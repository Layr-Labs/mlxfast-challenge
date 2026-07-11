import Foundation
import MLXFastCore

public struct TransformOptions: Equatable {
    public let referencePath: String
    public let outputPath: String

    public init(referencePath: String, outputPath: String) {
        self.referencePath = referencePath
        self.outputPath = outputPath
    }
}

public struct TransformReport: Equatable {
    public let referencePath: String
    public let outputPath: String
    public let denseTensorCount: Int
    public let denseShardCount: Int
    public let configPath: String
    public let indexPath: String

    public init(
        referencePath: String,
        outputPath: String,
        denseTensorCount: Int,
        denseShardCount: Int,
        configPath: String,
        indexPath: String
    ) {
        self.referencePath = referencePath
        self.outputPath = outputPath
        self.denseTensorCount = denseTensorCount
        self.denseShardCount = denseShardCount
        self.configPath = configPath
        self.indexPath = indexPath
    }
}

/// Offline transform for the Gemma 4 31B 4-bit checkpoint: selects ONLY the
/// text-tower tensors (the `language_model.` prefix in the source index),
/// drops every vision/audio/multimodal-projector tensor, and rewrites the
/// selected tensors into dense safetensors shard(s) plus a
/// `model.safetensors.index.json` and a runtime-authored `config.json`
/// (the flattened `text_config` fields the runtime needs, plus the
/// checkpoint's quantization metadata). Text/audio/vision are the only kinds
/// of tensors the reference checkpoint ships; there is no expert manifest --
/// the whole selected tree is one flat set of dense tensors, matching how a
/// single dense model (no MoE, no expert streaming) is loaded fully into RAM
/// at runtime init.
public enum SwiftTransform {
    /// Tensor name prefix that marks a checkpoint tensor as part of the text
    /// tower. Every other prefix (`vision_tower.`, `embed_vision.`,
    /// `audio_tower.`, `multi_modal_projector.`, ...) is vision/audio/
    /// multimodal-glue and is out of scope for this text-only challenge.
    static let textTowerPrefix = "language_model."

    public static func run(_ options: TransformOptions) throws -> TransformReport {
        try run(options, beforeSourceRevalidation: nil)
    }

    static func run(
        _ options: TransformOptions,
        beforeSourceRevalidation: (() throws -> Void)?
    ) throws -> TransformReport {
        let referenceDirectory = canonicalURL(
            try findReferenceDirectory(URL(fileURLWithPath: options.referencePath))
        )
        let outputDirectory = canonicalURL(URL(fileURLWithPath: options.outputPath))
        try validateDistinctDirectories(
            referenceDirectory: referenceDirectory,
            outputDirectory: outputDirectory
        )
        var outputIsDirectory = ObjCBool(false)
        if FileManager.default.fileExists(
            atPath: outputDirectory.path,
            isDirectory: &outputIsDirectory
        ), !outputIsDirectory.boolValue {
            throw MLXFastError.invalidInput(
                "transform output exists and is not a directory: \(outputDirectory.path)"
            )
        }

        let referenceConfigPath = referenceDirectory.appendingPathComponent("config.json")
        try requireFile(
            referenceConfigPath.path,
            description: "Gemma 4 31B 4-bit reference config"
        )
        let runtimeConfigData = try makeRuntimeConfigData(sourceConfigPath: referenceConfigPath)
        let metadataSnapshot = try captureMetadataFiles(from: referenceDirectory)

        let index = try loadIndex(referenceDirectory)
        let indexSnapshot = try index.canonicalData()
        let validatedHeaders = try validateCheckpointIndex(
            index,
            referenceDirectory: referenceDirectory
        )
        let textKeys = Set(index.weightMap.keys.filter(isTextTowerKey))
        guard !textKeys.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint index contains no text-tower tensors")
        }

        let textKeysByShard = Dictionary(grouping: textKeys) { key in
            index.weightMap[key] ?? ""
        }
        var totalTensorByteCount = 0
        for key in textKeys.sorted() {
            guard let shardName = index.weightMap[key],
                  let info = validatedHeaders[shardName]?.tensors[key]
            else {
                throw MLXFastError.invalidInput(
                    "missing validated tensor metadata for \(key)"
                )
            }
            let (nextTotal, overflow) = totalTensorByteCount.addingReportingOverflow(
                info.byteCount
            )
            guard !overflow else {
                throw MLXFastError.invalidInput(
                    "transformed tensor byte count overflows Int"
                )
            }
            totalTensorByteCount = nextTotal
        }

        let fileManager = FileManager.default
        let stagingDirectory = outputDirectory.deletingLastPathComponent().appendingPathComponent(
            ".\(outputDirectory.lastPathComponent).mlxfast-transform-\(UUID().uuidString)",
            isDirectory: true
        )
        try fileManager.createDirectory(
            at: stagingDirectory.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try fileManager.createDirectory(at: stagingDirectory, withIntermediateDirectories: false)
        var installed = false
        defer {
            if !installed {
                try? fileManager.removeItem(at: stagingDirectory)
            }
        }

        var copiedTensors = 0
        for shardName in textKeysByShard.keys.sorted() {
            let source = referenceDirectory.appendingPathComponent(shardName)
            let destination = stagingDirectory.appendingPathComponent(shardName)
            guard let header = validatedHeaders[shardName] else {
                throw MLXFastError.invalidInput("missing validated header for checkpoint shard \(shardName)")
            }
            copiedTensors += try Safetensors.copySubset(
                from: source,
                to: destination,
                tensorNames: textKeysByShard[shardName, default: []].sorted(),
                validatedHeader: header
            )
        }

        try writeMetadataFiles(metadataSnapshot, to: stagingDirectory)
        try index.writeStripped(
            to: stagingDirectory.appendingPathComponent("model.safetensors.index.json"),
            keeping: textKeys,
            totalTensorByteCount: totalTensorByteCount
        )

        try runtimeConfigData.write(
            to: stagingDirectory.appendingPathComponent("config.json")
        )
        try beforeSourceRevalidation?()
        try validateConfigAndIndexSnapshot(
            referenceDirectory: referenceDirectory,
            referenceConfigPath: referenceConfigPath,
            runtimeConfigData: runtimeConfigData,
            indexSnapshot: indexSnapshot,
            metadataSnapshot: metadataSnapshot
        )
        for shardName in validatedHeaders.keys.sorted() {
            guard let header = validatedHeaders[shardName] else {
                throw MLXFastError.invalidInput(
                    "missing validated header for checkpoint shard \(shardName)"
                )
            }
            try Safetensors.validateSourceIdentity(
                referenceDirectory.appendingPathComponent(shardName),
                against: header
            )
        }
        try validateConfigAndIndexSnapshot(
            referenceDirectory: referenceDirectory,
            referenceConfigPath: referenceConfigPath,
            runtimeConfigData: runtimeConfigData,
            indexSnapshot: indexSnapshot,
            metadataSnapshot: metadataSnapshot
        )
        try installTransformedDirectory(
            stagingDirectory,
            at: outputDirectory,
            fileManager: fileManager
        )
        installed = true

        let indexPath = outputDirectory.appendingPathComponent("model.safetensors.index.json")
        let configPath = outputDirectory.appendingPathComponent("config.json")

        return TransformReport(
            referencePath: referenceDirectory.path,
            outputPath: outputDirectory.path,
            denseTensorCount: copiedTensors,
            denseShardCount: textKeysByShard.count,
            configPath: configPath.path,
            indexPath: indexPath.path
        )
    }

    private static func loadIndex(_ referenceDirectory: URL) throws -> CheckpointIndex {
        let indexPath = referenceDirectory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexPath.path) {
            return try CheckpointIndex.load(from: indexPath)
        }
        return try CheckpointIndex.buildFromSafetensors(in: referenceDirectory)
    }

    private static func validateCheckpointIndex(
        _ index: CheckpointIndex,
        referenceDirectory: URL,
        fileManager: FileManager = .default
    ) throws -> [String: SafetensorsHeader] {
        guard !index.weightMap.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint index contains no tensors")
        }

        let keysByShard = Dictionary(grouping: index.weightMap.keys.sorted()) { key in
            index.weightMap[key] ?? ""
        }
        var headersByShard: [String: SafetensorsHeader] = [:]
        for shardName in keysByShard.keys.sorted() {
            try validateSafetensorsShardName(shardName, context: "checkpoint index")

            let shardURL = referenceDirectory.appendingPathComponent(shardName)
            try requireFile(shardURL.path, description: "checkpoint shard \(shardName)")
            let header = try Safetensors.readHeader(shardURL)
            headersByShard[shardName] = header
            let attributes = try fileManager.attributesOfItem(atPath: shardURL.path)
            let byteCount = try fileSizeByteCount(from: attributes, path: shardURL.path)
            guard let baseOffset = Int(exactly: header.dataBaseOffset) else {
                throw MLXFastError.invalidInput("checkpoint shard header is too large: \(shardName)")
            }

            for key in keysByShard[shardName, default: []].sorted() {
                guard let info = header.tensors[key] else {
                    throw MLXFastError.invalidInput(
                        "checkpoint index lists tensor \(key) in \(shardName), but the shard header does not contain it"
                    )
                }
                let dtype = try TensorDType.parse(info.dtype)
                let expectedByteLength = try expectedTensorByteCount(
                    name: key,
                    dtype: dtype,
                    shape: info.shape
                )
                guard info.byteCount == expectedByteLength else {
                    throw MLXFastError.invalidInput(
                        "checkpoint tensor \(key) byte length \(info.byteCount) does not match dtype \(info.dtype) and shape \(info.shape) expected \(expectedByteLength)"
                    )
                }
                let (end, overflow) = baseOffset.addingReportingOverflow(info.dataEnd)
                guard
                    !overflow,
                    info.dataStart >= 0,
                    info.byteCount > 0,
                    end <= byteCount
                else {
                    throw MLXFastError.invalidInput(
                        "checkpoint tensor \(key) byte range \(info.dataStart)..<\(info.dataEnd) exceeds shard size \(byteCount)"
                    )
                }
            }
        }
        return headersByShard
    }

    private static func canonicalURL(_ url: URL) -> URL {
        url.standardizedFileURL.resolvingSymlinksInPath()
    }

    static func validateDistinctDirectories(
        referenceDirectory: URL,
        outputDirectory: URL,
        workingDirectory: URL = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath,
            isDirectory: true
        )
    ) throws {
        guard referenceDirectory.path != outputDirectory.path else {
            throw MLXFastError.invalidInput(
                "transform reference and output directories must be different: \(referenceDirectory.path)"
            )
        }
        let outputPrefix = outputDirectory.path == "/" ? "/" : outputDirectory.path + "/"
        guard !referenceDirectory.path.hasPrefix(outputPrefix) else {
            throw MLXFastError.invalidInput(
                "transform output directory cannot contain the reference directory: \(outputDirectory.path)"
            )
        }
        let referencePrefix = referenceDirectory.path == "/"
            ? "/"
            : referenceDirectory.path + "/"
        guard !outputDirectory.path.hasPrefix(referencePrefix) else {
            throw MLXFastError.invalidInput(
                "transform output directory cannot be inside the reference directory: \(outputDirectory.path)"
            )
        }

        let canonicalWorkingDirectory = canonicalURL(workingDirectory)
        guard canonicalWorkingDirectory.path != outputDirectory.path,
              !canonicalWorkingDirectory.path.hasPrefix(outputPrefix)
        else {
            throw MLXFastError.invalidInput(
                "transform output directory cannot contain the current working directory: \(outputDirectory.path)"
            )
        }
    }

    private static func installTransformedDirectory(
        _ stagedDirectory: URL,
        at outputDirectory: URL,
        fileManager: FileManager
    ) throws {
        if fileManager.fileExists(atPath: outputDirectory.path) {
            _ = try fileManager.replaceItemAt(outputDirectory, withItemAt: stagedDirectory)
        } else {
            try fileManager.moveItem(at: stagedDirectory, to: outputDirectory)
        }
    }

    private static func findReferenceDirectory(_ base: URL) throws -> URL {
        if FileManager.default.fileExists(
            atPath: base.appendingPathComponent("config.json").path
        ) {
            return base
        }

        guard let enumerator = FileManager.default.enumerator(
            at: base,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw MLXFastError.missingFile("reference path not found at \(base.path)")
        }

        for case let url as URL in enumerator {
            if url.lastPathComponent == "config.json" {
                return url.deletingLastPathComponent()
            }
        }

        throw MLXFastError.missingFile(
            "no config.json found under \(base.path); place the Gemma 4 31B 4-bit checkpoint there"
        )
    }

    static func isTextTowerKey(_ key: String) -> Bool {
        key.hasPrefix(textTowerPrefix)
    }

    private static func captureMetadataFiles(from source: URL) throws -> [String: Data] {
        let files = try FileManager.default.contentsOfDirectory(
            at: source,
            includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey]
        )
        var snapshot: [String: Data] = [:]
        for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            if file.lastPathComponent == "model.safetensors.index.json" || file.lastPathComponent == "config.json" {
                continue
            }
            if shouldCopyMetadataFile(file) {
                let values = try file.resourceValues(
                    forKeys: [.isRegularFileKey, .isSymbolicLinkKey]
                )
                guard values.isRegularFile == true, values.isSymbolicLink != true else {
                    throw MLXFastError.invalidInput(
                        "reference metadata is not a regular file: \(file.path)"
                    )
                }
                snapshot[file.lastPathComponent] = try Data(contentsOf: file)
            }
        }
        return snapshot
    }

    private static func writeMetadataFiles(
        _ snapshot: [String: Data],
        to destination: URL
    ) throws {
        for name in snapshot.keys.sorted() {
            try snapshot[name]?.write(to: destination.appendingPathComponent(name))
        }
    }

    private static func shouldCopyMetadataFile(_ url: URL) -> Bool {
        let name = url.lastPathComponent
        if name.hasSuffix(".safetensors") {
            return false
        }
        switch url.pathExtension {
        case "json", "model", "tiktoken", "txt":
            return true
        default:
            return name == "tokenizer" || name == "vocab"
        }
    }

    /// Writes the runtime's `config.json`: the source checkpoint's
    /// `text_config` fields flattened to the top level (the exact schema
    /// `Gemma4Config.load` reads), plus the checkpoint-wide `quantization`
    /// block. The runtime controls this schema directly, so it carries only
    /// what `Gemma4Config`/`Gemma4WeightLoader` actually need -- no vision or
    /// audio config, no architecture/tokenizer metadata duplicated from
    /// `tokenizer_config.json`.
    static func makeRuntimeConfigData(sourceConfigPath: URL) throws -> Data {
        let data = try Data(contentsOf: sourceConfigPath)
        let object = try JSONSerialization.jsonObject(with: data)
        guard let root = object as? [String: Any] else {
            throw MLXFastError.invalidInput("reference config.json must be a JSON object")
        }
        guard let textConfig = root["text_config"] as? [String: Any] else {
            throw MLXFastError.invalidInput("reference config.json is missing text_config")
        }

        var runtimeConfig = textConfig
        if let quantization = root["quantization"] {
            runtimeConfig["quantization"] = quantization
        } else if let quantizationConfig = root["quantization_config"] {
            runtimeConfig["quantization"] = quantizationConfig
        }

        return try JSONSerialization.data(
            withJSONObject: runtimeConfig,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
    }

    private static func validateConfigAndIndexSnapshot(
        referenceDirectory: URL,
        referenceConfigPath: URL,
        runtimeConfigData: Data,
        indexSnapshot: Data,
        metadataSnapshot: [String: Data]
    ) throws {
        guard try makeRuntimeConfigData(sourceConfigPath: referenceConfigPath)
            == runtimeConfigData
        else {
            throw MLXFastError.invalidInput(
                "reference config changed while transform was running"
            )
        }
        guard try loadIndex(referenceDirectory).canonicalData() == indexSnapshot else {
            throw MLXFastError.invalidInput(
                "checkpoint index changed while transform was running"
            )
        }
        guard try captureMetadataFiles(from: referenceDirectory) == metadataSnapshot else {
            throw MLXFastError.invalidInput(
                "reference tokenizer metadata changed while transform was running"
            )
        }
    }
}
