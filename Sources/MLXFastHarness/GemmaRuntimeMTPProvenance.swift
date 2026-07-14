import CryptoKit
import Darwin
import Foundation
import MLXFastCore

public struct ExperimentalMTPArtifactReport: Equatable, Sendable {
    public let trackID: String
    public let targetModelID: String
    public let targetRevision: String
    public let targetByteCount: Int
    public let targetTreeSHA256: String
    public let assistantModelID: String
    public let assistantRevision: String
    public let assistantByteCount: Int
    public let assistantTreeSHA256: String
}

public struct ExperimentalMTPSourceTargetReport: Equatable, Sendable {
    public let modelID: String
    public let revision: String
    public let fileCount: Int
    public let byteCount: Int
    public let manifestSHA256: String
}

struct ExperimentalMTPTrackContract: Decodable {
    struct Target: Decodable {
        let upstreamModelID: String
        let upstreamRevision: String
        let runtimeModelID: String
        let runtimeRevision: String
        let manifestPath: String
        let manifestSHA256: String
        let expectedSourceBytes: Int
        let maximumTransformedBytes: Int
        let modelType: String
        let hiddenSize: Int
        let intermediateSize: Int
        let numHiddenLayers: Int
        let vocabSize: Int
        let quantizationBits: Int
        let quantizationGroupSize: Int

        enum CodingKeys: String, CodingKey {
            case upstreamModelID = "upstream_model_id"
            case upstreamRevision = "upstream_revision"
            case runtimeModelID = "runtime_model_id"
            case runtimeRevision = "runtime_revision"
            case manifestPath = "manifest_path"
            case manifestSHA256 = "manifest_sha256"
            case expectedSourceBytes = "expected_source_bytes"
            case maximumTransformedBytes = "maximum_transformed_bytes"
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case vocabSize = "vocab_size"
            case quantizationBits = "quantization_bits"
            case quantizationGroupSize = "quantization_group_size"
        }
    }

    struct AssistantFile: Decodable, Equatable {
        let path: String
        let sha256: String
        let bytes: Int
    }

    struct Assistant: Decodable {
        let modelID: String
        let revision: String
        let manifestPath: String
        let manifestSHA256: String
        let expectedTotalBytes: Int
        let maximumTotalBytes: Int
        let modelType: String
        let backboneHiddenSize: Int
        let hiddenSize: Int
        let intermediateSize: Int
        let numHiddenLayers: Int
        let numKVSharedLayers: Int
        let vocabSize: Int
        let files: [AssistantFile]

        enum CodingKeys: String, CodingKey {
            case modelID = "model_id"
            case revision
            case manifestPath = "manifest_path"
            case manifestSHA256 = "manifest_sha256"
            case expectedTotalBytes = "expected_total_bytes"
            case maximumTotalBytes = "maximum_total_bytes"
            case modelType = "model_type"
            case backboneHiddenSize = "backbone_hidden_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numKVSharedLayers = "num_kv_shared_layers"
            case vocabSize = "vocab_size"
            case files
        }
    }

    struct ProtocolContract: Decodable {
        let name: String
        let maximumBlockSize: Int
        let decodeTokens: Int
        let maximumDecodeTokens: Int
        let parentOracleRequired: Bool
        let workerTimingIsAuthoritative: Bool

        enum CodingKeys: String, CodingKey {
            case name
            case maximumBlockSize = "maximum_block_size"
            case decodeTokens = "decode_tokens"
            case maximumDecodeTokens = "maximum_decode_tokens"
            case parentOracleRequired = "parent_oracle_required"
            case workerTimingIsAuthoritative = "worker_timing_is_authoritative"
        }
    }

    struct ReferenceBaseline: Decodable {
        let status: String
        let publicationAllowed: Bool
        let requiredRebaselineHardware: String

        enum CodingKeys: String, CodingKey {
            case status
            case publicationAllowed = "publication_allowed"
            case requiredRebaselineHardware = "required_rebaseline_hardware"
        }
    }

    let schemaVersion: Int
    let trackID: String
    let officialScoringEnabled: Bool
    let mlxSwiftLMRevision: String
    let target: Target
    let assistant: Assistant
    let protocolContract: ProtocolContract
    let referenceBaseline: ReferenceBaseline

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case trackID = "track_id"
        case officialScoringEnabled = "official_scoring_enabled"
        case mlxSwiftLMRevision = "mlx_swift_lm_revision"
        case target
        case assistant
        case protocolContract = "protocol"
        case referenceBaseline = "reference_baseline"
    }
}

private struct ExperimentalMTPAssistantConfig: Decodable {
    struct TextConfig: Decodable {
        let modelType: String
        let hiddenSize: Int
        let intermediateSize: Int
        let numHiddenLayers: Int
        let numKVSharedLayers: Int
        let vocabSize: Int
        let numAttentionHeads: Int
        let numKeyValueHeads: Int
        let numGlobalKeyValueHeads: Int
        let headDim: Int
        let globalHeadDim: Int
        let attentionKEqV: Bool
        let slidingWindow: Int
        let maxPositionEmbeddings: Int
        let layerTypes: [String]

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numKVSharedLayers = "num_kv_shared_layers"
            case vocabSize = "vocab_size"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case numGlobalKeyValueHeads = "num_global_key_value_heads"
            case headDim = "head_dim"
            case globalHeadDim = "global_head_dim"
            case attentionKEqV = "attention_k_eq_v"
            case slidingWindow = "sliding_window"
            case maxPositionEmbeddings = "max_position_embeddings"
            case layerTypes = "layer_types"
        }
    }

    let modelType: String
    let backboneHiddenSize: Int
    let textConfig: TextConfig

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case backboneHiddenSize = "backbone_hidden_size"
        case textConfig = "text_config"
    }
}

extension GemmaRuntime {
    static let experimentalMTPTrackID = "gemma4-31b-it-mtp-v1"
    static let experimentalMTPDependencyRevision =
        "bc1c0ee67d15798343be17c9f8f61f7c0d977149"
    static let experimentalMTPTargetModelID =
        "mlx-community/gemma-4-31b-it-4bit"
    static let experimentalMTPTargetRevision =
        "696d436c404745a59f30e4939a658162b0a9e57f"
    static let experimentalMTPUpstreamTargetModelID =
        "google/gemma-4-31B-it"
    static let experimentalMTPUpstreamTargetRevision =
        "518276fb130dc81caf9a4f772e65e63ef2526493"
    static let experimentalMTPAssistantModelID =
        "google/gemma-4-31B-it-assistant"
    static let experimentalMTPAssistantRevision =
        "6c9152a7639e1f87626e4d4fd4dd9f3e20c9f3fb"

    static func validateExperimentalMTPSourceTarget(
        sourceTargetPath: String,
        contractPath: String
    ) throws -> ExperimentalMTPSourceTargetReport {
        let contractURL = URL(
            fileURLWithPath: contractPath
        ).standardizedFileURL
        try requireRegularFile(
            contractURL.path,
            description: "experimental MTP track contract"
        )
        let contract = try JSONDecoder().decode(
            ExperimentalMTPTrackContract.self,
            from: Data(contentsOf: contractURL)
        )
        try validateExperimentalMTPContract(contract)

        let repositoryRoot = contractURL
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let manifestURL = repositoryRoot.appendingPathComponent(
            contract.target.manifestPath
        ).standardizedFileURL
        guard manifestURL.path.hasPrefix(repositoryRoot.path + "/") else {
            throw MLXFastError.invalidInput(
                "experimental MTP target manifest escaped the contract root"
            )
        }
        try requireRegularFile(
            manifestURL.path,
            description: "experimental MTP target manifest"
        )
        let manifestDigest = try fileDigest(manifestURL)
        let manifestHex = manifestDigest.map {
            String(format: "%02x", $0)
        }.joined()
        guard manifestHex == contract.target.manifestSHA256 else {
            throw MLXFastError.invalidInput(
                "experimental MTP target manifest SHA256 mismatch"
            )
        }
        let entries = try parseExperimentalMTPManifest(
            Data(contentsOf: manifestURL)
        )
        try requireDirectory(
            sourceTargetPath,
            description: "experimental MTP source target"
        )
        let root = URL(
            fileURLWithPath: sourceTargetPath
        ).standardizedFileURL
        let actualNames = Set(
            try FileManager.default.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: nil,
                options: []
            ).map(\.lastPathComponent)
        )
        let expectedNames = Set(entries.map(\.path))
        guard actualNames == expectedNames else {
            throw MLXFastError.invalidInput(
                "experimental MTP source target inventory mismatch "
                    + "missing=\(expectedNames.subtracting(actualNames).sorted()) "
                    + "unexpected=\(actualNames.subtracting(expectedNames).sorted())"
            )
        }

        var byteCount = 0
        for entry in entries {
            let url = root.appendingPathComponent(entry.path)
            try requireRegularFile(
                url.path,
                description: "experimental MTP source target \(entry.path)"
            )
            let actualBytes = try fileSizeByteCount(
                from: FileManager.default.attributesOfItem(atPath: url.path),
                path: url.path
            )
            guard actualBytes == entry.bytes,
                  byteCount <= Int.max - actualBytes
            else {
                throw MLXFastError.invalidInput(
                    "experimental MTP source target \(entry.path) size mismatch"
                )
            }
            byteCount += actualBytes
            let actualHash = try fileDigest(url).map {
                String(format: "%02x", $0)
            }.joined()
            guard actualHash == entry.sha256 else {
                throw MLXFastError.invalidInput(
                    "experimental MTP source target \(entry.path) SHA256 mismatch"
                )
            }
        }
        guard byteCount == contract.target.expectedSourceBytes else {
            throw MLXFastError.invalidInput(
                "experimental MTP source target byte total mismatch"
            )
        }
        return ExperimentalMTPSourceTargetReport(
            modelID: contract.target.runtimeModelID,
            revision: contract.target.runtimeRevision,
            fileCount: entries.count,
            byteCount: byteCount,
            manifestSHA256: manifestHex
        )
    }

    static func validateExperimentalMTPArtifacts(
        targetWeightsPath: String,
        assistantPath: String,
        contractPath: String
    ) throws -> ExperimentalMTPArtifactReport {
        try requireRegularFile(
            contractPath,
            description: "experimental MTP track contract"
        )
        let contractURL = URL(fileURLWithPath: contractPath).standardizedFileURL
        let contractSize = try fileSizeByteCount(
            from: FileManager.default.attributesOfItem(atPath: contractURL.path),
            path: contractURL.path
        )
        guard contractSize > 0, contractSize <= 64 * 1024 else {
            throw MLXFastError.invalidInput(
                "experimental MTP track contract must be at most 65536 bytes"
            )
        }
        let contract = try JSONDecoder().decode(
            ExperimentalMTPTrackContract.self,
            from: Data(contentsOf: contractURL)
        )
        try validateExperimentalMTPContract(contract)

        try requireDirectory(
            targetWeightsPath,
            description: "experimental MTP transformed target"
        )
        let targetDigest = try directoryDigest(
            rootPath: targetWeightsPath,
            ignoredRelativePaths: [".benchmark-source.sha256", ".gitkeep"]
        )
        guard targetDigest.byteCount > 0,
              targetDigest.byteCount <= contract.target.maximumTransformedBytes
        else {
            throw MLXFastError.invalidInput(
                "experimental MTP transformed target is outside its artifact byte cap"
            )
        }
        try validateRuntimeWorkerPinnedConfiguration(
            weightsPath: targetWeightsPath
        )

        let assistantDigest = try validateExperimentalMTPAssistantDirectory(
            assistantPath,
            contract: contract
        )
        return ExperimentalMTPArtifactReport(
            trackID: contract.trackID,
            targetModelID: contract.target.runtimeModelID,
            targetRevision: contract.target.runtimeRevision,
            targetByteCount: targetDigest.byteCount,
            targetTreeSHA256: targetDigest.sha256,
            assistantModelID: contract.assistant.modelID,
            assistantRevision: contract.assistant.revision,
            assistantByteCount: assistantDigest.byteCount,
            assistantTreeSHA256: assistantDigest.sha256
        )
    }

    static func validateExperimentalMTPContract(
        _ contract: ExperimentalMTPTrackContract
    ) throws {
        let expectedFiles = [
            ExperimentalMTPTrackContract.AssistantFile(
                path: "config.json",
                sha256:
                    "d09487c1083a334a97545f204dd41ab1a40b0438bdead19558f0e1357e26d66a",
                bytes: 2_316
            ),
            ExperimentalMTPTrackContract.AssistantFile(
                path: "model.safetensors",
                sha256:
                    "9f80df6099fa1fd7db71220ec9ee864d5ecff769878697dd5763e4285b15a1da",
                bytes: 939_042_560
            ),
        ]
        guard contract.schemaVersion == 1,
              contract.trackID == experimentalMTPTrackID,
              !contract.officialScoringEnabled,
              contract.mlxSwiftLMRevision == experimentalMTPDependencyRevision,
              contract.target.upstreamModelID
                  == experimentalMTPUpstreamTargetModelID,
              contract.target.upstreamRevision
                  == experimentalMTPUpstreamTargetRevision,
              contract.target.runtimeModelID == experimentalMTPTargetModelID,
              contract.target.runtimeRevision == experimentalMTPTargetRevision,
              contract.target.manifestPath
                  == "fixtures/mtp_gemma_4_31b_it_4bit.sha256",
              contract.target.manifestSHA256
                  == "71c9cfe815075dbe072f4d9fcedae4452d7efea2720ee1f6feb0199c18bed5a9",
              contract.target.expectedSourceBytes == 18_444_420_181,
              contract.target.maximumTransformedBytes == 20 * (1 << 30),
              contract.target.modelType == "gemma4_text",
              contract.target.hiddenSize == MLXFastConstants.hiddenSize,
              contract.target.intermediateSize
                  == MLXFastConstants.intermediateSize,
              contract.target.numHiddenLayers
                  == MLXFastConstants.numHiddenLayers,
              contract.target.vocabSize == MLXFastConstants.vocabSize,
              contract.target.quantizationBits == 4,
              contract.target.quantizationGroupSize == 64,
              contract.assistant.modelID == experimentalMTPAssistantModelID,
              contract.assistant.revision == experimentalMTPAssistantRevision,
              contract.assistant.manifestPath
                  == "fixtures/mtp_gemma_4_31b_it_assistant_bf16.sha256",
              contract.assistant.manifestSHA256
                  == "9dde96d9e236bae641f54cb972b704cfeaceee305f700b0fd790e397e71fed50",
              contract.assistant.expectedTotalBytes == 939_044_876,
              contract.assistant.maximumTotalBytes == 1_000_000_000,
              contract.assistant.modelType == "gemma4_assistant",
              contract.assistant.backboneHiddenSize
                  == MLXFastConstants.hiddenSize,
              contract.assistant.hiddenSize == 1_024,
              contract.assistant.intermediateSize == 8_192,
              contract.assistant.numHiddenLayers == 4,
              contract.assistant.numKVSharedLayers == 4,
              contract.assistant.vocabSize == MLXFastConstants.vocabSize,
              contract.assistant.files == expectedFiles,
              contract.protocolContract.name == "trusted_mtp_block_v1",
              contract.protocolContract.maximumBlockSize
                  == MLXFastConstants.experimentalMTPMaxBlockSize,
              contract.protocolContract.decodeTokens
                  == MLXFastConstants.experimentalMTPMaxTotalTokens,
              contract.protocolContract.maximumDecodeTokens
                  == MLXFastConstants.experimentalMTPMaxConfiguredTotalTokens,
              contract.protocolContract.parentOracleRequired,
              !contract.protocolContract.workerTimingIsAuthoritative,
              contract.referenceBaseline.status == "not_established",
              !contract.referenceBaseline.publicationAllowed,
              contract.referenceBaseline.requiredRebaselineHardware
                  == "m5-bench"
        else {
            throw MLXFastError.invalidInput(
                "experimental MTP track contract does not match the organizer-pinned identity"
            )
        }
    }

    private static func validateExperimentalMTPAssistantDirectory(
        _ path: String,
        contract: ExperimentalMTPTrackContract
    ) throws -> DirectoryDigest {
        try requireDirectory(
            path,
            description: "experimental MTP assistant"
        )
        let root = URL(fileURLWithPath: path).standardizedFileURL
        let expectedNames = Set(contract.assistant.files.map(\.path))
        let entries = try FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [
                .isRegularFileKey,
                .isDirectoryKey,
                .isSymbolicLinkKey,
            ],
            options: []
        )
        let actualNames = Set(entries.map(\.lastPathComponent))
        guard actualNames == expectedNames else {
            let unexpected = actualNames.subtracting(expectedNames).sorted()
            let missing = expectedNames.subtracting(actualNames).sorted()
            throw MLXFastError.invalidInput(
                "experimental MTP assistant inventory mismatch "
                    + "missing=\(missing) unexpected=\(unexpected)"
            )
        }

        var byteCount = 0
        var treeHasher = SHA256()
        for expected in contract.assistant.files {
            guard expected.path == URL(fileURLWithPath: expected.path).lastPathComponent,
                  !expected.path.isEmpty
            else {
                throw MLXFastError.invalidInput(
                    "experimental MTP assistant contract contains an unsafe file path"
                )
            }
            let url = root.appendingPathComponent(expected.path)
            try requireRegularFile(
                url.path,
                description: "experimental MTP assistant \(expected.path)"
            )
            let actualBytes = try fileSizeByteCount(
                from: FileManager.default.attributesOfItem(atPath: url.path),
                path: url.path
            )
            guard actualBytes == expected.bytes,
                  byteCount <= Int.max - actualBytes
            else {
                throw MLXFastError.invalidInput(
                    "experimental MTP assistant \(expected.path) size mismatch"
                )
            }
            byteCount += actualBytes
            let digest = try fileDigest(url)
            let hex = digest.map { String(format: "%02x", $0) }.joined()
            guard hex == expected.sha256 else {
                throw MLXFastError.invalidInput(
                    "experimental MTP assistant \(expected.path) SHA256 mismatch"
                )
            }
            treeHasher.update(data: Data(expected.path.utf8))
            treeHasher.update(data: Data([0]))
            treeHasher.update(data: Data(digest))
            treeHasher.update(data: Data([0]))
        }
        guard byteCount == contract.assistant.expectedTotalBytes,
              byteCount <= contract.assistant.maximumTotalBytes
        else {
            throw MLXFastError.invalidInput(
                "experimental MTP assistant exceeds its organizer artifact budget"
            )
        }

        try validateExperimentalMTPAssistantConfig(
            at: root.appendingPathComponent("config.json"),
            contract: contract
        )
        return DirectoryDigest(
            fileCount: contract.assistant.files.count,
            byteCount: byteCount,
            sha256: treeHasher.finalize().map {
                String(format: "%02x", $0)
            }.joined()
        )
    }

    private static func validateExperimentalMTPAssistantConfig(
        at url: URL,
        contract: ExperimentalMTPTrackContract
    ) throws {
        let config = try JSONDecoder().decode(
            ExperimentalMTPAssistantConfig.self,
            from: Data(contentsOf: url)
        )
        let text = config.textConfig
        guard config.modelType == contract.assistant.modelType,
              config.backboneHiddenSize
                  == contract.assistant.backboneHiddenSize,
              text.modelType == "gemma4_text",
              text.hiddenSize == contract.assistant.hiddenSize,
              text.intermediateSize == contract.assistant.intermediateSize,
              text.numHiddenLayers == contract.assistant.numHiddenLayers,
              text.numKVSharedLayers
                  == contract.assistant.numKVSharedLayers,
              text.vocabSize == contract.assistant.vocabSize,
              text.numAttentionHeads == 32,
              text.numKeyValueHeads == 16,
              text.numGlobalKeyValueHeads == 4,
              text.headDim == 256,
              text.globalHeadDim == 512,
              text.attentionKEqV,
              text.slidingWindow == 1_024,
              text.maxPositionEmbeddings == 262_144,
              text.layerTypes == [
                  "sliding_attention",
                  "sliding_attention",
                  "sliding_attention",
                  "full_attention",
              ]
        else {
            throw MLXFastError.invalidInput(
                "experimental MTP assistant config is incompatible with the pinned target"
            )
        }
    }

    private struct ExperimentalMTPManifestEntry {
        let path: String
        let sha256: String
        let bytes: Int
    }

    private static func parseExperimentalMTPManifest(
        _ data: Data
    ) throws -> [ExperimentalMTPManifestEntry] {
        guard let text = String(data: data, encoding: .utf8) else {
            throw MLXFastError.invalidInput(
                "experimental MTP target manifest is not UTF-8"
            )
        }
        var entries: [ExperimentalMTPManifestEntry] = []
        var names = Set<String>()
        for rawLine in text.split(
            separator: "\n",
            omittingEmptySubsequences: false
        ) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") {
                continue
            }
            let fields = line.split(whereSeparator: \.isWhitespace)
            guard fields.count == 3,
                  fields[0].count == 64,
                  fields[0].allSatisfy({
                      ("0"..."9").contains($0)
                          || ("a"..."f").contains($0)
                  }),
                  let bytes = Int(fields[1]),
                  bytes > 0
            else {
                throw MLXFastError.invalidInput(
                    "experimental MTP target manifest has a malformed line"
                )
            }
            let path = String(fields[2])
            guard !path.isEmpty,
                  path == URL(fileURLWithPath: path).lastPathComponent,
                  names.insert(path).inserted
            else {
                throw MLXFastError.invalidInput(
                    "experimental MTP target manifest has an unsafe or duplicate path"
                )
            }
            entries.append(
                ExperimentalMTPManifestEntry(
                    path: path,
                    sha256: String(fields[0]),
                    bytes: bytes
                )
            )
        }
        guard !entries.isEmpty else {
            throw MLXFastError.invalidInput(
                "experimental MTP target manifest contains no files"
            )
        }
        return entries
    }
}
