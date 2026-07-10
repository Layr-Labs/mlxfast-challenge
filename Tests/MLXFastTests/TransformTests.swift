import Foundation
import Testing
@testable import MLXFastCore
@testable import MLXFastModel
@testable import MLXFastHarness
@testable import MLXFastTransform

@Test
func transformSelectsTextTowerTensorsAndDropsVisionTensors() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)

    try referenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try #"{"tokenizer":"fixture"}"#.write(
        to: reference.appendingPathComponent("tokenizer.json"),
        atomically: true,
        encoding: .utf8
    )

    let textName = "language_model.model.layers.0.self_attn.q_proj.weight"
    let visionName = "vision_tower.encoder.layers.0.self_attn.q_proj.weight"
    let embedVisionName = "embed_vision.embedding_projection.weight"
    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [
            TensorFixture(name: textName, dtype: "U8", shape: [4], data: Data([1, 2, 3, 4])),
            TensorFixture(name: visionName, dtype: "U8", shape: [3], data: Data([9, 8, 7])),
            TensorFixture(name: embedVisionName, dtype: "U8", shape: [2], data: Data([5, 6])),
        ]
    )
    try """
    {
      "metadata": {"total_size": 9},
      "weight_map": {
        "\(textName)": "\(shardName)",
        "\(visionName)": "\(shardName)",
        "\(embedVisionName)": "\(shardName)"
      }
    }
    """.write(
        to: reference.appendingPathComponent("model.safetensors.index.json"),
        atomically: true,
        encoding: .utf8
    )

    let report = try SwiftTransform.run(
        TransformOptions(referencePath: reference.path, outputPath: output.path)
    )

    #expect(report.denseTensorCount == 1)
    #expect(report.denseShardCount == 1)
    #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent("config.json").path))
    #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent("tokenizer.json").path))
    #expect(!FileManager.default.fileExists(atPath: output.appendingPathComponent("experts").path))

    let outputShard = output.appendingPathComponent(shardName)
    let outputHeader = try Safetensors.readHeader(outputShard)
    #expect(outputHeader.tensors.keys.sorted() == [textName])
    #expect(try tensorBytes(outputShard, header: outputHeader, name: textName) == Data([1, 2, 3, 4]))

    let strippedIndexData = try Data(
        contentsOf: output.appendingPathComponent("model.safetensors.index.json")
    )
    let strippedIndex = try JSONSerialization.jsonObject(with: strippedIndexData) as? [String: Any]
    let weightMap = try #require(strippedIndex?["weight_map"] as? [String: String])
    #expect(weightMap == [textName: shardName])
}

@Test
func transformWritesFlattenedRuntimeConfigFromTextConfig() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try referenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let textName = "language_model.model.embed_tokens.weight"
    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [TensorFixture(name: textName, dtype: "U8", shape: [1], data: Data([1]))]
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [textName: shardName]
    )

    _ = try SwiftTransform.run(
        TransformOptions(referencePath: reference.path, outputPath: output.path)
    )

    let configData = try Data(contentsOf: output.appendingPathComponent("config.json"))
    let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
    #expect(config?["num_hidden_layers"] as? Int == MLXFastConstants.numHiddenLayers)
    #expect(config?["vocab_size"] as? Int == MLXFastConstants.vocabSize)
    #expect(config?["text_config"] == nil)
    let quantization = try #require(config?["quantization"] as? [String: Any])
    #expect(quantization["group_size"] as? Int == 64)
    #expect(quantization["bits"] as? Int == 4)
}

@Test
func transformRejectsReferenceAsOutputWithoutMutatingCheckpoint() throws {
    let fixture = try writeTransformFixture()
    let originalConfig = try Data(
        contentsOf: fixture.reference.appendingPathComponent("config.json")
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(
                referencePath: fixture.reference.path,
                outputPath: fixture.reference.path
            )
        )
    }

    #expect(
        try Data(contentsOf: fixture.reference.appendingPathComponent("config.json"))
            == originalConfig
    )
    #expect(
        FileManager.default.fileExists(
            atPath: fixture.reference.appendingPathComponent("model-00001-of-00001.safetensors").path
        )
    )
}

@Test
func transformRejectsSymlinkAliasOfReferenceAsOutput() throws {
    let fixture = try writeTransformFixture()
    let outputAlias = fixture.root.appendingPathComponent("reference-alias", isDirectory: true)
    try FileManager.default.createSymbolicLink(
        at: outputAlias,
        withDestinationURL: fixture.reference
    )
    let originalConfig = try Data(
        contentsOf: fixture.reference.appendingPathComponent("config.json")
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(
                referencePath: fixture.reference.path,
                outputPath: outputAlias.path
            )
        )
    }

    #expect(
        try Data(contentsOf: fixture.reference.appendingPathComponent("config.json"))
            == originalConfig
    )
}

@Test
func transformRejectsOutputThatContainsWorkingDirectory() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference")
    let workingDirectory = root.appendingPathComponent("workspace/project")
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: workingDirectory, withIntermediateDirectories: true)

    #expect(throws: MLXFastError.self) {
        try SwiftTransform.validateDistinctDirectories(
            referenceDirectory: reference,
            outputDirectory: root,
            workingDirectory: workingDirectory
        )
    }
}

@Test
func transformRejectsOutputInsideReferenceDirectory() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference")
    let output = reference.appendingPathComponent("weights")
    let workingDirectory = root.appendingPathComponent("workspace")
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: workingDirectory, withIntermediateDirectories: true)

    #expect(throws: MLXFastError.self) {
        try SwiftTransform.validateDistinctDirectories(
            referenceDirectory: reference,
            outputDirectory: output,
            workingDirectory: workingDirectory
        )
    }
}

@Test
func transformRejectsExistingRegularFileOutputWithoutReplacingIt() throws {
    let fixture = try writeTransformFixture()
    let outputFile = fixture.root.appendingPathComponent("existing-output")
    try "preserve me".write(to: outputFile, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(
                referencePath: fixture.reference.path,
                outputPath: outputFile.path
            )
        )
    }

    #expect(try String(contentsOf: outputFile, encoding: .utf8) == "preserve me")
}

@Test
func transformAtomicallyReplacesExistingOutputAndRemovesStaleFiles() throws {
    let fixture = try writeTransformFixture()
    try FileManager.default.createDirectory(at: fixture.output, withIntermediateDirectories: true)
    let staleFile = fixture.output.appendingPathComponent("stale.txt")
    try "stale".write(to: staleFile, atomically: true, encoding: .utf8)

    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )

    #expect(!FileManager.default.fileExists(atPath: staleFile.path))
    #expect(
        FileManager.default.fileExists(
            atPath: fixture.output.appendingPathComponent("model.safetensors.index.json").path
        )
    )
}

@Test
func transformFailurePreservesExistingOutputAndRemovesStagingDirectory() throws {
    let fixture = try writeTransformFixture()
    try #"{"vision_config":{}}"#.write(
        to: fixture.reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try FileManager.default.createDirectory(at: fixture.output, withIntermediateDirectories: true)
    let sentinel = fixture.output.appendingPathComponent("sentinel.txt")
    try "original".write(to: sentinel, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
        )
    }

    #expect(try String(contentsOf: sentinel, encoding: .utf8) == "original")
    let siblings = try FileManager.default.contentsOfDirectory(
        at: fixture.output.deletingLastPathComponent(),
        includingPropertiesForKeys: nil
    )
    #expect(
        !siblings.contains {
            $0.lastPathComponent.hasPrefix(".weights.mlxfast-transform-")
        }
    )
}

@Test
func checkpointIndexBuildRejectsDuplicateTensorNamesAcrossShards() throws {
    let root = try temporaryDirectory()
    let tensorName = "language_model.model.embed_tokens.weight"
    try writeSafetensors(
        root.appendingPathComponent("model-00001-of-00002.safetensors"),
        tensors: [TensorFixture(name: tensorName, dtype: "U8", shape: [1], data: Data([1]))]
    )
    try writeSafetensors(
        root.appendingPathComponent("model-00002-of-00002.safetensors"),
        tensors: [TensorFixture(name: tensorName, dtype: "U8", shape: [1], data: Data([2]))]
    )

    #expect(throws: MLXFastError.self) {
        _ = try CheckpointIndex.buildFromSafetensors(in: root)
    }
}

@Test
func transformRejectsCheckpointWithoutTextTowerTensorsBeforeCreatingOutput() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try referenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let visionName = "vision_tower.encoder.layers.0.self_attn.q_proj.weight"
    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [TensorFixture(name: visionName, dtype: "U8", shape: [2], data: Data([1, 2]))]
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [visionName: shardName]
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: reference.path, outputPath: output.path)
        )
    }
    #expect(!FileManager.default.fileExists(atPath: output.path))
}

@Test
func transformVerifierAcceptsFreshSubmittedTransformOutputAndIgnoresLocalCacheMarkers() throws {
    let fixture = try writeTransformFixture()
    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )
    try "cache\n".write(
        to: fixture.output.appendingPathComponent(".benchmark-source.sha256"),
        atomically: true,
        encoding: .utf8
    )
    FileManager.default.createFile(
        atPath: fixture.output.appendingPathComponent(".gitkeep").path,
        contents: Data()
    )

    let report = try TransformVerifier.verify(
        TransformVerificationOptions(
            referencePath: fixture.reference.path,
            weightsPath: fixture.output.path,
            temporaryParentPath: fixture.root.path
        )
    )

    #expect(report.referencePath == fixture.reference.path)
    #expect(report.weightsPath == fixture.output.path)
    #expect(report.fileCount > 0)
    #expect(report.byteCount > 0)
    #expect(report.maxByteCount == MLXFastConstants.defaultMaxTransformedWeightsBytes)
    #expect(report.sha256.count == 64)
    #expect(report.deterministic)
}

@Test
func transformVerifierRejectsOutputThatDiffersFromFreshTransformRun() throws {
    let fixture = try writeTransformFixture()
    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )
    try "changed".write(
        to: fixture.output.appendingPathComponent("tokenizer.json"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try TransformVerifier.verify(
            TransformVerificationOptions(
                referencePath: fixture.reference.path,
                weightsPath: fixture.output.path,
                temporaryParentPath: fixture.root.path
            )
        )
    }
}

@Test
func transformVerifierRejectsStaleExtraGeneratedFile() throws {
    let fixture = try writeTransformFixture()
    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )
    try "extra".write(
        to: fixture.output.appendingPathComponent("extra.txt"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try TransformVerifier.verify(
            TransformVerificationOptions(
                referencePath: fixture.reference.path,
                weightsPath: fixture.output.path,
                temporaryParentPath: fixture.root.path
            )
        )
    }
}

@Test
func transformVerifierRejectsOutputAboveConfiguredByteLimit() throws {
    let fixture = try writeTransformFixture()
    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )

    #expect(throws: MLXFastError.self) {
        _ = try TransformVerifier.verify(
            TransformVerificationOptions(
                referencePath: fixture.reference.path,
                weightsPath: fixture.output.path,
                temporaryParentPath: fixture.root.path,
                maxByteCount: 1
            )
        )
    }
}

@Test
func transformRejectsUnsupportedIndexShardBeforeCreatingOutput() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try referenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            "language_model.model.layers.0.self_attn.q_proj.weight": "pytorch_model.bin",
        ]
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: reference.path, outputPath: output.path)
        )
    }
    #expect(!FileManager.default.fileExists(atPath: output.path))
}

@Test
func transformRejectsUnsafeIndexShardBeforeCreatingOutput() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try referenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            "language_model.model.layers.0.self_attn.q_proj.weight": "../model-00001.safetensors",
        ]
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: reference.path, outputPath: output.path)
        )
    }
    #expect(!FileManager.default.fileExists(atPath: output.path))
}

@Test
func transformRejectsIndexTensorMissingFromShardHeaderBeforeCreatingOutput() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try referenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [
            TensorFixture(name: "language_model.model.layers.0.self_attn.k_proj.weight", dtype: "U8", shape: [2], data: Data([1, 2])),
        ]
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            "language_model.model.layers.0.self_attn.q_proj.weight": shardName,
        ]
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: reference.path, outputPath: output.path)
        )
    }
    #expect(!FileManager.default.fileExists(atPath: output.path))
}

@Test
func transformAcceptsSparseShardLargerThanInt32() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try referenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let textName = "language_model.model.layers.0.self_attn.q_proj.weight"
    let visionName = "vision_tower.encoder.layers.0.self_attn.q_proj.weight"
    let shardName = "model-00001-of-00001.safetensors"
    let shard = reference.appendingPathComponent(shardName)
    try writeSafetensors(
        shard,
        tensors: [
            TensorFixture(name: textName, dtype: "U8", shape: [1], data: Data([4])),
            TensorFixture(name: visionName, dtype: "U8", shape: [1], data: Data([8])),
        ]
    )
    try truncateFile(shard, toByteCount: Int64(Int32.max) + 1024)
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            textName: shardName,
            visionName: shardName,
        ]
    )

    let report = try SwiftTransform.run(
        TransformOptions(referencePath: reference.path, outputPath: output.path)
    )

    #expect(report.denseTensorCount == 1)
}

private struct TensorFixture {
    let name: String
    let dtype: String
    let shape: [Int]
    let data: Data
}

private struct TransformFixturePaths {
    let root: URL
    let reference: URL
    let output: URL
}

private func writeTransformFixture() throws -> TransformFixturePaths {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try referenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try #"{"tokenizer":"fixture"}"#.write(
        to: reference.appendingPathComponent("tokenizer.json"),
        atomically: true,
        encoding: .utf8
    )

    let textName = "language_model.model.layers.0.self_attn.q_proj.weight"
    let visionName = "vision_tower.encoder.layers.0.self_attn.q_proj.weight"
    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [
            TensorFixture(name: textName, dtype: "U8", shape: [4], data: Data([1, 2, 3, 4])),
            TensorFixture(name: visionName, dtype: "U8", shape: [3], data: Data([9, 8, 7])),
        ]
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            textName: shardName,
            visionName: shardName,
        ]
    )

    return TransformFixturePaths(root: root, reference: reference, output: output)
}

private func referenceConfigJSON() -> String {
    """
    {
      "text_config": {
        "num_hidden_layers": \(MLXFastConstants.numHiddenLayers),
        "vocab_size": \(MLXFastConstants.vocabSize),
        "hidden_size": \(MLXFastConstants.hiddenSize)
      },
      "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
    }
    """
}

private func writeCheckpointIndex(_ path: URL, weightMap: [String: String]) throws {
    let data = try JSONSerialization.data(
        withJSONObject: ["weight_map": weightMap],
        options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    )
    try data.write(to: path)
}

private func writeSafetensors(_ path: URL, tensors: [TensorFixture]) throws {
    var object: [String: Any] = [:]
    var cursor = 0
    for tensor in tensors.sorted(by: { $0.name < $1.name }) {
        object[tensor.name] = [
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [cursor, cursor + tensor.data.count],
        ]
        cursor += tensor.data.count
    }

    var header = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    while header.count % 8 != 0 {
        header.append(0x20)
    }

    var output = Data()
    var headerLength = UInt64(header.count).littleEndian
    output.append(Data(bytes: &headerLength, count: 8))
    output.append(header)
    for tensor in tensors.sorted(by: { $0.name < $1.name }) {
        output.append(tensor.data)
    }
    try output.write(to: path)
}

private func truncateFile(_ path: URL, toByteCount byteCount: Int64) throws {
    let handle = try FileHandle(forWritingTo: path)
    defer {
        try? handle.close()
    }
    try handle.truncate(atOffset: UInt64(byteCount))
}

private func tensorBytes(_ path: URL, header: SafetensorsHeader, name: String) throws -> Data {
    let info = try #require(header.tensors[name])
    let data = try Data(contentsOf: path)
    let start = Int(header.dataBaseOffset) + info.dataStart
    return data.subdata(in: start..<(start + info.byteCount))
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
