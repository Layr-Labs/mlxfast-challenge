import Foundation
import MLX
import Testing
@testable import MLXFastCore
@testable import MLXFastModel

@Test
func gemma4WeightLoaderReadsPlainDenseTensor() throws {
    let weights = try makeWeightsFixture(tensors: [
        TensorFixture(
            name: "language_model.model.norm.weight",
            dtype: "U8",
            shape: [2, 2],
            data: Data([1, 2, 3, 4])
        )
    ])

    let loader = try Gemma4WeightLoader(weightsPath: weights.path)
    let tensor = try loader.materializedTensor(
        named: "language_model.model.norm.weight",
        expectedShape: [2, 2]
    )
    #expect(try tensor.uint8Values() == [1, 2, 3, 4])

    #expect(throws: MLXFastError.self) {
        _ = try loader.materializedTensor(
            named: "language_model.model.norm.weight",
            expectedShape: [4, 1]
        )
    }
}

@Test
func gemma4WeightLoaderRaisesForMissingTensor() throws {
    let weights = try makeWeightsFixture(tensors: [
        TensorFixture(name: "language_model.model.norm.weight", dtype: "U8", shape: [1], data: Data([1]))
    ])

    let loader = try Gemma4WeightLoader(weightsPath: weights.path)
    #expect(throws: MLXFastError.self) {
        _ = try loader.materializedTensor(named: "language_model.model.embed_tokens.weight")
    }
}

@Test
func gemma4WeightLoaderBuildsAffineQuantizedLinearWeightWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let weights = try makeWeightsFixture(tensors: [
        TensorFixture(
            name: "language_model.model.layers.0.self_attn.q_proj.weight",
            dtype: "U32",
            shape: [2, 1],
            data: uint32Bytes([1, 2])
        ),
        TensorFixture(
            name: "language_model.model.layers.0.self_attn.q_proj.scales",
            dtype: "BF16",
            shape: [2, 2],
            data: Data(repeating: 0, count: 8)
        ),
        TensorFixture(
            name: "language_model.model.layers.0.self_attn.q_proj.biases",
            dtype: "BF16",
            shape: [2, 2],
            data: Data(repeating: 0, count: 8)
        ),
    ])

    let loader = try Gemma4WeightLoader(weightsPath: weights.path)
    let weight = try loader.linearWeight(
        named: "language_model.model.layers.0.self_attn.q_proj.weight",
        outFeatures: 2,
        inFeatures: 8
    )

    #expect(weight.isQuantized)
    #expect(weight.shape == [2, 8])
    #expect(weight.weight.shape == [2, 1])
    #expect(weight.scales?.shape == [2, 2])
    #expect(weight.biases?.shape == [2, 2])
    #expect(weight.bits == 4)
    #expect(weight.groupSize == 4)
}

@Test
func gemma4WeightLoaderValidatesAffineQuantizedMetadataWithoutMaterializing() throws {
    let weights = try makeWeightsFixture(tensors: validQuantizedLinearFixtures())
    defer { try? FileManager.default.removeItem(at: weights.deletingLastPathComponent()) }
    let loader = try Gemma4WeightLoader(weightsPath: weights.path)

    let quantization = try loader.validateLinearMetadata(
        named: quantizedLinearName,
        expectedRows: 2,
        expectedInputFeatures: 8
    )

    #expect(quantization?.groupSize == 4)
    #expect(quantization?.bits == 4)
}

@Test
func gemma4WeightLoaderRejectsQuantizationMetadataThatDisagreesWithConfig() throws {
    let weights = try makeWeightsFixture(tensors: validQuantizedLinearFixtures())
    defer { try? FileManager.default.removeItem(at: weights.deletingLastPathComponent()) }
    let loader = try Gemma4WeightLoader(weightsPath: weights.path)

    #expect(throws: MLXFastError.self) {
        _ = try loader.validateLinearMetadata(
            named: quantizedLinearName,
            expectedRows: 2,
            expectedInputFeatures: 8,
            expectedGroupSize: 8,
            expectedBits: 4
        )
    }
    #expect(throws: MLXFastError.self) {
        _ = try loader.validateLinearMetadata(
            named: quantizedLinearName,
            expectedRows: 2,
            expectedInputFeatures: 8,
            expectedGroupSize: 4,
            expectedBits: 8
        )
    }
}

@Test
func gemma4WeightLoaderRequiresBothAffineQuantizationCompanions() throws {
    let weight = quantizedWeightFixture()
    let scales = quantizedCompanionFixture(suffix: "scales")
    let malformedFixtures = [
        [weight],
        [weight, scales],
    ]

    for tensors in malformedFixtures {
        let weights = try makeWeightsFixture(tensors: tensors)
        defer { try? FileManager.default.removeItem(at: weights.deletingLastPathComponent()) }
        let loader = try Gemma4WeightLoader(weightsPath: weights.path)

        #expect(throws: MLXFastError.self) {
            _ = try loader.validateLinearMetadata(
                named: quantizedLinearName,
                expectedRows: 2,
                expectedInputFeatures: 8
            )
        }
    }
}

@Test
func gemma4WeightLoaderRejectsNonBF16AffineQuantizationCompanions() throws {
    let malformedCompanions = [
        (
            quantizedCompanionFixture(suffix: "scales", dtype: "F16"),
            quantizedCompanionFixture(suffix: "biases")
        ),
        (
            quantizedCompanionFixture(suffix: "scales"),
            quantizedCompanionFixture(suffix: "biases", dtype: "F16")
        ),
    ]

    for (scales, biases) in malformedCompanions {
        let weights = try makeWeightsFixture(tensors: [quantizedWeightFixture(), scales, biases])
        defer { try? FileManager.default.removeItem(at: weights.deletingLastPathComponent()) }
        let loader = try Gemma4WeightLoader(weightsPath: weights.path)

        #expect(throws: MLXFastError.self) {
            _ = try loader.validateLinearMetadata(
                named: quantizedLinearName,
                expectedRows: 2,
                expectedInputFeatures: 8
            )
        }
    }
}

@Test
func gemma4WeightLoaderRejectsMalformedAffineQuantizationCompanionShapes() throws {
    let malformedCompanions = [
        (
            quantizedCompanionFixture(suffix: "scales", shape: [4]),
            quantizedCompanionFixture(suffix: "biases", shape: [4])
        ),
        (
            quantizedCompanionFixture(suffix: "scales", shape: [1, 4]),
            quantizedCompanionFixture(suffix: "biases", shape: [1, 4])
        ),
        (
            quantizedCompanionFixture(suffix: "scales"),
            quantizedCompanionFixture(suffix: "biases", shape: [2, 1])
        ),
    ]

    for (scales, biases) in malformedCompanions {
        let weights = try makeWeightsFixture(tensors: [quantizedWeightFixture(), scales, biases])
        defer { try? FileManager.default.removeItem(at: weights.deletingLastPathComponent()) }
        let loader = try Gemma4WeightLoader(weightsPath: weights.path)

        #expect(throws: MLXFastError.self) {
            _ = try loader.validateLinearMetadata(
                named: quantizedLinearName,
                expectedRows: 2,
                expectedInputFeatures: 8
            )
        }
    }
}

@Test
func runtimeShardInventoryRejectsMissingAndUnindexedShards() throws {
    #expect(try validateRuntimeShardInventory(
        referencedShards: ["b.safetensors", "a.safetensors"],
        discoveredShards: ["a.safetensors", "b.safetensors"]
    ) == ["a.safetensors", "b.safetensors"])

    #expect(throws: MLXFastError.self) {
        _ = try validateRuntimeShardInventory(
            referencedShards: ["a.safetensors", "missing.safetensors"],
            discoveredShards: ["a.safetensors"]
        )
    }
    #expect(throws: MLXFastError.self) {
        _ = try validateRuntimeShardInventory(
            referencedShards: ["a.safetensors"],
            discoveredShards: ["a.safetensors", "unindexed.safetensors"]
        )
    }
}

@Test
func runtimeTensorInventoryRejectsMissingAndUnindexedTensors() throws {
    try validateRuntimeTensorInventory(
        shardName: "a.safetensors",
        expectedNames: Set(["tensor.a", "tensor.b"]),
        discoveredNames: Set(["tensor.a", "tensor.b"])
    )
    #expect(throws: MLXFastError.self) {
        try validateRuntimeTensorInventory(
            shardName: "a.safetensors",
            expectedNames: Set(["tensor.a", "tensor.missing"]),
            discoveredNames: Set(["tensor.a"])
        )
    }
    #expect(throws: MLXFastError.self) {
        try validateRuntimeTensorInventory(
            shardName: "a.safetensors",
            expectedNames: Set(["tensor.a"]),
            discoveredNames: Set(["tensor.a", "tensor.unindexed"])
        )
    }
}

@Test
func runtimeWeightLoaderStreamsIndexedTensorsWithoutMLX() throws {
    let firstName = "language_model.model.a.weight"
    let secondName = "language_model.model.b.weight"
    let weights = try makeWeightsFixture(tensors: [
        TensorFixture(
            name: firstName,
            dtype: "U8",
            shape: [3],
            data: Data([1, 2, 3])
        ),
        TensorFixture(
            name: secondName,
            dtype: "U8",
            shape: [2],
            data: Data([4, 5])
        ),
    ])
    defer { try? FileManager.default.removeItem(at: weights.deletingLastPathComponent()) }
    let store = try DenseTensorStore(weightsPath: weights.path)
    var materializedNames: [String] = []

    let values: [String: [UInt8]] = try loadRuntimeWeightValues(
        denseStore: store
    ) { tensor in
        materializedNames.append(tensor.name)
        return Array(tensor.bytes)
    }

    #expect(materializedNames == [firstName, secondName])
    #expect(values["model.a.weight"] == [1, 2, 3])
    #expect(values["model.b.weight"] == [4, 5])
}

@Test
func runtimeWeightNameTrackerRejectsMisplacedDuplicateAndRenamedCollisions() throws {
    var misplaced = RuntimeWeightNameTracker()
    #expect(throws: MLXFastError.self) {
        _ = try misplaced.register(
            originalName: "tensor.b",
            shardName: "a.safetensors",
            expectedNames: Set(["tensor.a"])
        )
    }

    var duplicate = RuntimeWeightNameTracker()
    let duplicateNames = Set(["language_model.model.tensor"])
    #expect(try duplicate.register(
        originalName: "language_model.model.tensor",
        shardName: "a.safetensors",
        expectedNames: duplicateNames
    ) == "model.tensor")
    #expect(throws: MLXFastError.self) {
        _ = try duplicate.register(
            originalName: "language_model.model.tensor",
            shardName: "b.safetensors",
            expectedNames: duplicateNames
        )
    }

    var renamedCollision = RuntimeWeightNameTracker()
    let collidingNames = Set(["language_model.model.tensor", "model.tensor"])
    _ = try renamedCollision.register(
        originalName: "language_model.model.tensor",
        shardName: "a.safetensors",
        expectedNames: collidingNames
    )
    #expect(throws: MLXFastError.self) {
        _ = try renamedCollision.register(
            originalName: "model.tensor",
            shardName: "a.safetensors",
            expectedNames: collidingNames
        )
    }
}

@Test
func runtimeWeightNameTrackerRequiresEveryIndexedTensor() throws {
    var tracker = RuntimeWeightNameTracker()
    _ = try tracker.register(
        originalName: "language_model.model.present",
        shardName: "a.safetensors",
        expectedNames: Set(["language_model.model.present"])
    )

    #expect(throws: MLXFastError.self) {
        try tracker.validateComplete(expectedNames: Set([
            "language_model.model.present",
            "language_model.model.missing",
        ]))
    }
}

// A full end-to-end `validateRequiredMetadata` pass against the frozen (60
// real-size layer) config is covered by BenchmarkSupportTests'
// `benchmarkPreflightAcceptsRequiredArtifacts` and friends -- `Gemma4Config`
// enforces the frozen shape invariants on every load, so a tiny fixture
// config cannot exercise this loader method directly.

private struct TensorFixture {
    let name: String
    let dtype: String
    let shape: [Int]
    let data: Data
}

private let quantizedLinearName = "language_model.model.layers.0.self_attn.q_proj.weight"

private func quantizedWeightFixture() -> TensorFixture {
    TensorFixture(
        name: quantizedLinearName,
        dtype: "U32",
        shape: [2, 1],
        data: uint32Bytes([1, 2])
    )
}

private func quantizedCompanionFixture(
    suffix: String,
    dtype: String = "BF16",
    shape: [Int] = [2, 2]
) -> TensorFixture {
    let elementWidth = 2
    return TensorFixture(
        name: quantizedLinearName.replacingOccurrences(of: ".weight", with: ".\(suffix)"),
        dtype: dtype,
        shape: shape,
        data: Data(repeating: 0, count: shape.reduce(1, *) * elementWidth)
    )
}

private func validQuantizedLinearFixtures() -> [TensorFixture] {
    [
        quantizedWeightFixture(),
        quantizedCompanionFixture(suffix: "scales"),
        quantizedCompanionFixture(suffix: "biases"),
    ]
}

private func makeWeightsFixture(tensors: [TensorFixture]) throws -> URL {
    let root = try temporaryDirectory()
    let weights = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)

    let shard = "model-00001.safetensors"
    try writeSafetensors(weights.appendingPathComponent(shard), tensors: tensors)
    try writeIndex(
        weights.appendingPathComponent("model.safetensors.index.json"),
        tensors: tensors,
        shardName: shard
    )
    return weights
}

private func writeIndex(_ path: URL, tensors: [TensorFixture], shardName: String) throws {
    let entries = tensors.map { #""\#($0.name)": "\#(shardName)""# }.joined(separator: ",")
    try """
    {
      "weight_map": {
        \(entries)
      }
    }
    """.write(to: path, atomically: true, encoding: .utf8)
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

private func uint32Bytes(_ values: [UInt32]) -> Data {
    var data = Data()
    for value in values {
        var littleEndian = value.littleEndian
        data.append(Data(bytes: &littleEndian, count: 4))
    }
    return data
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
