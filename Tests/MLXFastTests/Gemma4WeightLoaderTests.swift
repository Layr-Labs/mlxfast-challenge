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
