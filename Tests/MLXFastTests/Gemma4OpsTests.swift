import Foundation
import MLX
@testable import MLXFastModel
import Testing

@Test
func gemma4OpsRunWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let ids = MLXArray([Int32(2), Int32(0)], [2])
    let embeddingWeight = Gemma4LinearWeight(MLXArray((1...6).map { Float($0) }, [3, 2]))
    let embedded = Gemma4Ops.embedding(inputIDs: ids, weight: embeddingWeight)
    #expect(embedded.shape == [2, 2])
    #expect(embedded.asArray(Float.self) == [5, 6, 1, 2])

    let input = MLXArray([Float(2), Float(3)], [1, 2])
    let weight = Gemma4LinearWeight(MLXArray([Float(1), Float(10), Float(2), Float(20)], [2, 2]))
    let projected = Gemma4Ops.linear(input, weight)
    #expect(projected.shape == [1, 2])
    #expect(projected.asArray(Float.self) == [32, 64])

    let denseQuantizedWeight = MLXArray((0..<128).map { Float($0) / 128 }, [2, 64])
    let (packedWeight, scales, quantBiases) = quantized(
        denseQuantizedWeight,
        groupSize: 64,
        bits: 4,
        mode: .affine
    )
    let quantizedWeight = Gemma4LinearWeight(
        weight: packedWeight,
        scales: scales,
        biases: quantBiases,
        logicalShape: [2, 64],
        groupSize: 64,
        bits: 4
    )
    let quantizedProjected = Gemma4Ops.linear(
        MLXArray(Array(repeating: Float(1), count: 64), [1, 64]),
        quantizedWeight
    )
    #expect(quantizedProjected.shape == [1, 2])

    let quantizedEmbedding = Gemma4Ops.embedding(
        inputIDs: MLXArray([Int32(1)], [1]),
        weight: quantizedWeight
    )
    #expect(quantizedEmbedding.shape == [1, 64])

    let norm = Gemma4Ops.rmsNorm(
        MLXArray([Float(3), Float(4)], [1, 2]),
        weight: MLXArray([Float(1), Float(1)], [2]),
        eps: 0
    ).asArray(Float.self)
    #expect(abs(norm[0] - 0.84852815) < 1e-5)
    #expect(abs(norm[1] - 1.1313709) < 1e-5)

    let normNoWeight = Gemma4Ops.rmsNormNoWeight(
        MLXArray([Float(3), Float(4)], [1, 2]),
        eps: 0
    ).asArray(Float.self)
    #expect(abs(normNoWeight[0] - 0.84852815) < 1e-5)
    #expect(abs(normNoWeight[1] - 1.1313709) < 1e-5)

    // gelu_pytorch_tanh(0) == 0, and the activation is monotonic increasing.
    let gelu = Gemma4Ops.geluTanh(MLXArray([Float(0), Float(1), Float(-1)], [3])).asArray(Float.self)
    #expect(abs(gelu[0] - 0) < 1e-6)
    #expect(gelu[1] > 0.5)
    #expect(gelu[2] < 0)

    let softcapped = Gemma4Ops.logitSoftcap(MLXArray([Float(60)], [1]), cap: 30.0).asArray(Float.self)
    #expect(softcapped[0] < 30.0)
    #expect(softcapped[0] > 0)

    let unchanged = Gemma4Ops.logitSoftcap(MLXArray([Float(5)], [1]), cap: 0).asArray(Float.self)
    #expect(unchanged[0] == 5)
}
