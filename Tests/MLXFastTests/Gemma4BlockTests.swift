import Foundation
import MLX
@testable import MLXFastModel
import Testing

@Test
func gemma4BlockAppliesSandwichNormResidualAndLayerScalarWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let weights = Gemma4BlockWeights(
        inputLayerNorm: ones([2], dtype: .float32),
        postAttentionLayerNorm: ones([2], dtype: .float32),
        preFeedForwardLayerNorm: ones([2], dtype: .float32),
        postFeedForwardLayerNorm: ones([2], dtype: .float32),
        layerScalar: MLXArray([Float(2)], [1])
    )

    var capturedAttentionInput: MLXArray?
    var capturedFeedForwardInput: MLXArray?
    let output = try Gemma4Block.forward(
        hidden: MLXArray([Float(1), 1], [1, 1, 2]),
        weights: weights,
        rmsNormEps: 0,
        attention: { normalized in
            capturedAttentionInput = normalized
            return normalized
        },
        feedForward: { normalized in
            capturedFeedForwardInput = normalized
            return normalized
        }
    )

    // See the worked arithmetic in the module doc comment for Gemma4Block:
    // rmsNorm([1,1]) == [1,1] (already unit RMS), so each sub-layer is a
    // pass-through here, and the residual doubles at each add before the
    // final layer_scalar multiply.
    #expect(output.shape == [1, 1, 2])
    #expect(output.asArray(Float.self) == [6, 6])

    let attentionInput = try #require(capturedAttentionInput).asArray(Float.self)
    let feedForwardInput = try #require(capturedFeedForwardInput).asArray(Float.self)
    #expect(attentionInput == [1, 1])
    #expect(feedForwardInput == [1, 1])
}
