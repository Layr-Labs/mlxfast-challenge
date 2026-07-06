import Foundation
import MLX
import MLXFastCore

public struct Gemma4BlockWeights {
    public let inputLayerNorm: MLXArray
    public let postAttentionLayerNorm: MLXArray
    public let preFeedForwardLayerNorm: MLXArray
    public let postFeedForwardLayerNorm: MLXArray
    public let layerScalar: MLXArray

    public init(
        inputLayerNorm: MLXArray,
        postAttentionLayerNorm: MLXArray,
        preFeedForwardLayerNorm: MLXArray,
        postFeedForwardLayerNorm: MLXArray,
        layerScalar: MLXArray
    ) {
        self.inputLayerNorm = inputLayerNorm
        self.postAttentionLayerNorm = postAttentionLayerNorm
        self.preFeedForwardLayerNorm = preFeedForwardLayerNorm
        self.postFeedForwardLayerNorm = postFeedForwardLayerNorm
        self.layerScalar = layerScalar
    }
}

/// Gemma 4 decoder layer: sandwich norms around both attention and the MLP
/// (pre-norm before the sub-layer, post-norm on the sub-layer's own output
/// before the residual add -- matching Gemma 2/3's convention), followed by a
/// per-layer learned scalar applied to the whole residual stream.
public enum Gemma4Block {
    public static func forward(
        hidden: MLXArray,
        weights: Gemma4BlockWeights,
        rmsNormEps: Double,
        attention: (_ normalized: MLXArray) throws -> MLXArray,
        feedForward: (_ normalized: MLXArray) throws -> MLXArray
    ) throws -> MLXArray {
        var residual = hidden
        var h = Gemma4Ops.rmsNorm(hidden, weight: weights.inputLayerNorm, eps: rmsNormEps)
        h = try attention(h)
        h = Gemma4Ops.rmsNorm(h, weight: weights.postAttentionLayerNorm, eps: rmsNormEps)
        var out = residual + h

        residual = out
        h = Gemma4Ops.rmsNorm(out, weight: weights.preFeedForwardLayerNorm, eps: rmsNormEps)
        h = try feedForward(h)
        h = Gemma4Ops.rmsNorm(h, weight: weights.postFeedForwardLayerNorm, eps: rmsNormEps)
        out = residual + h

        return out * weights.layerScalar
    }
}
