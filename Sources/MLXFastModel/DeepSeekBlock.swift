import Foundation
import MLX

public struct DeepSeekBlockSpec {
    public let hcMult: Int
    public let hcSinkhornIters: Int
    public let hcEps: Double
    public let rmsNormEps: Double

    public init(
        hcMult: Int,
        hcSinkhornIters: Int,
        hcEps: Double,
        rmsNormEps: Double
    ) {
        self.hcMult = hcMult
        self.hcSinkhornIters = hcSinkhornIters
        self.hcEps = hcEps
        self.rmsNormEps = rmsNormEps
    }

    public init(config: DeepSeekConfig) {
        self.init(
            hcMult: config.hcMult,
            hcSinkhornIters: config.hcSinkhornIters,
            hcEps: config.hcEps,
            rmsNormEps: config.rmsNormEps
        )
    }
}

public struct DeepSeekHyperConnectionWeights {
    public let fn: MLXArray
    public let base: MLXArray
    public let scale: MLXArray
    // Derived arrays shared across every forward pass. The collapse path
    // consumes fn transposed and mixes/scale/base in float32; hoisting the
    // transpose and the exact bf16->f32 widening here removes those nodes
    // from each of the per-layer collapse calls without changing any value.
    public let fnTransposedF32: MLXArray
    public let baseF32: MLXArray
    public let scaleF32: MLXArray

    public init(fn: MLXArray, base: MLXArray, scale: MLXArray) {
        self.fn = fn
        self.base = base
        self.scale = scale
        self.fnTransposedF32 = DeepSeekOps.cast(fn, to: .float32).T
        self.baseF32 = DeepSeekOps.cast(base, to: .float32)
        self.scaleF32 = DeepSeekOps.cast(scale, to: .float32)
    }
}

public struct DeepSeekBlockWeights {
    public let attentionNorm: MLXArray
    public let feedForwardNorm: MLXArray
    public let attentionHyperConnection: DeepSeekHyperConnectionWeights
    public let feedForwardHyperConnection: DeepSeekHyperConnectionWeights

    public init(
        attentionNorm: MLXArray,
        feedForwardNorm: MLXArray,
        attentionHyperConnection: DeepSeekHyperConnectionWeights,
        feedForwardHyperConnection: DeepSeekHyperConnectionWeights
    ) {
        self.attentionNorm = attentionNorm
        self.feedForwardNorm = feedForwardNorm
        self.attentionHyperConnection = attentionHyperConnection
        self.feedForwardHyperConnection = feedForwardHyperConnection
    }
}

public enum DeepSeekBlock {
    public static func forward(
        hidden: MLXArray,
        weights: DeepSeekBlockWeights,
        spec: DeepSeekBlockSpec,
        attention: (_ normalized: MLXArray) throws -> MLXArray,
        feedForward: (_ normalized: MLXArray) throws -> MLXArray
    ) throws -> MLXArray {
        let attentionResidual = hidden
        let attentionMix = try DeepSeekHyperConnection.collapse(
            hidden,
            fn: weights.attentionHyperConnection.fn,
            fnTransposed: weights.attentionHyperConnection.fnTransposedF32,
            base: weights.attentionHyperConnection.baseF32,
            scale: weights.attentionHyperConnection.scaleF32,
            hcMult: spec.hcMult,
            sinkhornIters: spec.hcSinkhornIters,
            eps: spec.hcEps,
            normEps: spec.rmsNormEps
        )
        let attentionInput = DeepSeekOps.rmsNorm(
            input: attentionMix.collapsed,
            weight: weights.attentionNorm,
            eps: spec.rmsNormEps
        )
        let attentionOutput = try attention(attentionInput)
        var hidden = DeepSeekHyperConnection.expand(
            attentionOutput,
            residual: attentionResidual,
            post: attentionMix.post,
            combination: attentionMix.combination
        )

        let feedForwardResidual = hidden
        let feedForwardMix = try DeepSeekHyperConnection.collapse(
            hidden,
            fn: weights.feedForwardHyperConnection.fn,
            fnTransposed: weights.feedForwardHyperConnection.fnTransposedF32,
            base: weights.feedForwardHyperConnection.baseF32,
            scale: weights.feedForwardHyperConnection.scaleF32,
            hcMult: spec.hcMult,
            sinkhornIters: spec.hcSinkhornIters,
            eps: spec.hcEps,
            normEps: spec.rmsNormEps
        )
        let feedForwardInput = DeepSeekOps.rmsNorm(
            input: feedForwardMix.collapsed,
            weight: weights.feedForwardNorm,
            eps: spec.rmsNormEps
        )
        let feedForwardOutput = try feedForward(feedForwardInput)
        hidden = DeepSeekHyperConnection.expand(
            feedForwardOutput,
            residual: feedForwardResidual,
            post: feedForwardMix.post,
            combination: feedForwardMix.combination
        )
        return hidden
    }
}
