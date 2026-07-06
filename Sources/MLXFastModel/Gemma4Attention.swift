import Foundation
import MLX
import MLXFastCore

public struct Gemma4AttentionSpec {
    public let layerType: Gemma4LayerType
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let useKEqV: Bool
    public let ropeSpec: Gemma4RopeSpec
    public let slidingWindow: Int
    public let rmsNormEps: Double

    public init(
        layerType: Gemma4LayerType,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        useKEqV: Bool,
        ropeSpec: Gemma4RopeSpec,
        slidingWindow: Int,
        rmsNormEps: Double
    ) {
        self.layerType = layerType
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.useKEqV = useKEqV
        self.ropeSpec = ropeSpec
        self.slidingWindow = slidingWindow
        self.rmsNormEps = rmsNormEps
    }

    public init(layerIndex: Int, config: Gemma4Config) {
        let layerType = config.layerTypes[layerIndex]
        self.init(
            layerType: layerType,
            numAttentionHeads: config.numAttentionHeads,
            numKeyValueHeads: config.numKeyValueHeads(for: layerType),
            headDim: config.headDim(for: layerType),
            useKEqV: config.usesKEqV(for: layerType),
            ropeSpec: config.rope(for: layerType),
            slidingWindow: config.slidingWindow,
            rmsNormEps: config.rmsNormEps
        )
    }
}

public struct Gemma4AttentionWeights {
    public let qProj: Gemma4LinearWeight
    public let kProj: Gemma4LinearWeight
    public let vProj: Gemma4LinearWeight?
    public let oProj: Gemma4LinearWeight
    public let qNorm: MLXArray
    public let kNorm: MLXArray

    public init(
        qProj: Gemma4LinearWeight,
        kProj: Gemma4LinearWeight,
        vProj: Gemma4LinearWeight?,
        oProj: Gemma4LinearWeight,
        qNorm: MLXArray,
        kNorm: MLXArray
    ) {
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.qNorm = qNorm
        self.kNorm = kNorm
    }
}

/// Gemma 4 self-attention: GQA with per-layer-type head dimensions, an
/// optional shared K/V projection on full-attention layers
/// (`attention_k_eq_v`), and a fixed attention scale of 1.0 -- matching both
/// the mlx-vlm and Hugging Face `transformers` Gemma 4 reference
/// implementations (`self.scaling = 1.0`; QK-norm replaces the usual
/// `1/sqrt(head_dim)` softmax scaling).
public enum Gemma4Attention {
    public static func forward(
        _ x: MLXArray,
        weights: Gemma4AttentionWeights,
        spec: Gemma4AttentionSpec,
        mask: MLXArray? = nil,
        cache: Gemma4LayerCache? = nil,
        positionOffset: Int = 0
    ) throws -> MLXArray {
        try validateInput(x, spec: spec)
        let batchSize = x.shape[0]
        let sequenceLength = x.shape[1]
        let rope = try Gemma4RoPECache.shared(dims: spec.headDim, spec: spec.ropeSpec)

        var q = Gemma4Ops.linear(x, weights.qProj)
            .reshaped([batchSize, sequenceLength, spec.numAttentionHeads, spec.headDim])
        q = Gemma4Ops.rmsNorm(q, weight: weights.qNorm, eps: spec.rmsNormEps)

        let rawK = Gemma4Ops.linear(x, weights.kProj)
            .reshaped([batchSize, sequenceLength, spec.numKeyValueHeads, spec.headDim])
        let rawV: MLXArray
        if spec.useKEqV {
            rawV = rawK
        } else {
            guard let vProj = weights.vProj else {
                throw MLXFastError.invalidInput("Gemma 4 attention layer is missing v_proj weights")
            }
            rawV = Gemma4Ops.linear(x, vProj)
                .reshaped([batchSize, sequenceLength, spec.numKeyValueHeads, spec.headDim])
        }

        var k = Gemma4Ops.rmsNorm(rawK, weight: weights.kNorm, eps: spec.rmsNormEps)
        var v = Gemma4Ops.rmsNormNoWeight(rawV, eps: spec.rmsNormEps)

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        q = rope.applied(to: q, offset: positionOffset)
        k = rope.applied(to: k, offset: positionOffset)

        var attentionMask = mask
        if let cache {
            let cachedK = try cache.keys.updateAndFetch(k)
            let cachedV = try cache.values.updateAndFetch(v)
            k = cachedK.value
            v = cachedV.value
            attentionMask = try Gemma4MaskCache.causal(
                queryLength: sequenceLength,
                keyLength: k.shape[2],
                queryOffset: positionOffset,
                keyOffset: cachedK.offset,
                windowSize: spec.layerType == .sliding ? spec.slidingWindow : nil
            )
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: 1.0,
            mask: attentionMask
        )
        let reshaped = out.transposed(0, 2, 1, 3).reshaped([batchSize, sequenceLength, -1])
        return Gemma4Ops.linear(reshaped, weights.oProj)
    }

    private static func validateInput(_ x: MLXArray, spec: Gemma4AttentionSpec) throws {
        guard x.shape.count == 3 else {
            throw MLXFastError.invalidInput("Gemma 4 attention input must have shape [batch, length, hidden]")
        }
        guard spec.numAttentionHeads > 0, spec.headDim > 0, spec.numKeyValueHeads > 0 else {
            throw MLXFastError.invalidInput("Gemma 4 attention spec dimensions must be positive")
        }
        guard spec.numAttentionHeads % spec.numKeyValueHeads == 0 else {
            throw MLXFastError.invalidInput("Gemma 4 attention heads must be divisible by key/value heads")
        }
    }
}
