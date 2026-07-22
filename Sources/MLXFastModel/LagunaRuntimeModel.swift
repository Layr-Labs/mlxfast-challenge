import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Correctness-first Laguna runtime (Poolside Laguna XS 2.1, 256-expert MoE).
//
// This module tree closely follows the vendored reference implementation at
// `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Laguna.swift` (`LagunaModel` /
// `LagunaModelInner`), which is the behavior oracle for this port. It is a
// reimplementation rather than a wrapper for two load-bearing reasons:
//
// 1. The pinned `mlx-community/Laguna-XS-2.1-4bit` checkpoint stores the MoE
//    router as a quantized linear submodule (`mlp.gate.proj.{weight,scales,
//    biases}`, 8-bit group-64) next to `mlp.gate.e_score_correction_bias`.
//    The vendored `LagunaMoEGate` models the router as a raw
//    `gate.weight` MLXArray (the original Poolside NVFP4 layout), so its
//    parameter paths cannot absorb this checkpoint's router tensors.
//    `LagunaRuntimeMoEGate` below restructures the router as a `proj`
//    Linear child while keeping the routing math identical.
// 2. The vendored `LagunaModelInner`/`LagunaDecoderLayer` types are
//    fileprivate and `LagunaConfiguration`'s stored properties are internal
//    to MLXLLM, so the runtime layers (cache geometry, future fast-engine
//    and exact-verification waves) could not reach the internals through a
//    plain wrapper.
//
// All math is expressed with standard MLX ops and the vendored shared
// primitives (`attentionWithCacheUpdate`, `initializeRope`,
// `applyRotaryPosition`, `SwitchGLU`, `weightedExpertSum`, `RMSNorm`,
// `createAttentionMask`). No custom Metal kernels in this increment; the
// fused fast-engine and exact-pair/exact-four style optimizations are a
// later layer on top of this reference target.

func lagunaLastTokenRange(sequenceLength: Int) -> Range<Int>? {
    sequenceLength > 1 ? (sequenceLength - 1)..<sequenceLength : nil
}

func lagunaLastTokenHidden(_ hidden: MLXArray) -> MLXArray {
    guard let range = lagunaLastTokenRange(sequenceLength: hidden.dim(1)) else {
        return hidden
    }
    return hidden[0..., range, 0...]
}

/// Builds the `initializeRope` scaling dictionary for a per-type Laguna RoPE
/// spec. For `default` RoPE only the type is consulted; for YaRN the factory
/// reads factor / original context / betas (Laguna leaves mscale and
/// mscale_all_dim at the shared defaults 1.0 / 0.0, which yields the vendored
/// attention scaling of `0.1 * ln(32) + 1`).
func lagunaRopeScalingConfig(_ spec: LagunaRopeSpec) -> [String: StringOrNumber] {
    var scalingConfig: [String: StringOrNumber] = ["rope_type": .string(spec.type)]
    if spec.type == "yarn" {
        scalingConfig["factor"] = .float(Float(spec.factor))
        scalingConfig["original_max_position_embeddings"] = .int(
            spec.originalMaxPositionEmbeddings)
        scalingConfig["beta_fast"] = .float(Float(spec.betaFast))
        scalingConfig["beta_slow"] = .float(Float(spec.betaSlow))
    }
    return scalingConfig
}

// MARK: - Attention

/// Laguna attention: GQA with per-head QK-norm, per-layer-type RoPE (YaRN on
/// full-attention layers over the first half of the head, plain RoPE on
/// sliding layers over the whole head), and per-head softplus output gating.
/// Mirrors the vendored `LagunaAttention` forward exactly.
final class LagunaRuntimeAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let gatingEnabled: Bool
    let gatePerHead: Bool
    let isSliding: Bool

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ config: LagunaConfig, layerIdx: Int) {
        let dim = config.hiddenSize
        self.nHeads = config.heads(forLayer: layerIdx)
        self.nKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)
        self.gatingEnabled = config.gating.enabled
        self.gatePerHead = config.gating.isPerHead

        let layerType = config.layerType(forLayer: layerIdx)
        self.isSliding = layerType == .sliding

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: config.qkvBias)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: config.qkvBias)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: config.qkvBias)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: config.attentionBias)

        if gatingEnabled {
            let gateDim = gatePerHead ? nHeads : nHeads * headDim
            self._gProj.wrappedValue = Linear(dim, gateDim, bias: false)
        }

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))

        let ropeSpec = config.rope(for: layerType)
        let ropeDims = Int(Float(headDim) * Float(ropeSpec.partialRotaryFactor))
        self.rope = initializeRope(
            dims: ropeDims,
            base: Float(ropeSpec.theta),
            traditional: false,
            scalingConfig: lagunaRopeScalingConfig(ropeSpec),
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = qNorm(queries.reshaped(B, L, nHeads, headDim)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, nKVHeads, headDim)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        if gatingEnabled, let gProj {
            // Per-head softplus gate computed in float32, then broadcast
            // across the head dimension (or applied elementwise for a
            // per-element gate).
            let gate = softplus(gProj(x).asType(.float32)).asType(output.dtype)
            if gatePerHead {
                output =
                    (output.reshaped(B, L, nHeads, headDim) * gate[.ellipsis, .newAxis])
                    .reshaped(B, L, -1)
            } else {
                output = output * gate
            }
        }

        return wo(output)
    }
}

// MARK: - Dense MLP (also used as the shared expert)

final class LagunaRuntimeMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - MoE

/// Sigmoid top-k router. The routing math mirrors the vendored
/// `LagunaMoEGate` exactly (sigmoid scores, correction bias added only for
/// expert CHOICE, mixture weights taken from the pre-bias scores, optional
/// top-k renormalization). The structural difference from the vendored type
/// is the projection: the pinned checkpoint ships the router as a quantized
/// linear child (`gate.proj.{weight,scales,biases}`, 8-bit group-64), so the
/// projection lives in a `proj` submodule instead of a raw `weight` array.
final class LagunaRuntimeMoEGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let routerLogitSoftcapping: Float

    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: LagunaConfig) {
        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.routerLogitSoftcapping = Float(config.moeRouterLogitSoftcapping)
        self._proj.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        self._eScoreCorrectionBias.wrappedValue = zeros([config.numExperts])
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var logits = proj(x).asType(.float32)
        if routerLogitSoftcapping > 0 {
            logits = tanh(logits / routerLogitSoftcapping) * routerLogitSoftcapping
        }

        let scores = sigmoid(logits)
        let scoresForChoice = scores + eScoreCorrectionBias.asType(scores.dtype)

        let inds = argPartition(-scoresForChoice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, inds, axis: -1)
        if normTopkProb {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
        return (inds, weights)
    }
}

final class LagunaRuntimeSparseMoEBlock: Module, UnaryLayer {
    let routedScalingFactor: Float

    @ModuleInfo(key: "gate") var gate: LagunaRuntimeMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: LagunaRuntimeMLP

    init(_ config: LagunaConfig) {
        self.routedScalingFactor = Float(config.moeRoutedScalingFactor)
        self._gate.wrappedValue = LagunaRuntimeMoEGate(config)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts
        )
        self._sharedExpert.wrappedValue = LagunaRuntimeMLP(
            dimensions: config.hiddenSize,
            hiddenDimensions: config.sharedExpertIntermediateSize
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, weights) = gate(x)
        var y = switchMLP(x, inds)
        y = weightedExpertSum(y, weights.asType(y.dtype))
        if routedScalingFactor != 1 {
            y = y * routedScalingFactor
        }
        return y + sharedExpert(x)
    }
}

// MARK: - Decoder Layer

final class LagunaRuntimeDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: LagunaRuntimeAttention
    let mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let attentionType: LagunaLayerType

    init(_ config: LagunaConfig, layerIdx: Int) {
        self._selfAttn.wrappedValue = LagunaRuntimeAttention(config, layerIdx: layerIdx)

        if config.isSparse(layer: layerIdx) {
            self.mlp = LagunaRuntimeSparseMoEBlock(config)
        } else {
            self.mlp = LagunaRuntimeMLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))

        self.attentionType = config.layerType(forLayer: layerIdx)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

// MARK: - Model

/// The Laguna text tower: embedding (NOT scaled -- unlike Gemma there is no
/// `sqrt(hidden)` embedding multiplier), 40 decoder layers, final RMSNorm.
/// Returns post-norm hidden states for every input position.
final class LagunaRuntimeModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [LagunaRuntimeDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let layerTypes: [LagunaLayerType]
    let slidingWindow: Int
    let fullAttentionIdx: Int
    let slidingAttentionIdx: Int

    init(_ config: LagunaConfig) {
        precondition(config.vocabSize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

        self._layers.wrappedValue = (0..<config.numHiddenLayers).map {
            LagunaRuntimeDecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))

        self.layerTypes = config.layerTypes
        self.slidingWindow = config.slidingWindow
        self.fullAttentionIdx = config.layerTypes.firstIndex(of: .full) ?? 0
        self.slidingAttentionIdx = config.layerTypes.firstIndex(of: .sliding) ?? 0
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        // One mask per attention family, derived from a representative
        // layer's cache offset: all full-attention caches advance in
        // lockstep, as do all sliding caches (vendored `LagunaModelInner`
        // convention).
        let fullMask = createAttentionMask(h: h, cache: cache?[fullAttentionIdx])
        let slidingMask = createAttentionMask(
            h: h, cache: cache?[slidingAttentionIdx], windowSize: slidingWindow)

        for (i, layer) in layers.enumerated() {
            let mask = layerTypes[i] == .full ? fullMask : slidingMask
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

/// Scored Laguna runtime model: last-token vocabulary head over the
/// reimplemented Laguna text tower.
///
/// Interface shape mirrors `Gemma4RuntimeModel` so the worker can swap the
/// model type: `callAsFunction(_:cache:)` serves both prompt prefill
/// (`[1, L]`) and single-token decode steps (`[1, 1]`) and returns
/// `[1, 1, vocab]` last-token logits; `newCache(parameters:)` creates the
/// per-layer cache stack (unbounded `StandardKVCache` for full-attention
/// layers, `RotatingKVCache(512)` for sliding layers). Laguna applies NO
/// final logit softcap and NO embedding scaling.
public final class LagunaRuntimeModel: Module, LanguageModel {
    @ModuleInfo(key: "model") var model: LagunaRuntimeModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let configuration: LagunaConfig

    public init(_ config: LagunaConfig) {
        self.configuration = config
        self._model.wrappedValue = LagunaRuntimeModelInner(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let fullHidden = model(inputs, cache: cache)
        // Every consumer of multi-token logits reads only the LAST
        // position's row; slicing before the head removes a
        // [length-1, vocab]-sized slab of dead work from every prefill.
        let hidden = lagunaLastTokenHidden(fullHidden)
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    public func prepare(
        _ input: LMInput,
        cache _: [KVCache],
        windowSize _: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    public func newCache(parameters _: GenerateParameters?) -> [KVCache] {
        (0..<configuration.numHiddenLayers).map { layerIndex in
            if configuration.layerTypes[layerIndex] == .full {
                StandardKVCache()
            } else {
                RotatingKVCache(maxSize: configuration.slidingWindow, keep: 0)
            }
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }
        // Drop precomputed rotary tables if a checkpoint ships them.
        return weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
    }
}
