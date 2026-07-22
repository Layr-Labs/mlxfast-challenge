import CoreFoundation
import Foundation
import MLXFastCore

/// Frozen invariants of the pinned Poolside Laguna XS 2.1 4-bit target
/// (`mlx-community/Laguna-XS-2.1-4bit`). They live here because
/// `MLXFastCore` is trusted harness code outside the editable surface.
///
/// Laguna is a 256-expert MoE decoder: 40 layers, hidden 2048, GQA with 8 KV
/// heads and head dim 128, mixed full-attention (48 query heads, YaRN partial
/// RoPE) and sliding-window layers (64 query heads, plain RoPE, window 512),
/// a dense MLP only at layer 0, and sigmoid top-8 routing with a shared
/// expert on layers 1-39. The vocabulary head is untied.
public enum LagunaConstants {
    public static let modelType = "laguna"
    public static let vocabSize = 100_352
    public static let hiddenSize = 2_048
    /// Dense MLP intermediate size (layer 0 only).
    public static let denseIntermediateSize = 8_192
    public static let numHiddenLayers = 40
    public static let numKeyValueHeads = 8
    public static let headDim = 128
    /// Query head count on full-attention layers (indices 0, 4, 8, ..., 36).
    public static let fullAttentionHeads = 48
    /// Query head count on sliding-window layers (the other 30 layers).
    public static let slidingAttentionHeads = 64
    public static let slidingWindow = 512
    public static let numExperts = 256
    public static let numExpertsPerTok = 8
    public static let moeIntermediateSize = 512
    public static let sharedExpertIntermediateSize = 512
    public static let moeRoutedScalingFactor = 2.5
    /// `bos_token_id` from the pinned checkpoint config; used only for
    /// prompt-independent warmup forwards (never for scored decoding).
    public static let bosTokenID = 2
    public static let eosTokenIDs = [2, 24]
    /// Global affine quantization of the mlx-community export. Every
    /// projection (embed, attention, MLP, experts, shared expert, untied
    /// lm_head) is affine 4-bit group-64 except the MoE router gate.
    public static let quantizationGroupSize = 64
    public static let quantizationBits = 4
    /// The sparse layers' router (`mlp.gate.proj`) is stored at 8 bits
    /// (group 64) via per-tensor overrides in the checkpoint's
    /// `quantization` config block.
    public static let routerGateQuantizationBits = 8
}

/// Attention layer type for a single Laguna decoder layer. Laguna alternates
/// three sliding-window layers with one full-attention layer per block of
/// four (`layer_types` in the source config), with per-type query head
/// counts and RoPE parameters.
public enum LagunaLayerType: String, Equatable {
    case sliding = "sliding_attention"
    case full = "full_attention"
}

/// MLP kind for a single decoder layer (`mlp_layer_types` in the source
/// config): layer 0 is a dense gated-SiLU MLP, layers 1-39 are sparse MoE
/// blocks (256 routed experts + 1 shared expert).
public enum LagunaMLPType: String, Equatable {
    case dense
    case sparse
}

/// Attention output gating mode (`gating` in the source config). The pinned
/// checkpoint uses per-head gating: `g_proj` produces one softplus gate per
/// query head, broadcast across the head dimension.
public enum LagunaGatingMode: Equatable {
    case disabled
    case perHead
    case perElement

    public var enabled: Bool { self != .disabled }
    public var isPerHead: Bool { self == .perHead }
}

/// Per-attention-type RoPE parameters. Sliding layers use standard rotary
/// embeddings over the full head dimension (theta 1e4); full-attention
/// layers use YaRN (theta 5e5, factor 32, original context 8192, beta
/// 64/1) over the first half of the head (`partial_rotary_factor` 0.5).
///
/// The YaRN fields are consulted only when `type == "yarn"`; they default to
/// the pinned Laguna values so an omitted field never silently falls back to
/// the different generic defaults inside `initializeRope`.
public struct LagunaRopeSpec: Equatable {
    public let theta: Double
    public let type: String
    public let partialRotaryFactor: Double
    public let factor: Double
    public let originalMaxPositionEmbeddings: Int
    public let betaFast: Double
    public let betaSlow: Double

    public init(
        theta: Double,
        type: String,
        partialRotaryFactor: Double,
        factor: Double = 1.0,
        originalMaxPositionEmbeddings: Int = 8_192,
        betaFast: Double = 64.0,
        betaSlow: Double = 1.0
    ) {
        self.theta = theta
        self.type = type
        self.partialRotaryFactor = partialRotaryFactor
        self.factor = factor
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
        self.betaFast = betaFast
        self.betaSlow = betaSlow
    }
}

/// Per-tensor quantization override. The checkpoint's `quantization` block
/// interleaves the global `{group_size, bits, mode}` scalars with
/// tensor-stem-keyed dictionaries (e.g.
/// `"language_model.model.layers.1.mlp.gate.proj": {"group_size": 64,
/// "bits": 8}` for the 8-bit router gates).
public struct LagunaQuantizationOverride: Equatable {
    public let groupSize: Int
    public let bits: Int

    public init(groupSize: Int, bits: Int) {
        self.groupSize = groupSize
        self.bits = bits
    }
}

public struct LagunaQuantizationSpec: Equatable {
    public let groupSize: Int
    public let bits: Int
    public let mode: String
    /// Keyed by the source tensor stem without the trailing `.weight`
    /// (e.g. `language_model.model.layers.1.mlp.gate.proj`).
    public let overrides: [String: LagunaQuantizationOverride]

    public init(
        groupSize: Int,
        bits: Int,
        mode: String,
        overrides: [String: LagunaQuantizationOverride]
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.overrides = overrides
    }

    /// Expected quantization for a quantized tensor, resolved from the
    /// per-tensor overrides with fallback to the global spec. `stem` is the
    /// source tensor name without the trailing `.weight`.
    public func expected(forTensorStem stem: String) -> (groupSize: Int, bits: Int) {
        if let override = overrides[stem] {
            return (override.groupSize, override.bits)
        }
        return (groupSize, bits)
    }
}

/// Text-tower configuration for Poolside Laguna XS 2.1 (256-expert MoE,
/// 4-bit). Parsed from the config.json in the transformed weights directory
/// -- the transform controls this schema directly, so it mirrors the source
/// Hugging Face config fields the runtime actually needs (the source config
/// is flat; the empty `vision_config` object is ignored).
public struct LagunaConfig: Equatable {
    public let modelType: String
    public let vocabSize: Int
    public let hiddenSize: Int
    /// Dense MLP intermediate size (used only by dense layers; layer 0).
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    /// Fallback query head count when `num_attention_heads_per_layer` is
    /// absent for a layer. The pinned config always ships the per-layer list.
    public let numAttentionHeads: Int
    public let numAttentionHeadsPerLayer: [Int]
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let rmsNormEps: Double
    public let maxPositionEmbeddings: Int
    public let attentionBias: Bool
    public let qkvBias: Bool
    public let attentionDropout: Double
    public let slidingWindow: Int
    public let layerTypes: [LagunaLayerType]
    public let mlpLayerTypes: [LagunaMLPType]
    public let gating: LagunaGatingMode
    public let tieWordEmbeddings: Bool
    public let numExperts: Int
    public let numExpertsPerTok: Int
    public let moeIntermediateSize: Int
    public let sharedExpertIntermediateSize: Int
    public let moeRoutedScalingFactor: Double
    public let normTopkProb: Bool
    public let moeRouterLogitSoftcapping: Double
    public let slidingRope: LagunaRopeSpec
    public let fullRope: LagunaRopeSpec
    public let quantization: LagunaQuantizationSpec

    public init(
        modelType: String,
        vocabSize: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        numAttentionHeadsPerLayer: [Int],
        numKeyValueHeads: Int,
        headDim: Int,
        rmsNormEps: Double,
        maxPositionEmbeddings: Int,
        attentionBias: Bool,
        qkvBias: Bool,
        attentionDropout: Double,
        slidingWindow: Int,
        layerTypes: [LagunaLayerType],
        mlpLayerTypes: [LagunaMLPType],
        gating: LagunaGatingMode,
        tieWordEmbeddings: Bool,
        numExperts: Int,
        numExpertsPerTok: Int,
        moeIntermediateSize: Int,
        sharedExpertIntermediateSize: Int,
        moeRoutedScalingFactor: Double,
        normTopkProb: Bool,
        moeRouterLogitSoftcapping: Double,
        slidingRope: LagunaRopeSpec,
        fullRope: LagunaRopeSpec,
        quantization: LagunaQuantizationSpec
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numAttentionHeadsPerLayer = numAttentionHeadsPerLayer
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.rmsNormEps = rmsNormEps
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.attentionBias = attentionBias
        self.qkvBias = qkvBias
        self.attentionDropout = attentionDropout
        self.slidingWindow = slidingWindow
        self.layerTypes = layerTypes
        self.mlpLayerTypes = mlpLayerTypes
        self.gating = gating
        self.tieWordEmbeddings = tieWordEmbeddings
        self.numExperts = numExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.moeIntermediateSize = moeIntermediateSize
        self.sharedExpertIntermediateSize = sharedExpertIntermediateSize
        self.moeRoutedScalingFactor = moeRoutedScalingFactor
        self.normTopkProb = normTopkProb
        self.moeRouterLogitSoftcapping = moeRouterLogitSoftcapping
        self.slidingRope = slidingRope
        self.fullRope = fullRope
        self.quantization = quantization
    }

    public func layerType(forLayer layerIndex: Int) -> LagunaLayerType {
        layerTypes[layerIndex]
    }

    public func heads(forLayer layerIndex: Int) -> Int {
        numAttentionHeadsPerLayer[layerIndex]
    }

    public func isSparse(layer layerIndex: Int) -> Bool {
        mlpLayerTypes[layerIndex] == .sparse
    }

    public func rope(for layerType: LagunaLayerType) -> LagunaRopeSpec {
        layerType == .full ? fullRope : slidingRope
    }

    /// Output feature count of a layer's attention gate projection
    /// (`g_proj`): one gate per query head for per-head gating, one gate per
    /// output element for per-element gating. Returns nil when gating is
    /// disabled (no `g_proj` tensors exist).
    public func gateProjectionOutputDim(forLayer layerIndex: Int) -> Int? {
        guard gating.enabled else { return nil }
        let layerHeads = heads(forLayer: layerIndex)
        return gating.isPerHead ? layerHeads : layerHeads * headDim
    }

    public static func load(from weightsPath: String) throws -> LagunaConfig {
        let path = URL(fileURLWithPath: weightsPath).appendingPathComponent("config.json")
        try requireFile(path.path, description: "transformed weights config")

        let data = try Data(contentsOf: path)
        let object = try JSONSerialization.jsonObject(with: data)
        guard let root = object as? [String: Any] else {
            throw MLXFastError.invalidInput("config.json must be a JSON object")
        }

        let numHiddenLayers = try intField(
            "num_hidden_layers", root: root, defaultValue: LagunaConstants.numHiddenLayers)
        guard numHiddenLayers == LagunaConstants.numHiddenLayers else {
            throw MLXFastError.invalidInput(
                "Laguna config invariant check failed: num_hidden_layers=\(numHiddenLayers) expected \(LagunaConstants.numHiddenLayers)"
            )
        }
        let numAttentionHeads = try intField(
            "num_attention_heads", root: root, defaultValue: LagunaConstants.fullAttentionHeads)
        let ropeParameters = root["rope_parameters"] as? [String: Any]
        let slidingRopeObject = ropeParameters?["sliding_attention"] as? [String: Any] ?? [:]
        let fullRopeObject = ropeParameters?["full_attention"] as? [String: Any] ?? [:]
        let quantizationObject =
            (root["quantization"] as? [String: Any])
            ?? (root["quantization_config"] as? [String: Any]) ?? [:]

        let layerTypes = try lagunaLayerTypesField(
            "layer_types", root: root, layerCount: numHiddenLayers)
        let config = LagunaConfig(
            modelType: try stringField(
                "model_type", root: root, defaultValue: LagunaConstants.modelType),
            vocabSize: try intField(
                "vocab_size", root: root, defaultValue: LagunaConstants.vocabSize),
            hiddenSize: try intField(
                "hidden_size", root: root, defaultValue: LagunaConstants.hiddenSize),
            intermediateSize: try intField(
                "intermediate_size", root: root,
                defaultValue: LagunaConstants.denseIntermediateSize),
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: numAttentionHeads,
            numAttentionHeadsPerLayer: try intArrayField(
                "num_attention_heads_per_layer", root: root,
                defaultValue: Array(repeating: numAttentionHeads, count: numHiddenLayers)),
            numKeyValueHeads: try intField(
                "num_key_value_heads", root: root,
                defaultValue: LagunaConstants.numKeyValueHeads),
            headDim: try intField("head_dim", root: root, defaultValue: LagunaConstants.headDim),
            rmsNormEps: try doubleField("rms_norm_eps", root: root, defaultValue: 1e-6),
            maxPositionEmbeddings: try intField(
                "max_position_embeddings", root: root, defaultValue: 262_144),
            attentionBias: try boolField("attention_bias", root: root, defaultValue: false),
            qkvBias: try boolField("qkv_bias", root: root, defaultValue: false),
            attentionDropout: try doubleField(
                "attention_dropout", root: root, defaultValue: 0.0),
            slidingWindow: try intField(
                "sliding_window", root: root, defaultValue: LagunaConstants.slidingWindow),
            layerTypes: layerTypes,
            mlpLayerTypes: try lagunaMLPLayerTypesField(root: root, layerCount: numHiddenLayers),
            gating: try lagunaGatingField("gating", root: root, defaultValue: .perHead),
            tieWordEmbeddings: try boolField(
                "tie_word_embeddings", root: root, defaultValue: false),
            numExperts: try intField(
                "num_experts", root: root, defaultValue: LagunaConstants.numExperts),
            numExpertsPerTok: try intField(
                "num_experts_per_tok", root: root,
                defaultValue: LagunaConstants.numExpertsPerTok),
            moeIntermediateSize: try intField(
                "moe_intermediate_size", root: root,
                defaultValue: LagunaConstants.moeIntermediateSize),
            sharedExpertIntermediateSize: try intField(
                "shared_expert_intermediate_size", root: root,
                defaultValue: LagunaConstants.sharedExpertIntermediateSize),
            moeRoutedScalingFactor: try doubleField(
                "moe_routed_scaling_factor", root: root,
                defaultValue: LagunaConstants.moeRoutedScalingFactor),
            normTopkProb: try boolField("norm_topk_prob", root: root, defaultValue: true),
            moeRouterLogitSoftcapping: try doubleField(
                "moe_router_logit_softcapping", root: root, defaultValue: 0.0),
            slidingRope: LagunaRopeSpec(
                theta: try doubleField(
                    "rope_theta", root: slidingRopeObject, defaultValue: 10_000.0),
                type: try stringField(
                    "rope_type", root: slidingRopeObject, defaultValue: "default"),
                partialRotaryFactor: try doubleField(
                    "partial_rotary_factor", root: slidingRopeObject, defaultValue: 1.0)
            ),
            fullRope: LagunaRopeSpec(
                theta: try doubleField(
                    "rope_theta", root: fullRopeObject, defaultValue: 500_000.0),
                type: try stringField("rope_type", root: fullRopeObject, defaultValue: "yarn"),
                partialRotaryFactor: try doubleField(
                    "partial_rotary_factor", root: fullRopeObject, defaultValue: 0.5),
                factor: try doubleField("factor", root: fullRopeObject, defaultValue: 32.0),
                originalMaxPositionEmbeddings: try intField(
                    "original_max_position_embeddings", root: fullRopeObject,
                    defaultValue: 8_192),
                betaFast: try doubleField("beta_fast", root: fullRopeObject, defaultValue: 64.0),
                betaSlow: try doubleField("beta_slow", root: fullRopeObject, defaultValue: 1.0)
            ),
            quantization: try lagunaQuantizationField(quantizationObject)
        )
        try config.validateFrozenInvariants()
        try config.validateStructuralValues()
        return config
    }

    public func validateFrozenInvariants() throws {
        let expected: [(String, Int, Int)] = [
            ("vocab_size", vocabSize, LagunaConstants.vocabSize),
            ("hidden_size", hiddenSize, LagunaConstants.hiddenSize),
            ("intermediate_size", intermediateSize, LagunaConstants.denseIntermediateSize),
            ("num_hidden_layers", numHiddenLayers, LagunaConstants.numHiddenLayers),
            ("num_key_value_heads", numKeyValueHeads, LagunaConstants.numKeyValueHeads),
            ("head_dim", headDim, LagunaConstants.headDim),
            ("sliding_window", slidingWindow, LagunaConstants.slidingWindow),
            ("num_experts", numExperts, LagunaConstants.numExperts),
            ("num_experts_per_tok", numExpertsPerTok, LagunaConstants.numExpertsPerTok),
            ("moe_intermediate_size", moeIntermediateSize, LagunaConstants.moeIntermediateSize),
            (
                "shared_expert_intermediate_size", sharedExpertIntermediateSize,
                LagunaConstants.sharedExpertIntermediateSize
            ),
        ]
        var errors = expected.compactMap { name, actual, expected in
            actual == expected ? nil : "\(name)=\(actual) expected \(expected)"
        }
        if layerTypes.count != numHiddenLayers {
            errors.append("layer_types count=\(layerTypes.count) expected \(numHiddenLayers)")
        }
        if mlpLayerTypes.count != numHiddenLayers {
            errors.append(
                "mlp_layer_types count=\(mlpLayerTypes.count) expected \(numHiddenLayers)")
        }
        if numAttentionHeadsPerLayer.count != numHiddenLayers {
            errors.append(
                "num_attention_heads_per_layer count=\(numAttentionHeadsPerLayer.count) expected \(numHiddenLayers)"
            )
        }
        if tieWordEmbeddings {
            errors.append("tie_word_embeddings=true expected false (Laguna's lm_head is untied)")
        }
        if gating != .perHead {
            errors.append("gating expected per-head for the pinned Laguna checkpoint")
        }
        if !errors.isEmpty {
            throw MLXFastError.invalidInput(
                "Laguna config invariant check failed: \(errors.joined(separator: ", "))"
            )
        }
    }

    public func validateStructuralValues() throws {
        guard numKeyValueHeads > 0 else {
            throw MLXFastError.invalidInput("Laguna KV head count must be positive")
        }
        guard headDim > 0, headDim.isMultiple(of: 2) else {
            throw MLXFastError.invalidInput("Laguna head dimension must be positive and even")
        }
        for (layerIndex, layerHeads) in numAttentionHeadsPerLayer.enumerated() {
            guard layerHeads > 0, layerHeads.isMultiple(of: numKeyValueHeads) else {
                throw MLXFastError.invalidInput(
                    "Laguna layer \(layerIndex) attention heads (\(layerHeads)) must be positive and divisible by num_key_value_heads (\(numKeyValueHeads))"
                )
            }
            let projection = layerHeads.multipliedReportingOverflow(by: headDim)
            guard !projection.overflow, projection.partialValue > 0 else {
                throw MLXFastError.invalidInput(
                    "Laguna layer \(layerIndex) attention projection dimensions overflow Int"
                )
            }
        }
        guard maxPositionEmbeddings > 0,
              slidingWindow > 0,
              slidingWindow <= maxPositionEmbeddings
        else {
            throw MLXFastError.invalidInput(
                "Laguna sliding window must be positive and no larger than max_position_embeddings"
            )
        }
        guard rmsNormEps.isFinite, rmsNormEps > 0 else {
            throw MLXFastError.invalidInput("Laguna rms_norm_eps must be finite and positive")
        }
        guard attentionDropout.isFinite, attentionDropout >= 0, attentionDropout < 1 else {
            throw MLXFastError.invalidInput("Laguna attention_dropout must be finite and in 0..<1")
        }
        guard numExperts > 0,
              numExpertsPerTok > 0,
              numExpertsPerTok <= numExperts
        else {
            throw MLXFastError.invalidInput(
                "Laguna expert counts must be positive with num_experts_per_tok <= num_experts"
            )
        }
        guard intermediateSize > 0, moeIntermediateSize > 0, sharedExpertIntermediateSize > 0
        else {
            throw MLXFastError.invalidInput("Laguna MLP intermediate sizes must be positive")
        }
        guard moeRoutedScalingFactor.isFinite, moeRoutedScalingFactor > 0 else {
            throw MLXFastError.invalidInput(
                "Laguna moe_routed_scaling_factor must be finite and positive")
        }
        guard moeRouterLogitSoftcapping.isFinite, moeRouterLogitSoftcapping >= 0 else {
            throw MLXFastError.invalidInput(
                "Laguna moe_router_logit_softcapping must be finite and non-negative")
        }
        try validateRopeSpec(slidingRope, dims: headDim, label: "sliding_attention")
        try validateRopeSpec(fullRope, dims: headDim, label: "full_attention")
        try validateQuantization()
    }

    private func validateRopeSpec(_ spec: LagunaRopeSpec, dims: Int, label: String) throws {
        guard spec.theta.isFinite, spec.theta > 0 else {
            throw MLXFastError.invalidInput("Laguna \(label) rope_theta must be finite and positive")
        }
        guard spec.partialRotaryFactor.isFinite,
              spec.partialRotaryFactor > 0,
              spec.partialRotaryFactor <= 1,
              spec.partialRotaryFactor * Double(dims) >= 2
        else {
            throw MLXFastError.invalidInput(
                "Laguna \(label) partial_rotary_factor must rotate at least one pair and no more than the head"
            )
        }
        let rotatedDims = Int(Float(dims) * Float(spec.partialRotaryFactor))
        guard rotatedDims.isMultiple(of: 2) else {
            throw MLXFastError.invalidInput(
                "Laguna \(label) partial_rotary_factor must rotate an even number of dimensions"
            )
        }
        // The runtime only knows how to configure the two RoPE families the
        // pinned checkpoint uses; anything else would silently pick up
        // non-Laguna defaults inside the shared `initializeRope` factory.
        switch spec.type {
        case "default":
            break
        case "yarn":
            guard spec.factor.isFinite, spec.factor >= 1 else {
                throw MLXFastError.invalidInput(
                    "Laguna \(label) yarn factor must be finite and at least 1")
            }
            guard spec.originalMaxPositionEmbeddings > 0 else {
                throw MLXFastError.invalidInput(
                    "Laguna \(label) yarn original_max_position_embeddings must be positive")
            }
            guard spec.betaFast.isFinite, spec.betaSlow.isFinite,
                  spec.betaFast >= spec.betaSlow, spec.betaSlow > 0
            else {
                throw MLXFastError.invalidInput(
                    "Laguna \(label) yarn beta_fast/beta_slow must be finite with beta_fast >= beta_slow > 0"
                )
            }
        default:
            throw MLXFastError.invalidInput(
                "Laguna \(label) rope_type \(spec.type) is not supported (expected default or yarn)"
            )
        }
    }

    private func validateQuantization() throws {
        guard quantization.mode == "affine" else {
            throw MLXFastError.invalidInput(
                "Laguna quantization mode must be affine, found \(quantization.mode)"
            )
        }
        guard quantization.groupSize > 0,
              hiddenSize.isMultiple(of: quantization.groupSize),
              intermediateSize.isMultiple(of: quantization.groupSize),
              moeIntermediateSize.isMultiple(of: quantization.groupSize),
              sharedExpertIntermediateSize.isMultiple(of: quantization.groupSize)
        else {
            throw MLXFastError.invalidInput(
                "Laguna quantization group_size must divide the hidden and intermediate dimensions"
            )
        }
        for layerHeads in numAttentionHeadsPerLayer {
            guard (layerHeads * headDim).isMultiple(of: quantization.groupSize) else {
                throw MLXFastError.invalidInput(
                    "Laguna quantization group_size must divide every attention projection width"
                )
            }
        }
        guard [2, 4, 8].contains(quantization.bits) else {
            throw MLXFastError.invalidInput("Laguna quantization bits must be 2, 4, or 8")
        }
        for (stem, override) in quantization.overrides {
            guard override.groupSize > 0, [2, 4, 8].contains(override.bits) else {
                throw MLXFastError.invalidInput(
                    "Laguna quantization override for \(stem) must use a positive group size and bits in {2, 4, 8}"
                )
            }
            // The runtime weight loader promotes quantized modules using the
            // GLOBAL group size for every tensor (it derives each module's
            // logical input width, and from it the bit width, as
            // `scales.dim(-1) * groupSize`). An override that changes only
            // the bit width (the pinned checkpoint's 8-bit router gates) is
            // representable; an override with a different group size is not,
            // and would otherwise surface as a shape-mismatch crash after
            // the multi-GB weight load instead of a clear config error here.
            guard override.groupSize == quantization.groupSize else {
                throw MLXFastError.invalidInput(
                    "Laguna quantization override for \(stem) must keep the global group size \(quantization.groupSize), found \(override.groupSize)"
                )
            }
        }
    }
}

private func fieldValue(_ key: String, root: [String: Any]) -> Any? {
    guard let value = root[key], !(value is NSNull) else {
        return nil
    }
    return value
}

private func intField(_ key: String, root: [String: Any], defaultValue: Int) throws -> Int {
    guard let value = fieldValue(key, root: root) else {
        return defaultValue
    }
    return try parseInt(value, field: key)
}

private func doubleField(_ key: String, root: [String: Any], defaultValue: Double) throws -> Double {
    guard let value = fieldValue(key, root: root) else {
        return defaultValue
    }
    return try parseDouble(value, field: key)
}

private func boolField(_ key: String, root: [String: Any], defaultValue: Bool) throws -> Bool {
    guard let value = fieldValue(key, root: root) else {
        return defaultValue
    }
    guard let number = value as? NSNumber,
          CFGetTypeID(number) == CFBooleanGetTypeID()
    else {
        throw MLXFastError.invalidInput("config field \(key) must be a boolean")
    }
    return number.boolValue
}

private func stringField(_ key: String, root: [String: Any], defaultValue: String) throws -> String {
    guard let value = fieldValue(key, root: root) else {
        return defaultValue
    }
    guard let string = value as? String else {
        throw MLXFastError.invalidInput("config field \(key) must be a string")
    }
    return string
}

private func intArrayField(_ key: String, root: [String: Any], defaultValue: [Int]) throws -> [Int] {
    guard let value = fieldValue(key, root: root) else {
        return defaultValue
    }
    guard let rawValues = value as? [Any] else {
        throw MLXFastError.invalidInput("config field \(key) must be an integer array")
    }
    return try rawValues.map { try parseInt($0, field: key) }
}

private func lagunaLayerTypesField(
    _ key: String, root: [String: Any], layerCount: Int
) throws -> [LagunaLayerType] {
    guard let value = fieldValue(key, root: root) else {
        // Default Laguna pattern: one full-attention layer, then three
        // sliding-window layers, repeating (full layers at 0, 4, 8, ...).
        return (0..<layerCount).map { $0 % 4 == 0 ? LagunaLayerType.full : LagunaLayerType.sliding }
    }
    guard let rawValues = value as? [String] else {
        throw MLXFastError.invalidInput("config field \(key) must be a string array")
    }
    return try rawValues.map { raw in
        guard let type = LagunaLayerType(rawValue: raw) else {
            throw MLXFastError.invalidInput("config field \(key) contains unsupported layer type \(raw)")
        }
        return type
    }
}

private func lagunaMLPLayerTypesField(
    root: [String: Any], layerCount: Int
) throws -> [LagunaMLPType] {
    if let value = fieldValue("mlp_layer_types", root: root) {
        guard let rawValues = value as? [String] else {
            throw MLXFastError.invalidInput("config field mlp_layer_types must be a string array")
        }
        return try rawValues.map { raw in
            guard let type = LagunaMLPType(rawValue: raw) else {
                throw MLXFastError.invalidInput(
                    "config field mlp_layer_types contains unsupported MLP type \(raw)")
            }
            return type
        }
    }
    // Fallback mirrors the vendored `LagunaConfiguration.isSparse(layer:)`
    // derivation: `mlp_only_layers` entries are dense, every
    // `decoder_sparse_step`-th remaining layer is sparse.
    let mlpOnlyLayers = try intArrayField("mlp_only_layers", root: root, defaultValue: [0])
    let decoderSparseStep = try intField("decoder_sparse_step", root: root, defaultValue: 1)
    let numExperts = try intField(
        "num_experts", root: root, defaultValue: LagunaConstants.numExperts)
    return (0..<layerCount).map { layerIndex -> LagunaMLPType in
        if mlpOnlyLayers.contains(layerIndex) {
            return .dense
        }
        let isSparse = numExperts > 0 && (layerIndex + 1) % max(decoderSparseStep, 1) == 0
        return isSparse ? .sparse : .dense
    }
}

private func lagunaGatingField(
    _ key: String, root: [String: Any], defaultValue: LagunaGatingMode
) throws -> LagunaGatingMode {
    guard let value = fieldValue(key, root: root) else {
        return defaultValue
    }
    if let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() {
        return number.boolValue ? .perHead : .disabled
    }
    guard let string = value as? String else {
        throw MLXFastError.invalidInput("config field \(key) must be a boolean or string")
    }
    switch string {
    case "per-element", "per_element":
        return .perElement
    case "false", "none", "":
        return .disabled
    case "per-head", "per_head":
        return .perHead
    default:
        // Mirror the vendored `LagunaGating` decoder: any other non-empty
        // string enables the default per-head gating.
        return .perHead
    }
}

private func lagunaQuantizationField(_ root: [String: Any]) throws -> LagunaQuantizationSpec {
    let groupSize = try intField(
        "group_size", root: root, defaultValue: LagunaConstants.quantizationGroupSize)
    let bits = try intField("bits", root: root, defaultValue: LagunaConstants.quantizationBits)
    let mode = try stringField("mode", root: root, defaultValue: "affine")
    var overrides: [String: LagunaQuantizationOverride] = [:]
    for (key, value) in root {
        guard let overrideObject = value as? [String: Any] else {
            continue
        }
        overrides[key] = LagunaQuantizationOverride(
            groupSize: try intField("group_size", root: overrideObject, defaultValue: groupSize),
            bits: try intField("bits", root: overrideObject, defaultValue: bits)
        )
    }
    return LagunaQuantizationSpec(
        groupSize: groupSize, bits: bits, mode: mode, overrides: overrides)
}

private func parseInt(_ value: Any, field: String) throws -> Int {
    guard let number = value as? NSNumber,
          CFGetTypeID(number) != CFBooleanGetTypeID(),
          !CFNumberIsFloatType(number),
          let integer = Int(number.stringValue)
    else {
        throw MLXFastError.invalidInput("config field \(field) must be a finite integer in Int range")
    }
    return integer
}

private func parseDouble(_ value: Any, field: String) throws -> Double {
    guard let number = value as? NSNumber,
          CFGetTypeID(number) != CFBooleanGetTypeID()
    else {
        throw MLXFastError.invalidInput("config field \(field) must be a finite number")
    }
    let double = number.doubleValue
    guard double.isFinite else {
        throw MLXFastError.invalidInput("config field \(field) must be a finite number")
    }
    return double
}
