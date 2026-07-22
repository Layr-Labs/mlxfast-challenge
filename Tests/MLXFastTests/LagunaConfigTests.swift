import Foundation
import Testing
@testable import MLXFastCore
@testable import MLXFastModel

// Direct unit coverage for `LagunaConfig.load` (the runtime side of the
// transformed-weights contract in docs/laguna-weight-contract.md). The
// runtime worker's pinned-configuration gate is covered separately in
// BenchmarkSupportTests; these tests pin the LagunaConfig parse/validation
// behavior itself, including the quantization-override rules the weight
// loader depends on.

/// The pinned Laguna XS 2.1 runtime config as the transform emits it,
/// including the checkpoint's per-router-gate 8-bit quantization overrides.
private func pinnedLagunaConfigObject() -> [String: Any] {
    var quantization: [String: Any] = [
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
    ]
    for layerIndex in 1..<LagunaConstants.numHiddenLayers {
        quantization["language_model.model.layers.\(layerIndex).mlp.gate.proj"] = [
            "group_size": 64,
            "bits": 8,
        ]
    }
    return [
        "model_type": "laguna",
        "vocab_size": LagunaConstants.vocabSize,
        "hidden_size": LagunaConstants.hiddenSize,
        "intermediate_size": LagunaConstants.denseIntermediateSize,
        "num_hidden_layers": LagunaConstants.numHiddenLayers,
        "num_attention_heads": LagunaConstants.fullAttentionHeads,
        "num_attention_heads_per_layer": (0..<LagunaConstants.numHiddenLayers).map {
            $0 % 4 == 0
                ? LagunaConstants.fullAttentionHeads
                : LagunaConstants.slidingAttentionHeads
        },
        "num_key_value_heads": LagunaConstants.numKeyValueHeads,
        "head_dim": LagunaConstants.headDim,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 262_144,
        "attention_bias": false,
        "qkv_bias": false,
        "attention_dropout": 0.0,
        "sliding_window": LagunaConstants.slidingWindow,
        "layer_types": (0..<LagunaConstants.numHiddenLayers).map {
            $0 % 4 == 0 ? "full_attention" : "sliding_attention"
        },
        "mlp_layer_types": (0..<LagunaConstants.numHiddenLayers).map {
            $0 == 0 ? "dense" : "sparse"
        },
        "mlp_only_layers": [0],
        "decoder_sparse_step": 1,
        "gating": "per-head",
        "tie_word_embeddings": false,
        "num_experts": LagunaConstants.numExperts,
        "num_experts_per_tok": LagunaConstants.numExpertsPerTok,
        "moe_intermediate_size": LagunaConstants.moeIntermediateSize,
        "shared_expert_intermediate_size": LagunaConstants.sharedExpertIntermediateSize,
        "moe_routed_scaling_factor": LagunaConstants.moeRoutedScalingFactor,
        "norm_topk_prob": true,
        "moe_router_logit_softcapping": 0.0,
        "rope_parameters": [
            "sliding_attention": [
                "rope_theta": 10_000.0,
                "rope_type": "default",
                "partial_rotary_factor": 1.0,
            ],
            "full_attention": [
                "rope_theta": 500_000.0,
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 8_192,
                "beta_fast": 64.0,
                "beta_slow": 1.0,
                "partial_rotary_factor": 0.5,
            ],
        ],
        "quantization": quantization,
    ]
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func loadLagunaConfig(_ object: [String: Any]) throws -> LagunaConfig {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let data = try JSONSerialization.data(withJSONObject: object)
    try data.write(to: root.appendingPathComponent("config.json"))
    return try LagunaConfig.load(from: root.path)
}

@Test
func lagunaConfigLoadsPinnedGeometryAndRouterOverrides() throws {
    let config = try loadLagunaConfig(pinnedLagunaConfigObject())

    #expect(config.modelType == LagunaConstants.modelType)
    #expect(config.vocabSize == LagunaConstants.vocabSize)
    #expect(config.hiddenSize == LagunaConstants.hiddenSize)
    #expect(config.numHiddenLayers == LagunaConstants.numHiddenLayers)
    #expect(config.numKeyValueHeads == LagunaConstants.numKeyValueHeads)
    #expect(config.headDim == LagunaConstants.headDim)
    #expect(config.slidingWindow == LagunaConstants.slidingWindow)
    #expect(!config.tieWordEmbeddings)
    #expect(config.gating == .perHead)
    #expect(config.moeRoutedScalingFactor == LagunaConstants.moeRoutedScalingFactor)

    // Per-layer schedule: full attention (48 heads, YaRN partial rotary) at
    // 0, 4, 8, ..., sliding (64 heads, default RoPE) elsewhere; dense MLP
    // only at layer 0.
    for layerIndex in 0..<config.numHiddenLayers {
        let isFull = layerIndex % 4 == 0
        #expect(config.layerType(forLayer: layerIndex) == (isFull ? .full : .sliding))
        #expect(
            config.heads(forLayer: layerIndex)
                == (isFull
                    ? LagunaConstants.fullAttentionHeads
                    : LagunaConstants.slidingAttentionHeads)
        )
        #expect(config.isSparse(layer: layerIndex) == (layerIndex != 0))
    }
    #expect(config.fullRope.type == "yarn")
    #expect(config.fullRope.partialRotaryFactor == 0.5)
    #expect(config.slidingRope.type == "default")
    #expect(config.slidingRope.partialRotaryFactor == 1.0)

    // Quantization: global affine 4-bit group-64 with the router gates
    // resolved to 8-bit through the per-tensor overrides.
    #expect(config.quantization.groupSize == LagunaConstants.quantizationGroupSize)
    #expect(config.quantization.bits == LagunaConstants.quantizationBits)
    #expect(config.quantization.overrides.count == LagunaConstants.numHiddenLayers - 1)
    let router = config.quantization.expected(
        forTensorStem: "language_model.model.layers.1.mlp.gate.proj"
    )
    #expect(router.groupSize == LagunaConstants.quantizationGroupSize)
    #expect(router.bits == LagunaConstants.routerGateQuantizationBits)
    let ordinary = config.quantization.expected(
        forTensorStem: "language_model.model.layers.1.self_attn.q_proj"
    )
    #expect(ordinary.groupSize == LagunaConstants.quantizationGroupSize)
    #expect(ordinary.bits == LagunaConstants.quantizationBits)
}

@Test
func lagunaConfigRejectsChangedArchitecture() throws {
    var object = pinnedLagunaConfigObject()
    object["num_hidden_layers"] = 12
    #expect(throws: MLXFastError.self) {
        _ = try loadLagunaConfig(object)
    }

    object = pinnedLagunaConfigObject()
    object["tie_word_embeddings"] = true
    #expect(throws: MLXFastError.self) {
        _ = try loadLagunaConfig(object)
    }

    object = pinnedLagunaConfigObject()
    object["num_experts"] = 128
    #expect(throws: MLXFastError.self) {
        _ = try loadLagunaConfig(object)
    }
}

// The runtime weight loader promotes every quantized module using the
// global group size (bits are derived per tensor from the packed width, so
// bit-width overrides like the 8-bit router gates are representable; group
// size overrides are not). A config declaring a different per-tensor group
// size must fail config validation up front instead of surfacing as a
// module shape mismatch after the multi-GB weight load.
@Test
func lagunaConfigRejectsQuantizationOverrideGroupSizeMismatch() throws {
    var object = pinnedLagunaConfigObject()
    var quantization = try #require(object["quantization"] as? [String: Any])
    quantization["language_model.model.layers.1.mlp.gate.proj"] = [
        "group_size": 32,
        "bits": 8,
    ]
    object["quantization"] = quantization
    #expect(throws: MLXFastError.self) {
        _ = try loadLagunaConfig(object)
    }
}

@Test
func lagunaConfigRejectsUnsupportedRopeType() throws {
    var object = pinnedLagunaConfigObject()
    var ropeParameters = try #require(object["rope_parameters"] as? [String: Any])
    ropeParameters["full_attention"] = [
        "rope_theta": 500_000.0,
        "rope_type": "llama3",
        "partial_rotary_factor": 0.5,
    ]
    object["rope_parameters"] = ropeParameters
    #expect(throws: MLXFastError.self) {
        _ = try loadLagunaConfig(object)
    }
}
