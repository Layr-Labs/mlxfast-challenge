import Foundation
import Testing
@testable import MLXFastCore
@testable import MLXFastModel

@Test
func gemma4ConfigLoadsFrozenInvariants() throws {
    let root = try temporaryDirectory()
    try fullConfigJSON().write(
        to: root.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let config = try Gemma4Config.load(from: root.path)

    #expect(config.numHiddenLayers == MLXFastConstants.numHiddenLayers)
    #expect(config.vocabSize == MLXFastConstants.vocabSize)
    #expect(config.hiddenSize == MLXFastConstants.hiddenSize)
    #expect(config.intermediateSize == MLXFastConstants.intermediateSize)
    #expect(config.numAttentionHeads == MLXFastConstants.attentionHeads)
    #expect(config.numKeyValueHeads == 16)
    #expect(config.numGlobalKeyValueHeads == 4)
    #expect(config.headDim == 256)
    #expect(config.globalHeadDim == 512)
    #expect(config.slidingWindow == 1024)
    #expect(config.attentionKEqV)
    #expect(config.finalLogitSoftcapping == 30.0)
    #expect(config.tieWordEmbeddings)
    #expect(config.quantizationGroupSize == 64)
    #expect(config.quantizationBits == 4)
}

@Test
func gemma4ConfigRejectsChangedArchitecture() throws {
    let root = try temporaryDirectory()
    try fullConfigJSON(layers: 12).write(
        to: root.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try Gemma4Config.load(from: root.path)
    }
}

@Test
func gemma4ConfigDefaultsLayerTypesToSlidingFullPattern() throws {
    let root = try temporaryDirectory()
    try """
    {
      "num_hidden_layers": \(MLXFastConstants.numHiddenLayers),
      "vocab_size": \(MLXFastConstants.vocabSize),
      "hidden_size": \(MLXFastConstants.hiddenSize),
      "intermediate_size": \(MLXFastConstants.intermediateSize),
      "num_attention_heads": \(MLXFastConstants.attentionHeads)
    }
    """.write(
        to: root.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let config = try Gemma4Config.load(from: root.path)
    #expect(config.layerTypes.count == MLXFastConstants.numHiddenLayers)
    for (index, layerType) in config.layerTypes.enumerated() {
        let expected: Gemma4LayerType = index % 6 == 5 ? .full : .sliding
        #expect(layerType == expected)
    }
}

@Test
func gemma4ConfigParsesExplicitLayerTypesAndRopeParameters() throws {
    let root = try temporaryDirectory()
    let layerTypesJSON = "[" + (0..<MLXFastConstants.numHiddenLayers).map {
        $0 % 6 == 5 ? "\"full_attention\"" : "\"sliding_attention\""
    }.joined(separator: ",") + "]"
    try """
    {
      "num_hidden_layers": \(MLXFastConstants.numHiddenLayers),
      "vocab_size": \(MLXFastConstants.vocabSize),
      "hidden_size": \(MLXFastConstants.hiddenSize),
      "intermediate_size": \(MLXFastConstants.intermediateSize),
      "num_attention_heads": \(MLXFastConstants.attentionHeads),
      "layer_types": \(layerTypesJSON),
      "rope_parameters": {
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        "full_attention": {"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25}
      }
    }
    """.write(
        to: root.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let config = try Gemma4Config.load(from: root.path)
    #expect(config.layerTypes[4] == .sliding)
    #expect(config.layerTypes[5] == .full)
    #expect(config.slidingRope.theta == 10_000.0)
    #expect(config.slidingRope.type == "default")
    #expect(config.fullRope.theta == 1_000_000.0)
    #expect(config.fullRope.type == "proportional")
    #expect(config.fullRope.partialRotaryFactor == 0.25)
    #expect(config.headDim(for: .sliding) == 256)
    #expect(config.headDim(for: .full) == 512)
    #expect(config.numKeyValueHeads(for: .sliding) == 16)
    #expect(config.numKeyValueHeads(for: .full) == 4)
    #expect(!config.usesKEqV(for: .sliding))
    #expect(config.usesKEqV(for: .full))
}

@Test
func gemma4ConfigRejectsMismatchedLayerTypeCount() throws {
    let root = try temporaryDirectory()
    try """
    {
      "num_hidden_layers": \(MLXFastConstants.numHiddenLayers),
      "vocab_size": \(MLXFastConstants.vocabSize),
      "hidden_size": \(MLXFastConstants.hiddenSize),
      "intermediate_size": \(MLXFastConstants.intermediateSize),
      "num_attention_heads": \(MLXFastConstants.attentionHeads),
      "layer_types": ["sliding_attention"]
    }
    """.write(
        to: root.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try Gemma4Config.load(from: root.path)
    }
}

@Test
func gemma4ConfigRejectsFractionalAndOutOfRangeIntegers() throws {
    let valid = fullConfigJSON()
    let fractional = valid.replacingOccurrences(
        of: "\"num_key_value_heads\": 16",
        with: "\"num_key_value_heads\": 1.5"
    )
    let outOfRange = valid.replacingOccurrences(
        of: "\"num_key_value_heads\": 16",
        with: "\"num_key_value_heads\": 9223372036854775808"
    )

    #expect(throws: MLXFastError.self) {
        _ = try loadConfigJSON(fractional)
    }
    #expect(throws: MLXFastError.self) {
        _ = try loadConfigJSON(outOfRange)
    }
}

@Test
func gemma4ConfigRejectsNumericBooleanFields() throws {
    let valid = fullConfigJSON()
    for numericBoolean in ["0", "1", "0.0", "1.0"] {
        let invalid = valid.replacingOccurrences(
            of: "\"attention_bias\": false",
            with: "\"attention_bias\": \(numericBoolean)"
        )
        #expect(throws: MLXFastError.self, "numeric boolean \(numericBoolean)") {
            _ = try loadConfigJSON(invalid)
        }
    }
}

@Test
func gemma4ConfigRejectsUnsafeLayerCountsBeforeDefaultLayerAllocation() {
    for layerCount in ["-1", "1000000000"] {
        let json = """
        {
          "num_hidden_layers": \(layerCount),
          "vocab_size": \(MLXFastConstants.vocabSize),
          "hidden_size": \(MLXFastConstants.hiddenSize),
          "intermediate_size": \(MLXFastConstants.intermediateSize),
          "num_attention_heads": \(MLXFastConstants.attentionHeads)
        }
        """
        #expect(throws: MLXFastError.self) {
            _ = try loadConfigJSON(json)
        }
    }
}

@Test
func gemma4ConfigRejectsOverflowingAttentionDimensions() throws {
    let invalid = fullConfigJSON().replacingOccurrences(
        of: "\"num_attention_heads\": \(MLXFastConstants.attentionHeads)",
        with: "\"num_attention_heads\": 9223372036854775807"
    )
    #expect(throws: MLXFastError.self) {
        _ = try loadConfigJSON(invalid)
    }
}

@Test
func gemma4ConfigRejectsNonFiniteNumbers() throws {
    let nonFinite = fullConfigJSON().replacingOccurrences(
        of: "\"rms_norm_eps\": 1e-6",
        with: "\"rms_norm_eps\": 1e400"
    )

    // Foundation rejects a JSON number that overflows Double before the
    // model parser sees it. Either way, a non-finite value cannot load.
    #expect(throws: Error.self) {
        _ = try loadConfigJSON(nonFinite)
    }
}

@Test
func gemma4ConfigRejectsInvalidStructuralValues() throws {
    let replacements: [(String, String)] = [
        ("\"num_key_value_heads\": 16", "\"num_key_value_heads\": 0"),
        ("\"num_global_key_value_heads\": 4", "\"num_global_key_value_heads\": 5"),
        ("\"head_dim\": 256", "\"head_dim\": 255"),
        ("\"global_head_dim\": 512", "\"global_head_dim\": 0"),
        ("\"rms_norm_eps\": 1e-6", "\"rms_norm_eps\": 0"),
        ("\"max_position_embeddings\": 262144", "\"max_position_embeddings\": 1000"),
        ("\"attention_dropout\": 0.0", "\"attention_dropout\": 1.0"),
        ("\"final_logit_softcapping\": 30.0", "\"final_logit_softcapping\": 0"),
        (
            "\"partial_rotary_factor\": 0.25",
            "\"partial_rotary_factor\": 0"
        ),
        (
            "\"quantization\": {\"group_size\": 64, \"bits\": 4, \"mode\": \"affine\"}",
            "\"quantization\": {\"group_size\": 63, \"bits\": 4, \"mode\": \"affine\"}"
        ),
        (
            "\"quantization\": {\"group_size\": 64, \"bits\": 4, \"mode\": \"affine\"}",
            "\"quantization\": {\"group_size\": 64, \"bits\": 3, \"mode\": \"affine\"}"
        ),
    ]

    for (validField, invalidField) in replacements {
        let invalid = fullConfigJSON().replacingOccurrences(of: validField, with: invalidField)
        #expect(invalid != fullConfigJSON())
        #expect(throws: MLXFastError.self) {
            _ = try loadConfigJSON(invalid)
        }
    }
}

private func fullConfigJSON(layers: Int = MLXFastConstants.numHiddenLayers) -> String {
    """
    {
      "model_type": "gemma4_text",
      "num_hidden_layers": \(layers),
      "vocab_size": \(MLXFastConstants.vocabSize),
      "hidden_size": \(MLXFastConstants.hiddenSize),
      "intermediate_size": \(MLXFastConstants.intermediateSize),
      "num_attention_heads": \(MLXFastConstants.attentionHeads),
      "num_key_value_heads": 16,
      "num_global_key_value_heads": 4,
      "head_dim": 256,
      "global_head_dim": 512,
      "rms_norm_eps": 1e-6,
      "hidden_activation": "gelu_pytorch_tanh",
      "max_position_embeddings": 262144,
      "attention_bias": false,
      "attention_dropout": 0.0,
      "attention_k_eq_v": true,
      "sliding_window": 1024,
      "final_logit_softcapping": 30.0,
      "tie_word_embeddings": true,
      "rope_parameters": {
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        "full_attention": {"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25}
      },
      "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
    }
    """
}

private func loadConfigJSON(_ json: String) throws -> Gemma4Config {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    try json.write(
        to: root.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    return try Gemma4Config.load(from: root.path)
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
