// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import MLXLLM
import Testing

@Suite("Gemma4AssistantConfiguration decoding")
struct Gemma4AssistantConfigurationTests {

    private func loadFixture(named name: String) throws -> Data {
        let url = Bundle.module.url(forResource: name, withExtension: "json")
        #expect(url != nil, "fixture \(name).json not found in test bundle")
        return try Data(contentsOf: url!)
    }

    @Test func decodesE4BDrafterConfig() throws {
        let data = try loadFixture(named: "gemma4-E4B-assistant-config")
        let config = try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: data)
        #expect(config.backboneHiddenSize == 2560)
        #expect(config.useOrderedEmbeddings == true)
        #expect(config.numCentroids == 2048)
        #expect(config.centroidIntermediateTopK == 32)
        #expect(config.textConfig.tieWordEmbeddings == true)
        #expect(config.textConfig.numHiddenLayers == 4)
        #expect(config.textConfig.numKvSharedLayers == 4)
        #expect(config.textConfig.attentionKeqV == false)
    }

    @Test func decodes26BA4BDrafterConfig() throws {
        let data = try loadFixture(named: "gemma4-26B-A4B-assistant-config")
        let config = try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: data)
        #expect(config.backboneHiddenSize == 2816)
        #expect(config.useOrderedEmbeddings == false)
        #expect(config.numCentroids == 2048)
        #expect(config.textConfig.tieWordEmbeddings == true)
        #expect(config.textConfig.numHiddenLayers == 4)
        #expect(config.textConfig.numKvSharedLayers == 4)
        #expect(config.textConfig.attentionKeqV == true)
        #expect(config.textConfig.numGlobalKeyValueHeads == 2)
    }

    @Test func clampsNumKvSharedLayersWhenMissing() throws {
        // Synthetic config: text_config omits num_kv_shared_layers; the
        // post-init should set it to num_hidden_layers = 4.
        let json = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 256,
            "use_ordered_embeddings": true,
            "num_centroids": 16,
            "centroid_intermediate_top_k": 4,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "intermediate_size": 512,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 32,
                "num_key_value_heads": 1,
                "sliding_window": 64,
                "tie_word_embeddings": true,
                "vocab_size": 1024,
                "vocab_size_per_layer_input": 1024,
                "rms_norm_eps": 1e-6
            }
        }
        """
        let data = Data(json.utf8)
        let config = try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: data)
        #expect(config.textConfig.numKvSharedLayers == 4)
    }

    @Test func clampsNumKvSharedLayersWhenZero() throws {
        let json = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 256,
            "use_ordered_embeddings": false,
            "num_centroids": 16,
            "centroid_intermediate_top_k": 4,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "intermediate_size": 512,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 32,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 0,
                "sliding_window": 64,
                "tie_word_embeddings": true,
                "vocab_size": 1024,
                "vocab_size_per_layer_input": 1024,
                "rms_norm_eps": 1e-6
            }
        }
        """
        let data = Data(json.utf8)
        let config = try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: data)
        #expect(config.textConfig.numKvSharedLayers == 4)
    }
}
