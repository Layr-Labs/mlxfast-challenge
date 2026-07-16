// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@Suite("Gemma4TextModel.rollbackSpeculativeCache")
struct Gemma4RollbackTests {

    private func smallConfig() throws -> Gemma4TextConfiguration {
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 64,
            "num_hidden_layers": 10,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "head_dim": 32,
            "global_head_dim": 32,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 5,
            "sliding_window": 16,
            "sliding_window_pattern": 5,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 32,
            "rms_norm_eps": 1e-6,
            "hidden_size_per_layer_input": 0
        }
        """
        let data = Data(json.utf8)
        return try JSONDecoder.json5().decode(Gemma4TextConfiguration.self, from: data)
    }

    @Test func scalarTrimsUniformly() throws {
        let config = try smallConfig()
        let model = Gemma4TextModel(config)
        eval(model)
        let tokens = MLXArray([Int32(1), 2, 3, 4, 5])[.newAxis, .ellipsis]  // [1, 5]
        let cache = model.newCache(parameters: nil)
        _ = model(tokens, cache: cache)
        let offsetBefore = cache[0].offset  // == 5
        model.rollbackSpeculativeCache(cache, accepted: .scalar(1), blockSize: 3)
        // Trimmed = blockSize - accepted - 1 = 3 - 1 - 1 = 1
        #expect(cache[0].offset == offsetBefore - 1)
    }

    @Test func scalarNoopWhenFullyAccepted() throws {
        let config = try smallConfig()
        let model = Gemma4TextModel(config)
        eval(model)
        let tokens = MLXArray([Int32(1), 2, 3, 4, 5])[.newAxis, .ellipsis]
        let cache = model.newCache(parameters: nil)
        _ = model(tokens, cache: cache)
        let offsetBefore = cache[0].offset
        // accepted == blockSize - 1 means all drafts were accepted; no trim.
        model.rollbackSpeculativeCache(cache, accepted: .scalar(2), blockSize: 3)
        #expect(cache[0].offset == offsetBefore)
    }

    // perRow case tested indirectly through the B>1 round-loop tests;
    // the isolated test here verifies the dispatch path calls both
    // trim and zeroTailPerRow without crashing on a batched cache.
    @Test func perRowDispatchesToBatchedPath() throws {
        // This test requires a BatchKVCache, which Gemma4TextModel.newCache
        // doesn't produce. We construct one manually and invoke the method
        // on a caches array that includes it.
        let b = BatchKVCache(leftPadding: [0, 0])
        let k = MLXArray.ones([2, 1, 6, 4], dtype: .float32)
        let v = MLXArray.ones([2, 1, 6, 4], dtype: .float32)
        _ = b.update(keys: k, values: v)

        let config = try smallConfig()
        let model = Gemma4TextModel(config)
        let accepted = MLXArray([Int32(0), 2])  // row 0 accepts 0, row 1 accepts 2
        model.rollbackSpeculativeCache(
            [b], accepted: .perRow(accepted), blockSize: 3
        )
        // Uniform trim = 3 - max(accepted) - 1 = 3 - 2 - 1 = 0 → no trim.
        #expect(b.offset == 6)
        // zeroTailPerRow: row 0 keeps (6 - max + 0) = 4 slots, row 1 keeps
        // (6 - max + 2) = 6 slots. Row 0's last 2 slots should be zero.
        let state = b.state
        let row0Tail = state[0][0, 0, 4..., 0]
        #expect(row0Tail.sum().item(Float.self) == 0.0)
        let row1All = state[0][1, 0, 0..<6, 0]
        #expect(row1All.sum().item(Float.self) == 6.0)  // untouched
    }
}
