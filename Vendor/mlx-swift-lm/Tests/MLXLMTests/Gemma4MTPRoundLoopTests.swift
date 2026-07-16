// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import Testing

@Suite("Gemma4MTPRoundLoop B=1")
struct Gemma4MTPRoundLoopTests {

    /// Minimal Gemma 4 target: 10 layers, 5 kv-shared, hidden=64, vocab=64.
    /// Matches the shapes the drafter expects in its forward.
    private func smallTarget() throws -> Gemma4TextModel {
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
            "sliding_window": 64,
            "sliding_window_pattern": 5,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "vocab_size": 64,
            "vocab_size_per_layer_input": 64,
            "rms_norm_eps": 1e-6,
            "hidden_size_per_layer_input": 0
        }
        """
        let data = Data(json.utf8)
        let config = try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: data)
        return Gemma4TextModel(config)
    }

    /// Minimal drafter config matching the target's hidden_size and vocab.
    private func smallDrafter() throws -> Gemma4AssistantDraftModel {
        let json = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 64,
            "use_ordered_embeddings": false,
            "num_centroids": 8,
            "centroid_intermediate_top_k": 2,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 32,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 4,
                "sliding_window": 64,
                "final_logit_softcapping": null,
                "tie_word_embeddings": true,
                "vocab_size": 64,
                "vocab_size_per_layer_input": 64,
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "layer_types": ["sliding_attention", "sliding_attention",
                                "sliding_attention", "full_attention"]
            }
        }
        """
        let data = Data(json.utf8)
        let cfg = try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: data)
        return Gemma4AssistantDraftModel(config: cfg)
    }

    /// Run prefill on the target to produce firstBonus/firstHidden/firstSharedKV
    /// for a given prompt. Drives the target's forwardForMTP over the prompt.
    private func prefill(
        target: Gemma4TextModel, promptTokens: [Int32]
    ) -> (firstBonus: Int, firstHidden: MLXArray, firstSharedKV: Gemma4SharedKV,
          cache: [KVCache]) {
        let tokens = MLXArray(promptTokens)[.newAxis, .ellipsis]  // [1, L]
        let cache = target.newCache(parameters: nil)
        let out = target.forwardForMTP(tokens, cache: cache)
        // Greedy bonus from the last position.
        let lastLogits = out.logits[0..., -1, 0...]
        let bonus = Int(lastLogits.asType(.float32).argMax(axis: -1).item(Int32.self))
        // Last-position hidden (the drafter's next-step input).
        let lastHidden = out.lastHidden[0..., -1 ..< out.lastHidden.dim(1), 0...]
        return (bonus, lastHidden, out.capturedSharedKV, cache)
    }

    @Test func mtpEmitsTokensAndTerminates() async throws {
        let target = try smallTarget()
        let drafter = try smallDrafter()
        eval(target, drafter)

        let prompt: [Int32] = [1, 2, 3, 4, 5]
        let prefillResult = prefill(target: target, promptTokens: prompt)

        // Collect generated tokens as ints for easy inspection.
        var produced: [Int] = []
        let stream = try runGemma4MTPRounds(
            target: target,
            drafter: drafter,
            targetCache: prefillResult.cache,
            firstBonus: prefillResult.firstBonus,
            firstHidden: prefillResult.firstHidden,
            firstSharedKV: prefillResult.firstSharedKV,
            maxTokens: 12,
            blockSize: 3
        )
        for try await gen in stream {
            if case .chunk(let s) = gen, let tok = Int(s) { produced.append(tok) }
        }
        // Should emit at least the first bonus + some more tokens, up to
        // the maxTokens cap. Exact count depends on acceptance rate.
        #expect(produced.count >= 1)
        #expect(produced.count <= 12)
    }

    @Test func mtpRespectsMaxTokens() async throws {
        let target = try smallTarget()
        let drafter = try smallDrafter()
        eval(target, drafter)

        let prefillResult = prefill(target: target, promptTokens: [1, 2, 3])

        var count = 0
        let stream = try runGemma4MTPRounds(
            target: target,
            drafter: drafter,
            targetCache: prefillResult.cache,
            firstBonus: prefillResult.firstBonus,
            firstHidden: prefillResult.firstHidden,
            firstSharedKV: prefillResult.firstSharedKV,
            maxTokens: 5,
            blockSize: 3
        )
        for try await gen in stream {
            if case .chunk = gen { count += 1 }
        }
        #expect(count <= 5)
    }
}

@Suite("Gemma4MTPRoundLoop B>1")
struct Gemma4MTPRoundLoopBatchedTests {

    /// Build a target that can handle batched input. Uses `BatchKVCache`
    /// for all non-shared attention layers. Shape is identical to the B=1
    /// tests but we pre-populate the cache via a batched prefill.
    private func smallTarget() throws -> Gemma4TextModel {
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
            "sliding_window": 64,
            "sliding_window_pattern": 5,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "vocab_size": 64,
            "vocab_size_per_layer_input": 64,
            "rms_norm_eps": 1e-6,
            "hidden_size_per_layer_input": 0
        }
        """
        let data = Data(json.utf8)
        let config = try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: data)
        return Gemma4TextModel(config)
    }

    private func smallDrafter() throws -> Gemma4AssistantDraftModel {
        let json = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 64,
            "use_ordered_embeddings": false,
            "num_centroids": 8,
            "centroid_intermediate_top_k": 2,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 32,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 4,
                "sliding_window": 64,
                "final_logit_softcapping": null,
                "tie_word_embeddings": true,
                "vocab_size": 64,
                "vocab_size_per_layer_input": 64,
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "layer_types": ["sliding_attention", "sliding_attention",
                                "sliding_attention", "full_attention"]
            }
        }
        """
        let data = Data(json.utf8)
        let cfg = try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: data)
        return Gemma4AssistantDraftModel(config: cfg)
    }

    /// Batched prefill: B rows with identical length for simplicity. Returns
    /// firstBonus (per row), firstHidden [B, 1, hidden], firstSharedKV.
    private func batchedPrefill(
        target: Gemma4TextModel, promptIds: [[Int32]]
    ) -> (firstBonus: [Int], firstHidden: MLXArray,
          firstSharedKV: Gemma4SharedKV, cache: [KVCache]) {
        let B = promptIds.count
        let L = promptIds[0].count
        precondition(promptIds.allSatisfy { $0.count == L },
                     "this test helper assumes uniform prompt lengths")
        let tokens = MLXArray(promptIds.flatMap { $0 }, [B, L])  // [B, L]

        // Use BatchKVCache for every non-shared layer so the B>1 path works.
        let firstKvShared = target.configuration.numHiddenLayers
            - target.configuration.numKvSharedLayers
        let leftPadding = Array(repeating: 0, count: B)
        var cache: [KVCache] = []
        for i in 0 ..< firstKvShared {
            let lt = target.configuration.layerTypes[i]
            if lt == "full_attention" {
                cache.append(BatchKVCache(leftPadding: leftPadding))
            } else {
                cache.append(BatchRotatingKVCache(
                    maxSize: target.configuration.slidingWindow,
                    leftPadding: leftPadding))
            }
        }

        let out = target.forwardForMTP(tokens, cache: cache)
        // Per-row greedy bonus from the last position.
        let lastLogits = out.logits[0..., -1, 0...]  // [B, vocab]
        let bonusArr = lastLogits.asType(.float32).argMax(axis: -1).asArray(Int32.self)
        let bonus = bonusArr.map { Int($0) }
        let lastHidden = out.lastHidden[0..., -1 ..< out.lastHidden.dim(1), 0...]
        return (bonus, lastHidden, out.capturedSharedKV, cache)
    }

    @Test func batchedMtpEmitsForAllRows() async throws {
        let target = try smallTarget()
        let drafter = try smallDrafter()
        eval(target, drafter)

        let prompts: [[Int32]] = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
        let p = batchedPrefill(target: target, promptIds: prompts)

        var emittedPerRow: [[Int]] = [[], []]
        let stream = try runGemma4MTPRoundsBatched(
            target: target,
            drafter: drafter,
            targetCache: p.cache,
            firstBonus: p.firstBonus,
            firstHidden: p.firstHidden,
            firstSharedKV: p.firstSharedKV,
            maxTokens: 8,
            blockSize: 3,
            eosTokenIds: nil
        )
        for try await step in stream {
            for slot in step.slots {
                if let tok = slot.token { emittedPerRow[slot.row].append(tok) }
            }
        }
        #expect(emittedPerRow[0].count >= 1)
        #expect(emittedPerRow[0].count <= 8)
        #expect(emittedPerRow[1].count >= 1)
        #expect(emittedPerRow[1].count <= 8)
    }

    @Test func batchedMtpRespectsEOS() async throws {
        // Pick an EOS that's definitely in the vocab (anything <= 63).
        // We can't force the model to emit a specific EOS on random weights,
        // but we CAN choose a set large enough that at least one sampled
        // token falls in it. Use the entire vocab as EOS -> every row
        // finishes after its first sample.
        let target = try smallTarget()
        let drafter = try smallDrafter()
        eval(target, drafter)

        let prompts: [[Int32]] = [[1, 2, 3], [4, 5, 6]]
        let p = batchedPrefill(target: target, promptIds: prompts)
        let allVocabEOS: Set<Int> = Set(0 ..< 64)

        var totalTokens = 0
        let stream = try runGemma4MTPRoundsBatched(
            target: target,
            drafter: drafter,
            targetCache: p.cache,
            firstBonus: p.firstBonus,
            firstHidden: p.firstHidden,
            firstSharedKV: p.firstSharedKV,
            maxTokens: 16,
            blockSize: 3,
            eosTokenIds: allVocabEOS
        )
        for try await step in stream {
            for slot in step.slots where slot.token != nil {
                totalTokens += 1
            }
        }
        // With every token in EOS, each row should finish almost immediately
        // (the first bonus is always emitted before EOS can fire, plus any
        // tokens emitted in the first round before the EOS check).
        // Strict bound: each row emits <= blockSize tokens.
        #expect(totalTokens <= 2 * 3)  // B=2, blockSize=3
    }

    /// Continuous-batching compaction: when some rows finish early (EOS),
    /// the remaining rows should keep emitting and the emission sequence
    /// should still use the ORIGINAL row indices. Verifies that cache
    /// filtering + state compaction don't corrupt the per-row output.
    @Test func continuousBatchingCompactsFinishedRows() async throws {
        let target = try smallTarget()
        let drafter = try smallDrafter()
        eval(target, drafter)

        let prompts: [[Int32]] = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        let p = batchedPrefill(target: target, promptIds: prompts)

        // Craft an EOS set that contains the first bonus for rows 0 and 2
        // but not rows 1 and 3 — rows 0/2 finish on their first emission,
        // rows 1/3 must continue for the full maxTokens budget.
        //
        // We don't know the bonuses a priori, but we can find out: firstBonus
        // is returned from batchedPrefill. Pick the EOS set accordingly.
        let eos: Set<Int> = [p.firstBonus[0], p.firstBonus[2]]

        var emittedPerRow: [[Int]] = [[], [], [], []]
        let stream = try runGemma4MTPRoundsBatched(
            target: target, drafter: drafter,
            targetCache: p.cache,
            firstBonus: p.firstBonus, firstHidden: p.firstHidden,
            firstSharedKV: p.firstSharedKV,
            maxTokens: 8, blockSize: 3,
            eosTokenIds: eos
        )
        for try await step in stream {
            for slot in step.slots {
                if let tok = slot.token {
                    emittedPerRow[slot.row].append(tok)
                }
            }
        }
        // Rows 0 and 2: bonus is in EOS so they finish after their very
        // first emission. They should have exactly 1 emitted token.
        #expect(emittedPerRow[0].count == 1,
                "row 0 should finish on bonus (EOS), got \(emittedPerRow[0].count)")
        #expect(emittedPerRow[2].count == 1,
                "row 2 should finish on bonus (EOS), got \(emittedPerRow[2].count)")
        // Rows 1 and 3: bonus is NOT in EOS (unless by coincidence the
        // random-weight bonuses happen to collide — rare at vocab=64 but
        // possible); check that they continue past 1.
        if p.firstBonus[1] != p.firstBonus[0] && p.firstBonus[1] != p.firstBonus[2] {
            #expect(emittedPerRow[1].count > 1,
                    "row 1's bonus is not EOS; should continue past 1")
        }
        if p.firstBonus[3] != p.firstBonus[0] && p.firstBonus[3] != p.firstBonus[2] {
            #expect(emittedPerRow[3].count > 1,
                    "row 3's bonus is not EOS; should continue past 1")
        }
    }
}

@Suite("generateGemma4MTP")
struct GenerateGemma4MTPTests {

    /// Build a minimal Gemma4 target + drafter + TestTokenizer-based context.
    /// Note: the tests use random weights, so output text is gibberish — we
    /// only verify structure (stream starts/ends, string chunks emitted).
    private func makeContext() throws -> (ModelContext, Gemma4AssistantDraftModel) {
        let targetJSON = """
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
            "sliding_window": 64,
            "sliding_window_pattern": 5,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "vocab_size": 64,
            "vocab_size_per_layer_input": 64,
            "rms_norm_eps": 1e-6,
            "hidden_size_per_layer_input": 0
        }
        """
        let targetCfg = try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(targetJSON.utf8))
        let target = Gemma4TextModel(targetCfg)

        let drafterJSON = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 64,
            "use_ordered_embeddings": false,
            "num_centroids": 8,
            "centroid_intermediate_top_k": 2,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 32,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 4,
                "sliding_window": 64,
                "final_logit_softcapping": null,
                "tie_word_embeddings": true,
                "vocab_size": 64,
                "vocab_size_per_layer_input": 64,
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "layer_types": ["sliding_attention", "sliding_attention",
                                "sliding_attention", "full_attention"]
            }
        }
        """
        let drafterCfg = try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: Data(drafterJSON.utf8))
        let drafter = Gemma4AssistantDraftModel(config: drafterCfg)
        eval(target, drafter)

        let processor = TestInputProcessor()
        let modelConfig = ModelConfiguration(
            id: "gemma4-test",
            defaultPrompt: "",
            extraEOSTokens: [],
            toolCallFormat: nil
        )
        let context = ModelContext(
            configuration: modelConfig,
            model: target,
            processor: processor,
            tokenizer: processor.tokenizer
        )
        return (context, drafter)
    }

    @Test func generateEmitsChunksAndFinishesCleanly() async throws {
        let (context, drafter) = try makeContext()
        // Minimal LMInput.
        let tokens = MLXArray([Int32(1), 2, 3, 4])[.newAxis, .ellipsis]
        let input = LMInput(text: .init(tokens: tokens))
        var params = GenerateParameters()
        params.maxTokens = 6

        let stream = try generateGemma4MTP(
            input: input,
            parameters: params,
            target: context,
            drafter: drafter,
            blockSize: 3
        )

        var chunks: [String] = []
        var hasInfo = false
        for try await gen in stream {
            switch gen {
            case .chunk(let s): chunks.append(s)
            case .info: hasInfo = true
            default: break
            }
        }
        #expect(chunks.count >= 1)
        #expect(chunks.count <= 6)
        #expect(hasInfo)
    }

    @Test func generateRejectsNonGemma4Target() async throws {
        // Build a Gemma3TextModel context — it should throw unsupportedTarget.
        let processor = TestInputProcessor()
        let modelConfig = ModelConfiguration(
            id: "gemma3-not-supported", defaultPrompt: "",
            extraEOSTokens: [], toolCallFormat: nil)
        let gemma3Config = Gemma3TextConfiguration(
            modelType: "text",
            hiddenSize: 64, hiddenLayers: 8, intermediateSize: 64,
            attentionHeads: 4, headDim: 64,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4,
            ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000,
            ropeTraditional: false, queryPreAttnScalar: 256,
            slidingWindow: 512, slidingWindowPattern: 6,
            maxPositionEmbeddings: 32768
        )
        let target = Gemma3TextModel(gemma3Config)
        let context = ModelContext(
            configuration: modelConfig, model: target,
            processor: processor, tokenizer: processor.tokenizer)

        let drafterJSON = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 64,
            "use_ordered_embeddings": false,
            "num_centroids": 8,
            "centroid_intermediate_top_k": 2,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 32,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 4,
                "sliding_window": 64,
                "final_logit_softcapping": null,
                "tie_word_embeddings": true,
                "vocab_size": 64,
                "vocab_size_per_layer_input": 64,
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "layer_types": ["sliding_attention", "sliding_attention",
                                "sliding_attention", "full_attention"]
            }
        }
        """
        let drafterCfg = try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: Data(drafterJSON.utf8))
        let drafter = Gemma4AssistantDraftModel(config: drafterCfg)
        eval(target, drafter)

        let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
        let input = LMInput(text: .init(tokens: tokens))
        var params = GenerateParameters()
        params.maxTokens = 4

        #expect(throws: Gemma4MTPError.self) {
            _ = try generateGemma4MTP(
                input: input, parameters: params, target: context,
                drafter: drafter, blockSize: 3
            )
        }
    }

    @Test func generateRejectsInvalidBlockSize() async throws {
        let (context, drafter) = try makeContext()
        let tokens = MLXArray([Int32(1), 2, 3])[.newAxis, .ellipsis]
        let input = LMInput(text: .init(tokens: tokens))
        var params = GenerateParameters()
        params.maxTokens = 4

        #expect(throws: Gemma4MTPError.self) {
            _ = try generateGemma4MTP(
                input: input, parameters: params, target: context,
                drafter: drafter, blockSize: 1
            )
        }
    }
}
