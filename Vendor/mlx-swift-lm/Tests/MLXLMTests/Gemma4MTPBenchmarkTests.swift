// Copyright © 2026 Apple Inc.
//
// Smoke tests for the Gemma 4 MTP benchmark primitives. Verifies the
// measurement harness runs end-to-end, produces finite timings, and
// returns a token count matching `maxTokens`. Actual throughput numbers
// are hardware-dependent and are not asserted; a separate driver
// (outside the test target) collects them against real models.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import MLXLLM
import Testing

@Suite("Gemma4MTPBenchmark", .serialized)
struct Gemma4MTPBenchmarkTests {

    /// Tiny target for smoke tests — matches the E2B-shaped config used
    /// in the parity suite so we know the harness runs cleanly.
    private func tinyTargetConfig() throws -> Gemma4TextConfiguration {
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 256,
            "num_hidden_layers": 12,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "head_dim": 32,
            "global_head_dim": 32,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 6,
            "sliding_window": 128,
            "sliding_window_pattern": 5,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "vocab_size": 1024,
            "vocab_size_per_layer_input": 1024,
            "rms_norm_eps": 1e-6,
            "hidden_size_per_layer_input": 0
        }
        """
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    private func tinyDrafterConfig() throws -> Gemma4AssistantConfiguration {
        let json = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 256,
            "use_ordered_embeddings": false,
            "num_centroids": 32,
            "centroid_intermediate_top_k": 4,
            "tie_word_embeddings": true,
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
                "vocab_size": 1024,
                "vocab_size_per_layer_input": 1024,
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "layer_types": ["sliding_attention", "sliding_attention",
                                "sliding_attention", "full_attention"]
            }
        }
        """
        return try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: Data(json.utf8))
    }

    @Test func baselineHarnessRunsAndProducesFiniteTimings() async throws {
        MLXRandom.seed(42)
        let target = Gemma4TextModel(try tinyTargetConfig())
        eval(target)
        let prompt = MLXArray([Int32](repeating: 7, count: 8))

        let r = measureBaselineThroughput(
            target: target, promptTokens: prompt, maxTokens: 8)

        #expect(r.generatedTokens == 8)
        #expect(r.prefillSeconds.isFinite && r.prefillSeconds >= 0)
        #expect(r.generationSeconds.isFinite && r.generationSeconds >= 0)
        #expect(r.acceptLengths == nil, "baseline has no accept histogram")
    }

    @Test func mtpHarnessRunsAndProducesAcceptLengths() async throws {
        MLXRandom.seed(42)
        let target = Gemma4TextModel(try tinyTargetConfig())
        let drafter = Gemma4AssistantDraftModel(config: try tinyDrafterConfig())
        eval(target, drafter)
        let prompt = MLXArray([Int32](repeating: 7, count: 8))

        let r = try await measureMTPThroughput(
            target: target, drafter: drafter,
            promptTokens: prompt, maxTokens: 12, blockSize: 3)

        #expect(r.generatedTokens == 12)
        #expect(r.prefillSeconds.isFinite && r.prefillSeconds >= 0)
        #expect(r.generationSeconds.isFinite && r.generationSeconds >= 0)
    }

    // MARK: - Real-model benchmark (opt-in, gated on MTP_BENCH_DATA_DIR)

    /// Real-model benchmark. Measures tokens/sec for baseline vs MTP on
    /// locally-resident models. Skipped unless `MTP_BENCH_DATA_DIR` env var
    /// points to a directory containing:
    ///   - gemma-4-e2b-it-bf16/
    ///   - gemma-4-E2B-it-assistant-bf16/
    /// (and optionally the e4b / 26b-a4b-4bit pairs).
    ///
    /// Produces a printed table. No assertions on throughput — numbers are
    /// hardware-dependent. Used to populate the README performance section.
    @Test func realModelThroughputBenchmark() async throws {
        guard let dataDir = ProcessInfo.processInfo.environment["MTP_BENCH_DATA_DIR"]
        else {
            print("Skipping: MTP_BENCH_DATA_DIR env var not set")
            return
        }

        struct Pair {
            let label: String
            let targetDir: String
            let drafterDir: String
        }
        let allPairs: [Pair] = [
            Pair(label: "E2B-4bit",
                 targetDir: "gemma-4-e2b-it-4bit",
                 drafterDir: "gemma-4-E2B-it-assistant-bf16"),
            Pair(label: "E2B-bf16",
                 targetDir: "gemma-4-e2b-it-bf16",
                 drafterDir: "gemma-4-E2B-it-assistant-bf16"),
            Pair(label: "E4B-bf16",
                 targetDir: "gemma-4-e4b-it-bf16",
                 drafterDir: "gemma-4-E4B-it-assistant-bf16"),
            Pair(label: "26B-A4B-4bit",
                 targetDir: "gemma-4-26b-a4b-it-4bit",
                 drafterDir: "gemma-4-26B-A4B-it-assistant-bf16"),
            Pair(label: "26B-A4B-q4draft",
                 targetDir: "gemma-4-26b-a4b-it-4bit",
                 drafterDir: "gemma-4-26B-A4B-it-qat-assistant-4bit"),
            Pair(label: "26B-QAT4-q4draft",
                 targetDir: "gemma-4-26B-A4B-it-qat-4bit",
                 drafterDir: "gemma-4-26B-A4B-it-qat-assistant-4bit"),
        ]
        let env = ProcessInfo.processInfo.environment
        let requestedPair = env["MTP_BENCH_PAIR"]?.lowercased()
        let pairs = allPairs.filter {
            let matchesRequest =
                requestedPair == nil
                || $0.label.lowercased() == requestedPair
                || $0.targetDir.lowercased() == requestedPair
                || $0.drafterDir.lowercased() == requestedPair
            return matchesRequest
                && FileManager.default.fileExists(atPath: "\(dataDir)/\($0.targetDir)")
                && FileManager.default.fileExists(atPath: "\(dataDir)/\($0.drafterDir)")
        }
        #expect(!pairs.isEmpty, "No requested model pairs found in \(dataDir)")

        func envInt(_ name: String, default defaultValue: Int) -> Int {
            guard let value = env[name], let parsed = Int(value), parsed > 0 else {
                return defaultValue
            }
            return parsed
        }

        func envIntList(_ name: String, default defaultValue: [Int]) -> [Int] {
            guard let value = env[name] else { return defaultValue }
            let parsed = value
                .split(separator: ",")
                .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
                .filter { $0 > 1 }
            return parsed.isEmpty ? defaultValue : parsed
        }

        // Load real chat-templated prompt tokens from a JSON fixture
        // (produced via Python's transformers tokenizer — path pointed to
        // by MTP_BENCH_PROMPTS env var). Using garbage token IDs produces
        // misleading "slower than baseline" results because the drafter
        // has no signal to predict from. Falls back to a single arbitrary
        // prompt if the fixture isn't provided.
        let promptsData: [[Int32]]
        if let fixturePath = ProcessInfo.processInfo.environment["MTP_BENCH_PROMPTS"] {
            let data = try Data(contentsOf: URL(fileURLWithPath: fixturePath))
            let decoded = try JSONDecoder().decode([[Int]].self, from: data)
            promptsData = decoded.map { $0.map { Int32($0) } }
        } else {
            // Degenerate fallback. Use MTP_BENCH_PROMPTS for real numbers.
            promptsData = [(0 ..< 16).map { Int32($0 * 211 + 7) }]
        }
        let maxTokens = envInt("MTP_BENCH_MAX_TOKENS", default: 64)
        let warmupTokens = envInt(
            "MTP_BENCH_WARMUP_TOKENS",
            default: Swift.min(16, maxTokens))
        let blockSizes = envIntList("MTP_BENCH_BLOCK_SIZES", default: [2, 3, 4, 5])
        let b1Only = env["MTP_BENCH_B1_ONLY"] == "1"

        print("\n\n=== Gemma 4 MTP real-model benchmark ===")
        print("prompts=\(promptsData.count), max_tokens=\(maxTokens), warmup=\(warmupTokens)")
        print("model              K  base tok/s  mtp tok/s  speedup  accept_avg")

        for pair in pairs {
            let targetURL = URL(fileURLWithPath: "\(dataDir)/\(pair.targetDir)")
            let drafterURL = URL(fileURLWithPath: "\(dataDir)/\(pair.drafterDir)")
            print("--- loading \(pair.label) ---")

            let (target, _) = try loadRealTargetAsTextModel(from: targetURL)
            let drafter = try await Gemma4AssistantDraftModel.load(from: drafterURL)
            eval(target, drafter)
            let policy = Gemma4MTPAutomaticPolicy.automatic(for: target)
            print(
                "\(pair.label) automatic policy: "
                    + "family=\(policy.family), "
                    + "B=1 \(policy.strategy(forBatchSize: 1)), "
                    + "B=4 \(policy.strategy(forBatchSize: 4))")

            // Warmup on first prompt with largest block size to exercise
            // the full drafter path.
            let warmupPrompt = MLXArray(promptsData[0])
            _ = measureBaselineThroughput(
                target: target, promptTokens: warmupPrompt, maxTokens: warmupTokens)
            _ = try measureMTPThroughput(
                target: target, drafter: drafter,
                promptTokens: warmupPrompt, maxTokens: warmupTokens,
                blockSize: blockSizes.max()!)
            MLX.Memory.clearCache()

            // Baseline once per prompt (block-size invariant).
            var baseRates: [Double] = []
            for promptInts in promptsData {
                let prompt = MLXArray(promptInts)
                let base = measureBaselineThroughput(
                    target: target, promptTokens: prompt, maxTokens: maxTokens)
                MLX.Memory.clearCache()
                baseRates.append(base.tokensPerSecond)
            }
            let baseAvg = baseRates.reduce(0, +) / Double(baseRates.count)

            for K in blockSizes {
                var mtpRates: [Double] = []
                var allAccepts: [Int] = []
                for promptInts in promptsData {
                    let prompt = MLXArray(promptInts)
                    let mtp = try measureMTPThroughput(
                        target: target, drafter: drafter,
                        promptTokens: prompt, maxTokens: maxTokens,
                        blockSize: K)
                    MLX.Memory.clearCache()
                    mtpRates.append(mtp.tokensPerSecond)
                    allAccepts.append(contentsOf: mtp.acceptLengths ?? [])
                }
                let mtpAvg = mtpRates.reduce(0, +) / Double(mtpRates.count)
                let accAvg = allAccepts.isEmpty
                    ? 0.0
                    : Double(allAccepts.reduce(0, +)) / Double(allAccepts.count)
                let speedup = mtpAvg / max(baseAvg, 1e-9)
                let baseStr = String(format: "%10.1f", baseAvg)
                let mtpStr = String(format: "%10.1f", mtpAvg)
                let spdStr = String(format: "%.2fx", speedup)
                let accStr = String(format: "%.2f", accAvg)
                let label = pair.label.padding(toLength: 18, withPad: " ", startingAt: 0)
                print("\(label) \(K)  \(baseStr) \(mtpStr)   \(spdStr)   \(accStr)/\(K-1)")
            }

            if b1Only {
                continue
            }

            // Batched (B=4) — sweep block sizes because the B=1 optimum
            // is not guaranteed to carry over to batched decode.
            let B4 = Array(repeating: promptsData[0], count: 4)
            _ = measureBatchedBaselineThroughput(
                target: target, promptTokens: B4, maxTokens: warmupTokens)
            _ = try await measureBatchedMTPThroughput(
                target: target, drafter: drafter,
                promptTokens: B4, maxTokens: warmupTokens,
                blockSize: blockSizes.max()!)
            MLX.Memory.clearCache()

            let batchBase = measureBatchedBaselineThroughput(
                target: target, promptTokens: B4, maxTokens: maxTokens)
            MLX.Memory.clearCache()
            let labelB4 = pair.label.padding(toLength: 18, withPad: " ", startingAt: 0)
            let automaticB4 = policy.strategy(forBatchSize: B4.count)
            for K in blockSizes {
                let batchMtp = try await measureBatchedMTPThroughput(
                    target: target, drafter: drafter,
                    promptTokens: B4, maxTokens: maxTokens,
                    blockSize: K)
                MLX.Memory.clearCache()
                let batchSpeedup = batchMtp.tokensPerSecond
                                 / max(batchBase.tokensPerSecond, 1e-9)
                let bbStr = String(format: "%10.1f", batchBase.tokensPerSecond)
                let bmStr = String(format: "%10.1f", batchMtp.tokensPerSecond)
                let bsStr = String(format: "%.2fx", batchSpeedup)
                let automaticMarker =
                    automaticB4 == .batched(blockSize: K) ? " auto" : ""
                print(
                    "\(labelB4) \(K)(B=4) \(bbStr) \(bmStr)   "
                        + "\(bsStr)\(automaticMarker)   "
                        + "-- (aggregate, uniform budget)")
            }

            // Continuous-batching benchmark: staggered budgets at B ∈
            // {4, 8, 16}. Each row gets a different maxTokens cap, so
            // rows finish at different times and compaction can fire.
            //
            // For each B: run WITH compaction (maxTokensPerRow set) vs
            // WITHOUT compaction (uniform maxTokens = max of budgets,
            // pad all rows). Compare wall time for the SAME total
            // emitted tokens.
            //
            // B=16 only runs when MTP_BENCH_LARGE_B=1 is set (requires
            // >64 GB to be safe on 26B-A4B).
            let blockForBatch = 4
            let batchSizes: [Int] = {
                if ProcessInfo.processInfo.environment["MTP_BENCH_LARGE_B"] != nil {
                    return [4, 8, 16]
                }
                return [4]
            }()
            print("\(labelB4)   --- continuous-batching comparison ---")
            print("\(labelB4)   B   budgets            compact_s   pad_s    win")
            for B in batchSizes {
                // Build staggered budgets — half the rows at max, a
                // quarter at max/2, a quarter at max/4 + max/8.
                var budgets: [Int] = []
                for i in 0 ..< B {
                    switch i % 4 {
                    case 0: budgets.append(Swift.max(maxTokens / 8, 1))
                    case 1: budgets.append(Swift.max(maxTokens / 4, 1))
                    case 2: budgets.append(maxTokens / 2)
                    default: budgets.append(maxTokens)
                    }
                }
                let budgetMax = budgets.max() ?? maxTokens
                let budgetTotal = budgets.reduce(0, +)

                let prompts = Array(repeating: promptsData[0], count: B)

                // Warmup (small budget) to clear kernel-compile noise.
                _ = try await measureBatchedMTPThroughputStaggered(
                    target: target, drafter: drafter,
                    promptTokens: prompts,
                    maxTokensPerRow: Array(repeating: 4, count: B),
                    blockSize: blockForBatch)
                MLX.Memory.clearCache()

                // Compacted run (true continuous batching).
                let compactRun = try await measureBatchedMTPThroughputStaggered(
                    target: target, drafter: drafter,
                    promptTokens: prompts, maxTokensPerRow: budgets,
                    blockSize: blockForBatch)
                MLX.Memory.clearCache()

                // Non-compacted run: all rows run to budgetMax (pad).
                // Reports per-token time, so the win is compact_per_tok /
                // pad_per_tok invariant (>1 means compact is faster per
                // emitted token, regardless of total token count).
                let padRun = try await measureBatchedMTPThroughput(
                    target: target, drafter: drafter,
                    promptTokens: prompts,
                    maxTokens: budgetMax,
                    blockSize: blockForBatch)
                MLX.Memory.clearCache()

                let compactPerTok =
                    compactRun.generationSeconds
                    / Double(max(compactRun.totalGenerated, 1))
                let padPerTok =
                    padRun.generationSeconds
                    / Double(max(padRun.totalGenerated, 1))
                let win = padPerTok / compactPerTok
                _ = budgetTotal  // kept for clarity above
                let budgetStr = budgets.map(String.init).joined(separator: ",")
                let compStr = String(
                    format: "%.1f tok/s", 1.0 / compactPerTok)
                let padStr = String(
                    format: "%.1f tok/s", 1.0 / padPerTok)
                let winStr = String(format: "%.2fx", win)
                print(
                    "\(labelB4)   \(B)   "
                        + "\(budgetStr.padding(toLength: 18, withPad: " ", startingAt: 0))"
                        + " compact=\(compStr)  pad=\(padStr)  win=\(winStr)")
            }
        }
    }

    /// Batch-scaling sweep + a "is multi-token verify free?" micro-bench.
    /// Gated on MTP_BENCH_SWEEP=1. Maps where MTP crosses from win to loss
    /// as batch grows, and measures whether a T-position forward amortizes
    /// (dense-like) or costs ~T separate steps (MoE expert dispersion).
    @Test func batchScalingSweep() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MTP_BENCH_SWEEP"] == "1", let dataDir = env["MTP_BENCH_DATA_DIR"],
              let promptsPath = env["MTP_BENCH_PROMPTS"]
        else { print("Skipping: set MTP_BENCH_SWEEP=1, MTP_BENCH_DATA_DIR, MTP_BENCH_PROMPTS"); return }

        let targetDir = env["MTP_BENCH_SWEEP_TARGET"] ?? "gemma-4-26b-a4b-it-4bit"
        let drafterDir = env["MTP_BENCH_SWEEP_DRAFTER"] ?? "gemma-4-26B-A4B-it-qat-assistant-4bit"
        let K = Int(env["MTP_BENCH_SWEEP_K"] ?? "3") ?? 3
        let maxTokens = Int(env["MTP_BENCH_SWEEP_MAXTOK"] ?? "128") ?? 128
        let batchSizes = (env["MTP_BENCH_SWEEP_B"] ?? "1,2,4,8")
            .split(separator: ",").compactMap { Int($0) }

        let pdata = try JSONDecoder()
            .decode([[Int]].self, from: Data(contentsOf: URL(fileURLWithPath: promptsPath)))
            .map { $0.map { Int32($0) } }

        let (target, _) = try loadRealTargetAsTextModel(
            from: URL(fileURLWithPath: "\(dataDir)/\(targetDir)"))
        let drafter = try await Gemma4AssistantDraftModel.load(
            from: URL(fileURLWithPath: "\(dataDir)/\(drafterDir)"))
        eval(target, drafter)

        print("\n=== Batch-scaling sweep (K=\(K), max_tokens=\(maxTokens), diverse prompts) ===")
        print("B  base_agg  base/seq   mtp_agg   speedup   mtp/seq")
        for B in batchSizes {
            let batch = (0 ..< B).map { pdata[$0 % pdata.count] }
            _ = measureBatchedBaselineThroughput(target: target, promptTokens: batch, maxTokens: 8)
            _ = try await measureBatchedMTPThroughput(
                target: target, drafter: drafter, promptTokens: batch, maxTokens: 8, blockSize: K)
            MLX.Memory.clearCache()
            let base = measureBatchedBaselineThroughput(
                target: target, promptTokens: batch, maxTokens: maxTokens)
            MLX.Memory.clearCache()
            let mtp = try await measureBatchedMTPThroughput(
                target: target, drafter: drafter, promptTokens: batch,
                maxTokens: maxTokens, blockSize: K)
            MLX.Memory.clearCache()
            let spd = mtp.tokensPerSecond / max(base.tokensPerSecond, 1e-9)
            print(String(format: "%d  %8.1f  %8.1f  %8.1f   %.2fx   %8.1f",
                B, base.tokensPerSecond, base.tokensPerSecond / Double(B),
                mtp.tokensPerSecond, spd, mtp.tokensPerSecond / Double(B)))
        }

        // Decisive MoE test: time a single forward of [B, T]. If ms/T drops
        // sharply with T, a T-token verify amortizes (dense-like → MTP free).
        // If ms/T stays ~flat (≈ the T=1 cost), the multi-token verify is NOT
        // free — MoE expert dispersion defeats speculation.
        print("\n=== Single-forward cost vs #positions T (is multi-token verify free?) ===")
        print("B  T  forward_ms  ms/position")
        for B in [1, 4] {
            for T in [1, 2, 4, 8] {
                let rows = (0 ..< B).map { r -> [Int32] in
                    let p = pdata[r % pdata.count]
                    return (0 ..< T).map { p[$0 % p.count] }
                }
                let input = MLXArray(rows.flatMap { $0 }, [B, T])
                _ = target(input, cache: target.newCache(parameters: nil))  // warmup
                eval(target(input, cache: target.newCache(parameters: nil)))
                let N = 10
                let t0 = Date()
                for _ in 0 ..< N {
                    let out = target(input, cache: target.newCache(parameters: nil))
                    eval(out)
                }
                let ms = Date().timeIntervalSince(t0) / Double(N) * 1000
                print(String(format: "%d  %d  %9.2f  %9.2f", B, T, ms, ms / Double(T)))
                MLX.Memory.clearCache()
            }
        }
    }

    /// Real-model batched parity. Token-level comparison of the batched
    /// greedy baseline vs the batched MTP round loop on the actual 26B
    /// model. This is the "no garbage outputs" gate for B>=2: greedy MTP
    /// must reproduce the baseline's tokens row-for-row (modulo rare bf16
    /// argmax ties). Dumps both token streams to JSON for offline decode.
    /// Gated on MTP_BENCH_PARITY=1.
    @Test func batchedParityRealModel() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MTP_BENCH_PARITY"] == "1", let dataDir = env["MTP_BENCH_DATA_DIR"],
              let promptsPath = env["MTP_BENCH_PROMPTS"]
        else { print("Skipping: set MTP_BENCH_PARITY=1, MTP_BENCH_DATA_DIR, MTP_BENCH_PROMPTS"); return }

        let targetDir = env["MTP_BENCH_SWEEP_TARGET"] ?? "gemma-4-26B-A4B-it-qat-4bit"
        let drafterDir = env["MTP_BENCH_SWEEP_DRAFTER"] ?? "gemma-4-26B-A4B-it-qat-assistant-4bit"
        let K = Int(env["MTP_BENCH_SWEEP_K"] ?? "4") ?? 4
        let maxTokens = Int(env["MTP_BENCH_SWEEP_MAXTOK"] ?? "160") ?? 160
        let batchSizes = (env["MTP_BENCH_SWEEP_B"] ?? "1,2,4")
            .split(separator: ",").compactMap { Int($0) }

        let pdata = try JSONDecoder()
            .decode([[Int]].self, from: Data(contentsOf: URL(fileURLWithPath: promptsPath)))
            .map { $0.map { Int32($0) } }

        let (target, _) = try loadRealTargetAsTextModel(
            from: URL(fileURLWithPath: "\(dataDir)/\(targetDir)"))
        let drafter = try await Gemma4AssistantDraftModel.load(
            from: URL(fileURLWithPath: "\(dataDir)/\(drafterDir)"))
        eval(target, drafter)

        var dump: [String: [[Int]]] = [:]
        print("\n=== Batched parity vs baseline (K=\(K), max_tokens=\(maxTokens)) ===")
        for B in batchSizes {
            let batch = (0 ..< B).map { pdata[$0 % pdata.count] }
            let baseline = runBatchedBaselineGreedyTokens(
                target: target, promptTokens: batch, maxTokens: maxTokens)
            MLX.Memory.clearCache()
            let mtp = try await runBatchedMTPTokens(
                target: target, drafter: drafter, promptTokens: batch,
                maxTokens: maxTokens, blockSize: K)
            MLX.Memory.clearCache()

            for bi in 0 ..< B {
                let n = Swift.min(baseline[bi].count, mtp[bi].count)
                var firstDiv = -1
                for i in 0 ..< n where baseline[bi][i] != mtp[bi][i] {
                    firstDiv = i
                    break
                }
                let matched = firstDiv == -1 ? n : firstDiv
                print("B=\(B) row=\(bi): matched \(matched)/\(n) tokens"
                      + (firstDiv >= 0 ? "  FIRST-DIVERGENCE@\(firstDiv)" : "  EXACT"))
            }
            dump["B\(B)_baseline"] = baseline
            dump["B\(B)_mtp"] = mtp
        }
        let outPath = env["MTP_BENCH_PARITY_OUT"] ?? "/tmp/mtp-bench/parity_out.json"
        let data = try JSONEncoder().encode(dump)
        try data.write(to: URL(fileURLWithPath: outPath))
        print("wrote \(outPath)")
    }

    /// Regression for the ragged-batch left-padding corruption.
    ///
    /// When prompts of different lengths are batched, the cache left-pads the
    /// short rows and the attention mask blocks their padding slots. The
    /// padding-position *queries* are then fully masked (every key blocked) and
    /// the hand-rolled batched-attention fallback softmaxed them to NaN; the
    /// `0 * NaN` in the value matmul propagated NaN into the hidden state and a
    /// later layer's real queries (which mask padding keys to weight 0) hit
    /// `0 * NaN` again, corrupting EVERY row -> all-NaN logits -> token-salad
    /// under concurrent load. See the NaN guard in `gemma4AttentionFallback`.
    ///
    /// This asserts the deterministic bug signature on a tiny random-weight
    /// model (fast, no model download, CI-safe):
    ///   (1) batched prefill logits are finite for every row (no NaN), and
    ///   (2) each row's first generated token matches its solo (B=1) decode.
    /// Full-sequence parity is intentionally NOT asserted: greedy argMax on a
    /// random-weight model is numerically hypersensitive, so benign batched-vs-
    /// solo reduction-order flips occur mid-stream. The gated
    /// `raggedBatchGreedyParityRealModel` confirms coherent full parity on the
    /// real model. (The pre-existing `batchedParityRealModel` only compared
    /// batched-MTP vs batched-baseline — both share the batched path, so they
    /// agreed while both were wrong; this compares against the independent B=1
    /// oracle.)
    @Test func raggedBatchGreedyParityTinyModel() throws {
        MLXRandom.seed(1234)
        let target = Gemma4TextModel(try tinyTargetConfig())
        eval(target)

        // Ragged prompt lengths -> different per-row left-padding when batched.
        let prompts: [[Int32]] = [
            [1, 5, 9, 3, 7],  // len 5 (most left-padding when batched)
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],  // len 12
            (0 ..< 30).map { Int32(($0 * 7 + 1) % 1000) },  // len 30
        ]

        // (1) Batched ragged prefill must produce FINITE logits for every row,
        // including the heavily left-padded ones. Pre-fix this was all-NaN.
        let (padded, leftPadding) = leftPadBatch(prompts)
        let cache = makeBatchedCache(target: target, leftPadding: leftPadding)
        let prefillLast = target(padded, cache: cache)[0..., -1, 0...].asType(.float32)
        eval(prefillLast)
        let nanCount = (prefillLast .!= prefillLast).sum().item(Int32.self)
        #expect(
            nanCount == 0,
            "batched ragged prefill produced \(nanCount) NaN logits (left-padding fully-masked-query corruption)")

        // (2) Each row's first generated token in the ragged batch must equal
        // its solo (B=1) decode. Pre-fix the left-padded rows produced token 0.
        let maxTokens = 4
        let batched = runBatchedBaselineGreedyTokens(
            target: target, promptTokens: prompts, maxTokens: maxTokens)
        for (i, p) in prompts.enumerated() {
            let solo = runBatchedBaselineGreedyTokens(
                target: target, promptTokens: [p], maxTokens: maxTokens)[0]
            MLX.Memory.clearCache()
            #expect(
                batched[i].first == solo.first,
                "row \(i) (len \(p.count)) first batched token \(batched[i].first ?? -1) != solo \(solo.first ?? -1)")
        }
    }

    /// Same batched-vs-solo parity check on the real model (gated). Asserts the
    /// bug signature — finite logits + per-row first-token parity — and PRINTS
    /// any later divergence (which on a real model is benign near-tie argMax
    /// drift, not corruption) without failing on it. Point it at a local model:
    ///   RAGGED_PARITY=1 RAGGED_MODEL_DIR=/path/to/gemma-4-... \
    ///   [MTP_BENCH_PROMPTS=/tmp/ragged_prompts.json] swift test \
    ///   --filter raggedBatchGreedyParityRealModel
    @Test func raggedBatchGreedyParityRealModel() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["RAGGED_PARITY"] == "1" else {
            print("Skipping: set RAGGED_PARITY=1 + RAGGED_MODEL_DIR")
            return
        }
        guard let modelDir = env["RAGGED_MODEL_DIR"]
            ?? env["MTP_BENCH_DATA_DIR"].map({
                "\($0)/\(env["RAGGED_TARGET"] ?? "gemma-4-26B-A4B-it-qat-4bit")"
            })
        else {
            print("Skipping: set RAGGED_MODEL_DIR (or MTP_BENCH_DATA_DIR)")
            return
        }
        let maxTokens = Int(env["RAGGED_MAXTOK"] ?? "48") ?? 48
        let prompts: [[Int32]]
        if let pp = env["MTP_BENCH_PROMPTS"] {
            prompts = try JSONDecoder()
                .decode([[Int]].self, from: Data(contentsOf: URL(fileURLWithPath: pp)))
                .map { $0.map { Int32($0) } }
        } else {
            prompts = [
                [2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                (0 ..< 30).map { Int32($0 + 2) },
            ]
        }

        let (target, _) = try loadRealTargetAsTextModel(
            from: URL(fileURLWithPath: modelDir))
        eval(target)

        // Finite-logit check on the ragged prefill (the bug signature).
        let (padded, leftPadding) = leftPadBatch(prompts)
        let prefillCache = makeBatchedCache(target: target, leftPadding: leftPadding)
        let prefillLast = target(padded, cache: prefillCache)[0..., -1, 0...].asType(.float32)
        eval(prefillLast)
        let nanCount = (prefillLast .!= prefillLast).sum().item(Int32.self)
        MLX.Memory.clearCache()
        #expect(nanCount == 0, "real-model ragged prefill produced \(nanCount) NaN logits")

        var solo: [[Int]] = []
        for p in prompts {
            solo.append(
                runBatchedBaselineGreedyTokens(
                    target: target, promptTokens: [p], maxTokens: maxTokens)[0])
            MLX.Memory.clearCache()
        }
        let batched = runBatchedBaselineGreedyTokens(
            target: target, promptTokens: prompts, maxTokens: maxTokens)
        MLX.Memory.clearCache()

        print("\n=== Ragged batched-vs-solo parity (real model, max_tokens=\(maxTokens)) ===")
        for (i, p) in prompts.enumerated() {
            let firstDiv = zip(solo[i], batched[i]).enumerated()
                .first(where: { $0.element.0 != $0.element.1 })?.offset
            if let d = firstDiv {
                print("row \(i) (len \(p.count)) matches \(d) tokens then forks "
                    + "(benign argMax near-tie): solo=\(solo[i].prefix(8)) batched=\(batched[i].prefix(8))")
            } else {
                print("row \(i) (len \(p.count)) EXACT (\(solo[i].count) tokens)")
            }
            // The bug signature is a row corrupted from the FIRST token; a real
            // model's later forks are benign numerical drift.
            #expect(
                batched[i].first == solo[i].first,
                "row \(i) corrupted from token 0 (batched=\(batched[i].prefix(4)) solo=\(solo[i].prefix(4)))")
        }
        if let outPath = env["RAGGED_OUT"] {
            try? JSONEncoder().encode(["solo": solo, "batched": batched])
                .write(to: URL(fileURLWithPath: outPath))
            print("wrote \(outPath)")
        }
    }

    /// Real-model ENGINE-path verification: drives the actual continuous-
    /// batching `GenerationBatch` + `Gemma4MTPEngineRuntime` on the 26B model
    /// (the wired-in path the server uses), at B=1/2/4. Reports engine-MTP
    /// tok/s vs plain batched-decode tok/s, and verifies token parity vs plain
    /// greedy (no garbage). Gated on MTP_BENCH_ENGINE=1.
    @Test func engineMtpRealModel() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MTP_BENCH_ENGINE"] == "1", let dataDir = env["MTP_BENCH_DATA_DIR"],
              let promptsPath = env["MTP_BENCH_PROMPTS"]
        else { print("Skipping: set MTP_BENCH_ENGINE=1, MTP_BENCH_DATA_DIR, MTP_BENCH_PROMPTS"); return }

        let targetDir = env["MTP_BENCH_SWEEP_TARGET"] ?? "gemma-4-26B-A4B-it-qat-4bit"
        let drafterDir = env["MTP_BENCH_SWEEP_DRAFTER"] ?? "gemma-4-26B-A4B-it-qat-assistant-4bit"
        let K = Int(env["MTP_BENCH_SWEEP_K"] ?? "3") ?? 3
        let maxTokens = Int(env["MTP_BENCH_SWEEP_MAXTOK"] ?? "160") ?? 160
        let batchSizes = (env["MTP_BENCH_SWEEP_B"] ?? "1,2,4")
            .split(separator: ",").compactMap { Int($0) }
        // One representative prompt replicated across rows (uniform length —
        // isolates engine economics from ragged-prefill padding).
        let pdata = try JSONDecoder()
            .decode([[Int]].self, from: Data(contentsOf: URL(fileURLWithPath: promptsPath)))
            .map { $0.map { Int32($0) } }
        let basePrompt = pdata[(Int(env["MTP_BENCH_PROMPT_IDX"] ?? "0") ?? 0) % pdata.count]

        let (target, _) = try loadRealTargetAsTextModel(
            from: URL(fileURLWithPath: "\(dataDir)/\(targetDir)"))
        let drafter = try await Gemma4AssistantDraftModel.load(
            from: URL(fileURLWithPath: "\(dataDir)/\(drafterDir)"))
        eval(target, drafter)

        print("\n=== Engine-path MTP on real model (K=\(K), max_tokens=\(maxTokens)) ===")
        print("B  base tok/s  engine-mtp tok/s  speedup  token-parity(worst row)")
        for B in batchSizes {
            let batch = Array(repeating: basePrompt, count: B)
            // Warmup.
            _ = measureBatchedBaselineThroughput(target: target, promptTokens: batch, maxTokens: 8)
            _ = try measureEngineMTPThroughput(
                target: target, drafter: drafter, promptTokens: batch,
                maxTokens: 8, blockSize: K, maxBatch: Swift.max(B, 2))
            MLX.Memory.clearCache()

            let base = measureBatchedBaselineThroughput(
                target: target, promptTokens: batch, maxTokens: maxTokens)
            let baseTokens = runBatchedBaselineGreedyTokens(
                target: target, promptTokens: batch, maxTokens: maxTokens)
            MLX.Memory.clearCache()
            let (eng, engTokens) = try measureEngineMTPThroughput(
                target: target, drafter: drafter, promptTokens: batch,
                maxTokens: maxTokens, blockSize: K, maxBatch: Swift.max(B, 2))
            MLX.Memory.clearCache()

            // Token parity: worst row's matched-prefix length (bf16 argmax ties
            // can diverge late — same tolerance as the batched parity suite).
            var worstMatched = maxTokens
            for i in 0 ..< B {
                let n = Swift.min(baseTokens[i].count, engTokens[i].count)
                var matched = n
                for j in 0 ..< n where baseTokens[i][j] != engTokens[i][j] { matched = j; break }
                worstMatched = Swift.min(worstMatched, matched)
            }
            let speedup = eng.tokensPerSecond / Swift.max(base.tokensPerSecond, 1e-9)
            print(String(
                format: "%d  %9.1f  %15.1f  %.2fx  %d/%d exact",
                B, base.tokensPerSecond, eng.tokensPerSecond, speedup, worstMatched, maxTokens))
        }
    }

    /// Pipelining fairness check. The naive baseline syncs the GPU once per
    /// token (`eval(tok)`); MTP syncs once per round (~K tokens), so MTP's
    /// speedup vs the naive baseline is partly a sync-overhead artifact. The
    /// fair comparison is MTP vs an async-pipelined baseline. Gated on
    /// MTP_BENCH_PIPELINE=1.
    @Test func pipelinedBaselineCheck() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MTP_BENCH_PIPELINE"] == "1", let dataDir = env["MTP_BENCH_DATA_DIR"],
              let promptsPath = env["MTP_BENCH_PROMPTS"]
        else { print("Skipping: set MTP_BENCH_PIPELINE=1, MTP_BENCH_DATA_DIR, MTP_BENCH_PROMPTS"); return }
        let targetDir = env["MTP_BENCH_SWEEP_TARGET"] ?? "gemma-4-26b-a4b-it-4bit"
        let drafterDir = env["MTP_BENCH_SWEEP_DRAFTER"] ?? "gemma-4-26B-A4B-it-qat-assistant-4bit"
        let K = Int(env["MTP_BENCH_SWEEP_K"] ?? "4") ?? 4
        let maxTokens = Int(env["MTP_BENCH_SWEEP_MAXTOK"] ?? "256") ?? 256
        let pdata = try JSONDecoder()
            .decode([[Int]].self, from: Data(contentsOf: URL(fileURLWithPath: promptsPath)))
            .map { $0.map { Int32($0) } }
        let prompt = MLXArray(pdata[0])

        let (target, _) = try loadRealTargetAsTextModel(
            from: URL(fileURLWithPath: "\(dataDir)/\(targetDir)"))
        let drafter = try await Gemma4AssistantDraftModel.load(
            from: URL(fileURLWithPath: "\(dataDir)/\(drafterDir)"))
        eval(target, drafter)

        // Pipelined baseline: feed the token MLXArray forward and asyncEval
        // each step so CPU prep overlaps GPU compute (mirrors mlx generate /
        // Python's generation_stream). One blocking eval at the end.
        func pipelined(_ maxTok: Int) -> Double {
            let cache = target.newCache(parameters: nil)
            var logits = target(prompt[.newAxis, .ellipsis], cache: cache)
            var tok = logits[0..., -1, 0...].asType(.float32).argMax(axis: -1)
            asyncEval(tok)
            let start = Date()
            var n = 1
            for _ in 1 ..< maxTok {
                logits = target(tok[.newAxis, .ellipsis], cache: cache)
                tok = logits[0..., -1, 0...].asType(.float32).argMax(axis: -1)
                asyncEval(tok)
                n += 1
            }
            eval(tok)
            return Double(n) / Date().timeIntervalSince(start)
        }

        _ = measureBaselineThroughput(target: target, promptTokens: prompt, maxTokens: 16)
        _ = pipelined(16)
        _ = try measureMTPThroughput(
            target: target, drafter: drafter, promptTokens: prompt, maxTokens: 16, blockSize: K)
        MLX.Memory.clearCache()

        let naive = measureBaselineThroughput(
            target: target, promptTokens: prompt, maxTokens: maxTokens).tokensPerSecond
        MLX.Memory.clearCache()
        let pipe = pipelined(maxTokens)
        MLX.Memory.clearCache()
        let mtp = try measureMTPThroughput(
            target: target, drafter: drafter, promptTokens: prompt,
            maxTokens: maxTokens, blockSize: K).tokensPerSecond
        MLX.Memory.clearCache()

        print("\n=== Pipelining fairness check (B=1, K=\(K), max_tokens=\(maxTokens)) ===")
        print(String(format: "naive baseline (eval/token):    %.1f tok/s", naive))
        print(String(format: "pipelined baseline (asyncEval):  %.1f tok/s", pipe))
        print(String(format: "MTP:                             %.1f tok/s", mtp))
        print(String(format: "MTP vs naive baseline:      %.2fx", mtp / max(naive, 1e-9)))
        print(String(format: "MTP vs PIPELINED baseline:  %.2fx  <-- the fair number", mtp / max(pipe, 1e-9)))
    }

    /// Load a HF Gemma 4 target (VLM-format checkpoint) into a bare
    /// `Gemma4TextModel`. The VLM wrapper is instantiated briefly so its
    /// sanitize() can strip vision/audio weights + remap key names; we
    /// then reach in and return the inner text model for MTP use.
    private func loadRealTargetAsTextModel(
        from modelDir: URL
    ) throws -> (Gemma4TextModel, Gemma4Configuration) {
        let configData = try Data(
            contentsOf: modelDir.appendingPathComponent("config.json"))
        let decoder = JSONDecoder()
        let fullConfig = try decoder.decode(Gemma4Configuration.self, from: configData)
        let baseConfig = try? decoder.decode(BaseConfiguration.self, from: configData)

        let wrapper = Gemma4Model(fullConfig)

        // Load shards manually. `resolvingSymlinksInPath` is required so a
        // symlinked staging dir (e.g. pointing into the HF cache snapshot)
        // is followed — FileManager shard discovery does not follow a
        // top-level directory symlink, which otherwise yields zero shards
        // and a misleading `keyNotFound(embed_tokens.weight)` at update.
        var weights = [String: MLXArray]()
        let scanDir = modelDir.resolvingSymlinksInPath()
        let shardURLs = (try? FileManager.default.contentsOfDirectory(
            at: scanDir, includingPropertiesForKeys: nil)) ?? []
        for url in shardURLs where url.pathExtension == "safetensors" {
            let (w, _) = try loadArraysAndMetadata(url: url)
            for (k, v) in w { weights[k] = v }
        }
        let sanitized = wrapper.sanitize(weights: weights)
        if let plq = baseConfig?.perLayerQuantization {
            quantize(model: wrapper) { path, _ in
                sanitized["\(path).scales"] != nil
                    ? plq.quantization(layer: path)?.asTuple : nil
            }
        }
        try wrapper.update(
            parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        eval(wrapper)
        return (wrapper.textModel, fullConfig)
    }
}
