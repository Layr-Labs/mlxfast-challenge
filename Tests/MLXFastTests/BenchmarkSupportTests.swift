import Foundation
@testable import MLXFastCore
@testable import MLXFastModel
@testable import MLXFastHarness
import Testing

@Test
func runtimeWorkerClientSkipsNonJSONStdoutLines() {
    #expect(runtimeWorkerLineLooksLikeJSONResponse(Data("  {\"id\":1,\"ok\":true}".utf8)))
    #expect(!runtimeWorkerLineLooksLikeJSONResponse(Data("Metal device initialized".utf8)))
    #expect(!runtimeWorkerLineLooksLikeJSONResponse(Data("".utf8)))
}

// metrics.commit must come from the trusted dispatch context when the ranked
// pipeline supplies it: on the ranked box the harness runs as the sandboxed
// bench uid where `git rev-parse` fails (dubious ownership in the runner-owned
// workspace copy), which produced empty commits that failed every ranked
// score's commit binding. git stays the local/dev fallback only.
@Test
func commitIdentifierPrefersTrustedDispatchSHAOverGit() {
    let fullSHA = "5f95c4bdce07a0ef79ea350c91d9eb0d7476cf2f"
    #expect(GemmaRuntime.commitIdentifier(environment: ["MLXFAST_COMMIT_SHA": fullSHA]) == fullSHA)
    // Short (rev-parse --short style) values and surrounding whitespace are fine.
    #expect(GemmaRuntime.commitIdentifier(environment: ["MLXFAST_COMMIT_SHA": "5f95c4bdce07"]) == "5f95c4bdce07")
    #expect(GemmaRuntime.commitIdentifier(environment: ["MLXFAST_COMMIT_SHA": " \(fullSHA)\n"]) == fullSHA)

    // Values that could not satisfy the trusted shell predicates fall back to
    // git instead of being stamped verbatim into the sealed score.
    for invalid in [
        "",
        "5f95c4",
        fullSHA.uppercased(),
        fullSHA + "0",
        "not-a-commit-sha",
        "5f95c4bdce07;rm -rf /",
    ] {
        let fallback = GemmaRuntime.commitIdentifier(environment: ["MLXFAST_COMMIT_SHA": invalid])
        #expect(fallback != invalid || invalid.isEmpty)
        #expect(!fallback.contains(";"))
    }

    #expect(GemmaRuntime.isCommitSHAHex(fullSHA))
    #expect(GemmaRuntime.isCommitSHAHex("abcdef0"))
    #expect(!GemmaRuntime.isCommitSHAHex("abcdef"))
    #expect(!GemmaRuntime.isCommitSHAHex(String(repeating: "a", count: 41)))
    #expect(!GemmaRuntime.isCommitSHAHex("ABCDEF0"))
}

@Test
func runtimeWorkerEnvironmentStripsOfficialRunAndCIIdentity() {
    let sanitized = sanitizedRuntimeWorkerEnvironment([
        "ANTHROPIC_API_KEY": "secret",
        "CI": "true",
        "GITHUB_ACTIONS": "true",
        "GITHUB_RUN_ID": "123",
        "RUNNER_TEMP": "/tmp/runner",
        "BLACKSMITH_RUNNER": "1",
        "MLXFAST_OFFICIAL_BENCHMARK_RUN": "1",
        "MLXFAST_RUN_BENCHMARK": "1",
        "MLXFAST_REFERENCE_DIR": "/private/reference",
        "MLXFAST_PRIVATE_DIR": "/private/golden",
        "MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE": "/tmp/profile.sb",
        "R2_ACCESS_KEY_ID": "key",
        "MLXFAST_MAX_WEIGHTS_BYTES": "42",
        "PATH": "/usr/bin",
        // A gates-only or timing-only parallel-split machine sets these to
        // tell the trusted CLI which half of the original single-machine run
        // this process covers -- on one machine, decode/prefill was always
        // timed at the same time gates were checked, so submitted code could
        // never previously tell "my speed doesn't count right now" from "my
        // correctness doesn't count right now." These must not reach the
        // sandboxed worker submitted code executes in.
        "MLXFAST_BENCHMARK_CHECK_GATES": "0",
        "MLXFAST_BENCHMARK_CORRECTNESS_STEPS": "0",
        "MLXFAST_BENCHMARK_SKIP_TIMED": "1",
        // Env-var forms of --base-case-only/--step-range: the slice machines'
        // equivalents of the split-phase vars above.
        "MLXFAST_CORRECTNESS_BASE_CASE_ONLY": "1",
        "MLXFAST_CORRECTNESS_STEP_RANGE": "21-42",
        // Same-session baseline timings from the trusted paired-baseline step;
        // submitted code must not observe the reference implementation's live
        // numbers (or that the run is paired at all).
        "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "0.17",
        "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "3.6",
    ])

    for key in [
        "ANTHROPIC_API_KEY",
        "CI",
        "GITHUB_ACTIONS",
        "GITHUB_RUN_ID",
        "RUNNER_TEMP",
        "BLACKSMITH_RUNNER",
        "MLXFAST_OFFICIAL_BENCHMARK_RUN",
        "MLXFAST_RUN_BENCHMARK",
        "MLXFAST_REFERENCE_DIR",
        "MLXFAST_PRIVATE_DIR",
        "MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE",
        "R2_ACCESS_KEY_ID",
        "MLXFAST_BENCHMARK_CHECK_GATES",
        "MLXFAST_BENCHMARK_CORRECTNESS_STEPS",
        "MLXFAST_BENCHMARK_SKIP_TIMED",
        "MLXFAST_CORRECTNESS_BASE_CASE_ONLY",
        "MLXFAST_CORRECTNESS_STEP_RANGE",
        "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN",
        "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN",
    ] {
        #expect(sanitized[key] == nil)
    }
    #expect(sanitized["MLXFAST_USE_RUNTIME_WORKER"] == "0")
    #expect(sanitized["MLXFAST_MAX_WEIGHTS_BYTES"] == "42")
    #expect(sanitized["PATH"] == "/usr/bin")
}

@Test
func pairedBaselineOverrideParsesTrustedEnvironmentFailClosed() throws {
    // Absent entirely: no pairing, callers fall back to golden/constants.
    #expect(try PairedBaselineOverride.fromEnvironment([:]) == nil)
    #expect(try PairedBaselineOverride.fromEnvironment([
        "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "",
        "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "  ",
    ]) == nil)

    // Present: both values parsed precisely.
    let override = try #require(try PairedBaselineOverride.fromEnvironment([
        "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "0.16518489738085937",
        "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "3.6977511595078125",
    ]))
    #expect(override.prefillSecondsPerToken == 0.16518489738085937)
    #expect(override.decodeSecondsPerToken == 3.6977511595078125)

    // Half-set pairs and non-positive/non-finite values are operator wiring
    // errors: fail closed rather than silently repricing against constants.
    for badEnvironment: [String: String] in [
        ["MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "0.17"],
        ["MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "3.6"],
        [
            "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "0",
            "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "3.6",
        ],
        [
            "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "0.17",
            "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "-1",
        ],
        [
            "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "inf",
            "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "3.6",
        ],
        [
            "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "fast",
            "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "3.6",
        ],
    ] {
        #expect(throws: MLXFastError.self) {
            _ = try PairedBaselineOverride.fromEnvironment(badEnvironment)
        }
    }
}

@Test
func semanticBehaviorGateRequiresPromptAndReferenceAnswer() {
    let exactOnly = GoldenBehaviorCase(
        name: "exact-only",
        promptTokens: [1],
        acceptedTokenSequences: [[2]],
        maxNewTokens: 1
    )
    let missingReference = GoldenBehaviorCase(
        name: "missing-reference",
        promptTokens: [1],
        acceptedTokenSequences: [[2]],
        maxNewTokens: 1,
        semanticPrompt: "question"
    )
    let semantic = GoldenBehaviorCase(
        name: "semantic",
        promptTokens: [1],
        acceptedTokenSequences: [[2]],
        maxNewTokens: 1,
        semanticPrompt: "question",
        semanticReferenceAnswer: "answer"
    )

    #expect(!GemmaRuntime.behaviorUsesSemanticJudge(exactOnly))
    #expect(!GemmaRuntime.behaviorUsesSemanticJudge(missingReference))
    #expect(GemmaRuntime.behaviorUsesSemanticJudge(semantic))
}

@Test
func correctnessAcceptsOnlyExactTopLogitTies() {
    #expect(correctnessTokenAccepted(
        expectedToken: 30,
        actualToken: 1,
        topLogits: [
            CorrectnessTraceLogit(token: 1, logit: 19.25),
            CorrectnessTraceLogit(token: 30, logit: 19.25),
        ]
    ))
    #expect(!correctnessTokenAccepted(
        expectedToken: 30,
        actualToken: 1,
        topLogits: [
            CorrectnessTraceLogit(token: 1, logit: 19.25),
            CorrectnessTraceLogit(token: 30, logit: 19.0),
        ]
    ))
    #expect(!correctnessTokenAccepted(expectedToken: 30, actualToken: 1, topLogits: nil))
}

@Test
func failedScoreRedactsCorrectnessTokenMismatchByDefault() {
    let report = CorrectnessReport(
        passed: false,
        checkedSteps: 7,
        caseCount: 1,
        firstFailingCase: "local-iterate",
        firstFailingStep: 6,
        expectedToken: 123,
        actualToken: 456,
        goldenHash: "golden",
        error: "token mismatch"
    )

    let payload = GemmaRuntime.failedScore(
        error: "token mismatch",
        correctness: report,
        passedCorrectness: false,
        runtime: "swift-local-iterate"
    )

    #expect(payload.score == nil)
    #expect(payload.passed == false)
    #expect(payload.metrics.runtime == "swift-local-iterate")
    #expect(payload.metrics.firstFailingCase == "local-iterate")
    #expect(payload.metrics.firstFailingStep == 6)
    #expect(payload.metrics.expectedToken == nil)
    #expect(payload.metrics.actualToken == nil)
    #expect(payload.metrics.bandwidthGBPerToken == 0)
}

@Test
func failedScorePreservesExplicitPublicMismatchTokensAndRuntimeLabel() {
    let report = CorrectnessReport(
        passed: false,
        checkedSteps: 7,
        caseCount: 1,
        firstFailingCase: "local-iterate",
        firstFailingStep: 6,
        expectedToken: 123,
        actualToken: 456,
        goldenHash: "golden",
        error: "token mismatch"
    )

    let payload = GemmaRuntime.failedScore(
        error: "token mismatch",
        correctness: report,
        passedCorrectness: false,
        expectedToken: report.expectedToken,
        actualToken: report.actualToken,
        runtime: "swift-local-iterate"
    )

    #expect(payload.score == nil)
    #expect(payload.passed == false)
    #expect(payload.metrics.runtime == "swift-local-iterate")
    #expect(payload.metrics.firstFailingCase == "local-iterate")
    #expect(payload.metrics.firstFailingStep == 6)
    #expect(payload.metrics.expectedToken == 123)
    #expect(payload.metrics.actualToken == 456)
}

@Test
func decodeTimingPlanStartsAfterSeedPrefill() throws {
    let plan = try DecodeTimingPlan(seedTokenCount: 32, decodeSteps: 4)
    var offsets: [Int] = []
    for step in 0..<plan.decodeSteps {
        offsets.append(try plan.positionOffset(forDecodedStep: step))
    }

    #expect(offsets == [32, 33, 34, 35])
}

@Test
func decodeTimingPlanRejectsInvalidRanges() throws {
    #expect(throws: MLXFastError.self) {
        _ = try DecodeTimingPlan(seedTokenCount: 0, decodeSteps: 4)
    }
    #expect(throws: MLXFastError.self) {
        _ = try DecodeTimingPlan(seedTokenCount: 32, decodeSteps: 0)
    }

    let plan = try DecodeTimingPlan(seedTokenCount: 32, decodeSteps: 4)
    #expect(throws: MLXFastError.self) {
        _ = try plan.positionOffset(forDecodedStep: 4)
    }
}

@Test
func submissionValidationDelayDefaultsToZero() throws {
    #expect(Gemma4SubmissionControls.measuredDecodeDelayMilliseconds == 0)
    #expect(try GemmaRuntime.submissionValidationDelayMilliseconds() == 0)
}

@Test
func benchmarkPromptPlanUsesHiddenBenchmarkOracle() throws {
    let prefill = Array(0..<MLXFastConstants.benchmarkPrefillPromptTokens)
    let seed = Array(0..<MLXFastConstants.benchmarkDecodeSeedTokens)
    let decode = Array(repeating: 9, count: MLXFastConstants.benchmarkDecodeSteps)
    let plan = try BenchmarkPrompt.plan(from: BenchmarkGolden(
        prefillPromptTokens: prefill,
        expectedPrefillToken: 17,
        decodeSeedTokens: seed,
        expectedDecodeSeedToken: 23,
        expectedDecodeTokens: decode
    ))

    #expect(plan.prefillTokens == prefill)
    #expect(plan.expectedPrefillToken == 17)
    #expect(plan.decodeSeedTokens == seed)
    #expect(plan.expectedDecodeSeedToken == 23)
    #expect(plan.expectedDecodeTokens == decode)
}

@Test
func benchmarkPromptPlanRejectsMalformedBenchmarkOracle() {
    #expect(throws: MLXFastError.self) {
        _ = try BenchmarkPrompt.plan(from: BenchmarkGolden(
            prefillPromptTokens: [1],
            expectedPrefillToken: 7,
            decodeSeedTokens: Array(repeating: 1, count: MLXFastConstants.benchmarkDecodeSeedTokens),
            expectedDecodeSeedToken: 7,
            expectedDecodeTokens: Array(repeating: 7, count: MLXFastConstants.benchmarkDecodeSteps)
        ))
    }
}

@Test
func defaultTransformedWeightsLimitIsTwentyFiveGiB() {
    #expect(MLXFastConstants.defaultMaxTransformedWeightsBytes == 25 * 1024 * 1024 * 1024)
}

@Test
func benchmarkPreflightAcceptsRequiredArtifacts() throws {
    let fixture = try makePreflightFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let report = try checkPreflight(fixture)

    #expect(report.weightsPath == fixture.weights.path)
    #expect(report.goldenPath == fixture.golden.path)
    #expect(report.weightsByteCount > 0)
    #expect(report.maxWeightsByteCount == MLXFastConstants.defaultMaxTransformedWeightsBytes)
}

@Test
func benchmarkPreflightRejectsWeightsAboveDefaultByteLimit() throws {
    let fixture = try makePreflightFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    try writeSparseFile(
        fixture.weights.appendingPathComponent("oversized.bin"),
        byteCount: MLXFastConstants.defaultMaxTransformedWeightsBytes + 1
    )

    #expect(throws: MLXFastError.self) {
        _ = try checkPreflight(fixture)
    }
}

@Test
func benchmarkPreflightHonorsConfiguredWeightsByteLimit() throws {
    let fixture = try makePreflightFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    try writeSparseFile(
        fixture.weights.appendingPathComponent("large-but-allowed.bin"),
        byteCount: MLXFastConstants.defaultMaxTransformedWeightsBytes + 1
    )
    let override = MLXFastConstants.defaultMaxTransformedWeightsBytes * 2

    let report = try checkPreflight(
        fixture,
        environment: [
            "MLXFAST_MAX_WEIGHTS_BYTES": "\(override)",
        ]
    )

    #expect(report.weightsByteCount > MLXFastConstants.defaultMaxTransformedWeightsBytes)
    #expect(report.maxWeightsByteCount == override)
}

@Test
func benchmarkPreflightRejectsMalformedGolden() throws {
    let fixture = try makePreflightFixture(goldenContents: "{}")
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: Error.self) {
        _ = try checkPreflight(fixture)
    }
}

@Test
func benchmarkPreflightRejectsShortBenchmarkPrompt() throws {
    let fixture = try makePreflightFixture(goldenContents: validGoldenJSON(benchmarkPromptTokens: [1]))
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try checkPreflight(fixture)
    }
}

@Test
func benchmarkPreflightRejectsMissingBenchmarkOracle() throws {
    let expected = arrayJSON(Array(repeating: 7, count: MLXFastConstants.correctnessSteps))
    let fixture = try makePreflightFixture(goldenContents: """
    {
      "version": 1,
      "cases": [
        {
          "name": "preflight",
          "prompt_tokens": \(arrayJSON(Array(repeating: 1, count: MLXFastConstants.correctnessPromptTokens))),
          "expected_tokens": \(expected)
        }
      ]
    }
    """)
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try checkPreflight(fixture)
    }
}

@Test
func correctnessPreflightAcceptsPublicGoldenWithoutBenchmarkOracle() throws {
    let fixture = try makePreflightFixture(goldenContents: correctnessOnlyGoldenJSON())
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let report = try checkCorrectnessPreflight(fixture)

    #expect(report.weightsPath == fixture.weights.path)
    #expect(report.goldenPath == fixture.golden.path)
}

@Test
func correctnessPreflightHonorsConfiguredWeightsByteLimit() throws {
    let fixture = try makePreflightFixture(goldenContents: correctnessOnlyGoldenJSON())
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try checkCorrectnessPreflight(
            fixture,
            environment: ["MLXFAST_MAX_WEIGHTS_BYTES": "1"]
        )
    }
}

@Test
func nonWorkerBenchmarkRejectsBehaviorGatesBecauseTTFTRequiresWorker() throws {
    let behaviorGate = """
    {
      "behavior": [
        {
          "name": "gpqa-hidden-ttft",
          "prompt_tokens": \(arrayJSON(Array(repeating: 1, count: MLXFastConstants.correctnessPromptTokens))),
          "accepted_token_sequences": [[7]],
          "max_new_tokens": 1,
          "semantic_prompt": "Hidden short-answer prompt",
          "semantic_reference_answer": "Reference answer"
        }
      ]
    }
    """
    let fixture = try makePreflightFixture(goldenContents: validGoldenJSON(correctnessGates: behaviorGate))
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let score = GemmaRuntime.benchmark(
        BenchmarkOptions(
            weightsPath: fixture.weights.path,
            goldenPath: fixture.golden.path,
            correctnessSteps: 1,
            benchmarkDecodeSteps: 1
        ),
        worker: nil
    )

    #expect(score.passed == false)
    #expect(score.metrics.passedCorrectness == false)
    #expect(score.metrics.error == "benchmark behavior and GPQA TTFT gates require runtime worker timing")
    #expect(score.metrics.gpqaTTFTCaseCount == 0)
    #expect(score.metrics.preflightSeconds == 0)
    #expect(score.metrics.weightsByteCount == 0)
    #expect(score.metrics.weightsFileCount == 0)
}

@Test
func benchmarkPreflightRejectsMissingSemanticTensor() throws {
    let fixture = try makePreflightFixture(omitDenseTensorName: Gemma4WeightNames.finalNorm)
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try checkPreflight(fixture)
    }
}

private struct PreflightFixture {
    let root: URL
    let weights: URL
    let golden: URL
}

private struct TensorFixture {
    let name: String
    let dtype: String
    let shape: [Int]
}

private func checkPreflight(
    _ fixture: PreflightFixture,
    environment: [String: String] = [:]
) throws -> BenchmarkPreflightReport {
    try BenchmarkPreflight.check(
        weightsPath: fixture.weights.path,
        goldenPath: fixture.golden.path,
        environment: environment
    )
}

private func checkCorrectnessPreflight(
    _ fixture: PreflightFixture,
    environment: [String: String] = [:]
) throws -> BenchmarkPreflightReport {
    try BenchmarkPreflight.checkCorrectnessArtifacts(
        weightsPath: fixture.weights.path,
        goldenPath: fixture.golden.path,
        environment: environment
    )
}

private func makePreflightFixture(
    goldenContents: String? = nil,
    omitDenseTensorName: String? = nil
) throws -> PreflightFixture {
    let directory = try temporaryDirectory()
    let weights = directory.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)

    try minimalGemma4ConfigJSON().write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    var denseTensors = requiredGemma4DenseTensorFixtures()
    if let omitDenseTensorName {
        denseTensors.removeAll { $0.name == omitDenseTensorName }
    }
    let denseShard = "model-00001.safetensors"
    try writeSafetensors(weights.appendingPathComponent(denseShard), tensors: denseTensors)
    try writeIndex(
        weights.appendingPathComponent("model.safetensors.index.json"),
        tensors: denseTensors,
        shardName: denseShard
    )

    let golden = directory.appendingPathComponent("correctness_golden.json")
    try (goldenContents ?? validGoldenJSON()).write(to: golden, atomically: true, encoding: .utf8)

    return PreflightFixture(root: directory, weights: weights, golden: golden)
}

private func minimalGemma4ConfigJSON() -> String {
    """
    {
      "model_type": "gemma4_text",
      "vocab_size": \(MLXFastConstants.vocabSize),
      "hidden_size": \(MLXFastConstants.hiddenSize),
      "intermediate_size": \(MLXFastConstants.intermediateSize),
      "num_hidden_layers": \(MLXFastConstants.numHiddenLayers),
      "num_attention_heads": \(MLXFastConstants.attentionHeads),
      "num_key_value_heads": 16,
      "num_global_key_value_heads": 4,
      "head_dim": 256,
      "global_head_dim": 512,
      "attention_k_eq_v": true,
      "sliding_window": 1024,
      "tie_word_embeddings": true
    }
    """
}

private func validGoldenJSON(
    correctnessPromptTokens: [Int] = Array(repeating: 1, count: MLXFastConstants.correctnessPromptTokens),
    benchmarkPromptTokens: [Int] = Array(repeating: 1, count: MLXFastConstants.benchmarkPrefillPromptTokens),
    correctnessGates: String? = nil
) -> String {
    let correctnessPrompt = arrayJSON(correctnessPromptTokens)
    let benchmarkPrompt = arrayJSON(benchmarkPromptTokens)
    let expected = arrayJSON(Array(repeating: 7, count: MLXFastConstants.correctnessSteps))
    let seed = arrayJSON(Array(benchmarkPromptTokens.prefix(MLXFastConstants.benchmarkDecodeSeedTokens)))
    let decode = arrayJSON(Array(repeating: 9, count: MLXFastConstants.benchmarkDecodeSteps))
    let gates = correctnessGates.map { ",\n  \"correctness_gates\": \($0)" } ?? ""
    return """
    {
      "version": 1,
      "cases": [
        {
          "name": "preflight",
          "prompt_tokens": \(correctnessPrompt),
          "expected_tokens": \(expected)
        }
      ],
      "benchmark": {
        "prefill_prompt_tokens": \(benchmarkPrompt),
        "expected_prefill_token": 8,
        "decode_seed_tokens": \(seed),
        "expected_decode_seed_token": 7,
        "expected_decode_tokens": \(decode)
      }\(gates)
    }
    """
}

private func correctnessOnlyGoldenJSON() -> String {
    let expected = arrayJSON(Array(repeating: 7, count: MLXFastConstants.correctnessSteps))
    return """
    {
      "version": 1,
      "cases": [
        {
          "name": "preflight",
          "prompt_tokens": \(arrayJSON(Array(repeating: 1, count: MLXFastConstants.correctnessPromptTokens))),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
}

/// Every tensor `Gemma4WeightLoader.validateRequiredMetadata` requires for the
/// full frozen shape (60 layers, alternating five sliding + one full-attention
/// layer per block). Real byte contents do not matter for these preflight
/// tests -- only declared dtype/shape -- so linear weights use the same
/// packed `U32` (affine 4-bit, 8 values/word) layout the real checkpoint
/// ships, which keeps the sparse fixture files under the default transformed-
/// weights byte cap the way the small norm/scalar tensors alone could not.
private func requiredGemma4DenseTensorFixtures() -> [TensorFixture] {
    let hidden = MLXFastConstants.hiddenSize
    let intermediate = MLXFastConstants.intermediateSize
    let vocab = MLXFastConstants.vocabSize
    let layers = MLXFastConstants.numHiddenLayers
    let heads = MLXFastConstants.attentionHeads
    let headDim = 256
    let globalHeadDim = 512
    let kvHeads = 16
    let globalKVHeads = 4

    func packedCols(_ inFeatures: Int) -> Int {
        inFeatures / 8
    }

    var tensors: [TensorFixture] = [
        TensorFixture(name: Gemma4WeightNames.embedTokens, dtype: "U32", shape: [vocab, packedCols(hidden)]),
        TensorFixture(name: Gemma4WeightNames.finalNorm, dtype: "F32", shape: [hidden]),
    ]

    for layerIndex in 0..<layers {
        let isFull = layerIndex % 6 == 5
        let effectiveHeadDim = isFull ? globalHeadDim : headDim
        let effectiveKVHeads = isFull ? globalKVHeads : kvHeads

        for suffix in [
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
        ] {
            tensors.append(
                TensorFixture(name: Gemma4WeightNames.layer(layerIndex, suffix), dtype: "F32", shape: [hidden])
            )
        }
        tensors.append(
            TensorFixture(name: Gemma4WeightNames.layer(layerIndex, "layer_scalar"), dtype: "F32", shape: [1])
        )

        tensors.append(TensorFixture(
            name: Gemma4WeightNames.attention(layerIndex, "q_proj.weight"),
            dtype: "U32",
            shape: [heads * effectiveHeadDim, packedCols(hidden)]
        ))
        tensors.append(TensorFixture(
            name: Gemma4WeightNames.attention(layerIndex, "k_proj.weight"),
            dtype: "U32",
            shape: [effectiveKVHeads * effectiveHeadDim, packedCols(hidden)]
        ))
        if !isFull {
            tensors.append(TensorFixture(
                name: Gemma4WeightNames.attention(layerIndex, "v_proj.weight"),
                dtype: "U32",
                shape: [effectiveKVHeads * effectiveHeadDim, packedCols(hidden)]
            ))
        }
        tensors.append(TensorFixture(
            name: Gemma4WeightNames.attention(layerIndex, "o_proj.weight"),
            dtype: "U32",
            shape: [hidden, packedCols(heads * effectiveHeadDim)]
        ))
        tensors.append(TensorFixture(
            name: Gemma4WeightNames.attention(layerIndex, "q_norm.weight"),
            dtype: "F32",
            shape: [effectiveHeadDim]
        ))
        tensors.append(TensorFixture(
            name: Gemma4WeightNames.attention(layerIndex, "k_norm.weight"),
            dtype: "F32",
            shape: [effectiveHeadDim]
        ))

        for suffix in ["gate_proj", "up_proj"] {
            tensors.append(TensorFixture(
                name: Gemma4WeightNames.mlp(layerIndex, "\(suffix).weight"),
                dtype: "U32",
                shape: [intermediate, packedCols(hidden)]
            ))
        }
        tensors.append(TensorFixture(
            name: Gemma4WeightNames.mlp(layerIndex, "down_proj.weight"),
            dtype: "U32",
            shape: [hidden, packedCols(intermediate)]
        ))
    }

    return tensors
}

private func writeIndex(_ path: URL, tensors: [TensorFixture], shardName: String) throws {
    let entries = tensors.map { #""\#($0.name)": "\#(shardName)""# }.joined(separator: ",")
    try """
    {
      "weight_map": {
        \(entries)
      }
    }
    """.write(to: path, atomically: true, encoding: .utf8)
}

@Test
func localIterateDecodeProgressIntervalReportsEveryStepForShortRuns() {
    // local-iterate: 16 decode steps get a running-number line on every token.
    #expect(GemmaRuntime.localIterateDecodeProgressInterval(totalDecodeSteps: 16, timingRepeats: 1) == 1)
    #expect(GemmaRuntime.localIterateDecodeProgressInterval(totalDecodeSteps: 32, timingRepeats: 1) == 1)
    // local-submit: 1023 steps keep the historical 8-step cadence.
    #expect(GemmaRuntime.localIterateDecodeProgressInterval(totalDecodeSteps: 1023, timingRepeats: 1) == 8)
    // Multi-repeat runs keep the sparser 64-step cadence.
    #expect(GemmaRuntime.localIterateDecodeProgressInterval(totalDecodeSteps: 512, timingRepeats: 4) == 64)
}

@Test
func localIterateProjectedDecodeSecondsPerTokenConvergesToChargedMean() {
    // Mid-run: charged 20s so far (18s seed + 2s steps), 2 of 16 tokens done at
    // 1s/step mean -> project 14 more step-seconds on top of the charged 20.
    let projected = GemmaRuntime.localIterateProjectedDecodeSecondsPerToken(
        chargedSecondsSoFar: 20,
        stepOnlySecondsSoFar: 2,
        decodedTokens: 2,
        totalDecodeTokens: 16
    )
    #expect(abs(projected - (20.0 + 14.0) / 16.0) < 1e-12)

    // Final token: exactly the charged mean the score payload will report.
    let final = GemmaRuntime.localIterateProjectedDecodeSecondsPerToken(
        chargedSecondsSoFar: 32,
        stepOnlySecondsSoFar: 14,
        decodedTokens: 16,
        totalDecodeTokens: 16
    )
    #expect(final == 2.0)

    // Guards.
    #expect(
        GemmaRuntime.localIterateProjectedDecodeSecondsPerToken(
            chargedSecondsSoFar: 1,
            stepOnlySecondsSoFar: 1,
            decodedTokens: 0,
            totalDecodeTokens: 16
        ) == 0
    )
}

@Test
func localIterateLiveDecodeStatusIncludesProjectedSpeedupAndScore() {
    let status = GemmaRuntime.localIterateLiveDecodeStatus(
        lastStepSeconds: 0.9,
        chargedSecondsSoFar: 20,
        stepOnlySecondsSoFar: 2,
        decodedTokens: 2,
        totalDecodeTokens: 16,
        prefillSecondsPerToken: 0.05
    )
    #expect(status.contains("last_step_seconds=0.900000"))
    #expect(status.contains("mean_step_seconds=1.000000"))
    // 14 remaining tokens at 1s mean -> 14s ETA.
    #expect(status.contains("decode_eta_seconds=14.0"))
    #expect(status.contains("projected_decode_seconds_per_token="))
    #expect(status.contains("projected_decode_speedup="))
    #expect(status.contains("projected_score="))
    // The RAM-resident dense runtime reports no expert-bandwidth live fields.
    #expect(!status.contains("expert_gb_per_token="))
    #expect(!status.contains("expert_hit_rate="))

    // The final step has no ETA.
    let finalStep = GemmaRuntime.localIterateLiveDecodeStatus(
        lastStepSeconds: 1,
        chargedSecondsSoFar: 32,
        stepOnlySecondsSoFar: 14,
        decodedTokens: 16,
        totalDecodeTokens: 16,
        prefillSecondsPerToken: 0.05
    )
    #expect(!finalStep.contains("decode_eta_seconds="))

    // Before prefill has a positive measurement there is no score estimate.
    let withoutPrefill = GemmaRuntime.localIterateLiveDecodeStatus(
        lastStepSeconds: 0.9,
        chargedSecondsSoFar: 20,
        stepOnlySecondsSoFar: 2,
        decodedTokens: 2,
        totalDecodeTokens: 16,
        prefillSecondsPerToken: nil
    )
    #expect(withoutPrefill.contains("projected_decode_speedup="))
    #expect(!withoutPrefill.contains("projected_score="))

    #expect(
        GemmaRuntime.localIterateLiveDecodeStatus(
            lastStepSeconds: 0,
            chargedSecondsSoFar: 0,
            stepOnlySecondsSoFar: 0,
            decodedTokens: 0,
            totalDecodeTokens: 16,
            prefillSecondsPerToken: nil
        ).isEmpty
    )
}

@Test
func workerStderrDrainForwardsRedactedLinesAndKeepsTailForDiagnostics() throws {
    final class LineBox: @unchecked Sendable {
        private let lock = NSLock()
        private var lines: [String] = []
        func append(_ line: String) {
            lock.lock()
            defer { lock.unlock() }
            lines.append(line)
        }
        func snapshot() -> [String] {
            lock.lock()
            defer { lock.unlock() }
            return lines
        }
    }

    let pipe = Pipe()
    let box = LineBox()
    let drain = WorkerStderrDrain(
        handle: pipe.fileHandleForReading,
        emit: { box.append($0) }
    )

    // Two complete lines (one needing token redaction) and one unterminated
    // partial line that must still be flushed at EOF.
    try pipe.fileHandleForWriting.write(contentsOf: Data("model debug: layer 3 routed\n".utf8))
    try pipe.fileHandleForWriting.write(contentsOf: Data("expected 5 actual 7\npartial".utf8))
    try pipe.fileHandleForWriting.close()

    let tail = drain.drainedOutput(timeoutSeconds: 5)
    let emitted = box.snapshot()

    #expect(emitted == [
        "mlxfast-worker: model debug: layer 3 routed\n",
        "mlxfast-worker: token-validation-failed\n",
        "mlxfast-worker: partial\n",
    ])
    // The diagnostic tail keeps the RAW content (workerExitDiagnostic applies
    // its own whole-blob sanitization, unchanged from before).
    #expect(tail.contains("model debug: layer 3 routed"))
    #expect(tail.contains("expected 5 actual 7"))
    #expect(tail.contains("partial"))

    // A second read must not block or lose the tail.
    #expect(drain.drainedOutput(timeoutSeconds: 1).contains("partial"))
}

@Test
func workerStderrDrainCapsRetainedTail() throws {
    let pipe = Pipe()
    let drain = WorkerStderrDrain(
        handle: pipe.fileHandleForReading,
        emit: { _ in }
    )

    let filler = String(repeating: "x", count: 1024)
    for index in 0..<128 {
        try pipe.fileHandleForWriting.write(contentsOf: Data("line-\(index) \(filler)\n".utf8))
    }
    try pipe.fileHandleForWriting.write(contentsOf: Data("final-marker\n".utf8))
    try pipe.fileHandleForWriting.close()

    let tail = drain.drainedOutput(timeoutSeconds: 5)
    #expect(tail.utf8.count <= WorkerStderrDrain.tailByteLimit + 16)
    #expect(tail.contains("final-marker"))
    #expect(!tail.contains("line-0 "))
}

@Test
func localIteratePrefillStatusReportsPerTokenAndSpeedup() {
    let status = GemmaRuntime.localIteratePrefillStatus(
        elapsedSeconds: 51.2,
        promptTokens: 512
    )
    #expect(status.contains("seconds=51.2"))
    #expect(status.contains("seconds_per_token=0.100000"))
    let expectedSpeedup = MLXFastConstants.officialBaselinePrefillSecondsPerToken / 0.1
    #expect(status.contains("prefill_speedup=\(String(format: "%.3f", expectedSpeedup))"))

    // Zero-duration or zero-token inputs fall back to the plain seconds field.
    #expect(
        GemmaRuntime.localIteratePrefillStatus(elapsedSeconds: 0, promptTokens: 512)
            == "seconds=0.0"
    )
}

@Test
func localIterateSummaryEmitsSpeedupsAndEstimatedScore() {
    let timing = GemmaRuntime.LocalIterateTimingResult(
        correctness: GemmaRuntime.localIterateCorrectnessReport(
            passed: true,
            checkedSteps: 18,
            caseCount: 1,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil,
            goldenHash: "hash",
            expertStats: .zero,
            error: "",
            modeName: "local-iterate"
        ),
        prefillSecondsPerToken: MLXFastConstants.officialBaselinePrefillSecondsPerToken / 2,
        decode: GemmaRuntime.DecodeMeasurement(
            secondsPerToken: MLXFastConstants.officialBaselineDecodeSecondsPerToken / 2,
            bandwidthGBPerToken: 0,
            bandwidthSource: "ram_resident_model"
        ),
        expertStats: .zero,
        peakRamGB: 24.5
    )

    var lines: [String] = []
    GemmaRuntime.emitLocalIterateSummary(
        modeName: "local-iterate",
        timing: timing,
        progress: { lines.append($0) }
    )

    let joined = lines.joined(separator: "\n")
    #expect(joined.contains("local-iterate summary prefill_seconds_per_token="))
    #expect(joined.contains("prefill_speedup=2.000"))
    #expect(joined.contains("local-iterate summary decode_seconds_per_token="))
    #expect(joined.contains("decode_speedup=2.000"))
    // 2x on both axes -> estimated score 2 under decode^0.75 * prefill^0.25.
    #expect(joined.contains("est_score=2.000"))
    #expect(joined.contains("published as this run's local estimated score, not a ranked score"))
    #expect(joined.contains("decode_bandwidth_gb_per_token=0"))
    #expect(joined.contains("peak_ram_gb=24.500"))
    #expect(!joined.contains("expert_hit_rate="))
}

// The Yukon participant CLI (`mlxfast run`) executes benchmarkCommand and then
// validates the contract scorePath as `{ "score": <finite number>, ... }`;
// `score: null` fails its schema. Local modes therefore publish the estimated
// score (the same decode_speedup^0.75 * prefill_speedup^0.25 estimate the
// summary prints) as a numeric, CLI-usable score.
@Test
func localIterateScorePublishesCLIUsableEstimatedScore() throws {
    let payload = GemmaRuntime.localIterateScore(
        peakRamGB: 24.5,
        bandwidthGBPerToken: 0,
        decodeSecondsPerToken: MLXFastConstants.officialBaselineDecodeSecondsPerToken / 2,
        prefillSecondsPerToken: MLXFastConstants.officialBaselinePrefillSecondsPerToken / 2,
        wallSeconds: 10,
        validationSeconds: 1,
        correctnessSeconds: 5,
        timedSeconds: 5,
        correctness: GemmaRuntime.localIterateCorrectnessReport(
            passed: true,
            checkedSteps: 18,
            caseCount: 1,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil,
            goldenHash: "hash",
            expertStats: .zero,
            error: "",
            modeName: "local-iterate"
        ),
        expertStats: .zero,
        bandwidthSource: "ram_resident_model",
        weightsDigest: nil,
        runtime: "swift-local-iterate"
    )

    // 2x on both axes -> estimated score 2 under decode^0.75 * prefill^0.25.
    let estimated = try #require(payload.score)
    #expect(abs(estimated - 2) < 1e-9)
    #expect(payload.passed)
    // The runtime label is the local-mode marker that keeps the estimate
    // clearly distinguishable from a ranked payload (runtime == "swift").
    #expect(payload.metrics.runtime == "swift-local-iterate")

    // The sealed scorePath JSON must carry the score as a finite JSON number
    // -- the exact thing the CLI's schema checks -- and not null.
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let path = directory.appendingPathComponent("score.json").path
    try writeScorePayload(payload, to: path)
    let raw = try String(contentsOfFile: path, encoding: .utf8)
    #expect(!raw.contains("\"score\" : null"))
    let object = try #require(
        try JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any]
    )
    let rawScore = try #require(object["score"] as? Double)
    #expect(rawScore.isFinite)
    let decoded = try JSONDecoder().decode(ScorePayload.self, from: Data(raw.utf8))
    #expect(decoded.score == estimated)
}

@Test
func localIterateScoreFallsBackToNullForUnusableTimings() {
    // Zero/invalid timings make the estimate non-finite; publish null rather
    // than a fabricated number (such runs are broken and should not read as
    // scoreable).
    let payload = GemmaRuntime.localIterateScore(
        peakRamGB: 0,
        bandwidthGBPerToken: 0,
        decodeSecondsPerToken: 0,
        prefillSecondsPerToken: 0,
        wallSeconds: 0,
        validationSeconds: 0,
        correctnessSeconds: 0,
        timedSeconds: 0,
        correctness: GemmaRuntime.localIterateCorrectnessReport(
            passed: true,
            checkedSteps: 18,
            caseCount: 1,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil,
            goldenHash: "hash",
            expertStats: .zero,
            error: "",
            modeName: "local-iterate"
        ),
        expertStats: .zero,
        bandwidthSource: "ram_resident_model",
        weightsDigest: nil,
        runtime: "swift-local-iterate"
    )

    #expect(payload.score == nil)
    #expect(payload.passed)
}

// Guard the ranked score semantics against the local-mode estimate: the
// official benchmark path still publishes exactly the score it was given on
// pass, and null on failure. Only localIterateScore (reached exclusively via
// --local-iterate/--local-submit) synthesizes an estimate.
@Test
func rankedScoreSemanticsAreUnchangedByLocalEstimatedScore() {
    let correctness = CorrectnessReport(
        passed: true,
        checkedSteps: 513,
        caseCount: 1,
        firstFailingCase: nil,
        firstFailingStep: nil,
        expectedToken: nil,
        actualToken: nil,
        goldenHash: "hash",
        error: ""
    )
    let passed = GemmaRuntime.passedScore(
        score: 1.25,
        peakRamGB: 20,
        bandwidthGBPerToken: 0,
        decodeSecondsPerToken: 0.1,
        prefillSecondsPerToken: 0.01,
        benchmarkWallSeconds: 100,
        preflightSeconds: 1,
        correctnessSeconds: 50,
        timedBenchmarkSeconds: 40,
        numLayers: MLXFastConstants.numHiddenLayers,
        correctness: correctness,
        expertStats: .zero,
        bandwidthSource: "ram_resident_model",
        weightsDigest: nil,
        gpqaTTFT: .zero
    )
    #expect(passed.score == 1.25)
    #expect(passed.metrics.runtime == "swift")

    let failed = GemmaRuntime.failedScore(
        error: "boom",
        correctness: nil,
        passedCorrectness: false,
        runtime: "swift"
    )
    #expect(failed.score == nil)
    #expect(failed.passed == false)
}

@Test
func localIteratePhaseHeartbeatFiresWhileBlockedAndStopsAfterCancel() throws {
    final class MessageBox: @unchecked Sendable {
        private let lock = NSLock()
        private var messages: [String] = []
        func append(_ message: String) {
            lock.lock()
            defer { lock.unlock() }
            messages.append(message)
        }
        func snapshot() -> [String] {
            lock.lock()
            defer { lock.unlock() }
            return messages
        }
    }

    // No progress sink means no timer at all.
    #expect(GemmaRuntime.startPhaseHeartbeat(label: "x", progress: nil) == nil)

    let box = MessageBox()
    let heartbeat = try #require(
        GemmaRuntime.startPhaseHeartbeat(
            label: "local-iterate prefill measured",
            intervalSeconds: 0.05,
            progress: { box.append($0) }
        )
    )
    Thread.sleep(forTimeInterval: 0.3)
    let cancellationDrained = DispatchSemaphore(value: 0)
    heartbeat.setCancelHandler {
        cancellationDrained.signal()
    }
    heartbeat.cancel()
    #expect(cancellationDrained.wait(timeout: .now() + 1) == .success)
    let firedWhileRunning = box.snapshot()
    #expect(!firedWhileRunning.isEmpty)
    #expect(
        firedWhileRunning.allSatisfy {
            $0.hasPrefix("local-iterate prefill measured still running phase_seconds=")
        }
    )

    // After cancellation has drained, the heartbeat must stay quiet.
    Thread.sleep(forTimeInterval: 0.2)
    #expect(box.snapshot().count == firedWhileRunning.count)
}

private func writeSafetensors(_ path: URL, tensors: [TensorFixture]) throws {
    var object: [String: Any] = [:]
    var cursor = 0
    for tensor in tensors.sorted(by: { $0.name < $1.name }) {
        let byteCount = try expectedTensorByteCount(
            name: tensor.name,
            dtype: TensorDType.parse(tensor.dtype),
            shape: tensor.shape
        )
        object[tensor.name] = [
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [cursor, cursor + byteCount],
        ]
        cursor += byteCount
    }

    var header = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    while header.count % 8 != 0 {
        header.append(0x20)
    }

    var output = Data()
    var headerLength = UInt64(header.count).littleEndian
    output.append(Data(bytes: &headerLength, count: 8))
    output.append(header)
    try output.write(to: path)

    let handle = try FileHandle(forWritingTo: path)
    defer {
        try? handle.close()
    }
    try handle.truncate(atOffset: UInt64(output.count + cursor))
}

private func arrayJSON(_ values: [Int]) -> String {
    "[\(values.map(String.init).joined(separator: ","))]"
}

private func writeSparseFile(_ path: URL, byteCount: Int) throws {
    _ = FileManager.default.createFile(atPath: path.path, contents: nil)
    let handle = try FileHandle(forWritingTo: path)
    defer {
        try? handle.close()
    }
    try handle.truncate(atOffset: UInt64(byteCount))
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
