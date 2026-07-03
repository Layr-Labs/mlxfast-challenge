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
        "MLXFAST_EXPERT_CACHE_TENSORS": "42",
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
    #expect(sanitized["MLXFAST_EXPERT_CACHE_TENSORS"] == "42")
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

    #expect(!DeepSeekRuntime.behaviorUsesSemanticJudge(exactOnly))
    #expect(!DeepSeekRuntime.behaviorUsesSemanticJudge(missingReference))
    #expect(DeepSeekRuntime.behaviorUsesSemanticJudge(semantic))
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
        expertCacheHits: 0,
        expertCacheMisses: 0,
        expertCacheEvictions: 0,
        expertBytesRead: 0,
        expertReadSeconds: 0,
        expertPeakCachedTensors: 0,
        expertHitRate: 0,
        firstFailingCase: "local-iterate",
        firstFailingStep: 6,
        expectedToken: 123,
        actualToken: 456,
        goldenHash: "golden",
        error: "token mismatch"
    )

    let payload = DeepSeekRuntime.failedScore(
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
}

@Test
func failedScorePreservesExplicitPublicMismatchTokensAndRuntimeLabel() {
    let report = CorrectnessReport(
        passed: false,
        checkedSteps: 7,
        caseCount: 1,
        expertCacheHits: 0,
        expertCacheMisses: 0,
        expertCacheEvictions: 0,
        expertBytesRead: 0,
        expertReadSeconds: 0,
        expertPeakCachedTensors: 0,
        expertHitRate: 0,
        firstFailingCase: "local-iterate",
        firstFailingStep: 6,
        expectedToken: 123,
        actualToken: 456,
        goldenHash: "golden",
        error: "token mismatch"
    )

    let payload = DeepSeekRuntime.failedScore(
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
    #expect(DeepSeekSubmissionControls.measuredDecodeDelayMilliseconds == 0)
    #expect(try DeepSeekRuntime.submissionValidationDelayMilliseconds() == 0)
}

@Test
func seedForwardExpertReadGateFlagsBypassedForward() throws {
    // A real seed prefill reads far more expert bytes than one decode step: passes.
    try DeepSeekRuntime.requirePlausibleSeedForwardExpertReads(
        seedForwardBytesRead: 40 * (1 << 30),
        decodeStepsBytesRead: 128 * (1 << 30),
        decodeSteps: 128
    )
    // A seed served from an in-process memo reads ~0 expert bytes while the
    // decode steps still stream: must be flagged.
    #expect(throws: MLXFastError.self) {
        try DeepSeekRuntime.requirePlausibleSeedForwardExpertReads(
            seedForwardBytesRead: 0,
            decodeStepsBytesRead: 128 * (1 << 30),
            decodeSteps: 128
        )
    }
    // A seed that read something but still below a single decode step's bytes is
    // implausible for a 512-token forward: flagged.
    #expect(throws: MLXFastError.self) {
        try DeepSeekRuntime.requirePlausibleSeedForwardExpertReads(
            seedForwardBytesRead: (1 << 30) - 1,
            decodeStepsBytesRead: 128 * (1 << 30),
            decodeSteps: 128
        )
    }
    #expect(throws: MLXFastError.self) {
        try DeepSeekRuntime.requirePlausibleSeedForwardExpertReads(
            seedForwardBytesRead: 1,
            decodeStepsBytesRead: 0,
            decodeSteps: 0
        )
    }
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
func benchmarkPreflightRejectsMissingExpertManifest() throws {
    let fixture = try makePreflightFixture(writeManifest: false)
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try checkPreflight(fixture)
    }
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
func correctnessPreflightRejectsExpertManifestOutsideAllowedReferenceRoots() throws {
    let fixture = try makePreflightFixture(goldenContents: correctnessOnlyGoldenJSON())
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try BenchmarkPreflight.checkCorrectnessArtifacts(
            weightsPath: fixture.weights.path,
            goldenPath: fixture.golden.path,
            environment: [
                "MLXFAST_REFERENCE_DIR": fixture.root.appendingPathComponent("different-reference").path,
            ]
        )
    }
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

    let score = DeepSeekRuntime.benchmark(
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
func runtimeCorrectnessRunsArtifactPreflightBeforeModelExecution() throws {
    let fixture = try makePreflightFixture(goldenContents: correctnessOnlyGoldenJSON())
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let report = try DeepSeekRuntime.runCorrectness(
        CorrectnessOptions(weightsPath: fixture.weights.path, goldenPath: fixture.golden.path)
    )

    #expect(!report.passed)
    #expect(report.checkedSteps == 0)
    #expect(report.error.contains("expert manifest reference_path must be under"))
}

@Test
func benchmarkPreflightRejectsMissingSemanticTensor() throws {
    let fixture = try makePreflightFixture(omitDenseTensorName: DeepSeekWeightNames.finalNorm[0])
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try checkPreflight(fixture)
    }
}

@Test
func benchmarkPreflightRejectsUnreadableExpertByteRange() throws {
    let fixture = try makePreflightFixture(expertByteLengthOverride: 1_000_000)
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try checkPreflight(fixture)
    }
}

@Test
func benchmarkPreflightAllowsConfiguredExternalReferenceRoot() throws {
    let fixture = try makePreflightFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let report = try checkPreflight(fixture)

    #expect(report.weightsPath == fixture.weights.path)
}

@Test
func benchmarkPreflightRejectsExpertManifestOutsideAllowedReferenceRoots() throws {
    let fixture = try makePreflightFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try BenchmarkPreflight.check(
            weightsPath: fixture.weights.path,
            goldenPath: fixture.golden.path,
            environment: [
                "MLXFAST_REFERENCE_DIR": fixture.root.appendingPathComponent("different-reference").path,
            ]
        )
    }
}

private struct PreflightFixture {
    let root: URL
    let weights: URL
    let reference: URL
    let golden: URL
}

private struct TensorFixture {
    let name: String
    let dtype: String
    let shape: [Int]
    let data: Data
}

private func checkPreflight(
    _ fixture: PreflightFixture,
    environment extraEnvironment: [String: String] = [:]
) throws -> BenchmarkPreflightReport {
    var environment = extraEnvironment
    environment["MLXFAST_REFERENCE_DIR"] = fixture.reference.path
    return try BenchmarkPreflight.check(
        weightsPath: fixture.weights.path,
        goldenPath: fixture.golden.path,
        environment: environment
    )
}

private func checkCorrectnessPreflight(
    _ fixture: PreflightFixture,
    environment extraEnvironment: [String: String] = [:]
) throws -> BenchmarkPreflightReport {
    var environment = extraEnvironment
    environment["MLXFAST_REFERENCE_DIR"] = fixture.reference.path
    return try BenchmarkPreflight.checkCorrectnessArtifacts(
        weightsPath: fixture.weights.path,
        goldenPath: fixture.golden.path,
        environment: environment
    )
}

private func makePreflightFixture(
    writeManifest: Bool = true,
    goldenContents: String? = nil,
    omitDenseTensorName: String? = nil,
    expertByteLengthOverride: Int? = nil
) throws -> PreflightFixture {
    let directory = try temporaryDirectory()
    let weights = directory.appendingPathComponent("weights", isDirectory: true)
    let reference = directory.appendingPathComponent("reference", isDirectory: true)
    let experts = weights.appendingPathComponent("experts", isDirectory: true)
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)

    try minimalDeepSeekConfigJSON().write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    var denseTensors = requiredDenseTensorFixtures()
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

    let expertTensors = requiredStackedExpertTensorFixtures()
    let expertShard = "expert-00001.safetensors"
    try writeSafetensors(reference.appendingPathComponent(expertShard), tensors: expertTensors)
    if writeManifest {
        try writeExpertManifest(
            experts.appendingPathComponent("manifest.json"),
            referencePath: reference.path,
            shardName: expertShard,
            tensors: expertTensors,
            expertByteLengthOverride: expertByteLengthOverride
        )
    }

    let golden = directory.appendingPathComponent("correctness_golden.json")
    try (goldenContents ?? validGoldenJSON()).write(to: golden, atomically: true, encoding: .utf8)

    return PreflightFixture(root: directory, weights: weights, reference: reference, golden: golden)
}

private func minimalDeepSeekConfigJSON() -> String {
    """
    {
      "model_type": "deepseek_v4",
      "vocab_size": \(MLXFastConstants.vocabSize),
      "num_hidden_layers": \(MLXFastConstants.numHiddenLayers),
      "n_routed_experts": \(MLXFastConstants.routedExperts),
      "num_experts_per_tok": \(MLXFastConstants.expertsPerToken)
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

private func requiredDenseTensorFixtures() -> [TensorFixture] {
    var tensors: [String: TensorFixture] = [:]
    func add(_ candidates: [String], shape: [Int]) {
        let name = candidates[0]
        tensors[name] = TensorFixture(
            name: name,
            dtype: "U8",
            shape: shape,
            data: Data([UInt8((tensors.count % 251) + 1)])
        )
    }

    let hidden = MLXFastConstants.hiddenSize
    let vocab = MLXFastConstants.vocabSize
    let layers = MLXFastConstants.numHiddenLayers
    let routedExperts = MLXFastConstants.routedExperts
    let moe = MLXFastConstants.moeIntermediateSize
    let heads = MLXFastConstants.attentionHeads
    let headDim = 512
    let qLoraRank = 1_024
    let outputGroups = 8
    let outputLoraRank = 1_024
    let groupedInput = heads * headDim / outputGroups
    let hcMult = 4
    let hcMix = (2 + hcMult) * hcMult
    let indexHeads = 64
    let indexHeadDim = 128
    let compressRatios = DeepSeekConfig.defaultCompressRatios(layerCount: layers)

    add(DeepSeekWeightNames.embedTokens, shape: [vocab, hidden])
    add(DeepSeekWeightNames.finalNorm, shape: [hidden])
    add(DeepSeekWeightNames.hcHeadFn, shape: [hcMult, hcMult * hidden])
    add(DeepSeekWeightNames.hcHeadBase, shape: [hcMult])
    add(DeepSeekWeightNames.hcHeadScale, shape: [1])
    add(DeepSeekWeightNames.lmHead, shape: [vocab, hidden])

    for layerIndex in 0..<layers {
        add(DeepSeekWeightNames.attentionNorm(layerIndex), shape: [hidden])
        add(DeepSeekWeightNames.feedForwardNorm(layerIndex), shape: [hidden])
        for block in [DeepSeekHyperConnectionBlock.attention, .feedForward] {
            add(
                DeepSeekWeightNames.hyperConnection(layerIndex: layerIndex, block: block, component: .fn),
                shape: [hcMix, hcMult * hidden]
            )
            add(
                DeepSeekWeightNames.hyperConnection(layerIndex: layerIndex, block: block, component: .base),
                shape: [hcMix]
            )
            add(
                DeepSeekWeightNames.hyperConnection(layerIndex: layerIndex, block: block, component: .scale),
                shape: [3]
            )
        }

        add(DeepSeekWeightNames.attention(layerIndex, "wq_a.weight"), shape: [qLoraRank, hidden])
        add(DeepSeekWeightNames.attention(layerIndex, "q_norm.weight"), shape: [qLoraRank])
        add(DeepSeekWeightNames.attention(layerIndex, "wq_b.weight"), shape: [heads * headDim, qLoraRank])
        add(DeepSeekWeightNames.attention(layerIndex, "wkv.weight"), shape: [headDim, hidden])
        add(DeepSeekWeightNames.attention(layerIndex, "kv_norm.weight"), shape: [headDim])
        add(
            DeepSeekWeightNames.attention(layerIndex, "wo_a.weight"),
            shape: [outputGroups, outputLoraRank, groupedInput]
        )
        add(
            DeepSeekWeightNames.attention(layerIndex, "wo_b.weight"),
            shape: [hidden, outputGroups * outputLoraRank]
        )

        let ratio = compressRatios[layerIndex]
        if ratio != 0 {
            let outDim = headDim * (ratio == 4 ? 2 : 1)
            add(DeepSeekWeightNames.attention(layerIndex, "compressor.wkv.weight"), shape: [outDim, hidden])
            add(DeepSeekWeightNames.attention(layerIndex, "compressor.wgate.weight"), shape: [outDim, hidden])
            add(DeepSeekWeightNames.attention(layerIndex, "compressor.ape"), shape: [ratio, outDim])
            add(DeepSeekWeightNames.attention(layerIndex, "compressor.norm.weight"), shape: [headDim])

            if ratio == 4 {
                let indexOutDim = indexHeadDim * 2
                add(
                    DeepSeekWeightNames.attention(layerIndex, "indexer.wq_b.weight"),
                    shape: [indexHeads * indexHeadDim, qLoraRank]
                )
                add(
                    DeepSeekWeightNames.attention(layerIndex, "indexer.weights_proj.weight"),
                    shape: [indexHeads, hidden]
                )
                add(
                    DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.wkv.weight"),
                    shape: [indexOutDim, hidden]
                )
                add(
                    DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.wgate.weight"),
                    shape: [indexOutDim, hidden]
                )
                add(
                    DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.ape"),
                    shape: [ratio, indexOutDim]
                )
                add(
                    DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.norm.weight"),
                    shape: [indexHeadDim]
                )
            }
        }

        add(DeepSeekWeightNames.feedForward(layerIndex, "gate.weight"), shape: [routedExperts, hidden])
        add(DeepSeekWeightNames.feedForward(layerIndex, "shared_experts.gate_proj.weight"), shape: [moe, hidden])
        add(DeepSeekWeightNames.feedForward(layerIndex, "shared_experts.up_proj.weight"), shape: [moe, hidden])
        add(DeepSeekWeightNames.feedForward(layerIndex, "shared_experts.down_proj.weight"), shape: [hidden, moe])
    }

    return tensors.values.sorted { $0.name < $1.name }
}

private func requiredStackedExpertTensorFixtures() -> [TensorFixture] {
    var tensors: [TensorFixture] = []
    let routedExperts = MLXFastConstants.routedExperts
    let hidden = MLXFastConstants.hiddenSize
    let moe = MLXFastConstants.moeIntermediateSize
    for layerIndex in 0..<MLXFastConstants.numHiddenLayers {
        tensors.append(
            TensorFixture(
                name: DeepSeekWeightNames.routedExpert(layerIndex: layerIndex, expertIndex: 0, projection: .gate)[0],
                dtype: "U8",
                shape: [routedExperts, moe, hidden],
                data: Data([1])
            )
        )
        tensors.append(
            TensorFixture(
                name: DeepSeekWeightNames.routedExpert(layerIndex: layerIndex, expertIndex: 0, projection: .up)[0],
                dtype: "U8",
                shape: [routedExperts, moe, hidden],
                data: Data([2])
            )
        )
        tensors.append(
            TensorFixture(
                name: DeepSeekWeightNames.routedExpert(layerIndex: layerIndex, expertIndex: 0, projection: .down)[0],
                dtype: "U8",
                shape: [routedExperts, hidden, moe],
                data: Data([3])
            )
        )
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

private func writeExpertManifest(
    _ path: URL,
    referencePath: String,
    shardName: String,
    tensors: [TensorFixture],
    expertByteLengthOverride: Int?
) throws {
    let header = try Safetensors.readHeader(URL(fileURLWithPath: referencePath).appendingPathComponent(shardName))
    let overrideName = tensors.first?.name
    let records = try tensors.map { tensor in
        let info = try #require(header.tensors[tensor.name])
        let byteLength = tensor.name == overrideName ? (expertByteLengthOverride ?? info.byteCount) : info.byteCount
        return """
        {
          "name": "\(tensor.name)",
          "shard": "\(shardName)",
          "dtype": "\(tensor.dtype)",
          "shape": \(arrayJSON(tensor.shape)),
          "data_offsets": [\(info.dataStart), \(info.dataEnd)],
          "byte_offset": \(Int(header.dataBaseOffset) + info.dataStart),
          "byte_length": \(byteLength)
        }
        """
    }.joined(separator: ",\n")
    try """
    {
      "version": 1,
      "source": "safetensors",
      "reference_path": "\(referencePath)",
      "expert_tensors": [
        \(records)
      ]
    }
    """.write(to: path, atomically: true, encoding: .utf8)
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

private func writeExecutableScript(_ path: URL, contents: String) throws -> URL {
    try contents.write(to: path, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: path.path
    )
    return path
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
