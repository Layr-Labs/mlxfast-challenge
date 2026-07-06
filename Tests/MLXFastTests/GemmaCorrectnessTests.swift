import Foundation
import CryptoKit
import MLX
import MLXFastCore
@testable import MLXFastModel
@testable import MLXFastHarness
import Testing

@Test
func deepSeekCorrectnessComparesExpectedTokenSequences() {
    let pass = GemmaCorrectness.compareTokens(
        expected: [4, 5, 6],
        actual: [4, 5, 6],
        steps: 3
    )
    #expect(pass.passed)
    #expect(pass.checkedSteps == 3)
    #expect(pass.firstFailingStep == nil)

    let fail = GemmaCorrectness.compareTokens(
        expected: [4, 5, 6],
        actual: [4, 9, 6],
        steps: 3
    )
    #expect(!fail.passed)
    #expect(fail.checkedSteps == 2)
    #expect(fail.firstFailingStep == 1)
    #expect(fail.expectedToken == 5)
    #expect(fail.actualToken == 9)

    let short = GemmaCorrectness.compareTokens(
        expected: [4, 5, 6],
        actual: [4],
        steps: 3
    )
    #expect(!short.passed)
    #expect(short.checkedSteps == 2)
    #expect(short.firstFailingStep == 1)
    #expect(short.expectedToken == 5)
    #expect(short.actualToken == nil)

    let expectedShort = GemmaCorrectness.compareTokens(
        expected: [4],
        actual: [4, 5],
        steps: 2
    )
    #expect(!expectedShort.passed)
    #expect(expectedShort.checkedSteps == 2)
    #expect(expectedShort.firstFailingStep == 1)
    #expect(expectedShort.expectedToken == nil)
    #expect(expectedShort.actualToken == 5)

    let bothShort = GemmaCorrectness.compareTokens(
        expected: [4],
        actual: [4],
        steps: 2
    )
    #expect(!bothShort.passed)
    #expect(bothShort.checkedSteps == 2)
    #expect(bothShort.firstFailingStep == 1)
    #expect(bothShort.expectedToken == nil)
    #expect(bothShort.actualToken == nil)
}

@Test
func deepSeekCorrectnessGeneratesGreedyTokensWithGrowingContext() throws {
    var contexts: [[Int]] = []
    let generated = try GemmaCorrectness.generateGreedyNoCache(
        promptTokens: [10, 11],
        steps: 3
    ) { context in
        contexts.append(context)
        return context.count
    }

    #expect(generated == [2, 3, 4])
    #expect(contexts == [[10, 11], [10, 11, 2], [10, 11, 2, 3]])
}

@Test
func deepSeekCorrectnessTeacherForcedUsesGoldenPrefix() throws {
    var contexts: [[Int]] = []
    let expected = [20, 21, 22]
    let comparison = try GemmaCorrectness.compareTeacherForcedNoCache(
        promptTokens: [10, 11],
        expectedTokens: expected,
        steps: expected.count
    ) { context in
        contexts.append(context)
        return expected[context.count - 2]
    }

    #expect(comparison.passed)
    #expect(comparison.checkedSteps == 3)
    #expect(contexts == [[10, 11], [10, 11, 20], [10, 11, 20, 21]])
}

@Test
func compareTeacherForcedNoCacheSeedsContextWithGoldenPrefixAtStartStep() throws {
    var contexts: [[Int]] = []
    let expected = [20, 21, 22, 23, 24]
    let comparison = try GemmaCorrectness.compareTeacherForcedNoCache(
        promptTokens: [10, 11],
        expectedTokens: expected,
        steps: 2,
        startStep: 3
    ) { context in
        contexts.append(context)
        return expected[context.count - 2]
    }

    #expect(comparison.passed)
    #expect(comparison.checkedSteps == 2)
    // The seed is one batch call over the prompt plus the golden prefix through
    // position 2 (indices 0..<3) -- never single-token-stepped through positions
    // 0-2, matching how a machine checking a later slice fast-forwards past
    // positions it isn't responsible for verifying.
    #expect(contexts == [[10, 11, 20, 21, 22], [10, 11, 20, 21, 22, 23]])
}

// Proves the core soundness claim behind splitting correctness checking across
// machines: because teacher forcing always feeds the known golden token as input
// (never the model's own prior output), a chunk [startStep, startStep + steps)
// checked in isolation must reach the exact same per-position verdict as that same
// range would reach inside one continuous run -- regardless of what happens in
// positions outside its own assigned range.
@Test
func compareTeacherForcedNoCacheChunksAreIndependentOfEachOther() throws {
    let expected = [100, 101, 102, 103, 104, 105]
    // A "model" that is correct everywhere except position 4, deterministic in
    // terms of context length alone so every chunk sees the same simulated model.
    func simulatedModel(_ context: [Int]) -> Int {
        let position = context.count - 2
        return position == 4 ? 999 : expected[position]
    }

    let full = try GemmaCorrectness.compareTeacherForcedNoCache(
        promptTokens: [1, 2],
        expectedTokens: expected,
        steps: expected.count
    ) { simulatedModel($0) }
    #expect(!full.passed)
    #expect(full.firstFailingStep == 4)
    #expect(full.expectedToken == 104)
    #expect(full.actualToken == 999)

    // A machine assigned [4, 6) never sees positions 0-3 checked at all, yet must
    // independently land on the exact same failure at the same absolute position.
    let chunkContainingFailure = try GemmaCorrectness.compareTeacherForcedNoCache(
        promptTokens: [1, 2],
        expectedTokens: expected,
        steps: 2,
        startStep: 4
    ) { simulatedModel($0) }
    #expect(!chunkContainingFailure.passed)
    #expect(chunkContainingFailure.firstFailingStep == 4)
    #expect(chunkContainingFailure.expectedToken == 104)
    #expect(chunkContainingFailure.actualToken == 999)
    #expect(chunkContainingFailure.checkedSteps == 1)

    // A machine assigned [0, 4), entirely before the failure, passes cleanly on
    // its own -- it has no way to know a later chunk will fail.
    let chunkBeforeFailure = try GemmaCorrectness.compareTeacherForcedNoCache(
        promptTokens: [1, 2],
        expectedTokens: expected,
        steps: 4,
        startStep: 0
    ) { simulatedModel($0) }
    #expect(chunkBeforeFailure.passed)
    #expect(chunkBeforeFailure.checkedSteps == 4)

    // A machine assigned [5, 6), entirely after the failure, also passes cleanly
    // on its own -- proving chunk verdicts are genuine independent checks against
    // the golden prefix, not an early-exit replay that would have inherited the
    // upstream failure.
    let chunkAfterFailure = try GemmaCorrectness.compareTeacherForcedNoCache(
        promptTokens: [1, 2],
        expectedTokens: expected,
        steps: 1,
        startStep: 5
    ) { simulatedModel($0) }
    #expect(chunkAfterFailure.passed)
    #expect(chunkAfterFailure.checkedSteps == 1)
}

@Test
func correctnessReportEncodesStableFailureFields() throws {
    let report = CorrectnessReport(
        passed: true,
        checkedSteps: MLXFastConstants.correctnessSteps,
        caseCount: 1,
        expertCacheHits: 4,
        expertCacheMisses: 6,
        expertCacheEvictions: 2,
        expertBytesRead: 2048,
        expertReadSeconds: 0.5,
        expertPeakCachedTensors: 8,
        expertHitRate: 0.4,
        firstFailingCase: nil,
        firstFailingStep: nil,
        expectedToken: nil,
        actualToken: nil,
        goldenHash: "abc123",
        error: ""
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(report)
    let raw = String(decoding: data, as: UTF8.self)

    #expect(raw.contains("\"first_failing_case\" : null"))
    #expect(raw.contains("\"first_failing_step\" : null"))
    #expect(raw.contains("\"expected_token\" : null"))
    #expect(raw.contains("\"actual_token\" : null"))
    #expect(raw.contains("\"checked_steps\" : \(MLXFastConstants.correctnessSteps)"))
    #expect(raw.contains("\"case_count\" : 1"))
    #expect(raw.contains("\"expert_cache_hits\" : 4"))
    #expect(raw.contains("\"expert_cache_misses\" : 6"))
    #expect(raw.contains("\"expert_cache_evictions\" : 2"))
    #expect(raw.contains("\"expert_bytes_read\" : 2048"))
    #expect(raw.contains("\"expert_read_seconds\" : 0.5"))
    #expect(raw.contains("\"expert_peak_cached_tensors\" : 8"))
    #expect(raw.contains("\"expert_hit_rate\" : 0.4"))
    #expect(raw.contains("\"golden_hash\" : \"abc123\""))
    #expect(report.expertStreamingStats.cacheHits == 4)
    #expect(report.expertStreamingStats.cacheMisses == 6)
    #expect(report.expertStreamingStats.hitRate == 0.4)
}

@Test
func gemmaRuntimeCorrectnessReportsMissingArtifacts() throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }

    let report = try GemmaRuntime.runCorrectness(
        CorrectnessOptions(
            weightsPath: directory.appendingPathComponent("missing-weights").path,
            goldenPath: directory.appendingPathComponent("missing-golden.json").path
        )
    )

    #expect(!report.passed)
    #expect(report.checkedSteps == 0)
    #expect(report.firstFailingCase == nil)
    #expect(report.error.contains("correctness golden file"))
}

@Test
func gemmaRuntimeCorrectnessReportsGoldenMetadataWhenWeightsAreMissing() throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }

    let goldenPath = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "valid-golden",
          "prompt_tokens": \(arrayJSON(Array(repeating: 1, count: MLXFastConstants.correctnessPromptTokens))),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: goldenPath, atomically: true, encoding: .utf8)

    let report = try GemmaRuntime.runCorrectness(
        CorrectnessOptions(
            weightsPath: directory.appendingPathComponent("missing-weights").path,
            goldenPath: goldenPath.path
        )
    )

    let digest = SHA256.hash(data: try Data(contentsOf: goldenPath))
    let expectedHash = digest.map { String(format: "%02x", $0) }.joined()
    #expect(!report.passed)
    #expect(report.checkedSteps == 0)
    #expect(report.caseCount == 1)
    #expect(report.goldenHash == expectedHash)
    #expect(report.firstFailingCase == nil)
}

@Test
func deepSeekCorrectnessSelectsGreedyTokenWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    #expect(try GemmaCorrectness.greedyToken(
        from: MLXArray([Float(0.1), 2.0, 1.0], [3])
    ) == 1)
    #expect(try GemmaCorrectness.greedyToken(
        from: MLXArray([Float(1), 2, 3, 2], [2, 2])
    ) == 0)
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func arrayJSON(_ values: [Int]) -> String {
    "[\(values.map(String.init).joined(separator: ","))]"
}
