import Foundation
import CryptoKit
import Testing
@testable import MLXFastCore

@Test
func checkedInPublicCorrectnessGoldenIsValid() throws {
    let promptPath = MLXFastConstants.defaultPublicCorrectnessPromptPath
    let promptData = try Data(contentsOf: URL(fileURLWithPath: promptPath))
    let promptDigest = SHA256.hash(data: promptData)
        .map { String(format: "%02x", $0) }
        .joined()

    #expect(promptDigest == "98f6a5c49523c891300978437074279c97bb8aa7af18cbf2645983cfbf15e781")
    #expect(promptData.count == 2_735)

    let path = MLXFastConstants.defaultPublicCorrectnessGoldenPath
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let digest = SHA256.hash(data: data)
        .map { String(format: "%02x", $0) }
        .joined()

    #expect(digest == "b9509697c08a2cf3c2943a85f0b76e39c485c441794690fa76835b40a58d7a63")

    let fixture = try loadGoldenFixture(from: path)
    #expect(fixture.sha256 == digest)
    #expect(fixture.benchmark == nil)
    #expect(fixture.correctnessGates == nil)
    #expect(fixture.cases.count == 1)
    #expect(fixture.cases[0].name == "longcopy-gate-english-512")
    #expect(fixture.cases[0].promptTokens.count == MLXFastConstants.correctnessPromptTokens)
    #expect(MLXFastConstants.correctnessSteps == 64)
    #expect(fixture.cases[0].expectedTokens.count == 256)

    let localSubmitPath = MLXFastConstants.defaultPublicLocalSubmitGoldenPath
    let localSubmitData = try Data(contentsOf: URL(fileURLWithPath: localSubmitPath))
    let localSubmitDigest = SHA256.hash(data: localSubmitData)
        .map { String(format: "%02x", $0) }
        .joined()

    #expect(localSubmitDigest == "f49e4c2cbc0d3ceee90195a3a12e1ff082636f8c031587485a9a2c10702b03d2")

    let localSubmitFixture = try loadGoldenFixture(
        from: localSubmitPath,
        requiredSteps: MLXFastConstants.localSubmitBenchmarkDecodeSteps + 1,
        requiredPromptTokens: MLXFastConstants.correctnessPromptTokens
    )
    #expect(localSubmitFixture.sha256 == localSubmitDigest)
    #expect(localSubmitFixture.benchmark == nil)
    #expect(localSubmitFixture.correctnessGates == nil)
    #expect(localSubmitFixture.cases.count == 1)
    #expect(localSubmitFixture.cases[0].name == fixture.cases[0].name)
    #expect(localSubmitFixture.cases[0].promptTokens == fixture.cases[0].promptTokens)
    #expect(localSubmitFixture.cases[0].expectedTokens.count == 1_024)
}

@Test
func goldenModelProvenanceIsStrictAndPinned() throws {
    func documentJSON(repository: String, revision: String, extra: String = "") -> Data {
        Data(
            """
            {
              "version": 1,
              "model_provenance": {
                "repository": "\(repository)",
                "revision": "\(revision)"\(extra)
              },
              "cases": [{
                "name": "provenance-contract",
                "prompt_tokens": \(correctnessPromptJSON()),
                "expected_tokens": \(Array(repeating: 7, count: MLXFastConstants.correctnessSteps))
              }]
            }
            """.utf8
        )
    }

    func load(_ data: Data) throws -> GoldenFixture {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("golden.json")
        try data.write(to: path)
        return try loadGoldenFixture(from: path.path)
    }

    let valid = try load(
        documentJSON(
            repository: MLXFastConstants.referenceModelRepository,
            revision: MLXFastConstants.referenceModelRevision
        )
    )
    #expect(valid.cases.count == 1)

    #expect(throws: MLXFastError.self) {
        _ = try load(
            documentJSON(
                repository: "mlx-community/Laguna-XS-2.1-4bit",
                revision: "c42e0a8f8d504ceacde015a535dcb286d65c8799"
            )
        )
    }
    #expect(throws: MLXFastError.self) {
        _ = try load(
            documentJSON(
                repository: MLXFastConstants.referenceModelRepository,
                revision: MLXFastConstants.referenceModelRevision,
                extra: ", \"unexpected\": true"
            )
        )
    }
}

@Test
func loadGoldenFixtureAcceptsLayeredCorrectnessGates() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "correctness_gates": {
        "anchors": [
          {
            "name": "anchor-0",
            "context_tokens": [1, 2, 3, 4],
            "expected_token": 5,
            "accepted_tokens": [6],
            "max_expected_rank": 2,
            "max_top_logit_delta": 0.001
          }
        ],
        "free_run": [
          {
            "name": "free-run-0",
            "prompt_tokens": \(correctnessPromptJSON(2)),
            "expected_tokens": [8, 9, 10],
            "exact_prefix_tokens": 2
          }
        ],
        "behavior": [
          {
            "name": "behavior-0",
            "prompt_tokens": [3, 4, 5],
            "accepted_token_sequences": [[11, 12], [12, 13]],
            "max_new_tokens": 2
          }
        ]
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    let fixture = try loadGoldenFixture(from: path.path)

    #expect(fixture.cases.count == 1)
    #expect(fixture.correctnessGates?.anchorCases.count == 1)
    #expect(fixture.correctnessGates?.freeRunCases.count == 1)
    #expect(fixture.correctnessGates?.behaviorCases.count == 1)
    #expect(fixture.totalCorrectnessCaseCount == 4)
}

@Test
func loadGoldenFixtureRejectsMalformedLayeredCorrectnessGate() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "correctness_gates": {
        "behavior": [
          {
            "name": "behavior-0",
            "prompt_tokens": \(correctnessPromptJSON(3)),
            "accepted_token_sequences": [[11, 12]],
            "max_new_tokens": 1
          }
        ]
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenFixture(from: path.path)
    }
}

@Test
func loadGoldenFixtureRejectsDuplicateLayeredCorrectnessNames() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "duplicate",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "correctness_gates": {
        "anchors": [
          {
            "name": "duplicate",
            "context_tokens": [1, 2, 3],
            "expected_token": 4
          }
        ]
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenFixture(from: path.path)
    }
}

@Test
func loadGoldenFixtureRejectsNoopAnchorDelta() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "correctness_gates": {
        "anchors": [
          {
            "name": "anchor-0",
            "context_tokens": [1, 2, 3],
            "expected_token": 4,
            "max_top_logit_delta": 0.001
          }
        ]
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenFixture(from: path.path)
    }
}

@Test
func loadGoldenFixtureRejectsUnknownCorrectnessGateKey() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "correctness_gates": {
        "free_runs": [
          {
            "name": "typo",
            "prompt_tokens": \(correctnessPromptJSON(2)),
            "expected_tokens": [8]
          }
        ]
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenFixture(from: path.path)
    }
}

@Test
func loadGoldenFixtureRejectsEmptyCorrectnessGateSection() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "correctness_gates": {
        "anchors": []
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenFixture(from: path.path)
    }
}

@Test
func goldenSequenceMatcherChecksExactPrefixes() {
    let pass = GoldenSequenceMatcher.firstPrefixMismatch(
        expected: [1, 2, 3],
        actual: [1, 2, 9],
        prefixTokens: 2
    )
    #expect(pass.passed)

    let fail = GoldenSequenceMatcher.firstPrefixMismatch(
        expected: [1, 2, 3],
        actual: [1, 8, 3],
        prefixTokens: 3
    )
    #expect(!fail.passed)
    #expect(fail.step == 1)
    #expect(fail.expectedToken == 2)
    #expect(fail.actualToken == 8)
}

@Test
func goldenSequenceMatcherAcceptsShortBehaviorPrefixes() {
    let pass = GoldenSequenceMatcher.matchesAnyAcceptedPrefix(
        acceptedSequences: [[101], [202, 203]],
        actual: [101, 999]
    )
    #expect(pass.passed)

    let fail = GoldenSequenceMatcher.matchesAnyAcceptedPrefix(
        acceptedSequences: [[101], [202, 203]],
        actual: [202, 999]
    )
    #expect(!fail.passed)
    #expect(fail.step == 1)
    #expect(fail.expectedToken == 203)
    #expect(fail.actualToken == 999)
}

@Test
func goldenSequenceMatcherAcceptsExactAnswerSequences() {
    let pass = GoldenSequenceMatcher.matchesAnyExactSequence(
        acceptedSequences: [[10, 11], [20, 21]],
        actual: [20, 21]
    )
    #expect(pass.passed)

    let fail = GoldenSequenceMatcher.matchesAnyExactSequence(
        acceptedSequences: [[10, 11], [20, 21]],
        actual: [20, 22, 99]
    )
    #expect(!fail.passed)
    #expect(fail.step == 1 || fail.step == 2)
}

@Test
func loadGoldenCasesAcceptsValidFixture() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    let cases = try loadGoldenCases(from: path.path)

    #expect(cases.count == 1)
    #expect(cases[0].name == "hidden-0")
    #expect(cases[0].promptTokens == correctnessPrompt())
    #expect(cases[0].expectedTokens.count == MLXFastConstants.correctnessSteps)

    let fixture = try loadGoldenFixture(from: path.path)
    let digest = SHA256.hash(data: try Data(contentsOf: path))
    let expectedHash = digest.map { String(format: "%02x", $0) }.joined()
    #expect(fixture.cases == cases)
    #expect(fixture.sha256 == expectedHash)
}

@Test
func loadGoldenFixtureAcceptsBenchmarkOracle() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let prefill = Array(repeating: 1, count: MLXFastConstants.benchmarkPrefillPromptTokens)
    let seed = Array(repeating: 2, count: MLXFastConstants.benchmarkDecodeSeedTokens)
    let decode = Array(repeating: 3, count: MLXFastConstants.benchmarkDecodeSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "benchmark": {
        "prefill_prompt_tokens": \(prefill),
        "expected_prefill_token": 4,
        "decode_seed_tokens": \(seed),
        "expected_decode_seed_token": 5,
        "expected_decode_tokens": \(decode)
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    let fixture = try loadGoldenFixture(from: path.path)

    #expect(fixture.benchmark?.prefillPromptTokens == prefill)
    #expect(fixture.benchmark?.expectedPrefillToken == 4)
    #expect(fixture.benchmark?.decodeSeedTokens == seed)
    #expect(fixture.benchmark?.expectedDecodeSeedToken == 5)
    #expect(fixture.benchmark?.expectedDecodeTokens == decode)
    // No per-prompt baselines carried: scoring resolves to the calibrated constants.
    #expect(fixture.benchmark?.baselinePrefillSecondsPerToken == nil)
    #expect(fixture.benchmark?.baselineDecodeSecondsPerToken == nil)
    #expect(
        fixture.benchmark?.resolvedBaselinePrefillSecondsPerToken
            == MLXFastConstants.officialBaselinePrefillSecondsPerToken
    )
    #expect(
        fixture.benchmark?.resolvedBaselineDecodeSecondsPerToken
            == MLXFastConstants.officialBaselineDecodeSecondsPerToken
    )
}

private func benchmarkOracleGoldenJSON(baselineFieldsJSON: String) -> String {
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let prefill = Array(repeating: 1, count: MLXFastConstants.benchmarkPrefillPromptTokens)
    let seed = Array(repeating: 2, count: MLXFastConstants.benchmarkDecodeSeedTokens)
    let decode = Array(repeating: 3, count: MLXFastConstants.benchmarkDecodeSteps)
    return """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "benchmark": {
        "prefill_prompt_tokens": \(prefill),
        "expected_prefill_token": 4,
        "decode_seed_tokens": \(seed),
        "expected_decode_seed_token": 5,
        "expected_decode_tokens": \(decode)\(baselineFieldsJSON)
      }
    }
    """
}

@Test
func loadGoldenFixtureAcceptsPerPromptBenchmarkBaselines() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let json = benchmarkOracleGoldenJSON(baselineFieldsJSON: """
    ,
        "baseline_prefill_seconds_per_token": 0.25,
        "baseline_decode_seconds_per_token": 4.5
    """)
    try json.write(to: path, atomically: true, encoding: .utf8)

    let fixture = try loadGoldenFixture(from: path.path)

    #expect(fixture.benchmark?.baselinePrefillSecondsPerToken == 0.25)
    #expect(fixture.benchmark?.baselineDecodeSecondsPerToken == 4.5)
    // Carried baselines win over the calibrated constants.
    #expect(fixture.benchmark?.resolvedBaselinePrefillSecondsPerToken == 0.25)
    #expect(fixture.benchmark?.resolvedBaselineDecodeSecondsPerToken == 4.5)
}

@Test
func loadGoldenFixtureRejectsUnknownBenchmarkKeys() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    // A typo'd scoring-critical key must fail loudly. Without strict nested
    // key validation, JSONDecoder drops the unknown key, both baselines decode
    // as nil, and the run silently scores against the calibrated constants
    // instead of the intended per-prompt baseline.
    let json = benchmarkOracleGoldenJSON(baselineFieldsJSON: """
    ,
        "baseline_decode_second_per_token": 4.5
    """)
    try json.write(to: path, atomically: true, encoding: .utf8)

    do {
        _ = try loadGoldenFixture(from: path.path)
        Issue.record("expected unknown benchmark key to be rejected")
    } catch let MLXFastError.invalidInput(message) {
        #expect(message.contains("unknown key"))
        #expect(message.contains("baseline_decode_second_per_token"))
    } catch {
        Issue.record("expected MLXFastError.invalidInput, got \(error)")
    }
}

@Test
func loadGoldenFixtureRejectsNullBenchmarkObject() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "benchmark": null
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    do {
        _ = try loadGoldenFixture(from: path.path)
        Issue.record("expected null benchmark object to be rejected")
    } catch let MLXFastError.invalidInput(message) {
        #expect(message.contains("benchmark must not be null"))
    } catch {
        Issue.record("expected MLXFastError.invalidInput, got \(error)")
    }
}

@Test
func loadGoldenFixtureRejectsHalfCalibratedBenchmarkBaselines() throws {
    let directory = try temporaryDirectory()
    for lonelyField in [
        "\"baseline_prefill_seconds_per_token\": 0.25",
        "\"baseline_decode_seconds_per_token\": 4.5",
    ] {
        let path = directory.appendingPathComponent("golden-\(UUID().uuidString).json")
        let json = benchmarkOracleGoldenJSON(baselineFieldsJSON: ",\n    \(lonelyField)")
        try json.write(to: path, atomically: true, encoding: .utf8)

        do {
            _ = try loadGoldenFixture(from: path.path)
            Issue.record("expected half-calibrated baseline pair to be rejected")
        } catch let MLXFastError.invalidInput(message) {
            #expect(message.contains("must be provided together"))
        } catch {
            Issue.record("expected MLXFastError.invalidInput, got \(error)")
        }
    }
}

@Test
func loadGoldenFixtureRejectsNonPositiveBenchmarkBaselines() throws {
    let directory = try temporaryDirectory()
    for badPair in [
        "\"baseline_prefill_seconds_per_token\": 0, \"baseline_decode_seconds_per_token\": 4.5",
        "\"baseline_prefill_seconds_per_token\": 0.25, \"baseline_decode_seconds_per_token\": -1.0",
    ] {
        let path = directory.appendingPathComponent("golden-\(UUID().uuidString).json")
        let json = benchmarkOracleGoldenJSON(baselineFieldsJSON: ",\n    \(badPair)")
        try json.write(to: path, atomically: true, encoding: .utf8)

        do {
            _ = try loadGoldenFixture(from: path.path)
            Issue.record("expected non-positive baseline to be rejected")
        } catch let MLXFastError.invalidInput(message) {
            #expect(message.contains("must be finite and positive"))
        } catch {
            Issue.record("expected MLXFastError.invalidInput, got \(error)")
        }
    }
}

@Test
func loadGoldenFixtureRejectsMalformedBenchmarkOracle() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "benchmark": {
        "prefill_prompt_tokens": [1],
        "expected_prefill_token": 4,
        "decode_seed_tokens": [2],
        "expected_decode_seed_token": 5,
        "expected_decode_tokens": [3]
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenFixture(from: path.path)
    }
}

@Test
func loadGoldenFixtureStaleBenchmarkOracleErrorMentionsPrecomputedFixture() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = arrayJSON(Array(repeating: 9, count: MLXFastConstants.correctnessSteps))
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "case-a",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "benchmark": {
        "prefill_prompt_tokens": \(arrayJSON(Array(repeating: 1, count: MLXFastConstants.benchmarkPrefillPromptTokens))),
        "expected_prefill_token": 2,
        "decode_seed_tokens": \(arrayJSON(Array(repeating: 3, count: 32))),
        "expected_decode_seed_token": 4,
        "expected_decode_tokens": \(arrayJSON(Array(repeating: 5, count: MLXFastConstants.benchmarkDecodeSteps)))
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    do {
        _ = try loadGoldenFixture(from: path.path)
        Issue.record("expected stale benchmark oracle error")
    } catch let MLXFastError.invalidInput(message) {
        #expect(message.contains("Replace stale local goldens with an updated precomputed golden fixture"))
    } catch {
        Issue.record("expected MLXFastError.invalidInput, got \(error)")
    }
}

@Test
func benchmarkOutputValidatorReportsTokenMismatches() {
    let oracle = BenchmarkGolden(
        prefillPromptTokens: Array(repeating: 1, count: MLXFastConstants.benchmarkPrefillPromptTokens),
        expectedPrefillToken: 10,
        decodeSeedTokens: Array(repeating: 2, count: MLXFastConstants.benchmarkDecodeSeedTokens),
        expectedDecodeSeedToken: 20,
        expectedDecodeTokens: [30, 31, 32]
    )

    let prefill = BenchmarkOutputValidator.comparePrefillToken(
        expected: oracle,
        actualToken: 11
    )
    #expect(!prefill.passed)
    #expect(prefill.expectedToken == 10)
    #expect(prefill.actualToken == 11)

    let seed = BenchmarkOutputValidator.compareDecodeSeedToken(
        expected: oracle,
        actualToken: 21
    )
    #expect(!seed.passed)
    #expect(seed.expectedToken == 20)
    #expect(seed.actualToken == 21)

    let decode = BenchmarkOutputValidator.compareDecodeTokens(
        expected: oracle,
        actualTokens: [30, 99, 32]
    )
    #expect(!decode.passed)
    #expect(decode.step == 1)
    #expect(decode.expectedToken == 31)
    #expect(decode.actualToken == 99)
}

@Test
func loadGoldenCasesRejectsOutOfRangeToken() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    var prompt = correctnessPrompt()
    prompt[0] = MLXFastConstants.vocabSize
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "bad",
          "prompt_tokens": \(arrayJSON(prompt)),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsMissingVersion() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "cases": [
        {
          "name": "missing-version",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsDuplicateCaseNames() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "duplicate",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        },
        {
          "name": "duplicate",
          "prompt_tokens": \(correctnessPromptJSON(2)),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsNamesWithSurroundingWhitespace() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": " ambiguous ",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsNamesWithControlCharacters() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "bad\\nname",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsWrongExpectedTokenCount() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps - 1)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "wrong-count",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsWrongPromptTokenCount() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "wrong-prompt-count",
          "prompt_tokens": [1],
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsNonPositiveRequiredSteps() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = [7]
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "bad-steps",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path, requiredSteps: 0)
    }
}

@Test
func loadGoldenCasesRejectsNonPositiveRequiredPromptTokens() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "bad-prompt-steps",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path, requiredPromptTokens: 0)
    }
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func correctnessPrompt(_ token: Int = 1) -> [Int] {
    Array(repeating: token, count: MLXFastConstants.correctnessPromptTokens)
}

private func correctnessPromptJSON(_ token: Int = 1) -> String {
    arrayJSON(correctnessPrompt(token))
}

private func arrayJSON(_ values: [Int]) -> String {
    "[\(values.map(String.init).joined(separator: ","))]"
}
