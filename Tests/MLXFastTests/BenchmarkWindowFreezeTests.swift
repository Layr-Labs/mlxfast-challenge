import Foundation
@testable import MLXFastCore
@testable import MLXFastModel
import Testing

// Guards the frozen timed-benchmark window (see docs/benchmark-window-freeze.md).
// The official prefill/decode baselines are measured on the Blacksmith runner at
// real cost, so any change to the charged work silently invalidates them. These
// tests are deliberately annoying to change: editing a window constant or the
// decode/prefill charged-forward structure fails CI until the baseline is
// re-measured and the freeze doc is updated in the same change.

private func packageFile(_ path: String) throws -> String {
    try String(contentsOfFile: path, encoding: .utf8)
}

// Returns the trimmed right-hand side of `let <name> = <literal>` in Swift source.
// Strips a trailing line comment so a future `let x = 3.6 // note` still extracts
// just `3.6` and matches the value quoted in the freeze doc.
private func swiftConstantLiteral(_ source: String, name: String) throws -> String {
    let marker = "let \(name) = "
    let start = try #require(
        source.range(of: marker),
        "expected constant \(name) in source"
    )
    let rest = source[start.upperBound...]
    let lineEnd = rest.firstIndex(of: "\n") ?? rest.endIndex
    var literal = String(rest[..<lineEnd])
    if let comment = literal.range(of: "//") {
        literal = String(literal[..<comment.lowerBound])
    }
    return literal.trimmingCharacters(in: .whitespaces)
}

private func slice(_ source: String, from startMarker: String, to endMarker: String) throws -> String {
    let start = try #require(source.range(of: startMarker), "expected \(startMarker)")
    let end = try #require(
        source.range(of: endMarker, range: start.upperBound..<source.endIndex),
        "expected \(endMarker) after \(startMarker)"
    )
    return String(source[start.lowerBound..<end.lowerBound])
}

@Test
func benchmarkWindowConstantsAreFrozen() {
    // Prefill axis: one cold, validated 512-token forward, no warmup.
    #expect(MLXFastConstants.benchmarkPrefillPromptTokens == 512)
    #expect(MLXFastConstants.benchmarkPrefillWarmupRuns == 0)
    #expect(MLXFastConstants.benchmarkPrefillTimedRuns == 1)
    // Decode axis: 512-token seed prefill charged to decode, then 128 validated steps.
    #expect(MLXFastConstants.benchmarkDecodeSeedTokens == 512)
    #expect(MLXFastConstants.benchmarkDecodeSteps == 128)
    // Ranking contract: geometric weights and floors the baseline maps through.
    #expect(MLXFastConstants.scoreDecodeWeight == 0.75)
    #expect(MLXFastConstants.scorePrefillWeight == 0.25)
    #expect(MLXFastConstants.scoreDecodeSpeedupFloor == 0.95)
    #expect(MLXFastConstants.scorePrefillSpeedupFloor == 0.95)
    #expect(MLXFastConstants.prefillBandUpTolerance == 0.02)
    #expect(MLXFastConstants.prefillBandDownTolerance == 0.05)
}

@Test
func benchmarkWindowFreezeDocMatchesConstants() throws {
    let doc = try packageFile("docs/benchmark-window-freeze.md")
    let constants = try packageFile("Sources/MLXFastCore/Constants.swift")

    // The doc must quote the current window knobs, so a constant edit forces a
    // doc edit in the same change (or this fails).
    #expect(doc.contains("benchmarkPrefillPromptTokens = \(MLXFastConstants.benchmarkPrefillPromptTokens)"))
    #expect(doc.contains("benchmarkPrefillWarmupRuns = \(MLXFastConstants.benchmarkPrefillWarmupRuns)"))
    #expect(doc.contains("benchmarkPrefillTimedRuns = \(MLXFastConstants.benchmarkPrefillTimedRuns)"))
    #expect(doc.contains("benchmarkDecodeSeedTokens = \(MLXFastConstants.benchmarkDecodeSeedTokens)"))
    #expect(doc.contains("benchmarkDecodeSteps = \(MLXFastConstants.benchmarkDecodeSteps)"))
    #expect(doc.contains("prefillBandUpTolerance = \(MLXFastConstants.prefillBandUpTolerance)"))
    #expect(doc.contains("prefillBandDownTolerance = \(MLXFastConstants.prefillBandDownTolerance)"))

    // The doc must quote the exact calibrated baseline literals from Constants,
    // so a re-baseline cannot land while the freeze doc still shows the old one.
    let decodeBaseline = try swiftConstantLiteral(constants, name: "officialBaselineDecodeSecondsPerToken")
    let prefillBaseline = try swiftConstantLiteral(constants, name: "officialBaselinePrefillSecondsPerToken")
    #expect(doc.contains(decodeBaseline), "freeze doc must quote officialBaselineDecodeSecondsPerToken=\(decodeBaseline)")
    #expect(doc.contains(prefillBaseline), "freeze doc must quote officialBaselinePrefillSecondsPerToken=\(prefillBaseline)")
}

@Test
func timedDecodeChargesOneValidatedSeedForward() throws {
    let worker = try packageFile("Sources/MLXFastHarness/GemmaRuntimeWorker.swift")
    let decodeBegin = try slice(worker, from: "case \"decode_begin\":", to: "case \"decode_step\":")
    // The one-forward/no-warmup property is also guarded by
    // BenchmarkScriptTests.decodeMeasurementRunsSingleUnmemoizableSeedForward;
    // this test intentionally supersets it (it additionally pins parent-timed,
    // oracle-validated measurement) so the freeze guard stands on its own.
    // Exactly one whole-prompt forward, and no warmup pass to memoize against it.
    #expect(decodeBegin.components(separatedBy: "Gemma4Model.logits(").count - 1 == 1)
    #expect(!decodeBegin.contains("warmupCache"))
    #expect(!decodeBegin.contains("warmupLogits"))

    // The decode phase is parent-timed and validated; worker-reported seconds
    // must not be the scored value, and both the seed and the steps are checked.
    let benchmark = try packageFile("Sources/MLXFastHarness/GemmaRuntimeBenchmark.swift")
    let measureWorkerDecode = try slice(
        benchmark,
        from: "static func measureWorkerDecode(",
        to: "static let bandwidthSource"
    )
    #expect(measureWorkerDecode.contains("secondsSince(decodePhaseStart)"))
    #expect(measureWorkerDecode.contains("compareDecodeSeedToken"))
    #expect(measureWorkerDecode.contains("compareDecodeTokens"))
    #expect(!measureWorkerDecode.contains("response.seconds"))
}

@Test
func timedPrefillChargesOneValidatedColdForward() throws {
    let benchmark = try packageFile("Sources/MLXFastHarness/GemmaRuntimeBenchmark.swift")
    let measureWorkerPrefill = try slice(
        benchmark,
        from: "static func measureWorkerPrefillSecondsPerToken(",
        to: "static func measureDecode("
    )
    // Parent-measured wall time around the worker request; validated against the
    // prefill oracle; worker-reported seconds are never the scored value.
    #expect(measureWorkerPrefill.contains("DispatchTime.now().uptimeNanoseconds"))
    #expect(measureWorkerPrefill.contains("secondsSince(prefillStart)"))
    #expect(measureWorkerPrefill.contains("comparePrefillToken"))
    #expect(!measureWorkerPrefill.contains("response.seconds"))
}

@Test
func scoredBaselinesResolveFromGoldenWithConstantsFallback() throws {
    // Prompt-pool rotation: the golden oracle may carry per-prompt baselines
    // (both axes together, validated positive at load). The scored speedups and
    // floors must use the golden-resolved values, with the calibrated constants
    // as the fallback for goldens that carry none -- so a pool prompt of
    // different intrinsic difficulty ranks on its own calibration instead of
    // the default prompt's.
    let golden = try packageFile("Sources/MLXFastCore/Golden.swift")
    #expect(golden.contains("baselinePrefillSecondsPerToken ?? MLXFastConstants.officialBaselinePrefillSecondsPerToken"))
    #expect(golden.contains("baselineDecodeSecondsPerToken ?? MLXFastConstants.officialBaselineDecodeSecondsPerToken"))
    #expect(golden.contains("must be provided together"))

    let benchmark = try packageFile("Sources/MLXFastHarness/GemmaRuntimeBenchmark.swift")
    // Both benchmark paths adopt the golden's resolved baselines...
    #expect(benchmark.components(separatedBy: "benchmarkGolden.resolvedBaselinePrefillSecondsPerToken").count - 1 == 2)
    #expect(benchmark.components(separatedBy: "benchmarkGolden.resolvedBaselineDecodeSecondsPerToken").count - 1 == 2)
    // ...and every scored speedup uses the resolved values, never the raw constants.
    #expect(benchmark.contains("baselineSecondsPerToken: baselineDecodeSecondsPerToken"))
    #expect(benchmark.contains("baselineSecondsPerToken: baselinePrefillSecondsPerToken"))
    #expect(!benchmark.contains("baselineSecondsPerToken: MLXFastConstants.officialBaselineDecodeSecondsPerToken"))
    #expect(!benchmark.contains("baselineSecondsPerToken: MLXFastConstants.officialBaselinePrefillSecondsPerToken"))
}

@Test
func officialTimingMachineMeasuresPairedBaseline() throws {
    // The ranking contract on official runs: speedups and floors are computed
    // against a baseline measured in the SAME session on the SAME VM (paired
    // baseline), cancelling the ~4-10% fleet/time drift that made fixed-
    // constant floor verdicts near the threshold a coin flip. See the
    // paired-baseline section of docs/benchmark-window-freeze.md.
    let workflow = try packageFile(".github/workflows/benchmark-timing-or-gates.yml")
    let constants = try packageFile("Sources/MLXFastCore/Constants.swift")
    let doc = try packageFile("docs/benchmark-window-freeze.md")

    // Timing machine only, ordered before the candidate's Benchmark step, on a
    // pinned trusted ref that submissions cannot repoint.
    let checkout = try #require(workflow.range(of: "- name: Checkout paired baseline reference"))
    let measure = try #require(workflow.range(of: "- name: Measure paired baseline"))
    let candidate = try #require(workflow.range(of: "- name: Benchmark"))
    #expect(checkout.lowerBound < measure.lowerBound)
    #expect(measure.lowerBound < candidate.lowerBound)
    let pairedBody = String(workflow[checkout.lowerBound..<candidate.lowerBound])
    #expect(pairedBody.components(separatedBy: "if: ${{ inputs.mode == 'timing' }}").count - 1 == 2)
    #expect(pairedBody.contains("ref: eff7e7f2c85a5a6cef11110442ba4624a6ab3986"))
    #expect(pairedBody.contains("persist-credentials: false"))

    // The measured pair reaches the candidate only through the env overrides,
    // which the harness validates fail-closed and strips from the worker env.
    #expect(pairedBody.contains("MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN="))
    #expect(pairedBody.contains("MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN="))
    // Baseline floor failures are tolerated (they are measured against the
    // baseline's own constants); anything else fails the run, and a sanity
    // band anchored to the calibrated constants rejects pathological sessions.
    #expect(pairedBody.contains("startswith(\"performance floor failed\")"))
    #expect(pairedBody.contains("$prefill_ratio >= 0.66 and $prefill_ratio <= 1.5"))
    let sanityPrefill = try #require(
        pairedBody.range(of: "MLXFAST_PAIRED_SANITY_PREFILL: \"").map { range in
            String(pairedBody[range.upperBound...].prefix(while: { $0 != "\"" }))
        }
    )
    let sanityDecode = try #require(
        pairedBody.range(of: "MLXFAST_PAIRED_SANITY_DECODE: \"").map { range in
            String(pairedBody[range.upperBound...].prefix(while: { $0 != "\"" }))
        }
    )
    #expect(sanityPrefill == (try swiftConstantLiteral(constants, name: "officialBaselinePrefillSecondsPerToken")))
    #expect(sanityDecode == (try swiftConstantLiteral(constants, name: "officialBaselineDecodeSecondsPerToken")))

    // Harness side: paired override outranks golden-carried baselines, which
    // outrank constants, and the override keys never reach the worker.
    let benchmark = try packageFile("Sources/MLXFastHarness/GemmaRuntimeBenchmark.swift")
    #expect(benchmark.components(separatedBy: "PairedBaselineOverride.fromEnvironment()").count - 1 == 2)
    #expect(benchmark.contains("pairedBaseline?.prefillSecondsPerToken\n                ?? benchmarkGolden.resolvedBaselinePrefillSecondsPerToken"))
    #expect(benchmark.contains("pairedBaseline?.decodeSecondsPerToken\n                ?? benchmarkGolden.resolvedBaselineDecodeSecondsPerToken"))
    let worker = try packageFile("Sources/MLXFastHarness/GemmaRuntimeWorker.swift")
    #expect(worker.contains("\"MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN\","))
    #expect(worker.contains("\"MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN\","))

    // The contract is documented in the freeze doc.
    #expect(doc.contains("Paired baseline measurement (official timing machine)"))
    #expect(doc.contains("paired override, then golden-carried baselines, then the\n  constants"))
}

@Test
func decodeValidationDelayHookDefaultsToNoOp() {
    // The one editable-surface knob that can add time to the trusted decode loop
    // must read zero on main/baseline. It can only ever slow a submission down,
    // never speed it up, but the frozen baseline is measured at zero delay, so a
    // nonzero default here would mean the baseline and submissions were timed
    // through different decode loops.
    //
    // BenchmarkSupportTests.submissionValidationDelayDefaultsToZero asserts the
    // same literal for a different reason (the general default of the hook). The
    // re-assert here is intentional: this file is meant to be the single,
    // self-contained guard for everything the frozen window depends on, so it
    // does not rely on an unrelated test staying green.
    #expect(Gemma4SubmissionControls.measuredDecodeDelayMilliseconds == 0)
}
