import Foundation
import MLX
@testable import MLXFastCore
@testable import MLXFastHarness
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
    #expect(MLXFastConstants.prefillBandUpTolerance == 0.05)
    #expect(MLXFastConstants.prefillBandDownTolerance == 0.05)
    #expect(MLXFastConstants.decodeBandUpTolerance == 0.02)
    #expect(MLXFastConstants.decodeBandDownTolerance == 0.05)
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
    #expect(doc.contains("decodeBandUpTolerance = \(MLXFastConstants.decodeBandUpTolerance)"))
    #expect(doc.contains("decodeBandDownTolerance = \(MLXFastConstants.decodeBandDownTolerance)"))

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

    // The in-process (non-worker) decode path must also run its single seed
    // forward with NO preceding warmup pass. A second identical whole-prompt
    // forward would let submitted model code memoize one pass and serve the
    // other from that memo, collapsing two charged forwards into one. (Unlike
    // decode_begin, this path inlines the per-step decode loop with its own
    // single-token logits calls, so assert the absence of the warmup rather
    // than a whole-prompt forward count.)
    let measureDecode = try slice(
        benchmark,
        from: "static func measureDecode(",
        to: "static func measureWorkerDecode("
    )
    #expect(!measureDecode.contains("warmupCache"))
    #expect(!measureDecode.contains("warmupLogits"))
    #expect(!measureDecode.contains("decode warmup start"))
    #expect(measureDecode.contains("preceding warmup pass"))
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
func workerPhaseStartsResetTrustedAllocatorBeforeForwardSetup() throws {
    let worker = try packageFile("Sources/MLXFastHarness/GemmaRuntimeWorker.swift")
    let resetCall = "try resetRuntimeWorkerAllocatorForPhaseStart()"

    // The phase-start value and reset live in the trusted harness, not in
    // editable model construction. MLX documents clearCache() as synchronously
    // deallocating every cached (free) buffer, so a nonzero postcondition must
    // fail closed. The reset pins the sequence-boundary state only (editable
    // code may change the limit inside the charged window); the doc must say
    // so instead of claiming a phase-long "policy" that is not enforced.
    #expect(worker.contains("static let trustedRuntimeWorkerPhaseStartCacheLimitBytes = 6 << 30"))
    let allocatorReset = try slice(
        worker,
        from: "static func resetRuntimeWorkerAllocatorForPhaseStart()",
        to: "static func handleWorkerRequest("
    )
    let setLimit = try #require(
        allocatorReset.range(of: "Memory.cacheLimit = trustedRuntimeWorkerPhaseStartCacheLimitBytes")
    )
    let clearCache = try #require(allocatorReset.range(of: "Memory.clearCache()"))
    let readCacheMemory = try #require(allocatorReset.range(of: "Memory.cacheMemory"))
    let zeroCheck = try #require(allocatorReset.range(of: "guard remainingCacheBytes == 0 else"))
    #expect(setLimit.lowerBound < clearCache.lowerBound)
    #expect(clearCache.lowerBound < readCacheMemory.lowerBound)
    #expect(readCacheMemory.lowerBound < zeroCheck.lowerBound)

    // Every new sequence resets exactly once before constructing its request
    // cache or entering the helper that performs the first logits call.
    let phaseBegins = [
        (
            start: "case \"correctness\":",
            end: "case \"correctness_begin\":",
            firstForwardSetup: "let tokens = try generateGreedyCached("
        ),
        (
            start: "case \"correctness_begin\":",
            end: "case \"correctness_step\":",
            firstForwardSetup: "let cache = Gemma4ModelCache("
        ),
        (
            start: "case \"prefill\":",
            end: "case \"decode_begin\":",
            firstForwardSetup: "let cache = Gemma4ModelCache("
        ),
        (
            start: "case \"decode_begin\":",
            end: "case \"decode_step\":",
            firstForwardSetup: "let cache = Gemma4ModelCache("
        ),
    ]
    for phase in phaseBegins {
        let body = try slice(worker, from: phase.start, to: phase.end)
        #expect(body.components(separatedBy: resetCall).count - 1 == 1)
        let reset = try #require(body.range(of: resetCall))
        let firstForwardSetup = try #require(body.range(of: phase.firstForwardSetup))
        #expect(reset.lowerBound < firstForwardSetup.lowerBound)
    }

    // Step requests are part of an already-started sequence. Clearing here
    // would destroy legitimate, charged KV/intermediate reuse.
    let correctnessStep = try slice(
        worker,
        from: "case \"correctness_step\":",
        to: "case \"prefill\":"
    )
    let decodeStep = try slice(
        worker,
        from: "case \"decode_step\":",
        to: "case \"phase_diagnostics\":"
    )
    #expect(!correctnessStep.contains(resetCall))
    #expect(!decodeStep.contains(resetCall))

    let doc = try packageFile("docs/benchmark-window-freeze.md")
    #expect(doc.contains("trusted 6 GiB MLX free-buffer cache reset"))
    #expect(doc.contains("The reset pins the phase-start state only"))
    #expect(doc.contains("not an enforced cap for the rest of the phase"))
    #expect(doc.contains("No allocator reset runs in `correctness_step` or `decode_step`"))
}

@Test
func trustedParentStartsPhaseTimerBeforeWorkerResetRequest() throws {
    let benchmark = try packageFile("Sources/MLXFastHarness/GemmaRuntimeBenchmark.swift")
    let prefill = try slice(
        benchmark,
        from: "static func measureWorkerPrefillSecondsPerToken(",
        to: "static func measureDecode("
    )
    let prefillTimer = try #require(
        prefill.range(of: "let prefillStart = DispatchTime.now().uptimeNanoseconds")
    )
    let prefillRequest = try #require(
        prefill.range(of: "let response = try worker.prefill(promptTokens: promptTokens)")
    )
    #expect(prefillTimer.lowerBound < prefillRequest.lowerBound)

    let decode = try slice(
        benchmark,
        from: "static func measureWorkerDecode(",
        to: "static let bandwidthSource"
    )
    let decodeTimer = try #require(
        decode.range(of: "let decodePhaseStart = DispatchTime.now().uptimeNanoseconds")
    )
    let decodeRequest = try #require(
        decode.range(of: "let beginResponse = try worker.beginDecode(seedTokens: seedTokens)")
    )
    #expect(decodeTimer.lowerBound < decodeRequest.lowerBound)
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
    let score = try packageFile("Sources/MLXFastCore/Score.swift")
    // Both benchmark paths adopt the golden's resolved baselines...
    #expect(benchmark.components(separatedBy: "benchmarkGolden.resolvedBaselinePrefillSecondsPerToken").count - 1 == 2)
    #expect(benchmark.components(separatedBy: "benchmarkGolden.resolvedBaselineDecodeSecondsPerToken").count - 1 == 2)
    // ...and every scored speedup uses the resolved values, never the raw constants.
    #expect(score.contains("baselineSecondsPerToken: baselineDecodeSecondsPerToken"))
    #expect(score.contains("baselineSecondsPerToken: baselinePrefillSecondsPerToken"))
    #expect(benchmark.contains("BenchmarkScore.evaluateTimedRun("))
    #expect(!score.contains("baselineSecondsPerToken: MLXFastConstants.officialBaselineDecodeSecondsPerToken"))
    #expect(!score.contains("baselineSecondsPerToken: MLXFastConstants.officialBaselinePrefillSecondsPerToken"))
    // The benchmark file feeds the baselines into BenchmarkScore/ScorePayload;
    // no call site there may bind a baseline argument straight to the raw
    // constants (parameter DEFAULTS spell ": Double = MLXFastConstants.", so
    // they do not match). Without this the two negative checks above are
    // vacuous: Score.swift only ever sees baselines as parameters.
    #expect(!benchmark.contains("baselineDecodeSecondsPerToken: MLXFastConstants."))
    #expect(!benchmark.contains("baselinePrefillSecondsPerToken: MLXFastConstants."))
}

@Test
func officialRankedRunMeasuresPairedBaselineOnTheSameSilicon() throws {
    // The ranking contract on official runs: speedups and floors are computed
    // against a baseline measured in the SAME session on the SAME silicon
    // (paired baseline), cancelling host/time drift that made fixed-constant
    // floor verdicts near the threshold a coin flip. On the single-machine M5
    // runner the pairing is owned by measure-job (trusted, runner-provisioned,
    // fixed thermal contract): it measures the PINNED baseline tree and the
    // candidate back to back, each behind the same thermal gate, and
    // overlay-paired-timing.sh applies the paired ratio and the frozen floors
    // in the trusted shell.
    let workflow = try packageFile(".github/workflows/serial-benchmark.yml")
    let overlay = try packageFile(".github/scripts/overlay-paired-timing.sh")
    let constants = try packageFile("Sources/MLXFastCore/Constants.swift")

    // The timed measurement runs LAST -- after the compute-heavy correctness/
    // gates pass, the hidden-material scrub, and the quiescence wait -- and
    // the overlay merges its result into the sealed gates score.
    let gates = try #require(workflow.range(of: "- name: Correctness and gates"))
    let scrub = try #require(workflow.range(of: "- name: Scrub hidden material from bench workspace"))
    let quiescence = try #require(workflow.range(of: "- name: Wait for quiescence before timing"))
    let measure = try #require(workflow.range(of: "- name: Timed paired benchmark (measure-job)"))
    let overlayStep = try #require(workflow.range(of: "- name: Overlay paired timing into final score"))
    #expect(gates.lowerBound < scrub.lowerBound)
    #expect(scrub.lowerBound < quiescence.lowerBound)
    #expect(quiescence.lowerBound < measure.lowerBound)
    #expect(measure.lowerBound < overlayStep.lowerBound)

    // The baseline is the PINNED tree provisioned on the box, sanity-banded
    // against its recorded calibration -- both consumed from the host
    // contract, never installed or repointable by the workflow run itself.
    #expect(workflow.contains("MLXFAST_MEASURE_JOB: /opt/bench-runner/measure-job.sh"))
    #expect(workflow.contains("MLXFAST_BASELINE_WS: /opt/bench-runner/baseline/current"))
    #expect(workflow.contains("MLXFAST_BASELINE_CALIBRATION: /opt/bench-runner/state/baseline-calibration.json"))
    let measureBody = String(workflow[measure.lowerBound..<overlayStep.lowerBound])
    #expect(measureBody.contains("--candidate \"${MLXFAST_JOB_WS}\""))
    #expect(measureBody.contains("--baseline \"${MLXFAST_BASELINE_WS}\""))
    #expect(measureBody.contains("BASELINE_CALIBRATION=\"${MLXFAST_BASELINE_CALIBRATION}\""))

    // Floors applied at overlay time are the frozen ranking-contract floors,
    // kept in sync with the constants; the paired score uses the frozen
    // geometric weights.
    let decodeFloor = try swiftConstantLiteral(constants, name: "scoreDecodeSpeedupFloor")
    let prefillFloor = try swiftConstantLiteral(constants, name: "scorePrefillSpeedupFloor")
    #expect(workflow.contains("MLXFAST_DECODE_SPEEDUP_FLOOR: \"\(decodeFloor)\""))
    #expect(workflow.contains("MLXFAST_PREFILL_SPEEDUP_FLOOR: \"\(prefillFloor)\""))
    #expect(overlay.contains("($ds >= $decode_floor) as $decode_ok"))
    #expect(overlay.contains("($ps >= $prefill_floor) as $prefill_ok"))
    #expect(overlay.contains(".score = (pow($ds; 0.75) * pow($ps; 0.25))"))
    #expect(overlay.contains("performance floor failed"))

    // Harness side (still shipped and armed): a paired override outranks
    // golden-carried baselines, which outrank constants, the pair is
    // fail-closed, and the override keys never reach the sandboxed worker.
    let benchmark = try packageFile("Sources/MLXFastHarness/GemmaRuntimeBenchmark.swift")
    #expect(benchmark.components(separatedBy: "PairedBaselineOverride.fromEnvironment()").count - 1 == 2)
    #expect(benchmark.contains("pairedBaseline?.prefillSecondsPerToken\n                ?? benchmarkGolden.resolvedBaselinePrefillSecondsPerToken"))
    #expect(benchmark.contains("pairedBaseline?.decodeSecondsPerToken\n                ?? benchmarkGolden.resolvedBaselineDecodeSecondsPerToken"))
    // The worker environment filter is a strict allowlist (see
    // sanitizedRuntimeWorkerEnvironment), so the override keys are excluded
    // structurally; assert that against the real filter rather than pinning
    // denylist source text.
    let workerEnvironment = sanitizedRuntimeWorkerEnvironment([
        "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "3.6977511595078125",
        "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "0.16518489738085937",
    ])
    #expect(workerEnvironment["MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN"] == nil)
    #expect(workerEnvironment["MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN"] == nil)
}

@Test
func decodeMeasurementInvokesNoPhaseVaryingEditableHook() throws {
    // The scored decode loop must invoke NO editable hook that is unique to it.
    // The former editable knob (Gemma4SubmissionControls.measuredDecodeDelay-
    // Milliseconds, read via submissionValidationDelayMilliseconds) was model
    // code called by trusted code ONLY on the timed decode path -- so "my getter
    // was invoked" was an unambiguous "I am being scored now" signal, and the
    // submission could run arbitrary phase-detection inside the getter. It is
    // removed entirely (editable file deleted, no trusted call site remains).
    // This file is the single self-contained guard for the frozen window, so it
    // freezes phase-independence here directly rather than relying on an
    // unrelated test.
    #expect(!FileManager.default.fileExists(
        atPath: "Sources/MLXFastModel/Gemma4SubmissionControls.swift"
    ))
    let worker = try packageFile("Sources/MLXFastHarness/GemmaRuntimeWorker.swift")
    let benchmark = try packageFile("Sources/MLXFastHarness/GemmaRuntimeBenchmark.swift")
    let localIterate = try packageFile("Sources/MLXFastHarness/GemmaRuntimeLocalIterate.swift")
    for source in [worker, benchmark, localIterate] {
        #expect(!source.contains("submissionValidationDelayMilliseconds"))
        #expect(!source.contains("measuredDecodeDelayMilliseconds"))
        #expect(!source.contains("Gemma4SubmissionControls"))
    }
    // The worker decode_step case invokes only the same editable entry points
    // the correctness path also invokes (Gemma4Model.logits / greedyToken), and
    // no per-token sleep that a delay hook used to drive.
    let decodeStepStart = try #require(worker.range(of: "case \"decode_step\":"))
    let decodeStepEnd = try #require(
        worker.range(
            of: "case \"phase_diagnostics\":",
            range: decodeStepStart.upperBound..<worker.endIndex
        )
    )
    let decodeStep = String(worker[decodeStepStart.upperBound..<decodeStepEnd.lowerBound])
    #expect(decodeStep.contains("Gemma4Model.logits("))
    #expect(!decodeStep.contains("Thread.sleep"))
}
