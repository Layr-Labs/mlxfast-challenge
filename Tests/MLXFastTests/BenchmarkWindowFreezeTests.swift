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
private func swiftConstantLiteral(_ source: String, name: String) throws -> String {
    let marker = "let \(name) = "
    let start = try #require(
        source.range(of: marker),
        "expected constant \(name) in source"
    )
    let rest = source[start.upperBound...]
    let lineEnd = rest.firstIndex(of: "\n") ?? rest.endIndex
    return String(rest[..<lineEnd]).trimmingCharacters(in: .whitespaces)
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

    // The doc must quote the exact calibrated baseline literals from Constants,
    // so a re-baseline cannot land while the freeze doc still shows the old one.
    let decodeBaseline = try swiftConstantLiteral(constants, name: "officialBaselineDecodeSecondsPerToken")
    let prefillBaseline = try swiftConstantLiteral(constants, name: "officialBaselinePrefillSecondsPerToken")
    #expect(doc.contains(decodeBaseline), "freeze doc must quote officialBaselineDecodeSecondsPerToken=\(decodeBaseline)")
    #expect(doc.contains(prefillBaseline), "freeze doc must quote officialBaselinePrefillSecondsPerToken=\(prefillBaseline)")
}

@Test
func timedDecodeChargesOneValidatedSeedForward() throws {
    let worker = try packageFile("Sources/MLXFastHarness/DeepSeekRuntimeWorker.swift")
    let decodeBegin = try slice(worker, from: "case \"decode_begin\":", to: "case \"decode_step\":")
    // Exactly one whole-prompt forward, and no warmup pass to memoize against it.
    #expect(decodeBegin.components(separatedBy: "DeepSeekModel.logits(").count - 1 == 1)
    #expect(!decodeBegin.contains("warmupCache"))
    #expect(!decodeBegin.contains("warmupLogits"))

    // The decode phase is parent-timed and validated; worker-reported seconds
    // must not be the scored value, and both the seed and the steps are checked.
    let benchmark = try packageFile("Sources/MLXFastHarness/DeepSeekRuntimeBenchmark.swift")
    let measureWorkerDecode = try slice(
        benchmark,
        from: "static func measureWorkerDecode(",
        to: "static func expertStreamingBandwidthGBPerToken("
    )
    #expect(measureWorkerDecode.contains("secondsSince(decodePhaseStart)"))
    #expect(measureWorkerDecode.contains("compareDecodeSeedToken"))
    #expect(measureWorkerDecode.contains("compareDecodeTokens"))
    #expect(!measureWorkerDecode.contains("response.seconds"))
}

@Test
func timedPrefillChargesOneValidatedColdForward() throws {
    let benchmark = try packageFile("Sources/MLXFastHarness/DeepSeekRuntimeBenchmark.swift")
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
func decodeValidationDelayHookDefaultsToNoOp() {
    // The one editable-surface knob that can add time to the trusted decode loop
    // must read zero on main/baseline. It can only ever slow a submission down,
    // never speed it up, but the frozen baseline is measured at zero delay, so a
    // nonzero default here would mean the baseline and submissions were timed
    // through different decode loops.
    #expect(DeepSeekSubmissionControls.measuredDecodeDelayMilliseconds == 0)
}
