import Foundation
@testable import MLXFastCore
import Testing

// Prefill acceptance band: a robust reference B is calibrated once from a fleet
// (drop the single slowest, average the rest); each run's single prefill is then
// checked against B ([-5%, +2%]). Numbers anchored on the real tenki fresh-VM run
// 28893815980 (prefills: 5 tight at ~0.0106, one spike at 0.01532).

private let freshVMPrefills = [
    0.010556244302734375, 0.01071322314453125, 0.010636167642578125,
    0.010626223958984375, 0.01532235343359375, 0.01049330069921875,
]

@Test
func robustReferenceDropsTheSlowestAndAverages() {
    let B = PrefillBand.robustReference(samples: freshVMPrefills)
    #expect(B != nil)
    // mean of the 5 after dropping the 0.01532 spike
    #expect(abs((B ?? 0) - 0.0106048) < 1e-5)
}

@Test
func robustReferenceRejectsTooFewOrInvalid() {
    #expect(PrefillBand.robustReference(samples: [0.0106, 0.0106]) == nil)
    #expect(PrefillBand.robustReference(samples: [0.0106, 0.0, 0.0106]) == nil)
    #expect(PrefillBand.robustReference(samples: [0.0106, .nan, 0.0106]) == nil)
}

@Test
func checkAcceptsAtReferenceAndModestlyFaster() {
    let B = 0.0106
    #expect(PrefillBand.check(prefill: B, reference: B).passed)          // exactly at B
    #expect(PrefillBand.check(prefill: B * 1.01, reference: B).passed)   // +1% slow: ok
    #expect(PrefillBand.check(prefill: B * 0.97, reference: B).passed)   // -3% fast: ok
}

@Test
func checkFailsWhenMoreThan2PercentSlow() {
    let B = 0.0106
    #expect(!PrefillBand.check(prefill: B * 1.021, reference: B).passed)
    #expect(PrefillBand.check(prefill: B * 1.021, reference: B).reason.contains("slowdown"))
    // the real 0.01532 spike vs B=0.010605 is +44% -> fail
    #expect(!PrefillBand.check(prefill: 0.01532235343359375, reference: 0.0106048).passed)
}

@Test
func checkFailsWhenMoreThan5PercentFast() {
    let B = 0.0106
    #expect(!PrefillBand.check(prefill: B * 0.949, reference: B).passed)
    #expect(PrefillBand.check(prefill: B * 0.949, reference: B).reason.contains("lucky"))
}

@Test
func checkRejectsNonFinite() {
    #expect(!PrefillBand.check(prefill: 0.0, reference: 0.0106).passed)
    #expect(!PrefillBand.check(prefill: 0.0106, reference: 0.0).passed)
    #expect(!PrefillBand.check(prefill: .nan, reference: 0.0106).passed)
}

@Test
func bandTolerancesMatchConstants() {
    #expect(MLXFastConstants.prefillBandUpTolerance == 0.02)
    #expect(MLXFastConstants.prefillBandDownTolerance == 0.05)
}
