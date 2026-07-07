import Foundation

/// Prefill acceptance band.
///
/// Prefill is a single cold forward, measured ONCE per benchmark run on one
/// prompt (bound to the correctness check) -- it is not re-measured. It is also
/// intrinsically noisy (see docs/thermal-variance-investigation.md): on a stable
/// machine it sits at a floor, but a given run's single measurement can spike slow
/// (transient contention) or dip suspiciously fast.
///
/// The defense is two parts:
///
/// 1. `robustReference(samples:)` -- computed ONCE from a calibration fleet of runs
///    (e.g. several fresh-VM measurements of the reference): drop the single
///    slowest sample, average the rest into `B`. This is a robust baseline number,
///    not a per-run measurement.
///
/// 2. `check(prefill:reference:)` -- applied to each scored run's single prefill
///    against `B`. The prefill may be faster than `B`, but:
///      - more than `upTolerance` ABOVE `B`  -> a real slowdown => FAIL.
///      - more than `downTolerance` BELOW `B` -> a suspiciously fast ("lucky")
///        reading => FAIL (so it cannot be variance-harvested).
///    Asymmetric on purpose: a small transient slowdown is normal noise, but a
///    large lucky-fast reading must not be rewarded.
public enum PrefillBand {
    /// Robust reference `B` from a calibration set: drop the single slowest sample,
    /// average the rest. Returns nil if there are too few / invalid samples.
    /// Computed once at calibration time, never per scored run.
    public static func robustReference(samples: [Double]) -> Double? {
        // Need >= 3 so that after dropping one outlier >= 2 remain to average.
        guard samples.count >= 3, samples.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            return nil
        }
        var rest = samples
        if let slowest = rest.indices.max(by: { rest[$0] < rest[$1] }) {
            rest.remove(at: slowest)
        }
        return rest.reduce(0.0, +) / Double(rest.count)
    }

    /// Per-run gate: validate one scored prefill measurement against reference `B`.
    public static func check(
        prefill: Double,
        reference: Double,
        upTolerance: Double = MLXFastConstants.prefillBandUpTolerance,
        downTolerance: Double = MLXFastConstants.prefillBandDownTolerance
    ) -> PrefillBandResult {
        guard prefill.isFinite, prefill > 0, reference.isFinite, reference > 0 else {
            return PrefillBandResult(
                passed: false,
                reason: "prefill (\(prefill)) and reference (\(reference)) must be finite and positive"
            )
        }
        let hi = reference * (1.0 + upTolerance)
        let lo = reference * (1.0 - downTolerance)
        if prefill > hi {
            return PrefillBandResult(
                passed: false,
                reason: "prefill \(prefill) exceeds +\(upTolerance * 100)% of reference "
                    + "\(reference) (> \(hi)): real slowdown"
            )
        }
        if prefill < lo {
            return PrefillBandResult(
                passed: false,
                reason: "prefill \(prefill) below -\(downTolerance * 100)% of reference "
                    + "\(reference) (< \(lo)): suspiciously lucky-fast reading"
            )
        }
        return PrefillBandResult(passed: true, reason: "")
    }
}

public struct PrefillBandResult: Equatable {
    public let passed: Bool
    /// Empty when `passed`; otherwise a human-readable failure reason.
    public let reason: String

    public init(passed: Bool, reason: String) {
        self.passed = passed
        self.reason = reason
    }
}
