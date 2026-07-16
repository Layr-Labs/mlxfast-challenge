import Darwin
import MLX

/// Selects a model-startup profile from the machine's physical-memory budget.
///
/// The optimized Gemma runtime can retain roughly 14.5 GiB of alternate
/// co-tiled/combined weight layouts in addition to the ~16.9 GiB checkpoint.
/// That full profile is appropriate for the 128 GiB ranked runner, but leaves
/// too little headroom for macOS, warmup activations, KV state, and Metal
/// buffers on the documented 36 GiB local minimum.
///
/// The low-memory feature defaults never clobber explicit settings: a flag
/// the user already exported keeps its value (so, for example, verifying a
/// co-tiled layout on a 48 GiB machine still works), and the automatic
/// selection itself can be overridden with
/// `DARKBLOOM_STARTUP_MEMORY_PROFILE=full|low|auto`. When the low-memory
/// profile engages it announces itself on stderr, including any user-set
/// flags it preserved.
public struct Gemma4StartupMemoryPolicy: Equatable, Sendable {
    public static let fullProfileMinimumPhysicalMemoryBytes = UInt64(64) << 30

    /// Environment name for the explicit profile override. It must keep the
    /// `DARKBLOOM_` prefix: the trusted harness forwards only that
    /// model-tuning family (plus MLX_/system prefixes) into the runtime
    /// worker's sanitized environment, so any other spelling would never
    /// reach the process that resolves the policy.
    public static let profileOverrideEnvironmentName =
        "DARKBLOOM_STARTUP_MEMORY_PROFILE"

    public let isLowMemory: Bool
    /// Why this profile was selected; quoted in the stderr notice.
    public let selectionReason: String
    public let cacheLimitBytes: Int
    public let maxMegabytesPerCommandBuffer: Int
    public let maxOperationsPerCommandBuffer: Int
    public let clearAllocatorCacheAfterWarmup: Bool
    public let environmentOverrides: [String: String]

    public static func resolve(
        physicalMemoryBytes: UInt64,
        requestedProfile: String? = nil
    ) -> Gemma4StartupMemoryPolicy {
        let lowMemory: Bool
        let selectionReason: String
        switch requestedProfile?.lowercased() ?? "" {
        case "", "auto":
            lowMemory = physicalMemoryBytes < fullProfileMinimumPhysicalMemoryBytes
            selectionReason = "physical memory \(physicalMemoryBytes >> 30) GiB "
                + (lowMemory ? "is below" : "meets")
                + " the \(fullProfileMinimumPhysicalMemoryBytes >> 30) GiB full-profile minimum"
        case "full":
            lowMemory = false
            selectionReason = "\(profileOverrideEnvironmentName)=full"
        case "low":
            lowMemory = true
            selectionReason = "\(profileOverrideEnvironmentName)=low"
        default:
            preconditionFailure(
                "\(profileOverrideEnvironmentName) must be auto, full, or low"
            )
        }

        if lowMemory {
            return Gemma4StartupMemoryPolicy(
                isLowMemory: true,
                selectionReason: selectionReason,
                cacheLimitBytes: 6 << 30,
                // Half the full profile's referenced-byte and op budgets:
                // shorter command buffers bound transient in-flight memory on
                // machines whose allocator cache is also capped to match the
                // trusted worker's 6 GiB phase-start value.
                maxMegabytesPerCommandBuffer: 128,
                maxOperationsPerCommandBuffer: 64,
                clearAllocatorCacheAfterWarmup: true,
                environmentOverrides: [
                    // Avoid retaining compiled closures and the largest alternate
                    // weight layouts in addition to the source model. The
                    // VERIFY_* companions matter too: a verify flag alone
                    // re-materializes the layout it compares against.
                    "MLX_COMPILED_DECODE": "0",
                    "MLX_COMPILED_MLP_TAIL": "0",
                    "DARKBLOOM_COMPILED_DECODE": "0",
                    "MLXFAST_COMBINED_ATTENTION_PREFILL": "0",
                    "MLXFAST_VERIFY_COMBINED_ATTENTION_PREFILL": "0",
                    "MLXFAST_FUSED_QKV": "0",
                    "MLXFAST_FUSED_FULL_QK": "0",
                    "MLXFAST_FUSED_GATE_UP": "0",
                    "MLXFAST_INDEXED_DOWN": "0",
                    "MLXFAST_INDEXED_OUTPUT_FAST": "0",
                    "DARKBLOOM_PACKED_GATE_UP_INDICES": "0",
                    "DARKBLOOM_VERIFY_PACKED_GATE_UP_BITS": "0",
                    "DARKBLOOM_GATE_UP_COTILED_FIXED12": "0",
                    "DARKBLOOM_VERIFY_GATE_UP_COTILED_FIXED12_BITS": "0",
                    "MLXFAST_PACKED_DOWN_INDICES": "0",
                    "MLXFAST_VERIFY_PACKED_DOWN_BITS": "0",
                    "DARKBLOOM_DOWN_COTILED_FIXED12": "0",
                    "DARKBLOOM_VERIFY_DOWN_COTILED_FIXED12_BITS": "0",
                    "MLXFAST_PACKED_OUTPUT_INDICES": "0",
                    "MLXFAST_VERIFY_PACKED_OUTPUT_BITS": "0",
                    "DARKBLOOM_OUTPUT_COTILED": "0",
                    "DARKBLOOM_OUTPUT_COTILED_SLIDING": "0",
                    "DARKBLOOM_OUTPUT_COTILED_FULL": "0",
                    "DARKBLOOM_VERIFY_OUTPUT_COTILED_BITS": "0",
                    "DARKBLOOM_VERIFY_OUTPUT_STOCK_BITS": "0",
                    "DARKBLOOM_TIED_HEAD_QMV": "0",
                    "DARKBLOOM_VERIFY_TIED_HEAD_BITS": "0",
                ]
            )
        }

        return Gemma4StartupMemoryPolicy(
            isLowMemory: false,
            selectionReason: selectionReason,
            // Ranked/full profile -- byte-identical to the constants this
            // policy replaced. The 32 GiB soft allocator cap lets the M5 Max
            // retain freed intermediates for reuse; it is not a reservation,
            // and model weights stay active allocations outside it.
            cacheLimitBytes: 32 << 30,
            // The MLX M5 Max default commits a command buffer after
            // referencing 50 MiB. Many 4-bit projections individually exceed
            // that, so 320 MiB groups adjacent kernels without long command
            // buffers; decode's explicit async-eval groups remain the outer
            // command-buffer boundary, and this referenced-buffer budget
            // governs within them.
            maxMegabytesPerCommandBuffer: 320,
            maxOperationsPerCommandBuffer: 128,
            clearAllocatorCacheAfterWarmup: false,
            environmentOverrides: [:]
        )
    }

    /// The environment work `apply()` will perform, split into defaults to
    /// install (name currently unset) and explicit user values to preserve,
    /// plus the stderr notice lines describing the outcome. Pure given
    /// `existingValue`, so tests can verify the no-overwrite semantics and
    /// the notice without mutating process state.
    func environmentPlan(
        existingValue: (String) -> String?
    ) -> Gemma4StartupMemoryEnvironmentPlan {
        var defaultsToApply: [String: String] = [:]
        var preservedUserValues: [String: String] = [:]
        for (name, value) in environmentOverrides {
            if let existing = existingValue(name) {
                preservedUserValues[name] = existing
            } else {
                defaultsToApply[name] = value
            }
        }
        var noticeLines: [String] = []
        if isLowMemory {
            noticeLines.append(
                "mlxfast: low-memory startup profile active (\(selectionReason)): "
                    + "applying \(defaultsToApply.count) feature-disable defaults and a "
                    + "\(cacheLimitBytes >> 30) GiB MLX allocator-cache cap; set "
                    + "\(Self.profileOverrideEnvironmentName)=full to opt out"
            )
            if !preservedUserValues.isEmpty {
                let preserved = preservedUserValues.keys.sorted()
                    .map { name in "\(name)=\(preservedUserValues[name] ?? "")" }
                    .joined(separator: " ")
                noticeLines.append(
                    "mlxfast: low-memory startup profile preserved user-set flags: "
                        + preserved
                )
            }
        }
        return Gemma4StartupMemoryEnvironmentPlan(
            defaultsToApply: defaultsToApply,
            preservedUserValues: preservedUserValues,
            noticeLines: noticeLines
        )
    }

    func apply() {
        // Command-buffer budgets are per-profile absolutes (the pre-policy
        // code force-set them identically); only the opt-in feature flags
        // below use no-overwrite semantics.
        setenv(
            "MLX_MAX_MB_PER_BUFFER",
            String(maxMegabytesPerCommandBuffer),
            1
        )
        setenv(
            "MLX_MAX_OPS_PER_BUFFER",
            String(maxOperationsPerCommandBuffer),
            1
        )
        let plan = environmentPlan { name in
            getenv(name).map { String(cString: $0) }
        }
        for (name, value) in plan.defaultsToApply {
            setenv(name, value, 0)
        }
        for line in plan.noticeLines {
            fputs(line + "\n", stderr)
        }
        Memory.cacheLimit = cacheLimitBytes
    }
}

/// See `Gemma4StartupMemoryPolicy.environmentPlan(existingValue:)`.
struct Gemma4StartupMemoryEnvironmentPlan: Equatable, Sendable {
    let defaultsToApply: [String: String]
    let preservedUserValues: [String: String]
    let noticeLines: [String]
}
