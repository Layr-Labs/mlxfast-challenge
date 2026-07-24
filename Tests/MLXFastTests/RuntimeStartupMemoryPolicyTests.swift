import Foundation
@testable import MLXFastModel
import Testing

@Test
func startupMemoryPolicyProtectsDocumented36GiBLocalMachine() {
    let policy = RuntimeStartupMemoryPolicy.resolve(
        physicalMemoryBytes: UInt64(36) << 30
    )

    #expect(policy.isLowMemory)
    #expect(policy.cacheLimitBytes == 6 << 30)
    #expect(policy.maxMegabytesPerCommandBuffer == 128)
    #expect(policy.maxOperationsPerCommandBuffer == 64)
    #expect(policy.clearAllocatorCacheAfterWarmup)
    // Exact-size pin plus the compiled-decode switches the profile exists
    // to disable; every name's spelling is cross-checked against the model
    // sources in lowMemoryOverridesNameOnlyFlagsReadByModelSources, so the
    // dictionary is not duplicated here verbatim.
    #expect(policy.environmentOverrides.count == 2)
    #expect(policy.environmentOverrides.values.allSatisfy { $0 == "0" })
    for name in [
        "MLX_COMPILED_DECODE",
        "DARKBLOOM_COMPILED_DECODE",
    ] {
        #expect(policy.environmentOverrides[name] == "0", "missing \(name)")
    }
}

// Every override must name a flag some model source actually reads;
// a typo or a rename that leaves the policy behind fails here. The policy
// file itself is excluded so the dictionary cannot self-satisfy the check.
// The scanned surface is the worker's model code: Sources/MLXFastModel plus
// the vendored MLXLMCommon runtime plumbing the Laguna forward dispatches
// (the compiled-decode flags are read there).
@Test
func lowMemoryOverridesNameOnlyFlagsReadByModelSources() throws {
    let modelSourceDirectories = [
        "Sources/MLXFastModel",
        "Vendor/mlx-swift-lm/Libraries/MLXLMCommon",
    ]
    var combinedSources = ""
    for directory in modelSourceDirectories {
        let sourceFiles = try FileManager.default
            .contentsOfDirectory(atPath: directory)
            .filter { $0.hasSuffix(".swift") && $0 != "RuntimeStartupMemoryPolicy.swift" }
        #expect(!sourceFiles.isEmpty)
        combinedSources += try sourceFiles
            .map { try String(contentsOfFile: "\(directory)/\($0)", encoding: .utf8) }
            .joined(separator: "\n")
    }

    let policy = RuntimeStartupMemoryPolicy.resolve(
        physicalMemoryBytes: UInt64(36) << 30
    )
    for name in policy.environmentOverrides.keys.sorted() {
        #expect(
            combinedSources.contains("\"\(name)\""),
            "override \(name) is not read anywhere in \(modelSourceDirectories)"
        )
    }
}

// benchmark.sh's end-of-run ranked-parity warning re-derives this policy's
// decision from the same inputs (physical memory, the profile override,
// per-flag environment presence), so the flag list, full-profile threshold,
// override name, and no-overwrite check it embeds must track this policy
// exactly. A policy change that leaves the shell mirror behind fails here.
@Test
func benchmarkScriptParityWarningMirrorsLowMemoryPolicy() throws {
    let script = try String(contentsOfFile: "benchmark.sh", encoding: .utf8)
    let policy = RuntimeStartupMemoryPolicy.resolve(
        physicalMemoryBytes: UInt64(36) << 30
    )

    // The shell flag array enumerates exactly the low-memory feature-disable
    // overrides -- no missing flag, no stale extra.
    let flagsLine = try #require(
        script.split(separator: "\n").first {
            $0.hasPrefix("readonly LOW_MEMORY_PROFILE_DISABLED_FLAGS=(")
        }
    )
    let openParenthesis = try #require(flagsLine.firstIndex(of: "("))
    let closeParenthesis = try #require(flagsLine.lastIndex(of: ")"))
    let shellFlags = Set(
        flagsLine[flagsLine.index(after: openParenthesis)..<closeParenthesis]
            .split(separator: " ")
            .map(String.init)
    )
    #expect(shellFlags == Set(policy.environmentOverrides.keys))
    #expect(policy.environmentOverrides.values.allSatisfy { $0 == "0" })

    // The shell threshold mirrors the policy's full-profile minimum, and the
    // profile override is read under the policy's worker-reachable name.
    let minimumGiB = RuntimeStartupMemoryPolicy.fullProfileMinimumPhysicalMemoryBytes >> 30
    #expect(script.contains(
        "readonly LOW_MEMORY_PROFILE_FULL_MIN_BYTES=$((\(minimumGiB) << 30))"
    ))
    #expect(script.contains(
        "${\(RuntimeStartupMemoryPolicy.profileOverrideEnvironmentName):-}"
    ))

    // The shell mirrors the policy's no-overwrite semantics with a per-flag
    // set/unset check (printenv treats set-but-empty as set, like getenv), so
    // explicitly exported flags are never warned about.
    #expect(script.contains("if ! printenv \"${flag}\" >/dev/null 2>&1; then"))
}

// The documented low-memory startup profile (<64 GiB machines: 6 GiB MLX
// allocator cap, feature-disable env defaults, warmup-buffer clear before
// the protocol hello) applies to the LAGUNA runtime worker. The Laguna
// weight cache must resolve the policy
// before its model load and honor clearAllocatorCacheAfterWarmup; the full
// profile stays a no-op there (the ranked box keeps stock allocator
// behavior, matching how the pinned baseline was measured).
@Test
func lagunaWeightCacheConsultsStartupMemoryPolicy() throws {
    let source = try String(
        contentsOfFile: "Sources/MLXFastModel/LagunaRuntimeWeights.swift",
        encoding: .utf8
    )
    #expect(source.contains("RuntimeStartupMemoryPolicy.resolve("))
    #expect(source.contains("RuntimeStartupMemoryPolicy.profileOverrideEnvironmentName"))
    #expect(source.contains("if policy.isLowMemory {"))
    #expect(source.contains("startupMemoryPolicy?.clearAllocatorCacheAfterWarmup == true"))
}

@Test
func startupMemoryPolicyKeepsRanked128GiBProfile() {
    let policy = RuntimeStartupMemoryPolicy.resolve(
        physicalMemoryBytes: UInt64(128) << 30
    )

    #expect(!policy.isLowMemory)
    #expect(policy.cacheLimitBytes == 32 << 30)
    #expect(policy.maxMegabytesPerCommandBuffer == 320)
    #expect(policy.maxOperationsPerCommandBuffer == 128)
    #expect(!policy.clearAllocatorCacheAfterWarmup)
    #expect(policy.environmentOverrides.isEmpty)

    // The ranked path must stay silent and write no feature defaults.
    let plan = policy.environmentPlan { _ in nil }
    #expect(plan.defaultsToApply.isEmpty)
    #expect(plan.preservedUserValues.isEmpty)
    #expect(plan.noticeLines.isEmpty)
}

@Test
func startupMemoryPolicyUses64GiBAsFullProfileBoundary() {
    #expect(
        RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: (UInt64(64) << 30) - 1
        ).isLowMemory
    )
    #expect(
        !RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(64) << 30
        ).isLowMemory
    )
}

@Test
func startupMemoryPolicyHonorsExplicitProfileRequest() {
    #expect(
        !RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(36) << 30,
            requestedProfile: "full"
        ).isLowMemory
    )
    #expect(
        RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(128) << 30,
            requestedProfile: "low"
        ).isLowMemory
    )
    #expect(
        !RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(36) << 30,
            requestedProfile: "FULL"
        ).isLowMemory
    )
    #expect(
        RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(36) << 30,
            requestedProfile: "auto"
        ).isLowMemory
    )
    #expect(
        RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(36) << 30,
            requestedProfile: ""
        ).isLowMemory
    )
    // The override must keep its DARKBLOOM_ prefix: the trusted worker
    // environment allowlist forwards that family, so renaming it (e.g. to
    // MLXFAST_*) would silently stop it from ever reaching the worker.
    #expect(
        RuntimeStartupMemoryPolicy.profileOverrideEnvironmentName
            == "DARKBLOOM_STARTUP_MEMORY_PROFILE"
    )
}

@Test
func lowMemoryPlanPreservesUserSetFlagsAndReportsThem() throws {
    let policy = RuntimeStartupMemoryPolicy.resolve(
        physicalMemoryBytes: UInt64(36) << 30
    )
    let userSet = "DARKBLOOM_COMPILED_DECODE"
    let plan = policy.environmentPlan { name in
        name == userSet ? "1" : nil
    }

    // The explicitly-set flag is preserved, every unset flag gets its
    // low-memory default, and nothing else is touched.
    #expect(plan.preservedUserValues == [userSet: "1"])
    #expect(plan.defaultsToApply[userSet] == nil)
    #expect(plan.defaultsToApply.count == policy.environmentOverrides.count - 1)
    #expect(plan.defaultsToApply["MLX_COMPILED_DECODE"] == "0")

    // The notice names the active profile, the opt-out, and the preserved flag.
    try #require(plan.noticeLines.count == 2)
    #expect(plan.noticeLines[0].contains("low-memory startup profile active"))
    #expect(plan.noticeLines[0].contains("DARKBLOOM_STARTUP_MEMORY_PROFILE=full"))
    #expect(plan.noticeLines[1].contains("\(userSet)=1"))
}

@Test
func lowMemoryPlanAppliesAllDefaultsWhenNoneAreUserSet() throws {
    let policy = RuntimeStartupMemoryPolicy.resolve(
        physicalMemoryBytes: UInt64(36) << 30
    )
    let plan = policy.environmentPlan { _ in nil }

    #expect(plan.defaultsToApply == policy.environmentOverrides)
    #expect(plan.preservedUserValues.isEmpty)
    try #require(plan.noticeLines.count == 1)
    #expect(plan.noticeLines[0].contains("low-memory startup profile active"))
}

@Test
func startupMemoryPolicyIsAppliedBeforeModelLoadAndCleansWarmupCache() throws {
    let source = try String(
        contentsOfFile: "Sources/MLXFastModel/LagunaRuntimeWeights.swift",
        encoding: .utf8
    )
    // Scope every assertion to the initializer body so this pins statement
    // order inside init, not merely textual order anywhere in the file.
    let initBody = try initializerBody(of: source)

    let resolve = try #require(
        initBody.range(of: "RuntimeStartupMemoryPolicy.resolve(")
    )
    let apply = try #require(initBody.range(of: "policy.apply()"))
    let load = try #require(initBody.range(of: "loadLibraryModel("))
    #expect(resolve.lowerBound < apply.lowerBound)
    #expect(apply.lowerBound < load.lowerBound)
    // The explicit profile request must come from the worker-reachable
    // DARKBLOOM_ override name, not a hardcoded or harness-only variable.
    #expect(
        initBody.contains("RuntimeStartupMemoryPolicy.profileOverrideEnvironmentName")
    )

    let warmup = try #require(initBody.range(of: "Self.warmLibraryModel(model)"))
    let cleanup = try #require(
        initBody.range(
            of: "if startupMemoryPolicy?.clearAllocatorCacheAfterWarmup == true"
        )
    )
    let clear = try #require(
        initBody.range(of: "Memory.clearCache()", range: cleanup.lowerBound..<initBody.endIndex)
    )
    #expect(warmup.lowerBound < cleanup.lowerBound)
    #expect(cleanup.lowerBound < clear.lowerBound)
}

private func initializerBody(of source: String) throws -> String {
    let start = try #require(
        source.range(of: "public init(loader: LagunaWeightLoader, config: LagunaConfig)")
    )
    let end = try #require(
        source.range(
            of: "private static func warmLibraryModel",
            range: start.upperBound..<source.endIndex
        )
    )
    return String(source[start.lowerBound..<end.lowerBound])
}
