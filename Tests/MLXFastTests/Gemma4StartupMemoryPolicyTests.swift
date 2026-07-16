import Foundation
@testable import MLXFastModel
import Testing

@Test
func startupMemoryPolicyProtectsDocumented36GiBLocalMachine() {
    let policy = Gemma4StartupMemoryPolicy.resolve(
        physicalMemoryBytes: UInt64(36) << 30
    )

    #expect(policy.isLowMemory)
    #expect(policy.cacheLimitBytes == 6 << 30)
    #expect(policy.maxMegabytesPerCommandBuffer == 128)
    #expect(policy.maxOperationsPerCommandBuffer == 64)
    #expect(policy.clearAllocatorCacheAfterWarmup)
    // Exact-size pin plus the multi-GiB layout switches the profile exists
    // to disable; every name's spelling is cross-checked against the model
    // sources in lowMemoryOverridesNameOnlyFlagsReadByModelSources, so the
    // dictionary is not duplicated here verbatim.
    #expect(policy.environmentOverrides.count == 27)
    #expect(policy.environmentOverrides.values.allSatisfy { $0 == "0" })
    for name in [
        "DARKBLOOM_GATE_UP_COTILED_FIXED12",
        "DARKBLOOM_DOWN_COTILED_FIXED12",
        "MLXFAST_COMBINED_ATTENTION_PREFILL",
        "DARKBLOOM_OUTPUT_COTILED",
        "MLX_COMPILED_DECODE",
        "DARKBLOOM_COMPILED_DECODE",
    ] {
        #expect(policy.environmentOverrides[name] == "0", "missing \(name)")
    }
}

// Every override must name a flag some model source actually reads;
// a typo or a rename that leaves the policy behind fails here. The policy
// file itself is excluded so the dictionary cannot self-satisfy the check.
@Test
func lowMemoryOverridesNameOnlyFlagsReadByModelSources() throws {
    let modelSourcesDirectory = "Sources/MLXFastModel"
    let sourceFiles = try FileManager.default
        .contentsOfDirectory(atPath: modelSourcesDirectory)
        .filter { $0.hasSuffix(".swift") && $0 != "Gemma4StartupMemoryPolicy.swift" }
    #expect(!sourceFiles.isEmpty)
    let combinedSources = try sourceFiles
        .map { try String(contentsOfFile: "\(modelSourcesDirectory)/\($0)", encoding: .utf8) }
        .joined(separator: "\n")

    let policy = Gemma4StartupMemoryPolicy.resolve(
        physicalMemoryBytes: UInt64(36) << 30
    )
    for name in policy.environmentOverrides.keys.sorted() {
        #expect(
            combinedSources.contains("\"\(name)\""),
            "override \(name) is not read anywhere in \(modelSourcesDirectory)"
        )
    }
}

@Test
func startupMemoryPolicyKeepsRanked128GiBProfile() {
    let policy = Gemma4StartupMemoryPolicy.resolve(
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
        Gemma4StartupMemoryPolicy.resolve(
            physicalMemoryBytes: (UInt64(64) << 30) - 1
        ).isLowMemory
    )
    #expect(
        !Gemma4StartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(64) << 30
        ).isLowMemory
    )
}

@Test
func startupMemoryPolicyHonorsExplicitProfileRequest() {
    #expect(
        !Gemma4StartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(36) << 30,
            requestedProfile: "full"
        ).isLowMemory
    )
    #expect(
        Gemma4StartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(128) << 30,
            requestedProfile: "low"
        ).isLowMemory
    )
    #expect(
        !Gemma4StartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(36) << 30,
            requestedProfile: "FULL"
        ).isLowMemory
    )
    #expect(
        Gemma4StartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(36) << 30,
            requestedProfile: "auto"
        ).isLowMemory
    )
    #expect(
        Gemma4StartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(36) << 30,
            requestedProfile: ""
        ).isLowMemory
    )
    // The override must keep its DARKBLOOM_ prefix: the trusted worker
    // environment allowlist forwards that family, so renaming it (e.g. to
    // MLXFAST_*) would silently stop it from ever reaching the worker.
    #expect(
        Gemma4StartupMemoryPolicy.profileOverrideEnvironmentName
            == "DARKBLOOM_STARTUP_MEMORY_PROFILE"
    )
}

@Test
func lowMemoryPlanPreservesUserSetFlagsAndReportsThem() throws {
    let policy = Gemma4StartupMemoryPolicy.resolve(
        physicalMemoryBytes: UInt64(36) << 30
    )
    let userSet = "DARKBLOOM_GATE_UP_COTILED_FIXED12"
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
    let policy = Gemma4StartupMemoryPolicy.resolve(
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
        contentsOfFile: "Sources/MLXFastModel/Gemma4RuntimeWeights.swift",
        encoding: .utf8
    )
    // Scope every assertion to the initializer body so this pins statement
    // order inside init, not merely textual order anywhere in the file.
    let initBody = try initializerBody(of: source)

    let resolve = try #require(
        initBody.range(of: "Gemma4StartupMemoryPolicy.resolve(")
    )
    let apply = try #require(initBody.range(of: "policy.apply()"))
    let load = try #require(initBody.range(of: "loadLibraryModel("))
    #expect(resolve.lowerBound < apply.lowerBound)
    #expect(apply.lowerBound < load.lowerBound)
    // The explicit profile request must come from the worker-reachable
    // DARKBLOOM_ override name, not a hardcoded or harness-only variable.
    #expect(
        initBody.contains("Gemma4StartupMemoryPolicy.profileOverrideEnvironmentName")
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
        source.range(of: "public init(loader: Gemma4WeightLoader, config: Gemma4Config)")
    )
    let end = try #require(
        source.range(
            of: "private static func warmLibraryModel",
            range: start.upperBound..<source.endIndex
        )
    )
    return String(source[start.lowerBound..<end.lowerBound])
}
