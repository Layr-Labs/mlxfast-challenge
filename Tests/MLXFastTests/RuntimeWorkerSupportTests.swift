import Dispatch
import Foundation
import MLX
@testable import MLXFastRuntimeWorkerSupport
import Testing

@Test
func lagunaCorrectnessSelectsGreedyTokenWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    #expect(try LagunaCorrectness.greedyToken(
        from: MLXArray([Float(0.1), 2.0, 1.0], [3])
    ) == 1)
    #expect(try LagunaCorrectness.greedyToken(
        from: MLXArray([Float(1), 2, 3, 2], [2, 2])
    ) == 0)
}

// The worker's orphan self-reaper is what frees the ~17 GB model residency
// when the harness parent dies while the worker is not blocked on protocol
// stdin (most importantly during the minutes-long model load).
@Test
func runtimeWorkerOrphanReaperFiresOnceTheParentIsGone() {
    let fired = DispatchSemaphore(value: 0)
    let thread = LagunaRuntime.startRuntimeWorkerOrphanReaper(
        pollIntervalSeconds: 0.01,
        isOrphaned: { true },
        onOrphaned: { fired.signal() }
    )
    defer { thread.cancel() }
    #expect(fired.wait(timeout: .now() + 2) == .success)
}

@Test
func runtimeWorkerOrphanReaperStaysQuietWhileTheParentIsAlive() {
    let fired = DispatchSemaphore(value: 0)
    let thread = LagunaRuntime.startRuntimeWorkerOrphanReaper(
        pollIntervalSeconds: 0.01,
        isOrphaned: { false },
        onOrphaned: { fired.signal() }
    )
    defer { thread.cancel() }
    #expect(fired.wait(timeout: .now() + 0.3) == .timedOut)
}

@Test
func phaseStartAllocatorResetLeavesExactlyEmptyCacheWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }
    Memory.cacheLimit = 32 << 30
    do {
        let scratch = MLXArray(Array(repeating: Float(1), count: 1 << 20), [1024, 1024])
        eval(scratch + scratch)
    }
    try LagunaRuntime.resetRuntimeWorkerAllocatorForPhaseStart()
    #expect(Memory.cacheMemory == 0)
    #expect(Memory.cacheLimit == LagunaRuntime.trustedRuntimeWorkerPhaseStartCacheLimitBytes)
}

@Test
func traceProtocolCarriesRequestedTopKAndExpectedTokenDiagnostics() throws {
    let request = RuntimeWorkerRequest(
        id: 4,
        kind: "correctness_step",
        token: 7,
        topK: 16,
        expectedToken: 23
    )
    let requestObject = try #require(
        JSONSerialization.jsonObject(
            with: JSONEncoder().encode(request)
        ) as? [String: Any]
    )
    #expect(requestObject["top_k"] as? Int == 16)
    #expect(requestObject["expected_token"] as? Int == 23)

    let response = RuntimeWorkerResponse(
        id: 4,
        nonce: "nonce",
        ok: true,
        token: 1,
        topLogits: [
            CorrectnessTraceLogit(token: 1, logit: 9),
        ],
        expectedTokenLogit: 3.5,
        expectedTokenRank: 19,
        topLogitMargin: 0.25
    )
    let decoded = try JSONDecoder().decode(
        RuntimeWorkerResponse.self,
        from: JSONEncoder().encode(response)
    )
    #expect(decoded.expectedTokenLogit == 3.5)
    #expect(decoded.expectedTokenRank == 19)
    #expect(decoded.topLogitMargin == 0.25)
}

@Test
func workerComputesExpectedTokenDiagnosticsOutsideReturnedTopSubset() throws {
    let diagnostics = try LagunaRuntime.correctnessLogitDiagnostics(
        values: (0..<20).map { Double(20 - $0) },
        topK: 12,
        expectedToken: 19
    )
    #expect(diagnostics.topLogits.count == 12)
    #expect(!diagnostics.topLogits.contains { $0.token == 19 })
    #expect(diagnostics.expectedTokenLogit == 1)
    #expect(diagnostics.expectedTokenRank == 20)
    #expect(diagnostics.topLogitMargin == 1)
}
