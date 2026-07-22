import Foundation
@testable import MLXFastCore
@testable import MLXFastModel
@testable import MLXFastRuntimeWorkerSupport
import Testing

@Test
func experimentalMTPSerialFallbackChainsTokensAndOffsetsExactly() throws {
    var inputTokens: [Int] = []
    var offsets: [Int] = []
    let tokens = try Gemma4TargetBlockGeneration.generateSerialBlock(
        previousToken: 10,
        maxBlockSize: 4,
        positionOffset: 512
    ) { inputToken, offset in
        inputTokens.append(inputToken)
        offsets.append(offset)
        return inputToken + 1
    }

    #expect(tokens == [11, 12, 13, 14])
    #expect(inputTokens == [10, 11, 12, 13])
    #expect(offsets == [512, 513, 514, 515])
}

@Test
func experimentalMTPSerialFallbackRejectsInvalidBounds() {
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4TargetBlockGeneration.generateSerialBlock(
            previousToken: 1,
            maxBlockSize: 0,
            positionOffset: 0
        ) { token, _ in token }
    }
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4TargetBlockGeneration.generateSerialBlock(
            previousToken: 1,
            maxBlockSize: MLXFastConstants.experimentalMTPMaxBlockSize + 1,
            positionOffset: 0
        ) { token, _ in token }
    }
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4TargetBlockGeneration.generateSerialBlock(
            previousToken: MLXFastConstants.vocabSize,
            maxBlockSize: 1,
            positionOffset: 0
        ) { token, _ in token }
    }
}

@Test
func shippedCheckpointReportsAssistantUnavailable() {
    let availability = Gemma4MTPAssistantAvailability.shippedCheckpoint
    #expect(!availability.isAvailable)
    #expect(availability.reason.contains("unavailable/incompatible"))
    #expect(availability.reason.contains("base checkpoint"))
    #expect(availability.reason.contains("IT models"))
    #expect(throws: MLXFastError.self) {
        try availability.requireAvailable()
    }
}

@Test
func decodeBlockRequestCodableShapeHasNoOracle() throws {
    let request = RuntimeWorkerRequest(
        id: 7,
        kind: "decode_block",
        token: 42,
        maxBlockSize: 4
    )
    let data = try JSONEncoder().encode(request)
    let object = try #require(
        JSONSerialization.jsonObject(with: data) as? [String: Any]
    )

    #expect(Set(object.keys) == Set(["id", "kind", "token", "max_block_size"]))
    #expect(object["expected_tokens"] == nil)
    #expect(object["expected_seed_token"] == nil)
    #expect(object["prompt_tokens"] == nil)
    #expect(data.count < BufferedFileLineReader.defaultMaximumLineByteCount)

    let decoded = try JSONDecoder().decode(RuntimeWorkerRequest.self, from: data)
    #expect(decoded.id == 7)
    #expect(decoded.kind == "decode_block")
    #expect(decoded.token == 42)
    #expect(decoded.maxBlockSize == 4)

    let requestWithOracleField = Data(
        """
        {"id":7,"kind":"decode_block","token":42,"max_block_size":4,"expected_tokens":[1,2]}
        """.utf8
    )
    #expect(throws: DecodingError.self) {
        _ = try JSONDecoder().decode(
            RuntimeWorkerRequest.self,
            from: requestWithOracleField
        )
    }
}

@Test
func legacyDecodeStepRequestShapeRemainsUnchanged() throws {
    let request = RuntimeWorkerRequest(
        id: 8,
        kind: "decode_step",
        token: 17
    )
    let data = try JSONEncoder().encode(request)
    let object = try #require(
        JSONSerialization.jsonObject(with: data) as? [String: Any]
    )

    #expect(Set(object.keys) == Set(["id", "kind", "token"]))
    #expect(object["max_block_size"] == nil)
    let decoded = try JSONDecoder().decode(RuntimeWorkerRequest.self, from: data)
    #expect(decoded.maxBlockSize == nil)
}

@Test
func decodeBlockResponseCodableShapeIsBounded() throws {
    let response = RuntimeWorkerResponse(
        id: 9,
        nonce: "nonce",
        ok: true,
        tokens: [1, 2, 3, 4]
    )
    let data = try JSONEncoder().encode(response)
    let object = try #require(
        JSONSerialization.jsonObject(with: data) as? [String: Any]
    )

    #expect(Set(object.keys) == Set(["id", "nonce", "ok", "tokens"]))
    #expect(data.count < BufferedFileLineReader.defaultMaximumLineByteCount)
}

@Test
func decodeBlockRequestValidationFailsClosed() throws {
    let valid = RuntimeWorkerRequest(
        id: 1,
        kind: "decode_block",
        token: 12,
        maxBlockSize: 4
    )
    #expect(
        try validateExperimentalDecodeBlockRequest(
            valid,
            decodedTokenCount: 124
        ) == ExperimentalDecodeBlockRequest(previousToken: 12, maxBlockSize: 4)
    )

    let invalidRequests = [
        RuntimeWorkerRequest(
            id: 1,
            kind: "decode_block",
            token: 12,
            maxBlockSize: 0
        ),
        RuntimeWorkerRequest(
            id: 1,
            kind: "decode_block",
            token: 12,
            maxBlockSize: 5
        ),
        RuntimeWorkerRequest(
            id: 1,
            kind: "decode_block",
            promptTokens: [1],
            token: 12,
            maxBlockSize: 1
        ),
        RuntimeWorkerRequest(
            id: 0,
            kind: "decode_block",
            token: 12,
            maxBlockSize: 1
        ),
        RuntimeWorkerRequest(
            id: 1,
            kind: "decode_block",
            token: MLXFastConstants.vocabSize,
            maxBlockSize: 1
        ),
    ]
    for request in invalidRequests {
        #expect(throws: MLXFastError.self) {
            _ = try validateExperimentalDecodeBlockRequest(
                request,
                decodedTokenCount: 0
            )
        }
    }
    #expect(throws: MLXFastError.self) {
        _ = try validateExperimentalDecodeBlockRequest(
            RuntimeWorkerRequest(
                id: 1,
                kind: "decode_block",
                token: 12,
                maxBlockSize: 2
            ),
            decodedTokenCount:
                MLXFastConstants.experimentalMTPMaxConfiguredTotalTokens - 1
        )
    }
}

@Test
func trustedMTPBlockValidatorAcceptsOnlyExactOrderedPrefix() throws {
    var validator = try ExperimentalMTPBlockValidator(
        expectedTokens: [10, 11, 12, 13, 14],
        totalTokenCount: 5,
        protocolMaxBlockSize: 4
    )
    try validator.accept([10, 11], requestedMaxBlockSize: 4)
    try validator.accept([12, 13, 14], requestedMaxBlockSize: 3)
    try validator.requireComplete()

    #expect(validator.committedTokenCount == 5)
    #expect(validator.remainingTokenCount == 0)
}

@Test
func trustedMTPBlockValidatorExercisesConfiguredTailLengths() throws {
    let expectedFinalRequests = [255: 3, 256: 4, 257: 1]
    for totalTokenCount in [255, 256, 257] {
        let expectedTokens = Array(0..<totalTokenCount)
        var validator = try ExperimentalMTPBlockValidator(
            expectedTokens: expectedTokens,
            totalTokenCount: totalTokenCount,
            protocolMaxBlockSize: 4
        )
        var finalRequest = 0
        while validator.remainingTokenCount > 0 {
            let requested = min(4, validator.remainingTokenCount)
            let start = validator.committedTokenCount
            try validator.accept(
                Array(expectedTokens[start..<(start + requested)]),
                requestedMaxBlockSize: requested
            )
            finalRequest = requested
        }
        try validator.requireComplete()
        #expect(finalRequest == expectedFinalRequests[totalTokenCount]!)
        #expect(validator.committedTokenCount == totalTokenCount)
    }
}

@Test
func trustedMTPBlockValidatorRejectsEmptyAndOversizedBlocks() throws {
    var emptyValidator = try ExperimentalMTPBlockValidator(
        expectedTokens: [1, 2, 3],
        totalTokenCount: 3
    )
    #expect(throws: MLXFastError.self) {
        try emptyValidator.accept([], requestedMaxBlockSize: 2)
    }

    var oversizedValidator = try ExperimentalMTPBlockValidator(
        expectedTokens: [1, 2, 3],
        totalTokenCount: 3
    )
    #expect(throws: MLXFastError.self) {
        try oversizedValidator.accept([1, 2, 3], requestedMaxBlockSize: 2)
    }
}

@Test
func trustedMTPBlockValidatorRejectsOverrunAndMismatch() throws {
    var overrunValidator = try ExperimentalMTPBlockValidator(
        expectedTokens: [1, 2, 3],
        totalTokenCount: 3
    )
    try overrunValidator.accept([1, 2], requestedMaxBlockSize: 2)
    #expect(throws: MLXFastError.self) {
        try overrunValidator.accept([3, 4], requestedMaxBlockSize: 2)
    }

    var mismatchValidator = try ExperimentalMTPBlockValidator(
        expectedTokens: [1, 2, 3],
        totalTokenCount: 3
    )
    do {
        try mismatchValidator.accept([1, 99], requestedMaxBlockSize: 2)
        Issue.record("expected token mismatch")
    } catch let mismatch as BenchmarkTokenMismatchError {
        #expect(mismatch.step == 1)
        #expect(mismatch.description == "experimental MTP decode token mismatch at step 1")
    }
    #expect(mismatchValidator.committedTokenCount == 0)
}

@Test
func experimentalMTPUsesTrustedConfigurableLengthAndBoundedBlocks() throws {
    #expect(MLXFastConstants.experimentalMTPMaxBlockSize == 4)
    #expect(MLXFastConstants.experimentalMTPMaxTotalTokens == 128)
    // 1,536 = 3x Laguna's 512-position sliding window: the trusted-parent
    // maximum keeps the untimed correctness legs able to wrap the rotating
    // cache while ranked timed decode stays fixed at 512 by workflow env.
    #expect(
        MLXFastConstants.experimentalMTPMaxConfiguredTotalTokens == 1_536
    )
    #expect(
        MLXFastConstants.experimentalMTPMaxTotalTokens
            == MLXFastConstants.benchmarkDecodeSteps
    )
    try GemmaRuntime.validateExperimentalMTPProbeOptions(
        ExperimentalMTPProbeOptions(
            weightsPath: "weights",
            goldenPath: "public.json",
            maxBlockSize: 4,
            totalTokenCount: 257
        )
    )
    #expect(throws: MLXFastError.self) {
        try GemmaRuntime.validateExperimentalMTPProbeOptions(
            ExperimentalMTPProbeOptions(
                weightsPath: "weights",
                goldenPath: "public.json",
                maxBlockSize: 5
            )
        )
    }
    #expect(throws: MLXFastError.self) {
        try GemmaRuntime.validateExperimentalMTPProbeOptions(
            ExperimentalMTPProbeOptions(
                weightsPath: "weights",
                goldenPath: "public.json",
                maxBlockSize: 4,
                totalTokenCount:
                    MLXFastConstants.experimentalMTPMaxConfiguredTotalTokens + 1
            )
        )
    }
}

@Test
func experimentalMTPCLIIsExplicitAndDoesNotAlterBenchmarkCommand() throws {
    let source = try String(
        contentsOfFile: "Sources/MLXFastCLI/main.swift",
        encoding: .utf8
    )
    #expect(source.contains("case \"mtp-probe\":"))
    #expect(
        source.contains(
            "mlxfast-swift mtp-probe --weights PATH --golden PATH [--block-size N] [--tokens N]"
        )
    )
    #expect(source.contains("optionName: \"--tokens\""))
    #expect(source.contains("mtp-probe requires --weights PATH"))
    #expect(source.contains("mtp-probe requires --golden PATH"))
    #expect(!source.contains("MLXFAST_MTP"))

    let benchmarkStart = try #require(
        source.range(of: "private static func runBenchmark(")
    )
    let probeStart = try #require(
        source.range(
            of: "private static func runExperimentalMTPProbe(",
            range: benchmarkStart.upperBound..<source.endIndex
        )
    )
    let benchmarkBody = source[benchmarkStart.lowerBound..<probeStart.lowerBound]
    #expect(!benchmarkBody.contains("experimentalMTPProbe"))
    #expect(!benchmarkBody.contains("decodeBlock"))
    #expect(!benchmarkBody.contains("mtp-probe"))
}

@Test
func experimentalMTPReportCannotRepresentOfficialScore() throws {
    let report = ExperimentalMTPProbeReport(
        experimental: true,
        protocolName: "decode_block_v1",
        generator: "serial_target_fallback",
        usesTrainedDrafter: false,
        assistantUnavailableReason: "unavailable",
        oracleSource: "first_golden_case",
        seedTokenCount: 512,
        decodeTokenCount: 128,
        maxBlockSize: 4,
        blockRequestCount: 32,
        elapsedSeconds: 1,
        seedPrefillSeconds: 0.2,
        blockDecodeSeconds: 0.8,
        meanBlockRequestSeconds: 0.025,
        p50BlockRequestSeconds: 0.024,
        maxBlockRequestSeconds: 0.03,
        parentMeasuredSecondsPerToken: 1.0 / 128.0,
        peakRamGB: 18,
        mlxActiveMemoryBytes: 17,
        mlxCacheMemoryBytes: 1,
        mlxPeakMemoryBytes: 18,
        allTokensMatched: true,
        officialScoreProduced: false
    )
    let object = try #require(
        JSONSerialization.jsonObject(with: JSONEncoder().encode(report))
            as? [String: Any]
    )
    #expect(object["score"] == nil)
    #expect(object["speedup"] == nil)
    #expect(object["official_score_produced"] as? Bool == false)
}

@Test
func trainedMTPRequestCarriesOnlyCommittedTokenAndBound() throws {
    let request = RuntimeWorkerRequest(
        id: 3,
        kind: "mtp_decode_block",
        token: 123,
        maxBlockSize: 4
    )
    let data = try JSONEncoder().encode(request)
    let object = try #require(
        JSONSerialization.jsonObject(with: data) as? [String: Any]
    )
    #expect(
        Set(object.keys)
            == Set(["id", "kind", "token", "max_block_size"])
    )
    #expect(object["prompt_tokens"] == nil)
    #expect(object["expected_tokens"] == nil)
    #expect(object["oracle"] == nil)
    #expect(object["accepted_count"] == nil)
    #expect(object["seconds"] == nil)

    let futureOracle = Data(
        """
        {"id":3,"kind":"mtp_decode_block","token":123,"max_block_size":4,"future_oracle":[5]}
        """.utf8
    )
    #expect(throws: DecodingError.self) {
        _ = try JSONDecoder().decode(
            RuntimeWorkerRequest.self,
            from: futureOracle
        )
    }
}

@Test
func trainedMTPBlockRequestRejectsOverrunAndCrossKindFields() throws {
    let valid = RuntimeWorkerRequest(
        id: 1,
        kind: "mtp_decode_block",
        token: 7,
        maxBlockSize: 4
    )
    #expect(
        try validateExperimentalTrainedMTPBlockRequest(
            valid,
            decodedTokenCount: 124
        ) == ExperimentalTrainedMTPBlockRequest(
            previousToken: 7,
            maxBlockSize: 4
        )
    )

    #expect(throws: MLXFastError.self) {
        _ = try validateExperimentalTrainedMTPBlockRequest(
            RuntimeWorkerRequest(
                id: 1,
                kind: "mtp_decode_block",
                token: 7,
                maxBlockSize: 2
            ),
            decodedTokenCount:
                MLXFastConstants.experimentalMTPMaxConfiguredTotalTokens - 1
        )
    }
    #expect(throws: MLXFastError.self) {
        _ = try validateExperimentalTrainedMTPBlockRequest(
            RuntimeWorkerRequest(
                id: 1,
                kind: "mtp_decode_block",
                promptTokens: [1],
                token: 7,
                maxBlockSize: 1
            ),
            decodedTokenCount: 0
        )
    }
}

@Test
func workerResponseRejectsFakeCountsTimingAndAcceptance() {
    let forgedFields = [
        "accepted_count": "4",
        "token_count": "128",
        "seconds": "0.000001",
        "future_tokens": "[1,2,3]",
    ]
    for (field, value) in forgedFields {
        let data = Data(
            """
            {"id":1,"nonce":"n","ok":true,"tokens":[1],"\(field)":\(value)}
            """.utf8
        )
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder().decode(
                RuntimeWorkerResponse.self,
                from: data
            )
        }
    }
}

@Test
func workerResponseRejectsPartialJSON() {
    let partial = Data(
        #"{"id":1,"nonce":"n","ok":true,"tokens":["#.utf8
    )
    #expect(throws: DecodingError.self) {
        _ = try JSONDecoder().decode(
            RuntimeWorkerResponse.self,
            from: partial
        )
    }
}

@Test
func trainedMTPAcceptanceAccountingCoversZeroPartialFullAndTail() throws {
    var accounting = ExperimentalMTPAcceptanceAccumulator()
    try accounting.record(requestedMaxBlockSize: 4, returnedTokenCount: 1)
    try accounting.record(requestedMaxBlockSize: 4, returnedTokenCount: 2)
    try accounting.record(requestedMaxBlockSize: 4, returnedTokenCount: 4)
    try accounting.record(requestedMaxBlockSize: 1, returnedTokenCount: 1)

    #expect(accounting.proposedDraftTokenCount == 9)
    #expect(accounting.acceptedDraftTokenCount == 4)
    #expect(accounting.rejectedDraftTokenCount == 5)
    #expect(accounting.rollbackRoundCount == 2)
    #expect(accounting.zeroAcceptanceRoundCount == 1)
    #expect(accounting.fullAcceptanceRoundCount == 1)
    #expect(accounting.targetOnlyTailTokenCount == 1)

    #expect(throws: MLXFastError.self) {
        try accounting.record(
            requestedMaxBlockSize: 4,
            returnedTokenCount: 5
        )
    }
}

@Test
func trainedMTPMemoryDiagnosticsRemainNonScoringProtocolFields() throws {
    let response = RuntimeWorkerResponse(
        id: 5,
        nonce: "nonce",
        ok: true,
        peakRamGB: 21.5,
        mlxActiveMemoryBytes: 20,
        mlxCacheMemoryBytes: 3,
        mlxPeakMemoryBytes: 24,
        targetVerificationMode: "exact-pair",
        exactPairSegmentCount: 31,
        exactPairRollbackRowCount: 2,
        serialVerificationRowCount: 1
    )
    let data = try JSONEncoder().encode(response)
    let object = try #require(
        JSONSerialization.jsonObject(with: data) as? [String: Any]
    )
    #expect(object["peak_ram_gb"] as? Double == 21.5)
    #expect(object["mlx_active_memory_bytes"] as? Int == 20)
    #expect(object["mlx_cache_memory_bytes"] as? Int == 3)
    #expect(object["mlx_peak_memory_bytes"] as? Int == 24)
    #expect(object["target_verification_mode"] as? String == "exact-pair")
    #expect(object["exact_pair_segment_count"] as? Int == 31)
    #expect(object["exact_pair_rollback_row_count"] as? Int == 2)
    #expect(object["serial_verification_row_count"] as? Int == 1)
    #expect(object["score"] == nil)
    #expect(object["seconds"] == nil)

    let decoded = try JSONDecoder().decode(
        RuntimeWorkerResponse.self,
        from: data
    )
    #expect(decoded.mlxPeakMemoryBytes == 24)
    #expect(decoded.targetVerificationMode == "exact-pair")
    #expect(decoded.exactPairSegmentCount == 31)
    #expect(decoded.exactPairRollbackRowCount == 2)
    #expect(decoded.serialVerificationRowCount == 1)
}

// The MTP ranked track is retired: its contract fixture
// (fixtures/laguna_xs_2_1_mtp_track.json), weight manifests, manifests, and
// local scripts are deleted, so the contract-pinning tests retired with
// them. The MTP protocol/validator unit tests below remain because the
// worker protocol code they cover still compiles into the runtime worker.

@Test
func trainedMTPOptionsFailClosedWithoutAssistantRequirement() throws {
    let missingRequirement = ExperimentalTrainedMTPOptions(
        sourceTargetPath: "source-target",
        targetWeightsPath: "mtp-weights",
        assistantPath: "assistant",
        contractPath: "contract.json",
        goldenPath: "it-golden.json",
        maxBlockSize: 4,
        requireTrainedAssistant: false
    )
    #expect(throws: MLXFastError.self) {
        try GemmaRuntime.validateExperimentalTrainedMTPOptions(
            missingRequirement
        )
    }

    let valid = ExperimentalTrainedMTPOptions(
        sourceTargetPath: "source-target",
        targetWeightsPath: "mtp-weights",
        assistantPath: "assistant",
        contractPath: "contract.json",
        goldenPath: "it-golden.json",
        maxBlockSize: 4,
        totalTokenCount: 257,
        requireTrainedAssistant: true
    )
    try GemmaRuntime.validateExperimentalTrainedMTPOptions(valid)
    try GemmaRuntime.validateExperimentalTrainedMTPOptions(
        ExperimentalTrainedMTPOptions(
            sourceTargetPath: "source-target",
            targetWeightsPath: "mtp-weights",
            assistantPath: "assistant",
            contractPath: "contract.json",
            goldenPath: "it-golden.json",
            maxBlockSize: 2,
            totalTokenCount: 255,
            verificationMode: .serial,
            requireTrainedAssistant: true
        )
    )

    // K=1 cannot draft, so the trained-assistant benchmark rejects it; the
    // serial K=1 control remains mtp-probe.
    #expect(throws: MLXFastError.self) {
        try GemmaRuntime.validateExperimentalTrainedMTPOptions(
            ExperimentalTrainedMTPOptions(
                sourceTargetPath: "source-target",
                targetWeightsPath: "mtp-weights",
                assistantPath: "assistant",
                contractPath: "contract.json",
                goldenPath: "it-golden.json",
                maxBlockSize: 1,
                totalTokenCount: 255,
                requireTrainedAssistant: true
            )
        )
    }

    #expect(throws: MLXFastError.self) {
        try GemmaRuntime.validateExperimentalTrainedMTPOptions(
            ExperimentalTrainedMTPOptions(
                sourceTargetPath: "source-target",
                targetWeightsPath: "mtp-weights",
                assistantPath: "assistant",
                contractPath: "contract.json",
                goldenPath: "it-golden.json",
                maxBlockSize: 4,
                totalTokenCount: 1_537,
                requireTrainedAssistant: true
            )
        )
    }
}

@Test
func trainedMTPTrackIsExplicitAndSerialBenchmarkRemainsUnchanged() throws {
    let trustedSource = try String(
        contentsOfFile: "Sources/MLXFastCLI/main.swift",
        encoding: .utf8
    )
    let workerSource = try String(
        contentsOfFile: "Sources/MLXFastRuntimeWorkerCLI/main.swift",
        encoding: .utf8
    )
    #expect(trustedSource.contains("case \"mtp-benchmark\":"))
    #expect(!trustedSource.contains("case \"runtime-worker\":"))
    #expect(!trustedSource.contains("case \"mtp-runtime-worker\":"))
    #expect(workerSource.contains("case \"runtime-worker\":"))
    #expect(workerSource.contains("case \"preflight\":"))
    #expect(workerSource.contains("case \"mtp-runtime-worker\":"))
    #expect(workerSource.contains("GemmaRuntime.runPreflightWorker"))
    #expect(trustedSource.contains("GemmaRuntime.runPreflightWithWorker"))
    #expect(workerSource.contains("mlxfast-runtime-worker runtime-worker"))
    #expect(workerSource.contains("[\"help\", \"--help\", \"-h\"]"))
    #expect(trustedSource.contains("Bundle.main.executableURL"))
    #expect(trustedSource.contains("_NSGetExecutablePath"))
    #expect(trustedSource.contains("try currentExecutablePath()"))
    #expect(trustedSource.contains("--require-trained-assistant"))
    #expect(trustedSource.contains("[--block-size N] [--tokens N]"))
    #expect(trustedSource.contains("[--target-verification exact-pair|serial]"))
    #expect(trustedSource.contains("mtp-benchmark requires --target-source PATH"))
    #expect(trustedSource.contains("mtp-benchmark requires --assistant PATH"))

    let benchmarkStart = try #require(
        trustedSource.range(of: "private static func runBenchmark(")
    )
    let trainedStart = try #require(
        trustedSource.range(
            of: "private static func runExperimentalTrainedMTPBenchmark(",
            range: benchmarkStart.upperBound..<trustedSource.endIndex
        )
    )
    let serialBenchmarkBody =
        trustedSource[benchmarkStart.lowerBound..<trainedStart.lowerBound]
    #expect(!serialBenchmarkBody.contains("experimentalTrainedMTPBenchmark"))
    #expect(!serialBenchmarkBody.contains("--assistant"))
    #expect(!serialBenchmarkBody.contains("mtp_decode_block"))
    #expect(!serialBenchmarkBody.contains("--target-verification"))
}

@Test
func trainedMTPUsesExplicitExactPairVerificationAndSerialControl() throws {
    let adapter = try String(
        contentsOfFile: "Sources/MLXFastModel/Gemma4MTPRuntime.swift",
        encoding: .utf8
    )
    let engine = try String(
        contentsOfFile: "Sources/MLXFastModel/Gemma4FastEngine.swift",
        encoding: .utf8
    )
    let worker = try String(
        contentsOfFile:
            "Sources/MLXFastHarness/GemmaRuntimeMTPWorker.swift",
        encoding: .utf8
    )
    #expect(adapter.contains("fastMTPForward(tokens, cache: cache)"))
    #expect(adapter.contains("newCache(parameters: parameters)"))
    #expect(adapter.contains("case .exactPair:"))
    #expect(adapter.contains("verifyWithExactPairs("))
    #expect(adapter.contains("target.exactMTPPair("))
    #expect(adapter.contains("Gemma4SharedKV.sliceTail("))
    #expect(adapter.contains("case .serial:"))
    #expect(adapter.contains("verifySerially("))
    #expect(!adapter.contains("replayRejectedRoundExactly"))
    #expect(!adapter.contains("PromptLookup"))
    #expect(engine.contains("func forwardForMTP("))
    #expect(engine.contains("func exactMTPPair("))
    #expect(engine.contains("gemma4ExactTwoTokenAttention("))
    #expect(engine.contains("exactTwoVectorPacked13Softcapped("))
    #expect(engine.contains("capturedSharedKV: Gemma4SharedKV("))
    #expect(engine.contains("let preNorm = inputs.dim(1) > 16"))
    #expect(worker.contains("let warmupSession"))
    #expect(worker.contains("repeating: 2"))
    #expect(worker.contains("verificationMode: verificationMode"))
    #expect(worker.contains("try target.warmMTPTargetKernels("))
    #expect(
        worker.contains("includeExactPair: verificationMode == .exactPair")
    )
    // The fixed warmup must not depend on assistant acceptance: four serial
    // rows always, two exact pairs when selected, plus the deep 1,024-key
    // sliding-attention switch.
    #expect(adapter.contains("for _ in 0..<4"))
    #expect(adapter.contains("for _ in 0..<2"))
    #expect(adapter.contains("warmDeepSlidingDecodeAttention()"))
    let warmup = try #require(worker.range(of: "let warmupSession"))
    let protocolHello = try #require(
        worker.range(of: "let sessionNonce = generateRuntimeWorkerNonce()")
    )
    #expect(warmup.lowerBound < protocolHello.lowerBound)
}

// setup.sh must stay free of the retired MTP track's provisioning: no MTP
// cache namespace and no assistant download path. (setup-mtp.sh itself is
// retired and deleted; DefaultTrackTests pins its absence.)
@Test
func setupStaysFreeOfRetiredMTPProvisioning() throws {
    let normalSetup = try String(
        contentsOfFile: "setup.sh",
        encoding: .utf8
    )
    #expect(!normalSetup.contains("laguna-xs-2.1-mtp-v1"))
    #expect(!normalSetup.contains("assistant"))
    #expect(!normalSetup.contains("setup-mtp.sh"))
}
