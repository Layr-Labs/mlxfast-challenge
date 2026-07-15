import Foundation
import MLX
@testable import MLXFastCore
@testable import MLXFastHarness
@testable import MLXFastModel
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Test
func exactPairDecisionCoversFullPartialAndZeroAcceptance() throws {
    #expect(
        try gemma4ExactPairDecision(
            drafts: [11, 12],
            targetTokens: [11, 12]
        ) == Gemma4ExactPairDecision(
            acceptedDrafts: 2,
            committedTokens: [11, 12]
        )
    )
    #expect(
        try gemma4ExactPairDecision(
            drafts: [11, 12],
            targetTokens: [11, 99]
        ) == Gemma4ExactPairDecision(
            acceptedDrafts: 1,
            committedTokens: [11, 99]
        )
    )
    #expect(
        try gemma4ExactPairDecision(
            drafts: [11, 12],
            targetTokens: [98, 99]
        ) == Gemma4ExactPairDecision(
            acceptedDrafts: 0,
            committedTokens: [98]
        )
    )
    #expect(
        try gemma4ExactPairDecision(
            drafts: [11],
            targetTokens: [11, 99]
        ) == Gemma4ExactPairDecision(
            acceptedDrafts: 1,
            committedTokens: [11, 99]
        )
    )
    #expect(
        try gemma4ExactPairDecision(
            drafts: [11],
            targetTokens: [98, 99]
        ) == Gemma4ExactPairDecision(
            acceptedDrafts: 0,
            committedTokens: [98]
        )
    )
    #expect(throws: MLXFastError.self) {
        _ = try gemma4ExactPairDecision(
            drafts: [11, 12],
            targetTokens: [11]
        )
    }
    #expect(throws: MLXFastError.self) {
        _ = try gemma4ExactPairDecision(
            drafts: [],
            targetTokens: [11, 12]
        )
    }
}

@Test
func exactPairEligibilityRequiresEveryBitExactComponent() {
    #expect(gemma4ExactTwoVectorShapeIsSupported([2, 5_376], width: 5_376))
    #expect(!gemma4ExactTwoVectorShapeIsSupported([1, 2, 5_376], width: 5_376))
    #expect(
        gemma4ExactTwoVectorLayerIsEligible(
            hasQKV: true,
            hasAttentionPreparation: true,
            hasAttentionBoundary: true,
            hasNextBoundary: true,
            hasOutput: true,
            hasGateUp: true,
            hasDown: true,
            usesFusedActivation: true
        )
    )
    #expect(
        !gemma4ExactTwoVectorLayerIsEligible(
            hasQKV: true,
            hasAttentionPreparation: true,
            hasAttentionBoundary: true,
            hasNextBoundary: true,
            hasOutput: true,
            hasGateUp: false,
            hasDown: true,
            usesFusedActivation: true
        )
    )
}

@Test
func exactPairAttentionMatchesSerialRowsAcrossSlidingDispatchBoundary() {
    guard ProcessInfo.processInfo.environment[
        "MLXFAST_RUN_MLX_RUNTIME_TESTS"
    ] == "1" else {
        return
    }

    for (headDimension, keyCount) in [
        (256, 514),
        (256, 1_023),
        (256, 1_024),
        (512, 514),
    ] {
        let keyElements = keyCount * headDimension
        let keyPositions = MLXArray(0..<keyElements).asType(.float32)
        let keysWithDraft = sin(keyPositions * 0.000_976_562_5)
            .asType(.bfloat16)
            .reshaped(1, 1, keyCount, headDimension)
        let valuesWithDraft = cos(keyPositions * 0.001_953_125)
            .asType(.bfloat16)
            .reshaped(1, 1, keyCount, headDimension)
        let keysBeforeDraft = keysWithDraft[
            0..., 0..., 0..<(keyCount - 1), 0...
        ]
        let valuesBeforeDraft = valuesWithDraft[
            0..., 0..., 0..<(keyCount - 1), 0...
        ]
        let queryPositions = MLXArray(0..<(2 * headDimension)).asType(.float32)
        let queries = sin(queryPositions * 0.003_906_25)
            .asType(.bfloat16)
            .reshaped(1, 1, 2, headDimension)

        let candidate = gemma4ExactTwoTokenAttention(
            queries: queries,
            keysBeforeDraft: keysBeforeDraft,
            valuesBeforeDraft: valuesBeforeDraft,
            keysWithDraft: keysWithDraft,
            valuesWithDraft: valuesWithDraft,
            scale: 1.0
        )
        let serialFirst = MLXFast.scaledDotProductAttention(
            queries: queries[0..., 0..., 0..<1, 0...],
            keys: keysBeforeDraft,
            values: valuesBeforeDraft,
            scale: 1.0,
            mask: .none
        )
        let serialSecond = MLXFast.scaledDotProductAttention(
            queries: queries[0..., 0..., 1..<2, 0...],
            keys: keysWithDraft,
            values: valuesWithDraft,
            scale: 1.0,
            mask: .none
        )
        expectExactBits(
            candidate,
            concatenated([serialFirst, serialSecond], axis: 2),
            dtype: .uint16,
            context:
                "attention D=\(headDimension) keys=\(keyCount)"
        )
    }
}

@Test
func exactPairRuntimeMatchesTwoSerialRowsBitForBit() throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["MLXFAST_RUN_MTP_EXACT_PAIR_TESTS"] == "1" else {
        return
    }
    let weightsPath = try #require(
        environment["MLXFAST_MTP_WEIGHTS_PATH"]
    )
    let goldenPath = try #require(
        environment["MLXFAST_MTP_PAIR_GOLDEN_PATH"]
    )
    let golden = try loadGoldenFixture(
        from: goldenPath,
        requiredSteps: 5,
        requiredPromptTokens: MLXFastConstants.correctnessPromptTokens
    )
    let testCase = try #require(golden.cases.first)
    let config = try Gemma4Config.load(from: weightsPath)
    let loader = try Gemma4WeightLoader(weightsPath: weightsPath)
    let weightCache = Gemma4RuntimeWeightCache(
        loader: loader,
        config: config
    )
    let model = try weightCache.requireLibraryModel()

    let pairCache = model.newCache(parameters: nil)
    let serialCache = model.newCache(parameters: nil)
    let rollbackCache = model.newCache(parameters: nil)
    let serialPrefixCache = model.newCache(parameters: nil)
    let prompt = MLXArray(
        testCase.promptTokens.map(Int32.init),
        [1, testCase.promptTokens.count]
    )
    let pairPrefill = try #require(model.fastMTPForward(prompt, cache: pairCache))
    let serialPrefill = try #require(model.fastMTPForward(prompt, cache: serialCache))
    _ = try #require(model.fastMTPForward(prompt, cache: rollbackCache))
    _ = try #require(model.fastMTPForward(prompt, cache: serialPrefixCache))
    let seedArray = pairPrefill.logits[
        0..., -1, 0...
    ].asType(.float32).argMax(axis: -1)
    let serialSeedArray = serialPrefill.logits[
        0..., -1, 0...
    ].asType(.float32).argMax(axis: -1)
    eval(seedArray, serialSeedArray)
    let seed = Int(seedArray.item(Int32.self))
    #expect(seed == testCase.expectedTokens[0])
    #expect(seedArray.item(Int32.self) == serialSeedArray.item(Int32.self))

    let firstDraft = testCase.expectedTokens[1]
    let pairInputs = MLXArray(
        [Int32(seed), Int32(firstDraft)],
        [1, 2]
    )
    let pair = try #require(
        model.exactMTPPair(pairInputs, cache: pairCache)
    )
    let rollbackPair = try #require(
        model.exactMTPPair(pairInputs, cache: rollbackCache)
    )
    let serial0 = try #require(
        model.fastMTPForward(
            MLXArray([Int32(seed)], [1, 1]),
            cache: serialCache
        )
    )
    let serialPrefix0 = try #require(
        model.fastMTPForward(
            MLXArray([Int32(seed)], [1, 1]),
            cache: serialPrefixCache
        )
    )
    let serial1 = try #require(
        model.fastMTPForward(
            MLXArray([Int32(firstDraft)], [1, 1]),
            cache: serialCache
        )
    )
    let pairTokens = pair.logits.asType(.float32).argMax(axis: -1)
    eval(pairTokens)
    #expect(
        pairTokens.asArray(Int32.self).map(Int.init)
            == Array(testCase.expectedTokens[1...2])
    )

    expectExactBits(
        pair.logits[0..., 0..<1, 0...],
        serial0.logits,
        dtype: .uint32,
        context: "row-0 logits"
    )
    expectExactBits(
        pair.logits[0..., 1..<2, 0...],
        serial1.logits,
        dtype: .uint32,
        context: "row-1 logits"
    )
    expectExactBits(
        pair.lastHidden[0..., 0..<1, 0...],
        serial0.lastHidden,
        dtype: .uint16,
        context: "row-0 pre-norm hidden"
    )
    expectExactBits(
        pair.lastHidden[0..., 1..<2, 0...],
        serial1.lastHidden,
        dtype: .uint16,
        context: "row-1 pre-norm hidden"
    )
    expectExactCacheState(
        pairCache,
        serialCache,
        context: "full pair commit"
    )

    let nextInput = testCase.expectedTokens[2]
    let nextDraft = testCase.expectedTokens[3]
    let pairAtNextOffset = try #require(
        model.exactMTPPair(
            MLXArray([Int32(nextInput), Int32(nextDraft)], [1, 2]),
            cache: pairCache
        )
    )
    let serial2 = try #require(
        model.fastMTPForward(
            MLXArray([Int32(nextInput)], [1, 1]),
            cache: serialCache
        )
    )
    let serial3 = try #require(
        model.fastMTPForward(
            MLXArray([Int32(nextDraft)], [1, 1]),
            cache: serialCache
        )
    )
    let nextPairTokens = pairAtNextOffset.logits
        .asType(.float32)
        .argMax(axis: -1)
    eval(nextPairTokens)
    #expect(
        nextPairTokens.asArray(Int32.self).map(Int.init)
            == Array(testCase.expectedTokens[3...4])
    )
    expectExactBits(
        pairAtNextOffset.logits[0..., 0..<1, 0...],
        serial2.logits,
        dtype: .uint32,
        context: "next-offset row-0 logits"
    )
    expectExactBits(
        pairAtNextOffset.logits[0..., 1..<2, 0...],
        serial3.logits,
        dtype: .uint32,
        context: "next-offset row-1 logits"
    )
    expectExactBits(
        pairAtNextOffset.lastHidden[0..., 0..<1, 0...],
        serial2.lastHidden,
        dtype: .uint16,
        context: "next-offset row-0 pre-norm hidden"
    )
    expectExactBits(
        pairAtNextOffset.lastHidden[0..., 1..<2, 0...],
        serial3.lastHidden,
        dtype: .uint16,
        context: "next-offset row-1 pre-norm hidden"
    )
    expectExactCacheState(
        pairCache,
        serialCache,
        context: "next-offset full pair commit"
    )

    for cache in rollbackCache {
        #expect(cache.trim(1) == 1)
    }
    let rolledShared = Gemma4SharedKV.sliceTail(
        from: rollbackPair.capturedSharedKV,
        rejected: 1
    )
    expectExactBits(
        rolledShared.fullAttention.0,
        serialPrefix0.capturedSharedKV.fullAttention.0,
        dtype: .uint16,
        context: "rolled full-attention keys"
    )
    expectExactBits(
        rolledShared.fullAttention.1,
        serialPrefix0.capturedSharedKV.fullAttention.1,
        dtype: .uint16,
        context: "rolled full-attention values"
    )
    expectExactBits(
        rolledShared.slidingAttention.0,
        serialPrefix0.capturedSharedKV.slidingAttention.0,
        dtype: .uint16,
        context: "rolled sliding-attention keys"
    )
    expectExactBits(
        rolledShared.slidingAttention.1,
        serialPrefix0.capturedSharedKV.slidingAttention.1,
        dtype: .uint16,
        context: "rolled sliding-attention values"
    )
    expectExactCacheState(
        rollbackCache,
        serialPrefixCache,
        context: "zero-acceptance rollback"
    )
    let resumedAfterRollback = try #require(
        model.fastMTPForward(
            MLXArray([Int32(firstDraft)], [1, 1]),
            cache: rollbackCache
        )
    )
    let serialAfterPrefix = try #require(
        model.fastMTPForward(
            MLXArray([Int32(firstDraft)], [1, 1]),
            cache: serialPrefixCache
        )
    )
    expectExactBits(
        resumedAfterRollback.logits,
        serialAfterPrefix.logits,
        dtype: .uint32,
        context: "post-rollback continuation logits"
    )
    expectExactBits(
        resumedAfterRollback.lastHidden,
        serialAfterPrefix.lastHidden,
        dtype: .uint16,
        context: "post-rollback continuation hidden"
    )
    expectExactCacheState(
        rollbackCache,
        serialPrefixCache,
        context: "post-rollback continuation cache"
    )
}

/// Forces zero, partial, full, and second-segment acceptance through the real
/// session verifier with fabricated drafts, so the seams do not depend on
/// assistant draft behavior. Every committed token, counter, and physical
/// cache byte must match an independent teacher-forced serial cache.
@Test
func exactPairSessionForcedAcceptanceSeamsMatchSerial() async throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["MLXFAST_RUN_MTP_EXACT_PAIR_TESTS"] == "1" else {
        return
    }
    let weightsPath = try #require(
        environment["MLXFAST_MTP_WEIGHTS_PATH"]
    )
    let assistantPath = try #require(
        environment["MLXFAST_MTP_ASSISTANT_DIR"]
    )
    let goldenPath = try #require(
        environment["MLXFAST_MTP_PAIR_GOLDEN_PATH"]
    )
    let golden = try loadGoldenFixture(
        from: goldenPath,
        requiredSteps: 17,
        requiredPromptTokens: MLXFastConstants.correctnessPromptTokens
    )
    let testCase = try #require(golden.cases.first)
    let expected = testCase.expectedTokens
    let config = try Gemma4Config.load(from: weightsPath)
    let loader = try Gemma4WeightLoader(weightsPath: weightsPath)
    let weightCache = Gemma4RuntimeWeightCache(
        loader: loader,
        config: config
    )
    let model = try weightCache.requireLibraryModel()
    let drafter = try await Gemma4AssistantDraftModel.load(
        from: URL(fileURLWithPath: assistantPath)
    )
    let session = try Gemma4TrainedMTPBlockSession(
        target: model,
        drafter: drafter,
        verificationMode: .exactPair
    )
    let seed = try session.begin(seedTokens: testCase.promptTokens)
    #expect(seed == expected[0])

    let serialCache = model.newCache(parameters: nil)
    _ = try #require(
        model.fastMTPForward(
            MLXArray(
                testCase.promptTokens.map(Int32.init),
                [1, testCase.promptTokens.count]
            ),
            cache: serialCache
        )
    )
    var serialInputs: [Int] = []
    func forwardSerial(_ tokens: [Int]) throws {
        for token in tokens {
            let row = try #require(
                model.fastMTPForward(
                    MLXArray([Int32(token)], [1, 1]),
                    cache: serialCache
                )
            )
            eval(row.logits)
            serialInputs.append(token)
        }
    }
    func mismatching(_ token: Int) -> Int {
        token == 0 ? 1 : 0
    }

    // Tail: K=1 is target-only, advances exactly one row, and never drafts.
    let targetOnlyTail = try session.generateBlock(
        previousCommittedToken: seed,
        maxBlockSize: 1
    )
    #expect(targetOnlyTail.tokens == [expected[1]])
    #expect(targetOnlyTail.acceptedDraftTokenCount == 0)
    #expect(!targetOnlyTail.usedDrafter)
    #expect(session.targetOnlyTailTokenCount == 1)
    #expect(session.exactPairSegmentCount == 0)
    #expect(session.serialVerificationRowCount == 1)
    try forwardSerial([seed])
    expectExactCacheState(
        session.targetCache,
        serialCache,
        context: "session target-only tail"
    )

    // Round A: K=2 full acceptance is exactly one two-row segment.
    let k2Block = try session.verifyWithExactPairs(
        previousToken: expected[1],
        drafts: [expected[2]]
    )
    #expect(k2Block.committedTokens == Array(expected[2...3]))
    #expect(k2Block.acceptedDrafts == 1)
    #expect(session.exactPairSegmentCount == 1)
    #expect(session.exactPairRollbackRowCount == 0)
    #expect(session.serialVerificationRowCount == 1)
    try forwardSerial(Array(expected[1...2]))
    expectExactCacheState(
        session.targetCache,
        serialCache,
        context: "session K=2 full acceptance"
    )

    // Round B: K=4 full acceptance composes two exact-pair segments.
    let fullBlock = try session.verifyWithExactPairs(
        previousToken: expected[3],
        drafts: Array(expected[4...6])
    )
    #expect(fullBlock.committedTokens == Array(expected[4...7]))
    #expect(fullBlock.acceptedDrafts == 3)
    #expect(session.exactPairSegmentCount == 3)
    #expect(session.exactPairRollbackRowCount == 0)
    #expect(session.serialVerificationRowCount == 1)
    try forwardSerial(Array(expected[3...6]))
    expectExactCacheState(
        session.targetCache,
        serialCache,
        context: "session full acceptance"
    )

    // Round C: zero acceptance rejects row zero and must physically remove
    // row one before continuing.
    let zeroBlock = try session.verifyWithExactPairs(
        previousToken: expected[7],
        drafts: [
            mismatching(expected[8]),
            mismatching(expected[9]),
            mismatching(expected[10]),
        ]
    )
    #expect(zeroBlock.committedTokens == [expected[8]])
    #expect(zeroBlock.acceptedDrafts == 0)
    #expect(session.exactPairSegmentCount == 4)
    #expect(session.exactPairRollbackRowCount == 1)
    try forwardSerial([expected[7]])
    expectExactCacheState(
        session.targetCache,
        serialCache,
        context: "session zero acceptance rollback"
    )

    // Round D: the second segment rejects its first row after a fully
    // accepted first segment.
    let secondSegmentBlock = try session.verifyWithExactPairs(
        previousToken: expected[8],
        drafts: [
            expected[9],
            expected[10],
            mismatching(expected[11]),
        ]
    )
    #expect(
        secondSegmentBlock.committedTokens == Array(expected[9...11])
    )
    #expect(secondSegmentBlock.acceptedDrafts == 2)
    #expect(session.exactPairSegmentCount == 6)
    #expect(session.exactPairRollbackRowCount == 2)
    try forwardSerial(Array(expected[8...10]))
    expectExactCacheState(
        session.targetCache,
        serialCache,
        context: "session second-segment rejection"
    )

    // Round E: row-one rejection keeps both physical rows because both pair
    // inputs are committed tokens.
    let partialBlock = try session.verifyWithExactPairs(
        previousToken: expected[11],
        drafts: [
            expected[12],
            mismatching(expected[13]),
            mismatching(expected[14]),
        ]
    )
    #expect(partialBlock.committedTokens == Array(expected[12...13]))
    #expect(partialBlock.acceptedDrafts == 1)
    #expect(session.exactPairSegmentCount == 7)
    #expect(session.exactPairRollbackRowCount == 2)
    #expect(session.serialVerificationRowCount == 1)
    try forwardSerial(Array(expected[11...12]))
    expectExactCacheState(
        session.targetCache,
        serialCache,
        context: "session row-one rejection"
    )

    // Round F: K=3 uses one exact pair plus one serial bonus row.
    let k3TailBlock = try session.verifyWithExactPairs(
        previousToken: expected[13],
        drafts: Array(expected[14...15])
    )
    #expect(k3TailBlock.committedTokens == Array(expected[14...16]))
    #expect(k3TailBlock.acceptedDrafts == 2)
    #expect(session.exactPairSegmentCount == 8)
    #expect(session.exactPairRollbackRowCount == 2)
    #expect(session.serialVerificationRowCount == 2)
    try forwardSerial(Array(expected[13...15]))
    expectExactCacheState(
        session.targetCache,
        serialCache,
        context: "session K=3 serial bonus tail"
    )

    // Physical row accounting across all rounds.
    let committedTotal = targetOnlyTail.tokens.count
        + k2Block.committedTokens.count
        + fullBlock.committedTokens.count
        + zeroBlock.committedTokens.count
        + secondSegmentBlock.committedTokens.count
        + partialBlock.committedTokens.count
        + k3TailBlock.committedTokens.count
    #expect(
        2 * session.exactPairSegmentCount
            - session.exactPairRollbackRowCount
            + session.serialVerificationRowCount
            == committedTotal
    )
    #expect(serialInputs.count == committedTotal)
    #expect(
        session.targetCache.allSatisfy {
            $0.offset == testCase.promptTokens.count + committedTotal
        }
    )
}

/// Drives identical fixed BOS token streams through the exact-pair path and a
/// serial mirror across the deep decode seams: the 768-position combined
/// cache growth, the odd-offset sliding refusal before the 1,024 wrap, the
/// library's 1,024-key sliding kernel switch, and post-wrap serial decode.
/// Inputs are constant, so this is pure physical/logit parity, not a language
/// test, and it needs no golden fixture.
@Test
func exactPairDeepOffsetsGrowthAndWrapMatchSerial() throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["MLXFAST_RUN_MTP_EXACT_PAIR_TESTS"] == "1",
          environment["MLXFAST_RUN_MTP_DEEP_OFFSET_TESTS"] == "1"
    else {
        return
    }
    let weightsPath = try #require(
        environment["MLXFAST_MTP_WEIGHTS_PATH"]
    )
    let config = try Gemma4Config.load(from: weightsPath)
    let loader = try Gemma4WeightLoader(weightsPath: weightsPath)
    let weightCache = Gemma4RuntimeWeightCache(
        loader: loader,
        config: config
    )
    let model = try weightCache.requireLibraryModel()

    let bos = Int32(2)
    let seedLength = MLXFastConstants.correctnessPromptTokens
    let prompt = MLXArray(
        Array(repeating: bos, count: seedLength),
        [1, seedLength]
    )
    let pairCache = model.newCache(parameters: nil)
    let serialCache = model.newCache(parameters: nil)
    _ = try #require(model.fastMTPForward(prompt, cache: pairCache))
    _ = try #require(model.fastMTPForward(prompt, cache: serialCache))

    let pairInput = MLXArray([bos, bos], [1, 2])
    let singleInput = MLXArray([bos], [1, 1])
    let slidingWindow = model.configuration.slidingWindow
    let checkpointOffsets: Set<Int> = [
        seedLength + 254,
        seedLength + 256,
        seedLength + 258,
        seedLength + 508,
    ]
    var offset = seedLength

    func serialRowPair() throws -> (Gemma4MTPForward, Gemma4MTPForward) {
        let first = try #require(
            model.fastMTPForward(singleInput, cache: serialCache)
        )
        let second = try #require(
            model.fastMTPForward(singleInput, cache: serialCache)
        )
        return (first, second)
    }

    // Even-offset pairs up to one row short of the sliding window boundary.
    while offset + 2 <= seedLength + 508 {
        let pair = try #require(
            model.exactMTPPair(pairInput, cache: pairCache)
        )
        let serial = try serialRowPair()
        offset += 2
        expectExactBits(
            pair.logits[0..., 0..<1, 0...],
            serial.0.logits,
            dtype: .uint32,
            context: "deep row-0 logits offset \(offset)"
        )
        expectExactBits(
            pair.logits[0..., 1..<2, 0...],
            serial.1.logits,
            dtype: .uint32,
            context: "deep row-1 logits offset \(offset)"
        )
        if checkpointOffsets.contains(offset) {
            expectExactCacheState(
                pairCache,
                serialCache,
                context: "deep checkpoint offset \(offset)"
            )
        }
    }
    #expect(offset == seedLength + 508)

    // One serial row makes the next refusal happen at an odd offset just
    // before the sliding wrap.
    let oddRow = try #require(
        model.fastMTPForward(singleInput, cache: pairCache)
    )
    let oddSerial = try #require(
        model.fastMTPForward(singleInput, cache: serialCache)
    )
    offset += 1
    expectExactBits(
        oddRow.logits,
        oddSerial.logits,
        dtype: .uint32,
        context: "odd serial row logits offset \(offset)"
    )
    let pairBeforeWrap = try #require(
        model.exactMTPPair(pairInput, cache: pairCache)
    )
    let serialBeforeWrap = try serialRowPair()
    offset += 2
    #expect(offset == slidingWindow - 1)
    expectExactBits(
        pairBeforeWrap.logits[0..., 1..<2, 0...],
        serialBeforeWrap.1.logits,
        dtype: .uint32,
        context: "pre-wrap pair row-1 logits"
    )

    // The sliding caches can no longer retain a prefix view for a pair.
    #expect(model.exactMTPPair(pairInput, cache: pairCache) == nil)

    // Serial rows continue across the wrap and the 1,024-key kernel switch.
    for step in 0..<4 {
        let row = try #require(
            model.fastMTPForward(singleInput, cache: pairCache)
        )
        let serialRow = try #require(
            model.fastMTPForward(singleInput, cache: serialCache)
        )
        offset += 1
        expectExactBits(
            row.logits,
            serialRow.logits,
            dtype: .uint32,
            context: "post-wrap serial logits step \(step) offset \(offset)"
        )
    }
    expectExactCacheState(
        pairCache,
        serialCache,
        context: "post-wrap cache state"
    )
    // The pair path stays unavailable after the wrap.
    #expect(model.exactMTPPair(pairInput, cache: pairCache) == nil)
}

private func expectExactBits(
    _ candidate: MLXArray,
    _ reference: MLXArray,
    dtype: DType,
    context: String
) {
    #expect(candidate.shape == reference.shape, Comment(rawValue: context))
    #expect(candidate.dtype == reference.dtype, Comment(rawValue: context))
    let equal = arrayEqual(
        candidate.view(dtype: dtype),
        reference.view(dtype: dtype)
    )
    eval(equal)
    #expect(equal.item(Bool.self), Comment(rawValue: context))
}

private func expectExactCacheState(
    _ candidate: [KVCache],
    _ reference: [KVCache],
    context: String
) {
    #expect(candidate.count == reference.count, Comment(rawValue: context))
    for index in candidate.indices {
        #expect(
            candidate[index].offset == reference[index].offset,
            Comment(rawValue: "\(context) layer \(index) offset")
        )
        let candidateState = candidate[index].state
        let referenceState = reference[index].state
        #expect(
            candidateState.count == referenceState.count,
            Comment(rawValue: "\(context) layer \(index) state count")
        )
        for stateIndex in candidateState.indices {
            expectExactBits(
                candidateState[stateIndex],
                referenceState[stateIndex],
                dtype: .uint16,
                context: "\(context) layer \(index) state \(stateIndex)"
            )
        }
    }
}

/// Load-path gate for the organizer-pinned QAT 4-bit assistant: proves the
/// pinned mlx-swift-lm revision materializes the quantized drafter
/// checkpoint (QuantizedLinear/QuantizedEmbedding construction from the
/// config's affine 4-bit group-64 quantization block) and that one drafter
/// forward over fabricated shared-KV produces a well-formed logits row.
/// Needs only the ~264 MB assistant sidecar, never the 17 GB target.
@Test
func qatAssistantCheckpointLoadsQuantizedAndForwards() async throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["MLXFAST_RUN_MTP_ASSISTANT_LOAD_TESTS"] == "1" else {
        return
    }
    let assistantPath = try #require(
        environment["MLXFAST_MTP_ASSISTANT_DIR"]
    )
    let drafter = try await Gemma4AssistantDraftModel.load(
        from: URL(fileURLWithPath: assistantPath)
    )
    let text = drafter.config.textConfig
    // The QAT checkpoint declares its quantization at the top level of
    // config.json (the level Gemma4AssistantDraftModel.load consumes via
    // BaseConfiguration), not inside text_config.
    let base = try JSONDecoder().decode(
        BaseConfiguration.self,
        from: Data(
            contentsOf: URL(fileURLWithPath: assistantPath)
                .appendingPathComponent("config.json")
        )
    )
    let quantization = try #require(
        base.perLayerQuantization?.quantization
    )
    #expect(quantization.bits == 4)
    #expect(quantization.groupSize == 64)
    // The load-time quantization pass must swap every module that ships
    // `.scales` tensors: the projections and the tied token embedding
    // become their Quantized counterparts.
    #expect(drafter.preProjection is QuantizedLinear)
    #expect(drafter.postProjection is QuantizedLinear)
    #expect(drafter.model.embedTokens is QuantizedEmbedding)

    // One drafter forward over fabricated, fixed shared-KV (target-free):
    // shapes follow Gemma4SharedKV's documented [B, heads, T, headDim]
    // layout for the 31B-IT tower (4x512 full, 16x256 sliding).
    let backbone = drafter.config.backboneHiddenSize
    let sharedKV = Gemma4SharedKV(
        fullAttention: (
            MLXArray.zeros([1, 4, 8, 512], dtype: .bfloat16),
            MLXArray.zeros([1, 4, 8, 512], dtype: .bfloat16)
        ),
        slidingAttention: (
            MLXArray.zeros([1, 16, 8, 256], dtype: .bfloat16),
            MLXArray.zeros([1, 16, 8, 256], dtype: .bfloat16)
        )
    )
    let output = drafter(
        inputsEmbeds: MLXArray.zeros([1, 1, 2 * backbone], dtype: .bfloat16),
        sharedKV: sharedKV,
        positionOffset: Gemma4.PositionOffset.scalar(8)
    )
    eval(output.logits, output.lastHidden)
    #expect(output.logits.shape == [1, 1, text.vocabSize])
    #expect(output.lastHidden.shape == [1, 1, backbone])
    let draft = output.logits[0..., -1, 0...].argMax(axis: -1)
    let token = Int(draft.item(Int32.self))
    #expect(token >= 0 && token < text.vocabSize)
    #expect(
        output.logits.asType(.float32).sum().item(Float.self).isFinite
    )
}
