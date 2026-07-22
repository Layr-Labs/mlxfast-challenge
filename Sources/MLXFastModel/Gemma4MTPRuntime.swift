import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon

private let gemma4MTPFastTargetEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_MTP_FAST_TARGET"
    ] else {
        return true
    }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

private let gemma4MTPExactFourEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_MTP_EXACT_FOUR"
    ] else {
        return true
    }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

private let gemma4MTPAdaptiveExactFourEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_MTP_ADAPTIVE_EXACT_FOUR"
    ] else {
        return true
    }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

private let gemma4MTPExactFourMarginThreshold: Float = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_MTP_EXACT_FOUR_MARGIN_THRESHOLD"
    ], let value = Float(raw), value.isFinite, value >= 0 else {
        return 3.8125
    }
    return value
}()

private let gemma4MTPLeadingMarginSelectorEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_MTP_LEADING_MARGIN_SELECTOR"
    ] else {
        return true
    }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

private let gemma4MTPSecondaryMarginSelectorEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_MTP_SECONDARY_MARGIN_SELECTOR"
    ] else {
        return true
    }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

private let gemma4MTPDrafterAsyncEvalEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_MTP_DRAFTER_ASYNC_EVAL"
    ] else {
        return true
    }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

private let gemma4MTPAssistantPositionDelta: Int = {
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_MTP_ASSISTANT_POSITION_DELTA"
    ] == "0" ? 0 : -1
}()

private let gemma4MTPTopTwoMarginKernel = MLXFast.metalKernel(
    name: "gemma4_mtp_top_two_margin_262144_v1",
    inputNames: ["logits"],
    outputNames: ["margin"],
    source: """
        threadgroup float group_best[8];
        threadgroup float group_second[8];

        float best = -INFINITY;
        float second = -INFINITY;
        for (uint index = thread_position_in_threadgroup.x;
             index < 262144;
             index += threads_per_threadgroup.x) {
            gemma4_mtp_insert_top_two(
                static_cast<float>(logits[index]),
                best,
                second
            );
        }

        for (ushort delta = 16; delta > 0; delta >>= 1) {
            const float other_best = simd_shuffle_down(best, delta);
            const float other_second = simd_shuffle_down(second, delta);
            if (thread_index_in_simdgroup + delta < 32) {
                gemma4_mtp_insert_top_two(other_best, best, second);
                gemma4_mtp_insert_top_two(other_second, best, second);
            }
        }
        if (thread_index_in_simdgroup == 0) {
            group_best[simdgroup_index_in_threadgroup] = best;
            group_second[simdgroup_index_in_threadgroup] = second;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simdgroup_index_in_threadgroup == 0) {
            if (thread_index_in_simdgroup < 8) {
                best = group_best[thread_index_in_simdgroup];
                second = group_second[thread_index_in_simdgroup];
            } else {
                best = -INFINITY;
                second = -INFINITY;
            }
            for (ushort delta = 16; delta > 0; delta >>= 1) {
                const float other_best = simd_shuffle_down(best, delta);
                const float other_second = simd_shuffle_down(second, delta);
                if (thread_index_in_simdgroup + delta < 32) {
                    gemma4_mtp_insert_top_two(other_best, best, second);
                    gemma4_mtp_insert_top_two(other_second, best, second);
                }
            }
            if (thread_index_in_simdgroup == 0) {
                margin[0] = best - second;
            }
        }
        """,
    header: """
        using namespace metal;

        inline void gemma4_mtp_insert_top_two(
            float value,
            thread float& best,
            thread float& second
        ) {
            if (value > best) {
                second = best;
                best = value;
            } else if (value > second) {
                second = value;
            }
        }
        """,
    ensureRowContiguous: true
)

func gemma4MTPTopTwoMargin(_ logits: MLXArray) -> MLXArray {
    gemma4MTPTopTwoMarginKernel(
        [logits],
        grid: (256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1]],
        outputDTypes: [.float32]
    )[0]
}

func gemma4MTPShouldUseExactFour(
    draftMargins: [Float],
    threshold: Float,
    leadingMarginSelectorEnabled: Bool = true,
    secondaryMarginSelectorEnabled: Bool = true
) -> Bool {
    guard draftMargins.count == 3 else {
        return false
    }
    if draftMargins.allSatisfy({ $0 >= threshold }) {
        return true
    }
    // Exact-pair verification needs a second target traversal whenever the
    // first two drafts survive; the third draft's outcome does not affect that
    // cost. High confidence in those two leading drafts can therefore select
    // exact-four independently of the third margin.
    if leadingMarginSelectorEnabled
        && draftMargins[0] >= 6.0
        && draftMargins[1] >= 8.0
    {
        return true
    }
    // A second, disjoint high-confidence region recovers cases where the
    // second margin is moderate but the first and third are both strong.
    // This selects the same bit-exact four-row verifier; it cannot alter the
    // draft tokens or acceptance decision.
    return secondaryMarginSelectorEnabled
        && draftMargins[0] >= 7.125
        && draftMargins[1] >= 3.25
        && draftMargins[2] >= 6.0
}

public enum MTPVerificationMode: String, Sendable {
    case exactPair = "exact-pair"
    case serial
}

/// Make the challenge's weighted runtime tower usable by the exact
/// `Gemma4MTPTarget` API pinned in Package.resolved.
///
/// The ordinary serial `callAsFunction` remains unchanged. MTP uses the
/// library trunk's capture hook so a multi-position target verification also
/// returns the pre-norm hidden state and the last full/sliding K/V pairs that
/// Google's trained assistant consumes.
extension Gemma4RuntimeModel: Gemma4MTPTarget {
    public var mtpConfiguration: Gemma4TextConfiguration {
        configuration
    }

    public func mtpNewCache(
        parameters: GenerateParameters?
    ) -> [any KVCache] {
        newCache(parameters: parameters)
    }

    public func embedTokensForDrafter(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(tokens) * Float(configuration.hiddenSize).squareRoot()
    }

    public func forwardForMTP(
        _ tokens: MLXArray,
        cache: [KVCache]
    ) -> Gemma4MTPForward {
        if gemma4MTPFastTargetEnabled,
           let fast = fastMTPForward(tokens, cache: cache)
        {
            return fast
        }
        let firstSharedLayer =
            configuration.numHiddenLayers - configuration.numKvSharedLayers
        var lastFull = -1
        var lastSliding = -1
        for layerIndex in 0..<firstSharedLayer {
            switch configuration.layerTypes[layerIndex] {
            case Gemma4LayerType.full.rawValue:
                lastFull = layerIndex
            case Gemma4LayerType.sliding.rawValue:
                lastSliding = layerIndex
            default:
                break
            }
        }

        let capture = Gemma4SharedKVCapture()
        let forward = model.callCapturingPreNorm(
            tokens,
            cache: cache
        ) { layerIndex, keyValue in
            if layerIndex == lastFull {
                capture.fullAttention = keyValue
            } else if layerIndex == lastSliding {
                capture.slidingAttention = keyValue
            }
        }
        let projected = model.embedTokens.asLinear(forward.postNorm)
        let cap = MLXArray(configuration.finalLogitSoftcapping)
        let logits = tanh(projected / cap) * cap
        return Gemma4MTPForward(
            logits: logits,
            lastHidden: forward.preNorm,
            capturedSharedKV: capture.snapshot()
        )
    }

    /// Compile every timed MTP target shape with fixed BOS inputs and
    /// throwaway state, independent of assistant acceptance behavior: the
    /// 512-token seed prefill, four serial K=1 verification rows (K=1 blocks,
    /// K=3 bonus rows, and geometry tails), the K=2 exact-pair segments used
    /// by K=2/K=4 when requested, and the deep sliding-attention decode
    /// shapes at the library's 1,024-key kernel switch. This is
    /// input-independent pipeline warmup only: no prompt-derived tensor,
    /// cache, token, or logit escapes the method.
    public func warmMTPTargetKernels(
        includeExactPair: Bool,
        includePrefill: Bool = true
    ) throws {
        guard !includeExactPair || supportsExactMTPPair else {
            throw MLXFastError.invalidInput(
                "exact-pair MTP warmup is unavailable for this target"
            )
        }
        let cache = mtpNewCache(parameters: nil)
        let bos = Int32(2)
        // The 512-token prefill shape is only needed when this warms from cold.
        // The post-allocator-reset re-warm skips it because the charged seed
        // prefill in `begin` reallocates that exact working set immediately
        // after, so warming it here twice would only add redundant timed cost.
        if includePrefill {
            let prefill = forwardForMTP(
                MLXArray(
                    Array(
                        repeating: bos,
                        count: MLXFastConstants.correctnessPromptTokens
                    ),
                    [1, MLXFastConstants.correctnessPromptTokens]
                ),
                cache: cache
            )
            eval(
                prefill.logits,
                prefill.lastHidden,
                prefill.capturedSharedKV.fullAttention.0,
                prefill.capturedSharedKV.fullAttention.1,
                prefill.capturedSharedKV.slidingAttention.0,
                prefill.capturedSharedKV.slidingAttention.1
            )
        }
        // A fixed assistant draft may reject at the first position, so the
        // drafter warmup round cannot guarantee later serial rows compiled.
        // Warm all four possible K=4 serial row positions unconditionally.
        for _ in 0..<4 {
            let row = forwardForMTP(
                MLXArray([bos], [1, 1]),
                cache: cache
            )
            eval(
                row.logits,
                row.lastHidden,
                row.capturedSharedKV.fullAttention.0,
                row.capturedSharedKV.fullAttention.1,
                row.capturedSharedKV.slidingAttention.0,
                row.capturedSharedKV.slidingAttention.1
            )
        }
        if includeExactPair {
            if gemma4MTPExactFourEnabled {
                let fourInput = MLXArray([bos, bos, bos, bos], [1, 4])
                if let four = exactMTPFourTokens(fourInput, cache: cache) {
                    eval(
                        four.tokenIDs,
                        four.lastHidden,
                        four.capturedSharedKV.fullAttention.0,
                        four.capturedSharedKV.fullAttention.1,
                        four.capturedSharedKV.slidingAttention.0,
                        four.capturedSharedKV.slidingAttention.1
                    )
                }
            }
            let pairInput = MLXArray([bos, bos], [1, 2])
            for _ in 0..<2 {
                guard let pair = exactMTPPair(pairInput, cache: cache) else {
                    throw MLXFastError.invalidInput(
                        "exact-pair MTP warmup could not append a fixed pair"
                    )
                }
                eval(
                    pair.logits,
                    pair.lastHidden,
                    pair.capturedSharedKV.fullAttention.0,
                    pair.capturedSharedKV.fullAttention.1,
                    pair.capturedSharedKV.slidingAttention.0,
                    pair.capturedSharedKV.slidingAttention.1
                )
            }
        }
        warmDeepSlidingDecodeAttention()
    }

    /// Long decodes cross the library's D=256 kernel switch at 1,024 keys,
    /// where sliding attention changes to its two-pass kernel and the
    /// exact-pair path serializes its rows. Warm those dispatches with fixed
    /// zero tensors so the switch is not first compiled inside a timed block.
    private func warmDeepSlidingDecodeAttention() {
        let queries = MLXArray.zeros([1, 32, 1, 256], dtype: .bfloat16)
        let keys = MLXArray.zeros([1, 16, 1_024, 256], dtype: .bfloat16)
        let values = MLXArray.zeros([1, 16, 1_024, 256], dtype: .bfloat16)
        let unmasked = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1.0,
            mask: .none
        )
        let windowMask = MLXArray.ones([1_024], dtype: .bool)
        let masked = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1.0,
            mask: .array(windowMask)
        )
        eval(unmasked, masked)
    }

    public func rollbackSpeculativeCache(
        _ caches: [KVCache],
        accepted: Gemma4AcceptCount,
        blockSize: Int
    ) {
        let maxAccepted = accepted.maxAccepted()
        let rejectedTail = max(0, blockSize - maxAccepted - 1)
        for cache in caches where cache.isTrimmable && rejectedTail > 0 {
            _ = cache.trim(rejectedTail)
        }
    }
}

public struct Gemma4MTPBlockResult: Equatable, Sendable {
    public let tokens: [Int]
    public let acceptedDraftTokenCount: Int
    public let targetCacheOffset: Int
    public let usedDrafter: Bool

    public init(
        tokens: [Int],
        acceptedDraftTokenCount: Int,
        targetCacheOffset: Int,
        usedDrafter: Bool
    ) {
        self.tokens = tokens
        self.acceptedDraftTokenCount = acceptedDraftTokenCount
        self.targetCacheOffset = targetCacheOffset
        self.usedDrafter = usedDrafter
    }
}

struct Gemma4ExactPairDecision: Equatable {
    let acceptedDrafts: Int
    let committedTokens: [Int]
}

func gemma4ExactPairDecision(
    drafts: [Int],
    targetTokens: [Int]
) throws -> Gemma4ExactPairDecision {
    guard (1...2).contains(drafts.count),
          targetTokens.count == 2
    else {
        throw MLXFastError.invalidInput(
            "exact MTP pair decision requires one or two drafts and two targets"
        )
    }
    for index in drafts.indices where drafts[index] != targetTokens[index] {
        return Gemma4ExactPairDecision(
            acceptedDrafts: index,
            committedTokens: Array(targetTokens[0...index])
        )
    }
    let committedCount = drafts.count == 1 ? 2 : drafts.count
    return Gemma4ExactPairDecision(
        acceptedDrafts: drafts.count,
        committedTokens: Array(targetTokens.prefix(committedCount))
    )
}

struct Gemma4MTPVerifiedBlock {
    let committedTokens: [Int]
    let acceptedDrafts: Int
    let hidden: MLXArray
    let sharedKV: Gemma4SharedKV
}

/// Stateful, greedy Gemma 4 MTP block decoder.
///
/// One instance owns one request. Target cache, target hidden state, shared
/// full/sliding K/V and the current bonus token persist across block calls.
/// Every returned token came from a target argmax. Drafts are never returned
/// directly: a draft is emitted only when it equals the corresponding
/// serial-equivalent K=1 target verification result. The block protocol still
/// amortizes assistant drafting and IPC, while shape-dependent target
/// reductions cannot alter the serial oracle or retained physical K/V state.
public final class Gemma4TrainedMTPBlockSession: @unchecked Sendable {
    public var implementationName: String {
        "google_gemma4_trained_mtp_\(verificationMode.rawValue)"
    }

    private let target: Gemma4RuntimeModel
    private let drafter: Gemma4AssistantDraftModel
    public let verificationMode: MTPVerificationMode
    // Internal (not public) so runtime seam tests can compare this cache's
    // physical bytes against an independent serial cache.
    private(set) var targetCache: [any KVCache] = []
    private var hidden: MLXArray?
    private var sharedKV: Gemma4SharedKV?
    private var bonusToken: Int?
    private var hostCacheOffset = 0

    public private(set) var decodedTokenCount = 0
    public private(set) var roundCount = 0
    public private(set) var acceptedDraftTokenCount = 0
    public private(set) var targetOnlyTailTokenCount = 0
    public private(set) var exactPairSegmentCount = 0
    public private(set) var exactPairRollbackRowCount = 0
    public private(set) var serialVerificationRowCount = 0

    public init(
        target: Gemma4RuntimeModel,
        drafter: Gemma4AssistantDraftModel,
        verificationMode: MTPVerificationMode = .exactPair
    ) throws {
        guard verificationMode != .exactPair
                || target.supportsExactMTPPair
        else {
            throw MLXFastError.invalidInput(
                "exact-pair MTP verification is unavailable for this target"
            )
        }
        self.target = target
        self.drafter = drafter
        self.verificationMode = verificationMode
        try drafter.bind(target: target)
    }

    /// Re-populate the MLX allocator free pool with the drafter and exact-pair
    /// target working set after the trusted phase-start cache clear, so the
    /// first *timed* decode block does not absorb a one-time buffer
    /// first-touch cost. Without this, every run spikes on block 0 (measured
    /// 0.6-1.6s versus a ~75ms steady block) because
    /// `resetRuntimeWorkerAllocatorForPhaseStart()` frees the buffers warmed
    /// during untimed init; the first exact-pair `eval` then reallocates the
    /// whole 60-layer working set inside the timed window.
    ///
    /// Safety: this runs fixed BOS shapes on throwaway caches and never reads,
    /// writes, or advances the real session cache, hidden state, shared-KV, or
    /// committed tokens, so it cannot change any returned token — the serial
    /// oracle and bit-exact parity are unaffected. It must be called after the
    /// trusted allocator reset and before `begin`; at that point no seed exists,
    /// so it is structurally input-independent (it cannot depend on the prompt).
    /// It only warms the allocator: the pinned pipelines were already compiled
    /// during untimed init, so this pays allocation, not compilation.
    public func warmWorkingSetAfterAllocatorReset() throws {
        // Target working set: exact-pair segments, serial rows, and the deep
        // sliding-attention kernel. This reuses the init warmup shapes so the
        // same buffers the first block needs are resident in the free pool.
        // The 512-token prefill working set is skipped here because `begin`'s
        // charged seed prefill reallocates it immediately after this returns.
        try target.warmMTPTargetKernels(
            includeExactPair: verificationMode == .exactPair,
            includePrefill: false
        )
        // Drafter working set: one throwaway K-1 draft chain on scratch
        // shared-KV so the first block's drafting also reuses cached buffers.
        let scratch = target.mtpNewCache(parameters: nil)
        let seedForward = target.forwardForMTP(
            MLXArray([Int32(2)], [1, 1]),
            cache: scratch
        )
        let lastIndex = seedForward.lastHidden.dim(1) - 1
        var draftHidden = seedForward.lastHidden[
            0..., lastIndex..<(lastIndex + 1), 0...
        ]
        let sharedKV = seedForward.capturedSharedKV
        var draftInput = MLXArray([Int32(2)], [1, 1])
        for _ in 0..<(MLXFastConstants.experimentalMTPMaxBlockSize - 1) {
            let embedding = target.embedTokensForDrafter(draftInput)
            let assistantInput = concatenated(
                [embedding, draftHidden],
                axis: -1
            )
            let output = drafter(
                inputsEmbeds: assistantInput,
                sharedKV: sharedKV,
                positionOffset: Gemma4.PositionOffset.scalar(
                    1 + gemma4MTPAssistantPositionDelta
                )
            )
            let logits = output.logits[0..., -1, 0...]
            let draft = logits.argMax(axis: -1).reshaped([1, 1])
            if gemma4MTPExactFourEnabled && gemma4MTPAdaptiveExactFourEnabled {
                eval(draft, output.lastHidden, gemma4MTPTopTwoMargin(logits))
            } else {
                eval(draft, output.lastHidden)
            }
            draftInput = draft
            draftHidden = output.lastHidden
        }
    }

    public func begin(seedTokens: [Int]) throws -> Int {
        guard !seedTokens.isEmpty,
              seedTokens.allSatisfy({
                  $0 >= 0 && $0 < MLXFastConstants.vocabSize
              })
        else {
            throw MLXFastError.invalidInput(
                "trained MTP seed tokens must be a nonempty in-vocabulary sequence"
            )
        }

        targetCache = target.mtpNewCache(parameters: nil)
        let input = MLXArray(seedTokens.map(Int32.init), [1, seedTokens.count])
        let prefill = target.forwardForMTP(input, cache: targetCache)
        let seedArray = prefill.logits[
            0..., -1, 0...
        ].asType(.float32).argMax(axis: -1)
        let nextHidden = prefill.lastHidden[
            0..., (prefill.lastHidden.dim(1) - 1)..<prefill.lastHidden.dim(1), 0...
        ]
        let nextSharedKV = prefill.capturedSharedKV
        eval(
            seedArray,
            nextHidden,
            nextSharedKV.fullAttention.0,
            nextSharedKV.fullAttention.1,
            nextSharedKV.slidingAttention.0,
            nextSharedKV.slidingAttention.1
        )

        let seedToken = Int(seedArray.item(Int32.self))
        hostCacheOffset = seedTokens.count
        try requireCacheOffsets(hostCacheOffset, context: "trained MTP prefill")
        hidden = nextHidden
        sharedKV = nextSharedKV
        bonusToken = seedToken
        decodedTokenCount = 0
        roundCount = 0
        acceptedDraftTokenCount = 0
        targetOnlyTailTokenCount = 0
        exactPairSegmentCount = 0
        exactPairRollbackRowCount = 0
        serialVerificationRowCount = 0
        return seedToken
    }

    public func generateBlock(
        previousCommittedToken: Int,
        maxBlockSize: Int
    ) throws -> Gemma4MTPBlockResult {
        guard maxBlockSize > 0,
              maxBlockSize <= MLXFastConstants.experimentalMTPMaxBlockSize
        else {
            throw MLXFastError.invalidInput(
                "trained MTP block size must be in "
                    + "1...\(MLXFastConstants.experimentalMTPMaxBlockSize)"
            )
        }
        guard decodedTokenCount <=
            MLXFastConstants.experimentalMTPMaxConfiguredTotalTokens
                - maxBlockSize
        else {
            throw MLXFastError.invalidInput(
                "trained MTP block would exceed the fixed decode window"
            )
        }
        guard let bonusToken, previousCommittedToken == bonusToken else {
            throw MLXFastError.invalidInput(
                "trained MTP block input is not the session's last committed token"
            )
        }
        guard hidden != nil, sharedKV != nil, !targetCache.isEmpty else {
            throw MLXFastError.invalidInput(
                "trained MTP block requested before prefill"
            )
        }
        try requireCacheOffsets(hostCacheOffset, context: "trained MTP block start")

        let result: Gemma4MTPBlockResult
        if maxBlockSize == 1 {
            result = try generateTargetOnlyTail(previousToken: bonusToken)
        } else {
            result = try generateSpeculativeRound(
                previousToken: bonusToken,
                blockSize: maxBlockSize
            )
        }
        decodedTokenCount += result.tokens.count
        return result
    }

    private func generateTargetOnlyTail(
        previousToken: Int
    ) throws -> Gemma4MTPBlockResult {
        let input = MLXArray([Int32(previousToken)], [1, 1])
        let output = target.forwardForMTP(input, cache: targetCache)
        let tokenArray = output.logits[
            0..., -1, 0...
        ].asType(.float32).argMax(axis: -1)
        let nextHidden = output.lastHidden[
            0..., (output.lastHidden.dim(1) - 1)..<output.lastHidden.dim(1), 0...
        ]
        let nextSharedKV = output.capturedSharedKV
        eval(tokenArray, nextHidden)

        let token = Int(tokenArray.item(Int32.self))
        hidden = nextHidden
        sharedKV = nextSharedKV
        bonusToken = token
        hostCacheOffset += 1
        targetOnlyTailTokenCount += 1
        serialVerificationRowCount += 1
        try requireCacheOffsets(hostCacheOffset, context: "trained MTP target-only tail")
        return Gemma4MTPBlockResult(
            tokens: [token],
            acceptedDraftTokenCount: 0,
            targetCacheOffset: hostCacheOffset,
            usedDrafter: false
        )
    }

    private func generateSpeculativeRound(
        previousToken: Int,
        blockSize: Int
    ) throws -> Gemma4MTPBlockResult {
        guard var draftHidden = hidden, let currentSharedKV = sharedKV else {
            throw MLXFastError.invalidInput("trained MTP state is incomplete")
        }

        let draftCount = blockSize - 1
        // The assistant consumes the target hidden captured at the preceding
        // input row. Align RoPE to that hidden-state position, not to the
        // bonus token predicted one position later.
        let positionOffset = Gemma4.PositionOffset.scalar(
            hostCacheOffset + gemma4MTPAssistantPositionDelta
        )
        var draftInput = MLXArray([Int32(previousToken)], [1, 1])
        var draftTokens: [MLXArray] = []
        var draftMarginArrays: [MLXArray] = []
        draftTokens.reserveCapacity(draftCount)
        draftMarginArrays.reserveCapacity(draftCount)

        let collectExactFourMargins = verificationMode == .exactPair
            && gemma4MTPExactFourEnabled
            && gemma4MTPAdaptiveExactFourEnabled
            && draftCount == 3

        for _ in 0..<draftCount {
            let targetEmbedding = target.embedTokensForDrafter(draftInput)
            let assistantInput = concatenated(
                [targetEmbedding, draftHidden],
                axis: -1
            )
            let assistantOutput = drafter(
                inputsEmbeds: assistantInput,
                sharedKV: currentSharedKV,
                positionOffset: positionOffset
            )
            let draftLogits = assistantOutput.logits[0..., -1, 0...]
            let draft = draftLogits.argMax(axis: -1)
            if collectExactFourMargins {
                draftMarginArrays.append(gemma4MTPTopTwoMargin(draftLogits))
            }
            let draft2D = draft.reshaped([1, 1])
            draftTokens.append(draft2D)
            draftInput = draft2D
            draftHidden = assistantOutput.lastHidden
            // Dispatch this draft's subgraph while the host constructs the
            // next draft's graph, mirroring the exact-pair verify path's
            // async cadence. Scheduling only: the per-array dependency graph
            // is unchanged, so drafted tokens and margins are bit-identical.
            if gemma4MTPDrafterAsyncEvalEnabled {
                if collectExactFourMargins, let margin = draftMarginArrays.last {
                    asyncEval(draft2D, margin)
                } else {
                    asyncEval(draft2D)
                }
            }
        }

        let draftConcat = concatenated(draftTokens, axis: 1)
        let draftMargins: [Float]
        if collectExactFourMargins {
            let marginArray = concatenated(draftMarginArrays, axis: 0)
            eval(draftConcat, marginArray)
            draftMargins = marginArray.asArray(Float.self)
        } else {
            eval(draftConcat)
            draftMargins = []
        }
        let draftValues = draftConcat
            .squeezed(axis: 0)
            .asArray(Int32.self)
            .map(Int.init)
        guard draftValues.count == draftCount else {
            throw MLXFastError.invalidInput(
                "trained MTP drafter returned an unexpected tensor shape"
            )
        }

        let verified: Gemma4MTPVerifiedBlock
        switch verificationMode {
        case .exactPair:
            verified = try verifyWithExactPairs(
                previousToken: previousToken,
                drafts: draftValues,
                preferExactFour: !gemma4MTPAdaptiveExactFourEnabled
                    || gemma4MTPShouldUseExactFour(
                        draftMargins: draftMargins,
                        threshold: gemma4MTPExactFourMarginThreshold,
                        leadingMarginSelectorEnabled:
                            gemma4MTPLeadingMarginSelectorEnabled,
                        secondaryMarginSelectorEnabled:
                            gemma4MTPSecondaryMarginSelectorEnabled
                    )
            )
        case .serial:
            verified = try verifySerially(
                previousToken: previousToken,
                drafts: draftValues
            )
        }

        guard verified.committedTokens.count
                == verified.acceptedDrafts + 1,
              let nextBonus = verified.committedTokens.last
        else {
            throw MLXFastError.invalidInput(
                "trained MTP exact verification produced an invalid block"
            )
        }

        hidden = verified.hidden
        sharedKV = verified.sharedKV
        bonusToken = nextBonus
        hostCacheOffset = targetCache[0].offset
        roundCount += 1
        acceptedDraftTokenCount += verified.acceptedDrafts
        try requireCacheOffsets(hostCacheOffset, context: "trained MTP commit/rollback")
        return Gemma4MTPBlockResult(
            tokens: verified.committedTokens,
            acceptedDraftTokenCount: verified.acceptedDrafts,
            targetCacheOffset: hostCacheOffset,
            usedDrafter: true
        )
    }

    private func verifySerially(
        previousToken: Int,
        drafts: [Int]
    ) throws -> Gemma4MTPVerifiedBlock {
        let verifyInputs = [previousToken] + drafts
        var committed: [Int] = []
        var accepted = 0
        var nextHidden: MLXArray?
        var nextSharedKV: Gemma4SharedKV?
        for (position, inputToken) in verifyInputs.enumerated() {
            let row = runSerialTargetRow(inputToken)
            committed.append(row.token)
            nextHidden = row.hidden
            nextSharedKV = row.sharedKV
            if position < drafts.count,
               row.token == drafts[position]
            {
                accepted += 1
                continue
            }
            break
        }
        guard committed.count == accepted + 1,
              let nextHidden,
              let nextSharedKV
        else {
            throw MLXFastError.invalidInput(
                "serial MTP verification produced an invalid block"
            )
        }
        return Gemma4MTPVerifiedBlock(
            committedTokens: committed,
            acceptedDrafts: accepted,
            hidden: nextHidden,
            sharedKV: nextSharedKV
        )
    }

    // Internal (not public) so runtime seam tests can force zero, partial,
    // full, and second-segment acceptance deterministically without depending
    // on assistant draft behavior.
    func verifyWithExactPairs(
        previousToken: Int,
        drafts: [Int],
        preferExactFour: Bool = true
    ) throws -> Gemma4MTPVerifiedBlock {
        guard !drafts.isEmpty else {
            throw MLXFastError.invalidInput(
                "exact-pair MTP verification requires at least one draft"
            )
        }

        if gemma4MTPExactFourEnabled, preferExactFour, drafts.count == 3 {
            let fourInput = MLXArray(
                ([previousToken] + drafts).map(Int32.init),
                [1, 4]
            )
            if let output = target.exactMTPFourTokens(
                fourInput,
                cache: targetCache
            ) {
                exactPairSegmentCount += 2
                let tokenArray = output.tokenIDs
                let fourHidden = output.lastHidden
                let fourSharedKV = output.capturedSharedKV
                eval(
                    tokenArray,
                    fourHidden,
                    fourSharedKV.fullAttention.0,
                    fourSharedKV.fullAttention.1,
                    fourSharedKV.slidingAttention.0,
                    fourSharedKV.slidingAttention.1
                )
                let targets = tokenArray.asArray(Int32.self).map(Int.init)
                guard targets.count == 4 else {
                    throw MLXFastError.invalidInput(
                        "exact-four MTP returned an unexpected logits shape"
                    )
                }
                let accepted = drafts.indices.first {
                    drafts[$0] != targets[$0]
                } ?? drafts.count
                let rejectedRows = drafts.count - accepted
                if rejectedRows > 0 {
                    for cache in targetCache {
                        guard let combined = cache as? Gemma4CombinedKVCache,
                              combined.trim(rejectedRows) == rejectedRows
                        else {
                            throw MLXFastError.invalidInput(
                                "exact-four MTP could not roll back rejected rows"
                            )
                        }
                    }
                    exactPairRollbackRowCount += rejectedRows
                }
                let retainedSharedKV = rejectedRows > 0
                    ? Gemma4SharedKV.sliceTail(
                        from: fourSharedKV,
                        rejected: rejectedRows
                    )
                    : fourSharedKV
                return Gemma4MTPVerifiedBlock(
                    committedTokens: Array(targets.prefix(accepted + 1)),
                    acceptedDrafts: accepted,
                    hidden: fourHidden[
                        0..., accepted..<(accepted + 1), 0...
                    ],
                    sharedKV: retainedSharedKV
                )
            }
        }
        var draftIndex = 0
        var currentInput = previousToken
        var committed: [Int] = []
        var accepted = 0

        while draftIndex < drafts.count {
            let pairInput = MLXArray(
                [Int32(currentInput), Int32(drafts[draftIndex])],
                [1, 2]
            )
            let pairOutput: Gemma4MTPTokenForward?
            if gemma4MTPExactPairFusedArgmaxEnabled {
                pairOutput = target.exactMTPPairTokens(
                    pairInput,
                    cache: targetCache
                )
            } else if let fullOutput = target.exactMTPPair(
                pairInput,
                cache: targetCache
            ) {
                pairOutput = Gemma4MTPTokenForward(
                    tokenIDs: fullOutput.logits.asType(.float32).argMax(axis: -1),
                    lastHidden: fullOutput.lastHidden,
                    capturedSharedKV: fullOutput.capturedSharedKV
                )
            } else {
                pairOutput = nil
            }
            guard let output = pairOutput else {
                // Only deterministic cache geometry may serialize: a rotating
                // cache at its wrap boundary cannot retain the prefix view
                // needed by exact-pair attention. Engine-level ineligibility
                // was checked at session construction and must stay a hard
                // failure, never a silent timed fallback.
                guard target.supportsExactMTPPair else {
                    throw MLXFastError.invalidInput(
                        "exact-pair MTP engine support disappeared mid-session"
                    )
                }
                let tail = try verifySerially(
                    previousToken: currentInput,
                    drafts: Array(drafts[draftIndex...])
                )
                return Gemma4MTPVerifiedBlock(
                    committedTokens: committed + tail.committedTokens,
                    acceptedDrafts: accepted + tail.acceptedDrafts,
                    hidden: tail.hidden,
                    sharedKV: tail.sharedKV
                )
            }
            exactPairSegmentCount += 1

            let tokenArray = output.tokenIDs
            let pairHidden = output.lastHidden
            let pairSharedKV = output.capturedSharedKV
            eval(
                tokenArray,
                pairHidden,
                pairSharedKV.fullAttention.0,
                pairSharedKV.fullAttention.1,
                pairSharedKV.slidingAttention.0,
                pairSharedKV.slidingAttention.1
            )
            let targets = tokenArray.asArray(Int32.self).map(Int.init)
            guard targets.count == 2 else {
                throw MLXFastError.invalidInput(
                    "exact MTP pair returned an unexpected logits shape"
                )
            }
            let remainingDraftCount = drafts.count - draftIndex
            let pairDraftCount = min(2, remainingDraftCount)
            let pairDrafts = Array(
                drafts[draftIndex..<(draftIndex + pairDraftCount)]
            )
            let decision = try gemma4ExactPairDecision(
                drafts: pairDrafts,
                targetTokens: targets
            )

            if decision.acceptedDrafts == 0 {
                for cache in targetCache {
                    guard let combined = cache as? Gemma4CombinedKVCache,
                          combined.trim(1) == 1
                    else {
                        throw MLXFastError.invalidInput(
                            "exact MTP pair could not roll back rejected row one"
                        )
                    }
                }
                exactPairRollbackRowCount += 1
                let rolledSharedKV = Gemma4SharedKV.sliceTail(
                    from: pairSharedKV,
                    rejected: 1
                )
                return Gemma4MTPVerifiedBlock(
                    committedTokens: committed + decision.committedTokens,
                    acceptedDrafts: accepted,
                    hidden: pairHidden[0..., 0..<1, 0...],
                    sharedKV: rolledSharedKV
                )
            }

            committed.append(contentsOf: decision.committedTokens)
            accepted += decision.acceptedDrafts
            if decision.acceptedDrafts < pairDraftCount {
                return Gemma4MTPVerifiedBlock(
                    committedTokens: committed,
                    acceptedDrafts: accepted,
                    hidden: pairHidden[0..., 1..<2, 0...],
                    sharedKV: pairSharedKV
                )
            }
            if pairDraftCount == 1 {
                return Gemma4MTPVerifiedBlock(
                    committedTokens: committed,
                    acceptedDrafts: accepted,
                    hidden: pairHidden[0..., 1..<2, 0...],
                    sharedKV: pairSharedKV
                )
            }

            draftIndex += pairDraftCount
            currentInput = drafts[draftIndex - 1]
            if draftIndex == drafts.count {
                // Both pair predictions were accepted, but row one contains
                // only the first draft input. Consume the second draft once to
                // obtain the required bonus token.
                let bonus = runSerialTargetRow(currentInput)
                committed.append(bonus.token)
                return Gemma4MTPVerifiedBlock(
                    committedTokens: committed,
                    acceptedDrafts: accepted,
                    hidden: bonus.hidden,
                    sharedKV: bonus.sharedKV
                )
            }
        }
        throw MLXFastError.invalidInput(
            "exact-pair MTP verification exhausted without a bonus token"
        )
    }

    private func runSerialTargetRow(
        _ inputToken: Int
    ) -> (token: Int, hidden: MLXArray, sharedKV: Gemma4SharedKV) {
        let output = target.forwardForMTP(
            MLXArray([Int32(inputToken)], [1, 1]),
            cache: targetCache
        )
        let tokenArray = output.logits[
            0..., -1, 0...
        ].asType(.float32).argMax(axis: -1)
        let outputHiddenLength = output.lastHidden.dim(1)
        let outputHidden = output.lastHidden[
            0...,
            (outputHiddenLength - 1)..<outputHiddenLength,
            0...
        ]
        let outputSharedKV = output.capturedSharedKV
        eval(
            tokenArray,
            outputHidden,
            outputSharedKV.fullAttention.0,
            outputSharedKV.fullAttention.1,
            outputSharedKV.slidingAttention.0,
            outputSharedKV.slidingAttention.1
        )
        serialVerificationRowCount += 1
        return (
            token: Int(tokenArray.item(Int32.self)),
            hidden: outputHidden,
            sharedKV: outputSharedKV
        )
    }

    private func requireCacheOffsets(
        _ expected: Int,
        context: String
    ) throws {
        guard expected >= 0,
              targetCache.allSatisfy({ $0.offset == expected })
        else {
            let offsets = targetCache.map(\.offset)
            throw MLXFastError.invalidInput(
                "\(context) target cache offsets \(offsets) do not match host offset \(expected)"
            )
        }
    }
}
