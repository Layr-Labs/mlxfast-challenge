import Foundation
import MLX
import MLXLMCommon

/// Gemma-specific eager cache that keeps K and V in one K/V-major allocation.
///
/// Supported prompt prefills can install a K/V-major parent allocation
/// directly, with one normal cache-growth quantum reserved for decode. This
/// avoids both split-to-combined conversion and first-token growth. Unsupported
/// shapes deliberately retain the upstream split-cache fallback. Slab zero is
/// K and slab one is V.
final class Gemma4CombinedKVCache: KVCache {
    enum Kind {
        case full(step: Int)
        case rotating(maxSize: Int, keep: Int, step: Int)
    }

    private let kind: Kind
    private var splitCache: (any KVCache)?
    private var combinedStorage: MLXArray?

    // Rotating-cache cursor metadata. These fields mirror RotatingKVCache and
    // remain dormant for a full cache.
    private var keep: Int = 0
    private var rotatingStep: Int = 256
    private var rotatingIndex: Int = 0
    private var rotatingMaxSize: Int?
    private var combinedMutationGeneration = 0
    private var lastPairAppendGeneration: Int?
    private var deferredCombinedPosition: Int?
    private var deferredCombinedGeneration: Int?
    private(set) var offset: Int = 0

    init(fullStep step: Int = 256) {
        precondition(step > 0)
        self.kind = .full(step: step)
        let cache = StandardKVCache()
        cache.step = step
        self.splitCache = cache
    }

    init(rotatingMaxSize maxSize: Int, keep: Int = 0, step: Int = 256) {
        precondition(maxSize > 0)
        precondition(step > 0)
        precondition(keep >= 0 && keep < maxSize)
        self.kind = .rotating(maxSize: maxSize, keep: keep, step: step)
        self.keep = keep
        self.rotatingStep = step
        self.rotatingMaxSize = maxSize
        self.splitCache = RotatingKVCache(maxSize: maxSize, keep: keep, step: step)
    }

    private init(
        kind: Kind,
        splitCache: (any KVCache)?,
        combinedStorage: MLXArray?,
        offset: Int,
        keep: Int,
        rotatingStep: Int,
        rotatingIndex: Int,
        rotatingMaxSize: Int?,
        combinedMutationGeneration: Int
    ) {
        self.kind = kind
        self.splitCache = splitCache
        self.combinedStorage = combinedStorage
        self.keep = keep
        self.rotatingStep = rotatingStep
        self.rotatingIndex = rotatingIndex
        self.rotatingMaxSize = rotatingMaxSize
        self.combinedMutationGeneration = combinedMutationGeneration
        self.offset = offset
    }

    var maxSize: Int? {
        rotatingMaxSize
    }

    /// True after the one-time split-to-combined conversion.
    var usesCombinedStorage: Bool {
        combinedStorage != nil
    }

    /// Capacity for a direct multi-token prefill. Exactly one ordinary growth
    /// quantum is reserved so decode does not immediately copy the seed cache.
    /// Using `length + step` rather than rounding first avoids disproportionate
    /// zero-fill work for short hidden/GPQA prompts. Rotating caches remain
    /// bounded by `maxSize`.
    func directPrefillCapacity(for length: Int) -> Int? {
        guard length > 1, offset == 0, combinedStorage == nil else { return nil }
        if let splitCache, !splitCache.innerState().isEmpty { return nil }

        let step: Int
        let upperBound: Int?
        switch kind {
        case .full(let fullStep):
            step = fullStep
            upperBound = nil
        case .rotating(let maxSize, _, let rotatingStep):
            guard length <= maxSize else { return nil }
            step = rotatingStep
            upperBound = maxSize
        }

        let (reserved, reserveOverflow) = length.addingReportingOverflow(step)
        guard !reserveOverflow else { return nil }
        let capacity = upperBound.map { min($0, reserved) } ?? reserved
        return capacity >= length ? capacity : nil
    }

    /// Adopt direct output from the fused multi-token K/V preparation kernel.
    /// No concatenate, stack, or slice update is needed at the seed boundary.
    func adoptDirectPrefill(
        _ combined: MLXArray,
        length: Int
    ) -> (MLXArray, MLXArray) {
        guard let capacity = directPrefillCapacity(for: length) else {
            preconditionFailure("combined KV cache cannot adopt this prefill")
        }
        precondition(combined.ndim == 5)
        precondition(combined.dim(0) == 2)
        precondition(combined.dim(1) == 1)
        precondition(combined.dim(3) == capacity)
        precondition(combined.dtype == .bfloat16)

        combinedStorage = combined
        splitCache = nil
        combinedMutationGeneration &+= 1
        lastPairAppendGeneration = nil
        deferredCombinedPosition = nil
        deferredCombinedGeneration = nil
        offset = length
        if rotatingMaxSize != nil {
            rotatingIndex = length
        }
        return views(range: 0..<length)
    }

    func innerState() -> [MLXArray] {
        if let combinedStorage {
            return [combinedStorage]
        }
        return splitCache?.innerState() ?? []
    }

    /// Generic protocol fallback. Prefill remains entirely on the upstream
    /// split cache. If an unsupported generic update arrives after conversion,
    /// convert back first so uncommon shapes retain upstream behavior.
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        precondition(keys.ndim == 4 && values.ndim == 4)
        precondition(keys.shape[0...2] == values.shape[0...2])
        precondition(keys.dtype == values.dtype)

        precondition(deferredCombinedPosition == nil)
        precondition(lastPairAppendGeneration == nil)
        if combinedStorage != nil {
            splitCache = makeUpstreamCache()
            combinedStorage = nil
            combinedMutationGeneration &+= 1
            lastPairAppendGeneration = nil
        }
        guard let splitCache else {
            preconditionFailure("combined KV cache lost its split fallback")
        }
        let result = splitCache.update(keys: keys, values: values)
        syncMetadata(from: splitCache)
        return result
    }

    /// Optimized path for one ordinary or two speculative positions.
    /// `combined` must be direct output from fused attention preparation, with
    /// shape `[2,B,Hkv,L,D]` where `L` is one or two.
    func updateCombined(_ combined: MLXArray) -> (MLXArray, MLXArray) {
        precondition(combined.ndim == 5)
        precondition(combined.dim(0) == 2)
        precondition(combined.dim(1) == 1)
        precondition((1...2).contains(combined.dim(3)))
        precondition(deferredCombinedPosition == nil)
        precondition(lastPairAppendGeneration == nil)

        convertSplitStorageIfNeeded(matching: combined)
        let result: (MLXArray, MLXArray)
        switch kind {
        case .full(let step):
            result = updateFull(combined, step: step)
        case .rotating:
            result = updateRotating(combined)
        }
        if combined.dim(3) == 2 {
            combinedMutationGeneration &+= 1
            lastPairAppendGeneration = combinedMutationGeneration
        } else {
            lastPairAppendGeneration = nil
        }
        return result
    }

    /// Speculative pair appends deliberately avoid cache growth and ring wrap.
    /// This keeps one-position rollback metadata-only and guarantees that the
    /// older position needed by token zero remains available as a prefix view.
    func canAppendCombinedPair() -> Bool {
        guard deferredCombinedPosition == nil,
              lastPairAppendGeneration == nil,
              splitCache == nil,
              let combinedStorage
        else { return false }
        switch kind {
        case .full:
            return offset + 2 <= combinedStorage.dim(3)
        case .rotating(let maxSize, _, _):
            return offset == rotatingIndex
                && offset + 2 <= maxSize
                && rotatingIndex + 2 <= combinedStorage.dim(3)
        }
    }

    /// Hide the newest speculative position from the public cache offset while
    /// retaining its bytes. The outer model cache advances by one API input per
    /// call, so the second row cannot become visible until the next call proves
    /// that its input token equals the draft.
    func canDeferNewestCombinedPosition() -> Bool {
        guard deferredCombinedPosition == nil,
              splitCache == nil,
              let combinedStorage,
              lastPairAppendGeneration == combinedMutationGeneration,
              offset > 0
        else { return false }
        if let maxSize = rotatingMaxSize {
            return offset == rotatingIndex
                && offset <= maxSize
                && rotatingIndex > 0
                && rotatingIndex <= combinedStorage.dim(3)
        }
        return offset <= combinedStorage.dim(3)
    }

    @discardableResult
    func deferNewestCombinedPosition() -> Bool {
        guard canDeferNewestCombinedPosition() else { return false }
        offset -= 1
        if rotatingMaxSize != nil {
            rotatingIndex -= 1
        }
        deferredCombinedPosition = offset
        deferredCombinedGeneration = combinedMutationGeneration
        lastPairAppendGeneration = nil
        return true
    }

    func canRecommitDeferredCombinedPosition() -> Bool {
        guard let deferredCombinedPosition,
              let deferredCombinedGeneration,
              splitCache == nil,
              let combinedStorage,
              deferredCombinedPosition == offset,
              deferredCombinedGeneration == combinedMutationGeneration
        else { return false }
        if let maxSize = rotatingMaxSize {
            return rotatingIndex == deferredCombinedPosition
                && offset + 1 <= maxSize
                && rotatingIndex + 1 <= combinedStorage.dim(3)
        }
        return offset + 1 <= combinedStorage.dim(3)
    }

    @discardableResult
    func recommitDeferredCombinedPosition() -> Bool {
        guard canRecommitDeferredCombinedPosition() else { return false }
        offset += 1
        if rotatingMaxSize != nil {
            rotatingIndex += 1
        }
        deferredCombinedPosition = nil
        deferredCombinedGeneration = nil
        return true
    }

    /// Reject the deferred row without changing the visible offset. The next
    /// ordinary cache update overwrites the retained storage slot.
    func canDiscardDeferredCombinedPosition() -> Bool {
        guard let deferredCombinedPosition,
              let deferredCombinedGeneration,
              deferredCombinedPosition == offset,
              deferredCombinedGeneration == combinedMutationGeneration
        else { return false }
        if rotatingMaxSize != nil {
            return rotatingIndex == deferredCombinedPosition
        }
        return true
    }

    @discardableResult
    func discardDeferredCombinedPosition() -> Bool {
        guard canDiscardDeferredCombinedPosition() else { return false }
        deferredCombinedPosition = nil
        deferredCombinedGeneration = nil
        return true
    }

    var hasDeferredCombinedPosition: Bool {
        deferredCombinedPosition != nil
    }

    /// Views the materialized prefix before the newest speculative positions.
    /// Rotating caches are supported only before their first wrap.
    func viewsExcludingNewest(_ count: Int) -> (MLXArray, MLXArray) {
        precondition(count >= 0 && count <= offset)
        guard let combinedStorage else {
            preconditionFailure("combined KV storage is empty")
        }
        if let maxSize = rotatingMaxSize {
            precondition(offset <= maxSize)
            precondition(offset <= combinedStorage.dim(3))
        }
        return views(range: 0..<(offset - count))
    }

    private func convertSplitStorageIfNeeded(matching incoming: MLXArray) {
        guard combinedStorage == nil else { return }
        guard let splitCache else { return }

        syncMetadata(from: splitCache)
        let inner = splitCache.innerState()
        if inner.isEmpty {
            self.splitCache = nil
            return
        }
        precondition(inner.count == 2)
        precondition(inner[0].ndim == 4 && inner[1].ndim == 4)
        precondition(inner[0].shape == inner[1].shape)
        precondition(inner[0].dtype == inner[1].dtype)
        precondition(inner[0].dtype == incoming.dtype)
        precondition(inner[0].dim(0) == incoming.dim(1))
        precondition(inner[0].dim(1) == incoming.dim(2))
        precondition(inner[0].dim(3) == incoming.dim(4))

        // This one-time stack happens at the decode boundary. The recurring
        // path receives the combined parent allocation directly from Metal.
        combinedStorage = stacked(inner, axis: 0)
        self.splitCache = nil
    }

    private func updateFull(_ incoming: MLXArray, step: Int) -> (MLXArray, MLXArray) {
        let previous = offset
        let length = incoming.dim(3)
        precondition((1...2).contains(length))

        if combinedStorage == nil || previous + length > combinedStorage!.dim(3) {
            let chunks = (step + length - 1) / step
            let appendCapacity = chunks * step
            let zeroShape = [
                2, incoming.dim(1), incoming.dim(2), appendCapacity, incoming.dim(4),
            ]
            let newSpace = MLXArray.zeros(zeroShape, dtype: incoming.dtype)
            if let current = combinedStorage {
                let retained = previous % step == 0
                    ? current
                    : current[0..., 0..., 0..., 0..<previous, 0...]
                combinedStorage = concatenated([retained, newSpace], axis: 3)
            } else {
                combinedStorage = newSpace
            }
        }

        offset += length
        combinedStorage![0..., 0..., 0..., previous..<offset, 0...] = incoming
        return views(range: 0..<offset)
    }

    private func updateRotating(_ incoming: MLXArray) -> (MLXArray, MLXArray) {
        guard let maxSize = rotatingMaxSize else {
            preconditionFailure("rotating cache has no maximum size")
        }
        let previous = offset
        let length = incoming.dim(3)
        precondition((1...2).contains(length))

        if combinedStorage == nil
            || (previous + length > combinedStorage!.dim(3)
                && combinedStorage!.dim(3) < maxSize)
        {
            let newSize = min(rotatingStep, maxSize - previous)
            precondition(newSize > 0)
            let newSpace = MLXArray.zeros(
                [2, incoming.dim(1), incoming.dim(2), newSize, incoming.dim(4)],
                dtype: incoming.dtype
            )
            if let current = combinedStorage {
                combinedStorage = concatenated([current, newSpace], axis: 3)
            } else {
                combinedStorage = newSpace
            }
            rotatingIndex = previous
        }

        let trimSize = combinedStorage!.dim(3) - maxSize
        if trimSize > 0 {
            combinedStorage = trimStorage(combinedStorage!, trimSize: trimSize)
            rotatingIndex = maxSize
        }
        if rotatingIndex == maxSize {
            rotatingIndex = keep
        }

        precondition(rotatingIndex + length <= combinedStorage!.dim(3))
        combinedStorage![
            0..., 0..., 0..., rotatingIndex..<(rotatingIndex + length), 0...
        ] = incoming
        offset += length
        rotatingIndex += length

        let validLength = offset < maxSize ? offset : combinedStorage!.dim(3)
        return views(range: 0..<validLength)
    }

    private func trimStorage(_ storage: MLXArray, trimSize: Int) -> MLXArray {
        if trimSize <= 0 { return storage }
        return concatenated(
            [
                storage[0..., 0..., 0..., 0..<keep, 0...],
                storage[0..., 0..., 0..., (trimSize + keep)..., 0...],
            ],
            axis: 3
        )
    }

    /// Range slicing plus squeeze is intentional: integer indexing lowers to
    /// `take`, while these are metadata-only slab views.
    private func views(range: Range<Int>) -> (MLXArray, MLXArray) {
        guard let combinedStorage else {
            preconditionFailure("combined KV storage is empty")
        }
        let keys = combinedStorage[
            0..<1, 0..., 0..., range, 0...
        ].squeezed(axis: 0)
        let values = combinedStorage[
            1..<2, 0..., 0..., range, 0...
        ].squeezed(axis: 0)
        return (keys, values)
    }

    var state: [MLXArray] {
        get {
            if let splitCache { return splitCache.state }
            guard let combinedStorage else { return [] }
            let length: Int
            if let maxSize = rotatingMaxSize {
                length = offset < combinedStorage.dim(3)
                    ? offset
                    : min(maxSize, combinedStorage.dim(3))
            } else {
                length = min(offset, combinedStorage.dim(3))
            }
            let result = views(range: 0..<length)
            return [result.0, result.1]
        }
        set {
            precondition(deferredCombinedPosition == nil)
            precondition(lastPairAppendGeneration == nil)
            precondition(newValue.count == 2)
            var upstream = freshUpstreamCache()
            upstream.state = newValue
            // Full-cache state defines offset. Rotating offset is restored by
            // its metadata setter, exactly like the upstream implementation.
            if rotatingMaxSize == nil {
                offset = upstream.offset
            }
            splitCache = upstream
            combinedStorage = nil
            combinedMutationGeneration &+= 1
            lastPairAppendGeneration = nil
        }
    }

    var metaState: [String] {
        get {
            if let splitCache { return splitCache.metaState }
            guard let maxSize = rotatingMaxSize else { return [""] }
            return [
                String(keep), String(maxSize), String(rotatingStep),
                String(offset), String(rotatingIndex),
            ]
        }
        set {
            precondition(deferredCombinedPosition == nil)
            precondition(lastPairAppendGeneration == nil)
            if rotatingMaxSize == nil {
                precondition(newValue.count == 1 && newValue[0].isEmpty)
                return
            }
            precondition(newValue.count == 5)
            guard let keep = Int(newValue[0]),
                  let maxSize = Int(newValue[1]),
                  let step = Int(newValue[2]),
                  let offset = Int(newValue[3]),
                  let index = Int(newValue[4])
            else {
                preconditionFailure("invalid rotating cache metadata")
            }
            self.keep = keep
            self.rotatingMaxSize = maxSize
            self.rotatingStep = step
            self.offset = offset
            self.rotatingIndex = index
            if var splitCache {
                splitCache.metaState = newValue
                self.splitCache = splitCache
            }
            combinedMutationGeneration &+= 1
            lastPairAppendGeneration = nil
        }
    }

    var isTrimmable: Bool {
        if let splitCache { return splitCache.isTrimmable }
        if let maxSize = rotatingMaxSize { return offset < maxSize }
        return true
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        precondition(deferredCombinedPosition == nil)
        precondition(lastPairAppendGeneration == nil)
        if let splitCache {
            let trimmed = splitCache.trim(n)
            syncMetadata(from: splitCache)
            return trimmed
        }
        let trimmed = min(offset, n)
        offset -= trimmed
        if rotatingMaxSize != nil {
            rotatingIndex -= trimmed
        }
        combinedMutationGeneration &+= 1
        lastPairAppendGeneration = nil
        return trimmed
    }

    func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if let splitCache {
            return splitCache.makeMask(
                n: n, windowSize: windowSize, returnArray: returnArray)
        }
        guard let maxSize = rotatingMaxSize else {
            if n == 1 { return .none }
            if returnArray || (windowSize != nil && n > windowSize!) {
                return .array(
                    createCausalMask(n: n, offset: offset, windowSize: windowSize)
                )
            }
            return .causal
        }
        if n > 1 {
            let actualWindow = windowSize ?? maxSize
            let cappedOffset = min(maxSize - 1, offset)
            if cappedOffset + n > actualWindow || returnArray {
                return .array(
                    createCausalMask(
                        n: n,
                        offset: cappedOffset,
                        windowSize: actualWindow
                    )
                )
            }
            return .causal
        }
        guard let windowSize else { return .none }
        if offset >= windowSize, maxSize > windowSize {
            var currentIndex = rotatingIndex
            if currentIndex >= maxSize { currentIndex = 0 }
            let maskSize = offset < maxSize ? offset + 1 : maxSize
            let mask = MLXArray(0..<Int32(maskSize)) .>= Int32(maskSize - windowSize)
            return .array(roll(mask, shift: currentIndex + 1))
        }
        return .none
    }

    func copy() -> any KVCache {
        precondition(deferredCombinedPosition == nil)
        precondition(lastPairAppendGeneration == nil)
        let splitCopy = splitCache?.copy()
        let storageCopy = combinedStorage.map { $0[0..., 0..., 0..., 0..., 0...] }
        return Gemma4CombinedKVCache(
            kind: kind,
            splitCache: splitCopy,
            combinedStorage: storageCopy,
            offset: offset,
            keep: keep,
            rotatingStep: rotatingStep,
            rotatingIndex: rotatingIndex,
            rotatingMaxSize: rotatingMaxSize,
            combinedMutationGeneration: combinedMutationGeneration
        )
    }

    /// Convert this editable cache back to an upstream concrete cache before
    /// CompiledDecode performs its strict type eligibility check.
    func makeUpstreamCache() -> any KVCache {
        precondition(deferredCombinedPosition == nil)
        precondition(lastPairAppendGeneration == nil)
        if let splitCache { return splitCache }
        var upstream = freshUpstreamCache()
        let currentState = state
        if !currentState.isEmpty {
            upstream.state = currentState
        }
        if rotatingMaxSize != nil {
            upstream.metaState = metaState
        }
        return upstream
    }

    private func freshUpstreamCache() -> any KVCache {
        switch kind {
        case .full(let step):
            let cache = StandardKVCache()
            cache.step = step
            return cache
        case .rotating(let maxSize, let keep, let step):
            return RotatingKVCache(maxSize: maxSize, keep: keep, step: step)
        }
    }

    private func syncMetadata(from cache: any KVCache) {
        offset = cache.offset
        guard rotatingMaxSize != nil else { return }
        let metadata = cache.metaState
        precondition(metadata.count == 5)
        guard let keep = Int(metadata[0]),
              let maxSize = Int(metadata[1]),
              let step = Int(metadata[2]),
              let index = Int(metadata[4])
        else {
            preconditionFailure("invalid upstream rotating cache metadata")
        }
        self.keep = keep
        self.rotatingMaxSize = maxSize
        self.rotatingStep = step
        self.rotatingIndex = index
    }
}

@inline(__always)
func gemma4CombinedKVCacheEnabled() -> Bool {
    gemma4CombinedKVCacheFeatureEnabled
}

@inline(__always)
func gemma4CombinedKVDirectEnabled() -> Bool {
    gemma4CombinedKVDirectFeatureEnabled
}

@inline(__always)
func gemma4CombinedKVPrefillEnabled() -> Bool {
    gemma4CombinedKVPrefillFeatureEnabled
}

@inline(__always)
func gemma4VerifyCombinedKVPrefillBitsEnabled() -> Bool {
    gemma4VerifyCombinedKVPrefillBitsFeatureEnabled
}

private let gemma4CombinedKVCacheFeatureEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_COMBINED_KV_CACHE"] else {
        return true
    }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

private let gemma4CombinedKVDirectFeatureEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_COMBINED_KV_DIRECT"] else {
        return true
    }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

private let gemma4CombinedKVPrefillFeatureEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_COMBINED_KV_PREFILL"
    ] else {
        return true
    }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

private let gemma4VerifyCombinedKVPrefillBitsFeatureEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_VERIFY_COMBINED_KV_PREFILL_BITS"
    ] else {
        return false
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()
