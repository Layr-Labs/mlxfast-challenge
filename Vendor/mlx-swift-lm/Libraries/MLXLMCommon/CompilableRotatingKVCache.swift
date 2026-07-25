// CompilableRotatingKVCache: compile-traceable rotating KV cache.
//
// Ported from osaurus-ai/vmlx-swift-lm
// (Libraries/MLXLMCommon/BatchEngine/CompilableRotatingKVCache.swift).
//
// Standard RotatingKVCache breaks compile() in two ways:
//   1. Buffer growth via concat rebinds the keys property — the trace
//      captures the original object and rebinding loses it.
//   2. Ring-buffer wrap via Swift Int reset doesn't match the trace's
//      assumed linear layout.
//
// This subclass fixes both by:
//   - Pre-allocating the unified buffer to maxCacheSize at promotion time.
//     No concat-growth during decode.
//   - Tracking the write index as idxArray (MLXArray [1] int32). Wrap
//     arithmetic uses MLXArray ops (where_ + modulo) so the compile tracer
//     can follow.
//   - Returning the FULL [B, H, maxCacheSize, D] buffer from update().
//     The attention mask restricts valid positions.
//
// Initialise via init(from:) on an existing RotatingKVCache populated by
// prefill, or via the static promote(from:maxLength:) helper.

import Foundation
import MLX
import MLXNN

/// Compile-traceable specialisation of ``RotatingKVCache``.
///
/// Compared to the parent:
///
/// - `keys` and `values` are eagerly allocated at `[B, H, maxCacheSize, D]`
///   at promotion time. No growth-via-concat during decode.
/// - `idxArray: MLXArray[1] int32` replaces Swift-Int `idx`. All wrap
///   arithmetic happens in MLXArray ops so the compile tracer can follow.
/// - `offsetArray: MLXArray[1] int32` mirrors `offset`. Used by `makeMask`
///   for the causal upper bound; tracked as an MLXArray so the tracer
///   follows per-step advances.
/// - `update(keys:values:)` writes new tokens at `idxArray` via
///   `dynamicSliceUpdate`, then advances `idxArray` with wrap semantics
///   entirely in MLXArray space.
/// - `makeMask` uses graph counters for partial rings; the production
///   single-token/full-window graph proves all 512 rows valid and returns `.none`.
public final class CompilableRotatingKVCache: RotatingKVCache, @unchecked Sendable {

    /// Graph-visible next-write slot, normalized into `[keep, maxCacheSize)`.
    public var idxArray: MLXArray

    /// Graph-visible total number of tokens committed to this cache.
    public var offsetArray: MLXArray

    private lazy var maskRinds = MLXArray(Int32(0) ..< Int32(maxCacheSize))

    /// The full-window `.none` mask branch is legal only after promotion has
    /// proved every physical ring slot valid.
    private var canElideFullWindowDecodeMask = false

    /// Deferred Laguna caches use stock rotating semantics through prefill.
    public private(set) var isCompiledMode: Bool

    /// Laguna mirrors graph counters on the host after a successful call.
    /// Generic compiled caches retain the historical graph-authoritative mode.
    private var mirrorsCompiledStateOnHost = false

    // MARK: - Init

    /// Historical direct-compiled constructor used by generic CompiledDecode.
    public override init(maxSize: Int, keep: Int = 0, step: Int = 256) {
        self.idxArray = MLXArray([Int32(0)])
        self.offsetArray = MLXArray([Int32(0)])
        self.isCompiledMode = true
        super.init(maxSize: maxSize, keep: keep, step: step)
    }

    /// Laguna constructor: ordinary cache first, in-place promotion later.
    public convenience init(
        maxSize: Int,
        keep: Int = 0,
        step: Int = 256,
        deferredPromotion: Bool
    ) {
        self.init(maxSize: maxSize, keep: keep, step: step)
        mirrorsCompiledStateOnHost = deferredPromotion
        isCompiledMode = !deferredPromotion
    }

    /// Host counters never trigger graph readback.
    public var hostOffset: Int { super.offset }
    public var hostIndex: Int { idx }

    /// Promote an already populated stock rotating cache.
    public convenience init(from rotating: RotatingKVCache) {
        self.init(
            maxSize: rotating.maxCacheSize,
            keep: rotating.keep,
            step: rotating.step
        )
        isCompiledMode = false
        idx = rotating.idx
        offset = rotating.offset
        keys = rotating.keys
        values = rotating.values
        _ = prepareForCompiledDecode()
    }

    public static func promote(
        from cache: RotatingKVCache, maxLength _: Int
    ) -> CompilableRotatingKVCache {
        CompilableRotatingKVCache(from: cache)
    }

    /// Pad the ordinary ring to fixed capacity and expose graph counters.
    ///
    /// A stock cache exactly filled by the 512-token seed leaves `idx == 512`
    /// and resets it to `keep` at the start of its next update. Promotion
    /// performs that pending normalization once, before the compiled write.
    @discardableResult
    public func prepareForCompiledDecode() -> Bool {
        if isCompiledMode {
            return keys?.dim(2) == maxCacheSize
                && values?.dim(2) == maxCacheSize
        }
        guard let currentKeys = keys,
            let currentValues = values,
            currentKeys.dim(2) >= min(super.offset, maxCacheSize),
            currentValues.dim(2) >= min(super.offset, maxCacheSize),
            currentKeys.dim(2) <= maxCacheSize,
            currentValues.dim(2) <= maxCacheSize
        else {
            return false
        }

        let keyPadding = maxCacheSize - currentKeys.dim(2)
        if keyPadding > 0 {
            let zeros = MLXArray.zeros(
                [
                    currentKeys.dim(0), currentKeys.dim(1), keyPadding,
                    currentKeys.dim(3),
                ],
                dtype: currentKeys.dtype
            )
            keys = concatenated([currentKeys, zeros], axis: 2)
        }

        let valuePadding = maxCacheSize - currentValues.dim(2)
        if valuePadding > 0 {
            let zeros = MLXArray.zeros(
                [
                    currentValues.dim(0), currentValues.dim(1), valuePadding,
                    currentValues.dim(3),
                ],
                dtype: currentValues.dtype
            )
            values = concatenated([currentValues, zeros], axis: 2)
        }

        if idx == maxCacheSize {
            idx = keep
        }
        guard idx >= keep, idx < maxCacheSize else {
            return false
        }

        idxArray = MLXArray([Int32(idx)])
        offsetArray = MLXArray([Int32(super.offset)])
        canElideFullWindowDecodeMask = super.offset >= maxCacheSize
        isCompiledMode = true
        return true
    }

    /// Resume the parent's eager update/mask behavior on the same ring.
    public func deactivateCompiledDecode() {
        guard isCompiledMode else { return }
        if !mirrorsCompiledStateOnHost {
            idx = idxArray[0].item(Int.self)
            super.offset = offsetArray[0].item(Int.self)
        }
        isCompiledMode = false
    }

    /// Replace every template state reference with one request's state.
    public func bindCompiledStorage(from source: CompilableRotatingKVCache) -> Bool {
        guard source.isCompiledMode,
            source.maxCacheSize == maxCacheSize,
            source.keep == keep,
            source.step == step,
            let sourceKeys = source.keys,
            let sourceValues = source.values,
            sourceKeys.dim(2) == maxCacheSize,
            sourceValues.dim(2) == maxCacheSize
        else {
            return false
        }

        keys = sourceKeys
        values = sourceValues
        idxArray = source.idxArray
        offsetArray = source.offsetArray
        idx = source.hostIndex
        super.offset = source.hostOffset
        canElideFullWindowDecodeMask = source.canElideFullWindowDecodeMask
        isCompiledMode = true
        return true
    }

    /// Allocation-free signature check used by Laguna before binding a
    /// request into its fixed production graph.
    public func compiledStorageMatches(
        batchSize: Int,
        heads: Int,
        length: Int,
        headDimension: Int,
        dtype: DType
    ) -> Bool {
        guard isCompiledMode, let keys, let values else { return false }
        return keys.dtype == dtype
            && keys.ndim == 4
            && keys.dim(0) == batchSize
            && keys.dim(1) == heads
            && keys.dim(2) == length
            && keys.dim(3) == headDimension
            && values.dtype == dtype
            && values.ndim == 4
            && values.dim(0) == batchSize
            && values.dim(1) == heads
            && values.dim(2) == length
            && values.dim(3) == headDimension
            && idxArray.dtype == .int32
            && idxArray.ndim == 1
            && idxArray.dim(0) == 1
            && offsetArray.dtype == .int32
            && offsetArray.ndim == 1
            && offsetArray.dim(0) == 1
    }

    /// Mirror the graph's one-token counter transition after a successful call.
    public func noteCompiledAdvance(_ count: Int = 1) {
        precondition(count >= 0)
        super.offset += count
        let advanced = idx + count
        if advanced < maxCacheSize {
            idx = advanced
        } else {
            idx = keep + (advanced - keep) % (maxCacheSize - keep)
        }
        canElideFullWindowDecodeMask = super.offset >= maxCacheSize
    }

    // MARK: - Update

    public override func update(
        keys newKeys: MLXArray, values newValues: MLXArray
    ) -> (MLXArray, MLXArray) {
        guard isCompiledMode else {
            return super.update(keys: newKeys, values: newValues)
        }

        let nTokens = newKeys.dim(2)
        if keys == nil {
            keys = MLXArray.zeros(
                [newKeys.dim(0), newKeys.dim(1), maxCacheSize, newKeys.dim(3)],
                dtype: newKeys.dtype
            )
            values = MLXArray.zeros(
                [newValues.dim(0), newValues.dim(1), maxCacheSize, newValues.dim(3)],
                dtype: newValues.dtype
            )
        }

        // Preserve physical ring order: write exactly the graph's next-write
        // slot, then advance both K and V state through the same index.
        keys!._updateInternal(
            dynamicSliceUpdate(keys!, update: newKeys, start: idxArray, axes: [2]))
        values!._updateInternal(
            dynamicSliceUpdate(values!, update: newValues, start: idxArray, axes: [2]))

        let advance = MLXArray([Int32(nTokens)])
        let advancedIdx = idxArray + advance
        let maxSizeArray = MLXArray([Int32(maxCacheSize)])
        let keepArray = MLXArray([Int32(keep)])
        let cycleLength = maxSizeArray - keepArray
        let rotatedIdx: MLXArray
        if keep > 0 {
            rotatedIdx = keepArray + ((advancedIdx - keepArray) % cycleLength)
        } else {
            rotatedIdx = advancedIdx % maxSizeArray
        }
        let nextIdx = MLX.`where`(
            advancedIdx .< maxSizeArray, advancedIdx, rotatedIdx)

        idxArray._updateInternal(nextIdx)
        offsetArray._updateInternal(offsetArray + advance)
        return (keys!, values!)
    }

    // MARK: - Mask

    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard isCompiledMode else {
            return super.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
        }

        // The production graph is traced only after a full 512-row ring. A
        // one-token query may attend every physical row, exactly like stock.
        if n == 1, windowSize == maxCacheSize, canElideFullWindowDecodeMask {
            return .none
        }

        let linds: MLXArray
        if n == 1 {
            linds = offsetArray.reshaped(1, 1)
        } else {
            linds = (MLXArray(Int32(0) ..< Int32(n)) + offsetArray).reshaped(n, 1)
        }
        let rinds = maskRinds.reshaped(1, maxCacheSize)
        let causal = linds .>= rinds
        let maxSizeArray = MLXArray([Int32(maxCacheSize)]).reshaped(1, 1)
        let allTrue = MLX.broadcast(
            MLXArray([true]).reshaped(1, 1),
            to: [linds.dim(0), rinds.dim(1)]
        )
        var mask = MLX.`where`(linds .>= maxSizeArray, allTrue, causal)

        if let windowSize {
            let tokenIndices =
                (rinds - idxArray + MLXArray(Int32(maxCacheSize))) % Int32(maxCacheSize)
            mask = mask & (tokenIndices .>= Int32(maxCacheSize - windowSize))
        }
        return .array(mask)
    }

    // MARK: - State

    /// Stable compiled state order: `[K, V, idx, offset]`.
    public override func innerState() -> [MLXArray] {
        guard isCompiledMode else {
            return super.innerState()
        }
        var state: [MLXArray] = []
        if let keys {
            state.append(keys)
        }
        if let values {
            state.append(values)
        }
        state.append(idxArray)
        state.append(offsetArray)
        return state
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        guard isCompiledMode else {
            return super.trim(n)
        }
        if !mirrorsCompiledStateOnHost {
            idx = idxArray[0].item(Int.self)
            super.offset = offsetArray[0].item(Int.self)
        }
        let trimmed = super.trim(n)
        idxArray = MLXArray([Int32(idx)])
        offsetArray = MLXArray([Int32(super.offset)])
        canElideFullWindowDecodeMask = super.offset >= maxCacheSize
        return trimmed
    }

    public override func copy() -> any KVCache {
        let logicalIndex: Int
        let logicalOffset: Int
        if isCompiledMode && !mirrorsCompiledStateOnHost {
            logicalIndex = idxArray[0].item(Int.self)
            logicalOffset = offsetArray[0].item(Int.self)
        } else {
            logicalIndex = idx
            logicalOffset = super.offset
        }

        let copy = CompilableRotatingKVCache(
            maxSize: maxCacheSize,
            keep: keep,
            step: step,
            deferredPromotion: mirrorsCompiledStateOnHost
        )
        if let keys {
            copy.keys = keys[.ellipsis]
        }
        if let values {
            copy.values = values[.ellipsis]
        }
        copy.idx = logicalIndex
        copy.offset = logicalOffset
        copy.idxArray = idxArray[0...]
        copy.offsetArray = offsetArray[0...]
        copy.canElideFullWindowDecodeMask = canElideFullWindowDecodeMask
        copy.isCompiledMode = isCompiledMode
        return copy
    }
}
