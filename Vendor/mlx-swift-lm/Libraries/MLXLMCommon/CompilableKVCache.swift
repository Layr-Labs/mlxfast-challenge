// CompilableKVCache: Fixed-size KV cache using the Overflow Bin pattern.
//
// Ported from osaurus-ai/vmlx-swift-lm (Libraries/MLXLMCommon/CompilableKVCache.swift).
// This is the foundation type for whole-step compiled decode: it is the only
// full-attention cache whose `update()` is graph-traceable, so a decode step
// (model forward + cache write) can be captured by `compile(inputs:outputs:)`.
//
// Standard KVCacheSimple returns keys[..<offset] — a dynamically sized slice that
// changes every decode step. This prevents compile() from tracing through the cache
// because DynamicSlice requires static slice_size.
//
// CompilableKVCache solves this by:
// 1. Pre-allocating a fixed-size buffer [B, H, maxLength, D]
// 2. Writing new keys/values via dynamicSliceUpdate (compile-traceable writes)
// 3. Returning the FULL buffer from update() — constant shape every step
// 4. Generating a boolean attention mask in makeMask() that marks active positions
//
// The attention kernel handles the masking, computing only on valid positions.
// This trades marginal redundant compute (masked zeros) for enabling compile()
// to fuse hundreds of FFI crossings into a single compiled call.
//
// Usage:
//   // After prefill with standard cache, convert for compiled decode:
//   let compilableCache = standardCache.map { c in
//       CompilableKVCache(from: c, maxLength: 2048)
//   }
//
// `CompiledDecode` uses this type for generic single-stream generation.
// Laguna additionally constructs it in deferred mode: seed prefill retains
// ordinary `KVCacheSimple` behavior, then the same object is promoted and
// rebound into a process-lifetime compiled decode executor.

import Foundation
import MLX
import MLXNN

/// A KV cache that returns fixed-size buffers to enable compile().
///
/// Key differences from KVCacheSimple:
/// - `offsetArray` (MLXArray) tracks position in the computation graph
/// - Pre-allocated buffer of fixed size (no dynamic growth during decode)
/// - `update()` returns the FULL buffer — mask handles which positions are valid
/// - `makeMask()` always returns an array mask covering the full buffer
/// - `innerState()` returns [keys, values, offsetArray] — all compile-tracked
public class CompilableKVCache: BaseKVCache {

    public var keys: MLXArray?
    public var values: MLXArray?

    /// Offset as MLXArray (1D [1] int32) — tracked by the compile tracer.
    /// It remains separate from the host mirror used by Laguna's worker
    /// protocol so compiled decode never needs a synchronous `.item()` call.
    public var offsetArray: MLXArray

    /// Maximum sequence length the fixed overflow-bin buffer can hold.
    public let maxLength: Int

    /// Ordinary-cache allocation step, matching `KVCacheSimple.step`.
    public var step: Int

    /// Fixed key-column indices used by the compiled overflow-bin mask.
    private lazy var maskRinds = MLXArray(Int32(0) ..< Int32(maxLength))

    /// Deferred Laguna caches remain ordinary until the seed is complete.
    /// Generic `CompiledDecode` caches retain their historical direct-compiled
    /// behavior by leaving `deferredPromotion` at its default value.
    public private(set) var isCompiledMode: Bool

    /// When true, `BaseKVCache.offset` is the authoritative host mirror and is
    /// advanced by the direct executor after each successful graph call.
    private let mirrorsCompiledOffsetOnHost: Bool

    public init(
        maxLength: Int = 4096,
        step: Int = 256,
        deferredPromotion: Bool = false
    ) {
        precondition(maxLength > 0)
        precondition(step > 0)
        self.maxLength = maxLength
        self.step = step
        self.offsetArray = MLXArray([Int32(0)])
        self.isCompiledMode = !deferredPromotion
        self.mirrorsCompiledOffsetOnHost = deferredPromotion
        super.init()
    }

    /// The host-only logical position. Unlike `offset`, this never reads back
    /// `offsetArray`; the direct executor and atlas routing use it for guards.
    public var hostOffset: Int { super.offset }

    /// Static promotion helper used by the generic compiled-decode path.
    public static func promote(from cache: KVCacheSimple, maxLength: Int) -> CompilableKVCache {
        CompilableKVCache(from: cache, maxLength: maxLength)
    }

    /// Create a direct-compiled cache from an already populated ordinary cache.
    public convenience init(from cache: KVCache, maxLength: Int = 4096) {
        self.init(maxLength: maxLength)
        let existingState = cache.state
        if existingState.count >= 2 {
            state = [existingState[0], existingState[1]]
        }
    }

    /// Promote the ordinary dynamic buffer in place to a fixed overflow bin.
    ///
    /// This is deliberately outside the compiled closure. The request keeps
    /// ownership of this cache object while the process-lifetime template is
    /// rebound to these exact array objects before every graph invocation.
    @discardableResult
    public func prepareForCompiledDecode() -> Bool {
        if isCompiledMode {
            return hostOffset < maxLength
                && keys?.dim(2) == maxLength
                && values?.dim(2) == maxLength
        }
        guard hostOffset < maxLength,
            let currentKeys = keys,
            let currentValues = values,
            currentKeys.dim(2) >= hostOffset,
            currentValues.dim(2) >= hostOffset,
            currentKeys.dim(2) <= maxLength,
            currentValues.dim(2) <= maxLength
        else {
            return false
        }

        let keyPadding = maxLength - currentKeys.dim(2)
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

        let valuePadding = maxLength - currentValues.dim(2)
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

        offsetArray = MLXArray([Int32(hostOffset)])
        isCompiledMode = true
        return true
    }

    /// Resume ordinary `KVCacheSimple` semantics on the same physical buffer.
    /// At offset 768 its next eager update performs the stock 256-row growth,
    /// writes row 768 once, and returns the valid 769-row prefix.
    public func deactivateCompiledDecode() {
        guard isCompiledMode else { return }
        if !mirrorsCompiledOffsetOnHost {
            super.offset = offsetArray[0].item(Int.self)
        }
        isCompiledMode = false
    }

    /// Rebind a persistent graph template to request-owned state.
    ///
    /// `compile(inputs:outputs:)` asks `innerState()` on every invocation, so
    /// replacing all three references here feeds the new request into the
    /// already compiled graph without retracing or retaining prior counters.
    public func bindCompiledStorage(from source: CompilableKVCache) -> Bool {
        guard source.isCompiledMode,
            source.maxLength == maxLength,
            let sourceKeys = source.keys,
            let sourceValues = source.values,
            sourceKeys.dim(2) == maxLength,
            sourceValues.dim(2) == maxLength
        else {
            return false
        }

        keys = sourceKeys
        values = sourceValues
        offsetArray = source.offsetArray
        super.offset = source.hostOffset
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
            && offsetArray.dtype == .int32
            && offsetArray.ndim == 1
            && offsetArray.dim(0) == 1
    }

    /// Advance only the host mirror after the graph has successfully advanced
    /// `offsetArray` and written one supplied token's K/V rows.
    public func noteCompiledAdvance(_ count: Int = 1) {
        precondition(count >= 0)
        super.offset += count
    }

    // MARK: - KVCache protocol

    public override var offset: Int {
        get {
            if isCompiledMode && !mirrorsCompiledOffsetOnHost {
                return offsetArray[0].item(Int.self)
            }
            return super.offset
        }
        set {
            super.offset = newValue
            if isCompiledMode {
                offsetArray = MLXArray([Int32(newValue)])
            }
        }
    }

    /// Stable compiled state order: `[K, V, offset]`.
    public override func innerState() -> [MLXArray] {
        guard isCompiledMode else {
            return [keys, values].compactMap { $0 }
        }
        if let keys, let values {
            return [keys, values, offsetArray]
        }
        return [offsetArray]
    }

    public override func update(keys newKeys: MLXArray, values newValues: MLXArray)
        -> (MLXArray, MLXArray)
    {
        guard isCompiledMode else {
            return eagerUpdate(keys: newKeys, values: newValues)
        }

        let nTokens = newKeys.dim(2)
        if keys == nil {
            keys = MLXArray.zeros(
                [newKeys.dim(0), newKeys.dim(1), maxLength, newKeys.dim(3)],
                dtype: newKeys.dtype
            )
            values = MLXArray.zeros(
                [newValues.dim(0), newValues.dim(1), maxLength, newValues.dim(3)],
                dtype: newValues.dtype
            )
        }

        // `prev` is the pre-update graph position. Both writes use it, then
        // the graph counter advances exactly once.
        let prev = offsetArray
        let newOffset = prev + MLXArray([Int32(nTokens)])
        keys!._updateInternal(
            dynamicSliceUpdate(keys!, update: newKeys, start: prev, axes: [2]))
        values!._updateInternal(
            dynamicSliceUpdate(values!, update: newValues, start: prev, axes: [2]))
        offsetArray._updateInternal(newOffset)

        // Overflow-bin attention always receives the fixed backing arrays.
        return (keys!, values!)
    }

    /// Verbatim `KVCacheSimple.update` semantics for deferred prefill and
    /// post-capacity fallback.
    private func eagerUpdate(keys newKeys: MLXArray, values newValues: MLXArray)
        -> (MLXArray, MLXArray)
    {
        let previous = super.offset
        let tokenCount = newKeys.dim(2)

        if keys == nil, previous == 0, tokenCount > 0,
            tokenCount.isMultiple(of: step)
        {
            keys = newKeys
            values = newValues
            super.offset = tokenCount
            return (newKeys, newValues)
        }

        let reset =
            if let currentKeys = keys, (previous + tokenCount) > currentKeys.dim(2) {
                true
            } else {
                keys == nil
            }
        if reset {
            let keyShape = [
                newKeys.dim(0), newKeys.dim(1),
                ((step + tokenCount - 1) / step) * step, newKeys.dim(3),
            ]
            let valueShape = [
                newValues.dim(0), newValues.dim(1),
                ((step + tokenCount - 1) / step) * step, newValues.dim(3),
            ]
            let newKeyStorage = MLXArray.zeros(keyShape, dtype: newKeys.dtype)
            let newValueStorage = MLXArray.zeros(valueShape, dtype: newValues.dtype)

            if var currentKeys = keys, var currentValues = values {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                keys = concatenated([currentKeys, newKeyStorage], axis: 2)
                values = concatenated([currentValues, newValueStorage], axis: 2)
            } else {
                keys = newKeyStorage
                values = newValueStorage
            }
        }

        super.offset += tokenCount
        keys?[.ellipsis, previous ..< super.offset, 0...] = newKeys
        values?[.ellipsis, previous ..< super.offset, 0...] = newValues
        return (
            keys![.ellipsis, ..<super.offset, 0...],
            values![.ellipsis, ..<super.offset, 0...]
        )
    }

    /// In compiled mode a boolean prefix mask admits precisely columns
    /// `0...preOffset`, including the row just written by the current token.
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard isCompiledMode else {
            return super.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
        }

        let linds: MLXArray
        if n == 1 {
            linds = offsetArray.reshaped(1, 1)
        } else {
            linds = (MLXArray(Int32(0) ..< Int32(n)) + offsetArray).reshaped(n, 1)
        }
        let rinds = maskRinds.reshaped(1, maxLength)
        var mask = linds .>= rinds
        if let windowSize {
            mask = mask & (rinds .>= linds - Int32(windowSize - 1))
        }
        return .array(mask)
    }

    public override var state: [MLXArray] {
        get {
            guard isCompiledMode else {
                guard let keys, let values else { return [] }
                if super.offset == keys.dim(2) {
                    return [keys, values]
                }
                return [
                    keys[.ellipsis, ..<super.offset, 0...],
                    values[.ellipsis, ..<super.offset, 0...],
                ]
            }

            guard let keys, let values else { return [] }
            let logicalOffset = offset
            if logicalOffset == keys.dim(2) {
                return [keys, values]
            }
            return [
                keys[.ellipsis, ..<logicalOffset, 0...],
                values[.ellipsis, ..<logicalOffset, 0...],
            ]
        }
        set {
            guard newValue.count == 2 else { return }
            let sequenceLength = newValue[0].dim(2)

            if isCompiledMode {
                guard sequenceLength <= maxLength else { return }
                keys = MLXArray.zeros(
                    [
                        newValue[0].dim(0), newValue[0].dim(1), maxLength,
                        newValue[0].dim(3),
                    ],
                    dtype: newValue[0].dtype
                )
                values = MLXArray.zeros(
                    [
                        newValue[1].dim(0), newValue[1].dim(1), maxLength,
                        newValue[1].dim(3),
                    ],
                    dtype: newValue[1].dtype
                )
                keys?[.ellipsis, ..<sequenceLength, 0...] = newValue[0]
                values?[.ellipsis, ..<sequenceLength, 0...] = newValue[1]
                super.offset = sequenceLength
                offsetArray = MLXArray([Int32(sequenceLength)])
            } else {
                keys = newValue[0]
                values = newValue[1]
                super.offset = sequenceLength
            }
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        if isCompiledMode && !mirrorsCompiledOffsetOnHost {
            super.offset = offsetArray[0].item(Int.self)
        }
        let trimmed = min(super.offset, n)
        super.offset -= trimmed
        if isCompiledMode {
            offsetArray = MLXArray([Int32(super.offset)])
        }
        return trimmed
    }

    public override func copy() -> any KVCache {
        let logicalOffset = offset
        let copy = CompilableKVCache(
            maxLength: maxLength,
            step: step,
            deferredPromotion: mirrorsCompiledOffsetOnHost
        )
        if let keys {
            copy.keys = keys[.ellipsis]
        }
        if let values {
            copy.values = values[.ellipsis]
        }
        copy.isCompiledMode = isCompiledMode
        copy.offsetArray = offsetArray[0...]
        copy.offset = logicalOffset
        if isCompiledMode {
            copy.offsetArray = offsetArray[0...]
        }
        return copy
    }

    public var debugDescription: String {
        "CompilableKVCache(offset=\(offset), maxLength=\(maxLength), "
            + "compiled=\(isCompiledMode), shape=\(keys?.shape.description ?? "nil"))"
    }
}
