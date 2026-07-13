import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon

public struct Gemma4CachedArray {
    public let value: MLXArray
    public let offset: Int

    public init(value: MLXArray, offset: Int) {
        self.value = value
        self.offset = offset
    }
}

/// Windowed (or effectively unbounded) KV array cache backed by a
/// preallocated buffer.
///
/// New positions are appended in place into spare capacity
/// (`buffer[..., end..<end+n, ...] = new`) and callers get a contiguous
/// slice view over the buffer, rather than re-concatenating and copying the
/// whole retained window on every step. Sliding-window layers pass
/// `maxSize == slidingWindow`; full-attention layers pass a very large
/// `maxSize` so the eviction branch is never exercised in practice (the
/// buffer still only grows on demand, in `capacityQuantum` increments, so
/// this does not preallocate the full bound up front) -- see
/// `AGENTS.md`/`CLAUDE.md` for why a simple full cache is an accepted
/// baseline as long as masking stays correct.
public final class Gemma4KVCache {
    public let maxSize: Int
    public private(set) var offset: Int
    public private(set) var startPosition: Int

    /// Backing storage padded to capacity. Absolute position `p` lives at
    /// buffer index `p - bufferStart`; the retained window occupies indices
    /// `[startPosition - bufferStart, writeEnd)`.
    private var buffer: MLXArray?
    private var bufferStart = 0
    private var writeEnd = 0

    private static let capacityQuantum = 256

    public init(maxSize: Int, offset: Int = 0, startPosition: Int = 0) {
        self.maxSize = maxSize
        self.offset = offset
        self.startPosition = startPosition
    }

    public func updateAndFetch(_ newValue: MLXArray) throws -> Gemma4CachedArray {
        guard maxSize > 0 else {
            throw MLXFastError.invalidInput("KV cache maxSize must be positive")
        }
        guard newValue.shape.count == 4 else {
            throw MLXFastError.invalidInput("KV cache input must have shape [batch, heads, length, hidden]")
        }
        let incoming = newValue.shape[2]
        guard incoming > 0 else {
            if let retained = retainedView() {
                return Gemma4CachedArray(value: retained, offset: startPosition)
            }
            return Gemma4CachedArray(value: newValue, offset: offset)
        }

        let combinedStart = startPosition
        append(newValue)
        offset += incoming

        let combinedLength = bufferStart + writeEnd - combinedStart
        let combined = buffer![
            0...,
            0...,
            (combinedStart - bufferStart)..<writeEnd,
            0...
        ]
        if combinedLength > maxSize {
            startPosition = combinedStart + (combinedLength - maxSize)
        }
        return Gemma4CachedArray(value: combined, offset: combinedStart)
    }

    func arraysForMaterialization() -> [MLXArray] {
        retainedView().map { [$0] } ?? []
    }

    private func append(_ newValue: MLXArray) {
        if let buffer,
            buffer.dtype == newValue.dtype,
            buffer.shape[0] == newValue.shape[0],
            buffer.shape[1] == newValue.shape[1],
            buffer.shape[3] == newValue.shape[3],
            writeEnd + newValue.shape[2] <= buffer.shape[2]
        {
            let end = writeEnd + newValue.shape[2]
            buffer[0..., 0..., writeEnd..<end, 0...] = newValue
            writeEnd = end
            return
        }

        let combined = retainedView().map { concatenated([$0, newValue], axis: 2) } ?? newValue
        let needed = combined.shape[2]
        let capacity = Self.paddedCapacity(for: needed)
        let storage = zeros(
            [combined.shape[0], combined.shape[1], capacity, combined.shape[3]],
            dtype: combined.dtype
        )
        storage[0..., 0..., 0..<needed, 0...] = combined
        buffer = storage
        bufferStart = startPosition
        writeEnd = needed
    }

    private func retainedView() -> MLXArray? {
        guard let buffer else {
            return nil
        }
        return buffer[0..., 0..., (startPosition - bufferStart)..<writeEnd, 0...]
    }

    private static func paddedCapacity(for length: Int) -> Int {
        let target = length + capacityQuantum
        return ((target + capacityQuantum - 1) / capacityQuantum) * capacityQuantum
    }
}

public final class Gemma4LayerCache {
    public let keys: Gemma4KVCache
    public let values: Gemma4KVCache

    public init(keys: Gemma4KVCache, values: Gemma4KVCache) {
        self.keys = keys
        self.values = values
    }

    func arraysForMaterialization() -> [MLXArray] {
        keys.arraysForMaterialization() + values.arraysForMaterialization()
    }
}

public final class Gemma4ModelCache {
    public let layers: [Gemma4LayerCache]

    /// Host-side mirror of the library KV-cache position. Compilable caches
    /// store their offset in an MLXArray, so reading `KVCache.offset` after
    /// compilation synchronizes the GPU on every decode step. Keep the cheap
    /// caller-contract check without that readback.
    public private(set) var expectedPositionOffset = 0

    /// A cap for full-attention layers' non-windowed cache, comfortably
    /// above every scored sequence length used by the correctness, GPQA, and
    /// benchmark protocols (see `MLXFastCore.Constants`), while staying small
    /// enough that pre-allocation never approaches this bound in practice --
    /// the buffer only grows in small on-demand increments.
    public static let unboundedCacheCap = 1 << 16

    /// The mlx-swift-lm per-layer KV caches. Created lazily from the loaded
    /// model on the first forward (the model owns `newCache`, which this holder's
    /// `init(config:)` has no access to), then reused across the seed prefill and
    /// every decode step so the cache offset advances and supplies RoPE
    /// positions for free.
    public private(set) var kvCaches: [any KVCache]?

    /// Compiled single-token decode step (mlx-swift-lm `CompiledDecode`). Built
    /// lazily on the first decode step, after the seed prefill has populated the
    /// caches, then reused for every subsequent step. nil when compiled decode is
    /// disabled/unsupported (the adapter falls back to the eager per-step forward).
    var compiledDecodeStep: (@Sendable ([MLXArray]) -> [MLXArray])?

    /// Adopt the compilable caches that `CompiledDecode.setupCompiledDecode`
    /// promoted in place, so later steps and `materializeCachedState` see them.
    func adoptKVCaches(_ caches: [any KVCache]) {
        kvCaches = caches
    }

    /// CompiledDecode only promotes its own concrete split-cache classes.
    /// Restore those types immediately before setup; eager decode otherwise
    /// keeps the K/V-major combined allocation.
    func convertCombinedCachesToUpstream(_ caches: inout [any KVCache]) {
        for index in caches.indices {
            if let combined = caches[index] as? Gemma4CombinedKVCache {
                caches[index] = combined.makeUpstreamCache()
            }
        }
        kvCaches = caches
    }

    func nextExpectedPositionOffset(
        positionOffset: Int,
        inputLength: Int
    ) throws -> Int {
        try advancedCachePosition(
            positionOffset: positionOffset,
            expectedPositionOffset: expectedPositionOffset,
            inputLength: inputLength
        )
    }

    func commitExpectedPositionOffset(_ positionOffset: Int) {
        expectedPositionOffset = positionOffset
    }

    public init(config: Gemma4Config) {
        self.layers = config.layerTypes.map { layerType in
            let maxSize = layerType == .sliding ? config.slidingWindow : Gemma4ModelCache.unboundedCacheCap
            return Gemma4LayerCache(
                keys: Gemma4KVCache(maxSize: maxSize),
                values: Gemma4KVCache(maxSize: maxSize)
            )
        }
    }

    /// Bespoke per-layer KV arrays (retained for the standalone Gemma4KVCache
    /// tests; the scored path uses the library `kvCaches` instead).
    func arraysForMaterialization() -> [MLXArray] {
        layers.flatMap { $0.arraysForMaterialization() }
    }

    /// Return this cache's `[KVCache]`, creating it from the model on first use.
    public func kvCache(for model: Gemma4RuntimeModel) -> [any KVCache] {
        if let kvCaches {
            return kvCaches
        }
        let created = model.newCache(parameters: nil)
        kvCaches = created
        return created
    }

    public func materializeCachedState() {
        if let kvCaches {
            eval(kvCaches)
        }
    }
}

func advancedCachePosition(
    positionOffset: Int,
    expectedPositionOffset: Int,
    inputLength: Int
) throws -> Int {
    guard positionOffset >= 0, expectedPositionOffset >= 0 else {
        throw MLXFastError.invalidInput("Gemma 4 position offset must be non-negative")
    }
    guard positionOffset == expectedPositionOffset else {
        throw MLXFastError.invalidInput(
            "Gemma 4 position offset \(positionOffset) does not match expected cache offset \(expectedPositionOffset)"
        )
    }
    guard inputLength > 0 else {
        throw MLXFastError.invalidInput("Gemma 4 cached input length must be positive")
    }
    let (nextPositionOffset, overflow) = expectedPositionOffset.addingReportingOverflow(
        inputLength
    )
    guard !overflow else {
        throw MLXFastError.invalidInput("Gemma 4 cache position offset overflows Int")
    }
    return nextPositionOffset
}
