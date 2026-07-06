import Foundation
import MLX
import MLXFastCore

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

    /// A cap for full-attention layers' non-windowed cache, comfortably
    /// above every scored sequence length used by the correctness, GPQA, and
    /// benchmark protocols (see `MLXFastCore.Constants`), while staying small
    /// enough that pre-allocation never approaches this bound in practice --
    /// the buffer only grows in small on-demand increments.
    public static let unboundedCacheCap = 1 << 16

    public init(config: Gemma4Config) {
        self.layers = config.layerTypes.map { layerType in
            let maxSize = layerType == .sliding ? config.slidingWindow : Gemma4ModelCache.unboundedCacheCap
            return Gemma4LayerCache(
                keys: Gemma4KVCache(maxSize: maxSize),
                values: Gemma4KVCache(maxSize: maxSize)
            )
        }
    }

    func arraysForMaterialization() -> [MLXArray] {
        layers.flatMap { $0.arraysForMaterialization() }
    }

    public func materializeCachedState() {
        for array in arraysForMaterialization() {
            eval(array)
        }
    }
}
