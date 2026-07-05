import Foundation
import MLX
import MLXFastCore

public struct DeepSeekCachedKV {
    public let kv: MLXArray
    public let keyOffset: Int

    public init(kv: MLXArray, keyOffset: Int) {
        self.kv = kv
        self.keyOffset = keyOffset
    }
}

/// Sliding-window KV cache backed by a preallocated buffer.
///
/// The previous implementation ran `concatenated([cached, new], axis: 2)`
/// followed by a window slice on every update, reallocating and copying the
/// whole retained window per layer per decode step. This version appends new
/// positions in place into spare capacity (`buffer[..., end..<end+n, ...] =
/// new`) and hands attention a contiguous slice view over the buffer.
///
/// Bit-exactness: the view returned to attention covers exactly the same
/// positions, in the same order, with the same stored values as the concat
/// version produced (retained window followed by the incoming keys). The
/// sliding-window drop is pure offset bookkeeping (`startPosition` advances;
/// the next step's view simply starts later), which matches slicing off the
/// oldest positions. In-place writes require an exact dtype/shape match, so
/// stored bytes equal the incoming array's bytes; every fallback path rebuilds
/// through the identical `concatenated` arithmetic the cache used before.
public final class DeepSeekLocalKVCache {
    public let maxSize: Int
    public private(set) var offset: Int
    public private(set) var startPosition: Int

    /// Backing storage padded to capacity. Absolute position `p` lives at
    /// buffer index `p - bufferStart`; the retained window occupies indices
    /// `[startPosition - bufferStart, writeEnd)`. Indices at `writeEnd` and
    /// beyond are unused capacity and are never part of any returned view.
    private var buffer: MLXArray?
    private var bufferStart = 0
    private var writeEnd = 0

    /// Capacity padding: a 512-token prompt allocates 768 positions, so a
    /// full 128-step decode (one appended position per step) never needs a
    /// second allocation. Overflow compacts the retained window (at most
    /// `maxSize` positions) into a fresh padded buffer.
    private static let capacityQuantum = 256

    public init(maxSize: Int, offset: Int = 0, startPosition: Int = 0) {
        self.maxSize = maxSize
        self.offset = offset
        self.startPosition = startPosition
    }

    public func updateAndFetch(_ newKV: MLXArray) throws -> DeepSeekCachedKV {
        guard maxSize > 0 else {
            throw MLXFastError.invalidInput("local KV cache maxSize must be positive")
        }
        guard newKV.shape.count == 4 else {
            throw MLXFastError.invalidInput("local KV cache input must have shape [batch, heads, length, hidden]")
        }
        let incoming = newKV.shape[2]
        guard incoming > 0 else {
            if let retained = retainedView() {
                return DeepSeekCachedKV(kv: retained, keyOffset: startPosition)
            }
            return DeepSeekCachedKV(kv: newKV, keyOffset: offset)
        }

        let combinedStart = startPosition
        append(newKV)
        offset += incoming

        // Same combined tensor the concat version returned: every retained
        // position from `combinedStart`, then the new keys, in append order.
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
        return DeepSeekCachedKV(kv: combined, keyOffset: combinedStart)
    }

    func arraysForMaterialization() -> [MLXArray] {
        retainedView().map { [$0] } ?? []
    }

    /// Appends `newKV` after the last written position. The fast path writes
    /// in place into spare capacity. The fallback (first allocation, capacity
    /// overflow, or defensive shape/dtype mismatch) rebuilds via the same
    /// `concatenated([retained, new])` arithmetic the cache used before
    /// preallocation and copies it into a fresh padded buffer, so the stored
    /// values are identical on every path.
    private func append(_ newKV: MLXArray) {
        if let buffer,
            buffer.dtype == newKV.dtype,
            buffer.shape[0] == newKV.shape[0],
            buffer.shape[1] == newKV.shape[1],
            buffer.shape[3] == newKV.shape[3],
            writeEnd + newKV.shape[2] <= buffer.shape[2]
        {
            // Writes touch only indices >= writeEnd, which no previously
            // returned view covers, so earlier views keep their values even
            // when MLX updates the buffer in place.
            let end = writeEnd + newKV.shape[2]
            buffer[0..., 0..., writeEnd..<end, 0...] = newKV
            writeEnd = end
            return
        }

        let combined = retainedView().map { concatenated([$0, newKV], axis: 2) } ?? newKV
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

public struct DeepSeekPoolingWindow {
    public let kv: MLXArray
    public let gate: MLXArray
    public let baseOffset: Int
}

public final class DeepSeekPoolingCache {
    public let ratio: Int
    private var bufferedKV: MLXArray?
    private var bufferedGate: MLXArray?

    /// Pooled history backed by a preallocated buffer. `pooledCount` rows are
    /// valid and `pooledView` memoizes the `[0..<pooledCount]` prefix view
    /// handed to attention, so steps that pool no new window (three of every
    /// four decode steps at ratio 4) add no graph nodes at all. The remainder
    /// accumulator (`bufferedKV`/`bufferedGate`) intentionally keeps its
    /// original concat form: it drains from the front each step, which does
    /// not fit an append-only buffer, and its arrays are at most `ratio`
    /// positions long.
    private var pooledBuffer: MLXArray?
    private var pooledCount = 0
    private var pooledView: MLXArray?

    private static let capacityQuantum = 64

    public init(ratio: Int) {
        self.ratio = ratio
    }

    public var pooledLength: Int {
        pooledCount
    }

    public func accumulateWindows(
        kv newKV: MLXArray,
        gate newGate: MLXArray,
        offset: Int
    ) throws -> DeepSeekPoolingWindow {
        guard ratio > 0 else {
            throw MLXFastError.invalidInput("pooling cache ratio must be positive")
        }
        guard newKV.shape.count == 3, newGate.shape.count == 3 else {
            throw MLXFastError.invalidInput("pooling cache inputs must have shape [batch, length, hidden]")
        }
        guard newKV.shape[0] == newGate.shape[0], newKV.shape[1] == newGate.shape[1] else {
            throw MLXFastError.invalidInput("pooling cache kv/gate batch and length must match")
        }

        let previousRemainder = bufferedKV?.shape[1] ?? 0
        let combinedKV = bufferedKV.map { concatenated([$0, newKV], axis: 1) } ?? newKV
        let combinedGate = bufferedGate.map { concatenated([$0, newGate], axis: 1) } ?? newGate
        let total = combinedKV.shape[1]
        let usable = (total / ratio) * ratio
        let remainder = total - usable

        let readyKV: MLXArray
        let readyGate: MLXArray
        if usable > 0 {
            readyKV = combinedKV[0..., 0..<usable, 0...]
            readyGate = combinedGate[0..., 0..<usable, 0...]
        } else {
            readyKV = zeros([newKV.shape[0], 0, newKV.shape[2]], dtype: newKV.dtype)
            readyGate = zeros([newGate.shape[0], 0, newGate.shape[2]], dtype: newGate.dtype)
        }

        if remainder > 0 {
            bufferedKV = combinedKV[0..., usable..., 0...]
            bufferedGate = combinedGate[0..., usable..., 0...]
        } else {
            bufferedKV = nil
            bufferedGate = nil
        }

        return DeepSeekPoolingWindow(
            kv: readyKV,
            gate: readyGate,
            baseOffset: offset - previousRemainder
        )
    }

    public func updateAndFetch(_ newPooled: MLXArray) -> MLXArray {
        let incoming = newPooled.shape[1]
        guard incoming > 0 else {
            if let pooledView {
                return pooledView
            }
            return zeros([newPooled.shape[0], 0, newPooled.shape[2]], dtype: newPooled.dtype)
        }

        if let pooledBuffer,
            pooledBuffer.dtype == newPooled.dtype,
            pooledBuffer.shape[0] == newPooled.shape[0],
            pooledBuffer.shape[2] == newPooled.shape[2],
            pooledCount + incoming <= pooledBuffer.shape[1]
        {
            // Release the memoized view before the in-place append so it does
            // not pin the buffer's previous node (letting MLX donate the
            // buffer instead of copying it). Rows [0..<pooledCount) are never
            // rewritten, so earlier views keep their values.
            pooledView = nil
            let end = pooledCount + incoming
            pooledBuffer[0..., pooledCount..<end, 0...] = newPooled
            pooledCount = end
            let view = pooledBuffer[0..., 0..<end, 0...]
            pooledView = view
            return view
        }

        // First allocation, capacity overflow, or defensive shape/dtype
        // mismatch: rebuild with the exact concat arithmetic used before
        // preallocation, then copy into a fresh padded buffer.
        let combined = pooledView.map { concatenated([$0, newPooled], axis: 1) } ?? newPooled
        let needed = combined.shape[1]
        let capacity = Self.paddedCapacity(for: needed)
        let storage = zeros(
            [combined.shape[0], capacity, combined.shape[2]],
            dtype: combined.dtype
        )
        storage[0..., 0..<needed, 0...] = combined
        pooledBuffer = storage
        pooledCount = needed
        let view = storage[0..., 0..<needed, 0...]
        pooledView = view
        return view
    }

    public func makeMask(queryLength: Int, offset: Int) -> MLXArray? {
        guard pooledCount > 0, queryLength > 1 else {
            return nil
        }
        return DeepSeekPooledMaskCache.mask(
            pooledLength: pooledCount,
            queryLength: queryLength,
            offset: offset,
            ratio: ratio
        )
    }

    func arraysForMaterialization() -> [MLXArray] {
        [bufferedKV, bufferedGate, pooledView].compactMap { $0 }
    }

    private static func paddedCapacity(for length: Int) -> Int {
        let target = length + capacityQuantum
        return ((target + capacityQuantum - 1) / capacityQuantum) * capacityQuantum
    }
}

public final class DeepSeekLayerCache {
    public let local: DeepSeekLocalKVCache
    public let pooled: DeepSeekPoolingCache?
    public let indexPooled: DeepSeekPoolingCache?

    public init(local: DeepSeekLocalKVCache, pooled: DeepSeekPoolingCache?, indexPooled: DeepSeekPoolingCache?) {
        self.local = local
        self.pooled = pooled
        self.indexPooled = indexPooled
    }

    func arraysForMaterialization() -> [MLXArray] {
        local.arraysForMaterialization()
            + (pooled?.arraysForMaterialization() ?? [])
            + (indexPooled?.arraysForMaterialization() ?? [])
    }
}

public final class DeepSeekModelCache {
    public let layers: [DeepSeekLayerCache]

    public init(config: DeepSeekConfig) {
        self.layers = config.compressRatios.map { ratio in
            DeepSeekLayerCache(
                local: DeepSeekLocalKVCache(maxSize: config.slidingWindow),
                pooled: ratio == 0 ? nil : DeepSeekPoolingCache(ratio: ratio),
                indexPooled: ratio == 4 ? DeepSeekPoolingCache(ratio: ratio) : nil
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
