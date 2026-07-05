import Foundation
import MLX
import MLXFastCore

/// Lock-guarded dictionary usable as a process-wide memo from Swift 6 code.
/// MLXArray and DeepSeekRoPE are not Sendable; all model execution happens on
/// one thread, and the lock makes the bookkeeping itself race-free, so the
/// unchecked conformance is sound for how the runtime uses these caches.
final class LockedCache<Key: Hashable, Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [Key: Value] = [:]
    private let capacity: Int

    init(capacity: Int = .max) {
        self.capacity = capacity
    }

    func value(for key: Key, make: () throws -> Value) rethrows -> Value {
        lock.lock()
        if let value = storage[key] {
            lock.unlock()
            return value
        }
        lock.unlock()

        let value = try make()

        lock.lock()
        if storage.count >= capacity {
            storage.removeAll(keepingCapacity: true)
        }
        storage[key] = value
        lock.unlock()
        return value
    }
}

/// Process-wide cache for causal attention masks.
///
/// A causal mask is a pure function of (queryLength, keyLength,
/// queryOffset - keyOffset, windowSize): the allowed predicate compares
/// query and key positions, so shifting both offsets by the same delta
/// produces the same boolean pattern. During decode every layer rebuilds an
/// identical mask each step; caching by the offset delta lets one mask per
/// shape class serve all layers and all steps.
public enum DeepSeekMaskCache {
    private struct Key: Hashable {
        let queryLength: Int
        let keyLength: Int
        let offsetDelta: Int
        let windowSize: Int
    }

    private static let cache = LockedCache<Key, MLXArray>(capacity: 128)

    public static func causal(
        queryLength: Int,
        keyLength: Int? = nil,
        queryOffset: Int = 0,
        keyOffset: Int = 0,
        windowSize: Int? = nil
    ) throws -> MLXArray {
        let resolvedKeyLength = keyLength ?? queryLength
        let key = Key(
            queryLength: queryLength,
            keyLength: resolvedKeyLength,
            offsetDelta: queryOffset - keyOffset,
            windowSize: windowSize ?? -1
        )
        return try cache.value(for: key) {
            try DeepSeekAttentionMask.causal(
                queryLength: queryLength,
                keyLength: resolvedKeyLength,
                queryOffset: queryOffset,
                keyOffset: keyOffset,
                windowSize: windowSize
            )
        }
    }
}

/// Process-wide cache for pooled (compressed-KV) causal masks.
///
/// `DeepSeekPoolingCache.makeMask` output is a pure function of
/// (pooledLength, queryLength, offset, ratio): it compares two fixed integer
/// ranges, `arange(pooledLength) < arange(offset + 1, offset + queryLength +
/// 1) / ratio`, so identical parameters always produce identical boolean
/// values. Every layer of a given ratio class asks for the same mask at a
/// given forward, so one entry serves all of them.
enum DeepSeekPooledMaskCache {
    private struct Key: Hashable {
        let pooledLength: Int
        let queryLength: Int
        let offset: Int
        let ratio: Int
    }

    private static let cache = LockedCache<Key, MLXArray>(capacity: 128)

    static func mask(pooledLength: Int, queryLength: Int, offset: Int, ratio: Int) -> MLXArray {
        let key = Key(
            pooledLength: pooledLength,
            queryLength: queryLength,
            offset: offset,
            ratio: ratio
        )
        return cache.value(for: key) {
            let poolIndex = arange(pooledLength, dtype: .int32)
            let queryIndex = arange(offset + 1, offset + queryLength + 1, dtype: .int32)
                .expandedDimensions(axis: 1)
            return poolIndex .< queryIndex.floorDivide(Int32(ratio))
        }
    }
}

/// Process-wide cache for the extended (local ++ pooled) attention masks that
/// `DeepSeekCompressedAttention` feeds to scaled-dot-product attention.
///
/// The extended mask is `concatenated([causalMask, pooledBlock], axis: -1)`.
/// Its causal part is a pure function of (queryLength, keyLength,
/// queryOffset - keyOffset, windowSize) -- the shift-invariance argument on
/// `DeepSeekMaskCache` -- and the pooled block is either all-ones (decode:
/// `makeMask` returns nil for queryLength == 1) or the broadcast pooled
/// causal mask, a pure function of (pooledLength, queryLength, absolute
/// offset, ratio). The key carries every one of those value-determining
/// parameters. Callers only consult this cache when both inputs came from
/// `DeepSeekMaskCache.causal` and `DeepSeekPoolingCache.makeMask` (always
/// `.bool`), so dtype is fixed by construction. During decode every
/// compressed layer previously rebuilt an identical ones+concat chain each
/// step; this memoizes one mask per shape class for all layers and steps.
enum DeepSeekExtendedMaskCache {
    struct Key: Hashable {
        let queryLength: Int
        let keyLength: Int
        let offsetDelta: Int
        let windowSize: Int
        let pooledLength: Int
        /// Absolute query offset, present only when a pooled causal mask
        /// participates (queryLength > 1); nil when the pooled block is
        /// all-ones and the absolute offset therefore cannot affect values.
        let pooledMaskOffset: Int?
        /// Pooling ratio backing `pooledMaskOffset`; 0 when the pooled block
        /// is all-ones.
        let poolRatio: Int
    }

    private static let cache = LockedCache<Key, MLXArray>(capacity: 256)

    static func mask(for key: Key, make: () throws -> MLXArray) rethrows -> MLXArray {
        try cache.value(for: key, make: make)
    }
}

/// Process-wide cache of DeepSeekRoPE instances keyed by their construction
/// parameters. The hot paths previously constructed a fresh instance per layer
/// per forward, recomputing base frequencies host-side and re-uploading the
/// frequency array to the GPU each time. Instances are immutable apart from an
/// internal lock-guarded frequency-array memo, so sharing instances is safe
/// for the single-threaded model runtime and for parallel unit tests.
public enum DeepSeekRoPECache {
    private struct Key: Hashable {
        let rotaryDimensions: Int
        let base: Double
        let freqScale: Int
        let maxPositionEmbeddings: Int
        // Every stored field of DeepSeekRopeScaling, spelled out so key
        // identity is guaranteed rather than a reflection detail.
        let scalingType: String?
        let scalingRopeType: String?
        let scalingFactor: Double?
        let scalingOriginalMaxPositionEmbeddings: Int?
        let scalingBetaFast: Double?
        let scalingBetaSlow: Double?
    }

    private static let cache = LockedCache<Key, DeepSeekRoPE>()

    public static func shared(
        rotaryDimensions: Int,
        base: Double,
        scaling: DeepSeekRopeScaling?,
        maxPositionEmbeddings: Int,
        freqScale: Int = 1
    ) throws -> DeepSeekRoPE {
        let key = Key(
            rotaryDimensions: rotaryDimensions,
            base: base,
            freqScale: freqScale,
            maxPositionEmbeddings: maxPositionEmbeddings,
            scalingType: scaling?.type,
            scalingRopeType: scaling?.ropeType,
            scalingFactor: scaling?.factor,
            scalingOriginalMaxPositionEmbeddings: scaling?.originalMaxPositionEmbeddings,
            scalingBetaFast: scaling?.betaFast,
            scalingBetaSlow: scaling?.betaSlow
        )
        return try cache.value(for: key) {
            try DeepSeekRoPE(
                rotaryDimensions: rotaryDimensions,
                base: base,
                scaling: scaling,
                maxPositionEmbeddings: maxPositionEmbeddings,
                freqScale: freqScale
            )
        }
    }
}
