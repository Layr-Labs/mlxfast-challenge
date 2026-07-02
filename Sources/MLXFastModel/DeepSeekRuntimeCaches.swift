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

/// Process-wide cache of DeepSeekRoPE instances keyed by their construction
/// parameters. The hot paths previously constructed a fresh instance per layer
/// per forward, recomputing base frequencies host-side and re-uploading the
/// frequency array to the GPU each time. Instances are immutable apart from an
/// internal frequency-array memo, and the model executes single-threaded, so
/// sharing instances is safe.
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
