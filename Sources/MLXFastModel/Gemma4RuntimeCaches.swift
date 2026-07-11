import Foundation
import MLX
import MLXFastCore

/// Lock-guarded dictionary usable as a process-wide memo from Swift 6 code.
/// MLXArray and Gemma4RoPE are not Sendable; all model execution happens on
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
        defer { lock.unlock() }
        if let value = storage[key] {
            return value
        }
        let value = try make()
        if storage.count >= capacity {
            storage.removeAll(keepingCapacity: true)
        }
        storage[key] = value
        return value
    }
}

public enum Gemma4AttentionMask {
    public static func causal(
        queryLength: Int,
        keyLength: Int? = nil,
        queryOffset: Int = 0,
        keyOffset: Int = 0,
        windowSize: Int? = nil
    ) throws -> MLXArray {
        guard queryLength > 0 else {
            throw MLXFastError.invalidInput("causal mask query length must be positive")
        }
        let keyLength = keyLength ?? queryLength
        guard keyLength > 0 else {
            throw MLXFastError.invalidInput("causal mask key length must be positive")
        }
        if let windowSize {
            guard windowSize > 0 else {
                throw MLXFastError.invalidInput("causal mask window size must be positive")
            }
        }

        let queryPositions = arange(
            queryOffset,
            queryOffset + queryLength,
            dtype: .int32
        ).expandedDimensions(axis: 1)
        let keyPositions = arange(
            keyOffset,
            keyOffset + keyLength,
            dtype: .int32
        ).expandedDimensions(axis: 0)

        var allowed = queryPositions .>= keyPositions
        if let windowSize {
            allowed = allowed .&& (queryPositions .< (keyPositions + windowSize))
        }
        return allowed
    }
}

/// Process-wide cache for causal attention masks; see `Gemma4AttentionMask`.
/// A causal mask is a pure function of (queryLength, keyLength,
/// queryOffset - keyOffset, windowSize), so one mask per shape class serves
/// every layer of that type and every decode step.
public enum Gemma4MaskCache {
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
            try Gemma4AttentionMask.causal(
                queryLength: queryLength,
                keyLength: resolvedKeyLength,
                queryOffset: queryOffset,
                keyOffset: keyOffset,
                windowSize: windowSize
            )
        }
    }
}
