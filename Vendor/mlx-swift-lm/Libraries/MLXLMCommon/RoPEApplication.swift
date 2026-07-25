// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation
import MLX
import MLXNN

// MARK: - BatchPositionedKVCache

/// Protocol for KV caches that expose per-sequence RoPE offsets.
///
/// This is a forward-compatible hook for batched caches. Current scalar-cache
/// code paths continue using `KVCache.offset`.
public protocol BatchPositionedKVCache: KVCache {
    /// Per-sequence RoPE offsets with shape `[B]`.
    var batchOffset: MLXArray { get }
}

// MARK: - graphOffsetArray Helper

/// Retain the current graph context without adding an MLX primitive. This is
/// the cross-module equivalent of MLXArray's internal `copyContext()`: later
/// `_updateInternal` calls replace the source wrapper's context while this
/// wrapper continues to reference the pre-update tracer.
private func graphContextSnapshot(_ array: MLXArray) -> MLXArray {
    var snapshot = mlx_array_new()
    mlx_array_set(&snapshot, array.ctx)
    return MLXArray(snapshot)
}

/// Returns a graph-visible cache offset when the cache exposes one.
///
/// `KVCache.offset` is an `Int` API for compatibility with upstream model
/// ports. Compile-safe caches keep the offset as an `MLXArray` so the value
/// can flow through `compile()` without an `.item()` readback. Model-side
/// helpers that need positions outside `applyRotaryPosition` should call this
/// before falling back to `cache.offset`.
public func graphOffsetArray(for cache: KVCache?) -> MLXArray? {
    // Snapshot the context so cache.update() advancing offsetArray doesn't
    // shift the caller's RoPE position. A direct alias would put the query
    // one step ahead of the keys in compiled decode. The former `+ 0`
    // snapshot preserved this ordering but also created an elementwise GPU
    // primitive on every call.
    if let compilableRot = cache as? CompilableRotatingKVCache {
        return graphContextSnapshot(compilableRot.offsetArray)
    }
    if let compilable = cache as? CompilableKVCache {
        return graphContextSnapshot(compilable.offsetArray)
    }
    if let batchCache = cache as? BatchPositionedKVCache {
        return graphContextSnapshot(batchCache.batchOffset)
    }
    return nil
}

// MARK: - applyRotaryPosition Helper

/// Apply rotary position embeddings, using the cache offset when available.
///
/// This function enables models to use a single call site instead of
/// repeating conditional offset handling:
/// ```swift
/// queries = applyRotaryPosition(rope, to: queries, cache: cache)
/// keys = applyRotaryPosition(rope, to: keys, cache: cache)
/// ```
///
/// When the cache exposes an `offsetArray` (e.g. `CompilableKVCache`), the
/// offset is passed as an `MLXArray` so the compile tracer can track it
/// through the graph without triggering a synchronous GPU readback. For all
/// other cache types, the standard `Int`-based offset path is used.
///
/// - Parameters:
///   - rope: A RoPE layer conforming to both `OffsetLayer` and `ArrayOffsetLayer`.
///   - x: The input tensor to apply RoPE to.
///   - cache: The KV cache (determines scalar or per-sequence offset), or `nil`
///     for offset 0.
/// - Returns: The input with rotary positional encoding applied.
public func applyRotaryPosition<R: RoPELayer>(_ rope: R, to x: MLXArray, cache: KVCache?)
    -> MLXArray
{
    if let offsetArray = graphOffsetArray(for: cache) {
        return rope(x, offset: offsetArray)
    }
    return rope(x, offset: cache?.offset ?? 0)
}
