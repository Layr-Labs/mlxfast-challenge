// CompiledDecode: whole-step compiled decode helper.
// See notes/MLXLMCommon-CompiledDecode.notes.md#line26

import Foundation
import MLX
import os

private let compiledDecodeLog = Logger(subsystem: "darkbloom", category: "CompiledDecode")
private let tieredCompiledAttentionEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_COMPILED_TIERED_ATTENTION"] != "0"

public enum CompiledDecode {

    /// Host-side selector between two compiled graphs over shared cache state.
    private final class TieredForward: @unchecked Sendable {
        let fastAttentionLength: Int
        var logicalOffset: Int
        let fastForward: @Sendable ([MLXArray]) -> [MLXArray]
        let fullForward: @Sendable ([MLXArray]) -> [MLXArray]
        let lock = NSLock()

        init(
            model: any LanguageModel,
            fastCache: [KVCache],
            fullCache: [KVCache],
            fastAttentionLength: Int,
            logicalOffset: Int
        ) {
            self.fastAttentionLength = fastAttentionLength
            self.logicalOffset = logicalOffset
            self.fastForward = CompiledDecode.compileForward(
                model: model, cacheRef: fastCache)
            self.fullForward = CompiledDecode.compileForward(
                model: model, cacheRef: fullCache)
        }

        func callAsFunction(_ args: [MLXArray]) -> [MLXArray] {
            lock.withLock {
                let tokenCount = args.first?.dim(1) ?? 0
                guard tokenCount > 0 else {
                    return logicalOffset < fastAttentionLength
                        ? fastForward(args) : fullForward(args)
                }

                let (needed, overflow) = logicalOffset.addingReportingOverflow(tokenCount)
                precondition(
                    !overflow && needed <= Int(Int32.max),
                    "Compiled decode cache offset exceeds its Int32 graph representation")
                logicalOffset = needed
                return needed <= fastAttentionLength ? fastForward(args) : fullForward(args)
            }
        }
    }

    /// Compiled decode is ON by default. Set `DARKBLOOM_COMPILED_DECODE=0`
    /// to disable. Guards in GenerationBatch ensure it only activates for
    /// B=1 solo decode with supported cache types (no MTP, no SSM).
    public static let isEnabled: Bool = {
        if let raw = ProcessInfo.processInfo.environment["DARKBLOOM_COMPILED_DECODE"] {
            return !["0", "false", "no", "off"].contains(raw.lowercased())
        }
        return true
    }()

    /// True iff every layer is a compilable cache type and thus
    /// compile-traceable by ``compileForward(model:cacheRef:)``.
    public static func eligible(_ cache: [KVCache]) -> Bool {
        !cache.isEmpty && cache.allSatisfy {
            $0 is CompilableKVCache || $0 is CompilableRotatingKVCache
                || $0 is CompilableBatchKVCache || $0 is CompilableBatchRotatingKVCache
        }
    }

    /// Build a compiled forward closure for a decode step.
    /// See notes/MLXLMCommon-CompiledDecode.notes.md#compileforward
    public static func compileForward(
        model: any LanguageModel,
        cacheRef: [KVCache]
    ) -> @Sendable ([MLXArray]) -> [MLXArray] {
        precondition(
            eligible(cacheRef),
            "CompiledDecode.compileForward requires a non-empty cache where every "
                + "layer is a compilable type (single-stream or batched).")

        let capturedModel = model
        let captured = cacheRef

        return compile(
            inputs: captured, outputs: captured
        ) { (args: [MLXArray]) -> [MLXArray] in
            let result = capturedModel(
                LMInput.Text(tokens: args[0]),
                cache: captured.isEmpty ? nil : captured,
                state: nil
            )
            return [result.logits]
        }
    }

    /// Attempt to set up compiled decode for a model + cache pair.
    ///
    /// This converts eligible cache layers to their compilable equivalents
    /// and builds a compiled forward closure. Per-layer promotion handles
    /// heterogeneous caches (e.g. Gemma4 with mixed KVCacheSimple +
    /// RotatingKVCache layers).
    ///
    /// The conversion is only performed when ALL of these conditions hold:
    /// - `DARKBLOOM_COMPILED_DECODE=1` env var is set
    /// - `MLXHardwareInfo.isCompiledDecodeSupported` is true
    /// - Every cache layer is either `KVCacheSimple` or `RotatingKVCache`
    ///
    public static func setupCompiledDecode(
        model: any LanguageModel,
        cache: inout [KVCache],
        maxCacheLength: Int = 4096
    ) -> (@Sendable ([MLXArray]) -> [MLXArray])? {
        guard isEnabled else { return nil }
        guard MLXHardwareInfo.isCompiledDecodeSupported else {
            compiledDecodeLog.info("Compiled decode skipped: hardware not supported")
            return nil
        }

        // Validate all layers are promotable before doing any conversion.
        for layer in cache {
            if !(layer is KVCacheSimple) && !(layer is RotatingKVCache) {
                compiledDecodeLog.info(
                    "Compiled decode skipped: unsupported cache type (\(type(of: layer)))")
                return nil
            }
        }

        // Materialize all pending cache operations before conversion.
        eval(cache)

        let currentOffset = cache.map(\.offset).max() ?? 0
        let growthStep =
            cache.compactMap { ($0 as? KVCacheSimple)?.step }.max() ?? 256
        guard currentOffset >= 0,
            let fastAttentionLength = initialAttentionLength(
                currentOffset: currentOffset,
                growthStep: growthStep,
                callerUpperBound: maxCacheLength)
        else {
            compiledDecodeLog.info("Compiled decode skipped: invalid cache capacity")
            return nil
        }

        // Per-layer promotion: each layer type gets its compilable equivalent.
        var simpleCount = 0
        var rotatingCount = 0
        for i in 0..<cache.count {
            if let rotating = cache[i] as? RotatingKVCache {
                // RotatingKVCache → CompilableRotatingKVCache
                // (uses its own maxCacheSize, not maxCacheLength)
                cache[i] = CompilableRotatingKVCache.promote(from: rotating, maxLength: maxCacheLength)
                rotatingCount += 1
            } else if let simple = cache[i] as? KVCacheSimple {
                // KVCacheSimple → CompilableKVCache
                cache[i] = CompilableKVCache.promote(
                    from: simple, maxLength: maxCacheLength)
                simpleCount += 1
            }
        }

        // Materialize the new compilable cache buffers
        eval(cache)

        let layerCount = cache.count
        compiledDecodeLog.info(
            "Compiled decode enabled: \(layerCount) layers (\(simpleCount) simple + \(rotatingCount) rotating), fastAttentionLength=\(fastAttentionLength), backingLength=\(maxCacheLength)")

        guard tieredCompiledAttentionEnabled,
            simpleCount > 0,
            fastAttentionLength < maxCacheLength
        else {
            return compileForward(model: model, cacheRef: cache)
        }
        let fastCache: [KVCache] = cache.map {
            if let full = $0 as? CompilableKVCache {
                return full.sharingStorage(attentionLength: fastAttentionLength)
            }
            return $0
        }
        let owner = TieredForward(
            model: model,
            fastCache: fastCache,
            fullCache: cache,
            fastAttentionLength: fastAttentionLength,
            logicalOffset: currentOffset)
        return { owner($0) }
    }

    /// Choose the shortest attention view that preserves the long-cache
    /// vector-SDPA reduction partition.
    /// See notes/MLXLMCommon-CompiledDecode.notes.md#initialattentionlength
    static func initialAttentionLength(
        currentOffset: Int,
        growthStep: Int,
        callerUpperBound: Int
    ) -> Int? {
        guard currentOffset >= 0, growthStep > 0, callerUpperBound >= currentOffset,
            currentOffset <= Int(Int32.max)
        else { return nil }
        let (withHeadroom, overflow) = currentOffset.addingReportingOverflow(growthStep)
        guard !overflow else { return nil }
        let needed = max(withHeadroom, 1025)
        let capacity =
            needed == 1025
            ? needed
            : roundedCapacity(atLeast: needed, step: growthStep)
        return min(callerUpperBound, capacity)
    }

    private static func roundedCapacity(atLeast needed: Int, step: Int) -> Int {
        precondition(needed >= 0 && step > 0)
        let quotient = needed / step
        let remainder = needed % step
        let roundedQuotient = quotient + (remainder == 0 ? 0 : 1)
        let (capacity, overflow) = roundedQuotient.multipliedReportingOverflow(by: step)
        precondition(
            !overflow && capacity <= Int(Int32.max),
            "Compiled decode cache capacity exceeds its Int32 graph representation")
        return capacity
    }

    /// Set up compiled decode for batched caches (B >= 1).
    /// See notes/MLXLMCommon-CompiledDecode.notes.md#setupbatchcompileddecode
    public static func setupBatchCompiledDecode(
        model: any LanguageModel,
        cache: inout [any BatchedCache],
        maxCacheLength: Int = 4096
    ) -> (@Sendable ([MLXArray]) -> [MLXArray])? {
        guard isEnabled else { return nil }
        guard MLXHardwareInfo.isCompiledDecodeSupported else {
            compiledDecodeLog.info("Batch compiled decode skipped: hardware not supported")
            return nil
        }

        // Validate all layers are promotable.
        for layer in cache {
            if layer is CompilableBatchKVCache || layer is CompilableBatchRotatingKVCache {
                continue  // Already compilable
            }
            if !(layer is BatchKVCache) && !(layer is BatchRotatingKVCache) {
                compiledDecodeLog.info(
                    "Batch compiled decode skipped: unsupported cache type (\(type(of: layer)))")
                return nil
            }
        }

        // Materialize all pending cache operations before conversion.
        eval(cache)

        // Per-layer promotion.
        var fullCount = 0
        var rotatingCount = 0
        for i in 0..<cache.count {
            if cache[i] is CompilableBatchKVCache || cache[i] is CompilableBatchRotatingKVCache {
                // Already compilable — count it.
                if cache[i] is CompilableBatchKVCache { fullCount += 1 }
                else { rotatingCount += 1 }
                continue
            }

            if let rotating = cache[i] as? BatchRotatingKVCache {
                cache[i] = CompilableBatchRotatingKVCache.promote(
                    from: rotating, maxLength: maxCacheLength)
                rotatingCount += 1
            } else if let full = cache[i] as? BatchKVCache {
                cache[i] = CompilableBatchKVCache.promote(
                    from: full, maxLength: maxCacheLength)
                fullCount += 1
            }
        }

        // Materialize the new compilable cache buffers.
        eval(cache)

        let layerCount = cache.count
        compiledDecodeLog.info(
            "Batch compiled decode enabled: \(layerCount) layers (\(fullCount) full + \(rotatingCount) rotating), maxLength=\(maxCacheLength)")

        // Build compiled forward with the caches cast to [KVCache].
        let cacheRef = cache.map { $0 as any KVCache }
        return compileForward(model: model, cacheRef: cacheRef)
    }
}
