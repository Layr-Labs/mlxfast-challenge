import Foundation
import MLX

/// Eagerly-prepared, RAM-resident weight cache for the Gemma 4 text tower.
///
/// The whole 4-bit checkpoint (~17 GB) is loaded once at construction time
/// (outside every scored window -- the runtime worker builds this before the
/// benchmark protocol handshake), so every scored forward pays no dense
/// loads or derived-view construction. There is no expert streaming or
/// residency machinery: the entire model is dense and lives in unified
/// memory for the whole process lifetime.
public final class Gemma4RuntimeWeightCache {
    public let loader: Gemma4WeightLoader
    public let config: Gemma4Config

    private var cachedModelWeights: Gemma4ModelWeights?
    private var cachedBlockWeights: [Int: Gemma4BlockWeights] = [:]
    private var cachedAttentionWeights: [Int: Gemma4AttentionWeights] = [:]
    private var cachedMLPWeights: [Int: Gemma4MLPWeights] = [:]
    private var cachedAttentionSpecs: [Int: Gemma4AttentionSpec] = [:]

    public init(loader: Gemma4WeightLoader, config: Gemma4Config) {
        self.loader = loader
        self.config = config
        eagerlyPrepareForFullModel()
    }

    public func attentionSpec(layerIndex: Int) -> Gemma4AttentionSpec {
        if let spec = cachedAttentionSpecs[layerIndex] {
            return spec
        }
        let spec = Gemma4AttentionSpec(layerIndex: layerIndex, config: config)
        cachedAttentionSpecs[layerIndex] = spec
        return spec
    }

    public func modelWeights() throws -> Gemma4ModelWeights {
        if let cachedModelWeights {
            return cachedModelWeights
        }
        let weights = try loader.modelWeights(config: config)
        cachedModelWeights = weights
        return weights
    }

    public func blockWeights(layerIndex: Int) throws -> Gemma4BlockWeights {
        if let weights = cachedBlockWeights[layerIndex] {
            return weights
        }
        let weights = try loader.blockWeights(layerIndex: layerIndex, config: config)
        cachedBlockWeights[layerIndex] = weights
        return weights
    }

    public func attentionWeights(layerIndex: Int) throws -> Gemma4AttentionWeights {
        if let weights = cachedAttentionWeights[layerIndex] {
            return weights
        }
        let weights = try loader.attentionWeights(layerIndex: layerIndex, config: config)
        cachedAttentionWeights[layerIndex] = weights
        return weights
    }

    public func mlpWeights(layerIndex: Int) throws -> Gemma4MLPWeights {
        if let weights = cachedMLPWeights[layerIndex] {
            return weights
        }
        let weights = try loader.mlpWeights(layerIndex: layerIndex, config: config)
        cachedMLPWeights[layerIndex] = weights
        return weights
    }

    /// For the full-size checkpoint, populate every memoized weight struct
    /// and warm the hot Metal kernels during construction. Small fixture
    /// configs (unit tests, convenience callers) skip this entirely, and
    /// every step is fail-soft so missing tensors surface on first use
    /// exactly as before.
    private func eagerlyPrepareForFullModel() {
        guard config.numHiddenLayers >= 16 else {
            return
        }
        // The default MLX buffer cache is effectively unbounded; bound it so
        // resident memory stays close to the ~17 GB checkpoint plus KV cache
        // and activation buffers instead of growing without limit across a
        // long decode run.
        Memory.cacheLimit = 6 << 30
        _ = try? modelWeights()
        for layerIndex in 0..<config.numHiddenLayers {
            _ = try? blockWeights(layerIndex: layerIndex)
            _ = try? attentionWeights(layerIndex: layerIndex)
            _ = try? mlpWeights(layerIndex: layerIndex)
            _ = attentionSpec(layerIndex: layerIndex)
        }
        Gemma4Warmup.run(weightCache: self)
    }
}
