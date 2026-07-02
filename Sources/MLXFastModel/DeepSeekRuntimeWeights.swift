import Foundation

public final class DeepSeekRuntimeWeightCache {
    public let loader: DeepSeekWeightLoader
    public let config: DeepSeekConfig

    private var cachedModelWeights: DeepSeekModelWeights?
    private var cachedBlockWeights: [Int: DeepSeekBlockWeights] = [:]
    private var cachedLocalAttentionWeights: [Int: DeepSeekLocalAttentionWeights] = [:]
    private var cachedCompressedAttentionWeights: [Int: DeepSeekCompressedAttentionWeights] = [:]
    private var cachedMoEWeights: [Int: DeepSeekMoEWeights] = [:]

    public init(loader: DeepSeekWeightLoader, config: DeepSeekConfig) {
        self.loader = loader
        self.config = config
    }

    public func modelWeights() throws -> DeepSeekModelWeights {
        if let cachedModelWeights {
            return cachedModelWeights
        }
        let weights = try loader.modelWeights(config: config)
        cachedModelWeights = weights
        return weights
    }

    public func blockWeights(layerIndex: Int) throws -> DeepSeekBlockWeights {
        if let weights = cachedBlockWeights[layerIndex] {
            return weights
        }
        let weights = try loader.blockWeights(layerIndex: layerIndex, config: config)
        cachedBlockWeights[layerIndex] = weights
        return weights
    }

    public func localAttentionWeights(layerIndex: Int) throws -> DeepSeekLocalAttentionWeights {
        if let weights = cachedLocalAttentionWeights[layerIndex] {
            return weights
        }
        let weights = try loader.localAttentionWeights(layerIndex: layerIndex, config: config)
        cachedLocalAttentionWeights[layerIndex] = weights
        return weights
    }

    public func compressedAttentionWeights(layerIndex: Int) throws -> DeepSeekCompressedAttentionWeights {
        if let weights = cachedCompressedAttentionWeights[layerIndex] {
            return weights
        }
        let weights = try loader.compressedAttentionWeights(layerIndex: layerIndex, config: config)
        cachedCompressedAttentionWeights[layerIndex] = weights
        return weights
    }

    public func moeWeights(layerIndex: Int) throws -> DeepSeekMoEWeights {
        if let weights = cachedMoEWeights[layerIndex] {
            return weights
        }
        let weights = try loader.moeWeights(layerIndex: layerIndex, config: config)
        cachedMoEWeights[layerIndex] = weights
        return weights
    }
}
