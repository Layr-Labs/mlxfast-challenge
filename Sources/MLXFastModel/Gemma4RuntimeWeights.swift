import Foundation
import MLX
import MLXLLM
import MLXLMCommon

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

    /// The mlx-swift-lm Gemma 4 text tower this benchmark's reference runs. It is
    /// loaded once here at construction (outside every scored window), so no
    /// checkpoint I/O or quantized-linear construction lands on the hot path.
    /// nil only if the load failed, in which case `loadError` carries the reason
    /// and the first `Gemma4Model.logits` rethrows it.
    public let libraryModel: Gemma4TextModel?
    public let loadError: Error?

    private var cachedModelWeights: Gemma4ModelWeights?
    private var cachedBlockWeights: [Int: Gemma4BlockWeights] = [:]
    private var cachedAttentionWeights: [Int: Gemma4AttentionWeights] = [:]
    private var cachedMLPWeights: [Int: Gemma4MLPWeights] = [:]
    private var cachedAttentionSpecs: [Int: Gemma4AttentionSpec] = [:]

    public init(loader: Gemma4WeightLoader, config: Gemma4Config) {
        self.loader = loader
        self.config = config
        // Bound the MLX buffer cache so resident memory stays near the ~17 GB
        // checkpoint plus KV/activation buffers instead of growing without limit
        // across a long decode run.
        if config.numHiddenLayers >= 16 {
            Memory.cacheLimit = 6 << 30
        }
        do {
            libraryModel = try Gemma4RuntimeWeightCache.loadLibraryModel(
                weightsPath: loader.denseStore.weightsPath,
                config: config
            )
            loadError = nil
        } catch {
            libraryModel = nil
            loadError = error
        }
    }

    /// Construct and weight-load the mlx-swift-lm Gemma 4 text tower from the
    /// transformed `weights/` tree: decode its flat `config.json`, build the
    /// module, and let the library's `loadWeights` sanitize + apply 4-bit affine
    /// quantization (group size / bits from our config) + update + eval.
    private static func loadLibraryModel(
        weightsPath: String,
        config: Gemma4Config
    ) throws -> Gemma4TextModel {
        let directory = URL(fileURLWithPath: weightsPath)
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let textConfig = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: configData)
        let model = Gemma4TextModel(textConfig)
        let quantization = BaseConfiguration.Quantization(
            groupSize: config.quantizationGroupSize,
            bits: config.quantizationBits
        )
        try loadWeights(modelDirectory: directory, model: model, quantization: quantization)
        return model
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

}
