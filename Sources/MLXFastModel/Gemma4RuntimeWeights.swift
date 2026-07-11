import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXNN

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
                denseStore: loader.denseStore,
                config: config
            )
            loadError = nil
        } catch {
            libraryModel = nil
            loadError = error
        }
        // Constructor-time warmup for the library model: the runtime worker builds this cache
        // before the benchmark protocol handshake, so the Metal pipeline-state
        // creation and MLX kernel-cache population triggered by the first
        // forward happen HERE, outside every scored window, instead of inside
        // the first scored prefill.
        if let model = libraryModel, config.numHiddenLayers >= 16 {
            Self.warmLibraryModel(model)
        }
    }

    /// One prefill-shaped forward (512 tokens) and one single-token decode
    /// step against a throwaway cache, evaluated and discarded. Inputs are
    /// constant BOS tokens, so this is prompt-independent and cannot affect
    /// model output; the buffer cache is cleared afterwards so warmup
    /// allocations do not linger into the measured run.
    private static func warmLibraryModel(_ model: Gemma4TextModel) {
        let bosToken = Int32(2)
        let warmupCache = model.newCache(parameters: nil)
        let prefillTokens = MLXArray(
            Array(repeating: bosToken, count: 512),
            [1, 512]
        )
        eval(model(prefillTokens, cache: warmupCache))
        let decodeToken = MLXArray([bosToken], [1, 1])
        eval(model(decodeToken, cache: warmupCache))
        Memory.clearCache()
    }

    /// Construct and weight-load the mlx-swift-lm Gemma 4 text tower from the
    /// transformed `weights/` tree. Mirrors the library's `loadWeights`
    /// (sanitize -> 4-bit affine quantize -> update -> eval), but reads the
    /// safetensor shards in memory first so we can strip the checkpoint's
    /// `language_model.` text-tower prefix: our transform preserves the source
    /// names (`language_model.model.layers...`), whereas `Gemma4TextModel`'s
    /// parameter paths are `model.layers...`. Doing the rename here keeps the
    /// transform output and the harness's DenseTensorStore validation unchanged.
    private static func loadLibraryModel(
        denseStore: DenseTensorStore,
        config: Gemma4Config
    ) throws -> Gemma4TextModel {
        let weightsPath = denseStore.weightsPath
        let directory = URL(fileURLWithPath: weightsPath)
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let textConfig = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: configData)
        let model = Gemma4TextModel(textConfig)

        var weights: [String: MLXArray] = [:]
        let entries = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )
        let discoveredShards = entries
            .filter { $0.pathExtension == "safetensors" }
            .map(\.lastPathComponent)
        let shardNames = try validateRuntimeShardInventory(
            referencedShards: denseStore.shardNames,
            discoveredShards: discoveredShards
        )
        var nameTracker = RuntimeWeightNameTracker()
        for shardName in shardNames {
            let shard = directory.appendingPathComponent(shardName)
            let expectedNames = denseStore.tensorNames(inShard: shardName)
            for (key, value) in try loadArrays(url: shard) {
                let renamed = try nameTracker.register(
                    originalName: key,
                    shardName: shardName,
                    expectedNames: expectedNames
                )
                weights[renamed] = value
            }
        }
        try nameTracker.validateComplete(expectedNames: Set(denseStore.tensorNames))

        let sanitized = model.sanitize(weights: weights)
        let quantization = BaseConfiguration.Quantization(
            groupSize: config.quantizationGroupSize,
            bits: config.quantizationBits
        )
        quantize(model: model) { path, _ in
            sanitized["\(path).scales"] != nil ? quantization.asTuple : nil
        }
        // The gemma-4-31b-4bit checkpoint already stores scales/biases/norms in
        // bf16 (only the 4-bit weight codes are U32), so the library's fp16->bf16
        // conversion pass is a no-op here and is intentionally omitted.
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        eval(model)
        return model
    }

    public func requireLibraryModel() throws -> Gemma4TextModel {
        guard let libraryModel else {
            throw loadError
                ?? MLXFastError.invalidInput("Gemma 4 reference model was not loaded")
        }
        return libraryModel
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

func validateRuntimeShardInventory(
    referencedShards: [String],
    discoveredShards: [String]
) throws -> [String] {
    let referenced = Set(referencedShards)
    let discovered = Set(discoveredShards)
    guard !referenced.isEmpty else {
        throw MLXFastError.missingFile("dense safetensors index references no shards")
    }
    let missing = referenced.subtracting(discovered).sorted()
    guard missing.isEmpty else {
        throw MLXFastError.missingFile(
            "dense safetensors index references missing shards: \(missing.joined(separator: ", "))"
        )
    }
    let unindexed = discovered.subtracting(referenced).sorted()
    guard unindexed.isEmpty else {
        throw MLXFastError.invalidInput(
            "weights directory contains unindexed safetensors shards: \(unindexed.joined(separator: ", "))"
        )
    }
    return referenced.sorted()
}

struct RuntimeWeightNameTracker {
    private static let languageModelPrefix = "language_model."
    private var originalNames: Set<String> = []
    private var runtimeNames: Set<String> = []

    mutating func register(
        originalName: String,
        shardName: String,
        expectedNames: Set<String>
    ) throws -> String {
        guard expectedNames.contains(originalName) else {
            throw MLXFastError.invalidInput(
                "safetensors shard \(shardName) contains unindexed or misplaced tensor \(originalName)"
            )
        }
        guard originalNames.insert(originalName).inserted else {
            throw MLXFastError.invalidInput("duplicate safetensors tensor \(originalName)")
        }
        let runtimeName = originalName.hasPrefix(Self.languageModelPrefix)
            ? String(originalName.dropFirst(Self.languageModelPrefix.count))
            : originalName
        guard runtimeNames.insert(runtimeName).inserted else {
            throw MLXFastError.invalidInput(
                "safetensors tensor names collide after runtime rename: \(runtimeName)"
            )
        }
        return runtimeName
    }

    func validateComplete(expectedNames: Set<String>) throws {
        let missing = expectedNames.subtracting(originalNames).sorted()
        guard missing.isEmpty else {
            throw MLXFastError.invalidInput(
                "indexed safetensors tensors were not loaded: \(missing.joined(separator: ", "))"
            )
        }
    }
}
