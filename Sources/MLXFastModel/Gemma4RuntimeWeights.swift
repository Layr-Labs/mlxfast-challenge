import Darwin
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
    public let libraryModel: Gemma4RuntimeModel?
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
        //
        // The ranked M5 Max has enough headroom to retain more freed
        // intermediate buffers for reuse. This is a soft allocator-cache cap,
        // not a reservation; model weights remain active allocations.
        if config.numHiddenLayers >= 16 {
            // The MLX M5 Max default commits after referencing 50 MiB. Many
            // 4-bit projections individually exceed that, so use a moderate
            // limit that can group adjacent kernels without long command buffers.
            setenv("MLX_MAX_MB_PER_BUFFER", "128", 1)
            Memory.cacheLimit = 32 << 30
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
        //
        // Retain freed, shape-relevant warmup buffers for the scored worker
        // request. Metal libraries and pipeline state are process-lifetime
        // caches independent of this allocator pool.
        if let model = libraryModel, config.numHiddenLayers >= 16 {
            Self.warmLibraryModel(model)
        }
    }

    /// One prefill-shaped forward (512 tokens) and one single-token decode
    /// step against a throwaway cache, evaluated and discarded. Inputs are
    /// constant BOS tokens, so this is prompt-independent and cannot affect
    /// model output; freed warmup buffers remain eligible for allocator reuse.
    private static func warmLibraryModel(_ model: Gemma4RuntimeModel) {
        let bosToken = Int32(2)
        let warmupCache = model.newCache(parameters: nil)
        let prefillTokens = MLXArray(
            Array(repeating: bosToken, count: 512),
            [1, 512]
        )
        eval(model(prefillTokens, cache: warmupCache))
        let decodeToken = MLXArray([bosToken], [1, 1])
        eval(model(decodeToken, cache: warmupCache))
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
    ) throws -> Gemma4RuntimeModel {
        let weightsPath = denseStore.weightsPath
        let directory = URL(fileURLWithPath: weightsPath)
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let textConfig = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: configData)
        let model = Gemma4RuntimeModel(textConfig)

        let loadedWeights = try loadRuntimeWeightArrays(denseStore: denseStore)
        let partition = try partitionRuntimeWeights(
            loadedWeights,
            expectedIndexedStems: expectedIndexedProjectionStems(config: config)
        )
        let weights = partition.modelParameters

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
        eval(partition.indexedMetadata.values.flatMap { [$0.indices, $0.lut] })
        try model.prepareFastEngine(indexedMetadata: partition.indexedMetadata)
        return model
    }

    public func requireLibraryModel() throws -> Gemma4RuntimeModel {
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

func loadRuntimeWeightArrays(
    denseStore: DenseTensorStore,
    restrictingTo requestedShards: Set<String>? = nil
) throws -> [String: MLXArray] {
    let directory = URL(fileURLWithPath: denseStore.weightsPath)
    let entries = try FileManager.default.contentsOfDirectory(
        at: directory, includingPropertiesForKeys: nil
    )
    let discoveredShards = entries
        .filter { $0.pathExtension == "safetensors" }
        .map(\.lastPathComponent)
    let allShardNames = try validateRuntimeShardInventory(
        referencedShards: denseStore.shardNames,
        discoveredShards: discoveredShards
    )

    let shardNames: [String]
    if let requestedShards {
        let unknown = requestedShards.subtracting(allShardNames).sorted()
        guard unknown.isEmpty else {
            throw MLXFastError.invalidInput(
                "requested runtime shards are not indexed: \(unknown.joined(separator: ", "))"
            )
        }
        guard !requestedShards.isEmpty else {
            throw MLXFastError.invalidInput("requested runtime shard set is empty")
        }
        shardNames = allShardNames.filter(requestedShards.contains)
    } else {
        shardNames = allShardNames
    }

    var loadedWeights: [String: MLXArray] = [:]
    var expectedLoadedNames: Set<String> = []
    var nameTracker = RuntimeWeightNameTracker()
    for shardName in shardNames {
        let expectedNames = denseStore.tensorNames(inShard: shardName)
        expectedLoadedNames.formUnion(expectedNames)
        let shard = directory.appendingPathComponent(shardName)
        for (key, value) in try loadArrays(url: shard) {
            let renamed = try nameTracker.register(
                originalName: key,
                shardName: shardName,
                expectedNames: expectedNames
            )
            loadedWeights[renamed] = value
        }
    }
    try nameTracker.validateComplete(expectedNames: expectedLoadedNames)
    return loadedWeights
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

struct RuntimeWeightPartition {
    let modelParameters: [String: MLXArray]
    let indexedMetadata: [String: IndexedAffineMetadata]
}

private struct RuntimeIndexedMetadataParts {
    var indices: MLXArray?
    var lut: MLXArray?
}

func expectedIndexedProjectionStems(config: Gemma4Config) -> Set<String> {
    Set((0..<config.numHiddenLayers).flatMap { layerIndex in
        var stems = [
            "model.layers.\(layerIndex).mlp.gate_proj",
            "model.layers.\(layerIndex).mlp.up_proj",
            "model.layers.\(layerIndex).mlp.down_proj",
            "model.layers.\(layerIndex).self_attn.q_proj",
            "model.layers.\(layerIndex).self_attn.k_proj",
        ]
        if !config.usesKEqV(for: config.layerTypes[layerIndex]) {
            stems.append("model.layers.\(layerIndex).self_attn.v_proj")
        }
        return stems
    })
}

func partitionRuntimeWeights(
    _ loaded: [String: MLXArray],
    expectedIndexedStems: Set<String>
) throws -> RuntimeWeightPartition {
    let indicesSuffix = ".metadata_indices"
    let lutSuffix = ".metadata_lut"
    var modelParameters: [String: MLXArray] = [:]
    var metadataParts: [String: RuntimeIndexedMetadataParts] = [:]

    for name in loaded.keys.sorted() {
        guard let array = loaded[name] else { continue }
        if name.hasSuffix(indicesSuffix) {
            let stem = String(name.dropLast(indicesSuffix.count))
            metadataParts[stem, default: RuntimeIndexedMetadataParts()].indices = array
        } else if name.hasSuffix(lutSuffix) {
            let stem = String(name.dropLast(lutSuffix.count))
            metadataParts[stem, default: RuntimeIndexedMetadataParts()].lut = array
        } else {
            modelParameters[name] = array
        }
    }

    guard !metadataParts.isEmpty else {
        return RuntimeWeightPartition(
            modelParameters: modelParameters,
            indexedMetadata: [:]
        )
    }
    let actualStems = Set(metadataParts.keys)
    let unknown = actualStems.subtracting(expectedIndexedStems).sorted()
    guard unknown.isEmpty else {
        throw MLXFastError.invalidInput(
            "indexed metadata contains unknown projection stems: \(unknown.joined(separator: ", "))"
        )
    }
    let missing = expectedIndexedStems.subtracting(actualStems).sorted()
    guard missing.isEmpty else {
        throw MLXFastError.invalidInput(
            "indexed metadata is missing projection stems: \(missing.joined(separator: ", "))"
        )
    }

    var indexedMetadata: [String: IndexedAffineMetadata] = [:]
    indexedMetadata.reserveCapacity(metadataParts.count)
    for stem in expectedIndexedStems.sorted() {
        guard let parts = metadataParts[stem],
              let indices = parts.indices,
              let lut = parts.lut,
              let weight = modelParameters["\(stem).weight"],
              let scales = modelParameters["\(stem).scales"],
              let biases = modelParameters["\(stem).biases"]
        else {
            throw MLXFastError.invalidInput(
                "indexed metadata or stock affine tensors are incomplete for \(stem)"
            )
        }
        guard weight.dtype == .uint32,
              scales.dtype == .bfloat16,
              biases.dtype == .bfloat16,
              scales.shape == biases.shape,
              scales.size > 0,
              indices.dtype == .uint16,
              indices.shape == scales.shape,
              lut.dtype == .uint32,
              lut.ndim == 1,
              (1...65_536).contains(lut.size)
        else {
            throw MLXFastError.invalidInput(
                "indexed metadata has invalid dtype or shape for \(stem)"
            )
        }
        try validateIndexedAffineMetadata(
            indices: indices,
            lut: lut,
            scales: scales,
            biases: biases,
            stem: stem
        )
        indexedMetadata[stem] = IndexedAffineMetadata(indices: indices, lut: lut)
    }
    return RuntimeWeightPartition(
        modelParameters: modelParameters,
        indexedMetadata: indexedMetadata
    )
}

private func validateIndexedAffineMetadata(
    indices: MLXArray,
    lut: MLXArray,
    scales: MLXArray,
    biases: MLXArray,
    stem: String
) throws {
    let maximumIndex = indices.max().item(UInt16.self)
    guard Int(maximumIndex) < lut.size else {
        throw MLXFastError.invalidInput(
            "indexed metadata index exceeds LUT bounds for \(stem)"
        )
    }

    let pairs = take(lut, indices.asType(.int32))
    let reconstructedScales = bitwiseAnd(
        pairs, MLXArray(UInt32(0xffff))
    ).asType(.uint16)
    let reconstructedBiases = rightShift(
        pairs, MLXArray(UInt32(16))
    ).asType(.uint16)
    let scalesMatch = arrayEqual(reconstructedScales, scales.view(dtype: .uint16))
    let biasesMatch = arrayEqual(reconstructedBiases, biases.view(dtype: .uint16))
    eval(scalesMatch, biasesMatch)
    guard scalesMatch.item(Bool.self), biasesMatch.item(Bool.self) else {
        throw MLXFastError.invalidInput(
            "indexed metadata does not reconstruct stock affine payloads for \(stem)"
        )
    }
}
