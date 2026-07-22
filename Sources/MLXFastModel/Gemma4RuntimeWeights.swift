import Darwin
import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXNN

/// Eagerly-prepared, RAM-resident weight cache retained for the unranked
/// experimental MTP commands.
///
/// The whole 4-bit checkpoint (~17 GB) is loaded once at construction time
/// before the worker protocol handshake, so each forward pays no dense
/// loads or derived-view construction. There is no expert streaming or
/// residency machinery: the entire model is dense and lives in unified
/// memory for the whole process lifetime.
public final class Gemma4RuntimeWeightCache {
    public let loader: Gemma4WeightLoader
    public let config: Gemma4Config

    /// The mlx-swift-lm Gemma 4 text tower used by the legacy MTP path. It is
    /// loaded once here at construction, so no checkpoint I/O or
    /// quantized-linear construction lands on the request path.
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
        let startupMemoryPolicy: RuntimeStartupMemoryPolicy?
        if config.numHiddenLayers >= 16 {
            // Keep the 128 GiB machine's full optimized layout, but
            // protect local Macs from retaining ~14.5 GiB of alternate weight
            // layouts on top of the ~16.9 GiB source model. The policy is
            // selected before model loading so file-global feature switches are
            // first observed after their low-memory defaults are installed;
            // the defaults never overwrite flags the user set explicitly.
            let policy = RuntimeStartupMemoryPolicy.resolve(
                physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
                requestedProfile: ProcessInfo.processInfo.environment[
                    RuntimeStartupMemoryPolicy.profileOverrideEnvironmentName
                ]
            )
            policy.apply()
            startupMemoryPolicy = policy
        } else {
            startupMemoryPolicy = nil
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
        // Constructor-time warmup for the library model: the runtime worker
        // builds this cache before the protocol handshake, so Metal pipeline-state
        // creation and MLX kernel-cache population triggered by the first
        // forward happen here instead of inside the first request.
        //
        // Retain freed, shape-relevant warmup buffers for the worker request.
        // Metal libraries and pipeline state are process-lifetime caches
        // independent of this allocator pool.
        if let model = libraryModel, config.numHiddenLayers >= 16 {
            Self.warmLibraryModel(model)
            if startupMemoryPolicy?.clearAllocatorCacheAfterWarmup == true {
                // Pipeline state is process-lifetime state, while free warmup
                // allocations are exactly the pressure a low-memory machine
                // cannot afford to retain before the worker protocol hello.
                Memory.clearCache()
            }
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
    /// (sanitize -> 4-bit affine quantize -> update -> eval), but streams each
    /// tensor from its shard into MLX-owned storage while stripping the
    /// checkpoint's `language_model.` text-tower prefix: our transform
    /// preserves the source names (`language_model.model.layers...`), whereas `Gemma4TextModel`'s
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

        let validatedTiedHeadPacked13Metadata =
            try validateGemma4TiedHeadPacked13MetadataBytes(denseStore: denseStore)
        let loadedWeights = try loadRuntimeWeightArrays(denseStore: denseStore)
        let partition = try partitionRuntimeWeights(
            loadedWeights,
            expectedIndexedStems: expectedIndexedProjectionStems(config: config),
            validatedTiedHeadPacked13Metadata: validatedTiedHeadPacked13Metadata
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
        // The constructor's decode warmup consumes the packed head before the
        // worker handshake and materializes these arrays. Avoid a standalone
        // GPU command before FastEngine finishes its host-side preparation.
        try model.prepareFastEngine(
            indexedMetadata: partition.indexedMetadata,
            tiedHeadPacked13Metadata: partition.tiedHeadPacked13Metadata
        )
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

// The model-agnostic tensor loading/inventory helpers formerly here
// (`loadRuntimeWeightArrays`, `loadRuntimeWeightValues`,
// `validateRuntimeTensorInventory`, `validateRuntimeShardInventory`,
// `RuntimeWeightNameTracker`) now live in `RuntimeWeightLoading.swift`,
// shared with the Laguna loader. Everything below is Gemma 4 specific.

struct RuntimeWeightPartition {
    let modelParameters: [String: MLXArray]
    let indexedMetadata: [String: IndexedAffineMetadata]
    let tiedHeadPacked13Metadata: Gemma4TiedHeadPacked13Metadata?
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
    expectedIndexedStems: Set<String>,
    validatedTiedHeadPacked13Metadata:
        ValidatedGemma4TiedHeadPacked13Metadata? = nil
) throws -> RuntimeWeightPartition {
    let tiedHeadPackedIndicesName =
        "model.embed_tokens.tied_head_packed13_indices"
    let tiedHeadLUTName = "model.embed_tokens.tied_head_packed13_lut"
    let indicesSuffix = ".metadata_indices"
    let lutSuffix = ".metadata_lut"
    var modelParameters: [String: MLXArray] = [:]
    var metadataParts: [String: RuntimeIndexedMetadataParts] = [:]
    var tiedHeadPackedIndices: MLXArray?
    var tiedHeadLUT: MLXArray?

    for name in loaded.keys.sorted() {
        guard let array = loaded[name] else { continue }
        if name == tiedHeadPackedIndicesName {
            tiedHeadPackedIndices = array
        } else if name == tiedHeadLUTName {
            tiedHeadLUT = array
        } else if name.hasSuffix(indicesSuffix) {
            let stem = String(name.dropLast(indicesSuffix.count))
            metadataParts[stem, default: RuntimeIndexedMetadataParts()].indices = array
        } else if name.hasSuffix(lutSuffix) {
            let stem = String(name.dropLast(lutSuffix.count))
            metadataParts[stem, default: RuntimeIndexedMetadataParts()].lut = array
        } else {
            modelParameters[name] = array
        }
    }

    let tiedHeadPacked13Metadata: Gemma4TiedHeadPacked13Metadata?
    switch (tiedHeadPackedIndices, tiedHeadLUT) {
    case (nil, nil):
        tiedHeadPacked13Metadata = nil
    case let (packedIndices?, lut?):
        guard let weight = modelParameters["model.embed_tokens.weight"],
              let scales = modelParameters["model.embed_tokens.scales"],
              let biases = modelParameters["model.embed_tokens.biases"]
        else {
            throw MLXFastError.invalidInput(
                "tied-head packed13 metadata is missing stock affine tensors"
            )
        }
        let metadata = Gemma4TiedHeadPacked13Metadata(
            packedIndices: packedIndices,
            lut: lut
        )
        try validateGemma4TiedHeadPacked13MetadataLayout(
            metadata,
            weight: weight,
            scales: scales,
            biases: biases
        )
        guard let validatedTiedHeadPacked13Metadata,
              validatedTiedHeadPacked13Metadata.lutCount == lut.size
        else {
            throw MLXFastError.invalidInput(
                "tied-head packed13 metadata was not validated from raw bytes"
            )
        }
        tiedHeadPacked13Metadata = metadata
    default:
        throw MLXFastError.invalidInput(
            "tied-head packed13 metadata payload is incomplete"
        )
    }

    guard !metadataParts.isEmpty else {
        return RuntimeWeightPartition(
            modelParameters: modelParameters,
            indexedMetadata: [:],
            tiedHeadPacked13Metadata: tiedHeadPacked13Metadata
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
        indexedMetadata: indexedMetadata,
        tiedHeadPacked13Metadata: tiedHeadPacked13Metadata
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
