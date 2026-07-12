import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon

public struct Gemma4ModelWeights {
    public let embedTokens: Gemma4LinearWeight
    public let finalNorm: MLXArray
    /// Explicit LM head, present only when the checkpoint does not tie word
    /// embeddings. The Gemma 4 31B checkpoint ties embeddings
    /// (`tie_word_embeddings: true`) and ships no separate `lm_head` tensor,
    /// so this is nil in the shipped configuration and `embedTokens` doubles
    /// as the head (matching mlx-vlm's `embed_tokens.as_linear(hidden)`).
    public let lmHead: Gemma4LinearWeight?

    public init(embedTokens: Gemma4LinearWeight, finalNorm: MLXArray, lmHead: Gemma4LinearWeight?) {
        self.embedTokens = embedTokens
        self.finalNorm = finalNorm
        self.lmHead = lmHead
    }
}

public struct Gemma4ModelSpec: Equatable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let rmsNormEps: Double
    public let finalLogitSoftcapping: Double

    public init(
        vocabSize: Int,
        hiddenSize: Int,
        numHiddenLayers: Int,
        rmsNormEps: Double,
        finalLogitSoftcapping: Double
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.rmsNormEps = rmsNormEps
        self.finalLogitSoftcapping = finalLogitSoftcapping
    }

    public init(config: Gemma4Config) {
        self.init(
            vocabSize: config.vocabSize,
            hiddenSize: config.hiddenSize,
            numHiddenLayers: config.numHiddenLayers,
            rmsNormEps: config.rmsNormEps,
            finalLogitSoftcapping: config.finalLogitSoftcapping
        )
    }
}

/// Top-level Gemma 4 (text tower) forward pass: scaled embedding lookup,
/// decoder layers, final RMSNorm, tied-embedding LM head, and final logit
/// softcapping (`30 * tanh(x / 30)`).
public enum Gemma4Model {
    /// Reference forward, delegated to the mlx-swift-lm Gemma 4 text tower held
    /// by `weightCache`. Returns `[1, length, vocab]` logits with the final
    /// `30 * tanh(x / 30)` softcap already applied by the library; downstream
    /// consumers read only the last position's row.
    ///
    /// The library derives each layer's RoPE position from its KV cache offset.
    /// `positionOffset` is checked against those offsets so a stale or reused
    /// harness cache fails before silently producing logits at the wrong position.
    public static func logits(
        inputIDs: MLXArray,
        weightCache: Gemma4RuntimeWeightCache,
        cache: Gemma4ModelCache? = nil,
        positionOffset: Int = 0
    ) throws -> MLXArray {
        guard positionOffset >= 0 else {
            throw MLXFastError.invalidInput("Gemma 4 position offset must be non-negative")
        }
        let model = try weightCache.requireLibraryModel()

        guard let cache else {
            let kvCaches = model.newCache(parameters: nil)
            try verifyCachePosition(
                positionOffset: positionOffset,
                cacheOffsets: kvCaches.map(\.offset)
            )
            return model(inputIDs, cache: kvCaches)
        }

        let inputLength = inputIDs.dim(1)
        let compiledDecodeStep = cache.compiledDecodeStep
        let validatesLibraryCacheOffsets: Bool
        if compiledDecodeStep == nil {
            validatesLibraryCacheOffsets = true
        } else {
            validatesLibraryCacheOffsets = false
        }
        return try executeGemma4CachedForward(
            cache: cache,
            positionOffset: positionOffset,
            inputLength: inputLength,
            validatesLibraryCacheOffsets: validatesLibraryCacheOffsets,
            validateLibraryCacheOffsets: {
                let kvCaches = cache.kvCache(for: model)
                try verifyCachePosition(
                    positionOffset: positionOffset,
                    cacheOffsets: kvCaches.map(\.offset)
                )
            },
            forward: {
                var kvCaches = cache.kvCache(for: model)
                // Single-token decode step: optionally route through mlx-swift-lm's
                // compiled decode (fused, no per-step graph rebuild). The compiled
                // closure is built once, on the first step, from the seed-populated
                // caches (promoted in place to compilable types); prefill and the seed
                // forward (length > 1) stay eager.
                //
                // Compiled decode is OPT-IN (DARKBLOOM_COMPILED_DECODE=1): its one-time
                // compile cost lands inside the scored decode window, which is not
                // amortized at the ranked 128-step length. Gating here in the harness
                // adapter -- rather than relying on the fork's internal default --
                // makes ranked runs deterministically eager wherever the env is unset.
                if inputLength == 1 {
                    if let step = compiledDecodeStep {
                        return step([inputIDs])[0]
                    }
                    if ProcessInfo.processInfo.environment["DARKBLOOM_COMPILED_DECODE"] == "1" {
                        cache.convertCombinedCachesToUpstream(&kvCaches)
                        if let step = CompiledDecode.setupCompiledDecode(
                            model: model,
                            cache: &kvCaches
                        ) {
                            cache.adoptKVCaches(kvCaches)
                            cache.compiledDecodeStep = step
                            return step([inputIDs])[0]
                        }
                    }
                    // Compiled decode disabled or unsupported: eager per-step forward.
                    cache.adoptKVCaches(kvCaches)
                    return model(inputIDs, cache: kvCaches)
                }
                return model(inputIDs, cache: kvCaches)
            }
        )
    }

    public static func logits(
        inputIDs: MLXArray,
        weights: Gemma4ModelWeights,
        spec: Gemma4ModelSpec,
        positionOffset: Int = 0,
        layer: (_ layerIndex: Int, _ hidden: MLXArray) throws -> MLXArray
    ) throws -> MLXArray {
        let hidden = try finalHidden(
            inputIDs: inputIDs,
            weights: weights,
            spec: spec,
            positionOffset: positionOffset,
            layer: layer
        )
        let logits = Gemma4Ops.linear(hidden, weights.lmHead ?? weights.embedTokens)
        return Gemma4Ops.logitSoftcap(logits, cap: spec.finalLogitSoftcapping)
    }

    public static func finalHidden(
        inputIDs: MLXArray,
        weights: Gemma4ModelWeights,
        spec: Gemma4ModelSpec,
        positionOffset: Int = 0,
        layer: (_ layerIndex: Int, _ hidden: MLXArray) throws -> MLXArray
    ) throws -> MLXArray {
        var hidden = try initialHidden(inputIDs: inputIDs, embedding: weights.embedTokens, spec: spec)

        for layerIndex in 0..<spec.numHiddenLayers {
            hidden = try layer(layerIndex, hidden)
        }

        // Every consumer of multi-token logits (greedy next-token selection,
        // top-logit traces, the benchmark oracle validators) reads only the
        // LAST position's row; slicing before the final norm/head removes a
        // [length-1, vocab]-sized slab of dead work from every scored forward.
        if hidden.shape[1] > 1 {
            hidden = hidden[0..., (hidden.shape[1] - 1)..., 0...]
        }

        return Gemma4Ops.rmsNorm(hidden, weight: weights.finalNorm, eps: spec.rmsNormEps)
    }

    public static func initialHidden(
        inputIDs: MLXArray,
        embedding: Gemma4LinearWeight,
        spec: Gemma4ModelSpec
    ) throws -> MLXArray {
        try validateInputIDs(inputIDs, spec: spec)
        guard embedding.logicalShape == [spec.vocabSize, spec.hiddenSize] else {
            throw MLXFastError.invalidInput(
                "embedding shape \(embedding.logicalShape) expected [\(spec.vocabSize), \(spec.hiddenSize)]"
            )
        }

        let embedded = Gemma4Ops.embedding(inputIDs: inputIDs, weight: embedding)
        let scale = Float(Double(spec.hiddenSize).squareRoot())
        return embedded * scale
    }

    private static func validateInputIDs(_ inputIDs: MLXArray, spec: Gemma4ModelSpec) throws {
        guard inputIDs.shape.count == 2 else {
            throw MLXFastError.invalidInput("Gemma 4 input IDs must have shape [batch, length]")
        }
        guard inputIDs.shape[0] > 0, inputIDs.shape[1] > 0 else {
            throw MLXFastError.invalidInput("Gemma 4 input IDs must have non-empty batch and length")
        }
        guard spec.vocabSize > 0, spec.hiddenSize > 0 else {
            throw MLXFastError.invalidInput("Gemma 4 model spec dimensions must be positive")
        }
    }
}

@inline(__always)
func executeGemma4CachedForward<Result>(
    cache: Gemma4ModelCache,
    positionOffset: Int,
    inputLength: Int,
    validatesLibraryCacheOffsets: Bool,
    validateLibraryCacheOffsets: () throws -> Void,
    forward: () throws -> Result
) throws -> Result {
    let nextExpectedPositionOffset = try cache.nextExpectedPositionOffset(
        positionOffset: positionOffset,
        inputLength: inputLength
    )
    if validatesLibraryCacheOffsets {
        try validateLibraryCacheOffsets()
    }
    let result = try forward()
    cache.commitExpectedPositionOffset(nextExpectedPositionOffset)
    return result
}

func verifyCachePosition(positionOffset: Int, cacheOffsets: [Int]) throws {
    guard positionOffset >= 0 else {
        throw MLXFastError.invalidInput("Gemma 4 position offset must be non-negative")
    }
    guard let cacheOffset = cacheOffsets.first else {
        throw MLXFastError.invalidInput("Gemma 4 model returned no KV caches")
    }
    guard cacheOffsets.allSatisfy({ $0 == cacheOffset }) else {
        throw MLXFastError.invalidInput("Gemma 4 KV cache layer offsets are inconsistent")
    }
    guard positionOffset == cacheOffset else {
        throw MLXFastError.invalidInput(
            "Gemma 4 position offset \(positionOffset) does not match KV cache offset \(cacheOffset)"
        )
    }
}
