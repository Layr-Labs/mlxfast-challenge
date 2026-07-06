import Foundation
import MLX
import MLXFastCore

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
    public static func logits(
        inputIDs: MLXArray,
        weightCache: Gemma4RuntimeWeightCache,
        cache: Gemma4ModelCache? = nil,
        positionOffset: Int = 0
    ) throws -> MLXArray {
        let config = weightCache.config
        let spec = Gemma4ModelSpec(config: config)
        let weights = try weightCache.modelWeights()
        return try logits(
            inputIDs: inputIDs,
            weights: weights,
            spec: spec,
            positionOffset: positionOffset
        ) { layerIndex, hidden in
            try layer(
                index: layerIndex,
                hidden: hidden,
                weightCache: weightCache,
                cache: cache?.layers[layerIndex],
                positionOffset: positionOffset
            )
        }
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

    public static func layer(
        index layerIndex: Int,
        hidden: MLXArray,
        weightCache: Gemma4RuntimeWeightCache,
        cache: Gemma4LayerCache? = nil,
        positionOffset: Int = 0
    ) throws -> MLXArray {
        let config = weightCache.config
        let blockWeights = try weightCache.blockWeights(layerIndex: layerIndex)
        let attentionWeights = try weightCache.attentionWeights(layerIndex: layerIndex)
        let mlpWeights = try weightCache.mlpWeights(layerIndex: layerIndex)
        let attentionSpec = weightCache.attentionSpec(layerIndex: layerIndex)
        let layerType = config.layerTypes[layerIndex]

        // Both attention paths rebuild their mask from the KV cache's key
        // offset whenever a cache is present, so this mask is consumed only
        // on the cache-free path (used by tests and single-shot forwards).
        let mask = cache == nil
            ? try Gemma4MaskCache.causal(
                queryLength: hidden.shape[1],
                keyLength: hidden.shape[1],
                queryOffset: positionOffset,
                keyOffset: positionOffset,
                windowSize: layerType == .sliding ? config.slidingWindow : nil
            )
            : nil

        return try Gemma4Block.forward(
            hidden: hidden,
            weights: blockWeights,
            rmsNormEps: config.rmsNormEps,
            attention: { normalized in
                try Gemma4Attention.forward(
                    normalized,
                    weights: attentionWeights,
                    spec: attentionSpec,
                    mask: mask,
                    cache: cache,
                    positionOffset: positionOffset
                )
            },
            feedForward: { normalized in
                Gemma4MLP.forward(normalized, weights: mlpWeights)
            }
        )
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
