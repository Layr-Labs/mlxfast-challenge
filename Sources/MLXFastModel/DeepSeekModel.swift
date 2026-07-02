import Foundation
import MLX
import MLXFastCore

public struct DeepSeekHeadHyperConnectionWeights {
    public let fn: MLXArray
    public let base: MLXArray
    public let scale: MLXArray

    public init(fn: MLXArray, base: MLXArray, scale: MLXArray) {
        self.fn = fn
        self.base = base
        self.scale = scale
    }
}

public struct DeepSeekModelWeights {
    public let embedTokens: DeepSeekLinearWeight
    public let finalNorm: MLXArray
    public let headHyperConnection: DeepSeekHeadHyperConnectionWeights
    public let lmHead: DeepSeekLinearWeight

    public init(
        embedTokens: DeepSeekLinearWeight,
        finalNorm: MLXArray,
        headHyperConnection: DeepSeekHeadHyperConnectionWeights,
        lmHead: DeepSeekLinearWeight
    ) {
        self.embedTokens = embedTokens
        self.finalNorm = finalNorm
        self.headHyperConnection = headHyperConnection
        self.lmHead = lmHead
    }

    public init(
        embedTokens: MLXArray,
        finalNorm: MLXArray,
        headHyperConnection: DeepSeekHeadHyperConnectionWeights,
        lmHead: MLXArray
    ) {
        self.init(
            embedTokens: DeepSeekLinearWeight(embedTokens),
            finalNorm: finalNorm,
            headHyperConnection: headHyperConnection,
            lmHead: DeepSeekLinearWeight(lmHead)
        )
    }
}

public struct DeepSeekModelSpec: Equatable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let compressRatios: [Int]
    public let slidingWindow: Int
    public let hcMult: Int
    public let hcSinkhornIters: Int
    public let hcEps: Double
    public let rmsNormEps: Double

    public init(
        vocabSize: Int,
        hiddenSize: Int,
        numHiddenLayers: Int,
        compressRatios: [Int],
        slidingWindow: Int,
        hcMult: Int,
        hcSinkhornIters: Int,
        hcEps: Double,
        rmsNormEps: Double
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.compressRatios = compressRatios
        self.slidingWindow = slidingWindow
        self.hcMult = hcMult
        self.hcSinkhornIters = hcSinkhornIters
        self.hcEps = hcEps
        self.rmsNormEps = rmsNormEps
    }

    public init(config: DeepSeekConfig) {
        self.init(
            vocabSize: config.vocabSize,
            hiddenSize: config.hiddenSize,
            numHiddenLayers: config.numHiddenLayers,
            compressRatios: config.compressRatios,
            slidingWindow: config.slidingWindow,
            hcMult: config.hcMult,
            hcSinkhornIters: config.hcSinkhornIters,
            hcEps: config.hcEps,
            rmsNormEps: config.rmsNormEps
        )
    }
}

public enum DeepSeekModel {
    public static func logits(
        inputIDs: MLXArray,
        loader: DeepSeekWeightLoader,
        config: DeepSeekConfig,
        cache: DeepSeekModelCache? = nil,
        positionOffset: Int = 0
    ) throws -> MLXArray {
        try logits(
            inputIDs: inputIDs,
            weightCache: DeepSeekRuntimeWeightCache(loader: loader, config: config),
            cache: cache,
            positionOffset: positionOffset
        )
    }

    public static func logits(
        inputIDs: MLXArray,
        weightCache: DeepSeekRuntimeWeightCache,
        cache: DeepSeekModelCache? = nil,
        positionOffset: Int = 0
    ) throws -> MLXArray {
        let config = weightCache.config
        let spec = DeepSeekModelSpec(config: config)
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
                inputIDs: inputIDs,
                weightCache: weightCache,
                cache: cache?.layers[layerIndex],
                positionOffset: positionOffset
            )
        }
    }

    public static func logits(
        inputIDs: MLXArray,
        weights: DeepSeekModelWeights,
        spec: DeepSeekModelSpec,
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
        return DeepSeekOps.linear(input: hidden, weight: weights.lmHead)
    }

    public static func finalHidden(
        inputIDs: MLXArray,
        weights: DeepSeekModelWeights,
        spec: DeepSeekModelSpec,
        positionOffset: Int = 0,
        layer: (_ layerIndex: Int, _ hidden: MLXArray) throws -> MLXArray
    ) throws -> MLXArray {
        var hidden = try initialHidden(
            inputIDs: inputIDs,
            embedding: weights.embedTokens,
            spec: spec
        )

        for layerIndex in 0..<spec.numHiddenLayers {
            guard layerIndex < spec.compressRatios.count else {
                throw MLXFastError.invalidInput(
                    "missing compress ratio for DeepSeek layer \(layerIndex)"
                )
            }
            hidden = try layer(layerIndex, hidden)
        }

        let collapsed = try DeepSeekHyperConnection.head(
            hidden,
            fn: weights.headHyperConnection.fn,
            base: weights.headHyperConnection.base,
            scale: weights.headHyperConnection.scale,
            hcMult: spec.hcMult,
            eps: spec.hcEps,
            normEps: spec.rmsNormEps
        )
        return DeepSeekOps.rmsNorm(
            input: collapsed,
            weight: weights.finalNorm,
            eps: spec.rmsNormEps
        )
    }

    public static func initialHidden(
        inputIDs: MLXArray,
        embedding: DeepSeekLinearWeight,
        spec: DeepSeekModelSpec
    ) throws -> MLXArray {
        try validateInputIDs(inputIDs, spec: spec)
        guard embedding.logicalShape == [spec.vocabSize, spec.hiddenSize] else {
            throw MLXFastError.invalidInput(
                "embedding shape \(embedding.logicalShape) expected [\(spec.vocabSize), \(spec.hiddenSize)]"
            )
        }

        let embedded = DeepSeekOps.embedding(inputIDs: inputIDs, weight: embedding)
        return broadcast(
            embedded.expandedDimensions(axis: 2),
            to: [inputIDs.shape[0], inputIDs.shape[1], spec.hcMult, spec.hiddenSize]
        )
    }

    public static func initialHidden(
        inputIDs: MLXArray,
        embedding: MLXArray,
        spec: DeepSeekModelSpec
    ) throws -> MLXArray {
        try initialHidden(
            inputIDs: inputIDs,
            embedding: DeepSeekLinearWeight(embedding),
            spec: spec
        )
    }

    public static func layer(
        index layerIndex: Int,
        hidden: MLXArray,
        inputIDs: MLXArray,
        loader: DeepSeekWeightLoader,
        config: DeepSeekConfig,
        cache: DeepSeekLayerCache? = nil,
        positionOffset: Int = 0
    ) throws -> MLXArray {
        try layer(
            index: layerIndex,
            hidden: hidden,
            inputIDs: inputIDs,
            weightCache: DeepSeekRuntimeWeightCache(loader: loader, config: config),
            cache: cache,
            positionOffset: positionOffset
        )
    }

    public static func layer(
        index layerIndex: Int,
        hidden: MLXArray,
        inputIDs: MLXArray,
        weightCache: DeepSeekRuntimeWeightCache,
        cache: DeepSeekLayerCache? = nil,
        positionOffset: Int = 0
    ) throws -> MLXArray {
        let config = weightCache.config
        let compressRatio = config.compressRatios[layerIndex]
        let blockWeights = try weightCache.blockWeights(layerIndex: layerIndex)
        let moeWeights = try weightCache.moeWeights(layerIndex: layerIndex)
        let blockSpec = DeepSeekBlockSpec(config: config)
        let moeSpec = try DeepSeekMoESpec(layerIndex: layerIndex, config: config)
        let mask = try DeepSeekAttentionMask.causal(
            queryLength: inputIDs.shape[1],
            keyLength: inputIDs.shape[1],
            queryOffset: positionOffset,
            keyOffset: positionOffset,
            windowSize: config.slidingWindow
        )

        return try DeepSeekBlock.forward(
            hidden: hidden,
            weights: blockWeights,
            spec: blockSpec,
            attention: { normalized in
                switch compressRatio {
                case 0:
                    return try DeepSeekLocalAttention.forward(
                        normalized,
                        weights: weightCache.localAttentionWeights(layerIndex: layerIndex),
                        spec: DeepSeekLocalAttentionSpec(config: config),
                        mask: mask,
                        cache: cache?.local,
                        windowSize: config.slidingWindow,
                        positionOffset: positionOffset
                    )
                case 4, 128:
                    return try DeepSeekCompressedAttention.forward(
                        normalized,
                        weights: weightCache.compressedAttentionWeights(layerIndex: layerIndex),
                        spec: DeepSeekCompressedAttentionSpec(config: config, layerIndex: layerIndex),
                        mask: mask,
                        cache: cache,
                        windowSize: config.slidingWindow,
                        positionOffset: positionOffset
                    )
                default:
                    throw MLXFastError.invalidInput(
                        "Swift DeepSeek attention ratio \(compressRatio) is unsupported for layer \(layerIndex)"
                    )
                }
            },
            feedForward: { normalized in
                try DeepSeekMoE.forward(
                    normalized,
                    inputIDs: inputIDs,
                    weights: moeWeights,
                    loader: weightCache.loader,
                    spec: moeSpec
                )
            }
        )
    }

    private static func validateInputIDs(_ inputIDs: MLXArray, spec: DeepSeekModelSpec) throws {
        guard inputIDs.shape.count == 2 else {
            throw MLXFastError.invalidInput("DeepSeek input IDs must have shape [batch, length]")
        }
        guard inputIDs.shape[0] > 0, inputIDs.shape[1] > 0 else {
            throw MLXFastError.invalidInput("DeepSeek input IDs must have non-empty batch and length")
        }
        guard spec.vocabSize > 0, spec.hiddenSize > 0, spec.hcMult > 0 else {
            throw MLXFastError.invalidInput("DeepSeek model spec dimensions must be positive")
        }
        guard spec.compressRatios.count == spec.numHiddenLayers else {
            throw MLXFastError.invalidInput(
                "compress ratio count \(spec.compressRatios.count) expected \(spec.numHiddenLayers)"
            )
        }
    }
}
