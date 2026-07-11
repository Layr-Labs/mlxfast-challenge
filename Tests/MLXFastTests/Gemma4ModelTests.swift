import Foundation
import MLX
@testable import MLXFastCore
@testable import MLXFastModel
import Testing

@Test
func gemma4ModelInitialHiddenScalesEmbeddingBySqrtHiddenSizeWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let spec = Gemma4ModelSpecFixture.make(vocabSize: 3, hiddenSize: 2)
    let inputIDs = MLXArray([Int32(2), Int32(0)], [1, 2])
    let embedding = Gemma4LinearWeight(MLXArray((1...6).map { Float($0) }, [3, 2]))

    let hidden = try Gemma4Model.initialHidden(inputIDs: inputIDs, embedding: embedding, spec: spec)

    let scale = Float(2.0.squareRoot())
    #expect(hidden.shape == [1, 2, 2])
    let values = hidden.asArray(Float.self)
    #expect(abs(values[0] - 5 * scale) < 1e-5)
    #expect(abs(values[1] - 6 * scale) < 1e-5)
    #expect(abs(values[2] - 1 * scale) < 1e-5)
    #expect(abs(values[3] - 2 * scale) < 1e-5)
}

@Test
func gemma4ModelTopLevelLogitsRunInjectedLayersAndSoftcapsWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let spec = Gemma4ModelSpecFixture.make(vocabSize: 3, hiddenSize: 2, numHiddenLayers: 1, finalLogitSoftcapping: 0)
    let weights = Gemma4ModelWeights(
        embedTokens: Gemma4LinearWeight(MLXArray([Float(1), 0, 0, 1, 1, 1], [3, 2])),
        finalNorm: ones([2], dtype: .float32),
        lmHead: Gemma4LinearWeight(MLXArray([Float(1), 0, 0, 1, 1, 1], [3, 2]))
    )
    let inputIDs = MLXArray([Int32(1)], [1, 1])
    var seenLayers: [Int] = []

    let logits = try Gemma4Model.logits(
        inputIDs: inputIDs,
        weights: weights,
        spec: spec
    ) { layerIndex, hidden in
        seenLayers.append(layerIndex)
        return hidden
    }

    #expect(seenLayers == [0])
    #expect(logits.shape == [1, 1, 3])
    let scale = Float(2.0.squareRoot())
    let values = logits.asArray(Float.self)
    #expect(abs(values[0] - 0) < 1e-4)
    #expect(abs(values[1] - scale) < 1e-4)
    #expect(abs(values[2] - scale) < 1e-4)
}

@Test
func gemma4ModelAppliesFinalLogitSoftcapWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let spec = Gemma4ModelSpecFixture.make(vocabSize: 1, hiddenSize: 1, numHiddenLayers: 0, finalLogitSoftcapping: 30.0)
    let weights = Gemma4ModelWeights(
        embedTokens: Gemma4LinearWeight(MLXArray([Float(60)], [1, 1])),
        finalNorm: ones([1], dtype: .float32),
        lmHead: Gemma4LinearWeight(MLXArray([Float(1)], [1, 1]))
    )
    let inputIDs = MLXArray([Int32(0)], [1, 1])

    let logits = try Gemma4Model.logits(
        inputIDs: inputIDs,
        weights: weights,
        spec: spec
    ) { _, hidden in hidden }

    let value = logits.asArray(Float.self)[0]
    #expect(value < 30.0)
    #expect(value > 0)
}

@Test
func gemma4ModelCachePositionAcceptsMatchingLayerOffsets() throws {
    try verifyCachePosition(positionOffset: 0, cacheOffsets: [0])
    try verifyCachePosition(positionOffset: 512, cacheOffsets: [512, 512, 512])
}

@Test
func gemma4ModelCachePositionRejectsMismatchInconsistencyNegativeAndEmptyOffsets() {
    #expect(throws: MLXFastError.self) {
        try verifyCachePosition(positionOffset: 7, cacheOffsets: [8, 8])
    }
    #expect(throws: MLXFastError.self) {
        try verifyCachePosition(positionOffset: 8, cacheOffsets: [8, 9])
    }
    #expect(throws: MLXFastError.self) {
        try verifyCachePosition(positionOffset: -1, cacheOffsets: [0])
    }
    #expect(throws: MLXFastError.self) {
        try verifyCachePosition(positionOffset: 0, cacheOffsets: [])
    }
}

@Test
func gemma4ModelHostCachePositionAdvancesAndRejectsInvalidTransitions() throws {
    #expect(try advancedCachePosition(
        positionOffset: 0,
        expectedPositionOffset: 0,
        inputLength: 512
    ) == 512)
    #expect(try advancedCachePosition(
        positionOffset: 512,
        expectedPositionOffset: 512,
        inputLength: 1
    ) == 513)

    #expect(throws: MLXFastError.self) {
        _ = try advancedCachePosition(
            positionOffset: 511,
            expectedPositionOffset: 512,
            inputLength: 1
        )
    }
    #expect(throws: MLXFastError.self) {
        _ = try advancedCachePosition(
            positionOffset: 0,
            expectedPositionOffset: 0,
            inputLength: 0
        )
    }
    #expect(throws: MLXFastError.self) {
        _ = try advancedCachePosition(
            positionOffset: Int.max,
            expectedPositionOffset: Int.max,
            inputLength: 1
        )
    }
}

@Test
func gemma4CachedForwardValidatesEagerPathAndCommitsAfterSuccess() throws {
    let cache = makeGemma4ModelCacheFixture()
    var events: [String] = []

    let prefill: Int = try executeGemma4CachedForward(
        cache: cache,
        positionOffset: 0,
        inputLength: 512,
        validatesLibraryCacheOffsets: true,
        validateLibraryCacheOffsets: { events.append("validate-prefill") },
        forward: {
            events.append("forward-prefill")
            return 41
        }
    )

    #expect(prefill == 41)
    #expect(events == ["validate-prefill", "forward-prefill"])
    #expect(cache.expectedPositionOffset == 512)

    let decode: Int = try executeGemma4CachedForward(
        cache: cache,
        positionOffset: 512,
        inputLength: 1,
        validatesLibraryCacheOffsets: false,
        validateLibraryCacheOffsets: { events.append("unexpected-compiled-validation") },
        forward: {
            events.append("forward-compiled")
            return 42
        }
    )

    #expect(decode == 42)
    #expect(events == ["validate-prefill", "forward-prefill", "forward-compiled"])
    #expect(cache.expectedPositionOffset == 513)
}

@Test
func gemma4CachedForwardCommitsOnlyAfterValidationAndForwardSucceed() {
    let cache = makeGemma4ModelCacheFixture()
    var forwardRan = false

    #expect(throws: Gemma4CachedForwardTestError.self) {
        let _: Int = try executeGemma4CachedForward(
            cache: cache,
            positionOffset: 0,
            inputLength: 8,
            validatesLibraryCacheOffsets: true,
            validateLibraryCacheOffsets: { throw Gemma4CachedForwardTestError.validation },
            forward: {
                forwardRan = true
                return 1
            }
        )
    }
    #expect(!forwardRan)
    #expect(cache.expectedPositionOffset == 0)

    #expect(throws: Gemma4CachedForwardTestError.self) {
        let _: Int = try executeGemma4CachedForward(
            cache: cache,
            positionOffset: 0,
            inputLength: 8,
            validatesLibraryCacheOffsets: true,
            validateLibraryCacheOffsets: {},
            forward: { throw Gemma4CachedForwardTestError.forward }
        )
    }
    #expect(cache.expectedPositionOffset == 0)

    #expect(throws: MLXFastError.self) {
        let _: Int = try executeGemma4CachedForward(
            cache: cache,
            positionOffset: 1,
            inputLength: 1,
            validatesLibraryCacheOffsets: false,
            validateLibraryCacheOffsets: {},
            forward: {
                forwardRan = true
                return 1
            }
        )
    }
    #expect(!forwardRan)
    #expect(cache.expectedPositionOffset == 0)
}

private enum Gemma4CachedForwardTestError: Error {
    case validation
    case forward
}

private func makeGemma4ModelCacheFixture() -> Gemma4ModelCache {
    Gemma4ModelCache(config: Gemma4Config(
        modelType: "gemma4_text",
        vocabSize: 1,
        hiddenSize: 2,
        intermediateSize: 2,
        numHiddenLayers: 0,
        numAttentionHeads: 1,
        numKeyValueHeads: 1,
        numGlobalKeyValueHeads: 1,
        headDim: 2,
        globalHeadDim: 2,
        rmsNormEps: 1e-6,
        hiddenActivation: "gelu_pytorch_tanh",
        maxPositionEmbeddings: 16,
        attentionBias: false,
        attentionDropout: 0,
        attentionKEqV: true,
        slidingWindow: 4,
        layerTypes: [],
        finalLogitSoftcapping: 30,
        tieWordEmbeddings: true,
        slidingRope: Gemma4RopeSpec(theta: 10_000, type: "default"),
        fullRope: Gemma4RopeSpec(theta: 1_000_000, type: "proportional", partialRotaryFactor: 1),
        quantizationGroupSize: 1,
        quantizationBits: 4
    ))
}

private enum Gemma4ModelSpecFixture {
    static func make(
        vocabSize: Int,
        hiddenSize: Int,
        numHiddenLayers: Int = 0,
        rmsNormEps: Double = 0,
        finalLogitSoftcapping: Double = 30.0
    ) -> Gemma4ModelSpec {
        Gemma4ModelSpec(
            vocabSize: vocabSize,
            hiddenSize: hiddenSize,
            numHiddenLayers: numHiddenLayers,
            rmsNormEps: rmsNormEps,
            finalLogitSoftcapping: finalLogitSoftcapping
        )
    }
}
