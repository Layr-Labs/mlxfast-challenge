import Foundation
import MLX
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
