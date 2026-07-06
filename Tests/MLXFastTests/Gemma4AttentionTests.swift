import Foundation
import MLX
@testable import MLXFastCore
@testable import MLXFastModel
import Testing

@Test
func gemma4AttentionRejectsHeadsNotDivisibleByKeyValueHeads() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let spec = Gemma4AttentionSpec(
        layerType: .sliding,
        numAttentionHeads: 3,
        numKeyValueHeads: 2,
        headDim: 4,
        useKEqV: false,
        ropeSpec: Gemma4RopeSpec(theta: 10_000, type: "default"),
        slidingWindow: 4,
        rmsNormEps: 0
    )
    let weights = Gemma4AttentionSpecFixture.identityWeights(spec: spec, hiddenSize: 8)

    #expect(throws: MLXFastError.self) {
        _ = try Gemma4Attention.forward(
            MLXArray.zeros([1, 1, 8]),
            weights: weights,
            spec: spec
        )
    }
}

@Test
func gemma4AttentionSlidingLayerProducesExpectedOutputShapeWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let spec = Gemma4AttentionSpec(
        layerType: .sliding,
        numAttentionHeads: 2,
        numKeyValueHeads: 1,
        headDim: 4,
        useKEqV: false,
        ropeSpec: Gemma4RopeSpec(theta: 10_000, type: "default"),
        slidingWindow: 4,
        rmsNormEps: 0
    )
    let hiddenSize = 8
    let weights = Gemma4AttentionSpecFixture.identityWeights(spec: spec, hiddenSize: hiddenSize)

    let x = MLXArray((0..<(3 * hiddenSize)).map { Float($0) / 10 }, [1, 3, hiddenSize])
    let output = try Gemma4Attention.forward(x, weights: weights, spec: spec)

    #expect(output.shape == [1, 3, hiddenSize])
}

@Test
func gemma4AttentionFullLayerWithSharedKVProjectionRunsWithoutVProjWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let spec = Gemma4AttentionSpec(
        layerType: .full,
        numAttentionHeads: 2,
        numKeyValueHeads: 1,
        headDim: 4,
        useKEqV: true,
        ropeSpec: Gemma4RopeSpec(theta: 1_000_000, type: "proportional", partialRotaryFactor: 0.5),
        slidingWindow: 4,
        rmsNormEps: 0
    )
    let hiddenSize = 8
    let weights = Gemma4AttentionSpecFixture.identityWeights(spec: spec, hiddenSize: hiddenSize)
    #expect(weights.vProj == nil)

    let x = MLXArray((0..<(2 * hiddenSize)).map { Float($0) / 10 }, [1, 2, hiddenSize])
    let output = try Gemma4Attention.forward(x, weights: weights, spec: spec)

    #expect(output.shape == [1, 2, hiddenSize])
}

@Test
func gemma4AttentionCachedDecodeAppliesSlidingWindowMaskWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let spec = Gemma4AttentionSpec(
        layerType: .sliding,
        numAttentionHeads: 1,
        numKeyValueHeads: 1,
        headDim: 4,
        useKEqV: false,
        ropeSpec: Gemma4RopeSpec(theta: 10_000, type: "default"),
        slidingWindow: 2,
        rmsNormEps: 0
    )
    let hiddenSize = 4
    let weights = Gemma4AttentionSpecFixture.identityWeights(spec: spec, hiddenSize: hiddenSize)
    let cache = Gemma4LayerCache(
        keys: Gemma4KVCache(maxSize: spec.slidingWindow),
        values: Gemma4KVCache(maxSize: spec.slidingWindow)
    )

    for step in 0..<5 {
        let x = MLXArray((0..<hiddenSize).map { Float($0 + step) }, [1, 1, hiddenSize])
        let output = try Gemma4Attention.forward(
            x,
            weights: weights,
            spec: spec,
            cache: cache,
            positionOffset: step
        )
        #expect(output.shape == [1, 1, hiddenSize])
    }

    // The sliding-window cache never retains more than slidingWindow positions.
    #expect(cache.keys.arraysForMaterialization().first?.shape[2] ?? 0 <= spec.slidingWindow)
}

private enum Gemma4AttentionSpecFixture {
    static func identityWeights(spec: Gemma4AttentionSpec, hiddenSize: Int) -> Gemma4AttentionWeights {
        let qOut = spec.numAttentionHeads * spec.headDim
        let kvOut = spec.numKeyValueHeads * spec.headDim
        return Gemma4AttentionWeights(
            qProj: Gemma4LinearWeight(MLXArray.zeros([qOut, hiddenSize])),
            kProj: Gemma4LinearWeight(MLXArray.zeros([kvOut, hiddenSize])),
            vProj: spec.useKEqV ? nil : Gemma4LinearWeight(MLXArray.zeros([kvOut, hiddenSize])),
            oProj: Gemma4LinearWeight(MLXArray.zeros([hiddenSize, qOut])),
            qNorm: ones([spec.headDim], dtype: .float32),
            kNorm: ones([spec.headDim], dtype: .float32)
        )
    }
}
