import Foundation
import MLX
@testable import MLXFastCore
@testable import MLXFastModel
import Testing

@Test
func gemma4RoPERejectsOddOrNonPositiveDimensions() throws {
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4RoPE(dims: 3, spec: Gemma4RopeSpec(theta: 10_000, type: "default"))
    }
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4RoPE(dims: 0, spec: Gemma4RopeSpec(theta: 10_000, type: "default"))
    }
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4RoPE(dims: 256, spec: Gemma4RopeSpec(theta: 0, type: "default"))
    }
}

@Test
func gemma4RoPERejectsUnsupportedType() throws {
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4RoPE(dims: 256, spec: Gemma4RopeSpec(theta: 10_000, type: "yarn"))
    }
}

@Test
func gemma4RoPEStandardIsIdentityAtZeroOffsetWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let rope = try Gemma4RoPE(dims: 8, spec: Gemma4RopeSpec(theta: 10_000, type: "default"))
    let input = MLXArray((0..<8).map { Float($0) }, [1, 1, 1, 8])
    let output = rope.applied(to: input, offset: 0)

    #expect(output.shape == [1, 1, 1, 8])
}

@Test
func gemma4RoPEProportionalOnlyRotatesPartialDimensionsWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    // headDim=512, partialRotaryFactor=0.25 -> rotatedDims=128 (matches the
    // full-attention layer configuration).
    let rope = try Gemma4RoPE(
        dims: 512,
        spec: Gemma4RopeSpec(theta: 1_000_000, type: "proportional", partialRotaryFactor: 0.25)
    )
    let input = MLXArray((0..<512).map { Float($0) }, [1, 1, 1, 512])

    // At offset 0 every rotation angle is zero, so the proportional RoPE
    // should be the identity regardless of which dimensions are rotated.
    let output = rope.applied(to: input, offset: 0)
    #expect(output.shape == [1, 1, 1, 512])
    let outputValues = output.asArray(Float.self)
    let inputValues = input.asArray(Float.self)
    for (actual, expected) in zip(outputValues, inputValues) {
        #expect(abs(actual - expected) <= 1e-3)
    }

    // At a non-zero offset, only the first 64 elements of each half (indices
    // 0..<64 and 256..<320) should move; the remaining elements in each half
    // are untouched by construction.
    let offsetOutput = rope.applied(to: input, offset: 5)
    let offsetValues = offsetOutput.asArray(Float.self)
    for index in 64..<256 {
        #expect(abs(offsetValues[index] - inputValues[index]) <= 1e-3)
    }
    for index in 320..<512 {
        #expect(abs(offsetValues[index] - inputValues[index]) <= 1e-3)
    }
}

@Test
func gemma4RoPECacheReturnsSameInstanceForSameParameters() throws {
    let first = try Gemma4RoPECache.shared(
        dims: 256,
        spec: Gemma4RopeSpec(theta: 10_000, type: "default")
    )
    let second = try Gemma4RoPECache.shared(
        dims: 256,
        spec: Gemma4RopeSpec(theta: 10_000, type: "default")
    )
    #expect(first === second)

    let different = try Gemma4RoPECache.shared(
        dims: 512,
        spec: Gemma4RopeSpec(theta: 1_000_000, type: "proportional", partialRotaryFactor: 0.25)
    )
    #expect(first !== different)
}
