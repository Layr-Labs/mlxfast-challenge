import Foundation
import MLX
@testable import MLXFastModel
import Testing

@Test
func gemma4MLPMatchesGatedGeluTanhReferenceWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    // Identity gate/up/down projections isolate the activation: down(gelu(x) * x).
    let identity = MLXArray([Float(1), 0, 0, 1], [2, 2])
    let weights = Gemma4MLPWeights(
        gate: Gemma4LinearWeight(identity),
        up: Gemma4LinearWeight(identity),
        down: Gemma4LinearWeight(identity)
    )

    let x = MLXArray([Float(1), Float(-1)], [1, 2])
    let output = Gemma4MLP.forward(x, weights: weights).asArray(Float.self)

    let expectedGelu1 = Float(geluTanhReference(1.0))
    let expectedGeluNeg1 = Float(geluTanhReference(-1.0))
    #expect(abs(output[0] - expectedGelu1 * 1.0) < 1e-4)
    #expect(abs(output[1] - expectedGeluNeg1 * (-1.0)) < 1e-4)
}

private func geluTanhReference(_ x: Double) -> Double {
    0.5 * x * (1 + tanh((2.0 / Double.pi).squareRoot() * (x + 0.044715 * x * x * x)))
}
