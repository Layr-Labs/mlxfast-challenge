import Foundation
import MLX
import MLXFastCore

public struct DeepSeekHyperConnectionMix {
    public let pre: MLXArray
    public let post: MLXArray
    public let combination: MLXArray
}

public struct DeepSeekHyperConnectionOutput {
    public let collapsed: MLXArray
    public let post: MLXArray
    public let combination: MLXArray
}

public enum DeepSeekHyperConnection {
    public static func collapse(
        _ x: MLXArray,
        fn: MLXArray,
        base: MLXArray,
        scale: MLXArray,
        hcMult: Int,
        sinkhornIters: Int,
        eps: Double,
        normEps: Double
    ) throws -> DeepSeekHyperConnectionOutput {
        try validateInput(x, hcMult: hcMult)

        let y = x.asType(.float32)
        let normalized = weightlessRMSNorm(y.flattened(start: -2), eps: normEps)
        let mixes = matmul(normalized, fn.T)
        let mix = try splitSinkhorn(
            mixes: mixes,
            scale: scale,
            base: base,
            hcMult: hcMult,
            sinkhornIters: sinkhornIters,
            eps: eps
        )

        let collapsed = (mix.pre.expandedDimensions(axis: -1) * y)
            .sum(axis: 2)
            .asType(x.dtype)
        return DeepSeekHyperConnectionOutput(
            collapsed: collapsed,
            post: mix.post,
            combination: mix.combination
        )
    }

    public static func splitSinkhorn(
        mixes: MLXArray,
        scale: MLXArray,
        base: MLXArray,
        hcMult: Int,
        sinkhornIters: Int,
        eps: Double
    ) throws -> DeepSeekHyperConnectionMix {
        guard hcMult > 0 else {
            throw MLXFastError.invalidInput("HyperConnection hcMult must be positive")
        }
        let mixWidth = (2 + hcMult) * hcMult
        guard mixes.shape.last == mixWidth else {
            throw MLXFastError.invalidInput(
                "HyperConnection mixes last dimension \(mixes.shape.last ?? -1) expected \(mixWidth)"
            )
        }

        let mixes = mixes.asType(.float32)
        let scale = scale.asType(.float32)
        let base = base.asType(.float32)
        let eps = Float(eps)

        let pre = sigmoid(mixes[.ellipsis, 0..<hcMult] * scale[0] + base[0..<hcMult]) + eps
        let post = 2.0 * sigmoid(
            mixes[.ellipsis, hcMult..<(2 * hcMult)] * scale[1] + base[hcMult..<(2 * hcMult)]
        )

        let combFlat = mixes[.ellipsis, (2 * hcMult)...] * scale[2] + base[(2 * hcMult)...]
        let combShape = Array(mixes.shape.dropLast()) + [hcMult, hcMult]
        var combination = combFlat.reshaped(combShape)
        combination = softmax(combination, axis: -1, precise: true) + eps
        combination = combination / (combination.sum(axis: -2, keepDims: true) + eps)

        for _ in 0..<max(sinkhornIters - 1, 0) {
            combination = combination / (combination.sum(axis: -1, keepDims: true) + eps)
            combination = combination / (combination.sum(axis: -2, keepDims: true) + eps)
        }

        return DeepSeekHyperConnectionMix(
            pre: pre,
            post: post,
            combination: combination
        )
    }

    public static func expand(
        _ x: MLXArray,
        residual: MLXArray,
        post: MLXArray,
        combination: MLXArray
    ) -> MLXArray {
        let postScaled = post.expandedDimensions(axis: -1)
            * x.expandedDimensions(axis: -2).asType(.float32)
        let residualMixed = matmul(
            combination.swappedAxes(-1, -2),
            residual.asType(.float32)
        )
        return (postScaled + residualMixed).asType(x.dtype)
    }

    public static func head(
        _ x: MLXArray,
        fn: MLXArray,
        base: MLXArray,
        scale: MLXArray,
        hcMult: Int,
        eps: Double,
        normEps: Double
    ) throws -> MLXArray {
        try validateInput(x, hcMult: hcMult)

        let y = x.asType(.float32)
        let normalized = weightlessRMSNorm(y.flattened(start: -2), eps: normEps)
        let mixes = matmul(normalized, fn.T)
        let pre = sigmoid(mixes * scale[0] + base) + Float(eps)
        return (pre.expandedDimensions(axis: -1) * y)
            .sum(axis: 2)
            .asType(x.dtype)
    }

    public static func weightlessRMSNorm(_ x: MLXArray, eps: Double) -> MLXArray {
        x * rsqrt(mean(square(x), axis: -1, keepDims: true) + Float(eps))
    }

    private static func validateInput(_ x: MLXArray, hcMult: Int) throws {
        guard x.shape.count == 4 else {
            throw MLXFastError.invalidInput("HyperConnection input must have shape [batch, length, hc, hidden]")
        }
        guard x.shape[2] == hcMult else {
            throw MLXFastError.invalidInput(
                "HyperConnection input hc dimension \(x.shape[2]) expected \(hcMult)"
            )
        }
    }
}
