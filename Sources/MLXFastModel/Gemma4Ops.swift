import Foundation
import MLX
import MLXFastCore

public enum Gemma4Ops {
    public static func embedding(inputIDs: MLXArray, weight: Gemma4LinearWeight) -> MLXArray {
        guard let scales = weight.scales else {
            return weight.weight[inputIDs]
        }
        return dequantized(
            weight.weight[inputIDs],
            scales: scales[inputIDs],
            biases: weight.biases.map { $0[inputIDs] },
            groupSize: weight.groupSize,
            bits: weight.bits,
            mode: .affine
        )
    }

    public static func linear(_ input: MLXArray, _ weight: Gemma4LinearWeight) -> MLXArray {
        guard let scales = weight.scales else {
            return matmul(input, weight.weight.T)
        }
        return quantizedMM(
            input,
            weight.weight,
            scales: scales,
            biases: weight.biases,
            transpose: true,
            groupSize: weight.groupSize,
            bits: weight.bits,
            mode: .affine
        )
    }

    /// Gemma-style RMSNorm: `x / rms(x) * weight`. Unlike the classic
    /// Gemma 1/2/3 `(1 + weight)` convention, the mlx-community 4-bit Gemma 4
    /// checkpoint already bakes the `+1` shift into the stored weight values
    /// (confirmed both by the mlx-vlm/`transformers` reference implementations,
    /// which apply the learned weight directly with no offset, and by
    /// inspecting the actual checkpoint tensor values, which center near the
    /// shifted magnitude rather than near zero). Applying an extra `+1` here
    /// would double-shift and diverge from the reference numerically.
    public static func rmsNorm(_ x: MLXArray, weight: MLXArray, eps: Double) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: Float(eps))
    }

    /// Scale-less RMSNorm (Gemma 4's `v_norm`, applied to attention values).
    /// `MLXFast.rmsNorm` requires a non-optional weight array in this MLX
    /// Swift version, so the weightless case is computed directly -- the same
    /// convention already used by the hyper-connection weightless norm
    /// elsewhere in this package.
    public static func rmsNormNoWeight(_ x: MLXArray, eps: Double) -> MLXArray {
        x * rsqrt(mean(square(x), axis: -1, keepDims: true) + Float(eps))
    }

    /// `gelu_pytorch_tanh` activation: the tanh approximation of GELU used by
    /// Gemma's gated MLP.
    public static func geluTanh(_ x: MLXArray) -> MLXArray {
        let sqrt2OverPi = Float((2.0 / Double.pi).squareRoot())
        return 0.5 * x * (1 + tanh(sqrt2OverPi * (x + 0.044715 * x * x * x)))
    }

    public static func logitSoftcap(_ x: MLXArray, cap: Double) -> MLXArray {
        guard cap > 0 else {
            return x
        }
        let capValue = Float(cap)
        return tanh(x / capValue) * capValue
    }
}
