import Foundation
import MLX
import MLXFastCore

public enum DeepSeekOps {
    public static func embedding(inputIDs: MLXArray, weight: MLXArray) -> MLXArray {
        weight[inputIDs]
    }

    public static func embedding(inputIDs: MLXArray, weight: DeepSeekLinearWeight) -> MLXArray {
        guard let scales = weight.scales else {
            return embedding(inputIDs: inputIDs, weight: weight.weight)
        }
        return dequantized(
            weight.weight[inputIDs],
            scales: scales[inputIDs],
            biases: weight.biases.map { $0[inputIDs] },
            groupSize: weight.groupSize,
            bits: weight.bits,
            mode: weight.mode
        )
    }

    public static func linear(input: MLXArray, weight: MLXArray, bias: MLXArray? = nil) -> MLXArray {
        if let bias {
            return addMM(bias, input, weight.T)
        }
        return matmul(input, weight.T)
    }

    public static func linear(
        input: MLXArray,
        weight: DeepSeekLinearWeight,
        bias: MLXArray? = nil
    ) -> MLXArray {
        guard let scales = weight.scales else {
            return linear(input: input, weight: weight.weight, bias: bias)
        }
        var projected = quantizedMM(
            input,
            weight.weight,
            scales: scales,
            biases: weight.biases,
            transpose: true,
            groupSize: weight.groupSize,
            bits: weight.bits,
            mode: weight.mode
        )
        if let bias {
            projected = projected + bias
        }
        return projected
    }

    public static func multiLinear(input: MLXArray, weight: MLXArray, transpose: Bool = true) -> MLXArray {
        if transpose {
            return matmul(input, weight.swappedAxes(-1, -2))
        }
        return matmul(input, weight)
    }

    public static func multiLinear(
        input: MLXArray,
        weight: DeepSeekLinearWeight,
        transpose: Bool = true
    ) throws -> MLXArray {
        guard weight.isQuantized else {
            return multiLinear(input: input, weight: weight.weight, transpose: transpose)
        }
        guard transpose else {
            throw MLXFastError.invalidInput("quantized grouped linear requires transpose=true")
        }
        guard input.shape.count == 4, weight.logicalShape.count == 3 else {
            throw MLXFastError.invalidInput(
                "quantized grouped linear expects input [batch, groups, length, input] and weight [groups, output, input]"
            )
        }

        let groupCount = input.shape[1]
        guard groupCount == weight.logicalShape[0] else {
            throw MLXFastError.invalidInput(
                "quantized grouped linear input group count \(groupCount) does not match weight group count \(weight.logicalShape[0])"
            )
        }
        let outputDimensions = weight.logicalShape[1]
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(groupCount)
        for groupIndex in 0..<groupCount {
            let groupInput = input[0..., groupIndex, 0..., 0...]
            let rowStart = groupIndex * outputDimensions
            let rowEnd = rowStart + outputDimensions
            let groupWeight = weight.rows(
                rowStart..<rowEnd,
                logicalShape: [outputDimensions, weight.logicalShape[2]]
            )
            outputs.append(
                linear(input: groupInput, weight: groupWeight)
                    .expandedDimensions(axis: 1)
            )
        }
        return concatenated(outputs, axis: 1)
    }

    public static func rmsNorm(input: MLXArray, weight: MLXArray, eps: Double) -> MLXArray {
        MLXFast.rmsNorm(input, weight: weight, eps: Float(eps))
    }

    public static func silu(_ input: MLXArray) -> MLXArray {
        input * sigmoid(input)
    }

    public static func limitedSwiGLU(gate: MLXArray, up: MLXArray, limit: Double) -> MLXArray {
        guard limit > 0 else {
            return silu(gate) * up
        }
        let cappedGate = minimum(gate, Float(limit))
        let clippedUp = clip(up, min: Float(-limit), max: Float(limit))
        return silu(cappedGate) * clippedUp
    }
}
