import Foundation
import MLX
import MLXFastCore

public struct DeepSeekRoutedExpertSpec: Equatable {
    public let layerIndex: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let swigluLimit: Double

    public init(
        layerIndex: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        swigluLimit: Double
    ) {
        self.layerIndex = layerIndex
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.swigluLimit = swigluLimit
    }

    public init(layerIndex: Int, config: DeepSeekConfig) {
        self.init(
            layerIndex: layerIndex,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize,
            swigluLimit: config.swigluLimit
        )
    }
}

public enum DeepSeekRoutedExperts {
    public static func forward(
        _ x: MLXArray,
        expertIndices: MLXArray,
        loader: DeepSeekWeightLoader,
        spec: DeepSeekRoutedExpertSpec
    ) throws -> MLXArray {
        guard x.shape.count == 3 else {
            throw MLXFastError.invalidInput("routed expert input must have shape [batch, length, hidden]")
        }
        guard expertIndices.shape.count == 3 else {
            throw MLXFastError.invalidInput("expert indices must have shape [batch, length, topK]")
        }
        guard x.shape[0] == expertIndices.shape[0], x.shape[1] == expertIndices.shape[1] else {
            throw MLXFastError.invalidInput("expert indices batch/length must match routed expert input")
        }
        guard x.shape[2] == spec.hiddenSize else {
            throw MLXFastError.invalidInput(
                "routed expert input hidden size \(x.shape[2]) expected \(spec.hiddenSize)"
            )
        }

        let batchSize = x.shape[0]
        let sequenceLength = x.shape[1]
        let topK = expertIndices.shape[2]
        let selectedExperts = expertIndices.asArray(Int32.self).map(Int.init)

        let outputCount = batchSize * sequenceLength * topK
        guard outputCount > 0 else {
            return zeros([batchSize, sequenceLength, topK, spec.hiddenSize], dtype: x.dtype)
        }

        var flatIndicesByExpert: [Int: [Int]] = [:]
        flatIndicesByExpert.reserveCapacity(min(outputCount, 256))
        for (flatIndex, expertIndex) in selectedExperts.enumerated() {
            flatIndicesByExpert[expertIndex, default: []].append(flatIndex)
        }

        var outputs = Array<MLXArray?>(repeating: nil, count: outputCount)
        for (expertIndex, flatIndices) in flatIndicesByExpert {
            let expertWeights = try weights(
                forExpert: expertIndex,
                loader: loader,
                spec: spec
            )
            let tokens = concatenated(
                flatIndices.map { flatIndex in
                    let tokenIndex = flatIndex / topK
                    let batch = tokenIndex / sequenceLength
                    let position = tokenIndex % sequenceLength
                    return x[batch, position].reshaped([1, spec.hiddenSize])
                },
                axis: 0
            )
            let expertOutput = DeepSeekMLP.forward(
                tokens,
                weights: expertWeights,
                swigluLimit: spec.swigluLimit
            )
            for (indexInExpertBatch, flatIndex) in flatIndices.enumerated() {
                outputs[flatIndex] = expertOutput[indexInExpertBatch].reshaped([1, spec.hiddenSize])
            }
        }

        let orderedOutputs = try outputs.enumerated().map { flatIndex, output in
            guard let output else {
                throw MLXFastError.invalidInput("missing routed expert output at flat index \(flatIndex)")
            }
            return output
        }

        return concatenated(orderedOutputs, axis: 0)
            .reshaped([batchSize, sequenceLength, topK, spec.hiddenSize])
    }

    public static func weights(
        forExpert expertIndex: Int,
        loader: DeepSeekWeightLoader,
        spec: DeepSeekRoutedExpertSpec
    ) throws -> DeepSeekMLPWeights {
        try DeepSeekMLPWeights(
            gate: loader.expertLinearWeight(
                candidates: DeepSeekWeightNames.routedExpert(
                    layerIndex: spec.layerIndex,
                    expertIndex: expertIndex,
                    projection: .gate
                ),
                expectedShape: [spec.intermediateSize, spec.hiddenSize],
                expertIndex: expertIndex
            ),
            up: loader.expertLinearWeight(
                candidates: DeepSeekWeightNames.routedExpert(
                    layerIndex: spec.layerIndex,
                    expertIndex: expertIndex,
                    projection: .up
                ),
                expectedShape: [spec.intermediateSize, spec.hiddenSize],
                expertIndex: expertIndex
            ),
            down: loader.expertLinearWeight(
                candidates: DeepSeekWeightNames.routedExpert(
                    layerIndex: spec.layerIndex,
                    expertIndex: expertIndex,
                    projection: .down
                ),
                expectedShape: [spec.hiddenSize, spec.intermediateSize],
                expertIndex: expertIndex
            )
        )
    }
}
