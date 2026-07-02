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
        let hiddenSize = spec.hiddenSize
        let tokenCount = batchSize * sequenceLength

        // Prefill-shaped calls touch essentially every expert, so schedule
        // whole-stacked-tensor staging for this layer (and the next) BEFORE
        // the routing sync below: the sequential ~1 GiB reads then overlap
        // the GPU drain instead of following it. Staging needs no routing
        // indices, and a failed stage falls back to the per-slice path.
        var stagingScheduled = false
        if tokenCount >= stagingMinimumTokenCount,
           let stager = loader.expertLayerStager,
           let plan = loader.stagedExpertLayerPlan(layerIndex: spec.layerIndex)
        {
            stager.schedule(plan)
            if let nextPlan = loader.stagedExpertLayerPlan(layerIndex: spec.layerIndex + 1) {
                stager.schedule(nextPlan)
            }
            stagingScheduled = true
        }

        let selectedExperts = expertIndices.asArray(Int32.self).map(Int.init)
        let useStaged = stagingScheduled
            && loader.expertLayerStager?.waitForLayer(spec.layerIndex) == true
        defer {
            if useStaged {
                loader.expertLayerStager?.releaseLayer(spec.layerIndex)
            }
        }
        if !useStaged {
            // Kernel read-ahead for every byte range this layer is about to
            // pread, so SSD I/O overlaps the per-expert GPU compute below.
            loader.expertPrefetcher.prefetch(layerIndex: spec.layerIndex, expertIndices: selectedExperts)
        }

        let outputCount = tokenCount * topK
        guard outputCount > 0 else {
            return zeros([batchSize, sequenceLength, topK, hiddenSize], dtype: x.dtype)
        }

        // Group activation flat-indices by expert so each expert runs one batched
        // matmul over all of its tokens instead of one matmul per token.
        var flatIndicesByExpert: [Int: [Int]] = [:]
        flatIndicesByExpert.reserveCapacity(min(outputCount, 256))
        for (flatIndex, expertIndex) in selectedExperts.enumerated() {
            flatIndicesByExpert[expertIndex, default: []].append(flatIndex)
        }

        // Flatten the token axis once. Row (batch * sequenceLength + position)
        // equals x[batch, position], so an activation flat index maps to token row
        // flatIndex / topK. Gathering rows with a single `take` replaces the
        // per-token slice+concat that built each expert batch previously.
        let xFlat = x.reshaped([tokenCount, hiddenSize])

        var expertOutputs: [MLXArray] = []
        expertOutputs.reserveCapacity(flatIndicesByExpert.count)
        var scatterOrder: [Int] = []
        scatterOrder.reserveCapacity(outputCount)

        for (expertIndex, flatIndices) in flatIndicesByExpert {
            let expertWeights = try weights(
                forExpert: expertIndex,
                loader: loader,
                spec: spec,
                preferStaged: useStaged
            )
            let tokenRows = flatIndices.map { Int32($0 / topK) }
            let tokens = xFlat.take(MLXArray(tokenRows), axis: 0)
            let expertOutput = DeepSeekMLP.forward(
                tokens,
                weights: expertWeights,
                swigluLimit: spec.swigluLimit
            )
            expertOutputs.append(expertOutput)
            scatterOrder.append(contentsOf: flatIndices)
        }

        let combined = concatenated(expertOutputs, axis: 0)

        // scatterOrder[row] is the activation flat index that produced combined
        // row `row`. Invert it so a single gather places every output back into
        // activation order, replacing the previous per-row scatter loop.
        var inverse = [Int32](repeating: 0, count: outputCount)
        for (row, flatIndex) in scatterOrder.enumerated() {
            inverse[flatIndex] = Int32(row)
        }
        let ordered = combined.take(MLXArray(inverse), axis: 0)

        return ordered.reshaped([batchSize, sequenceLength, topK, hiddenSize])
    }

    public static func weights(
        forExpert expertIndex: Int,
        loader: DeepSeekWeightLoader,
        spec: DeepSeekRoutedExpertSpec,
        preferStaged: Bool = false
    ) throws -> DeepSeekMLPWeights {
        try DeepSeekMLPWeights(
            gate: loader.expertLinearWeight(
                candidates: DeepSeekWeightNames.routedExpert(
                    layerIndex: spec.layerIndex,
                    expertIndex: expertIndex,
                    projection: .gate
                ),
                expectedShape: [spec.intermediateSize, spec.hiddenSize],
                expertIndex: expertIndex,
                preferStaged: preferStaged
            ),
            up: loader.expertLinearWeight(
                candidates: DeepSeekWeightNames.routedExpert(
                    layerIndex: spec.layerIndex,
                    expertIndex: expertIndex,
                    projection: .up
                ),
                expectedShape: [spec.intermediateSize, spec.hiddenSize],
                expertIndex: expertIndex,
                preferStaged: preferStaged
            ),
            down: loader.expertLinearWeight(
                candidates: DeepSeekWeightNames.routedExpert(
                    layerIndex: spec.layerIndex,
                    expertIndex: expertIndex,
                    projection: .down
                ),
                expectedShape: [spec.hiddenSize, spec.intermediateSize],
                expertIndex: expertIndex,
                preferStaged: preferStaged
            )
        )
    }
}

/// Below this many tokens the unique-expert count is small enough that
/// per-slice streaming reads less than a whole stacked tensor; decode and the
/// hidden one-token gates stay on the existing path. At 64 tokens (384
/// activations) the expected unique-expert coverage already exceeds 3/4 of
/// the stack, and the scored 512-token prefills touch essentially all of it.
private let stagingMinimumTokenCount = 64
