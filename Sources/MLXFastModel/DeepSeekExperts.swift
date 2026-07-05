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
        spec: DeepSeekRoutedExpertSpec,
        onRoutingSynced: (() -> Void)? = nil
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
        // RAM-pinned layers have no plan of their own, but the NEXT layer's
        // staging must still start now — otherwise the first streamed layer
        // after the pinned prefix reads its ~3 GiB cold instead of during the
        // pinned layers' compute.
        var stagingScheduled = false
        if tokenCount >= stagingMinimumTokenCount,
           let stager = loader.expertLayerStager
        {
            if let plan = loader.stagedExpertLayerPlan(layerIndex: spec.layerIndex) {
                stager.schedule(plan)
                stagingScheduled = true
            }
            if let nextPlan = loader.stagedExpertLayerPlan(layerIndex: spec.layerIndex + 1) {
                stager.schedule(nextPlan)
            }
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

        // Prefill-shaped batched path: when the layer's stacked code tensors
        // are already in RAM (staged whole-layer buffer or pinned hash-layer
        // store) run all experts through three sorted gatherQuantizedMM calls
        // instead of ~256 per-expert QMM dispatches plus per-expert
        // SwiGLU/gather/concat chains. Row values are the same per-row
        // quantized dot products over the same bytes; the correctness gate
        // guards the math. Any missing piece falls through to the existing
        // per-expert loop unchanged.
        if tokenCount >= stagingMinimumTokenCount,
           useStaged || loader.pinnedLayerIsFullyResident(layerIndex: spec.layerIndex),
           let batched = batchedResidentForward(
               x,
               selectedExperts: selectedExperts,
               batchSize: batchSize,
               sequenceLength: sequenceLength,
               topK: topK,
               loader: loader,
               spec: spec
           )
        {
            return batched
        }

        // Group activation flat-indices by expert so each expert runs one batched
        // matmul over all of its tokens instead of one matmul per token.
        var flatIndicesByExpert: [Int: [Int]] = [:]
        flatIndicesByExpert.reserveCapacity(min(outputCount, 256))
        var expertOrder: [Int] = []
        expertOrder.reserveCapacity(min(outputCount, 256))
        for (flatIndex, expertIndex) in selectedExperts.enumerated() {
            if flatIndicesByExpert[expertIndex] == nil {
                flatIndicesByExpert[expertIndex] = []
                expertOrder.append(expertIndex)
            }
            flatIndicesByExpert[expertIndex]?.append(flatIndex)
        }

        // Flatten the token axis once. Row (batch * sequenceLength + position)
        // equals x[batch, position], so an activation flat index maps to token row
        // flatIndex / topK. Gathering rows with a single `take` replaces the
        // per-token slice+concat that built each expert batch previously.
        let xFlat = x.reshaped([tokenCount, hiddenSize])

        // Decode/1-token path: the per-expert code slices are otherwise read
        // one blocking pread at a time on the compute thread. Read them
        // concurrently through a capacity-0 side bank (byte-identical
        // ranges) with PER-EXPERT completion signaling, so the per-expert
        // loop below builds expert e's subgraph as soon as e's OWN three
        // slices are in RAM — overlapping graph-build CPU work with the SSD
        // tail of the remaining experts' reads instead of barriering on the
        // whole batch. Anything not prefetched falls back to the normal
        // per-expert bank read.
        //
        // The reads are LAUNCHED on a side queue before the shared-expert
        // overlap eval runs: the shared expert depends only on x
        // (RAM-resident weights, no SSD read), so its eval is the one piece
        // of GPU work that can fill the read window — but its CPU wait must
        // not delay the start of the reads. Same reads, same arrays, same
        // eval; only the launch order changes.
        //
        // NOTE: temporal next-step expert prediction was measured on the
        // official runner (d4e4f946) and REGRESSED ~33 ms/step — decode is
        // disk-saturated, so speculative reads for layer n+1 steal
        // bandwidth from layer n's demand reads. Only the hash layers'
        // EXACT token-derived prefetch (scheduled at decode entry, no
        // speculation) remains.
        var decodePrefetch: [String: StagedExpertCode]?
        var decodeStream: DecodeExpertCodeStream?
        if tokenCount == 1, !useStaged {
            let group = DispatchGroup()
            let box = DecodePrefetchResultBox()
            let prefetchExpertOrder = expertOrder
            group.enter()
            decodePrefetchLaunchQueue.async {
                let consumed = loader.consumeScheduledPinnedDecodeExpertCodes(
                    layerIndex: spec.layerIndex
                )
                // Per-expert completion streaming: schedules the same
                // side-bank slice reads as the full-barrier prefetch, then
                // returns immediately (the box is filled as soon as the
                // stream object exists; reads continue in the background and
                // the loop below waits per expert).
                if let stream = loader.streamDecodeExpertCodes(
                    layerIndex: spec.layerIndex,
                    expertIndices: prefetchExpertOrder,
                    hiddenSize: spec.hiddenSize,
                    intermediateSize: spec.intermediateSize,
                    consumed: consumed
                ) {
                    box.stream = stream
                } else {
                    // Fast path unavailable (no side bank / disabled byte
                    // cache / nothing resolvable): keep the existing
                    // full-barrier prefetch unchanged as the fallback.
                    box.value = loader.mergeMissingDecodeExpertCodes(
                        consumed,
                        layerIndex: spec.layerIndex,
                        experts: prefetchExpertOrder,
                        hiddenSize: spec.hiddenSize,
                        intermediateSize: spec.intermediateSize
                    )
                }
                group.leave()
            }
            onRoutingSynced?()
            group.wait()
            decodePrefetch = box.value
            decodeStream = box.stream
        } else if useStaged {
            // Prefill/warmup staged path: build the active experts' base
            // MLXArrays from the staged layer buffer concurrently so the loop
            // below skips the serial Data->Metal copy (same win as decode).
            decodePrefetch = loader.prefetchStagedExpertCodes(
                layerIndex: spec.layerIndex,
                expertIndices: expertOrder,
                hiddenSize: spec.hiddenSize,
                intermediateSize: spec.intermediateSize
            )
        }

        var expertOutputs: [MLXArray] = []
        expertOutputs.reserveCapacity(flatIndicesByExpert.count)
        var scatterOrder: [Int] = []
        scatterOrder.reserveCapacity(outputCount)

        for expertIndex in expertOrder {
            guard let flatIndices = flatIndicesByExpert[expertIndex] else {
                continue
            }
            // Streamed decode reads: block only on THIS expert's slices,
            // then snapshot the (add-only) results so the expert's own
            // prebuilt entries are visible; later experts' reads keep
            // landing in the background while this subgraph is built.
            var expertPrefetch = decodePrefetch
            if let decodeStream {
                decodeStream.waitForExpert(expertIndex)
                expertPrefetch = decodeStream.snapshotResults()
            }
            let expertWeights = try weights(
                forExpert: expertIndex,
                loader: loader,
                spec: spec,
                preferStaged: useStaged,
                decodePrefetch: expertPrefetch
            )
            let tokens: MLXArray
            if tokenCount == 1, flatIndices.count == 1 {
                tokens = xFlat
            } else {
                let tokenRows = flatIndices.map { Int32($0 / topK) }
                tokens = xFlat.take(MLXArray(tokenRows), axis: 0)
            }
            let expertOutput = DeepSeekMLP.forward(
                tokens,
                weights: expertWeights,
                swigluLimit: spec.swigluLimit
            )
            // NOTE: an early asyncEval of the first expert's output was
            // measured locally at ~+5-7% decode s/token — the extra command
            // buffer per layer (43/step) costs more than the earlier dispatch
            // saves, matching the official finding that 43 extra prefill-side
            // command buffers raised decode per-step time 0.504 -> 0.549.
            // The streaming win is the per-expert read wait + graph-build
            // overlap alone; the layer-exit asyncEval below dispatches once.
            expertOutputs.append(expertOutput)
            scatterOrder.append(contentsOf: flatIndices)
        }

        let combined = concatenated(expertOutputs, axis: 0)

        // scatterOrder[row] is the activation flat index that produced combined
        // row `row`. Invert it so a single gather places every output back into
        // activation order, replacing the previous per-row scatter loop. When
        // the scatter order is already the identity (decode's first-seen
        // expert order visits slots in order), the gather is a no-op node --
        // skip it; the tensor values are identical by definition.
        var inverse = [Int32](repeating: 0, count: outputCount)
        var isIdentity = true
        for (row, flatIndex) in scatterOrder.enumerated() {
            inverse[flatIndex] = Int32(row)
            if flatIndex != row {
                isIdentity = false
            }
        }
        let ordered = isIdentity ? combined : combined.take(MLXArray(inverse), axis: 0)

        let result = ordered.reshaped([batchSize, sequenceLength, topK, hiddenSize])
        // Decode path: dispatch this layer's routed-expert graph now so the
        // GPU runs the QMMs while the CPU builds the next block's graph,
        // matching the batched prefill path's layer-exit dispatch. Scheduling
        // only; identical nodes evaluate either way.
        if tokenCount == 1 {
            asyncEval(result)
        }
        return result
    }

    /// Whole-layer routed-expert forward over RAM-resident stacked tensors.
    ///
    /// Bit-identity contract: every expert still runs the SAME per-expert
    /// quantizedMM kernel over the SAME token batch (an [n_e, hidden] GEMM)
    /// and the same weight bytes — the weights are just zero-copy first-axis
    /// views into ONE stacked MLXArray per projection instead of 256
    /// separately copied slice buffers. The limitedSwiGLU chain is elementwise
    /// and batch-size invariant, so evaluating it once over the concatenated
    /// [N, intermediate] activations produces the same elements as 256
    /// per-expert evaluations. gatherQuantizedMM is deliberately NOT used: its
    /// per-row matvec accumulation order differs from the per-expert GEMM and
    /// shifts bf16 argmax ties.
    private static func batchedResidentForward(
        _ x: MLXArray,
        selectedExperts: [Int],
        batchSize: Int,
        sequenceLength: Int,
        topK: Int,
        loader: DeepSeekWeightLoader,
        spec: DeepSeekRoutedExpertSpec
    ) -> MLXArray? {
        // Build the three stacked projections concurrently: each is one big
        // Data->Metal copy (vs 256 slice copies on the per-expert path).
        var projections = [DeepSeekWeightLoader.StackedExpertProjection?](repeating: nil, count: 3)
        let expectedShapes: [[Int]] = [
            [spec.intermediateSize, spec.hiddenSize],
            [spec.intermediateSize, spec.hiddenSize],
            [spec.hiddenSize, spec.intermediateSize],
        ]
        let kinds: [DeepSeekExpertProjection] = [.gate, .up, .down]
        projections.withUnsafeMutableBufferPointer { buffer in
            let sink = StackedProjectionSink(buffer: buffer)
            DispatchQueue.concurrentPerform(iterations: 3) { index in
                sink.buffer[index] = loader.stackedExpertProjection(
                    layerIndex: spec.layerIndex,
                    projection: kinds[index],
                    expectedShape: expectedShapes[index]
                )
            }
        }
        guard let gate = projections[0], let up = projections[1], let down = projections[2] else {
            return nil
        }

        let outputCount = selectedExperts.count
        // Group activation flat-indices by expert exactly like the per-expert
        // path; iteration order only affects concat order, which the inverse
        // permutation undoes.
        var flatIndicesByExpert: [Int: [Int]] = [:]
        flatIndicesByExpert.reserveCapacity(min(outputCount, 256))
        for (flatIndex, expertIndex) in selectedExperts.enumerated() {
            flatIndicesByExpert[expertIndex, default: []].append(flatIndex)
        }

        let xFlat = x.reshaped([batchSize * sequenceLength, spec.hiddenSize])
        let gateLogical = [spec.intermediateSize, spec.hiddenSize]
        let downLogical = [spec.hiddenSize, spec.intermediateSize]

        // One gather for every expert's token batch. Each expert's segment
        // view of xSorted holds the same rows in the same within-expert order
        // as the per-expert take, so the per-expert GEMM inputs are identical.
        var scatterOrder: [Int] = []
        scatterOrder.reserveCapacity(outputCount)
        var segmentLengths: [Int] = []
        segmentLengths.reserveCapacity(flatIndicesByExpert.count)
        var expertOrder: [Int] = []
        expertOrder.reserveCapacity(flatIndicesByExpert.count)
        for (expertIndex, flatIndices) in flatIndicesByExpert {
            expertOrder.append(expertIndex)
            segmentLengths.append(flatIndices.count)
            scatterOrder.append(contentsOf: flatIndices)
        }
        let sortedRows = scatterOrder.map { Int32($0 / topK) }
        let xSorted = xFlat.take(MLXArray(sortedRows), axis: 0)

        var gateOutputs: [MLXArray] = []
        var upOutputs: [MLXArray] = []
        var downWeights: [DeepSeekLinearWeight] = []
        gateOutputs.reserveCapacity(expertOrder.count)
        upOutputs.reserveCapacity(expertOrder.count)
        downWeights.reserveCapacity(expertOrder.count)

        var tokenStart = 0
        for (position, expertIndex) in expertOrder.enumerated() {
            let gateWeight = DeepSeekLinearWeight(
                weight: gate.weight[expertIndex],
                scales: gate.scales[expertIndex],
                biases: gate.biases.map { $0[expertIndex] },
                logicalShape: gateLogical,
                groupSize: gate.groupSize,
                bits: gate.bits,
                mode: gate.mode
            )
            let upWeight = DeepSeekLinearWeight(
                weight: up.weight[expertIndex],
                scales: up.scales[expertIndex],
                biases: up.biases.map { $0[expertIndex] },
                logicalShape: gateLogical,
                groupSize: up.groupSize,
                bits: up.bits,
                mode: up.mode
            )
            downWeights.append(
                DeepSeekLinearWeight(
                    weight: down.weight[expertIndex],
                    scales: down.scales[expertIndex],
                    biases: down.biases.map { $0[expertIndex] },
                    logicalShape: downLogical,
                    groupSize: down.groupSize,
                    bits: down.bits,
                    mode: down.mode
                )
            )
            let tokens = xSorted[tokenStart..<(tokenStart + segmentLengths[position])]
            tokenStart += segmentLengths[position]
            gateOutputs.append(DeepSeekOps.linear(input: tokens, weight: gateWeight))
            upOutputs.append(DeepSeekOps.linear(input: tokens, weight: upWeight))
        }

        // One elementwise SwiGLU over all experts' activations. Elementwise
        // ops are batch-size invariant, so each row matches the per-expert
        // evaluation bit for bit.
        let hiddenAll = DeepSeekOps.limitedSwiGLU(
            gate: concatenated(gateOutputs, axis: 0),
            up: concatenated(upOutputs, axis: 0),
            limit: spec.swigluLimit
        )

        var expertOutputs: [MLXArray] = []
        expertOutputs.reserveCapacity(downWeights.count)
        var segmentStart = 0
        for (index, weight) in downWeights.enumerated() {
            let segment = hiddenAll[segmentStart..<(segmentStart + segmentLengths[index])]
            expertOutputs.append(DeepSeekOps.linear(input: segment, weight: weight))
            segmentStart += segmentLengths[index]
        }

        let combined = concatenated(expertOutputs, axis: 0)
        var inverse = [Int32](repeating: 0, count: outputCount)
        for (row, flatIndex) in scatterOrder.enumerated() {
            inverse[flatIndex] = Int32(row)
        }
        let ordered = combined.take(MLXArray(inverse), axis: 0)
        return ordered.reshaped([batchSize, sequenceLength, topK, spec.hiddenSize])
    }

    public static func weights(
        forExpert expertIndex: Int,
        loader: DeepSeekWeightLoader,
        spec: DeepSeekRoutedExpertSpec,
        preferStaged: Bool = false,
        decodePrefetch: [String: StagedExpertCode]? = nil
    ) throws -> DeepSeekMLPWeights {
        try DeepSeekMLPWeights(
            gate: loader.routedExpertLinearWeight(
                layerIndex: spec.layerIndex,
                projection: .gate,
                expectedShape: [spec.intermediateSize, spec.hiddenSize],
                expertIndex: expertIndex,
                preferStaged: preferStaged,
                decodePrefetch: decodePrefetch
            ),
            up: loader.routedExpertLinearWeight(
                layerIndex: spec.layerIndex,
                projection: .up,
                expectedShape: [spec.intermediateSize, spec.hiddenSize],
                expertIndex: expertIndex,
                preferStaged: preferStaged,
                decodePrefetch: decodePrefetch
            ),
            down: loader.routedExpertLinearWeight(
                layerIndex: spec.layerIndex,
                projection: .down,
                expectedShape: [spec.hiddenSize, spec.intermediateSize],
                expertIndex: expertIndex,
                preferStaged: preferStaged,
                decodePrefetch: decodePrefetch
            )
        )
    }
}

// Lets concurrentPerform write the three stacked projections into distinct
// slots from worker threads; each index is written by exactly one iteration.
private struct StackedProjectionSink: @unchecked Sendable {
    let buffer: UnsafeMutableBufferPointer<DeepSeekWeightLoader.StackedExpertProjection?>
}

// Single-producer result slot for the decode-step prefetch launched on the
// side queue; the DispatchGroup wait orders the writes before the reads.
// Exactly one of `stream` (per-expert completion streaming) or `value`
// (full-barrier fallback map) is set per decode step.
private final class DecodePrefetchResultBox: @unchecked Sendable {
    var value: [String: StagedExpertCode]?
    var stream: DecodeExpertCodeStream?
}

// Side queue that only launches the decode-step slice prefetch, so the
// shared-expert overlap eval on the compute thread runs concurrently with the
// reads instead of preceding them.
private let decodePrefetchLaunchQueue = DispatchQueue(
    label: "mlxfast.decode.prefetch-launch",
    qos: .userInitiated
)

/// Below this many tokens the unique-expert count is small enough that
/// per-slice streaming reads less than a whole stacked tensor; decode and the
/// hidden one-token gates stay on the existing path. At 64 tokens (384
/// activations) the expected unique-expert coverage already exceeds 3/4 of
/// the stack, and the scored 512-token prefills touch essentially all of it.
private let stagingMinimumTokenCount = 64
