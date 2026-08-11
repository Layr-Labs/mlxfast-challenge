import Foundation
import MLX
import MLXFast
import MLXNN


public let compiledSiluProduct: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { gate, up in
        MLXNN.silu(gate) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

public let weightedExpertSum: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { outputs, weights in
    (outputs * MLX.expandedDimensions(weights, axis: -1)).sum(axis: -2)
}


public let safeGeluApproximate: @Sendable (MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray) -> MLXArray = { (x: MLXArray) -> MLXArray in
        0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

public class SafeGELU: Module, UnaryLayer {
    public override init() { super.init() }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        safeGeluApproximate(x)
    }
}

private let compiledSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, up: MLXArray) -> MLXArray in
        MLXNN.silu(gate) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

private let compiledGeGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, up: MLXArray) -> MLXArray in
        (0.5 * gate * (1 + tanh(sqrt(2 / Float.pi) * (gate + 0.044715 * gate * gate * gate)))) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

private let inversePermutationScatterEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_INVERSE_SCATTER"] != "0"

private let inversePermutationScatterKernel = MLXFast.metalKernel(
    name: "mlx_lm_inverse_permutation_scatter_u32_v1",
    inputNames: ["order"],
    outputNames: ["inverse"],
    source: """
        uint i = thread_position_in_grid.x;
        inverse[order[i]] = i;
        """,
    ensureRowContiguous: false
)

private let routeCountingSortEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ROUTE_COUNTING_SORT"] != "0"

private let routeSortTile = 128

private let routeTileHistKernel = MLXFast.metalKernel(
    name: "mlx_lm_route_csort_hist_u32_v1",
    inputNames: ["keys"],
    outputNames: ["tile_hist"],
    source: """
        constexpr uint TILE = \(routeSortTile);
        uint t = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        threadgroup atomic_uint counts[256];
        atomic_store_explicit(&counts[lid], 0u, memory_order_relaxed);
        atomic_store_explicit(&counts[lid + 128], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint key = keys[t * TILE + lid];
        atomic_fetch_add_explicit(&counts[key], 1u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_hist[t * 256 + lid] =
            atomic_load_explicit(&counts[lid], memory_order_relaxed);
        tile_hist[t * 256 + lid + 128] =
            atomic_load_explicit(&counts[lid + 128], memory_order_relaxed);
        """,
    ensureRowContiguous: false
)

private let routeScanKernel = MLXFast.metalKernel(
    name: "mlx_lm_route_csort_scan_u32_v1",
    inputNames: ["tile_hist"],
    outputNames: ["base"],
    source: """
        // One threadgroup of 256 threads: total per key, then exclusive scan
        // over keys, serial on lane 0 (256 adds, launched once per forward).
        uint k = thread_position_in_threadgroup.x;
        uint tiles = tile_hist_shape[0] / 256;
        uint total = 0;
        for (uint t = 0; t < tiles; ++t) {
            total += tile_hist[t * 256 + k];
        }
        threadgroup uint totals[256];
        totals[k] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k == 0) {
            uint acc = 0;
            for (uint i = 0; i < 256; ++i) {
                uint c = totals[i];
                totals[i] = acc;
                acc += c;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        base[k] = totals[k];
        """,
    ensureRowContiguous: false
)

private let routeScatterKernel = MLXFast.metalKernel(
    name: "mlx_lm_route_csort_scatter_u32_v1",
    inputNames: ["keys", "tile_hist", "base"],
    outputNames: ["order"],
    source: """
        constexpr uint TILE = \(routeSortTile);
        uint t = threadgroup_position_in_grid.x;
        uint k = thread_position_in_threadgroup.x;
        // Rank base for key k in tile t: global base + counts in earlier tiles.
        uint off = base[k];
        for (uint tp = 0; tp < t; ++tp) {
            off += tile_hist[tp * 256 + k];
        }
        // Walk this tile's slice in input order: stability by construction.
        for (uint i = 0; i < TILE; ++i) {
            uint idx = t * TILE + i;
            if (keys[idx] == k) {
                order[off++] = idx;
            }
        }
        """,
    ensureRowContiguous: false
)

private func routeCountingSort(_ indices: MLXArray) -> MLXArray? {
    let n = indices.size
    guard routeCountingSortEnabled, n > 0, n % routeSortTile == 0 else { return nil }
    let tiles = n / routeSortTile
    let hist = routeTileHistKernel(
        [indices],
        grid: (tiles * routeSortTile, 1, 1),
        threadGroup: (routeSortTile, 1, 1),
        outputShapes: [[tiles * 256]],
        outputDTypes: [.uint32]
    )[0]
    let base = routeScanKernel(
        [hist],
        grid: (256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[256]],
        outputDTypes: [.uint32]
    )[0]
    return routeScatterKernel(
        [indices, hist, base],
        grid: (tiles * 256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[n]],
        outputDTypes: [.uint32]
    )[0]
}

private let routeFusedScatterEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ROUTE_FUSED_SCATTER"] != "0"

private let routeFusedScatterTopK = 8

private let routeFusedScatterKernel: MLXFast.MLXFastKernel = {
    let m = routeFusedScatterTopK
    return MLXFast.metalKernel(
        name: "mlx_lm_route_csort_scatter_fused_m\(m)_u32_v4",
        inputNames: ["keys"],
        outputNames: ["row_order", "sorted_keys", "inverse_order"],
        source: """
            constexpr uint TILE = \(routeSortTile);
            constexpr uint M = \(m);
            uint t = threadgroup_position_in_grid.x;
            uint k = thread_position_in_threadgroup.x;
            uint simd_id = k / 32;
            uint lane = k % 32;
            uint n = keys_shape[0];
            // In-threadgroup histograms replace both the standalone hist
            // dispatch and the scan dispatch: one cooperative pass counts
            // every key (totals) and every key in earlier tiles (before),
            // then a simd exclusive prefix over the 256 totals yields the
            // base table. Counts and sums are commutative integer adds, so
            // any accumulation order produces the byte-identical tables.
            threadgroup atomic_uint tg_total[256];
            threadgroup atomic_uint tg_before[256];
            atomic_store_explicit(&tg_total[k], 0u, memory_order_relaxed);
            atomic_store_explicit(&tg_before[k], 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Split at the before-limit boundary so the tail segment
            // carries no branch; identical counters, identical adds.
            uint before_limit = t * TILE;
            uint idx = k;
            for (; idx < before_limit; idx += 256) {
                uint key = keys[idx];
                atomic_fetch_add_explicit(
                    &tg_total[key], 1u, memory_order_relaxed);
                atomic_fetch_add_explicit(
                    &tg_before[key], 1u, memory_order_relaxed);
            }
            for (; idx < n; idx += 256) {
                atomic_fetch_add_explicit(
                    &tg_total[keys[idx]], 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint total = atomic_load_explicit(&tg_total[k], memory_order_relaxed);
            uint lane_excl = simd_prefix_exclusive_sum(total);
            threadgroup uint simd_totals[8];
            if (lane == 31) {
                simd_totals[simd_id] = lane_excl + total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint simd_base = 0;
            for (uint s = 0; s < simd_id; ++s) {
                simd_base += simd_totals[s];
            }
            // Rank base for key k in tile t: global base + earlier tiles.
            uint off = simd_base + lane_excl +
                atomic_load_explicit(&tg_before[k], memory_order_relaxed);
            // Walk this tile's slice in input order: stability by
            // construction, exactly the stock scatter's write order.
            for (uint i = 0; i < TILE; ++i) {
                uint idx = t * TILE + i;
                if (keys[idx] == k) {
                    row_order[off] = idx / M;
                    sorted_keys[off] = k;
                    inverse_order[idx] = off;
                    ++off;
                }
            }
            """,
        ensureRowContiguous: false
    )
}()

private func routeCountingSortFused(
    _ indices: MLXArray, m: Int
) -> (rowOrder: MLXArray, sortedKeys: MLXArray, inverseOrder: MLXArray)? {
    let n = indices.size
    guard routeFusedScatterEnabled, routeCountingSortEnabled,
        indices.dtype == .uint32,
        n > 0, n % routeSortTile == 0,
        m == routeFusedScatterTopK
    else { return nil }
    let tiles = n / routeSortTile
    let outputs = routeFusedScatterKernel(
        [indices],
        grid: (tiles * 256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[n], [n], [n]],
        outputDTypes: [.uint32, .uint32, .uint32]
    )
    return (outputs[0], outputs[1], outputs[2])
}

public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    if let fused = routeCountingSortFused(indices, m: m) {
        return (
            x.flattened(start: 0, end: -3)[fused.rowOrder],
            fused.sortedKeys,
            fused.inverseOrder
        )
    }
    let order = routeCountingSort(indices) ?? argSort(indices)
    let inverseOrder: MLXArray
    if inversePermutationScatterEnabled && order.size > 0 {
        inverseOrder = inversePermutationScatterKernel(
            [order],
            grid: (order.size, 1, 1),
            threadGroup: (min(order.size, 256), 1, 1),
            outputShapes: [[order.size]],
            outputDTypes: [.uint32]
        )[0]
    } else {
        inverseOrder = argSort(order)
    }

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}


public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear?
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear?
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: SwitchLinear?
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    let isSiluActivation: Bool
    let isGeluActivation: Bool

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false,
        fuseGateUp: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = MLXNN.silu
        self.activationProduct = compiledSiluProduct
        self.isSiluActivation = true
        self.isGeluActivation = false

        if fuseGateUp {
            self._gateUpProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims * 2, numExperts: numExperts, bias: bias)
        } else {
            self._gateProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
            self._upProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        }
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray,
        bias: Bool = false,
        fuseGateUp: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.activationProduct = nil
        let probe = MLXArray([Float(1.0)])
        let probeOut = activation(probe)
        let detectedSilu = (probeOut .== MLXNN.silu(probe)).all().item(Bool.self)
        self.isSiluActivation = detectedSilu
        self.isGeluActivation =
            !detectedSilu && (probeOut .== safeGeluApproximate(probe)).all().item(Bool.self)

        if fuseGateUp {
            self._gateUpProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims * 2, numExperts: numExperts, bias: bias)
        } else {
            self._gateProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
            self._upProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        }
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let xGate: MLXArray
        let xUp: MLXArray
        if let gateUpProj {
            let xGateUp = gateUpProj(x, idx, sortedIndices: doSort)
            xGate = xGateUp[.ellipsis, ..<hiddenDims]
            xUp = xGateUp[.ellipsis, hiddenDims...]
        } else {
            guard let gateProj, let upProj else {
                fatalError("SwitchGLU requires either gate_up_proj or gate_proj/up_proj")
            }
            xUp = upProj(x, idx, sortedIndices: doSort)
            xGate = gateProj(x, idx, sortedIndices: doSort)
        }

        let activated: MLXArray
        if let activationProduct {
            activated = activationProduct(xGate, xUp)
        } else if isSiluActivation {
            activated = compiledSwiGLU(xGate, xUp)
        } else if isGeluActivation {
            activated = compiledGeGLU(xGate, xUp)
        } else {
            activated = activation(xGate) * xUp
        }

        x = downProj(activated, idx, sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        var result = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }
}
