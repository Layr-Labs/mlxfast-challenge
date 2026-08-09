import Foundation
import MLX
import MLXFast
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

/// Compiled SiLU-gated product (`silu(gate) * up`) for the common MoE GLU path.
/// Fusing activation + product into one compiled, shapeless kernel cuts kernel
/// dispatches and intermediates on the hot decode path. Upstream ef85ed0.
///
/// Gated by `MLXHardwareInfo.isCompiledDecodeSupported` (env `MLX_COMPILED_DECODE`,
/// default on) like the sibling `compiledSwiGLU` / `safeGeluApproximate` fusions.
/// The default SiLU `SwitchGLU` path wires this in as `activationProduct` (the
/// highest-precedence branch in `callAsFunction`) and `LFM2MoE` calls it directly,
/// so without the gate both would keep hitting compiled kernels on the very M1/M2 +
/// macOS Tahoe machines the opt-out (MLX #3329) is meant to protect. Falls back to
/// the plain uncompiled closure when off; the default (env unset) stays compiled.
public let compiledSiluProduct: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { gate, up in
        MLXNN.silu(gate) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Compiled weighted expert-output combine (`(outputs * weights[..., None]).sum(-2)`).
/// Shared by MoE routers (e.g. Gemma 4) to fuse the scale + reduce. Upstream ef85ed0.
public let weightedExpertSum: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { outputs, weights in
    (outputs * MLX.expandedDimensions(weights, axis: -1)).sum(axis: -2)
}

// MARK: - Compiled activation fusions (vMLX / osaurus-main port)

/// Approximate (tanh) GELU written with `x * x * x` instead of the Power
/// primitive (`x ** 3`). The Power primitive returns zero results under the
/// macOS Tahoe Metal JIT (MLX #3329), so the explicit multiplies keep it safe
/// under `compile(shapeless: true)`. Numerically identical to
/// `MLXNN.geluApproximate`.
///
/// Gated by `MLXHardwareInfo.isCompiledDecodeSupported` (env `MLX_COMPILED_DECODE`,
/// default on); falls back to the plain closure when compiled fusions are off.
public let safeGeluApproximate: @Sendable (MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray) -> MLXArray = { (x: MLXArray) -> MLXArray in
        0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Drop-in replacement for `MLXNN.GELU(approximation: .tanh)` that avoids the
/// Power primitive crash. Use anywhere a tanh-approx GELU unary layer is needed.
public class SafeGELU: Module, UnaryLayer {
    public override init() { super.init() }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        safeGeluApproximate(x)
    }
}

/// Compiled SiLU-gated GLU product (`silu(gate) * up`). Same math as
/// `compiledSiluProduct` above, but gated by `MLXHardwareInfo` so M1/M2 + macOS
/// Tahoe can opt out. Used by `SwitchGLU` when a SiLU activation is supplied via
/// the custom-activation initializer (where `activationProduct` is nil).
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

/// Compiled GELU-gated GLU product (`geluApprox(gate) * up`), fusing the tanh
/// GELU and the element-wise multiply into one shapeless kernel. Uses the
/// Power-free `x * x * x` GELU so it is safe under `compile(shapeless: true)`.
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

private let routeSortTile = 128

/// Fused stable counting sort that directly emits every downstream index
/// product of the sorted-MoE chain: `idx / m` is the gathered row, the tested
/// key `k` is the sorted index, and `off` is the inverse permutation entry.
/// DEFAULT ON; `DARKBLOOM_ROUTE_FUSED_SCATTER=0` restores the generic chain.
private let routeFusedScatterEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ROUTE_FUSED_SCATTER"] != "0"

private let routeFusedScatterTopK = 8

private let routeFusedScatterKernel: MLXFast.MLXFastKernel = {
    let m = routeFusedScatterTopK
    return MLXFast.metalKernel(
        name: "mlx_lm_route_csort_scatter_fused_m\(m)_u32_v3",
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
            // One cooperative pass counts every key (totals) and every key
            // in earlier tiles (before), then a simd exclusive prefix over
            // the 256 totals yields the base table. Counts and sums are
            // commutative, so accumulation order cannot change the tables.
            threadgroup atomic_uint tg_total[256];
            threadgroup atomic_uint tg_before[256];
            atomic_store_explicit(&tg_total[k], 0u, memory_order_relaxed);
            atomic_store_explicit(&tg_before[k], 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint before_limit = t * TILE;
            for (uint idx = k; idx < n; idx += 256) {
                uint key = keys[idx];
                atomic_fetch_add_explicit(
                    &tg_total[key], 1u, memory_order_relaxed);
                if (idx < before_limit) {
                    atomic_fetch_add_explicit(
                        &tg_before[key], 1u, memory_order_relaxed);
                }
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
            // Walk this tile's slice in input order for stable ties.
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
    guard routeFusedScatterEnabled,
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
    let order = argSort(indices)
    let inverseOrder = argSort(order)

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

// MARK: - SwitchGLU

public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear?
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear?
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: SwitchLinear?
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    /// Optional fused (activation * up) kernel. Set for the default SiLU path so
    /// the GLU product runs as one compiled op; nil when a custom activation is
    /// supplied (we then fall back to `activation(gate) * up`). Upstream ef85ed0.
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    /// Activation-type flags detected once at init from a tiny test input (vMLX
    /// approach — no per-token check). Only consulted when `activationProduct` is
    /// nil (the custom-activation path): they let SiLU/GELU custom activations use
    /// the compiled `compiledSwiGLU` / `compiledGeGLU` fusions instead of the
    /// uncompiled `activation(gate) * up`. On any mismatch we fall back to that
    /// exact uncompiled path, so detection only ever enables a numerically
    /// equivalent fast path — it can never change results.
    let isSiluActivation: Bool
    let isGeluActivation: Bool

    /// Default SiLU GLU path -- uses the compiled fused (silu * up) kernel.
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
        // Default path is SiLU and `activationProduct` is non-nil, so these are
        // not consulted on the hot path; set them accurately for completeness
        // (and to avoid a needless probe eval at load for every MoE layer).
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

    /// Custom-activation GLU path -- runs `activation(gate) * up` uncompiled.
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
        // Detect SiLU/GELU once via a tiny test input (vMLX approach) so the hot
        // path can select the compiled fusion without a per-token check. Exact
        // equality is intentional: a match means the supplied closure computes
        // that exact function; any non-match falls back to `activation(gate) * up`
        // in callAsFunction, so this can only ever enable an equivalent fast path.
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
            // Pre-fused gate_up_proj weight from checkpoint — one gathered
            // matmul via the polymorphic SwitchLinear call, then split.
            let xGateUp = gateUpProj(x, idx, sortedIndices: doSort)
            xGate = xGateUp[.ellipsis, ..<hiddenDims]
            xUp = xGateUp[.ellipsis, hiddenDims...]
        } else {
            // Separate gate_proj / up_proj checkpoints — two gathered matmuls.
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

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
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
