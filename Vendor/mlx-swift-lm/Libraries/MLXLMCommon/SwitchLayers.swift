import Foundation
import MLX
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

public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
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

// MARK: - Direct final-row stores for sorted expert gather-QMMs
//
// `gatherSort`/`scatterUnsort` above round-trip a sorted gather-QMM through
// TWO argSorts: `order = argSort(indices)` to build the sorted row order,
// then `inverseOrder = argSort(order)` purely to undo it, followed by a full
// `x[invOrder]` gather (`scatterUnsort`) that copies the whole
// `[M, hiddenDims]` output back into original-slot order. For Laguna's
// down-projection, the expert-aligned gather-QMM kernel
// (`fp_gather_qmm_rhs_expert_nax` in `fp_quantized_nax.cpp`) already visits
// every output row exactly once; the only thing missing is knowing, at
// store time, which original slot each sorted row came from.
//
// `gatherSortKeyed` supplies that by folding each row's original flattened
// routing slot into the low 20 bits of the sort key (expert id in the high
// bits), so a single `argSort` over the combined key both sorts by expert
// AND carries the slot needed to store directly to it -- no second argSort,
// no `inverseOrder`. `laguna_sorted_lower_bound` in `fp_quantized_nax.cpp`
// decodes the expert id back out (`>> 20`) for its run-boundary binary
// search, and the down-projection's store epilogue decodes the slot
// (`& 0xFFFFF`) to pick its device row directly. See
// `darkbloom_direct_final_row_stores_enabled` in `quantized.cpp` for the
// C++-side gate and `DARKBLOOM_DIRECT_FINAL_ROW_STORES` in the top-level
// CLAUDE.md-style module notes.
//
// SAFETY CONTRACT (read before adding a new sorted-gather-QMM call site):
// the C++ side turns this on for ANY call that reaches the expert-aligned
// kernel at Laguna's down-projection shape (`K == 512 && N == 2048`,
// `laguna_moe_shape`'s second leg) -- it has no way to know which Swift
// call produced the indices it was given. That is only safe because EVERY
// call in this editable surface that can reach that shape (this file's
// `SwitchGLU.callAsFunction` below, and
// `Sources/MLXFastModel/LagunaRuntimeModel.swift`'s
// `lagunaFusedSortedRoutedGateUp`) is updated in lockstep here to encode
// its indices and skip `scatterUnsort` under the exact same eligibility
// check (`lagunaDirectFinalRowStoreEligible`). If a future change adds
// another down-projection-shaped sorted gather-QMM call anywhere in the
// editable surface, it MUST go through this same helper pair or explicitly
// opt out -- passing plain indices to a call that happens to match the
// shape while `DARKBLOOM_DIRECT_FINAL_ROW_STORES` is on will silently
// scatter output rows to wrong addresses.
//
// `DARKBLOOM_DIRECT_GATHER_LOADS` (`lagunaDirectGatherLoadEligible` below)
// is the symmetric input-side lever for the gate/up leg, with a NARROWER
// safety contract by construction: the shape it keys on (`K == 2048 &&
// N == 1024`, `laguna_moe_shape`'s FIRST leg) is only ever reached by
// `lagunaFusedSortedRoutedGateUp`'s fused-bank gate/up call -- `SwitchGLU`'s
// stock `gate_proj`/`up_proj` are separate banks at `N == moeIntermediateSize`
// each and never reach this shape (Laguna's checkpoint does not use
// `fuseGateUp`) -- so `SwitchGLU.callAsFunction` needs no matching change
// for loads the way it needed one for stores.

/// `DARKBLOOM_DIRECT_FINAL_ROW_STORES` (default on; set "0" to fully revert
/// every caller below to the stock `gatherSort`/`scatterUnsort` round trip).
public let lagunaDirectFinalRowStoresEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_DIRECT_FINAL_ROW_STORES"] != "0"

/// Mirrors `darkbloom_expert_aligned_gather()` in `quantized.cpp`: direct
/// final-row stores only ever apply on top of the expert-aligned kernel, so
/// if that kernel itself is ablated off, fall back to the stock path too.
private let lagunaExpertAlignedGatherStageEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_EXPERT_ALIGNED_GATHER"] != "0"

/// Mirrors `darkbloom_stage_bm128_variant() == 4` (unset, or explicitly
/// "4") in `quantized.cpp` -- the only tiling (`BM=64, WM=4, WN=2`) the
/// expert-aligned kernel's `expert_aligned` gate accepts. Any other
/// `DARKBLOOM_STAGE_BM128` value routes the C++ dispatch to the generic,
/// non-expert-aligned gather-QMM kernel for every shape, so this mirror
/// must stay in lockstep with that gate.
private let lagunaStageBM128IsDefaultTiling: Bool = {
    guard let value = ProcessInfo.processInfo.environment["DARKBLOOM_STAGE_BM128"] else {
        return true
    }
    return value.isEmpty || value == "4"
}()

/// Mirrors `metal::is_nax_available()`
/// (`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/device.cpp:913-932`),
/// the FIRST gate `gather_qmm_rhs` (`quantized.cpp:1708`) checks before ever
/// entering `gather_qmm_rhs_nax` -- and therefore before `expert_aligned` is
/// even computed. `fp_gather_qmm_rhs_expert_nax` is an M5-generation ("NAX")
/// kernel; on a device where this is false (confirmed on an M3 Max: local
/// correctness diverges from both the clean tree AND itself across
/// `DARKBLOOM_DIRECT_FINAL_ROW_STORES` on/off -- "coherent but different"
/// logits, not a near-tie reshuffle), `gather_qmm_rhs` falls through to the
/// plain (non-NAX) `_gather_qmm_rhs_nt_`/`_gather_qmm_rhs_nn_` kernel family
/// in this same file, which reads every `indices[...]` entry directly as a
/// plain expert id for weight-pointer arithmetic
/// (`wl + index * stride_w`-style addressing in that kernel's Metal source)
/// -- an encoded `(expertId << 20) | slot` value there is wrong by roughly
/// 2^20x, corrupting weight selection outright, with no corresponding
/// decode and no scatterUnsort to undo the skipped unsort. This predicate
/// makes the WHOLE lever inert (falls back to the fully stock path) on any
/// device where that kernel is not the one actually executing.
///
/// Parses `GPU.deviceInfo().architecture`
/// (`Vendor/mlx-swift/Source/MLX/GPU+Metal.swift`, `MTLDevice.architecture.name`)
/// with the IDENTICAL two-digit-generation extraction `Device::Device()`
/// uses (`device.cpp:564-572`: the two characters immediately before the
/// last one, each clamped to 0 if not an ASCII digit) and the identical
/// `arch.back() == 'p' ? 18 : 17` generation floor
/// (`device.cpp:919-927`), gated on the same `macOS 26.2` availability
/// check. `MTLDevice.architecture.name` and the Metal-cpp
/// `device_->architecture()->name()` the C++ side reads are the same
/// underlying Objective-C property, so this is an exact mirror for Laguna's
/// call pattern, not merely a conservative approximation: the OTHER two
/// conjuncts in `gather_qmm_rhs`'s NAX gate (`transpose_` and
/// `x.dtype() != float32`) are both structurally always-true for every
/// caller in this file (Laguna's routed-expert weights are always
/// `transpose: true`, and `x`/`activated` are always bfloat16), so
/// `is_nax_available()` is the ONLY variable term left, and this reproduces
/// it exactly rather than bounding it. The one acknowledged gap: a debug
/// override via `MLX_METAL_GPU_ARCH`-style env vars
/// (`env::metal_gpu_arch()`, read only on the C++ side with no Swift
/// mirror) is not tracked here; the ranked runner sets no such override
/// (CLAUDE.md: "the ranked runner sets no DARKBLOOM_* variables", and this
/// override predates and is unrelated to that family), so this is a
/// debug-only residual, not a ranked-path risk.
private let lagunaIsNAXAvailable: Bool = {
    guard #available(macOS 26.2, *) else {
        return false
    }
    let arch = GPU.deviceInfo().architecture
    let chars = Array(arch)
    guard let lastChar = chars.last, chars.count >= 3 else {
        return false
    }
    func asciiDigit(_ c: Character) -> Int {
        guard let a = c.asciiValue, a >= 48, a <= 57 else {
            return 0
        }
        return Int(a) - 48
    }
    let tens = asciiDigit(chars[chars.count - 3])
    let ones = asciiDigit(chars[chars.count - 2])
    let generation = tens * 10 + ones
    return generation >= (lastChar == "p" ? 18 : 17)
}()

/// True exactly when a sorted down-projection gather-QMM built from `down`
/// over `m` rows will land on `fp_gather_qmm_rhs_expert_nax` -- i.e. every
/// condition `gather_qmm_rhs_nax` checks before setting `expert_aligned`,
/// PLUS the two outer gates `gather_qmm_rhs`'s caller (`GatherQMM::eval_gpu`,
/// `quantized.cpp:1933`) checks before ever calling `gather_qmm_rhs` at all:
/// `metal::is_nax_available()` (`lagunaIsNAXAvailable` above) and
/// `B / E >= 4` (`quantized.cpp:1961`, `B` the sorted-row count = `m` here
/// since `x.shape(-2) == 1` for every caller in this file, `E == 256` fixed
/// routed experts for Laguna -- i.e. `m >= 1024`; the frozen 512-token
/// prefill window always has `m == 4096` and clears this trivially, but
/// `doSort`'s own threshold is `m >= 64`, so a hidden gate with a shorter
/// multi-token prompt -- 8 to 127 tokens -- could hit `doSort == true` while
/// failing `B / E >= 4`, routing `eval_gpu` to the generic `gather_qmm`
/// matrix-matrix path instead of `gather_qmm_rhs_nax` and bypassing
/// `expert_aligned` entirely; unguarded, the fused runtime path would still
/// encode indices and skip `scatterUnsort` for a call that never reaches
/// the kernel that understands either).
///
/// Mirrored from the Swift side for Laguna's fixed down-projection shape
/// (`K == 512` == `moeIntermediateSize`, `N == 2048` == `hiddenSize`, the
/// second leg of `laguna_moe_shape`). `align_N`/`align_K` are not mirrored
/// here because they are unconditionally true for this fixed shape (2048
/// and 512 are both multiples of the kernel's fixed `bn = bk = 64`
/// regardless of the `DARKBLOOM_STAGE_BM128` tiling choice).
public func lagunaDirectFinalRowStoreEligible(down: QuantizedSwitchLinear, m: Int) -> Bool {
    lagunaDirectFinalRowStoresEnabled
        && lagunaExpertAlignedGatherStageEnabled
        && lagunaStageBM128IsDefaultTiling
        && lagunaIsNAXAvailable
        && m >= 64
        && m >= 1024
        && m < (1 << 20)
        && down.mode == .nvfp4
        && down.groupSize == 16
        && down.bits == 4
        && down.bias == nil
        && down.inputDims == 512
        && down.outputDims == 2048
}

/// `DARKBLOOM_DIRECT_GATHER_LOADS` (default on; the store-side sibling
/// `DARKBLOOM_DIRECT_FINAL_ROW_STORES` is ranked-validated at 307360f5, and
/// this flag mirrors its `!= "0"` semantics; set "0" to fully revert to
/// physically gathering x for the gate/up leg). Symmetric, input-side
/// counterpart to `DARKBLOOM_DIRECT_FINAL_ROW_STORES` -- see
/// `darkbloom_direct_gather_loads_enabled()` in `quantized.cpp` and the
/// `kDirectLoadTokenShift` load epilogue in `fp_quantized_nax.cpp`.
/// Independent of `lagunaDirectFinalRowStoresEnabled`: `lagunaFusedSortedRoutedGateUp`
/// handles all four on/off combinations explicitly (see its doc comment).
///
/// This flag alone is still all-or-nothing (every sparse layer's gate/up
/// leg is a candidate). The PARTIAL-strength dial that chunks a full-loads
/// prefill gain across submissions -- needed because full-strength direct
/// loads plausibly overshoots the paired prefill acceptance band on its own
/// (re-reading the gate/up A-tile once per N-tile, BN=64 over N=1024 -> 16
/// re-reads, cuts O(10 GB) of DRAM traffic per prefill pass) -- lives
/// entirely in `LagunaRuntimeModel.swift`
/// (`DARKBLOOM_DIRECT_LOADS_LAYER_PCT`, `lagunaDirectLoadsLayerSelected`),
/// downstream of this flag and `lagunaDirectGatherLoadEligible` below: a
/// per-layer dial selects WHICH eligible layers actually take the
/// direct-load addressing, using the exact same eligibility this function
/// establishes. See that file for the worked selection and the per-layer
/// dataflow table (this file's `lagunaDirectGatherLoadEligible` and
/// `lagunaFusedSortedRoutedGateUp`'s "four combinations" doc comment both
/// stay layer-agnostic on purpose: they describe when a layer's ENCODING is
/// eligible, not when its ADDRESSING is selected).
public let lagunaDirectGatherLoadsEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_DIRECT_GATHER_LOADS"] != "0"

/// True exactly when a sorted gate/up gather-QMM at Laguna's fused-bank
/// shape (`k`, `n`) over `m` sorted rows will land on
/// `fp_gather_qmm_rhs_expert_nax` AND is safe to feed with raw,
/// un-physically-gathered x plus an explicit (dummy-valued) `lhsIndices` --
/// i.e. every condition `GatherQMM::eval_gpu`'s `direct_gather_loads_capable`
/// checks (`quantized.cpp`, right after the `right_sorted_ == true` branch),
/// mirrored from the Swift side. `mode`/`groupSize`/`bits`/bias-free are not
/// re-checked here (unlike `lagunaDirectFinalRowStoreEligible`, which reads
/// them off a live `QuantizedSwitchLinear`): the only caller of this
/// function, `lagunaFusedSortedRoutedGateUp`, is only ever reached after
/// `prepareFusedRoutedGateUp` (`LagunaRuntimeModel.swift`) has already
/// verified the fused bank is bias-free NVFP4 group-16 4-bit, so those are
/// invariants of this call site rather than runtime conditions to re-derive.
/// `k`/`n` ARE still checked explicitly against the fixed shape
/// (`k == hiddenSize == 2048`, `n == 2 * moeIntermediateSize == 1024`, the
/// first leg of `laguna_moe_shape`) so this predicate stays self-contained
/// and does not silently pass for a differently-shaped future caller.
public func lagunaDirectGatherLoadEligible(m: Int, k: Int, n: Int) -> Bool {
    lagunaDirectGatherLoadsEnabled
        && lagunaExpertAlignedGatherStageEnabled
        && lagunaStageBM128IsDefaultTiling
        && lagunaIsNAXAvailable
        && m >= 64
        && m >= 1024
        && m < (1 << 20)
        // Post-mortem shape restriction (ranked no-scores c2e292c7 and
        // 835b22f7, both isolated to the direct-loads route): only the
        // exact frozen-window shape -- 512 tokens x 8 routed experts, the
        // one shape the timed benchmark and the hidden 512-token
        // teacher-forced base case both exercise -- takes the direct-load
        // path. Every other prefill length (hidden anchor / GPQA / free-run
        // prompts, non-full chunks) keeps the ranked-proven physical-gather
        // path, removing all odd-shape exposure (tail-chunk masking, run
        // fragmentation at unusual M) from the lever while it is being
        // ranked-qualified. Loosen deliberately, if ever, with ranked
        // evidence.
        && m == 4096
        && k == 2048
        && n == 1024
}

/// Keyed counterpart to `gatherSort`: folds each row's original flattened
/// routing slot (its position in the pre-sort `[0, indices.size)` range)
/// into the low 20 bits of the sort key, expert id in the high bits, and
/// returns the sorted key array in place of `gatherSort`'s separate
/// `(idx, inverseOrder)` pair -- there is no second `argSort` to compute.
///
/// Exactness vs. `gatherSort`: `argSort(indices)` and `argSort(key)` produce
/// the identical row order. `key` is a strict total order (no two rows
/// share a key, since the low 20 bits are a distinct rank per row), so its
/// `argSort` is deterministic independent of any sort-stability guarantee;
/// `gatherSort`'s plain `argSort(indices)` is ALSO already stable on MLX's
/// Metal backend (`ArgPartition`/`argSort` both route through the same
/// stable merge sort -- see the comment on `laguna_router_key_before` in
/// `LagunaRuntimeModel.swift`), so the two orders coincide row-for-row.
public func gatherSortKeyed(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let flatIndices = indices.flattened().asType(.uint32)
    let total = flatIndices.size
    precondition(
        total < (1 << 20),
        "gatherSortKeyed: \(total) routing slots exceeds the 20-bit slot encoding"
    )
    let slot = MLXArray.arange(total, dtype: .uint32)
    let key = (flatIndices << 20) | slot
    let order = argSort(key)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        key[order]
    )
}

/// Companion to `scatterUnsort` for the direct-final-row-store path: the
/// down-projection kernel already wrote every row to its final (pre-sort)
/// slot, so restoring the logical `[..., numExpertsPerTok, hiddenDims]`
/// shape is a pure view reshape -- no gather, no copy.
public func unflattenDirectStored(x: MLXArray, shape: [Int]) -> MLXArray {
    unflatten(x, axis: 0, shape: shape)
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

        // `idx` feeds gate_proj/up_proj (and gate_up_proj), which for
        // Laguna's checkpoint (separate, bias-free NVFP4 gate_proj/up_proj
        // banks at N == 512 each) never matches `laguna_moe_shape` and so
        // never reaches the expert-aligned kernel -- it MUST stay plain
        // (undecoded) expert ids regardless of `directStored` below, or the
        // generic gather-QMM kernel's `index * stride_w` weight-offset
        // arithmetic corrupts on an encoded value. `downIdx` feeds only
        // down_proj, and is the encoded key when the direct-final-row-store
        // fast path is taken (see the "Direct final-row stores" section
        // above `gatherSort`).
        var idx = indices
        var downIdx = indices
        var inverseOrder = MLXArray()
        var directStored = false

        if doSort {
            if let downQuantized = downProj as? QuantizedSwitchLinear,
                lagunaDirectFinalRowStoreEligible(down: downQuantized, m: indices.size)
            {
                let (sortedX, key) = gatherSortKeyed(x: x, indices: indices)
                x = sortedX
                idx = key >> 20
                downIdx = key
                directStored = true
            } else {
                (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
                downIdx = idx
            }
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

        x = downProj(activated, downIdx, sortedIndices: doSort)

        if doSort {
            if directStored {
                x = unflattenDirectStored(x: x, shape: indices.shape)
            } else {
                x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
            }
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
