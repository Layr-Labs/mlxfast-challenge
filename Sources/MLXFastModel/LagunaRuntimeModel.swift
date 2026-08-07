import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// Correctness-first Laguna XS 2.1 runtime, behavior-checked against the
// vendored reference implementation and specialized by guarded fast paths.

func lagunaLastTokenRange(sequenceLength: Int) -> Range<Int>? {
    sequenceLength > 1 ? (sequenceLength - 1)..<sequenceLength : nil
}

func lagunaLastTokenHidden(_ hidden: MLXArray) -> MLXArray {
    guard let range = lagunaLastTokenRange(sequenceLength: hidden.dim(1)) else {
        return hidden
    }
    return hidden[0..., range, 0...]
}

// MARK: - Allocation-free shape checks
//
// `MLXArray.shape` builds a fresh Swift `[Int]` and comparing it against a
// dimension literal builds a second one, so each such guard on the decode path
// costs two heap allocations. `ndim` and `shapeN` are direct C accessors that
// allocate nothing, and the `ndim` test short-circuits `shapeN`'s
// dimensionality precondition.

extension MLXArray {
    /// True when the shape is exactly the listed dimensions.
    @inline(__always) func dims(_ d0: Int) -> Bool { ndim == 1 && dim(0) == d0 }
    @inline(__always) func dims(_ d0: Int, _ d1: Int) -> Bool { ndim == 2 && shape2 == (d0, d1) }
    @inline(__always) func dims(_ d0: Int, _ d1: Int, _ d2: Int) -> Bool {
        ndim == 3 && shape3 == (d0, d1, d2)
    }
    @inline(__always) func dims(_ d0: Int, _ d1: Int, _ d2: Int, _ d3: Int) -> Bool {
        ndim == 4 && shape4 == (d0, d1, d2, d3)
    }
    @inline(__always) func dims(_ d0: Int, _ d1: Int, _ d2: Int, _ d3: Int, _ d4: Int) -> Bool {
        ndim == 5 && dim(0) == d0 && dim(1) == d1 && dim(2) == d2 && dim(3) == d3 && dim(4) == d4
    }

    /// True when `other` has an identical shape.
    @inline(__always) func sameDims(_ other: MLXArray) -> Bool {
        guard ndim == other.ndim else { return false }
        for axis in 0..<ndim where dim(axis) != other.dim(axis) { return false }
        return true
    }
}

/// Builds the `initializeRope` scaling dictionary for a per-type Laguna RoPE
/// spec. For `default` RoPE only the type is consulted; for YaRN the factory
/// reads factor / original context / betas. The XS config also serializes
/// `attention_factor: 1.0`, but both vendored MLX Laguna implementations
/// intentionally ignore that Hugging Face field. Do not forward it here:
/// leaving MLX's mscale/mscale_all_dim defaults at 1.0/0.0 yields the upstream
/// attention scaling of `0.1 * ln(32) + 1` (~1.34657).
func lagunaRopeScalingConfig(_ spec: LagunaRopeSpec) -> [String: StringOrNumber] {
    var scalingConfig: [String: StringOrNumber] = ["rope_type": .string(spec.type)]
    if spec.type == "yarn" {
        scalingConfig["factor"] = .float(Float(spec.factor))
        scalingConfig["original_max_position_embeddings"] = .int(
            spec.originalMaxPositionEmbeddings)
        scalingConfig["beta_fast"] = .float(Float(spec.betaFast))
        scalingConfig["beta_slow"] = .float(Float(spec.betaSlow))
    }
    return scalingConfig
}

/// `DARKBLOOM_TRACE_FUSION=1` prints one stderr line the first time each fused
/// decode path is taken. Every fusion here is guarded on dtype, rank, exact
/// shape and module identity and falls back silently when a guard declines, so
/// a change that quietly stops firing looks exactly like a change that does
/// nothing. This makes "did it actually run" observable without a debugger.
private let lagunaTraceFusion =
    ProcessInfo.processInfo.environment["DARKBLOOM_TRACE_FUSION"] == "1"
private let lagunaTracedFusions = LagunaFusionTraceLog()

final class LagunaFusionTraceLog: @unchecked Sendable {
    private var seen: Set<String> = []
    private let lock = NSLock()

    func note(_ site: String) {
        lock.lock()
        let isNew = seen.insert(site).inserted
        lock.unlock()
        if isNew {
            FileHandle.standardError.write(Data("mlxfast: fusion active: \(site)\n".utf8))
        }
    }
}

@inline(__always)
func lagunaTrace(_ site: @autoclosure () -> String) {
    guard lagunaTraceFusion else { return }
    lagunaTracedFusions.note(site())
}

// MARK: - Runtime fusion feature flags

// Each fusion below concatenates the OUTPUT ROWS of same-dtype projections
// that consume the same input. Per-row gemv/qmv/gather-qmv arithmetic is
// independent of which rows share a dispatch (every output row keeps its own
// K-loop and scale application in the original order), so the fused dispatch
// is bit-exact against the separate dispatches it replaces. The per-head
// g_proj (N=64) uses a different split-K gemv variant and is never fused.

/// `DARKBLOOM_FUSED_QKV` (default ON; set "0" to disable): after checkpoint
/// load, retain one row-concatenated `[Wq; Wk; Wv]` BF16 weight per attention
/// layer and serve Q/K/V from a single projection dispatch. Bit-exact for
/// bias-free `Linear` projections; prefill-only (L > 1).
let lagunaFusedQKVEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_QKV"] != "0"

/// `DARKBLOOM_FUSED_SHARED_GATE_UP` (default on; set "0" to disable): after
/// checkpoint load, retain one row-concatenated NVFP4 `[gate; up]` bank per
/// shared expert and serve single-token decode from one quantized matmul.
/// Multi-token prefill remains on the stock separate banks so the ranked
/// prefill path and its smaller gather/GEMM shapes are unchanged.
let lagunaFusedSharedGateUpEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_SHARED_GATE_UP"] != "0"

/// Decode-only shared-expert NVFP4 QMV + SwiGLU fusion. This consumes the
/// retained row-concatenated `[gate; up]` bank and emits only the 512-wide
/// BF16 activation, preserving the two independent QMV casts and every BF16
/// boundary in the compiled SiLU product.
let lagunaFusedSharedSwiGLUQMVEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_SHARED_SWIGLU_QMV"] != "0"

/// Decode-only shared-expert down QMV plus both sparse-block residual adds.
/// The kernel preserves the stock BF16 down-projection result, the inner
/// `routed + shared` rounding, and the outer `h + r2` rounding while avoiding
/// the intermediate shared/r2 materializations and the final elementwise
/// dispatch.
let lagunaFusedSharedDownResidualEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_SHARED_DOWN_RESIDUAL"] != "0"

/// Higher-fusion decode path: the eight routed down projections and the
/// shared down projection share one 288-thread dispatch, which also performs
/// the exact router reduction, routed scale, and both BF16 residual adds.
let lagunaFusedRoutedSharedDownResidualEnabled =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_FUSED_ROUTED_SHARED_DOWN_RESIDUAL"] != "0"

/// Routed-expert counterpart to the shared QMV + SwiGLU fusion. Each decode
/// request supplies exactly eight current-token expert indices; the kernel
/// reads those banks directly and emits `[1, 1, 8, 1, 512]`.
let lagunaFusedRoutedSwiGLUQMVEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_ROUTED_SWIGLU_QMV"] != "0"

/// `DARKBLOOM_PACKED_SCALES` (default ON; set "0" to disable): decode-only
/// scale-interleaved side copy of the fused routed gate/up NVFP4 bank. The
/// stock `lagunaRoutedSwiGLUQMV` reads codes and E4M3 scales from two separate
/// tensors (four device streams per simdgroup iteration: gate codes, up
/// codes, gate scales, up scales). The packed side bank stores only the 32
/// scale bytes for each row and K block, in the kernel's exact walk order
/// `[expert][tile 128][k-block 4][row-pair sub 8]`; the resident fused code
/// bank is reused directly. Load widths, dequant expressions, accumulation
/// order, and every BF16 boundary are identical to the stock kernel — only
/// scale address computation changes, so the packed dispatch is bit-exact
/// (class A).
/// Memory: +~32 MB resident per sparse layer while enabled (the stock fused
/// code bank stays resident for prefill and fallback paths).
let lagunaPackedScalesEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PACKED_SCALES"] != "0"

/// The fused residual+RMSNorm+router producer kernel formerly emitted a 4th
/// output buffer (`router_keys`, 256-uint32 = 1KB per sparse layer per decode
/// step). No kernel ever consumed `router_keys` as input — it was only used as
/// a nil-check guard to select the R1-halved gate/up dispatch branch. The keys
/// output is now eliminated: the v2 kernel variant (no keys) is always used,
/// saving ~5MB of write traffic and 5,120 allocations per decode window.

/// One-shot stderr visibility for the packed-scales arm: with the flag set,
/// the arm MUST announce either "active" (bank built / packed dispatch taken)
/// or "inactive" (a guard declined and the stock kernel ran instead), so a
/// silently-declining guard can never measure its own control.
final class LagunaPackedScalesLog: @unchecked Sendable {
    private var seen: Set<String> = []
    private let lock = NSLock()

    func note(_ state: String, _ site: String) {
        lock.lock()
        let isNew = seen.insert(site).inserted
        lock.unlock()
        if isNew {
            FileHandle.standardError.write(
                Data("mlxfast: packed-scales \(state): \(site)\n".utf8))
        }
    }
}

let lagunaPackedScalesLog = LagunaPackedScalesLog()

/// Decode-only routed NVFP4 down-QMV plus BF16 router weighting, fixed-order
/// expert reduction, and the Laguna 2.5 routed scale. The custom kernel emits
/// one 2048-wide branch instead of materializing eight expert rows.
let lagunaFusedRoutedDownReduceEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_ROUTED_DOWN_REDUCE"] != "0"

/// `DARKBLOOM_FUSED_ROUTED_GATE_UP` (default on; set "0" to disable): after
/// checkpoint load, retain one row-concatenated NVFP4 `[gate; up]` bank per
/// sparse layer's routed experts and serve single-token decode's gate/up from
/// one gather-QMM dispatch. DECODE-ONLY: the module tree, checkpoint keys,
/// and every multi-token (prefill) forward stay fully stock -- ablation
/// showed the fused bank helps decode (~+1.9%) but badly hurts the M=512
/// sorted gather-GEMM prefill path, so prefill always dispatches the stock
/// separate banks.
///
/// That prefill finding pre-dates RUNSKIP. See
/// `DARKBLOOM_PREFILL_FUSED_GATE_UP` immediately below for the current,
/// separately-flagged, post-RUNSKIP re-measurement of the same fusion idea
/// applied to the sorted prefill path -- this flag and its history are left
/// as-is (decode-only) rather than folded together, so each can be ablated
/// independently.
let lagunaFusedRoutedGateUpEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_ROUTED_GATE_UP"] != "0"

/// Multi-token sorted gather-GEMM counterpart to the decode gate/up fusion;
/// independently ablatable and row-arithmetic-identical to separate banks.
let lagunaPrefillFusedRoutedGateUpEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_FUSED_GATE_UP"] != "0"

let lagunaPrefillFusedRoutedGateUpHalvedEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_FUSED_GATE_UP_HALVED"] != "0"

func lagunaNAXAvailable(architecture: String, osSupportsNAX: Bool) -> Bool {
    guard osSupportsNAX,
        let generation = Int(architecture.suffix(3).prefix(2))
    else { return false }
    return generation >= (architecture.hasSuffix("p") ? 18 : 17)
}

func lagunaExpertAlignedStageEnabled(_ value: String?) -> Bool {
    ["", "4", "5"].contains(value ?? "")
}

func lagunaNAXGate(_ envKey: String) -> Bool {
    let e = ProcessInfo.processInfo.environment
    guard e[envKey] != "0", #available(macOS 26.2, *) else { return false }
    let a = e["MLX_METAL_GPU_ARCH"]
    return lagunaNAXAvailable(architecture: a.flatMap { $0.isEmpty ? nil : $0 } ?? GPU.deviceInfo().architecture, osSupportsNAX: true)
}

let lagunaExpertAlignedGatherEnabled = lagunaExpertAlignedStageEnabled(ProcessInfo.processInfo.environment["DARKBLOOM_STAGE_BM128"]) && lagunaNAXGate("DARKBLOOM_EXPERT_ALIGNED_GATHER")

let lagunaPrefillSharedHalvedEnabled = lagunaNAXGate("DARKBLOOM_PREFILL_SHARED_HALVED")

/// Decode post-attention residual + RMSNorm fusion. The kernel emits
/// both the rounded BF16 residual (needed by the following skip connection)
/// and the normalized row (consumed immediately by the MLP), eliminating a
/// separate residual-add materialization/read. Restricted to the single-row
/// (`x.size == hiddenSize`) decode shape at its call site; see
/// `lagunaPrefillFusedResidualRMSNormEnabled` immediately below for the
/// multi-token counterpart, gated independently.
let lagunaFusedResidualRMSNormEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_RESIDUAL_RMS"] != "0"

/// Multi-token use of the row-general residual+RMSNorm kernel; its independent
/// row dispatch preserves the stock add and normalization order per token.
let lagunaPrefillFusedResidualRMSNormEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_FUSED_RESIDUAL_RMS"] != "0"


/// One output row per simdgroup for the default split routed gate/up decode
/// QMV. Official submission `b56a6d9` passed all 1,344 exact-token checks:
/// every row retains its K-block order and 32-lane reduction while the grid
/// exposes twice as many independent simdgroups to cover memory latency.
/// Set `DARKBLOOM_QMV_R1=0` to restore the two-row control.
let lagunaSwiGLUQMVRows1Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_QMV_R1"] != "0"

/// Folds the per-head softplus gate into the output projection's GEMV (see
/// `lagunaGatedOutputProjectionSource`), with one kernel variant per attention
/// family. Set `DARKBLOOM_FUSED_GATED_OUTPUT=0` to ablate.
let lagunaFusedGatedOutputProjectionEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_GATED_OUTPUT"] != "0"

/// Issues Q, K and V as one dispatch over the three stock weights (see
/// `lagunaFusedQKVProjectionSource`). Unlike `DARKBLOOM_FUSED_QKV` this keeps
/// no concatenated bank, so prefill is untouched. Set
/// `DARKBLOOM_FUSED_QKV_PROJECTION=0` to ablate.
let lagunaFusedQKVProjectionEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_QKV_PROJECTION"] != "0"

/// TensorFold-derived within-token batching for the serial decode stream.
/// A native group-32 affine INT8 side layout packs Q/K/V into one batched
/// quantized matmul, cutting their weight traffic without speculating future
/// tokens or changing the KV dependency. Prefill stays on the original BF16
/// projections. Two ranked chunks proved 28 layers; this final bounded chunk
/// widens the same layout to all 40 layers.
private let lagunaNativeAffineQKVLayerCount: Int = {
    guard ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_QKV"] != "0"
    else { return 0 }
    let requested = Int(
        ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_QKV_LAYERS"]
            ?? "40") ?? 40
    return min(max(requested, 0), LagunaConstants.numHiddenLayers)
}()
let lagunaNativeAffineQKVEnabled = lagunaNativeAffineQKVLayerCount > 0

/// Depth-selection mode for both native affine INT8 attention layouts.
///
/// **Default (unset or anything but `1`) is the shipped PREFIX predicate**
/// `layer < count`, so the shipped semantics are byte-for-byte what the ranked
/// chunks proved. `DARKBLOOM_NATIVE_AFFINE_SUFFIX=1` selects the LAST `count`
/// layers (`layer >= numHiddenLayers - count`) instead. It exists to measure
/// whether the quantization perturbation the argmax gate sees depends on the
/// DEPTH of the quantized site (an early layer's error has 39 more layers of
/// amplification runway than a late one's) at matched weight-traffic coverage.
private let lagunaNativeAffineSuffixSelection =
    ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_SUFFIX"] == "1"

/// Restricts a native affine layout to exactly one layer index, for
/// amplification-vs-depth probes. Unset (or out of range) keeps the normal
/// prefix/suffix coverage. The layer count must still be non-zero, so the
/// layout is prepared and dispatched exactly as it would be in a ranked run.
private func lagunaNativeAffineOnlyLayer(_ key: String) -> Int? {
    guard let raw = ProcessInfo.processInfo.environment[key],
        let value = Int(raw),
        value >= 0,
        value < LagunaConstants.numHiddenLayers
    else {
        return nil
    }
    return value
}

private let lagunaNativeAffineQKVOnlyLayer = lagunaNativeAffineOnlyLayer(
    "DARKBLOOM_NATIVE_AFFINE_QKV_ONLY_LAYER")

private let lagunaNativeAffineOProjOnlyLayer = lagunaNativeAffineOnlyLayer(
    "DARKBLOOM_NATIVE_AFFINE_OPROJ_ONLY_LAYER")

private func lagunaNativeAffineCovers(layer: Int, count: Int, onlyLayer: Int?) -> Bool {
    guard count > 0 else { return false }
    if let onlyLayer { return layer == onlyLayer }
    if lagunaNativeAffineSuffixSelection {
        return layer >= LagunaConstants.numHiddenLayers - count
    }
    return layer < count
}

private func lagunaUseNativeAffineQKV(layer: Int) -> Bool {
    lagunaNativeAffineCovers(
        layer: layer,
        count: lagunaNativeAffineQKVLayerCount,
        onlyLayer: lagunaNativeAffineQKVOnlyLayer)
}

/// The same native group-32 affine INT8 side layout applied to the attention
/// output projection. `o_proj` is the single largest BF16 decode weight read
/// left in the attention block — 30 sliding layers at `[2048, 8192]` plus 10
/// full-attention layers at `[2048, 6144]` is ~1.2 GB of the decode token's
/// weight traffic — and unlike Q/K/V it is read *after* SDPA, so quantizing it
/// changes nothing about the KV dependency or the cache contents.
///
/// Decode only: prefill and every non-`[1, 1, ·]` call keep the BF16 parameter,
/// which stays authoritative and resident. The first 16 layers are the
/// acceptance-band-safe first chunk; later submissions can widen the same
/// layout the way `DARKBLOOM_NATIVE_AFFINE_QKV_LAYERS` was widened.
///
/// Set `DARKBLOOM_NATIVE_AFFINE_OPROJ=0` (or `..._LAYERS=0`) to fall back to
/// the exact stock gated projection inside the same binary.
private let lagunaNativeAffineOProjLayerCount: Int = {
    guard ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_OPROJ"] != "0"
    else { return 0 }
    let requested = Int(
        ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_OPROJ_LAYERS"]
            ?? "40") ?? 40
    return min(max(requested, 0), LagunaConstants.numHiddenLayers)
}()
let lagunaNativeAffineOProjEnabled = lagunaNativeAffineOProjLayerCount > 0

private func lagunaUseNativeAffineOProj(layer: Int) -> Bool {
    lagunaNativeAffineCovers(
        layer: layer,
        count: lagunaNativeAffineOProjLayerCount,
        onlyLayer: lagunaNativeAffineOProjOnlyLayer)
}

/// The same native group-32 affine INT8 side layout applied to the attention
/// per-head gate projection (`g_proj`), admitted to the accepted quantization
/// envelope by the g_proj amendment. `g_proj` is a tiny `[heads, 2048]` BF16
/// read (48/64 rows), but its decode GEMV is one extra dispatch per layer
/// against the same normalized row the Q/K/V batch already computes, and its
/// output feeds only the softplus gate — never the KV cache — so requantizing
/// it perturbs the model strictly less than the already-shipped Q/K/V and
/// o_proj layouts. Unlike those layouts the gate bank is ALWAYS group-32
/// affine INT8: the envelope caps `g_proj` there, so the NVFP4 tail window
/// (`lagunaNativeAffineNVFP4From`) and the measurement-only probe format do
/// not apply to it. On layers whose QKV bank is itself group-32 INT8 the gate
/// rows concatenate into that same bank and ride the same dispatch for free;
/// on the NVFP4 tail layers the gate keeps a separate group-32 INT8 bank and
/// replaces the BF16 GEMV one dispatch for one dispatch.
///
/// Decode only: prefill and every non-`[1, 1, ·]` call keep the BF16
/// parameter, which stays authoritative and resident. Coverage is prepared
/// inside `prepareNativeAffineQKVWeight`, so a layer only gets the gate
/// layout when its QKV layout is also active.
///
/// Set `DARKBLOOM_NATIVE_AFFINE_GPROJ=0` (or `..._LAYERS=0`) to fall back to
/// the exact stock BF16 gate projection inside the same binary.
private let lagunaNativeAffineGProjLayerCount: Int = {
    guard ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_GPROJ"] != "0"
    else { return 0 }
    let requested = Int(
        ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_GPROJ_LAYERS"]
            ?? "40") ?? 40
    return min(max(requested, 0), LagunaConstants.numHiddenLayers)
}()
let lagunaNativeAffineGProjEnabled = lagunaNativeAffineGProjLayerCount > 0

private let lagunaNativeAffineGProjOnlyLayer = lagunaNativeAffineOnlyLayer(
    "DARKBLOOM_NATIVE_AFFINE_GPROJ_ONLY_LAYER")

private func lagunaUseNativeAffineGProj(layer: Int) -> Bool {
    lagunaNativeAffineCovers(
        layer: layer,
        count: lagunaNativeAffineGProjLayerCount,
        onlyLayer: lagunaNativeAffineGProjOnlyLayer)
}

/// Builds the gate projection's side layout. Unlike
/// `lagunaNativeAffineWeight` this NEVER takes the NVFP4 tail window or the
/// probe round-trip: the accepted envelope admits `g_proj` only as group-32
/// affine INT8, so the layout is fixed regardless of layer depth.
private func lagunaNativeAffineGProjWeight(_ weight: MLXArray) -> LagunaNativeAffineWeight? {
    guard weight.dtype == .bfloat16, weight.ndim == 2,
        weight.dim(1).isMultiple(of: 32)
    else {
        return nil
    }
    let (packedCodes, scales, biases) = quantized(
        weight, groupSize: 32, bits: 8, mode: .affine)
    guard biases != nil else { return nil }
    var w = LagunaNativeAffineWeight(
        packedCodes: packedCodes,
        scales: scales,
        biases: biases,
        originalShape: weight.shape
    )
    w.interleavedMetadata = contiguous(
        stacked([scales, biases!], axis: -1).reshaped(scales.shape[0], -1))
    return w
}

/// Sliding-layer per-head RMSNorm + plain RoPE fusion (see
/// `lagunaSlidingQKNormRoPEKernel`).
///
/// **DEFAULT ON, deliberately** (`!= "0"`; set
/// `DARKBLOOM_FUSED_SLIDING_QK_NORM_ROPE=0` to ablate).
///
/// Default ON since `9e06de6` (+1.73%): the original -0.19% was caused by a
/// redundant `simdgroup_barrier` for inverse-RMS broadcast, not kernel shape.
let lagunaFusedSlidingQKNormRoPEEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_SLIDING_QK_NORM_ROPE"] != "0"

/// Multi-token (prefill) twin of the two decode QK-norm+RoPE fusions above
/// (see `lagunaPrefillSlidingQKNormRoPEKernel` /
/// `lagunaPrefillFullQKNormYaRNKernel`). One dispatch per layer replaces the
/// four stock dispatches on sliding layers (q RMSNorm, k RMSNorm, RoPE q,
/// RoPE k) and the six on full-attention layers (the partial-YaRN RoPE first
/// materializes a general copy of the transposed view). The cos/sin rows
/// come from the same load-time probe-seed atlas the decode kernels can
/// consume, so every rotary factor is a float the stock RoPE kernel itself
/// produced.
///
/// **DEFAULT ON** (`!= "0"`; set `DARKBLOOM_PREFILL_QK_NORM_ROPE=0` to
/// ablate). Guarded on shape/dtype/family and a host-known cache offset
/// with `offset + L <= lagunaRoPEAngleAtlasLength`; every other case takes
/// the verbatim stock path.
private let lagunaPrefillQKNormRoPEEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_QK_NORM_ROPE"] != "0"

/// Heads-per-threadgroup repartition for the prefill QK-norm+RoPE kernels.
/// The shipped kernels pack four heads (four SIMDs) per threadgroup; this
/// selects a one-head-per-threadgroup twin (one SIMD) instead -- the proven
/// DECODE shape. Bit-exact in the EG256 class: each head is one SIMD and all
/// per-head arithmetic is SIMD-local, so only threadgroup composition changes.
/// Default `1` selects H1; `DARKBLOOM_PREFILL_QK_HEADS=4` restores the control.
let lagunaPrefillQKHeadsPerGroup: Int = {
    let raw =
        ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_QK_HEADS"]
        ?? "1"
    return raw == "4" ? 4 : 1
}()

/// Terminal-prefill projection banking. The last decoder layer consumes Q and
/// the per-head gate for only the final supplied row, while K/V must still be
/// produced for every row so the cache advances normally. Retained `[Q; G]`
/// and `[K; V]` BF16 banks therefore turn four independent projections into
/// two without changing any output row's contraction or rounding. Set
/// `DARKBLOOM_LAST_PREFILL_PROJECTION_BANKS=0` to restore the four stock
/// `Linear` calls and skip allocating the two derived banks.
private let lagunaLastPrefillProjectionBanksEnabled =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_LAST_PREFILL_PROJECTION_BANKS"] != "0"

/// `DARKBLOOM_TERMINAL_FUSION` (default ON; set "0" to disable): current-API
/// port of overlay-dropped `9f98995`. `callLastPrefillRow` reuses the ordinary
/// path's accepted row-local fused residual+RMSNorm(+router)+MoE-tail helpers.
private let lagunaTerminalPrefillFusionEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_TERMINAL_FUSION"] != "0"

/// Full-attention counterpart: fuses per-head Q/K RMSNorm with partial YaRN
/// RoPE. One stock FP32 probe row carries the authoritative rotary factors,
/// while the custom kernel preserves the normalized BF16 boundary and tail.
/// Folds the MoE router's `[256, 2048]` projection into the post-attention
/// residual + RMSNorm kernel, which is the dispatch immediately before it and
/// its only producer. Set `DARKBLOOM_FUSED_RESIDUAL_RMS_ROUTER=0` to ablate.
let lagunaFusedResidualRMSNormRouterEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_RESIDUAL_RMS_ROUTER"] != "0"

let lagunaFusedFullQKNormYaRNEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_FULL_QK_NORM_YARN"] != "0"

/// Decode-only carrier for the two authoritative RoPE angle rows consumed by
/// the fused Q/K kernels. At load time each attention family's own stock RoPE
/// materializes an exact FP32 position atlas. A single custom kernel then
/// replaces the token embedding gather and copies both selected atlas rows,
/// removing the two per-token probe RoPE dispatches without changing their
/// values.
///
/// Default ON since the r=1-regime re-sweep (2026-08-02, M5 Max driver rig):
/// under the current one-row QMV geometry + counting-sort frontier the arm
/// measures −0.55..−0.7% steady decode, 4/4 mirrored pairs favoring ON
/// (medians 4.521/4.535/4.536/4.533 vs controls 4.563/4.557/4.574/4.543),
/// inverting the earlier fusion-stack-audit conclusion (−0.23% against ON
/// under the pre-r=1 regime) — the same regime-rot pattern the tail QKV
/// fusion showed in reverse. Values are unchanged by construction (the
/// atlas rows are the family's own stock RoPE outputs, copied); free-run
/// token hash and 1,600 teacher-forced steps are identical across arms.
/// Set `DARKBLOOM_ROPE_ANGLE_ATLAS=0` to restore the stock fallback
/// (`embedTokens` gather + `ropeAngleTable` probes).
let lagunaRoPEAngleAtlasEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ROPE_ANGLE_ATLAS"] != "0"

/// Zero-dispatch decode angle carrier: serve the two per-step RoPE angle rows
/// as contiguous row VIEWS of the load-time FP32 position atlases instead of
/// running the two probe RoPE dispatches every token. Unlike the fused
/// embedding+atlas kernel above (default OFF; its fixed kernel cost measured
/// −0.23%), this path adds no kernel at all: the atlas row for position `p`
/// is bit-identical to the probe output at `p` by construction (the atlas IS
/// the family's own stock RoPE run over the broadcast probe seed), and a
/// row slice of the contiguous `[1, 1, 4096, D]` atlas is a zero-copy
/// row-contiguous view, so the two probe dispatches vanish from the front of
/// every decode step with no replacement work. The stock `embedTokens`
/// gather is unchanged.
///
/// MEASURED (2026-08-01, M5 Max 128 GB, driver rig, 150-step cool-floor
/// ABBA): views are +0.01..+0.07 ms/step vs the probe dispatches — the two
/// probes are off the critical path (they overlap the embedding gather and
/// layer-0 front), so removing them buys nothing, and aliasing the ~3 MB
/// atlas buffers as per-step kernel inputs appears to add slight
/// resource-tracking cost. Default OFF, same promoted-era conclusion as the
/// fused embedding+atlas kernel above. Set `DARKBLOOM_ROPE_ATLAS_VIEWS=1`
/// to re-measure.
let lagunaRoPEAtlasViewsEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ROPE_ATLAS_VIEWS"] == "1"

/// `DARKBLOOM_FUSED_DENSE_GATE_UP_SWIGLU` (default on; set "0" to disable):
/// after checkpoint load, retain one row-concatenated BF16 `[gate; up]` bank
/// for layer 0's dense (non-quantized) MLP and serve single-token decode's
/// gate/up projections plus the SiLU-gated product from one dispatch (see
/// `LagunaRuntimeMLP.fusedDenseDownResidual` and `lagunaDenseGateUpSwiGLU`).
/// Layer 0 is the only layer whose MLP is plain BF16 `Linear` rather than
/// NVFP4 `QuantizedLinear`, so every gate/up fusion flag above (all guarded
/// on `QuantizedLinear`) always declines for it; this is its dedicated
/// counterpart.
let lagunaFusedDenseGateUpSwiGLUEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_DENSE_GATE_UP_SWIGLU"] != "0"

/// `DARKBLOOM_FUSED_DENSE_DOWN_RESIDUAL` (default on; set "0" to disable):
/// layer-0-only decode fusion of the dense MLP's down projection with the
/// decoder layer's `h + r2` residual add (see `lagunaDenseDownResidual`).
/// Every other down+residual fusion flag in this file requires
/// `mlp as? LagunaRuntimeSparseMoEBlock`, which layer 0 never satisfies, so
/// layer 0's residual add was the one MLP-side decode dispatch left with no
/// fusion counterpart at all before this flag.
let lagunaFusedDenseDownResidualEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_DENSE_DOWN_RESIDUAL"] != "0"

/// `DARKBLOOM_ROUTER_ROWS_PER_GROUP` (default `8`; set `64` to restore the
/// pre-widening shape, `32`/`16` for intermediate points, `4`/`2`/`1` for the
/// sub-8 shapes): router output rows owned by one threadgroup in
/// `laguna_residual_rms_router_bf16_2048`.
///
/// At `64` only 4 threadgroups (140 GB/s, 24% of box). Halving doubles tiles: 8->32.
/// Bit-exact: only changes which threadgroup owns which row; no add regrouped.
/// Sub-8 is measured null (redundant norm cancels extra-tile overlap).
let lagunaRouterRowsPerGroup: Int = {
    guard
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_ROUTER_ROWS_PER_GROUP"],
        let value = Int(raw), [1, 2, 4, 8, 16, 32, 64].contains(value)
    else {
        return 8
    }
    return value
}()

/// `DARKBLOOM_DECODE_ASYNC_STAGE` (default `at:1,7,15,23,31,39`): process-once
/// boundary schedule for decode-step async scheduling. Active only when the
/// invocation input shape is exactly `[1, 1]`; prefill and multi-token shapes
/// are never asyncEval'd. `off`/`0` disables it; `norm` and `logits` remain
/// process-once ablation points, as does any single layer index `0`-`39`.
/// No operation, cache row, or token is added.
private enum LagunaDecodeAsyncStage {
    case off
    case layer(Int)
    case ladder(Int)
    case explicit(UInt64)
    case norm
    case logits
}

/// Two schedule families, both special cases of the `at:i,j,k` boundary set:
/// `ladderN` fires after every `N`th layer, and `at:` names the boundaries
/// outright. `asyncEval` adds no operation, cache row, dtype boundary or
/// token — it only enqueues already-constructed work earlier — so every
/// schedule here is bit-exact and the choice is purely a measurement.
///
/// MEASURED (`notes/52`, 66 runs, all passed): ladder8=1.000, ladder6=1.0064,
/// ladder2=1.0169, ladder1=1.0178, at:1,7,15,23,31,39=1.0170. Adding one front
/// rung to ladder8 matches 40-fire ladder1 for one extra scheduler round trip.
private let lagunaDecodeAsyncStage: LagunaDecodeAsyncStage = {
    let raw =
        ProcessInfo.processInfo.environment["DARKBLOOM_DECODE_ASYNC_STAGE"]?
        .lowercased() ?? "at:0,1,7,15,23,31,39"
    switch raw {
    case "off", "0", "":
        return .off
    case "norm":
        return .norm
    case "logits":
        return .logits
    default:
        // `at:i,j,k` — an arbitrary boundary set, as a bitmask over decoder
        // layer indices (bit `i` ⇒ fire after layer `i`). `ladderN` and the
        // single-rung forms are both special cases of it, so every schedule
        // shape can be measured without a rebuild. Indices ≥ 64 are rejected
        // rather than silently dropped; Laguna has 40 layers.
        if raw.hasPrefix("at:") {
            var mask: UInt64 = 0
            for field in raw.dropFirst(3).split(separator: ",") {
                guard let index = Int(field), (0..<64).contains(index) else { return .off }
                mask |= 1 << UInt64(index)
            }
            return mask == 0 ? .off : .explicit(mask)
        }
        if raw.hasPrefix("ladder"), let stride = Int(raw.dropFirst("ladder".count)),
            (1...40).contains(stride)
        {
            return .ladder(stride)
        }
        if let index = Int(raw), (0...39).contains(index) {
            return .layer(index)
        }
        return .off
    }
}()

/// Diagnostic front-edge rung: enqueue layer 0's already-constructed QKV and
/// gate projections before the rest of that layer's graph is built.
private let lagunaAttentionProjectionAsyncEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ATTN_PROJECTION_ASYNC"] != "0"

/// `DARKBLOOM_PREFILL_ASYNC_LADDER` (default `1`; `0`/`off` disables;
/// `8` restores the prior default): a ranked measurement on the
/// 1.87782 base scored stride 1 at 1.88526 (+0.40% vs that base, rejected
/// only because a larger win promoted mid-queue), and the decode-side
/// ladder sweep showed denser firing pays until graph-build cost
/// dominates. Stride 1 fires `asyncEval` after every layer:
/// prefill-side twin of the decode ladder above. Multi-token forwards build
/// a ~400-op graph with the GPU idle until the final eval; firing `asyncEval`
/// after every Nth layer streams completed segments exactly as the promoted
/// decode ladder does. Same exactness ground: no operation, order, cache
/// write, or token changes — only when already-constructed work is enqueued.
/// This pays into both score components: the prefill phase itself and the
/// 512-token seed prefill charged to the decode window.
private let lagunaPrefillAsyncLadderStride: Int = {
    let raw = ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_ASYNC_LADDER"]?.lowercased() ?? "1"
    if raw == "off" || raw == "0" || raw.isEmpty { return 0 }
    guard let n = Int(raw), (1...40).contains(n) else { return 0 }
    return n
}()

private let lagunaRoPEAngleAtlasLength = 4096

/// The shared 512-thread RMSNorm prologue emitted by three decode kernels.
///
/// 512 threads / 16 simdgroups square one 2048-wide row, `simd_sum` inside
/// each simdgroup, the sixteen partials meet in `local_sums`, and one inverse
/// RMS comes back out. Gathering those sixteen partials genuinely needs
/// threadgroup memory -- `simd_shuffle*` reaches only the 32 lanes of the
/// calling simdgroup, so no shuffle crosses from simdgroup 5 to simdgroup 0.
///
/// SHUF_NORM_BCAST/INIT barrier removal: rejected on ranked box (-0.07%), removed.
private let lagunaNormInvMeanScratch = "threadgroup float local_inv_mean[1];"

/// Emits the cross-simdgroup half of that prologue, from the sixteen partial
/// writes through to a `float laguna_inv_mean` the normalize loop consumes.
/// The emitted text is line for line what the three kernels shipped, plus a
/// register alias for `local_inv_mean[0]`.
private func lagunaNormReductionTail(
    lane: String,
    simdGroup: String,
    denominator: String,
    epsilon: String
) -> String {
    let inverseRMS =
        "metal::precise::rsqrt(acc / \(denominator) + \(epsilon))"
    let lines: [String] = [
        "if (\(simdGroup) == 0) {",
        "    local_sums[\(lane)] = 0.0f;",
        "}",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        "if (\(lane) == 0) {",
        "    local_sums[\(simdGroup)] = acc;",
        "}",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        "if (\(simdGroup) == 0) {",
        "    acc = simd_sum(local_sums[\(lane)]);",
        "    if (\(lane) == 0) {",
        "        local_inv_mean[0] = \(inverseRMS);",
        "    }",
        "}",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        "float laguna_inv_mean = local_inv_mean[0];",
    ]
    // The first line inherits the enclosing literal's indentation, exactly as
    // the `\(epilogue)` interpolation in the router kernel does.
    return lines.joined(separator: "\n        ")
}

/// The 2048-row prologue shared by both residual+RMSNorm kernels.
private let lagunaNormReductionTail2048 = lagunaNormReductionTail(
    lane: "simd_lane", simdGroup: "simd_group",
    denominator: "2048.0f", epsilon: "1.0e-6f")

/// The fused QKV kernel names the same two builtins `lane` and `simd_group`
/// and spells its constants differently, but the reduction is the same shape.
private let lagunaNormReductionTailQKV = lagunaNormReductionTail(
    lane: "lane", simdGroup: "simd_group",
    denominator: "float(in_vec_size)", epsilon: "norm_eps")

/// Post-attention residual add + RMSNorm with the MoE router's projection
/// folded in.
///
/// Folds the `[256,2048]` BF16 router GEMV into the norm kernel (redundant 4KB read,
/// removes a dispatch). Router half replicates MLX gemv (BM4/BN1/SM1/SN32, `i` then
/// `tn` FP32 accumulation, `simd_shuffle_down` reduce, one BF16 round). `rowsPerGroup`
/// only changes WHICH threadgroup owns which row; no add is regrouped. **LOADS ONLY**:
/// `router_result[0]` is a single accumulator in strict `(block,i)` order — tree-summing
/// the 4-block unroll would regroup 64 FP32 adds and break bit-exactness. At
/// `rows_per_group==1` the block loop unrolls 4-deep (256KB in-flight vs 64KB).
private func lagunaResidualRMSNormRouterSource(rowsPerGroup: Int) -> String {
    let simdGroups = 512 / 32
    let rowsPerThread = rowsPerGroup >= simdGroups ? rowsPerGroup / simdGroups : 1
    let activeSimdGroups = rowsPerGroup / rowsPerThread
    let zeros = Array(repeating: "0.0f", count: rowsPerThread).joined(separator: ", ")
    let guardOpen = activeSimdGroups < simdGroups
        ? "        if (simd_group < active_simd_groups) {\n" : ""
    let guardClose = activeSimdGroups < simdGroups ? "        }\n" : ""
    let routerStore = "router_logits[router_row + r] = bfloat(router_result[r]);"

    let accumulate: String
    if rowsPerThread == 1 {
        accumulate = """
                    uint column = simd_lane * n_reads;
                    for (uint block = 0; block < router_blocks; block += 4) {
                        vec<bfloat, 4> rw[4];
                        for (uint u = 0; u < 4; ++u) {
                            const device vec<bfloat, 4>* row_values =
                                (const device vec<bfloat, 4>*)(
                                    router_weight + router_row * axis_size +
                                        column + u * block_width);
                            rw[u] = row_values[0];
                        }
                        for (uint u = 0; u < 4; ++u) {
                            uint column_u = column + u * block_width;
                            for (uint i = 0; i < n_reads; ++i) {
                                router_result[0] += float(rw[u][i]) *
                                    float(normalized_row[column_u + i]);
                            }
                        }
                        column += 4 * block_width;
                    }
            """
    } else {
        accumulate = """
                    thread float router_input[n_reads];

                    uint column = simd_lane * n_reads;
                    for (uint block = 0; block < router_blocks; ++block) {
                        for (uint i = 0; i < n_reads; ++i) {
                            router_input[i] = float(normalized_row[column + i]);
                        }
                        for (uint r = 0; r < rows_per_thread; ++r) {
                            const device vec<bfloat, 4>* row_values =
                                (const device vec<bfloat, 4>*)(
                                    router_weight + (router_row + r) * axis_size +
                                        column);
                            const vec<bfloat, 4> rw = row_values[0];
                            for (uint i = 0; i < n_reads; ++i) {
                                router_result[r] += float(rw[i]) * router_input[i];
                            }
                        }
                        column += block_width;
                    }
            """
    }

    return """
        constexpr uint axis_size = 2048;
        constexpr uint n_reads = 4;
        constexpr uint simd_size = 32;
        constexpr uint rows_per_group = \(rowsPerGroup);
        constexpr uint rows_per_thread = \(rowsPerThread);
        constexpr uint active_simd_groups = \(activeSimdGroups);
        constexpr uint block_width = 128;
        constexpr uint router_blocks = axis_size / block_width;

        uint tile = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint base = lid * n_reads;

        \(lagunaNormInvMeanScratch)
        threadgroup float local_sums[simd_size];
        threadgroup bfloat normalized_row[axis_size];

        thread bfloat values[n_reads];
        float acc = 0.0f;
        for (uint i = 0; i < n_reads; ++i) {
            bfloat value = bfloat(residual[base + i] + branch[base + i]);
            values[i] = value;
            if (tile == 0) {
                summed[base + i] = value;
            }
            float fv = float(value);
            acc += fv * fv;
        }

        acc = simd_sum(acc);
        \(lagunaNormReductionTail2048)

        for (uint i = 0; i < n_reads; ++i) {
            bfloat value =
                weight[base + i] *
                bfloat(float(values[i]) * laguna_inv_mean);
            normalized_row[base + i] = value;
            if (tile == 0) {
                normalized[base + i] = value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- router projection ---
        \(guardOpen)\
        uint router_row = tile * rows_per_group + simd_group * rows_per_thread;
        thread float router_result[rows_per_thread] = {\(zeros)};
        \(accumulate)

        for (uint r = 0; r < rows_per_thread; ++r) {
            for (ushort delta = 16; delta >= 1; delta >>= 1) {
                router_result[r] +=
                    metal::simd_shuffle_down(router_result[r], delta);
            }
        }
        if (simd_lane == 0) {
            for (uint r = 0; r < rows_per_thread; ++r) {
                \(routerStore)
            }
        }
        \(guardClose)
        """
}

/// One kernel per supported `rows_per_group`, all built eagerly so that every
/// arm of an ablation is served by the same binary (`notes/00`'s one-binary
/// rule). MLX keys its JIT library cache by name and clears it when a name's
/// source changes (`custom_kernel.cpp:58-68`), so the variant MUST be in the
/// name or four sources would thrash one cache entry.
private let lagunaResidualRMSNormRouterKernels: [Int: MLXFast.MLXFastKernel] =
    Dictionary(
        uniqueKeysWithValues: [1, 2, 4, 8, 16, 32, 64].map { rowsPerGroup in
            (
                rowsPerGroup,
                MLXFast.metalKernel(
                    name: "laguna_residual_rms_router_bf16_2048_rpg\(rowsPerGroup)_v2",
                    inputNames: ["residual", "branch", "weight", "router_weight"],
                    outputNames: ["summed", "normalized", "router_logits"],
                    source: lagunaResidualRMSNormRouterSource(rowsPerGroup: rowsPerGroup),
                    header: "",
                    ensureRowContiguous: true
                )
            )
        })

/// Residual add + RMSNorm for the layers whose MLP is not a sparse block
/// (layer 0) and for any shape the router fusion above declines.
private let lagunaResidualRMSNormKernel = MLXFast.metalKernel(
    name: "laguna_residual_rms_bf16_2048_v1",
    inputNames: ["residual", "branch", "weight"],
    outputNames: ["summed", "normalized"],
    source: """
        constexpr uint axis_size = 2048;
        constexpr uint n_reads = 4;
        constexpr uint simd_size = 32;

        uint row = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint base = row * axis_size + lid * n_reads;

        \(lagunaNormInvMeanScratch)
        threadgroup float local_sums[simd_size];

        thread bfloat values[n_reads];
        float acc = 0.0f;
        for (uint i = 0; i < n_reads; ++i) {
            bfloat value = bfloat(residual[base + i] + branch[base + i]);
            values[i] = value;
            summed[base + i] = value;
            float fv = float(value);
            acc += fv * fv;
        }

        acc = simd_sum(acc);
        \(lagunaNormReductionTail2048)

        for (uint i = 0; i < n_reads; ++i) {
            normalized[base + i] =
                weight[lid * n_reads + i] *
                bfloat(float(values[i]) * laguna_inv_mean);
        }
        """,
    ensureRowContiguous: true
)

func lagunaResidualRMSNormRouter(
    residual: MLXArray, branch: MLXArray, weight: MLXArray,
    routerWeight: MLXArray, correctionBias: MLXArray
) -> (summed: MLXArray, normalized: MLXArray, routerLogits: MLXArray) {
    let hidden = LagunaConstants.hiddenSize
    let experts = LagunaConstants.numExperts
    precondition(residual.dtype == .bfloat16)
    precondition(branch.dtype == .bfloat16)
    precondition(weight.dtype == .bfloat16)
    precondition(routerWeight.dtype == .bfloat16)
    precondition(correctionBias.dtype == .float32 || correctionBias.dtype == .bfloat16)
    precondition(residual.dims(1, 1, hidden))
    precondition(branch.dims(1, 1, hidden))
    precondition(weight.dims(hidden))
    precondition(routerWeight.dims(experts, hidden))
    precondition(correctionBias.dims(experts))

    // `rows_per_group` router rows per threadgroup, so 256 / rows_per_group
    // tiles. Divides exactly for 64/32/16/8/4/2/1 (4..256 tiles), so no partial
    // tile is dispatched and no row is computed twice or missed. The 512-thread
    // threadgroup and `n_reads == 4` are NOT knobs: they are load-bearing for
    // the `rms_single_row` correspondence (each thread squares its own
    // contiguous four elements), and moving either regroups the FP32 RMS
    // summation and forfeits bit-exactness.
    let rowsPerGroup = lagunaRouterRowsPerGroup
    let tiles = experts / rowsPerGroup
    lagunaTrace("residual+rmsnorm+router rpg\(rowsPerGroup)")
    let inputs = [residual, branch, weight, routerWeight]
    let outputs = lagunaResidualRMSNormRouterKernels[rowsPerGroup]!(
        inputs,
        grid: (tiles * 512, 1, 1),
        threadGroup: (512, 1, 1),
        outputShapes: [[1, 1, hidden], [1, 1, hidden], [1, 1, experts]],
        outputDTypes: [.bfloat16, .bfloat16, .bfloat16]
    )
    return (outputs[0], outputs[1], outputs[2])
}

func lagunaResidualRMSNorm(
    residual: MLXArray, branch: MLXArray, weight: MLXArray
) -> (MLXArray, MLXArray) {
    precondition(residual.dtype == .bfloat16)
    precondition(branch.dtype == .bfloat16)
    precondition(weight.dtype == .bfloat16)
    precondition(residual.sameDims(branch))
    precondition(residual.dim(-1) == LagunaConstants.hiddenSize)
    precondition(weight.dims(LagunaConstants.hiddenSize))

    let rows = residual.size / LagunaConstants.hiddenSize
    let outputs = lagunaResidualRMSNormKernel(
        [residual, branch, weight],
        grid: (rows * 512, 1, 1),
        threadGroup: (512, 1, 1),
        outputShapes: [residual.shape, residual.shape],
        outputDTypes: [.bfloat16, .bfloat16]
    )
    return (outputs[0], outputs[1])
}

// MARK: - Attention

private let lagunaFullQKNormYaRNKernel = MLXFast.metalKernel(
    name: "laguna_full_qk_norm_yarn_bf16_128_v4",
    inputNames: ["raw_queries", "raw_keys", "query_weight", "key_weight", "angles"],
    outputNames: ["queries", "keys"],
    source: """
        constexpr uint head_dim = 128;
        constexpr uint rotary_dims = 64;
        constexpr uint rotary_pairs = 32;
        constexpr uint query_heads = 48;
        constexpr float yarn_mscale = 1.3465735912322998f;

        uint head = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;


        const device bfloat* input;
        const device bfloat* weight;
        if (head < query_heads) {
            input = raw_queries + head * head_dim;
            weight = query_weight;
        } else {
            input = raw_keys + (head - query_heads) * head_dim;
            weight = key_weight;
        }

        uint base = lane * 4;
        thread bfloat normalized[4];
        float sum = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            float value = float(input[base + i]);
            sum += value * value;
        }
        // `simd_sum` already returns the total to every lane, so each lane
        // derives the same `precise::rsqrt` locally. That removes the
        // threadgroup slot and the barrier this one-simdgroup-per-head kernel
        // would otherwise pay for on every head.
        sum = simd_sum(sum);
        float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

        for (uint i = 0; i < 4; ++i) {
            normalized[i] =
                weight[base + i] *
                bfloat(float(input[base + i]) * inverse_rms);
        }

        thread float paired[4];
        for (uint i = 0; i < 4; ++i) {
            paired[i] = simd_shuffle(float(normalized[i]), lane ^ 8);
        }

        device bfloat* output =
            head < query_heads
            ? queries + head * head_dim
            : keys + (head - query_heads) * head_dim;
        if (lane < 8) {
            bfloat rounded_mscale = bfloat(yarn_mscale);
            for (uint i = 0; i < 4; ++i) {
                uint pair = base + i;
                float first =
                    float(bfloat(normalized[i] * rounded_mscale));
                float second =
                    float(bfloat(bfloat(paired[i]) * rounded_mscale));
                float cosine = angles[pair];
                float sine = angles[pair + rotary_pairs];
                output[pair] = bfloat(first * cosine - second * sine);
                output[pair + rotary_pairs] =
                    bfloat(first * sine + second * cosine);
            }
        } else if (lane >= 16) {
            for (uint i = 0; i < 4; ++i) {
                output[base + i] = normalized[i];
            }
        }
        """,
    ensureRowContiguous: true
)

func lagunaFullQKNormYaRN(
    rawQueries: MLXArray,
    rawKeys: MLXArray,
    queryWeight: MLXArray,
    keyWeight: MLXArray,
    angles: MLXArray
) -> (MLXArray, MLXArray) {
    precondition(rawQueries.dtype == .bfloat16)
    precondition(rawKeys.dtype == .bfloat16)
    precondition(queryWeight.dtype == .bfloat16)
    precondition(keyWeight.dtype == .bfloat16)
    precondition(rawQueries.dims(1, 1, 48 * LagunaConstants.headDim))
    precondition(rawKeys.dims(1, 1, 8 * LagunaConstants.headDim))
    precondition(queryWeight.dims(LagunaConstants.headDim))
    precondition(keyWeight.dims(LagunaConstants.headDim))
    precondition(angles.dtype == .float32)
    precondition(angles.dims(1, 1, 1, LagunaConstants.headDim / 2))

    lagunaTrace("full qk norm+yarn")
    let outputs = lagunaFullQKNormYaRNKernel(
        [rawQueries, rawKeys, queryWeight, keyWeight, angles],
        grid: (56 * 32, 1, 1),
        threadGroup: (32, 1, 1),
        outputShapes: [
            [1, 48, 1, LagunaConstants.headDim],
            [1, 8, 1, LagunaConstants.headDim],
        ],
        outputDTypes: [.bfloat16, .bfloat16]
    )
    return (outputs[0], outputs[1])
}

/// Sliding-layer twin of the full-attention QK-norm+RoPE kernel above. The
/// thirty sliding layers carry plain RoPE -- the whole 128-element head
/// rotates, the angle scale is one, and there is no YaRN mscale -- so their
/// per-head RMSNorm and rotation stayed on the stock four-dispatch path
/// (`q_norm`, `k_norm`, RoPE(q), RoPE(k)) while the ten full-attention layers
/// were fused. This kernel closes that gap: one dispatch per decode step per
/// layer for all 72 heads, emitting the transposed `[1, heads, 1, 128]` layout
/// attention consumes directly.
///
/// Exactness, link for link with the pair it replaces:
///  * The RMSNorm half mirrors `rms_single_row` (rms_norm.metal) at
///    axis_size 128 with N_READS 4 and a 32-thread group: lane `l` owns the
///    contiguous block `[4l, 4l+4)`, accumulates `float(x)^2` in index order,
///    `simd_sum`s, and applies `precise::rsqrt(acc / 128 + eps)`. The
///    `bfloat(...)` inside `w[i] * bfloat(x[i] * inv_mean)` is load-bearing:
///    it is the same rounding the separate kernel would have written out and
///    the rotation would have read back.
///  * The rotation mirrors `rope_single_impl<T, false>` for `dims == 128`:
///    pair `p` couples elements `p` and `p + 64`, and `cos`/`sin` come from a
///    table produced by that very kernel (see `_slidingRoPEAngleSeed`), so
///    they are the same floats, not a re-derivation.
private let lagunaSlidingQKNormRoPEKernel = MLXFast.metalKernel(
    name: "laguna_sliding_qk_norm_rope_bf16_128_v1",
    inputNames: ["raw_queries", "raw_keys", "query_weight", "key_weight", "angles"],
    outputNames: ["queries", "keys"],
    source: """
        constexpr uint head_dim = 128;
        constexpr uint rotary_pairs = 64;
        constexpr uint query_heads = 64;

        uint head = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;


        const device bfloat* input;
        const device bfloat* weight;
        if (head < query_heads) {
            input = raw_queries + head * head_dim;
            weight = query_weight;
        } else {
            input = raw_keys + (head - query_heads) * head_dim;
            weight = key_weight;
        }

        uint base = lane * 4;
        thread bfloat normalized[4];
        float sum = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            float value = float(input[base + i]);
            sum += value * value;
        }
        // `simd_sum` already returns the total to every lane, so each lane
        // derives the same `precise::rsqrt` locally. That removes the
        // threadgroup slot and the barrier this one-simdgroup-per-head kernel
        // would otherwise pay for on every head.
        sum = simd_sum(sum);
        float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

        for (uint i = 0; i < 4; ++i) {
            normalized[i] =
                weight[base + i] *
                bfloat(float(input[base + i]) * inverse_rms);
        }

        // Element `p + 64`, the partner of pair `p`, lives 16 lanes away.
        thread float paired[4];
        for (uint i = 0; i < 4; ++i) {
            paired[i] = simd_shuffle(float(normalized[i]), lane ^ 16);
        }

        device bfloat* output =
            head < query_heads
            ? queries + head * head_dim
            : keys + (head - query_heads) * head_dim;
        // Every element rotates, so the lower sixteen lanes own all 64 pairs
        // and write both halves of each.
        if (lane < 16) {
            for (uint i = 0; i < 4; ++i) {
                uint pair = base + i;
                float first = float(normalized[i]);
                float second = paired[i];
                float cosine = angles[pair];
                float sine = angles[pair + rotary_pairs];
                output[pair] = bfloat(first * cosine - second * sine);
                output[pair + rotary_pairs] =
                    bfloat(first * sine + second * cosine);
            }
        }
        """,
    ensureRowContiguous: true
)

func lagunaSlidingQKNormRoPE(
    rawQueries: MLXArray,
    rawKeys: MLXArray,
    queryWeight: MLXArray,
    keyWeight: MLXArray,
    angles: MLXArray
) -> (MLXArray, MLXArray) {
    let heads = LagunaConstants.slidingAttentionHeads
    let kvHeads = LagunaConstants.numKeyValueHeads
    precondition(rawQueries.dtype == .bfloat16)
    precondition(rawKeys.dtype == .bfloat16)
    precondition(queryWeight.dtype == .bfloat16)
    precondition(keyWeight.dtype == .bfloat16)
    precondition(rawQueries.dims(1, 1, heads * LagunaConstants.headDim))
    precondition(rawKeys.dims(1, 1, kvHeads * LagunaConstants.headDim))
    precondition(queryWeight.dims(LagunaConstants.headDim))
    precondition(keyWeight.dims(LagunaConstants.headDim))
    precondition(angles.dtype == .float32)
    precondition(angles.dims(1, 1, 1, LagunaConstants.headDim))

    lagunaTrace("sliding qk norm+rope")
    let outputs = lagunaSlidingQKNormRoPEKernel(
        [rawQueries, rawKeys, queryWeight, keyWeight, angles],
        grid: ((heads + kvHeads) * 32, 1, 1),
        threadGroup: (32, 1, 1),
        outputShapes: [
            [1, heads, 1, LagunaConstants.headDim],
            [1, kvHeads, 1, LagunaConstants.headDim],
        ],
        outputDTypes: [.bfloat16, .bfloat16]
    )
    return (outputs[0], outputs[1])
}

/// `DARKBLOOM_FUSED_SLIDING_ATTN` (default on; set "0" to disable): decode
/// fused attention for the thirty sliding-window layers in the steady
/// wrapped regime. ONE dispatch replaces the four-stage dependency chain
/// [QK-norm+RoPE kernel] -> [K cache slice-assign] -> [V cache
/// slice-assign] -> [sdpa_vector]: it computes the new token's per-head
/// Q/K RMSNorm + plain RoPE in threadgroup memory (textual replica of
/// `laguna_sliding_qk_norm_rope_bf16_128_v1`), persists the new K/V row
/// into the ring backing at the slot `RotatingKVCache.updateInPlace` would
/// have written, and attends over the full 512-slot ring in slot order with
/// the GQA-pair schedule of the shipped `sdpa_vector` pair path (textual
/// replica: same key visit order per simdgroup, same online-softmax text
/// including the alpha-skip rescale, same two-plane combine and reduction
/// trees). Bit-exactness of the substitution: at slot `write_idx` every
/// threadgroup substitutes the just-computed row from threadgroup memory —
/// the values pass through the same `bfloat` storage rounding the separate
/// kernels would have written to and re-read from the cache, so scores and
/// output are bit-identical, and no threadgroup ever reads slot
/// `write_idx` from device memory, making the concurrent slot write
/// race-free by construction (its only consumers are future steps, ordered
/// by command-buffer sequencing). Removes 3 dispatches + their encoder-wide
/// barriers per sliding layer per decode step.
let lagunaFusedSlidingAttentionEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_SLIDING_ATTN"] != "0"

private let lagunaSlidingFusedAttentionKernel = MLXFast.metalKernel(
    name: "laguna_sliding_fused_attn_ring_v1",
    inputNames: [
        "raw_queries", "raw_keys", "raw_values",
        "query_weight", "key_weight", "angles",
        "k_cache", "v_cache", "params", "scale_arr",
    ],
    outputNames: ["attended"],
    source: """
        constexpr uint head_dim = 128;
        constexpr uint window = 512;
        constexpr uint gqa = 8;
        constexpr int BN = 32;
        constexpr int BD = 32;
        // Pad the epilogue exchange stride off a power of two: the
        // transposing write below is `lane * stride + sg`, which at
        // stride 32 puts all 32 lanes of a simdgroup in one threadgroup
        // memory bank. The odd stride keeps the read contiguous.
        constexpr int BDP = BD + 1;
        constexpr int qk_per_thread = 4;
        constexpr int v_per_thread = 4;
        constexpr uint rotary_pairs = 64;
        constexpr int N = 512;

        typedef float U;

        uint pair_tg = threadgroup_position_in_grid.x;
        uint head0 = pair_tg * 2;
        uint head1 = head0 + 1;
        uint kv_head = head0 / gqa;
        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint widx = params[0];
        float scale = scale_arr[0];

        threadgroup bfloat tg_q0[head_dim];
        threadgroup bfloat tg_q1[head_dim];
        threadgroup bfloat tg_k[head_dim];
        threadgroup bfloat tg_v[head_dim];

        // Phase 1: per-head RMSNorm + plain RoPE, textual replica of
        // laguna_sliding_qk_norm_rope_bf16_128_v1 with the device row
        // writes retargeted at threadgroup memory. simdgroups 0/1/2 own
        // q0/q1/k; simdgroup 3 copies the raw V row (stored unmodified).
        if (sg < 3) {
            const device bfloat* input =
                sg == 0 ? raw_queries + head0 * head_dim
                : sg == 1 ? raw_queries + head1 * head_dim
                          : raw_keys + kv_head * head_dim;
            const device bfloat* weight =
                sg == 2 ? key_weight : query_weight;
            threadgroup bfloat* outrow =
                sg == 0 ? tg_q0 : sg == 1 ? tg_q1 : tg_k;

            uint base = lane * 4;
            thread bfloat normalized[4];
            float sum = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                float value = float(input[base + i]);
                sum += value * value;
            }
            sum = simd_sum(sum);
            float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);
            for (uint i = 0; i < 4; ++i) {
                normalized[i] =
                    weight[base + i] *
                    bfloat(float(input[base + i]) * inverse_rms);
            }
            thread float paired[4];
            for (uint i = 0; i < 4; ++i) {
                paired[i] = simd_shuffle(float(normalized[i]), lane ^ 16);
            }
            if (lane < 16) {
                for (uint i = 0; i < 4; ++i) {
                    uint pair = base + i;
                    float first = float(normalized[i]);
                    float second = paired[i];
                    float cosine = angles[pair];
                    float sine = angles[pair + rotary_pairs];
                    outrow[pair] = bfloat(first * cosine - second * sine);
                    outrow[pair + rotary_pairs] =
                        bfloat(first * sine + second * cosine);
                }
            }
        } else if (sg == 3) {
            const device bfloat* vin = raw_values + kv_head * head_dim;
            for (uint i = lane; i < head_dim; i += 32) {
                tg_v[i] = vin[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: one writer threadgroup per KV head persists the new row
        // for future steps. No threadgroup reads slot widx from device this
        // dispatch (all substitute the threadgroup copy), so cross-group
        // ordering is irrelevant; the next step observes the write through
        // command-buffer sequencing.
        if ((head0 % gqa) == 0 && sg == 0) {
            device bfloat* kc = (device bfloat*)k_cache +
                (size_t)kv_head * (window * head_dim) +
                (size_t)widx * head_dim;
            device bfloat* vc = (device bfloat*)v_cache +
                (size_t)kv_head * (window * head_dim) +
                (size_t)widx * head_dim;
            for (uint i = lane; i < head_dim; i += 32) {
                kc[i] = tg_k[i];
                vc[i] = tg_v[i];
            }
        }

        // Phase 3: GQA-pair attention over the ring in slot order, textual
        // replica of the sdpa_vector pair path at fixed kL = 512 (steady
        // ring: the 8-trip two-deep pipeline covers all 16 slots per
        // simdgroup with no tail).
        threadgroup U outputs[4 * BN * BDP];
        threadgroup U max_scores[2 * BN];
        threadgroup U sum_exp_scores[2 * BN];

        const device bfloat* pair_keys = k_cache +
            (size_t)kv_head * (window * head_dim) +
            (size_t)sg * head_dim + lane * qk_per_thread;
        const device bfloat* pair_values = v_cache +
            (size_t)kv_head * (window * head_dim) +
            (size_t)sg * head_dim + lane * v_per_thread;
        const int inner_k_stride = BN * int(head_dim);
        const int inner_v_stride = BN * int(head_dim);

        thread U pair_q0[qk_per_thread];
        thread U pair_q1[qk_per_thread];
        thread U pair_o0[v_per_thread];
        thread U pair_o1[v_per_thread];

        for (int j = 0; j < qk_per_thread; ++j) {
            pair_q0[j] =
                static_cast<U>(scale) * tg_q0[lane * qk_per_thread + j];
            pair_q1[j] =
                static_cast<U>(scale) * tg_q1[lane * qk_per_thread + j];
        }
        for (int j = 0; j < v_per_thread; ++j) {
            pair_o0[j] = 0;
            pair_o1[j] = 0;
        }

        U pair_max0 = metal::numeric_limits<U>::lowest();
        U pair_max1 = metal::numeric_limits<U>::lowest();
        U pair_sum0 = 0;
        U pair_sum1 = 0;

        int i = sg;
        for (; i + BN < N; i += 2 * BN) {
            const device bfloat* pipe_keys_b = pair_keys + inner_k_stride;
            const device bfloat* pipe_values_b = pair_values + inner_v_stride;
            const bool sub_a = uint(i) == widx;
            const bool sub_b = uint(i + BN) == widx;
            U pipe_ka[4];
            U pipe_kb[4];
            T_LOAD_K(pipe_ka, sub_a, pair_keys);
            T_LOAD_K(pipe_kb, sub_b, pipe_keys_b);
            bfloat pipe_va0, pipe_va1, pipe_va2, pipe_va3;
            bfloat pipe_vb0, pipe_vb1, pipe_vb2, pipe_vb3;
            T_LOAD_V(pipe_va0, pipe_va1, pipe_va2, pipe_va3, sub_a,
                pair_values);
            T_LOAD_V(pipe_vb0, pipe_vb1, pipe_vb2, pipe_vb3, sub_b,
                pipe_values_b);

            U pair_score0 = 0;
            U pair_score1 = 0;
            pair_score0 += pair_q0[0] * pipe_ka[0];
            pair_score1 += pair_q1[0] * pipe_ka[0];
            pair_score0 += pair_q0[1] * pipe_ka[1];
            pair_score1 += pair_q1[1] * pipe_ka[1];
            pair_score0 += pair_q0[2] * pipe_ka[2];
            pair_score1 += pair_q1[2] * pipe_ka[2];
            pair_score0 += pair_q0[3] * pipe_ka[3];
            pair_score1 += pair_q1[3] * pipe_ka[3];
            pair_score0 = simd_sum(pair_score0);
            pair_score1 = simd_sum(pair_score1);

            U pair_new_max0 = metal::max(pair_max0, pair_score0);
            U pair_new_max1 = metal::max(pair_max1, pair_score1);
            U pair_factor0;
            U pair_factor1;
            LAGUNA_RESCALE(pair_factor0, pair_max0 - pair_new_max0);
            LAGUNA_RESCALE(pair_factor1, pair_max1 - pair_new_max1);
            U pair_exp0 = metal::fast::exp(pair_score0 - pair_new_max0);
            U pair_exp1 = metal::fast::exp(pair_score1 - pair_new_max1);

            pair_max0 = pair_new_max0;
            pair_max1 = pair_new_max1;
            pair_sum0 = pair_sum0 * pair_factor0 + pair_exp0;
            pair_sum1 = pair_sum1 * pair_factor1 + pair_exp1;

            pair_o0[0] = pair_o0[0] * pair_factor0 + pair_exp0 * pipe_va0;
            pair_o1[0] = pair_o1[0] * pair_factor1 + pair_exp1 * pipe_va0;
            pair_o0[1] = pair_o0[1] * pair_factor0 + pair_exp0 * pipe_va1;
            pair_o1[1] = pair_o1[1] * pair_factor1 + pair_exp1 * pipe_va1;
            pair_o0[2] = pair_o0[2] * pair_factor0 + pair_exp0 * pipe_va2;
            pair_o1[2] = pair_o1[2] * pair_factor1 + pair_exp1 * pipe_va2;
            pair_o0[3] = pair_o0[3] * pair_factor0 + pair_exp0 * pipe_va3;
            pair_o1[3] = pair_o1[3] * pair_factor1 + pair_exp1 * pipe_va3;

            U pipeb_score0 = 0;
            U pipeb_score1 = 0;
            pipeb_score0 += pair_q0[0] * pipe_kb[0];
            pipeb_score1 += pair_q1[0] * pipe_kb[0];
            pipeb_score0 += pair_q0[1] * pipe_kb[1];
            pipeb_score1 += pair_q1[1] * pipe_kb[1];
            pipeb_score0 += pair_q0[2] * pipe_kb[2];
            pipeb_score1 += pair_q1[2] * pipe_kb[2];
            pipeb_score0 += pair_q0[3] * pipe_kb[3];
            pipeb_score1 += pair_q1[3] * pipe_kb[3];
            pipeb_score0 = simd_sum(pipeb_score0);
            pipeb_score1 = simd_sum(pipeb_score1);

            U pipeb_new_max0 = metal::max(pair_max0, pipeb_score0);
            U pipeb_new_max1 = metal::max(pair_max1, pipeb_score1);
            U pipeb_factor0;
            U pipeb_factor1;
            LAGUNA_RESCALE(pipeb_factor0, pair_max0 - pipeb_new_max0);
            LAGUNA_RESCALE(pipeb_factor1, pair_max1 - pipeb_new_max1);
            U pipeb_exp0 = metal::fast::exp(pipeb_score0 - pipeb_new_max0);
            U pipeb_exp1 = metal::fast::exp(pipeb_score1 - pipeb_new_max1);

            pair_max0 = pipeb_new_max0;
            pair_max1 = pipeb_new_max1;
            pair_sum0 = pair_sum0 * pipeb_factor0 + pipeb_exp0;
            pair_sum1 = pair_sum1 * pipeb_factor1 + pipeb_exp1;

            pair_o0[0] = pair_o0[0] * pipeb_factor0 + pipeb_exp0 * pipe_vb0;
            pair_o1[0] = pair_o1[0] * pipeb_factor1 + pipeb_exp1 * pipe_vb0;
            pair_o0[1] = pair_o0[1] * pipeb_factor0 + pipeb_exp0 * pipe_vb1;
            pair_o1[1] = pair_o1[1] * pipeb_factor1 + pipeb_exp1 * pipe_vb1;
            pair_o0[2] = pair_o0[2] * pipeb_factor0 + pipeb_exp0 * pipe_vb2;
            pair_o1[2] = pair_o1[2] * pipeb_factor1 + pipeb_exp1 * pipe_vb2;
            pair_o0[3] = pair_o0[3] * pipeb_factor0 + pipeb_exp0 * pipe_vb3;
            pair_o1[3] = pair_o1[3] * pipeb_factor1 + pipeb_exp1 * pipe_vb3;

            pair_keys += 2 * inner_k_stride;
            pair_values += 2 * inner_v_stride;
        }

        // Combine: promoted two-plane exchange, textual replica of the
        // sdpa_vector pair path epilogue.
        constexpr int pair_planes = 2;
        constexpr int pair_plane_size = BN * BDP;
        if (lane == 0) {
            max_scores[sg] = pair_max0;
            max_scores[BN + sg] = pair_max1;
            sum_exp_scores[sg] = pair_sum0;
            sum_exp_scores[BN + sg] = pair_sum1;
        }
        for (int p = 0; p < pair_planes; ++p) {
            outputs[p * pair_plane_size + lane * BDP + sg] = pair_o0[p];
            outputs[
                (pair_planes + p) * pair_plane_size + lane * BDP + sg] =
                pair_o1[p];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        pair_max0 = max_scores[lane];
        pair_max1 = max_scores[BN + lane];
        U pair_global_max0 = simd_max(pair_max0);
        U pair_global_max1 = simd_max(pair_max1);
        U pair_global_factor0 = metal::fast::exp(pair_max0 - pair_global_max0);
        U pair_global_factor1 = metal::fast::exp(pair_max1 - pair_global_max1);
        pair_sum0 = simd_sum(sum_exp_scores[lane] * pair_global_factor0);
        pair_sum1 = simd_sum(sum_exp_scores[BN + lane] * pair_global_factor1);

        for (int p = 0; p < pair_planes; ++p) {
            U acc0 = simd_sum(
                outputs[p * pair_plane_size + sg * BDP + lane] *
                pair_global_factor0);
            U acc1 = simd_sum(
                outputs[
                    (pair_planes + p) * pair_plane_size + sg * BDP + lane] *
                pair_global_factor1);
            pair_o0[p] = pair_sum0 == 0 ? acc0 : (acc0 / pair_sum0);
            pair_o1[p] = pair_sum1 == 0 ? acc1 : (acc1 / pair_sum1);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int p = 0; p < pair_planes; ++p) {
            outputs[p * pair_plane_size + lane * BDP + sg] =
                pair_o0[pair_planes + p];
            outputs[
                (pair_planes + p) * pair_plane_size + lane * BDP + sg] =
                pair_o1[pair_planes + p];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int p = 0; p < pair_planes; ++p) {
            U acc0 = simd_sum(
                outputs[p * pair_plane_size + sg * BDP + lane] *
                pair_global_factor0);
            U acc1 = simd_sum(
                outputs[
                    (pair_planes + p) * pair_plane_size + sg * BDP + lane] *
                pair_global_factor1);
            pair_o0[pair_planes + p] =
                pair_sum0 == 0 ? acc0 : (acc0 / pair_sum0);
            pair_o1[pair_planes + p] =
                pair_sum1 == 0 ? acc1 : (acc1 / pair_sum1);
        }

        if (lane == 0) {
            device bfloat* pair_out0 =
                attended + head0 * head_dim + sg * v_per_thread;
            device bfloat* pair_out1 =
                attended + head1 * head_dim + sg * v_per_thread;
            for (int p = 0; p < v_per_thread; ++p) {
                pair_out0[p] = static_cast<bfloat>(pair_o0[p]);
                pair_out1[p] = static_cast<bfloat>(pair_o1[p]);
            }
        }
        """,
    header: """
        // Alpha-skip rescale, replica of sdpa_vector.h's shipped
        // DARKBLOOM_RESCALE_FACTOR (DARKBLOOM_ALPHASKIP == 1 arm).
        #define LAGUNA_RESCALE(dst, delta_expr)         \\
          do {                                          \\
            const float db_delta_ = (delta_expr);       \\
            if (as_type<uint>(db_delta_) == 0u) {       \\
              dst = float(1.0f);                        \\
            } else {                                    \\
              dst = metal::fast::exp(db_delta_);        \\
            }                                           \\
          } while (false)

        // K loads: 8-byte vec loads from the ring, or the threadgroup
        // substitute for the just-written slot. Same elements, same order,
        // same bfloat -> float conversion points as the scalar form.
        #define T_LOAD_K(dst, substitute, ptr)                     \\
          do {                                                     \\
            if (substitute) {                                      \\
              dst[0] = tg_k[lane * qk_per_thread + 0];             \\
              dst[1] = tg_k[lane * qk_per_thread + 1];             \\
              dst[2] = tg_k[lane * qk_per_thread + 2];             \\
              dst[3] = tg_k[lane * qk_per_thread + 3];             \\
            } else {                                               \\
              const vec<bfloat, 4> v_ =                            \\
                  *reinterpret_cast<const device vec<bfloat, 4>*>( \\
                      ptr);                                        \\
              dst[0] = v_.x;                                       \\
              dst[1] = v_.y;                                       \\
              dst[2] = v_.z;                                       \\
              dst[3] = v_.w;                                       \\
            }                                                      \\
          } while (false)

        #define T_LOAD_V(d0, d1, d2, d3, substitute, ptr)          \\
          do {                                                     \\
            if (substitute) {                                      \\
              d0 = tg_v[lane * v_per_thread + 0];                  \\
              d1 = tg_v[lane * v_per_thread + 1];                  \\
              d2 = tg_v[lane * v_per_thread + 2];                  \\
              d3 = tg_v[lane * v_per_thread + 3];                  \\
            } else {                                               \\
              const vec<bfloat, 4> v_ =                            \\
                  *reinterpret_cast<const device vec<bfloat, 4>*>( \\
                      ptr);                                        \\
              d0 = v_.x;                                           \\
              d1 = v_.y;                                           \\
              d2 = v_.z;                                           \\
              d3 = v_.w;                                           \\
            }                                                      \\
          } while (false)

        // (trailing newline required: the JIT concatenates the generated
        // [[kernel]] signature directly after this header string)

        """,
    ensureRowContiguous: true
)

/// Fused decode attention for a sliding layer in the steady ring regime.
/// Returns `[1, heads, 1, headDim]` attended output; the caller advances the
/// cache clock via `RotatingKVCache.fusedRingAdvance()`.
func lagunaSlidingFusedAttention(
    rawQueries: MLXArray,
    rawKeys: MLXArray,
    rawValues: MLXArray,
    queryWeight: MLXArray,
    keyWeight: MLXArray,
    angles: MLXArray,
    cacheKeys: MLXArray,
    cacheValues: MLXArray,
    writeIdx: Int,
    scale: MLXArray
) -> MLXArray {
    let heads = LagunaConstants.slidingAttentionHeads
    let kvHeads = LagunaConstants.numKeyValueHeads
    let window = LagunaConstants.slidingWindow
    precondition(rawQueries.dtype == .bfloat16)
    precondition(rawKeys.dtype == .bfloat16)
    precondition(rawValues.dtype == .bfloat16)
    precondition(rawQueries.dims(1, 1, heads * LagunaConstants.headDim))
    precondition(rawKeys.dims(1, 1, kvHeads * LagunaConstants.headDim))
    precondition(rawValues.dims(1, 1, kvHeads * LagunaConstants.headDim))
    precondition(queryWeight.dims(LagunaConstants.headDim))
    precondition(keyWeight.dims(LagunaConstants.headDim))
    precondition(angles.dtype == .float32)
    precondition(angles.dims(1, 1, 1, LagunaConstants.headDim))
    precondition(cacheKeys.dtype == .bfloat16)
    precondition(
        cacheKeys.dims(1, kvHeads, window, LagunaConstants.headDim))
    precondition(
        cacheValues.dims(1, kvHeads, window, LagunaConstants.headDim))
    precondition(writeIdx >= 0 && writeIdx < window)
    precondition(scale.dtype == .float32 && scale.size == 1)

    lagunaTrace("sliding fused attention")
    let params = lagunaParamsAtlasEnabled
        ? lagunaRingIdxAtlas[writeIdx] : MLXArray([UInt32(writeIdx)])
    return lagunaSlidingFusedAttentionKernel(
        [
            rawQueries, rawKeys, rawValues,
            queryWeight, keyWeight, angles,
            cacheKeys, cacheValues, params, scale,
        ],
        grid: ((heads / 2) * 1024, 1, 1),
        threadGroup: (1024, 1, 1),
        outputShapes: [[1, heads, 1, LagunaConstants.headDim]],
        outputDTypes: [.bfloat16]
    )[0]
}

/// Pre-materialized 4-byte uniform buffers for every possible sliding ring
/// write index (2 KB total, built on first touch during untimed warmup):
/// replaces a fresh 1-element MLXArray allocation per sliding-attention call
/// (30/step). Input-independent: all 512 values built unconditionally; the
/// lookup indexes by the cache's ring position (request-local state), the
/// same contract as the RoPE angle atlases. Worker decode is
/// single-threaded, so the write-once unsafe opt-out is sound.
private enum LagunaRingIdxAtlasStore {
    nonisolated(unsafe) static let entries: [MLXArray] = {
        let atlas = (0..<LagunaConstants.slidingWindow).map {
            MLXArray([UInt32($0)])
        }
        for entry in atlas { eval(entry) }
        return atlas
    }()
}

private var lagunaRingIdxAtlas: [MLXArray] { LagunaRingIdxAtlasStore.entries }

/// `DARKBLOOM_PARAMS_ATLAS=0` restores the per-call fresh 1-element array
/// (ablation control for the atlas above; identical bytes either way).
let lagunaParamsAtlasEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PARAMS_ATLAS"] != "0"

/// Pre-materialized 12-byte uniform buffers for the full-attention decode
/// kernel's 3-element params array ([writeIdx, writeIdx+1, capacity]). Unlike
/// the sliding ring atlas, the values depend on the runtime cache capacity
/// and the absolute write index, so the atlas is built lazily on the first
/// fused full-attention call and rebuilt if the capacity changes. Worker
/// decode is single-threaded, so the write-once unsafe opt-out is sound.
private enum LagunaFullIdxAtlasStore {
    nonisolated(unsafe) private static var entries: [MLXArray] = []
    nonisolated(unsafe) private static var baseWriteIdx: Int = -1
    nonisolated(unsafe) private static var capacity: Int = -1
    private static let count = 128

    static func lookup(writeIdx: Int, capacity: Int) -> MLXArray? {
        if self.capacity != capacity {
            build(baseWriteIdx: writeIdx, capacity: capacity)
        }
        let offset = writeIdx - baseWriteIdx
        guard offset >= 0, offset < entries.count else { return nil }
        return entries[offset]
    }

    private static func build(baseWriteIdx: Int, capacity: Int) {
        let atlas = (0..<count).map {
            MLXArray([
                UInt32(baseWriteIdx + $0),
                UInt32(baseWriteIdx + $0 + 1),
                UInt32(capacity),
            ])
        }
        for entry in atlas { eval(entry) }
        entries = atlas
        self.baseWriteIdx = baseWriteIdx
        self.capacity = capacity
    }
}

/// `DARKBLOOM_FULL_PARAMS_ATLAS=0` restores the per-call fresh 3-element
/// array (ablation control; identical bytes either way).
let lagunaFullIdxAtlasEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FULL_PARAMS_ATLAS"] != "0"

/// `DARKBLOOM_FUSED_FULL_ATTN` (default on; set "0" to disable): decode
/// fused attention for the ten full-attention layers once the cache backing
/// has spare capacity (from the second decode step on; the first step's
/// stock growth concat is kept). Same design as the sliding twin above —
/// ONE dispatch replaces [QK-norm+YaRN kernel] -> [K slice-assign] ->
/// [V slice-assign] -> [sdpa_vector] — with the full-attention phase-1 text
/// (textual replica of `laguna_full_qk_norm_yarn_bf16_128_v4`: 64-dim
/// partial rotary, folded mscale roundings, passthrough tail) and the
/// pair path's runtime-length loop + single-row tail at gqa_factor 6.
let lagunaFusedFullAttentionEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_FULL_ATTN"] != "0"

/// Diagnostic-only coupling control for the historical second whole-model
/// constructor decode. Full-attention fusion no longer implies this rewarm;
/// set the flag explicitly only when reproducing the retired bundled arm.
let lagunaFusedFullAttentionWholeModelWarmupEnabled =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_FUSED_FULL_ATTN_WHOLE_MODEL_WARMUP"] == "1"

/// Compile only the full-attention custom kernel during untimed construction,
/// using tiny throwaway arrays. Unlike the retired whole-model rewarm, this
/// does not execute another Laguna layer or retain request/cache state.
let lagunaFusedFullAttentionKernelWarmupEnabled =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_FUSED_FULL_ATTN_KERNEL_WARMUP"] != "0"

private let lagunaFullFusedAttentionKernel = MLXFast.metalKernel(
    name: "laguna_full_fused_attn_grow_v1",
    inputNames: [
        "raw_queries", "raw_keys", "raw_values",
        "query_weight", "key_weight", "angles",
        "k_cache", "v_cache", "params", "scale_arr",
    ],
    outputNames: ["attended"],
    source: """
        constexpr uint head_dim = 128;
        constexpr uint gqa = 6;
        constexpr int BN = 32;
        constexpr int BD = 32;
        // Pad the epilogue exchange stride off a power of two: the
        // transposing write below is `lane * stride + sg`, which at
        // stride 32 puts all 32 lanes of a simdgroup in one threadgroup
        // memory bank. The odd stride keeps the read contiguous.
        constexpr int BDP = BD + 1;
        constexpr int qk_per_thread = 4;
        constexpr int v_per_thread = 4;
        constexpr uint rotary_pairs = 32;
        constexpr float yarn_mscale = 1.3465735912322998f;

        typedef float U;

        uint pair_tg = threadgroup_position_in_grid.x;
        uint head0 = pair_tg * 2;
        uint head1 = head0 + 1;
        uint kv_head = head0 / gqa;
        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint widx = params[0];
        int N = int(params[1]);
        uint capacity = params[2];
        float scale = scale_arr[0];

        threadgroup bfloat tg_q0[head_dim];
        threadgroup bfloat tg_q1[head_dim];
        threadgroup bfloat tg_k[head_dim];
        threadgroup bfloat tg_v[head_dim];

        // Phase 1: per-head RMSNorm + partial YaRN RoPE, textual replica of
        // laguna_full_qk_norm_yarn_bf16_128_v4 with the device row writes
        // retargeted at threadgroup memory.
        if (sg < 3) {
            const device bfloat* input =
                sg == 0 ? raw_queries + head0 * head_dim
                : sg == 1 ? raw_queries + head1 * head_dim
                          : raw_keys + kv_head * head_dim;
            const device bfloat* weight =
                sg == 2 ? key_weight : query_weight;
            threadgroup bfloat* outrow =
                sg == 0 ? tg_q0 : sg == 1 ? tg_q1 : tg_k;

            uint base = lane * 4;
            thread bfloat normalized[4];
            float sum = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                float value = float(input[base + i]);
                sum += value * value;
            }
            sum = simd_sum(sum);
            float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);
            for (uint i = 0; i < 4; ++i) {
                normalized[i] =
                    weight[base + i] *
                    bfloat(float(input[base + i]) * inverse_rms);
            }
            thread float paired[4];
            for (uint i = 0; i < 4; ++i) {
                paired[i] = simd_shuffle(float(normalized[i]), lane ^ 8);
            }
            if (lane < 8) {
                bfloat rounded_mscale = bfloat(yarn_mscale);
                for (uint i = 0; i < 4; ++i) {
                    uint pair = base + i;
                    float first =
                        float(bfloat(normalized[i] * rounded_mscale));
                    float second =
                        float(bfloat(bfloat(paired[i]) * rounded_mscale));
                    float cosine = angles[pair];
                    float sine = angles[pair + rotary_pairs];
                    outrow[pair] = bfloat(first * cosine - second * sine);
                    outrow[pair + rotary_pairs] =
                        bfloat(first * sine + second * cosine);
                }
            } else if (lane >= 16) {
                for (uint i = 0; i < 4; ++i) {
                    outrow[base + i] = normalized[i];
                }
            }
        } else if (sg == 3) {
            const device bfloat* vin = raw_values + kv_head * head_dim;
            for (uint i = lane; i < head_dim; i += 32) {
                tg_v[i] = vin[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: one writer threadgroup per KV head persists the new row.
        if ((head0 % gqa) == 0 && sg == 0) {
            device bfloat* kc = (device bfloat*)k_cache +
                (size_t)kv_head * (capacity * head_dim) +
                (size_t)widx * head_dim;
            device bfloat* vc = (device bfloat*)v_cache +
                (size_t)kv_head * (capacity * head_dim) +
                (size_t)widx * head_dim;
            for (uint i = lane; i < head_dim; i += 32) {
                kc[i] = tg_k[i];
                vc[i] = tg_v[i];
            }
        }

        // Phase 3: GQA-pair attention over the first N rows in slot order,
        // textual replica of the sdpa_vector pair path (runtime N, tail
        // row included).
        threadgroup U outputs[4 * BN * BDP];
        threadgroup U max_scores[2 * BN];
        threadgroup U sum_exp_scores[2 * BN];

        const device bfloat* pair_keys = k_cache +
            (size_t)kv_head * (capacity * head_dim) +
            (size_t)sg * head_dim + lane * qk_per_thread;
        const device bfloat* pair_values = v_cache +
            (size_t)kv_head * (capacity * head_dim) +
            (size_t)sg * head_dim + lane * v_per_thread;
        const int inner_k_stride = BN * int(head_dim);
        const int inner_v_stride = BN * int(head_dim);

        thread U pair_q0[qk_per_thread];
        thread U pair_q1[qk_per_thread];
        thread U pair_k[qk_per_thread];
        thread U pair_o0[v_per_thread];
        thread U pair_o1[v_per_thread];

        for (int j = 0; j < qk_per_thread; ++j) {
            pair_q0[j] =
                static_cast<U>(scale) * tg_q0[lane * qk_per_thread + j];
            pair_q1[j] =
                static_cast<U>(scale) * tg_q1[lane * qk_per_thread + j];
        }
        for (int j = 0; j < v_per_thread; ++j) {
            pair_o0[j] = 0;
            pair_o1[j] = 0;
        }

        U pair_max0 = metal::numeric_limits<U>::lowest();
        U pair_max1 = metal::numeric_limits<U>::lowest();
        U pair_sum0 = 0;
        U pair_sum1 = 0;

        int i = sg;
        for (; i + BN < N; i += 2 * BN) {
            const device bfloat* pipe_keys_b = pair_keys + inner_k_stride;
            const device bfloat* pipe_values_b = pair_values + inner_v_stride;
            const bool sub_a = uint(i) == widx;
            const bool sub_b = uint(i + BN) == widx;
            U pipe_ka[4];
            U pipe_kb[4];
            T_LOAD_K(pipe_ka, sub_a, pair_keys);
            T_LOAD_K(pipe_kb, sub_b, pipe_keys_b);
            bfloat pipe_va0, pipe_va1, pipe_va2, pipe_va3;
            bfloat pipe_vb0, pipe_vb1, pipe_vb2, pipe_vb3;
            T_LOAD_V(pipe_va0, pipe_va1, pipe_va2, pipe_va3, sub_a,
                pair_values);
            T_LOAD_V(pipe_vb0, pipe_vb1, pipe_vb2, pipe_vb3, sub_b,
                pipe_values_b);

            U pair_score0 = 0;
            U pair_score1 = 0;
            pair_score0 += pair_q0[0] * pipe_ka[0];
            pair_score1 += pair_q1[0] * pipe_ka[0];
            pair_score0 += pair_q0[1] * pipe_ka[1];
            pair_score1 += pair_q1[1] * pipe_ka[1];
            pair_score0 += pair_q0[2] * pipe_ka[2];
            pair_score1 += pair_q1[2] * pipe_ka[2];
            pair_score0 += pair_q0[3] * pipe_ka[3];
            pair_score1 += pair_q1[3] * pipe_ka[3];
            pair_score0 = simd_sum(pair_score0);
            pair_score1 = simd_sum(pair_score1);

            U pair_new_max0 = metal::max(pair_max0, pair_score0);
            U pair_new_max1 = metal::max(pair_max1, pair_score1);
            U pair_factor0;
            U pair_factor1;
            LAGUNA_RESCALE(pair_factor0, pair_max0 - pair_new_max0);
            LAGUNA_RESCALE(pair_factor1, pair_max1 - pair_new_max1);
            U pair_exp0 = metal::fast::exp(pair_score0 - pair_new_max0);
            U pair_exp1 = metal::fast::exp(pair_score1 - pair_new_max1);

            pair_max0 = pair_new_max0;
            pair_max1 = pair_new_max1;
            pair_sum0 = pair_sum0 * pair_factor0 + pair_exp0;
            pair_sum1 = pair_sum1 * pair_factor1 + pair_exp1;

            pair_o0[0] = pair_o0[0] * pair_factor0 + pair_exp0 * pipe_va0;
            pair_o1[0] = pair_o1[0] * pair_factor1 + pair_exp1 * pipe_va0;
            pair_o0[1] = pair_o0[1] * pair_factor0 + pair_exp0 * pipe_va1;
            pair_o1[1] = pair_o1[1] * pair_factor1 + pair_exp1 * pipe_va1;
            pair_o0[2] = pair_o0[2] * pair_factor0 + pair_exp0 * pipe_va2;
            pair_o1[2] = pair_o1[2] * pair_factor1 + pair_exp1 * pipe_va2;
            pair_o0[3] = pair_o0[3] * pair_factor0 + pair_exp0 * pipe_va3;
            pair_o1[3] = pair_o1[3] * pair_factor1 + pair_exp1 * pipe_va3;

            U pipeb_score0 = 0;
            U pipeb_score1 = 0;
            pipeb_score0 += pair_q0[0] * pipe_kb[0];
            pipeb_score1 += pair_q1[0] * pipe_kb[0];
            pipeb_score0 += pair_q0[1] * pipe_kb[1];
            pipeb_score1 += pair_q1[1] * pipe_kb[1];
            pipeb_score0 += pair_q0[2] * pipe_kb[2];
            pipeb_score1 += pair_q1[2] * pipe_kb[2];
            pipeb_score0 += pair_q0[3] * pipe_kb[3];
            pipeb_score1 += pair_q1[3] * pipe_kb[3];
            pipeb_score0 = simd_sum(pipeb_score0);
            pipeb_score1 = simd_sum(pipeb_score1);

            U pipeb_new_max0 = metal::max(pair_max0, pipeb_score0);
            U pipeb_new_max1 = metal::max(pair_max1, pipeb_score1);
            U pipeb_factor0;
            U pipeb_factor1;
            LAGUNA_RESCALE(pipeb_factor0, pair_max0 - pipeb_new_max0);
            LAGUNA_RESCALE(pipeb_factor1, pair_max1 - pipeb_new_max1);
            U pipeb_exp0 = metal::fast::exp(pipeb_score0 - pipeb_new_max0);
            U pipeb_exp1 = metal::fast::exp(pipeb_score1 - pipeb_new_max1);

            pair_max0 = pipeb_new_max0;
            pair_max1 = pipeb_new_max1;
            pair_sum0 = pair_sum0 * pipeb_factor0 + pipeb_exp0;
            pair_sum1 = pair_sum1 * pipeb_factor1 + pipeb_exp1;

            pair_o0[0] = pair_o0[0] * pipeb_factor0 + pipeb_exp0 * pipe_vb0;
            pair_o1[0] = pair_o1[0] * pipeb_factor1 + pipeb_exp1 * pipe_vb0;
            pair_o0[1] = pair_o0[1] * pipeb_factor0 + pipeb_exp0 * pipe_vb1;
            pair_o1[1] = pair_o1[1] * pipeb_factor1 + pipeb_exp1 * pipe_vb1;
            pair_o0[2] = pair_o0[2] * pipeb_factor0 + pipeb_exp0 * pipe_vb2;
            pair_o1[2] = pair_o1[2] * pipeb_factor1 + pipeb_exp1 * pipe_vb2;
            pair_o0[3] = pair_o0[3] * pipeb_factor0 + pipeb_exp0 * pipe_vb3;
            pair_o1[3] = pair_o1[3] * pipeb_factor1 + pipeb_exp1 * pipe_vb3;

            pair_keys += 2 * inner_k_stride;
            pair_values += 2 * inner_v_stride;
        }
        if (i < N) {
            const bool sub_t = uint(i) == widx;
            T_LOAD_K(pair_k, sub_t, pair_keys);
            bfloat pipe_va0, pipe_va1, pipe_va2, pipe_va3;
            T_LOAD_V(pipe_va0, pipe_va1, pipe_va2, pipe_va3, sub_t,
                pair_values);

            U pair_score0 = 0;
            U pair_score1 = 0;
            pair_score0 += pair_q0[0] * pair_k[0];
            pair_score1 += pair_q1[0] * pair_k[0];
            pair_score0 += pair_q0[1] * pair_k[1];
            pair_score1 += pair_q1[1] * pair_k[1];
            pair_score0 += pair_q0[2] * pair_k[2];
            pair_score1 += pair_q1[2] * pair_k[2];
            pair_score0 += pair_q0[3] * pair_k[3];
            pair_score1 += pair_q1[3] * pair_k[3];
            pair_score0 = simd_sum(pair_score0);
            pair_score1 = simd_sum(pair_score1);

            U pair_new_max0 = metal::max(pair_max0, pair_score0);
            U pair_new_max1 = metal::max(pair_max1, pair_score1);
            U pair_factor0;
            U pair_factor1;
            LAGUNA_RESCALE(pair_factor0, pair_max0 - pair_new_max0);
            LAGUNA_RESCALE(pair_factor1, pair_max1 - pair_new_max1);
            U pair_exp0 = metal::fast::exp(pair_score0 - pair_new_max0);
            U pair_exp1 = metal::fast::exp(pair_score1 - pair_new_max1);

            pair_max0 = pair_new_max0;
            pair_max1 = pair_new_max1;
            pair_sum0 = pair_sum0 * pair_factor0 + pair_exp0;
            pair_sum1 = pair_sum1 * pair_factor1 + pair_exp1;

            pair_o0[0] = pair_o0[0] * pair_factor0 + pair_exp0 * pipe_va0;
            pair_o1[0] = pair_o1[0] * pair_factor1 + pair_exp1 * pipe_va0;
            pair_o0[1] = pair_o0[1] * pair_factor0 + pair_exp0 * pipe_va1;
            pair_o1[1] = pair_o1[1] * pair_factor1 + pair_exp1 * pipe_va1;
            pair_o0[2] = pair_o0[2] * pair_factor0 + pair_exp0 * pipe_va2;
            pair_o1[2] = pair_o1[2] * pair_factor1 + pair_exp1 * pipe_va2;
            pair_o0[3] = pair_o0[3] * pair_factor0 + pair_exp0 * pipe_va3;
            pair_o1[3] = pair_o1[3] * pair_factor1 + pair_exp1 * pipe_va3;
        }

        // Combine: promoted two-plane exchange, textual replica of the
        // sdpa_vector pair path epilogue.
        constexpr int pair_planes = 2;
        constexpr int pair_plane_size = BN * BDP;
        if (lane == 0) {
            max_scores[sg] = pair_max0;
            max_scores[BN + sg] = pair_max1;
            sum_exp_scores[sg] = pair_sum0;
            sum_exp_scores[BN + sg] = pair_sum1;
        }
        for (int p = 0; p < pair_planes; ++p) {
            outputs[p * pair_plane_size + lane * BDP + sg] = pair_o0[p];
            outputs[
                (pair_planes + p) * pair_plane_size + lane * BDP + sg] =
                pair_o1[p];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        pair_max0 = max_scores[lane];
        pair_max1 = max_scores[BN + lane];
        U pair_global_max0 = simd_max(pair_max0);
        U pair_global_max1 = simd_max(pair_max1);
        U pair_global_factor0 = metal::fast::exp(pair_max0 - pair_global_max0);
        U pair_global_factor1 = metal::fast::exp(pair_max1 - pair_global_max1);
        pair_sum0 = simd_sum(sum_exp_scores[lane] * pair_global_factor0);
        pair_sum1 = simd_sum(sum_exp_scores[BN + lane] * pair_global_factor1);

        for (int p = 0; p < pair_planes; ++p) {
            U acc0 = simd_sum(
                outputs[p * pair_plane_size + sg * BDP + lane] *
                pair_global_factor0);
            U acc1 = simd_sum(
                outputs[
                    (pair_planes + p) * pair_plane_size + sg * BDP + lane] *
                pair_global_factor1);
            pair_o0[p] = pair_sum0 == 0 ? acc0 : (acc0 / pair_sum0);
            pair_o1[p] = pair_sum1 == 0 ? acc1 : (acc1 / pair_sum1);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int p = 0; p < pair_planes; ++p) {
            outputs[p * pair_plane_size + lane * BDP + sg] =
                pair_o0[pair_planes + p];
            outputs[
                (pair_planes + p) * pair_plane_size + lane * BDP + sg] =
                pair_o1[pair_planes + p];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int p = 0; p < pair_planes; ++p) {
            U acc0 = simd_sum(
                outputs[p * pair_plane_size + sg * BDP + lane] *
                pair_global_factor0);
            U acc1 = simd_sum(
                outputs[
                    (pair_planes + p) * pair_plane_size + sg * BDP + lane] *
                pair_global_factor1);
            pair_o0[pair_planes + p] =
                pair_sum0 == 0 ? acc0 : (acc0 / pair_sum0);
            pair_o1[pair_planes + p] =
                pair_sum1 == 0 ? acc1 : (acc1 / pair_sum1);
        }

        if (lane == 0) {
            device bfloat* pair_out0 =
                attended + head0 * head_dim + sg * v_per_thread;
            device bfloat* pair_out1 =
                attended + head1 * head_dim + sg * v_per_thread;
            for (int p = 0; p < v_per_thread; ++p) {
                pair_out0[p] = static_cast<bfloat>(pair_o0[p]);
                pair_out1[p] = static_cast<bfloat>(pair_o1[p]);
            }
        }
        """,
    header: """
        #define LAGUNA_RESCALE(dst, delta_expr)         \\
          do {                                          \\
            const float db_delta_ = (delta_expr);       \\
            if (as_type<uint>(db_delta_) == 0u) {       \\
              dst = float(1.0f);                        \\
            } else {                                    \\
              dst = metal::fast::exp(db_delta_);        \\
            }                                           \\
          } while (false)

        #define T_LOAD_K(dst, substitute, ptr)                     \\
          do {                                                     \\
            if (substitute) {                                      \\
              dst[0] = tg_k[lane * qk_per_thread + 0];             \\
              dst[1] = tg_k[lane * qk_per_thread + 1];             \\
              dst[2] = tg_k[lane * qk_per_thread + 2];             \\
              dst[3] = tg_k[lane * qk_per_thread + 3];             \\
            } else {                                               \\
              const vec<bfloat, 4> v_ =                            \\
                  *reinterpret_cast<const device vec<bfloat, 4>*>( \\
                      ptr);                                        \\
              dst[0] = v_.x;                                       \\
              dst[1] = v_.y;                                       \\
              dst[2] = v_.z;                                       \\
              dst[3] = v_.w;                                       \\
            }                                                      \\
          } while (false)

        #define T_LOAD_V(d0, d1, d2, d3, substitute, ptr)          \\
          do {                                                     \\
            if (substitute) {                                      \\
              d0 = tg_v[lane * v_per_thread + 0];                  \\
              d1 = tg_v[lane * v_per_thread + 1];                  \\
              d2 = tg_v[lane * v_per_thread + 2];                  \\
              d3 = tg_v[lane * v_per_thread + 3];                  \\
            } else {                                               \\
              const vec<bfloat, 4> v_ =                            \\
                  *reinterpret_cast<const device vec<bfloat, 4>*>( \\
                      ptr);                                        \\
              d0 = v_.x;                                           \\
              d1 = v_.y;                                           \\
              d2 = v_.z;                                           \\
              d3 = v_.w;                                           \\
            }                                                      \\
          } while (false)

        // (trailing newline required: the JIT concatenates the generated
        // [[kernel]] signature directly after this header string)

        """,
    ensureRowContiguous: true
)

/// Fused decode attention for a full-attention layer with spare backing
/// capacity. Returns `[1, heads, 1, headDim]`; the caller advances the
/// cache clock via `KVCacheSimple.fusedAppendAdvance()`.
func lagunaFullFusedAttention(
    rawQueries: MLXArray,
    rawKeys: MLXArray,
    rawValues: MLXArray,
    queryWeight: MLXArray,
    keyWeight: MLXArray,
    angles: MLXArray,
    cacheKeys: MLXArray,
    cacheValues: MLXArray,
    writeIdx: Int,
    scale: MLXArray
) -> MLXArray {
    let heads = LagunaConstants.fullAttentionHeads
    let kvHeads = LagunaConstants.numKeyValueHeads
    let capacity = cacheKeys.dim(2)
    precondition(rawQueries.dtype == .bfloat16)
    precondition(rawKeys.dtype == .bfloat16)
    precondition(rawValues.dtype == .bfloat16)
    precondition(rawQueries.dims(1, 1, heads * LagunaConstants.headDim))
    precondition(rawKeys.dims(1, 1, kvHeads * LagunaConstants.headDim))
    precondition(rawValues.dims(1, 1, kvHeads * LagunaConstants.headDim))
    precondition(queryWeight.dims(LagunaConstants.headDim))
    precondition(keyWeight.dims(LagunaConstants.headDim))
    precondition(angles.dtype == .float32)
    precondition(angles.dims(1, 1, 1, LagunaConstants.headDim / 2))
    precondition(cacheKeys.dtype == .bfloat16)
    precondition(
        cacheKeys.dims(1, kvHeads, capacity, LagunaConstants.headDim))
    precondition(
        cacheValues.dims(1, kvHeads, capacity, LagunaConstants.headDim))
    precondition(writeIdx >= 0 && writeIdx < capacity)
    precondition(scale.dtype == .float32 && scale.size == 1)

    lagunaTrace("full fused attention")
    let params = lagunaFullIdxAtlasEnabled
        ? (LagunaFullIdxAtlasStore.lookup(writeIdx: writeIdx, capacity: capacity)
            ?? MLXArray([UInt32(writeIdx), UInt32(writeIdx + 1), UInt32(capacity)]))
        : MLXArray([UInt32(writeIdx), UInt32(writeIdx + 1), UInt32(capacity)])
    return lagunaFullFusedAttentionKernel(
        [
            rawQueries, rawKeys, rawValues,
            queryWeight, keyWeight, angles,
            cacheKeys, cacheValues, params, scale,
        ],
        grid: ((heads / 2) * 1024, 1, 1),
        threadGroup: (1024, 1, 1),
        outputShapes: [[1, heads, 1, LagunaConstants.headDim]],
        outputDTypes: [.bfloat16]
    )[0]
}

/// Force creation of `lagunaFullFusedAttentionKernel`'s pipeline state with
/// production Q/K/V geometry and a minimal two-row cache. Every tensor is
/// deterministic, input-independent, evaluated once, and released before the
/// constructor clears transient allocator cache and wires resident weights.
func lagunaWarmFullFusedAttentionKernel() {
    let headDim = LagunaConstants.headDim
    let heads = LagunaConstants.fullAttentionHeads
    let kvHeads = LagunaConstants.numKeyValueHeads
    let rawQueries = MLXArray.zeros(
        [1, 1, heads * headDim], dtype: .bfloat16)
    let rawKeys = MLXArray.zeros(
        [1, 1, kvHeads * headDim], dtype: .bfloat16)
    let rawValues = MLXArray.zeros(
        [1, 1, kvHeads * headDim], dtype: .bfloat16)
    let queryWeight = MLXArray.ones([headDim], dtype: .bfloat16)
    let keyWeight = MLXArray.ones([headDim], dtype: .bfloat16)
    let angles = MLXArray.zeros(
        [1, 1, 1, headDim / 2], dtype: .float32)
    let cacheKeys = MLXArray.zeros(
        [1, kvHeads, 2, headDim], dtype: .bfloat16)
    let cacheValues = MLXArray.zeros(
        [1, kvHeads, 2, headDim], dtype: .bfloat16)
    let scale = MLXArray([pow(Float(headDim), -0.5)])
    eval(lagunaFullFusedAttention(
        rawQueries: rawQueries,
        rawKeys: rawKeys,
        rawValues: rawValues,
        queryWeight: queryWeight,
        keyWeight: keyWeight,
        angles: angles,
        cacheKeys: cacheKeys,
        cacheValues: cacheValues,
        writeIdx: 1,
        scale: scale
    ))
}

/// Multi-token sliding-layer Q/K RMSNorm + plain RoPE fusion. One dispatch
/// replaces the stock four (`rms_single_row` q, `rms_single_row` k,
/// `rope_bfloat16` q, `rope_bfloat16` k) for a whole prefill layer and
/// writes both outputs directly in the `[1, heads, L, 128]` layout SDPA and
/// the cache update consume, so the transposed-view round trip is gone too.
///
/// Grid mapping: one threadgroup of 128 threads (four simdgroups) per
/// (head-block, token); each simdgroup owns one (token, head) row, exactly
/// the row `rms_single_row` gets at axis_size 128 (N_READS 4, one 32-lane
/// simdgroup per row). `threadgroups_per_grid.y` is L, so no shape constant
/// is baked and any L dispatches the same compiled kernel.
///
/// Exactness, link for link with the stock chain:
///  * The norm replicates `rms_single_row` at axis_size 128: lane `l` owns
///    the contiguous block `[4l, 4l+4)`, squares in index order, `simd_sum`
///    (the stock single-simdgroup threadgroup then sums `local_sums`, which
///    adds only zeros), `precise::rsqrt(acc / 128 + 1e-6)`, and the BF16
///    rounding inside `w[i] * bfloat(x[i] * inv)` — the same expression the
///    shipped decode kernels use against the same stock kernel.
///  * The rotation replicates `rope_impl<T, _, 4>` (`rope_bfloat16`,
///    non-traditional, dims 128): pair `p` couples `p` and `p + 64`, and the
///    cos/sin floats are read from the probe-seed atlas row for the token's
///    absolute position — values the stock RoPE kernel computed, not a
///    re-derivation. The decode twin (`laguna_sliding_qk_norm_rope_bf16_128_v1`)
///    consumes the same table with the same expression.
private let lagunaPrefillSlidingQKNormRoPEKernel = MLXFast.metalKernel(
    name: "laguna_prefill_sliding_qk_norm_rope_bf16_128_v2",
    inputNames: [
        "raw_queries", "raw_keys", "query_weight", "key_weight", "angles",
        "offsets",
    ],
    outputNames: ["queries", "keys"],
    source: """
        constexpr uint head_dim = 128;
        constexpr uint rotary_pairs = 64;
        constexpr uint query_heads = 64;
        constexpr uint kv_heads = 8;

        uint t = threadgroup_position_in_grid.y;
        uint length = threadgroups_per_grid.y;
        uint head = threadgroup_position_in_grid.x * 4
            + simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device bfloat* input;
        const device bfloat* weight;
        device bfloat* output;
        if (head < query_heads) {
            input = raw_queries + (t * query_heads + head) * head_dim;
            weight = query_weight;
            output = queries + (head * length + t) * head_dim;
        } else {
            uint khead = head - query_heads;
            input = raw_keys + (t * kv_heads + khead) * head_dim;
            weight = key_weight;
            output = keys + (khead * length + t) * head_dim;
        }

        uint base = lane * 4;
        thread bfloat normalized[4];
        float sum = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            float value = float(input[base + i]);
            sum += value * value;
        }
        // `simd_sum` already returns the total to every lane, so each lane
        // derives the same `precise::rsqrt` locally, matching the shipped
        // decode kernels. The stock single-simdgroup threadgroup's extra
        // `local_sums` round adds only zeros and cannot change the total.
        sum = simd_sum(sum);
        float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            normalized[i] =
                weight[base + i] *
                bfloat(float(input[base + i]) * inverse_rms);
        }

        // Element `p + 64`, the partner of pair `p`, lives 16 lanes away.
        // Every fixed-four loop in this prefill-only kernel is explicitly
        // scalarized. This removes loop-control ALU while preserving the
        // exact source order of the dependent RMS sum and rotary arithmetic.
        thread float paired[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            paired[i] = simd_shuffle(float(normalized[i]), lane ^ 16);
        }

        const device float* angle_row =
            angles + (uint(offsets[0]) + t) * (2 * rotary_pairs);
        // Every element rotates, so the lower sixteen lanes own all 64
        // pairs and write both halves of each.
        if (lane < 16) {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                uint pair = base + i;
                float first = float(normalized[i]);
                float second = paired[i];
                float cosine = angle_row[pair];
                float sine = angle_row[pair + rotary_pairs];
                output[pair] = bfloat(first * cosine - second * sine);
                output[pair + rotary_pairs] =
                    bfloat(first * sine + second * cosine);
            }
        }
        """,
    ensureRowContiguous: true
)

/// One-head-per-threadgroup twin of the prefill sliding QK-norm+RoPE kernel
/// (H1). BIT-EXACT repartition in the EG256 class: the `*4` original packs
/// four heads (four SIMDs) per threadgroup; this twin packs one (one SIMD).
/// Each head's work is fully SIMD-local -- RMSNorm `simd_sum`, the within-SIMD
/// `lane ^ 16` rotary-partner shuffle, and every BF16 rounding boundary never
/// cross heads -- so the value each head writes is identical regardless of how
/// many heads share a threadgroup. This matches the proven DECODE shape
/// (`lagunaSlidingQKNormRoPEKernel`, one SIMD/head, the project's largest
/// single win); the prefill `*4` was an unaudited divergence from it.
private let lagunaPrefillSlidingQKNormRoPEH1Kernel = MLXFast.metalKernel(
    name: "laguna_prefill_sliding_qk_norm_rope_bf16_128_h1_v2",
    inputNames: [
        "raw_queries", "raw_keys", "query_weight", "key_weight", "angles",
        "offsets",
    ],
    outputNames: ["queries", "keys"],
    source: """
        constexpr uint head_dim = 128;
        constexpr uint rotary_pairs = 64;
        constexpr uint query_heads = 64;
        constexpr uint kv_heads = 8;

        uint t = threadgroup_position_in_grid.y;
        uint length = threadgroups_per_grid.y;
        uint head = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;

        const device bfloat* input;
        const device bfloat* weight;
        device bfloat* output;
        if (head < query_heads) {
            input = raw_queries + (t * query_heads + head) * head_dim;
            weight = query_weight;
            output = queries + (head * length + t) * head_dim;
        } else {
            uint khead = head - query_heads;
            input = raw_keys + (t * kv_heads + khead) * head_dim;
            weight = key_weight;
            output = keys + (khead * length + t) * head_dim;
        }

        uint base = lane * 4;
        thread bfloat normalized[4];
        float sum = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            float value = float(input[base + i]);
            sum += value * value;
        }
        sum = simd_sum(sum);
        float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            normalized[i] =
                weight[base + i] *
                bfloat(float(input[base + i]) * inverse_rms);
        }

        // Every fixed-four loop in this prefill-only kernel is explicitly
        // scalarized. This removes loop-control ALU while preserving the
        // exact source order of the dependent RMS sum and rotary arithmetic.
        thread float paired[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            paired[i] = simd_shuffle(float(normalized[i]), lane ^ 16);
        }

        const device float* angle_row =
            angles + (uint(offsets[0]) + t) * (2 * rotary_pairs);
        if (lane < 16) {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                uint pair = base + i;
                float first = float(normalized[i]);
                float second = paired[i];
                float cosine = angle_row[pair];
                float sine = angle_row[pair + rotary_pairs];
                output[pair] = bfloat(first * cosine - second * sine);
                output[pair + rotary_pairs] =
                    bfloat(first * sine + second * cosine);
            }
        }
        """,
    ensureRowContiguous: true
)

/// Multi-token full-attention twin: per-head Q/K RMSNorm + partial YaRN
/// RoPE (rotary half 64, mscale on the rotary inputs, tail passes through).
/// One dispatch replaces the stock six (`rms_single_row` ×2, the general
/// copy each partial RoPE materializes first ×2, `rope_freqs_bfloat16` ×2).
///
/// Exactness mirrors the shipped decode kernel
/// (`laguna_full_qk_norm_yarn_bf16_128_v4`) against the same stock chain:
/// the same rms_single_row reproduction; the same mscale round-trip
/// `float(bfloat(x * bfloat(mscale)))` the stock
/// `rope_input_with_mscale<bfloat16, true>` applies under the negative-scale
/// sentinel; the same probe-seed angle row (the FP32 probe recovers
/// `fl(fl(1/mscale) * mscale) == 1.0f`, so the atlas carries pure cos/sin);
/// and the tail elements 64…127 written verbatim, matching the values the
/// stock pre-RoPE copy leaves behind.
private let lagunaPrefillFullQKNormYaRNKernel = MLXFast.metalKernel(
    name: "laguna_prefill_full_qk_norm_yarn_bf16_128_v2",
    inputNames: [
        "raw_queries", "raw_keys", "query_weight", "key_weight", "angles",
        "offsets",
    ],
    outputNames: ["queries", "keys"],
    source: """
        constexpr uint head_dim = 128;
        constexpr uint rotary_pairs = 32;
        constexpr uint query_heads = 48;
        constexpr uint kv_heads = 8;
        constexpr float yarn_mscale = 1.3465735912322998f;

        uint t = threadgroup_position_in_grid.y;
        uint length = threadgroups_per_grid.y;
        uint head = threadgroup_position_in_grid.x * 4
            + simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device bfloat* input;
        const device bfloat* weight;
        device bfloat* output;
        if (head < query_heads) {
            input = raw_queries + (t * query_heads + head) * head_dim;
            weight = query_weight;
            output = queries + (head * length + t) * head_dim;
        } else {
            uint khead = head - query_heads;
            input = raw_keys + (t * kv_heads + khead) * head_dim;
            weight = key_weight;
            output = keys + (khead * length + t) * head_dim;
        }

        uint base = lane * 4;
        thread bfloat normalized[4];
        float sum = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            float value = float(input[base + i]);
            sum += value * value;
        }
        sum = simd_sum(sum);
        float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            normalized[i] =
                weight[base + i] *
                bfloat(float(input[base + i]) * inverse_rms);
        }

        // Element `p + 32`, the rotary partner of pair `p` inside the
        // 64-wide YaRN half, lives 8 lanes away. As in the sliding twin,
        // scalarize the fixed-four plumbing without changing arithmetic.
        thread float paired[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            paired[i] = simd_shuffle(float(normalized[i]), lane ^ 8);
        }

        const device float* angle_row =
            angles + (uint(offsets[0]) + t) * (2 * rotary_pairs);
        if (lane < 8) {
            bfloat rounded_mscale = bfloat(yarn_mscale);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                uint pair = base + i;
                float first =
                    float(bfloat(normalized[i] * rounded_mscale));
                float second =
                    float(bfloat(bfloat(paired[i]) * rounded_mscale));
                float cosine = angle_row[pair];
                float sine = angle_row[pair + rotary_pairs];
                output[pair] = bfloat(first * cosine - second * sine);
                output[pair + rotary_pairs] =
                    bfloat(first * sine + second * cosine);
            }
        } else if (lane >= 16) {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                output[base + i] = normalized[i];
            }
        }
        """,
    ensureRowContiguous: true
)

/// One-head-per-threadgroup twin of the prefill full-attention QK-norm+YaRN
/// kernel (H1). Bit-exact repartition (EG256 class): each head is one SIMD and
/// all per-head arithmetic (RMSNorm `simd_sum`, the within-SIMD `lane ^ 8` YaRN
/// partner shuffle, the mscale round-trip) is SIMD-local, so grouping one head
/// per threadgroup instead of four changes only launch count/occupancy, not any
/// head's output value. Matches the proven decode shape.
private let lagunaPrefillFullQKNormYaRNH1Kernel = MLXFast.metalKernel(
    name: "laguna_prefill_full_qk_norm_yarn_bf16_128_h1_v2",
    inputNames: [
        "raw_queries", "raw_keys", "query_weight", "key_weight", "angles",
        "offsets",
    ],
    outputNames: ["queries", "keys"],
    source: """
        constexpr uint head_dim = 128;
        constexpr uint rotary_pairs = 32;
        constexpr uint query_heads = 48;
        constexpr uint kv_heads = 8;
        constexpr float yarn_mscale = 1.3465735912322998f;

        uint t = threadgroup_position_in_grid.y;
        uint length = threadgroups_per_grid.y;
        uint head = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;

        const device bfloat* input;
        const device bfloat* weight;
        device bfloat* output;
        if (head < query_heads) {
            input = raw_queries + (t * query_heads + head) * head_dim;
            weight = query_weight;
            output = queries + (head * length + t) * head_dim;
        } else {
            uint khead = head - query_heads;
            input = raw_keys + (t * kv_heads + khead) * head_dim;
            weight = key_weight;
            output = keys + (khead * length + t) * head_dim;
        }

        uint base = lane * 4;
        thread bfloat normalized[4];
        float sum = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            float value = float(input[base + i]);
            sum += value * value;
        }
        sum = simd_sum(sum);
        float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            normalized[i] =
                weight[base + i] *
                bfloat(float(input[base + i]) * inverse_rms);
        }

        // As in the sliding twin, scalarize the fixed-four plumbing without
        // changing arithmetic.
        thread float paired[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            paired[i] = simd_shuffle(float(normalized[i]), lane ^ 8);
        }

        const device float* angle_row =
            angles + (uint(offsets[0]) + t) * (2 * rotary_pairs);
        if (lane < 8) {
            bfloat rounded_mscale = bfloat(yarn_mscale);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                uint pair = base + i;
                float first =
                    float(bfloat(normalized[i] * rounded_mscale));
                float second =
                    float(bfloat(bfloat(paired[i]) * rounded_mscale));
                float cosine = angle_row[pair];
                float sine = angle_row[pair + rotary_pairs];
                output[pair] = bfloat(first * cosine - second * sine);
                output[pair + rotary_pairs] =
                    bfloat(first * sine + second * cosine);
            }
        } else if (lane >= 16) {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                output[base + i] = normalized[i];
            }
        }
        """,
    ensureRowContiguous: true
)

private func lagunaPrefillSlidingQKNormRoPE(
    rawQueries: MLXArray,
    rawKeys: MLXArray,
    queryWeight: MLXArray,
    keyWeight: MLXArray,
    angles: MLXArray,
    offsets: MLXArray,
    length: Int
) -> (MLXArray, MLXArray) {
    let heads = LagunaConstants.slidingAttentionHeads
    let kvHeads = LagunaConstants.numKeyValueHeads
    precondition(rawQueries.dtype == .bfloat16)
    precondition(rawKeys.dtype == .bfloat16)
    precondition(queryWeight.dtype == .bfloat16)
    precondition(keyWeight.dtype == .bfloat16)
    precondition(rawQueries.dims(1, length, heads * LagunaConstants.headDim))
    precondition(rawKeys.dims(1, length, kvHeads * LagunaConstants.headDim))
    precondition(queryWeight.dims(LagunaConstants.headDim))
    precondition(keyWeight.dims(LagunaConstants.headDim))
    precondition(angles.dtype == .float32)
    precondition(
        angles.dims(1, 1, lagunaRoPEAngleAtlasLength, LagunaConstants.headDim))
    precondition(offsets.dtype == .int32 && offsets.size == 1)
    precondition((heads + kvHeads) % 4 == 0)

    lagunaTrace("prefill sliding qk norm+rope")
    let useH1 = lagunaPrefillQKHeadsPerGroup == 1
    let headsPerGroup = useH1 ? 1 : 4
    let threadGroupSize = headsPerGroup * 32
    let kernel = useH1
        ? lagunaPrefillSlidingQKNormRoPEH1Kernel
        : lagunaPrefillSlidingQKNormRoPEKernel
    let outputs = kernel(
        [rawQueries, rawKeys, queryWeight, keyWeight, angles, offsets],
        grid: ((heads + kvHeads) / headsPerGroup * threadGroupSize, length, 1),
        threadGroup: (threadGroupSize, 1, 1),
        outputShapes: [
            [1, heads, length, LagunaConstants.headDim],
            [1, kvHeads, length, LagunaConstants.headDim],
        ],
        outputDTypes: [.bfloat16, .bfloat16]
    )
    return (outputs[0], outputs[1])
}

private func lagunaPrefillFullQKNormYaRN(
    rawQueries: MLXArray,
    rawKeys: MLXArray,
    queryWeight: MLXArray,
    keyWeight: MLXArray,
    angles: MLXArray,
    offsets: MLXArray,
    length: Int
) -> (MLXArray, MLXArray) {
    let heads = LagunaConstants.fullAttentionHeads
    let kvHeads = LagunaConstants.numKeyValueHeads
    precondition(rawQueries.dtype == .bfloat16)
    precondition(rawKeys.dtype == .bfloat16)
    precondition(queryWeight.dtype == .bfloat16)
    precondition(keyWeight.dtype == .bfloat16)
    precondition(rawQueries.dims(1, length, heads * LagunaConstants.headDim))
    precondition(rawKeys.dims(1, length, kvHeads * LagunaConstants.headDim))
    precondition(queryWeight.dims(LagunaConstants.headDim))
    precondition(keyWeight.dims(LagunaConstants.headDim))
    precondition(angles.dtype == .float32)
    precondition(
        angles.dims(1, 1, lagunaRoPEAngleAtlasLength, LagunaConstants.headDim / 2))
    precondition(offsets.dtype == .int32 && offsets.size == 1)
    precondition((heads + kvHeads) % 4 == 0)

    lagunaTrace("prefill full qk norm+yarn")
    let useH1 = lagunaPrefillQKHeadsPerGroup == 1
    let headsPerGroup = useH1 ? 1 : 4
    let threadGroupSize = headsPerGroup * 32
    let kernel = useH1
        ? lagunaPrefillFullQKNormYaRNH1Kernel
        : lagunaPrefillFullQKNormYaRNKernel
    let outputs = kernel(
        [rawQueries, rawKeys, queryWeight, keyWeight, angles, offsets],
        grid: ((heads + kvHeads) / headsPerGroup * threadGroupSize, length, 1),
        threadGroup: (threadGroupSize, 1, 1),
        outputShapes: [
            [1, heads, length, LagunaConstants.headDim],
            [1, kvHeads, length, LagunaConstants.headDim],
        ],
        outputDTypes: [.bfloat16, .bfloat16]
    )
    return (outputs[0], outputs[1])
}

struct LagunaIndexedAffineMetadata {
    let indices: MLXArray
    let lut: MLXArray

    var arrays: [MLXArray] { [indices, lut] }
}

private let lagunaAffineMetadataIndexedEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_AFFINE_METADATA_INDEXED"] != "0"

/// Exact UInt16 indices into an insertion-ordered BF16 `(scale,bias)` LUT.
private func lagunaIndexedAffineMetadata(
    scales: MLXArray, biases: MLXArray
) -> LagunaIndexedAffineMetadata? {
    guard lagunaAffineMetadataIndexedEnabled,
        scales.dtype == .bfloat16, biases.dtype == .bfloat16,
        scales.sameDims(biases), scales.size > 0
    else {
        return nil
    }

    let scaleBits = scales.view(dtype: .uint16).asArray(UInt16.self)
    let biasBits = biases.view(dtype: .uint16).asArray(UInt16.self)
    guard scaleBits.count == biasBits.count else { return nil }

    var pairToIndex: [UInt32: UInt16] = [:]
    pairToIndex.reserveCapacity(min(scaleBits.count, 65_536))
    var lut: [UInt32] = []
    lut.reserveCapacity(min(scaleBits.count, 65_536))
    var indices: [UInt16] = []
    indices.reserveCapacity(scaleBits.count)

    for i in scaleBits.indices {
        let pair = UInt32(scaleBits[i]) | (UInt32(biasBits[i]) << 16)
        if let index = pairToIndex[pair] {
            indices.append(index)
            continue
        }
        guard lut.count < 65_536 else { return nil }
        let index = UInt16(lut.count)
        pairToIndex[pair] = index
        lut.append(pair)
        indices.append(index)
    }

    return LagunaIndexedAffineMetadata(
        indices: MLXArray(indices).reshaped(scales.shape),
        lut: MLXArray(lut)
    )
}

struct LagunaNativeAffineWeight {
    let packedCodes: MLXArray
    var scales: MLXArray
    let biases: MLXArray?
    let originalShape: [Int]
    /// Shipped group-32 affine INT8 or the inherited group-16 NVFP4 tail.
    var groupSize: Int = 32
    var bits: Int = 8
    var mode: QuantizationMode = .affine
    var indexedMetadata: LagunaIndexedAffineMetadata? = nil
    /// Scale-plane-halving escape byte [1] uint8 for NVFP4 O-proj: original
    /// scale[0, 1] retained for the k=0 pair where pairwise constancy may fail.
    var scalesEscape: MLXArray? = nil
    var qkvEscape: MLXArray? = nil
    /// Interleaved [scale, bias] per group for g_proj gate-softplus cache locality.
    var interleavedMetadata: MLXArray? = nil

    var arrays: [MLXArray] {
        [packedCodes, scales]
            + (biases.map { [$0] } ?? [])
            + (indexedMetadata?.arrays ?? [])
            + (scalesEscape.map { [$0] } ?? [])
            + (qkvEscape.map { [$0] } ?? [])
            + (interleavedMetadata.map { [$0] } ?? [])
    }
}

/// Group-16 NVFP4 attention tail start layer. NVFP4 as shipped costs
/// 0.5625 B/param vs 1.125 for the group-32 affine INT8 side layout, so
/// each layer kept on NVFP4 halves its attention weight traffic (decode is
/// bandwidth-bound; local sweep monotone ~-0.5%/layer). Numerically this is
/// the shipped representation the goldens came from — envelope option (1),
/// which never requires the INT8 re-quant.
private let lagunaNativeAffineNVFP4From: Int? = {
    guard ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_NVFP4"] != "0"
    else { return nil }
    let raw = ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_NVFP4_FROM"] ?? "0"
    guard let value = Int(raw), value < LagunaConstants.numHiddenLayers else { return nil }
    return min(max(value, 0), LagunaConstants.numHiddenLayers)
}()

/// Diagnostic-only numeric-format probe; unset on the shipped path.
private let lagunaNativeAffineProbeFormat: (mode: QuantizationMode, groupSize: Int, bits: Int)? = {
    switch ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_PROBE_FORMAT"] {
    case "nvfp4": return (.nvfp4, 16, 4)
    case "mxfp8": return (.mxfp8, 32, 8)
    case "mxfp4": return (.mxfp4, 32, 4)
    default: return nil
    }
}()

private let lagunaNativeAffineProbeFormatFrom: Int = {
    guard
        let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_NATIVE_AFFINE_PROBE_FORMAT_FROM"],
        let value = Int(raw)
    else {
        return 0
    }
    return min(max(value, 0), LagunaConstants.numHiddenLayers)
}()

private func lagunaNativeAffineProbeRoundTrip(_ weight: MLXArray, layer: Int?) -> MLXArray {
    guard let probe = lagunaNativeAffineProbeFormat,
        weight.dim(1).isMultiple(of: probe.groupSize),
        layer.map({ $0 >= lagunaNativeAffineProbeFormatFrom }) ?? true
    else {
        return weight
    }
    let (codes, scales, biases) = quantized(
        weight, groupSize: probe.groupSize, bits: probe.bits, mode: probe.mode)
    return dequantized(
        codes, scales: scales, biases: biases,
        groupSize: probe.groupSize, bits: probe.bits, mode: probe.mode,
        dtype: .bfloat16)
}

func lagunaNativeAffineWeight(
    _ weight: MLXArray, layer: Int? = nil
) -> LagunaNativeAffineWeight? {
    guard weight.dtype == .bfloat16, weight.ndim == 2,
        weight.dim(1).isMultiple(of: 32)
    else {
        return nil
    }
    let source = lagunaNativeAffineProbeRoundTrip(weight, layer: layer)
    if let from = lagunaNativeAffineNVFP4From,
        (layer ?? 0) >= from,
        weight.dim(1).isMultiple(of: 16)
    {
        let (codes, scales, biases) = quantized(
            source, groupSize: 16, bits: 4, mode: .nvfp4)
        return LagunaNativeAffineWeight(
            packedCodes: codes,
            scales: scales,
            biases: biases,
            originalShape: weight.shape,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4
        )
    }
    let (packedCodes, scales, biases) = quantized(
        source, groupSize: 32, bits: 8, mode: .affine)
    guard biases != nil else { return nil }
    return LagunaNativeAffineWeight(
        packedCodes: packedCodes,
        scales: scales,
        biases: biases,
        originalShape: weight.shape
    )
}

/// Decode-only, exact fused input RMSNorm plus Q/K/V/gate projections.
private func lagunaFusedQKVProjectionSource(
    heads: Int, compact: Bool = false, mxfp8: Bool = false
) -> String {
    let projectionPointerSetup =
        compact
        ? """
        const device bfloat* weight;
        const device uint8_t* weight_low;
        const device uint8_t* weight_codes;
        const device uint8_t* weight_palettes;
        const device uint8_t* weight_modes;
        device bfloat* out;
        uint row_base;
        if (global_row < query_rows) {
            weight = query_weight;
            weight_low = query_low;
            weight_codes = query_codes;
            weight_palettes = query_palettes;
            weight_modes = query_modes;
            out = queries;
            row_base = global_row;
        } else if (global_row < query_rows + kv_rows) {
            weight = key_weight;
            weight_low = key_low;
            weight_codes = key_codes;
            weight_palettes = key_palettes;
            weight_modes = key_modes;
            out = keys;
            row_base = global_row - query_rows;
        } else {
            weight = value_weight;
            weight_low = value_low;
            weight_codes = value_codes;
            weight_palettes = value_palettes;
            weight_modes = value_modes;
            out = values;
            row_base = global_row - query_rows - kv_rows;
        }
        """
        : mxfp8
        ? """
        const device bfloat* weight;
        const device uint8_t* weight_codes8;
        const device uint8_t* weight_scales8;
        device bfloat* out;
        uint row_base;
        if (global_row < query_rows) {
            weight = query_weight;
            weight_codes8 = query_codes8;
            weight_scales8 = query_scales8;
            out = queries;
            row_base = global_row;
        } else if (global_row < query_rows + kv_rows) {
            weight = key_weight;
            weight_codes8 = key_codes8;
            weight_scales8 = key_scales8;
            out = keys;
            row_base = global_row - query_rows;
        } else {
            weight = value_weight;
            weight_codes8 = value_codes8;
            weight_scales8 = value_scales8;
            out = values;
            row_base = global_row - query_rows - kv_rows;
        }
        """
        : mxfp8
        ? """
        thread float result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
        constexpr uint scale_groups = in_vec_size / 32;

        // Contiguous TensorFold mapping: every lane owns two complete
        // 32-value MXFP8 groups. Codes and activations are loaded as eight
        // packed words per group, each scale is read once, and the final
        // simd reduction combines the 32 contiguous lane partials.
        for (uint row = 0; row < rows_per_thread; ++row) {
            float row_acc = 0.0f;
            for (uint gg = 0; gg < 2; ++gg) {
                uint group = 2 * lane + gg;
                float scale = laguna_attn_e8m0_decode(
                    weight_scales8[
                        size_t(row_base + row) * scale_groups + group]);
                const device uint4* cptr = (const device uint4*)(
                    weight_codes8
                    + size_t(row_base + row) * in_vec_size
                    + group * 32);
                uint4 packed0 = cptr[0];
                uint4 packed1 = cptr[1];
                const threadgroup ushort4* xrow =
                    (const threadgroup ushort4*)(
                        normalized_row + group * 32);
                #pragma clang loop unroll(full)
                for (uint w = 0; w < 8; ++w) {
                    uint packed =
                        (w < 4u) ? packed0[w & 3u] : packed1[w & 3u];
                    float4 weights =
                        laguna_attn_e4m3_decode4(packed) * scale;
                    float4 values =
                        as_type<float4>(uint4(xrow[w]) << 16);
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < 4; ++i) {
                        row_acc += weights[i] * values[i];
                    }
                }
            }
            result[row] = row_acc;
        }
        """
        : """
        const device bfloat* weight;
        device bfloat* out;
        uint row_base;
        if (global_row < query_rows) {
            weight = query_weight;
            out = queries;
            row_base = global_row;
        } else if (global_row < query_rows + kv_rows) {
            weight = key_weight;
            out = keys;
            row_base = global_row - query_rows;
        } else {
            weight = value_weight;
            out = values;
            row_base = global_row - query_rows - kv_rows;
        }
        """

    let projectionLoop =
        compact
        ? """
        thread float result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
        thread float coefficients[values_per_thread];
        constexpr uint palette_block_width = 1024;
        constexpr uint palette_blocks = in_vec_size / palette_block_width;
        constexpr uint subblocks_per_palette = palette_block_width / block_width;

        // One simdgroup owns four output rows. Lanes 0..<16 hold those rows'
        // sixteen literal high bytes, and `simd_shuffle` turns each packed
        // nibble into the exact high byte without a scattered device lookup.
        for (uint palette_segment = 0;
            palette_segment < palette_blocks; ++palette_segment)
        {
            thread uint palette_lane[rows_per_thread];
            thread uint raw_mode[rows_per_thread];
            for (uint row = 0; row < rows_per_thread; ++row) {
                size_t palette_block =
                    size_t(row_base + row) * palette_blocks + palette_segment;
                palette_lane[row] =
                    lane < 16
                    ? uint(weight_palettes[palette_block * 16 + lane])
                    : 0u;
                uint mode = lane == 0 ? uint(weight_modes[palette_block]) : 0u;
                raw_mode[row] = simd_shuffle(mode, ushort(0));
            }

            for (uint subblock = 0;
                subblock < subblocks_per_palette; ++subblock)
            {
                uint column =
                    palette_segment * palette_block_width
                    + subblock * block_width
                    + lane * values_per_thread;
                for (uint i = 0; i < values_per_thread; ++i) {
                    coefficients[i] = float(normalized_row[column + i]);
                }

                for (uint row = 0; row < rows_per_thread; ++row) {
                    size_t value_index =
                        size_t(row_base + row) * in_vec_size + column;
                    if (raw_mode[row] != 0) {
                        const device vec<bfloat, 4>* row_values =
                            (const device vec<bfloat, 4>*)(
                                weight + value_index);
                        const vec<bfloat, 4> w = row_values[0];
                        for (uint i = 0; i < values_per_thread; ++i) {
                            result[row] += float(w[i]) * coefficients[i];
                        }
                    } else {
                        uint8_t packed0 = weight_codes[value_index / 2];
                        uint8_t packed1 = weight_codes[value_index / 2 + 1];
                        thread float unpacked[values_per_thread];
                        uint high0 = simd_shuffle(
                            palette_lane[row], ushort(packed0 & 0x0fu));
                        uint high1 = simd_shuffle(
                            palette_lane[row], ushort(packed0 >> 4));
                        uint high2 = simd_shuffle(
                            palette_lane[row], ushort(packed1 & 0x0fu));
                        uint high3 = simd_shuffle(
                            palette_lane[row], ushort(packed1 >> 4));
                        unpacked[0] = as_type<float>(
                            (high0 << 24) | (uint(weight_low[value_index]) << 16));
                        unpacked[1] = as_type<float>(
                            (high1 << 24)
                            | (uint(weight_low[value_index + 1]) << 16));
                        unpacked[2] = as_type<float>(
                            (high2 << 24)
                            | (uint(weight_low[value_index + 2]) << 16));
                        unpacked[3] = as_type<float>(
                            (high3 << 24)
                            | (uint(weight_low[value_index + 3]) << 16));
                        for (uint i = 0; i < values_per_thread; ++i) {
                            result[row] += unpacked[i] * coefficients[i];
                        }
                    }
                }
            }
        }
        """
        : """
        thread float result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
        thread float coefficients[values_per_thread];

        uint column = lane * values_per_thread;
        for (uint block = 0; block < blocks; ++block) {
            for (uint i = 0; i < values_per_thread; ++i) {
                coefficients[i] = float(normalized_row[column + i]);
            }

            for (uint row = 0; row < rows_per_thread; ++row) {
                const device vec<bfloat, 4>* row_values =
                    (const device vec<bfloat, 4>*)(
                        weight + (row_base + row) * in_vec_size + column);
                const vec<bfloat, 4> w = row_values[0];
                for (uint i = 0; i < values_per_thread; ++i) {
                    result[row] += float(w[i]) * coefficients[i];
                }
            }

            column += block_width;
        }
        """

    return """
        constexpr uint in_vec_size = \(LagunaConstants.hiddenSize);
        constexpr uint query_rows = \(heads * LagunaConstants.headDim);
        constexpr uint kv_rows =
            \(LagunaConstants.numKeyValueHeads * LagunaConstants.headDim);
        constexpr uint rows_per_thread = 4;
        constexpr uint values_per_thread = 4;
        constexpr uint block_width = 128;
        constexpr uint blocks = in_vec_size / block_width;
        constexpr uint rows_per_group = 64;
        constexpr uint query_tiles = query_rows / rows_per_group;
        constexpr uint kv_tiles = kv_rows / rows_per_group;
        constexpr uint gate_tiles = \(heads / 8);
        constexpr uint query_tiles_per_round = query_tiles / kv_tiles;
        constexpr float norm_eps = 1.0e-6f;

        // The projection used to launch every Q tile, then every K tile,
        // then every V tile. Decode on the ranked M5 is latency-bound rather
        // than bandwidth-bound, so that order presents only one independent
        // weight bank to each scheduling wave. Preserve sequential row order
        // within every bank, but issue one K, one V, and (while available)
        // one gate tile before each proportional run of Q tiles.
        //
        // This is a pure bijection over the existing threadgroups. `tile`
        // below is the old logical tile number, so row ownership, K-loop
        // order, reductions, writes, and total work are unchanged.
        uint scheduled_tile = threadgroup_position_in_grid.x;
        uint round;
        uint position;
        constexpr uint gated_round_width = query_tiles_per_round + 3;
        constexpr uint plain_round_width = query_tiles_per_round + 2;
        constexpr uint gated_span = gate_tiles * gated_round_width;
        bool round_has_gate = scheduled_tile < gated_span;
        if (round_has_gate) {
            round = scheduled_tile / gated_round_width;
            position = scheduled_tile % gated_round_width;
        } else {
            uint tail = scheduled_tile - gated_span;
            round = gate_tiles + tail / plain_round_width;
            position = tail % plain_round_width;
        }

        uint tile;
        if (position == 0) {
            tile = query_tiles + round;
        } else if (position == 1) {
            tile = query_tiles + kv_tiles + round;
        } else if (round_has_gate && position == 2) {
            tile = query_tiles + 2 * kv_tiles + round;
        } else {
            uint projection_prefix = round_has_gate ? 3u : 2u;
            uint query_position = position - projection_prefix;
            tile = round * query_tiles_per_round + query_position;
        }

        uint local_id = thread_position_in_threadgroup.x;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        // --- input RMSNorm, mirroring rms_single_row at 512 threads ---
        \(lagunaNormInvMeanScratch)
        threadgroup float local_sums[32];
        threadgroup bfloat normalized_row[in_vec_size];

        uint norm_base = local_id * values_per_thread;
        thread float raw[values_per_thread];
        float acc = 0.0f;
        for (uint i = 0; i < values_per_thread; ++i) {
            raw[i] = float(residual[norm_base + i]);
            acc += raw[i] * raw[i];
        }
        acc = simd_sum(acc);
        \(lagunaNormReductionTailQKV)

        for (uint i = 0; i < values_per_thread; ++i) {
            bfloat value =
                norm_weight[norm_base + i] *
                bfloat(raw[i] * laguna_inv_mean);
            normalized_row[norm_base + i] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- per-head gate projection, on the tiles past the Q/K/V rows ---
        //
        // `g_proj` is the one Laguna projection MLX does not run with the
        // plain ladder: out_vec 64 with in_vec 2048 satisfies
        // `K >= 16 * out_vec`, so gemv_axbpy switches to BM 1 / BN 8, i.e.
        // eight simdgroups split K eight ways and then reduce through
        // threadgroup memory in ascending simdgroup order. Reproduced here
        // verbatim: two of those eight-simdgroup groups per 16-simdgroup
        // threadgroup, four rows each.
        constexpr uint gate_rows = 64;
        constexpr uint gate_simds = 8;
        constexpr uint gate_block_width = 1024;
        constexpr uint gate_blocks = in_vec_size / gate_block_width;
        constexpr uint qkv_tiles = query_tiles + 2 * kv_tiles;

        // Flat, because Metal will not take a multidimensional threadgroup
        // array here: [gate_half][split][row] laid out row-major by hand.
        threadgroup float gate_partials[2 * gate_simds * rows_per_thread];

        if (tile >= qkv_tiles) {
            uint gate_half = simd_group / gate_simds;
            uint split = simd_group % gate_simds;
            uint gate_row =
                ((tile - qkv_tiles) * 2 + gate_half) * rows_per_thread;

            thread float gate_result[rows_per_thread] = {
                0.0f, 0.0f, 0.0f, 0.0f
            };
            thread float gate_input[values_per_thread];

            uint gate_column =
                (split * 32 + lane) * values_per_thread;
            for (uint block = 0; block < gate_blocks; ++block) {
                for (uint i = 0; i < values_per_thread; ++i) {
                    gate_input[i] = float(normalized_row[gate_column + i]);
                }
                for (uint r = 0; r < rows_per_thread; ++r) {
                    const device vec<bfloat, 4>* row_values =
                        (const device vec<bfloat, 4>*)(
                            gate_weight + (gate_row + r) * in_vec_size +
                                gate_column);
                    const vec<bfloat, 4> gw = row_values[0];
                    for (uint i = 0; i < values_per_thread; ++i) {
                        gate_result[r] += float(gw[i]) * gate_input[i];
                    }
                }
                gate_column += gate_block_width;
            }

            for (uint r = 0; r < rows_per_thread; ++r) {
                for (ushort delta = 16; delta >= 1; delta >>= 1) {
                    gate_result[r] +=
                        metal::simd_shuffle_down(gate_result[r], delta);
                }
            }
            if (lane == 0) {
                for (uint r = 0; r < rows_per_thread; ++r) {
                    gate_partials[
                        (gate_half * gate_simds + split) * rows_per_thread + r
                    ] = gate_result[r];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (split == 0 && lane == 0) {
                for (uint r = 0; r < rows_per_thread; ++r) {
                    float total = gate_result[r];
                    for (uint sgn = 1; sgn < gate_simds; ++sgn) {
                        total += gate_partials[
                            (gate_half * gate_simds + sgn) * rows_per_thread + r
                        ];
                    }
                    // Preserve the stock boundary: the projection first
                    // rounds to BF16, then softplus widens that rounded logit
                    // to FP32 and rounds the activated value back to BF16.
                    bfloat rounded_logit = bfloat(total);
                    float logit = float(rounded_logit);
                    float gate;
                    if (metal::isnan(logit)) {
                        gate = NAN;
                    } else {
                        float maxval = metal::max(logit, 0.0f);
                        float minval = metal::min(logit, 0.0f);
                        gate = (metal::isinf(minval) || metal::isinf(maxval))
                            ? maxval
                            : maxval + log1p(metal::exp(minval - maxval));
                    }
                    gate_values[gate_row + r] = bfloat(gate);
                }
            }
            return;
        }

        // --- projections ---
        uint global_row = tile * rows_per_group + simd_group * rows_per_thread;

        \(projectionPointerSetup)

        \(projectionLoop)

        for (uint row = 0; row < rows_per_thread; ++row) {
            for (ushort delta = 16; delta >= 1; delta >>= 1) {
                result[row] += metal::simd_shuffle_down(result[row], delta);
            }
        }
        if (lane == 0) {
            for (uint row = 0; row < rows_per_thread; ++row) {
                out[row_base + row] = bfloat(result[row]);
            }
        }
        """
}

private let lagunaFusedQKVProjectionKernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        kernels[heads] = MLXFast.metalKernel(
            name: "laguna_fused_norm_qkv_projection_bf16_h\(heads)_v3",
            inputNames: [
                "residual", "norm_weight", "query_weight", "key_weight",
                "value_weight", "gate_weight",
            ],
            outputNames: ["queries", "keys", "values", "gate_values"],
            source: lagunaFusedQKVProjectionSource(heads: heads),
            ensureRowContiguous: true
        )
    }
    return kernels
}()

func lagunaFusedNormQKVProjection(
    residual: MLXArray,
    normWeight: MLXArray,
    queryWeight: MLXArray,
    keyWeight: MLXArray,
    valueWeight: MLXArray,
    gateWeight: MLXArray,
    heads: Int
) -> (
    queries: MLXArray, keys: MLXArray, values: MLXArray, gateValues: MLXArray,
    gateActivated: Bool
)? {
    guard let kernel = lagunaFusedQKVProjectionKernels[heads] else { return nil }
    let hidden = LagunaConstants.hiddenSize
    let queryRows = heads * LagunaConstants.headDim
    let kvRows = LagunaConstants.numKeyValueHeads * LagunaConstants.headDim
    precondition(residual.dtype == .bfloat16)
    precondition(residual.dims(1, 1, hidden))
    precondition(normWeight.dtype == .bfloat16)
    precondition(normWeight.dims(hidden))
    precondition(queryWeight.dims(queryRows, hidden))
    precondition(keyWeight.dims(kvRows, hidden))
    precondition(valueWeight.dims(kvRows, hidden))
    precondition(gateWeight.dtype == .bfloat16)
    precondition(gateWeight.dims(heads, hidden))

    // Q/K/V tiles at 64 rows each, then 8 more tiles carrying the 64 gate
    // rows as two eight-simdgroup split-K groups apiece.
    let projectionTiles = (queryRows + 2 * kvRows) / 64
    let gateTiles = heads / 8
    lagunaTrace("norm+qkv+gate projection h\(heads)")
    let outputs = kernel(
        [residual, normWeight, queryWeight, keyWeight, valueWeight, gateWeight],
        grid: ((projectionTiles + gateTiles) * 512, 1, 1),
        threadGroup: (512, 1, 1),
        outputShapes: [
            [1, 1, queryRows], [1, 1, kvRows], [1, 1, kvRows], [1, 1, heads],
        ],
        outputDTypes: [.bfloat16, .bfloat16, .bfloat16, .bfloat16]
    )
    return (outputs[0], outputs[1], outputs[2], outputs[3], true)
}

/// Decode-only fusion of the per-head attention gate with the output
/// projection. Folds the softplus gate into the GEMV's vector loads so the
/// 8192-wide gated row is never materialized — one dispatch instead of two.
/// Bit-exact: same BF16 gate product and GEMV accumulation order as stock.
private func lagunaGatedOutputProjectionSource(
    heads: Int, unroll: Int, compact: Bool = false
) -> String {
    let singleWeightLoad =
        compact
        ? """
                            size_t value_index =
                                size_t(out_row + row) * in_vec_size + column;
                            size_t palette_block = value_index / 1024;
                            vec<bfloat, 4> w;
                            if (weight_modes[palette_block] != 0) {
                                const device vec<bfloat, 4>* row_values =
                                    (const device vec<bfloat, 4>*)(
                                        weight + value_index);
                                w = row_values[0];
                            } else {
                                uint8_t packed0 =
                                    weight_codes[value_index / 2];
                                uint8_t packed1 =
                                    weight_codes[value_index / 2 + 1];
                                size_t palette_base = palette_block * 16;
                                ushort bits0 = ushort(weight_low[value_index])
                                    | (ushort(weight_palettes[
                                        palette_base + (packed0 & 0x0fu)]) << 8);
                                ushort bits1 = ushort(weight_low[value_index + 1])
                                    | (ushort(weight_palettes[
                                        palette_base + (packed0 >> 4)]) << 8);
                                ushort bits2 = ushort(weight_low[value_index + 2])
                                    | (ushort(weight_palettes[
                                        palette_base + (packed1 & 0x0fu)]) << 8);
                                ushort bits3 = ushort(weight_low[value_index + 3])
                                    | (ushort(weight_palettes[
                                        palette_base + (packed1 >> 4)]) << 8);
                                w[0] = as_type<bfloat>(bits0);
                                w[1] = as_type<bfloat>(bits1);
                                w[2] = as_type<bfloat>(bits2);
                                w[3] = as_type<bfloat>(bits3);
                            }
        """
        : """
                            const device vec<bfloat, 4>* row_values =
                                (const device vec<bfloat, 4>*)(
                                    weight + (out_row + row) * in_vec_size + column);
                            const vec<bfloat, 4> w = row_values[0];
        """

    let unrolledWeightLoad =
        compact
        ? """
                                size_t value_index =
                                    size_t(out_row + row) * in_vec_size + column_u;
                                size_t palette_block = value_index / 1024;
                                if (weight_modes[palette_block] != 0) {
                                    const device vec<bfloat, 4>* row_values =
                                        (const device vec<bfloat, 4>*)(
                                            weight + value_index);
                                    weight_values[u][row] = row_values[0];
                                } else {
                                    uint8_t packed0 =
                                        weight_codes[value_index / 2];
                                    uint8_t packed1 =
                                        weight_codes[value_index / 2 + 1];
                                    size_t palette_base = palette_block * 16;
                                    ushort bits0 = ushort(weight_low[value_index])
                                        | (ushort(weight_palettes[
                                            palette_base + (packed0 & 0x0fu)]) << 8);
                                    ushort bits1 = ushort(weight_low[value_index + 1])
                                        | (ushort(weight_palettes[
                                            palette_base + (packed0 >> 4)]) << 8);
                                    ushort bits2 = ushort(weight_low[value_index + 2])
                                        | (ushort(weight_palettes[
                                            palette_base + (packed1 & 0x0fu)]) << 8);
                                    ushort bits3 = ushort(weight_low[value_index + 3])
                                        | (ushort(weight_palettes[
                                            palette_base + (packed1 >> 4)]) << 8);
                                    weight_values[u][row][0] =
                                        as_type<bfloat>(bits0);
                                    weight_values[u][row][1] =
                                        as_type<bfloat>(bits1);
                                    weight_values[u][row][2] =
                                        as_type<bfloat>(bits2);
                                    weight_values[u][row][3] =
                                        as_type<bfloat>(bits3);
                                }
        """
        : """
                                const device vec<bfloat, 4>* row_values =
                                    (const device vec<bfloat, 4>*)(
                                        weight + (out_row + row) * in_vec_size +
                                            column_u);
                                weight_values[u][row] = row_values[0];
        """

    let body: String
    if unroll == 1 {
        body = """
                    uint column = lane * values_per_thread;
                    for (uint block = 0; block < blocks; ++block) {
                        // Column `4 * lane + 128 * block` sits in head `block`.
                        float gate = float(gate_values[block]);
                        const device vec<bfloat, 4>* gated =
                            (const device vec<bfloat, 4>*)(attention_output + column);
                        const vec<bfloat, 4> values = gated[0];
                        for (uint i = 0; i < values_per_thread; ++i) {
                            coefficients[i] = float(bfloat(float(values[i]) * gate));
                        }

                        for (uint row = 0; row < rows_per_thread; ++row) {
                            \(singleWeightLoad)
                            for (uint i = 0; i < values_per_thread; ++i) {
                                result[row] += float(w[i]) * coefficients[i];
                            }
                        }

                        column += block_width;
                    }
            """
    } else {
        body = """
                    uint column = lane * values_per_thread;
                    for (uint block = 0; block < blocks; block += unroll) {
                        vec<bfloat, 4> gated_values[unroll];
                        vec<bfloat, 4> weight_values[unroll][rows_per_thread];
                        for (uint u = 0; u < unroll; ++u) {
                            uint column_u = column + u * block_width;
                            const device vec<bfloat, 4>* gated =
                                (const device vec<bfloat, 4>*)(
                                    attention_output + column_u);
                            gated_values[u] = gated[0];
                            for (uint row = 0; row < rows_per_thread; ++row) {
                                \(unrolledWeightLoad)
                            }
                        }

                        for (uint u = 0; u < unroll; ++u) {
                            // Column `4 * lane + 128 * (block + u)` is in head
                            // `block + u`.
                            float gate = float(gate_values[block + u]);
                            for (uint i = 0; i < values_per_thread; ++i) {
                                coefficients[i] =
                                    float(bfloat(float(gated_values[u][i]) * gate));
                            }
                            for (uint row = 0; row < rows_per_thread; ++row) {
                                for (uint i = 0; i < values_per_thread; ++i) {
                                    result[row] +=
                                        float(weight_values[u][row][i]) *
                                            coefficients[i];
                                }
                            }
                        }

                        column += unroll * block_width;
                    }
            """
    }
    return """
        constexpr uint unroll = \(unroll);
        constexpr uint in_vec_size = \(heads * LagunaConstants.headDim);
        constexpr uint heads = \(heads);
        constexpr uint head_dim = 128;
        constexpr uint rows_per_thread = 4;
        constexpr uint values_per_thread = 4;
        constexpr uint block_width = 128;
        constexpr uint blocks = in_vec_size / block_width;
        constexpr uint rows_per_group = 16;

        uint tile = threadgroup_position_in_grid.x;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        uint out_row = tile * rows_per_group + simd_group * rows_per_thread;
        thread float result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
        thread float coefficients[values_per_thread];

        \(body)

        for (uint row = 0; row < rows_per_thread; ++row) {
            for (ushort delta = 16; delta >= 1; delta >>= 1) {
                result[row] += metal::simd_shuffle_down(result[row], delta);
            }
        }
        if (lane == 0) {
            for (uint row = 0; row < rows_per_thread; ++row) {
                projected[out_row + row] = bfloat(result[row]);
            }
        }
        """
}

/// `DARKBLOOM_L5_UNROLL` (default `2`; `1` restores the pre-unroll loop
/// verbatim, `4`/`8` deepen it): block-loop unroll depth for the gated output
/// projection. Every depth divides both block counts — 64 heads and 48 — so no
/// tail loop is ever needed, and depth `1` emits the pre-patch loop, which
/// makes it a true ablation control rather than an approximation of one.
///
/// The depth sweep {1, 2, 4} on this kernel is the highest-information
/// measurement left on this box. It decides whether outstanding loads per
/// thread — rather than bandwidth or occupancy — is what limits this whole
/// kernel family. A monotone rise toward 596 GB/s would mean the 462.9 µs /
/// 4.52% ceiling that L1+L5 have been sized against is itself too low.
let lagunaGatedOutputUnroll: Int = {
    guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_L5_UNROLL"],
        let value = Int(raw), [1, 2, 4, 8].contains(value)
    else {
        return 2
    }
    return value
}()

/// Every head count x every unroll depth, built eagerly so that one binary
/// serves every arm of an ablation (`notes/00`'s one-binary rule) and so MLX's
/// name-keyed JIT library cache never sees two sources under one name.
private let lagunaGatedOutputProjectionKernels:
    [Int: [Int: MLXFast.MLXFastKernel]] = {
    var kernels: [Int: [Int: MLXFast.MLXFastKernel]] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        var byDepth: [Int: MLXFast.MLXFastKernel] = [:]
        for depth in [1, 2, 4, 8] {
            byDepth[depth] = MLXFast.metalKernel(
                name: "laguna_gated_output_projection_bf16_h\(heads)_u\(depth)_v3",
                inputNames: ["attention_output", "gate_values", "weight"],
                outputNames: ["projected"],
                source: lagunaGatedOutputProjectionSource(
                    heads: heads, unroll: depth),
                ensureRowContiguous: true
            )
        }
        kernels[heads] = byDepth
    }
    return kernels
}()

func lagunaGatedOutputProjection(
    attentionOutput: MLXArray, gateValues: MLXArray, weight: MLXArray, heads: Int
) -> MLXArray? {
    guard let kernel = lagunaGatedOutputProjectionKernels[heads]?[lagunaGatedOutputUnroll]
    else { return nil }
    let inVec = heads * LagunaConstants.headDim
    precondition(attentionOutput.dtype == .bfloat16)
    precondition(attentionOutput.dims(1, 1, inVec))
    precondition(gateValues.dtype == .bfloat16)
    precondition(gateValues.dims(1, 1, heads))
    precondition(weight.dtype == .bfloat16)
    precondition(weight.dims(LagunaConstants.hiddenSize, inVec))

    lagunaTrace("gated output projection h\(heads)")
    return kernel(
        [attentionOutput, gateValues, weight],
        grid: ((LagunaConstants.hiddenSize / 16) * 128, 1, 1),
        threadGroup: (128, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.hiddenSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

/// Decode-only producer/consumer fusion for the per-head output gate. The
/// stock chain between the gate projection and the output projection is four
/// dispatches — BF16→FP32 cast, `LogAddExp(x, 0)`, FP32→BF16 cast, and the
/// broadcast product against the attention row — all over a 48/64-element
/// logit vector and a 6144/8192-element row. This kernel performs the same
/// four steps in one dispatch, reproducing each rounding boundary exactly:
///
/// - the FP32 softplus is MLX's `LogAddExp<float>` verbatim (same
///   `maxval + log1p(exp(minval - maxval))` form, same NaN/inf guards, and
///   the same `log1p`: MLX's own Goldberg implementation from its metal
///   utils preamble, which is also what the eager softplus dispatch uses);
/// - the gate is rounded to BF16 exactly where the stock `.asType(.bfloat16)`
///   rounds it, and the product rounds once to BF16 exactly where MLX's BF16
///   binary multiply rounds `float(bfloat(float(values[i]) * gate))`.
///
/// Every output element is therefore bit-identical to the four-dispatch
/// chain; the only change is dispatch count. One thread per output element;
/// the softplus is recomputed per element of a head, which is the same FP32
/// op stream the standalone softplus dispatch would run once per head.
private func lagunaGateProductSoftplusSource(heads: Int) -> String {
    """
    constexpr int HEAD_DIM = \(LagunaConstants.headDim);
    uint gid = thread_position_in_grid.x;
    int head = gid / HEAD_DIM;
    float logit = float(gate_logits[head]);
    float gate;
    if (metal::isnan(logit)) {
        gate = NAN;
    } else {
        float maxval = metal::max(logit, 0.0f);
        float minval = metal::min(logit, 0.0f);
        gate = (metal::isinf(minval) || metal::isinf(maxval))
            ? maxval
            : maxval + log1p(metal::exp(minval - maxval));
    }
    bfloat gate_bf = bfloat(gate);
    gated[gid] = bfloat(float(attention_output[gid]) * float(gate_bf));
    """
}

private let lagunaGateProductSoftplusKernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        kernels[heads] = MLXFast.metalKernel(
            name: "laguna_gate_product_softplus_bf16_h\(heads)_v1",
            inputNames: ["attention_output", "gate_logits"],
            outputNames: ["gated"],
            source: lagunaGateProductSoftplusSource(heads: heads),
            ensureRowContiguous: true
        )
    }
    return kernels
}()

/// Set `DARKBLOOM_FUSED_GATE_PRODUCT=0` to ablate and restore the exact
/// four-dispatch stock chain (eager softplus + donated in-place multiply).
private let lagunaFusedGateProductEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_GATE_PRODUCT"] != "0"

/// Returns the gated attention row for one decode token, or nil when the
/// preconditions do not hold (caller falls back to the stock chain). The
/// result is bit-identical to `output * softplus(gateLogits)` as the stock
/// chain computes it; see the kernel commentary above.
func lagunaGateProductSoftplus(
    attentionOutput: MLXArray, gateLogits: MLXArray, heads: Int
) -> MLXArray? {
    guard lagunaFusedGateProductEnabled,
        let kernel = lagunaGateProductSoftplusKernels[heads]
    else { return nil }
    let inVec = heads * LagunaConstants.headDim
    precondition(attentionOutput.dtype == .bfloat16)
    precondition(attentionOutput.dims(1, 1, inVec))
    precondition(gateLogits.dtype == .bfloat16)
    precondition(gateLogits.dims(1, 1, heads))

    lagunaTrace("gate product softplus h\(heads)")
    return kernel(
        [attentionOutput, gateLogits],
        grid: (inVec, 1, 1),
        threadGroup: (128, 1, 1),
        outputShapes: [[1, 1, inVec]],
        outputDTypes: [.bfloat16]
    )[0]
}

// MARK: - Gated native-affine INT8 output projection (one dispatch)

/// Exact per-head softplus gate plus group-32 affine INT8 output GEMV.
private func lagunaGatedAffineOProjSource(heads: Int, indexed: Bool = false) -> String {
    let metadataPointers = indexed
        ? """
        const device ushort* mi = metadata_indices + out_row * in_vec_size_g +
            simd_lid / scale_step_per_thread;
        """
        : """
        const device bfloat* sc = weight_scales + out_row * in_vec_size_g +
            simd_lid / scale_step_per_thread;
        const device bfloat* bs = weight_biases + out_row * in_vec_size_g +
            simd_lid / scale_step_per_thread;
        """
    let metadataLoad = indexed
        ? """
            float scale, bias;
            uint gl = (simd_lid / scale_step_per_thread) * scale_step_per_thread;
            if (simd_lid == gl) {
                uint pair = metadata_lut[mi[row * in_vec_size_g]];
                scale = float(as_type<bfloat>(ushort(pair)));
                bias = float(as_type<bfloat>(ushort(pair >> 16)));
            }
            scale = simd_shuffle(scale, gl);
            bias = simd_shuffle(bias, gl);
        """
        : """
            float scale, bias;
            uint gl = (simd_lid / scale_step_per_thread) * scale_step_per_thread;
            if (simd_lid == gl) {
                scale = float(sc[row * in_vec_size_g]);
                bias = float(bs[row * in_vec_size_g]);
            }
            scale = simd_shuffle(scale, gl);
            bias = simd_shuffle(bias, gl);
        """
    let metadataAdvance = indexed
        ? "mi += block_size / group_size;"
        : """
        sc += block_size / group_size;
        bs += block_size / group_size;
        """
    return """
    constexpr uint in_vec_size = \(heads * LagunaConstants.headDim);
    constexpr uint out_vec_size = \(LagunaConstants.hiddenSize);
    constexpr uint gate_heads = \(heads);
    constexpr uint head_shift = 7;              // head_dim == 128
    constexpr uint values_per_thread = 8;       // pack_factor 4 * packs_per_thread 2
    constexpr uint block_size = 256;            // values_per_thread * SIMD_SIZE
    constexpr uint results_per_simdgroup = 4;
    constexpr uint num_simdgroups = 2;
    constexpr uint group_size = 32;
    constexpr uint scale_step_per_thread = group_size / values_per_thread;
    constexpr uint in_vec_size_g = in_vec_size / group_size;

    uint tile = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    // One softplus per head, in the FP32 form and at the BF16 rounding point
    // `lagunaGateProductSoftplusSource` uses.
    threadgroup float gate_table[gate_heads];
    if (lid < gate_heads) {
        float logit = float(gate_logits[lid]);
        float gate;
        if (metal::isnan(logit)) {
            gate = NAN;
        } else {
            float maxval = metal::max(logit, 0.0f);
            float minval = metal::min(logit, 0.0f);
            gate = (metal::isinf(minval) || metal::isinf(maxval))
                ? maxval
                : maxval + log1p(metal::exp(minval - maxval));
        }
        gate_table[lid] = float(bfloat(gate));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint out_row = tile * (num_simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;

    const device uint8_t* ws = (const device uint8_t*)weight_codes +
        out_row * in_vec_size + simd_lid * values_per_thread;
    \(metadataPointers)
    const device bfloat* xp = attention_output + simd_lid * values_per_thread;

    thread float x_thread[values_per_thread];
    thread float result[results_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint column = simd_lid * values_per_thread;
    for (uint k = 0; k < in_vec_size; k += block_size) {
        float gate = gate_table[column >> head_shift];
        float sum = 0.0f;
        for (uint i = 0; i < values_per_thread; ++i) {
            float value = float(bfloat(float(xp[i]) * gate));
            sum += value;
            x_thread[i] = value;
        }

        for (uint row = 0; row < results_per_simdgroup; ++row) {
            const device uint8_t* wl = ws + row * in_vec_size;
            \(metadataLoad)
            float accum = dot(float4(x_thread[0], x_thread[1], x_thread[2], x_thread[3]),
                              float4(float(wl[0]), float(wl[1]), float(wl[2]), float(wl[3])));
            accum += dot(float4(x_thread[4], x_thread[5], x_thread[6], x_thread[7]),
                         float4(float(wl[4]), float(wl[5]), float(wl[6]), float(wl[7])));
            result[row] += scale * accum + sum * bias;
        }

        ws += block_size;
        \(metadataAdvance)
        xp += block_size;
        column += block_size;
    }

    {
        const vec<float, 4> packed = simd_sum(
            vec<float, 4>(result[0], result[1], result[2], result[3]));
        result[0] = packed.x; result[1] = packed.y;
        result[2] = packed.z; result[3] = packed.w;
    }
    if (simd_lid == 0) {
        for (uint row = 0; row < results_per_simdgroup; ++row) {
            projected[out_row + row] = bfloat(result[row]);
        }
    }
    """
}

/// One kernel per attention head count, built eagerly so one binary serves
/// every arm of an ablation and MLX's name-keyed JIT cache never sees two
/// sources under one name.
private let lagunaGatedAffineOProjKernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        kernels[heads] = MLXFast.metalKernel(
            name: "laguna_gated_affine_oproj_qmv_i8g32_h\(heads)_v2",
            inputNames: [
                "attention_output", "gate_logits", "weight_codes",
                "weight_scales", "weight_biases",
            ],
            outputNames: ["projected"],
            source: lagunaGatedAffineOProjSource(heads: heads),
            ensureRowContiguous: true
        )
    }
    return kernels
}()

private let lagunaGatedAffineOProjIndexedKernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        kernels[heads] = MLXFast.metalKernel(
            name: "laguna_gated_affine_oproj_qmv_i8g32_h\(heads)_idx_v2",
            inputNames: [
                "attention_output", "gate_logits", "weight_codes",
                "metadata_indices", "metadata_lut",
            ],
            outputNames: ["projected"],
            source: lagunaGatedAffineOProjSource(heads: heads, indexed: true),
            ensureRowContiguous: true
        )
    }
    return kernels
}()

/// `DARKBLOOM_FUSED_GATED_AFFINE_OPROJ` (default ON; set "0" to disable).
/// Fuses the per-head softplus gate product into the native group-32 affine
/// INT8 o_proj GEMV so the decode attention tail is one dispatch instead of
/// two. Bit-exact: the gate factor is applied at the same FP32 -> BF16
/// rounding point as `lagunaGateProductSoftplus`, and the contraction keeps
/// MLX's affine GEMV geometry and accumulation order exactly. Falls back to
/// the exact two-dispatch chain whenever the affine o proj is not group-32
/// INT8 (the NVFP4 tail layers) or any guard declines.
let lagunaFusedGatedAffineOProjEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_GATED_AFFINE_OPROJ"] != "0"

/// Gates only the NVFP4 tail-layer twin so it can be ablated independently.
let lagunaGatedAffineOProjNVFP4Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_GATED_AFFINE_OPROJ_NVFP4"] != "0"

/// `DARKBLOOM_NVFP4_QMV_SIGN_CARRY` (default OFF): in the standalone NVFP4
/// o_proj QMV decode (`lagunaGatedAffineOProjNVFP4Source`), fold the E4M3
/// group-scale sign bit into the half bit pattern instead of a conditional
/// negate — the same bit-exact transform ivanfioravanti's 71b80b1f applied to
/// the fused tail qdot (`DARKBLOOM_TAIL_NVFP4_SCALE_FOLD`), which this
/// standalone kernel did not carry. For `bits = 128 + m` the carry
/// `bits + (bits & 128)` yields `256 + m`, and `(256 + m) << 7 ==
/// 0x8000 | (m << 7)` because `m <= 127` keeps `m << 7 <= 16256 < 0x8000`;
/// IEEE half is sign-magnitude, so that pattern IS the negation, including
/// `-0.0h`. For `bits < 128` the add is the identity. Replaces the branch and
/// the intermediate `half` with one carrying add. Exhaustively bit-identical
/// over all 256 E4M3 bytes (`LagunaNVFP4QMVFoldTests`); the two power-of-two
/// scale factors were already folded to the per-row `* 4194304.0f` epilogue.
let lagunaNvfp4QmvSignCarryEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_NVFP4_QMV_SIGN_CARRY"] != "0"

/// `DARKBLOOM_E4M3_SIGN_DOMAIN` (default ON; set "0" to keep the sign-carry
/// forms): census-certified sign-domain specialization of the E4M3 scale
/// decode. A full scan of the pinned checkpoint's 234 U8 scale tensors
/// (1,970,601,984 bytes) measures min 1, max 73, zero sign bits, and the
/// init-derived attention side banks inherit nonnegativity from the vendored
/// `fp_quantize` producer (`simd_max(abs(w)) / 6` converted to E4M3), so
/// `bits & 128 == 0` on every scale byte any kernel can read. The carrying
/// add `bits + (bits & 128)` therefore reduces algebraically to `bits` and
/// the sign select to the identity — same bit pattern, same float, fewer
/// dependent ops. The kill switch restores the carry forms, which remain
/// exact over all 256 bytes. (Mechanism first priced by 0xkydo's 3d6f202,
/// which passed every hidden gate and the static review; census reproduced
/// independently on this machine before adoption.)
let lagunaE4M3SignDomainCertified =
    ProcessInfo.processInfo.environment["DARKBLOOM_E4M3_SIGN_DOMAIN"] != "0"

/// `DARKBLOOM_NVFP4_QMV_SEED_ELIDE` (default OFF): in the same o_proj QMV
/// decode, assign the first four-term product group to the accumulator instead
/// of adding it to a `+0.0f` seed, removing one dead FP add per output row per
/// K block. `fadd 0.0, %t` is not foldable under `setFastMathEnabled(false)`
/// (`0.0 + (-0.0) == +0.0 != -0.0`), so the add is really emitted. Eliding it
/// can only flip the sign of an all-`-0.0` group's zero, which the
/// `+0.0f`-seeded `result[row]` accumulator (`+0.0 + -0.0 == +0.0`) and the
/// BF16 epilogue absorb, so the kernel output is bit-identical
/// (`LagunaNVFP4QMVFoldTests`).
let lagunaNvfp4QmvSeedElisionEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_NVFP4_QMV_SEED_ELIDE"] != "0"

/// Gate product + native-affine INT8 output projection in one dispatch, or
/// `nil` when any shape, dtype or wire-format guard declines (caller then runs
/// the exact two-dispatch chain).
func lagunaGatedAffineOProj(
    attentionOutput: MLXArray,
    gateLogits: MLXArray,
    codes: MLXArray,
    scales: MLXArray,
    biases: MLXArray,
    indexedMetadata: LagunaIndexedAffineMetadata? = nil,
    heads: Int
) -> MLXArray? {
    let inVec = heads * LagunaConstants.headDim
    let outVec = LagunaConstants.hiddenSize
    guard attentionOutput.dtype == .bfloat16,
        attentionOutput.dims(1, 1, inVec),
        gateLogits.dtype == .bfloat16,
        gateLogits.dims(1, 1, heads),
        codes.dtype == .uint32,
        codes.dims(outVec, inVec / 4),
        scales.dtype == .bfloat16,
        scales.dims(outVec, inVec / 32),
        biases.dtype == .bfloat16,
        biases.dims(outVec, inVec / 32)
    else {
        return nil
    }

    if let metadata = indexedMetadata,
        let kernel = lagunaGatedAffineOProjIndexedKernels[heads],
        metadata.indices.dtype == .uint16,
        metadata.indices.dims(outVec, inVec / 32),
        metadata.lut.dtype == .uint32,
        metadata.lut.ndim == 1,
        metadata.lut.size <= 65_536
    {
        lagunaTrace("gated affine oproj qmv h\(heads) indexed")
        return kernel(
            [attentionOutput, gateLogits, codes, metadata.indices, metadata.lut],
            grid: ((outVec / 8) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[1, 1, outVec]], outputDTypes: [.bfloat16]
        )[0]
    }
    guard let kernel = lagunaGatedAffineOProjKernels[heads] else { return nil }
    lagunaTrace("gated affine oproj qmv h\(heads)")
    return kernel(
        [attentionOutput, gateLogits, codes, scales, biases],
        grid: ((outVec / 8) * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, outVec]],
        outputDTypes: [.bfloat16]
    )[0]
}

// MARK: - Gated NVFP4 output projection for the affine tail layers

/// NVFP4 twin of `lagunaGatedAffineOProjSource` for layers using the native
/// group-16 NVFP4 output projection. It folds the softplus gate, broadcast
/// product, and contraction into one dispatch while preserving the BF16 gate
/// rounding point and the stock NVFP4 accumulation geometry.
func lagunaGatedAffineOProjNVFP4Source(
    heads: Int,
    signCarry: Bool = lagunaNvfp4QmvSignCarryEnabled,
    seedElide: Bool = lagunaNvfp4QmvSeedElisionEnabled,
    preActivatedGate: Bool = false,
    halvedScales: Bool = false
) -> String {
    let scaleFold = lagunaNvfp4ScaleFoldEnabled
    let weightScale = scaleFold ? "" : " * 16384.0f"
    // Sign-carry fold: E4M3 is sign-magnitude, so carrying the sign bit into
    // the half pattern is the exact negation over all 256 bytes (incl. -0.0h);
    // the OFF arm keeps the negate-after-convert form verbatim.
    let scaleDecode = signCarry
        ? (lagunaE4M3SignDomainCertified
            ? "ushort sraw = ushort(sbits) << 7;\n"
                + "        float scale = float(as_type<half>(sraw));"
            : "ushort sraw = ushort(sbits + (sbits & 128)) << 7;\n"
                + "        float scale = float(as_type<half>(sraw));")
        : "ushort sraw = ushort(sbits & 127) << 7;\n"
            + "        half sconverted = as_type<half>(sraw);\n"
            + "        float scale = float((sbits & 128) ? -sconverted : sconverted);"
    // Seed elision: assign the first four-term group instead of adding it to a
    // dead `+0.0f` seed. Only a signed-zero can differ, and the `+0.0f`-seeded
    // `result[row]` plus BF16 epilogue absorb it. OFF arm keeps the seed.
    let accumDecl = seedElide ? "float accum;" : "float accum = 0.0f;"
    let firstAccum = seedElide
        ? "if (j == 0) {\n"
            + "                accum =\n"
            + "                    (x_thread[8 * j] * v04.x +\n"
            + "                     x_thread[8 * j + 1] * v15.x +\n"
            + "                     x_thread[8 * j + 2] * v26.x +\n"
            + "                     x_thread[8 * j + 3] * v37.x);\n"
            + "            } else {\n"
            + "                accum +=\n"
            + "                    (x_thread[8 * j] * v04.x +\n"
            + "                     x_thread[8 * j + 1] * v15.x +\n"
            + "                     x_thread[8 * j + 2] * v26.x +\n"
            + "                     x_thread[8 * j + 3] * v37.x);\n"
            + "            }"
        : "accum +=\n"
            + "                (x_thread[8 * j] * v04.x +\n"
            + "                 x_thread[8 * j + 1] * v15.x +\n"
            + "                 x_thread[8 * j + 2] * v26.x +\n"
            + "                 x_thread[8 * j + 3] * v37.x);"
    let extract = """
                    const uint xe = c & 0x0F0F0F0Fu;
                    const uint ge = xe | (xe << 3);
                    const uint yo = c & 0xF0F0F0F0u;
                    const uint go = yo | (yo >> 3);
                    const uint p0 = (ge << 9) & 0x8E008E00u;
                    const uint p1 = (go << 8) & 0x8E008E00u;
                    const uint p2 = (ge << 1) & 0x8E008E00u;
                    const uint p3 = go & 0x8E008E00u;
    """
    let gateSetup = preActivatedGate ? "" : """
    threadgroup float gt[gate_heads];
    if(lid<gate_heads){
        float l=float(gate_logits[lid]);
        float g;
        if(metal::isnan(l)) g=NAN;
        else {
            float hi=metal::max(l,0.0f);
            float lo=metal::min(l,0.0f);
            g=(metal::isinf(lo)||metal::isinf(hi))?hi:hi+log1p(metal::exp(lo-hi));
        }
        gt[lid]=float(bfloat(g));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    """
    let loadInput = preActivatedGate
        ? """
        float g=float(gate_values[column>>head_shift]);
        for(uint i=0;i<values_per_thread;++i)
            x_thread[i]=float(bfloat(float(xp[i])*g));
        """
        : """
        float g=gt[column>>head_shift];
        for(uint i=0;i<values_per_thread;++i)
            x_thread[i]=float(bfloat(float(xp[i])*g));
        """
    // Scale-halving: when enabled, scales are stored as [outVec, inVec/32]
    // (every other byte). The kernel reads simd_lid/2 with half stride.
    // The escape byte corrects the k=0 pair read for row 0, lane 1, where
    // the quantizer may break pairwise constancy (scale[0] != scale[1]).
    let scaleStride = halvedScales ? "in_vec_size_g / 2" : "in_vec_size_g"
    let scaleLane = halvedScales ? "simd_lid / 2" : "simd_lid"
    let scaleStep = halvedScales ? "block_size / (group_size * 2)" : "block_size / group_size"
    let escapeDecl = halvedScales
        ? "    uint8_t oproj_escape_val = oproj_escape_buf[0];\n    "
        : ""
    let escapeCheck = halvedScales
        ? "            if (out_row + row == 0 && simd_lid == 1 && k == 0)\n"
            + "                sbits = oproj_escape_val;\n"
        : ""
    return """
    constexpr uint in_vec_size = \(heads * LagunaConstants.headDim);
    constexpr uint out_vec_size = \(LagunaConstants.hiddenSize);
    constexpr uint gate_heads = \(heads);
    constexpr uint head_shift = 7;
    constexpr uint group_size = 16;
    constexpr uint values_per_thread = 16;
    constexpr uint codes_per_thread = values_per_thread / 8;
    constexpr uint block_size = values_per_thread * 32;
    constexpr uint results_per_simdgroup = 4;
    constexpr uint num_simdgroups = 2;
    constexpr uint in_vec_size_g = in_vec_size / group_size;

    uint tile = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    \(gateSetup)

    uint out_row = tile * (num_simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;
    const device uint32_t* ws =
        (const device uint32_t*)weight_codes +
        out_row * (in_vec_size / 8) + simd_lid * codes_per_thread;
    const device uint8_t* sc = weight_scales +
        out_row * (\(scaleStride)) + (\(scaleLane));
    const device bfloat* xp = attention_output + simd_lid * values_per_thread;
    \(escapeDecl)thread float x_thread[values_per_thread];
    thread float result[results_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint column = simd_lid * values_per_thread;
    for (uint k = 0; k < in_vec_size; k += block_size) {
        \(loadInput)

        for (uint row = 0; row < results_per_simdgroup; ++row) {
            const device uint32_t* wl = ws + row * (in_vec_size / 8);
            // Defer the exact E4M3 2^22 renormalization to the per-row
            // epilogue. Every partial remains the exact 2^-22 rescaling of
            // the control until the multiply before the existing BF16 round.
            uint8_t sbits = sc[row * (\(scaleStride))];
            \(escapeCheck)\(scaleDecode)
            \(accumDecl)
            #pragma unroll
            for (uint j = 0; j < codes_per_thread; ++j) {
                const uint c = wl[j];
                \(extract)
                const float2 v04 = float2(as_type<half2>(p0))\(weightScale);
                const float2 v15 = float2(as_type<half2>(p1))\(weightScale);
                const float2 v26 = float2(as_type<half2>(p2))\(weightScale);
                const float2 v37 = float2(as_type<half2>(p3))\(weightScale);
                \(firstAccum)
                accum +=
                    (x_thread[8 * j + 4] * v04.y +
                     x_thread[8 * j + 5] * v15.y +
                     x_thread[8 * j + 6] * v26.y +
                     x_thread[8 * j + 7] * v37.y);
            }
            result[row] += scale * accum;
        }

        ws += block_size / 8;
        sc += \(scaleStep);
        xp += block_size;
        column += block_size;
    }

    {
        const vec<float, 4> packed = simd_sum(
            vec<float, 4>(result[0], result[1], result[2], result[3]) * 4194304.0f);
        result[0] = packed.x; result[1] = packed.y;
        result[2] = packed.z; result[3] = packed.w;
    }
    if (simd_lid == 0) {
        for (uint row = 0; row < results_per_simdgroup; ++row) {
            projected[out_row + row] = bfloat(result[row]);
        }
    }
    """
}

/// O-proj NVFP4 scale-plane halving. NVFP4 quantization produces pairwise-
/// constant scale bytes (scale[2k] == scale[2k+1]) for most groups, so one
/// byte per 32 elements suffices instead of one per 16. The kernel reads
/// `simd_lid / 2` from the halved scale row, halving ~40 MiB/step of O-proj
/// scale traffic. Default ON; set "0" to restore full scales.
private let lagunaOProjScaleHalvingEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_OPROJ_SCALE_HALVING"] != "0"

private let lagunaGatedAffineOProjNVFP4Kernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        kernels[heads] = MLXFast.metalKernel(
            name: "laguna_gated_affine_oproj_nvfp4_qmv_h\(heads)_v1"
                + (lagunaNvfp4QmvSignCarryEnabled ? "_sc1" : "")
                + (lagunaNvfp4QmvSeedElisionEnabled ? "_se1" : ""),
            inputNames: [
                "attention_output", "gate_logits", "weight_codes",
                "weight_scales",
            ],
            outputNames: ["projected"],
            source: lagunaGatedAffineOProjNVFP4Source(heads: heads),
            ensureRowContiguous: true
        )
    }
    return kernels
}()

private let lagunaGatedAffineOProjNVFP4HalvedKernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        kernels[heads] = MLXFast.metalKernel(
            name: "laguna_gated_affine_oproj_nvfp4_qmv_h\(heads)_v1"
                + (lagunaNvfp4QmvSignCarryEnabled ? "_sc1" : "")
                + (lagunaNvfp4QmvSeedElisionEnabled ? "_se1" : "")
                + "_hs1",
            inputNames: [
                "attention_output", "gate_logits", "weight_codes",
                "weight_scales", "oproj_escape_buf",
            ],
            outputNames: ["projected"],
            source: lagunaGatedAffineOProjNVFP4Source(heads: heads, halvedScales: true),
            ensureRowContiguous: true
        )
    }
    return kernels
}()

private let lagunaGateSoftplusEnabled = ProcessInfo.processInfo.environment[
    "DARKBLOOM_AFFINE_GATE_SOFTPLUS"] != "0"

private func lagunaGateSoftplusSource(heads: Int) -> String {
    """
    constexpr uint K=\(LagunaConstants.hiddenSize),GS=32,V=8;
    constexpr uint BK=V*32,R=4,NS=2,KG=K/GS,SS=GS/V;
    uint tile=threadgroup_position_in_grid.x;
    uint sg=simdgroup_index_in_threadgroup;
    uint lane=thread_index_in_simdgroup;
    uint orow=tile*(NS*R)+sg*R;
    const device uint8_t* ws=(const device uint8_t*)packed_codes+orow*K+lane*V;
    const device bfloat* pm=packed_metadata+orow*KG*2+(lane/SS)*2;
    thread float x[V];
    thread float r[R]={0.0f,0.0f,0.0f,0.0f};
    uint col=lane*V;
    for(uint k=0;k<K;k+=BK){
        float sum=0.0f;
        for(uint i=0;i<V;++i){
            x[i]=float(input[col+i]);
            sum+=x[i];
        }
        for(uint row=0;row<R;++row){
            const device uint8_t* wl=ws+row*K;
            float s,b;
            uint gl=(lane/SS)*SS;
            if(lane==gl){ s=float(pm[row*KG*2]); b=float(pm[row*KG*2+1]); }
            s=simd_shuffle(s,gl); b=simd_shuffle(b,gl);
            float a=0.0f;
            for(uint i=0;i<V;++i) a+=x[i]*wl[i];
            r[row]+=s*a+sum*b;
        }
        ws+=BK; pm+=(BK/GS)*2; col+=BK;
    }
    {
        const vec<float, 4> packed = simd_sum(
            vec<float, 4>(r[0], r[1], r[2], r[3]));
        r[0] = packed.x; r[1] = packed.y;
        r[2] = packed.z; r[3] = packed.w;
    }
    for(uint row=0;row<R;++row){
        if(lane==0){
            float l=float(bfloat(r[row]));
            float g;
            if(metal::isnan(l)) g=NAN;
            else {
                float hi=metal::max(l,0.0f);
                float lo=metal::min(l,0.0f);
                g=(metal::isinf(lo)||metal::isinf(hi))?hi:hi+log1p(metal::exp(lo-hi));
            }
            gate_values[orow+row]=bfloat(g);
        }
    }
    """
}

private let lagunaGateSoftplusKernels: [Int: MLXFast.MLXFastKernel] = {
    var result: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        result[heads] = MLXFast.metalKernel(
            name: "laguna_gate_sp_h\(heads)_v3",
            inputNames: ["input", "packed_codes", "packed_metadata"],
            outputNames: ["gate_values"],
            source: lagunaGateSoftplusSource(heads: heads),
            ensureRowContiguous: true)
    }
    return result
}()

private func lagunaGateSoftplus(
    input: MLXArray, bank: LagunaNativeAffineWeight, heads: Int
) -> MLXArray? {
    guard lagunaGateSoftplusEnabled,
        bank.mode == .affine, bank.bits == 8, bank.groupSize == 32,
        let biases = bank.biases,
        let interleaved = bank.interleavedMetadata,
        let kernel = lagunaGateSoftplusKernels[heads],
        input.dtype == .bfloat16,
        input.dims(1, 1, LagunaConstants.hiddenSize),
        bank.packedCodes.dims(heads, LagunaConstants.hiddenSize / 4),
        interleaved.dims(heads, LagunaConstants.hiddenSize / 16),
        biases.dims(heads, LagunaConstants.hiddenSize / 32)
    else { return nil }

    return kernel(
        [input, bank.packedCodes, interleaved],
        grid: ((heads / 8) * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, heads]],
        outputDTypes: [.bfloat16])[0]
}

private let lagunaActivatedOProjKernels: [Int: MLXFast.MLXFastKernel] = {
    var result: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        result[heads] = MLXFast.metalKernel(
            name: "laguna_oproj_act_h\(heads)_v1"
                + (lagunaNvfp4QmvSignCarryEnabled ? "_sc1" : "")
                + (lagunaNvfp4QmvSeedElisionEnabled ? "_se1" : ""),
            inputNames: [
                "attention_output", "gate_values", "weight_codes",
                "weight_scales",
            ],
            outputNames: ["projected"],
            source: lagunaGatedAffineOProjNVFP4Source(heads: heads, preActivatedGate: true),
            ensureRowContiguous: true)
    }
    return result
}()

private let lagunaActivatedOProjHalvedKernels: [Int: MLXFast.MLXFastKernel] = {
    var result: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        result[heads] = MLXFast.metalKernel(
            name: "laguna_oproj_act_h\(heads)_v1"
                + (lagunaNvfp4QmvSignCarryEnabled ? "_sc1" : "")
                + (lagunaNvfp4QmvSeedElisionEnabled ? "_se1" : "")
                + "_hs1",
            inputNames: [
                "attention_output", "gate_values", "weight_codes",
                "weight_scales", "oproj_escape_buf",
            ],
            outputNames: ["projected"],
            source: lagunaGatedAffineOProjNVFP4Source(heads: heads, preActivatedGate: true, halvedScales: true),
            ensureRowContiguous: true)
    }
    return result
}()

func lagunaGatedAffineOProjNVFP4(
    attentionOutput: MLXArray,
    gateLogits: MLXArray,
    codes: MLXArray,
    scales: MLXArray,
    heads: Int,
    gateIsActivated: Bool = false,
    scalesEscape: MLXArray? = nil
) -> MLXArray? {
    let halved = lagunaOProjScaleHalvingEnabled
    let selected: MLXFast.MLXFastKernel?
    if gateIsActivated {
        guard lagunaGateSoftplusEnabled else { return nil }
        selected = halved
            ? lagunaActivatedOProjHalvedKernels[heads]
            : lagunaActivatedOProjKernels[heads]
    } else {
        selected = halved
            ? lagunaGatedAffineOProjNVFP4HalvedKernels[heads]
            : lagunaGatedAffineOProjNVFP4Kernels[heads]
    }
    guard let kernel = selected else { return nil }
    let inVec = heads * LagunaConstants.headDim
    let outVec = LagunaConstants.hiddenSize
    guard attentionOutput.dtype == .bfloat16,
        attentionOutput.dims(1, 1, inVec),
        gateLogits.dtype == .bfloat16,
        gateLogits.dims(1, 1, heads),
        codes.dtype == .uint32,
        codes.dims(outVec, inVec / 8),
        scales.dtype == .uint8,
        scales.dims(outVec, halved ? inVec / 32 : inVec / 16)
    else {
        return nil
    }

    if halved, let escape = scalesEscape {
        guard escape.dtype == .uint8, escape.size == 1 else { return nil }
        lagunaTrace("gated affine oproj nvfp4 qmv h\(heads)")
        return kernel(
            [attentionOutput, gateLogits, codes, scales, escape],
            grid: ((outVec / 8) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[1, 1, outVec]],
            outputDTypes: [.bfloat16]
        )[0]
    }
    if halved { return nil }

    lagunaTrace("gated affine oproj nvfp4 qmv h\(heads)")
    return kernel(
        [attentionOutput, gateLogits, codes, scales],
        grid: ((outVec / 8) * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, outVec]],
        outputDTypes: [.bfloat16]
    )[0]
}


/// Exact E4M3 scale decode and stock-order 16-value qdot for the fused tail.
/// The stock path shifts the 7-bit magnitude into a half (thereby dividing by
/// 256), restores that factor in half, applies the sign, then multiplies its
/// float scale by 16384 before the accumulated qdot. The enabled path folds
/// both powers of two into one exact float multiply (`256 * 16384 = 2^22`).
/// `bits + (bits & 128)` moves E4M3's sign to half bit 15 when shifted while
/// retaining the magnitude. Every E4M3 magnitude is finite; the half-form
/// intermediate is at most 1.875, so neither formulation rounds or overflows.
/// An exhaustive comparison over all 256 scale bytes is bit-identical,
/// including both signed zeros. The environment switch keeps the original
/// expression as a direct timing and correctness control.
private let lagunaTailNVFP4ScaleFoldEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_TAIL_NVFP4_SCALE_FOLD"] != "0"

/// `DARKBLOOM_QKV_TAIL_FOLD` (default ON for the ranked run; set "0" to restore
/// the frontier q/k/v QMV): ports the two exactness-proven ALU
/// micro-elisions the o_proj (`lagunaGatedAffineOProjNVFP4Source`) and MoE
/// (`lagunaSharedSwiGLUQMVHeader`) NVFP4 QMVs already ship into the decode
/// q/k/v NVFP4 tail QMV (`lagunaDecodeNVFP4QKVR1Source` +
/// `lagunaTailNVFP4QMVHeader`): (1) seed elision -- assign the first four-term
/// product group instead of adding it to a dead `+0.0f` seed; and (2) scale
/// defer -- return the RAW half from `laguna_tail_nvfp4_scale` and apply the
/// folded `2^22` once per output row at the R1 epilogue instead of once per
/// 16-value group. Both are pure instruction-count reductions: identical bytes,
/// loads, geometry and reduction order (the R1 one-row-per-simdgroup schedule
/// and the `simd_sum` order are untouched). When OFF the generated kernel
/// source, name and dispatch are byte-identical to the frontier q/k/v QMV;
/// ranked correctness is the CXXC exact-token gate.
private let lagunaTailNVFP4QKVTailFoldEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_QKV_TAIL_FOLD"] != "0"

/// Seed-elision half of `DARKBLOOM_QKV_TAIL_FOLD`, kept as its own constant so
/// the two elisions can later split behind independent env sub-flags.
let lagunaTailNVFP4QKVSeedElisionEnabled = lagunaTailNVFP4QKVTailFoldEnabled

/// Scale-defer half of `DARKBLOOM_QKV_TAIL_FOLD`. Composes only with the active
/// fold arm, exactly like MoE's
/// `lagunaNvfp4ScaleDeferEnabled && lagunaNvfp4ScaleFoldEnabled`: deferring the
/// `2^22` only makes sense once the fold has parked it in the scale.
let lagunaTailNVFP4QKVScaleDeferEnabled =
    lagunaTailNVFP4QKVTailFoldEnabled && lagunaTailNVFP4ScaleFoldEnabled

/// Body of `laguna_tail_nvfp4_scale`. The folded arm parks `256 * 16384 == 2^22`
/// in the scale; under scale-defer it returns the RAW half and the `2^22` is
/// re-applied once per output row by `lagunaTailNVFP4RowScaleSuffixSource` at
/// the R1 epilogue. Explicit bool params (no env) so both arms are exercisable
/// under plain `swift test`, mirroring `lagunaGatedAffineOProjNVFP4Source`. The
/// non-defer and unfolded arms reproduce the frontier text verbatim.
func lagunaTailNVFP4ScaleDecodeSource(scaleFold: Bool, scaleDefer: Bool) -> String {
    guard scaleFold else {
        return "    half converted = as_type<half>(raw);\n"
            + "    converted *= 256.0;\n"
            + "    half signed_value = (bits & 128) ? -converted : converted;\n"
            + "    return float(signed_value);"
    }
    return scaleDefer
        ? "    return float(as_type<half>(raw));"
        : "    return float(as_type<half>(raw)) * 4194304.0f;"
}

/// Accumulator declaration for `laguna_tail_nvfp4_qdot`: the seed-elided arm
/// drops the dead `= 0` initializer because the first group now assigns.
func lagunaTailNVFP4QDotAccumDeclSource(seedElide: Bool) -> String {
    seedElide ? "float accum;" : "float accum = 0;"
}

/// First four-term product group of `laguna_tail_nvfp4_qdot`. Seed-elided, the
/// first code word (`j == 0`) assigns the accumulator; every other word adds.
/// The `if (j == 0)` / `else` split and the multiply/add expressions are the
/// exact text o_proj's `firstAccum` emits, so the association is untouched --
/// only a signed-zero can differ, absorbed by the `+0.0f`-seeded `result` and
/// the BF16 epilogue. The non-elided arm reproduces the frontier group verbatim.
func lagunaTailNVFP4QDotFirstGroupSource(seedElide: Bool) -> String {
    seedElide
        ? "if (j == 0) {\n"
            + "                accum =\n"
            + "                    (x_thread[8 * j] * v04.x +\n"
            + "                     x_thread[8 * j + 1] * v15.x +\n"
            + "                     x_thread[8 * j + 2] * v26.x +\n"
            + "                     x_thread[8 * j + 3] * v37.x);\n"
            + "            } else {\n"
            + "                accum +=\n"
            + "                    (x_thread[8 * j] * v04.x +\n"
            + "                     x_thread[8 * j + 1] * v15.x +\n"
            + "                     x_thread[8 * j + 2] * v26.x +\n"
            + "                     x_thread[8 * j + 3] * v37.x);\n"
            + "            }"
        : "accum +=\n"
            + "            (x_thread[8 * j] * v04.x +\n"
            + "             x_thread[8 * j + 1] * v15.x +\n"
            + "             x_thread[8 * j + 2] * v26.x +\n"
            + "             x_thread[8 * j + 3] * v37.x);"
}

/// The deferred `2^22` re-applied once per output row at the R1 epilogue when
/// scale-defer is active (empty otherwise), mirroring MoE's
/// `lagunaNvfp4RowScaleSuffix`.
func lagunaTailNVFP4RowScaleSuffixSource(scaleDefer: Bool) -> String {
    scaleDefer ? " * 4194304.0f" : ""
}

private let lagunaTailNVFP4QDotReturn = lagunaTailNVFP4ScaleFoldEnabled
    ? "return scale * accum;"
    : "return (scale * 16384.0f) * accum;"

private let lagunaTailNVFP4QMVHeader = """
    static inline float laguna_tail_nvfp4_scale(uint8_t bits) {
        \(lagunaTailNVFP4ScaleFoldEnabled
            ? (lagunaE4M3SignDomainCertified
                ? "ushort raw = ushort(bits) << 7;"
                : "ushort raw = ushort(bits + (bits & 128)) << 7;")
            : "ushort raw = ushort(bits & 127) << 7;")
        \(lagunaTailNVFP4ScaleDecodeSource(scaleFold: lagunaTailNVFP4ScaleFoldEnabled, scaleDefer: lagunaTailNVFP4QKVScaleDeferEnabled))
    }

    static inline float laguna_tail_nvfp4_qdot(
        const device uint8_t* w,
        const thread float* x_thread,
        float scale
    ) {
        \(lagunaTailNVFP4QDotAccumDeclSource(seedElide: lagunaTailNVFP4QKVSeedElisionEnabled))
        const device uint2* wq = (const device uint2*)w;
        const uint2 codes = wq[0];
    #pragma unroll
        for (int j = 0; j < 2; j++) {
            const uint32_t c = (j == 0) ? codes.x : codes.y;
            // Split-nibble decode: the same eight `half` bit patterns per
            // code word as the original shift+mask sequence, in fewer
            // integer ops with three mask constants instead of eight — the
            // form the current stock `fp_qmv_fast` compiles (every form is
            // an OR of masked shifts, so the decode is bit-identical).
            const uint32_t xe = c & 0x0F0F0F0Fu;
            const uint32_t ge = xe | (xe << 3);
            const uint32_t yo = c & 0xF0F0F0F0u;
            const uint32_t go = yo | (yo >> 3);
            const uint32_t p0 = (ge << 9) & 0x8E008E00u;
            const uint32_t p1 = (go << 8) & 0x8E008E00u;
            const uint32_t p2 = (ge << 1) & 0x8E008E00u;
            const uint32_t p3 = go & 0x8E008E00u;
            const float2 v04 = float2(as_type<half2>(p0));
            const float2 v15 = float2(as_type<half2>(p1));
            const float2 v26 = float2(as_type<half2>(p2));
            const float2 v37 = float2(as_type<half2>(p3));
            \(lagunaTailNVFP4QDotFirstGroupSource(seedElide: lagunaTailNVFP4QKVSeedElisionEnabled))
            accum +=
                (x_thread[8 * j + 4] * v04.y +
                 x_thread[8 * j + 5] * v15.y +
                 x_thread[8 * j + 6] * v26.y +
                 x_thread[8 * j + 7] * v37.y);
        }
        \(lagunaTailNVFP4QDotReturn)
    }
    """

/// Decode-only, static-shape R1 schedule for the live group-16 NVFP4 QKV
/// bank. The generated QMV gives each SIMD group four output rows. This twin
/// keeps every row's K order, FP32 accumulator, simd reduction and BF16 cast
/// intact while giving each SIMD one row, matching the M5-positive routed and
/// shared expert schedules. Multi-token prefill cannot pass the shape guard.
private let lagunaDecodeNVFP4QKVR1Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_DECODE_NVFP4_QKV_R1"] != "0"

private let lagunaQKVScaleHalvingEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_QKV_SCALE_HALVING"] != "0"

private func lagunaDecodeNVFP4QKVR1Source(
    heads: Int, halvedScales: Bool = false, withGProj: Bool = false
) -> String {
    let ss = halvedScales ? "in_vec_size_g / 2" : "in_vec_size_g"
    let sl = halvedScales ? "simd_lid / 2" : "simd_lid"
    let st = halvedScales ? "block_size / 32" : "block_size / 16"
    let d = halvedScales ? """
        constexpr uint q_rows = \(heads * LagunaConstants.headDim);
        constexpr uint kv_rows = \(LagunaConstants.numKeyValueHeads * LagunaConstants.headDim);
""" : ""
    let r: String
    if halvedScales {
        r = """
        uint8_t sbits = sc[0];
        if (simd_lid == 1 && k == 0) {
            if (out_row == 0) sbits = qkv_escape[0];
            else if (out_row == q_rows) sbits = qkv_escape[1];
            else if (out_row == q_rows + kv_rows) sbits = qkv_escape[2];
        }
        result += laguna_tail_nvfp4_qdot(ws, x_thread, laguna_tail_nvfp4_scale(sbits));
"""
    } else {
        r = """
        result += laguna_tail_nvfp4_qdot(ws, x_thread, laguna_tail_nvfp4_scale(sc[0]));
"""
    }
    let qkvRows = "(\(heads) + 2 * \(LagunaConstants.numKeyValueHeads)) * \(LagunaConstants.headDim)"
    let gProjDecl = withGProj ? """
    constexpr uint gk = axis_size, ggs = 32, gv = 8, gkg = gk / ggs, gss = ggs / gv;
    constexpr uint g_bk = gv * 32;
""" : ""
    let qkvRowsConst = withGProj
        ? "    constexpr uint qkv_rows = \(qkvRows);"
        : ""
    let gateRowDecl = withGProj ? "    uint gate_row = out_row - qkv_rows;" : ""
    let qkvBranch = withGProj ? "    if (out_row < qkv_rows) {" : ""
    let gProjBranch = withGProj ? """
    } else {
        const device uint8_t* gws =
            (const device uint8_t*)gproj_codes + gate_row * gk + simd_lid * gv;
        const device bfloat* gpm =
            gproj_metadata + gate_row * gkg * 2 + (simd_lid / gss) * 2;
        thread float gate_result = 0.0f;

        for (uint gk_iter = 0; gk_iter < gk; gk_iter += g_bk) {
            thread float gx[gv];
            uint gcol = simd_lid * gv + gk_iter;
            float sum = 0.0f;
            for (uint i = 0; i < gv; ++i) {
                gx[i] = float(normalized[gcol + i]);
                sum += gx[i];
            }
            float s, b;
            uint gl = (simd_lid / gss) * gss;
            if (simd_lid == gl) {
                s = float(gpm[0]); b = float(gpm[1]);
            }
            s = simd_shuffle(s, gl); b = simd_shuffle(b, gl);
            float a = 0.0f;
            for (uint i = 0; i < gv; ++i)
                a += gx[i] * gws[gk_iter + i];
            gate_result += s * a + sum * b;
            gpm += (g_bk / ggs) * 2;
        }

        gate_result = simd_sum(gate_result);
        if (simd_lid == 0) {
            float l = float(bfloat(gate_result));
            float g;
            if (metal::isnan(l)) g = NAN;
            else {
                float hi = metal::max(l, 0.0f);
                float lo = metal::min(l, 0.0f);
                g = (metal::isinf(lo) || metal::isinf(hi))
                    ? hi : hi + log1p(metal::exp(lo - hi));
            }
            gate_values[gate_row] = bfloat(g);
        }
    }
""" : ""
    let outputCode = withGProj ? """
    if (out_row < qkv_rows) {
        result = simd_sum(result\(lagunaTailNVFP4RowScaleSuffixSource(scaleDefer: lagunaTailNVFP4QKVScaleDeferEnabled)));
        if (simd_lid == 0) {
            projected[out_row] = bfloat(result);
        }
    }
""" : """
    result = simd_sum(result\(lagunaTailNVFP4RowScaleSuffixSource(scaleDefer: lagunaTailNVFP4QKVScaleDeferEnabled)));
    if (simd_lid == 0) {
        projected[out_row] = bfloat(result);
    }
"""
    return """
    constexpr uint axis_size = 2048;
    constexpr uint num_simdgroups = 2;
    constexpr uint values_per_thread = 16;
    constexpr uint block_size = 512;
    constexpr uint in_vec_size_w = axis_size / 2;
    constexpr uint in_vec_size_g = axis_size / 16;
    \(d)
    \(gProjDecl)
    \(qkvRowsConst)

    uint tile = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    uint out_row = tile * num_simdgroups + simd_gid;
    \(gateRowDecl)

    thread float result = 0.0f;
    \(qkvBranch)

    thread float x_thread[values_per_thread];
    const device uint8_t* ws = (const device uint8_t*)weight_codes +
        out_row * in_vec_size_w + simd_lid * 8;
    const device uint8_t* sc = weight_scales +
        out_row * (\(ss)) + (\(sl));

    uint column = simd_lid * values_per_thread;
    for (uint k = 0; k < axis_size; k += block_size) {
        for (uint i = 0; i < values_per_thread; ++i) {
            x_thread[i] = float(normalized[column + i]);
        }
        \(r)
        ws += block_size / 2;
        sc += \(st);
        column += block_size;
    }
    \(gProjBranch)

    \(outputCode)
    """
}

private let lagunaDecodeNVFP4QKVR1Kernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        kernels[heads] = MLXFast.metalKernel(
            name: "laguna_decode_nvfp4_qkv_h\(heads)_r1_v1"
                + (lagunaTailNVFP4QKVSeedElisionEnabled ? "_se1" : "")
                + (lagunaTailNVFP4QKVScaleDeferEnabled ? "_sd1" : ""),
            inputNames: ["normalized", "weight_codes", "weight_scales"],
            outputNames: ["projected"],
            source: lagunaDecodeNVFP4QKVR1Source(heads: heads),
            header: lagunaTailNVFP4QMVHeader,
            ensureRowContiguous: true)
    }
    return kernels
}()

private let lagunaDecodeNVFP4QKVR1HalvedKernels: [Int: MLXFast.MLXFastKernel] = {
    var k: [Int: MLXFast.MLXFastKernel] = [:]
    for h in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        let s = (lagunaTailNVFP4QKVSeedElisionEnabled ? "_se1" : "")
            + (lagunaTailNVFP4QKVScaleDeferEnabled ? "_sd1" : "")
        k[h] = MLXFast.metalKernel(
            name: "laguna_decode_nvfp4_qkv_h\(h)_r1_v1\(s)_hs1",
            inputNames: ["normalized", "weight_codes", "weight_scales", "qkv_escape"],
            outputNames: ["projected"],
            source: lagunaDecodeNVFP4QKVR1Source(heads: h, halvedScales: true),
            header: lagunaTailNVFP4QMVHeader, ensureRowContiguous: true)
    }
    return k
}()

private func lagunaDecodeNVFP4QKVR1(
    normalized: MLXArray,
    bank: LagunaNativeAffineWeight,
    heads: Int
) -> MLXArray? {
    guard lagunaDecodeNVFP4QKVR1Enabled else { return nil }
    let halved = lagunaQKVScaleHalvingEnabled
    let rows = (heads + 2 * LagunaConstants.numKeyValueHeads) * LagunaConstants.headDim
    let hidden = LagunaConstants.hiddenSize
    guard normalized.dtype == .bfloat16,
        normalized.dims(1, 1, hidden),
        bank.mode == .nvfp4, bank.bits == 4, bank.groupSize == 16,
        bank.biases == nil,
        bank.originalShape == [rows, hidden],
        bank.packedCodes.dtype == .uint32,
        bank.packedCodes.dims(rows, hidden / 8),
        bank.scales.dtype == .uint8,
        bank.scales.dims(rows, halved ? hidden / 32 : hidden / 16),
        rows % 2 == 0,
        let kernel = halved
            ? lagunaDecodeNVFP4QKVR1HalvedKernels[heads]
            : lagunaDecodeNVFP4QKVR1Kernels[heads]
    else { return nil }
    var inputs = [normalized, bank.packedCodes, bank.scales]
    if halved, let esc = bank.qkvEscape, esc.dims(3) { inputs.append(esc) } else if halved { return nil }
    return kernel(inputs, grid: ((rows / 2) * 64, 1, 1), threadGroup: (64, 1, 1), outputShapes: [[1, 1, rows]], outputDTypes: [.bfloat16])[0]
}

private let lagunaFusedGProjQKVKernels: [Int: MLXFast.MLXFastKernel] = {
    var k: [Int: MLXFast.MLXFastKernel] = [:]
    for h in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        let s = (lagunaTailNVFP4QKVSeedElisionEnabled ? "_se1" : "")
            + (lagunaTailNVFP4QKVScaleDeferEnabled ? "_sd1" : "")
        k[h] = MLXFast.metalKernel(
            name: "laguna_decode_nvfp4_qkv_h\(h)_r1_v1\(s)_fgp",
            inputNames: ["normalized", "weight_codes", "weight_scales",
                         "gproj_codes", "gproj_metadata"],
            outputNames: ["projected", "gate_values"],
            source: lagunaDecodeNVFP4QKVR1Source(heads: h, withGProj: true),
            header: lagunaTailNVFP4QMVHeader, ensureRowContiguous: true)
    }
    return k
}()

private let lagunaFusedGProjQKVHalvedKernels: [Int: MLXFast.MLXFastKernel] = {
    var k: [Int: MLXFast.MLXFastKernel] = [:]
    for h in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        let s = (lagunaTailNVFP4QKVSeedElisionEnabled ? "_se1" : "")
            + (lagunaTailNVFP4QKVScaleDeferEnabled ? "_sd1" : "")
        k[h] = MLXFast.metalKernel(
            name: "laguna_decode_nvfp4_qkv_h\(h)_r1_v1\(s)_hs1_fgp",
            inputNames: ["normalized", "weight_codes", "weight_scales",
                         "qkv_escape", "gproj_codes", "gproj_metadata"],
            outputNames: ["projected", "gate_values"],
            source: lagunaDecodeNVFP4QKVR1Source(heads: h, halvedScales: true, withGProj: true),
            header: lagunaTailNVFP4QMVHeader, ensureRowContiguous: true)
    }
    return k
}()

/// Fuses the g_proj INT8 affine matmul + softplus into the NVFP4 QKV R1
/// dispatch, producing both QKV projections and activated gate values in a
/// single Metal dispatch. Bit-exact: the g_proj computation matches
/// `lagunaGateSoftplusSource` exactly (same K-loop, same group-32 affine
/// scale/bias, same simd_sum reduction order, same softplus formula). The
/// only difference is WHERE the computation runs (extra threadgroups in the
/// QKV grid instead of a separate dispatch).
private func lagunaFusedGProjQKV(
    normalized: MLXArray,
    bank: LagunaNativeAffineWeight,
    gproj: LagunaNativeAffineWeight,
    heads: Int
) -> (qkv: MLXArray, gateValues: MLXArray)? {
    guard lagunaDecodeNVFP4QKVR1Enabled,
        lagunaGateSoftplusEnabled,
        let gprojMeta = gproj.interleavedMetadata
    else { return nil }
    let halved = lagunaQKVScaleHalvingEnabled
    let rows = (heads + 2 * LagunaConstants.numKeyValueHeads) * LagunaConstants.headDim
    let hidden = LagunaConstants.hiddenSize
    guard normalized.dtype == .bfloat16,
        normalized.dims(1, 1, hidden),
        bank.mode == .nvfp4, bank.bits == 4, bank.groupSize == 16,
        bank.biases == nil,
        bank.originalShape == [rows, hidden],
        bank.packedCodes.dtype == .uint32,
        bank.packedCodes.dims(rows, hidden / 8),
        bank.scales.dtype == .uint8,
        bank.scales.dims(rows, halved ? hidden / 32 : hidden / 16),
        gproj.mode == .affine, gproj.bits == 8, gproj.groupSize == 32,
        gproj.packedCodes.dims(heads, hidden / 4),
        gprojMeta.dims(heads, hidden / 16),
        rows % 2 == 0, heads % 2 == 0,
        let kernel = halved
            ? lagunaFusedGProjQKVHalvedKernels[heads]
            : lagunaFusedGProjQKVKernels[heads]
    else { return nil }
    var inputs: [MLXArray] = [normalized, bank.packedCodes, bank.scales]
    if halved, let esc = bank.qkvEscape, esc.dims(3) {
        inputs.append(esc)
    } else if halved { return nil }
    inputs.append(gproj.packedCodes)
    inputs.append(gprojMeta)
    let totalRows = rows + heads
    let outputs = kernel(
        inputs,
        grid: ((totalRows / 2) * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, rows], [1, 1, heads]],
        outputDTypes: [.bfloat16, .bfloat16])
    return (outputs[0], outputs[1])
}


private func lagunaNormAffineQKVSource(rows: Int, staged: Bool) -> String {
    let stagedNormalize = """
            for (uint j = 0; j < virtual_per_thread; ++j) {
                uint base = (lid + j * real_threads) * n_reads;
                for (uint i = 0; i < n_reads; ++i) {
                    norm_row[base + i] =
                        norm_weight[base + i] *
                        bfloat(float(residual[base + i]) * laguna_inv_mean);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        """
    let scratch =
        staged
        ? "threadgroup bfloat norm_row[axis_size];"
        : "// inline: no staged row, no third barrier"
    let normalize = staged ? stagedNormalize : ""
    let loadValue =
        staged
        ? "float value = float(norm_row[column + i]);"
        : """
        float value = float(bfloat(
                        norm_weight[column + i] *
                        bfloat(float(residual[column + i]) * laguna_inv_mean)));
        """
    return """
    \(lagunaNormAffineQKVBody(rows: rows, scratch: scratch, normalize: normalize, loadValue: loadValue))
    """
}

private func lagunaNormAffineQKVBody(
    rows: Int, scratch: String, normalize: String, loadValue: String
) -> String {
    return """
    constexpr uint axis_size = \(LagunaConstants.hiddenSize);
    constexpr uint out_vec_size = \(rows);
    constexpr uint n_reads = 4;                 // RMS_N_READS
    constexpr uint norm_threads = axis_size / n_reads;   // 512 virtual threads
    constexpr uint real_threads = 64;
    constexpr uint virtual_per_thread = norm_threads / real_threads;  // 8
    constexpr uint simd_size = 32;
    constexpr float norm_eps = 1.0e-6f;
    constexpr uint values_per_thread = 8;       // pack_factor 4 * packs_per_thread 2
    constexpr uint block_size = 256;            // values_per_thread * SIMD_SIZE
    constexpr uint results_per_simdgroup = 4;
    constexpr uint num_simdgroups = 2;
    constexpr uint group_size = 32;
    constexpr uint scale_step_per_thread = group_size / values_per_thread;
    constexpr uint in_vec_size_g = axis_size / group_size;

    uint tile = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[simd_size];
    \(scratch)

    // --- rms_single_row replica, 512 virtual threads over 64 real ones ---
    if (lid < simd_size) {
        local_sums[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < virtual_per_thread; ++j) {
        uint base = (lid + j * real_threads) * n_reads;
        float acc = 0.0f;
        for (uint i = 0; i < n_reads; ++i) {
            float xi = float(residual[base + i]);
            acc += xi * xi;
        }
        acc = simd_sum(acc);
        if (simd_lid == 0) {
            local_sums[simd_gid + num_simdgroups * j] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_gid == 0) {
        float total = simd_sum(local_sums[simd_lid]);
        if (simd_lid == 0) {
            local_inv_mean[0] =
                metal::precise::rsqrt(total / float(axis_size) + norm_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float laguna_inv_mean = local_inv_mean[0];
    \(normalize)
    // --- affine_qmv_fast replica over the normalized row ---
    uint out_row = tile * (num_simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;

    const device uint8_t* ws = (const device uint8_t*)weight_codes +
        out_row * axis_size + simd_lid * values_per_thread;
    const device bfloat* sc = weight_scales + out_row * in_vec_size_g +
        simd_lid / scale_step_per_thread;
    const device bfloat* bs = weight_biases + out_row * in_vec_size_g +
        simd_lid / scale_step_per_thread;

    thread float x_thread[values_per_thread];
    thread float result[results_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint column = simd_lid * values_per_thread;
    for (uint k = 0; k < axis_size; k += block_size) {
        float sum = 0.0f;
        for (uint i = 0; i < values_per_thread; ++i) {
            \(loadValue)
            sum += value;
            x_thread[i] = value;
        }

        for (uint row = 0; row < results_per_simdgroup; ++row) {
            const device uint8_t* wl = ws + row * axis_size;
            float scale, bias;
            uint gl = (simd_lid / scale_step_per_thread) * scale_step_per_thread;
            if (simd_lid == gl) {
                scale = float(sc[row * in_vec_size_g]);
                bias = float(bs[row * in_vec_size_g]);
            }
            scale = simd_shuffle(scale, gl);
            bias = simd_shuffle(bias, gl);
            float accum = 0.0f;
            for (uint i = 0; i < values_per_thread; ++i) {
                accum += x_thread[i] * wl[i];
            }
            result[row] += scale * accum + sum * bias;
        }

        ws += block_size;
        sc += block_size / group_size;
        bs += block_size / group_size;
        column += block_size;
    }

    {
        const vec<float, 4> packed = simd_sum(
            vec<float, 4>(result[0], result[1], result[2], result[3]));
        result[0] = packed.x; result[1] = packed.y;
        result[2] = packed.z; result[3] = packed.w;
    }
    if (simd_lid == 0) {
        for (uint row = 0; row < results_per_simdgroup; ++row) {
            projected[out_row + row] = bfloat(result[row]);
        }
    }
    """
}

/// `DARKBLOOM_NORM_AFFINE_QKV_STAGE` = `inline` (default) or `tg`.
private let lagunaNormAffineQKVStaged =
    ProcessInfo.processInfo.environment["DARKBLOOM_NORM_AFFINE_QKV_STAGE"] == "tg"

/// `DARKBLOOM_NORM_AFFINE_QKV_PF` (default 4 = register-prefetch depth 4,
/// **DEFAULT ON**; set "0" to restore the stock inline kernel; 1...4
/// clamped). QmvLimiter study (notes/exp-qmvlimiter.md): the fused kernel's
/// RMS prologue (three barriers + the 2048-element reduction) runs
/// dead-serial BEFORE the first weight byte of each threadgroup's short
/// 18 KB stream, costing ~9% of the dispatch (harness noprol probe); the
/// prefetch variant issues the first `depth` k-blocks' code/scale/bias
/// loads ABOVE the prologue so the weight stream is in flight while the
/// reduction runs, then consumes them from registers in the exact stock
/// order. Pure reads of immutable weights and an identical FP schedule ->
/// bit-identical (standalone-harness memcmp on all output rows, random
/// banks, r10304 and r8240; model-level 130/130 max_abs_diff=0). Depth-4
/// measured -11.9%/-12.2% kernel-level at decode clocks (48.61->42.84 us
/// r10304, 39.34->34.55 us r8240; 554/549 GB/s vs 488/482 stock; projected
/// ~-171 us/step), depth curve monotone pf1->pf4, occupancy unchanged
/// (maxTotalThreadsPerThreadgroup 1024, +64 B thread state). Ignored under
/// the `tg` staged variant.
let lagunaNormAffineQKVPrefetchDepth: Int = {
    let raw =
        ProcessInfo.processInfo.environment["DARKBLOOM_NORM_AFFINE_QKV_PF"] ?? "4"
    return min(max(Int(raw) ?? 4, 0), 4)
}()

/// Register-prefetch twin of the inline `lagunaNormAffineQKVSource`. The
/// NORM half and the k-loop arithmetic are textually the stock inline
/// variant's; the only changes are (a) the stream-pointer setup and the
/// first `depth` k-blocks' weight/scale/bias loads hoisted above the
/// prologue into registers, and (b) those blocks peeled off the k-loop,
/// consuming the registers with the identical per-i accumulation order.
/// Same values, same operation order per output row -> bit-exact.
private func lagunaNormAffineQKVPrefetchSource(
    rows: Int, depth: Int, indexed: Bool = false
) -> String {
    let metadataPointers = indexed
        ? """
        const device ushort* mi = metadata_indices + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        uint g = simd_lid & ~3u;
        """
        : """
        const device bfloat* sc = weight_scales + out_row * in_vec_size_g +
            simd_lid / scale_step_per_thread;
        const device bfloat* bs = weight_biases + out_row * in_vec_size_g +
            simd_lid / scale_step_per_thread;
        """
    let prefetchMetadata = indexed
        ? """
            uint pair = simd_shuffle(simd_lid == g ? metadata_lut[mi[d * (block_size / group_size) + row * in_vec_size_g]] : 0, g);
            pf_s[d][row] = float(as_type<bfloat>(ushort(pair)));
            pf_b[d][row] = float(as_type<bfloat>(ushort(pair >> 16)));
        """
        : """
            float ps, pb;
            uint gl_pf = (simd_lid / scale_step_per_thread) * scale_step_per_thread;
            if (simd_lid == gl_pf) {
                ps = float(sc[d * (block_size / group_size) + row * in_vec_size_g]);
                pb = float(bs[d * (block_size / group_size) + row * in_vec_size_g]);
            }
            ps = simd_shuffle(ps, gl_pf);
            pb = simd_shuffle(pb, gl_pf);
            pf_s[d][row] = ps;
            pf_b[d][row] = pb;
        """
    let metadataLoad = indexed
        ? """
            uint pair = simd_shuffle(simd_lid == g ? metadata_lut[mi[row * in_vec_size_g]] : 0, g);
            float scale = float(as_type<bfloat>(ushort(pair)));
            float bias = float(as_type<bfloat>(ushort(pair >> 16)));
        """
        : """
            float scale, bias;
            uint gl = (simd_lid / scale_step_per_thread) * scale_step_per_thread;
            if (simd_lid == gl) {
                scale = float(sc[row * in_vec_size_g]);
                bias = float(bs[row * in_vec_size_g]);
            }
            scale = simd_shuffle(scale, gl);
            bias = simd_shuffle(bias, gl);
        """
    let metadataAdvance = indexed
        ? "mi += block_size / group_size;"
        : """
        sc += block_size / group_size;
        bs += block_size / group_size;
        """
    return """
    constexpr uint axis_size = \(LagunaConstants.hiddenSize);
    constexpr uint out_vec_size = \(rows);
    constexpr uint n_reads = 4;                 // RMS_N_READS
    constexpr uint norm_threads = axis_size / n_reads;   // 512 virtual threads
    constexpr uint real_threads = 64;
    constexpr uint virtual_per_thread = norm_threads / real_threads;  // 8
    constexpr uint simd_size = 32;
    constexpr float norm_eps = 1.0e-6f;
    constexpr uint values_per_thread = 8;       // pack_factor 4 * packs_per_thread 2
    constexpr uint block_size = 256;            // values_per_thread * SIMD_SIZE
    constexpr uint results_per_simdgroup = 4;
    constexpr uint num_simdgroups = 2;
    constexpr uint group_size = 32;
    constexpr uint scale_step_per_thread = group_size / values_per_thread;
    constexpr uint in_vec_size_g = axis_size / group_size;
    constexpr uint pf_depth = \(depth);

    uint tile = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[simd_size];

    // Stream pointers and the first pf_depth k-blocks' loads issued BEFORE
    // the norm reduction: the weight stream is in flight while the prologue
    // runs. Pure reads of immutable weights, consumed below in the stock
    // k-loop's exact per-i order.
    uint out_row = tile * (num_simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;

    const device uint8_t* ws = (const device uint8_t*)weight_codes +
        out_row * axis_size + simd_lid * values_per_thread;
    \(metadataPointers)

    uint8_t pf_w[pf_depth][results_per_simdgroup][values_per_thread];
    float pf_s[pf_depth][results_per_simdgroup];
    float pf_b[pf_depth][results_per_simdgroup];
    for (uint d = 0; d < pf_depth; ++d) {
        for (uint row = 0; row < results_per_simdgroup; ++row) {
            const device uint8_t* wl = ws + d * block_size + row * axis_size;
            for (uint i = 0; i < values_per_thread; ++i) {
                pf_w[d][row][i] = wl[i];
            }
            \(prefetchMetadata)
        }
    }

    // --- rms_single_row replica, textually the stock inline variant's ---
    if (lid < simd_size) {
        local_sums[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < virtual_per_thread; ++j) {
        uint base = (lid + j * real_threads) * n_reads;
        float acc = 0.0f;
        for (uint i = 0; i < n_reads; ++i) {
            float xi = float(residual[base + i]);
            acc += xi * xi;
        }
        acc = simd_sum(acc);
        if (simd_lid == 0) {
            local_sums[simd_gid + num_simdgroups * j] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_gid == 0) {
        float total = simd_sum(local_sums[simd_lid]);
        if (simd_lid == 0) {
            local_inv_mean[0] =
                metal::precise::rsqrt(total / float(axis_size) + norm_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float laguna_inv_mean = local_inv_mean[0];

    // --- affine_qmv_fast replica; first pf_depth blocks consume the
    // prefetched registers with the identical accumulation order ---
    thread float x_thread[values_per_thread];
    thread float result[results_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint column = simd_lid * values_per_thread;
    for (uint d = 0; d < pf_depth; ++d) {
        float sum = 0.0f;
        for (uint i = 0; i < values_per_thread; ++i) {
            float value = float(bfloat(
                            norm_weight[column + i] *
                            bfloat(float(residual[column + i]) * laguna_inv_mean)));
            sum += value;
            x_thread[i] = value;
        }
        for (uint row = 0; row < results_per_simdgroup; ++row) {
            float accum = 0.0f;
            for (uint i = 0; i < values_per_thread; ++i) {
                accum += x_thread[i] * pf_w[d][row][i];
            }
            result[row] += pf_s[d][row] * accum + sum * pf_b[d][row];
        }
        ws += block_size;
        \(metadataAdvance)
        column += block_size;
    }
    for (uint k = pf_depth * block_size; k < axis_size; k += block_size) {
        float sum = 0.0f;
        for (uint i = 0; i < values_per_thread; ++i) {
            float value = float(bfloat(
                            norm_weight[column + i] *
                            bfloat(float(residual[column + i]) * laguna_inv_mean)));
            sum += value;
            x_thread[i] = value;
        }

        for (uint row = 0; row < results_per_simdgroup; ++row) {
            const device uint8_t* wl = ws + row * axis_size;
            \(metadataLoad)
            float accum = 0.0f;
            for (uint i = 0; i < values_per_thread; ++i) {
                accum += x_thread[i] * wl[i];
            }
            result[row] += scale * accum + sum * bias;
        }

        ws += block_size;
        \(metadataAdvance)
        column += block_size;
    }

    for (uint row = 0; row < results_per_simdgroup; ++row) {
        result[row] = simd_sum(result[row]);
        if (simd_lid == 0) {
            projected[out_row + row] = bfloat(result[row]);
        }
    }
    """
}

/// One kernel per reachable `[Q; K; V; (G)]` row count: both head families,
/// gate rows folded in or not. All four are multiples of 8, so `qmv`'s `fast`
/// predicate holds and no tail threadgroup is ever dispatched.
private let lagunaNormAffineQKVKernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    let kvRows = 2 * LagunaConstants.numKeyValueHeads * LagunaConstants.headDim
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        for gateRows in [0, heads] {
            let rows = heads * LagunaConstants.headDim + kvRows + gateRows
            if kernels[rows] != nil { continue }
            let staged = lagunaNormAffineQKVStaged
            let pf = staged ? 0 : lagunaNormAffineQKVPrefetchDepth
            kernels[rows] = MLXFast.metalKernel(
                name: pf > 0
                    ? "laguna_norm_affine_qkv_qmv_i8g32_r\(rows)_pf\(pf)_v2"
                    : "laguna_norm_affine_qkv_qmv_i8g32_r\(rows)_"
                        + (staged ? "tg" : "inl") + "_v2",
                inputNames: [
                    "residual", "norm_weight", "weight_codes", "weight_scales",
                    "weight_biases",
                ],
                outputNames: ["projected"],
                source: pf > 0
                    ? lagunaNormAffineQKVPrefetchSource(rows: rows, depth: pf)
                    : lagunaNormAffineQKVSource(rows: rows, staged: staged),
                ensureRowContiguous: true
            )
        }
    }
    return kernels
}()

private let lagunaNormAffineQKVIndexedKernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    guard !lagunaNormAffineQKVStaged, lagunaNormAffineQKVPrefetchDepth > 0 else {
        return kernels
    }
    let kvRows = 2 * LagunaConstants.numKeyValueHeads * LagunaConstants.headDim
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        for gateRows in [0, heads] {
            let rows = heads * LagunaConstants.headDim + kvRows + gateRows
            if kernels[rows] != nil { continue }
            kernels[rows] = MLXFast.metalKernel(
                name: "laguna_norm_affine_qkv_qmv_i8g32_r\(rows)_"
                    + "pf\(lagunaNormAffineQKVPrefetchDepth)_idx_v2",
                inputNames: [
                    "residual", "norm_weight", "weight_codes",
                    "metadata_indices", "metadata_lut",
                ],
                outputNames: ["projected"],
                source: lagunaNormAffineQKVPrefetchSource(
                    rows: rows, depth: lagunaNormAffineQKVPrefetchDepth,
                    indexed: true),
                ensureRowContiguous: true
            )
        }
    }
    return kernels
}()

/// `DARKBLOOM_FUSED_NORM_AFFINE_QKV` (default ON; set "0" to disable). Folds
/// the input RMSNorm into the native group-32 affine INT8 QKV GEMV so the
/// decode attention head is one dispatch instead of two. Bit-exact (inline
/// variant re-derives normalized values from L1-resident rows with no
/// occupancy cost). Applies only on group-32 INT8 layers with the gate rows
/// folded into the QKV bank; the NVFP4 tail layers and any guard decline keep
/// the separate norm + projection.
let lagunaFusedNormAffineQKVEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_NORM_AFFINE_QKV"] != "0"

/// Input RMSNorm + native-affine INT8 `[Q; K; V; (G)]` projection in one
/// dispatch, or `nil` when any shape, dtype or wire-format guard declines
/// (caller then runs the exact two-dispatch chain).
func lagunaNormAffineQKV(
    residual: MLXArray,
    normWeight: MLXArray,
    codes: MLXArray,
    scales: MLXArray,
    biases: MLXArray,
    indexedMetadata: LagunaIndexedAffineMetadata? = nil,
    rows: Int
) -> MLXArray? {
    guard let kernel = lagunaNormAffineQKVKernels[rows] else { return nil }
    let hidden = LagunaConstants.hiddenSize
    guard residual.dtype == .bfloat16,
        residual.dims(1, 1, hidden),
        normWeight.dtype == .bfloat16,
        normWeight.dims(hidden),
        codes.dtype == .uint32,
        codes.dims(rows, hidden / 4),
        scales.dtype == .bfloat16,
        scales.dims(rows, hidden / 32),
        biases.dtype == .bfloat16,
        biases.dims(rows, hidden / 32)
    else {
        return nil
    }

    if let metadata = indexedMetadata,
        let indexedKernel = lagunaNormAffineQKVIndexedKernels[rows],
        metadata.indices.dtype == .uint16,
        metadata.indices.dims(rows, hidden / 32),
        metadata.lut.dtype == .uint32,
        metadata.lut.ndim == 1,
        metadata.lut.size <= 65_536
    {
        lagunaTrace(
            "norm+affine qkv qmv r\(rows) "
                + "pf\(lagunaNormAffineQKVPrefetchDepth) indexed")
        return indexedKernel(
            [residual, normWeight, codes, metadata.indices, metadata.lut],
            grid: ((rows / 8) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[1, 1, rows]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    lagunaTrace(
        lagunaNormAffineQKVPrefetchDepth > 0 && !lagunaNormAffineQKVStaged
            ? "norm+affine qkv qmv r\(rows) pf\(lagunaNormAffineQKVPrefetchDepth)"
            : "norm+affine qkv qmv r\(rows)")
    return kernel(
        [residual, normWeight, codes, scales, biases],
        grid: ((rows / 8) * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, rows]],
        outputDTypes: [.bfloat16]
    )[0]
}

/// Keep the stock shapeless unary gate for prefill. Ranked measurement showed
/// the larger gate/product graph regressing the complete prefill schedule even
/// though its isolated steady-state subpath was slightly faster.
private let lagunaCompiledSoftplusGate: @Sendable (MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray) -> MLXArray = { gate in
        softplus(gate.asType(.float32)).asType(gate.dtype)
    }
    return MLXHardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

/// Decode-only outer compilation of the same gate product with the following
/// bias-free BF16 output projection. MLX keeps the matmul primitive intact but
/// schedules the elementwise producer and projection as one compiled graph,
/// avoiding a separate frontend boundary and shortening the gated vector's
/// lifetime. Prefill deliberately uses the smaller gate-only fusion.
private func makeLagunaAttentionGateProjection(
    heads: Int, headDim: Int
) -> @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray {
    let body: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = {
        output, projectedGate, weight in
        let batch = output.dim(0)
        let length = output.dim(1)
        let gate = softplus(projectedGate.asType(.float32)).asType(output.dtype)
        let gated = (
            output.reshaped(batch, length, heads, headDim)
                * gate[.ellipsis, .newAxis]
        ).reshaped(batch, length, heads * headDim)
        return matmul(gated, weight.T)
    }
    return MLXHardwareInfo.isCompiledDecodeSupported ? compile(body) : body
}

private let lagunaFullAttentionGateProjection = makeLagunaAttentionGateProjection(
    heads: LagunaConstants.fullAttentionHeads,
    headDim: LagunaConstants.headDim
)

private let lagunaSlidingAttentionGateProjection = makeLagunaAttentionGateProjection(
    heads: LagunaConstants.slidingAttentionHeads,
    headDim: LagunaConstants.headDim
)

/// Laguna attention: GQA with per-head QK-norm, per-layer-type RoPE (YaRN on
/// full-attention layers over the first half of the head, plain RoPE on
/// sliding layers over the whole head), and per-head softplus output gating.
/// Mirrors the vendored `LagunaAttention` forward exactly.
final class LagunaRuntimeAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let gatingEnabled: Bool
    let gatePerHead: Bool
    let isSliding: Bool
    let layerIdx: Int
    /// Retained `[1]` FP32 carrier of `scale` for the fused decode
    /// attention kernel (same float the stock SDPA call passes), built once
    /// so the per-step graph adds no fresh scalar upload for it.
    lazy var _fusedAttnScale: MLXArray = MLXArray([scale])
    let attentionGateProjection: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    /// Retained fused `[Wq; Wk; Wv]` weight (output rows concatenated, query
    /// rows first), built once after checkpoint load when
    /// `DARKBLOOM_FUSED_QKV` is enabled. Plain stored property with a leading
    /// underscore so Module reflection never treats this derived layout as a
    /// checkpoint parameter; the q/k/v `Linear` modules keep the original
    /// arrays for parameter integrity.
    var _fusedQKVWeight: MLXArray?

    /// Terminal-prefill-only BF16 side banks. Q and the per-head gate share
    /// the singleton final normalized row; K and V share every normalized
    /// supplied row. The authoritative modules remain intact for checkpoint
    /// loading and every fallback path.
    var _lastPrefillQGateWeight: MLXArray?
    var _lastPrefillKVWeight: MLXArray?

    /// Derived native group-32 affine layout for one serial decode token's
    /// Q/K/V batch. The original BF16 parameters remain authoritative and
    /// continue to serve prefill.
    var _nativeAffineQKV: LagunaNativeAffineWeight?

    /// Derived native group-32 affine layout for the attention output
    /// projection, used only by the serial decode call. `wo.weight` remains the
    /// authoritative parameter and continues to serve prefill, the last-row
    /// prefill path, and every decode fallback.
    var _nativeAffineOProj: LagunaNativeAffineWeight?

    /// Derived native group-32 affine INT8 layout for the attention per-head
    /// gate projection, used only by the serial decode call. Retained
    /// separately only on layers whose QKV bank is NOT group-32 INT8 (the
    /// NVFP4 tail window); on INT8-bank layers the gate rows concatenate into
    /// `_nativeAffineQKV` instead and `_nativeAffineQKVGateRows` records them.
    /// `gProj.weight` remains the authoritative parameter and continues to
    /// serve prefill, the last-row prefill path, and every decode fallback.
    var _nativeAffineGProj: LagunaNativeAffineWeight?

    /// Number of gate rows appended at the tail of the fused QKV bank (0 when
    /// the gate is not folded into the bank). The appended rows sit after the
    /// Q/K/V rows, so the decode call slices them out of the same dispatch's
    /// output at `queryDim + 2 * kvDim`.
    var _nativeAffineQKVGateRows = 0

    func prepareNativeAffineOProjWeight() -> [MLXArray] {
        guard _nativeAffineOProj == nil,
            type(of: wo) == Linear.self,
            wo.bias == nil,
            wo.weight.dims(LagunaConstants.hiddenSize, nHeads * headDim),
            let quantizedWO = lagunaNativeAffineWeight(wo.weight, layer: layerIdx)
        else {
            return []
        }
        var preparedWO = quantizedWO
        if preparedWO.mode == .affine, preparedWO.bits == 8,
            preparedWO.groupSize == 32, let biases = preparedWO.biases
        {
            preparedWO.indexedMetadata = lagunaIndexedAffineMetadata(
                scales: preparedWO.scales, biases: biases)
        }
        if preparedWO.mode == .nvfp4, preparedWO.bits == 4,
            preparedWO.groupSize == 16,
            lagunaOProjScaleHalvingEnabled
        {
            // Halve the scale plane: [outVec, inVec/16] -> [outVec, inVec/32].
            // Take even-indexed bytes; the kernel reads simd_lid/2.
            // Retain the original scale[0, 1] as the escape byte for the k=0
            // pair where the quantizer may break pairwise constancy.
            let oprojEscape = contiguous(preparedWO.scales[0, 1].reshaped([1]))
            let halved = contiguous(preparedWO.scales[0..., .stride(from: 0, by: 2)])
            preparedWO.scales = halved
            preparedWO.scalesEscape = oprojEscape
        }
        _nativeAffineOProj = preparedWO
        return preparedWO.arrays
    }

    func prepareNativeAffineQKVWeight() -> [MLXArray] {
        guard _nativeAffineQKV == nil,
            let q = lagunaNativeAffineWeight(wq.weight, layer: layerIdx),
            let k = lagunaNativeAffineWeight(wk.weight, layer: layerIdx),
            let v = lagunaNativeAffineWeight(wv.weight, layer: layerIdx)
        else {
            return []
        }
        // The per-head gate joins the same side layout where the envelope
        // allows it. It is always group-32 affine INT8 (never the NVFP4 tail
        // window), so it can concatenate into the QKV bank only when that
        // bank is itself group-32 INT8; on the NVFP4 tail layers it keeps a
        // separate bank and its own dispatch. Either way every output row's
        // K loop is independent of which rows share the dispatch, so folding
        // the gate rows into the bank does not move a single Q/K/V value.
        var gate: LagunaNativeAffineWeight?
        if lagunaNativeAffineGProjEnabled,
            lagunaUseNativeAffineGProj(layer: layerIdx),
            gatingEnabled, gatePerHead,
            let gProj,
            type(of: gProj) == Linear.self,
            gProj.bias == nil,
            gProj.weight.dtype == .bfloat16,
            gProj.weight.dims(nHeads, LagunaConstants.hiddenSize)
        {
            gate = lagunaNativeAffineGProjWeight(gProj.weight)
        }
        let foldGateIntoBank =
            gate != nil && q.groupSize == 32 && q.bits == 8 && q.mode == .affine
        var packedBlocks = [q.packedCodes, k.packedCodes, v.packedCodes]
        var scaleBlocks = [q.scales, k.scales, v.scales]
        var biasBlocks = [q.biases, k.biases, v.biases]
        var totalRows = wq.weight.dim(0) + wk.weight.dim(0) + wv.weight.dim(0)
        if foldGateIntoBank, let gate {
            packedBlocks.append(gate.packedCodes)
            scaleBlocks.append(gate.scales)
            biasBlocks.append(gate.biases)
            totalRows += nHeads
            _nativeAffineQKVGateRows = nHeads
        } else if let gate {
            _nativeAffineGProj = gate
        }
        let packedCodes = concatenated(packedBlocks, axis: 0)
        let scales = concatenated(scaleBlocks, axis: 0)
        var biases: MLXArray?
        if biasBlocks.allSatisfy({ $0 != nil }) {
            biases = concatenated(biasBlocks.compactMap { $0 }, axis: 0)
        }
        var fused = LagunaNativeAffineWeight(
            packedCodes: packedCodes,
            scales: scales,
            biases: biases,
            originalShape: [
                totalRows,
                wq.weight.dim(1),
            ],
            groupSize: q.groupSize,
            bits: q.bits,
            mode: q.mode
        )
        if fused.mode == .affine, fused.bits == 8, fused.groupSize == 32,
            let biases = fused.biases
        {
            fused.indexedMetadata = lagunaIndexedAffineMetadata(
                scales: fused.scales, biases: biases)
        }
        if fused.mode == .nvfp4, fused.bits == 4, fused.groupSize == 16,
            lagunaQKVScaleHalvingEnabled
        {
            let qR = wq.weight.dim(0), kvR = wk.weight.dim(0)
            fused.qkvEscape = contiguous(stacked([
                fused.scales[0, 1], fused.scales[qR, 1], fused.scales[qR + kvR, 1]
            ]).reshaped([3]))
            fused.scales = contiguous(fused.scales[0..., .stride(from: 0, by: 2)])
        }
        _nativeAffineQKV = fused
        return fused.arrays + (_nativeAffineGProj?.arrays ?? [])
    }

    /// Builds and retains the fused QKV weight from the loaded q/k/v
    /// projection weights. Called once after weights are installed and
    /// evaluated (before warmup); returns the new array so the caller can
    /// batch a single eval. Fuses only the exact stock configuration: three
    /// plain bias-free `Linear` projections of one dtype over the same input
    /// width, so the fused matmul is `matmul(x, w.T)` with every original
    /// output row unchanged.
    func prepareFusedQKVWeight() -> MLXArray? {
        guard _fusedQKVWeight == nil,
            type(of: wq) == Linear.self,
            type(of: wk) == Linear.self,
            type(of: wv) == Linear.self,
            wq.bias == nil, wk.bias == nil, wv.bias == nil,
            wq.weight.ndim == 2, wk.weight.ndim == 2, wv.weight.ndim == 2,
            wq.weight.dtype == wk.weight.dtype,
            wk.weight.dtype == wv.weight.dtype,
            wq.weight.dim(1) == wk.weight.dim(1),
            wk.weight.dim(1) == wv.weight.dim(1),
            wq.weight.dim(0) == nHeads * headDim,
            wk.weight.dim(0) == nKVHeads * headDim,
            wv.weight.dim(0) == nKVHeads * headDim
        else {
            return nil
        }
        let fused = concatenated([wq.weight, wk.weight, wv.weight], axis: 0)
        _fusedQKVWeight = fused
        return fused
    }

    /// Build the two terminal-prefill projection banks once after checkpoint
    /// load. Only the final sliding layer can dispatch them. Concatenating
    /// output rows is exact for bias-free `Linear`: each row retains the same
    /// K loop, BF16 inputs and BF16 weight bytes, independent of which other
    /// output rows share the matmul dispatch.
    func prepareLastPrefillProjectionWeights() -> [MLXArray] {
        guard lagunaLastPrefillProjectionBanksEnabled,
            _lastPrefillQGateWeight == nil,
            _lastPrefillKVWeight == nil,
            layerIdx == LagunaConstants.numHiddenLayers - 1,
            isSliding, gatingEnabled, gatePerHead,
            let gProj,
            type(of: wq) == Linear.self,
            type(of: wk) == Linear.self,
            type(of: wv) == Linear.self,
            type(of: gProj) == Linear.self,
            wq.bias == nil, wk.bias == nil, wv.bias == nil, gProj.bias == nil,
            wq.weight.dtype == .bfloat16,
            wk.weight.dtype == .bfloat16,
            wv.weight.dtype == .bfloat16,
            gProj.weight.dtype == .bfloat16,
            wq.weight.dims(nHeads * headDim, LagunaConstants.hiddenSize),
            wk.weight.dims(nKVHeads * headDim, LagunaConstants.hiddenSize),
            wv.weight.dims(nKVHeads * headDim, LagunaConstants.hiddenSize),
            gProj.weight.dims(nHeads, LagunaConstants.hiddenSize)
        else {
            return []
        }
        let qGate = concatenated([wq.weight, gProj.weight], axis: 0)
        let kv = concatenated([wk.weight, wv.weight], axis: 0)
        _lastPrefillQGateWeight = qGate
        _lastPrefillKVWeight = kv
        return [qGate, kv]
    }

    init(_ config: LagunaConfig, layerIdx: Int) {
        let dim = config.hiddenSize
        self.layerIdx = layerIdx
        self.nHeads = config.heads(forLayer: layerIdx)
        self.nKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)
        let layerGating = config.gatingMode(forLayer: layerIdx)
        self.gatingEnabled = layerGating.enabled
        self.gatePerHead = layerGating.isPerHead

        let layerType = config.layerType(forLayer: layerIdx)
        self.isSliding = layerType == .sliding
        self.attentionGateProjection =
            layerType == .sliding
            ? lagunaSlidingAttentionGateProjection
            : lagunaFullAttentionGateProjection

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: config.qkvBias)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: config.qkvBias)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: config.qkvBias)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: config.attentionBias)

        if gatingEnabled {
            let gateDim = gatePerHead ? nHeads : nHeads * headDim
            self._gProj.wrappedValue = Linear(dim, gateDim, bias: false)
        }

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))

        let ropeSpec = config.rope(for: layerType)
        let ropeDims = Int(Float(headDim) * Float(ropeSpec.partialRotaryFactor))
        self.rope = initializeRope(
            dims: ropeDims,
            base: Float(ropeSpec.theta),
            traditional: false,
            scalingConfig: lagunaRopeScalingConfig(ropeSpec),
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        inputNorm: RMSNorm,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        qkRoPEAngles: MLXArray? = nil,
        qkRoPEOffsets: MLXArray? = nil
    ) -> MLXArray {
        let (B, L) = (input.dim(0), input.dim(1))

        // One dispatch for the input RMSNorm and all three projections when
        // the decode preconditions hold; otherwise normalize separately and
        // fall through to the stock projections below.
        var fusedNormQKV:
            (
                queries: MLXArray, keys: MLXArray, values: MLXArray,
                gateValues: MLXArray, gateActivated: Bool
            )?
        if lagunaFusedQKVProjectionEnabled, _fusedQKVWeight == nil,
            B == 1, L == 1,
            headDim == LagunaConstants.headDim,
            nKVHeads == LagunaConstants.numKeyValueHeads,
            input.dtype == .bfloat16,
            input.dims(1, 1, LagunaConstants.hiddenSize),
            inputNorm.weight.dtype == .bfloat16,
            inputNorm.weight.dims(LagunaConstants.hiddenSize),
            wq.bias == nil, wk.bias == nil, wv.bias == nil,
            type(of: wq) == Linear.self, type(of: wk) == Linear.self,
            type(of: wv) == Linear.self,
            wq.weight.dtype == .bfloat16, wk.weight.dtype == .bfloat16,
            wv.weight.dtype == .bfloat16,
            gatingEnabled, gatePerHead,
            let gateProjection = gProj,
            gateProjection.bias == nil,
            type(of: gateProjection) == Linear.self,
            gateProjection.weight.dtype == .bfloat16,
            gateProjection.weight.dims(nHeads, LagunaConstants.hiddenSize)
        {
            if lagunaUseNativeAffineQKV(layer: layerIdx),
                let fusedAffine = _nativeAffineQKV
            {
                // One dispatch for the input RMSNorm AND the INT8 projection
                // (see `lagunaNormAffineQKVSource`). Requires the group-32
                // affine INT8 wire format AND the gate rows folded into the
                // bank, so no consumer downstream of here needs a
                // device-visible normalized row; the NVFP4 tail layers and
                // any guard decline keep the separate norm.
                var fusedQKV: MLXArray?
                if lagunaFusedNormAffineQKVEnabled,
                    fusedAffine.mode == .affine, fusedAffine.bits == 8,
                    fusedAffine.groupSize == 32,
                    _nativeAffineQKVGateRows == nHeads,
                    inputNorm.eps == Float(LagunaConstants.rmsNormEpsilon),
                    let affineBiases = fusedAffine.biases
                {
                    fusedQKV = lagunaNormAffineQKV(
                        residual: input,
                        normWeight: inputNorm.weight,
                        codes: fusedAffine.packedCodes,
                        scales: fusedAffine.scales,
                        biases: affineBiases,
                        indexedMetadata: fusedAffine.indexedMetadata,
                        rows: fusedAffine.originalShape[0])
                }

                // Fused tail norm+QKV+gate kernel removed (r=1 re-sweep: +2.7%); placeholder only.
                let fusedTailGateLogits: MLXArray? = nil
                // Only materialized when the fused kernel declined; the gate
                // branches below that read it are unreachable when it fired.
                let normalized = fusedQKV ?? inputNorm(input)
                // Try fused QKV+g_proj: one dispatch produces both QKV and
                // activated gate values, eliminating the separate gate-softplus
                // dispatch for NVFP4-tail layers.
                let fusedGProjQKV: (qkv: MLXArray, gateValues: MLXArray)?
                if fusedQKV == nil,
                    let affineGate = _nativeAffineGProj
                {
                    fusedGProjQKV = lagunaFusedGProjQKV(
                        normalized: normalized, bank: fusedAffine,
                        gproj: affineGate, heads: nHeads)
                } else {
                    fusedGProjQKV = nil
                }
                let decodeNVFP4QKVR1 =
                    fusedQKV == nil && fusedGProjQKV == nil
                    ? lagunaDecodeNVFP4QKVR1(
                        normalized: normalized, bank: fusedAffine, heads: nHeads)
                    : nil
                let qkv =
                    fusedQKV
                    ?? fusedGProjQKV?.qkv
                    ?? decodeNVFP4QKVR1
                    ?? quantizedMM(
                        normalized,
                        fusedAffine.packedCodes,
                        scales: fusedAffine.scales,
                        biases: fusedAffine.biases,
                        transpose: true,
                        groupSize: fusedAffine.groupSize,
                        bits: fusedAffine.bits,
                        mode: fusedAffine.mode
                    )
                let queryDim = nHeads * headDim
                let kvDim = nKVHeads * headDim
                let gateStart = queryDim + 2 * kvDim
                let gateLogits: MLXArray
                var gateProjectionActivated = false
                if let fusedTailGateLogits {
                    // Removed tail-fusion placeholder: always nil since the
                    // r=1-regime re-sweep; kept so the defer/eager plumbing
                    // below stays structurally unchanged.
                    gateLogits = fusedTailGateLogits
                } else if _nativeAffineQKVGateRows == nHeads {
                    // The gate rows rode the fused bank's single dispatch;
                    // slice them out of its tail. Same row-local math as a
                    // standalone group-32 INT8 gate qmv.
                    gateLogits = qkv[.ellipsis, gateStart ..< (gateStart + nHeads)]
                } else if let fusedGate = fusedGProjQKV {
                    // Fused QKV+g_proj: the gate values are already computed
                    // and softplus-activated inside the QKV dispatch.
                    gateLogits = fusedGate.gateValues
                    gateProjectionActivated = true
                } else if let affineGate = _nativeAffineGProj {
                    // NVFP4-tail layer: the gate keeps its own group-32 INT8
                    // bank (the envelope caps g_proj there) and replaces the
                    // BF16 GEMV one dispatch for one dispatch.
                    if lagunaFusedGatedAffineOProjEnabled,
                        lagunaGatedAffineOProjNVFP4Enabled,
                        lagunaUseNativeAffineOProj(layer: layerIdx),
                        let affineWO = _nativeAffineOProj,
                        affineWO.mode == .nvfp4, affineWO.bits == 4,
                        affineWO.groupSize == 16,
                        let activated = lagunaGateSoftplus(
                            input: normalized, bank: affineGate, heads: nHeads)
                    {
                        gateLogits = activated
                        gateProjectionActivated = true
                    } else {
                        gateLogits = quantizedMM(
                            normalized,
                            affineGate.packedCodes,
                            scales: affineGate.scales,
                            biases: affineGate.biases,
                            transpose: true,
                            groupSize: affineGate.groupSize,
                            bits: affineGate.bits,
                            mode: affineGate.mode
                        )
                    }
                } else {
                    gateLogits = gateProjection(normalized)
                }
                if lagunaAttentionProjectionAsyncEnabled,
                    layerIdx == 0, B == 1, L == 1
                {
                    lagunaTrace("attention projection async rung layer 0")
                    asyncEval(qkv, gateLogits)
                }
                // When the fused gate-product kernel will consume the gate at
                // the output projection, hand it the RAW BF16 logits and skip
                // the eager softplus chain entirely; the kernel reproduces
                // those exact rounding boundaries inside its single dispatch.
                // Otherwise keep the stock eager activation.
                let deferGateActivation =
                    lagunaFusedGateProductEnabled
                    && lagunaUseNativeAffineOProj(layer: layerIdx)
                    && _nativeAffineOProj != nil
                    && wo.bias == nil
                let gateValues =
                    gateProjectionActivated || deferGateActivation
                    ? gateLogits
                    : softplus(gateLogits.asType(.float32)).asType(.bfloat16)
                fusedNormQKV = (
                    qkv[.ellipsis, 0 ..< queryDim],
                    qkv[.ellipsis, queryDim ..< (queryDim + kvDim)],
                    qkv[.ellipsis, (queryDim + kvDim) ..< gateStart],
                    gateValues,
                    gateProjectionActivated || !deferGateActivation
                )
            } else {
                fusedNormQKV = lagunaFusedNormQKVProjection(
                    residual: input,
                    normWeight: inputNorm.weight,
                    queryWeight: wq.weight,
                    keyWeight: wk.weight,
                    valueWeight: wv.weight,
                    gateWeight: gateProjection.weight,
                    heads: nHeads
                )
            }
        }
        // The fused result already contains every consumer of the normalized
        // row. Materialize that row only for the stock projections or the
        // retained row-concatenated QKV bank. Checking actual bank presence
        // above (rather than its environment flag) preserves the custom
        // fallback if fused-weight preparation declined.
        let normalizedInput: MLXArray? =
            fusedNormQKV == nil ? inputNorm(input) : nil

        var queries: MLXArray
        var keys: MLXArray
        var values: MLXArray
        // The retained BF16 [Wq; Wk; Wv] bank is PREFILL-ONLY: at decode it
        // would override the INT8 fused norm+QKV path (measured +1.4 ms/step
        // when force-enabled), while at L > 1 it collapses three steel GEMMs
        // into one.
        if let fusedQKVWeight = _fusedQKVWeight, L > 1 {
            guard let normalizedInput else {
                preconditionFailure("retained fused QKV requires normalized input")
            }
            // One dispatch over the row-concatenated [Wq; Wk; Wv] weight,
            // identical math to the three bias-free `Linear` calls
            // (`matmul(x, w.T)`). Each output row's K-loop is independent of
            // which rows share the dispatch, so every Q/K/V element is
            // bit-exact; the slices are views and the reshapes below may
            // copy, which does not change values.
            let qkv = matmul(normalizedInput, fusedQKVWeight.T)
            let queryDim = nHeads * headDim
            let kvDim = nKVHeads * headDim
            queries = qkv[.ellipsis, 0 ..< queryDim]
            keys = qkv[.ellipsis, queryDim ..< (queryDim + kvDim)]
            values = qkv[.ellipsis, (queryDim + kvDim) ..< (queryDim + 2 * kvDim)]
        } else if let fused = fusedNormQKV {
            queries = fused.queries
            keys = fused.keys
            values = fused.values
        } else {
            guard let normalizedInput else {
                preconditionFailure("stock QKV projections require normalized input")
            }
            queries = wq(normalizedInput)
            keys = wk(normalizedInput)
            values = wv(normalizedInput)
        }

        let fusedQKNormShapesMatch =
            B == 1 && L == 1 &&
            nKVHeads == LagunaConstants.numKeyValueHeads &&
            headDim == LagunaConstants.headDim &&
            queries.dtype == .bfloat16 && keys.dtype == .bfloat16 &&
            qNorm.weight.dtype == .bfloat16 && kNorm.weight.dtype == .bfloat16 &&
            queries.dims(1, 1, nHeads * headDim) &&
            keys.dims(1, 1, nKVHeads * headDim)

        let useFusedFullQKNormYaRN =
            lagunaFusedFullQKNormYaRNEnabled && !isSliding &&
            fusedQKNormShapesMatch &&
            nHeads == LagunaConstants.fullAttentionHeads &&
            qkRoPEAngles?.dtype == .float32 &&
            qkRoPEAngles?.dims(1, 1, 1, headDim / 2) == true

        let useFusedSlidingQKNormRoPE =
            lagunaFusedSlidingQKNormRoPEEnabled && isSliding &&
            fusedQKNormShapesMatch &&
            nHeads == LagunaConstants.slidingAttentionHeads &&
            qkRoPEAngles?.dtype == .float32 &&
            qkRoPEAngles?.dims(1, 1, 1, headDim) == true

        // Multi-token twins of the decode fusions. The angle input is the
        // full load-time atlas (one cos/sin row per absolute position) and
        // `qkRoPEOffsets` carries the cache offset the stock
        // `applyRotaryPosition` would have used, both prepared once per
        // forward by the inner model; the guards fall through to the stock
        // four/six-dispatch chain for any other shape, dtype, or cache
        // state.
        let prefillQKNormShapesMatch =
            B == 1 && L > 1 &&
            nKVHeads == LagunaConstants.numKeyValueHeads &&
            headDim == LagunaConstants.headDim &&
            queries.dtype == .bfloat16 && keys.dtype == .bfloat16 &&
            qNorm.weight.dtype == .bfloat16 && kNorm.weight.dtype == .bfloat16 &&
            queries.dims(1, L, nHeads * headDim) &&
            keys.dims(1, L, nKVHeads * headDim) &&
            qkRoPEAngles?.dtype == .float32 &&
            qkRoPEOffsets?.dtype == .int32 && qkRoPEOffsets?.size == 1

        let usePrefillFusedSlidingQKNormRoPE =
            lagunaPrefillQKNormRoPEEnabled && isSliding &&
            prefillQKNormShapesMatch &&
            nHeads == LagunaConstants.slidingAttentionHeads &&
            qkRoPEAngles?.dims(1, 1, lagunaRoPEAngleAtlasLength, headDim) == true

        let usePrefillFusedFullQKNormYaRN =
            lagunaPrefillQKNormRoPEEnabled && !isSliding &&
            prefillQKNormShapesMatch &&
            nHeads == LagunaConstants.fullAttentionHeads &&
            qkRoPEAngles?.dims(1, 1, lagunaRoPEAngleAtlasLength, headDim / 2) == true

        var qkNormRoPEFused = false
        var fusedAttended: MLXArray?
        if lagunaFusedSlidingAttentionEnabled,
            useFusedSlidingQKNormRoPE,
            let fusedAngles = qkRoPEAngles,
            values.dtype == .bfloat16,
            values.dims(1, 1, nKVHeads * headDim),
            let rotating = cache as? RotatingKVCache,
            rotating.maxSize == LagunaConstants.slidingWindow,
            let ring = rotating.fusedRingPrepare()
        {
            // One dispatch replaces the QK-norm+RoPE kernel, both cache
            // slice-assign dispatches, and sdpa_vector; see the kernel doc.
            // The clock advance below mirrors updateInPlace(tokenCount: 1).
            fusedAttended = lagunaSlidingFusedAttention(
                rawQueries: queries,
                rawKeys: keys,
                rawValues: values,
                queryWeight: qNorm.weight,
                keyWeight: kNorm.weight,
                angles: fusedAngles,
                cacheKeys: ring.keys,
                cacheValues: ring.values,
                writeIdx: ring.writeIdx,
                scale: _fusedAttnScale
            )
            rotating.fusedRingAdvance()
            qkNormRoPEFused = true
        } else if lagunaFusedFullAttentionEnabled,
            useFusedFullQKNormYaRN,
            let fusedAngles = qkRoPEAngles,
            values.dtype == .bfloat16,
            values.dims(1, 1, nKVHeads * headDim),
            let simple = cache as? KVCacheSimple,
            let append = simple.fusedAppendPrepare()
        {
            // Full-attention twin of the fused branch above; engages from
            // the second decode step (the first step's growth concat stays
            // stock). The clock advance mirrors the stock single-token
            // update.
            fusedAttended = lagunaFullFusedAttention(
                rawQueries: queries,
                rawKeys: keys,
                rawValues: values,
                queryWeight: qNorm.weight,
                keyWeight: kNorm.weight,
                angles: fusedAngles,
                cacheKeys: append.keys,
                cacheValues: append.values,
                writeIdx: append.writeIdx,
                scale: _fusedAttnScale
            )
            simple.fusedAppendAdvance()
            qkNormRoPEFused = true
        } else if useFusedFullQKNormYaRN, let qkRoPEAngles {
            (queries, keys) = lagunaFullQKNormYaRN(
                rawQueries: queries,
                rawKeys: keys,
                queryWeight: qNorm.weight,
                keyWeight: kNorm.weight,
                angles: qkRoPEAngles
            )
            qkNormRoPEFused = true
        } else if useFusedSlidingQKNormRoPE, let qkRoPEAngles {
            (queries, keys) = lagunaSlidingQKNormRoPE(
                rawQueries: queries,
                rawKeys: keys,
                queryWeight: qNorm.weight,
                keyWeight: kNorm.weight,
                angles: qkRoPEAngles
            )
            qkNormRoPEFused = true
        } else if usePrefillFusedSlidingQKNormRoPE,
            let angles = qkRoPEAngles, let offsets = qkRoPEOffsets
        {
            (queries, keys) = lagunaPrefillSlidingQKNormRoPE(
                rawQueries: queries,
                rawKeys: keys,
                queryWeight: qNorm.weight,
                keyWeight: kNorm.weight,
                angles: angles,
                offsets: offsets,
                length: L
            )
            qkNormRoPEFused = true
        } else if usePrefillFusedFullQKNormYaRN,
            let angles = qkRoPEAngles, let offsets = qkRoPEOffsets
        {
            (queries, keys) = lagunaPrefillFullQKNormYaRN(
                rawQueries: queries,
                rawKeys: keys,
                queryWeight: qNorm.weight,
                keyWeight: kNorm.weight,
                angles: angles,
                offsets: offsets,
                length: L
            )
            qkNormRoPEFused = true
        } else {
            queries =
                qNorm(queries.reshaped(B, L, nHeads, headDim))
                .transposed(0, 2, 1, 3)
            keys =
                kNorm(keys.reshaped(B, L, nKVHeads, headDim))
                .transposed(0, 2, 1, 3)
        }
        // With a singleton sequence axis, `[B, 1, H, D]` and
        // `[B, H, 1, D]` have the same contiguous byte order. Reshape
        // directly so decode does not carry a no-op transpose view through
        // the lazy graph. Multi-token calls still require the real axis swap.
        // A fused attention branch consumed the raw values already, so the
        // head-major layout only exists for the stock SDPA fallback.
        if fusedAttended == nil {
            values =
                L == 1
                ? values.reshaped(B, nKVHeads, L, headDim)
                : values.reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)
        }

        if !qkNormRoPEFused {
            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)
        }

        let attended =
            fusedAttended
            ?? attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask
            )
        // SDPA returns `[B, H, L, D]`. When `L == 1`, flattening its
        // contiguous head-major payload directly produces the exact
        // `[B, 1, H*D]` byte order; the transpose only changes singleton-axis
        // metadata. Preserve the real transpose for prefill.
        var output =
            L == 1
            ? attended.reshaped(B, L, -1)
            : attended.transposed(0, 2, 1, 3).reshaped(B, L, -1)

        if gatingEnabled, let gProj {
            // Per-head softplus gate computed in float32, then broadcast
            // across the head dimension (or applied elementwise for a
            // per-element gate).
            let projectedGate: MLXArray
            let gateIsActivated: Bool
            if let fusedNormQKV {
                projectedGate = fusedNormQKV.gateValues
                gateIsActivated = fusedNormQKV.gateActivated
            } else {
                guard let normalizedInput else {
                    preconditionFailure("attention gate requires normalized input")
                }
                projectedGate = gProj(normalizedInput)
                gateIsActivated = false
            }
            // Native group-32 affine INT8 output projection for the serial
            // decode token. The stock fused kernel folds the gate into the
            // GEMV's own vector loads; this path cannot, because MLX's
            // `quantizedMM` owns the contraction. It therefore reproduces that
            // kernel's element-wise ordering explicitly: the per-head gate
            // multiplies the attention output *first*, through the same single
            // BF16 rounding boundary the kernel spells as
            // `float(bfloat(float(values[i]) * gate))` (an MLX BF16 binary
            // product rounds once, identically), and only then does the
            // contraction run. Gate-then-project is what every stock decode
            // form computes — the fused kernel, the compiled
            // `attentionGateProjection`, and the plain
            // `(output * gate); wo(output)` tail all apply the gate per input
            // element before the K loop — so the only perturbation this branch
            // introduces is the weight quantization itself.
            //
            // The broadcast multiply stays an MLX binary op deliberately: its
            // input is row-contiguous and refcount-1, so MLX donates the
            // attention output buffer and runs the product in place, whereas a
            // custom kernel would have to allocate and first-touch a fresh
            // 8192-wide output.
            if lagunaUseNativeAffineOProj(layer: layerIdx),
                let affineWO = _nativeAffineOProj,
                gatePerHead, B == 1, L == 1, wo.bias == nil,
                headDim == LagunaConstants.headDim,
                output.dtype == .bfloat16, projectedGate.dtype == .bfloat16,
                output.dims(1, 1, nHeads * headDim),
                projectedGate.dims(1, 1, nHeads)
            {
                // Raw logits + gated affine GEMV: ONE dispatch for the softplus
                // chain, the broadcast product AND the INT8 contraction (see
                // `lagunaGatedAffineOProjSource`). Only the group-32 affine
                // INT8 wire format is served; the NVFP4 tail layers and any
                // guard decline fall through to the two-dispatch chain below.
                if lagunaFusedGatedAffineOProjEnabled, !gateIsActivated,
                    affineWO.mode == .affine, affineWO.bits == 8,
                    affineWO.groupSize == 32,
                    affineWO.indexedMetadata != nil,
                    let affineBiases = affineWO.biases,
                    let fusedProjection = lagunaGatedAffineOProj(
                        attentionOutput: output,
                        gateLogits: projectedGate,
                        codes: affineWO.packedCodes,
                        scales: affineWO.scales,
                        biases: affineBiases,
                        indexedMetadata: affineWO.indexedMetadata,
                        heads: nHeads)
                {
                    return fusedProjection
                }
                if lagunaFusedGatedAffineOProjEnabled, !gateIsActivated,
                    affineWO.mode == .affine, affineWO.bits == 8,
                    affineWO.groupSize == 32,
                    let affineBiases = affineWO.biases,
                    let fusedProjection = lagunaGatedAffineOProj(
                        attentionOutput: output,
                        gateLogits: projectedGate,
                        codes: affineWO.packedCodes,
                        scales: affineWO.scales,
                        biases: affineBiases,
                        heads: nHeads)
                {
                    return fusedProjection
                }
                if lagunaFusedGatedAffineOProjEnabled,
                    lagunaGatedAffineOProjNVFP4Enabled,
                    gateIsActivated,
                    affineWO.mode == .nvfp4, affineWO.bits == 4,
                    affineWO.groupSize == 16,
                    let fusedProjection = lagunaGatedAffineOProjNVFP4(
                        attentionOutput: output,
                        gateLogits: projectedGate,
                        codes: affineWO.packedCodes,
                        scales: affineWO.scales,
                        heads: nHeads,
                        gateIsActivated: true,
                        scalesEscape: affineWO.scalesEscape)
                {
                    return fusedProjection
                }
                if lagunaFusedGatedAffineOProjEnabled,
                    lagunaGatedAffineOProjNVFP4Enabled,
                    !gateIsActivated,
                    affineWO.mode == .nvfp4, affineWO.bits == 4,
                    affineWO.groupSize == 16,
                    let fusedProjection = lagunaGatedAffineOProjNVFP4(
                        attentionOutput: output,
                        gateLogits: projectedGate,
                        codes: affineWO.packedCodes,
                        scales: affineWO.scales,
                        heads: nHeads,
                        scalesEscape: affineWO.scalesEscape)
                {
                    return fusedProjection
                }
                // Raw logits + fused kernel: one dispatch reproduces the
                // softplus chain AND the broadcast product bit-exactly (see
                // `lagunaGateProductSoftplusSource`). Falls back to the stock
                // compiled-softplus + donated in-place multiply whenever the
                // gate is already activated or the kernel declines.
                let gated: MLXArray
                if !gateIsActivated,
                    let fusedGated = lagunaGateProductSoftplus(
                        attentionOutput: output, gateLogits: projectedGate,
                        heads: nHeads)
                {
                    gated = fusedGated
                } else {
                    let gate =
                        gateIsActivated
                        ? projectedGate
                        : lagunaCompiledSoftplusGate(projectedGate)
                    gated =
                        (output.reshaped(B, L, nHeads, headDim)
                            * gate[.ellipsis, .newAxis])
                        .reshaped(B, L, -1)
                }
                lagunaTrace("native affine gated output projection h\(nHeads)")
                return quantizedMM(
                    gated,
                    affineWO.packedCodes,
                    scales: affineWO.scales,
                    biases: affineWO.biases,
                    transpose: true,
                    groupSize: affineWO.groupSize,
                    bits: affineWO.bits,
                    mode: affineWO.mode
                )
            }
            if lagunaFusedGatedOutputProjectionEnabled,
                gateIsActivated, gatePerHead, L == 1, B == 1, wo.bias == nil,
                headDim == LagunaConstants.headDim,
                output.dtype == .bfloat16, projectedGate.dtype == .bfloat16,
                wo.weight.dtype == .bfloat16,
                output.dims(1, 1, nHeads * headDim),
                projectedGate.dims(1, 1, nHeads),
                wo.weight.dims(LagunaConstants.hiddenSize, nHeads * headDim)
            {
                let projection = lagunaGatedOutputProjection(
                    attentionOutput: output,
                    gateValues: projectedGate,
                    weight: wo.weight,
                    heads: nHeads
                )
                if let projection {
                    return projection
                }
            }
            if !gateIsActivated,
                gatePerHead && projectedGate.dtype == output.dtype,
                L == 1, wo.bias == nil, MLXHardwareInfo.isCompiledDecodeSupported
            {
                return attentionGateProjection(output, projectedGate, wo.weight)
            }
            let gate =
                gateIsActivated
                ? projectedGate
                : gatePerHead && projectedGate.dtype == output.dtype
                ? lagunaCompiledSoftplusGate(projectedGate)
                : softplus(projectedGate.asType(.float32)).asType(output.dtype)
            if gatePerHead {
                output =
                    (output.reshaped(B, L, nHeads, headDim) * gate[.ellipsis, .newAxis])
                    .reshaped(B, L, -1)
            } else {
                output = output * gate
            }
        }

        return wo(output)
    }

    /// Prefill-only final-layer attention when the caller consumes just the
    /// last hidden row. K/V and the cache update still cover every supplied
    /// token. Q projection, Q normalization, Q RoPE, SDPA, and the output
    /// gate/projection run only for the last query; its RoPE offset is advanced
    /// by the discarded query-row count so it remains at the supplied
    /// sequence's final absolute position.
    func callLastPrefillRow(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        precondition(L > 1)

        let lastInput = lagunaLastTokenHidden(x)
        var queries: MLXArray
        var keys: MLXArray
        var values: MLXArray
        let bankedGate: MLXArray?
        if lagunaLastPrefillProjectionBanksEnabled,
            let qGateWeight = _lastPrefillQGateWeight,
            let kvWeight = _lastPrefillKVWeight,
            B == 1,
            isSliding, gatingEnabled, gatePerHead,
            lastInput.dtype == .bfloat16,
            x.dtype == .bfloat16,
            lastInput.dims(1, 1, LagunaConstants.hiddenSize),
            x.dims(1, L, LagunaConstants.hiddenSize),
            qGateWeight.dtype == .bfloat16,
            kvWeight.dtype == .bfloat16,
            qGateWeight.dims(nHeads * headDim + nHeads, LagunaConstants.hiddenSize),
            kvWeight.dims(2 * nKVHeads * headDim, LagunaConstants.hiddenSize)
        {
            let qGate = matmul(lastInput, qGateWeight.T)
            let queryDim = nHeads * headDim
            queries = qGate[.ellipsis, 0 ..< queryDim]
            bankedGate = qGate[.ellipsis, queryDim ..< (queryDim + nHeads)]

            let kv = matmul(x, kvWeight.T)
            let kvDim = nKVHeads * headDim
            keys = kv[.ellipsis, 0 ..< kvDim]
            values = kv[.ellipsis, kvDim ..< (2 * kvDim)]
            lagunaTrace("last prefill Q+gate / K+V projection banks")
        } else {
            queries = wq(lastInput)
            keys = wk(x)
            values = wv(x)
            bankedGate = nil
        }

        queries = qNorm(queries.reshaped(B, 1, nHeads, headDim)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, nKVHeads, headDim)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)

        if let offsetArray = graphOffsetArray(for: cache) {
            queries = rope(queries, offset: offsetArray + Int32(L - 1))
        } else {
            queries = rope(queries, offset: (cache?.offset ?? 0) + L - 1)
        }
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let attended = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: .causal
        )
        // The last-row query length is exactly one, so the SDPA result's
        // `[B, H, 1, D]` storage is already the desired flattened head order.
        var output = attended.reshaped(B, 1, -1)

        if gatingEnabled, let gProj {
            let projectedGate = bankedGate ?? gProj(lastInput)
            let gate =
                gatePerHead && projectedGate.dtype == output.dtype
                ? lagunaCompiledSoftplusGate(projectedGate)
                : softplus(projectedGate.asType(.float32)).asType(output.dtype)
            if gatePerHead {
                output =
                    (output.reshaped(B, 1, nHeads, headDim) * gate[.ellipsis, .newAxis])
                    .reshaped(B, 1, -1)
            } else {
                output = output * gate
            }
        }

        return wo(output)
    }
}

// MARK: - Dense MLP (also used as the shared expert)

/// `DARKBLOOM_NVFP4_SCALE_FOLD` (default on; set "0" to restore the pre-fold
/// arithmetic): hoists the `2^14` out of `laguna_nvfp4_qdot_16`'s sixteen
/// per-call multiplies and folds it into the one multiply
/// Hoists `2^14` into `laguna_nvfp4_scale`: -16 multiplies/call, -1104M/token.
/// Bit-exact: power-of-two scaling touches only exponents. Dtype move range-checked
/// (half peak 1.875, scale 480, 136x headroom). Compiler can't fold (fastMath off).
let lagunaNvfp4ScaleFoldEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_NVFP4_SCALE_FOLD"] != "0"

/// `DARKBLOOM_NVFP4_NIBBLE_SPLIT` (default `1` = split; set `0` for the stock
/// shuffle, `2` for the 2-constant control arm): a strictly shorter
/// instruction sequence that produces the SAME eight `half` bit patterns per
/// code word. Values, dtypes, FP32 accumulation and its order, and the single
/// final BF16 round are untouched -- only the integer sequence that builds the
/// half bit patterns changes.
/// 0=stock (19 ops/word, 8 masks), 1=split (13 ops/word, 3 masks, -12/group),
/// 2=op-count control (19 ops, 2 masks). Bit-exact by construction: OR of masked
/// shifts, verified against 33 basis words + 300K random words. Composes with SCALE_FOLD.
let lagunaNvfp4NibbleSplit: Int = {
    guard
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_NVFP4_NIBBLE_SPLIT"],
        let value = Int(raw), (0 ... 2).contains(value)
    else { return 1 }
    return value
}()

/// Folds the e4m3 group-scale's sign bit into the half bit pattern instead of
/// negating after conversion. For `bits = 128 + m` the carry `bits + (bits &
/// 128)` yields `256 + m`, and `(256 + m) << 7 == 0x8000 | (m << 7)` because
/// `m <= 127` keeps `m << 7 <= 16256 < 0x8000`; IEEE half is sign-magnitude,
/// so that pattern IS the negation, including `-0.0h`. For `bits < 128` the
/// add is the identity. Drops `laguna_nvfp4_scale` from seven AIR ops to five
/// in the innermost K loop of every routed/shared decode QMV.
/// `DARKBLOOM_NVFP4_SCALE_CARRY=0` restores the negate-after-convert form.
let lagunaNvfp4ScaleCarry: Bool =
    ProcessInfo.processInfo.environment["DARKBLOOM_NVFP4_SCALE_CARRY"] != "0"

/// `DARKBLOOM_NVFP4_QDOT_SEED_ELIDE` (default on; set "0" to restore the
/// `float accum = 0.0f;` seed): removes the one dead FP add per 16-value NVFP4
/// group. `laguna_nvfp4_qdot_codes_16` seeds its accumulator with the literal
/// `+0.0f` and then performs four `accum +=`, so the very first of those four
/// is `fl(+0.0f + t)`. **`fadd float 0.0, %t` is NOT foldable to `%t`** —
/// `0.0f + (-0.0f)` is `+0.0f`, not `-0.0f`, so eliminating it needs the
/// no-signed-zeros flag, and `device.cpp:631` sets
/// `setFastMathEnabled(false)`. The add really is emitted, once per group, and
/// with ~69 M groups per decoded token across the nine routed/shared
/// SwiGLU-QMV and down kernels that is ~69 M dead FP adds per token, ~1.4% of
/// this loop's ALU.
/// Bit-exact by signed-zero case analysis (not tolerance): eliding `+0.0f` seed
/// can only flip `+0.0` to `-0.0` when all 16 products are `-0.0`, and every call
/// site absorbs that sign via `+0.0`-seeded accumulators. Tested by `LagunaNVFP4QdotSeedTests`.
let lagunaNvfp4QdotSeedElisionEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_NVFP4_QDOT_SEED_ELIDE"] != "0"

/// `DARKBLOOM_NVFP4_SCALE_DEFER` (default ON; set "0" to restore): moves
/// the `2^22` that `DARKBLOOM_NVFP4_SCALE_FOLD` parked in
/// `laguna_nvfp4_scale` out of the per-group scale and onto the per-row
/// accumulator, one multiply per output row instead of one per 16-value
/// group. `laguna_nvfp4_scale` drops from five ops to four (and, add, shift,
/// convert) and the group cost falls by one FP multiply -- the same ~69 M
/// removals per decoded token as the seed elision above.
/// Exact under a range condition (not case analysis): diverges only if some group has
/// `0 < |scale*accum| < 2^-104`, which needs all 16 activations < ~2^-99. This model's
/// BF16 activations are O(1) RMSNorm quantities, so the margin is ~60 binades.
let lagunaNvfp4ScaleDeferEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_NVFP4_SCALE_DEFER"] != "0"
    && lagunaNvfp4ScaleFoldEnabled

/// The `2^22` that `laguna_nvfp4_scale` stops applying under
/// `DARKBLOOM_NVFP4_SCALE_DEFER`, re-applied once per output row at the
/// accumulator-to-BF16 boundary. It appears at exactly one site per
/// `laguna_nvfp4_scale` call site in every kernel that uses
/// `lagunaSharedSwiGLUQMVHeader`; `nvfp4EveryScaleCallSiteHasARowRescaleSite`
/// pins those two counts equal per kernel so a missed epilogue cannot ship.
let lagunaNvfp4RowScaleSuffix = lagunaNvfp4ScaleDeferEnabled ? " * 4194304.0f" : ""

let lagunaSharedSwiGLUQMVHeader: String = {
    // The two halves of one power-of-two regrouping. They MUST move together:
    // the scale absorbs `2^14` exactly when the weights stop applying it.
    // `4194304.0f == 256 · 16384 == 2^22`.
    // Under `DARKBLOOM_NVFP4_SCALE_DEFER` the `2^22` leaves the per-group
    // scale entirely and is re-applied once per output row by
    // `lagunaNvfp4RowScaleSuffix`; the tail is then the same expression the
    // unfolded arm uses, for the opposite reason.
    let scaleTail =
        (lagunaNvfp4ScaleFoldEnabled && !lagunaNvfp4ScaleDeferEnabled)
        ? "        return float(signed_value) * 4194304.0f;"
        : "        return float(signed_value);"
    let scale256 = lagunaNvfp4ScaleFoldEnabled ? "" : "        converted *= 256.0;\n"
    let weightScale = lagunaNvfp4ScaleFoldEnabled ? "" : " * 16384.0f"
    let extract: String
    switch lagunaNvfp4NibbleSplit {
    case 1:
        extract = """
                    const uint xe = c & 0x0F0F0F0Fu;
                    const uint ge = xe | (xe << 3);
                    const uint yo = c & 0xF0F0F0F0u;
                    const uint go = yo | (yo >> 3);
                    const uint p0 = (ge << 9) & 0x8E008E00u;
                    const uint p1 = (go << 8) & 0x8E008E00u;
                    const uint p2 = (ge << 1) & 0x8E008E00u;
                    const uint p3 = go & 0x8E008E00u;
            """
    case 2:
        extract = """
                    const uint p0 =
                        ((c << 9) & 0x0E000E00u) | ((c << 12) & 0x80008000u);
                    const uint p1 =
                        ((c << 5) & 0x0E000E00u) | ((c << 8) & 0x80008000u);
                    const uint p2 =
                        ((c << 1) & 0x0E000E00u) | ((c << 4) & 0x80008000u);
                    const uint p3 =
                        ((c >> 3) & 0x0E000E00u) | (c & 0x80008000u);
            """
    default:
        extract = """
                    const uint p0 =
                        ((c & 0x00070007u) << 9) | ((c & 0x00080008u) << 12);
                    const uint p1 =
                        ((c & 0x00700070u) << 5) | ((c & 0x00800080u) << 8);
                    const uint p2 =
                        ((c & 0x07000700u) << 1) | ((c & 0x08000800u) << 4);
                    const uint p3 =
                        ((c & 0x70007000u) >> 3) | (c & 0x80008000u);
            """
    }
    // The carry form only composes with the folded tail, which is where the
    // sign lands before any further scaling; keep the negate form otherwise.
    let scaleCarryActive = lagunaNvfp4ScaleCarry && lagunaNvfp4ScaleFoldEnabled
    let scaleRawExpression =
        scaleCarryActive
        ? (lagunaE4M3SignDomainCertified
            ? "ushort(uint(bits) << 7)"
            : "ushort((uint(bits) + (bits & 128u)) << 7)")
        : "ushort(bits & 127) << 7"
    let scaleSignExpression =
        scaleCarryActive
        ? "converted"
        : "(bits & 128) ? -converted : converted"
    // E4M3 bytes 0...15 are a linear positive run.  In the default folded
    // deferred-scale arm the half bit pattern is already the exact value
    // needed by the qdot; skip the carry/sign setup for this overwhelmingly
    // common case.  Keep every ablation's source unchanged.
    let lowScaleFastPath = lagunaNvfp4ScaleDeferEnabled
        ? """
        if (bits < 16u) {
            ushort fast_raw = ushort(bits) << 7;
            return float(as_type<half>(fast_raw));
        }
        """
        : ""
    // One packed 32-bit code word: eight NVFP4 values, four `half2` patterns,
    // two four-term FP groups. The first group of the FIRST word seeds the
    // accumulator when the seed elision is enabled; every other group adds.
    // Word `w` owns `input[8w .. 8w+7]`, exactly the indices the `8 * j` form
    // produced, so the multiply/add expressions and their association are
    // untouched.
    func packedWordBody(_ word: Int) -> String {
        let codeWord = word == 0 ? "codes.x" : "codes.y"
        let base = 8 * word
        let seedStart =
            (word == 0 && lagunaNvfp4QdotSeedElisionEnabled)
            ? "0.0f" : "accum"
        return """
                {
                    const uint c = \(codeWord);
            \(extract)
                    const float2 v04 = float2(as_type<half2>(p0))\(weightScale);
                    const float2 v15 = float2(as_type<half2>(p1))\(weightScale);
                    const float2 v26 = float2(as_type<half2>(p2))\(weightScale);
                    const float2 v37 = float2(as_type<half2>(p3))\(weightScale);
                    accum =
                        fma(input[\(base)], v04.x,
                        fma(input[\(base + 1)], v15.x,
                        fma(input[\(base + 2)], v26.x,
                        fma(input[\(base + 3)], v37.x, \(seedStart)))));
                    accum =
                        fma(input[\(base + 4)], v04.y,
                        fma(input[\(base + 5)], v15.y,
                        fma(input[\(base + 6)], v26.y,
                        fma(input[\(base + 7)], v37.y, accum))));
                }
            """
    }
    let accumDeclaration =
        lagunaNvfp4QdotSeedElisionEnabled
        ? "float accum;" : "float accum = 0.0f;"
    return """
    static inline float laguna_nvfp4_scale(uint8_t bits) {
    \(lowScaleFastPath)
        ushort raw = \(scaleRawExpression);
        half converted = as_type<half>(raw);
    \(scale256)    half signed_value = \(scaleSignExpression);
    \(scaleTail)
    }

    static inline float laguna_nvfp4_qdot_codes_16(
        uint2 codes,
        const thread float* input,
        float scale
    ) {
        \(accumDeclaration)
    \(packedWordBody(0))
    \(packedWordBody(1))
        return scale * accum;
    }

    static inline float laguna_nvfp4_qdot_16(
        const device uint8_t* weight,
        const thread float* input,
        float scale
    ) {
        const device uint2* packed = (const device uint2*)weight;
        return laguna_nvfp4_qdot_codes_16(packed[0], input, scale);
    }
    """
}()

/// Scale-plane-halved shared SwiGLU QMV (R1 scheduling: one row per SIMD).
private let lagunaSharedSwiGLUQMVRows1Kernel = MLXFast.metalKernel(
    name: "laguna_shared_nvfp4_swiglu_qmv_rows1_bf16_v1_halved",
    inputNames: ["input", "fused_weight", "packed_scales", "gate_up_escape"],
    outputNames: ["activated"],
    source: """
        constexpr uint input_width = 2048;
        constexpr uint output_width = 512;
        constexpr uint packed_row_bytes = 1024;
        constexpr uint scale_row_bytes = 64;
        constexpr uint block_width = 512;
        constexpr uint values_per_lane = 16;

        uint tile = threadgroup_position_in_grid.x;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint row = tile * 2 + simd_group;

        const device uint8_t* gate_row_weight =
            (const device uint8_t*)fused_weight +
            row * packed_row_bytes + lane * 8;
        const device uint8_t* up_row_weight =
            (const device uint8_t*)fused_weight +
            (row + output_width) * packed_row_bytes + lane * 8;
        const device uint8_t* gate_row_scale =
            packed_scales + row * scale_row_bytes + lane / 2;
        const device uint8_t* up_row_scale =
            packed_scales + (row + output_width) * scale_row_bytes + lane / 2;
        uint8_t gate_escape = gate_up_escape[0];
        uint8_t up_escape = gate_up_escape[1];

        thread float gate_result = 0.0f;
        thread float up_result = 0.0f;
        thread float input_values[values_per_lane];

        for (uint block = 0; block < input_width; block += block_width) {
            const device vec<bfloat, 4>* input_vectors =
                (const device vec<bfloat, 4>*) (
                    input + block + lane * values_per_lane);
            for (uint i = 0; i < values_per_lane / 4; ++i) {
                *(thread float4*)(input_values + 4 * i) =
                    float4(input_vectors[i]);
            }

            uint8_t gate_sb = gate_row_scale[block / 16];
            uint8_t up_sb = up_row_scale[block / 16];
            if (row == 0 && lane == 1) {
                gate_sb = gate_escape;
                up_sb = up_escape;
            }
            gate_result += laguna_nvfp4_qdot_16(
                gate_row_weight + block / 2,
                input_values,
                laguna_nvfp4_scale(gate_sb));
            up_result += laguna_nvfp4_qdot_16(
                up_row_weight + block / 2,
                input_values,
                laguna_nvfp4_scale(up_sb));
        }

        {
            const vec<float, 2> packed = simd_sum(
                vec<float, 2>(gate_result, up_result));
            gate_result = packed.x;
            up_result = packed.y;
        }
        if (lane == 0) {
            bfloat gate = bfloat(gate_result\(lagunaNvfp4RowScaleSuffix));
            bfloat up = bfloat(up_result\(lagunaNvfp4RowScaleSuffix));
            bfloat exp_abs = metal::exp(metal::abs(gate));
            bfloat denominator = bfloat(1) + exp_abs;
            bfloat y = bfloat(1) / denominator;
            bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
            bfloat silu = bfloat(gate * sigmoid);
            activated[row] = bfloat(silu * up);
        }
        """,
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)

func lagunaSharedSwiGLUQMV(
    _ input: MLXArray,
    fusedWeight: MLXArray,
    packedScales: MLXArray,
    gateUpEscape: MLXArray
) -> MLXArray {
    precondition(input.dtype == .bfloat16)
    precondition(input.dims(1, 1, LagunaConstants.hiddenSize))
    precondition(fusedWeight.dtype == .uint32)
    precondition(
        fusedWeight.dims(2 * LagunaConstants.sharedExpertIntermediateSize,
            LagunaConstants.hiddenSize / 8))
    precondition(packedScales.dtype == .uint8)
    precondition(
        packedScales.dims(2 * LagunaConstants.sharedExpertIntermediateSize,
            LagunaConstants.hiddenSize / 32))
    precondition(gateUpEscape.dtype == .uint8)

    return lagunaSharedSwiGLUQMVRows1Kernel(
        [input, fusedWeight, packedScales, gateUpEscape],
        grid: (256 * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.sharedExpertIntermediateSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

private let lagunaSharedDownResidualKernel = MLXFast.metalKernel(
    name: "laguna_shared_nvfp4_down_residual_bf16_v1_halved",
    inputNames: [
        "activated", "down_weight", "down_scales", "down_scales_escape",
        "routed", "residual",
    ],
    outputNames: ["output"],
    source: """
        constexpr uint input_width = 512;
        constexpr uint output_width = 2048;
        constexpr uint outputs_per_simd = 4;
        constexpr uint values_per_lane = 16;
        constexpr uint packed_row_bytes = 256;
        constexpr uint scale_row_bytes = 16;

        uint group = threadgroup_position_in_grid.x;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint first_row =
            group * 2 * outputs_per_simd +
            simd_group * outputs_per_simd;

        thread float input_values[values_per_lane];
        const device vec<bfloat, 4>* input_vectors =
            (const device vec<bfloat, 4>*)(
                activated + lane * values_per_lane);
        for (uint i = 0; i < values_per_lane / 4; ++i) {
            const vec<bfloat, 4> values = input_vectors[i];
            input_values[4 * i] = values[0];
            input_values[4 * i + 1] = values[1];
            input_values[4 * i + 2] = values[2];
            input_values[4 * i + 3] = values[3];
        }

        uint8_t escape_val = down_scales_escape[0];

        thread float result[outputs_per_simd] = {
            0.0f, 0.0f, 0.0f, 0.0f
        };
        for (uint row = 0; row < outputs_per_simd; ++row) {
            uint output_row = first_row + row;
            const device uint8_t* weight =
                (const device uint8_t*)down_weight +
                output_row * packed_row_bytes + lane * 8;
            const device uint8_t* scale =
                down_scales + output_row * scale_row_bytes + lane / 2;
            uint8_t sb = scale[0];
            if (output_row == 0 && lane == 1)
                sb = escape_val;
            result[row] = laguna_nvfp4_qdot_16(
                weight,
                input_values,
                laguna_nvfp4_scale(sb));
            result[row] = simd_sum(result[row]);
        }

        if (lane == 0) {
            for (uint row = 0; row < outputs_per_simd; ++row) {
                uint output_row = first_row + row;
                bfloat shared = bfloat(result[row]\(lagunaNvfp4RowScaleSuffix));
                bfloat r2 = bfloat(routed[output_row] + shared);
                output[output_row] =
                    bfloat(residual[output_row] + r2);
            }
        }
        """,
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)

func lagunaSharedDownResidual(
    _ activated: MLXArray,
    downWeight: MLXArray,
    downScales: MLXArray,
    downScalesEscape: MLXArray,
    routed: MLXArray,
    residual: MLXArray
) -> MLXArray {
    precondition(activated.dtype == .bfloat16)
    precondition(
        activated.dims(1, 1, LagunaConstants.sharedExpertIntermediateSize))
    precondition(downWeight.dtype == .uint32)
    precondition(
        downWeight.dims(LagunaConstants.hiddenSize,
            LagunaConstants.sharedExpertIntermediateSize / 8))
    precondition(downScales.dtype == .uint8)
    precondition(
        downScales.dims(LagunaConstants.hiddenSize,
            LagunaConstants.sharedExpertIntermediateSize / 32))
    precondition(downScalesEscape.dtype == .uint8)
    precondition(routed.dtype == .bfloat16)
    precondition(routed.dims(1, 1, LagunaConstants.hiddenSize))
    precondition(residual.dtype == .bfloat16)
    precondition(residual.dims(1, 1, LagunaConstants.hiddenSize))

    return lagunaSharedDownResidualKernel(
        [activated, downWeight, downScales, downScalesEscape, routed, residual],
        grid: ((LagunaConstants.hiddenSize / 8) * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.hiddenSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

private let lagunaRoutedSwiGLUQMVKernel = MLXFast.metalKernel(
    name: "laguna_routed_nvfp4_swiglu_qmv_bf16_v2",
    inputNames: ["input", "fused_weight", "fused_scales", "indices"],
    outputNames: ["activated"],
    source: """
        constexpr uint input_width = 2048;
        constexpr uint output_width = 512;
        constexpr uint fused_width = 1024;
        constexpr uint packed_row_bytes = 1024;
        constexpr uint scale_row_bytes = 128;
        constexpr uint packed_expert_bytes = fused_width * packed_row_bytes;
        constexpr uint scale_expert_bytes = fused_width * scale_row_bytes;
        constexpr uint block_width = 512;
        constexpr uint values_per_lane = 16;
        constexpr uint tiles_per_expert = 128;
        constexpr uint routed_experts = 8;

        // Tile-major order keeps one threadgroup per expert exactly as before,
        // but places the eight independent weight banks next to one another in
        // the dispatch stream. This exposes expert-bank memory latency across
        // a scheduling wave instead of issuing all 128 tiles of one expert
        // before touching the next bank.
        uint group = threadgroup_position_in_grid.x;
        uint expert_slot = group % routed_experts;
        uint tile = group / routed_experts;
        uint expert = uint(indices[expert_slot]);
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint first_row = tile * 4 + simd_group * 2;

        const device uint8_t* expert_weight =
            (const device uint8_t*)fused_weight +
            expert * packed_expert_bytes;
        const device uint8_t* expert_scales =
            fused_scales + expert * scale_expert_bytes;

        thread float gate_result[2] = {0.0f, 0.0f};
        thread float up_result[2] = {0.0f, 0.0f};
        thread float input_values[values_per_lane];

        for (uint block = 0; block < input_width; block += block_width) {
            const device vec<bfloat, 4>* input_vectors =
                (const device vec<bfloat, 4>*)(
                    input + block + lane * values_per_lane);
            for (uint i = 0; i < values_per_lane / 4; ++i) {
                const vec<bfloat, 4> values = input_vectors[i];
                input_values[4 * i] = values[0];
                input_values[4 * i + 1] = values[1];
                input_values[4 * i + 2] = values[2];
                input_values[4 * i + 3] = values[3];
            }

            for (uint row = 0; row < 2; ++row) {
                uint logical_row = first_row + row;
                uint pair_tile = logical_row / 32;
                uint gate_row = pair_tile * 64 + logical_row % 32;
                uint up_row = gate_row + 32;
                const device uint8_t* gate_weight =
                    expert_weight + gate_row * packed_row_bytes +
                    block / 2 + lane * 8;
                const device uint8_t* up_weight =
                    expert_weight + up_row * packed_row_bytes +
                    block / 2 + lane * 8;
                const device uint8_t* gate_scale =
                    expert_scales + gate_row * scale_row_bytes +
                    block / 16 + lane;
                const device uint8_t* up_scale =
                    expert_scales + up_row * scale_row_bytes +
                    block / 16 + lane;

                gate_result[row] += laguna_nvfp4_qdot_16(
                    gate_weight,
                    input_values,
                    laguna_nvfp4_scale(gate_scale[0]));
                up_result[row] += laguna_nvfp4_qdot_16(
                    up_weight,
                    input_values,
                    laguna_nvfp4_scale(up_scale[0]));
            }
        }

        {
            const vec<float, 4> packed = simd_sum(
                vec<float, 4>(gate_result[0], gate_result[1],
                              up_result[0], up_result[1]));
            gate_result[0] = packed.x; gate_result[1] = packed.y;
            up_result[0] = packed.z; up_result[1] = packed.w;
        }
        for (uint row = 0; row < 2; ++row) {
            if (lane == 0) {
                bfloat gate = bfloat(gate_result[row]\(lagunaNvfp4RowScaleSuffix));
                bfloat up = bfloat(up_result[row]\(lagunaNvfp4RowScaleSuffix));
                bfloat exp_abs = metal::exp(metal::abs(gate));
                bfloat denominator = bfloat(1) + exp_abs;
                bfloat y = bfloat(1) / denominator;
                bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
                bfloat silu = bfloat(gate * sigmoid);
                activated[
                    expert_slot * output_width + first_row + row
                ] = bfloat(silu * up);
            }
        }
        """,
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)

/// R1 scheduling twin of `lagunaRoutedSwiGLUQMVKernel`: each simdgroup owns
/// one output row rather than two. Two simdgroups per 64-thread group and 256
/// tiles cover all 512 expert rows exactly once.
private let lagunaRoutedSwiGLUQMVRows1Kernel = MLXFast.metalKernel(
    name: "laguna_routed_nvfp4_swiglu_qmv_rows1_bf16_v1",
    inputNames: ["input", "fused_weight", "fused_scales", "indices"],
    outputNames: ["activated"],
    source: """
        constexpr uint input_width = 2048;
        constexpr uint output_width = 512;
        constexpr uint fused_width = 1024;
        constexpr uint packed_row_bytes = 1024;
        constexpr uint scale_row_bytes = 128;
        constexpr uint packed_expert_bytes = fused_width * packed_row_bytes;
        constexpr uint scale_expert_bytes = fused_width * scale_row_bytes;
        constexpr uint block_width = 512;
        constexpr uint values_per_lane = 16;
        constexpr uint tiles_per_expert = 256;
        constexpr uint routed_experts = 8;

        uint group = threadgroup_position_in_grid.x;
        uint expert_slot = group % routed_experts;
        uint tile = group / routed_experts;
        uint expert = uint(indices[expert_slot]);
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint logical_row = tile * 2 + simd_group;

        const device uint8_t* expert_weight =
            (const device uint8_t*)fused_weight +
            expert * packed_expert_bytes;
        const device uint8_t* expert_scales =
            fused_scales + expert * scale_expert_bytes;

        uint pair_tile = logical_row / 32;
        uint gate_row = pair_tile * 64 + logical_row % 32;
        uint up_row = gate_row + 32;
        const device uint8_t* gate_row_weight =
            expert_weight + gate_row * packed_row_bytes + lane * 8;
        const device uint8_t* up_row_weight =
            expert_weight + up_row * packed_row_bytes + lane * 8;
        const device uint8_t* gate_row_scale =
            expert_scales + gate_row * scale_row_bytes + lane;
        const device uint8_t* up_row_scale =
            expert_scales + up_row * scale_row_bytes + lane;

        thread float gate_result = 0.0f;
        thread float up_result = 0.0f;
        thread float input_values[values_per_lane];

        for (uint block = 0; block < input_width; block += block_width) {
            const device vec<bfloat, 4>* input_vectors =
                (const device vec<bfloat, 4>*) (
                    input + block + lane * values_per_lane);
            for (uint i = 0; i < values_per_lane / 4; ++i) {
                const vec<bfloat, 4> values = input_vectors[i];
                input_values[4 * i] = values[0];
                input_values[4 * i + 1] = values[1];
                input_values[4 * i + 2] = values[2];
                input_values[4 * i + 3] = values[3];
            }

            const device uint8_t* gate_weight =
                gate_row_weight + block / 2;
            const device uint8_t* up_weight =
                up_row_weight + block / 2;
            const device uint8_t* gate_scale =
                gate_row_scale + block / 16;
            const device uint8_t* up_scale =
                up_row_scale + block / 16;

            gate_result += laguna_nvfp4_qdot_16(
                gate_weight,
                input_values,
                laguna_nvfp4_scale(gate_scale[0]));
            up_result += laguna_nvfp4_qdot_16(
                up_weight,
                input_values,
                laguna_nvfp4_scale(up_scale[0]));
        }

        {
            const vec<float, 2> packed = simd_sum(
                vec<float, 2>(gate_result, up_result));
            gate_result = packed.x;
            up_result = packed.y;
        }
        if (lane == 0) {
            bfloat gate = bfloat(gate_result\(lagunaNvfp4RowScaleSuffix));
            bfloat up = bfloat(up_result\(lagunaNvfp4RowScaleSuffix));
            bfloat exp_abs = metal::exp(metal::abs(gate));
            bfloat denominator = bfloat(1) + exp_abs;
            bfloat y = bfloat(1) / denominator;
            bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
            bfloat silu = bfloat(gate * sigmoid);
            activated[expert_slot * output_width + logical_row] =
                bfloat(silu * up);
        }
        """,
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)

func lagunaRoutedSwiGLUQMV(
    _ input: MLXArray,
    fusedWeight: MLXArray,
    fusedScales: MLXArray,
    indices: MLXArray
) -> MLXArray {
    precondition(input.dtype == .bfloat16)
    precondition(input.dims(1, 1, LagunaConstants.hiddenSize))
    precondition(fusedWeight.dtype == .uint32)
    precondition(
        fusedWeight.dims(LagunaConstants.numExperts, 2 * LagunaConstants.moeIntermediateSize,
            LagunaConstants.hiddenSize / 8))
    precondition(fusedScales.dtype == .uint8)
    precondition(
        fusedScales.dims(LagunaConstants.numExperts, 2 * LagunaConstants.moeIntermediateSize,
            LagunaConstants.hiddenSize / 16))
    precondition(indices.dtype == .uint32)
    precondition(indices.dims(1, 1, LagunaConstants.numExpertsPerTok))

    let kernel =
        lagunaSwiGLUQMVRows1Enabled
        ? lagunaRoutedSwiGLUQMVRows1Kernel
        : lagunaRoutedSwiGLUQMVKernel
    let tilesPerSlot = lagunaSwiGLUQMVRows1Enabled ? 256 : 128
    return kernel(
        [input, fusedWeight, fusedScales, indices],
        grid: (LagunaConstants.numExpertsPerTok * tilesPerSlot * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[
            1, 1, LagunaConstants.numExpertsPerTok, 1,
            LagunaConstants.moeIntermediateSize,
        ]],
        outputDTypes: [.bfloat16]
    )[0]
}

/// `DARKBLOOM_PACKED_SCALES` twin of `lagunaRoutedSwiGLUQMVKernel` consuming
/// the walk-order scale side bank built by
/// `preparePackedRoutedGateUpBank`. Bank layout, per expert:
/// `[tile 128][k-block 4][sub 8][32 scale bytes]` where `sub =
/// (simd_group*2 + row)*2 + {0 gate, 1 up}`. The stock kernel's
/// `gate_row/up_row` remap is baked into the scale bank while its fused code
/// bank is reused directly, so per (row, k-block, lane) this kernel issues
/// the identical one-byte scale and uint2 code loads and runs the textually
/// identical dequant/accumulate/SwiGLU chain — only scale address computation
/// differs.
private let lagunaRoutedSwiGLUQMVPackedKernel = MLXFast.metalKernel(
    name: "laguna_routed_nvfp4_swiglu_qmv_packed_bf16_v1",
    inputNames: ["input", "fused_weight", "packed_scales", "indices"],
    outputNames: ["activated"],
    source: """
        constexpr uint input_width = 2048;
        constexpr uint output_width = 512;
        constexpr uint block_width = 512;
        constexpr uint values_per_lane = 16;
        constexpr uint routed_experts = 8;
        constexpr uint fused_row_bytes = 1024;
        constexpr uint fused_expert_bytes = 1024 * fused_row_bytes;
        constexpr uint scale_row_bytes = 32;
        constexpr uint scale_sub_bytes = 8 * scale_row_bytes;
        constexpr uint scale_kblock_bytes = scale_sub_bytes;
        constexpr uint scale_tile_bytes = 4 * scale_kblock_bytes;
        constexpr uint packed_expert_bytes = 128 * scale_tile_bytes;

        uint group = threadgroup_position_in_grid.x;
        uint expert_slot = group % routed_experts;
        uint tile = group / routed_experts;
        uint expert = uint(indices[expert_slot]);
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint first_row = tile * 4 + simd_group * 2;

        const device uint8_t* expert_weight =
            (const device uint8_t*)fused_weight +
            expert * fused_expert_bytes;
        const device uint8_t* tile_scales =
            packed_scales + expert * packed_expert_bytes
            + tile * scale_tile_bytes;

        thread float gate_result[2] = {0.0f, 0.0f};
        thread float up_result[2] = {0.0f, 0.0f};
        thread float input_values[values_per_lane];

        for (uint block = 0; block < input_width; block += block_width) {
            const device vec<bfloat, 4>* input_vectors =
                (const device vec<bfloat, 4>*)(
                    input + block + lane * values_per_lane);
            for (uint i = 0; i < values_per_lane / 4; ++i) {
                const vec<bfloat, 4> values = input_vectors[i];
                input_values[4 * i] = values[0];
                input_values[4 * i + 1] = values[1];
                input_values[4 * i + 2] = values[2];
                input_values[4 * i + 3] = values[3];
            }

            const device uint8_t* block_scales =
                tile_scales + (block / block_width) * scale_kblock_bytes;
            for (uint row = 0; row < 2; ++row) {
                uint logical_row = tile * 4 + simd_group * 2 + row;
                uint gate_row = (logical_row / 32) * 64 + logical_row % 32;
                uint up_row = gate_row + 32;
                uint sub = simd_group * 2 + row;
                const device uint8_t* gate_scale =
                    block_scales + sub * 2 * scale_row_bytes + lane;
                const device uint8_t* up_scale =
                    gate_scale + scale_row_bytes;
                const device uint8_t* gate_weight =
                    expert_weight + gate_row * fused_row_bytes
                    + block / 2 + lane * 8;
                const device uint8_t* up_weight =
                    expert_weight + up_row * fused_row_bytes
                    + block / 2 + lane * 8;

                gate_result[row] += laguna_nvfp4_qdot_16(
                    gate_weight,
                    input_values,
                    laguna_nvfp4_scale(gate_scale[0]));
                up_result[row] += laguna_nvfp4_qdot_16(
                    up_weight,
                    input_values,
                    laguna_nvfp4_scale(up_scale[0]));
            }
        }

        {
            const vec<float, 4> packed = simd_sum(
                vec<float, 4>(gate_result[0], gate_result[1],
                              up_result[0], up_result[1]));
            gate_result[0] = packed.x; gate_result[1] = packed.y;
            up_result[0] = packed.z; up_result[1] = packed.w;
        }
        for (uint row = 0; row < 2; ++row) {
            if (lane == 0) {
                bfloat gate = bfloat(gate_result[row]\(lagunaNvfp4RowScaleSuffix));
                bfloat up = bfloat(up_result[row]\(lagunaNvfp4RowScaleSuffix));
                bfloat exp_abs = metal::exp(metal::abs(gate));
                bfloat denominator = bfloat(1) + exp_abs;
                bfloat y = bfloat(1) / denominator;
                bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
                bfloat silu = bfloat(gate * sigmoid);
                activated[
                    expert_slot * output_width + first_row + row
                ] = bfloat(silu * up);
            }
        }
        """,
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)

func lagunaRoutedSwiGLUQMVPacked(
    _ input: MLXArray,
    fusedWeight: MLXArray,
    packedScales: MLXArray,
    indices: MLXArray
) -> MLXArray {
    precondition(input.dtype == .bfloat16)
    precondition(input.dims(1, 1, LagunaConstants.hiddenSize))
    precondition(fusedWeight.dtype == .uint32)
    precondition(
        fusedWeight.dims(LagunaConstants.numExperts, 2 * LagunaConstants.moeIntermediateSize,
            LagunaConstants.hiddenSize / 8))
    precondition(packedScales.dtype == .uint8)
    precondition(
        packedScales.dims(LagunaConstants.numExperts, 2 * LagunaConstants.moeIntermediateSize * 4,
            LagunaConstants.hiddenSize / 64))
    precondition(indices.dtype == .uint32)
    precondition(indices.dims(1, 1, LagunaConstants.numExpertsPerTok))

    return lagunaRoutedSwiGLUQMVPackedKernel(
        [input, fusedWeight, packedScales, indices],
        grid: (LagunaConstants.numExpertsPerTok * 128 * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[
            1, 1, LagunaConstants.numExpertsPerTok, 1,
            LagunaConstants.moeIntermediateSize,
        ]],
        outputDTypes: [.bfloat16]
    )[0]
}

/// Packed routed QMV body specialized with an alternate expert-selection
/// prologue. The ordinary accepted kernel above stays byte-for-byte unchanged;
/// this generator is used only by the exact router-key twin below.
func lagunaRoutedSwiGLUQMVPackedSelectedSource(
    prologue: String, expertExpression: String
) -> String {
    """
        constexpr uint input_width = 2048;
        constexpr uint output_width = 512;
        constexpr uint block_width = 512;
        constexpr uint values_per_lane = 16;
        constexpr uint routed_experts = 8;
        constexpr uint fused_row_bytes = 1024;
        constexpr uint fused_expert_bytes = 1024 * fused_row_bytes;
        constexpr uint scale_row_bytes = 32;
        constexpr uint scale_sub_bytes = 8 * scale_row_bytes;
        constexpr uint scale_kblock_bytes = scale_sub_bytes;
        constexpr uint scale_tile_bytes = 4 * scale_kblock_bytes;
        constexpr uint packed_expert_bytes = 128 * scale_tile_bytes;

        uint group = threadgroup_position_in_grid.x;
        uint expert_slot = group % routed_experts;
        uint tile = group / routed_experts;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint first_row = tile * 4 + simd_group * 2;
        \(prologue)
        uint expert = \(expertExpression);

        const device uint8_t* expert_weight =
            (const device uint8_t*)fused_weight + expert * fused_expert_bytes;
        const device uint8_t* tile_scales =
            packed_scales + expert * packed_expert_bytes
            + tile * scale_tile_bytes;

        thread float gate_result[2] = {0.0f, 0.0f};
        thread float up_result[2] = {0.0f, 0.0f};
        thread float input_values[values_per_lane];

        for (uint block = 0; block < input_width; block += block_width) {
            const device vec<bfloat, 4>* input_vectors =
                (const device vec<bfloat, 4>*) (
                    input + block + lane * values_per_lane);
            for (uint i = 0; i < values_per_lane / 4; ++i) {
                const vec<bfloat, 4> values = input_vectors[i];
                input_values[4 * i] = values[0];
                input_values[4 * i + 1] = values[1];
                input_values[4 * i + 2] = values[2];
                input_values[4 * i + 3] = values[3];
            }

            const device uint8_t* block_scales =
                tile_scales + (block / block_width) * scale_kblock_bytes;
            for (uint row = 0; row < 2; ++row) {
                uint logical_row = tile * 4 + simd_group * 2 + row;
                uint gate_row = (logical_row / 32) * 64 + logical_row % 32;
                uint up_row = gate_row + 32;
                uint sub = simd_group * 2 + row;
                const device uint8_t* gate_scale =
                    block_scales + sub * 2 * scale_row_bytes + lane;
                const device uint8_t* up_scale = gate_scale + scale_row_bytes;
                const device uint8_t* gate_weight =
                    expert_weight + gate_row * fused_row_bytes
                    + block / 2 + lane * 8;
                const device uint8_t* up_weight =
                    expert_weight + up_row * fused_row_bytes
                    + block / 2 + lane * 8;

                gate_result[row] += laguna_nvfp4_qdot_16(
                    gate_weight, input_values,
                    laguna_nvfp4_scale(gate_scale[0]));
                up_result[row] += laguna_nvfp4_qdot_16(
                    up_weight, input_values,
                    laguna_nvfp4_scale(up_scale[0]));
            }
        }

        {
            const vec<float, 4> packed = simd_sum(
                vec<float, 4>(gate_result[0], gate_result[1],
                              up_result[0], up_result[1]));
            gate_result[0] = packed.x; gate_result[1] = packed.y;
            up_result[0] = packed.z; up_result[1] = packed.w;
        }
        for (uint row = 0; row < 2; ++row) {
            if (lane == 0) {
                bfloat gate = bfloat(gate_result[row]\(lagunaNvfp4RowScaleSuffix));
                bfloat up = bfloat(up_result[row]\(lagunaNvfp4RowScaleSuffix));
                bfloat exp_abs = metal::exp(metal::abs(gate));
                bfloat denominator = bfloat(1) + exp_abs;
                bfloat y = bfloat(1) / denominator;
                bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
                bfloat silu = bfloat(gate * sigmoid);
                activated[expert_slot * output_width + first_row + row] =
                    bfloat(silu * up);
            }
        }
        """
}

private let lagunaRoutedSwiGLUQMVPackedTop8Kernel = MLXFast.metalKernel(
    name: "laguna_routed_nvfp4_swiglu_qmv_packed_top8keys_bf16_v1",
    inputNames: ["input", "fused_weight", "packed_scales", "indices"],
    outputNames: ["activated"],
    source: lagunaRoutedSwiGLUQMVPackedSelectedSource(
        prologue: "",
        expertExpression: "uint(indices[expert_slot])"),
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)

/// `DARKBLOOM_ROUTED_GATEUP_R1` (default ON; set "0" to restore the accepted
/// two-rows-per-simdgroup pipeline): one output row per simdgroup for the
/// routed gate/up packed QMV, with twice the threadgroups — the promoted
/// down-kernel R1 retile's ownership geometry (isolated receipt `8d35b19d`,
/// composed in promoted `05e7894f`) applied to its gate/up sibling. Per
/// output row the operation sequence is identical: same bank bytes via
/// `bank_tile = logical_row / 4`, `sub = logical_row % 4`, same qdot and
/// `simd_sum` order, same suffix/SwiGLU/BF16 boundaries, one writer per row.
let lagunaRoutedGateUpR1Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ROUTED_GATEUP_R1"] != "0"

private let lagunaRoutedSwiGLUQMVPackedTop8R1Kernel = MLXFast.metalKernel(
    name: "laguna_routed_nvfp4_swiglu_qmv_packed_top8keys_r1_bf16_v2_halved",
    inputNames: ["input", "fused_weight", "packed_scales", "gate_up_escape", "indices",
                 "shared_weight", "shared_scales", "shared_escape"],
    outputNames: ["activated"],
    source: """
        constexpr uint input_width = 2048;
        constexpr uint output_width = 512;
        constexpr uint block_width = 512;
        constexpr uint values_per_lane = 16;
        constexpr uint total_slots = 9;
        constexpr uint routed_experts = 8;
        constexpr uint fused_row_bytes = 1024;
        constexpr uint fused_expert_bytes = 1024 * fused_row_bytes;
        constexpr uint r_scale_row_bytes = 16;
        constexpr uint r_scale_kblock_bytes = 8 * r_scale_row_bytes;
        constexpr uint r_scale_tile_bytes = 4 * r_scale_kblock_bytes;
        constexpr uint r_packed_expert_bytes = 128 * r_scale_tile_bytes;
        constexpr uint s_scale_row_bytes = 64;

        uint group = threadgroup_position_in_grid.x;
        uint expert_slot = group % total_slots;
        uint tile = group / total_slots;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint logical_row = tile * 2 + simd_group;

        thread float gate_result = 0.0f;
        thread float up_result = 0.0f;
        thread float input_values[values_per_lane];

        if (expert_slot < routed_experts) {
            uint expert = uint(indices[expert_slot]);
            const device uint8_t* expert_weight =
                (const device uint8_t*)fused_weight + expert * fused_expert_bytes;
            const device uint8_t* row_scales =
                packed_scales + expert * r_packed_expert_bytes
                + (logical_row / 4) * r_scale_tile_bytes;
            const device uint8_t* expert_escape =
                gate_up_escape + expert * 2;
            uint sub = logical_row % 4;
            uint gate_row = (logical_row / 32) * 64 + logical_row % 32;
            uint up_row = gate_row + 32;

            uint2 gate_codes;
            uint2 up_codes;
            uint8_t gate_sb;
            uint8_t up_sb;
            {
                const device uint8_t* first_scales =
                    row_scales + sub * 2 * r_scale_row_bytes + lane / 2;
                gate_sb = first_scales[0];
                up_sb = first_scales[r_scale_row_bytes];
                if (logical_row == 0 && lane == 1) {
                    gate_sb = expert_escape[0];
                    up_sb = expert_escape[1];
                }
                gate_codes = *(const device uint2*)(
                    expert_weight + gate_row * fused_row_bytes + lane * 8);
                up_codes = *(const device uint2*)(
                    expert_weight + up_row * fused_row_bytes + lane * 8);
            }

            for (uint block = 0; block < input_width; block += block_width) {
                const device vec<bfloat, 4>* input_vectors =
                    (const device vec<bfloat, 4>*) (
                        input + block + lane * values_per_lane);
                for (uint i = 0; i < values_per_lane / 4; ++i) {
                    *(thread float4*)(input_values + 4 * i) =
                        float4(input_vectors[i]);
                }

                const uint2 cur_gate_codes = gate_codes;
                const uint2 cur_up_codes = up_codes;
                const uint8_t cur_gate_sb = gate_sb;
                const uint8_t cur_up_sb = up_sb;
                const uint next_block = block + block_width;
                if (next_block < input_width) {
                    const device uint8_t* next_scales =
                        row_scales + (next_block / block_width) * r_scale_kblock_bytes
                        + sub * 2 * r_scale_row_bytes + lane / 2;
                    gate_sb = next_scales[0];
                    up_sb = next_scales[r_scale_row_bytes];
                    gate_codes = *(const device uint2*)(
                        expert_weight + gate_row * fused_row_bytes
                        + next_block / 2 + lane * 8);
                    up_codes = *(const device uint2*)(
                        expert_weight + up_row * fused_row_bytes
                        + next_block / 2 + lane * 8);
                }

                gate_result += laguna_nvfp4_qdot_codes_16(
                    cur_gate_codes, input_values,
                    laguna_nvfp4_scale(cur_gate_sb));
                up_result += laguna_nvfp4_qdot_codes_16(
                    cur_up_codes, input_values,
                    laguna_nvfp4_scale(cur_up_sb));
            }
        } else {
            uint row = logical_row;
            const device uint8_t* gate_row_weight =
                (const device uint8_t*)shared_weight +
                row * fused_row_bytes + lane * 8;
            const device uint8_t* up_row_weight =
                (const device uint8_t*)shared_weight +
                (row + output_width) * fused_row_bytes + lane * 8;
            const device uint8_t* gate_row_scale =
                shared_scales + row * s_scale_row_bytes + lane / 2;
            const device uint8_t* up_row_scale =
                shared_scales + (row + output_width) * s_scale_row_bytes + lane / 2;
            uint8_t gate_escape = shared_escape[0];
            uint8_t up_escape = shared_escape[1];

            for (uint block = 0; block < input_width; block += block_width) {
                const device vec<bfloat, 4>* input_vectors =
                    (const device vec<bfloat, 4>*) (
                        input + block + lane * values_per_lane);
                for (uint i = 0; i < values_per_lane / 4; ++i) {
                    *(thread float4*)(input_values + 4 * i) =
                        float4(input_vectors[i]);
                }

                uint8_t gate_sb = gate_row_scale[block / 16];
                uint8_t up_sb = up_row_scale[block / 16];
                if (row == 0 && lane == 1) {
                    gate_sb = gate_escape;
                    up_sb = up_escape;
                }
                gate_result += laguna_nvfp4_qdot_16(
                    gate_row_weight + block / 2, input_values,
                    laguna_nvfp4_scale(gate_sb));
                up_result += laguna_nvfp4_qdot_16(
                    up_row_weight + block / 2, input_values,
                    laguna_nvfp4_scale(up_sb));
            }
        }

        {
            const vec<float, 2> packed = simd_sum(
                vec<float, 2>(gate_result, up_result));
            gate_result = packed.x;
            up_result = packed.y;
        }
        if (lane == 0) {
            bfloat gate = bfloat(gate_result\(lagunaNvfp4RowScaleSuffix));
            bfloat up = bfloat(up_result\(lagunaNvfp4RowScaleSuffix));
            bfloat exp_abs = metal::exp(metal::abs(gate));
            bfloat denominator = bfloat(1) + exp_abs;
            bfloat y = bfloat(1) / denominator;
            bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
            bfloat silu = bfloat(gate * sigmoid);
            activated[expert_slot * output_width + logical_row] =
                bfloat(silu * up);
        }
        """,
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)

func lagunaRoutedSwiGLUQMVPackedTop8(
    _ input: MLXArray,
    fusedWeight: MLXArray,
    packedScales: MLXArray,
    gateUpEscape: MLXArray,
    indices: MLXArray,
    sharedWeight: MLXArray? = nil,
    sharedScales: MLXArray? = nil,
    sharedEscape: MLXArray? = nil
) -> MLXArray {
    precondition(input.dtype == .bfloat16)
    precondition(input.dims(1, 1, LagunaConstants.hiddenSize))
    precondition(fusedWeight.dtype == .uint32)
    precondition(packedScales.dtype == .uint8)
    precondition(gateUpEscape.dtype == .uint8)
    precondition(indices.dtype == .uint32)
    precondition(indices.dims(1, 1, LagunaConstants.numExpertsPerTok))

    if lagunaRoutedGateUpR1Enabled,
        let sw = sharedWeight, let ss = sharedScales, let se = sharedEscape
    {
        precondition(sw.dtype == .uint32)
        precondition(ss.dtype == .uint8)
        precondition(se.dtype == .uint8)
        return lagunaRoutedSwiGLUQMVPackedTop8R1Kernel(
            [input, fusedWeight, packedScales, gateUpEscape, indices, sw, ss, se],
            grid: (9 * 256 * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[
                1, 1, 9, 1,
                LagunaConstants.moeIntermediateSize,
            ]],
            outputDTypes: [.bfloat16]
        )[0]
    }
    return lagunaRoutedSwiGLUQMVPackedTop8Kernel(
        [input, fusedWeight, packedScales, indices],
        grid: (LagunaConstants.numExpertsPerTok * 128 * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[
            1, 1, LagunaConstants.numExpertsPerTok, 1,
            LagunaConstants.moeIntermediateSize,
        ]],
        outputDTypes: [.bfloat16]
    )[0]
}

private let lagunaRoutedDownReduceKernel = MLXFast.metalKernel(
    name: "laguna_routed_nvfp4_down_reduce_bf16_v2",
    inputNames: [
        "activated", "down_weight", "down_scales", "indices", "router_weights",
    ],
    outputNames: ["routed"],
    source: """
        constexpr uint input_width = 512;
        constexpr uint output_width = 2048;
        constexpr uint experts_per_token = 8;
        constexpr uint outputs_per_simd = 4;
        constexpr uint values_per_lane = 16;
        constexpr uint packed_row_bytes = 256;
        constexpr uint scale_row_bytes = 32;
        constexpr uint packed_expert_bytes =
            output_width * packed_row_bytes;
        constexpr uint scale_expert_bytes =
            output_width * scale_row_bytes;

        uint tile = threadgroup_position_in_grid.x;
        uint expert_slot = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint first_row = tile * outputs_per_simd;
        uint expert = uint(indices[expert_slot]);

        const device bfloat* expert_input =
            activated + expert_slot * input_width;
        const device uint8_t* expert_weight =
            (const device uint8_t*)down_weight +
            expert * packed_expert_bytes;
        const device uint8_t* expert_scales =
            down_scales + expert * scale_expert_bytes;

        thread float input_values[values_per_lane];
        const device vec<bfloat, 4>* input_vectors =
            (const device vec<bfloat, 4>*)(
                expert_input + lane * values_per_lane);
        for (uint i = 0; i < values_per_lane / 4; ++i) {
            const vec<bfloat, 4> values = input_vectors[i];
            input_values[4 * i] = values[0];
            input_values[4 * i + 1] = values[1];
            input_values[4 * i + 2] = values[2];
            input_values[4 * i + 3] = values[3];
        }

        thread float result[outputs_per_simd] = {
            0.0f, 0.0f, 0.0f, 0.0f
        };
        // Rows' code words/scales stage first; one packed simd_sum, same
        // bytes/decode/order per component as stock.
        uint2 row_codes[outputs_per_simd];
        uint8_t row_sb[outputs_per_simd];
        for (uint row = 0; row < outputs_per_simd; ++row) {
            uint output_row = first_row + row;
            row_codes[row] = *(const device uint2*)(
                expert_weight + output_row * packed_row_bytes + lane * 8);
            row_sb[row] =
                expert_scales[output_row * scale_row_bytes + lane];
        }
        for (uint row = 0; row < outputs_per_simd; ++row) {
            result[row] = laguna_nvfp4_qdot_codes_16(
                row_codes[row],
                input_values,
                laguna_nvfp4_scale(row_sb[row]));
        }
        {
            const vec<float, 4> packed_rows = simd_sum(
                vec<float, 4>(result[0], result[1], result[2], result[3]));
            result[0] = packed_rows.x;
            result[1] = packed_rows.y;
            result[2] = packed_rows.z;
            result[3] = packed_rows.w;
        }

        threadgroup bfloat expert_outputs[
            experts_per_token * outputs_per_simd
        ];
        if (lane == 0) {
            for (uint row = 0; row < outputs_per_simd; ++row) {
                expert_outputs[
                    expert_slot * outputs_per_simd + row
                ] = bfloat(result[row]\(lagunaNvfp4RowScaleSuffix));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // `weightedExpertSum` first multiplies BF16 expert outputs by router
        // weights cast from FP32 to BF16. Its small strided BF16 reduction
        // initializes with zero, then visits expert slots 0 through 7 in
        // order. The scalar 2.5 is constructed in the BF16 result dtype.
        if (expert_slot == 0 && lane < outputs_per_simd) {
            bfloat total = bfloat(0);
            for (uint slot = 0; slot < experts_per_token; ++slot) {
                bfloat route_weight = bfloat(router_weights[slot]);
                bfloat product = bfloat(
                    expert_outputs[slot * outputs_per_simd + lane] *
                    route_weight);
                total = bfloat(product + total);
            }
            routed[first_row + lane] = bfloat(total * bfloat(2.5f));
        }
        """,
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)

func lagunaRoutedDownReduce(
    _ activated: MLXArray,
    downWeight: MLXArray,
    downScales: MLXArray,
    indices: MLXArray,
    routerWeights: MLXArray
) -> MLXArray {
    precondition(activated.dtype == .bfloat16)
    precondition(
        activated.dims(1, 1, LagunaConstants.numExpertsPerTok, 1,
            LagunaConstants.moeIntermediateSize))
    precondition(downWeight.dtype == .uint32)
    precondition(
        downWeight.dims(LagunaConstants.numExperts, LagunaConstants.hiddenSize,
            LagunaConstants.moeIntermediateSize / 8))
    precondition(downScales.dtype == .uint8)
    precondition(
        downScales.dims(LagunaConstants.numExperts, LagunaConstants.hiddenSize,
            LagunaConstants.moeIntermediateSize / 16))
    precondition(indices.dtype == .uint32)
    precondition(indices.dims(1, 1, LagunaConstants.numExpertsPerTok))
    precondition(routerWeights.dtype == .float32)
    precondition(routerWeights.dims(1, 1, LagunaConstants.numExpertsPerTok))

    return lagunaRoutedDownReduceKernel(
        [activated, downWeight, downScales, indices, routerWeights],
        grid: ((LagunaConstants.hiddenSize / 4) * 256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.hiddenSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

/// Encode-order lever for the 9-slot down+residual dispatch. MLX's eval
/// traversal visits a node's inputs in declaration order, and Metal memory
/// barriers are encoder-wide. Hypothesis was that listing the shared-expert
/// inputs first would let the shared SwiGLU QMV overlap the router top-8
/// latency; MEASURED (2026-08-01, M5 Max 128 GB, driver rig, 150-step
/// cool-floor windows): shared-first REGRESSES ~+0.10 ms/step. Cause: in
/// the shipped routed-first order the shared QMV is encoded after the
/// routed QMV with no intervening barrier, so it already overlaps the
/// routed QMV on the GPU; moving it before the top-8 barrier makes the
/// barrier ahead of the routed QMV wait on the shared QMV too, LENGTHENING
/// the critical path (barriers are encoder-wide, not per-resource). Kept as
/// an opt-in A/B arm (`DARKBLOOM_SHARED_FIRST_DOWN=1`) for re-measurement
/// if the surrounding dispatch anatomy changes; both orders are bit-exact
/// (pure input permutation, kernel body addresses inputs by name; the
/// reordered variant carries a new kernel name because the JIT cache keys
/// signatures by name).
let lagunaSharedFirstDownOrderEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_SHARED_FIRST_DOWN"] == "1"

private let lagunaRoutedSharedDownResidualKernel = MLXFast.metalKernel(
    name: lagunaSharedFirstDownOrderEnabled
        ? "laguna_routed_shared_nvfp4_down_residual_bf16_r1_v5sf_halved"
        : "laguna_routed_shared_nvfp4_down_residual_bf16_r1_v5_halved",
    inputNames: lagunaSharedFirstDownOrderEnabled
        ? [
            "shared_activated", "shared_down_weight", "shared_down_scales",
            "shared_down_escape",
            "routed_activated", "routed_down_weight", "routed_down_scales",
            "routed_down_escape",
            "indices", "router_weights", "residual",
        ]
        : [
            "routed_activated", "routed_down_weight", "routed_down_scales",
            "routed_down_escape",
            "indices", "router_weights", "shared_activated",
            "shared_down_weight", "shared_down_scales",
            "shared_down_escape",
            "residual",
        ],
    outputNames: ["output"],
    source: """
        constexpr uint input_width = 512;
        constexpr uint output_width = 2048;
        constexpr uint routed_experts = 8;
        constexpr uint shared_slot = 8;
        constexpr uint outputs_per_simd = 8;
        constexpr uint values_per_lane = 16;
        constexpr uint packed_row_bytes = 256;
        constexpr uint scale_row_bytes = 16;
        constexpr uint packed_expert_bytes =
            output_width * packed_row_bytes;
        constexpr uint scale_expert_bytes =
            output_width * scale_row_bytes;

        uint tile = threadgroup_position_in_grid.x;
        uint slot = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint first_row = tile * outputs_per_simd;
        bool is_shared = slot == shared_slot;
        uint expert = is_shared ? 0 : uint(indices[slot]);

        const device bfloat* expert_input = is_shared
            ? shared_activated
            : routed_activated + slot * input_width;
        const device uint8_t* expert_weight = is_shared
            ? (const device uint8_t*)shared_down_weight
            : (const device uint8_t*)routed_down_weight +
                expert * packed_expert_bytes;
        const device uint8_t* expert_scales = is_shared
            ? shared_down_scales
            : routed_down_scales + expert * scale_expert_bytes;
        uint8_t escape_val;
        if (is_shared) {
            escape_val = shared_down_escape[0];
        } else {
            escape_val = routed_down_escape[expert];
        }

        thread float input_values[values_per_lane];
        const device vec<bfloat, 4>* input_vectors =
            (const device vec<bfloat, 4>*)(
                expert_input + lane * values_per_lane);
        for (uint i = 0; i < values_per_lane / 4; ++i) {
            *(thread float4*)(input_values + 4 * i) =
                float4(input_vectors[i]);
        }

        thread float result[outputs_per_simd] = {0.0f};
        for (uint row = 0; row < outputs_per_simd; ++row) {
            uint output_row = first_row + row;
            const device uint8_t* weight =
                expert_weight + output_row * packed_row_bytes + lane * 8;
            const device uint8_t* scale =
                expert_scales + output_row * scale_row_bytes + lane / 2;
            uint8_t sb = scale[0];
            if (output_row == 0 && lane == 1)
                sb = escape_val;
            result[row] = laguna_nvfp4_qdot_16(
                weight,
                input_values,
                laguna_nvfp4_scale(sb));
            result[row] = simd_sum(result[row]);
        }

        threadgroup bfloat down_outputs[
            (routed_experts + 1) * outputs_per_simd
        ];
        if (lane == 0) {
            for (uint row = 0; row < outputs_per_simd; ++row) {
                down_outputs[slot * outputs_per_simd + row] =
                    bfloat(result[row]\(lagunaNvfp4RowScaleSuffix));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (slot == 0 && lane < outputs_per_simd) {
            bfloat routed_total = bfloat(0);
            for (uint routed_slot = 0;
                 routed_slot < routed_experts;
                 ++routed_slot) {
                bfloat route_weight =
                    bfloat(router_weights[routed_slot]);
                bfloat product = bfloat(
                    down_outputs[
                        routed_slot * outputs_per_simd + lane
                    ] * route_weight);
                routed_total = bfloat(product + routed_total);
            }
            bfloat routed = bfloat(
                routed_total * bfloat(2.5f));
            bfloat shared =
                down_outputs[shared_slot * outputs_per_simd + lane];
            bfloat r2 = bfloat(routed + shared);
            output[first_row + lane] =
                bfloat(residual[first_row + lane] + r2);
        }
        """,
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)

func lagunaRoutedSharedDownResidual(
    routedActivated: MLXArray,
    routedDownWeight: MLXArray,
    routedDownScales: MLXArray,
    routedDownScalesEscape: MLXArray,
    indices: MLXArray,
    routerWeights: MLXArray,
    sharedActivated: MLXArray,
    sharedDownWeight: MLXArray,
    sharedDownScales: MLXArray,
    sharedDownScalesEscape: MLXArray,
    residual: MLXArray
) -> MLXArray {
    precondition(routedActivated.dtype == .bfloat16)
    precondition(
        routedActivated.dims(1, 1, LagunaConstants.numExpertsPerTok, 1,
            LagunaConstants.moeIntermediateSize))
    precondition(routedDownWeight.dtype == .uint32)
    precondition(
        routedDownWeight.dims(LagunaConstants.numExperts, LagunaConstants.hiddenSize,
            LagunaConstants.moeIntermediateSize / 8))
    precondition(routedDownScales.dtype == .uint8)
    precondition(
        routedDownScales.dims(LagunaConstants.numExperts, LagunaConstants.hiddenSize,
            LagunaConstants.moeIntermediateSize / 32))
    precondition(routedDownScalesEscape.dtype == .uint8)
    precondition(indices.dtype == .uint32)
    precondition(indices.dims(1, 1, LagunaConstants.numExpertsPerTok))
    precondition(routerWeights.dtype == .float32)
    precondition(routerWeights.dims(1, 1, LagunaConstants.numExpertsPerTok))
    precondition(sharedActivated.dtype == .bfloat16)
    precondition(
        sharedActivated.dims(1, 1, LagunaConstants.sharedExpertIntermediateSize))
    precondition(sharedDownWeight.dtype == .uint32)
    precondition(
        sharedDownWeight.dims(LagunaConstants.hiddenSize,
            LagunaConstants.sharedExpertIntermediateSize / 8))
    precondition(sharedDownScales.dtype == .uint8)
    precondition(
        sharedDownScales.dims(LagunaConstants.hiddenSize,
            LagunaConstants.sharedExpertIntermediateSize / 32))
    precondition(sharedDownScalesEscape.dtype == .uint8)
    precondition(residual.dtype == .bfloat16)
    precondition(residual.dims(1, 1, LagunaConstants.hiddenSize))

    return lagunaRoutedSharedDownResidualKernel(
        lagunaSharedFirstDownOrderEnabled
            ? [
                sharedActivated, sharedDownWeight, sharedDownScales,
                sharedDownScalesEscape,
                routedActivated, routedDownWeight, routedDownScales,
                routedDownScalesEscape,
                indices, routerWeights, residual,
            ]
            : [
                routedActivated, routedDownWeight, routedDownScales,
                routedDownScalesEscape,
                indices, routerWeights, sharedActivated,
                sharedDownWeight, sharedDownScales,
                sharedDownScalesEscape,
                residual,
            ],
        grid: (LagunaConstants.hiddenSize / 8 * 288, 1, 1),
        threadGroup: (288, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.hiddenSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

// MARK: - Layer-0 dense MLP fusion (BF16, no quantization)
//
// Layer 0's gate/up/down projections are plain BF16 `Linear` (never NVFP4),
// so stock ran four separate dispatches plus the residual add. The two
// kernels below fuse gate+up+SiLU-product (3 -> 1) and down+residual
// (2 -> 1). Exactness: each reuses a row loop already shipped in this file
// for the identical out_vec/in_vec shape — the gate/up kernel the
// `lagunaFusedQKVProjectionSource` tiling (8192 rows over 2048, the sliding
// Q projection's shape) with the BF16-round-first SiLU epilogue copied
// verbatim from `lagunaSharedSwiGLUQMVKernel`'s dtype-generic tail; the
// down kernel the `lagunaGatedOutputProjectionSource` loop (2048 rows over
// 8192) minus its gate multiply, with `lagunaSharedDownResidualKernel`'s
// round-then-add-then-round epilogue reproducing stock `h + r2`
// bit-for-bit.
private let lagunaDenseGateUpSwiGLUKernel = MLXFast.metalKernel(
    name: "laguna_dense_gate_up_swiglu_bf16_v1",
    inputNames: ["input", "fused_weight"],
    outputNames: ["activated"],
    source: """
        constexpr uint in_vec_size = 2048;
        constexpr uint output_width = 8192;
        constexpr uint rows_per_thread = 4;
        constexpr uint values_per_thread = 4;
        constexpr uint block_width = 128;
        constexpr uint blocks = in_vec_size / block_width;
        constexpr uint rows_per_group = 64;

        uint tile = threadgroup_position_in_grid.x;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        uint row_base = tile * rows_per_group + simd_group * rows_per_thread;

        thread float gate_result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
        thread float up_result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
        thread float coefficients[values_per_thread];

        uint column = lane * values_per_thread;
        for (uint block = 0; block < blocks; ++block) {
            // vec<bfloat,4> activation load: same bytes, same per-element
            // float conversion in the same order as the scalar loop — the
            // pattern this kernel already uses for its weight rows.
            // Alignment: column = lane * 4 elements = 8-byte multiples.
            const vec<bfloat, 4> c4 =
                *((const device vec<bfloat, 4>*)(input + column));
            for (uint i = 0; i < values_per_thread; ++i) {
                coefficients[i] = float(c4[i]);
            }
            for (uint row = 0; row < rows_per_thread; ++row) {
                const device vec<bfloat, 4>* gate_row_values =
                    (const device vec<bfloat, 4>*)(
                        fused_weight + (row_base + row) * in_vec_size + column);
                const vec<bfloat, 4> gw = gate_row_values[0];
                const device vec<bfloat, 4>* up_row_values =
                    (const device vec<bfloat, 4>*)(
                        fused_weight +
                        (output_width + row_base + row) * in_vec_size + column);
                const vec<bfloat, 4> uw = up_row_values[0];
                for (uint i = 0; i < values_per_thread; ++i) {
                    gate_result[row] += float(gw[i]) * coefficients[i];
                    up_result[row] += float(uw[i]) * coefficients[i];
                }
            }
            column += block_width;
        }

        for (uint row = 0; row < rows_per_thread; ++row) {
            for (ushort delta = 16; delta >= 1; delta >>= 1) {
                gate_result[row] +=
                    metal::simd_shuffle_down(gate_result[row], delta);
                up_result[row] +=
                    metal::simd_shuffle_down(up_result[row], delta);
            }
        }
        if (lane == 0) {
            for (uint row = 0; row < rows_per_thread; ++row) {
                bfloat gate = bfloat(gate_result[row]);
                bfloat up = bfloat(up_result[row]);
                bfloat exp_abs = metal::exp(metal::abs(gate));
                bfloat denominator = bfloat(1) + exp_abs;
                bfloat y = bfloat(1) / denominator;
                bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
                bfloat silu = bfloat(gate * sigmoid);
                activated[row_base + row] = bfloat(silu * up);
            }
        }
        """,
    ensureRowContiguous: true
)

/// `fusedWeight` is `concatenated([gateProj.weight, upProj.weight], axis: 0)`
/// -- gate rows first -- built once by `LagunaRuntimeMLP.prepareFusedDenseGateUp()`.
func lagunaDenseGateUpSwiGLU(
    _ input: MLXArray,
    fusedWeight: MLXArray
) -> MLXArray {
    precondition(input.dtype == .bfloat16)
    precondition(input.dims(1, 1, LagunaConstants.hiddenSize))
    precondition(fusedWeight.dtype == .bfloat16)
    precondition(
        fusedWeight.dims(2 * LagunaConstants.denseIntermediateSize, LagunaConstants.hiddenSize))

    return lagunaDenseGateUpSwiGLUKernel(
        [input, fusedWeight],
        grid: ((LagunaConstants.denseIntermediateSize / 64) * 512, 1, 1),
        threadGroup: (512, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.denseIntermediateSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

private let lagunaDenseDownResidualKernel = MLXFast.metalKernel(
    name: "laguna_dense_down_residual_bf16_v1",
    inputNames: ["activated", "down_weight", "residual"],
    outputNames: ["output"],
    source: """
        constexpr uint in_vec_size = 8192;
        constexpr uint rows_per_thread = 4;
        constexpr uint values_per_thread = 4;
        constexpr uint block_width = 128;
        constexpr uint blocks = in_vec_size / block_width;
        constexpr uint rows_per_group = 16;

        uint tile = threadgroup_position_in_grid.x;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        uint row_base = tile * rows_per_group + simd_group * rows_per_thread;

        thread float result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
        thread float coefficients[values_per_thread];

        uint column = lane * values_per_thread;
        for (uint block = 0; block < blocks; ++block) {
            // vec<bfloat,4> activation load — see the gate/up twin's note.
            const vec<bfloat, 4> c4 =
                *((const device vec<bfloat, 4>*)(activated + column));
            for (uint i = 0; i < values_per_thread; ++i) {
                coefficients[i] = float(c4[i]);
            }
            for (uint row = 0; row < rows_per_thread; ++row) {
                const device vec<bfloat, 4>* row_values =
                    (const device vec<bfloat, 4>*)(
                        down_weight + (row_base + row) * in_vec_size + column);
                const vec<bfloat, 4> w = row_values[0];
                for (uint i = 0; i < values_per_thread; ++i) {
                    result[row] += float(w[i]) * coefficients[i];
                }
            }
            column += block_width;
        }

        for (uint row = 0; row < rows_per_thread; ++row) {
            for (ushort delta = 16; delta >= 1; delta >>= 1) {
                result[row] += metal::simd_shuffle_down(result[row], delta);
            }
        }
        if (lane == 0) {
            for (uint row = 0; row < rows_per_thread; ++row) {
                bfloat down = bfloat(result[row]);
                output[row_base + row] =
                    bfloat(residual[row_base + row] + down);
            }
        }
        """,
    ensureRowContiguous: true
)

func lagunaDenseDownResidual(
    _ activated: MLXArray,
    downWeight: MLXArray,
    residual: MLXArray
) -> MLXArray {
    precondition(activated.dtype == .bfloat16)
    precondition(activated.dims(1, 1, LagunaConstants.denseIntermediateSize))
    precondition(downWeight.dtype == .bfloat16)
    precondition(
        downWeight.dims(LagunaConstants.hiddenSize, LagunaConstants.denseIntermediateSize))
    precondition(residual.dtype == .bfloat16)
    precondition(residual.dims(1, 1, LagunaConstants.hiddenSize))

    return lagunaDenseDownResidualKernel(
        [activated, downWeight, residual],
        grid: ((LagunaConstants.hiddenSize / 16) * 128, 1, 1),
        threadGroup: (128, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.hiddenSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

final class LagunaRuntimeMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    /// Retained fused NVFP4 `[gate; up]` layout (gate output rows first),
    /// built once after checkpoint load for the shared expert when
    /// `DARKBLOOM_FUSED_SHARED_GATE_UP` is enabled. Plain stored properties
    /// with a leading underscore so Module reflection never treats the
    /// derived layout as checkpoint parameters; the quantized gate/up
    /// modules keep the original arrays for parameter integrity. Never set
    /// on the dense (BF16) layer-0 MLP.
    var _fusedGateUpWeight: MLXArray?
    var _fusedGateUpScales: MLXArray?
    var _fusedGateUpSplit: Int = 0

    /// Retained fused BF16 `[gate; up]` bank for the dense (non-quantized)
    /// layer-0 MLP, built once after checkpoint load when
    /// `DARKBLOOM_FUSED_DENSE_GATE_UP_SWIGLU` is enabled. Mutually exclusive
    /// with `_fusedGateUpWeight`/`_fusedGateUpScales` above: those guard on
    /// `QuantizedLinear` (the NVFP4 shared-expert instance of this class),
    /// this one guards on plain `Linear` (the dense layer-0 instance), and
    /// `gateProj`/`upProj` are always both-or-neither quantized, so at most
    /// one of the two banks is ever non-nil on a given instance.
    var _fusedDenseGateUpWeight: MLXArray?
    /// Scale-plane-halved shared down scales [2048, 16] uint8 and escape [1]
    /// uint8, exploiting the NVFP4 pairwise-constancy invariant.
    var _halvedSharedDownScales: MLXArray?
    var _sharedDownScalesEscape: MLXArray?
    /// Scale-plane-halved shared gate/up fused scales [1024, 64] uint8 and
    /// escape [2] (gate row 0, up row 0) uint8.
    var _halvedFusedGateUpScales: MLXArray?
    var _fusedGateUpScalesEscape: MLXArray?
    /// Precomputed full group-32 scale arrays for the prefill halved path,
    /// built once in `prepareFusedSharedGateUp` to eliminate per-call
    /// reconstruction dispatches.
    var _prefillGateUpFullScales: MLXArray?
    var _prefillDownFullScales: MLXArray?

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    /// Builds and retains the fused gate/up NVFP4 bank from the loaded
    /// shared-expert projections. Called once after weights are installed
    /// and evaluated (before warmup); returns the new arrays so the caller
    /// can batch a single eval. Fuses only the exact stock shared-expert
    /// configuration: two bias-free NVFP4 group-16 4-bit `QuantizedLinear`
    /// projections with identical packed shapes and no affine biases.
    func prepareFusedSharedGateUp() -> [MLXArray] {
        guard _fusedGateUpWeight == nil, _fusedGateUpScales == nil,
            let gate = gateProj as? QuantizedLinear,
            let up = upProj as? QuantizedLinear,
            type(of: gate) == QuantizedLinear.self,
            type(of: up) == QuantizedLinear.self,
            gate.mode == .nvfp4, up.mode == .nvfp4,
            gate.groupSize == 16, up.groupSize == 16,
            gate.bits == 4, up.bits == 4,
            gate.bias == nil, up.bias == nil,
            gate.biases == nil, up.biases == nil,
            gate.weight.ndim == 2, up.weight.ndim == 2,
            gate.weight.dtype == .uint32, up.weight.dtype == .uint32,
            gate.scales.ndim == 2, up.scales.ndim == 2,
            gate.scales.dtype == .uint8, up.scales.dtype == .uint8,
            gate.weight.sameDims(up.weight),
            gate.scales.sameDims(up.scales),
            gate.scales.dim(0) == gate.weight.dim(0),
            gate.weight.dim(1) * 8 == gate.scales.dim(1) * 16
        else {
            return []
        }
        let fusedWeight = concatenated([gate.weight, up.weight], axis: 0)
        let fusedScales = concatenated([gate.scales, up.scales], axis: 0)
        _fusedGateUpWeight = fusedWeight
        _fusedGateUpScales = fusedScales
        _fusedGateUpSplit = gate.weight.dim(0)
        // Halve fused gate/up scales: [1024, 128] -> [1024, 64] + escape [2].
        // Gate exception at row 0 byte 1, up exception at row 512 byte 1.
        let fuseRows = fusedScales.dim(0)
        let fuseGroups = fusedScales.dim(1)
        let halfFuseGroups = fuseGroups / 2
        let fuseReshaped = fusedScales.reshaped(
            [fuseRows, halfFuseGroups, 2])
        let halvedFuse = contiguous(
            take(fuseReshaped, MLXArray([0]), axis: 2)
                .squeezed(axis: 2))
        let gateEsc = fusedScales[0, 1]
        let upEsc = fusedScales[fuseRows / 2, 1]
        let fuseEscape = contiguous(
            stacked([gateEsc, upEsc]).reshaped([2]))
        _halvedFusedGateUpScales = halvedFuse
        _fusedGateUpScalesEscape = fuseEscape
        let hg = halvedFuse.dim(1)
        _prefillGateUpFullScales = contiguous(concatenated([halvedFuse,
            concatenated([fuseEscape.reshaped([1, 2]),
                MLXArray.zeros([1, hg - 2], dtype: .uint8)], axis: 1)], axis: 0))
        // Halve shared down scales: [2048, 32] -> [2048, 16] + escape [1].
        if let down = downProj as? QuantizedLinear,
            type(of: down) == QuantizedLinear.self,
            down.mode == .nvfp4, down.groupSize == 16, down.bits == 4,
            down.bias == nil, down.biases == nil,
            down.scales.dtype == .uint8, down.scales.ndim == 2
        {
            let dsRows = down.scales.dim(0)
            let dsGroups = down.scales.dim(1)
            let halfDsGroups = dsGroups / 2
            let dsReshaped = down.scales.reshaped(
                [dsRows, halfDsGroups, 2])
            let halvedDs = contiguous(
                take(dsReshaped, MLXArray([0]), axis: 2)
                    .squeezed(axis: 2))
            let dsEscape = contiguous(down.scales[0, 1].reshaped([1]))
            _halvedSharedDownScales = halvedDs
            _sharedDownScalesEscape = dsEscape
            let r = halvedDs.dim(0)
            let nd = halvedDs.dim(1)
            let dsEsc = contiguous(concatenated([stacked([dsEscape[0], halvedDs[r / 2, 0]]).reshaped([1, 2]),
                MLXArray.zeros([1, nd - 2], dtype: .uint8)], axis: 1))
            _prefillDownFullScales = contiguous(concatenated([halvedDs, dsEsc], axis: 0))
            return [fusedWeight, fusedScales, halvedFuse, fuseEscape,
                    halvedDs, dsEscape, _prefillGateUpFullScales!,
                    _prefillDownFullScales!]
        }
        return [fusedWeight, fusedScales, halvedFuse, fuseEscape, _prefillGateUpFullScales!]
    }

    /// Builds and retains the fused BF16 gate/up bank from layer 0's dense
    /// (non-quantized) `gate_proj`/`up_proj`. Called once after weights are
    /// installed and evaluated (before warmup); returns the new array so the
    /// caller can batch a single eval. Fuses only the exact stock dense
    /// configuration: two bias-free plain `Linear` projections of identical
    /// shape and dtype. Never fires on the NVFP4 shared-expert instance of
    /// this class -- `type(of: gateProj) == Linear.self` is false there
    /// because `QuantizedLinear` is a distinct type, not this base type.
    func prepareFusedDenseGateUp() -> MLXArray? {
        let hidden = LagunaConstants.hiddenSize
        let intermediate = LagunaConstants.denseIntermediateSize
        guard _fusedDenseGateUpWeight == nil,
            type(of: gateProj) == Linear.self,
            type(of: upProj) == Linear.self,
            gateProj.bias == nil, upProj.bias == nil,
            gateProj.weight.dtype == .bfloat16,
            upProj.weight.dtype == .bfloat16,
            gateProj.weight.dims(intermediate, hidden),
            upProj.weight.dims(intermediate, hidden)
        else {
            return nil
        }
        let fusedWeight = concatenated([gateProj.weight, upProj.weight], axis: 0)
        _fusedDenseGateUpWeight = fusedWeight
        return fusedWeight
    }

    /// The shared expert's fused gate/up bank and its down bank, when every
    /// precondition of the fused decode path holds — without running the
    /// gate/up QMV, so a caller can batch that QMV with the routed one.
    func fusedSharedBanks(
        _ x: MLXArray
    ) -> (
        gateUpWeight: MLXArray,
        gateUpScales: MLXArray,
        downWeight: MLXArray,
        downScales: MLXArray
    )? {
        guard let banks = fusedSharedBankGuard(x) else { return nil }
        return banks
    }

    /// `sharedActivation` is the shared expert's gate/up result when the
    /// caller already issued it in this same invocation, batched into the
    /// routed gate/up dispatch. Passing it in just avoids issuing the
    /// identical QMV twice within one forward; nothing is retained across
    /// invocations.
    func fusedSharedDownInputs(
        _ x: MLXArray,
        sharedActivation: MLXArray? = nil
    ) -> (
        activated: MLXArray,
        downWeight: MLXArray,
        downScales: MLXArray
    )? {
        guard let banks = fusedSharedBankGuard(x),
            let halvedScales = _halvedFusedGateUpScales,
            let gateUpEscape = _fusedGateUpScalesEscape
        else { return nil }
        let activated =
            sharedActivation
            ?? lagunaSharedSwiGLUQMV(
                x,
                fusedWeight: banks.gateUpWeight,
                packedScales: halvedScales,
                gateUpEscape: gateUpEscape
            )
        return (activated, banks.downWeight, banks.downScales)
    }

    private func fusedSharedBankGuard(
        _ x: MLXArray
    ) -> (
        gateUpWeight: MLXArray,
        gateUpScales: MLXArray,
        downWeight: MLXArray,
        downScales: MLXArray
    )? {
        guard lagunaFusedSharedSwiGLUQMVEnabled,
            let fusedWeight = _fusedGateUpWeight,
            let fusedScales = _fusedGateUpScales,
            let down = downProj as? QuantizedLinear,
            type(of: down) == QuantizedLinear.self,
            down.mode == .nvfp4,
            down.groupSize == 16,
            down.bits == 4,
            down.bias == nil,
            down.biases == nil,
            x.dtype == .bfloat16,
            x.dims(1, 1, LagunaConstants.hiddenSize),
            fusedWeight.dtype == .uint32,
            fusedScales.dtype == .uint8,
            _fusedGateUpSplit == LagunaConstants.sharedExpertIntermediateSize,
            down.weight.dtype == .uint32,
            down.weight.dims(LagunaConstants.hiddenSize,
                LagunaConstants.sharedExpertIntermediateSize / 8),
            down.scales.dtype == .uint8,
            down.scales.dims(LagunaConstants.hiddenSize,
                LagunaConstants.sharedExpertIntermediateSize / 16)
        else {
            return nil
        }

        return (fusedWeight, fusedScales, down.weight, down.scales)
    }

    func fusedSharedDownResidual(
        _ x: MLXArray,
        routed: MLXArray,
        residual: MLXArray
    ) -> MLXArray? {
        guard lagunaFusedSharedDownResidualEnabled,
            let inputs = fusedSharedDownInputs(x),
            let halvedDownScales = _halvedSharedDownScales,
            let downScalesEscape = _sharedDownScalesEscape,
            routed.dtype == .bfloat16,
            routed.dims(1, 1, LagunaConstants.hiddenSize),
            residual.dtype == .bfloat16,
            residual.dims(1, 1, LagunaConstants.hiddenSize)
        else {
            return nil
        }

        lagunaTrace("shared down residual")
        return lagunaSharedDownResidual(
            inputs.activated,
            downWeight: inputs.downWeight,
            downScales: halvedDownScales,
            downScalesEscape: downScalesEscape,
            routed: routed,
            residual: residual
        )
    }

    /// Layer-0-only decode fusion: the dense gate/up GEMV + SiLU product and
    /// the down GEMV + decoder-layer residual add, each independently
    /// ablatable via its own `DARKBLOOM_FUSED_DENSE_*` flag. Every guard here
    /// mirrors the stock configuration exactly (bias-free plain `Linear`,
    /// BF16, the fixed layer-0 shapes), so when a flag is off, its retained
    /// bank was never built, or a guard declines, that half falls back to the
    /// exact stock op it replaces -- `compiledSiluProduct(gateProj(x),
    /// upProj(x))` for the gate/up half, `residual + downProj(activated)` for
    /// the down+residual half (the same elementwise BF16 add the decoder
    /// layer's stock `h + r2` performs). Returns `nil` only when the outer
    /// decode/layer-0/dense-BF16 guard itself declines, in which case the
    /// caller falls back to the fully stock `let r2 = mlp(normalized); return
    /// h + r2` path.
    func fusedDenseDownResidual(
        _ x: MLXArray, residual: MLXArray
    ) -> MLXArray? {
        let hidden = LagunaConstants.hiddenSize
        let intermediate = LagunaConstants.denseIntermediateSize
        guard x.dim(1) == 1,
            x.dtype == .bfloat16,
            x.dims(1, 1, hidden),
            residual.dtype == .bfloat16,
            residual.dims(1, 1, hidden),
            type(of: gateProj) == Linear.self,
            type(of: upProj) == Linear.self,
            type(of: downProj) == Linear.self,
            gateProj.bias == nil, upProj.bias == nil, downProj.bias == nil,
            gateProj.weight.dtype == .bfloat16,
            upProj.weight.dtype == .bfloat16,
            downProj.weight.dtype == .bfloat16,
            gateProj.weight.dims(intermediate, hidden),
            upProj.weight.dims(intermediate, hidden),
            downProj.weight.dims(hidden, intermediate)
        else {
            return nil
        }

        let activated: MLXArray
        if lagunaFusedDenseGateUpSwiGLUEnabled,
            let fusedWeight = _fusedDenseGateUpWeight,
            fusedWeight.dtype == .bfloat16,
            fusedWeight.dims(2 * intermediate, hidden)
        {
            lagunaTrace("dense gate/up GEMV + SwiGLU")
            activated = lagunaDenseGateUpSwiGLU(x, fusedWeight: fusedWeight)
        } else {
            activated = compiledSiluProduct(gateProj(x), upProj(x))
        }

        if lagunaFusedDenseDownResidualEnabled {
            lagunaTrace("dense down GEMV + residual")
            return lagunaDenseDownResidual(
                activated, downWeight: downProj.weight, residual: residual)
        }
        return residual + downProj(activated)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if x.dim(1) == 1,
            let fusedWeight = _fusedGateUpWeight, let fusedScales = _fusedGateUpScales
        {
            if lagunaFusedSharedSwiGLUQMVEnabled,
                let halvedScales = _halvedFusedGateUpScales,
                let gateUpEscape = _fusedGateUpScalesEscape,
                x.dtype == .bfloat16,
                x.dims(1, 1, LagunaConstants.hiddenSize),
                fusedWeight.dtype == .uint32,
                fusedScales.dtype == .uint8,
                _fusedGateUpSplit == LagunaConstants.sharedExpertIntermediateSize
            {
                lagunaTrace("shared gate/up QMV + SwiGLU")
                return downProj(
                    lagunaSharedSwiGLUQMV(
                        x,
                        fusedWeight: fusedWeight,
                        packedScales: halvedScales,
                        gateUpEscape: gateUpEscape
                    )
                )
            }

            // One NVFP4 dispatch over the row-concatenated [gate; up] bank,
            // mirroring `QuantizedLinear.callAsFunction` exactly (transpose,
            // group 16, 4-bit, .nvfp4, no affine biases, no bias add; the
            // guards in `prepareFusedSharedGateUp` pin those literals). Each
            // quantized output row is computed independently, so the split
            // halves are bit-exact vs. the separate gate/up dispatches.
            lagunaTrace("shared fused [gate; up] bank QMM")
            let gateUp = MLX.quantizedMM(
                x,
                fusedWeight,
                scales: fusedScales,
                biases: nil,
                transpose: true,
                groupSize: 16,
                bits: 4,
                mode: .nvfp4
            )
            let gate = gateUp[.ellipsis, 0 ..< _fusedGateUpSplit]
            let up = gateUp[.ellipsis, _fusedGateUpSplit...]
            return downProj(compiledSiluProduct(gate, up))
        }
        if x.dim(1) > 1,
            lagunaPrefillSharedHalvedEnabled,
            let fusedWeight = _fusedGateUpWeight, let fusedScales = _fusedGateUpScales,
            let cs = _prefillGateUpFullScales,
            let cds = _prefillDownFullScales,
            let down = downProj as? QuantizedLinear,
            x.dtype == .bfloat16,
            fusedWeight.dtype == .uint32,
            fusedScales.dtype == .uint8,
            _fusedGateUpSplit == LagunaConstants.sharedExpertIntermediateSize
        {
            lagunaTrace("shared fused [gate; up] bank QMM halved (prefill)")
            let gu = MLX.quantizedMM(x, fusedWeight, scales: cs, biases: nil, transpose: true, groupSize: 32, bits: 4, mode: .nvfp4)
            let g = gu[.ellipsis, 0 ..< _fusedGateUpSplit], u = gu[.ellipsis, _fusedGateUpSplit...]
            let act = compiledSiluProduct(g, u)
            return MLX.quantizedMM(act, down.weight, scales: cds, biases: nil, transpose: true, groupSize: 32, bits: down.bits, mode: down.mode)
        }
        return downProj(compiledSiluProduct(gateProj(x), upProj(x)))
    }
}

// MARK: - MoE

/// Decode-only router post-processing. The stock path materializes sigmoid
/// scores, corrected choice scores, their negation, a full 256-entry argsort,
/// and a gather before retaining just eight entries. This fixed-shape kernel
/// computes the same FP32 sigmoid values and stable choice order, then emits
/// only the selected indices and their scores.
///
/// `normalizing` additionally folds in the top-k renormalization, which the
/// stock path spends two more dependent dispatches on. That is reproducible
/// exactly: `weights.sum(axis: -1)` over a row of eight FP32 values takes
/// MLX's `row_reduce_small` path (`row_size <= 64` with a single non-row
/// reduction), whose `thread_reduce` walks the row in index order from
/// `Op::init == 0`, and the following divide is an elementwise FP32 IEEE
/// division -- MLX builds every runtime library, this kernel included, with
/// fast math disabled, so `scores[lane] / total` is the same division the
/// binary kernel would perform.
private func lagunaDecodeRouterTop8KernelSource(normalizing: Bool) -> String {
    let epilogue =
        normalizing
        ? """
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            total = simd_shuffle(my_score, ushort(i)) + total;
        }
        if (lane < 8) {
            router_indices[lane] = my_index;
            router_scores[lane] = my_score / total;
        }
        """
        : """
        if (lane < 8) {
            router_indices[lane] = my_index;
            router_scores[lane] = my_score;
        }
        """
    return """
        uint lane = thread_position_in_threadgroup.x;

        threadgroup float xchg_keys[256];
        threadgroup uint xchg_indices[256];
        threadgroup float xchg_scores[256];

        float x = float(logits[lane]);
        float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
        float my_score = x < 0.0f ? y : 1.0f - y;
        float my_key = -(my_score + float(correction_bias[lane]));
        uint my_index = lane;

        // A total order (choice key, then original expert index) makes this
        // network match the stock stable merge sort even for exact ties,
        // signed zero, and NaNs. The lower half of each final sequence keeps
        // the better entries, so ranks 0..<8 are the desired top experts.
        //
        // The network's schedule, comparator, and pair roles are unchanged
        // from the threadgroup-memory version; only WHERE a pair exchanges
        // its operands differs. For stride < 32, `partner = lane ^ stride`
        // never leaves the calling simdgroup (only bits 0-4 flip), so those
        // 30 stages exchange through registers with `simd_shuffle_xor` --
        // the same value-passing idiom the promoted QK-norm kernels use --
        // touching no memory and needing no barrier. Shuffles are
        // bit-preserving, both partners compute the identical swap decision
        // from identical operands (`lane & sequence` agrees across a pair
        // because stride < sequence), and each keeps its side of the
        // exchange, so every stage's resulting values are bit-identical to
        // the memory version's. Only the six stages with stride >= 32 cross
        // a simdgroup boundary and go through threadgroup memory with full
        // barriers.
        for (uint sequence = 2; sequence <= 256; sequence <<= 1) {
            for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
                float other_key;
                uint other_index;
                float other_score;
                if (stride < 32) {
                    other_key = simd_shuffle_xor(my_key, ushort(stride));
                    other_index = simd_shuffle_xor(my_index, ushort(stride));
                    other_score = simd_shuffle_xor(my_score, ushort(stride));
                } else {
                    xchg_keys[lane] = my_key;
                    xchg_indices[lane] = my_index;
                    xchg_scores[lane] = my_score;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    uint partner = lane ^ stride;
                    other_key = xchg_keys[partner];
                    other_index = xchg_indices[partner];
                    other_score = xchg_scores[partner];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                bool is_lower = (lane & stride) == 0;
                float a_key = is_lower ? my_key : other_key;
                uint a_index = is_lower ? my_index : other_index;
                float a_score = is_lower ? my_score : other_score;
                float b_key = is_lower ? other_key : my_key;
                uint b_index = is_lower ? other_index : my_index;
                float b_score = is_lower ? other_score : my_score;

                bool lower_wants_better = (lane & sequence) == 0;
                bool b_before_a = laguna_router_key_before(
                    b_key, b_index, a_key, a_index);
                bool a_before_b = laguna_router_key_before(
                    a_key, a_index, b_key, b_index);
                bool swap = lower_wants_better ? b_before_a : a_before_b;
                if (swap) {
                    my_key = is_lower ? b_key : a_key;
                    my_index = is_lower ? b_index : a_index;
                    my_score = is_lower ? b_score : a_score;
                }
            }
        }

        // Ranks 0..<8 live in lanes 0..<8 of simdgroup 0. The epilogue runs
        // unguarded so every shuffle source lane is active; only lanes < 8
        // write. The rank-order left fold reproduces the stock epilogue's
        // `total = scores[i] + total` operand order exactly.
        \(epilogue)
        """
}

private let lagunaDecodeRouterTop8Header = """
    METAL_FUNC bool laguna_router_key_before(
        float a, uint a_index, float b, uint b_index) {
        bool a_nan = metal::isnan(a);
        bool b_nan = metal::isnan(b);
        if (a_nan | b_nan) {
            if (a_nan != b_nan) {
                return !a_nan;
            }
            return a_index < b_index;
        }
        if (a < b) {
            return true;
        }
        if (b < a) {
            return false;
        }
        return a_index < b_index;
    }
    """

private let lagunaDecodeRouterTop8Kernel = MLXFast.metalKernel(
    name: "laguna_decode_router_top8_v3",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaDecodeRouterTop8KernelSource(normalizing: false),
    header: lagunaDecodeRouterTop8Header,
    ensureRowContiguous: true
)

private let lagunaDecodeRouterTop8NormalizingKernel = MLXFast.metalKernel(
    name: "laguna_decode_router_top8_norm_v2",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaDecodeRouterTop8KernelSource(normalizing: true),
    header: lagunaDecodeRouterTop8Header,
    ensureRowContiguous: true
)

/// Default-on decode-router payload optimization. Set
/// `DARKBLOOM_ROUTER_ORDINAL=0` for the accepted float-payload fallback. The
/// accepted bitonic
/// network carries `(float key, uint index, float score)` through all 36
/// compare/exchange stages. The ordinal arm preserves that network's exact
/// stage, shuffle, barrier, and pair-role geometry, but replaces the live
/// payload with `(uint ordinal, uint index)`. `laguna_router_key_ordinal`
/// canonicalizes both signed zeros and every NaN before applying the usual
/// monotone IEEE-754 bit transform, so unsigned ordinal comparison plus the
/// original expert-index tie break is exactly `laguna_router_key_before`.
/// Only final lanes 0...7 recompute their pre-bias sigmoid score.
private func lagunaDecodeRouterOrdinalKernelSource(
    normalizing: Bool, scoreTable: Bool = false
) -> String {
    let scoreStorage =
        scoreTable
        ? "threadgroup float original_scores[256];"
        : ""
    let scoreStore =
        scoreTable
        ? "original_scores[lane] = score;"
        : ""
    let winnerScore =
        scoreTable
        ? """
            my_score = original_scores[my_index];
        """
        : """
            float winner_x = float(logits[my_index]);
            float winner_y = 1.0f / (1.0f + metal::exp(metal::abs(winner_x)));
            my_score = winner_x < 0.0f ? winner_y : 1.0f - winner_y;
        """
    let epilogue =
        normalizing
        ? """
        float my_score = 0.0f;
        if (lane < 8) {
        \(winnerScore)
        }
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            total = simd_shuffle(my_score, ushort(i)) + total;
        }
        if (lane < 8) {
            router_indices[lane] = my_index;
            router_scores[lane] = my_score / total;
        }
        """
        : """
        if (lane < 8) {
            float my_score = 0.0f;
        \(winnerScore)
            router_indices[lane] = my_index;
            router_scores[lane] = my_score;
        }
        """
    return """
        uint lane = thread_position_in_threadgroup.x;

        threadgroup uint xchg_ordinals[256];
        threadgroup uint xchg_indices[256];
        \(scoreStorage)

        float x = float(logits[lane]);
        float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
        float score = x < 0.0f ? y : 1.0f - y;
        \(scoreStore)
        float key = -(score + float(correction_bias[lane]));
        uint my_ordinal = laguna_router_key_ordinal(key);
        uint my_index = lane;

        // Byte-for-byte the accepted 256-element Batcher schedule and pair
        // roles: 30 intra-simdgroup stages and six cross-simdgroup stages.
        // Only the exact sortable payload representation differs.
        for (uint sequence = 2; sequence <= 256; sequence <<= 1) {
            for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
                uint other_ordinal;
                uint other_index;
                if (stride < 32) {
                    other_ordinal = simd_shuffle_xor(my_ordinal, ushort(stride));
                    other_index = simd_shuffle_xor(my_index, ushort(stride));
                } else {
                    xchg_ordinals[lane] = my_ordinal;
                    xchg_indices[lane] = my_index;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    uint partner = lane ^ stride;
                    other_ordinal = xchg_ordinals[partner];
                    other_index = xchg_indices[partner];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                bool is_lower = (lane & stride) == 0;
                bool lower_wants_better = (lane & sequence) == 0;
                bool want_better = lower_wants_better == is_lower;
                bool other_before_my = laguna_router_ordinal_before(
                    other_ordinal, other_index, my_ordinal, my_index);
                // Expert indices are globally unique, so `my` and `other`
                // can never compare equal. This is the accepted a/b pair-role
                // rule reduced algebraically to a direct take-other decision.
                bool take_other = want_better ? other_before_my : !other_before_my;
                if (take_other) {
                    my_ordinal = other_ordinal;
                    my_index = other_index;
                }
            }
        }

        \(epilogue)
        """
}

private let lagunaDecodeRouterOrdinalHeader = """
    METAL_FUNC uint laguna_router_key_ordinal(float key) {
        uint bits = as_type<uint>(key);
        uint magnitude = bits & 0x7FFFFFFFu;
        if (magnitude > 0x7F800000u) {
            return 0xFFFFFFFFu;
        }
        // The accepted comparator considers -0 and +0 equal and breaks that
        // tie only by original expert index.
        if (magnitude == 0u) {
            return 0x80000000u;
        }
        return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
    }

    METAL_FUNC bool laguna_router_ordinal_before(
        uint a, uint a_index, uint b, uint b_index) {
        if (a < b) {
            return true;
        }
        if (b < a) {
            return false;
        }
        return a_index < b_index;
    }
    """

private let lagunaDecodeRouterOrdinalKernel = MLXFast.metalKernel(
    name: "laguna_decode_router_top8_ordinal_v1",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaDecodeRouterOrdinalKernelSource(normalizing: false),
    header: lagunaDecodeRouterOrdinalHeader,
    ensureRowContiguous: true
)

private let lagunaDecodeRouterOrdinalNormalizingKernel = MLXFast.metalKernel(
    name: "laguna_decode_router_top8_ordinal_norm_v1",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaDecodeRouterOrdinalKernelSource(normalizing: true),
    header: lagunaDecodeRouterOrdinalHeader,
    ensureRowContiguous: true
)

private let lagunaDecodeRouterOrdinalScoreTableKernel = MLXFast.metalKernel(
    name: "laguna_decode_router_top8_ordinal_table_v1",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaDecodeRouterOrdinalKernelSource(normalizing: false, scoreTable: true),
    header: lagunaDecodeRouterOrdinalHeader,
    ensureRowContiguous: true
)

private let lagunaDecodeRouterOrdinalScoreTableNormalizingKernel = MLXFast.metalKernel(
    name: "laguna_decode_router_top8_ordinal_table_norm_v1",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaDecodeRouterOrdinalKernelSource(normalizing: true, scoreTable: true),
    header: lagunaDecodeRouterOrdinalHeader,
    ensureRowContiguous: true
)

private let lagunaDecodeRouterOrdinalEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ROUTER_ORDINAL"] != "0"

/// The default ordinal arm preserves each original score once in TG memory;
/// set `DARKBLOOM_ROUTER_ORDINAL_SCORE_TABLE=0` to recompute only the final
/// eight sigmoid scores instead.
private let lagunaDecodeRouterOrdinalScoreTableEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ROUTER_ORDINAL_SCORE_TABLE"] != "0"

func lagunaDecodeRouterTop8AcceptedForTesting(
    logits: MLXArray, correctionBias: MLXArray, normalizing: Bool = false
) -> (MLXArray, MLXArray) {
    precondition(logits.dtype == .bfloat16 || logits.dtype == .float32)
    precondition(correctionBias.dtype == .float32)
    precondition(logits.size == 256)
    precondition(correctionBias.size == 256)

    let kernel =
        normalizing ? lagunaDecodeRouterTop8NormalizingKernel : lagunaDecodeRouterTop8Kernel
    let outputs = kernel(
        [logits, correctionBias],
        grid: (256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, 1, 8], [1, 1, 8]],
        outputDTypes: [.uint32, .float32]
    )
    return (outputs[0], outputs[1])
}

func lagunaDecodeRouterTop8OrdinalForTesting(
    logits: MLXArray, correctionBias: MLXArray, normalizing: Bool = false
) -> (MLXArray, MLXArray) {
    precondition(logits.dtype == .bfloat16 || logits.dtype == .float32)
    precondition(correctionBias.dtype == .float32)
    precondition(logits.size == 256)
    precondition(correctionBias.size == 256)

    let kernel =
        normalizing
        ? lagunaDecodeRouterOrdinalNormalizingKernel : lagunaDecodeRouterOrdinalKernel
    let outputs = kernel(
        [logits, correctionBias],
        grid: (256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, 1, 8], [1, 1, 8]],
        outputDTypes: [.uint32, .float32]
    )
    return (outputs[0], outputs[1])
}

func lagunaDecodeRouterTop8OrdinalScoreTableForTesting(
    logits: MLXArray, correctionBias: MLXArray, normalizing: Bool = false
) -> (MLXArray, MLXArray) {
    precondition(logits.dtype == .bfloat16 || logits.dtype == .float32)
    precondition(correctionBias.dtype == .float32)
    precondition(logits.size == 256)
    precondition(correctionBias.size == 256)

    let kernel =
        normalizing
        ? lagunaDecodeRouterOrdinalScoreTableNormalizingKernel
        : lagunaDecodeRouterOrdinalScoreTableKernel
    let outputs = kernel(
        [logits, correctionBias],
        grid: (256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, 1, 8], [1, 1, 8]],
        outputDTypes: [.uint32, .float32]
    )
    return (outputs[0], outputs[1])
}

private func lagunaDecodeRouterTop8(
    logits: MLXArray, correctionBias: MLXArray, normalizing: Bool = false
) -> (MLXArray, MLXArray) {
    if lagunaDecodeRouterOrdinalEnabled {
        if lagunaDecodeRouterOrdinalScoreTableEnabled {
            return lagunaDecodeRouterTop8OrdinalScoreTableForTesting(
                logits: logits,
                correctionBias: correctionBias,
                normalizing: normalizing
            )
        }
        return lagunaDecodeRouterTop8OrdinalForTesting(
            logits: logits, correctionBias: correctionBias, normalizing: normalizing)
    }
    return lagunaDecodeRouterTop8AcceptedForTesting(
        logits: logits,
        correctionBias: correctionBias,
        normalizing: normalizing
    )
}

/// Default-on after same-binary bitwise checks over smooth, tied, and extreme
/// rows plus a 39-stage compiled latency probe. Set
/// `DARKBLOOM_FUSED_ROUTER=0` for a stock-path ablation.
private let lagunaDecodeRouterTop8Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_ROUTER"] != "0"

/// Decode-only cast sinking for the fused router. The BF16 router GEMV result
/// is consumed directly and converted to FP32 by the top-8 kernel's first
/// instruction, removing an otherwise standalone 256-element cast dispatch.
private let lagunaDecodeRouterCastSinkEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_ROUTER_CAST"] != "0"

/// Decode-only top-k renormalization sinking, the companion to the cast sink
/// above: the eight selected scores are summed and divided inside the top-8
/// kernel, removing the standalone eight-element reduce and the broadcast
/// divide that followed it. Set `DARKBLOOM_FUSED_ROUTER_NORM=0` to ablate.
private let lagunaDecodeRouterNormSinkEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_ROUTER_NORM"] != "0"

/// Prefill counterpart of the fused decode router: one dispatch per sparse
/// layer replaces the stock multi-token routing chain (FP32 cast, sigmoid,
/// correction-bias add, negate, `argPartition`'s full 256-wide merge argsort,
/// the top-8 slice, `takeAlong`, and — when `norm_topk_prob` is set — the
/// row sum and broadcast divide).
///
/// DEFAULT OFF: submission `fe01af9` shipped this together with the prefill
/// MoE tail and ranked **-0.68%** against its own base (1.11254 vs 1.12019).
/// The per-lane predecessor-count selection is ~10x the ALU of the batched
/// merge sort it replaced, and at 512 rows the stock sort amortizes to a few
/// microseconds per layer — there was nothing to save, only kernel shape to
/// lose. Kept behind `DARKBLOOM_PREFILL_ROUTER_TOP8=1` because the
/// bit-exactness argument (Metal `ArgPartition` IS the stable merge argsort)
/// is verified and useful.
private let lagunaPrefillRouterTop8Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_ROUTER_TOP8"] == "1"

/// Prefill MoE tail fusion: the weighted expert-output combine, the fixed
/// 2.5 routed scale, the shared-expert add and the residual add collapse
/// into one elementwise kernel, so the `[1, L, 8, 2048]` expert bank is read
/// once instead of materializing three more `[1, L, 2048]`-sized
/// intermediates. Set `DARKBLOOM_PREFILL_MOE_TAIL=0` to restore the stock
/// ops.
private let lagunaPrefillMoETailEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_MOE_TAIL"] != "0"

/// Default-off probe: keep the routed down projection in expert-sorted order
/// and let the fused MoE tail gather each original `(token, slot)` row through
/// `gatherSort`'s already-computed inverse permutation. This removes
/// `scatterUnsort`'s full expert-bank copy without changing the eight-slot
/// weighted reduction order.
private let lagunaPrefillSortedMoETailEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_SORTED_MOE_TAIL"] != "0"

/// Batched top-8 selection for multi-token (prefill) routing.
///
/// Exactness against the stock chain it replaces, per row:
///  * The sigmoid is the same numerically-stable form the FP32 `sigmoid`
///    kernel computes after the standalone cast (`float(bfloat)` widening is
///    exact), already ranked-validated by the decode router kernel.
///  * The selection reproduces `argPartition(-scoresForChoice, kth: 7)`
///    exactly: on Metal `ArgPartition::eval_gpu` IS `gpu_merge_sort`
///    (sort.cpp routes it to the same stable merge argsort as `argSort`), so
///    the stock "partition" is a fully sorted row. `laguna_router_key_before`
///    is a strict total order (choice key, then original expert index, with
///    the sort's NaN placement), so counting predecessors gives every expert
///    a unique rank equal to its stable-argsort position; ranks 0..<8 emit in
///    rank order, which is byte-identical to the stock argsort slice.
///  * Mixture weights are the pre-bias sigmoid scores of the selected
///    experts, exactly `takeAlong(scores, inds)`.
///  * The normalizing epilogue reproduces `weights.sum(axis: -1)` (an
///    8-element `row_reduce_small` walked in index order from zero) and the
///    IEEE FP32 broadcast divide — the same two dispatches the decode norm
///    sink already replaces, one row at a time.
private func lagunaPrefillRouterTop8KernelSource(normalizing: Bool) -> String {
    let epilogue =
        normalizing
        ? """
                float total = 0.0f;
                for (uint i = 0; i < 8; ++i) {
                    total = selected_scores[i] + total;
                }
                router_scores[row * 8 + lane] = selected_scores[lane] / total;
        """
        : """
                router_scores[row * 8 + lane] = selected_scores[lane];
        """
    return """
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.y;

        threadgroup float choice_keys[256];
        threadgroup float selected_scores[8];

        float x = float(logits[row * 256 + lane]);
        float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
        float score = x < 0.0f ? y : 1.0f - y;
        float corrected = score + float(correction_bias[lane]);
        float my_key = -corrected;
        choice_keys[lane] = my_key;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stable-argsort rank by predecessor count under the strict total
        // order (key, then original index). Ranks are a permutation of
        // 0..255, so the eight winners land in distinct output slots in
        // exactly the stock argsort-slice order.
        uint rank = 0;
        for (uint j = 0; j < 256; ++j) {
            rank += laguna_router_key_before(
                choice_keys[j], j, my_key, lane) ? 1 : 0;
        }
        if (rank < 8) {
            router_indices[row * 8 + rank] = lane;
            selected_scores[rank] = score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < 8) {
        \(epilogue)
        }
        """
}

private let lagunaPrefillRouterTop8Kernel = MLXFast.metalKernel(
    name: "laguna_prefill_router_top8_v1",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaPrefillRouterTop8KernelSource(normalizing: false),
    header: lagunaDecodeRouterTop8Header,
    ensureRowContiguous: true
)

private let lagunaPrefillRouterTop8NormalizingKernel = MLXFast.metalKernel(
    name: "laguna_prefill_router_top8_norm_v1",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaPrefillRouterTop8KernelSource(normalizing: true),
    header: lagunaDecodeRouterTop8Header,
    ensureRowContiguous: true
)

private func lagunaPrefillRouterTop8(
    logits: MLXArray, correctionBias: MLXArray, rows: Int, normalizing: Bool
) -> (MLXArray, MLXArray) {
    precondition(logits.dtype == .bfloat16 || logits.dtype == .float32)
    precondition(correctionBias.dtype == .float32)
    precondition(logits.size == rows * 256)
    precondition(correctionBias.size == 256)

    let kernel =
        normalizing
        ? lagunaPrefillRouterTop8NormalizingKernel : lagunaPrefillRouterTop8Kernel
    let outputs = kernel(
        [logits, correctionBias],
        grid: (256, rows, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, rows, 8], [1, rows, 8]],
        outputDTypes: [.uint32, .float32]
    )
    return (outputs[0], outputs[1])
}

/// Two-phase tournament replacing O(256) predecessor count: (1) 8 independent 32-lane
/// bitonic sorts per simdgroup (same decode kernel network, no threadgroup memory);
/// (2) repack 64 candidates, bitonic-sort the union, take first 8. Same comparator and
/// total order as decode router. Local-top-8 provably contains global top-8. Normalizing
/// epilogue via `simd_shuffle` is bit-identical to stock `weights.sum(axis: -1)`.
private func lagunaPrefillRouterTournamentKernelSource(normalizing: Bool) -> String {
    let epilogue =
        normalizing
        ? """
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            total = simd_shuffle(my_score2, ushort(i)) + total;
        }
        if (lane < 8) {
            router_indices[row * 8 + lane] = my_index2;
            router_scores[row * 8 + lane] = my_score2 / total;
        }
        """
        : """
        if (lane < 8) {
            router_indices[row * 8 + lane] = my_index2;
            router_scores[row * 8 + lane] = my_score2;
        }
        """
    return """
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.y;

        threadgroup float xchg_keys[256];
        threadgroup uint xchg_indices[256];
        threadgroup float xchg_scores[256];
        threadgroup float candidate_keys[64];
        threadgroup uint candidate_indices[64];
        threadgroup float candidate_scores[64];

        float x = float(logits[row * 256 + lane]);
        float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
        float my_score = x < 0.0f ? y : 1.0f - y;
        float my_key = -(my_score + float(correction_bias[lane]));
        uint my_index = lane;

        // Phase 1: eight independent 32-lane bitonic sorts (one per
        // simdgroup), entirely via simd_shuffle_xor. Identical stage
        // structure and comparator calls to the promoted decode router's
        // low-stride stages; just not continued past sequence == 32.
        for (uint sequence = 2; sequence <= 32; sequence <<= 1) {
            for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
                float other_key = simd_shuffle_xor(my_key, ushort(stride));
                uint other_index = simd_shuffle_xor(my_index, ushort(stride));
                float other_score = simd_shuffle_xor(my_score, ushort(stride));

                bool is_lower = (lane & stride) == 0;
                float a_key = is_lower ? my_key : other_key;
                uint a_index = is_lower ? my_index : other_index;
                float a_score = is_lower ? my_score : other_score;
                float b_key = is_lower ? other_key : my_key;
                uint b_index = is_lower ? other_index : my_index;
                float b_score = is_lower ? other_score : my_score;

                bool lower_wants_better = (lane & sequence) == 0;
                bool b_before_a = laguna_router_key_before(
                    b_key, b_index, a_key, a_index);
                bool a_before_b = laguna_router_key_before(
                    a_key, a_index, b_key, b_index);
                bool swap = lower_wants_better ? b_before_a : a_before_b;
                if (swap) {
                    my_key = is_lower ? b_key : a_key;
                    my_index = is_lower ? b_index : a_index;
                    my_score = is_lower ? b_score : a_score;
                }
            }
        }

        // Each simdgroup's 32 lanes are now fully sorted by (key, index) --
        // but NOT all eight blocks ascending: this is stage `sequence ==
        // 32` of the standard Batcher network, whose direction test
        // `(lane & sequence) == 0` reads bit 5 of `lane` at that stage,
        // which is exactly the block-parity bit. Even-indexed blocks
        // (0, 2, 4, 6) sort ascending (within_block 0 = best); odd-indexed
        // blocks (1, 3, 5, 7) sort DESCENDING (within_block 31 = best) --
        // required so a continued network could merge each adjacent
        // ascending/descending pair into a bitonic sequence at sequence ==
        // 64, even though this network stops here instead of continuing.
        // Extract each block's true local top-8 in rank order (0 = best)
        // from whichever end that block actually sorted its best element
        // to; the local-top-8-contains-global-top-8 proof above depends
        // only on each block being internally sorted by the total order,
        // not on a particular direction.
        uint block = lane >> 5;
        uint within_block = lane & 31;
        bool block_ascending = (block & 1) == 0;
        uint rank_in_block = block_ascending ? within_block : (31 - within_block);
        bool is_local_top8 = block_ascending ? (within_block < 8) : (within_block >= 24);
        if (is_local_top8) {
            candidate_keys[block * 8 + rank_in_block] = my_key;
            candidate_indices[block * 8 + rank_in_block] = my_index;
            candidate_scores[block * 8 + rank_in_block] = my_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: bitonic-sort the 64-candidate union. Every thread
        // participates uniformly -- lanes 64-255 load a harmless wrapped
        // duplicate of the real 64 candidates (`lane & 63`) so every
        // thread in the threadgroup reaches the stride >= 32 barrier
        // below identically; only lanes < 8 are ever read.
        float my_key2 = candidate_keys[lane & 63];
        uint my_index2 = candidate_indices[lane & 63];
        float my_score2 = candidate_scores[lane & 63];
        for (uint sequence = 2; sequence <= 64; sequence <<= 1) {
            for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
                float other_key;
                uint other_index;
                float other_score;
                if (stride < 32) {
                    other_key = simd_shuffle_xor(my_key2, ushort(stride));
                    other_index = simd_shuffle_xor(my_index2, ushort(stride));
                    other_score = simd_shuffle_xor(my_score2, ushort(stride));
                } else {
                    xchg_keys[lane] = my_key2;
                    xchg_indices[lane] = my_index2;
                    xchg_scores[lane] = my_score2;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    uint partner = lane ^ stride;
                    other_key = xchg_keys[partner];
                    other_index = xchg_indices[partner];
                    other_score = xchg_scores[partner];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                bool is_lower = (lane & stride) == 0;
                float a_key = is_lower ? my_key2 : other_key;
                uint a_index = is_lower ? my_index2 : other_index;
                float a_score = is_lower ? my_score2 : other_score;
                float b_key = is_lower ? other_key : my_key2;
                uint b_index = is_lower ? other_index : my_index2;
                float b_score = is_lower ? other_score : my_score2;

                bool lower_wants_better = (lane & sequence) == 0;
                bool b_before_a = laguna_router_key_before(
                    b_key, b_index, a_key, a_index);
                bool a_before_b = laguna_router_key_before(
                    a_key, a_index, b_key, b_index);
                bool swap = lower_wants_better ? b_before_a : a_before_b;
                if (swap) {
                    my_key2 = is_lower ? b_key : a_key;
                    my_index2 = is_lower ? b_index : a_index;
                    my_score2 = is_lower ? b_score : a_score;
                }
            }
        }

        // Ranks 0..<8 of the 64-candidate merge are the row's global
        // top-8, in exactly the stock argsort-slice order.
        \(epilogue)
        """
}

/// Ordinal-payload mirror of the default-on prefill tournament, selected by
/// the same default-on `DARKBLOOM_ROUTER_ORDINAL` switch as decode.
/// Its two-phase schedule, extraction geometry, single inter-phase barrier,
/// wrapped 64-candidate duplicate lanes, and final rank order are identical
/// to the accepted kernel above. Only the sorting payload changes from
/// `(float key, uint index, float score)` to `(uint ordinal, uint index)`.
/// One per-row score table preserves the original sigmoid bytes for the
/// final eight indexed loads without carrying scores through either network.
private func lagunaPrefillRouterTournamentOrdinalKernelSource(normalizing: Bool) -> String {
    let epilogue =
        normalizing
        ? """
        float my_score2 = lane < 8 ? original_scores[my_index2] : 0.0f;
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            total = simd_shuffle(my_score2, ushort(i)) + total;
        }
        if (lane < 8) {
            router_indices[row * 8 + lane] = my_index2;
            router_scores[row * 8 + lane] = my_score2 / total;
        }
        """
        : """
        if (lane < 8) {
            router_indices[row * 8 + lane] = my_index2;
            router_scores[row * 8 + lane] = original_scores[my_index2];
        }
        """
    return """
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.y;

        threadgroup uint xchg_ordinals[256];
        threadgroup uint xchg_indices[256];
        threadgroup uint candidate_ordinals[64];
        threadgroup uint candidate_indices[64];
        threadgroup float original_scores[256];

        float x = float(logits[row * 256 + lane]);
        float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
        float score = x < 0.0f ? y : 1.0f - y;
        original_scores[lane] = score;
        float key = -(score + float(correction_bias[lane]));
        uint my_ordinal = laguna_router_key_ordinal(key);
        uint my_index = lane;

        // Phase 1: identical 15-stage local 32-lane Batcher sorts.
        for (uint sequence = 2; sequence <= 32; sequence <<= 1) {
            for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
                uint other_ordinal = simd_shuffle_xor(my_ordinal, ushort(stride));
                uint other_index = simd_shuffle_xor(my_index, ushort(stride));

                bool is_lower = (lane & stride) == 0;
                bool lower_wants_better = (lane & sequence) == 0;
                bool want_better = lower_wants_better == is_lower;
                bool other_before_my = laguna_router_ordinal_before(
                    other_ordinal, other_index, my_ordinal, my_index);
                bool take_other = want_better ? other_before_my : !other_before_my;
                if (take_other) {
                    my_ordinal = other_ordinal;
                    my_index = other_index;
                }
            }
        }

        // Identical direction-aware local-top-8 extraction.
        uint block = lane >> 5;
        uint within_block = lane & 31;
        bool block_ascending = (block & 1) == 0;
        uint rank_in_block = block_ascending ? within_block : (31 - within_block);
        bool is_local_top8 = block_ascending ? (within_block < 8) : (within_block >= 24);
        if (is_local_top8) {
            candidate_ordinals[block * 8 + rank_in_block] = my_ordinal;
            candidate_indices[block * 8 + rank_in_block] = my_index;
        }
        // This accepted inter-phase barrier also makes every lane's initial
        // original_scores store visible before the final indexed reads.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: identical wrapped 64-candidate, 21-stage Batcher sort.
        uint my_ordinal2 = candidate_ordinals[lane & 63];
        uint my_index2 = candidate_indices[lane & 63];
        for (uint sequence = 2; sequence <= 64; sequence <<= 1) {
            for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
                uint other_ordinal;
                uint other_index;
                if (stride < 32) {
                    other_ordinal = simd_shuffle_xor(my_ordinal2, ushort(stride));
                    other_index = simd_shuffle_xor(my_index2, ushort(stride));
                } else {
                    xchg_ordinals[lane] = my_ordinal2;
                    xchg_indices[lane] = my_index2;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    uint partner = lane ^ stride;
                    other_ordinal = xchg_ordinals[partner];
                    other_index = xchg_indices[partner];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                bool is_lower = (lane & stride) == 0;
                bool lower_wants_better = (lane & sequence) == 0;
                bool want_better = lower_wants_better == is_lower;
                bool other_before_my = laguna_router_ordinal_before(
                    other_ordinal, other_index, my_ordinal2, my_index2);
                bool take_other = want_better ? other_before_my : !other_before_my;
                if (take_other) {
                    my_ordinal2 = other_ordinal;
                    my_index2 = other_index;
                }
            }
        }

        \(epilogue)
        """
}

private let lagunaPrefillRouterTournamentKernel = MLXFast.metalKernel(
    name: "laguna_prefill_router_tournament_v1",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaPrefillRouterTournamentKernelSource(normalizing: false),
    header: lagunaDecodeRouterTop8Header,
    ensureRowContiguous: true
)

private let lagunaPrefillRouterTournamentNormalizingKernel = MLXFast.metalKernel(
    name: "laguna_prefill_router_tournament_norm_v1",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaPrefillRouterTournamentKernelSource(normalizing: true),
    header: lagunaDecodeRouterTop8Header,
    ensureRowContiguous: true
)

private let lagunaPrefillRouterTournamentOrdinalKernel = MLXFast.metalKernel(
    name: "laguna_prefill_router_tournament_ordinal_v1",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaPrefillRouterTournamentOrdinalKernelSource(normalizing: false),
    header: lagunaDecodeRouterOrdinalHeader,
    ensureRowContiguous: true
)

private let lagunaPrefillRouterTournamentOrdinalNormalizingKernel = MLXFast.metalKernel(
    name: "laguna_prefill_router_tournament_ordinal_norm_v1",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaPrefillRouterTournamentOrdinalKernelSource(normalizing: true),
    header: lagunaDecodeRouterOrdinalHeader,
    ensureRowContiguous: true
)

func lagunaPrefillRouterTournamentAcceptedForTesting(
    logits: MLXArray, correctionBias: MLXArray, rows: Int, normalizing: Bool
) -> (MLXArray, MLXArray) {
    precondition(logits.dtype == .bfloat16 || logits.dtype == .float32)
    precondition(correctionBias.dtype == .float32)
    precondition(logits.size == rows * 256)
    precondition(correctionBias.size == 256)

    let kernel =
        normalizing
        ? lagunaPrefillRouterTournamentNormalizingKernel : lagunaPrefillRouterTournamentKernel
    let outputs = kernel(
        [logits, correctionBias],
        grid: (256, rows, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, rows, 8], [1, rows, 8]],
        outputDTypes: [.uint32, .float32]
    )
    return (outputs[0], outputs[1])
}

func lagunaPrefillRouterTournamentOrdinalForTesting(
    logits: MLXArray, correctionBias: MLXArray, rows: Int, normalizing: Bool
) -> (MLXArray, MLXArray) {
    precondition(logits.dtype == .bfloat16 || logits.dtype == .float32)
    precondition(correctionBias.dtype == .float32)
    precondition(logits.size == rows * 256)
    precondition(correctionBias.size == 256)

    let kernel =
        normalizing
        ? lagunaPrefillRouterTournamentOrdinalNormalizingKernel
        : lagunaPrefillRouterTournamentOrdinalKernel
    let outputs = kernel(
        [logits, correctionBias],
        grid: (256, rows, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, rows, 8], [1, rows, 8]],
        outputDTypes: [.uint32, .float32]
    )
    return (outputs[0], outputs[1])
}

private func lagunaPrefillRouterTournament(
    logits: MLXArray, correctionBias: MLXArray, rows: Int, normalizing: Bool
) -> (MLXArray, MLXArray) {
    if lagunaDecodeRouterOrdinalEnabled {
        return lagunaPrefillRouterTournamentOrdinalForTesting(
            logits: logits,
            correctionBias: correctionBias,
            rows: rows,
            normalizing: normalizing
        )
    }
    return lagunaPrefillRouterTournamentAcceptedForTesting(
        logits: logits,
        correctionBias: correctionBias,
        rows: rows,
        normalizing: normalizing
    )
}

private let lagunaPrefillRouterTournamentEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_ROUTER_TOURNAMENT"] != "0"

/// Sigmoid top-k router. The routing math mirrors the vendored
/// `LagunaMoEGate` exactly (sigmoid scores, correction bias added only for
/// expert CHOICE, mixture weights taken from the pre-bias scores, optional
/// top-k renormalization).
final class LagunaRuntimeMoEGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let routerLogitSoftcapping: Float

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: LagunaConfig) {
        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.routerLogitSoftcapping = Float(config.moeRouterLogitSoftcapping)
        self._weight.wrappedValue = zeros([config.numExperts, config.hiddenSize])
        self._eScoreCorrectionBias.wrappedValue = zeros([config.numExperts])
    }

    /// `logits` is this layer's router projection when an upstream kernel in
    /// the same invocation already produced it (the fused residual + RMSNorm +
    /// router dispatch). It is the identical `x @ weight.T` this method would
    /// otherwise issue.
    func callAsFunction(_ x: MLXArray, logits: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let projectedLogits = logits ?? x.matmul(weight.T)
        let inds: MLXArray
        var weights: MLXArray
        if lagunaPrefillRouterTournamentEnabled,
            routerLogitSoftcapping == 0,
            topK == 8,
            projectedLogits.dtype == .bfloat16,
            projectedLogits.ndim == 3,
            projectedLogits.dim(0) == 1,
            projectedLogits.dim(1) > 1,
            projectedLogits.dim(2) == 256,
            eScoreCorrectionBias.size == 256
        {
            lagunaTrace("prefill router tournament")
            return lagunaPrefillRouterTournament(
                logits: projectedLogits,
                correctionBias: eScoreCorrectionBias.asType(.float32),
                rows: projectedLogits.dim(1),
                normalizing: normTopkProb
            )
        }
        if lagunaPrefillRouterTop8Enabled,
            routerLogitSoftcapping == 0,
            topK == 8,
            projectedLogits.dtype == .bfloat16,
            projectedLogits.ndim == 3,
            projectedLogits.dim(0) == 1,
            projectedLogits.dim(1) > 1,
            projectedLogits.dim(2) == 256,
            eScoreCorrectionBias.size == 256
        {
            lagunaTrace("prefill router top8")
            return lagunaPrefillRouterTop8(
                logits: projectedLogits,
                correctionBias: eScoreCorrectionBias.asType(.float32),
                rows: projectedLogits.dim(1),
                normalizing: normTopkProb
            )
        }
        if lagunaDecodeRouterTop8Enabled,
            lagunaDecodeRouterCastSinkEnabled,
            routerLogitSoftcapping == 0,
            projectedLogits.dtype == .bfloat16,
            projectedLogits.size == 256, topK == 8,
            eScoreCorrectionBias.size == 256
        {
            // Cast-sink path: consumes the BF16 router GEMV directly. The
            // norm sink is a separate flag, so name it separately.
            let sinkNormalization = normTopkProb && lagunaDecodeRouterNormSinkEnabled
            lagunaTrace(
                sinkNormalization
                    ? "decode router top8 (cast sink + norm sink)"
                    : "decode router top8 (cast sink)")
            (inds, weights) = lagunaDecodeRouterTop8(
                logits: projectedLogits,
                correctionBias: eScoreCorrectionBias.asType(.float32),
                normalizing: sinkNormalization
            )
            if sinkNormalization {
                return (inds, weights)
            }
        } else {
            var logits = projectedLogits.asType(.float32)
            if routerLogitSoftcapping > 0 {
                logits = tanh(logits / routerLogitSoftcapping) * routerLogitSoftcapping
            }
            if lagunaDecodeRouterTop8Enabled,
                logits.size == 256, topK == 8,
                eScoreCorrectionBias.size == 256
            {
                // Stock-cast path: FP32 logits, no cast sink.
                lagunaTrace("decode router top8 (fp32 logits)")
                (inds, weights) = lagunaDecodeRouterTop8(
                    logits: logits,
                    correctionBias: eScoreCorrectionBias.asType(.float32)
                )
            } else {
                let scores = sigmoid(logits)
                let scoresForChoice =
                    scores + eScoreCorrectionBias.asType(scores.dtype)
                inds =
                    argPartition(-scoresForChoice, kth: topK - 1, axis: -1)[
                        .ellipsis, ..<topK]
                weights = takeAlong(scores, inds, axis: -1)
            }
        }
        if normTopkProb {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
        return (inds, weights)
    }
}

/// Prefill MoE tail: weighted expert combine + routed scale + shared add +
/// residual add in one elementwise dispatch.
///
/// Exactness, op for op against the stock chain (`weightedExpertSum`, the
/// scalar multiply, and the two adds), whose arithmetic the promoted decode
/// down-reduce kernel already reproduces bit-exactly one row at a time:
///  * `weights.asType(y.dtype)` is the FP32→BF16 convert of each router
///    weight, done here per weight before any product.
///  * The multiply materializes `bfloat(y * w)` per element — the same
///    single-rounding BF16 product the compiled elementwise kernel writes.
///  * The `.sum(axis: -2)` over eight expert slots takes MLX's
///    `col_reduce_small` path (reduction size 8, stride 2048): each slot is
///    `op(value, init == 0)` and the combine walks slots in ascending order
///    with a BF16 accumulator, i.e. `total = bfloat(product + total)` from
///    zero in slot order. (`x + 0` is exact in BF16 except `-0`, which both
///    forms canonicalize identically.)
///  * The routed scale is `y * 2.5` with the scalar constructed in the BF16
///    result dtype (2.5 is exactly representable), one rounding.
///  * `r2 = scaled + shared` and `residual + r2` keep the stock operand
///    order and one BF16 rounding each.
private let lagunaPrefillMoETailKernel = MLXFast.metalKernel(
    name: "laguna_prefill_moe_tail_bf16_v1",
    inputNames: ["expert_outputs", "router_weights", "shared_output", "residual"],
    outputNames: ["output"],
    source: """
        constexpr uint hidden = 2048;
        constexpr uint experts = 8;
        constexpr uint n_cols = 4;

        uint row = thread_position_in_grid.y;
        uint col = thread_position_in_grid.x * n_cols;

        const device bfloat* expert_row =
            expert_outputs + (row * experts) * hidden + col;
        const device float* weight_row = router_weights + row * experts;

        bfloat expert_weights[experts];
        for (uint e = 0; e < experts; ++e) {
            expert_weights[e] = bfloat(weight_row[e]);
        }

        for (uint i = 0; i < n_cols; ++i) {
            bfloat total = bfloat(0);
            for (uint e = 0; e < experts; ++e) {
                bfloat product =
                    bfloat(expert_row[e * hidden + i] * expert_weights[e]);
                total = bfloat(product + total);
            }
            bfloat scaled = bfloat(total * bfloat(2.5f));
            bfloat r2 = bfloat(scaled + shared_output[row * hidden + col + i]);
            output[row * hidden + col + i] =
                bfloat(residual[row * hidden + col + i] + r2);
        }
        """,
    ensureRowContiguous: true
)

/// Sorted-input twin of `lagunaPrefillMoETailKernel`. `inverse_order[p]` is
/// the row in the expert-sorted down-projection output that
/// `scatterUnsort(...)[p]` would copy to original flattened slot `p`.
/// Reading that row directly preserves the stock slot-0-through-slot-7 BF16
/// multiply/add sequence while deleting the intervening 16 MiB copy at the
/// ranked 512-token window.
private let lagunaPrefillSortedMoETailKernel = MLXFast.metalKernel(
    name: "laguna_prefill_sorted_moe_tail_bf16_v1",
    inputNames: [
        "sorted_expert_outputs", "inverse_order", "router_weights",
        "shared_output", "residual",
    ],
    outputNames: ["output"],
    source: """
        constexpr uint hidden = 2048;
        constexpr uint experts = 8;
        constexpr uint n_cols = 4;

        uint row = thread_position_in_grid.y;
        uint col = thread_position_in_grid.x * n_cols;
        const device float* weight_row = router_weights + row * experts;

        bfloat expert_weights[experts];
        uint sorted_rows[experts];
        for (uint e = 0; e < experts; ++e) {
            expert_weights[e] = bfloat(weight_row[e]);
            sorted_rows[e] = inverse_order[row * experts + e];
        }

        for (uint i = 0; i < n_cols; ++i) {
            bfloat total = bfloat(0);
            for (uint e = 0; e < experts; ++e) {
                bfloat product = bfloat(
                    sorted_expert_outputs[sorted_rows[e] * hidden + col + i] *
                    expert_weights[e]);
                total = bfloat(product + total);
            }
            bfloat scaled = bfloat(total * bfloat(2.5f));
            bfloat r2 = bfloat(scaled + shared_output[row * hidden + col + i]);
            output[row * hidden + col + i] =
                bfloat(residual[row * hidden + col + i] + r2);
        }
        """,
    ensureRowContiguous: true
)

private func lagunaPrefillMoETail(
    expertOutputs: MLXArray,
    routerWeights: MLXArray,
    sharedOutput: MLXArray,
    residual: MLXArray
) -> MLXArray {
    let rows = expertOutputs.dim(1)
    precondition(expertOutputs.dtype == .bfloat16)
    precondition(
        expertOutputs.dims(1, rows, LagunaConstants.numExpertsPerTok, LagunaConstants.hiddenSize))
    precondition(routerWeights.dtype == .float32)
    precondition(routerWeights.dims(1, rows, LagunaConstants.numExpertsPerTok))
    precondition(sharedOutput.dtype == .bfloat16)
    precondition(sharedOutput.dims(1, rows, LagunaConstants.hiddenSize))
    precondition(residual.dtype == .bfloat16)
    precondition(residual.dims(1, rows, LagunaConstants.hiddenSize))

    return lagunaPrefillMoETailKernel(
        [expertOutputs, routerWeights, sharedOutput, residual],
        grid: (LagunaConstants.hiddenSize / 4, rows, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, rows, LagunaConstants.hiddenSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

private func lagunaPrefillSortedMoETail(
    sortedExpertOutputs: MLXArray,
    inverseOrder: MLXArray,
    routerWeights: MLXArray,
    sharedOutput: MLXArray,
    residual: MLXArray
) -> MLXArray {
    let rows = routerWeights.dim(1)
    precondition(sortedExpertOutputs.dtype == .bfloat16)
    precondition(
        sortedExpertOutputs.size
            == rows * LagunaConstants.numExpertsPerTok * LagunaConstants.hiddenSize)
    precondition(inverseOrder.dtype == .uint32)
    precondition(inverseOrder.size == rows * LagunaConstants.numExpertsPerTok)
    precondition(routerWeights.dtype == .float32)
    precondition(routerWeights.dims(1, rows, LagunaConstants.numExpertsPerTok))
    precondition(sharedOutput.dtype == .bfloat16)
    precondition(sharedOutput.dims(1, rows, LagunaConstants.hiddenSize))
    precondition(residual.dtype == .bfloat16)
    precondition(residual.dims(1, rows, LagunaConstants.hiddenSize))

    return lagunaPrefillSortedMoETailKernel(
        [
            sortedExpertOutputs, inverseOrder, routerWeights, sharedOutput,
            residual,
        ],
        grid: (LagunaConstants.hiddenSize / 4, rows, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, rows, LagunaConstants.hiddenSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

/// Reconstructs the stock SwiGLU result from the retained bank's physical
/// `[gate32, up32]` tile order. This is the correctness-preserving fallback
/// when the expert-aligned backend is disabled.
private func lagunaInterleavedSwiGLU(
    _ gateUp: MLXArray,
    split: Int
) -> MLXArray {
    precondition(split % 32 == 0)
    precondition(gateUp.dim(-1) == 2 * split)

    var tiledShape = gateUp.shape
    tiledShape.removeLast()
    tiledShape.append(split / 32)
    tiledShape.append(64)
    let tiled = gateUp.reshaped(tiledShape)

    var halfShape = gateUp.shape
    halfShape[halfShape.count - 1] = split
    let gate = tiled[.ellipsis, 0 ..< 32].reshaped(halfShape)
    let up = tiled[.ellipsis, 32 ..< 64].reshaped(halfShape)
    return compiledSiluProduct(gate, up)
}

/// Prefill (multi-token, SORTED-regime) counterpart to the decode-only fused
/// gate/up dispatch in `LagunaRuntimeSparseMoEBlock.forward`. One gather-QMM
/// consumes the retained `[gate32, up32]`-interleaved NVFP4 bank in place of
/// `SwitchGLU`'s separate `gate_proj` and `up_proj` calls. On the ranked
/// expert-aligned path the backend also applies the same rounded-BF16 SiLU
/// product and packs the 512-wide activation into the first half of the
/// nominal 1024-wide output allocation, avoiding that intermediate's device
/// round trip. `down_proj`, sorting, and unsorting remain the stock calls.
private func lagunaFusedSortedRoutedGateUp(
    _ x: MLXArray,
    indices: MLXArray,
    fusedWeight: MLXArray,
    fusedScales: MLXArray,
    halvedScales: MLXArray?,
    scalesEscape: MLXArray?,
    split: Int,
    downProj: SwitchLinear,
    downWeight: MLXArray?,
    halvedDownScales: MLXArray?,
    downScalesEscape: MLXArray?,
    deferUnsort: Bool
) -> (output: MLXArray, inverseOrder: MLXArray?) {
    // SwitchGLU: `var x = MLX.expandedDimensions(x, axes: [-2, -3])`
    var sortedX = MLX.expandedDimensions(x, axes: [-2, -3])
    // SwitchGLU: `let doSort = indices.size >= 64`; call site already guards this.
    let doSort = indices.size >= 64
    // SwitchGLU: `var idx = indices` / `var inverseOrder = MLXArray()`
    var idx = indices
    var inverseOrder = MLXArray()
    // SwitchGLU: `if doSort { (x, idx, inverseOrder) = gatherSort(x: x, indices: indices) }`
    //
    if doSort {
        (sortedX, idx, inverseOrder) = gatherSort(x: sortedX, indices: indices)
    }
    // Fused counterpart of SwitchGLU's separate-bank branch:
    //   xUp = upProj(x, idx, sortedIndices: doSort)
    //   xGate = gateProj(x, idx, sortedIndices: doSort)
    // Each of those is exactly `QuantizedSwitchLinear.callAsFunction` with
    // `biases: nil` (both banks are bias-free per the `prepareFusedRoutedGateUp`
    // guard): `MLX.gatherQuantizedMM(x, weight, scales: scales, biases: nil,
    // rhsIndices: indices, transpose: true, groupSize: groupSize, bits: bits,
    // mode: mode, sortedIndices: sortedIndices)`. Issuing that once over the
    // tile-interleaved `fusedWeight`/`fusedScales` bank instead of twice over
    // the separate banks is the fusion; every other argument matches the
    // stock call exactly (group 16, 4-bit, NVFP4, transpose, doSort).
    let gateUp: MLXArray
    // Halved scales: embed escape bytes in the scales tensor's extra row (like
    // PR #243's shared expert qmm_nax path), pass biases: nil with groupSize: 32
    // so ops.cpp validation accepts half-resolution scales. quantized.cpp
    // detects group_size==32, overrides to 16, and extracts the escape buffer.
    let useHalved = lagunaExpertAlignedGatherEnabled && halvedScales != nil
    gateUp = MLX.gatherQuantizedMM(
        sortedX, fusedWeight,
        scales: useHalved ? halvedScales! : fusedScales,
        biases: nil,
        rhsIndices: idx, transpose: true,
        groupSize: useHalved ? 32 : 16,
        bits: 4, mode: .nvfp4, sortedIndices: doSort)
    let activated: MLXArray
    if lagunaExpertAlignedGatherEnabled {
        // The expert kernel writes rows with a physical stride of `split`
        // into the allocation's contiguous prefix. Slice that prefix before
        // restoring the logical shape expected by down_proj.
        var activatedShape = gateUp.shape
        activatedShape[activatedShape.count - 1] = split
        activated = gateUp.reshaped([-1])[0 ..< gateUp.size / 2]
            .reshaped(activatedShape)
    } else {
        activated = lagunaInterleavedSwiGLU(gateUp, split: split)
    }
    // SwitchGLU: `x = downProj(activated, idx, sortedIndices: doSort)`.
    // Down halved remains disabled: the expert kernel reads escape at a
    // fixed stride of 2 (gate+up layout), incompatible with the down path's
    // 1-byte-per-expert escape.
    let useHalvedDown = false
    var result: MLXArray
    if useHalvedDown {
        result = MLX.gatherQuantizedMM(
            activated, downWeight!,
            scales: halvedDownScales!, biases: nil,
            rhsIndices: idx, transpose: true, groupSize: 16,
            bits: 4, mode: .nvfp4, sortedIndices: doSort)
    } else {
        result = downProj(activated, idx, sortedIndices: doSort)
    }
    // SwitchGLU: `if doSort { x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape) }`
    if doSort && !deferUnsort {
        result = scatterUnsort(x: result, invOrder: inverseOrder, shape: indices.shape)
    }
    if doSort && deferUnsort {
        return (result, inverseOrder)
    }
    // SwitchGLU: `return MLX.squeezed(x, axis: -2)`
    return (MLX.squeezed(result, axis: -2), nil)
}

final class LagunaRuntimeSparseMoEBlock: Module, UnaryLayer {
    let routedScalingFactor: Float

    @ModuleInfo(key: "gate") var gate: LagunaRuntimeMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: LagunaRuntimeMLP

    /// Retained fused NVFP4 `[gate32, up32]` routed-expert banks (per-expert
    /// output rows interleaved in matched 32-row tiles), built once after
    /// checkpoint load when `DARKBLOOM_FUSED_ROUTED_GATE_UP` is enabled, plus
    /// a reference to the stock `switch_mlp.down_proj` module for the fused
    /// decode path. Plain stored properties with a leading underscore so
    /// Module reflection never treats the derived layout as checkpoint
    /// parameters or a second child module; `switchMLP` keeps the original
    /// separate banks for checkpoint parameter integrity.
    var _fusedRoutedGateUpWeight: MLXArray?
    var _fusedRoutedGateUpScales: MLXArray?
    var _fusedRoutedGateUpSplit: Int = 0
    var _routedDownProj: SwitchLinear?
    var _routedDownWeight: MLXArray?
    var _routedDownScales: MLXArray?
    /// `DARKBLOOM_PACKED_SCALES` walk-order scale-interleaved copy of the
    /// fused routed gate/up scales ([experts, 4096, 32] uint8); see
    /// `lagunaRoutedSwiGLUQMVPackedKernel` for the layout contract. Nil
    /// when the flag is set to zero (default ON).
    var _packedRoutedGateUpBank: MLXArray?
    /// Scale-plane-halved routed down scales [experts, 2048, 16] uint8 and
    /// per-expert escape bytes [experts] uint8, exploiting the NVFP4
    /// pairwise-constancy invariant (scale[2k]==scale[2k+1] for k>=1).
    var _halvedRoutedDownScales: MLXArray?
    var _routedDownScalesEscape: MLXArray?
    /// Scale-plane-halved packed routed gate/up bank [experts, 4096, 16] and
    /// per-expert escape [experts, 2] (gate, up) uint8.
    var _halvedPackedRoutedGateUpBank: MLXArray?
    var _packedRoutedGateUpEscape: MLXArray?
    /// Halved fused gate/up scales [experts, 2*split, K/32] + escape [experts, 2].
    var _halvedFusedRoutedGateUpScales: MLXArray?
    var _fusedRoutedGateUpScalesEscape: MLXArray?

    /// Builds and retains the fused routed gate/up NVFP4 banks from the
    /// loaded stock `SwitchGLU` submodules (reached through the public
    /// `children()`/`parameters()` Module APIs). Called once after weights
    /// are installed and evaluated (before warmup); returns the new arrays
    /// so the caller can batch a single eval. Fuses only the exact stock
    /// configuration: two bias-free NVFP4 group-16 4-bit
    /// `QuantizedSwitchLinear` banks with identical packed shapes.
    func prepareFusedRoutedGateUp() -> [MLXArray] {
        guard _fusedRoutedGateUpWeight == nil, _fusedRoutedGateUpScales == nil else {
            return []
        }
        let children = Dictionary(uniqueKeysWithValues: switchMLP.children().flattened())
        guard let gateModule = children["gate_proj"] as? QuantizedSwitchLinear,
            let upModule = children["up_proj"] as? QuantizedSwitchLinear,
            let downModule = children["down_proj"] as? QuantizedSwitchLinear,
            type(of: gateModule) == QuantizedSwitchLinear.self,
            type(of: upModule) == QuantizedSwitchLinear.self,
            type(of: downModule) == QuantizedSwitchLinear.self,
            gateModule.mode == .nvfp4, upModule.mode == .nvfp4,
            downModule.mode == .nvfp4,
            gateModule.groupSize == 16, upModule.groupSize == 16,
            downModule.groupSize == 16,
            gateModule.bits == 4, upModule.bits == 4, downModule.bits == 4
        else {
            return []
        }
        let gateParams = Dictionary(uniqueKeysWithValues: gateModule.parameters().flattened())
        let upParams = Dictionary(uniqueKeysWithValues: upModule.parameters().flattened())
        let downParams = Dictionary(uniqueKeysWithValues: downModule.parameters().flattened())
        guard let gateWeight = gateParams["weight"], let gateScales = gateParams["scales"],
            let upWeight = upParams["weight"], let upScales = upParams["scales"],
            let downWeight = downParams["weight"],
            let downScales = downParams["scales"],
            gateParams["bias"] == nil, gateParams["biases"] == nil,
            upParams["bias"] == nil, upParams["biases"] == nil,
            downParams["bias"] == nil, downParams["biases"] == nil,
            gateWeight.ndim == 3, upWeight.ndim == 3,
            gateScales.ndim == 3, upScales.ndim == 3,
            downWeight.ndim == 3, downScales.ndim == 3,
            gateWeight.dtype == .uint32, upWeight.dtype == .uint32,
            gateScales.dtype == .uint8, upScales.dtype == .uint8,
            downWeight.dtype == .uint32, downScales.dtype == .uint8,
            gateWeight.sameDims(upWeight),
            gateScales.sameDims(upScales),
            gateScales.dim(0) == gateWeight.dim(0),
            gateScales.dim(1) == gateWeight.dim(1),
            gateWeight.dim(2) * 8 == gateScales.dim(2) * 16,
            downWeight.dim(0) == gateWeight.dim(0),
            downScales.dim(0) == downWeight.dim(0),
            downScales.dim(1) == downWeight.dim(1),
            downWeight.dim(2) * 8 == downScales.dim(2) * 16
        else {
            return []
        }
        // Interleave one 32-row gate tile with its matching 32-row up tile.
        // The expert-aligned prefill kernel's two WN simdgroups then own a
        // matched pair inside one 64-column threadgroup and can exchange the
        // rounded BF16 results through its existing weight scratch.
        let experts = gateWeight.dim(0)
        let split = gateWeight.dim(1)
        let pairRows = 32
        let weightDepth = gateWeight.dim(2)
        let scaleDepth = gateScales.dim(2)
        let gateWeightTiles = gateWeight.reshaped(
            [experts, split / pairRows, pairRows, weightDepth])
        let upWeightTiles = upWeight.reshaped(
            [experts, split / pairRows, pairRows, weightDepth])
        let gateScaleTiles = gateScales.reshaped(
            [experts, split / pairRows, pairRows, scaleDepth])
        let upScaleTiles = upScales.reshaped(
            [experts, split / pairRows, pairRows, scaleDepth])
        let fusedWeight = concatenated(
            [gateWeightTiles, upWeightTiles], axis: 2
        ).reshaped([experts, 2 * split, weightDepth])
        let fusedScales = concatenated(
            [gateScaleTiles, upScaleTiles], axis: 2
        ).reshaped([experts, 2 * split, scaleDepth])
        _fusedRoutedGateUpWeight = fusedWeight
        _fusedRoutedGateUpScales = fusedScales
        _fusedRoutedGateUpSplit = split
        _routedDownProj = downModule
        _routedDownWeight = downWeight
        _routedDownScales = downScales
        let halvedFuse = contiguous(
            fusedScales[0..., 0..., .stride(from: 0, by: 2)])
        let gateEsc = fusedScales[0..., 0, 1].reshaped([experts, 1])
        let upEsc = fusedScales[0..., 32, 1].reshaped([experts, 1])
        _fusedRoutedGateUpScalesEscape = contiguous(stacked([
            gateEsc, upEsc], axis: 1))
        let hg = halvedFuse.dim(2)
        let escapeRow = concatenated([
            concatenated([gateEsc, upEsc], axis: 1).reshaped([experts, 1, 2]),
            MLXArray.zeros([experts, 1, hg - 2], dtype: .uint8)], axis: 2)
        _halvedFusedRoutedGateUpScales = contiguous(
            concatenated([halvedFuse, escapeRow], axis: 1))
        var prepared = [fusedWeight, fusedScales,
            _halvedFusedRoutedGateUpScales!, _fusedRoutedGateUpScalesEscape!]
        // Halve routed down scales: [experts, 2048, 32] -> [experts, 2048, 16]
        // + escape [experts, 1]. The NVFP4 quantizer pairwise-constancy
        // invariant guarantees scale[2k]==scale[2k+1] for k>=1 in each
        // flattened weight matrix; only k=0 (row 0, bytes 0-1) can differ.
        let downScaleRows = downScales.dim(1)
        let downScaleGroups = downScales.dim(2)
        let halfDownGroups = downScaleGroups / 2
        let downReshaped = downScales.reshaped(
            [experts, downScaleRows, halfDownGroups, 2])
        let halvedDown = contiguous(
            take(downReshaped, MLXArray([0]), axis: 3)
                .squeezed(axis: 3))
        let downEscape = contiguous(
            downScales[0..., 0, 1].reshaped([experts, 1]))
        _halvedRoutedDownScales = halvedDown
        _routedDownScalesEscape = downEscape
        prepared.append(
            contentsOf: preparePackedRoutedGateUpBank(
                fusedScales: fusedScales,
                experts: experts,
                split: split))
        prepared.append(contentsOf: [halvedDown, downEscape])
        return prepared
    }

    /// Builds the `DARKBLOOM_PACKED_SCALES` side bank from the (lazy) fused
    /// routed gate/up arrays: bytes are only reordered, never recomputed.
    /// Per expert the packed layout is `[tile 128][k-block 4][sub 8][32 B]`
    /// with `sub = (simd_group*2 + row)*2 + {0 gate, 1 up}`. The row remap
    /// below (gateRow = (logical/32)*64 + logical%32, up = +32) is the stock
    /// kernel's mapping over the 32-row gate/up-interleaved fused bank, baked
    /// into scale storage order. The code bytes remain in the resident fused
    /// weight bank, so this side copy is ~32 MB per sparse layer instead of
    /// duplicating the ~256 MB code bank.
    func preparePackedRoutedGateUpBank(
        fusedScales: MLXArray,
        experts: Int,
        split: Int
    ) -> [MLXArray] {
        guard lagunaPackedScalesEnabled else { return [] }
        guard split == LagunaConstants.moeIntermediateSize,
            experts == LagunaConstants.numExperts,
            LagunaConstants.hiddenSize == 2048
        else {
            lagunaPackedScalesLog.note(
                "inactive", "packed routed gate/up bank (geometry guard declined)")
            return []
        }
        let rows = 2 * split  // 1024 fused (gate/up-interleaved) rows
        let rowBlocks = fusedScales.reshaped([experts, rows * 4, 32])
        // Walk-order gather over scale row-blocks: packed position (tile,
        // kblock, sub) reads fused scale row-block (fusedRow, kblock).
        var order = [Int32]()
        order.reserveCapacity(rows * 4)
        for tile in 0..<(rows / 8) {
            for kblock in 0..<4 {
                for sub in 0..<8 {
                    let logicalRow = tile * 4 + sub / 2
                    let gateRow = (logicalRow / 32) * 64 + logicalRow % 32
                    let fusedRow = sub % 2 == 0 ? gateRow : gateRow + 32
                    order.append(Int32(fusedRow * 4 + kblock))
                }
            }
        }
        // `take(axis: 1)` materializes with permuted strides (NOT
        // row-contiguous), and the custom kernel's `ensureRowContiguous`
        // would then re-copy the side bank on EVERY dispatch. Force the
        // one-time row-contiguous materialization here, at init, so dispatches
        // bind the bank buffer directly.
        let packed = contiguous(take(rowBlocks, MLXArray(order), axis: 1))
        _packedRoutedGateUpBank = packed
        // Halve the packed bank: [experts, 4096, 32] -> [experts, 4096, 16].
        // Within each 32-byte k-block row, byte 2j==byte 2j+1 for j>=1 by the
        // NVFP4 pairwise-constancy invariant. Escape: gate row 0 byte 1 at
        // packed[expert, 0, 1] and up row 0 byte 1 at packed[expert, 128, 1]
        // (walk-order indices 0 and 128 map to gate/up row 0 k-block 0).
        let halfGroups = 16
        let packedReshaped = packed.reshaped(
            [experts, rows * 4, halfGroups, 2])
        let halvedPacked = contiguous(
            take(packedReshaped, MLXArray([0]), axis: 3)
                .squeezed(axis: 3))
        let gateEscape = packed[0..., 0, 1]
        let upEscape = packed[0..., 128, 1]
        let gateUpEscape = contiguous(
            stacked([gateEscape, upEscape], axis: 1))
        _halvedPackedRoutedGateUpBank = halvedPacked
        _packedRoutedGateUpEscape = gateUpEscape
        lagunaPackedScalesLog.note("active", "packed routed gate/up bank prepared")
        return [packed, halvedPacked, gateUpEscape]
    }

    init(_ config: LagunaConfig) {
        self.routedScalingFactor = Float(config.moeRoutedScalingFactor)
        self._gate.wrappedValue = LagunaRuntimeMoEGate(config)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts
        )
        self._sharedExpert.wrappedValue = LagunaRuntimeMLP(
            dimensions: config.hiddenSize,
            hiddenDimensions: config.sharedExpertIntermediateSize
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        forward(x, residual: nil, routerLogits: nil)
    }

    func callAsFunction(
        _ x: MLXArray, residual: MLXArray, routerLogits: MLXArray? = nil
    ) -> MLXArray {
        forward(x, residual: residual, routerLogits: routerLogits)
    }

    private func forward(
        _ x: MLXArray, residual: MLXArray?, routerLogits: MLXArray?
    ) -> MLXArray {
        let (inds, weights) = gate(x, logits: routerLogits)
        var y: MLXArray
        var routedAlreadyReduced = false
        var sortedTailInverseOrder: MLXArray?
        if let fusedWeight = _fusedRoutedGateUpWeight,
            let fusedScales = _fusedRoutedGateUpScales,
            let downProj = _routedDownProj,
            x.dim(1) == 1, inds.size < 64
        {
            // DECODE-ONLY fused gate/up: replicate exactly SwitchGLU's
            // unsorted small-batch path (`indices.size < 64`, so no
            // gatherSort/scatterUnsort) with one gather-QMM over the
            // row-concatenated [gate; up] bank instead of two. The gather
            // call mirrors `QuantizedSwitchLinear.callAsFunction` (biases
            // nil, rhsIndices, transpose, group 16, 4-bit, .nvfp4,
            // sortedIndices false; the prepare guards pin those literals).
            // Each gathered output row is computed independently, so the
            // split halves (gate rows first) are bit-exact vs. the separate
            // banks; down_proj is the stock module invoked exactly as
            // SwitchGLU does. Multi-token forwards (prefill) below keep the
            // fully stock sorted gather-GEMM path and never see the fused
            // bank.
            let activated: MLXArray
            // Set when the routed and shared gate/up QMVs were issued as one
            // dispatch below, so the shared half of that same dispatch is
            // handed to the down projection instead of being issued again.
            // Purely within this invocation; nothing survives it.
            var mergedSharedActivated: MLXArray?
            if lagunaFusedRoutedSwiGLUQMVEnabled,
                x.dtype == .bfloat16,
                x.dims(1, 1, LagunaConstants.hiddenSize),
                fusedWeight.dtype == .uint32,
                fusedScales.dtype == .uint8,
                inds.dtype == .uint32,
                inds.dims(1, 1, LagunaConstants.numExpertsPerTok),
                _fusedRoutedGateUpSplit == LagunaConstants.moeIntermediateSize
            {
                if lagunaPackedScalesEnabled,
                    let packedBank = _packedRoutedGateUpBank
                {
                    lagunaPackedScalesLog.note(
                        "active", "routed swiglu qmv packed dispatch")
                    if gate.topK == LagunaConstants.numExpertsPerTok,
                        gate.routerLogitSoftcapping == 0,
                        gate.eScoreCorrectionBias.size == LagunaConstants.numExperts
                    {
                        if lagunaRoutedGateUpR1Enabled,
                            let halvedBank = _halvedPackedRoutedGateUpBank,
                            let gateUpEscape = _packedRoutedGateUpEscape
                        {
                            lagunaTrace("routed gate/up QMV + SwiGLU (packed, halved)")
                            let sharedW = sharedExpert._fusedGateUpWeight
                            let sharedS = sharedExpert._halvedFusedGateUpScales
                            let sharedE = sharedExpert._fusedGateUpScalesEscape
                            let merged = lagunaRoutedSwiGLUQMVPackedTop8(
                                x,
                                fusedWeight: fusedWeight,
                                packedScales: halvedBank,
                                gateUpEscape: gateUpEscape,
                                indices: inds,
                                sharedWeight: sharedW,
                                sharedScales: sharedS,
                                sharedEscape: sharedE
                            )
                            activated = merged[0..<1, 0..<1, 0..<8]
                            if sharedW != nil {
                                mergedSharedActivated =
                                    merged[0, 0, 8..<9, .ellipsis]
                            }
                        } else {
                            lagunaTrace("routed gate/up QMV + SwiGLU (packed, top8)")
                            activated = lagunaRoutedSwiGLUQMVPackedTop8(
                                x,
                                fusedWeight: fusedWeight,
                                packedScales: packedBank,
                                gateUpEscape: _packedRoutedGateUpEscape
                                    ?? MLXArray.zeros([LagunaConstants.numExperts, 2], dtype: .uint8),
                                indices: inds
                            )
                        }
                    } else {
                        lagunaTrace("routed gate/up QMV + SwiGLU (packed scales)")
                        activated = lagunaRoutedSwiGLUQMVPacked(
                            x,
                            fusedWeight: fusedWeight,
                            packedScales: packedBank,
                            indices: inds
                        )
                    }
                } else {
                    if lagunaPackedScalesEnabled {
                        lagunaPackedScalesLog.note(
                            "inactive",
                            "routed swiglu qmv packed (bank missing; stock kernel dispatched)")
                    }
                    lagunaTrace("routed gate/up QMV + SwiGLU")
                    activated = lagunaRoutedSwiGLUQMV(
                        x,
                        fusedWeight: fusedWeight,
                        fusedScales: fusedScales,
                        indices: inds
                    )
                }
            } else {
                let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
                let gateUp = MLX.gatherQuantizedMM(
                    expanded,
                    fusedWeight,
                    scales: fusedScales,
                    biases: nil,
                    rhsIndices: inds,
                    transpose: true,
                    groupSize: 16,
                    bits: 4,
                    mode: .nvfp4,
                    sortedIndices: false
                )
                activated = lagunaInterleavedSwiGLU(
                    gateUp, split: _fusedRoutedGateUpSplit)
            }
            if lagunaFusedRoutedSharedDownResidualEnabled,
                let residual,
                let downWeight = _routedDownWeight,
                let halvedDownScales = _halvedRoutedDownScales,
                let downScalesEscape = _routedDownScalesEscape,
                let sharedInputs = sharedExpert.fusedSharedDownInputs(
                    x, sharedActivation: mergedSharedActivated),
                let halvedSharedDownScales = sharedExpert._halvedSharedDownScales,
                let sharedDownScalesEscape = sharedExpert._sharedDownScalesEscape,
                activated.dtype == .bfloat16,
                activated.dims(1, 1, LagunaConstants.numExpertsPerTok, 1,
                    LagunaConstants.moeIntermediateSize),
                downWeight.dtype == .uint32,
                downWeight.dims(LagunaConstants.numExperts, LagunaConstants.hiddenSize,
                    LagunaConstants.moeIntermediateSize / 8),
                halvedDownScales.dtype == .uint8,
                halvedDownScales.dims(LagunaConstants.numExperts, LagunaConstants.hiddenSize,
                    LagunaConstants.moeIntermediateSize / 32),
                downScalesEscape.dtype == .uint8,
                weights.dtype == .float32,
                weights.dims(1, 1, LagunaConstants.numExpertsPerTok),
                routedScalingFactor == Float(LagunaConstants.moeRoutedScalingFactor),
                residual.dtype == .bfloat16,
                residual.dims(1, 1, LagunaConstants.hiddenSize)
            {
                lagunaTrace("routed+shared down residual")
                return lagunaRoutedSharedDownResidual(
                    routedActivated: activated,
                    routedDownWeight: downWeight,
                    routedDownScales: halvedDownScales,
                    routedDownScalesEscape: downScalesEscape,
                    indices: inds,
                    routerWeights: weights,
                    sharedActivated: sharedInputs.activated,
                    sharedDownWeight: sharedInputs.downWeight,
                    sharedDownScales: halvedSharedDownScales,
                    sharedDownScalesEscape: sharedDownScalesEscape,
                    residual: residual
                )
            } else if lagunaFusedRoutedDownReduceEnabled,
                let downWeight = _routedDownWeight,
                let downScales = _routedDownScales,
                activated.dtype == .bfloat16,
                activated.dims(1, 1, LagunaConstants.numExpertsPerTok, 1,
                    LagunaConstants.moeIntermediateSize),
                downWeight.dtype == .uint32,
                downWeight.dims(LagunaConstants.numExperts, LagunaConstants.hiddenSize,
                    LagunaConstants.moeIntermediateSize / 8),
                downScales.dtype == .uint8,
                downScales.dims(LagunaConstants.numExperts, LagunaConstants.hiddenSize,
                    LagunaConstants.moeIntermediateSize / 16),
                weights.dtype == .float32,
                weights.dims(1, 1, LagunaConstants.numExpertsPerTok),
                routedScalingFactor == Float(LagunaConstants.moeRoutedScalingFactor)
            {
                lagunaTrace("routed down reduce")
                y = lagunaRoutedDownReduce(
                    activated,
                    downWeight: downWeight,
                    downScales: downScales,
                    indices: inds,
                    routerWeights: weights
                )
                routedAlreadyReduced = true
            } else {
                y = MLX.squeezed(
                    downProj(activated, inds, sortedIndices: false),
                    axis: -2)
            }
        } else {
            // PREFILL sorted-regime fused gate/up: same retained
            // row-concatenated NVFP4 bank the decode branch above uses, but
            // driven through `lagunaFusedSortedRoutedGateUp`, which mirrors
            // `SwitchGLU.callAsFunction`'s `doSort == true` path op for op
            // (see that function's doc comment for the line-by-line
            // correspondence). Falls back to the fully stock `switchMLP(x,
            // inds)` -- unchanged from before this fusion -- whenever the
            // flag is off, the fused bank wasn't built, or the guarded
            // shapes/dtypes/regime don't match; either way `y` ends up with
            // the exact same shape/dtype `switchMLP` alone would have
            // produced, so every consumer below (including the
            // `lagunaPrefillMoETailEnabled` tail fusion) is unaffected by
            // which branch ran.
            if lagunaPrefillFusedRoutedGateUpEnabled,
                let fusedWeight = _fusedRoutedGateUpWeight,
                let fusedScales = _fusedRoutedGateUpScales,
                let downProj = _routedDownProj,
                x.dim(1) > 1,
                inds.size >= 64,
                fusedWeight.dtype == .uint32,
                fusedScales.dtype == .uint8,
                _fusedRoutedGateUpSplit == LagunaConstants.moeIntermediateSize
            {
                lagunaTrace("prefill fused routed gate/up")
                let routed = lagunaFusedSortedRoutedGateUp(
                    x,
                    indices: inds,
                    fusedWeight: fusedWeight,
                    fusedScales: fusedScales,
                    halvedScales: _halvedFusedRoutedGateUpScales,
                    scalesEscape: _fusedRoutedGateUpScalesEscape,
                    split: _fusedRoutedGateUpSplit,
                    downProj: downProj,
                    downWeight: _routedDownWeight,
                    halvedDownScales: _halvedRoutedDownScales,
                    downScalesEscape: _routedDownScalesEscape,
                    deferUnsort:
                        lagunaPrefillSortedMoETailEnabled
                        && lagunaPrefillMoETailEnabled
                        && residual != nil
                )
                y = routed.output
                sortedTailInverseOrder = routed.inverseOrder
            } else {
                y = switchMLP(x, inds)
            }
            if let inverseOrder = sortedTailInverseOrder,
                lagunaPrefillMoETailEnabled,
                let residual,
                x.dim(1) > 1,
                y.dtype == .bfloat16,
                y.size
                    == x.dim(1) * LagunaConstants.numExpertsPerTok
                        * LagunaConstants.hiddenSize,
                inverseOrder.dtype == .uint32,
                inverseOrder.size == x.dim(1) * LagunaConstants.numExpertsPerTok,
                weights.dtype == .float32,
                weights.dims(1, x.dim(1), LagunaConstants.numExpertsPerTok),
                routedScalingFactor == Float(LagunaConstants.moeRoutedScalingFactor),
                residual.dtype == .bfloat16,
                residual.dims(1, x.dim(1), LagunaConstants.hiddenSize)
            {
                let sharedOut = sharedExpert(x)
                if sharedOut.dtype == .bfloat16, sharedOut.sameDims(residual) {
                    lagunaTrace("prefill sorted moe tail")
                    return lagunaPrefillSortedMoETail(
                        sortedExpertOutputs: y,
                        inverseOrder: inverseOrder,
                        routerWeights: weights,
                        sharedOutput: sharedOut,
                        residual: residual
                    )
                }
                // Preserve the stock fallback for an unexpected shared-expert
                // shape while reusing the already-built shared output.
                y = MLX.squeezed(
                    scatterUnsort(x: y, invOrder: inverseOrder, shape: inds.shape),
                    axis: -2)
                var reduced = weightedExpertSum(y, weights.asType(y.dtype))
                if routedScalingFactor != 1 {
                    reduced = reduced * routedScalingFactor
                }
                return residual + (reduced + sharedOut)
            }
            if let inverseOrder = sortedTailInverseOrder {
                // A generic guard declined after down_proj was deliberately
                // left sorted. Restore the exact SwitchGLU representation
                // before entering any stock consumer.
                y = MLX.squeezed(
                    scatterUnsort(x: y, invOrder: inverseOrder, shape: inds.shape),
                    axis: -2)
                sortedTailInverseOrder = nil
            }
            if lagunaPrefillMoETailEnabled,
                let residual,
                x.dim(1) > 1,
                y.dtype == .bfloat16,
                y.ndim == 4,
                y.dim(0) == 1,
                y.dim(1) == x.dim(1),
                y.dim(2) == LagunaConstants.numExpertsPerTok,
                y.dim(3) == LagunaConstants.hiddenSize,
                weights.dtype == .float32,
                weights.dims(1, x.dim(1), LagunaConstants.numExpertsPerTok),
                routedScalingFactor == Float(LagunaConstants.moeRoutedScalingFactor),
                residual.dtype == .bfloat16,
                residual.dims(1, x.dim(1), LagunaConstants.hiddenSize)
            {
                let sharedOut = sharedExpert(x)
                if sharedOut.dtype == .bfloat16, sharedOut.sameDims(residual) {
                    lagunaTrace("prefill moe tail")
                    return lagunaPrefillMoETail(
                        expertOutputs: y,
                        routerWeights: weights,
                        sharedOutput: sharedOut,
                        residual: residual
                    )
                }
                // Unreachable with the stock shared expert; keep the stock
                // arithmetic while reusing the already-built shared output.
                var reduced = weightedExpertSum(y, weights.asType(y.dtype))
                if routedScalingFactor != 1 {
                    reduced = reduced * routedScalingFactor
                }
                return residual + (reduced + sharedOut)
            }
        }
        if !routedAlreadyReduced {
            y = weightedExpertSum(y, weights.asType(y.dtype))
            if routedScalingFactor != 1 {
                y = y * routedScalingFactor
            }
        }
        if let residual,
            let output = sharedExpert.fusedSharedDownResidual(
                x,
                routed: y,
                residual: residual
            )
        {
            return output
        }
        let r2 = y + sharedExpert(x)
        return residual.map { $0 + r2 } ?? r2
    }
}

// MARK: - Decoder Layer

final class LagunaRuntimeDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: LagunaRuntimeAttention
    let mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let attentionType: LagunaLayerType

    init(_ config: LagunaConfig, layerIdx: Int) {
        self._selfAttn.wrappedValue = LagunaRuntimeAttention(config, layerIdx: layerIdx)

        if config.isSparse(layer: layerIdx) {
            self.mlp = LagunaRuntimeSparseMoEBlock(config)
        } else {
            self.mlp = LagunaRuntimeMLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))

        self.attentionType = config.layerType(forLayer: layerIdx)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        qkRoPEAngles: MLXArray? = nil,
        qkRoPEOffsets: MLXArray? = nil
    ) -> MLXArray {
        let r = selfAttn(
            x,
            inputNorm: inputLayerNorm,
            mask: mask,
            cache: cache,
            qkRoPEAngles: qkRoPEAngles,
            qkRoPEOffsets: qkRoPEOffsets
        )
        let h: MLXArray
        let normalized: MLXArray
        var routerLogits: MLXArray?
        if lagunaFusedResidualRMSNormRouterEnabled,
            x.dtype == .bfloat16, r.dtype == .bfloat16,
            postAttentionLayerNorm.weight.dtype == .bfloat16,
            x.dims(1, 1, LagunaConstants.hiddenSize), x.sameDims(r),
            let sparse = mlp as? LagunaRuntimeSparseMoEBlock,
            sparse.gate.weight.dtype == .bfloat16,
            sparse.gate.weight.dims(LagunaConstants.numExperts, LagunaConstants.hiddenSize)
        {
            let fused = lagunaResidualRMSNormRouter(
                residual: x,
                branch: r,
                weight: postAttentionLayerNorm.weight,
                routerWeight: sparse.gate.weight,
                correctionBias: sparse.gate.eScoreCorrectionBias)
            h = fused.summed
            normalized = fused.normalized
            routerLogits = fused.routerLogits
        } else if lagunaFusedResidualRMSNormEnabled,
            x.dtype == .bfloat16, r.dtype == .bfloat16,
            postAttentionLayerNorm.weight.dtype == .bfloat16,
            x.sameDims(r), x.dim(-1) == LagunaConstants.hiddenSize,
            x.size == LagunaConstants.hiddenSize
        {
            lagunaTrace("residual+rmsnorm")
            (h, normalized) = lagunaResidualRMSNorm(
                residual: x, branch: r, weight: postAttentionLayerNorm.weight)
        } else if lagunaPrefillFusedResidualRMSNormEnabled,
            x.dtype == .bfloat16, r.dtype == .bfloat16,
            postAttentionLayerNorm.weight.dtype == .bfloat16,
            x.sameDims(r), x.ndim == 3, x.dim(0) == 1,
            x.dim(-1) == LagunaConstants.hiddenSize,
            x.dim(1) > 1
        {
            // Prefill (multi-token) counterpart of the fused decode branch
            // above: same row-count-general `lagunaResidualRMSNorm` kernel,
            // only the call-site guard differs. Full exactness argument in
            // `lagunaPrefillFusedResidualRMSNormEnabled`'s doc comment.
            lagunaTrace("prefill residual+rmsnorm")
            (h, normalized) = lagunaResidualRMSNorm(
                residual: x, branch: r, weight: postAttentionLayerNorm.weight)
        } else {
            h = x + r
            normalized = postAttentionLayerNorm(h)
        }
        if (
            lagunaFusedSharedDownResidualEnabled ||
                lagunaFusedRoutedSharedDownResidualEnabled
        ),
            normalized.dtype == .bfloat16,
            normalized.dims(1, 1, LagunaConstants.hiddenSize),
            h.dtype == .bfloat16,
            h.sameDims(normalized),
            let sparse = mlp as? LagunaRuntimeSparseMoEBlock
        {
            return sparse(
                normalized, residual: h, routerLogits: routerLogits)
        }
        // Multi-token prefill: hand the residual to the sparse block so the
        // prefill MoE tail kernel can fold the final residual add. When any
        // guard inside declines, the block computes `residual + (y + shared)`
        // itself — the identical stock ops this call site would otherwise
        // issue.
        if lagunaPrefillMoETailEnabled,
            x.dim(1) > 1,
            let sparse = mlp as? LagunaRuntimeSparseMoEBlock
        {
            return sparse(
                normalized, residual: h, routerLogits: routerLogits)
        }
        // Layer-0-only decode fusion: `fusedDenseDownResidual` returns nil off
        // layer 0's decode shape (or if a guard declines); stock path then runs.
        if let dense = mlp as? LagunaRuntimeMLP,
            let fused = dense.fusedDenseDownResidual(normalized, residual: h)
        {
            return fused
        }
        let r2 = mlp(normalized)
        return h + r2
    }

    /// Final-layer prefill specialization: every row commits K/V, but only the
    /// last query/output row runs attention output projection + the terminal MLP.
    func callLastPrefillRow(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        if lagunaTerminalPrefillFusionEnabled {
            // Fused terminal row (see flag doc). Reuses the ordinary path's
            // accepted row-local fusion; `else` is the exact stock fallback.
            let normalized = inputLayerNorm(x)
            let r = selfAttn.callLastPrefillRow(normalized, cache: cache)
            let lastResidual = lagunaLastTokenHidden(x)
            let h: MLXArray
            let normalizedAfterAttention: MLXArray
            var routerLogits: MLXArray?
            if lagunaFusedResidualRMSNormRouterEnabled,
                lastResidual.dtype == .bfloat16, r.dtype == .bfloat16,
                postAttentionLayerNorm.weight.dtype == .bfloat16,
                lastResidual.dims(1, 1, LagunaConstants.hiddenSize),
                lastResidual.sameDims(r),
                let sparse = mlp as? LagunaRuntimeSparseMoEBlock,
                sparse.gate.weight.dtype == .bfloat16,
                sparse.gate.weight.dims(LagunaConstants.numExperts, LagunaConstants.hiddenSize)
            {
                let fused = lagunaResidualRMSNormRouter(
                    residual: lastResidual,
                    branch: r,
                    weight: postAttentionLayerNorm.weight,
                    routerWeight: sparse.gate.weight,
                    correctionBias: sparse.gate.eScoreCorrectionBias)
                h = fused.summed
                normalizedAfterAttention = fused.normalized
                routerLogits = fused.routerLogits
            } else if lagunaFusedResidualRMSNormEnabled,
                lastResidual.dtype == .bfloat16, r.dtype == .bfloat16,
                postAttentionLayerNorm.weight.dtype == .bfloat16,
                lastResidual.sameDims(r),
                lastResidual.dim(-1) == LagunaConstants.hiddenSize,
                lastResidual.size == LagunaConstants.hiddenSize
            {
                lagunaTrace("terminal prefill residual+rmsnorm")
                (h, normalizedAfterAttention) = lagunaResidualRMSNorm(
                    residual: lastResidual, branch: r,
                    weight: postAttentionLayerNorm.weight)
            } else {
                h = lastResidual + r
                normalizedAfterAttention = postAttentionLayerNorm(h)
            }
            if (
                lagunaFusedSharedDownResidualEnabled ||
                    lagunaFusedRoutedSharedDownResidualEnabled
            ),
                normalizedAfterAttention.dtype == .bfloat16,
                normalizedAfterAttention.dims(1, 1, LagunaConstants.hiddenSize),
                h.dtype == .bfloat16,
                h.sameDims(normalizedAfterAttention),
                let sparse = mlp as? LagunaRuntimeSparseMoEBlock
            {
                return sparse(
                    normalizedAfterAttention, residual: h,
                    routerLogits: routerLogits)
            }
            if let dense = mlp as? LagunaRuntimeMLP,
                let fused = dense.fusedDenseDownResidual(
                    normalizedAfterAttention, residual: h)
            {
                return fused
            }
            let r2 = mlp(normalizedAfterAttention)
            return h + r2
        } else {
            let normalized = inputLayerNorm(x)
            let r = selfAttn.callLastPrefillRow(normalized, cache: cache)
            let h = lagunaLastTokenHidden(x) + r
            let r2 = mlp(postAttentionLayerNorm(h))
            return h + r2
        }
    }
}

// MARK: - Model

/// Single-token embedding gather plus position-atlas row selection. The
/// embedding row is copied as BF16 bits; the angle rows are copied as FP32
/// bits. The stock embedding and the two stock probe RoPE calls produce the
/// same three output buffers separately.
private let lagunaDecodeEmbeddingRoPEAtlasKernel = MLXFast.metalKernel(
    name: "laguna_decode_embedding_rope_atlas_bf16_2048_v2",
    inputNames: [
        "tokens", "embedding_weight", "full_atlas", "sliding_atlas",
        "atlas_position",
    ],
    outputNames: ["hidden", "full_angles", "sliding_angles"],
    source: """
        constexpr uint hidden_size = 2048;
        constexpr uint hidden_vectors = hidden_size / 4;
        constexpr uint full_width = 64;
        constexpr uint sliding_width = 128;

        uint lane = thread_position_in_grid.x;
        uint token = uint(tokens[0]);
        uint position = uint(atlas_position);

        const device vec<bfloat, 4>* embedding_vectors =
            (const device vec<bfloat, 4>*)(
                embedding_weight + token * hidden_size);
        device vec<bfloat, 4>* hidden_vectors_out =
            (device vec<bfloat, 4>*)(hidden);
        if (lane < hidden_vectors) {
            hidden_vectors_out[lane] = embedding_vectors[lane];
        }

        if (lane < full_width / 4) {
            const device vec<float, 4>* atlas_vectors =
                (const device vec<float, 4>*)(
                    full_atlas + position * full_width);
            ((device vec<float, 4>*)(full_angles))[lane] =
                atlas_vectors[lane];
        }
        if (lane < sliding_width / 4) {
            const device vec<float, 4>* atlas_vectors =
                (const device vec<float, 4>*)(
                    sliding_atlas + position * sliding_width);
            ((device vec<float, 4>*)(sliding_angles))[lane] =
                atlas_vectors[lane];
        }
        """,
    ensureRowContiguous: true
)

private func lagunaDecodeEmbeddingRoPEAtlas(
    tokens: MLXArray,
    embeddingWeight: MLXArray,
    fullAtlas: MLXArray,
    slidingAtlas: MLXArray,
    position: Int
) -> (hidden: MLXArray, fullAngles: MLXArray, slidingAngles: MLXArray)? {
    guard tokens.dtype == .int32,
        tokens.dims(1, 1),
        embeddingWeight.dtype == .bfloat16,
        embeddingWeight.dims(LagunaConstants.vocabSize, LagunaConstants.hiddenSize),
        fullAtlas.dtype == .float32,
        fullAtlas.dims(1, 1, lagunaRoPEAngleAtlasLength, LagunaConstants.headDim / 2),
        slidingAtlas.dtype == .float32,
        slidingAtlas.dims(1, 1, lagunaRoPEAngleAtlasLength, LagunaConstants.headDim),
        position >= 0, position < lagunaRoPEAngleAtlasLength
    else {
        return nil
    }

    let kernelInputs: [any ScalarOrArray] = [
        tokens,
        embeddingWeight,
        fullAtlas,
        slidingAtlas,
        Int32(position),
    ]
    let outputs = lagunaDecodeEmbeddingRoPEAtlasKernel(
        kernelInputs,
        grid: (512, 1, 1),
        threadGroup: (512, 1, 1),
        outputShapes: [
            [1, 1, LagunaConstants.hiddenSize],
            [1, 1, 1, LagunaConstants.headDim / 2],
            [1, 1, 1, LagunaConstants.headDim],
        ],
        outputDTypes: [.bfloat16, .float32, .float32]
    )
    lagunaTrace("decode embedding+rope atlas")
    return (outputs[0], outputs[1], outputs[2])
}

/// The Laguna text tower: unscaled embedding and 40 decoder layers. The final
/// RMSNorm remains a child of this module for checkpoint compatibility, but
/// the scored wrapper applies it after selecting the only consumed row.
final class LagunaRuntimeModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [LagunaRuntimeDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let layerTypes: [LagunaLayerType]
    let slidingWindow: Int
    let fullAttentionIdx: Int
    let slidingAttentionIdx: Int
    let _fullRoPEAngleSeed: MLXArray
    let _slidingRoPEAngleSeed: MLXArray
    var _fullRoPEAngleAtlas: MLXArray?
    var _slidingRoPEAngleAtlas: MLXArray?
    /// Which layers fire `asyncEval` during single-token decode. Derived from
    /// the process-wide async-stage flag and the fixed layer count.
    let decodeFireMask: UInt64

    init(_ config: LagunaConfig) {
        precondition(config.vocabSize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

        self._layers.wrappedValue = (0..<config.numHiddenLayers).map {
            LagunaRuntimeDecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))

        self.layerTypes = config.layerTypes
        self.slidingWindow = config.slidingWindow
        self.fullAttentionIdx = config.layerTypes.firstIndex(of: .full) ?? 0
        self.slidingAttentionIdx = config.layerTypes.firstIndex(of: .sliding) ?? 0
        self._fullRoPEAngleSeed = MLXArray(
            Array(repeating: Float(0.7426255941390991), count: LagunaConstants.headDim / 4)
                + Array(repeating: Float(0), count: LagunaConstants.headDim / 4),
            [1, 1, 1, LagunaConstants.headDim / 2]
        )
        // Plain RoPE rotates the pair (p, p + 64) as
        // `(x_p cos - x_{p+64} sin, x_p sin + x_{p+64} cos)`, so a row of ones
        // followed by zeros comes back as exactly `[cos..., sin...]`. The
        // full-attention seed above carries `1 / mscale` instead because YaRN
        // scales its rotary inputs; sliding layers apply no mscale.
        self._slidingRoPEAngleSeed = MLXArray(
            Array(repeating: Float(1), count: LagunaConstants.headDim / 2)
                + Array(repeating: Float(0), count: LagunaConstants.headDim / 2),
            [1, 1, 1, LagunaConstants.headDim]
        )
        self.decodeFireMask =
            switch lagunaDecodeAsyncStage {
            case .off, .norm, .logits:
                0
            case .layer(let idx):
                UInt64(1) << UInt64(idx)
            case .ladder(let n):
                (0..<UInt64(config.numHiddenLayers)).reduce(UInt64(0)) { acc, i in
                    (Int(i) + 1) % n == 0 ? acc | (UInt64(1) << i) : acc
                }
            case .explicit(let mask):
                mask
            }
    }

    /// Materialize exact position rows with the same stock RoPE instances the
    /// two attention families use. Broadcasting the probe seeds along the
    /// sequence dimension makes row `p` exactly the scalar-offset probe at
    /// position `p`, including YaRN's authoritative FP32 rounding.
    func prepareRoPEAngleAtlases() -> [MLXArray] {
        // The decode atlas consumer keys on `lagunaRoPEAngleAtlasEnabled`;
        // the prefill QK-norm+RoPE fusion consumes the same two tables
        // whenever it is enabled, so either flag builds them.
        guard lagunaRoPEAngleAtlasEnabled || lagunaPrefillQKNormRoPEEnabled,
            lagunaFusedFullQKNormYaRNEnabled,
            lagunaFusedSlidingQKNormRoPEEnabled,
            layerTypes.contains(.full),
            layerTypes.contains(.sliding)
        else {
            return []
        }
        if let fullAtlas = _fullRoPEAngleAtlas,
            let slidingAtlas = _slidingRoPEAngleAtlas
        {
            return [fullAtlas, slidingAtlas]
        }

        let fullSeed = broadcast(
            _fullRoPEAngleSeed,
            to: [
                1, 1, lagunaRoPEAngleAtlasLength, LagunaConstants.headDim / 2,
            ])
        let slidingSeed = broadcast(
            _slidingRoPEAngleSeed,
            to: [
                1, 1, lagunaRoPEAngleAtlasLength, LagunaConstants.headDim,
            ])
        let fullAtlas = layers[fullAttentionIdx].selfAttn.rope(fullSeed, offset: 0)
        let slidingAtlas = layers[slidingAttentionIdx].selfAttn.rope(
            slidingSeed, offset: 0)
        _fullRoPEAngleAtlas = fullAtlas
        _slidingRoPEAngleAtlas = slidingAtlas
        return [fullAtlas, slidingAtlas]
    }

    /// Return a host position only for the exact direct-decode cache pair.
    /// Exact runtime type checks deliberately exclude compilable subclasses,
    /// whose compatibility `offset` getter may synchronize a graph value.
    private func decodeRoPEAtlasPosition(
        inputs: MLXArray, cache: [KVCache]?
    ) -> Int? {
        guard lagunaRoPEAngleAtlasEnabled || lagunaRoPEAtlasViewsEnabled,
            lagunaFusedFullQKNormYaRNEnabled,
            lagunaFusedSlidingQKNormRoPEEnabled,
            inputs.dtype == .int32,
            inputs.dims(1, 1),
            _fullRoPEAngleAtlas != nil,
            _slidingRoPEAngleAtlas != nil,
            let cache,
            fullAttentionIdx < cache.count,
            slidingAttentionIdx < cache.count
        else {
            return nil
        }

        let fullCache = cache[fullAttentionIdx]
        let slidingCache = cache[slidingAttentionIdx]
        guard type(of: fullCache) == KVCacheSimple.self,
            type(of: slidingCache) == RotatingKVCache.self,
            slidingCache.maxSize == slidingWindow
        else {
            return nil
        }

        let fullPosition = fullCache.offset
        let slidingPosition = slidingCache.offset
        guard fullPosition == slidingPosition,
            fullPosition >= 0,
            fullPosition < lagunaRoPEAngleAtlasLength
        else {
            return nil
        }
        return fullPosition
    }

    /// Runs `attention`'s own RoPE layer over `seed` at the cache's current
    /// position, honoring a graph-valued offset when the cache carries one.
    private func ropeAngleTable(
        seed: MLXArray, attention: LagunaRuntimeAttention, cache: KVCache?
    ) -> MLXArray {
        if let graphOffset = graphOffsetArray(for: cache) {
            return attention.rope(seed, offset: graphOffset)
        }
        return attention.rope(seed, offset: cache?.offset ?? 0)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h: MLXArray
        var fullRoPEAngles: MLXArray?
        var slidingRoPEAngles: MLXArray?
        var qkRoPEOffsets: MLXArray?
        if lagunaRoPEAngleAtlasEnabled,
            let position = decodeRoPEAtlasPosition(inputs: inputs, cache: cache),
            let fullAtlas = _fullRoPEAngleAtlas,
            let slidingAtlas = _slidingRoPEAngleAtlas,
            let atlasOutputs = lagunaDecodeEmbeddingRoPEAtlas(
                tokens: inputs,
                embeddingWeight: embedTokens.weight,
                fullAtlas: fullAtlas,
                slidingAtlas: slidingAtlas,
                position: position)
        {
            h = atlasOutputs.hidden
            fullRoPEAngles = atlasOutputs.fullAngles
            slidingRoPEAngles = atlasOutputs.slidingAngles
        } else if lagunaRoPEAtlasViewsEnabled,
            let position = decodeRoPEAtlasPosition(inputs: inputs, cache: cache),
            let fullAtlas = _fullRoPEAngleAtlas,
            let slidingAtlas = _slidingRoPEAngleAtlas
        {
            // Zero-dispatch angle carrier: stock embedding gather plus two
            // zero-copy row views of the load-time atlases. The atlas row at
            // `position` carries the same FP32 floats the probe dispatch
            // would have produced (the atlas is that probe, broadcast over
            // positions at load time), and the row slice of the contiguous
            // atlas is row-contiguous, so the fused QK-norm+RoPE kernels
            // consume it without any copy or added kernel.
            h = embedTokens(inputs)
            fullRoPEAngles = fullAtlas[0..., 0..., position..<(position + 1), 0...]
            slidingRoPEAngles = slidingAtlas[0..., 0..., position..<(position + 1), 0...]
        } else {
            // Verbatim stock fallback for prefill, unsupported caches and
            // positions outside the precomputed atlas.
            h = embedTokens(inputs)
            let isSingleTokenDecode = h.dim(0) == 1 && h.dim(1) == 1
            fullRoPEAngles =
                lagunaFusedFullQKNormYaRNEnabled && isSingleTokenDecode
                ? ropeAngleTable(
                    seed: _fullRoPEAngleSeed,
                    attention: layers[fullAttentionIdx].selfAttn,
                    cache: cache?[fullAttentionIdx])
                : nil
            slidingRoPEAngles =
                lagunaFusedSlidingQKNormRoPEEnabled && isSingleTokenDecode
                ? ropeAngleTable(
                    seed: _slidingRoPEAngleSeed,
                    attention: layers[slidingAttentionIdx].selfAttn,
                    cache: cache?[slidingAttentionIdx])
                : nil
            // Prefill: hand every layer the family's load-time angle atlas
            // plus the cache offset the stock `applyRotaryPosition` would
            // have used, so the fused multi-token QK-norm+RoPE kernels read
            // the same cos/sin floats the stock rope dispatches would have
            // computed. Requires a host-known offset (a graph-valued offset
            // could not be bounds-checked against the atlas) inside the
            // atlas range; anything else keeps the stock path.
            if lagunaPrefillQKNormRoPEEnabled, !isSingleTokenDecode,
                h.dim(0) == 1,
                let fullAtlas = _fullRoPEAngleAtlas,
                let slidingAtlas = _slidingRoPEAngleAtlas
            {
                let length = h.dim(1)
                let familyCache =
                    fullAttentionIdx < (cache?.count ?? 0)
                    ? cache?[fullAttentionIdx] : nil
                if graphOffsetArray(for: familyCache) == nil {
                    let offset = familyCache?.offset ?? 0
                    if offset >= 0, offset + length <= lagunaRoPEAngleAtlasLength {
                        fullRoPEAngles = fullAtlas
                        slidingRoPEAngles = slidingAtlas
                        qkRoPEOffsets = MLXArray([Int32(offset)])
                    }
                }
            }
        }

        // One mask per attention family, derived from a representative
        // layer's cache offset: all full-attention caches advance in
        // lockstep, as do all sliding caches (vendored `LagunaModelInner`
        // convention).
        let fullMask = createAttentionMask(h: h, cache: cache?[fullAttentionIdx])
        let slidingMask = createAttentionMask(
            h: h, cache: cache?[slidingAttentionIdx], windowSize: slidingWindow)

        let isSingleTokenDecode = inputs.dims(1, 1)

        // One cos/sin table per attention family per decode step, shared by
        // every layer of that family (their caches advance in lockstep). Each
        // table is produced by running the family's own RoPE layer over a
        // seed row, so the angles are the exact floats that layer's kernel
        // would have computed rather than a re-derivation.

        for (i, layer) in layers.enumerated() {
            let isFull = layerTypes[i] == .full
            let mask = isFull ? fullMask : slidingMask
            let qkRoPEAngles = isFull ? fullRoPEAngles : slidingRoPEAngles
            if i == layers.count - 1, h.dim(1) > 1 {
                if case .causal = mask {
                    h = layer.callLastPrefillRow(h, cache: cache?[i])
                } else {
                    h = layer(
                        h,
                        mask: mask,
                        cache: cache?[i],
                        qkRoPEAngles: qkRoPEAngles,
                        qkRoPEOffsets: qkRoPEOffsets
                    )
                    if isSingleTokenDecode, (decodeFireMask >> UInt64(i)) & 1 == 1 {
                        asyncEval(h)
                    }
                }
            } else {
                h = layer(
                    h,
                    mask: mask,
                    cache: cache?[i],
                    qkRoPEAngles: qkRoPEAngles,
                    qkRoPEOffsets: qkRoPEOffsets
                )
                if isSingleTokenDecode, (decodeFireMask >> UInt64(i)) & 1 == 1 {
                    asyncEval(h)
                }
                if lagunaPrefillAsyncLadderStride > 0, h.dim(1) > 1,
                    (i + 1) % lagunaPrefillAsyncLadderStride == 0
                {
                    asyncEval(h)
                }
            }
        }

        return h
    }
}

/// Scored Laguna runtime model: last-token vocabulary head over the
/// reimplemented Laguna text tower.
///
/// `callAsFunction(_:cache:)` serves both prompt prefill
/// (`[1, L]`) and single-token decode steps (`[1, 1]`) and returns
/// `[1, 1, vocab]` last-token logits; `newCache(parameters:)` creates the
/// per-layer cache stack (unbounded `StandardKVCache` for full-attention
/// layers, `RotatingKVCache(512)` for sliding layers). Laguna applies NO
/// final logit softcap and NO embedding scaling.
public final class LagunaRuntimeModel: Module, LanguageModel {
    @ModuleInfo(key: "model") var model: LagunaRuntimeModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let configuration: LagunaConfig

    /// Certified two-pass lm_head elision for single-token decode
    /// (notes/68). Non-nil only when `lagunaLmHeadPruneEnabled` (default ON;
    /// set `DARKBLOOM_LM_HEAD_PRUNE=0` to disable) and the coarse copy built
    /// cleanly; the stock full pass is used otherwise.
    private var lmHeadPruner: LagunaLmHeadPruner?

    public init(_ config: LagunaConfig) {
        self.configuration = config
        self._model.wrappedValue = LagunaRuntimeModelInner(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
        super.init()

        // Match the vendored Poolside Laguna model exactly: only the routed
        // experts and shared expert are NVFP4. Quantizing one sparse decoder
        // layer at a time avoids asking Module.update to descend through the
        // dense layer 0, which has no quantized child.
        for layer in model.layers where layer.mlp is LagunaRuntimeSparseMoEBlock {
            quantize(model: layer) { path, _ in
                if path.contains("switch_mlp") || path.contains("shared_expert") {
                    return (
                        groupSize: config.quantization.groupSize,
                        bits: config.quantization.bits,
                        mode: .nvfp4
                    )
                }
                return nil
            }
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let fullHidden = model(inputs, cache: cache)
        // Every consumer of multi-token logits reads only the LAST
        // position's row. Slice before the row-independent final RMSNorm and
        // vocabulary head so prefill neither normalizes nor projects the
        // preceding rows. For single-token decode the slice is a no-op.
        let hidden = model.norm(lagunaLastTokenHidden(fullHidden))
        if case .norm = lagunaDecodeAsyncStage, inputs.dims(1, 1) {
            asyncEval(hidden)
        }

        let result: MLXArray
        if let lmHead {
            if let pruner = lmHeadPruner,
                inputs.dims(1, 1) || lagunaLmHeadPrunePrefillEnabled
            {
                // Certified two-pass final-row head (notes/68): full BF16
                // logits, bit-identical to stock in every argmax-reachable
                // slot. When enabled, prefill has already sliced to the last
                // hidden row; decode always retains this pruner. Only
                // single-token decode takes the three-level screen, whose
                // level-one pass reads 25.7 MB/step less of the int5 planes.
                result = pruner.logits(
                    hidden: hidden,
                    lmHeadWeight: lmHead.weight,
                    useFusedRefinement: inputs.dims(1, 1))
            } else {
                result = lmHead(hidden)
            }
        } else {
            result = model.embedTokens.asLinear(hidden)
        }
        if case .logits = lagunaDecodeAsyncStage, inputs.dims(1, 1) {
            asyncEval(result)
        }
        return result
    }

    public func prepare(
        _ input: LMInput,
        cache _: [KVCache],
        windowSize _: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    public func newCache(parameters _: GenerateParameters?) -> [KVCache] {
        (0..<configuration.numHiddenLayers).map { layerIndex in
            if configuration.layerTypes[layerIndex] == .full {
                StandardKVCache()
            } else {
                RotatingKVCache(maxSize: configuration.slidingWindow, keep: 0)
            }
        }
    }

    /// Builds the retained fused runtime weight layouts (fused QKV, fused
    /// shared-expert gate/up, fused routed gate/up decode banks) once the
    /// checkpoint parameters are installed and evaluated. Called by the
    /// weight cache after `update` + `eval`, before constructor-time warmup,
    /// so the concatenations read materialized weights and the fused arrays
    /// are resident before the first forward. The module tree and its
    /// checkpoint parameters are never restructured; every fused layout is a
    /// derived side copy.
    func prepareFusedRuntimeWeights() {
        var fusedArrays = model.prepareRoPEAngleAtlases()
        for layer in model.layers {
            if lagunaUseNativeAffineQKV(layer: layer.selfAttn.layerIdx) {
                fusedArrays.append(
                    contentsOf: layer.selfAttn.prepareNativeAffineQKVWeight())
            }
            if lagunaUseNativeAffineOProj(layer: layer.selfAttn.layerIdx) {
                fusedArrays.append(
                    contentsOf: layer.selfAttn.prepareNativeAffineOProjWeight())
            }
            if lagunaFusedQKVEnabled, let fused = layer.selfAttn.prepareFusedQKVWeight() {
                fusedArrays.append(fused)
            }
            fusedArrays.append(
                contentsOf: layer.selfAttn.prepareLastPrefillProjectionWeights())
            if let sparse = layer.mlp as? LagunaRuntimeSparseMoEBlock {
                if lagunaFusedSharedGateUpEnabled {
                    fusedArrays.append(contentsOf: sparse.sharedExpert.prepareFusedSharedGateUp())
                }
                if lagunaFusedRoutedGateUpEnabled {
                    fusedArrays.append(contentsOf: sparse.prepareFusedRoutedGateUp())
                }
            } else if let dense = layer.mlp as? LagunaRuntimeMLP {
                if lagunaFusedDenseGateUpSwiGLUEnabled,
                    let fused = dense.prepareFusedDenseGateUp()
                {
                    fusedArrays.append(fused)
                }
            }
        }
        if !fusedArrays.isEmpty {
            eval(fusedArrays)
        }
        // Certified two-pass lm_head coarse copy (notes/68), gated by
        // `lagunaLmHeadPruneEnabled` (DARKBLOOM_LM_HEAD_PRUNE, default ON;
        // set "0" to disable). Built after the fused layouts so it reads
        // materialized BF16 weights; quantized() runs one untimed dispatch
        // over 205M elements (~ms).
        if lagunaLmHeadPruneEnabled, let lmHead {
            lmHeadPruner = LagunaLmHeadPruner(lmHeadWeight: lmHead.weight)
            if let pruner = lmHeadPruner {
                eval(pruner.residentArrays)
                FileHandle.standardError.write(
                    Data("mlxfast: lm_head prune active (coarse copy resident)\n".utf8))
            }
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }
        // Drop precomputed rotary tables if a checkpoint ships them.
        return weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
    }
}
