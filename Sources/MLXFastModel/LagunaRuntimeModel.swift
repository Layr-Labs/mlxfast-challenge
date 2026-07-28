import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// Correctness-first Laguna runtime (Poolside Laguna XS 2.1, 256-expert MoE).
//
// This module tree closely follows the vendored reference implementation at
// `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Laguna.swift` (`LagunaModel` /
// `LagunaModelInner`), which is the behavior oracle for this port. It is a
// reimplementation rather than a wrapper for two load-bearing reasons:
//
// 1. The Poolside checkpoint stores the MoE router as a raw BF16
//    `mlp.gate.weight` matrix next to the F32
//    `mlp.gate.e_score_correction_bias`, while only expert projections are
//    NVFP4. The runtime mirrors those parameter paths exactly.
// 2. The vendored `LagunaModelInner`/`LagunaDecoderLayer` types are
//    fileprivate and `LagunaConfiguration`'s stored properties are internal
//    to MLXLLM, so the runtime layers (cache geometry, future fast-engine
//    and exact-verification waves) could not reach the internals through a
//    plain wrapper.
//
// All math is expressed with standard MLX ops and the vendored shared
// primitives (`attentionWithCacheUpdate`, `initializeRope`,
// `applyRotaryPosition`, `SwitchGLU`, `weightedExpertSum`, `RMSNorm`,
// `createAttentionMask`). No custom Metal kernels in this increment; the
// fused fast-engine and exact-pair/exact-four style optimizations are a
// later layer on top of this reference target.

func lagunaLastTokenRange(sequenceLength: Int) -> Range<Int>? {
    sequenceLength > 1 ? (sequenceLength - 1)..<sequenceLength : nil
}

func lagunaLastTokenHidden(_ hidden: MLXArray) -> MLXArray {
    guard let range = lagunaLastTokenRange(sequenceLength: hidden.dim(1)) else {
        return hidden
    }
    return hidden[0..., range, 0...]
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
func lagunaTrace(_ site: String) {
    guard lagunaTraceFusion else { return }
    lagunaTracedFusions.note(site)
}

// MARK: - Runtime fusion feature flags

// Each fusion below concatenates the OUTPUT ROWS of same-dtype projections
// that consume the same input. Per-row gemv/qmv/gather-qmv arithmetic is
// independent of which rows share a dispatch (every output row keeps its own
// K-loop and scale application in the original order), so the fused dispatch
// is bit-exact against the separate dispatches it replaces. The per-head
// g_proj (N=64) uses a different split-K gemv variant and is never fused.

/// `DARKBLOOM_FUSED_QKV` (default OFF; set "1" to enable): after checkpoint
/// load, retain one row-concatenated `[Wq; Wk; Wv]` BF16 weight per attention
/// layer and serve Q/K/V from a single projection dispatch. Ablation on the
/// paired local benchmark showed a mild prefill cost with no decode gain, so
/// this ships opt-in.
let lagunaFusedQKVEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_QKV"] == "1"

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

/// `DARKBLOOM_PREFILL_FUSED_GATE_UP` (default on; set "0" to disable):
/// prefill/SORTED-regime counterpart to `DARKBLOOM_FUSED_ROUTED_GATE_UP`
/// above. Serves the multi-token sorted gather-GEMM path (`indices.size >=
/// 64` -- `SwitchGLU`'s own threshold for taking `gatherSort`, which every
/// timed 512-token prefill request clears) from the same retained
/// row-concatenated NVFP4 `[gate; up]` bank the decode path already keeps
/// resident, instead of the stock two separate sorted gather-QMMs
/// (`gate_proj` then `up_proj`). Mechanism: one N=1024 gather-GEMM has half
/// the run-loop iterations and dispatch overhead of two N=512 ones, and,
/// like the decode fusion above, each *gathered* output row's K-loop and
/// scale application reads only its own weight/scale row independent of
/// which bank that row lives in -- so the fused dispatch is bit-exact
/// against the two separate ones it replaces.
///
/// The `DARKBLOOM_FUSED_ROUTED_GATE_UP` comment above records an earlier
/// ablation that measured this same idea hurting the M=512 sorted prefill
/// path; that measurement pre-dates RUNSKIP. A later measurement on a
/// RUNSKIP-era tree (note ba4561c, never landed) found this specific
/// prefill fusion a ~4% prefill win (373.5us -> 358.5us) instead. Shipped
/// default ON, behind its own flag, so it can be ablated and re-measured on
/// the ranked box independently of both the older decode-only flag and the
/// stale prefill finding. See `lagunaFusedSortedRoutedGateUp` and its call
/// site in `LagunaRuntimeSparseMoEBlock.forward` for the exact op-for-op
/// mirror of `SwitchGLU.callAsFunction`'s sorted branch.
let lagunaPrefillFusedRoutedGateUpEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_FUSED_GATE_UP"] != "0"

/// The expert-aligned gather-QMM consumes a 32-row gate/up-interleaved bank
/// and writes the packed 512-wide SwiGLU result into the first half of its
/// oversized MLX output allocation. The same environment switch controls the
/// backend dispatch and this view interpretation, keeping its ablation path
/// coherent.
let lagunaExpertAlignedGatherEnabled =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_EXPERT_ALIGNED_GATHER"] != "0"

/// Decode post-attention residual + RMSNorm fusion. The kernel emits
/// both the rounded BF16 residual (needed by the following skip connection)
/// and the normalized row (consumed immediately by the MLP), eliminating a
/// separate residual-add materialization/read. Restricted to the single-row
/// (`x.size == hiddenSize`) decode shape at its call site; see
/// `lagunaPrefillFusedResidualRMSNormEnabled` immediately below for the
/// multi-token counterpart, gated independently.
let lagunaFusedResidualRMSNormEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_RESIDUAL_RMS"] != "0"

/// `DARKBLOOM_PREFILL_FUSED_RESIDUAL_RMS` (default on; set "0" to ablate):
/// prefill (multi-token) counterpart of `lagunaFusedResidualRMSNormEnabled`
/// above, gated independently per this file's convention of separate
/// per-regime flags (see the router fusion split, `DARKBLOOM_FUSED_ROUTER`
/// vs. `DARKBLOOM_PREFILL_ROUTER_TOURNAMENT`). Reuses the IDENTICAL
/// `lagunaResidualRMSNorm` kernel the decode branch already calls -- no new
/// Metal source -- because that kernel's Swift wrapper and Metal body are
/// already fully row-count-general: `rows` is derived from
/// `residual.size / hiddenSize` (not hardcoded), the dispatch grid is
/// `rows` independent threadgroups, and the kernel body's own row index
/// (`threadgroup_position_in_grid.x`) plus all of its scratch
/// (`local_sums`, the running mean-square accumulator) are per-threadgroup
/// -- one row's computation never reads another row's data. Only the
/// call-site guard restricted it to decode; this flag lifts that
/// restriction for `x.dim(1) > 1` specifically, leaving the decode branch
/// above completely untouched (mutually exclusive guards: decode requires
/// `x.size == hiddenSize`, i.e. exactly one row; this branch requires
/// `x.dim(1) > 1`).
///
/// Exactness: with this flag off, EVERY prefill forward already falls
/// through this same call site's stock `else` arm (`h = x + r`; `normalized
/// = postAttentionLayerNorm(h)`) unconditionally, since the decode branch's
/// `x.size == hiddenSize` guard is always false for `L > 1` -- confirmed by
/// reading the guard, not assumed. That is the exact two-op sequence the
/// kernel already reproduces bit-for-bit for decode's `L == 1` case, and
/// RMSNorm has no cross-token interaction in either the stock ops or the
/// kernel, so the per-row equivalence already proven for one row holds row
/// by row for any `L`.
let lagunaPrefillFusedResidualRMSNormEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_FUSED_RESIDUAL_RMS"] != "0"

/// Issues the routed and shared gate/up NVFP4 QMVs as one nine-slot dispatch
/// (see `lagunaRoutedSharedSwiGLUQMVKernel`). Set
/// `DARKBLOOM_FUSED_ROUTED_SHARED_SWIGLU_QMV=0` to ablate.
let lagunaFusedRoutedSharedSwiGLUQMVEnabled =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_FUSED_ROUTED_SHARED_SWIGLU_QMV"] != "0"

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

/// Sliding-layer per-head RMSNorm + plain RoPE fusion (see
/// `lagunaSlidingQKNormRoPEKernel`).
///
/// **DEFAULT ON, deliberately** (`!= "0"`; set
/// `DARKBLOOM_FUSED_SLIDING_QK_NORM_ROPE=0` to ablate).
///
/// History, because the negative result and its resolution are the
/// instructive part — but note the default is ON today:
///  * Submission `7333473` ranked this fusion at **-0.19%** (1.09995 against
///    a 1.10187 frontier) and it shipped default-off. The diagnosis at the
///    time — "one simdgroup per head is a bad kernel shape" — was wrong.
///  * The actual cause was one redundant line. The kernel parked the inverse
///    RMS in a `threadgroup` slot and issued a `simdgroup_barrier` to
///    broadcast it, but `simd_sum` already returns the total to *every* lane,
///    so each lane can derive the same `precise::rsqrt` locally and
///    bit-identically. At 72 threadgroups per layer across 30 sliding layers
///    that barrier was paid **2160 times per decode token** for nothing, and
///    the full-attention twin had the identical pattern (another 560).
///  * Deleting both and **re-enabling** this fusion measured 10.456 -> 10.326
///    ms steady step (+1.19%, 4/4 pairs) and promoted as `9e06de6` at
///    **1.12019, +1.73%** — the largest single win in the project.
///
/// So the -0.19% figure describes a kernel that no longer exists. Do not
/// spend measurement pairs re-ablating this on the strength of that number.
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
/// Default OFF since the decode fusion-stack audit: with every other lever
/// at default, the atlas measures −0.23% steady decode (se 0.03%, 0/2 ABBA
/// pairs favoring ON, quiescent-machine rig) — the fused kernel's fixed cost
/// now exceeds the two tiny probe dispatches it removes, the same
/// promoted-era-value rot its prefill sibling showed (ranked −0.79%
/// re-land). The OFF path is the verbatim stock fallback below
/// (`embedTokens` gather + `ropeAngleTable` probes), exercised with zero
/// token mismatches in every audit arm. Set `DARKBLOOM_ROPE_ANGLE_ATLAS=1`
/// to re-enable.
let lagunaRoPEAngleAtlasEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ROPE_ANGLE_ATLAS"] == "1"

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
/// pre-widening shape, `32`/`16` for intermediate points): router output rows
/// owned by one threadgroup in `laguna_residual_rms_router_bf16_2048`.
///
/// The router GEMV reads the whole `[256, 2048]` BF16 gate — 1,048,576 B —
/// once per sparse layer. At `64` (16 simdgroups x 4 rows) the 256 rows need
/// `256/64 = 4` threadgroups, and it measures 140.2 GB/s against a 575 GB/s
/// box (`notes/47` §2a): four threadgroups cannot cover the machine's cores.
/// Each halving doubles the tile count at constant total work, 8 -> 32
/// threadgroups. `rows_out` stays 256 in every setting, so no wave
/// quantization hole is created (`notes/50` §6b-§6d).
///
/// Bit-exact. `rows_per_group` changes only WHICH THREADGROUP OWNS WHICH ROW.
/// Every output row keeps its own private FP32 accumulator, its own K-loop
/// over `router_blocks` in `(block, i)` order, its own `simd_shuffle_down`
/// ladder, and one BF16 round. No add is regrouped: the reduction tree exists
/// only at lane level and this knob does not touch it.
let lagunaRouterRowsPerGroup: Int = {
    guard
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_ROUTER_ROWS_PER_GROUP"],
        let value = Int(raw), [8, 16, 32, 64].contains(value)
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
/// MEASURED, `notes/52` (two Latin squares, 66 runs, 66/66 `passed_correctness`,
/// steady step 8..128, all contrasts 6/6 paired). `off` is 10.3735 ms and the
/// previous `ladder8` default 9.4533, so overlap was already worth +9.7%; the
/// remaining prize is 0.15 ms and this default takes essentially all of it:
///
///     ladder8   (5 fires)  1.0000   the promoted default, unswept
///     ladder6   (6 fires)  1.0064
///     ladder2  (20 fires)  1.0169
///     ladder1  (40 fires)  1.0178
///     at:1,7,15,23,31,39   1.0170   <- six fires, ties forty
///
/// `ladderN`'s first fire is at layer `N-1`, so it structurally skips the
/// widest GPU-idle window in the step: the front. Adding ONE rung at layer 1
/// to `ladder8`'s own boundaries is worth as much as quadrupling the ladder,
/// for one extra scheduler round trip instead of thirty-five. The front rung
/// is worthless alone — a lone fire at layer 1 measures 0.9476, the worst
/// schedule tested — and only pays once the rest of the step is covered.
private let lagunaDecodeAsyncStage: LagunaDecodeAsyncStage = {
    let raw =
        ProcessInfo.processInfo.environment["DARKBLOOM_DECODE_ASYNC_STAGE"]?
        .lowercased() ?? "at:1,7,15,23,31,39"
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

/// `DARKBLOOM_PREFILL_ASYNC_LADDER` (default `8`; `0`/`off` disables):
/// prefill-side twin of the decode ladder above. Multi-token forwards build
/// a ~400-op graph with the GPU idle until the final eval; firing `asyncEval`
/// after every Nth layer streams completed segments exactly as the promoted
/// decode ladder does. Same exactness ground: no operation, order, cache
/// write, or token changes — only when already-constructed work is enqueued.
/// This pays into both score components: the prefill phase itself and the
/// 512-token seed prefill charged to the decode window.
private let lagunaPrefillAsyncLadderStride: Int = {
    let raw = ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_ASYNC_LADDER"]?.lowercased() ?? "8"
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
/// A `DARKBLOOM_SHUF_NORM_BCAST` / `DARKBLOOM_SHUF_NORM_INIT` pair once made
/// the two non-gather barriers here optional. Both were measured locally at
/// -0.70% steady step (6/6 pairs, t=4.77, 95% CI excluding zero) and shipped
/// as submission `58864bf4`, which the ranked runner **rejected at -0.07%**.
/// The effect did not exist on the ranked box. Barrier removal has only ever
/// paid in narrow 32-thread kernels (`9e06de6`, +1.73%), where the rendezvous
/// is a large fraction of the kernel; in a 512-thread kernel it is free.
/// Removed rather than left default-OFF so nobody re-derives it.
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
/// Every sparse layer follows this norm with a `[256, 2048]` BF16 GEMV whose
/// only input is the normalized row, so that GEMV is the very next link in the
/// dependency chain and nothing can overlap it. Folding it in costs each
/// threadgroup a redundant 4 KB read of the normalized row it just produced
/// and removes a kernel from the chain.
///
/// Exactness: the router half replicates MLX's gemv for out_vec 256 and in_vec
/// 2048, which selects BM 4, BN 1, SM 1, SN 32, TM 4, TN 4. Lane `l` covers
/// columns `4l + 128i`, products accumulate in `i` then `tn` order in FP32,
/// and the simdgroup reduces with the same `simd_shuffle_down` ladder before
/// one BF16 round. The norm half is untouched.
///
/// `rowsPerGroup` (see `DARKBLOOM_ROUTER_ROWS_PER_GROUP`) chooses only WHICH
/// THREADGROUP OWNS WHICH ROW. 256 divides evenly by 64/32/16/8, every row
/// keeps its own private accumulator and its own `(block, i)` K-loop, and no
/// add is regrouped. **At `rowsPerGroup == 64` this emits the pre-widening
/// kernel** — no guard, no unroll, the same four-element initializer, and
/// `tile * rows_per_group` is the literal 64 the old
/// `tile * (simd_size * rows_per_thread / 2)` folded to. That is what makes
/// `DARKBLOOM_ROUTER_ROWS_PER_GROUP=64` a null by construction and therefore a
/// usable control (`notes/50` §7e).
///
/// Below 16 rows per group there are fewer rows than simdgroups, so
/// `rows_per_thread` bottoms out at 1 and the surplus simdgroups sit out the
/// router phase behind `active_simd_groups`. They still run the norm, which
/// needs all 512 threads, and the guard opens *after* the norm's
/// `threadgroup_barrier` and closes *after* the logit write, so no thread is
/// skipped past a barrier and no row goes unwritten.
///
/// At `rows_per_thread == 1` the block loop is also unrolled four deep. This
/// is the load-level-parallelism half of `notes/50` §6b-ter: `tiles *
/// rows_per_group == 256` at every tiling, so retiling alone cannot add a
/// single outstanding load and leaves in-flight bytes pinned at 64 KB — which
/// is the whole of the measured 140 GB/s. Hoisting four blocks' weight loads
/// takes that to 256 KB.
///
/// **LOADS ONLY.** `router_result[0]` stays a single accumulator stepped in
/// strict `(block, i)` order: block 0's four products, then block 1's, and so
/// on into the same register. Giving each unrolled step its own partial and
/// summing the four at the end would regroup 64 sequential FP32 adds into a
/// tree — bit-exactness lost, every local check still green, the hidden
/// exact-token gate failed. `router_blocks == 16` and `16 % 4 == 0`, so there
/// is no tail. The `normalized_row` coefficients are read inline rather than
/// staged: at one row per thread both cost `n_reads` threadgroup reads per
/// block, so staging would buy nothing and cost 16 registers per unroll step.
private func lagunaResidualRMSNormRouterSource(rowsPerGroup: Int) -> String {
    let simdGroups = 512 / 32
    let rowsPerThread = rowsPerGroup >= simdGroups ? rowsPerGroup / simdGroups : 1
    let activeSimdGroups = rowsPerGroup / rowsPerThread
    let zeros = Array(repeating: "0.0f", count: rowsPerThread).joined(separator: ", ")
    let guardOpen = activeSimdGroups < simdGroups
        ? "        if (simd_group < active_simd_groups) {\n" : ""
    let guardClose = activeSimdGroups < simdGroups ? "        }\n" : ""

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
                router_logits[router_row + r] = bfloat(router_result[r]);
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
        uniqueKeysWithValues: [8, 16, 32, 64].map { rowsPerGroup in
            (
                rowsPerGroup,
                MLXFast.metalKernel(
                    name: "laguna_residual_rms_router_bf16_2048_rpg\(rowsPerGroup)_v2",
                    inputNames: ["residual", "branch", "weight", "router_weight"],
                    outputNames: ["summed", "normalized", "router_logits"],
                    source: lagunaResidualRMSNormRouterSource(rowsPerGroup: rowsPerGroup),
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
    residual: MLXArray, branch: MLXArray, weight: MLXArray, routerWeight: MLXArray
) -> (summed: MLXArray, normalized: MLXArray, routerLogits: MLXArray) {
    let hidden = LagunaConstants.hiddenSize
    let experts = LagunaConstants.numExperts
    precondition(residual.dtype == .bfloat16)
    precondition(branch.dtype == .bfloat16)
    precondition(weight.dtype == .bfloat16)
    precondition(routerWeight.dtype == .bfloat16)
    precondition(residual.shape == [1, 1, hidden])
    precondition(branch.shape == [1, 1, hidden])
    precondition(weight.shape == [hidden])
    precondition(routerWeight.shape == [experts, hidden])

    // `rows_per_group` router rows per threadgroup, so 256 / rows_per_group
    // tiles. Divides exactly for 64/32/16/8 (4/8/16/32 tiles), so no partial
    // tile is dispatched and no row is computed twice or missed. The 512-thread
    // threadgroup and `n_reads == 4` are NOT knobs: they are load-bearing for
    // the `rms_single_row` correspondence (each thread squares its own
    // contiguous four elements), and moving either regroups the FP32 RMS
    // summation and forfeits bit-exactness.
    let rowsPerGroup = lagunaRouterRowsPerGroup
    let tiles = experts / rowsPerGroup
    lagunaTrace("residual+rmsnorm+router")
    let outputs = lagunaResidualRMSNormRouterKernels[rowsPerGroup]!(
        [residual, branch, weight, routerWeight],
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
    precondition(residual.shape == branch.shape)
    precondition(residual.dim(-1) == LagunaConstants.hiddenSize)
    precondition(weight.shape == [LagunaConstants.hiddenSize])

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
    precondition(rawQueries.shape == [1, 1, 48 * LagunaConstants.headDim])
    precondition(rawKeys.shape == [1, 1, 8 * LagunaConstants.headDim])
    precondition(queryWeight.shape == [LagunaConstants.headDim])
    precondition(keyWeight.shape == [LagunaConstants.headDim])
    precondition(angles.dtype == .float32)
    precondition(angles.shape == [1, 1, 1, LagunaConstants.headDim / 2])

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
    precondition(rawQueries.shape == [1, 1, heads * LagunaConstants.headDim])
    precondition(rawKeys.shape == [1, 1, kvHeads * LagunaConstants.headDim])
    precondition(queryWeight.shape == [LagunaConstants.headDim])
    precondition(keyWeight.shape == [LagunaConstants.headDim])
    precondition(angles.dtype == .float32)
    precondition(angles.shape == [1, 1, 1, LagunaConstants.headDim])

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
    precondition(rawQueries.shape == [1, length, heads * LagunaConstants.headDim])
    precondition(rawKeys.shape == [1, length, kvHeads * LagunaConstants.headDim])
    precondition(queryWeight.shape == [LagunaConstants.headDim])
    precondition(keyWeight.shape == [LagunaConstants.headDim])
    precondition(angles.dtype == .float32)
    precondition(
        angles.shape == [1, 1, lagunaRoPEAngleAtlasLength, LagunaConstants.headDim])
    precondition(offsets.dtype == .int32 && offsets.size == 1)
    precondition((heads + kvHeads) % 4 == 0)

    lagunaTrace("prefill sliding qk norm+rope")
    let outputs = lagunaPrefillSlidingQKNormRoPEKernel(
        [rawQueries, rawKeys, queryWeight, keyWeight, angles, offsets],
        grid: ((heads + kvHeads) / 4 * 128, length, 1),
        threadGroup: (128, 1, 1),
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
    precondition(rawQueries.shape == [1, length, heads * LagunaConstants.headDim])
    precondition(rawKeys.shape == [1, length, kvHeads * LagunaConstants.headDim])
    precondition(queryWeight.shape == [LagunaConstants.headDim])
    precondition(keyWeight.shape == [LagunaConstants.headDim])
    precondition(angles.dtype == .float32)
    precondition(
        angles.shape == [1, 1, lagunaRoPEAngleAtlasLength, LagunaConstants.headDim / 2])
    precondition(offsets.dtype == .int32 && offsets.size == 1)
    precondition((heads + kvHeads) % 4 == 0)

    lagunaTrace("prefill full qk norm+yarn")
    let outputs = lagunaPrefillFullQKNormYaRNKernel(
        [rawQueries, rawKeys, queryWeight, keyWeight, angles, offsets],
        grid: ((heads + kvHeads) / 4 * 128, length, 1),
        threadGroup: (128, 1, 1),
        outputShapes: [
            [1, heads, length, LagunaConstants.headDim],
            [1, kvHeads, length, LagunaConstants.headDim],
        ],
        outputDTypes: [.bfloat16, .bfloat16]
    )
    return (outputs[0], outputs[1])
}

/// Decode-only fusion of the three attention input projections into one
/// dispatch. Q, K and V all read the same normalized row and are mutually
/// independent, so MLX already issues them into one barrier group; what this
/// removes is two dispatches per layer, and it does so without the
/// row-concatenated `[Wq; Wk; Wv]` bank behind `DARKBLOOM_FUSED_QKV` — the
/// kernel reads the three stock weights in place, so prefill's GEMM shapes,
/// scheduling and resident memory are all untouched.
///
/// Exactness: MLX's gemv gives every output row its own K loop and its own
/// simdgroup reduction, and the tiling it picks for all three shapes shares
/// SM 1, SN 32, TM 4, TN 4, BN 1 (only BM differs, which just regroups rows
/// across threadgroups). So a row's arithmetic does not depend on which
/// dispatch or which simdgroup computes it: lane `l` covers columns
/// `4l + 128i`, products accumulate in `i` then `tn` order in FP32, and the
/// simdgroup reduces with the same `simd_shuffle_down` ladder before lane 0
/// rounds once to BF16. Row blocks are sized so no simdgroup ever straddles
/// two of the three matrices.
/// The kernel also absorbs the layer's input RMSNorm, which every one of these
/// projections consumes. Each threadgroup recomputes the 2048-element norm
/// from the raw residual row — 4 KB read and one 2048-element reduction
/// against 32 MB of weight traffic — and keeps the normalized row in
/// threadgroup memory for its own K loop, so the norm leaves the dependency
/// chain entirely. The per-head gate projection is folded into this same
/// kernel and also reads the threadgroup row, so no device-visible normalized
/// output is needed.
///
/// The norm reproduces `rms_single_row` at `axis_size == 2048`, `N_READS == 4`
/// and a 512-thread group, which is exactly the shape MLX dispatches for this
/// row: thread `lid` squares its own contiguous four elements in index order,
/// `simd_sum` inside each of the sixteen simdgroups, lane 0 of each writes into
/// `local_sums`, simdgroup 0 `simd_sum`s those, and
/// `precise::rsqrt(acc / 2048 + eps)` is broadcast. The BF16 rounding stays
/// inside `w[i] * bfloat(x[i] * inv)`, which is the value the separate kernel
/// would have written and these projections would have read back.
private func lagunaFusedQKVProjectionSource(heads: Int) -> String {
    """
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
    queries: MLXArray, keys: MLXArray, values: MLXArray, gateValues: MLXArray
)? {
    guard let kernel = lagunaFusedQKVProjectionKernels[heads] else { return nil }
    let hidden = LagunaConstants.hiddenSize
    let queryRows = heads * LagunaConstants.headDim
    let kvRows = LagunaConstants.numKeyValueHeads * LagunaConstants.headDim
    precondition(residual.dtype == .bfloat16)
    precondition(residual.shape == [1, 1, hidden])
    precondition(normWeight.dtype == .bfloat16)
    precondition(normWeight.shape == [hidden])
    precondition(queryWeight.shape == [queryRows, hidden])
    precondition(keyWeight.shape == [kvRows, hidden])
    precondition(valueWeight.shape == [kvRows, hidden])
    precondition(gateWeight.dtype == .bfloat16)
    precondition(gateWeight.shape == [heads, hidden])

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
    return (outputs[0], outputs[1], outputs[2], outputs[3])
}

/// Decode-only fusion of the per-head attention gate with the output
/// projection. The stock decode path is two dispatches: one compiled
/// elementwise kernel that softplus-gates the attention output, and one GEMV
/// over `o_proj`. This kernel folds the gate into the GEMV's vector loads, so
/// the 8192-wide gated row is never materialized and the layer spends one
/// dispatch instead of two.
///
/// Exactness. The fused QKV producer has already reproduced
/// `softplus(gate.asType(.float32)).asType(.bfloat16)` after preserving the
/// projection's intermediate BF16 rounding boundary. This consumer applies
/// the same BF16 gate product as stock. The projection reproduces MLX's
/// `gemv` for this shape exactly: out_vec 2048 and in_vec 8192 select BM 4,
/// BN 1, SM 1, SN 32, TM 4, TN 4, so a thread owns four output rows, lane `l`
/// covers input columns `4l + 128i`, products accumulate in `i` then `tn`
/// order in FP32, and the simdgroup reduces with the same
/// `simd_shuffle_down` ladder (16, 8, 4, 2, 1) before lane 0 rounds once to
/// BF16. Because column `4l + 128i` always lies inside head `i`, the gate a
/// thread needs at step `i` is simply `gate_values[i]`.
/// Depth-2 block unroll, `notes/54` §11: L5 is the only large kernel whose
/// in-flight budget is small enough for memory-level parallelism to bind at
/// all. It holds 512 KB against `lm_head`'s 1280 KB, and at the top of the
/// measured 287–947 ns latency bracket 512 KB supports 554 GB/s against L5's
/// measured 553.8. Hoisting two blocks' loads takes that to 1.05 MB, which
/// clears the 596.1 GB/s fabric ceiling under every calibration in the
/// bracket. If L5 is fabric-bound instead this is flat — a result, not a
/// failure. L5's 0.40 waves are what make the ~20 extra registers free: at
/// 3.2 threadgroups per core against a capacity of 8 there is no occupancy to
/// lose (`notes/46` §6).
///
/// **LOADS ONLY.** `result[row]` stays one accumulator per row, stepped in
/// strict `(block, i)` order — block 0's four products then block 1's, into
/// the same register. Per-unroll partial sums combined at the end would
/// regroup the FP32 chain into a tree and forfeit bit-exactness while passing
/// every local check. `blocks == heads` is 64 or 48, both even, so no tail.
private func lagunaGatedOutputProjectionSource(heads: Int, unroll: Int) -> String {
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
                            const device vec<bfloat, 4>* row_values =
                                (const device vec<bfloat, 4>*)(
                                    weight + (out_row + row) * in_vec_size + column);
                            const vec<bfloat, 4> w = row_values[0];
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
                                const device vec<bfloat, 4>* row_values =
                                    (const device vec<bfloat, 4>*)(
                                        weight + (out_row + row) * in_vec_size +
                                            column_u);
                                weight_values[u][row] = row_values[0];
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

/// `DARKBLOOM_L5_UNROLL` (default `8`; `1` restores the pre-unroll loop
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
        return 8
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
    precondition(attentionOutput.shape == [1, 1, inVec])
    precondition(gateValues.dtype == .bfloat16)
    precondition(gateValues.shape == [1, 1, heads])
    precondition(weight.dtype == .bfloat16)
    precondition(weight.shape == [LagunaConstants.hiddenSize, inVec])

    lagunaTrace("gated output projection h\(heads)")
    return kernel(
        [attentionOutput, gateValues, weight],
        grid: ((LagunaConstants.hiddenSize / 16) * 128, 1, 1),
        threadGroup: (128, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.hiddenSize]],
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

    init(_ config: LagunaConfig, layerIdx: Int) {
        let dim = config.hiddenSize
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
                gateValues: MLXArray
            )?
        if lagunaFusedQKVProjectionEnabled, _fusedQKVWeight == nil,
            B == 1, L == 1,
            headDim == LagunaConstants.headDim,
            nKVHeads == LagunaConstants.numKeyValueHeads,
            input.dtype == .bfloat16,
            input.shape == [1, 1, LagunaConstants.hiddenSize],
            inputNorm.weight.dtype == .bfloat16,
            inputNorm.weight.shape == [LagunaConstants.hiddenSize],
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
            gateProjection.weight.shape == [nHeads, LagunaConstants.hiddenSize]
        {
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
        if let fusedQKVWeight = _fusedQKVWeight {
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
            queries.shape == [1, 1, nHeads * headDim] &&
            keys.shape == [1, 1, nKVHeads * headDim]

        let useFusedFullQKNormYaRN =
            lagunaFusedFullQKNormYaRNEnabled && !isSliding &&
            fusedQKNormShapesMatch &&
            nHeads == LagunaConstants.fullAttentionHeads &&
            qkRoPEAngles?.dtype == .float32 &&
            qkRoPEAngles?.shape == [1, 1, 1, headDim / 2]

        let useFusedSlidingQKNormRoPE =
            lagunaFusedSlidingQKNormRoPEEnabled && isSliding &&
            fusedQKNormShapesMatch &&
            nHeads == LagunaConstants.slidingAttentionHeads &&
            qkRoPEAngles?.dtype == .float32 &&
            qkRoPEAngles?.shape == [1, 1, 1, headDim]

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
            queries.shape == [1, L, nHeads * headDim] &&
            keys.shape == [1, L, nKVHeads * headDim] &&
            qkRoPEAngles?.dtype == .float32 &&
            qkRoPEOffsets?.dtype == .int32 && qkRoPEOffsets?.size == 1

        let usePrefillFusedSlidingQKNormRoPE =
            lagunaPrefillQKNormRoPEEnabled && isSliding &&
            prefillQKNormShapesMatch &&
            nHeads == LagunaConstants.slidingAttentionHeads &&
            qkRoPEAngles?.shape == [1, 1, lagunaRoPEAngleAtlasLength, headDim]

        let usePrefillFusedFullQKNormYaRN =
            lagunaPrefillQKNormRoPEEnabled && !isSliding &&
            prefillQKNormShapesMatch &&
            nHeads == LagunaConstants.fullAttentionHeads &&
            qkRoPEAngles?.shape == [1, 1, lagunaRoPEAngleAtlasLength, headDim / 2]

        var qkNormRoPEFused = false
        if useFusedFullQKNormYaRN, let qkRoPEAngles {
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
        values =
            L == 1
            ? values.reshaped(B, nKVHeads, L, headDim)
            : values.reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)

        if !qkNormRoPEFused {
            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)
        }

        let attended = attentionWithCacheUpdate(
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
                gateIsActivated = true
            } else {
                guard let normalizedInput else {
                    preconditionFailure("attention gate requires normalized input")
                }
                projectedGate = gProj(normalizedInput)
                gateIsActivated = false
            }
            if lagunaFusedGatedOutputProjectionEnabled,
                gateIsActivated, gatePerHead, L == 1, B == 1, wo.bias == nil,
                headDim == LagunaConstants.headDim,
                output.dtype == .bfloat16, projectedGate.dtype == .bfloat16,
                wo.weight.dtype == .bfloat16,
                output.shape == [1, 1, nHeads * headDim],
                projectedGate.shape == [1, 1, nHeads],
                wo.weight.shape == [LagunaConstants.hiddenSize, nHeads * headDim],
                let projection = lagunaGatedOutputProjection(
                    attentionOutput: output,
                    gateValues: projectedGate,
                    weight: wo.weight,
                    heads: nHeads
                )
            {
                return projection
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
        var queries = wq(lastInput)
        var keys = wk(x)
        var values = wv(x)

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
            let projectedGate = gProj(lastInput)
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
/// `laguna_nvfp4_scale` already performs. **−16 scalar multiplies per
/// `qdot_16` call, −16.5% of the dequantize ALU, ~−1104 M float multiplies per
/// token across L8 + L9, and nothing added** (`notes/57` §10).
///
/// Bit-exact. `16384 == 2^14`, and scaling a binary float by an exact power of
/// two touches only the exponent field, so every product, partial sum and
/// rounding decision in the accumulator chain is exactly `2^-14 ×` its old
/// value — same bits, different exponent — and the `2^14` reappears once in
/// the scale before the single final rounding.
///
/// **The dtype move is range-checked, not assumed** (`notes/58` §1a). The
/// multiply moves from half to float because `4194304` overflows half, and a
/// power-of-two argument does NOT by itself survive a dtype change — so all
/// 256 E4M3 scale bytes were enumerated through both paths, in half and in
/// float, with an explicit `isfinite` check on the old path. **Zero
/// divergence, and no overflow is reachable:** the shuffle
/// `(bits & 127) << 7` maps E4M3 into half format, and since E4M3's exponent
/// bias is 7 against half's 15 it already yields the scale divided by 256 —
/// which is exactly what the old `*= 256.0` corrected. The half-domain
/// intermediate therefore peaks at **1.875**, and the scale at **480**,
/// against half's finite max of 65504. **136x headroom.**
///
/// The compiler cannot do this fold itself: `device.cpp:631` sets
/// `setFastMathEnabled(false)`, so reassociating `Σ(a·h·2^14)` into
/// `2^14·Σ(a·h)` is forbidden and all sixteen multiplies really are emitted.
/// `device.cpp` is outside `editablePaths`, so it is done by hand.
///
/// Safe under the `notes/00` kernel-selection rule: this is a pure arithmetic
/// identity **inside our own Metal source**. No shape, dtype, tile count or
/// reduction order that MLX can observe changes, so it cannot alter which
/// kernel MLX selects.
let lagunaNvfp4ScaleFoldEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_NVFP4_SCALE_FOLD"] != "0"
private let lagunaSharedSwiGLUQMVHeader: String = {
    // The two halves of one power-of-two regrouping. They MUST move together:
    // the scale absorbs `2^14` exactly when the weights stop applying it.
    // `4194304.0f == 256 · 16384 == 2^22`.
    let scaleTail =
        lagunaNvfp4ScaleFoldEnabled
        ? "        return float(signed_value) * 4194304.0f;"
        : "        return float(signed_value);"
    let scale256 = lagunaNvfp4ScaleFoldEnabled ? "" : "        converted *= 256.0;\n"
    let weightScale = lagunaNvfp4ScaleFoldEnabled ? "" : " * 16384.0f"
    return """
    static inline float laguna_nvfp4_scale(uint8_t bits) {
        ushort raw = ushort(bits & 127) << 7;
        half converted = as_type<half>(raw);
    \(scale256)    half signed_value = (bits & 128) ? -converted : converted;
    \(scaleTail)
    }

    static inline float laguna_nvfp4_qdot_16(
        const device uint8_t* weight,
        const thread float* input,
        float scale
    ) {
        float accum = 0.0f;
        const device uint2* packed = (const device uint2*)weight;
        const uint2 codes = packed[0];
        for (uint j = 0; j < 2; ++j) {
            const uint c = (j == 0) ? codes.x : codes.y;
            const uint p0 =
                ((c & 0x00070007u) << 9) | ((c & 0x00080008u) << 12);
            const uint p1 =
                ((c & 0x00700070u) << 5) | ((c & 0x00800080u) << 8);
            const uint p2 =
                ((c & 0x07000700u) << 1) | ((c & 0x08000800u) << 4);
            const uint p3 =
                ((c & 0x70007000u) >> 3) | (c & 0x80008000u);
            const float2 v04 = float2(as_type<half2>(p0))\(weightScale);
            const float2 v15 = float2(as_type<half2>(p1))\(weightScale);
            const float2 v26 = float2(as_type<half2>(p2))\(weightScale);
            const float2 v37 = float2(as_type<half2>(p3))\(weightScale);
            accum +=
                (input[8 * j] * v04.x +
                 input[8 * j + 1] * v15.x +
                 input[8 * j + 2] * v26.x +
                 input[8 * j + 3] * v37.x);
            accum +=
                (input[8 * j + 4] * v04.y +
                 input[8 * j + 5] * v15.y +
                 input[8 * j + 6] * v26.y +
                 input[8 * j + 7] * v37.y);
        }
        return scale * accum;
    }
    """
}()

private let lagunaSharedSwiGLUQMVKernel = MLXFast.metalKernel(
    name: "laguna_shared_nvfp4_swiglu_qmv_bf16_v1",
    inputNames: ["input", "fused_weight", "fused_scales"],
    outputNames: ["activated"],
    source: """
        constexpr uint input_width = 2048;
        constexpr uint output_width = 512;
        constexpr uint fused_width = 1024;
        constexpr uint packed_row_bytes = 1024;
        constexpr uint scale_row_bytes = 128;
        constexpr uint block_width = 512;
        constexpr uint values_per_lane = 16;

        uint tile = threadgroup_position_in_grid.x;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint first_row = tile * 4 + simd_group * 2;

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
                uint gate_row = first_row + row;
                uint up_row = gate_row + output_width;
                const device uint8_t* gate_weight =
                    (const device uint8_t*)fused_weight +
                    gate_row * packed_row_bytes + block / 2 + lane * 8;
                const device uint8_t* up_weight =
                    (const device uint8_t*)fused_weight +
                    up_row * packed_row_bytes + block / 2 + lane * 8;
                const device uint8_t* gate_scale =
                    fused_scales + gate_row * scale_row_bytes +
                    block / 16 + lane;
                const device uint8_t* up_scale =
                    fused_scales + up_row * scale_row_bytes +
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

        for (uint row = 0; row < 2; ++row) {
            gate_result[row] = simd_sum(gate_result[row]);
            up_result[row] = simd_sum(up_result[row]);
            if (lane == 0) {
                bfloat gate = bfloat(gate_result[row]);
                bfloat up = bfloat(up_result[row]);
                bfloat exp_abs = metal::exp(metal::abs(gate));
                bfloat denominator = bfloat(1) + exp_abs;
                bfloat y = bfloat(1) / denominator;
                bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
                bfloat silu = bfloat(gate * sigmoid);
                activated[first_row + row] = bfloat(silu * up);
            }
        }
        """,
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)

func lagunaSharedSwiGLUQMV(
    _ input: MLXArray,
    fusedWeight: MLXArray,
    fusedScales: MLXArray
) -> MLXArray {
    precondition(input.dtype == .bfloat16)
    precondition(input.shape == [1, 1, LagunaConstants.hiddenSize])
    precondition(fusedWeight.dtype == .uint32)
    precondition(
        fusedWeight.shape == [
            2 * LagunaConstants.sharedExpertIntermediateSize,
            LagunaConstants.hiddenSize / 8,
        ])
    precondition(fusedScales.dtype == .uint8)
    precondition(
        fusedScales.shape == [
            2 * LagunaConstants.sharedExpertIntermediateSize,
            LagunaConstants.hiddenSize / 16,
        ])

    return lagunaSharedSwiGLUQMVKernel(
        [input, fusedWeight, fusedScales],
        grid: (128 * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.sharedExpertIntermediateSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

private let lagunaSharedDownResidualKernel = MLXFast.metalKernel(
    name: "laguna_shared_nvfp4_down_residual_bf16_v1",
    inputNames: [
        "activated", "down_weight", "down_scales", "routed", "residual",
    ],
    outputNames: ["output"],
    source: """
        constexpr uint input_width = 512;
        constexpr uint output_width = 2048;
        constexpr uint outputs_per_simd = 4;
        constexpr uint values_per_lane = 16;
        constexpr uint packed_row_bytes = 256;
        constexpr uint scale_row_bytes = 32;

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

        thread float result[outputs_per_simd] = {
            0.0f, 0.0f, 0.0f, 0.0f
        };
        for (uint row = 0; row < outputs_per_simd; ++row) {
            uint output_row = first_row + row;
            const device uint8_t* weight =
                (const device uint8_t*)down_weight +
                output_row * packed_row_bytes + lane * 8;
            const device uint8_t* scale =
                down_scales + output_row * scale_row_bytes + lane;
            result[row] = laguna_nvfp4_qdot_16(
                weight,
                input_values,
                laguna_nvfp4_scale(scale[0]));
            result[row] = simd_sum(result[row]);
        }

        if (lane == 0) {
            for (uint row = 0; row < outputs_per_simd; ++row) {
                uint output_row = first_row + row;
                bfloat shared = bfloat(result[row]);
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
    routed: MLXArray,
    residual: MLXArray
) -> MLXArray {
    precondition(activated.dtype == .bfloat16)
    precondition(
        activated.shape == [1, 1, LagunaConstants.sharedExpertIntermediateSize])
    precondition(downWeight.dtype == .uint32)
    precondition(
        downWeight.shape == [
            LagunaConstants.hiddenSize,
            LagunaConstants.sharedExpertIntermediateSize / 8,
        ])
    precondition(downScales.dtype == .uint8)
    precondition(
        downScales.shape == [
            LagunaConstants.hiddenSize,
            LagunaConstants.sharedExpertIntermediateSize / 16,
        ])
    precondition(routed.dtype == .bfloat16)
    precondition(routed.shape == [1, 1, LagunaConstants.hiddenSize])
    precondition(residual.dtype == .bfloat16)
    precondition(residual.shape == [1, 1, LagunaConstants.hiddenSize])

    return lagunaSharedDownResidualKernel(
        [activated, downWeight, downScales, routed, residual],
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

        for (uint row = 0; row < 2; ++row) {
            gate_result[row] = simd_sum(gate_result[row]);
            up_result[row] = simd_sum(up_result[row]);
            if (lane == 0) {
                bfloat gate = bfloat(gate_result[row]);
                bfloat up = bfloat(up_result[row]);
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

func lagunaRoutedSwiGLUQMV(
    _ input: MLXArray,
    fusedWeight: MLXArray,
    fusedScales: MLXArray,
    indices: MLXArray
) -> MLXArray {
    precondition(input.dtype == .bfloat16)
    precondition(input.shape == [1, 1, LagunaConstants.hiddenSize])
    precondition(fusedWeight.dtype == .uint32)
    precondition(
        fusedWeight.shape == [
            LagunaConstants.numExperts,
            2 * LagunaConstants.moeIntermediateSize,
            LagunaConstants.hiddenSize / 8,
        ])
    precondition(fusedScales.dtype == .uint8)
    precondition(
        fusedScales.shape == [
            LagunaConstants.numExperts,
            2 * LagunaConstants.moeIntermediateSize,
            LagunaConstants.hiddenSize / 16,
        ])
    precondition(indices.dtype == .uint32)
    precondition(indices.shape == [1, 1, LagunaConstants.numExpertsPerTok])

    return lagunaRoutedSwiGLUQMVKernel(
        [input, fusedWeight, fusedScales, indices],
        grid: (LagunaConstants.numExpertsPerTok * 128 * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[
            1, 1, LagunaConstants.numExpertsPerTok, 1,
            LagunaConstants.moeIntermediateSize,
        ]],
        outputDTypes: [.bfloat16]
    )[0]
}

/// The routed and shared gate/up QMVs read the same activation row, write
/// different outputs, and share an identical tile shape: 128 tiles of four
/// output rows, two rows per simdgroup, four 512-wide K blocks. They are also
/// independent of each other, so MLX issues them into the same barrier group
/// anyway. Merging them into one nine-slot dispatch (slots 0-7 routed, slot 8
/// shared) removes one dispatch per sparse layer without touching either
/// slot's arithmetic: a threadgroup does exactly the work it did before, over
/// the same bank, in the same order.
private let lagunaRoutedSharedSwiGLUQMVKernel = MLXFast.metalKernel(
    name: "laguna_routed_shared_nvfp4_swiglu_qmv_bf16_v2",
    inputNames: [
        "input", "routed_weight", "routed_scales", "indices",
        "shared_weight", "shared_scales",
    ],
    outputNames: ["routed_activated", "shared_activated"],
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

        // Preserve each expert tile's arithmetic and output address while
        // exposing all eight routed banks plus the shared bank in each
        // scheduling wave.
        uint group = threadgroup_position_in_grid.x;
        uint expert_slot = group % (routed_experts + 1);
        uint tile = group / (routed_experts + 1);
        bool is_routed = expert_slot < routed_experts;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint first_row = tile * 4 + simd_group * 2;

        const device uint8_t* expert_weight;
        const device uint8_t* expert_scales;
        if (is_routed) {
            uint expert = uint(indices[expert_slot]);
            expert_weight =
                (const device uint8_t*)routed_weight +
                expert * packed_expert_bytes;
            expert_scales = routed_scales + expert * scale_expert_bytes;
        } else {
            expert_weight = (const device uint8_t*)shared_weight;
            expert_scales = shared_scales;
        }

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
                uint gate_row;
                uint up_row;
                if (is_routed) {
                    uint pair_tile = logical_row / 32;
                    gate_row = pair_tile * 64 + logical_row % 32;
                    up_row = gate_row + 32;
                } else {
                    gate_row = logical_row;
                    up_row = gate_row + output_width;
                }
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

        for (uint row = 0; row < 2; ++row) {
            gate_result[row] = simd_sum(gate_result[row]);
            up_result[row] = simd_sum(up_result[row]);
            if (lane == 0) {
                bfloat gate = bfloat(gate_result[row]);
                bfloat up = bfloat(up_result[row]);
                bfloat exp_abs = metal::exp(metal::abs(gate));
                bfloat denominator = bfloat(1) + exp_abs;
                bfloat y = bfloat(1) / denominator;
                bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
                bfloat silu = bfloat(gate * sigmoid);
                bfloat activation = bfloat(silu * up);
                if (is_routed) {
                    routed_activated[
                        expert_slot * output_width + first_row + row
                    ] = activation;
                } else {
                    shared_activated[first_row + row] = activation;
                }
            }
        }
        """,
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)

func lagunaRoutedSharedSwiGLUQMV(
    _ input: MLXArray,
    routedWeight: MLXArray,
    routedScales: MLXArray,
    indices: MLXArray,
    sharedWeight: MLXArray,
    sharedScales: MLXArray
) -> (routed: MLXArray, shared: MLXArray) {
    precondition(input.dtype == .bfloat16)
    precondition(input.shape == [1, 1, LagunaConstants.hiddenSize])
    precondition(routedWeight.dtype == .uint32)
    precondition(
        routedWeight.shape == [
            LagunaConstants.numExperts,
            2 * LagunaConstants.moeIntermediateSize,
            LagunaConstants.hiddenSize / 8,
        ])
    precondition(routedScales.dtype == .uint8)
    precondition(
        routedScales.shape == [
            LagunaConstants.numExperts,
            2 * LagunaConstants.moeIntermediateSize,
            LagunaConstants.hiddenSize / 16,
        ])
    precondition(indices.dtype == .uint32)
    precondition(indices.shape == [1, 1, LagunaConstants.numExpertsPerTok])
    precondition(sharedWeight.dtype == .uint32)
    precondition(
        sharedWeight.shape == [
            2 * LagunaConstants.sharedExpertIntermediateSize,
            LagunaConstants.hiddenSize / 8,
        ])
    precondition(sharedScales.dtype == .uint8)
    precondition(
        sharedScales.shape == [
            2 * LagunaConstants.sharedExpertIntermediateSize,
            LagunaConstants.hiddenSize / 16,
        ])

    lagunaTrace("routed+shared gate/up QMV")
    let outputs = lagunaRoutedSharedSwiGLUQMVKernel(
        [input, routedWeight, routedScales, indices, sharedWeight, sharedScales],
        grid: ((LagunaConstants.numExpertsPerTok + 1) * 128 * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [
            [
                1, 1, LagunaConstants.numExpertsPerTok, 1,
                LagunaConstants.moeIntermediateSize,
            ],
            [1, 1, LagunaConstants.sharedExpertIntermediateSize],
        ],
        outputDTypes: [.bfloat16, .bfloat16]
    )
    return (outputs[0], outputs[1])
}

// Decode-only Laguna XS routed down projection. The stock graph materializes
// eight 2048-wide BF16 expert outputs, casts eight FP32 router weights to
// BF16, multiplies, reduces the expert axis, then applies the fixed BF16 2.5
// routed scale. This kernel preserves each of those arithmetic boundaries but
// keeps the eight expert rows in threadgroup memory and emits only the final
// 2048-wide routed branch.
private let lagunaRoutedDownReduceKernel = MLXFast.metalKernel(
    name: "laguna_routed_nvfp4_down_reduce_bf16_v1",
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
        for (uint row = 0; row < outputs_per_simd; ++row) {
            uint output_row = first_row + row;
            const device uint8_t* weight =
                expert_weight + output_row * packed_row_bytes + lane * 8;
            const device uint8_t* scale =
                expert_scales + output_row * scale_row_bytes + lane;
            result[row] = laguna_nvfp4_qdot_16(
                weight,
                input_values,
                laguna_nvfp4_scale(scale[0]));
            result[row] = simd_sum(result[row]);
        }

        threadgroup bfloat expert_outputs[
            experts_per_token * outputs_per_simd
        ];
        if (lane == 0) {
            for (uint row = 0; row < outputs_per_simd; ++row) {
                expert_outputs[
                    expert_slot * outputs_per_simd + row
                ] = bfloat(result[row]);
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
        activated.shape == [
            1, 1, LagunaConstants.numExpertsPerTok, 1,
            LagunaConstants.moeIntermediateSize,
        ])
    precondition(downWeight.dtype == .uint32)
    precondition(
        downWeight.shape == [
            LagunaConstants.numExperts,
            LagunaConstants.hiddenSize,
            LagunaConstants.moeIntermediateSize / 8,
        ])
    precondition(downScales.dtype == .uint8)
    precondition(
        downScales.shape == [
            LagunaConstants.numExperts,
            LagunaConstants.hiddenSize,
            LagunaConstants.moeIntermediateSize / 16,
        ])
    precondition(indices.dtype == .uint32)
    precondition(indices.shape == [1, 1, LagunaConstants.numExpertsPerTok])
    precondition(routerWeights.dtype == .float32)
    precondition(routerWeights.shape == [1, 1, LagunaConstants.numExpertsPerTok])

    return lagunaRoutedDownReduceKernel(
        [activated, downWeight, downScales, indices, routerWeights],
        grid: ((LagunaConstants.hiddenSize / 4) * 256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.hiddenSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

private let lagunaRoutedSharedDownResidualKernel = MLXFast.metalKernel(
    name: "laguna_routed_shared_nvfp4_down_residual_bf16_v1",
    inputNames: [
        "routed_activated", "routed_down_weight", "routed_down_scales",
        "indices", "router_weights", "shared_activated",
        "shared_down_weight", "shared_down_scales", "residual",
    ],
    outputNames: ["output"],
    source: """
        constexpr uint input_width = 512;
        constexpr uint output_width = 2048;
        constexpr uint routed_experts = 8;
        constexpr uint shared_slot = 8;
        constexpr uint outputs_per_simd = 4;
        constexpr uint values_per_lane = 16;
        constexpr uint packed_row_bytes = 256;
        constexpr uint scale_row_bytes = 32;
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
        for (uint row = 0; row < outputs_per_simd; ++row) {
            uint output_row = first_row + row;
            const device uint8_t* weight =
                expert_weight + output_row * packed_row_bytes + lane * 8;
            const device uint8_t* scale =
                expert_scales + output_row * scale_row_bytes + lane;
            result[row] = laguna_nvfp4_qdot_16(
                weight,
                input_values,
                laguna_nvfp4_scale(scale[0]));
            result[row] = simd_sum(result[row]);
        }

        threadgroup bfloat down_outputs[
            (routed_experts + 1) * outputs_per_simd
        ];
        if (lane == 0) {
            for (uint row = 0; row < outputs_per_simd; ++row) {
                down_outputs[slot * outputs_per_simd + row] =
                    bfloat(result[row]);
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
    indices: MLXArray,
    routerWeights: MLXArray,
    sharedActivated: MLXArray,
    sharedDownWeight: MLXArray,
    sharedDownScales: MLXArray,
    residual: MLXArray
) -> MLXArray {
    precondition(routedActivated.dtype == .bfloat16)
    precondition(
        routedActivated.shape == [
            1, 1, LagunaConstants.numExpertsPerTok, 1,
            LagunaConstants.moeIntermediateSize,
        ])
    precondition(routedDownWeight.dtype == .uint32)
    precondition(
        routedDownWeight.shape == [
            LagunaConstants.numExperts,
            LagunaConstants.hiddenSize,
            LagunaConstants.moeIntermediateSize / 8,
        ])
    precondition(routedDownScales.dtype == .uint8)
    precondition(
        routedDownScales.shape == [
            LagunaConstants.numExperts,
            LagunaConstants.hiddenSize,
            LagunaConstants.moeIntermediateSize / 16,
        ])
    precondition(indices.dtype == .uint32)
    precondition(indices.shape == [1, 1, LagunaConstants.numExpertsPerTok])
    precondition(routerWeights.dtype == .float32)
    precondition(routerWeights.shape == [1, 1, LagunaConstants.numExpertsPerTok])
    precondition(sharedActivated.dtype == .bfloat16)
    precondition(
        sharedActivated.shape == [
            1, 1, LagunaConstants.sharedExpertIntermediateSize,
        ])
    precondition(sharedDownWeight.dtype == .uint32)
    precondition(
        sharedDownWeight.shape == [
            LagunaConstants.hiddenSize,
            LagunaConstants.sharedExpertIntermediateSize / 8,
        ])
    precondition(sharedDownScales.dtype == .uint8)
    precondition(
        sharedDownScales.shape == [
            LagunaConstants.hiddenSize,
            LagunaConstants.sharedExpertIntermediateSize / 16,
        ])
    precondition(residual.dtype == .bfloat16)
    precondition(residual.shape == [1, 1, LagunaConstants.hiddenSize])

    return lagunaRoutedSharedDownResidualKernel(
        [
            routedActivated, routedDownWeight, routedDownScales,
            indices, routerWeights, sharedActivated,
            sharedDownWeight, sharedDownScales, residual,
        ],
        grid: ((LagunaConstants.hiddenSize / 4) * 288, 1, 1),
        threadGroup: (288, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.hiddenSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

// MARK: - Layer-0 dense MLP fusion (BF16, no quantization)
//
// Layer 0's `gate_proj`/`up_proj`/`down_proj` are plain BF16 `Linear`, never
// NVFP4 `QuantizedLinear` -- every fused-bank guard above this point casts to
// `QuantizedLinear` and always declines for layer 0, so it fell all the way
// through to four fully separate stock dispatches (gate GEMV, up GEMV,
// `compiledSiluProduct`, down GEMV) plus the decoder layer's own separate
// `h + r2` residual add. The two kernels below close that gap.
//
// `laguna_dense_gate_up_swiglu_bf16_v1` fuses the gate and up projections
// plus the SiLU-gated product into one dispatch (3 dispatches -> 1).
// Exactness: the row loop is the plain-BF16 "projections" tiling already
// shipped and documented on `lagunaFusedQKVProjectionSource` above (SM 1,
// SN 32, TM 4, TN 4, BN 1 -- see that kernel's doc comment) for this exact
// out_vec/in_vec pair: 8192 output rows over a 2048-wide input is precisely
// the sliding-attention Q projection's shape (`slidingAttentionHeads(64) *
// headDim(128) == 8192`, dispatched at lines 932-983 above), so that same row
// loop -- `block_width 128`, `blocks = in_vec_size / 128`, `rows_per_group
// 64`, `rows_per_thread 4`, one `vec<bfloat, 4>` read per block, FP32
// accumulation, the `simd_shuffle_down` ladder (16, 8, 4, 2, 1), one BF16
// round -- is bit-exact here too. The loop is duplicated per output row (one
// pass over `gate_weight` and one over the row-shifted `up_weight` half of
// the same fused bank) so a single shared read of the input row produces both
// the gate row and its matching up row, generalizing the two-lane-per-simdgroup
// pairing already shipped in `lagunaSharedSwiGLUQMVKernel` above (lines
// 1664-1688, `outputs_per_simd == 2`) to this file's standard
// `rows_per_thread == 4`. The SiLU epilogue -- round each accumulator to
// BF16 first (matching stock `gateProj`/`upProj`'s own output rounding), then
// the numerically-stable sigmoid and the BF16 product -- is copied verbatim
// from that same kernel's tail (lines 1691-1704), which is dtype-generic (not
// NVFP4-specific) and already reproduces `compiledSiluProduct`'s exact BF16
// rounding boundary.
//
// `laguna_dense_down_residual_bf16_v1` fuses the down projection with the
// decoder layer's residual add (2 dispatches -> 1). Exactness: the row loop
// is the plain-BF16 row loop already shipped on
// `lagunaGatedOutputProjectionSource` above for the identical out_vec/in_vec
// pair -- 2048 output rows over an 8192-wide input is exactly the
// sliding-attention output projection's shape (in_vec = heads(64) *
// headDim(128) == 8192, dispatched at lines 1064-1117 above) -- minus that
// kernel's per-head gate multiply (layer 0's down projection has no gate to
// fold in), so `block_width 128`, `blocks = in_vec_size / 128`, `rows_per_group
// 16`, `rows_per_thread 4` reproduce the identical MLX gemv tiling for this
// shape. The epilogue mirrors `lagunaSharedDownResidualKernel`'s tail
// rounding order above (lines 1789-1797): round the GEMV accumulator to BF16
// first, matching stock `downProj`'s own output rounding, then add the
// residual and round once more -- reproducing stock `h + r2` bit-for-bit.
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
            for (uint i = 0; i < values_per_thread; ++i) {
                coefficients[i] = float(input[column + i]);
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
    precondition(input.shape == [1, 1, LagunaConstants.hiddenSize])
    precondition(fusedWeight.dtype == .bfloat16)
    precondition(
        fusedWeight.shape == [
            2 * LagunaConstants.denseIntermediateSize, LagunaConstants.hiddenSize,
        ])

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
            for (uint i = 0; i < values_per_thread; ++i) {
                coefficients[i] = float(activated[column + i]);
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
    precondition(activated.shape == [1, 1, LagunaConstants.denseIntermediateSize])
    precondition(downWeight.dtype == .bfloat16)
    precondition(
        downWeight.shape == [
            LagunaConstants.hiddenSize, LagunaConstants.denseIntermediateSize,
        ])
    precondition(residual.dtype == .bfloat16)
    precondition(residual.shape == [1, 1, LagunaConstants.hiddenSize])

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
            gate.weight.shape == up.weight.shape,
            gate.scales.shape == up.scales.shape,
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
        return [fusedWeight, fusedScales]
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
            gateProj.weight.shape == [intermediate, hidden],
            upProj.weight.shape == [intermediate, hidden]
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
        guard let banks = fusedSharedBankGuard(x) else { return nil }
        let activated =
            sharedActivation
            ?? lagunaSharedSwiGLUQMV(
                x,
                fusedWeight: banks.gateUpWeight,
                fusedScales: banks.gateUpScales
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
            x.shape == [1, 1, LagunaConstants.hiddenSize],
            fusedWeight.dtype == .uint32,
            fusedScales.dtype == .uint8,
            _fusedGateUpSplit == LagunaConstants.sharedExpertIntermediateSize,
            down.weight.dtype == .uint32,
            down.weight.shape == [
                LagunaConstants.hiddenSize,
                LagunaConstants.sharedExpertIntermediateSize / 8,
            ],
            down.scales.dtype == .uint8,
            down.scales.shape == [
                LagunaConstants.hiddenSize,
                LagunaConstants.sharedExpertIntermediateSize / 16,
            ]
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
            routed.dtype == .bfloat16,
            routed.shape == [1, 1, LagunaConstants.hiddenSize],
            residual.dtype == .bfloat16,
            residual.shape == [1, 1, LagunaConstants.hiddenSize]
        else {
            return nil
        }

        lagunaTrace("shared down residual")
        return lagunaSharedDownResidual(
            inputs.activated,
            downWeight: inputs.downWeight,
            downScales: inputs.downScales,
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
            x.shape == [1, 1, hidden],
            residual.dtype == .bfloat16,
            residual.shape == [1, 1, hidden],
            type(of: gateProj) == Linear.self,
            type(of: upProj) == Linear.self,
            type(of: downProj) == Linear.self,
            gateProj.bias == nil, upProj.bias == nil, downProj.bias == nil,
            gateProj.weight.dtype == .bfloat16,
            upProj.weight.dtype == .bfloat16,
            downProj.weight.dtype == .bfloat16,
            gateProj.weight.shape == [intermediate, hidden],
            upProj.weight.shape == [intermediate, hidden],
            downProj.weight.shape == [hidden, intermediate]
        else {
            return nil
        }

        let activated: MLXArray
        if lagunaFusedDenseGateUpSwiGLUEnabled,
            let fusedWeight = _fusedDenseGateUpWeight,
            fusedWeight.dtype == .bfloat16,
            fusedWeight.shape == [2 * intermediate, hidden]
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
                x.dtype == .bfloat16,
                x.shape == [1, 1, LagunaConstants.hiddenSize],
                fusedWeight.dtype == .uint32,
                fusedScales.dtype == .uint8,
                _fusedGateUpSplit == LagunaConstants.sharedExpertIntermediateSize
            {
                lagunaTrace("shared gate/up QMV + SwiGLU")
                return downProj(
                    lagunaSharedSwiGLUQMV(
                        x,
                        fusedWeight: fusedWeight,
                        fusedScales: fusedScales
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

private func lagunaDecodeRouterTop8(
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

/// `DARKBLOOM_PREFILL_ROUTER_TOURNAMENT` (default on; set "0" to ablate):
/// credited re-land of saucegod's `aeabc27` two-stage tournament, the
/// mechanism this replaces `lagunaPrefillRouterTop8` above's O(256) per-lane
/// predecessor count with (that one stays in the tree, default off, as its
/// own independent ablation point -- `DARKBLOOM_PREFILL_ROUTER_TOP8=1`).
///
/// Same comparator, same total order, same normalization idiom as the
/// promoted decode router (`laguna_router_key_before`,
/// `lagunaDecodeRouterTop8Header` above) -- reused verbatim, not
/// reimplemented -- but a genuinely cheaper selection network instead of a
/// full 256-element sort or an O(256^2) predecessor count:
///
/// Phase 1 -- eight independent 32-lane bitonic sorts, one per simdgroup.
/// This is exactly the promoted decode kernel's own low-stride bitonic
/// network code (`sequence` from 2 to 32, `stride` from `sequence>>1` down
/// to 1, `simd_shuffle_xor`-only exchanges, identical comparator calls),
/// simply not continued past `sequence == 32`: since `stride <
/// sequence <= 32` throughout, no exchange's `lane ^ stride` ever crosses a
/// 32-lane simdgroup boundary (XORing bits 0-4 cannot flip bit 5), so this
/// is EXACTLY 8 independent, fully-correct bitonic sorts of each
/// simdgroup's own 32-lane block, needing no threadgroup memory. Each
/// block IS fully sorted by the total order after this phase, but NOT all
/// eight ascending: standard Batcher-network direction alternates by block
/// parity at an intermediate stage like this one (needed if the network
/// continued merging into larger blocks, which this one does not) --
/// even-indexed blocks land ascending (rank 0 at `within_block == 0`),
/// odd-indexed blocks land descending (rank 0 at `within_block == 31`).
/// The extraction step below reads each block's true rank-0..7 from
/// whichever end it actually sorted to.
///
/// Exactness of the local-top-8-is-sufficient claim: if an expert `e` is in
/// the row's GLOBAL top-8, it cannot rank below 7 within its own 32-lane
/// block -- if it did, that one block alone would already contain 8
/// experts strictly better than `e` (its within-block betters, all real,
/// all in the same 256-row), giving `e` a global rank of at least 9,
/// contradicting global top-8 membership. So the 8 blocks' local top-8
/// sets (64 candidates total) provably contain the row's true top-8 as a
/// SET, for any partition into blocks -- this holds regardless of block
/// size or which 32 experts land in which block.
///
/// Phase 2 -- repack the 64 candidates into one contiguous threadgroup
/// array (unavoidably a real cross-simdgroup data movement, one barrier)
/// then bitonic-sort THAT 64-element union using the same comparator
/// (`sequence` 2 to 64). All 256 threads participate uniformly (Metal
/// requires uniform control flow to reach a `threadgroup_barrier`); lanes
/// 64-255 operate on a harmless wrapped duplicate of the same 64
/// candidates (`lane & 63`) and are never read. Because a strict total
/// order applied consistently preserves relative order within any subset,
/// the sorted union's first 8 entries are the row's true top-8 IN THE SAME
/// ORDER the full 256-element stable argsort would have produced them --
/// same proof structure the promoted decode kernel and the existing
/// (default-off) `lagunaPrefillRouterTop8` predecessor-count kernel both
/// already rely on for their own exactness arguments.
///
/// The normalizing epilogue reuses the decode kernel's own trick verbatim:
/// after phase 2, ranks 0..<8 are physical lanes 0..<8, all within
/// simdgroup 0, so `simd_shuffle(my_score2, i)` gathers all eight winning
/// scores through registers (no threadgroup memory) and folds them in
/// ascending-lane order -- bit-identical to stock `weights.sum(axis: -1)`'s
/// left fold and the IEEE FP32 divide that follows it.
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

private func lagunaPrefillRouterTournament(
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
        expertOutputs.shape == [
            1, rows, LagunaConstants.numExpertsPerTok, LagunaConstants.hiddenSize,
        ])
    precondition(routerWeights.dtype == .float32)
    precondition(routerWeights.shape == [1, rows, LagunaConstants.numExpertsPerTok])
    precondition(sharedOutput.dtype == .bfloat16)
    precondition(sharedOutput.shape == [1, rows, LagunaConstants.hiddenSize])
    precondition(residual.dtype == .bfloat16)
    precondition(residual.shape == [1, rows, LagunaConstants.hiddenSize])

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
    precondition(routerWeights.shape == [1, rows, LagunaConstants.numExpertsPerTok])
    precondition(sharedOutput.dtype == .bfloat16)
    precondition(sharedOutput.shape == [1, rows, LagunaConstants.hiddenSize])
    precondition(residual.dtype == .bfloat16)
    precondition(residual.shape == [1, rows, LagunaConstants.hiddenSize])

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
    split: Int,
    downProj: SwitchLinear,
    deferUnsort: Bool
) -> (output: MLXArray, inverseOrder: MLXArray?) {
    // SwitchGLU: `var x = MLX.expandedDimensions(x, axes: [-2, -3])`
    var sortedX = MLX.expandedDimensions(x, axes: [-2, -3])
    // SwitchGLU: `let doSort = indices.size >= 64`. The call site already
    // guards `indices.size >= 64` before calling in, so this is always true
    // here; recomputed anyway so this function mirrors SwitchGLU verbatim
    // and stays correct if that guard is ever loosened.
    let doSort = indices.size >= 64
    // SwitchGLU: `var idx = indices` / `var inverseOrder = MLXArray()`
    var idx = indices
    var inverseOrder = MLXArray()
    // SwitchGLU: `if doSort { (x, idx, inverseOrder) = gatherSort(x: x, indices: indices) }`
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
    let gateUp = MLX.gatherQuantizedMM(
        sortedX,
        fusedWeight,
        scales: fusedScales,
        biases: nil,
        rhsIndices: idx,
        transpose: true,
        groupSize: 16,
        bits: 4,
        mode: .nvfp4,
        sortedIndices: doSort
    )
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
    // SwitchGLU: `x = downProj(activated, idx, sortedIndices: doSort)`
    var result = downProj(activated, idx, sortedIndices: doSort)
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
            gateWeight.shape == upWeight.shape,
            gateScales.shape == upScales.shape,
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
        return [fusedWeight, fusedScales]
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
                x.shape == [1, 1, LagunaConstants.hiddenSize],
                fusedWeight.dtype == .uint32,
                fusedScales.dtype == .uint8,
                inds.dtype == .uint32,
                inds.shape == [1, 1, LagunaConstants.numExpertsPerTok],
                _fusedRoutedGateUpSplit == LagunaConstants.moeIntermediateSize
            {
                if lagunaFusedRoutedSharedSwiGLUQMVEnabled,
                    let sharedBanks = sharedExpert.fusedSharedBanks(x)
                {
                    let merged = lagunaRoutedSharedSwiGLUQMV(
                        x,
                        routedWeight: fusedWeight,
                        routedScales: fusedScales,
                        indices: inds,
                        sharedWeight: sharedBanks.gateUpWeight,
                        sharedScales: sharedBanks.gateUpScales
                    )
                    activated = merged.routed
                    mergedSharedActivated = merged.shared
                } else {
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
                let downScales = _routedDownScales,
                let sharedInputs = sharedExpert.fusedSharedDownInputs(
                    x, sharedActivation: mergedSharedActivated),
                activated.dtype == .bfloat16,
                activated.shape == [
                    1, 1, LagunaConstants.numExpertsPerTok, 1,
                    LagunaConstants.moeIntermediateSize,
                ],
                downWeight.dtype == .uint32,
                downWeight.shape == [
                    LagunaConstants.numExperts,
                    LagunaConstants.hiddenSize,
                    LagunaConstants.moeIntermediateSize / 8,
                ],
                downScales.dtype == .uint8,
                downScales.shape == [
                    LagunaConstants.numExperts,
                    LagunaConstants.hiddenSize,
                    LagunaConstants.moeIntermediateSize / 16,
                ],
                weights.dtype == .float32,
                weights.shape == [1, 1, LagunaConstants.numExpertsPerTok],
                routedScalingFactor == Float(LagunaConstants.moeRoutedScalingFactor),
                residual.dtype == .bfloat16,
                residual.shape == [1, 1, LagunaConstants.hiddenSize]
            {
                lagunaTrace("routed+shared down residual")
                return lagunaRoutedSharedDownResidual(
                    routedActivated: activated,
                    routedDownWeight: downWeight,
                    routedDownScales: downScales,
                    indices: inds,
                    routerWeights: weights,
                    sharedActivated: sharedInputs.activated,
                    sharedDownWeight: sharedInputs.downWeight,
                    sharedDownScales: sharedInputs.downScales,
                    residual: residual
                )
            } else if lagunaFusedRoutedDownReduceEnabled,
                let downWeight = _routedDownWeight,
                let downScales = _routedDownScales,
                activated.dtype == .bfloat16,
                activated.shape == [
                    1, 1, LagunaConstants.numExpertsPerTok, 1,
                    LagunaConstants.moeIntermediateSize,
                ],
                downWeight.dtype == .uint32,
                downWeight.shape == [
                    LagunaConstants.numExperts,
                    LagunaConstants.hiddenSize,
                    LagunaConstants.moeIntermediateSize / 8,
                ],
                downScales.dtype == .uint8,
                downScales.shape == [
                    LagunaConstants.numExperts,
                    LagunaConstants.hiddenSize,
                    LagunaConstants.moeIntermediateSize / 16,
                ],
                weights.dtype == .float32,
                weights.shape == [1, 1, LagunaConstants.numExpertsPerTok],
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
                    split: _fusedRoutedGateUpSplit,
                    downProj: downProj,
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
                weights.shape == [1, x.dim(1), LagunaConstants.numExpertsPerTok],
                routedScalingFactor == Float(LagunaConstants.moeRoutedScalingFactor),
                residual.dtype == .bfloat16,
                residual.shape == [1, x.dim(1), LagunaConstants.hiddenSize]
            {
                let sharedOut = sharedExpert(x)
                if sharedOut.dtype == .bfloat16, sharedOut.shape == residual.shape {
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
                weights.shape == [1, x.dim(1), LagunaConstants.numExpertsPerTok],
                routedScalingFactor == Float(LagunaConstants.moeRoutedScalingFactor),
                residual.dtype == .bfloat16,
                residual.shape == [1, x.dim(1), LagunaConstants.hiddenSize]
            {
                let sharedOut = sharedExpert(x)
                if sharedOut.dtype == .bfloat16, sharedOut.shape == residual.shape {
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
            x.shape == [1, 1, LagunaConstants.hiddenSize], x.shape == r.shape,
            let sparse = mlp as? LagunaRuntimeSparseMoEBlock,
            sparse.gate.weight.dtype == .bfloat16,
            sparse.gate.weight.shape == [
                LagunaConstants.numExperts, LagunaConstants.hiddenSize,
            ]
        {
            let fused = lagunaResidualRMSNormRouter(
                residual: x,
                branch: r,
                weight: postAttentionLayerNorm.weight,
                routerWeight: sparse.gate.weight)
            h = fused.summed
            normalized = fused.normalized
            routerLogits = fused.routerLogits
        } else if lagunaFusedResidualRMSNormEnabled,
            x.dtype == .bfloat16, r.dtype == .bfloat16,
            postAttentionLayerNorm.weight.dtype == .bfloat16,
            x.shape == r.shape, x.dim(-1) == LagunaConstants.hiddenSize,
            x.size == LagunaConstants.hiddenSize
        {
            lagunaTrace("residual+rmsnorm")
            (h, normalized) = lagunaResidualRMSNorm(
                residual: x, branch: r, weight: postAttentionLayerNorm.weight)
        } else if lagunaPrefillFusedResidualRMSNormEnabled,
            x.dtype == .bfloat16, r.dtype == .bfloat16,
            postAttentionLayerNorm.weight.dtype == .bfloat16,
            x.shape == r.shape, x.ndim == 3, x.dim(0) == 1,
            x.dim(-1) == LagunaConstants.hiddenSize,
            x.dim(1) > 1
        {
            // Prefill (multi-token) counterpart of the fused decode branch
            // just above: same kernel (`lagunaResidualRMSNorm`), which is
            // already fully row-count-general (its Swift wrapper derives
            // `rows` from `residual.size / hiddenSize` and its grid is
            // `rows` independent threadgroups; the Metal body indexes
            // everything from `threadgroup_position_in_grid.x` and
            // per-threadgroup scratch, with no cross-row state of any
            // kind) -- only the call-site guard above was decode-only
            // (`x.size == hiddenSize` forces a single row). The stock ops
            // it replaces are identical for prefill: with this branch off,
            // `x.dim(1) > 1` always falls through to the same `h = x + r`
            // / `normalized = postAttentionLayerNorm(h)` pair the decode
            // branch already replaces at L == 1, applied row-independently
            // by both the stock ops and the kernel alike (RMSNorm has no
            // cross-token interaction in either form). See
            // `lagunaPrefillFusedResidualRMSNormEnabled`'s doc comment for
            // the full exactness argument.
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
            normalized.shape == [1, 1, LagunaConstants.hiddenSize],
            h.dtype == .bfloat16,
            h.shape == normalized.shape,
            let sparse = mlp as? LagunaRuntimeSparseMoEBlock
        {
            return sparse(normalized, residual: h, routerLogits: routerLogits)
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
            return sparse(normalized, residual: h, routerLogits: routerLogits)
        }
        // Layer-0-only decode fusion: the dense MLP has no
        // `LagunaRuntimeSparseMoEBlock` branch above to catch it, so its
        // residual add was the one MLP-side decode dispatch left completely
        // unfused. `fusedDenseDownResidual` returns `nil` whenever it isn't
        // layer 0's decode shape (or a guard inside it declines), in which
        // case the stock path below runs unchanged.
        if let dense = mlp as? LagunaRuntimeMLP,
            let fused = dense.fusedDenseDownResidual(normalized, residual: h)
        {
            return fused
        }
        let r2 = mlp(normalized)
        return h + r2
    }

    /// Final-layer prefill specialization. Every supplied row still produces
    /// and commits its K/V state, but only the last query/output row proceeds
    /// through attention output projection and the terminal MLP.
    func callLastPrefillRow(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let normalized = inputLayerNorm(x)
        let r = selfAttn.callLastPrefillRow(normalized, cache: cache)
        let h = lagunaLastTokenHidden(x) + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
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
        tokens.shape == [1, 1],
        embeddingWeight.dtype == .bfloat16,
        embeddingWeight.shape == [
            LagunaConstants.vocabSize, LagunaConstants.hiddenSize,
        ],
        fullAtlas.dtype == .float32,
        fullAtlas.shape == [
            1, 1, lagunaRoPEAngleAtlasLength, LagunaConstants.headDim / 2,
        ],
        slidingAtlas.dtype == .float32,
        slidingAtlas.shape == [
            1, 1, lagunaRoPEAngleAtlasLength, LagunaConstants.headDim,
        ],
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
        guard lagunaRoPEAngleAtlasEnabled,
            lagunaFusedFullQKNormYaRNEnabled,
            lagunaFusedSlidingQKNormRoPEEnabled,
            inputs.dtype == .int32,
            inputs.shape == [1, 1],
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
        if let position = decodeRoPEAtlasPosition(inputs: inputs, cache: cache),
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
                    if case .layer(let idx) = lagunaDecodeAsyncStage, idx == i, inputs.shape == [1, 1] {
                        asyncEval(h)
                    }
                    if case .ladder(let n) = lagunaDecodeAsyncStage, (i + 1) % n == 0,
                        inputs.shape == [1, 1]
                    {
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
                if case .layer(let idx) = lagunaDecodeAsyncStage, idx == i, inputs.shape == [1, 1] {
                    asyncEval(h)
                }
                if case .ladder(let n) = lagunaDecodeAsyncStage, (i + 1) % n == 0,
                    inputs.shape == [1, 1]
                {
                    asyncEval(h)
                }
                if case .explicit(let mask) = lagunaDecodeAsyncStage,
                    (mask >> UInt64(i)) & 1 == 1, inputs.shape == [1, 1]
                {
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
        if case .norm = lagunaDecodeAsyncStage, inputs.shape == [1, 1] {
            asyncEval(hidden)
        }

        let result: MLXArray
        if let lmHead {
            if inputs.shape == [1, 1], let pruner = lmHeadPruner {
                // Certified two-pass decode head (notes/68): full BF16 logits
                // row, bit-identical to stock in every argmax-reachable slot.
                result = pruner.logits(hidden: hidden, lmHeadWeight: lmHead.weight)
            } else {
                result = lmHead(hidden)
            }
        } else {
            result = model.embedTokens.asLinear(hidden)
        }
        if case .logits = lagunaDecodeAsyncStage, inputs.shape == [1, 1] {
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
            if lagunaFusedQKVEnabled, let fused = layer.selfAttn.prepareFusedQKVWeight() {
                fusedArrays.append(fused)
            }
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
                eval(pruner.codes, pruner.scales)
                FileHandle.standardError.write(
                    Data("mlxfast: lm_head prune active (mxfp8 coarse copy resident)\n".utf8))
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
