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
let lagunaFusedRoutedGateUpEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_ROUTED_GATE_UP"] != "0"

/// Decode post-attention residual + RMSNorm fusion. The kernel emits
/// both the rounded BF16 residual (needed by the following skip connection)
/// and the normalized row (consumed immediately by the MLP), eliminating a
/// separate residual-add materialization/read. Prefill stays on the stock path
/// to keep its dispatch and scheduling unchanged.
let lagunaFusedResidualRMSNormEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_RESIDUAL_RMS"] != "0"

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
/// DEFAULT OFF: the ranked runner measured it at **-0.19%** (submission
/// `7333473`, 1.09995 against a 1.10187 frontier). It is bit-exact and removes
/// 90 dispatches per decode token, so the loss has to be the kernel itself:
/// one simdgroup per head is 72 threadgroups of 32 threads with a barrier in
/// the middle, against four stock dispatches (`rms_norm` twice, `rope` twice)
/// that are already well shaped at this size. Kept behind the flag rather than
/// deleted because the negative result is the useful part, and a wider variant
/// (several heads per threadgroup) might still pay.
let lagunaFusedSlidingQKNormRoPEEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_SLIDING_QK_NORM_ROPE"] != "0"

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
/// one BF16 round. Sixteen simdgroups of four rows cover the 64 rows this
/// threadgroup owns, and 256 divides evenly by 64. The norm half is untouched.
private let lagunaResidualRMSNormRouterKernel = MLXFast.metalKernel(
    name: "laguna_residual_rms_router_bf16_2048_v2",
    inputNames: ["residual", "branch", "weight", "router_weight"],
    outputNames: ["summed", "normalized", "router_logits"],
    source: """
        constexpr uint axis_size = 2048;
        constexpr uint n_reads = 4;
        constexpr uint simd_size = 32;
        constexpr uint rows_per_thread = 4;
        constexpr uint block_width = 128;
        constexpr uint router_blocks = axis_size / block_width;

        uint tile = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint base = lid * n_reads;

        threadgroup float local_inv_mean[1];
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
        if (simd_lane == 0) {
            local_sums[simd_group] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0) {
            // The threadgroup is always dispatched at 512 threads (16
            // simdgroups), so only local_sums[0..15] are ever written. An
            // in-register 0.0f for lanes 16-31 reproduces the previously
            // zero-filled slots exactly (adding +0.0f leaves every partial
            // in simd_sum's reduction tree bit-identical for these
            // non-negative sums of squares), removing the separate
            // zero-init pass and its barrier without changing any value.
            acc = simd_sum(simd_lane < 16 ? local_sums[simd_lane] : 0.0f);
            if (simd_lane == 0) {
                local_inv_mean[0] =
                    metal::precise::rsqrt(acc / 2048.0f + 1.0e-6f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < n_reads; ++i) {
            bfloat value =
                weight[base + i] *
                bfloat(float(values[i]) * local_inv_mean[0]);
            normalized_row[base + i] = value;
            if (tile == 0) {
                normalized[base + i] = value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- router projection ---
        uint router_row = tile * (simd_size * rows_per_thread / 2) +
            simd_group * rows_per_thread;
        thread float router_result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
        thread float router_input[n_reads];

        uint column = simd_lane * n_reads;
        for (uint block = 0; block < router_blocks; ++block) {
            for (uint i = 0; i < n_reads; ++i) {
                router_input[i] = float(normalized_row[column + i]);
            }
            for (uint r = 0; r < rows_per_thread; ++r) {
                const device vec<bfloat, 4>* row_values =
                    (const device vec<bfloat, 4>*)(
                        router_weight + (router_row + r) * axis_size + column);
                const vec<bfloat, 4> rw = row_values[0];
                for (uint i = 0; i < n_reads; ++i) {
                    router_result[r] += float(rw[i]) * router_input[i];
                }
            }
            column += block_width;
        }

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
        """,
    ensureRowContiguous: true
)

/// Residual add + RMSNorm for the layers whose MLP is not a sparse block
/// (layer 0) and for any shape the router fusion above declines.
private let lagunaResidualRMSNormKernel = MLXFast.metalKernel(
    name: "laguna_residual_rms_bf16_2048_v2",
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

        threadgroup float local_inv_mean[1];
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
        if (simd_lane == 0) {
            local_sums[simd_group] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0) {
            // The threadgroup is always dispatched at 512 threads (16
            // simdgroups), so only local_sums[0..15] are ever written. An
            // in-register 0.0f for lanes 16-31 reproduces the previously
            // zero-filled slots exactly (adding +0.0f leaves every partial
            // in simd_sum's reduction tree bit-identical for these
            // non-negative sums of squares), removing the separate
            // zero-init pass and its barrier without changing any value.
            acc = simd_sum(simd_lane < 16 ? local_sums[simd_lane] : 0.0f);
            if (simd_lane == 0) {
                local_inv_mean[0] =
                    metal::precise::rsqrt(acc / 2048.0f + 1.0e-6f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < n_reads; ++i) {
            normalized[base + i] =
                weight[lid * n_reads + i] *
                bfloat(float(values[i]) * local_inv_mean[0]);
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

    // Sixteen simdgroups of four router rows each; 256 rows / 64 = 4 tiles.
    let tiles = experts / 64
    lagunaTrace("residual+rmsnorm+router")
    let outputs = lagunaResidualRMSNormRouterKernel(
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
/// chain entirely. `normalized` is still emitted (from tile 0) because the
/// per-head gate projection reads it.
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
        constexpr float norm_eps = 1.0e-6f;

        uint tile = threadgroup_position_in_grid.x;
        uint local_id = thread_position_in_threadgroup.x;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        // --- input RMSNorm, mirroring rms_single_row at 512 threads ---
        threadgroup float local_inv_mean[1];
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
        if (lane == 0) {
            local_sums[simd_group] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            // The threadgroup is always dispatched at 512 threads (16
            // simdgroups), so only local_sums[0..15] are ever written. An
            // in-register 0.0f for lanes 16-31 reproduces the previously
            // zero-filled slots exactly (adding +0.0f leaves every partial
            // in simd_sum's reduction tree bit-identical for these
            // non-negative sums of squares), removing the separate
            // zero-init pass and its barrier without changing any value.
            acc = simd_sum(lane < 16 ? local_sums[lane] : 0.0f);
            if (lane == 0) {
                local_inv_mean[0] =
                    metal::precise::rsqrt(acc / float(in_vec_size) + norm_eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < values_per_thread; ++i) {
            bfloat value =
                norm_weight[norm_base + i] *
                bfloat(raw[i] * local_inv_mean[0]);
            normalized_row[norm_base + i] = value;
            if (tile == 0) {
                normalized[norm_base + i] = value;
            }
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
        constexpr uint qkv_tiles =
            (query_rows + 2 * kv_rows) / rows_per_group;

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
                    gate_logits[gate_row + r] = bfloat(total);
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
            name: "laguna_fused_norm_qkv_projection_bf16_h\(heads)_v2",
            inputNames: [
                "residual", "norm_weight", "query_weight", "key_weight",
                "value_weight", "gate_weight",
            ],
            outputNames: ["normalized", "queries", "keys", "values", "gate_logits"],
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
    normalized: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray,
    gateLogits: MLXArray
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
            [1, 1, hidden], [1, 1, queryRows], [1, 1, kvRows], [1, 1, kvRows],
            [1, 1, heads],
        ],
        outputDTypes: [.bfloat16, .bfloat16, .bfloat16, .bfloat16, .bfloat16]
    )
    return (outputs[0], outputs[1], outputs[2], outputs[3], outputs[4])
}

/// Decode-only fusion of the per-head attention gate with the output
/// projection. The stock decode path is two dispatches: one compiled
/// elementwise kernel that softplus-gates the attention output, and one GEMV
/// over `o_proj`. This kernel folds the gate into the GEMV's vector loads, so
/// the 8192-wide gated row is never materialized and the layer spends one
/// dispatch instead of two.
///
/// Exactness. The gate reproduces `softplus(gate.asType(.float32))`, which is
/// MLX's `logAddExp(x, 0)`: `max(x, 0) + log1p(exp(min(x, 0) - max(x, 0)))`,
/// with the same NaN and infinity short-circuits, then the same
/// `.asType(output.dtype)` rounding to BF16 and the same BF16 product with the
/// attention output. The projection reproduces MLX's `gemv` for this shape
/// exactly: out_vec 2048 and in_vec 8192 select BM 4, BN 1, SM 1, SN 32, TM 4,
/// TN 4, so a thread owns four output rows, lane `l` covers input columns
/// `4l + 128i`, products accumulate in `i` then `tn` order in FP32, and the
/// simdgroup reduces with the same `simd_shuffle_down` ladder (16, 8, 4, 2, 1)
/// before lane 0 rounds once to BF16. Because column `4l + 128i` always lies
/// inside head `i`, the gate a thread needs at step `i` is simply `gates[i]`.
private func lagunaGatedOutputProjectionSource(heads: Int) -> String {
    """
        constexpr uint in_vec_size = \(heads * LagunaConstants.headDim);
        constexpr uint heads = \(heads);
        constexpr uint head_dim = 128;
        constexpr uint rows_per_thread = 4;
        constexpr uint values_per_thread = 4;
        constexpr uint block_width = 128;
        constexpr uint blocks = in_vec_size / block_width;
        constexpr uint rows_per_group = 16;

        uint tile = threadgroup_position_in_grid.x;
        uint local_id = thread_position_in_threadgroup.x;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        // softplus(x) == logaddexp(x, 0), rounded to BF16 exactly where
        // `.asType(output.dtype)` rounds it.
        threadgroup bfloat gates[heads];
        if (local_id < heads) {
            float logit = float(gate_logits[local_id]);
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
            gates[local_id] = bfloat(gate);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint out_row = tile * rows_per_group + simd_group * rows_per_thread;
        thread float result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
        thread float coefficients[values_per_thread];

        uint column = lane * values_per_thread;
        for (uint block = 0; block < blocks; ++block) {
            // Column `4 * lane + 128 * block` sits in head `block`.
            float gate = float(gates[block]);
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

private let lagunaGatedOutputProjectionKernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        kernels[heads] = MLXFast.metalKernel(
            name: "laguna_gated_output_projection_bf16_h\(heads)_v1",
            inputNames: ["attention_output", "gate_logits", "weight"],
            outputNames: ["projected"],
            source: lagunaGatedOutputProjectionSource(heads: heads),
            ensureRowContiguous: true
        )
    }
    return kernels
}()

func lagunaGatedOutputProjection(
    attentionOutput: MLXArray, gateLogits: MLXArray, weight: MLXArray, heads: Int
) -> MLXArray? {
    guard let kernel = lagunaGatedOutputProjectionKernels[heads] else { return nil }
    let inVec = heads * LagunaConstants.headDim
    precondition(attentionOutput.dtype == .bfloat16)
    precondition(attentionOutput.shape == [1, 1, inVec])
    precondition(gateLogits.dtype == .bfloat16)
    precondition(gateLogits.shape == [1, 1, heads])
    precondition(weight.dtype == .bfloat16)
    precondition(weight.shape == [LagunaConstants.hiddenSize, inVec])

    lagunaTrace("gated output projection h\(heads)")
    return kernel(
        [attentionOutput, gateLogits, weight],
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
        qkRoPEAngles: MLXArray? = nil
    ) -> MLXArray {
        let (B, L) = (input.dim(0), input.dim(1))

        // One dispatch for the input RMSNorm and all three projections when
        // the decode preconditions hold; otherwise normalize separately and
        // fall through to the stock projections below.
        var fusedNormQKV:
            (
                normalized: MLXArray, queries: MLXArray, keys: MLXArray,
                values: MLXArray, gateLogits: MLXArray
            )?
        if lagunaFusedQKVProjectionEnabled, B == 1, L == 1,
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
        let x = fusedNormQKV?.normalized ?? inputNorm(input)

        var queries: MLXArray
        var keys: MLXArray
        var values: MLXArray
        if let fusedQKVWeight = _fusedQKVWeight {
            // One dispatch over the row-concatenated [Wq; Wk; Wv] weight,
            // identical math to the three bias-free `Linear` calls
            // (`matmul(x, w.T)`). Each output row's K-loop is independent of
            // which rows share the dispatch, so every Q/K/V element is
            // bit-exact; the slices are views and the reshapes below may
            // copy, which does not change values.
            let qkv = matmul(x, fusedQKVWeight.T)
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
            queries = wq(x)
            keys = wk(x)
            values = wv(x)
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

        if useFusedFullQKNormYaRN, let qkRoPEAngles {
            (queries, keys) = lagunaFullQKNormYaRN(
                rawQueries: queries,
                rawKeys: keys,
                queryWeight: qNorm.weight,
                keyWeight: kNorm.weight,
                angles: qkRoPEAngles
            )
        } else if useFusedSlidingQKNormRoPE, let qkRoPEAngles {
            (queries, keys) = lagunaSlidingQKNormRoPE(
                rawQueries: queries,
                rawKeys: keys,
                queryWeight: qNorm.weight,
                keyWeight: kNorm.weight,
                angles: qkRoPEAngles
            )
        } else {
            queries =
                qNorm(queries.reshaped(B, L, nHeads, headDim))
                .transposed(0, 2, 1, 3)
            keys =
                kNorm(keys.reshaped(B, L, nKVHeads, headDim))
                .transposed(0, 2, 1, 3)
        }
        values = values.reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)

        if !useFusedFullQKNormYaRN && !useFusedSlidingQKNormRoPE {
            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)
        }

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        if gatingEnabled, let gProj {
            // Per-head softplus gate computed in float32, then broadcast
            // across the head dimension (or applied elementwise for a
            // per-element gate).
            let projectedGate = fusedNormQKV?.gateLogits ?? gProj(x)
            if lagunaFusedGatedOutputProjectionEnabled,
                gatePerHead, L == 1, B == 1, wo.bias == nil,
                headDim == LagunaConstants.headDim,
                output.dtype == .bfloat16, projectedGate.dtype == .bfloat16,
                wo.weight.dtype == .bfloat16,
                output.shape == [1, 1, nHeads * headDim],
                projectedGate.shape == [1, 1, nHeads],
                wo.weight.shape == [LagunaConstants.hiddenSize, nHeads * headDim],
                let projection = lagunaGatedOutputProjection(
                    attentionOutput: output,
                    gateLogits: projectedGate,
                    weight: wo.weight,
                    heads: nHeads
                )
            {
                return projection
            }
            if gatePerHead && projectedGate.dtype == output.dtype,
                L == 1, wo.bias == nil, MLXHardwareInfo.isCompiledDecodeSupported
            {
                return attentionGateProjection(output, projectedGate, wo.weight)
            }
            let gate =
                gatePerHead && projectedGate.dtype == output.dtype
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

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: .causal
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, 1, -1)

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

private let lagunaSharedSwiGLUQMVHeader = """
    static inline float laguna_nvfp4_scale(uint8_t bits) {
        ushort raw = ushort(bits & 127) << 7;
        half converted = as_type<half>(raw);
        converted *= 256.0;
        half signed_value = (bits & 128) ? -converted : converted;
        return float(signed_value);
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
            const float2 v04 = float2(as_type<half2>(p0)) * 16384.0f;
            const float2 v15 = float2(as_type<half2>(p1)) * 16384.0f;
            const float2 v26 = float2(as_type<half2>(p2)) * 16384.0f;
            const float2 v37 = float2(as_type<half2>(p3)) * 16384.0f;
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
    name: "laguna_routed_nvfp4_swiglu_qmv_bf16_v1",
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

        uint group = threadgroup_position_in_grid.x;
        uint expert_slot = group / tiles_per_expert;
        uint tile = group % tiles_per_expert;
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
                uint gate_row = first_row + row;
                uint up_row = gate_row + output_width;
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
    name: "laguna_routed_shared_nvfp4_swiglu_qmv_bf16_v1",
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

        uint group = threadgroup_position_in_grid.x;
        uint expert_slot = group / tiles_per_expert;
        uint tile = group % tiles_per_expert;
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
                uint gate_row = first_row + row;
                uint up_row = gate_row + output_width;
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

        return lagunaSharedDownResidual(
            inputs.activated,
            downWeight: inputs.downWeight,
            downScales: inputs.downScales,
            routed: routed,
            residual: residual
        )
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
                    total = scores[i] + total;
                }
                router_scores[lane] = scores[lane] / total;
        """
        : """
                router_scores[lane] = scores[lane];
        """
    return """
        uint lane = thread_position_in_threadgroup.x;

        threadgroup float scores[256];
        threadgroup float choice_keys[256];
        threadgroup uint expert_indices[256];

        float x = float(logits[lane]);
        float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
        float score = x < 0.0f ? y : 1.0f - y;
        scores[lane] = score;
        float corrected = score + float(correction_bias[lane]);
        choice_keys[lane] = -corrected;
        expert_indices[lane] = lane;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A total order (choice key, then original expert index) makes this
        // network match the stock stable merge sort even for exact ties,
        // signed zero, and NaNs. The lower half of each final sequence keeps
        // the better entries, so ranks 0..<8 are the desired top experts.
        for (uint sequence = 2; sequence <= 256; sequence <<= 1) {
            for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
                uint partner = lane ^ stride;
                if (partner > lane) {
                    float a_key = choice_keys[lane];
                    uint a_index = expert_indices[lane];
                    float a_score = scores[lane];
                    float b_key = choice_keys[partner];
                    uint b_index = expert_indices[partner];
                    float b_score = scores[partner];

                    bool lower_wants_better = (lane & sequence) == 0;
                    bool b_before_a = laguna_router_key_before(
                        b_key, b_index, a_key, a_index);
                    bool a_before_b = laguna_router_key_before(
                        a_key, a_index, b_key, b_index);
                    bool swap = lower_wants_better ? b_before_a : a_before_b;
                    if (swap) {
                        choice_keys[lane] = b_key;
                        expert_indices[lane] = b_index;
                        scores[lane] = b_score;
                        choice_keys[partner] = a_key;
                        expert_indices[partner] = a_index;
                        scores[partner] = a_score;
                    }
                }
                // `partner = lane ^ stride` never leaves the calling
                // simdgroup while `stride < 32` (only bits 0-4 flip), so
                // those stages' compare-swaps touch only the calling
                // simdgroup's own disjoint 32-slot range of choice_keys /
                // expert_indices / scores; a simdgroup-scoped fence is
                // sufficient there. Stages with stride >= 32 cross the
                // 32-lane boundary and keep the full threadgroup barrier.
                // The comparator, swap predicate, and total order are
                // unchanged, so the selected ranks are identical.
                if (stride < 32) {
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                } else {
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }

        if (lane < 8) {
            router_indices[lane] = expert_indices[lane];
        \(epilogue)
        }
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
            let sinkNormalization = normTopkProb && lagunaDecodeRouterNormSinkEnabled
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

final class LagunaRuntimeSparseMoEBlock: Module, UnaryLayer {
    let routedScalingFactor: Float

    @ModuleInfo(key: "gate") var gate: LagunaRuntimeMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: LagunaRuntimeMLP

    /// Retained fused NVFP4 `[gate; up]` routed-expert banks (per-expert
    /// output rows concatenated, gate rows first), built once after
    /// checkpoint load when `DARKBLOOM_FUSED_ROUTED_GATE_UP` is enabled, plus
    /// a reference to the stock `switch_mlp.down_proj` module for the fused
    /// decode path. Plain stored properties with a leading underscore so
    /// Module reflection never treats the derived layout as checkpoint
    /// parameters or a second child module; `switchMLP` keeps the original
    /// separate banks (they still serve every multi-token forward and
    /// parameter integrity).
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
        let fusedWeight = concatenated([gateWeight, upWeight], axis: 1)
        let fusedScales = concatenated([gateScales, upScales], axis: 1)
        _fusedRoutedGateUpWeight = fusedWeight
        _fusedRoutedGateUpScales = fusedScales
        _fusedRoutedGateUpSplit = gateWeight.dim(1)
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
                let xGate = gateUp[.ellipsis, 0 ..< _fusedRoutedGateUpSplit]
                let xUp = gateUp[.ellipsis, _fusedRoutedGateUpSplit...]
                activated = compiledSiluProduct(xGate, xUp)
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
            y = switchMLP(x, inds)
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
        qkRoPEAngles: MLXArray? = nil
    ) -> MLXArray {
        let r = selfAttn(
            x,
            inputNorm: inputLayerNorm,
            mask: mask,
            cache: cache,
            qkRoPEAngles: qkRoPEAngles
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
        var h = embedTokens(inputs)

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
        let isSingleTokenDecode = h.dim(0) == 1 && h.dim(1) == 1
        let fullRoPEAngles = lagunaFusedFullQKNormYaRNEnabled && isSingleTokenDecode
            ? ropeAngleTable(
                seed: _fullRoPEAngleSeed,
                attention: layers[fullAttentionIdx].selfAttn,
                cache: cache?[fullAttentionIdx])
            : nil
        let slidingRoPEAngles = lagunaFusedSlidingQKNormRoPEEnabled && isSingleTokenDecode
            ? ropeAngleTable(
                seed: _slidingRoPEAngleSeed,
                attention: layers[slidingAttentionIdx].selfAttn,
                cache: cache?[slidingAttentionIdx])
            : nil

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
                        qkRoPEAngles: qkRoPEAngles
                    )
                }
            } else {
                h = layer(
                    h,
                    mask: mask,
                    cache: cache?[i],
                    qkRoPEAngles: qkRoPEAngles
                )
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
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
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
        var fusedArrays: [MLXArray] = []
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
            }
        }
        if !fusedArrays.isEmpty {
            eval(fusedArrays)
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
