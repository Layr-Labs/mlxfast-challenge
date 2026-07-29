import Foundation
import MLX
import MLXFast

// Certified two-pass lm_head elision for the final-token projection (notes/68).
//
// Stock lm_head reads the full BF16 [100352, 2048] weight (411 MB) for the
// final hidden row at the DRAM wall. This module, gated by
// DARKBLOOM_LM_HEAD_PRUNE (DEFAULT ON; set "0" to disable; unset = shipped
// path), replaces it for
// both prefill's already-sliced last hidden row and single-token decode with:
//
//   1. COARSE pass (`lagunaLmHeadCoarseKernel`): one fused GEMV over an
//      init-time MXFP8 copy of lm_head (gs32 e8m0+e4m3, 211.9 MB) built with
//      the repo's own `quantized(..., mode: .mxfp8)`, producing per-row coarse
//      logit c_i, a certified bound delta_i, and a BF16 pre-fill of the output
//      row. delta_i = d_i*(1+gamma) + 2*gamma*m_i with
//      d_i = sum_g sd_g * sum_{j in g} |x_j| * hs8(code_ij)
//          >= sum_j |x_j| * |w_ij - what_ij|   (half-ulp cells, top cell 186)
//      and m_i = sum_j |x_j| * |what_ij|, so delta_i covers BOTH the
//      quantization error and both kernels' float rounding (depth <= 96
//      roundings/element-path << gamma = 2^-15 relative; notes/68 section 6).
//      The e4m3/e8m0 decoders below are bit-exact replicas of the vendored
//      fp8.h / fp_quantized.h semantics (no libm: exponent-bit construction).
//   2. SELECTION (`lagunaLmHeadSelectKernel`): a dense one-byte-per-row mask
//      marking rows with c_i + delta_i >= max_j(c_j - delta_j) - beta,
//      beta = |L| / 64 (>= 2 BF16 ulps at the logit scale, so a
//      non-candidate's BF16(coarse) is PROVABLY strictly below the winner's
//      BF16 value). No host readback and no atomics: the exact pass keys on
//      "is row r a candidate?", so a mask is both cheaper and race-free.
//   3. EXACT pass (`lagunaLmHeadExactKernel`): each simdgroup owns a FIXED
//      block of four output rows and runs a full BF16 GEMV over that block
//      only when one of its rows is marked, writing coarse values otherwise.
//      The per-row arithmetic is a TEXTUAL replica of the stock
//      `gemv_al_bfloat16` (bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0; see gemv.h
//      GEMVKernel::run with kAligned=true) -- same lane partition, same
//      sequential f32 order, same vec4 loads, same simd tree, same BF16 cast
//      -- and, because the row-to-thread mapping is the stock one rather than
//      an indirection, each candidate row's output is bit-identical to the
//      stock full GEMV's (R1). Every vocabulary slot is written by exactly one
//      lane on exactly one path, so the row is fully covered with no race and
//      no uninitialized slot. Non-candidate slots keep the BF16 coarse value,
//      which the certificate shows is strictly below the winner; the harness
//      argmaxes the returned row (LagunaCorrectness.swift:108), so the emitted
//      token is the stock token.
// The threshold beta widens the candidate set slightly vs the raw lower bound
// L; it is the BF16-cast safety margin from the assembly proof.

private let lagunaLmHeadPruneVocab = 100_352
private let lagunaLmHeadPruneHidden = 2048

/// Master switch for the certified two-pass final-row lm_head (notes/68).
/// DEFAULT ON: unset, or any value other than "0", enables the certified
/// two-pass head and builds the MXFP8 coarse copy at init time.
/// Set `DARKBLOOM_LM_HEAD_PRUNE=0` to disable and restore the byte-identical
/// stock full lm_head pass.
let lagunaLmHeadPruneEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LM_HEAD_PRUNE"] != "0"

/// Same-binary A/B switch for applying the certified pruner to prefill's
/// already-sliced final hidden row. DEFAULT ON; set
/// `DARKBLOOM_LM_HEAD_PRUNE_PREFILL=0` to restore the stock prefill head
/// without disabling the existing decode pruner.
let lagunaLmHeadPrunePrefillEnabled =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_LM_HEAD_PRUNE_PREFILL"] != "0"

/// Kernel header: bit-exact MXFP8 element decoders + the certified
/// half-cell-width table, all inlinable and libm-free.
private let lagunaLmHeadPruneHeader = """
    // e4m3fn decode, identical to fp8.h:32-38 (half bit pattern (b&127)<<7,
    // times 256, sign from bit 7). Exact in half/float for all 256 codes.
    static inline float laguna_e4m3_decode(uint8_t b) {
        half converted = as_type<half>(ushort(uint(b & 127u) << 7));
        converted = converted * (half)256.0f;
        return (b & 128u) ? -float(converted) : float(converted);
    }

    // e8m0 decode, identical to fp8.h:70-77 (bits<<7 as bf16; bits==0 ->
    // 0x40 as bf16 = 2^-127). Exponent-bit construction, exact.
    static inline float laguna_e8m0_decode(uint8_t b) {
        if (b == 0u) {
            return as_type<float>(0x00400000u);  // 2^-127
        }
        return as_type<float>(uint(b) << 23);
    }

    // Certified |ratio - code| bound for an e4m3 element: half the enclosing
    // RNE cell (denormal half-ulp 2^-10; normal half-ulp 2^(e-11)), except the
    // saturated top code 0x7E whose cell is open: the e8m0 scale may round
    // down by up to a factor 2^0.5, so ratio <= 448*2^0.5 and the bound is
    // 448*(2^0.5-1) = 185.6, rounded up to 186.
    //
    // Max-form: the denormal branch 2^-10 equals 2^(1-11), so both non-top
    // cases collapse to 2^(max(e,1)-11) -- identical float for all 256 codes
    // to the original three-branch form (e==0 -> 2^-10; e>0 -> 2^(e-11)).
    static inline float laguna_hs8(uint8_t b) {
        uint mag = uint(b) & 127u;
        uint e = mag >> 3;
        float h = as_type<float>((metal::max(e, 1u) + 116u) << 23);  // 2^(max(e,1)-11)
        return (mag == 126u) ? 186.0f : h;
    }

    // Bit-parallel e4m3 decode of one packed word (4 codes) into 4 floats.
    // Per byte b the half bit pattern is sign<<15 | (b&127)<<7, i.e. exactly
    // fp8.h's (b&127)<<7 construction with the sign applied as the half sign
    // bit instead of a post-float negate. IEEE multiply is sign-magnitude
    // symmetric, so (sign-packed half)*256h == sign*((b&127)-half * 256h)
    // bit-for-bit for every code, including -0 (code 0x80). The four decoded
    // floats are byte-order b0,b1,b2,b3 in out.x,out.y,out.z,out.w.
    static inline float4 laguna_e4m3_decode4(uint w) {
        uint lo = ((w & 0x007F007Fu) << 7) | ((w & 0x00800080u) << 8);
        uint hs = w >> 8;
        uint hi = ((hs & 0x007F007Fu) << 7) | ((hs & 0x00800080u) << 8);
        half2 h02 = as_type<half2>(lo) * half2((half)256.0f);
        half2 h13 = as_type<half2>(hi) * half2((half)256.0f);
        return float4(float(h02.x), float(h13.x), float(h02.y), float(h13.y));
    }
    """

/// Fused MXFP8 coarse GEMV + certified bound + BF16 pre-fill.
/// One simdgroup per row; lane covers 64 consecutive elements (2 groups).
///
/// v4 preserves the v2 arithmetic and row ownership while packing 28
/// independent SIMD rows into each 896-thread threadgroup. The vocabulary
/// size is exactly divisible by 28, so this removes threadgroups without a
/// tail branch. The lane mapping, FP accumulation text, reduction order,
/// vectorized decode, and every output bit remain unchanged.
private func lagunaLmHeadCoarseKernelSource(rowsPerThreadgroup: Int) -> String {
    """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * \(rowsPerThreadgroup) +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint8_t* crow = codes + size_t(row) * 2048;
        const device uint8_t* srow = scales + size_t(row) * 64;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        float m_acc = 0.0f;
        for (uint gg = 0; gg < 2; ++gg) {
            uint g = 2 * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            const device uint4* cptr = (const device uint4*)(crow + g * 32);
            uint4 packed0 = cptr[0];
            uint4 packed1 = cptr[1];
            float cg = 0.0f;
            float dg = 0.0f;
            float mg = 0.0f;
            const device ushort4* xrow = (const device ushort4*)(x + g * 32);
            #pragma clang loop unroll(full)
            for (uint w = 0; w < 8; ++w) {
                uint word = (w < 4u) ? packed0[w & 3u] : packed1[w & 3u];
                float4 cv4 = laguna_e4m3_decode4(word);
                // bf16 -> f32 is exactly bits<<16 for every value class.
                float4 xv4 = as_type<float4>(uint4(xrow[w]) << 16);
                float4 ax4 = metal::abs(xv4);
                uint4 b4 = (uint4(word) >> uint4(0u, 8u, 16u, 24u)) & 255u;
                uint4 mag4 = b4 & 127u;
                uint4 e4 = mag4 >> 3;
                float4 hsf = as_type<float4>((metal::max(e4, uint4(1u)) + 116u) << 23);
                float4 hs4 = metal::select(hsf, float4(186.0f), mag4 == 126u);
                float4 acv4 = metal::abs(cv4);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4; ++k) {
                    float cv = cv4[k];
                    float xv = xv4[k];
                    float ax = ax4[k];
                    cg += xv * cv;
                    dg += ax * hs4[k];
                    mg += ax * acv4[k];
                }
            }
            c_acc += sd * cg;
            d_acc += sd * dg;
            m_acc += sd * mg;
        }
        c_acc = simd_sum(c_acc);
        d_acc = simd_sum(d_acc);
        m_acc = simd_sum(m_acc);
        if (lane == 0) {
            coarse[row] = c_acc;
            delta[row] = d_acc * (1.0f + GAMMA) + (2.0f * GAMMA) * m_acc;
            coarse_bf[row] = bfloat(c_acc);
        }
        """
}

private let lagunaLmHeadCoarseKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_mxfp8_coarse_pack28_v4",
    inputNames: ["x", "codes", "scales"],
    outputNames: ["coarse", "delta", "coarse_bf"],
    source: lagunaLmHeadCoarseKernelSource(rowsPerThreadgroup: 28),
    header: lagunaLmHeadPruneHeader,
    ensureRowContiguous: true
)

private let lagunaLmHeadCoarseKernelPack16 = MLXFast.metalKernel(
    name: "laguna_lmhead_mxfp8_coarse_pack16_v3",
    inputNames: ["x", "codes", "scales"],
    outputNames: ["coarse", "delta", "coarse_bf"],
    source: lagunaLmHeadCoarseKernelSource(rowsPerThreadgroup: 16),
    header: lagunaLmHeadPruneHeader,
    ensureRowContiguous: true
)
/// v1 coarse kernel, kept verbatim for same-binary A/B (the paired
/// measurement protocol requires both arms in one binary). Selected by
/// `DARKBLOOM_LMHEAD_COARSE=v1`; the shipped default is pack28 above. The two
/// kernels are bit-identical in all three outputs for every input.
private let lagunaLmHeadCoarseKernelV1 = MLXFast.metalKernel(
    name: "laguna_lmhead_mxfp8_coarse_v1",
    inputNames: ["x", "codes", "scales"],
    outputNames: ["coarse", "delta", "coarse_bf"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 8 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint8_t* crow = codes + size_t(row) * 2048;
        const device uint8_t* srow = scales + size_t(row) * 64;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        float m_acc = 0.0f;
        for (uint gg = 0; gg < 2; ++gg) {
            uint g = 2 * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            const device uint4* cptr = (const device uint4*)(crow + g * 32);
            uint4 packed0 = cptr[0];
            uint4 packed1 = cptr[1];
            float cg = 0.0f;
            float dg = 0.0f;
            float mg = 0.0f;
            for (uint j = 0; j < 32; ++j) {
                uint word = (j < 16) ? packed0[j / 4] : packed1[(j - 16) / 4];
                uint8_t b = uint8_t(word >> (8 * (j % 4)));
                float cv = laguna_e4m3_decode(b);
                float xv = float(x[g * 32 + j]);
                float ax = metal::abs(xv);
                cg += xv * cv;
                dg += ax * laguna_hs8(b);
                mg += ax * metal::abs(cv);
            }
            c_acc += sd * cg;
            d_acc += sd * dg;
            m_acc += sd * mg;
        }
        c_acc = simd_sum(c_acc);
        d_acc = simd_sum(d_acc);
        m_acc = simd_sum(m_acc);
        if (lane == 0) {
            coarse[row] = c_acc;
            delta[row] = d_acc * (1.0f + GAMMA) + (2.0f * GAMMA) * m_acc;
            coarse_bf[row] = bfloat(c_acc);
        }
        """,
    header: lagunaLmHeadPruneHeader,
    ensureRowContiguous: true
)

/// Same-binary A/B selector for the coarse kernel (v2 default).
private let lagunaLmHeadCoarseUseV1 =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_COARSE"] == "v1"

private let lagunaLmHeadCoarseUsePack16 =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_COARSE"] == "pack16"

/// `lower.max()` uses MLX's two-pass `all_reduce_max` for this 100352-element
/// row. The first pass partitions it into 128 contiguous 784-element rows,
/// with 224 threads reading four values each (196 active threads), and the
/// second pass reduces those 128 partials with one 32-lane simdgroup.
///
/// These two custom kernels reproduce that exact geometry while fusing the
/// elementwise `coarse - delta` into pass one and the scalar threshold
/// arithmetic into pass two. For finite inputs, max only selects an existing
/// float, so matching the stock partitions plus `simd_max` gives the same
/// `L` bit pattern. The helper also preserves MLX's NaN propagation and its
/// pairwise `a > b ? a : b` behavior for completeness.
private let lagunaLmHeadLowerMaxHeader = """
    static inline float laguna_lmhead_max_pair(float a, float b) {
        if (metal::isnan(a) || metal::isnan(b)) {
            return NAN;
        }
        return a > b ? a : b;
    }

    static inline float laguna_lmhead_simd_max(float value) {
        if (simd_any(value != value)) {
            return NAN;
        }
        return simd_max(value);
    }
    """

/// Pass one of the fused lower-bound reduction. Its launch shape and read
/// order are the stock MLX `all_reduce_max` first pass for exactly 100352
/// float32 values: grid (224, 128), threadgroup (224, 1), four consecutive
/// values per active thread, then the stock simdgroup/shared-memory tree.
private let lagunaLmHeadLowerMaxStage1Kernel = MLXFast.metalKernel(
    name: "laguna_lmhead_lower_max_stage1_v1",
    inputNames: ["coarse", "delta"],
    outputNames: ["partial_max"],
    source: """
        constexpr uint ROW_SIZE = 784;
        constexpr uint READS = 4;
        constexpr uint ACTIVE_THREADS = ROW_SIZE / READS;
        constexpr uint SIMD_GROUPS = 7;

        uint row = threadgroup_position_in_grid.y;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        threadgroup float shared_vals[32];

        float total = -metal::numeric_limits<float>::infinity();
        if (lid < ACTIVE_THREADS) {
            uint base = row * ROW_SIZE + lid * READS;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < READS; ++i) {
                float lower = coarse[base + i] - delta[base + i];
                total = laguna_lmhead_max_pair(lower, total);
            }
        }

        total = laguna_lmhead_simd_max(total);
        if (simd_lane == 0) {
            shared_vals[simd_group] = total;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        total = lid < SIMD_GROUPS
            ? shared_vals[lid]
            : -metal::numeric_limits<float>::infinity();
        total = laguna_lmhead_simd_max(total);
        if (lid == 0) {
            partial_max[row] = total;
        }
        """,
    header: lagunaLmHeadLowerMaxHeader,
    ensureRowContiguous: true
)

/// Pass two reduces the 128 partials with the same four-values-per-lane
/// order as MLX, then computes `L - abs(L) * 2^-6`. The temporary
/// threadgroup store plus barrier preserves the separate float32 rounding of
/// MLX's multiply before the final subtraction (and prevents contraction).
private let lagunaLmHeadLowerMaxThresholdKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_lower_max_threshold_v1",
    inputNames: ["partial_max"],
    outputNames: ["threshold"],
    source: """
        constexpr uint READS = 4;
        uint lid = thread_position_in_threadgroup.x;
        threadgroup float rounded_beta[1];

        float total = -metal::numeric_limits<float>::infinity();
        uint base = lid * READS;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < READS; ++i) {
            total = laguna_lmhead_max_pair(partial_max[base + i], total);
        }
        total = laguna_lmhead_simd_max(total);

        if (lid == 0) {
            rounded_beta[0] = metal::abs(total) * 0x1p-6f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            threshold[0] = total - rounded_beta[0];
        }
        """,
    header: lagunaLmHeadLowerMaxHeader,
    ensureRowContiguous: true
)

/// Default-off fusion probe for one complete 784-row lower-max partition.
/// Twenty-eight SIMD groups preserve the accepted coarse row arithmetic while
/// each group processes twenty-eight rows in ascending batches. The input row
/// is staged once, and lane zero stores each row's separately rounded lower
/// bound in threadgroup memory. The final 224-thread reduction is the exact
/// stage-one tree above, so the only numerical change under test is none: the
/// compiler emits the same FP32 delta fmuladd followed by the same fsub/fadd.
private let lagunaLmHeadCoarseLowerFusedKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_mxfp8_coarse_lower_fused_v1",
    inputNames: ["x", "codes", "scales"],
    outputNames: ["upper", "coarse_bf", "partial_max"],
    source: """
        constexpr float GAMMA = 0x1p-15f;
        constexpr uint PARTITION_ROWS = 784;
        constexpr uint SIMD_GROUPS = 28;
        constexpr uint ROWS_PER_SIMD = PARTITION_ROWS / SIMD_GROUPS;
        constexpr uint THREADS = SIMD_GROUPS * 32;
        constexpr uint REDUCE_THREADS = 224;
        constexpr uint REDUCE_SIMD_GROUPS = 7;

        uint partition = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        threadgroup bfloat staged_x[2048];
        threadgroup float lower_values[PARTITION_ROWS];
        threadgroup float shared_vals[32];

        for (uint i = lid; i < 2048; i += THREADS) {
            staged_x[i] = x[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(disable)
        for (uint batch = 0; batch < ROWS_PER_SIMD; ++batch) {
            uint local_row = batch * SIMD_GROUPS + simd_group;
            uint row = partition * PARTITION_ROWS + local_row;
            const device uint8_t* crow = codes + size_t(row) * 2048;
            const device uint8_t* srow = scales + size_t(row) * 64;

            float c_acc = 0.0f;
            float d_acc = 0.0f;
            float m_acc = 0.0f;
            for (uint gg = 0; gg < 2; ++gg) {
                uint g = 2 * lane + gg;
                float sd = laguna_e8m0_decode(srow[g]);
                const device uint4* cptr =
                    (const device uint4*)(crow + g * 32);
                uint4 packed0 = cptr[0];
                uint4 packed1 = cptr[1];
                float cg = 0.0f;
                float dg = 0.0f;
                float mg = 0.0f;
                const threadgroup ushort4* xrow =
                    (const threadgroup ushort4*)(staged_x + g * 32);
                #pragma clang loop unroll(full)
                for (uint w = 0; w < 8; ++w) {
                    uint word = (w < 4u) ? packed0[w & 3u] : packed1[w & 3u];
                    float4 cv4 = laguna_e4m3_decode4(word);
                    float4 xv4 = as_type<float4>(uint4(xrow[w]) << 16);
                    float4 ax4 = metal::abs(xv4);
                    uint4 b4 =
                        (uint4(word) >> uint4(0u, 8u, 16u, 24u)) & 255u;
                    uint4 mag4 = b4 & 127u;
                    uint4 e4 = mag4 >> 3;
                    float4 hsf = as_type<float4>(
                        (metal::max(e4, uint4(1u)) + 116u) << 23);
                    float4 hs4 = metal::select(
                        hsf, float4(186.0f), mag4 == 126u);
                    float4 acv4 = metal::abs(cv4);
                    #pragma clang loop unroll(full)
                    for (uint k = 0; k < 4; ++k) {
                        float cv = cv4[k];
                        float xv = xv4[k];
                        float ax = ax4[k];
                        cg += xv * cv;
                        dg += ax * hs4[k];
                        mg += ax * acv4[k];
                    }
                }
                c_acc += sd * cg;
                d_acc += sd * dg;
                m_acc += sd * mg;
            }
            c_acc = simd_sum(c_acc);
            d_acc = simd_sum(d_acc);
            m_acc = simd_sum(m_acc);
            if (lane == 0) {
                float rounded_delta =
                    d_acc * (1.0f + GAMMA) + (2.0f * GAMMA) * m_acc;
                lower_values[local_row] = c_acc - rounded_delta;
                upper[row] = c_acc + rounded_delta;
                coarse_bf[row] = bfloat(c_acc);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = -metal::numeric_limits<float>::infinity();
        if (lid < REDUCE_THREADS - 28) {
            uint base = lid * 4;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                total = laguna_lmhead_max_pair(lower_values[base + i], total);
            }
        }
        total = laguna_lmhead_simd_max(total);
        if (lane == 0) {
            shared_vals[simd_group] = total;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        total = lid < REDUCE_SIMD_GROUPS
            ? shared_vals[lid]
            : -metal::numeric_limits<float>::infinity();
        total = laguna_lmhead_simd_max(total);
        if (lid == 0) {
            partial_max[partition] = total;
        }
        """,
    header: lagunaLmHeadPruneHeader + lagunaLmHeadLowerMaxHeader,
    ensureRowContiguous: true
)

/// GPU candidate marking: one byte per vocabulary row, set when the row's
/// certified upper bound reaches the threshold. A dense mask rather than a
/// compacted index list, because the exact pass below owns a FIXED output
/// block per simdgroup and therefore needs "is row r a candidate?" keyed by
/// r, not "what is the r-th candidate?". No atomics, no compaction, and the
/// output is a pure function of its inputs.
private let lagunaLmHeadSelectKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_select_v2",
    inputNames: ["coarse", "delta", "thr"],
    outputNames: ["is_cand"],
    source: """
        uint i = thread_position_in_grid.x;
        is_cand[i] = (coarse[i] + delta[i] >= thr[0]) ? uint8_t(1) : uint8_t(0);
        """,
    ensureRowContiguous: true
)

/// Exact pass. Each simdgroup owns a FIXED block of four output rows -- the
/// same static row-to-simdgroup mapping the stock kernel uses -- and runs the
/// full-precision GEMV for that block only when at least one of its four rows
/// is a candidate; otherwise it writes those rows' coarse values. Because the
/// block is fixed, `assembled[r]` is written by exactly ONE lane (lane 0 of
/// the owning simdgroup) on exactly one path, so the output is fully covered
/// with no race and no uninitialized slot.
///
/// Per-row arithmetic is a textual replica of the stock `gemv_al_bfloat16`
/// (GEMVKernel<bfloat16_t, 8,1,1,32, 4,4, false, true>::run with
/// matrix_ld = 2048, in_vec_size = 2048, no leftover, no tgp reduction):
/// same lane partition, same sequential f32 accumulation order, same vec4
/// loads, same simd_shuffle_down tree, same single BF16 cast. There is no row
/// indirection at all -- row `r` is computed by the thread that owns output
/// slot `r` -- so a candidate row's value is bit-identical to the stock full
/// GEMV's by construction (R1).
///
/// The skipped work is the byte saving: with |C| in the single-to-low-double
/// digits, all but a handful of threadgroups take the coarse branch and never
/// touch `lm_head`. The default packs thirty-two fixed four-row SIMD blocks
/// into each 1,024-thread threadgroup, quartering the original launch count.
private func lagunaLmHeadExactKernelSource(rowsPerThreadgroup: Int) -> String {
    """
        constexpr uint VOCAB = 100352;
        constexpr uint K = 2048;

        uint tgid = threadgroup_position_in_grid.x;
        uint sgid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        uint base = tgid * \(rowsPerThreadgroup) + sgid * 4;

        // Simdgroup-uniform: every lane reads the same four mask bytes.
        bool any_candidate = false;
        #pragma unroll
        for (uint tm = 0; tm < 4; ++tm) {
            uint r = base + tm;
            any_candidate = any_candidate || (r < VOCAB && is_cand[r] != 0);
        }

        if (!any_candidate) {
            if (lane < 4 && base + lane < VOCAB) {
                assembled[base + lane] = coarse_bf[base + lane];
            }
            return;
        }

        // --- stock gemv_al replica begin (gemv.h:151-289) ---
        thread float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        thread bfloat inter[4];
        thread float v_coeff[4];
        uint bn = lane * 4;
        for (uint i = 0; i < 16; ++i) {
            vec<bfloat, 4> xv =
                *((const device vec<bfloat, 4>*)(x + bn));
            v_coeff[0] = float(xv.x);
            v_coeff[1] = float(xv.y);
            v_coeff[2] = float(xv.z);
            v_coeff[3] = float(xv.w);
            #pragma unroll
            for (uint tm = 0; tm < 4; ++tm) {
                const device bfloat* mrow = lm_head + size_t(base + tm) * K;
                vec<bfloat, 4> mv =
                    *((const device vec<bfloat, 4>*)(mrow + bn));
                inter[0] = mv.x;
                inter[1] = mv.y;
                inter[2] = mv.z;
                inter[3] = mv.w;
                result[tm] += inter[0] * v_coeff[0];
                result[tm] += inter[1] * v_coeff[1];
                result[tm] += inter[2] * v_coeff[2];
                result[tm] += inter[3] * v_coeff[3];
            }
            bn += 128;
        }
        #pragma unroll
        for (uint tm = 0; tm < 4; ++tm) {
            #pragma unroll
            for (ushort sn = 16; sn >= 1; sn >>= 1) {
                result[tm] += simd_shuffle_down(result[tm], sn);
            }
        }
        // --- stock gemv_al replica end ---
        if (lane == 0) {
            #pragma unroll
            for (uint tm = 0; tm < 4; ++tm) {
                uint r = base + tm;
                if (r < VOCAB) {
                    assembled[r] = (is_cand[r] != 0)
                        ? bfloat(result[tm])
                        : coarse_bf[r];
                }
            }
        }
        """
}

private let lagunaLmHeadExactKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_block_pack32_v4",
    inputNames: ["coarse_bf", "lm_head", "x", "is_cand"],
    outputNames: ["assembled"],
    source: lagunaLmHeadExactKernelSource(rowsPerThreadgroup: 128),
    ensureRowContiguous: true
)

private func lagunaLmHeadExactMaskBroadcastSource() -> String {
    let source = lagunaLmHeadExactKernelSource(rowsPerThreadgroup: 128)
    let candidate = source.replacingOccurrences(
        of: "// Simdgroup-uniform: every lane reads the same four mask bytes.\n"
            + "bool any_candidate = false;\n#pragma unroll\n"
            + "for (uint tm = 0; tm < 4; ++tm) {\n"
            + "    uint r = base + tm;\n"
            + "    any_candidate = any_candidate || (r < VOCAB && is_cand[r] != 0);\n}",
        with: "// base is four-byte aligned and every fixed block is in bounds.\n"
            + "uint any_candidate = 0;\nif (lane == 0) {\n"
            + "    any_candidate = *((const device uint*)(is_cand + base));\n}\n"
            + "any_candidate = simd_broadcast(any_candidate, 0);")
    precondition(candidate != source)
    return candidate
}

private let lagunaLmHeadExactMaskBroadcastKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_block_mask_broadcast_v5",
    inputNames: ["coarse_bf", "lm_head", "x", "is_cand"],
    outputNames: ["assembled"],
    source: lagunaLmHeadExactMaskBroadcastSource(),
    ensureRowContiguous: true
)

private let lagunaLmHeadPackedSelectKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_select_packed_v3",
    inputNames: ["coarse", "delta", "thr"],
    outputNames: ["packed_cand"],
    source: """
        uint i = thread_position_in_grid.x;
        uint base = i * 4;
        float4 c = *((const device float4*)(coarse + base));
        float4 d = *((const device float4*)(delta + base));
        bool4 selected = c + d >= float4(thr[0]);
        packed_cand[i] = uint(selected.x)
            | (uint(selected.y) << 8)
            | (uint(selected.z) << 16)
            | (uint(selected.w) << 24);
        """,
    ensureRowContiguous: true
)

private func lagunaLmHeadExactPackedMaskRepeatedLoadsSource() -> String {
    let source = lagunaLmHeadExactMaskBroadcastSource()
    let loadReplaced = source.replacingOccurrences(
        of: "any_candidate = *((const device uint*)(is_cand + base));",
        with: "any_candidate = is_cand[base >> 2];")
    precondition(loadReplaced != source)
    let candidate = loadReplaced.replacingOccurrences(
        of: "assembled[r] = (is_cand[r] != 0)\n"
            + "                ? bfloat(result[tm])\n"
            + "                : coarse_bf[r];",
        with: "assembled[r] = ((is_cand[base >> 2] >> (tm * 8)) & 255u) != 0\n"
            + "                ? bfloat(result[tm])\n"
            + "                : coarse_bf[r];")
    precondition(candidate != loadReplaced)
    return candidate
}

private let lagunaLmHeadExactPackedMaskRepeatedLoadsKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_block_packed_mask_v6",
    inputNames: ["coarse_bf", "lm_head", "x", "is_cand"],
    outputNames: ["assembled"],
    source: lagunaLmHeadExactPackedMaskRepeatedLoadsSource(),
    ensureRowContiguous: true
)

private func lagunaLmHeadExactPackedMaskSource() -> String {
    let source = lagunaLmHeadExactPackedMaskRepeatedLoadsSource()
    let candidate = source.replacingOccurrences(
        of: "((is_cand[base >> 2] >> (tm * 8)) & 255u)",
        with: "((any_candidate >> (tm * 8)) & 255u)")
    precondition(candidate != source)
    return candidate
}

private let lagunaLmHeadExactPackedMaskWithBoundsKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_block_packed_mask_v7",
    inputNames: ["coarse_bf", "lm_head", "x", "is_cand"],
    outputNames: ["assembled"],
    source: lagunaLmHeadExactPackedMaskSource(),
    ensureRowContiguous: true
)

private func lagunaLmHeadExactPackedMaskNoBoundsSource() -> String {
    let source = lagunaLmHeadExactPackedMaskSource()
    let fastPath = source.replacingOccurrences(
        of: "if (lane < 4 && base + lane < VOCAB)",
        with: "if (lane < 4)")
    precondition(fastPath != source)
    let candidate = fastPath.replacingOccurrences(
        of: "if (r < VOCAB) {",
        with: "if (true) {")
    precondition(candidate != fastPath)
    return candidate
}

private let lagunaLmHeadExactPackedMaskScalarCopyKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_block_packed_mask_v8",
    inputNames: ["coarse_bf", "lm_head", "x", "is_cand"],
    outputNames: ["assembled"],
    source: lagunaLmHeadExactPackedMaskNoBoundsSource(),
    ensureRowContiguous: true
)

private func lagunaLmHeadExactPackedMaskVectorCopySource() -> String {
    let source = lagunaLmHeadExactPackedMaskNoBoundsSource()
    let candidate = source.replacingOccurrences(
        of: "if (lane < 4) {\n"
            + "        assembled[base + lane] = coarse_bf[base + lane];\n"
            + "    }",
        with: "if (lane == 0) {\n"
            + "        *((device vec<bfloat, 4>*)(assembled + base)) =\n"
            + "            *((const device vec<bfloat, 4>*)(coarse_bf + base));\n"
            + "    }")
    precondition(candidate != source)
    return candidate
}

private let lagunaLmHeadExactPackedMaskKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_block_packed_mask_v9",
    inputNames: ["coarse_bf", "lm_head", "x", "is_cand"],
    outputNames: ["assembled"],
    source: lagunaLmHeadExactPackedMaskVectorCopySource(),
    ensureRowContiguous: true
)

private func lagunaLmHeadExactFusedSelectSource() -> String {
    let source = lagunaLmHeadExactPackedMaskVectorCopySource()
    let candidate = source.replacingOccurrences(
        of: "any_candidate = is_cand[base >> 2];",
        with: "float4 c = *((const device float4*)(coarse + base));\n"
            + "    float4 d = *((const device float4*)(delta + base));\n"
            + "    bool4 selected = c + d >= float4(thr[0]);\n"
            + "    any_candidate = uint(selected.x)\n"
            + "        | (uint(selected.y) << 8)\n"
            + "        | (uint(selected.z) << 16)\n"
            + "        | (uint(selected.w) << 24);")
    precondition(candidate != source)
    return candidate
}

private let lagunaLmHeadExactFusedSelectKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_block_fused_select_v1",
    inputNames: ["coarse_bf", "lm_head", "x", "coarse", "delta", "thr"],
    outputNames: ["assembled"],
    source: lagunaLmHeadExactFusedSelectSource(),
    ensureRowContiguous: true
)

private func lagunaLmHeadExactUpperBoundSource() -> String {
    let source = lagunaLmHeadExactPackedMaskVectorCopySource()
    let candidate = source.replacingOccurrences(
        of: "any_candidate = is_cand[base >> 2];",
        with: "float4 u = *((const device float4*)(upper + base));\n"
            + "    bool4 selected = u >= float4(thr[0]);\n"
            + "    any_candidate = uint(selected.x)\n"
            + "        | (uint(selected.y) << 8)\n"
            + "        | (uint(selected.z) << 16)\n"
            + "        | (uint(selected.w) << 24);")
    precondition(candidate != source)
    return candidate
}

private let lagunaLmHeadExactUpperBoundKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_block_upper_bound_v1",
    inputNames: ["coarse_bf", "lm_head", "x", "upper", "thr"],
    outputNames: ["assembled"],
    source: lagunaLmHeadExactUpperBoundSource(),
    ensureRowContiguous: true
)

private let lagunaLmHeadExactKernelPack16 = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_block_pack16_v3",
    inputNames: ["coarse_bf", "lm_head", "x", "is_cand"],
    outputNames: ["assembled"],
    source: lagunaLmHeadExactKernelSource(rowsPerThreadgroup: 64),
    ensureRowContiguous: true
)

private let lagunaLmHeadExactKernelPack8 = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_block_v2",
    inputNames: ["coarse_bf", "lm_head", "x", "is_cand"],
    outputNames: ["assembled"],
    source: lagunaLmHeadExactKernelSource(rowsPerThreadgroup: 32),
    ensureRowContiguous: true
)

private let lagunaLmHeadExactUsePack8 =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_EXACT_PACK"] == "8"

private let lagunaLmHeadExactUsePack16 =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_EXACT_PACK"] == "16"

private let lagunaLmHeadExactUseRepeatedMaskLoads =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_LMHEAD_EXACT_MASK_BROADCAST"] == "0"

private let lagunaLmHeadPackedMaskEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_PACKED_MASK"] != "0"

private let lagunaLmHeadPackedMaskReuseEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_PACKED_MASK_REUSE"] != "0"

private let lagunaLmHeadPackedMaskNoBoundsEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_PACKED_MASK_NO_BOUNDS"] != "0"

private let lagunaLmHeadPackedMaskVectorCopyEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_PACKED_MASK_VECTOR_COPY"] != "0"

private let lagunaLmHeadExactFusedSelectEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_FUSED_SELECT"] != "0"

private let lagunaLmHeadCoarseLowerFusedEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_COARSE_LOWER_FUSED"] != "0"

/// Retained init-time MXFP8 coarse copy of lm_head plus the pruned final-row
/// forward. Built once (untimed init) by
/// `LagunaRuntimeModel.prepareFusedRuntimeWeights` when
/// `lagunaLmHeadPruneEnabled` (DARKBLOOM_LM_HEAD_PRUNE, default ON; set "0"
/// to disable); ~212 MB additional resident memory.
final class LagunaLmHeadPruner {
    let codes: MLXArray   // [100352, 2048] uint8 e4m3 elements
    let scales: MLXArray  // [100352, 64] uint8 e8m0 group scales

    init?(lmHeadWeight: MLXArray) {
        guard lmHeadWeight.shape == [lagunaLmHeadPruneVocab, lagunaLmHeadPruneHidden],
            lmHeadWeight.dtype == .bfloat16
        else {
            FileHandle.standardError.write(
                Data("mlxfast: lm_head prune: unrecognized lm_head shape/dtype; disabled\n".utf8))
            return nil
        }
        // The repo's own quantizer (ops.cpp fp_quantize gs32/bits8 ->
        // fp_quantized.h fp_quantize kernel): e8m0 group scale = 2^round(log2(
        // gmax/448)), e4m3 elements of w/sd. Returns (wq uint32 viewed as
        // [V, 512], scales uint8 [V, 64]); the uint32 view is the same bytes
        // as per-element uint8 codes in order.
        let (wq, scales, _) = quantized(
            lmHeadWeight, groupSize: 32, bits: 8, mode: .mxfp8)
        self.codes = wq.view(dtype: .uint8)
        self.scales = scales
    }

    /// Pruned final-row lm_head: full [vocab] BF16 logits row, bit-identical to
    /// the stock pass in every candidate slot and certified-below elsewhere,
    /// so the downstream argmax emits the stock token.
    func logits(hidden: MLXArray, lmHeadWeight: MLXArray) -> MLXArray {
        precondition(hidden.dtype == .bfloat16 && hidden.size == lagunaLmHeadPruneHidden)
        let vocab = lagunaLmHeadPruneVocab
        let x = hidden.reshaped([lagunaLmHeadPruneHidden])

        let useCoarseV1 = lagunaLmHeadCoarseUseV1
        let useCoarsePack16 = lagunaLmHeadCoarseUsePack16
        let useCoarseLowerFused = lagunaLmHeadCoarseLowerFusedEnabled
            && !useCoarseV1 && !useCoarsePack16
            && !lagunaLmHeadExactUsePack8 && !lagunaLmHeadExactUsePack16
            && !lagunaLmHeadExactUseRepeatedMaskLoads
            && lagunaLmHeadPackedMaskEnabled
            && lagunaLmHeadPackedMaskReuseEnabled
            && lagunaLmHeadPackedMaskNoBoundsEnabled
            && lagunaLmHeadPackedMaskVectorCopyEnabled
        let coarse: MLXArray?
        let delta: MLXArray?
        let upper: MLXArray?
        let coarseBF: MLXArray
        let lowerMaxPartials: MLXArray
        if useCoarseLowerFused {
            let outputs = lagunaLmHeadCoarseLowerFusedKernel(
                [x, codes, scales],
                grid: (128 * 896, 1, 1),
                threadGroup: (896, 1, 1),
                outputShapes: [[vocab], [vocab], [128]],
                outputDTypes: [.float32, .bfloat16, .float32]
            )
            upper = outputs[0]
            coarseBF = outputs[1]
            lowerMaxPartials = outputs[2]
            coarse = nil
            delta = nil
        } else {
            let coarseKernel = useCoarseV1 ? lagunaLmHeadCoarseKernelV1
                : (useCoarsePack16
                    ? lagunaLmHeadCoarseKernelPack16 : lagunaLmHeadCoarseKernel)
            let coarseRowsPerThreadgroup = useCoarseV1
                ? 8 : (useCoarsePack16 ? 16 : 28)
            let coarseThreadsPerThreadgroup = coarseRowsPerThreadgroup * 32
            let outputs = coarseKernel(
                [x, codes, scales],
                grid: (
                    vocab / coarseRowsPerThreadgroup * coarseThreadsPerThreadgroup,
                    1, 1),
                threadGroup: (coarseThreadsPerThreadgroup, 1, 1),
                outputShapes: [[vocab], [vocab], [vocab]],
                outputDTypes: [.float32, .float32, .bfloat16]
            )
            coarse = outputs[0]
            delta = outputs[1]
            coarseBF = outputs[2]
            lowerMaxPartials = lagunaLmHeadLowerMaxStage1Kernel(
                [coarse!, delta!],
                grid: (224, 128, 1),
                threadGroup: (224, 1, 1),
                outputShapes: [[128]],
                outputDTypes: [.float32]
            )[0]
            upper = nil
        }

        let thr = lagunaLmHeadLowerMaxThresholdKernel(
            [lowerMaxPartials],
            grid: (32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[1]],
            outputDTypes: [.float32]
        )[0]

        let useExactPack8 = lagunaLmHeadExactUsePack8
        let useExactPack16 = lagunaLmHeadExactUsePack16
        let usePackedMask = lagunaLmHeadPackedMaskEnabled
            && !useExactPack8 && !useExactPack16
            && !lagunaLmHeadExactUseRepeatedMaskLoads
        let useFusedSelect = usePackedMask
            && lagunaLmHeadPackedMaskReuseEnabled
            && lagunaLmHeadPackedMaskNoBoundsEnabled
            && lagunaLmHeadPackedMaskVectorCopyEnabled
            && lagunaLmHeadExactFusedSelectEnabled
        let isCandidate: MLXArray?
        if useCoarseLowerFused || useFusedSelect {
            isCandidate = nil
        } else if usePackedMask {
            isCandidate = lagunaLmHeadPackedSelectKernel(
                [coarse!, delta!, thr],
                grid: (vocab / 4, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[vocab / 4]],
                outputDTypes: [.uint32]
            )[0]
        } else {
            isCandidate = lagunaLmHeadSelectKernel(
                [coarse!, delta!, thr],
                grid: (vocab, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[vocab]],
                outputDTypes: [.uint8]
            )[0]
        }

        let exactKernel = useCoarseLowerFused
            ? lagunaLmHeadExactUpperBoundKernel
            : useFusedSelect ? lagunaLmHeadExactFusedSelectKernel
            : usePackedMask
            ? (lagunaLmHeadPackedMaskReuseEnabled
                ? (lagunaLmHeadPackedMaskNoBoundsEnabled
                    ? (lagunaLmHeadPackedMaskVectorCopyEnabled
                        ? lagunaLmHeadExactPackedMaskKernel
                        : lagunaLmHeadExactPackedMaskScalarCopyKernel)
                    : lagunaLmHeadExactPackedMaskWithBoundsKernel)
                : lagunaLmHeadExactPackedMaskRepeatedLoadsKernel)
            : (useExactPack8 ? lagunaLmHeadExactKernelPack8
                : (useExactPack16 ? lagunaLmHeadExactKernelPack16
                    : (lagunaLmHeadExactUseRepeatedMaskLoads
                        ? lagunaLmHeadExactKernel
                        : lagunaLmHeadExactMaskBroadcastKernel)))
        let exactSIMDGroupsPerThreadgroup = useExactPack8 ? 8 : (useExactPack16 ? 16 : 32)
        let exactThreadsPerThreadgroup = exactSIMDGroupsPerThreadgroup * 32
        let exactRowsPerThreadgroup = exactSIMDGroupsPerThreadgroup * 4
        let exactInputs = useCoarseLowerFused
            ? [coarseBF, lmHeadWeight, x, upper!, thr]
            : useFusedSelect
                ? [coarseBF, lmHeadWeight, x, coarse!, delta!, thr]
                : [coarseBF, lmHeadWeight, x, isCandidate!]
        let assembled = exactKernel(
            exactInputs,
            grid: (
                vocab / exactRowsPerThreadgroup * exactThreadsPerThreadgroup,
                1, 1),
            threadGroup: (exactThreadsPerThreadgroup, 1, 1),
            outputShapes: [[vocab]],
            outputDTypes: [.bfloat16]
        )[0]
        return assembled.reshaped([1, 1, vocab])
    }
}
