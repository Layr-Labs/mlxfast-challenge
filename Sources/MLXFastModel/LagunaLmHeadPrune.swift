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
//   1. COARSE pass (`lagunaLmHeadInlineCoarseKernel`): one fused GEMV over an
//      init-time MXFP8 copy of lm_head (gs32 e8m0+e4m3, 211.9 MB) built with
//      the repo's own `quantized(..., mode: .mxfp8)`, producing per-row coarse
//      logit c_i and a certified bound delta_i. delta_i =
//      d_i*(1+gamma) + 2*gamma*m_i with
//      d_i = sum_g sd_g * sum_{j in g} |x_j| * hs8(code_ij)
//          >= sum_j |x_j| * |w_ij - what_ij|   (half-ulp cells, top cell 186)
//      and m_i = sum_j |x_j| * |what_ij|, so delta_i covers BOTH the
//      quantization error and both kernels' float rounding (depth <= 96
//      roundings/element-path << gamma = 2^-15 relative; notes/68 section 6).
//      The e4m3/e8m0 decoders below are bit-exact replicas of the vendored
//      fp8.h / fp_quantized.h semantics (no libm: exponent-bit construction).
//   2. EXACT pass (`lagunaLmHeadInlineExactKernel`): each simdgroup owns a FIXED
//      block of four output rows and runs a full BF16 GEMV over that block
//      only when `coarse[r] + delta[r] >= threshold` for one of its rows,
//      writing `bfloat(coarse[r])` otherwise. The predicate and BF16 cast are
//      textually the same operations as the retained mask/coarse_bf path, so
//      NaNs, signed zero, and every stored FP32 coarse bit keep their existing
//      behavior while the selector dispatch and both temporary buffers vanish.
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
// The default threshold is BF16(L)'s exact lower rounding boundary, constructed
// from integer bits; `DARKBLOOM_LMHEAD_EXACT_BETA=0` retains the older
// two-ulp beta for same-binary ablation.
//
// `DARKBLOOM_LMHEAD_INLINE_MASK=0` restores the new tip's three-output coarse
// kernels, fused two-pass lower-bound reduction, dense uint8 selector mask,
// and coarse_bf-fed exact kernel inside the same binary.

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

/// Inline candidate testing for the certified exact pass. The exact kernel
/// copies the retained selector's `coarse + delta >= threshold` sequence
/// textually. The retained coarse pass stored FP32 `c_acc` and cast that same
/// value to BF16; the inline path reloads the unchanged FP32 bits and applies
/// the same BF16 cast, preserving finite, NaN, and signed-zero behavior.
/// DEFAULT ON; set `DARKBLOOM_LMHEAD_INLINE_MASK=0` to restore the dense mask
/// dispatch and stored `coarse_bf` output.
private let lagunaLmHeadInlineMaskEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_INLINE_MASK"] != "0"

/// Exact candidate threshold for the final BF16 logits. DEFAULT ON; set
/// `DARKBLOOM_LMHEAD_EXACT_BETA=0` to restore the retained `L - |L|/64`
/// threshold kernel byte-for-byte.
private let lagunaLmHeadExactBetaEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_EXACT_BETA"] != "0"

/// Candidate membership is computed by lane 0 and broadcast as a four-bit
/// integer mask. DEFAULT ON; set `DARKBLOOM_LMHEAD_LANE_UNIFORM=0` to restore
/// the retained every-lane candidate expressions.
private let lagunaLmHeadLaneUniformEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_LANE_UNIFORM"] != "0"

/// Within an active four-row exact block, issue the copied stock GEMV body
/// only for candidate rows. DEFAULT ON; set `DARKBLOOM_LMHEAD_ROW_SKIP=0` to
/// restore computation of all four rows.
private let lagunaLmHeadRowSkipEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_ROW_SKIP"] != "0"

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
/// v2 (H3 audit, R1): same grid, same lane->element mapping, same FP
/// accumulation text and j-order -- only the per-element decode plumbing is
/// vectorized. Word-parallel e4m3 decode (laguna_e4m3_decode4, bit-identical
/// construction), vectorized hs8 (max-form, identical floats), x loaded as
/// ushort4 and converted bf16->f32 by the exact bits<<16 construction, and
/// both loops fully unrolled with static trip counts so the packed words and
/// vector components resolve to static indices. Coarse, delta, and coarse_bf
/// outputs are bit-identical to v1 for every input, so the notes/68
/// certificate is untouched.
private let lagunaLmHeadCoarseKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_mxfp8_coarse_v2",
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
        """,
    header: lagunaLmHeadPruneHeader,
    ensureRowContiguous: true
)
/// v1 coarse kernel, kept verbatim for same-binary A/B (the paired
/// measurement protocol requires both arms in one binary). Selected by
/// `DARKBLOOM_LMHEAD_COARSE=v1`; the shipped default is v2 above. The two
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

/// Two-output coarse kernels for the inline-mask path. Their accumulation and
/// bound sequences are copied textually from the retained v2/v1 kernels above;
/// only the final `coarse_bf[row] = bfloat(c_acc)` store is absent. The exact
/// pass reloads the stored FP32 bits and applies that same BF16 conversion,
/// which is byte-identical for finite values, NaNs, and signed zero. Setting
/// `DARKBLOOM_LMHEAD_INLINE_MASK=0` selects the original three-output kernels.
private let lagunaLmHeadInlineCoarseKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_mxfp8_inline_coarse_v2",
    inputNames: ["x", "codes", "scales"],
    outputNames: ["coarse", "delta"],
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
        }
        """,
    header: lagunaLmHeadPruneHeader,
    ensureRowContiguous: true
)

private let lagunaLmHeadInlineCoarseKernelV1 = MLXFast.metalKernel(
    name: "laguna_lmhead_mxfp8_inline_coarse_v1",
    inputNames: ["x", "codes", "scales"],
    outputNames: ["coarse", "delta"],
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
        }
        """,
    header: lagunaLmHeadPruneHeader,
    ensureRowContiguous: true
)

/// Same-binary A/B selector for the coarse kernel (v2 default).
private let lagunaLmHeadCoarseUseV1 =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_COARSE"] == "v1"

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

/// Integer construction of the exact lower rounding boundary for BF16(L).
/// The FP32-to-BF16 RNE step and the midpoint's FP32 bits are derived without
/// floating-point arithmetic. Candidate comparison remains `upper >= T`, so
/// an exact midpoint is retained regardless of the BF16 tie-even parity.
///
/// Exceptional handling is conservative: NaN returns the same NaN threshold
/// as the incumbent path (all ordered comparisons are false); negative
/// infinity returns -infinity (all non-NaN rows remain candidates); positive
/// infinity uses the finite BF16 overflow midpoint 0x7f7f8000.
private let lagunaLmHeadExactBetaHeader = lagunaLmHeadLowerMaxHeader + """

    static inline float laguna_lmhead_bf16_lower_boundary(float value) {
        uint raw = as_type<uint>(value);
        uint magnitude = raw & 0x7fffffffu;

        // Preserve NaN as NaN without a floating-point operation.
        if (magnitude > 0x7f800000u) {
            return as_type<float>(raw);
        }

        // Round the finite/infinite FP32 bit pattern to BF16, ties-to-even.
        uint upper = raw >> 16;
        uint remainder = raw & 0xffffu;
        uint increment =
            (remainder > 0x8000u ||
             (remainder == 0x8000u && (upper & 1u) != 0u))
            ? 1u : 0u;
        uint bf = upper + increment;

        // -infinity has no lower finite neighbor. Keeping -infinity is the
        // conservative threshold and matches the retained beta path.
        if (bf == 0xff80u) {
            return as_type<float>(0xff800000u);
        }

        // The predecessor of either signed zero is -min-subnormal BF16, whose
        // midpoint with zero is exactly FP32 -2^-134 (0x80008000).
        if ((bf & 0x7fffu) == 0u) {
            return as_type<float>(0x80008000u);
        }

        // Adjacent BF16 values differ by 0x10000 in their exact FP32
        // encodings. The midpoint below a positive value (including +inf) is
        // -0x8000 bits; below a negative finite value it is +0x8000 bits.
        uint fp = bf << 16;
        uint midpoint =
            (bf & 0x8000u) != 0u ? fp + 0x8000u : fp - 0x8000u;
        return as_type<float>(midpoint);
    }
    """

private let lagunaLmHeadLowerMaxExactThresholdKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_lower_max_exact_bf16_threshold_v1",
    inputNames: ["partial_max"],
    outputNames: ["threshold"],
    source: """
        constexpr uint READS = 4;
        uint lid = thread_position_in_threadgroup.x;

        float total = -metal::numeric_limits<float>::infinity();
        uint base = lid * READS;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < READS; ++i) {
            total = laguna_lmhead_max_pair(partial_max[base + i], total);
        }
        total = laguna_lmhead_simd_max(total);

        if (lid == 0) {
            threshold[0] = laguna_lmhead_bf16_lower_boundary(total);
        }
        """,
    header: lagunaLmHeadExactBetaHeader,
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
/// digits, all but a handful of the 3136 threadgroups take the coarse branch
/// and never touch `lm_head`.
private let lagunaLmHeadExactKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_block_v2",
    inputNames: ["coarse_bf", "lm_head", "x", "is_cand"],
    outputNames: ["assembled"],
    source: """
        constexpr uint VOCAB = 100352;
        constexpr uint K = 2048;

        uint tgid = threadgroup_position_in_grid.x;
        uint sgid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        // This simdgroup's fixed four output rows. VOCAB is 3136 * 32, so the
        // grid tiles it exactly; the bounds test is belt-and-braces.
        uint base = tgid * 32 + sgid * 4;

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
        """,
    ensureRowContiguous: true
)

/// Default exact pass with candidate testing inlined. The membership
/// expression is copied textually from `lagunaLmHeadSelectKernel`, and skipped
/// rows apply the same BF16 conversion after reloading the FP32 value formerly
/// cast by the coarse kernel. The memory round-trip adds no arithmetic, so NaN
/// payloads and signed zero reach the same cast; candidate NaNs compare false
/// on both paths. The stock GEMV block below is otherwise a textual copy of
/// `lagunaLmHeadExactKernel`. Set `DARKBLOOM_LMHEAD_INLINE_MASK=0` to restore
/// that kernel plus its selector and `coarse_bf` input. Both paths consume the
/// same selected lower-bound threshold.
private let lagunaLmHeadInlineExactKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_inline_mask_block_v1",
    inputNames: ["coarse", "delta", "thr", "lm_head", "x"],
    outputNames: ["assembled"],
    source: """
        constexpr uint VOCAB = 100352;
        constexpr uint K = 2048;

        uint tgid = threadgroup_position_in_grid.x;
        uint sgid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        // This simdgroup's fixed four output rows. VOCAB is 3136 * 32, so the
        // grid tiles it exactly; the bounds test is belt-and-braces.
        uint base = tgid * 32 + sgid * 4;

        // Simdgroup-uniform. This is textually the selector's predicate; the
        // fixed row mapping still gives one owner per output slot.
        bool any_candidate = false;
        #pragma unroll
        for (uint tm = 0; tm < 4; ++tm) {
            uint r = base + tm;
            any_candidate = any_candidate ||
                (r < VOCAB && coarse[r] + delta[r] >= thr[0]);
        }

        if (!any_candidate) {
            if (lane < 4 && base + lane < VOCAB) {
                assembled[base + lane] = bfloat(coarse[base + lane]);
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
                    assembled[r] = (coarse[r] + delta[r] >= thr[0])
                        ? bfloat(result[tm])
                        : bfloat(coarse[r]);
                }
            }
        }
        """,
    ensureRowContiguous: true
)

/// Builds the independently switchable lane-uniform and per-row-skip exact
/// variants. The candidate expression is copied from the retained selector,
/// and every floating-point line between the stock-replica markers is copied
/// textually from `lagunaLmHeadInlineExactKernel`. The only inserted control
/// is a simdgroup-uniform integer-mask guard around a row's existing loads,
/// accumulations, shuffle tree, and BF16 cast.
private func makeLagunaLmHeadSpecializedExactKernel(
    name: String,
    inlineCandidate: Bool,
    laneUniform: Bool,
    rowSkip: Bool
) -> MLXFast.MLXFastKernel {
    let candidateExpression =
        inlineCandidate
        ? "(r < VOCAB && coarse[r] + delta[r] >= thr[0])"
        : "(r < VOCAB && is_cand[r] != 0)"
    let skippedExpression =
        inlineCandidate ? "bfloat(coarse[r])" : "coarse_bf[r]"
    let inputNames =
        inlineCandidate
        ? ["coarse", "delta", "thr", "lm_head", "x"]
        : ["coarse_bf", "lm_head", "x", "is_cand"]

    return MLXFast.metalKernel(
        name: name,
        inputNames: inputNames,
        outputNames: ["assembled"],
        source: """
            constexpr uint VOCAB = 100352;
            constexpr uint K = 2048;
            constexpr bool LANE_UNIFORM = \(laneUniform ? "true" : "false");
            constexpr bool ROW_SKIP = \(rowSkip ? "true" : "false");

            uint tgid = threadgroup_position_in_grid.x;
            uint sgid = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;

            // This simdgroup's fixed four output rows.
            uint base = tgid * 32 + sgid * 4;

            // With LANE_UNIFORM, lane 0 alone executes the retained predicate
            // and broadcasts only its integer result. Otherwise every lane
            // computes the same mask, preserving an independent #5 ablation.
            uint candidate_mask = 0u;
            if (!LANE_UNIFORM || lane == 0u) {
                #pragma unroll
                for (uint tm = 0; tm < 4; ++tm) {
                    uint r = base + tm;
                    if \(candidateExpression) {
                        candidate_mask |= 1u << tm;
                    }
                }
            }
            if (LANE_UNIFORM) {
                candidate_mask = simd_shuffle(candidate_mask, ushort(0));
            }

            if (candidate_mask == 0u) {
                if (lane < 4 && base + lane < VOCAB) {
                    uint r = base + lane;
                    assembled[r] = \(skippedExpression);
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
                    if (!ROW_SKIP || (candidate_mask & (1u << tm)) != 0u) {
                        const device bfloat* mrow =
                            lm_head + size_t(base + tm) * K;
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
                }
                bn += 128;
            }
            #pragma unroll
            for (uint tm = 0; tm < 4; ++tm) {
                if (!ROW_SKIP || (candidate_mask & (1u << tm)) != 0u) {
                    #pragma unroll
                    for (ushort sn = 16; sn >= 1; sn >>= 1) {
                        result[tm] += simd_shuffle_down(result[tm], sn);
                    }
                }
            }
            // --- stock gemv_al replica end ---
            if (lane == 0) {
                #pragma unroll
                for (uint tm = 0; tm < 4; ++tm) {
                    uint r = base + tm;
                    if (r < VOCAB) {
                        assembled[r] =
                            (candidate_mask & (1u << tm)) != 0u
                            ? bfloat(result[tm])
                            : \(skippedExpression);
                    }
                }
            }
            """,
        ensureRowContiguous: true
    )
}

private let lagunaLmHeadInlineLaneUniformKernel =
    makeLagunaLmHeadSpecializedExactKernel(
        name: "laguna_lmhead_exact_inline_lane_uniform_v1",
        inlineCandidate: true,
        laneUniform: true,
        rowSkip: false
    )

private let lagunaLmHeadInlineRowSkipKernel =
    makeLagunaLmHeadSpecializedExactKernel(
        name: "laguna_lmhead_exact_inline_row_skip_v1",
        inlineCandidate: true,
        laneUniform: false,
        rowSkip: true
    )

private let lagunaLmHeadInlineLaneUniformRowSkipKernel =
    makeLagunaLmHeadSpecializedExactKernel(
        name: "laguna_lmhead_exact_inline_lane_uniform_row_skip_v1",
        inlineCandidate: true,
        laneUniform: true,
        rowSkip: true
    )

private let lagunaLmHeadMaskLaneUniformKernel =
    makeLagunaLmHeadSpecializedExactKernel(
        name: "laguna_lmhead_exact_mask_lane_uniform_v1",
        inlineCandidate: false,
        laneUniform: true,
        rowSkip: false
    )

private let lagunaLmHeadMaskRowSkipKernel =
    makeLagunaLmHeadSpecializedExactKernel(
        name: "laguna_lmhead_exact_mask_row_skip_v1",
        inlineCandidate: false,
        laneUniform: false,
        rowSkip: true
    )

private let lagunaLmHeadMaskLaneUniformRowSkipKernel =
    makeLagunaLmHeadSpecializedExactKernel(
        name: "laguna_lmhead_exact_mask_lane_uniform_row_skip_v1",
        inlineCandidate: false,
        laneUniform: true,
        rowSkip: true
    )

private func lagunaLmHeadSelectedExactKernel(
    inlineCandidate: Bool
) -> MLXFast.MLXFastKernel {
    if inlineCandidate {
        if lagunaLmHeadLaneUniformEnabled && lagunaLmHeadRowSkipEnabled {
            return lagunaLmHeadInlineLaneUniformRowSkipKernel
        }
        if lagunaLmHeadLaneUniformEnabled {
            return lagunaLmHeadInlineLaneUniformKernel
        }
        if lagunaLmHeadRowSkipEnabled {
            return lagunaLmHeadInlineRowSkipKernel
        }
        return lagunaLmHeadInlineExactKernel
    }

    if lagunaLmHeadLaneUniformEnabled && lagunaLmHeadRowSkipEnabled {
        return lagunaLmHeadMaskLaneUniformRowSkipKernel
    }
    if lagunaLmHeadLaneUniformEnabled {
        return lagunaLmHeadMaskLaneUniformKernel
    }
    if lagunaLmHeadRowSkipEnabled {
        return lagunaLmHeadMaskRowSkipKernel
    }
    return lagunaLmHeadExactKernel
}

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

        let coarseOut: [MLXArray]
        if lagunaLmHeadInlineMaskEnabled {
            let coarseKernel =
                lagunaLmHeadCoarseUseV1
                ? lagunaLmHeadInlineCoarseKernelV1 : lagunaLmHeadInlineCoarseKernel
            coarseOut = coarseKernel(
                [x, codes, scales],
                grid: (vocab / 8 * 256, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[vocab], [vocab]],
                outputDTypes: [.float32, .float32]
            )
        } else {
            // Kill-switch fallback: the new tip's original three-output
            // coarse kernels still materialize `coarse_bf` for the retained
            // selector/exact path.
            let coarseKernel =
                lagunaLmHeadCoarseUseV1 ? lagunaLmHeadCoarseKernelV1 : lagunaLmHeadCoarseKernel
            coarseOut = coarseKernel(
                [x, codes, scales],
                grid: (vocab / 8 * 256, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[vocab], [vocab], [vocab]],
                outputDTypes: [.float32, .float32, .bfloat16]
            )
        }
        let coarse = coarseOut[0]
        let delta = coarseOut[1]

        // Threshold on GPU: L = max(coarse - delta). The exact-beta default
        // constructs BF16(L)'s lower midpoint with integer bit manipulation;
        // its kill switch retains the previous `L - |L|/64` second pass.
        // Both variants share the identical fused first-pass reduction.
        let lowerMaxPartials = lagunaLmHeadLowerMaxStage1Kernel(
            [coarse, delta],
            grid: (224, 128, 1),
            threadGroup: (224, 1, 1),
            outputShapes: [[128]],
            outputDTypes: [.float32]
        )[0]
        let thresholdKernel =
            lagunaLmHeadExactBetaEnabled
            ? lagunaLmHeadLowerMaxExactThresholdKernel
            : lagunaLmHeadLowerMaxThresholdKernel
        let thr = thresholdKernel(
            [lowerMaxPartials],
            grid: (32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[1]],
            outputDTypes: [.float32]
        )[0]

        // One threadgroup per 32 output rows, covering the vocabulary exactly
        // once (100352 == 3136 * 32). Every slot has exactly one owning lane.
        let assembled: MLXArray
        if lagunaLmHeadInlineMaskEnabled {
            let exactKernel = lagunaLmHeadSelectedExactKernel(inlineCandidate: true)
            assembled = exactKernel(
                [coarse, delta, thr, lmHeadWeight, x],
                grid: (vocab / 32 * 256, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[vocab]],
                outputDTypes: [.bfloat16]
            )[0]
        } else {
            // Kill-switch fallback: byte-for-byte the new tip's selector call
            // and exact-kernel inputs, including the stored BF16 coarse output.
            let coarseBF = coarseOut[2]
            let isCandidate = lagunaLmHeadSelectKernel(
                [coarse, delta, thr],
                grid: (vocab, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[vocab]],
                outputDTypes: [.uint8]
            )[0]
            let exactKernel = lagunaLmHeadSelectedExactKernel(inlineCandidate: false)
            assembled = exactKernel(
                [coarseBF, lmHeadWeight, x, isCandidate],
                grid: (vocab / 32 * 256, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[vocab]],
                outputDTypes: [.bfloat16]
            )[0]
        }
        return assembled.reshaped([1, 1, vocab])
    }
}
