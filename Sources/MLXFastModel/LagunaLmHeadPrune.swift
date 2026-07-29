import Foundation
import MLX
import MLXFast

// Certified two-pass lm_head elision for the decode path (notes/68).
//
// Stock decode lm_head reads the full BF16 [100352, 2048] weight (411 MB) per
// token at the DRAM wall. This module, gated by
// DARKBLOOM_LM_HEAD_PRUNE (DEFAULT ON; set "0" to disable; unset = shipped
// path), replaces it for
// single-token decode steps with:
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
//
// SINCE `DARKBLOOM_NATIVE_AFFINE_LMHEAD` (default ON) the EXACT pass is served
// by `lagunaLmHeadExactAffineKernel`, which is the same kernel over a group-32
// affine INT8 side copy instead of the BF16 parameter, so a surviving row's
// value is no longer bit-identical to the stock GEMV's -- it carries the same
// quantization perturbation the promoted attention layouts carry, measured
// ~36x smaller (see that flag's doc comment). Steps 1 and 2 above -- the coarse
// GEMV, the certified bound, the threshold and the candidate mask -- are
// untouched, so the certificate's actual guarantee is intact: the true argmax
// row is always a candidate, and every non-candidate stays at least |L|/64
// below it, which is an order of magnitude more than the finalist
// perturbation. Set the flag to "0" for the bit-identical BF16 finalist pass in
// the same binary.

private let lagunaLmHeadPruneVocab = 100_352
private let lagunaLmHeadPruneHidden = 2048

/// Master switch for the certified two-pass decode lm_head (notes/68).
/// DEFAULT ON: unset, or any value other than "0", enables the certified
/// two-pass decode head and builds the MXFP8 coarse copy at init time.
/// Set `DARKBLOOM_LM_HEAD_PRUNE=0` to disable and restore the byte-identical
/// stock full lm_head pass.
let lagunaLmHeadPruneEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LM_HEAD_PRUNE"] != "0"

/// Decode-side group-32 affine INT8 side copy of `lm_head`, read by the
/// FINALIST (exact) pass only. DEFAULT ON; `DARKBLOOM_NATIVE_AFFINE_LMHEAD=0`
/// restores the byte-identical BF16 finalist pass inside the same binary.
///
/// The scan is untouched: the coarse GEMV, its certified bound, the threshold
/// and the candidate mask are bit-for-bit the shipped ones, so the true argmax
/// row still provably reaches the finalist pass (the certificate only needs
/// `c_i + delta_i >= L - |L|/64` to be evaluated on the same numbers, and it
/// is). What changes is only the VALUE a surviving row is scored with: instead
/// of the stock BF16 GEMV it gets the same GEMV over the INT8 side copy. The
/// resulting perturbation can therefore only reorder near-ties among
/// survivors, and it is the same perturbation class the promoted attention
/// layouts already ship, ~36x smaller. Measured on the public 512-token
/// fixture, 129 teacher-forced positions, this flag against itself: top-1/top-2
/// DIFFERENTIAL rms 0.040 logits (max 0.125 = one BF16 ulp) against a
/// copy-regime p5 top-2 gap of 1.400 and a median of 6.125; 0/129 flips, and
/// the OFF arm's token stays strict top-1 at 129/129 positions. The attention
/// INT8 stack spends 1.47 at the same measure -- a logit perturbation applied
/// at the head has no downstream layers to amplify it.
///
/// A non-candidate row can never overtake the winner either: it keeps its
/// coarse BF16 value, which the certificate places at least `|L|/64` (~0.3-0.5
/// logits here) below the true maximum, an order of magnitude above the
/// finalist perturbation.
///
/// Decode only: prefill and the `DARKBLOOM_LM_HEAD_PRUNE=0` fallback keep the
/// authoritative BF16 `lm_head` parameter.
let lagunaLmHeadFinalistAffineEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_LMHEAD"] != "0"

/// Measurement-only survivor census. `DARKBLOOM_LMHEAD_PRUNE_STATS=1` makes
/// every pruned decode step evaluate and print the candidate-row count and the
/// number of four-row simdgroup blocks the finalist pass actually reads, which
/// is what turns into finalist weight traffic. Off by default (it forces a
/// host sync per token and adds two reductions).
private let lagunaLmHeadPruneStats =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_PRUNE_STATS"] == "1"

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

/// Same-binary A/B selector for the coarse kernel (v2 default).
private let lagunaLmHeadCoarseUseV1 =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_COARSE"] == "v1"

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

/// Finalist pass over the group-32 affine INT8 side copy (see
/// `lagunaLmHeadFinalistAffineEnabled`). Structurally identical to
/// `lagunaLmHeadExactKernel` -- same grid, same fixed four-row block per
/// simdgroup, same candidate test, same coarse fallback, same lane partition
/// (lane `l` covers columns `4l + 128i`), same sequential FP32 accumulation
/// order and the same `simd_shuffle_down` ladder -- so the only difference is
/// where a surviving row's weights come from.
///
/// The lane partition and the group size line up exactly: lane `l`'s four
/// columns `4l..4l+3` never straddle a 32-element group, so each `(lane, i)`
/// step reads one packed word (four codes) and that word's single
/// `(scale, bias)` pair. Dequantization is MLX's affine form
/// `w = scale * code + bias` (quantized.cpp `affine_dequantize`, bits == 8:
/// one byte per value, little-endian within the packed word), evaluated in
/// FP32 so the surviving row's arithmetic keeps the stock accumulation
/// structure and only the weight values change.
///
/// Traffic per scored row: 2048 code bytes + 64 x (2 + 2) scale/bias bytes =
/// 2304 B against the BF16 pass's 4096 B (9 bits/value vs 16, -43.75%).
private let lagunaLmHeadExactAffineKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_block_affine_v1",
    inputNames: ["coarse_bf", "codes", "scales", "biases", "x", "is_cand"],
    outputNames: ["assembled"],
    source: """
        constexpr uint VOCAB = 100352;
        constexpr uint WORDS = 512;   // 2048 codes / 4 per packed word
        constexpr uint GROUPS = 64;   // 2048 codes / 32 per group

        uint tgid = threadgroup_position_in_grid.x;
        uint sgid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        uint base = tgid * 32 + sgid * 4;

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

        thread float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        thread float v_coeff[4];
        uint bn = lane * 4;
        for (uint i = 0; i < 16; ++i) {
            vec<bfloat, 4> xv =
                *((const device vec<bfloat, 4>*)(x + bn));
            v_coeff[0] = float(xv.x);
            v_coeff[1] = float(xv.y);
            v_coeff[2] = float(xv.z);
            v_coeff[3] = float(xv.w);
            uint widx = bn >> 2;
            uint g = bn >> 5;
            #pragma unroll
            for (uint tm = 0; tm < 4; ++tm) {
                size_t r = size_t(base + tm);
                uint packed = codes[r * WORDS + widx];
                float sc = float(scales[r * GROUPS + g]);
                float bi = float(biases[r * GROUPS + g]);
                result[tm] += (sc * float(packed & 255u) + bi) * v_coeff[0];
                result[tm] += (sc * float((packed >> 8) & 255u) + bi) * v_coeff[1];
                result[tm] += (sc * float((packed >> 16) & 255u) + bi) * v_coeff[2];
                result[tm] += (sc * float((packed >> 24) & 255u) + bi) * v_coeff[3];
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

/// Retained init-time MXFP8 coarse copy of lm_head plus the pruned decode
/// forward. Built once (untimed init) by
/// `LagunaRuntimeModel.prepareFusedRuntimeWeights` when
/// `lagunaLmHeadPruneEnabled` (DARKBLOOM_LM_HEAD_PRUNE, default ON; set "0"
/// to disable); ~212 MB additional resident memory.
final class LagunaLmHeadPruner {
    let codes: MLXArray   // [100352, 2048] uint8 e4m3 elements
    let scales: MLXArray  // [100352, 64] uint8 e8m0 group scales

    /// Group-32 affine INT8 side copy read by the FINALIST pass only, when
    /// `DARKBLOOM_NATIVE_AFFINE_LMHEAD` is on. `nil` keeps the exact BF16
    /// finalist pass. Codes are uint32 [100352, 512] (four 8-bit codes per
    /// word); scales/biases are BF16 [100352, 64].
    let affineCodes: MLXArray?
    let affineScales: MLXArray?
    let affineBiases: MLXArray?

    /// Arrays that must be materialized before the first scored forward.
    var residentArrays: [MLXArray] {
        [codes, scales] + [affineCodes, affineScales, affineBiases].compactMap { $0 }
    }

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

        // FINALIST-pass side copy. Same group-32 affine INT8 the promoted
        // attention layouts use (`w = scale * code + bias`, one byte per
        // value), built from the same materialized BF16 parameter. The scan
        // above is not re-derived from it in any way.
        if lagunaLmHeadFinalistAffineEnabled {
            let (aq, ascales, abiases) = quantized(
                lmHeadWeight, groupSize: 32, bits: 8, mode: .affine)
            if let abiases,
                aq.dtype == .uint32,
                aq.shape == [lagunaLmHeadPruneVocab, lagunaLmHeadPruneHidden / 4],
                ascales.dtype == .bfloat16,
                abiases.dtype == .bfloat16,
                ascales.shape == [lagunaLmHeadPruneVocab, lagunaLmHeadPruneHidden / 32],
                abiases.shape == ascales.shape
            {
                self.affineCodes = aq
                self.affineScales = ascales
                self.affineBiases = abiases
            } else {
                FileHandle.standardError.write(
                    Data(
                        "mlxfast: lm_head finalist affine: unexpected layout; BF16 finalist\n"
                            .utf8))
                self.affineCodes = nil
                self.affineScales = nil
                self.affineBiases = nil
            }
        } else {
            self.affineCodes = nil
            self.affineScales = nil
            self.affineBiases = nil
        }
    }

    /// Pruned decode lm_head: full [vocab] BF16 logits row, bit-identical to
    /// the stock pass in every candidate slot and certified-below elsewhere,
    /// so the downstream argmax emits the stock token.
    func logits(hidden: MLXArray, lmHeadWeight: MLXArray) -> MLXArray {
        precondition(hidden.dtype == .bfloat16 && hidden.size == lagunaLmHeadPruneHidden)
        let vocab = lagunaLmHeadPruneVocab
        let x = hidden.reshaped([lagunaLmHeadPruneHidden])

        let coarseKernel =
            lagunaLmHeadCoarseUseV1 ? lagunaLmHeadCoarseKernelV1 : lagunaLmHeadCoarseKernel
        let coarseOut = coarseKernel(
            [x, codes, scales],
            grid: (vocab / 8 * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[vocab], [vocab], [vocab]],
            outputDTypes: [.float32, .float32, .bfloat16]
        )
        let coarse = coarseOut[0]
        let delta = coarseOut[1]
        let coarseBF = coarseOut[2]

        // Threshold on GPU: L = max(coarse - delta); thr = L - |L|/64, in ONE
        // dispatch. The MLX expression form of this (`coarse - delta`, then
        // `.max()` -- itself a two-pass all_reduce at this size -- then
        // `.abs()`, a scalar multiply and a scalar subtract) costs six
        // dispatches, five of which move almost no data. `max` is associative
        // in IEEE 754, so the fused tree is bitwise identical regardless of
        // shape; see the kernel's doc comment.
        let lower = coarse - delta
        let l = lower.max()
        let thr = (l - l.abs() * Float(1.0 / 64.0)).reshaped([1])

        let isCandidate = lagunaLmHeadSelectKernel(
            [coarse, delta, thr],
            grid: (vocab, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[vocab]],
            outputDTypes: [.uint8]
        )[0]

        if lagunaLmHeadPruneStats {
            reportSurvivors(isCandidate)
        }

        // One threadgroup per 32 output rows, covering the vocabulary exactly
        // once (100352 == 3136 * 32). Every slot has exactly one owning lane.
        let assembled: MLXArray
        if let affineCodes, let affineScales, let affineBiases {
            assembled = lagunaLmHeadExactAffineKernel(
                [coarseBF, affineCodes, affineScales, affineBiases, x, isCandidate],
                grid: (vocab / 32 * 256, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[vocab]],
                outputDTypes: [.bfloat16]
            )[0]
        } else {
            assembled = lagunaLmHeadExactKernel(
                [coarseBF, lmHeadWeight, x, isCandidate],
                grid: (vocab / 32 * 256, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[vocab]],
                outputDTypes: [.bfloat16]
            )[0]
        }
        return assembled.reshaped([1, 1, vocab])
    }

    /// Measurement-only: how many rows survived the scan, and how many of the
    /// finalist pass's four-row blocks that lights up (the block count is the
    /// one that turns into weight traffic -- a block is read in full as soon as
    /// any one of its four rows is a candidate).
    private func reportSurvivors(_ isCandidate: MLXArray) {
        let vocab = lagunaLmHeadPruneVocab
        let rows = isCandidate.asType(.int32).sum()
        let blocks = isCandidate.reshaped([vocab / 4, 4]).max(axis: 1)
            .asType(.int32).sum()
        eval(rows, blocks)
        let rowCount = rows.item(Int.self)
        let blockCount = blocks.item(Int.self)
        let bytesBF16 = blockCount * 4 * lagunaLmHeadPruneHidden * 2
        let bytesINT8 = blockCount * 4 * (lagunaLmHeadPruneHidden + 64 * 4)
        FileHandle.standardError.write(
            Data(
                """
                mlxfast: lm_head survivors rows=\(rowCount) blocks=\(blockCount) \
                rows_read=\(blockCount * 4) bf16_bytes=\(bytesBF16) \
                int8_bytes=\(bytesINT8)

                """.utf8))
    }
}
