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
// notes/69 REPLACES STAGE 1 ONLY. The scanned coarse copy is now a
// directed-rounding affine INT6 quantization at group size 128 (1600 B/row,
// 160.6 MB/token) instead of MXFP8 gs32 (2112 B/row, 211.9 MB/token) --
// 24% fewer bytes at the DRAM wall AND a strictly tighter certified bound, so
// the candidate set shrinks too (|S| median 15 -> 1 on 512 real decode steps).
// Stages 2 and 3 and the entire correctness argument above are untouched: the
// int6 kernel emits the same three outputs with the same certified meaning.
// See the block above `lagunaLmHeadInt6CoarseKernel` for the format, the
// certificate, and why 4-bit does NOT work here. The MXFP8 copy stays
// resident (+160.6 MB total) so `DARKBLOOM_LMHEAD_INT6=0` selects the previous
// screen in the SAME binary, which the paired measurement protocol requires.

private let lagunaLmHeadPruneVocab = 100_352
private let lagunaLmHeadPruneHidden = 2048

/// Master switch for the certified two-pass decode lm_head (notes/68).
/// DEFAULT ON: unset, or any value other than "0", enables the certified
/// two-pass decode head and builds the MXFP8 coarse copy at init time.
/// Set `DARKBLOOM_LM_HEAD_PRUNE=0` to disable and restore the byte-identical
/// stock full lm_head pass.
let lagunaLmHeadPruneEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LM_HEAD_PRUNE"] != "0"

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

// ---------------------------------------------------------------------------
// INT6 coarse screen (notes/69). Same three-stage structure as the MXFP8 screen
// above -- coarse+certified-bound scan, dense candidate mask, blocked exact
// GEMV -- but the scanned copy is a directed-rounding affine int6 quantization
// at group size 128 instead of MXFP8 gs32. 1600 B/row instead of 2112 B/row
// (160.6 MB instead of 211.9 MB per decode token), and a TIGHTER certified
// bound, so the candidate set shrinks as well.
//
// WHY int6 AND NOT int4/mxfp4. The screen's headroom is thin: over 512 real
// captured decode activations the vocabulary bulk sits at coarse+delta ~ 11.6
// against a threshold of ~ 14.3, i.e. ~2.5 logits of slack. Scaling the MXFP8
// bound by a multiplier m and recounting the survivors gives
//     m       1.00  1.10  1.25  1.50   1.75    2.00    2.50
//     median   15    30    103  1703  17092   58789   99308   (of 100352)
// so any bound looser than ~1.25x collapses the screen. Every 4-bit format
// certifies at 2.0x (affine gs32) to 3.1x (mxfp4 gs32) and blows the screen
// wide open; int6 gs128 certifies at 0.60x -- TIGHTER than MXFP8 -- while
// reading 24% fewer bytes. That is the whole idea.
//
// FORMAT. For each group g of 128 consecutive elements of row i:
//     bias_g = largest bf16 <= min_j W_ij          (rounded DOWN)
//     s_g    = smallest bf16 >= (max_j W_ij - bias_g) / 63   (rounded UP)
//     q_ij   = round_to_nearest((W_ij - bias_g) / s_g)
//     what_ij = q_ij * s_g + bias_g
// The directed rounding is what makes the certificate clean:
//   (a) W_ij - bias_g >= min_j W_ij - bias_g >= 0, so q_ij >= 0;
//   (b) W_ij - bias_g <= max_j W_ij - bias_g <= 63 * s_g, so q_ij <= 63.
// The code therefore NEVER clamps, and
//       |W_ij - what_ij| = s_g * |(W_ij - bias_g)/s_g - q_ij| <= s_g / 2
// EXACTLY -- with no widening term for the storage format, because s_g and
// bias_g are themselves bf16 values used verbatim by both the packer and the
// kernel, not roundings of some other target. Round-to-nearest-away and
// round-to-nearest-even both satisfy |ratio - q| <= 1/2, so the bound does not
// depend on which one `round` picks.
// The build-time division (W - bias)/s is evaluated in f32, so the ratio the
// rounding sees can differ from the real ratio by <= 63 * 2^-24 ulp; that is
// 7.5e-6 relative to s/2, an order of magnitude inside the (1 + GAMMA) factor
// applied below. `buildLagunaLmHeadInt6Planes` additionally VERIFIES
// q in [0, 63] and |W - what| <= s/2 over every one of the 205.5M elements and
// refuses the format (falling back to the MXFP8 screen) if either fails, so
// the certificate does not rest on the float analysis alone.
//
// BOUND. Writing d_i = sum_g (s_g/2) * sum_{j in g} |x_j| and
// M_i = sum_g (64 s_g + |bias_g|) * sum_{j in g} |x_j|:
//   * d_i >= sum_j |x_j| * |W_ij - what_ij|, the quantization term;
//   * M_i >= sum_j |x_j| * |what_ij| since |what_ij| = |q s_g + bias_g|
//     <= 63 s_g + |bias_g| < 64 s_g + |bias_g|, AND M_i also dominates the sum
//     of the ABSOLUTE VALUES OF THE TERMS this kernel actually accumulates
//     (s_g * x_j q_ij and bias_g * x_j separately), which is what the float
//     rounding analysis needs -- the split into a q-part and a bias-part can
//     cancel, so bounding sum|x_j what_ij| alone would NOT be enough here.
//   * 64 * s_g is exact in f32 (a power-of-two scaling), so that coefficient
//     carries no rounding of its own.
// Every value on both the coarse path and the exact path is accumulated in f32
// through a chain of depth <= ~75 (64 sequential adds per lane, a 5-deep simd
// tree, then a handful of combines), so each kernel's rounding is bounded by
// 75 * 2^-24 = 4.5e-6 relative -- comfortably inside GAMMA = 2^-15 = 3.05e-5,
// the same margin notes/68 establishes for the MXFP8 screen (depth <= 96).
// delta_i = d_i*(1+GAMMA) + 2*GAMMA*M_i therefore covers the quantization
// error AND both kernels' float rounding, exactly as in the MXFP8 screen, and
// the downstream select/exact stages are used UNCHANGED.
//
// LAYOUT. Split-plane, so every load is uint4-aligned and every field extract
// is a static shift-and-mask (no 6-bit field ever straddles a word):
//     planeA  1024 B/row : high 4 bits of q, two elements per byte
//     planeB   512 B/row : low  2 bits of q, four elements per byte
//     meta       64 B/row: per group, (bf16 s_g, bf16 bias_g)
// A lane owns 64 consecutive elements = exactly HALF a group, so it needs
// exactly one (s, bias) pair and reads 2 uint4 of planeA + 1 uint4 of planeB.

/// Master switch for the int6 coarse screen. DEFAULT ON; set
/// `DARKBLOOM_LMHEAD_INT6=0` to fall back to the MXFP8 screen in the same
/// binary (the paired measurement protocol needs both arms in one build).
/// Ignored entirely when `DARKBLOOM_LM_HEAD_PRUNE=0` disables the screen.
let lagunaLmHeadInt6Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_INT6"] != "0"

/// Next declared band-sized chunk: the same construction at 5 bits (1344 B/row,
/// 134.9 MB/token). DEFAULT OFF -- set `DARKBLOOM_LMHEAD_INT5=1` to select it.
/// When on it takes priority over int6; if its certificate check fails the
/// screen falls back to int6, and then to MXFP8, in the same binary.
let lagunaLmHeadInt5Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_INT5"] == "1"

/// Debug-only per-token cross-check against the stock full BF16 lm_head.
/// OFF unless `DARKBLOOM_LMHEAD_VERIFY=1`; adds a full 411 MB GEMV and two
/// host readbacks per decode step, so it is for local verification only.
let lagunaLmHeadVerifyEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_VERIFY"] == "1"

/// Fused int6 coarse GEMV + certified bound + BF16 pre-fill. Structurally the
/// twin of `laguna_lmhead_mxfp8_coarse_v2`: one simdgroup per row, lane L owns
/// elements [64L, 64L+64), three simd_sum reductions, identical output
/// semantics (`coarse` f32, `delta` f32, `coarse_bf` bf16).
///
/// The per-lane work is a single FMA chain over the elements plus two
/// x-only sums:
///     S1 = sum x_j q_j   S2 = sum x_j   S3 = sum |x_j|
///     coarse += s*S1 + bias*S2      (linearity of what = q*s + bias)
///     d      += (s/2)*S3
///     M      += (64 s + |bias|)*S3
/// S2 and S3 are row-independent, but recomputing them per row costs two adds
/// per element and no extra loads, and keeps the kernel a pure function of its
/// inputs with the same one-row-per-simdgroup mapping as the MXFP8 arm.
private let lagunaLmHeadInt6CoarseKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_int6_coarse_v1",
    inputNames: ["x", "planeA", "planeB", "meta"],
    outputNames: ["coarse", "delta", "coarse_bf"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 8 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint4* pa =
            (const device uint4*)(planeA + size_t(row) * 1024 + lane * 32);
        const device uint4* pb =
            (const device uint4*)(planeB + size_t(row) * 512 + lane * 16);
        const device ushort4* xr = (const device ushort4*)(x + lane * 64);
        // meta is [vocab, 16, 2] ushort: group g occupies one uint32, low half
        // the bf16 scale bits, high half the bf16 bias bits. Lane L covers the
        // second half of group L/2.
        uint mt = ((const device uint*)(meta + size_t(row) * 32))[lane >> 1];
        float sc = as_type<float>((mt & 0xFFFFu) << 16);
        float bi = as_type<float>((mt >> 16) << 16);

        uint4 a0 = pa[0];
        uint4 a1 = pa[1];
        uint4 b0 = pb[0];

        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        #pragma clang loop unroll(full)
        for (uint w = 0; w < 8; ++w) {
            // planeA word w holds elements 8w..8w+7 as nibbles at bit 4i.
            uint aw = (w < 4u) ? a0[w & 3u] : a1[w & 3u];
            // planeB word w/2 holds elements 16(w/2)..+15 as 2-bit fields;
            // this word's elements start at bit 16*(w&1).
            uint bw = b0[w >> 1];
            uint bsh = (w & 1u) ? 16u : 0u;
            #pragma clang loop unroll(full)
            for (uint h = 0; h < 2; ++h) {
                uint4 hi4 =
                    (uint4(aw) >> (uint4(0u, 4u, 8u, 12u) + 16u * h)) & 15u;
                uint4 lo2 =
                    (uint4(bw) >> (uint4(0u, 2u, 4u, 6u) + 8u * h + bsh)) & 3u;
                float4 qf = float4((hi4 << 2) | lo2);
                // bf16 -> f32 is exactly bits<<16 for every value class.
                float4 xv = as_type<float4>(uint4(xr[w * 2 + h]) << 16);
                float4 ax = metal::abs(xv);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4; ++k) {
                    s1 += xv[k] * qf[k];
                    s2 += xv[k];
                    s3 += ax[k];
                }
            }
        }
        float c_acc = simd_sum(sc * s1 + bi * s2);
        float d_acc = simd_sum((sc * 0.5f) * s3);
        float m_acc = simd_sum((64.0f * sc + metal::abs(bi)) * s3);
        if (lane == 0) {
            coarse[row] = c_acc;
            delta[row] = d_acc * (1.0f + GAMMA) + (2.0f * GAMMA) * m_acc;
            coarse_bf[row] = bfloat(c_acc);
        }
        """,
    ensureRowContiguous: true
)

/// int5 gs128 twin of `laguna_lmhead_int6_coarse_v1`. IDENTICAL arithmetic,
/// identical lane mapping, identical certified-bound algebra; the only change
/// is that the low plane carries ONE bit per element instead of two, so it is
/// 256 B/row instead of 512 B/row and a lane reads a uint2 instead of a uint4.
/// Total 1344 B/row = 134.9 MB/token (vs int6's 1600 B/row = 160.6 MB/token).
///
/// The certificate carries over verbatim with 63 -> 31 and 64 -> 32:
///   bias_g = largest bf16 <= min_j W_ij;  s_g = smallest bf16 >= (max-bias)/31
///   0 <= q <= 31 unreachable-clamp by construction;  cell = s_g/2 exactly;
///   M_i = sum_g (32 s_g + |bias_g|) * sum_{j in g} |x_j|   (32 s_g exact in f32)
/// The cell is 63/31 = 2.03x wider than int6's for the same group, which is
/// the entire tail-risk story: see notes/70 for the measured |S| distribution.
private let lagunaLmHeadInt5CoarseKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_int5_coarse_v1",
    inputNames: ["x", "planeA", "planeLow", "meta"],
    outputNames: ["coarse", "delta", "coarse_bf"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 8 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint4* pa =
            (const device uint4*)(planeA + size_t(row) * 1024 + lane * 32);
        // 64 elements at one bit each = 8 bytes = one uint2.
        const device uint2* pc =
            (const device uint2*)(planeLow + size_t(row) * 256 + lane * 8);
        const device ushort4* xr = (const device ushort4*)(x + lane * 64);
        uint mt = ((const device uint*)(meta + size_t(row) * 32))[lane >> 1];
        float sc = as_type<float>((mt & 0xFFFFu) << 16);
        float bi = as_type<float>((mt >> 16) << 16);

        uint4 a0 = pa[0];
        uint4 a1 = pa[1];
        uint2 c0 = pc[0];

        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        #pragma clang loop unroll(full)
        for (uint w = 0; w < 8; ++w) {
            // planeA word w holds elements 8w..8w+7 as nibbles at bit 4i.
            uint aw = (w < 4u) ? a0[w & 3u] : a1[w & 3u];
            // Low plane: element 8w+i is bit 8w+i of the lane's 8 bytes, i.e.
            // bit 8*(w&3) + i of low word w/4.
            uint cw = (w < 4u) ? c0.x : c0.y;
            uint csh = 8u * (w & 3u);
            #pragma clang loop unroll(full)
            for (uint h = 0; h < 2; ++h) {
                uint4 hi4 =
                    (uint4(aw) >> (uint4(0u, 4u, 8u, 12u) + 16u * h)) & 15u;
                uint4 lo1 =
                    (uint4(cw) >> (uint4(0u, 1u, 2u, 3u) + csh + 4u * h)) & 1u;
                float4 qf = float4((hi4 << 1) | lo1);
                float4 xv = as_type<float4>(uint4(xr[w * 2 + h]) << 16);
                float4 ax = metal::abs(xv);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4; ++k) {
                    s1 += xv[k] * qf[k];
                    s2 += xv[k];
                    s3 += ax[k];
                }
            }
        }
        float c_acc = simd_sum(sc * s1 + bi * s2);
        float d_acc = simd_sum((sc * 0.5f) * s3);
        float m_acc = simd_sum((32.0f * sc + metal::abs(bi)) * s3);
        if (lane == 0) {
            coarse[row] = c_acc;
            delta[row] = d_acc * (1.0f + GAMMA) + (2.0f * GAMMA) * m_acc;
            coarse_bf[row] = bfloat(c_acc);
        }
        """,
    ensureRowContiguous: true
)

/// Largest bf16 value <= `v` (v arbitrary sign), as float32. Bit truncation
/// rounds toward ZERO, which is toward +inf for negatives, so a negative that
/// truncated upward is stepped one bf16 ulp further from zero.
private func lagunaBF16RoundedDown(_ v: MLXArray) -> MLXArray {
    let truncBits = v.view(dtype: .uint32) & UInt32(0xFFFF_0000)
    let trunc = truncBits.view(dtype: .float32)
    let stepped = (truncBits + UInt32(0x0001_0000)).view(dtype: .float32)
    return which(trunc .> v, stepped, trunc)
}

/// Smallest bf16 value >= `v`, for v >= 0. Truncation never overshoots a
/// non-negative value, so one ulp up is enough whenever it undershoots.
private func lagunaBF16RoundedUp(_ v: MLXArray) -> MLXArray {
    let truncBits = v.view(dtype: .uint32) & UInt32(0xFFFF_0000)
    let trunc = truncBits.view(dtype: .float32)
    let stepped = (truncBits + UInt32(0x0001_0000)).view(dtype: .float32)
    return which(trunc .< v, stepped, trunc)
}

/// The three resident int-`bits` planes. `bits` is 6 (low plane 512 B/row,
/// 1600 B/row total, 160.6 MB) or 5 (low plane 256 B/row, 1344 B/row total,
/// 134.9 MB).
struct LagunaLmHeadIntBPlanes {
    let bits: Int
    let planeA: MLXArray    // [vocab, 1024] uint8, high 4 bits, 2 elements/byte
    let planeLow: MLXArray  // [vocab, 512 or 256] uint8, low bits(s)
    let meta: MLXArray      // [vocab, 32] uint16, per gs128 (bf16 s, bf16 bias)
}

/// Builds the int-`bits` planes from the materialized BF16 lm_head, verifying
/// the certificate on the real weight before returning. Returns nil (-> the
/// next screen down: int5 falls back to int6, int6 to MXFP8) if any element
/// would clamp or exceed the half-cell, or if the shape is not the expected
/// [100352, 2048].
///
/// `bits` is 6 or 5. The high nibble always goes to planeA (1024 B/row); the
/// remaining `bits - 4` bits go to the low plane, packed `8/(bits-4)` elements
/// per byte -- 4/byte (512 B/row) at int6, 8/byte (256 B/row) at int5. The
/// arithmetic, the directed rounding and the certificate are identical for
/// both; only the code width and the low-plane packing density change.
///
/// Runs in row chunks so the f32 working set stays ~100 MB rather than
/// materializing an 822 MB f32 view of the whole head at once.
func buildLagunaLmHeadIntBPlanes(
    lmHeadWeight w: MLXArray, bits: Int
) -> LagunaLmHeadIntBPlanes? {
    let vocab = lagunaLmHeadPruneVocab
    let hidden = lagunaLmHeadPruneHidden
    let groups = hidden / 128
    guard w.shape == [vocab, hidden], w.dtype == .bfloat16 else { return nil }
    guard bits == 5 || bits == 6 else { return nil }
    let lowBits = bits - 4                  // 1 or 2
    let perLowByte = 8 / lowBits            // 8 or 4
    let maxCode = Float((1 << bits) - 1)    // 31 or 63

    var planeAChunks: [MLXArray] = []
    var planeLowChunks: [MLXArray] = []
    var metaChunks: [MLXArray] = []
    let chunk = 6272  // 16 chunks; 100352 == 16 * 6272

    var rowStart = 0
    while rowStart < vocab {
        let rows = Swift.min(chunk, vocab - rowStart)
        let wc = w[rowStart ..< (rowStart + rows)].asType(.float32)
        let g = wc.reshaped([rows, groups, 128])
        let lo = MLX.min(g, axis: 2)
        let hi = MLX.max(g, axis: 2)
        let bias = lagunaBF16RoundedDown(lo)
        // Guard the all-equal group: hi - bias is then 0 (lm_head is bf16, so
        // lo is already a bf16 value and bias == lo), which would divide by
        // zero. Any strictly positive scale reconstructs the group exactly,
        // because every element equals bias and therefore codes to q = 0.
        let rawScale = lagunaBF16RoundedUp((hi - bias) / maxCode)
        let scale = which(rawScale .> Float(0), rawScale, MLXArray(Float.leastNormalMagnitude))
        let q = round((g - bias.reshaped([rows, groups, 1]))
            / scale.reshaped([rows, groups, 1]))

        // Certificate check on the REAL weight, before anything is retained.
        let what = q * scale.reshaped([rows, groups, 1]) + bias.reshaped([rows, groups, 1])
        let err = abs(g - what)
        let cell = (scale * Float(0.5)).reshaped([rows, groups, 1])
        let ok = MLX.min(q).item(Float.self) >= 0
            && MLX.max(q).item(Float.self) <= maxCode
            && MLX.max(err - cell).item(Float.self) <= 0
        guard ok else {
            FileHandle.standardError.write(Data(
                ("mlxfast: lm_head int\(bits): certificate check failed at row "
                    + "\(rowStart); falling back\n").utf8))
            return nil
        }

        let qi = q.asType(.uint8).reshaped([rows, hidden])
        let hi4 = (qi >> UInt8(lowBits)) & UInt8(0x0F)
        let low = qi & UInt8((1 << lowBits) - 1)
        let hiPair = hi4.reshaped([rows, hidden / 2, 2])
        planeAChunks.append(hiPair[0..., 0..., 0] | (hiPair[0..., 0..., 1] << UInt8(4)))
        // Low plane: `perLowByte` consecutive elements per byte, element t at
        // bit lowBits*(t % perLowByte) -- the same ascending order planeA uses.
        let lowGroup = low.reshaped([rows, hidden / perLowByte, perLowByte])
        var lowByte = lowGroup[0..., 0..., 0]
        for slot in 1 ..< perLowByte {
            lowByte = lowByte | (lowGroup[0..., 0..., slot] << UInt8(lowBits * slot))
        }
        planeLowChunks.append(lowByte)
        let sBits = (scale.view(dtype: .uint32) >> UInt32(16)).asType(.uint16)
        let bBits = (bias.view(dtype: .uint32) >> UInt32(16)).asType(.uint16)
        metaChunks.append(
            stacked([sBits, bBits], axis: 2).reshaped([rows, groups * 2]))
        eval(planeAChunks.last!, planeLowChunks.last!, metaChunks.last!)
        rowStart += rows
    }

    let planes = LagunaLmHeadIntBPlanes(
        bits: bits,
        planeA: concatenated(planeAChunks, axis: 0),
        planeLow: concatenated(planeLowChunks, axis: 0),
        meta: concatenated(metaChunks, axis: 0))
    eval(planes.planeA, planes.planeLow, planes.meta)
    return planes
}

/// Retained init-time MXFP8 coarse copy of lm_head plus the pruned decode
/// forward. Built once (untimed init) by
/// `LagunaRuntimeModel.prepareFusedRuntimeWeights` when
/// `lagunaLmHeadPruneEnabled` (DARKBLOOM_LM_HEAD_PRUNE, default ON; set "0"
/// to disable); ~212 MB additional resident memory.
final class LagunaLmHeadPruner {
    let codes: MLXArray   // [100352, 2048] uint8 e4m3 elements
    let scales: MLXArray  // [100352, 64] uint8 e8m0 group scales
    /// Non-nil when an int-b gs128 screen is enabled AND its build-time
    /// certificate check passed on the real weight. When set it REPLACES the
    /// MXFP8 coarse scan (160.6 MB/token at int6, 134.9 MB/token at int5,
    /// instead of 211.9 MB/token); the MXFP8 copy stays resident so
    /// `DARKBLOOM_LMHEAD_INT6=0` is a same-binary A/B.
    var intB: LagunaLmHeadIntBPlanes?

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

    /// Pruned decode lm_head: full [vocab] BF16 logits row, bit-identical to
    /// the stock pass in every candidate slot and certified-below elsewhere,
    /// so the downstream argmax emits the stock token.
    func logits(hidden: MLXArray, lmHeadWeight: MLXArray) -> MLXArray {
        precondition(hidden.dtype == .bfloat16 && hidden.size == lagunaLmHeadPruneHidden)
        let vocab = lagunaLmHeadPruneVocab
        let x = hidden.reshaped([lagunaLmHeadPruneHidden])

        // Coarse scan: int6 gs128 when it built and certified, else MXFP8.
        // Both produce the same three outputs with the same certified
        // semantics, so everything downstream is shared verbatim.
        let coarseOut: [MLXArray]
        if let intB {
            let kernel = intB.bits == 5
                ? lagunaLmHeadInt5CoarseKernel : lagunaLmHeadInt6CoarseKernel
            coarseOut = kernel(
                [x, intB.planeA, intB.planeLow, intB.meta],
                grid: (vocab / 8 * 256, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[vocab], [vocab], [vocab]],
                outputDTypes: [.float32, .float32, .bfloat16]
            )
        } else {
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

        // One threadgroup per 32 output rows, covering the vocabulary exactly
        // once (100352 == 3136 * 32). Every slot has exactly one owning lane.
        let assembled = lagunaLmHeadExactKernel(
            [coarseBF, lmHeadWeight, x, isCandidate],
            grid: (vocab / 32 * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[vocab]],
            outputDTypes: [.bfloat16]
        )[0]
        if lagunaLmHeadVerifyEnabled {
            verifyAgainstStock(assembled: assembled, x: x, lmHeadWeight: lmHeadWeight,
                               coarse: coarse, delta: delta)
        }
        return assembled.reshaped([1, 1, vocab])
    }

    /// Debug-only (`DARKBLOOM_LMHEAD_VERIFY=1`): re-run the stock full BF16
    /// GEMV for this token and assert that the screened row argmaxes to the
    /// same vocabulary slot and that the certified bound really contains the
    /// stock logit for every row. Reports to stderr and never mutates the
    /// returned logits, so a verified run and a scored run execute the same
    /// screen; it is not on the timed path.
    private func verifyAgainstStock(
        assembled: MLXArray, x: MLXArray, lmHeadWeight: MLXArray,
        coarse: MLXArray, delta: MLXArray
    ) {
        let stock = matmul(lmHeadWeight, x.reshaped([lagunaLmHeadPruneHidden, 1]))
            .reshaped([lagunaLmHeadPruneVocab])
        let stockF = stock.asType(.float32)
        let ours = argMax(assembled).item(Int.self)
        let theirs = argMax(stock).item(Int.self)
        let escapes = (abs(stockF - coarse) .> delta).sum().item(Int.self)
        let candidates = ((coarse + delta) .>= (coarse - delta).max()).sum().item(Int.self)
        let tag = (ours == theirs && escapes == 0) ? "ok" : "MISMATCH"
        FileHandle.standardError.write(Data(
            ("mlxfast: lmhead verify \(tag) screen_argmax=\(ours) stock_argmax=\(theirs) "
                + "bound_escapes=\(escapes) candidates=\(candidates)\n").utf8))
    }
}
