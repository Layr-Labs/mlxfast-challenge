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
//      DEFAULT (DARKBLOOM_LMHEAD_RATIO_BOUND, default ON) the kernel emits the
//      strictly-not-smaller closed form d_i*(1+61*gamma), which is legal
//      because |decode_e4m3(code)| <= 30*hs8(code) for all 256 codes gives
//      m_i <= 30*d_i termwise; that drops the m_i accumulator and its
//      SIMD reduction. Set the variable to "0" for the accepted two-term form.
//      Ratio certificate, all 256 codes, using this file's own decoders
//      (mag = b & 127, e = mag >> 3, m = mag & 7):
//        e == 0 (denormal, mag 0..7): decode = m*2^-9, hs8 = 2^-10,
//          ratio = 2m <= 14 (attained mag 7).
//        e >= 1, mag != 126: decode = (8+m)*2^(e-10), hs8 = 2^(e-11),
//          ratio = 16 + 2m <= 30 (attained m = 7, e.g. mag 15 and mag 127,
//          the latter decoding to 480 with hs8 = 16).
//        mag == 126 (saturated top, decode 448, open cell hs8 186):
//          ratio = 448/186 = 2.409.
//      So max ratio = 30, exactly attained. The inequality is per element
//      over the SAME elements with the SAME non-negative group scales sd_g
//      (e8m0 decodes to a positive power of two for every byte), and |x_j|
//      >= 0, so it lifts termwise to the row sums: m_i <= 30*d_i, hence
//      d_i*(1+gamma) + 2*gamma*m_i <= d_i*(1 + gamma + 60*gamma)
//      = d_i*(1 + 61*gamma). Both scalar constants are exact in FP32
//      (1+2^-15 and 1+61*2^-15 need 15 mantissa bits) and 2*gamma = 2^-14
//      is a power of two, so the substitution is exact at the constant
//      level: (1+gamma) + 30*(2*gamma) == (1+61*gamma) bit-for-bit.
//      FP32 evaluation note: the one-term form can land at most ~3 ulps
//      (~1.8e-7 relative) below the two-term form when m_acc sits exactly
//      at 30*d_acc. That is absorbed many times over by the certificate's
//      own rounding headroom -- gamma = 2^-15 = 3.05e-5 against the
//      <= 96*2^-24 = 5.7e-6 of accumulated FP32 rounding it must cover,
//      a >5x margin -- so the emitted delta still bounds the true error.
//      DEFAULT (DARKBLOOM_LMHEAD_DELTA_BF16, default ON, nested inside the
//      ratio bound) delta_i is stored as BF16 rounded toward +infinity
//      instead of FP32, halving that buffer's write and both of its reads
//      (401,408 -> 200,704 bytes per token per traversal, ~0.6 MB/token in
//      all). delta is only ever COMPARED downstream and candidacy is
//      MONOTONE in it -- a larger delta lowers L = max(coarse - delta) and
//      hence the threshold, and raises each row's `coarse + delta` -- so
//      rounding it up only widens the certified bound and only grows the
//      candidate set. `coarse` stays FP32: it would have to round DOWN for
//      the L path and UP for the candidate test, which one buffer cannot do.
//      Set the variable to "0" for the FP32 delta round trip.
//      BOTH selectors are ARM-ORTHOGONAL: they apply identically on the
//      DARKBLOOM_LMHEAD_COARSE_V4 int6 arm (the shipped default), where the
//      flat half-cell makes the ratio 2*|q| instead of a format table. The
//      int6 codes decode to q = u - 32 and `buildInt6Planes` VERIFIES
//      max|q| <= 31 on the actual tensor (declining to the MXFP8 copy if it
//      ever fails), so per element the m term sd*|x_j|*|q_ij| is at most 62x
//      the d term (0.5*sd)*|x_j| -- same elements, same positive power-of-two
//      sd_g -- giving m_i <= 62*d_i and the closed form d_i*(1 + 125*gamma).
//      That substitution is likewise exact in FP32: (1+gamma) + 62*(2*gamma)
//      == 1 + 125*gamma bit-for-bit. See the int6 ratio-bound kernel below.
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
// The threshold beta widens the candidate set slightly vs the raw lower bound
// L; it is the BF16-cast safety margin from the assembly proof.
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

/// v4 coarse copy (D4): replaces the MXFP8 coarse pass with a planar-packed
/// symmetric int6 copy (nibble plane + 2-bit plane + e8m0-style power-of-two
/// group scales, 1600 B/row vs MXFP8's 2112 B/row, -24.2% coarse bytes).
/// The certified bound TIGHTENS: a uniform grid has a flat sd/2 absolute
/// half-cell, and on this tensor the resulting per-row delta is ~22% smaller
/// than the e4m3 half-ulp bound (offline, 130 captured decode hidden states:
/// candidates p50 2 / p90 7 / max 135 vs MXFP8's p50 12 / p90 37 / max 545).
/// The downstream lower-max/threshold/exact-mask machinery is unchanged; the
/// exact pass still recomputes every candidate row with the textual stock
/// BF16 GEMV, so the emitted token stays the stock token (same certificate
/// chain as notes/68, with hs8 replaced by the flat half-cell 0.5*sd).
/// DEFAULT ON (ranked-shipped); set `DARKBLOOM_LMHEAD_COARSE_V4=0` to restore
/// the MXFP8 pack16 coarse. Requires the
/// inline-mask path (ignored under DARKBLOOM_LMHEAD_INLINE_MASK=0).
private let lagunaLmHeadCoarseV4Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_COARSE_V4"] != "0"

/// One-line stderr trace hooks (DARKBLOOM_TRACE_FUSION=1) so a silently
/// declining v4 guard is visible in run logs.
private let lagunaTraceFusionEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_TRACE_FUSION"] == "1"

/// Replace the per-element `m = sum |x|*|what|` accumulation with its exact
/// global E4M3 bound `m <= 30*d`. This removes one multiply/add chain and one
/// SIMD reduction from the bandwidth-heavy coarse pass. Set to "0" for the
/// accepted three-accumulator implementation.
///
/// The bound is per-row arithmetic (see the ratio-bound kernels below), so it
/// is orthogonal to the pack16 threadgroup geometry: rows-per-threadgroup only
/// changes which simdgroup owns a row, never a row's lane partition, its
/// 32-lane `simd_sum` width, or its `sd_g` scale handling. This selector is
/// likewise orthogonal to `DARKBLOOM_LMHEAD_INLINE_MASK` -- both the
/// two-output inline twin and the three-output kill-switch twin honor it -- and
/// it is nested inside `DARKBLOOM_LMHEAD_COARSE`, whose `v1` arm stays the
/// verbatim scalar pack8 A/B reference.
///
/// It is ALSO orthogonal to `DARKBLOOM_LMHEAD_COARSE_V4`: the int6 arm carries
/// its own twin of this bound with the format-specific constant `m <= 62*d`
/// (flat half-cell, `|q| <= 31`) instead of the E4M3 table's `m <= 30*d`, so
/// `delta = d*(1 + 125*gamma)` there and `d*(1 + 61*gamma)` on the MXFP8 arm.
/// Setting this to "0" restores the accepted three-accumulator form on
/// whichever coarse arm is active.
private let lagunaLmHeadRatioBoundEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_RATIO_BOUND"] != "0"

/// Narrow the certified bound's round-trip buffer from FP32 to BF16, rounding
/// every stored value UP so the bound can only widen. Set to "0" to restore the
/// FP32 `delta` buffer.
///
/// `delta` is only ever COMPARED downstream -- `max_i(coarse - delta)` feeds the
/// threshold and `coarse + delta >= thr` selects candidates -- and candidacy is
/// MONOTONE in delta: a larger delta lowers `L` and therefore `thr`, and raises
/// each row's upper bound, so both movements admit candidates and neither drops
/// one. Rounding delta toward +infinity is therefore certified-safe by the same
/// argument the ratio bound uses, and the exact pass's extra candidates receive
/// the stock-exact GEMV value in place of their certified-below coarse value.
///
/// `coarse` deliberately stays FP32. It appears in `max_i(coarse - delta)`,
/// where it must round DOWN to stay conservative, and in `coarse + delta >= thr`
/// and `bfloat(coarse[r])`, where it must round UP -- contradictory directions
/// that a single narrowed buffer cannot satisfy. Pushing coarse's BF16 rounding
/// error (a half ulp, up to 2^-8 relative of |coarse|) into delta instead would
/// dwarf delta itself and collapse the pruning, so only delta narrows.
///
/// Nested inside `DARKBLOOM_LMHEAD_RATIO_BOUND` (and, on the MXFP8 arm, inside
/// the inline-mask and non-`v1` coarse arms): the accepted three-accumulator
/// form, the dense-selector kill switch and the scalar pack8 A/B reference all
/// keep their FP32 `delta` kernels byte-for-byte, so every OFF arm reaches text
/// that predates this change.
///
/// Arm-orthogonal like the ratio bound: the `DARKBLOOM_LMHEAD_COARSE_V4` int6
/// arm has its own narrowed-store twin, and both arms feed the SAME two BF16
/// consumers below, so one arm selection always implies one `delta` dtype for
/// every downstream reader.
private let lagunaLmHeadDeltaBF16Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LMHEAD_DELTA_BF16"] != "0"

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
    name: "laguna_lmhead_mxfp8_coarse_pack16_v3",
    inputNames: ["x", "codes", "scales"],
    outputNames: ["coarse", "delta", "coarse_bf"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 16 +
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

/// Same MXFP8 coarse logits and quantization-error sum as the pack16 kernel
/// above, but the roundoff term uses the exact format-wide inequality
/// `m <= 30*d` certified at the top of this file: over all 256 codes the
/// decoded magnitude divided by its `laguna_hs8` half-cell width is `2m <= 14`
/// for the e == 0 denormals, `16 + 2m <= 30` for every e >= 1 code other than
/// the saturated 126 (attained at magnitude 15 and at 127), and 448/186 =
/// 2.409 for magnitude 126 -- so the maximum is exactly 30. Thus the retained
/// `d*(1+gamma) + 2*gamma*m` is bounded by `d*(1+61*gamma)`, and the two
/// scalars agree exactly in FP32: `(1+gamma) + 30*(2*gamma) == 1+61*gamma`.
///
/// The inequality is per element, and both `d` and `m` accumulate the same
/// per-group `sd_g` scale over the same 2048 elements of one row, so it lifts
/// to the row sums unchanged under any threadgroup packing. The pack16 row
/// mapping (`threadgroup_position_in_grid.x * 16 + simdgroup_index_in_...`)
/// keeps one simdgroup per row, 32 lanes per row, and two groups per lane, so
/// `c_acc` and `d_acc` are the bit-identical FP32 values the pack16 kernel
/// computes; only the `m_acc` chain and its `simd_sum` are gone. These kernels
/// use no threadgroup shared memory, so widening the threadgroup to 512 threads
/// adds no shared array to size and no barrier to place.
private let lagunaLmHeadCoarseRatioBoundKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_mxfp8_coarse_ratio_bound_pack16_v1",
    inputNames: ["x", "codes", "scales"],
    outputNames: ["coarse", "delta", "coarse_bf"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 16 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint8_t* crow = codes + size_t(row) * 2048;
        const device uint8_t* srow = scales + size_t(row) * 64;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        for (uint gg = 0; gg < 2; ++gg) {
            uint g = 2 * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            const device uint4* cptr = (const device uint4*)(crow + g * 32);
            uint4 packed0 = cptr[0];
            uint4 packed1 = cptr[1];
            float cg = 0.0f;
            float dg = 0.0f;
            const device ushort4* xrow = (const device ushort4*)(x + g * 32);
            #pragma clang loop unroll(full)
            for (uint w = 0; w < 8; ++w) {
                uint word = (w < 4u) ? packed0[w & 3u] : packed1[w & 3u];
                float4 cv4 = laguna_e4m3_decode4(word);
                float4 xv4 = as_type<float4>(uint4(xrow[w]) << 16);
                float4 ax4 = metal::abs(xv4);
                uint4 b4 = (uint4(word) >> uint4(0u, 8u, 16u, 24u)) & 255u;
                uint4 mag4 = b4 & 127u;
                uint4 e4 = mag4 >> 3;
                float4 hsf =
                    as_type<float4>((metal::max(e4, uint4(1u)) + 116u) << 23);
                float4 hs4 =
                    metal::select(hsf, float4(186.0f), mag4 == 126u);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4; ++k) {
                    cg += xv4[k] * cv4[k];
                    dg += ax4[k] * hs4[k];
                }
            }
            c_acc += sd * cg;
            d_acc += sd * dg;
        }
        c_acc = simd_sum(c_acc);
        d_acc = simd_sum(d_acc);
        if (lane == 0) {
            coarse[row] = c_acc;
            delta[row] = d_acc * (1.0f + 61.0f * GAMMA);
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
    name: "laguna_lmhead_mxfp8_inline_coarse_pack16_v3",
    inputNames: ["x", "codes", "scales"],
    outputNames: ["coarse", "delta"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 16 +
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

/// Same MXFP8 coarse logits and quantization-error sum as the pack16 inline
/// kernel above, but the roundoff term uses the exact format-wide inequality
/// `m <= 30*d` certified at the top of this file: over all 256 codes the
/// decoded magnitude divided by its `laguna_hs8` half-cell width is `2m <= 14`
/// for the e == 0 denormals, `16 + 2m <= 30` for every e >= 1 code other than
/// the saturated 126 (attained at magnitude 15 and at 127), and 448/186 =
/// 2.409 for magnitude 126 -- so the maximum is exactly 30. Thus the retained
/// `d*(1+gamma) + 2*gamma*m` is bounded by `d*(1+61*gamma)`, and the two
/// scalars agree exactly in FP32: `(1+gamma) + 30*(2*gamma) == 1+61*gamma`.
///
/// As with the three-output twin, the inequality is per element and lifts to
/// the row sums under any threadgroup packing; the pack16 mapping leaves one
/// simdgroup per row and a 32-lane `simd_sum`, so `coarse` is bit-identical to
/// the retained kernel's and `delta` only widens. Widening `delta` can only
/// lower the threshold and admit MORE candidate rows, and an extra candidate
/// row is written with the stock-exact GEMV value instead of its certified-
/// below coarse value, so the inline-mask output contract is unchanged: every
/// slot still has exactly one owning lane, candidate slots are still
/// bit-identical to the stock full GEMV, and non-candidate slots still carry
/// `bfloat(coarse)` from the same unchanged FP32 bits.
private let lagunaLmHeadInlineCoarseRatioBoundKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_mxfp8_inline_coarse_ratio_bound_pack16_v1",
    inputNames: ["x", "codes", "scales"],
    outputNames: ["coarse", "delta"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 16 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint8_t* crow = codes + size_t(row) * 2048;
        const device uint8_t* srow = scales + size_t(row) * 64;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        for (uint gg = 0; gg < 2; ++gg) {
            uint g = 2 * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            const device uint4* cptr = (const device uint4*)(crow + g * 32);
            uint4 packed0 = cptr[0];
            uint4 packed1 = cptr[1];
            float cg = 0.0f;
            float dg = 0.0f;
            const device ushort4* xrow = (const device ushort4*)(x + g * 32);
            #pragma clang loop unroll(full)
            for (uint w = 0; w < 8; ++w) {
                uint word = (w < 4u) ? packed0[w & 3u] : packed1[w & 3u];
                float4 cv4 = laguna_e4m3_decode4(word);
                float4 xv4 = as_type<float4>(uint4(xrow[w]) << 16);
                float4 ax4 = metal::abs(xv4);
                uint4 b4 = (uint4(word) >> uint4(0u, 8u, 16u, 24u)) & 255u;
                uint4 mag4 = b4 & 127u;
                uint4 e4 = mag4 >> 3;
                float4 hsf =
                    as_type<float4>((metal::max(e4, uint4(1u)) + 116u) << 23);
                float4 hs4 =
                    metal::select(hsf, float4(186.0f), mag4 == 126u);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4; ++k) {
                    cg += xv4[k] * cv4[k];
                    dg += ax4[k] * hs4[k];
                }
            }
            c_acc += sd * cg;
            d_acc += sd * dg;
        }
        c_acc = simd_sum(c_acc);
        d_acc = simd_sum(d_acc);
        if (lane == 0) {
            coarse[row] = c_acc;
            delta[row] = d_acc * (1.0f + 61.0f * GAMMA);
        }
        """,
    header: lagunaLmHeadPruneHeader,
    ensureRowContiguous: true
)

/// The shipped default: the ratio-bound inline coarse kernel with `delta`
/// narrowed to BF16. Everything above the epilogue is a TEXTUAL copy of the
/// kernel directly above -- same lane partition, same j-order, same FP32
/// accumulation, same `sd_g` handling, same two `simd_sum`s -- so `coarse` and
/// the FP32 bound `d_acc * (1 + 61*GAMMA)` are bit-identical to that kernel's;
/// only the store narrows.
///
/// Round-toward-+infinity, exactly: `d_up` is a sum of products of `|x_j|`
/// (>= 0), `laguna_hs8` (> 0) and `laguna_e8m0_decode` group scales (a positive
/// power of two for every one of the 256 scale bytes), so it is non-negative
/// and its FP32 sign bit is clear. For such a value, masking the low 16
/// mantissa bits truncates toward zero, and adding one BF16 ulp whenever any
/// bit was actually dropped yields the SMALLEST BF16 that is >= the FP32 bound.
/// Metal offers no directed-rounding float->bfloat conversion, and the RNE
/// `bfloat(...)` cast can round DOWN, which would shrink the certified bound;
/// the bit form is the provably-conservative one and costs three integer ops.
///
/// Edge behavior is preserved rather than special-cased: a NaN bound keeps a
/// set mantissa through the mask-and-bump (a quiet NaN is BF16-exact and passes
/// through; a NaN whose low mantissa bits alone are set becomes 0x7F81xxxx,
/// still NaN), and a finite bound large enough to carry out of the exponent
/// saturates to +infinity, which is conservative (every row becomes a candidate
/// and the exact pass runs the stock GEMV for all of them). Neither is
/// reachable for a real hidden row.
///
/// Bytes: `delta` shrinks from 401,408 to 200,704 B per token, saving that on
/// the store plus the same again on each of the two consumers below.
private let lagunaLmHeadInlineCoarseRatioBoundDeltaBF16Kernel = MLXFast.metalKernel(
    name: "laguna_lmhead_mxfp8_inline_coarse_ratio_bound_delta_bf16_pack16_v1",
    inputNames: ["x", "codes", "scales"],
    outputNames: ["coarse", "delta"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 16 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint8_t* crow = codes + size_t(row) * 2048;
        const device uint8_t* srow = scales + size_t(row) * 64;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        for (uint gg = 0; gg < 2; ++gg) {
            uint g = 2 * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            const device uint4* cptr = (const device uint4*)(crow + g * 32);
            uint4 packed0 = cptr[0];
            uint4 packed1 = cptr[1];
            float cg = 0.0f;
            float dg = 0.0f;
            const device ushort4* xrow = (const device ushort4*)(x + g * 32);
            #pragma clang loop unroll(full)
            for (uint w = 0; w < 8; ++w) {
                uint word = (w < 4u) ? packed0[w & 3u] : packed1[w & 3u];
                float4 cv4 = laguna_e4m3_decode4(word);
                float4 xv4 = as_type<float4>(uint4(xrow[w]) << 16);
                float4 ax4 = metal::abs(xv4);
                uint4 b4 = (uint4(word) >> uint4(0u, 8u, 16u, 24u)) & 255u;
                uint4 mag4 = b4 & 127u;
                uint4 e4 = mag4 >> 3;
                float4 hsf =
                    as_type<float4>((metal::max(e4, uint4(1u)) + 116u) << 23);
                float4 hs4 =
                    metal::select(hsf, float4(186.0f), mag4 == 126u);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4; ++k) {
                    cg += xv4[k] * cv4[k];
                    dg += ax4[k] * hs4[k];
                }
            }
            c_acc += sd * cg;
            d_acc += sd * dg;
        }
        c_acc = simd_sum(c_acc);
        d_acc = simd_sum(d_acc);
        if (lane == 0) {
            coarse[row] = c_acc;
            // Same FP32 bound as the kernel above, then rounded UP to BF16.
            float d_up = d_acc * (1.0f + 61.0f * GAMMA);
            uint dbits = as_type<uint>(d_up);
            uint dtrunc = dbits & 0xFFFF0000u;
            if (dtrunc != dbits) {
                dtrunc += 0x00010000u;
            }
            delta[row] = as_type<bfloat>(ushort(dtrunc >> 16));
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

/// v4 coarse pass over the planar int6 copy (DARKBLOOM_LMHEAD_COARSE_V4=1).
/// Same launch geometry as the pack16 MXFP8 kernel (16 rows/threadgroup, one
/// simdgroup per row, lane = 2 consecutive 32-element groups), same fused
/// coarse+delta outputs, ~3/4 of the bytes.
///
/// Certificate (mirrors notes/68 with a flat half-cell):
///   codes are u = q + 32, q = round(w/sd) in [-31, 31], sd = 2^e chosen at
///   init so gmax/sd < 31.5 (no clamp; verified at init). sd is a power of
///   two, so w/sd is exact in float and |w_ij - sd*q_ij| <= sd/2 EXACTLY.
///   Hence d_i = sum_g (0.5*sd_g) * sum_{j in g} |x_j|
///             >= sum_j |x_j| * |w_ij - what_ij|
///   (0.5*sd is exact: both factors are powers of two), and with
///   m_i = sum_g sd_g * sum_{j in g} |x_j|*|q_ij|,
///   delta_i = d_i*(1+gamma) + 2*gamma*m_i covers the quantization error
///   plus both kernels' float rounding exactly as in the MXFP8 argument
///   (accumulation depth here is ~45 roundings/element-path, under the
///   depth <= 96 budget assumed by gamma = 2^-15).
/// Decode exactness: float4(uint4) of values <= 63 and the -32.0f offset are
/// exact; sd*q multiplies a power of two by a <=5-bit integer float: exact.
private let lagunaLmHeadInt6CoarseKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_int6_inline_coarse_v4",
    inputNames: ["x", "codes_lo", "codes_hi", "scales"],
    outputNames: ["coarse", "delta"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 16 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint8_t* lorow = codes_lo + size_t(row) * 1024;
        const device uint8_t* hirow = codes_hi + size_t(row) * 512;
        const device uint8_t* srow = scales + size_t(row) * 64;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        float m_acc = 0.0f;
        for (uint gg = 0; gg < 2; ++gg) {
            uint g = 2 * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            uint4 lo4 = ((const device uint4*)(lorow + g * 16))[0];
            uint2 hi2 = ((const device uint2*)(hirow + g * 8))[0];
            const device ushort4* xrow = (const device ushort4*)(x + g * 32);
            float cg = 0.0f;
            float ag = 0.0f;
            float mg = 0.0f;
            #pragma clang loop unroll(full)
            for (uint w = 0; w < 4; ++w) {
                // Word w: elements 8w..8w+7 of the group. Nibble plane byte
                // b holds elements 2b (low) / 2b+1 (high); 2-bit plane byte
                // b holds elements 4b..4b+3 at bits 0,2,4,6.
                uint lw = lo4[w];
                uint hw = ((w & 2u) ? hi2.y : hi2.x) >> ((w & 1u) * 16u);
                uint4 ne = (uint4(lw) >> uint4(0u, 8u, 16u, 24u)) & 15u;
                uint4 no = (uint4(lw) >> uint4(4u, 12u, 20u, 28u)) & 15u;
                uint4 he = (uint4(hw) >> uint4(0u, 4u, 8u, 12u)) & 3u;
                uint4 ho = (uint4(hw) >> uint4(2u, 6u, 10u, 14u)) & 3u;
                // Offset-binary decode: value = u - 32 in [-31, 31], exact.
                float4 ve = float4(ne | (he << 4u)) - 32.0f;
                float4 vo = float4(no | (ho << 4u)) - 32.0f;
                // bf16 -> f32 is exactly bits<<16 for every value class.
                float4 xa = as_type<float4>(uint4(xrow[2 * w]) << 16);
                float4 xb = as_type<float4>(uint4(xrow[2 * w + 1]) << 16);
                float4 xe = float4(xa.x, xa.z, xb.x, xb.z);
                float4 xo = float4(xa.y, xa.w, xb.y, xb.w);
                float4 axe = metal::abs(xe);
                float4 axo = metal::abs(xo);
                float4 ave = metal::abs(ve);
                float4 avo = metal::abs(vo);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4; ++k) {
                    cg += xe[k] * ve[k];
                    cg += xo[k] * vo[k];
                    ag += axe[k];
                    ag += axo[k];
                    mg += axe[k] * ave[k];
                    mg += axo[k] * avo[k];
                }
            }
            c_acc += sd * cg;
            d_acc += (0.5f * sd) * ag;
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

/// The v4 int6 coarse pass under the ratio bound
/// (`DARKBLOOM_LMHEAD_RATIO_BOUND`, default ON): same `coarse` and same `d`
/// term as the kernel directly above, but the roundoff term uses the exact
/// format-wide inequality `m <= 62*d`, so the `m_acc` chain and its SIMD
/// reduction are gone and the epilogue emits the closed form
/// `d*(1 + 125*gamma)`.
///
/// Certificate (the int6 twin of the MXFP8 `m <= 30*d` argument at the top of
/// this file; mirrors the notes/68 chain with the flat half-cell):
///   * Codes. The planar planes carry `u` in six bits and the kernel decodes
///     `q = u - 32`. `buildInt6Planes` picks `sd = 2^e` so that
///     `gmax/sd < 31.5` and then VERIFIES `max|q| <= 31` on the actual tensor,
///     declining to the MXFP8 copy if that ever fails. There is no clamp, so
///     the guard is what establishes the range: on this arm `u` is in [1, 63]
///     and `|q| <= 31` for every element the kernel can ever decode. (Were
///     `u = 0` reachable the ratio would be 64, not 62 -- the init guard is
///     load-bearing, which is why it is a hard fallback rather than a
///     warning.)
///   * Per element. The `m` term contributes `sd*|x_j|*|q_ij|` and the `d`
///     term contributes `(0.5*sd)*|x_j|` -- the SAME elements under the SAME
///     positive power-of-two group scale `sd_g`. Their ratio is exactly
///     `2*|q_ij| <= 62`, so the inequality lifts termwise to the row sums:
///     `m_i <= 62*d_i`.
///   * Substitution. `d_i*(1+gamma) + 2*gamma*m_i <= d_i*(1 + gamma +
///     124*gamma) = d_i*(1 + 125*gamma)`, and the two scalars agree exactly in
///     FP32: `(1+gamma) + 62*(2*gamma) == 1 + 125*gamma` bit-for-bit
///     (`125*2^-15` spans 16 significand bits, well inside FP32's 24).
///
/// Widening `delta` is always safe: candidacy is MONOTONE in delta -- a larger
/// delta lowers `L = max(coarse - delta)` and therefore the threshold, and
/// raises each row's `coarse + delta` upper bound -- so the exact pass's
/// candidate set can only GROW, and a newly admitted row is written with the
/// stock-exact GEMV value in place of its certified-below coarse value, which
/// cannot move the argmax.
///
/// `c_acc` and `d_acc` are bit-identical to the kernel above: the lane
/// partition, the `gg`/`w`/`k` loop order and unrolls, the plane decode, the
/// `sd_g` handling and the 32-lane `simd_sum` width are all textually
/// unchanged. The kernel text is exactly that kernel's with the `m` chain
/// deleted -- `m_acc`, `mg`, `ave`/`avo`, the two `mg` MADs per k-iteration
/// and the third `simd_sum` -- and the epilogue's store replaced; the tests
/// machine-check that derivation.
///
/// Op delta per row (2048 elements, 32 lanes): -2048 `metal::abs`, -2048
/// fmul, -2048 fadd, -64 fmul, -64 fadd (the `m_acc += sd*mg` folds), -1
/// `simd_sum`, -1 fmul, -1 fadd in the epilogue. No byte-traffic change.
private let lagunaLmHeadInt6CoarseRatioBoundKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_int6_inline_coarse_ratio_bound_v4",
    inputNames: ["x", "codes_lo", "codes_hi", "scales"],
    outputNames: ["coarse", "delta"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 16 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint8_t* lorow = codes_lo + size_t(row) * 1024;
        const device uint8_t* hirow = codes_hi + size_t(row) * 512;
        const device uint8_t* srow = scales + size_t(row) * 64;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        for (uint gg = 0; gg < 2; ++gg) {
            uint g = 2 * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            uint4 lo4 = ((const device uint4*)(lorow + g * 16))[0];
            uint2 hi2 = ((const device uint2*)(hirow + g * 8))[0];
            const device ushort4* xrow = (const device ushort4*)(x + g * 32);
            float cg = 0.0f;
            float ag = 0.0f;
            #pragma clang loop unroll(full)
            for (uint w = 0; w < 4; ++w) {
                // Word w: elements 8w..8w+7 of the group. Nibble plane byte
                // b holds elements 2b (low) / 2b+1 (high); 2-bit plane byte
                // b holds elements 4b..4b+3 at bits 0,2,4,6.
                uint lw = lo4[w];
                uint hw = ((w & 2u) ? hi2.y : hi2.x) >> ((w & 1u) * 16u);
                uint4 ne = (uint4(lw) >> uint4(0u, 8u, 16u, 24u)) & 15u;
                uint4 no = (uint4(lw) >> uint4(4u, 12u, 20u, 28u)) & 15u;
                uint4 he = (uint4(hw) >> uint4(0u, 4u, 8u, 12u)) & 3u;
                uint4 ho = (uint4(hw) >> uint4(2u, 6u, 10u, 14u)) & 3u;
                // Offset-binary decode: value = u - 32 in [-31, 31], exact.
                float4 ve = float4(ne | (he << 4u)) - 32.0f;
                float4 vo = float4(no | (ho << 4u)) - 32.0f;
                // bf16 -> f32 is exactly bits<<16 for every value class.
                float4 xa = as_type<float4>(uint4(xrow[2 * w]) << 16);
                float4 xb = as_type<float4>(uint4(xrow[2 * w + 1]) << 16);
                float4 xe = float4(xa.x, xa.z, xb.x, xb.z);
                float4 xo = float4(xa.y, xa.w, xb.y, xb.w);
                float4 axe = metal::abs(xe);
                float4 axo = metal::abs(xo);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4; ++k) {
                    cg += xe[k] * ve[k];
                    cg += xo[k] * vo[k];
                    ag += axe[k];
                    ag += axo[k];
                }
            }
            c_acc += sd * cg;
            d_acc += (0.5f * sd) * ag;
        }
        c_acc = simd_sum(c_acc);
        d_acc = simd_sum(d_acc);
        if (lane == 0) {
            coarse[row] = c_acc;
            delta[row] = d_acc * (1.0f + 125.0f * GAMMA);
        }
        """,
    header: lagunaLmHeadPruneHeader,
    ensureRowContiguous: true
)

/// The shipped default on the v4 arm: the int6 ratio-bound coarse kernel with
/// `delta` narrowed to BF16 (`DARKBLOOM_LMHEAD_DELTA_BF16`, default ON, nested
/// inside the ratio bound). Everything above the epilogue is a TEXTUAL copy of
/// the kernel directly above -- same lane partition, same loop order, same
/// plane decode, same FP32 accumulation, same `sd_g` handling, same two
/// `simd_sum`s -- so `coarse` and the FP32 bound `d_acc * (1 + 125*GAMMA)` are
/// bit-identical to that kernel's; only the store narrows.
///
/// Round-toward-+infinity, exactly, by the same bit surgery the MXFP8 twin
/// uses. The sign-clear argument holds here for the same reason: `d_acc` is a
/// sum of products of `|x_j|` (>= 0) and `(0.5 * laguna_e8m0_decode(sd_byte))`
/// (a positive power of two for every one of the 256 scale bytes, halved --
/// still positive), so `d_up` is non-negative and its FP32 sign bit is clear.
/// For such a value, masking the low 16 mantissa bits truncates toward zero,
/// and adding one BF16 ulp whenever any bit was actually dropped yields the
/// SMALLEST BF16 that is >= the FP32 bound. Metal offers no directed-rounding
/// float->bfloat conversion, and the RNE `bfloat(...)` cast can round DOWN,
/// which would shrink the certified bound.
///
/// Edge behavior is preserved rather than special-cased, exactly as on the
/// MXFP8 arm: a NaN bound keeps a set mantissa through the mask-and-bump, and
/// a finite bound large enough to carry out of the exponent saturates to
/// +infinity, which is conservative (every row becomes a candidate and the
/// exact pass runs the stock GEMV for all of them). Neither is reachable for a
/// real hidden row.
///
/// Downstream this reuses the SAME narrowed-delta consumers as the MXFP8 arm
/// -- `lagunaLmHeadLowerMaxStage1DeltaBF16Kernel` and
/// `lagunaLmHeadInlineExactDeltaBF16Kernel` -- so one arm selection still
/// implies one `delta` dtype for every reader. Bytes: `delta` shrinks from
/// 401,408 to 200,704 B per token on the store plus the same again on each of
/// those two consumers.
private let lagunaLmHeadInt6CoarseRatioBoundDeltaBF16Kernel = MLXFast.metalKernel(
    name: "laguna_lmhead_int6_inline_coarse_ratio_bound_delta_bf16_v4",
    inputNames: ["x", "codes_lo", "codes_hi", "scales"],
    outputNames: ["coarse", "delta"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 16 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint8_t* lorow = codes_lo + size_t(row) * 1024;
        const device uint8_t* hirow = codes_hi + size_t(row) * 512;
        const device uint8_t* srow = scales + size_t(row) * 64;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        for (uint gg = 0; gg < 2; ++gg) {
            uint g = 2 * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            uint4 lo4 = ((const device uint4*)(lorow + g * 16))[0];
            uint2 hi2 = ((const device uint2*)(hirow + g * 8))[0];
            const device ushort4* xrow = (const device ushort4*)(x + g * 32);
            float cg = 0.0f;
            float ag = 0.0f;
            #pragma clang loop unroll(full)
            for (uint w = 0; w < 4; ++w) {
                // Word w: elements 8w..8w+7 of the group. Nibble plane byte
                // b holds elements 2b (low) / 2b+1 (high); 2-bit plane byte
                // b holds elements 4b..4b+3 at bits 0,2,4,6.
                uint lw = lo4[w];
                uint hw = ((w & 2u) ? hi2.y : hi2.x) >> ((w & 1u) * 16u);
                uint4 ne = (uint4(lw) >> uint4(0u, 8u, 16u, 24u)) & 15u;
                uint4 no = (uint4(lw) >> uint4(4u, 12u, 20u, 28u)) & 15u;
                uint4 he = (uint4(hw) >> uint4(0u, 4u, 8u, 12u)) & 3u;
                uint4 ho = (uint4(hw) >> uint4(2u, 6u, 10u, 14u)) & 3u;
                // Offset-binary decode: value = u - 32 in [-31, 31], exact.
                float4 ve = float4(ne | (he << 4u)) - 32.0f;
                float4 vo = float4(no | (ho << 4u)) - 32.0f;
                // bf16 -> f32 is exactly bits<<16 for every value class.
                float4 xa = as_type<float4>(uint4(xrow[2 * w]) << 16);
                float4 xb = as_type<float4>(uint4(xrow[2 * w + 1]) << 16);
                float4 xe = float4(xa.x, xa.z, xb.x, xb.z);
                float4 xo = float4(xa.y, xa.w, xb.y, xb.w);
                float4 axe = metal::abs(xe);
                float4 axo = metal::abs(xo);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4; ++k) {
                    cg += xe[k] * ve[k];
                    cg += xo[k] * vo[k];
                    ag += axe[k];
                    ag += axo[k];
                }
            }
            c_acc += sd * cg;
            d_acc += (0.5f * sd) * ag;
        }
        c_acc = simd_sum(c_acc);
        d_acc = simd_sum(d_acc);
        if (lane == 0) {
            coarse[row] = c_acc;
            // Same FP32 bound as the kernel above, then rounded UP to BF16.
            float d_up = d_acc * (1.0f + 125.0f * GAMMA);
            uint dbits = as_type<uint>(d_up);
            uint dtrunc = dbits & 0xFFFF0000u;
            if (dtrunc != dbits) {
                dtrunc += 0x00010000u;
            }
            delta[row] = as_type<bfloat>(ushort(dtrunc >> 16));
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

/// Stage one over a BF16 `delta`. Textually the kernel above with the single
/// substitution `delta[base + i]` -> `float(delta[base + i])`: the launch shape,
/// partition, read order, NaN handling and reduction tree are unchanged, and
/// BF16 -> FP32 widening is exact, so this recomputes the same expression on
/// whatever value the buffer holds.
///
/// Because the stored delta was rounded UP, each `lower` is <= the FP32 arm's
/// and FP32 subtraction is monotone in its subtrahend, so `L` -- and therefore
/// the threshold -- can only move DOWN, admitting candidates and never dropping
/// one.
private let lagunaLmHeadLowerMaxStage1DeltaBF16Kernel = MLXFast.metalKernel(
    name: "laguna_lmhead_lower_max_stage1_delta_bf16_v1",
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
                float lower = coarse[base + i] - float(delta[base + i]);
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
/// that kernel plus its selector and `coarse_bf` input. The new tip's fused
/// lower-bound reduction is shared unchanged by both paths.
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

/// The shipped default: the inline-mask exact pass over a BF16 `delta`.
/// Textually the kernel above with the two `delta[r]` reads widened by
/// `float(...)` -- an exact conversion -- and nothing else changed: the same
/// fixed four-row block per simdgroup, the same stock `gemv_al_bfloat16`
/// replica, the same single owning lane per output slot, the same `bfloat`
/// casts.
///
/// The stored delta was rounded UP, so `coarse[r] + float(delta[r])` is >= the
/// FP32 arm's sum (FP32 addition is monotone) while `thr[0]` moved DOWN, and
/// every row the FP32 arm calls a candidate is still a candidate here. Newly
/// admitted rows are written with the stock-exact GEMV value instead of their
/// certified-below `bfloat(coarse[r])`, which cannot move the argmax.
/// `coarse` is untouched FP32, so skipped slots keep the exact bits the FP32
/// arm would store.
private let lagunaLmHeadInlineExactDeltaBF16Kernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_inline_mask_block_delta_bf16_v1",
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
                (r < VOCAB && coarse[r] + float(delta[r]) >= thr[0]);
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
                    assembled[r] = (coarse[r] + float(delta[r]) >= thr[0])
                        ? bfloat(result[tm])
                        : bfloat(coarse[r]);
                }
            }
        }
        """,
    ensureRowContiguous: true
)

/// Retained init-time MXFP8 coarse copy of lm_head plus the pruned final-row
/// forward. Built once (untimed init) by
/// `LagunaRuntimeModel.prepareFusedRuntimeWeights` when
/// `lagunaLmHeadPruneEnabled` (DARKBLOOM_LM_HEAD_PRUNE, default ON; set "0"
/// to disable); ~212 MB additional resident memory.
final class LagunaLmHeadPruner {
    /// MXFP8 copy for the shipped coarse pass. Not built when the v4 int6
    /// copy is active (one coarse copy is resident per arm).
    let codes: MLXArray?   // [100352, 2048] uint8 e4m3 elements
    let scales: MLXArray?  // [100352, 64] uint8 e8m0 group scales
    /// v4 planar int6 coarse copy (DARKBLOOM_LMHEAD_COARSE_V4=1): nibble
    /// plane [V, 1024], 2-bit plane [V, 512], power-of-two scale bytes
    /// [V, 64] (e8m0 byte semantics: 0 -> 2^-127, else 2^(b-127)).
    let int6CodesLo: MLXArray?
    let int6CodesHi: MLXArray?
    let int6Scales: MLXArray?

    /// The resident coarse-copy arrays of the ACTIVE arm, for the untimed
    /// init-time eval in `prepareFusedRuntimeWeights` (the shipped call named
    /// `codes`/`scales` directly, which are nil under v4).
    var residentArrays: [MLXArray] {
        if let lo = int6CodesLo, let hi = int6CodesHi, let s6 = int6Scales {
            return [lo, hi, s6]
        }
        return [codes, scales].compactMap { $0 }
    }

    init?(lmHeadWeight: MLXArray) {
        guard lmHeadWeight.shape == [lagunaLmHeadPruneVocab, lagunaLmHeadPruneHidden],
            lmHeadWeight.dtype == .bfloat16
        else {
            FileHandle.standardError.write(
                Data("mlxfast: lm_head prune: unrecognized lm_head shape/dtype; disabled\n".utf8))
            return nil
        }
        if lagunaLmHeadCoarseV4Enabled, lagunaLmHeadInlineMaskEnabled,
            let planes = LagunaLmHeadPruner.buildInt6Planes(lmHeadWeight)
        {
            // v4: the int6 copy replaces the MXFP8 copy entirely on this arm.
            self.int6CodesLo = planes.lo
            self.int6CodesHi = planes.hi
            self.int6Scales = planes.scales
            self.codes = nil
            self.scales = nil
            if lagunaTraceFusionEnabled {
                FileHandle.standardError.write(
                    Data("fusion active: lmhead-int6-coarse-v4\n".utf8))
            }
            return
        }
        if lagunaLmHeadCoarseV4Enabled, !lagunaLmHeadInlineMaskEnabled {
            FileHandle.standardError.write(
                Data(
                    "mlxfast: lm_head coarse v4 ignored (inline mask disabled)\n"
                        .utf8))
        }
        self.int6CodesLo = nil
        self.int6CodesHi = nil
        self.int6Scales = nil
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

    /// Builds the v4 planar int6 copy (untimed init).
    ///
    /// Scale rule: for each 32-element group with gmax = max|w|, sd = 2^e
    /// with e = floor_exp(gmax) - 4, bumped by one when the gmax mantissa is
    /// >= 1.96875, so that gmax/sd < 31.5 EXACTLY. Then q = round(w/sd) (the
    /// quotient is exact: sd is a power of two) satisfies |q| <= 31 and
    /// |w - sd*q| <= sd/2 exactly -- the flat half-cell the kernel's d-term
    /// uses. The no-overflow property is additionally verified here on the
    /// actual tensor; on violation the pruner falls back to the MXFP8 copy.
    private static func buildInt6Planes(
        _ lmHeadWeight: MLXArray
    ) -> (lo: MLXArray, hi: MLXArray, scales: MLXArray)? {
        let vocab = lagunaLmHeadPruneVocab
        let hidden = lagunaLmHeadPruneHidden
        let w = lmHeadWeight.asType(.float32).reshaped([vocab, hidden / 32, 32])
        let gmax = MLX.abs(w).max(axis: 2)  // [V, 64] float32, contiguous
        let gbits = gmax.view(dtype: .uint32)
        let biasedE = (gbits >> 23).asType(.int32)
        let mant = gbits & MLXArray(UInt32(0x007F_FFFF))
        // bump when mantissa >= 0.96875 * 2^23 (i.e. m >= 31.5/16).
        let bump = (mant .>= MLXArray(UInt32(0x7C_0000))).asType(.int32)
        let sdByte = clip(biasedE - 4 + bump, min: 0, max: 255)
        let sd = which(
            sdByte .== 0,
            MLXArray(Float(bitPattern: 0x0040_0000)),  // 2^-127, e8m0 semantics
            (sdByte.asType(.uint32) << 23).view(dtype: .float32))
        let q = (w / sd.expandedDimensions(axis: 2)).round()
        // Init-time certificate guard: no code may leave [-31, 31].
        let maxCode = MLX.abs(q).max().item(Float.self)
        guard maxCode <= 31.0 else {
            FileHandle.standardError.write(
                Data(
                    "mlxfast: lm_head coarse v4: int6 code overflow (\(maxCode)); using MXFP8\n"
                        .utf8))
            return nil
        }
        // Offset-binary u = q + 32 in [1, 63]; planar-pack 4+2 bits.
        let u = (q + 32).asType(.uint8).reshaped([vocab, hidden])
        let u16 = u.view(dtype: .uint16)  // [V, 1024]: elem 2b low byte
        let lo =
            ((u16 & MLXArray(UInt16(0x000F)))
            | ((u16 >> 4) & MLXArray(UInt16(0x00F0)))).asType(.uint8)
        let u32 = u.view(dtype: .uint32)  // [V, 512]: elem 4b low byte
        let hi =
            (((u32 >> 4) & MLXArray(UInt32(0x03)))
            | ((u32 >> 10) & MLXArray(UInt32(0x0C)))
            | ((u32 >> 16) & MLXArray(UInt32(0x30)))
            | ((u32 >> 22) & MLXArray(UInt32(0xC0)))).asType(.uint8)
        return (lo, hi, sdByte.asType(.uint8))
    }

    /// Pruned final-row lm_head: full [vocab] BF16 logits row, bit-identical to
    /// the stock pass in every candidate slot and certified-below elsewhere,
    /// so the downstream argmax emits the stock token.
    func logits(hidden: MLXArray, lmHeadWeight: MLXArray) -> MLXArray {
        precondition(hidden.dtype == .bfloat16 && hidden.size == lagunaLmHeadPruneHidden)
        let vocab = lagunaLmHeadPruneVocab
        let x = hidden.reshaped([lagunaLmHeadPruneHidden])
        let useCoarseV1 = lagunaLmHeadCoarseUseV1
        let coarseRowsPerThreadgroup = useCoarseV1 ? 8 : 16
        let coarseThreadsPerThreadgroup = coarseRowsPerThreadgroup * 32
        // The BF16 `delta` round trip is nested inside the ratio bound and, on
        // the MXFP8 arm, inside the inline mask and the pack16 coarse arm:
        // turning any of them off restores the FP32 `delta` kernels those arms
        // already shipped. The v4 int6 arm is inline-mask-only and pack16 by
        // construction -- its init declines under DARKBLOOM_LMHEAD_INLINE_MASK=0
        // and its kernels ignore the pack8 `v1` A/B selector -- so it carries
        // only the two nested selectors. Either way this single flag drives the
        // coarse store AND both downstream readers, so one arm selection always
        // means one `delta` dtype.
        let useInt6 =
            int6CodesLo != nil && int6CodesHi != nil && int6Scales != nil
        let useDeltaBF16 =
            (useInt6 || (lagunaLmHeadInlineMaskEnabled && !useCoarseV1))
            && lagunaLmHeadRatioBoundEnabled && lagunaLmHeadDeltaBF16Enabled

        let coarseOut: [MLXArray]
        if let lo = int6CodesLo, let hi = int6CodesHi, let s6 = int6Scales {
            // v4 int6 coarse pass: 16 rows per threadgroup like pack16.
            let coarseKernel: MLXFast.MLXFastKernel =
                if useDeltaBF16 {
                    lagunaLmHeadInt6CoarseRatioBoundDeltaBF16Kernel
                } else if lagunaLmHeadRatioBoundEnabled {
                    lagunaLmHeadInt6CoarseRatioBoundKernel
                } else {
                    lagunaLmHeadInt6CoarseKernel
                }
            coarseOut = coarseKernel(
                [x, lo, hi, s6],
                grid: (vocab / 16 * 512, 1, 1),
                threadGroup: (512, 1, 1),
                outputShapes: [[vocab], [vocab]],
                outputDTypes: [.float32, useDeltaBF16 ? .bfloat16 : .float32]
            )
        } else if lagunaLmHeadInlineMaskEnabled {
            let coarseKernel: MLXFast.MLXFastKernel =
                if useCoarseV1 {
                    lagunaLmHeadInlineCoarseKernelV1
                } else if useDeltaBF16 {
                    lagunaLmHeadInlineCoarseRatioBoundDeltaBF16Kernel
                } else if lagunaLmHeadRatioBoundEnabled {
                    lagunaLmHeadInlineCoarseRatioBoundKernel
                } else {
                    lagunaLmHeadInlineCoarseKernel
                }
            coarseOut = coarseKernel(
                [x, codes!, scales!],
                grid: (
                    vocab / coarseRowsPerThreadgroup * coarseThreadsPerThreadgroup,
                    1,
                    1
                ),
                threadGroup: (coarseThreadsPerThreadgroup, 1, 1),
                outputShapes: [[vocab], [vocab]],
                outputDTypes: [.float32, useDeltaBF16 ? .bfloat16 : .float32]
            )
        } else {
            // Kill-switch fallback: the new tip's original three-output
            // coarse kernels still materialize `coarse_bf` for the retained
            // selector/exact path.
            let coarseKernel: MLXFast.MLXFastKernel =
                if useCoarseV1 {
                    lagunaLmHeadCoarseKernelV1
                } else if lagunaLmHeadRatioBoundEnabled {
                    lagunaLmHeadCoarseRatioBoundKernel
                } else {
                    lagunaLmHeadCoarseKernel
                }
            coarseOut = coarseKernel(
                [x, codes!, scales!],
                grid: (
                    vocab / coarseRowsPerThreadgroup * coarseThreadsPerThreadgroup,
                    1,
                    1
                ),
                threadGroup: (coarseThreadsPerThreadgroup, 1, 1),
                outputShapes: [[vocab], [vocab], [vocab]],
                outputDTypes: [.float32, .float32, .bfloat16]
            )
        }
        let coarse = coarseOut[0]
        let delta = coarseOut[1]

        // Threshold on GPU: L = max(coarse - delta); thr = L - |L|/64.
        // The custom pair fuses the six-dispatch MLX expression into two
        // dispatches while reproducing MLX's exact two-pass reduction layout.
        let lowerMaxStage1Kernel =
            useDeltaBF16
            ? lagunaLmHeadLowerMaxStage1DeltaBF16Kernel
            : lagunaLmHeadLowerMaxStage1Kernel
        let lowerMaxPartials = lowerMaxStage1Kernel(
            [coarse, delta],
            grid: (224, 128, 1),
            threadGroup: (224, 1, 1),
            outputShapes: [[128]],
            outputDTypes: [.float32]
        )[0]
        let thr = lagunaLmHeadLowerMaxThresholdKernel(
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
            let exactKernel =
                useDeltaBF16
                ? lagunaLmHeadInlineExactDeltaBF16Kernel
                : lagunaLmHeadInlineExactKernel
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
}
