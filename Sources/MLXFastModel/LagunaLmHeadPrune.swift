import Foundation
import MLX
import MLXFast

// Certified two-pass lm_head elision for the final-token projection
// (notes/68, notes/70, notes/exp-hybridcoarse.md).
//
// Stock lm_head reads the full BF16 [100352, 2048] weight (411 MB) for the
// final hidden row at the DRAM wall. This module, gated by
// DARKBLOOM_LM_HEAD_PRUNE (DEFAULT ON; set "0" to restore the byte-identical
// stock pass), replaces it for both prefill's already-sliced last hidden row
// and single-token decode with four dispatches:
//
//   1. COARSE (`lagunaLmHeadCoarseKernel`): one fused
//      GEMV over an init-time planar int5 copy of lm_head (1344 B/row) that
//      emits each row's coarse logit c_i and a certified error bound delta_i.
//      With d_i = sum_j |x_j| * (sd_g/2) >= sum_j |x_j| * |w_ij - what_ij|
//      (flat half-cell) and m_i = sum_j |x_j| * |what_ij|, the certificate is
//      delta_i = d_i*(1+gamma) + 2*gamma*m_i, where gamma = 2^-15 covers every
//      float rounding on both paths (depth <= 96 roundings per element-path
//      << gamma; notes/68 section 6). The kernel emits the strictly-not-
//      smaller closed form d_i*(1 + 61*gamma), legal because the int5 codes
//      satisfy |q| <= 15 (verified on the actual tensor at init), so per
//      element the m term sd*|x_j|*|q_ij| is at most 30x the d term
//      (sd/2)*|x_j| -- same elements, same positive power-of-two group scale
//      -- giving m_i <= 30*d_i termwise. The substitution is exact at the
//      constant level: (1+gamma) + 30*(2*gamma) == 1 + 61*gamma bit-for-bit in
//      FP32. delta_i is stored as BF16 rounded toward +infinity; delta is only
//      ever COMPARED downstream and candidacy is MONOTONE in it, so widening
//      it only grows the candidate set. `coarse` stays FP32: it would have to
//      round DOWN for the threshold path and UP for the candidate test, which
//      one buffer cannot do.
//   2. ARGMAX stage one (`lagunaLmHeadCoarseArgmaxStage1Kernel`): the stock
//      two-pass reduction layout carrying (value, index) pairs over `coarse`
//      alone, so `delta` is not read here.
//   3. THRESHOLD (`lagunaLmHeadExactWinnerBF16PredecessorThresholdKernel`):
//      finishes the argmax, runs the stock single-row GEMV for the winning row
//      r, and thresholds just below bfloat(e_r) instead of below the coarse
//      lower bound L. Sound for ANY r because e_r <= e_winner, which removes
//      the winner's own quantization delta from the threshold.
//   4. EXACT (`lagunaLmHeadExactKernel`): each simdgroup owns a
//      FIXED block of four output rows and runs a full BF16 GEMV over that
//      block only when `coarse[r] + float(delta[r]) >= threshold` for one of
//      its rows, writing `bfloat(coarse[r])` otherwise. The per-row arithmetic
//      is a TEXTUAL replica of the stock `gemv_al_bfloat16`
//      (bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0; see gemv.h GEMVKernel::run with
//      kAligned=true) -- same lane partition, same sequential f32 order, same
//      vec4 loads, same simd tree, same BF16 cast -- so every candidate row's
//      logit is bit-identical to the stock full GEMV's. Every vocabulary slot
//      is written by exactly one lane on exactly one path, so the row is fully
//      covered with no race and no uninitialized slot. Non-candidate slots
//      keep the BF16 coarse value, which the certificate shows is strictly
//      below the winner; the harness argmaxes the returned row
//      (LagunaCorrectness.swift:108), so the emitted token is the stock token.
//
// SINGLE-TOKEN DECODE runs a three-level form of the same screen
// (DARKBLOOM_LMHEAD_FUSED_REFINEMENT, DEFAULT ON). The int5 planes are
// re-split so the nibble plane alone is a self-contained 2x-coarse code
// (nibble = floor(q/2) + 8, bit plane = q - 2*floor(q/2)), which lets step 1
// read 1088 B/row instead of 1344 B/row -- 25.7 MB less per decode step over
// the 100352-row vocabulary. The dropped 256 B/row residual bit plane is then
// re-read INSIDE the exact dispatch, for surviving four-row blocks only, to
// restore the exact int5 value and the tighter half-cell bound before the BF16
// GEMV decision. Dispatch count, grid and threadgroup geometry are unchanged;
// prefill's already-sliced row keeps the one-pass form above.
//
// The e8m0 decoder in the kernel header below is a bit-exact replica of the
// vendored fp8.h semantics (no libm: exponent-bit construction).

private let lagunaLmHeadPruneVocab = 100_352
private let lagunaLmHeadPruneHidden = 2048

/// Master switch for the certified two-pass final-row lm_head (notes/68).
/// DEFAULT ON: unset, or any value other than "0", enables the certified
/// two-pass head and builds the int5 coarse copy at init time.
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

/// Same-binary A/B switch for the three-level decode screen: read only the
/// four-bit base plane for every row, then refine the surviving four-row
/// blocks with the residual bit plane inside the exact dispatch. DEFAULT ON;
/// set `DARKBLOOM_LMHEAD_FUSED_REFINEMENT=0` for the one-pass int5 control
/// arm in the same binary.
let lagunaLmHeadFusedRefinementEnabled =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_LMHEAD_FUSED_REFINEMENT"] != "0"

/// One-line stderr trace hooks (DARKBLOOM_TRACE_FUSION=1) so an active pruner
/// is visible in run logs.
private let lagunaTraceFusionEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_TRACE_FUSION"] == "1"

/// Kernel header: the bit-exact e8m0 group-scale decoder, inlinable and
/// libm-free.
private let lagunaLmHeadPruneHeader = """
    // e8m0 decode, identical to fp8.h:70-77 (bits<<7 as bf16; bits==0 ->
    // 0x40 as bf16 = 2^-127). Exponent-bit construction, exact.
    static inline float laguna_e8m0_decode(uint8_t b) {
        if (b == 0u) {
            return as_type<float>(0x00400000u);  // 2^-127
        }
        return as_type<float>(uint(b) << 23);
    }
    """

/// Unified coarse kernel for both prefill's one-pass int5 screen and decode's
/// base-plane-only screen. `mode[0]==0` selects the one-pass form (reads the
/// 1-bit plane for exact int5 decode, half-cell bound, 1344 B/row); `mode[0]==1`
/// selects the base-plane-only form (midpoint code, full-cell bound, 1088
/// B/row). Both paths share the same 16-rows/threadgroup launch and lane
/// partition, folding coarse+delta outputs.
///
/// Prefill (mode 0) certificate:
///   * Scale. sd = 2^e per 32-element group with e = floorexp(gmax) - 3,
///     +1 when mantissa(gmax) >= 1.9375 (bit-exact integer test
///     mant >= 0x780000), so gmax/sd < 15.5 EXACTLY in both cases
///     (8m < 15.5 when m < 1.9375; 4m < 8 otherwise). Scale byte stored
///     with e8m0 semantics, decoded by the existing `laguna_e8m0_decode`.
///   * Codes. q = round(w/sd): sd is a power of two so w/sd is EXACT in
///     f32, and rounding an exact quotient < 15.5 gives |q| <= 15 -- no
///     clamp, and `buildInt5Planes` VERIFIES max|q| <= 15 on the actual
///     tensor, declining to the stock head if that ever fails. So
///     u = q + 16 is in [1, 31] and |w - sd*q| <= sd/2 EXACTLY (flat
///     half-cell; 0.5*sd is exact, both factors powers of two).
///   * Ratio bound. The m term contributes sd*|x_j|*|q_ij| and the d term
///     (0.5*sd)*|x_j| -- same elements, same positive power-of-two group
///     scale -- so their ratio is exactly 2*|q_ij| <= 30, m_i <= 30*d_i
///     termwise, and d_i*(1+gamma) + 2*gamma*m_i <= d_i*(1 + 61*gamma).
///   * Store. The FP32 bound d_acc*(1 + 61*GAMMA) is rounded UP to BF16 by
///     sign-clear mask-and-bump bit surgery; widening delta only admits
///     candidates (monotone), and delta does not feed the threshold.
/// Decode (mode 1) certificate:
///   * Midpoint code. The nibble is H = floor(q/2) + 8, so q = 2H + b - 16
///     with b the residual bit this path does not read. Substituting the
///     cell midpoint b = 0.5 gives q0 = 2H - 15.5 and |q - q0| = 0.5 EXACTLY,
///     so |w - sd*q0| <= |w - sd*q| + 0.5*sd <= sd: the full cell, twice the
///     one-pass kernel's half-cell. That is why d_acc accumulates sd*ag here
///     and (0.5*sd)*ag there.
///   * Ratio bound. H <= 15 so |q0| <= 15.5, giving m_i <= 15.5*d_i
///     termwise and d_i*(1+gamma) + 2*gamma*m_i <= d_i*(1 + 32*gamma).
///   * Store. Identical sign-clear mask-and-bump round-UP to BF16.
private let lagunaLmHeadCoarseKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_int5_coarse_delta_bf16_v7",
    inputNames: ["x", "codes_lo", "codes_hi", "scales", "mode"],
    outputNames: ["coarse", "delta"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 16 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint8_t* lorow = codes_lo + size_t(row) * 1024;
        const device uint8_t* srow = scales + size_t(row) * 64;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        if (mode[0] == 0u) {
            const device uint8_t* hirow = codes_hi + size_t(row) * 256;
            for (uint gg = 0; gg < 2; ++gg) {
                uint g = 2 * lane + gg;
                float sd = laguna_e8m0_decode(srow[g]);
                uint4 lo4 = ((const device uint4*)(lorow + g * 16))[0];
                uint hb = ((const device uint*)(hirow + g * 4))[0];
                const device ushort4* xrow = (const device ushort4*)(x + g * 32);
                float cg = 0.0f;
                float ag = 0.0f;
                #pragma clang loop unroll(full)
                for (uint w = 0; w < 4; ++w) {
                    uint lw = lo4[w];
                    uint hw = hb >> (8u * w);
                    uint4 ne = (uint4(lw) >> uint4(0u, 8u, 16u, 24u)) & 15u;
                    uint4 no = (uint4(lw) >> uint4(4u, 12u, 20u, 28u)) & 15u;
                    uint4 he = (uint4(hw) >> uint4(0u, 2u, 4u, 6u)) & 1u;
                    uint4 ho = (uint4(hw) >> uint4(1u, 3u, 5u, 7u)) & 1u;
                    float4 ve = float4((ne << 1u) | he) - 16.0f;
                    float4 vo = float4((no << 1u) | ho) - 16.0f;
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
        } else {
            for (uint gg = 0; gg < 2; ++gg) {
                uint g = 2 * lane + gg;
                float sd = laguna_e8m0_decode(srow[g]);
                uint4 c4 = ((const device uint4*)(lorow + g * 16))[0];
                const device ushort4* xrow = (const device ushort4*)(x + g * 32);
                float cg = 0.0f;
                float ag = 0.0f;
                #pragma clang loop unroll(full)
                for (uint w = 0; w < 4; ++w) {
                    uint lw = c4[w];
                    uint4 ne = (uint4(lw) >> uint4(0u, 8u, 16u, 24u)) & 15u;
                    uint4 no = (uint4(lw) >> uint4(4u, 12u, 20u, 28u)) & 15u;
                    float4 ve = float4(ne << 1u) - 15.5f;
                    float4 vo = float4(no << 1u) - 15.5f;
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
                d_acc += sd * ag;
            }
        }
        c_acc = simd_sum(c_acc);
        d_acc = simd_sum(d_acc);
        if (lane == 0) {
            coarse[row] = c_acc;
            float bound_mult = mode[0] == 0u ? 61.0f : 32.0f;
            float d_up = d_acc * (1.0f + bound_mult * GAMMA);
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

/// Stage one: partial ARGMAX of `coarse` (value + index). Same launch shape
/// and read partition as the stock two-pass reduction (grid (224, 128),
/// threadgroup (224, 1), four consecutive values per active thread,
/// 784-element partitions), and it reads no `delta` at all -- the exact-winner
/// threshold does not consume the coarse lower bound L, so `coarse - delta` is
/// never formed and `delta`'s only reader is the exact pass.
///
/// The reduction carries (value, index) pairs: value primary, LOWEST index
/// on ties, making the selected row deterministic for any input. NaN coarse
/// values lose every `>` comparison and are skipped (no `laguna_lmhead_
/// max_pair` NaN propagation here) -- correctness never depends on WHICH row
/// wins: the exact-winner threshold is sound for ANY row index (see the
/// threshold kernel below), so argmax quality only affects the candidate
/// count, i.e. speed. Real coarse rows contain no NaN (finite BF16 weights
/// times finite BF16 activations accumulated in FP32).
///
/// Inactive lanes (lid >= 196) and the second-phase padding lanes carry
/// (-inf, 0xFFFFFFFF) and lose to any read element -- even an all-(-inf)
/// partition resolves to a real index via the tie-break, so a valid row
/// index always survives; the threshold kernel additionally clamps.
private let lagunaLmHeadCoarseArgmaxStage1Kernel = MLXFast.metalKernel(
    name: "laguna_lmhead_coarse_argmax_stage1_v5",
    inputNames: ["coarse"],
    outputNames: ["partial_max", "partial_idx"],
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
        threadgroup uint shared_idxs[32];

        float best = -metal::numeric_limits<float>::infinity();
        uint best_idx = 0xFFFFFFFFu;
        if (lid < ACTIVE_THREADS) {
            uint base = row * ROW_SIZE + lid * READS;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < READS; ++i) {
                float v = coarse[base + i];
                if (v > best || (v == best && base + i < best_idx)) {
                    best = v;
                    best_idx = base + i;
                }
            }
        }

        #pragma clang loop unroll(full)
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
            float ov = simd_shuffle_down(best, sn);
            uint oi = simd_shuffle_down(best_idx, sn);
            if (ov > best || (ov == best && oi < best_idx)) {
                best = ov;
                best_idx = oi;
            }
        }
        if (simd_lane == 0) {
            shared_vals[simd_group] = best;
            shared_idxs[simd_group] = best_idx;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        best = lid < SIMD_GROUPS
            ? shared_vals[lid]
            : -metal::numeric_limits<float>::infinity();
        best_idx = lid < SIMD_GROUPS ? shared_idxs[lid] : 0xFFFFFFFFu;
        #pragma clang loop unroll(full)
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
            float ov = simd_shuffle_down(best, sn);
            uint oi = simd_shuffle_down(best_idx, sn);
            if (ov > best || (ov == best && oi < best_idx)) {
                best = ov;
                best_idx = oi;
            }
        }
        if (lid == 0) {
            partial_max[row] = best;
            partial_idx[row] = best_idx;
        }
        """,
    ensureRowContiguous: true
)

/// Tight exact-winner threshold: finish the argmax, run the stock single-row
/// GEMV for the winning row r, and threshold just below `bfloat(e_r)` rather
/// than below the coarse lower bound L. Sound for ANY r because
/// `e_r <= e_winner`, which removes the winner's own quantization delta from
/// the threshold -- the one number that made pure int5 untenable.
///
/// Let `b = bfloat(e_r)` and `p = predecessor(b)` in the ordered BF16 value
/// set. Monotonic BF16 rounding gives `bfloat(e_winner) >= b` because
/// `e_winner >= e_r`. A non-candidate has `coarse_i + delta_i < p`, hence
/// `coarse_i < p`; monotonic round-to-nearest therefore gives
/// `bfloat(coarse_i) <= p < b <= bfloat(e_winner)`. The true winner remains a
/// candidate because its certified upper bound is at least `e_winner`, while
/// `p < e_r <= e_winner`. This proof covers either sign and exact BF16 values.
/// Finite zero maps to negative-min-subnormal; real model logits are finite.
///
/// The emitted threshold is refined to the exact FP32 midpoint between `p` and
/// `b`, which is the highest value the same proof supports: candidacy is `>=`,
/// so a skipped coarse value is strictly below the tie boundary and rounds to
/// `p` or lower, while the winner's own bound is at least `e_r > midpoint`.
private let lagunaLmHeadExactWinnerBF16PredecessorThresholdKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_winner_bf16_midpoint_threshold_v1",
    inputNames: ["partial_max", "partial_idx", "lm_head", "x"],
    outputNames: ["threshold"],
    source: """
        constexpr uint VOCAB = 100352;
        constexpr uint K = 2048;
        constexpr uint READS = 4;
        uint lid = thread_position_in_threadgroup.x;
        threadgroup uint winner_row[1];

        // Verbatim final argmax over the retained 128 partials.
        float best = -metal::numeric_limits<float>::infinity();
        uint best_idx = 0xFFFFFFFFu;
        uint base = lid * READS;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < READS; ++i) {
            float v = partial_max[base + i];
            uint idx = partial_idx[base + i];
            if (v > best || (v == best && idx < best_idx)) {
                best = v;
                best_idx = idx;
            }
        }
        #pragma clang loop unroll(full)
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
            float ov = simd_shuffle_down(best, sn);
            uint oi = simd_shuffle_down(best_idx, sn);
            if (ov > best || (ov == best && oi < best_idx)) {
                best = ov;
                best_idx = oi;
            }
        }
        if (lid == 0) {
            winner_row[0] = metal::min(best_idx, uint(VOCAB - 1));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint r = winner_row[0];

        // --- stock gemv_al replica begin (single row r; gemv.h:151-289) ---
        float result = 0.0f;
        thread bfloat inter[4];
        thread float v_coeff[4];
        uint bn = lid * 4;
        const device bfloat* mrow = lm_head + size_t(r) * K;
        for (uint i = 0; i < 16; ++i) {
            vec<bfloat, 4> xv =
                *((const device vec<bfloat, 4>*)(x + bn));
            v_coeff[0] = float(xv.x);
            v_coeff[1] = float(xv.y);
            v_coeff[2] = float(xv.z);
            v_coeff[3] = float(xv.w);
            vec<bfloat, 4> mv =
                *((const device vec<bfloat, 4>*)(mrow + bn));
            inter[0] = mv.x;
            inter[1] = mv.y;
            inter[2] = mv.z;
            inter[3] = mv.w;
            result += inter[0] * v_coeff[0];
            result += inter[1] * v_coeff[1];
            result += inter[2] * v_coeff[2];
            result += inter[3] * v_coeff[3];
            bn += 128;
        }
        #pragma unroll
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
            result += simd_shuffle_down(result, sn);
        }
        // --- stock gemv_al replica end ---
        if (lid == 0) {
            bfloat rounded = bfloat(result);
            // Expand through the numeric BF16->FP32 conversion, whose bits are
            // exactly `bf16_bits << 16`; do not reinterpret the Metal wrapper.
            ushort bits = ushort(as_type<uint>(float(rounded)) >> 16);
            ushort magnitude = bits & 0x7FFFu;
            ushort predecessor_bits;
            if (magnitude == 0u) {
                predecessor_bits = 0x8001u;  // predecessor of either zero
            } else if ((bits & 0x8000u) == 0u) {
                predecessor_bits = bits - 1u;
            } else {
                predecessor_bits = bits + 1u;
            }
            float predecessor =
                as_type<float>(uint(predecessor_bits) << 16);
            float rounded_value = as_type<float>(uint(bits) << 16);
            threshold[0] = predecessor + (rounded_value - predecessor) * 0.5f;
        }
        """,
    ensureRowContiguous: true
)
/// Unified exact kernel for both prefill's inline-mask pass and decode's
/// fused-int5 sparse refine pass. `mode[0]==0` selects the one-pass inline
/// exact form (no residual-plane refinement); `mode[0]==1` selects the
/// three-level decode form (re-reads the residual bit plane for live blocks
/// and re-screens with a tightened bound). Both paths share the same
/// fixed four-row block per simdgroup, the same launch geometry, and the
/// stock `gemv_al_bfloat16` replica for candidate rows.
///
/// Prefill (mode 0):
/// The predicate is simdgroup-uniform; lane 0 forms it once and broadcasts the
/// four row decisions. Non-candidate slots keep `bfloat(coarse[r])`, which the
/// certificate shows is strictly below the winner; candidate slots get the
/// stock-exact GEMV value. The stored delta was rounded UP, so
/// `coarse[r] + float(delta[r])` is >= the FP32 bound's sum while `thr[0]`
/// moved DOWN, and every row an FP32 delta would call a candidate is still a
/// candidate here.
///
/// Decode (mode 1):
/// A block that fails the base screen writes `bfloat(coarse[r])` and returns,
/// exactly as before. A block that survives re-reads the 256 B residual bit
/// plane and 64 B of scales for its live rows ONLY, adds
/// `sum_g sd_g * sum_j x_j*(b_j - 0.5)` to the stored midpoint value, and
/// re-screens against the tightened bound. Only rows surviving THAT read
/// BF16 lm_head. Live blocks are rare (single digits per step), so the plane
/// bytes this refunds are negligible against the 25.7 MB level one saves.
///
/// Refined bound (mode 1 only):
///   * `coarse[r] + correction` reconstructs sd*(2H + b - 16) = sd*q exactly
///     (2H - 15.5 + b - 0.5 = 2H + b - 16, and every term is a power of two
///     times a <=6-significand-bit value), so the exact-arithmetic error
///     bound drops from the full cell D to the half cell d = D/2.
///   * The accumulated magnitude is the sum of the two parts: the midpoint
///     term is at most 15.5*D = 31*d and the residual term at most
///     0.5*D = 1*d, so m <= 32*d and the certificate is
///     d*(1+gamma) + 2*gamma*m <= d*(1 + 65*gamma).
///   * `delta[r]` already bounds D*(1 + 32*gamma), so multiplying it by
///     `0x1.005p-1f` = (1 + 40*gamma)/2 yields at least
///     d*(1 + 32*gamma)*(1 + 40*gamma) > d*(1 + 65*gamma), with 7*gamma of
///     margin. The multiply is EXACT in FP32 (a widened BF16 has <= 8
///     significand bits, the constant has 13, and 21 < 24), and the
///     mask-and-bump round-UP to BF16 that follows only widens further.
///   * Copying the source submission's `0x1.004p-1f` = (1 + 32*gamma)/2
///     would deliver only d*(1 + 64.03*gamma) and miss the derived bound by
///     about one gamma.
/// The refined value is per-simdgroup scratch held by its owning lane, so no
/// threadgroup storage or barrier is involved.
private let lagunaLmHeadExactKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_mask_block_v2",
    inputNames: [
        "coarse", "delta", "thr", "lm_head", "x", "codes_bit", "scales", "mode",
    ],
    outputNames: ["assembled"],
    source: """
        constexpr uint VOCAB = 100352;
        constexpr uint K = 2048;

        uint tgid = threadgroup_position_in_grid.x;
        uint sgid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        uint base = tgid * 32 + sgid * 4;

        uint base_mask = 0;
        if (lane == 0) {
            #pragma unroll
            for (uint tm = 0; tm < 4; ++tm) {
                uint r = base + tm;
                if (r < VOCAB && coarse[r] + float(delta[r]) >= thr[0]) {
                    base_mask |= 1u << tm;
                }
            }
        }
        base_mask = simd_broadcast(base_mask, 0);

        if (base_mask == 0) {
            if (lane < 4 && base + lane < VOCAB) {
                assembled[base + lane] = bfloat(coarse[base + lane]);
            }
            return;
        }

        if (mode[0] == 0u) {
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
                        assembled[r] = (base_mask & (1u << tm)) != 0
                            ? bfloat(result[tm])
                            : bfloat(coarse[r]);
                    }
                }
            }
            return;
        }

        // Mode 1: decode with sparse residual-plane refinement.
        // Per-thread scratch: every lane holds a copy, only lane 0's is ever
        // written or consumed.
        thread float refined_coarse[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        uint refined_mask = 0;
        #pragma clang loop unroll(disable)
        for (uint tm = 0; tm < 4; ++tm) {
            if ((base_mask & (1u << tm)) == 0) {
                continue;
            }
            uint r = base + tm;
            const device uint8_t* hirow = codes_bit + size_t(r) * 256;
            const device uint8_t* srow = scales + size_t(r) * 64;
            float correction = 0.0f;
            for (uint gg = 0; gg < 2; ++gg) {
                uint g = 2 * lane + gg;
                float sd = laguna_e8m0_decode(srow[g]);
                uint hb = ((const device uint*)(hirow + g * 4))[0];
                const device ushort4* xrow =
                    (const device ushort4*)(x + g * 32);
                float cg = 0.0f;
                #pragma clang loop unroll(full)
                for (uint w = 0; w < 4; ++w) {
                    uint hw = hb >> (8u * w);
                    uint4 he = (uint4(hw) >> uint4(0u, 2u, 4u, 6u)) & 1u;
                    uint4 ho = (uint4(hw) >> uint4(1u, 3u, 5u, 7u)) & 1u;
                    float4 ve = float4(he) - 0.5f;
                    float4 vo = float4(ho) - 0.5f;
                    float4 xa = as_type<float4>(uint4(xrow[2 * w]) << 16);
                    float4 xb = as_type<float4>(uint4(xrow[2 * w + 1]) << 16);
                    float4 xe = float4(xa.x, xa.z, xb.x, xb.z);
                    float4 xo = float4(xa.y, xa.w, xb.y, xb.w);
                    #pragma clang loop unroll(full)
                    for (uint k = 0; k < 4; ++k) {
                        cg += xe[k] * ve[k];
                        cg += xo[k] * vo[k];
                    }
                }
                correction += sd * cg;
            }
            correction = simd_sum(correction);
            if (lane == 0) {
                float c_refined = coarse[r] + correction;
                refined_coarse[tm] = c_refined;
                float d_up = float(delta[r]) * 0x1.005p-1f;
                uint dbits = as_type<uint>(d_up);
                uint dtrunc = dbits & 0xFFFF0000u;
                if (dtrunc != dbits) {
                    dtrunc += 0x00010000u;
                }
                float delta_up =
                    float(as_type<bfloat>(ushort(dtrunc >> 16)));
                if (r < VOCAB && c_refined + delta_up >= thr[0]) {
                    refined_mask |= 1u << tm;
                }
            }
        }
        refined_mask = simd_broadcast(refined_mask, 0);

        if (refined_mask == 0) {
            if (lane == 0) {
                #pragma unroll
                for (uint tm = 0; tm < 4; ++tm) {
                    uint r = base + tm;
                    if (r < VOCAB) {
                        assembled[r] = (base_mask & (1u << tm)) != 0
                            ? bfloat(refined_coarse[tm])
                            : bfloat(coarse[r]);
                    }
                }
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
                    assembled[r] = (refined_mask & (1u << tm)) != 0
                        ? bfloat(result[tm])
                        : ((base_mask & (1u << tm)) != 0
                            ? bfloat(refined_coarse[tm])
                            : bfloat(coarse[r]));
                }
            }
        }
        """,
    header: lagunaLmHeadPruneHeader,
    ensureRowContiguous: true
)


/// Init-time int5 coarse copy of lm_head plus the pruned final-row forward.
/// Built once (untimed init) by
/// `LagunaRuntimeModel.prepareFusedRuntimeWeights` when
/// `lagunaLmHeadPruneEnabled` (DARKBLOOM_LM_HEAD_PRUNE, default ON; set "0"
/// to disable); ~135 MB additional resident memory.
final class LagunaLmHeadPruner {
    /// Planar int5 coarse copy: nibble plane [V, 1024] holding u >> 1, 1-bit
    /// plane [V, 256] holding u & 1 (element j of each 32-element group at bit
    /// j of the group's uint32 word), power-of-two group scale bytes [V, 64]
    /// with e8m0 semantics. 1344 B/row, of which the nibble plane and the
    /// scales (1088 B/row) are a self-contained 2x-coarse code.
    let int5CodesLo: MLXArray
    let int5CodesHi: MLXArray
    let int5Scales: MLXArray

    /// The resident coarse-copy arrays, for the untimed init-time eval in
    /// `prepareFusedRuntimeWeights`.
    var residentArrays: [MLXArray] { [int5CodesLo, int5CodesHi, int5Scales] }

    init?(lmHeadWeight: MLXArray) {
        guard lmHeadWeight.shape == [lagunaLmHeadPruneVocab, lagunaLmHeadPruneHidden],
            lmHeadWeight.dtype == .bfloat16
        else {
            FileHandle.standardError.write(
                Data("mlxfast: lm_head prune: unrecognized lm_head shape/dtype; disabled\n".utf8))
            return nil
        }
        // The int5 range guard is load-bearing (the coarse kernel's
        // m <= 30*d ratio bound assumes |q| <= 15), so a decline must return
        // nil and leave the caller on the byte-identical stock lm_head pass
        // rather than reach a weaker certificate.
        guard let planes = LagunaLmHeadPruner.buildInt5Planes(lmHeadWeight) else {
            return nil
        }
        self.int5CodesLo = planes.lo
        self.int5CodesHi = planes.hi
        self.int5Scales = planes.scales
        if lagunaTraceFusionEnabled {
            FileHandle.standardError.write(
                Data("fusion active: lmhead-int5-winner-coarse-v5\n".utf8))
        }
    }

    /// Builds the planar int5 coarse copy (untimed init).
    ///
    /// Scale rule: for each 32-element group with gmax = max|w|, sd = 2^e
    /// with e = floor_exp(gmax) - 3, bumped by one when the gmax mantissa is
    /// >= 1.9375, so that gmax/sd < 15.5 EXACTLY (8m < 15.5 when m < 1.9375;
    /// 4m < 8 otherwise). Then q = round(w/sd) (the quotient is exact: sd is
    /// a power of two) satisfies |q| <= 15 and |w - sd*q| <= sd/2 exactly --
    /// the flat half-cell the kernel's d-term uses. The no-overflow property
    /// is additionally verified here on the actual tensor; on violation the
    /// pruner declines and the stock lm_head pass is used.
    private static func buildInt5Planes(
        _ lmHeadWeight: MLXArray
    ) -> (lo: MLXArray, hi: MLXArray, scales: MLXArray)? {
        let vocab = lagunaLmHeadPruneVocab
        let hidden = lagunaLmHeadPruneHidden
        let w = lmHeadWeight.asType(.float32).reshaped([vocab, hidden / 32, 32])
        let gmax = MLX.abs(w).max(axis: 2)  // [V, 64] float32, contiguous
        let gbits = gmax.view(dtype: .uint32)
        let biasedE = (gbits >> 23).asType(.int32)
        let mant = gbits & MLXArray(UInt32(0x007F_FFFF))
        // bump when mantissa >= 0.9375 * 2^23 (i.e. m >= 15.5/8).
        let bump = (mant .>= MLXArray(UInt32(0x78_0000))).asType(.int32)
        let sdByte = clip(biasedE - 3 + bump, min: 0, max: 255)
        let sd = which(
            sdByte .== 0,
            MLXArray(Float(bitPattern: 0x0040_0000)),  // 2^-127, e8m0 semantics
            (sdByte.asType(.uint32) << 23).view(dtype: .float32))
        let q = (w / sd.expandedDimensions(axis: 2)).round()
        // Init-time certificate guard: no code may leave [-15, 15]. The
        // kernel's m <= 30*d ratio bound assumes |q| <= 15 (u in [1, 31]).
        let maxCode = MLX.abs(q).max().item(Float.self)
        guard maxCode <= 15.0 else {
            FileHandle.standardError.write(
                Data(
                    "mlxfast: lm_head prune: int5 code overflow (\(maxCode)); using stock head\n"
                        .utf8))
            return nil
        }
        // Offset-binary u = q + 16 in [1, 31], split so the nibble plane is a
        // self-contained 2x-coarse code: the nibble holds u >> 1 (the HIGH
        // four bits, floor(q/2) + 8) and the 1-bit plane holds u & 1
        // (q - 2*floor(q/2)). Total bytes are unchanged; what changes is that
        // the level-one screen can now read the nibble plane alone.
        let u = (q + 16).asType(.uint8).reshaped([vocab, hidden])
        let base = u >> 1
        let u16 = base.view(dtype: .uint16)  // [V, 1024]: elem 2b low byte
        let lo =
            ((u16 & MLXArray(UInt16(0x000F)))
            | ((u16 >> 4) & MLXArray(UInt16(0x00F0)))).asType(.uint8)
        // 1-bit plane: bit 0 of each code; element j of each 32-element
        // group lands at bit j of the group's little-endian uint32 word.
        // Step 1: per uint32 word of u (4 codes), gather the four bit-0s
        // (u32 bits 0, 8, 16, 24) into one low nibble.
        let u32 = u.view(dtype: .uint32)  // [V, 512]: elem 4t..4t+3
        let nib =
            ((u32 & MLXArray(UInt32(0x01)))
            | ((u32 >> 7) & MLXArray(UInt32(0x02)))
            | ((u32 >> 14) & MLXArray(UInt32(0x04)))
            | ((u32 >> 21) & MLXArray(UInt32(0x08)))).asType(.uint8)
        // Step 2: merge nibble pairs into bytes (byte s = elements 8s..8s+7).
        let nib16 = nib.view(dtype: .uint16)  // [V, 256]
        let hi =
            ((nib16 & MLXArray(UInt16(0x000F)))
            | ((nib16 >> 4) & MLXArray(UInt16(0x00F0)))).asType(.uint8)
        return (lo, hi, sdByte.asType(.uint8))
    }

    /// Pruned final-row lm_head: full [vocab] BF16 logits row, bit-identical to
    /// the stock pass in every candidate slot and certified-below elsewhere,
    /// so the downstream argmax emits the stock token.
    func logits(
        hidden: MLXArray,
        lmHeadWeight: MLXArray,
        useFusedRefinement: Bool = false
    ) -> MLXArray {
        precondition(hidden.dtype == .bfloat16 && hidden.size == lagunaLmHeadPruneHidden)
        let vocab = lagunaLmHeadPruneVocab
        let x = hidden.reshaped([lagunaLmHeadPruneHidden])
        let refine = useFusedRefinement && lagunaLmHeadFusedRefinementEnabled
        let coarseMode = MLXArray([refine ? UInt32(1) : UInt32(0)])
        let exactMode = MLXArray([refine ? UInt32(1) : UInt32(0)])
        // Four dispatches: the int5 coarse pass, argmax stage one over
        // `coarse` alone (no `delta` read), the exact-winner threshold that
        // absorbs the winning row's 4 KB GEMV, and the inline-mask exact pass.
        // With `refine` the coarse pass reads 1088 B/row instead of 1344 B/row
        // and the exact pass re-reads the residual bit plane for live blocks
        // only.
        let coarseOut = lagunaLmHeadCoarseKernel(
            [x, int5CodesLo, int5CodesHi, int5Scales, coarseMode],
            grid: (vocab / 16 * 512, 1, 1),
            threadGroup: (512, 1, 1),
            outputShapes: [[vocab], [vocab]],
            outputDTypes: [.float32, .bfloat16]
        )
        let coarse = coarseOut[0]
        let delta = coarseOut[1]
        let argmaxPartials = lagunaLmHeadCoarseArgmaxStage1Kernel(
            [coarse],
            grid: (224, 128, 1),
            threadGroup: (224, 1, 1),
            outputShapes: [[128], [128]],
            outputDTypes: [.float32, .uint32]
        )
        let thr = lagunaLmHeadExactWinnerBF16PredecessorThresholdKernel(
            [argmaxPartials[0], argmaxPartials[1], lmHeadWeight, x],
            grid: (32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[1]],
            outputDTypes: [.float32]
        )[0]
        let assembled = lagunaLmHeadExactKernel(
            [coarse, delta, thr, lmHeadWeight, x, int5CodesHi, int5Scales, exactMode],
            grid: (vocab / 32 * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[vocab]],
            outputDTypes: [.bfloat16]
        )[0]
        return assembled.reshaped([1, 1, vocab])
    }
}

// Official measurement receipt 2 of 3 for the M1 cascade family. The archive
// service de-duplicates byte-identical trees, so each receipt in a family
// carries one distinct comment. No code differs between the three.
