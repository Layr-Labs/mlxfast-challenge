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

private let lagunaLmHeadPruneVocab = 100_352
private let lagunaLmHeadPruneHidden = 2048

/// Master switch for the certified two-pass decode lm_head (notes/68).
/// Operator override 2026-07-28: default flipped to OFF on the reference
/// branch. This module arrived via submission e36827ee default-ON; leaving a
/// single submission's optimization as the shipped default at the reference
/// HEAD would silently apply it to anyone building from main and could leak
/// into a future baseline re-pin, so on main it is opt-in only. The
/// as-submitted design and certification below are unchanged; enable it with
/// `DARKBLOOM_LM_HEAD_PRUNE=1`. Any other value (including unset) keeps the
/// byte-identical stock full lm_head pass. (The algorithm comments below were
/// authored against the original default-ON contract; the switch here is the
/// authority.)
let lagunaLmHeadPruneEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LM_HEAD_PRUNE"] == "1"

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
    static inline float laguna_hs8(uint8_t b) {
        uint mag = uint(b) & 127u;
        if (mag == 126u) {
            return 186.0f;
        }
        uint e = mag >> 3;
        if (e == 0u) {
            return 0x1p-10f;
        }
        return as_type<float>((e + 116u) << 23);  // 2^(e-11), exact
    }
    """

/// Fused MXFP8 coarse GEMV + certified bound + BF16 pre-fill.
/// One simdgroup per row; lane covers 64 consecutive elements (2 groups).
private let lagunaLmHeadCoarseKernel = MLXFast.metalKernel(
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

/// Retained init-time MXFP8 coarse copy of lm_head plus the pruned decode
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

    /// Pruned decode lm_head: full [vocab] BF16 logits row, bit-identical to
    /// the stock pass in every candidate slot and certified-below elsewhere,
    /// so the downstream argmax emits the stock token.
    func logits(hidden: MLXArray, lmHeadWeight: MLXArray) -> MLXArray {
        precondition(hidden.dtype == .bfloat16 && hidden.size == lagunaLmHeadPruneHidden)
        let vocab = lagunaLmHeadPruneVocab
        let x = hidden.reshaped([lagunaLmHeadPruneHidden])

        let coarseOut = lagunaLmHeadCoarseKernel(
            [x, codes, scales],
            grid: (vocab / 8 * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[vocab], [vocab], [vocab]],
            outputDTypes: [.float32, .float32, .bfloat16]
        )
        let coarse = coarseOut[0]
        let delta = coarseOut[1]
        let coarseBF = coarseOut[2]

        // Threshold on GPU: L = max(coarse - delta); thr = L - |L|/64.
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
        return assembled.reshaped([1, 1, vocab])
    }
}
