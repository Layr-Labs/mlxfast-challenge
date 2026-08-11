import Foundation
import MLX
import MLXFast


private let lagunaLmHeadPruneVocab = 100_352
private let lagunaLmHeadPruneHidden = 2048

let lagunaLmHeadPruneEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LM_HEAD_PRUNE"] != "0"

let lagunaLmHeadPrunePrefillEnabled =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_LM_HEAD_PRUNE_PREFILL"] != "0"

let lagunaLmHeadFusedRefinementEnabled =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_LMHEAD_FUSED_REFINEMENT"] != "0"

let lagunaLmHeadRow32Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_LM_HEAD_ROW32"] != "0"

private let lagunaTraceFusionEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_TRACE_FUSION"] == "1"

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

private let lagunaLmHeadInt5CoarseRatioBoundDeltaBF16Kernel = MLXFast.metalKernel(
    name: "laguna_lmhead_int5_inline_coarse_ratio_bound_delta_bf16_v6",
    inputNames: ["x", "codes_lo", "codes_hi", "scales"],
    outputNames: ["coarse", "delta"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 16 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint8_t* lorow = codes_lo + size_t(row) * 1024;
        const device uint8_t* hirow = codes_hi + size_t(row) * 256;
        const device uint8_t* srow = scales + size_t(row) * 64;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
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
                // Word w: elements 8w..8w+7 of the group. Nibble plane byte
                // b holds elements 2b (low) / 2b+1 (high); 1-bit plane bit j
                // of the group's word holds element j's residual bit.
                uint lw = lo4[w];
                uint hw = hb >> (8u * w);
                uint4 ne = (uint4(lw) >> uint4(0u, 8u, 16u, 24u)) & 15u;
                uint4 no = (uint4(lw) >> uint4(4u, 12u, 20u, 28u)) & 15u;
                uint4 he = (uint4(hw) >> uint4(0u, 2u, 4u, 6u)) & 1u;
                uint4 ho = (uint4(hw) >> uint4(1u, 3u, 5u, 7u)) & 1u;
                // The nibble stores floor(q/2)+8 and the bit plane stores
                // q-2*floor(q/2), so joining them rebuilds u = q + 16 in
                // [1, 31]; offset-binary decode is exact.
                float4 ve = float4((ne << 1u) | he) - 16.0f;
                float4 vo = float4((no << 1u) | ho) - 16.0f;
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
            // FP32 bound, then rounded UP to BF16 (mask-and-bump, sign clear).
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

private let lagunaLmHeadInt5BaseCoarseKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_int5_base_coarse_delta_bf16_v1",
    inputNames: ["x", "codes_base", "scales"],
    outputNames: ["coarse", "delta"],
    source: """
        constexpr float GAMMA = 0x1p-15f;

        uint row = threadgroup_position_in_grid.x * 16 +
            simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        const device uint8_t* crow = codes_base + size_t(row) * 1024;
        const device uint8_t* srow = scales + size_t(row) * 64;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        for (uint gg = 0; gg < 2; ++gg) {
            uint g = 2 * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            uint4 c4 = ((const device uint4*)(crow + g * 16))[0];
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
        c_acc = simd_sum(c_acc);
        d_acc = simd_sum(d_acc);
        if (lane == 0) {
            coarse[row] = c_acc;
            float d_up = d_acc * (1.0f + 32.0f * GAMMA);
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

private let lagunaLmHeadInlineExactDeltaBF16Kernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_inline_mask_block_delta_bf16_lane0_mask_v1",
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

        // The predicate is simdgroup-uniform, so lane 0 forms it once and
        // broadcasts the four row decisions. Reusing the mask below removes
        // the same coarse/delta/threshold reads from the final write path.
        uint candidate_mask = 0;
        if (lane == 0) {
            #pragma unroll
            for (uint tm = 0; tm < 4; ++tm) {
                uint r = base + tm;
                if (r < VOCAB && coarse[r] + float(delta[r]) >= thr[0]) {
                    candidate_mask |= 1u << tm;
                }
            }
        }
        candidate_mask = simd_broadcast(candidate_mask, 0);

        if (candidate_mask == 0) {
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
                    assembled[r] = (candidate_mask & (1u << tm)) != 0
                        ? bfloat(result[tm])
                        : bfloat(coarse[r]);
                }
            }
        }
        """,
    ensureRowContiguous: true
)

private let lagunaLmHeadInlineExactDeltaBF16Row32Kernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_inline_mask_delta_bf16_row32_v1",
    inputNames: ["coarse", "delta", "thr", "lm_head", "x"],
    outputNames: ["assembled"],
    source: """
        constexpr uint K = 2048;

        uint tgid = threadgroup_position_in_grid.x;
        uint sgid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint base = tgid * 256 + sgid * 32;
        uint row = base + lane;

        bool is_candidate =
            coarse[row] + float(delta[row]) >= thr[0];
        bfloat assembled_value = bfloat(coarse[row]);

        // The owner loop is simdgroup-uniform. Across the whole grid it makes
        // exactly one candidate test per vocabulary row, matching the
        // accepted four-row mapping, but launches eight times fewer groups.
        #pragma clang loop unroll(disable)
        for (ushort owner = 0; owner < 32; ++owner) {
            uint owner_is_candidate = simd_broadcast(
                uint(is_candidate), owner);
            if (owner_is_candidate == 0u) {
                continue;
            }
            uint exact_row = base + uint(owner);

            // --- stock gemv_al replica begin (gemv.h:151-289) ---
            float result = 0.0f;
            thread bfloat inter[4];
            thread float v_coeff[4];
            uint bn = lane * 4;
            const device bfloat* mrow =
                lm_head + size_t(exact_row) * K;
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

            // Only lane zero owns the completed reduction. Broadcast that
            // exact bit pattern before handing it to the row-owning lane.
            result = simd_broadcast(result, 0);
            if (lane == owner) {
                assembled_value = bfloat(result);
            }
        }

        assembled[row] = assembled_value;
        """,
    ensureRowContiguous: true
)

private let lagunaLmHeadRefinedExactKernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_fused_int5_sparse_refine_v1",
    inputNames: [
        "coarse", "delta", "thr", "lm_head", "x", "codes_bit", "scales",
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

private let lagunaLmHeadRefinedExactRow32Kernel = MLXFast.metalKernel(
    name: "laguna_lmhead_exact_fused_int5_sparse_refine_row32_v1",
    inputNames: [
        "coarse", "delta", "thr", "lm_head", "x", "codes_bit", "scales",
    ],
    outputNames: ["assembled"],
    source: """
        constexpr uint K = 2048;

        uint tgid = threadgroup_position_in_grid.x;
        uint sgid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint base = tgid * 256 + sgid * 32;
        uint row = base + lane;

        bool is_base_candidate =
            coarse[row] + float(delta[row]) >= thr[0];
        bfloat assembled_value = bfloat(coarse[row]);

        #pragma clang loop unroll(disable)
        for (ushort owner = 0; owner < 32; ++owner) {
            uint owner_is_candidate = simd_broadcast(
                uint(is_base_candidate), owner);
            if (owner_is_candidate == 0u) {
                continue;
            }
            uint exact_row = base + uint(owner);

            const device uint8_t* hirow =
                codes_bit + size_t(exact_row) * 256;
            const device uint8_t* srow =
                scales + size_t(exact_row) * 64;
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
                    uint4 he =
                        (uint4(hw) >> uint4(0u, 2u, 4u, 6u)) & 1u;
                    uint4 ho =
                        (uint4(hw) >> uint4(1u, 3u, 5u, 7u)) & 1u;
                    float4 ve = float4(he) - 0.5f;
                    float4 vo = float4(ho) - 0.5f;
                    float4 xa =
                        as_type<float4>(uint4(xrow[2 * w]) << 16);
                    float4 xb =
                        as_type<float4>(uint4(xrow[2 * w + 1]) << 16);
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

            // Preserve the accepted kernel's lane-zero epilogue exactly, then
            // broadcast its values so the owning lane can perform the sole
            // output write after all candidate work is complete.
            float refined_coarse = 0.0f;
            uint is_refined_candidate = 0u;
            if (lane == 0) {
                refined_coarse = coarse[exact_row] + correction;
                float d_up = float(delta[exact_row]) * 0x1.005p-1f;
                uint dbits = as_type<uint>(d_up);
                uint dtrunc = dbits & 0xFFFF0000u;
                if (dtrunc != dbits) {
                    dtrunc += 0x00010000u;
                }
                float delta_up =
                    float(as_type<bfloat>(ushort(dtrunc >> 16)));
                is_refined_candidate = uint(
                    refined_coarse + delta_up >= thr[0]);
            }
            refined_coarse = simd_broadcast(refined_coarse, 0);
            is_refined_candidate =
                simd_broadcast(is_refined_candidate, 0);

            float selected_value = refined_coarse;
            if (is_refined_candidate != 0u) {
                // --- stock gemv_al replica begin (gemv.h:151-289) ---
                float result = 0.0f;
                thread bfloat inter[4];
                thread float v_coeff[4];
                uint bn = lane * 4;
                const device bfloat* mrow =
                    lm_head + size_t(exact_row) * K;
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
                selected_value = simd_broadcast(result, 0);
            }

            if (lane == owner) {
                assembled_value = bfloat(selected_value);
            }
        }

        assembled[row] = assembled_value;
        """,
    header: lagunaLmHeadPruneHeader,
    ensureRowContiguous: true
)

private func lagunaLmHeadAssembleInlineExact(
    coarse: MLXArray,
    delta: MLXArray,
    threshold: MLXArray,
    lmHeadWeight: MLXArray,
    hidden: MLXArray,
    useRow32: Bool
) -> MLXArray {
    let kernel: MLXFast.MLXFastKernel =
        useRow32
        ? lagunaLmHeadInlineExactDeltaBF16Row32Kernel
        : lagunaLmHeadInlineExactDeltaBF16Kernel
    let gridThreads =
        useRow32
        ? lagunaLmHeadPruneVocab
        : lagunaLmHeadPruneVocab / 32 * 256
    return kernel(
        [coarse, delta, threshold, lmHeadWeight, hidden],
        grid: (gridThreads, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[lagunaLmHeadPruneVocab]],
        outputDTypes: [.bfloat16]
    )[0]
}
private func lagunaLmHeadAssembleRefinedExact(
    coarse: MLXArray,
    delta: MLXArray,
    threshold: MLXArray,
    lmHeadWeight: MLXArray,
    hidden: MLXArray,
    codesBit: MLXArray,
    scales: MLXArray,
    useRow32: Bool
) -> MLXArray {
    let kernel: MLXFast.MLXFastKernel =
        useRow32
        ? lagunaLmHeadRefinedExactRow32Kernel
        : lagunaLmHeadRefinedExactKernel
    let gridThreads =
        useRow32
        ? lagunaLmHeadPruneVocab
        : lagunaLmHeadPruneVocab / 32 * 256
    return kernel(
        [coarse, delta, threshold, lmHeadWeight, hidden, codesBit, scales],
        grid: (gridThreads, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[lagunaLmHeadPruneVocab]],
        outputDTypes: [.bfloat16]
    )[0]
}

func lagunaLmHeadAssembleForTesting(
    coarse: MLXArray,
    delta: MLXArray,
    threshold: MLXArray,
    lmHeadWeight: MLXArray,
    hidden: MLXArray,
    useRow32: Bool,
    refinement: (codesBit: MLXArray, scales: MLXArray)? = nil
) -> MLXArray {
    precondition(coarse.shape == [lagunaLmHeadPruneVocab])
    precondition(coarse.dtype == .float32)
    precondition(delta.shape == [lagunaLmHeadPruneVocab])
    precondition(delta.dtype == .bfloat16)
    precondition(threshold.shape == [1])
    precondition(threshold.dtype == .float32)
    precondition(
        lmHeadWeight.shape
            == [lagunaLmHeadPruneVocab, lagunaLmHeadPruneHidden])
    precondition(lmHeadWeight.dtype == .bfloat16)
    precondition(hidden.shape == [lagunaLmHeadPruneHidden])
    precondition(hidden.dtype == .bfloat16)
    if let refinement {
        precondition(
            refinement.codesBit.shape
                == [lagunaLmHeadPruneVocab, lagunaLmHeadPruneHidden / 8])
        precondition(refinement.codesBit.dtype == .uint8)
        precondition(
            refinement.scales.shape
                == [lagunaLmHeadPruneVocab, lagunaLmHeadPruneHidden / 32])
        precondition(refinement.scales.dtype == .uint8)
        return lagunaLmHeadAssembleRefinedExact(
            coarse: coarse,
            delta: delta,
            threshold: threshold,
            lmHeadWeight: lmHeadWeight,
            hidden: hidden,
            codesBit: refinement.codesBit,
            scales: refinement.scales,
            useRow32: useRow32
        )
    }
    return lagunaLmHeadAssembleInlineExact(
        coarse: coarse,
        delta: delta,
        threshold: threshold,
        lmHeadWeight: lmHeadWeight,
        hidden: hidden,
        useRow32: useRow32
    )
}

final class LagunaLmHeadPruner {
    let int5CodesLo: MLXArray
    let int5CodesHi: MLXArray
    let int5Scales: MLXArray

    var residentArrays: [MLXArray] { [int5CodesLo, int5CodesHi, int5Scales] }

    init?(lmHeadWeight: MLXArray) {
        guard lmHeadWeight.shape == [lagunaLmHeadPruneVocab, lagunaLmHeadPruneHidden],
            lmHeadWeight.dtype == .bfloat16
        else {
            FileHandle.standardError.write(
                Data("mlxfast: lm_head prune: unrecognized lm_head shape/dtype; disabled\n".utf8))
            return nil
        }
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
        let bump = (mant .>= MLXArray(UInt32(0x78_0000))).asType(.int32)
        let sdByte = clip(biasedE - 3 + bump, min: 0, max: 255)
        let sd = which(
            sdByte .== 0,
            MLXArray(Float(bitPattern: 0x0040_0000)),  // 2^-127, e8m0 semantics
            (sdByte.asType(.uint32) << 23).view(dtype: .float32))
        let q = (w / sd.expandedDimensions(axis: 2)).round()
        let maxCode = MLX.abs(q).max().item(Float.self)
        guard maxCode <= 15.0 else {
            FileHandle.standardError.write(
                Data(
                    "mlxfast: lm_head prune: int5 code overflow (\(maxCode)); using stock head\n"
                        .utf8))
            return nil
        }
        let u = (q + 16).asType(.uint8).reshaped([vocab, hidden])
        let base = u >> 1
        let u16 = base.view(dtype: .uint16)  // [V, 1024]: elem 2b low byte
        let lo =
            ((u16 & MLXArray(UInt16(0x000F)))
            | ((u16 >> 4) & MLXArray(UInt16(0x00F0)))).asType(.uint8)
        let u32 = u.view(dtype: .uint32)  // [V, 512]: elem 4t..4t+3
        let nib =
            ((u32 & MLXArray(UInt32(0x01)))
            | ((u32 >> 7) & MLXArray(UInt32(0x02)))
            | ((u32 >> 14) & MLXArray(UInt32(0x04)))
            | ((u32 >> 21) & MLXArray(UInt32(0x08)))).asType(.uint8)
        let nib16 = nib.view(dtype: .uint16)  // [V, 256]
        let hi =
            ((nib16 & MLXArray(UInt16(0x000F)))
            | ((nib16 >> 4) & MLXArray(UInt16(0x00F0)))).asType(.uint8)
        return (lo, hi, sdByte.asType(.uint8))
    }

    func logits(
        hidden: MLXArray,
        lmHeadWeight: MLXArray,
        useFusedRefinement: Bool = false
    ) -> MLXArray {
        precondition(hidden.dtype == .bfloat16 && hidden.size == lagunaLmHeadPruneHidden)
        let vocab = lagunaLmHeadPruneVocab
        let x = hidden.reshaped([lagunaLmHeadPruneHidden])
        let refine = useFusedRefinement && lagunaLmHeadFusedRefinementEnabled
        let coarseOut =
            refine
            ? lagunaLmHeadInt5BaseCoarseKernel(
                [x, int5CodesLo, int5Scales],
                grid: (vocab / 16 * 512, 1, 1),
                threadGroup: (512, 1, 1),
                outputShapes: [[vocab], [vocab]],
                outputDTypes: [.float32, .bfloat16]
            )
            : lagunaLmHeadInt5CoarseRatioBoundDeltaBF16Kernel(
                [x, int5CodesLo, int5CodesHi, int5Scales],
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
        let assembled =
            refine
            ? lagunaLmHeadAssembleRefinedExact(
                coarse: coarse,
                delta: delta,
                threshold: thr,
                lmHeadWeight: lmHeadWeight,
                hidden: x,
                codesBit: int5CodesHi,
                scales: int5Scales,
                useRow32: lagunaLmHeadRow32Enabled
            )
            : lagunaLmHeadAssembleInlineExact(
                coarse: coarse,
                delta: delta,
                threshold: thr,
                lmHeadWeight: lmHeadWeight,
                hidden: x,
                useRow32: lagunaLmHeadRow32Enabled
            )
        return assembled.reshaped([1, 1, vocab])
    }
}
