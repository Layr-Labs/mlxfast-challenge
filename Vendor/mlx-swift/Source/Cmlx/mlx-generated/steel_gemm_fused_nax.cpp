namespace mlx::core::metal {

const char* steel_gemm_fused_nax() {
  return R"preamble(
// Copyright © 2025 Apple Inc.

// Auto generated source for mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused_nax.h

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused_nax.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused_nax.h"
// Copyright © 2025 Apple Inc.

using namespace mlx::steel;

constant bool has_batch [[function_constant(10)]];

constant bool use_out_source [[function_constant(100)]];
constant bool do_axpby [[function_constant(110)]];

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];
// Private Laguna prefill specialization.  The host always supplies this
// constant; false leaves the upstream kernel resource- and instruction-neutral.
constant bool laguna_qk_prefill [[function_constant(203)]];

METAL_FUNC bool laguna_qk_metadata_matches(
    const device bfloat* metadata,
    const ushort query_heads) {
  const device ushort* bits =
      reinterpret_cast<const device ushort*>(metadata);
  const ushort angle_width = query_heads == 48 ? 64 : 128;
  return bits[256] == 0x4c51 && bits[257] == 0x514b &&
      bits[258] == 0x4e52 && bits[259] == 0x5031 && bits[260] == 1 &&
      bits[261] == query_heads && bits[262] == angle_width &&
      bits[263] == 512;
}

// Consume the BF16-rounded GEMM tile exactly where the stock kernel would
// have stored it, then reproduce Laguna's head-128 RMSNorm and rotary step.
// The destination is intentionally canonical [head, token, dim] storage:
// all query heads first, followed by all key heads.
template <int BM, int BN, int WM, int WN>
METAL_FUNC void laguna_qk_norm_rope_epilogue(
    const threadgroup bfloat* tile,
    const device bfloat* metadata,
    device bfloat* output,
    const uint c_row,
    const uint c_col,
    const ushort query_heads,
    const uint simd_group_id) {
  constexpr uint kLength = 512;
  constexpr uint kHeadDim = 128;
  constexpr uint kSimdgroups = WM * WN;
  constexpr float kNormEps = 1.0e-6f;
  constexpr float kYaRNMScale = 1.3465735912322998f;

  const uint head = c_col / kHeadDim;
  const bool is_query = head < query_heads;
  const uint family_head = is_query ? head : head - query_heads;
  const device bfloat* norm_weight = metadata + (is_query ? 0 : kHeadDim);
  const device float* angles =
      reinterpret_cast<const device float*>(metadata + 2048);
  const uint angle_width = query_heads == 48 ? 64 : 128;
  const uint lane = __metal_get_thread_index_in_simdgroup(ushort());
  const uint base = lane * 4;

  for (uint local_row = simd_group_id; local_row < BM;
       local_row += kSimdgroups) {
    const uint token = c_row + local_row;
    const threadgroup bfloat* input = tile + local_row * BN;
    thread bfloat normalized[4];
    float sum = 0.0f;
    STEEL_PRAGMA_UNROLL
    for (uint i = 0; i < 4; ++i) {
      const float value = float(input[base + i]);
      sum += value * value;
    }
    sum = simd_sum(sum);
    const float inverse_rms =
        metal::precise::rsqrt(sum / 128.0f + kNormEps);
    STEEL_PRAGMA_UNROLL
    for (uint i = 0; i < 4; ++i) {
      normalized[i] = norm_weight[base + i] *
          bfloat(float(input[base + i]) * inverse_rms);
    }

    const uint partner_delta = query_heads == 48 ? 8 : 16;
    thread float paired[4];
    STEEL_PRAGMA_UNROLL
    for (uint i = 0; i < 4; ++i) {
      paired[i] = simd_shuffle(
          float(normalized[i]), ushort(lane ^ partner_delta));
    }

    const device float* token_angles = angles + token * angle_width;
    const ulong family_base = is_query
        ? 0
        : ulong(query_heads) * kLength * kHeadDim;
    device bfloat* head_output = output + family_base +
        (ulong(family_head) * kLength + token) * kHeadDim;

    if (query_heads == 48) {
      // Full attention: YaRN over the first 64 dimensions, with the same two
      // BF16 mscale boundaries as the stock RoPE path.
      if (lane < 8) {
        const bfloat rounded_mscale = bfloat(kYaRNMScale);
        STEEL_PRAGMA_UNROLL
        for (uint i = 0; i < 4; ++i) {
          const uint pair = base + i;
          const float first =
              float(bfloat(normalized[i] * rounded_mscale));
          const float second =
              float(bfloat(bfloat(paired[i]) * rounded_mscale));
          const float cosine = token_angles[pair];
          const float sine = token_angles[pair + 32];
          head_output[pair] = bfloat(first * cosine - second * sine);
          head_output[pair + 32] = bfloat(first * sine + second * cosine);
        }
      } else if (lane >= 16) {
        STEEL_PRAGMA_UNROLL
        for (uint i = 0; i < 4; ++i) {
          head_output[base + i] = normalized[i];
        }
      }
    } else if (lane < 16) {
      // Sliding attention: ordinary RoPE over all 128 dimensions.
      STEEL_PRAGMA_UNROLL
      for (uint i = 0; i < 4; ++i) {
        const uint pair = base + i;
        const float first = float(normalized[i]);
        const float second = paired[i];
        const float cosine = token_angles[pair];
        const float sine = token_angles[pair + 64];
        head_output[pair] = bfloat(first * cosine - second * sine);
        head_output[pair + 64] = bfloat(first * sine + second * cosine);
      }
    }
  }
}

// clang-format off
template <
    bool kAlignedM,
    bool kAlignedN,
    class NAXTile_t,
    typename T>
void gemm_epilogue(
    thread NAXTile_t& Dtile,
    const device T* C,
    const constant GEMMParams* params,
    const constant GEMMAddMMParams* addmm_params,
    const short sgp_sm, 
    const short sgp_sn) { // clang-format on

  (void)params;

  using V = typename NAXTile_t::elem_type;

  constexpr short TM = NAXTile_t::kTileRows;
  constexpr short TN = NAXTile_t::kTileCols;
  constexpr short kElemsPerFrag = NAXTile_t::kElemsPerFrag;

  using CFrag = typename NAXTile_t::NAXFrag_t;
  using cfrag_t = typename CFrag::template dtype_frag_t<T>;

  const_for_loop<0, TM, 1>([&](auto mm) {
    const_for_loop<0, TN, 1>([&](auto nn) {
      auto m = mm * Int<CFrag::kFragRows>{};
      auto n = nn * Int<CFrag::kFragCols>{};

      cfrag_t celems;

      if constexpr (kAlignedM && kAlignedN) {
        CFrag::load(celems, C, addmm_params->ldc, addmm_params->fdc, m, n);
      } else {
        CFrag::load_safe(
            celems,
            C,
            addmm_params->ldc,
            addmm_params->fdc,
            sgp_sm,
            sgp_sn,
            m,
            n);
      }

      thread auto& delems = Dtile.template frag_at<mm, nn>();

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kElemsPerFrag; i++) {
        if (do_axpby) {
          delems[i] = addmm_params->alpha * delems[i] +
              addmm_params->beta * static_cast<V>(celems[i]);
        } else {
          delems[i] += static_cast<V>(celems[i]);
        }
      }
    });
  });
}

// clang-format off
template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void gemm(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device T* C [[buffer(2), function_constant(use_out_source)]],
    device T* D [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant GEMMAddMMParams* addmm_params [[buffer(5), function_constant(use_out_source)]],
    const constant int* batch_shape [[buffer(6), function_constant(has_batch)]],
    const constant int64_t* batch_strides [[buffer(7), function_constant(has_batch)]],
    threadgroup bfloat* laguna_tile
        [[threadgroup(0), function_constant(laguna_qk_prefill)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) { // clang-format on
  // Find block
  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  // Exit early if out of bounds
  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  // Adjust for batch
  if (has_batch) {
    const constant auto* A_bstrides = batch_strides;
    const constant auto* B_bstrides = batch_strides + params->batch_ndim;

    ulong2 batch_offsets = elem_to_loc_broadcast(
        tid.z, batch_shape, A_bstrides, B_bstrides, params->batch_ndim);

    A += batch_offsets.x;
    B += batch_offsets.y;

    if (use_out_source) {
      const constant auto* C_bstrides = B_bstrides + params->batch_ndim;
      C += elem_to_loc(tid.z, batch_shape, C_bstrides, params->batch_ndim);
    }
  } else {
    A += params->batch_stride_a * tid.z;
    B += params->batch_stride_b * tid.z;

    if (use_out_source) {
      C += addmm_params->batch_stride_c * tid.z;
    }
  }

  D += params->batch_stride_d * tid.z;
  const device T* B_root = B;
  device T* D_root = D;

  // Prepare threadgroup memory
  threadgroup_barrier(mem_flags::mem_none);

  // Find block in A, B, C
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  bool laguna_qk_special = false;
  ushort laguna_query_heads = 0;
  const device bfloat* laguna_metadata = nullptr;
  if constexpr (
      metal::is_same_v<T, bfloat> && !transpose_a && transpose_b &&
      BM == 64 && BN == 128 && BK == 256 && WM == 2 && WN == 4) {
    if (laguna_qk_prefill && params->M == 512 && params->K == 2048 &&
        (params->N == 7296 || params->N == 9344)) {
      laguna_query_heads = params->N == 7296 ? 48 : 64;
      const uint semantic_rows = params->N - BN;
      laguna_metadata = reinterpret_cast<const device bfloat*>(B_root) +
          ulong(semantic_rows) * params->K;
      // The version header is uniform.  One lane per simdgroup performs its
      // eight loads, then broadcasts a ushort so every branch remains
      // simdgroup-uniform without a threadgroup barrier.
      const ushort metadata_match =
          __metal_get_thread_index_in_simdgroup(ushort()) == 0 &&
              laguna_qk_metadata_matches(laguna_metadata, laguna_query_heads)
          ? 1
          : 0;
      laguna_qk_special = simd_broadcast_first(metadata_match) != 0;
      // The final N tile is only a metadata carrier.  It must not write the
      // ordinary row-major locations because those overlap the canonical
      // flattened Q/K regions produced by semantic tiles.
      if (laguna_qk_special && uint(c_col) >= semantic_rows) {
        device bfloat* tail = reinterpret_cast<device bfloat*>(D_root) +
            ulong(semantic_rows) * 512 + ulong(c_row) * BN;
        const uint local_id = simd_group_id * 32 +
            __metal_get_thread_index_in_simdgroup(ushort());
        STEEL_PRAGMA_NO_UNROLL
        for (uint i = local_id; i < BM * BN; i += WM * WN * 32) {
          tail[i] = bfloat(0.0f);
        }
        return;
      }
    }
  }

  A += transpose_a ? c_row_long : c_row_long * params->lda;
  B += transpose_b ? c_col_long * params->ldb : c_col_long;
  D += c_row_long * params->ldd + c_col_long;

  if (use_out_source) {
    C += c_row_long * addmm_params->ldc + c_col_long * addmm_params->fdc;
  }

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const int sgp_sm_int =
      align_M ? int(SM) : min(int(SM), params->M - (c_row + tm));
  const short sgp_sm = short(sgp_sm_int);
  const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);

  const int sgp_sn_int =
      align_N ? int(SN) : min(int(SN), params->N - (c_col + tn));
  const short sgp_sn = short(sgp_sn_int);
  const bool is_unaligned_sn = align_N ? false : (sgp_sn != SN);

  A += transpose_a ? tm : (tm * params->lda);
  B += transpose_b ? (tn * params->ldb) : tn;
  D += tm * params->ldd + tn;

  if (use_out_source) {
    C += tm * addmm_params->ldc + tn * addmm_params->fdc;
  }

  NAXTile<AccumType, TM, TN> Dtile;

  dispatch_bool(align_K, [&](auto kAlignedK) {
    dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(align_N || !is_unaligned_sn, [&](auto kAlignedN) {
        Dtile = gemm_loop<
            T,
            SM,
            SN,
            SK,
            BK,
            transpose_a,
            transpose_b,
            kAlignedM.value,
            kAlignedN.value,
            kAlignedK.value,
            AccumType>(
            A,
            B,
            params->lda,
            params->ldb,
            params->K,
            params->gemm_k_iterations_aligned,
            sgp_sm,
            sgp_sn);
        if (use_out_source) {
          gemm_epilogue<kAlignedM.value, kAlignedN.value>(
              Dtile, C, params, addmm_params, sgp_sm, sgp_sn);
        }
        if constexpr (
            metal::is_same_v<T, bfloat> && !transpose_a && transpose_b &&
            BM == 64 && BN == 128 && BK == 256 && WM == 2 && WN == 4) {
          if (laguna_qk_special) {
            Dtile.template store<bfloat, BN, 1>(
                laguna_tile + tm * BN + tn);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            laguna_qk_norm_rope_epilogue<BM, BN, WM, WN>(
                laguna_tile,
                laguna_metadata,
                reinterpret_cast<device bfloat*>(D_root),
                uint(c_row),
                uint(c_col),
                laguna_query_heads,
                simd_group_id);
          } else if constexpr (kAlignedM && kAlignedN) {
            Dtile.store(D, int(params->ldd));
          } else {
            Dtile.store_safe(D, int(params->ldd), short2(sgp_sn, sgp_sm));
          }
        } else if constexpr (kAlignedM && kAlignedN) {
          Dtile.store(D, int(params->ldd));
        } else {
          Dtile.store_safe(D, int(params->ldd), short2(sgp_sn, sgp_sm));
        }
      });
    });
  });
}

///////////////////////////////////////////////////////////////////////////////
)preamble";
}

} // namespace mlx::core::metal
