// Copyright © 2025 Apple Inc.

using namespace mlx::steel;

constant bool has_batch [[function_constant(10)]];

constant bool use_out_source [[function_constant(100)]];
constant bool do_axpby [[function_constant(110)]];

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

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

///////////////////////////////////////////////////////////////////////////////
// Lossless packed-B BF16 path used by Laguna's multi-token QKV projection.
//
// The packer keeps every BF16 sign, exponent, and mantissa bit. Normal
// 32-value blocks use 49 bytes; exceptional exponent spans point at one of
// fifteen raw 64-byte slots in the same original 4,096-byte row. Decoding
// changes only where B fragment values are loaded from. The NAX cooperative
// tensors, matmul descriptor, K traversal, accumulator type, and store path
// below are identical to gemm_loop / gemm.
///////////////////////////////////////////////////////////////////////////////

template <typename T>
METAL_FUNC T laguna_lossless_bf16_value(
    const device T* packed,
    const int row,
    const int column) {
  constexpr int K = 2048;
  constexpr int kBlockValues = 32;
  constexpr int kRecordBytes = 49;
  constexpr int kPrimaryBytes = (K / kBlockValues) * kRecordBytes;
  constexpr int kRawSlotBytes = kBlockValues * int(sizeof(ushort));

  const device uchar* row_base =
      reinterpret_cast<const device uchar*>(packed) + size_t(row) * K * 2;
  const int block = column / kBlockValues;
  const int within = column % kBlockValues;
  const device uchar* record = row_base + block * kRecordBytes;
  const uint base = record[48];

  if (base == 255u) {
    const uint slot = record[32];
    if (slot >= 15u) {
      return as_type<T>(ushort(0x7fc1));
    }
    const device ushort* raw =
        reinterpret_cast<const device ushort*>(
            row_base + kPrimaryBytes + slot * kRawSlotBytes);
    return as_type<T>(raw[within]);
  }
  const uint sign_mantissa = record[within];
  const uint pair = record[32 + within / 2];
  const uint delta = (pair >> ((within & 1) * 4)) & 15u;
  const ushort bits = ushort(
      ((sign_mantissa & 128u) << 8) |
      ((base + delta) << 7) |
      (sign_mantissa & 127u));
  return as_type<T>(bits);
}

template <typename T, short TR, short TC>
METAL_FUNC void laguna_load_lossless_btile(
    thread NAXTile<T, TR, TC>& tile,
    const device T* packed,
    const int row_base,
    const int column_base) {
  const short2 sc = BaseNAXFrag::get_coord();
  STEEL_PRAGMA_UNROLL
  for (short fr = 0; fr < TR; ++fr) {
    STEEL_PRAGMA_UNROLL
    for (short fc = 0; fc < TC; ++fc) {
      thread auto& frag = tile.frag_at(fr, fc);
      STEEL_PRAGMA_UNROLL
      for (short er = 0; er < BaseNAXFrag::kElemRows; ++er) {
        const int row =
            row_base + fr * BaseNAXFrag::kFragRows + sc.y +
            er * BaseNAXFrag::kElemRowsJump;
        STEEL_PRAGMA_UNROLL
        for (short ec = 0; ec < BaseNAXFrag::kElemCols; ++ec) {
          const int column =
              column_base + fc * BaseNAXFrag::kFragCols + sc.x + ec;
          frag[er * BaseNAXFrag::kElemCols + ec] =
              laguna_lossless_bf16_value(packed, row, column);
        }
      }
    }
  }
}

template <
    typename T,
    short SM,
    short SN,
    short SK,
    short BK,
    typename AccumType = float>
METAL_FUNC auto laguna_lossless_bf16_gemm_loop(
    const device T* A,
    const device T* packed_B,
    const int lda,
    const int K,
    const int output_row_base) {
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  STEEL_PRAGMA_NO_UNROLL
  for (int kk0 = 0; kk0 < K / BK; ++kk0) {
    threadgroup_barrier(mem_flags::mem_none);
    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      NAXTile<T, TM, TK> Atile;
      NAXTile<T, TN, TK> Btile;
      Atile.load(A + kk1, lda);
      laguna_load_lossless_btile(
          Btile,
          packed_B,
          output_row_base,
          kk0 * BK + kk1);
      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
    }
    A += BK;
  }
  return Dtile;
}

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
gemm_lossless_bf16_b(
    const device T* A [[buffer(0)]],
    const device T* packed_B [[buffer(1)]],
    device T* D [[buffer(2)]],
    const constant GEMMParams* params [[buffer(3)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;
  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  threadgroup_barrier(mem_flags::mem_none);
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  A += size_t(c_row) * params->lda;
  D += size_t(c_row) * params->ldd + c_col;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  A += tm * params->lda;
  D += tm * params->ldd + tn;
  auto Dtile = laguna_lossless_bf16_gemm_loop<
      T, SM, SN, SK, BK, AccumType>(
      A, packed_B, params->lda, params->K, c_col + tn);
  Dtile.store(D, int(params->ldd));
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

  // Prepare threadgroup memory
  threadgroup_barrier(mem_flags::mem_none);

  // Find block in A, B, C
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

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
        if constexpr (kAlignedM && kAlignedN) {
          Dtile.store(D, int(params->ldd));
        } else {
          Dtile.store_safe(D, int(params->ldd), short2(sgp_sn, sgp_sm));
        }
      });
    });
  });
}
