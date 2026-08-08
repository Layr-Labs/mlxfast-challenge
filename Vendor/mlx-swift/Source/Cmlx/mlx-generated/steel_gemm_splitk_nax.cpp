namespace mlx::core::metal {

const char* steel_gemm_splitk_nax() {
  return R"preamble(
// Copyright © 2025 Apple Inc.

// Auto generated source for mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk_nax.h

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk_nax.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk_nax.h"
// Copyright © 2026 Apple Inc.

using namespace mlx::steel;

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];

///////////////////////////////////////////////////////////////////////////////
// NAX Split-K GEMM kernel
///////////////////////////////////////////////////////////////////////////////

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
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void gemm_splitk_nax(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device AccumType* C [[buffer(2)]],
    const constant GEMMSpiltKParams* params [[buffer(3)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) { // clang-format on

  const int linear_tid = tid.x;

  // Compute swizzled tile dimensions
  const int tn_swizzled = params->tiles_n << params->swizzle_log;
  const int tm_swizzled =
      (params->tiles_m + (1 << params->swizzle_log) - 1) >> params->swizzle_log;
  const int tiles_per_partition = tn_swizzled * tm_swizzled;

  const int tid_z = linear_tid / tiles_per_partition;
  const int xy_flat = linear_tid % tiles_per_partition;

  // Decode 2D grid coordinates in swizzled space
  const int grid_x = xy_flat % tn_swizzled;
  const int grid_y = xy_flat / tn_swizzled;

  // Apply X-Y swizzle
  const int tid_y = (grid_y << params->swizzle_log) +
      (grid_x & ((1 << params->swizzle_log) - 1));
  const int tid_x = grid_x >> params->swizzle_log;

  // Exit early
  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  // Calculate partition bounds
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const int k_start = params->split_k_partition_size * tid_z;
  const int k_end = min(k_start + params->split_k_partition_size, params->K);

  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);
  const size_t k_start_long = size_t(k_start);

  // Adjust pointers for split-K partition
  A += transpose_a ? (c_row_long + k_start_long * params->lda)
                   : (k_start_long + c_row_long * params->lda);
  B += transpose_b ? (k_start_long + c_col_long * params->ldb)
                   : (c_col_long + k_start_long * params->ldb);
  C += (size_t(params->split_k_partition_stride) * tid_z) +
      (c_row_long * params->ldc + c_col_long);

  // NAX tile configuration
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;

  // Calculate simdgroup offsets and alignment
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
  C += tm * params->ldc + tn;

  NAXTile<AccumType, TM, TN> Dtile;

  // gemm_loop through the partition
  // Check K-alignment at runtime (partition-specific)
  const int partition_k_size = k_end - k_start;
  const int partition_k_iters = partition_k_size / BK;
  const bool partition_k_aligned = (partition_k_size % BK) == 0;

  dispatch_bool(partition_k_aligned, [&](auto kAlignedK) {
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
            partition_k_size,
            partition_k_iters,
            sgp_sm,
            sgp_sn);
      });
    });
  });

  // Store result
  dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(align_N || !is_unaligned_sn, [&](auto kAlignedN) {
      if constexpr (kAlignedM && kAlignedN) {
        Dtile.store(C, int(params->ldc));
      } else {
        Dtile.store_safe(C, int(params->ldc), short2(sgp_sn, sgp_sm));
      }
    });
  });
}

// Laguna prefill-only entry: A is contiguous [H,M,128], G is contiguous
// [M,H], and their product rounds to BF16 before the unchanged NAX MMA chain.
template <typename T, int BM, int BN, int BK, int WM, int WN,
          bool transpose_a, bool transpose_b, typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]]
void gemm_splitk_nax_hmgate(
    const device T* A [[buffer(0)]], const device T* B [[buffer(1)]],
    device AccumType* C [[buffer(2)]],
    const constant GEMMSpiltKParams* p [[buffer(3)]],
    const device T* G [[buffer(4)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  static_assert(!transpose_a && transpose_b);
  const int tns = p->tiles_n << p->swizzle_log;
  const int tms = (p->tiles_m + (1 << p->swizzle_log) - 1) >> p->swizzle_log;
  const int per_part = tns * tms;
  const int part = int(tid.x) / per_part, xy = int(tid.x) % per_part;
  const int gx = xy % tns, gy = xy / tns;
  const int ty = (gy << p->swizzle_log) + (gx & ((1 << p->swizzle_log) - 1));
  const int tx = gx >> p->swizzle_log;
  if (p->tiles_n <= tx || p->tiles_m <= ty) return;

  // One head-major threadgroup owns two adjacent BN panels. The rounded A
  // fragment is identical for both, so keeping both accumulators live halves
  // the attended/gate loads and products without changing either output's
  // MMA chain.
  const int row0 = ty * BM, col0 = tx * (2 * BN);
  const int k0 = p->split_k_partition_size * part;
  const int kend = min(k0 + p->split_k_partition_size, p->K);
  B += size_t(k0) + size_t(col0) * p->ldb;
  C += size_t(p->split_k_partition_stride) * part + size_t(row0) * p->ldc + col0;

  constexpr short SM = BM / WM, SN = BN / WN, SK = 32;
  constexpr short TM = SM / 16, TN = SN / 16, TK = SK / 16;
  const short tm = SM * (sg / WN), tn = SN * (sg % WN);
  const short ms = short(align_M ? SM : min(int(SM), p->M - row0 - tm));
  const short ns0 = short(
      align_N ? SN : max(0, min(int(SN), p->N - col0 - tn)));
  const short ns1 = short(
      align_N ? SN : max(0, min(int(SN), p->N - col0 - BN - tn)));
  B += size_t(tn) * p->ldb;
  C += tm * p->ldc + tn;

  NAXTile<AccumType, TM, TN> D0;
  NAXTile<AccumType, TM, TN> D1;
  D0.clear();
  D1.clear();
  const int heads = p->K / 128;
  const short2 sc = BaseNAXFrag::get_coord();
  const int iters = (kend - k0) / BK;
  STEEL_PRAGMA_NO_UNROLL
  for (int kb = 0; kb < iters; ++kb) {
    threadgroup_barrier(mem_flags::mem_none);
    STEEL_PRAGMA_NO_UNROLL
    for (int hk = 0; hk < BK; hk += 128) {
      static_assert(BK % 128 == 0);
      thread T gate_rows[TM * 2];
      const int head = (k0 + kb * BK + hk) / 128;
      STEEL_PRAGMA_UNROLL
      for (short fi = 0; fi < TM; ++fi) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; ++i) {
          const int lm = fi * 16 + sc.y + i * 8;
          const int m = row0 + tm + lm;
          gate_rows[fi * 2 + i] =
              (align_M || lm < ms) ? G[m * heads + head] : T(0);
        }
      }
      STEEL_PRAGMA_UNROLL
      for (int kk = hk; kk < hk + 128; kk += SK) {
        NAXTile<T, TM, TK> At;
        NAXTile<T, TN, TK> Bt;
        volatile int compiler_barrier;
        STEEL_PRAGMA_UNROLL
        for (short fi = 0; fi < TM; ++fi) {
          STEEL_PRAGMA_UNROLL
          for (short fj = 0; fj < TK; ++fj) {
            thread auto& f = At.frag_at(fi, fj);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; ++i) {
              const int lm = fi * 16 + sc.y + i * 8;
              const int m = row0 + tm + lm;
              STEEL_PRAGMA_UNROLL
              for (short j = 0; j < 4; ++j) {
                const int d = kk - hk + fj * 16 + sc.x + j;
                T v = T(0);
                if (align_M || lm < ms) {
                  v = T(float(A[size_t(head) * (p->M * 128) + m * 128 + d]) *
                        float(gate_rows[fi * 2 + i]));
                }
                f[i * 4 + j] = v;
              }
            }
          }
        }
        const int boff = kb * BK + kk;
        if (align_N) Bt.load(B + boff, p->ldb);
        else Bt.load_safe(B + boff, p->ldb, short2(SK, ns0));
        tile_matmad_nax(
            D0, At, metal::bool_constant<false>{}, Bt,
            metal::bool_constant<true>{});
        if (align_N) {
          Bt.load(B + size_t(BN) * p->ldb + boff, p->ldb);
        } else if (ns1 > 0) {
          Bt.load_safe(
              B + size_t(BN) * p->ldb + boff, p->ldb, short2(SK, ns1));
        }
        if (align_N || ns1 > 0) {
          tile_matmad_nax(
              D1, At, metal::bool_constant<false>{}, Bt,
              metal::bool_constant<true>{});
        }
        (void)compiler_barrier;
      }
    }
  }
  if (align_M && align_N) {
    D0.store(C, int(p->ldc));
    D1.store(C + BN, int(p->ldc));
  } else {
    D0.store_safe(C, int(p->ldc), short2(ns0, ms));
    if (ns1 > 0) {
      D1.store_safe(C + BN, int(p->ldc), short2(ns1, ms));
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
)preamble";
}

} // namespace mlx::core::metal
