// Copyright © 2025 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/fp4.h"
#include "mlx/backend/metal/kernels/fp8.h"

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];
// Set by gather_qmm_rhs_nax when DARKBLOOM_PREFILL_GATHER_RUNSKIP selects this
// dispatch. Default OFF: the host always supplies it, and when false the kernel
// is byte-for-byte the upstream algorithm.
constant bool gather_run_skip [[function_constant(203)]];

// DARKBLOOM staging levers for fp_gather_qmm_rhs_nax. The host always supplies
// every function constant and ships all four ON. Each is fixed once per
// process, so no tunable magnitude ever enters the pipeline specialization
// key.
constant bool stage_widest [[function_constant(204)]];
constant bool stage_wideld [[function_constant(205)]];
constant bool stage_runbar [[function_constant(206)]];
constant bool stage_novol [[function_constant(207)]];

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

template <int wsize = 8, int bits>
inline constexpr short get_pack_factor() {
  return wsize / bits;
}

template <int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  return wsize / 8;
}

template <typename T, int group_size>
static inline T dequantize_scale(uint8_t s) {
  if constexpr (group_size == 16) {
    // Use nv scale
    return T(*(thread fp8_e4m3*)(&s));
  } else {
    return T(*(thread fp8_e8m0*)(&s));
  }
}

template <int bits>
struct Quantize {
  uint8_t operator()(float x) {
    if (bits == 8) {
      return fp8_e4m3(x).bits;
    } else {
      return fp4_e2m1(x).bits;
    }
  }
};

template <int bits, typename U = float>
struct Dequantize {
  U operator()(uint8_t x) {
    if constexpr (bits == 8) {
      return U(*(thread fp8_e4m3*)(&x));
    } else {
      return U(*(thread fp4_e2m1*)(&x));
    }
  }
};

template <typename U, int bits>
inline void dequantize(uint8_t w, U scale, threadgroup U* w_local) {
  if constexpr (bits == 4) {
    w_local[0] = scale * Dequantize<4, U>{}(w);
    w_local[1] = scale * Dequantize<4, U>{}(w >> 4);
  } else {
    w_local[0] = scale * Dequantize<8, U>{}(w);
  }
}

// 16B-aligned chunk used to give the Ws staging buffer a guaranteed 16B base
// address. Metal gives no alignas on a threadgroup array of scalars, and MSL
// has no pointer-to-integer cast for threadgroup addresses, so the alignment
// has to come from the element type. Declaring Ws as an array of these and
// reinterpreting to Wtype* changes nothing about the buffer's size, element
// count, layout, or contents -- it only pins the base address.
template <typename T>
struct alignas(16) NAXWsChunk16 {
  T v[16 / sizeof(T)];
};

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
struct QuantizedBlockLoader {
  MLX_MTL_CONST short pack_factor = get_pack_factor<8, bits>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;

  MLX_MTL_CONST short n_reads_per_scale = (n_reads * pack_factor) <= group_size
      ? n_reads
      : (group_size / pack_factor);
  MLX_MTL_CONST short n_steps_per_read = n_reads / n_reads_per_scale;

  MLX_MTL_CONST short n_groups = BCOLS / group_size;

  const int src_ld;
  const int tile_stride;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  const short group_id;

  threadgroup T* dst;
  const device uint8_t* src;
  const device uint8_t* scales;

  QuantizedBlockLoader(
      const device uint8_t* src_,
      const device uint8_t* scales_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED * bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_stride(BROWS * src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        group_id((bj * pack_factor) / group_size),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size + group_id) {}

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    int k = 0;
    for (int i = 0; i < n_steps_per_read; i++) {
      T scale = dequantize_scale<T, group_size>(scales[i]);
      for (int j = 0; j < n_reads_per_scale; j++) {
        dequantize<T, bits>(
            src[k * bytes_per_pack], scale, dst + k * pack_factor);
        k++;
      }
    }
  }

  // DARKBLOOM_STAGE_WIDEST / DARKBLOOM_STAGE_WIDELD.
  //
  // BIT-EXACTNESS. This writes exactly the same values to exactly the same
  // threadgroup addresses as load_unsafe(), decoded from exactly the same
  // source bytes with exactly the same scale for every element. NO FLOAT
  // ARITHMETIC IS TOUCHED AT ALL -- only the *width* of the device loads and
  // the threadgroup stores changes. The per-element expression is character
  // for character the one in dequantize<T, 4>:
  //     scale * Dequantize<4, T>{}(byte)        (low nibble)
  //     scale * Dequantize<4, T>{}(byte >> 4)   (high nibble)
  // so this is a strictly stronger exactness class than a barrier removal:
  // there is no reassociation, no accumulation-order change, and no rounding
  // boundary anywhere in the diff.
  //
  // ALIGNMENT IS A CORRECTNESS PRECONDITION, NOT AN ASSUMPTION. A misaligned
  // wide access is silent corruption, not a fault, so the preconditions are
  // split between the two places that can actually see them:
  //   * the HOST checks what only it knows -- the weight buffer's own byte
  //     offset, the per-expert stride, and the tile column base (see
  //     darkbloom_stage_wide_load_ok in quantized.cpp),
  //   * this loader checks what only it knows -- each thread's own offset
  //     within the tile, below, using integer arithmetic on the same
  //     expressions the constructor used (no pointer-to-integer casts, which
  //     MSL does not provide for threadgroup addresses).
  // If any precondition fails the thread runs the untouched scalar path, so
  // an unexpected shape degrades to today's code rather than corrupting.

  // Elements per 16B threadgroup store, and how many such stores cover the
  // 32 contiguous elements this thread owns.
  MLX_MTL_CONST short kWideElems = 16 / sizeof(T);
  MLX_MTL_CONST short kElemsPerThread = n_reads * pack_factor;
  MLX_MTL_CONST short kWideChunks = kElemsPerThread / kWideElems;
  MLX_MTL_CONST short kSrcBytesPerChunk = kWideElems / pack_factor;
  // Total packed source bytes this thread reads per k-iteration.
  MLX_MTL_CONST short kSrcBytes = n_reads * bytes_per_pack;

  // Shape preconditions that depend only on the instantiation.
  MLX_MTL_CONST bool kWidenShapeOk = (bits == 4) && (bytes_per_pack == 1) &&
      (kWideChunks >= 1) && (kWideChunks * kWideElems == kElemsPerThread) &&
      (kSrcBytesPerChunk * pack_factor == kWideElems) &&
      // Every chunk must fall inside a single scale group, so that one scale
      // covers it exactly as the scalar loop's `i` would.
      ((n_reads_per_scale % kSrcBytesPerChunk) == 0) &&
      (BCOLS_PACKED * BROWS >= tgp_size);
  // A single 16B device load covers this thread's whole source run.
  MLX_MTL_CONST bool kWideLoadShapeOk = kWidenShapeOk && (kSrcBytes == 16);

  struct alignas(16) WideChunk {
    T v[kWideElems];
  };
  // Sized by kSrcBytes rather than a literal 16 so the copy loop below stays
  // in bounds for every instantiation, including the 8-bit ones where
  // kSrcBytes is 32 and the wide-load path is statically disabled.
  struct alignas(16) WideSrc {
    uint8_t b[kSrcBytes];
  };

  // Byte offset of this thread's threadgroup destination, relative to the Ws
  // base -- the same expression the constructor used for `dst`. Ws itself is
  // 16B aligned by construction (see the NAXWsChunk16 backing store), so this
  // offset alone decides whether a 16B store is legal.
  short dst_byte_off() const {
    return short((bi * dst_ld + bj * pack_factor) * sizeof(T));
  }

  // Byte offset of this thread's device source, relative to the per-expert
  // tile base the constructor was handed. The host certifies that base.
  int src_byte_off() const {
    return bi * src_ld * bytes_per_pack / pack_factor + bj * bytes_per_pack;
  }

  // Exact thread-space twin of dequantize<T, bits> for bits == 4.
  static void dequantize_pair(uint8_t w, T scale, thread T* out) {
    out[0] = scale * Dequantize<4, T>{}(w);
    out[1] = scale * Dequantize<4, T>{}(w >> 4);
  }

  template <bool wide_store, bool wide_load>
  void load_unsafe_wide() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    const bool store_ok =
        wide_store && kWidenShapeOk && ((dst_byte_off() & 15) == 0);
    const bool load_ok =
        wide_load && kWideLoadShapeOk && ((src_byte_off() & 15) == 0);

    // Nothing widened for this thread: run the untouched scalar path.
    if (!store_ok && !load_ok) {
      load_unsafe();
      return;
    }

    uint8_t sb[kSrcBytes];
    // if constexpr: on instantiations where a single 16B load cannot cover
    // this thread's source run the wide-load branch is not just unreachable,
    // it is not emitted at all.
    bool took_wide_load = false;
    if constexpr (kWideLoadShapeOk) {
      if (load_ok) {
        WideSrc packed = *((const device WideSrc*)src);
        STEEL_PRAGMA_UNROLL
        for (short b = 0; b < kSrcBytes; b++) {
          sb[b] = packed.b[b];
        }
        took_wide_load = true;
      }
    }
    if (!took_wide_load) {
      STEEL_PRAGMA_UNROLL
      for (short b = 0; b < kSrcBytes; b++) {
        sb[b] = src[b * bytes_per_pack];
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short c = 0; c < kWideChunks; c++) {
      const short e0 = c * kWideElems;
      const short k0 = c * kSrcBytesPerChunk;
      // Same scale the scalar loop selects for every k in this chunk:
      // i = k / n_reads_per_scale, constant across the chunk because
      // kSrcBytesPerChunk divides n_reads_per_scale.
      T scale =
          dequantize_scale<T, group_size>(scales[k0 / n_reads_per_scale]);

      WideChunk out;
      STEEL_PRAGMA_UNROLL
      for (short b = 0; b < kSrcBytesPerChunk; b++) {
        dequantize_pair(sb[k0 + b], scale, &out.v[b * pack_factor]);
      }

      if (store_ok) {
        *((threadgroup WideChunk*)(dst + e0)) = out;
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kWideElems; j++) {
          dst[e0 + j] = out.v[j];
        }
      }
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    int k = 0;
    for (int i = 0; i < n_steps_per_read; i++) {
      T scale = dequantize_scale<T, group_size>(scales[i]);
      for (int j = 0; j < n_reads_per_scale; j++) {
        dequantize<T, bits>(
            src[k * bytes_per_pack], scale, dst + k * pack_factor);
        k++;
      }
    }
  }

  void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      scales += n_groups;
    } else {
      scales += n_groups * group_stride;
    }
  }
};

using namespace mlx::steel;

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    typename Wtype = bfloat>
METAL_FUNC void fp_qmm_t_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    threadgroup Wtype* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();

  constexpr int BK_padded = (BK + 16 / sizeof(Wtype));

  // Instantiate Loader
  using loader_w_t = QuantizedBlockLoader<
      Wtype,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  // Set the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the weight loader
  loader_w_t loader_w(wl, scales, K, Ws, simd_gid, simd_lid);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;

  const short sgp_sm = min(int(SM), M - (y_row + tm));
  const bool is_unaligned_sm = (sgp_sm != SM);

  const short sgp_sn = aligned_N ? SN : min(int(SN), N - (y_col + tn));

  const short tgp_bn = aligned_N ? BN : min(BN, int(N - (y_col)));
  const bool is_unaligned_bn = aligned_N ? false : (tgp_bn != BN);

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (kAlignedN.value) {
          loader_w.load_unsafe();
        } else {
          loader_w.load_safe(short2(BK, tgp_bn));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          NAXTile<T, TM, TK> Atile;
          NAXTile<Wtype, TN, TK> Btile;

          volatile int compiler_barrier;

          if constexpr (kAlignedM.value) {
            Atile.load(x + kk1, K);
          } else {
            Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
          }

          Btile.template load<Wtype, BK_padded, 1>(Ws + tn * BK_padded + kk1);

          tile_matmad_nax(
              Dtile,
              Atile,
              metal::bool_constant<transpose_a>{},
              Btile,
              metal::bool_constant<transpose_b>{});

          (void)compiler_barrier;
        }

        x += BK;
        loader_w.next();
      }

      // Store results to device memory
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if constexpr (kAlignedM.value && kAlignedN.value) {
        Dtile.store(y + tm * N + tn, N);
      } else if (kAlignedM.value && sgp_sn == SN) {
        Dtile.store(y + tm * N + tn, N);
      } else {
        Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
      }
    });
  });
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    typename Wtype = bfloat>
METAL_FUNC void fp_qmm_n_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;
  (void)M;

  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();

  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      0,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  // Set the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the x loader and mma operation
  // const short num_els = min(BM, M - y_row);
  // const short num_outs = min(BN, N - y_col);
  loader_w_t loader_w(wl, scales, K, Ws, simd_gid, simd_lid);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  const short ldb_tgp = BN_padded;

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = false;

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      NAXTile<T, TM, TK> Atile;
      NAXTile<Wtype, TK, TN> Btile;

      volatile int compiler_barrier;

      Atile.load(x + kk1, K);
      Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * ldb_tgp);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<transpose_a>{},
          Btile,
          metal::bool_constant<transpose_b>{});

      (void)compiler_barrier;
    }

    x += BK;
    loader_w.next();
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);

  Dtile.store(y + tm * N + tn, N);
}

template <typename T, typename S>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device S*& scales,
    device T*& y,
    int output_stride,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx = tid.z;
  uint32_t w_idx = tid.z;
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
  }
  y += tid.z * output_stride;
}

template <typename T, typename S>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device S*& scales,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T*& y,
    int output_stride,
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx;
  uint32_t w_idx;
  if (batch_ndims == 1) {
    x_idx = lhs_indices[tid.z * lhs_strides[0]];
    w_idx = rhs_indices[tid.z * rhs_strides[0]];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        tid.z, batch_shape, lhs_strides, rhs_strides, batch_ndims);
    x_idx = lhs_indices[idx.x];
    w_idx = rhs_indices[idx.y];
  }
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
  }
  y += tid.z * output_stride;
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const bool batched,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    typename Wtype = bfloat>
[[kernel]] void fp_qmm_t_nax(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(Wtype));

  threadgroup Wtype Ws[BN * BK_padded];

  if (batched) {
    adjust_matrix_offsets(
        x,
        w,
        scales,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        tid);
  }
  fp_qmm_t_impl<T, group_size, bits, aligned_N, BM, BK, BN, WM, WN, Wtype>(
      w, scales, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool batched,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    typename Wtype = bfloat>
[[kernel]] void fp_qmm_n_nax(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  if (batched) {
    adjust_matrix_offsets(
        x,
        w,
        scales,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        tid);
  }

  fp_qmm_n_impl<T, group_size, bits, BM, BK, BN, WM, WN, Wtype>(
      w, scales, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    typename Wtype = bfloat>
[[kernel]] void fp_gather_qmm_t_nax(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(Wtype));

  threadgroup Wtype Ws[BN * BK_padded];

  adjust_matrix_offsets(
      x,
      w,
      scales,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      tid);
  fp_qmm_t_impl<T, group_size, bits, aligned_N, BM, BK, BN, WM, WN, Wtype>(
      w, scales, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    typename Wtype = bfloat>
[[kernel]] void fp_gather_qmm_n_nax(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  adjust_matrix_offsets(
      x,
      w,
      scales,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      tid);
  fp_qmm_n_impl<T, group_size, bits, BM, BK, BN, WM, WN, Wtype>(
      w, scales, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    int group_size,
    const int bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose,
    typename Wtype = bfloat>
[[kernel]] void fp_gather_qmm_rhs_nax(
    const device T* x,
    const device uint32_t* w,
    const device uint8_t* scales,
    const device uint32_t* indices,
    device T* y,
    const constant int& M,
    const constant int& N,
    const constant int& K,
    // Magnitude dial for DARKBLOOM_PREFILL_GATHER_RUNSKIP, 1..100. A RUNTIME
    // scalar, deliberately NOT a function constant: it must never participate
    // in the pipeline specialization key, so one variant is compiled per
    // process and no JIT compile can land inside a timed forward.
    const constant int& run_skip_pct,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = (BK + 16 / sizeof(Wtype));
  constexpr int BN_padded = (BN + 16 / sizeof(Wtype));

  using loader_w_t = QuantizedBlockLoader<
      Wtype,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  // 16B-aligned backing store for Ws: identical element count, identical
  // contents, identical relative addresses. DARKBLOOM_STAGE_WIDEST needs Ws
  // itself 16B-aligned so that every thread's dst = Ws + bi*BK_padded +
  // bj*pack_factor is too (BK_padded*sizeof == 144 and bj*pack_factor*sizeof
  // in {0, 64} are all multiples of 16).
  constexpr int kWsElems = transpose ? BN * BK_padded : BK * BN_padded;
  constexpr int kWsPerChunk = 16 / sizeof(Wtype);
  threadgroup NAXWsChunk16<Wtype>
      Ws_storage[(kWsElems + kWsPerChunk - 1) / kWsPerChunk];
  threadgroup Wtype* Ws = (threadgroup Wtype*)Ws_storage;

  // Compute the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int N_w = N * bytes_per_pack / pack_factor;
  const int N_g = N / group_size;
  const int K_it = K / BK;
  const size_t stride_w = transpose ? N * K_w : K * N_w;
  const size_t stride_s = transpose ? N * K_g : K * N_g;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds
  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  // Calculate the final tiles in the case that K is not aligned
  const int k_remain = K - K_it * BK;
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  // Move x and output to the correct block
  auto wl = (const device uint8_t*)w;
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
  scales += transpose ? y_col_long * K_g : y_col / group_size;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const short sgp_sm = align_M ? SM : min(int(SM), max(0, (M - (y_row + tm))));
  const short sgp_sn = align_N ? SN : min(int(SN), max(0, (N - (y_col + tn))));

  const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);
  const bool is_unaligned_bn = align_N ? false : (tgp_bn != BN);

  constexpr short BR = transpose ? TN : TK;
  constexpr short BC = transpose ? TK : TN;

  using AccumType = float;

  // Do as many matmuls as necessary
  uint32_t index;
  short offset;
  uint32_t index_next = indices[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (indices[y_row + n] != index) {
        offset_next = n;
        index_next = indices[y_row + n];
        break;
      }
    }
    // DARKBLOOM_STAGE_RUNBAR: mem_none is an execution-only sync. Everything
    // between it and the next mem_threadgroup barrier (Dtile.clear and the
    // loader constructor) is register work, so that later barrier already
    // orders every threadgroup access on both sides of this point.
    if (!stage_runbar) {
      threadgroup_barrier(mem_flags::mem_none);
    }

    // --- DARKBLOOM_PREFILL_GATHER_RUNSKIP (function constant 203) ---
    // Rows this simdgroup owns are [tm, tm + sgp_sm) inside the tile; this run
    // covers tile rows [offset, offset_next). The store below writes exactly
    // rows [m_lo_lim, m_hi_lim) of this simdgroup's band, so when that range is
    // empty every matmul performed for this run is dead work for this
    // simdgroup -- store_slice's per-element guard already discards all of it.
    //
    // Exactness: this elides only arithmetic whose result is provably never
    // written. Elements that ARE stored keep an identical accumulation order
    // over an identical K sequence, so results are bit-for-bit unchanged. This
    // holds for ANY value of run_skip_pct: the dial only chooses how many tiles
    // take the elision, never what any surviving element computes.
    //
    // Magnitude dial: tile_enabled selects a deterministic subset of output
    // row-tiles by tid.y, which is a grid coordinate -- it depends only on the
    // dispatch geometry, never on token content or expert ids, so the elided
    // set is identical for every prompt. Monotone in run_skip_pct by
    // construction: the set {r : r*100 < pct} grows with pct, and pct>=100
    // enables every tile.
    //
    // Barrier uniformity: offset/offset_next/tgp_bm are threadgroup-uniform, so
    // the enclosing while-loop trip count is identical for every thread. tm and
    // sgp_sm depend only on simd_group_id, and tile_enabled only on tid.y and a
    // constant scalar, so sg_active is simdgroup-uniform. It gates ONLY
    // per-simdgroup register work (Atile/Btile + tile_matmad) and the store;
    // every threadgroup_barrier and the threadgroup-wide weight loader
    // (load_unsafe / load_safe / next) stay unconditional below.
    const short m_lo_lim = min(int(sgp_sm), max(0, offset - tm));
    const short m_hi_lim = min(int(sgp_sm), max(0, offset_next - tm));
    const bool tile_enabled =
        (run_skip_pct >= 100) ||
        (int((tid.y * 61u) % 100u) < run_skip_pct);
    const bool sg_active =
        !gather_run_skip || !tile_enabled || (m_lo_lim < m_hi_lim);

    // Prepare threadgroup mma operation
    NAXTile<AccumType, TM, TN> Dtile;
    Dtile.clear();

    const device T* xn = x + tm * K;

    // Prepare threadgroup loading operations
    thread loader_w_t loader_w(
        wl + index * stride_w,
        scales + index * stride_s,
        transpose ? K : N,
        Ws,
        simd_group_id,
        simd_lane_id);

    dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(align_N || !is_unaligned_bn, [&](auto kAlignedN) {
        for (int k = 0; k < K_it; k++) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if constexpr (kAlignedN.value) {
            // Same bytes, same addresses, same nibble decode, same scale
            // mapping -- only the access width changes. See load_unsafe_wide.
            if (stage_widest) {
              if (stage_wideld) {
                loader_w.template load_unsafe_wide<true, true>();
              } else {
                loader_w.template load_unsafe_wide<true, false>();
              }
            } else if (stage_wideld) {
              loader_w.template load_unsafe_wide<false, true>();
            } else {
              loader_w.load_unsafe();
            }
          } else {
            loader_w.load_safe(
                transpose ? short2(BK, tgp_bn) : short2(tgp_bn, BK));
          }

          threadgroup_barrier(mem_flags::mem_threadgroup);

          if (sg_active) {
            STEEL_PRAGMA_NO_UNROLL
            for (int kk1 = 0; kk1 < BK; kk1 += SK) {
              NAXTile<T, TM, TK> Atile;
              NAXTile<Wtype, BR, BC> Btile;

              volatile int compiler_barrier;

              if constexpr (kAlignedM.value) {
                Atile.load(xn + kk1, K);
              } else {
                Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
              }

              if constexpr (transpose) {
                Btile.template load<Wtype, BK_padded, 1>(
                    Ws + tn * BK_padded + kk1);
              } else {
                Btile.template load<Wtype, BN_padded, 1>(
                    Ws + tn + kk1 * BN_padded);
              }

              tile_matmad_nax(
                  Dtile,
                  Atile,
                  metal::bool_constant<false>{},
                  Btile,
                  metal::bool_constant<transpose>{});

              // DARKBLOOM_STAGE_NOVOL: the volatile read forces a stack load
              // and an optimization barrier on every inner step, which blocks
              // software-pipelining the Atile load against the MMA. Dropping
              // it touches no arithmetic and no memory the kernel reads or
              // writes.
              if (!stage_novol) {
                (void)compiler_barrier;
              }
            }
          }

          xn += BK;
          loader_w.next();
        }

        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          loader_w.load_safe(tile_w);
          threadgroup_barrier(mem_flags::mem_threadgroup);

          if (sg_active) {
            STEEL_PRAGMA_NO_UNROLL
            for (int kk1 = 0; kk1 < BK; kk1 += SK) {
              NAXTile<T, TM, TK> Atile;
              NAXTile<Wtype, BR, BC> Btile;

              volatile int compiler_barrier;

              const short psk = min(int(SK), max(0, (BK - kk1)));
              Atile.load_safe(xn + kk1, K, short2(psk, sgp_sm));

              if constexpr (transpose) {
                Btile.template load<Wtype, BK_padded, 1>(
                    Ws + tn * BK_padded + kk1);
              } else {
                Btile.template load<Wtype, BN_padded, 1>(
                    Ws + tn + kk1 * BN_padded);
              }

              tile_matmad_nax(
                  Dtile,
                  Atile,
                  metal::bool_constant<false>{},
                  Btile,
                  metal::bool_constant<transpose>{});

              if (!stage_novol) {
                (void)compiler_barrier;
              }
            }
          }
        }

        // DARKBLOOM_STAGE_RUNBAR: this barrier fences nothing. The only
        // threadgroup array is Ws; the code between here and the next Ws
        // access is Dtile.store*/store_slice, which reads registers and
        // writes device memory. The write-after-read hazard against the next
        // run's Ws stores is already covered by the mem_threadgroup barrier
        // that immediately precedes those stores.
        if (!stage_runbar) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Store results to device memory. A skipped run stored nothing anyway
        // (m_lo_lim >= m_hi_lim makes every store_slice range empty), so this
        // guard removes work without changing any written element.
        if (sg_active) {
          if constexpr (kAlignedN.value) {
            if (m_lo_lim == 0 && m_hi_lim == SM) {
              Dtile.store(y + tm * N + tn, N);
            } else {
              Dtile.store_slice(
                  y + tm * N + tn,
                  N,
                  short2(0, m_lo_lim),
                  short2(SN, m_hi_lim));
            }
          } else {
            Dtile.store_slice(
                y + tm * N + tn,
                N,
                short2(0, m_lo_lim),
                short2(sgp_sn, m_hi_lim));
          }
        }
      });
    });
  }
}
