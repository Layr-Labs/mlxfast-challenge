namespace mlx::core::metal {

const char* fp_quantized_nax() {
  return R"preamble(
// Copyright © 2025 Apple Inc.

// Auto generated source for mlx/backend/metal/kernels/fp_quantized_nax.h

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/fp4.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/fp4.h"

struct fp4_e2m1 {
  fp4_e2m1(float x) {
    if (metal::isnan(x)) {
      bits = 0x7;
      return;
    }

    const uint8_t sign_bit = (metal::signbit(x)) ? 0x8 : 0x0;
    x = metal::abs(x);

    if (x > 5.0f) {
      bits = 0x7;
    } else if (x >= 3.5f) {
      bits = 0x6;
    } else if (x > 2.5f) {
      bits = 0x5;
    } else if (x >= 1.75f) {
      bits = 0x4;
    } else if (x > 1.25f) {
      bits = 0x3;
    } else if (x >= 0.75f) {
      bits = 0x2;
    } else if (x > 0.25f) {
      bits = 0x1;
    } else {
      bits = 0x0;
    }
    bits |= sign_bit;
  }

  operator float16_t() {
    half converted = as_type<half>(ushort((bits & 7) << 9));
    converted *= 16384.0;
    return bits & 8 ? -converted : converted;
  }

  operator float() {
    return static_cast<float>(this->operator float16_t());
  }

  operator bfloat16_t() {
    return static_cast<bfloat16_t>(this->operator float16_t());
  }

  uint8_t bits;
};

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/fp8.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/fp8.h"

struct fp8_e4m3 {
  template <typename T>
  fp8_e4m3(T f) {
    // From PyTorch
    // https://github.com/pytorch/pytorch/blob/e3643e1e0e923f0fc063dfab6f45c956d568919d/c10/util/Float8_e4m3fn.h#L148
    uint32_t fp8_max = 543 << 21;
    uint32_t denorm_mask = 141 << 23;
    uint32_t f_bits = as_type<uint32_t>(static_cast<float>(f));
    uint32_t sign = f_bits & 0x80000000;
    f_bits ^= sign;
    if (f_bits >= fp8_max) {
      // Default behavior saturates to min/max
      bits = 0x7E;
    } else {
      if (f_bits < (121 << 23)) {
        f_bits = as_type<uint32_t>(
            as_type<float>(f_bits) + as_type<float>(denorm_mask));
        bits = static_cast<uint8_t>(f_bits - denorm_mask);
      } else {
        // resulting mantissa is odd
        uint8_t mant_odd = (f_bits >> 20) & 1;
        f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
        f_bits += mant_odd;
        bits = static_cast<uint8_t>(f_bits >> 20);
      }
    }
    bits |= static_cast<uint8_t>(sign >> 24);
  }

  operator float16_t() {
    uint16_t v = (bits & 127) << 7;
    half converted = as_type<half>(v);
    converted *= 256.0;
    auto sign = bits & 128;
    return (sign ? -converted : converted);
  }

  operator bfloat16_t() {
    return static_cast<bfloat16_t>(this->operator float16_t());
  }

  operator float() {
    return static_cast<float>(this->operator float16_t());
  }

  uint8_t bits;
};

struct fp8_e8m0 {
  fp8_e8m0(float x) {
    if (!metal::isfinite(x)) {
      bits = 0xFF;
      return;
    }
    if (x < 0.0f) {
      bits = 0x00;
      return;
    }
    float le = metal::log2(x);
    int n = int(metal::round(le));

    n = n < -127 ? -127 : n;
    n = n > 127 ? 127 : n;
    bits = static_cast<uint8_t>(n + 127);
  }

  operator bfloat16_t() {
    uint16_t out = (bits == 0 ? 0x40 : (static_cast<uint16_t>(bits) << 7));
    return as_type<bfloat16_t>(out);
  }

  operator float() {
    uint32_t out = (bits == 0 ? 0x400000 : (static_cast<uint16_t>(bits) << 23));
    return as_type<float>(out);
  }

  uint8_t bits;
};

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/fp_quantized_nax.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/fp_quantized_nax.h"

#include <metal_simdgroup>
#include <metal_stdlib>


constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];
constant bool gather_run_skip [[function_constant(203)]];

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


static inline float fp4nv_scale_x16384(uint8_t s) {
  if (s < 16u) {
    return float(uint(s) << 5);
  }
  return float(*(thread fp8_e4m3*)(&s)) * 16384.0f;
}

static inline uint32_t fp4nv_pack4(const device uint8_t* p) {
  return as_type<uint32_t>(uchar4(*(const device packed_uchar4*)p));
}
static inline uint32_t fp4nv_pack4(const thread uint8_t* p) {
  return as_type<uint32_t>(uchar4(p[0], p[1], p[2], p[3]));
}

template <typename T>
static inline void fp4nv_decode8(uint32_t c, float scale, thread T* out) {
  const uint32_t xe = c & 0x0F0F0F0Fu;
  const uint32_t ge = xe | (xe << 3);
  const uint32_t yo = c & 0xF0F0F0F0u;
  const uint32_t go = yo | (yo >> 3);
  const float2 v0 =
      float2(as_type<half2>((ge << 9) & 0x8E008E00u)) * scale;
  const float2 v1 =
      float2(as_type<half2>((go << 8) & 0x8E008E00u)) * scale;
  const float2 v2 =
      float2(as_type<half2>((ge << 1) & 0x8E008E00u)) * scale;
  const float2 v3 = float2(as_type<half2>(go & 0x8E008E00u)) * scale;
  out[0] = T(v0.x);
  out[1] = T(v1.x);
  out[2] = T(v2.x);
  out[3] = T(v3.x);
  out[4] = T(v0.y);
  out[5] = T(v1.y);
  out[6] = T(v2.y);
  out[7] = T(v3.y);
}

template <typename T>
struct alignas(16) NAXWsChunk16 {
  T v[16 / sizeof(T)];
};

template <short role>
METAL_FUNC int laguna_co_original_row(int row) {
  if constexpr (role == 3) {
    return (row >> 6) * 32 + (row & 31);
  } else if constexpr (role == 7) {
    return row & 511;
  } else {
    return row;
  }
}

template <short role>
METAL_FUNC int laguna_co_plane(int row) {
  if constexpr (role == 2 || role == 6) {
    return 1;
  } else if constexpr (role == 3) {
    return (row & 63) >= 32;
  } else if constexpr (role == 7) {
    return row >= 512;
  } else {
    return 0;
  }
}

template <short role>
METAL_FUNC const device uint8_t* laguna_co_code(
    const device uint8_t* base, int expert, int row, int k_byte) {
  const int logical_row = laguna_co_original_row<role>(row);
  if constexpr (role == 4 || role == 8) {
    const size_t record = role == 4
        ? size_t(expert) * 2048 + logical_row
        : size_t(logical_row);
    return base + 128 + record * 272 + k_byte;
  } else {
    const int block = k_byte >> 8;
    const size_t record = (role <= 3 ? size_t(expert) * 512 : 0)
        + logical_row;
    return base + 128 + (record * 4 + block) * 544
        + laguna_co_plane<role>(row) * 272 + (k_byte & 255);
  }
}

template <short role>
METAL_FUNC uint8_t laguna_co_scale(
    const device uint8_t* base, int expert, int row, int group) {
  const int logical_row = laguna_co_original_row<role>(row);
  const int plane = laguna_co_plane<role>(row);
  if (expert == 0 && logical_row == 0 && group == 1) {
    return base[plane];
  }
  if constexpr (role == 4 || role == 8) {
    const size_t record = role == 4
        ? size_t(expert) * 2048 + logical_row
        : size_t(logical_row);
    return base[128 + record * 272 + 256 + (group >> 1)];
  } else {
    const int block = group >> 5;
    const size_t record = (role <= 3 ? size_t(expert) * 512 : 0)
        + logical_row;
    return base[128 + (record * 4 + block) * 544
        + plane * 272 + 256 + ((group & 31) >> 1)];
  }
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits,
    short pairwise_scale_layout = 0>
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
  const device uint8_t* pairwise_patch_base;
  const device uint8_t* pairwise_row_base;
  int pairwise_patch_slot;
  int logical_group;
  const device uint8_t* colayout_base;
  int colayout_expert;
  int colayout_row;
  int colayout_k_byte;

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
        scales(scales_ + bi * src_ld / group_size + group_id),
        pairwise_patch_base(scales_),
        pairwise_row_base(scales_),
        pairwise_patch_slot(-1),
        logical_group(group_id),
        colayout_base(src_),
        colayout_expert(0),
        colayout_row(0),
        colayout_k_byte(bj * bytes_per_pack) {
    static_assert(
        pairwise_scale_layout == 0 || reduction_dim == 1,
        "pairwise scales are only defined for reduction-dimension staging");
    static_assert(
        pairwise_scale_layout >= 0 && pairwise_scale_layout <= 16,
        "unknown Laguna pairwise scale layout");
  }


  short row_in_tile() const { return bi; }

  void set_pairwise_packed(
      const device uint8_t* patch_base,
      const device uint8_t* expert_base,
      int row,
      int patch_slot) {
    const int within_block = row & 63;
    const int logical_row = (row >> 6) * 32 + (within_block & 31);
    const int tile = logical_row >> 2;
    const int sub = ((logical_row & 3) << 1) + (within_block >> 5);
    pairwise_patch_base = patch_base;
    pairwise_row_base = expert_base + tile * 512 + sub * 16;
    pairwise_patch_slot = patch_slot;
  }

  void set_pairwise_rowmajor(
      const device uint8_t* patch_base,
      const device uint8_t* expert_base,
      int row,
      int patch_slot) {
    pairwise_patch_base = patch_base;
    pairwise_row_base =
        expert_base + size_t(row) * (src_ld / group_size / 2);
    pairwise_patch_slot = patch_slot;
  }

  void set_colayout(int expert, int row, int k_start = 0) {
    if constexpr (pairwise_scale_layout >= 3) {
      colayout_expert = expert;
      colayout_row = row + bi;
      colayout_k_byte = k_start / pack_factor + bj * bytes_per_pack;
      src = laguna_co_code<
          pairwise_scale_layout >= 9 ? pairwise_scale_layout - 8
                                     : pairwise_scale_layout>(
          colayout_base, colayout_expert, colayout_row, colayout_k_byte);
      logical_group = k_start / group_size + group_id;
    }
  }

  uint8_t scale_code(int step) const {
    if constexpr (pairwise_scale_layout >= 3) {
      return laguna_co_scale<
          pairwise_scale_layout >= 9 ? pairwise_scale_layout - 8
                                     : pairwise_scale_layout>(
          colayout_base, colayout_expert, colayout_row, logical_group + step);
    } else if constexpr (pairwise_scale_layout == 1) {
      const int group = logical_group + step;
      return group == 1 && pairwise_patch_slot >= 0
          ? pairwise_patch_base[pairwise_patch_slot]
          : pairwise_row_base[(group >> 5) * 128 + ((group & 31) >> 1)];
    } else if constexpr (pairwise_scale_layout == 2) {
      const int group = logical_group + step;
      return group == 1 && pairwise_patch_slot >= 0
          ? pairwise_patch_base[pairwise_patch_slot]
          : pairwise_row_base[group >> 1];
    } else {
      return scales[step];
    }
  }

  MLX_MTL_CONST bool fp4nv_fast = (bits == 4) && (group_size == 16) &&
      (bytes_per_pack == 1) && (n_reads_per_scale >= 4) &&
      ((n_reads_per_scale % 4) == 0);

  void stage() const {
    if constexpr (fp4nv_fast) {
      float pair_scales[n_steps_per_read];
      if constexpr (pairwise_scale_layout != 0) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < n_steps_per_read; ++i) {
          const int group = logical_group + i;
          const bool patched_group1 = group == 1 &&
              (pairwise_patch_slot >= 0 || pairwise_scale_layout >= 3);
          const bool shares_previous =
              i > 0 && ((group >> 1) == ((group - 1) >> 1)) &&
              !patched_group1;
          pair_scales[i] = shares_previous
              ? pair_scales[i - 1]
              : fp4nv_scale_x16384(scale_code(i));
        }
      }
      int k = 0;
      for (int i = 0; i < n_steps_per_read; i++) {
        float scale;
        if constexpr (pairwise_scale_layout != 0) {
          scale = pair_scales[i];
        } else {
          scale = fp4nv_scale_x16384(scale_code(i));
        }
        for (int j = 0; j < n_reads_per_scale / 4; j++) {
          T vals[8];
          fp4nv_decode8<T>(fp4nv_pack4(src + k), scale, vals);
          for (int e = 0; e < 8; e++) {
            dst[k * pack_factor + e] = vals[e];
          }
          k += 4;
        }
      }
    } else {
      int k = 0;
      for (int i = 0; i < n_steps_per_read; i++) {
        T scale = dequantize_scale<T, group_size>(scale_code(i));
        for (int j = 0; j < n_reads_per_scale; j++) {
          dequantize<T, bits>(
              src[k * bytes_per_pack], scale, dst + k * pack_factor);
          k++;
        }
      }
    }
  }

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    stage();
  }


  MLX_MTL_CONST short kWideElems = 16 / sizeof(T);
  MLX_MTL_CONST short kElemsPerThread = n_reads * pack_factor;
  MLX_MTL_CONST short kWideChunks = kElemsPerThread / kWideElems;
  MLX_MTL_CONST short kSrcBytesPerChunk = kWideElems / pack_factor;
  MLX_MTL_CONST short kSrcBytes = n_reads * bytes_per_pack;

  MLX_MTL_CONST bool kWidenShapeOk = (bits == 4) && (bytes_per_pack == 1) &&
      (kWideChunks >= 1) && (kWideChunks * kWideElems == kElemsPerThread) &&
      (kSrcBytesPerChunk * pack_factor == kWideElems) &&
      ((n_reads_per_scale % kSrcBytesPerChunk) == 0) &&
      (BCOLS_PACKED * BROWS >= tgp_size);
  MLX_MTL_CONST bool kWideLoadShapeOk = kWidenShapeOk && (kSrcBytes == 16);
  MLX_MTL_CONST bool kWideLoad8ShapeOk = kWidenShapeOk && (kSrcBytes == 8);

  struct alignas(16) WideChunk {
    T v[kWideElems];
  };
  struct alignas(16) WideSrc {
    uint8_t b[kSrcBytes];
  };
  struct alignas(8) WideSrc8 {
    uint8_t b[kSrcBytes];
  };

  short dst_byte_off() const {
    return short((bi * dst_ld + bj * pack_factor) * sizeof(T));
  }

  int src_byte_off() const {
    return bi * src_ld * bytes_per_pack / pack_factor + bj * bytes_per_pack;
  }

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
    const bool load_ok = wide_load &&
        ((kWideLoadShapeOk && ((src_byte_off() & 15) == 0)) ||
         (kWideLoad8ShapeOk && ((src_byte_off() & 7) == 0)));

    if (!store_ok && !load_ok) {
      load_unsafe();
      return;
    }

    uint8_t sb[kSrcBytes];
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
    if constexpr (kWideLoad8ShapeOk) {
      if (load_ok) {
        WideSrc8 packed = *((const device WideSrc8*)src);
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

    float pair_scales[n_steps_per_read];
    if constexpr (fp4nv_fast && pairwise_scale_layout != 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_steps_per_read; ++i) {
        const int group = logical_group + i;
        const bool patched_group1 = group == 1 &&
            (pairwise_patch_slot >= 0 || pairwise_scale_layout >= 3);
        const bool shares_previous =
            i > 0 && ((group >> 1) == ((group - 1) >> 1)) &&
            !patched_group1;
        pair_scales[i] = shares_previous
            ? pair_scales[i - 1]
            : fp4nv_scale_x16384(scale_code(i));
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short c = 0; c < kWideChunks; c++) {
      const short e0 = c * kWideElems;
      const short k0 = c * kSrcBytesPerChunk;
      WideChunk out;
      if constexpr (fp4nv_fast && (kSrcBytesPerChunk % 4) == 0) {
        float scale;
        if constexpr (pairwise_scale_layout != 0) {
          scale = pair_scales[k0 / n_reads_per_scale];
        } else {
          scale = fp4nv_scale_x16384(scale_code(k0 / n_reads_per_scale));
        }
        STEEL_PRAGMA_UNROLL
        for (short b = 0; b < kSrcBytesPerChunk / 4; b++) {
          fp4nv_decode8<T>(fp4nv_pack4(sb + k0 + b * 4), scale, &out.v[b * 8]);
        }
      } else {
        T scale =
            dequantize_scale<T, group_size>(
                scale_code(k0 / n_reads_per_scale));
        STEEL_PRAGMA_UNROLL
        for (short b = 0; b < kSrcBytesPerChunk; b++) {
          dequantize_pair(sb[k0 + b], scale, &out.v[b * pack_factor]);
        }
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

    stage();
  }

  void next() {
    if constexpr (pairwise_scale_layout >= 3) {
      colayout_k_byte += tile_stride;
      logical_group += n_groups;
      src = laguna_co_code<
          pairwise_scale_layout >= 9 ? pairwise_scale_layout - 8
                                     : pairwise_scale_layout>(
          colayout_base, colayout_expert, colayout_row, colayout_k_byte);
      return;
    }
    src += tile_stride;
    if (reduction_dim == 1) {
      if constexpr (pairwise_scale_layout != 0) {
        logical_group += n_groups;
      } else {
        scales += n_groups;
      }
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
    typename Wtype = bfloat,
    const int fixed_K = 0,
    const int fixed_N = 0,
    const bool aligned_M = false,
    short storage_layout = 0>
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
  const int kernel_K = fixed_K > 0 ? fixed_K : K;
  const int kernel_N = fixed_N > 0 ? fixed_N : N;

  constexpr int BK_padded = (BK + 16 / sizeof(Wtype));

  using loader_w_t = QuantizedBlockLoader<
      Wtype,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits,
      storage_layout>;

  const int K_w = kernel_K * bytes_per_pack / pack_factor;
  const int K_g = kernel_K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(kernel_K);
  if constexpr (storage_layout == 0) {
    wl += y_col * K_w;
    scales += y_col * K_g;
  }
  y += y_row * static_cast<int64_t>(kernel_N) + y_col;

  loader_w_t loader_w(wl, scales, kernel_K, Ws, simd_gid, simd_lid);
  loader_w.set_colayout(0, y_col);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  static_assert(SK == 32, "dense NAX fragment width");
  static_assert(SK % 16 == 0, "dense NAX fragment divisibility");

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;

  const short sgp_sm =
      aligned_M ? SM : min(int(SM), M - (y_row + tm));
  const bool is_unaligned_sm = aligned_M ? false : (sgp_sm != SM);

  const short sgp_sn =
      aligned_N ? SN : min(int(SN), kernel_N - (y_col + tn));

  const short tgp_bn =
      aligned_N ? BN : min(BN, int(kernel_N - y_col));
  const bool is_unaligned_bn = aligned_N ? false : (tgp_bn != BN);

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * kernel_K;

  dispatch_bool(aligned_M || !is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {
      for (int k = 0; k < kernel_K; k += BK) {
        if (fixed_K == 0 || k > 0) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
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
            Atile.load(x + kk1, kernel_K);
          } else {
            Atile.load_safe(
                x + kk1, kernel_K, short2(SK, sgp_sm));
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

      if (fixed_K == 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      if constexpr (kAlignedM.value && kAlignedN.value) {
        Dtile.store(y + tm * kernel_N + tn, kernel_N);
      } else if (kAlignedM.value && sgp_sn == SN) {
        Dtile.store(y + tm * kernel_N + tn, kernel_N);
      } else {
        Dtile.store_safe(
            y + tm * kernel_N + tn,
            kernel_N,
            short2(sgp_sn, sgp_sm));
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

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

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
    typename Wtype = bfloat,
    short storage_layout = 0>
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
  fp_qmm_t_impl<
      T, group_size, bits, aligned_N, BM, BK, BN, WM, WN, Wtype,
      0, 0, false, storage_layout>(
      w, scales, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int fixed_K,
    const int fixed_N,
    const bool aligned_M,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    typename Wtype = bfloat,
    short storage_layout = 0>
[[kernel]] void fp_qmm_t_nax_static(
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
  (void)K;
  (void)N;
  (void)x_batch_ndims;
  (void)x_shape;
  (void)x_strides;
  (void)w_batch_ndims;
  (void)w_shape;
  (void)w_strides;
  (void)s_strides;
  static_assert(fixed_K > 0 && fixed_N > 0);

  constexpr int BK_padded = BK + 16 / sizeof(Wtype);
  threadgroup Wtype Ws[BN * BK_padded];

  fp_qmm_t_impl<
      T,
      group_size,
      bits,
      true,
      BM,
      BK,
      BN,
      WM,
      WN,
      Wtype,
      fixed_K,
      fixed_N,
      aligned_M,
      storage_layout>(
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

  constexpr int kWsElems = transpose ? BN * BK_padded : BK * BN_padded;
  constexpr int kWsPerChunk = 16 / sizeof(Wtype);
  threadgroup NAXWsChunk16<Wtype>
      Ws_storage[(kWsElems + kWsPerChunk - 1) / kWsPerChunk];
  threadgroup Wtype* Ws = (threadgroup Wtype*)Ws_storage;

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

  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  const int k_remain = K - K_it * BK;
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

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
    if (!stage_runbar) {
      threadgroup_barrier(mem_flags::mem_none);
    }

    const short m_lo_lim = min(int(sgp_sm), max(0, offset - tm));
    const short m_hi_lim = min(int(sgp_sm), max(0, offset_next - tm));
    const bool tile_enabled =
        (run_skip_pct >= 100) ||
        (int((tid.y * 61u) % 100u) < run_skip_pct);
    const bool sg_active =
        !gather_run_skip || !tile_enabled || (m_lo_lim < m_hi_lim);

    NAXTile<AccumType, TM, TN> Dtile;
    Dtile.clear();

    const device T* xn = x + tm * K;

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
            STEEL_PRAGMA_UNROLL
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
            STEEL_PRAGMA_UNROLL
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

        if (!stage_runbar) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

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

METAL_FUNC int laguna_sorted_lower_bound(
    const device uint32_t* indices,
    const int count,
    const uint32_t value) {
  int lo = 0;
  int hi = count;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (indices[mid] < value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
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
    const int fixed_K = 0,
    const int fixed_N = 0,
    typename Wtype = bfloat,
    int tg_expert_groups = 64,
    bool wide_store = false,
    bool wide_load = false,
    short pairwise_scale_layout = 0>
[[kernel]] void fp_gather_qmm_rhs_expert_nax(
    const device T* x,
    const device uint32_t* w,
    const device uint8_t* scales,
    const device uint32_t* indices,
    device T* y,
    const constant int& M,
    const constant int& N,
    const constant int& K,
    const constant int& run_skip_pct,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  (void)run_skip_pct;
  static_assert(transpose, "expert-aligned Laguna QMM requires NT weights");
  static_assert(group_size == 16, "expert-aligned Laguna QMM requires gs16");
  static_assert(bits == 4, "expert-aligned Laguna QMM requires NVFP4");

  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = BK + 16 / sizeof(Wtype);
  constexpr int BN_padded = BN + 16 / sizeof(Wtype);
  constexpr int expert_groups = tg_expert_groups;
  constexpr int experts = 256;
  const int kernel_K = fixed_K > 0 ? fixed_K : K;
  const int kernel_N = fixed_N > 0 ? fixed_N : N;
  static_assert(experts % expert_groups == 0);

  using loader_w_t = QuantizedBlockLoader<
      Wtype,
      BN,
      BK,
      BK_padded,
      true,
      WM * WN * SIMD_SIZE,
      group_size,
      bits,
      pairwise_scale_layout>;

  constexpr int kWsElems = BN * BK_padded;
  constexpr int kWsPerChunk = 16 / sizeof(Wtype);
  threadgroup NAXWsChunk16<Wtype>
      Ws_storage[(kWsElems + kWsPerChunk - 1) / kWsPerChunk];
  threadgroup Wtype* Ws = (threadgroup Wtype*)Ws_storage;
  threadgroup bfloat* gate_up_stage =
      (threadgroup bfloat*)Ws_storage;
#if defined(DARKBLOOM_BSEARCH_HOIST) && \
    !defined(DARKBLOOM_EXPERT_BOUNDS_SIDECAR)
  threadgroup int bounds[experts / expert_groups + 1];
#elif !defined(DARKBLOOM_EXPERT_BOUNDS_SIDECAR)
  threadgroup int bounds[2];
#endif

  const int K_w = kernel_K * bytes_per_pack / pack_factor;
  const int K_g = kernel_K / group_size;
  const int K_it = kernel_K / BK;
  const size_t stride_w = size_t(kernel_N) * K_w;
  const size_t stride_s = size_t(kernel_N) * K_g;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;
  if constexpr (pairwise_scale_layout < 9) {
    wl += size_t(y_col) * K_w;
  }
  const device uint8_t* scale_base = scales;
  if constexpr (pairwise_scale_layout == 0) {
    scale_base += size_t(y_col) * K_g;
  }

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

#ifdef DARKBLOOM_SWIGLU_REGLOCAL
  constexpr bool kSwigluRegLocal =
      (WN == 1) && (BN == 64) && ((BM / WM) == 16);
#endif // DARKBLOOM_SWIGLU_REGLOCAL

#if defined(DARKBLOOM_BSEARCH_HOIST) && \
    !defined(DARKBLOOM_EXPERT_BOUNDS_SIDECAR)
  // Hoist: all slot bounds once, one lower_bound per thread (same integers
  // as the per-slot lid==0 searches), one barrier instead of two per slot.
  for (int b = int(lid); b <= experts / expert_groups;
       b += WM * WN * SIMD_SIZE) {
    bounds[b] = laguna_sorted_lower_bound(
        indices,
        M,
        static_cast<uint32_t>(tid.y * (experts / expert_groups) + b));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
#endif

  for (int expert_slot = 0; expert_slot < experts / expert_groups;
       ++expert_slot) {
    const uint32_t expert =
        static_cast<uint32_t>(
            tid.y * (experts / expert_groups) + expert_slot);

#ifdef DARKBLOOM_EXPERT_BOUNDS_SIDECAR
    // N1b: the sorter already published exact global prefixes at logical
    // indices [0, 256]. One lane in every simdgroup loads the same pair and
    // broadcasts it locally. This removes both binary searches, the
    // threadgroup bounds allocation, and its synchronization while ensuring
    // every simdgroup begins with identical endpoints.
    int run_start = simd_lane_id == 0 ? int(indices[expert]) : 0;
    int run_end = simd_lane_id == 0 ? int(indices[expert + 1]) : 0;
    run_start = simd_broadcast(run_start, 0);
    run_end = simd_broadcast(run_end, 0);
#elif defined(DARKBLOOM_BSEARCH_HOIST)
    const int run_start = bounds[expert_slot];
    const int run_end = bounds[expert_slot + 1];
#else
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
      bounds[0] = laguna_sorted_lower_bound(indices, M, expert);
      bounds[1] = laguna_sorted_lower_bound(indices, M, expert + 1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int run_start = bounds[0];
    const int run_end = bounds[1];
#endif
    for (int chunk_start = run_start; chunk_start < run_end;
         chunk_start += BM) {
      const short chunk_rows =
          short(min(BM, run_end - chunk_start));
      const short sgp_sm =
          min(int(SM), max(0, int(chunk_rows) - int(tm)));
      const bool sg_active = sgp_sm > 0;

      NAXTile<float, TM, TN> Dtile;
      Dtile.clear();

      const device T* xn =
          x + size_t(chunk_start + tm) * kernel_K;
      const device uint8_t* loader_scales = pairwise_scale_layout != 0
          ? scales
          : scale_base + size_t(expert) * stride_s;
      thread loader_w_t loader_w(
          pairwise_scale_layout >= 9 ? wl : wl + size_t(expert) * stride_w,
          loader_scales,
          kernel_K,
          Ws,
          simd_group_id,
          simd_lane_id);
      if constexpr (pairwise_scale_layout >= 9) {
        loader_w.set_colayout(expert, y_col);
      } else if constexpr (pairwise_scale_layout == 1) {
        const int scale_row = y_col + int(loader_w.row_in_tile());
        const int patch_slot = expert == 0 && scale_row == 0
            ? 0
            : (expert == 0 && scale_row == 32 ? 1 : -1);
        const device uint8_t* packed_expert = scales + 128
            + size_t(expert) * (size_t(kernel_N) * K_g / 2);
        loader_w.set_pairwise_packed(
            scales, packed_expert, scale_row, patch_slot);
      } else if constexpr (pairwise_scale_layout == 2) {
        const int scale_row = y_col + int(loader_w.row_in_tile());
        const int patch_slot = expert == 0 && scale_row == 0 ? 0 : -1;
        const device uint8_t* packed_expert = scales + 128
            + size_t(expert) * (size_t(kernel_N) * K_g / 2);
        loader_w.set_pairwise_rowmajor(
            scales, packed_expert, scale_row, patch_slot);
      }

      for (int k = 0; k < K_it; ++k) {
        NAXTile<T, TM, TK> Atile[BK / SK];
        if (sg_active) {
          STEEL_PRAGMA_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            if (sgp_sm == SM) {
              Atile[kk1 / SK].load_contig(xn + kk1, kernel_K);
            } else {
              Atile[kk1 / SK].load_rows_contig(xn + kk1, kernel_K, sgp_sm);
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (wide_store || wide_load) {
          loader_w.template load_unsafe_wide<wide_store, wide_load>();
        } else {
          loader_w.load_unsafe();
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_active) {
          STEEL_PRAGMA_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            NAXTile<Wtype, TN, TK> Btile;

            Btile.template load_contig_tg<Wtype, BK_padded>(
                Ws + tn * BK_padded + kk1);

            tile_matmad_nax(
                Dtile,
                Atile[kk1 / SK],
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<true>{});

          }
        }

        xn += BK;
        loader_w.next();
      }

#ifndef DARKBLOOM_SWIGLU_REGLOCAL
      threadgroup_barrier(mem_flags::mem_threadgroup);
#endif // DARKBLOOM_SWIGLU_REGLOCAL
      const bool fuse_swiglu =
          kernel_N == 1024 && kernel_K == 2048;
      if (fuse_swiglu) {
#ifdef DARKBLOOM_SWIGLU_REGLOCAL
        if constexpr (kSwigluRegLocal) {
#pragma clang fp contract(off)
          constexpr int activated_cols = BN / 2;
          const short qid = short(simd_lane_id >> 2);
          const short fm = (qid & 4) | ((short(simd_lane_id) >> 1) & 3);
          const short fn = ((qid & 2) | (short(simd_lane_id) & 1)) * 4;
          STEEL_PRAGMA_UNROLL
          for (short jf = 0; jf < 2; ++jf) {
            STEEL_PRAGMA_UNROLL
            for (short ie = 0; ie < 2; ++ie) {
              const short row = fm + ie * 8;
              if (row < sgp_sm) {
                STEEL_PRAGMA_UNROLL
                for (short jj = 0; jj < 4; ++jj) {
                  const int col = jf * 16 + fn + jj;
                  const bfloat gate = static_cast<bfloat>(
                      Dtile.frag_at(0, jf)[ie * 4 + jj]);
                  const bfloat up = static_cast<bfloat>(
                      Dtile.frag_at(0, jf + 2)[ie * 4 + jj]);
                  const bfloat exp_abs = metal::exp(metal::abs(gate));
                  const bfloat denominator = bfloat(1) + exp_abs;
                  const bfloat z = bfloat(1) / denominator;
                  const bfloat sigmoid =
                      gate < bfloat(0) ? z : bfloat(1) - z;
                  const bfloat silu = bfloat(gate * sigmoid);
                  y[size_t(chunk_start + tm + row) * (kernel_N / 2) +
                    size_t(tid.x) * activated_cols + col] =
                      bfloat(silu * up);
                }
              }
            }
          }
        }
        if constexpr (!kSwigluRegLocal) {
#endif // DARKBLOOM_SWIGLU_REGLOCAL
        if (sg_active) {
          Dtile.template store<bfloat, BN, 1>(
              gate_up_stage + tm * BN + tn);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_active && (simd_group_id % WN) == 0) {
          constexpr int activated_cols = BN / 2;
          for (int linear = simd_lane_id;
               linear < int(sgp_sm) * activated_cols;
               linear += SIMD_SIZE) {
            const int row = linear / activated_cols;
            const int col = linear % activated_cols;
            const bfloat gate =
                gate_up_stage[(tm + row) * BN + col];
            const bfloat up =
                gate_up_stage[(tm + row) * BN + activated_cols + col];
            const bfloat exp_abs = metal::exp(metal::abs(gate));
            const bfloat denominator = bfloat(1) + exp_abs;
            const bfloat z = bfloat(1) / denominator;
            const bfloat sigmoid =
                gate < bfloat(0) ? z : bfloat(1) - z;
            const bfloat silu = bfloat(gate * sigmoid);
            y[size_t(chunk_start + tm + row) * (kernel_N / 2) +
              size_t(tid.x) * activated_cols + col] =
                bfloat(silu * up);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#ifdef DARKBLOOM_SWIGLU_REGLOCAL
        }
#endif // DARKBLOOM_SWIGLU_REGLOCAL
      } else if (sg_active) {
        device T* yn =
            y + size_t(chunk_start + tm) * kernel_N + y_col + tn;
        if (sgp_sm == SM) {
          Dtile.store(yn, kernel_N);
        } else {
          Dtile.store_slice(
              yn,
              kernel_N,
              short2(0, 0),
              short2(SN, sgp_sm));
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
)preamble";
}

} // namespace mlx::core::metal
