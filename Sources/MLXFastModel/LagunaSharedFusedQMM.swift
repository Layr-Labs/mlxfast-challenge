// Fused shared-expert [gate; up] NVFP4 QMM as ONE custom Metal dispatch.
//
// WHAT THIS IS. `MLXFast.metalKernel` wrapper around a byte-identical clone of
// the Metal source MLX itself JIT-compiles for `fp_qmm_t_nax`, instantiated at
// exactly the template arguments MLX's own dispatch picks for the shared
// expert's NVFP4 projections. The kernel is run once over the row-concatenated
// [gate; up] weight bank instead of twice over the two 512-row banks.
//
// WHY NOT `MLX.quantizedMM` ON THE DOUBLED-N BANK. It does not run the same
// kernel. `quantized.cpp`'s `qmm_nax()` carries a repo-local specialization
// switch, `use_static_laguna_shape`, keyed on
//   transpose && N % 64 == 0 && !batched && mode == "nvfp4" &&
//   bfloat16_t && group_size == 16 && bits == 4 && !biases &&
//   ((K == 2048 && N == 1024) || (K == 512 && N == 2048))
// -- and the fused bank is precisely K == 2048, N == 1024. So the fused call
// selects `fp_qmm_t_nax_static` (fixed_K/fixed_N/aligned_M baked in as
// compile-time constants, two threadgroup barriers dropped as provably dead)
// while the two separate 512-row banks (K == 2048, N == 512) miss the switch
// and select plain `fp_qmm_t_nax`. Two different compiled kernels. The tiling
// is NOT what differs: `qmm_nax()` hardcodes bm = bn = bk = 64, wm = wn = 2 for
// every shape, and `aligned_N` is true for both N = 512 and N = 1024.
//
// This wrapper therefore pins the NON-static instantiation, which is the one
// the split dispatches use.
//
// EXACTNESS ARGUMENT. Read `fp_qmm_t_impl` (fp_quantized_nax.cpp): with
// `aligned_N == true` and `batched == false` the only places N enters are
//   * `y_col = tid.x * BN`             -- which 64-column slab this threadgroup
//                                         owns; 512 is a multiple of BN = 64,
//                                         so fused slabs 0..7 are exactly the
//                                         gate bank's slabs and 8..15 exactly
//                                         the up bank's,
//   * `wl += y_col * K_w`, `scales += y_col * K_g` -- row addressing; the fused
//                                         bank's row r >= 512 holds byte for
//                                         byte the up bank's row r - 512,
//   * `y += ... + y_col` and the `kernel_N` passed to `Dtile.store` -- the
//                                         output row stride, which is what
//                                         makes the result the concatenation,
//   * `sgp_sn` / `tgp_bn` -- dead, folded to the compile-time BN/SN because
//                                         `aligned_N` is true.
// Nothing N-dependent enters the per-row K-loop: the loader's stride, its
// scale walk and its `n_reads` geometry are functions of BN/BK/tgp_size only;
// `Atile` depends on M and K; `Btile` reads threadgroup memory; `tile_matmad_nax`
// accumulates into a per-thread register tile with no cross-N reduction; there
// is no split-K. Every output element's accumulation order is therefore the
// same one the stock kernel produces for its own 512-row bank.
//
// SOURCE PROVENANCE. `lagunaSharedFusedQMMHeader` below is the verbatim
// concatenation of `metal::gemm_nax()`, `metal::quantized_utils()` and the
// prefix of `metal::fp_quantized_nax()` through the end of `fp_qmm_t_impl` --
// the same three strings, in the same order, that `get_qmm_nax_kernel()`
// concatenates (jit_kernels.cpp:1233-1255). MLX prepends `metal::utils()` to a
// custom kernel's source itself (custom_kernel.cpp), so the preamble the Metal
// compiler sees here is identical to the stock one. Nothing is renamed:
// a custom kernel gets its own `MTLLibrary`, built from `metal::utils() +
// source` and nothing else, so there is no symbol to clash with -- and keeping
// the text byte-identical is a strictly stronger guarantee than a rename.

import Foundation
import MLX

/// Metal preamble for ``lagunaSharedFusedQMMKernel``. See the file comment for
/// provenance; do not edit by hand.
internal let lagunaSharedFusedQMMHeader = #"""
// Copyright © 2025 Apple Inc.

// Auto generated source for mlx/backend/metal/kernels/steel/gemm/gemm_nax.h

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/defines.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/defines.h"
// Copyright © 2024 Apple Inc.


#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define STEEL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")

///////////////////////////////////////////////////////////////////////////////
// Contents from "/private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.5.188.0.9CLJ6d/Metal.xctoolchain/usr/metal/32023/lib/clang/32023.883/include/metal/__exec/units.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "/private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.5.188.0.9CLJ6d/Metal.xctoolchain/usr/metal/32023/lib/clang/32023.883/include/metal/__exec/units.h"

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/utils/type_traits.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/utils/type_traits.h"
// Copyright © 2024 Apple Inc.


#include <metal_stdlib>

#pragma METAL internals : enable

namespace metal {

template <typename T>
struct is_empty : metal::bool_constant<__is_empty(T)> {};

#ifdef __cpp_variable_templates
template <typename T>
constexpr constant bool is_empty_v = is_empty<T>::value;
#endif

template <typename... Ts>
struct make_void {
  typedef void type;
};

template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

template <class T>
struct is_static : metal::bool_constant<is_empty<remove_cv_t<T>>::value> {};

template <typename T>
struct pointer_element {};

template <typename T>
struct pointer_element<thread T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<device T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<constant T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<threadgroup T*> {
  using type = remove_cv_t<T>;
};

template <typename T>
using pointer_element_t = typename pointer_element<remove_cv_t<T>>::type;

} // namespace metal

#pragma METAL internals : disable

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
// Copyright © 2024 Apple Inc.


#include <metal_stdlib>

#pragma METAL internals : enable

namespace mlx {
namespace steel {

///////////////////////////////////////////////////////////////////////////////
// Integral constant with casting
///////////////////////////////////////////////////////////////////////////////

template <typename T, T v>
struct integral_constant {
  static constexpr constant T value = v;
  using value_type = T;
  using type = integral_constant;

  METAL_FUNC constexpr operator value_type() const noexcept {
    return value;
  }
};

template <bool B>
using bool_constant = integral_constant<bool, B>;
using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template <class T>
struct is_integral : bool_constant<metal::is_integral<T>::value> {};

template <class T, T v>
struct is_integral<integral_constant<T, v>>
    : bool_constant<metal::is_integral<T>::value> {};

template <typename T>
constexpr constant bool is_integral_v = is_integral<T>::value;

template <int val>
using Int = integral_constant<int, val>;

///////////////////////////////////////////////////////////////////////////////
// Binary Operators on Integral constants
///////////////////////////////////////////////////////////////////////////////

#define integral_const_binop(__op__, __operator__)          \
  template <typename T, T tv, typename U, U uv>             \
  METAL_FUNC constexpr auto __operator__(                   \
      integral_constant<T, tv>, integral_constant<U, uv>) { \
    constexpr auto res = tv __op__ uv;                      \
    return integral_constant<decltype(res), res>{};         \
  }

integral_const_binop(+, operator+);
integral_const_binop(-, operator-);
integral_const_binop(*, operator*);
integral_const_binop(/, operator/);

integral_const_binop(==, operator==);
integral_const_binop(!=, operator!=);
integral_const_binop(<, operator<);
integral_const_binop(>, operator>);
integral_const_binop(<=, operator<=);
integral_const_binop(>=, operator>=);

integral_const_binop(&&, operator&&);
integral_const_binop(||, operator||);

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator||(true_type, T) {
  return true_type{};
}
template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator||(T, true_type) {
  return true_type{};
}

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator&&(false_type, T) {
  return false_type{};
}

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator&&(T, false_type) {
  return false_type{};
}

// Dispatch utilities
template <typename F>
void dispatch_bool(bool v, F f) {
  if (v) {
    f(true_type{});
  } else {
    f(false_type{});
  }
}

template <int start, int stop, int step, typename F>
constexpr void const_for_loop(F f) {
  if constexpr (start < stop) {
    constexpr auto idx = Int<start>{};
    f(idx);
    const_for_loop<start + step, stop, step, F>(f);
  }
}

#undef integral_const_binop

///////////////////////////////////////////////////////////////////////////////
// Reduction operators
///////////////////////////////////////////////////////////////////////////////

template <typename T>
METAL_FUNC constexpr T sum(T x) {
  return x;
}

template <typename T, typename... Us>
METAL_FUNC constexpr auto sum(T x, Us... us) {
  return x + sum(us...);
}

} // namespace steel
} // namespace mlx

#pragma METAL internals : disable

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/gemm/nax.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/gemm/nax.h"
// Copyright © 2025 Apple Inc.


#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>


#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// MMA helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

///////////////////////////////////////////////////////////////////////////////
// NAX Steel with new tiles
///////////////////////////////////////////////////////////////////////////////

struct BaseNAXFrag {
  STEEL_CONST short kFragRows = 16;
  STEEL_CONST short kFragCols = 16;

  STEEL_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32;

  STEEL_CONST short kElemRows = 2;
  STEEL_CONST short kElemCols = 4;

  STEEL_CONST short kElemRowsJump = 8;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MMAFrag shape is not consistent with MMAFrag size");

  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static short2 get_coord() {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;
    return short2{fn, fm};
  }

  METAL_FUNC static short2 get_coord(short idx) {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3)) + (idx >> 2) * 8;
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4 + idx % 4;
    return short2{fn, fm};
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + c + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + j) * str_y]);
        }
      }
    }
  }


  // Contiguous vec4 `load` for 8B-aligned Int<1> call sites (certified per
  // site): identical elements, identical slots, pure access-width change.
  template <
      typename T,
      typename StrX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_contig(
      thread dtype_frag_t<T>& dst,
      const device T* src,
      StrX str_x,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      static_assert(kElemCols == 4, "load_contig assumes 4-element runs");
      const metal::vec<T, 4> v =
          *reinterpret_cast<const device metal::vec<T, 4>*>(
              &src[r * str_x + c]);
      dst[i * kElemCols + 0] = v.x;
      dst[i * kElemCols + 1] = v.y;
      dst[i * kElemCols + 2] = v.z;
      dst[i * kElemCols + 3] = v.w;
    }
  }

  // Row-limited device vec4 twin (certified call sites only): identical
  // row predicate and zero-fills as load_rows, vec4 on whole runs.
  template <
      typename T,
      typename StrX,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_rows_contig(
      thread dtype_frag_t<T>& dst,
      const device T* src,
      StrX str_x,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x;
    auto lx = lim_x - sc.y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if (r < lx) {
        const metal::vec<T, 4> v =
            *reinterpret_cast<const device metal::vec<T, 4>*>(
                &src[r * str_x + c]);
        dst[i * kElemCols + 0] = v.x;
        dst[i * kElemCols + 1] = v.y;
        dst[i * kElemCols + 2] = v.z;
        dst[i * kElemCols + 3] = v.w;
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  // Threadgroup twin of load_contig; same certification contract.
  template <
      typename T,
      typename StrX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_contig_tg(
      thread dtype_frag_t<T>& dst,
      const threadgroup T* src,
      StrX str_x,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      static_assert(kElemCols == 4, "load_contig_tg assumes 4-element runs");
      const metal::vec<T, 4> v =
          *reinterpret_cast<const threadgroup metal::vec<T, 4>*>(
              &src[r * str_x + c]);
      dst[i * kElemCols + 0] = v.x;
      dst[i * kElemCols + 1] = v.y;
      dst[i * kElemCols + 2] = v.z;
      dst[i * kElemCols + 3] = v.w;
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_rows(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + (c + j)]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] =
                static_cast<T>(src[r * str_x + (c + j) * str_y]);
          }
        }

      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_safe(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((r < lx) && ((c + j) < ly)) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + j) * str_y]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + (c + j) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_rows(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + (c + j) * str_y] =
                static_cast<U>(src[i * kElemCols + j]);
          }
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_safe(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if (r < lx && (c + j) < ly) {
          dst[r * str_x + (c + j) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename StartX,
      typename StopX,
      typename StartY,
      typename StopY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_slice(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      StartX start_x,
      StopX stop_x,
      StartY start_y,
      StopY stop_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();

    const_for_loop<0, kElemRows, 1>([&](auto idx_row) {
      const auto r = off_x + idx_row * Int<kElemRowsJump>{};
      if (r >= stop_x - sc.y || r < start_x - sc.y) {
        return;
      }

      const_for_loop<0, kElemCols, 1>([&](auto idx_col) {
        const auto c = off_y + idx_col;
        if (c >= stop_y - sc.x || c < start_y - sc.x) {
          return;
        }

        const auto src_idx = idx_row * Int<kElemCols>{} + idx_col;
        dst[(r + sc.y) * str_x + (c + sc.x) * str_y] =
            static_cast<U>(src[src_idx]);
      });
    });
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_reduce(
      thread const dtype_frag_t<T>& inp_vals,
      thread T* reduced_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      T thr_reduce = Op::apply(
          Op::apply(inp_vals[i * kElemCols + 0], inp_vals[i * kElemCols + 1]),
          Op::apply(inp_vals[i * kElemCols + 2], inp_vals[i * kElemCols + 3]));

      T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
      qgr_reduce = Op::apply(thr_reduce, qgr_reduce);

      T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
      sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);

      reduced_vals[i] = Op::apply(reduced_vals[i], sgr_reduce);
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_bin_op(
      thread dtype_frag_t<T>& inp_vals,
      thread T* row_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] =
            Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }

  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cn0,
      thread dtype_frag_t<CType>& Cn1,
      const thread dtype_frag_t<AType>& A,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& Bn0,
      const thread dtype_frag_t<BType>& Bn1,
      metal::bool_constant<transpose_b>) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    // Create matmul op
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    // Create matmul operands in registers
    auto ct_a =
        gemm_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b =
        gemm_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();

    // Create matmul output in register
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        decltype(ct_a),
        decltype(ct_b),
        CType>();

    // Load A in to left operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_a[i] = A[i];
    }

    // Load B into right operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_b[i] = Bn0[i];
      ct_b[kElemsPerFrag + i] = Bn1[i];
    }

    // Load C into output registers (op handles accumulation)
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_c[i] = Cn0[i];
      ct_c[kElemsPerFrag + i] = Cn1[i];
    }

    // Do matmul
    gemm_op.run(ct_a, ct_b, ct_c);

    // Copy out results
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      Cn0[i] = ct_c[i];
      Cn1[i] = ct_c[kElemsPerFrag + i];
    }
  }

  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cm0,
      thread dtype_frag_t<CType>& Cm1,
      const thread dtype_frag_t<AType>& Am0,
      const thread dtype_frag_t<AType>& Am1,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& B,
      metal::bool_constant<transpose_b>) {
    // Create Matmul descriptor
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    // Create matmul op
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    // Create matmul operands in registers
    auto ct_a =
        gemm_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b =
        gemm_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();

    // Create matmul output in register
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        decltype(ct_a),
        decltype(ct_b),
        CType>();

    // Load A in to left operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_a[i] = Am0[i];
      ct_a[kElemsPerFrag + i] = Am1[i];
    }

    // Load B into right operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_b[i] = B[i];
    }

    // Load C into output registers (op handles accumulation)
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_c[i] = Cm0[i];
      ct_c[kElemsPerFrag + i] = Cm1[i];
    }

    // Do matmul
    gemm_op.run(ct_a, ct_b, ct_c);

    // Copy out results
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      Cm0[i] = ct_c[i];
      Cm1[i] = ct_c[kElemsPerFrag + i];
    }
  }
};

template <
    typename T,
    short kTileRows_,
    short kTileCols_,
    class NAXFrag_ = BaseNAXFrag>
struct NAXTile {
  using NAXFrag_t = NAXFrag_;
  using elem_type = T;

  STEEL_CONST short kFragRows = NAXFrag_t::kFragRows;
  STEEL_CONST short kFragCols = NAXFrag_t::kFragCols;
  STEEL_CONST short kElemsPerFrag = NAXFrag_t::kElemsPerFrag;

  STEEL_CONST short kTileRows = kTileRows_;
  STEEL_CONST short kTileCols = kTileCols_;

  STEEL_CONST short kRows = kTileRows * kFragRows;
  STEEL_CONST short kCols = kTileCols * kFragCols;

  STEEL_CONST short kNumFrags = kTileRows * kTileCols;
  STEEL_CONST short kElemsPerTile = kNumFrags * kElemsPerFrag;

  STEEL_CONST short kFragThrRows = NAXFrag_t::kElemRows;
  STEEL_CONST short kFragThrCols = NAXFrag_t::kElemCols;
  STEEL_CONST short kFragRowsJump = NAXFrag_t::kElemRowsJump;

  STEEL_CONST short kRowsPerThread = kTileRows * NAXFrag_t::kElemRows;
  STEEL_CONST short kColsPerThread = kTileCols * NAXFrag_t::kElemCols;

  typedef typename NAXFrag_t::template dtype_frag_t<T> frag_type;

  frag_type val_frags[kNumFrags]; // = {frag_type(0)};

  METAL_FUNC NAXTile() thread {}

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j) const {
    return val_frags[i * kTileCols + j];
  }

  template <int i, int j>
  METAL_FUNC constexpr thread frag_type& frag_at() {
    return val_frags[i * kTileCols + j];
  }

  template <int i, int j>
  METAL_FUNC constexpr const thread frag_type& frag_at() const {
    return val_frags[i * kTileCols + j];
  }

  template <bool transpose>
  METAL_FUNC constexpr thread frag_type&
  frag_at(const short i, const short j, metal::bool_constant<transpose>) {
    if constexpr (transpose) {
      return frag_at(j, i);
    } else {
      return frag_at(i, j);
    }
  }

  template <bool transpose>
  METAL_FUNC constexpr const thread frag_type&
  frag_at(const short i, const short j, metal::bool_constant<transpose>) const {
    if constexpr (transpose) {
      return frag_at(j, i);
    } else {
      return frag_at(i, j);
    }
  }

  template <int i, int j, bool transpose>
  METAL_FUNC constexpr thread frag_type& frag_at() {
    if constexpr (transpose) {
      return frag_at<j, i>();
    } else {
      return frag_at<i, j>();
    }
  }

  template <int i, int j, bool transpose>
  METAL_FUNC constexpr const thread frag_type& frag_at() const {
    if constexpr (transpose) {
      return frag_at<j, i>();
    } else {
      return frag_at<i, j>();
    }
  }

  METAL_FUNC thread elem_type* elems() {
    return reinterpret_cast<thread elem_type*>(val_frags);
  }

  METAL_FUNC const thread elem_type* elems() const {
    return reinterpret_cast<const thread elem_type*>(val_frags);
  }

  template <typename Op>
  METAL_FUNC void row_reduce(thread metal::vec<T, kRowsPerThread>& vals) const {
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        NAXFrag_t::template row_reduce<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread metal::vec<T, kRowsPerThread>& vals) {
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        NAXFrag_t::template row_bin_op<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename U, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            Int<str_x>{},
            Int<str_y>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U, int str_x, int str_y>
  METAL_FUNC void store(threadgroup U* dst) const {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            Int<str_x>{},
            Int<str_y>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void load(const device U* src, const int ld) {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void
  load_rows_contig(const device U* src, const int ld, const short n_rows) {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load_rows_contig(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            n_rows,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U, int str_x>
  METAL_FUNC void load_contig_tg(const threadgroup U* src) {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load_contig_tg(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            Int<str_x>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void load_contig(const device U* src, const int ld) {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load_contig(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }


  template <typename U>
  METAL_FUNC void store(device U* dst, const int ld) const {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void
  load_rows(const device U* src, const int ld, const short n_rows) {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load_rows(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            n_rows,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void
  load_safe(const device U* src, const int ld, const short2 src_tile_dims) {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load_safe(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            src_tile_dims.y,
            src_tile_dims.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_rows(device U* dst, const int ld, const short n_rows)
      const {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_rows(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            n_rows,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void
  store_safe(device U* dst, const int ld, const short2 dst_tile_dims) const {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_safe(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            dst_tile_dims.y,
            dst_tile_dims.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_slice(
      device U* dst,
      const int ld,
      const short2 start,
      const short2 stop) const {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_slice(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            start.y,
            stop.y,
            start.x,
            stop.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }
};

template <
    class CTile,
    class ATile,
    class BTile,
    bool transpose_a,
    bool transpose_b>
METAL_FUNC void tile_matmad_nax(
    thread CTile& C,
    thread ATile& A,
    metal::bool_constant<transpose_a>,
    thread BTile& B,
    metal::bool_constant<transpose_b>) {
  // Static checks
  constexpr short TMa = transpose_a ? ATile::kTileCols : ATile::kTileRows;
  constexpr short TM = CTile::kTileRows;
  static_assert(TMa == TM, "MXU tile matmul: M dimensions do not match");

  constexpr short TNb = transpose_b ? BTile::kTileRows : BTile::kTileCols;
  constexpr short TN = CTile::kTileCols;
  static_assert(TNb == TN, "MXU tile matmul: N dimensions do not match");

  constexpr short TKa = transpose_a ? ATile::kTileRows : ATile::kTileCols;
  constexpr short TK = transpose_b ? BTile::kTileCols : BTile::kTileRows;
  static_assert(TKa == TK, "MXU tile matmul: K dimensions do not match");

  constexpr auto ta = metal::bool_constant<transpose_a>{};
  constexpr auto tb = metal::bool_constant<transpose_b>{};

  if constexpr (TN == 1 && TM % 2 == 0) {
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; mm += 2) {
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; ++nn) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          CTile::NAXFrag_t::mma(
              C.frag_at(mm, nn),
              C.frag_at(mm + 1, nn),
              A.frag_at(mm, kk, ta),
              A.frag_at(mm + 1, kk, ta),
              metal::bool_constant<transpose_a>{},
              B.frag_at(kk, nn, tb),
              metal::bool_constant<transpose_b>{});
        }
      }
    }
  } else if constexpr (TN % 2 == 0) {
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; ++mm) {
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; nn += 2) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          CTile::NAXFrag_t::mma(
              C.frag_at(mm, nn),
              C.frag_at(mm, nn + 1),
              A.frag_at(mm, kk, ta),
              metal::bool_constant<transpose_a>{},
              B.frag_at(kk, nn, tb),
              B.frag_at(kk, nn + 1, tb),
              metal::bool_constant<transpose_b>{});
        }
      }
    }
  }
}

} // namespace steel
} // namespace mlx

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/gemm/params.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/gemm/params.h"
// Copyright © 2024 Apple Inc.


///////////////////////////////////////////////////////////////////////////////
// GEMM param classes
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

struct GEMMParams {
  const int M;
  const int N;
  const int K;

  const int lda;
  const int ldb;
  const int ldd;

  const int tiles_n;
  const int tiles_m;

  const int64_t batch_stride_a;
  const int64_t batch_stride_b;
  const int64_t batch_stride_d;

  const int swizzle_log;
  const int gemm_k_iterations_aligned;

  const int batch_ndim;
};

struct GEMMSpiltKParams {
  const int M;
  const int N;
  const int K;

  const int lda;
  const int ldb;
  const int ldc;

  const int tiles_n;
  const int tiles_m;

  const int split_k_partitions;
  const int split_k_partition_stride;
  const int split_k_partition_size;

  const int swizzle_log;
  const int gemm_k_iterations_aligned;
};

struct GEMMAddMMParams {
  const int ldc;
  const int fdc;

  const int64_t batch_stride_c;

  const float alpha;
  const float beta;
};

} // namespace steel
} // namespace mlx

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/utils.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/utils.h"
// Copyright © 2024 Apple Inc.


#include <metal_stdlib>

METAL_FUNC ulong2 elem_to_loc_broadcast(
    uint elem,
    constant const int* shape,
    constant const int64_t* a_strides,
    constant const int64_t* b_strides,
    int ndim) {
  ulong loc_a{0};
  ulong loc_b{0};
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    int pos_in_dim = (elem % shape[i]);
    elem /= shape[i];
    loc_a += pos_in_dim * a_strides[i];
    loc_b += pos_in_dim * b_strides[i];
  }
  return ulong2(loc_a, loc_b);
}

METAL_FUNC ulong3 elem_to_loc_broadcast(
    uint elem,
    constant const int* shape,
    constant const int64_t* a_strides,
    constant const int64_t* b_strides,
    constant const int64_t* c_strides,
    int ndim) {
  ulong loc_a{0};
  ulong loc_b{0};
  ulong loc_c{0};
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    int pos_in_dim = (elem % shape[i]);
    elem /= shape[i];
    loc_a += pos_in_dim * a_strides[i];
    loc_b += pos_in_dim * b_strides[i];
    loc_c += pos_in_dim * c_strides[i];
  }
  return ulong3(loc_a, loc_b, loc_c);
}

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/gemm/transforms.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/gemm/transforms.h"
// Copyright © 2024 Apple Inc.



///////////////////////////////////////////////////////////////////////////////
// Transforms and Epilogues
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

template <typename OutT, typename InT>
struct TransformNone {
  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  static METAL_FUNC OutT apply(InT x, OutT) {
    return static_cast<OutT>(x);
  }
};

template <typename OutT, typename InT>
struct TransformAdd {
  TransformAdd(const float, const float) {}

  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  static METAL_FUNC OutT apply(InT x, OutT c) {
    return static_cast<OutT>(x) + c;
  }
};

template <typename OutT, typename InT>
struct TransformAxpby {
  const float alpha;
  const float beta;

  TransformAxpby(const float alpha_, const float beta_)
      : alpha(alpha_), beta(beta_) {}

  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  METAL_FUNC OutT apply(InT x, OutT c) const {
    return static_cast<OutT>(
        x * static_cast<InT>(alpha) + (static_cast<OutT>(beta) * c));
  }
};

template <typename T>
struct AccumHelper {
  typedef float accum_type;
};

struct BlockSwizzle {
  static METAL_FUNC int2
  swizzle(uint3 tid [[threadgroup_position_in_grid]], const int swizzle_log) {
    const int tid_x = (tid.x) >> swizzle_log;
    const int tid_y =
        ((tid.y) << swizzle_log) + ((tid.x) & ((1 << swizzle_log) - 1));
    return int2(tid_x, tid_y);
  }
};

} // namespace steel
} // namespace mlx

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
// Copyright © 2025 Apple Inc.



using namespace metal;

namespace mlx::steel {

template <
    typename T,
    short SM,
    short SN,
    short SK,
    short BK,
    bool transpose_a,
    bool transpose_b,
    bool kAlignedM,
    bool kAlignedN,
    bool kAlignedK,
    typename AccumType = float>
auto gemm_loop(
    const device T* A,
    const device T* B,
    int lda,
    int ldb,
    int K,
    int gemm_k_iterations_aligned,
    const short sgp_sm,
    const short sgp_sn) {
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  constexpr int RA = transpose_a ? TK : TM;
  constexpr int CA = transpose_a ? TM : TK;

  constexpr int RB = transpose_b ? TN : TK;
  constexpr int CB = transpose_b ? TK : TN;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  int gemm_k_iterations_ = gemm_k_iterations_aligned;

  STEEL_PRAGMA_NO_UNROLL
  for (int kk0 = 0; kk0 < gemm_k_iterations_; kk0++) {
    threadgroup_barrier(mem_flags::mem_none);

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      NAXTile<T, RA, CA> Atile;
      NAXTile<T, RB, CB> Btile;
      const int k = kk1;

      volatile int compiler_barrier;

      const int A_offset = transpose_a ? k * lda : k;
      const int B_offset = transpose_b ? k : k * ldb;

      if constexpr (kAlignedM) {
        Atile.load(A + A_offset, lda);
      } else {
        const short rmax = transpose_a ? SK : sgp_sm;
        const short cmax = transpose_a ? sgp_sm : SK;
        Atile.load_safe(A + A_offset, lda, short2(cmax, rmax));
      }

      if constexpr (kAlignedN) {
        Btile.load(B + B_offset, ldb);
      } else {
        const short rmax = transpose_b ? sgp_sn : SK;
        const short cmax = transpose_b ? SK : sgp_sn;
        Btile.load_safe(B + B_offset, ldb, short2(cmax, rmax));
      }

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<transpose_a>{},
          Btile,
          metal::bool_constant<transpose_b>{});

      (void)compiler_barrier;
    }

    A += transpose_a ? (BK * lda) : BK;
    B += transpose_b ? BK : (BK * ldb);
  }

  if constexpr (!kAlignedK) {
    simdgroup_barrier(mem_flags::mem_none);

    const short rem_bk = K - gemm_k_iterations_ * BK;

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < rem_bk; kk1 += SK) {
      NAXTile<T, RA, CA> Atile;
      NAXTile<T, RB, CB> Btile;

      const int k = kk1;
      const short psk = max(0, rem_bk - k);

      const short2 Aklims =
          transpose_a ? short2(sgp_sm, psk) : short2(psk, sgp_sm);
      const short2 Bklims =
          transpose_b ? short2(psk, sgp_sn) : short2(sgp_sn, psk);

      const int A_offset = transpose_a ? k * lda : k;
      const int B_offset = transpose_b ? k : k * ldb;

      Atile.load_safe(A + A_offset, lda, Aklims);
      Btile.load_safe(B + B_offset, ldb, Bklims);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<transpose_a>{},
          Btile,
          metal::bool_constant<transpose_b>{});
    }
  }

  return Dtile;
}

} // namespace mlx::steel

///////////////////////////////////////////////////////////////////////////////

// Copyright © 2025 Apple Inc.

// Auto generated source for mlx/backend/metal/kernels/quantized_utils.h

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/quantized_utils.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/quantized_utils.h"
// Copyright © 2023-2024 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>

template <typename T, typename mma_t, typename loader_a_t, typename loader_b_t>
METAL_FUNC void gemm_loop_aligned(
    threadgroup T* As,
    threadgroup T* Bs,
    thread mma_t& mma_op,
    thread loader_a_t& loader_a,
    thread loader_b_t& loader_b,
    const int k_iterations) {
  for (int k = 0; k < k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load elements into threadgroup memory
    loader_a.load_unsafe();
    loader_b.load_unsafe();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Multiply and accumulate threadgroup elements
    mma_op.mma(As, Bs);

    // Prepare for next iteration
    loader_a.next();
    loader_b.next();
  }
}

template <
    bool rows_aligned,
    bool cols_aligned,
    bool transpose,
    typename T,
    typename mma_t,
    typename loader_a_t,
    typename loader_b_t>
METAL_FUNC void gemm_loop_unaligned(
    threadgroup T* As,
    threadgroup T* Bs,
    thread mma_t& mma_op,
    thread loader_a_t& loader_a,
    thread loader_b_t& loader_b,
    const int k_iterations,
    const short tgp_bm,
    const short tgp_bn,
    const short tgp_bk) {
  for (int k = 0; k < k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load elements into threadgroup memory
    if (rows_aligned) {
      loader_a.load_unsafe();
    } else {
      loader_a.load_safe(short2(tgp_bk, tgp_bm));
    }
    if (cols_aligned) {
      loader_b.load_unsafe();
    } else {
      loader_b.load_safe(
          transpose ? short2(tgp_bk, tgp_bn) : short2(tgp_bn, tgp_bk));
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Multiply and accumulate threadgroup elements
    mma_op.mma(As, Bs);

    // Prepare for next iteration
    loader_a.next();
    loader_b.next();
  }
}

template <typename T, typename mma_t, typename loader_a_t, typename loader_b_t>
METAL_FUNC void gemm_loop_finalize(
    threadgroup T* As,
    threadgroup T* Bs,
    thread mma_t& mma_op,
    thread loader_a_t& loader_a,
    thread loader_b_t& loader_b,
    const short2 tile_a,
    const short2 tile_b) {
  loader_a.load_safe(tile_a);
  loader_b.load_safe(tile_b);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mma_op.mma(As, Bs);
}

///////////////////////////////////////////////////////////////////////////////

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
// Copyright © 2025 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>


constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];
// Set by gather_qmm_rhs_nax when DARKBLOOM_PREFILL_GATHER_RUNSKIP selects this
// dispatch. Default OFF: the host always supplies it, and when false the kernel
// is byte-for-byte the upstream algorithm.
constant bool gather_run_skip [[function_constant(203)]];

// DARKBLOOM staging levers for fp_gather_qmm_rhs_nax. All default OFF: an
// undefined bool function constant reads as false, exactly as the kernel
// behaves today. Each is resolved once per process on the host side, so no
// tunable magnitude ever enters the pipeline specialization key.
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

///////////////////////////////////////////////////////////////////////////////
// NVFP4 block-loader staging fast path.
//
// Two bit-exact rewrites of the fp4 staging chain QuantizedBlockLoader runs
// (the qmm / gather-qmm prefill kernels). Both rest on one observation about
// `fp4_e2m1::operator float16_t()`:
//
//     half converted = as_type<half>(ushort((bits & 7) << 9));
//     converted *= 16384.0;                        // 2^14
//     return bits & 8 ? -converted : converted;
//
// The 3-bit magnitude field is *bit-embedded* into a half -- fp4's 2-bit
// exponent lands in the low two bits of half's 5-bit exponent field and fp4's
// single mantissa bit lands in half mantissa bit 9 -- so the reinterpreted
// half is already the right number up to a fixed power of two: exactly
// {0, .5, 1, 1.5, 2, 3, 4, 6} * 2^-14. The `* 16384.0` is a pure
// renormalization, never a rounding step. (0.5 * 2^-14 == 2^-15 is a half
// subnormal, and its bit pattern is precisely the one we started from, so
// nothing rounds there either.)
//
// CHANGE 1 -- hoist the 2^14 out of the per-value converts into the one
// per-group scale. The loader stores `scale * value`, so with
//     s = the e4m3 group scale (at most 4 significant bits, |s| in
//         [2^-9, 448] or NaN)
//     m = an fp4 magnitude in {0, .5, 1, 1.5, 2, 3, 4, 6}
// today's chain rounds `s * (m * 2^-14 * 2^14)` once and the folded chain
// rounds `(s * 2^14) * (m * 2^-14)` once. Every factor is exact in binary FP:
//   * s * 2^14 only shifts an exponent -- no rounding -- and can neither
//     overflow (448 * 2^14 = 7340032, far inside float) nor underflow
//     (2^-9 * 2^14 = 2^5),
//   * m * 2^-14 is exactly representable in half, bfloat and float,
//   * so both orderings are the SAME real number rounded once to the same
//     destination type: identical bits.
// The per-value multiply count drops from `n_reads * pack_factor` to one.
// This is the loader-side sibling of the 2^22 fold `laguna_nvfp4_scale`
// already carries in the decode custom kernels.
//
// Restricted to group_size == 16, the e4m3 (NVFP4) scale. mxfp4's e8m0 scales
// (group_size 32) reach 2^127, where s * 2^14 would overflow to inf, so those
// instantiations keep the original chain byte for byte.
//
// CHANGE 2 -- spread eight nibbles per uint32 instead of two per byte. The
// byte-at-a-time chain costs AND + SHL + half multiply + AND + compare +
// select per nibble plus a SHR per byte (~104 scalar ops per thread per
// k-iteration at 16 values). The uint-at-a-time spread is four masked-
// shift-OR groups, 19 integer ops per uint32, each producing a half2 whose
// two lanes are two nibbles with the sign folded into the same OR. Ported
// from `laguna_nvfp4_qdot_16` in the decode custom kernels. The half bit
// patterns it builds are exactly the ones fp4_e2m1 builds one lane at a time,
// so the staged values are unchanged.
//
// Verified by exhaustive GPU enumeration against the byte-at-a-time chain for
// bfloat16_t, float16_t and float: all 256 scale bytes x all 256 packed
// bytes, and all 256 scale bytes x all 65536 four-nibble codes -- 0 bit
// mismatches out of 404,226,048 staged values.
///////////////////////////////////////////////////////////////////////////////

// Per-group NVFP4 scale with fp4's 2^14 renormalization folded in (Change 1).
static inline float fp4nv_scale_x16384(uint8_t s) {
  if (s < 16u) {
    return float(uint(s) << 5);
  }
  return float(*(thread fp8_e4m3*)(&s)) * 16384.0f;
}

// Four packed bytes -> one uint32 in little-endian nibble order. Read through
// packed_uchar4, whose alignment is 1, so widening the access adds no address
// precondition the byte-at-a-time loop did not already satisfy.
static inline uint32_t fp4nv_pack4(const device uint8_t* p) {
  return as_type<uint32_t>(uchar4(*(const device packed_uchar4*)p));
}
static inline uint32_t fp4nv_pack4(const thread uint8_t* p) {
  return as_type<uint32_t>(uchar4(p[0], p[1], p[2], p[3]));
}

// Decode the eight fp4 nibbles packed in `c` and apply the folded scale
// (Change 2). `out[k]` is nibble k -- byte k/2's low half for even k, high
// half for odd k -- which is exactly the order `dequantize<U, 4>` produces
// when walking those four bytes.
template <typename T>
static inline void fp4nv_decode8(uint32_t c, float scale, thread T* out) {
  // Split-nibble decode: identical half bit patterns to stock with fewer
  // integer ops and fewer live constant registers. See fp_quantized.cpp
  // qdot() for the bit-exactness argument.
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

  // The NVFP4 staging fast path applies when the packing is one byte per two
  // values, the scale is e4m3, and the byte run governed by ONE scale splits
  // evenly into uint32s (so a uint32 never straddles a scale boundary). Every
  // fp4 instantiation here qualifies: n_reads = 16 with n_reads_per_scale = 8.
  // mxfp8 (bits 8) and mxfp4 (e8m0 scales) keep the original scalar chain.
  MLX_MTL_CONST bool fp4nv_fast = (bits == 4) && (group_size == 16) &&
      (bytes_per_pack == 1) && (n_reads_per_scale >= 4) &&
      ((n_reads_per_scale % 4) == 0);

  // Stage this thread's n_reads packed bytes into `dst`. Identical values at
  // identical addresses on both paths; see the note above dequantize().
  void stage() const {
    if constexpr (fp4nv_fast) {
      int k = 0;
      for (int i = 0; i < n_steps_per_read; i++) {
        const float scale = fp4nv_scale_x16384(scales[i]);
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
        T scale = dequantize_scale<T, group_size>(scales[i]);
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
  // A single 8B device load covers it instead (the 256-thread expert-aligned
  // geometry: n_reads 8, one packed byte each). Same exactness class as the
  // 16B form -- the same bytes reach the same sb[] slots -- and the host
  // certification for 16B bases is strictly stronger than the 8B one, so it
  // is reused unchanged; only the per-thread offset check relaxes to 8B.
  MLX_MTL_CONST bool kWideLoad8ShapeOk = kWidenShapeOk && (kSrcBytes == 8);

  struct alignas(16) WideChunk {
    T v[kWideElems];
  };
  // Sized by kSrcBytes rather than a literal 16 so the copy loop below stays
  // in bounds for every instantiation, including the 8-bit ones where
  // kSrcBytes is 32 and the wide-load path is statically disabled.
  struct alignas(16) WideSrc {
    uint8_t b[kSrcBytes];
  };
  struct alignas(8) WideSrc8 {
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
    const bool load_ok = wide_load &&
        ((kWideLoadShapeOk && ((src_byte_off() & 15) == 0)) ||
         (kWideLoad8ShapeOk && ((src_byte_off() & 7) == 0)));

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

    STEEL_PRAGMA_UNROLL
    for (short c = 0; c < kWideChunks; c++) {
      const short e0 = c * kWideElems;
      const short k0 = c * kSrcBytesPerChunk;
      // Same scale the scalar loop selects for every k in this chunk:
      // i = k / n_reads_per_scale, constant across the chunk because
      // kSrcBytesPerChunk divides n_reads_per_scale.
      WideChunk out;
      // The NVFP4 spread needs 4 source bytes per call, so it is used only
      // when a 16B chunk covers a whole multiple of them (kSrcBytesPerChunk
      // is 4 for bfloat/half staging, 2 for float). Same values either way.
      if constexpr (fp4nv_fast && (kSrcBytesPerChunk % 4) == 0) {
        const float scale =
            fp4nv_scale_x16384(scales[k0 / n_reads_per_scale]);
        STEEL_PRAGMA_UNROLL
        for (short b = 0; b < kSrcBytesPerChunk / 4; b++) {
          fp4nv_decode8<T>(fp4nv_pack4(sb + k0 + b * 4), scale, &out.v[b * 8]);
        }
      } else {
        T scale =
            dequantize_scale<T, group_size>(scales[k0 / n_reads_per_scale]);
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
    typename Wtype = bfloat,
    const int fixed_K = 0,
    const int fixed_N = 0,
    const bool aligned_M = false>
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
  const int K_w = kernel_K * bytes_per_pack / pack_factor;
  const int K_g = kernel_K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(kernel_K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  y += y_row * static_cast<int64_t>(kernel_N) + y_col;

  // Make the weight loader
  loader_w_t loader_w(wl, scales, kernel_K, Ws, simd_gid, simd_lid);

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
        // Dead at k==0 for fixed_K>0: no prior-iteration Ws read to order.
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

      // Dead for fixed_K>0: no next iteration, epilogue never touches Ws.
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


"""#

/// Kernel body. `BM`/`BK`/`BN`/`WM`/`WN` and `aligned_N` are the constants
/// `qmm_nax()` uses for every NVFP4 `qmm_t` shape; `Wtype = bfloat` is the
/// template default the stock instantiation takes.
internal let lagunaSharedFusedQMMSource = #"""
    constexpr int BM = 64;
    constexpr int BK = 64;
    constexpr int BN = 64;
    constexpr int WM = 2;
    constexpr int WN = 2;
    constexpr int BK_padded = BK + 16 / sizeof(bfloat);

    threadgroup bfloat Ws[BN * BK_padded];

    fp_qmm_t_impl<
        bfloat16_t,
        /* group_size = */ 16,
        /* bits = */ 4,
        /* aligned_N = */ true,
        BM,
        BK,
        BN,
        WM,
        WN,
        bfloat>(
        w,
        scales,
        x,
        out,
        Ws,
        K,
        N,
        M,
        threadgroup_position_in_grid,
        thread_index_in_threadgroup,
        simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
    """#

/// Single-dispatch NVFP4 `y = x @ W^T` over the row-concatenated [gate; up]
/// bank, bit-exact against the two stock `fp_qmm_t_nax` dispatches.
internal let lagunaSharedFusedQMMKernel = MLXFast.metalKernel(
    name: "laguna_sfq_qmm_t_nax_bf16_gs16_b4_bm64_bn64_bk64_wm2_wn2_alN_true_batch0_v1",
    inputNames: ["w", "scales", "x", "K", "N", "M"],
    outputNames: ["out"],
    source: lagunaSharedFusedQMMSource,
    header: lagunaSharedFusedQMMHeader,
    ensureRowContiguous: true
)

/// Computes `y = x @ fusedWeight^T` for an NVFP4 group-16 4-bit bank in one
/// dispatch.
///
/// - Parameters:
///   - x: `[1, M, K]` bfloat16 activations.
///   - fusedWeight: `[N, K / 8]` uint32 NVFP4 codes, rows `0..<N/2` the gate
///     bank and `N/2..<N` the up bank.
///   - fusedScales: `[N, K / 16]` uint8 e4m3 group scales, same row order.
/// - Returns: `[1, M, N]` bfloat16, column-wise the concatenation of the two
///   banks' stock outputs.
///
/// `K % 64 == 0` and `N % 64 == 0` are the preconditions behind the pinned
/// `aligned_N == true` instantiation and the `BK = 64` K-loop; `aligned_M` is
/// left `false` exactly as the stock non-static dispatch leaves it, so any `M`
/// is accepted.
internal func lagunaSharedFusedGateUpQMM(
    x: MLXArray,
    fusedWeight: MLXArray,
    fusedScales: MLXArray
) -> MLXArray {
    precondition(x.dtype == .bfloat16, "x must be bfloat16, got \(x.dtype)")
    precondition(fusedWeight.dtype == .uint32, "fusedWeight must be uint32")
    precondition(fusedScales.dtype == .uint8, "fusedScales must be uint8")
    precondition(x.ndim == 3 && x.dim(0) == 1, "x must be [1, M, K], got \(x.shape)")
    precondition(fusedWeight.ndim == 2, "fusedWeight must be [N, K / 8]")
    precondition(fusedScales.ndim == 2, "fusedScales must be [N, K / 16]")

    let m = x.dim(1)
    let k = x.dim(2)
    let n = fusedWeight.dim(0)

    precondition(k % 64 == 0, "K must be a multiple of 64, got \(k)")
    precondition(n % 64 == 0, "N must be a multiple of 64, got \(n)")
    precondition(
        fusedWeight.dim(1) == k / 8,
        "fusedWeight must be [N, K / 8], got \(fusedWeight.shape) for K \(k)"
    )
    precondition(
        fusedScales.dim(0) == n && fusedScales.dim(1) == k / 16,
        "fusedScales must be [\(n), \(k / 16)], got \(fusedScales.shape)"
    )

    // `qmm_nax()`: grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B),
    // group_dims(32, wn, wm). MLX's custom-kernel primitive dispatches THREADS,
    // so the threadgroup counts are multiplied through; both extents divide
    // exactly, so the threadgroup grid and every builtin index match the stock
    // `dispatch_threadgroups` launch.
    let tilesN = (n + 63) / 64
    let tilesM = (m + 63) / 64

    return lagunaSharedFusedQMMKernel(
        [
            fusedWeight,
            fusedScales,
            x,
            MLXArray(Int32(k)),
            MLXArray(Int32(n)),
            MLXArray(Int32(m)),
        ],
        grid: (tilesN * 32, tilesM * 2, 2),
        threadGroup: (32, 2, 2),
        outputShapes: [[1, m, n]],
        outputDTypes: [.bfloat16]
    )[0]
}
