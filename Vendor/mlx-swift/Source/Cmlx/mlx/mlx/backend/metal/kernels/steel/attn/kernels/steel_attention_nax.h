// Copyright © 2024-25 Apple Inc.

#include "mlx/backend/metal/kernels/steel/attn/nax.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/backend/metal/kernels/steel/attn/transforms.h"
#include "mlx/backend/metal/kernels/steel/utils.h"

using namespace mlx::steel;

// DARKBLOOM_ATTN_QHOIST default. DEFAULT OFF: unless the compiler is invoked
// with -DDARKBLOOM_ATTN_QHOIST=1, the kernel below is byte-for-byte the
// upstream algorithm.
//
// NOTE ON WHICH TWIN RUNS. steel/attn is a JIT family: the runtime-effective
// source is the string in Cmlx/mlx-generated/steel_attention_nax.cpp, and the
// DARKBLOOM_ATTN_QHOIST env var is read there, by
// get_steel_attention_nax_kernel in mlx/backend/metal/jit_kernels.cpp, which
// prepends the define to that string. This AOT header is kept in sync so the
// two twins do not diverge, but nothing wires the env var to
// tools/build-mlx-metallib.sh -- the AOT copy is reachable only by adding the
// define to the metallib build flags by hand.
#ifndef DARKBLOOM_ATTN_QHOIST
#define DARKBLOOM_ATTN_QHOIST 0
#endif

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];

constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];
constant bool has_sinks [[function_constant(302)]];

template <typename T>
struct TransformScale {
  T scale;
  METAL_FUNC TransformScale(T scale_) : scale(scale_) {}

  METAL_FUNC T apply(T x) const {
    return scale * x;
  }
};

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct SubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x - y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_nax(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device MaskType* mask [[buffer(6), function_constant(has_mask)]],
    const device T* sinks [[buffer(7), function_constant(has_sinks)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;
  (void)simd_lane_id;

  // Move to correct block
  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      tidl.x * BQ * params->Q_strides[2]; // Sequence

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + // Batch
      kv_head_idx * params->K_strides[1]; // Head

  V += tidl.z * params->V_strides[0] + // Batch
      kv_head_idx * params->V_strides[1]; // Head

  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      tidl.x * BQ * params->O_strides[2]; // Sequence

  if (has_mask) {
    mask += tidl.z * mask_params->M_strides[0] + // Batch
        tidl.y * mask_params->M_strides[1]; // Head
  }

  const metal::uniform<float> scale2 =
      make_uniform(params->scale) * make_uniform(1.44269504089f);

  // Prepare MMA tiles
  constexpr short kU = 16;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kU);
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kU;
  // KV seq frags per warp
  constexpr short TK = BK / kU;

  static_assert(TQ == 1, "Check TQ");
  using otile_t = NAXTile<AccumType, TQ, TD>;
  otile_t Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short tm = kU * TQ * simd_group_id;
  Q += tm * int(params->Q_strides[2]);

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  // Init row reduction variables
  constexpr short kRowsPT = otile_t::kRowsPerThread;

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

  // Init to -Inf
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  if (has_sinks) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
      sum_score[i] = 1;
    }
  }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);

    int q_min = tid.x * BQ + params->qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  // Per-simdgroup causal K-block elision (level 1, always on). kb_lim above
  // derives from the THREADGROUP's last row; this simdgroup owns only rows
  // [tid.x * BQ + tm, tid.x * BQ + tm + kU * TQ). K blocks at or beyond
  // sg_kb_lim lie entirely above its causal diagonal: the causal mask would
  // set every element of its Stile rows to neg_inf, making the P tile
  // exactly the all-+0.0 tile Stile.clear() already produces, new_max equal
  // to max_score (factor == exp2(+0.0) == 1.0), and the sum_score update a
  // +0.0 add into a value that is always >= +0.0. Skipping QK^T, the scale,
  // both masks and the softmax for those blocks is therefore bit-exact.
  // Level 2 also skips the P@V loads and MMAs: every Stile multiplicand is
  // +0.0, so those instructions can only add a signed zero to Otile. That
  // leaves every finite nonzero accumulator bit-identical; an exactly-zero
  // accumulator can differ only in its zero sign, which is numerically equal
  // through the final positive normalization and all downstream arithmetic.
  // The kb trip count and every barrier stay untouched (the P@V loop contains
  // a threadgroup_barrier at BD == 128, so a per-simdgroup trip count would be
  // undefined behaviour); sg_active is simdgroup-uniform (tid.x and tm only).
  // Restricted to
  // do_causal && !has_mask so the all-masked proof rests on the causal mask
  // alone; the timed window passes no array mask.
  int sg_kb_lim = kb_lim;
  if (do_causal && !has_mask) {
    int sg_q_max = int(tid.x) * BQ + params->qL_off + int(tm) + kU * TQ;
    sg_kb_lim = min(kb_lim, (sg_q_max + BK - 1) / BK);
  }

  const bool is_last_bq = int(tid.x) == (params->NQ_aligned);
  // const bool is_last_tq = int(simd_group_id) >= (params->qL_rem / UQ);
  const bool is_last_q = is_last_bq;

  const short lim_rows_q = params->qL_rem - tm;
  const short lim_rows_k = params->kL_rem;

#if DARKBLOOM_ATTN_QHOIST
  // DARKBLOOM_ATTN_QHOIST -- hoist the loop-invariant Q fragments.
  //
  // The kb loop advances K and V (`K += BK * K_strides[2]` at the bottom) but
  // NEVER advances Q: the Q pointer is finalised above and is constant for the
  // whole loop. The QK^T phase nevertheless re-executed `Qtile.load(...)` on
  // every iteration, re-reading the identical TQ*TD fragments from device
  // memory ~9 times per q-block at the frozen 512-token prefill window.
  //
  // Staging them once here is a PURE HOIST: the loads use the same pointer,
  // the same offsets and the same bounds predicate as the in-loop version, and
  // the mma consumes the identical fragment values in the identical order. No
  // float arithmetic is touched, so no rounding boundary can move.
  //
  // Both loop nests are STEEL_PRAGMA_UNROLL, so every Qhoist index is a
  // compile-time constant and the array stays in registers. If either unroll
  // were ever dropped, Qhoist would spill to thread-local memory and this
  // would become a pessimisation, not an optimisation.
  //
  // COST (BQ=64 BK=32 BD=128 WM=4 WN=1 => TQ=1 TD=8, T=bfloat16): TQ*TD = 8
  // fragments x 8 elems x 2 B = 128 B/thread = 32 x 32-bit registers, live for
  // the whole loop instead of one fragment (4 registers) live for part of it,
  // so +28 registers/thread and +16 KB/threadgroup at 128 threads. See
  // notes/21-attn-analysis.md for why that is expected to fit: this kernel
  // allocates ZERO threadgroup memory, and on Apple family 9+ the on-chip pool
  // is shared between registers and threadgroup memory.
  typename NAXTile<T, 1, 1>::frag_type Qhoist[TQ * TD];

  STEEL_PRAGMA_UNROLL
  for (short iq = 0; iq < TQ; iq++) {
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      NAXTile<T, 1, 1> Qstage;
      const int Q_load_off = iq * kU * int(params->Q_strides[2]) + id * kU;

      if (!align_Q && is_last_q) {
        Qstage.load_rows(
            Q + Q_load_off, int(params->Q_strides[2]), lim_rows_q - iq * kU);
      } else {
        Qstage.load(Q + Q_load_off, int(params->Q_strides[2]));
      }

      Qhoist[iq * TD + id] = Qstage.frag_at(0, 0);
    }
  }
#endif

  // DARKBLOOM K-TILE THREADGROUP STAGING (always on).
  //
  // Upstream, every one of the WM*WN simdgroups loaded the SAME K block
  // fragments straight from device memory inside the QK^T nest, so the
  // threadgroup paid for the identical block WM*WN times (4x at the ranked
  // shape). Stage the block ONCE per threadgroup here and let all simdgroups
  // read their fragments out of threadgroup memory instead. V is deliberately
  // left on the device path.
  //
  // EXACTNESS: this is a bitwise relocation of the same T values, nothing
  // else. The staging copy is storage-type to storage-type (no
  // bfloat -> float -> bfloat round trip), the fragment fetch below goes
  // through the same BaseNAXFrag::load / BaseNAXFrag::load_rows entry points
  // with the same per-lane coordinates, the same per-fragment row offsets and
  // the same row bound, and the fragment orientation, the transpose flags and
  // the mma call sequence are untouched. NO FLOAT ARITHMETIC IS TOUCHED AT
  // ALL -- nothing is reassociated, no accumulation order changes, no rounding
  // boundary moves. Only WHERE the fragment bits are fetched from changes.
  //
  // LAYOUT: BK rows of BD elements at a padded row stride of BD + 8 elements.
  // At the ranked shape (BK=32, BD=128, T=bfloat16) that is 32 x 136 elements
  // = 8704 B, of which 8192 B are written; the 8 pad elements per row are
  // never written and never read. The pad keeps every row 16 B aligned
  // (136 * 2 B = 272 B = 17 * 16 B) while avoiding the pathological
  // power-of-two 256 B stride. This kernel allocated ZERO threadgroup memory
  // before, so 8704 B is the whole threadgroup-memory budget it now takes.
  constexpr int kKStageLD = BD + 8;
  constexpr int kKStageThreads = WM * WN * 32;
  constexpr int kKStageElemsPerThread = (BK * BD) / kKStageThreads;
  constexpr int kKStageThreadsPerRow = BD / kKStageElemsPerThread;
  static_assert(
      kKStageElemsPerThread * kKStageThreads == BK * BD &&
          kKStageThreadsPerRow * kKStageElemsPerThread == BD,
      "K staging must tile the block exactly");

  threadgroup T Kstage[BK * kKStageLD];

  // Flat thread index in the threadgroup. The launch is (32, WM, WN), so
  // simd_group_id * 32 + simd_lane_id enumerates all kKStageThreads lanes
  // exactly once. At the ranked shape that is 128 threads, four per row:
  // row = tid_in_tg >> 2, col0 = (tid_in_tg & 3) * 32.
  const int tid_in_tg = int(simd_group_id) * 32 + int(simd_lane_id);
  const short kstage_row = short(tid_in_tg / kKStageThreadsPerRow);
  const short kstage_col =
      short((tid_in_tg % kKStageThreadsPerRow) * kKStageElemsPerThread);

  // Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {
    const int is_last_k = (kb == (params->NK_aligned));

    // Stage this K block. K still points at the current block here -- the
    // advance happens at the bottom of the loop.
    //
    // UNIFORMITY: kb_lim depends only on tid.x and params, so the trip count
    // is threadgroup-uniform and every thread reaches the staging and both
    // barriers on every iteration. They therefore sit OUTSIDE the
    // per-simdgroup `sg_active` causal elision below, which continues to
    // guard fragment CONSUMPTION only. No barrier sits inside a
    // simdgroup-varying conditional.
    //
    // TAIL: on the ragged last block the rows at or past lim_rows_k do not
    // exist in device memory, so they are staged as T(0) rather than read --
    // no out-of-range device read is ever issued. Columns are always full
    // (BD is the head dim). The bounded fragment fetch below keeps the same
    // lim_rows_k bound, so those staged zeros are never actually read back,
    // matching upstream where load_rows zero-fills without touching memory.
    {
      threadgroup T* kstage_dst = Kstage + kstage_row * kKStageLD + kstage_col;

      if ((!align_K && is_last_k) && kstage_row >= lim_rows_k) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < kKStageElemsPerThread; e++) {
          kstage_dst[e] = T(0);
        }
      } else {
        const device T* kstage_src =
            K + kstage_row * int(params->K_strides[2]) + kstage_col;

        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < kKStageElemsPerThread; e++) {
          kstage_dst[e] = kstage_src[e];
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Do S = Q @ K.T
    using stile_t = NAXTile<AccumType, TQ, TK>;
    using ktile_t = NAXTile<T, 2, 1>;
    stile_t Stile;

    Stile.clear();

    // Causal elision: guard the score computation and the zero P@V work, but
    // never a barrier or the outer-loop pointer advance. See the sg_kb_lim
    // comment above for the exactness argument.
    const bool sg_active = kb < sg_kb_lim;
    if (sg_active) {

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TD; id++) {
          NAXTile<T, 1, 1> Qtile;
          ktile_t Ktile;

#if !DARKBLOOM_ATTN_QHOIST
          const int Q_load_off = iq * kU * int(params->Q_strides[2]) + id * kU;
#endif
          const int K_stage_off = ik * kU * kKStageLD + id * kU;

#if DARKBLOOM_ATTN_QHOIST
          // Q is loop-invariant: the kb loop advances K and V but never Q, so
          // the load in the #else branch re-read the same addresses on every
          // one of the ~9 K-block iterations. Consume the fragment staged
          // before the loop instead.
          //
          // EXACTNESS: the staged load used the same base pointer, the same
          // Q_load_off, the same stride and the same bounds predicate, so the
          // bits in Qhoist are the bits this load would have returned. The mma
          // below consumes them in the identical order. NO FLOAT ARITHMETIC IS
          // TOUCHED AT ALL -- nothing is reassociated, no accumulation order
          // changes, no rounding boundary moves. The only difference is WHEN
          // the device read happened.
          Qtile.frag_at(0, 0) = Qhoist[iq * TD + id];
#else
          if (!align_Q && is_last_q) {
            Qtile.load_rows(
                Q + Q_load_off,
                int(params->Q_strides[2]),
                lim_rows_q - iq * kU);
          } else {
            Qtile.load(Q + Q_load_off, int(params->Q_strides[2]));
          }
#endif

          // Consume the block staged above instead of re-reading it from
          // device memory. Same fragment entry points, same per-fragment row
          // offsets (0 and kU) and -- for the ragged tail -- the same
          // `lim_rows_k - ik * kU` bound as the device path; only the source
          // pointer and the row stride change. Keeping the bound on top of
          // the zero-staging is belt and braces: the staged zero rows are
          // never read, exactly as upstream never read them.
          const threadgroup T* Kstage_src = Kstage + K_stage_off;

          if (!align_K && is_last_k) {
            const short k_lim_rows = lim_rows_k - ik * kU;

            ktile_t::NAXFrag_t::load_rows(
                Ktile.frag_at(0, 0),
                Kstage_src,
                kKStageLD,
                Int<1>{},
                k_lim_rows,
                Int<0>{},
                Int<0>{});
            ktile_t::NAXFrag_t::load_rows(
                Ktile.frag_at(1, 0),
                Kstage_src,
                kKStageLD,
                Int<1>{},
                k_lim_rows,
                Int<kU>{},
                Int<0>{});
          } else {
            ktile_t::NAXFrag_t::load(
                Ktile.frag_at(0, 0),
                Kstage_src,
                kKStageLD,
                Int<1>{},
                Int<0>{},
                Int<0>{});
            ktile_t::NAXFrag_t::load(
                Ktile.frag_at(1, 0),
                Kstage_src,
                kKStageLD,
                Int<1>{},
                Int<kU>{},
                Int<0>{});
          }

          stile_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qtile.frag_at(0, 0),
              metal::false_type{},
              Ktile.frag_at(0, 0),
              Ktile.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    // Scale S
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= float(scale2);
    }

    // Mask out length sequence
    if (!align_K && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = ik * kU + sn;

          thread auto& fg = Stile.frag_at(iq, ik);

          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < params->kL_rem) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= kb_min_causal) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);

          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto r =
                  base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
              const auto c = base_col + ik * kU + jj + sn;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + tm;
      const int base_col = kb * BK;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, AccumType>;
      using mtile_t = NAXTile<melem_t, TQ, TK>;
      using mfrag_t = typename mtile_t::frag_type;

      if (base_row + BQ <= params->qL && base_col + BK <= params->kL) {
        for (short iq = 0; iq < TQ; iq++) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            const int row_pos = base_row + iq * kU;
            const int col_pos = base_col + ik * kU;

            mfrag_t mfrag;
            mtile_t::NAXFrag_t::load(
                mfrag,
                mask,
                int64_t(mask_params->M_strides[2]),
                Int<1>{},
                row_pos,
                col_pos);

            thread auto& fg = Stile.frag_at(iq, ik);

            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
              if constexpr (is_bool) {
                fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
              } else {
                fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
              }
            }
          }
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            const int row_pos = base_row + iq * kU;
            const int col_pos = base_col + ik * kU;

            mfrag_t mfrag;
            mtile_t::NAXFrag_t::load_safe(
                mfrag,
                mask,
                int64_t(mask_params->M_strides[2]),
                Int<1>{},
                params->qL,
                params->kL,
                row_pos,
                col_pos);

            thread auto& fg = Stile.frag_at(iq, ik);

            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
              if constexpr (is_bool) {
                fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
              } else {
                fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
              }
            }
          }
        }
      }
    }

    // Do softmax

    // Temp variables
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    // Row Sum
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }

    Stile.template row_reduce<SumOp>(sum_score);

    // Update O
    Otile.template row_bin_op<MulOp>(factor);
    }

    simdgroup_barrier(mem_flags::mem_none);

    // Do O = P @ V
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id += 2) {
        if constexpr (BD == 128) {
          if (id == 4) {
            threadgroup_barrier(mem_flags::mem_none);
          }
        }

        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if (sg_active) {
          NAXTile<T, 1, 2> Vtile;

          const int V_load_off = ik * kU * int(params->V_strides[2]) + id * kU;

          if (!align_K && is_last_k) {
            Vtile.load_rows(
                V + V_load_off,
                int(params->V_strides[2]),
                lim_rows_k - ik * kU);
          } else {
            Vtile.load(V + V_load_off, int(params->V_strides[2]));
          }

          otile_t::NAXFrag_t::mma(
              Otile.frag_at(iq, id),
              Otile.frag_at(iq, id + 1),
              Stile.frag_at(iq, ik),
              metal::false_type{},
              Vtile.frag_at(0, 0),
              Vtile.frag_at(0, 1),
              metal::false_type{});
          }
        }
      }
    }

    // Prepare for next iteration
    K += BK * int(params->K_strides[2]);
    V += BK * int(params->V_strides[2]);

    // The whole block has now been consumed: the P@V nest above reads V from
    // device memory and never touches Kstage. Fence before the next
    // iteration's staging overwrites the tile. Unconditional and uniform --
    // it sits at the loop bottom, outside every sg_active guard.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Normalize output

  threadgroup_barrier(mem_flags::mem_none);

  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }

  Otile.template row_bin_op<MulOp>(rcp);

  // Store results
  O += tm * int(params->O_strides[2]);

  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;

    Otile.store_rows(O, int(params->O_strides[2]), lim_rows_q);
  } else {
    Otile.store(O, int(params->O_strides[2]));
  }
}
