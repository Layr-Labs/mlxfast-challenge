// Copyright © 2024 Apple Inc.

#include <metal_simdgroup>

using namespace metal;

// ---------------------------------------------------------------------------
// DARKBLOOM AOT: sdpa_vector exchange-plane width
//
// This header is AOT-only. It has no `mlx-generated/*.cpp` twin and is served
// from `mlx.metallib`, which `tools/build-mlx-metallib.sh` compiles. EVERY
// change here needs that rebuild to take effect, and a stale `mlx.metallib`
// silently serves the previous kernel -- a measurement taken without a rebuild
// is a false null in EITHER direction.
//
// WHAT THE WIDTH DOES
//
// `sdpa_vector`'s combine loop transposes the per-simdgroup output partials
// through one recycled threadgroup plane, so every element pays a RAW barrier
// after its write and a WAR barrier before the next write. Both hazards are
// artifacts of the reuse, not of the reduction. Give the elements separate
// planes and the hazards disappear; the one surviving rendezvous then absorbs
// the max/sum publish that already precedes the loop. At Laguna's D = V = 128
// (v_per_thread = 4), per threadgroup:
//
//   PLANES  planes  outputs   kernel threadgroup mem   barriers
//   1       1        4096 B   4352 B                   8   (upstream, OFF)
//   2       2        8192 B   8448 B                   3
//   4       4       16384 B   16640 B                  1
//
// Two live widths because the trade buys barriers with threadgroup memory and
// the residency cost of that memory is a hardware constant we have not
// measured. If the kernel is thread-limited (1024 threads per threadgroup is
// the binding constraint) PLANES=4 is free and dominates. If a core can
// co-reside two of these threadgroups against a 32 KiB threadgroup-memory
// pool, 2 x 16640 B = 33280 B does not fit and PLANES=4 would pay for its
// barriers with occupancy, while 2 x 8448 B = 16896 B always does. PLANES=2 is
// therefore safe under every hypothesis. Measure all three.
//
// HOW AN ARM IS SELECTED
//
// Both mechanisms exist because they serve different masters:
//
//  * SHIPPED (packaged, authoritative): the `DARKBLOOM_AOT_SDPA_PLANES` macro
//    below picks the width behind the unsuffixed kernel name the stock MLX
//    dispatcher asks for. Whatever this is set to at `mlxfast submit` time IS
//    WHAT SHIPS. Default 1 = upstream = OFF.
//
//  * MEASUREMENT (local only, NEVER packaged): every width is also
//    instantiated under its own kernel name (`..._planes1/2/4`) by
//    `scaled_dot_product_attention.metal`, and a one-hunk local patch to
//    `backend/metal/scaled_dot_product_attention.cpp` appends that suffix from
//    the `DARKBLOOM_AOT_SDPA_PLANES` environment variable. The point of the
//    named variants is that all three arms live in ONE metallib and ONE
//    binary, so an A/B/C series can interleave without a rebuild between runs
//    and without crossing a build boundary.
//
//    THAT PATCH MUST BE REVERTED BY HAND BEFORE `mlxfast submit`.
//    `scaled_dot_product_attention.cpp` is outside `benchmark.json`
//    editablePaths, and AGENTS.md is explicit that submit "rejects ... source
//    changes outside the editable surface" -- it REJECTS them, it does not
//    silently drop them, so leaving the hunk in place FAILS THE UPLOAD.
//    (An earlier revision of this comment claimed it was "dropped by submit".
//    That was wrong, and a wrong assertion sitting next to the shipping
//    switch is how the next person gets caught.)
//
//    The named `_planes*` instantiations are likewise measurement scaffolding
//    and are compiled only under `-DDARKBLOOM_AOT_SDPA_MEASURE`; see the guard
//    in `scaled_dot_product_attention.metal`. The shipped metallib therefore
//    contains exactly the upstream kernel set, at the width set below.
//
// Set the macro to the width you intend to ship before packaging. An arm
// measured only through the environment variable ships as a no-op otherwise.
//
// MEASURED, session 7, 3-arm Latin square, one binary, n=12 pairs, median
// steady decode step over steps 8..128, "+" = faster:
//     PLANES=1 -> 2   10.4045 -> 10.3673 ms   +0.24%  t=1.21   8/12  (decays)
//     PLANES=1 -> 4   10.4045 -> 10.3323 ms   +0.60%  t=4.64  12/12
// 95% CI for PLANES=4 is [+0.35%, +0.86%], halves +0.67%/+0.54% (0.14 pp
// swing). PLANES=2 is the RUNBAR shape -- +0.58% 4/4 at n=4 collapsing to
// +0.24% 8/12 at n=12 -- and is NOT shipped. The residency worry that once
// argued for the narrower width is empirically dead: at 16640 B the pipeline
// still reports maxTotalThreadsPerThreadgroup 1024, and sdpa_vector is
// already residency-1, so there was no residency left to lose.
#ifndef DARKBLOOM_AOT_SDPA_PLANES
#define DARKBLOOM_AOT_SDPA_PLANES 4
#endif

// Same encoding for `sdpa_vector_2pass_2`. Compile-time only, with no named
// variants: that kernel is off the scored decode path (the host routes to
// 2-pass only at KV length >= 1024 and the frozen timed window peaks at
// 512 + 128 = 640), so it does not earn a slot in the measurement protocol.
#ifndef DARKBLOOM_AOT_SDPA_2PASS_PLANES
#define DARKBLOOM_AOT_SDPA_2PASS_PLANES 1
#endif

// Laguna decode uses eight query heads per KV head on sliding layers and six
// on full-attention layers. The stock vector kernel launches one 1024-thread
// group per query head, so each resident K/V row is reread for every owned
// query head. Group adjacent query heads inside one threadgroup: they retain
// independent online-softmax state and the exact stock 32-simdgroup reduction
// tree, but share each K/V load. The host grid is unchanged; its upper half
// returns uniformly before touching memory.
//
// The group size is chosen per GQA factor so it divides evenly and no group
// crosses a KV-head ownership boundary: 3 for GQA6 (6/3=2 groups per KV head)
// and 4 for GQA8 (8/4=2 groups per KV head). Both yield exactly two active
// query groups per KV head — the maximum K/V sharing. Six exchange planes
// serve both: 3 heads × 2 planes = 6 for GQA6, and 4 heads use a 3+1 staggered
// exchange reusing the same 6 planes. GROUP_MAX stays at 3; head 3 uses
// explicit register variables so the 2/3-head compiled code is unchanged.
// The path is restricted to the scored D=V=128, Laguna GQA (8 or 6),
// one-query, unmasked, sink-free vector shape.
#ifndef DARKBLOOM_GQA_GROUP_FULL
#define DARKBLOOM_GQA_GROUP_FULL 3
#endif
#ifndef DARKBLOOM_GQA_GROUP_SLIDING
#define DARKBLOOM_GQA_GROUP_SLIDING 4
#endif
#ifndef DARKBLOOM_GQA_GROUP_MAX
#define DARKBLOOM_GQA_GROUP_MAX 3
#endif

// ---------------------------------------------------------------------------
// DARKBLOOM_ALPHASKIP: exact online-softmax rescale elision, keyed on the
// bit-exact alpha == 1 case. Default ON; `0` restores the stock statement
// textually (the `#else` arm below IS the upstream line), so an OFF build is
// a byte-identical control. There is no function-constant or environment
// switch: this header is AOT-only and the dispatcher that would have to
// select a second kernel name lives in non-editable host code, so the two
// arms are selected by building two metallibs.
//
// HOW OFTEN alpha == 1 (measured, not assumed). Instrumented decode run on
// this tree (device counters in the rescale site, sampled over 4 of the 32
// simdgroups, `--local-iterate`, 128 decode steps):
//
//   layer family        steady sites     factor == +1.0f      fraction
//   full  (48 q heads)     3 264 288          2 663 629        81.60%
//   sliding (64 q heads)  11 531 520          9 990 456        86.64%
//   combined              14 795 808         12 654 085        85.52%
//
// The first key each simdgroup visits is never an alpha == 1 site (max_score
// starts at Limits<U>::finite_min, so factor is +0.0 there): 0 / 960 960.
// The dispatched kernel is this one-pass `sdpa_vector`, on the GQA-pair path,
// for BOTH families -- the host routes to `sdpa_vector_2pass` only at
// kL >= 1024 and the frozen window peaks at 640, and the generic (non-pair)
// path of this kernel logged zero decode sites.
//
// WHAT IS *NOT* ELIDED, AND WHY. The obvious elision -- skipping the
// `sum * factor` and `o[j] * factor` multiplies -- is both INEXACT and
// WORTHLESS here, and the reason is codegen, not arithmetic. This toolchain
// contracts
//     o[j] = o[j] * factor + exp_score * value
// into `fma(o[j], factor, fmul(exp_score, value))`: it fuses the FACTOR
// multiply and rounds `exp_score * value` separately. Measured over 4 194 304
// randomized (o, factor, exp, value) quadruples, `fma(o, factor, e * v)`
// reproduces the stock spelling on all of them while `fma(e, v, o * factor)`
// differs on 742 292. Two consequences:
//   * The "pure-FMA accumulate" `fma(exp_score, value, o[j])` is NOT bitwise
//     equal to the stock expression at factor == 1.0f -- it rounds the
//     product once where stock rounds it twice -- and differed on
//     372 418 / 4 194 304 factor == 1 quadruples. It would move tokens.
//   * The factor multiply costs nothing to begin with: it is absorbed into an
//     FMA the accumulate needs anyway, so `o[j] + P` and
//     `fma(o[j], factor, P)` are both one instruction. Same for
//     `sum_exp_score * factor + exp_score`. There is no work there to remove.
//
// WHAT *IS* ELIDED. The only work the alpha == 1 case actually contains is
// the evaluation of `factor` itself. When the running max does not advance,
// `max_score - new_max` is bit-exactly +0.0f (x - x, finite x, round-to-
// nearest), and `fast::exp(+0.0f)` is bit-exactly 1.0f on this toolchain
// (verified against every finite float32 exponent class, and against the
// literal). The branch therefore substitutes the constant the call provably
// returns and leaves every downstream expression CHARACTER-IDENTICAL:
// `factor` is still a runtime value the compiler cannot fold, so
// `sum_exp_score * factor + exp_score` and `o[j] * factor + exp_score * v`
// emit the same instructions on the same bits as stock. Exactness reduces to
// one checkable fact -- fast::exp(+0.0f) == 1.0f -- instead of to a claim
// about rounding, and no accumulator, reduction, or rounding boundary moves.
//
// PREDICATE. It is the bit pattern of the delta, not `score <= max_score`.
// That routes every non-finite case through the unchanged stock call: if
// max_score or new_max is +/-inf the delta is NaN, whose bits are not +0.0,
// so `fast::exp` runs exactly as before. It is simdgroup-uniform -- `score`
// is the broadcast result of `simd_sum` and `max_score` is updated
// identically in all 32 lanes -- so the whole simdgroup takes one arm and the
// branch costs no divergence. Masked and sink handling is untouched: the
// sink seed only changes the initial `max_score`, and the mask decides
// whether the site executes at all, both upstream of this substitution.
//
// `sdpa_vector_2pass_1` carries the same rescale but is deliberately NOT
// changed: it is off the scored path (see the 2PASS_PLANES note above) and
// the instrumented run logged zero 2-pass launches.
//
// MEASURED OUTCOME: NEUTRAL. Do not re-run this experiment expecting a win.
// Validation: 347 280 000 rescale sites / 79 872 000 bf16 output elements
// compared bitwise between an ALPHASKIP=1 and an ALPHASKIP=0 metallib on
// identical random Q/K/V across four decode shapes (denormals, +/-0, +/-inf,
// NaN, and planted duplicate K rows to force exact score ties) -- 0
// mismatches; and the `--local-iterate` drift signature was unchanged in
// every run (checked_step=0 expected_token=5991 actual_token=8550, golden
// b9509697c08a), so the elision does not move tokens. But the timing is a
// null: position-balanced 8-run A/B (OFF,ON,ON,OFF then ON,OFF,OFF,ON,
// metallib copy-swap, one worker binary), steady decode s/token
//     ON  10.6930 10.5872 10.5863 10.5455 ms   mean 10.6030
//     OFF 10.6618 10.5626 10.5584 10.6211 ms   mean 10.6010
// = -0.02% for ON, i.e. dead even against a ~0.3% adjacent-pair noise floor.
// One `fast::exp` per elided (key, head) is simply not what this loop is
// bound by. It is left ON because it is exact and free, not because it pays.
#ifndef DARKBLOOM_ALPHASKIP
#define DARKBLOOM_ALPHASKIP 1
#endif

// Bit-exact +1.0f test on the online-softmax rescale delta. `d` is
// `max_score - new_max`; the stock value of the rescale factor is
// `fast::exp(d)`.
#if DARKBLOOM_ALPHASKIP
#define DARKBLOOM_RESCALE_FACTOR(dst, delta_expr)   \
  do {                                              \
    const U db_delta_ = (delta_expr);               \
    if (as_type<uint>(db_delta_) == 0u) {           \
      dst = U(1.0f);                                \
    } else {                                        \
      dst = fast::exp(db_delta_);                   \
    }                                               \
  } while (false)
#else
#define DARKBLOOM_RESCALE_FACTOR(dst, delta_expr) \
  do {                                            \
    dst = fast::exp(delta_expr);                  \
  } while (false)
#endif

constant bool has_mask [[function_constant(20)]];
constant bool query_transposed [[function_constant(21)]];
constant bool do_causal [[function_constant(22)]];
constant bool bool_mask [[function_constant(23)]];
constant bool float_mask [[function_constant(24)]];
constant bool has_sinks [[function_constant(25)]];
constant int blocks [[function_constant(26)]];

// `PLANES` is the requested exchange width, clamped to `v_per_thread` below.
// `DEFAULT_ENTRY` carries no code: it only keeps the unsuffixed default entry
// point a distinct specialization from the equal-width named variant, so both
// can be instantiated from one template without a redefinition.
template <
    typename T,
    int D,
    int V = D,
    int PLANES = DARKBLOOM_AOT_SDPA_PLANES,
    bool DEFAULT_ENTRY = true>
[[kernel]] void sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    const device bool* bmask [[buffer(11), function_constant(bool_mask)]],
    const device T* fmask [[buffer(12), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(13), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(14), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(15), function_constant(has_mask)]],
    const device T* sinks [[buffer(16), function_constant(has_sinks)]],
    const constant int& num_q_heads
    [[buffer(17), function_constant(has_sinks)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  // Clamp: more planes than elements would allocate threadgroup memory nobody
  // writes. The clamp is also the compile-safety net for D = V = 256, where
  // v_per_thread = 8 and eight 4 KiB planes (32768 B, plus max_scores and
  // sum_exp_scores) would exceed the 32 KiB per-threadgroup limit - a hard
  // metallib compile error, not a slow kernel. PLANES never exceeds 4.
  constexpr int v_planes = PLANES < v_per_thread ? PLANES : v_per_thread;
  constexpr int gqa_group_enabled =
      DARKBLOOM_GQA_GROUP_FULL >= 2 && DARKBLOOM_GQA_GROUP_SLIDING >= 2;
  // 3 heads × 2 planes = 6 for GQA6; 4 heads reuse the same 6 staggered.
  constexpr int gqa_max_group =
      DARKBLOOM_GQA_GROUP_FULL > DARKBLOOM_GQA_GROUP_SLIDING
      ? DARKBLOOM_GQA_GROUP_FULL
      : DARKBLOOM_GQA_GROUP_SLIDING;
  constexpr int exchange_planes =
      (D == 128 && V == 128 && gqa_group_enabled)
      ? (gqa_max_group >= 3 ? 6 : 4)
      : v_planes;
  threadgroup U outputs[exchange_planes * BN * BD];
  threadgroup U max_scores[DARKBLOOM_GQA_GROUP_MAX * BN];
  threadgroup U sum_exp_scores[DARKBLOOM_GQA_GROUP_MAX * BN];

  // GQA group sharing: preserve each head's exact key order and reduction
  // tree while sharing the K/V device reads across grouped heads.
  if constexpr (D == 128 && V == 128 && gqa_group_enabled) {
  // Vector-load eligibility: the group path issues 8-byte vec<T,4> K/V loads
  // (T is 2 bytes at D == V == 128), so every element offset in the K/V
  // indexing must be a multiple of 4 elements and the base pointers 8-byte
  // aligned. simd_lid * qk_per_thread is always a multiple of 4 here; the
  // strides and bases are checked at runtime. Ineligible layouts fall back
  // to the stock per-head path below, which the group path reproduces
  // bit-for-bit by construction (same key order, same reduction tree), so
  // the fallback changes nothing but speed.
  const bool pair_vec_aligned =
      (((k_seq_stride | v_seq_stride | k_head_stride | v_head_stride) & 3) ==
       0) &&
      ((reinterpret_cast<uintptr_t>(keys) |
        reinterpret_cast<uintptr_t>(values)) &
       7) == 0;
  const int gqa_group = (gqa_factor == 6) ? DARKBLOOM_GQA_GROUP_FULL
                    : (gqa_factor == 8) ? DARKBLOOM_GQA_GROUP_SLIDING
                    : 0;
  const bool use_gqa_pair =
      gqa_group >= 2 &&
      tpg.y == 1 && (tpg.x % gqa_group) == 0 &&
      pair_vec_aligned &&
      !has_mask && !do_causal && !has_sinks;
  if (use_gqa_pair) {
    const int group_idx = tid.x;
    const int q_head0 = gqa_group * group_idx;
    if (q_head0 >= int(tpg.x)) {
      return;
    }
    const int n_active = (q_head0 + gqa_group <= int(tpg.x)) ? gqa_group
                          : (int(tpg.x) - q_head0);
    const int kv_head_idx = q_head0 / gqa_factor;

    // When GROUP_SLIDING >= 4, n_active can be 4 but arrays are sized
    // GROUP_MAX (3). Cap the array-loop bound and handle head 3 with
    // explicit variables. When GROUP_SLIDING < 4, n_active <= GROUP_MAX so
    // n_active_arr == n_active and the code is unchanged.
#if DARKBLOOM_GQA_GROUP_SLIDING >= 4
    const int n_active_arr =
        (n_active > DARKBLOOM_GQA_GROUP_MAX) ? DARKBLOOM_GQA_GROUP_MAX
                                             : n_active;
#else
    const int n_active_arr = n_active;
#endif

    // Query pointers and output pointers for each active head.
    const device T* group_qptrs[DARKBLOOM_GQA_GROUP_MAX];
    device T* group_optrs[DARKBLOOM_GQA_GROUP_MAX];
    for (int h = 0; h < n_active_arr; ++h) {
      group_qptrs[h] =
          queries + (q_head0 + h) * D + simd_lid * qk_per_thread;
      group_optrs[h] =
          out + (q_head0 + h) * V + simd_gid * v_per_thread;
    }
#if DARKBLOOM_GQA_GROUP_SLIDING >= 4
    const device T* qptr3 = nullptr;
    device T* optr3 = nullptr;
    if (n_active > DARKBLOOM_GQA_GROUP_MAX) {
      qptr3 = queries + (q_head0 + 3) * D + simd_lid * qk_per_thread;
      optr3 = out + (q_head0 + 3) * V + simd_gid * v_per_thread;
    }
#endif
    const device T* group_keys =
        keys + kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
        simd_lid * qk_per_thread;
    const device T* group_values =
        values + kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
        simd_lid * v_per_thread;

    thread U gq[DARKBLOOM_GQA_GROUP_MAX][qk_per_thread];
    thread U go[DARKBLOOM_GQA_GROUP_MAX][v_per_thread];
    U gmax[DARKBLOOM_GQA_GROUP_MAX];
    U gsum[DARKBLOOM_GQA_GROUP_MAX];
#if DARKBLOOM_GQA_GROUP_SLIDING >= 4
    thread U gq3[qk_per_thread];
    thread U go3[v_per_thread];
    U gmax3;
    U gsum3;
#endif

    for (int h = 0; h < n_active_arr; ++h) {
      for (int j = 0; j < qk_per_thread; ++j)
        gq[h][j] = static_cast<U>(scale) * group_qptrs[h][j];
      for (int j = 0; j < v_per_thread; ++j)
        go[h][j] = 0;
      gmax[h] = Limits<U>::finite_min;
      gsum[h] = 0;
    }
#if DARKBLOOM_GQA_GROUP_SLIDING >= 4
    if (n_active > DARKBLOOM_GQA_GROUP_MAX) {
      for (int j = 0; j < qk_per_thread; ++j)
        gq3[j] = static_cast<U>(scale) * qptr3[j];
      for (int j = 0; j < v_per_thread; ++j)
        go3[j] = 0;
      gmax3 = Limits<U>::finite_min;
      gsum3 = 0;
    }
#endif

    // Two-deep software pipeline, loads hoisted: both positions' K and V
    // rows are read at the top of the trip, then the two positions are
    // accumulated strictly in order (i before i+BN). Per-position FP
    // sequence is character-identical to stock; only load placement moved,
    // and loads have no side effects (no device stores occur in the loop).
    int i = simd_gid;
    for (; i + BN < N; i += 2 * BN) {
      const device T* pipe_keys_b = group_keys + inner_k_stride;
      const device T* pipe_values_b = group_values + inner_v_stride;
      const vec<T, 4> vec_ka =
          *reinterpret_cast<const device vec<T, 4>*>(group_keys);
      const vec<T, 4> vec_kb =
          *reinterpret_cast<const device vec<T, 4>*>(pipe_keys_b);
      U pipe_ka[4];
      U pipe_kb[4];
      pipe_ka[0] = vec_ka.x;
      pipe_ka[1] = vec_ka.y;
      pipe_ka[2] = vec_ka.z;
      pipe_ka[3] = vec_ka.w;
      pipe_kb[0] = vec_kb.x;
      pipe_kb[1] = vec_kb.y;
      pipe_kb[2] = vec_kb.z;
      pipe_kb[3] = vec_kb.w;
      const vec<T, 4> vec_va =
          *reinterpret_cast<const device vec<T, 4>*>(group_values);
      const vec<T, 4> vec_vb =
          *reinterpret_cast<const device vec<T, 4>*>(pipe_values_b);
      const T pipe_va0 = vec_va.x;
      const T pipe_va1 = vec_va.y;
      const T pipe_va2 = vec_va.z;
      const T pipe_va3 = vec_va.w;
      const T pipe_vb0 = vec_vb.x;
      const T pipe_vb1 = vec_vb.y;
      const T pipe_vb2 = vec_vb.z;
      const T pipe_vb3 = vec_vb.w;

      // Position a: per-head score and update. Each head's FP sequence
      // is identical to stock (same q*k dot-product chain, same FMA order
      // for the output accumulate). K/V are shared reads.
      for (int h = 0; h < n_active_arr; ++h) {
        U score = 0;
        score += gq[h][0] * pipe_ka[0];
        score += gq[h][1] * pipe_ka[1];
        score += gq[h][2] * pipe_ka[2];
        score += gq[h][3] * pipe_ka[3];
        score = simd_sum(score);

        U new_max = max(gmax[h], score);
        U factor;
        DARKBLOOM_RESCALE_FACTOR(factor, gmax[h] - new_max);
        U exp_score = fast::exp(score - new_max);
        gmax[h] = new_max;
        gsum[h] = gsum[h] * factor + exp_score;

        go[h][0] = go[h][0] * factor + exp_score * pipe_va0;
        go[h][1] = go[h][1] * factor + exp_score * pipe_va1;
        go[h][2] = go[h][2] * factor + exp_score * pipe_va2;
        go[h][3] = go[h][3] * factor + exp_score * pipe_va3;
      }
#if DARKBLOOM_GQA_GROUP_SLIDING >= 4
      if (n_active > DARKBLOOM_GQA_GROUP_MAX) {
        U score = 0;
        score += gq3[0] * pipe_ka[0];
        score += gq3[1] * pipe_ka[1];
        score += gq3[2] * pipe_ka[2];
        score += gq3[3] * pipe_ka[3];
        score = simd_sum(score);

        U new_max = max(gmax3, score);
        U factor;
        DARKBLOOM_RESCALE_FACTOR(factor, gmax3 - new_max);
        U exp_score = fast::exp(score - new_max);
        gmax3 = new_max;
        gsum3 = gsum3 * factor + exp_score;

        go3[0] = go3[0] * factor + exp_score * pipe_va0;
        go3[1] = go3[1] * factor + exp_score * pipe_va1;
        go3[2] = go3[2] * factor + exp_score * pipe_va2;
        go3[3] = go3[3] * factor + exp_score * pipe_va3;
      }
#endif

      // Position b: same per-head pattern with pipe_kb/vb.
      for (int h = 0; h < n_active_arr; ++h) {
        U score = 0;
        score += gq[h][0] * pipe_kb[0];
        score += gq[h][1] * pipe_kb[1];
        score += gq[h][2] * pipe_kb[2];
        score += gq[h][3] * pipe_kb[3];
        score = simd_sum(score);

        U new_max = max(gmax[h], score);
        U factor;
        DARKBLOOM_RESCALE_FACTOR(factor, gmax[h] - new_max);
        U exp_score = fast::exp(score - new_max);
        gmax[h] = new_max;
        gsum[h] = gsum[h] * factor + exp_score;

        go[h][0] = go[h][0] * factor + exp_score * pipe_vb0;
        go[h][1] = go[h][1] * factor + exp_score * pipe_vb1;
        go[h][2] = go[h][2] * factor + exp_score * pipe_vb2;
        go[h][3] = go[h][3] * factor + exp_score * pipe_vb3;
      }
#if DARKBLOOM_GQA_GROUP_SLIDING >= 4
      if (n_active > DARKBLOOM_GQA_GROUP_MAX) {
        U score = 0;
        score += gq3[0] * pipe_kb[0];
        score += gq3[1] * pipe_kb[1];
        score += gq3[2] * pipe_kb[2];
        score += gq3[3] * pipe_kb[3];
        score = simd_sum(score);

        U new_max = max(gmax3, score);
        U factor;
        DARKBLOOM_RESCALE_FACTOR(factor, gmax3 - new_max);
        U exp_score = fast::exp(score - new_max);
        gmax3 = new_max;
        gsum3 = gsum3 * factor + exp_score;

        go3[0] = go3[0] * factor + exp_score * pipe_vb0;
        go3[1] = go3[1] * factor + exp_score * pipe_vb1;
        go3[2] = go3[2] * factor + exp_score * pipe_vb2;
        go3[3] = go3[3] * factor + exp_score * pipe_vb3;
      }
#endif

      group_keys += 2 * inner_k_stride;
      group_values += 2 * inner_v_stride;
    }
    if (i < N) {
      const vec<T, 4> vec_kt =
          *reinterpret_cast<const device vec<T, 4>*>(group_keys);
      const vec<T, 4> vec_vt =
          *reinterpret_cast<const device vec<T, 4>*>(group_values);
      U kt[4];
      kt[0] = vec_kt.x;
      kt[1] = vec_kt.y;
      kt[2] = vec_kt.z;
      kt[3] = vec_kt.w;
      const T pipe_va0 = vec_vt.x;
      const T pipe_va1 = vec_vt.y;
      const T pipe_va2 = vec_vt.z;
      const T pipe_va3 = vec_vt.w;

      for (int h = 0; h < n_active_arr; ++h) {
        U score = 0;
        score += gq[h][0] * kt[0];
        score += gq[h][1] * kt[1];
        score += gq[h][2] * kt[2];
        score += gq[h][3] * kt[3];
        score = simd_sum(score);

        U new_max = max(gmax[h], score);
        U factor;
        DARKBLOOM_RESCALE_FACTOR(factor, gmax[h] - new_max);
        U exp_score = fast::exp(score - new_max);
        gmax[h] = new_max;
        gsum[h] = gsum[h] * factor + exp_score;

        go[h][0] = go[h][0] * factor + exp_score * pipe_va0;
        go[h][1] = go[h][1] * factor + exp_score * pipe_va1;
        go[h][2] = go[h][2] * factor + exp_score * pipe_va2;
        go[h][3] = go[h][3] * factor + exp_score * pipe_va3;
      }
#if DARKBLOOM_GQA_GROUP_SLIDING >= 4
      if (n_active > DARKBLOOM_GQA_GROUP_MAX) {
        U score = 0;
        score += gq3[0] * kt[0];
        score += gq3[1] * kt[1];
        score += gq3[2] * kt[2];
        score += gq3[3] * kt[3];
        score = simd_sum(score);

        U new_max = max(gmax3, score);
        U factor;
        DARKBLOOM_RESCALE_FACTOR(factor, gmax3 - new_max);
        U exp_score = fast::exp(score - new_max);
        gmax3 = new_max;
        gsum3 = gsum3 * factor + exp_score;

        go3[0] = go3[0] * factor + exp_score * pipe_va0;
        go3[1] = go3[1] * factor + exp_score * pipe_va1;
        go3[2] = go3[2] * factor + exp_score * pipe_va2;
        go3[3] = go3[3] * factor + exp_score * pipe_va3;
      }
#endif
    }

    // Exchange epilogue. Each head uses 2 planes. The plane base for head h
    // is h * 2 * pair_plane_size. Only the first n_active_arr * 2 planes are
    // touched. The additive head-bank offset changes no producer/consumer
    // pairing or simd_sum tree.
    constexpr int pair_planes = 2;
    constexpr int pair_plane_size = BN * BD;
    if (simd_lid == 0) {
      for (int h = 0; h < n_active_arr; ++h) {
        max_scores[h * BN + simd_gid] = gmax[h];
        sum_exp_scores[h * BN + simd_gid] = gsum[h];
      }
    }
    for (int i = 0; i < pair_planes; ++i) {
      for (int h = 0; h < n_active_arr; ++h) {
        outputs[(h * pair_planes + i) * pair_plane_size +
                simd_lid * BD + simd_gid] = go[h][i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    U g_global_max[DARKBLOOM_GQA_GROUP_MAX];
    U g_global_factor[DARKBLOOM_GQA_GROUP_MAX];
    for (int h = 0; h < n_active_arr; ++h) {
      U local_max = max_scores[h * BN + simd_lid];
      g_global_max[h] = simd_max(local_max);
      g_global_factor[h] = fast::exp(local_max - g_global_max[h]);
      gsum[h] =
          simd_sum(sum_exp_scores[h * BN + simd_lid] * g_global_factor[h]);
    }

    for (int i = 0; i < pair_planes; ++i) {
      for (int h = 0; h < n_active_arr; ++h) {
        U acc = simd_sum(
            outputs[(h * pair_planes + i) * pair_plane_size +
                    simd_gid * BD + simd_lid] *
            g_global_factor[h]);
        go[h][i] = gsum[h] == 0 ? acc : (acc / gsum[h]);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < pair_planes; ++i) {
      for (int h = 0; h < n_active_arr; ++h) {
        outputs[(h * pair_planes + i) * pair_plane_size +
                simd_lid * BD + simd_gid] = go[h][pair_planes + i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < pair_planes; ++i) {
      for (int h = 0; h < n_active_arr; ++h) {
        U acc = simd_sum(
            outputs[(h * pair_planes + i) * pair_plane_size +
                    simd_gid * BD + simd_lid] *
            g_global_factor[h]);
        go[h][pair_planes + i] =
            gsum[h] == 0 ? acc : (acc / gsum[h]);
      }
    }

#if DARKBLOOM_GQA_GROUP_SLIDING >= 4
    // Staggered exchange for head 3: reuse planes 0,1 and max_scores/
    // sum_exp_scores slots 0,1 (head 0's, now free). The exchange for
    // each head is independent (separate max/sum broadcast, separate
    // simd_sum reduction, separate factor/divide), so processing head 3
    // after heads 0-2 is bit-exact.
    if (n_active > DARKBLOOM_GQA_GROUP_MAX) {
      // Ensure all reads from the 3-head exchange are complete.
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (simd_lid == 0) {
        max_scores[0 * BN + simd_gid] = gmax3;
        sum_exp_scores[0 * BN + simd_gid] = gsum3;
      }
      for (int i = 0; i < pair_planes; ++i) {
        outputs[i * pair_plane_size + simd_lid * BD + simd_gid] = go3[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      U local_max3 = max_scores[0 * BN + simd_lid];
      U g_global_max3 = simd_max(local_max3);
      U g_global_factor3 = fast::exp(local_max3 - g_global_max3);
      gsum3 = simd_sum(sum_exp_scores[0 * BN + simd_lid] * g_global_factor3);

      for (int i = 0; i < pair_planes; ++i) {
        U acc = simd_sum(
            outputs[i * pair_plane_size + simd_gid * BD + simd_lid] *
            g_global_factor3);
        go3[i] = gsum3 == 0 ? acc : (acc / gsum3);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = 0; i < pair_planes; ++i) {
        outputs[i * pair_plane_size + simd_lid * BD + simd_gid] =
            go3[pair_planes + i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = 0; i < pair_planes; ++i) {
        U acc = simd_sum(
            outputs[i * pair_plane_size + simd_gid * BD + simd_lid] *
            g_global_factor3);
        go3[pair_planes + i] = gsum3 == 0 ? acc : (acc / gsum3);
      }
    }
#endif

    if (simd_lid == 0) {
      for (int h = 0; h < n_active_arr; ++h) {
        for (int i = 0; i < v_per_thread; ++i) {
          group_optrs[h][i] = static_cast<T>(go[h][i]);
        }
      }
#if DARKBLOOM_GQA_GROUP_SLIDING >= 4
      if (n_active > DARKBLOOM_GQA_GROUP_MAX) {
        for (int i = 0; i < v_per_thread; ++i) {
          optr3[i] = static_cast<T>(go3[i]);
        }
      }
#endif
    }
    return;
  }
  }

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  // Adjust positions
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;
  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx % num_q_heads]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Read the key
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor;
      DARKBLOOM_RESCALE_FACTOR(factor, max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
    if (bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // `v_planes` is a compile-time constant, so this selector is folded and only
  // one arm survives in each specialization; the barriers inside it are never
  // reached divergently.
  U factor;
  if (v_planes > 1) {
    // Widened exchange. Elements are combined in groups of `v_planes`; within
    // a group every element owns its own plane, so neither the RAW hazard
    // (write then read) nor the WAR hazard (read then rewrite) recurs inside
    // the group. The group's single rendezvous also absorbs the max/sum
    // publish, because the plane stores depend only on registers (`o[]` is
    // final here) and target a threadgroup array disjoint from
    // max_scores/sum_exp_scores, so hoisting them above that barrier
    // introduces no dependency.
    //
    // At D = V = 128 (v_per_thread = 4): 8 barriers -> 3 at PLANES=2 (two
    // groups of two), -> 1 at PLANES=4 (one group of four).
    //
    // Exactness, per lane. Baseline: lane l of simdgroup g writes o[i] to
    // outputs[l * BD + g] and reads outputs[g * BD + l], so the simd_sum over
    // the 32 lanes of simdgroup g reduces, in lane order l = 0..31, the o[i]
    // produced by lane g of simdgroup l. Widened: the same lane writes
    // o[base+p] to outputs[p * (BN * BD) + l * BD + g] and reads
    // outputs[p * (BN * BD) + g * BD + l]. The plane base p * (BN * BD) is the
    // same additive constant for the writer and the reader of a given element,
    // so the producer/consumer pairing, the lane ordering and the reduction
    // tree are identical; only the base address differs. `factor`, the divide
    // and the `sum_exp_score == 0` guard are untouched. No reassociation, no
    // rounding boundary moved.
    if (simd_lid == 0) {
      max_scores[simd_gid] = max_score;
      sum_exp_scores[simd_gid] = sum_exp_score;
    }
    for (int i = 0; i < v_planes; i++) {
      outputs[i * (BN * BD) + simd_lid * BD + simd_gid] = o[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = max_scores[simd_lid];
    U new_max = simd_max(max_score);
    factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

    for (int i = 0; i < v_planes; i++) {
      U acc =
          simd_sum(outputs[i * (BN * BD) + simd_gid * BD + simd_lid] * factor);
      o[i] = sum_exp_score == 0 ? acc : (acc / sum_exp_score);
    }
    // Only entered when v_per_thread exceeds v_planes (PLANES=2 at D >= 96,
    // PLANES=4 at D = 256). Each extra group costs one WAR plus one RAW.
    for (int base = v_planes; base < v_per_thread; base += v_planes) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = 0; i < v_planes && base + i < v_per_thread; i++) {
        outputs[i * (BN * BD) + simd_lid * BD + simd_gid] = o[base + i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = 0; i < v_planes && base + i < v_per_thread; i++) {
        U acc = simd_sum(
            outputs[i * (BN * BD) + simd_gid * BD + simd_lid] * factor);
        o[base + i] = sum_exp_score == 0 ? acc : (acc / sum_exp_score);
      }
    }
  } else {
    // Upstream shape, preserved verbatim so PLANES=1 is a true control.
    // First let's communicate the max and sum_exp
    if (simd_lid == 0) {
      max_scores[simd_gid] = max_score;
      sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = max_scores[simd_lid];
    U new_max = simd_max(max_score);
    factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

    // Now we need to aggregate all the outputs. The trailing barrier only
    // protects the exchange plane's reuse by the NEXT iteration, so the last
    // iteration's trailing barrier guards nothing and is skipped; every
    // exchanged value, slot, and reduction is unchanged.
    for (int i = 0; i < v_per_thread; i++) {
      outputs[simd_lid * BD + simd_gid] = o[i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
      o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
      if (i + 1 < v_per_thread) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    const device bool* bmask [[buffer(13), function_constant(bool_mask)]],
    const device T* fmask [[buffer(14), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(15), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(16), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(17), function_constant(has_mask)]],
    const device T* sinks [[buffer(18), function_constant(has_sinks)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  // Adjust positions
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
  const int q_offset =
      query_transposed ? num_q_heads * q_seq_idx + q_batch_head_idx : o_offset;

  queries += q_offset * D + simd_lid * qk_per_thread;

  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
  keys += kv_batch_head_idx * k_head_stride + block_idx * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_batch_head_idx * v_head_stride + block_idx * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  // Read the query
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && block_idx == 0) {
    max_score = static_cast<U>(sinks[q_head_idx]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = block_idx; i < N; i += blocks) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - q_seq_len + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Compute the i-th score
      U score = 0;
      for (int i = 0; i < qk_per_thread; i++) {
        score += q[i] * keys[i];
      }
      score = simd_sum(score);

      if (float_mask) {
        score += fmask[0];
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int i = 0; i < v_per_thread; i++) {
        o[i] = o[i] * factor + exp_score * values[i];
      }
    }

    // Move the pointers to the next kv
    keys += blocks * int(k_seq_stride);
    values += blocks * int(v_seq_stride);
    if (bool_mask) {
      bmask += blocks * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += blocks * mask_kv_seq_stride;
    }
  }

  // Write the sum and max and outputs
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }

  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<T>(o[i]);
  }
}

template <typename T, int D, int PLANES = DARKBLOOM_AOT_SDPA_2PASS_PLANES>
[[kernel]] void sdpa_vector_2pass_2(
    const device T* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& blocks [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  typedef float U;

  thread U o[elem_per_thread] = {0};
  constexpr int o_planes = PLANES < elem_per_thread ? PLANES : elem_per_thread;
  threadgroup U outputs[o_planes * BN * BD];

  // Adjust positions
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int q_offset = head_idx * tpg.y + q_seq_idx;
  partials += q_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += q_offset * blocks;
  maxs += q_offset * blocks;
  out += q_offset * D + simd_gid * elem_per_thread;

  // Set defaults
  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;

  // Reduce the max
  for (int b = 0; b < blocks / BN; ++b) {
    max_score = max(max_score, maxs[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);

  // Reduce the d
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sums[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);

  // Reduce the sum exp and partials
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_gid] - max_score);

    // Update the output accumulator
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(partials[i]);
    }
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }

  if (o_planes > 1) {
    // Same widening as `sdpa_vector`: one plane per element removes the
    // RAW/WAR pair that only existed because a single plane was recycled.
    // 7 barriers -> 1 at D = 128. Unlike `sdpa_vector` there is no earlier
    // threadgroup publish to merge with, so one rendezvous remains.
    //
    // Exactness: lane l of simdgroup g still writes o[i] to slot (l * BD + g)
    // of plane i and reads slot (g * BD + l) of plane i, so each simd_sum
    // reduces the same 32 values from the same producers in the same order.
    // Only the plane base address changes.
    for (int i = 0; i < o_planes; i++) {
      outputs[i * (BN * BD) + simd_lid * BD + simd_gid] = o[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < o_planes; i++) {
      U acc = simd_sum(outputs[i * (BN * BD) + simd_gid * BD + simd_lid]);
      o[i] = sum_exp_score == 0 ? acc : (acc / sum_exp_score);
    }
    for (int base = o_planes; base < elem_per_thread; base += o_planes) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = 0; i < o_planes && base + i < elem_per_thread; i++) {
        outputs[i * (BN * BD) + simd_lid * BD + simd_gid] = o[base + i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = 0; i < o_planes && base + i < elem_per_thread; i++) {
        U acc = simd_sum(outputs[i * (BN * BD) + simd_gid * BD + simd_lid]);
        o[base + i] = sum_exp_score == 0 ? acc : (acc / sum_exp_score);
      }
    }
  } else {
    // Use shared memory to transpose and reduce the final block. As in
    // sdpa_vector above, the trailing barrier only protects the exchange
    // plane's reuse by the NEXT iteration, so the last iteration's trailing
    // barrier guards nothing and is skipped; no value, order, or write set
    // changes.
    for (int i = 0; i < elem_per_thread; i++) {
      outputs[simd_lid * BD + simd_gid] = o[i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
      o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
      if (i + 1 < elem_per_thread) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
