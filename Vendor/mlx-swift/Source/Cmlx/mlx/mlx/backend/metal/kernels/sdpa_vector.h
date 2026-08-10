// Copyright © 2024 Apple Inc.

#include <metal_simdgroup>

using namespace metal;

#ifndef DARKBLOOM_AOT_SDPA_PLANES
#define DARKBLOOM_AOT_SDPA_PLANES 4
#endif

#ifndef DARKBLOOM_AOT_SDPA_2PASS_PLANES
#define DARKBLOOM_AOT_SDPA_2PASS_PLANES 1
#endif

#ifndef DARKBLOOM_GQA_PAIR_HEADS
#define DARKBLOOM_GQA_PAIR_HEADS 2
#endif

#ifndef DARKBLOOM_ALPHASKIP
#define DARKBLOOM_ALPHASKIP 1
#endif

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

  constexpr int v_planes = PLANES < v_per_thread ? PLANES : v_per_thread;
  constexpr int exchange_planes =
      (D == 128 && V == 128 && DARKBLOOM_GQA_PAIR_HEADS == 2)
      ? 4
      : v_planes;
  threadgroup U outputs[exchange_planes * BN * BD];
  threadgroup U max_scores[DARKBLOOM_GQA_PAIR_HEADS * BN];
  threadgroup U sum_exp_scores[DARKBLOOM_GQA_PAIR_HEADS * BN];

  if constexpr (D == 128 && V == 128 && DARKBLOOM_GQA_PAIR_HEADS == 2) {
  const bool pair_vec_aligned =
      (((k_seq_stride | v_seq_stride | k_head_stride | v_head_stride) & 3) ==
       0) &&
      ((reinterpret_cast<uintptr_t>(keys) |
        reinterpret_cast<uintptr_t>(values)) &
       7) == 0;
  const bool use_gqa_pair =
      (gqa_factor == 8 || gqa_factor == 6) &&
      tpg.y == 1 && (tpg.x % 2) == 0 &&
      pair_vec_aligned &&
      !has_mask && !do_causal && !has_sinks;
  if (use_gqa_pair) {
    const int pair_idx = tid.x;
    const int q_head0 = 2 * pair_idx;
    if (q_head0 >= int(tpg.x)) {
      return;
    }
    const int q_head1 = q_head0 + 1;
    const int kv_head_idx = q_head0 / gqa_factor;

    const device T* pair_query0 =
        queries + q_head0 * D + simd_lid * qk_per_thread;
    const device T* pair_query1 =
        queries + q_head1 * D + simd_lid * qk_per_thread;
    const device T* pair_keys =
        keys + kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
        simd_lid * qk_per_thread;
    const device T* pair_values =
        values + kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
        simd_lid * v_per_thread;
    device T* pair_out0 =
        out + q_head0 * V + simd_gid * v_per_thread;
    device T* pair_out1 =
        out + q_head1 * V + simd_gid * v_per_thread;

    thread U pair_q0[qk_per_thread];
    thread U pair_q1[qk_per_thread];
    thread U pair_k[qk_per_thread];
    thread U pair_o0[v_per_thread];
    thread U pair_o1[v_per_thread];

    for (int j = 0; j < qk_per_thread; ++j) {
      pair_q0[j] = static_cast<U>(scale) * pair_query0[j];
      pair_q1[j] = static_cast<U>(scale) * pair_query1[j];
    }
    for (int j = 0; j < v_per_thread; ++j) {
      pair_o0[j] = 0;
      pair_o1[j] = 0;
    }

    U pair_max0 = Limits<U>::finite_min;
    U pair_max1 = Limits<U>::finite_min;
    U pair_sum0 = 0;
    U pair_sum1 = 0;

    int i = simd_gid;
    for (; i + BN < N; i += 2 * BN) {
      const device T* pipe_keys_b = pair_keys + inner_k_stride;
      const device T* pipe_values_b = pair_values + inner_v_stride;
      const vec<T, 4> vec_ka =
          *reinterpret_cast<const device vec<T, 4>*>(pair_keys);
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
          *reinterpret_cast<const device vec<T, 4>*>(pair_values);
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

      U pair_score0 = 0;
      U pair_score1 = 0;
      pair_score0 += pair_q0[0] * pipe_ka[0];
      pair_score1 += pair_q1[0] * pipe_ka[0];
      pair_score0 += pair_q0[1] * pipe_ka[1];
      pair_score1 += pair_q1[1] * pipe_ka[1];
      pair_score0 += pair_q0[2] * pipe_ka[2];
      pair_score1 += pair_q1[2] * pipe_ka[2];
      pair_score0 += pair_q0[3] * pipe_ka[3];
      pair_score1 += pair_q1[3] * pipe_ka[3];
      pair_score0 = simd_sum(pair_score0);
      pair_score1 = simd_sum(pair_score1);

      U pair_new_max0 = max(pair_max0, pair_score0);
      U pair_new_max1 = max(pair_max1, pair_score1);
      U pair_factor0;
      U pair_factor1;
      DARKBLOOM_RESCALE_FACTOR(pair_factor0, pair_max0 - pair_new_max0);
      DARKBLOOM_RESCALE_FACTOR(pair_factor1, pair_max1 - pair_new_max1);
      U pair_exp0 = fast::exp(pair_score0 - pair_new_max0);
      U pair_exp1 = fast::exp(pair_score1 - pair_new_max1);

      pair_max0 = pair_new_max0;
      pair_max1 = pair_new_max1;
      pair_sum0 = pair_sum0 * pair_factor0 + pair_exp0;
      pair_sum1 = pair_sum1 * pair_factor1 + pair_exp1;

      pair_o0[0] = pair_o0[0] * pair_factor0 + pair_exp0 * pipe_va0;
      pair_o1[0] = pair_o1[0] * pair_factor1 + pair_exp1 * pipe_va0;
      pair_o0[1] = pair_o0[1] * pair_factor0 + pair_exp0 * pipe_va1;
      pair_o1[1] = pair_o1[1] * pair_factor1 + pair_exp1 * pipe_va1;
      pair_o0[2] = pair_o0[2] * pair_factor0 + pair_exp0 * pipe_va2;
      pair_o1[2] = pair_o1[2] * pair_factor1 + pair_exp1 * pipe_va2;
      pair_o0[3] = pair_o0[3] * pair_factor0 + pair_exp0 * pipe_va3;
      pair_o1[3] = pair_o1[3] * pair_factor1 + pair_exp1 * pipe_va3;


      U pipeb_score0 = 0;
      U pipeb_score1 = 0;
      pipeb_score0 += pair_q0[0] * pipe_kb[0];
      pipeb_score1 += pair_q1[0] * pipe_kb[0];
      pipeb_score0 += pair_q0[1] * pipe_kb[1];
      pipeb_score1 += pair_q1[1] * pipe_kb[1];
      pipeb_score0 += pair_q0[2] * pipe_kb[2];
      pipeb_score1 += pair_q1[2] * pipe_kb[2];
      pipeb_score0 += pair_q0[3] * pipe_kb[3];
      pipeb_score1 += pair_q1[3] * pipe_kb[3];
      pipeb_score0 = simd_sum(pipeb_score0);
      pipeb_score1 = simd_sum(pipeb_score1);

      U pipeb_new_max0 = max(pair_max0, pipeb_score0);
      U pipeb_new_max1 = max(pair_max1, pipeb_score1);
      U pipeb_factor0;
      U pipeb_factor1;
      DARKBLOOM_RESCALE_FACTOR(pipeb_factor0, pair_max0 - pipeb_new_max0);
      DARKBLOOM_RESCALE_FACTOR(pipeb_factor1, pair_max1 - pipeb_new_max1);
      U pipeb_exp0 = fast::exp(pipeb_score0 - pipeb_new_max0);
      U pipeb_exp1 = fast::exp(pipeb_score1 - pipeb_new_max1);

      pair_max0 = pipeb_new_max0;
      pair_max1 = pipeb_new_max1;
      pair_sum0 = pair_sum0 * pipeb_factor0 + pipeb_exp0;
      pair_sum1 = pair_sum1 * pipeb_factor1 + pipeb_exp1;

      pair_o0[0] = pair_o0[0] * pipeb_factor0 + pipeb_exp0 * pipe_vb0;
      pair_o1[0] = pair_o1[0] * pipeb_factor1 + pipeb_exp1 * pipe_vb0;
      pair_o0[1] = pair_o0[1] * pipeb_factor0 + pipeb_exp0 * pipe_vb1;
      pair_o1[1] = pair_o1[1] * pipeb_factor1 + pipeb_exp1 * pipe_vb1;
      pair_o0[2] = pair_o0[2] * pipeb_factor0 + pipeb_exp0 * pipe_vb2;
      pair_o1[2] = pair_o1[2] * pipeb_factor1 + pipeb_exp1 * pipe_vb2;
      pair_o0[3] = pair_o0[3] * pipeb_factor0 + pipeb_exp0 * pipe_vb3;
      pair_o1[3] = pair_o1[3] * pipeb_factor1 + pipeb_exp1 * pipe_vb3;

      pair_keys += 2 * inner_k_stride;
      pair_values += 2 * inner_v_stride;
    }
    if (i < N) {
      const vec<T, 4> vec_kt =
          *reinterpret_cast<const device vec<T, 4>*>(pair_keys);
      const vec<T, 4> vec_vt =
          *reinterpret_cast<const device vec<T, 4>*>(pair_values);
      pair_k[0] = vec_kt.x;
      pair_k[1] = vec_kt.y;
      pair_k[2] = vec_kt.z;
      pair_k[3] = vec_kt.w;
      const T pipe_va0 = vec_vt.x;
      const T pipe_va1 = vec_vt.y;
      const T pipe_va2 = vec_vt.z;
      const T pipe_va3 = vec_vt.w;

      U pair_score0 = 0;
      U pair_score1 = 0;
      pair_score0 += pair_q0[0] * pair_k[0];
      pair_score1 += pair_q1[0] * pair_k[0];
      pair_score0 += pair_q0[1] * pair_k[1];
      pair_score1 += pair_q1[1] * pair_k[1];
      pair_score0 += pair_q0[2] * pair_k[2];
      pair_score1 += pair_q1[2] * pair_k[2];
      pair_score0 += pair_q0[3] * pair_k[3];
      pair_score1 += pair_q1[3] * pair_k[3];
      pair_score0 = simd_sum(pair_score0);
      pair_score1 = simd_sum(pair_score1);

      U pair_new_max0 = max(pair_max0, pair_score0);
      U pair_new_max1 = max(pair_max1, pair_score1);
      U pair_factor0;
      U pair_factor1;
      DARKBLOOM_RESCALE_FACTOR(pair_factor0, pair_max0 - pair_new_max0);
      DARKBLOOM_RESCALE_FACTOR(pair_factor1, pair_max1 - pair_new_max1);
      U pair_exp0 = fast::exp(pair_score0 - pair_new_max0);
      U pair_exp1 = fast::exp(pair_score1 - pair_new_max1);

      pair_max0 = pair_new_max0;
      pair_max1 = pair_new_max1;
      pair_sum0 = pair_sum0 * pair_factor0 + pair_exp0;
      pair_sum1 = pair_sum1 * pair_factor1 + pair_exp1;

      pair_o0[0] = pair_o0[0] * pair_factor0 + pair_exp0 * pipe_va0;
      pair_o1[0] = pair_o1[0] * pair_factor1 + pair_exp1 * pipe_va0;
      pair_o0[1] = pair_o0[1] * pair_factor0 + pair_exp0 * pipe_va1;
      pair_o1[1] = pair_o1[1] * pair_factor1 + pair_exp1 * pipe_va1;
      pair_o0[2] = pair_o0[2] * pair_factor0 + pair_exp0 * pipe_va2;
      pair_o1[2] = pair_o1[2] * pair_factor1 + pair_exp1 * pipe_va2;
      pair_o0[3] = pair_o0[3] * pair_factor0 + pair_exp0 * pipe_va3;
      pair_o1[3] = pair_o1[3] * pair_factor1 + pair_exp1 * pipe_va3;

    }

    constexpr int pair_planes = 2;
    constexpr int pair_plane_size = BN * BD;
    if (simd_lid == 0) {
      max_scores[simd_gid] = pair_max0;
      max_scores[BN + simd_gid] = pair_max1;
      sum_exp_scores[simd_gid] = pair_sum0;
      sum_exp_scores[BN + simd_gid] = pair_sum1;
    }
    for (int i = 0; i < pair_planes; ++i) {
      outputs[i * pair_plane_size + simd_lid * BD + simd_gid] = pair_o0[i];
      outputs[
          (pair_planes + i) * pair_plane_size + simd_lid * BD + simd_gid] =
          pair_o1[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    pair_max0 = max_scores[simd_lid];
    pair_max1 = max_scores[BN + simd_lid];
    U pair_global_max0 = simd_max(pair_max0);
    U pair_global_max1 = simd_max(pair_max1);
    U pair_global_factor0 = fast::exp(pair_max0 - pair_global_max0);
    U pair_global_factor1 = fast::exp(pair_max1 - pair_global_max1);
    pair_sum0 =
        simd_sum(sum_exp_scores[simd_lid] * pair_global_factor0);
    pair_sum1 =
        simd_sum(sum_exp_scores[BN + simd_lid] * pair_global_factor1);

    for (int i = 0; i < pair_planes; ++i) {
      U acc0 = simd_sum(
          outputs[i * pair_plane_size + simd_gid * BD + simd_lid] *
          pair_global_factor0);
      U acc1 = simd_sum(
          outputs[
              (pair_planes + i) * pair_plane_size +
              simd_gid * BD + simd_lid] *
          pair_global_factor1);
      pair_o0[i] = pair_sum0 == 0 ? acc0 : (acc0 / pair_sum0);
      pair_o1[i] = pair_sum1 == 0 ? acc1 : (acc1 / pair_sum1);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < pair_planes; ++i) {
      outputs[i * pair_plane_size + simd_lid * BD + simd_gid] =
          pair_o0[pair_planes + i];
      outputs[
          (pair_planes + i) * pair_plane_size + simd_lid * BD + simd_gid] =
          pair_o1[pair_planes + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < pair_planes; ++i) {
      U acc0 = simd_sum(
          outputs[i * pair_plane_size + simd_gid * BD + simd_lid] *
          pair_global_factor0);
      U acc1 = simd_sum(
          outputs[
              (pair_planes + i) * pair_plane_size +
              simd_gid * BD + simd_lid] *
          pair_global_factor1);
      pair_o0[pair_planes + i] =
          pair_sum0 == 0 ? acc0 : (acc0 / pair_sum0);
      pair_o1[pair_planes + i] =
          pair_sum1 == 0 ? acc1 : (acc1 / pair_sum1);
    }

    if (simd_lid == 0) {
      for (int i = 0; i < v_per_thread; ++i) {
        pair_out0[i] = static_cast<T>(pair_o0[i]);
        pair_out1[i] = static_cast<T>(pair_o1[i]);
      }
    }
    return;
  }
  }

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

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
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      U new_max = max(max_score, score);
      U factor;
      DARKBLOOM_RESCALE_FACTOR(factor, max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    keys += inner_k_stride;
    values += inner_v_stride;
    if (bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }


  U factor;
  if (v_planes > 1) {
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
    if (simd_lid == 0) {
      max_scores[simd_gid] = max_score;
      sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = max_scores[simd_lid];
    U new_max = simd_max(max_score);
    factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

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

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && block_idx == 0) {
    max_score = static_cast<U>(sinks[q_head_idx]);
    sum_exp_score = 1;
  }

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
      U score = 0;
      for (int i = 0; i < qk_per_thread; i++) {
        score += q[i] * keys[i];
      }
      score = simd_sum(score);

      if (float_mask) {
        score += fmask[0];
      }

      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      for (int i = 0; i < v_per_thread; i++) {
        o[i] = o[i] * factor + exp_score * values[i];
      }
    }

    keys += blocks * int(k_seq_stride);
    values += blocks * int(v_seq_stride);
    if (bool_mask) {
      bmask += blocks * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += blocks * mask_kv_seq_stride;
    }
  }

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

  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int q_offset = head_idx * tpg.y + q_seq_idx;
  partials += q_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += q_offset * blocks;
  maxs += q_offset * blocks;
  out += q_offset * D + simd_gid * elem_per_thread;

  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;

  for (int b = 0; b < blocks / BN; ++b) {
    max_score = max(max_score, maxs[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);

  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sums[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);

  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_gid] - max_score);

    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(partials[i]);
    }
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }

  if (o_planes > 1) {
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

  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
