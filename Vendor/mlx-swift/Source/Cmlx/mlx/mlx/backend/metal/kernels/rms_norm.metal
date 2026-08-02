// Copyright © 2024 Apple Inc.

#include <metal_common>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

constant bool has_w [[function_constant(20)]];

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void rms_single_row(
    const device T* x,
    const device T* w,
    device T* out,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint grid_size [[threads_per_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;

  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  // Laguna BF16 single-row 2048 specialization. The grid-width guard keeps
  // the 512-row prefill launch on the generic path: one 2048-wide row is
  // exactly 512 threads at RMS_N_READS=4, while prefill dispatches 512 of
  // those threadgroups as a 262144-thread grid. The fixed row has sixteen
  // simdgroup partials. Write those to slots 0...15 while lanes 16...31 of
  // simdgroup zero initialize the disjoint unused slots, then rendezvous once.
  // This produces the exact 32-lane second-reduction input of the generic
  // path while removing its preceding zero-fill barrier. The square order,
  // simd reductions, precise rsqrt, BF16 cast point, and weight multiply are
  // unchanged.
  if constexpr (metal::is_same_v<T, bfloat16_t> && N_READS == 4) {
    if (axis_size == 2048 && grid_size == 512 && w_stride == 1) {
      constexpr uint laguna_simdgroups = 16;
      const device T* row_x =
          x + gid * size_t(axis_size) + lid * N_READS;
      const device T* row_w = w + lid * N_READS;
      device T* row_out =
          out + gid * size_t(axis_size) + lid * N_READS;

      float acc = 0;
      float xcache[N_READS];
      for (int i = 0; i < N_READS; i++) {
        float xi = row_x[i];
        xcache[i] = xi;
        acc += xi * xi;
      }
      acc = simd_sum(acc);

      if (simd_group_id == 0 && simd_lane_id >= laguna_simdgroups) {
        local_sums[simd_lane_id] = 0;
      }
      if (simd_lane_id == 0) {
        local_sums[simd_group_id] = acc;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
          local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (int i = 0; i < N_READS; i++) {
        row_out[i] =
            row_w[i] * static_cast<T>(xcache[i] * local_inv_mean[0]);
      }
      return;
    }
  }

  // Generic RMSNorm path.
  float acc = 0;
  x += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  // Upstream ml-explore/mlx PR #3754: cache the row values read during the
  // accumulation pass so the output pass does not re-read the same device
  // bytes. Pure load elimination: the cached float is the exact bf16->float
  // promotion the output expression performs anyway (T * float promotes T
  // first), device memory is immutable during the dispatch, and every
  // arithmetic op, its order, and all barriers are unchanged -- bit-exact.
  float xcache[N_READS];
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      float xi = x[i];
      xcache[i] = xi;
      acc += xi * xi;
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        float xi = x[i];
        xcache[i] = xi;
        acc += xi * xi;
      }
    }
  }
  acc = simd_sum(acc);
  //  Initialize shared memory
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write simd accumulations into shared memory
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulate over simd groups
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write the outputs
  out += gid * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      out[i] = w[w_stride * i] * static_cast<T>(xcache[i] * local_inv_mean[0]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        out[i] = w[w_stride * i] * static_cast<T>(xcache[i] * local_inv_mean[0]);
      }
    }
  }
}

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void rms_looped(
    const device T* x,
    const device T* w,
    device T* out,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  float acc = 0;
  x += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = x[i + r];
        acc += xi * xi;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = x[i + r];
          acc += xi * xi;
        }
      }
    }
  }
  acc = simd_sum(acc);
  //  Initialize shared memory
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write simd accumulations into shared memory
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulate over simd groups
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write the outputs
  out += gid * size_t(axis_size) + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        out[r + i] = w[w_stride * (i + r)] *
            static_cast<T>(x[r + i] * local_inv_mean[0]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          out[r + i] = w[w_stride * (i + r)] *
              static_cast<T>(x[r + i] * local_inv_mean[0]);
        }
      }
    }
  }
}

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void vjp_rms_single_row(
    const device T* x,
    const device T* w,
    const device T* g,
    device T* gx,
    device T* gw,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  // Advance the input pointers
  x += gid * size_t(axis_size) + lid * N_READS;
  g += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;

  // Allocate registers for the computation and accumulators
  float thread_x[N_READS];
  float thread_w[N_READS];
  float thread_g[N_READS];
  float sumx2 = 0;
  float sumgwx = 0;

  // Allocate shared memory to implement the reduction
  constexpr int SIMD_SIZE = 32;
  threadgroup float local_sumx2[SIMD_SIZE];
  threadgroup float local_sumgwx[SIMD_SIZE];
  threadgroup float local_normalizer[1];
  threadgroup float local_meangwx[1];

  // Read and accumulate locally
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = x[i];
      thread_w[i] = w[w_stride * i];
      thread_g[i] = g[i];

      sumx2 += thread_x[i] * thread_x[i];
      sumgwx += thread_x[i] * thread_w[i] * thread_g[i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        thread_x[i] = x[i];
        thread_w[i] = w[w_stride * i];
        thread_g[i] = g[i];

        sumx2 += thread_x[i] * thread_x[i];
        sumgwx += thread_x[i] * thread_w[i] * thread_g[i];
      }
    }
  }

  // Accumulate across threads
  sumx2 = simd_sum(sumx2);
  sumgwx = simd_sum(sumgwx);
  if (simd_group_id == 0) {
    local_sumx2[simd_lane_id] = 0;
    local_sumgwx[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    local_sumx2[simd_group_id] = sumx2;
    local_sumgwx[simd_group_id] = sumgwx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    sumx2 = simd_sum(local_sumx2[simd_lane_id]);
    sumgwx = simd_sum(local_sumgwx[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_meangwx[0] = sumgwx / axis_size;
      local_normalizer[0] = metal::precise::rsqrt(sumx2 / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float meangwx = local_meangwx[0];
  float normalizer = local_normalizer[0];
  float normalizer3 = normalizer * normalizer * normalizer;

  // Write the outputs
  gx += gid * size_t(axis_size) + lid * N_READS;
  gw += gid * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      gx[i] = static_cast<T>(
          thread_g[i] * thread_w[i] * normalizer -
          thread_x[i] * meangwx * normalizer3);
      if (has_w) {
        gw[i] = static_cast<T>(thread_g[i] * thread_x[i] * normalizer);
      }
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        gx[i] = static_cast<T>(
            thread_g[i] * thread_w[i] * normalizer -
            thread_x[i] * meangwx * normalizer3);
        if (has_w) {
          gw[i] = static_cast<T>(thread_g[i] * thread_x[i] * normalizer);
        }
      }
    }
  }
}

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void vjp_rms_looped(
    const device T* x,
    const device T* w,
    const device T* g,
    device T* gx,
    device T* gw,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  // Advance the input pointers
  x += gid * size_t(axis_size) + lid * N_READS;
  g += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;

  // Allocate registers for the accumulators
  float sumx2 = 0;
  float sumgwx = 0;

  // Allocate shared memory to implement the reduction
  constexpr int SIMD_SIZE = 32;
  threadgroup float local_sumx2[SIMD_SIZE];
  threadgroup float local_sumgwx[SIMD_SIZE];
  threadgroup float local_normalizer[1];
  threadgroup float local_meangwx[1];

  // Read and accumulate locally
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = x[i + r];
        float wi = w[w_stride * (i + r)];
        float gi = g[i + r];

        sumx2 += xi * xi;
        sumgwx += xi * wi * gi;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = x[i + r];
          float wi = w[w_stride * (i + r)];
          float gi = g[i + r];

          sumx2 += xi * xi;
          sumgwx += xi * wi * gi;
        }
      }
    }
  }

  // Accumulate across threads
  sumx2 = simd_sum(sumx2);
  sumgwx = simd_sum(sumgwx);
  if (simd_group_id == 0) {
    local_sumx2[simd_lane_id] = 0;
    local_sumgwx[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    local_sumx2[simd_group_id] = sumx2;
    local_sumgwx[simd_group_id] = sumgwx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    sumx2 = simd_sum(local_sumx2[simd_lane_id]);
    sumgwx = simd_sum(local_sumgwx[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_meangwx[0] = sumgwx / axis_size;
      local_normalizer[0] = metal::precise::rsqrt(sumx2 / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float meangwx = local_meangwx[0];
  float normalizer = local_normalizer[0];
  float normalizer3 = normalizer * normalizer * normalizer;

  // Write the outputs
  gx += gid * size_t(axis_size) + lid * N_READS;
  gw += gid * size_t(axis_size) + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = x[i + r];
        float wi = w[w_stride * (i + r)];
        float gi = g[i + r];

        gx[i + r] =
            static_cast<T>(gi * wi * normalizer - xi * meangwx * normalizer3);
        if (has_w) {
          gw[i + r] = static_cast<T>(gi * xi * normalizer);
        }
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = x[i + r];
          float wi = w[w_stride * (i + r)];
          float gi = g[i + r];

          gx[i + r] =
              static_cast<T>(gi * wi * normalizer - xi * meangwx * normalizer3);
          if (has_w) {
            gw[i + r] = static_cast<T>(gi * xi * normalizer);
          }
        }
      }
    }
  }
}

// clang-format off
#define instantiate_rms(name, itype)                                \
  instantiate_kernel("rms" #name, rms_single_row, itype)            \
  instantiate_kernel("vjp_rms" #name, vjp_rms_single_row, itype)    \
  instantiate_kernel("rms_looped" #name, rms_looped, itype)         \
  instantiate_kernel("vjp_rms_looped" #name, vjp_rms_looped, itype)

instantiate_rms(float32, float)
instantiate_rms(float16, half)
instantiate_rms(bfloat16, bfloat16_t) // clang-format on
