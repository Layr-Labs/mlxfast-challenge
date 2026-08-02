// Copyright © 2023 Apple Inc.

#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename U>
struct IndexValPair {
  uint32_t index;
  U val;
};

template <typename U>
struct ArgMin {
  static constexpr constant U init = Limits<U>::max;

  IndexValPair<U> reduce(IndexValPair<U> best, IndexValPair<U> current) {
    if (best.val > current.val ||
        (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }

  template <int N>
  IndexValPair<U>
  reduce_many(IndexValPair<U> best, thread U* vals, uint32_t offset) {
    for (int i = 0; i < N; i++) {
      if (vals[i] < best.val) {
        best.val = vals[i];
        best.index = offset + i;
      }
    }
    return best;
  }
};

template <typename U>
struct ArgMax {
  static constexpr constant U init = Limits<U>::min;

  IndexValPair<U> reduce(IndexValPair<U> best, IndexValPair<U> current) {
    if (best.val < current.val ||
        (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }

  template <int N>
  IndexValPair<U>
  reduce_many(IndexValPair<U> best, thread U* vals, uint32_t offset) {
    for (int i = 0; i < N; i++) {
      if (vals[i] > best.val) {
        best.val = vals[i];
        best.index = offset + i;
      }
    }
    return best;
  }
};

template <typename U>
IndexValPair<U> simd_shuffle_down(IndexValPair<U> data, uint16_t delta) {
  return IndexValPair<U>{
      simd_shuffle_down(data.index, delta), simd_shuffle_down(data.val, delta)};
}

template <typename T, typename Op, int N_READS>
METAL_FUNC IndexValPair<T> arg_reduce_generic(
    const device T* in,
    int64_t in_idx,
    int64_t axis_stride,
    size_t axis_size,
    uint3 lid,
    uint3 lsize) {
  Op op;
  IndexValPair<T> best{0, Op::init};

  // Loop over the reduction axis in lsize*N_READS buckets.
  for (uint r = 0; r < ceildiv(axis_size, N_READS * lsize.x); r++) {
    uint32_t current_index = r * lsize.x * N_READS + lid.x * N_READS;
    uint32_t offset = current_index;
    const device T* current_in = in + in_idx + current_index * axis_stride;
    T vals[N_READS];
    for (int i = 0; i < N_READS; i++) {
      vals[i] = (current_index < axis_size) ? *current_in : T(Op::init);
      current_index++;
      current_in += axis_stride;
    }
    best = op.template reduce_many<N_READS>(best, vals, offset);
  }
  return best;
}

template <typename Op>
METAL_FUNC IndexValPair<bfloat16_t> argmax_bfloat16_100352(
    const device bfloat16_t* in,
    int64_t in_idx,
    uint32_t lid) {
  constexpr uint32_t reads = 4;
  constexpr uint32_t wave_size = 4096;
  constexpr uint32_t full_waves = 24;
  Op op;
  IndexValPair<bfloat16_t> best{0, Op::init};

  // The fixed vocabulary is 24 complete 4,096-element waves followed by one
  // 2,048-element half wave. Preserve the generic loop's per-lane read order,
  // but remove its 100,352 dynamic bounds checks.
  for (uint32_t r = 0; r < full_waves; r++) {
    uint32_t offset = r * wave_size + lid * reads;
    const device bfloat16_t* current_in = in + in_idx + offset;
    bfloat16_t vals[reads] = {
        current_in[0], current_in[1], current_in[2], current_in[3]};
    best = op.template reduce_many<reads>(best, vals, offset);
  }
  if (lid < 512) {
    uint32_t offset = 98304 + lid * 4;
    const device bfloat16_t* current_in = in + in_idx + offset;
    bfloat16_t vals[reads] = {
        current_in[0], current_in[1], current_in[2], current_in[3]};
    best = op.template reduce_many<reads>(best, vals, offset);
  }
  return best;
}

template <typename T, typename Op, int N_READS = 4>
[[kernel]] void arg_reduce_general(
    const device T* in [[buffer(0)]],
    device uint32_t* out [[buffer(1)]],
    const constant int* shape [[buffer(2)]],
    const constant int64_t* in_strides [[buffer(3)]],
    const constant int64_t* out_strides [[buffer(4)]],
    const constant size_t& ndim [[buffer(5)]],
    const constant int64_t& axis_stride [[buffer(6)]],
    const constant size_t& axis_size [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 gsize [[threads_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  // Shapes and strides *do not* contain the reduction axis. The reduction size
  // and stride are provided in axis_stride and axis_size.
  //
  // Note: in shape == out shape with this convention.
  //
  // The sketch of the kernel is as follows.
  //    1. Launch prod(shape) * thread_group_size threads.
  //    2. Loop ceildiv(axis_size / lsize) times
  //    3. Read input values
  //    4. Reduce among them and go to 3
  //    4. Reduce in each simd_group
  //    6. Write in the thread local memory
  //    6. Reduce them across thread group
  //    7. Write the output without need for atomic
  Op op;

  // Compute the input/output index. There is one beginning and one output for
  // the whole threadgroup.
  int64_t row_idx = gid.y + static_cast<int64_t>(gsize.y) * gid.z;
  auto in_idx = elem_to_loc(row_idx, shape, in_strides, ndim);
  auto out_idx = elem_to_loc(row_idx, shape, out_strides, ndim);

  IndexValPair<T> best;
  if constexpr (metal::is_same_v<T, bfloat16_t> &&
                metal::is_same_v<Op, ArgMax<bfloat16_t>>) {
    if (axis_size == 100352 && axis_stride == 1 && ndim == 0 &&
        lsize.x == 1024) {
      best = argmax_bfloat16_100352<Op>(in, in_idx, lid.x);
    } else {
      best = arg_reduce_generic<T, Op, N_READS>(
          in, in_idx, axis_stride, axis_size, lid, lsize);
    }
  } else {
    best = arg_reduce_generic<T, Op, N_READS>(
        in, in_idx, axis_stride, axis_size, lid, lsize);
  }

  threadgroup IndexValPair<T> local_data[32];
  // At this point we have reduced the axis into thread group best values so we
  // need to reduce across the thread group.

  // First per simd reduction.
  for (uint offset = simd_size / 2; offset > 0; offset /= 2) {
    IndexValPair<T> neighbor = simd_shuffle_down(best, offset);
    best = op.reduce(best, neighbor);
  }

  // Write to the threadgroup memory
  if (simd_lane_id == 0) {
    local_data[simd_group_id] = best;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id != 0) {
    return;
  }

  // Read the appropriate value from local data and perform one simd reduction
  uint simd_groups = ceildiv(lsize.x, simd_size);
  if (simd_lane_id < simd_groups) {
    best = local_data[simd_lane_id];
  }
  for (uint offset = simd_size / 2; offset > 0; offset /= 2) {
    IndexValPair<T> neighbor = simd_shuffle_down(best, offset);
    best = op.reduce(best, neighbor);
  }

  // Finally write the output
  if (lid.x == 0) {
    out[out_idx] = best.index;
  }
}

// clang-format off
#define instantiate_arg_reduce(name, itype)                      \
  instantiate_kernel(                                            \
      "argmin_" #name, arg_reduce_general, itype, ArgMin<itype>) \
  instantiate_kernel(                                            \
      "argmax_" #name, arg_reduce_general, itype, ArgMax<itype>)

instantiate_arg_reduce(bool_, bool)
instantiate_arg_reduce(uint8, uint8_t)
instantiate_arg_reduce(uint16, uint16_t)
instantiate_arg_reduce(uint32, uint32_t)
instantiate_arg_reduce(uint64, uint64_t)
instantiate_arg_reduce(int8, int8_t)
instantiate_arg_reduce(int16, int16_t)
instantiate_arg_reduce(int32, int32_t)
instantiate_arg_reduce(int64, int64_t)
instantiate_arg_reduce(float16, half)
instantiate_arg_reduce(float32, float)
instantiate_arg_reduce(bfloat16, bfloat16_t) // clang-format on
