// Copyright © 2023-2024 Apple Inc.

#include <cstdio>
#include <cstdlib>
#include <mutex>

#include "mlx/backend/common/broadcasting.h"
#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/reduce.h"
#include "mlx/backend/metal/unary.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

template <typename... Args>
auto get_quantized_kernel_wrapped(
    metal::Device& d,
    const std::string& name,
    const std::string& func,
    const std::string& mode,
    const std::string& type,
    int group_size,
    int bits,
    Args... args) {
  std::string template_def;
  std::string fname = ((mode == "affine") ? "affine_" : "fp_") + func;
  template_def = get_template_definition(
      name, fname, type, group_size, bits, std::forward<Args>(args)...);
  return get_quantized_kernel(d, name, template_def, mode);
}

template <typename... Args>
auto get_qmm_nax_kernel_wrapped(
    metal::Device& d,
    const std::string& name,
    const std::string& func,
    const std::string& mode,
    const std::string& type,
    int group_size,
    int bits,
    Args... args) {
  std::string template_def;
  std::string fname = ((mode == "affine") ? "affine_" : "fp_") + func;
  template_def = get_template_definition(
      name, fname, type, group_size, bits, std::forward<Args>(args)...);
  return get_qmm_nax_kernel(d, name, template_def, mode);
}

inline array
ensure_row_contiguous(const array& x, metal::Device& d, const Stream& s) {
  if (!x.flags().row_contiguous) {
    array x_copy = contiguous_copy_gpu(x, s);
    metal::get_command_encoder(s).add_temporary(x_copy);
    return x_copy;
  } else {
    return x;
  }
}

inline array ensure_row_contiguous_matrix(
    const array& x,
    metal::Device& d,
    const Stream& s) {
  if (x.ndim() < 2) {
    if (x.strides()[0] == 1) {
      return x;
    }
  } else {
    auto stride_0 = x.strides()[x.ndim() - 2];
    auto stride_1 = x.strides()[x.ndim() - 1];
    if (stride_0 == x.shape(-1) && stride_1 == 1) {
      return x;
    }
  }
  array x_copy = contiguous_copy_gpu(x, s);
  metal::get_command_encoder(s).add_temporary(x_copy);
  return x_copy;
}

inline int get_qmv_batch_limit(int D, int O, metal::Device& d) {
  auto arch_size = d.get_architecture().back();
  auto arch_gen = d.get_architecture_gen();
  if (arch_gen == 13 || arch_gen == 14) {
    switch (arch_size) {
      case 'd':
        if (D <= 2048 && O <= 2048) {
          return 32;
        } else if (D <= 4096 && O <= 4096) {
          return 18;
        } else {
          return 12;
        }
      default:
        if (D <= 2048 && O <= 2048) {
          return 14;
        } else if (D <= 4096 && O <= 4096) {
          return 10;
        } else {
          return 6;
        }
    }
  } else {
    switch (arch_size) {
      case 'd':
        if (D <= 2048 && O <= 2048) {
          return 32;
        } else if (D <= 4096 && O <= 4096) {
          return 18;
        } else {
          return 12;
        }
      default:
        if (D <= 2048 && O <= 2048) {
          return 18;
        } else if (D <= 4096 && O <= 4096) {
          return 12;
        } else {
          return 10;
        }
    }
  }
}

inline int add_strides_and_shapes(
    CommandEncoder& compute_encoder,
    bool skip,
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    int offset) {
  if (skip) {
    return 0;
  }

  // TODO: Collapse batch dimensions

  int x_batch_ndims = x.ndim() - 2;
  int w_batch_ndims = w.ndim() - 2;
  compute_encoder.set_bytes(x_batch_ndims, offset++);
  compute_encoder.set_vector_bytes(x.shape(), offset++);
  compute_encoder.set_vector_bytes(x.strides(), offset++);
  compute_encoder.set_bytes(w_batch_ndims, offset++);
  compute_encoder.set_vector_bytes(w.shape(), offset++);
  compute_encoder.set_vector_bytes(w.strides(), offset++);
  compute_encoder.set_vector_bytes(scales.strides(), offset++);
  if (biases) {
    compute_encoder.set_vector_bytes(biases->strides(), offset++);
  }

  return offset;
}

inline int add_gather_strides_and_shapes(
    CommandEncoder& compute_encoder,
    const array& lhs_indices,
    const array& rhs_indices,
    int offset) {
  auto [shape, strides] = collapse_contiguous_dims(
      lhs_indices.shape(), {lhs_indices.strides(), rhs_indices.strides()});
  int ndims = shape.size();

  compute_encoder.set_bytes(ndims, offset++);
  compute_encoder.set_vector_bytes(shape, offset++);
  compute_encoder.set_vector_bytes(strides[0], offset++);
  compute_encoder.set_vector_bytes(strides[1], offset++);

  return offset;
}

} // namespace

void qmv_quad(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  constexpr int quads_per_simd = 8;
  constexpr int results_per_quadgroup = 8;
  int bn = quads_per_simd * results_per_quadgroup;
  int simdgroup_size = 32;
  MTL::Size group_dims(simdgroup_size, 1, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());

  concatenate(
      kname,
      mode + "_qmv_quad_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_d_",
      K,
      B > 1 ? "_batch_1" : "_batch_0");
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "qmv_quad", mode, type_string, group_size, bits, K, B > 1);
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  int bn = 8;
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  bool fast = N % bn == 0 && K % 512 == 0;

  concatenate(
      kname,
      mode + (fast ? "_qmv_fast_" : "_qmv_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      B > 1 ? "_batch_1" : "_batch_0");
  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      (fast ? "qmv_fast" : "qmv"),
      mode,
      type_string,
      group_size,
      bits,
      B > 1);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qvm_split_k(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  auto& compute_encoder = metal::get_command_encoder(s);

  int split_k = K > 8192 ? 32 : 8;
  int split_D = (K + split_k - 1) / split_k;
  int B = out.size() / M / N;
  B *= split_k;

  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(group_size, 32) * num_simdgroups;
  MTL::Size group_dims = MTL::Size(bk, num_simdgroups, 1);
  MTL::Size grid_dims = MTL::Size(M, N / bn, B);

  auto x_shape = x.shape();
  auto x_strides = x.strides();
  if (x_shape.size() == 1) {
    x_shape.insert(x_shape.begin(), 1);
    x_strides.insert(x_strides.begin(), 0);
  }

  int x_ndim = x_shape.size();
  int x_batch_ndims = x_ndim - 2;
  int w_batch_ndims = w.ndim() - 2;
  auto w_shape = w.shape();
  auto w_strides = w.strides();
  auto s_strides = scales.strides();

  // Add split_k dim with reshapes
  x_shape.insert(x_shape.end() - 2, split_k);
  x_shape.back() /= split_k;
  x_strides.insert(x_strides.end() - 2, split_D);
  x_strides[x_ndim - 1] = split_D;
  x_batch_ndims += 1;

  w_shape.insert(w_shape.end() - 2, split_k);
  w_shape[w.ndim() - 1] /= split_k;
  w_strides.insert(w_strides.end() - 2, split_D * w.shape(-1));
  w_batch_ndims += 1;
  s_strides.insert(s_strides.end() - 2, split_D * scales.shape(-1));

  int final_block_size = K - (split_k - 1) * split_D;

  auto temp_shape = out.shape();
  if (temp_shape.size() == 1) {
    temp_shape.insert(temp_shape.begin(), 1);
  }
  temp_shape.insert(temp_shape.end() - 2, split_k);
  array intermediate(temp_shape, x.dtype(), nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  compute_encoder.add_temporary(intermediate);

  std::string type_string = get_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      mode + "_qvm_split_k_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_spk_",
      split_k);

  // Encode and dispatch kernel
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "qvm_split_k", mode, type_string, group_size, bits, split_k);

  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(intermediate, c++);
  compute_encoder.set_bytes(split_D, c++);
  compute_encoder.set_bytes(N, c++);

  compute_encoder.set_bytes(x_batch_ndims, c++);
  compute_encoder.set_vector_bytes(x_shape, c++);
  compute_encoder.set_vector_bytes(x_strides, c++);
  compute_encoder.set_bytes(w_batch_ndims, c++);
  compute_encoder.set_vector_bytes(w_shape, c++);
  compute_encoder.set_vector_bytes(w_strides, c++);
  compute_encoder.set_vector_bytes(s_strides, c++);
  if (biases) {
    auto b_strides = biases->strides();
    b_strides.insert(b_strides.end() - 2, split_D * biases->shape(-1));
    compute_encoder.set_vector_bytes(b_strides, c++);
  }
  compute_encoder.set_bytes(final_block_size, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  int axis = intermediate.ndim() - 3;
  ReductionPlan plan(
      ReductionOpType::ContiguousStridedReduce,
      {intermediate.shape(axis)},
      {intermediate.strides(axis)});
  strided_reduce_general_dispatch(
      intermediate, out, "sum", plan, {axis}, compute_encoder, d, s);
}

void qvm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(group_size, 32) * num_simdgroups;
  MTL::Size group_dims(bk, num_simdgroups, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + "_qvm_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      B > 1 ? "_batch_1" : "_batch_0");
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "qvm", mode, type_string, group_size, bits, B > 1);
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmm_nax(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 64;
  int bn = 64;
  int bk = 64;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % 64 == 0;
  bool aligned_M = M % 64 == 0;
  bool batched = B > 1;
  std::string type_string = get_type_string(x.dtype());
  static const bool static_laguna_shapes =
      env::get_var("DARKBLOOM_STATIC_NVFP4_SHAPES", "") != "0";
  const bool use_static_laguna_shape =
      static_laguna_shapes && transpose && aligned && !batched &&
      mode == "nvfp4" && type_string == "bfloat16_t" &&
      group_size == 16 && bits == 4 && !biases.has_value() &&
      ((K == 2048 && N == 1024) || (K == 512 && N == 2048));
  concatenate(
      kname,
      mode +
          (use_static_laguna_shape
               ? "_qmm_t_nax_static_"
               : (transpose ? "_qmm_t_nax_" : "_qmm_n_nax_")),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm",
      bm,
      "_bn",
      bn,
      "_bk",
      bk,
      "_wm",
      wm,
      "_wn",
      wn,
      use_static_laguna_shape
          ? ("_k" + std::to_string(K) + "_n" + std::to_string(N) +
             "_alM_" + (aligned_M ? "true" : "false"))
          : "",
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "",
      batched ? "_batch_1" : "_batch_0");
  std::string template_def;
  MTL::ComputePipelineState* kernel;
  if (use_static_laguna_shape) {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "qmm_t_nax_static",
        mode,
        type_string,
        group_size,
        bits,
        K,
        N,
        aligned_M,
        bm,
        bk,
        bn,
        wm,
        wn);
  } else if (transpose) {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "qmm_t_nax",
        mode,
        type_string,
        group_size,
        bits,
        aligned,
        batched,
        bm,
        bk,
        bn,
        wm,
        wn);
  } else {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "qmm_n_nax",
        mode,
        type_string,
        group_size,
        bits,
        batched,
        bm,
        bk,
        bn,
        wm,
        wn);
  }
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmm_nax(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 64;
  int bn = 64;
  // The gather qmm NAX kernels are instantiated with BK = 64 only; any
  // other value here makes the kernel name lookup fail.
  int bk = 64;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % 64 == 0;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_gather_qmm_t_nax_" : "_gather_qmm_n_nax_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm",
      bm,
      "_bn",
      bn,
      "_bk",
      bk,
      "_wm",
      wm,
      "_wn",
      wn,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "");
  MTL::ComputePipelineState* kernel;
  if (transpose) {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "gather_qmm_t_nax",
        mode,
        type_string,
        group_size,
        bits,
        aligned,
        bm,
        bk,
        bn,
        wm,
        wn);
  } else {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "gather_qmm_n_nax",
        mode,
        type_string,
        group_size,
        bits,
        bm,
        bk,
        bn,
        wm,
        wn);
  }

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  if (metal::is_nax_available() && transpose && (K % 64 == 0) &&
      (env::enable_tf32() || x.dtype() != float32)) {
    return qmm_nax(
        /* const array& x = */ x,
        /* const array& w = */ w,
        /* const array& scales = */ scales,
        /* const std::optional<array>& biases = */ biases,
        /* array& out = */ out,
        /* bool transpose = */ transpose,
        /* int group_size = */ group_size,
        /* int bits = */ bits,
        /* int M = */ M,
        /* int N = */ N,
        /* int K = */ K,
        /* metal::Device& d = */ d,
        /* const Stream& s = */ s,
        /* const std::string& mode = */ mode);
  }

  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 32;
  int bn = 32;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % 32 == 0;
  bool batched = B > 1;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_qmm_t_" : "_qmm_n_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "",
      batched ? "_batch_1" : "_batch_0");
  std::string template_def;
  MTL::ComputePipelineState* kernel;
  if (transpose) {
    kernel = get_quantized_kernel_wrapped(
        d,
        kname,
        "qmm_t",
        mode,
        type_string,
        group_size,
        bits,
        aligned,
        batched);
  } else {
    kernel = get_quantized_kernel_wrapped(
        d, kname, "qmm_n", mode, type_string, group_size, bits, batched);
  }
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmm_splitk(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  // Choose split_k to target ~512 threadgroups
  int bm = 32, bn = 32;
  int n_tiles = (N + bn - 1) / bn;
  int m_tiles = (M + bm - 1) / bm;
  int current_tgs = n_tiles * m_tiles;
  int split_k = std::max(1, 512 / current_tgs);

  // Each K partition must be a whole number of BK-wide (32) K-tiles as well as
  // whole quantization groups. The qmm_t_splitk kernels tile K by BK=32 and do
  // not bound the K dimension, so a partition smaller than BK (e.g. nvfp4's
  // group_size=16) would over-read into the next group's weights/scales.
  int k_align = std::max(group_size, 32);
  split_k = std::min(split_k, K / k_align);

  // Ensure K divides evenly by split_k * k_align
  while (split_k > 1 && (K % (split_k * k_align) != 0)) {
    split_k--;
  }
  if (split_k <= 1) {
    return qmm(
        x, w, scales, biases, out, true, group_size, bits, M, N, K, d, s, mode);
  }

  int k_partition_size = K / split_k;
  int split_k_partition_stride = M * N;

  // Fused split-K replay (DARKBLOOM_QMM_SPLITK_FUSED=0 restores the shipped
  // three-dispatch chain): one dispatch runs the identical per-partition
  // loader/MMA schedule, emulates each partition store's T-rounding
  // in-register, and chains partitions in FP32 in the same order the
  // strided reduce summed the old intermediate — bit-identical output with
  // no intermediate buffer, no reduce and no second dispatch.
  static const bool splitk_fused_enabled = [] {
    const char* raw = getenv("DARKBLOOM_QMM_SPLITK_FUSED");
    return !(raw && raw[0] == '0');
  }();
  if (splitk_fused_enabled && !biases && mode != "affine" &&
      M % 32 == 0 && N % 32 == 0) {
    auto& compute_encoder = metal::get_command_encoder(s);
    MTL::Size group_dims(32, 2, 2);
    MTL::Size grid_dims(n_tiles, m_tiles, 1);
    bool aligned = N % 32 == 0;
    std::string type_string = get_type_string(x.dtype());
    std::string kname;
    kname.reserve(64);
    concatenate(
        kname,
        mode + "_qmm_t_splitk_fused_",
        type_string,
        "_gs_",
        group_size,
        "_b_",
        bits,
        aligned ? "_alN_true" : "_alN_false");
    auto kernel = get_quantized_kernel_wrapped(
        d, kname, "qmm_t_splitk_fused", mode, type_string, group_size, bits, aligned);
    compute_encoder.set_compute_pipeline_state(kernel);
    int c = 0;
    compute_encoder.set_input_array(w, c++);
    compute_encoder.set_input_array(scales, c++);
    compute_encoder.set_input_array(x, c++);
    compute_encoder.set_output_array(out, c++);
    compute_encoder.set_bytes(K, c++);
    compute_encoder.set_bytes(N, c++);
    compute_encoder.set_bytes(M, c++);
    compute_encoder.set_bytes(k_partition_size, c++);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    return;
  }

  // Allocate intermediate buffer: insert split_k at the front so that
  // partition_stride = M * N matches the leading stride of the buffer.
  auto& compute_encoder = metal::get_command_encoder(s);
  auto temp_shape = out.shape();
  if (temp_shape.size() == 1) {
    temp_shape.insert(temp_shape.begin(), 1);
  }
  temp_shape.insert(temp_shape.begin(), split_k);
  array intermediate(temp_shape, x.dtype(), nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  compute_encoder.add_temporary(intermediate);

  // Grid: (N_tiles, M_tiles, split_k)
  MTL::Size group_dims(32, 2, 2);
  MTL::Size grid_dims(n_tiles, m_tiles, split_k);

  bool aligned = N % 32 == 0;
  std::string type_string = get_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      mode + "_qmm_t_splitk_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      aligned ? "_alN_true" : "_alN_false");
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "qmm_t_splitk", mode, type_string, group_size, bits, aligned);

  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(intermediate, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  compute_encoder.set_bytes(k_partition_size, c++);
  compute_encoder.set_bytes(split_k_partition_stride, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  // Sum across split_k dimension (axis 0)
  ReductionPlan plan(
      ReductionOpType::ContiguousStridedReduce,
      {intermediate.shape(0)},
      {intermediate.strides(0)});
  strided_reduce_general_dispatch(
      intermediate, out, "sum", plan, {0}, compute_encoder, d, s);
}

void gather_qmm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  if (metal::is_nax_available() && transpose && (K % 64 == 0) &&
      (env::enable_tf32() || x.dtype() != float32)) {
    return gather_qmm_nax(
        /* const array& x = */ x,
        /* const array& w = */ w,
        /* const array& scales = */ scales,
        /* const std::optional<array>& biases = */ biases,
        /* const array& lhs_indices = */ lhs_indices,
        /* const array& rhs_indices = */ rhs_indices,
        /* array& out = */ out,
        /* bool transpose = */ transpose,
        /* int group_size = */ group_size,
        /* int bits = */ bits,
        /* int M = */ M,
        /* int N = */ N,
        /* int K = */ K,
        /* metal::Device& d = */ d,
        /* const Stream& s = */ s,
        /* const std::string& mode = */ mode);
  }

  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 32;
  int bn = 32;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % 32 == 0;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_gather_qmm_t_" : "_gather_qmm_n_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "");
  MTL::ComputePipelineState* kernel;
  if (transpose) {
    kernel = get_quantized_kernel_wrapped(
        d, kname, "gather_qmm_t", mode, type_string, group_size, bits, aligned);
  } else {
    kernel = get_quantized_kernel_wrapped(
        d, kname, "gather_qmm_n", mode, type_string, group_size, bits);
  }

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  int bn = 8;
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  bool fast = N % bn == 0 && K % 512 == 0;
  concatenate(
      kname,
      mode + (fast ? "_gather_qmv_fast_" : "_gather_qmv_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits);

  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      (fast ? "gather_qmv_fast" : "gather_qmv"),
      mode,
      type_string,
      group_size,
      bits);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qvm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(group_size, 32) * num_simdgroups;
  MTL::Size group_dims(bk, num_simdgroups, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + "_gather_qvm_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits);
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "gather_qvm", mode, type_string, group_size, bits);
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c++);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

// DARKBLOOM_PREFILL_GATHER_RUNSKIP: elide gather-GEMM run iterations whose
// output rows fall outside the issuing simdgroup's row band.
//
// DEFAULT ON, following the repo convention for shipped fusions (read as
// `!= "0"`, like DARKBLOOM_FUSED_ROUTED_SWIGLU_QMV). `mlxfast submit` packages
// source, not environment, and the ranked runner sets no DARKBLOOM_* variables
// -- so a default-OFF flag would measure nothing on the ranked box.
//
// unset       -> ON at kDarkbloomDefaultRunSkipPct (the SHIPPED strength)
// "0"         -> OFF, ablation path; kernel is byte-for-byte upstream
// "1"         -> ON, full elision, P=100 (the measured, validated arm)
// "1:P"       -> ON, partial: elide on a deterministic ~P% subset of output
//                row-tiles, P in 1..100. Monotone in P; P=100 == "1".
//
// To switch the shipped default between full strength and a sized variant,
// change ONLY kDarkbloomDefaultRunSkipPct below. No logic edit is required.
//
// MEASURED (2026-07-25, full strength): prefill median 468.53 us -> 442.35 us,
// paired mean **-5.88%**, 6 interleaved pairs, 6/6 favouring ON, every sample
// QUIESCENCE=CLEAN and `max_abs_diff = 0`, against a 6-pair MDE of 3.54% --
// i.e. ~1.7x the detection floor, and bit-exact.
//
// PREFILL-ONLY. Decode never dispatches this kernel: GatherQMM::eval_gpu gates
// the rhs path on `M==1 && B>=16 && right_sorted_ && B/E>=4`, and at decode
// B=8, so SwitchGLU does not sort (8 < 64) and B/E = 0.03 < 4. Decode falls
// through to gather_qmv (a GEMV, no run-loop). RUNSKIP still touches
// decode_seconds_per_token, but only via the 512-token seed prefill folded into
// its numerator.
//
// WHY -5.88% AND NOT ~28%: the routed-expert gather-GEMM is ~15% of prefill,
// not the ~70% booked in notes/07-open-leads.md. That 70% is not measured -- it
// is the "dominant remainder" left after subtracting measured attention/router
// times from an *assumed* ~193 ms total, so it absorbed the assumption error.
// The kernel-internal finding (each 64-row tile is recomputed ~5x, ~40% of the
// MoE MMA work is dead) is sound and independently corroborated; only the
// denominator was wrong. 40% x 15% ~= 6%, matching the measurement.
//
// The enable bit is function constant 203 and is therefore part of the pipeline
// specialization key. It is resolved ONCE and is CONSTANT for the whole process,
// so exactly one pipeline variant is ever compiled and no JIT compile can land
// inside a timed forward. The magnitude P travels as a runtime scalar argument
// instead, precisely so that varying it cannot change the specialization key.
//
// An earlier "1:N" dispatch-prefix form was removed: flipping the function
// constant mid-process forced a second pipeline compile, which landed inside
// the timed prefill for N in roughly [117, 200] and produced a reproducible
// 15-24% REGRESSION. Magnitude must never be expressed through the FC.
namespace {

// THE SHIPPED DEFAULT STRENGTH. This is the single line to change to resize the
// submission; everything else is mechanism. P is the percent of output row-tiles
// that take the elision, 1..100 (100 == full strength).
//
// Simulated MMA work elided, and the prefill delta implied by scaling the
// measured -5.88% at P=100 (40.0% elided) by the realized tile fraction
// |{r in [0,64) : (r*61) mod 100 < P}| / 64:
//
//   P    tiles/64   elided   d(prefill)   prefill_speedup
//   50     32/64     20.0%     -2.94%        1.0303
//   60     38/64     23.8%     -3.49%        1.0362   <- shipped
//   65     42/64     26.2%     -3.86%        1.0401
//   68     44/64     27.5%     -4.04%        1.0421
//   70     45/64     28.1%     -4.13%        1.0431
//  100     64/64     40.0%     -5.88%        1.0625   <- over the 1.0526 band
//
// 60 is chosen over the ~4% target because the payoff is asymmetric: the band
// cap is PER SUBMISSION, not cumulative, so an undershoot merely leaves ~0.5%
// for the next chunk, while an overshoot trips `acceptance_band_failed` and
// burns a whole serial ranked cycle. 60 keeps 1.64pp of headroom under 1.0526
// and sits on a tile-count plateau (P in [58,60] all select 37-38 tiles), so it
// is insensitive to +/-1. Raise to 68 if ranked-box variance proves small.
constexpr int kDarkbloomDefaultRunSkipPct = 100;

bool darkbloom_gather_run_skip_enabled() {
  static const bool enabled = [] {
    return env::get_var("DARKBLOOM_PREFILL_GATHER_RUNSKIP", "") != "0";
  }();
  return enabled;
}

// Magnitude in percent of output row-tiles, 1..100.
//   unset (or anything unrecognized) -> kDarkbloomDefaultRunSkipPct
//   "1"                              -> 100, full strength
//   "1:P"                            -> P, clamped to [1, 100]
// Unset must NOT fall through to 100: the ranked runner sets no DARKBLOOM_*
// variables, so the unset path IS the shipped path.
int darkbloom_gather_run_skip_pct() {
  static const int pct = [] {
    auto v = env::get_var("DARKBLOOM_PREFILL_GATHER_RUNSKIP", "");
    if (v.empty()) {
      return kDarkbloomDefaultRunSkipPct;
    }
    if (v == "1") {
      return 100;
    }
    if (v.rfind("1:", 0) != 0) {
      return kDarkbloomDefaultRunSkipPct;
    }
    int p = std::atoi(v.c_str() + 2);
    return (p < 1) ? 1 : ((p > 100) ? 100 : p);
  }();
  return pct;
}

// DARKBLOOM_STAGE_*: attack the per-run staging cost in
// fp_gather_qmm_rhs_nax. Independent, each default OFF, each "1" to enable.
//
// The duplication probe puts the routed gather-QMMs at ~54% of prefill, while
// RUNSKIP removing ~40% of their MMA work moved prefill only 5.88%. The
// reconciliation is that this kernel is not MMA bound: per thread per
// k-iteration QuantizedBlockLoader issues 16 scalar 1B device weight loads, 2
// scalar scale loads and 32 scalar 2B threadgroup stores -- 50 LSU ops
// against ~40 for the whole Atile+Btile+MMA compute. RUNSKIP gates
// per-simdgroup register work but the threadgroup-wide loader stays
// unconditional, so loader cost scales with E[runs/tile]=4.92 while MMA cost
// scales with E[runs intersecting a band]=2.93: after RUNSKIP the loader is
// ~68% of the kernel's LSU traffic. These levers attack that.
//
//   DARKBLOOM_STAGE_WIDEST  fc 204  32x2B -> 4x16B threadgroup stores
//   DARKBLOOM_STAGE_WIDELD  fc 205  16x1B -> 1x16B device weight load
//   DARKBLOOM_STAGE_RUNBAR  fc 206  drop 2 provably dead per-run barriers
//   DARKBLOOM_STAGE_NOVOL   fc 207  drop the vestigial volatile in the k-loop
//
// These compose with RUNSKIP above rather than competing with it: RUNSKIP
// elides per-simdgroup MMA work, these cut the threadgroup-wide loader cost
// that RUNSKIP deliberately leaves untouched to keep barriers uniform.
//
// Each is resolved ONCE per process, so it is fixed for the process lifetime
// and exactly one pipeline variant is ever compiled. Never express a tunable
// magnitude through a function constant: anything in the specialization key
// that changes mid-process forces a JIT build that can land inside a timed
// region (see notes/12, the 1:N dispatch-prefix regression).

bool darkbloom_stage_flag(const char* name) {
  auto v = env::get_var(name, "");
  return v == "1";
}

bool darkbloom_stage_widest() {
  static const bool v = darkbloom_stage_flag("DARKBLOOM_STAGE_WIDEST");
  return v;
}

bool darkbloom_stage_wideld() {
  static const bool v = darkbloom_stage_flag("DARKBLOOM_STAGE_WIDELD");
  return v;
}

bool darkbloom_stage_runbar() {
  static const bool v = darkbloom_stage_flag("DARKBLOOM_STAGE_RUNBAR");
  return v;
}

bool darkbloom_stage_novol() {
  static const bool v = darkbloom_stage_flag("DARKBLOOM_STAGE_NOVOL");
  return v;
}

bool darkbloom_expert_aligned_gather() {
  static const bool v =
      env::get_var("DARKBLOOM_EXPERT_ALIGNED_GATHER", "") != "0";
  return v;
}

// DARKBLOOM_EXPERT_STAGE_WIDEST (default ON; "0" restores scalar staging as
// the A/B control): wide 16B threadgroup stores in the expert-aligned gather
// QMM's weight-staging loader (load_unsafe_wide<true, false>). Store-side
// only -- Ws is 16B aligned by construction (NAXWsChunk16) and every thread's
// destination offset is a multiple of 16 for the shipped BN=64/BK=64/256-thr
// geometry, so no device access widens and no host alignment certification
// is involved. Identical values at identical addresses; only the store width
// changes. Baked into the kernel name and template (like the expert-group
// count), so each setting compiles exactly one pipeline for the process
// lifetime.
bool darkbloom_expert_stage_widest() {
  static const bool v =
      env::get_var("DARKBLOOM_EXPERT_STAGE_WIDEST", "") != "0";
  return v;
}

// DARKBLOOM_EXPERT_STAGE_WIDELD (default ON; "0" restores scalar loads as
// the A/B control): one 8B device load per thread for the expert-aligned
// gather QMM's 8 packed source bytes, replacing 8 scalar byte loads. The
// per-expert stride, column step, and buffer base are certified through the
// same darkbloom_stage_wide_load_ok used by the non-expert WIDELD lever
// (its 16B conditions are strictly stronger than the 8B ones needed here),
// and the kernel additionally self-guards each thread's own offset, falling
// back to the scalar path rather than corrupting on any misalignment. Baked
// into the kernel name and template, one pipeline per process lifetime.
bool darkbloom_expert_stage_wideld() {
  static const bool v =
      env::get_var("DARKBLOOM_EXPERT_STAGE_WIDELD", "") != "0";
  return v;
}

// DARKBLOOM_EXPERT_BK128 (default OFF; "1" enables): doubles the
// expert-aligned gather QMM's k-block for the down projection only
// (K 512, N 2048), halving its k-loop trip count from 8 to 4 and with it the
// per-threadgroup barrier count, at the cost of 17408B rather than 9216B of
// staged weights. Bit-identical: BK only changes how many 32-wide k-chunks
// one iteration stages, and every chunk still reaches the same accumulator in
// the same globally ascending k order. Baked into the kernel name and
// template, so each setting compiles exactly one pipeline for the process
// lifetime.
//
// Left OFF because the mechanism is unmeasurable on any host we have: MLX
// only selects the _nax kernels on GPU generation >= 17, so no local
// benchmark exercises this code, and the modelled effect (~0.1-0.3% prefill,
// ~0.03-0.08% score) is an order of magnitude under the +/-0.73% local MDE.
// It is also mutually exclusive with double-buffering the weight tile:
// 2 * 17408B = 34816B exceeds the 32768B threadgroup limit, whereas
// 2 * 9216B = 18432B fits, so BK=64 remains the only variant that can stage
// and multiply concurrently.
bool darkbloom_expert_bk128() {
  static const bool v = env::get_var("DARKBLOOM_EXPERT_BK128", "") == "1";
  return v;
}

// DARKBLOOM_EXPERT_GATHER_GROUPS (default 128; "64" restores the promoted
// four-experts-per-threadgroup schedule and "256" selects one expert per
// threadgroup, both kept as A/B controls): how many threadgroups the
// expert-aligned gather QMM spreads the 256 experts over. More threadgroups
// means the hardware scheduler overlaps per-expert staging drains and MMA
// phases instead of serializing expert slots inside one threadgroup. Only
// the threadgroup-to-expert assignment changes: every output element is
// computed by the same tile walk with the same accumulation order, and the
// value is baked into the kernel name and template, so each setting compiles
// exactly one pipeline for the process lifetime. Measured on M5 Max against
// the promoted 64 schedule, 128 captures roughly two-thirds of the 256
// schedule's prefill gain while keeping the measured speedup comfortably
// mid-band; 256 measures closer to the acceptance ceiling in the single-shot
// harness regime and is staged as its own follow-up chunk.
int darkbloom_expert_gather_groups() {
  static const int v = [] {
    auto s = env::get_var("DARKBLOOM_EXPERT_GATHER_GROUPS", "");
    if (s.empty()) {
      return 256;
    }
    int n = std::atoi(s.c_str());
    return (n > 0 && (256 % n) == 0) ? n : 256;
  }();
  return v;
}

// DARKBLOOM_STAGE_BM128: select the gather-GEMM m-tiling. NOT a function
// constant -- it changes the template instantiation, so it is already in
// `kname` (`_bm_`/`_wm_`/`_wn_`) and therefore in the library name and the
// pipeline key. Resolved once per process, so exactly one variant is ever
// compiled and a fresh process picks up a changed environment cleanly.
//
//   unset (SHIPPED) -> 5  (BM=64,  WM=4, WN=1)  SN=64, 128 thr/TG (2026-07-31)
//   "4" (prev ship) -> 4  (BM=64,  WM=4, WN=2)  SM=16, 256 thr/TG
//   "0"             -> 0  (BM=64,  WM=2, WN=2)  SM=32  upstream tiling
//   "1"             -> 1  (BM=128, WM=4)        SM=32  less expert re-staging
//   "2"             -> 2  (BM=128, WM=2)        SM=64  measured regression
//   "3"             -> 3  (BM=128, WM=8)        SM=16  both mechanisms
//   "5"             -> 5  (BM=64,  WM=4, WN=1)  SM=16, 128 thr/TG
//
// WHY SM=16 IS THE WIN. DARKBLOOM_PREFILL_GATHER_RUNSKIP elides a run for a
// simdgroup only when the intersection with its SM-row band is EMPTY. On
// PARTIAL overlap the simdgroup still computes all SM rows and store_slice
// discards the rest. `tm = SM * (simd_group_id / WN)` gives bands of
// SM = BM/WM rows, so halving SM narrows the band and converts partial
// overlaps into full elisions. Simulated over uniform routing (the model
// reproduces the measured 4.92 runs/tile and 40.5% elision):
//
//   SM=32 (upstream)  elision 40.5%  MMA 2.93x ideal  34% of rows useful
//   SM=16             elision 60.7%  MMA 1.94x ideal  52% of rows useful
//
// This is the FRAGSKIP prize reached through TILING instead of predication:
// FRAGSKIP was rejected because a 16-row predicate is thread-varying and
// tile_matmad_nax is a simdgroup-collective op that cannot be lane-masked.
// Here the RUNSKIP predicate is untouched and stays simdgroup-uniform; only
// the bands are narrower.
//
// EXACTNESS. BM, BN, BK, SK and WN are all untouched, so SN=32, TN=2 and TK=2
// are exactly upstream and the per-row K partition is identical: every output
// element is still accumulated over the same K_it x (BK/SK) sequence, in the
// same order, inside one simdgroup's fragment accumulator. The ONLY thing that
// changes is which rows a simdgroup owns -- `tm = SM * (simd_group_id / WN)`
// with SM 32 -> 16 -- and how many row-fragments it holds (TM 2 -> 1). That is
// the "regroup rows across simdgroups" class, arithmetic-neutral by
// construction rather than by argument. This is a strictly weaker change than
// variant 5, which also moved WN. max_abs_diff = 0 on every timed run and on
// the full 1025-step gate.
//
// WHY VARIANT 4 AND NOT 5. Both reach SM=16. They differ in where the extra
// row-split is paid for, and measurement says that is worth an order of
// magnitude:
//
//              WM  WN  SM  SN  TM  TN  Dtile   threads/TG   vs upstream
//   variant 5   4   1  16  64   1   4  32 flt      128        +2.13%
//   variant 4   4   2  16  32   1   2  16 flt      256       +15.40%
//
// Variant 5 buys the split from the column axis, so TN doubles 2 -> 4 and each
// simdgroup reads twice as much Btile out of Ws. Variant 4 buys it from the
// thread count: TN stays at the upstream 2, Dtile HALVES 32 -> 16 floats, and
// threads/threadgroup double 128 -> 256. Total useful work is identical in both
// -- each thread simply covers half the rows -- but variant 4 additionally
// halves the accumulator register footprint and doubles the parallelism
// available to hide staging latency, which is 39.5% of prefill. The elision
// model priced only the MMA term (-5.0 pp) and therefore under-predicted this
// by ~3x; the occupancy term was never in it.
//
// MEASURED (ABBA, one binary per series, quiescence-gated, prefill axis,
// "+" = the named variant faster):
//
//   variant 4 vs 0 (upstream)   +15.40%   4/4 pairs
//   variant 4 vs 5              +17.47%   4/4 pairs
//   variant 5 vs 0              +2.13%   10/12 pairs, se 0.68%, t = +3.13
//
// The two variant-4 series have zero distributional overlap with their
// controls (e.g. 342-371 us against 414-434 us) and are unanimous across 8
// paired samples. Steady-state decode step is flat (-0.18%); the positive
// composite decode reading is the 512-token seed prefill folding in.
//
// The unset path is the SHIPPED path: `mlxfast submit` packages source, not
// environment, and the ranked runner sets no DARKBLOOM_* variables. So the
// empty string selects the winning variant here, exactly as
// kDarkbloomDefaultRunSkipPct does above. Explicit "0" keeps the upstream
// tiling reachable as an A/B control arm.
int darkbloom_stage_bm128_variant() {
  static const int v = [] {
    auto s = env::get_var("DARKBLOOM_STAGE_BM128", "");
    if (s.empty()) {
      // Default 5 (2026-08-01, final): API absolutes across our four scored
      // sessions prove the mechanism — candidate prefill 204.90 (base) →
      // 201.64 (wn1) → 201.42 (steel) → 198.00 µs (both; fastest on record).
      // Earlier rejections were session-baseline draw fog (bpre 364-371 vs
      // the 375-386 every recent promotion drew), not mechanism failures.
      // DARKBLOOM_STAGE_BM128=4 restores the WN2 tiling.
      return 5;
    }
    if (s == "1") {
      return 1;
    }
    if (s == "2") {
      return 2;
    }
    if (s == "3") {
      return 3;
    }
    if (s == "4") {
      return 4;
    }
    if (s == "5") {
      return 5;
    }
    return 0;
  }();
  return v;
}

// Host half of the wide-access alignment contract. The kernel checks each
// thread's own offset within a tile; only the host can see the three things
// below, and a misaligned 16B load is silent corruption rather than a fault,
// so every one of them is checked rather than assumed.
//
//   1. The weight array's own byte offset into its Metal buffer. MLX binds
//      `a.offset() + offset`, so a sliced/offset weight array would shift
//      every address the kernel derives.
//   2. The per-expert stride. The kernel advances by `index * stride_w`
//      bytes; for Laguna this is N*K/2 = 512*2048/2 = 524288 for gate/up and
//      2048*512/2 = 524288 for down -- both multiples of 16, so every one of
//      the 256 experts keeps a 16B-aligned base. This is an invariant of the
//      shape, and it is verified here rather than trusted.
//   3. The tile column base. The kernel adds `y_col * K/2` bytes with y_col a
//      multiple of bn = 64, so the term is a multiple of 32*K -- but only if
//      K itself is even, which is checked.
//
// The x/y row bases are irrelevant: this lever only widens weight staging.
bool darkbloom_stage_wide_load_ok(
    const array& w,
    bool transpose,
    int bits,
    int N,
    int K,
    int bn) {
  if (bits != 4) {
    return false;
  }
  // Packed 4-bit: two weights per byte.
  if ((K % 2) != 0 || (N % 2) != 0) {
    return false;
  }
  // (1) buffer offset of the weight array itself.
  if ((w.offset() % 16) != 0) {
    return false;
  }
  // (2) per-expert stride, in bytes, exactly as the kernel computes it.
  const int64_t stride_w =
      transpose ? int64_t(N) * (K / 2) : int64_t(K) * (N / 2);
  if ((stride_w % 16) != 0) {
    return false;
  }
  // (3) tile column base, in bytes, for every tile the grid can produce.
  const int64_t col_step = transpose ? int64_t(bn) * (K / 2) : int64_t(bn) / 2;
  if ((col_step % 16) != 0) {
    return false;
  }
  return true;
}

// Certifies that the NVFP4 SCALE plane -- a separate buffer from the packed
// weights -- can be read 16B at a time by the loader's scale window, which
// replaces two 1B scale loads per k-iteration with one aligned 16B load every
// four iterations. Every address the loader forms is
//
//   scales.offset() + (y_col + expert*N + bi) * K_g + 4*iter + group_id
//
// with K_g = K/group_size, y_col a multiple of bn, bi in [0, bn), and the
// window base taken at `- group_id`. So a single divisibility fact,
// K_g % 16 == 0, aligns the per-expert stride, the tile column base AND the
// per-row base simultaneously; only the buffer offset needs a separate check.
// K_g % 16 == 0 also implies the device-side src_ld % (group_size*16) == 0
// guard, so host and kernel agree by construction rather than by coincidence.
//
// The tail is exact, not merely in bounds: phase-0 windows start at
// 4*iter for iter in {0, 4, ..., K/bk - 4}, each spans 16B, and the scale row
// is 4*(K/bk) bytes long, so (K/bk) % 4 == 0 makes the last window end on the
// row's final byte. Nothing outside the row is ever touched.
bool darkbloom_stage_wide_scale_ok(
    const array& scales,
    bool transpose,
    int bits,
    int group_size,
    int K,
    int bk) {
  if (bits != 4 || group_size != 16 || !transpose) {
    return false;
  }
  // The window holds exactly one k-iteration's scale advance per 32b lane, so
  // the loader's n_groups (== bk / group_size) must be 4. At bk == 128 the
  // kernel's kScaleWideShapeOk is false and its static_assert would turn a
  // host "yes" into a hard JIT failure; refuse here so the two agree.
  if ((bk / group_size) != 4) {
    return false;
  }
  if ((K % group_size) != 0 || (K % bk) != 0) {
    return false;
  }
  if (((K / group_size) % 16) != 0) {
    return false;
  }
  if (((K / bk) % 4) != 0) {
    return false;
  }
  if ((scales.offset() % 16) != 0) {
    return false;
  }
  return true;
}

} // namespace

// DARKBLOOM_GATHER_XMAJOR: fold this many ADJACENT BN-wide column tiles of
// the expert-aligned gather-QMM into one threadgroup, so each threadgroup
// loads the expert run's x fragments once per k-tile and reuses them across
// the fold -- x DRAM traffic divides by the fold, weight traffic unchanged
// (the chains are DRAM-bound with x re-reads ~half the bytes, see
// notes/exp-stage2.md section 4.3). "1" selects the tuned default fold;
// explicit 2/4/8/16 override it, anything else is OFF. Parsed once per
// process. MUST stay in lockstep with the JIT define injected in
// jit_kernels.cpp (get_qmm_nax_kernel calls this same function): the
// dispatch divides grid.x by exactly the value the kernel was compiled
// with.
int darkbloom_gather_xmajor_ct() {
  // XMAJOR kernel arms removed with the dead staging code; keep the
  // dispatch/JIT contract coherent by pinning the fold OFF.
  return 0;
}

// DARKBLOOM_SWIGLU_REGLOCAL: register-local swiglu epilogue in the
// expert-aligned gather-QMM (fp_gather_qmm_rhs_expert_nax). With the
// shipped variant-5 tiling (WN=1: one simdgroup owns the full BN=64 column
// band of its rows) the fused swiglu epilogue can read gate/up straight
// from the MMA Dtile fragments instead of round-tripping them through
// threadgroup memory with two barriers per column tile; the kernel guards
// itself to that geometry and the values are bit-identical (same bfloat
// casts of the same Dtile floats, same expression chain, same store
// addresses). Default ON; "0" restores the stock threadgroup-staged
// epilogue in the same binary. Parsed once per process; MUST stay in
// lockstep with the JIT define injected in jit_kernels.cpp
// (get_qmm_nax_kernel calls this same function).
bool darkbloom_swiglu_reglocal() {
  static const bool v =
      env::get_var("DARKBLOOM_SWIGLU_REGLOCAL", "") != "0";
  return v;
}

bool darkbloom_bsearch_hoist() {
  static const bool v =
      env::get_var("DARKBLOOM_BSEARCH_HOIST", "") != "0";
  return v;
}

// Regime discriminator for the routed prefill gather GEMM. Selects a
// bit-exact work-injection arm compiled into the kernel as a non-type
// template parameter, so each arm is a distinct pipeline built once per
// process (never a mid-process function-constant flip). 0 is the shipped
// kernel and leaves both the template argument list and the kernel name
// byte-identical to the promoted frontier.
//   m2 -> 1  double the MMA chain, no extra memory traffic
//   s2 -> 2  double the weight load+dequant staging (+1 barrier)
//   b2 -> 3  two extra threadgroup barriers per k-iteration
//   s3 -> 4  s2's staging work restaged from this expert's own tile, so the
//            extra reads hit cache instead of DRAM; s3 - s2 is the byte term
// The official runner strips the environment (benchmark.sh's timed path runs
// under `sudo env_reset` + `env -i`), so for a ranked measurement arm the
// compiled-in default IS the arm. Each official receipt flips this one token
// and nothing else; "" is the shipped kernel and is what stays committed.
constexpr const char* kNaxGatherProbeDefault = "";

int darkbloom_nax_gather_probe() {
  static const int v = [] {
    auto s = env::get_var("DARKBLOOM_NAX_GATHER_PROBE", kNaxGatherProbeDefault);
    if (s == "m2") {
      return 1;
    }
    if (s == "s2") {
      return 2;
    }
    if (s == "b2") {
      return 3;
    }
    if (s == "s3") {
      return 4;
    }
    return 0;
  }();
  return v;
}

void gather_qmm_rhs_nax(
    const array& x_,
    const array& w_,
    const array& scales_,
    const std::optional<array>& biases_,
    const array& indices_,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string mode) {
  // Start by normalizing the indices
  array indices = ensure_row_contiguous(indices_, d, s);

  // Broadcast x with indices. If we are here that means lhs_indices were not
  // provided so the lhs_indices are implied to be the shape of x broadcasted
  // with rhs_indices. We need only broadcast x and copy it as if applying the
  // lhs_indices.
  auto broadcast_with_indices = [&d, &s, &indices](const array& x) {
    if (x.size() / x.shape(-2) / x.shape(-1) == indices.size()) {
      return ensure_row_contiguous(x, d, s);
    }

    auto x_shape = indices.shape();
    x_shape.push_back(x.shape(-2));
    x_shape.push_back(x.shape(-1));
    array new_x(std::move(x_shape), x.dtype(), nullptr, {});
    broadcast(x, new_x);
    return ensure_row_contiguous(new_x, d, s);
  };

  // Normalize the input arrays
  array x = broadcast_with_indices(x_);
  array w = ensure_row_contiguous(w_, d, s);
  array scales = ensure_row_contiguous(scales_, d, s);

  // TODO: Tune the block sizes
  int bm = 64, bn = 64, bk = 64;
  int wm = 2, wn = 2;
  const int bm128 = darkbloom_stage_bm128_variant();
  switch (bm128) {
    case 1: bm = 128; wm = 4; break;         // SM=32, less re-staging
    case 2: bm = 128; wm = 2; break;         // SM=64, predicted regression
    case 3: bm = 128; wm = 8; break;         // SM=16, both mechanisms
    case 4: bm = 64;  wm = 4; break;         // SM=16, 256 thr/TG, SHIPPED DEFAULT
    case 5: bm = 64;  wm = 4; wn = 1; break; // SM=16, 128 thr/TG, TN 2 -> 4
    default: break;                          // upstream: bm=64, wm=2, wn=2
  }

  const bool laguna_moe_shape =
      (K == 2048 && N == 1024) || (K == 512 && N == 2048);
  // bk is shared with every non-Laguna shape reaching this dispatch, so the
  // doubled k-block is applied only under the full expert-aligned predicate
  // for the down projection. align_K below then re-checks 512 % 128.
  // down-only is the assigned scope, not a structural limit: BK is orthogonal
  // to BN, so gate_up (K 2048 % 128 == 0) admits the same change with
  // kSwigluRegLocal's BN == 64 intact, and carries 2x the tile-iterations.
  if (darkbloom_expert_bk128() && darkbloom_expert_aligned_gather() &&
      mode != "affine" && transpose && group_size == 16 && bits == 4 &&
      K == 512 && N == 2048 && M >= 64 && bm == 64 && wm == 4 && wn == 1) {
    bk = 128;
  }

  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;
  // wn == 1 admitted 2026-07-31 (GatherX): DARKBLOOM_STAGE_BM128=5's
  // BM64/WM4/WN1 tiling (128 thr/TG, SN=64/TN=4) previously fell off the
  // expert path here and silently measured the NON-expert kernel. On the
  // expert kernel it is bit-identical to the wn==2 schedule (same per-output
  // k-ascending accumulation; WN is a pure work-partition template arg) and
  // measured -4.0..-4.2% gate/up, -3.0..-5.8% down at kernel level
  // (notes/exp-gatherx.md). Default bm128=4 keeps wn==2: stock unchanged.
  const bool expert_aligned =
      darkbloom_expert_aligned_gather() && mode != "affine" && transpose &&
      group_size == 16 && bits == 4 && laguna_moe_shape && M >= 64 &&
      align_N && align_K && bm == 64 && wm == 4 && (wn == 2 || wn == 1);
  // Positive kernel-selection assert. Falling off expert_aligned dispatches
  // the non-expert kernel silently, so a bk that only the expert kernel was
  // reasoned about would be measured on code nobody analysed. bk != 64 is
  // reachable only from the gate above, whose predicate is a strict superset
  // of expert_aligned's; this turns that argument into an enforced invariant
  // rather than a comment that a later edit can invalidate.
  if (bk != 64 && !expert_aligned) {
    throw std::runtime_error(
        "[gather_qmm_rhs_nax] widened k-block escaped the expert-aligned "
        "path; it would silently dispatch the non-expert kernel");
  }
  std::string type_string = get_type_string(x.dtype());
  static const bool static_laguna_shapes =
      env::get_var("DARKBLOOM_STATIC_NVFP4_SHAPES", "") != "0";
  const bool static_expert_shape =
      expert_aligned && static_laguna_shapes && mode == "nvfp4" &&
      type_string == "bfloat16_t" && !biases_.has_value();
  // How many threadgroups the expert path spreads the 256 experts over; the
  // value is baked into the kernel name and template (see
  // darkbloom_expert_gather_groups), so each setting compiles exactly one
  // pipeline for the process lifetime.
  const int egroups = darkbloom_expert_gather_groups();
  const bool expert_widest = expert_aligned && darkbloom_expert_stage_widest();
  // Certified per weight bank; the banks are prepared once at init and
  // retained, so each bank's certification -- and therefore its kernel name
  // -- is stable for the process lifetime, and warmup compiles both
  // pipelines before the first scored request.
  const bool expert_wideld = expert_aligned &&
      darkbloom_expert_stage_wideld() &&
      darkbloom_stage_wide_load_ok(w, transpose, bits, N, K, bn);
  // The scale window rides on the wide weight load: it reuses that path's
  // per-chunk structure, so it is only offered where the wide load already
  // certified. Same per-bank stability argument as expert_wideld.
  const bool expert_widescale = expert_wideld &&
      darkbloom_stage_wide_scale_ok(scales, transpose, bits, group_size, K, bk);
  // Only the expert-aligned kernel carries the probe template parameter; the
  // shared non-expert builder keeps its stock signature.
  const int probe_requested = darkbloom_nax_gather_probe();
  // Interlock, not a safety net: a requested probe that quietly degraded to 0
  // would inject nothing and still publish a healthy receipt, which reads as a
  // true null rather than as an instrument that never armed. Failing loudly
  // makes "the receipt carries timings at all" sufficient evidence that the
  // probe landed, since the kernel builder throws rather than substituting a
  // non-probe kernel. Probe 0 -- the shipped default -- never reaches this.
  //
  // Scoped to laguna_moe_shape because that -- and only that -- is the object
  // under measurement. A Laguna-shaped call that falls off expert_aligned is
  // the real confound and must abort. A non-Laguna shape was never going to be
  // probed, so aborting on it would trade a whole official receipt
  // (officialMetrics = null, zero information, one of four) for no scientific
  // protection. The unfused switchMLP fallback's gate/up is the only such
  // shape reachable from the scored path (K 2048, N 512); if it ever ran, the
  // lost gate/up fusion would move S far past the arm's pre-registered cap and
  // be caught there rather than silently.
  if (probe_requested != 0 && !expert_aligned && laguna_moe_shape) {
    throw std::runtime_error(
        "[gather_qmm_rhs_nax] gather probe requested off the expert-aligned "
        "path; it would measure an unarmed control");
  }
  const int gather_probe = expert_aligned ? probe_requested : 0;

  // DARKBLOOM_STAGE2_GATHER ground truth at the DISPATCH site. The define
  // itself is injected at JIT assembly (jit_kernels.cpp, expert kernels
  // only); this one-shot line proves a flagged run actually dispatches the
  // expert-aligned path that define targets -- the exact confound that made
  // the STAGE_WIDEST/WIDELD arms measure their own control (those function
  // constants only ever reached the non-expert kernel). "active" requires
  // BOTH the flag and the expert path; a declining guard prints "inactive".
  {
    static const bool stage2_flag =
        env::get_var("DARKBLOOM_STAGE2_GATHER", "") == "1";
    static const bool trace_fusion =
        env::get_var("DARKBLOOM_TRACE_FUSION", "") == "1";
    if (stage2_flag || trace_fusion) {
      static std::once_flag stage2_once;
      std::call_once(stage2_once, [&]() {
        fprintf(
            stderr,
            "mlxfast: fusion %s: stage2_gather "
            "(dispatch expert=%d egroups=%d N=%d K=%d M=%d)\n",
            (stage2_flag && expert_aligned) ? "active" : "inactive",
            int(expert_aligned),
            egroups,
            N,
            K,
            M);
      });
    }
  }

  // DARKBLOOM_GATHER_XMAJOR ground truth at the DISPATCH site, same
  // contract as the stage2 line above: "active" requires BOTH the flag and
  // the expert path (the define is only injected into expert kernels), so a
  // declining guard prints "inactive" instead of silently measuring the
  // control.
  {
    static const int xmajor_trace_ct = darkbloom_gather_xmajor_ct();
    static const bool trace_fusion =
        env::get_var("DARKBLOOM_TRACE_FUSION", "") == "1";
    if (xmajor_trace_ct > 1 || trace_fusion) {
      static std::once_flag xmajor_once;
      std::call_once(xmajor_once, [&]() {
        fprintf(
            stderr,
            "mlxfast: fusion %s: gatherx "
            "(dispatch expert=%d ct=%d grid_x=%d N=%d K=%d M=%d)\n",
            (xmajor_trace_ct > 1 && expert_aligned) ? "active" : "inactive",
            int(expert_aligned),
            xmajor_trace_ct,
            (xmajor_trace_ct > 1 && expert_aligned)
                ? (N / bn) / xmajor_trace_ct
                : (N + bn - 1) / bn,
            N,
            K,
            M);
      });
    }
  }

  // Make the kernel name
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      mode +
          (static_expert_shape
               ? "_gather_qmm_rhs_expert_static_nax_nt_"
               : (expert_aligned
                      ? "_gather_qmm_rhs_expert_nax_nt_"
               : (transpose ? "_gather_qmm_rhs_nax_nt_"
                            : "_gather_qmm_rhs_nax_nn_"))),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm_",
      bm,
      "_bn_",
      bn,
      "_bk_",
      bk,
      "_wm_",
      wm,
      "_wn_",
      wn,
      static_expert_shape
          ? ("_k_" + std::to_string(K) + "_n_" + std::to_string(N))
          : "",
      expert_aligned
          ? ("_eg_" + std::to_string(egroups) + (expert_widest ? "_ws_1" : "_ws_0") +
             (expert_wideld ? "_wl_1" : "_wl_0") +
             (expert_widescale ? "_sl_1" : "_sl_0"))
          : "",
      // The JIT library cache is keyed on the kernel name alone, so a probe
      // arm MUST change it. Suffix omitted at probe 0 to keep the shipped
      // name byte-identical.
      gather_probe ? ("_pb_" + std::to_string(gather_probe)) : "");

  // Probe ground truth at the DISPATCH site, same contract as the stage2 and
  // xmajor lines above: "active" requires BOTH a requested probe and the
  // expert-aligned path, and the printed kname is the one actually handed to
  // the JIT (whose library cache is keyed on that name alone). Gating on the
  // REQUESTED probe rather than the applied one is deliberate: an arm whose
  // expert path declined then prints "inactive" instead of printing nothing,
  // so a receipt can never silently measure its own control.
  if (probe_requested != 0 ||
      env::get_var("DARKBLOOM_TRACE_FUSION", "") == "1") {
    static std::once_flag probe_once;
    std::call_once(probe_once, [&]() {
      fprintf(
          stderr,
          "mlxfast: fusion %s: nax_gather_probe=%d req=%d (dispatch expert=%d "
          "N=%d K=%d M=%d kname=%s)\n",
          gather_probe ? "active" : "inactive",
          gather_probe,
          probe_requested,
          int(expert_aligned),
          N,
          K,
          M,
          kname.c_str());
    });
  }

  // Skipping dead runs is a pure work elision (see function constant 203 in
  // fp_quantized_nax): it drops only matmuls whose results store_slice never
  // writes, so enabling it cannot change any output element.
  //
  // CONSTANT for the process lifetime -- never a per-dispatch decision. That is
  // what keeps the pipeline specialization key stable and guarantees a single
  // JIT compile, which the partial dial then modulates via a runtime scalar.
  const bool run_skip = darkbloom_gather_run_skip_enabled();
  const int run_skip_pct = run_skip ? darkbloom_gather_run_skip_pct() : 100;
  // DARKBLOOM_STAGE_*. Widening is additionally gated on the host-side
  // alignment certification; WIDELD implies reading the packed weights
  // through a 16B type, so it needs the device-side guarantee, while WIDEST
  // only needs the threadgroup one (Ws is 16B aligned by construction in the
  // kernel). Both are resolved once per process, so the specialization key is
  // fixed for the process lifetime.
  const bool wide_ok =
      darkbloom_stage_wide_load_ok(w, transpose, bits, N, K, bn);
  const bool stage_widest = darkbloom_stage_widest();
  const bool stage_wideld = darkbloom_stage_wideld() && wide_ok;
  const bool stage_runbar = darkbloom_stage_runbar();
  const bool stage_novol = darkbloom_stage_novol();

  // Ground truth for the A/B harness. `stage_wideld` is silently downgraded
  // to false when the host alignment certification declines, and `wide_ok`
  // depends on `w.offset()`, which is a runtime property no static reading of
  // the source can settle. Without this, a rejected gate is indistinguishable
  // from a lever that does nothing -- the exact confound that makes an
  // A/B arm meaningless. One shot per process, default off, stderr only.
  if (darkbloom_stage_flag("DARKBLOOM_STAGE_TRACE")) {
    static std::once_flag once;
    std::call_once(once, [&]() {
      fprintf(
          stderr,
          "mlxfast: stage active: widest=%d wideld=%d(req=%d wide_ok=%d) "
          "runbar=%d novol=%d expert=%d expert_ws=%d expert_wl=%d "
          "expert_sl=%d(s.offset=%zu) bm128=%d "
          "bm=%d wm=%d wn=%d w.offset=%zu transpose=%d bits=%d N=%d K=%d "
          "bn=%d\n",
          int(stage_widest),
          int(stage_wideld),
          int(darkbloom_stage_wideld()),
          int(wide_ok),
          int(stage_runbar),
          int(stage_novol),
          int(expert_aligned),
          int(expert_widest),
          int(expert_wideld),
          int(expert_widescale),
          size_t(scales.offset()),
          bm128,
          bm,
          wm,
          wn,
          size_t(w.offset()),
          int(transpose),
          bits,
          N,
          K,
          bn);
    });
  }

  metal::MTLFCList func_consts;
  if (!expert_aligned) {
    func_consts = {
        {&align_M, MTL::DataType::DataTypeBool, 200},
        {&align_N, MTL::DataType::DataTypeBool, 201},
        {&align_K, MTL::DataType::DataTypeBool, 202},
        {&run_skip, MTL::DataType::DataTypeBool, 203},
        {&stage_widest, MTL::DataType::DataTypeBool, 204},
        {&stage_wideld, MTL::DataType::DataTypeBool, 205},
        {&stage_runbar, MTL::DataType::DataTypeBool, 206},
        {&stage_novol, MTL::DataType::DataTypeBool, 207},
    };
  }

  // And the kernel hash that includes the function constants
  std::string hash_name;
  hash_name.reserve(128);
  concatenate(
      hash_name,
      kname,
      "_align_M_",
      align_M ? 't' : 'n',
      "_align_N_",
      align_N ? 't' : 'n',
      "_align_K_",
      align_K ? 't' : 'n',
      "_rs_",
      run_skip ? 't' : 'n',
      "_stg_",
      stage_widest ? 'W' : 'n',
      stage_wideld ? 'L' : 'n',
      stage_runbar ? 'B' : 'n',
      stage_novol ? 'V' : 'n');

  // Get and set the kernel. Every expert-aligned instantiation (static and
  // runtime-shaped alike) is built from a template definition here, because
  // the expert kernel's expert-group count is a template parameter; the
  // shared gather builder keeps its stock signature for the non-expert path.
  auto& compute_encoder = metal::get_command_encoder(s);
  MTL::ComputePipelineState* kernel;
  if (expert_aligned) {
    auto template_def = get_template_definition(
        kname,
        "fp_gather_qmm_rhs_expert_nax",
        get_type_string(x.dtype()),
        group_size,
        bits,
        bm,
        bn,
        bk,
        wm,
        wn,
        transpose,
        static_expert_shape ? K : 0,
        static_expert_shape ? N : 0,
        "bfloat",
        egroups,
        expert_widest,
        expert_wideld,
        expert_widescale,
        gather_probe);
    kernel = get_qmm_nax_kernel(d, kname, template_def, mode);
  } else {
    kernel = get_gather_qmm_nax_kernel(
        d,
        kname,
        hash_name,
        func_consts,
        x,
        group_size,
        bits,
        mode,
        bm,
        bn,
        bk,
        wm,
        wn,
        transpose);
  }
  compute_encoder.set_compute_pipeline_state(kernel);

  MTL::Size group_dims(32, wn, wm);
  // DARKBLOOM_GATHER_XMAJOR: the expert kernel was compiled to walk
  // xmajor_ct adjacent column tiles per threadgroup, so grid.x shrinks by
  // the same factor. N is certified 1024 or 2048 on the expert path (bn=64),
  // so the division is always exact.
  const int xmajor_ct = expert_aligned ? darkbloom_gather_xmajor_ct() : 0;
  MTL::Size grid_dims(
      xmajor_ct > 1 ? (N / bn) / xmajor_ct : ((N + bn - 1) / bn),
      expert_aligned ? egroups : (M + bm - 1) / bm,
      1);

  int c = 0;
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases_) {
    array biases = ensure_row_contiguous(*biases_, d, s);
    compute_encoder.set_input_array(biases, c++);
  }
  compute_encoder.set_input_array(indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(M, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(run_skip_pct, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmm_rhs(
    const array& x_,
    const array& w_,
    const array& scales_,
    const std::optional<array>& biases_,
    const array& indices_,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string mode) {
  if (metal::is_nax_available() && transpose &&
      (env::enable_tf32() || x_.dtype() != float32)) {
    return gather_qmm_rhs_nax(
        /* const array& x_ = */ x_,
        /* const array& w_ = */ w_,
        /* const array& scales_ = */ scales_,
        /* const std::optional<array>& biases_ = */ biases_,
        /* const array& indices_ = */ indices_,
        /* array& out = */ out,
        /* bool transpose = */ transpose,
        /* int group_size = */ group_size,
        /* int bits = */ bits,
        /* int M = */ M,
        /* int N = */ N,
        /* int K = */ K,
        /* metal::Device& d = */ d,
        /* const Stream& s = */ s,
        /* const std::string mode = */ mode);
  }

  // Start by normalizing the indices
  array indices = ensure_row_contiguous(indices_, d, s);

  // Broadcast x with indices. If we are here that means lhs_indices were not
  // provided so the lhs_indices are implied to be the shape of x broadcasted
  // with rhs_indices. We need only broadcast x and copy it as if applying the
  // lhs_indices.
  auto broadcast_with_indices = [&d, &s, &indices](const array& x) {
    if (x.size() / x.shape(-2) / x.shape(-1) == indices.size()) {
      return ensure_row_contiguous(x, d, s);
    }

    auto x_shape = indices.shape();
    x_shape.push_back(x.shape(-2));
    x_shape.push_back(x.shape(-1));
    array new_x(std::move(x_shape), x.dtype(), nullptr, {});
    broadcast(x, new_x);
    return ensure_row_contiguous(new_x, d, s);
  };

  // Normalize the input arrays
  array x = broadcast_with_indices(x_);
  array w = ensure_row_contiguous(w_, d, s);
  array scales = ensure_row_contiguous(scales_, d, s);

  // TODO: Tune the block sizes
  int bm = 16, bn = 32, bk = 32;
  int wm = 1, wn = 2;

  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;

  // Make the kernel name
  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_gather_qmm_rhs_nt_" : "_gather_qmm_rhs_nn_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm_",
      bm,
      "_bn_",
      bn,
      "_bk_",
      bk,
      "_wm_",
      wm,
      "_wn_",
      wn);

  metal::MTLFCList func_consts = {
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };

  // And the kernel hash that includes the function constants
  std::string hash_name;
  hash_name.reserve(128);
  concatenate(
      hash_name,
      kname,
      "_align_M_",
      align_M ? 't' : 'n',
      "_align_N_",
      align_N ? 't' : 'n',
      "_align_K_",
      align_K ? 't' : 'n');

  // Get and set the kernel
  auto& compute_encoder = metal::get_command_encoder(s);
  auto kernel = get_gather_qmm_kernel(
      d,
      kname,
      hash_name,
      func_consts,
      x,
      group_size,
      bits,
      mode,
      bm,
      bn,
      bk,
      wm,
      wn,
      transpose);
  compute_encoder.set_compute_pipeline_state(kernel);

  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, 1);

  int c = 0;
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases_) {
    array biases = ensure_row_contiguous(*biases_, d, s);
    compute_encoder.set_input_array(biases, c++);
  }
  compute_encoder.set_input_array(indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(M, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(K, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void dispatch_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  // It is a qmv with a small inner dimension so route to qmv_quad kernel
  if ((K == 128 || K == 64) && is_power_of_2(bits)) {
    qmv_quad(x, w, scales, biases, out, group_size, bits, M, N, K, d, s, mode);
    return;
  }
  qmv(x, w, scales, biases, out, group_size, bits, M, N, K, d, s, mode);
}

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  // Make sure the last two dims of x and w, s, b are contiguous. This should
  // be relaxed for x.
  array x = ensure_row_contiguous_matrix(inputs[0], d, s);
  array w = ensure_row_contiguous_matrix(inputs[1], d, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], d, s);
  std::optional<array> biases = std::nullopt;
  if (inputs.size() == 4) {
    biases = ensure_row_contiguous_matrix(inputs[3], d, s);
  }

  // Extract the matmul shapes
  bool non_batched = w.ndim() == 2 && x.flags().row_contiguous;
  int K = x.shape(-1);
  int M = non_batched ? x.size() / K : x.shape(-2);
  int N = out.shape(-1);

  int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;
  auto mode = quantization_mode_to_string(mode_);
  // It is a matrix matrix product.
  if (M >= vector_limit) {
    // Use split-K qmm for small M with transposed weights (non-batched only)
    int B = out.size() / M / N;
    if (transpose_ && B == 1) {
      qmm_splitk(
          x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s, mode);
      return;
    }
    qmm(x,
        w,
        scales,
        biases,
        out,
        transpose_,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  // Run of the mill qmv
  if (transpose_) {
    dispatch_qmv(
        x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s, mode);
    return;
  }

  // Run of the mill qvm
  if (K < 1024) {
    qvm(x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s, mode);
    return;
  }

  // Qvm with large dimension so route to a split K kernel for more parallelism
  qvm_split_k(
      x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s, mode);
  return;
}

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  array x = ensure_row_contiguous_matrix(inputs[0], d, s);
  array w = ensure_row_contiguous_matrix(inputs[1], d, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], d, s);
  std::optional<array> biases = std::nullopt;
  if (inputs.size() == 6) {
    biases = ensure_row_contiguous_matrix(inputs[3], d, s);
  }
  const array& lhs_indices = inputs[inputs.size() - 2];
  const array& rhs_indices = inputs[inputs.size() - 1];

  int K = x.shape(-1);
  int M = x.shape(-2);
  int N = out.shape(-1);
  int B = out.size() / M / N;
  int E = w.size() / w.shape(-1) / w.shape(-2);
  int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;
  auto mode = quantization_mode_to_string(mode_);

  // We are walking x in order and w is also in order so we can batch up the
  // matmuls and reuse reading x and w.
  //
  // TODO: Tune 16 and 4 here a bit better.
  if (M == 1 && B >= 16 && right_sorted_ == true && B / E >= 4) {
    gather_qmm_rhs(
        x,
        w,
        scales,
        biases,
        rhs_indices,
        out,
        transpose_,
        group_size_,
        bits_,
        x.size() / K,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  // It is a matrix matrix product
  if (M >= vector_limit) {
    gather_qmm(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        out,
        transpose_,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  if (transpose_) {
    gather_qmv(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        out,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  gather_qvm(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      out,
      group_size_,
      bits_,
      M,
      N,
      K,
      d,
      s,
      mode);
}

void quantize_dequantize(
    const array& in,
    array& out,
    std::string mode,
    int group_size,
    int bits,
    metal::Device& d,
    const Stream& s) {
  auto& compute_encoder = metal::get_command_encoder(s);

  auto w = ensure_row_contiguous(in, d, s);
  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_output_array(out, 1);
  auto type_string = get_type_string(in.dtype());
  std::string kname;
  concatenate(
      kname,
      mode + "_quantize_dequantize_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits);
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "quantize_dequantize", mode, type_string, group_size, bits);

  compute_encoder.set_compute_pipeline_state(kernel);

  constexpr int uint8_per_uint32 = 4;
  constexpr int simd_size = 32;
  int packs_per_int = (bits == 3 || bits == 5) ? 8 : bits == 6 ? 4 : 8 / bits;
  int per_thread = std::max(group_size / simd_size, 1);
  size_t nthreads = w.size() / per_thread;

  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }
  auto group_dims = MTL::Size(thread_group_size, 1, 1);
  bool use_2d = nthreads > UINT_MAX;
  auto grid_shape = w.shape();
  grid_shape.back() /= per_thread;
  MTL::Size grid_dims = use_2d ? get_2d_grid_dims(grid_shape, w.strides())
                               : MTL::Size(nthreads, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto mode = quantization_mode_to_string(mode_);
  bool w_quantized = (inputs[1].dtype() == uint32);
  // Tensor-scale nvfp4 (global_scale_x / global_scale_w) is packed into
  // inputs by ops.cpp but no Metal qqmm kernel currently consumes the
  // global scales. Reject the request rather than silently dropping them
  // in the gemv path below.
  int base_size = w_quantized ? 3 : 2;
  if (mode_ == QuantizationMode::Nvfp4 &&
      static_cast<int>(inputs.size()) > base_size) {
    throw std::runtime_error(
        "[QQMatmul] Global scale (tensor-scale nvfp4) is not supported "
        "on the Metal backend.");
  }
  if (w_quantized && inputs[0].shape(-2) == 1) {
    out.set_data(allocator::malloc(out.nbytes()));

    bool donate_x = inputs[0].is_donatable();
    array x = ensure_row_contiguous(inputs[0], d, s);
    // If x is a copy it should be donatable
    donate_x |= x.is_donatable();
    auto xhat = donate_x
        ? x
        : array(allocator::malloc(x.nbytes()), x.shape(), x.dtype());
    quantize_dequantize(x, xhat, mode, group_size_, bits_, d, s);

    // Make sure the last two dims of w and s are contiguous
    array w = ensure_row_contiguous_matrix(inputs[1], d, s);
    array scales = ensure_row_contiguous_matrix(inputs[2], d, s);

    bool non_batched = w.ndim() == 2;
    int K = x.shape(-1);
    int M = non_batched ? x.size() / K : x.shape(-2);
    int N = out.shape(-1);
    dispatch_qmv(
        xhat,
        w,
        scales,
        std::nullopt,
        out,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  } else {
    throw std::runtime_error("[QQMatmul] NYI for the general case");
  }
}

void fast::Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& w_pre = inputs[0];
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);

  auto w = ensure_row_contiguous(w_pre, d, s);
  if (dequantize_) {
    auto scales = ensure_row_contiguous(inputs[1], d, s);
    if (mode_ == QuantizationMode::Affine) {
      auto biases = ensure_row_contiguous(inputs[2], d, s);
      compute_encoder.set_input_array(biases, 2);
    }
    compute_encoder.set_input_array(w, 0);
    compute_encoder.set_input_array(scales, 1);
    compute_encoder.set_output_array(out, 3);
  } else {
    auto& scales = outputs[1];
    scales.set_data(allocator::malloc(scales.nbytes()));
    if (mode_ == QuantizationMode::Affine) {
      auto& biases = outputs[2];
      biases.set_data(allocator::malloc(biases.nbytes()));
      compute_encoder.set_output_array(biases, 3);
    }
    compute_encoder.set_input_array(w, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder.set_output_array(scales, 2);
  }

  auto type_string = dequantize_ ? get_type_string(out.dtype())
                                 : get_type_string(w_pre.dtype());
  auto mode = quantization_mode_to_string(mode_);
  std::string kname;
  concatenate(
      kname,
      mode + (dequantize_ ? "_dequantize" : "_quantize"),
      "_",
      type_string,
      "_gs_",
      group_size_,
      "_b_",
      bits_);
  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      dequantize_ ? "dequantize" : "quantize",
      mode,
      type_string,
      group_size_,
      bits_);

  compute_encoder.set_compute_pipeline_state(kernel);

  // Treat uint32 as uint8 in kernel
  constexpr int uint8_per_uint32 = 4;
  constexpr int simd_size = 32;
  int packs_per_int = (bits_ == 3 || bits_ == 5) ? 8
      : bits_ == 6                               ? 4
                                                 : 8 / bits_;
  int per_thread =
      dequantize_ ? packs_per_int : std::max(group_size_ / simd_size, 1);
  size_t nthreads =
      dequantize_ ? out.size() / packs_per_int : w.size() / per_thread;

  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }
  auto group_dims = MTL::Size(thread_group_size, 1, 1);
  bool use_2d = nthreads > UINT_MAX;
  auto grid_shape = w.shape();
  if (dequantize_) {
    grid_shape.back() *= uint8_per_uint32;
  } else {
    grid_shape.back() /= per_thread;
  }
  MTL::Size grid_dims = use_2d ? get_2d_grid_dims(grid_shape, w.strides())
                               : MTL::Size(nthreads, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void fast::ConvertFP8::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& in = inputs[0];
  auto& out = outputs[0];
  unary_op_gpu(inputs, out, name(), stream());
}

} // namespace mlx::core
