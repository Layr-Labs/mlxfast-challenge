// Copyright © 2024 Apple Inc.
#include <cstdio>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/utils.h"

using namespace fmt::literals;

namespace mlx::core {

MTL::ComputePipelineState* get_arange_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out) {
  auto lib = d.get_library(kernel_name, [&]() {
    std::string kernel_source = metal::utils();
    kernel_source += metal::arange();
    kernel_source += get_template_definition(
        kernel_name, "arange", get_type_string(out.dtype()));
    return kernel_source;
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_unary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype in_type,
    Dtype out_type,
    const char* op) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name, [&]() {
    auto in_t = get_type_string(in_type);
    auto out_t = get_type_string(out_type);
    std::string kernel_source = metal::utils();
    concatenate(kernel_source, metal::unary_ops(), metal::unary());
    kernel_source +=
        get_template_definition("v_" + lib_name, "unary_v", in_t, out_t, op, 1);
    if (get_work_per_thread(in_type) > 1) {
      kernel_source +=
          get_template_definition("vn_" + lib_name, "unary_v", in_t, out_t, op);
    }
    kernel_source +=
        get_template_definition("v2_" + lib_name, "unary_v2", in_t, out_t, op);
    kernel_source += get_template_definition(
        "gn1_" + lib_name, "unary_g", in_t, out_t, op, 1, "int");
    kernel_source += get_template_definition(
        "gn4large_" + lib_name, "unary_g", in_t, out_t, op, 4);
    return kernel_source;
  });
  return d.get_kernel(kernel_name, lib);
}

void append_binary_kernels(
    const std::string& lib_name,
    Dtype in_type,
    Dtype out_type,
    const char* op,
    std::string& kernel_source) {
  const std::array<std::pair<std::string, std::string>, 7> kernel_types = {{
      {"ss", "binary_ss"},
      {"vs2", "binary_vs2"},
      {"sv2", "binary_sv2"},
      {"vv2", "binary_vv2"},
      {"g1large", "binary_g_nd1"},
      {"g2large", "binary_g_nd2"},
      {"g3large", "binary_g_nd3"},
  }};
  auto in_t = get_type_string(in_type);
  auto out_t = get_type_string(out_type);

  for (auto& [name, func] : kernel_types) {
    kernel_source +=
        get_template_definition(name + "_" + lib_name, func, in_t, out_t, op);
  }
  kernel_source += get_template_definition(
      "vs_" + lib_name, "binary_vs", in_t, out_t, op, 1);
  kernel_source += get_template_definition(
      "sv_" + lib_name, "binary_sv", in_t, out_t, op, 1);
  kernel_source += get_template_definition(
      "vv_" + lib_name, "binary_vv", in_t, out_t, op, 1);

  if (get_work_per_thread(in_type) > 1) {
    kernel_source += get_template_definition(
        "vsn_" + lib_name, "binary_vs", in_t, out_t, op);
    kernel_source += get_template_definition(
        "svn_" + lib_name, "binary_sv", in_t, out_t, op);
    kernel_source += get_template_definition(
        "vvn_" + lib_name, "binary_vv", in_t, out_t, op);
  }

  kernel_source += get_template_definition(
      "g1_" + lib_name, "binary_g_nd1", in_t, out_t, op, "int");
  kernel_source += get_template_definition(
      "g2_" + lib_name, "binary_g_nd2", in_t, out_t, op, "int");
  kernel_source += get_template_definition(
      "g3_" + lib_name, "binary_g_nd3", in_t, out_t, op, "int");
  kernel_source += get_template_definition(
      "gn2_" + lib_name, "binary_g", in_t, out_t, op, 2, "int");
  kernel_source += get_template_definition(
      "gn4large_" + lib_name, "binary_g", in_t, out_t, op, 4);
}

MTL::ComputePipelineState* get_binary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype in_type,
    Dtype out_type,
    const char* op) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name, [&]() {
    std::string kernel_source;
    kernel_source = metal::utils();
    concatenate(kernel_source, metal::binary_ops(), metal::binary());
    append_binary_kernels(lib_name, in_type, out_type, op, kernel_source);
    return kernel_source;
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_binary_two_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype in_type,
    Dtype out_type,
    const char* op) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name, [&]() {
    std::string kernel_source = metal::utils();
    concatenate(kernel_source, metal::binary_ops(), metal::binary_two());
    append_binary_kernels(lib_name, in_type, out_type, op, kernel_source);
    return kernel_source;
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_ternary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype type,
    const char* op) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name, [&]() {
    auto t_str = get_type_string(type);
    std::string kernel_source = metal::utils();
    concatenate(kernel_source, metal::ternary_ops(), metal::ternary());
    const std::array<std::pair<std::string, std::string>, 3> kernel_types = {{
        {"g1large", "ternary_g_nd1"},
        {"g2large", "ternary_g_nd2"},
        {"g3large", "ternary_g_nd3"},
    }};
    for (auto& [name, func] : kernel_types) {
      kernel_source +=
          get_template_definition(name + "_" + lib_name, func, t_str, op);
    }

    kernel_source += get_template_definition(
        "v2_" + lib_name, "ternary_v2", t_str, op, false, false);
    kernel_source += get_template_definition(
        "sv2_" + lib_name, "ternary_v2", t_str, op, true, false);
    kernel_source += get_template_definition(
        "vs2_" + lib_name, "ternary_v2", t_str, op, false, true);

    if (get_work_per_thread(type) > 1) {
      kernel_source += get_template_definition(
          "vn_" + lib_name, "ternary_v", t_str, op, false, false);
      kernel_source += get_template_definition(
          "svn_" + lib_name, "ternary_v", t_str, op, true, false);
      kernel_source += get_template_definition(
          "vsn_" + lib_name, "ternary_v", t_str, op, false, true);
    }

    kernel_source += get_template_definition(
        "v_" + lib_name, "ternary_v", t_str, op, false, false, 1);
    kernel_source += get_template_definition(
        "sv_" + lib_name, "ternary_v", t_str, op, true, false, 1);
    kernel_source += get_template_definition(
        "vs_" + lib_name, "ternary_v", t_str, op, false, true, 1);
    kernel_source += get_template_definition(
        "g1_" + lib_name, "ternary_g_nd1", t_str, op, "int");
    kernel_source += get_template_definition(
        "g2_" + lib_name, "ternary_g_nd2", t_str, op, "int");
    kernel_source += get_template_definition(
        "g3_" + lib_name, "ternary_g_nd3", t_str, op, "int");
    kernel_source += get_template_definition(
        "gn2_" + lib_name, "ternary_g", t_str, op, 2, "int");
    kernel_source += get_template_definition(
        "gn4large_" + lib_name, "ternary_g", t_str, op, 4);
    return kernel_source;
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_copy_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name, [&]() {
    std::string kernel_source = metal::utils();
    kernel_source += metal::copy();
    auto in_type = get_type_string(in.dtype());
    auto out_type = get_type_string(out.dtype());
    kernel_source += get_template_definition(
        "s_" + lib_name, "copy_s", in_type, out_type, 1);
    kernel_source +=
        get_template_definition("s2_" + lib_name, "copy_s2", in_type, out_type);
    kernel_source += get_template_definition(
        "v_" + lib_name, "copy_v", in_type, out_type, 1);
    kernel_source +=
        get_template_definition("v2_" + lib_name, "copy_v2", in_type, out_type);

    if (get_work_per_thread(out.dtype()) > 1) {
      kernel_source += get_template_definition(
          "sn_" + lib_name, "copy_s", in_type, out_type);
      kernel_source += get_template_definition(
          "vn_" + lib_name, "copy_v", in_type, out_type);
    }

    kernel_source += get_template_definition(
        "g1_" + lib_name, "copy_g_nd1", in_type, out_type, "int");
    kernel_source += get_template_definition(
        "g2_" + lib_name, "copy_g_nd2", in_type, out_type, "int");
    kernel_source += get_template_definition(
        "g3_" + lib_name, "copy_g_nd3", in_type, out_type, "int");
    kernel_source += get_template_definition(
        "gn2_" + lib_name, "copy_g", in_type, out_type, 2, "int");
    kernel_source += get_template_definition(
        "gg1_" + lib_name, "copy_gg_nd1", in_type, out_type, "int");
    kernel_source += get_template_definition(
        "gg2_" + lib_name, "copy_gg_nd2", in_type, out_type, "int");
    kernel_source += get_template_definition(
        "gg3_" + lib_name, "copy_gg_nd3", in_type, out_type, "int");
    kernel_source += get_template_definition(
        "ggn2_" + lib_name, "copy_gg", in_type, out_type, 2, "int");
    kernel_source += get_template_definition(
        "g1large_" + lib_name, "copy_g_nd1", in_type, out_type);
    kernel_source += get_template_definition(
        "g2large_" + lib_name, "copy_g_nd2", in_type, out_type);
    kernel_source += get_template_definition(
        "g3large_" + lib_name, "copy_g_nd3", in_type, out_type);
    kernel_source += get_template_definition(
        "gn4large_" + lib_name, "copy_g", in_type, out_type, 4);
    kernel_source += get_template_definition(
        "gg1large_" + lib_name, "copy_gg_nd1", in_type, out_type);
    kernel_source += get_template_definition(
        "gg2large_" + lib_name, "copy_gg_nd2", in_type, out_type);
    kernel_source += get_template_definition(
        "gg3large_" + lib_name, "copy_gg_nd3", in_type, out_type);
    kernel_source += get_template_definition(
        "ggn4large_" + lib_name, "copy_gg", in_type, out_type, 4);
    return kernel_source;
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_dynamic_copy_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name, [&]() {
    std::string kernel_source = metal::utils();
    kernel_source += metal::copy();
    auto in_type = get_type_string(in.dtype());
    auto out_type = get_type_string(out.dtype());
    kernel_source += get_template_definition(
        "gg1_" + lib_name, "copy_gg_dynamic_nd1", in_type, out_type, "int");
    kernel_source += get_template_definition(
        "gg2_" + lib_name, "copy_gg_dynamic_nd2", in_type, out_type, "int");
    kernel_source += get_template_definition(
        "gg3_" + lib_name, "copy_gg_dynamic_nd3", in_type, out_type, "int");
    kernel_source += get_template_definition(
        "ggn2_" + lib_name, "copy_gg_dynamic", in_type, out_type, 2, "int");
    kernel_source += get_template_definition(
        "gg1large_" + lib_name, "copy_gg_dynamic_nd1", in_type, out_type);
    kernel_source += get_template_definition(
        "gg2large_" + lib_name, "copy_gg_dynamic_nd2", in_type, out_type);
    kernel_source += get_template_definition(
        "gg3large_" + lib_name, "copy_gg_dynamic_nd3", in_type, out_type);
    kernel_source += get_template_definition(
        "ggn4large_" + lib_name, "copy_gg_dynamic", in_type, out_type, 4);
    return kernel_source;
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_softmax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    bool precise,
    const array& out) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name, [&] {
    std::string kernel_source = metal::utils();
    auto in_type = get_type_string(out.dtype());
    auto acc_type = get_type_string(precise ? float32 : out.dtype());
    kernel_source += metal::softmax();
    kernel_source += get_template_definition(
        "block_" + lib_name, "softmax_single_row", in_type, acc_type);
    kernel_source += get_template_definition(
        "looped_" + lib_name, "softmax_looped", in_type, acc_type);
    return kernel_source;
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_logsumexp_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name, [&] {
    auto t_str = get_type_string(out.dtype());
    std::string kernel_source;
    kernel_source = metal::utils();
    kernel_source += metal::logsumexp();
    kernel_source +=
        get_template_definition("block_" + lib_name, "logsumexp", t_str);
    kernel_source += get_template_definition(
        "looped_" + lib_name, "logsumexp_looped", t_str);
    return kernel_source;
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_scan_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    bool reverse,
    bool inclusive,
    const std::string& reduce_type,
    const array& in,
    const array& out) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name, [&]() {
    auto out_type = get_type_string(out.dtype());
    std::string op = "Cum" + reduce_type + "<" + out_type + ">";
    op[3] = toupper(op[3]);
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::scan();
    const std::array<std::pair<std::string, std::string>, 2> scan_kernels = {{
        {"contig_", "contiguous_scan"},
        {"strided_", "strided_scan"},
    }};
    for (auto& [prefix, kernel] : scan_kernels) {
      kernel_source << get_template_definition(
          prefix + lib_name,
          kernel,
          get_type_string(in.dtype()),
          get_type_string(out.dtype()),
          op,
          in.itemsize() <= 4 ? 4 : 2,
          inclusive,
          reverse);
    }
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_sort_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out,
    int bn,
    int tn) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    auto in_type = get_type_string(in.dtype());
    auto out_type = get_type_string(out.dtype());
    kernel_source << metal::utils() << metal::sort();
    for (bool is_argsort : {true, false}) {
      std::string bool_string = is_argsort ? "true" : "false";
      std::string func_string = is_argsort ? "carg_" : "c_";
      kernel_source << get_template_definition(
          func_string + lib_name,
          "block_sort",
          in_type,
          out_type,
          bool_string,
          bn,
          tn);
      kernel_source << get_template_definition(
          "n" + func_string + lib_name,
          "block_sort_nc",
          in_type,
          out_type,
          bool_string,
          bn,
          tn);
    }
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_mb_sort_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& idx,
    int bn,
    int tn) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::sort();
    std::array<std::pair<std::string, std::string>, 3> kernel_types = {
        {{"sort_", "mb_block_sort"},
         {"partition_", "mb_block_partition"},
         {"merge_", "mb_block_merge"}}};
    for (auto& [name, func] : kernel_types) {
      kernel_source << get_template_definition(
          name + lib_name,
          func,
          get_type_string(in.dtype()),
          get_type_string(idx.dtype()),
          "true",
          bn,
          tn);
    }
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_reduce_init_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& func_name,
    const std::string& op_name,
    const Dtype& out_type) {
  auto lib = d.get_library(kernel_name, [&]() {
    std::string op_type = op_name;
    op_type[0] = std::toupper(op_name[0]);
    auto out_t = get_type_string(out_type);
    std::string op = op_type + "<" + out_t + ">";
    std::string kernel_source = metal::utils();
    kernel_source += metal::reduce_utils();
    kernel_source += metal::reduce();
    kernel_source += get_template_definition(kernel_name, func_name, out_t, op);
    return kernel_source;
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_reduce_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& func_name,
    const std::string& op_name,
    const Dtype& in_type,
    const Dtype& out_type,
    const std::string& idx_t,
    int ndim /* = -1 */,
    int bm /* = -1 */,
    int bn /* = -1 */) {
  auto lib = d.get_library(kernel_name, [&]() {
    std::string op_type = op_name;
    op_type[0] = std::toupper(op_name[0]);
    auto in_t = get_type_string(in_type);
    auto out_t = get_type_string(out_type);
    std::string op = op_type + "<" + out_t + ">";
    std::string kernel_source = metal::utils();
    concatenate(kernel_source, metal::reduce_utils(), metal::reduce());
    if (bm >= 0) {
      kernel_source += get_template_definition(
          kernel_name, func_name, in_t, out_t, op, idx_t, ndim, bm, bn);
    } else if (ndim >= 0) {
      kernel_source += get_template_definition(
          kernel_name, func_name, in_t, out_t, op, idx_t, ndim);
    } else {
      kernel_source += get_template_definition(
          kernel_name, func_name, in_t, out_t, op, idx_t);
    }
    return kernel_source;
  });
  auto st = d.get_kernel(kernel_name, lib);
  return st;
}

MTL::ComputePipelineState* get_steel_gemm_fused_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& out,
    bool transpose_a,
    bool transpose_b,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::gemm()
                  << metal::steel_gemm_fused()
                  << get_template_definition(
                         lib_name,
                         "gemm",
                         get_type_string(out.dtype()),
                         bm,
                         bn,
                         bk,
                         wm,
                         wn,
                         transpose_a,
                         transpose_b);
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_gemm_splitk_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out,
    bool transpose_a,
    bool transpose_b,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn,
    bool mn_aligned,
    bool k_aligned) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::gemm()
                  << metal::steel_gemm_splitk()
                  << get_template_definition(
                         lib_name,
                         "gemm_splitk",
                         get_type_string(in.dtype()),
                         get_type_string(out.dtype()),
                         bm,
                         bn,
                         bk,
                         wm,
                         wn,
                         transpose_a,
                         transpose_b,
                         mn_aligned,
                         k_aligned);
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_steel_gemm_splitk_accum_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out,
    bool axbpy) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::gemm()
                  << metal::steel_gemm_splitk()
                  << get_template_definition(
                         lib_name,
                         axbpy ? "gemm_splitk_accum_axpby"
                               : "gemm_splitk_accum",
                         get_type_string(in.dtype()),
                         get_type_string(out.dtype()));
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_steel_gemm_masked_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out,
    const std::optional<array>& mask_out,
    const std::optional<array>& mask_op,
    bool transpose_a,
    bool transpose_b,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn,
    bool mn_aligned,
    bool k_aligned) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    auto out_mask_type = mask_out.has_value()
        ? get_type_string((*mask_out).dtype())
        : "nomask_t";
    auto op_mask_type =
        mask_op.has_value() ? get_type_string((*mask_op).dtype()) : "nomask_t";
    kernel_source << metal::utils() << metal::gemm()
                  << metal::steel_gemm_masked()
                  << get_template_definition(
                         lib_name,
                         "block_masked_gemm",
                         get_type_string(out.dtype()),
                         out_mask_type,
                         op_mask_type,
                         bm,
                         bn,
                         bk,
                         wm,
                         wn,
                         transpose_a,
                         transpose_b,
                         mn_aligned,
                         k_aligned);
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_steel_gemm_gather_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& out,
    bool transpose_a,
    bool transpose_b,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn,
    bool rhs) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::string kernel_source;
    concatenate(
        kernel_source,
        metal::utils(),
        metal::gemm(),
        metal::steel_gemm_gather(),
        get_template_definition(
            lib_name,
            rhs ? "gather_mm_rhs" : "gather_mm",
            get_type_string(out.dtype()),
            bm,
            bn,
            bk,
            wm,
            wn,
            transpose_a,
            transpose_b));
    return kernel_source;
  });
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_gemm_segmented_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& out,
    bool transpose_a,
    bool transpose_b,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::string kernel_source;
    concatenate(
        kernel_source,
        metal::utils(),
        metal::gemm(),
        metal::steel_gemm_segmented(),
        get_template_definition(
            lib_name,
            "segmented_mm",
            get_type_string(out.dtype()),
            bm,
            bn,
            bk,
            wm,
            wn,
            transpose_a,
            transpose_b));
    return kernel_source;
  });
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

MTL::ComputePipelineState* get_gemv_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out,
    bool transpose_mat,
    int bm,
    int bn,
    int sm,
    int sn,
    int tm,
    int tn,
    bool nc,
    bool axpby) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    // The aligned non-transposed variant is a distinct kernel template
    // ("gemv_al"); the host encodes it in the kernel name prefix.
    bool aligned = kernel_name.compare(0, 8, "gemv_al_") == 0;
    std::ostringstream kernel_source;
    kernel_source << metal::gemv()
                  << get_template_definition(
                         lib_name,
                         transpose_mat ? "gemv_t" : (aligned ? "gemv_al" : "gemv"),
                         get_type_string(out.dtype()),
                         bm,
                         bn,
                         sm,
                         sn,
                         tm,
                         tn,
                         nc ? 1 : 0,
                         axpby ? 1 : 0);
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_gemv_gather_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out,
    bool transpose_mat,
    int bm,
    int bn,
    int sm,
    int sn,
    int tm,
    int tn) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::gemv()
                  << get_template_definition(
                         lib_name,
                         transpose_mat ? "gemv_t_gather" : "gemv_gather",
                         get_type_string(out.dtype()),
                         bm,
                         bn,
                         sm,
                         sn,
                         tm,
                         tn);
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_gemv_masked_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out,
    const std::optional<array>& mask_out,
    const std::optional<array>& mask_op,
    bool transpose_mat,
    int bm,
    int bn,
    int sm,
    int sn,
    int tm,
    int tn,
    bool contiguous) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    auto out_mask_type = mask_out.has_value()
        ? get_type_string((*mask_out).dtype())
        : "nomask_t";
    auto op_mask_type =
        mask_op.has_value() ? get_type_string((*mask_op).dtype()) : "nomask_t";
    kernel_source << metal::utils() << metal::gemv_masked()
                  << get_template_definition(
                         lib_name,
                         (transpose_mat) ? "gemv_t_masked" : "gemv_masked",
                         get_type_string(out.dtype()),
                         out_mask_type,
                         op_mask_type,
                         bm,
                         bn,
                         sm,
                         sn,
                         tm,
                         tn,
                         contiguous ? 0 : 1);
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_steel_conv_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn,
    int n_channel_specialization,
    bool small_filter) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::conv() << metal::steel_conv()
                  << get_template_definition(
                         lib_name,
                         "implicit_gemm_conv_2d",
                         get_type_string(out.dtype()),
                         bm,
                         bn,
                         bk,
                         wm,
                         wn,
                         n_channel_specialization,
                         small_filter);
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_steel_conv_3d_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn,
    bool small_filter) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::conv() << metal::steel_conv_3d()
                  << get_template_definition(
                         lib_name,
                         "implicit_gemm_conv_3d",
                         get_type_string(out.dtype()),
                         bm,
                         bn,
                         bk,
                         wm,
                         wn,
                         small_filter);
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_steel_conv_general_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& out,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::conv()
                  << metal::steel_conv_general()
                  << get_template_definition(
                         lib_name,
                         "implicit_gemm_conv_2d_general",
                         get_type_string(out.dtype()),
                         bm,
                         bn,
                         bk,
                         wm,
                         wn);
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

MTL::ComputePipelineState* get_fft_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const std::string& template_def) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    std::string kernel_string;
    kernel_source << metal::fft() << template_def;
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

MTL::ComputePipelineState* get_quantized_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& template_def,
    const std::string& mode) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::string kernel_source;
    concatenate(
        kernel_source,
        metal::utils(),
        metal::gemm(),
        metal::quantized_utils(),
        (mode == "affine") ? metal::quantized() : metal::fp_quantized(),
        template_def);
    return kernel_source;
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_gather_qmm_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& x,
    int group_size,
    int bits,
    const std::string& mode,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn,
    bool transpose) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::string kernel_source;
    concatenate(
        kernel_source, metal::utils(), metal::quantized_utils(), metal::gemm());
    bool is_affine = mode == "affine";
    concatenate(
        kernel_source,
        is_affine ? metal::quantized() : metal::fp_quantized(),
        get_template_definition(
            lib_name,
            (is_affine ? "affine" : "fp") + std::string("_gather_qmm_rhs"),
            get_type_string(x.dtype()),
            group_size,
            bits,
            bm,
            bn,
            bk,
            wm,
            wn,
            transpose));
    return kernel_source;
  });
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_gemm_fused_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& out,
    bool transpose_a,
    bool transpose_b,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::gemm_nax()
                  << metal::steel_gemm_fused_nax()
                  << get_template_definition(
                         lib_name,
                         "gemm",
                         get_type_string(out.dtype()),
                         bm,
                         bn,
                         bk,
                         wm,
                         wn,
                         transpose_a,
                         transpose_b);
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_gemm_gather_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& out,
    bool transpose_a,
    bool transpose_b,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn,
    bool rhs) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::string kernel_source;
    concatenate(
        kernel_source,
        metal::utils(),
        metal::gemm_nax(),
        metal::steel_gemm_gather_nax(),
        get_template_definition(
            lib_name,
            rhs ? "gather_mm_rhs_nax" : "gather_mm_nax",
            get_type_string(out.dtype()),
            bm,
            bn,
            bk,
            wm,
            wn,
            transpose_a,
            transpose_b));
    return kernel_source;
  });
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_gemm_splitk_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& in,
    bool transpose_a,
    bool transpose_b,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::gemm_nax()
                  << metal::steel_gemm_splitk_nax()
                  << get_template_definition(
                         lib_name,
                         "gemm_splitk_nax",
                         get_type_string(in.dtype()),
                         bm,
                         bn,
                         bk,
                         wm,
                         wn,
                         transpose_a,
                         transpose_b);
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_gemm_segmented_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& out,
    bool transpose_a,
    bool transpose_b,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::gemm_nax()
                  << metal::steel_gemm_segmented_nax()
                  << get_template_definition(
                         lib_name,
                         "segmented_mm_nax",
                         get_type_string(out.dtype()),
                         bm,
                         bn,
                         bk,
                         wm,
                         wn,
                         transpose_a,
                         transpose_b);
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

MTL::ComputePipelineState* get_qmm_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& template_def,
    const std::string& mode) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::string kernel_source;
    concatenate(
        kernel_source,
        metal::utils(),
        metal::gemm_nax(),
        metal::quantized_utils(),
        (mode == "affine") ? metal::quantized_nax() : metal::fp_quantized_nax(),
        template_def);
    return kernel_source;
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_gather_qmm_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& x,
    int group_size,
    int bits,
    const std::string& mode,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn,
    bool transpose) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::string kernel_source;
    concatenate(
        kernel_source,
        metal::utils(),
        metal::gemm_nax(),
        metal::quantized_utils());
    bool is_affine = mode == "affine";
    concatenate(
        kernel_source,
        is_affine ? metal::quantized_nax() : metal::fp_quantized_nax(),
        get_template_definition(
            lib_name,
            (is_affine ? "affine" : "fp") + std::string("_gather_qmm_rhs_nax"),
            get_type_string(x.dtype()),
            group_size,
            bits,
            bm,
            bn,
            bk,
            wm,
            wn,
            transpose));
    return kernel_source;
  });
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_attention_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& q,
    int bq,
    int bk,
    int bd,
    int wm,
    int wn,
    const array& m) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::string kernel_source;
    concatenate(
        kernel_source,
        metal::utils(),
        metal::steel_attention(),
        get_template_definition(
            lib_name,
            "attention",
            get_type_string(q.dtype()),
            bq,
            bk,
            bd,
            wm,
            wn,
            get_type_string(m.dtype())));
    return kernel_source;
  });
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

namespace {

// DARKBLOOM_ATTN_QHOIST: hoist the loop-invariant Q fragments out of the
// steel-attention K-block loop.
//
// The kb loop in attention_nax advances K and V but never Q, yet the QK^T
// phase re-executed `Qtile.load(...)` on every iteration, re-reading the same
// TQ*TD fragments from device memory. At the frozen 512-token prefill window
// (BQ=64 BK=32 BD=128 WM=4 WN=1 => TQ=1 TD=8 TK=2, kb_lim averaging 9.00) that
// is 8 of every 40 device fragment-loads per simdgroup per iteration, ~17.8%
// of the loader traffic, and it drops the LSU:MMA issue ratio from 5.00 to
// 4.11. See notes/21-attn-analysis.md.
//
// DEFAULT OFF, read as `== "1"`. This is an unmeasured arm: the hoist extends
// 8 fragments' live range across the whole loop (+28 registers/thread), and if
// that crosses an occupancy threshold it shows up as a regression, not a win.
// It must not ship until a paired measurement says otherwise.
//
// WHY A #define AND NOT A FUNCTION CONSTANT. The natural home for a host-side
// switch is scaled_dot_product_attention.cpp, but that file is not in
// benchmark.json editablePaths, so it cannot carry one. jit_kernels.cpp is
// editable and is where the JIT source string is assembled, which turns out to
// be the better place anyway: a define is baked into the source text before
// compilation and therefore CANNOT enter the Metal pipeline specialization
// key. A function constant can, and flipping one mid-process forces a second
// pipeline compile that can land inside a timed forward -- the exact trap that
// produced a reproducible 15-24% regression for
// DARKBLOOM_PREFILL_GATHER_RUNSKIP (see the note in quantized.cpp).
//
// Resolved once via a function-local static, so the value is constant for the
// process. Device::get_library caches the built library in memory keyed on
// lib_name with no on-disk persistence, so exactly one variant is ever
// compiled and a fresh process picks up a changed environment cleanly.
//
// Returns a `const char*` rather than a std::string because concatenate()
// takes its arguments by value; the empty string appends nothing.
const char* darkbloom_attn_qhoist_define() {
  static const bool enabled = [] {
    const bool v = env::get_var("DARKBLOOM_ATTN_QHOIST", "") == "1";
    // Same ground-truth discipline the STAGE arms needed: prove the arm is
    // live before trusting its number. A #define that silently fails to reach
    // the source string produces an arm that measures its own control.
    if (env::get_var("DARKBLOOM_ATTN_TRACE", "") == "1") {
      fprintf(stderr, "mlxfast: attn qhoist: enabled=%d\n", int(v));
    }
    return v;
  }();
  return enabled ? "\n#define DARKBLOOM_ATTN_QHOIST 1\n" : "";
}

// DARKBLOOM_ATTN_SGCAUSAL: per-simdgroup causal K-block limit in attention_nax.
//
// kb_lim is derived from the THREADGROUP's row span, but each simdgroup owns
// only kU*TQ of those BQ rows. Simdgroups holding the earlier rows therefore
// run K blocks entirely above their own causal diagonal, where every Stile
// element is masked to neg_inf and the block is the identity. At the frozen
// prefill window that is 16 of 288 simdgroup x K-block units -- 5.56% of the
// executed block work. The trip count and every barrier stay exactly where
// they are (a per-simdgroup trip count would make the in-loop
// threadgroup_barrier be reached a different number of times per simdgroup,
// which is UB); only the work is predicated.
//
// Levels: 1 = skip QK^T, scale, masking and softmax but still run P@V on the
// cleared +0.0 Stile, which is bit-for-bit the P upstream would have fed it.
// 2 = also skip P@V; see the -0.0 note in the kernel.
//
// THE EMPTY STRING SELECTS LEVEL 1, AND THAT IS DELIBERATE. `mlxfast submit`
// packages source, not environment, and the ranked runner sets no DARKBLOOM_*
// variables -- so the UNSET PATH IS THE SHIPPED PATH. A winning flag left at
// `raw == "1"` ships as a literal no-op, which this project has done twice.
// Explicit "0" keeps upstream reachable as the A/B control arm, and level 2
// stays opt-in behind its -0.0 asterisk.
int darkbloom_attn_sgcausal_level() {
  static const int level = [] {
    const auto raw = env::get_var("DARKBLOOM_ATTN_SGCAUSAL", "");
    const int v = (raw == "0") ? 0 : ((raw == "2") ? 2 : 1);
    if (env::get_var("DARKBLOOM_ATTN_TRACE", "") == "1") {
      fprintf(stderr, "mlxfast: attn sgcausal: level=%d\n", v);
    }
    return v;
  }();
  return level;
}

const char* darkbloom_attn_sgcausal_define() {
  switch (darkbloom_attn_sgcausal_level()) {
    case 1:
      return "\n#define DARKBLOOM_ATTN_SGCAUSAL 1\n";
    case 2:
      return "\n#define DARKBLOOM_ATTN_SGCAUSAL 2\n";
    default:
      return "";
  }
}

// Arm state, folded into the kernel/library name.
//
// `d.get_library()` keys its cache on the library name and
// `get_steel_attention_nax_kernel` passed the bare `kernel_name`, which encodes
// tile shape and dtype but NOT which DARKBLOOM define was prepended. Two arms
// therefore shared one cache entry, so whichever ran first could have its
// binary served to the other -- an arm silently measuring its control. Folding
// the state into the name gives each arm its own entry. Empty when every arm is
// off, so the default path keeps the exact upstream name.
const char* darkbloom_attn_variant_suffix() {
  static const std::string suffix = [] {
    std::string s;
    if (darkbloom_attn_qhoist_define()[0] != '\0') {
      s += "_qhoist";
    }
    const int sg = darkbloom_attn_sgcausal_level();
    if (sg != 0) {
      s += "_sgcausal";
      s += char('0' + sg);
    }
    return s;
  }();
  return suffix.c_str();
}

} // namespace

MTL::ComputePipelineState* get_steel_attention_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& q,
    int bq,
    int bk,
    int bd,
    int wm,
    int wn,
    const array& m) {
  // Arm-qualified so a cached library can never be served to the wrong arm.
  // Identical to `kernel_name` when no arm is enabled.
  const std::string lib_name = kernel_name + darkbloom_attn_variant_suffix();
  auto lib = d.get_library(lib_name, [&]() {
    std::string kernel_source;
    concatenate(
        kernel_source,
        metal::utils(),
        darkbloom_attn_qhoist_define(),
        darkbloom_attn_sgcausal_define(),
        metal::steel_attention_nax(),
        get_template_definition(
            lib_name,
            "attention_nax",
            get_type_string(q.dtype()),
            bq,
            bk,
            bd,
            wm,
            wn,
            get_type_string(m.dtype())));
    return kernel_source;
  });
  return d.get_kernel(lib_name, lib, hash_name, func_consts);
}

} // namespace mlx::core
