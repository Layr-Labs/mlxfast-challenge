#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sdpa_vector.h"

using namespace metal;

#define instantiate_sdpa_vector_aggregation(type, value_dim) \
  instantiate_kernel(                                        \
      "sdpa_vector_2pass_2_" #type "_" #value_dim,           \
      sdpa_vector_2pass_2,                                   \
      type,                                                  \
      value_dim)

#define instantiate_sdpa_vector(type, qk_dim, value_dim)       \
  instantiate_kernel(                                          \
      "sdpa_vector_" #type "_" #qk_dim "_" #value_dim,         \
      sdpa_vector,                                             \
      type,                                                    \
      qk_dim,                                                  \
      value_dim)                                               \
  instantiate_kernel(                                          \
      "sdpa_vector_2pass_1_" #type "_" #qk_dim "_" #value_dim, \
      sdpa_vector_2pass_1,                                     \
      type,                                                    \
      qk_dim,                                                  \
      value_dim)                                               \
  instantiate_sdpa_vector_measure(type, qk_dim, value_dim)

#ifdef DARKBLOOM_AOT_SDPA_MEASURE
#define instantiate_sdpa_vector_measure(type, qk_dim, value_dim) \
  instantiate_sdpa_vector_planes(type, qk_dim, value_dim)
#else
#define instantiate_sdpa_vector_measure(type, qk_dim, value_dim)
#endif

#define instantiate_sdpa_vector_planes(type, qk_dim, value_dim)          \
  instantiate_kernel(                                                    \
      "sdpa_vector_" #type "_" #qk_dim "_" #value_dim "_planes1",        \
      sdpa_vector, type, qk_dim, value_dim, 1, false)                    \
  instantiate_kernel(                                                    \
      "sdpa_vector_" #type "_" #qk_dim "_" #value_dim "_planes2",        \
      sdpa_vector, type, qk_dim, value_dim, 2, false)                    \
  instantiate_kernel(                                                    \
      "sdpa_vector_" #type "_" #qk_dim "_" #value_dim "_planes4",        \
      sdpa_vector, type, qk_dim, value_dim, 4, false)

#define instantiate_sdpa_vector_heads(type)      \
  instantiate_sdpa_vector(type, 64, 64)          \
  instantiate_sdpa_vector(type, 96, 96)          \
  instantiate_sdpa_vector(type, 128, 128)        \
  instantiate_sdpa_vector(type, 256, 256)        \
  instantiate_sdpa_vector_aggregation(type, 64)  \
  instantiate_sdpa_vector_aggregation(type, 96)  \
  instantiate_sdpa_vector_aggregation(type, 128) \
  instantiate_sdpa_vector_aggregation(type, 256)

instantiate_sdpa_vector_heads(float)
instantiate_sdpa_vector_heads(bfloat16_t)
instantiate_sdpa_vector_heads(float16_t)
    // clang-format on
