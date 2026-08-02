import Foundation
import MLX
import MLXFast

/// Fuse the already-last-row final RMSNorm with the 64 activation-group L1
/// bounds consumed by the INT5 vocabulary-head certificate.  The switch is
/// default on so the submitted arm is exercised without runner configuration;
/// setting `DARKBLOOM_FINAL_NORM_L1=0` restores the stock RMSNorm and the
/// shipped v5 coarse kernel byte-for-byte at their call sites.
let lagunaFinalNormL1Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FINAL_NORM_L1"] != "0"

struct LagunaFinalNormL1Output {
    let normalized: MLXArray
    let groupL1Bounds: MLXArray
}

/// Exact twin of MLX's `rms_single_row<bfloat, RMS_N_READS=4>` for a single
/// 2048-element Laguna row.  The launch remains 512 threads / 16 simdgroups,
/// every thread squares the same four consecutive BF16 values, and all three
/// cross-simdgroup barriers and the precise rsqrt are retained.  The output
/// expression is the stock `weight * bfloat(x * inv_rms)` expression.
///
/// Once that exact BF16 row exists in threadgroup memory, the first 64 threads
/// each sum one consecutive 32-element group in index order.  This is the same
/// addition order used by the retained v5 INT5 coarse kernel.  Advancing the
/// non-negative finite FP32 result by one representable value makes the stored
/// value an outward bound; using it in place of the old exact group sum can
/// only widen `delta`, hence can only admit extra exact-head candidates.
private let lagunaFinalNormL1Kernel = MLXFast.metalKernel(
    name: "laguna_final_rmsnorm_l1_bf16_2048_v1",
    inputNames: ["x", "weight"],
    outputNames: ["normalized", "group_l1_bounds"],
    source: """
        constexpr uint axis_size = 2048;
        constexpr uint n_reads = 4;
        constexpr uint simd_size = 32;
        constexpr uint l1_group_size = 32;
        constexpr uint l1_groups = axis_size / l1_group_size;
        constexpr float norm_eps = 1.0e-6f;

        uint lid = thread_position_in_threadgroup.x;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint base = lid * n_reads;

        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[simd_size];
        threadgroup bfloat normalized_row[axis_size];

        // rms_single_row replica: keep the per-thread and cross-simdgroup
        // reduction topology exactly equal to the stock 512-thread kernel.
        float acc = 0.0f;
        for (int i = 0; i < n_reads; ++i) {
            float xi = x[base + i];
            acc += xi * xi;
        }
        acc = simd_sum(acc);
        if (simd_group == 0) {
            local_sums[simd_lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0) {
            local_sums[simd_group] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            acc = simd_sum(local_sums[simd_lane]);
            if (simd_lane == 0) {
                local_inv_mean[0] =
                    metal::precise::rsqrt(acc / axis_size + norm_eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < n_reads; ++i) {
            bfloat value =
                weight[base + i] *
                static_cast<bfloat>(x[base + i] * local_inv_mean[0]);
            normalized[base + i] = value;
            normalized_row[base + i] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < l1_groups) {
            float l1 = 0.0f;
            uint group_base = lid * l1_group_size;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < l1_group_size; ++i) {
                l1 += metal::abs(float(normalized_row[group_base + i]));
            }

            // l1 is non-negative.  One next representable FP32 value is an
            // outward rounding of the exact retained sequential result.  Keep
            // +inf/NaN unchanged; ordinary model activations are finite.
            uint bits = as_type<uint>(l1);
            if ((bits & 0x7F800000u) != 0x7F800000u) {
                bits += 1u;
            }
            group_l1_bounds[lid] = as_type<float>(bits);
        }
        """,
    ensureRowContiguous: true
)

func lagunaFinalNormL1(
    _ input: MLXArray, weight: MLXArray, eps: Float
) -> LagunaFinalNormL1Output? {
    guard lagunaFinalNormL1Enabled,
        eps == Float(LagunaConstants.rmsNormEpsilon),
        input.dtype == .bfloat16,
        weight.dtype == .bfloat16,
        input.shape == [1, 1, LagunaConstants.hiddenSize],
        weight.shape == [LagunaConstants.hiddenSize]
    else {
        return nil
    }

    let outputs = lagunaFinalNormL1Kernel(
        [input, weight],
        grid: (512, 1, 1),
        threadGroup: (512, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.hiddenSize], [64]],
        outputDTypes: [.bfloat16, .float32]
    )
    return LagunaFinalNormL1Output(
        normalized: outputs[0], groupL1Bounds: outputs[1])
}
