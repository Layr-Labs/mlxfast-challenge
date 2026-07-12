import MLX

private func makeGemma4FusedAttentionRMSKernel(
    name: String,
    headDim: Int,
    kvHeads: Int
) -> MLXFast.MLXFastKernel {
    precondition(headDim == 256 || headDim == 512)
    precondition(kvHeads == 16 || kvHeads == 4)
    return MLXFast.metalKernel(
        name: name,
        inputNames: ["raw_q", "raw_k", "raw_v", "q_weight", "k_weight"],
        outputNames: ["queries", "keys", "values"],
        source: """
            constexpr uint kHeadDim = \(headDim);
            constexpr uint kQHeads = 32;
            constexpr uint kKVHeads = \(kvHeads);
            constexpr uint kReads = 4;
            constexpr uint kSIMDSize = 32;

            const uint combined_row = threadgroup_position_in_grid.y;
            const bool is_q = combined_row < kQHeads;
            const bool is_k = !is_q && combined_row < kQHeads + kKVHeads;
            const uint projection_row = is_q
                ? combined_row
                : (is_k ? combined_row - kQHeads
                        : combined_row - kQHeads - kKVHeads);

            const device bfloat* input = is_q
                ? raw_q + projection_row * kHeadDim
                : (is_k ? raw_k : raw_v) + projection_row * kHeadDim;
            const device bfloat* weight = is_q ? q_weight : k_weight;
            device bfloat* output = is_q
                ? queries + projection_row * kHeadDim
                : (is_k ? keys : values) + projection_row * kHeadDim;
            const bool has_weight = is_q || is_k;

            float accumulator = 0;
            input += thread_position_in_threadgroup.x * kReads;
            if (thread_position_in_threadgroup.x * kReads + kReads <= kHeadDim) {
                for (uint index = 0; index < kReads; ++index) {
                    const float value = input[index];
                    accumulator += value * value;
                }
            }
            accumulator = simd_sum(accumulator);

            threadgroup float inverse_mean[1];
            threadgroup float local_sums[kSIMDSize];
            if (simdgroup_index_in_threadgroup == 0) {
                local_sums[thread_index_in_simdgroup] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (thread_index_in_simdgroup == 0) {
                local_sums[simdgroup_index_in_threadgroup] = accumulator;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simdgroup_index_in_threadgroup == 0) {
                accumulator = simd_sum(local_sums[thread_index_in_simdgroup]);
                if (thread_index_in_simdgroup == 0) {
                    inverse_mean[0] = metal::precise::rsqrt(
                        accumulator / kHeadDim + 1.0e-6f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            output += thread_position_in_threadgroup.x * kReads;
            const device bfloat* row_weight =
                weight + thread_position_in_threadgroup.x * kReads;
            for (uint index = 0; index < kReads; ++index) {
                const bfloat normalized = static_cast<bfloat>(
                    input[index] * inverse_mean[0]);
                output[index] = has_weight
                    ? row_weight[index] * normalized
                    : static_cast<bfloat>(1.0f) * normalized;
            }
            """,
        header: "using namespace metal;",
        ensureRowContiguous: true
    )
}

private let gemma4FusedSlidingAttentionRMS = makeGemma4FusedAttentionRMSKernel(
    name: "gemma4_fused_sliding_attention_rms_256_v1",
    headDim: 256,
    kvHeads: 16
)

private let gemma4FusedFullAttentionRMS = makeGemma4FusedAttentionRMSKernel(
    name: "gemma4_fused_full_attention_rms_512_v1",
    headDim: 512,
    kvHeads: 4
)

struct FusedAttentionRMSPreparation: @unchecked Sendable {
    let isSliding: Bool
    let headDim: Int
    let kvHeads: Int
    let qNormWeight: MLXArray
    let kNormWeight: MLXArray

    init?(
        isSliding: Bool,
        headDim: Int,
        kvHeads: Int,
        qNormWeight: MLXArray,
        kNormWeight: MLXArray?,
        eps: Float
    ) {
        guard let kNormWeight,
              eps == 1.0e-6,
              qNormWeight.dtype == .bfloat16,
              kNormWeight.dtype == .bfloat16,
              qNormWeight.shape == [headDim],
              kNormWeight.shape == [headDim],
              (isSliding && headDim == 256 && kvHeads == 16)
                || (!isSliding && headDim == 512 && kvHeads == 4)
        else { return nil }
        self.isSliding = isSliding
        self.headDim = headDim
        self.kvHeads = kvHeads
        self.qNormWeight = qNormWeight
        self.kNormWeight = kNormWeight
    }

    func callAsFunction(
        rawQueries: MLXArray,
        rawKeys: MLXArray,
        rawValues: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray) {
        let queryWidth = 32 * headDim
        let kvWidth = kvHeads * headDim
        precondition(rawQueries.dtype == .bfloat16)
        precondition(rawKeys.dtype == .bfloat16)
        precondition(rawQueries.shape == [1, 1, queryWidth])
        precondition(rawKeys.shape == [1, 1, kvWidth])
        let valueInput = rawValues ?? rawKeys
        precondition(valueInput.dtype == .bfloat16)
        precondition(valueInput.shape == [1, 1, kvWidth])

        let threads = headDim / 4
        let kernel = isSliding
            ? gemma4FusedSlidingAttentionRMS
            : gemma4FusedFullAttentionRMS
        let outputs = kernel(
            [rawQueries, rawKeys, valueInput, qNormWeight, kNormWeight],
            grid: (threads, 32 + 2 * kvHeads, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [
                [1, 32, 1, headDim],
                [1, kvHeads, 1, headDim],
                [1, kvHeads, 1, headDim],
            ],
            outputDTypes: [.bfloat16, .bfloat16, .bfloat16]
        )

        return (outputs[0], outputs[1], outputs[2])
    }
}
