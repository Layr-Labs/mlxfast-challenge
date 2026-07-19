import MLX

private let gemma4ExactFourVectorSlidingQKVSource = """
    constexpr int kInputWidth = 5376;
    constexpr int kQueryRows = 8192;
    constexpr int kKVRows = 4096;
    constexpr int kGroupsPerRow = 84;
    constexpr int kWeightWordsPerRow = 672;
    constexpr int kRowsPerSIMD = 4;
    constexpr int kSIMDSize = 32;

    const int projection = simdgroup_index_in_threadgroup;
    const bool is_q = projection < 2;
    const bool is_k = projection == 2;
    const int output_row = is_q
        ? threadgroup_position_in_grid.y * 8 + projection * kRowsPerSIMD
        : threadgroup_position_in_grid.y * kRowsPerSIMD;
    const int output_width = is_q ? kQueryRows : kKVRows;

    const device uint* weight = is_q
        ? q_weight
        : (is_k ? k_weight : v_weight);
    const device ushort* indices = is_q
        ? q_indices
        : (is_k ? k_indices : v_indices);
    const device uint* lut = is_q
        ? q_lut
        : (is_k ? k_lut : v_lut);
    device bfloat* output = is_q
        ? q_output
        : (is_k ? k_output : v_output);

    const uint lane = thread_index_in_simdgroup;
    const device uint* weight_words = weight
        + output_row * kWeightWordsPerRow + lane;
    const device ushort* row_indices = indices
        + output_row * kGroupsPerRow + lane / 8;
    const device bfloat* input0 = x + lane * 8;
    const device bfloat* input1 = x + kInputWidth + lane * 8;
    const device bfloat* input2 = x + 2 * kInputWidth + lane * 8;
    const device bfloat* input3 = x + 3 * kInputWidth + lane * 8;

    float result0[kRowsPerSIMD] = {0};
    float result1[kRowsPerSIMD] = {0};
    float result2[kRowsPerSIMD] = {0};
    float result3[kRowsPerSIMD] = {0};
    for (int block = 0; block < 21; ++block) {
        {
            float values0[8];
            float values1[8];
            const float input_sum0 =
                gemma4_exact_four_vector_sliding_qkv_load_values(
                    input0, values0);
            const float input_sum1 =
                gemma4_exact_four_vector_sliding_qkv_load_values(
                    input1, values1);
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uint* row_weight_words =
                    weight_words + row * kWeightWordsPerRow;
                const ushort metadata_index =
                    row_indices[row * kGroupsPerRow];
                const uint pair = lut[metadata_index];
                gemma4_exact_four_vector_sliding_qkv_qdot_pair_4bit(
                    row_weight_words[0],
                    values0,
                    values1,
                    gemma4_exact_four_vector_sliding_qkv_pair_scale(pair),
                    gemma4_exact_four_vector_sliding_qkv_pair_bias(pair),
                    input_sum0,
                    input_sum1,
                    result0[row],
                    result1[row]);
            }
        }
        {
            float values2[8];
            float values3[8];
            const float input_sum2 =
                gemma4_exact_four_vector_sliding_qkv_load_values(
                    input2, values2);
            const float input_sum3 =
                gemma4_exact_four_vector_sliding_qkv_load_values(
                    input3, values3);
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uint* row_weight_words =
                    weight_words + row * kWeightWordsPerRow;
                const ushort metadata_index =
                    row_indices[row * kGroupsPerRow];
                const uint pair = lut[metadata_index];
                gemma4_exact_four_vector_sliding_qkv_qdot_pair_4bit(
                    row_weight_words[0],
                    values2,
                    values3,
                    gemma4_exact_four_vector_sliding_qkv_pair_scale(pair),
                    gemma4_exact_four_vector_sliding_qkv_pair_bias(pair),
                    input_sum2,
                    input_sum3,
                    result2[row],
                    result3[row]);
            }
        }

        weight_words += kSIMDSize;
        row_indices += 4;
        input0 += 256;
        input1 += 256;
        input2 += 256;
        input3 += 256;
    }

    #pragma clang loop unroll(full)
    for (int row = 0; row < kRowsPerSIMD; ++row) {
        result0[row] = simd_sum(result0[row]);
        result1[row] = simd_sum(result1[row]);
        result2[row] = simd_sum(result2[row]);
        result3[row] = simd_sum(result3[row]);
        if (thread_index_in_simdgroup == 0) {
            output[output_row + row] = static_cast<bfloat>(result0[row]);
            output[output_width + output_row + row] =
                static_cast<bfloat>(result1[row]);
            output[2 * output_width + output_row + row] =
                static_cast<bfloat>(result2[row]);
            output[3 * output_width + output_row + row] =
                static_cast<bfloat>(result3[row]);
        }
    }
    """

private let gemma4ExactFourVectorSlidingQKVHeader = """
    using namespace metal;

    inline float gemma4_exact_four_vector_sliding_qkv_pair_scale(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair)));
    }

    inline float gemma4_exact_four_vector_sliding_qkv_pair_bias(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair >> 16)));
    }

    inline float gemma4_exact_four_vector_sliding_qkv_load_values(
        const device bfloat* input,
        thread float* values
    ) {
        float sum = 0;
        for (int index = 0; index < 8; index += 4) {
            sum += input[index] + input[index + 1]
                + input[index + 2] + input[index + 3];
            values[index] = input[index];
            values[index + 1] = input[index + 1] / 16.0f;
            values[index + 2] = input[index + 2] / 256.0f;
            values[index + 3] = input[index + 3] / 4096.0f;
        }
        return sum;
    }

    inline void gemma4_exact_four_vector_sliding_qkv_qdot_pair_4bit(
        uint packed_word,
        const thread float* values0,
        const thread float* values1,
        float scale,
        float bias,
        float input_sum0,
        float input_sum1,
        thread float& result0,
        thread float& result1
    ) {
        float accumulator = 0;
        for (int index = 0; index < 2; ++index) {
            const ushort packed = static_cast<ushort>(
                packed_word >> (index * 16));
            accumulator +=
                (values0[4 * index] * (packed & 0x000f)
                + values0[4 * index + 1] * (packed & 0x00f0)
                + values0[4 * index + 2] * (packed & 0x0f00)
                + values0[4 * index + 3] * (packed & 0xf000));
        }
        result0 += scale * accumulator + input_sum0 * bias;

        accumulator = 0;
        for (int index = 0; index < 2; ++index) {
            const ushort packed = static_cast<ushort>(
                packed_word >> (index * 16));
            accumulator +=
                (values1[4 * index] * (packed & 0x000f)
                + values1[4 * index + 1] * (packed & 0x00f0)
                + values1[4 * index + 2] * (packed & 0x0f00)
                + values1[4 * index + 3] * (packed & 0xf000));
        }
        result1 += scale * accumulator + input_sum1 * bias;
    }
    """

private let gemma4ExactFourVectorIndexedSlidingQKV = MLXFast.metalKernel(
    name: "gemma4_exact_four_vector_indexed_sliding_qkv_qmv_5376_mtp_v1",
    inputNames: [
        "q_weight", "q_indices", "q_lut",
        "k_weight", "k_indices", "k_lut",
        "v_weight", "v_indices", "v_lut", "x",
    ],
    outputNames: ["q_output", "k_output", "v_output"],
    source: gemma4ExactFourVectorSlidingQKVSource,
    header: gemma4ExactFourVectorSlidingQKVHeader,
    ensureRowContiguous: true
)

extension FusedSlidingQKVProjection {
    func exactFourVector(_ input: MLXArray) -> (
        queries: MLXArray, keys: MLXArray, values: MLXArray
    ) {
        precondition(input.dtype == .bfloat16 && input.shape == [4, 5_376])
        let outputs = gemma4ExactFourVectorIndexedSlidingQKV(
            [
                q.weight, qMetadata.indices, qMetadata.lut,
                k.weight, kMetadata.indices, kMetadata.lut,
                v.weight, vMetadata.indices, vMetadata.lut, input,
            ],
            grid: (32, 4_096, 1),
            threadGroup: (32, 4, 1),
            outputShapes: [[4, 8_192], [4, 4_096], [4, 4_096]],
            outputDTypes: [.bfloat16, .bfloat16, .bfloat16]
        )
        return (outputs[0], outputs[1], outputs[2])
    }
}

private let gemma4ExactFourVectorFullQKSource = """
    constexpr int kInputWidth = 5376;
    constexpr int kQueryRows = 16384;
    constexpr int kKeyRows = 2048;
    constexpr int kGroupsPerRow = 84;
    constexpr int kWeightWordsPerRow = 672;
    constexpr int kRowsPerSIMD = 4;
    constexpr int kSIMDSize = 32;

    const int projection = simdgroup_index_in_threadgroup;
    const bool is_q = projection < 8;
    const int output_row = is_q
        ? threadgroup_position_in_grid.y * 32 + projection * kRowsPerSIMD
        : threadgroup_position_in_grid.y * kRowsPerSIMD;
    const int output_width = is_q ? kQueryRows : kKeyRows;

    const device uint* weight = is_q ? q_weight : k_weight;
    const device ushort* indices = is_q ? q_indices : k_indices;
    const device uint* lut = is_q ? q_lut : k_lut;
    device bfloat* output = is_q ? q_output : k_output;

    const uint lane = thread_index_in_simdgroup;
    const device uint* weight_words = weight
        + output_row * kWeightWordsPerRow + lane;
    const device ushort* row_indices = indices
        + output_row * kGroupsPerRow + lane / 8;
    const device bfloat* input0 = x + lane * 8;
    const device bfloat* input1 = x + kInputWidth + lane * 8;
    const device bfloat* input2 = x + 2 * kInputWidth + lane * 8;
    const device bfloat* input3 = x + 3 * kInputWidth + lane * 8;

    float result0[kRowsPerSIMD] = {0};
    float result1[kRowsPerSIMD] = {0};
    float result2[kRowsPerSIMD] = {0};
    float result3[kRowsPerSIMD] = {0};
    for (int block = 0; block < 21; ++block) {
        {
            float values0[8];
            float values1[8];
            const float input_sum0 =
                gemma4_exact_four_vector_sliding_qkv_load_values(
                    input0, values0);
            const float input_sum1 =
                gemma4_exact_four_vector_sliding_qkv_load_values(
                    input1, values1);
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uint* row_weight_words =
                    weight_words + row * kWeightWordsPerRow;
                const ushort metadata_index =
                    row_indices[row * kGroupsPerRow];
                const uint pair = lut[metadata_index];
                gemma4_exact_four_vector_sliding_qkv_qdot_pair_4bit(
                    row_weight_words[0],
                    values0,
                    values1,
                    gemma4_exact_four_vector_sliding_qkv_pair_scale(pair),
                    gemma4_exact_four_vector_sliding_qkv_pair_bias(pair),
                    input_sum0,
                    input_sum1,
                    result0[row],
                    result1[row]);
            }
        }
        {
            float values2[8];
            float values3[8];
            const float input_sum2 =
                gemma4_exact_four_vector_sliding_qkv_load_values(
                    input2, values2);
            const float input_sum3 =
                gemma4_exact_four_vector_sliding_qkv_load_values(
                    input3, values3);
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uint* row_weight_words =
                    weight_words + row * kWeightWordsPerRow;
                const ushort metadata_index =
                    row_indices[row * kGroupsPerRow];
                const uint pair = lut[metadata_index];
                gemma4_exact_four_vector_sliding_qkv_qdot_pair_4bit(
                    row_weight_words[0],
                    values2,
                    values3,
                    gemma4_exact_four_vector_sliding_qkv_pair_scale(pair),
                    gemma4_exact_four_vector_sliding_qkv_pair_bias(pair),
                    input_sum2,
                    input_sum3,
                    result2[row],
                    result3[row]);
            }
        }

        weight_words += kSIMDSize;
        row_indices += 4;
        input0 += 256;
        input1 += 256;
        input2 += 256;
        input3 += 256;
    }

    #pragma clang loop unroll(full)
    for (int row = 0; row < kRowsPerSIMD; ++row) {
        result0[row] = simd_sum(result0[row]);
        result1[row] = simd_sum(result1[row]);
        result2[row] = simd_sum(result2[row]);
        result3[row] = simd_sum(result3[row]);
        if (thread_index_in_simdgroup == 0) {
            output[output_row + row] = static_cast<bfloat>(result0[row]);
            output[output_width + output_row + row] =
                static_cast<bfloat>(result1[row]);
            output[2 * output_width + output_row + row] =
                static_cast<bfloat>(result2[row]);
            output[3 * output_width + output_row + row] =
                static_cast<bfloat>(result3[row]);
        }
    }
    """

private let gemma4ExactFourVectorIndexedFullQK = MLXFast.metalKernel(
    name: "gemma4_exact_four_vector_indexed_full_qk_qmv_5376_mtp_v1",
    inputNames: [
        "q_weight", "q_indices", "q_lut",
        "k_weight", "k_indices", "k_lut", "x",
    ],
    outputNames: ["q_output", "k_output"],
    source: gemma4ExactFourVectorFullQKSource,
    header: gemma4ExactFourVectorSlidingQKVHeader,
    ensureRowContiguous: true
)

extension FusedFullQKProjection {
    func exactFourVector(_ input: MLXArray) -> (
        queries: MLXArray, keys: MLXArray
    ) {
        precondition(input.dtype == .bfloat16 && input.shape == [4, 5_376])
        let outputs = gemma4ExactFourVectorIndexedFullQK(
            [
                q.weight, qMetadata.indices, qMetadata.lut,
                k.weight, kMetadata.indices, kMetadata.lut, input,
            ],
            grid: (32, 4_608, 1),
            threadGroup: (32, 9, 1),
            outputShapes: [[4, 16_384], [4, 2_048]],
            outputDTypes: [.bfloat16, .bfloat16]
        )
        return (outputs[0], outputs[1])
    }
}
