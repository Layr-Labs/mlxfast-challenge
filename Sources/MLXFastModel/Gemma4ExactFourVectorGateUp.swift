import MLX

private let gemma4ExactFourVectorGateUpHeader = """
    using namespace metal;

    inline float gemma4_exact_four_vector_gate_up_pair_scale(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair)));
    }

    inline float gemma4_exact_four_vector_gate_up_pair_bias(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair >> 16)));
    }

    inline float gemma4_exact_four_vector_gate_up_load_values(
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

    inline void gemma4_exact_four_vector_gate_up_qdot_pair_4bit(
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

    inline bfloat gemma4_exact_four_vector_gate_up_activate(
        bfloat gate,
        bfloat up
    ) {
        const bfloat cubic0 = static_cast<bfloat>(0.044715f) * gate;
        const bfloat cubic1 = cubic0 * gate;
        const bfloat cubic2 = cubic1 * gate;
        const bfloat inner0 = gate + cubic2;
        const bfloat inner1 =
            static_cast<bfloat>(0.7978845834732056f) * inner0;
        const bfloat tanh_value =
            static_cast<bfloat>(metal::precise::tanh(inner1));
        const bfloat shifted = static_cast<bfloat>(1.0f) + tanh_value;
        const bfloat scaled = static_cast<bfloat>(0.5f) * gate;
        const bfloat gelu = scaled * shifted;
        return gelu * up;
    }
    """

private let gemma4ExactFourVectorFixed12GateUpSource = """
    constexpr int kInputWidth = 5376;
    constexpr int kOutputRows = 21504;
    constexpr int kRowsPerSIMD = 4;
    constexpr int kSIMDSize = 32;
    constexpr int kWordsPerProjectionBlock = kRowsPerSIMD * kSIMDSize;
    constexpr int kWeightWordsPerPair = 4 * kWordsPerProjectionBlock;
    constexpr int kMetadataBytesPerProjectionPair = 48;
    constexpr int kWordsPerPair = 536;
    constexpr int kPairCount = 10;
    constexpr int kTailWeightWords = 2 * kWordsPerProjectionBlock;
    constexpr int kTailMetadataBytesPerProjection = 32;
    constexpr int kWordsPerTail = 272;
    constexpr int kWordsPerThreadgroup =
        kPairCount * kWordsPerPair + kWordsPerTail;

    const bool is_up = simdgroup_index_in_threadgroup == 1;
    const int threadgroup_row = threadgroup_position_in_grid.y;
    const int output_row = threadgroup_row * kRowsPerSIMD;
    const uint lane = thread_index_in_simdgroup;
    const uint lane_group = lane >> 3;
    const device uint* tile_words =
        cotiled_payload + threadgroup_row * kWordsPerThreadgroup;
    const device bfloat* input0 = x + lane * 8;
    const device bfloat* input1 = x + kInputWidth + lane * 8;
    const device bfloat* input2 = x + 2 * kInputWidth + lane * 8;
    const device bfloat* input3 = x + 3 * kInputWidth + lane * 8;
    const device uint* lut = is_up ? up_lut : gate_lut;

    float result0[kRowsPerSIMD] = {0};
    float result1[kRowsPerSIMD] = {0};
    float result2[kRowsPerSIMD] = {0};
    float result3[kRowsPerSIMD] = {0};
    for (int block_pair = 0; block_pair < kPairCount; ++block_pair) {
        const device uint* even_weight_words = tile_words
            + (is_up ? kWordsPerProjectionBlock : 0) + lane;
        const device uint* odd_weight_words = tile_words
            + 2 * kWordsPerProjectionBlock
            + (is_up ? kWordsPerProjectionBlock : 0) + lane;
        const device uchar* metadata_bytes =
            reinterpret_cast<const device uchar*>(
                tile_words + kWeightWordsPerPair)
            + (is_up ? kMetadataBytesPerProjectionPair : 0);
        {
            float values0[8];
            float values1[8];
            const float input_sum0 =
                gemma4_exact_four_vector_gate_up_load_values(input0, values0);
            const float input_sum1 =
                gemma4_exact_four_vector_gate_up_load_values(input1, values1);
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_metadata = metadata_bytes + row * 12;
                const uint even_low = row_metadata[lane_group];
                const uint middle = row_metadata[4 + lane_group];
                const uint odd_high = row_metadata[8 + lane_group];
                const uint even_index = even_low | ((middle & 0x0f) << 8);
                const uint even_pair = lut[even_index];
                const uint packed_word =
                    even_weight_words[row * kSIMDSize];
                gemma4_exact_four_vector_gate_up_qdot_pair_4bit(
                    packed_word, values0, values1,
                    gemma4_exact_four_vector_gate_up_pair_scale(
                        even_pair),
                    gemma4_exact_four_vector_gate_up_pair_bias(
                        even_pair),
                    input_sum0, input_sum1,
                    result0[row], result1[row]);
            }
        }
        input0 += 256;
        input1 += 256;
        {
            float values0[8];
            float values1[8];
            const float input_sum0 =
                gemma4_exact_four_vector_gate_up_load_values(input0, values0);
            const float input_sum1 =
                gemma4_exact_four_vector_gate_up_load_values(input1, values1);
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_metadata = metadata_bytes + row * 12;
                const uint middle = row_metadata[4 + lane_group];
                const uint odd_high = row_metadata[8 + lane_group];
                const uint odd_pair =
                    lut[(middle >> 4) | (odd_high << 4)];
                const uint packed_word = odd_weight_words[row * kSIMDSize];
                gemma4_exact_four_vector_gate_up_qdot_pair_4bit(
                    packed_word, values0, values1,
                    gemma4_exact_four_vector_gate_up_pair_scale(
                        odd_pair),
                    gemma4_exact_four_vector_gate_up_pair_bias(
                        odd_pair),
                    input_sum0, input_sum1,
                    result0[row], result1[row]);
            }
        }
        input0 += 256;
        input1 += 256;
        {
            float values2[8];
            float values3[8];
            const float input_sum2 =
                gemma4_exact_four_vector_gate_up_load_values(input2, values2);
            const float input_sum3 =
                gemma4_exact_four_vector_gate_up_load_values(input3, values3);
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_metadata = metadata_bytes + row * 12;
                const uint even_low = row_metadata[lane_group];
                const uint middle = row_metadata[4 + lane_group];
                const uint even_pair =
                    lut[even_low | ((middle & 0x0f) << 8)];
                const uint packed_word =
                    even_weight_words[row * kSIMDSize];
                gemma4_exact_four_vector_gate_up_qdot_pair_4bit(
                    packed_word, values2, values3,
                    gemma4_exact_four_vector_gate_up_pair_scale(
                        even_pair),
                    gemma4_exact_four_vector_gate_up_pair_bias(
                        even_pair),
                    input_sum2, input_sum3,
                    result2[row], result3[row]);
            }
        }
        input2 += 256;
        input3 += 256;
        {
            float values2[8];
            float values3[8];
            const float input_sum2 =
                gemma4_exact_four_vector_gate_up_load_values(input2, values2);
            const float input_sum3 =
                gemma4_exact_four_vector_gate_up_load_values(input3, values3);
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_metadata = metadata_bytes + row * 12;
                const uint middle = row_metadata[4 + lane_group];
                const uint odd_high = row_metadata[8 + lane_group];
                const uint odd_pair =
                    lut[(middle >> 4) | (odd_high << 4)];
                const uint packed_word = odd_weight_words[row * kSIMDSize];
                gemma4_exact_four_vector_gate_up_qdot_pair_4bit(
                    packed_word, values2, values3,
                    gemma4_exact_four_vector_gate_up_pair_scale(
                        odd_pair),
                    gemma4_exact_four_vector_gate_up_pair_bias(
                        odd_pair),
                    input_sum2, input_sum3,
                    result2[row], result3[row]);
            }
        }
        input2 += 256;
        input3 += 256;
        tile_words += kWordsPerPair;
    }

    const device uint* tail_weight_words = tile_words
        + (is_up ? kWordsPerProjectionBlock : 0) + lane;
    const device uchar* tail_metadata =
        reinterpret_cast<const device uchar*>(
            tile_words + kTailWeightWords)
        + (is_up ? kTailMetadataBytesPerProjection : 0);
    const uint tail_lane_offset = lane_group << 1;
    {
        float values0[8];
        float values1[8];
        const float input_sum0 =
            gemma4_exact_four_vector_gate_up_load_values(input0, values0);
        const float input_sum1 =
            gemma4_exact_four_vector_gate_up_load_values(input1, values1);
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            const device uchar* row_metadata = tail_metadata + row * 8;
            const uint low = row_metadata[tail_lane_offset];
            const uint high = row_metadata[tail_lane_offset + 1];
            const uint tail_pair = lut[low | (high << 8)];
            const uint packed_word = tail_weight_words[row * kSIMDSize];
            gemma4_exact_four_vector_gate_up_qdot_pair_4bit(
                packed_word, values0, values1,
                gemma4_exact_four_vector_gate_up_pair_scale(tail_pair),
                gemma4_exact_four_vector_gate_up_pair_bias(tail_pair),
                input_sum0, input_sum1,
                result0[row], result1[row]);
        }
    }
    {
        float values2[8];
        float values3[8];
        const float input_sum2 =
            gemma4_exact_four_vector_gate_up_load_values(input2, values2);
        const float input_sum3 =
            gemma4_exact_four_vector_gate_up_load_values(input3, values3);
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            const device uchar* row_metadata = tail_metadata + row * 8;
            const uint low = row_metadata[tail_lane_offset];
            const uint high = row_metadata[tail_lane_offset + 1];
            const uint tail_pair = lut[low | (high << 8)];
            const uint packed_word = tail_weight_words[row * kSIMDSize];
            gemma4_exact_four_vector_gate_up_qdot_pair_4bit(
                packed_word, values2, values3,
                gemma4_exact_four_vector_gate_up_pair_scale(tail_pair),
                gemma4_exact_four_vector_gate_up_pair_bias(tail_pair),
                input_sum2, input_sum3,
                result2[row], result3[row]);
        }
    }

    threadgroup bfloat projections[2][4][kRowsPerSIMD];
    for (int row = 0; row < kRowsPerSIMD; ++row) {
        result0[row] = simd_sum(result0[row]);
        result1[row] = simd_sum(result1[row]);
        result2[row] = simd_sum(result2[row]);
        result3[row] = simd_sum(result3[row]);
        if (thread_index_in_simdgroup == 0) {
            projections[is_up ? 1 : 0][0][row] =
                static_cast<bfloat>(result0[row]);
            projections[is_up ? 1 : 0][1][row] =
                static_cast<bfloat>(result1[row]);
            projections[is_up ? 1 : 0][2][row] =
                static_cast<bfloat>(result2[row]);
            projections[is_up ? 1 : 0][3][row] =
                static_cast<bfloat>(result3[row]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index_in_threadgroup == 0
        && thread_index_in_simdgroup < kRowsPerSIMD
    ) {
        const int row = thread_index_in_simdgroup;
        for (int input_row = 0; input_row < 4; ++input_row) {
            activated[input_row * kOutputRows + output_row + row] =
                gemma4_exact_four_vector_gate_up_activate(
                    projections[0][input_row][row],
                    projections[1][input_row][row]);
        }
    }
    """

private let gemma4ExactFourVectorU16GateUpSource = """
    constexpr int kInputWidth = 5376;
    constexpr int kOutputRows = 21504;
    constexpr int kGroupsPerRow = 84;
    constexpr int kWeightBytesPerRow = 2688;
    constexpr int kRowsPerSIMD = 4;

    const bool is_up = simdgroup_index_in_threadgroup == 1;
    const int output_row = threadgroup_position_in_grid.y * kRowsPerSIMD;
    const uint lane = thread_index_in_simdgroup;
    const device uint* weight = is_up ? up_weight : gate_weight;
    const device ushort* indices = is_up ? up_indices : gate_indices;
    const device uint* lut = is_up ? up_lut : gate_lut;
    const device uchar* weight_bytes =
        reinterpret_cast<const device uchar*>(weight)
        + output_row * kWeightBytesPerRow + lane * 4;
    const device ushort* row_indices =
        indices + output_row * kGroupsPerRow + lane / 8;
    const device bfloat* input0 = x + lane * 8;
    const device bfloat* input1 = x + kInputWidth + lane * 8;
    const device bfloat* input2 = x + 2 * kInputWidth + lane * 8;
    const device bfloat* input3 = x + 3 * kInputWidth + lane * 8;

    float result0[kRowsPerSIMD] = {0};
    float result1[kRowsPerSIMD] = {0};
    float result2[kRowsPerSIMD] = {0};
    float result3[kRowsPerSIMD] = {0};
    for (int block = 0; block < 21; ++block) {
        uint pairs[kRowsPerSIMD];
        {
            float values0[8];
            float values1[8];
            const float input_sum0 =
                gemma4_exact_four_vector_gate_up_load_values(input0, values0);
            const float input_sum1 =
                gemma4_exact_four_vector_gate_up_load_values(input1, values1);
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                pairs[row] = lut[row_indices[row * kGroupsPerRow]];
                const uint packed_word =
                    *reinterpret_cast<const device uint*>(row_weight);
                gemma4_exact_four_vector_gate_up_qdot_pair_4bit(
                    packed_word, values0, values1,
                    gemma4_exact_four_vector_gate_up_pair_scale(pairs[row]),
                    gemma4_exact_four_vector_gate_up_pair_bias(pairs[row]),
                    input_sum0, input_sum1,
                    result0[row], result1[row]);
            }
        }
        {
            float values2[8];
            float values3[8];
            const float input_sum2 =
                gemma4_exact_four_vector_gate_up_load_values(input2, values2);
            const float input_sum3 =
                gemma4_exact_four_vector_gate_up_load_values(input3, values3);
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const uint packed_word =
                    *reinterpret_cast<const device uint*>(row_weight);
                gemma4_exact_four_vector_gate_up_qdot_pair_4bit(
                    packed_word, values2, values3,
                    gemma4_exact_four_vector_gate_up_pair_scale(pairs[row]),
                    gemma4_exact_four_vector_gate_up_pair_bias(pairs[row]),
                    input_sum2, input_sum3,
                    result2[row], result3[row]);
            }
        }
        weight_bytes += 128;
        row_indices += 4;
        input0 += 256;
        input1 += 256;
        input2 += 256;
        input3 += 256;
    }

    threadgroup bfloat projections[2][4][kRowsPerSIMD];
    for (int row = 0; row < kRowsPerSIMD; ++row) {
        result0[row] = simd_sum(result0[row]);
        result1[row] = simd_sum(result1[row]);
        result2[row] = simd_sum(result2[row]);
        result3[row] = simd_sum(result3[row]);
        if (thread_index_in_simdgroup == 0) {
            projections[is_up ? 1 : 0][0][row] =
                static_cast<bfloat>(result0[row]);
            projections[is_up ? 1 : 0][1][row] =
                static_cast<bfloat>(result1[row]);
            projections[is_up ? 1 : 0][2][row] =
                static_cast<bfloat>(result2[row]);
            projections[is_up ? 1 : 0][3][row] =
                static_cast<bfloat>(result3[row]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index_in_threadgroup == 0
        && thread_index_in_simdgroup < kRowsPerSIMD
    ) {
        const int row = thread_index_in_simdgroup;
        for (int input_row = 0; input_row < 4; ++input_row) {
            activated[input_row * kOutputRows + output_row + row] =
                gemma4_exact_four_vector_gate_up_activate(
                    projections[0][input_row][row],
                    projections[1][input_row][row]);
        }
    }
    """

/// Four-row fixed12 topology with the exact-two kernel duplicated across two
/// input-pair SIMDgroup pairs in one threadgroup. This avoids the register
/// pressure of carrying four accumulators through one SIMDgroup while still
/// eliminating the second dispatch and placing identical weight reads in the
/// same threadgroup scheduling window.
private let gemma4ExactFourVectorParallelFixed12GateUpSource: String = {
    var source = gemma4ExactTwoVectorGateUpSource
    source = source.replacingOccurrences(
        of: "gemma4_exact_two_vector_gate_up_load_values",
        with: "gemma4_exact_four_vector_gate_up_load_values"
    )
    source = source.replacingOccurrences(
        of: "gemma4_exact_two_vector_gate_up_pair_scale",
        with: "gemma4_exact_four_vector_gate_up_pair_scale"
    )
    source = source.replacingOccurrences(
        of: "gemma4_exact_two_vector_gate_up_pair_bias",
        with: "gemma4_exact_four_vector_gate_up_pair_bias"
    )
    source = source.replacingOccurrences(
        of: "gemma4_exact_two_vector_gate_up_qdot_4bit",
        with: "gemma4_exact_four_vector_gate_up_qdot_pair_4bit"
    )
    source = source.replacingOccurrences(
        of: "const bool is_up = simdgroup_index_in_threadgroup == 1;",
        with: """
            const int input_pair = simdgroup_index_in_threadgroup / 2;
            const bool is_up = (simdgroup_index_in_threadgroup & 1) == 1;
            """
    )
    source = source.replacingOccurrences(
        of: "const device bfloat* input0 = x + lane * 8;",
        with: """
            const device bfloat* input0 =
                x + (2 * input_pair) * kInputWidth + lane * 8;
            """
    )
    source = source.replacingOccurrences(
        of: "const device bfloat* input1 = x + kInputWidth + lane * 8;",
        with: """
            const device bfloat* input1 =
                x + (2 * input_pair + 1) * kInputWidth + lane * 8;
            """
    )
    source = source.replacingOccurrences(
        of: "threadgroup bfloat projections[2][2][kRowsPerSIMD];",
        with: "threadgroup bfloat projections[2][2][2][kRowsPerSIMD];"
    )
    source = source.replacingOccurrences(
        of: "projections[is_up ? 1 : 0][0][row]",
        with: "projections[input_pair][is_up ? 1 : 0][0][row]"
    )
    source = source.replacingOccurrences(
        of: "projections[is_up ? 1 : 0][1][row]",
        with: "projections[input_pair][is_up ? 1 : 0][1][row]"
    )
    source = source.replacingOccurrences(
        of: "simdgroup_index_in_threadgroup == 0",
        with: "!is_up"
    )
    source = source.replacingOccurrences(
        of: "projections[0][0][row]",
        with: "projections[input_pair][0][0][row]"
    )
    source = source.replacingOccurrences(
        of: "projections[1][0][row]",
        with: "projections[input_pair][1][0][row]"
    )
    source = source.replacingOccurrences(
        of: "projections[0][1][row]",
        with: "projections[input_pair][0][1][row]"
    )
    source = source.replacingOccurrences(
        of: "projections[1][1][row]",
        with: "projections[input_pair][1][1][row]"
    )
    source = source.replacingOccurrences(
        of: "activated[output_row + row] = gelu * up;",
        with: """
            activated[(2 * input_pair) * kOutputRows + output_row + row] =
                gelu * up;
            """
    )
    source = source.replacingOccurrences(
        of: "activated[kOutputRows + output_row + row] = gelu * up;",
        with: """
            activated[(2 * input_pair + 1) * kOutputRows + output_row + row] =
                gelu * up;
            """
    )
    return source
}()

private let gemma4ExactFourVectorFixed12GateUp = MLXFast.metalKernel(
    name: "gemma4_exact_four_vector_cotiled_fixed12_gate_up_mtp_v2",
    inputNames: ["cotiled_payload", "gate_lut", "up_lut", "x"],
    outputNames: ["activated"],
    source: gemma4ExactFourVectorParallelFixed12GateUpSource,
    header: gemma4ExactFourVectorGateUpHeader,
    ensureRowContiguous: true
)

private let gemma4ExactFourVectorU16GateUp = MLXFast.metalKernel(
    name: "gemma4_exact_four_vector_u16_gate_up_mtp_v1",
    inputNames: [
        "gate_weight", "gate_indices", "gate_lut",
        "up_weight", "up_indices", "up_lut", "x",
    ],
    outputNames: ["activated"],
    source: gemma4ExactFourVectorU16GateUpSource,
    header: gemma4ExactFourVectorGateUpHeader,
    ensureRowContiguous: true
)

func gemma4ExactFourVectorGateUpActivated(
    payload: MLXArray,
    gateLUT: MLXArray,
    upLUT: MLXArray,
    input: MLXArray
) -> MLXArray {
    precondition(payload.dtype == .uint32 && payload.shape == [5_376, 5_632])
    precondition(gateLUT.dtype == .uint32 && gateLUT.ndim == 1)
    precondition(upLUT.dtype == .uint32 && upLUT.ndim == 1)
    precondition(input.dtype == .bfloat16 && input.shape == [4, 5_376])
    return gemma4ExactFourVectorFixed12GateUp(
        [payload, gateLUT, upLUT, input],
        grid: (32, 21_504, 1),
        threadGroup: (32, 4, 1),
        outputShapes: [[4, 21_504]],
        outputDTypes: [.bfloat16]
    )[0]
}

func gemma4ExactFourVectorU16GateUpActivated(
    gateWeight: MLXArray,
    gateIndices: MLXArray,
    gateLUT: MLXArray,
    upWeight: MLXArray,
    upIndices: MLXArray,
    upLUT: MLXArray,
    input: MLXArray
) -> MLXArray {
    precondition(gateWeight.dtype == .uint32)
    precondition(gateWeight.shape == [21_504, 672])
    precondition(upWeight.dtype == .uint32 && upWeight.shape == gateWeight.shape)
    precondition(gateIndices.dtype == .uint16)
    precondition(gateIndices.shape == [21_504, 84])
    precondition(upIndices.dtype == .uint16 && upIndices.shape == gateIndices.shape)
    precondition(gateLUT.dtype == .uint32 && gateLUT.ndim == 1)
    precondition((1...65_536).contains(gateLUT.size))
    precondition(upLUT.dtype == .uint32 && upLUT.ndim == 1)
    precondition((1...65_536).contains(upLUT.size))
    precondition(input.dtype == .bfloat16 && input.shape == [4, 5_376])
    return gemma4ExactFourVectorU16GateUp(
        [
            gateWeight, gateIndices, gateLUT,
            upWeight, upIndices, upLUT, input,
        ],
        grid: (32, 10_752, 1),
        threadGroup: (32, 2, 1),
        outputShapes: [[4, 21_504]],
        outputDTypes: [.bfloat16]
    )[0]
}
