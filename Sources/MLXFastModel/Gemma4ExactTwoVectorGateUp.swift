import MLX

let gemma4ExactTwoVectorGateUpSource = """
    constexpr int kInputWidth = 5376;
    constexpr int kOutputRows = 21504;
    constexpr int kRowsPerSIMD = 4;
    constexpr int kSIMDSize = 32;
    constexpr int kWordsPerProjectionBlock =
        kRowsPerSIMD * kSIMDSize;
    constexpr int kWeightWordsPerPair =
        4 * kWordsPerProjectionBlock;
    constexpr int kMetadataBytesPerProjectionPair = 48;
    constexpr int kWordsPerPair = 536;
    constexpr int kPairCount = 10;
    constexpr int kTailWeightWords =
        2 * kWordsPerProjectionBlock;
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
    const device uint* lut = is_up ? up_lut : gate_lut;

    float result0[kRowsPerSIMD] = {0};
    float result1[kRowsPerSIMD] = {0};
    for (int block_pair = 0; block_pair < kPairCount; ++block_pair) {
        uint odd_pairs[kRowsPerSIMD];
        const device uint* even_weight_words = tile_words
            + (is_up ? kWordsPerProjectionBlock : 0)
            + lane;
        const device uint* odd_weight_words = tile_words
            + 2 * kWordsPerProjectionBlock
            + (is_up ? kWordsPerProjectionBlock : 0)
            + lane;
        const device uchar* metadata_bytes =
            reinterpret_cast<const device uchar*>(
                tile_words + kWeightWordsPerPair)
            + (is_up ? kMetadataBytesPerProjectionPair : 0);

        {
            float even_values0[8];
            float even_values1[8];
            const float even_input_sum0 =
                gemma4_exact_two_vector_gate_up_load_values(
                    input0, even_values0);
            const float even_input_sum1 =
                gemma4_exact_two_vector_gate_up_load_values(
                    input1, even_values1);

            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_metadata = metadata_bytes + row * 12;
                const uint even_low = row_metadata[lane_group];
                const uint middle = row_metadata[4 + lane_group];
                const uint odd_high = row_metadata[8 + lane_group];
                const uint even_index =
                    even_low | ((middle & 0x0f) << 8);
                const uint odd_index =
                    (middle >> 4) | (odd_high << 4);
                const uint even_pair = lut[even_index];
                odd_pairs[row] = lut[odd_index];
                const device uint* row_weight_words =
                    even_weight_words + row * kSIMDSize;
                const uint packed_word = row_weight_words[0];
                gemma4_exact_two_vector_gate_up_qdot_4bit(
                    packed_word,
                    even_values0,
                    even_values1,
                    gemma4_exact_two_vector_gate_up_pair_scale(even_pair),
                    gemma4_exact_two_vector_gate_up_pair_bias(even_pair),
                    even_input_sum0,
                    even_input_sum1,
                    result0[row],
                    result1[row]);
            }
        }

        input0 += 256;
        input1 += 256;
        {
            float odd_values0[8];
            float odd_values1[8];
            const float odd_input_sum0 =
                gemma4_exact_two_vector_gate_up_load_values(
                    input0, odd_values0);
            const float odd_input_sum1 =
                gemma4_exact_two_vector_gate_up_load_values(
                    input1, odd_values1);
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const uint odd_pair = odd_pairs[row];
                const device uint* row_weight_words =
                    odd_weight_words + row * kSIMDSize;
                const uint packed_word = row_weight_words[0];
                gemma4_exact_two_vector_gate_up_qdot_4bit(
                    packed_word,
                    odd_values0,
                    odd_values1,
                    gemma4_exact_two_vector_gate_up_pair_scale(odd_pair),
                    gemma4_exact_two_vector_gate_up_pair_bias(odd_pair),
                    odd_input_sum0,
                    odd_input_sum1,
                    result0[row],
                    result1[row]);
            }
        }

        input0 += 256;
        input1 += 256;
        tile_words += kWordsPerPair;
    }

    float tail_values0[8];
    float tail_values1[8];
    const float tail_input_sum0 =
        gemma4_exact_two_vector_gate_up_load_values(input0, tail_values0);
    const float tail_input_sum1 =
        gemma4_exact_two_vector_gate_up_load_values(input1, tail_values1);
    const device uint* tail_weight_words = tile_words
        + (is_up ? kWordsPerProjectionBlock : 0)
        + lane;
    const device uchar* tail_metadata =
        reinterpret_cast<const device uchar*>(
            tile_words + kTailWeightWords)
        + (is_up ? kTailMetadataBytesPerProjection : 0);
    const uint tail_lane_offset = lane_group << 1;
    #pragma clang loop unroll(full)
    for (int row = 0; row < kRowsPerSIMD; ++row) {
        const device uchar* row_metadata = tail_metadata + row * 8;
        const uint low = row_metadata[tail_lane_offset];
        const uint high = row_metadata[tail_lane_offset + 1];
        const uint metadata_index = low | (high << 8);
        const uint pair = lut[metadata_index];
        const device uint* row_weight_words =
            tail_weight_words + row * kSIMDSize;
        const uint packed_word = row_weight_words[0];
        gemma4_exact_two_vector_gate_up_qdot_4bit(
            packed_word,
            tail_values0,
            tail_values1,
            gemma4_exact_two_vector_gate_up_pair_scale(pair),
            gemma4_exact_two_vector_gate_up_pair_bias(pair),
            tail_input_sum0,
            tail_input_sum1,
            result0[row],
            result1[row]);
    }

    threadgroup bfloat projections[2][2][kRowsPerSIMD];
    #pragma clang loop unroll(full)
    for (int row = 0; row < kRowsPerSIMD; ++row) {
        result0[row] = simd_sum(result0[row]);
        result1[row] = simd_sum(result1[row]);
        if (thread_index_in_simdgroup == 0) {
            projections[is_up ? 1 : 0][0][row] =
                static_cast<bfloat>(result0[row]);
            projections[is_up ? 1 : 0][1][row] =
                static_cast<bfloat>(result1[row]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index_in_threadgroup == 0
        && thread_index_in_simdgroup < kRowsPerSIMD
    ) {
        const int row = thread_index_in_simdgroup;
        {
            const bfloat gate = projections[0][0][row];
            const bfloat up = projections[1][0][row];
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
            activated[output_row + row] = gelu * up;
        }
        {
            const bfloat gate = projections[0][1][row];
            const bfloat up = projections[1][1][row];
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
            activated[kOutputRows + output_row + row] = gelu * up;
        }
    }
    """

private let gemma4ExactTwoVectorU16GateUpSource = """
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

    const device uchar* weight_bytes =
        reinterpret_cast<const device uchar*>(weight)
        + output_row * kWeightBytesPerRow
        + lane * 4;
    const device ushort* row_indices =
        indices + output_row * kGroupsPerRow + lane / 8;
    const device bfloat* input0 = x + lane * 8;
    const device bfloat* input1 = x + kInputWidth + lane * 8;

    float result0[kRowsPerSIMD] = {0};
    float result1[kRowsPerSIMD] = {0};
    for (int block = 0; block < 21; ++block) {
        float values0[8];
        float values1[8];
        const float input_sum0 =
            gemma4_exact_two_vector_gate_up_load_values(input0, values0);
        const float input_sum1 =
            gemma4_exact_two_vector_gate_up_load_values(input1, values1);

        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            const device uchar* row_weight =
                weight_bytes + row * kWeightBytesPerRow;
            const ushort metadata_index =
                row_indices[row * kGroupsPerRow];
            const uint pair = is_up
                ? up_lut[metadata_index]
                : gate_lut[metadata_index];
            const uint packed_word =
                *reinterpret_cast<const device uint*>(row_weight);
            gemma4_exact_two_vector_gate_up_qdot_4bit(
                packed_word,
                values0,
                values1,
                gemma4_exact_two_vector_gate_up_pair_scale(pair),
                gemma4_exact_two_vector_gate_up_pair_bias(pair),
                input_sum0,
                input_sum1,
                result0[row],
                result1[row]);
        }

        weight_bytes += 128;
        row_indices += 4;
        input0 += 256;
        input1 += 256;
    }

    threadgroup bfloat projections[2][2][kRowsPerSIMD];
    #pragma clang loop unroll(full)
    for (int row = 0; row < kRowsPerSIMD; ++row) {
        result0[row] = simd_sum(result0[row]);
        if (thread_index_in_simdgroup == 0) {
            projections[is_up ? 1 : 0][0][row] =
                static_cast<bfloat>(result0[row]);
        }
    }
    #pragma clang loop unroll(full)
    for (int row = 0; row < kRowsPerSIMD; ++row) {
        result1[row] = simd_sum(result1[row]);
        if (thread_index_in_simdgroup == 0) {
            projections[is_up ? 1 : 0][1][row] =
                static_cast<bfloat>(result1[row]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index_in_threadgroup == 0
        && thread_index_in_simdgroup < kRowsPerSIMD
    ) {
        const int row = thread_index_in_simdgroup;
        {
            const bfloat gate = projections[0][0][row];
            const bfloat up = projections[1][0][row];
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
            activated[output_row + row] = gelu * up;
        }
        {
            const bfloat gate = projections[0][1][row];
            const bfloat up = projections[1][1][row];
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
            activated[kOutputRows + output_row + row] = gelu * up;
        }
    }
    """

private let gemma4ExactTwoVectorGateUpHeader = """
    using namespace metal;

    inline float gemma4_exact_two_vector_gate_up_pair_scale(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair)));
    }

    inline float gemma4_exact_two_vector_gate_up_pair_bias(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair >> 16)));
    }

    inline float gemma4_exact_two_vector_gate_up_load_values(
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

    inline void gemma4_exact_two_vector_gate_up_qdot_4bit(
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

private let gemma4ExactTwoVectorGateUpKernel = MLXFast.metalKernel(
    name: "gemma4_exact_two_vector_cotiled_fixed12_gate_up_activation_qmv_5376_mtp_v2",
    inputNames: ["cotiled_payload", "gate_lut", "up_lut", "x"],
    outputNames: ["activated"],
    source: gemma4ExactTwoVectorGateUpSource,
    header: gemma4ExactTwoVectorGateUpHeader,
    ensureRowContiguous: true
)

private let gemma4ExactTwoVectorU16GateUpKernel = MLXFast.metalKernel(
    name: "gemma4_exact_two_vector_u16_indexed_gate_up_activation_qmv_5376_mtp_v2",
    inputNames: [
        "gate_weight", "gate_indices", "gate_lut",
        "up_weight", "up_indices", "up_lut", "x",
    ],
    outputNames: ["activated"],
    source: gemma4ExactTwoVectorU16GateUpSource,
    header: gemma4ExactTwoVectorGateUpHeader,
    ensureRowContiguous: true
)

func gemma4ExactTwoVectorGateUpActivated(
    payload: MLXArray,
    gateLUT: MLXArray,
    upLUT: MLXArray,
    input: MLXArray
) -> MLXArray {
    precondition(payload.dtype == .uint32 && payload.shape == [5_376, 5_632])
    precondition(gateLUT.dtype == .uint32 && gateLUT.ndim == 1)
    precondition(upLUT.dtype == .uint32 && upLUT.ndim == 1)
    precondition(input.dtype == .bfloat16 && input.shape == [2, 5_376])
    return gemma4ExactTwoVectorGateUpKernel(
        [payload, gateLUT, upLUT, input],
        grid: (32, 10_752, 1),
        threadGroup: (32, 2, 1),
        outputShapes: [[2, 21_504]],
        outputDTypes: [.bfloat16]
    )[0]
}

func gemma4ExactTwoVectorU16GateUpActivated(
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
    precondition(input.dtype == .bfloat16 && input.shape == [2, 5_376])
    return gemma4ExactTwoVectorU16GateUpKernel(
        [
            gateWeight, gateIndices, gateLUT,
            upWeight, upIndices, upLUT, input,
        ],
        grid: (32, 10_752, 1),
        threadGroup: (32, 2, 1),
        outputShapes: [[2, 21_504]],
        outputDTypes: [.bfloat16]
    )[0]
}
