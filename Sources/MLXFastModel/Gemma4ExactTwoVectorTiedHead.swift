import MLX

private let gemma4ExactTwoVectorTiedHeadSource = """
    constexpr int kInputWidth = 5376;
    constexpr int kOutputWidth = 262144;
    constexpr int kGroupsPerRow = 84;
    constexpr int kPackedWordsPerRow = 35;
    constexpr int kWeightBytesPerRow = 2688;
    constexpr int kRowsPerSIMD = 4;
    constexpr int kSIMDGroupsPerThreadgroup = 4;

    const int output_row =
        threadgroup_position_in_grid.y
            * kRowsPerSIMD * kSIMDGroupsPerThreadgroup
        + simdgroup_index_in_threadgroup * kRowsPerSIMD;

    const device uchar* weight_bytes =
        reinterpret_cast<const device uchar*>(weight)
        + output_row * kWeightBytesPerRow
        + thread_index_in_simdgroup * 4;
    const device uint* row_packed_indices =
        packed_indices + output_row * kPackedWordsPerRow;
    const device bfloat* input0 = x + thread_index_in_simdgroup * 8;
    const device bfloat* input1 =
        x + kInputWidth + thread_index_in_simdgroup * 8;

    float result0[kRowsPerSIMD] = {0};
    float result1[kRowsPerSIMD] = {0};
    for (int block = 0; block < 21; ++block) {
        float values0[8];
        float values1[8];
        const float input_sum0 =
            gemma4_exact_two_vector_tied_head_load_values(input0, values0);
        const float input_sum1 =
            gemma4_exact_two_vector_tied_head_load_values(input1, values1);
        const uint metadata_column =
            block * 4 + thread_index_in_simdgroup / 8;

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            const device uchar* row_weight =
                weight_bytes + row * kWeightBytesPerRow;
            const ushort metadata_index =
                gemma4_exact_two_vector_tied_head_extract_packed13(
                    row_packed_indices + row * kPackedWordsPerRow,
                    metadata_column);
            const uint pair = lut[metadata_index];
            const uint packed_word =
                reinterpret_cast<const device uint*>(row_weight)[0];
            gemma4_exact_two_vector_tied_head_qdot_4bit(
                packed_word,
                values0,
                values1,
                gemma4_exact_two_vector_tied_head_pair_scale(pair),
                gemma4_exact_two_vector_tied_head_pair_bias(pair),
                input_sum0,
                input_sum1,
                result0[row],
                result1[row]);
        }

        weight_bytes += 128;
        input0 += 256;
        input1 += 256;
    }

    for (int row = 0; row < kRowsPerSIMD; ++row) {
        result0[row] = simd_sum(result0[row]);
        result1[row] = simd_sum(result1[row]);
        if (thread_index_in_simdgroup == 0) {
            {
                const bfloat projected = static_cast<bfloat>(result0[row]);
                const float quotient = static_cast<float>(projected) / cap;
                const float softened = metal::precise::tanh(quotient);
                output[output_row + row] = softened * cap;
            }
            {
                const bfloat projected = static_cast<bfloat>(result1[row]);
                const float quotient = static_cast<float>(projected) / cap;
                const float softened = metal::precise::tanh(quotient);
                output[kOutputWidth + output_row + row] = softened * cap;
            }
        }
    }
    """

private let gemma4ExactTwoVectorTiedHeadHeader = """
    using namespace metal;

    inline ushort gemma4_exact_two_vector_tied_head_extract_packed13(
        const device uint* words,
        uint column
    ) {
        const uint bit_offset = column * 13;
        const uint word_index = bit_offset >> 5;
        const uint shift = bit_offset & 31;
        uint value = words[word_index] >> shift;
        if (shift > 19) {
            value |= words[word_index + 1] << (32 - shift);
        }
        return static_cast<ushort>(value & 0x1fff);
    }

    inline float gemma4_exact_two_vector_tied_head_pair_scale(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair)));
    }

    inline float gemma4_exact_two_vector_tied_head_pair_bias(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair >> 16)));
    }

    inline float gemma4_exact_two_vector_tied_head_load_values(
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

    inline void gemma4_exact_two_vector_tied_head_qdot_4bit(
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

private let gemma4ExactTwoVectorTiedHeadKernel = MLXFast.metalKernel(
    name: "gemma4_exact_two_vector_tied_head_packed13_softcap_qmv_262144x5376_mtp_v2",
    inputNames: ["weight", "packed_indices", "lut", "x", "cap"],
    outputNames: ["output"],
    source: gemma4ExactTwoVectorTiedHeadSource,
    header: gemma4ExactTwoVectorTiedHeadHeader,
    ensureRowContiguous: true
)

func gemma4ExactTwoVectorTiedHead(
    weight: MLXArray,
    packedIndices: MLXArray,
    lut: MLXArray,
    input: MLXArray,
    cap: MLXArray
) -> MLXArray {
    precondition(weight.dtype == .uint32 && weight.shape == [262_144, 672])
    precondition(packedIndices.dtype == .uint32)
    precondition(packedIndices.shape == [262_144, 35])
    precondition(lut.dtype == .uint32 && lut.ndim == 1)
    precondition(input.dtype == .bfloat16 && input.shape == [2, 5_376])
    precondition(cap.dtype == .float32 && cap.size == 1)
    return gemma4ExactTwoVectorTiedHeadKernel(
        [weight, packedIndices, lut, input, cap],
        grid: (32, 65_536, 1),
        threadGroup: (32, 4, 1),
        outputShapes: [[2, 262_144]],
        outputDTypes: [.float32]
    )[0]
}

private let gemma4ExactTwoVectorTiedHeadArgmaxSource = """
    constexpr int kInputWidth = 5376;
    constexpr int kOutputWidth = 262144;
    constexpr int kPackedWordsPerRow = 35;
    constexpr int kWeightBytesPerRow = 2688;
    constexpr int kRowsPerSIMD = 4;
    constexpr int kSIMDGroupsPerThreadgroup = 4;
    constexpr int kPartialCount = 16384;

    const int group_index = threadgroup_position_in_grid.y;
    const int output_row =
        group_index * kRowsPerSIMD * kSIMDGroupsPerThreadgroup
        + simdgroup_index_in_threadgroup * kRowsPerSIMD;

    const device uchar* weight_bytes =
        reinterpret_cast<const device uchar*>(weight)
        + output_row * kWeightBytesPerRow
        + thread_index_in_simdgroup * 4;
    const device uint* row_packed_indices =
        packed_indices + output_row * kPackedWordsPerRow;
    const device bfloat* input0 = x + thread_index_in_simdgroup * 8;
    const device bfloat* input1 =
        x + kInputWidth + thread_index_in_simdgroup * 8;

    float result0[kRowsPerSIMD] = {0};
    float result1[kRowsPerSIMD] = {0};
    for (int block = 0; block < 21; ++block) {
        float values0[8];
        float values1[8];
        const float input_sum0 =
            gemma4_exact_two_vector_tied_head_load_values(input0, values0);
        const float input_sum1 =
            gemma4_exact_two_vector_tied_head_load_values(input1, values1);
        const uint metadata_column =
            block * 4 + thread_index_in_simdgroup / 8;

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            const device uchar* row_weight =
                weight_bytes + row * kWeightBytesPerRow;
            const ushort metadata_index =
                gemma4_exact_two_vector_tied_head_extract_packed13(
                    row_packed_indices + row * kPackedWordsPerRow,
                    metadata_column);
            const uint pair = lut[metadata_index];
            const uint packed_word =
                reinterpret_cast<const device uint*>(row_weight)[0];
            gemma4_exact_two_vector_tied_head_qdot_4bit(
                packed_word,
                values0,
                values1,
                gemma4_exact_two_vector_tied_head_pair_scale(pair),
                gemma4_exact_two_vector_tied_head_pair_bias(pair),
                input_sum0,
                input_sum1,
                result0[row],
                result1[row]);
        }

        weight_bytes += 128;
        input0 += 256;
        input1 += 256;
    }

    for (int row = 0; row < kRowsPerSIMD; ++row) {
        result0[row] = simd_sum(result0[row]);
        result1[row] = simd_sum(result1[row]);
    }

    threadgroup float group_values[8];
    threadgroup uint group_indices[8];
    if (thread_index_in_simdgroup == 0) {
        float best0 = -INFINITY;
        float best1 = -INFINITY;
        uint index0 = static_cast<uint>(output_row);
        uint index1 = index0;
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            const uint vocabulary_index = static_cast<uint>(output_row + row);
            const bfloat projected0 = static_cast<bfloat>(result0[row]);
            const bfloat projected1 = static_cast<bfloat>(result1[row]);
            const float value0 = metal::precise::tanh(
                static_cast<float>(projected0) / cap) * cap;
            const float value1 = metal::precise::tanh(
                static_cast<float>(projected1) / cap) * cap;
            if (value0 > best0) { best0 = value0; index0 = vocabulary_index; }
            if (value1 > best1) { best1 = value1; index1 = vocabulary_index; }
        }
        const uint slot = simdgroup_index_in_threadgroup;
        group_values[slot] = best0;
        group_values[4 + slot] = best1;
        group_indices[slot] = index0;
        group_indices[4 + slot] = index1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index_in_threadgroup == 0
        && thread_index_in_simdgroup < 2) {
        const uint token = thread_index_in_simdgroup;
        float best = -INFINITY;
        uint best_index = 0;
        for (uint group = 0; group < 4; ++group) {
            const uint slot = token * 4 + group;
            const float value = group_values[slot];
            if (value > best) {
                best = value;
                best_index = group_indices[slot];
            }
        }
        const uint destination = token * kPartialCount + group_index;
        partial_values[destination] = best;
        partial_indices[destination] = best_index;
    }
    """

private let gemma4ExactTwoVectorTiedHeadArgmaxKernel = MLXFast.metalKernel(
    name: "gemma4_exact_two_vector_tied_head_packed13_argmax_262144x5376_mtp_v1",
    inputNames: ["weight", "packed_indices", "lut", "x", "cap"],
    outputNames: ["partial_values", "partial_indices"],
    source: gemma4ExactTwoVectorTiedHeadArgmaxSource,
    header: gemma4ExactTwoVectorTiedHeadHeader,
    ensureRowContiguous: true
)

func gemma4ExactTwoVectorTiedHeadArgmax(
    weight: MLXArray,
    packedIndices: MLXArray,
    lut: MLXArray,
    input: MLXArray,
    cap: MLXArray
) -> MLXArray {
    precondition(weight.dtype == .uint32 && weight.shape == [262_144, 672])
    precondition(packedIndices.dtype == .uint32)
    precondition(packedIndices.shape == [262_144, 35])
    precondition(lut.dtype == .uint32 && lut.ndim == 1)
    precondition(input.dtype == .bfloat16 && input.shape == [2, 5_376])
    precondition(cap.dtype == .float32 && cap.size == 1)
    let partials = gemma4ExactTwoVectorTiedHeadArgmaxKernel(
        [weight, packedIndices, lut, input, cap],
        grid: (32, 65_536, 1),
        threadGroup: (32, 4, 1),
        outputShapes: [[2, 16_384], [2, 16_384]],
        outputDTypes: [.float32, .uint32]
    )
    let localWinners = partials[0].argMax(axis: -1).reshaped([2, 1])
    return takeAlong(partials[1], localWinners, axis: 1)
        .squeezed(axis: 1)
        .asType(.int32)
}
