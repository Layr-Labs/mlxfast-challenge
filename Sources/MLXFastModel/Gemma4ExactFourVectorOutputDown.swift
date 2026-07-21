import MLX

private func gemma4ExactFourVectorCoTiledProjectionSource(
    inputWidth: Int,
    indexBits: Int
) -> String {
    precondition([8_192, 16_384, 21_504].contains(inputWidth))
    precondition(indexBits == 12 || indexBits == 13)
    let metadataExtraction = indexBits == 12
        ? """
                const uint low = row_metadata[lane_group];
                const uint packed_high = row_metadata[8 + lane_group / 2];
                const uint high =
                    (packed_high >> ((lane_group & 1) * 4)) & 0x0f;
                const uint metadata_index = low | (high << 8);
            """
        : """
                const uint low = row_metadata[lane_group];
                const uint packed_middle =
                    row_metadata[8 + lane_group / 2];
                const uint middle =
                    (packed_middle >> ((lane_group & 1) * 4)) & 0x0f;
                const uint top = (row_metadata[12] >> lane_group) & 1;
                const uint metadata_index =
                    low | (middle << 8) | (top << 12);
            """
    return """
        constexpr int kInputWidth = \(inputWidth);
        constexpr int kOutputRows = 5376;
        constexpr int kRowsPerSIMD = 4;
        constexpr int kBlockSize = 512;
        constexpr int kBlocks = \(inputWidth / 512);
        constexpr int kWordsPerLane = 2;
        constexpr int kWordsPerRow = 32 * kWordsPerLane;
        constexpr int kWordsPerSIMD = kRowsPerSIMD * kWordsPerRow;
        constexpr int kWeightWordsPerTile = 2 * kWordsPerSIMD;
        constexpr int kMetadataBytesPerRow = \(indexBits);
        constexpr int kMetadataWordsPerTile = \(2 * indexBits);
        constexpr int kPayloadWords =
            kWeightWordsPerTile + kMetadataWordsPerTile;
        constexpr int kWordsPerThreadgroup = kBlocks * kPayloadWords;

        const int threadgroup_row = threadgroup_position_in_grid.y;
        const int simd_group = simdgroup_index_in_threadgroup;
        const int output_row =
            threadgroup_row * 8 + simd_group * kRowsPerSIMD;
        const uint lane_group = thread_index_in_simdgroup / 4;
        const device bfloat* input0 =
            x + thread_index_in_simdgroup * 16;
        const device bfloat* input1 =
            x + kInputWidth + thread_index_in_simdgroup * 16;
        const device bfloat* input2 =
            x + 2 * kInputWidth + thread_index_in_simdgroup * 16;
        const device bfloat* input3 =
            x + 3 * kInputWidth + thread_index_in_simdgroup * 16;
        const device uint* tile_words =
            cotiled_payload + threadgroup_row * kWordsPerThreadgroup;

        float result0[kRowsPerSIMD] = {0};
        float result1[kRowsPerSIMD] = {0};
        float result2[kRowsPerSIMD] = {0};
        float result3[kRowsPerSIMD] = {0};
        for (int block = 0; block < kInputWidth; block += kBlockSize) {
            const device uint* weight_words = tile_words
                + simd_group * kWordsPerSIMD
                + thread_index_in_simdgroup * kWordsPerLane;
            const device uchar* metadata_bytes =
                reinterpret_cast<const device uchar*>(
                    tile_words + kWeightWordsPerTile)
                + simd_group * kRowsPerSIMD * kMetadataBytesPerRow;
            {
                float values0[16];
                float values1[16];
                const float input_sum0 =
                    gemma4_exact_four_vector_projection_load_values(
                        input0, values0);
                const float input_sum1 =
                    gemma4_exact_four_vector_projection_load_values(
                        input1, values1);
                for (int row = 0; row < kRowsPerSIMD; ++row) {
                    const device uint* row_weight_words =
                        weight_words + row * kWordsPerRow;
                    const device uchar* row_metadata =
                        metadata_bytes + row * kMetadataBytesPerRow;
        \(metadataExtraction)
                    const uint pair = lut[metadata_index];
                    gemma4_exact_four_vector_projection_qdot_pair_4bit(
                        row_weight_words[0],
                        row_weight_words[1],
                        values0,
                        values1,
                        gemma4_exact_four_vector_projection_pair_scale(pair),
                        gemma4_exact_four_vector_projection_pair_bias(pair),
                        input_sum0,
                        input_sum1,
                        result0[row],
                        result1[row]);
                }
            }
            {
                float values2[16];
                float values3[16];
                const float input_sum2 =
                    gemma4_exact_four_vector_projection_load_values(
                        input2, values2);
                const float input_sum3 =
                    gemma4_exact_four_vector_projection_load_values(
                        input3, values3);
                for (int row = 0; row < kRowsPerSIMD; ++row) {
                    const device uint* row_weight_words =
                        weight_words + row * kWordsPerRow;
                    const device uchar* row_metadata =
                        metadata_bytes + row * kMetadataBytesPerRow;
        \(metadataExtraction)
                    const uint pair = lut[metadata_index];
                    gemma4_exact_four_vector_projection_qdot_pair_4bit(
                        row_weight_words[0],
                        row_weight_words[1],
                        values2,
                        values3,
                        gemma4_exact_four_vector_projection_pair_scale(pair),
                        gemma4_exact_four_vector_projection_pair_bias(pair),
                        input_sum2,
                        input_sum3,
                        result2[row],
                        result3[row]);
                }
            }
            tile_words += kPayloadWords;
            input0 += kBlockSize;
            input1 += kBlockSize;
            input2 += kBlockSize;
            input3 += kBlockSize;
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result0[row] = simd_sum(result0[row]);
            result1[row] = simd_sum(result1[row]);
            result2[row] = simd_sum(result2[row]);
            result3[row] = simd_sum(result3[row]);
            if (thread_index_in_simdgroup == 0) {
                output[output_row + row] =
                    static_cast<bfloat>(result0[row]);
                output[kOutputRows + output_row + row] =
                    static_cast<bfloat>(result1[row]);
                output[2 * kOutputRows + output_row + row] =
                    static_cast<bfloat>(result2[row]);
                output[3 * kOutputRows + output_row + row] =
                    static_cast<bfloat>(result3[row]);
            }
        }
        """
}

private let gemma4ExactFourVectorProjectionHeader = """
    using namespace metal;

    inline float gemma4_exact_four_vector_projection_pair_scale(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair)));
    }

    inline float gemma4_exact_four_vector_projection_pair_bias(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair >> 16)));
    }

    inline float gemma4_exact_four_vector_projection_load_values(
        const device bfloat* input,
        thread float* values
    ) {
        float sum = 0;
        for (int index = 0; index < 16; index += 4) {
            sum += input[index] + input[index + 1]
                + input[index + 2] + input[index + 3];
            values[index] = input[index];
            values[index + 1] = input[index + 1] / 16.0f;
            values[index + 2] = input[index + 2] / 256.0f;
            values[index + 3] = input[index + 3] / 4096.0f;
        }
        return sum;
    }

    inline void gemma4_exact_four_vector_projection_qdot_pair_4bit(
        uint packed_word0,
        uint packed_word1,
        const thread float* values0,
        const thread float* values1,
        float scale,
        float bias,
        float input_sum0,
        float input_sum1,
        thread float& result0,
        thread float& result1
    ) {
        const ushort packed0 = static_cast<ushort>(packed_word0);
        const ushort packed1 = static_cast<ushort>(packed_word0 >> 16);
        const ushort packed2 = static_cast<ushort>(packed_word1);
        const ushort packed3 = static_cast<ushort>(packed_word1 >> 16);
        float accumulator = 0;
        accumulator +=
            (values0[0] * (packed0 & 0x000f)
            + values0[1] * (packed0 & 0x00f0)
            + values0[2] * (packed0 & 0x0f00)
            + values0[3] * (packed0 & 0xf000));
        accumulator +=
            (values0[4] * (packed1 & 0x000f)
            + values0[5] * (packed1 & 0x00f0)
            + values0[6] * (packed1 & 0x0f00)
            + values0[7] * (packed1 & 0xf000));
        accumulator +=
            (values0[8] * (packed2 & 0x000f)
            + values0[9] * (packed2 & 0x00f0)
            + values0[10] * (packed2 & 0x0f00)
            + values0[11] * (packed2 & 0xf000));
        accumulator +=
            (values0[12] * (packed3 & 0x000f)
            + values0[13] * (packed3 & 0x00f0)
            + values0[14] * (packed3 & 0x0f00)
            + values0[15] * (packed3 & 0xf000));
        result0 += scale * accumulator + input_sum0 * bias;

        accumulator = 0;
        accumulator +=
            (values1[0] * (packed0 & 0x000f)
            + values1[1] * (packed0 & 0x00f0)
            + values1[2] * (packed0 & 0x0f00)
            + values1[3] * (packed0 & 0xf000));
        accumulator +=
            (values1[4] * (packed1 & 0x000f)
            + values1[5] * (packed1 & 0x00f0)
            + values1[6] * (packed1 & 0x0f00)
            + values1[7] * (packed1 & 0xf000));
        accumulator +=
            (values1[8] * (packed2 & 0x000f)
            + values1[9] * (packed2 & 0x00f0)
            + values1[10] * (packed2 & 0x0f00)
            + values1[11] * (packed2 & 0xf000));
        accumulator +=
            (values1[12] * (packed3 & 0x000f)
            + values1[13] * (packed3 & 0x00f0)
            + values1[14] * (packed3 & 0x0f00)
            + values1[15] * (packed3 & 0xf000));
        result1 += scale * accumulator + input_sum1 * bias;
    }
    """

private let gemma4ExactFourVectorSlidingFixed12Output = MLXFast.metalKernel(
    name: "gemma4_exact_four_vector_cotiled_fixed12_output_8192_mtp_v1",
    inputNames: ["cotiled_payload", "lut", "x"],
    outputNames: ["output"],
    source: gemma4ExactFourVectorCoTiledProjectionSource(
        inputWidth: 8_192, indexBits: 12),
    header: gemma4ExactFourVectorProjectionHeader,
    ensureRowContiguous: true
)

private let gemma4ExactFourVectorSlidingFixed13Output = MLXFast.metalKernel(
    name: "gemma4_exact_four_vector_cotiled_fixed13_output_8192_mtp_v1",
    inputNames: ["cotiled_payload", "lut", "x"],
    outputNames: ["output"],
    source: gemma4ExactFourVectorCoTiledProjectionSource(
        inputWidth: 8_192, indexBits: 13),
    header: gemma4ExactFourVectorProjectionHeader,
    ensureRowContiguous: true
)

private let gemma4ExactFourVectorFullFixed12Output = MLXFast.metalKernel(
    name: "gemma4_exact_four_vector_cotiled_fixed12_output_16384_mtp_v1",
    inputNames: ["cotiled_payload", "lut", "x"],
    outputNames: ["output"],
    source: gemma4ExactFourVectorCoTiledProjectionSource(
        inputWidth: 16_384, indexBits: 12),
    header: gemma4ExactFourVectorProjectionHeader,
    ensureRowContiguous: true
)

private let gemma4ExactFourVectorFullFixed13Output = MLXFast.metalKernel(
    name: "gemma4_exact_four_vector_cotiled_fixed13_output_16384_mtp_v1",
    inputNames: ["cotiled_payload", "lut", "x"],
    outputNames: ["output"],
    source: gemma4ExactFourVectorCoTiledProjectionSource(
        inputWidth: 16_384, indexBits: 13),
    header: gemma4ExactFourVectorProjectionHeader,
    ensureRowContiguous: true
)

private let gemma4ExactFourVectorFixed12Down = MLXFast.metalKernel(
    name: "gemma4_exact_four_vector_cotiled_fixed12_down_21504_mtp_v1",
    inputNames: ["cotiled_payload", "lut", "x"],
    outputNames: ["output"],
    source: gemma4ExactFourVectorCoTiledProjectionSource(
        inputWidth: 21_504, indexBits: 12),
    header: gemma4ExactFourVectorProjectionHeader,
    ensureRowContiguous: true
)

func gemma4ExactFourVectorIndexedOutput(
    payload: MLXArray,
    lut: MLXArray,
    input: MLXArray,
    inputWidth: Int,
    indexBits: Int
) -> MLXArray {
    precondition(input.dtype == .bfloat16 && input.shape == [4, inputWidth])
    precondition(lut.dtype == .uint32 && lut.ndim == 1)
    let kernel: MLXFast.MLXFastKernel
    switch (inputWidth, indexBits) {
    case (8_192, 12): kernel = gemma4ExactTwoVectorSlidingFixed12Output
    case (8_192, 13): kernel = gemma4ExactTwoVectorSlidingFixed13Output
    case (16_384, 12): kernel = gemma4ExactTwoVectorFullFixed12Output
    case (16_384, 13): kernel = gemma4ExactTwoVectorFullFixed13Output
    default: preconditionFailure("unsupported exact four-vector output layout")
    }
    return kernel(
        [payload, lut, input],
        grid: (32, 1_344, 2),
        threadGroup: (32, 2, 1),
        outputShapes: [[4, 5_376]],
        outputDTypes: [.bfloat16]
    )[0]
}

func gemma4ExactFourVectorIndexedDown(
    payload: MLXArray,
    lut: MLXArray,
    input: MLXArray
) -> MLXArray {
    precondition(payload.dtype == .uint32 && payload.shape == [672, 42, 536])
    precondition(lut.dtype == .uint32 && lut.ndim == 1)
    precondition(input.dtype == .bfloat16 && input.shape == [4, 21_504])
    return gemma4ExactTwoVectorDownKernel(
        [payload, lut, input],
        grid: (32, 1_344, 2),
        threadGroup: (32, 2, 1),
        outputShapes: [[4, 5_376]],
        outputDTypes: [.bfloat16]
    )[0]
}

private let gemma4ExactFourVectorU16DownSource = """
    constexpr int kInputWidth = 21504;
    constexpr int kOutputRows = 5376;
    constexpr int kGroupsPerRow = 336;
    constexpr int kWeightBytesPerRow = 10752;
    constexpr int kRowsPerSIMD = 4;
    constexpr int kBlockSize = 512;

    const int output_row =
        threadgroup_position_in_grid.y * 8
        + simdgroup_index_in_threadgroup * kRowsPerSIMD;
    const device uchar* weight_bytes =
        reinterpret_cast<const device uchar*>(weight)
        + output_row * kWeightBytesPerRow
        + thread_index_in_simdgroup * 8;
    const device ushort* row_indices =
        indices + output_row * kGroupsPerRow
        + thread_index_in_simdgroup / 4;
    const device bfloat* input0 =
        x + thread_index_in_simdgroup * 16;
    const device bfloat* input1 =
        x + kInputWidth + thread_index_in_simdgroup * 16;
    const device bfloat* input2 =
        x + 2 * kInputWidth + thread_index_in_simdgroup * 16;
    const device bfloat* input3 =
        x + 3 * kInputWidth + thread_index_in_simdgroup * 16;

    float result0[kRowsPerSIMD] = {0};
    float result1[kRowsPerSIMD] = {0};
    float result2[kRowsPerSIMD] = {0};
    float result3[kRowsPerSIMD] = {0};
    for (int block = 0; block < kInputWidth; block += kBlockSize) {
        {
            float values0[16];
            float values1[16];
            const float input_sum0 =
                gemma4_exact_four_vector_projection_load_values(
                    input0, values0);
            const float input_sum1 =
                gemma4_exact_four_vector_projection_load_values(
                    input1, values1);
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uint* row_weight_words =
                    reinterpret_cast<const device uint*>(
                        weight_bytes + row * kWeightBytesPerRow);
                const ushort metadata_index =
                    row_indices[row * kGroupsPerRow];
                const uint pair = lut[metadata_index];
                gemma4_exact_four_vector_projection_qdot_pair_4bit(
                    row_weight_words[0], row_weight_words[1],
                    values0, values1,
                    gemma4_exact_four_vector_projection_pair_scale(pair),
                    gemma4_exact_four_vector_projection_pair_bias(pair),
                    input_sum0, input_sum1,
                    result0[row], result1[row]);
            }
        }
        {
            float values2[16];
            float values3[16];
            const float input_sum2 =
                gemma4_exact_four_vector_projection_load_values(
                    input2, values2);
            const float input_sum3 =
                gemma4_exact_four_vector_projection_load_values(
                    input3, values3);
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uint* row_weight_words =
                    reinterpret_cast<const device uint*>(
                        weight_bytes + row * kWeightBytesPerRow);
                const ushort metadata_index =
                    row_indices[row * kGroupsPerRow];
                const uint pair = lut[metadata_index];
                gemma4_exact_four_vector_projection_qdot_pair_4bit(
                    row_weight_words[0], row_weight_words[1],
                    values2, values3,
                    gemma4_exact_four_vector_projection_pair_scale(pair),
                    gemma4_exact_four_vector_projection_pair_bias(pair),
                    input_sum2, input_sum3,
                    result2[row], result3[row]);
            }
        }
        weight_bytes += 256;
        row_indices += 8;
        input0 += kBlockSize;
        input1 += kBlockSize;
        input2 += kBlockSize;
        input3 += kBlockSize;
    }

    for (int row = 0; row < kRowsPerSIMD; ++row) {
        result0[row] = simd_sum(result0[row]);
        result1[row] = simd_sum(result1[row]);
        result2[row] = simd_sum(result2[row]);
        result3[row] = simd_sum(result3[row]);
        if (thread_index_in_simdgroup == 0) {
            output[output_row + row] = static_cast<bfloat>(result0[row]);
            output[kOutputRows + output_row + row] =
                static_cast<bfloat>(result1[row]);
            output[2 * kOutputRows + output_row + row] =
                static_cast<bfloat>(result2[row]);
            output[3 * kOutputRows + output_row + row] =
                static_cast<bfloat>(result3[row]);
        }
    }
    """

private let gemma4ExactFourVectorU16Down = MLXFast.metalKernel(
    name: "gemma4_exact_four_vector_u16_down_21504_mtp_v1",
    inputNames: ["weight", "indices", "lut", "x"],
    outputNames: ["output"],
    source: gemma4ExactFourVectorU16DownSource,
    header: gemma4ExactFourVectorProjectionHeader,
    ensureRowContiguous: true
)

func gemma4ExactFourVectorU16IndexedDown(
    weight: MLXArray,
    indices: MLXArray,
    lut: MLXArray,
    input: MLXArray
) -> MLXArray {
    precondition(weight.dtype == .uint32 && weight.shape == [5_376, 2_688])
    precondition(indices.dtype == .uint16 && indices.shape == [5_376, 336])
    precondition(lut.dtype == .uint32 && lut.ndim == 1)
    precondition((1...65_536).contains(lut.size))
    precondition(input.dtype == .bfloat16 && input.shape == [4, 21_504])
    return gemma4ExactTwoVectorU16DownKernel(
        [weight, indices, lut, input],
        grid: (32, 1_344, 2),
        threadGroup: (32, 2, 1),
        outputShapes: [[4, 5_376]],
        outputDTypes: [.bfloat16]
    )[0]
}
