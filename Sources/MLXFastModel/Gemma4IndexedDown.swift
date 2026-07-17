import Foundation
import MLX

private let gemma4DownCoTiledPayloadWords = 536
private let gemma4DownCoTiledFixed13PayloadWords = 538

/// Pack each run of eight 12-bit indexes as eight low bytes followed by four
/// bytes containing pairs of high nibbles. This remains 96 bits per eight
/// indexes, but every lane can extract its value without a cross-word branch.
private func gemma4PackFixed12BitPlanes(
    _ indices: [UInt16],
    rows: Int,
    groupsPerRow: Int
) -> [UInt32]? {
    guard rows > 0,
          groupsPerRow > 0,
          groupsPerRow.isMultiple(of: 8)
    else {
        return nil
    }
    let (elementCount, elementOverflow) = rows.multipliedReportingOverflow(
        by: groupsPerRow)
    guard !elementOverflow, indices.count == elementCount else { return nil }
    let blocksPerRow = groupsPerRow / 8
    let (blockCount, blockOverflow) = rows.multipliedReportingOverflow(
        by: blocksPerRow)
    guard !blockOverflow else { return nil }
    let (wordCount, wordOverflow) = blockCount.multipliedReportingOverflow(by: 3)
    guard !wordOverflow else { return nil }

    var words = [UInt32](repeating: 0, count: wordCount)
    for row in 0..<rows {
        let inputRow = row * groupsPerRow
        let outputRow = row * blocksPerRow * 3
        for block in 0..<blocksPerRow {
            let inputBlock = inputRow + block * 8
            let outputBlock = outputRow + block * 3
            for group in 0..<8 {
                let value = UInt32(indices[inputBlock + group])
                guard value < 4_096 else { return nil }
                let lowShift = (group % 4) * 8
                words[outputBlock + group / 4] |= (value & 0xff) << lowShift

                let highShift = (group / 2) * 8 + (group % 2) * 4
                words[outputBlock + 2] |= ((value >> 8) & 0x0f) << highShift
            }
        }
    }
    return words
}

/// Reorder the down weight and fixed-plane metadata into the decode kernel's
/// exact threadgroup-major ownership. Each `(threadgroup, K-block)` payload is
/// `[512 weight U32, 24 metadata U32]`, or 536 U32 total.
private func gemma4MakeCoTiledFixed12DownPayload(
    weight: MLXArray,
    indices: MLXArray,
    materialize: Bool = true
) -> MLXArray? {
    precondition(weight.dtype == .uint32)
    precondition(weight.shape == [5_376, 2_688])
    precondition(indices.dtype == .uint16)
    precondition(indices.shape == [5_376, 336])
    guard let fixedPlanes = gemma4PackFixed12BitPlanes(
        indices.asArray(UInt16.self),
        rows: 5_376,
        groupsPerRow: 336
    ) else {
        return nil
    }

    // output row = 8 * threadgroup + 4 * SIMD-group + row
    // input word = 64 * K-block + 2 * lane + word-within-lane
    let weightPayload = weight
        .reshaped(672, 2, 4, 42, 32, 2)
        .transposed(0, 3, 1, 2, 4, 5)
        .contiguous()
        .reshaped(672, 42, 512)
    let metadataPayload = MLXArray(fixedPlanes, [5_376, 42, 3])
        .reshaped(672, 2, 4, 42, 3)
        .transposed(0, 3, 1, 2, 4)
        .contiguous()
        .reshaped(672, 42, 24)
    let payload = concatenated([weightPayload, metadataPayload], axis: 2)
    precondition(payload.dtype == .uint32)
    precondition(payload.shape == [672, 42, gemma4DownCoTiledPayloadWords])
    if materialize {
        // The sidecar is input-independent and constructed during untimed
        // model initialization, so decode never inherits this conversion.
        eval(payload)
    }
    return payload
}

/// Fixed13 sibling of the promoted down co-tile for the U16-fallback layers
/// whose LUT needs more than 12 bits. The weight plane is byte-identical to
/// the fixed12 payload; the metadata plane packs each `(threadgroup,
/// K-block)` tile as eight 13-byte rows via the promoted output-family
/// packer (eight low bytes, four high-nibble bytes, one top-bit byte), so a
/// tile is `[512 weight U32, 26 metadata U32]`, or 538 U32 total.
private func gemma4MakeCoTiledFixed13DownPayload(
    weight: MLXArray,
    indices: MLXArray,
    materialize: Bool = true
) -> MLXArray? {
    precondition(weight.dtype == .uint32)
    precondition(weight.shape == [5_376, 2_688])
    precondition(indices.dtype == .uint16)
    precondition(indices.shape == [5_376, 336])
    guard let packedMetadata = gemma4PackOutputCoTileMetadata(
        indices.asArray(UInt16.self),
        rows: 5_376,
        groupsPerRow: 336,
        indexBits: 13
    ) else {
        return nil
    }

    // output row = 8 * threadgroup + 4 * SIMD-group + row
    // input word = 64 * K-block + 2 * lane + word-within-lane
    let weightPayload = weight
        .reshaped(672, 2, 4, 42, 32, 2)
        .transposed(0, 3, 1, 2, 4, 5)
        .contiguous()
        .reshaped(672, 42, 512)
    let metadataPayload = MLXArray(packedMetadata, [672, 42, 26])
    let payload = concatenated([weightPayload, metadataPayload], axis: 2)
    precondition(payload.dtype == .uint32)
    precondition(payload.shape == [672, 42, gemma4DownCoTiledFixed13PayloadWords])
    if materialize {
        // The sidecar is input-independent and constructed during untimed
        // model initialization, so decode never inherits this conversion.
        eval(payload)
    }
    return payload
}

private let gemma4Packed12IndexedDownQMV = MLXFast.metalKernel(
    name: "gemma4_packed12_indexed_down_qmv_21504_v1",
    inputNames: ["weight", "packed_indices", "lut", "x"],
    outputNames: ["output"],
    source: """
        constexpr int kInputWidth = 21504;
        constexpr int kOutputWidth = 5376;
        constexpr int kGroupsPerRow = 336;
        constexpr int kPackedWordsPerRow = 126;
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

        // One qmv_fast block consumes eight group-64 metadata entries. Eight
        // 12-bit indexes occupy exactly three U32 words, so the per-lane bit
        // location is invariant across all 42 K blocks.
        const uint lane_group = thread_index_in_simdgroup / 4;
        const uint lane_bit = lane_group * 12;
        const uint lane_word = lane_bit / 32;
        const uint lane_shift = lane_bit % 32;
        const device uint* row_packed_indices =
            packed_indices + output_row * kPackedWordsPerRow + lane_word;
        const device bfloat* input = x + thread_index_in_simdgroup * 16;

        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < kInputWidth; block += kBlockSize) {
            float values[16];
            const float input_sum = gemma4_down_load_values(input, values);

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const device uint* row_words =
                    row_packed_indices + row * kPackedWordsPerRow;
                uint metadata_index = row_words[0] >> lane_shift;
                if (lane_shift > 20) {
                    metadata_index |= row_words[1] << (32 - lane_shift);
                }
                metadata_index &= 0x0fff;
                const uint pair = lut[metadata_index];
                result[row] += gemma4_down_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_down_pair_scale(pair),
                    gemma4_down_pair_bias(pair),
                    input_sum);
            }

            weight_bytes += 256;
            row_packed_indices += 3;
            input += kBlockSize;
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                output[output_row + row] = static_cast<bfloat>(result[row]);
            }
        }
        """,
    header: """
        using namespace metal;

        inline float gemma4_down_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float gemma4_down_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float gemma4_down_load_values(
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

        inline float gemma4_down_qdot_4bit(
            const device uchar* weight,
            const thread float* values,
            float scale,
            float bias,
            float input_sum
        ) {
            const device ushort* packed =
                reinterpret_cast<const device ushort*>(weight);
            float accumulator = 0;
            for (int index = 0; index < 4; ++index) {
                accumulator +=
                    (values[4 * index] * (packed[index] & 0x000f)
                    + values[4 * index + 1] * (packed[index] & 0x00f0)
                    + values[4 * index + 2] * (packed[index] & 0x0f00)
                    + values[4 * index + 3] * (packed[index] & 0xf000));
            }
            return scale * accumulator + input_sum * bias;
        }
        """,
    ensureRowContiguous: true
)

private let gemma4CoTiledFixed12IndexedDownQMV = MLXFast.metalKernel(
    name: "gemma4_cotiled_fixed12_indexed_down_qmv_21504_v1",
    inputNames: ["cotiled_payload", "lut", "x"],
    outputNames: ["output"],
    source: """
        constexpr int kInputWidth = 21504;
        constexpr int kRowsPerSIMD = 4;
        constexpr int kBlockSize = 512;
        constexpr int kBlocks = 42;
        constexpr int kWordsPerLane = 2;
        constexpr int kWordsPerRow = 32 * kWordsPerLane;
        constexpr int kWordsPerSIMD = kRowsPerSIMD * kWordsPerRow;
        constexpr int kWeightWordsPerTile = 2 * kWordsPerSIMD;
        constexpr int kMetadataBytesPerRow = 12;
        constexpr int kMetadataWordsPerTile = 24;
        constexpr int kPayloadWords =
            kWeightWordsPerTile + kMetadataWordsPerTile;
        constexpr int kWordsPerThreadgroup = kBlocks * kPayloadWords;

        const int threadgroup_row = threadgroup_position_in_grid.y;
        const int simd_group = simdgroup_index_in_threadgroup;
        const int output_row =
            threadgroup_row * 8 + simd_group * kRowsPerSIMD;
        const uint lane_group = thread_index_in_simdgroup / 4;
        const device bfloat* input = x + thread_index_in_simdgroup * 16;
        const device uint* tile_words =
            cotiled_payload + threadgroup_row * kWordsPerThreadgroup;

        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < kInputWidth; block += kBlockSize) {
            float values[16];
            const float input_sum = gemma4_cotiled_down_load_values(
                input,
                values);
            const device uint* weight_words = tile_words
                + simd_group * kWordsPerSIMD
                + thread_index_in_simdgroup * kWordsPerLane;
            const device uchar* metadata_bytes =
                reinterpret_cast<const device uchar*>(
                    tile_words + kWeightWordsPerTile)
                + simd_group * kRowsPerSIMD * kMetadataBytesPerRow;

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    reinterpret_cast<const device uchar*>(
                        weight_words + row * kWordsPerRow);
                const device uchar* row_metadata =
                    metadata_bytes + row * kMetadataBytesPerRow;
                const uint low = row_metadata[lane_group];
                const uint packed_high = row_metadata[8 + lane_group / 2];
                const uint high =
                    (packed_high >> ((lane_group & 1) * 4)) & 0x0f;
                const uint metadata_index = low | (high << 8);
                const uint pair = lut[metadata_index];
                result[row] += gemma4_cotiled_down_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_cotiled_down_pair_scale(pair),
                    gemma4_cotiled_down_pair_bias(pair),
                    input_sum);
            }

            tile_words += kPayloadWords;
            input += kBlockSize;
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                output[output_row + row] = static_cast<bfloat>(result[row]);
            }
        }
        """,
    header: """
        using namespace metal;

        inline float gemma4_cotiled_down_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float gemma4_cotiled_down_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float gemma4_cotiled_down_load_values(
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

        inline float gemma4_cotiled_down_qdot_4bit(
            const device uchar* weight,
            const thread float* values,
            float scale,
            float bias,
            float input_sum
        ) {
            const device ushort* packed =
                reinterpret_cast<const device ushort*>(weight);
            float accumulator = 0;
            for (int index = 0; index < 4; ++index) {
                accumulator +=
                    (values[4 * index] * (packed[index] & 0x000f)
                    + values[4 * index + 1] * (packed[index] & 0x00f0)
                    + values[4 * index + 2] * (packed[index] & 0x0f00)
                    + values[4 * index + 3] * (packed[index] & 0xf000));
            }
            return scale * accumulator + input_sum * bias;
        }
        """,
    ensureRowContiguous: true
)

private let gemma4CoTiledFixed13IndexedDownQMV = MLXFast.metalKernel(
    name: "gemma4_cotiled_fixed13_indexed_down_qmv_21504_v1",
    inputNames: ["cotiled_payload", "lut", "x"],
    outputNames: ["output"],
    source: """
        constexpr int kInputWidth = 21504;
        constexpr int kRowsPerSIMD = 4;
        constexpr int kBlockSize = 512;
        constexpr int kBlocks = 42;
        constexpr int kWordsPerLane = 2;
        constexpr int kWordsPerRow = 32 * kWordsPerLane;
        constexpr int kWordsPerSIMD = kRowsPerSIMD * kWordsPerRow;
        constexpr int kWeightWordsPerTile = 2 * kWordsPerSIMD;
        constexpr int kMetadataBytesPerRow = 13;
        constexpr int kMetadataWordsPerTile = 26;
        constexpr int kPayloadWords =
            kWeightWordsPerTile + kMetadataWordsPerTile;
        constexpr int kWordsPerThreadgroup = kBlocks * kPayloadWords;

        const int threadgroup_row = threadgroup_position_in_grid.y;
        const int simd_group = simdgroup_index_in_threadgroup;
        const int output_row =
            threadgroup_row * 8 + simd_group * kRowsPerSIMD;
        const uint lane_group = thread_index_in_simdgroup / 4;
        const device bfloat* input = x + thread_index_in_simdgroup * 16;
        const device uint* tile_words =
            cotiled_payload + threadgroup_row * kWordsPerThreadgroup;

        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < kInputWidth; block += kBlockSize) {
            float values[16];
            const float input_sum = gemma4_cotiled13_down_load_values(
                input,
                values);
            const device uint* weight_words = tile_words
                + simd_group * kWordsPerSIMD
                + thread_index_in_simdgroup * kWordsPerLane;
            const device uchar* metadata_bytes =
                reinterpret_cast<const device uchar*>(
                    tile_words + kWeightWordsPerTile)
                + simd_group * kRowsPerSIMD * kMetadataBytesPerRow;

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    reinterpret_cast<const device uchar*>(
                        weight_words + row * kWordsPerRow);
                const device uchar* row_metadata =
                    metadata_bytes + row * kMetadataBytesPerRow;
                const uint low = row_metadata[lane_group];
                const uint packed_high = row_metadata[8 + lane_group / 2];
                const uint high =
                    (packed_high >> ((lane_group & 1) * 4)) & 0x0f;
                const uint top = (row_metadata[12] >> lane_group) & 1;
                const uint metadata_index =
                    low | (high << 8) | (top << 12);
                const uint pair = lut[metadata_index];
                result[row] += gemma4_cotiled13_down_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_cotiled13_down_pair_scale(pair),
                    gemma4_cotiled13_down_pair_bias(pair),
                    input_sum);
            }

            tile_words += kPayloadWords;
            input += kBlockSize;
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                output[output_row + row] = static_cast<bfloat>(result[row]);
            }
        }
        """,
    header: """
        using namespace metal;

        inline float gemma4_cotiled13_down_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float gemma4_cotiled13_down_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float gemma4_cotiled13_down_load_values(
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

        inline float gemma4_cotiled13_down_qdot_4bit(
            const device uchar* weight,
            const thread float* values,
            float scale,
            float bias,
            float input_sum
        ) {
            const device ushort* packed =
                reinterpret_cast<const device ushort*>(weight);
            float accumulator = 0;
            for (int index = 0; index < 4; ++index) {
                accumulator +=
                    (values[4 * index] * (packed[index] & 0x000f)
                    + values[4 * index + 1] * (packed[index] & 0x00f0)
                    + values[4 * index + 2] * (packed[index] & 0x0f00)
                    + values[4 * index + 3] * (packed[index] & 0xf000));
            }
            return scale * accumulator + input_sum * bias;
        }
        """,
    ensureRowContiguous: true
)

private let gemma4IndexedDownQMV = MLXFast.metalKernel(
    name: "gemma4_indexed_down_qmv_21504_v1",
    inputNames: ["weight", "indices", "lut", "x"],
    outputNames: ["output"],
    source: """
        constexpr int kInputWidth = 21504;
        constexpr int kOutputWidth = 5376;
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
        const device bfloat* input = x + thread_index_in_simdgroup * 16;

        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < kInputWidth; block += kBlockSize) {
            float values[16];
            const float input_sum = gemma4_down_load_values(input, values);

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const ushort metadata_index =
                    row_indices[row * kGroupsPerRow];
                const uint pair = lut[metadata_index];
                result[row] += gemma4_down_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_down_pair_scale(pair),
                    gemma4_down_pair_bias(pair),
                    input_sum);
            }

            weight_bytes += 256;
            row_indices += 8;
            input += kBlockSize;
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                output[output_row + row] = static_cast<bfloat>(result[row]);
            }
        }
        """,
    header: """
        using namespace metal;

        inline float gemma4_down_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float gemma4_down_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float gemma4_down_load_values(
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

        inline float gemma4_down_qdot_4bit(
            const device uchar* weight,
            const thread float* values,
            float scale,
            float bias,
            float input_sum
        ) {
            const device ushort* packed =
                reinterpret_cast<const device ushort*>(weight);
            float accumulator = 0;
            for (int index = 0; index < 4; ++index) {
                accumulator +=
                    (values[4 * index] * (packed[index] & 0x000f)
                    + values[4 * index + 1] * (packed[index] & 0x00f0)
                    + values[4 * index + 2] * (packed[index] & 0x0f00)
                    + values[4 * index + 3] * (packed[index] & 0xf000));
            }
            return scale * accumulator + input_sum * bias;
        }
        """,
    ensureRowContiguous: true
)

/// Losslessly row-pack U16 indexes into a little-endian 12-bit U32 stream.
/// Rows are independently padded to a whole U32 so a Metal extraction can
/// never cross into the following row. nil is fail-closed for malformed input
/// or an index that does not fit the 12-bit representation.
func gemma4Pack12BitIndices(
    _ indices: [UInt16],
    rows: Int,
    groupsPerRow: Int
) -> [UInt32]? {
    guard rows > 0, groupsPerRow > 0 else { return nil }
    let (elementCount, elementOverflow) = rows.multipliedReportingOverflow(
        by: groupsPerRow)
    guard !elementOverflow, indices.count == elementCount else { return nil }
    let (rowBits, rowBitsOverflow) = groupsPerRow.multipliedReportingOverflow(by: 12)
    guard !rowBitsOverflow else { return nil }
    let (roundedRowBits, roundedOverflow) = rowBits.addingReportingOverflow(31)
    guard !roundedOverflow else { return nil }
    let wordsPerRow = roundedRowBits / 32
    let (wordCount, wordOverflow) = rows.multipliedReportingOverflow(by: wordsPerRow)
    guard !wordOverflow else { return nil }

    var words = [UInt32](repeating: 0, count: wordCount)
    for row in 0..<rows {
        let inputBase = row * groupsPerRow
        let outputBase = row * wordsPerRow
        for group in 0..<groupsPerRow {
            let value = UInt32(indices[inputBase + group])
            guard value < 4_096 else { return nil }
            let bit = group * 12
            let word = bit / 32
            let shift = bit % 32
            words[outputBase + word] |= value << shift
            if shift > 20 {
                words[outputBase + word + 1] |= value >> (32 - shift)
            }
        }
    }
    return words
}

private func environmentFlag(_ name: String, default defaultValue: Bool) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return defaultValue
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

private let gemma4DownCoTiledFixed12Enabled = environmentFlag(
    "DARKBLOOM_DOWN_COTILED_FIXED12",
    default: true
)

private let gemma4VerifyDownCoTiledFixed12Bits = environmentFlag(
    "DARKBLOOM_VERIFY_DOWN_COTILED_FIXED12_BITS",
    default: false
)

private let gemma4DownCoTiledFixed13Enabled = environmentFlag(
    "DARKBLOOM_DOWN_COTILED_FIXED13",
    default: true
)

private let gemma4VerifyDownCoTiledFixed13Bits = environmentFlag(
    "DARKBLOOM_VERIFY_DOWN_COTILED_FIXED13_BITS",
    default: false
)

private struct Packed12DownMetadata: @unchecked Sendable {
    let words: MLXArray

    init?(metadata: IndexedAffineMetadata) {
        guard metadata.indices.dtype == .uint16,
              metadata.indices.shape == [5_376, 336],
              metadata.lut.dtype == .uint32,
              metadata.lut.ndim == 1,
              (1...4_096).contains(metadata.lut.size),
              let packed = gemma4Pack12BitIndices(
                  metadata.indices.asArray(UInt16.self),
                  rows: 5_376,
                  groupsPerRow: 336
              )
        else {
            return nil
        }
        let words = MLXArray(packed, [5_376, 126])
        // This metadata is input-independent and constructed while the model
        // is still in its untimed initialization phase. Materialize it now so
        // the first single-token down projection cannot inherit a deferred
        // host-to-Metal conversion for roughly 2.7 MB per eligible layer.
        eval(words)
        self.words = words
    }
}

private struct CoTiledFixed12DownPayload: @unchecked Sendable {
    let words: MLXArray

    init?(
        projection: FastQuantizedProjection,
        metadata: IndexedAffineMetadata
    ) {
        guard metadata.lut.dtype == .uint32,
              metadata.lut.ndim == 1,
              (1...4_096).contains(metadata.lut.size),
              let words = gemma4MakeCoTiledFixed12DownPayload(
                  weight: projection.weight,
                  indices: metadata.indices
              )
        else {
            return nil
        }
        self.words = words
    }
}

/// U16-fallback layers only: a LUT wider than 12 bits disqualifies every
/// promoted fixed12 form, so this payload can never coexist with (or shadow)
/// the fixed12 co-tile, the packed12 sidecar, or their rollback switches.
private struct CoTiledFixed13DownPayload: @unchecked Sendable {
    let words: MLXArray

    init?(
        projection: FastQuantizedProjection,
        metadata: IndexedAffineMetadata
    ) {
        guard metadata.lut.dtype == .uint32,
              metadata.lut.ndim == 1,
              (4_097...8_192).contains(metadata.lut.size),
              let words = gemma4MakeCoTiledFixed13DownPayload(
                  weight: projection.weight,
                  indices: metadata.indices
              )
        else {
            return nil
        }
        self.words = words
    }
}

func supportsGemma4IndexedDown(
    projection: FastQuantizedProjection,
    metadata: IndexedAffineMetadata
) -> Bool {
    guard let biases = projection.biases else { return false }
    return projection.groupSize == 64
        && projection.bits == 4
        && projection.weight.dtype == .uint32
        && projection.weight.shape == [5_376, 2_688]
        && projection.scales.dtype == .bfloat16
        && projection.scales.shape == [5_376, 336]
        && biases.dtype == .bfloat16
        && biases.shape == [5_376, 336]
        && metadata.indices.dtype == .uint16
        && metadata.indices.shape == [5_376, 336]
        && metadata.lut.dtype == .uint32
        && metadata.lut.ndim == 1
        && (1...65_536).contains(metadata.lut.size)
}

struct IndexedDownProjection: @unchecked Sendable {
    let projection: FastQuantizedProjection
    let metadata: IndexedAffineMetadata
    private let packed12: Packed12DownMetadata?
    private let coTiledFixed12: CoTiledFixed12DownPayload?
    private let coTiledFixed13: CoTiledFixed13DownPayload?
    private let verifyPacked12Bits: Bool
    private let verifyCoTiledFixed12Bits: Bool
    private let verifyCoTiledFixed13Bits: Bool

    init(
        projection: FastQuantizedProjection,
        metadata: IndexedAffineMetadata
    ) {
        precondition(supportsGemma4IndexedDown(
            projection: projection,
            metadata: metadata
        ))
        self.projection = projection
        self.metadata = metadata
        let packedIndicesEnabled = environmentFlag(
            "MLXFAST_PACKED_DOWN_INDICES",
            default: true
        )
        let verifyPacked12Bits = environmentFlag(
            "MLXFAST_VERIFY_PACKED_DOWN_BITS",
            default: false
        )
        self.verifyPacked12Bits = verifyPacked12Bits
        self.verifyCoTiledFixed12Bits =
            gemma4VerifyDownCoTiledFixed12Bits
        self.verifyCoTiledFixed13Bits =
            gemma4VerifyDownCoTiledFixed13Bits

        let wantsCoTiled = packedIndicesEnabled
            && (gemma4DownCoTiledFixed12Enabled
                || gemma4VerifyDownCoTiledFixed12Bits)
        let coTiledFixed12 = wantsCoTiled
            ? CoTiledFixed12DownPayload(
                projection: projection,
                metadata: metadata)
            : nil
        self.coTiledFixed12 = coTiledFixed12

        // The fixed13 payload's own guard restricts it to LUTs the fixed12
        // forms reject, so promoted layers and rollback semantics are
        // untouched.
        let wantsCoTiledFixed13 = gemma4DownCoTiledFixed13Enabled
            || gemma4VerifyDownCoTiledFixed13Bits
        self.coTiledFixed13 = wantsCoTiledFixed13
            ? CoTiledFixed13DownPayload(
                projection: projection,
                metadata: metadata)
            : nil

        // The production co-tile already contains its fixed12 metadata. Keep
        // the promoted packed12 sidecar only for rollback, fallback, or a
        // verifier so the default path avoids another ~2.6 MiB per layer.
        let needsPacked12 = packedIndicesEnabled
            && (coTiledFixed12 == nil
                || !gemma4DownCoTiledFixed12Enabled
                || gemma4VerifyDownCoTiledFixed12Bits
                || verifyPacked12Bits)
        self.packed12 = needsPacked12
            ? Packed12DownMetadata(metadata: metadata)
            : nil
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 21_504])
        let outputShape = [1, 1, 5_376]

        var verifiedPacked12: MLXArray?
        if let coTiledFixed12 {
            let candidate = gemma4CoTiledFixed12IndexedDownQMV(
                [coTiledFixed12.words, metadata.lut, input],
                grid: (32, 1_344, 1),
                threadGroup: (32, 2, 1),
                outputShapes: [outputShape],
                outputDTypes: [.bfloat16]
            )[0]

            if verifyCoTiledFixed12Bits {
                guard let packed12 else {
                    preconditionFailure(
                        "co-tiled fixed12 verifier requires packed12 metadata"
                    )
                }
                let reference = packed12Output(
                    input,
                    packed12: packed12,
                    outputShape: outputShape
                )
                let matches = arrayEqual(
                    candidate.view(dtype: .uint16),
                    reference.view(dtype: .uint16)
                )
                eval(matches)
                precondition(
                    matches.item(Bool.self),
                    "co-tiled fixed12 down projection differs from "
                        + "promoted packed12 qmv"
                )
                verifyPacked12Output(
                    reference,
                    input: input,
                    outputShape: outputShape
                )
                verifiedPacked12 = reference
            }

            if gemma4DownCoTiledFixed12Enabled {
                if verifyPacked12Bits && verifiedPacked12 == nil {
                    guard let packed12 else {
                        preconditionFailure(
                            "packed12 verifier requires packed12 metadata"
                        )
                    }
                    let reference = packed12Output(
                        input,
                        packed12: packed12,
                        outputShape: outputShape
                    )
                    verifyPacked12Output(
                        reference,
                        input: input,
                        outputShape: outputShape
                    )
                }
                return candidate
            }
        }

        // U16-fallback layers (a LUT wider than 12 bits): the fixed12 forms
        // above are structurally absent, so this payload is the only
        // alternative to the promoted U16 kernel.
        if let coTiledFixed13 {
            let candidate = gemma4CoTiledFixed13IndexedDownQMV(
                [coTiledFixed13.words, metadata.lut, input],
                grid: (32, 1_344, 1),
                threadGroup: (32, 2, 1),
                outputShapes: [outputShape],
                outputDTypes: [.bfloat16]
            )[0]
            if verifyCoTiledFixed13Bits {
                let reference = u16Output(input, outputShape: outputShape)
                let matches = arrayEqual(
                    candidate.view(dtype: .uint16),
                    reference.view(dtype: .uint16)
                )
                eval(matches)
                precondition(
                    matches.item(Bool.self),
                    "co-tiled fixed13 down projection differs from "
                        + "U16 indexed qmv_fast"
                )
                if !gemma4DownCoTiledFixed13Enabled {
                    // Verifier-only mode must not change the selected output.
                    return reference
                }
            }
            if gemma4DownCoTiledFixed13Enabled {
                return candidate
            }
        }

        guard let packed12 else {
            return u16Output(input, outputShape: outputShape)
        }
        let output = verifiedPacked12 ?? packed12Output(
            input,
            packed12: packed12,
            outputShape: outputShape
        )
        if verifiedPacked12 == nil {
            verifyPacked12Output(output, input: input, outputShape: outputShape)
        }
        return output
    }

    private func packed12Output(
        _ input: MLXArray,
        packed12: Packed12DownMetadata,
        outputShape: [Int]
    ) -> MLXArray {
        gemma4Packed12IndexedDownQMV(
            [projection.weight, packed12.words, metadata.lut, input],
            grid: (32, 1_344, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [outputShape],
            outputDTypes: [.bfloat16]
        )[0]
    }

    private func u16Output(
        _ input: MLXArray,
        outputShape: [Int]
    ) -> MLXArray {
        gemma4IndexedDownQMV(
            [projection.weight, metadata.indices, metadata.lut, input],
            grid: (32, 1_344, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [outputShape],
            outputDTypes: [.bfloat16]
        )[0]
    }

    private func verifyPacked12Output(
        _ output: MLXArray,
        input: MLXArray,
        outputShape: [Int]
    ) {
        guard verifyPacked12Bits else { return }
        let reference = u16Output(input, outputShape: outputShape)
        let matches = arrayEqual(
            output.view(dtype: .uint16),
            reference.view(dtype: .uint16)
        )
        eval(matches)
        precondition(
            matches.item(Bool.self),
            "packed 12-bit down projection differs from U16 indexed qmv_fast"
        )
    }

    var supportsExactTwoVector: Bool {
        !verifyPacked12Bits
            && !verifyCoTiledFixed12Bits
            && !verifyCoTiledFixed13Bits
            && gemma4ExactTwoVectorDownMode(
                lutCount: metadata.lut.size,
                hasFixed12CoTile: coTiledFixed12 != nil,
                fixed12CoTileEnabled: gemma4DownCoTiledFixed12Enabled
            ) != nil
    }

    func exactTwoVector(_ input: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16 && input.shape == [2, 21_504])
        guard supportsExactTwoVector else {
            preconditionFailure("exact two-vector down payload is unavailable")
        }
        let mode = gemma4ExactTwoVectorDownMode(
            lutCount: metadata.lut.size,
            hasFixed12CoTile: coTiledFixed12 != nil,
            fixed12CoTileEnabled: gemma4DownCoTiledFixed12Enabled
        )
        if mode == .fixed12, let coTiledFixed12 {
            return gemma4ExactTwoVectorIndexedDown(
                payload: coTiledFixed12.words,
                lut: metadata.lut,
                input: input
            )
        }
        return gemma4ExactTwoVectorU16IndexedDown(
            weight: projection.weight,
            indices: metadata.indices,
            lut: metadata.lut,
            input: input
        )
    }
}

enum Gemma4ExactTwoVectorDownMode: Equatable {
    case fixed12
    case u16
}

func gemma4ExactTwoVectorDownMode(
    lutCount: Int,
    hasFixed12CoTile: Bool,
    fixed12CoTileEnabled: Bool
) -> Gemma4ExactTwoVectorDownMode? {
    if (1...4_096).contains(lutCount) {
        return hasFixed12CoTile && fixed12CoTileEnabled ? .fixed12 : nil
    }
    return (4_097...65_536).contains(lutCount) ? .u16 : nil
}
