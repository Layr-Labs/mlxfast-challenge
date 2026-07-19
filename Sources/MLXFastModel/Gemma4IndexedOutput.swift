import Foundation
import MLX

private let gemma4OutputRows = 5_376
private let gemma4OutputRowsPerThreadgroup = 8
private let gemma4OutputWeightWordsPerTile = 512

/// Return the row-major source word owned by a threadgroup-major weight tile.
/// Each tile contains eight rows of one 512-input block, or 64 U32 per row.
func gemma4OutputCoTileWeightSourceWordIndex(
    rows: Int,
    blocks: Int,
    threadgroup: Int,
    block: Int,
    tileWord: Int
) -> Int? {
    guard rows > 0,
          rows.isMultiple(of: gemma4OutputRowsPerThreadgroup),
          blocks > 0,
          threadgroup >= 0,
          threadgroup < rows / gemma4OutputRowsPerThreadgroup,
          block >= 0,
          block < blocks,
          tileWord >= 0,
          tileWord < gemma4OutputWeightWordsPerTile
    else {
        return nil
    }
    let localRow = tileWord / 64
    let wordInBlock = tileWord % 64
    let sourceRow = threadgroup * gemma4OutputRowsPerThreadgroup + localRow
    return (sourceRow * blocks + block) * 64 + wordInBlock
}

/// Pack output metadata directly in `(threadgroup, K-block, local row)` order.
/// One row/block owns eight indexes. Fixed12 uses eight low bytes plus four
/// high-nibble bytes; fixed13 adds one byte containing the eight top bits.
/// Thus each eight-row tile occupies exactly 24 or 26 U32, without padding.
func gemma4PackOutputCoTileMetadata(
    _ indices: [UInt16],
    rows: Int,
    groupsPerRow: Int,
    indexBits: Int
) -> [UInt32]? {
    guard rows > 0,
          rows.isMultiple(of: gemma4OutputRowsPerThreadgroup),
          groupsPerRow > 0,
          groupsPerRow.isMultiple(of: 8),
          indexBits == 12 || indexBits == 13
    else {
        return nil
    }
    let (elementCount, elementOverflow) = rows.multipliedReportingOverflow(
        by: groupsPerRow)
    guard !elementOverflow, indices.count == elementCount else { return nil }

    let blocks = groupsPerRow / 8
    let threadgroups = rows / gemma4OutputRowsPerThreadgroup
    let metadataBytesPerRow = indexBits
    let (metadataBytesPerTile, tileBytesOverflow) =
        gemma4OutputRowsPerThreadgroup.multipliedReportingOverflow(
            by: metadataBytesPerRow)
    guard !tileBytesOverflow, metadataBytesPerTile.isMultiple(of: 4) else {
        return nil
    }
    let (tileCount, tileCountOverflow) = threadgroups.multipliedReportingOverflow(
        by: blocks)
    guard !tileCountOverflow else { return nil }
    let (byteCount, byteCountOverflow) = tileCount.multipliedReportingOverflow(
        by: metadataBytesPerTile)
    guard !byteCountOverflow else { return nil }

    var words = [UInt32](repeating: 0, count: byteCount / 4)
    for threadgroup in 0..<threadgroups {
        for block in 0..<blocks {
            let tile = threadgroup * blocks + block
            let tileByteBase = tile * metadataBytesPerTile
            for localRow in 0..<gemma4OutputRowsPerThreadgroup {
                let sourceRow = threadgroup * gemma4OutputRowsPerThreadgroup
                    + localRow
                let sourceBase = sourceRow * groupsPerRow + block * 8
                let rowByteBase = tileByteBase + localRow * metadataBytesPerRow
                for laneGroup in 0..<8 {
                    let value = UInt32(indices[sourceBase + laneGroup])
                    guard value < UInt32(1 << indexBits) else { return nil }

                    let lowByte = rowByteBase + laneGroup
                    words[lowByte / 4] |= (value & 0xff)
                        << UInt32((lowByte % 4) * 8)

                    let middleByte = rowByteBase + 8 + laneGroup / 2
                    let middleShift = (middleByte % 4) * 8
                        + (laneGroup % 2) * 4
                    words[middleByte / 4] |= ((value >> 8) & 0x0f)
                        << UInt32(middleShift)

                    if indexBits == 13 {
                        let topByte = rowByteBase + 12
                        let topShift = (topByte % 4) * 8 + laneGroup
                        words[topByte / 4] |= ((value >> 12) & 1)
                            << UInt32(topShift)
                    }
                }
            }
        }
    }
    return words
}

private func gemma4MakeCoTiledOutputPayload(
    projection: FastQuantizedProjection,
    metadata: IndexedAffineMetadata,
    inputWidth: Int,
    indexBits: Int,
    materialize: Bool = true
) -> MLXArray? {
    precondition(inputWidth == 8_192 || inputWidth == 16_384)
    precondition(indexBits == 12 || indexBits == 13)
    let blocks = inputWidth / 512
    let groupsPerRow = inputWidth / 64
    precondition(projection.weight.dtype == .uint32)
    precondition(projection.weight.shape == [gemma4OutputRows, inputWidth / 8])
    precondition(metadata.indices.dtype == .uint16)
    precondition(metadata.indices.shape == [gemma4OutputRows, groupsPerRow])

    let indices = metadata.indices.asArray(UInt16.self)
    guard indices.allSatisfy({ Int($0) < metadata.lut.size }),
          let packedMetadata = gemma4PackOutputCoTileMetadata(
              indices,
              rows: gemma4OutputRows,
              groupsPerRow: groupsPerRow,
              indexBits: indexBits
          )
    else {
        return nil
    }

    // row = 8 * threadgroup + 4 * SIMD-group + row-within-SIMD
    // word = 64 * K-block + 2 * lane + word-within-lane
    let threadgroups = gemma4OutputRows / gemma4OutputRowsPerThreadgroup
    let weightPayload = projection.weight
        .reshaped(threadgroups, 2, 4, blocks, 32, 2)
        .transposed(0, 3, 1, 2, 4, 5)
        .contiguous()
        .reshaped(threadgroups, blocks, gemma4OutputWeightWordsPerTile)
    let metadataWordsPerTile = 2 * indexBits
    let metadataPayload = MLXArray(
        packedMetadata,
        [threadgroups, blocks, metadataWordsPerTile]
    )
    let payload = concatenated([weightPayload, metadataPayload], axis: 2)
    precondition(payload.dtype == .uint32)
    precondition(payload.shape == [
        threadgroups,
        blocks,
        gemma4OutputWeightWordsPerTile + metadataWordsPerTile,
    ])
    if materialize {
        // The input-independent sidecar is built during untimed model init.
        eval(payload)
    }
    return payload
}

private let gemma4CoTiledFixed12IndexedSlidingOutputQMV = MLXFast.metalKernel(
    name: "gemma4_cotiled_fixed12_indexed_output_qmv_fast_8192_v1",
    inputNames: ["cotiled_payload", "lut", "x"],
    outputNames: ["output"],
    source: gemma4CoTiledIndexedOutputBody(inputWidth: 8_192, indexBits: 12),
    header: gemma4IndexedOutputHeader,
    ensureRowContiguous: true
)

private let gemma4CoTiledFixed13IndexedSlidingOutputQMV = MLXFast.metalKernel(
    name: "gemma4_cotiled_fixed13_indexed_output_qmv_fast_8192_v1",
    inputNames: ["cotiled_payload", "lut", "x"],
    outputNames: ["output"],
    source: gemma4CoTiledIndexedOutputBody(inputWidth: 8_192, indexBits: 13),
    header: gemma4IndexedOutputHeader,
    ensureRowContiguous: true
)

private let gemma4CoTiledFixed12IndexedFullOutputQMV = MLXFast.metalKernel(
    name: "gemma4_cotiled_fixed12_indexed_output_qmv_fast_16384_v1",
    inputNames: ["cotiled_payload", "lut", "x"],
    outputNames: ["output"],
    source: gemma4CoTiledIndexedOutputBody(inputWidth: 16_384, indexBits: 12),
    header: gemma4IndexedOutputHeader,
    ensureRowContiguous: true
)

private let gemma4CoTiledFixed13IndexedFullOutputQMV = MLXFast.metalKernel(
    name: "gemma4_cotiled_fixed13_indexed_output_qmv_fast_16384_v1",
    inputNames: ["cotiled_payload", "lut", "x"],
    outputNames: ["output"],
    source: gemma4CoTiledIndexedOutputBody(inputWidth: 16_384, indexBits: 13),
    header: gemma4IndexedOutputHeader,
    ensureRowContiguous: true
)

// Packed-output additions authored by GPT 5.6 Sol through Gaj's OpenCode Harness.
private let gemma4Packed12IndexedSlidingOutputQMV = MLXFast.metalKernel(
    name: "gemma4_packed12_indexed_output_qmv_fast_8192_v1",
    inputNames: ["weight", "packed_indices", "lut", "x"],
    outputNames: ["output"],
    source: gemma4Packed12IndexedOutputBody(
        inputWidth: 8_192,
        groupsPerRow: 128,
        packedWordsPerRow: 48,
        weightBytesPerRow: 4_096
    ),
    header: gemma4IndexedOutputHeader,
    ensureRowContiguous: true
)

private let gemma4Packed12IndexedFullOutputQMV = MLXFast.metalKernel(
    name: "gemma4_packed12_indexed_output_qmv_fast_16384_v1",
    inputNames: ["weight", "packed_indices", "lut", "x"],
    outputNames: ["output"],
    source: gemma4Packed12IndexedOutputBody(
        inputWidth: 16_384,
        groupsPerRow: 256,
        packedWordsPerRow: 96,
        weightBytesPerRow: 8_192
    ),
    header: gemma4IndexedOutputHeader,
    ensureRowContiguous: true
)

private let gemma4IndexedSlidingOutputQMV = MLXFast.metalKernel(
    name: "gemma4_indexed_output_qmv_fast_8192_v1",
    inputNames: ["weight", "indices", "lut", "x"],
    outputNames: ["output"],
    source: gemma4IndexedOutputBody(
        inputWidth: 8_192,
        groupsPerRow: 128,
        weightBytesPerRow: 4_096
    ),
    header: gemma4IndexedOutputHeader,
    ensureRowContiguous: true
)

private let gemma4IndexedFullOutputQMV = MLXFast.metalKernel(
    name: "gemma4_indexed_output_qmv_fast_16384_v1",
    inputNames: ["weight", "indices", "lut", "x"],
    outputNames: ["output"],
    source: gemma4IndexedOutputBody(
        inputWidth: 16_384,
        groupsPerRow: 256,
        weightBytesPerRow: 8_192
    ),
    header: gemma4IndexedOutputHeader,
    ensureRowContiguous: true
)

private func gemma4CoTiledIndexedOutputBody(
    inputWidth: Int,
    indexBits: Int
) -> String {
    precondition(inputWidth == 8_192 || inputWidth == 16_384)
    precondition(indexBits == 12 || indexBits == 13)
    let blocks = inputWidth / 512
    let metadataWordsPerTile = 2 * indexBits
    let metadataExtraction: String
    if indexBits == 12 {
        metadataExtraction = """
                const uint low = row_metadata[lane_group];
                const uint packed_high =
                    row_metadata[8 + lane_group / 2];
                const uint high =
                    (packed_high >> ((lane_group & 1) * 4)) & 0x0f;
                const uint metadata_index = low | (high << 8);
            """
    } else {
        metadataExtraction = """
                const uint low = row_metadata[lane_group];
                const uint packed_middle =
                    row_metadata[8 + lane_group / 2];
                const uint middle =
                    (packed_middle >> ((lane_group & 1) * 4)) & 0x0f;
                const uint top = (row_metadata[12] >> lane_group) & 1;
                const uint metadata_index =
                    low | (middle << 8) | (top << 12);
            """
    }

    return """
        constexpr int kInputWidth = \(inputWidth);
        constexpr int kRowsPerSIMD = 4;
        constexpr int kBlockSize = 512;
        constexpr int kBlocks = \(blocks);
        constexpr int kWordsPerLane = 2;
        constexpr int kWordsPerRow = 32 * kWordsPerLane;
        constexpr int kWordsPerSIMD = kRowsPerSIMD * kWordsPerRow;
        constexpr int kWeightWordsPerTile = 2 * kWordsPerSIMD;
        constexpr int kMetadataBytesPerRow = \(indexBits);
        constexpr int kMetadataWordsPerTile = \(metadataWordsPerTile);
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
            const float input_sum = gemma4_output_load_values(input, values);
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
        \(metadataExtraction)
                const uint pair = lut[metadata_index];
                result[row] += gemma4_output_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_output_pair_scale(pair),
                    gemma4_output_pair_bias(pair),
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
        """
}

private func gemma4Packed12IndexedOutputBody(
    inputWidth: Int,
    groupsPerRow: Int,
    packedWordsPerRow: Int,
    weightBytesPerRow: Int
) -> String {
    """
        constexpr int kInputWidth = \(inputWidth);
        constexpr int kGroupsPerRow = \(groupsPerRow);
        constexpr int kPackedWordsPerRow = \(packedWordsPerRow);
        constexpr int kWeightBytesPerRow = \(weightBytesPerRow);
        constexpr int kRowsPerSIMD = 4;
        constexpr int kBlockSize = 512;

        const int output_row =
            threadgroup_position_in_grid.y * 8
            + simdgroup_index_in_threadgroup * kRowsPerSIMD;
        const device uchar* weight_bytes =
            reinterpret_cast<const device uchar*>(weight)
            + output_row * kWeightBytesPerRow
            + thread_index_in_simdgroup * 8;

        // Each 512-element block consumes eight group-64 indexes. The block's
        // 96 metadata bits are exactly three U32 words, so no block crosses a
        // row boundary for either supported output geometry.
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
            const float input_sum = gemma4_output_load_values(input, values);
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
                result[row] += gemma4_output_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_output_pair_scale(pair),
                    gemma4_output_pair_bias(pair),
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
        """
}

private func gemma4IndexedOutputBody(
    inputWidth: Int,
    groupsPerRow: Int,
    weightBytesPerRow: Int
) -> String {
    """
        constexpr int kInputWidth = \(inputWidth);
        constexpr int kGroupsPerRow = \(groupsPerRow);
        constexpr int kWeightBytesPerRow = \(weightBytesPerRow);
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
            const float input_sum = gemma4_output_load_values(input, values);
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const ushort metadata_index = row_indices[row * kGroupsPerRow];
                const uint pair = lut[metadata_index];
                result[row] += gemma4_output_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_output_pair_scale(pair),
                    gemma4_output_pair_bias(pair),
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
        """
}

private let gemma4IndexedOutputHeader = """
    using namespace metal;

    inline float gemma4_output_pair_scale(uint pair) {
        return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
    }

    inline float gemma4_output_pair_bias(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair >> 16)));
    }

    inline float gemma4_output_load_values(
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

    inline float gemma4_output_qdot_4bit(
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
    """

private func gemma4OutputEnvironmentFlag(
    _ name: String,
    default defaultValue: Bool
) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return defaultValue
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

private let gemma4OutputCoTiledEnabled = gemma4OutputEnvironmentFlag(
    "DARKBLOOM_OUTPUT_COTILED",
    default: true
)

// Keep the two decode geometries independently fail-closed while preserving
// the promoted master switch as their production default.
private let gemma4OutputCoTiledSlidingEnabled = gemma4OutputEnvironmentFlag(
    "DARKBLOOM_OUTPUT_COTILED_SLIDING",
    default: gemma4OutputCoTiledEnabled
)
private let gemma4OutputCoTiledFullEnabled = gemma4OutputEnvironmentFlag(
    "DARKBLOOM_OUTPUT_COTILED_FULL",
    default: gemma4OutputCoTiledEnabled
)

private let gemma4VerifyOutputCoTiledBits = gemma4OutputEnvironmentFlag(
    "DARKBLOOM_VERIFY_OUTPUT_COTILED_BITS",
    default: false
)

private let gemma4VerifyOutputStockBits = gemma4OutputEnvironmentFlag(
    "DARKBLOOM_VERIFY_OUTPUT_STOCK_BITS",
    default: false
)

private struct Packed12OutputMetadata: @unchecked Sendable {
    let words: MLXArray

    init?(metadata: IndexedAffineMetadata, groupsPerRow: Int) {
        guard groupsPerRow == 128 || groupsPerRow == 256,
              metadata.indices.dtype == .uint16,
              metadata.indices.shape == [5_376, groupsPerRow],
              metadata.lut.dtype == .uint32,
              metadata.lut.ndim == 1,
              (1...4_096).contains(metadata.lut.size)
        else {
            return nil
        }

        let (rowBits, rowBitsOverflow) = groupsPerRow.multipliedReportingOverflow(by: 12)
        guard !rowBitsOverflow, rowBits.isMultiple(of: 32) else { return nil }
        let wordsPerRow = rowBits / 32
        let (wordCount, wordCountOverflow) = 5_376.multipliedReportingOverflow(
            by: wordsPerRow)
        guard !wordCountOverflow else { return nil }

        let indices = metadata.indices.asArray(UInt16.self)
        guard indices.allSatisfy({ Int($0) < metadata.lut.size }),
              let packed = gemma4Pack12BitIndices(
                  indices,
                  rows: 5_376,
                  groupsPerRow: groupsPerRow
              ),
              packed.count == wordCount
        else {
            return nil
        }

        let words = MLXArray(packed, [5_376, wordsPerRow])
        // Packing and host-to-Metal materialization happen while the model is
        // initialized, before any scored prefill or decode phase begins.
        eval(words)
        self.words = words
    }
}

private struct CoTiledOutputPayload: @unchecked Sendable {
    let words: MLXArray
    let indexBits: Int

    init?(
        projection: FastQuantizedProjection,
        metadata: IndexedAffineMetadata,
        inputWidth: Int
    ) {
        guard metadata.lut.dtype == .uint32,
              metadata.lut.ndim == 1,
              (1...8_192).contains(metadata.lut.size)
        else {
            return nil
        }
        let indexBits = metadata.lut.size <= 4_096 ? 12 : 13
        guard let words = gemma4MakeCoTiledOutputPayload(
            projection: projection,
            metadata: metadata,
            inputWidth: inputWidth,
            indexBits: indexBits
        ) else {
            return nil
        }
        self.words = words
        self.indexBits = indexBits
    }
}

private func gemma4VerifyRawOutputBF16(
    _ candidate: MLXArray,
    reference: MLXArray,
    candidateName: String,
    referenceName: String
) {
    precondition(candidate.dtype == .bfloat16)
    precondition(reference.dtype == .bfloat16)
    precondition(candidate.shape == reference.shape)
    let candidateBits = candidate.view(dtype: .uint16)
    let referenceBits = reference.view(dtype: .uint16)
    let matches = arrayEqual(candidateBits, referenceBits)
    eval(matches)
    guard matches.item(Bool.self) else {
        let candidateValues = candidateBits.asArray(UInt16.self)
        let referenceValues = referenceBits.asArray(UInt16.self)
        let mismatch = zip(candidateValues, referenceValues)
            .enumerated()
            .first { $0.element.0 != $0.element.1 }
        preconditionFailure(
            "attention output raw BF16 mismatch \(candidateName) vs "
                + "\(referenceName) at index \(mismatch?.offset ?? -1): "
                + "candidate=\(mismatch?.element.0 ?? 0), "
                + "reference=\(mismatch?.element.1 ?? 0)"
        )
    }
}

struct IndexedOutputProjection: @unchecked Sendable {
    let projection: FastQuantizedProjection
    let metadata: IndexedAffineMetadata
    let inputWidth: Int
    private let packed12: Packed12OutputMetadata?
    private let coTiled: CoTiledOutputPayload?
    private let verifyPacked12Bits: Bool
    private let verifyCoTiledBits: Bool
    private let verifyStockBits: Bool
    private let coTiledEnabled: Bool

    init?(projection: FastQuantizedProjection) {
        guard let biases = projection.biases else { return nil }
        let inputWidth: Int
        if projection.weight.shape == [5_376, 1_024]
            && projection.scales.shape == [5_376, 128]
        {
            inputWidth = 8_192
        } else if projection.weight.shape == [5_376, 2_048]
            && projection.scales.shape == [5_376, 256]
        {
            inputWidth = 16_384
        } else {
            return nil
        }
        guard projection.groupSize == 64,
              projection.bits == 4,
              projection.weight.dtype == .uint32,
              projection.scales.dtype == .bfloat16,
              biases.dtype == .bfloat16,
              biases.shape == projection.scales.shape
        else {
            return nil
        }
        let metadata = makeIndexedAffineMetadata(
            scales: projection.scales,
            biases: biases
        )
        self.projection = projection
        self.metadata = metadata
        self.inputWidth = inputWidth
        let packed12Enabled = gemma4OutputEnvironmentFlag(
            "MLXFAST_PACKED_OUTPUT_INDICES",
            default: true
        )
        let verifyPacked12Bits = gemma4OutputEnvironmentFlag(
            "MLXFAST_VERIFY_PACKED_OUTPUT_BITS",
            default: false
        )
        self.verifyPacked12Bits = verifyPacked12Bits
        self.verifyCoTiledBits = gemma4VerifyOutputCoTiledBits
        self.verifyStockBits = gemma4VerifyOutputStockBits
        let coTiledEnabled = inputWidth == 8_192
            ? gemma4OutputCoTiledSlidingEnabled
            : gemma4OutputCoTiledFullEnabled
        self.coTiledEnabled = coTiledEnabled

        let wantsCoTiled = coTiledEnabled
            || gemma4VerifyOutputCoTiledBits
            || gemma4VerifyOutputStockBits
        let coTiled = wantsCoTiled
            ? CoTiledOutputPayload(
                projection: projection,
                metadata: metadata,
                inputWidth: inputWidth
            )
            : nil
        self.coTiled = coTiled

        // Keep the promoted packed12 sidecar only when rollback or its own
        // verifier can use it. The original projection and U16 metadata above
        // always remain resident for prefill, raw verification, and fallback.
        let needsPacked12 = packed12Enabled
            && (coTiled == nil
                || !coTiledEnabled
                || verifyPacked12Bits)
        self.packed12 = needsPacked12
            ? Packed12OutputMetadata(
                metadata: metadata,
                groupsPerRow: inputWidth / 64
            )
            : nil
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, inputWidth])
        let outputShape = [1, 1, gemma4OutputRows]
        var rawReference: MLXArray?

        if let coTiled {
            let candidate = coTiledOutput(
                input,
                payload: coTiled,
                outputShape: outputShape
            )
            if verifyCoTiledBits {
                let reference = rawOutput(input, outputShape: outputShape)
                gemma4VerifyRawOutputBF16(
                    candidate,
                    reference: reference,
                    candidateName: "co-tiled fixed\(coTiled.indexBits)",
                    referenceName: "U16 indexed qmv_fast"
                )
                rawReference = reference
            }
            if verifyStockBits {
                let stock = projection(input)
                gemma4VerifyRawOutputBF16(
                    candidate,
                    reference: stock,
                    candidateName: "co-tiled fixed\(coTiled.indexBits)",
                    referenceName: "stock quantizedMM"
                )
                // Qualification mode deliberately consumes the trusted oracle.
                return stock
            }

            if coTiledEnabled {
                if verifyPacked12Bits, let packed12 {
                    let packedOutput = packed12Output(
                        input,
                        packed12: packed12,
                        outputShape: outputShape
                    )
                    verifyPacked12Output(
                        packedOutput,
                        input: input,
                        outputShape: outputShape,
                        rawReference: rawReference
                    )
                }
                return candidate
            }
        }

        if verifyStockBits {
            // A missing co-tiled payload cannot be qualified; still preserve the
            // verifier contract that only stock output reaches the model.
            return projection(input)
        }
        guard let packed12 else {
            return rawReference ?? rawOutput(input, outputShape: outputShape)
        }
        let output = packed12Output(
            input,
            packed12: packed12,
            outputShape: outputShape
        )
        verifyPacked12Output(
            output,
            input: input,
            outputShape: outputShape,
            rawReference: rawReference
        )
        return output
    }

    private func coTiledOutput(
        _ input: MLXArray,
        payload: CoTiledOutputPayload,
        outputShape: [Int]
    ) -> MLXArray {
        let inputs = [payload.words, metadata.lut, input]
        let outputs: [MLXArray]
        switch (inputWidth, payload.indexBits) {
        case (8_192, 12):
            outputs = gemma4CoTiledFixed12IndexedSlidingOutputQMV(
                inputs,
                grid: (32, 1_344, 1),
                threadGroup: (32, 2, 1),
                outputShapes: [outputShape],
                outputDTypes: [.bfloat16]
            )
        case (8_192, 13):
            outputs = gemma4CoTiledFixed13IndexedSlidingOutputQMV(
                inputs,
                grid: (32, 1_344, 1),
                threadGroup: (32, 2, 1),
                outputShapes: [outputShape],
                outputDTypes: [.bfloat16]
            )
        case (16_384, 12):
            outputs = gemma4CoTiledFixed12IndexedFullOutputQMV(
                inputs,
                grid: (32, 1_344, 1),
                threadGroup: (32, 2, 1),
                outputShapes: [outputShape],
                outputDTypes: [.bfloat16]
            )
        case (16_384, 13):
            outputs = gemma4CoTiledFixed13IndexedFullOutputQMV(
                inputs,
                grid: (32, 1_344, 1),
                threadGroup: (32, 2, 1),
                outputShapes: [outputShape],
                outputDTypes: [.bfloat16]
            )
        default:
            preconditionFailure("unsupported co-tiled output projection layout")
        }
        return outputs[0]
    }

    private func packed12Output(
        _ input: MLXArray,
        packed12: Packed12OutputMetadata,
        outputShape: [Int]
    ) -> MLXArray {
        let kernel = inputWidth == 8_192
            ? gemma4Packed12IndexedSlidingOutputQMV
            : gemma4Packed12IndexedFullOutputQMV
        return kernel(
            [projection.weight, packed12.words, metadata.lut, input],
            grid: (32, 1_344, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [outputShape],
            outputDTypes: [.bfloat16]
        )[0]
    }

    private func rawOutput(
        _ input: MLXArray,
        outputShape: [Int]
    ) -> MLXArray {
        let kernel = inputWidth == 8_192
            ? gemma4IndexedSlidingOutputQMV
            : gemma4IndexedFullOutputQMV
        return kernel(
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
        outputShape: [Int],
        rawReference: MLXArray?
    ) {
        guard verifyPacked12Bits else { return }
        let reference = rawReference ?? rawOutput(input, outputShape: outputShape)
        gemma4VerifyRawOutputBF16(
            output,
            reference: reference,
            candidateName: "packed12",
            referenceName: "U16 indexed qmv_fast"
        )
    }

    var supportsExactTwoVector: Bool {
        coTiledEnabled
            && coTiled != nil
            && !verifyPacked12Bits
            && !verifyCoTiledBits
            && !verifyStockBits
    }

    var supportsExactFourVector: Bool {
        supportsExactTwoVector
    }

    func exactTwoVector(_ input: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [2, inputWidth])
        guard supportsExactTwoVector, let coTiled else {
            preconditionFailure("exact two-vector output payload is unavailable")
        }
        return gemma4ExactTwoVectorIndexedOutput(
            payload: coTiled.words,
            lut: metadata.lut,
            input: input,
            inputWidth: inputWidth,
            indexBits: coTiled.indexBits
        )
    }

    func exactFourVector(_ input: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [4, inputWidth])
        guard supportsExactFourVector, let coTiled else {
            preconditionFailure("exact four-vector output payload is unavailable")
        }
        return gemma4ExactFourVectorIndexedOutput(
            payload: coTiled.words,
            lut: metadata.lut,
            input: input,
            inputWidth: inputWidth,
            indexBits: coTiled.indexBits
        )
    }
}
