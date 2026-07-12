import Foundation
import MLX

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
    private let verifyPacked12Bits: Bool

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
        if environmentFlag("MLXFAST_PACKED_DOWN_INDICES", default: true) {
            self.packed12 = Packed12DownMetadata(metadata: metadata)
        } else {
            self.packed12 = nil
        }
        self.verifyPacked12Bits = environmentFlag(
            "MLXFAST_VERIFY_PACKED_DOWN_BITS",
            default: false
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 21_504])
        let outputShape = [1, 1, 5_376]
        guard let packed12 else {
            return gemma4IndexedDownQMV(
                [projection.weight, metadata.indices, metadata.lut, input],
                grid: (32, 1_344, 1),
                threadGroup: (32, 2, 1),
                outputShapes: [outputShape],
                outputDTypes: [.bfloat16]
            )[0]
        }

        let output = gemma4Packed12IndexedDownQMV(
            [projection.weight, packed12.words, metadata.lut, input],
            grid: (32, 1_344, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [outputShape],
            outputDTypes: [.bfloat16]
        )[0]
        if verifyPacked12Bits {
            let reference = gemma4IndexedDownQMV(
                [projection.weight, metadata.indices, metadata.lut, input],
                grid: (32, 1_344, 1),
                threadGroup: (32, 2, 1),
                outputShapes: [outputShape],
                outputDTypes: [.bfloat16]
            )[0]
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
        return output
    }
}
