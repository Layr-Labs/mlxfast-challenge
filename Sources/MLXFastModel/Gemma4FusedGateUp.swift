import Foundation
import MLX
import MLXNN

struct IndexedAffineMetadata: @unchecked Sendable {
    let indices: MLXArray
    let lut: MLXArray
}

/// Losslessly row-pack U16 gate/up metadata indexes into fixed two-block
/// tiles. Every pair of four-group decode blocks occupies twelve bytes:
/// four even low bytes, four shared even-high/odd-low bytes, and four odd
/// high bytes. An unpaired final block uses eight lane-major bytes (the two
/// otherwise-padding bytes buy a branch-free high nibble per lane). Rows are
/// padded to four-byte alignment, so Gemma's 84 indexes occupy 128 bytes.
///
/// This is still a dense 12-bit representation, but its decode schedule has
/// no variable word crossing: one lane loads three coalesced bytes for two
/// blocks and extracts both indexes with fixed nibble operations.
func gemma4Pack12BitGateUpIndices(
    _ indices: [UInt16],
    rows: Int,
    groupsPerRow: Int
) -> [UInt8]? {
    guard rows > 0,
          groupsPerRow > 0,
          groupsPerRow.isMultiple(of: 4)
    else {
        return nil
    }
    let (elementCount, elementOverflow) = rows.multipliedReportingOverflow(
        by: groupsPerRow)
    guard !elementOverflow, indices.count == elementCount else { return nil }

    let blockCount = groupsPerRow / 4
    let blockPairCount = blockCount / 2
    let tailBlockCount = blockCount % 2
    let (pairBytes, pairBytesOverflow) = blockPairCount.multipliedReportingOverflow(by: 12)
    guard !pairBytesOverflow else { return nil }
    let (payloadBytes, payloadOverflow) = pairBytes.addingReportingOverflow(
        tailBlockCount * 8)
    guard !payloadOverflow else { return nil }
    let (roundedBytes, roundedOverflow) = payloadBytes.addingReportingOverflow(3)
    guard !roundedOverflow else { return nil }
    let bytesPerRow = (roundedBytes / 4) * 4
    let (byteCount, byteOverflow) = rows.multipliedReportingOverflow(by: bytesPerRow)
    guard !byteOverflow else { return nil }

    var bytes = [UInt8](repeating: 0, count: byteCount)
    for row in 0..<rows {
        let inputBase = row * groupsPerRow
        let outputBase = row * bytesPerRow

        for blockPair in 0..<blockPairCount {
            let inputPairBase = inputBase + blockPair * 8
            let outputPairBase = outputBase + blockPair * 12
            for laneGroup in 0..<4 {
                let even = indices[inputPairBase + laneGroup]
                let odd = indices[inputPairBase + 4 + laneGroup]
                guard even < 4_096, odd < 4_096 else { return nil }
                bytes[outputPairBase + laneGroup] = UInt8(truncatingIfNeeded: even)
                bytes[outputPairBase + 4 + laneGroup] =
                    UInt8((even >> 8) | ((odd & 0x000f) << 4))
                bytes[outputPairBase + 8 + laneGroup] = UInt8(odd >> 4)
            }
        }

        if tailBlockCount == 1 {
            let inputTailBase = inputBase + blockPairCount * 8
            let outputTailBase = outputBase + blockPairCount * 12
            for laneGroup in 0..<4 {
                let value = indices[inputTailBase + laneGroup]
                guard value < 4_096 else { return nil }
                let laneOffset = outputTailBase + laneGroup * 2
                bytes[laneOffset] = UInt8(truncatingIfNeeded: value)
                bytes[laneOffset + 1] = UInt8(value >> 8)
            }
        }
    }
    return bytes
}

private func gemma4GateUpEnvironmentFlag(
    _ name: String,
    default defaultValue: Bool
) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return defaultValue
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

private struct Packed12GateUpMetadata: @unchecked Sendable {
    let bytes: MLXArray

    init?(metadata: IndexedAffineMetadata) {
        guard metadata.indices.dtype == .uint16,
              metadata.indices.shape == [21_504, 84],
              metadata.lut.dtype == .uint32,
              metadata.lut.ndim == 1,
              (1...4_096).contains(metadata.lut.size),
              let packed = gemma4Pack12BitGateUpIndices(
                  metadata.indices.asArray(UInt16.self),
                  rows: 21_504,
                  groupsPerRow: 84
              )
        else {
            return nil
        }
        let bytes = MLXArray(packed, [21_504, 128])
        // Packing happens during untimed model initialization. Materialize the
        // input-independent buffer now so first decode cannot inherit a lazy
        // host-to-Metal upload of roughly 2.63 MiB per eligible projection.
        eval(bytes)
        self.bytes = bytes
    }
}

private struct Packed12GateUpMetadataPair: @unchecked Sendable {
    let gate: Packed12GateUpMetadata
    let up: Packed12GateUpMetadata
}

func supportsGemma4FusedGateUpInput(_ input: MLXArray) -> Bool {
    input.dtype == .bfloat16
        && input.shape == [1, 1, 5_376]
}

private func supportsFusedGateUpKernelBuffers(
    gate: FastQuantizedProjection,
    up: FastQuantizedProjection
) -> Bool {
    let inputWidth = 5_376
    let groupSize = 64
    let bits = 4

    func supports(_ projection: FastQuantizedProjection) -> Bool {
        guard let biases = projection.biases,
              projection.weight.ndim == 2,
              projection.weight.dim(1) == inputWidth / (32 / bits),
              projection.weight.dim(0).isMultiple(of: 4)
        else {
            return false
        }
        let metadataShape = [projection.weight.dim(0), inputWidth / groupSize]
        return projection.groupSize == groupSize
            && projection.bits == bits
            && projection.weight.dtype == .uint32
            && projection.scales.dtype == .bfloat16
            && projection.scales.shape == metadataShape
            && biases.dtype == .bfloat16
            && biases.shape == metadataShape
    }

    return supports(gate)
        && supports(up)
        && gate.weight.dim(0) == up.weight.dim(0)
}

private func supportsIndexedAffineMetadata(
    _ metadata: IndexedAffineMetadata,
    shape: [Int]
) -> Bool {
    metadata.indices.dtype == .uint16
        && metadata.indices.shape == shape
        && metadata.lut.dtype == .uint32
        && metadata.lut.ndim == 1
        && (1...65_536).contains(metadata.lut.size)
}

func supportsGemma4FusedGateUp(
    gate: FastQuantizedProjection,
    up: FastQuantizedProjection
) -> Bool {
    let inputWidth = 5_376
    let outputWidth = 21_504
    let groupSize = 64
    let bits = 4
    let weightShape = [outputWidth, inputWidth / (32 / bits)]
    let metadataShape = [outputWidth, inputWidth / groupSize]

    return supportsFusedGateUpKernelBuffers(gate: gate, up: up)
        && gate.weight.shape == weightShape
        && up.weight.shape == weightShape
        && gate.scales.shape == metadataShape
        && up.scales.shape == metadataShape
}

func makeIndexedAffineMetadata(
    scales: MLXArray,
    biases: MLXArray
) -> IndexedAffineMetadata {
    precondition(scales.dtype == .bfloat16)
    precondition(biases.dtype == .bfloat16)
    precondition(scales.shape == biases.shape)

    let scaleBits = scales.view(dtype: .uint16).asArray(UInt16.self)
    let biasBits = biases.view(dtype: .uint16).asArray(UInt16.self)
    var pairToIndex: [UInt32: UInt16] = [:]
    pairToIndex.reserveCapacity(8_192)
    var indices: [UInt16] = []
    indices.reserveCapacity(scaleBits.count)
    var lut: [UInt32] = []
    lut.reserveCapacity(8_192)

    for (scale, bias) in zip(scaleBits, biasBits) {
        let pair = UInt32(scale) | (UInt32(bias) << 16)
        if let index = pairToIndex[pair] {
            indices.append(index)
        } else {
            precondition(lut.count < 65_536, "affine metadata LUT exceeds UInt16 capacity")
            let index = UInt16(lut.count)
            pairToIndex[pair] = index
            indices.append(index)
            lut.append(pair)
        }
    }

    return IndexedAffineMetadata(
        indices: MLXArray(indices, scales.shape),
        lut: MLXArray(lut)
    )
}

enum FusedGateUpMetadataMode: String {
    case raw
    case indexed
}

private let gemma4FusedGateUpQMV = MLXFast.metalKernel(
    name: "gemma4_fused_gate_up_qmv_5376",
    inputNames: [
        "gate_weight", "gate_scales", "gate_biases",
        "up_weight", "up_scales", "up_biases", "x",
    ],
    outputNames: ["gate_output", "up_output"],
    source: """
        constexpr int kInputWidth = 5376;
        constexpr int kGroupsPerRow = 84;
        constexpr int kWeightBytesPerRow = 2688;
        constexpr int kRowsPerSIMD = 4;

        const bool is_up = simdgroup_index_in_threadgroup == 1;
        const int output_row = threadgroup_position_in_grid.y * kRowsPerSIMD;
        const device uint* weight = is_up ? up_weight : gate_weight;
        const device bfloat* scales = is_up ? up_scales : gate_scales;
        const device bfloat* biases = is_up ? up_biases : gate_biases;
        device bfloat* output = is_up ? up_output : gate_output;

        const device uchar* weight_bytes =
            reinterpret_cast<const device uchar*>(weight)
            + output_row * kWeightBytesPerRow
            + thread_index_in_simdgroup * 4;
        const device bfloat* row_scales =
            scales + output_row * kGroupsPerRow
            + thread_index_in_simdgroup / 8;
        const device bfloat* row_biases =
            biases + output_row * kGroupsPerRow
            + thread_index_in_simdgroup / 8;
        const device bfloat* input = x + thread_index_in_simdgroup * 8;

        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < 21; ++block) {
            float values[8];
            const float input_sum = gemma4_load_qmv_values(input, values);

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const device bfloat* scale =
                    row_scales + row * kGroupsPerRow;
                const device bfloat* bias =
                    row_biases + row * kGroupsPerRow;
                result[row] += gemma4_qdot_4bit(
                    row_weight, values, scale[0], bias[0], input_sum);
            }

            weight_bytes += 128;
            row_scales += 4;
            row_biases += 4;
            input += 256;
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

        inline float gemma4_load_qmv_values(
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

        inline float gemma4_qdot_4bit(
            const device uchar* weight,
            const thread float* values,
            float scale,
            float bias,
            float input_sum
        ) {
            const device ushort* packed =
                reinterpret_cast<const device ushort*>(weight);
            float accumulator = 0;
            for (int index = 0; index < 2; ++index) {
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

private let gemma4IndexedFusedGateUpQMV = MLXFast.metalKernel(
    name: "gemma4_indexed_fused_gate_up_qmv_5376_v1",
    inputNames: [
        "gate_weight", "gate_indices", "gate_lut",
        "up_weight", "up_indices", "up_lut", "x",
    ],
    outputNames: ["gate_output", "up_output"],
    source: """
        constexpr int kInputWidth = 5376;
        constexpr int kGroupsPerRow = 84;
        constexpr int kWeightBytesPerRow = 2688;
        constexpr int kRowsPerSIMD = 4;

        const bool is_up = simdgroup_index_in_threadgroup == 1;
        const int output_row = threadgroup_position_in_grid.y * kRowsPerSIMD;
        const device uint* weight = is_up ? up_weight : gate_weight;
        const device ushort* indices = is_up ? up_indices : gate_indices;
        device bfloat* output = is_up ? up_output : gate_output;

        const device uchar* weight_bytes =
            reinterpret_cast<const device uchar*>(weight)
            + output_row * kWeightBytesPerRow
            + thread_index_in_simdgroup * 4;
        const device ushort* row_indices =
            indices + output_row * kGroupsPerRow
            + thread_index_in_simdgroup / 8;
        const device bfloat* input = x + thread_index_in_simdgroup * 8;

        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < 21; ++block) {
            float values[8];
            const float input_sum = gemma4_load_qmv_values(input, values);

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const ushort metadata_index = row_indices[row * kGroupsPerRow];
                const uint pair = is_up
                    ? up_lut[metadata_index]
                    : gate_lut[metadata_index];
                result[row] += gemma4_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_pair_scale(pair),
                    gemma4_pair_bias(pair),
                    input_sum);
            }

            weight_bytes += 128;
            row_indices += 4;
            input += 256;
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

        inline float gemma4_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float gemma4_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float gemma4_load_qmv_values(
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

        inline float gemma4_qdot_4bit(
            const device uchar* weight,
            const thread float* values,
            float scale,
            float bias,
            float input_sum
        ) {
            const device ushort* packed =
                reinterpret_cast<const device ushort*>(weight);
            float accumulator = 0;
            for (int index = 0; index < 2; ++index) {
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

private let gemma4Packed12IndexedFusedGateUpActivationQMV = MLXFast.metalKernel(
    name: "gemma4_packed12_indexed_fused_gate_up_activation_qmv_5376_v2",
    inputNames: [
        "gate_weight", "gate_packed_indices", "gate_lut",
        "up_weight", "up_packed_indices", "up_lut", "x",
    ],
    outputNames: ["activated"],
    source: """
        constexpr int kInputWidth = 5376;
        constexpr int kGroupsPerRow = 84;
        constexpr int kPackedBytesPerRow = 128;
        constexpr int kWeightBytesPerRow = 2688;
        constexpr int kRowsPerSIMD = 4;

        const bool is_up = simdgroup_index_in_threadgroup == 1;
        const int output_row = threadgroup_position_in_grid.y * kRowsPerSIMD;
        const device uint* weight = is_up ? up_weight : gate_weight;
        const device uchar* packed_indices = is_up
            ? up_packed_indices
            : gate_packed_indices;
        const device uint* lut = is_up ? up_lut : gate_lut;

        const device uchar* weight_bytes =
            reinterpret_cast<const device uchar*>(weight)
            + output_row * kWeightBytesPerRow
            + thread_index_in_simdgroup * 4;
        const device uchar* row_packed_indices =
            packed_indices + output_row * kPackedBytesPerRow;
        const device bfloat* input = x + thread_index_in_simdgroup * 8;
        const uint lane_group = thread_index_in_simdgroup >> 3;

        float result[kRowsPerSIMD] = {0};
        for (int block_pair = 0; block_pair < 10; ++block_pair) {
            float even_values[8];
            const float even_input_sum =
                gemma4_load_qmv_values(input, even_values);
            uint odd_pairs[kRowsPerSIMD];

            // One 96-bit tile encodes the eight metadata indexes consumed by
            // two consecutive 256-input blocks. For this lane's group, the
            // three byte planes reconstruct both indexes without variable
            // bit offsets, integer division/modulo, or word-crossing control.
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const device uchar* row_tile =
                    row_packed_indices + row * kPackedBytesPerRow;
                const uint even_low = row_tile[lane_group];
                const uint middle = row_tile[4 + lane_group];
                const uint odd_high = row_tile[8 + lane_group];
                const uint even_index =
                    even_low | ((middle & 0x0f) << 8);
                const uint odd_index =
                    (middle >> 4) | (odd_high << 4);
                const uint even_pair = lut[even_index];
                odd_pairs[row] = lut[odd_index];
                result[row] += gemma4_qdot_4bit(
                    row_weight,
                    even_values,
                    gemma4_pair_scale(even_pair),
                    gemma4_pair_bias(even_pair),
                    even_input_sum);
            }

            weight_bytes += 128;
            input += 256;
            float odd_values[8];
            const float odd_input_sum =
                gemma4_load_qmv_values(input, odd_values);
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const uint odd_pair = odd_pairs[row];
                result[row] += gemma4_qdot_4bit(
                    row_weight,
                    odd_values,
                    gemma4_pair_scale(odd_pair),
                    gemma4_pair_bias(odd_pair),
                    odd_input_sum);
            }

            weight_bytes += 128;
            input += 256;
            row_packed_indices += 12;
        }

        // The 21st block uses the row's eight remaining bytes as four
        // lane-major (low, high-nibble) pairs. This spends what was padding in
        // the dense 12-bit row to avoid one more variable nibble extraction.
        float tail_values[8];
        const float tail_input_sum =
            gemma4_load_qmv_values(input, tail_values);
        const uint tail_lane_offset = lane_group << 1;
        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            const device uchar* row_weight =
                weight_bytes + row * kWeightBytesPerRow;
            const device uchar* row_tail =
                row_packed_indices + row * kPackedBytesPerRow;
            const uint low = row_tail[tail_lane_offset];
            const uint high = row_tail[tail_lane_offset + 1];
            const uint metadata_index = low | (high << 8);
            const uint pair = lut[metadata_index];
            result[row] += gemma4_qdot_4bit(
                row_weight,
                tail_values,
                gemma4_pair_scale(pair),
                gemma4_pair_bias(pair),
                tail_input_sum);
        }

        threadgroup bfloat projections[2][kRowsPerSIMD];
        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                projections[is_up ? 1 : 0][row] =
                    static_cast<bfloat>(result[row]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simdgroup_index_in_threadgroup == 0
            && thread_index_in_simdgroup < kRowsPerSIMD
        ) {
            const int row = thread_index_in_simdgroup;
            const bfloat gate = projections[0][row];
            const bfloat up = projections[1][row];

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
        """,
    header: """
        using namespace metal;

        inline float gemma4_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float gemma4_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float gemma4_load_qmv_values(
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

        inline float gemma4_qdot_4bit(
            const device uchar* weight,
            const thread float* values,
            float scale,
            float bias,
            float input_sum
        ) {
            const device ushort* packed =
                reinterpret_cast<const device ushort*>(weight);
            float accumulator = 0;
            for (int index = 0; index < 2; ++index) {
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

private let gemma4IndexedFusedGateUpActivationQMV = MLXFast.metalKernel(
    name: "gemma4_indexed_fused_gate_up_activation_qmv_5376_v1",
    inputNames: [
        "gate_weight", "gate_indices", "gate_lut",
        "up_weight", "up_indices", "up_lut", "x",
    ],
    outputNames: ["activated"],
    source: """
        constexpr int kInputWidth = 5376;
        constexpr int kGroupsPerRow = 84;
        constexpr int kWeightBytesPerRow = 2688;
        constexpr int kRowsPerSIMD = 4;

        const bool is_up = simdgroup_index_in_threadgroup == 1;
        const int output_row = threadgroup_position_in_grid.y * kRowsPerSIMD;
        const device uint* weight = is_up ? up_weight : gate_weight;
        const device ushort* indices = is_up ? up_indices : gate_indices;

        const device uchar* weight_bytes =
            reinterpret_cast<const device uchar*>(weight)
            + output_row * kWeightBytesPerRow
            + thread_index_in_simdgroup * 4;
        const device ushort* row_indices =
            indices + output_row * kGroupsPerRow
            + thread_index_in_simdgroup / 8;
        const device bfloat* input = x + thread_index_in_simdgroup * 8;

        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < 21; ++block) {
            float values[8];
            const float input_sum = gemma4_load_qmv_values(input, values);

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const ushort metadata_index = row_indices[row * kGroupsPerRow];
                const uint pair = is_up
                    ? up_lut[metadata_index]
                    : gate_lut[metadata_index];
                result[row] += gemma4_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_pair_scale(pair),
                    gemma4_pair_bias(pair),
                    input_sum);
            }

            weight_bytes += 128;
            row_indices += 4;
            input += 256;
        }

        threadgroup bfloat projections[2][kRowsPerSIMD];
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                projections[is_up ? 1 : 0][row] =
                    static_cast<bfloat>(result[row]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simdgroup_index_in_threadgroup == 0
            && thread_index_in_simdgroup < kRowsPerSIMD
        ) {
            const int row = thread_index_in_simdgroup;
            const bfloat gate = projections[0][row];
            const bfloat up = projections[1][row];

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
        """,
    header: """
        using namespace metal;

        inline float gemma4_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float gemma4_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float gemma4_load_qmv_values(
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

        inline float gemma4_qdot_4bit(
            const device uchar* weight,
            const thread float* values,
            float scale,
            float bias,
            float input_sum
        ) {
            const device ushort* packed =
                reinterpret_cast<const device ushort*>(weight);
            float accumulator = 0;
            for (int index = 0; index < 2; ++index) {
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

struct FusedGateUpProjection: @unchecked Sendable {
    let gate: FastQuantizedProjection
    let up: FastQuantizedProjection
    let metadataMode: FusedGateUpMetadataMode
    let indexedGate: IndexedAffineMetadata?
    let indexedUp: IndexedAffineMetadata?
    private let packed12IndexedGateUp: Packed12GateUpMetadataPair?
    private let usePacked12IndexedGateUp: Bool
    private let verifyPacked12IndexedBits: Bool

    init(
        gate: QuantizedLinear,
        up: QuantizedLinear,
        metadataMode: FusedGateUpMetadataMode = .raw
    ) {
        self.init(
            gate: FastQuantizedProjection(gate),
            up: FastQuantizedProjection(up),
            metadataMode: metadataMode
        )
    }

    init(
        gate: FastQuantizedProjection,
        up: FastQuantizedProjection,
        metadataMode: FusedGateUpMetadataMode = .raw,
        gateIndexedMetadata: IndexedAffineMetadata? = nil,
        upIndexedMetadata: IndexedAffineMetadata? = nil
    ) {
        self.gate = gate
        self.up = up
        self.metadataMode = metadataMode
        precondition(supportsFusedGateUpKernelBuffers(gate: gate, up: up))
        guard let gateBiases = gate.biases, let upBiases = up.biases else {
            preconditionFailure("fused gate/up QMV requires affine biases")
        }
        let usePacked12IndexedGateUp = gemma4GateUpEnvironmentFlag(
            "DARKBLOOM_PACKED_GATE_UP_INDICES",
            default: true
        )
        let verifyPacked12IndexedBits = gemma4GateUpEnvironmentFlag(
            "DARKBLOOM_VERIFY_PACKED_GATE_UP_BITS",
            default: false
        )
        var packed12IndexedGateUp: Packed12GateUpMetadataPair?
        switch metadataMode {
        case .raw:
            self.indexedGate = nil
            self.indexedUp = nil
        case .indexed:
            let indexedGate = gateIndexedMetadata ?? makeIndexedAffineMetadata(
                scales: gate.scales,
                biases: gateBiases
            )
            let indexedUp = upIndexedMetadata ?? makeIndexedAffineMetadata(
                scales: up.scales,
                biases: upBiases
            )
            precondition(supportsIndexedAffineMetadata(
                indexedGate, shape: gate.scales.shape))
            precondition(supportsIndexedAffineMetadata(
                indexedUp, shape: up.scales.shape))
            self.indexedGate = indexedGate
            self.indexedUp = indexedUp
            if usePacked12IndexedGateUp || verifyPacked12IndexedBits,
               (1...4_096).contains(indexedGate.lut.size),
               (1...4_096).contains(indexedUp.lut.size),
               let packedGate = Packed12GateUpMetadata(metadata: indexedGate),
               let packedUp = Packed12GateUpMetadata(metadata: indexedUp)
            {
                packed12IndexedGateUp = Packed12GateUpMetadataPair(
                    gate: packedGate,
                    up: packedUp
                )
            }
        }
        self.packed12IndexedGateUp = packed12IndexedGateUp
        self.usePacked12IndexedGateUp = usePacked12IndexedGateUp
        self.verifyPacked12IndexedBits = verifyPacked12IndexedBits
    }

    func callAsFunction(_ input: MLXArray) -> (MLXArray, MLXArray) {
        precondition(supportsGemma4FusedGateUpInput(input))
        guard let gateBiases = gate.biases, let upBiases = up.biases else {
            preconditionFailure("fused gate/up QMV requires affine biases")
        }

        let outputWidth = gate.weight.dim(0)
        var outputShape = input.shape
        outputShape[outputShape.count - 1] = outputWidth
        let outputs: [MLXArray]
        switch metadataMode {
        case .raw:
            outputs = gemma4FusedGateUpQMV(
                [
                    gate.weight, gate.scales, gateBiases,
                    up.weight, up.scales, upBiases, input,
                ],
                grid: (32, outputWidth / 2, 1),
                threadGroup: (32, 2, 1),
                outputShapes: [outputShape, outputShape],
                outputDTypes: [.bfloat16, .bfloat16]
            )
        case .indexed:
            guard let indexedGate, let indexedUp else {
                preconditionFailure("indexed gate/up metadata was not prepared")
            }
            outputs = gemma4IndexedFusedGateUpQMV(
                [
                    gate.weight, indexedGate.indices, indexedGate.lut,
                    up.weight, indexedUp.indices, indexedUp.lut, input,
                ],
                grid: (32, outputWidth / 2, 1),
                threadGroup: (32, 2, 1),
                outputShapes: [outputShape, outputShape],
                outputDTypes: [.bfloat16, .bfloat16]
            )
        }
        return (outputs[0], outputs[1])
    }

    func activated(_ input: MLXArray) -> MLXArray {
        precondition(supportsGemma4FusedGateUpInput(input))
        precondition(metadataMode == .indexed)
        guard let indexedGate, let indexedUp else {
            preconditionFailure("indexed gate/up metadata was not prepared")
        }
        var outputShape = input.shape
        let outputWidth = gate.weight.dim(0)
        outputShape[outputShape.count - 1] = outputWidth
        let runPacked = usePacked12IndexedGateUp || verifyPacked12IndexedBits

        let packedOutput: MLXArray?
        if runPacked, let packed12IndexedGateUp {
            packedOutput = gemma4Packed12IndexedFusedGateUpActivationQMV(
                [
                    gate.weight, packed12IndexedGateUp.gate.bytes, indexedGate.lut,
                    up.weight, packed12IndexedGateUp.up.bytes, indexedUp.lut, input,
                ],
                grid: (32, outputWidth / 2, 1),
                threadGroup: (32, 2, 1),
                outputShapes: [outputShape],
                outputDTypes: [.bfloat16]
            )[0]
        } else {
            packedOutput = nil
        }

        let needsPromoted = !usePacked12IndexedGateUp
            || verifyPacked12IndexedBits
            || packed12IndexedGateUp == nil
        let promotedOutput: MLXArray? = needsPromoted
            ? gemma4IndexedFusedGateUpActivationQMV(
                [
                    gate.weight, indexedGate.indices, indexedGate.lut,
                    up.weight, indexedUp.indices, indexedUp.lut, input,
                ],
                grid: (32, outputWidth / 2, 1),
                threadGroup: (32, 2, 1),
                outputShapes: [outputShape],
                outputDTypes: [.bfloat16]
            )[0]
            : nil

        if verifyPacked12IndexedBits,
           let packedOutput,
           let promotedOutput
        {
            let matches = arrayEqual(
                packedOutput.view(dtype: .uint16),
                promotedOutput.view(dtype: .uint16)
            )
            eval(matches)
            precondition(
                matches.item(Bool.self),
                "fixed-tile packed gate/up activation differs from promoted kernel"
            )
            return promotedOutput
        }
        if let packedOutput {
            return packedOutput
        }
        guard let promotedOutput else {
            preconditionFailure("gate/up activation kernel was not selected")
        }
        return promotedOutput
    }
}
