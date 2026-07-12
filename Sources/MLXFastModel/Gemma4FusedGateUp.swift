import MLX
import MLXNN

struct IndexedAffineMetadata: @unchecked Sendable {
    let indices: MLXArray
    let lut: MLXArray
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
        }
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
        outputShape[outputShape.count - 1] = gate.weight.dim(0)
        return gemma4IndexedFusedGateUpActivationQMV(
            [
                gate.weight, indexedGate.indices, indexedGate.lut,
                up.weight, indexedUp.indices, indexedUp.lut, input,
            ],
            grid: (32, gate.weight.dim(0) / 2, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [outputShape],
            outputDTypes: [.bfloat16]
        )[0]
    }
}
