import MLX
import MLXNN

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
    ensureRowContiguous: false
)

struct FusedGateUpProjection: @unchecked Sendable {
    let gate: FastQuantizedProjection
    let up: FastQuantizedProjection

    init(gate: QuantizedLinear, up: QuantizedLinear) {
        self.gate = FastQuantizedProjection(gate)
        self.up = FastQuantizedProjection(up)
    }

    init(gate: FastQuantizedProjection, up: FastQuantizedProjection) {
        self.gate = gate
        self.up = up
    }

    func callAsFunction(_ input: MLXArray) -> (MLXArray, MLXArray) {
        precondition(input.dim(-1) == 5_376)
        precondition(input.size == 5_376)
        precondition(gate.weight.dim(0) == up.weight.dim(0))
        precondition(gate.weight.dim(0).isMultiple(of: 4))
        guard let gateBiases = gate.biases, let upBiases = up.biases else {
            preconditionFailure("fused gate/up QMV requires affine biases")
        }

        let outputWidth = gate.weight.dim(0)
        var outputShape = input.shape
        outputShape[outputShape.count - 1] = outputWidth
        let outputs = gemma4FusedGateUpQMV(
            [
                gate.weight, gate.scales, gateBiases,
                up.weight, up.scales, upBiases, input,
            ],
            grid: (32, outputWidth / 2, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [outputShape, outputShape],
            outputDTypes: [.bfloat16, .bfloat16]
        )
        return (outputs[0], outputs[1])
    }
}
