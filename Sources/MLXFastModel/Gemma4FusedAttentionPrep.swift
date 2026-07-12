import MLX

private func makeGemma4FusedAttentionRMSKernel(
    name: String,
    headDim: Int,
    kvHeads: Int,
    sharesFullKVReduction: Bool
) -> MLXFast.MLXFastKernel {
    precondition(headDim == 256 || headDim == 512)
    precondition(kvHeads == 16 || kvHeads == 4)
    return MLXFast.metalKernel(
        name: name,
        inputNames: [
            "raw_q", "raw_k", "raw_v", "q_weight", "k_weight",
            "position", "rope_cosines", "rope_sines",
        ],
        outputNames: ["queries", "keys", "values"],
        source: """
            constexpr uint kHeadDim = \(headDim);
            constexpr uint kQHeads = 32;
            constexpr uint kKVHeads = \(kvHeads);
            constexpr uint kReads = 4;
            constexpr uint kSIMDSize = 32;
            constexpr bool kSharesFullKVReduction = \(sharesFullKVReduction);

            const uint combined_row = threadgroup_position_in_grid.y;
            const bool is_q = combined_row < kQHeads;
            const bool is_k = !is_q && (
                kSharesFullKVReduction
                    || combined_row < kQHeads + kKVHeads);
            const uint projection_row = is_q
                ? combined_row
                : (is_k ? combined_row - kQHeads
                        : combined_row - kQHeads - kKVHeads);

            const device bfloat* input = is_q
                ? raw_q + projection_row * kHeadDim
                : (is_k ? raw_k : raw_v) + projection_row * kHeadDim;
            const device bfloat* weight = is_q ? q_weight : k_weight;
            device bfloat* output = is_q
                ? queries + projection_row * kHeadDim
                : (is_k ? keys : values) + projection_row * kHeadDim;
            const bool has_weight = is_q || is_k;

            float accumulator = 0;
            input += thread_position_in_threadgroup.x * kReads;
            if (thread_position_in_threadgroup.x * kReads + kReads <= kHeadDim) {
                for (uint index = 0; index < kReads; ++index) {
                    const float value = input[index];
                    accumulator += value * value;
                }
            }
            accumulator = simd_sum(accumulator);

            threadgroup float inverse_mean[1];
            threadgroup float local_sums[kSIMDSize];
            threadgroup bfloat normalized_row[kHeadDim];
            if (simdgroup_index_in_threadgroup == 0) {
                local_sums[thread_index_in_simdgroup] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (thread_index_in_simdgroup == 0) {
                local_sums[simdgroup_index_in_threadgroup] = accumulator;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simdgroup_index_in_threadgroup == 0) {
                accumulator = simd_sum(local_sums[thread_index_in_simdgroup]);
                if (thread_index_in_simdgroup == 0) {
                    inverse_mean[0] = metal::precise::rsqrt(
                        accumulator / kHeadDim + 1.0e-6f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const device bfloat* row_weight =
                weight + thread_position_in_threadgroup.x * kReads;
            for (uint index = 0; index < kReads; ++index) {
                const uint dimension =
                    thread_position_in_threadgroup.x * kReads + index;
                const bfloat normalized = static_cast<bfloat>(
                    input[index] * inverse_mean[0]);
                const bfloat weighted = has_weight
                    ? row_weight[index] * normalized
                    : static_cast<bfloat>(1.0f) * normalized;
                normalized_row[dimension] = weighted;
                if (!has_weight) {
                    output[dimension] = weighted;
                }
                if (kSharesFullKVReduction && is_k) {
                    values[projection_row * kHeadDim + dimension] =
                        static_cast<bfloat>(1.0f) * normalized;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (has_weight) {
                constexpr uint kPairs = kHeadDim / 2;
                constexpr uint kThreads = kHeadDim / kReads;
                for (uint pair = thread_position_in_threadgroup.x;
                     pair < kPairs;
                     pair += kThreads) {
                    const uint rope_index =
                        static_cast<uint>(position) * kPairs + pair;
                    const float cosine = rope_cosines[rope_index];
                    const float sine = rope_sines[rope_index];
                    const float left = static_cast<float>(normalized_row[pair]);
                    const float right = static_cast<float>(
                        normalized_row[pair + kPairs]);
                    output[pair] = static_cast<bfloat>(
                        left * cosine - right * sine);
                    output[pair + kPairs] = static_cast<bfloat>(
                        left * sine + right * cosine);
                }
            }
            """,
        header: "using namespace metal;",
        ensureRowContiguous: true
    )
}

private let gemma4FusedSlidingAttentionRMS = makeGemma4FusedAttentionRMSKernel(
    name: "gemma4_fused_sliding_attention_rms_rope_table_256_v4",
    headDim: 256,
    kvHeads: 16,
    sharesFullKVReduction: false
)

private let gemma4FusedFullAttentionRMS = makeGemma4FusedAttentionRMSKernel(
    name: "gemma4_fused_full_attention_rms_rope_table_shared_kv_512_v5",
    headDim: 512,
    kvHeads: 4,
    sharesFullKVReduction: true
)

private struct Gemma4PreparedArray: @unchecked Sendable {
    let value: MLXArray
}

private let gemma4FullAttentionFrequencies: Gemma4PreparedArray = {
    let exponents = MLXArray(stride(from: 0, to: 128, by: 2))
        .asType(.float32) / Float(512)
    let realFrequencies = MLX.pow(Float(1_000_000), exponents)
    let passThrough = MLXArray(Array(repeating: Float.infinity, count: 192))
    let frequencies = concatenated([realFrequencies, passThrough], axis: -1)
    eval(frequencies)
    return Gemma4PreparedArray(value: frequencies)
}()

private let gemma4AttentionRopeTableKernel = MLXFast.metalKernel(
    name: "gemma4_attention_rope_tables_4096_v1",
    inputNames: ["full_freqs"],
    outputNames: [
        "sliding_cosines", "sliding_sines", "full_cosines", "full_sines",
    ],
    source: """
        constexpr uint kPositions = 4096;
        constexpr uint kSlidingPairs = 128;
        constexpr uint kFullPairs = 256;
        const uint pair = thread_position_in_grid.x;
        const uint position = thread_position_in_grid.y;
        if (position >= kPositions || pair >= kFullPairs) {
            return;
        }

        const float position_value = 1.0f * static_cast<float>(position);
        const float full_inverse_frequency = 1.0f / full_freqs[pair];
        const float full_theta = position_value * full_inverse_frequency;
        const uint full_index = position * kFullPairs + pair;
        full_cosines[full_index] = metal::fast::cos(full_theta);
        full_sines[full_index] = metal::fast::sin(full_theta);

        if (pair < kSlidingPairs) {
            const float sliding_inverse_frequency = metal::exp2(
                -static_cast<float>(pair) / 128.0f
                    * as_type<float>(0x41549a78u));
            const float sliding_theta =
                position_value * sliding_inverse_frequency;
            const uint sliding_index = position * kSlidingPairs + pair;
            sliding_cosines[sliding_index] = metal::fast::cos(sliding_theta);
            sliding_sines[sliding_index] = metal::fast::sin(sliding_theta);
        }
        """,
    header: "using namespace metal;",
    ensureRowContiguous: true
)

private struct Gemma4AttentionRopeTables: @unchecked Sendable {
    let positions: MLXArray
    let positionViews: [MLXArray]
    let slidingCosines: MLXArray
    let slidingSines: MLXArray
    let fullCosines: MLXArray
    let fullSines: MLXArray
}

private let gemma4AttentionRopeTables: Gemma4AttentionRopeTables = {
    let positions = MLXArray(Int32(0)..<Int32(4096))
    let positionViews = (0..<4096).map { positions[$0] }
    let outputs = gemma4AttentionRopeTableKernel(
        [gemma4FullAttentionFrequencies.value],
        grid: (256, 4096, 1),
        threadGroup: (32, 8, 1),
        outputShapes: [
            [4096, 128], [4096, 128], [4096, 256], [4096, 256],
        ],
        outputDTypes: [.float32, .float32, .float32, .float32]
    )
    eval(positions, outputs[0], outputs[1], outputs[2], outputs[3])
    return Gemma4AttentionRopeTables(
        positions: positions,
        positionViews: positionViews,
        slidingCosines: outputs[0],
        slidingSines: outputs[1],
        fullCosines: outputs[2],
        fullSines: outputs[3]
    )
}()

struct FusedAttentionRMSPreparation: @unchecked Sendable {
    let isSliding: Bool
    let headDim: Int
    let kvHeads: Int
    let qNormWeight: MLXArray
    let kNormWeight: MLXArray
    let positions: MLXArray
    let positionViews: [MLXArray]
    let ropeCosines: MLXArray
    let ropeSines: MLXArray

    init?(
        isSliding: Bool,
        headDim: Int,
        kvHeads: Int,
        qNormWeight: MLXArray,
        kNormWeight: MLXArray?,
        eps: Float
    ) {
        guard let kNormWeight,
              eps == 1.0e-6,
              qNormWeight.dtype == .bfloat16,
              kNormWeight.dtype == .bfloat16,
              qNormWeight.shape == [headDim],
              kNormWeight.shape == [headDim],
              (isSliding && headDim == 256 && kvHeads == 16)
                || (!isSliding && headDim == 512 && kvHeads == 4)
        else { return nil }
        self.isSliding = isSliding
        self.headDim = headDim
        self.kvHeads = kvHeads
        self.qNormWeight = qNormWeight
        self.kNormWeight = kNormWeight
        self.positions = gemma4AttentionRopeTables.positions
        self.positionViews = gemma4AttentionRopeTables.positionViews
        self.ropeCosines = isSliding
            ? gemma4AttentionRopeTables.slidingCosines
            : gemma4AttentionRopeTables.fullCosines
        self.ropeSines = isSliding
            ? gemma4AttentionRopeTables.slidingSines
            : gemma4AttentionRopeTables.fullSines
    }

    func supports(offset: Int) -> Bool {
        (0..<4096).contains(offset)
    }

    func callAsFunction(
        rawQueries: MLXArray,
        rawKeys: MLXArray,
        rawValues: MLXArray?,
        offset: Int
    ) -> (MLXArray, MLXArray, MLXArray) {
        let queryWidth = 32 * headDim
        let kvWidth = kvHeads * headDim
        precondition(rawQueries.dtype == .bfloat16)
        precondition(rawKeys.dtype == .bfloat16)
        precondition(rawQueries.shape == [1, 1, queryWidth])
        precondition(rawKeys.shape == [1, 1, kvWidth])
        let valueInput = rawValues ?? rawKeys
        precondition(valueInput.dtype == .bfloat16)
        precondition(valueInput.shape == [1, 1, kvWidth])

        let threads = headDim / 4
        let kernel = isSliding
            ? gemma4FusedSlidingAttentionRMS
            : gemma4FusedFullAttentionRMS
        precondition(supports(offset: offset))
        let position = positionViews[offset]
        let inputs = [
            rawQueries, rawKeys, valueInput, qNormWeight, kNormWeight,
            position, ropeCosines, ropeSines,
        ]
        let outputs = kernel(
            inputs,
            grid: (threads, 32 + (isSliding ? 2 * kvHeads : kvHeads), 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [
                [1, 32, 1, headDim],
                [1, kvHeads, 1, headDim],
                [1, kvHeads, 1, headDim],
            ],
            outputDTypes: [.bfloat16, .bfloat16, .bfloat16]
        )

        return (outputs[0], outputs[1], outputs[2])
    }
}
