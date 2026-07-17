import Foundation
import MLX
import MLXFastCore
import MLXNN

private func gemma4TiedHeadEnvironmentFlag(
    _ name: String,
    default defaultValue: Bool
) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return defaultValue
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

private let gemma4TiedHeadCoTiledEnabled = gemma4TiedHeadEnvironmentFlag(
    "DARKBLOOM_TIED_HEAD_COTILED",
    default: true
)

private let gemma4VerifyTiedHeadCoTiledBits = gemma4TiedHeadEnvironmentFlag(
    "DARKBLOOM_VERIFY_TIED_HEAD_COTILED_BITS",
    default: false
)

private let gemma4TiedVocabularyHeadQMV = MLXFast.metalKernel(
    name: "gemma4_tied_vocabulary_head_qmv_262144x5376_v1",
    inputNames: ["weight", "scales", "biases", "x"],
    outputNames: ["output"],
    source: """
        constexpr int kInputWidth = 5376;
        constexpr int kOutputWidth = 262144;
        constexpr int kGroupsPerRow = 84;
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
            const float input_sum =
                gemma4_tied_head_load_values(input, values);

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                result[row] += gemma4_tied_head_qdot_4bit(
                    row_weight,
                    values,
                    static_cast<float>(row_scales[row * kGroupsPerRow]),
                    static_cast<float>(row_biases[row * kGroupsPerRow]),
                    input_sum);
            }

            weight_bytes += 128;
            row_scales += 4;
            row_biases += 4;
            input += 256;
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                output[output_row + row] =
                    static_cast<bfloat>(result[row]);
            }
        }
        """,
    header: """
        using namespace metal;

        inline float gemma4_tied_head_load_values(
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

        inline float gemma4_tied_head_qdot_4bit(
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

private let gemma4TiedVocabularyHeadPacked13SoftcapQMV = MLXFast.metalKernel(
    name: "gemma4_tied_vocabulary_head_packed13_softcap_qmv_262144x5376_v2",
    inputNames: ["weight", "packed_indices", "lut", "x", "cap"],
    outputNames: ["output"],
    source: """
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
        const device bfloat* input = x + thread_index_in_simdgroup * 8;

        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < 21; ++block) {
            float values[8];
            const float input_sum =
                gemma4_tied_head_packed13_load_values(input, values);
            const uint metadata_column =
                block * 4 + thread_index_in_simdgroup / 8;

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const ushort metadata_index =
                    gemma4_tied_head_extract_packed13(
                        row_packed_indices + row * kPackedWordsPerRow,
                        metadata_column);
                const uint pair = lut[metadata_index];
                result[row] += gemma4_tied_head_packed13_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_tied_head_pair_scale(pair),
                    gemma4_tied_head_pair_bias(pair),
                    input_sum);
            }

            weight_bytes += 128;
            input += 256;
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                // Match the stock graph's materialized BF16 projection before
                // applying its Float32 softcap suffix in the same command.
                const bfloat projected = static_cast<bfloat>(result[row]);
                const float quotient = static_cast<float>(projected) / cap;
                const float softened = metal::precise::tanh(quotient);
                output[output_row + row] = softened * cap;
            }
        }
        """,
    header: """
        using namespace metal;

        inline ushort gemma4_tied_head_extract_packed13(
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

        inline float gemma4_tied_head_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float gemma4_tied_head_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float gemma4_tied_head_packed13_load_values(
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

        inline float gemma4_tied_head_packed13_qdot_4bit(
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

/// Co-tiled variant of the packed13 softcap QMV above. The traversal
/// consumes one forward-contiguous per-threadgroup tile stream
/// (`Gemma4TiedHeadCoTiledPayloadLayout`): ten pair tiles of `[even-block
/// weights | odd-block weights | 13-byte packed13 pair runs]` followed by
/// one tail tile of `[block-20 weights | 8-byte tail metadata runs]`. The
/// arithmetic, LUT pair decode, per-row accumulation order (block 2p, then
/// 2p + 1, then the tail block 20), reductions, and the BF16-projection +
/// Float32 softcap epilogue are identical to the packed13 kernel — only the
/// addresses change, so outputs are bit-identical by construction once the
/// payload reproduces the same weight words and metadata bytes. Block pair
/// `p` consumes packed13 columns `8p..<8p+8`, whose bits `[104p, 104p+104)`
/// are exactly bytes `[13p, 13p+13)` of the source row, so within a copied
/// run every column sits at bit offset `13 * column_in_pair` and the
/// extraction below reconstructs the exact 13-bit indexes the packed13
/// kernel's word-based extraction produces.
private let gemma4CoTiledTiedVocabularyHeadPacked13SoftcapQMV = MLXFast.metalKernel(
    name: "gemma4_cotiled_tied_vocabulary_head_packed13_softcap_qmv_262144x5376_v1",
    inputNames: ["cotiled_payload", "lut", "x", "cap"],
    outputNames: ["output"],
    source: """
        constexpr int kRowsPerSIMD = 4;
        constexpr int kSIMDSize = 32;
        constexpr int kSlotWeightWords = kRowsPerSIMD * kSIMDSize;
        constexpr int kSlotsPerThreadgroup = 4;
        constexpr int kBlockWeightWords =
            kSlotsPerThreadgroup * kSlotWeightWords;
        constexpr int kPairWeightWords = 2 * kBlockWeightWords;
        constexpr int kPairCount = 10;
        constexpr int kPairMetadataBytesPerRow = 13;
        constexpr int kSlotPairMetadataBytes =
            kRowsPerSIMD * kPairMetadataBytesPerRow;
        constexpr int kWordsPerPair = kPairWeightWords
            + kSlotsPerThreadgroup * kSlotPairMetadataBytes / 4;
        constexpr int kWordsPerTail =
            kBlockWeightWords + 8 * kSlotsPerThreadgroup;
        constexpr int kWordsPerThreadgroup =
            kPairCount * kWordsPerPair + kWordsPerTail;

        const int slot = simdgroup_index_in_threadgroup;
        const int output_row =
            threadgroup_position_in_grid.y
                * kRowsPerSIMD * kSlotsPerThreadgroup
            + slot * kRowsPerSIMD;

        const uint lane = thread_index_in_simdgroup;
        const uint lane_group = lane >> 3;
        const device uint* tile_words = cotiled_payload
            + threadgroup_position_in_grid.y * kWordsPerThreadgroup;
        const device bfloat* input = x + lane * 8;

        float result[kRowsPerSIMD] = {0};
        for (int block_pair = 0; block_pair < kPairCount; ++block_pair) {
            float even_values[8];
            const float even_input_sum =
                gemma4_cotiled_tied_head_load_values(input, even_values);
            uint odd_luts[kRowsPerSIMD];

            const device uint* even_weight_words = tile_words
                + slot * kSlotWeightWords + lane;
            const device uint* odd_weight_words = tile_words
                + kBlockWeightWords + slot * kSlotWeightWords + lane;
            const device uchar* metadata_bytes =
                reinterpret_cast<const device uchar*>(
                    tile_words + kPairWeightWords)
                + slot * kSlotPairMetadataBytes;

            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_tile =
                    metadata_bytes + row * kPairMetadataBytesPerRow;
                const ushort even_index =
                    gemma4_cotiled_tied_head_extract13(row_tile, lane_group);
                const ushort odd_index =
                    gemma4_cotiled_tied_head_extract13(
                        row_tile, 4 + lane_group);
                const uint even_pair = lut[even_index];
                odd_luts[row] = lut[odd_index];
                const device uchar* row_weight =
                    reinterpret_cast<const device uchar*>(
                        even_weight_words + row * kSIMDSize);
                result[row] += gemma4_cotiled_tied_head_qdot_4bit(
                    row_weight,
                    even_values,
                    gemma4_cotiled_tied_head_pair_scale(even_pair),
                    gemma4_cotiled_tied_head_pair_bias(even_pair),
                    even_input_sum);
            }

            input += 256;
            float odd_values[8];
            const float odd_input_sum =
                gemma4_cotiled_tied_head_load_values(input, odd_values);
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const uint odd_pair = odd_luts[row];
                const device uchar* row_weight =
                    reinterpret_cast<const device uchar*>(
                        odd_weight_words + row * kSIMDSize);
                result[row] += gemma4_cotiled_tied_head_qdot_4bit(
                    row_weight,
                    odd_values,
                    gemma4_cotiled_tied_head_pair_scale(odd_pair),
                    gemma4_cotiled_tied_head_pair_bias(odd_pair),
                    odd_input_sum);
            }

            input += 256;
            tile_words += kWordsPerPair;
        }

        // The 21st block is the unpaired tail tile: block-20 weights for
        // every slot, then the packed13 row's bytes [130, 138) per row.
        // Columns 80..83 start exactly at bit 1040 = byte 130, so the
        // within-run bit offsets are again 13 * lane_group.
        float tail_values[8];
        const float tail_input_sum =
            gemma4_cotiled_tied_head_load_values(input, tail_values);
        const device uint* tail_weight_words = tile_words
            + slot * kSlotWeightWords + lane;
        const device uchar* tail_metadata =
            reinterpret_cast<const device uchar*>(
                tile_words + kBlockWeightWords)
            + slot * 32;
        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            const device uchar* row_tail = tail_metadata + row * 8;
            const ushort metadata_index =
                gemma4_cotiled_tied_head_extract13(row_tail, lane_group);
            const uint pair = lut[metadata_index];
            const device uchar* row_weight =
                reinterpret_cast<const device uchar*>(
                    tail_weight_words + row * kSIMDSize);
            result[row] += gemma4_cotiled_tied_head_qdot_4bit(
                row_weight,
                tail_values,
                gemma4_cotiled_tied_head_pair_scale(pair),
                gemma4_cotiled_tied_head_pair_bias(pair),
                tail_input_sum);
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                // Match the stock graph's materialized BF16 projection before
                // applying its Float32 softcap suffix in the same command.
                const bfloat projected = static_cast<bfloat>(result[row]);
                const float quotient = static_cast<float>(projected) / cap;
                const float softened = metal::precise::tanh(quotient);
                output[output_row + row] = softened * cap;
            }
        }
        """,
    header: """
        using namespace metal;

        // Extracts the 13-bit index at bit offset 13 * column of a copied
        // packed13 run. Three-byte little-endian load; the post-shift
        // 0x1fff mask discards every bit the run does not own, so the
        // third byte never leaks into the result when only two bytes are
        // needed (the payload always continues past a pair run, so the
        // load stays in-bounds).
        inline ushort gemma4_cotiled_tied_head_extract13(
            const device uchar* bytes,
            uint column
        ) {
            const uint bit_offset = column * 13;
            const uint byte_offset = bit_offset >> 3;
            const uint shift = bit_offset & 7;
            const uint low = bytes[byte_offset];
            const uint middle = bytes[byte_offset + 1];
            const uint high = bytes[byte_offset + 2];
            const uint value = (low | (middle << 8) | (high << 16)) >> shift;
            return static_cast<ushort>(value & 0x1fff);
        }

        inline float gemma4_cotiled_tied_head_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float gemma4_cotiled_tied_head_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float gemma4_cotiled_tied_head_load_values(
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

        inline float gemma4_cotiled_tied_head_qdot_4bit(
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

private let gemma4TiedHeadMaximumPacked13LUTCount = 8_192

struct Gemma4TiedHeadPacked13Metadata: @unchecked Sendable {
    let packedIndices: MLXArray
    let lut: MLXArray
}

/// The transform-authored co-tiled tied-head decode payload
/// (`model.embed_tokens.tied_head_cotiled_payload`), already validated
/// byte-for-byte against the separate weight and packed13 index tensors by
/// `validateGemma4TiedHeadCoTiledPayloadBytes` before arrays were loaded.
/// The layout is documented in `Gemma4TiedHeadCoTiledPayloadValidation.swift`
/// and in the transform's `CoTiledTiedHeadPayloadCoding`.
struct Gemma4TiedHeadCoTiledPayload: @unchecked Sendable {
    let words: MLXArray
}

func supportsGemma4TiedHeadCoTiledPayload(
    _ payload: Gemma4TiedHeadCoTiledPayload,
    metadata: Gemma4TiedHeadPacked13Metadata
) -> Bool {
    payload.words.dtype == .uint32
        && payload.words.shape == [
            Gemma4TiedHeadCoTiledPayloadLayout.threadgroups,
            Gemma4TiedHeadCoTiledPayloadLayout.wordsPerThreadgroup,
        ]
        && (1...gemma4TiedHeadMaximumPacked13LUTCount).contains(
            metadata.lut.size
        )
}

func validateGemma4TiedHeadPacked13MetadataLayout(
    _ metadata: Gemma4TiedHeadPacked13Metadata,
    weight: MLXArray,
    scales: MLXArray,
    biases: MLXArray
) throws {
    guard weight.dtype == .uint32,
          weight.shape == [262_144, 672],
          scales.dtype == .bfloat16,
          scales.shape == [262_144, 84],
          biases.dtype == .bfloat16,
          biases.shape == scales.shape,
          metadata.packedIndices.dtype == .uint32,
          metadata.packedIndices.shape == [262_144, 35],
          metadata.lut.dtype == .uint32,
          metadata.lut.ndim == 1,
          (1...gemma4TiedHeadMaximumPacked13LUTCount).contains(
              metadata.lut.size
          )
    else {
        throw MLXFastError.invalidInput(
            "tied-head packed13 metadata has invalid dtype or shape"
        )
    }
}

func supportsGemma4TiedVocabularyHead(
    weight: MLXArray,
    scales: MLXArray,
    biases: MLXArray?,
    groupSize: Int,
    bits: Int
) -> Bool {
    guard let biases else { return false }
    return groupSize == 64
        && bits == 4
        && weight.dtype == .uint32
        && weight.shape == [262_144, 672]
        && scales.dtype == .bfloat16
        && scales.shape == [262_144, 84]
        && biases.dtype == .bfloat16
        && biases.shape == [262_144, 84]
}

func isGemma4ProductionTiedVocabularyHead(_ embedding: Embedding) -> Bool {
    guard let embedding = embedding as? QuantizedEmbedding else { return false }
    return embedding.mode == .affine
        && embedding.groupSize == 64
        && embedding.bits == 4
        && embedding.weight.dtype == .uint32
        && embedding.weight.shape == [262_144, 672]
}

struct Gemma4TiedVocabularyHead: @unchecked Sendable {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let packed13Metadata: Gemma4TiedHeadPacked13Metadata?
    let coTiledPayload: Gemma4TiedHeadCoTiledPayload?
    private let useCoTiled: Bool
    private let verifyCoTiledBits: Bool

    init?(
        _ embedding: Embedding,
        packed13Metadata: Gemma4TiedHeadPacked13Metadata?,
        coTiledPayload: Gemma4TiedHeadCoTiledPayload? = nil
    ) {
        guard let embedding = embedding as? QuantizedEmbedding,
              embedding.mode == .affine,
              let biases = embedding.biases,
              supportsGemma4TiedVocabularyHead(
                weight: embedding.weight,
                scales: embedding.scales,
                biases: biases,
                groupSize: embedding.groupSize,
                bits: embedding.bits
              )
        else { return nil }
        self.weight = embedding.weight
        self.scales = embedding.scales
        self.biases = biases
        self.packed13Metadata = packed13Metadata
        self.useCoTiled = gemma4TiedHeadCoTiledEnabled
        self.verifyCoTiledBits = gemma4VerifyTiedHeadCoTiledBits
        // The co-tiled kernel engages only when the transform authored the
        // payload, the packed13 metadata (which carries the LUT the co-tiled
        // kernel indexes) is present, and the payload shape matches exactly
        // (fail-open to the packed13 kernel below, never mixed).
        if gemma4TiedHeadCoTiledEnabled || gemma4VerifyTiedHeadCoTiledBits,
            let coTiledPayload,
            let packed13Metadata,
            supportsGemma4TiedHeadCoTiledPayload(
                coTiledPayload,
                metadata: packed13Metadata
            )
        {
            self.coTiledPayload = coTiledPayload
        } else {
            self.coTiledPayload = nil
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 5_376])
        return gemma4TiedVocabularyHeadQMV(
            [weight, scales, biases, input],
            grid: (32, 65_536, 1),
            threadGroup: (32, 4, 1),
            outputShapes: [[1, 1, 262_144]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    func packed13Softcapped(_ input: MLXArray, cap: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 5_376])
        precondition(cap.dtype == .float32)
        precondition(cap.size == 1)
        guard let packed13Metadata else {
            preconditionFailure("tied vocabulary packed13 metadata was not prepared")
        }
        guard let coTiledPayload else {
            return packed13ReferenceSoftcapped(
                input,
                cap: cap,
                metadata: packed13Metadata
            )
        }
        let candidate = coTiledPacked13Softcapped(
            input,
            cap: cap,
            payload: coTiledPayload,
            metadata: packed13Metadata
        )
        if verifyCoTiledBits {
            let reference = packed13ReferenceSoftcapped(
                input,
                cap: cap,
                metadata: packed13Metadata
            )
            verifyRawFloat32(
                candidate,
                stock: reference,
                candidateName: "co-tiled packed13 fused-softcap",
                stockName: "packed13 fused-softcap"
            )
            if !useCoTiled {
                // Verifier-only mode must not change the selected output.
                return reference
            }
        }
        return candidate
    }

    /// The pre-co-tile packed13 fused-softcap kernel over the separate
    /// weight and metadata streams. This is the fail-open path and the
    /// co-tile verifier's reference.
    private func packed13ReferenceSoftcapped(
        _ input: MLXArray,
        cap: MLXArray,
        metadata: Gemma4TiedHeadPacked13Metadata
    ) -> MLXArray {
        gemma4TiedVocabularyHeadPacked13SoftcapQMV(
            [weight, metadata.packedIndices, metadata.lut, input, cap],
            grid: (32, 65_536, 1),
            threadGroup: (32, 4, 1),
            outputShapes: [[1, 1, 262_144]],
            outputDTypes: [.float32]
        )[0]
    }

    private func coTiledPacked13Softcapped(
        _ input: MLXArray,
        cap: MLXArray,
        payload: Gemma4TiedHeadCoTiledPayload,
        metadata: Gemma4TiedHeadPacked13Metadata
    ) -> MLXArray {
        gemma4CoTiledTiedVocabularyHeadPacked13SoftcapQMV(
            [payload.words, metadata.lut, input, cap],
            grid: (32, 65_536, 1),
            threadGroup: (32, 4, 1),
            outputShapes: [[1, 1, 262_144]],
            outputDTypes: [.float32]
        )[0]
    }

    var supportsExactTwoVectorPacked13: Bool {
        packed13Metadata != nil
    }

    func exactTwoVectorPacked13Softcapped(
        _ input: MLXArray,
        cap: MLXArray
    ) -> MLXArray {
        precondition(input.dtype == .bfloat16 && input.shape == [2, 5_376])
        precondition(cap.dtype == .float32 && cap.size == 1)
        guard let packed13Metadata else {
            preconditionFailure("exact two-vector tied-head metadata is unavailable")
        }
        return gemma4ExactTwoVectorTiedHead(
            weight: weight,
            packedIndices: packed13Metadata.packedIndices,
            lut: packed13Metadata.lut,
            input: input,
            cap: cap
        )
    }

    func verifyRawFloat32(
        _ candidate: MLXArray,
        stock: MLXArray,
        candidateName: String,
        stockName: String
    ) {
        precondition(candidate.dtype == .float32)
        precondition(stock.dtype == .float32)
        precondition(candidate.shape == stock.shape)
        let candidateBits = candidate.view(dtype: .uint32)
        let stockBits = stock.view(dtype: .uint32)
        let equal = arrayEqual(candidateBits, stockBits)
        eval(equal)
        guard equal.item(Bool.self) else {
            let candidateValues = candidateBits.asArray(UInt32.self)
            let stockValues = stockBits.asArray(UInt32.self)
            let mismatch = zip(candidateValues, stockValues).enumerated().first {
                $0.element.0 != $0.element.1
            }
            preconditionFailure(
                "tied vocabulary head raw Float32 mismatch \(candidateName) vs "
                    + "\(stockName) at index "
                    + "\(mismatch?.offset ?? -1): candidate="
                    + "\(mismatch?.element.0 ?? 0), stock="
                    + "\(mismatch?.element.1 ?? 0)"
            )
        }
    }

    func verifyRawBF16(
        _ candidate: MLXArray,
        stock: MLXArray,
        candidateName: String,
        stockName: String
    ) {
        precondition(candidate.dtype == .bfloat16)
        precondition(stock.dtype == .bfloat16)
        precondition(candidate.shape == stock.shape)
        let candidateBits = candidate.view(dtype: .uint16)
        let stockBits = stock.view(dtype: .uint16)
        let equal = arrayEqual(candidateBits, stockBits)
        eval(equal)
        guard equal.item(Bool.self) else {
            let candidateValues = candidateBits.asArray(UInt16.self)
            let stockValues = stockBits.asArray(UInt16.self)
            let mismatch = zip(candidateValues, stockValues).enumerated().first {
                $0.element.0 != $0.element.1
            }
            preconditionFailure(
                "tied vocabulary head raw BF16 mismatch \(candidateName) vs "
                    + "\(stockName) at index "
                    + "\(mismatch?.offset ?? -1): candidate="
                    + "\(mismatch?.element.0 ?? 0), stock="
                    + "\(mismatch?.element.1 ?? 0)"
            )
        }
    }
}
