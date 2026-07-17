import Foundation
import MLX

/// One transform-authored fixed 12/13-bit projection index stream
/// (`<stem>.metadata_packed_fixed12` / `..._fixed13`), already validated
/// byte-for-byte against the stem's U16 `.metadata_indices` tensor by
/// `validateGemma4PackedProjectionMetadataBytes` before arrays were loaded.
/// The layout is documented in `Gemma4PackedProjectionMetadataValidation.swift`.
struct Gemma4PackedQKVIndexMetadata: @unchecked Sendable {
    let bytes: MLXArray
    let indexBits: Int
}

func supportsGemma4PackedQKVIndexMetadata(
    _ packed: Gemma4PackedQKVIndexMetadata,
    metadata: IndexedAffineMetadata,
    outputWidth: Int
) -> Bool {
    guard let bytesPerRow = Gemma4PackedProjectionMetadataLayout.bytesPerRow(
        groupsPerRow: 84,
        indexBits: packed.indexBits
    ), let maximumLUTCount = Gemma4PackedProjectionMetadataLayout.maximumLUTCount(
        indexBits: packed.indexBits
    ) else {
        return false
    }
    return packed.bytes.dtype == .uint8
        && packed.bytes.shape == [outputWidth, bytesPerRow]
        && (1...maximumLUTCount).contains(metadata.lut.size)
}

private func gemma4QKVEnvironmentFlag(
    _ name: String,
    default defaultValue: Bool
) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return defaultValue
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

private let gemma4PackedSlidingQKVIndicesEnabled = gemma4QKVEnvironmentFlag(
    "DARKBLOOM_PACKED_QKV_INDICES",
    default: true
)

private let gemma4VerifyPackedSlidingQKVBits = gemma4QKVEnvironmentFlag(
    "DARKBLOOM_VERIFY_PACKED_QKV_BITS",
    default: false
)

private let gemma4PackedFullQKIndicesEnabled = gemma4QKVEnvironmentFlag(
    "DARKBLOOM_PACKED_FULL_QK_INDICES",
    default: true
)

private let gemma4VerifyPackedFullQKBits = gemma4QKVEnvironmentFlag(
    "DARKBLOOM_VERIFY_PACKED_FULL_QK_BITS",
    default: false
)

private let gemma4CoTiledSlidingQKVEnabled = gemma4QKVEnvironmentFlag(
    "DARKBLOOM_QKV_COTILED",
    default: true
)

private let gemma4VerifyCoTiledSlidingQKVBits = gemma4QKVEnvironmentFlag(
    "DARKBLOOM_VERIFY_QKV_COTILED_BITS",
    default: false
)

private let gemma4CoTiledFullQKEnabled = gemma4QKVEnvironmentFlag(
    "DARKBLOOM_FULL_QK_COTILED",
    default: true
)

private let gemma4VerifyCoTiledFullQKBits = gemma4QKVEnvironmentFlag(
    "DARKBLOOM_VERIFY_FULL_QK_COTILED_BITS",
    default: false
)

/// One transform-authored co-tiled attention decode payload
/// (`<layer>.self_attn.qkv_cotiled_payload` / `..._qk_cotiled_payload`),
/// already validated byte-for-byte against the layer's separate weight and
/// packed-metadata tensors by `validateGemma4CoTiledAttentionPayloadBytes`
/// before arrays were loaded. The layout is documented in
/// `Gemma4CoTiledAttentionPayloadValidation.swift` and in the transform's
/// `CoTiledAttentionPayloadCoding`.
struct Gemma4CoTiledAttentionPayload: @unchecked Sendable {
    let kind: Gemma4CoTiledAttentionPayloadKind
    let words: MLXArray
    let qBits: Int
    let kBits: Int
    let vBits: Int?
}

func supportsGemma4CoTiledSlidingQKVPayload(
    _ payload: Gemma4CoTiledAttentionPayload,
    qMetadata: IndexedAffineMetadata,
    kMetadata: IndexedAffineMetadata,
    vMetadata: IndexedAffineMetadata
) -> Bool {
    guard payload.kind == .slidingQKV,
          let vBits = payload.vBits,
          let qMaximumLUTCount = Gemma4PackedProjectionMetadataLayout
              .maximumLUTCount(indexBits: payload.qBits),
          let kMaximumLUTCount = Gemma4PackedProjectionMetadataLayout
              .maximumLUTCount(indexBits: payload.kBits),
          let vMaximumLUTCount = Gemma4PackedProjectionMetadataLayout
              .maximumLUTCount(indexBits: vBits),
          let wordsPerThreadgroup = Gemma4CoTiledAttentionPayloadLayout
              .wordsPerThreadgroup(
                  slotIndexBits: Gemma4CoTiledAttentionPayloadLayout
                      .slidingSlotIndexBits(
                          qBits: payload.qBits,
                          kBits: payload.kBits,
                          vBits: vBits
                      )
              )
    else {
        return false
    }
    return payload.words.dtype == .uint32
        && payload.words.shape == [1_024, wordsPerThreadgroup]
        && (1...qMaximumLUTCount).contains(qMetadata.lut.size)
        && (1...kMaximumLUTCount).contains(kMetadata.lut.size)
        && (1...vMaximumLUTCount).contains(vMetadata.lut.size)
}

func supportsGemma4CoTiledFullQKPayload(
    _ payload: Gemma4CoTiledAttentionPayload,
    qMetadata: IndexedAffineMetadata,
    kMetadata: IndexedAffineMetadata
) -> Bool {
    guard payload.kind == .fullQK,
          payload.vBits == nil,
          let qMaximumLUTCount = Gemma4PackedProjectionMetadataLayout
              .maximumLUTCount(indexBits: payload.qBits),
          let kMaximumLUTCount = Gemma4PackedProjectionMetadataLayout
              .maximumLUTCount(indexBits: payload.kBits),
          let wordsPerThreadgroup = Gemma4CoTiledAttentionPayloadLayout
              .wordsPerThreadgroup(
                  slotIndexBits: Gemma4CoTiledAttentionPayloadLayout
                      .fullSlotIndexBits(
                          qBits: payload.qBits,
                          kBits: payload.kBits
                      )
              )
    else {
        return false
    }
    return payload.words.dtype == .uint32
        && payload.words.shape == [512, wordsPerThreadgroup]
        && (1...qMaximumLUTCount).contains(qMetadata.lut.size)
        && (1...kMaximumLUTCount).contains(kMetadata.lut.size)
}

private func gemma4VerifyPackedQKVRawBF16(
    _ candidate: MLXArray,
    reference: MLXArray,
    label: String
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
            "\(label) raw BF16 mismatch vs U16 indexed qmv at index "
                + "\(mismatch?.offset ?? -1): "
                + "candidate=\(mismatch?.element.0 ?? 0), "
                + "reference=\(mismatch?.element.1 ?? 0)"
        )
    }
}

/// Per-projection metadata decode constants for one packed kernel variant.
/// `pairStride`/`bytesPerRow` come from the fixed12/13 row layout; the
/// weight traversal, accumulation order, and reductions are identical to the
/// U16 kernels, so outputs are bit-identical by construction once the packed
/// bytes reconstruct the same LUT indexes.
private func gemma4PackedQKVProjectionMetadataConstants(
    prefix: String,
    indexBits: Int
) -> String {
    guard let bytesPerRow = Gemma4PackedProjectionMetadataLayout.bytesPerRow(
        groupsPerRow: 84,
        indexBits: indexBits
    ) else {
        preconditionFailure("unsupported packed QKV metadata width \(indexBits)")
    }
    return """
        constexpr int k\(prefix)PackedBytesPerRow = \(bytesPerRow);
        constexpr int k\(prefix)PairStride = \(indexBits);
        constexpr bool k\(prefix)HasTopBits = \(indexBits == 13 ? "true" : "false");
        """
}

/// Shared metadata-pair/tail decode loop for the packed QKV/QK kernel
/// bodies. The traversal mirrors the promoted packed12 gate/up kernel: one
/// pair tile carries the eight indexes consumed by two consecutive 256-input
/// blocks, and each row's contributions accumulate in the exact block order
/// (2p, then 2p+1, then the tail block 20) of the U16 kernels.
private func gemma4PackedQKVDecodeLoops(functionPrefix: String) -> String {
    """
        float result[kRowsPerSIMD] = {0};
        for (int block_pair = 0; block_pair < kPairCount; ++block_pair) {
            float even_values[8];
            const float even_input_sum =
                \(functionPrefix)_load_values(input, even_values);
            uint odd_luts[kRowsPerSIMD];

            // One pair tile encodes the eight metadata indexes consumed by
            // two consecutive 256-input blocks. For this lane's group, the
            // three byte planes (plus the fixed13 top-bit byte) reconstruct
            // both indexes without variable bit offsets, integer
            // division/modulo, or word-crossing control.
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const device uchar* row_tile =
                    row_packed_indices + row * packed_bytes_per_row;
                const uint even_low = row_tile[lane_group];
                const uint middle = row_tile[4 + lane_group];
                const uint odd_high = row_tile[8 + lane_group];
                uint even_index = even_low | ((middle & 0x0f) << 8);
                uint odd_index = (middle >> 4) | (odd_high << 4);
                if (has_top_bits) {
                    const uint top = row_tile[12];
                    even_index |= ((top >> lane_group) & 1) << 12;
                    odd_index |= ((top >> (4 + lane_group)) & 1) << 12;
                }
                const uint even_pair = lut[even_index];
                odd_luts[row] = lut[odd_index];
                result[row] += \(functionPrefix)_qdot_4bit(
                    row_weight,
                    even_values,
                    \(functionPrefix)_pair_scale(even_pair),
                    \(functionPrefix)_pair_bias(even_pair),
                    even_input_sum);
            }

            weight_bytes += 128;
            input += 256;
            float odd_values[8];
            const float odd_input_sum =
                \(functionPrefix)_load_values(input, odd_values);
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const uint odd_pair = odd_luts[row];
                result[row] += \(functionPrefix)_qdot_4bit(
                    row_weight,
                    odd_values,
                    \(functionPrefix)_pair_scale(odd_pair),
                    \(functionPrefix)_pair_bias(odd_pair),
                    odd_input_sum);
            }

            weight_bytes += 128;
            input += 256;
            row_packed_indices += pair_stride;
        }

        // The 21st block is the unpaired tail: two lane-major bytes per
        // index (natively wide enough for both fixed12 and fixed13).
        float tail_values[8];
        const float tail_input_sum =
            \(functionPrefix)_load_values(input, tail_values);
        const uint tail_lane_offset = lane_group << 1;
        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            const device uchar* row_weight =
                weight_bytes + row * kWeightBytesPerRow;
            const device uchar* row_tail =
                row_packed_indices + row * packed_bytes_per_row;
            const uint low = row_tail[tail_lane_offset];
            const uint high = row_tail[tail_lane_offset + 1];
            const uint metadata_index = low | (high << 8);
            const uint pair = lut[metadata_index];
            result[row] += \(functionPrefix)_qdot_4bit(
                row_weight,
                tail_values,
                \(functionPrefix)_pair_scale(pair),
                \(functionPrefix)_pair_bias(pair),
                tail_input_sum);
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                output[output_row + row] = static_cast<bfloat>(result[row]);
            }
        }
        """
}

private func gemma4PackedQKVHelperHeader(functionPrefix: String) -> String {
    """
    using namespace metal;

    inline float \(functionPrefix)_pair_scale(uint pair) {
        return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
    }

    inline float \(functionPrefix)_pair_bias(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair >> 16)));
    }

    inline float \(functionPrefix)_load_values(
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

    inline float \(functionPrefix)_qdot_4bit(
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
    """
}

private func gemma4PackedSlidingQKVBody(
    qBits: Int,
    kBits: Int,
    vBits: Int
) -> String {
    """
        constexpr int kWeightBytesPerRow = 2688;
        constexpr int kRowsPerSIMD = 4;
        constexpr int kPairCount = 10;
        \(gemma4PackedQKVProjectionMetadataConstants(prefix: "Q", indexBits: qBits))
        \(gemma4PackedQKVProjectionMetadataConstants(prefix: "K", indexBits: kBits))
        \(gemma4PackedQKVProjectionMetadataConstants(prefix: "V", indexBits: vBits))

        const int projection = simdgroup_index_in_threadgroup;
        const bool is_q = projection < 2;
        const bool is_k = projection == 2;
        const int output_row = is_q
            ? threadgroup_position_in_grid.y * 8 + projection * kRowsPerSIMD
            : threadgroup_position_in_grid.y * kRowsPerSIMD;

        const device uint* weight = is_q
            ? q_weight
            : (is_k ? k_weight : v_weight);
        const device uchar* packed_indices = is_q
            ? q_packed_indices
            : (is_k ? k_packed_indices : v_packed_indices);
        const device uint* lut = is_q
            ? q_lut
            : (is_k ? k_lut : v_lut);
        device bfloat* output = is_q
            ? q_output
            : (is_k ? k_output : v_output);
        const int packed_bytes_per_row = is_q
            ? kQPackedBytesPerRow
            : (is_k ? kKPackedBytesPerRow : kVPackedBytesPerRow);
        const int pair_stride = is_q
            ? kQPairStride
            : (is_k ? kKPairStride : kVPairStride);
        const bool has_top_bits = is_q
            ? kQHasTopBits
            : (is_k ? kKHasTopBits : kVHasTopBits);

        const device uchar* weight_bytes =
            reinterpret_cast<const device uchar*>(weight)
            + output_row * kWeightBytesPerRow
            + thread_index_in_simdgroup * 4;
        const device uchar* row_packed_indices =
            packed_indices + output_row * packed_bytes_per_row;
        const device bfloat* input = x + thread_index_in_simdgroup * 8;
        const uint lane_group = thread_index_in_simdgroup >> 3;

        \(gemma4PackedQKVDecodeLoops(functionPrefix: "gemma4_packed_qkv"))
        """
}

private func gemma4PackedFullQKBody(qBits: Int, kBits: Int) -> String {
    """
        constexpr int kWeightBytesPerRow = 2688;
        constexpr int kRowsPerSIMD = 4;
        constexpr int kPairCount = 10;
        \(gemma4PackedQKVProjectionMetadataConstants(prefix: "Q", indexBits: qBits))
        \(gemma4PackedQKVProjectionMetadataConstants(prefix: "K", indexBits: kBits))

        const int projection = simdgroup_index_in_threadgroup;
        const bool is_q = projection < 8;
        const int output_row = is_q
            ? threadgroup_position_in_grid.y * 32 + projection * kRowsPerSIMD
            : threadgroup_position_in_grid.y * kRowsPerSIMD;

        const device uint* weight = is_q ? q_weight : k_weight;
        const device uchar* packed_indices = is_q
            ? q_packed_indices
            : k_packed_indices;
        const device uint* lut = is_q ? q_lut : k_lut;
        device bfloat* output = is_q ? q_output : k_output;
        const int packed_bytes_per_row = is_q
            ? kQPackedBytesPerRow
            : kKPackedBytesPerRow;
        const int pair_stride = is_q ? kQPairStride : kKPairStride;
        const bool has_top_bits = is_q ? kQHasTopBits : kKHasTopBits;

        const device uchar* weight_bytes =
            reinterpret_cast<const device uchar*>(weight)
            + output_row * kWeightBytesPerRow
            + thread_index_in_simdgroup * 4;
        const device uchar* row_packed_indices =
            packed_indices + output_row * packed_bytes_per_row;
        const device bfloat* input = x + thread_index_in_simdgroup * 8;
        const uint lane_group = thread_index_in_simdgroup >> 3;

        \(gemma4PackedQKVDecodeLoops(functionPrefix: "gemma4_packed_full_qk"))
        """
}

struct Gemma4PackedSlidingQKVFormats: Hashable {
    let qBits: Int
    let kBits: Int
    let vBits: Int
}

struct Gemma4PackedFullQKFormats: Hashable {
    let qBits: Int
    let kBits: Int
}

/// Every per-projection width combination is pre-registered; only the
/// variants the loaded checkpoint actually selects are compiled by Metal
/// (pipeline creation is lazy on first dispatch).
private let gemma4PackedIndexedSlidingQKVKernels:
    [Gemma4PackedSlidingQKVFormats: MLXFast.MLXFastKernel] = {
        var kernels = [Gemma4PackedSlidingQKVFormats: MLXFast.MLXFastKernel]()
        for qBits in [12, 13] {
            for kBits in [12, 13] {
                for vBits in [12, 13] {
                    let formats = Gemma4PackedSlidingQKVFormats(
                        qBits: qBits,
                        kBits: kBits,
                        vBits: vBits
                    )
                    kernels[formats] = MLXFast.metalKernel(
                        name: "gemma4_packed_indexed_sliding_qkv_qmv_5376"
                            + "_q\(qBits)_k\(kBits)_v\(vBits)_v1",
                        inputNames: [
                            "q_weight", "q_packed_indices", "q_lut",
                            "k_weight", "k_packed_indices", "k_lut",
                            "v_weight", "v_packed_indices", "v_lut", "x",
                        ],
                        outputNames: ["q_output", "k_output", "v_output"],
                        source: gemma4PackedSlidingQKVBody(
                            qBits: qBits,
                            kBits: kBits,
                            vBits: vBits
                        ),
                        header: gemma4PackedQKVHelperHeader(
                            functionPrefix: "gemma4_packed_qkv"
                        ),
                        ensureRowContiguous: true
                    )
                }
            }
        }
        return kernels
    }()

private let gemma4PackedIndexedFullQKKernels:
    [Gemma4PackedFullQKFormats: MLXFast.MLXFastKernel] = {
        var kernels = [Gemma4PackedFullQKFormats: MLXFast.MLXFastKernel]()
        for qBits in [12, 13] {
            for kBits in [12, 13] {
                let formats = Gemma4PackedFullQKFormats(qBits: qBits, kBits: kBits)
                kernels[formats] = MLXFast.metalKernel(
                    name: "gemma4_packed_indexed_full_qk_qmv_5376"
                        + "_q\(qBits)_k\(kBits)_v1",
                    inputNames: [
                        "q_weight", "q_packed_indices", "q_lut",
                        "k_weight", "k_packed_indices", "k_lut", "x",
                    ],
                    outputNames: ["q_output", "k_output"],
                    source: gemma4PackedFullQKBody(qBits: qBits, kBits: kBits),
                    header: gemma4PackedQKVHelperHeader(
                        functionPrefix: "gemma4_packed_full_qk"
                    ),
                    ensureRowContiguous: true
                )
            }
        }
        return kernels
    }()

/// Shared decode loops for the co-tiled QKV/QK kernel bodies. The traversal
/// consumes one forward-contiguous per-threadgroup tile stream: ten pair
/// tiles of `[even-block weights | odd-block weights | packed metadata]`
/// followed by one tail tile of `[block-20 weights | lane-major tail
/// metadata]` (layout in `Gemma4CoTiledAttentionPayloadValidation.swift`).
/// The arithmetic, LUT index reconstruction, per-row accumulation order
/// (block 2p, then 2p+1, then the tail block 20), reductions, and BF16
/// output boundary are identical to the packed fixed12/13 kernels above —
/// only the addresses change, so outputs are bit-identical by construction
/// once the payload reproduces the same weight words and metadata bytes.
private func gemma4CoTiledQKVDecodeLoops(functionPrefix: String) -> String {
    """
        float result[kRowsPerSIMD] = {0};
        for (int block_pair = 0; block_pair < kPairCount; ++block_pair) {
            float even_values[8];
            const float even_input_sum =
                \(functionPrefix)_load_values(input, even_values);
            uint odd_luts[kRowsPerSIMD];

            const device uint* even_weight_words = tile_words
                + projection * kSlotWeightWords + lane;
            const device uint* odd_weight_words = tile_words
                + kBlockWeightWords + projection * kSlotWeightWords + lane;
            const device uchar* metadata_bytes =
                reinterpret_cast<const device uchar*>(
                    tile_words + kPairWeightWords)
                + slot_meta_offset;

            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_tile =
                    metadata_bytes + row * pair_stride;
                const uint even_low = row_tile[lane_group];
                const uint middle = row_tile[4 + lane_group];
                const uint odd_high = row_tile[8 + lane_group];
                uint even_index = even_low | ((middle & 0x0f) << 8);
                uint odd_index = (middle >> 4) | (odd_high << 4);
                if (has_top_bits) {
                    const uint top = row_tile[12];
                    even_index |= ((top >> lane_group) & 1) << 12;
                    odd_index |= ((top >> (4 + lane_group)) & 1) << 12;
                }
                const uint even_pair = lut[even_index];
                odd_luts[row] = lut[odd_index];
                const device uchar* row_weight =
                    reinterpret_cast<const device uchar*>(
                        even_weight_words + row * kSIMDSize);
                result[row] += \(functionPrefix)_qdot_4bit(
                    row_weight,
                    even_values,
                    \(functionPrefix)_pair_scale(even_pair),
                    \(functionPrefix)_pair_bias(even_pair),
                    even_input_sum);
            }

            input += 256;
            float odd_values[8];
            const float odd_input_sum =
                \(functionPrefix)_load_values(input, odd_values);
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const uint odd_pair = odd_luts[row];
                const device uchar* row_weight =
                    reinterpret_cast<const device uchar*>(
                        odd_weight_words + row * kSIMDSize);
                result[row] += \(functionPrefix)_qdot_4bit(
                    row_weight,
                    odd_values,
                    \(functionPrefix)_pair_scale(odd_pair),
                    \(functionPrefix)_pair_bias(odd_pair),
                    odd_input_sum);
            }

            input += 256;
            tile_words += kWordsPerPair;
        }

        // The 21st block is the unpaired tail tile: block-20 weights for
        // every slot, then two lane-major bytes per index (32 bytes per
        // slot, no padding bytes).
        float tail_values[8];
        const float tail_input_sum =
            \(functionPrefix)_load_values(input, tail_values);
        const device uint* tail_weight_words = tile_words
            + projection * kSlotWeightWords + lane;
        const device uchar* tail_metadata =
            reinterpret_cast<const device uchar*>(
                tile_words + kBlockWeightWords)
            + projection * 32;
        const uint tail_lane_offset = lane_group << 1;
        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            const device uchar* row_tail = tail_metadata + row * 8;
            const uint low = row_tail[tail_lane_offset];
            const uint high = row_tail[tail_lane_offset + 1];
            const uint metadata_index = low | (high << 8);
            const uint pair = lut[metadata_index];
            const device uchar* row_weight =
                reinterpret_cast<const device uchar*>(
                    tail_weight_words + row * kSIMDSize);
            result[row] += \(functionPrefix)_qdot_4bit(
                row_weight,
                tail_values,
                \(functionPrefix)_pair_scale(pair),
                \(functionPrefix)_pair_bias(pair),
                tail_input_sum);
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                output[output_row + row] = static_cast<bfloat>(result[row]);
            }
        }
        """
}

private func gemma4CoTiledSlidingQKVBody(
    qBits: Int,
    kBits: Int,
    vBits: Int
) -> String {
    """
        constexpr int kRowsPerSIMD = 4;
        constexpr int kSIMDSize = 32;
        constexpr int kSlotWeightWords = kRowsPerSIMD * kSIMDSize;
        constexpr int kSlotsPerThreadgroup = 4;
        constexpr int kBlockWeightWords =
            kSlotsPerThreadgroup * kSlotWeightWords;
        constexpr int kPairWeightWords = 2 * kBlockWeightWords;
        constexpr int kPairCount = 10;
        constexpr int kQPairStride = \(qBits);
        constexpr int kKPairStride = \(kBits);
        constexpr int kVPairStride = \(vBits);
        constexpr bool kQHasTopBits = \(qBits == 13 ? "true" : "false");
        constexpr bool kKHasTopBits = \(kBits == 13 ? "true" : "false");
        constexpr bool kVHasTopBits = \(vBits == 13 ? "true" : "false");
        constexpr int kPairMetadataBytes =
            8 * kQPairStride + 4 * kKPairStride + 4 * kVPairStride;
        constexpr int kWordsPerPair =
            kPairWeightWords + kPairMetadataBytes / 4;
        constexpr int kWordsPerTail =
            kBlockWeightWords + 8 * kSlotsPerThreadgroup;
        constexpr int kWordsPerThreadgroup =
            kPairCount * kWordsPerPair + kWordsPerTail;

        const int projection = simdgroup_index_in_threadgroup;
        const bool is_q = projection < 2;
        const bool is_k = projection == 2;
        const int output_row = is_q
            ? threadgroup_position_in_grid.y * 8 + projection * kRowsPerSIMD
            : threadgroup_position_in_grid.y * kRowsPerSIMD;

        const device uint* lut = is_q
            ? q_lut
            : (is_k ? k_lut : v_lut);
        device bfloat* output = is_q
            ? q_output
            : (is_k ? k_output : v_output);
        const int pair_stride = is_q
            ? kQPairStride
            : (is_k ? kKPairStride : kVPairStride);
        const bool has_top_bits = is_q
            ? kQHasTopBits
            : (is_k ? kKHasTopBits : kVHasTopBits);
        const int slot_meta_offset = is_q
            ? projection * 4 * kQPairStride
            : (is_k
                ? 8 * kQPairStride
                : 8 * kQPairStride + 4 * kKPairStride);

        const uint lane = thread_index_in_simdgroup;
        const uint lane_group = lane >> 3;
        const device uint* tile_words = cotiled_payload
            + threadgroup_position_in_grid.y * kWordsPerThreadgroup;
        const device bfloat* input = x + lane * 8;

        \(gemma4CoTiledQKVDecodeLoops(functionPrefix: "gemma4_cotiled_qkv"))
        """
}

private func gemma4CoTiledFullQKBody(qBits: Int, kBits: Int) -> String {
    """
        constexpr int kRowsPerSIMD = 4;
        constexpr int kSIMDSize = 32;
        constexpr int kSlotWeightWords = kRowsPerSIMD * kSIMDSize;
        constexpr int kSlotsPerThreadgroup = 9;
        constexpr int kBlockWeightWords =
            kSlotsPerThreadgroup * kSlotWeightWords;
        constexpr int kPairWeightWords = 2 * kBlockWeightWords;
        constexpr int kPairCount = 10;
        constexpr int kQPairStride = \(qBits);
        constexpr int kKPairStride = \(kBits);
        constexpr bool kQHasTopBits = \(qBits == 13 ? "true" : "false");
        constexpr bool kKHasTopBits = \(kBits == 13 ? "true" : "false");
        constexpr int kPairMetadataBytes =
            32 * kQPairStride + 4 * kKPairStride;
        constexpr int kWordsPerPair =
            kPairWeightWords + kPairMetadataBytes / 4;
        constexpr int kWordsPerTail =
            kBlockWeightWords + 8 * kSlotsPerThreadgroup;
        constexpr int kWordsPerThreadgroup =
            kPairCount * kWordsPerPair + kWordsPerTail;

        const int projection = simdgroup_index_in_threadgroup;
        const bool is_q = projection < 8;
        const int output_row = is_q
            ? threadgroup_position_in_grid.y * 32 + projection * kRowsPerSIMD
            : threadgroup_position_in_grid.y * kRowsPerSIMD;

        const device uint* lut = is_q ? q_lut : k_lut;
        device bfloat* output = is_q ? q_output : k_output;
        const int pair_stride = is_q ? kQPairStride : kKPairStride;
        const bool has_top_bits = is_q ? kQHasTopBits : kKHasTopBits;
        const int slot_meta_offset = is_q
            ? projection * 4 * kQPairStride
            : 32 * kQPairStride;

        const uint lane = thread_index_in_simdgroup;
        const uint lane_group = lane >> 3;
        const device uint* tile_words = cotiled_payload
            + threadgroup_position_in_grid.y * kWordsPerThreadgroup;
        const device bfloat* input = x + lane * 8;

        \(gemma4CoTiledQKVDecodeLoops(functionPrefix: "gemma4_cotiled_full_qk"))
        """
}

/// Every per-projection width combination is pre-registered; only the
/// variants the loaded checkpoint actually selects are compiled by Metal
/// (pipeline creation is lazy on first dispatch).
private let gemma4CoTiledIndexedSlidingQKVKernels:
    [Gemma4PackedSlidingQKVFormats: MLXFast.MLXFastKernel] = {
        var kernels = [Gemma4PackedSlidingQKVFormats: MLXFast.MLXFastKernel]()
        for qBits in [12, 13] {
            for kBits in [12, 13] {
                for vBits in [12, 13] {
                    let formats = Gemma4PackedSlidingQKVFormats(
                        qBits: qBits,
                        kBits: kBits,
                        vBits: vBits
                    )
                    kernels[formats] = MLXFast.metalKernel(
                        name: "gemma4_cotiled_indexed_sliding_qkv_qmv_5376"
                            + "_q\(qBits)_k\(kBits)_v\(vBits)_v1",
                        inputNames: [
                            "cotiled_payload", "q_lut", "k_lut", "v_lut", "x",
                        ],
                        outputNames: ["q_output", "k_output", "v_output"],
                        source: gemma4CoTiledSlidingQKVBody(
                            qBits: qBits,
                            kBits: kBits,
                            vBits: vBits
                        ),
                        header: gemma4PackedQKVHelperHeader(
                            functionPrefix: "gemma4_cotiled_qkv"
                        ),
                        ensureRowContiguous: true
                    )
                }
            }
        }
        return kernels
    }()

private let gemma4CoTiledIndexedFullQKKernels:
    [Gemma4PackedFullQKFormats: MLXFast.MLXFastKernel] = {
        var kernels = [Gemma4PackedFullQKFormats: MLXFast.MLXFastKernel]()
        for qBits in [12, 13] {
            for kBits in [12, 13] {
                let formats = Gemma4PackedFullQKFormats(qBits: qBits, kBits: kBits)
                kernels[formats] = MLXFast.metalKernel(
                    name: "gemma4_cotiled_indexed_full_qk_qmv_5376"
                        + "_q\(qBits)_k\(kBits)_v1",
                    inputNames: [
                        "cotiled_payload", "q_lut", "k_lut", "x",
                    ],
                    outputNames: ["q_output", "k_output"],
                    source: gemma4CoTiledFullQKBody(qBits: qBits, kBits: kBits),
                    header: gemma4PackedQKVHelperHeader(
                        functionPrefix: "gemma4_cotiled_full_qk"
                    ),
                    ensureRowContiguous: true
                )
            }
        }
        return kernels
    }()

private let gemma4IndexedSlidingQKV = MLXFast.metalKernel(
    name: "gemma4_indexed_sliding_qkv_qmv_5376_v1",
    inputNames: [
        "q_weight", "q_indices", "q_lut",
        "k_weight", "k_indices", "k_lut",
        "v_weight", "v_indices", "v_lut", "x",
    ],
    outputNames: ["q_output", "k_output", "v_output"],
    source: """
        constexpr int kInputWidth = 5376;
        constexpr int kGroupsPerRow = 84;
        constexpr int kWeightBytesPerRow = 2688;
        constexpr int kRowsPerSIMD = 4;

        const int projection = simdgroup_index_in_threadgroup;
        const bool is_q = projection < 2;
        const bool is_k = projection == 2;
        const int output_row = is_q
            ? threadgroup_position_in_grid.y * 8 + projection * kRowsPerSIMD
            : threadgroup_position_in_grid.y * kRowsPerSIMD;

        const device uint* weight = is_q
            ? q_weight
            : (is_k ? k_weight : v_weight);
        const device ushort* indices = is_q
            ? q_indices
            : (is_k ? k_indices : v_indices);
        const device uint* lut = is_q
            ? q_lut
            : (is_k ? k_lut : v_lut);
        device bfloat* output = is_q
            ? q_output
            : (is_k ? k_output : v_output);

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
            const float input_sum = gemma4_qkv_load_values(input, values);

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const ushort metadata_index = row_indices[row * kGroupsPerRow];
                const uint pair = lut[metadata_index];
                result[row] += gemma4_qkv_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_qkv_pair_scale(pair),
                    gemma4_qkv_pair_bias(pair),
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

        inline float gemma4_qkv_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float gemma4_qkv_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float gemma4_qkv_load_values(
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

        inline float gemma4_qkv_qdot_4bit(
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

private func supportsSlidingProjection(
    _ projection: FastQuantizedProjection,
    metadata: IndexedAffineMetadata,
    outputWidth: Int
) -> Bool {
    guard let biases = projection.biases else { return false }
    let metadataShape = [outputWidth, 84]
    return projection.groupSize == 64
        && projection.bits == 4
        && projection.weight.dtype == .uint32
        && projection.weight.shape == [outputWidth, 672]
        && projection.scales.dtype == .bfloat16
        && projection.scales.shape == metadataShape
        && biases.dtype == .bfloat16
        && biases.shape == metadataShape
        && metadata.indices.dtype == .uint16
        && metadata.indices.shape == metadataShape
        && metadata.lut.dtype == .uint32
        && metadata.lut.ndim == 1
        && (1...65_536).contains(metadata.lut.size)
}

func supportsGemma4FusedSlidingQKV(
    q: FastQuantizedProjection,
    k: FastQuantizedProjection,
    v: FastQuantizedProjection,
    qMetadata: IndexedAffineMetadata,
    kMetadata: IndexedAffineMetadata,
    vMetadata: IndexedAffineMetadata
) -> Bool {
    supportsSlidingProjection(q, metadata: qMetadata, outputWidth: 8_192)
        && supportsSlidingProjection(k, metadata: kMetadata, outputWidth: 4_096)
        && supportsSlidingProjection(v, metadata: vMetadata, outputWidth: 4_096)
}

struct FusedSlidingQKVProjection: @unchecked Sendable {
    let q: FastQuantizedProjection
    let k: FastQuantizedProjection
    let v: FastQuantizedProjection
    let qMetadata: IndexedAffineMetadata
    let kMetadata: IndexedAffineMetadata
    let vMetadata: IndexedAffineMetadata
    private let packed: PackedMetadataTriple?
    private let usePackedIndices: Bool
    private let verifyPackedBits: Bool
    private let coTiled: CoTiledPayload?
    private let useCoTiled: Bool
    private let verifyCoTiledBits: Bool

    private struct PackedMetadataTriple {
        let q: Gemma4PackedQKVIndexMetadata
        let k: Gemma4PackedQKVIndexMetadata
        let v: Gemma4PackedQKVIndexMetadata

        var formats: Gemma4PackedSlidingQKVFormats {
            Gemma4PackedSlidingQKVFormats(
                qBits: q.indexBits,
                kBits: k.indexBits,
                vBits: v.indexBits
            )
        }
    }

    private struct CoTiledPayload {
        let words: MLXArray
        let formats: Gemma4PackedSlidingQKVFormats
    }

    init(
        q: FastQuantizedProjection,
        k: FastQuantizedProjection,
        v: FastQuantizedProjection,
        qMetadata: IndexedAffineMetadata,
        kMetadata: IndexedAffineMetadata,
        vMetadata: IndexedAffineMetadata,
        qPackedMetadata: Gemma4PackedQKVIndexMetadata? = nil,
        kPackedMetadata: Gemma4PackedQKVIndexMetadata? = nil,
        vPackedMetadata: Gemma4PackedQKVIndexMetadata? = nil,
        coTiledPayload: Gemma4CoTiledAttentionPayload? = nil
    ) {
        precondition(supportsGemma4FusedSlidingQKV(
            q: q,
            k: k,
            v: v,
            qMetadata: qMetadata,
            kMetadata: kMetadata,
            vMetadata: vMetadata
        ))
        self.q = q
        self.k = k
        self.v = v
        self.qMetadata = qMetadata
        self.kMetadata = kMetadata
        self.vMetadata = vMetadata
        self.usePackedIndices = gemma4PackedSlidingQKVIndicesEnabled
        self.verifyPackedBits = gemma4VerifyPackedSlidingQKVBits
        self.useCoTiled = gemma4CoTiledSlidingQKVEnabled
        self.verifyCoTiledBits = gemma4VerifyCoTiledSlidingQKVBits
        // The packed kernels require all three streams: a partially packed
        // layer keeps the promoted U16 kernel (fail-open, never mixed).
        if gemma4PackedSlidingQKVIndicesEnabled
            || gemma4VerifyPackedSlidingQKVBits,
            let qPackedMetadata,
            let kPackedMetadata,
            let vPackedMetadata,
            supportsGemma4PackedQKVIndexMetadata(
                qPackedMetadata, metadata: qMetadata, outputWidth: 8_192),
            supportsGemma4PackedQKVIndexMetadata(
                kPackedMetadata, metadata: kMetadata, outputWidth: 4_096),
            supportsGemma4PackedQKVIndexMetadata(
                vPackedMetadata, metadata: vMetadata, outputWidth: 4_096)
        {
            self.packed = PackedMetadataTriple(
                q: qPackedMetadata,
                k: kPackedMetadata,
                v: vPackedMetadata
            )
        } else {
            self.packed = nil
        }
        // The co-tiled kernel engages only when the transform authored the
        // payload and its shape matches the loaded metadata formats exactly
        // (fail-open to the packed/U16 selection below, never mixed).
        if gemma4CoTiledSlidingQKVEnabled || gemma4VerifyCoTiledSlidingQKVBits,
            let coTiledPayload,
            let vBits = coTiledPayload.vBits,
            supportsGemma4CoTiledSlidingQKVPayload(
                coTiledPayload,
                qMetadata: qMetadata,
                kMetadata: kMetadata,
                vMetadata: vMetadata
            )
        {
            self.coTiled = CoTiledPayload(
                words: coTiledPayload.words,
                formats: Gemma4PackedSlidingQKVFormats(
                    qBits: coTiledPayload.qBits,
                    kBits: coTiledPayload.kBits,
                    vBits: vBits
                )
            )
        } else {
            self.coTiled = nil
        }
    }

    func callAsFunction(_ input: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 5_376])
        guard let coTiled else {
            return referenceOutputs(input)
        }
        let candidate = coTiledOutputs(input, coTiled: coTiled)
        if verifyCoTiledBits {
            let reference = referenceOutputs(input)
            gemma4VerifyPackedQKVRawBF16(
                candidate.0,
                reference: reference.0,
                label: "co-tiled sliding qkv q_proj"
            )
            gemma4VerifyPackedQKVRawBF16(
                candidate.1,
                reference: reference.1,
                label: "co-tiled sliding qkv k_proj"
            )
            gemma4VerifyPackedQKVRawBF16(
                candidate.2,
                reference: reference.2,
                label: "co-tiled sliding qkv v_proj"
            )
            if !useCoTiled {
                // Verifier-only mode must not change the selected output.
                return reference
            }
        }
        return candidate
    }

    /// The pre-co-tile selection (packed fixed12/13 when authored, else the
    /// promoted U16 kernel), including its own verify mode. This is the
    /// fail-open path and the co-tile verifier's reference.
    private func referenceOutputs(
        _ input: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray) {
        guard let packed else {
            return u16Outputs(input)
        }
        let candidate = packedOutputs(input, packed: packed)
        if verifyPackedBits {
            let reference = u16Outputs(input)
            gemma4VerifyPackedQKVRawBF16(
                candidate.0,
                reference: reference.0,
                label: "packed fixed\(packed.q.indexBits) sliding qkv q_proj"
            )
            gemma4VerifyPackedQKVRawBF16(
                candidate.1,
                reference: reference.1,
                label: "packed fixed\(packed.k.indexBits) sliding qkv k_proj"
            )
            gemma4VerifyPackedQKVRawBF16(
                candidate.2,
                reference: reference.2,
                label: "packed fixed\(packed.v.indexBits) sliding qkv v_proj"
            )
            if !usePackedIndices {
                // Verifier-only mode must not change the selected output.
                return reference
            }
        }
        return candidate
    }

    private func coTiledOutputs(
        _ input: MLXArray,
        coTiled: CoTiledPayload
    ) -> (MLXArray, MLXArray, MLXArray) {
        guard let kernel = gemma4CoTiledIndexedSlidingQKVKernels[coTiled.formats]
        else {
            preconditionFailure("missing co-tiled sliding QKV kernel variant")
        }
        let outputs = kernel(
            [
                coTiled.words,
                qMetadata.lut, kMetadata.lut, vMetadata.lut, input,
            ],
            grid: (32, 4_096, 1),
            threadGroup: (32, 4, 1),
            outputShapes: [
                [1, 1, 8_192],
                [1, 1, 4_096],
                [1, 1, 4_096],
            ],
            outputDTypes: [.bfloat16, .bfloat16, .bfloat16]
        )
        return (outputs[0], outputs[1], outputs[2])
    }

    private func packedOutputs(
        _ input: MLXArray,
        packed: PackedMetadataTriple
    ) -> (MLXArray, MLXArray, MLXArray) {
        guard let kernel = gemma4PackedIndexedSlidingQKVKernels[packed.formats]
        else {
            preconditionFailure("missing packed sliding QKV kernel variant")
        }
        let outputs = kernel(
            [
                q.weight, packed.q.bytes, qMetadata.lut,
                k.weight, packed.k.bytes, kMetadata.lut,
                v.weight, packed.v.bytes, vMetadata.lut, input,
            ],
            grid: (32, 4_096, 1),
            threadGroup: (32, 4, 1),
            outputShapes: [
                [1, 1, 8_192],
                [1, 1, 4_096],
                [1, 1, 4_096],
            ],
            outputDTypes: [.bfloat16, .bfloat16, .bfloat16]
        )
        return (outputs[0], outputs[1], outputs[2])
    }

    private func u16Outputs(
        _ input: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray) {
        let outputs = gemma4IndexedSlidingQKV(
            [
                q.weight, qMetadata.indices, qMetadata.lut,
                k.weight, kMetadata.indices, kMetadata.lut,
                v.weight, vMetadata.indices, vMetadata.lut, input,
            ],
            grid: (32, 4_096, 1),
            threadGroup: (32, 4, 1),
            outputShapes: [
                [1, 1, 8_192],
                [1, 1, 4_096],
                [1, 1, 4_096],
            ],
            outputDTypes: [.bfloat16, .bfloat16, .bfloat16]
        )
        return (outputs[0], outputs[1], outputs[2])
    }
}

private let gemma4IndexedFullQK = MLXFast.metalKernel(
    name: "gemma4_indexed_full_qk_qmv_5376_v1",
    inputNames: [
        "q_weight", "q_indices", "q_lut",
        "k_weight", "k_indices", "k_lut", "x",
    ],
    outputNames: ["q_output", "k_output"],
    source: """
        constexpr int kGroupsPerRow = 84;
        constexpr int kWeightBytesPerRow = 2688;
        constexpr int kRowsPerSIMD = 4;

        const int projection = simdgroup_index_in_threadgroup;
        const bool is_q = projection < 8;
        const int output_row = is_q
            ? threadgroup_position_in_grid.y * 32 + projection * kRowsPerSIMD
            : threadgroup_position_in_grid.y * kRowsPerSIMD;

        const device uint* weight = is_q ? q_weight : k_weight;
        const device ushort* indices = is_q ? q_indices : k_indices;
        const device uint* lut = is_q ? q_lut : k_lut;
        device bfloat* output = is_q ? q_output : k_output;

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
            const float input_sum = gemma4_full_qk_load_values(input, values);

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const ushort metadata_index = row_indices[row * kGroupsPerRow];
                const uint pair = lut[metadata_index];
                result[row] += gemma4_full_qk_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_full_qk_pair_scale(pair),
                    gemma4_full_qk_pair_bias(pair),
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

        inline float gemma4_full_qk_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float gemma4_full_qk_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float gemma4_full_qk_load_values(
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

        inline float gemma4_full_qk_qdot_4bit(
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

func supportsGemma4FusedFullQK(
    q: FastQuantizedProjection,
    k: FastQuantizedProjection,
    qMetadata: IndexedAffineMetadata,
    kMetadata: IndexedAffineMetadata
) -> Bool {
    supportsSlidingProjection(q, metadata: qMetadata, outputWidth: 16_384)
        && supportsSlidingProjection(k, metadata: kMetadata, outputWidth: 2_048)
}

struct FusedFullQKProjection: @unchecked Sendable {
    let q: FastQuantizedProjection
    let k: FastQuantizedProjection
    let qMetadata: IndexedAffineMetadata
    let kMetadata: IndexedAffineMetadata
    private let packed: PackedMetadataPair?
    private let usePackedIndices: Bool
    private let verifyPackedBits: Bool
    private let coTiled: CoTiledPayload?
    private let useCoTiled: Bool
    private let verifyCoTiledBits: Bool

    private struct PackedMetadataPair {
        let q: Gemma4PackedQKVIndexMetadata
        let k: Gemma4PackedQKVIndexMetadata

        var formats: Gemma4PackedFullQKFormats {
            Gemma4PackedFullQKFormats(qBits: q.indexBits, kBits: k.indexBits)
        }
    }

    private struct CoTiledPayload {
        let words: MLXArray
        let formats: Gemma4PackedFullQKFormats
    }

    init(
        q: FastQuantizedProjection,
        k: FastQuantizedProjection,
        qMetadata: IndexedAffineMetadata,
        kMetadata: IndexedAffineMetadata,
        qPackedMetadata: Gemma4PackedQKVIndexMetadata? = nil,
        kPackedMetadata: Gemma4PackedQKVIndexMetadata? = nil,
        coTiledPayload: Gemma4CoTiledAttentionPayload? = nil
    ) {
        precondition(supportsGemma4FusedFullQK(
            q: q,
            k: k,
            qMetadata: qMetadata,
            kMetadata: kMetadata
        ))
        self.q = q
        self.k = k
        self.qMetadata = qMetadata
        self.kMetadata = kMetadata
        self.usePackedIndices = gemma4PackedFullQKIndicesEnabled
        self.verifyPackedBits = gemma4VerifyPackedFullQKBits
        self.useCoTiled = gemma4CoTiledFullQKEnabled
        self.verifyCoTiledBits = gemma4VerifyCoTiledFullQKBits
        // The packed kernel requires both streams: a partially packed layer
        // keeps the promoted U16 kernel (fail-open, never mixed).
        if gemma4PackedFullQKIndicesEnabled || gemma4VerifyPackedFullQKBits,
            let qPackedMetadata,
            let kPackedMetadata,
            supportsGemma4PackedQKVIndexMetadata(
                qPackedMetadata, metadata: qMetadata, outputWidth: 16_384),
            supportsGemma4PackedQKVIndexMetadata(
                kPackedMetadata, metadata: kMetadata, outputWidth: 2_048)
        {
            self.packed = PackedMetadataPair(
                q: qPackedMetadata,
                k: kPackedMetadata
            )
        } else {
            self.packed = nil
        }
        // The co-tiled kernel engages only when the transform authored the
        // payload and its shape matches the loaded metadata formats exactly
        // (fail-open to the packed/U16 selection below, never mixed).
        if gemma4CoTiledFullQKEnabled || gemma4VerifyCoTiledFullQKBits,
            let coTiledPayload,
            supportsGemma4CoTiledFullQKPayload(
                coTiledPayload,
                qMetadata: qMetadata,
                kMetadata: kMetadata
            )
        {
            self.coTiled = CoTiledPayload(
                words: coTiledPayload.words,
                formats: Gemma4PackedFullQKFormats(
                    qBits: coTiledPayload.qBits,
                    kBits: coTiledPayload.kBits
                )
            )
        } else {
            self.coTiled = nil
        }
    }

    func callAsFunction(_ input: MLXArray) -> (MLXArray, MLXArray) {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 5_376])
        guard let coTiled else {
            return referenceOutputs(input)
        }
        let candidate = coTiledOutputs(input, coTiled: coTiled)
        if verifyCoTiledBits {
            let reference = referenceOutputs(input)
            gemma4VerifyPackedQKVRawBF16(
                candidate.0,
                reference: reference.0,
                label: "co-tiled full qk q_proj"
            )
            gemma4VerifyPackedQKVRawBF16(
                candidate.1,
                reference: reference.1,
                label: "co-tiled full qk k_proj"
            )
            if !useCoTiled {
                // Verifier-only mode must not change the selected output.
                return reference
            }
        }
        return candidate
    }

    /// The pre-co-tile selection (packed fixed12/13 when authored, else the
    /// promoted U16 kernel), including its own verify mode. This is the
    /// fail-open path and the co-tile verifier's reference.
    private func referenceOutputs(_ input: MLXArray) -> (MLXArray, MLXArray) {
        guard let packed else {
            return u16Outputs(input)
        }
        let candidate = packedOutputs(input, packed: packed)
        if verifyPackedBits {
            let reference = u16Outputs(input)
            gemma4VerifyPackedQKVRawBF16(
                candidate.0,
                reference: reference.0,
                label: "packed fixed\(packed.q.indexBits) full qk q_proj"
            )
            gemma4VerifyPackedQKVRawBF16(
                candidate.1,
                reference: reference.1,
                label: "packed fixed\(packed.k.indexBits) full qk k_proj"
            )
            if !usePackedIndices {
                // Verifier-only mode must not change the selected output.
                return reference
            }
        }
        return candidate
    }

    private func coTiledOutputs(
        _ input: MLXArray,
        coTiled: CoTiledPayload
    ) -> (MLXArray, MLXArray) {
        guard let kernel = gemma4CoTiledIndexedFullQKKernels[coTiled.formats]
        else {
            preconditionFailure("missing co-tiled full QK kernel variant")
        }
        let outputs = kernel(
            [
                coTiled.words,
                qMetadata.lut, kMetadata.lut, input,
            ],
            grid: (32, 4_608, 1),
            threadGroup: (32, 9, 1),
            outputShapes: [[1, 1, 16_384], [1, 1, 2_048]],
            outputDTypes: [.bfloat16, .bfloat16]
        )
        return (outputs[0], outputs[1])
    }

    private func packedOutputs(
        _ input: MLXArray,
        packed: PackedMetadataPair
    ) -> (MLXArray, MLXArray) {
        guard let kernel = gemma4PackedIndexedFullQKKernels[packed.formats]
        else {
            preconditionFailure("missing packed full QK kernel variant")
        }
        let outputs = kernel(
            [
                q.weight, packed.q.bytes, qMetadata.lut,
                k.weight, packed.k.bytes, kMetadata.lut, input,
            ],
            grid: (32, 4_608, 1),
            threadGroup: (32, 9, 1),
            outputShapes: [[1, 1, 16_384], [1, 1, 2_048]],
            outputDTypes: [.bfloat16, .bfloat16]
        )
        return (outputs[0], outputs[1])
    }

    private func u16Outputs(_ input: MLXArray) -> (MLXArray, MLXArray) {
        let outputs = gemma4IndexedFullQK(
            [
                q.weight, qMetadata.indices, qMetadata.lut,
                k.weight, kMetadata.indices, kMetadata.lut, input,
            ],
            grid: (32, 4_608, 1),
            threadGroup: (32, 9, 1),
            outputShapes: [[1, 1, 16_384], [1, 1, 2_048]],
            outputDTypes: [.bfloat16, .bfloat16]
        )
        return (outputs[0], outputs[1])
    }
}
