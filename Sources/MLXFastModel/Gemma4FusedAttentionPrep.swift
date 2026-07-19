import Foundation
import MLX

private func gemma4PrefillAttentionEnvironmentFlag(
    _ name: String,
    default defaultValue: Bool
) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return defaultValue
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

// Raw BF16 exact against the promoted path at L512 for both attention modes,
// both prefill preparation routes, and on the real 31B public prefill.
private let gemma4DirectPrefillAttentionRMSRoPEDefault = true

private func gemma4RecordDirectPrefillAttentionRMSRoPEVerification(
    kind: String,
    arrays: Int,
    values: Int
) {
    FileHandle.standardError.write(Data(
        (
            "verify_direct_prefill_attention_rms_rope kind=\(kind) "
                + "arrays=\(arrays) values=\(values)\n"
        ).utf8
    ))
}

/// Replaces only the final BF16 staging/redistribution portion of the K/V
/// prefill kernel. The exact promoted per-row RMS reduction remains byte-for-
/// byte unchanged. RoPE-owning threads reload their two source elements and
/// recreate the same explicit BF16 normalization and weight multiplications.
func gemma4DirectCombinedKVPrefillRMSRoPESource(
    _ reference: String
) -> String {
    var source = reference

    let declaration = "threadgroup bfloat normalized_row[kHeadDim];"
    precondition(source.components(separatedBy: declaration).count == 2)
    source = source.replacingOccurrences(of: declaration, with: "")

    let stagedBodyStart = "const device bfloat* row_weight ="
    precondition(source.components(separatedBy: stagedBodyStart).count == 2)
    guard let stagedRange = source.range(of: stagedBodyStart) else {
        preconditionFailure("missing staged combined K/V prefill body")
    }

    let directBody = """
    if (!is_k) {
                    for (uint index = 0; index < kReads; ++index) {
                        const uint dimension =
                            thread_position_in_threadgroup.x * kReads + index;
                        const bfloat normalized = static_cast<bfloat>(
                            input[static_cast<int64_t>(index) * dimension_stride]
                                * inverse_mean[0]);
                        output[dimension] =
                            static_cast<bfloat>(1.0f) * normalized;
                    }
                } else {
                    constexpr uint kPairs = kHeadDim / 2;
                    constexpr uint kThreads = kHeadDim / kReads;
                    const device bfloat* row_input = input
                        - static_cast<int64_t>(
                            thread_position_in_threadgroup.x * kReads
                        ) * dimension_stride;
                    for (uint pair = thread_position_in_threadgroup.x;
                         pair < kPairs;
                         pair += kThreads) {
                        const bfloat normalized_left = static_cast<bfloat>(
                            row_input[static_cast<int64_t>(pair)
                                * dimension_stride] * inverse_mean[0]);
                        const bfloat normalized_right = static_cast<bfloat>(
                            row_input[static_cast<int64_t>(pair + kPairs)
                                * dimension_stride] * inverse_mean[0]);
                        const bfloat weighted_left =
                            k_weight[pair] * normalized_left;
                        const bfloat weighted_right =
                            k_weight[pair + kPairs] * normalized_right;
                        if (kSharesFullKVReduction) {
                            combined_kv[
                                slab_elements
                                + (projection_head * output_capacity + token)
                                    * kHeadDim
                                + pair
                            ] = static_cast<bfloat>(1.0f) * normalized_left;
                            combined_kv[
                                slab_elements
                                + (projection_head * output_capacity + token)
                                    * kHeadDim
                                + pair + kPairs
                            ] = static_cast<bfloat>(1.0f) * normalized_right;
                        }
                        const uint rope_index =
                            (static_cast<uint>(start_position) + token)
                                * kPairs + pair;
                        const float cosine = rope_cosines[rope_index];
                        const float sine = rope_sines[rope_index];
                        const float left = static_cast<float>(weighted_left);
                        const float right = static_cast<float>(weighted_right);
                        output[pair] = static_cast<bfloat>(
                            left * cosine - right * sine);
                        output[pair + kPairs] = static_cast<bfloat>(
                            left * sine + right * cosine);
                    }
                }
    """
    source.replaceSubrange(
        stagedRange.lowerBound..<source.endIndex,
        with: directBody
    )
    precondition(!source.contains("normalized_row"))
    return source
}

/// Direct-normalization specialization for the combined Q/K/V prefill kernel.
/// Like the K/V-only variant, it changes no accumulation, SIMD reduction,
/// reciprocal-square-root, or BF16 conversion boundary.
func gemma4DirectCombinedQKVPrefillRMSRoPESource(
    _ reference: String
) -> String {
    var source = reference

    let declaration = "threadgroup bfloat normalized_row[kHeadDim];"
    precondition(source.components(separatedBy: declaration).count == 2)
    source = source.replacingOccurrences(of: declaration, with: "")

    let stagedBodyStart = "const bool has_weight = is_q || is_k;"
    precondition(source.components(separatedBy: stagedBodyStart).count == 2)
    guard let stagedRange = source.range(of: stagedBodyStart) else {
        preconditionFailure("missing staged combined Q/K/V prefill body")
    }

    let directBody = """
    const bool has_weight = is_q || is_k;
                const device bfloat* weight = is_q ? q_weight : k_weight;
                if (!has_weight) {
                    for (uint index = 0; index < kReads; ++index) {
                        const uint dimension =
                            thread_position_in_threadgroup.x * kReads + index;
                        const bfloat normalized = static_cast<bfloat>(
                            input[static_cast<int64_t>(index) * dimension_stride]
                                * inverse_mean[0]);
                        output[dimension] =
                            static_cast<bfloat>(1.0f) * normalized;
                    }
                } else {
                    constexpr uint kPairs = kHeadDim / 2;
                    constexpr uint kThreads = kHeadDim / kReads;
                    const device bfloat* row_input = input
                        - static_cast<int64_t>(
                            thread_position_in_threadgroup.x * kReads
                        ) * dimension_stride;
                    for (uint pair = thread_position_in_threadgroup.x;
                         pair < kPairs;
                         pair += kThreads) {
                        const bfloat normalized_left = static_cast<bfloat>(
                            row_input[static_cast<int64_t>(pair)
                                * dimension_stride] * inverse_mean[0]);
                        const bfloat normalized_right = static_cast<bfloat>(
                            row_input[static_cast<int64_t>(pair + kPairs)
                                * dimension_stride] * inverse_mean[0]);
                        const bfloat weighted_left =
                            weight[pair] * normalized_left;
                        const bfloat weighted_right =
                            weight[pair + kPairs] * normalized_right;
                        if (kSharesFullKVReduction && is_k) {
                            combined_kv[
                                slab_elements
                                + (projection_head * output_capacity + token)
                                    * kHeadDim
                                + pair
                            ] = static_cast<bfloat>(1.0f) * normalized_left;
                            combined_kv[
                                slab_elements
                                + (projection_head * output_capacity + token)
                                    * kHeadDim
                                + pair + kPairs
                            ] = static_cast<bfloat>(1.0f) * normalized_right;
                        }
                        const uint rope_index =
                            (static_cast<uint>(start_position) + token)
                                * kPairs + pair;
                        const float cosine = rope_cosines[rope_index];
                        const float sine = rope_sines[rope_index];
                        const float left = static_cast<float>(weighted_left);
                        const float right = static_cast<float>(weighted_right);
                        output[pair] = static_cast<bfloat>(
                            left * cosine - right * sine);
                        output[pair + kPairs] = static_cast<bfloat>(
                            left * sine + right * cosine);
                    }
                }
    """
    source.replaceSubrange(
        stagedRange.lowerBound..<source.endIndex,
        with: directBody
    )
    precondition(!source.contains("normalized_row"))
    return source
}

private func makeGemma4FusedAttentionRMSKernel(
    name: String,
    headDim: Int,
    kvHeads: Int,
    sharesFullKVReduction: Bool,
    combinedKVOutput: Bool = false
) -> MLXFast.MLXFastKernel {
    precondition(headDim == 256 || headDim == 512)
    precondition(kvHeads == 16 || kvHeads == 4)
    let outputNames = combinedKVOutput
        ? ["queries", "combined_kv"]
        : ["queries", "keys", "values"]
    let kvOutputPointer = combinedKVOutput
        ? "(combined_kv + (is_k ? 0 : kKVSlabElements) + projection_row * kHeadDim)"
        : "(is_k ? keys : values) + projection_row * kHeadDim"
    let sharedValueOutput = combinedKVOutput
        ? "combined_kv[kKVSlabElements + projection_row * kHeadDim + dimension]"
        : "values[projection_row * kHeadDim + dimension]"
    return MLXFast.metalKernel(
        name: name,
        inputNames: [
            "raw_q", "raw_k", "raw_v", "q_weight", "k_weight",
            "position", "rope_cosines", "rope_sines",
        ],
        outputNames: outputNames,
        source: """
            constexpr uint kHeadDim = \(headDim);
            constexpr uint kQHeads = 32;
            constexpr uint kKVHeads = \(kvHeads);
            constexpr uint kKVSlabElements = kKVHeads * kHeadDim;
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
                : \(kvOutputPointer);
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
                    \(sharedValueOutput) =
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

private let gemma4FusedSlidingAttentionRMSCombined = makeGemma4FusedAttentionRMSKernel(
    name: "gemma4_fused_sliding_attention_rms_rope_table_combined_kv_256_v1",
    headDim: 256,
    kvHeads: 16,
    sharesFullKVReduction: false,
    combinedKVOutput: true
)

private let gemma4FusedFullAttentionRMSCombined = makeGemma4FusedAttentionRMSKernel(
    name: "gemma4_fused_full_attention_rms_rope_table_shared_combined_kv_512_v1",
    headDim: 512,
    kvHeads: 4,
    sharesFullKVReduction: true,
    combinedKVOutput: true
)

private func makeGemma4CombinedKVPrefillKernel(
    name: String,
    headDim: Int,
    kvHeads: Int,
    sharesFullKVReduction: Bool,
    directRMSRoPE: Bool = false
) -> MLXFast.MLXFastKernel {
    precondition(headDim == 256 || headDim == 512)
    precondition(kvHeads == 16 || kvHeads == 4)
    let referenceSource = """
            constexpr uint kHeadDim = \(headDim);
            constexpr uint kKVHeads = \(kvHeads);
            constexpr uint kRowsPerToken =
                kKVHeads * (\(sharesFullKVReduction) ? 1 : 2);
            constexpr uint kReads = 4;
            constexpr uint kSIMDSize = 32;
            constexpr bool kSharesFullKVReduction = \(sharesFullKVReduction);

            const uint combined_row = threadgroup_position_in_grid.y;
            const uint token = combined_row / kRowsPerToken;
            const uint token_row = combined_row - token * kRowsPerToken;
            const bool is_k = kSharesFullKVReduction || token_row < kKVHeads;
            const uint projection_head = is_k
                ? token_row
                : token_row - kKVHeads;
            const uint input_length = static_cast<uint>(valid_length);
            const uint output_capacity = static_cast<uint>(capacity);
            const uint slab_elements = kKVHeads * output_capacity * kHeadDim;

            device bfloat* output = combined_kv
                + (is_k ? 0 : slab_elements)
                + (projection_head * output_capacity + token) * kHeadDim;
            if (token >= input_length) {
                for (uint index = 0; index < kReads; ++index) {
                    const uint dimension =
                        thread_position_in_threadgroup.x * kReads + index;
                    output[dimension] = static_cast<bfloat>(0.0f);
                    if (kSharesFullKVReduction && is_k) {
                        combined_kv[
                            slab_elements
                            + (projection_head * output_capacity + token)
                                * kHeadDim
                            + dimension
                        ] = static_cast<bfloat>(0.0f);
                    }
                }
                return;
            }

            // The promoted combined-attention prefill projection returns K/V
            // as last-axis slices of one wider Q/K/V parent. Their innermost
            // dimension remains unit-stride, but adjacent tokens are separated
            // by the full combined projection width. Consume the actual view
            // strides so MLX need not materialize row-contiguous K/V copies.
            const constant int64_t* input_strides = is_k
                ? raw_k_strides
                : raw_v_strides;
            const int64_t dimension_stride = input_strides[2];
            const device bfloat* input = (is_k ? raw_k : raw_v)
                + static_cast<int64_t>(token) * input_strides[1]
                + static_cast<int64_t>(projection_head * kHeadDim)
                    * dimension_stride;

            float accumulator = 0;
            input += static_cast<int64_t>(
                thread_position_in_threadgroup.x * kReads
            ) * dimension_stride;
            if (thread_position_in_threadgroup.x * kReads + kReads <= kHeadDim) {
                for (uint index = 0; index < kReads; ++index) {
                    const float value = input[
                        static_cast<int64_t>(index) * dimension_stride];
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
                k_weight + thread_position_in_threadgroup.x * kReads;
            for (uint index = 0; index < kReads; ++index) {
                const uint dimension =
                    thread_position_in_threadgroup.x * kReads + index;
                const bfloat normalized = static_cast<bfloat>(
                    input[static_cast<int64_t>(index) * dimension_stride]
                        * inverse_mean[0]);
                const bfloat weighted = is_k
                    ? row_weight[index] * normalized
                    : static_cast<bfloat>(1.0f) * normalized;
                normalized_row[dimension] = weighted;
                if (!is_k) {
                    output[dimension] = weighted;
                }
                if (kSharesFullKVReduction && is_k) {
                    combined_kv[
                        slab_elements
                        + (projection_head * output_capacity + token) * kHeadDim
                        + dimension
                    ] = static_cast<bfloat>(1.0f) * normalized;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (is_k) {
                constexpr uint kPairs = kHeadDim / 2;
                constexpr uint kThreads = kHeadDim / kReads;
                for (uint pair = thread_position_in_threadgroup.x;
                     pair < kPairs;
                     pair += kThreads) {
                    const uint rope_index =
                        (static_cast<uint>(start_position) + token) * kPairs + pair;
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
            """
    let source = directRMSRoPE
        ? gemma4DirectCombinedKVPrefillRMSRoPESource(referenceSource)
        : referenceSource
    return MLXFast.metalKernel(
        name: name,
        inputNames: [
            "raw_k", "raw_v", "k_weight", "start_position", "valid_length",
            "capacity", "rope_cosines", "rope_sines",
        ],
        outputNames: ["combined_kv"],
        source: source,
        header: "using namespace metal;",
        ensureRowContiguous: false
    )
}

private let gemma4SlidingCombinedKVPrefill = makeGemma4CombinedKVPrefillKernel(
    name: "gemma4_sliding_combined_kv_prefill_strided_256_v3",
    headDim: 256,
    kvHeads: 16,
    sharesFullKVReduction: false
)

private let gemma4FullCombinedKVPrefill = makeGemma4CombinedKVPrefillKernel(
    name: "gemma4_full_combined_kv_prefill_shared_strided_512_v3",
    headDim: 512,
    kvHeads: 4,
    sharesFullKVReduction: true
)

private let gemma4DirectSlidingCombinedKVPrefill =
    makeGemma4CombinedKVPrefillKernel(
        name: "gemma4_direct_sliding_combined_kv_prefill_strided_256_v1",
        headDim: 256,
        kvHeads: 16,
        sharesFullKVReduction: false,
        directRMSRoPE: true
    )

private let gemma4DirectFullCombinedKVPrefill =
    makeGemma4CombinedKVPrefillKernel(
        name: "gemma4_direct_full_combined_kv_prefill_shared_strided_512_v1",
        headDim: 512,
        kvHeads: 4,
        sharesFullKVReduction: true,
        directRMSRoPE: true
    )

/// Prefill preparation for all attention projections. Q/K/V remain strided
/// slices of the combined projection; each threadgroup owns one token/head row
/// and writes the final query/cache layouts directly.
private func makeGemma4CombinedQKVPrefillPreparationKernel(
    name: String,
    headDim: Int,
    kvHeads: Int,
    sharesFullKVReduction: Bool,
    directRMSRoPE: Bool = false
) -> MLXFast.MLXFastKernel {
    precondition(headDim == 256 || headDim == 512)
    precondition(kvHeads == 16 || kvHeads == 4)
    let referenceSource = """
            constexpr uint kHeadDim = \(headDim);
            constexpr uint kQHeads = 32;
            constexpr uint kKVHeads = \(kvHeads);
            constexpr uint kKVRowsPerToken =
                kKVHeads * (\(sharesFullKVReduction) ? 1 : 2);
            constexpr uint kReads = 4;
            constexpr uint kSIMDSize = 32;
            constexpr bool kSharesFullKVReduction = \(sharesFullKVReduction);

            const uint combined_row = threadgroup_position_in_grid.y;
            const uint input_length = static_cast<uint>(valid_length);
            const uint output_capacity = static_cast<uint>(capacity);
            const uint query_rows = input_length * kQHeads;
            const bool is_q = combined_row < query_rows;

            uint token;
            uint projection_head;
            bool is_k;
            if (is_q) {
                token = combined_row / kQHeads;
                projection_head = combined_row - token * kQHeads;
                is_k = false;
            } else {
                const uint kv_row = combined_row - query_rows;
                token = kv_row / kKVRowsPerToken;
                const uint token_row = kv_row - token * kKVRowsPerToken;
                is_k = kSharesFullKVReduction || token_row < kKVHeads;
                projection_head = is_k
                    ? token_row
                    : token_row - kKVHeads;
            }

            const uint slab_elements =
                kKVHeads * output_capacity * kHeadDim;
            device bfloat* output = is_q
                ? queries + (projection_head * input_length + token) * kHeadDim
                : combined_kv
                    + (is_k ? 0 : slab_elements)
                    + (projection_head * output_capacity + token) * kHeadDim;

            // Query rows exist only for valid tokens. K/V rows span capacity so
            // the reserved cache suffix is initialized without a concatenate.
            if (!is_q && token >= input_length) {
                for (uint index = 0; index < kReads; ++index) {
                    const uint dimension =
                        thread_position_in_threadgroup.x * kReads + index;
                    output[dimension] = static_cast<bfloat>(0.0f);
                    if (kSharesFullKVReduction && is_k) {
                        combined_kv[
                            slab_elements
                            + (projection_head * output_capacity + token)
                                * kHeadDim
                            + dimension
                        ] = static_cast<bfloat>(0.0f);
                    }
                }
                return;
            }

            // Q/K/V are last-axis views of a wider combined projection. Use
            // their signed MLX strides rather than materializing contiguous
            // copies before normalization.
            const constant int64_t* input_strides = is_q
                ? raw_q_strides
                : (is_k ? raw_k_strides : raw_v_strides);
            const int64_t dimension_stride = input_strides[2];
            const device bfloat* input_base = is_q
                ? raw_q
                : (is_k ? raw_k : raw_v);
            const device bfloat* input = input_base
                + static_cast<int64_t>(token) * input_strides[1]
                + static_cast<int64_t>(projection_head * kHeadDim)
                    * dimension_stride;

            float accumulator = 0;
            input += static_cast<int64_t>(
                thread_position_in_threadgroup.x * kReads
            ) * dimension_stride;
            if (thread_position_in_threadgroup.x * kReads + kReads <= kHeadDim) {
                for (uint index = 0; index < kReads; ++index) {
                    const float value = input[
                        static_cast<int64_t>(index) * dimension_stride];
                    accumulator += value * value;
                }
            }
            accumulator = simd_sum(accumulator);

            // Match MLXFast.rmsNorm's established reduction/cast topology used
            // by the exact single-token and K/V-prefill preparation kernels.
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

            const bool has_weight = is_q || is_k;
            const device bfloat* weight = is_q ? q_weight : k_weight;
            const device bfloat* row_weight =
                weight + thread_position_in_threadgroup.x * kReads;
            for (uint index = 0; index < kReads; ++index) {
                const uint dimension =
                    thread_position_in_threadgroup.x * kReads + index;
                const bfloat normalized = static_cast<bfloat>(
                    input[static_cast<int64_t>(index) * dimension_stride]
                        * inverse_mean[0]);
                const bfloat weighted = has_weight
                    ? row_weight[index] * normalized
                    : static_cast<bfloat>(1.0f) * normalized;
                normalized_row[dimension] = weighted;
                if (!has_weight) {
                    output[dimension] = weighted;
                }
                if (kSharesFullKVReduction && is_k) {
                    combined_kv[
                        slab_elements
                        + (projection_head * output_capacity + token) * kHeadDim
                        + dimension
                    ] = static_cast<bfloat>(1.0f) * normalized;
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
                        (static_cast<uint>(start_position) + token)
                            * kPairs + pair;
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
            """
    let source = directRMSRoPE
        ? gemma4DirectCombinedQKVPrefillRMSRoPESource(referenceSource)
        : referenceSource
    return MLXFast.metalKernel(
        name: name,
        inputNames: [
            "raw_q", "raw_k", "raw_v", "q_weight", "k_weight",
            "start_position", "valid_length", "capacity", "rope_cosines",
            "rope_sines",
        ],
        outputNames: ["queries", "combined_kv"],
        source: source,
        header: "using namespace metal;",
        ensureRowContiguous: false
    )
}

private let gemma4SlidingCombinedQKVPrefillPreparation =
    makeGemma4CombinedQKVPrefillPreparationKernel(
        name: "gemma4_sliding_combined_qkv_prefill_prep_strided_256_v1",
        headDim: 256,
        kvHeads: 16,
        sharesFullKVReduction: false
    )

private let gemma4FullCombinedQKVPrefillPreparation =
    makeGemma4CombinedQKVPrefillPreparationKernel(
        name: "gemma4_full_combined_qkv_prefill_prep_shared_strided_512_v1",
        headDim: 512,
        kvHeads: 4,
        sharesFullKVReduction: true
    )

private let gemma4DirectSlidingCombinedQKVPrefillPreparation =
    makeGemma4CombinedQKVPrefillPreparationKernel(
        name: "gemma4_direct_sliding_combined_qkv_prefill_strided_256_v1",
        headDim: 256,
        kvHeads: 16,
        sharesFullKVReduction: false,
        directRMSRoPE: true
    )

private let gemma4DirectFullCombinedQKVPrefillPreparation =
    makeGemma4CombinedQKVPrefillPreparationKernel(
        name: "gemma4_direct_full_combined_qkv_prefill_shared_strided_512_v1",
        headDim: 512,
        kvHeads: 4,
        sharesFullKVReduction: true,
        directRMSRoPE: true
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

private func gemma4VerifyDirectPrefillAttentionRMSRoPEOutputs(
    candidate: [MLXArray],
    reference: [MLXArray],
    kind: String
) {
    precondition(candidate.count == reference.count)
    var valueCount = 0
    for (outputIndex, pair) in zip(candidate, reference).enumerated() {
        let candidateOutput = pair.0
        let referenceOutput = pair.1
        precondition(candidateOutput.dtype == .bfloat16)
        precondition(referenceOutput.dtype == .bfloat16)
        precondition(candidateOutput.shape == referenceOutput.shape)
        valueCount += candidateOutput.size
        let candidateBits = candidateOutput.view(dtype: .uint16)
        let referenceBits = referenceOutput.view(dtype: .uint16)
        let matches = arrayEqual(candidateBits, referenceBits)
        eval(matches)
        guard matches.item(Bool.self) else {
            let candidateValues = candidateBits.asArray(UInt16.self)
            let referenceValues = referenceBits.asArray(UInt16.self)
            let mismatch = zip(candidateValues, referenceValues)
                .enumerated()
                .first { $0.element.0 != $0.element.1 }
            preconditionFailure(
                "direct prefill attention RMS/RoPE raw BF16 mismatch "
                    + "kind=\(kind) output=\(outputIndex) "
                    + "index=\(mismatch?.offset ?? -1) "
                    + "candidate=\(mismatch?.element.0 ?? 0) "
                    + "reference=\(mismatch?.element.1 ?? 0)"
            )
        }
    }
    gemma4RecordDirectPrefillAttentionRMSRoPEVerification(
        kind: kind,
        arrays: candidate.count,
        values: valueCount
    )
}

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
    private let directPrefillRMSRoPE: Bool
    private let verifyDirectPrefillRMSRoPEBits: Bool

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
        self.directPrefillRMSRoPE = gemma4PrefillAttentionEnvironmentFlag(
            "DARKBLOOM_DIRECT_PREFILL_ATTENTION_RMS_ROPE",
            default: gemma4DirectPrefillAttentionRMSRoPEDefault
        )
        self.verifyDirectPrefillRMSRoPEBits =
            gemma4PrefillAttentionEnvironmentFlag(
                "DARKBLOOM_VERIFY_DIRECT_PREFILL_ATTENTION_RMS_ROPE_BITS",
                default: false
            )
    }

    func supports(offset: Int) -> Bool {
        (0..<4096).contains(offset)
    }

    func supportsPrefill(offset: Int, length: Int) -> Bool {
        guard offset >= 0, length > 1 else { return false }
        let (end, overflow) = offset.addingReportingOverflow(length)
        return !overflow && end <= 4096
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

    /// Direct-output variant for `Gemma4CombinedKVCache`. The second output is
    /// the parent K/V-major allocation `[2,1,Hkv,1,D]`; the cache extracts K/V
    /// consumers with range slices, never integer-index/take operations.
    func callCombined(
        rawQueries: MLXArray,
        rawKeys: MLXArray,
        rawValues: MLXArray?,
        offset: Int
    ) -> (queries: MLXArray, combinedKV: MLXArray) {
        let queryWidth = 32 * headDim
        let kvWidth = kvHeads * headDim
        precondition(rawQueries.dtype == .bfloat16)
        precondition(rawKeys.dtype == .bfloat16)
        precondition(rawQueries.shape == [1, 1, queryWidth])
        precondition(rawKeys.shape == [1, 1, kvWidth])
        let valueInput = rawValues ?? rawKeys
        precondition(valueInput.dtype == .bfloat16)
        precondition(valueInput.shape == [1, 1, kvWidth])
        precondition(supports(offset: offset))

        let threads = headDim / 4
        let kernel = isSliding
            ? gemma4FusedSlidingAttentionRMSCombined
            : gemma4FusedFullAttentionRMSCombined
        let outputs = kernel(
            [
                rawQueries, rawKeys, valueInput, qNormWeight, kNormWeight,
                positionViews[offset], ropeCosines, ropeSines,
            ],
            grid: (threads, 32 + (isSliding ? 2 * kvHeads : kvHeads), 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [
                [1, 32, 1, headDim],
                [2, 1, kvHeads, 1, headDim],
            ],
            outputDTypes: [.bfloat16, .bfloat16]
        )
        return (outputs[0], outputs[1])
    }

    /// Multi-token K/V preparation that writes the seed cache in its final
    /// K/V-major layout. The reserved suffix is zero-filled in the same kernel
    /// so the entire cache allocation is deterministic without a concatenate.
    func callCombinedPrefill(
        rawKeys: MLXArray,
        rawValues: MLXArray?,
        offset: Int,
        length: Int,
        capacity: Int
    ) -> MLXArray {
        let kvWidth = kvHeads * headDim
        precondition(rawKeys.dtype == .bfloat16)
        precondition(rawKeys.shape == [1, length, kvWidth])
        let valueInput = rawValues ?? rawKeys
        precondition(valueInput.dtype == .bfloat16)
        precondition(valueInput.shape == [1, length, kvWidth])
        precondition(capacity >= length)
        precondition(supportsPrefill(offset: offset, length: length))

        let threads = headDim / 4
        let rowsPerToken = isSliding ? 2 * kvHeads : kvHeads
        let inputs = [
            rawKeys, valueInput, kNormWeight, positionViews[offset],
            MLXArray(Int32(length)), MLXArray(Int32(capacity)),
            ropeCosines, ropeSines,
        ]
        let grid = (threads, capacity * rowsPerToken, 1)
        let shapes = [[2, 1, kvHeads, capacity, headDim]]
        let candidate: [MLXArray]?
        if directPrefillRMSRoPE || verifyDirectPrefillRMSRoPEBits {
            let candidateKernel = isSliding
                ? gemma4DirectSlidingCombinedKVPrefill
                : gemma4DirectFullCombinedKVPrefill
            candidate = candidateKernel(
                inputs,
                grid: grid,
                threadGroup: (threads, 1, 1),
                outputShapes: shapes,
                outputDTypes: [.bfloat16]
            )
        } else {
            candidate = nil
        }
        let reference: [MLXArray]?
        if !directPrefillRMSRoPE || verifyDirectPrefillRMSRoPEBits {
            let referenceKernel = isSliding
                ? gemma4SlidingCombinedKVPrefill
                : gemma4FullCombinedKVPrefill
            reference = referenceKernel(
                inputs,
                grid: grid,
                threadGroup: (threads, 1, 1),
                outputShapes: shapes,
                outputDTypes: [.bfloat16]
            )
        } else {
            reference = nil
        }
        if verifyDirectPrefillRMSRoPEBits,
           let candidate,
           let reference
        {
            gemma4VerifyDirectPrefillAttentionRMSRoPEOutputs(
                candidate: candidate,
                reference: reference,
                kind: isSliding ? "sliding_kv" : "full_kv"
            )
        }
        let outputs = directPrefillRMSRoPE
            ? (candidate ?? reference)
            : (reference ?? candidate)
        guard let outputs else {
            preconditionFailure(
                "combined K/V prefill RMS/RoPE kernel was not selected"
            )
        }
        return outputs[0]
    }

    /// Multi-token Q/K/V preparation. Queries are emitted as `[1,32,L,D]`
    /// after RMSNorm, transpose, and RoPE; K/V are emitted directly into the
    /// reserved combined-cache parent allocation.
    func callCombinedQKVPrefill(
        rawQueries: MLXArray,
        rawKeys: MLXArray,
        rawValues: MLXArray?,
        offset: Int,
        length: Int,
        capacity: Int
    ) -> (queries: MLXArray, combinedKV: MLXArray) {
        let queryWidth = 32 * headDim
        let kvWidth = kvHeads * headDim
        precondition(rawQueries.dtype == .bfloat16)
        precondition(rawQueries.shape == [1, length, queryWidth])
        precondition(rawKeys.dtype == .bfloat16)
        precondition(rawKeys.shape == [1, length, kvWidth])
        let valueInput = rawValues ?? rawKeys
        precondition(valueInput.dtype == .bfloat16)
        precondition(valueInput.shape == [1, length, kvWidth])
        precondition(capacity >= length)
        precondition(supportsPrefill(offset: offset, length: length))

        let threads = headDim / 4
        let kvRowsPerToken = isSliding ? 2 * kvHeads : kvHeads
        let inputs = [
            rawQueries, rawKeys, valueInput, qNormWeight, kNormWeight,
            positionViews[offset], MLXArray(Int32(length)),
            MLXArray(Int32(capacity)), ropeCosines, ropeSines,
        ]
        let grid = (
            threads,
            length * 32 + capacity * kvRowsPerToken,
            1
        )
        let shapes = [
            [1, 32, length, headDim],
            [2, 1, kvHeads, capacity, headDim],
        ]
        let candidate: [MLXArray]?
        if directPrefillRMSRoPE || verifyDirectPrefillRMSRoPEBits {
            let candidateKernel = isSliding
                ? gemma4DirectSlidingCombinedQKVPrefillPreparation
                : gemma4DirectFullCombinedQKVPrefillPreparation
            candidate = candidateKernel(
                inputs,
                grid: grid,
                threadGroup: (threads, 1, 1),
                outputShapes: shapes,
                outputDTypes: [.bfloat16, .bfloat16]
            )
        } else {
            candidate = nil
        }
        let reference: [MLXArray]?
        if !directPrefillRMSRoPE || verifyDirectPrefillRMSRoPEBits {
            let referenceKernel = isSliding
                ? gemma4SlidingCombinedQKVPrefillPreparation
                : gemma4FullCombinedQKVPrefillPreparation
            reference = referenceKernel(
                inputs,
                grid: grid,
                threadGroup: (threads, 1, 1),
                outputShapes: shapes,
                outputDTypes: [.bfloat16, .bfloat16]
            )
        } else {
            reference = nil
        }
        if verifyDirectPrefillRMSRoPEBits,
           let candidate,
           let reference
        {
            gemma4VerifyDirectPrefillAttentionRMSRoPEOutputs(
                candidate: candidate,
                reference: reference,
                kind: isSliding ? "sliding_qkv" : "full_qkv"
            )
        }
        let outputs = directPrefillRMSRoPE
            ? (candidate ?? reference)
            : (reference ?? candidate)
        guard let outputs else {
            preconditionFailure(
                "combined Q/K/V prefill RMS/RoPE kernel was not selected"
            )
        }
        return (outputs[0], outputs[1])
    }
}
