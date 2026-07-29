import MLX
import MLXFast

/// Certified Middle-Out Attention (CMOA) configuration for Laguna.
///
/// A small certified attention-output error does not prove that all later
/// layers preserve the dense model's greedy token, which is the challenge's
/// actual correctness contract. The ranked correctness pass remains the
/// authority for this fixed candidate configuration.
///
/// The first implementation uses exact per-page query/key score maxima. Those
/// are tighter bounds than a centroid/radius sphere and make the residual-mass
/// certificate exact, at the cost of reading all keys during page discovery.
/// Selected pages are then read again with their values. A future persistent
/// KV hierarchy can replace the discovery pass without changing the selection
/// or certification logic in the attention kernel.
struct LagunaCMOAConfiguration: Sendable {
    let pageSize: Int
    let minimumContext: Int
    let recentTokens: Int
    let maximumSelectedFraction: Float
    let tailMassTolerance: Float
    let expandNeighbours: Bool

    static let current = LagunaCMOAConfiguration(
        pageSize: 16,
        minimumContext: 512,
        recentTokens: 128,
        maximumSelectedFraction: 0.5,
        tailMassTolerance: 0.001,
        expandNeighbours: true
    )
}

let lagunaCMOAConfiguration = LagunaCMOAConfiguration.current

/// The kernel supports the compiled decode cache's 4096-token backing store.
/// With 16-token pages this is 256 page records, comfortably inside one
/// threadgroup's memory budget.
private let lagunaCMOAMaximumPages = 256

private func lagunaCMOAFloatLiteral(_ value: Float) -> String {
    // Configuration values are clamped finite values. Appending `f` keeps the
    // generated MSL constants in float rather than double syntax.
    "\(value)f"
}

private let lagunaCMOAKernel: MLXFast.MLXFastKernel = {
    let configuration = lagunaCMOAConfiguration
    let pageSize = configuration.pageSize
    let recentPages = max(
        1, (configuration.recentTokens + pageSize - 1) / pageSize)
    let expandNeighbours = configuration.expandNeighbours ? "true" : "false"

    return MLXFast.metalKernel(
        name:
            "laguna_cmoa_decode_bf16_p\(pageSize)_r\(recentPages)_v1",
        inputNames: ["queries", "keys", "values", "valid_length"],
        outputNames: ["output"],
        source: """
            constexpr uint head_dim = 128;
            constexpr uint query_heads = 48;
            constexpr uint kv_heads = 8;
            constexpr uint queries_per_kv_head = 6;
            constexpr uint page_size = \(pageSize);
            constexpr uint maximum_pages = \(lagunaCMOAMaximumPages);
            constexpr uint recent_pages = \(recentPages);
            constexpr uint minimum_context = \(configuration.minimumContext);
            constexpr float maximum_selected_fraction =
                \(lagunaCMOAFloatLiteral(configuration.maximumSelectedFraction));
            constexpr float tail_mass_tolerance =
                \(lagunaCMOAFloatLiteral(configuration.tailMassTolerance));
            constexpr bool expand_neighbours = \(expandNeighbours);
            constexpr float attention_scale = 0.08838834764831845f;

            uint query_head = threadgroup_position_in_grid.x;
            uint lane = thread_index_in_simdgroup;
            uint capacity = keys_shape[2];
            uint valid = min(uint(max(valid_length[0], 0)), capacity);
            uint page_count = (valid + page_size - 1) / page_size;
            uint kv_head = query_head / queries_per_kv_head;

            threadgroup float page_upper[maximum_pages];
            threadgroup ushort page_tokens[maximum_pages];
            threadgroup uchar selected[maximum_pages];
            threadgroup uint next_pages[2];
            threadgroup uint next_page_count;
            threadgroup uint selected_page_count;
            threadgroup bool stop_search;
            threadgroup bool dense_fallback;

            uint component_base = lane * 4;
            thread float query_values[4];
            thread float numerator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint i = 0; i < 4; ++i) {
                query_values[i] =
                    float(queries[query_head * head_dim + component_base + i]);
            }

            if (lane == 0) {
                for (uint page = 0; page < maximum_pages; ++page) {
                    page_upper[page] = -INFINITY;
                    page_tokens[page] = 0;
                    selected[page] = 0;
                }
                next_page_count = 0;
                selected_page_count = 0;
                stop_search = false;
                dense_fallback = false;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Page discovery. The maximum exact q·k score in a page is a
            // query-dependent upper bound for every token in that page. This
            // first pass reads K but not V.
            for (uint token = 0; token < valid; ++token) {
                uint key_base =
                    ((kv_head * capacity + token) * head_dim) + component_base;
                float partial = 0.0f;
                for (uint i = 0; i < 4; ++i) {
                    partial += query_values[i] * float(keys[key_base + i]);
                }
                float score = simd_sum(partial) * attention_scale;
                if (lane == 0) {
                    uint page = token / page_size;
                    page_upper[page] = max(page_upper[page], score);
                    page_tokens[page] += 1;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (valid == 0 || page_count == 0) {
                for (uint i = 0; i < 4; ++i) {
                    output[query_head * head_dim + component_base + i] =
                        bfloat(0.0f);
                }
                return;
            }

            // For short contexts, retain the kernel's dense path. The Swift
            // caller normally avoids dispatching this kernel for a short
            // direct cache; this device-side guard also handles a compiled
            // fixed-capacity cache whose valid length is graph-valued.
            if (lane == 0) {
                if (valid < minimum_context) {
                    for (uint page = 0; page < page_count; ++page) {
                        selected[page] = 1;
                    }
                    selected_page_count = page_count;
                    dense_fallback = true;
                } else {
                    // Page zero is the attention-sink page. The newest pages
                    // form the mandatory local window.
                    selected[0] = 1;
                    selected_page_count = 1;
                    uint recent_start =
                        page_count > recent_pages ? page_count - recent_pages : 0;
                    for (uint page = recent_start; page < page_count; ++page) {
                        if (selected[page] == 0) {
                            selected[page] = 1;
                            selected_page_count += 1;
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Online softmax state. Every lane carries the same maximum and
            // denominator; each lane owns four output dimensions.
            float softmax_max = -INFINITY;
            float denominator = 0.0f;

            // Evaluate the mandatory pages in temporal order.
            for (uint page = 0; page < page_count; ++page) {
                if (selected[page] == 0) {
                    continue;
                }
                uint start = page * page_size;
                uint end = min(start + page_size, valid);
                for (uint token = start; token < end; ++token) {
                    uint key_base =
                        ((kv_head * capacity + token) * head_dim) + component_base;
                    float partial = 0.0f;
                    for (uint i = 0; i < 4; ++i) {
                        partial += query_values[i] * float(keys[key_base + i]);
                    }
                    float score = simd_sum(partial) * attention_scale;
                    float new_max = max(softmax_max, score);
                    float previous_scale =
                        isfinite(softmax_max) ? metal::exp(softmax_max - new_max) : 0.0f;
                    float token_scale = metal::exp(score - new_max);
                    uint value_base =
                        ((kv_head * capacity + token) * head_dim) + component_base;
                    for (uint i = 0; i < 4; ++i) {
                        numerator[i] =
                            numerator[i] * previous_scale
                            + token_scale * float(values[value_base + i]);
                    }
                    denominator =
                        denominator * previous_scale + token_scale;
                    softmax_max = new_max;
                }
            }

            // Repeatedly open the page with largest n*exp(U). Its strongest
            // unselected neighbour is opened in the same wave, producing the
            // "centre then expand" traversal while retaining a global route
            // to additional, disjoint semantic centres.
            for (uint iteration = 0; iteration < maximum_pages; ++iteration) {
                if (lane == 0) {
                    float residual = 0.0f;
                    float best_priority = -INFINITY;
                    uint best_page = maximum_pages;

                    for (uint page = 0; page < page_count; ++page) {
                        if (selected[page] != 0) {
                            continue;
                        }
                        float shifted_bound =
                            page_upper[page] - softmax_max;
                        residual +=
                            float(page_tokens[page]) * metal::exp(shifted_bound);
                        float priority =
                            page_upper[page] + metal::log(float(page_tokens[page]));
                        if (priority > best_priority) {
                            best_priority = priority;
                            best_page = page;
                        }
                    }

                    float tail_mass =
                        residual / max(denominator + residual, 1.0e-30f);
                    stop_search =
                        best_page == maximum_pages
                        || tail_mass <= tail_mass_tolerance;
                    next_page_count = 0;

                    if (!stop_search) {
                        uint budget = max(
                            1u,
                            uint(ceil(
                                maximum_selected_fraction * float(page_count))));
                        if (selected_page_count >= budget) {
                            // The certificate did not close within budget.
                            // Select every remaining page so the result is
                            // mathematically dense rather than returning an
                            // uncertified approximation.
                            dense_fallback = true;
                            for (uint page = 0; page < page_count; ++page) {
                                if (selected[page] == 0) {
                                    selected[page] = 2;
                                    selected_page_count += 1;
                                }
                            }
                        } else {
                            selected[best_page] = 2;
                            selected_page_count += 1;
                            next_pages[next_page_count++] = best_page;

                            if (expand_neighbours && selected_page_count < budget) {
                                uint neighbour = maximum_pages;
                                float neighbour_priority = -INFINITY;
                                if (best_page > 0 && selected[best_page - 1] == 0) {
                                    uint candidate = best_page - 1;
                                    neighbour = candidate;
                                    neighbour_priority =
                                        page_upper[candidate]
                                        + metal::log(float(page_tokens[candidate]));
                                }
                                if (best_page + 1 < page_count
                                    && selected[best_page + 1] == 0)
                                {
                                    uint candidate = best_page + 1;
                                    float candidate_priority =
                                        page_upper[candidate]
                                        + metal::log(float(page_tokens[candidate]));
                                    if (candidate_priority > neighbour_priority) {
                                        neighbour = candidate;
                                    }
                                }
                                if (neighbour != maximum_pages) {
                                    selected[neighbour] = 2;
                                    selected_page_count += 1;
                                    next_pages[next_page_count++] = neighbour;
                                }
                            }
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (stop_search) {
                    break;
                }

                // A budget miss marks every remaining page with state 2.
                // A normal centre wave records one or two explicit page IDs.
                if (dense_fallback) {
                    for (uint page = 0; page < page_count; ++page) {
                        if (selected[page] != 2) {
                            continue;
                        }
                        uint start = page * page_size;
                        uint end = min(start + page_size, valid);
                        for (uint token = start; token < end; ++token) {
                            uint key_base =
                                ((kv_head * capacity + token) * head_dim)
                                + component_base;
                            float partial = 0.0f;
                            for (uint i = 0; i < 4; ++i) {
                                partial +=
                                    query_values[i] * float(keys[key_base + i]);
                            }
                            float score = simd_sum(partial) * attention_scale;
                            float new_max = max(softmax_max, score);
                            float previous_scale =
                                metal::exp(softmax_max - new_max);
                            float token_scale = metal::exp(score - new_max);
                            uint value_base =
                                ((kv_head * capacity + token) * head_dim)
                                + component_base;
                            for (uint i = 0; i < 4; ++i) {
                                numerator[i] =
                                    numerator[i] * previous_scale
                                    + token_scale * float(values[value_base + i]);
                            }
                            denominator =
                                denominator * previous_scale + token_scale;
                            softmax_max = new_max;
                        }
                    }
                    break;
                }

                for (uint wave = 0; wave < next_page_count; ++wave) {
                    uint page = next_pages[wave];
                    uint start = page * page_size;
                    uint end = min(start + page_size, valid);
                    for (uint token = start; token < end; ++token) {
                        uint key_base =
                            ((kv_head * capacity + token) * head_dim)
                            + component_base;
                        float partial = 0.0f;
                        for (uint i = 0; i < 4; ++i) {
                            partial +=
                                query_values[i] * float(keys[key_base + i]);
                        }
                        float score = simd_sum(partial) * attention_scale;
                        float new_max = max(softmax_max, score);
                        float previous_scale =
                            metal::exp(softmax_max - new_max);
                        float token_scale = metal::exp(score - new_max);
                        uint value_base =
                            ((kv_head * capacity + token) * head_dim)
                            + component_base;
                        for (uint i = 0; i < 4; ++i) {
                            numerator[i] =
                                numerator[i] * previous_scale
                                + token_scale * float(values[value_base + i]);
                        }
                        denominator =
                            denominator * previous_scale + token_scale;
                        softmax_max = new_max;
                    }
                }

                if (lane == 0) {
                    for (uint page = 0; page < page_count; ++page) {
                        if (selected[page] == 2) {
                            selected[page] = 1;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float inverse_denominator = 1.0f / max(denominator, 1.0e-30f);
            uint output_base = query_head * head_dim + component_base;
            for (uint i = 0; i < 4; ++i) {
                output[output_base + i] =
                    bfloat(numerator[i] * inverse_denominator);
            }
            """,
        ensureRowContiguous: true
    )
}()

/// Returns a CMOA decode result when the exact Laguna full-attention geometry
/// and configured long-context threshold are supported, otherwise nil.
///
/// `validLength` is graph-valued for `CompilableKVCache`, allowing the same
/// kernel to ignore unwritten overflow-bin rows without a CPU readback.
public func lagunaCertifiedMiddleOutDecodeAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    validLength: MLXArray
) -> MLXArray? {
    let configuration = lagunaCMOAConfiguration
    guard queries.dtype == .bfloat16,
        keys.dtype == .bfloat16,
        values.dtype == .bfloat16,
        validLength.dtype == .int32,
        queries.shape == [
            1, LagunaConstants.fullAttentionHeads, 1, LagunaConstants.headDim,
        ],
        keys.ndim == 4,
        values.shape == keys.shape,
        keys.dim(0) == 1,
        keys.dim(1) == LagunaConstants.numKeyValueHeads,
        keys.dim(3) == LagunaConstants.headDim,
        validLength.size == 1
    else {
        return nil
    }

    let capacity = keys.dim(2)
    let pageCount = (capacity + configuration.pageSize - 1) / configuration.pageSize
    guard capacity >= configuration.minimumContext,
        pageCount <= lagunaCMOAMaximumPages
    else {
        return nil
    }

    lagunaTrace("certified middle-out attention")
    return lagunaCMOAKernel(
        [queries, keys, values, validLength],
        grid: (LagunaConstants.fullAttentionHeads * 32, 1, 1),
        threadGroup: (32, 1, 1),
        outputShapes: [[
            1, LagunaConstants.fullAttentionHeads, 1, LagunaConstants.headDim,
        ]],
        outputDTypes: [.bfloat16]
    )[0]
}
