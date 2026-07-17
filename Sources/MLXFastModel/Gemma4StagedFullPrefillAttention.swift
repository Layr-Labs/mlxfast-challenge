import Foundation
import MLX

/// CPU-side mirror of the staged FULL-attention prefill kernel's tile
/// geometry.
///
/// The staged full-attention kernel runs the ranked prefill shape of the ten
/// full-attention layers: B=1, L=512, 32 query heads, 4 KV heads (GQA 8:1),
/// head dim 512, causal mask, cache offset 0. Tests use these mirrors to
/// prove, without touching the MLX runtime, that
/// - the P1-style skip bound skips exactly the fully-masked QK tile set,
/// - the P3-style token-major tile addressing is the exact transpose
///   bijection of the head-major addressing, and
/// - the kernel's GQA head mapping matches the stock fallback's
///   `unflatten(q, 1, {4, 8})` grouping.
///
/// Any change to the Metal-side constants or addressing below must be
/// reflected here; the arithmetic is duplicated intentionally so the tests
/// stay pure CPU.
enum Gemma4StagedFullPrefillGeometry {
    /// Sequence length of the staged prefill (queries and keys).
    static let length = 512
    /// Per-head attention dimension of the full-attention layers
    /// (`global_head_dim`).
    static let headDim = 512
    /// Query rows owned by one threadgroup (M tile).
    static let queryRows = 16
    /// Number of query heads.
    static let queryHeads = 32
    /// Number of KV heads (`num_global_key_value_heads`).
    static let kvHeads = 4
    /// Query heads per KV head (GQA broadcast factor).
    static var gqaFactor: Int { queryHeads / kvHeads }
    /// Key columns covered by one QK tensor-op tile (N tile).
    static let keyTileColumns = 32
    /// Output columns covered by one PV tensor-op tile (N tile).
    static let valueTileColumns = 32
    /// Cache offset of the staged prefill dispatch.
    static let cacheOffset = 0

    /// Total 32-key QK tiles per query block.
    static var keyBlocksTotal: Int { length / keyTileColumns }
    /// Total 16-query blocks per head.
    static var queryBlocksTotal: Int { length / queryRows }

    /// Mask keep predicate for the ranked staged full-attention shape.
    ///
    /// Full-attention layers have no sliding window. The stock fallback
    /// (`mlx::core::fast::scaled_dot_product_attention` with mask mode
    /// `.causal`) builds `q_idx >= k_idx` with `offset = kL - qL = 0`, so a
    /// (query, key) score survives iff `key <= query`.
    static func maskKeeps(queryPosition: Int, keyPosition: Int) -> Bool {
        keyPosition <= queryPosition + cacheOffset
    }

    /// KV head consumed by a query head. Mirrors the stock fallback's
    /// `unflatten(q, 1, {n_kv_heads, n_repeats})` + broadcast K/V grouping:
    /// query head `h = kv * 8 + r` reads KV head `kv = h / 8`. Must match the
    /// Metal-side `kv_head = query_head / kGQAFactor`.
    static func kvHead(forQueryHead queryHead: Int) -> Int {
        queryHead / gqaFactor
    }

    /// P1 bound: number of leading 32-key QK tiles computed for the 16-query
    /// tile `queryBlock`; every tile index at or beyond this bound is skipped.
    ///
    /// Derivation (identical to the sliding staged kernel -- it depends only
    /// on the 16-row/32-column tile shape, not the head dim): the tile
    /// starting at `key_start = 32*keyBlock` is fully masked iff its smallest
    /// key exceeds the tile's largest query position, i.e.
    /// `32*keyBlock > 16*queryBlock + 15`, equivalently
    /// `32*keyBlock >= 16*(queryBlock + 1)`, so the first fully-masked tile
    /// index is `ceil((queryBlock + 1) / 2) == (queryBlock + 2) / 2` in
    /// integer division. Must match the Metal-side `key_block_limit`.
    static func computedKeyBlockCount(queryBlock: Int) -> Int {
        (queryBlock + 2) / 2
    }

    /// P5 bound: number of leading probability columns the PV stage consumes
    /// for the 16-query tile `queryBlock`; every key column at or beyond this
    /// bound is causally masked for all 16 rows of the tile, so its softmax
    /// probability is exactly +0.0 bf16.
    ///
    /// Granularity: the PV matmul reduces over K with a single dynamic-K
    /// tensor op, so any column bound would be dispatchable; the bound is
    /// kept at the same 32-key block granularity as the P1 QK skip
    /// (`keyTileColumns * computedKeyBlockCount`), which is conservative
    /// (partially-live blocks stay fully included) and keeps the truncated
    /// reduction length 32-aligned. Identical to the sliding staged kernel's
    /// bound -- it depends only on the 16-row/32-column tile shape, not the
    /// head dim. Must match the Metal-side `pv_key_column_limit`.
    static func pvKeyColumnLimit(queryBlock: Int) -> Int {
        keyTileColumns * computedKeyBlockCount(queryBlock: queryBlock)
    }

    /// Token-major linear offset of logical element (token, head, dim) in the
    /// P3 output layout `[1, length, queryHeads * headDim]`.
    static func tokenMajorOffset(token: Int, head: Int, dim: Int) -> Int {
        (token * queryHeads + head) * headDim + dim
    }

    /// Head-major linear offset of logical element (head, token, dim) in the
    /// legacy output layout `[1, queryHeads, length, headDim]`.
    static func headMajorOffset(head: Int, token: Int, dim: Int) -> Int {
        (head * length + token) * headDim + dim
    }

    /// Mirrors the kernel's PV output tile addressing: linear offset written
    /// by tile coordinate (queryHead, queryStart, valueStart) at tile-local
    /// (row, lane), where `row` indexes the 16 query rows and `lane` the 32
    /// output columns. Must match the Metal-side `output_tile` base and the
    /// output tensor strides (`{1, kOutputRowStride}`).
    static func outputTileElementOffset(
        tokenMajor: Bool,
        queryHead: Int,
        queryStart: Int,
        valueStart: Int,
        row: Int,
        lane: Int
    ) -> Int {
        if tokenMajor {
            let base = queryStart * (queryHeads * headDim)
                + queryHead * headDim
                + valueStart
            return base + row * (queryHeads * headDim) + lane
        }
        let base = (queryHead * length + queryStart) * headDim + valueStart
        return base + row * headDim + lane
    }
}

/// Pipeline name of the staged full prefill kernel. Internal so CPU tests
/// can pin the DARKBLOOM flag/name-suffix contract (`_skip`/`_tokmaj`/
/// `_pvskip`); the suffix keeps differently-configured pipelines from
/// aliasing in MLX's kernel cache.
let gemma4StagedFullPrefill512KernelName: String =
    "gemma4_staged_full_prefill_16x512x512_mpp_v1"
    + "_skip\(gemma4StagedPrefillCausalTileSkipEnabled ? 1 : 0)"
    + "_tokmaj\(gemma4StagedPrefillTokenMajorOutputEnabled ? 1 : 0)"
    + "_pvskip\(gemma4StagedPrefillPVTileSkipEnabled ? 1 : 0)"

/// Metal source of the staged full prefill kernel. Internal so CPU tests
/// can prove the P5 zero-initialization and `+ 0.0f` canonicalization text
/// survives verbatim in the exact source string handed to MLX's Metal
/// compiler (which disables fast math, so the compiler cannot elide
/// `x + 0.0f` -- it is not an identity under IEEE signed zeros).
let gemma4StagedFullPrefill512KernelSource: String = """
        constexpr uint kLength = 512;
        constexpr uint kHeadDim = 512;
        constexpr uint kQueryRows = 16;
        constexpr uint kQHeads = 32;
        constexpr uint kKVHeads = 4;
        constexpr uint kGQAFactor = 8;
        constexpr uint kThreads = 128;
        constexpr uint kSIMDSize = 32;
        constexpr uint kSIMDGroups = 4;
        constexpr uint kSoftmaxReads = 4;
        constexpr bool kCausalTileSkip =
            \(gemma4StagedPrefillCausalTileSkipEnabled);
        constexpr bool kTokenMajorOutput =
            \(gemma4StagedPrefillTokenMajorOutputEnabled);
        constexpr bool kPVTileSkip =
            \(gemma4StagedPrefillPVTileSkipEnabled);

        const uint thread_index = thread_position_in_threadgroup.x;
        const uint simd_lane = thread_index_in_simdgroup;
        const uint simd_group = simdgroup_index_in_threadgroup;
        const uint query_head = threadgroup_position_in_grid.y;
        const uint query_block = threadgroup_position_in_grid.z;
        const uint query_start = query_block * kQueryRows;
        const uint kv_head = query_head / kGQAFactor;

        threadgroup bfloat scores[kQueryRows * kLength];

        // Four SIMDgroups cooperatively cover 32-key column tiles. MPP keeps
        // the 512-wide head-dim reduction inside the tensor operation, and
        // writes only BF16 scores to on-chip threadgroup storage. Gemma 4's
        // attention scale is exactly 1.0, so Q can come directly from device
        // memory; Apple's MPP guidance recommends relying on cache instead of
        // staging GEMM sources through threadgroup memory.
        //
        // P1 causal tile skip: the 32-key tile at key_start = 32*key_block is
        // fully masked iff its smallest key exceeds this query tile's largest
        // position, i.e. 32*key_block > query_start + kQueryRows - 1, so the
        // first fully-masked tile index is (query_block + 2) / 2 (integer
        // ceil((query_block + 1) / 2)). Tiles at or beyond that bound are
        // never computed or loaded; the causal fill below overwrites every
        // one of their score elements with bfloat lowest anyway, so the
        // threadgroup contents entering softmax are bit-identical. Partially
        // masked tiles keep the unchanged mask arithmetic.
        const uint key_block_limit = kCausalTileSkip
            ? (query_block + 2) / 2
            : kLength / 32;
        constexpr auto qk_descriptor = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 512, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<
            qk_descriptor, metal::execution_simdgroup> qk;
        device bfloat* mutable_queries = const_cast<device bfloat*>(queries)
            + static_cast<int64_t>(query_head) * queries_strides[1]
            + static_cast<int64_t>(query_start) * queries_strides[2];
        for (uint key_block = simd_group;
             key_block < key_block_limit;
             key_block += kSIMDGroups) {
            const uint key_start = key_block * 32;
            device bfloat* mutable_keys = const_cast<device bfloat*>(keys)
                + static_cast<int64_t>(kv_head) * keys_strides[1]
                + static_cast<int64_t>(key_start) * keys_strides[2];
            auto q_tensor = metal::tensor(
                mutable_queries,
                metal::dextents<int, 2>{512, 16},
                metal::array<int64_t, 2>{
                    queries_strides[3], queries_strides[2]});
            auto k_tensor = metal::tensor(
                mutable_keys,
                metal::dextents<int, 2>{512, 32},
                metal::array<int64_t, 2>{keys_strides[3], keys_strides[2]});
            auto score_tensor = metal::tensor(
                scores + key_start,
                metal::dextents<int, 2>{32, 16},
                metal::array<int, 2>{1, 512});
            qk.run(q_tensor, k_tensor, score_tensor);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Offset-zero, L=512 full attention is causal only (no window).
        // Match the stock boolean-mask fill value in the BF16 score dtype:
        // the fallback's where() writes finfo(bfloat16).min, which is
        // numeric_limits<bfloat>::lowest(). This fill covers every element of
        // every P1-skipped tile: a skipped tile satisfies
        // key_start > query_start + row for all of its rows.
        for (uint index = thread_index;
             index < kQueryRows * kLength;
             index += kThreads) {
            const uint row = index / kLength;
            const uint key_position = index - row * kLength;
            if (key_position > query_start + row) {
                scores[index] = metal::numeric_limits<bfloat>::lowest();
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reproduce MLX precise block_softmax for axis size 512 exactly while
        // running four rows concurrently. One physical SIMDgroup owns a row
        // and emulates stock's four virtual SIMDgroups. Each virtual group
        // retains the identical lane/read mapping and simd_max/simd_sum; the
        // second reduction places its four partials in lanes 0...3 before the
        // same SIMD intrinsic. This removes all softmax threadgroup barriers
        // without changing reduction or cast order. Identical topology to the
        // (raw-bit-verified) sliding staged kernel: full-attention score rows
        // are the same 512 elements wide.
        for (uint row_group = 0; row_group < kQueryRows;
             row_group += kSIMDGroups) {
            const uint row = row_group + simd_group;
            const uint row_offset = row * kLength;
            float loaded[kSIMDGroups][kSoftmaxReads];
            float virtual_maxima[kSIMDGroups];

            for (uint virtual_group = 0;
                 virtual_group < kSIMDGroups;
                 ++virtual_group) {
                const uint read_offset =
                    virtual_group * kSIMDSize * kSoftmaxReads
                    + simd_lane * kSoftmaxReads;
                float maximum = metal::numeric_limits<float>::lowest();
                for (uint read = 0; read < kSoftmaxReads; ++read) {
                    const float value = static_cast<float>(
                        scores[row_offset + read_offset + read]);
                    loaded[virtual_group][read] = value;
                    maximum = maximum < value ? value : maximum;
                }
                virtual_maxima[virtual_group] = simd_max(maximum);
            }

            float maximum = simd_lane < kSIMDGroups
                ? virtual_maxima[simd_lane]
                : metal::numeric_limits<float>::lowest();
            maximum = simd_max(maximum);

            float virtual_normalizers[kSIMDGroups];
            for (uint virtual_group = 0;
                 virtual_group < kSIMDGroups;
                 ++virtual_group) {
                float normalizer = 0.0f;
                for (uint read = 0; read < kSoftmaxReads; ++read) {
                    const float exponential = fast::exp(
                        loaded[virtual_group][read] - maximum);
                    loaded[virtual_group][read] = exponential;
                    normalizer += exponential;
                }
                virtual_normalizers[virtual_group] = simd_sum(normalizer);
            }
            float normalizer = simd_lane < kSIMDGroups
                ? virtual_normalizers[simd_lane]
                : 0.0f;
            normalizer = 1.0f / simd_sum(normalizer);

            for (uint virtual_group = 0;
                 virtual_group < kSIMDGroups;
                 ++virtual_group) {
                const uint read_offset =
                    virtual_group * kSIMDSize * kSoftmaxReads
                    + simd_lane * kSoftmaxReads;
                for (uint read = 0; read < kSoftmaxReads; ++read) {
                    scores[row_offset + read_offset + read] =
                        static_cast<bfloat>(
                            loaded[virtual_group][read] * normalizer);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Consume the on-chip BF16 probabilities directly. Four SIMDgroups
        // cover 32 output columns each, in four waves (kHeadDim / 32 = 16
        // value tiles), and write only the final 16x512 attention output to
        // device memory. The PV descriptor (16, 32, 512) is the identical
        // MPP reduction the sliding staged kernel bit-verified.
        //
        // P3 token-major emission: element (row, lane) of the tile is the
        // output for token (query_start + row), head query_head, dimension
        // (value_start + lane). Token-major places it at
        //   (query_start + row) * kQHeads * kHeadDim
        //     + query_head * kHeadDim + value_start + lane
        // (output shape [1, 512, 32*512], row stride 16384); head-major keeps
        // the v1 addressing (output shape [1, 32, 512, 512], row stride 512).
        // Identical computed values, different destination addresses only.
        //
        // P5 PV causal column skip: probability columns at or beyond
        // pv_key_column_limit = 32 * ((query_block + 2) / 2) are causally
        // masked for every row of this query tile, so softmax wrote exactly
        // +0.0 bf16 there (fast::exp underflows the bfloat lowest fill to
        // +0.0f and +0.0f * normalizer stays +0.0f). In the full-width
        // reduction each such column contributes acc += (+0.0) * v, and
        // under IEEE-754 round-to-nearest a signed-zero term can only
        // canonicalize a -0.0 accumulator to +0.0 (when the product is
        // +0.0-signed) -- it can never change a nonzero accumulator. The
        // truncated path reproduces that net effect exactly: it reduces only
        // the causally live prefix into a +0.0f-initialized f32 cooperative
        // accumulator (a +0.0-seeded round-to-nearest accumulation can never
        // produce -0.0), applies the same trailing + 0.0f the skipped
        // columns would have applied, and rounds once to bf16 via
        // static_cast, the same final conversion the full-width run
        // performs. Query blocks without a skipped tail (query_block >= 30)
        // keep the untouched full-width run: the full loop applies no masked
        // terms for them, so no canonicalization may be applied either. The
        // bound derivation is head-dim independent and identical to the
        // sliding staged kernel's. See
        // notes/agent-p5-pv-skip-2026-07-15.md for the exactness argument,
        // including chunk-granularity accumulation.
        constexpr int kOutputRowStride = kTokenMajorOutput
            ? int(kQHeads * kHeadDim)
            : int(kHeadDim);
        constexpr auto pv_descriptor = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 512, false, false, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<
            pv_descriptor, metal::execution_simdgroup> pv;
        constexpr auto pv_truncated_descriptor =
            mpp::tensor_ops::matmul2d_descriptor(
                16, 32, static_cast<int>(metal::dynamic_extent),
                false, false, false,
                mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<
            pv_truncated_descriptor, metal::execution_simdgroup> pv_truncated;
        const uint pv_key_column_limit = kPVTileSkip
            ? ((query_block + 2) / 2) * 32
            : kLength;
        const uint output_tile_base = kTokenMajorOutput
            ? query_start * (kQHeads * kHeadDim) + query_head * kHeadDim
            : (query_head * kLength + query_start) * kHeadDim;
        for (uint value_block = simd_group;
             value_block < kHeadDim / 32;
             value_block += kSIMDGroups) {
            const uint value_start = value_block * 32;
            device bfloat* mutable_values = const_cast<device bfloat*>(values)
                + static_cast<int64_t>(kv_head) * values_strides[1]
                + static_cast<int64_t>(value_start) * values_strides[3];
            device bfloat* output_tile = output
                + output_tile_base
                + value_start;
            \(gemma4StagedPrefillPVSkipGuardLine)
                auto probability_tensor = metal::tensor(
                    scores,
                    metal::dextents<int, 2>{int(pv_key_column_limit), 16},
                    metal::array<int, 2>{1, 512});
                auto value_tensor = metal::tensor(
                    mutable_values,
                    metal::dextents<int, 2>{32, int(pv_key_column_limit)},
                    metal::array<int64_t, 2>{
                        values_strides[3], values_strides[2]});
                auto accumulator =
                    pv_truncated.get_destination_cooperative_tensor<
                        decltype(probability_tensor),
                        decltype(value_tensor),
                        float>();
                #pragma clang loop unroll(full)
                for (uint16_t i = 0; i < accumulator.get_capacity(); ++i) {
                    if (accumulator.is_valid_element(i)) {
                        \(gemma4StagedPrefillPVSkipZeroInitializationLine)
                    }
                }
                pv_truncated.run(
                    probability_tensor, value_tensor, accumulator);
                #pragma clang loop unroll(full)
                for (uint16_t i = 0; i < accumulator.get_capacity(); ++i) {
                    if (accumulator.is_valid_element(i)) {
                        \(gemma4StagedPrefillPVSkipCanonicalizationLine)
                        const auto coords =
                            accumulator.get_multidimensional_index(i);
                        output_tile[coords[0] + coords[1] * kOutputRowStride] =
                            static_cast<bfloat>(canonicalized);
                    }
                }
            } else {
                auto probability_tensor = metal::tensor(
                    scores,
                    metal::dextents<int, 2>{512, 16},
                    metal::array<int, 2>{1, 512});
                auto value_tensor = metal::tensor(
                    mutable_values,
                    metal::dextents<int, 2>{32, 512},
                    metal::array<int64_t, 2>{
                        values_strides[3], values_strides[2]});
                auto output_tensor = metal::tensor(
                    output_tile,
                    metal::dextents<int, 2>{32, 16},
                    metal::array<int, 2>{1, kOutputRowStride});
                pv.run(probability_tensor, value_tensor, output_tensor);
            }
        }
        """

/// Exact-shape MPP kernel for Gemma 4 FULL-attention prefill (P2).
///
/// The ten full-attention layers previously ran MLX's unfused SDPA fallback
/// graph at L=512 (D=512 has no fused kernel: supported full-attention head
/// dims are 64/80/128): a bf16 QK^T matmul materializing [1,4,8,512,512]
/// scores in device memory (~16.8 MB), a `where` causal fill, a precise f32
/// block softmax through device memory, and a PV matmul -- ~100 MB of
/// intermediate device traffic per layer, plus a ~33.6 MB transpose-reshape
/// merge.
///
/// This kernel is the full-attention analog of the promoted staged sliding
/// kernel (`Gemma4StagedPrefillAttention.swift`): one 128-thread threadgroup
/// owns one `(query head, 16-query)` tile. QK writes the causally-live prefix
/// of a 16x512 BF16 score tile to threadgroup memory (fully-masked 32-key
/// tiles are skipped and mask-filled instead -- the P1 derivation is head-dim
/// independent), the same four-SIMD/4-read reduction topology as MLX's
/// precise block softmax for axis 512 rewrites that tile in place, and PV
/// consumes only the causally live prefix of it (P5) without materializing
/// either scores or probabilities in device memory. Output is emitted
/// token-major `[1, 512, 32*512]` (P3), directly in the merged layout
/// `o_proj` consumes.
///
/// Differences from the sliding kernel are geometry only:
/// - GQA 8:1 (`kv_head = query_head / 8`, matching the stock fallback's
///   `unflatten(q, 1, {4, 8})` + broadcast K/V),
/// - QK reduces over head dim 512 (descriptor K=512; the sliding kernel's PV
///   already bit-verified this MPP reduction length against stock matmul),
/// - PV covers 16 32-column output tiles (four waves of four SIMDgroups),
/// - no sliding window: the mask is pure causal, exactly the stock `.causal`
///   fallback path (`q_idx >= k_idx`, offset 0), filled with
///   `bfloat::lowest()` -- the same `finfo(bfloat16).min` the stock `where`
///   uses.
///
/// Threadgroup memory: the score tile is 16x512 bf16 = 16 KB (unchanged from
/// the sliding kernel -- its size depends on rows x keys, not head dim), the
/// only threadgroup allocation. PV accumulation lives in per-SIMDgroup
/// registers (16x32 f32 per in-flight tile), so D=512 costs no extra
/// threadgroup memory; 16 KB is half the 32 KB Apple GPU threadgroup budget,
/// preserving the sliding kernel's occupancy.
///
/// Rollback: honors the same P1/P3/P5 switches as the sliding kernel
/// (`DARKBLOOM_STAGED_PREFILL_CAUSAL_TILE_SKIP`,
/// `DARKBLOOM_STAGED_PREFILL_TOKEN_MAJOR_OUTPUT`,
/// `DARKBLOOM_STAGED_PREFILL_PV_TILE_SKIP`); the whole kernel is gated
/// by `DARKBLOOM_STAGED_FULL_PREFILL_ATTENTION` in the engine.
private let gemma4StagedFullPrefill512Kernel = MLXFast.metalKernel(
    name: gemma4StagedFullPrefill512KernelName,
    inputNames: ["queries", "keys", "values"],
    outputNames: ["output"],
    source: gemma4StagedFullPrefill512KernelSource,
    header: """
        #include <metal_stdlib>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

        """,
    ensureRowContiguous: false
)

/// Runs the staged full-attention prefill kernel on the ranked shape.
///
/// Returns `[1, 512, 32*512]` (token-major, already the merged layout
/// `o_proj` consumes) when `gemma4StagedPrefillTokenMajorOutputEnabled`,
/// otherwise the legacy head-major `[1, 32, 512, 512]` that the caller merges
/// via `transposed(0, 2, 1, 3).reshaped(B, L, -1)`.
func gemma4StagedFullPrefill512(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray
) -> MLXArray {
    precondition(queries.dtype == .bfloat16)
    precondition(keys.dtype == .bfloat16)
    precondition(values.dtype == .bfloat16)
    precondition(queries.shape == [1, 32, 512, 512])
    precondition(keys.shape == [1, 4, 512, 512])
    precondition(values.shape == [1, 4, 512, 512])

    let outputShape: [Int] = gemma4StagedPrefillTokenMajorOutputEnabled
        ? [1, 512, 32 * 512]
        : [1, 32, 512, 512]
    return gemma4StagedFullPrefill512Kernel(
        [queries, keys, values],
        grid: (128, 32, 32),
        threadGroup: (128, 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [.bfloat16]
    )[0]
}
