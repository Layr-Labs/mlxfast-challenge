import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// Correctness-first Laguna runtime (Poolside Laguna XS 2.1, 256-expert MoE).
//
// This module tree closely follows the vendored reference implementation at
// `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Laguna.swift` (`LagunaModel` /
// `LagunaModelInner`), which is the behavior oracle for this port. It is a
// reimplementation rather than a wrapper for two load-bearing reasons:
//
// 1. The Poolside checkpoint stores the MoE router as a raw BF16
//    `mlp.gate.weight` matrix next to the F32
//    `mlp.gate.e_score_correction_bias`, while only expert projections are
//    NVFP4. The runtime mirrors those parameter paths exactly.
// 2. The vendored `LagunaModelInner`/`LagunaDecoderLayer` types are
//    fileprivate and `LagunaConfiguration`'s stored properties are internal
//    to MLXLLM, so the runtime layers (cache geometry, future fast-engine
//    and exact-verification waves) could not reach the internals through a
//    plain wrapper.
//
// All math is expressed with standard MLX ops and the vendored shared
// primitives (`attentionWithCacheUpdate`, `initializeRope`,
// `applyRotaryPosition`, `SwitchGLU`, `weightedExpertSum`, `RMSNorm`,
// `createAttentionMask`). No custom Metal kernels in this increment; the
// fused fast-engine and exact-pair/exact-four style optimizations are a
// later layer on top of this reference target.

func lagunaLastTokenRange(sequenceLength: Int) -> Range<Int>? {
    sequenceLength > 1 ? (sequenceLength - 1)..<sequenceLength : nil
}

func lagunaLastTokenHidden(_ hidden: MLXArray) -> MLXArray {
    guard let range = lagunaLastTokenRange(sequenceLength: hidden.dim(1)) else {
        return hidden
    }
    return hidden[0..., range, 0...]
}

/// Builds the `initializeRope` scaling dictionary for a per-type Laguna RoPE
/// spec. For `default` RoPE only the type is consulted; for YaRN the factory
/// reads factor / original context / betas. The XS config also serializes
/// `attention_factor: 1.0`, but both vendored MLX Laguna implementations
/// intentionally ignore that Hugging Face field. Do not forward it here:
/// leaving MLX's mscale/mscale_all_dim defaults at 1.0/0.0 yields the upstream
/// attention scaling of `0.1 * ln(32) + 1` (~1.34657).
func lagunaRopeScalingConfig(_ spec: LagunaRopeSpec) -> [String: StringOrNumber] {
    var scalingConfig: [String: StringOrNumber] = ["rope_type": .string(spec.type)]
    if spec.type == "yarn" {
        scalingConfig["factor"] = .float(Float(spec.factor))
        scalingConfig["original_max_position_embeddings"] = .int(
            spec.originalMaxPositionEmbeddings)
        scalingConfig["beta_fast"] = .float(Float(spec.betaFast))
        scalingConfig["beta_slow"] = .float(Float(spec.betaSlow))
    }
    return scalingConfig
}

// MARK: - Runtime fusion feature flags

// Each fusion below concatenates the OUTPUT ROWS of same-dtype projections
// that consume the same input. Per-row gemv/qmv/gather-qmv arithmetic is
// independent of which rows share a dispatch (every output row keeps its own
// K-loop and scale application in the original order), so the fused dispatch
// is bit-exact against the separate dispatches it replaces. The per-head
// g_proj (N=64) uses a different split-K gemv variant and is never fused.

/// `DARKBLOOM_FUSED_QKV` (default OFF; set "1" to enable): after checkpoint
/// load, retain one row-concatenated `[Wq; Wk; Wv]` BF16 weight per attention
/// layer and serve Q/K/V from a single projection dispatch. Ablation on the
/// paired local benchmark showed a mild prefill cost with no decode gain, so
/// this ships opt-in.
let lagunaFusedQKVEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_QKV"] == "1"

/// `DARKBLOOM_FUSED_SHARED_GATE_UP` (default on; set "0" to disable): after
/// checkpoint load, retain one row-concatenated NVFP4 `[gate; up]` bank per
/// shared expert and serve single-token decode from one quantized matmul.
/// Multi-token prefill remains on the stock separate banks so the ranked
/// prefill path and its smaller gather/GEMM shapes are unchanged.
let lagunaFusedSharedGateUpEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_SHARED_GATE_UP"] != "0"

/// Decode-only shared-expert NVFP4 QMV + SwiGLU fusion. This consumes the
/// retained row-concatenated `[gate; up]` bank and emits only the 512-wide
/// BF16 activation, preserving the two independent QMV casts and every BF16
/// boundary in the compiled SiLU product.
let lagunaFusedSharedSwiGLUQMVEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_SHARED_SWIGLU_QMV"] != "0"

/// Routed-expert counterpart to the shared QMV + SwiGLU fusion. Each decode
/// request supplies exactly eight current-token expert indices; the kernel
/// reads those banks directly and emits `[1, 1, 8, 1, 512]`.
let lagunaFusedRoutedSwiGLUQMVEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_ROUTED_SWIGLU_QMV"] != "0"

/// Decode-only routed NVFP4 down-QMV plus BF16 router weighting, fixed-order
/// expert reduction, and the Laguna 2.5 routed scale. The custom kernel emits
/// one 2048-wide branch instead of materializing eight expert rows.
let lagunaFusedRoutedDownReduceEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_ROUTED_DOWN_REDUCE"] != "0"

/// `DARKBLOOM_FUSED_ROUTED_GATE_UP` (default on; set "0" to disable): after
/// checkpoint load, retain one row-concatenated NVFP4 `[gate; up]` bank per
/// sparse layer's routed experts and serve single-token decode's gate/up from
/// one gather-QMM dispatch. DECODE-ONLY: the module tree, checkpoint keys,
/// and every multi-token (prefill) forward stay fully stock -- ablation
/// showed the fused bank helps decode (~+1.9%) but badly hurts the M=512
/// sorted gather-GEMM prefill path, so prefill always dispatches the stock
/// separate banks.
let lagunaFusedRoutedGateUpEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_ROUTED_GATE_UP"] != "0"

/// Decode post-attention residual + RMSNorm fusion. The kernel emits
/// both the rounded BF16 residual (needed by the following skip connection)
/// and the normalized row (consumed immediately by the MLP), eliminating a
/// separate residual-add materialization/read. Prefill stays on the stock path
/// to keep its dispatch and scheduling unchanged.
let lagunaFusedResidualRMSNormEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_RESIDUAL_RMS"] != "0"

private let lagunaResidualRMSNormKernel = MLXFast.metalKernel(
    name: "laguna_residual_rms_bf16_2048_v1",
    inputNames: ["residual", "branch", "weight"],
    outputNames: ["summed", "normalized"],
    source: """
        constexpr uint axis_size = 2048;
        constexpr uint n_reads = 4;
        constexpr uint simd_size = 32;

        uint row = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint base = row * axis_size + lid * n_reads;

        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[simd_size];

        thread bfloat values[n_reads];
        float acc = 0.0f;
        for (uint i = 0; i < n_reads; ++i) {
            bfloat value = bfloat(residual[base + i] + branch[base + i]);
            values[i] = value;
            summed[base + i] = value;
            float fv = float(value);
            acc += fv * fv;
        }

        acc = simd_sum(acc);
        if (simd_group == 0) {
            local_sums[simd_lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_lane == 0) {
            local_sums[simd_group] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0) {
            acc = simd_sum(local_sums[simd_lane]);
            if (simd_lane == 0) {
                local_inv_mean[0] =
                    metal::precise::rsqrt(acc / 2048.0f + 1.0e-6f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < n_reads; ++i) {
            normalized[base + i] =
                weight[lid * n_reads + i] *
                bfloat(float(values[i]) * local_inv_mean[0]);
        }
        """,
    ensureRowContiguous: true
)

func lagunaResidualRMSNorm(
    residual: MLXArray, branch: MLXArray, weight: MLXArray
) -> (MLXArray, MLXArray) {
    precondition(residual.dtype == .bfloat16)
    precondition(branch.dtype == .bfloat16)
    precondition(weight.dtype == .bfloat16)
    precondition(residual.shape == branch.shape)
    precondition(residual.dim(-1) == LagunaConstants.hiddenSize)
    precondition(weight.shape == [LagunaConstants.hiddenSize])

    let rows = residual.size / LagunaConstants.hiddenSize
    let outputs = lagunaResidualRMSNormKernel(
        [residual, branch, weight],
        grid: (rows * 512, 1, 1),
        threadGroup: (512, 1, 1),
        outputShapes: [residual.shape, residual.shape],
        outputDTypes: [.bfloat16, .bfloat16]
    )
    return (outputs[0], outputs[1])
}

// MARK: - Attention

/// Keep the stock shapeless unary gate for prefill. Ranked measurement showed
/// the larger gate/product graph regressing the complete prefill schedule even
/// though its isolated steady-state subpath was slightly faster.
private let lagunaCompiledSoftplusGate: @Sendable (MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray) -> MLXArray = { gate in
        softplus(gate.asType(.float32)).asType(gate.dtype)
    }
    return MLXHardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

/// Decode-only outer compilation of the same gate product with the following
/// bias-free BF16 output projection. MLX keeps the matmul primitive intact but
/// schedules the elementwise producer and projection as one compiled graph,
/// avoiding a separate frontend boundary and shortening the gated vector's
/// lifetime. Prefill deliberately uses the smaller gate-only fusion.
private func makeLagunaAttentionGateProjection(
    heads: Int, headDim: Int
) -> @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray {
    let body: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = {
        output, projectedGate, weight in
        let batch = output.dim(0)
        let length = output.dim(1)
        let gate = softplus(projectedGate.asType(.float32)).asType(output.dtype)
        let gated = (
            output.reshaped(batch, length, heads, headDim)
                * gate[.ellipsis, .newAxis]
        ).reshaped(batch, length, heads * headDim)
        return matmul(gated, weight.T)
    }
    return MLXHardwareInfo.isCompiledDecodeSupported ? compile(body) : body
}

private let lagunaFullAttentionGateProjection = makeLagunaAttentionGateProjection(
    heads: LagunaConstants.fullAttentionHeads,
    headDim: LagunaConstants.headDim
)

private let lagunaSlidingAttentionGateProjection = makeLagunaAttentionGateProjection(
    heads: LagunaConstants.slidingAttentionHeads,
    headDim: LagunaConstants.headDim
)

/// Laguna attention: GQA with per-head QK-norm, per-layer-type RoPE (YaRN on
/// full-attention layers over the first half of the head, plain RoPE on
/// sliding layers over the whole head), and per-head softplus output gating.
/// Mirrors the vendored `LagunaAttention` forward exactly.
final class LagunaRuntimeAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let gatingEnabled: Bool
    let gatePerHead: Bool
    let isSliding: Bool
    let attentionGateProjection: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    /// Retained fused `[Wq; Wk; Wv]` weight (output rows concatenated, query
    /// rows first), built once after checkpoint load when
    /// `DARKBLOOM_FUSED_QKV` is enabled. Plain stored property with a leading
    /// underscore so Module reflection never treats this derived layout as a
    /// checkpoint parameter; the q/k/v `Linear` modules keep the original
    /// arrays for parameter integrity.
    var _fusedQKVWeight: MLXArray?

    /// Builds and retains the fused QKV weight from the loaded q/k/v
    /// projection weights. Called once after weights are installed and
    /// evaluated (before warmup); returns the new array so the caller can
    /// batch a single eval. Fuses only the exact stock configuration: three
    /// plain bias-free `Linear` projections of one dtype over the same input
    /// width, so the fused matmul is `matmul(x, w.T)` with every original
    /// output row unchanged.
    func prepareFusedQKVWeight() -> MLXArray? {
        guard _fusedQKVWeight == nil,
            type(of: wq) == Linear.self,
            type(of: wk) == Linear.self,
            type(of: wv) == Linear.self,
            wq.bias == nil, wk.bias == nil, wv.bias == nil,
            wq.weight.ndim == 2, wk.weight.ndim == 2, wv.weight.ndim == 2,
            wq.weight.dtype == wk.weight.dtype,
            wk.weight.dtype == wv.weight.dtype,
            wq.weight.dim(1) == wk.weight.dim(1),
            wk.weight.dim(1) == wv.weight.dim(1),
            wq.weight.dim(0) == nHeads * headDim,
            wk.weight.dim(0) == nKVHeads * headDim,
            wv.weight.dim(0) == nKVHeads * headDim
        else {
            return nil
        }
        let fused = concatenated([wq.weight, wk.weight, wv.weight], axis: 0)
        _fusedQKVWeight = fused
        return fused
    }

    init(_ config: LagunaConfig, layerIdx: Int) {
        let dim = config.hiddenSize
        self.nHeads = config.heads(forLayer: layerIdx)
        self.nKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)
        let layerGating = config.gatingMode(forLayer: layerIdx)
        self.gatingEnabled = layerGating.enabled
        self.gatePerHead = layerGating.isPerHead

        let layerType = config.layerType(forLayer: layerIdx)
        self.isSliding = layerType == .sliding
        self.attentionGateProjection =
            layerType == .sliding
            ? lagunaSlidingAttentionGateProjection
            : lagunaFullAttentionGateProjection

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: config.qkvBias)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: config.qkvBias)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: config.qkvBias)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: config.attentionBias)

        if gatingEnabled {
            let gateDim = gatePerHead ? nHeads : nHeads * headDim
            self._gProj.wrappedValue = Linear(dim, gateDim, bias: false)
        }

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))

        let ropeSpec = config.rope(for: layerType)
        let ropeDims = Int(Float(headDim) * Float(ropeSpec.partialRotaryFactor))
        self.rope = initializeRope(
            dims: ropeDims,
            base: Float(ropeSpec.theta),
            traditional: false,
            scalingConfig: lagunaRopeScalingConfig(ropeSpec),
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries: MLXArray
        var keys: MLXArray
        var values: MLXArray
        if let fusedQKVWeight = _fusedQKVWeight {
            // One dispatch over the row-concatenated [Wq; Wk; Wv] weight,
            // identical math to the three bias-free `Linear` calls
            // (`matmul(x, w.T)`). Each output row's K-loop is independent of
            // which rows share the dispatch, so every Q/K/V element is
            // bit-exact; the slices are views and the reshapes below may
            // copy, which does not change values.
            let qkv = matmul(x, fusedQKVWeight.T)
            let queryDim = nHeads * headDim
            let kvDim = nKVHeads * headDim
            queries = qkv[.ellipsis, 0 ..< queryDim]
            keys = qkv[.ellipsis, queryDim ..< (queryDim + kvDim)]
            values = qkv[.ellipsis, (queryDim + kvDim) ..< (queryDim + 2 * kvDim)]
        } else {
            queries = wq(x)
            keys = wk(x)
            values = wv(x)
        }

        queries = qNorm(queries.reshaped(B, L, nHeads, headDim)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, nKVHeads, headDim)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        if gatingEnabled, let gProj {
            // Per-head softplus gate computed in float32, then broadcast
            // across the head dimension (or applied elementwise for a
            // per-element gate).
            let projectedGate = gProj(x)
            if gatePerHead && projectedGate.dtype == output.dtype,
                L == 1, wo.bias == nil, MLXHardwareInfo.isCompiledDecodeSupported
            {
                return attentionGateProjection(output, projectedGate, wo.weight)
            }
            let gate =
                gatePerHead && projectedGate.dtype == output.dtype
                ? lagunaCompiledSoftplusGate(projectedGate)
                : softplus(projectedGate.asType(.float32)).asType(output.dtype)
            if gatePerHead {
                output =
                    (output.reshaped(B, L, nHeads, headDim) * gate[.ellipsis, .newAxis])
                    .reshaped(B, L, -1)
            } else {
                output = output * gate
            }
        }

        return wo(output)
    }

    /// Prefill-only final-layer attention when the caller consumes just the
    /// last hidden row. K/V and the cache update still cover every supplied
    /// token. Q projection, Q normalization, Q RoPE, SDPA, and the output
    /// gate/projection run only for the last query; its RoPE offset is advanced
    /// by the discarded query-row count so it remains at the supplied
    /// sequence's final absolute position.
    func callLastPrefillRow(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        precondition(L > 1)

        let lastInput = lagunaLastTokenHidden(x)
        var queries = wq(lastInput)
        var keys = wk(x)
        var values = wv(x)

        queries = qNorm(queries.reshaped(B, 1, nHeads, headDim)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, nKVHeads, headDim)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)

        if let offsetArray = graphOffsetArray(for: cache) {
            queries = rope(queries, offset: offsetArray + Int32(L - 1))
        } else {
            queries = rope(queries, offset: (cache?.offset ?? 0) + L - 1)
        }
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: .causal
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, 1, -1)

        if gatingEnabled, let gProj {
            let projectedGate = gProj(lastInput)
            let gate =
                gatePerHead && projectedGate.dtype == output.dtype
                ? lagunaCompiledSoftplusGate(projectedGate)
                : softplus(projectedGate.asType(.float32)).asType(output.dtype)
            if gatePerHead {
                output =
                    (output.reshaped(B, 1, nHeads, headDim) * gate[.ellipsis, .newAxis])
                    .reshaped(B, 1, -1)
            } else {
                output = output * gate
            }
        }

        return wo(output)
    }
}

// MARK: - Dense MLP (also used as the shared expert)

private let lagunaSharedSwiGLUQMVHeader = """
    static inline float laguna_nvfp4_scale(uint8_t bits) {
        ushort raw = ushort(bits & 127) << 7;
        half converted = as_type<half>(raw);
        converted *= 256.0;
        half signed_value = (bits & 128) ? -converted : converted;
        return float(signed_value);
    }

    static inline float laguna_nvfp4_qdot_16(
        const device uint8_t* weight,
        const thread float* input,
        float scale
    ) {
        float accum = 0.0f;
        const device uint2* packed = (const device uint2*)weight;
        const uint2 codes = packed[0];
        for (uint j = 0; j < 2; ++j) {
            const uint c = (j == 0) ? codes.x : codes.y;
            const uint p0 =
                ((c & 0x00070007u) << 9) | ((c & 0x00080008u) << 12);
            const uint p1 =
                ((c & 0x00700070u) << 5) | ((c & 0x00800080u) << 8);
            const uint p2 =
                ((c & 0x07000700u) << 1) | ((c & 0x08000800u) << 4);
            const uint p3 =
                ((c & 0x70007000u) >> 3) | (c & 0x80008000u);
            const float2 v04 = float2(as_type<half2>(p0)) * 16384.0f;
            const float2 v15 = float2(as_type<half2>(p1)) * 16384.0f;
            const float2 v26 = float2(as_type<half2>(p2)) * 16384.0f;
            const float2 v37 = float2(as_type<half2>(p3)) * 16384.0f;
            accum +=
                (input[8 * j] * v04.x +
                 input[8 * j + 1] * v15.x +
                 input[8 * j + 2] * v26.x +
                 input[8 * j + 3] * v37.x);
            accum +=
                (input[8 * j + 4] * v04.y +
                 input[8 * j + 5] * v15.y +
                 input[8 * j + 6] * v26.y +
                 input[8 * j + 7] * v37.y);
        }
        return scale * accum;
    }
    """

private let lagunaSharedSwiGLUQMVKernel = MLXFast.metalKernel(
    name: "laguna_shared_nvfp4_swiglu_qmv_bf16_v1",
    inputNames: ["input", "fused_weight", "fused_scales"],
    outputNames: ["activated"],
    source: """
        constexpr uint input_width = 2048;
        constexpr uint output_width = 512;
        constexpr uint fused_width = 1024;
        constexpr uint packed_row_bytes = 1024;
        constexpr uint scale_row_bytes = 128;
        constexpr uint block_width = 512;
        constexpr uint values_per_lane = 16;

        uint tile = threadgroup_position_in_grid.x;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint first_row = tile * 4 + simd_group * 2;

        thread float gate_result[2] = {0.0f, 0.0f};
        thread float up_result[2] = {0.0f, 0.0f};
        thread float input_values[values_per_lane];

        for (uint block = 0; block < input_width; block += block_width) {
            const device vec<bfloat, 4>* input_vectors =
                (const device vec<bfloat, 4>*)(
                    input + block + lane * values_per_lane);
            for (uint i = 0; i < values_per_lane / 4; ++i) {
                const vec<bfloat, 4> values = input_vectors[i];
                input_values[4 * i] = values[0];
                input_values[4 * i + 1] = values[1];
                input_values[4 * i + 2] = values[2];
                input_values[4 * i + 3] = values[3];
            }

            for (uint row = 0; row < 2; ++row) {
                uint gate_row = first_row + row;
                uint up_row = gate_row + output_width;
                const device uint8_t* gate_weight =
                    (const device uint8_t*)fused_weight +
                    gate_row * packed_row_bytes + block / 2 + lane * 8;
                const device uint8_t* up_weight =
                    (const device uint8_t*)fused_weight +
                    up_row * packed_row_bytes + block / 2 + lane * 8;
                const device uint8_t* gate_scale =
                    fused_scales + gate_row * scale_row_bytes +
                    block / 16 + lane;
                const device uint8_t* up_scale =
                    fused_scales + up_row * scale_row_bytes +
                    block / 16 + lane;

                gate_result[row] += laguna_nvfp4_qdot_16(
                    gate_weight,
                    input_values,
                    laguna_nvfp4_scale(gate_scale[0]));
                up_result[row] += laguna_nvfp4_qdot_16(
                    up_weight,
                    input_values,
                    laguna_nvfp4_scale(up_scale[0]));
            }
        }

        for (uint row = 0; row < 2; ++row) {
            gate_result[row] = simd_sum(gate_result[row]);
            up_result[row] = simd_sum(up_result[row]);
            if (lane == 0) {
                bfloat gate = bfloat(gate_result[row]);
                bfloat up = bfloat(up_result[row]);
                bfloat exp_abs = metal::exp(metal::abs(gate));
                bfloat denominator = bfloat(1) + exp_abs;
                bfloat y = bfloat(1) / denominator;
                bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
                bfloat silu = bfloat(gate * sigmoid);
                activated[first_row + row] = bfloat(silu * up);
            }
        }
        """,
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)

func lagunaSharedSwiGLUQMV(
    _ input: MLXArray,
    fusedWeight: MLXArray,
    fusedScales: MLXArray
) -> MLXArray {
    precondition(input.dtype == .bfloat16)
    precondition(input.shape == [1, 1, LagunaConstants.hiddenSize])
    precondition(fusedWeight.dtype == .uint32)
    precondition(
        fusedWeight.shape == [
            2 * LagunaConstants.sharedExpertIntermediateSize,
            LagunaConstants.hiddenSize / 8,
        ])
    precondition(fusedScales.dtype == .uint8)
    precondition(
        fusedScales.shape == [
            2 * LagunaConstants.sharedExpertIntermediateSize,
            LagunaConstants.hiddenSize / 16,
        ])

    return lagunaSharedSwiGLUQMVKernel(
        [input, fusedWeight, fusedScales],
        grid: (128 * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.sharedExpertIntermediateSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

private let lagunaRoutedSwiGLUQMVKernel = MLXFast.metalKernel(
    name: "laguna_routed_nvfp4_swiglu_qmv_bf16_v1",
    inputNames: ["input", "fused_weight", "fused_scales", "indices"],
    outputNames: ["activated"],
    source: """
        constexpr uint input_width = 2048;
        constexpr uint output_width = 512;
        constexpr uint fused_width = 1024;
        constexpr uint packed_row_bytes = 1024;
        constexpr uint scale_row_bytes = 128;
        constexpr uint packed_expert_bytes = fused_width * packed_row_bytes;
        constexpr uint scale_expert_bytes = fused_width * scale_row_bytes;
        constexpr uint block_width = 512;
        constexpr uint values_per_lane = 16;
        constexpr uint tiles_per_expert = 128;

        uint group = threadgroup_position_in_grid.x;
        uint expert_slot = group / tiles_per_expert;
        uint tile = group % tiles_per_expert;
        uint expert = uint(indices[expert_slot]);
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint first_row = tile * 4 + simd_group * 2;

        const device uint8_t* expert_weight =
            (const device uint8_t*)fused_weight +
            expert * packed_expert_bytes;
        const device uint8_t* expert_scales =
            fused_scales + expert * scale_expert_bytes;

        thread float gate_result[2] = {0.0f, 0.0f};
        thread float up_result[2] = {0.0f, 0.0f};
        thread float input_values[values_per_lane];

        for (uint block = 0; block < input_width; block += block_width) {
            const device vec<bfloat, 4>* input_vectors =
                (const device vec<bfloat, 4>*)(
                    input + block + lane * values_per_lane);
            for (uint i = 0; i < values_per_lane / 4; ++i) {
                const vec<bfloat, 4> values = input_vectors[i];
                input_values[4 * i] = values[0];
                input_values[4 * i + 1] = values[1];
                input_values[4 * i + 2] = values[2];
                input_values[4 * i + 3] = values[3];
            }

            for (uint row = 0; row < 2; ++row) {
                uint gate_row = first_row + row;
                uint up_row = gate_row + output_width;
                const device uint8_t* gate_weight =
                    expert_weight + gate_row * packed_row_bytes +
                    block / 2 + lane * 8;
                const device uint8_t* up_weight =
                    expert_weight + up_row * packed_row_bytes +
                    block / 2 + lane * 8;
                const device uint8_t* gate_scale =
                    expert_scales + gate_row * scale_row_bytes +
                    block / 16 + lane;
                const device uint8_t* up_scale =
                    expert_scales + up_row * scale_row_bytes +
                    block / 16 + lane;

                gate_result[row] += laguna_nvfp4_qdot_16(
                    gate_weight,
                    input_values,
                    laguna_nvfp4_scale(gate_scale[0]));
                up_result[row] += laguna_nvfp4_qdot_16(
                    up_weight,
                    input_values,
                    laguna_nvfp4_scale(up_scale[0]));
            }
        }

        for (uint row = 0; row < 2; ++row) {
            gate_result[row] = simd_sum(gate_result[row]);
            up_result[row] = simd_sum(up_result[row]);
            if (lane == 0) {
                bfloat gate = bfloat(gate_result[row]);
                bfloat up = bfloat(up_result[row]);
                bfloat exp_abs = metal::exp(metal::abs(gate));
                bfloat denominator = bfloat(1) + exp_abs;
                bfloat y = bfloat(1) / denominator;
                bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
                bfloat silu = bfloat(gate * sigmoid);
                activated[
                    expert_slot * output_width + first_row + row
                ] = bfloat(silu * up);
            }
        }
        """,
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)

func lagunaRoutedSwiGLUQMV(
    _ input: MLXArray,
    fusedWeight: MLXArray,
    fusedScales: MLXArray,
    indices: MLXArray
) -> MLXArray {
    precondition(input.dtype == .bfloat16)
    precondition(input.shape == [1, 1, LagunaConstants.hiddenSize])
    precondition(fusedWeight.dtype == .uint32)
    precondition(
        fusedWeight.shape == [
            LagunaConstants.numExperts,
            2 * LagunaConstants.moeIntermediateSize,
            LagunaConstants.hiddenSize / 8,
        ])
    precondition(fusedScales.dtype == .uint8)
    precondition(
        fusedScales.shape == [
            LagunaConstants.numExperts,
            2 * LagunaConstants.moeIntermediateSize,
            LagunaConstants.hiddenSize / 16,
        ])
    precondition(indices.dtype == .uint32)
    precondition(indices.shape == [1, 1, LagunaConstants.numExpertsPerTok])

    return lagunaRoutedSwiGLUQMVKernel(
        [input, fusedWeight, fusedScales, indices],
        grid: (LagunaConstants.numExpertsPerTok * 128 * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[
            1, 1, LagunaConstants.numExpertsPerTok, 1,
            LagunaConstants.moeIntermediateSize,
        ]],
        outputDTypes: [.bfloat16]
    )[0]
}

// Decode-only Laguna XS routed down projection. The stock graph materializes
// eight 2048-wide BF16 expert outputs, casts eight FP32 router weights to
// BF16, multiplies, reduces the expert axis, then applies the fixed BF16 2.5
// routed scale. This kernel preserves each of those arithmetic boundaries but
// keeps the eight expert rows in threadgroup memory and emits only the final
// 2048-wide routed branch.
private let lagunaRoutedDownReduceKernel = MLXFast.metalKernel(
    name: "laguna_routed_nvfp4_down_reduce_bf16_v1",
    inputNames: [
        "activated", "down_weight", "down_scales", "indices", "router_weights",
    ],
    outputNames: ["routed"],
    source: """
        constexpr uint input_width = 512;
        constexpr uint output_width = 2048;
        constexpr uint experts_per_token = 8;
        constexpr uint outputs_per_simd = 4;
        constexpr uint values_per_lane = 16;
        constexpr uint packed_row_bytes = 256;
        constexpr uint scale_row_bytes = 32;
        constexpr uint packed_expert_bytes =
            output_width * packed_row_bytes;
        constexpr uint scale_expert_bytes =
            output_width * scale_row_bytes;

        uint tile = threadgroup_position_in_grid.x;
        uint expert_slot = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint first_row = tile * outputs_per_simd;
        uint expert = uint(indices[expert_slot]);

        const device bfloat* expert_input =
            activated + expert_slot * input_width;
        const device uint8_t* expert_weight =
            (const device uint8_t*)down_weight +
            expert * packed_expert_bytes;
        const device uint8_t* expert_scales =
            down_scales + expert * scale_expert_bytes;

        thread float input_values[values_per_lane];
        const device vec<bfloat, 4>* input_vectors =
            (const device vec<bfloat, 4>*)(
                expert_input + lane * values_per_lane);
        for (uint i = 0; i < values_per_lane / 4; ++i) {
            const vec<bfloat, 4> values = input_vectors[i];
            input_values[4 * i] = values[0];
            input_values[4 * i + 1] = values[1];
            input_values[4 * i + 2] = values[2];
            input_values[4 * i + 3] = values[3];
        }

        thread float result[outputs_per_simd] = {
            0.0f, 0.0f, 0.0f, 0.0f
        };
        for (uint row = 0; row < outputs_per_simd; ++row) {
            uint output_row = first_row + row;
            const device uint8_t* weight =
                expert_weight + output_row * packed_row_bytes + lane * 8;
            const device uint8_t* scale =
                expert_scales + output_row * scale_row_bytes + lane;
            result[row] = laguna_nvfp4_qdot_16(
                weight,
                input_values,
                laguna_nvfp4_scale(scale[0]));
            result[row] = simd_sum(result[row]);
        }

        threadgroup bfloat expert_outputs[
            experts_per_token * outputs_per_simd
        ];
        if (lane == 0) {
            for (uint row = 0; row < outputs_per_simd; ++row) {
                expert_outputs[
                    expert_slot * outputs_per_simd + row
                ] = bfloat(result[row]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // `weightedExpertSum` first multiplies BF16 expert outputs by router
        // weights cast from FP32 to BF16. Its small strided BF16 reduction
        // initializes with zero, then visits expert slots 0 through 7 in
        // order. The scalar 2.5 is constructed in the BF16 result dtype.
        if (expert_slot == 0 && lane < outputs_per_simd) {
            bfloat total = bfloat(0);
            for (uint slot = 0; slot < experts_per_token; ++slot) {
                bfloat route_weight = bfloat(router_weights[slot]);
                bfloat product = bfloat(
                    expert_outputs[slot * outputs_per_simd + lane] *
                    route_weight);
                total = bfloat(product + total);
            }
            routed[first_row + lane] = bfloat(total * bfloat(2.5f));
        }
        """,
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)

func lagunaRoutedDownReduce(
    _ activated: MLXArray,
    downWeight: MLXArray,
    downScales: MLXArray,
    indices: MLXArray,
    routerWeights: MLXArray
) -> MLXArray {
    precondition(activated.dtype == .bfloat16)
    precondition(
        activated.shape == [
            1, 1, LagunaConstants.numExpertsPerTok, 1,
            LagunaConstants.moeIntermediateSize,
        ])
    precondition(downWeight.dtype == .uint32)
    precondition(
        downWeight.shape == [
            LagunaConstants.numExperts,
            LagunaConstants.hiddenSize,
            LagunaConstants.moeIntermediateSize / 8,
        ])
    precondition(downScales.dtype == .uint8)
    precondition(
        downScales.shape == [
            LagunaConstants.numExperts,
            LagunaConstants.hiddenSize,
            LagunaConstants.moeIntermediateSize / 16,
        ])
    precondition(indices.dtype == .uint32)
    precondition(indices.shape == [1, 1, LagunaConstants.numExpertsPerTok])
    precondition(routerWeights.dtype == .float32)
    precondition(routerWeights.shape == [1, 1, LagunaConstants.numExpertsPerTok])

    return lagunaRoutedDownReduceKernel(
        [activated, downWeight, downScales, indices, routerWeights],
        grid: ((LagunaConstants.hiddenSize / 4) * 256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.hiddenSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

final class LagunaRuntimeMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    /// Retained fused NVFP4 `[gate; up]` layout (gate output rows first),
    /// built once after checkpoint load for the shared expert when
    /// `DARKBLOOM_FUSED_SHARED_GATE_UP` is enabled. Plain stored properties
    /// with a leading underscore so Module reflection never treats the
    /// derived layout as checkpoint parameters; the quantized gate/up
    /// modules keep the original arrays for parameter integrity. Never set
    /// on the dense (BF16) layer-0 MLP.
    var _fusedGateUpWeight: MLXArray?
    var _fusedGateUpScales: MLXArray?
    var _fusedGateUpSplit: Int = 0

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    /// Builds and retains the fused gate/up NVFP4 bank from the loaded
    /// shared-expert projections. Called once after weights are installed
    /// and evaluated (before warmup); returns the new arrays so the caller
    /// can batch a single eval. Fuses only the exact stock shared-expert
    /// configuration: two bias-free NVFP4 group-16 4-bit `QuantizedLinear`
    /// projections with identical packed shapes and no affine biases.
    func prepareFusedSharedGateUp() -> [MLXArray] {
        guard _fusedGateUpWeight == nil, _fusedGateUpScales == nil,
            let gate = gateProj as? QuantizedLinear,
            let up = upProj as? QuantizedLinear,
            type(of: gate) == QuantizedLinear.self,
            type(of: up) == QuantizedLinear.self,
            gate.mode == .nvfp4, up.mode == .nvfp4,
            gate.groupSize == 16, up.groupSize == 16,
            gate.bits == 4, up.bits == 4,
            gate.bias == nil, up.bias == nil,
            gate.biases == nil, up.biases == nil,
            gate.weight.ndim == 2, up.weight.ndim == 2,
            gate.weight.dtype == .uint32, up.weight.dtype == .uint32,
            gate.scales.ndim == 2, up.scales.ndim == 2,
            gate.scales.dtype == .uint8, up.scales.dtype == .uint8,
            gate.weight.shape == up.weight.shape,
            gate.scales.shape == up.scales.shape,
            gate.scales.dim(0) == gate.weight.dim(0),
            gate.weight.dim(1) * 8 == gate.scales.dim(1) * 16
        else {
            return []
        }
        let fusedWeight = concatenated([gate.weight, up.weight], axis: 0)
        let fusedScales = concatenated([gate.scales, up.scales], axis: 0)
        _fusedGateUpWeight = fusedWeight
        _fusedGateUpScales = fusedScales
        _fusedGateUpSplit = gate.weight.dim(0)
        return [fusedWeight, fusedScales]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if x.dim(1) == 1,
            let fusedWeight = _fusedGateUpWeight, let fusedScales = _fusedGateUpScales
        {
            if lagunaFusedSharedSwiGLUQMVEnabled,
                x.dtype == .bfloat16,
                x.shape == [1, 1, LagunaConstants.hiddenSize],
                fusedWeight.dtype == .uint32,
                fusedScales.dtype == .uint8,
                _fusedGateUpSplit == LagunaConstants.sharedExpertIntermediateSize
            {
                return downProj(
                    lagunaSharedSwiGLUQMV(
                        x,
                        fusedWeight: fusedWeight,
                        fusedScales: fusedScales
                    )
                )
            }

            // One NVFP4 dispatch over the row-concatenated [gate; up] bank,
            // mirroring `QuantizedLinear.callAsFunction` exactly (transpose,
            // group 16, 4-bit, .nvfp4, no affine biases, no bias add; the
            // guards in `prepareFusedSharedGateUp` pin those literals). Each
            // quantized output row is computed independently, so the split
            // halves are bit-exact vs. the separate gate/up dispatches.
            let gateUp = MLX.quantizedMM(
                x,
                fusedWeight,
                scales: fusedScales,
                biases: nil,
                transpose: true,
                groupSize: 16,
                bits: 4,
                mode: .nvfp4
            )
            let gate = gateUp[.ellipsis, 0 ..< _fusedGateUpSplit]
            let up = gateUp[.ellipsis, _fusedGateUpSplit...]
            return downProj(compiledSiluProduct(gate, up))
        }
        return downProj(compiledSiluProduct(gateProj(x), upProj(x)))
    }
}

// MARK: - MoE

/// Decode-only router post-processing. The stock path materializes sigmoid
/// scores, corrected choice scores, their negation, a full 256-entry argsort,
/// and a gather before retaining just eight entries. This fixed-shape kernel
/// computes the same FP32 sigmoid values and stable choice order, then emits
/// only the selected indices and their *uncorrected* scores. Normalization
/// deliberately remains on the stock MLX reduction below so its accumulation
/// order and rounding are unchanged.
private let lagunaDecodeRouterTop8Kernel = MLXFast.metalKernel(
    name: "laguna_decode_router_top8_v2",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: """
        uint lane = thread_position_in_threadgroup.x;

        threadgroup float scores[256];
        threadgroup float choice_keys[256];
        threadgroup uint expert_indices[256];

        float x = float(logits[lane]);
        float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
        float score = x < 0.0f ? y : 1.0f - y;
        scores[lane] = score;
        float corrected = score + float(correction_bias[lane]);
        choice_keys[lane] = -corrected;
        expert_indices[lane] = lane;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A total order (choice key, then original expert index) makes this
        // network match the stock stable merge sort even for exact ties,
        // signed zero, and NaNs. The lower half of each final sequence keeps
        // the better entries, so ranks 0..<8 are the desired top experts.
        for (uint sequence = 2; sequence <= 256; sequence <<= 1) {
            for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
                uint partner = lane ^ stride;
                if (partner > lane) {
                    float a_key = choice_keys[lane];
                    uint a_index = expert_indices[lane];
                    float a_score = scores[lane];
                    float b_key = choice_keys[partner];
                    uint b_index = expert_indices[partner];
                    float b_score = scores[partner];

                    bool lower_wants_better = (lane & sequence) == 0;
                    bool b_before_a = laguna_router_key_before(
                        b_key, b_index, a_key, a_index);
                    bool a_before_b = laguna_router_key_before(
                        a_key, a_index, b_key, b_index);
                    bool swap = lower_wants_better ? b_before_a : a_before_b;
                    if (swap) {
                        choice_keys[lane] = b_key;
                        expert_indices[lane] = b_index;
                        scores[lane] = b_score;
                        choice_keys[partner] = a_key;
                        expert_indices[partner] = a_index;
                        scores[partner] = a_score;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        if (lane < 8) {
            router_indices[lane] = expert_indices[lane];
            router_scores[lane] = scores[lane];
        }
        """,
    header: """
        METAL_FUNC bool laguna_router_key_before(
            float a, uint a_index, float b, uint b_index) {
            bool a_nan = metal::isnan(a);
            bool b_nan = metal::isnan(b);
            if (a_nan | b_nan) {
                if (a_nan != b_nan) {
                    return !a_nan;
                }
                return a_index < b_index;
            }
            if (a < b) {
                return true;
            }
            if (b < a) {
                return false;
            }
            return a_index < b_index;
        }
        """,
    ensureRowContiguous: true
)

private func lagunaDecodeRouterTop8(
    logits: MLXArray, correctionBias: MLXArray
) -> (MLXArray, MLXArray) {
    precondition(logits.dtype == .float32)
    precondition(correctionBias.dtype == .float32)
    precondition(logits.size == 256)
    precondition(correctionBias.size == 256)

    let outputs = lagunaDecodeRouterTop8Kernel(
        [logits, correctionBias],
        grid: (256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, 1, 8], [1, 1, 8]],
        outputDTypes: [.uint32, .float32]
    )
    return (outputs[0], outputs[1])
}

/// Default-on after same-binary bitwise checks over smooth, tied, and extreme
/// rows plus a 39-stage compiled latency probe. Set
/// `DARKBLOOM_FUSED_ROUTER=0` for a stock-path ablation.
private let lagunaDecodeRouterTop8Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_ROUTER"] != "0"

/// Sigmoid top-k router. The routing math mirrors the vendored
/// `LagunaMoEGate` exactly (sigmoid scores, correction bias added only for
/// expert CHOICE, mixture weights taken from the pre-bias scores, optional
/// top-k renormalization).
final class LagunaRuntimeMoEGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let routerLogitSoftcapping: Float

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: LagunaConfig) {
        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.routerLogitSoftcapping = Float(config.moeRouterLogitSoftcapping)
        self._weight.wrappedValue = zeros([config.numExperts, config.hiddenSize])
        self._eScoreCorrectionBias.wrappedValue = zeros([config.numExperts])
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var logits = x.matmul(weight.T).asType(.float32)
        if routerLogitSoftcapping > 0 {
            logits = tanh(logits / routerLogitSoftcapping) * routerLogitSoftcapping
        }

        let inds: MLXArray
        var weights: MLXArray
        if lagunaDecodeRouterTop8Enabled,
            logits.size == 256, topK == 8,
            eScoreCorrectionBias.size == 256
        {
            (inds, weights) = lagunaDecodeRouterTop8(
                logits: logits,
                correctionBias: eScoreCorrectionBias.asType(.float32)
            )
        } else {
            let scores = sigmoid(logits)
            let scoresForChoice = scores + eScoreCorrectionBias.asType(scores.dtype)
            inds =
                argPartition(-scoresForChoice, kth: topK - 1, axis: -1)[
                    .ellipsis, ..<topK]
            weights = takeAlong(scores, inds, axis: -1)
        }
        if normTopkProb {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
        return (inds, weights)
    }
}

final class LagunaRuntimeSparseMoEBlock: Module, UnaryLayer {
    let routedScalingFactor: Float

    @ModuleInfo(key: "gate") var gate: LagunaRuntimeMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: LagunaRuntimeMLP

    /// Retained fused NVFP4 `[gate; up]` routed-expert banks (per-expert
    /// output rows concatenated, gate rows first), built once after
    /// checkpoint load when `DARKBLOOM_FUSED_ROUTED_GATE_UP` is enabled, plus
    /// a reference to the stock `switch_mlp.down_proj` module for the fused
    /// decode path. Plain stored properties with a leading underscore so
    /// Module reflection never treats the derived layout as checkpoint
    /// parameters or a second child module; `switchMLP` keeps the original
    /// separate banks (they still serve every multi-token forward and
    /// parameter integrity).
    var _fusedRoutedGateUpWeight: MLXArray?
    var _fusedRoutedGateUpScales: MLXArray?
    var _fusedRoutedGateUpSplit: Int = 0
    var _routedDownProj: SwitchLinear?
    var _routedDownWeight: MLXArray?
    var _routedDownScales: MLXArray?

    /// Builds and retains the fused routed gate/up NVFP4 banks from the
    /// loaded stock `SwitchGLU` submodules (reached through the public
    /// `children()`/`parameters()` Module APIs). Called once after weights
    /// are installed and evaluated (before warmup); returns the new arrays
    /// so the caller can batch a single eval. Fuses only the exact stock
    /// configuration: two bias-free NVFP4 group-16 4-bit
    /// `QuantizedSwitchLinear` banks with identical packed shapes.
    func prepareFusedRoutedGateUp() -> [MLXArray] {
        guard _fusedRoutedGateUpWeight == nil, _fusedRoutedGateUpScales == nil else {
            return []
        }
        let children = Dictionary(uniqueKeysWithValues: switchMLP.children().flattened())
        guard let gateModule = children["gate_proj"] as? QuantizedSwitchLinear,
            let upModule = children["up_proj"] as? QuantizedSwitchLinear,
            let downModule = children["down_proj"] as? QuantizedSwitchLinear,
            type(of: gateModule) == QuantizedSwitchLinear.self,
            type(of: upModule) == QuantizedSwitchLinear.self,
            type(of: downModule) == QuantizedSwitchLinear.self,
            gateModule.mode == .nvfp4, upModule.mode == .nvfp4,
            downModule.mode == .nvfp4,
            gateModule.groupSize == 16, upModule.groupSize == 16,
            downModule.groupSize == 16,
            gateModule.bits == 4, upModule.bits == 4, downModule.bits == 4
        else {
            return []
        }
        let gateParams = Dictionary(uniqueKeysWithValues: gateModule.parameters().flattened())
        let upParams = Dictionary(uniqueKeysWithValues: upModule.parameters().flattened())
        let downParams = Dictionary(uniqueKeysWithValues: downModule.parameters().flattened())
        guard let gateWeight = gateParams["weight"], let gateScales = gateParams["scales"],
            let upWeight = upParams["weight"], let upScales = upParams["scales"],
            let downWeight = downParams["weight"],
            let downScales = downParams["scales"],
            gateParams["bias"] == nil, gateParams["biases"] == nil,
            upParams["bias"] == nil, upParams["biases"] == nil,
            downParams["bias"] == nil, downParams["biases"] == nil,
            gateWeight.ndim == 3, upWeight.ndim == 3,
            gateScales.ndim == 3, upScales.ndim == 3,
            downWeight.ndim == 3, downScales.ndim == 3,
            gateWeight.dtype == .uint32, upWeight.dtype == .uint32,
            gateScales.dtype == .uint8, upScales.dtype == .uint8,
            downWeight.dtype == .uint32, downScales.dtype == .uint8,
            gateWeight.shape == upWeight.shape,
            gateScales.shape == upScales.shape,
            gateScales.dim(0) == gateWeight.dim(0),
            gateScales.dim(1) == gateWeight.dim(1),
            gateWeight.dim(2) * 8 == gateScales.dim(2) * 16,
            downWeight.dim(0) == gateWeight.dim(0),
            downScales.dim(0) == downWeight.dim(0),
            downScales.dim(1) == downWeight.dim(1),
            downWeight.dim(2) * 8 == downScales.dim(2) * 16
        else {
            return []
        }
        let fusedWeight = concatenated([gateWeight, upWeight], axis: 1)
        let fusedScales = concatenated([gateScales, upScales], axis: 1)
        _fusedRoutedGateUpWeight = fusedWeight
        _fusedRoutedGateUpScales = fusedScales
        _fusedRoutedGateUpSplit = gateWeight.dim(1)
        _routedDownProj = downModule
        _routedDownWeight = downWeight
        _routedDownScales = downScales
        return [fusedWeight, fusedScales]
    }

    init(_ config: LagunaConfig) {
        self.routedScalingFactor = Float(config.moeRoutedScalingFactor)
        self._gate.wrappedValue = LagunaRuntimeMoEGate(config)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts
        )
        self._sharedExpert.wrappedValue = LagunaRuntimeMLP(
            dimensions: config.hiddenSize,
            hiddenDimensions: config.sharedExpertIntermediateSize
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, weights) = gate(x)
        var y: MLXArray
        var routedAlreadyReduced = false
        if let fusedWeight = _fusedRoutedGateUpWeight,
            let fusedScales = _fusedRoutedGateUpScales,
            let downProj = _routedDownProj,
            x.dim(1) == 1, inds.size < 64
        {
            // DECODE-ONLY fused gate/up: replicate exactly SwitchGLU's
            // unsorted small-batch path (`indices.size < 64`, so no
            // gatherSort/scatterUnsort) with one gather-QMM over the
            // row-concatenated [gate; up] bank instead of two. The gather
            // call mirrors `QuantizedSwitchLinear.callAsFunction` (biases
            // nil, rhsIndices, transpose, group 16, 4-bit, .nvfp4,
            // sortedIndices false; the prepare guards pin those literals).
            // Each gathered output row is computed independently, so the
            // split halves (gate rows first) are bit-exact vs. the separate
            // banks; down_proj is the stock module invoked exactly as
            // SwitchGLU does. Multi-token forwards (prefill) below keep the
            // fully stock sorted gather-GEMM path and never see the fused
            // bank.
            let activated: MLXArray
            if lagunaFusedRoutedSwiGLUQMVEnabled,
                x.dtype == .bfloat16,
                x.shape == [1, 1, LagunaConstants.hiddenSize],
                fusedWeight.dtype == .uint32,
                fusedScales.dtype == .uint8,
                inds.dtype == .uint32,
                inds.shape == [1, 1, LagunaConstants.numExpertsPerTok],
                _fusedRoutedGateUpSplit == LagunaConstants.moeIntermediateSize
            {
                activated = lagunaRoutedSwiGLUQMV(
                    x,
                    fusedWeight: fusedWeight,
                    fusedScales: fusedScales,
                    indices: inds
                )
            } else {
                let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
                let gateUp = MLX.gatherQuantizedMM(
                    expanded,
                    fusedWeight,
                    scales: fusedScales,
                    biases: nil,
                    rhsIndices: inds,
                    transpose: true,
                    groupSize: 16,
                    bits: 4,
                    mode: .nvfp4,
                    sortedIndices: false
                )
                let xGate = gateUp[.ellipsis, 0 ..< _fusedRoutedGateUpSplit]
                let xUp = gateUp[.ellipsis, _fusedRoutedGateUpSplit...]
                activated = compiledSiluProduct(xGate, xUp)
            }
            if lagunaFusedRoutedDownReduceEnabled,
                let downWeight = _routedDownWeight,
                let downScales = _routedDownScales,
                activated.dtype == .bfloat16,
                activated.shape == [
                    1, 1, LagunaConstants.numExpertsPerTok, 1,
                    LagunaConstants.moeIntermediateSize,
                ],
                downWeight.dtype == .uint32,
                downWeight.shape == [
                    LagunaConstants.numExperts,
                    LagunaConstants.hiddenSize,
                    LagunaConstants.moeIntermediateSize / 8,
                ],
                downScales.dtype == .uint8,
                downScales.shape == [
                    LagunaConstants.numExperts,
                    LagunaConstants.hiddenSize,
                    LagunaConstants.moeIntermediateSize / 16,
                ],
                weights.dtype == .float32,
                weights.shape == [1, 1, LagunaConstants.numExpertsPerTok],
                routedScalingFactor == Float(LagunaConstants.moeRoutedScalingFactor)
            {
                y = lagunaRoutedDownReduce(
                    activated,
                    downWeight: downWeight,
                    downScales: downScales,
                    indices: inds,
                    routerWeights: weights
                )
                routedAlreadyReduced = true
            } else {
                y = MLX.squeezed(
                    downProj(activated, inds, sortedIndices: false),
                    axis: -2)
            }
        } else {
            y = switchMLP(x, inds)
        }
        if !routedAlreadyReduced {
            y = weightedExpertSum(y, weights.asType(y.dtype))
            if routedScalingFactor != 1 {
                y = y * routedScalingFactor
            }
        }
        return y + sharedExpert(x)
    }
}

// MARK: - Decoder Layer

final class LagunaRuntimeDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: LagunaRuntimeAttention
    let mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let attentionType: LagunaLayerType

    init(_ config: LagunaConfig, layerIdx: Int) {
        self._selfAttn.wrappedValue = LagunaRuntimeAttention(config, layerIdx: layerIdx)

        if config.isSparse(layer: layerIdx) {
            self.mlp = LagunaRuntimeSparseMoEBlock(config)
        } else {
            self.mlp = LagunaRuntimeMLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))

        self.attentionType = config.layerType(forLayer: layerIdx)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h: MLXArray
        let normalized: MLXArray
        if lagunaFusedResidualRMSNormEnabled,
            x.dtype == .bfloat16, r.dtype == .bfloat16,
            postAttentionLayerNorm.weight.dtype == .bfloat16,
            x.shape == r.shape, x.dim(-1) == LagunaConstants.hiddenSize,
            x.size == LagunaConstants.hiddenSize
        {
            (h, normalized) = lagunaResidualRMSNorm(
                residual: x, branch: r, weight: postAttentionLayerNorm.weight)
        } else {
            h = x + r
            normalized = postAttentionLayerNorm(h)
        }
        let r2 = mlp(normalized)
        return h + r2
    }

    /// Final-layer prefill specialization. Every supplied row still produces
    /// and commits its K/V state, but only the last query/output row proceeds
    /// through attention output projection and the terminal MLP.
    func callLastPrefillRow(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let normalized = inputLayerNorm(x)
        let r = selfAttn.callLastPrefillRow(normalized, cache: cache)
        let h = lagunaLastTokenHidden(x) + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

// MARK: - Model

/// The Laguna text tower: unscaled embedding and 40 decoder layers. The final
/// RMSNorm remains a child of this module for checkpoint compatibility, but
/// the scored wrapper applies it after selecting the only consumed row.
final class LagunaRuntimeModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [LagunaRuntimeDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let layerTypes: [LagunaLayerType]
    let slidingWindow: Int
    let fullAttentionIdx: Int
    let slidingAttentionIdx: Int

    init(_ config: LagunaConfig) {
        precondition(config.vocabSize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

        self._layers.wrappedValue = (0..<config.numHiddenLayers).map {
            LagunaRuntimeDecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))

        self.layerTypes = config.layerTypes
        self.slidingWindow = config.slidingWindow
        self.fullAttentionIdx = config.layerTypes.firstIndex(of: .full) ?? 0
        self.slidingAttentionIdx = config.layerTypes.firstIndex(of: .sliding) ?? 0
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        // One mask per attention family, derived from a representative
        // layer's cache offset: all full-attention caches advance in
        // lockstep, as do all sliding caches (vendored `LagunaModelInner`
        // convention).
        let fullMask = createAttentionMask(h: h, cache: cache?[fullAttentionIdx])
        let slidingMask = createAttentionMask(
            h: h, cache: cache?[slidingAttentionIdx], windowSize: slidingWindow)

        for (i, layer) in layers.enumerated() {
            let mask = layerTypes[i] == .full ? fullMask : slidingMask
            if i == layers.count - 1, h.dim(1) > 1 {
                if case .causal = mask {
                    h = layer.callLastPrefillRow(h, cache: cache?[i])
                } else {
                    h = layer(h, mask: mask, cache: cache?[i])
                }
            } else {
                h = layer(h, mask: mask, cache: cache?[i])
            }
        }

        return h
    }
}

/// Scored Laguna runtime model: last-token vocabulary head over the
/// reimplemented Laguna text tower.
///
/// `callAsFunction(_:cache:)` serves both prompt prefill
/// (`[1, L]`) and single-token decode steps (`[1, 1]`) and returns
/// `[1, 1, vocab]` last-token logits; `newCache(parameters:)` creates the
/// per-layer cache stack (unbounded `StandardKVCache` for full-attention
/// layers, `RotatingKVCache(512)` for sliding layers). Laguna applies NO
/// final logit softcap and NO embedding scaling.
public final class LagunaRuntimeModel: Module, LanguageModel {
    @ModuleInfo(key: "model") var model: LagunaRuntimeModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let configuration: LagunaConfig

    public init(_ config: LagunaConfig) {
        self.configuration = config
        self._model.wrappedValue = LagunaRuntimeModelInner(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
        super.init()

        // Match the vendored Poolside Laguna model exactly: only the routed
        // experts and shared expert are NVFP4. Quantizing one sparse decoder
        // layer at a time avoids asking Module.update to descend through the
        // dense layer 0, which has no quantized child.
        for layer in model.layers where layer.mlp is LagunaRuntimeSparseMoEBlock {
            quantize(model: layer) { path, _ in
                if path.contains("switch_mlp") || path.contains("shared_expert") {
                    return (
                        groupSize: config.quantization.groupSize,
                        bits: config.quantization.bits,
                        mode: .nvfp4
                    )
                }
                return nil
            }
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let fullHidden = model(inputs, cache: cache)
        // Every consumer of multi-token logits reads only the LAST
        // position's row. Slice before the row-independent final RMSNorm and
        // vocabulary head so prefill neither normalizes nor projects the
        // preceding rows. For single-token decode the slice is a no-op.
        let hidden = model.norm(lagunaLastTokenHidden(fullHidden))
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    public func prepare(
        _ input: LMInput,
        cache _: [KVCache],
        windowSize _: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    public func newCache(parameters _: GenerateParameters?) -> [KVCache] {
        (0..<configuration.numHiddenLayers).map { layerIndex in
            if configuration.layerTypes[layerIndex] == .full {
                StandardKVCache()
            } else {
                RotatingKVCache(maxSize: configuration.slidingWindow, keep: 0)
            }
        }
    }

    /// Builds the retained fused runtime weight layouts (fused QKV, fused
    /// shared-expert gate/up, fused routed gate/up decode banks) once the
    /// checkpoint parameters are installed and evaluated. Called by the
    /// weight cache after `update` + `eval`, before constructor-time warmup,
    /// so the concatenations read materialized weights and the fused arrays
    /// are resident before the first forward. The module tree and its
    /// checkpoint parameters are never restructured; every fused layout is a
    /// derived side copy.
    func prepareFusedRuntimeWeights() {
        var fusedArrays: [MLXArray] = []
        for layer in model.layers {
            if lagunaFusedQKVEnabled, let fused = layer.selfAttn.prepareFusedQKVWeight() {
                fusedArrays.append(fused)
            }
            if let sparse = layer.mlp as? LagunaRuntimeSparseMoEBlock {
                if lagunaFusedSharedGateUpEnabled {
                    fusedArrays.append(contentsOf: sparse.sharedExpert.prepareFusedSharedGateUp())
                }
                if lagunaFusedRoutedGateUpEnabled {
                    fusedArrays.append(contentsOf: sparse.prepareFusedRoutedGateUp())
                }
            }
        }
        if !fusedArrays.isEmpty {
            eval(fusedArrays)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }
        // Drop precomputed rotary tables if a checkpoint ships them.
        return weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
    }
}
