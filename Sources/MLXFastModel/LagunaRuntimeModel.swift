import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN




func lagunaLastTokenRange(sequenceLength: Int) -> Range<Int>? {
    sequenceLength > 1 ? (sequenceLength - 1)..<sequenceLength : nil
}

func lagunaLastTokenHidden(_ hidden: MLXArray) -> MLXArray {
    guard let range = lagunaLastTokenRange(sequenceLength: hidden.dim(1)) else {
        return hidden
    }
    return hidden[0..., range, 0...]
}









extension MLXArray {

    @inline(__always) func dims(_ d0: Int) -> Bool { ndim == 1 && dim(0) == d0 }
    @inline(__always) func dims(_ d0: Int, _ d1: Int) -> Bool { ndim == 2 && shape2 == (d0, d1) }
    @inline(__always) func dims(_ d0: Int, _ d1: Int, _ d2: Int) -> Bool {
        ndim == 3 && shape3 == (d0, d1, d2)
    }
    @inline(__always) func dims(_ d0: Int, _ d1: Int, _ d2: Int, _ d3: Int) -> Bool {
        ndim == 4 && shape4 == (d0, d1, d2, d3)
    }
    @inline(__always) func dims(_ d0: Int, _ d1: Int, _ d2: Int, _ d3: Int, _ d4: Int) -> Bool {
        ndim == 5 && dim(0) == d0 && dim(1) == d1 && dim(2) == d2 && dim(3) == d3 && dim(4) == d4
    }


    @inline(__always) func sameDims(_ other: MLXArray) -> Bool {
        guard ndim == other.ndim else { return false }
        for axis in 0..<ndim where dim(axis) != other.dim(axis) { return false }
        return true
    }
}








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






private let lagunaTraceFusion =
    ProcessInfo.processInfo.environment["DARKBLOOM_TRACE_FUSION"] == "1"
private let lagunaTracedFusions = LagunaFusionTraceLog()

final class LagunaFusionTraceLog: @unchecked Sendable {
    private var seen: Set<String> = []
    private let lock = NSLock()

    func note(_ site: String) {
        lock.lock()
        let isNew = seen.insert(site).inserted
        lock.unlock()
        if isNew {
            FileHandle.standardError.write(Data("mlxfast: fusion active: \(site)\n".utf8))
        }
    }
}

@inline(__always)
func lagunaTrace(_ site: @autoclosure () -> String) {
    guard lagunaTraceFusion else { return }
    lagunaTracedFusions.note(site())
}















let lagunaFusedQKVEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_QKV"] == "1"






let lagunaFusedSharedGateUpEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_SHARED_GATE_UP"] != "0"





let lagunaFusedSharedSwiGLUQMVEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_SHARED_SWIGLU_QMV"] != "0"






let lagunaFusedSharedDownResidualEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_SHARED_DOWN_RESIDUAL"] != "0"




let lagunaFusedRoutedSharedDownResidualEnabled =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_FUSED_ROUTED_SHARED_DOWN_RESIDUAL"] != "0"




let lagunaFusedRoutedSwiGLUQMVEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_ROUTED_SWIGLU_QMV"] != "0"














let lagunaPackedScalesEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PACKED_SCALES"] != "0"




private let lagunaRouterPrecomputedKeysEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ROUTER_PRECOMPUTED_KEYS"] != "0"





final class LagunaPackedScalesLog: @unchecked Sendable {
    private var seen: Set<String> = []
    private let lock = NSLock()

    func note(_ state: String, _ site: String) {
        lock.lock()
        let isNew = seen.insert(site).inserted
        lock.unlock()
        if isNew {
            FileHandle.standardError.write(
                Data("mlxfast: packed-scales \(state): \(site)\n".utf8))
        }
    }
}

let lagunaPackedScalesLog = LagunaPackedScalesLog()




let lagunaFusedRoutedDownReduceEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_ROUTED_DOWN_REDUCE"] != "0"
















let lagunaFusedRoutedGateUpEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_ROUTED_GATE_UP"] != "0"



let lagunaPrefillFusedRoutedGateUpEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_FUSED_GATE_UP"] != "0"




let lagunaPrefillExpertPairwiseScalesEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_EXPERT_PAIRWISE_SCALES"] != "0"


let lagunaPrefillExpertDownPairwiseScalesEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_EXPERT_DOWN_PAIRWISE_SCALES"] != "0"





func lagunaPrefillExpertPairwiseScalesAdmitted(routedRows: Int) -> Bool {
    routedRows >= 4 * LagunaConstants.numExperts
}

func lagunaNAXAvailable(architecture: String, osSupportsNAX: Bool) -> Bool {
    guard osSupportsNAX,
        let generation = Int(architecture.suffix(3).prefix(2))
    else { return false }
    return generation >= (architecture.hasSuffix("p") ? 18 : 17)
}

func lagunaExpertAlignedStageEnabled(_ value: String?) -> Bool {
    ["", "4", "5"].contains(value ?? "")
}

let lagunaExpertAlignedGatherEnabled = {
    let environment = ProcessInfo.processInfo.environment
    guard environment["DARKBLOOM_EXPERT_ALIGNED_GATHER"] != "0",
        lagunaExpertAlignedStageEnabled(environment["DARKBLOOM_STAGE_BM128"]),
        #available(macOS 26.2, *)
    else { return false }
    let configured = environment["MLX_METAL_GPU_ARCH"]
    return lagunaNAXAvailable(
        architecture: configured.flatMap { $0.isEmpty ? nil : $0 }
            ?? GPU.deviceInfo().architecture,
        osSupportsNAX: true
    )
}()








let lagunaFusedResidualRMSNormEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_RESIDUAL_RMS"] != "0"



let lagunaPrefillFusedResidualRMSNormEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_FUSED_RESIDUAL_RMS"] != "0"







let lagunaSwiGLUQMVRows1Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_QMV_R1"] != "0"





let lagunaSharedSwiGLUQMVRows1Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_SHARED_QMV_R1"] != "0"














let lagunaSharedScaleHalvedEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_SHARED_SCALE_HALVED"] != "0"











let lagunaSharedQMVWideCodesEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_QMV_WIDE_CODES"] == "1"




let lagunaFusedGatedOutputProjectionEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_GATED_OUTPUT"] != "0"





let lagunaFusedQKVProjectionEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_QKV_PROJECTION"] != "0"







private let lagunaNativeAffineQKVLayerCount: Int = {
    guard ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_QKV"] != "0"
    else { return 0 }
    let requested = Int(
        ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_QKV_LAYERS"]
            ?? "40") ?? 40
    return min(max(requested, 0), LagunaConstants.numHiddenLayers)
}()
let lagunaNativeAffineQKVEnabled = lagunaNativeAffineQKVLayerCount > 0










private let lagunaNativeAffineSuffixSelection =
    ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_SUFFIX"] == "1"





private func lagunaNativeAffineOnlyLayer(_ key: String) -> Int? {
    guard let raw = ProcessInfo.processInfo.environment[key],
        let value = Int(raw),
        value >= 0,
        value < LagunaConstants.numHiddenLayers
    else {
        return nil
    }
    return value
}

private let lagunaNativeAffineQKVOnlyLayer = lagunaNativeAffineOnlyLayer(
    "DARKBLOOM_NATIVE_AFFINE_QKV_ONLY_LAYER")

private let lagunaNativeAffineOProjOnlyLayer = lagunaNativeAffineOnlyLayer(
    "DARKBLOOM_NATIVE_AFFINE_OPROJ_ONLY_LAYER")

private func lagunaNativeAffineCovers(layer: Int, count: Int, onlyLayer: Int?) -> Bool {
    guard count > 0 else { return false }
    if let onlyLayer { return layer == onlyLayer }
    if lagunaNativeAffineSuffixSelection {
        return layer >= LagunaConstants.numHiddenLayers - count
    }
    return layer < count
}

private func lagunaUseNativeAffineQKV(layer: Int) -> Bool {
    lagunaNativeAffineCovers(
        layer: layer,
        count: lagunaNativeAffineQKVLayerCount,
        onlyLayer: lagunaNativeAffineQKVOnlyLayer)
}















private let lagunaNativeAffineOProjLayerCount: Int = {
    guard ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_OPROJ"] != "0"
    else { return 0 }
    let requested = Int(
        ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_OPROJ_LAYERS"]
            ?? "40") ?? 40
    return min(max(requested, 0), LagunaConstants.numHiddenLayers)
}()
let lagunaNativeAffineOProjEnabled = lagunaNativeAffineOProjLayerCount > 0

private func lagunaUseNativeAffineOProj(layer: Int) -> Bool {
    lagunaNativeAffineCovers(
        layer: layer,
        count: lagunaNativeAffineOProjLayerCount,
        onlyLayer: lagunaNativeAffineOProjOnlyLayer)
}























private let lagunaNativeAffineGProjLayerCount: Int = {
    guard ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_GPROJ"] != "0"
    else { return 0 }
    let requested = Int(
        ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_GPROJ_LAYERS"]
            ?? "40") ?? 40
    return min(max(requested, 0), LagunaConstants.numHiddenLayers)
}()
let lagunaNativeAffineGProjEnabled = lagunaNativeAffineGProjLayerCount > 0

private let lagunaNativeAffineGProjOnlyLayer = lagunaNativeAffineOnlyLayer(
    "DARKBLOOM_NATIVE_AFFINE_GPROJ_ONLY_LAYER")

private func lagunaUseNativeAffineGProj(layer: Int) -> Bool {
    lagunaNativeAffineCovers(
        layer: layer,
        count: lagunaNativeAffineGProjLayerCount,
        onlyLayer: lagunaNativeAffineGProjOnlyLayer)
}





private func lagunaNativeAffineGProjWeight(_ weight: MLXArray) -> LagunaNativeAffineWeight? {
    guard weight.dtype == .bfloat16, weight.ndim == 2,
        weight.dim(1).isMultiple(of: 32)
    else {
        return nil
    }
    let (packedCodes, scales, biases) = quantized(
        weight, groupSize: 32, bits: 8, mode: .affine)
    guard biases != nil else { return nil }
    return LagunaNativeAffineWeight(
        packedCodes: packedCodes,
        scales: scales,
        biases: biases,
        originalShape: weight.shape
    )
}

























let lagunaFusedSlidingQKNormRoPEEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_SLIDING_QK_NORM_ROPE"] != "0"















private let lagunaPrefillQKNormRoPEEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_QK_NORM_ROPE"] != "0"







let lagunaPrefillQKHeadsPerGroup: Int = {
    let raw =
        ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_QK_HEADS"]
        ?? "1"
    return raw == "4" ? 4 : 1
}()








private let lagunaLastPrefillProjectionBanksEnabled =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_LAST_PREFILL_PROJECTION_BANKS"] != "0"




private let lagunaTerminalPrefillFusionEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_TERMINAL_FUSION"] != "0"







let lagunaFusedResidualRMSNormRouterEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_RESIDUAL_RMS_ROUTER"] != "0"

let lagunaFusedFullQKNormYaRNEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_FULL_QK_NORM_YARN"] != "0"



















let lagunaRoPEAngleAtlasEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ROPE_ANGLE_ATLAS"] != "0"





















let lagunaRoPEAtlasViewsEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ROPE_ATLAS_VIEWS"] == "1"










let lagunaFusedDenseGateUpSwiGLUEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_DENSE_GATE_UP_SWIGLU"] != "0"








let lagunaFusedDenseDownResidualEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_DENSE_DOWN_RESIDUAL"] != "0"


























let lagunaRouterRowsPerGroup: Int = {
    guard
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_ROUTER_ROWS_PER_GROUP"],
        let value = Int(raw), [1, 2, 4, 8, 16, 32, 64].contains(value)
    else {
        return 8
    }
    return value
}()











let lagunaRouterWeightPrefetch: Int = {
    guard
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_ROUTER_WEIGHT_PREFETCH"],
        let value = Int(raw), [0, 1, 5].contains(value)
    else {
        return 1
    }
    return value
}()







private enum LagunaDecodeAsyncStage {
    case off
    case layer(Int)
    case ladder(Int)
    case explicit(UInt64)
    case norm
    case logits
}
























private let lagunaDecodeAsyncStage: LagunaDecodeAsyncStage = {
    let raw =
        ProcessInfo.processInfo.environment["DARKBLOOM_DECODE_ASYNC_STAGE"]?
        .lowercased() ?? "at:0,1,7,15,23,31,39"
    switch raw {
    case "off", "0", "":
        return .off
    case "norm":
        return .norm
    case "logits":
        return .logits
    default:





        if raw.hasPrefix("at:") {
            var mask: UInt64 = 0
            for field in raw.dropFirst(3).split(separator: ",") {
                guard let index = Int(field), (0..<64).contains(index) else { return .off }
                mask |= 1 << UInt64(index)
            }
            return mask == 0 ? .off : .explicit(mask)
        }
        if raw.hasPrefix("ladder"), let stride = Int(raw.dropFirst("ladder".count)),
            (1...40).contains(stride)
        {
            return .ladder(stride)
        }
        if let index = Int(raw), (0...39).contains(index) {
            return .layer(index)
        }
        return .off
    }
}()



private let lagunaAttentionProjectionAsyncEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ATTN_PROJECTION_ASYNC"] != "0"














private let lagunaPrefillAsyncLadderStride: Int = {
    let raw = ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_ASYNC_LADDER"]?.lowercased() ?? "1"
    if raw == "off" || raw == "0" || raw.isEmpty { return 0 }
    guard let n = Int(raw), (1...40).contains(n) else { return 0 }
    return n
}()

private let lagunaRoPEAngleAtlasLength = 4096

















private let lagunaNormInvMeanScratch = "threadgroup float local_inv_mean[1];"





private func lagunaNormReductionTail(
    lane: String,
    simdGroup: String,
    denominator: String,
    epsilon: String
) -> String {
    let inverseRMS =
        "metal::precise::rsqrt(acc / \(denominator) + \(epsilon))"
    let lines: [String] = [
        "if (\(simdGroup) == 0) {",
        "    local_sums[\(lane)] = 0.0f;",
        "}",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        "if (\(lane) == 0) {",
        "    local_sums[\(simdGroup)] = acc;",
        "}",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        "if (\(simdGroup) == 0) {",
        "    acc = simd_sum(local_sums[\(lane)]);",
        "    if (\(lane) == 0) {",
        "        local_inv_mean[0] = \(inverseRMS);",
        "    }",
        "}",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        "float laguna_inv_mean = local_inv_mean[0];",
    ]


    return lines.joined(separator: "\n        ")
}


private let lagunaNormReductionTail2048 = lagunaNormReductionTail(
    lane: "simd_lane", simdGroup: "simd_group",
    denominator: "2048.0f", epsilon: "1.0e-6f")



private let lagunaNormReductionTailQKV = lagunaNormReductionTail(
    lane: "lane", simdGroup: "simd_group",
    denominator: "float(in_vec_size)", epsilon: "norm_eps")





private func lagunaRouterPrefetchGroups(rowsPerThread: Int, prefetch: Int) -> Int {
    guard rowsPerThread == 1 else { return 0 }
    return prefetch == 5 ? 1 : prefetch
}

















































private func lagunaResidualRMSNormRouterSource(
    rowsPerGroup: Int, prefetch: Int = 0
) -> String {
    let simdGroups = 512 / 32
    let rowsPerThread = rowsPerGroup >= simdGroups ? rowsPerGroup / simdGroups : 1
    let activeSimdGroups = rowsPerGroup / rowsPerThread
    let prefetchGroups = lagunaRouterPrefetchGroups(
        rowsPerThread: rowsPerThread, prefetch: prefetch)
    let prefetchBlock = prefetchGroups == 0 ? "" : """
thread vec<bfloat, 4> laguna_pf[\(prefetchGroups * 4)];
if (simd_group < active_simd_groups) {
    uint laguna_pf_row = tile * rows_per_group + simd_group * rows_per_thread;
    uint laguna_pf_column = simd_lane * n_reads;
    for (uint k = 0; k < \(prefetchGroups * 4); ++k) {
        const device vec<bfloat, 4>* pf_values =
            (const device vec<bfloat, 4>*)(
                router_weight + laguna_pf_row * axis_size +
                    laguna_pf_column + k * block_width);
        laguna_pf[k] = pf_values[0];
    }
}

"""
    let prefetchEarly = prefetch == 5 ? "" : prefetchBlock
    let prefetchLate = prefetch == 5 ? prefetchBlock : ""
    let zeros = Array(repeating: "0.0f", count: rowsPerThread).joined(separator: ", ")
    let guardOpen = activeSimdGroups < simdGroups
        ? "        if (simd_group < active_simd_groups) {\n" : ""
    let guardClose = activeSimdGroups < simdGroups ? "        }\n" : ""
    let routerStore = lagunaRouterPrecomputedKeysEnabled
        ? """
        bfloat logit = bfloat(router_result[r]);
        router_logits[router_row + r] = logit;
        float x = float(logit);
        float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
        float score = x < 0.0f ? y : 1.0f - y;
        router_keys[router_row + r] = laguna_router_key_ordinal(
            -(score + float(correction_bias[router_row + r])));
"""
        : "router_logits[router_row + r] = bfloat(router_result[r]);"

    let accumulate: String
    if prefetchGroups > 0 {
        accumulate = """
        uint column = simd_lane * n_reads;
        for (uint g = 0; g < \(prefetchGroups); ++g) {
            for (uint u = 0; u < 4; ++u) {
                uint column_u = column + u * block_width;
                for (uint i = 0; i < n_reads; ++i) {
                    router_result[0] += float(laguna_pf[g * 4 + u][i]) *
                        float(normalized_row[column_u + i]);
                }
            }
            column += 4 * block_width;
        }
        for (uint block = \(prefetchGroups * 4); block < router_blocks; block += 4) {
            vec<bfloat, 4> rw[4];
            for (uint u = 0; u < 4; ++u) {
                const device vec<bfloat, 4>* row_values =
                    (const device vec<bfloat, 4>*)(
                        router_weight + router_row * axis_size +
                            column + u * block_width);
                rw[u] = row_values[0];
            }
            for (uint u = 0; u < 4; ++u) {
                uint column_u = column + u * block_width;
                for (uint i = 0; i < n_reads; ++i) {
                    router_result[0] += float(rw[u][i]) *
                        float(normalized_row[column_u + i]);
                }
            }
            column += 4 * block_width;
        }
"""
    } else if rowsPerThread == 1 {
        accumulate = """
        uint column = simd_lane * n_reads;
        for (uint block = 0; block < router_blocks; block += 4) {
            vec<bfloat, 4> rw[4];
            for (uint u = 0; u < 4; ++u) {
                const device vec<bfloat, 4>* row_values =
                    (const device vec<bfloat, 4>*)(
                        router_weight + router_row * axis_size +
                            column + u * block_width);
                rw[u] = row_values[0];
            }
            for (uint u = 0; u < 4; ++u) {
                uint column_u = column + u * block_width;
                for (uint i = 0; i < n_reads; ++i) {
                    router_result[0] += float(rw[u][i]) *
                        float(normalized_row[column_u + i]);
                }
            }
            column += 4 * block_width;
        }
"""
    } else {
        accumulate = """
        thread float router_input[n_reads];

        uint column = simd_lane * n_reads;
        for (uint block = 0; block < router_blocks; ++block) {
            for (uint i = 0; i < n_reads; ++i) {
                router_input[i] = float(normalized_row[column + i]);
            }
            for (uint r = 0; r < rows_per_thread; ++r) {
                const device vec<bfloat, 4>* row_values =
                    (const device vec<bfloat, 4>*)(
                        router_weight + (router_row + r) * axis_size +
                            column);
                const vec<bfloat, 4> rw = row_values[0];
                for (uint i = 0; i < n_reads; ++i) {
                    router_result[r] += float(rw[i]) * router_input[i];
                }
            }
            column += block_width;
        }
"""
    }

    return """
constexpr uint axis_size = 2048;
constexpr uint n_reads = 4;
constexpr uint simd_size = 32;
constexpr uint rows_per_group = \(rowsPerGroup);
constexpr uint rows_per_thread = \(rowsPerThread);
constexpr uint active_simd_groups = \(activeSimdGroups);
constexpr uint block_width = 128;
constexpr uint router_blocks = axis_size / block_width;

uint tile = threadgroup_position_in_grid.x;
uint lid = thread_position_in_threadgroup.x;
uint simd_lane = thread_index_in_simdgroup;
uint simd_group = simdgroup_index_in_threadgroup;
uint base = lid * n_reads;

\(lagunaNormInvMeanScratch)
threadgroup float local_sums[simd_size];
threadgroup bfloat normalized_row[axis_size];

thread bfloat values[n_reads];
float acc = 0.0f;
for (uint i = 0; i < n_reads; ++i) {
    bfloat value = bfloat(residual[base + i] + branch[base + i]);
    values[i] = value;
    if (tile == 0) {
        summed[base + i] = value;
    }
    float fv = float(value);
    acc += fv * fv;
}

acc = simd_sum(acc);
\(prefetchEarly)\(lagunaNormReductionTail2048)

for (uint i = 0; i < n_reads; ++i) {
    bfloat value =
        weight[base + i] *
        bfloat(float(values[i]) * laguna_inv_mean);
    normalized_row[base + i] = value;
    if (tile == 0) {
        normalized[base + i] = value;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

\(prefetchLate)\(guardOpen)\
uint router_row = tile * rows_per_group + simd_group * rows_per_thread;
thread float router_result[rows_per_thread] = {\(zeros)};
\(accumulate)

for (uint r = 0; r < rows_per_thread; ++r) {
    for (ushort delta = 16; delta >= 1; delta >>= 1) {
        router_result[r] +=
            metal::simd_shuffle_down(router_result[r], delta);
    }
}
if (simd_lane == 0) {
    for (uint r = 0; r < rows_per_thread; ++r) {
        \(routerStore)
    }
}
\(guardClose)
"""
}






private let lagunaResidualRMSNormRouterKernels: [Int: MLXFast.MLXFastKernel] =
    Dictionary(
        uniqueKeysWithValues: [1, 2, 4, 8, 16, 32, 64].flatMap { rowsPerGroup in
            [0, 1, 5].map { prefetch -> (Int, MLXFast.MLXFastKernel) in
            let groups = lagunaRouterPrefetchGroups(
                rowsPerThread: rowsPerGroup >= 16 ? rowsPerGroup / 16 : 1,
                prefetch: prefetch)
            let armSuffix = groups == 0 ? "" : (prefetch == 5 ? "_pf1c" : "_pf\(groups)")
            return (
                rowsPerGroup * 8 + prefetch,
                MLXFast.metalKernel(
                    name: "laguna_residual_rms_router_bf16_2048_rpg\(rowsPerGroup)_"
                        + (lagunaRouterPrecomputedKeysEnabled ? "keys_v1" : "v2")
                        + armSuffix,
                    inputNames: lagunaRouterPrecomputedKeysEnabled
                        ? ["residual", "branch", "weight", "router_weight", "correction_bias"]
                        : ["residual", "branch", "weight", "router_weight"],
                    outputNames: lagunaRouterPrecomputedKeysEnabled
                        ? ["summed", "normalized", "router_logits", "router_keys"]
                        : ["summed", "normalized", "router_logits"],
                    source: lagunaResidualRMSNormRouterSource(
                        rowsPerGroup: rowsPerGroup, prefetch: prefetch),
                    header: lagunaRouterPrecomputedKeysEnabled
                        ? lagunaDecodeRouterOrdinalHeader : "",
                    ensureRowContiguous: true
                )
            )
            }
        })



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

\(lagunaNormInvMeanScratch)
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
\(lagunaNormReductionTail2048)

for (uint i = 0; i < n_reads; ++i) {
    normalized[base + i] =
        weight[lid * n_reads + i] *
        bfloat(float(values[i]) * laguna_inv_mean);
}
""",
    ensureRowContiguous: true
)

func lagunaResidualRMSNormRouter(
    residual: MLXArray, branch: MLXArray, weight: MLXArray,
    routerWeight: MLXArray, correctionBias: MLXArray
) -> (summed: MLXArray, normalized: MLXArray, routerLogits: MLXArray,
    routerKeys: MLXArray?) {
    let hidden = LagunaConstants.hiddenSize
    let experts = LagunaConstants.numExperts
    precondition(residual.dtype == .bfloat16)
    precondition(branch.dtype == .bfloat16)
    precondition(weight.dtype == .bfloat16)
    precondition(routerWeight.dtype == .bfloat16)
    precondition(correctionBias.dtype == .float32 || correctionBias.dtype == .bfloat16)
    precondition(residual.dims(1, 1, hidden))
    precondition(branch.dims(1, 1, hidden))
    precondition(weight.dims(hidden))
    precondition(routerWeight.dims(experts, hidden))
    precondition(correctionBias.dims(experts))








    let rowsPerGroup = lagunaRouterRowsPerGroup
    let tiles = experts / rowsPerGroup
    lagunaTrace("residual+rmsnorm+router rpg\(rowsPerGroup) pf\(lagunaRouterWeightPrefetch)")
    let inputs = lagunaRouterPrecomputedKeysEnabled
        ? [residual, branch, weight, routerWeight, correctionBias]
        : [residual, branch, weight, routerWeight]
    let outputs = lagunaResidualRMSNormRouterKernels[
        rowsPerGroup * 8 + lagunaRouterWeightPrefetch]!(
        inputs,
        grid: (tiles * 512, 1, 1),
        threadGroup: (512, 1, 1),
        outputShapes: [[1, 1, hidden], [1, 1, hidden], [1, 1, experts]]
            + (lagunaRouterPrecomputedKeysEnabled ? [[1, 1, experts]] : []),
        outputDTypes: [.bfloat16, .bfloat16, .bfloat16]
            + (lagunaRouterPrecomputedKeysEnabled ? [.uint32] : [])
    )
    return (outputs[0], outputs[1], outputs[2], outputs.count > 3 ? outputs[3] : nil)
}

func lagunaResidualRMSNorm(
    residual: MLXArray, branch: MLXArray, weight: MLXArray
) -> (MLXArray, MLXArray) {
    precondition(residual.dtype == .bfloat16)
    precondition(branch.dtype == .bfloat16)
    precondition(weight.dtype == .bfloat16)
    precondition(residual.sameDims(branch))
    precondition(residual.dim(-1) == LagunaConstants.hiddenSize)
    precondition(weight.dims(LagunaConstants.hiddenSize))

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



private let lagunaFullQKNormYaRNKernel = MLXFast.metalKernel(
    name: "laguna_full_qk_norm_yarn_bf16_128_v4",
    inputNames: ["raw_queries", "raw_keys", "query_weight", "key_weight", "angles"],
    outputNames: ["queries", "keys"],
    source: """
constexpr uint head_dim = 128;
constexpr uint rotary_dims = 64;
constexpr uint rotary_pairs = 32;
constexpr uint query_heads = 48;
constexpr float yarn_mscale = 1.3465735912322998f;

uint head = threadgroup_position_in_grid.x;
uint lane = thread_index_in_simdgroup;


const device bfloat* input;
const device bfloat* weight;
if (head < query_heads) {
    input = raw_queries + head * head_dim;
    weight = query_weight;
} else {
    input = raw_keys + (head - query_heads) * head_dim;
    weight = key_weight;
}

uint base = lane * 4;
thread bfloat normalized[4];
float sum = 0.0f;
for (uint i = 0; i < 4; ++i) {
    float value = float(input[base + i]);
    sum += value * value;
}
sum = simd_sum(sum);
float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

for (uint i = 0; i < 4; ++i) {
    normalized[i] =
        weight[base + i] *
        bfloat(float(input[base + i]) * inverse_rms);
}

thread float paired[4];
for (uint i = 0; i < 4; ++i) {
    paired[i] = simd_shuffle(float(normalized[i]), lane ^ 8);
}

device bfloat* output =
    head < query_heads
    ? queries + head * head_dim
    : keys + (head - query_heads) * head_dim;
if (lane < 8) {
    bfloat rounded_mscale = bfloat(yarn_mscale);
    for (uint i = 0; i < 4; ++i) {
        uint pair = base + i;
        float first =
            float(bfloat(normalized[i] * rounded_mscale));
        float second =
            float(bfloat(bfloat(paired[i]) * rounded_mscale));
        float cosine = angles[pair];
        float sine = angles[pair + rotary_pairs];
        output[pair] = bfloat(first * cosine - second * sine);
        output[pair + rotary_pairs] =
            bfloat(first * sine + second * cosine);
    }
} else if (lane >= 16) {
    for (uint i = 0; i < 4; ++i) {
        output[base + i] = normalized[i];
    }
}
""",
    ensureRowContiguous: true
)

func lagunaFullQKNormYaRN(
    rawQueries: MLXArray,
    rawKeys: MLXArray,
    queryWeight: MLXArray,
    keyWeight: MLXArray,
    angles: MLXArray
) -> (MLXArray, MLXArray) {
    precondition(rawQueries.dtype == .bfloat16)
    precondition(rawKeys.dtype == .bfloat16)
    precondition(queryWeight.dtype == .bfloat16)
    precondition(keyWeight.dtype == .bfloat16)
    precondition(rawQueries.dims(1, 1, 48 * LagunaConstants.headDim))
    precondition(rawKeys.dims(1, 1, 8 * LagunaConstants.headDim))
    precondition(queryWeight.dims(LagunaConstants.headDim))
    precondition(keyWeight.dims(LagunaConstants.headDim))
    precondition(angles.dtype == .float32)
    precondition(angles.dims(1, 1, 1, LagunaConstants.headDim / 2))

    lagunaTrace("full qk norm+yarn")
    let outputs = lagunaFullQKNormYaRNKernel(
        [rawQueries, rawKeys, queryWeight, keyWeight, angles],
        grid: (56 * 32, 1, 1),
        threadGroup: (32, 1, 1),
        outputShapes: [
            [1, 48, 1, LagunaConstants.headDim],
            [1, 8, 1, LagunaConstants.headDim],
        ],
        outputDTypes: [.bfloat16, .bfloat16]
    )
    return (outputs[0], outputs[1])
}






















private let lagunaSlidingQKNormRoPEKernel = MLXFast.metalKernel(
    name: "laguna_sliding_qk_norm_rope_bf16_128_v1",
    inputNames: ["raw_queries", "raw_keys", "query_weight", "key_weight", "angles"],
    outputNames: ["queries", "keys"],
    source: """
constexpr uint head_dim = 128;
constexpr uint rotary_pairs = 64;
constexpr uint query_heads = 64;

uint head = threadgroup_position_in_grid.x;
uint lane = thread_index_in_simdgroup;


const device bfloat* input;
const device bfloat* weight;
if (head < query_heads) {
    input = raw_queries + head * head_dim;
    weight = query_weight;
} else {
    input = raw_keys + (head - query_heads) * head_dim;
    weight = key_weight;
}

uint base = lane * 4;
thread bfloat normalized[4];
float sum = 0.0f;
for (uint i = 0; i < 4; ++i) {
    float value = float(input[base + i]);
    sum += value * value;
}
sum = simd_sum(sum);
float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

for (uint i = 0; i < 4; ++i) {
    normalized[i] =
        weight[base + i] *
        bfloat(float(input[base + i]) * inverse_rms);
}

thread float paired[4];
for (uint i = 0; i < 4; ++i) {
    paired[i] = simd_shuffle(float(normalized[i]), lane ^ 16);
}

device bfloat* output =
    head < query_heads
    ? queries + head * head_dim
    : keys + (head - query_heads) * head_dim;
if (lane < 16) {
    for (uint i = 0; i < 4; ++i) {
        uint pair = base + i;
        float first = float(normalized[i]);
        float second = paired[i];
        float cosine = angles[pair];
        float sine = angles[pair + rotary_pairs];
        output[pair] = bfloat(first * cosine - second * sine);
        output[pair + rotary_pairs] =
            bfloat(first * sine + second * cosine);
    }
}
""",
    ensureRowContiguous: true
)

func lagunaSlidingQKNormRoPE(
    rawQueries: MLXArray,
    rawKeys: MLXArray,
    queryWeight: MLXArray,
    keyWeight: MLXArray,
    angles: MLXArray
) -> (MLXArray, MLXArray) {
    let heads = LagunaConstants.slidingAttentionHeads
    let kvHeads = LagunaConstants.numKeyValueHeads
    precondition(rawQueries.dtype == .bfloat16)
    precondition(rawKeys.dtype == .bfloat16)
    precondition(queryWeight.dtype == .bfloat16)
    precondition(keyWeight.dtype == .bfloat16)
    precondition(rawQueries.dims(1, 1, heads * LagunaConstants.headDim))
    precondition(rawKeys.dims(1, 1, kvHeads * LagunaConstants.headDim))
    precondition(queryWeight.dims(LagunaConstants.headDim))
    precondition(keyWeight.dims(LagunaConstants.headDim))
    precondition(angles.dtype == .float32)
    precondition(angles.dims(1, 1, 1, LagunaConstants.headDim))

    lagunaTrace("sliding qk norm+rope")
    let outputs = lagunaSlidingQKNormRoPEKernel(
        [rawQueries, rawKeys, queryWeight, keyWeight, angles],
        grid: ((heads + kvHeads) * 32, 1, 1),
        threadGroup: (32, 1, 1),
        outputShapes: [
            [1, heads, 1, LagunaConstants.headDim],
            [1, kvHeads, 1, LagunaConstants.headDim],
        ],
        outputDTypes: [.bfloat16, .bfloat16]
    )
    return (outputs[0], outputs[1])
}






















let lagunaFusedSlidingAttentionEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_SLIDING_ATTN"] != "0"

private let lagunaSlidingFusedAttentionKernel = MLXFast.metalKernel(
    name: "laguna_sliding_fused_attn_ring_v1",
    inputNames: [
        "raw_queries", "raw_keys", "raw_values",
        "query_weight", "key_weight", "angles",
        "k_cache", "v_cache", "params", "scale_arr",
    ],
    outputNames: ["attended"],
    source: """
constexpr uint head_dim = 128;
constexpr uint window = 512;
constexpr uint gqa = 8;
constexpr int BN = 32;
constexpr int BD = 32;
constexpr int BDP = BD + 1;
constexpr int qk_per_thread = 4;
constexpr int v_per_thread = 4;
constexpr uint rotary_pairs = 64;
constexpr int N = 512;

typedef float U;

uint pair_tg = threadgroup_position_in_grid.x;
uint head0 = pair_tg * 2;
uint head1 = head0 + 1;
uint kv_head = head0 / gqa;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint widx = params[0];
float scale = scale_arr[0];

threadgroup bfloat tg_q0[head_dim];
threadgroup bfloat tg_q1[head_dim];
threadgroup bfloat tg_k[head_dim];
threadgroup bfloat tg_v[head_dim];

if (sg < 3) {
    const device bfloat* input =
        sg == 0 ? raw_queries + head0 * head_dim
        : sg == 1 ? raw_queries + head1 * head_dim
                  : raw_keys + kv_head * head_dim;
    const device bfloat* weight =
        sg == 2 ? key_weight : query_weight;
    threadgroup bfloat* outrow =
        sg == 0 ? tg_q0 : sg == 1 ? tg_q1 : tg_k;

    uint base = lane * 4;
    thread bfloat normalized[4];
    float sum = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        float value = float(input[base + i]);
        sum += value * value;
    }
    sum = simd_sum(sum);
    float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);
    for (uint i = 0; i < 4; ++i) {
        normalized[i] =
            weight[base + i] *
            bfloat(float(input[base + i]) * inverse_rms);
    }
    thread float paired[4];
    for (uint i = 0; i < 4; ++i) {
        paired[i] = simd_shuffle(float(normalized[i]), lane ^ 16);
    }
    if (lane < 16) {
        for (uint i = 0; i < 4; ++i) {
            uint pair = base + i;
            float first = float(normalized[i]);
            float second = paired[i];
            float cosine = angles[pair];
            float sine = angles[pair + rotary_pairs];
            outrow[pair] = bfloat(first * cosine - second * sine);
            outrow[pair + rotary_pairs] =
                bfloat(first * sine + second * cosine);
        }
    }
} else if (sg == 3) {
    const device bfloat* vin = raw_values + kv_head * head_dim;
    for (uint i = lane; i < head_dim; i += 32) {
        tg_v[i] = vin[i];
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if ((head0 % gqa) == 0 && sg == 0) {
    device bfloat* kc = (device bfloat*)k_cache +
        (size_t)kv_head * (window * head_dim) +
        (size_t)widx * head_dim;
    device bfloat* vc = (device bfloat*)v_cache +
        (size_t)kv_head * (window * head_dim) +
        (size_t)widx * head_dim;
    for (uint i = lane; i < head_dim; i += 32) {
        kc[i] = tg_k[i];
        vc[i] = tg_v[i];
    }
}

threadgroup float4 outputs4[BN * BDP];
threadgroup U max_scores[2 * BN];
threadgroup U sum_exp_scores[2 * BN];

const device bfloat* pair_keys = k_cache +
    (size_t)kv_head * (window * head_dim) +
    (size_t)sg * head_dim + lane * qk_per_thread;
const device bfloat* pair_values = v_cache +
    (size_t)kv_head * (window * head_dim) +
    (size_t)sg * head_dim + lane * v_per_thread;
const int inner_k_stride = BN * int(head_dim);
const int inner_v_stride = BN * int(head_dim);

thread U pair_q0[qk_per_thread];
thread U pair_q1[qk_per_thread];
thread U pair_o0[v_per_thread];
thread U pair_o1[v_per_thread];

for (int j = 0; j < qk_per_thread; ++j) {
    pair_q0[j] =
        static_cast<U>(scale) * tg_q0[lane * qk_per_thread + j];
    pair_q1[j] =
        static_cast<U>(scale) * tg_q1[lane * qk_per_thread + j];
}
for (int j = 0; j < v_per_thread; ++j) {
    pair_o0[j] = 0;
    pair_o1[j] = 0;
}

U pair_max0 = metal::numeric_limits<U>::lowest();
U pair_max1 = metal::numeric_limits<U>::lowest();
U pair_sum0 = 0;
U pair_sum1 = 0;

int i = sg;
for (; i + 3 * BN < N; i += 4 * BN) {
    const device bfloat* pipe_keys_b = pair_keys + inner_k_stride;
    const device bfloat* pipe_keys_c = pair_keys + 2 * inner_k_stride;
    const device bfloat* pipe_keys_d = pair_keys + 3 * inner_k_stride;
    const device bfloat* pipe_values_b = pair_values + inner_v_stride;
    const device bfloat* pipe_values_c = pair_values + 2 * inner_v_stride;
    const device bfloat* pipe_values_d = pair_values + 3 * inner_v_stride;
    const bool sub_a = uint(i) == widx;
    const bool sub_b = uint(i + BN) == widx;
    const bool sub_c = uint(i + 2 * BN) == widx;
    const bool sub_d = uint(i + 3 * BN) == widx;
    U pipe_ka[4];
    U pipe_kb[4];
    U pipe_kc[4];
    U pipe_kd[4];
    T_LOAD_K(pipe_ka, sub_a, pair_keys);
    T_LOAD_K(pipe_kb, sub_b, pipe_keys_b);
    T_LOAD_K(pipe_kc, sub_c, pipe_keys_c);
    T_LOAD_K(pipe_kd, sub_d, pipe_keys_d);
    bfloat pipe_va0, pipe_va1, pipe_va2, pipe_va3;
    bfloat pipe_vb0, pipe_vb1, pipe_vb2, pipe_vb3;
    bfloat pipe_vc0, pipe_vc1, pipe_vc2, pipe_vc3;
    bfloat pipe_vd0, pipe_vd1, pipe_vd2, pipe_vd3;
    T_LOAD_V(pipe_va0, pipe_va1, pipe_va2, pipe_va3, sub_a,
        pair_values);
    T_LOAD_V(pipe_vb0, pipe_vb1, pipe_vb2, pipe_vb3, sub_b,
        pipe_values_b);
    T_LOAD_V(pipe_vc0, pipe_vc1, pipe_vc2, pipe_vc3, sub_c,
        pipe_values_c);
    T_LOAD_V(pipe_vd0, pipe_vd1, pipe_vd2, pipe_vd3, sub_d,
        pipe_values_d);

    U pair_score0 = 0;
    U pair_score1 = 0;
    pair_score0 += pair_q0[0] * pipe_ka[0];
    pair_score1 += pair_q1[0] * pipe_ka[0];
    pair_score0 += pair_q0[1] * pipe_ka[1];
    pair_score1 += pair_q1[1] * pipe_ka[1];
    pair_score0 += pair_q0[2] * pipe_ka[2];
    pair_score1 += pair_q1[2] * pipe_ka[2];
    pair_score0 += pair_q0[3] * pipe_ka[3];
    pair_score1 += pair_q1[3] * pipe_ka[3];
    pair_score0 = simd_sum(pair_score0);
    pair_score1 = simd_sum(pair_score1);

    U pair_new_max0 = metal::max(pair_max0, pair_score0);
    U pair_new_max1 = metal::max(pair_max1, pair_score1);
    U pair_factor0;
    U pair_factor1;
    LAGUNA_RESCALE(pair_factor0, pair_max0 - pair_new_max0);
    LAGUNA_RESCALE(pair_factor1, pair_max1 - pair_new_max1);
    U pair_exp0 = metal::fast::exp(pair_score0 - pair_new_max0);
    U pair_exp1 = metal::fast::exp(pair_score1 - pair_new_max1);

    pair_max0 = pair_new_max0;
    pair_max1 = pair_new_max1;
    pair_sum0 = pair_sum0 * pair_factor0 + pair_exp0;
    pair_sum1 = pair_sum1 * pair_factor1 + pair_exp1;

    pair_o0[0] = pair_o0[0] * pair_factor0 + pair_exp0 * pipe_va0;
    pair_o1[0] = pair_o1[0] * pair_factor1 + pair_exp1 * pipe_va0;
    pair_o0[1] = pair_o0[1] * pair_factor0 + pair_exp0 * pipe_va1;
    pair_o1[1] = pair_o1[1] * pair_factor1 + pair_exp1 * pipe_va1;
    pair_o0[2] = pair_o0[2] * pair_factor0 + pair_exp0 * pipe_va2;
    pair_o1[2] = pair_o1[2] * pair_factor1 + pair_exp1 * pipe_va2;
    pair_o0[3] = pair_o0[3] * pair_factor0 + pair_exp0 * pipe_va3;
    pair_o1[3] = pair_o1[3] * pair_factor1 + pair_exp1 * pipe_va3;

    U pipeb_score0 = 0;
    U pipeb_score1 = 0;
    pipeb_score0 += pair_q0[0] * pipe_kb[0];
    pipeb_score1 += pair_q1[0] * pipe_kb[0];
    pipeb_score0 += pair_q0[1] * pipe_kb[1];
    pipeb_score1 += pair_q1[1] * pipe_kb[1];
    pipeb_score0 += pair_q0[2] * pipe_kb[2];
    pipeb_score1 += pair_q1[2] * pipe_kb[2];
    pipeb_score0 += pair_q0[3] * pipe_kb[3];
    pipeb_score1 += pair_q1[3] * pipe_kb[3];
    pipeb_score0 = simd_sum(pipeb_score0);
    pipeb_score1 = simd_sum(pipeb_score1);

    U pipeb_new_max0 = metal::max(pair_max0, pipeb_score0);
    U pipeb_new_max1 = metal::max(pair_max1, pipeb_score1);
    U pipeb_factor0;
    U pipeb_factor1;
    LAGUNA_RESCALE(pipeb_factor0, pair_max0 - pipeb_new_max0);
    LAGUNA_RESCALE(pipeb_factor1, pair_max1 - pipeb_new_max1);
    U pipeb_exp0 = metal::fast::exp(pipeb_score0 - pipeb_new_max0);
    U pipeb_exp1 = metal::fast::exp(pipeb_score1 - pipeb_new_max1);

    pair_max0 = pipeb_new_max0;
    pair_max1 = pipeb_new_max1;
    pair_sum0 = pair_sum0 * pipeb_factor0 + pipeb_exp0;
    pair_sum1 = pair_sum1 * pipeb_factor1 + pipeb_exp1;

    pair_o0[0] = pair_o0[0] * pipeb_factor0 + pipeb_exp0 * pipe_vb0;
    pair_o1[0] = pair_o1[0] * pipeb_factor1 + pipeb_exp1 * pipe_vb0;
    pair_o0[1] = pair_o0[1] * pipeb_factor0 + pipeb_exp0 * pipe_vb1;
    pair_o1[1] = pair_o1[1] * pipeb_factor1 + pipeb_exp1 * pipe_vb1;
    pair_o0[2] = pair_o0[2] * pipeb_factor0 + pipeb_exp0 * pipe_vb2;
    pair_o1[2] = pair_o1[2] * pipeb_factor1 + pipeb_exp1 * pipe_vb2;
    pair_o0[3] = pair_o0[3] * pipeb_factor0 + pipeb_exp0 * pipe_vb3;
    pair_o1[3] = pair_o1[3] * pipeb_factor1 + pipeb_exp1 * pipe_vb3;

    U pipec_score0 = 0;
    U pipec_score1 = 0;
    pipec_score0 += pair_q0[0] * pipe_kc[0];
    pipec_score1 += pair_q1[0] * pipe_kc[0];
    pipec_score0 += pair_q0[1] * pipe_kc[1];
    pipec_score1 += pair_q1[1] * pipe_kc[1];
    pipec_score0 += pair_q0[2] * pipe_kc[2];
    pipec_score1 += pair_q1[2] * pipe_kc[2];
    pipec_score0 += pair_q0[3] * pipe_kc[3];
    pipec_score1 += pair_q1[3] * pipe_kc[3];
    pipec_score0 = simd_sum(pipec_score0);
    pipec_score1 = simd_sum(pipec_score1);

    U pipec_new_max0 = metal::max(pair_max0, pipec_score0);
    U pipec_new_max1 = metal::max(pair_max1, pipec_score1);
    U pipec_factor0;
    U pipec_factor1;
    LAGUNA_RESCALE(pipec_factor0, pair_max0 - pipec_new_max0);
    LAGUNA_RESCALE(pipec_factor1, pair_max1 - pipec_new_max1);
    U pipec_exp0 = metal::fast::exp(pipec_score0 - pipec_new_max0);
    U pipec_exp1 = metal::fast::exp(pipec_score1 - pipec_new_max1);

    pair_max0 = pipec_new_max0;
    pair_max1 = pipec_new_max1;
    pair_sum0 = pair_sum0 * pipec_factor0 + pipec_exp0;
    pair_sum1 = pair_sum1 * pipec_factor1 + pipec_exp1;

    pair_o0[0] = pair_o0[0] * pipec_factor0 + pipec_exp0 * pipe_vc0;
    pair_o1[0] = pair_o1[0] * pipec_factor1 + pipec_exp1 * pipe_vc0;
    pair_o0[1] = pair_o0[1] * pipec_factor0 + pipec_exp0 * pipe_vc1;
    pair_o1[1] = pair_o1[1] * pipec_factor1 + pipec_exp1 * pipe_vc1;
    pair_o0[2] = pair_o0[2] * pipec_factor0 + pipec_exp0 * pipe_vc2;
    pair_o1[2] = pair_o1[2] * pipec_factor1 + pipec_exp1 * pipe_vc2;
    pair_o0[3] = pair_o0[3] * pipec_factor0 + pipec_exp0 * pipe_vc3;
    pair_o1[3] = pair_o1[3] * pipec_factor1 + pipec_exp1 * pipe_vc3;

    U piped_score0 = 0;
    U piped_score1 = 0;
    piped_score0 += pair_q0[0] * pipe_kd[0];
    piped_score1 += pair_q1[0] * pipe_kd[0];
    piped_score0 += pair_q0[1] * pipe_kd[1];
    piped_score1 += pair_q1[1] * pipe_kd[1];
    piped_score0 += pair_q0[2] * pipe_kd[2];
    piped_score1 += pair_q1[2] * pipe_kd[2];
    piped_score0 += pair_q0[3] * pipe_kd[3];
    piped_score1 += pair_q1[3] * pipe_kd[3];
    piped_score0 = simd_sum(piped_score0);
    piped_score1 = simd_sum(piped_score1);

    U piped_new_max0 = metal::max(pair_max0, piped_score0);
    U piped_new_max1 = metal::max(pair_max1, piped_score1);
    U piped_factor0;
    U piped_factor1;
    LAGUNA_RESCALE(piped_factor0, pair_max0 - piped_new_max0);
    LAGUNA_RESCALE(piped_factor1, pair_max1 - piped_new_max1);
    U piped_exp0 = metal::fast::exp(piped_score0 - piped_new_max0);
    U piped_exp1 = metal::fast::exp(piped_score1 - piped_new_max1);

    pair_max0 = piped_new_max0;
    pair_max1 = piped_new_max1;
    pair_sum0 = pair_sum0 * piped_factor0 + piped_exp0;
    pair_sum1 = pair_sum1 * piped_factor1 + piped_exp1;

    pair_o0[0] = pair_o0[0] * piped_factor0 + piped_exp0 * pipe_vd0;
    pair_o1[0] = pair_o1[0] * piped_factor1 + piped_exp1 * pipe_vd0;
    pair_o0[1] = pair_o0[1] * piped_factor0 + piped_exp0 * pipe_vd1;
    pair_o1[1] = pair_o1[1] * piped_factor1 + piped_exp1 * pipe_vd1;
    pair_o0[2] = pair_o0[2] * piped_factor0 + piped_exp0 * pipe_vd2;
    pair_o1[2] = pair_o1[2] * piped_factor1 + piped_exp1 * pipe_vd2;
    pair_o0[3] = pair_o0[3] * piped_factor0 + piped_exp0 * pipe_vd3;
    pair_o1[3] = pair_o1[3] * piped_factor1 + piped_exp1 * pipe_vd3;

    pair_keys += 4 * inner_k_stride;
    pair_values += 4 * inner_v_stride;
}

if (lane == 0) {
    max_scores[sg] = pair_max0;
    max_scores[BN + sg] = pair_max1;
    sum_exp_scores[sg] = pair_sum0;
    sum_exp_scores[BN + sg] = pair_sum1;
}
outputs4[lane * BDP + sg] =
    float4(pair_o0[0], pair_o0[1], pair_o0[2], pair_o0[3]);
threadgroup_barrier(mem_flags::mem_threadgroup);

pair_max0 = max_scores[lane];
pair_max1 = max_scores[BN + lane];
U pair_global_max0 = simd_max(pair_max0);
U pair_global_max1 = simd_max(pair_max1);
U pair_global_factor0 = metal::fast::exp(pair_max0 - pair_global_max0);
U pair_global_factor1 = metal::fast::exp(pair_max1 - pair_global_max1);
pair_sum0 = simd_sum(sum_exp_scores[lane] * pair_global_factor0);
pair_sum1 = simd_sum(sum_exp_scores[BN + lane] * pair_global_factor1);

float4 pair_v0 = outputs4[sg * BDP + lane];
U acc00 = simd_sum(pair_v0.x * pair_global_factor0);
U acc01 = simd_sum(pair_v0.y * pair_global_factor0);
U acc02 = simd_sum(pair_v0.z * pair_global_factor0);
U acc03 = simd_sum(pair_v0.w * pair_global_factor0);
pair_o0[0] = pair_sum0 == 0 ? acc00 : (acc00 / pair_sum0);
pair_o0[1] = pair_sum0 == 0 ? acc01 : (acc01 / pair_sum0);
pair_o0[2] = pair_sum0 == 0 ? acc02 : (acc02 / pair_sum0);
pair_o0[3] = pair_sum0 == 0 ? acc03 : (acc03 / pair_sum0);

threadgroup_barrier(mem_flags::mem_threadgroup);
outputs4[lane * BDP + sg] =
    float4(pair_o1[0], pair_o1[1], pair_o1[2], pair_o1[3]);
threadgroup_barrier(mem_flags::mem_threadgroup);
float4 pair_v1 = outputs4[sg * BDP + lane];
U acc10 = simd_sum(pair_v1.x * pair_global_factor1);
U acc11 = simd_sum(pair_v1.y * pair_global_factor1);
U acc12 = simd_sum(pair_v1.z * pair_global_factor1);
U acc13 = simd_sum(pair_v1.w * pair_global_factor1);
pair_o1[0] = pair_sum1 == 0 ? acc10 : (acc10 / pair_sum1);
pair_o1[1] = pair_sum1 == 0 ? acc11 : (acc11 / pair_sum1);
pair_o1[2] = pair_sum1 == 0 ? acc12 : (acc12 / pair_sum1);
pair_o1[3] = pair_sum1 == 0 ? acc13 : (acc13 / pair_sum1);

if (lane == 0) {
    device bfloat* pair_out0 =
        attended + head0 * head_dim + sg * v_per_thread;
    device bfloat* pair_out1 =
        attended + head1 * head_dim + sg * v_per_thread;
    for (int p = 0; p < v_per_thread; ++p) {
        pair_out0[p] = static_cast<bfloat>(pair_o0[p]);
        pair_out1[p] = static_cast<bfloat>(pair_o1[p]);
    }
}
""",
    header: """
#define LAGUNA_RESCALE(dst, delta_expr)         \\
  do {                                          \\
    const float db_delta_ = (delta_expr);       \\
    if (as_type<uint>(db_delta_) == 0u) {       \\
      dst = float(1.0f);                        \\
    } else {                                    \\
      dst = metal::fast::exp(db_delta_);        \\
    }                                           \\
  } while (false)

#define T_LOAD_K(dst, substitute, ptr)                     \\
  do {                                                     \\
    if (substitute) {                                      \\
      dst[0] = tg_k[lane * qk_per_thread + 0];             \\
      dst[1] = tg_k[lane * qk_per_thread + 1];             \\
      dst[2] = tg_k[lane * qk_per_thread + 2];             \\
      dst[3] = tg_k[lane * qk_per_thread + 3];             \\
    } else {                                               \\
      const vec<bfloat, 4> v_ =                            \\
          *reinterpret_cast<const device vec<bfloat, 4>*>( \\
              ptr);                                        \\
      dst[0] = v_.x;                                       \\
      dst[1] = v_.y;                                       \\
      dst[2] = v_.z;                                       \\
      dst[3] = v_.w;                                       \\
    }                                                      \\
  } while (false)

#define T_LOAD_V(d0, d1, d2, d3, substitute, ptr)          \\
  do {                                                     \\
    if (substitute) {                                      \\
      d0 = tg_v[lane * v_per_thread + 0];                  \\
      d1 = tg_v[lane * v_per_thread + 1];                  \\
      d2 = tg_v[lane * v_per_thread + 2];                  \\
      d3 = tg_v[lane * v_per_thread + 3];                  \\
    } else {                                               \\
      const vec<bfloat, 4> v_ =                            \\
          *reinterpret_cast<const device vec<bfloat, 4>*>( \\
              ptr);                                        \\
      d0 = v_.x;                                           \\
      d1 = v_.y;                                           \\
      d2 = v_.z;                                           \\
      d3 = v_.w;                                           \\
    }                                                      \\
  } while (false)


""",
    ensureRowContiguous: true
)




func lagunaSlidingFusedAttention(
    rawQueries: MLXArray,
    rawKeys: MLXArray,
    rawValues: MLXArray,
    queryWeight: MLXArray,
    keyWeight: MLXArray,
    angles: MLXArray,
    cacheKeys: MLXArray,
    cacheValues: MLXArray,
    writeIdx: Int,
    scale: MLXArray
) -> MLXArray {
    let heads = LagunaConstants.slidingAttentionHeads
    let kvHeads = LagunaConstants.numKeyValueHeads
    let window = LagunaConstants.slidingWindow
    precondition(rawQueries.dtype == .bfloat16)
    precondition(rawKeys.dtype == .bfloat16)
    precondition(rawValues.dtype == .bfloat16)
    precondition(rawQueries.dims(1, 1, heads * LagunaConstants.headDim))
    precondition(rawKeys.dims(1, 1, kvHeads * LagunaConstants.headDim))
    precondition(rawValues.dims(1, 1, kvHeads * LagunaConstants.headDim))
    precondition(queryWeight.dims(LagunaConstants.headDim))
    precondition(keyWeight.dims(LagunaConstants.headDim))
    precondition(angles.dtype == .float32)
    precondition(angles.dims(1, 1, 1, LagunaConstants.headDim))
    precondition(cacheKeys.dtype == .bfloat16)
    precondition(
        cacheKeys.dims(1, kvHeads, window, LagunaConstants.headDim))
    precondition(
        cacheValues.dims(1, kvHeads, window, LagunaConstants.headDim))
    precondition(writeIdx >= 0 && writeIdx < window)
    precondition(scale.dtype == .float32 && scale.size == 1)

    lagunaTrace("sliding fused attention")
    let params = lagunaParamsAtlasEnabled
        ? lagunaRingIdxAtlas[writeIdx] : MLXArray([UInt32(writeIdx)])
    return lagunaSlidingFusedAttentionKernel(
        [
            rawQueries, rawKeys, rawValues,
            queryWeight, keyWeight, angles,
            cacheKeys, cacheValues, params, scale,
        ],
        grid: ((heads / 2) * 1024, 1, 1),
        threadGroup: (1024, 1, 1),
        outputShapes: [[1, heads, 1, LagunaConstants.headDim]],
        outputDTypes: [.bfloat16]
    )[0]
}








private enum LagunaRingIdxAtlasStore {
    nonisolated(unsafe) static let entries: [MLXArray] = {
        let atlas = (0..<LagunaConstants.slidingWindow).map {
            MLXArray([UInt32($0)])
        }
        for entry in atlas { eval(entry) }
        return atlas
    }()
}

private var lagunaRingIdxAtlas: [MLXArray] { LagunaRingIdxAtlasStore.entries }



let lagunaParamsAtlasEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PARAMS_ATLAS"] != "0"










let lagunaFusedFullAttentionEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_FULL_ATTN"] != "0"




let lagunaFusedFullAttentionWholeModelWarmupEnabled =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_FUSED_FULL_ATTN_WHOLE_MODEL_WARMUP"] == "1"




let lagunaFusedFullAttentionKernelWarmupEnabled =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_FUSED_FULL_ATTN_KERNEL_WARMUP"] != "0"

private let lagunaFullFusedAttentionKernel = MLXFast.metalKernel(
    name: "laguna_full_fused_attn_grow_v1",
    inputNames: [
        "raw_queries", "raw_keys", "raw_values",
        "query_weight", "key_weight", "angles",
        "k_cache", "v_cache", "params", "scale_arr",
    ],
    outputNames: ["attended"],
    source: """
constexpr uint head_dim = 128;
constexpr uint gqa = 6;
constexpr int BN = 32;
constexpr int BD = 32;
constexpr int BDP = BD + 1;
constexpr int qk_per_thread = 4;
constexpr int v_per_thread = 4;
constexpr uint rotary_pairs = 32;
constexpr float yarn_mscale = 1.3465735912322998f;

typedef float U;

uint pair_tg = threadgroup_position_in_grid.x;
uint head0 = pair_tg * 2;
uint head1 = head0 + 1;
uint kv_head = head0 / gqa;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint widx = params[0];
int N = int(params[1]);
uint capacity = params[2];
float scale = scale_arr[0];

threadgroup bfloat tg_q0[head_dim];
threadgroup bfloat tg_q1[head_dim];
threadgroup bfloat tg_k[head_dim];
threadgroup bfloat tg_v[head_dim];

if (sg < 3) {
    const device bfloat* input =
        sg == 0 ? raw_queries + head0 * head_dim
        : sg == 1 ? raw_queries + head1 * head_dim
                  : raw_keys + kv_head * head_dim;
    const device bfloat* weight =
        sg == 2 ? key_weight : query_weight;
    threadgroup bfloat* outrow =
        sg == 0 ? tg_q0 : sg == 1 ? tg_q1 : tg_k;

    uint base = lane * 4;
    thread bfloat normalized[4];
    float sum = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        float value = float(input[base + i]);
        sum += value * value;
    }
    sum = simd_sum(sum);
    float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);
    for (uint i = 0; i < 4; ++i) {
        normalized[i] =
            weight[base + i] *
            bfloat(float(input[base + i]) * inverse_rms);
    }
    thread float paired[4];
    for (uint i = 0; i < 4; ++i) {
        paired[i] = simd_shuffle(float(normalized[i]), lane ^ 8);
    }
    if (lane < 8) {
        bfloat rounded_mscale = bfloat(yarn_mscale);
        for (uint i = 0; i < 4; ++i) {
            uint pair = base + i;
            float first =
                float(bfloat(normalized[i] * rounded_mscale));
            float second =
                float(bfloat(bfloat(paired[i]) * rounded_mscale));
            float cosine = angles[pair];
            float sine = angles[pair + rotary_pairs];
            outrow[pair] = bfloat(first * cosine - second * sine);
            outrow[pair + rotary_pairs] =
                bfloat(first * sine + second * cosine);
        }
    } else if (lane >= 16) {
        for (uint i = 0; i < 4; ++i) {
            outrow[base + i] = normalized[i];
        }
    }
} else if (sg == 3) {
    const device bfloat* vin = raw_values + kv_head * head_dim;
    for (uint i = lane; i < head_dim; i += 32) {
        tg_v[i] = vin[i];
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if ((head0 % gqa) == 0 && sg == 0) {
    device bfloat* kc = (device bfloat*)k_cache +
        (size_t)kv_head * (capacity * head_dim) +
        (size_t)widx * head_dim;
    device bfloat* vc = (device bfloat*)v_cache +
        (size_t)kv_head * (capacity * head_dim) +
        (size_t)widx * head_dim;
    for (uint i = lane; i < head_dim; i += 32) {
        kc[i] = tg_k[i];
        vc[i] = tg_v[i];
    }
}

threadgroup float4 outputs4[BN * BDP];
threadgroup U max_scores[2 * BN];
threadgroup U sum_exp_scores[2 * BN];

const device bfloat* pair_keys = k_cache +
    (size_t)kv_head * (capacity * head_dim) +
    (size_t)sg * head_dim + lane * qk_per_thread;
const device bfloat* pair_values = v_cache +
    (size_t)kv_head * (capacity * head_dim) +
    (size_t)sg * head_dim + lane * v_per_thread;
const int inner_k_stride = BN * int(head_dim);
const int inner_v_stride = BN * int(head_dim);

thread U pair_q0[qk_per_thread];
thread U pair_q1[qk_per_thread];
thread U pair_k[qk_per_thread];
thread U pair_o0[v_per_thread];
thread U pair_o1[v_per_thread];

for (int j = 0; j < qk_per_thread; ++j) {
    pair_q0[j] =
        static_cast<U>(scale) * tg_q0[lane * qk_per_thread + j];
    pair_q1[j] =
        static_cast<U>(scale) * tg_q1[lane * qk_per_thread + j];
}
for (int j = 0; j < v_per_thread; ++j) {
    pair_o0[j] = 0;
    pair_o1[j] = 0;
}

U pair_max0 = metal::numeric_limits<U>::lowest();
U pair_max1 = metal::numeric_limits<U>::lowest();
U pair_sum0 = 0;
U pair_sum1 = 0;

int i = sg;
for (; i + BN < N; i += 2 * BN) {
    const device bfloat* pipe_keys_b = pair_keys + inner_k_stride;
    const device bfloat* pipe_values_b = pair_values + inner_v_stride;
    const bool sub_a = uint(i) == widx;
    const bool sub_b = uint(i + BN) == widx;
    U pipe_ka[4];
    U pipe_kb[4];
    T_LOAD_K(pipe_ka, sub_a, pair_keys);
    T_LOAD_K(pipe_kb, sub_b, pipe_keys_b);
    bfloat pipe_va0, pipe_va1, pipe_va2, pipe_va3;
    bfloat pipe_vb0, pipe_vb1, pipe_vb2, pipe_vb3;
    T_LOAD_V(pipe_va0, pipe_va1, pipe_va2, pipe_va3, sub_a,
        pair_values);
    T_LOAD_V(pipe_vb0, pipe_vb1, pipe_vb2, pipe_vb3, sub_b,
        pipe_values_b);

    U pair_score0 = 0;
    U pair_score1 = 0;
    pair_score0 += pair_q0[0] * pipe_ka[0];
    pair_score1 += pair_q1[0] * pipe_ka[0];
    pair_score0 += pair_q0[1] * pipe_ka[1];
    pair_score1 += pair_q1[1] * pipe_ka[1];
    pair_score0 += pair_q0[2] * pipe_ka[2];
    pair_score1 += pair_q1[2] * pipe_ka[2];
    pair_score0 += pair_q0[3] * pipe_ka[3];
    pair_score1 += pair_q1[3] * pipe_ka[3];
    pair_score0 = simd_sum(pair_score0);
    pair_score1 = simd_sum(pair_score1);

    U pair_new_max0 = metal::max(pair_max0, pair_score0);
    U pair_new_max1 = metal::max(pair_max1, pair_score1);
    U pair_factor0;
    U pair_factor1;
    LAGUNA_RESCALE(pair_factor0, pair_max0 - pair_new_max0);
    LAGUNA_RESCALE(pair_factor1, pair_max1 - pair_new_max1);
    U pair_exp0 = metal::fast::exp(pair_score0 - pair_new_max0);
    U pair_exp1 = metal::fast::exp(pair_score1 - pair_new_max1);

    pair_max0 = pair_new_max0;
    pair_max1 = pair_new_max1;
    pair_sum0 = pair_sum0 * pair_factor0 + pair_exp0;
    pair_sum1 = pair_sum1 * pair_factor1 + pair_exp1;

    pair_o0[0] = pair_o0[0] * pair_factor0 + pair_exp0 * pipe_va0;
    pair_o1[0] = pair_o1[0] * pair_factor1 + pair_exp1 * pipe_va0;
    pair_o0[1] = pair_o0[1] * pair_factor0 + pair_exp0 * pipe_va1;
    pair_o1[1] = pair_o1[1] * pair_factor1 + pair_exp1 * pipe_va1;
    pair_o0[2] = pair_o0[2] * pair_factor0 + pair_exp0 * pipe_va2;
    pair_o1[2] = pair_o1[2] * pair_factor1 + pair_exp1 * pipe_va2;
    pair_o0[3] = pair_o0[3] * pair_factor0 + pair_exp0 * pipe_va3;
    pair_o1[3] = pair_o1[3] * pair_factor1 + pair_exp1 * pipe_va3;

    U pipeb_score0 = 0;
    U pipeb_score1 = 0;
    pipeb_score0 += pair_q0[0] * pipe_kb[0];
    pipeb_score1 += pair_q1[0] * pipe_kb[0];
    pipeb_score0 += pair_q0[1] * pipe_kb[1];
    pipeb_score1 += pair_q1[1] * pipe_kb[1];
    pipeb_score0 += pair_q0[2] * pipe_kb[2];
    pipeb_score1 += pair_q1[2] * pipe_kb[2];
    pipeb_score0 += pair_q0[3] * pipe_kb[3];
    pipeb_score1 += pair_q1[3] * pipe_kb[3];
    pipeb_score0 = simd_sum(pipeb_score0);
    pipeb_score1 = simd_sum(pipeb_score1);

    U pipeb_new_max0 = metal::max(pair_max0, pipeb_score0);
    U pipeb_new_max1 = metal::max(pair_max1, pipeb_score1);
    U pipeb_factor0;
    U pipeb_factor1;
    LAGUNA_RESCALE(pipeb_factor0, pair_max0 - pipeb_new_max0);
    LAGUNA_RESCALE(pipeb_factor1, pair_max1 - pipeb_new_max1);
    U pipeb_exp0 = metal::fast::exp(pipeb_score0 - pipeb_new_max0);
    U pipeb_exp1 = metal::fast::exp(pipeb_score1 - pipeb_new_max1);

    pair_max0 = pipeb_new_max0;
    pair_max1 = pipeb_new_max1;
    pair_sum0 = pair_sum0 * pipeb_factor0 + pipeb_exp0;
    pair_sum1 = pair_sum1 * pipeb_factor1 + pipeb_exp1;

    pair_o0[0] = pair_o0[0] * pipeb_factor0 + pipeb_exp0 * pipe_vb0;
    pair_o1[0] = pair_o1[0] * pipeb_factor1 + pipeb_exp1 * pipe_vb0;
    pair_o0[1] = pair_o0[1] * pipeb_factor0 + pipeb_exp0 * pipe_vb1;
    pair_o1[1] = pair_o1[1] * pipeb_factor1 + pipeb_exp1 * pipe_vb1;
    pair_o0[2] = pair_o0[2] * pipeb_factor0 + pipeb_exp0 * pipe_vb2;
    pair_o1[2] = pair_o1[2] * pipeb_factor1 + pipeb_exp1 * pipe_vb2;
    pair_o0[3] = pair_o0[3] * pipeb_factor0 + pipeb_exp0 * pipe_vb3;
    pair_o1[3] = pair_o1[3] * pipeb_factor1 + pipeb_exp1 * pipe_vb3;

    pair_keys += 2 * inner_k_stride;
    pair_values += 2 * inner_v_stride;
}
if (i < N) {
    const bool sub_t = uint(i) == widx;
    T_LOAD_K(pair_k, sub_t, pair_keys);
    bfloat pipe_va0, pipe_va1, pipe_va2, pipe_va3;
    T_LOAD_V(pipe_va0, pipe_va1, pipe_va2, pipe_va3, sub_t,
        pair_values);

    U pair_score0 = 0;
    U pair_score1 = 0;
    pair_score0 += pair_q0[0] * pair_k[0];
    pair_score1 += pair_q1[0] * pair_k[0];
    pair_score0 += pair_q0[1] * pair_k[1];
    pair_score1 += pair_q1[1] * pair_k[1];
    pair_score0 += pair_q0[2] * pair_k[2];
    pair_score1 += pair_q1[2] * pair_k[2];
    pair_score0 += pair_q0[3] * pair_k[3];
    pair_score1 += pair_q1[3] * pair_k[3];
    pair_score0 = simd_sum(pair_score0);
    pair_score1 = simd_sum(pair_score1);

    U pair_new_max0 = metal::max(pair_max0, pair_score0);
    U pair_new_max1 = metal::max(pair_max1, pair_score1);
    U pair_factor0;
    U pair_factor1;
    LAGUNA_RESCALE(pair_factor0, pair_max0 - pair_new_max0);
    LAGUNA_RESCALE(pair_factor1, pair_max1 - pair_new_max1);
    U pair_exp0 = metal::fast::exp(pair_score0 - pair_new_max0);
    U pair_exp1 = metal::fast::exp(pair_score1 - pair_new_max1);

    pair_max0 = pair_new_max0;
    pair_max1 = pair_new_max1;
    pair_sum0 = pair_sum0 * pair_factor0 + pair_exp0;
    pair_sum1 = pair_sum1 * pair_factor1 + pair_exp1;

    pair_o0[0] = pair_o0[0] * pair_factor0 + pair_exp0 * pipe_va0;
    pair_o1[0] = pair_o1[0] * pair_factor1 + pair_exp1 * pipe_va0;
    pair_o0[1] = pair_o0[1] * pair_factor0 + pair_exp0 * pipe_va1;
    pair_o1[1] = pair_o1[1] * pair_factor1 + pair_exp1 * pipe_va1;
    pair_o0[2] = pair_o0[2] * pair_factor0 + pair_exp0 * pipe_va2;
    pair_o1[2] = pair_o1[2] * pair_factor1 + pair_exp1 * pipe_va2;
    pair_o0[3] = pair_o0[3] * pair_factor0 + pair_exp0 * pipe_va3;
    pair_o1[3] = pair_o1[3] * pair_factor1 + pair_exp1 * pipe_va3;
}

if (lane == 0) {
    max_scores[sg] = pair_max0;
    max_scores[BN + sg] = pair_max1;
    sum_exp_scores[sg] = pair_sum0;
    sum_exp_scores[BN + sg] = pair_sum1;
}
outputs4[lane * BDP + sg] =
    float4(pair_o0[0], pair_o0[1], pair_o0[2], pair_o0[3]);
threadgroup_barrier(mem_flags::mem_threadgroup);

pair_max0 = max_scores[lane];
pair_max1 = max_scores[BN + lane];
U pair_global_max0 = simd_max(pair_max0);
U pair_global_max1 = simd_max(pair_max1);
U pair_global_factor0 = metal::fast::exp(pair_max0 - pair_global_max0);
U pair_global_factor1 = metal::fast::exp(pair_max1 - pair_global_max1);
pair_sum0 = simd_sum(sum_exp_scores[lane] * pair_global_factor0);
pair_sum1 = simd_sum(sum_exp_scores[BN + lane] * pair_global_factor1);

float4 pair_v0 = outputs4[sg * BDP + lane];
U acc00 = simd_sum(pair_v0.x * pair_global_factor0);
U acc01 = simd_sum(pair_v0.y * pair_global_factor0);
U acc02 = simd_sum(pair_v0.z * pair_global_factor0);
U acc03 = simd_sum(pair_v0.w * pair_global_factor0);
pair_o0[0] = pair_sum0 == 0 ? acc00 : (acc00 / pair_sum0);
pair_o0[1] = pair_sum0 == 0 ? acc01 : (acc01 / pair_sum0);
pair_o0[2] = pair_sum0 == 0 ? acc02 : (acc02 / pair_sum0);
pair_o0[3] = pair_sum0 == 0 ? acc03 : (acc03 / pair_sum0);

threadgroup_barrier(mem_flags::mem_threadgroup);
outputs4[lane * BDP + sg] =
    float4(pair_o1[0], pair_o1[1], pair_o1[2], pair_o1[3]);
threadgroup_barrier(mem_flags::mem_threadgroup);
float4 pair_v1 = outputs4[sg * BDP + lane];
U acc10 = simd_sum(pair_v1.x * pair_global_factor1);
U acc11 = simd_sum(pair_v1.y * pair_global_factor1);
U acc12 = simd_sum(pair_v1.z * pair_global_factor1);
U acc13 = simd_sum(pair_v1.w * pair_global_factor1);
pair_o1[0] = pair_sum1 == 0 ? acc10 : (acc10 / pair_sum1);
pair_o1[1] = pair_sum1 == 0 ? acc11 : (acc11 / pair_sum1);
pair_o1[2] = pair_sum1 == 0 ? acc12 : (acc12 / pair_sum1);
pair_o1[3] = pair_sum1 == 0 ? acc13 : (acc13 / pair_sum1);

if (lane == 0) {
    device bfloat* pair_out0 =
        attended + head0 * head_dim + sg * v_per_thread;
    device bfloat* pair_out1 =
        attended + head1 * head_dim + sg * v_per_thread;
    for (int p = 0; p < v_per_thread; ++p) {
        pair_out0[p] = static_cast<bfloat>(pair_o0[p]);
        pair_out1[p] = static_cast<bfloat>(pair_o1[p]);
    }
}
""",
    header: """
#define LAGUNA_RESCALE(dst, delta_expr)         \\
  do {                                          \\
    const float db_delta_ = (delta_expr);       \\
    if (as_type<uint>(db_delta_) == 0u) {       \\
      dst = float(1.0f);                        \\
    } else {                                    \\
      dst = metal::fast::exp(db_delta_);        \\
    }                                           \\
  } while (false)

#define T_LOAD_K(dst, substitute, ptr)                     \\
  do {                                                     \\
    if (substitute) {                                      \\
      dst[0] = tg_k[lane * qk_per_thread + 0];             \\
      dst[1] = tg_k[lane * qk_per_thread + 1];             \\
      dst[2] = tg_k[lane * qk_per_thread + 2];             \\
      dst[3] = tg_k[lane * qk_per_thread + 3];             \\
    } else {                                               \\
      const vec<bfloat, 4> v_ =                            \\
          *reinterpret_cast<const device vec<bfloat, 4>*>( \\
              ptr);                                        \\
      dst[0] = v_.x;                                       \\
      dst[1] = v_.y;                                       \\
      dst[2] = v_.z;                                       \\
      dst[3] = v_.w;                                       \\
    }                                                      \\
  } while (false)

#define T_LOAD_V(d0, d1, d2, d3, substitute, ptr)          \\
  do {                                                     \\
    if (substitute) {                                      \\
      d0 = tg_v[lane * v_per_thread + 0];                  \\
      d1 = tg_v[lane * v_per_thread + 1];                  \\
      d2 = tg_v[lane * v_per_thread + 2];                  \\
      d3 = tg_v[lane * v_per_thread + 3];                  \\
    } else {                                               \\
      const vec<bfloat, 4> v_ =                            \\
          *reinterpret_cast<const device vec<bfloat, 4>*>( \\
              ptr);                                        \\
      d0 = v_.x;                                           \\
      d1 = v_.y;                                           \\
      d2 = v_.z;                                           \\
      d3 = v_.w;                                           \\
    }                                                      \\
  } while (false)


""",
    ensureRowContiguous: true
)




func lagunaFullFusedAttention(
    rawQueries: MLXArray,
    rawKeys: MLXArray,
    rawValues: MLXArray,
    queryWeight: MLXArray,
    keyWeight: MLXArray,
    angles: MLXArray,
    cacheKeys: MLXArray,
    cacheValues: MLXArray,
    writeIdx: Int,
    scale: MLXArray
) -> MLXArray {
    let heads = LagunaConstants.fullAttentionHeads
    let kvHeads = LagunaConstants.numKeyValueHeads
    let capacity = cacheKeys.dim(2)
    precondition(rawQueries.dtype == .bfloat16)
    precondition(rawKeys.dtype == .bfloat16)
    precondition(rawValues.dtype == .bfloat16)
    precondition(rawQueries.dims(1, 1, heads * LagunaConstants.headDim))
    precondition(rawKeys.dims(1, 1, kvHeads * LagunaConstants.headDim))
    precondition(rawValues.dims(1, 1, kvHeads * LagunaConstants.headDim))
    precondition(queryWeight.dims(LagunaConstants.headDim))
    precondition(keyWeight.dims(LagunaConstants.headDim))
    precondition(angles.dtype == .float32)
    precondition(angles.dims(1, 1, 1, LagunaConstants.headDim / 2))
    precondition(cacheKeys.dtype == .bfloat16)
    precondition(
        cacheKeys.dims(1, kvHeads, capacity, LagunaConstants.headDim))
    precondition(
        cacheValues.dims(1, kvHeads, capacity, LagunaConstants.headDim))
    precondition(writeIdx >= 0 && writeIdx < capacity)
    precondition(scale.dtype == .float32 && scale.size == 1)

    lagunaTrace("full fused attention")
    let params = MLXArray([
        UInt32(writeIdx), UInt32(writeIdx + 1), UInt32(capacity),
    ])
    return lagunaFullFusedAttentionKernel(
        [
            rawQueries, rawKeys, rawValues,
            queryWeight, keyWeight, angles,
            cacheKeys, cacheValues, params, scale,
        ],
        grid: ((heads / 2) * 1024, 1, 1),
        threadGroup: (1024, 1, 1),
        outputShapes: [[1, heads, 1, LagunaConstants.headDim]],
        outputDTypes: [.bfloat16]
    )[0]
}





func lagunaWarmFullFusedAttentionKernel() {
    let headDim = LagunaConstants.headDim
    let heads = LagunaConstants.fullAttentionHeads
    let kvHeads = LagunaConstants.numKeyValueHeads
    let rawQueries = MLXArray.zeros(
        [1, 1, heads * headDim], dtype: .bfloat16)
    let rawKeys = MLXArray.zeros(
        [1, 1, kvHeads * headDim], dtype: .bfloat16)
    let rawValues = MLXArray.zeros(
        [1, 1, kvHeads * headDim], dtype: .bfloat16)
    let queryWeight = MLXArray.ones([headDim], dtype: .bfloat16)
    let keyWeight = MLXArray.ones([headDim], dtype: .bfloat16)
    let angles = MLXArray.zeros(
        [1, 1, 1, headDim / 2], dtype: .float32)
    let cacheKeys = MLXArray.zeros(
        [1, kvHeads, 2, headDim], dtype: .bfloat16)
    let cacheValues = MLXArray.zeros(
        [1, kvHeads, 2, headDim], dtype: .bfloat16)
    let scale = MLXArray([pow(Float(headDim), -0.5)])
    eval(lagunaFullFusedAttention(
        rawQueries: rawQueries,
        rawKeys: rawKeys,
        rawValues: rawValues,
        queryWeight: queryWeight,
        keyWeight: keyWeight,
        angles: angles,
        cacheKeys: cacheKeys,
        cacheValues: cacheValues,
        writeIdx: 1,
        scale: scale
    ))
}


























private let lagunaPrefillSlidingQKNormRoPEKernel = MLXFast.metalKernel(
    name: "laguna_prefill_sliding_qk_norm_rope_bf16_128_v2",
    inputNames: [
        "raw_queries", "raw_keys", "query_weight", "key_weight", "angles",
        "offsets",
    ],
    outputNames: ["queries", "keys"],
    source: """
constexpr uint head_dim = 128;
constexpr uint rotary_pairs = 64;
constexpr uint query_heads = 64;
constexpr uint kv_heads = 8;

uint t = threadgroup_position_in_grid.y;
uint length = threadgroups_per_grid.y;
uint head = threadgroup_position_in_grid.x * 4
    + simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;

const device bfloat* input;
const device bfloat* weight;
device bfloat* output;
if (head < query_heads) {
    input = raw_queries + (t * query_heads + head) * head_dim;
    weight = query_weight;
    output = queries + (head * length + t) * head_dim;
} else {
    uint khead = head - query_heads;
    input = raw_keys + (t * kv_heads + khead) * head_dim;
    weight = key_weight;
    output = keys + (khead * length + t) * head_dim;
}

uint base = lane * 4;
thread bfloat normalized[4];
float sum = 0.0f;
#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    float value = float(input[base + i]);
    sum += value * value;
}
sum = simd_sum(sum);
float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    normalized[i] =
        weight[base + i] *
        bfloat(float(input[base + i]) * inverse_rms);
}

thread float paired[4];
#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    paired[i] = simd_shuffle(float(normalized[i]), lane ^ 16);
}

const device float* angle_row =
    angles + (uint(offsets[0]) + t) * (2 * rotary_pairs);
if (lane < 16) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        uint pair = base + i;
        float first = float(normalized[i]);
        float second = paired[i];
        float cosine = angle_row[pair];
        float sine = angle_row[pair + rotary_pairs];
        output[pair] = bfloat(first * cosine - second * sine);
        output[pair + rotary_pairs] =
            bfloat(first * sine + second * cosine);
    }
}
""",
    ensureRowContiguous: true
)










private let lagunaPrefillSlidingQKNormRoPEH1Kernel = MLXFast.metalKernel(
    name: "laguna_prefill_sliding_qk_norm_rope_bf16_128_h1_v2",
    inputNames: [
        "raw_queries", "raw_keys", "query_weight", "key_weight", "angles",
        "offsets",
    ],
    outputNames: ["queries", "keys"],
    source: """
constexpr uint head_dim = 128;
constexpr uint rotary_pairs = 64;
constexpr uint query_heads = 64;
constexpr uint kv_heads = 8;

uint t = threadgroup_position_in_grid.y;
uint length = threadgroups_per_grid.y;
uint head = threadgroup_position_in_grid.x;
uint lane = thread_index_in_simdgroup;

const device bfloat* input;
const device bfloat* weight;
device bfloat* output;
if (head < query_heads) {
    input = raw_queries + (t * query_heads + head) * head_dim;
    weight = query_weight;
    output = queries + (head * length + t) * head_dim;
} else {
    uint khead = head - query_heads;
    input = raw_keys + (t * kv_heads + khead) * head_dim;
    weight = key_weight;
    output = keys + (khead * length + t) * head_dim;
}

uint base = lane * 4;
thread bfloat normalized[4];
float sum = 0.0f;
#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    float value = float(input[base + i]);
    sum += value * value;
}
sum = simd_sum(sum);
float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    normalized[i] =
        weight[base + i] *
        bfloat(float(input[base + i]) * inverse_rms);
}

thread float paired[4];
#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    paired[i] = simd_shuffle(float(normalized[i]), lane ^ 16);
}

const device float* angle_row =
    angles + (uint(offsets[0]) + t) * (2 * rotary_pairs);
if (lane < 16) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        uint pair = base + i;
        float first = float(normalized[i]);
        float second = paired[i];
        float cosine = angle_row[pair];
        float sine = angle_row[pair + rotary_pairs];
        output[pair] = bfloat(first * cosine - second * sine);
        output[pair + rotary_pairs] =
            bfloat(first * sine + second * cosine);
    }
}
""",
    ensureRowContiguous: true
)















private let lagunaPrefillFullQKNormYaRNKernel = MLXFast.metalKernel(
    name: "laguna_prefill_full_qk_norm_yarn_bf16_128_v2",
    inputNames: [
        "raw_queries", "raw_keys", "query_weight", "key_weight", "angles",
        "offsets",
    ],
    outputNames: ["queries", "keys"],
    source: """
constexpr uint head_dim = 128;
constexpr uint rotary_pairs = 32;
constexpr uint query_heads = 48;
constexpr uint kv_heads = 8;
constexpr float yarn_mscale = 1.3465735912322998f;

uint t = threadgroup_position_in_grid.y;
uint length = threadgroups_per_grid.y;
uint head = threadgroup_position_in_grid.x * 4
    + simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;

const device bfloat* input;
const device bfloat* weight;
device bfloat* output;
if (head < query_heads) {
    input = raw_queries + (t * query_heads + head) * head_dim;
    weight = query_weight;
    output = queries + (head * length + t) * head_dim;
} else {
    uint khead = head - query_heads;
    input = raw_keys + (t * kv_heads + khead) * head_dim;
    weight = key_weight;
    output = keys + (khead * length + t) * head_dim;
}

uint base = lane * 4;
thread bfloat normalized[4];
float sum = 0.0f;
#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    float value = float(input[base + i]);
    sum += value * value;
}
sum = simd_sum(sum);
float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    normalized[i] =
        weight[base + i] *
        bfloat(float(input[base + i]) * inverse_rms);
}

thread float paired[4];
#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    paired[i] = simd_shuffle(float(normalized[i]), lane ^ 8);
}

const device float* angle_row =
    angles + (uint(offsets[0]) + t) * (2 * rotary_pairs);
if (lane < 8) {
    bfloat rounded_mscale = bfloat(yarn_mscale);
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        uint pair = base + i;
        float first =
            float(bfloat(normalized[i] * rounded_mscale));
        float second =
            float(bfloat(bfloat(paired[i]) * rounded_mscale));
        float cosine = angle_row[pair];
        float sine = angle_row[pair + rotary_pairs];
        output[pair] = bfloat(first * cosine - second * sine);
        output[pair + rotary_pairs] =
            bfloat(first * sine + second * cosine);
    }
} else if (lane >= 16) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        output[base + i] = normalized[i];
    }
}
""",
    ensureRowContiguous: true
)







private let lagunaPrefillFullQKNormYaRNH1Kernel = MLXFast.metalKernel(
    name: "laguna_prefill_full_qk_norm_yarn_bf16_128_h1_v2",
    inputNames: [
        "raw_queries", "raw_keys", "query_weight", "key_weight", "angles",
        "offsets",
    ],
    outputNames: ["queries", "keys"],
    source: """
constexpr uint head_dim = 128;
constexpr uint rotary_pairs = 32;
constexpr uint query_heads = 48;
constexpr uint kv_heads = 8;
constexpr float yarn_mscale = 1.3465735912322998f;

uint t = threadgroup_position_in_grid.y;
uint length = threadgroups_per_grid.y;
uint head = threadgroup_position_in_grid.x;
uint lane = thread_index_in_simdgroup;

const device bfloat* input;
const device bfloat* weight;
device bfloat* output;
if (head < query_heads) {
    input = raw_queries + (t * query_heads + head) * head_dim;
    weight = query_weight;
    output = queries + (head * length + t) * head_dim;
} else {
    uint khead = head - query_heads;
    input = raw_keys + (t * kv_heads + khead) * head_dim;
    weight = key_weight;
    output = keys + (khead * length + t) * head_dim;
}

uint base = lane * 4;
thread bfloat normalized[4];
float sum = 0.0f;
#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    float value = float(input[base + i]);
    sum += value * value;
}
sum = simd_sum(sum);
float inverse_rms = metal::precise::rsqrt(sum / 128.0f + 1.0e-6f);

#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    normalized[i] =
        weight[base + i] *
        bfloat(float(input[base + i]) * inverse_rms);
}

thread float paired[4];
#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
    paired[i] = simd_shuffle(float(normalized[i]), lane ^ 8);
}

const device float* angle_row =
    angles + (uint(offsets[0]) + t) * (2 * rotary_pairs);
if (lane < 8) {
    bfloat rounded_mscale = bfloat(yarn_mscale);
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        uint pair = base + i;
        float first =
            float(bfloat(normalized[i] * rounded_mscale));
        float second =
            float(bfloat(bfloat(paired[i]) * rounded_mscale));
        float cosine = angle_row[pair];
        float sine = angle_row[pair + rotary_pairs];
        output[pair] = bfloat(first * cosine - second * sine);
        output[pair + rotary_pairs] =
            bfloat(first * sine + second * cosine);
    }
} else if (lane >= 16) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        output[base + i] = normalized[i];
    }
}
""",
    ensureRowContiguous: true
)

private func lagunaPrefillSlidingQKNormRoPE(
    rawQueries: MLXArray,
    rawKeys: MLXArray,
    queryWeight: MLXArray,
    keyWeight: MLXArray,
    angles: MLXArray,
    offsets: MLXArray,
    length: Int
) -> (MLXArray, MLXArray) {
    let heads = LagunaConstants.slidingAttentionHeads
    let kvHeads = LagunaConstants.numKeyValueHeads
    precondition(rawQueries.dtype == .bfloat16)
    precondition(rawKeys.dtype == .bfloat16)
    precondition(queryWeight.dtype == .bfloat16)
    precondition(keyWeight.dtype == .bfloat16)
    precondition(rawQueries.dims(1, length, heads * LagunaConstants.headDim))
    precondition(rawKeys.dims(1, length, kvHeads * LagunaConstants.headDim))
    precondition(queryWeight.dims(LagunaConstants.headDim))
    precondition(keyWeight.dims(LagunaConstants.headDim))
    precondition(angles.dtype == .float32)
    precondition(
        angles.dims(1, 1, lagunaRoPEAngleAtlasLength, LagunaConstants.headDim))
    precondition(offsets.dtype == .int32 && offsets.size == 1)
    precondition((heads + kvHeads) % 4 == 0)

    lagunaTrace("prefill sliding qk norm+rope")
    let useH1 = lagunaPrefillQKHeadsPerGroup == 1
    let headsPerGroup = useH1 ? 1 : 4
    let threadGroupSize = headsPerGroup * 32
    let kernel = useH1
        ? lagunaPrefillSlidingQKNormRoPEH1Kernel
        : lagunaPrefillSlidingQKNormRoPEKernel
    let outputs = kernel(
        [rawQueries, rawKeys, queryWeight, keyWeight, angles, offsets],
        grid: ((heads + kvHeads) / headsPerGroup * threadGroupSize, length, 1),
        threadGroup: (threadGroupSize, 1, 1),
        outputShapes: [
            [1, heads, length, LagunaConstants.headDim],
            [1, kvHeads, length, LagunaConstants.headDim],
        ],
        outputDTypes: [.bfloat16, .bfloat16]
    )
    return (outputs[0], outputs[1])
}

private func lagunaPrefillFullQKNormYaRN(
    rawQueries: MLXArray,
    rawKeys: MLXArray,
    queryWeight: MLXArray,
    keyWeight: MLXArray,
    angles: MLXArray,
    offsets: MLXArray,
    length: Int
) -> (MLXArray, MLXArray) {
    let heads = LagunaConstants.fullAttentionHeads
    let kvHeads = LagunaConstants.numKeyValueHeads
    precondition(rawQueries.dtype == .bfloat16)
    precondition(rawKeys.dtype == .bfloat16)
    precondition(queryWeight.dtype == .bfloat16)
    precondition(keyWeight.dtype == .bfloat16)
    precondition(rawQueries.dims(1, length, heads * LagunaConstants.headDim))
    precondition(rawKeys.dims(1, length, kvHeads * LagunaConstants.headDim))
    precondition(queryWeight.dims(LagunaConstants.headDim))
    precondition(keyWeight.dims(LagunaConstants.headDim))
    precondition(angles.dtype == .float32)
    precondition(
        angles.dims(1, 1, lagunaRoPEAngleAtlasLength, LagunaConstants.headDim / 2))
    precondition(offsets.dtype == .int32 && offsets.size == 1)
    precondition((heads + kvHeads) % 4 == 0)

    lagunaTrace("prefill full qk norm+yarn")
    let useH1 = lagunaPrefillQKHeadsPerGroup == 1
    let headsPerGroup = useH1 ? 1 : 4
    let threadGroupSize = headsPerGroup * 32
    let kernel = useH1
        ? lagunaPrefillFullQKNormYaRNH1Kernel
        : lagunaPrefillFullQKNormYaRNKernel
    let outputs = kernel(
        [rawQueries, rawKeys, queryWeight, keyWeight, angles, offsets],
        grid: ((heads + kvHeads) / headsPerGroup * threadGroupSize, length, 1),
        threadGroup: (threadGroupSize, 1, 1),
        outputShapes: [
            [1, heads, length, LagunaConstants.headDim],
            [1, kvHeads, length, LagunaConstants.headDim],
        ],
        outputDTypes: [.bfloat16, .bfloat16]
    )
    return (outputs[0], outputs[1])
}

struct LagunaIndexedAffineMetadata {
    let indices: MLXArray
    let lut: MLXArray

    var arrays: [MLXArray] { [indices, lut] }
}

private let lagunaAffineMetadataIndexedEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_AFFINE_METADATA_INDEXED"] != "0"


private func lagunaIndexedAffineMetadata(
    scales: MLXArray, biases: MLXArray
) -> LagunaIndexedAffineMetadata? {
    guard lagunaAffineMetadataIndexedEnabled,
        scales.dtype == .bfloat16, biases.dtype == .bfloat16,
        scales.sameDims(biases), scales.size > 0
    else {
        return nil
    }

    let scaleBits = scales.view(dtype: .uint16).asArray(UInt16.self)
    let biasBits = biases.view(dtype: .uint16).asArray(UInt16.self)
    guard scaleBits.count == biasBits.count else { return nil }

    var pairToIndex: [UInt32: UInt16] = [:]
    pairToIndex.reserveCapacity(min(scaleBits.count, 65_536))
    var lut: [UInt32] = []
    lut.reserveCapacity(min(scaleBits.count, 65_536))
    var indices: [UInt16] = []
    indices.reserveCapacity(scaleBits.count)

    for i in scaleBits.indices {
        let pair = UInt32(scaleBits[i]) | (UInt32(biasBits[i]) << 16)
        if let index = pairToIndex[pair] {
            indices.append(index)
            continue
        }
        guard lut.count < 65_536 else { return nil }
        let index = UInt16(lut.count)
        pairToIndex[pair] = index
        lut.append(pair)
        indices.append(index)
    }

    return LagunaIndexedAffineMetadata(
        indices: MLXArray(indices).reshaped(scales.shape),
        lut: MLXArray(lut)
    )
}

struct LagunaNativeAffineWeight {
    let packedCodes: MLXArray
    let scales: MLXArray
    let biases: MLXArray?
    let originalShape: [Int]

    var groupSize: Int = 32
    var bits: Int = 8
    var mode: QuantizationMode = .affine
    var indexedMetadata: LagunaIndexedAffineMetadata? = nil



    var narrowScales: LagunaNarrowScaleBank? = nil



    var laneMajorScales: LagunaLaneMajorScaleBank? = nil

    var arrays: [MLXArray] {
        [packedCodes, scales]
            + (biases.map { [$0] } ?? [])
            + (indexedMetadata?.arrays ?? [])
            + (narrowScales?.arrays ?? [])
            + (laneMajorScales?.arrays ?? [])
    }
}







private let lagunaNativeAffineNVFP4From: Int? = {
    guard ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_NVFP4"] != "0"
    else { return nil }
    let raw = ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_NVFP4_FROM"] ?? "0"
    guard let value = Int(raw), value < LagunaConstants.numHiddenLayers else { return nil }
    return min(max(value, 0), LagunaConstants.numHiddenLayers)
}()


private let lagunaNativeAffineProbeFormat: (mode: QuantizationMode, groupSize: Int, bits: Int)? = {
    switch ProcessInfo.processInfo.environment["DARKBLOOM_NATIVE_AFFINE_PROBE_FORMAT"] {
    case "nvfp4": return (.nvfp4, 16, 4)
    case "mxfp8": return (.mxfp8, 32, 8)
    case "mxfp4": return (.mxfp4, 32, 4)
    default: return nil
    }
}()

private let lagunaNativeAffineProbeFormatFrom: Int = {
    guard
        let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_NATIVE_AFFINE_PROBE_FORMAT_FROM"],
        let value = Int(raw)
    else {
        return 0
    }
    return min(max(value, 0), LagunaConstants.numHiddenLayers)
}()

private func lagunaNativeAffineProbeRoundTrip(_ weight: MLXArray, layer: Int?) -> MLXArray {
    guard let probe = lagunaNativeAffineProbeFormat,
        weight.dim(1).isMultiple(of: probe.groupSize),
        layer.map({ $0 >= lagunaNativeAffineProbeFormatFrom }) ?? true
    else {
        return weight
    }
    let (codes, scales, biases) = quantized(
        weight, groupSize: probe.groupSize, bits: probe.bits, mode: probe.mode)
    return dequantized(
        codes, scales: scales, biases: biases,
        groupSize: probe.groupSize, bits: probe.bits, mode: probe.mode,
        dtype: .bfloat16)
}

func lagunaNativeAffineWeight(
    _ weight: MLXArray, layer: Int? = nil
) -> LagunaNativeAffineWeight? {
    guard weight.dtype == .bfloat16, weight.ndim == 2,
        weight.dim(1).isMultiple(of: 32)
    else {
        return nil
    }
    let source = lagunaNativeAffineProbeRoundTrip(weight, layer: layer)
    if let from = lagunaNativeAffineNVFP4From,
        (layer ?? 0) >= from,
        weight.dim(1).isMultiple(of: 16)
    {
        let (codes, scales, biases) = quantized(
            source, groupSize: 16, bits: 4, mode: .nvfp4)
        return LagunaNativeAffineWeight(
            packedCodes: codes,
            scales: scales,
            biases: biases,
            originalShape: weight.shape,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4
        )
    }
    let (packedCodes, scales, biases) = quantized(
        source, groupSize: 32, bits: 8, mode: .affine)
    guard biases != nil else { return nil }
    return LagunaNativeAffineWeight(
        packedCodes: packedCodes,
        scales: scales,
        biases: biases,
        originalShape: weight.shape
    )
}


private func lagunaFusedQKVProjectionSource(
    heads: Int, compact: Bool = false, mxfp8: Bool = false
) -> String {
    let projectionPointerSetup =
        compact
        ? """
const device bfloat* weight;
const device uint8_t* weight_low;
const device uint8_t* weight_codes;
const device uint8_t* weight_palettes;
const device uint8_t* weight_modes;
device bfloat* out;
uint row_base;
if (global_row < query_rows) {
    weight = query_weight;
    weight_low = query_low;
    weight_codes = query_codes;
    weight_palettes = query_palettes;
    weight_modes = query_modes;
    out = queries;
    row_base = global_row;
} else if (global_row < query_rows + kv_rows) {
    weight = key_weight;
    weight_low = key_low;
    weight_codes = key_codes;
    weight_palettes = key_palettes;
    weight_modes = key_modes;
    out = keys;
    row_base = global_row - query_rows;
} else {
    weight = value_weight;
    weight_low = value_low;
    weight_codes = value_codes;
    weight_palettes = value_palettes;
    weight_modes = value_modes;
    out = values;
    row_base = global_row - query_rows - kv_rows;
}
"""
        : mxfp8
        ? """
const device bfloat* weight;
const device uint8_t* weight_codes8;
const device uint8_t* weight_scales8;
device bfloat* out;
uint row_base;
if (global_row < query_rows) {
    weight = query_weight;
    weight_codes8 = query_codes8;
    weight_scales8 = query_scales8;
    out = queries;
    row_base = global_row;
} else if (global_row < query_rows + kv_rows) {
    weight = key_weight;
    weight_codes8 = key_codes8;
    weight_scales8 = key_scales8;
    out = keys;
    row_base = global_row - query_rows;
} else {
    weight = value_weight;
    weight_codes8 = value_codes8;
    weight_scales8 = value_scales8;
    out = values;
    row_base = global_row - query_rows - kv_rows;
}
"""
        : mxfp8
        ? """
thread float result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
constexpr uint scale_groups = in_vec_size / 32;

for (uint row = 0; row < rows_per_thread; ++row) {
    float row_acc = 0.0f;
    for (uint gg = 0; gg < 2; ++gg) {
        uint group = 2 * lane + gg;
        float scale = laguna_attn_e8m0_decode(
            weight_scales8[
                size_t(row_base + row) * scale_groups + group]);
        const device uint4* cptr = (const device uint4*)(
            weight_codes8
            + size_t(row_base + row) * in_vec_size
            + group * 32);
        uint4 packed0 = cptr[0];
        uint4 packed1 = cptr[1];
        const threadgroup ushort4* xrow =
            (const threadgroup ushort4*)(
                normalized_row + group * 32);
        #pragma clang loop unroll(full)
        for (uint w = 0; w < 8; ++w) {
            uint packed =
                (w < 4u) ? packed0[w & 3u] : packed1[w & 3u];
            float4 weights =
                laguna_attn_e4m3_decode4(packed) * scale;
            float4 values =
                as_type<float4>(uint4(xrow[w]) << 16);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                row_acc += weights[i] * values[i];
            }
        }
    }
    result[row] = row_acc;
}
"""
        : """
const device bfloat* weight;
device bfloat* out;
uint row_base;
if (global_row < query_rows) {
    weight = query_weight;
    out = queries;
    row_base = global_row;
} else if (global_row < query_rows + kv_rows) {
    weight = key_weight;
    out = keys;
    row_base = global_row - query_rows;
} else {
    weight = value_weight;
    out = values;
    row_base = global_row - query_rows - kv_rows;
}
"""

    let projectionLoop =
        compact
        ? """
thread float result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
thread float coefficients[values_per_thread];
constexpr uint palette_block_width = 1024;
constexpr uint palette_blocks = in_vec_size / palette_block_width;
constexpr uint subblocks_per_palette = palette_block_width / block_width;

for (uint palette_segment = 0;
    palette_segment < palette_blocks; ++palette_segment)
{
    thread uint palette_lane[rows_per_thread];
    thread uint raw_mode[rows_per_thread];
    for (uint row = 0; row < rows_per_thread; ++row) {
        size_t palette_block =
            size_t(row_base + row) * palette_blocks + palette_segment;
        palette_lane[row] =
            lane < 16
            ? uint(weight_palettes[palette_block * 16 + lane])
            : 0u;
        uint mode = lane == 0 ? uint(weight_modes[palette_block]) : 0u;
        raw_mode[row] = simd_shuffle(mode, ushort(0));
    }

    for (uint subblock = 0;
        subblock < subblocks_per_palette; ++subblock)
    {
        uint column =
            palette_segment * palette_block_width
            + subblock * block_width
            + lane * values_per_thread;
        for (uint i = 0; i < values_per_thread; ++i) {
            coefficients[i] = float(normalized_row[column + i]);
        }

        for (uint row = 0; row < rows_per_thread; ++row) {
            size_t value_index =
                size_t(row_base + row) * in_vec_size + column;
            if (raw_mode[row] != 0) {
                const device vec<bfloat, 4>* row_values =
                    (const device vec<bfloat, 4>*)(
                        weight + value_index);
                const vec<bfloat, 4> w = row_values[0];
                for (uint i = 0; i < values_per_thread; ++i) {
                    result[row] += float(w[i]) * coefficients[i];
                }
            } else {
                uint8_t packed0 = weight_codes[value_index / 2];
                uint8_t packed1 = weight_codes[value_index / 2 + 1];
                thread float unpacked[values_per_thread];
                uint high0 = simd_shuffle(
                    palette_lane[row], ushort(packed0 & 0x0fu));
                uint high1 = simd_shuffle(
                    palette_lane[row], ushort(packed0 >> 4));
                uint high2 = simd_shuffle(
                    palette_lane[row], ushort(packed1 & 0x0fu));
                uint high3 = simd_shuffle(
                    palette_lane[row], ushort(packed1 >> 4));
                unpacked[0] = as_type<float>(
                    (high0 << 24) | (uint(weight_low[value_index]) << 16));
                unpacked[1] = as_type<float>(
                    (high1 << 24)
                    | (uint(weight_low[value_index + 1]) << 16));
                unpacked[2] = as_type<float>(
                    (high2 << 24)
                    | (uint(weight_low[value_index + 2]) << 16));
                unpacked[3] = as_type<float>(
                    (high3 << 24)
                    | (uint(weight_low[value_index + 3]) << 16));
                for (uint i = 0; i < values_per_thread; ++i) {
                    result[row] += unpacked[i] * coefficients[i];
                }
            }
        }
    }
}
"""
        : """
thread float result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
thread float coefficients[values_per_thread];

uint column = lane * values_per_thread;
for (uint block = 0; block < blocks; ++block) {
    for (uint i = 0; i < values_per_thread; ++i) {
        coefficients[i] = float(normalized_row[column + i]);
    }

    for (uint row = 0; row < rows_per_thread; ++row) {
        const device vec<bfloat, 4>* row_values =
            (const device vec<bfloat, 4>*)(
                weight + (row_base + row) * in_vec_size + column);
        const vec<bfloat, 4> w = row_values[0];
        for (uint i = 0; i < values_per_thread; ++i) {
            result[row] += float(w[i]) * coefficients[i];
        }
    }

    column += block_width;
}
"""

    return """
constexpr uint in_vec_size = \(LagunaConstants.hiddenSize);
constexpr uint query_rows = \(heads * LagunaConstants.headDim);
constexpr uint kv_rows =
    \(LagunaConstants.numKeyValueHeads * LagunaConstants.headDim);
constexpr uint rows_per_thread = 4;
constexpr uint values_per_thread = 4;
constexpr uint block_width = 128;
constexpr uint blocks = in_vec_size / block_width;
constexpr uint rows_per_group = 64;
constexpr uint query_tiles = query_rows / rows_per_group;
constexpr uint kv_tiles = kv_rows / rows_per_group;
constexpr uint gate_tiles = \(heads / 8);
constexpr uint query_tiles_per_round = query_tiles / kv_tiles;
constexpr float norm_eps = 1.0e-6f;

uint scheduled_tile = threadgroup_position_in_grid.x;
uint round;
uint position;
constexpr uint gated_round_width = query_tiles_per_round + 3;
constexpr uint plain_round_width = query_tiles_per_round + 2;
constexpr uint gated_span = gate_tiles * gated_round_width;
bool round_has_gate = scheduled_tile < gated_span;
if (round_has_gate) {
    round = scheduled_tile / gated_round_width;
    position = scheduled_tile % gated_round_width;
} else {
    uint tail = scheduled_tile - gated_span;
    round = gate_tiles + tail / plain_round_width;
    position = tail % plain_round_width;
}

uint tile;
if (position == 0) {
    tile = query_tiles + round;
} else if (position == 1) {
    tile = query_tiles + kv_tiles + round;
} else if (round_has_gate && position == 2) {
    tile = query_tiles + 2 * kv_tiles + round;
} else {
    uint projection_prefix = round_has_gate ? 3u : 2u;
    uint query_position = position - projection_prefix;
    tile = round * query_tiles_per_round + query_position;
}

uint local_id = thread_position_in_threadgroup.x;
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;

\(lagunaNormInvMeanScratch)
threadgroup float local_sums[32];
threadgroup bfloat normalized_row[in_vec_size];

uint norm_base = local_id * values_per_thread;
thread float raw[values_per_thread];
float acc = 0.0f;
for (uint i = 0; i < values_per_thread; ++i) {
    raw[i] = float(residual[norm_base + i]);
    acc += raw[i] * raw[i];
}
acc = simd_sum(acc);
\(lagunaNormReductionTailQKV)

for (uint i = 0; i < values_per_thread; ++i) {
    bfloat value =
        norm_weight[norm_base + i] *
        bfloat(raw[i] * laguna_inv_mean);
    normalized_row[norm_base + i] = value;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

constexpr uint gate_rows = 64;
constexpr uint gate_simds = 8;
constexpr uint gate_block_width = 1024;
constexpr uint gate_blocks = in_vec_size / gate_block_width;
constexpr uint qkv_tiles = query_tiles + 2 * kv_tiles;

threadgroup float gate_partials[2 * gate_simds * rows_per_thread];

if (tile >= qkv_tiles) {
    uint gate_half = simd_group / gate_simds;
    uint split = simd_group % gate_simds;
    uint gate_row =
        ((tile - qkv_tiles) * 2 + gate_half) * rows_per_thread;

    thread float gate_result[rows_per_thread] = {
        0.0f, 0.0f, 0.0f, 0.0f
    };
    thread float gate_input[values_per_thread];

    uint gate_column =
        (split * 32 + lane) * values_per_thread;
    for (uint block = 0; block < gate_blocks; ++block) {
        for (uint i = 0; i < values_per_thread; ++i) {
            gate_input[i] = float(normalized_row[gate_column + i]);
        }
        for (uint r = 0; r < rows_per_thread; ++r) {
            const device vec<bfloat, 4>* row_values =
                (const device vec<bfloat, 4>*)(
                    gate_weight + (gate_row + r) * in_vec_size +
                        gate_column);
            const vec<bfloat, 4> gw = row_values[0];
            for (uint i = 0; i < values_per_thread; ++i) {
                gate_result[r] += float(gw[i]) * gate_input[i];
            }
        }
        gate_column += gate_block_width;
    }

    for (uint r = 0; r < rows_per_thread; ++r) {
        for (ushort delta = 16; delta >= 1; delta >>= 1) {
            gate_result[r] +=
                metal::simd_shuffle_down(gate_result[r], delta);
        }
    }
    if (lane == 0) {
        for (uint r = 0; r < rows_per_thread; ++r) {
            gate_partials[
                (gate_half * gate_simds + split) * rows_per_thread + r
            ] = gate_result[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0 && lane == 0) {
        for (uint r = 0; r < rows_per_thread; ++r) {
            float total = gate_result[r];
            for (uint sgn = 1; sgn < gate_simds; ++sgn) {
                total += gate_partials[
                    (gate_half * gate_simds + sgn) * rows_per_thread + r
                ];
            }
            bfloat rounded_logit = bfloat(total);
            float logit = float(rounded_logit);
            float gate;
            if (metal::isnan(logit)) {
                gate = NAN;
            } else {
                float maxval = metal::max(logit, 0.0f);
                float minval = metal::min(logit, 0.0f);
                gate = (metal::isinf(minval) || metal::isinf(maxval))
                    ? maxval
                    : maxval + log1p(metal::exp(minval - maxval));
            }
            gate_values[gate_row + r] = bfloat(gate);
        }
    }
    return;
}

uint global_row = tile * rows_per_group + simd_group * rows_per_thread;

\(projectionPointerSetup)

\(projectionLoop)

for (uint row = 0; row < rows_per_thread; ++row) {
    for (ushort delta = 16; delta >= 1; delta >>= 1) {
        result[row] += metal::simd_shuffle_down(result[row], delta);
    }
}
if (lane == 0) {
    for (uint row = 0; row < rows_per_thread; ++row) {
        out[row_base + row] = bfloat(result[row]);
    }
}
"""
}

private let lagunaFusedQKVProjectionKernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        kernels[heads] = MLXFast.metalKernel(
            name: "laguna_fused_norm_qkv_projection_bf16_h\(heads)_v3",
            inputNames: [
                "residual", "norm_weight", "query_weight", "key_weight",
                "value_weight", "gate_weight",
            ],
            outputNames: ["queries", "keys", "values", "gate_values"],
            source: lagunaFusedQKVProjectionSource(heads: heads),
            ensureRowContiguous: true
        )
    }
    return kernels
}()

func lagunaFusedNormQKVProjection(
    residual: MLXArray,
    normWeight: MLXArray,
    queryWeight: MLXArray,
    keyWeight: MLXArray,
    valueWeight: MLXArray,
    gateWeight: MLXArray,
    heads: Int
) -> (
    queries: MLXArray, keys: MLXArray, values: MLXArray, gateValues: MLXArray,
    gateActivated: Bool
)? {
    guard let kernel = lagunaFusedQKVProjectionKernels[heads] else { return nil }
    let hidden = LagunaConstants.hiddenSize
    let queryRows = heads * LagunaConstants.headDim
    let kvRows = LagunaConstants.numKeyValueHeads * LagunaConstants.headDim
    precondition(residual.dtype == .bfloat16)
    precondition(residual.dims(1, 1, hidden))
    precondition(normWeight.dtype == .bfloat16)
    precondition(normWeight.dims(hidden))
    precondition(queryWeight.dims(queryRows, hidden))
    precondition(keyWeight.dims(kvRows, hidden))
    precondition(valueWeight.dims(kvRows, hidden))
    precondition(gateWeight.dtype == .bfloat16)
    precondition(gateWeight.dims(heads, hidden))



    let projectionTiles = (queryRows + 2 * kvRows) / 64
    let gateTiles = heads / 8
    lagunaTrace("norm+qkv+gate projection h\(heads)")
    let outputs = kernel(
        [residual, normWeight, queryWeight, keyWeight, valueWeight, gateWeight],
        grid: ((projectionTiles + gateTiles) * 512, 1, 1),
        threadGroup: (512, 1, 1),
        outputShapes: [
            [1, 1, queryRows], [1, 1, kvRows], [1, 1, kvRows], [1, 1, heads],
        ],
        outputDTypes: [.bfloat16, .bfloat16, .bfloat16, .bfloat16]
    )
    return (outputs[0], outputs[1], outputs[2], outputs[3], true)
}



































private func lagunaGatedOutputProjectionSource(
    heads: Int, unroll: Int, compact: Bool = false
) -> String {
    let singleWeightLoad =
        compact
        ? """
                    size_t value_index =
                        size_t(out_row + row) * in_vec_size + column;
                    size_t palette_block = value_index / 1024;
                    vec<bfloat, 4> w;
                    if (weight_modes[palette_block] != 0) {
                        const device vec<bfloat, 4>* row_values =
                            (const device vec<bfloat, 4>*)(
                                weight + value_index);
                        w = row_values[0];
                    } else {
                        uint8_t packed0 =
                            weight_codes[value_index / 2];
                        uint8_t packed1 =
                            weight_codes[value_index / 2 + 1];
                        size_t palette_base = palette_block * 16;
                        ushort bits0 = ushort(weight_low[value_index])
                            | (ushort(weight_palettes[
                                palette_base + (packed0 & 0x0fu)]) << 8);
                        ushort bits1 = ushort(weight_low[value_index + 1])
                            | (ushort(weight_palettes[
                                palette_base + (packed0 >> 4)]) << 8);
                        ushort bits2 = ushort(weight_low[value_index + 2])
                            | (ushort(weight_palettes[
                                palette_base + (packed1 & 0x0fu)]) << 8);
                        ushort bits3 = ushort(weight_low[value_index + 3])
                            | (ushort(weight_palettes[
                                palette_base + (packed1 >> 4)]) << 8);
                        w[0] = as_type<bfloat>(bits0);
                        w[1] = as_type<bfloat>(bits1);
                        w[2] = as_type<bfloat>(bits2);
                        w[3] = as_type<bfloat>(bits3);
                    }
"""
        : """
                    const device vec<bfloat, 4>* row_values =
                        (const device vec<bfloat, 4>*)(
                            weight + (out_row + row) * in_vec_size + column);
                    const vec<bfloat, 4> w = row_values[0];
"""

    let unrolledWeightLoad =
        compact
        ? """
                        size_t value_index =
                            size_t(out_row + row) * in_vec_size + column_u;
                        size_t palette_block = value_index / 1024;
                        if (weight_modes[palette_block] != 0) {
                            const device vec<bfloat, 4>* row_values =
                                (const device vec<bfloat, 4>*)(
                                    weight + value_index);
                            weight_values[u][row] = row_values[0];
                        } else {
                            uint8_t packed0 =
                                weight_codes[value_index / 2];
                            uint8_t packed1 =
                                weight_codes[value_index / 2 + 1];
                            size_t palette_base = palette_block * 16;
                            ushort bits0 = ushort(weight_low[value_index])
                                | (ushort(weight_palettes[
                                    palette_base + (packed0 & 0x0fu)]) << 8);
                            ushort bits1 = ushort(weight_low[value_index + 1])
                                | (ushort(weight_palettes[
                                    palette_base + (packed0 >> 4)]) << 8);
                            ushort bits2 = ushort(weight_low[value_index + 2])
                                | (ushort(weight_palettes[
                                    palette_base + (packed1 & 0x0fu)]) << 8);
                            ushort bits3 = ushort(weight_low[value_index + 3])
                                | (ushort(weight_palettes[
                                    palette_base + (packed1 >> 4)]) << 8);
                            weight_values[u][row][0] =
                                as_type<bfloat>(bits0);
                            weight_values[u][row][1] =
                                as_type<bfloat>(bits1);
                            weight_values[u][row][2] =
                                as_type<bfloat>(bits2);
                            weight_values[u][row][3] =
                                as_type<bfloat>(bits3);
                        }
"""
        : """
                        const device vec<bfloat, 4>* row_values =
                            (const device vec<bfloat, 4>*)(
                                weight + (out_row + row) * in_vec_size +
                                    column_u);
                        weight_values[u][row] = row_values[0];
"""

    let body: String
    if unroll == 1 {
        body = """
        uint column = lane * values_per_thread;
        for (uint block = 0; block < blocks; ++block) {
            float gate = float(gate_values[block]);
            const device vec<bfloat, 4>* gated =
                (const device vec<bfloat, 4>*)(attention_output + column);
            const vec<bfloat, 4> values = gated[0];
            for (uint i = 0; i < values_per_thread; ++i) {
                coefficients[i] = float(bfloat(float(values[i]) * gate));
            }

            for (uint row = 0; row < rows_per_thread; ++row) {
                \(singleWeightLoad)
                for (uint i = 0; i < values_per_thread; ++i) {
                    result[row] += float(w[i]) * coefficients[i];
                }
            }

            column += block_width;
        }
"""
    } else {
        body = """
        uint column = lane * values_per_thread;
        for (uint block = 0; block < blocks; block += unroll) {
            vec<bfloat, 4> gated_values[unroll];
            vec<bfloat, 4> weight_values[unroll][rows_per_thread];
            for (uint u = 0; u < unroll; ++u) {
                uint column_u = column + u * block_width;
                const device vec<bfloat, 4>* gated =
                    (const device vec<bfloat, 4>*)(
                        attention_output + column_u);
                gated_values[u] = gated[0];
                for (uint row = 0; row < rows_per_thread; ++row) {
                    \(unrolledWeightLoad)
                }
            }

            for (uint u = 0; u < unroll; ++u) {
                float gate = float(gate_values[block + u]);
                for (uint i = 0; i < values_per_thread; ++i) {
                    coefficients[i] =
                        float(bfloat(float(gated_values[u][i]) * gate));
                }
                for (uint row = 0; row < rows_per_thread; ++row) {
                    for (uint i = 0; i < values_per_thread; ++i) {
                        result[row] +=
                            float(weight_values[u][row][i]) *
                                coefficients[i];
                    }
                }
            }

            column += unroll * block_width;
        }
"""
    }
    return """
constexpr uint unroll = \(unroll);
constexpr uint in_vec_size = \(heads * LagunaConstants.headDim);
constexpr uint heads = \(heads);
constexpr uint head_dim = 128;
constexpr uint rows_per_thread = 4;
constexpr uint values_per_thread = 4;
constexpr uint block_width = 128;
constexpr uint blocks = in_vec_size / block_width;
constexpr uint rows_per_group = 16;

uint tile = threadgroup_position_in_grid.x;
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;

uint out_row = tile * rows_per_group + simd_group * rows_per_thread;
thread float result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
thread float coefficients[values_per_thread];

\(body)

for (uint row = 0; row < rows_per_thread; ++row) {
    for (ushort delta = 16; delta >= 1; delta >>= 1) {
        result[row] += metal::simd_shuffle_down(result[row], delta);
    }
}
if (lane == 0) {
    for (uint row = 0; row < rows_per_thread; ++row) {
        projected[out_row + row] = bfloat(result[row]);
    }
}
"""
}












let lagunaGatedOutputUnroll: Int = {
    guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_L5_UNROLL"],
        let value = Int(raw), [1, 2, 4, 8].contains(value)
    else {
        return 2
    }
    return value
}()




private let lagunaGatedOutputProjectionKernels:
    [Int: [Int: MLXFast.MLXFastKernel]] = {
    var kernels: [Int: [Int: MLXFast.MLXFastKernel]] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        var byDepth: [Int: MLXFast.MLXFastKernel] = [:]
        for depth in [1, 2, 4, 8] {
            byDepth[depth] = MLXFast.metalKernel(
                name: "laguna_gated_output_projection_bf16_h\(heads)_u\(depth)_v3",
                inputNames: ["attention_output", "gate_values", "weight"],
                outputNames: ["projected"],
                source: lagunaGatedOutputProjectionSource(
                    heads: heads, unroll: depth),
                ensureRowContiguous: true
            )
        }
        kernels[heads] = byDepth
    }
    return kernels
}()

func lagunaGatedOutputProjection(
    attentionOutput: MLXArray, gateValues: MLXArray, weight: MLXArray, heads: Int
) -> MLXArray? {
    guard let kernel = lagunaGatedOutputProjectionKernels[heads]?[lagunaGatedOutputUnroll]
    else { return nil }
    let inVec = heads * LagunaConstants.headDim
    precondition(attentionOutput.dtype == .bfloat16)
    precondition(attentionOutput.dims(1, 1, inVec))
    precondition(gateValues.dtype == .bfloat16)
    precondition(gateValues.dims(1, 1, heads))
    precondition(weight.dtype == .bfloat16)
    precondition(weight.dims(LagunaConstants.hiddenSize, inVec))

    lagunaTrace("gated output projection h\(heads)")
    return kernel(
        [attentionOutput, gateValues, weight],
        grid: ((LagunaConstants.hiddenSize / 16) * 128, 1, 1),
        threadGroup: (128, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.hiddenSize]],
        outputDTypes: [.bfloat16]
    )[0]
}




















private func lagunaGateProductSoftplusSource(heads: Int) -> String {
    """
constexpr int HEAD_DIM = \(LagunaConstants.headDim);
uint gid = thread_position_in_grid.x;
int head = gid / HEAD_DIM;
float logit = float(gate_logits[head]);
float gate;
if (metal::isnan(logit)) {
    gate = NAN;
} else {
    float maxval = metal::max(logit, 0.0f);
    float minval = metal::min(logit, 0.0f);
    gate = (metal::isinf(minval) || metal::isinf(maxval))
        ? maxval
        : maxval + log1p(metal::exp(minval - maxval));
}
bfloat gate_bf = bfloat(gate);
gated[gid] = bfloat(float(attention_output[gid]) * float(gate_bf));
"""
}

private let lagunaGateProductSoftplusKernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        kernels[heads] = MLXFast.metalKernel(
            name: "laguna_gate_product_softplus_bf16_h\(heads)_v1",
            inputNames: ["attention_output", "gate_logits"],
            outputNames: ["gated"],
            source: lagunaGateProductSoftplusSource(heads: heads),
            ensureRowContiguous: true
        )
    }
    return kernels
}()



private let lagunaFusedGateProductEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_GATE_PRODUCT"] != "0"





func lagunaGateProductSoftplus(
    attentionOutput: MLXArray, gateLogits: MLXArray, heads: Int
) -> MLXArray? {
    guard lagunaFusedGateProductEnabled,
        let kernel = lagunaGateProductSoftplusKernels[heads]
    else { return nil }
    let inVec = heads * LagunaConstants.headDim
    precondition(attentionOutput.dtype == .bfloat16)
    precondition(attentionOutput.dims(1, 1, inVec))
    precondition(gateLogits.dtype == .bfloat16)
    precondition(gateLogits.dims(1, 1, heads))

    lagunaTrace("gate product softplus h\(heads)")
    return kernel(
        [attentionOutput, gateLogits],
        grid: (inVec, 1, 1),
        threadGroup: (128, 1, 1),
        outputShapes: [[1, 1, inVec]],
        outputDTypes: [.bfloat16]
    )[0]
}




private func lagunaGatedAffineOProjSource(heads: Int, indexed: Bool = false) -> String {
    let metadataPointers = indexed
        ? """
const device ushort* mi = metadata_indices + out_row * in_vec_size_g +
    simd_lid / scale_step_per_thread;
"""
        : """
const device bfloat* sc = weight_scales + out_row * in_vec_size_g +
    simd_lid / scale_step_per_thread;
const device bfloat* bs = weight_biases + out_row * in_vec_size_g +
    simd_lid / scale_step_per_thread;
"""
    let metadataLoad = indexed
        ? """
    uint pair = metadata_lut[mi[row * in_vec_size_g]];
    float scale = float(as_type<bfloat>(ushort(pair)));
    float bias = float(as_type<bfloat>(ushort(pair >> 16)));
"""
        : """
    float scale = float(sc[row * in_vec_size_g]);
    float bias = float(bs[row * in_vec_size_g]);
"""
    let metadataAdvance = indexed
        ? "mi += block_size / group_size;"
        : """
sc += block_size / group_size;
bs += block_size / group_size;
"""
    return """
constexpr uint in_vec_size = \(heads * LagunaConstants.headDim);
constexpr uint out_vec_size = \(LagunaConstants.hiddenSize);
constexpr uint gate_heads = \(heads);
constexpr uint head_shift = 7;
constexpr uint values_per_thread = 8;
constexpr uint block_size = 256;
constexpr uint results_per_simdgroup = 4;
constexpr uint num_simdgroups = 2;
constexpr uint group_size = 32;
constexpr uint scale_step_per_thread = group_size / values_per_thread;
constexpr uint in_vec_size_g = in_vec_size / group_size;

uint tile = threadgroup_position_in_grid.x;
uint lid = thread_position_in_threadgroup.x;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;

threadgroup float gate_table[gate_heads];
if (lid < gate_heads) {
    float logit = float(gate_logits[lid]);
    float gate;
    if (metal::isnan(logit)) {
        gate = NAN;
    } else {
        float maxval = metal::max(logit, 0.0f);
        float minval = metal::min(logit, 0.0f);
        gate = (metal::isinf(minval) || metal::isinf(maxval))
            ? maxval
            : maxval + log1p(metal::exp(minval - maxval));
    }
    gate_table[lid] = float(bfloat(gate));
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint out_row = tile * (num_simdgroups * results_per_simdgroup) +
    simd_gid * results_per_simdgroup;

const device uint8_t* ws = (const device uint8_t*)weight_codes +
    out_row * in_vec_size + simd_lid * values_per_thread;
\(metadataPointers)
const device bfloat* xp = attention_output + simd_lid * values_per_thread;

thread float x_thread[values_per_thread];
thread float result[results_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};

uint column = simd_lid * values_per_thread;
for (uint k = 0; k < in_vec_size; k += block_size) {
    float gate = gate_table[column >> head_shift];
    float sum = 0.0f;
    for (uint i = 0; i < values_per_thread; ++i) {
        float value = float(bfloat(float(xp[i]) * gate));
        sum += value;
        x_thread[i] = value;
    }

    for (uint row = 0; row < results_per_simdgroup; ++row) {
        const device uint8_t* wl = ws + row * in_vec_size;
        \(metadataLoad)
        float accum = 0.0f;
        for (uint i = 0; i < values_per_thread; ++i) {
            accum += x_thread[i] * wl[i];
        }
        result[row] += scale * accum + sum * bias;
    }

    ws += block_size;
    \(metadataAdvance)
    xp += block_size;
    column += block_size;
}

for (uint row = 0; row < results_per_simdgroup; ++row) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
        projected[out_row + row] = bfloat(result[row]);
    }
}
"""
}




private let lagunaGatedAffineOProjKernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        kernels[heads] = MLXFast.metalKernel(
            name: "laguna_gated_affine_oproj_qmv_i8g32_h\(heads)_v1",
            inputNames: [
                "attention_output", "gate_logits", "weight_codes",
                "weight_scales", "weight_biases",
            ],
            outputNames: ["projected"],
            source: lagunaGatedAffineOProjSource(heads: heads),
            ensureRowContiguous: true
        )
    }
    return kernels
}()

private let lagunaGatedAffineOProjIndexedKernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        kernels[heads] = MLXFast.metalKernel(
            name: "laguna_gated_affine_oproj_qmv_i8g32_h\(heads)_idx_v1",
            inputNames: [
                "attention_output", "gate_logits", "weight_codes",
                "metadata_indices", "metadata_lut",
            ],
            outputNames: ["projected"],
            source: lagunaGatedAffineOProjSource(heads: heads, indexed: true),
            ensureRowContiguous: true
        )
    }
    return kernels
}()









let lagunaFusedGatedAffineOProjEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_GATED_AFFINE_OPROJ"] != "0"


let lagunaGatedAffineOProjNVFP4Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_GATED_AFFINE_OPROJ_NVFP4"] != "0"














let lagunaNvfp4QmvSignCarryEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_NVFP4_QMV_SIGN_CARRY"] != "0"














let lagunaE4M3SignDomainCertified =
    ProcessInfo.processInfo.environment["DARKBLOOM_E4M3_SIGN_DOMAIN"] != "0"










let lagunaNvfp4QmvSeedElisionEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_NVFP4_QMV_SEED_ELIDE"] != "0"




func lagunaGatedAffineOProj(
    attentionOutput: MLXArray,
    gateLogits: MLXArray,
    codes: MLXArray,
    scales: MLXArray,
    biases: MLXArray,
    indexedMetadata: LagunaIndexedAffineMetadata? = nil,
    heads: Int
) -> MLXArray? {
    let inVec = heads * LagunaConstants.headDim
    let outVec = LagunaConstants.hiddenSize
    guard attentionOutput.dtype == .bfloat16,
        attentionOutput.dims(1, 1, inVec),
        gateLogits.dtype == .bfloat16,
        gateLogits.dims(1, 1, heads),
        codes.dtype == .uint32,
        codes.dims(outVec, inVec / 4),
        scales.dtype == .bfloat16,
        scales.dims(outVec, inVec / 32),
        biases.dtype == .bfloat16,
        biases.dims(outVec, inVec / 32)
    else {
        return nil
    }

    if let metadata = indexedMetadata,
        let kernel = lagunaGatedAffineOProjIndexedKernels[heads],
        metadata.indices.dtype == .uint16,
        metadata.indices.dims(outVec, inVec / 32),
        metadata.lut.dtype == .uint32,
        metadata.lut.ndim == 1,
        metadata.lut.size <= 65_536
    {
        lagunaTrace("gated affine oproj qmv h\(heads) indexed")
        return kernel(
            [attentionOutput, gateLogits, codes, metadata.indices, metadata.lut],
            grid: ((outVec / 8) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[1, 1, outVec]], outputDTypes: [.bfloat16]
        )[0]
    }
    guard let kernel = lagunaGatedAffineOProjKernels[heads] else { return nil }
    lagunaTrace("gated affine oproj qmv h\(heads)")
    return kernel(
        [attentionOutput, gateLogits, codes, scales, biases],
        grid: ((outVec / 8) * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, outVec]],
        outputDTypes: [.bfloat16]
    )[0]
}







func lagunaGatedAffineOProjNVFP4Source(
    heads: Int,
    signCarry: Bool = lagunaNvfp4QmvSignCarryEnabled,
    seedElide: Bool = lagunaNvfp4QmvSeedElisionEnabled,
    preActivatedGate: Bool = false,
    laneMajor: Bool = false,
    pairwise: Bool = false
) -> String {
    let scaleFold = lagunaNvfp4ScaleFoldEnabled
    let weightScale = scaleFold ? "" : " * 16384.0f"



    let scaleDecode = signCarry
        ? (lagunaE4M3SignDomainCertified
            ? "ushort sraw = ushort(sbits) << 7;\n"
                + "        float scale = float(as_type<half>(sraw));"
            : "ushort sraw = ushort(sbits + (sbits & 128)) << 7;\n"
                + "        float scale = float(as_type<half>(sraw));")
        : "ushort sraw = ushort(sbits & 127) << 7;\n"
            + "        half sconverted = as_type<half>(sraw);\n"
            + "        float scale = float((sbits & 128) ? -sconverted : sconverted);"



    let accumDecl = seedElide ? "float accum;" : "float accum = 0.0f;"
    let firstAccum = seedElide
        ? "if (j == 0) {\n"
            + "                accum =\n"
            + "                    (x_thread[8 * j] * v04.x +\n"
            + "                     x_thread[8 * j + 1] * v15.x +\n"
            + "                     x_thread[8 * j + 2] * v26.x +\n"
            + "                     x_thread[8 * j + 3] * v37.x);\n"
            + "            } else {\n"
            + "                accum +=\n"
            + "                    (x_thread[8 * j] * v04.x +\n"
            + "                     x_thread[8 * j + 1] * v15.x +\n"
            + "                     x_thread[8 * j + 2] * v26.x +\n"
            + "                     x_thread[8 * j + 3] * v37.x);\n"
            + "            }"
        : "accum +=\n"
            + "                (x_thread[8 * j] * v04.x +\n"
            + "                 x_thread[8 * j + 1] * v15.x +\n"
            + "                 x_thread[8 * j + 2] * v26.x +\n"
            + "                 x_thread[8 * j + 3] * v37.x);"
    let extract = """
                const uint xe = c & 0x0F0F0F0Fu;
                const uint ge = xe | (xe << 3);
                const uint yo = c & 0xF0F0F0F0u;
                const uint go = yo | (yo >> 3);
                const uint p0 = (ge << 9) & 0x8E008E00u;
                const uint p1 = (go << 8) & 0x8E008E00u;
                const uint p2 = (ge << 1) & 0x8E008E00u;
                const uint p3 = go & 0x8E008E00u;
"""
    let gateSetup = preActivatedGate ? "" : """
threadgroup float gt[gate_heads];
if(lid<gate_heads){
    float l=float(gate_logits[lid]);
    float g;
    if(metal::isnan(l)) g=NAN;
    else {
        float hi=metal::max(l,0.0f);
        float lo=metal::min(l,0.0f);
        g=(metal::isinf(lo)||metal::isinf(hi))?hi:hi+log1p(metal::exp(lo-hi));
    }
    gt[lid]=float(bfloat(g));
}
threadgroup_barrier(mem_flags::mem_threadgroup);
"""
    let loadInput = preActivatedGate
        ? """
float g=float(gate_values[column>>head_shift]);
for(uint i=0;i<values_per_thread;++i)
    x_thread[i]=float(bfloat(float(xp[i])*g));
"""
        : """
float g=gt[column>>head_shift];
for(uint i=0;i<values_per_thread;++i)
    x_thread[i]=float(bfloat(float(xp[i])*g));
"""







    let nibDiv = pairwise ? 4 : 2
    let laneIdx = pairwise ? "(simd_lid >> 1)" : "simd_lid"
    let scaleSetup =
        laneMajor
        ? """
const device uint8_t* nq = scale_nibbles +
    out_row * (in_vec_size_g / \(nibDiv)) +
    \(laneIdx) * (in_vec_size_g / 64);
const device uint8_t* bs = scale_bases + out_row;
const device uint8_t* sc = weight_scales +
    out_row * in_vec_size_g + simd_lid;
uint nsh = 0;
"""
        : """
const device uint8_t* sc = weight_scales +
    out_row * in_vec_size_g + simd_lid;
"""
    let scaleRead =
        laneMajor
        ? """
const uint8_t rb = bs[row];
    const bool esc = rb == 0xFFu;
    const device uint8_t* sp = esc
        ? (sc + row * in_vec_size_g)
        : (nq + row * (in_vec_size_g / \(nibDiv)));
    const uint8_t raw = sp[0];
    uint8_t sbits = esc ? raw : uint8_t(rb + ((raw >> nsh) & 0x0Fu));
"""
        : "uint8_t sbits = sc[row * in_vec_size_g];"
    let scaleAdvance =
        laneMajor
        ? """
sc += block_size / group_size;
nq += nsh >> 2;
nsh ^= 4;
"""
        : "sc += block_size / group_size;"
    return """
constexpr uint in_vec_size = \(heads * LagunaConstants.headDim);
constexpr uint out_vec_size = \(LagunaConstants.hiddenSize);
constexpr uint gate_heads = \(heads);
constexpr uint head_shift = 7;
constexpr uint group_size = 16;
constexpr uint values_per_thread = 16;
constexpr uint codes_per_thread = values_per_thread / 8;
constexpr uint block_size = values_per_thread * 32;
constexpr uint results_per_simdgroup = 4;
constexpr uint num_simdgroups = 2;
constexpr uint in_vec_size_g = in_vec_size / group_size;

uint tile = threadgroup_position_in_grid.x;
uint lid = thread_position_in_threadgroup.x;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;

\(gateSetup)

uint out_row = tile * (num_simdgroups * results_per_simdgroup) +
    simd_gid * results_per_simdgroup;
const device uint32_t* ws =
    (const device uint32_t*)weight_codes +
    out_row * (in_vec_size / 8) + simd_lid * codes_per_thread;
\(scaleSetup)
const device bfloat* xp = attention_output + simd_lid * values_per_thread;

thread float x_thread[values_per_thread];
thread float result[results_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};

uint column = simd_lid * values_per_thread;
for (uint k = 0; k < in_vec_size; k += block_size) {
    \(loadInput)

    for (uint row = 0; row < results_per_simdgroup; ++row) {
        const device uint32_t* wl = ws + row * (in_vec_size / 8);
        \(scaleRead)
        \(scaleDecode)
        \(accumDecl)
        #pragma unroll
        for (uint j = 0; j < codes_per_thread; ++j) {
            const uint c = wl[j];
            \(extract)
            const float2 v04 = float2(as_type<half2>(p0))\(weightScale);
            const float2 v15 = float2(as_type<half2>(p1))\(weightScale);
            const float2 v26 = float2(as_type<half2>(p2))\(weightScale);
            const float2 v37 = float2(as_type<half2>(p3))\(weightScale);
            \(firstAccum)
            accum +=
                (x_thread[8 * j + 4] * v04.y +
                 x_thread[8 * j + 5] * v15.y +
                 x_thread[8 * j + 6] * v26.y +
                 x_thread[8 * j + 7] * v37.y);
        }
        result[row] += scale * accum;
    }

    ws += block_size / 8;
    \(scaleAdvance)
    xp += block_size;
    column += block_size;
}

for (uint row = 0; row < results_per_simdgroup; ++row) {
    result[row] = simd_sum(result[row] * 4194304.0f);
    if (simd_lid == 0) {
        projected[out_row + row] = bfloat(result[row]);
    }
}
"""
}

private let lagunaGatedAffineOProjNVFP4Kernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        kernels[heads] = MLXFast.metalKernel(
            name: "laguna_gated_affine_oproj_nvfp4_qmv_h\(heads)_v1"
                + (lagunaNvfp4QmvSignCarryEnabled ? "_sc1" : "")
                + (lagunaNvfp4QmvSeedElisionEnabled ? "_se1" : ""),
            inputNames: [
                "attention_output", "gate_logits", "weight_codes",
                "weight_scales",
            ],
            outputNames: ["projected"],
            source: lagunaGatedAffineOProjNVFP4Source(heads: heads),
            ensureRowContiguous: true
        )
    }
    return kernels
}()




private let lagunaGatedAffineOProjNVFP4LaneMajorKernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        kernels[heads] = MLXFast.metalKernel(
            name: "laguna_gated_affine_oproj_nvfp4_qmv_h\(heads)_v1_lm1"
                + (lagunaAttnScalePairwiseOProjEnabled ? "_pw1" : "")
                + (lagunaNvfp4QmvSignCarryEnabled ? "_sc1" : "")
                + (lagunaNvfp4QmvSeedElisionEnabled ? "_se1" : ""),
            inputNames: [
                "attention_output", "gate_logits", "weight_codes",
                "scale_nibbles", "scale_bases", "weight_scales",
            ],
            outputNames: ["projected"],
            source: lagunaGatedAffineOProjNVFP4Source(
                heads: heads, laneMajor: true,
                pairwise: lagunaAttnScalePairwiseOProjEnabled),
            ensureRowContiguous: true
        )
    }
    return kernels
}()

private let lagunaGateSoftplusEnabled = ProcessInfo.processInfo.environment[
    "DARKBLOOM_AFFINE_GATE_SOFTPLUS"] != "0"

private func lagunaGateSoftplusSource(
    heads: Int,
    inputName: String = "input",
    codesName: String = "packed_codes",
    scalesName: String = "scales",
    biasesName: String = "biases"
) -> String {
    """
constexpr uint K=\(LagunaConstants.hiddenSize),GS=32,V=8;
constexpr uint BK=V*32,R=4,NS=2,KG=K/GS,SS=GS/V;
uint tile=threadgroup_position_in_grid.x;
uint sg=simdgroup_index_in_threadgroup;
uint lane=thread_index_in_simdgroup;
uint orow=tile*(NS*R)+sg*R;
const device uint8_t* ws=(const device uint8_t*)\(codesName)+orow*K+lane*V;
const device bfloat* sc=\(scalesName)+orow*KG+lane/SS;
const device bfloat* bs=\(biasesName)+orow*KG+lane/SS;
thread float x[V];
thread float r[R]={0.0f,0.0f,0.0f,0.0f};
uint col=lane*V;
for(uint k=0;k<K;k+=BK){
    float sum=0.0f;
    for(uint i=0;i<V;++i){
        x[i]=float(\(inputName)[col+i]);
        sum+=x[i];
    }
    for(uint row=0;row<R;++row){
        const device uint8_t* wl=ws+row*K;
        float s=float(sc[row*KG]),b=float(bs[row*KG]),a=0.0f;
        for(uint i=0;i<V;++i) a+=x[i]*wl[i];
        r[row]+=s*a+sum*b;
    }
    ws+=BK; sc+=BK/GS; bs+=BK/GS; col+=BK;
}
for(uint row=0;row<R;++row){
    r[row]=simd_sum(r[row]);
    if(lane==0){
        float l=float(bfloat(r[row]));
        float g;
        if(metal::isnan(l)) g=NAN;
        else {
            float hi=metal::max(l,0.0f);
            float lo=metal::min(l,0.0f);
            g=(metal::isinf(lo)||metal::isinf(hi))?hi:hi+log1p(metal::exp(lo-hi));
        }
        gate_values[orow+row]=bfloat(g);
    }
}
"""
}

private let lagunaGateSoftplusKernels: [Int: MLXFast.MLXFastKernel] = {
    var result: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        result[heads] = MLXFast.metalKernel(
            name: "laguna_gate_sp_h\(heads)_v1",
            inputNames: ["input", "packed_codes", "scales", "biases"],
            outputNames: ["gate_values"],
            source: lagunaGateSoftplusSource(heads: heads),
            ensureRowContiguous: true)
    }
    return result
}()

private func lagunaGateSoftplus(
    input: MLXArray, bank: LagunaNativeAffineWeight, heads: Int
) -> MLXArray? {
    guard lagunaGateSoftplusEnabled,
        bank.mode == .affine, bank.bits == 8, bank.groupSize == 32,
        let biases = bank.biases,
        let kernel = lagunaGateSoftplusKernels[heads],
        input.dtype == .bfloat16,
        input.dims(1, 1, LagunaConstants.hiddenSize),
        bank.packedCodes.dims(heads, LagunaConstants.hiddenSize / 4),
        bank.scales.dims(heads, LagunaConstants.hiddenSize / 32),
        biases.dims(heads, LagunaConstants.hiddenSize / 32)
    else { return nil }

    return kernel(
        [input, bank.packedCodes, bank.scales, biases],
        grid: ((heads / 8) * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, heads]],
        outputDTypes: [.bfloat16])[0]
}

private let lagunaActivatedOProjKernels: [Int: MLXFast.MLXFastKernel] = {
    var result: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        result[heads] = MLXFast.metalKernel(
            name: "laguna_oproj_act_h\(heads)_v1"
                + (lagunaNvfp4QmvSignCarryEnabled ? "_sc1" : "")
                + (lagunaNvfp4QmvSeedElisionEnabled ? "_se1" : ""),
            inputNames: [
                "attention_output", "gate_values", "weight_codes",
                "weight_scales",
            ],
            outputNames: ["projected"],
            source: lagunaGatedAffineOProjNVFP4Source(heads: heads, preActivatedGate: true),
            ensureRowContiguous: true)
    }
    return result
}()

private let lagunaActivatedOProjLaneMajorKernels: [Int: MLXFast.MLXFastKernel] = {
    var result: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        result[heads] = MLXFast.metalKernel(
            name: "laguna_oproj_act_h\(heads)_v1_lm1"
                + (lagunaAttnScalePairwiseOProjEnabled ? "_pw1" : "")
                + (lagunaNvfp4QmvSignCarryEnabled ? "_sc1" : "")
                + (lagunaNvfp4QmvSeedElisionEnabled ? "_se1" : ""),
            inputNames: [
                "attention_output", "gate_values", "weight_codes",
                "scale_nibbles", "scale_bases", "weight_scales",
            ],
            outputNames: ["projected"],
            source: lagunaGatedAffineOProjNVFP4Source(
                heads: heads, preActivatedGate: true, laneMajor: true,
                pairwise: lagunaAttnScalePairwiseOProjEnabled),
            ensureRowContiguous: true)
    }
    return result
}()

func lagunaGatedAffineOProjNVFP4(
    attentionOutput: MLXArray,
    gateLogits: MLXArray,
    codes: MLXArray,
    scales: MLXArray,
    laneMajorScales: LagunaLaneMajorScaleBank? = nil,
    heads: Int,
    gateIsActivated: Bool = false
) -> MLXArray? {
    let inVec = heads * LagunaConstants.headDim
    let outVec = LagunaConstants.hiddenSize
    guard attentionOutput.dtype == .bfloat16,
        attentionOutput.dims(1, 1, inVec),
        gateLogits.dtype == .bfloat16,
        gateLogits.dims(1, 1, heads),
        codes.dtype == .uint32,
        codes.dims(outVec, inVec / 8),
        scales.dtype == .uint8,
        scales.dims(outVec, inVec / 16)
    else {
        return nil
    }

    if let lane = laneMajorScales,
        lane.pairwise == lagunaAttnScalePairwiseOProjEnabled,
        lane.nibbles.dtype == .uint8, lane.nibbles.dims(outVec, lane.nibbleBytes),
        lane.bases.dtype == .uint8, lane.bases.dims(outVec),
        lane.groups == inVec / 16,
        let kernel = gateIsActivated
            ? (lagunaGateSoftplusEnabled ? lagunaActivatedOProjLaneMajorKernels[heads] : nil)
            : lagunaGatedAffineOProjNVFP4LaneMajorKernels[heads]
    {
        lagunaTrace("gated affine oproj nvfp4 qmv h\(heads) lane-major")
        lagunaNarrowScaleLog.noteDispatch("lane-major", "oproj h\(heads)")
        return kernel(
            [
                attentionOutput, gateLogits, codes, lane.nibbles, lane.bases,
                scales,
            ],
            grid: ((outVec / 8) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[1, 1, outVec]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    let selected = gateIsActivated
        ? (lagunaGateSoftplusEnabled ? lagunaActivatedOProjKernels[heads] : nil)
        : lagunaGatedAffineOProjNVFP4Kernels[heads]
    guard let kernel = selected else { return nil }
    lagunaTrace("gated affine oproj nvfp4 qmv h\(heads)")
    lagunaNarrowScaleLog.noteDispatch("inactive", "oproj h\(heads)")
    return kernel(
        [attentionOutput, gateLogits, codes, scales],
        grid: ((outVec / 8) * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, outVec]],
        outputDTypes: [.bfloat16]
    )[0]
}













private let lagunaTailNVFP4ScaleFoldEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_TAIL_NVFP4_SCALE_FOLD"] != "0"















private let lagunaTailNVFP4QKVTailFoldEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_QKV_TAIL_FOLD"] != "0"



let lagunaTailNVFP4QKVSeedElisionEnabled = lagunaTailNVFP4QKVTailFoldEnabled





let lagunaTailNVFP4QKVScaleDeferEnabled =
    lagunaTailNVFP4QKVTailFoldEnabled && lagunaTailNVFP4ScaleFoldEnabled







func lagunaTailNVFP4ScaleDecodeSource(scaleFold: Bool, scaleDefer: Bool) -> String {
    guard scaleFold else {
        return "    half converted = as_type<half>(raw);\n"
            + "    converted *= 256.0;\n"
            + "    half signed_value = (bits & 128) ? -converted : converted;\n"
            + "    return float(signed_value);"
    }
    return scaleDefer
        ? "    return float(as_type<half>(raw));"
        : "    return float(as_type<half>(raw)) * 4194304.0f;"
}



func lagunaTailNVFP4QDotAccumDeclSource(seedElide: Bool) -> String {
    seedElide ? "float accum;" : "float accum = 0;"
}







func lagunaTailNVFP4QDotFirstGroupSource(seedElide: Bool) -> String {
    seedElide
        ? "if (j == 0) {\n"
            + "                accum =\n"
            + "                    (x_thread[8 * j] * v04.x +\n"
            + "                     x_thread[8 * j + 1] * v15.x +\n"
            + "                     x_thread[8 * j + 2] * v26.x +\n"
            + "                     x_thread[8 * j + 3] * v37.x);\n"
            + "            } else {\n"
            + "                accum +=\n"
            + "                    (x_thread[8 * j] * v04.x +\n"
            + "                     x_thread[8 * j + 1] * v15.x +\n"
            + "                     x_thread[8 * j + 2] * v26.x +\n"
            + "                     x_thread[8 * j + 3] * v37.x);\n"
            + "            }"
        : "accum +=\n"
            + "            (x_thread[8 * j] * v04.x +\n"
            + "             x_thread[8 * j + 1] * v15.x +\n"
            + "             x_thread[8 * j + 2] * v26.x +\n"
            + "             x_thread[8 * j + 3] * v37.x);"
}




func lagunaTailNVFP4RowScaleSuffixSource(scaleDefer: Bool) -> String {
    scaleDefer ? " * 4194304.0f" : ""
}

private let lagunaTailNVFP4QDotReturn = lagunaTailNVFP4ScaleFoldEnabled
    ? "return scale * accum;"
    : "return (scale * 16384.0f) * accum;"

private let lagunaTailNVFP4QMVHeader = """
    static inline float laguna_tail_nvfp4_scale(uint8_t bits) {
        \(lagunaTailNVFP4ScaleFoldEnabled
            ? (lagunaE4M3SignDomainCertified
                ? "ushort raw = ushort(bits) << 7;"
                : "ushort raw = ushort(bits + (bits & 128)) << 7;")
            : "ushort raw = ushort(bits & 127) << 7;")
        \(lagunaTailNVFP4ScaleDecodeSource(scaleFold: lagunaTailNVFP4ScaleFoldEnabled, scaleDefer: lagunaTailNVFP4QKVScaleDeferEnabled))
    }

    static inline float laguna_tail_nvfp4_qdot(
        const device uint8_t* w,
        const thread float* x_thread,
        float scale
    ) {
        \(lagunaTailNVFP4QDotAccumDeclSource(seedElide: lagunaTailNVFP4QKVSeedElisionEnabled))
        const device uint2* wq = (const device uint2*)w;
        const uint2 codes = wq[0];
    #pragma unroll
        for (int j = 0; j < 2; j++) {
            const uint32_t c = (j == 0) ? codes.x : codes.y;
            // Split-nibble decode: the same eight `half` bit patterns per
            // code word as the original shift+mask sequence, in fewer
            // integer ops with three mask constants instead of eight — the
            // form the current stock `fp_qmv_fast` compiles (every form is
            // an OR of masked shifts, so the decode is bit-identical).
            const uint32_t xe = c & 0x0F0F0F0Fu;
            const uint32_t ge = xe | (xe << 3);
            const uint32_t yo = c & 0xF0F0F0F0u;
            const uint32_t go = yo | (yo >> 3);
            const uint32_t p0 = (ge << 9) & 0x8E008E00u;
            const uint32_t p1 = (go << 8) & 0x8E008E00u;
            const uint32_t p2 = (ge << 1) & 0x8E008E00u;
            const uint32_t p3 = go & 0x8E008E00u;
            const float2 v04 = float2(as_type<half2>(p0));
            const float2 v15 = float2(as_type<half2>(p1));
            const float2 v26 = float2(as_type<half2>(p2));
            const float2 v37 = float2(as_type<half2>(p3));
            \(lagunaTailNVFP4QDotFirstGroupSource(seedElide: lagunaTailNVFP4QKVSeedElisionEnabled))
            accum +=
                (x_thread[8 * j + 4] * v04.y +
                 x_thread[8 * j + 5] * v15.y +
                 x_thread[8 * j + 6] * v26.y +
                 x_thread[8 * j + 7] * v37.y);
        }
        \(lagunaTailNVFP4QDotReturn)
    }
    """






private let lagunaDecodeNVFP4QKVR1Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_DECODE_NVFP4_QKV_R1"] != "0"

private func lagunaDecodeNVFP4QKVR1Source(narrow: Bool = false) -> String {




    let scaleSetup =
        narrow
        ? """
const device uint8_t* nb = scale_nibbles +
    out_row * (in_vec_size_g / 2) + (simd_lid >> 1);
const device uint8_t* hb = scale_high_bits +
    out_row * (in_vec_size_g / 8) + (simd_lid >> 3);
const device uint8_t* bs = scale_bases + out_row * (in_vec_size_g / 32);
"""
        : """
const device uint8_t* sc = weight_scales +
    out_row * in_vec_size_g + simd_lid;
"""
    let scaleCode =
        narrow
        ? "uint8_t(bs[0] + ((nb[0] >> ((simd_lid & 1) << 2)) & 0x0Fu) + "
            + "(((hb[0] >> (simd_lid & 7)) & 0x01u) << 4))"
        : "sc[0]"
    let scaleAdvance =
        narrow
        ? """
nb += block_size / 32;
    hb += block_size / 128;
    bs += block_size / 512;
"""
        : "sc += block_size / 16;"
    return """
constexpr uint axis_size = 2048;
constexpr uint num_simdgroups = 2;
constexpr uint values_per_thread = 16;
constexpr uint block_size = 512;
constexpr uint in_vec_size_w = axis_size / 2;
constexpr uint in_vec_size_g = axis_size / 16;

uint tile = threadgroup_position_in_grid.x;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;
uint out_row = tile * num_simdgroups + simd_gid;

const device uint8_t* ws = (const device uint8_t*)weight_codes +
    out_row * in_vec_size_w + simd_lid * 8;
\(scaleSetup)

thread float x_thread[values_per_thread];
thread float result = 0.0f;

uint column = simd_lid * values_per_thread;
for (uint k = 0; k < axis_size; k += block_size) {
    for (uint i = 0; i < values_per_thread; ++i) {
        x_thread[i] = float(normalized[column + i]);
    }
    result += laguna_tail_nvfp4_qdot(
        ws, x_thread, laguna_tail_nvfp4_scale(\(scaleCode)));
    ws += block_size / 2;
    \(scaleAdvance)
    column += block_size;
}

result = simd_sum(result\(lagunaTailNVFP4RowScaleSuffixSource(scaleDefer: lagunaTailNVFP4QKVScaleDeferEnabled)));
if (simd_lid == 0) {
    projected[out_row] = bfloat(result);
}
"""
}

private let lagunaDecodeNVFP4QKVR1Kernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        kernels[heads] = MLXFast.metalKernel(
            name: "laguna_decode_nvfp4_qkv_h\(heads)_r1_v1"
                + (lagunaTailNVFP4QKVSeedElisionEnabled ? "_se1" : "")
                + (lagunaTailNVFP4QKVScaleDeferEnabled ? "_sd1" : ""),
            inputNames: ["normalized", "weight_codes", "weight_scales"],
            outputNames: ["projected"],
            source: lagunaDecodeNVFP4QKVR1Source(),
            header: lagunaTailNVFP4QMVHeader,
            ensureRowContiguous: true)
    }
    return kernels
}()

private let lagunaDecodeNVFP4QKVR1NarrowKernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        kernels[heads] = MLXFast.metalKernel(
            name: "laguna_decode_nvfp4_qkv_h\(heads)_r1_v1_ns1"
                + (lagunaTailNVFP4QKVSeedElisionEnabled ? "_se1" : "")
                + (lagunaTailNVFP4QKVScaleDeferEnabled ? "_sd1" : ""),
            inputNames: [
                "normalized", "weight_codes", "scale_nibbles",
                "scale_high_bits", "scale_bases",
            ],
            outputNames: ["projected"],
            source: lagunaDecodeNVFP4QKVR1Source(narrow: true),
            header: lagunaTailNVFP4QMVHeader,
            ensureRowContiguous: true)
    }
    return kernels
}()








private func lagunaDecodeNVFP4QKVLaneMajorSource(
    pairwise: Bool, tileOffset: String? = nil
) -> String {
    let tileExpr =
        tileOffset.map { "threadgroup_position_in_grid.x - \($0)" }
        ?? "threadgroup_position_in_grid.x"
    return """
constexpr uint axis_size = 2048;
constexpr uint num_simdgroups = 2;
constexpr uint values_per_thread = 16;
constexpr uint block_size = 512;
constexpr uint in_vec_size_w = axis_size / 2;
constexpr uint in_vec_size_g = axis_size / 16;
constexpr uint blocks_per_row = in_vec_size_g / 32;

uint tile = \(tileExpr);
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;
uint out_row = tile * num_simdgroups + simd_gid;

const device uint8_t* ws = (const device uint8_t*)weight_codes +
    out_row * in_vec_size_w + simd_lid * 8;

thread uint8_t sb[blocks_per_row];
const uint8_t row_base = scale_bases[out_row];
if (row_base != 0xFFu) {
    const device ushort* nb = (const device ushort*)(
        scale_nibbles + out_row * (in_vec_size_g / \(pairwise ? 4 : 2)))
        + \(pairwise ? "(simd_lid >> 1)" : "simd_lid");
    const ushort packed = nb[0];
#pragma unroll
    for (uint b = 0; b < blocks_per_row; ++b) {
        sb[b] = uint8_t(row_base + ((packed >> (b << 2)) & 0x0Fu));
    }
} else {
    const device uint8_t* sc = weight_scales +
        out_row * in_vec_size_g + simd_lid;
#pragma unroll
    for (uint b = 0; b < blocks_per_row; ++b) {
        sb[b] = sc[b * (block_size / 16)];
    }
}

thread float x_thread[values_per_thread];
thread float result = 0.0f;

uint column = simd_lid * values_per_thread;
for (uint k = 0; k < axis_size; k += block_size) {
    for (uint i = 0; i < values_per_thread; ++i) {
        x_thread[i] = float(normalized[column + i]);
    }
    result += laguna_tail_nvfp4_qdot(
        ws, x_thread, laguna_tail_nvfp4_scale(sb[k / block_size]));
    ws += block_size / 2;
    column += block_size;
}

result = simd_sum(result\(lagunaTailNVFP4RowScaleSuffixSource(scaleDefer: lagunaTailNVFP4QKVScaleDeferEnabled)));
if (simd_lid == 0) {
    projected[out_row] = bfloat(result);
}
"""
}

private let lagunaDecodeNVFP4QKVLaneMajorKernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        kernels[heads] = MLXFast.metalKernel(
            name: "laguna_decode_nvfp4_qkv_h\(heads)_r1_v1_lm1"
                + (lagunaAttnScalePairwiseQKVEnabled ? "_pw1" : "")
                + (lagunaTailNVFP4QKVSeedElisionEnabled ? "_se1" : "")
                + (lagunaTailNVFP4QKVScaleDeferEnabled ? "_sd1" : ""),
            inputNames: [
                "normalized", "weight_codes", "scale_nibbles", "scale_bases",
                "weight_scales",
            ],
            outputNames: ["projected"],
            source: lagunaDecodeNVFP4QKVLaneMajorSource(
                pairwise: lagunaAttnScalePairwiseQKVEnabled),
            header: lagunaTailNVFP4QMVHeader,
            ensureRowContiguous: true)
    }
    return kernels
}()

private func lagunaDecodeNVFP4QKVR1(
    normalized: MLXArray,
    bank: LagunaNativeAffineWeight,
    heads: Int
) -> MLXArray? {
    guard lagunaDecodeNVFP4QKVR1Enabled else { return nil }
    let rows = (heads + 2 * LagunaConstants.numKeyValueHeads) * LagunaConstants.headDim
    let hidden = LagunaConstants.hiddenSize
    guard normalized.dtype == .bfloat16,
        normalized.dims(1, 1, hidden),
        bank.mode == .nvfp4, bank.bits == 4, bank.groupSize == 16,
        bank.biases == nil,
        bank.originalShape == [rows, hidden],
        bank.packedCodes.dtype == .uint32,
        bank.packedCodes.dims(rows, hidden / 8),
        bank.scales.dtype == .uint8,
        bank.scales.dims(rows, hidden / 16),
        rows % 2 == 0
    else { return nil }
    if let lane = bank.laneMajorScales,
        lane.pairwise == lagunaAttnScalePairwiseQKVEnabled,
        lane.nibbles.dtype == .uint8,
        lane.nibbles.dims(rows, hidden / (lane.pairwise ? 64 : 32)),
        lane.bases.dtype == .uint8, lane.bases.dims(rows),
        let kernel = lagunaDecodeNVFP4QKVLaneMajorKernels[heads]
    {
        lagunaTrace("decode nvfp4 qkv r1 h\(heads) lane-major")
        lagunaNarrowScaleLog.noteDispatch("lane-major", "qkv h\(heads)")
        return kernel(
            [normalized, bank.packedCodes, lane.nibbles, lane.bases, bank.scales],
            grid: ((rows / 2) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[1, 1, rows]],
            outputDTypes: [.bfloat16]
        )[0]
    }
    if let narrow = bank.narrowScales,
        narrow.nibbles.dtype == .uint8, narrow.nibbles.dims(rows, hidden / 32),
        narrow.highBits.dtype == .uint8, narrow.highBits.dims(rows, hidden / 128),
        narrow.bases.dtype == .uint8, narrow.bases.dims(rows, hidden / 512),
        let kernel = lagunaDecodeNVFP4QKVR1NarrowKernels[heads]
    {
        lagunaTrace("decode nvfp4 qkv r1 h\(heads) narrow")
        lagunaNarrowScaleLog.noteDispatch("active", "qkv h\(heads)")
        return kernel(
            [normalized, bank.packedCodes, narrow.nibbles, narrow.highBits, narrow.bases],
            grid: ((rows / 2) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[1, 1, rows]],
            outputDTypes: [.bfloat16]
        )[0]
    }
    guard let kernel = lagunaDecodeNVFP4QKVR1Kernels[heads] else { return nil }
    lagunaTrace("decode nvfp4 qkv r1 h\(heads)")
    lagunaNarrowScaleLog.noteDispatch("inactive", "qkv h\(heads)")
    return kernel(
        [normalized, bank.packedCodes, bank.scales],
        grid: ((rows / 2) * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, rows]],
        outputDTypes: [.bfloat16]
    )[0]
}

private let lagunaDecodeNVFP4QKVGateFusedEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_DECODE_QKV_GATE_FUSED"] != "0"

/// QKV and `g_proj` both read the same normalized row and neither consumes the
/// other, so the two grids are appended into one dispatch instead of adding a
/// dependency edge. The `heads / 8` gate tiles lead so they are scheduled in
/// the first wave rather than in the QKV drain tail.
private func lagunaDecodeNVFP4QKVGateSource(pairwise: Bool, heads: Int) -> String {
    let gateBody = lagunaGateSoftplusSource(
        heads: heads,
        inputName: "normalized",
        codesName: "gate_codes",
        scalesName: "gate_scales",
        biasesName: "gate_biases")
    return """
constexpr uint laguna_gate_tiles = \(heads / 8);
if (threadgroup_position_in_grid.x < laguna_gate_tiles) {
\(gateBody)
    return;
}
\(lagunaDecodeNVFP4QKVLaneMajorSource(
    pairwise: pairwise, tileOffset: "laguna_gate_tiles"))
"""
}

private let lagunaDecodeNVFP4QKVGateKernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        kernels[heads] = MLXFast.metalKernel(
            name: "laguna_decode_nvfp4_qkv_gate_h\(heads)_r1_v1_lm1"
                + (lagunaAttnScalePairwiseQKVEnabled ? "_pw1" : "")
                + (lagunaTailNVFP4QKVSeedElisionEnabled ? "_se1" : "")
                + (lagunaTailNVFP4QKVScaleDeferEnabled ? "_sd1" : ""),
            inputNames: [
                "normalized", "weight_codes", "scale_nibbles", "scale_bases",
                "weight_scales", "gate_codes", "gate_scales", "gate_biases",
            ],
            outputNames: ["projected", "gate_values"],
            source: lagunaDecodeNVFP4QKVGateSource(
                pairwise: lagunaAttnScalePairwiseQKVEnabled, heads: heads),
            header: lagunaTailNVFP4QMVHeader,
            ensureRowContiguous: true)
    }
    return kernels
}()

private func lagunaDecodeNVFP4QKVGate(
    normalized: MLXArray,
    bank: LagunaNativeAffineWeight,
    gateBank: LagunaNativeAffineWeight,
    heads: Int
) -> (qkv: MLXArray, gate: MLXArray)? {
    guard lagunaDecodeNVFP4QKVGateFusedEnabled,
        lagunaDecodeNVFP4QKVR1Enabled,
        lagunaGateSoftplusEnabled
    else { return nil }
    let rows = (heads + 2 * LagunaConstants.numKeyValueHeads) * LagunaConstants.headDim
    let hidden = LagunaConstants.hiddenSize
    guard normalized.dtype == .bfloat16,
        normalized.dims(1, 1, hidden),
        bank.mode == .nvfp4, bank.bits == 4, bank.groupSize == 16,
        bank.biases == nil,
        bank.originalShape == [rows, hidden],
        bank.packedCodes.dtype == .uint32,
        bank.packedCodes.dims(rows, hidden / 8),
        bank.scales.dtype == .uint8,
        bank.scales.dims(rows, hidden / 16),
        rows % 2 == 0,
        let lane = bank.laneMajorScales,
        lane.pairwise == lagunaAttnScalePairwiseQKVEnabled,
        lane.nibbles.dtype == .uint8,
        lane.nibbles.dims(rows, hidden / (lane.pairwise ? 64 : 32)),
        lane.bases.dtype == .uint8, lane.bases.dims(rows),
        gateBank.mode == .affine, gateBank.bits == 8, gateBank.groupSize == 32,
        let gateBiases = gateBank.biases,
        gateBank.packedCodes.dims(heads, hidden / 4),
        gateBank.scales.dims(heads, hidden / 32),
        gateBiases.dims(heads, hidden / 32),
        heads % 8 == 0,
        let kernel = lagunaDecodeNVFP4QKVGateKernels[heads]
    else { return nil }
    lagunaTrace("decode nvfp4 qkv+gate h\(heads) lane-major")
    lagunaNarrowScaleLog.noteDispatch("lane-major", "qkv+gate h\(heads)")
    let outputs = kernel(
        [
            normalized, bank.packedCodes, lane.nibbles, lane.bases, bank.scales,
            gateBank.packedCodes, gateBank.scales, gateBiases,
        ],
        grid: ((heads / 8 + rows / 2) * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, rows], [1, 1, heads]],
        outputDTypes: [.bfloat16, .bfloat16]
    )
    return (outputs[0], outputs[1])
}


private func lagunaNormAffineQKVSource(rows: Int, staged: Bool) -> String {
    let stagedNormalize = """
    for (uint j = 0; j < virtual_per_thread; ++j) {
        uint base = (lid + j * real_threads) * n_reads;
        for (uint i = 0; i < n_reads; ++i) {
            norm_row[base + i] =
                norm_weight[base + i] *
                bfloat(float(residual[base + i]) * laguna_inv_mean);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""
    let scratch =
        staged
        ? "threadgroup bfloat norm_row[axis_size];"
        : "// inline: no staged row, no third barrier"
    let normalize = staged ? stagedNormalize : ""
    let loadValue =
        staged
        ? "float value = float(norm_row[column + i]);"
        : """
float value = float(bfloat(
                norm_weight[column + i] *
                bfloat(float(residual[column + i]) * laguna_inv_mean)));
"""
    return """
\(lagunaNormAffineQKVBody(rows: rows, scratch: scratch, normalize: normalize, loadValue: loadValue))
"""
}

private func lagunaNormAffineQKVBody(
    rows: Int, scratch: String, normalize: String, loadValue: String
) -> String {
    return """
constexpr uint axis_size = \(LagunaConstants.hiddenSize);
constexpr uint out_vec_size = \(rows);
constexpr uint n_reads = 4;
constexpr uint norm_threads = axis_size / n_reads;
constexpr uint real_threads = 64;
constexpr uint virtual_per_thread = norm_threads / real_threads;
constexpr uint simd_size = 32;
constexpr float norm_eps = 1.0e-6f;
constexpr uint values_per_thread = 8;
constexpr uint block_size = 256;
constexpr uint results_per_simdgroup = 4;
constexpr uint num_simdgroups = 2;
constexpr uint group_size = 32;
constexpr uint scale_step_per_thread = group_size / values_per_thread;
constexpr uint in_vec_size_g = axis_size / group_size;

uint tile = threadgroup_position_in_grid.x;
uint lid = thread_position_in_threadgroup.x;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;

threadgroup float local_inv_mean[1];
threadgroup float local_sums[simd_size];
\(scratch)

if (lid < simd_size) {
    local_sums[lid] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint j = 0; j < virtual_per_thread; ++j) {
    uint base = (lid + j * real_threads) * n_reads;
    float acc = 0.0f;
    for (uint i = 0; i < n_reads; ++i) {
        float xi = float(residual[base + i]);
        acc += xi * xi;
    }
    acc = simd_sum(acc);
    if (simd_lid == 0) {
        local_sums[simd_gid + num_simdgroups * j] = acc;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (simd_gid == 0) {
    float total = simd_sum(local_sums[simd_lid]);
    if (simd_lid == 0) {
        local_inv_mean[0] =
            metal::precise::rsqrt(total / float(axis_size) + norm_eps);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float laguna_inv_mean = local_inv_mean[0];
\(normalize)
uint out_row = tile * (num_simdgroups * results_per_simdgroup) +
    simd_gid * results_per_simdgroup;

const device uint8_t* ws = (const device uint8_t*)weight_codes +
    out_row * axis_size + simd_lid * values_per_thread;
const device bfloat* sc = weight_scales + out_row * in_vec_size_g +
    simd_lid / scale_step_per_thread;
const device bfloat* bs = weight_biases + out_row * in_vec_size_g +
    simd_lid / scale_step_per_thread;

thread float x_thread[values_per_thread];
thread float result[results_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};

uint column = simd_lid * values_per_thread;
for (uint k = 0; k < axis_size; k += block_size) {
    float sum = 0.0f;
    for (uint i = 0; i < values_per_thread; ++i) {
        \(loadValue)
        sum += value;
        x_thread[i] = value;
    }

    for (uint row = 0; row < results_per_simdgroup; ++row) {
        const device uint8_t* wl = ws + row * axis_size;
        float scale = float(sc[row * in_vec_size_g]);
        float bias = float(bs[row * in_vec_size_g]);
        float accum = 0.0f;
        for (uint i = 0; i < values_per_thread; ++i) {
            accum += x_thread[i] * wl[i];
        }
        result[row] += scale * accum + sum * bias;
    }

    ws += block_size;
    sc += block_size / group_size;
    bs += block_size / group_size;
    column += block_size;
}

for (uint row = 0; row < results_per_simdgroup; ++row) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
        projected[out_row + row] = bfloat(result[row]);
    }
}
"""
}


private let lagunaNormAffineQKVStaged =
    ProcessInfo.processInfo.environment["DARKBLOOM_NORM_AFFINE_QKV_STAGE"] == "tg"


















let lagunaNormAffineQKVPrefetchDepth: Int = {
    let raw =
        ProcessInfo.processInfo.environment["DARKBLOOM_NORM_AFFINE_QKV_PF"] ?? "4"
    return min(max(Int(raw) ?? 4, 0), 4)
}()








private func lagunaNormAffineQKVPrefetchSource(
    rows: Int, depth: Int, indexed: Bool = false
) -> String {
    let metadataPointers = indexed
        ? """
const device ushort* mi = metadata_indices + out_row * in_vec_size_g +
    simd_lid / scale_step_per_thread;
"""
        : """
const device bfloat* sc = weight_scales + out_row * in_vec_size_g +
    simd_lid / scale_step_per_thread;
const device bfloat* bs = weight_biases + out_row * in_vec_size_g +
    simd_lid / scale_step_per_thread;
"""
    let prefetchMetadata = indexed
        ? """
    uint pair = metadata_lut[
        mi[d * (block_size / group_size) + row * in_vec_size_g]];
    pf_s[d][row] = float(as_type<bfloat>(ushort(pair)));
    pf_b[d][row] = float(as_type<bfloat>(ushort(pair >> 16)));
"""
        : """
    pf_s[d][row] =
        float(sc[d * (block_size / group_size) + row * in_vec_size_g]);
    pf_b[d][row] =
        float(bs[d * (block_size / group_size) + row * in_vec_size_g]);
"""
    let metadataLoad = indexed
        ? """
    uint pair = metadata_lut[mi[row * in_vec_size_g]];
    float scale = float(as_type<bfloat>(ushort(pair)));
    float bias = float(as_type<bfloat>(ushort(pair >> 16)));
"""
        : """
    float scale = float(sc[row * in_vec_size_g]);
    float bias = float(bs[row * in_vec_size_g]);
"""
    let metadataAdvance = indexed
        ? "mi += block_size / group_size;"
        : """
sc += block_size / group_size;
bs += block_size / group_size;
"""
    return """
constexpr uint axis_size = \(LagunaConstants.hiddenSize);
constexpr uint out_vec_size = \(rows);
constexpr uint n_reads = 4;
constexpr uint norm_threads = axis_size / n_reads;
constexpr uint real_threads = 64;
constexpr uint virtual_per_thread = norm_threads / real_threads;
constexpr uint simd_size = 32;
constexpr float norm_eps = 1.0e-6f;
constexpr uint values_per_thread = 8;
constexpr uint block_size = 256;
constexpr uint results_per_simdgroup = 4;
constexpr uint num_simdgroups = 2;
constexpr uint group_size = 32;
constexpr uint scale_step_per_thread = group_size / values_per_thread;
constexpr uint in_vec_size_g = axis_size / group_size;
constexpr uint pf_depth = \(depth);

uint tile = threadgroup_position_in_grid.x;
uint lid = thread_position_in_threadgroup.x;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;

threadgroup float local_inv_mean[1];
threadgroup float local_sums[simd_size];

uint out_row = tile * (num_simdgroups * results_per_simdgroup) +
    simd_gid * results_per_simdgroup;

const device uint8_t* ws = (const device uint8_t*)weight_codes +
    out_row * axis_size + simd_lid * values_per_thread;
\(metadataPointers)

uint8_t pf_w[pf_depth][results_per_simdgroup][values_per_thread];
float pf_s[pf_depth][results_per_simdgroup];
float pf_b[pf_depth][results_per_simdgroup];
for (uint d = 0; d < pf_depth; ++d) {
    for (uint row = 0; row < results_per_simdgroup; ++row) {
        const device uint8_t* wl = ws + d * block_size + row * axis_size;
        for (uint i = 0; i < values_per_thread; ++i) {
            pf_w[d][row][i] = wl[i];
        }
        \(prefetchMetadata)
    }
}

if (lid < simd_size) {
    local_sums[lid] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint j = 0; j < virtual_per_thread; ++j) {
    uint base = (lid + j * real_threads) * n_reads;
    float acc = 0.0f;
    for (uint i = 0; i < n_reads; ++i) {
        float xi = float(residual[base + i]);
        acc += xi * xi;
    }
    acc = simd_sum(acc);
    if (simd_lid == 0) {
        local_sums[simd_gid + num_simdgroups * j] = acc;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (simd_gid == 0) {
    float total = simd_sum(local_sums[simd_lid]);
    if (simd_lid == 0) {
        local_inv_mean[0] =
            metal::precise::rsqrt(total / float(axis_size) + norm_eps);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float laguna_inv_mean = local_inv_mean[0];

thread float x_thread[values_per_thread];
thread float result[results_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};

uint column = simd_lid * values_per_thread;
for (uint d = 0; d < pf_depth; ++d) {
    float sum = 0.0f;
    for (uint i = 0; i < values_per_thread; ++i) {
        float value = float(bfloat(
                        norm_weight[column + i] *
                        bfloat(float(residual[column + i]) * laguna_inv_mean)));
        sum += value;
        x_thread[i] = value;
    }
    for (uint row = 0; row < results_per_simdgroup; ++row) {
        float accum = 0.0f;
        for (uint i = 0; i < values_per_thread; ++i) {
            accum += x_thread[i] * pf_w[d][row][i];
        }
        result[row] += pf_s[d][row] * accum + sum * pf_b[d][row];
    }
    ws += block_size;
    \(metadataAdvance)
    column += block_size;
}
for (uint k = pf_depth * block_size; k < axis_size; k += block_size) {
    float sum = 0.0f;
    for (uint i = 0; i < values_per_thread; ++i) {
        float value = float(bfloat(
                        norm_weight[column + i] *
                        bfloat(float(residual[column + i]) * laguna_inv_mean)));
        sum += value;
        x_thread[i] = value;
    }

    for (uint row = 0; row < results_per_simdgroup; ++row) {
        const device uint8_t* wl = ws + row * axis_size;
        \(metadataLoad)
        float accum = 0.0f;
        for (uint i = 0; i < values_per_thread; ++i) {
            accum += x_thread[i] * wl[i];
        }
        result[row] += scale * accum + sum * bias;
    }

    ws += block_size;
    \(metadataAdvance)
    column += block_size;
}

for (uint row = 0; row < results_per_simdgroup; ++row) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
        projected[out_row + row] = bfloat(result[row]);
    }
}
"""
}




private let lagunaNormAffineQKVKernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    let kvRows = 2 * LagunaConstants.numKeyValueHeads * LagunaConstants.headDim
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        for gateRows in [0, heads] {
            let rows = heads * LagunaConstants.headDim + kvRows + gateRows
            if kernels[rows] != nil { continue }
            let staged = lagunaNormAffineQKVStaged
            let pf = staged ? 0 : lagunaNormAffineQKVPrefetchDepth
            kernels[rows] = MLXFast.metalKernel(
                name: pf > 0
                    ? "laguna_norm_affine_qkv_qmv_i8g32_r\(rows)_pf\(pf)_v1"
                    : "laguna_norm_affine_qkv_qmv_i8g32_r\(rows)_"
                        + (staged ? "tg" : "inl") + "_v1",
                inputNames: [
                    "residual", "norm_weight", "weight_codes", "weight_scales",
                    "weight_biases",
                ],
                outputNames: ["projected"],
                source: pf > 0
                    ? lagunaNormAffineQKVPrefetchSource(rows: rows, depth: pf)
                    : lagunaNormAffineQKVSource(rows: rows, staged: staged),
                ensureRowContiguous: true
            )
        }
    }
    return kernels
}()

private let lagunaNormAffineQKVIndexedKernels: [Int: MLXFast.MLXFastKernel] = {
    var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    guard !lagunaNormAffineQKVStaged, lagunaNormAffineQKVPrefetchDepth > 0 else {
        return kernels
    }
    let kvRows = 2 * LagunaConstants.numKeyValueHeads * LagunaConstants.headDim
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        for gateRows in [0, heads] {
            let rows = heads * LagunaConstants.headDim + kvRows + gateRows
            if kernels[rows] != nil { continue }
            kernels[rows] = MLXFast.metalKernel(
                name: "laguna_norm_affine_qkv_qmv_i8g32_r\(rows)_"
                    + "pf\(lagunaNormAffineQKVPrefetchDepth)_idx_v1",
                inputNames: [
                    "residual", "norm_weight", "weight_codes",
                    "metadata_indices", "metadata_lut",
                ],
                outputNames: ["projected"],
                source: lagunaNormAffineQKVPrefetchSource(
                    rows: rows, depth: lagunaNormAffineQKVPrefetchDepth,
                    indexed: true),
                ensureRowContiguous: true
            )
        }
    }
    return kernels
}()








let lagunaFusedNormAffineQKVEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_NORM_AFFINE_QKV"] != "0"




func lagunaNormAffineQKV(
    residual: MLXArray,
    normWeight: MLXArray,
    codes: MLXArray,
    scales: MLXArray,
    biases: MLXArray,
    indexedMetadata: LagunaIndexedAffineMetadata? = nil,
    rows: Int
) -> MLXArray? {
    guard let kernel = lagunaNormAffineQKVKernels[rows] else { return nil }
    let hidden = LagunaConstants.hiddenSize
    guard residual.dtype == .bfloat16,
        residual.dims(1, 1, hidden),
        normWeight.dtype == .bfloat16,
        normWeight.dims(hidden),
        codes.dtype == .uint32,
        codes.dims(rows, hidden / 4),
        scales.dtype == .bfloat16,
        scales.dims(rows, hidden / 32),
        biases.dtype == .bfloat16,
        biases.dims(rows, hidden / 32)
    else {
        return nil
    }

    if let metadata = indexedMetadata,
        let indexedKernel = lagunaNormAffineQKVIndexedKernels[rows],
        metadata.indices.dtype == .uint16,
        metadata.indices.dims(rows, hidden / 32),
        metadata.lut.dtype == .uint32,
        metadata.lut.ndim == 1,
        metadata.lut.size <= 65_536
    {
        lagunaTrace(
            "norm+affine qkv qmv r\(rows) "
                + "pf\(lagunaNormAffineQKVPrefetchDepth) indexed")
        return indexedKernel(
            [residual, normWeight, codes, metadata.indices, metadata.lut],
            grid: ((rows / 8) * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[1, 1, rows]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    lagunaTrace(
        lagunaNormAffineQKVPrefetchDepth > 0 && !lagunaNormAffineQKVStaged
            ? "norm+affine qkv qmv r\(rows) pf\(lagunaNormAffineQKVPrefetchDepth)"
            : "norm+affine qkv qmv r\(rows)")
    return kernel(
        [residual, normWeight, codes, scales, biases],
        grid: ((rows / 8) * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, rows]],
        outputDTypes: [.bfloat16]
    )[0]
}




private let lagunaCompiledSoftplusGate: @Sendable (MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray) -> MLXArray = { gate in
        softplus(gate.asType(.float32)).asType(gate.dtype)
    }
    return MLXHardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
}()






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





final class LagunaRuntimeAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let gatingEnabled: Bool
    let gatePerHead: Bool
    let isSliding: Bool
    let layerIdx: Int



    lazy var _fusedAttnScale: MLXArray = MLXArray([scale])
    let attentionGateProjection: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer







    var _fusedQKVWeight: MLXArray?





    var _lastPrefillQGateWeight: MLXArray?
    var _lastPrefillKVWeight: MLXArray?




    var _nativeAffineQKV: LagunaNativeAffineWeight?





    var _nativeAffineOProj: LagunaNativeAffineWeight?








    var _nativeAffineGProj: LagunaNativeAffineWeight?





    var _nativeAffineQKVGateRows = 0

    func prepareNativeAffineOProjWeight() -> [MLXArray] {
        guard _nativeAffineOProj == nil,
            type(of: wo) == Linear.self,
            wo.bias == nil,
            wo.weight.dims(LagunaConstants.hiddenSize, nHeads * headDim),
            let quantizedWO = lagunaNativeAffineWeight(wo.weight, layer: layerIdx)
        else {
            return []
        }
        var preparedWO = quantizedWO
        if preparedWO.mode == .affine, preparedWO.bits == 8,
            preparedWO.groupSize == 32, let biases = preparedWO.biases
        {
            preparedWO.indexedMetadata = lagunaIndexedAffineMetadata(
                scales: preparedWO.scales, biases: biases)
        }
        if lagunaAttnScaleNarrowOProjEnabled, preparedWO.mode == .nvfp4,
            preparedWO.bits == 4, preparedWO.groupSize == 16
        {
            preparedWO.laneMajorScales = lagunaLaneMajorNVFP4ScaleBank(
                preparedWO.scales, site: "oproj", layer: layerIdx,
                pairwise: lagunaAttnScalePairwiseOProjEnabled)
        }
        _nativeAffineOProj = preparedWO
        return preparedWO.arrays
    }

    func prepareNativeAffineQKVWeight() -> [MLXArray] {
        guard _nativeAffineQKV == nil,
            let q = lagunaNativeAffineWeight(wq.weight, layer: layerIdx),
            let k = lagunaNativeAffineWeight(wk.weight, layer: layerIdx),
            let v = lagunaNativeAffineWeight(wv.weight, layer: layerIdx)
        else {
            return []
        }







        var gate: LagunaNativeAffineWeight?
        if lagunaNativeAffineGProjEnabled,
            lagunaUseNativeAffineGProj(layer: layerIdx),
            gatingEnabled, gatePerHead,
            let gProj,
            type(of: gProj) == Linear.self,
            gProj.bias == nil,
            gProj.weight.dtype == .bfloat16,
            gProj.weight.dims(nHeads, LagunaConstants.hiddenSize)
        {
            gate = lagunaNativeAffineGProjWeight(gProj.weight)
        }
        let foldGateIntoBank =
            gate != nil && q.groupSize == 32 && q.bits == 8 && q.mode == .affine
        var packedBlocks = [q.packedCodes, k.packedCodes, v.packedCodes]
        var scaleBlocks = [q.scales, k.scales, v.scales]
        var biasBlocks = [q.biases, k.biases, v.biases]
        var totalRows = wq.weight.dim(0) + wk.weight.dim(0) + wv.weight.dim(0)
        if foldGateIntoBank, let gate {
            packedBlocks.append(gate.packedCodes)
            scaleBlocks.append(gate.scales)
            biasBlocks.append(gate.biases)
            totalRows += nHeads
            _nativeAffineQKVGateRows = nHeads
        } else if let gate {
            _nativeAffineGProj = gate
        }
        let packedCodes = concatenated(packedBlocks, axis: 0)
        let scales = concatenated(scaleBlocks, axis: 0)
        var biases: MLXArray?
        if biasBlocks.allSatisfy({ $0 != nil }) {
            biases = concatenated(biasBlocks.compactMap { $0 }, axis: 0)
        }
        var fused = LagunaNativeAffineWeight(
            packedCodes: packedCodes,
            scales: scales,
            biases: biases,
            originalShape: [
                totalRows,
                wq.weight.dim(1),
            ],
            groupSize: q.groupSize,
            bits: q.bits,
            mode: q.mode
        )
        if fused.mode == .affine, fused.bits == 8, fused.groupSize == 32,
            let biases = fused.biases
        {
            fused.indexedMetadata = lagunaIndexedAffineMetadata(
                scales: fused.scales, biases: biases)
        }
        if lagunaAttnScaleNarrowQKVEnabled, fused.mode == .nvfp4,
            fused.bits == 4, fused.groupSize == 16
        {



            fused.laneMajorScales = lagunaLaneMajorNVFP4ScaleBank(
                fused.scales, site: "qkv", layer: layerIdx,
                pairwise: lagunaAttnScalePairwiseQKVEnabled)
            if fused.laneMajorScales == nil {
                fused.narrowScales = lagunaNarrowNVFP4ScaleBank(
                    fused.scales, site: "qkv", layer: layerIdx)
            }
        }
        _nativeAffineQKV = fused
        return fused.arrays + (_nativeAffineGProj?.arrays ?? [])
    }








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






    func prepareLastPrefillProjectionWeights() -> [MLXArray] {
        guard lagunaLastPrefillProjectionBanksEnabled,
            _lastPrefillQGateWeight == nil,
            _lastPrefillKVWeight == nil,
            layerIdx == LagunaConstants.numHiddenLayers - 1,
            isSliding, gatingEnabled, gatePerHead,
            let gProj,
            type(of: wq) == Linear.self,
            type(of: wk) == Linear.self,
            type(of: wv) == Linear.self,
            type(of: gProj) == Linear.self,
            wq.bias == nil, wk.bias == nil, wv.bias == nil, gProj.bias == nil,
            wq.weight.dtype == .bfloat16,
            wk.weight.dtype == .bfloat16,
            wv.weight.dtype == .bfloat16,
            gProj.weight.dtype == .bfloat16,
            wq.weight.dims(nHeads * headDim, LagunaConstants.hiddenSize),
            wk.weight.dims(nKVHeads * headDim, LagunaConstants.hiddenSize),
            wv.weight.dims(nKVHeads * headDim, LagunaConstants.hiddenSize),
            gProj.weight.dims(nHeads, LagunaConstants.hiddenSize)
        else {
            return []
        }
        let qGate = concatenated([wq.weight, gProj.weight], axis: 0)
        let kv = concatenated([wk.weight, wv.weight], axis: 0)
        _lastPrefillQGateWeight = qGate
        _lastPrefillKVWeight = kv
        return [qGate, kv]
    }

    init(_ config: LagunaConfig, layerIdx: Int) {
        let dim = config.hiddenSize
        self.layerIdx = layerIdx
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
        _ input: MLXArray,
        inputNorm: RMSNorm,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        qkRoPEAngles: MLXArray? = nil,
        qkRoPEOffsets: MLXArray? = nil
    ) -> MLXArray {
        let (B, L) = (input.dim(0), input.dim(1))




        var fusedNormQKV:
            (
                queries: MLXArray, keys: MLXArray, values: MLXArray,
                gateValues: MLXArray, gateActivated: Bool
            )?
        if lagunaFusedQKVProjectionEnabled, _fusedQKVWeight == nil,
            B == 1, L == 1,
            headDim == LagunaConstants.headDim,
            nKVHeads == LagunaConstants.numKeyValueHeads,
            input.dtype == .bfloat16,
            input.dims(1, 1, LagunaConstants.hiddenSize),
            inputNorm.weight.dtype == .bfloat16,
            inputNorm.weight.dims(LagunaConstants.hiddenSize),
            wq.bias == nil, wk.bias == nil, wv.bias == nil,
            type(of: wq) == Linear.self, type(of: wk) == Linear.self,
            type(of: wv) == Linear.self,
            wq.weight.dtype == .bfloat16, wk.weight.dtype == .bfloat16,
            wv.weight.dtype == .bfloat16,
            gatingEnabled, gatePerHead,
            let gateProjection = gProj,
            gateProjection.bias == nil,
            type(of: gateProjection) == Linear.self,
            gateProjection.weight.dtype == .bfloat16,
            gateProjection.weight.dims(nHeads, LagunaConstants.hiddenSize)
        {
            if lagunaUseNativeAffineQKV(layer: layerIdx),
                let fusedAffine = _nativeAffineQKV
            {






                var fusedQKV: MLXArray?
                if lagunaFusedNormAffineQKVEnabled,
                    fusedAffine.mode == .affine, fusedAffine.bits == 8,
                    fusedAffine.groupSize == 32,
                    _nativeAffineQKVGateRows == nHeads,
                    inputNorm.eps == Float(LagunaConstants.rmsNormEpsilon),
                    let affineBiases = fusedAffine.biases
                {
                    fusedQKV = lagunaNormAffineQKV(
                        residual: input,
                        normWeight: inputNorm.weight,
                        codes: fusedAffine.packedCodes,
                        scales: fusedAffine.scales,
                        biases: affineBiases,
                        indexedMetadata: fusedAffine.indexedMetadata,
                        rows: fusedAffine.originalShape[0])
                }





                let fusedTailGateLogits: MLXArray? = nil


                let normalized = fusedQKV ?? inputNorm(input)
                var fusedQKVGate: (qkv: MLXArray, gate: MLXArray)?
                if fusedQKV == nil,
                    _nativeAffineQKVGateRows != nHeads,
                    lagunaFusedGatedAffineOProjEnabled,
                    lagunaGatedAffineOProjNVFP4Enabled,
                    lagunaUseNativeAffineOProj(layer: layerIdx),
                    let affineGate = _nativeAffineGProj,
                    let affineWO = _nativeAffineOProj,
                    affineWO.mode == .nvfp4, affineWO.bits == 4,
                    affineWO.groupSize == 16
                {
                    fusedQKVGate = lagunaDecodeNVFP4QKVGate(
                        normalized: normalized, bank: fusedAffine,
                        gateBank: affineGate, heads: nHeads)
                }
                let decodeNVFP4QKVR1 =
                    fusedQKV == nil
                    ? (fusedQKVGate?.qkv
                        ?? lagunaDecodeNVFP4QKVR1(
                            normalized: normalized, bank: fusedAffine, heads: nHeads))
                    : nil
                let qkv =
                    fusedQKV
                    ?? decodeNVFP4QKVR1
                    ?? quantizedMM(
                        normalized,
                        fusedAffine.packedCodes,
                        scales: fusedAffine.scales,
                        biases: fusedAffine.biases,
                        transpose: true,
                        groupSize: fusedAffine.groupSize,
                        bits: fusedAffine.bits,
                        mode: fusedAffine.mode
                    )
                let queryDim = nHeads * headDim
                let kvDim = nKVHeads * headDim
                let gateStart = queryDim + 2 * kvDim
                let gateLogits: MLXArray
                var gateProjectionActivated = false
                if let fusedTailGateLogits {



                    gateLogits = fusedTailGateLogits
                } else if _nativeAffineQKVGateRows == nHeads {



                    gateLogits = qkv[.ellipsis, gateStart ..< (gateStart + nHeads)]
                } else if let affineGate = _nativeAffineGProj {



                    if let fusedGate = fusedQKVGate?.gate {
                        gateLogits = fusedGate
                        gateProjectionActivated = true
                    } else if lagunaFusedGatedAffineOProjEnabled,
                        lagunaGatedAffineOProjNVFP4Enabled,
                        lagunaUseNativeAffineOProj(layer: layerIdx),
                        let affineWO = _nativeAffineOProj,
                        affineWO.mode == .nvfp4, affineWO.bits == 4,
                        affineWO.groupSize == 16,
                        let activated = lagunaGateSoftplus(
                            input: normalized, bank: affineGate, heads: nHeads)
                    {
                        gateLogits = activated
                        gateProjectionActivated = true
                    } else {
                        gateLogits = quantizedMM(
                            normalized,
                            affineGate.packedCodes,
                            scales: affineGate.scales,
                            biases: affineGate.biases,
                            transpose: true,
                            groupSize: affineGate.groupSize,
                            bits: affineGate.bits,
                            mode: affineGate.mode
                        )
                    }
                } else {
                    gateLogits = gateProjection(normalized)
                }
                if lagunaAttentionProjectionAsyncEnabled,
                    layerIdx == 0, B == 1, L == 1
                {
                    lagunaTrace("attention projection async rung layer 0")
                    asyncEval(qkv, gateLogits)
                }





                let deferGateActivation =
                    lagunaFusedGateProductEnabled
                    && lagunaUseNativeAffineOProj(layer: layerIdx)
                    && _nativeAffineOProj != nil
                    && wo.bias == nil
                let gateValues =
                    gateProjectionActivated || deferGateActivation
                    ? gateLogits
                    : softplus(gateLogits.asType(.float32)).asType(.bfloat16)
                fusedNormQKV = (
                    qkv[.ellipsis, 0 ..< queryDim],
                    qkv[.ellipsis, queryDim ..< (queryDim + kvDim)],
                    qkv[.ellipsis, (queryDim + kvDim) ..< gateStart],
                    gateValues,
                    gateProjectionActivated || !deferGateActivation
                )
            } else {
                fusedNormQKV = lagunaFusedNormQKVProjection(
                    residual: input,
                    normWeight: inputNorm.weight,
                    queryWeight: wq.weight,
                    keyWeight: wk.weight,
                    valueWeight: wv.weight,
                    gateWeight: gateProjection.weight,
                    heads: nHeads
                )
            }
        }





        let normalizedInput: MLXArray? =
            fusedNormQKV == nil ? inputNorm(input) : nil

        var queries: MLXArray
        var keys: MLXArray
        var values: MLXArray




        if let fusedQKVWeight = _fusedQKVWeight, L > 1 {
            guard let normalizedInput else {
                preconditionFailure("retained fused QKV requires normalized input")
            }






            let qkv = matmul(normalizedInput, fusedQKVWeight.T)
            let queryDim = nHeads * headDim
            let kvDim = nKVHeads * headDim
            queries = qkv[.ellipsis, 0 ..< queryDim]
            keys = qkv[.ellipsis, queryDim ..< (queryDim + kvDim)]
            values = qkv[.ellipsis, (queryDim + kvDim) ..< (queryDim + 2 * kvDim)]
        } else if let fused = fusedNormQKV {
            queries = fused.queries
            keys = fused.keys
            values = fused.values
        } else {
            guard let normalizedInput else {
                preconditionFailure("stock QKV projections require normalized input")
            }
            queries = wq(normalizedInput)
            keys = wk(normalizedInput)
            values = wv(normalizedInput)
        }

        let fusedQKNormShapesMatch =
            B == 1 && L == 1 &&
            nKVHeads == LagunaConstants.numKeyValueHeads &&
            headDim == LagunaConstants.headDim &&
            queries.dtype == .bfloat16 && keys.dtype == .bfloat16 &&
            qNorm.weight.dtype == .bfloat16 && kNorm.weight.dtype == .bfloat16 &&
            queries.dims(1, 1, nHeads * headDim) &&
            keys.dims(1, 1, nKVHeads * headDim)

        let useFusedFullQKNormYaRN =
            lagunaFusedFullQKNormYaRNEnabled && !isSliding &&
            fusedQKNormShapesMatch &&
            nHeads == LagunaConstants.fullAttentionHeads &&
            qkRoPEAngles?.dtype == .float32 &&
            qkRoPEAngles?.dims(1, 1, 1, headDim / 2) == true

        let useFusedSlidingQKNormRoPE =
            lagunaFusedSlidingQKNormRoPEEnabled && isSliding &&
            fusedQKNormShapesMatch &&
            nHeads == LagunaConstants.slidingAttentionHeads &&
            qkRoPEAngles?.dtype == .float32 &&
            qkRoPEAngles?.dims(1, 1, 1, headDim) == true








        let prefillQKNormShapesMatch =
            B == 1 && L > 1 &&
            nKVHeads == LagunaConstants.numKeyValueHeads &&
            headDim == LagunaConstants.headDim &&
            queries.dtype == .bfloat16 && keys.dtype == .bfloat16 &&
            qNorm.weight.dtype == .bfloat16 && kNorm.weight.dtype == .bfloat16 &&
            queries.dims(1, L, nHeads * headDim) &&
            keys.dims(1, L, nKVHeads * headDim) &&
            qkRoPEAngles?.dtype == .float32 &&
            qkRoPEOffsets?.dtype == .int32 && qkRoPEOffsets?.size == 1

        let usePrefillFusedSlidingQKNormRoPE =
            lagunaPrefillQKNormRoPEEnabled && isSliding &&
            prefillQKNormShapesMatch &&
            nHeads == LagunaConstants.slidingAttentionHeads &&
            qkRoPEAngles?.dims(1, 1, lagunaRoPEAngleAtlasLength, headDim) == true

        let usePrefillFusedFullQKNormYaRN =
            lagunaPrefillQKNormRoPEEnabled && !isSliding &&
            prefillQKNormShapesMatch &&
            nHeads == LagunaConstants.fullAttentionHeads &&
            qkRoPEAngles?.dims(1, 1, lagunaRoPEAngleAtlasLength, headDim / 2) == true

        var qkNormRoPEFused = false
        var fusedAttended: MLXArray?
        if lagunaFusedSlidingAttentionEnabled,
            useFusedSlidingQKNormRoPE,
            let fusedAngles = qkRoPEAngles,
            values.dtype == .bfloat16,
            values.dims(1, 1, nKVHeads * headDim),
            let rotating = cache as? RotatingKVCache,
            rotating.maxSize == LagunaConstants.slidingWindow,
            let ring = rotating.fusedRingPrepare()
        {



            fusedAttended = lagunaSlidingFusedAttention(
                rawQueries: queries,
                rawKeys: keys,
                rawValues: values,
                queryWeight: qNorm.weight,
                keyWeight: kNorm.weight,
                angles: fusedAngles,
                cacheKeys: ring.keys,
                cacheValues: ring.values,
                writeIdx: ring.writeIdx,
                scale: _fusedAttnScale
            )
            rotating.fusedRingAdvance()
            qkNormRoPEFused = true
        } else if lagunaFusedFullAttentionEnabled,
            useFusedFullQKNormYaRN,
            let fusedAngles = qkRoPEAngles,
            values.dtype == .bfloat16,
            values.dims(1, 1, nKVHeads * headDim),
            let simple = cache as? KVCacheSimple,
            let append = simple.fusedAppendPrepare()
        {




            fusedAttended = lagunaFullFusedAttention(
                rawQueries: queries,
                rawKeys: keys,
                rawValues: values,
                queryWeight: qNorm.weight,
                keyWeight: kNorm.weight,
                angles: fusedAngles,
                cacheKeys: append.keys,
                cacheValues: append.values,
                writeIdx: append.writeIdx,
                scale: _fusedAttnScale
            )
            simple.fusedAppendAdvance()
            qkNormRoPEFused = true
        } else if useFusedFullQKNormYaRN, let qkRoPEAngles {
            (queries, keys) = lagunaFullQKNormYaRN(
                rawQueries: queries,
                rawKeys: keys,
                queryWeight: qNorm.weight,
                keyWeight: kNorm.weight,
                angles: qkRoPEAngles
            )
            qkNormRoPEFused = true
        } else if useFusedSlidingQKNormRoPE, let qkRoPEAngles {
            (queries, keys) = lagunaSlidingQKNormRoPE(
                rawQueries: queries,
                rawKeys: keys,
                queryWeight: qNorm.weight,
                keyWeight: kNorm.weight,
                angles: qkRoPEAngles
            )
            qkNormRoPEFused = true
        } else if usePrefillFusedSlidingQKNormRoPE,
            let angles = qkRoPEAngles, let offsets = qkRoPEOffsets
        {
            (queries, keys) = lagunaPrefillSlidingQKNormRoPE(
                rawQueries: queries,
                rawKeys: keys,
                queryWeight: qNorm.weight,
                keyWeight: kNorm.weight,
                angles: angles,
                offsets: offsets,
                length: L
            )
            qkNormRoPEFused = true
        } else if usePrefillFusedFullQKNormYaRN,
            let angles = qkRoPEAngles, let offsets = qkRoPEOffsets
        {
            (queries, keys) = lagunaPrefillFullQKNormYaRN(
                rawQueries: queries,
                rawKeys: keys,
                queryWeight: qNorm.weight,
                keyWeight: kNorm.weight,
                angles: angles,
                offsets: offsets,
                length: L
            )
            qkNormRoPEFused = true
        } else {
            queries =
                qNorm(queries.reshaped(B, L, nHeads, headDim))
                .transposed(0, 2, 1, 3)
            keys =
                kNorm(keys.reshaped(B, L, nKVHeads, headDim))
                .transposed(0, 2, 1, 3)
        }






        if fusedAttended == nil {
            values =
                L == 1
                ? values.reshaped(B, nKVHeads, L, headDim)
                : values.reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)
        }

        if !qkNormRoPEFused {
            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)
        }

        let attended =
            fusedAttended
            ?? attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask
            )




        var output =
            L == 1
            ? attended.reshaped(B, L, -1)
            : attended.transposed(0, 2, 1, 3).reshaped(B, L, -1)

        if gatingEnabled, let gProj {



            let projectedGate: MLXArray
            let gateIsActivated: Bool
            if let fusedNormQKV {
                projectedGate = fusedNormQKV.gateValues
                gateIsActivated = fusedNormQKV.gateActivated
            } else {
                guard let normalizedInput else {
                    preconditionFailure("attention gate requires normalized input")
                }
                projectedGate = gProj(normalizedInput)
                gateIsActivated = false
            }





















            if lagunaUseNativeAffineOProj(layer: layerIdx),
                let affineWO = _nativeAffineOProj,
                gatePerHead, B == 1, L == 1, wo.bias == nil,
                headDim == LagunaConstants.headDim,
                output.dtype == .bfloat16, projectedGate.dtype == .bfloat16,
                output.dims(1, 1, nHeads * headDim),
                projectedGate.dims(1, 1, nHeads)
            {





                if lagunaFusedGatedAffineOProjEnabled, !gateIsActivated,
                    affineWO.mode == .affine, affineWO.bits == 8,
                    affineWO.groupSize == 32,
                    affineWO.indexedMetadata != nil,
                    let affineBiases = affineWO.biases,
                    let fusedProjection = lagunaGatedAffineOProj(
                        attentionOutput: output,
                        gateLogits: projectedGate,
                        codes: affineWO.packedCodes,
                        scales: affineWO.scales,
                        biases: affineBiases,
                        indexedMetadata: affineWO.indexedMetadata,
                        heads: nHeads)
                {
                    return fusedProjection
                }
                if lagunaFusedGatedAffineOProjEnabled, !gateIsActivated,
                    affineWO.mode == .affine, affineWO.bits == 8,
                    affineWO.groupSize == 32,
                    let affineBiases = affineWO.biases,
                    let fusedProjection = lagunaGatedAffineOProj(
                        attentionOutput: output,
                        gateLogits: projectedGate,
                        codes: affineWO.packedCodes,
                        scales: affineWO.scales,
                        biases: affineBiases,
                        heads: nHeads)
                {
                    return fusedProjection
                }
                if lagunaFusedGatedAffineOProjEnabled,
                    lagunaGatedAffineOProjNVFP4Enabled,
                    gateIsActivated,
                    affineWO.mode == .nvfp4, affineWO.bits == 4,
                    affineWO.groupSize == 16,
                    let fusedProjection = lagunaGatedAffineOProjNVFP4(
                        attentionOutput: output,
                        gateLogits: projectedGate,
                        codes: affineWO.packedCodes,
                        scales: affineWO.scales,
                        laneMajorScales: affineWO.laneMajorScales,
                        heads: nHeads,
                        gateIsActivated: true)
                {
                    return fusedProjection
                }
                if lagunaFusedGatedAffineOProjEnabled,
                    lagunaGatedAffineOProjNVFP4Enabled,
                    !gateIsActivated,
                    affineWO.mode == .nvfp4, affineWO.bits == 4,
                    affineWO.groupSize == 16,
                    let fusedProjection = lagunaGatedAffineOProjNVFP4(
                        attentionOutput: output,
                        gateLogits: projectedGate,
                        codes: affineWO.packedCodes,
                        scales: affineWO.scales,
                        laneMajorScales: affineWO.laneMajorScales,
                        heads: nHeads)
                {
                    return fusedProjection
                }





                let gated: MLXArray
                if !gateIsActivated,
                    let fusedGated = lagunaGateProductSoftplus(
                        attentionOutput: output, gateLogits: projectedGate,
                        heads: nHeads)
                {
                    gated = fusedGated
                } else {
                    let gate =
                        gateIsActivated
                        ? projectedGate
                        : lagunaCompiledSoftplusGate(projectedGate)
                    gated =
                        (output.reshaped(B, L, nHeads, headDim)
                            * gate[.ellipsis, .newAxis])
                        .reshaped(B, L, -1)
                }
                lagunaTrace("native affine gated output projection h\(nHeads)")
                return quantizedMM(
                    gated,
                    affineWO.packedCodes,
                    scales: affineWO.scales,
                    biases: affineWO.biases,
                    transpose: true,
                    groupSize: affineWO.groupSize,
                    bits: affineWO.bits,
                    mode: affineWO.mode
                )
            }
            if lagunaFusedGatedOutputProjectionEnabled,
                gateIsActivated, gatePerHead, L == 1, B == 1, wo.bias == nil,
                headDim == LagunaConstants.headDim,
                output.dtype == .bfloat16, projectedGate.dtype == .bfloat16,
                wo.weight.dtype == .bfloat16,
                output.dims(1, 1, nHeads * headDim),
                projectedGate.dims(1, 1, nHeads),
                wo.weight.dims(LagunaConstants.hiddenSize, nHeads * headDim)
            {
                let projection = lagunaGatedOutputProjection(
                    attentionOutput: output,
                    gateValues: projectedGate,
                    weight: wo.weight,
                    heads: nHeads
                )
                if let projection {
                    return projection
                }
            }
            if !gateIsActivated,
                gatePerHead && projectedGate.dtype == output.dtype,
                L == 1, wo.bias == nil, MLXHardwareInfo.isCompiledDecodeSupported
            {
                return attentionGateProjection(output, projectedGate, wo.weight)
            }
            let gate =
                gateIsActivated
                ? projectedGate
                : gatePerHead && projectedGate.dtype == output.dtype
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







    func callLastPrefillRow(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        precondition(L > 1)

        let lastInput = lagunaLastTokenHidden(x)
        var queries: MLXArray
        var keys: MLXArray
        var values: MLXArray
        let bankedGate: MLXArray?
        if lagunaLastPrefillProjectionBanksEnabled,
            let qGateWeight = _lastPrefillQGateWeight,
            let kvWeight = _lastPrefillKVWeight,
            B == 1,
            isSliding, gatingEnabled, gatePerHead,
            lastInput.dtype == .bfloat16,
            x.dtype == .bfloat16,
            lastInput.dims(1, 1, LagunaConstants.hiddenSize),
            x.dims(1, L, LagunaConstants.hiddenSize),
            qGateWeight.dtype == .bfloat16,
            kvWeight.dtype == .bfloat16,
            qGateWeight.dims(nHeads * headDim + nHeads, LagunaConstants.hiddenSize),
            kvWeight.dims(2 * nKVHeads * headDim, LagunaConstants.hiddenSize)
        {
            let qGate = matmul(lastInput, qGateWeight.T)
            let queryDim = nHeads * headDim
            queries = qGate[.ellipsis, 0 ..< queryDim]
            bankedGate = qGate[.ellipsis, queryDim ..< (queryDim + nHeads)]

            let kv = matmul(x, kvWeight.T)
            let kvDim = nKVHeads * headDim
            keys = kv[.ellipsis, 0 ..< kvDim]
            values = kv[.ellipsis, kvDim ..< (2 * kvDim)]
            lagunaTrace("last prefill Q+gate / K+V projection banks")
        } else {
            queries = wq(lastInput)
            keys = wk(x)
            values = wv(x)
            bankedGate = nil
        }

        queries = qNorm(queries.reshaped(B, 1, nHeads, headDim)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, nKVHeads, headDim)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)

        if let offsetArray = graphOffsetArray(for: cache) {
            queries = rope(queries, offset: offsetArray + Int32(L - 1))
        } else {
            queries = rope(queries, offset: (cache?.offset ?? 0) + L - 1)
        }
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let attended = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: .causal
        )


        var output = attended.reshaped(B, 1, -1)

        if gatingEnabled, let gProj {
            let projectedGate = bankedGate ?? gProj(lastInput)
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





































let lagunaNvfp4ScaleFoldEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_NVFP4_SCALE_FOLD"] != "0"












































let lagunaNvfp4NibbleSplit: Int = {
    guard
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_NVFP4_NIBBLE_SPLIT"],
        let value = Int(raw), (0 ... 2).contains(value)
    else { return 1 }
    return value
}()









let lagunaNvfp4ScaleCarry: Bool =
    ProcessInfo.processInfo.environment["DARKBLOOM_NVFP4_SCALE_CARRY"] != "0"


















































let lagunaNvfp4QdotSeedElisionEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_NVFP4_QDOT_SEED_ELIDE"] != "0"































let lagunaNvfp4ScaleDeferEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_NVFP4_SCALE_DEFER"] != "0"
    && lagunaNvfp4ScaleFoldEnabled







let lagunaNvfp4RowScaleSuffix = lagunaNvfp4ScaleDeferEnabled ? " * 4194304.0f" : ""

let lagunaSharedSwiGLUQMVHeader: String = {







    let scaleTail =
        (lagunaNvfp4ScaleFoldEnabled && !lagunaNvfp4ScaleDeferEnabled)
        ? "        return float(signed_value) * 4194304.0f;"
        : "        return float(signed_value);"
    let scale256 = lagunaNvfp4ScaleFoldEnabled ? "" : "        converted *= 256.0;\n"
    let weightScale = lagunaNvfp4ScaleFoldEnabled ? "" : " * 16384.0f"
    let extract: String
    switch lagunaNvfp4NibbleSplit {
    case 1:
        extract = """
        const uint xe = c & 0x0F0F0F0Fu;
        const uint ge = xe | (xe << 3);
        const uint yo = c & 0xF0F0F0F0u;
        const uint go = yo | (yo >> 3);
        const uint p0 = (ge << 9) & 0x8E008E00u;
        const uint p1 = (go << 8) & 0x8E008E00u;
        const uint p2 = (ge << 1) & 0x8E008E00u;
        const uint p3 = go & 0x8E008E00u;
"""
    case 2:
        extract = """
        const uint p0 =
            ((c << 9) & 0x0E000E00u) | ((c << 12) & 0x80008000u);
        const uint p1 =
            ((c << 5) & 0x0E000E00u) | ((c << 8) & 0x80008000u);
        const uint p2 =
            ((c << 1) & 0x0E000E00u) | ((c << 4) & 0x80008000u);
        const uint p3 =
            ((c >> 3) & 0x0E000E00u) | (c & 0x80008000u);
"""
    default:
        extract = """
        const uint p0 =
            ((c & 0x00070007u) << 9) | ((c & 0x00080008u) << 12);
        const uint p1 =
            ((c & 0x00700070u) << 5) | ((c & 0x00800080u) << 8);
        const uint p2 =
            ((c & 0x07000700u) << 1) | ((c & 0x08000800u) << 4);
        const uint p3 =
            ((c & 0x70007000u) >> 3) | (c & 0x80008000u);
"""
    }


    let scaleCarryActive = lagunaNvfp4ScaleCarry && lagunaNvfp4ScaleFoldEnabled
    let scaleRawExpression =
        scaleCarryActive
        ? (lagunaE4M3SignDomainCertified
            ? "ushort(uint(bits) << 7)"
            : "ushort((uint(bits) + (bits & 128u)) << 7)")
        : "ushort(bits & 127) << 7"
    let scaleSignExpression =
        scaleCarryActive
        ? "converted"
        : "(bits & 128) ? -converted : converted"




    let lowScaleFastPath = lagunaNvfp4ScaleDeferEnabled
        ? """
if (bits < 16u) {
    ushort fast_raw = ushort(bits) << 7;
    return float(as_type<half>(fast_raw));
}
"""
        : ""






    func packedWordBody(_ word: Int) -> String {
        let codeWord = word == 0 ? "codes.x" : "codes.y"
        let base = 8 * word
        let seedOperator =
            (word == 0 && lagunaNvfp4QdotSeedElisionEnabled)
            ? "accum =" : "accum +="
        return """
    {
        const uint c = \(codeWord);
\(extract)
        const float2 v04 = float2(as_type<half2>(p0))\(weightScale);
        const float2 v15 = float2(as_type<half2>(p1))\(weightScale);
        const float2 v26 = float2(as_type<half2>(p2))\(weightScale);
        const float2 v37 = float2(as_type<half2>(p3))\(weightScale);
        \(seedOperator)
            (input[\(base)] * v04.x +
             input[\(base + 1)] * v15.x +
             input[\(base + 2)] * v26.x +
             input[\(base + 3)] * v37.x);
        accum +=
            (input[\(base + 4)] * v04.y +
             input[\(base + 5)] * v15.y +
             input[\(base + 6)] * v26.y +
             input[\(base + 7)] * v37.y);
    }
"""
    }
    let accumDeclaration =
        lagunaNvfp4QdotSeedElisionEnabled
        ? "float accum;" : "float accum = 0.0f;"
    return """
static inline float laguna_nvfp4_scale(uint8_t bits) {
\(lowScaleFastPath)
    ushort raw = \(scaleRawExpression);
    half converted = as_type<half>(raw);
\(scale256)    half signed_value = \(scaleSignExpression);
\(scaleTail)
}

static inline float laguna_nvfp4_qdot_codes_16(
    uint2 codes,
    const thread float* input,
    float scale
) {
    \(accumDeclaration)
\(packedWordBody(0))
\(packedWordBody(1))
    return scale * accum;
}

static inline float laguna_nvfp4_qdot_16(
    const device uint8_t* weight,
    const thread float* input,
    float scale
) {
    const device uint2* packed = (const device uint2*)weight;
    return laguna_nvfp4_qdot_codes_16(packed[0], input, scale);
}
"""
}()

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
        bfloat gate = bfloat(gate_result[row]\(lagunaNvfp4RowScaleSuffix));
        bfloat up = bfloat(up_result[row]\(lagunaNvfp4RowScaleSuffix));
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










private func lagunaSharedSwiGLUQMVRows1Source(halved: Bool) -> String {
    let scaleRowBytes = halved ? 64 : 128
    let patch =
        halved
        ? "constexpr uint scale_patch_bytes = \(lagunaScalePatchHeaderBytes);\n" : ""
    let base = halved ? "fused_scales + scale_patch_bytes" : "fused_scales"
    let laneTerm = halved ? "(lane >> 1)" : "lane"
    let blockDiv = halved ? 32 : 16
    func value(_ pointer: String, _ slot: Int) -> String {
        let read = "\(pointer)[block / \(blockDiv)]"
        guard halved else { return read }
        return "(row == 0 && block == 0 && lane == 1)"
            + " ? fused_scales[\(slot)] : \(read)"
    }
    return """
constexpr uint input_width = 2048;
constexpr uint output_width = 512;
constexpr uint packed_row_bytes = 1024;
constexpr uint scale_row_bytes = \(scaleRowBytes);
\(patch)constexpr uint block_width = 512;
constexpr uint values_per_lane = 16;

uint tile = threadgroup_position_in_grid.x;
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint row = tile * 2 + simd_group;

const device uint8_t* gate_row_weight =
    (const device uint8_t*)fused_weight +
    row * packed_row_bytes + lane * 8;
const device uint8_t* up_row_weight =
    (const device uint8_t*)fused_weight +
    (row + output_width) * packed_row_bytes + lane * 8;
const device uint8_t* gate_row_scale =
    \(base) + row * scale_row_bytes + \(laneTerm);
const device uint8_t* up_row_scale =
    \(base) + (row + output_width) * scale_row_bytes + \(laneTerm);

thread float gate_result = 0.0f;
thread float up_result = 0.0f;
thread float input_values[values_per_lane];

for (uint block = 0; block < input_width; block += block_width) {
    const device vec<bfloat, 4>* input_vectors =
        (const device vec<bfloat, 4>*) (
            input + block + lane * values_per_lane);
    for (uint i = 0; i < values_per_lane / 4; ++i) {
        const vec<bfloat, 4> values = input_vectors[i];
        input_values[4 * i] = values[0];
        input_values[4 * i + 1] = values[1];
        input_values[4 * i + 2] = values[2];
        input_values[4 * i + 3] = values[3];
    }

    gate_result += laguna_nvfp4_qdot_16(
        gate_row_weight + block / 2,
        input_values,
        laguna_nvfp4_scale(\(value("gate_row_scale", 0))));
    up_result += laguna_nvfp4_qdot_16(
        up_row_weight + block / 2,
        input_values,
        laguna_nvfp4_scale(\(value("up_row_scale", 1))));
}

gate_result = simd_sum(gate_result);
up_result = simd_sum(up_result);
if (lane == 0) {
    bfloat gate = bfloat(gate_result\(lagunaNvfp4RowScaleSuffix));
    bfloat up = bfloat(up_result\(lagunaNvfp4RowScaleSuffix));
    bfloat exp_abs = metal::exp(metal::abs(gate));
    bfloat denominator = bfloat(1) + exp_abs;
    bfloat y = bfloat(1) / denominator;
    bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
    bfloat silu = bfloat(gate * sigmoid);
    activated[row] = bfloat(silu * up);
}
"""
}

private let lagunaSharedSwiGLUQMVRows1Kernel = MLXFast.metalKernel(
    name: "laguna_shared_nvfp4_swiglu_qmv_rows1_bf16_v1",
    inputNames: ["input", "fused_weight", "fused_scales"],
    outputNames: ["activated"],
    source: lagunaSharedSwiGLUQMVRows1Source(halved: false),
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)

private let lagunaSharedSwiGLUQMVRows1HalvedKernel = MLXFast.metalKernel(
    name: "laguna_shared_nvfp4_swiglu_qmv_rows1_halved_bf16_v1",
    inputNames: ["input", "fused_weight", "fused_scales"],
    outputNames: ["activated"],
    source: lagunaSharedSwiGLUQMVRows1Source(halved: true),
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)







private let lagunaSharedSwiGLUQMVRows1WideKernel = MLXFast.metalKernel(
    name: "laguna_shared_nvfp4_swiglu_qmv_rows1_halved_wide_bf16_v1",
    inputNames: ["input", "fused_weight", "fused_scales"],
    outputNames: ["activated"],
    source: """
constexpr uint input_width = 2048;
constexpr uint output_width = 512;
constexpr uint packed_row_bytes = 1024;
constexpr uint scale_row_bytes = 64;
constexpr uint scale_patch_bytes = \(lagunaScalePatchHeaderBytes);
constexpr uint slab_width = 1024;
constexpr uint values_per_lane = 32;

uint tile = threadgroup_position_in_grid.x;
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint row = tile * 2 + simd_group;

const device uint8_t* gate_row_weight =
    (const device uint8_t*)fused_weight +
    row * packed_row_bytes + lane * 16;
const device uint8_t* up_row_weight =
    (const device uint8_t*)fused_weight +
    (row + output_width) * packed_row_bytes + lane * 16;
const device uint8_t* gate_row_scale =
    fused_scales + scale_patch_bytes + row * scale_row_bytes + lane;
const device uint8_t* up_row_scale =
    fused_scales + scale_patch_bytes
    + (row + output_width) * scale_row_bytes + lane;

thread float gate_result = 0.0f;
thread float up_result = 0.0f;
thread float input_values[values_per_lane];

for (uint slab = 0; slab < input_width; slab += slab_width) {
    const device vec<bfloat, 4>* input_vectors =
        (const device vec<bfloat, 4>*) (
            input + slab + lane * values_per_lane);
    for (uint i = 0; i < values_per_lane / 4; ++i) {
        const vec<bfloat, 4> values = input_vectors[i];
        input_values[4 * i] = values[0];
        input_values[4 * i + 1] = values[1];
        input_values[4 * i + 2] = values[2];
        input_values[4 * i + 3] = values[3];
    }

    // The pair's halved byte covers both groups; the header byte restores the
    // one quantizer exception per plane (gate row 0 / up row 0, pair 0, whose
    // odd member is the second group of lane 0's first pair).
    uint8_t gate_sb = gate_row_scale[slab / 32];
    uint8_t up_sb = up_row_scale[slab / 32];
    bool patch = row == 0 && slab == 0 && lane == 0;
    uint8_t gate_sb_b = patch ? fused_scales[0] : gate_sb;
    uint8_t up_sb_b = patch ? fused_scales[1] : up_sb;

    const uint4 gate_codes =
        *(const device uint4*)(gate_row_weight + slab / 2);
    const uint4 up_codes =
        *(const device uint4*)(up_row_weight + slab / 2);

    gate_result += laguna_nvfp4_qdot_codes_16(
        gate_codes.xy, input_values, laguna_nvfp4_scale(gate_sb));
    gate_result += laguna_nvfp4_qdot_codes_16(
        gate_codes.zw, input_values + 16, laguna_nvfp4_scale(gate_sb_b));
    up_result += laguna_nvfp4_qdot_codes_16(
        up_codes.xy, input_values, laguna_nvfp4_scale(up_sb));
    up_result += laguna_nvfp4_qdot_codes_16(
        up_codes.zw, input_values + 16, laguna_nvfp4_scale(up_sb_b));
}

gate_result = simd_sum(gate_result);
up_result = simd_sum(up_result);
if (lane == 0) {
    bfloat gate = bfloat(gate_result\(lagunaNvfp4RowScaleSuffix));
    bfloat up = bfloat(up_result\(lagunaNvfp4RowScaleSuffix));
    bfloat exp_abs = metal::exp(metal::abs(gate));
    bfloat denominator = bfloat(1) + exp_abs;
    bfloat y = bfloat(1) / denominator;
    bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
    bfloat silu = bfloat(gate * sigmoid);
    activated[row] = bfloat(silu * up);
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
    precondition(input.dims(1, 1, LagunaConstants.hiddenSize))
    precondition(fusedWeight.dtype == .uint32)
    precondition(
        fusedWeight.dims(2 * LagunaConstants.sharedExpertIntermediateSize,
            LagunaConstants.hiddenSize / 8))
    precondition(fusedScales.dtype == .uint8)



    let halved = fusedScales.ndim == 1
    if halved {
        precondition(lagunaSharedSwiGLUQMVRows1Enabled)
        precondition(
            fusedScales.size == lagunaScalePatchHeaderBytes
                + 2 * LagunaConstants.sharedExpertIntermediateSize
                * (LagunaConstants.hiddenSize / 32))
    } else {
        precondition(
            fusedScales.dims(2 * LagunaConstants.sharedExpertIntermediateSize,
                LagunaConstants.hiddenSize / 16))
    }

    let kernel =
        halved
        ? (lagunaSharedQMVWideCodesEnabled
            ? lagunaSharedSwiGLUQMVRows1WideKernel
            : lagunaSharedSwiGLUQMVRows1HalvedKernel)
        : (lagunaSharedSwiGLUQMVRows1Enabled
            ? lagunaSharedSwiGLUQMVRows1Kernel
            : lagunaSharedSwiGLUQMVKernel)
    let tiles = lagunaSharedSwiGLUQMVRows1Enabled ? 256 : 128
    return kernel(
        [input, fusedWeight, fusedScales],
        grid: (tiles * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.sharedExpertIntermediateSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

private let lagunaSharedDownResidualKernel = MLXFast.metalKernel(
    name: "laguna_shared_nvfp4_down_residual_bf16_v1",
    inputNames: [
        "activated", "down_weight", "down_scales", "routed", "residual",
    ],
    outputNames: ["output"],
    source: lagunaSharedDownResidualSource(halved: false),
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)

private let lagunaSharedDownResidualHalvedKernel = MLXFast.metalKernel(
    name: "laguna_shared_nvfp4_down_residual_halved_bf16_v1",
    inputNames: [
        "activated", "down_weight", "down_scales", "routed", "residual",
    ],
    outputNames: ["output"],
    source: lagunaSharedDownResidualSource(halved: true),
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)




private func lagunaSharedDownResidualSource(halved: Bool) -> String {
    let scaleRowBytes = halved ? 16 : 32
    let patch =
        halved
        ? "constexpr uint scale_patch_bytes = \(lagunaScalePatchHeaderBytes);\n" : ""
    let base = halved ? "down_scales + scale_patch_bytes" : "down_scales"
    let laneTerm = halved ? "(lane >> 1)" : "lane"
    let read =
        halved
        ? "(output_row == 0 && lane == 1) ? down_scales[0] : scale[0]" : "scale[0]"
    return """
constexpr uint input_width = 512;
constexpr uint output_width = 2048;
constexpr uint outputs_per_simd = 4;
constexpr uint values_per_lane = 16;
constexpr uint packed_row_bytes = 256;
constexpr uint scale_row_bytes = \(scaleRowBytes);
\(patch)
uint group = threadgroup_position_in_grid.x;
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint first_row =
    group * 2 * outputs_per_simd +
    simd_group * outputs_per_simd;

thread float input_values[values_per_lane];
const device vec<bfloat, 4>* input_vectors =
    (const device vec<bfloat, 4>*)(
        activated + lane * values_per_lane);
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
        (const device uint8_t*)down_weight +
        output_row * packed_row_bytes + lane * 8;
    const device uint8_t* scale =
        \(base) + output_row * scale_row_bytes + \(laneTerm);
    result[row] = laguna_nvfp4_qdot_16(
        weight,
        input_values,
        laguna_nvfp4_scale(\(read)));
    result[row] = simd_sum(result[row]);
}

if (lane == 0) {
    for (uint row = 0; row < outputs_per_simd; ++row) {
        uint output_row = first_row + row;
        bfloat shared = bfloat(result[row]\(lagunaNvfp4RowScaleSuffix));
        bfloat r2 = bfloat(routed[output_row] + shared);
        output[output_row] =
            bfloat(residual[output_row] + r2);
    }
}
"""
}

func lagunaSharedDownResidual(
    _ activated: MLXArray,
    downWeight: MLXArray,
    downScales: MLXArray,
    routed: MLXArray,
    residual: MLXArray
) -> MLXArray {
    precondition(activated.dtype == .bfloat16)
    precondition(
        activated.dims(1, 1, LagunaConstants.sharedExpertIntermediateSize))
    precondition(downWeight.dtype == .uint32)
    precondition(
        downWeight.dims(LagunaConstants.hiddenSize,
            LagunaConstants.sharedExpertIntermediateSize / 8))
    precondition(downScales.dtype == .uint8)
    let halved = downScales.ndim == 1
    if halved {
        precondition(
            downScales.size == lagunaScalePatchHeaderBytes
                + LagunaConstants.hiddenSize
                * (LagunaConstants.sharedExpertIntermediateSize / 32))
    } else {
        precondition(
            downScales.dims(LagunaConstants.hiddenSize,
                LagunaConstants.sharedExpertIntermediateSize / 16))
    }
    precondition(routed.dtype == .bfloat16)
    precondition(routed.dims(1, 1, LagunaConstants.hiddenSize))
    precondition(residual.dtype == .bfloat16)
    precondition(residual.dims(1, 1, LagunaConstants.hiddenSize))

    let kernel =
        halved
        ? lagunaSharedDownResidualHalvedKernel : lagunaSharedDownResidualKernel
    return kernel(
        [activated, downWeight, downScales, routed, residual],
        grid: ((LagunaConstants.hiddenSize / 8) * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.hiddenSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

private let lagunaRoutedSwiGLUQMVKernel = MLXFast.metalKernel(
    name: "laguna_routed_nvfp4_swiglu_qmv_bf16_v2",
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
constexpr uint routed_experts = 8;

uint group = threadgroup_position_in_grid.x;
uint expert_slot = group % routed_experts;
uint tile = group / routed_experts;
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
        uint logical_row = first_row + row;
        uint pair_tile = logical_row / 32;
        uint gate_row = pair_tile * 64 + logical_row % 32;
        uint up_row = gate_row + 32;
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
        bfloat gate = bfloat(gate_result[row]\(lagunaNvfp4RowScaleSuffix));
        bfloat up = bfloat(up_result[row]\(lagunaNvfp4RowScaleSuffix));
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




private let lagunaRoutedSwiGLUQMVRows1Kernel = MLXFast.metalKernel(
    name: "laguna_routed_nvfp4_swiglu_qmv_rows1_bf16_v1",
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
constexpr uint tiles_per_expert = 256;
constexpr uint routed_experts = 8;

uint group = threadgroup_position_in_grid.x;
uint expert_slot = group % routed_experts;
uint tile = group / routed_experts;
uint expert = uint(indices[expert_slot]);
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint logical_row = tile * 2 + simd_group;

const device uint8_t* expert_weight =
    (const device uint8_t*)fused_weight +
    expert * packed_expert_bytes;
const device uint8_t* expert_scales =
    fused_scales + expert * scale_expert_bytes;

uint pair_tile = logical_row / 32;
uint gate_row = pair_tile * 64 + logical_row % 32;
uint up_row = gate_row + 32;
const device uint8_t* gate_row_weight =
    expert_weight + gate_row * packed_row_bytes + lane * 8;
const device uint8_t* up_row_weight =
    expert_weight + up_row * packed_row_bytes + lane * 8;
const device uint8_t* gate_row_scale =
    expert_scales + gate_row * scale_row_bytes + lane;
const device uint8_t* up_row_scale =
    expert_scales + up_row * scale_row_bytes + lane;

thread float gate_result = 0.0f;
thread float up_result = 0.0f;
thread float input_values[values_per_lane];

for (uint block = 0; block < input_width; block += block_width) {
    const device vec<bfloat, 4>* input_vectors =
        (const device vec<bfloat, 4>*) (
            input + block + lane * values_per_lane);
    for (uint i = 0; i < values_per_lane / 4; ++i) {
        const vec<bfloat, 4> values = input_vectors[i];
        input_values[4 * i] = values[0];
        input_values[4 * i + 1] = values[1];
        input_values[4 * i + 2] = values[2];
        input_values[4 * i + 3] = values[3];
    }

    const device uint8_t* gate_weight =
        gate_row_weight + block / 2;
    const device uint8_t* up_weight =
        up_row_weight + block / 2;
    const device uint8_t* gate_scale =
        gate_row_scale + block / 16;
    const device uint8_t* up_scale =
        up_row_scale + block / 16;

    gate_result += laguna_nvfp4_qdot_16(
        gate_weight,
        input_values,
        laguna_nvfp4_scale(gate_scale[0]));
    up_result += laguna_nvfp4_qdot_16(
        up_weight,
        input_values,
        laguna_nvfp4_scale(up_scale[0]));
}

gate_result = simd_sum(gate_result);
up_result = simd_sum(up_result);
if (lane == 0) {
    bfloat gate = bfloat(gate_result\(lagunaNvfp4RowScaleSuffix));
    bfloat up = bfloat(up_result\(lagunaNvfp4RowScaleSuffix));
    bfloat exp_abs = metal::exp(metal::abs(gate));
    bfloat denominator = bfloat(1) + exp_abs;
    bfloat y = bfloat(1) / denominator;
    bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
    bfloat silu = bfloat(gate * sigmoid);
    activated[expert_slot * output_width + logical_row] =
        bfloat(silu * up);
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
    precondition(input.dims(1, 1, LagunaConstants.hiddenSize))
    precondition(fusedWeight.dtype == .uint32)
    precondition(
        fusedWeight.dims(LagunaConstants.numExperts, 2 * LagunaConstants.moeIntermediateSize,
            LagunaConstants.hiddenSize / 8))
    precondition(fusedScales.dtype == .uint8)
    precondition(
        fusedScales.dims(LagunaConstants.numExperts, 2 * LagunaConstants.moeIntermediateSize,
            LagunaConstants.hiddenSize / 16))
    precondition(indices.dtype == .uint32)
    precondition(indices.dims(1, 1, LagunaConstants.numExpertsPerTok))

    let kernel =
        lagunaSwiGLUQMVRows1Enabled
        ? lagunaRoutedSwiGLUQMVRows1Kernel
        : lagunaRoutedSwiGLUQMVKernel
    let tilesPerSlot = lagunaSwiGLUQMVRows1Enabled ? 256 : 128
    return kernel(
        [input, fusedWeight, fusedScales, indices],
        grid: (LagunaConstants.numExpertsPerTok * tilesPerSlot * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[
            1, 1, LagunaConstants.numExpertsPerTok, 1,
            LagunaConstants.moeIntermediateSize,
        ]],
        outputDTypes: [.bfloat16]
    )[0]
}











private let lagunaRoutedSwiGLUQMVPackedKernel = MLXFast.metalKernel(
    name: "laguna_routed_nvfp4_swiglu_qmv_packed_bf16_v1",
    inputNames: ["input", "fused_weight", "packed_scales", "indices"],
    outputNames: ["activated"],
    source: """
constexpr uint input_width = 2048;
constexpr uint output_width = 512;
constexpr uint block_width = 512;
constexpr uint values_per_lane = 16;
constexpr uint routed_experts = 8;
constexpr uint fused_row_bytes = 1024;
constexpr uint fused_expert_bytes = 1024 * fused_row_bytes;
constexpr uint scale_patch_bytes = \(lagunaScalePatchHeaderBytes);
constexpr uint scale_row_bytes = 16;
constexpr uint scale_sub_bytes = 8 * scale_row_bytes;
constexpr uint scale_kblock_bytes = scale_sub_bytes;
constexpr uint scale_tile_bytes = 4 * scale_kblock_bytes;
constexpr uint packed_expert_bytes = 128 * scale_tile_bytes;

uint group = threadgroup_position_in_grid.x;
uint expert_slot = group % routed_experts;
uint tile = group / routed_experts;
uint expert = uint(indices[expert_slot]);
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint first_row = tile * 4 + simd_group * 2;

const device uint8_t* expert_weight =
    (const device uint8_t*)fused_weight +
    expert * fused_expert_bytes;
const device uint8_t* tile_scales =
    packed_scales + scale_patch_bytes
    + expert * packed_expert_bytes
    + tile * scale_tile_bytes;

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

    const device uint8_t* block_scales =
        tile_scales + (block / block_width) * scale_kblock_bytes;
    for (uint row = 0; row < 2; ++row) {
        uint logical_row = tile * 4 + simd_group * 2 + row;
        uint gate_row = (logical_row / 32) * 64 + logical_row % 32;
        uint up_row = gate_row + 32;
        uint sub = simd_group * 2 + row;
        const device uint8_t* gate_scale =
            block_scales + sub * 2 * scale_row_bytes + (lane >> 1);
        const device uint8_t* up_scale =
            gate_scale + scale_row_bytes;
        bool patch_lane = expert == 0 && logical_row == 0
            && block == 0 && lane == 1;
        uint8_t gate_sb =
            patch_lane ? packed_scales[0] : gate_scale[0];
        uint8_t up_sb = patch_lane ? packed_scales[1] : up_scale[0];
        const device uint8_t* gate_weight =
            expert_weight + gate_row * fused_row_bytes
            + block / 2 + lane * 8;
        const device uint8_t* up_weight =
            expert_weight + up_row * fused_row_bytes
            + block / 2 + lane * 8;

        gate_result[row] += laguna_nvfp4_qdot_16(
            gate_weight,
            input_values,
            laguna_nvfp4_scale(gate_sb));
        up_result[row] += laguna_nvfp4_qdot_16(
            up_weight,
            input_values,
            laguna_nvfp4_scale(up_sb));
    }
}

for (uint row = 0; row < 2; ++row) {
    gate_result[row] = simd_sum(gate_result[row]);
    up_result[row] = simd_sum(up_result[row]);
    if (lane == 0) {
        bfloat gate = bfloat(gate_result[row]\(lagunaNvfp4RowScaleSuffix));
        bfloat up = bfloat(up_result[row]\(lagunaNvfp4RowScaleSuffix));
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

func lagunaRoutedSwiGLUQMVPacked(
    _ input: MLXArray,
    fusedWeight: MLXArray,
    packedScales: MLXArray,
    indices: MLXArray
) -> MLXArray {
    precondition(input.dtype == .bfloat16)
    precondition(input.dims(1, 1, LagunaConstants.hiddenSize))
    precondition(fusedWeight.dtype == .uint32)
    precondition(
        fusedWeight.dims(LagunaConstants.numExperts, 2 * LagunaConstants.moeIntermediateSize,
            LagunaConstants.hiddenSize / 8))
    precondition(packedScales.dtype == .uint8)
    precondition(packedScales.size == lagunaPackedRoutedGateUpScaleBytes)
    precondition(indices.dtype == .uint32)
    precondition(indices.dims(1, 1, LagunaConstants.numExpertsPerTok))

    return lagunaRoutedSwiGLUQMVPackedKernel(
        [input, fusedWeight, packedScales, indices],
        grid: (LagunaConstants.numExpertsPerTok * 128 * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[
            1, 1, LagunaConstants.numExpertsPerTok, 1,
            LagunaConstants.moeIntermediateSize,
        ]],
        outputDTypes: [.bfloat16]
    )[0]
}




func lagunaRoutedSwiGLUQMVPackedSelectedSource(
    prologue: String, expertExpression: String
) -> String {
    """
constexpr uint input_width = 2048;
constexpr uint output_width = 512;
constexpr uint block_width = 512;
constexpr uint values_per_lane = 16;
constexpr uint routed_experts = 8;
constexpr uint fused_row_bytes = 1024;
constexpr uint fused_expert_bytes = 1024 * fused_row_bytes;
constexpr uint scale_patch_bytes = \(lagunaScalePatchHeaderBytes);
constexpr uint scale_row_bytes = 16;
constexpr uint scale_sub_bytes = 8 * scale_row_bytes;
constexpr uint scale_kblock_bytes = scale_sub_bytes;
constexpr uint scale_tile_bytes = 4 * scale_kblock_bytes;
constexpr uint packed_expert_bytes = 128 * scale_tile_bytes;

uint group = threadgroup_position_in_grid.x;
uint expert_slot = group % routed_experts;
uint tile = group / routed_experts;
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint first_row = tile * 4 + simd_group * 2;
\(prologue)
uint expert = \(expertExpression);

const device uint8_t* expert_weight =
    (const device uint8_t*)fused_weight + expert * fused_expert_bytes;
const device uint8_t* tile_scales =
    packed_scales + scale_patch_bytes + expert * packed_expert_bytes
    + tile * scale_tile_bytes;

thread float gate_result[2] = {0.0f, 0.0f};
thread float up_result[2] = {0.0f, 0.0f};
thread float input_values[values_per_lane];

for (uint block = 0; block < input_width; block += block_width) {
    const device vec<bfloat, 4>* input_vectors =
        (const device vec<bfloat, 4>*) (
            input + block + lane * values_per_lane);
    for (uint i = 0; i < values_per_lane / 4; ++i) {
        const vec<bfloat, 4> values = input_vectors[i];
        input_values[4 * i] = values[0];
        input_values[4 * i + 1] = values[1];
        input_values[4 * i + 2] = values[2];
        input_values[4 * i + 3] = values[3];
    }

    const device uint8_t* block_scales =
        tile_scales + (block / block_width) * scale_kblock_bytes;
    for (uint row = 0; row < 2; ++row) {
        uint logical_row = tile * 4 + simd_group * 2 + row;
        uint gate_row = (logical_row / 32) * 64 + logical_row % 32;
        uint up_row = gate_row + 32;
        uint sub = simd_group * 2 + row;
        const device uint8_t* gate_scale =
            block_scales + sub * 2 * scale_row_bytes + (lane >> 1);
        const device uint8_t* up_scale = gate_scale + scale_row_bytes;
        const device uint8_t* gate_weight =
            expert_weight + gate_row * fused_row_bytes
            + block / 2 + lane * 8;
        const device uint8_t* up_weight =
            expert_weight + up_row * fused_row_bytes
            + block / 2 + lane * 8;

        bool patch_lane =
            expert == 0 && logical_row == 0 && block == 0 && lane == 1;
        uint8_t gate_sb = patch_lane ? packed_scales[0] : gate_scale[0];
        uint8_t up_sb = patch_lane ? packed_scales[1] : up_scale[0];

        gate_result[row] += laguna_nvfp4_qdot_16(
            gate_weight, input_values,
            laguna_nvfp4_scale(gate_sb));
        up_result[row] += laguna_nvfp4_qdot_16(
            up_weight, input_values,
            laguna_nvfp4_scale(up_sb));
    }
}

for (uint row = 0; row < 2; ++row) {
    gate_result[row] = simd_sum(gate_result[row]);
    up_result[row] = simd_sum(up_result[row]);
    if (lane == 0) {
        bfloat gate = bfloat(gate_result[row]\(lagunaNvfp4RowScaleSuffix));
        bfloat up = bfloat(up_result[row]\(lagunaNvfp4RowScaleSuffix));
        bfloat exp_abs = metal::exp(metal::abs(gate));
        bfloat denominator = bfloat(1) + exp_abs;
        bfloat y = bfloat(1) / denominator;
        bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
        bfloat silu = bfloat(gate * sigmoid);
        activated[expert_slot * output_width + first_row + row] =
            bfloat(silu * up);
    }
}
"""
}




let lagunaRouterTop8PrologueHeader = """
METAL_FUNC uint laguna_router_top8_extract_round(
    thread const uint* keys, thread uint& mask, uint lane) {
    uint best_ordinal = 0xFFFFFFFFu;
    uint best_index = 256u;
    for (uint j = 0; j < 8; ++j) {
        if ((mask & (1u << j)) != 0u) continue;
        uint e = lane + 32u * j;
        uint o = keys[j];
        if (laguna_router_ordinal_before(o, e, best_ordinal, best_index)) {
            best_ordinal = o;
            best_index = e;
        }
    }
    // Transport the comparator's (ordinal, expert-index) state as one uint2
    // through each butterfly step. simd_shuffle_xor moves both components
    // bit-for-bit from the same source lane; comparator order is unchanged.
    uint2 best_pair = uint2(best_ordinal, best_index);
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        const uint2 other_pair = simd_shuffle_xor(best_pair, offset);
        if (laguna_router_ordinal_before(
            other_pair.x, other_pair.y, best_pair.x, best_pair.y)) {
            best_pair = other_pair;
        }
    }
    best_index = best_pair.y;
    if ((best_index & 31u) == lane) {
        mask |= 1u << (best_index >> 5u);
    }
    return best_index;
}
"""

private let lagunaRouterTop8PrecomputedPrelude = """
thread uint top8_keys[8];
    for (uint j = 0; j < 8; ++j) {
        top8_keys[j] = router_keys[lane + 32u * j];
    }
    uint top8_mask = 0u;
    uint top8_winner = 0u;
    for (uint r = 0; r <= expert_slot; ++r) {
        top8_winner = laguna_router_top8_extract_round(
            top8_keys, top8_mask, lane);
    }
"""

private let lagunaRoutedSwiGLUQMVPackedTop8Kernel = MLXFast.metalKernel(
    name: "laguna_routed_nvfp4_swiglu_qmv_packed_top8keys_bf16_v1",
    inputNames: ["input", "fused_weight", "packed_scales", "router_keys"],
    outputNames: ["activated"],
    source: lagunaRoutedSwiGLUQMVPackedSelectedSource(
        prologue: lagunaRouterTop8PrecomputedPrelude,
        expertExpression: "top8_winner"),
    header: lagunaSharedSwiGLUQMVHeader + "\n" + lagunaDecodeRouterOrdinalHeader
        + "\n" + lagunaRouterTop8PrologueHeader,
    ensureRowContiguous: true
)









let lagunaRoutedGateUpR1Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ROUTED_GATEUP_R1"] != "0"

private let lagunaRoutedSwiGLUQMVPackedTop8R1Kernel = MLXFast.metalKernel(
    name: "laguna_routed_nvfp4_swiglu_qmv_packed_top8keys_r1_bf16_v2",
    inputNames: ["input", "fused_weight", "packed_scales", "router_keys"],
    outputNames: ["activated"],
    source: """
constexpr uint input_width = 2048;
constexpr uint output_width = 512;
constexpr uint block_width = 512;
constexpr uint values_per_lane = 16;
constexpr uint routed_experts = 8;
constexpr uint fused_row_bytes = 1024;
constexpr uint fused_expert_bytes = 1024 * fused_row_bytes;
constexpr uint scale_patch_bytes = \(lagunaScalePatchHeaderBytes);
constexpr uint scale_row_bytes = 16;
constexpr uint scale_sub_bytes = 8 * scale_row_bytes;
constexpr uint scale_kblock_bytes = scale_sub_bytes;
constexpr uint scale_tile_bytes = 4 * scale_kblock_bytes;
constexpr uint packed_expert_bytes = 128 * scale_tile_bytes;

uint group = threadgroup_position_in_grid.x;
uint expert_slot = group % routed_experts;
uint tile = group / routed_experts;
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint logical_row = tile * 2 + simd_group;
\(lagunaRouterTop8PrecomputedPrelude)
uint expert = top8_winner;

const device uint8_t* expert_weight =
    (const device uint8_t*)fused_weight + expert * fused_expert_bytes;
const device uint8_t* row_scales =
    packed_scales + scale_patch_bytes + expert * packed_expert_bytes
    + (logical_row / 4) * scale_tile_bytes;
uint sub = logical_row % 4;
uint gate_row = (logical_row / 32) * 64 + logical_row % 32;
uint up_row = gate_row + 32;

thread float gate_result = 0.0f;
thread float up_result = 0.0f;
thread float input_values[values_per_lane];

uint2 gate_codes;
uint2 up_codes;
uint8_t gate_sb;
uint8_t up_sb;
{
    const device uint8_t* first_scales =
        row_scales + sub * 2 * scale_row_bytes + (lane >> 1);
    bool patch_lane = expert == 0 && logical_row == 0 && lane == 1;
    gate_sb = patch_lane ? packed_scales[0] : first_scales[0];
    up_sb = patch_lane ? packed_scales[1] : first_scales[scale_row_bytes];
    gate_codes = *(const device uint2*)(
        expert_weight + gate_row * fused_row_bytes + lane * 8);
    up_codes = *(const device uint2*)(
        expert_weight + up_row * fused_row_bytes + lane * 8);
}

for (uint block = 0; block < input_width; block += block_width) {
    const device vec<bfloat, 4>* input_vectors =
        (const device vec<bfloat, 4>*) (
            input + block + lane * values_per_lane);
    for (uint i = 0; i < values_per_lane / 4; ++i) {
        const vec<bfloat, 4> values = input_vectors[i];
        input_values[4 * i] = values[0];
        input_values[4 * i + 1] = values[1];
        input_values[4 * i + 2] = values[2];
        input_values[4 * i + 3] = values[3];
    }

    const uint2 cur_gate_codes = gate_codes;
    const uint2 cur_up_codes = up_codes;
    const uint8_t cur_gate_sb = gate_sb;
    const uint8_t cur_up_sb = up_sb;
    const uint next_block = block + block_width;
    if (next_block < input_width) {
        const device uint8_t* next_scales =
            row_scales + (next_block / block_width) * scale_kblock_bytes
            + sub * 2 * scale_row_bytes + (lane >> 1);
        gate_sb = next_scales[0];
        up_sb = next_scales[scale_row_bytes];
        gate_codes = *(const device uint2*)(
            expert_weight + gate_row * fused_row_bytes
            + next_block / 2 + lane * 8);
        up_codes = *(const device uint2*)(
            expert_weight + up_row * fused_row_bytes
            + next_block / 2 + lane * 8);
    }

    gate_result += laguna_nvfp4_qdot_codes_16(
        cur_gate_codes, input_values,
        laguna_nvfp4_scale(cur_gate_sb));
    up_result += laguna_nvfp4_qdot_codes_16(
        cur_up_codes, input_values,
        laguna_nvfp4_scale(cur_up_sb));
}

gate_result = simd_sum(gate_result);
up_result = simd_sum(up_result);
if (lane == 0) {
    bfloat gate = bfloat(gate_result\(lagunaNvfp4RowScaleSuffix));
    bfloat up = bfloat(up_result\(lagunaNvfp4RowScaleSuffix));
    bfloat exp_abs = metal::exp(metal::abs(gate));
    bfloat denominator = bfloat(1) + exp_abs;
    bfloat y = bfloat(1) / denominator;
    bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
    bfloat silu = bfloat(gate * sigmoid);
    activated[expert_slot * output_width + logical_row] =
        bfloat(silu * up);
}
""",
    header: lagunaSharedSwiGLUQMVHeader + "\n" + lagunaDecodeRouterOrdinalHeader
        + "\n" + lagunaRouterTop8PrologueHeader,
    ensureRowContiguous: true
)

func lagunaRoutedSwiGLUQMVPackedTop8(
    _ input: MLXArray,
    fusedWeight: MLXArray,
    packedScales: MLXArray,
    routerKeys: MLXArray
) -> MLXArray {
    precondition(input.dtype == .bfloat16)
    precondition(input.dims(1, 1, LagunaConstants.hiddenSize))
    precondition(fusedWeight.dtype == .uint32)
    precondition(packedScales.dtype == .uint8)
    precondition(packedScales.size == lagunaPackedRoutedGateUpScaleBytes)
    precondition(routerKeys.dtype == .uint32)
    precondition(routerKeys.size == LagunaConstants.numExperts)

    if lagunaRoutedGateUpR1Enabled {
        return lagunaRoutedSwiGLUQMVPackedTop8R1Kernel(
            [input, fusedWeight, packedScales, routerKeys],
            grid: (LagunaConstants.numExpertsPerTok * 256 * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[
                1, 1, LagunaConstants.numExpertsPerTok, 1,
                LagunaConstants.moeIntermediateSize,
            ]],
            outputDTypes: [.bfloat16]
        )[0]
    }
    return lagunaRoutedSwiGLUQMVPackedTop8Kernel(
        [input, fusedWeight, packedScales, routerKeys],
        grid: (LagunaConstants.numExpertsPerTok * 128 * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[
            1, 1, LagunaConstants.numExpertsPerTok, 1,
            LagunaConstants.moeIntermediateSize,
        ]],
        outputDTypes: [.bfloat16]
    )[0]
}

private let lagunaRoutedDownReduceKernel = MLXFast.metalKernel(
    name: "laguna_routed_nvfp4_down_reduce_bf16_v2",
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
constexpr uint scale_patch_bytes = \(lagunaScalePatchHeaderBytes);
constexpr uint scale_row_bytes = 16;
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
    down_scales + scale_patch_bytes + expert * scale_expert_bytes;

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
uint2 row_codes[outputs_per_simd];
uint8_t row_sb[outputs_per_simd];
for (uint row = 0; row < outputs_per_simd; ++row) {
    uint output_row = first_row + row;
    row_codes[row] = *(const device uint2*)(
        expert_weight + output_row * packed_row_bytes + lane * 8);
    row_sb[row] =
        (expert == 0 && output_row == 0 && lane == 1)
        ? down_scales[0]
        : expert_scales[output_row * scale_row_bytes + (lane >> 1)];
}
for (uint row = 0; row < outputs_per_simd; ++row) {
    result[row] = laguna_nvfp4_qdot_codes_16(
        row_codes[row],
        input_values,
        laguna_nvfp4_scale(row_sb[row]));
}
{
    const vec<float, 4> packed_rows = simd_sum(
        vec<float, 4>(result[0], result[1], result[2], result[3]));
    result[0] = packed_rows.x;
    result[1] = packed_rows.y;
    result[2] = packed_rows.z;
    result[3] = packed_rows.w;
}

threadgroup bfloat expert_outputs[
    experts_per_token * outputs_per_simd
];
if (lane == 0) {
    for (uint row = 0; row < outputs_per_simd; ++row) {
        expert_outputs[
            expert_slot * outputs_per_simd + row
        ] = bfloat(result[row]\(lagunaNvfp4RowScaleSuffix));
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

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
        activated.dims(1, 1, LagunaConstants.numExpertsPerTok, 1,
            LagunaConstants.moeIntermediateSize))
    precondition(downWeight.dtype == .uint32)
    precondition(
        downWeight.dims(LagunaConstants.numExperts, LagunaConstants.hiddenSize,
            LagunaConstants.moeIntermediateSize / 8))
    precondition(downScales.dtype == .uint8)
    precondition(downScales.size == lagunaRoutedDownScaleBytes)
    precondition(indices.dtype == .uint32)
    precondition(indices.dims(1, 1, LagunaConstants.numExpertsPerTok))
    precondition(routerWeights.dtype == .float32)
    precondition(routerWeights.dims(1, 1, LagunaConstants.numExpertsPerTok))

    return lagunaRoutedDownReduceKernel(
        [activated, downWeight, downScales, indices, routerWeights],
        grid: ((LagunaConstants.hiddenSize / 4) * 256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.hiddenSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

















let lagunaSharedFirstDownOrderEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_SHARED_FIRST_DOWN"] == "1"






let lagunaFusedDownRowStagingEnabled =
    ProcessInfo.processInfo.environment[
        "DARKBLOOM_FUSED_DOWN_ROW_STAGING"] != "0"

private let lagunaRoutedSharedDownResidualKernel = MLXFast.metalKernel(
    name: lagunaSharedFirstDownOrderEnabled
        ? "laguna_routed_shared_nvfp4_down_residual_bf16_r1_v5sf"
        : "laguna_routed_shared_nvfp4_down_residual_bf16_r1_v5",
    inputNames: lagunaSharedFirstDownOrderEnabled
        ? [
            "shared_activated", "shared_down_weight", "shared_down_scales",
            "routed_activated", "routed_down_weight", "routed_down_scales",
            "indices", "router_weights", "residual",
        ]
        : [
            "routed_activated", "routed_down_weight", "routed_down_scales",
            "indices", "router_weights", "shared_activated",
            "shared_down_weight", "shared_down_scales", "residual",
        ],
    outputNames: ["output"],
    source: lagunaRoutedSharedDownResidualSource(sharedHalved: false),
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)





private let lagunaRoutedSharedDownResidualSharedHalvedKernel =
    MLXFast.metalKernel(
        name: lagunaSharedFirstDownOrderEnabled
            ? "laguna_routed_shared_nvfp4_down_residual_bf16_r1_sh_v5sf"
            : "laguna_routed_shared_nvfp4_down_residual_bf16_r1_sh_v5",
        inputNames: lagunaSharedFirstDownOrderEnabled
            ? [
                "shared_activated", "shared_down_weight", "shared_down_scales",
                "routed_activated", "routed_down_weight", "routed_down_scales",
                "indices", "router_weights", "residual",
            ]
            : [
                "routed_activated", "routed_down_weight", "routed_down_scales",
                "indices", "router_weights", "shared_activated",
                "shared_down_weight", "shared_down_scales", "residual",
            ],
        outputNames: ["output"],
        source: lagunaRoutedSharedDownResidualSource(sharedHalved: true),
        header: lagunaSharedSwiGLUQMVHeader,
        ensureRowContiguous: true
    )

private func lagunaRoutedSharedDownResidualSource(
    sharedHalved: Bool, staged: Bool = false
) -> String {
    let sharedRowBytes = sharedHalved ? 16 : 32
    let sharedBase =
        sharedHalved ? "shared_down_scales + scale_patch_bytes" : "shared_down_scales"
    let scaleLane = sharedHalved ? "(lane >> 1)" : "is_shared ? lane : (lane >> 1)"
    let patch =
        sharedHalved
        ? "(output_row == 0 && lane == 1 && (is_shared || expert == 0))\n"
            + "        ? (is_shared ? shared_down_scales[0] : routed_down_scales[0])"
        : "(!is_shared && expert == 0 && output_row == 0 && lane == 1)\n"
            + "        ? routed_down_scales[0]"



    let qdots =
        staged
        ? """
        thread float result[outputs_per_simd] = {0.0f};
        uint2 row_codes[outputs_per_simd];
        uint8_t row_sb[outputs_per_simd];
        for (uint row = 0; row < outputs_per_simd; ++row) {
            uint output_row = first_row + row;
            row_codes[row] = *(const device uint2*)(
                expert_weight + output_row * packed_row_bytes + lane * 8);
            const device uint8_t* scale =
                expert_scales + output_row * scale_row_bytes + scale_lane;
            row_sb[row] =
                \(patch)
                : scale[0];
        }
        for (uint row = 0; row < outputs_per_simd; ++row) {
            result[row] = laguna_nvfp4_qdot_codes_16(
                row_codes[row],
                input_values,
                laguna_nvfp4_scale(row_sb[row]));
            result[row] = simd_sum(result[row]);
        }
        """
        : """
        thread float result[outputs_per_simd] = {0.0f};
        for (uint row = 0; row < outputs_per_simd; ++row) {
            uint output_row = first_row + row;
            const device uint8_t* weight =
                expert_weight + output_row * packed_row_bytes + lane * 8;
            const device uint8_t* scale =
                expert_scales + output_row * scale_row_bytes + scale_lane;
            uint8_t sb =
                \(patch)
                : scale[0];
            result[row] = laguna_nvfp4_qdot_16(
                weight,
                input_values,
                laguna_nvfp4_scale(sb));
            result[row] = simd_sum(result[row]);
        }
        """
    return """
constexpr uint input_width = 512;
constexpr uint output_width = 2048;
constexpr uint routed_experts = 8;
constexpr uint shared_slot = 8;
constexpr uint outputs_per_simd = 4;
constexpr uint values_per_lane = 16;
constexpr uint packed_row_bytes = 256;
constexpr uint scale_patch_bytes = \(lagunaScalePatchHeaderBytes);
constexpr uint shared_scale_row_bytes = \(sharedRowBytes);
constexpr uint routed_scale_row_bytes = 16;
constexpr uint packed_expert_bytes =
    output_width * packed_row_bytes;
constexpr uint scale_expert_bytes =
    output_width * routed_scale_row_bytes;

uint tile = threadgroup_position_in_grid.x;
uint slot = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint first_row = tile * outputs_per_simd;
bool is_shared = slot == shared_slot;
uint expert = is_shared ? 0 : uint(indices[slot]);

const device bfloat* expert_input = is_shared
    ? shared_activated
    : routed_activated + slot * input_width;
const device uint8_t* expert_weight = is_shared
    ? (const device uint8_t*)shared_down_weight
    : (const device uint8_t*)routed_down_weight +
        expert * packed_expert_bytes;
const device uint8_t* expert_scales = is_shared
    ? \(sharedBase)
    : routed_down_scales + scale_patch_bytes
        + expert * scale_expert_bytes;
uint scale_row_bytes =
    is_shared ? shared_scale_row_bytes : routed_scale_row_bytes;
uint scale_lane = \(scaleLane);

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

\(qdots)

threadgroup bfloat down_outputs[
    (routed_experts + 1) * outputs_per_simd
];
if (lane == 0) {
    for (uint row = 0; row < outputs_per_simd; ++row) {
        down_outputs[slot * outputs_per_simd + row] =
            bfloat(result[row]\(lagunaNvfp4RowScaleSuffix));
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (slot == 0 && lane < outputs_per_simd) {
    bfloat routed_total = bfloat(0);
    for (uint routed_slot = 0;
         routed_slot < routed_experts;
         ++routed_slot) {
        bfloat route_weight =
            bfloat(router_weights[routed_slot]);
        bfloat product = bfloat(
            down_outputs[
                routed_slot * outputs_per_simd + lane
            ] * route_weight);
        routed_total = bfloat(product + routed_total);
    }
    bfloat routed = bfloat(
        routed_total * bfloat(2.5f));
    bfloat shared =
        down_outputs[shared_slot * outputs_per_simd + lane];
    bfloat r2 = bfloat(routed + shared);
    output[first_row + lane] =
        bfloat(residual[first_row + lane] + r2);
}
"""
}






private let lagunaRoutedSharedDownResidualStagedKernel = MLXFast.metalKernel(
    name: lagunaSharedFirstDownOrderEnabled
        ? "laguna_routed_shared_nvfp4_down_residual_bf16_stage4_v6sf"
        : "laguna_routed_shared_nvfp4_down_residual_bf16_stage4_v6",
    inputNames: lagunaSharedFirstDownOrderEnabled
        ? [
            "shared_activated", "shared_down_weight", "shared_down_scales",
            "routed_activated", "routed_down_weight", "routed_down_scales",
            "indices", "router_weights", "residual",
        ]
        : [
            "routed_activated", "routed_down_weight", "routed_down_scales",
            "indices", "router_weights", "shared_activated",
            "shared_down_weight", "shared_down_scales", "residual",
        ],
    outputNames: ["output"],
    source: """
constexpr uint input_width = 512;
constexpr uint output_width = 2048;
constexpr uint routed_experts = 8;
constexpr uint shared_slot = 8;
constexpr uint outputs_per_simd = 4;
constexpr uint values_per_lane = 16;
constexpr uint packed_row_bytes = 256;
constexpr uint scale_patch_bytes = \(lagunaScalePatchHeaderBytes);
constexpr uint shared_scale_row_bytes = 32;
constexpr uint routed_scale_row_bytes = 16;
constexpr uint packed_expert_bytes =
    output_width * packed_row_bytes;
constexpr uint scale_expert_bytes =
    output_width * routed_scale_row_bytes;

uint tile = threadgroup_position_in_grid.x;
uint slot = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint first_row = tile * outputs_per_simd;
bool is_shared = slot == shared_slot;
uint expert = is_shared ? 0 : uint(indices[slot]);

const device bfloat* expert_input = is_shared
    ? shared_activated
    : routed_activated + slot * input_width;
const device uint8_t* expert_weight = is_shared
    ? (const device uint8_t*)shared_down_weight
    : (const device uint8_t*)routed_down_weight +
        expert * packed_expert_bytes;
const device uint8_t* expert_scales = is_shared
    ? shared_down_scales
    : routed_down_scales + scale_patch_bytes
        + expert * scale_expert_bytes;
uint scale_row_bytes =
    is_shared ? shared_scale_row_bytes : routed_scale_row_bytes;
uint scale_lane = is_shared ? lane : (lane >> 1);

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

thread float result[outputs_per_simd] = {0.0f};
uint2 row_codes[outputs_per_simd];
uint8_t row_sb[outputs_per_simd];
for (uint row = 0; row < outputs_per_simd; ++row) {
    uint output_row = first_row + row;
    row_codes[row] = *(const device uint2*)(
        expert_weight + output_row * packed_row_bytes + lane * 8);
    const device uint8_t* scale =
        expert_scales + output_row * scale_row_bytes + scale_lane;
    row_sb[row] =
        (!is_shared && expert == 0 && output_row == 0 && lane == 1)
        ? routed_down_scales[0]
        : scale[0];
}
for (uint row = 0; row < outputs_per_simd; ++row) {
    result[row] = laguna_nvfp4_qdot_codes_16(
        row_codes[row],
        input_values,
        laguna_nvfp4_scale(row_sb[row]));
    result[row] = simd_sum(result[row]);
}

threadgroup bfloat down_outputs[
    (routed_experts + 1) * outputs_per_simd
];
if (lane == 0) {
    for (uint row = 0; row < outputs_per_simd; ++row) {
        down_outputs[slot * outputs_per_simd + row] =
            bfloat(result[row]\(lagunaNvfp4RowScaleSuffix));
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (slot == 0 && lane < outputs_per_simd) {
    bfloat routed_total = bfloat(0);
    for (uint routed_slot = 0;
         routed_slot < routed_experts;
         ++routed_slot) {
        bfloat route_weight =
            bfloat(router_weights[routed_slot]);
        bfloat product = bfloat(
            down_outputs[
                routed_slot * outputs_per_simd + lane
            ] * route_weight);
        routed_total = bfloat(product + routed_total);
    }
    bfloat routed = bfloat(
        routed_total * bfloat(2.5f));
    bfloat shared =
        down_outputs[shared_slot * outputs_per_simd + lane];
    bfloat r2 = bfloat(routed + shared);
    output[first_row + lane] =
        bfloat(residual[first_row + lane] + r2);
}
""",
    header: lagunaSharedSwiGLUQMVHeader,
    ensureRowContiguous: true
)




private let lagunaRoutedSharedDownResidualStagedSharedHalvedKernel =
    MLXFast.metalKernel(
        name: lagunaSharedFirstDownOrderEnabled
            ? "laguna_routed_shared_nvfp4_down_residual_bf16_sh_stage4_v6sf"
            : "laguna_routed_shared_nvfp4_down_residual_bf16_sh_stage4_v6",
        inputNames: lagunaSharedFirstDownOrderEnabled
            ? [
                "shared_activated", "shared_down_weight", "shared_down_scales",
                "routed_activated", "routed_down_weight", "routed_down_scales",
                "indices", "router_weights", "residual",
            ]
            : [
                "routed_activated", "routed_down_weight", "routed_down_scales",
                "indices", "router_weights", "shared_activated",
                "shared_down_weight", "shared_down_scales", "residual",
            ],
        outputNames: ["output"],
        source: lagunaRoutedSharedDownResidualSource(
            sharedHalved: true, staged: true),
        header: lagunaSharedSwiGLUQMVHeader,
        ensureRowContiguous: true
    )

func lagunaRoutedSharedDownResidual(
    routedActivated: MLXArray,
    routedDownWeight: MLXArray,
    routedDownScales: MLXArray,
    indices: MLXArray,
    routerWeights: MLXArray,
    sharedActivated: MLXArray,
    sharedDownWeight: MLXArray,
    sharedDownScales: MLXArray,
    residual: MLXArray,
    staged: Bool = lagunaFusedDownRowStagingEnabled
) -> MLXArray {
    precondition(routedActivated.dtype == .bfloat16)
    precondition(
        routedActivated.dims(1, 1, LagunaConstants.numExpertsPerTok, 1,
            LagunaConstants.moeIntermediateSize))
    precondition(routedDownWeight.dtype == .uint32)
    precondition(
        routedDownWeight.dims(LagunaConstants.numExperts, LagunaConstants.hiddenSize,
            LagunaConstants.moeIntermediateSize / 8))
    precondition(routedDownScales.dtype == .uint8)
    precondition(routedDownScales.size == lagunaRoutedDownScaleBytes)
    precondition(indices.dtype == .uint32)
    precondition(indices.dims(1, 1, LagunaConstants.numExpertsPerTok))
    precondition(routerWeights.dtype == .float32)
    precondition(routerWeights.dims(1, 1, LagunaConstants.numExpertsPerTok))
    precondition(sharedActivated.dtype == .bfloat16)
    precondition(
        sharedActivated.dims(1, 1, LagunaConstants.sharedExpertIntermediateSize))
    precondition(sharedDownWeight.dtype == .uint32)
    precondition(
        sharedDownWeight.dims(LagunaConstants.hiddenSize,
            LagunaConstants.sharedExpertIntermediateSize / 8))
    precondition(sharedDownScales.dtype == .uint8)
    let sharedHalved = sharedDownScales.ndim == 1
    if sharedHalved {
        precondition(
            sharedDownScales.size == lagunaScalePatchHeaderBytes
                + LagunaConstants.hiddenSize
                * (LagunaConstants.sharedExpertIntermediateSize / 32))
    } else {
        precondition(
            sharedDownScales.dims(LagunaConstants.hiddenSize,
                LagunaConstants.sharedExpertIntermediateSize / 16))
    }
    precondition(residual.dtype == .bfloat16)
    precondition(residual.dims(1, 1, LagunaConstants.hiddenSize))

    let fusedKernel =
        sharedHalved
        ? (staged
            ? lagunaRoutedSharedDownResidualStagedSharedHalvedKernel
            : lagunaRoutedSharedDownResidualSharedHalvedKernel)
        : (staged
            ? lagunaRoutedSharedDownResidualStagedKernel
            : lagunaRoutedSharedDownResidualKernel)
    return fusedKernel(
        lagunaSharedFirstDownOrderEnabled
            ? [
                sharedActivated, sharedDownWeight, sharedDownScales,
                routedActivated, routedDownWeight, routedDownScales,
                indices, routerWeights, residual,
            ]
            : [
                routedActivated, routedDownWeight, routedDownScales,
                indices, routerWeights, sharedActivated,
                sharedDownWeight, sharedDownScales, residual,
            ],
        grid: (LagunaConstants.hiddenSize / 4 * 288, 1, 1),
        threadGroup: (288, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.hiddenSize]],
        outputDTypes: [.bfloat16]
    )[0]
}















private let lagunaDenseGateUpSwiGLUKernel = MLXFast.metalKernel(
    name: "laguna_dense_gate_up_swiglu_bf16_v1",
    inputNames: ["input", "fused_weight"],
    outputNames: ["activated"],
    source: """
constexpr uint in_vec_size = 2048;
constexpr uint output_width = 8192;
constexpr uint rows_per_thread = 4;
constexpr uint values_per_thread = 4;
constexpr uint block_width = 128;
constexpr uint blocks = in_vec_size / block_width;
constexpr uint rows_per_group = 64;

uint tile = threadgroup_position_in_grid.x;
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;

uint row_base = tile * rows_per_group + simd_group * rows_per_thread;

thread float gate_result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
thread float up_result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
thread float coefficients[values_per_thread];

uint column = lane * values_per_thread;
for (uint block = 0; block < blocks; ++block) {
    const vec<bfloat, 4> c4 =
        *((const device vec<bfloat, 4>*)(input + column));
    for (uint i = 0; i < values_per_thread; ++i) {
        coefficients[i] = float(c4[i]);
    }
    for (uint row = 0; row < rows_per_thread; ++row) {
        const device vec<bfloat, 4>* gate_row_values =
            (const device vec<bfloat, 4>*)(
                fused_weight + (row_base + row) * in_vec_size + column);
        const vec<bfloat, 4> gw = gate_row_values[0];
        const device vec<bfloat, 4>* up_row_values =
            (const device vec<bfloat, 4>*)(
                fused_weight +
                (output_width + row_base + row) * in_vec_size + column);
        const vec<bfloat, 4> uw = up_row_values[0];
        for (uint i = 0; i < values_per_thread; ++i) {
            gate_result[row] += float(gw[i]) * coefficients[i];
            up_result[row] += float(uw[i]) * coefficients[i];
        }
    }
    column += block_width;
}

for (uint row = 0; row < rows_per_thread; ++row) {
    for (ushort delta = 16; delta >= 1; delta >>= 1) {
        gate_result[row] +=
            metal::simd_shuffle_down(gate_result[row], delta);
        up_result[row] +=
            metal::simd_shuffle_down(up_result[row], delta);
    }
}
if (lane == 0) {
    for (uint row = 0; row < rows_per_thread; ++row) {
        bfloat gate = bfloat(gate_result[row]);
        bfloat up = bfloat(up_result[row]);
        bfloat exp_abs = metal::exp(metal::abs(gate));
        bfloat denominator = bfloat(1) + exp_abs;
        bfloat y = bfloat(1) / denominator;
        bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
        bfloat silu = bfloat(gate * sigmoid);
        activated[row_base + row] = bfloat(silu * up);
    }
}
""",
    ensureRowContiguous: true
)



func lagunaDenseGateUpSwiGLU(
    _ input: MLXArray,
    fusedWeight: MLXArray
) -> MLXArray {
    precondition(input.dtype == .bfloat16)
    precondition(input.dims(1, 1, LagunaConstants.hiddenSize))
    precondition(fusedWeight.dtype == .bfloat16)
    precondition(
        fusedWeight.dims(2 * LagunaConstants.denseIntermediateSize, LagunaConstants.hiddenSize))

    return lagunaDenseGateUpSwiGLUKernel(
        [input, fusedWeight],
        grid: ((LagunaConstants.denseIntermediateSize / 64) * 512, 1, 1),
        threadGroup: (512, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.denseIntermediateSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

private let lagunaDenseDownResidualKernel = MLXFast.metalKernel(
    name: "laguna_dense_down_residual_bf16_v1",
    inputNames: ["activated", "down_weight", "residual"],
    outputNames: ["output"],
    source: """
constexpr uint in_vec_size = 8192;
constexpr uint rows_per_thread = 4;
constexpr uint values_per_thread = 4;
constexpr uint block_width = 128;
constexpr uint blocks = in_vec_size / block_width;
constexpr uint rows_per_group = 16;

uint tile = threadgroup_position_in_grid.x;
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;

uint row_base = tile * rows_per_group + simd_group * rows_per_thread;

thread float result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
thread float coefficients[values_per_thread];

uint column = lane * values_per_thread;
for (uint block = 0; block < blocks; ++block) {
    const vec<bfloat, 4> c4 =
        *((const device vec<bfloat, 4>*)(activated + column));
    for (uint i = 0; i < values_per_thread; ++i) {
        coefficients[i] = float(c4[i]);
    }
    for (uint row = 0; row < rows_per_thread; ++row) {
        const device vec<bfloat, 4>* row_values =
            (const device vec<bfloat, 4>*)(
                down_weight + (row_base + row) * in_vec_size + column);
        const vec<bfloat, 4> w = row_values[0];
        for (uint i = 0; i < values_per_thread; ++i) {
            result[row] += float(w[i]) * coefficients[i];
        }
    }
    column += block_width;
}

for (uint row = 0; row < rows_per_thread; ++row) {
    for (ushort delta = 16; delta >= 1; delta >>= 1) {
        result[row] += metal::simd_shuffle_down(result[row], delta);
    }
}
if (lane == 0) {
    for (uint row = 0; row < rows_per_thread; ++row) {
        bfloat down = bfloat(result[row]);
        output[row_base + row] =
            bfloat(residual[row_base + row] + down);
    }
}
""",
    ensureRowContiguous: true
)

func lagunaDenseDownResidual(
    _ activated: MLXArray,
    downWeight: MLXArray,
    residual: MLXArray
) -> MLXArray {
    precondition(activated.dtype == .bfloat16)
    precondition(activated.dims(1, 1, LagunaConstants.denseIntermediateSize))
    precondition(downWeight.dtype == .bfloat16)
    precondition(
        downWeight.dims(LagunaConstants.hiddenSize, LagunaConstants.denseIntermediateSize))
    precondition(residual.dtype == .bfloat16)
    precondition(residual.dims(1, 1, LagunaConstants.hiddenSize))

    return lagunaDenseDownResidualKernel(
        [activated, downWeight, residual],
        grid: ((LagunaConstants.hiddenSize / 16) * 128, 1, 1),
        threadGroup: (128, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.hiddenSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

final class LagunaRuntimeMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear








    var _fusedGateUpWeight: MLXArray?
    var _fusedGateUpScales: MLXArray?
    var _fusedGateUpSplit: Int = 0





    var _fusedGateUpScalesHalved: MLXArray?
    var _sharedDownScalesHalved: MLXArray?









    var _fusedDenseGateUpWeight: MLXArray?

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }







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
            gate.weight.sameDims(up.weight),
            gate.scales.sameDims(up.scales),
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
        var prepared = [fusedWeight, fusedScales]
        guard lagunaSharedScaleHalvedEnabled else { return prepared }




        let gateUpPairs = gate.scales.dim(0) * gate.scales.dim(1) / 2


        if lagunaSharedSwiGLUQMVRows1Enabled,
            let halved = lagunaHalvedGroup32ScalePlane(
                fusedScales, allowedFlatPairs: [0, gateUpPairs])
        {
            _fusedGateUpScalesHalved = halved
            prepared.append(halved)
            lagunaPackedScalesLog.note("active", "shared gate/up halved")
        }
        if let down = downProj as? QuantizedLinear,
            type(of: down) == QuantizedLinear.self,
            down.mode == .nvfp4, down.groupSize == 16, down.bits == 4,
            down.scales.dtype == .uint8, down.scales.ndim == 2,
            let halvedDown = lagunaHalvedGroup32ScalePlane(
                down.scales, allowedFlatPairs: [0])
        {
            _sharedDownScalesHalved = halvedDown
            prepared.append(halvedDown)
            lagunaPackedScalesLog.note("active", "shared down halved")
        }
        return prepared
    }









    func prepareFusedDenseGateUp() -> MLXArray? {
        let hidden = LagunaConstants.hiddenSize
        let intermediate = LagunaConstants.denseIntermediateSize
        guard _fusedDenseGateUpWeight == nil,
            type(of: gateProj) == Linear.self,
            type(of: upProj) == Linear.self,
            gateProj.bias == nil, upProj.bias == nil,
            gateProj.weight.dtype == .bfloat16,
            upProj.weight.dtype == .bfloat16,
            gateProj.weight.dims(intermediate, hidden),
            upProj.weight.dims(intermediate, hidden)
        else {
            return nil
        }
        let fusedWeight = concatenated([gateProj.weight, upProj.weight], axis: 0)
        _fusedDenseGateUpWeight = fusedWeight
        return fusedWeight
    }




    func fusedSharedBanks(
        _ x: MLXArray
    ) -> (
        gateUpWeight: MLXArray,
        gateUpScales: MLXArray,
        downWeight: MLXArray,
        downScales: MLXArray
    )? {
        guard let banks = fusedSharedBankGuard(x) else { return nil }
        return banks
    }






    func fusedSharedDownInputs(
        _ x: MLXArray,
        sharedActivation: MLXArray? = nil
    ) -> (
        activated: MLXArray,
        downWeight: MLXArray,
        downScales: MLXArray
    )? {
        guard let banks = fusedSharedBankGuard(x) else { return nil }
        let activated =
            sharedActivation
            ?? lagunaSharedSwiGLUQMV(
                x,
                fusedWeight: banks.gateUpWeight,
                fusedScales: banks.gateUpScales
            )
        return (activated, banks.downWeight, banks.downScales)
    }

    private func fusedSharedBankGuard(
        _ x: MLXArray
    ) -> (
        gateUpWeight: MLXArray,
        gateUpScales: MLXArray,
        downWeight: MLXArray,
        downScales: MLXArray
    )? {
        guard lagunaFusedSharedSwiGLUQMVEnabled,
            let fusedWeight = _fusedGateUpWeight,
            let fusedScales = _fusedGateUpScales,
            let down = downProj as? QuantizedLinear,
            type(of: down) == QuantizedLinear.self,
            down.mode == .nvfp4,
            down.groupSize == 16,
            down.bits == 4,
            down.bias == nil,
            down.biases == nil,
            x.dtype == .bfloat16,
            x.dims(1, 1, LagunaConstants.hiddenSize),
            fusedWeight.dtype == .uint32,
            fusedScales.dtype == .uint8,
            _fusedGateUpSplit == LagunaConstants.sharedExpertIntermediateSize,
            down.weight.dtype == .uint32,
            down.weight.dims(LagunaConstants.hiddenSize,
                LagunaConstants.sharedExpertIntermediateSize / 8),
            down.scales.dtype == .uint8,
            down.scales.dims(LagunaConstants.hiddenSize,
                LagunaConstants.sharedExpertIntermediateSize / 16)
        else {
            return nil
        }



        return (
            fusedWeight,
            _fusedGateUpScalesHalved ?? fusedScales,
            down.weight,
            _sharedDownScalesHalved ?? down.scales
        )
    }

    func fusedSharedDownResidual(
        _ x: MLXArray,
        routed: MLXArray,
        residual: MLXArray
    ) -> MLXArray? {
        guard lagunaFusedSharedDownResidualEnabled,
            let inputs = fusedSharedDownInputs(x),
            routed.dtype == .bfloat16,
            routed.dims(1, 1, LagunaConstants.hiddenSize),
            residual.dtype == .bfloat16,
            residual.dims(1, 1, LagunaConstants.hiddenSize)
        else {
            return nil
        }

        lagunaTrace("shared down residual")
        return lagunaSharedDownResidual(
            inputs.activated,
            downWeight: inputs.downWeight,
            downScales: inputs.downScales,
            routed: routed,
            residual: residual
        )
    }














    func fusedDenseDownResidual(
        _ x: MLXArray, residual: MLXArray
    ) -> MLXArray? {
        let hidden = LagunaConstants.hiddenSize
        let intermediate = LagunaConstants.denseIntermediateSize
        guard x.dim(1) == 1,
            x.dtype == .bfloat16,
            x.dims(1, 1, hidden),
            residual.dtype == .bfloat16,
            residual.dims(1, 1, hidden),
            type(of: gateProj) == Linear.self,
            type(of: upProj) == Linear.self,
            type(of: downProj) == Linear.self,
            gateProj.bias == nil, upProj.bias == nil, downProj.bias == nil,
            gateProj.weight.dtype == .bfloat16,
            upProj.weight.dtype == .bfloat16,
            downProj.weight.dtype == .bfloat16,
            gateProj.weight.dims(intermediate, hidden),
            upProj.weight.dims(intermediate, hidden),
            downProj.weight.dims(hidden, intermediate)
        else {
            return nil
        }

        let activated: MLXArray
        if lagunaFusedDenseGateUpSwiGLUEnabled,
            let fusedWeight = _fusedDenseGateUpWeight,
            fusedWeight.dtype == .bfloat16,
            fusedWeight.dims(2 * intermediate, hidden)
        {
            lagunaTrace("dense gate/up GEMV + SwiGLU")
            activated = lagunaDenseGateUpSwiGLU(x, fusedWeight: fusedWeight)
        } else {
            activated = compiledSiluProduct(gateProj(x), upProj(x))
        }

        if lagunaFusedDenseDownResidualEnabled {
            lagunaTrace("dense down GEMV + residual")
            return lagunaDenseDownResidual(
                activated, downWeight: downProj.weight, residual: residual)
        }
        return residual + downProj(activated)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if x.dim(1) == 1,
            let fusedWeight = _fusedGateUpWeight, let fusedScales = _fusedGateUpScales
        {
            if lagunaFusedSharedSwiGLUQMVEnabled,
                x.dtype == .bfloat16,
                x.dims(1, 1, LagunaConstants.hiddenSize),
                fusedWeight.dtype == .uint32,
                fusedScales.dtype == .uint8,
                _fusedGateUpSplit == LagunaConstants.sharedExpertIntermediateSize
            {
                lagunaTrace("shared gate/up QMV + SwiGLU")
                return downProj(
                    lagunaSharedSwiGLUQMV(
                        x,
                        fusedWeight: fusedWeight,
                        fusedScales: fusedScales
                    )
                )
            }







            lagunaTrace("shared fused [gate; up] bank QMM")
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


















private func lagunaDecodeRouterTop8KernelSource(normalizing: Bool) -> String {
    let epilogue =
        normalizing
        ? """
float total = 0.0f;
for (uint i = 0; i < 8; ++i) {
    total = simd_shuffle(my_score, ushort(i)) + total;
}
if (lane < 8) {
    router_indices[lane] = my_index;
    router_scores[lane] = my_score / total;
}
"""
        : """
if (lane < 8) {
    router_indices[lane] = my_index;
    router_scores[lane] = my_score;
}
"""
    return """
uint lane = thread_position_in_threadgroup.x;

threadgroup float xchg_keys[256];
threadgroup uint xchg_indices[256];
threadgroup float xchg_scores[256];

float x = float(logits[lane]);
float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
float my_score = x < 0.0f ? y : 1.0f - y;
float my_key = -(my_score + float(correction_bias[lane]));
uint my_index = lane;

for (uint sequence = 2; sequence <= 256; sequence <<= 1) {
    for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
        float other_key;
        uint other_index;
        float other_score;
        if (stride < 32) {
            other_key = simd_shuffle_xor(my_key, ushort(stride));
            other_index = simd_shuffle_xor(my_index, ushort(stride));
            other_score = simd_shuffle_xor(my_score, ushort(stride));
        } else {
            xchg_keys[lane] = my_key;
            xchg_indices[lane] = my_index;
            xchg_scores[lane] = my_score;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint partner = lane ^ stride;
            other_key = xchg_keys[partner];
            other_index = xchg_indices[partner];
            other_score = xchg_scores[partner];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        bool is_lower = (lane & stride) == 0;
        float a_key = is_lower ? my_key : other_key;
        uint a_index = is_lower ? my_index : other_index;
        float a_score = is_lower ? my_score : other_score;
        float b_key = is_lower ? other_key : my_key;
        uint b_index = is_lower ? other_index : my_index;
        float b_score = is_lower ? other_score : my_score;

        bool lower_wants_better = (lane & sequence) == 0;
        bool b_before_a = laguna_router_key_before(
            b_key, b_index, a_key, a_index);
        bool a_before_b = laguna_router_key_before(
            a_key, a_index, b_key, b_index);
        bool swap = lower_wants_better ? b_before_a : a_before_b;
        if (swap) {
            my_key = is_lower ? b_key : a_key;
            my_index = is_lower ? b_index : a_index;
            my_score = is_lower ? b_score : a_score;
        }
    }
}

\(epilogue)
"""
}

private let lagunaDecodeRouterTop8Header = """
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
"""

private let lagunaDecodeRouterTop8Kernel = MLXFast.metalKernel(
    name: "laguna_decode_router_top8_v3",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaDecodeRouterTop8KernelSource(normalizing: false),
    header: lagunaDecodeRouterTop8Header,
    ensureRowContiguous: true
)

private let lagunaDecodeRouterTop8NormalizingKernel = MLXFast.metalKernel(
    name: "laguna_decode_router_top8_norm_v2",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaDecodeRouterTop8KernelSource(normalizing: true),
    header: lagunaDecodeRouterTop8Header,
    ensureRowContiguous: true
)












private func lagunaDecodeRouterOrdinalKernelSource(
    normalizing: Bool, scoreTable: Bool = false
) -> String {
    let scoreStorage =
        scoreTable
        ? "threadgroup float original_scores[256];"
        : ""
    let scoreStore =
        scoreTable
        ? "original_scores[lane] = score;"
        : ""
    let winnerScore =
        scoreTable
        ? """
    my_score = original_scores[my_index];
"""
        : """
    float winner_x = float(logits[my_index]);
    float winner_y = 1.0f / (1.0f + metal::exp(metal::abs(winner_x)));
    my_score = winner_x < 0.0f ? winner_y : 1.0f - winner_y;
"""
    let epilogue =
        normalizing
        ? """
float my_score = 0.0f;
if (lane < 8) {
\(winnerScore)
}
float total = 0.0f;
for (uint i = 0; i < 8; ++i) {
    total = simd_shuffle(my_score, ushort(i)) + total;
}
if (lane < 8) {
    router_indices[lane] = my_index;
    router_scores[lane] = my_score / total;
}
"""
        : """
if (lane < 8) {
    float my_score = 0.0f;
\(winnerScore)
    router_indices[lane] = my_index;
    router_scores[lane] = my_score;
}
"""
    return """
uint lane = thread_position_in_threadgroup.x;

threadgroup uint xchg_ordinals[256];
threadgroup uint xchg_indices[256];
\(scoreStorage)

float x = float(logits[lane]);
float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
float score = x < 0.0f ? y : 1.0f - y;
\(scoreStore)
float key = -(score + float(correction_bias[lane]));
uint my_ordinal = laguna_router_key_ordinal(key);
uint my_index = lane;

for (uint sequence = 2; sequence <= 256; sequence <<= 1) {
    for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
        uint other_ordinal;
        uint other_index;
        if (stride < 32) {
            other_ordinal = simd_shuffle_xor(my_ordinal, ushort(stride));
            other_index = simd_shuffle_xor(my_index, ushort(stride));
        } else {
            xchg_ordinals[lane] = my_ordinal;
            xchg_indices[lane] = my_index;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint partner = lane ^ stride;
            other_ordinal = xchg_ordinals[partner];
            other_index = xchg_indices[partner];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        bool is_lower = (lane & stride) == 0;
        bool lower_wants_better = (lane & sequence) == 0;
        bool want_better = lower_wants_better == is_lower;
        bool other_before_my = laguna_router_ordinal_before(
            other_ordinal, other_index, my_ordinal, my_index);
        bool take_other = want_better ? other_before_my : !other_before_my;
        if (take_other) {
            my_ordinal = other_ordinal;
            my_index = other_index;
        }
    }
}

\(epilogue)
"""
}

private let lagunaDecodeRouterOrdinalHeader = """
METAL_FUNC uint laguna_router_key_ordinal(float key) {
    uint bits = as_type<uint>(key);
    uint magnitude = bits & 0x7FFFFFFFu;
    if (magnitude > 0x7F800000u) {
        return 0xFFFFFFFFu;
    }
    if (magnitude == 0u) {
        return 0x80000000u;
    }
    return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
}

METAL_FUNC bool laguna_router_ordinal_before(
    uint a, uint a_index, uint b, uint b_index) {
    if (a < b) {
        return true;
    }
    if (b < a) {
        return false;
    }
    return a_index < b_index;
}
"""

private let lagunaDecodeRouterOrdinalKernel = MLXFast.metalKernel(
    name: "laguna_decode_router_top8_ordinal_v1",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaDecodeRouterOrdinalKernelSource(normalizing: false),
    header: lagunaDecodeRouterOrdinalHeader,
    ensureRowContiguous: true
)

private let lagunaDecodeRouterOrdinalNormalizingKernel = MLXFast.metalKernel(
    name: "laguna_decode_router_top8_ordinal_norm_v1",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaDecodeRouterOrdinalKernelSource(normalizing: true),
    header: lagunaDecodeRouterOrdinalHeader,
    ensureRowContiguous: true
)

private let lagunaDecodeRouterOrdinalScoreTableKernel = MLXFast.metalKernel(
    name: "laguna_decode_router_top8_ordinal_table_v1",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaDecodeRouterOrdinalKernelSource(normalizing: false, scoreTable: true),
    header: lagunaDecodeRouterOrdinalHeader,
    ensureRowContiguous: true
)

private let lagunaDecodeRouterOrdinalScoreTableNormalizingKernel = MLXFast.metalKernel(
    name: "laguna_decode_router_top8_ordinal_table_norm_v1",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaDecodeRouterOrdinalKernelSource(normalizing: true, scoreTable: true),
    header: lagunaDecodeRouterOrdinalHeader,
    ensureRowContiguous: true
)

private let lagunaDecodeRouterOrdinalEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ROUTER_ORDINAL"] != "0"




private let lagunaDecodeRouterOrdinalScoreTableEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ROUTER_ORDINAL_SCORE_TABLE"] != "0"












































private let lagunaDecodeRouterTournamentEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_DECODE_ROUTER_TOURNAMENT"] != "0"

func lagunaDecodeRouterTop8AcceptedForTesting(
    logits: MLXArray, correctionBias: MLXArray, normalizing: Bool = false
) -> (MLXArray, MLXArray) {
    precondition(logits.dtype == .bfloat16 || logits.dtype == .float32)
    precondition(correctionBias.dtype == .float32)
    precondition(logits.size == 256)
    precondition(correctionBias.size == 256)

    let kernel =
        normalizing ? lagunaDecodeRouterTop8NormalizingKernel : lagunaDecodeRouterTop8Kernel
    let outputs = kernel(
        [logits, correctionBias],
        grid: (256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, 1, 8], [1, 1, 8]],
        outputDTypes: [.uint32, .float32]
    )
    return (outputs[0], outputs[1])
}

func lagunaDecodeRouterTop8OrdinalForTesting(
    logits: MLXArray, correctionBias: MLXArray, normalizing: Bool = false
) -> (MLXArray, MLXArray) {
    precondition(logits.dtype == .bfloat16 || logits.dtype == .float32)
    precondition(correctionBias.dtype == .float32)
    precondition(logits.size == 256)
    precondition(correctionBias.size == 256)

    let kernel =
        normalizing
        ? lagunaDecodeRouterOrdinalNormalizingKernel : lagunaDecodeRouterOrdinalKernel
    let outputs = kernel(
        [logits, correctionBias],
        grid: (256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, 1, 8], [1, 1, 8]],
        outputDTypes: [.uint32, .float32]
    )
    return (outputs[0], outputs[1])
}

func lagunaDecodeRouterTop8OrdinalScoreTableForTesting(
    logits: MLXArray, correctionBias: MLXArray, normalizing: Bool = false
) -> (MLXArray, MLXArray) {
    precondition(logits.dtype == .bfloat16 || logits.dtype == .float32)
    precondition(correctionBias.dtype == .float32)
    precondition(logits.size == 256)
    precondition(correctionBias.size == 256)

    let kernel =
        normalizing
        ? lagunaDecodeRouterOrdinalScoreTableNormalizingKernel
        : lagunaDecodeRouterOrdinalScoreTableKernel
    let outputs = kernel(
        [logits, correctionBias],
        grid: (256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, 1, 8], [1, 1, 8]],
        outputDTypes: [.uint32, .float32]
    )
    return (outputs[0], outputs[1])
}

private func lagunaDecodeRouterTop8(
    logits: MLXArray, correctionBias: MLXArray, normalizing: Bool = false
) -> (MLXArray, MLXArray) {
    if lagunaDecodeRouterOrdinalEnabled {
        if lagunaDecodeRouterOrdinalScoreTableEnabled {
            if lagunaDecodeRouterTournamentEnabled {
                return lagunaPrefillRouterTournamentOrdinalForTesting(
                    logits: logits,
                    correctionBias: correctionBias,
                    rows: 1,
                    normalizing: normalizing
                )
            }
            return lagunaDecodeRouterTop8OrdinalScoreTableForTesting(
                logits: logits,
                correctionBias: correctionBias,
                normalizing: normalizing
            )
        }
        return lagunaDecodeRouterTop8OrdinalForTesting(
            logits: logits, correctionBias: correctionBias, normalizing: normalizing)
    }
    return lagunaDecodeRouterTop8AcceptedForTesting(
        logits: logits,
        correctionBias: correctionBias,
        normalizing: normalizing
    )
}




private let lagunaDecodeRouterTop8Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_ROUTER"] != "0"




private let lagunaDecodeRouterCastSinkEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_ROUTER_CAST"] != "0"





private let lagunaDecodeRouterNormSinkEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_FUSED_ROUTER_NORM"] != "0"















private let lagunaPrefillRouterTop8Enabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_ROUTER_TOP8"] == "1"







private let lagunaPrefillMoETailEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_MOE_TAIL"] != "0"






private let lagunaPrefillSortedMoETailEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_SORTED_MOE_TAIL"] != "0"





















private func lagunaPrefillRouterTop8KernelSource(normalizing: Bool) -> String {
    let epilogue =
        normalizing
        ? """
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            total = selected_scores[i] + total;
        }
        router_scores[row * 8 + lane] = selected_scores[lane] / total;
"""
        : """
        router_scores[row * 8 + lane] = selected_scores[lane];
"""
    return """
uint lane = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.y;

threadgroup float choice_keys[256];
threadgroup float selected_scores[8];

float x = float(logits[row * 256 + lane]);
float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
float score = x < 0.0f ? y : 1.0f - y;
float corrected = score + float(correction_bias[lane]);
float my_key = -corrected;
choice_keys[lane] = my_key;
threadgroup_barrier(mem_flags::mem_threadgroup);

uint rank = 0;
for (uint j = 0; j < 256; ++j) {
    rank += laguna_router_key_before(
        choice_keys[j], j, my_key, lane) ? 1 : 0;
}
if (rank < 8) {
    router_indices[row * 8 + rank] = lane;
    selected_scores[rank] = score;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (lane < 8) {
\(epilogue)
}
"""
}

private let lagunaPrefillRouterTop8Kernel = MLXFast.metalKernel(
    name: "laguna_prefill_router_top8_v1",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaPrefillRouterTop8KernelSource(normalizing: false),
    header: lagunaDecodeRouterTop8Header,
    ensureRowContiguous: true
)

private let lagunaPrefillRouterTop8NormalizingKernel = MLXFast.metalKernel(
    name: "laguna_prefill_router_top8_norm_v1",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaPrefillRouterTop8KernelSource(normalizing: true),
    header: lagunaDecodeRouterTop8Header,
    ensureRowContiguous: true
)

private func lagunaPrefillRouterTop8(
    logits: MLXArray, correctionBias: MLXArray, rows: Int, normalizing: Bool
) -> (MLXArray, MLXArray) {
    precondition(logits.dtype == .bfloat16 || logits.dtype == .float32)
    precondition(correctionBias.dtype == .float32)
    precondition(logits.size == rows * 256)
    precondition(correctionBias.size == 256)

    let kernel =
        normalizing
        ? lagunaPrefillRouterTop8NormalizingKernel : lagunaPrefillRouterTop8Kernel
    let outputs = kernel(
        [logits, correctionBias],
        grid: (256, rows, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, rows, 8], [1, rows, 8]],
        outputDTypes: [.uint32, .float32]
    )
    return (outputs[0], outputs[1])
}





























































private func lagunaPrefillRouterTournamentKernelSource(normalizing: Bool) -> String {
    let epilogue =
        normalizing
        ? """
float total = 0.0f;
for (uint i = 0; i < 8; ++i) {
    total = simd_shuffle(my_score2, ushort(i)) + total;
}
if (lane < 8) {
    router_indices[row * 8 + lane] = my_index2;
    router_scores[row * 8 + lane] = my_score2 / total;
}
"""
        : """
if (lane < 8) {
    router_indices[row * 8 + lane] = my_index2;
    router_scores[row * 8 + lane] = my_score2;
}
"""
    return """
uint lane = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.y;

threadgroup float xchg_keys[256];
threadgroup uint xchg_indices[256];
threadgroup float xchg_scores[256];
threadgroup float candidate_keys[64];
threadgroup uint candidate_indices[64];
threadgroup float candidate_scores[64];

float x = float(logits[row * 256 + lane]);
float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
float my_score = x < 0.0f ? y : 1.0f - y;
float my_key = -(my_score + float(correction_bias[lane]));
uint my_index = lane;

for (uint sequence = 2; sequence <= 32; sequence <<= 1) {
    for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
        float other_key = simd_shuffle_xor(my_key, ushort(stride));
        uint other_index = simd_shuffle_xor(my_index, ushort(stride));
        float other_score = simd_shuffle_xor(my_score, ushort(stride));

        bool is_lower = (lane & stride) == 0;
        float a_key = is_lower ? my_key : other_key;
        uint a_index = is_lower ? my_index : other_index;
        float a_score = is_lower ? my_score : other_score;
        float b_key = is_lower ? other_key : my_key;
        uint b_index = is_lower ? other_index : my_index;
        float b_score = is_lower ? other_score : my_score;

        bool lower_wants_better = (lane & sequence) == 0;
        bool b_before_a = laguna_router_key_before(
            b_key, b_index, a_key, a_index);
        bool a_before_b = laguna_router_key_before(
            a_key, a_index, b_key, b_index);
        bool swap = lower_wants_better ? b_before_a : a_before_b;
        if (swap) {
            my_key = is_lower ? b_key : a_key;
            my_index = is_lower ? b_index : a_index;
            my_score = is_lower ? b_score : a_score;
        }
    }
}

uint block = lane >> 5;
uint within_block = lane & 31;
bool block_ascending = (block & 1) == 0;
uint rank_in_block = block_ascending ? within_block : (31 - within_block);
bool is_local_top8 = block_ascending ? (within_block < 8) : (within_block >= 24);
if (is_local_top8) {
    candidate_keys[block * 8 + rank_in_block] = my_key;
    candidate_indices[block * 8 + rank_in_block] = my_index;
    candidate_scores[block * 8 + rank_in_block] = my_score;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float my_key2 = candidate_keys[lane & 63];
uint my_index2 = candidate_indices[lane & 63];
float my_score2 = candidate_scores[lane & 63];
for (uint sequence = 2; sequence <= 64; sequence <<= 1) {
    for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
        float other_key;
        uint other_index;
        float other_score;
        if (stride < 32) {
            other_key = simd_shuffle_xor(my_key2, ushort(stride));
            other_index = simd_shuffle_xor(my_index2, ushort(stride));
            other_score = simd_shuffle_xor(my_score2, ushort(stride));
        } else {
            xchg_keys[lane] = my_key2;
            xchg_indices[lane] = my_index2;
            xchg_scores[lane] = my_score2;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint partner = lane ^ stride;
            other_key = xchg_keys[partner];
            other_index = xchg_indices[partner];
            other_score = xchg_scores[partner];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        bool is_lower = (lane & stride) == 0;
        float a_key = is_lower ? my_key2 : other_key;
        uint a_index = is_lower ? my_index2 : other_index;
        float a_score = is_lower ? my_score2 : other_score;
        float b_key = is_lower ? other_key : my_key2;
        uint b_index = is_lower ? other_index : my_index2;
        float b_score = is_lower ? other_score : my_score2;

        bool lower_wants_better = (lane & sequence) == 0;
        bool b_before_a = laguna_router_key_before(
            b_key, b_index, a_key, a_index);
        bool a_before_b = laguna_router_key_before(
            a_key, a_index, b_key, b_index);
        bool swap = lower_wants_better ? b_before_a : a_before_b;
        if (swap) {
            my_key2 = is_lower ? b_key : a_key;
            my_index2 = is_lower ? b_index : a_index;
            my_score2 = is_lower ? b_score : a_score;
        }
    }
}

\(epilogue)
"""
}









private func lagunaPrefillRouterTournamentOrdinalKernelSource(normalizing: Bool) -> String {
    let epilogue =
        normalizing
        ? """
float my_score2 = lane < 8 ? original_scores[my_index2] : 0.0f;
float total = 0.0f;
for (uint i = 0; i < 8; ++i) {
    total = simd_shuffle(my_score2, ushort(i)) + total;
}
if (lane < 8) {
    router_indices[row * 8 + lane] = my_index2;
    router_scores[row * 8 + lane] = my_score2 / total;
}
"""
        : """
if (lane < 8) {
    router_indices[row * 8 + lane] = my_index2;
    router_scores[row * 8 + lane] = original_scores[my_index2];
}
"""
    return """
uint lane = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.y;

threadgroup uint xchg_ordinals[64];
threadgroup uint xchg_indices[64];
threadgroup uint candidate_ordinals[64];
threadgroup uint candidate_indices[64];
threadgroup float original_scores[256];

float x = float(logits[row * 256 + lane]);
float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
float score = x < 0.0f ? y : 1.0f - y;
original_scores[lane] = score;
float key = -(score + float(correction_bias[lane]));
uint my_ordinal = laguna_router_key_ordinal(key);
uint my_index = lane;

for (uint sequence = 2; sequence <= 32; sequence <<= 1) {
    for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
        uint other_ordinal = simd_shuffle_xor(my_ordinal, ushort(stride));
        uint other_index = simd_shuffle_xor(my_index, ushort(stride));

        bool is_lower = (lane & stride) == 0;
        bool lower_wants_better = (lane & sequence) == 0;
        bool want_better = lower_wants_better == is_lower;
        bool other_before_my = laguna_router_ordinal_before(
            other_ordinal, other_index, my_ordinal, my_index);
        bool take_other = want_better ? other_before_my : !other_before_my;
        if (take_other) {
            my_ordinal = other_ordinal;
            my_index = other_index;
        }
    }
}

uint block = lane >> 5;
uint within_block = lane & 31;
bool block_ascending = (block & 1) == 0;
uint rank_in_block = block_ascending ? within_block : (31 - within_block);
bool is_local_top8 = block_ascending ? (within_block < 8) : (within_block >= 24);
if (is_local_top8) {
    candidate_ordinals[block * 8 + rank_in_block] = my_ordinal;
    candidate_indices[block * 8 + rank_in_block] = my_index;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint my_ordinal2 = 0u;
uint my_index2 = 0u;
if (lane < 64) {
    my_ordinal2 = candidate_ordinals[lane];
    my_index2 = candidate_indices[lane];
}

// Sort one 64-candidate set instead of four duplicate copies. Sequences up to
// 32 are simdgroup-local, so inactive simdgroups can skip them entirely.
if (lane < 64) {
for (uint sequence = 2; sequence <= 32; sequence <<= 1) {
    for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
        uint other_ordinal = simd_shuffle_xor(my_ordinal2, ushort(stride));
        uint other_index = simd_shuffle_xor(my_index2, ushort(stride));

        bool is_lower = (lane & stride) == 0;
        bool lower_wants_better = (lane & sequence) == 0;
        bool want_better = lower_wants_better == is_lower;
        bool other_before_my = laguna_router_ordinal_before(
            other_ordinal, other_index, my_ordinal2, my_index2);
        bool take_other = want_better ? other_before_my : !other_before_my;
        if (take_other) {
            my_ordinal2 = other_ordinal;
            my_index2 = other_index;
        }
    }
}
}

// The first stage of sequence 64 crosses the two active simdgroups. All 256
// threads reach the barrier, but only the 64 live candidates touch memory or
// execute the comparator.
if (lane < 64) {
    xchg_ordinals[lane] = my_ordinal2;
    xchg_indices[lane] = my_index2;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane < 64) {
    uint partner = lane ^ 32u;
    uint other_ordinal = xchg_ordinals[partner];
    uint other_index = xchg_indices[partner];
    bool is_lower = (lane & 32u) == 0;
    bool other_before_my = laguna_router_ordinal_before(
        other_ordinal, other_index, my_ordinal2, my_index2);
    bool take_other = is_lower ? other_before_my : !other_before_my;
    if (take_other) {
        my_ordinal2 = other_ordinal;
        my_index2 = other_index;
    }

    for (uint stride = 16; stride > 0; stride >>= 1) {
        other_ordinal = simd_shuffle_xor(my_ordinal2, ushort(stride));
        other_index = simd_shuffle_xor(my_index2, ushort(stride));
        is_lower = (lane & stride) == 0;
        other_before_my = laguna_router_ordinal_before(
            other_ordinal, other_index, my_ordinal2, my_index2);
        take_other = is_lower ? other_before_my : !other_before_my;
        if (take_other) {
            my_ordinal2 = other_ordinal;
            my_index2 = other_index;
        }
    }
}

\(epilogue)
"""
}

private let lagunaPrefillRouterTournamentKernel = MLXFast.metalKernel(
    name: "laguna_prefill_router_tournament_v1",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaPrefillRouterTournamentKernelSource(normalizing: false),
    header: lagunaDecodeRouterTop8Header,
    ensureRowContiguous: true
)

private let lagunaPrefillRouterTournamentNormalizingKernel = MLXFast.metalKernel(
    name: "laguna_prefill_router_tournament_norm_v1",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaPrefillRouterTournamentKernelSource(normalizing: true),
    header: lagunaDecodeRouterTop8Header,
    ensureRowContiguous: true
)

private let lagunaPrefillRouterTournamentOrdinalKernel = MLXFast.metalKernel(
    name: "laguna_prefill_router_tournament_ordinal_active64_v2",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaPrefillRouterTournamentOrdinalKernelSource(normalizing: false),
    header: lagunaDecodeRouterOrdinalHeader,
    ensureRowContiguous: true
)

private let lagunaPrefillRouterTournamentOrdinalNormalizingKernel = MLXFast.metalKernel(
    name: "laguna_prefill_router_tournament_ordinal_norm_active64_v2",
    inputNames: ["logits", "correction_bias"],
    outputNames: ["router_indices", "router_scores"],
    source: lagunaPrefillRouterTournamentOrdinalKernelSource(normalizing: true),
    header: lagunaDecodeRouterOrdinalHeader,
    ensureRowContiguous: true
)

func lagunaPrefillRouterTournamentAcceptedForTesting(
    logits: MLXArray, correctionBias: MLXArray, rows: Int, normalizing: Bool
) -> (MLXArray, MLXArray) {
    precondition(logits.dtype == .bfloat16 || logits.dtype == .float32)
    precondition(correctionBias.dtype == .float32)
    precondition(logits.size == rows * 256)
    precondition(correctionBias.size == 256)

    let kernel =
        normalizing
        ? lagunaPrefillRouterTournamentNormalizingKernel : lagunaPrefillRouterTournamentKernel
    let outputs = kernel(
        [logits, correctionBias],
        grid: (256, rows, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, rows, 8], [1, rows, 8]],
        outputDTypes: [.uint32, .float32]
    )
    return (outputs[0], outputs[1])
}

func lagunaPrefillRouterTournamentOrdinalForTesting(
    logits: MLXArray, correctionBias: MLXArray, rows: Int, normalizing: Bool
) -> (MLXArray, MLXArray) {
    precondition(logits.dtype == .bfloat16 || logits.dtype == .float32)
    precondition(correctionBias.dtype == .float32)
    precondition(logits.size == rows * 256)
    precondition(correctionBias.size == 256)

    let kernel =
        normalizing
        ? lagunaPrefillRouterTournamentOrdinalNormalizingKernel
        : lagunaPrefillRouterTournamentOrdinalKernel
    let outputs = kernel(
        [logits, correctionBias],
        grid: (256, rows, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, rows, 8], [1, rows, 8]],
        outputDTypes: [.uint32, .float32]
    )
    return (outputs[0], outputs[1])
}

private func lagunaPrefillRouterTournament(
    logits: MLXArray, correctionBias: MLXArray, rows: Int, normalizing: Bool
) -> (MLXArray, MLXArray) {
    if lagunaDecodeRouterOrdinalEnabled {
        return lagunaPrefillRouterTournamentOrdinalForTesting(
            logits: logits,
            correctionBias: correctionBias,
            rows: rows,
            normalizing: normalizing
        )
    }
    return lagunaPrefillRouterTournamentAcceptedForTesting(
        logits: logits,
        correctionBias: correctionBias,
        rows: rows,
        normalizing: normalizing
    )
}

private let lagunaPrefillRouterTournamentEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_PREFILL_ROUTER_TOURNAMENT"] != "0"





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





    func callAsFunction(_ x: MLXArray, logits: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let projectedLogits = logits ?? x.matmul(weight.T)
        let inds: MLXArray
        var weights: MLXArray
        if lagunaPrefillRouterTournamentEnabled,
            routerLogitSoftcapping == 0,
            topK == 8,
            projectedLogits.dtype == .bfloat16,
            projectedLogits.ndim == 3,
            projectedLogits.dim(0) == 1,
            projectedLogits.dim(1) > 1,
            projectedLogits.dim(2) == 256,
            eScoreCorrectionBias.size == 256
        {
            lagunaTrace("prefill router tournament")
            return lagunaPrefillRouterTournament(
                logits: projectedLogits,
                correctionBias: eScoreCorrectionBias.asType(.float32),
                rows: projectedLogits.dim(1),
                normalizing: normTopkProb
            )
        }
        if lagunaPrefillRouterTop8Enabled,
            routerLogitSoftcapping == 0,
            topK == 8,
            projectedLogits.dtype == .bfloat16,
            projectedLogits.ndim == 3,
            projectedLogits.dim(0) == 1,
            projectedLogits.dim(1) > 1,
            projectedLogits.dim(2) == 256,
            eScoreCorrectionBias.size == 256
        {
            lagunaTrace("prefill router top8")
            return lagunaPrefillRouterTop8(
                logits: projectedLogits,
                correctionBias: eScoreCorrectionBias.asType(.float32),
                rows: projectedLogits.dim(1),
                normalizing: normTopkProb
            )
        }
        if lagunaDecodeRouterTop8Enabled,
            lagunaDecodeRouterCastSinkEnabled,
            routerLogitSoftcapping == 0,
            projectedLogits.dtype == .bfloat16,
            projectedLogits.size == 256, topK == 8,
            eScoreCorrectionBias.size == 256
        {


            let sinkNormalization = normTopkProb && lagunaDecodeRouterNormSinkEnabled
            lagunaTrace(
                sinkNormalization
                    ? "decode router top8 (cast sink + norm sink)"
                    : "decode router top8 (cast sink)")
            (inds, weights) = lagunaDecodeRouterTop8(
                logits: projectedLogits,
                correctionBias: eScoreCorrectionBias.asType(.float32),
                normalizing: sinkNormalization
            )
            if sinkNormalization {
                return (inds, weights)
            }
        } else {
            var logits = projectedLogits.asType(.float32)
            if routerLogitSoftcapping > 0 {
                logits = tanh(logits / routerLogitSoftcapping) * routerLogitSoftcapping
            }
            if lagunaDecodeRouterTop8Enabled,
                logits.size == 256, topK == 8,
                eScoreCorrectionBias.size == 256
            {

                lagunaTrace("decode router top8 (fp32 logits)")
                (inds, weights) = lagunaDecodeRouterTop8(
                    logits: logits,
                    correctionBias: eScoreCorrectionBias.asType(.float32)
                )
            } else {
                let scores = sigmoid(logits)
                let scoresForChoice =
                    scores + eScoreCorrectionBias.asType(scores.dtype)
                inds =
                    argPartition(-scoresForChoice, kth: topK - 1, axis: -1)[
                        .ellipsis, ..<topK]
                weights = takeAlong(scores, inds, axis: -1)
            }
        }
        if normTopkProb {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
        return (inds, weights)
    }
}





















private let lagunaPrefillMoETailKernel = MLXFast.metalKernel(
    name: "laguna_prefill_moe_tail_bf16_v1",
    inputNames: ["expert_outputs", "router_weights", "shared_output", "residual"],
    outputNames: ["output"],
    source: """
constexpr uint hidden = 2048;
constexpr uint experts = 8;
constexpr uint n_cols = 4;

uint row = thread_position_in_grid.y;
uint col = thread_position_in_grid.x * n_cols;

const device bfloat* expert_row =
    expert_outputs + (row * experts) * hidden + col;
const device float* weight_row = router_weights + row * experts;

bfloat expert_weights[experts];
for (uint e = 0; e < experts; ++e) {
    expert_weights[e] = bfloat(weight_row[e]);
}

for (uint i = 0; i < n_cols; ++i) {
    bfloat total = bfloat(0);
    for (uint e = 0; e < experts; ++e) {
        bfloat product =
            bfloat(expert_row[e * hidden + i] * expert_weights[e]);
        total = bfloat(product + total);
    }
    bfloat scaled = bfloat(total * bfloat(2.5f));
    bfloat r2 = bfloat(scaled + shared_output[row * hidden + col + i]);
    output[row * hidden + col + i] =
        bfloat(residual[row * hidden + col + i] + r2);
}
""",
    ensureRowContiguous: true
)







private let lagunaPrefillSortedMoETailKernel = MLXFast.metalKernel(
    name: "laguna_prefill_sorted_moe_tail_bf16_v1",
    inputNames: [
        "sorted_expert_outputs", "inverse_order", "router_weights",
        "shared_output", "residual",
    ],
    outputNames: ["output"],
    source: """
constexpr uint hidden = 2048;
constexpr uint experts = 8;
constexpr uint n_cols = 4;

uint row = thread_position_in_grid.y;
uint col = thread_position_in_grid.x * n_cols;
const device float* weight_row = router_weights + row * experts;

bfloat expert_weights[experts];
uint sorted_rows[experts];
for (uint e = 0; e < experts; ++e) {
    expert_weights[e] = bfloat(weight_row[e]);
    sorted_rows[e] = inverse_order[row * experts + e];
}

for (uint i = 0; i < n_cols; ++i) {
    bfloat total = bfloat(0);
    for (uint e = 0; e < experts; ++e) {
        bfloat product = bfloat(
            sorted_expert_outputs[sorted_rows[e] * hidden + col + i] *
            expert_weights[e]);
        total = bfloat(product + total);
    }
    bfloat scaled = bfloat(total * bfloat(2.5f));
    bfloat r2 = bfloat(scaled + shared_output[row * hidden + col + i]);
    output[row * hidden + col + i] =
        bfloat(residual[row * hidden + col + i] + r2);
}
""",
    ensureRowContiguous: true
)

private func lagunaPrefillMoETail(
    expertOutputs: MLXArray,
    routerWeights: MLXArray,
    sharedOutput: MLXArray,
    residual: MLXArray
) -> MLXArray {
    let rows = expertOutputs.dim(1)
    precondition(expertOutputs.dtype == .bfloat16)
    precondition(
        expertOutputs.dims(1, rows, LagunaConstants.numExpertsPerTok, LagunaConstants.hiddenSize))
    precondition(routerWeights.dtype == .float32)
    precondition(routerWeights.dims(1, rows, LagunaConstants.numExpertsPerTok))
    precondition(sharedOutput.dtype == .bfloat16)
    precondition(sharedOutput.dims(1, rows, LagunaConstants.hiddenSize))
    precondition(residual.dtype == .bfloat16)
    precondition(residual.dims(1, rows, LagunaConstants.hiddenSize))

    return lagunaPrefillMoETailKernel(
        [expertOutputs, routerWeights, sharedOutput, residual],
        grid: (LagunaConstants.hiddenSize / 4, rows, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, rows, LagunaConstants.hiddenSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

private func lagunaPrefillSortedMoETail(
    sortedExpertOutputs: MLXArray,
    inverseOrder: MLXArray,
    routerWeights: MLXArray,
    sharedOutput: MLXArray,
    residual: MLXArray
) -> MLXArray {
    let rows = routerWeights.dim(1)
    precondition(sortedExpertOutputs.dtype == .bfloat16)
    precondition(
        sortedExpertOutputs.size
            == rows * LagunaConstants.numExpertsPerTok * LagunaConstants.hiddenSize)
    precondition(inverseOrder.dtype == .uint32)
    precondition(inverseOrder.size == rows * LagunaConstants.numExpertsPerTok)
    precondition(routerWeights.dtype == .float32)
    precondition(routerWeights.dims(1, rows, LagunaConstants.numExpertsPerTok))
    precondition(sharedOutput.dtype == .bfloat16)
    precondition(sharedOutput.dims(1, rows, LagunaConstants.hiddenSize))
    precondition(residual.dtype == .bfloat16)
    precondition(residual.dims(1, rows, LagunaConstants.hiddenSize))

    return lagunaPrefillSortedMoETailKernel(
        [
            sortedExpertOutputs, inverseOrder, routerWeights, sharedOutput,
            residual,
        ],
        grid: (LagunaConstants.hiddenSize / 4, rows, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[1, rows, LagunaConstants.hiddenSize]],
        outputDTypes: [.bfloat16]
    )[0]
}




private func lagunaInterleavedSwiGLU(
    _ gateUp: MLXArray,
    split: Int
) -> MLXArray {
    precondition(split % 32 == 0)
    precondition(gateUp.dim(-1) == 2 * split)

    var tiledShape = gateUp.shape
    tiledShape.removeLast()
    tiledShape.append(split / 32)
    tiledShape.append(64)
    let tiled = gateUp.reshaped(tiledShape)

    var halfShape = gateUp.shape
    halfShape[halfShape.count - 1] = split
    let gate = tiled[.ellipsis, 0 ..< 32].reshaped(halfShape)
    let up = tiled[.ellipsis, 32 ..< 64].reshaped(halfShape)
    return compiledSiluProduct(gate, up)
}












private func lagunaFusedSortedRoutedGateUp(
    _ x: MLXArray,
    indices: MLXArray,
    fusedWeight: MLXArray,
    fusedScales: MLXArray,
    pairwiseScales: MLXArray?,
    split: Int,
    downProj: SwitchLinear,
    downWeight: MLXArray?,
    downPairwiseScales: MLXArray?,
    deferUnsort: Bool
) -> (output: MLXArray, inverseOrder: MLXArray?) {

    var sortedX = MLX.expandedDimensions(x, axes: [-2, -3])




    let doSort = indices.size >= 64

    var idx = indices
    var inverseOrder = MLXArray()


    if doSort {
        (sortedX, idx, inverseOrder) = gatherSort(x: sortedX, indices: indices)
    }











    let gateUp = MLX.gatherQuantizedMM(
        sortedX,
        fusedWeight,
        scales: pairwiseScales ?? fusedScales,
        biases: nil,
        rhsIndices: idx,
        transpose: true,
        groupSize: 16,
        bits: 4,
        mode: .nvfp4,
        sortedIndices: doSort
    )
    let activated: MLXArray
    if lagunaExpertAlignedGatherEnabled {



        var activatedShape = gateUp.shape
        activatedShape[activatedShape.count - 1] = split
        activated = gateUp.reshaped([-1])[0 ..< gateUp.size / 2]
            .reshaped(activatedShape)
    } else {
        activated = lagunaInterleavedSwiGLU(gateUp, split: split)
    }




    var result: MLXArray
    if let downWeight, let downPairwiseScales {
        result = MLX.gatherQuantizedMM(
            activated,
            downWeight,
            scales: downPairwiseScales,
            biases: nil,
            rhsIndices: idx,
            transpose: true,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4,
            sortedIndices: doSort
        )
    } else {
        result = downProj(activated, idx, sortedIndices: doSort)
    }

    if doSort && !deferUnsort {
        result = scatterUnsort(x: result, invOrder: inverseOrder, shape: indices.shape)
    }
    if doSort && deferUnsort {
        return (result, inverseOrder)
    }

    return (MLX.squeezed(result, axis: -2), nil)
}

final class LagunaRuntimeSparseMoEBlock: Module, UnaryLayer {
    let routedScalingFactor: Float

    @ModuleInfo(key: "gate") var gate: LagunaRuntimeMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: LagunaRuntimeMLP









    var _fusedRoutedGateUpWeight: MLXArray?
    var _fusedRoutedGateUpScales: MLXArray?


    var _fusedRoutedGateUpPairwiseScales: MLXArray?
    var _fusedRoutedGateUpSplit: Int = 0
    var _routedDownProj: SwitchLinear?
    var _routedDownWeight: MLXArray?





    var _routedDownScales: MLXArray?


    var _routedDownPairwiseScales: MLXArray?






    var _packedRoutedGateUpBank: MLXArray?








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
            gateWeight.sameDims(upWeight),
            gateScales.sameDims(upScales),
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




        let experts = gateWeight.dim(0)
        let split = gateWeight.dim(1)
        let pairRows = 32
        let weightDepth = gateWeight.dim(2)
        let scaleDepth = gateScales.dim(2)
        let gateWeightTiles = gateWeight.reshaped(
            [experts, split / pairRows, pairRows, weightDepth])
        let upWeightTiles = upWeight.reshaped(
            [experts, split / pairRows, pairRows, weightDepth])
        let gateScaleTiles = gateScales.reshaped(
            [experts, split / pairRows, pairRows, scaleDepth])
        let upScaleTiles = upScales.reshaped(
            [experts, split / pairRows, pairRows, scaleDepth])
        let fusedWeight = concatenated(
            [gateWeightTiles, upWeightTiles], axis: 2
        ).reshaped([experts, 2 * split, weightDepth])
        let fusedScales = concatenated(
            [gateScaleTiles, upScaleTiles], axis: 2
        ).reshaped([experts, 2 * split, scaleDepth])
        _fusedRoutedGateUpWeight = fusedWeight
        _fusedRoutedGateUpScales = fusedScales
        _fusedRoutedGateUpSplit = split
        _routedDownProj = downModule
        _routedDownWeight = downWeight
        var prepared = [fusedWeight, fusedScales]



        if let halvedDown = lagunaHalvedGroup32ScalePlane(
            downScales, allowedFlatPairs: [0])
        {
            _routedDownScales = halvedDown
            prepared.append(halvedDown)
            if lagunaPrefillExpertDownPairwiseScalesEnabled,
                lagunaExpertAlignedGatherEnabled,
                let pairwiseDown = lagunaPackedPrefillDownScaleView(halvedDown)
            {
                _routedDownPairwiseScales = pairwiseDown
                prepared.append(pairwiseDown)
            }
        }
        prepared.append(
            contentsOf: preparePackedRoutedGateUpBank(
                fusedScales: fusedScales,
                experts: experts,
                split: split))
        if lagunaPrefillExpertPairwiseScalesEnabled,
            lagunaExpertAlignedGatherEnabled,
            let packedScales = _packedRoutedGateUpBank,
            let pairwiseView = lagunaPackedPrefillScaleView(packedScales)
        {
            _fusedRoutedGateUpPairwiseScales = pairwiseView
            prepared.append(pairwiseView)
        }
        return prepared
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
        forward(x, residual: nil, routerLogits: nil)
    }

    func callAsFunction(
        _ x: MLXArray, residual: MLXArray, routerLogits: MLXArray? = nil,
        routerKeys: MLXArray? = nil
    ) -> MLXArray {
        forward(x, residual: residual, routerLogits: routerLogits, routerKeys: routerKeys)
    }

    private func forward(
        _ x: MLXArray, residual: MLXArray?, routerLogits: MLXArray?,
        routerKeys: MLXArray? = nil
    ) -> MLXArray {
        let (inds, weights) = gate(x, logits: routerLogits)
        var y: MLXArray
        var routedAlreadyReduced = false
        var sortedTailInverseOrder: MLXArray?
        if let fusedWeight = _fusedRoutedGateUpWeight,
            let fusedScales = _fusedRoutedGateUpScales,
            let downProj = _routedDownProj,
            x.dim(1) == 1, inds.size < 64
        {













            let activated: MLXArray




            var mergedSharedActivated: MLXArray?
            if lagunaFusedRoutedSwiGLUQMVEnabled,
                x.dtype == .bfloat16,
                x.dims(1, 1, LagunaConstants.hiddenSize),
                fusedWeight.dtype == .uint32,
                fusedScales.dtype == .uint8,
                inds.dtype == .uint32,
                inds.dims(1, 1, LagunaConstants.numExpertsPerTok),
                _fusedRoutedGateUpSplit == LagunaConstants.moeIntermediateSize
            {
                if lagunaPackedScalesEnabled,
                    let packedBank = _packedRoutedGateUpBank
                {
                    lagunaPackedScalesLog.note(
                        "active", "routed swiglu qmv packed dispatch")
                    if lagunaRouterPrecomputedKeysEnabled,
                        let routerKeys,
                        routerKeys.dtype == .uint32,
                        routerKeys.size == LagunaConstants.numExperts,
                        gate.topK == LagunaConstants.numExpertsPerTok,
                        gate.routerLogitSoftcapping == 0,
                        gate.eScoreCorrectionBias.size == LagunaConstants.numExperts
                    {
                        lagunaTrace("routed gate/up QMV + SwiGLU (packed, producer keys)")
                        activated = lagunaRoutedSwiGLUQMVPackedTop8(
                            x,
                            fusedWeight: fusedWeight,
                            packedScales: packedBank,
                            routerKeys: routerKeys
                        )
                    } else {
                        lagunaTrace("routed gate/up QMV + SwiGLU (packed scales)")
                        activated = lagunaRoutedSwiGLUQMVPacked(
                            x,
                            fusedWeight: fusedWeight,
                            packedScales: packedBank,
                            indices: inds
                        )
                    }
                } else {
                    if lagunaPackedScalesEnabled {
                        lagunaPackedScalesLog.note(
                            "inactive",
                            "routed swiglu qmv packed (bank missing; stock kernel dispatched)")
                    }
                    lagunaTrace("routed gate/up QMV + SwiGLU")
                    activated = lagunaRoutedSwiGLUQMV(
                        x,
                        fusedWeight: fusedWeight,
                        fusedScales: fusedScales,
                        indices: inds
                    )
                }
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
                activated = lagunaInterleavedSwiGLU(
                    gateUp, split: _fusedRoutedGateUpSplit)
            }
            if lagunaFusedRoutedSharedDownResidualEnabled,
                let residual,
                let downWeight = _routedDownWeight,
                let downScales = _routedDownScales,
                let sharedInputs = sharedExpert.fusedSharedDownInputs(
                    x, sharedActivation: mergedSharedActivated),
                activated.dtype == .bfloat16,
                activated.dims(1, 1, LagunaConstants.numExpertsPerTok, 1,
                    LagunaConstants.moeIntermediateSize),
                downWeight.dtype == .uint32,
                downWeight.dims(LagunaConstants.numExperts, LagunaConstants.hiddenSize,
                    LagunaConstants.moeIntermediateSize / 8),
                downScales.dtype == .uint8,
                downScales.size == lagunaRoutedDownScaleBytes,
                weights.dtype == .float32,
                weights.dims(1, 1, LagunaConstants.numExpertsPerTok),
                routedScalingFactor == Float(LagunaConstants.moeRoutedScalingFactor),
                residual.dtype == .bfloat16,
                residual.dims(1, 1, LagunaConstants.hiddenSize)
            {
                lagunaTrace("routed+shared down residual")
                return lagunaRoutedSharedDownResidual(
                    routedActivated: activated,
                    routedDownWeight: downWeight,
                    routedDownScales: downScales,
                    indices: inds,
                    routerWeights: weights,
                    sharedActivated: sharedInputs.activated,
                    sharedDownWeight: sharedInputs.downWeight,
                    sharedDownScales: sharedInputs.downScales,
                    residual: residual
                )
            } else if lagunaFusedRoutedDownReduceEnabled,
                let downWeight = _routedDownWeight,
                let downScales = _routedDownScales,
                activated.dtype == .bfloat16,
                activated.dims(1, 1, LagunaConstants.numExpertsPerTok, 1,
                    LagunaConstants.moeIntermediateSize),
                downWeight.dtype == .uint32,
                downWeight.dims(LagunaConstants.numExperts, LagunaConstants.hiddenSize,
                    LagunaConstants.moeIntermediateSize / 8),
                downScales.dtype == .uint8,
                downScales.size == lagunaRoutedDownScaleBytes,
                weights.dtype == .float32,
                weights.dims(1, 1, LagunaConstants.numExpertsPerTok),
                routedScalingFactor == Float(LagunaConstants.moeRoutedScalingFactor)
            {
                lagunaTrace("routed down reduce")
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













            if lagunaPrefillFusedRoutedGateUpEnabled,
                let fusedWeight = _fusedRoutedGateUpWeight,
                let fusedScales = _fusedRoutedGateUpScales,
                let downProj = _routedDownProj,
                x.dim(1) > 1,
                inds.size >= 64,
                fusedWeight.dtype == .uint32,
                fusedScales.dtype == .uint8,
                _fusedRoutedGateUpSplit == LagunaConstants.moeIntermediateSize
            {
                lagunaTrace("prefill fused routed gate/up")
                let pairwiseScales =
                    lagunaPrefillExpertPairwiseScalesAdmitted(routedRows: inds.size)
                    ? _fusedRoutedGateUpPairwiseScales : nil
                let pairwiseDownScales =
                    lagunaPrefillExpertPairwiseScalesAdmitted(routedRows: inds.size)
                    ? _routedDownPairwiseScales : nil
                if lagunaPrefillExpertPairwiseScalesEnabled {
                    lagunaPackedScalesLog.note(
                        pairwiseScales == nil ? "inactive" : "active",
                        "packed routed gate/up prefill scale view consumed")
                }
                if lagunaPrefillExpertDownPairwiseScalesEnabled {
                    lagunaPackedScalesLog.note(
                        pairwiseDownScales == nil ? "inactive" : "active",
                        "packed routed down prefill scale view consumed")
                }
                let routed = lagunaFusedSortedRoutedGateUp(
                    x,
                    indices: inds,
                    fusedWeight: fusedWeight,
                    fusedScales: fusedScales,
                    pairwiseScales: pairwiseScales,
                    split: _fusedRoutedGateUpSplit,
                    downProj: downProj,
                    downWeight: _routedDownWeight,
                    downPairwiseScales: pairwiseDownScales,
                    deferUnsort:
                        lagunaPrefillSortedMoETailEnabled
                        && lagunaPrefillMoETailEnabled
                        && residual != nil
                )
                y = routed.output
                sortedTailInverseOrder = routed.inverseOrder
            } else {
                y = switchMLP(x, inds)
            }
            if let inverseOrder = sortedTailInverseOrder,
                lagunaPrefillMoETailEnabled,
                let residual,
                x.dim(1) > 1,
                y.dtype == .bfloat16,
                y.size
                    == x.dim(1) * LagunaConstants.numExpertsPerTok
                        * LagunaConstants.hiddenSize,
                inverseOrder.dtype == .uint32,
                inverseOrder.size == x.dim(1) * LagunaConstants.numExpertsPerTok,
                weights.dtype == .float32,
                weights.dims(1, x.dim(1), LagunaConstants.numExpertsPerTok),
                routedScalingFactor == Float(LagunaConstants.moeRoutedScalingFactor),
                residual.dtype == .bfloat16,
                residual.dims(1, x.dim(1), LagunaConstants.hiddenSize)
            {
                let sharedOut = sharedExpert(x)
                if sharedOut.dtype == .bfloat16, sharedOut.sameDims(residual) {
                    lagunaTrace("prefill sorted moe tail")
                    return lagunaPrefillSortedMoETail(
                        sortedExpertOutputs: y,
                        inverseOrder: inverseOrder,
                        routerWeights: weights,
                        sharedOutput: sharedOut,
                        residual: residual
                    )
                }


                y = MLX.squeezed(
                    scatterUnsort(x: y, invOrder: inverseOrder, shape: inds.shape),
                    axis: -2)
                var reduced = weightedExpertSum(y, weights.asType(y.dtype))
                if routedScalingFactor != 1 {
                    reduced = reduced * routedScalingFactor
                }
                return residual + (reduced + sharedOut)
            }
            if let inverseOrder = sortedTailInverseOrder {



                y = MLX.squeezed(
                    scatterUnsort(x: y, invOrder: inverseOrder, shape: inds.shape),
                    axis: -2)
                sortedTailInverseOrder = nil
            }
            if lagunaPrefillMoETailEnabled,
                let residual,
                x.dim(1) > 1,
                y.dtype == .bfloat16,
                y.ndim == 4,
                y.dim(0) == 1,
                y.dim(1) == x.dim(1),
                y.dim(2) == LagunaConstants.numExpertsPerTok,
                y.dim(3) == LagunaConstants.hiddenSize,
                weights.dtype == .float32,
                weights.dims(1, x.dim(1), LagunaConstants.numExpertsPerTok),
                routedScalingFactor == Float(LagunaConstants.moeRoutedScalingFactor),
                residual.dtype == .bfloat16,
                residual.dims(1, x.dim(1), LagunaConstants.hiddenSize)
            {
                let sharedOut = sharedExpert(x)
                if sharedOut.dtype == .bfloat16, sharedOut.sameDims(residual) {
                    lagunaTrace("prefill moe tail")
                    return lagunaPrefillMoETail(
                        expertOutputs: y,
                        routerWeights: weights,
                        sharedOutput: sharedOut,
                        residual: residual
                    )
                }


                var reduced = weightedExpertSum(y, weights.asType(y.dtype))
                if routedScalingFactor != 1 {
                    reduced = reduced * routedScalingFactor
                }
                return residual + (reduced + sharedOut)
            }
        }
        if !routedAlreadyReduced {
            y = weightedExpertSum(y, weights.asType(y.dtype))
            if routedScalingFactor != 1 {
                y = y * routedScalingFactor
            }
        }
        if let residual,
            let output = sharedExpert.fusedSharedDownResidual(
                x,
                routed: y,
                residual: residual
            )
        {
            return output
        }
        let r2 = y + sharedExpert(x)
        return residual.map { $0 + r2 } ?? r2
    }
}



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
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        qkRoPEAngles: MLXArray? = nil,
        qkRoPEOffsets: MLXArray? = nil
    ) -> MLXArray {
        let r = selfAttn(
            x,
            inputNorm: inputLayerNorm,
            mask: mask,
            cache: cache,
            qkRoPEAngles: qkRoPEAngles,
            qkRoPEOffsets: qkRoPEOffsets
        )
        let h: MLXArray
        let normalized: MLXArray
        var routerLogits: MLXArray?
        var routerKeys: MLXArray?
        if lagunaFusedResidualRMSNormRouterEnabled,
            x.dtype == .bfloat16, r.dtype == .bfloat16,
            postAttentionLayerNorm.weight.dtype == .bfloat16,
            x.dims(1, 1, LagunaConstants.hiddenSize), x.sameDims(r),
            let sparse = mlp as? LagunaRuntimeSparseMoEBlock,
            sparse.gate.weight.dtype == .bfloat16,
            sparse.gate.weight.dims(LagunaConstants.numExperts, LagunaConstants.hiddenSize)
        {
            let fused = lagunaResidualRMSNormRouter(
                residual: x,
                branch: r,
                weight: postAttentionLayerNorm.weight,
                routerWeight: sparse.gate.weight,
                correctionBias: sparse.gate.eScoreCorrectionBias)
            h = fused.summed
            normalized = fused.normalized
            routerLogits = fused.routerLogits
            routerKeys = fused.routerKeys
        } else if lagunaFusedResidualRMSNormEnabled,
            x.dtype == .bfloat16, r.dtype == .bfloat16,
            postAttentionLayerNorm.weight.dtype == .bfloat16,
            x.sameDims(r), x.dim(-1) == LagunaConstants.hiddenSize,
            x.size == LagunaConstants.hiddenSize
        {
            lagunaTrace("residual+rmsnorm")
            (h, normalized) = lagunaResidualRMSNorm(
                residual: x, branch: r, weight: postAttentionLayerNorm.weight)
        } else if lagunaPrefillFusedResidualRMSNormEnabled,
            x.dtype == .bfloat16, r.dtype == .bfloat16,
            postAttentionLayerNorm.weight.dtype == .bfloat16,
            x.sameDims(r), x.ndim == 3, x.dim(0) == 1,
            x.dim(-1) == LagunaConstants.hiddenSize,
            x.dim(1) > 1
        {




            lagunaTrace("prefill residual+rmsnorm")
            (h, normalized) = lagunaResidualRMSNorm(
                residual: x, branch: r, weight: postAttentionLayerNorm.weight)
        } else {
            h = x + r
            normalized = postAttentionLayerNorm(h)
        }
        if (
            lagunaFusedSharedDownResidualEnabled ||
                lagunaFusedRoutedSharedDownResidualEnabled
        ),
            normalized.dtype == .bfloat16,
            normalized.dims(1, 1, LagunaConstants.hiddenSize),
            h.dtype == .bfloat16,
            h.sameDims(normalized),
            let sparse = mlp as? LagunaRuntimeSparseMoEBlock
        {
            return sparse(
                normalized, residual: h, routerLogits: routerLogits,
                routerKeys: routerKeys)
        }





        if lagunaPrefillMoETailEnabled,
            x.dim(1) > 1,
            let sparse = mlp as? LagunaRuntimeSparseMoEBlock
        {
            return sparse(
                normalized, residual: h, routerLogits: routerLogits,
                routerKeys: routerKeys)
        }


        if let dense = mlp as? LagunaRuntimeMLP,
            let fused = dense.fusedDenseDownResidual(normalized, residual: h)
        {
            return fused
        }
        let r2 = mlp(normalized)
        return h + r2
    }



    func callLastPrefillRow(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        if lagunaTerminalPrefillFusionEnabled {


            let normalized = inputLayerNorm(x)
            let r = selfAttn.callLastPrefillRow(normalized, cache: cache)
            let lastResidual = lagunaLastTokenHidden(x)
            let h: MLXArray
            let normalizedAfterAttention: MLXArray
            var routerLogits: MLXArray?
            var routerKeys: MLXArray?
            if lagunaFusedResidualRMSNormRouterEnabled,
                lastResidual.dtype == .bfloat16, r.dtype == .bfloat16,
                postAttentionLayerNorm.weight.dtype == .bfloat16,
                lastResidual.dims(1, 1, LagunaConstants.hiddenSize),
                lastResidual.sameDims(r),
                let sparse = mlp as? LagunaRuntimeSparseMoEBlock,
                sparse.gate.weight.dtype == .bfloat16,
                sparse.gate.weight.dims(LagunaConstants.numExperts, LagunaConstants.hiddenSize)
            {
                let fused = lagunaResidualRMSNormRouter(
                    residual: lastResidual,
                    branch: r,
                    weight: postAttentionLayerNorm.weight,
                    routerWeight: sparse.gate.weight,
                    correctionBias: sparse.gate.eScoreCorrectionBias)
                h = fused.summed
                normalizedAfterAttention = fused.normalized
                routerLogits = fused.routerLogits
                routerKeys = fused.routerKeys
            } else if lagunaFusedResidualRMSNormEnabled,
                lastResidual.dtype == .bfloat16, r.dtype == .bfloat16,
                postAttentionLayerNorm.weight.dtype == .bfloat16,
                lastResidual.sameDims(r),
                lastResidual.dim(-1) == LagunaConstants.hiddenSize,
                lastResidual.size == LagunaConstants.hiddenSize
            {
                lagunaTrace("terminal prefill residual+rmsnorm")
                (h, normalizedAfterAttention) = lagunaResidualRMSNorm(
                    residual: lastResidual, branch: r,
                    weight: postAttentionLayerNorm.weight)
            } else {
                h = lastResidual + r
                normalizedAfterAttention = postAttentionLayerNorm(h)
            }
            if (
                lagunaFusedSharedDownResidualEnabled ||
                    lagunaFusedRoutedSharedDownResidualEnabled
            ),
                normalizedAfterAttention.dtype == .bfloat16,
                normalizedAfterAttention.dims(1, 1, LagunaConstants.hiddenSize),
                h.dtype == .bfloat16,
                h.sameDims(normalizedAfterAttention),
                let sparse = mlp as? LagunaRuntimeSparseMoEBlock
            {
                return sparse(
                    normalizedAfterAttention, residual: h,
                    routerLogits: routerLogits, routerKeys: routerKeys)
            }
            if let dense = mlp as? LagunaRuntimeMLP,
                let fused = dense.fusedDenseDownResidual(
                    normalizedAfterAttention, residual: h)
            {
                return fused
            }
            let r2 = mlp(normalizedAfterAttention)
            return h + r2
        } else {
            let normalized = inputLayerNorm(x)
            let r = selfAttn.callLastPrefillRow(normalized, cache: cache)
            let h = lagunaLastTokenHidden(x) + r
            let r2 = mlp(postAttentionLayerNorm(h))
            return h + r2
        }
    }
}







private let lagunaDecodeEmbeddingRoPEAtlasKernel = MLXFast.metalKernel(
    name: "laguna_decode_embedding_rope_atlas_bf16_2048_v2",
    inputNames: [
        "tokens", "embedding_weight", "full_atlas", "sliding_atlas",
        "atlas_position",
    ],
    outputNames: ["hidden", "full_angles", "sliding_angles"],
    source: """
constexpr uint hidden_size = 2048;
constexpr uint hidden_vectors = hidden_size / 4;
constexpr uint full_width = 64;
constexpr uint sliding_width = 128;

uint lane = thread_position_in_grid.x;
uint token = uint(tokens[0]);
uint position = uint(atlas_position);

const device vec<bfloat, 4>* embedding_vectors =
    (const device vec<bfloat, 4>*)(
        embedding_weight + token * hidden_size);
device vec<bfloat, 4>* hidden_vectors_out =
    (device vec<bfloat, 4>*)(hidden);
if (lane < hidden_vectors) {
    hidden_vectors_out[lane] = embedding_vectors[lane];
}

if (lane < full_width / 4) {
    const device vec<float, 4>* atlas_vectors =
        (const device vec<float, 4>*)(
            full_atlas + position * full_width);
    ((device vec<float, 4>*)(full_angles))[lane] =
        atlas_vectors[lane];
}
if (lane < sliding_width / 4) {
    const device vec<float, 4>* atlas_vectors =
        (const device vec<float, 4>*)(
            sliding_atlas + position * sliding_width);
    ((device vec<float, 4>*)(sliding_angles))[lane] =
        atlas_vectors[lane];
}
""",
    ensureRowContiguous: true
)

private func lagunaDecodeEmbeddingRoPEAtlas(
    tokens: MLXArray,
    embeddingWeight: MLXArray,
    fullAtlas: MLXArray,
    slidingAtlas: MLXArray,
    position: Int
) -> (hidden: MLXArray, fullAngles: MLXArray, slidingAngles: MLXArray)? {
    guard tokens.dtype == .int32,
        tokens.dims(1, 1),
        embeddingWeight.dtype == .bfloat16,
        embeddingWeight.dims(LagunaConstants.vocabSize, LagunaConstants.hiddenSize),
        fullAtlas.dtype == .float32,
        fullAtlas.dims(1, 1, lagunaRoPEAngleAtlasLength, LagunaConstants.headDim / 2),
        slidingAtlas.dtype == .float32,
        slidingAtlas.dims(1, 1, lagunaRoPEAngleAtlasLength, LagunaConstants.headDim),
        position >= 0, position < lagunaRoPEAngleAtlasLength
    else {
        return nil
    }

    let kernelInputs: [any ScalarOrArray] = [
        tokens,
        embeddingWeight,
        fullAtlas,
        slidingAtlas,
        Int32(position),
    ]
    let outputs = lagunaDecodeEmbeddingRoPEAtlasKernel(
        kernelInputs,
        grid: (512, 1, 1),
        threadGroup: (512, 1, 1),
        outputShapes: [
            [1, 1, LagunaConstants.hiddenSize],
            [1, 1, 1, LagunaConstants.headDim / 2],
            [1, 1, 1, LagunaConstants.headDim],
        ],
        outputDTypes: [.bfloat16, .float32, .float32]
    )
    lagunaTrace("decode embedding+rope atlas")
    return (outputs[0], outputs[1], outputs[2])
}




final class LagunaRuntimeModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [LagunaRuntimeDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let layerTypes: [LagunaLayerType]
    let slidingWindow: Int
    let fullAttentionIdx: Int
    let slidingAttentionIdx: Int
    let _fullRoPEAngleSeed: MLXArray
    let _slidingRoPEAngleSeed: MLXArray
    var _fullRoPEAngleAtlas: MLXArray?
    var _slidingRoPEAngleAtlas: MLXArray?


    let decodeFireMask: UInt64

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
        self._fullRoPEAngleSeed = MLXArray(
            Array(repeating: Float(0.7426255941390991), count: LagunaConstants.headDim / 4)
                + Array(repeating: Float(0), count: LagunaConstants.headDim / 4),
            [1, 1, 1, LagunaConstants.headDim / 2]
        )





        self._slidingRoPEAngleSeed = MLXArray(
            Array(repeating: Float(1), count: LagunaConstants.headDim / 2)
                + Array(repeating: Float(0), count: LagunaConstants.headDim / 2),
            [1, 1, 1, LagunaConstants.headDim]
        )
        self.decodeFireMask =
            switch lagunaDecodeAsyncStage {
            case .off, .norm, .logits:
                0
            case .layer(let idx):
                UInt64(1) << UInt64(idx)
            case .ladder(let n):
                (0..<UInt64(config.numHiddenLayers)).reduce(UInt64(0)) { acc, i in
                    (Int(i) + 1) % n == 0 ? acc | (UInt64(1) << i) : acc
                }
            case .explicit(let mask):
                mask
            }
    }





    func prepareRoPEAngleAtlases() -> [MLXArray] {



        guard lagunaRoPEAngleAtlasEnabled || lagunaPrefillQKNormRoPEEnabled,
            lagunaFusedFullQKNormYaRNEnabled,
            lagunaFusedSlidingQKNormRoPEEnabled,
            layerTypes.contains(.full),
            layerTypes.contains(.sliding)
        else {
            return []
        }
        if let fullAtlas = _fullRoPEAngleAtlas,
            let slidingAtlas = _slidingRoPEAngleAtlas
        {
            return [fullAtlas, slidingAtlas]
        }

        let fullSeed = broadcast(
            _fullRoPEAngleSeed,
            to: [
                1, 1, lagunaRoPEAngleAtlasLength, LagunaConstants.headDim / 2,
            ])
        let slidingSeed = broadcast(
            _slidingRoPEAngleSeed,
            to: [
                1, 1, lagunaRoPEAngleAtlasLength, LagunaConstants.headDim,
            ])
        let fullAtlas = layers[fullAttentionIdx].selfAttn.rope(fullSeed, offset: 0)
        let slidingAtlas = layers[slidingAttentionIdx].selfAttn.rope(
            slidingSeed, offset: 0)
        _fullRoPEAngleAtlas = fullAtlas
        _slidingRoPEAngleAtlas = slidingAtlas
        return [fullAtlas, slidingAtlas]
    }




    private func decodeRoPEAtlasPosition(
        inputs: MLXArray, cache: [KVCache]?
    ) -> Int? {
        guard lagunaRoPEAngleAtlasEnabled || lagunaRoPEAtlasViewsEnabled,
            lagunaFusedFullQKNormYaRNEnabled,
            lagunaFusedSlidingQKNormRoPEEnabled,
            inputs.dtype == .int32,
            inputs.dims(1, 1),
            _fullRoPEAngleAtlas != nil,
            _slidingRoPEAngleAtlas != nil,
            let cache,
            fullAttentionIdx < cache.count,
            slidingAttentionIdx < cache.count
        else {
            return nil
        }

        let fullCache = cache[fullAttentionIdx]
        let slidingCache = cache[slidingAttentionIdx]
        guard type(of: fullCache) == KVCacheSimple.self,
            type(of: slidingCache) == RotatingKVCache.self,
            slidingCache.maxSize == slidingWindow
        else {
            return nil
        }

        let fullPosition = fullCache.offset
        let slidingPosition = slidingCache.offset
        guard fullPosition == slidingPosition,
            fullPosition >= 0,
            fullPosition < lagunaRoPEAngleAtlasLength
        else {
            return nil
        }
        return fullPosition
    }



    private func ropeAngleTable(
        seed: MLXArray, attention: LagunaRuntimeAttention, cache: KVCache?
    ) -> MLXArray {
        if let graphOffset = graphOffsetArray(for: cache) {
            return attention.rope(seed, offset: graphOffset)
        }
        return attention.rope(seed, offset: cache?.offset ?? 0)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h: MLXArray
        var fullRoPEAngles: MLXArray?
        var slidingRoPEAngles: MLXArray?
        var qkRoPEOffsets: MLXArray?
        if lagunaRoPEAngleAtlasEnabled,
            let position = decodeRoPEAtlasPosition(inputs: inputs, cache: cache),
            let fullAtlas = _fullRoPEAngleAtlas,
            let slidingAtlas = _slidingRoPEAngleAtlas,
            let atlasOutputs = lagunaDecodeEmbeddingRoPEAtlas(
                tokens: inputs,
                embeddingWeight: embedTokens.weight,
                fullAtlas: fullAtlas,
                slidingAtlas: slidingAtlas,
                position: position)
        {
            h = atlasOutputs.hidden
            fullRoPEAngles = atlasOutputs.fullAngles
            slidingRoPEAngles = atlasOutputs.slidingAngles
        } else if lagunaRoPEAtlasViewsEnabled,
            let position = decodeRoPEAtlasPosition(inputs: inputs, cache: cache),
            let fullAtlas = _fullRoPEAngleAtlas,
            let slidingAtlas = _slidingRoPEAngleAtlas
        {







            h = embedTokens(inputs)
            fullRoPEAngles = fullAtlas[0..., 0..., position..<(position + 1), 0...]
            slidingRoPEAngles = slidingAtlas[0..., 0..., position..<(position + 1), 0...]
        } else {


            h = embedTokens(inputs)
            let isSingleTokenDecode = h.dim(0) == 1 && h.dim(1) == 1
            fullRoPEAngles =
                lagunaFusedFullQKNormYaRNEnabled && isSingleTokenDecode
                ? ropeAngleTable(
                    seed: _fullRoPEAngleSeed,
                    attention: layers[fullAttentionIdx].selfAttn,
                    cache: cache?[fullAttentionIdx])
                : nil
            slidingRoPEAngles =
                lagunaFusedSlidingQKNormRoPEEnabled && isSingleTokenDecode
                ? ropeAngleTable(
                    seed: _slidingRoPEAngleSeed,
                    attention: layers[slidingAttentionIdx].selfAttn,
                    cache: cache?[slidingAttentionIdx])
                : nil







            if lagunaPrefillQKNormRoPEEnabled, !isSingleTokenDecode,
                h.dim(0) == 1,
                let fullAtlas = _fullRoPEAngleAtlas,
                let slidingAtlas = _slidingRoPEAngleAtlas
            {
                let length = h.dim(1)
                let familyCache =
                    fullAttentionIdx < (cache?.count ?? 0)
                    ? cache?[fullAttentionIdx] : nil
                if graphOffsetArray(for: familyCache) == nil {
                    let offset = familyCache?.offset ?? 0
                    if offset >= 0, offset + length <= lagunaRoPEAngleAtlasLength {
                        fullRoPEAngles = fullAtlas
                        slidingRoPEAngles = slidingAtlas
                        qkRoPEOffsets = MLXArray([Int32(offset)])
                    }
                }
            }
        }





        let fullMask = createAttentionMask(h: h, cache: cache?[fullAttentionIdx])
        let slidingMask = createAttentionMask(
            h: h, cache: cache?[slidingAttentionIdx], windowSize: slidingWindow)

        let isSingleTokenDecode = inputs.dims(1, 1)







        for (i, layer) in layers.enumerated() {
            let isFull = layerTypes[i] == .full
            let mask = isFull ? fullMask : slidingMask
            let qkRoPEAngles = isFull ? fullRoPEAngles : slidingRoPEAngles
            if i == layers.count - 1, h.dim(1) > 1 {
                if case .causal = mask {
                    h = layer.callLastPrefillRow(h, cache: cache?[i])
                } else {
                    h = layer(
                        h,
                        mask: mask,
                        cache: cache?[i],
                        qkRoPEAngles: qkRoPEAngles,
                        qkRoPEOffsets: qkRoPEOffsets
                    )
                    if isSingleTokenDecode, (decodeFireMask >> UInt64(i)) & 1 == 1 {
                        asyncEval(h)
                    }
                }
            } else {
                h = layer(
                    h,
                    mask: mask,
                    cache: cache?[i],
                    qkRoPEAngles: qkRoPEAngles,
                    qkRoPEOffsets: qkRoPEOffsets
                )
                if isSingleTokenDecode, (decodeFireMask >> UInt64(i)) & 1 == 1 {
                    asyncEval(h)
                }
                if lagunaPrefillAsyncLadderStride > 0, h.dim(1) > 1,
                    (i + 1) % lagunaPrefillAsyncLadderStride == 0
                {
                    asyncEval(h)
                }
            }
            lagunaInjectLayerWork(layer: i, isSingleTokenDecode: isSingleTokenDecode)
        }

        return h
    }
}










public final class LagunaRuntimeModel: Module, LanguageModel {
    @ModuleInfo(key: "model") var model: LagunaRuntimeModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let configuration: LagunaConfig





    private var lmHeadPruner: LagunaLmHeadPruner?

    public init(_ config: LagunaConfig) {
        self.configuration = config
        self._model.wrappedValue = LagunaRuntimeModelInner(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
        super.init()





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




        let hidden = model.norm(lagunaLastTokenHidden(fullHidden))
        if case .norm = lagunaDecodeAsyncStage, inputs.dims(1, 1) {
            asyncEval(hidden)
        }

        let result: MLXArray
        if let lmHead {
            if let pruner = lmHeadPruner,
                inputs.dims(1, 1) || lagunaLmHeadPrunePrefillEnabled
            {






                result = pruner.logits(
                    hidden: hidden,
                    lmHeadWeight: lmHead.weight,
                    useFusedRefinement: inputs.dims(1, 1))
            } else {
                result = lmHead(hidden)
            }
        } else {
            result = model.embedTokens.asLinear(hidden)
        }
        if case .logits = lagunaDecodeAsyncStage, inputs.dims(1, 1) {
            asyncEval(result)
        }
        return result
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









    func prepareFusedRuntimeWeights() {
        var fusedArrays = model.prepareRoPEAngleAtlases()
        for layer in model.layers {
            if lagunaUseNativeAffineQKV(layer: layer.selfAttn.layerIdx) {
                fusedArrays.append(
                    contentsOf: layer.selfAttn.prepareNativeAffineQKVWeight())
            }
            if lagunaUseNativeAffineOProj(layer: layer.selfAttn.layerIdx) {
                fusedArrays.append(
                    contentsOf: layer.selfAttn.prepareNativeAffineOProjWeight())
            }
            if lagunaFusedQKVEnabled, let fused = layer.selfAttn.prepareFusedQKVWeight() {
                fusedArrays.append(fused)
            }
            fusedArrays.append(
                contentsOf: layer.selfAttn.prepareLastPrefillProjectionWeights())
            if let sparse = layer.mlp as? LagunaRuntimeSparseMoEBlock {
                if lagunaFusedSharedGateUpEnabled {
                    fusedArrays.append(contentsOf: sparse.sharedExpert.prepareFusedSharedGateUp())
                }
                if lagunaFusedRoutedGateUpEnabled {
                    fusedArrays.append(contentsOf: sparse.prepareFusedRoutedGateUp())
                }
            } else if let dense = layer.mlp as? LagunaRuntimeMLP {
                if lagunaFusedDenseGateUpSwiGLUEnabled,
                    let fused = dense.prepareFusedDenseGateUp()
                {
                    fusedArrays.append(fused)
                }
            }
        }
        if !fusedArrays.isEmpty {
            eval(fusedArrays)
        }





        if lagunaLmHeadPruneEnabled, let lmHead {
            lmHeadPruner = LagunaLmHeadPruner(lmHeadWeight: lmHead.weight)
            if let pruner = lmHeadPruner {
                eval(pruner.residentArrays)
                FileHandle.standardError.write(
                    Data("mlxfast: lm_head prune active (coarse copy resident)\n".utf8))
            }
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        return weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
    }
}













































private func lagunaInjectEnvInt(_ key: String, _ fallback: Int) -> Int {
    guard let raw = ProcessInfo.processInfo.environment[key], let value = Int(raw),
        value >= 0
    else { return fallback }
    return value
}











private let lagunaInjectDecodeSweeps = lagunaInjectEnvInt(
    "DARKBLOOM_INJECT_DECODE_SWEEPS", 0)


private let lagunaInjectSweepPasses = max(
    1, lagunaInjectEnvInt("DARKBLOOM_INJECT_SWEEP_PASSES", 1))

private let lagunaInjectPrefillMatmuls = lagunaInjectEnvInt(
    "DARKBLOOM_INJECT_PREFILL_MATMULS", 0)

private let lagunaInjectDecodeEmpty = lagunaInjectEnvInt(
    "DARKBLOOM_INJECT_DECODE_EMPTY", 0)

private let lagunaInjectPrefillEmpty = lagunaInjectEnvInt(
    "DARKBLOOM_INJECT_PREFILL_EMPTY", 0)



private let lagunaInjectEmptySpread = lagunaInjectEnvInt(
    "DARKBLOOM_INJECT_EMPTY_SPREAD", 1) != 0


private let lagunaInjectEmptyThreadgroups = lagunaInjectEnvInt(
    "DARKBLOOM_INJECT_EMPTY_TG", 160)



private let lagunaInjectEmptyChain = lagunaInjectEnvInt(
    "DARKBLOOM_INJECT_EMPTY_CHAIN", 1) != 0

private let lagunaInjectPoolUInt4 = 1 << 24










private let lagunaInjectSweepThreads = 1 << 16
private let lagunaInjectSweepPerThread = lagunaInjectPoolUInt4 / lagunaInjectSweepThreads
private let lagunaInjectMatmulM = 512
private let lagunaInjectMatmulK = 8192
private let lagunaInjectMatmulN = 2048


let lagunaInjectSweepBytes = lagunaInjectPoolUInt4 * 16 * lagunaInjectSweepPasses

let lagunaInjectMatmulFlops = 2 * lagunaInjectMatmulM * lagunaInjectMatmulN * lagunaInjectMatmulK

private let lagunaInjectSweepKernel = MLXFast.metalKernel(
    name: "laguna_inject_dram_sweep_u4_v2",
    inputNames: ["pool", "control"],
    outputNames: ["sink"],
    source: """
            constexpr uint kThreads = \(lagunaInjectSweepThreads);
            constexpr uint kPerThread = \(lagunaInjectSweepPerThread);
            constexpr uint kMask = \(lagunaInjectPoolUInt4 - 1);
            const device uint4* quads = (const device uint4*)pool;
            uint gid = thread_position_in_grid.x;
            uint idx = (gid + control[0]) & kMask;
            uint passes = control[1];
            uint4 acc = uint4(0u);
            for (uint p = 0; p < passes; ++p) {
                for (uint i = 0; i < kPerThread; ++i) {
                    acc ^= quads[idx];
                    idx = (idx + kThreads) & kMask;
                }
            }
            uint folded = acc.x ^ acc.y ^ acc.z ^ acc.w;
            if (folded == 0xFFFFFFFFu) {
                sink[gid & 255u] = folded;
            }
        """,
    ensureRowContiguous: true
)







private let lagunaInjectEmptyKernel = MLXFast.metalKernel(
    name: "laguna_inject_empty_dispatch_v1",
    inputNames: ["control", "prev"],
    outputNames: ["sink"],
    source: """
            uint gid = thread_position_in_grid.x;
            if (control[0] == 0xFFFFFFFFu) {
                sink[gid & 255u] = gid + prev[0];
            }
        """,
    ensureRowContiguous: true
)




private enum LagunaInjectStore {
    struct Scratch {
        let pool: MLXArray
        let control: [MLXArray]
        let matA: MLXArray
        let matB: MLXArray
    }

    nonisolated(unsafe) static let scratch: Scratch = {
        let pool = MLXArray.zeros([lagunaInjectPoolUInt4 * 4], dtype: .uint32)
        let control = (0..<8).map {
            MLXArray([UInt32($0 + 1), UInt32(lagunaInjectSweepPasses)])
        }
        let matA = MLXArray.zeros(
            [lagunaInjectMatmulM, lagunaInjectMatmulK], dtype: .bfloat16)
        let matB = MLXArray.zeros(
            [lagunaInjectMatmulK, lagunaInjectMatmulN], dtype: .bfloat16)
        eval([pool, matA, matB] + control)
        return Scratch(pool: pool, control: control, matA: matA, matB: matB)
    }()
}








private enum LagunaInjectChain {
    nonisolated(unsafe) static var tail: MLXArray?
}


private func lagunaInjectShare(_ total: Int, layer: Int) -> Int {
    guard total > 0 else { return 0 }
    let layers = LagunaConstants.numHiddenLayers
    return (layer + 1) * total / layers - layer * total / layers
}

private let lagunaInjectActive =
    lagunaInjectDecodeSweeps + lagunaInjectPrefillMatmuls + lagunaInjectDecodeEmpty
    + lagunaInjectPrefillEmpty > 0

func lagunaInjectLayerWork(layer: Int, isSingleTokenDecode: Bool) {
    guard lagunaInjectActive else { return }
    let sweeps = lagunaInjectShare(
        isSingleTokenDecode ? lagunaInjectDecodeSweeps : 0, layer: layer)
    let matmuls = lagunaInjectShare(
        isSingleTokenDecode ? 0 : lagunaInjectPrefillMatmuls, layer: layer)
    let emptyTotal = isSingleTokenDecode ? lagunaInjectDecodeEmpty : lagunaInjectPrefillEmpty
    let empties =
        lagunaInjectEmptySpread
        ? lagunaInjectShare(emptyTotal, layer: layer) : (layer == 0 ? emptyTotal : 0)
    let scratch = LagunaInjectStore.scratch
    var pending: [MLXArray] = []
    pending.reserveCapacity(sweeps + matmuls + empties)
    for k in 0..<sweeps {
        pending.append(
            lagunaInjectSweepKernel(
                [scratch.pool, scratch.control[(layer + k) & 7]],
                grid: (lagunaInjectSweepThreads, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[256]],
                outputDTypes: [.uint32]
            )[0])
    }
    for _ in 0..<matmuls {
        pending.append(matmul(scratch.matA, scratch.matB))
    }
    if empties > 0 {
        var tail = LagunaInjectChain.tail ?? scratch.control[0]
        for k in 0..<empties {
            tail = lagunaInjectEmptyKernel(
                [
                    scratch.control[(layer + k) & 7],
                    lagunaInjectEmptyChain ? tail : scratch.control[7],
                ],
                grid: (lagunaInjectEmptyThreadgroups * 256, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[256]],
                outputDTypes: [.uint32]
            )[0]
            if !lagunaInjectEmptyChain { pending.append(tail) }
        }
        LagunaInjectChain.tail = tail
        if lagunaInjectEmptyChain { pending.append(tail) }
    }
    guard !pending.isEmpty else { return }
    asyncEval(pending)
}



