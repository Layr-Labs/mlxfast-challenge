import Foundation
import MLX
import MLXFastCore
import MLXLMCommon
import MLXNN

/// Tensor name helpers for the Poolside Laguna text tower. The source
/// checkpoint is already text-only and uses the runtime-native `model.*` /
/// `lm_head.*` names, which the transform preserves unchanged.
public enum LagunaWeightNames {
    private static let prefix = "model"

    public static let embedTokens = "\(prefix).embed_tokens.weight"
    public static let finalNorm = "\(prefix).norm.weight"
    public static let lmHead = "lm_head.weight"

    public static func layer(_ layerIndex: Int, _ suffix: String) -> String {
        "\(prefix).layers.\(layerIndex).\(suffix)"
    }

    public static func attention(_ layerIndex: Int, _ suffix: String) -> String {
        layer(layerIndex, "self_attn.\(suffix)")
    }

    public static func mlp(_ layerIndex: Int, _ suffix: String) -> String {
        layer(layerIndex, "mlp.\(suffix)")
    }
}

/// Metadata-level access and validation for the transformed Laguna weights
/// tree. `validateRequiredMetadata` checks that every tensor the runtime model
/// needs is present with the
/// expected dtype/shape/quantization WITHOUT materializing any `MLXArray`s,
/// so a malformed weights directory fails fast before the (expensive) full
/// weight load.
public struct LagunaWeightLoader {
    public let denseStore: DenseTensorStore

    public init(weightsPath: String) throws {
        self.denseStore = try DenseTensorStore(weightsPath: weightsPath)
    }

    public init(denseStore: DenseTensorStore) {
        self.denseStore = denseStore
    }

    public func validateRequiredMetadata(config: LagunaConfig) throws {
        let tensorNames = denseStore.tensorNames
        let forbiddenSuffixes = [
            ".weight_packed",
            ".input_global_scale",
            ".weight_global_scale",
            ".k_scale",
            ".v_scale",
            ".biases",
        ]
        if let forbiddenName = tensorNames.first(where: { name in
            forbiddenSuffixes.contains(where: name.hasSuffix)
        }) {
            throw MLXFastError.invalidInput(
                "Poolside Laguna MLX checkpoint must not contain compressed-tensors/global-scale, "
                    + "FP8 KV-scale, or affine-bias tensor \(forbiddenName)"
            )
        }
        guard tensorNames.count == LagunaConstants.tensorCount else {
            throw MLXFastError.invalidInput(
                "Poolside Laguna tensor inventory contains \(tensorNames.count) tensors; "
                    + "expected exactly \(LagunaConstants.tensorCount)"
            )
        }
        var dtypeCounts: [String: Int] = [:]
        for name in tensorNames {
            guard let record = denseStore.record(named: name) else {
                throw MLXFastError.invalidInput("dense tensor not found: \(name)")
            }
            dtypeCounts[record.dtype, default: 0] += 1
        }
        let expectedDTypeCounts = [
            "BF16": LagunaConstants.bfloat16TensorCount,
            "F32": LagunaConstants.float32TensorCount,
            "U32": LagunaConstants.packedUInt32TensorCount,
            "U8": LagunaConstants.e4m3ScaleUInt8TensorCount,
        ]
        guard dtypeCounts == expectedDTypeCounts else {
            throw MLXFastError.invalidInput(
                "Poolside Laguna tensor dtype inventory \(dtypeCounts) does not match "
                    + "expected \(expectedDTypeCounts)"
            )
        }

        try validateBFloat16ProjectionMetadata(
            named: LagunaWeightNames.embedTokens,
            expectedShape: [config.vocabSize, config.hiddenSize]
        )
        try validateDenseTensorMetadata(
            named: LagunaWeightNames.finalNorm,
            expectedShape: [config.hiddenSize],
            expectedDType: "BF16"
        )
        if !config.tieWordEmbeddings {
            try validateBFloat16ProjectionMetadata(
                named: LagunaWeightNames.lmHead,
                expectedShape: [config.vocabSize, config.hiddenSize]
            )
        }

        for layerIndex in 0..<config.numHiddenLayers {
            let layerHeads = config.heads(forLayer: layerIndex)

            for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                try validateDenseTensorMetadata(
                    named: LagunaWeightNames.layer(layerIndex, suffix),
                    expectedShape: [config.hiddenSize],
                    expectedDType: "BF16"
                )
            }

            try validateBFloat16ProjectionMetadata(
                named: LagunaWeightNames.attention(layerIndex, "q_proj.weight"),
                expectedShape: [layerHeads * config.headDim, config.hiddenSize]
            )
            for suffix in ["k_proj.weight", "v_proj.weight"] {
                try validateBFloat16ProjectionMetadata(
                    named: LagunaWeightNames.attention(layerIndex, suffix),
                    expectedShape: [
                        config.numKeyValueHeads * config.headDim,
                        config.hiddenSize,
                    ]
                )
            }
            try validateBFloat16ProjectionMetadata(
                named: LagunaWeightNames.attention(layerIndex, "o_proj.weight"),
                expectedShape: [config.hiddenSize, layerHeads * config.headDim]
            )
            if let gateDim = config.gateProjectionOutputDim(forLayer: layerIndex) {
                try validateBFloat16ProjectionMetadata(
                    named: LagunaWeightNames.attention(layerIndex, "g_proj.weight"),
                    expectedShape: [gateDim, config.hiddenSize]
                )
            }
            for suffix in ["q_norm.weight", "k_norm.weight"] {
                try validateDenseTensorMetadata(
                    named: LagunaWeightNames.attention(layerIndex, suffix),
                    expectedShape: [config.headDim],
                    expectedDType: "BF16"
                )
            }

            if config.isSparse(layer: layerIndex) {
                // Poolside keeps routing in full precision; only expert
                // projections carry NVFP4 companions.
                try validateBFloat16ProjectionMetadata(
                    named: LagunaWeightNames.mlp(layerIndex, "gate.weight"),
                    expectedShape: [config.numExperts, config.hiddenSize]
                )
                try validateDenseTensorMetadata(
                    named: LagunaWeightNames.mlp(layerIndex, "gate.e_score_correction_bias"),
                    expectedShape: [config.numExperts],
                    expectedDType: "F32"
                )
                // Routed experts: SwitchGLU-stacked tensors with a leading
                // experts axis.
                for suffix in ["switch_mlp.gate_proj.weight", "switch_mlp.up_proj.weight"] {
                    try validateNVFP4TensorMetadata(
                        named: LagunaWeightNames.mlp(layerIndex, suffix),
                        expectedLeadingShape: [config.numExperts, config.moeIntermediateSize],
                        expectedInputFeatures: config.hiddenSize,
                        quantization: config.quantization
                    )
                }
                try validateNVFP4TensorMetadata(
                    named: LagunaWeightNames.mlp(layerIndex, "switch_mlp.down_proj.weight"),
                    expectedLeadingShape: [config.numExperts, config.hiddenSize],
                    expectedInputFeatures: config.moeIntermediateSize,
                    quantization: config.quantization
                )
                for suffix in ["shared_expert.gate_proj.weight", "shared_expert.up_proj.weight"] {
                    try validateNVFP4TensorMetadata(
                        named: LagunaWeightNames.mlp(layerIndex, suffix),
                        expectedLeadingShape: [config.sharedExpertIntermediateSize],
                        expectedInputFeatures: config.hiddenSize,
                        quantization: config.quantization
                    )
                }
                try validateNVFP4TensorMetadata(
                    named: LagunaWeightNames.mlp(layerIndex, "shared_expert.down_proj.weight"),
                    expectedLeadingShape: [config.hiddenSize],
                    expectedInputFeatures: config.sharedExpertIntermediateSize,
                    quantization: config.quantization
                )
            } else {
                for suffix in ["gate_proj.weight", "up_proj.weight"] {
                    try validateBFloat16ProjectionMetadata(
                        named: LagunaWeightNames.mlp(layerIndex, suffix),
                        expectedShape: [config.intermediateSize, config.hiddenSize]
                    )
                }
                try validateBFloat16ProjectionMetadata(
                    named: LagunaWeightNames.mlp(layerIndex, "down_proj.weight"),
                    expectedShape: [config.hiddenSize, config.intermediateSize]
                )
            }
        }

        var expectedTensorCount = config.tieWordEmbeddings ? 2 : 3
        for layerIndex in 0..<config.numHiddenLayers {
            // Layer norms (2), q/k/v/o projections (4), q/k norms (2),
            // and the Poolside per-head gate projection (1).
            expectedTensorCount += 8
            if config.gateProjectionOutputDim(forLayer: layerIndex) != nil {
                expectedTensorCount += 1
            }
            // Sparse layers have two router tensors plus six NVFP4
            // projections, each represented by weight + scales. Layer 0 is
            // the sole dense three-projection MLP.
            expectedTensorCount += config.isSparse(layer: layerIndex) ? 14 : 3
        }
        guard expectedTensorCount == LagunaConstants.tensorCount else {
            throw MLXFastError.invalidInput(
                "internal Poolside Laguna tensor contract computed \(expectedTensorCount) tensors; "
                    + "expected \(LagunaConstants.tensorCount)"
            )
        }
    }

    /// Validates a plain tensor's exact dtype and shape without materializing it.
    private func validateDenseTensorMetadata(
        named name: String,
        expectedShape: [Int],
        expectedDType: String
    ) throws {
        guard let record = denseStore.record(named: name) else {
            throw MLXFastError.invalidInput("dense tensor not found: \(name)")
        }
        guard record.dtype == expectedDType, record.shape == expectedShape else {
            throw MLXFastError.invalidInput(
                "tensor \(name) dtype/shape \(record.dtype) \(record.shape) does not match expected \(expectedDType) \(expectedShape)"
            )
        }
    }

    private func validateBFloat16ProjectionMetadata(
        named name: String,
        expectedShape: [Int]
    ) throws {
        try validateDenseTensorMetadata(
            named: name,
            expectedShape: expectedShape,
            expectedDType: "BF16"
        )
        for suffix in ["scales", "biases"] {
            let companionName = Self.companionName(for: name, suffix: suffix)
            guard denseStore.record(named: companionName) == nil else {
                throw MLXFastError.invalidInput(
                    "BF16 Poolside projection \(name) must not contain \(companionName)"
                )
            }
        }
    }

    /// Validates a Poolside NVFP4 expert tensor: packed U32 codes, one U8
    /// E4M3 scale per 16 inputs, and no affine-bias companion.
    private func validateNVFP4TensorMetadata(
        named name: String,
        expectedLeadingShape: [Int],
        expectedInputFeatures: Int,
        quantization: LagunaQuantizationSpec
    ) throws {
        let (groupSize, bits) = quantization.expected(forTensorStem: Self.tensorStem(name))
        guard expectedInputFeatures > 0,
              quantization.mode == LagunaConstants.quantizationMode,
              groupSize == LagunaConstants.quantizationGroupSize,
              bits == LagunaConstants.quantizationBits,
              (expectedInputFeatures * bits).isMultiple(of: 32),
              expectedInputFeatures.isMultiple(of: groupSize)
        else {
            throw MLXFastError.invalidInput(
                "NVFP4 tensor \(name) logical input \(expectedInputFeatures) is incompatible with 4-bit group size 16"
            )
        }
        guard let record = denseStore.record(named: name) else {
            throw MLXFastError.invalidInput("dense tensor not found: \(name)")
        }
        guard record.dtype == "U32" else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) must use U32 packed codes, found \(record.dtype)"
            )
        }
        let expectedWeightShape = expectedLeadingShape + [expectedInputFeatures * bits / 32]
        guard record.shape == expectedWeightShape else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) shape \(record.shape) does not match expected shape \(expectedWeightShape)"
            )
        }

        let scalesName = Self.companionName(for: name, suffix: "scales")
        guard let scales = denseStore.record(named: scalesName) else {
            throw MLXFastError.invalidInput(
                "NVFP4 tensor \(name) is missing U8 scales \(scalesName)"
            )
        }
        let expectedCompanionShape = expectedLeadingShape + [expectedInputFeatures / groupSize]
        guard scales.dtype == "U8", scales.shape == expectedCompanionShape else {
            throw MLXFastError.invalidInput(
                "NVFP4 tensor \(name) scales dtype/shape \(scales.dtype) \(scales.shape) does not match expected U8 \(expectedCompanionShape)"
            )
        }
        let biasesName = Self.companionName(for: name, suffix: "biases")
        guard denseStore.record(named: biasesName) == nil else {
            throw MLXFastError.invalidInput(
                "NVFP4 tensor \(name) must not contain affine biases \(biasesName)"
            )
        }
    }

    static func tensorStem(_ name: String) -> String {
        guard name.hasSuffix(".weight") else {
            return name
        }
        return String(name.dropLast(".weight".count))
    }

    private static func companionName(for baseName: String, suffix: String) -> String {
        "\(tensorStem(baseName)).\(suffix)"
    }
}

/// Eagerly-prepared, RAM-resident weight cache for the Laguna text tower. The
/// whole Poolside NVFP4 checkpoint (~21.6 GB) is loaded once at construction
/// time (outside every scored window -- the runtime worker builds this before the
/// benchmark protocol handshake), so every scored forward pays no dense
/// loads or quantized-module construction. All expert tensors are
/// RAM-resident SwitchGLU stacks; there is no expert streaming or residency
/// machinery.
public final class LagunaRuntimeWeightCache {
    public let loader: LagunaWeightLoader
    public let config: LagunaConfig

    /// The Laguna runtime model this benchmark's reference runs. Loaded once
    /// here at construction (outside every scored window). nil only if the
    /// load failed, in which case `loadError` carries the reason and
    /// `requireLibraryModel()` rethrows it.
    public let libraryModel: LagunaRuntimeModel?
    public let loadError: Error?

    public init(loader: LagunaWeightLoader, config: LagunaConfig) {
        self.loader = loader
        self.config = config
        // Select the startup memory profile BEFORE the model load. Laguna
        // retains no alternate weight layouts. The full profile installs the
        // qualified BFS scheduler and post-wire command-buffer defaults below;
        // the documented low-memory profile for <64 GiB machines caps the MLX allocator
        // cache at 6 GiB, shortens command buffers, and clears free
        // warmup buffers before the worker protocol hello -- pure memory
        // management: compiled decode and every other ranked code path stay
        // enabled, matching the ranked box. The layer-count
        // guard keeps tiny unit-test configurations on stock behavior.
        let startupMemoryPolicy: RuntimeStartupMemoryPolicy?
        if config.numHiddenLayers >= 16 {
            let policy = RuntimeStartupMemoryPolicy.resolve(
                physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
                requestedProfile: ProcessInfo.processInfo.environment[
                    RuntimeStartupMemoryPolicy.profileOverrideEnvironmentName
                ]
            )
            if policy.isLowMemory {
                policy.apply()
                startupMemoryPolicy = policy
            } else {
                // The full 128 GiB ranked profile wires the complete live
                // model into Metal's residency set after load. Once those
                // allocations are permanently resident, MLX's stock 50 MiB
                // referenced-byte commit threshold splits every sparse layer
                // across several command buffers even though the layer's
                // weights no longer create residency pressure. A 512 MiB
                // budget fits one complete decode layer (attention plus the
                // routed/shared gate-up and down banks are ~507 MiB for the
                // 200 MB / 200-op command buffers: the post-anupsv-loader regime
                // re-test winner (6 Latin pairs: decode 5/6, prefill 4/6). Explicit
                // MLX_ values win; DARKBLOOM kill switch supports same-binary A/B.
                let env = ProcessInfo.processInfo.environment
                // P4: full-profile BFS width default from 776a79e1 / dda29d26.
                // setenv overwrite=0 keeps any explicit MLX_BFS_MAX_WIDTH.
                setenv("MLX_BFS_MAX_WIDTH", "50", 0)
                if env["DARKBLOOM_POST_WIRE_COMMAND_BUFFER"] != "0" {
                    setenv("MLX_MAX_MB_PER_BUFFER", "200", 0)
                    setenv("MLX_MAX_OPS_PER_BUFFER", "200", 0)
                }
                startupMemoryPolicy = nil
            }
        } else {
            startupMemoryPolicy = nil
        }
        do {
            libraryModel = try LagunaRuntimeWeightCache.loadLibraryModel(
                loader: loader,
                config: config
            )
            loadError = nil
        } catch {
            libraryModel = nil
            loadError = error
        }
        // Constructor-time warmup: the runtime worker builds this cache
        // before the benchmark protocol handshake, so the Metal
        // pipeline-state creation and MLX kernel-cache population triggered
        // by the first forward happen HERE, outside every scored window,
        // instead of inside the first scored prefill. The layer-count guard
        // keeps tiny unit-test configurations from paying a full-size
        // warmup.
        if let model = libraryModel, config.numHiddenLayers >= 16 {
            Self.warmLibraryModel(model)
            if startupMemoryPolicy?.clearAllocatorCacheAfterWarmup == true {
                // Pipeline state is process-lifetime state, while free
                // warmup allocations are exactly the pressure a low-memory
                // machine cannot afford to retain before the protocol hello.
                Memory.clearCache()
            }
        }
        // Zero-headroom wired residency (notes/47 §4e follow-up, session
        // H6): the vendored MLX Device attaches a `MTLResidencySet` to every
        // command queue, but `ResidencySet::capacity_` defaults to 0, so the
        // set stays empty and the driver re-establishes residency for the
        // whole ~21.6 GB RAM-resident text tower on every command buffer
        // (notes/47 §4d-4e: driver-busy tracks the prefill span, 9-15 ms
        // kernelStart gaps). notes/47 §6-§7 measured the naive fix -- a
        // 32 GiB wired limit -- removing that driver work (167.1 -> 9.9 ms)
        // but regressing under the scored seatbelt, because ~10 GiB of spare
        // capacity made `ResidencySet::insert`/`erase` issue a Metal
        // `commit()` for every scored-window allocation and eviction
        // (resident.cpp:28-50).
        //
        // This variant closes that hole with capacity discipline instead of
        // new API: wire ONCE at construction time (outside every scored
        // window) with capacity = live-bytes-now + a small page-rounding
        // slack, so `ResidencySet::resize` migrates the weights in a single
        // commit (resident.cpp:52-90) and every later transient allocation
        // FAILS the fit test (`allocatedSize() + buf > capacity_`) and takes
        // the commit-free `unwired_set_` branch. Live weight buffers never
        // enter the buffer cache, so the trusted per-phase
        // `Memory.clearCache()` reset cannot unwire them; cached transients
        // allocated after this point are never wired, so their eviction
        // never commits either. Pure driver-side residency: no kernel, op,
        // dtype, order, or token behavior changes (register- and
        // shape-neutral by construction).
        //
        // Ships DEFAULT-ON at a dosed 42 MiB capacity (the ranked runner
        // sets no environment variables): `DARKBLOOM_WIRED_ZH=0` is the
        // kill switch, `DARKBLOOM_WIRED_ZH_FRACTION` /
        // `DARKBLOOM_WIRED_ZH_SLACK_MB` override the dose for local A/B
        // (fraction of live bytes + flat slack; the slack also covers the
        // MTLBuffer allocatedSize page-rounding inflation over
        // `Memory.activeMemory` at full wire). A >=96 GiB physical-memory
        // guard keeps sub-128GB machines on stock behavior. Target machine
        // (local and ranked) is an M5 Max with 128 GB unified memory:
        // recommendedMaxWorkingSetSize ~= 115 GB, so even a full ~31 GB
        // wire is far from the OS cap that `metal::set_wired_limit`
        // fail-closes on (allocator.cpp:305-312). See the dose table on
        // `wireResidentWeightsIfEnabled`.
        if libraryModel != nil, config.numHiddenLayers >= 16 {
            Self.wireResidentWeightsIfEnabled()
        }
    }

    /// One prefill-shaped forward (512 tokens) and one single-token decode
    /// step against a throwaway cache, evaluated and discarded. Inputs are
    /// constant BOS tokens, so this is prompt-independent and cannot affect
    /// model output; freed warmup buffers remain eligible for allocator
    /// reuse.
    private static func warmLibraryModel(_ model: LagunaRuntimeModel) {
        let bosToken = Int32(LagunaConstants.bosTokenID)
        let warmupCache = model.newCache(parameters: nil)
        let prefillTokens = MLXArray(
            Array(repeating: bosToken, count: 512),
            [1, 512]
        )
        eval(model(prefillTokens, cache: warmupCache))
        let decodeToken = MLXArray([bosToken], [1, 1])
        var warmDecodeLogits = model(decodeToken, cache: warmupCache)
        eval(warmDecodeLogits)
        // The historical full-attention bundle coupled this second whole-model
        // decode to the fusion selector and regressed ranked prefill 11.3%.
        // Reproducing that retired rewarm now requires its own explicit
        // diagnostic selector; the default-on fused kernel must not silently
        // execute all 40 layers again. The first decode still preserves the
        // promoted constructor-warmup contract, and the kernel-only call below
        // creates the full-attention PSO without model/cache state.
        if lagunaFusedFullAttentionEnabled,
            lagunaFusedFullAttentionWholeModelWarmupEnabled
        {
            warmDecodeLogits = model(decodeToken, cache: warmupCache)
            eval(warmDecodeLogits)
        }
        if lagunaFusedFullAttentionEnabled,
            lagunaFusedFullAttentionKernelWarmupEnabled
        {
            lagunaWarmFullFusedAttentionKernel()
        }
        // Warm the greedy-token pipeline too. Every scored worker request ends
        // in `LagunaCorrectness.greedyToken` (reshape -> last row -> argMax),
        // and the forwards above never run an argmax, so its first use
        // otherwise creates the `argmax_bfloat16` compute pipeline state
        // INSIDE the measured window: a timestamped PSO-miss log showed the
        // compile firing ~0.23 s into the scored prefill request and again in
        // the decode seed, matching a recurring ~17 ms MTLCompilerService
        // interval inside both timed phases in Metal System Trace. Replicating
        // the same ops here moves that one-time compile to untimed init.
        // Input-independent kernel-cache warmup only (TASK.md explicitly
        // allows caches for kernels); constant BOS input, output discarded.
        // `DARKBLOOM_WARM_GREEDY_ARGMAX=0` restores the stock warmup.
        if ProcessInfo.processInfo.environment["DARKBLOOM_WARM_GREEDY_ARGMAX"] != "0",
            let vocabSize = warmDecodeLogits.shape.last, vocabSize > 0
        {
            let rows = warmDecodeLogits.reshaped([-1, vocabSize])
            eval(rows[-1].argMax())
        }
    }
    /// See the construction-time comment: one `set_wired_limit` call sized
    /// to the live footprint, applied through the public async ticket path
    /// (`WiredMemoryTicket.start` -> `mlx_set_wired_limit`), bridged
    /// synchronously because this runs before the worker protocol hello.
    /// SHIPPED DOSE (full wire, operator-directed): capacity = 1.0 x live
    /// bytes + 64 MiB ~= 31.4 GiB -- the entire live footprint (weights +
    /// fused banks + lm_head coarse copy) wired in one resize commit, the
    /// full mechanism from notes/47 §4-§7 with the zero-headroom fit-test
    /// discipline. Measured loaded-local (cool-gated Latin squares, all vs
    /// unwired control, correctness green every sample):
    ///   42 MiB -> -4.2% prefill | 350 MiB -> -8.2% | 0.10x -> -11.2%
    ///   0.20x -> -17.2% | 0.35x -> -20.8% | 1.0x -> -28.3% prefill,
    ///   -4.2% decode composite (seed-prefill share; steady step null).
    /// Chunk 1 (42 MiB) ranked +1.24% score (promoted 1.38531), validating
    /// the ~1:1 loaded-local -> ranked transfer. This configuration ships
    /// the WHOLE curve in one submission per operator instruction; the
    /// documented acceptance band (prefill speedup vs calibration in
    /// [0.952, 1.053]) is expected to reject gains this large in one step
    /// -- if the ranked run fails with acceptance_band_failed, revert to
    /// band-sized dose increments (the curve above is the roadmap).
    ///
    /// Engagement guards: `DARKBLOOM_WIRED_ZH=0` kills it; machines under
    /// 96 GiB physical memory keep stock behavior (the ranked box is an
    /// M5 Max 128 GB; smaller local boxes must not wire a ~31 GB live set
    /// against a much smaller working-set cap).
    private static let wiredZHDefaultFraction = 1.0
    private static let wiredZHDefaultSlackMB = 64

    private static func wireResidentWeightsIfEnabled() {
        let env = ProcessInfo.processInfo.environment
        guard env["DARKBLOOM_WIRED_ZH"] != "0" else { return }
        guard ProcessInfo.processInfo.physicalMemory >= (96 << 30) else { return }

        // Evict cached warmup transients FIRST so only live buffers
        // (weights + persistent runtime tensors) are tracked when the
        // resize walk runs, and the computed capacity leaves no headroom
        // for scored-window scratch to fit into.
        Memory.clearCache()

        let active = Memory.activeMemory
        guard active > 0 else { return }

        let fraction =
            env["DARKBLOOM_WIRED_ZH_FRACTION"].flatMap(Double.init)
            ?? wiredZHDefaultFraction
        let slackMB =
            env["DARKBLOOM_WIRED_ZH_SLACK_MB"].flatMap(Int.init)
            ?? wiredZHDefaultSlackMB
        var target = Int(Double(active) * min(max(fraction, 0.0), 1.0))
        target += max(0, slackMB) << 20

        // metal::set_wired_limit THROWS (uncaught in the Swift backend) on
        // limits above recommendedMaxWorkingSetSize; clamp with margin.
        if let maxRec = GPU.maxRecommendedWorkingSetBytes() {
            target = min(target, maxRec - (256 << 20))
        }
        guard target > 0 else { return }

        let ticket = WiredMemoryTicket(
            size: target,
            policy: MLXLMCommon.WiredSumPolicy(cap: target),
            manager: .shared,
            kind: .active
        )
        let appliedBox = LagunaWiredLimitBox()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            let applied = await ticket.start()
            appliedBox.value = applied
            semaphore.signal()
        }
        // Bounded wait: a manager stall must not hang worker construction.
        let outcome = semaphore.wait(timeout: .now() + .seconds(30))
        Self.wiredTicketRetainer = ticket
        let applied = outcome == .success ? appliedBox.value : -1
        let maxRec = GPU.maxRecommendedWorkingSetBytes() ?? -1
        var line = "mlxfast: wired-zh request=\(target) applied=\(applied)"
        line += " active=\(active)"
        line += " slack_mb=\(max(0, slackMB))"
        line += " fraction=\(fraction)"
        line += " maxrec=\(maxRec)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    /// Never ended: ending would restore the 0 baseline and unwire the
    /// weights (resident.cpp resize-shrink evicts every allocation).
    nonisolated(unsafe) private static var wiredTicketRetainer: WiredMemoryTicket?
    /// Crossing the applied wired limit back from the async ticket task to
    /// the synchronous constructor; the semaphore orders the accesses.
    private final class LagunaWiredLimitBox: @unchecked Sendable {
        var value: Int = 0
    }

    /// Construct and weight-load the Laguna runtime model from the
    /// transformed weights tree. Mirrors the library's `loadWeights`
    /// pipeline (sanitize -> NVFP4 module wiring -> update -> eval) while
    /// streaming each tensor from its shard into MLX-owned storage and
    /// preserving its runtime-native `model.*` / `lm_head.*` parameter paths.
    ///
    /// `LagunaRuntimeModel` promotes exactly each sparse layer's routed/shared
    /// expert projections to NVFP4 during construction. All attention,
    /// embedding, dense-MLP, router, and vocabulary-head modules stay BF16.
    private static func loadLibraryModel(
        loader: LagunaWeightLoader,
        config: LagunaConfig
    ) throws -> LagunaRuntimeModel {
        try loader.validateRequiredMetadata(config: config)
        let model = LagunaRuntimeModel(config)

        let loadedWeights = try loadRuntimeWeightArrays(denseStore: loader.denseStore)
        let sanitized = model.sanitize(weights: loadedWeights)
        // Poolside stores dense parameters in BF16 and NVFP4 scales in U8, so
        // the library's fp16->bf16 conversion pass is a no-op and is omitted.
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        eval(model)
        // Build the retained fused weight layouts (fused QKV, fused
        // shared-expert gate/up; see the DARKBLOOM_FUSED_* flags) from the
        // now-materialized checkpoint arrays, before the constructor-time
        // warmup so the fused kernels warm with their production shapes.
        model.prepareFusedRuntimeWeights()
        return model
    }

    public func requireLibraryModel() throws -> LagunaRuntimeModel {
        guard let libraryModel else {
            throw loadError
                ?? MLXFastError.invalidInput("Laguna runtime model was not loaded")
        }
        return libraryModel
    }
}

// ---------------------------------------------------------------------
// Narrow NVFP4 attention scale planes (DARKBLOOM_ATTN_SCALE_NARROW).
// Init-time packing and its byte-exact reconstruction certificate live
// here with the other weight preparation; the QMV kernels that read the
// planes stay in LagunaRuntimeModel.swift.
// ---------------------------------------------------------------------

/// `DARKBLOOM_ATTN_SCALE_NARROW` (default ON; set "0" to read the stock uint8
/// scale plane): 21-byte-per-32-group storage for the decode-only attention
/// NVFP4 scale planes. A measured census (`research/frieren-pr35-scale-census.md`)
/// found `max - min <= 31` for 100.00% of the 2.78M attention 32-group blocks a
/// decode simdgroup covers, so a 5-bit index plus a per-block uint8 base
/// RECONSTRUCTS the original byte rather than re-deriving it:
/// `code = base + nibble + (bit << 4)` feeds the unchanged
/// `laguna_tail_nvfp4_scale`. 21 B vs 32 B is -34.4% of the attention scale
/// traffic with no escape path and no data-dependent branch. Routed/shared
/// planes reach span 39 and are out of this envelope.
let lagunaAttnScaleNarrowEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ATTN_SCALE_NARROW"] != "0"

/// Per-site kill switches so the q/k/v and o_proj rungs are separable.
let lagunaAttnScaleNarrowQKVEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ATTN_SCALE_NARROW_QKV"] != "0"

let lagunaAttnScaleNarrowOProjEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ATTN_SCALE_NARROW_OPROJ"] != "0"

/// `DARKBLOOM_ATTN_SCALE_LANEMAJOR` (default ON; set "0" to fall back to the
/// 32-group-block planes above): one uint8 base per ROW plus a 4-bit index per
/// group, permuted so that the `blocks = groups / 32` nibbles a decode lane
/// consumes are adjacent. Lane `simd_lid` then reads all of them in a single
/// `blocks / 2`-byte load and the 32 lanes of a simdgroup cover one contiguous
/// `groups / 2`-byte run, against three strided byte loads per 32-group block.
/// Storage is `groups / 2 + 1` bytes per row (65 vs 84 vs 128 for fused QKV).
/// The census measured full-row spans <= 15 for 98.1-99.6% of attention rows;
/// the rest carry base `0xFF` (real bases are <= 41) and read the stock plane,
/// which stays resident for prefill, so an escape costs traffic but no memory.
let lagunaAttnScaleLaneMajorEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_ATTN_SCALE_LANEMAJOR"] != "0"

/// `DARKBLOOM_ATTN_SCALE_NARROW_LOG=1` reports which scale plane each attention
/// QMV dispatch reads. Off by default so the scored dispatch pays no lock and
/// no string interpolation, exactly as `lagunaTrace` is gated.
let lagunaNarrowScaleDispatchLog =
    ProcessInfo.processInfo.environment["DARKBLOOM_ATTN_SCALE_NARROW_LOG"] == "1"

final class LagunaNarrowScaleLog: @unchecked Sendable {
    private var seen: Set<String> = []
    private let lock = NSLock()

    /// Init-time notes (bank built or declined); always reported once.
    func note(_ state: String, _ site: String) {
        lock.lock()
        let isNew = seen.insert("\(state)|\(site)").inserted
        lock.unlock()
        if isNew {
            FileHandle.standardError.write(
                Data("mlxfast: narrow-scales \(state): \(site)\n".utf8))
        }
    }

    @inline(__always) func noteDispatch(
        _ state: @autoclosure () -> String, _ site: @autoclosure () -> String
    ) {
        guard lagunaNarrowScaleDispatchLog else { return }
        note(state(), site())
    }
}

let lagunaNarrowScaleLog = LagunaNarrowScaleLog()

/// Three row-contiguous planes holding the same uint8 E4M3 scale codes as a
/// stock NVFP4 scale plane: a 4-bit index nibble per group, its 5th bit, and
/// one uint8 base per 32-group block. Sizes per 32 groups are 16 + 4 + 1 = 21
/// bytes against the 32 the stock plane spends.
struct LagunaNarrowScaleBank {
    let nibbles: MLXArray
    let highBits: MLXArray
    let bases: MLXArray
    let rows: Int
    let groups: Int

    var arrays: [MLXArray] { [nibbles, highBits, bases] }
}

/// Packs a uint8 NVFP4 scale plane into `LagunaNarrowScaleBank`, or declines
/// when any 32-group block spans more than 31 codes or the packing does not
/// reproduce the plane byte for byte. Both checks run here, at init, on the
/// real bank: the returned bank is only ever a lossless re-encoding.
func lagunaNarrowNVFP4ScaleBank(
    _ scales: MLXArray, site: String, layer: Int
) -> LagunaNarrowScaleBank? {
    guard lagunaAttnScaleNarrowEnabled,
        scales.dtype == .uint8, scales.ndim == 2,
        scales.dim(1).isMultiple(of: 32)
    else {
        return nil
    }
    let rows = scales.dim(0)
    let groups = scales.dim(1)
    let blocks = groups / 32
    let wide = contiguous(scales).reshaped([rows, blocks, 32])
    let blockBase = wide.min(axis: 2, keepDims: true)
    let index = contiguous(wide - blockBase)
    let span = index.max().asType(.int32).item(Int32.self)
    guard span <= 31 else {
        lagunaNarrowScaleLog.note("declined L\(layer) (block span \(span) > 31)", site)
        return nil
    }

    // Nibble plane: byte b holds group 2b in bits 0-3 and group 2b+1 in bits
    // 4-7, read through the uint16 view exactly as `buildInt5Planes` packs the
    // pruned lm_head nibble plane.
    let u = index.reshaped([rows, groups])
    let u16 = u.view(dtype: .uint16)
    let nibbles = contiguous(
        ((u16 & MLXArray(UInt16(0x000F)))
            | ((u16 >> 4) & MLXArray(UInt16(0x00F0)))).asType(.uint8))
    // Bit plane: bit j of byte s holds bit 4 of group 8s+j. Step one gathers
    // the four bit-4s of each little-endian uint32 word (word bits 4, 12, 20,
    // 28) into one nibble; step two merges nibble pairs into bytes.
    let u32 = u.view(dtype: .uint32)
    let bitNibble =
        (((u32 >> 4) & MLXArray(UInt32(0x01)))
        | ((u32 >> 11) & MLXArray(UInt32(0x02)))
        | ((u32 >> 18) & MLXArray(UInt32(0x04)))
        | ((u32 >> 25) & MLXArray(UInt32(0x08)))).asType(.uint8)
    let bitNibble16 = contiguous(bitNibble).view(dtype: .uint16)
    let highBits = contiguous(
        ((bitNibble16 & MLXArray(UInt16(0x000F)))
            | ((bitNibble16 >> 4) & MLXArray(UInt16(0x00F0)))).asType(.uint8))
    let bases = contiguous(blockBase.reshaped([rows, blocks]))

    let bank = LagunaNarrowScaleBank(
        nibbles: nibbles, highBits: highBits, bases: bases,
        rows: rows, groups: groups)
    guard lagunaNarrowScaleBankReproducesScales(bank, scales) else {
        lagunaNarrowScaleLog.note("declined L\(layer) (reconstruction mismatch)", site)
        return nil
    }
    lagunaNarrowScaleLog.note("built", site)
    return bank
}

/// Init-time certificate: decode the three planes with MLX and require every
/// byte to equal the plane the kernels read today. A bank that fails this is
/// discarded, so no dispatch can ever consume an approximate scale.
func lagunaNarrowScaleBankReproducesScales(
    _ bank: LagunaNarrowScaleBank, _ scales: MLXArray
) -> Bool {
    let rows = bank.rows
    let groups = bank.groups
    guard bank.nibbles.dtype == .uint8, bank.nibbles.dims(rows, groups / 2),
        bank.highBits.dtype == .uint8, bank.highBits.dims(rows, groups / 8),
        bank.bases.dtype == .uint8, bank.bases.dims(rows, groups / 32)
    else {
        return false
    }
    let nib = bank.nibbles.asType(.int32).reshaped([rows, groups / 2, 1])
    let nibValues = concatenated([nib & 0x0F, (nib >> 4) & 0x0F], axis: 2)
        .reshaped([rows, groups])
    let hb = bank.highBits.asType(.int32).reshaped([rows, groups / 8, 1])
    let bitValues = concatenated((0..<8).map { (hb >> $0) & 0x01 }, axis: 2)
        .reshaped([rows, groups])
    let baseValues = broadcast(
        bank.bases.asType(.int32).reshaped([rows, groups / 32, 1]),
        to: [rows, groups / 32, 32]
    ).reshaped([rows, groups])
    let decoded = (baseValues + nibValues + (bitValues << 4)).asType(.uint8)
    let mismatches = (decoded .!= scales).asType(.int32).sum().item(Int32.self)
    return mismatches == 0
}

/// Two row-contiguous planes holding the same uint8 E4M3 scale codes as a stock
/// NVFP4 scale plane: one uint8 base per ROW, and a 4-bit index per group
/// permuted to lane-major order, so the `groups / 32` nibbles decode lane `l`
/// consumes occupy bytes `l * groups / 64 ..< (l + 1) * groups / 64`. That is
/// `groups / 2 + 1` bytes per row against the block planes'
/// `21 * groups / 32`, and -- the point -- one load per lane per row instead of
/// three per 32-group block. Rows spanning more than 15 codes carry base
/// `0xFF` and are read from the stock plane, which every other reader keeps
/// resident, so an escape costs traffic but no memory.
struct LagunaLaneMajorScaleBank {
    let nibbles: MLXArray
    let bases: MLXArray
    let rows: Int
    let groups: Int
    let escapedRows: Int

    var arrays: [MLXArray] { [nibbles, bases] }
}

/// Packs a uint8 NVFP4 scale plane into `LagunaLaneMajorScaleBank`. `groups`
/// must be a multiple of 64 so a row's per-lane nibble run is a whole number of
/// bytes and no byte straddles two lanes. Out-of-span rows are escaped rather
/// than declining the whole plane; the certificate then requires every
/// non-escaped row to reproduce the plane byte for byte.
func lagunaLaneMajorNVFP4ScaleBank(
    _ scales: MLXArray, site: String, layer: Int
) -> LagunaLaneMajorScaleBank? {
    guard lagunaAttnScaleLaneMajorEnabled,
        scales.dtype == .uint8, scales.ndim == 2,
        scales.dim(1).isMultiple(of: 64)
    else {
        return nil
    }
    let rows = scales.dim(0)
    let groups = scales.dim(1)
    let blocks = groups / 32
    let plane = contiguous(scales)
    let rowMin = plane.min(axis: 1, keepDims: true)
    let span =
        plane.max(axis: 1, keepDims: true).asType(.int32) - rowMin.asType(.int32)
    // A measured attention base never reaches 0xFF (codes top out at 41), and a
    // row whose minimum somehow did would simply read the stock plane, so the
    // sentinel cannot silently lose a byte either way.
    let fits = span .<= 15
    let bases = contiguous(which(fits, rowMin, MLXArray(UInt8(0xFF))).reshaped([rows]))
    let fitting = Int(fits.asType(.int32).sum().item(Int32.self))
    // [rows, block, group-in-block] -> [rows, lane, block]: lane `l` owns group
    // `l` of every block, so the codes it reads become adjacent.
    let lanes = contiguous(plane.reshaped([rows, blocks, 32]).transposed(0, 2, 1))
    let index = which(
        fits.reshaped([rows, 1, 1]),
        lanes.asType(.int32) - rowMin.asType(.int32).reshaped([rows, 1, 1]),
        MLXArray(Int32(0))
    ).asType(.uint8).reshaped([rows, groups])
    let u16 = contiguous(index).view(dtype: .uint16)
    let nibbles = contiguous(
        ((u16 & MLXArray(UInt16(0x000F)))
            | ((u16 >> 4) & MLXArray(UInt16(0x00F0)))).asType(.uint8))

    let bank = LagunaLaneMajorScaleBank(
        nibbles: nibbles, bases: bases, rows: rows, groups: groups,
        escapedRows: rows - fitting)
    guard lagunaLaneMajorScaleBankReproducesScales(bank, scales) else {
        lagunaNarrowScaleLog.note("declined L\(layer) (lane-major mismatch)", site)
        return nil
    }
    lagunaNarrowScaleLog.noteDispatch(
        "lane-major L\(layer) escaped \(bank.escapedRows)/\(rows)", site)
    lagunaNarrowScaleLog.note("built lane-major", site)
    return bank
}

/// Init-time certificate: undo the lane-major permutation with MLX and require
/// every non-escaped row to equal the plane the kernels read today. A bank that
/// fails is discarded, so no dispatch can consume an approximate scale.
func lagunaLaneMajorScaleBankReproducesScales(
    _ bank: LagunaLaneMajorScaleBank, _ scales: MLXArray
) -> Bool {
    let rows = bank.rows
    let groups = bank.groups
    guard bank.nibbles.dtype == .uint8, bank.nibbles.dims(rows, groups / 2),
        bank.bases.dtype == .uint8, bank.bases.dims(rows),
        scales.dims(rows, groups)
    else {
        return false
    }
    let nib = bank.nibbles.asType(.int32).reshaped([rows, groups / 2, 1])
    let nibValues = concatenated([nib & 0x0F, (nib >> 4) & 0x0F], axis: 2)
        .reshaped([rows, 32, groups / 32])
    let decoded = contiguous(
        (bank.bases.asType(.int32).reshaped([rows, 1, 1]) + nibValues)
            .transposed(0, 2, 1)
    ).reshaped([rows, groups]).asType(.uint8)
    let escaped = (bank.bases .== MLXArray(UInt8(0xFF))).reshaped([rows, 1])
    let mismatches = (which(escaped, scales, decoded) .!= scales)
        .asType(.int32).sum().item(Int32.self)
    return mismatches == 0
}

/// Size of the patch header that `lagunaHalvedGroup32ScalePlane` puts in
/// front of a group-32 halved scale plane. One byte per allowed exception
/// pair is used; the rest is padding that keeps the plane itself aligned to a
/// full Apple GPU cache line.
let lagunaScalePatchHeaderBytes = 128

/// Byte length of the halved packed routed gate/up scale bank, header included.
let lagunaPackedRoutedGateUpScaleBytes =
    lagunaScalePatchHeaderBytes
    + LagunaConstants.numExperts * 2 * LagunaConstants.moeIntermediateSize * 4
    * (LagunaConstants.hiddenSize / 128)

/// Byte length of the halved routed down-projection scale plane, header
/// included.
let lagunaRoutedDownScaleBytes =
    lagunaScalePatchHeaderBytes
    + LagunaConstants.numExperts * LagunaConstants.hiddenSize
    * (LagunaConstants.moeIntermediateSize / 32)

/// Halves a group-16 NVFP4 uint8 scale plane along its last (group) axis by
/// keeping only the even-indexed byte of every adjacent group pair, so a
/// decode kernel reads one scale byte per 32 weights instead of two.
///
/// The shipped Laguna checkpoint stores one scale byte per 16 weights, but its
/// expert planes were produced by MLX's Metal `fp_quantize`, whose
/// per-simdgroup absmax makes both halves of every 32-weight span carry the
/// same byte. Census over all 39 sparse layers (234 tensors, 985,300,992
/// pairs): 985,300,824 pairs are byte-identical and all 168 exceptions are the
/// very first pair of a tensor, the one span the quantizer's first simdgroup
/// writes twice.
///
/// `allowedFlatPairs` lists the flat pair indices allowed to break the rule.
/// Their odd byte is copied into a `lagunaScalePatchHeaderBytes` header placed
/// in front of the halved plane, which both keeps the plane cache-line aligned
/// and lets the kernels restore the exact byte without a second buffer
/// binding. Returns nil unless every other discarded odd byte is bitwise equal
/// to its even partner, so the halved plane is installed only when it is
/// provably lossless for the loaded checkpoint; callers keep their
/// full-resolution path for the nil case.
func lagunaHalvedGroup32ScalePlane(
    _ scales: MLXArray, allowedFlatPairs: [Int]
) -> MLXArray? {
    let pairCount = scales.size / 2
    guard scales.dtype == .uint8, scales.ndim >= 1, scales.dim(-1) % 2 == 0,
        !allowedFlatPairs.isEmpty,
        allowedFlatPairs.count <= lagunaScalePatchHeaderBytes,
        allowedFlatPairs.allSatisfy({ $0 >= 0 && $0 < pairCount })
    else {
        return nil
    }
    let pairs = scales.reshaped([pairCount, 2])
    let even = pairs[0..., 0]
    let odd = pairs[0..., 1]
    let mismatch = MLX.notEqual(even, odd).asType(.int32)
    var violations = mismatch.sum()
    for index in allowedFlatPairs {
        violations = violations - mismatch[index]
    }
    guard violations.item(Int32.self) == 0 else { return nil }
    var header = [UInt8](repeating: 0, count: lagunaScalePatchHeaderBytes)
    for (slot, index) in allowedFlatPairs.enumerated() {
        header[slot] = odd[index].item(UInt8.self)
    }
    return contiguous(concatenated([MLXArray(header), even]))
}

extension LagunaRuntimeSparseMoEBlock {
    /// Builds the `DARKBLOOM_PACKED_SCALES` side bank from the (lazy) fused
    /// routed gate/up arrays: bytes are only reordered, never recomputed.
    /// Per expert the packed layout is `[tile 128][k-block 4][sub 8][16 B]`
    /// with `sub = (simd_group*2 + row)*2 + {0 gate, 1 up}`, behind the shared
    /// `lagunaScalePatchHeaderBytes` header. The row remap
    /// below (gateRow = (logical/32)*64 + logical%32, up = +32) is the stock
    /// kernel's mapping over the 32-row gate/up-interleaved fused bank, baked
    /// into scale storage order. The code bytes remain in the resident fused
    /// weight bank, so this side copy is ~32 MB per sparse layer instead of
    /// duplicating the ~256 MB code bank.
    func preparePackedRoutedGateUpBank(
        fusedScales: MLXArray,
        experts: Int,
        split: Int
    ) -> [MLXArray] {
        guard lagunaPackedScalesEnabled else { return [] }
        guard split == LagunaConstants.moeIntermediateSize,
            experts == LagunaConstants.numExperts,
            LagunaConstants.hiddenSize == 2048
        else {
            lagunaPackedScalesLog.note(
                "inactive", "packed routed gate/up bank (geometry guard declined)")
            return []
        }
        let rows = 2 * split  // 1024 fused (gate/up-interleaved) rows
        let rowBlocks = fusedScales.reshaped([experts, rows * 4, 32])
        // Walk-order gather over scale row-blocks: packed position (tile,
        // kblock, sub) reads fused scale row-block (fusedRow, kblock).
        var order = [Int32]()
        order.reserveCapacity(rows * 4)
        for tile in 0..<(rows / 8) {
            for kblock in 0..<4 {
                for sub in 0..<8 {
                    let logicalRow = tile * 4 + sub / 2
                    let gateRow = (logicalRow / 32) * 64 + logicalRow % 32
                    let fusedRow = sub % 2 == 0 ? gateRow : gateRow + 32
                    order.append(Int32(fusedRow * 4 + kblock))
                }
            }
        }
        // `take(axis: 1)` materializes with permuted strides (NOT
        // row-contiguous), and the custom kernel's `ensureRowContiguous`
        // would then re-copy the side bank on EVERY dispatch. Force the
        // one-time row-contiguous materialization here, at init, so dispatches
        // bind the bank buffer directly.
        let packed = contiguous(take(rowBlocks, MLXArray(order), axis: 1))
        // Group-32 halving: inside a packed 32-byte row-block lane `l` reads
        // original group `l`, so the checkpoint's byte-identical (2k, 2k+1)
        // group pairs collapse to `half[l >> 1]`. The walk order puts fused
        // gate row 0 of expert 0 at row-block 0 and fused up row 0 at
        // row-block 1, so their first pairs are flat pair 0 and 16 -- the
        // only two the quantizer can leave unequal in this bank.
        guard let halved = lagunaHalvedGroup32ScalePlane(
            packed, allowedFlatPairs: [0, 16])
        else {
            lagunaPackedScalesLog.note(
                "inactive", "packed routed gate/up bank (scale halving declined)")
            return []
        }
        _packedRoutedGateUpBank = halved
        lagunaPackedScalesLog.note("active", "packed routed gate/up bank prepared")
        return [halved]
    }
}
