import Darwin
import Foundation
import MLX
import MLXFastCore
import MLXFastModel
import MLXLMCommon

// LagunaRuntime is split across LagunaRuntime*.swift for auditability.
// Generated split; behavior identical to the original single file.

extension LagunaRuntime {
    public static func runWorker(weightsPath: String) throws {
        // The worker holds the ~21.6 GB model for its whole lifetime, so it must
        // never outlive the harness parent that spawned it. Reading protocol
        // stdin already ends the worker on parent death (pipe EOF), but only
        // while the worker is blocked reading -- NOT during the minutes-long
        // model load below, or while a forward is in flight. Start the orphan
        // self-reaper first so a parent that dies during those windows cannot
        // leave a resident-model orphan that out-of-memories the next run.
        startRuntimeWorkerOrphanReaper()
        // Move the protocol away from fd 0/1 before any editable model code
        // runs. Startup validation and model construction may log or otherwise
        // use standard I/O; none of that may be confused with protocol traffic.
        let protocolIO = try RuntimeWorkerProtocolIO.isolatingStandardIO()
        try validateRuntimeWorkerPinnedConfiguration(weightsPath: weightsPath)
        let config = try LagunaConfig.load(from: weightsPath)
        let loader = try LagunaWeightLoader(weightsPath: weightsPath)
        // Validate transformed-weight structure HERE, inside the sandboxed worker,
        // rather than in the trusted parent. These checks execute editable
        // MLXFastModel code (DenseTensorStore / LagunaWeightLoader); the parent
        // used to run the equivalent via BenchmarkPreflight.check, which meant
        // submitted code ran in the unsandboxed process that authors score.json.
        // Failing here throws before the protocol hello below, so the parent's
        // worker client sees the worker fail to start and records a failed
        // benchmark -- same coverage, no submitted code in the score-writing
        // parent.
        try loader.denseStore.validateReadableByteRanges()
        try loader.validateRequiredMetadata(config: config)
        // Constructing the weight cache loads the whole 4-bit Laguna text
        // tower and runs its constructor-time kernel warmup, all before the
        // protocol hello -- outside every scored window.
        let weightCache = LagunaRuntimeWeightCache(loader: loader, config: config)
        _ = try weightCache.requireLibraryModel()
        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let sessionNonce = generateRuntimeWorkerNonce()
        // expertStats is always the zero struct for this RAM-resident dense
        // runtime (no expert-streaming machinery); kept in the protocol hello
        // so the schema/field shape stays unchanged from earlier submissions.
        try protocolIO.writeLine(try encoder.encode(RuntimeWorkerResponse(
            id: 0,
            nonce: sessionNonce,
            ok: true,
            expertStats: expertStats(from: weightCache)
        )))
        var state = RuntimeWorkerState()

        while let line = try protocolIO.readLine() {
            guard !line.isEmpty else {
                continue
            }
            let response: RuntimeWorkerResponse
            do {
                let request = try decoder.decode(RuntimeWorkerRequest.self, from: Data(line.utf8))
                do {
                    response = try handleWorkerRequest(
                        request,
                        sessionNonce: sessionNonce,
                        weightCache: weightCache,
                        state: &state
                    )
                } catch {
                    response = RuntimeWorkerResponse(
                        id: request.id,
                        nonce: sessionNonce,
                        ok: false,
                        error: "\(error)"
                    )
                }
            } catch {
                response = RuntimeWorkerResponse(id: -1, nonce: sessionNonce, ok: false, error: "\(error)")
            }
            let data = try encoder.encode(response)
            try protocolIO.writeLine(data)
        }
    }

    /// One-shot structural validation used by the trusted `preflight` command.
    /// Protocol stdout is isolated before any editable model code runs so
    /// participant logging cannot forge the JSON result.
    public static func runPreflightWorker(weightsPath: String) throws {
        startRuntimeWorkerOrphanReaper()
        let protocolIO = try RuntimeWorkerProtocolIO.isolatingStandardIO()
        let response: RuntimeWorkerPreflightResponse
        do {
            try validateRuntimeWorkerPinnedConfiguration(weightsPath: weightsPath)
            let config = try LagunaConfig.load(from: weightsPath)
            let loader = try LagunaWeightLoader(weightsPath: weightsPath)
            try loader.denseStore.validateReadableByteRanges()
            try loader.validateRequiredMetadata(config: config)
            response = RuntimeWorkerPreflightResponse(ok: true)
        } catch {
            response = RuntimeWorkerPreflightResponse(
                ok: false,
                error: "\(error)"
            )
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        try protocolIO.writeLine(try encoder.encode(response))
        if let error = response.error {
            throw MLXFastError.invalidInput(
                "participant worker preflight failed: \(error)"
            )
        }
    }

    /// Poll interval for the worker's orphan self-reaper. Coarse on purpose:
    /// the check is two syscalls, and a couple of seconds of residual
    /// residency after a dead parent is harmless.
    static let runtimeWorkerOrphanPollSeconds = 2.0

    /// Background self-reaper: exit the worker promptly once the spawning
    /// parent is gone, instead of relying solely on protocol-stdin EOF (which
    /// the worker only observes while blocked reading between requests).
    ///
    /// The worker is always spawned by the harness's RuntimeWorkerClient, so
    /// on macOS its parent dying re-parents it to launchd and `getppid()`
    /// becomes 1 -- an unambiguous "the harness that owns me is dead" signal
    /// that cannot fire in any healthy run (local or ranked). Exiting frees
    /// the ~21.6 GB model residency so the next run cannot double-load into an
    /// out-of-memory. The seams exist for tests only; production callers use
    /// the defaults.
    @discardableResult
    static func startRuntimeWorkerOrphanReaper(
        pollIntervalSeconds: Double = LagunaRuntime.runtimeWorkerOrphanPollSeconds,
        isOrphaned: @escaping @Sendable () -> Bool = { getppid() == 1 },
        onOrphaned: @escaping @Sendable () -> Void = {
            fputs(
                "mlxfast-swift: runtime worker parent exited; shutting down to release model memory\n",
                stderr
            )
            exit(1)
        }
    ) -> Thread {
        let thread = Thread {
            while !Thread.current.isCancelled {
                if isOrphaned() {
                    onOrphaned()
                    return
                }
                Thread.sleep(forTimeInterval: pollIntervalSeconds)
            }
        }
        thread.name = "mlxfast.worker-orphan-reaper"
        thread.start()
        return thread
    }

    /// Trusted allocator state applied at the START of every new worker forward
    /// sequence, after the parent has already started the phase timer.
    /// Submitted MLXFastModel code runs during worker initialization and may
    /// change the process-global MLX cache policy, so the trusted request
    /// handler re-normalizes the allocator at the sequence boundary.
    ///
    /// Scope: the boundary only. This is NOT an enforced cap for the rest of
    /// the phase -- editable code may change `Memory.cacheLimit` again inside
    /// the charged window, and any allocation that follows is charged like all
    /// other work. The substantive defense is `Memory.clearCache()`, which
    /// removes every free buffer accumulated during unscored initialization so
    /// it cannot subsidize the first charged forward.
    static let trustedRuntimeWorkerPhaseStartCacheLimitBytes = 6 << 30

    static func resetRuntimeWorkerAllocatorForPhaseStart() throws {
        Memory.cacheLimit = trustedRuntimeWorkerPhaseStartCacheLimitBytes
        Memory.clearCache()
        let remainingCacheBytes = Memory.cacheMemory
        // The pinned MLX clearCache contract synchronously deallocates every
        // cached (free) buffer under its evaluation lock. Live model weights
        // and KV state are active memory, not cacheMemory, so exact zero is the
        // safe fail-closed postcondition rather than a tolerance. This also
        // makes "no MLX allocator activity in flight across a request
        // boundary" part of the submission contract: background work that
        // repopulates the cache here fails the run closed.
        guard remainingCacheBytes == 0 else {
            throw MLXFastError.invalidInput(
                "runtime worker failed to clear the MLX allocator cache at phase start"
            )
        }
    }

    /// One forward through the RAM-resident Laguna runtime model. Laguna is
    /// an INSTANCE model whose per-layer `[KVCache]`
    /// stack both stores K/V and supplies RoPE positions, so the model
    /// takes no explicit offset. `positionOffset` is kept as the caller's
    /// statement of where the sequence should be and is validated against
    /// the cache offsets, preserving the old adapter's fail-loudly contract
    /// for a stale or reused cache. Returns `[1, 1, vocab]` LAST-token
    /// logits (Laguna applies no final softcap and no embedding scaling).
    static func lagunaLogits(
        inputIDs: MLXArray,
        model: LagunaRuntimeModel,
        cache: [KVCache],
        positionOffset: Int
    ) throws -> MLXArray {
        try verifyLagunaCachePosition(positionOffset: positionOffset, cache: cache)
        return model(inputIDs, cache: cache)
    }

    /// Every layer cache must agree on one logical offset
    /// (`StandardKVCache` and `RotatingKVCache` both count total positions
    /// seen), and it must equal the caller's expected position.
    static func verifyLagunaCachePosition(
        positionOffset: Int,
        cache: [KVCache]
    ) throws {
        guard positionOffset >= 0 else {
            throw MLXFastError.invalidInput("Laguna position offset must be non-negative")
        }
        guard let cacheOffset = cache.first?.offset else {
            throw MLXFastError.invalidInput("Laguna model returned no KV caches")
        }
        guard cache.allSatisfy({ $0.offset == cacheOffset }) else {
            throw MLXFastError.invalidInput("Laguna KV cache layer offsets are inconsistent")
        }
        guard positionOffset == cacheOffset else {
            throw MLXFastError.invalidInput(
                "Laguna position offset \(positionOffset) does not match KV cache offset \(cacheOffset)"
            )
        }
    }

    /// Force-evaluate the per-layer KV state so the seed prefill's cache
    /// writes are complete before decode steps are timed against it.
    static func materializeLagunaCacheState(_ cache: [KVCache]) {
        eval(cache)
    }

    static func handleWorkerRequest(
        _ request: RuntimeWorkerRequest,
        sessionNonce: String,
        weightCache: LagunaRuntimeWeightCache,
        state: inout RuntimeWorkerState
    ) throws -> RuntimeWorkerResponse {
        let carriesTraceDiagnostics =
            request.topK != nil || request.expectedToken != nil
        if carriesTraceDiagnostics {
            guard request.kind == "correctness_begin"
                || request.kind == "correctness_step"
            else {
                throw MLXFastError.invalidInput(
                    "runtime worker trace diagnostics are valid only for correctness requests"
                )
            }
            guard let topK = request.topK, topK > 0,
                  let expectedToken = request.expectedToken,
                  expectedToken >= 0,
                  expectedToken < MLXFastConstants.vocabSize
            else {
                throw MLXFastError.invalidInput(
                    "runtime worker trace diagnostics require positive top_k and a valid expected_token"
                )
            }
        }
        switch request.kind {
        case "correctness":
            guard let promptTokens = request.promptTokens, let steps = request.steps else {
                throw MLXFastError.invalidInput("runtime worker correctness request missing prompt_tokens or steps")
            }
            try resetRuntimeWorkerAllocatorForPhaseStart()
            let tokens = try generateGreedyCached(
                promptTokens: promptTokens,
                steps: steps,
                weightCache: weightCache
            )
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                tokens: tokens,
                expertStats: expertStats(from: weightCache),
                peakRamGB: peakResidentMemoryGB()
            )

        case "correctness_begin":
            guard let promptTokens = request.promptTokens else {
                throw MLXFastError.invalidInput("runtime worker teacher-forced correctness request missing prompt_tokens")
            }
            try resetRuntimeWorkerAllocatorForPhaseStart()
            let model = try weightCache.requireLibraryModel()
            let cache = model.newCache(parameters: nil)
            let logits = try lagunaLogits(
                inputIDs: inputIDsArray(promptTokens),
                model: model,
                cache: cache,
                positionOffset: 0
            )
            let token = try LagunaCorrectness.greedyToken(from: logits)
            let diagnostics = try correctnessLogitDiagnostics(
                from: logits,
                topK: request.topK
                    ?? MLXFastConstants.correctnessTopLogits,
                expectedToken: request.expectedToken
            )
            state.correctnessCache = cache
            state.correctnessPromptTokenCount = promptTokens.count
            state.correctnessStep = 0
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                token: token,
                topLogits: diagnostics.topLogits,
                expectedTokenLogit: diagnostics.expectedTokenLogit,
                expectedTokenRank: diagnostics.expectedTokenRank,
                topLogitMargin: diagnostics.topLogitMargin,
                expertStats: expertStats(from: weightCache),
                peakRamGB: peakResidentMemoryGB()
            )

        case "correctness_step":
            guard let previousToken = request.token else {
                throw MLXFastError.invalidInput("runtime worker teacher-forced correctness request missing token")
            }
            guard let cache = state.correctnessCache else {
                throw MLXFastError.invalidInput("runtime worker teacher-forced correctness step before begin")
            }
            let logits = try lagunaLogits(
                inputIDs: inputIDsArray([previousToken]),
                model: try weightCache.requireLibraryModel(),
                cache: cache,
                positionOffset: state.correctnessPromptTokenCount + state.correctnessStep
            )
            let token = try LagunaCorrectness.greedyToken(from: logits)
            let diagnostics = try correctnessLogitDiagnostics(
                from: logits,
                topK: request.topK
                    ?? MLXFastConstants.correctnessTopLogits,
                expectedToken: request.expectedToken
            )
            state.correctnessStep += 1
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                token: token,
                topLogits: diagnostics.topLogits,
                expectedTokenLogit: diagnostics.expectedTokenLogit,
                expectedTokenRank: diagnostics.expectedTokenRank,
                topLogitMargin: diagnostics.topLogitMargin,
                expertStats: expertStats(from: weightCache),
                peakRamGB: peakResidentMemoryGB()
            )

        case "prefill":
            guard let promptTokens = request.promptTokens else {
                throw MLXFastError.invalidInput("runtime worker prefill request missing prompt_tokens")
            }
            try resetRuntimeWorkerAllocatorForPhaseStart()
            let model = try weightCache.requireLibraryModel()
            let cache = model.newCache(parameters: nil)
            let logits = try lagunaLogits(
                inputIDs: inputIDsArray(promptTokens),
                model: model,
                cache: cache,
                positionOffset: 0
            )
            eval(logits)
            let token = try LagunaCorrectness.greedyToken(from: logits)
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                token: token
            )

        case "decode_begin":
            guard let seedTokens = request.seedTokens else {
                throw MLXFastError.invalidInput("runtime worker decode_begin request missing seed_tokens")
            }
            try resetRuntimeWorkerAllocatorForPhaseStart()
            // Exactly one whole-prompt (seed) forward runs here, with NO preceding
            // warmup pass. The decode measurement deliberately charges this seed
            // prefill to the decode phase (see measureWorkerDecode). A second,
            // identical whole-prompt forward -- the warmup this used to run before
            // the seed -- let submitted model code memoize one pass and serve the
            // other from that memo (both had the same tokens at offset 0), so two
            // charged forwards collapsed into one and inflated decode_speedup with
            // no real speedup. The trusted harness cannot force editable code to
            // recompute a forward it issues, so the only robust defense is to never
            // issue two identical forwards in the timed window: with a single seed
            // forward there is no identical predecessor to reuse, and the 128
            // single-token decode steps are input-dependent and cannot be
            // precomputed. Prefill/decode/correctness each run in their own worker
            // process, so no model-owned memo persists across phases; the trusted
            // reset above separately removes allocator free-buffer state.
            let model = try weightCache.requireLibraryModel()
            let cache = model.newCache(parameters: nil)
            let logits = try lagunaLogits(
                inputIDs: inputIDsArray(seedTokens),
                model: model,
                cache: cache,
                positionOffset: 0
            )
            let token = try LagunaCorrectness.greedyToken(from: logits)
            let seedToken = token
            materializeLagunaCacheState(cache)
            state.decodeCache = cache
            state.decodeSeedTokenCount = seedTokens.count
            state.decodeStep = 0
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                seedToken: seedToken
            )

        case "decode_step":
            guard let inputToken = request.token else {
                throw MLXFastError.invalidInput("runtime worker decode_step request missing token")
            }
            guard let cache = state.decodeCache else {
                throw MLXFastError.invalidInput("runtime worker decode_step before decode_begin")
            }
            // decode_step invokes only the same editable entry points the
            // correctness path invokes (the Laguna model forward /
            // greedyToken); it must never call an editable hook that is
            // unique to the scored decode path. The former editable
            // decode-delay knob (removed) was exactly such a phase oracle:
            // because submitted model code is editable, the mere fact that it
            // was invoked ONLY on the timed decode path told the submission
            // "I am being scored now", which lets it serve a slow/correct
            // path while checked and a cheap path while timed. Keep
            // trusted->editable calls phase-agnostic.
            let logits = try lagunaLogits(
                inputIDs: inputIDsArray([inputToken]),
                model: try weightCache.requireLibraryModel(),
                cache: cache,
                positionOffset: state.decodeSeedTokenCount + state.decodeStep
            )
            let token = try LagunaCorrectness.greedyToken(from: logits)
            state.decodeStep += 1
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                token: token
            )

        case "phase_diagnostics":
            let peakRamGB = peakResidentMemoryGB()
            let stats = expertStats(from: weightCache)
            let mlxActiveMemoryBytes = Memory.activeMemory
            let mlxCacheMemoryBytes = Memory.cacheMemory
            let mlxPeakMemoryBytes = Memory.peakMemory
            Memory.clearCache()
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                expertStats: stats,
                peakRamGB: peakRamGB,
                mlxActiveMemoryBytes: mlxActiveMemoryBytes,
                mlxCacheMemoryBytes: mlxCacheMemoryBytes,
                mlxPeakMemoryBytes: mlxPeakMemoryBytes
            )

        default:
            throw MLXFastError.invalidInput("runtime worker received unknown request kind \(request.kind)")
        }
    }

}

private struct RuntimeWorkerPinnedConfiguration: Decodable {
    let modelType: String
    let hiddenSize: Int
    let numHiddenLayers: Int
    let intermediateSize: Int
    let numAttentionHeads: Int
    let numAttentionHeadsPerLayer: [Int]
    let numKeyValueHeads: Int
    let headDim: Int
    let rmsNormEps: Double
    let vocabSize: Int
    let slidingWindow: Int
    let maxPositionEmbeddings: Int
    let attentionBias: Bool?
    let qkvBias: Bool?
    let attentionDropout: Double?
    let gating: String
    let gatingTypes: [String]
    let tieWordEmbeddings: Bool
    let numExperts: Int
    let numExpertsPerTok: Int
    let moeIntermediateSize: Int
    let sharedExpertIntermediateSize: Int
    let moeRoutedScalingFactor: Double
    let normTopkProb: Bool
    let moeApplyRouterWeightOnInput: Bool
    let moeRouterLogitSoftcapping: Double?
    let routerAuxLossCoef: Double
    let useCache: Bool
    let layerTypes: [String]
    let mlpLayerTypes: [String]
    let mlpOnlyLayers: [Int]
    let decoderSparseStep: Int
    let ropeParameters: RuntimeWorkerPinnedRopeParameters
    let quantization: RuntimeWorkerPinnedQuantization?
    let quantizationConfig: RuntimeWorkerPinnedQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numAttentionHeadsPerLayer = "num_attention_heads_per_layer"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case slidingWindow = "sliding_window"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case qkvBias = "qkv_bias"
        case attentionDropout = "attention_dropout"
        case gating
        case gatingTypes = "gating_types"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeRoutedScalingFactor = "moe_routed_scaling_factor"
        case normTopkProb = "norm_topk_prob"
        case moeApplyRouterWeightOnInput = "moe_apply_router_weight_on_input"
        case moeRouterLogitSoftcapping = "moe_router_logit_softcapping"
        case routerAuxLossCoef = "router_aux_loss_coef"
        case useCache = "use_cache"
        case layerTypes = "layer_types"
        case mlpLayerTypes = "mlp_layer_types"
        case mlpOnlyLayers = "mlp_only_layers"
        case decoderSparseStep = "decoder_sparse_step"
        case ropeParameters = "rope_parameters"
        case quantization
        case quantizationConfig = "quantization_config"
    }
}

private struct RuntimeWorkerPinnedRopeParameters: Decodable {
    let slidingAttention: RuntimeWorkerPinnedRopeSpec
    let fullAttention: RuntimeWorkerPinnedRopeSpec

    enum CodingKeys: String, CodingKey {
        case slidingAttention = "sliding_attention"
        case fullAttention = "full_attention"
    }
}

private struct RuntimeWorkerPinnedRopeSpec: Decodable {
    let ropeTheta: Double
    let ropeType: String
    let partialRotaryFactor: Double?
    let factor: Double?
    let originalMaxPositionEmbeddings: Int?
    let betaFast: Double?
    let betaSlow: Double?
    let attentionFactor: Double?

    enum CodingKeys: String, CodingKey, CaseIterable {
        case ropeTheta = "rope_theta"
        case ropeType = "rope_type"
        case partialRotaryFactor = "partial_rotary_factor"
        case factor
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        case betaFast = "beta_fast"
        case betaSlow = "beta_slow"
        case attentionFactor = "attention_factor"
    }

    init(from decoder: Decoder) throws {
        let wire = try decoder.container(keyedBy: RuntimeWorkerQuantizationKey.self)
        let allowed = Set(CodingKeys.allCases.map(\.rawValue))
        guard wire.allKeys.allSatisfy({ allowed.contains($0.stringValue) }) else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription:
                        "Poolside Laguna RoPE contains unsupported fields"
                )
            )
        }
        let container = try decoder.container(keyedBy: CodingKeys.self)
        ropeTheta = try container.decode(Double.self, forKey: .ropeTheta)
        ropeType = try container.decode(String.self, forKey: .ropeType)
        partialRotaryFactor = try container.decodeIfPresent(
            Double.self,
            forKey: .partialRotaryFactor
        )
        factor = try container.decodeIfPresent(Double.self, forKey: .factor)
        originalMaxPositionEmbeddings = try container.decodeIfPresent(
            Int.self,
            forKey: .originalMaxPositionEmbeddings
        )
        betaFast = try container.decodeIfPresent(Double.self, forKey: .betaFast)
        betaSlow = try container.decodeIfPresent(Double.self, forKey: .betaSlow)
        attentionFactor = try container.decodeIfPresent(
            Double.self,
            forKey: .attentionFactor
        )
    }
}

private struct RuntimeWorkerPinnedQuantization: Decodable, Equatable {
    let bits: Int
    let groupSize: Int
    let mode: String

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: RuntimeWorkerQuantizationKey.self)
        let allowed = Set(["bits", "group_size", "mode"])
        guard container.allKeys.allSatisfy({ allowed.contains($0.stringValue) }) else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription:
                        "Poolside Laguna NVFP4 does not permit per-tensor quantization overrides"
                )
            )
        }
        bits = try container.decode(Int.self, forKey: .init("bits"))
        groupSize = try container.decode(Int.self, forKey: .init("group_size"))
        mode = try container.decode(String.self, forKey: .init("mode"))
    }
}

private struct RuntimeWorkerQuantizationKey: CodingKey {
    let stringValue: String
    let intValue: Int? = nil

    init(_ stringValue: String) {
        self.stringValue = stringValue
    }

    init?(stringValue: String) {
        self.init(stringValue)
    }

    init?(intValue: Int) {
        return nil
    }
}

func validateRuntimeWorkerPinnedConfiguration(weightsPath: String) throws {
    let path = URL(fileURLWithPath: weightsPath).appendingPathComponent("config.json")
    let values = try path.resourceValues(
        forKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey]
    )
    let maximumConfigByteCount = 1 * 1024 * 1024
    guard values.isRegularFile == true,
          values.isSymbolicLink != true,
          let byteCount = values.fileSize,
          byteCount > 0,
          byteCount <= maximumConfigByteCount
    else {
        throw MLXFastError.invalidInput(
            "runtime worker config.json must be a non-symlink regular file no larger than \(maximumConfigByteCount) bytes"
        )
    }
    try validateRuntimeWorkerPinnedConfigurationData(Data(contentsOf: path))
}

func validateRuntimeWorkerPinnedConfigurationData(_ data: Data) throws {
    let decoded: RuntimeWorkerPinnedConfiguration
    do {
        decoded = try JSONDecoder().decode(RuntimeWorkerPinnedConfiguration.self, from: data)
    } catch {
        throw MLXFastError.invalidInput(
            "runtime worker config.json is missing or has invalid pinned architecture fields"
        )
    }

    // Laguna XS 2.1 layer schedule: one full-attention layer (48 query
    // heads, YaRN partial RoPE) then three sliding-window layers (64 query
    // heads, plain RoPE), repeating -- full at 0, 4, 8, ..., 36 -- with a
    // dense MLP only at layer 0 and 256-expert top-8 MoE blocks elsewhere.
    let expectedLayerTypes = (0..<MLXFastConstants.numHiddenLayers).map {
        $0 % 4 == 0 ? "full_attention" : "sliding_attention"
    }
    let expectedHeadsPerLayer = (0..<MLXFastConstants.numHiddenLayers).map {
        $0 % 4 == 0 ? 48 : 64
    }
    let expectedMLPLayerTypes = (0..<MLXFastConstants.numHiddenLayers).map {
        $0 == 0 ? "dense" : "sparse"
    }
    let expectedGatingTypes = [String](
        repeating: "per_head",
        count: MLXFastConstants.numHiddenLayers
    )
    guard let quantization = decoded.quantization,
          let quantizationConfig = decoded.quantizationConfig,
          quantization == quantizationConfig
    else {
        throw MLXFastError.invalidInput(
            "runtime worker config.json requires matching quantization and quantization_config blocks"
        )
    }
    // Match the immutable Poolside artifact's behavior-bearing fields exactly.
    // qkv_bias and moe_router_logit_softcapping are absent in the source
    // config (an explicit JSON null decodes equivalently); concrete false/zero
    // substitutions are rejected rather than treated as a synthetic schema.
    guard decoded.modelType == "laguna",
          decoded.hiddenSize == MLXFastConstants.hiddenSize,
          decoded.numHiddenLayers == MLXFastConstants.numHiddenLayers,
          decoded.intermediateSize == MLXFastConstants.intermediateSize,
          decoded.numAttentionHeads == MLXFastConstants.attentionHeads,
          decoded.numAttentionHeadsPerLayer == expectedHeadsPerLayer,
          decoded.numKeyValueHeads == 8,
          decoded.headDim == 128,
          decoded.rmsNormEps == 1e-6,
          decoded.vocabSize == MLXFastConstants.vocabSize,
          decoded.slidingWindow == 512,
          decoded.maxPositionEmbeddings == 262_144,
          decoded.attentionBias == false,
          decoded.qkvBias == nil,
          decoded.attentionDropout == 0,
          decoded.gating == "per-head",
          decoded.gatingTypes == expectedGatingTypes,
          !decoded.tieWordEmbeddings,
          decoded.numExperts == 256,
          decoded.numExpertsPerTok == 8,
          decoded.moeIntermediateSize == 512,
          decoded.sharedExpertIntermediateSize == 512,
          decoded.moeRoutedScalingFactor == 2.5,
          decoded.normTopkProb,
          !decoded.moeApplyRouterWeightOnInput,
          decoded.moeRouterLogitSoftcapping == nil,
          decoded.routerAuxLossCoef == 0,
          decoded.useCache == true,
          decoded.layerTypes == expectedLayerTypes,
          decoded.mlpLayerTypes == expectedMLPLayerTypes,
          decoded.mlpOnlyLayers == [0],
          decoded.decoderSparseStep == 1,
          decoded.ropeParameters.slidingAttention.ropeTheta == 10_000,
          decoded.ropeParameters.slidingAttention.ropeType == "default",
          decoded.ropeParameters.slidingAttention.partialRotaryFactor == 1,
          decoded.ropeParameters.slidingAttention.factor == nil,
          decoded.ropeParameters.slidingAttention.originalMaxPositionEmbeddings == nil,
          decoded.ropeParameters.slidingAttention.betaFast == nil,
          decoded.ropeParameters.slidingAttention.betaSlow == nil,
          decoded.ropeParameters.slidingAttention.attentionFactor == nil,
          decoded.ropeParameters.fullAttention.ropeTheta == 500_000,
          decoded.ropeParameters.fullAttention.ropeType == "yarn",
          decoded.ropeParameters.fullAttention.partialRotaryFactor == 0.5,
          decoded.ropeParameters.fullAttention.factor == 32,
          decoded.ropeParameters.fullAttention.originalMaxPositionEmbeddings
              == 8_192,
          decoded.ropeParameters.fullAttention.betaFast == 64,
          decoded.ropeParameters.fullAttention.betaSlow == 1,
          decoded.ropeParameters.fullAttention.attentionFactor == 1,
          quantization.bits == 4,
          quantization.groupSize == 16,
          quantization.mode == "nvfp4"
    else {
        throw MLXFastError.invalidInput(
            "runtime worker config.json does not match the pinned Laguna XS 2.1 MoE architecture"
        )
    }
}

struct RuntimeWorkerRequest: Codable {
    let id: Int
    let kind: String
    let promptTokens: [Int]?
    let token: Int?
    let seedTokens: [Int]?
    let steps: Int?
    let topK: Int?
    let expectedToken: Int?

    init(
        id: Int,
        kind: String,
        promptTokens: [Int]? = nil,
        token: Int? = nil,
        seedTokens: [Int]? = nil,
        steps: Int? = nil,
        topK: Int? = nil,
        expectedToken: Int? = nil
    ) {
        self.id = id
        self.kind = kind
        self.promptTokens = promptTokens
        self.token = token
        self.seedTokens = seedTokens
        self.steps = steps
        self.topK = topK
        self.expectedToken = expectedToken
    }

    init(from decoder: Swift.Decoder) throws {
        let wireContainer = try decoder.container(
            keyedBy: RuntimeWorkerWireCodingKey.self
        )
        let allowedKeys = Set(CodingKeys.allCases.map(\.rawValue))
        if let unknownKey = wireContainer.allKeys.first(
            where: { !allowedKeys.contains($0.stringValue) }
        ) {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: decoder.codingPath + [unknownKey],
                    debugDescription:
                        "runtime worker request contains unknown field \(unknownKey.stringValue)"
                )
            )
        }

        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(Int.self, forKey: .id)
        kind = try container.decode(String.self, forKey: .kind)
        promptTokens = try container.decodeIfPresent(
            [Int].self,
            forKey: .promptTokens
        )
        token = try container.decodeIfPresent(Int.self, forKey: .token)
        seedTokens = try container.decodeIfPresent(
            [Int].self,
            forKey: .seedTokens
        )
        steps = try container.decodeIfPresent(Int.self, forKey: .steps)
        topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        expectedToken = try container.decodeIfPresent(
            Int.self,
            forKey: .expectedToken
        )
    }

    func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(kind, forKey: .kind)
        try container.encodeIfPresent(promptTokens, forKey: .promptTokens)
        try container.encodeIfPresent(token, forKey: .token)
        try container.encodeIfPresent(seedTokens, forKey: .seedTokens)
        try container.encodeIfPresent(steps, forKey: .steps)
        try container.encodeIfPresent(topK, forKey: .topK)
        try container.encodeIfPresent(expectedToken, forKey: .expectedToken)
    }

    enum CodingKeys: String, CodingKey, CaseIterable {
        case id
        case kind
        case promptTokens = "prompt_tokens"
        case token
        case seedTokens = "seed_tokens"
        case steps
        case topK = "top_k"
        case expectedToken = "expected_token"
    }
}

private struct RuntimeWorkerWireCodingKey: CodingKey {
    let stringValue: String
    let intValue: Int?

    init?(stringValue: String) {
        self.stringValue = stringValue
        self.intValue = nil
    }

    init?(intValue: Int) {
        self.stringValue = String(intValue)
        self.intValue = intValue
    }
}

struct RuntimeWorkerState {
    var correctnessCache: [KVCache]?
    var correctnessPromptTokenCount = 0
    var correctnessStep = 0
    var decodeCache: [KVCache]?
    var decodeSeedTokenCount = 0
    var decodeStep = 0
}

struct RuntimeWorkerPreflightResponse: Codable, Equatable {
    let ok: Bool
    let error: String?

    init(ok: Bool, error: String? = nil) {
        self.ok = ok
        self.error = error
    }
}

struct RuntimeWorkerResponse: Codable {
    let id: Int
    let nonce: String?
    let ok: Bool
    let error: String?
    let token: Int?
    let topLogits: [CorrectnessTraceLogit]?
    let expectedTokenLogit: Double?
    let expectedTokenRank: Int?
    let topLogitMargin: Double?
    let seedToken: Int?
    let tokens: [Int]?
    let expertStats: ExpertStreamingStats?
    let peakRamGB: Double?
    let mlxActiveMemoryBytes: Int?
    let mlxCacheMemoryBytes: Int?
    let mlxPeakMemoryBytes: Int?
    let targetVerificationMode: String?
    let exactPairSegmentCount: Int?
    let exactPairRollbackRowCount: Int?
    let serialVerificationRowCount: Int?

    init(
        id: Int,
        nonce: String? = nil,
        ok: Bool,
        error: String? = nil,
        token: Int? = nil,
        topLogits: [CorrectnessTraceLogit]? = nil,
        expectedTokenLogit: Double? = nil,
        expectedTokenRank: Int? = nil,
        topLogitMargin: Double? = nil,
        seedToken: Int? = nil,
        tokens: [Int]? = nil,
        expertStats: ExpertStreamingStats? = nil,
        peakRamGB: Double? = nil,
        mlxActiveMemoryBytes: Int? = nil,
        mlxCacheMemoryBytes: Int? = nil,
        mlxPeakMemoryBytes: Int? = nil,
        targetVerificationMode: String? = nil,
        exactPairSegmentCount: Int? = nil,
        exactPairRollbackRowCount: Int? = nil,
        serialVerificationRowCount: Int? = nil
    ) {
        self.id = id
        self.nonce = nonce
        self.ok = ok
        self.error = error
        self.token = token
        self.topLogits = topLogits
        self.expectedTokenLogit = expectedTokenLogit
        self.expectedTokenRank = expectedTokenRank
        self.topLogitMargin = topLogitMargin
        self.seedToken = seedToken
        self.tokens = tokens
        self.expertStats = expertStats
        self.peakRamGB = peakRamGB
        self.mlxActiveMemoryBytes = mlxActiveMemoryBytes
        self.mlxCacheMemoryBytes = mlxCacheMemoryBytes
        self.mlxPeakMemoryBytes = mlxPeakMemoryBytes
        self.targetVerificationMode = targetVerificationMode
        self.exactPairSegmentCount = exactPairSegmentCount
        self.exactPairRollbackRowCount = exactPairRollbackRowCount
        self.serialVerificationRowCount = serialVerificationRowCount
    }

    init(from decoder: Swift.Decoder) throws {
        let wireContainer = try decoder.container(
            keyedBy: RuntimeWorkerWireCodingKey.self
        )
        let allowedKeys = Set(CodingKeys.allCases.map(\.rawValue))
        if let unknownKey = wireContainer.allKeys.first(
            where: { !allowedKeys.contains($0.stringValue) }
        ) {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: decoder.codingPath + [unknownKey],
                    debugDescription:
                        "runtime worker response contains unknown field "
                        + unknownKey.stringValue
                )
            )
        }

        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(Int.self, forKey: .id)
        nonce = try container.decodeIfPresent(String.self, forKey: .nonce)
        ok = try container.decode(Bool.self, forKey: .ok)
        error = try container.decodeIfPresent(String.self, forKey: .error)
        token = try container.decodeIfPresent(Int.self, forKey: .token)
        topLogits = try container.decodeIfPresent(
            [CorrectnessTraceLogit].self,
            forKey: .topLogits
        )
        expectedTokenLogit = try container.decodeIfPresent(
            Double.self,
            forKey: .expectedTokenLogit
        )
        expectedTokenRank = try container.decodeIfPresent(
            Int.self,
            forKey: .expectedTokenRank
        )
        topLogitMargin = try container.decodeIfPresent(
            Double.self,
            forKey: .topLogitMargin
        )
        seedToken = try container.decodeIfPresent(Int.self, forKey: .seedToken)
        tokens = try container.decodeIfPresent([Int].self, forKey: .tokens)
        expertStats = try container.decodeIfPresent(
            ExpertStreamingStats.self,
            forKey: .expertStats
        )
        peakRamGB = try container.decodeIfPresent(
            Double.self,
            forKey: .peakRamGB
        )
        mlxActiveMemoryBytes = try container.decodeIfPresent(
            Int.self,
            forKey: .mlxActiveMemoryBytes
        )
        mlxCacheMemoryBytes = try container.decodeIfPresent(
            Int.self,
            forKey: .mlxCacheMemoryBytes
        )
        mlxPeakMemoryBytes = try container.decodeIfPresent(
            Int.self,
            forKey: .mlxPeakMemoryBytes
        )
        targetVerificationMode = try container.decodeIfPresent(
            String.self,
            forKey: .targetVerificationMode
        )
        exactPairSegmentCount = try container.decodeIfPresent(
            Int.self,
            forKey: .exactPairSegmentCount
        )
        exactPairRollbackRowCount = try container.decodeIfPresent(
            Int.self,
            forKey: .exactPairRollbackRowCount
        )
        serialVerificationRowCount = try container.decodeIfPresent(
            Int.self,
            forKey: .serialVerificationRowCount
        )
    }

    func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encodeIfPresent(nonce, forKey: .nonce)
        try container.encode(ok, forKey: .ok)
        try container.encodeIfPresent(error, forKey: .error)
        try container.encodeIfPresent(token, forKey: .token)
        try container.encodeIfPresent(topLogits, forKey: .topLogits)
        try container.encodeIfPresent(
            expectedTokenLogit,
            forKey: .expectedTokenLogit
        )
        try container.encodeIfPresent(
            expectedTokenRank,
            forKey: .expectedTokenRank
        )
        try container.encodeIfPresent(
            topLogitMargin,
            forKey: .topLogitMargin
        )
        try container.encodeIfPresent(seedToken, forKey: .seedToken)
        try container.encodeIfPresent(tokens, forKey: .tokens)
        try container.encodeIfPresent(expertStats, forKey: .expertStats)
        try container.encodeIfPresent(peakRamGB, forKey: .peakRamGB)
        try container.encodeIfPresent(
            mlxActiveMemoryBytes,
            forKey: .mlxActiveMemoryBytes
        )
        try container.encodeIfPresent(
            mlxCacheMemoryBytes,
            forKey: .mlxCacheMemoryBytes
        )
        try container.encodeIfPresent(
            mlxPeakMemoryBytes,
            forKey: .mlxPeakMemoryBytes
        )
        try container.encodeIfPresent(
            targetVerificationMode,
            forKey: .targetVerificationMode
        )
        try container.encodeIfPresent(
            exactPairSegmentCount,
            forKey: .exactPairSegmentCount
        )
        try container.encodeIfPresent(
            exactPairRollbackRowCount,
            forKey: .exactPairRollbackRowCount
        )
        try container.encodeIfPresent(
            serialVerificationRowCount,
            forKey: .serialVerificationRowCount
        )
    }

    enum CodingKeys: String, CodingKey, CaseIterable {
        case id
        case nonce
        case ok
        case error
        case token
        case topLogits = "top_logits"
        case expectedTokenLogit = "expected_token_logit"
        case expectedTokenRank = "expected_token_rank"
        case topLogitMargin = "top_logit_margin"
        case seedToken = "seed_token"
        case tokens
        case expertStats = "expert_stats"
        case peakRamGB = "peak_ram_gb"
        case mlxActiveMemoryBytes = "mlx_active_memory_bytes"
        case mlxCacheMemoryBytes = "mlx_cache_memory_bytes"
        case mlxPeakMemoryBytes = "mlx_peak_memory_bytes"
        case targetVerificationMode = "target_verification_mode"
        case exactPairSegmentCount = "exact_pair_segment_count"
        case exactPairRollbackRowCount = "exact_pair_rollback_row_count"
        case serialVerificationRowCount = "serial_verification_row_count"
    }
}

final class BufferedFileLineReader {
    static let defaultMaximumLineByteCount = 4 * 1024 * 1024

    private let handle: FileHandle
    private let maximumLineByteCount: Int
    private var buffer = Data()

    init(
        handle: FileHandle,
        maximumLineByteCount: Int = BufferedFileLineReader.defaultMaximumLineByteCount
    ) {
        self.handle = handle
        self.maximumLineByteCount = maximumLineByteCount
    }

    func readLine() throws -> Data? {
        guard maximumLineByteCount > 0 else {
            throw MLXFastError.invalidInput("runtime worker protocol line limit must be positive")
        }
        while true {
            if let newlineIndex = buffer.firstIndex(of: 0x0a) {
                let lineByteCount = buffer.distance(from: buffer.startIndex, to: newlineIndex)
                guard lineByteCount <= maximumLineByteCount else {
                    throw MLXFastError.invalidInput(
                        "runtime worker protocol line exceeds \(maximumLineByteCount) bytes"
                    )
                }
                let line = Data(buffer.prefix(lineByteCount))
                buffer.removeSubrange(buffer.startIndex...newlineIndex)
                return line
            }
            guard buffer.count <= maximumLineByteCount else {
                throw MLXFastError.invalidInput(
                    "runtime worker protocol line exceeds \(maximumLineByteCount) bytes"
                )
            }
            let chunk = handle.availableData
            if chunk.isEmpty {
                guard !buffer.isEmpty else {
                    return nil
                }
                guard buffer.count <= maximumLineByteCount else {
                    throw MLXFastError.invalidInput(
                        "runtime worker protocol line exceeds \(maximumLineByteCount) bytes"
                    )
                }
                defer { buffer.removeAll(keepingCapacity: true) }
                return buffer
            }
            buffer.append(chunk)
        }
    }
}

final class RuntimeWorkerProtocolIO {
    private let input: BufferedFileLineReader
    private let output: FileHandle

    private init(inputDescriptor: Int32, outputDescriptor: Int32) {
        self.input = BufferedFileLineReader(
            handle: FileHandle(fileDescriptor: inputDescriptor, closeOnDealloc: true)
        )
        self.output = FileHandle(fileDescriptor: outputDescriptor, closeOnDealloc: true)
    }

    static func isolatingStandardIO() throws -> RuntimeWorkerProtocolIO {
        let descriptors = try duplicateRuntimeWorkerProtocolDescriptors(
            inputDescriptor: STDIN_FILENO,
            outputDescriptor: STDOUT_FILENO
        )
        let inputFD = descriptors.input
        let outputFD = descriptors.output
        do {
            try redirectDescriptorToDevNull(STDIN_FILENO, flags: O_RDONLY, label: "stdin")
            try redirectDescriptorToDevNull(STDOUT_FILENO, flags: O_WRONLY, label: "stdout")
        } catch {
            close(inputFD)
            close(outputFD)
            throw error
        }
        return RuntimeWorkerProtocolIO(inputDescriptor: inputFD, outputDescriptor: outputFD)
    }

    func readLine() throws -> String? {
        guard let data = try input.readLine() else {
            return nil
        }
        guard let line = String(data: data, encoding: .utf8) else {
            throw MLXFastError.invalidInput("runtime worker received non-UTF8 protocol input")
        }
        return line
    }

    func writeLine(_ data: Data) throws {
        guard data.count <= BufferedFileLineReader.defaultMaximumLineByteCount else {
            throw MLXFastError.invalidInput(
                "runtime worker protocol response exceeds "
                    + "\(BufferedFileLineReader.defaultMaximumLineByteCount) bytes"
            )
        }
        try output.write(contentsOf: data)
        try output.write(contentsOf: Data([0x0a]))
    }
}

func duplicateRuntimeWorkerProtocolDescriptors(
    inputDescriptor: Int32,
    outputDescriptor: Int32,
    duplicate: (_ descriptor: Int32, _ label: String) throws -> Int32 = duplicatePrivateDescriptor,
    closeDescriptor: (_ descriptor: Int32) -> Void = { _ = Darwin.close($0) }
) throws -> (input: Int32, output: Int32) {
    let input = try duplicate(inputDescriptor, "stdin")
    do {
        let output = try duplicate(outputDescriptor, "stdout")
        return (input: input, output: output)
    } catch {
        closeDescriptor(input)
        throw error
    }
}

func duplicatePrivateDescriptor(_ descriptor: Int32, label: String) throws -> Int32 {
    // F_DUPFD requires its lower bound to be below RLIMIT_NOFILE. Standard
    // macOS launchd jobs may inherit a soft limit of 256, so a randomized
    // 64...512 bound makes worker startup fail nondeterministically.
    let lowerBound = STDERR_FILENO + 1
    let duplicatedFD = fcntl(descriptor, F_DUPFD_CLOEXEC, lowerBound)
    guard duplicatedFD >= 0 else {
        throw MLXFastError.invalidInput("runtime worker failed to duplicate \(label) for protocol I/O")
    }
    return duplicatedFD
}

func redirectDescriptorToDevNull(_ descriptor: Int32, flags: Int32, label: String) throws {
    let devNullFD = open("/dev/null", flags)
    guard devNullFD >= 0 else {
        throw MLXFastError.invalidInput("runtime worker failed to open /dev/null for \(label) redirection")
    }
    defer {
        close(devNullFD)
    }
    guard dup2(devNullFD, descriptor) >= 0 else {
        throw MLXFastError.invalidInput("runtime worker failed to redirect \(label) away from protocol I/O")
    }
}

/// Continuously drains a runtime worker's stderr pipe on a background thread,
/// forwarding each completed line to `emit` (prefixed and token-redacted) and
/// keeping a capped raw tail for the exit diagnostic. Local modes attach this
/// so participants' debug prints in model code show up live during the edit
/// loop; it also means a chatty worker can no longer fill the undrained pipe
/// buffer and stall the run. Official runs attach the same drain with a no-op
/// emitter, so submitted output is consumed but never forwarded to CI logs.
final class WorkerStderrDrain: @unchecked Sendable {
    private let handle: FileHandle
    private let emit: (String) -> Void
    private let lock = NSLock()
    private var tail = Data()
    private var pendingLine = Data()
    private var pendingLineWasTruncated = false
    private let finished = DispatchSemaphore(value: 0)
    static let tailByteLimit = 64 * 1024
    static let forwardedLinePrefix = "mlxfast-worker: "
    static let truncatedLine = "[worker stderr line exceeded 65536 bytes]"

    init(
        handle: FileHandle,
        emit: ((String) -> Void)? = nil
    ) {
        self.handle = handle
        self.emit = emit ?? { line in
            fputs(line, stderr)
            fflush(stderr)
        }
        let thread = Thread { [self] in
            drainToEOF()
            finished.signal()
        }
        thread.name = "mlxfast.worker-stderr-drain"
        thread.start()
    }

    /// Blocks until the reader thread hits EOF (the worker exited and all
    /// output was ingested) or the timeout passes, then returns the raw tail
    /// for the exit diagnostic (which applies its own sanitization).
    func drainedOutput(timeoutSeconds: Double) -> String {
        if finished.wait(timeout: .now() + timeoutSeconds) == .success {
            // Re-signal so later calls (or repeated diagnostics) do not block.
            finished.signal()
        }
        lock.lock()
        defer {
            lock.unlock()
        }
        var data = tail
        if pendingLineWasTruncated {
            data.append(Data(Self.truncatedLine.utf8))
        } else {
            data.append(pendingLine)
        }
        if data.count > Self.tailByteLimit {
            data = Data(data.suffix(Self.tailByteLimit))
        }
        return String(decoding: data, as: UTF8.self)
    }

    private func drainToEOF() {
        while true {
            let chunk = handle.readData(ofLength: 8192)
            if chunk.isEmpty {
                break
            }
            ingest(chunk)
        }
        flushPendingLine()
    }

    private func ingest(_ chunk: Data) {
        var completedLines: [String] = []
        lock.lock()
        pendingLine.append(chunk)
        while let newlineIndex = pendingLine.firstIndex(of: 0x0a) {
            let lineLength = pendingLine.distance(from: pendingLine.startIndex, to: newlineIndex)
            let lineWasTruncated = pendingLineWasTruncated || lineLength > Self.tailByteLimit
            let lineData = lineWasTruncated
                ? Data(Self.truncatedLine.utf8)
                : Data(pendingLine.prefix(lineLength))
            pendingLine = Data(pendingLine.dropFirst(lineLength + 1))
            pendingLineWasTruncated = false
            appendToTailLocked(lineData + Data([0x0a]))
            completedLines.append(String(decoding: lineData, as: UTF8.self))
        }
        if pendingLine.count > Self.tailByteLimit {
            pendingLine.removeAll(keepingCapacity: true)
            pendingLineWasTruncated = true
        }
        lock.unlock()
        for line in completedLines {
            emitLine(line)
        }
    }

    private func flushPendingLine() {
        lock.lock()
        let remainder = pendingLineWasTruncated
            ? Data(Self.truncatedLine.utf8)
            : pendingLine
        pendingLine = Data()
        pendingLineWasTruncated = false
        if !remainder.isEmpty {
            appendToTailLocked(remainder + Data([0x0a]))
        }
        lock.unlock()
        if !remainder.isEmpty {
            emitLine(String(decoding: remainder, as: UTF8.self))
        }
    }

    private func appendToTailLocked(_ data: Data) {
        tail.append(data)
        if tail.count > Self.tailByteLimit {
            tail = Data(tail.suffix(Self.tailByteLimit))
        }
    }

    private func emitLine(_ line: String) {
        emit("\(Self.forwardedLinePrefix)\(redactedWorkerStderrLine(line))\n")
    }
}

/// Per-line redaction for forwarded worker stderr: worker output comes from
/// submitted model code that has seen the (possibly private) golden, so lines
/// that look like token comparisons are collapsed exactly like the shared
/// error-path redaction.
func redactedWorkerStderrLine(_ line: String) -> String {
    if line.range(of: "expected", options: .caseInsensitive) != nil
        || line.range(of: "actual", options: .caseInsensitive) != nil
    {
        return "token-validation-failed"
    }
    return line
}

final class RuntimeWorkerWatchdog: @unchecked Sendable {
    private let process: Process
    private let timer: DispatchSourceTimer
    private let lock = NSLock()
    private var active = true
    private var fired = false

    init(process: Process, timeoutSeconds: Double, terminationGraceSeconds: Double) {
        self.process = process
        self.timer = DispatchSource.makeTimerSource(queue: .global(qos: .userInitiated))
        timer.schedule(deadline: .now() + timeoutSeconds)
        timer.setEventHandler { [weak self] in
            self?.fire(terminationGraceSeconds: terminationGraceSeconds)
        }
        timer.resume()
    }

    @discardableResult
    func cancelAndReturnDidFire() -> Bool {
        lock.lock()
        active = false
        let result = fired
        lock.unlock()
        timer.cancel()
        return result
    }

    private func fire(terminationGraceSeconds: Double) {
        lock.lock()
        guard active else {
            lock.unlock()
            return
        }
        active = false
        fired = true
        lock.unlock()

        if process.isRunning {
            process.terminate()
        }
        DispatchQueue.global(qos: .userInitiated).asyncAfter(
            deadline: .now() + max(terminationGraceSeconds, 0)
        ) { [process] in
            if process.isRunning {
                kill(process.processIdentifier, SIGKILL)
            }
        }
    }
}

@discardableResult
func stopRuntimeWorkerProcess(
    _ process: Process,
    timeoutSeconds: Double,
    pollIntervalMicroseconds: useconds_t = 10_000
) -> Bool {
    guard process.isRunning else {
        return true
    }
    process.terminate()
    let boundedTimeout = timeoutSeconds.isFinite
        ? min(max(timeoutSeconds, 0), 24 * 60 * 60)
        : 0
    let now = DispatchTime.now().uptimeNanoseconds
    let timeoutNanoseconds = UInt64(boundedTimeout * 1_000_000_000)
    let (deadline, overflow) = now.addingReportingOverflow(timeoutNanoseconds)
    let resolvedDeadline = overflow ? UInt64.max : deadline
    while process.isRunning, DispatchTime.now().uptimeNanoseconds < resolvedDeadline {
        usleep(pollIntervalMicroseconds)
    }
    if process.isRunning {
        kill(process.processIdentifier, SIGKILL)
    }
    let killDeadline = DispatchTime.now().uptimeNanoseconds + 2_000_000_000
    while process.isRunning, DispatchTime.now().uptimeNanoseconds < killDeadline {
        usleep(pollIntervalMicroseconds)
    }
    return !process.isRunning
}

extension LagunaRuntime {
    public static func runPreflightWithWorker(
        weightsPath: String,
        worker options: RuntimeWorkerOptions
    ) throws {
        guard options.requestTimeoutSeconds.isFinite,
              options.requestTimeoutSeconds > 0,
              options.shutdownTimeoutSeconds.isFinite,
              options.shutdownTimeoutSeconds >= 0,
              options.terminationGraceSeconds.isFinite,
              options.terminationGraceSeconds >= 0
        else {
            throw MLXFastError.invalidInput(
                "runtime worker preflight timeouts must be valid"
            )
        }

        let process = Process()
        let stdout = Pipe()
        let stderr = Pipe()
        let workerArguments = [
            "preflight",
            "--weights",
            weightsPath,
        ]
        if let sandboxProfilePath = options.sandboxProfilePath {
            process.executableURL = URL(fileURLWithPath: "/usr/bin/sandbox-exec")
            process.arguments = [
                "-f",
                sandboxProfilePath,
                options.executablePath,
            ] + workerArguments
        } else {
            process.executableURL = URL(fileURLWithPath: options.executablePath)
            process.arguments = workerArguments
        }
        process.environment = sanitizedRuntimeWorkerEnvironment(
            ProcessInfo.processInfo.environment
        )
        process.standardInput = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()

        let stderrDrain = WorkerStderrDrain(
            handle: stderr.fileHandleForReading,
            emit: options.forwardsWorkerStderr ? nil : { _ in }
        )
        let watchdog = RuntimeWorkerWatchdog(
            process: process,
            timeoutSeconds: options.requestTimeoutSeconds,
            terminationGraceSeconds: options.terminationGraceSeconds
        )
        let output = BufferedFileLineReader(
            handle: stdout.fileHandleForReading
        )
        do {
            guard let data = try output.readLine() else {
                throw MLXFastError.invalidInput(
                    "runtime worker preflight closed stdout without a response"
                )
            }
            let response = try JSONDecoder().decode(
                RuntimeWorkerPreflightResponse.self,
                from: data
            )
            process.waitUntilExit()
            if watchdog.cancelAndReturnDidFire() {
                throw MLXFastError.invalidInput(
                    "runtime worker preflight timed out"
                )
            }
            _ = stderrDrain.drainedOutput(
                timeoutSeconds: options.shutdownTimeoutSeconds
                    + options.terminationGraceSeconds + 1
            )
            guard response.ok, process.terminationStatus == 0 else {
                throw MLXFastError.invalidInput(
                    response.error
                        ?? "runtime worker preflight exited with status "
                        + "\(process.terminationStatus)"
                )
            }
        } catch {
            let timedOut = watchdog.cancelAndReturnDidFire()
            if process.isRunning {
                _ = stopRuntimeWorkerProcess(
                    process,
                    timeoutSeconds: options.shutdownTimeoutSeconds
                )
            }
            _ = stderrDrain.drainedOutput(
                timeoutSeconds: options.shutdownTimeoutSeconds
                    + options.terminationGraceSeconds + 1
            )
            if timedOut {
                throw MLXFastError.invalidInput(
                    "runtime worker preflight timed out"
                )
            }
            throw error
        }
    }
}

final class RuntimeWorkerClient {
    private let process: Process
    private let input: FileHandle
    private let output: BufferedFileLineReader
    private let stderrDrain: WorkerStderrDrain
    private let requestTimeoutSeconds: Double
    private let shutdownTimeoutSeconds: Double
    private let terminationGraceSeconds: Double
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private var sessionNonce = ""
    private var nextID = 1
    private var closed = false

    init(
        options: RuntimeWorkerOptions,
        weightsPath: String
    ) throws {
        guard options.helloTimeoutSeconds.isFinite,
              options.helloTimeoutSeconds > 0,
              options.requestTimeoutSeconds.isFinite,
              options.requestTimeoutSeconds > 0,
              options.shutdownTimeoutSeconds.isFinite,
              options.shutdownTimeoutSeconds >= 0,
              options.terminationGraceSeconds.isFinite,
              options.terminationGraceSeconds >= 0
        else {
            throw MLXFastError.invalidInput("runtime worker timeouts must be positive")
        }
        let process = Process()
        let stdin = Pipe()
        let stdout = Pipe()
        let stderr = Pipe()
        let workerArguments = [
            "runtime-worker",
            "--weights",
            weightsPath,
        ]
        if let sandboxProfilePath = options.sandboxProfilePath {
            process.executableURL = URL(fileURLWithPath: "/usr/bin/sandbox-exec")
            process.arguments = [
                "-f",
                sandboxProfilePath,
                options.executablePath,
            ] + workerArguments
        } else {
            process.executableURL = URL(fileURLWithPath: options.executablePath)
            process.arguments = workerArguments
        }
        process.environment = sanitizedRuntimeWorkerEnvironment(ProcessInfo.processInfo.environment)
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()

        self.process = process
        self.input = stdin.fileHandleForWriting
        self.output = BufferedFileLineReader(handle: stdout.fileHandleForReading)
        self.requestTimeoutSeconds = options.requestTimeoutSeconds
        self.shutdownTimeoutSeconds = options.shutdownTimeoutSeconds
        self.terminationGraceSeconds = options.terminationGraceSeconds
        // Always consume the pipe. Official runs use a no-op emitter so worker
        // output cannot reach logs, while local modes retain live forwarding.
        self.stderrDrain = WorkerStderrDrain(
            handle: stderr.fileHandleForReading,
            emit: options.forwardsWorkerStderr ? nil : { _ in }
        )
        let helloWatchdog = RuntimeWorkerWatchdog(
            process: process,
            timeoutSeconds: options.helloTimeoutSeconds,
            terminationGraceSeconds: options.terminationGraceSeconds
        )
        let hello: RuntimeWorkerResponse
        do {
            hello = try readResponseLine(validateNonce: false)
            if helloWatchdog.cancelAndReturnDidFire() {
                throw MLXFastError.invalidInput("runtime worker timed out waiting for protocol hello")
            }
            guard hello.id == 0, hello.ok, let nonce = hello.nonce, !nonce.isEmpty else {
                throw MLXFastError.invalidInput("runtime worker did not return a valid protocol hello")
            }
            self.sessionNonce = nonce
        } catch {
            let helloTimedOut = helloWatchdog.cancelAndReturnDidFire()
            _ = stopRuntimeWorkerProcess(process, timeoutSeconds: options.shutdownTimeoutSeconds)
            _ = stderrDrain.drainedOutput(
                timeoutSeconds: options.shutdownTimeoutSeconds + options.terminationGraceSeconds + 1
            )
            if helloTimedOut {
                throw MLXFastError.invalidInput("runtime worker timed out waiting for protocol hello")
            }
            throw error
        }
    }

    deinit {
        close()
    }

    func close() {
        guard !closed else {
            return
        }
        closed = true
        try? input.close()
        if process.isRunning {
            _ = stopRuntimeWorkerProcess(process, timeoutSeconds: shutdownTimeoutSeconds)
        }
        _ = stderrDrain.drainedOutput(timeoutSeconds: shutdownTimeoutSeconds + terminationGraceSeconds + 1)
    }

    func generateCorrectness(promptTokens: [Int], steps: Int) throws -> RuntimeWorkerResponse {
        try send(
            kind: "correctness",
            promptTokens: promptTokens,
            steps: steps
        )
    }

    func beginTeacherForcedCorrectness(
        promptTokens: [Int],
        topK: Int? = nil,
        expectedToken: Int? = nil
    ) throws -> RuntimeWorkerResponse {
        try send(
            kind: "correctness_begin",
            promptTokens: promptTokens,
            topK: topK,
            expectedToken: expectedToken
        )
    }

    func teacherForcedCorrectnessStep(
        previousToken: Int,
        topK: Int? = nil,
        expectedToken: Int? = nil
    ) throws -> RuntimeWorkerResponse {
        try send(
            kind: "correctness_step",
            token: previousToken,
            topK: topK,
            expectedToken: expectedToken
        )
    }

    func prefill(promptTokens: [Int]) throws -> RuntimeWorkerResponse {
        try send(
            kind: "prefill",
            promptTokens: promptTokens
        )
    }

    func beginDecode(seedTokens: [Int]) throws -> RuntimeWorkerResponse {
        try send(
            kind: "decode_begin",
            seedTokens: seedTokens
        )
    }

    func decodeStep(inputToken: Int) throws -> RuntimeWorkerResponse {
        try send(
            kind: "decode_step",
            token: inputToken
        )
    }

    func phaseDiagnostics() throws -> RuntimeWorkerResponse {
        try send(kind: "phase_diagnostics")
    }

    private func send(
        kind: String,
        promptTokens: [Int]? = nil,
        token: Int? = nil,
        seedTokens: [Int]? = nil,
        steps: Int? = nil,
        topK: Int? = nil,
        expectedToken: Int? = nil
    ) throws -> RuntimeWorkerResponse {
        guard process.isRunning else {
            throw MLXFastError.invalidInput("runtime worker exited before request \(kind): \(workerExitDiagnostic())")
        }
        let id = nextID
        nextID += 1
        let request = RuntimeWorkerRequest(
            id: id,
            kind: kind,
            promptTokens: promptTokens,
            token: token,
            seedTokens: seedTokens,
            steps: steps,
            topK: topK,
            expectedToken: expectedToken
        )
        var data = try encoder.encode(request)
        guard data.count <= BufferedFileLineReader.defaultMaximumLineByteCount else {
            throw MLXFastError.invalidInput(
                "runtime worker protocol request exceeds "
                    + "\(BufferedFileLineReader.defaultMaximumLineByteCount) bytes"
            )
        }
        data.append(0x0a)
        let watchdog = RuntimeWorkerWatchdog(
            process: process,
            timeoutSeconds: requestTimeoutSeconds,
            terminationGraceSeconds: terminationGraceSeconds
        )
        let response: RuntimeWorkerResponse
        do {
            try input.write(contentsOf: data)
            response = try readResponseLine(validateNonce: true)
        } catch {
            if watchdog.cancelAndReturnDidFire() {
                throw MLXFastError.invalidInput("runtime worker timed out handling request \(kind)")
            }
            throw error
        }
        guard !watchdog.cancelAndReturnDidFire() else {
            throw MLXFastError.invalidInput("runtime worker timed out handling request \(kind)")
        }
        guard response.id == id else {
            throw MLXFastError.invalidInput("runtime worker returned response id \(response.id), expected \(id)")
        }
        guard response.ok else {
            throw MLXFastError.invalidInput("runtime worker \(kind) failed: \(response.error ?? "unknown error")")
        }
        return response
    }

    private func readResponseLine(validateNonce: Bool) throws -> RuntimeWorkerResponse {
        while true {
            let data = try readWorkerOutputLine()
            guard runtimeWorkerLineLooksLikeJSONResponse(data) else {
                continue
            }
            let response = try decoder.decode(RuntimeWorkerResponse.self, from: data)
            if validateNonce, response.nonce != sessionNonce {
                throw MLXFastError.invalidInput("runtime worker returned a response with an invalid nonce")
            }
            return response
        }
    }

    private func readWorkerOutputLine() throws -> Data {
        guard let data = try output.readLine() else {
            throw MLXFastError.invalidInput(
                "runtime worker closed stdout before returning a response: \(workerExitDiagnostic())"
            )
        }
        return data
    }

    private func workerExitDiagnostic() -> String {
        if process.isRunning {
            _ = stopRuntimeWorkerProcess(process, timeoutSeconds: shutdownTimeoutSeconds)
        }
        let stderr = stderrDrain.drainedOutput(
            timeoutSeconds: shutdownTimeoutSeconds + terminationGraceSeconds + 1
        )
        let trimmed = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
        let redacted = sanitizeWorkerDiagnostic(trimmed)
        let status = process.isRunning ? "timeout" : String(process.terminationStatus)
        if redacted.isEmpty {
            return "exit_status=\(status)"
        }
        return "exit_status=\(status) stderr=\(redacted)"
    }

    private func sanitizeWorkerDiagnostic(_ value: String) -> String {
        let singleLine = value
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
        if singleLine.range(of: "expected", options: .caseInsensitive) != nil
            || singleLine.range(of: "actual", options: .caseInsensitive) != nil
        {
            return "token-validation-failed"
        }
        return singleLine
    }
}

/// Runtime-worker child environment policy: STRICT ALLOWLIST, not a denylist.
///
/// The runtime worker is the only process that executes submitted model code,
/// and submitted code can read its whole environment via
/// `ProcessInfo.processInfo.environment`. The ranked pipeline runs that same
/// code in two separate passes with different harness environments -- the
/// unscored correctness/gates pass and the scored timed pass -- so ANY
/// inherited variable whose value differs between the passes is a phase
/// oracle: a submission could serve correct-but-slow behavior while its
/// tokens are checked and a cheaper path while its speed is measured,
/// inflating the paired score without a real optimization.
///
/// A remove-by-default denylist structurally cannot close that class: every
/// new harness/CI/workflow variable reopens it by default (MLXFAST_NOTE,
/// MLXFAST_SCORE_PATH, MLXFAST_INTEGRITY_PATH, the semantic-GPQA knobs,
/// BENCH_GOLDEN_PATH, and GIT_CONFIG_* all leaked through the previous
/// denylist, and the first three differ between the gates and timed passes).
/// So this filter starts from an EMPTY environment and copies in only the
/// names below, which makes the child environment byte-identical across
/// phases by construction -- the phase-isolation property is tested directly
/// by `runtimeWorkerEnvironmentIsIdenticalAcrossPipelinePhases`.
///
/// Keep-set rationale (everything else is dropped):
/// - Exact POSIX/login/session basics (`PATH`, `HOME`, `TMPDIR`, ...): the
///   dynamic loader, Foundation, and Metal's shader-cache paths rely on
///   them; their values are fixed per host/user, never per phase.
/// - `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`: constant "1" wherever trusted
///   scripts set them; they only ever remove (network) work.
/// - `LC_`/`DYLD_`/`MTL_`/`METAL_` prefixes: locale, dynamic-loader, and
///   Metal-framework configuration families. System-level, operator-owned,
///   phase-independent; dropping loader/Metal config could break how the
///   worker loads MLX and its metallib.
/// - `MLX_` prefix: MLX core tuning knobs read by mlx::core (e.g.
///   MLX_DISABLE_COMPILE, MLX_MAX_OPS_PER_BUFFER, MLX_RESOURCE_LIMIT) and by
///   the mlx-swift-lm fork (MLX_COMPILED_DECODE). Note "MLX_" does NOT match
///   harness "MLXFAST_*" names -- harness variables stay excluded.
/// - `DARKBLOOM_` prefix: model-runtime opt-ins read only by model-side code
///   and the mlx-swift-lm fork. The ranked workflow never sets them (absent
///   in BOTH ranked phases); they exist for operator/participant tuning on
///   local machines and must keep reaching the worker there.
/// - `MLXFAST_USE_RUNTIME_WORKER` is force-set to "0" so the child can never
///   recursively spawn another worker.
///
/// `SSH_AUTH_SOCK` is deliberately NOT allowlisted. The worker needs no SSH
/// agent (weights arrive via argv, dependencies are pre-resolved, network is
/// denied by its Seatbelt profile), and forwarding a live agent socket hands
/// submitted model code a usable authentication channel in any context where
/// the parent process happens to hold one. It is harmless on the ranked box
/// (no agent is present and PF blocks egress) but has no legitimate use here,
/// so it is dropped like every other non-essential name.
///
/// Maintainer contract: do NOT regress this to keep-by-default, do NOT add a
/// broad `MLXFAST_` (or `BENCH_`) allowance, do NOT re-add `SSH_AUTH_SOCK` (or
/// any other credential/agent socket), and never allowlist a name whose value
/// trusted code could set differently between the gates and timed passes. The
/// worker itself needs no MLXFAST_* configuration: its weights path arrives
/// via argv (`runtime-worker --weights ...`).
func sanitizedRuntimeWorkerEnvironment(_ environment: [String: String]) -> [String: String] {
    let allowedExactKeys: Set<String> = [
        "HF_HUB_OFFLINE",
        "HOME",
        "LANG",
        "LOGNAME",
        "PATH",
        "SHELL",
        "TERM",
        "TMPDIR",
        "TRANSFORMERS_OFFLINE",
        "USER",
        // macOS per-user default text encoding consulted by CoreFoundation.
        "__CF_USER_TEXT_ENCODING",
    ]
    let allowedPrefixes = [
        "DARKBLOOM_",
        "DYLD_",
        "LC_",
        "METAL_",
        "MLX_",
        "MTL_",
    ]
    var sanitized: [String: String] = [:]
    for (key, value) in environment
        where allowedExactKeys.contains(key)
        || allowedPrefixes.contains(where: { key.hasPrefix($0) })
    {
        sanitized[key] = value
    }
    sanitized["MLXFAST_USE_RUNTIME_WORKER"] = "0"
    return sanitized
}

func generateRuntimeWorkerNonce() -> String {
    var bytes = [UInt8](repeating: 0, count: 16)
    bytes.withUnsafeMutableBytes { buffer in
        if let baseAddress = buffer.baseAddress {
            arc4random_buf(baseAddress, buffer.count)
        }
    }
    return bytes
        .map { String(format: "%02x", $0) }
        .joined()
}

func runtimeWorkerLineLooksLikeJSONResponse(_ data: Data) -> Bool {
    for byte in data where byte != 0x20 && byte != 0x09 && byte != 0x0d {
        return byte == 0x7b
    }
    return false
}
