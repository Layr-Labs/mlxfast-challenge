import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXNN

private let gemma4CombinedQKVPrefillPreparationEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_COMBINED_QKV_PREFILL_PREP"
    ] else {
        return true
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

private let gemma4VerifyCombinedQKVPrefillPreparationBits: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_VERIFY_COMBINED_QKV_PREFILL_PREP_BITS"
    ] else {
        return false
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

private let gemma4StagedSlidingPrefillAttentionEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_STAGED_SLIDING_PREFILL_ATTENTION"
    ] else {
        return true
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

private let gemma4VerifyStagedSlidingPrefillAttentionBits: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_VERIFY_STAGED_SLIDING_PREFILL_ATTENTION_BITS"
    ] else {
        return false
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

private let gemma4StagedFullPrefillAttentionEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_STAGED_FULL_PREFILL_ATTENTION"
    ] else {
        return true
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

private let gemma4VerifyStagedFullPrefillAttentionBits: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_VERIFY_STAGED_FULL_PREFILL_ATTENTION_BITS"
    ] else {
        return false
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// Rollback switch for the last-layer M=64 tail prune.
///
/// Default ON. At prefill lengths >= 128 the final transformer layer's
/// post-attention chain (o_proj -> post-attn boundary -> gate/up -> gelu*mul
/// -> down -> post-FFN boundary) influences only the last row's logits, so
/// the chain runs on the last 64 supplied rows instead of all L. The
/// layer's qkv/prep/attention/KV-write stays full-width (decode needs KV at
/// every position). Every dispatched op in the pruned chain is
/// row-independent at M=64 vs M=512: the frozen host dispatch sends both to
/// the identical `affine_qmm_t_nax` pipeline (qmm_splitk picks split_k=1
/// for N=5376 and N=21504 at both M values, and the kernel has a strictly
/// ascending per-element K chain with no cross-row reduction), and the
/// RMS/elementwise ops are per-row at any M, so the retained rows are
/// bit-identical to the full-width chain. Set
/// `DARKBLOOM_LAST_LAYER_TAIL_PRUNE=0` to restore the full-width chain.
let gemma4LastLayerTailPruneEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_LAST_LAYER_TAIL_PRUNE"
    ] else {
        return true
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// Raw-bit verify switch for the last-layer tail prune. Default OFF. When
/// enabled, the final layer additionally runs the unpruned full-width chain
/// and preconditions that its last 64 rows bit-match the pruned output.
let gemma4VerifyLastLayerTailPruneBits: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_VERIFY_LAST_LAYER_TAIL_PRUNE_BITS"
    ] else {
        return false
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// Rollback switch for the last-layer Q-side prune.
///
/// Default ON. At prefill lengths >= 128 the final layer (full attention,
/// K==V shared) computes Q only for the last 64 supplied rows: the pruned
/// post-attention chain consumes only those rows, Q is never cached, and
/// logits are last-row only. K/V projection, K/V preparation, and the KV
/// cache write stay full-width (decode needs KV at every position). The
/// pruned attention runs the stock C++ SDPA fallback at [1, 32, 64, 512]
/// with `.causal` (whose mask offset kL - qL = L - 64 is exactly the global
/// causal bound of the retained rows). Every dispatched op is
/// row-independent: separate q/k QMMs preserve the combined projection's
/// per-element K chains (the CombinedAttentionPrefillProjection.verifyBits
/// invariant), qmm_splitk selects split_k=1 at both M=512 and M=64 for
/// N=16384/2048, and the attention fallback does not split K at these
/// shapes, so the retained rows are bit-identical to the full-width path.
/// Set `DARKBLOOM_LAST_LAYER_Q_PRUNE=0` to restore full-width Q.
let gemma4LastLayerQPruneEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_LAST_LAYER_Q_PRUNE"
    ] else {
        return true
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// Raw-bit verify switch for the last-layer Q-side prune. Default OFF. When
/// enabled, the final layer additionally runs the full-width stock-SDPA
/// attention and chain, and preconditions that the pruned 64-row attention
/// output and the pruned layer output bit-match its last 64 rows.
let gemma4VerifyLastLayerQPruneBits: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_VERIFY_LAST_LAYER_Q_PRUNE_BITS"
    ] else {
        return false
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// Affine 4-bit projection extracted from a loaded QuantizedLinear.
struct FastQuantizedProjection: @unchecked Sendable {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray?
    let groupSize: Int
    let bits: Int

    init(_ linear: QuantizedLinear) {
        self.weight = linear.weight
        self.scales = linear.scales
        self.biases = linear.biases
        self.groupSize = linear.groupSize
        self.bits = linear.bits
    }

    init(
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int
    ) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.groupSize = groupSize
        self.bits = bits
    }

    @inline(__always)
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return quantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: .affine
        )
    }

}

private struct FastMLPTailWeights: @unchecked Sendable {
    let preNorm: MLXArray
    let postNorm: MLXArray
    let layerScalar: MLXArray
}

/// Approximate tanh-GELU using `x*x*x` (compile-safe), matching the library.
/// Internal (not private) so GPU verification harnesses can reference the
/// exact closure the engine compiles into its MLP tails.
let fastGeluApproximate: @Sendable (MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray) -> MLXArray = { x in
        0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
    }
    let enabled: Bool
    if let raw = ProcessInfo.processInfo.environment["MLX_COMPILED_DECODE"] {
        enabled = ["1", "true", "yes", "on"].contains(raw.lowercased())
    } else {
        enabled = true
    }
    return enabled ? compile(shapeless: true, body) : body
}()

/// One decoder layer rebuilt from the loaded library modules.
///
/// Attention stays eager with active-length KV caches (bit-parity with the
/// baseline). The gated MLP is fused into one compiled closure to collapse
/// gate/up/GELU/product/down graph-construction overhead.
struct Gemma4FastLayerResult {
    let hidden: MLXArray
    let nextNormalized: MLXArray?
    let keyValue: (MLXArray, MLXArray)
}

final class Gemma4FastLayer {
    let isSliding: Bool
    let nHeads: Int
    let nKvHeads: Int
    let headDim: Int
    let useKEqV: Bool
    let scale: Float = 1.0
    let eps: Float

    let qProj: FastQuantizedProjection
    let kProj: FastQuantizedProjection
    let vProj: FastQuantizedProjection?
    let oProj: FastQuantizedProjection
    let indexedOutput: IndexedOutputProjection?
    let fusedQKV: FusedSlidingQKVProjection?
    let fusedQK: FusedFullQKProjection?
    let combinedAttentionPrefill: CombinedAttentionPrefillProjection?
    let fusedAttentionRMS: FusedAttentionRMSPreparation?
    let fusedAttentionToMLPBoundary: FusedAttentionToMLPBoundary?
    let fusedMLPToNextBoundary: FusedMLPToNextBoundary?

    let qNormWeight: MLXArray
    let kNormWeight: MLXArray?
    let inputNormWeight: MLXArray
    let postAttnNormWeight: MLXArray
    let preFfnNormWeight: MLXArray
    let postFfnNormWeight: MLXArray
    let layerScalar: MLXArray

    let rope: RoPELayer
    let fusedMLP: @Sendable (MLXArray) -> MLXArray
    let fusedMLPTail: (@Sendable (MLXArray, MLXArray) -> MLXArray)?
    let prefillGeluEpilogue: Gemma4PrefillGeluEpilogueMLPTail?
    let fusedGateUp: FusedGateUpProjection?
    let fusedGateUpPostTail: (@Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray)?
    let fusedGateUpActivation: (@Sendable (MLXArray, MLXArray) -> MLXArray)?
    let indexedDown: IndexedDownProjection?
    let indexedDownPostTail: (@Sendable (MLXArray, MLXArray) -> MLXArray)?
    let useFusedGateUpActivation: Bool

    init(
        isSliding: Bool,
        nHeads: Int,
        nKvHeads: Int,
        headDim: Int,
        useKEqV: Bool,
        eps: Float,
        qProj: QuantizedLinear,
        kProj: QuantizedLinear,
        vProj: QuantizedLinear?,
        oProj: QuantizedLinear,
        qNorm: RMSNorm,
        kNorm: RMSNorm?,
        inputNorm: RMSNorm,
        postAttnNorm: RMSNorm,
        preFfnNorm: RMSNorm,
        postFfnNorm: RMSNorm,
        gate: QuantizedLinear,
        up: QuantizedLinear,
        down: QuantizedLinear,
        layerScalar: MLXArray,
        nextInputNormWeight: MLXArray?,
        rope: RoPELayer,
        qIndexedMetadata: IndexedAffineMetadata?,
        kIndexedMetadata: IndexedAffineMetadata?,
        vIndexedMetadata: IndexedAffineMetadata?,
        gateIndexedMetadata: IndexedAffineMetadata?,
        upIndexedMetadata: IndexedAffineMetadata?,
        downIndexedMetadata: IndexedAffineMetadata?,
        qPackedIndexMetadata: Gemma4PackedQKVIndexMetadata? = nil,
        kPackedIndexMetadata: Gemma4PackedQKVIndexMetadata? = nil,
        vPackedIndexMetadata: Gemma4PackedQKVIndexMetadata? = nil,
        coTiledAttentionPayload: Gemma4CoTiledAttentionPayload? = nil
    ) {
        self.isSliding = isSliding
        self.nHeads = nHeads
        self.nKvHeads = nKvHeads
        self.headDim = headDim
        self.useKEqV = useKEqV
        self.eps = eps
        let qProjection = FastQuantizedProjection(qProj)
        let kProjection = FastQuantizedProjection(kProj)
        let vProjection = vProj.map(FastQuantizedProjection.init)
        self.qProj = qProjection
        self.kProj = kProjection
        self.vProj = vProjection
        let outputProjection = FastQuantizedProjection(oProj)
        self.oProj = outputProjection
        let indexedOutputEnabled: Bool
        if let raw = ProcessInfo.processInfo.environment["MLXFAST_INDEXED_OUTPUT_FAST"] {
            indexedOutputEnabled = ["1", "true", "yes", "on"].contains(
                raw.lowercased())
        } else {
            indexedOutputEnabled = true
        }
        self.indexedOutput = indexedOutputEnabled
            ? IndexedOutputProjection(projection: outputProjection)
            : nil
        let fusedQKVEnabled: Bool
        if let raw = ProcessInfo.processInfo.environment["MLXFAST_FUSED_QKV"] {
            fusedQKVEnabled = ["1", "true", "yes", "on"].contains(
                raw.lowercased())
        } else {
            fusedQKVEnabled = true
        }
        if fusedQKVEnabled,
           isSliding,
           let vProjection,
           let qIndexedMetadata,
           let kIndexedMetadata,
           let vIndexedMetadata,
           supportsGemma4FusedSlidingQKV(
               q: qProjection,
               k: kProjection,
               v: vProjection,
               qMetadata: qIndexedMetadata,
               kMetadata: kIndexedMetadata,
               vMetadata: vIndexedMetadata
           )
        {
            self.fusedQKV = FusedSlidingQKVProjection(
                q: qProjection,
                k: kProjection,
                v: vProjection,
                qMetadata: qIndexedMetadata,
                kMetadata: kIndexedMetadata,
                vMetadata: vIndexedMetadata,
                qPackedMetadata: qPackedIndexMetadata,
                kPackedMetadata: kPackedIndexMetadata,
                vPackedMetadata: vPackedIndexMetadata,
                coTiledPayload: coTiledAttentionPayload?.kind == .slidingQKV
                    ? coTiledAttentionPayload
                    : nil
            )
        } else {
            self.fusedQKV = nil
        }
        let fusedQKEnabled: Bool
        if let raw = ProcessInfo.processInfo.environment["MLXFAST_FUSED_FULL_QK"] {
            fusedQKEnabled = ["1", "true", "yes", "on"].contains(
                raw.lowercased())
        } else {
            fusedQKEnabled = true
        }
        if fusedQKEnabled,
           !isSliding,
           useKEqV,
           let qIndexedMetadata,
           let kIndexedMetadata,
           supportsGemma4FusedFullQK(
               q: qProjection,
               k: kProjection,
               qMetadata: qIndexedMetadata,
               kMetadata: kIndexedMetadata
           )
        {
            self.fusedQK = FusedFullQKProjection(
                q: qProjection,
                k: kProjection,
                qMetadata: qIndexedMetadata,
                kMetadata: kIndexedMetadata,
                qPackedMetadata: qPackedIndexMetadata,
                kPackedMetadata: kPackedIndexMetadata,
                coTiledPayload: coTiledAttentionPayload?.kind == .fullQK
                    ? coTiledAttentionPayload
                    : nil
            )
        } else {
            self.fusedQK = nil
        }
        let combinedAttentionPrefillEnabled: Bool
        if let raw = ProcessInfo.processInfo.environment[
            "MLXFAST_COMBINED_ATTENTION_PREFILL"
        ] {
            combinedAttentionPrefillEnabled = ["1", "true", "yes", "on"]
                .contains(raw.lowercased())
        } else {
            combinedAttentionPrefillEnabled = true
        }
        let verifyCombinedAttentionPrefill: Bool
        if let raw = ProcessInfo.processInfo.environment[
            "MLXFAST_VERIFY_COMBINED_ATTENTION_PREFILL"
        ] {
            verifyCombinedAttentionPrefill = ["1", "true", "yes", "on"]
                .contains(raw.lowercased())
        } else {
            verifyCombinedAttentionPrefill = false
        }
        self.combinedAttentionPrefill = combinedAttentionPrefillEnabled
            ? CombinedAttentionPrefillProjection(
                q: qProjection,
                k: kProjection,
                v: vProjection,
                verifyBits: verifyCombinedAttentionPrefill
            )
            : nil
        self.qNormWeight = qNorm.weight
        self.kNormWeight = kNorm?.weight
        let fusedAttentionRMSEnabled: Bool
        if let raw = ProcessInfo.processInfo.environment[
            "MLXFAST_FUSED_ATTENTION_RMS"
        ] {
            fusedAttentionRMSEnabled = ["1", "true", "yes", "on"]
                .contains(raw.lowercased())
        } else {
            fusedAttentionRMSEnabled = true
        }
        self.fusedAttentionRMS = fusedAttentionRMSEnabled
            ? FusedAttentionRMSPreparation(
                isSliding: isSliding,
                headDim: headDim,
                kvHeads: nKvHeads,
                qNormWeight: qNorm.weight,
                kNormWeight: kNorm?.weight,
                eps: eps
            )
            : nil
        self.inputNormWeight = inputNorm.weight
        self.postAttnNormWeight = postAttnNorm.weight
        self.preFfnNormWeight = preFfnNorm.weight
        self.postFfnNormWeight = postFfnNorm.weight
        self.layerScalar = layerScalar
        let fusedAttentionToMLPBoundaryEnabled: Bool
        if let raw = ProcessInfo.processInfo.environment[
            "MLXFAST_FUSED_ATTENTION_TO_MLP_BOUNDARY"
        ] {
            fusedAttentionToMLPBoundaryEnabled = ["1", "true", "yes", "on"]
                .contains(raw.lowercased())
        } else {
            fusedAttentionToMLPBoundaryEnabled = true
        }
        self.fusedAttentionToMLPBoundary = fusedAttentionToMLPBoundaryEnabled
            ? FusedAttentionToMLPBoundary(
                postAttentionWeight: postAttnNorm.weight,
                preFFNWeight: preFfnNorm.weight,
                eps: eps
            )
            : nil
        let fusedMLPToNextBoundaryEnabled: Bool
        if let raw = ProcessInfo.processInfo.environment[
            "MLXFAST_FUSED_MLP_TO_NEXT_BOUNDARY"
        ] {
            fusedMLPToNextBoundaryEnabled = ["1", "true", "yes", "on"]
                .contains(raw.lowercased())
        } else {
            fusedMLPToNextBoundaryEnabled = true
        }
        self.fusedMLPToNextBoundary = fusedMLPToNextBoundaryEnabled
            ? nextInputNormWeight.flatMap {
                FusedMLPToNextBoundary(
                    postFFNWeight: postFfnNorm.weight,
                    layerScalar: layerScalar,
                    nextNormWeight: $0,
                    eps: eps
                )
            }
            : nil
        self.rope = rope

        // Fuse the gated MLP into one compiled closure: collapses gate/up/GELU/
        // product/down graph construction. Keep two quantizedMMs (not a packed
        // gate+up): a single 2x-wide matmul was slower on this silicon.
        let gateP = FastQuantizedProjection(gate)
        let upP = FastQuantizedProjection(up)
        let downP = FastQuantizedProjection(down)
        let gelu = fastGeluApproximate
        let body: @Sendable (MLXArray) -> MLXArray = { x in
            downP(gelu(gateP(x)) * upP(x))
        }
        let compileEnabled: Bool
        if let raw = ProcessInfo.processInfo.environment["MLX_COMPILED_DECODE"] {
            compileEnabled = ["1", "true", "yes", "on"].contains(raw.lowercased())
        } else {
            compileEnabled = true
        }
        self.fusedMLP = compileEnabled ? compile(shapeless: true, body) : body

        let tailWeights = FastMLPTailWeights(
            preNorm: preFfnNorm.weight,
            postNorm: postFfnNorm.weight,
            layerScalar: layerScalar
        )
        let tailBody: @Sendable (MLXArray, MLXArray) -> MLXArray = { x, residual in
            let normalized = MLXFast.rmsNorm(x, weight: tailWeights.preNorm, eps: eps)
            let mlp = downP(gelu(gateP(normalized)) * upP(normalized))
            let postNormalized = MLXFast.rmsNorm(mlp, weight: tailWeights.postNorm, eps: eps)
            return (residual + postNormalized) * tailWeights.layerScalar
        }
        let tailEnabled: Bool
        if let raw = ProcessInfo.processInfo.environment["MLX_COMPILED_MLP_TAIL"] {
            tailEnabled = ["1", "true", "yes", "on"].contains(raw.lowercased())
        } else {
            // Keep the extra fusion aligned with the library's compiled
            // decode capability switch. It preserves attention's eager
            // reduction path while removing four MLP-side graph boundaries.
            tailEnabled = compileEnabled
        }
        self.fusedMLPTail = tailEnabled ? compile(shapeless: true, tailBody) : nil

        // Fused prefill up-projection with GELU(gate)*up store epilogue
        // (DARKBLOOM_PREFILL_GELU_EPILOGUE, default on): replaces the stock
        // up qmm + compiled gelu·mul inside the prefill MLP tail with one NAX
        // qmm clone whose store epilogue emits the activation directly,
        // bit-exactly. Requires the stock compiled tail as the verify
        // reference and the standard affine g64/b4 projection layouts;
        // anything else keeps the stock tail.
        if gemma4PrefillGeluEpilogueEnabled, let stockTail = self.fusedMLPTail {
            self.prefillGeluEpilogue = Gemma4PrefillGeluEpilogueMLPTail(
                gate: gateP,
                up: upP,
                down: downP,
                preNormWeight: preFfnNorm.weight,
                postNormWeight: postFfnNorm.weight,
                layerScalar: layerScalar,
                eps: eps,
                stockTail: stockTail
            )
        } else {
            self.prefillGeluEpilogue = nil
        }

        let fusedGateUpEnabled: Bool
        if let raw = ProcessInfo.processInfo.environment["MLXFAST_FUSED_GATE_UP"] {
            fusedGateUpEnabled = ["1", "true", "yes", "on"].contains(raw.lowercased())
        } else {
            fusedGateUpEnabled = true
        }
        if fusedGateUpEnabled && tailEnabled
            && supportsGemma4FusedGateUp(gate: gateP, up: upP)
        {
            let metadataMode: FusedGateUpMetadataMode
            if let raw = ProcessInfo.processInfo.environment[
                "MLXFAST_FUSED_GATE_UP_METADATA"
            ] {
                guard let parsed = FusedGateUpMetadataMode(rawValue: raw.lowercased()) else {
                    preconditionFailure(
                        "MLXFAST_FUSED_GATE_UP_METADATA must be raw or indexed")
                }
                metadataMode = parsed
            } else {
                metadataMode = gateIndexedMetadata != nil && upIndexedMetadata != nil
                    ? .indexed
                    : .raw
            }
            self.fusedGateUp = FusedGateUpProjection(
                gate: gateP,
                up: upP,
                metadataMode: metadataMode,
                gateIndexedMetadata: gateIndexedMetadata,
                upIndexedMetadata: upIndexedMetadata
            )
            let postTailBody: @Sendable (
                MLXArray, MLXArray, MLXArray
            ) -> MLXArray = { gateOutput, upOutput, residual in
                let mlp = downP(gelu(gateOutput) * upOutput)
                let postNormalized = MLXFast.rmsNorm(
                    mlp, weight: tailWeights.postNorm, eps: eps)
                return (residual + postNormalized) * tailWeights.layerScalar
            }
            // CustomKernel cannot participate in MLX compile because it cannot
            // infer output shapes. Keep the QMV eager and compile its suffix.
            self.fusedGateUpPostTail = compile(shapeless: true, postTailBody)

            let indexedDownEnabled: Bool
            if let raw = ProcessInfo.processInfo.environment["MLXFAST_INDEXED_DOWN"] {
                indexedDownEnabled = ["1", "true", "yes", "on"].contains(
                    raw.lowercased())
            } else {
                indexedDownEnabled = downIndexedMetadata != nil
            }
            if indexedDownEnabled,
               metadataMode == .indexed,
               let downIndexedMetadata,
               supportsGemma4IndexedDown(
                   projection: downP,
                   metadata: downIndexedMetadata
               )
            {
                let activationBody: @Sendable (MLXArray, MLXArray) -> MLXArray = {
                    gateOutput, upOutput in
                    gelu(gateOutput) * upOutput
                }
                self.fusedGateUpActivation = compile(
                    shapeless: true,
                    activationBody
                )
                self.indexedDown = IndexedDownProjection(
                    projection: downP,
                    metadata: downIndexedMetadata
                )
                let postDownBody: @Sendable (MLXArray, MLXArray) -> MLXArray = {
                    mlp, residual in
                    let postNormalized = MLXFast.rmsNorm(
                        mlp, weight: tailWeights.postNorm, eps: eps)
                    return (residual + postNormalized) * tailWeights.layerScalar
                }
                self.indexedDownPostTail = compile(
                    shapeless: true,
                    postDownBody
                )
                if let raw = ProcessInfo.processInfo.environment[
                    "MLXFAST_FUSED_GATE_UP_ACTIVATION"
                ] {
                    self.useFusedGateUpActivation = ["1", "true", "yes", "on"]
                        .contains(raw.lowercased())
                } else {
                    self.useFusedGateUpActivation = true
                }
            } else {
                self.fusedGateUpActivation = nil
                self.indexedDown = nil
                self.indexedDownPostTail = nil
                self.useFusedGateUpActivation = false
            }
        } else {
            self.fusedGateUp = nil
            self.fusedGateUpPostTail = nil
            self.fusedGateUpActivation = nil
            self.indexedDown = nil
            self.indexedDownPostTail = nil
            self.useFusedGateUpActivation = false
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        normalizedInput: MLXArray?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        pruneTail: Bool = false
    ) -> Gemma4FastLayerResult {
        let residual = x
        let h: MLXArray
        if let normalizedInput {
            precondition(x.dim(0) == 1 && x.dim(1) == 1)
            precondition(normalizedInput.shape == x.shape)
            precondition(normalizedInput.dtype == .bfloat16)
            h = normalizedInput
        } else {
            h = MLXFast.rmsNorm(x, weight: inputNormWeight, eps: eps)
        }

        let (B, L, _) = (h.dim(0), h.dim(1), h.dim(2))
        let offset = cache?.offset ?? 0

        // C1: last-layer Q-side prune. Only full-attention K==V layers at
        // prefill lengths >= 128, and only when the K/V-only combined
        // prefill path is available (the ranked prefill shape always has
        // both). The retained Q rows are the last 64 supplied rows, at their
        // global RoPE offset.
        let qPruneCapacity = pruneTail
            && gemma4LastLayerQPruneEnabled
            && B == 1
            && L >= 128
            && !isSliding
            && vProj == nil
            && h.dtype == .bfloat16
            && gemma4CombinedKVPrefillEnabled()
            && fusedAttentionRMS?.supportsPrefill(
                offset: offset, length: L) == true
            ? (cache as? Gemma4CombinedKVCache)?.directPrefillCapacity(for: L)
            : nil
        let qPruneRows = qPruneCapacity != nil ? 64 : 0
        let qPruneStart = L - qPruneRows

        let rawQueries: MLXArray
        let rawKeys: MLXArray
        let rawValues: MLXArray?
        if B == 1, L == 1, let fusedQKV {
            let projected = fusedQKV(h)
            rawQueries = projected.0
            rawKeys = projected.1
            rawValues = projected.2
        } else if B == 1, L == 1, let fusedQK {
            let projected = fusedQK(h)
            rawQueries = projected.0
            rawKeys = projected.1
            rawValues = nil
        } else if qPruneRows > 0 {
            // K full-width (decode needs every row's KV); Q on the retained
            // 64 rows only. Separate q/k QMMs preserve the combined
            // projection's per-element K chains (the
            // CombinedAttentionPrefillProjection.verifyBits invariant), and
            // qmm_splitk selects split_k=1 at both M=512 and M=64 for
            // N=16384 and N=2048, so the retained rows are bit-identical.
            // The BN32 clone dispatches are bit-identical to those stock
            // QMMs at both widths, so the invariant is preserved.
            rawQueries = gemma4PrefillBN32QMMDispatchIfSupported(
                h[0..., qPruneStart..<L, 0...], projection: qProj)
            rawKeys = gemma4PrefillBN32QMMDispatchIfSupported(
                h, projection: kProj)
            rawValues = nil
        } else if B == 1,
                  L > 1,
                  h.dtype == .bfloat16,
                  let combinedAttentionPrefill
        {
            let projected = combinedAttentionPrefill(h)
            rawQueries = projected.queries
            rawKeys = projected.keys
            rawValues = projected.values
        } else {
            rawQueries = qProj(h)
            rawKeys = kProj(h)
            rawValues = vProj?(h)
        }

        var queries: MLXArray
        var keys: MLXArray
        var values: MLXArray
        let usesFusedAttentionPreparation = B == 1
            && L == 1
            && fusedAttentionRMS?.supports(offset: offset) == true
        let combinedCache = gemma4CombinedKVDirectEnabled()
            ? cache as? Gemma4CombinedKVCache
            : nil
        let usesCombinedKVDecodePreparation = usesFusedAttentionPreparation
            && combinedCache != nil
        let combinedPrefillCapacity = B == 1
            && L > 1
            && gemma4CombinedKVPrefillEnabled()
            && fusedAttentionRMS?.supportsPrefill(offset: offset, length: L) == true
            ? combinedCache?.directPrefillCapacity(for: L)
            : nil
        let usesCombinedKVPrefillPreparation = combinedPrefillCapacity != nil
        let usesCombinedQKVPrefillPreparation =
            usesCombinedKVPrefillPreparation
            && offset == 0
            && h.dtype == .bfloat16
            && combinedAttentionPrefill != nil
            && (gemma4CombinedQKVPrefillPreparationEnabled
                || gemma4VerifyCombinedQKVPrefillPreparationBits)
        if usesCombinedKVDecodePreparation,
           let fusedAttentionRMS,
           let combinedCache
        {
            if gemma4DecodeKVDirectWriteEnabled(),
               let target = combinedCache.directDecodeWriteTarget()
            {
                // The preparation kernel writes the K/V rows straight into the
                // combined slab at `target.position`; no slice update dispatch.
                queries = fusedAttentionRMS.callCombinedDecodeDirect(
                    rawQueries: rawQueries,
                    rawKeys: rawKeys,
                    rawValues: rawValues,
                    offset: offset,
                    cacheStorage: target.storage,
                    writePosition: target.position,
                    capacity: target.storage.dim(3)
                )
                let updated = combinedCache.adoptDirectDecodeWrite(
                    position: target.position
                )
                keys = updated.0
                values = updated.1
                if gemma4VerifyDecodeKVDirectWriteBitsEnabled() {
                    gemma4VerifyDecodeKVDirectWrite(
                        queries: queries,
                        cacheStorage: target.storage,
                        position: target.position,
                        fusedAttentionRMS: fusedAttentionRMS,
                        rawQueries: rawQueries,
                        rawKeys: rawKeys,
                        rawValues: rawValues,
                        offset: offset
                    )
                }
            } else {
                let prepared = fusedAttentionRMS.callCombined(
                    rawQueries: rawQueries,
                    rawKeys: rawKeys,
                    rawValues: rawValues,
                    offset: offset
                )
                queries = prepared.queries
                let updated = combinedCache.updateCombined(prepared.combinedKV)
                keys = updated.0
                values = updated.1
            }
        } else if usesFusedAttentionPreparation, let fusedAttentionRMS {
            let prepared = fusedAttentionRMS(
                rawQueries: rawQueries,
                rawKeys: rawKeys,
                rawValues: rawValues,
                offset: offset
            )
            queries = prepared.0
            keys = prepared.1
            values = prepared.2
        } else if qPruneRows > 0,
                  let fusedAttentionRMS,
                  let combinedCache,
                  let qPruneCapacity
        {
            // Q: stock eager path on the 64 retained rows at their global
            // RoPE offset (rmsNorm and RoPE are row-independent, so these
            // rows are bit-identical to the full-width preparation).
            queries = rawQueries.reshaped(B, qPruneRows, nHeads, headDim)
            queries = MLXFast.rmsNorm(queries, weight: qNormWeight, eps: eps)
            queries = queries.transposed(0, 2, 1, 3)
            queries = rope(queries, offset: qPruneStart)
            // K/V: the same K/V-only combined prefill the unpruned path
            // uses, full-width, so decode keeps KV at every position.
            let combined = fusedAttentionRMS.callCombinedPrefill(
                rawKeys: rawKeys,
                rawValues: rawValues,
                offset: offset,
                length: L,
                capacity: qPruneCapacity
            )
            let updated = combinedCache.adoptDirectPrefill(combined, length: L)
            keys = updated.0
            values = updated.1
        } else if usesCombinedQKVPrefillPreparation,
                  let fusedAttentionRMS,
                  let combinedCache,
                  let combinedPrefillCapacity
        {
            let candidate = fusedAttentionRMS.callCombinedQKVPrefill(
                rawQueries: rawQueries,
                rawKeys: rawKeys,
                rawValues: rawValues,
                offset: offset,
                length: L,
                capacity: combinedPrefillCapacity
            )
            var selectedQueries = candidate.queries
            var selectedCombinedKV = candidate.combinedKV

            if gemma4VerifyCombinedQKVPrefillPreparationBits {
                var referenceQueries = rawQueries.reshaped(
                    B, L, nHeads, headDim)
                referenceQueries = MLXFast.rmsNorm(
                    referenceQueries, weight: qNormWeight, eps: eps)
                referenceQueries = referenceQueries.transposed(0, 2, 1, 3)
                referenceQueries = rope(referenceQueries, offset: offset)
                let referenceCombinedKV = fusedAttentionRMS.callCombinedPrefill(
                    rawKeys: rawKeys,
                    rawValues: rawValues,
                    offset: offset,
                    length: L,
                    capacity: combinedPrefillCapacity
                )
                let queriesMatch = arrayEqual(
                    candidate.queries.view(dtype: .uint16),
                    referenceQueries.view(dtype: .uint16)
                )
                let combinedKVMatches = arrayEqual(
                    candidate.combinedKV[0..., 0..., 0..., 0..<L, 0...]
                        .view(dtype: .uint16),
                    referenceCombinedKV[0..., 0..., 0..., 0..<L, 0...]
                        .view(dtype: .uint16)
                )
                eval(queriesMatch, combinedKVMatches)
                precondition(
                    queriesMatch.item(Bool.self),
                    "combined QKV prefill queries differ from stock RMSNorm/RoPE"
                )
                precondition(
                    combinedKVMatches.item(Bool.self),
                    "combined QKV prefill cache differs from K/V-only preparation"
                )
                if !gemma4CombinedQKVPrefillPreparationEnabled {
                    selectedQueries = referenceQueries
                    selectedCombinedKV = referenceCombinedKV
                }
            }

            queries = selectedQueries
            let updated = combinedCache.adoptDirectPrefill(
                selectedCombinedKV, length: L)
            keys = updated.0
            values = updated.1
        } else if usesCombinedKVPrefillPreparation,
                  let fusedAttentionRMS,
                  let combinedCache,
                  let combinedPrefillCapacity
        {
            // Preserve the stock query path exactly. K/V normalization,
            // transposition, RoPE, cache layout, and capacity reservation are
            // emitted directly by one multi-token Metal kernel. `rawQueries`
            // may come from the promoted combined Q/K/V prefill projection;
            // keeping it here preserves that dispatch and its single QMM.
            queries = rawQueries.reshaped(B, L, nHeads, headDim)
            queries = MLXFast.rmsNorm(queries, weight: qNormWeight, eps: eps)
            let combined = fusedAttentionRMS.callCombinedPrefill(
                rawKeys: rawKeys,
                rawValues: rawValues,
                offset: offset,
                length: L,
                capacity: combinedPrefillCapacity
            )
            let updated = combinedCache.adoptDirectPrefill(combined, length: L)
            keys = updated.0
            values = updated.1
            if gemma4VerifyCombinedKVPrefillBitsEnabled() {
                let shapedKeys = rawKeys.reshaped(B, L, nKvHeads, headDim)
                var referenceKeys = MLXFast.rmsNorm(
                    shapedKeys, weight: kNormWeight!, eps: eps)
                    .transposed(0, 2, 1, 3)
                referenceKeys = rope(referenceKeys, offset: offset)
                let referenceValueInput = rawValues?.reshaped(
                    B, L, nKvHeads, headDim
                ) ?? shapedKeys
                let referenceValues = MLXFast.rmsNorm(
                    referenceValueInput,
                    weight: MLXArray.mlxNone,
                    eps: eps
                ).transposed(0, 2, 1, 3)
                let keysMatch = arrayEqual(
                    keys.view(dtype: .uint16),
                    referenceKeys.view(dtype: .uint16)
                )
                let valuesMatch = arrayEqual(
                    values.view(dtype: .uint16),
                    referenceValues.view(dtype: .uint16)
                )
                eval(keysMatch, valuesMatch)
                precondition(
                    keysMatch.item(Bool.self) && valuesMatch.item(Bool.self),
                    "direct combined KV prefill differs from stock RMSNorm/RoPE"
                )
            }
        } else {
            queries = rawQueries.reshaped(B, L, nHeads, headDim)
            queries = MLXFast.rmsNorm(queries, weight: qNormWeight, eps: eps)

            let shapedKeys = rawKeys.reshaped(B, L, nKvHeads, headDim)
            keys = MLXFast.rmsNorm(shapedKeys, weight: kNormWeight!, eps: eps)
            keys = keys.transposed(0, 2, 1, 3)

            if let rawValues {
                values = rawValues.reshaped(B, L, nKvHeads, headDim)
            } else {
                values = shapedKeys
            }
            values = MLXFast.rmsNorm(
                values, weight: MLXArray.mlxNone, eps: eps)
            values = values.transposed(0, 2, 1, 3)
        }
        if !usesFusedAttentionPreparation && !usesCombinedKVPrefillPreparation {
            keys = rope(keys, offset: offset)
        }

        if let cache,
           !usesCombinedKVDecodePreparation,
           !usesCombinedKVPrefillPreparation
        {
            let updated = cache.update(keys: keys, values: values)
            keys = updated.0
            values = updated.1
        }

        if !usesFusedAttentionPreparation && !usesCombinedQKVPrefillPreparation {
            queries = queries.transposed(0, 2, 1, 3)
        }
        if !usesFusedAttentionPreparation && !usesCombinedQKVPrefillPreparation {
            queries = rope(queries, offset: offset)
        }

        var attentionMask = mask
        if case .array(let maskArray) = mask {
            let keysSeqLen = keys.dim(2)
            if maskArray.dim(-1) != keysSeqLen {
                attentionMask = .array(maskArray[.ellipsis, 0..<keysSeqLen])
            }
        }

        let mergedAttention: MLXArray
        let canUseStagedSlidingPrefill = B == 1
            && L == 512
            && offset == 0
            && isSliding
            && queries.dtype == .bfloat16
            && queries.shape == [1, 32, 512, 256]
            && keys.shape == [1, 16, 512, 256]
            && values.shape == [1, 16, 512, 256]
        // Full-attention analog (P2): the ten full layers at the ranked
        // prefill shape. At B=1, L=512, offset 0 the engine-constructed full
        // mask is always the symbolic `.causal` (no window), which is exactly
        // the mask the staged kernel reproduces.
        let canUseStagedFullPrefill = B == 1
            && L == 512
            && offset == 0
            && !isSliding
            && queries.dtype == .bfloat16
            && queries.shape == [1, 32, 512, 512]
            && keys.shape == [1, 4, 512, 512]
            && values.shape == [1, 4, 512, 512]
        if qPruneRows > 0 {
            // Stock C++ SDPA fallback on the 64 retained query rows. Its
            // `.causal` mask uses offset kL - qL = L - 64 (fast.cpp), which
            // is exactly the retained rows' global causal bound; every row's
            // QK/softmax/PV reduction is row-independent, so the output rows
            // are bit-identical to the full-width attention's last 64 rows.
            let attention = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .causal
            )
            mergedAttention = attention.transposed(0, 2, 1, 3)
                .reshaped(B, qPruneRows, -1)
        } else if canUseStagedSlidingPrefill
            && (gemma4StagedSlidingPrefillAttentionEnabled
                || gemma4VerifyStagedSlidingPrefillAttentionBits)
        {
            // Merges head-major [1, 32, 512, 256] attention into the
            // token-major [1, 512, 32*256] layout o_proj consumes. The
            // reshape of the transposed view materializes a copy.
            func mergeHeadMajor(_ attention: MLXArray) -> MLXArray {
                attention.transposed(0, 2, 1, 3).reshaped(B, L, -1)
            }
            // With token-major emission enabled the staged kernel already
            // writes [1, 512, 32*256] directly (identical values, different
            // addresses), so no transpose-reshape copy is needed here.
            let candidate = gemma4StagedSlidingPrefill512(
                queries: queries,
                keys: keys,
                values: values
            )
            let mergedCandidate = gemma4StagedPrefillTokenMajorOutputEnabled
                ? candidate
                : mergeHeadMajor(candidate)
            if gemma4VerifyStagedSlidingPrefillAttentionBits {
                let reference = MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: attentionMask
                )
                // Compare raw bits in the staged kernel's own output layout.
                // For token-major emission the reference is relaid out with
                // the same transpose+reshape merge -- a bijective relayout,
                // so every element is still compared exactly once.
                let comparisonReference = gemma4StagedPrefillTokenMajorOutputEnabled
                    ? mergeHeadMajor(reference)
                    : reference
                let matches = arrayEqual(
                    candidate.view(dtype: .uint16),
                    comparisonReference.view(dtype: .uint16)
                )
                eval(matches)
                precondition(
                    matches.item(Bool.self),
                    "staged sliding prefill attention differs from stock SDPA"
                )
                mergedAttention = gemma4StagedSlidingPrefillAttentionEnabled
                    ? mergedCandidate
                    : mergeHeadMajor(reference)
            } else {
                mergedAttention = mergedCandidate
            }
        } else if canUseStagedFullPrefill
            && (gemma4StagedFullPrefillAttentionEnabled
                || gemma4VerifyStagedFullPrefillAttentionBits)
        {
            // Merges head-major [1, 32, 512, 512] attention into the
            // token-major [1, 512, 32*512] layout o_proj consumes. The
            // reshape of the transposed view materializes a copy.
            func mergeHeadMajor(_ attention: MLXArray) -> MLXArray {
                attention.transposed(0, 2, 1, 3).reshaped(B, L, -1)
            }
            // With token-major emission enabled the staged kernel already
            // writes [1, 512, 32*512] directly (identical values, different
            // addresses), so no transpose-reshape copy is needed here.
            let candidate = gemma4StagedFullPrefill512(
                queries: queries,
                keys: keys,
                values: values
            )
            let mergedCandidate = gemma4StagedPrefillTokenMajorOutputEnabled
                ? candidate
                : mergeHeadMajor(candidate)
            if gemma4VerifyStagedFullPrefillAttentionBits {
                let reference = MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: attentionMask
                )
                // Compare raw bits in the staged kernel's own output layout.
                // For token-major emission the reference is relaid out with
                // the same transpose+reshape merge -- a bijective relayout,
                // so every element is still compared exactly once.
                let comparisonReference = gemma4StagedPrefillTokenMajorOutputEnabled
                    ? mergeHeadMajor(reference)
                    : reference
                let matches = arrayEqual(
                    candidate.view(dtype: .uint16),
                    comparisonReference.view(dtype: .uint16)
                )
                eval(matches)
                precondition(
                    matches.item(Bool.self),
                    "staged full prefill attention differs from stock SDPA"
                )
                mergedAttention = gemma4StagedFullPrefillAttentionEnabled
                    ? mergedCandidate
                    : mergeHeadMajor(reference)
            } else {
                mergedAttention = mergedCandidate
            }
        } else {
            let attention: MLXArray
            if L > 1 && offset > 0 {
                attention = gemma4FastAttentionFallback(
                    queries: queries,
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: attentionMask
                )
            } else {
                // Prefer library SDPA: D=256 sliding uses fused vector kernel;
                // D=512 full uses its internal fallback. Compiling our own D=512
                // fallback changes the public near-tie reduction order.
                attention = MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: attentionMask
                )
            }
            mergedAttention = attention.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        }
        // Last-layer tail prune: at prefill lengths >= 128 the post-attention
        // chain of the final layer influences only the last row's logits, so
        // o_proj, the boundary, and the whole MLP tail run on the last 64
        // supplied rows (a metadata-only view -- 64 real rows, no padding).
        // The qkv/prep/attention/KV-write above already ran full-width, so
        // decode keeps KV at every position. Every op in the pruned chain is
        // row-independent at M=64 vs full width: the frozen host dispatch
        // (QuantizedMatmul::eval_gpu -> qmm_splitk) selects split_k=1 for
        // N=5376 and N=21504 at both M=512 and M=64 and forwards to the same
        // affine_qmm_t_nax pipeline, whose per-element K chain is strictly
        // ascending with no cross-row reduction; RMS norms and elementwise
        // ops are per-row at any M. The retained rows are therefore
        // bit-identical to the full-width chain.
        let tailStart = pruneTail && gemma4LastLayerTailPruneEnabled
            && qPruneRows == 0 && B == 1 && L >= 128 ? L - 64 : 0
        let chainAttention = tailStart > 0
            ? mergedAttention[0..., tailStart..<L, 0...]
            : mergedAttention
        let chainResidual = tailStart > 0
            ? residual[0..., tailStart..<L, 0...]
            : qPruneRows > 0
                ? residual[0..., qPruneStart..<L, 0...]
                : residual
        let attnOut: MLXArray
        if B == 1, L == 1, let indexedOutput {
            attnOut = indexedOutput(mergedAttention)
        } else {
            // BN32 narrow-residency qmm clone
            // (DARKBLOOM_PREFILL_BN32_QMM, default on): bit-exact vs the
            // stock o_proj quantizedMM at both full width and the M=64
            // pruned tail.
            attnOut = gemma4PrefillBN32QMMDispatchIfSupported(
                chainAttention, projection: oProj)
        }
        var out: MLXArray
        let residual2: MLXArray
        let fusedPreFFNNormalized: MLXArray?
        var nextNormalized: MLXArray? = nil
        if B == 1,
           L == 1,
           fusedGateUp != nil,
           fusedGateUpPostTail != nil,
           let fusedAttentionToMLPBoundary
        {
            let prepared = fusedAttentionToMLPBoundary(
                attentionOutput: attnOut,
                residual: residual
            )
            out = prepared.0
            residual2 = prepared.0
            fusedPreFFNNormalized = prepared.1
        } else {
            out = chainResidual + MLXFast.rmsNorm(
                attnOut, weight: postAttnNormWeight, eps: eps)
            residual2 = out
            fusedPreFFNNormalized = nil
        }
        if B == 1, L == 1, let fusedGateUp, let fusedGateUpPostTail {
            let normalized = fusedPreFFNNormalized ?? MLXFast.rmsNorm(
                out, weight: preFfnNormWeight, eps: eps)
            if let fusedGateUpActivation, let indexedDown, let indexedDownPostTail {
                let activated: MLXArray
                if useFusedGateUpActivation {
                    activated = fusedGateUp.activated(normalized)
                } else {
                    let (gateOutput, upOutput) = fusedGateUp(normalized)
                    activated = fusedGateUpActivation(gateOutput, upOutput)
                }
                let mlp = indexedDown(activated)
                if let fusedMLPToNextBoundary {
                    let prepared = fusedMLPToNextBoundary(
                        mlpOutput: mlp,
                        residual: residual2
                    )
                    out = prepared.0
                    nextNormalized = prepared.1
                } else {
                    out = indexedDownPostTail(mlp, residual2)
                }
            } else {
                let (gateOutput, upOutput) = fusedGateUp(normalized)
                out = fusedGateUpPostTail(gateOutput, upOutput, residual2)
            }
        } else if let fusedMLPTail {
            if let geluEpilogue = prefillGeluEpilogue, geluEpilogue.supports(B: B, L: L) {
                if gemma4VerifyPrefillGeluEpilogueBits {
                    let reference = fusedMLPTail(out, residual2)
                    let candidate = geluEpilogue(out, residual2)
                    let matches = arrayEqual(
                        candidate.view(dtype: .uint16),
                        reference.view(dtype: .uint16)
                    )
                    eval(matches)
                    precondition(
                        matches.item(Bool.self),
                        "prefill GELU epilogue differs from stock MLP tail"
                    )
                    out = candidate
                } else {
                    out = geluEpilogue(out, residual2)
                }
            } else {
                out = fusedMLPTail(out, residual2)
            }
        } else {
            out = MLXFast.rmsNorm(out, weight: preFfnNormWeight, eps: eps)
            out = fusedMLP(out)
            out = MLXFast.rmsNorm(out, weight: postFfnNormWeight, eps: eps)
            out = residual2 + out
            out = out * layerScalar
        }
        if qPruneRows > 0 && gemma4VerifyLastLayerQPruneBits {
            // Full-width reference: stock eager Q over every row plus the
            // stock C++ SDPA fallback -- the unpruned path. The reference Q
            // projection is bit-identical to the combined projection's Q
            // slice (the CombinedAttentionPrefillProjection.verifyBits
            // invariant), and the K/V arrays are the same ones the pruned
            // attention consumed.
            var referenceQueries = qProj(h).reshaped(B, L, nHeads, headDim)
            referenceQueries = MLXFast.rmsNorm(
                referenceQueries, weight: qNormWeight, eps: eps)
            referenceQueries = referenceQueries.transposed(0, 2, 1, 3)
            referenceQueries = rope(referenceQueries, offset: offset)
            let referenceAttention = MLXFast.scaledDotProductAttention(
                queries: referenceQueries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .causal
            )
            let referenceMerged = referenceAttention.transposed(0, 2, 1, 3)
                .reshaped(B, L, -1)
            let attentionMatches = arrayEqual(
                mergedAttention.view(dtype: .uint16),
                referenceMerged[0..., qPruneStart..<L, 0...].view(dtype: .uint16)
            )
            eval(attentionMatches)
            precondition(
                attentionMatches.item(Bool.self),
                "last-layer Q-side prune attention differs from full-width"
            )
            // Chain-level reference: the full-width post-attention chain's
            // last 64 rows must bit-match the pruned chain output. The
            // fusedMLPTail reference is bit-identical to the production
            // chain (the GELU epilogue carries its own raw-bit contract
            // against fusedMLPTail).
            let referenceAttnOut = oProj(referenceMerged)
            let referenceBoundary = residual + MLXFast.rmsNorm(
                referenceAttnOut, weight: postAttnNormWeight, eps: eps)
            let referenceOut: MLXArray
            if let fusedMLPTail {
                referenceOut = fusedMLPTail(
                    referenceBoundary, referenceBoundary)
            } else {
                var full = MLXFast.rmsNorm(
                    referenceBoundary, weight: preFfnNormWeight, eps: eps)
                full = fusedMLP(full)
                full = MLXFast.rmsNorm(
                    full, weight: postFfnNormWeight, eps: eps)
                referenceOut = (referenceBoundary + full) * layerScalar
            }
            let chainMatches = arrayEqual(
                out.view(dtype: .uint16),
                referenceOut[0..., qPruneStart..<L, 0...].view(dtype: .uint16)
            )
            eval(chainMatches)
            precondition(
                chainMatches.item(Bool.self),
                "last-layer Q-side prune output differs from full-width chain"
            )
        }
        if tailStart > 0 && gemma4VerifyLastLayerTailPruneBits {
            // Full-width reference over the identical post-attention ops:
            // the pruned output must bit-match its last 64 rows. The layer's
            // KV slabs are upstream of the prune point (attention ran
            // full-width), so they are identical by construction.
            let referenceAttnOut = oProj(mergedAttention)
            let referenceBoundary = residual + MLXFast.rmsNorm(
                referenceAttnOut, weight: postAttnNormWeight, eps: eps)
            let referenceOut: MLXArray
            if let fusedMLPTail {
                referenceOut = fusedMLPTail(referenceBoundary, referenceBoundary)
            } else {
                var full = MLXFast.rmsNorm(
                    referenceBoundary, weight: preFfnNormWeight, eps: eps)
                full = fusedMLP(full)
                full = MLXFast.rmsNorm(full, weight: postFfnNormWeight, eps: eps)
                referenceOut = (referenceBoundary + full) * layerScalar
            }
            let reference = referenceOut[0..., tailStart..<L, 0...]
            let matches = arrayEqual(
                out.view(dtype: .uint16),
                reference.view(dtype: .uint16)
            )
            eval(matches)
            precondition(
                matches.item(Bool.self),
                "last-layer tail prune output differs from the full-width chain"
            )
        }
        return Gemma4FastLayerResult(
            hidden: out,
            nextNormalized: nextNormalized,
            keyValue: (keys, values)
        )
    }

    var supportsExactMTPPair: Bool {
        let hasQKV = isSliding ? fusedQKV != nil : fusedQK != nil
        return gemma4ExactTwoVectorLayerIsEligible(
            hasQKV: hasQKV,
            hasAttentionPreparation: fusedAttentionRMS != nil,
            hasAttentionBoundary: fusedAttentionToMLPBoundary != nil,
            hasNextBoundary: fusedMLPToNextBoundary != nil,
            hasOutput: indexedOutput?.supportsExactTwoVector == true,
            hasGateUp: fusedGateUp?.supportsExactTwoVector == true,
            hasDown: indexedDown?.supportsExactTwoVector == true,
            usesFusedActivation: useFusedGateUpActivation
        )
    }

    func canRunExactMTPPair(offset: Int) -> Bool {
        supportsExactMTPPair
            && fusedAttentionRMS?.supportsPrefill(
                offset: offset,
                length: 2
            ) == true
    }

    /// Two MTP target rows with the same per-row accumulation order as K=1.
    /// Dense projections share each packed-weight traversal; normalization and
    /// attention retain serial row boundaries.
    func exactMTPPair(
        _ x: MLXArray,
        normalizedInput: MLXArray?,
        cache: Gemma4CombinedKVCache
    ) -> Gemma4FastLayerResult {
        precondition(supportsExactMTPPair)
        precondition(x.dtype == .bfloat16 && x.shape == [2, 5_376])
        precondition(cache.canAppendExactPair())
        let offset = cache.offset

        let h: MLXArray
        if let normalizedInput {
            precondition(
                normalizedInput.dtype == .bfloat16
                    && normalizedInput.shape == x.shape
            )
            h = normalizedInput
        } else {
            h = gemma4SerializedTwoRowRMSNorm(
                x,
                weight: inputNormWeight,
                eps: eps
            )
        }

        let rawQueries: MLXArray
        let rawKeys: MLXArray
        let rawValues: MLXArray?
        if isSliding, let fusedQKV {
            let projected = fusedQKV.exactTwoVector(h)
            rawQueries = projected.queries.reshaped(1, 2, 8_192)
            rawKeys = projected.keys.reshaped(1, 2, 4_096)
            rawValues = projected.values.reshaped(1, 2, 4_096)
        } else if let fusedQK {
            let projected = fusedQK.exactTwoVector(h)
            rawQueries = projected.queries.reshaped(1, 2, 16_384)
            rawKeys = projected.keys.reshaped(1, 2, 2_048)
            rawValues = nil
        } else {
            preconditionFailure("exact MTP pair QKV projection is unavailable")
        }

        guard let fusedAttentionRMS,
              fusedAttentionRMS.supportsPrefill(offset: offset, length: 2)
        else {
            preconditionFailure(
                "exact MTP pair attention preparation is unavailable"
            )
        }
        let prepared = fusedAttentionRMS.callCombinedQKVPrefill(
            rawQueries: rawQueries,
            rawKeys: rawKeys,
            rawValues: rawValues,
            offset: offset,
            length: 2,
            capacity: 2
        )
        let updated = cache.updateCombined(prepared.combinedKV)
        let beforeSecondRow = cache.viewsExcludingNewest(1)
        let attention = gemma4ExactTwoTokenAttention(
            queries: prepared.queries,
            keysBeforeDraft: beforeSecondRow.0,
            valuesBeforeDraft: beforeSecondRow.1,
            keysWithDraft: updated.0,
            valuesWithDraft: updated.1,
            scale: scale
        )
        let mergedAttention = attention.transposed(0, 2, 1, 3)
            .reshaped(2, nHeads * headDim)
        guard let indexedOutput,
              let fusedAttentionToMLPBoundary,
              let fusedGateUp,
              let indexedDown,
              let fusedMLPToNextBoundary
        else {
            preconditionFailure("exact MTP pair layer payload is unavailable")
        }
        let attentionOutput = indexedOutput.exactTwoVector(mergedAttention)
        let attentionBoundary = fusedAttentionToMLPBoundary(
            attentionOutput: attentionOutput,
            residual: x
        )
        let activated = fusedGateUp.exactTwoVectorActivated(
            attentionBoundary.1
        )
        let mlp = indexedDown.exactTwoVector(activated)
        let mlpBoundary = fusedMLPToNextBoundary(
            mlpOutput: mlp,
            residual: attentionBoundary.0
        )
        return Gemma4FastLayerResult(
            hidden: mlpBoundary.0,
            nextNormalized: mlpBoundary.1,
            keyValue: updated
        )
    }
}

func gemma4ExactTwoVectorLayerIsEligible(
    hasQKV: Bool,
    hasAttentionPreparation: Bool,
    hasAttentionBoundary: Bool,
    hasNextBoundary: Bool,
    hasOutput: Bool,
    hasGateUp: Bool,
    hasDown: Bool,
    usesFusedActivation: Bool
) -> Bool {
    hasQKV
        && hasAttentionPreparation
        && hasAttentionBoundary
        && hasNextBoundary
        && hasOutput
        && hasGateUp
        && hasDown
        && usesFusedActivation
}

func gemma4ExactTwoVectorShapeIsSupported(
    _ shape: [Int],
    width: Int
) -> Bool {
    width > 0 && shape == [2, width]
}

private func gemma4SerializedTwoRowRMSNorm(
    _ input: MLXArray,
    weight: MLXArray,
    eps: Float
) -> MLXArray {
    precondition(input.dtype == .bfloat16 && input.shape == [2, 5_376])
    let shaped = input.reshaped(1, 2, 5_376)
    let first = MLXFast.rmsNorm(
        shaped[0..., 0..<1, 0...],
        weight: weight,
        eps: eps
    )
    let second = MLXFast.rmsNorm(
        shaped[0..., 1..<2, 0...],
        weight: weight,
        eps: eps
    )
    return concatenated([first, second], axis: 1).reshaped(2, 5_376)
}

/// Manual attention fallback matching the library's batched/ragged path.
private func gemma4FastAttentionFallback(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray {
    let (B, nQHeads, L, D) = (
        queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3)
    )
    let nKVHeads = keys.dim(1)
    let repeats = nQHeads / nKVHeads

    var q = queries * scale
    var k = keys
    var v = values
    if repeats > 1 {
        q = q.reshaped([B, nKVHeads, repeats, L, D])
        k = expandedDimensions(k, axis: 2)
        v = expandedDimensions(v, axis: 2)
    }

    var scores = matmul(q, k.swappedAxes(-1, -2))

    func applyMask(_ maskArray: MLXArray) {
        var mask = maskArray
        if scores.ndim == 5 && mask.ndim == 4 && mask.dim(0) == scores.dim(0) {
            mask = expandedDimensions(mask, axis: 2)
        }
        if mask.dtype == .bool {
            scores = MLX.where(
                mask, scores, MLXArray(-Float.infinity, dtype: scores.dtype))
        } else {
            scores = scores + mask
        }
    }

    switch mask {
    case .none:
        break
    case .causal:
        let qL = scores.dim(-2)
        let kL = scores.dim(-1)
        let qIndices = MLXArray(0..<qL) + MLXArray(kL - qL)
        let kIndices = MLXArray(0..<kL)
        let causalMask = greaterEqual(
            expandedDimensions(qIndices, axis: -1),
            expandedDimensions(kIndices, axis: -2))
        applyMask(causalMask)
    case .array(let maskArray):
        applyMask(maskArray)
    case .arrays(let maskArrays):
        if let maskArray = maskArrays.first {
            applyMask(maskArray)
        }
    }

    var probs = softmax(scores.asType(.float32), axis: -1, precise: true)
    probs = MLX.where(probs .!= probs, MLXArray(Float(0)), probs)
    scores = probs.asType(scores.dtype)
    var output = matmul(scores, v)
    if repeats > 1 {
        output = output.reshaped([B, nQHeads, L, values.dim(3)])
    }
    return output
}

/// High-performance Gemma 4 trunk built from the already-loaded library modules.
final class Gemma4FastEngine {
    let embedTokens: Embedding
    let embedScale: Float
    let finalNormWeight: MLXArray
    let eps: Float
    let softcap: Float
    let layers: [Gemma4FastLayer]
    let slidingWindow: Int
    let asyncLayerGroup: Int
    let asyncLayerLead: Int
    let tiedVocabularyHead: Gemma4TiedVocabularyHead?
    let usePacked13TiedVocabularyHead: Bool
    let verifyTiedVocabularyHead: Bool
    let supportsExactMTPPair: Bool
    private let logitSoftcap: @Sendable (MLXArray, MLXArray) -> MLXArray
    /// Persistent 0-d softcap array reused on every decode step when
    /// `DARKBLOOM_PRECOMPUTED_SCALAR_VIEWS` is on; nil restores the per-call
    /// `MLXArray(softcap)` allocation. Same value, same kernel input.
    private let precomputedSoftcap: MLXArray?
    /// Fused single-dispatch decode embedding (gather + dequantize + scale).
    /// nil when unsupported or `DARKBLOOM_FUSED_DECODE_EMBED=0`.
    private let fusedDecodeEmbed: Gemma4FusedDecodeEmbed?

    init(
        model: Gemma4RuntimeModel,
        indexedMetadata: [String: IndexedAffineMetadata] = [:],
        packedIndexMetadata: [String: Gemma4PackedQKVIndexMetadata] = [:],
        coTiledAttentionPayloads: [String: Gemma4CoTiledAttentionPayload] = [:],
        tiedHeadPacked13Metadata: Gemma4TiedHeadPacked13Metadata? = nil,
        tiedHeadCoTiledPayload: Gemma4TiedHeadCoTiledPayload? = nil
    ) throws {
        let config = model.configuration
        self.embedScale = Float(config.hiddenSize).squareRoot()
        self.eps = config.rmsNormEps
        self.softcap = config.finalLogitSoftcapping
        self.slidingWindow = config.slidingWindow
        let asyncLayerGroup = max(
            0,
            Int(ProcessInfo.processInfo.environment["MLXFAST_ASYNC_LAYER_GROUP"] ?? "10") ?? 10
        )
        self.asyncLayerGroup = asyncLayerGroup
        if asyncLayerGroup > 0 {
            self.asyncLayerLead = min(
                asyncLayerGroup,
                max(
                    1,
                    Int(ProcessInfo.processInfo.environment[
                        "MLXFAST_ASYNC_LAYER_LEAD"
                    ] ?? "1") ?? 1
                )
            )
        } else {
            self.asyncLayerLead = 0
        }

        let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        let params = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

        func module<T: Module>(_ key: String, as type: T.Type) throws -> T {
            guard let value = modules[key] as? T else {
                throw MLXFastError.invalidInput("missing module \(key) as \(T.self)")
            }
            return value
        }

        func array(_ key: String) throws -> MLXArray {
            guard let value = params[key] else {
                throw MLXFastError.invalidInput("missing parameter \(key)")
            }
            return value
        }

        let loadedEmbedTokens = try module("model.embed_tokens", as: Embedding.self)
        self.embedTokens = loadedEmbedTokens
        if gemma4PrecomputedScalarViewsEnabled() {
            let cap = MLXArray(softcap)
            eval(cap)
            self.precomputedSoftcap = cap
        } else {
            self.precomputedSoftcap = nil
        }
        self.fusedDecodeEmbed = Gemma4FusedDecodeEmbed(
            loadedEmbedTokens,
            embedScale: embedScale
        )
        let tiedHeadRequested = ["1", "true", "yes", "on"].contains(
            ProcessInfo.processInfo.environment["DARKBLOOM_TIED_HEAD_QMV"]?
                .lowercased() ?? "0"
        )
        let verifyTiedHead = ["1", "true", "yes", "on"].contains(
            ProcessInfo.processInfo.environment["DARKBLOOM_VERIFY_TIED_HEAD_BITS"]?
                .lowercased() ?? "0"
        )
        let packed13Rollback: Bool
        if let rawPacked13 = ProcessInfo.processInfo.environment[
            "DARKBLOOM_TIED_HEAD_PACKED13"
        ]?.lowercased() {
            switch rawPacked13 {
            case "0", "false", "no", "off":
                packed13Rollback = true
            case "1", "true", "yes", "on":
                packed13Rollback = false
            default:
                throw MLXFastError.invalidInput(
                    "DARKBLOOM_TIED_HEAD_PACKED13 must be 0 or 1"
                )
            }
        } else {
            packed13Rollback = false
        }
        let productionTiedHead = isGemma4ProductionTiedVocabularyHead(
            loadedEmbedTokens
        )
        self.usePacked13TiedVocabularyHead = productionTiedHead
            && !packed13Rollback
        self.verifyTiedVocabularyHead = verifyTiedHead
        if productionTiedHead || tiedHeadRequested || verifyTiedHead {
            guard let tiedVocabularyHead = Gemma4TiedVocabularyHead(
                loadedEmbedTokens,
                packed13Metadata: tiedHeadPacked13Metadata,
                coTiledPayload: tiedHeadCoTiledPayload
            ) else {
                throw MLXFastError.invalidInput(
                    "opt-in tied vocabulary head requires affine 4-bit "
                        + "QuantizedEmbedding [262144, 672] with BF16 "
                        + "scales and biases [262144, 84]"
                )
            }
            guard !usePacked13TiedVocabularyHead
                    || tiedVocabularyHead.packed13Metadata != nil
            else {
                throw MLXFastError.invalidInput(
                    "default tied vocabulary packed13 head requires validated "
                        + "transform-authored metadata"
                )
            }
            self.tiedVocabularyHead = usePacked13TiedVocabularyHead
                    || tiedHeadRequested
                    || verifyTiedHead
                ? tiedVocabularyHead
                : nil
        } else {
            self.tiedVocabularyHead = nil
        }
        let finalNorm = try module("model.norm", as: RMSNorm.self)
        self.finalNormWeight = finalNorm.weight

        var built: [Gemma4FastLayer] = []
        built.reserveCapacity(config.numHiddenLayers)
        for index in 0..<config.numHiddenLayers {
            let prefix = "model.layers.\(index)"
            let isSliding = config.layerTypes[index] == "sliding_attention"
            let headDim = isSliding ? config.headDim : config.globalHeadDim
            let useKEqV = config.attentionKeqV && !isSliding
            let nKvHeads: Int
            if useKEqV, let global = config.numGlobalKeyValueHeads {
                nKvHeads = global
            } else {
                nKvHeads = config.numKeyValueHeads
            }

            let rope: RoPELayer
            if isSliding {
                rope = initializeRope(
                    dims: headDim,
                    base: config.slidingRopeTheta,
                    traditional: false,
                    scalingConfig: nil,
                    maxPositionEmbeddings: nil
                )
            } else {
                rope = initializeRope(
                    dims: headDim,
                    base: config.fullRopeTheta,
                    traditional: false,
                    scalingConfig: [
                        "type": .string("proportional"),
                        "partial_rotary_factor": .float(config.fullPartialRotaryFactor),
                    ],
                    maxPositionEmbeddings: nil
                )
            }

            let vProj: QuantizedLinear?
            if useKEqV {
                vProj = nil
            } else {
                vProj = try module("\(prefix).self_attn.v_proj", as: QuantizedLinear.self)
            }

            let kNorm: RMSNorm?
            if useKEqV {
                // Still present for non-shared layers; k_eq_v still has k_norm.
                kNorm = try module("\(prefix).self_attn.k_norm", as: RMSNorm.self)
            } else {
                kNorm = try module("\(prefix).self_attn.k_norm", as: RMSNorm.self)
            }

            let nextInputNormWeight: MLXArray?
            if index + 1 < config.numHiddenLayers {
                let nextPrefix = "model.layers.\(index + 1)"
                nextInputNormWeight = try module(
                    "\(nextPrefix).input_layernorm",
                    as: RMSNorm.self
                ).weight
            } else {
                nextInputNormWeight = finalNorm.weight
            }

            built.append(
                Gemma4FastLayer(
                    isSliding: isSliding,
                    nHeads: config.numAttentionHeads,
                    nKvHeads: nKvHeads,
                    headDim: headDim,
                    useKEqV: useKEqV,
                    eps: config.rmsNormEps,
                    qProj: try module("\(prefix).self_attn.q_proj", as: QuantizedLinear.self),
                    kProj: try module("\(prefix).self_attn.k_proj", as: QuantizedLinear.self),
                    vProj: vProj,
                    oProj: try module("\(prefix).self_attn.o_proj", as: QuantizedLinear.self),
                    qNorm: try module("\(prefix).self_attn.q_norm", as: RMSNorm.self),
                    kNorm: kNorm,
                    inputNorm: try module("\(prefix).input_layernorm", as: RMSNorm.self),
                    postAttnNorm: try module("\(prefix).post_attention_layernorm", as: RMSNorm.self),
                    preFfnNorm: try module("\(prefix).pre_feedforward_layernorm", as: RMSNorm.self),
                    postFfnNorm: try module("\(prefix).post_feedforward_layernorm", as: RMSNorm.self),
                    gate: try module("\(prefix).mlp.gate_proj", as: QuantizedLinear.self),
                    up: try module("\(prefix).mlp.up_proj", as: QuantizedLinear.self),
                    down: try module("\(prefix).mlp.down_proj", as: QuantizedLinear.self),
                    layerScalar: try array("\(prefix).layer_scalar"),
                    nextInputNormWeight: nextInputNormWeight,
                    rope: rope,
                    qIndexedMetadata: indexedMetadata["\(prefix).self_attn.q_proj"],
                    kIndexedMetadata: indexedMetadata["\(prefix).self_attn.k_proj"],
                    vIndexedMetadata: indexedMetadata["\(prefix).self_attn.v_proj"],
                    gateIndexedMetadata: indexedMetadata["\(prefix).mlp.gate_proj"],
                    upIndexedMetadata: indexedMetadata["\(prefix).mlp.up_proj"],
                    downIndexedMetadata: indexedMetadata["\(prefix).mlp.down_proj"],
                    qPackedIndexMetadata: packedIndexMetadata["\(prefix).self_attn.q_proj"],
                    kPackedIndexMetadata: packedIndexMetadata["\(prefix).self_attn.k_proj"],
                    vPackedIndexMetadata: packedIndexMetadata["\(prefix).self_attn.v_proj"],
                    coTiledAttentionPayload:
                        coTiledAttentionPayloads["\(prefix).self_attn"]
                )
            )
        }
        self.layers = built
        self.supportsExactMTPPair = usePacked13TiedVocabularyHead
            && !verifyTiedVocabularyHead
            && tiedVocabularyHead?.supportsExactTwoVectorPacked13 == true
            && built.allSatisfy(\.supportsExactMTPPair)

        let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { logits, cap in
            tanh(logits / cap) * cap
        }
        let compileEnabled: Bool
        if let raw = ProcessInfo.processInfo.environment["MLX_COMPILED_DECODE"] {
            compileEnabled = ["1", "true", "yes", "on"].contains(raw.lowercased())
        } else {
            compileEnabled = true
        }
        self.logitSoftcap = compileEnabled ? compile(shapeless: true, body) : body
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden: MLXArray
        if inputs.ndim == 2, inputs.dim(0) == 1, inputs.dim(1) == 1,
           inputs.dtype == .int32, let fusedDecodeEmbed
        {
            // Single-token decode: one fused gather+dequantize+scale kernel
            // replaces the five-dispatch stock chain, bit-identical output.
            hidden = fusedDecodeEmbed(inputs)
        } else {
            hidden = embedTokens(inputs) * embedScale
        }

        func layerCache(_ index: Int) -> KVCache? {
            guard let cache, index < cache.count else { return nil }
            return cache[index]
        }

        let slidingIndex = layers.firstIndex(where: \.isSliding)
        let fullIndex = layers.firstIndex(where: { !$0.isSliding })
        let slidingMask = slidingIndex.map {
            createAttentionMask(
                h: hidden,
                cache: layerCache($0),
                windowSize: slidingWindow
            )
        }
        let fullMask = fullIndex.map {
            createAttentionMask(h: hidden, cache: layerCache($0), windowSize: nil)
        }

        var normalizedInput: MLXArray? = nil
        for (index, layer) in layers.enumerated() {
            let mask = layer.isSliding ? (slidingMask ?? .none) : (fullMask ?? .none)
            let result = layer(
                hidden,
                normalizedInput: normalizedInput,
                mask: mask,
                cache: layerCache(index),
                pruneTail: index == layers.count - 1
            )
            hidden = result.hidden
            normalizedInput = result.nextNormalized
            let layerNumber = index + 1
            if inputs.dim(1) == 1,
               asyncLayerGroup > 0,
               layerNumber >= asyncLayerLead,
               (layerNumber - asyncLayerLead).isMultiple(of: asyncLayerGroup)
            {
                if let normalizedInput {
                    asyncEval(hidden, normalizedInput)
                } else {
                    asyncEval(hidden)
                }
            }
        }

        if inputs.dim(1) == 1, let normalizedInput {
            hidden = normalizedInput
        } else {
            hidden = gemma4LastTokenHidden(hidden)
            hidden = MLXFast.rmsNorm(hidden, weight: finalNormWeight, eps: eps)
        }
        let cap = precomputedSoftcap ?? MLXArray(softcap)
        if let tiedVocabularyHead, usePacked13TiedVocabularyHead {
            let candidate = tiedVocabularyHead.packed13Softcapped(
                hidden,
                cap: cap
            )
            if verifyTiedVocabularyHead {
                let stock = logitSoftcap(embedTokens.asLinear(hidden), cap)
                tiedVocabularyHead.verifyRawFloat32(
                    candidate,
                    stock: stock,
                    candidateName: "packed13 fused-softcap",
                    stockName: "embedTokens.asLinear + compiled softcap"
                )
            }
            return candidate
        }

        let logits: MLXArray
        if let tiedVocabularyHead {
            let candidate = tiedVocabularyHead(hidden)
            if verifyTiedVocabularyHead {
                tiedVocabularyHead.verifyRawBF16(
                    candidate,
                    stock: embedTokens.asLinear(hidden),
                    candidateName: "stock-metadata custom",
                    stockName: "embedTokens.asLinear"
                )
            }
            logits = candidate
        } else {
            logits = embedTokens.asLinear(hidden)
        }
        return logitSoftcap(logits, cap)
    }

    func canRunExactMTPPair(cache: [KVCache]) -> Bool {
        guard supportsExactMTPPair, cache.count == layers.count else {
            return false
        }
        let combinedCaches = cache.compactMap {
            $0 as? Gemma4CombinedKVCache
        }
        guard combinedCaches.count == layers.count,
              combinedCaches.allSatisfy({ $0.canAppendExactPair() })
        else {
            return false
        }
        return layers.indices.allSatisfy {
            layers[$0].canRunExactMTPPair(
                offset: combinedCaches[$0].offset
            )
        }
    }

    /// MTP-only exact pair target forward. It shares packed weight reads while
    /// retaining K=1 arithmetic independently for each row.
    func exactMTPPair(
        _ inputs: MLXArray,
        cache: [KVCache]
    ) -> Gemma4MTPForward {
        precondition(canRunExactMTPPair(cache: cache))
        precondition(inputs.dtype == .int32 && inputs.shape == [1, 2])
        let combinedCaches = cache.compactMap {
            $0 as? Gemma4CombinedKVCache
        }

        // Preserve the serial gather/scaling boundary. Weight-sharing begins
        // only at the dense projections where the exact kernels retain each
        // row's K=1 reduction order.
        let first = embedTokens(inputs[0..., 0..<1]) * embedScale
        let second = embedTokens(inputs[0..., 1..<2]) * embedScale
        var hidden = concatenated([first, second], axis: 1)
            .reshaped(2, 5_376)
        var normalizedInput: MLXArray?
        var capturedFull: (MLXArray, MLXArray)?
        var capturedSliding: (MLXArray, MLXArray)?
        for (index, layer) in layers.enumerated() {
            let result = layer.exactMTPPair(
                hidden,
                normalizedInput: normalizedInput,
                cache: combinedCaches[index]
            )
            hidden = result.hidden
            normalizedInput = result.nextNormalized
            if layer.isSliding {
                capturedSliding = result.keyValue
            } else {
                capturedFull = result.keyValue
            }
            let layerNumber = index + 1
            if asyncLayerGroup > 0,
               layerNumber >= asyncLayerLead,
               (layerNumber - asyncLayerLead).isMultiple(of: asyncLayerGroup)
            {
                if let normalizedInput {
                    asyncEval(hidden, normalizedInput)
                } else {
                    asyncEval(hidden)
                }
            }
        }
        guard let normalizedInput,
              let tiedVocabularyHead,
              let capturedFull,
              let capturedSliding
        else {
            preconditionFailure("exact MTP pair final state is unavailable")
        }
        let logits = tiedVocabularyHead.exactTwoVectorPacked13Softcapped(
            normalizedInput,
            cap: precomputedSoftcap ?? MLXArray(softcap)
        )
        return Gemma4MTPForward(
            logits: logits.reshaped(1, 2, logits.dim(-1)),
            lastHidden: hidden.reshaped(1, 2, hidden.dim(-1)),
            capturedSharedKV: Gemma4SharedKV(
                fullAttention: capturedFull,
                slidingAttention: capturedSliding
            )
        )
    }

    /// Multi-position target verification for the trained Gemma 4 assistant.
    ///
    /// Unlike the ordinary runtime entry point, this preserves every verify
    /// position and captures the last full/sliding K/V views. Long prompt
    /// prefill still projects only its final position because the MTP session
    /// consumes only the seed logit; K<=4 verification projects every row.
    func forwardForMTP(
        _ inputs: MLXArray,
        cache: [KVCache]
    ) -> Gemma4MTPForward {
        precondition(inputs.dim(0) == 1)
        var hidden = embedTokens(inputs) * embedScale

        func layerCache(_ index: Int) -> KVCache? {
            index < cache.count ? cache[index] : nil
        }

        let slidingIndex = layers.firstIndex(where: \.isSliding)
        let fullIndex = layers.firstIndex(where: { !$0.isSliding })
        let slidingMask = slidingIndex.map {
            createAttentionMask(
                h: hidden,
                cache: layerCache($0),
                windowSize: slidingWindow
            )
        }
        let fullMask = fullIndex.map {
            createAttentionMask(
                h: hidden,
                cache: layerCache($0),
                windowSize: nil
            )
        }

        var normalizedInput: MLXArray? = nil
        var capturedFull: (MLXArray, MLXArray)?
        var capturedSliding: (MLXArray, MLXArray)?
        for (index, layer) in layers.enumerated() {
            let mask = layer.isSliding
                ? (slidingMask ?? .none)
                : (fullMask ?? .none)
            let result = layer(
                hidden,
                normalizedInput: normalizedInput,
                mask: mask,
                cache: layerCache(index)
            )
            hidden = result.hidden
            normalizedInput = result.nextNormalized
            if layer.isSliding {
                capturedSliding = result.keyValue
            } else {
                capturedFull = result.keyValue
            }
        }

        guard let capturedFull, let capturedSliding else {
            preconditionFailure(
                "Gemma 4 MTP verification requires full and sliding K/V"
            )
        }
        let preNorm = inputs.dim(1) > 16
            ? gemma4LastTokenHidden(hidden)
            : hidden
        let postNorm: MLXArray
        if inputs.dim(1) == 1, let normalizedInput {
            postNorm = normalizedInput
        } else {
            postNorm = MLXFast.rmsNorm(
                preNorm,
                weight: finalNormWeight,
                eps: eps
            )
        }
        let cap = precomputedSoftcap ?? MLXArray(softcap)
        let logits: MLXArray
        if let tiedVocabularyHead, usePacked13TiedVocabularyHead {
            let positionLogits = (0..<postNorm.dim(1)).map {
                positionIndex in
                tiedVocabularyHead.packed13Softcapped(
                    postNorm[
                        0...,
                        positionIndex..<(positionIndex + 1),
                        0...
                    ],
                    cap: cap
                )
            }
            logits = positionLogits.count == 1
                ? positionLogits[0]
                : concatenated(positionLogits, axis: 1)
        } else {
            logits = logitSoftcap(
                embedTokens.asLinear(postNorm),
                cap
            )
        }
        return Gemma4MTPForward(
            logits: logits,
            lastHidden: preNorm,
            capturedSharedKV: Gemma4SharedKV(
                fullAttention: capturedFull,
                slidingAttention: capturedSliding
            )
        )
    }
}
