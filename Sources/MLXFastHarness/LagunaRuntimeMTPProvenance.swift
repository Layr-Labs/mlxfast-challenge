import CryptoKit
import Darwin
import Foundation
import MLXFastCore

public struct ExperimentalMTPArtifactReport: Equatable, Sendable {
    public let trackID: String
    public let referenceBaselineStatus: String
    public let targetModelID: String
    public let targetRevision: String
    public let targetByteCount: Int
    public let targetTreeSHA256: String
    public let assistantModelID: String
    public let assistantRevision: String
    public let assistantByteCount: Int
    public let assistantTreeSHA256: String
}

public struct ExperimentalMTPSourceTargetReport: Equatable, Sendable {
    public let modelID: String
    public let revision: String
    public let fileCount: Int
    public let byteCount: Int
    public let manifestSHA256: String
}

struct ExperimentalMTPTrackContract: Decodable {
    struct Target: Decodable {
        let upstreamModelID: String
        let upstreamRevision: String
        let runtimeModelID: String
        let runtimeRevision: String
        let manifestPath: String
        let manifestSHA256: String
        let expectedSourceBytes: Int
        let maximumTransformedBytes: Int
        let modelType: String
        let hiddenSize: Int
        let intermediateSize: Int
        let numHiddenLayers: Int
        let vocabSize: Int
        let headDim: Int
        let slidingWindow: Int
        let numExperts: Int
        let numExpertsPerTok: Int
        let moeIntermediateSize: Int
        let sharedExpertIntermediateSize: Int
        let mlpOnlyLayers: [Int]
        let quantizationBits: Int
        let quantizationGroupSize: Int

        enum CodingKeys: String, CodingKey {
            case upstreamModelID = "upstream_model_id"
            case upstreamRevision = "upstream_revision"
            case runtimeModelID = "runtime_model_id"
            case runtimeRevision = "runtime_revision"
            case manifestPath = "manifest_path"
            case manifestSHA256 = "manifest_sha256"
            case expectedSourceBytes = "expected_source_bytes"
            case maximumTransformedBytes = "maximum_transformed_bytes"
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case vocabSize = "vocab_size"
            case headDim = "head_dim"
            case slidingWindow = "sliding_window"
            case numExperts = "num_experts"
            case numExpertsPerTok = "num_experts_per_tok"
            case moeIntermediateSize = "moe_intermediate_size"
            case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
            case mlpOnlyLayers = "mlp_only_layers"
            case quantizationBits = "quantization_bits"
            case quantizationGroupSize = "quantization_group_size"
        }
    }

    struct AssistantFile: Decodable, Equatable {
        let path: String
        let sha256: String
        /// Pinned byte count, or nil while the contract carries the
        /// operator placeholder string ("REGENERATE_ON_M5") instead of an
        /// on-box pin. Identity validation tolerates the placeholder;
        /// artifact validation fails closed on a nil byte pin.
        let bytes: Int?

        init(path: String, sha256: String, bytes: Int?) {
            self.path = path
            self.sha256 = sha256
            self.bytes = bytes
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            path = try container.decode(String.self, forKey: .path)
            sha256 = try container.decode(String.self, forKey: .sha256)
            if let pinnedBytes = try? container.decode(
                Int.self,
                forKey: .bytes
            ) {
                bytes = pinnedBytes
            } else {
                // The placeholder must still be a string; anything else is
                // a malformed contract and fails decoding outright.
                _ = try container.decode(String.self, forKey: .bytes)
                bytes = nil
            }
        }

        enum CodingKeys: String, CodingKey {
            case path
            case sha256
            case bytes
        }
    }

    struct Assistant: Decodable {
        let modelID: String
        let revision: String
        let manifestPath: String
        let manifestSHA256: String
        let expectedTotalBytes: Int
        let maximumTotalBytes: Int
        let modelType: String
        let backboneHiddenSize: Int
        let hiddenSize: Int
        let intermediateSize: Int
        let numHiddenLayers: Int
        let numAttentionHeads: Int
        let numKeyValueHeads: Int
        let headDim: Int
        let slidingWindow: Int
        let layerTypes: [String]
        let vocabSize: Int
        let draftVocabSize: Int
        let numTargetLayers: Int
        let blockSize: Int
        let maskTokenID: Int
        let targetLayerIDs: [Int]
        let eagleAuxHiddenStateLayerIDs: [Int]
        let causal: Bool
        let sourceDtype: String
        let quantizationBits: Int
        let quantizationGroupSize: Int
        let files: [AssistantFile]

        enum CodingKeys: String, CodingKey {
            case modelID = "model_id"
            case revision
            case manifestPath = "manifest_path"
            case manifestSHA256 = "manifest_sha256"
            case expectedTotalBytes = "expected_total_bytes"
            case maximumTotalBytes = "maximum_total_bytes"
            case modelType = "model_type"
            case backboneHiddenSize = "backbone_hidden_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case slidingWindow = "sliding_window"
            case layerTypes = "layer_types"
            case vocabSize = "vocab_size"
            case draftVocabSize = "draft_vocab_size"
            case numTargetLayers = "num_target_layers"
            case blockSize = "block_size"
            case maskTokenID = "mask_token_id"
            case targetLayerIDs = "target_layer_ids"
            case eagleAuxHiddenStateLayerIDs =
                "eagle_aux_hidden_state_layer_ids"
            case causal
            case sourceDtype = "source_dtype"
            case quantizationBits = "quantization_bits"
            case quantizationGroupSize = "quantization_group_size"
            case files
        }
    }

    struct ProtocolContract: Decodable {
        let name: String
        let maximumBlockSize: Int
        let decodeTokens: Int
        let maximumDecodeTokens: Int
        let parentOracleRequired: Bool
        let workerTimingIsAuthoritative: Bool

        enum CodingKeys: String, CodingKey {
            case name
            case maximumBlockSize = "maximum_block_size"
            case decodeTokens = "decode_tokens"
            case maximumDecodeTokens = "maximum_decode_tokens"
            case parentOracleRequired = "parent_oracle_required"
            case workerTimingIsAuthoritative = "worker_timing_is_authoritative"
        }
    }

    struct ReferenceBaseline: Decodable {
        let status: String
        let publicationAllowed: Bool
        let requiredRebaselineHardware: String

        enum CodingKeys: String, CodingKey {
            case status
            case publicationAllowed = "publication_allowed"
            case requiredRebaselineHardware = "required_rebaseline_hardware"
        }
    }

    let schemaVersion: Int
    let trackID: String
    let officialScoringEnabled: Bool
    let mlxSwiftLMRevision: String
    let target: Target
    let assistant: Assistant
    let protocolContract: ProtocolContract
    let referenceBaseline: ReferenceBaseline

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case trackID = "track_id"
        case officialScoringEnabled = "official_scoring_enabled"
        case mlxSwiftLMRevision = "mlx_swift_lm_revision"
        case target
        case assistant
        case protocolContract = "protocol"
        case referenceBaseline = "reference_baseline"
    }
}

/// The pinned DFlash speculator ships a flat Laguna-family config.json
/// (model_type "laguna", architectures ["DFlashLagunaForCausalLM"]) rather
/// than a nested assistant wrapper: backbone geometry, the drafter's own
/// attention geometry, and the DFlash pairing metadata (block size, mask
/// token, target layer taps) all live at the top level or under
/// `dflash_config`.
private struct ExperimentalMTPAssistantConfig: Decodable {
    struct Quantization: Decodable {
        let groupSize: Int
        let bits: Int
        let mode: String?

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
            case mode
        }
    }

    struct DFlashConfig: Decodable {
        let blockSize: Int
        let maskTokenID: Int
        let numTargetLayers: Int
        let targetLayerIDs: [Int]
        let causal: Bool

        enum CodingKeys: String, CodingKey {
            case blockSize = "block_size"
            case maskTokenID = "mask_token_id"
            case numTargetLayers = "num_target_layers"
            case targetLayerIDs = "target_layer_ids"
            case causal
        }
    }

    let modelType: String
    let architectures: [String]?
    let hiddenSize: Int
    let intermediateSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let rmsNormEps: Double
    let ropeTheta: Double
    let slidingWindow: Int
    let maxPositionEmbeddings: Int
    let attentionBias: Bool?
    let gating: ExperimentalMTPAssistantGating?
    let layerTypes: [String]
    let vocabSize: Int
    let draftVocabSize: Int
    let eagleAuxHiddenStateLayerIDs: [Int]
    let dflashConfig: DFlashConfig
    let numExperts: Int?
    let torchDtype: String?
    let quantization: Quantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case gating
        case layerTypes = "layer_types"
        case vocabSize = "vocab_size"
        case draftVocabSize = "draft_vocab_size"
        case eagleAuxHiddenStateLayerIDs =
            "eagle_aux_hidden_state_layer_ids"
        case dflashConfig = "dflash_config"
        case numExperts = "num_experts"
        case torchDtype = "torch_dtype"
        case quantization
    }
}

/// Attention output gating flag: the Laguna family encodes per-head gating
/// as either a bool (`true`) or a string (`"per-head"` / `"per_head"`; any
/// other non-empty, non-disabling string also selects the default per-head
/// mode). Mirrors the runtime worker's pinned-config gating decoder.
private struct ExperimentalMTPAssistantGating: Decodable {
    let isPerHead: Bool

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let flag = try? container.decode(Bool.self) {
            isPerHead = flag
            return
        }
        switch try container.decode(String.self) {
        case "per-element", "per_element", "false", "none", "":
            isPerHead = false
        default:
            isPerHead = true
        }
    }
}

extension LagunaRuntime {
    static let experimentalMTPTrackID = "laguna-xs-2.1-mtp-v1"
    static let experimentalMTPDependencyRevision =
        "bc1c0ee67d15798343be17c9f8f61f7c0d977149"
    static let experimentalMTPTargetModelID =
        "mlx-community/Laguna-XS-2.1-4bit"
    static let experimentalMTPTargetRevision =
        "c42e0a8f8d504ceacde015a535dcb286d65c8799"
    static let experimentalMTPUpstreamTargetModelID =
        "poolside/Laguna-XS-2.1"
    static let experimentalMTPUpstreamTargetRevision =
        "c405648833500615a2efde76886b8aed4fb9324e"
    static let experimentalMTPAssistantModelID =
        "poolside/Laguna-XS-2.1-DFlash"
    static let experimentalMTPAssistantRevision =
        "5c36361aab23c8ed3afbd079c10c426b677bc607"

    static func validateExperimentalMTPSourceTarget(
        sourceTargetPath: String,
        contractPath: String
    ) throws -> ExperimentalMTPSourceTargetReport {
        let contractURL = URL(
            fileURLWithPath: contractPath
        ).standardizedFileURL
        try requireRegularFile(
            contractURL.path,
            description: "experimental MTP track contract"
        )
        let contract = try JSONDecoder().decode(
            ExperimentalMTPTrackContract.self,
            from: Data(contentsOf: contractURL)
        )
        try validateExperimentalMTPContract(contract)

        let repositoryRoot = contractURL
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let manifestURL = repositoryRoot.appendingPathComponent(
            contract.target.manifestPath
        ).standardizedFileURL
        guard manifestURL.path.hasPrefix(repositoryRoot.path + "/") else {
            throw MLXFastError.invalidInput(
                "experimental MTP target manifest escaped the contract root"
            )
        }
        try requireRegularFile(
            manifestURL.path,
            description: "experimental MTP target manifest"
        )
        let manifestDigest = try fileDigest(manifestURL)
        let manifestHex = manifestDigest.map {
            String(format: "%02x", $0)
        }.joined()
        guard manifestHex == contract.target.manifestSHA256 else {
            throw MLXFastError.invalidInput(
                "experimental MTP target manifest SHA256 mismatch"
            )
        }
        let entries = try parseExperimentalMTPManifest(
            Data(contentsOf: manifestURL)
        )
        try requireDirectory(
            sourceTargetPath,
            description: "experimental MTP source target"
        )
        let root = URL(
            fileURLWithPath: sourceTargetPath
        ).standardizedFileURL
        let actualNames = Set(
            try FileManager.default.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: nil,
                options: []
            ).map(\.lastPathComponent)
        )
        let expectedNames = Set(entries.map(\.path))
        guard actualNames == expectedNames else {
            throw MLXFastError.invalidInput(
                "experimental MTP source target inventory mismatch "
                    + "missing=\(expectedNames.subtracting(actualNames).sorted()) "
                    + "unexpected=\(actualNames.subtracting(expectedNames).sorted())"
            )
        }

        var byteCount = 0
        for entry in entries {
            let url = root.appendingPathComponent(entry.path)
            try requireRegularFile(
                url.path,
                description: "experimental MTP source target \(entry.path)"
            )
            let actualBytes = try fileSizeByteCount(
                from: FileManager.default.attributesOfItem(atPath: url.path),
                path: url.path
            )
            guard actualBytes == entry.bytes,
                  byteCount <= Int.max - actualBytes
            else {
                throw MLXFastError.invalidInput(
                    "experimental MTP source target \(entry.path) size mismatch"
                )
            }
            byteCount += actualBytes
            let actualHash = try fileDigest(url).map {
                String(format: "%02x", $0)
            }.joined()
            guard actualHash == entry.sha256 else {
                throw MLXFastError.invalidInput(
                    "experimental MTP source target \(entry.path) SHA256 mismatch"
                )
            }
        }
        guard byteCount == contract.target.expectedSourceBytes else {
            throw MLXFastError.invalidInput(
                "experimental MTP source target byte total mismatch"
            )
        }
        return ExperimentalMTPSourceTargetReport(
            modelID: contract.target.runtimeModelID,
            revision: contract.target.runtimeRevision,
            fileCount: entries.count,
            byteCount: byteCount,
            manifestSHA256: manifestHex
        )
    }

    static func validateExperimentalMTPArtifacts(
        targetWeightsPath: String,
        assistantPath: String,
        contractPath: String
    ) throws -> ExperimentalMTPArtifactReport {
        try requireRegularFile(
            contractPath,
            description: "experimental MTP track contract"
        )
        let contractURL = URL(fileURLWithPath: contractPath).standardizedFileURL
        let contractSize = try fileSizeByteCount(
            from: FileManager.default.attributesOfItem(atPath: contractURL.path),
            path: contractURL.path
        )
        guard contractSize > 0, contractSize <= 64 * 1024 else {
            throw MLXFastError.invalidInput(
                "experimental MTP track contract must be at most 65536 bytes"
            )
        }
        let contract = try JSONDecoder().decode(
            ExperimentalMTPTrackContract.self,
            from: Data(contentsOf: contractURL)
        )
        try validateExperimentalMTPContract(contract)

        try requireDirectory(
            targetWeightsPath,
            description: "experimental MTP transformed target"
        )
        let targetDigest = try directoryDigest(
            rootPath: targetWeightsPath,
            ignoredRelativePaths: [".benchmark-source.sha256", ".gitkeep"]
        )
        guard targetDigest.byteCount > 0,
              targetDigest.byteCount <= contract.target.maximumTransformedBytes
        else {
            throw MLXFastError.invalidInput(
                "experimental MTP transformed target is outside its artifact byte cap"
            )
        }
        try validateRuntimeWorkerPinnedConfiguration(
            weightsPath: targetWeightsPath
        )

        let assistantDigest = try validateExperimentalMTPAssistantDirectory(
            assistantPath,
            contract: contract
        )
        return ExperimentalMTPArtifactReport(
            trackID: contract.trackID,
            referenceBaselineStatus: contract.referenceBaseline.status,
            targetModelID: contract.target.runtimeModelID,
            targetRevision: contract.target.runtimeRevision,
            targetByteCount: targetDigest.byteCount,
            targetTreeSHA256: targetDigest.sha256,
            assistantModelID: contract.assistant.modelID,
            assistantRevision: contract.assistant.revision,
            assistantByteCount: assistantDigest.byteCount,
            assistantTreeSHA256: assistantDigest.sha256
        )
    }

    static func validateExperimentalMTPContract(
        _ contract: ExperimentalMTPTrackContract
    ) throws {
        // TODO(operator): the DFlash file pins below carry the
        // REGENERATE_ON_M5 placeholders (nil byte counts) until the
        // assistant manifest is regenerated from an on-box download on
        // m5-bench (go-live runbook step A). Identity validation accepts
        // the placeholder so provenance-only callers keep working;
        // artifact validation fails closed on the nil byte pins.
        let expectedFiles = [
            ExperimentalMTPTrackContract.AssistantFile(
                path: "config.json",
                sha256: "REGENERATE_ON_M5",
                bytes: nil
            ),
            ExperimentalMTPTrackContract.AssistantFile(
                path: "model.safetensors",
                sha256: "REGENERATE_ON_M5",
                bytes: nil
            ),
        ]
        let expectedAssistantLayerTypes = [String](
            repeating: "sliding_attention",
            count: 5
        )
        guard contract.schemaVersion == 1,
              contract.trackID == experimentalMTPTrackID,
              // Laguna re-pin: official scoring is DISABLED, fail closed,
              // until the hidden goldens, the weight manifests, and the
              // paired M5 reference baseline are regenerated on m5-bench
              // with the real Laguna weights
              // (docs/mtp-track-golive-runbook.md). TODO(operator): the
              // enablement commit flips the contract fixture and this pin
              // together. A contract claiming anything else is not the
              // pinned track identity and fails closed.
              !contract.officialScoringEnabled,
              contract.mlxSwiftLMRevision == experimentalMTPDependencyRevision,
              contract.target.upstreamModelID
                  == experimentalMTPUpstreamTargetModelID,
              contract.target.upstreamRevision
                  == experimentalMTPUpstreamTargetRevision,
              contract.target.runtimeModelID == experimentalMTPTargetModelID,
              contract.target.runtimeRevision == experimentalMTPTargetRevision,
              contract.target.manifestPath
                  == "fixtures/mtp_laguna_xs_2_1_4bit.sha256",
              // TODO(operator): pin the real manifest-file SHA-256 once the
              // target manifest is regenerated and byte-verified on
              // m5-bench; until then the placeholder makes every manifest
              // digest comparison fail closed.
              contract.target.manifestSHA256 == "REGENERATE_ON_M5",
              contract.target.expectedSourceBytes == 18_829_720_326,
              contract.target.maximumTransformedBytes == 20 * (1 << 30),
              contract.target.modelType == "laguna",
              contract.target.hiddenSize == MLXFastConstants.hiddenSize,
              contract.target.intermediateSize
                  == MLXFastConstants.intermediateSize,
              contract.target.numHiddenLayers
                  == MLXFastConstants.numHiddenLayers,
              contract.target.vocabSize == MLXFastConstants.vocabSize,
              contract.target.headDim == 128,
              contract.target.slidingWindow == 512,
              contract.target.numExperts == 256,
              contract.target.numExpertsPerTok == 8,
              contract.target.moeIntermediateSize == 512,
              contract.target.sharedExpertIntermediateSize == 512,
              contract.target.mlpOnlyLayers == [0],
              contract.target.quantizationBits == 4,
              contract.target.quantizationGroupSize == 64,
              contract.assistant.modelID == experimentalMTPAssistantModelID,
              contract.assistant.revision == experimentalMTPAssistantRevision,
              contract.assistant.manifestPath
                  == "fixtures/mtp_laguna_xs_2_1_dflash.sha256",
              // TODO(operator): pin the real manifest-file SHA-256 once the
              // DFlash manifest is regenerated on m5-bench.
              contract.assistant.manifestSHA256 == "REGENERATE_ON_M5",
              // The upstream BF16 model.safetensors payload at the pinned
              // revision (Hugging Face LFS metadata); setup converts it to
              // MLX affine 4-bit group-64 on-box.
              contract.assistant.expectedTotalBytes == 924_135_848,
              contract.assistant.maximumTotalBytes == 1_073_741_824,
              contract.assistant.modelType == "laguna_dflash",
              // The DFlash drafter consumes target hidden states, so its
              // widths are the backbone's, not a narrowed drafter tower.
              contract.assistant.backboneHiddenSize
                  == MLXFastConstants.hiddenSize,
              contract.assistant.hiddenSize == MLXFastConstants.hiddenSize,
              contract.assistant.intermediateSize == 8_192,
              contract.assistant.numHiddenLayers == 5,
              contract.assistant.numAttentionHeads == 64,
              contract.assistant.numKeyValueHeads == 8,
              contract.assistant.headDim == 128,
              contract.assistant.slidingWindow == 512,
              contract.assistant.layerTypes == expectedAssistantLayerTypes,
              contract.assistant.vocabSize == MLXFastConstants.vocabSize,
              contract.assistant.draftVocabSize == MLXFastConstants.vocabSize,
              contract.assistant.numTargetLayers
                  == MLXFastConstants.numHiddenLayers,
              contract.assistant.blockSize == 16,
              contract.assistant.maskTokenID == 12,
              contract.assistant.targetLayerIDs == [1, 13, 25, 33, 39],
              contract.assistant.eagleAuxHiddenStateLayerIDs
                  == [2, 14, 26, 34, 40],
              contract.assistant.causal,
              contract.assistant.sourceDtype == "bfloat16",
              contract.assistant.quantizationBits == 4,
              contract.assistant.quantizationGroupSize == 64,
              contract.assistant.files == expectedFiles,
              contract.protocolContract.name == "trusted_mtp_block_v1",
              contract.protocolContract.maximumBlockSize
                  == MLXFastConstants.experimentalMTPMaxBlockSize,
              contract.protocolContract.decodeTokens
                  == MLXFastConstants.experimentalMTPMaxTotalTokens,
              contract.protocolContract.maximumDecodeTokens
                  == MLXFastConstants.experimentalMTPMaxConfiguredTotalTokens,
              contract.protocolContract.parentOracleRequired,
              !contract.protocolContract.workerTimingIsAuthoritative,
              // TODO(operator): flips to "established" with
              // publication_allowed=true in the enablement commit after the
              // paired Laguna baseline is re-established on m5-bench
              // (runbook steps B-C).
              contract.referenceBaseline.status == "pending_m5_rebaseline",
              !contract.referenceBaseline.publicationAllowed,
              contract.referenceBaseline.requiredRebaselineHardware
                  == "m5-bench"
        else {
            throw MLXFastError.invalidInput(
                "experimental MTP track contract does not match the organizer-pinned identity"
            )
        }
    }

    private static func validateExperimentalMTPAssistantDirectory(
        _ path: String,
        contract: ExperimentalMTPTrackContract
    ) throws -> DirectoryDigest {
        try requireDirectory(
            path,
            description: "experimental MTP assistant"
        )
        let root = URL(fileURLWithPath: path).standardizedFileURL
        let expectedNames = Set(contract.assistant.files.map(\.path))
        let entries = try FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [
                .isRegularFileKey,
                .isDirectoryKey,
                .isSymbolicLinkKey,
            ],
            options: []
        )
        let actualNames = Set(entries.map(\.lastPathComponent))
        guard actualNames == expectedNames else {
            let unexpected = actualNames.subtracting(expectedNames).sorted()
            let missing = expectedNames.subtracting(actualNames).sorted()
            throw MLXFastError.invalidInput(
                "experimental MTP assistant inventory mismatch "
                    + "missing=\(missing) unexpected=\(unexpected)"
            )
        }

        var byteCount = 0
        var treeHasher = SHA256()
        for expected in contract.assistant.files {
            guard expected.path == URL(fileURLWithPath: expected.path).lastPathComponent,
                  !expected.path.isEmpty
            else {
                throw MLXFastError.invalidInput(
                    "experimental MTP assistant contract contains an unsafe file path"
                )
            }
            guard let expectedBytes = expected.bytes else {
                throw MLXFastError.invalidInput(
                    "experimental MTP assistant \(expected.path) byte pin is an "
                        + "operator placeholder; regenerate the manifest on m5-bench"
                )
            }
            let url = root.appendingPathComponent(expected.path)
            try requireRegularFile(
                url.path,
                description: "experimental MTP assistant \(expected.path)"
            )
            let actualBytes = try fileSizeByteCount(
                from: FileManager.default.attributesOfItem(atPath: url.path),
                path: url.path
            )
            guard actualBytes == expectedBytes,
                  byteCount <= Int.max - actualBytes
            else {
                throw MLXFastError.invalidInput(
                    "experimental MTP assistant \(expected.path) size mismatch"
                )
            }
            byteCount += actualBytes
            let digest = try fileDigest(url)
            let hex = digest.map { String(format: "%02x", $0) }.joined()
            guard hex == expected.sha256 else {
                throw MLXFastError.invalidInput(
                    "experimental MTP assistant \(expected.path) SHA256 mismatch"
                )
            }
            treeHasher.update(data: Data(expected.path.utf8))
            treeHasher.update(data: Data([0]))
            treeHasher.update(data: Data(digest))
            treeHasher.update(data: Data([0]))
        }
        guard byteCount == contract.assistant.expectedTotalBytes,
              byteCount <= contract.assistant.maximumTotalBytes
        else {
            throw MLXFastError.invalidInput(
                "experimental MTP assistant exceeds its organizer artifact budget"
            )
        }

        try validateExperimentalMTPAssistantConfig(
            at: root.appendingPathComponent("config.json"),
            contract: contract
        )
        return DirectoryDigest(
            fileCount: contract.assistant.files.count,
            byteCount: byteCount,
            sha256: treeHasher.finalize().map {
                String(format: "%02x", $0)
            }.joined()
        )
    }

    private static func validateExperimentalMTPAssistantConfig(
        at url: URL,
        contract: ExperimentalMTPTrackContract
    ) throws {
        let config = try JSONDecoder().decode(
            ExperimentalMTPAssistantConfig.self,
            from: Data(contentsOf: url)
        )
        // The checkpoint config declares the Laguna model family; the
        // contract's "laguna_dflash" model_type is the track's own artifact
        // label. Optional fields follow the pinned BF16 upstream: absent
        // means the pinned default, present must equal the pinned value.
        // Quantization metadata is absent in the BF16 upstream and must
        // match the contract if the on-disk checkpoint is the converted MLX
        // 4-bit form.
        guard config.modelType == "laguna",
              config.architectures == nil
                  || config.architectures == ["DFlashLagunaForCausalLM"],
              config.hiddenSize == contract.assistant.hiddenSize,
              config.hiddenSize == contract.assistant.backboneHiddenSize,
              config.intermediateSize == contract.assistant.intermediateSize,
              config.numHiddenLayers == contract.assistant.numHiddenLayers,
              config.numAttentionHeads
                  == contract.assistant.numAttentionHeads,
              config.numKeyValueHeads
                  == contract.assistant.numKeyValueHeads,
              config.headDim == contract.assistant.headDim,
              config.rmsNormEps == 1e-6,
              config.ropeTheta == 500_000,
              config.slidingWindow == contract.assistant.slidingWindow,
              config.maxPositionEmbeddings == 262_144,
              config.attentionBias == nil || config.attentionBias == false,
              config.gating?.isPerHead ?? true,
              config.layerTypes == contract.assistant.layerTypes,
              config.vocabSize == contract.assistant.vocabSize,
              config.draftVocabSize == contract.assistant.draftVocabSize,
              config.eagleAuxHiddenStateLayerIDs
                  == contract.assistant.eagleAuxHiddenStateLayerIDs,
              config.dflashConfig.blockSize == contract.assistant.blockSize,
              config.dflashConfig.maskTokenID
                  == contract.assistant.maskTokenID,
              config.dflashConfig.numTargetLayers
                  == contract.assistant.numTargetLayers,
              config.dflashConfig.targetLayerIDs
                  == contract.assistant.targetLayerIDs,
              config.dflashConfig.causal == contract.assistant.causal,
              config.numExperts == nil || config.numExperts == 0,
              config.torchDtype == nil
                  || config.torchDtype == contract.assistant.sourceDtype,
              config.quantization == nil
                  || (config.quantization?.bits
                      == contract.assistant.quantizationBits
                      && config.quantization?.groupSize
                          == contract.assistant.quantizationGroupSize
                      && (config.quantization?.mode ?? "affine") == "affine")
        else {
            throw MLXFastError.invalidInput(
                "experimental MTP assistant config is incompatible with the pinned target"
            )
        }
    }

    private struct ExperimentalMTPManifestEntry {
        let path: String
        let sha256: String
        let bytes: Int
    }

    private static func parseExperimentalMTPManifest(
        _ data: Data
    ) throws -> [ExperimentalMTPManifestEntry] {
        guard let text = String(data: data, encoding: .utf8) else {
            throw MLXFastError.invalidInput(
                "experimental MTP target manifest is not UTF-8"
            )
        }
        var entries: [ExperimentalMTPManifestEntry] = []
        var names = Set<String>()
        for rawLine in text.split(
            separator: "\n",
            omittingEmptySubsequences: false
        ) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") {
                continue
            }
            let fields = line.split(whereSeparator: \.isWhitespace)
            guard fields.count == 3,
                  fields[0].count == 64,
                  fields[0].allSatisfy({
                      ("0"..."9").contains($0)
                          || ("a"..."f").contains($0)
                  }),
                  let bytes = Int(fields[1]),
                  bytes > 0
            else {
                throw MLXFastError.invalidInput(
                    "experimental MTP target manifest has a malformed line"
                )
            }
            let path = String(fields[2])
            guard !path.isEmpty,
                  path == URL(fileURLWithPath: path).lastPathComponent,
                  names.insert(path).inserted
            else {
                throw MLXFastError.invalidInput(
                    "experimental MTP target manifest has an unsafe or duplicate path"
                )
            }
            entries.append(
                ExperimentalMTPManifestEntry(
                    path: path,
                    sha256: String(fields[0]),
                    bytes: bytes
                )
            )
        }
        guard !entries.isEmpty else {
            throw MLXFastError.invalidInput(
                "experimental MTP target manifest contains no files"
            )
        }
        return entries
    }
}
