import CoreFoundation
import Foundation
import MLXFastCore

/// Per-tensor quantization expectations parsed from the Laguna source
/// config's `quantization` (or `quantization_config`) block. Mirrors the
/// runtime's `LagunaQuantizationSpec`
/// (`Sources/MLXFastModel/LagunaConfig.swift`) without importing the
/// MLX-linked model target: the block carries the global scalars plus
/// dictionary-valued per-tensor overrides keyed by the source tensor stem
/// without the trailing `.weight` -- in the pinned mlx-community checkpoint,
/// the 39 sparse-layer router gates
/// (`language_model.model.layers.<N>.mlp.gate.proj`) at 8 bits.
struct LagunaTransformQuantizationSpec: Equatable {
    struct Override: Equatable {
        let groupSize: Int
        let bits: Int
    }

    let groupSize: Int
    let bits: Int
    let mode: String
    let overrides: [String: Override]

    /// Expected quantization for a quantized tensor, resolved from the
    /// per-tensor overrides with fallback to the global spec. `stem` is the
    /// source tensor name without the trailing `.weight`.
    func expected(forTensorStem stem: String) -> (groupSize: Int, bits: Int) {
        if let override = overrides[stem] {
            return (override.groupSize, override.bits)
        }
        return (groupSize, bits)
    }
}

/// Transform-side structural validation of the Laguna text-tower tensor set
/// against `docs/laguna-weight-contract.md`. The source
/// checkpoint (`mlx-community/Laguna-XS-2.1-4bit`) is already MLX
/// affine-quantized, so the transform passes tensors through unchanged; this
/// pass fails fast -- before the multi-GB copy -- when the set it would copy
/// cannot satisfy the runtime loader (`LagunaWeightLoader`):
///
/// - every recognized quantized projection stored as packed U32 codes must
///   ship its BF16 `.scales`/`.biases` companions with matching leading
///   dimensions;
/// - the SwitchGLU expert tensors (`mlp.switch_mlp.{gate,up,down}_proj`)
///   must keep the stacked 3D layout (leading experts axis; never split per
///   expert), while every other projection -- attention q/k/v/o and the
///   per-head `g_proj` gate, the layer-0 dense MLP, the shared expert, the
///   router, the embedding, and the untied `lm_head` -- is 2D;
/// - each tensor's packed width must match the group size and bit width the
///   emitted config.json declares for its stem (global 4-bit group-64 with
///   the 8-bit router-gate overrides);
/// - a quantized router pairs with a BF16 `mlp.gate.e_score_correction_bias`
///   vector carrying one entry per expert, and the stacked expert tensors
///   agree with the router on the expert count.
///
/// Unquantized tensors (norms, tiny synthetic test fixtures, BF16 exports)
/// are passed through without affine checks; the runtime validates the full
/// exact geometry against `LagunaConfig` before the first forward.
enum LagunaCheckpointValidation {
    struct RecognizedQuantizedStem: Equatable {
        enum Kind: Equatable {
            /// Ordinary 2D projection: `[rows, in]`.
            case matrix
            /// Sparse-layer MoE router (`mlp.gate.proj`): 2D
            /// `[experts, in]`, paired with a raw
            /// `mlp.gate.e_score_correction_bias` vector.
            case routerGate
            /// SwitchGLU stacked expert projection: 3D
            /// `[experts, rows, in]`.
            case stackedExperts
        }

        let kind: Kind
        let layerIndex: Int?
    }

    private static let layerPrefix = "language_model.model.layers."

    /// Recognizes the quantized-projection stems of the Laguna text tower.
    /// Norm weights (`*_layernorm`, `q_norm`, `k_norm`, `model.norm`) and
    /// the router correction bias are raw tensors and are not returned here.
    static func recognizedQuantizedStem(forWeightName name: String) -> RecognizedQuantizedStem? {
        guard name.hasSuffix(".weight") else {
            return nil
        }
        if name == "language_model.model.embed_tokens.weight"
            || name == "language_model.lm_head.weight"
        {
            return RecognizedQuantizedStem(kind: .matrix, layerIndex: nil)
        }
        guard name.hasPrefix(layerPrefix) else {
            return nil
        }
        let remainder = name.dropFirst(layerPrefix.count)
        guard let separator = remainder.firstIndex(of: "."),
              let layerIndex = Int(remainder[..<separator])
        else {
            return nil
        }
        switch String(remainder[separator...]) {
        case ".self_attn.q_proj.weight",
             ".self_attn.k_proj.weight",
             ".self_attn.v_proj.weight",
             ".self_attn.o_proj.weight",
             ".self_attn.g_proj.weight",
             ".mlp.gate_proj.weight",
             ".mlp.up_proj.weight",
             ".mlp.down_proj.weight",
             ".mlp.shared_expert.gate_proj.weight",
             ".mlp.shared_expert.up_proj.weight",
             ".mlp.shared_expert.down_proj.weight":
            return RecognizedQuantizedStem(kind: .matrix, layerIndex: layerIndex)
        case ".mlp.gate.proj.weight":
            return RecognizedQuantizedStem(kind: .routerGate, layerIndex: layerIndex)
        case ".mlp.switch_mlp.gate_proj.weight",
             ".mlp.switch_mlp.up_proj.weight",
             ".mlp.switch_mlp.down_proj.weight":
            return RecognizedQuantizedStem(kind: .stackedExperts, layerIndex: layerIndex)
        default:
            return nil
        }
    }

    /// Parses the quantization spec the emitted runtime config.json will
    /// carry (`quantization` with fallback to the `quantization_config`
    /// mirror, matching both `makeRuntimeConfigData` and
    /// `LagunaConfig.load`). Defaults mirror the pinned checkpoint's global
    /// affine 4-bit group-64 spec.
    static func quantizationSpec(
        fromConfigRoot root: [String: Any]
    ) throws -> LagunaTransformQuantizationSpec {
        let block =
            (root["quantization"] as? [String: Any])
            ?? (root["quantization_config"] as? [String: Any]) ?? [:]
        let groupSize = try intField("group_size", in: block, defaultValue: 64)
        let bits = try intField("bits", in: block, defaultValue: 4)
        let mode = try stringField("mode", in: block, defaultValue: "affine")
        guard mode == "affine" else {
            throw MLXFastError.invalidInput(
                "Laguna quantization mode must be affine, found \(mode)"
            )
        }
        try validateGroupAndBits(
            groupSize: groupSize,
            bits: bits,
            context: "global quantization spec"
        )

        var overrides: [String: LagunaTransformQuantizationSpec.Override] = [:]
        for (key, value) in block {
            guard let overrideObject = value as? [String: Any] else {
                continue
            }
            let override = LagunaTransformQuantizationSpec.Override(
                groupSize: try intField("group_size", in: overrideObject, defaultValue: groupSize),
                bits: try intField("bits", in: overrideObject, defaultValue: bits)
            )
            try validateGroupAndBits(
                groupSize: override.groupSize,
                bits: override.bits,
                context: "quantization override for \(key)"
            )
            overrides[key] = override
        }
        return LagunaTransformQuantizationSpec(
            groupSize: groupSize,
            bits: bits,
            mode: mode,
            overrides: overrides
        )
    }

    static func validateSelectedTensors(
        selectedKeys: Set<String>,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader],
        quantization: LagunaTransformQuantizationSpec
    ) throws {
        var routerExpertsByLayer: [Int: Int] = [:]
        var stackedExpertsByLayer: [Int: Int] = [:]

        for name in selectedKeys.sorted() {
            guard let recognized = recognizedQuantizedStem(forWeightName: name) else {
                continue
            }
            let stem = String(name.dropLast(".weight".count))
            let scalesName = "\(stem).scales"
            let biasesName = "\(stem).biases"
            let hasScales = selectedKeys.contains(scalesName)
            let hasBiases = selectedKeys.contains(biasesName)
            let weightInfo = try tensorInfo(named: name, index: index, headers: headers)
            guard weightInfo.dtype == TensorDType.u32.rawValue else {
                // Unquantized exports and tiny synthetic fixtures are passed
                // through without affine companions; companions without
                // packed U32 codes are structural corruption.
                guard !hasScales, !hasBiases else {
                    throw MLXFastError.invalidInput(
                        "Laguna projection \(stem) has affine scales or biases but no packed U32 weight"
                    )
                }
                continue
            }
            guard hasScales, hasBiases else {
                throw MLXFastError.invalidInput(
                    "quantized Laguna projection \(name) is missing BF16 scales or biases"
                )
            }
            let scalesInfo = try tensorInfo(named: scalesName, index: index, headers: headers)
            let biasesInfo = try tensorInfo(named: biasesName, index: index, headers: headers)
            let expectedRank = recognized.kind == .stackedExperts ? 3 : 2
            guard scalesInfo.dtype == TensorDType.bf16.rawValue,
                  biasesInfo.dtype == TensorDType.bf16.rawValue,
                  scalesInfo.shape == biasesInfo.shape,
                  weightInfo.shape.count == expectedRank,
                  scalesInfo.shape.count == expectedRank,
                  weightInfo.shape.dropLast() == scalesInfo.shape.dropLast(),
                  weightInfo.shape.allSatisfy({ $0 > 0 }),
                  scalesInfo.shape.allSatisfy({ $0 > 0 })
            else {
                throw MLXFastError.invalidInput(
                    "Laguna projection \(stem) has incompatible weight, scale, or bias metadata"
                )
            }

            let expected = quantization.expected(forTensorStem: stem)
            let packedWidth = weightInfo.shape[expectedRank - 1]
            let groupCount = scalesInfo.shape[expectedRank - 1]
            let (inputFeatures, inputOverflow) = groupCount.multipliedReportingOverflow(
                by: expected.groupSize
            )
            guard !inputOverflow else {
                throw MLXFastError.invalidInput(
                    "Laguna projection \(stem) input dimension overflows Int"
                )
            }
            let (packedBits, packedOverflow) = packedWidth.multipliedReportingOverflow(by: 32)
            let (expectedPackedBits, expectedOverflow) = inputFeatures.multipliedReportingOverflow(
                by: expected.bits
            )
            guard !packedOverflow, !expectedOverflow else {
                throw MLXFastError.invalidInput(
                    "Laguna projection \(stem) packed width overflows Int"
                )
            }
            guard packedBits == expectedPackedBits else {
                throw MLXFastError.invalidInput(
                    "quantized Laguna projection \(stem) stored width \(packedWidth) does not "
                        + "match config quantization group_size \(expected.groupSize) "
                        + "bits \(expected.bits) for input dimension \(inputFeatures)"
                )
            }

            switch recognized.kind {
            case .matrix:
                break
            case .routerGate:
                try validateRouterCorrectionBias(
                    routerStem: stem,
                    routerRows: weightInfo.shape[0],
                    selectedKeys: selectedKeys,
                    index: index,
                    headers: headers
                )
                if let layerIndex = recognized.layerIndex {
                    routerExpertsByLayer[layerIndex] = weightInfo.shape[0]
                }
            case .stackedExperts:
                if let layerIndex = recognized.layerIndex {
                    let experts = weightInfo.shape[0]
                    if let existing = stackedExpertsByLayer[layerIndex], existing != experts {
                        throw MLXFastError.invalidInput(
                            "Laguna layer \(layerIndex) stacked expert tensors disagree on the "
                                + "expert count (\(existing) vs \(experts))"
                        )
                    }
                    stackedExpertsByLayer[layerIndex] = experts
                }
            }
        }

        for (layerIndex, experts) in stackedExpertsByLayer.sorted(by: { $0.key < $1.key }) {
            if let routerExperts = routerExpertsByLayer[layerIndex], routerExperts != experts {
                throw MLXFastError.invalidInput(
                    "Laguna layer \(layerIndex) router rows (\(routerExperts)) do not match the "
                        + "stacked expert count (\(experts))"
                )
            }
        }
    }

    /// A quantized router (`<layer>.mlp.gate.proj`) always pairs with the
    /// raw `<layer>.mlp.gate.e_score_correction_bias` vector: BF16, one
    /// entry per routed expert.
    private static func validateRouterCorrectionBias(
        routerStem: String,
        routerRows: Int,
        selectedKeys: Set<String>,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader]
    ) throws {
        let biasName = String(routerStem.dropLast(".proj".count)) + ".e_score_correction_bias"
        guard selectedKeys.contains(biasName) else {
            throw MLXFastError.invalidInput(
                "Laguna router \(routerStem) is missing its e_score_correction_bias tensor"
            )
        }
        let biasInfo = try tensorInfo(named: biasName, index: index, headers: headers)
        guard biasInfo.dtype == TensorDType.bf16.rawValue,
              biasInfo.shape == [routerRows]
        else {
            throw MLXFastError.invalidInput(
                "Laguna router correction bias \(biasName) must be BF16 with one entry per expert"
            )
        }
    }

    private static func tensorInfo(
        named name: String,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader]
    ) throws -> SafetensorInfo {
        guard let shardName = index.weightMap[name],
              let info = headers[shardName]?.tensors[name]
        else {
            throw MLXFastError.invalidInput("missing validated tensor metadata for \(name)")
        }
        return info
    }

    private static func validateGroupAndBits(
        groupSize: Int,
        bits: Int,
        context: String
    ) throws {
        guard groupSize > 0, [2, 4, 8].contains(bits) else {
            throw MLXFastError.invalidInput(
                "Laguna \(context) must use a positive group size and bits in {2, 4, 8}"
            )
        }
    }

    private static func intField(
        _ key: String,
        in object: [String: Any],
        defaultValue: Int
    ) throws -> Int {
        guard let value = object[key], !(value is NSNull) else {
            return defaultValue
        }
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              !CFNumberIsFloatType(number),
              let integer = Int(number.stringValue)
        else {
            throw MLXFastError.invalidInput(
                "Laguna quantization field \(key) must be a finite integer in Int range"
            )
        }
        return integer
    }

    private static func stringField(
        _ key: String,
        in object: [String: Any],
        defaultValue: String
    ) throws -> String {
        guard let value = object[key], !(value is NSNull) else {
            return defaultValue
        }
        guard let string = value as? String else {
            throw MLXFastError.invalidInput(
                "Laguna quantization field \(key) must be a string"
            )
        }
        return string
    }
}
