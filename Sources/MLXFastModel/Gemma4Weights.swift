import Foundation
import MLX
import MLXFastCore

/// Tensor name helpers for the Gemma 4 text tower. The transform keeps the
/// source checkpoint's tensor names (including the `language_model.model.`
/// prefix) unchanged when it copies text-tower tensors into the transformed
/// weights tree, so the runtime loader can address them directly without a
/// rename pass.
public enum Gemma4WeightNames {
    private static let prefix = "language_model.model"

    public static let embedTokens = "\(prefix).embed_tokens.weight"
    public static let finalNorm = "\(prefix).norm.weight"
    public static let lmHead = "language_model.lm_head.weight"

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

public struct Gemma4WeightLoader {
    public let denseStore: DenseTensorStore
    private let bridge: MLXArrayTensorBridge

    public init(weightsPath: String, bridge: MLXArrayTensorBridge = MLXArrayTensorBridge()) throws {
        self.denseStore = try DenseTensorStore(weightsPath: weightsPath)
        self.bridge = bridge
    }

    public init(denseStore: DenseTensorStore, bridge: MLXArrayTensorBridge = MLXArrayTensorBridge()) {
        self.denseStore = denseStore
        self.bridge = bridge
    }

    // MARK: - Primitive tensor access

    public func materializedTensor(named name: String, expectedShape: [Int]? = nil) throws -> MaterializedTensor {
        let tensor = try denseStore.materializedTensor(named: name)
        try validateShape(tensor.shape, expectedShape: expectedShape, tensorName: name)
        return tensor
    }

    public func denseArray(named name: String, expectedShape: [Int]? = nil) throws -> MLXArray {
        try bridge.makeArray(from: materializedTensor(named: name, expectedShape: expectedShape))
    }

    public func optionalDenseArray(named name: String, expectedShape: [Int]? = nil) throws -> MLXArray? {
        guard denseStore.record(named: name) != nil else {
            return nil
        }
        return try denseArray(named: name, expectedShape: expectedShape)
    }

    /// Loads a (possibly quantized) linear weight by its logical `.weight`
    /// tensor name. When the underlying tensor is `U32`, the matching
    /// `.scales`/`.biases` companions (BF16) are required, and `bits`/
    /// `groupSize` are derived from the packed shapes -- the same recipe
    /// `DenseTensorStore`-backed loaders elsewhere in this package use for
    /// affine 4-bit quantized checkpoints.
    public func linearWeight(named baseName: String, outFeatures: Int, inFeatures: Int) throws -> Gemma4LinearWeight {
        let quantization = try validateLinearMetadata(
            named: baseName,
            expectedRows: outFeatures,
            expectedInputFeatures: inFeatures
        )
        let tensor = try materializedTensor(named: baseName)
        guard let quantization else {
            return Gemma4LinearWeight(try bridge.makeArray(from: tensor))
        }

        let scalesName = Self.companionName(for: baseName, suffix: "scales")
        let biasesName = Self.companionName(for: baseName, suffix: "biases")
        let scalesTensor = try materializedTensor(named: scalesName)
        let biasesTensor = try materializedTensor(named: biasesName)

        return Gemma4LinearWeight(
            weight: try bridge.makeArray(from: tensor),
            scales: try bridge.makeArray(from: scalesTensor),
            biases: try bridge.makeArray(from: biasesTensor),
            logicalShape: [outFeatures, inFeatures],
            groupSize: quantization.groupSize,
            bits: quantization.bits
        )
    }

    /// Validates that every tensor the model needs is present with the
    /// expected shape, without materializing any `MLXArray`s. Used by
    /// preflight/worker-boundary checks that must fail fast on a malformed
    /// transformed weights directory before the (expensive) full weight load.
    public func validateRequiredMetadata(config: Gemma4Config) throws {
        try validateLinearMetadata(
            named: Gemma4WeightNames.embedTokens,
            expectedRows: config.vocabSize,
            expectedInputFeatures: config.hiddenSize,
            expectedGroupSize: config.quantizationGroupSize,
            expectedBits: config.quantizationBits
        )
        try validateDenseTensorMetadata(
            named: Gemma4WeightNames.finalNorm,
            expectedShape: [config.hiddenSize]
        )
        if !config.tieWordEmbeddings {
            try validateLinearMetadata(
                named: Gemma4WeightNames.lmHead,
                expectedRows: config.vocabSize,
                expectedInputFeatures: config.hiddenSize,
                expectedGroupSize: config.quantizationGroupSize,
                expectedBits: config.quantizationBits
            )
        }

        for layerIndex in 0..<config.numHiddenLayers {
            let layerType = config.layerTypes[layerIndex]
            let headDim = config.headDim(for: layerType)
            let kvHeads = config.numKeyValueHeads(for: layerType)
            let useKEqV = config.usesKEqV(for: layerType)

            for suffix in [
                "input_layernorm.weight", "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
            ] {
                try validateDenseTensorMetadata(
                    named: Gemma4WeightNames.layer(layerIndex, suffix),
                    expectedShape: [config.hiddenSize]
                )
            }
            try validateDenseTensorMetadata(
                named: Gemma4WeightNames.layer(layerIndex, "layer_scalar"),
                expectedShape: [1]
            )

            try validateLinearMetadata(
                named: Gemma4WeightNames.attention(layerIndex, "q_proj.weight"),
                expectedRows: config.numAttentionHeads * headDim,
                expectedInputFeatures: config.hiddenSize,
                expectedGroupSize: config.quantizationGroupSize,
                expectedBits: config.quantizationBits
            )
            try validateLinearMetadata(
                named: Gemma4WeightNames.attention(layerIndex, "k_proj.weight"),
                expectedRows: kvHeads * headDim,
                expectedInputFeatures: config.hiddenSize,
                expectedGroupSize: config.quantizationGroupSize,
                expectedBits: config.quantizationBits
            )
            if !useKEqV {
                try validateLinearMetadata(
                    named: Gemma4WeightNames.attention(layerIndex, "v_proj.weight"),
                    expectedRows: kvHeads * headDim,
                    expectedInputFeatures: config.hiddenSize,
                    expectedGroupSize: config.quantizationGroupSize,
                    expectedBits: config.quantizationBits
                )
            }
            try validateLinearMetadata(
                named: Gemma4WeightNames.attention(layerIndex, "o_proj.weight"),
                expectedRows: config.hiddenSize,
                expectedInputFeatures: config.numAttentionHeads * headDim,
                expectedGroupSize: config.quantizationGroupSize,
                expectedBits: config.quantizationBits
            )
            try validateDenseTensorMetadata(
                named: Gemma4WeightNames.attention(layerIndex, "q_norm.weight"),
                expectedShape: [headDim]
            )
            try validateDenseTensorMetadata(
                named: Gemma4WeightNames.attention(layerIndex, "k_norm.weight"),
                expectedShape: [headDim]
            )

            try validateLinearMetadata(
                named: Gemma4WeightNames.mlp(layerIndex, "gate_proj.weight"),
                expectedRows: config.intermediateSize,
                expectedInputFeatures: config.hiddenSize,
                expectedGroupSize: config.quantizationGroupSize,
                expectedBits: config.quantizationBits
            )
            try validateLinearMetadata(
                named: Gemma4WeightNames.mlp(layerIndex, "up_proj.weight"),
                expectedRows: config.intermediateSize,
                expectedInputFeatures: config.hiddenSize,
                expectedGroupSize: config.quantizationGroupSize,
                expectedBits: config.quantizationBits
            )
            try validateLinearMetadata(
                named: Gemma4WeightNames.mlp(layerIndex, "down_proj.weight"),
                expectedRows: config.hiddenSize,
                expectedInputFeatures: config.intermediateSize,
                expectedGroupSize: config.quantizationGroupSize,
                expectedBits: config.quantizationBits
            )
        }
    }

    /// Validates a plain (non-quantized) dense tensor's shape without
    /// materializing it.
    private func validateDenseTensorMetadata(named name: String, expectedShape: [Int]) throws {
        guard let record = denseStore.record(named: name) else {
            throw MLXFastError.invalidInput("dense tensor not found: \(name)")
        }
        guard record.shape == expectedShape else {
            throw MLXFastError.invalidInput(
                "tensor \(name) shape \(record.shape) does not match expected shape \(expectedShape)"
            )
        }
    }

    /// Validates a (possibly quantized) linear tensor's logical row/input
    /// dimensions without materializing it. Quantized tensors are packed
    /// (`U32` codes), so their on-disk shape does not equal the logical
    /// `[outFeatures, inFeatures]` shape directly.
    @discardableResult
    func validateLinearMetadata(
        named name: String,
        expectedRows: Int,
        expectedInputFeatures: Int,
        expectedGroupSize: Int? = nil,
        expectedBits: Int? = nil
    ) throws -> (groupSize: Int, bits: Int)? {
        guard let record = denseStore.record(named: name) else {
            throw MLXFastError.invalidInput("dense tensor not found: \(name)")
        }
        if record.dtype != "U32" {
            try validateDenseTensorMetadata(named: name, expectedShape: [expectedRows, expectedInputFeatures])
            return nil
        }
        guard expectedInputFeatures > 0 else {
            throw MLXFastError.invalidInput("quantized tensor \(name) logical input must be positive")
        }
        guard record.shape.count == 2, record.shape[0] == expectedRows, record.shape[1] > 0 else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) shape \(record.shape) must be [\(expectedRows), packedInput]"
            )
        }
        let (packedBits, overflow) = record.shape[1].multipliedReportingOverflow(by: 32)
        guard !overflow, packedBits % expectedInputFeatures == 0 else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) packed input \(record.shape[1]) is incompatible with logical input \(expectedInputFeatures)"
            )
        }
        let bits = packedBits / expectedInputFeatures
        guard [2, 4, 8].contains(bits) else {
            throw MLXFastError.invalidInput("quantized tensor \(name) has unsupported bit width \(bits)")
        }

        let scalesName = Self.companionName(for: name, suffix: "scales")
        let biasesName = Self.companionName(for: name, suffix: "biases")
        guard let scales = denseStore.record(named: scalesName) else {
            throw MLXFastError.invalidInput("quantized tensor \(name) is missing companion \(scalesName)")
        }
        guard let biases = denseStore.record(named: biasesName) else {
            throw MLXFastError.invalidInput("quantized tensor \(name) is missing companion \(biasesName)")
        }
        guard scales.dtype == "BF16" else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) scales must use BF16, found \(scales.dtype)"
            )
        }
        guard biases.dtype == "BF16" else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) biases must use BF16, found \(biases.dtype)"
            )
        }
        guard scales.shape.count == 2,
              scales.shape[0] == expectedRows,
              scales.shape[1] > 0,
              expectedInputFeatures % scales.shape[1] == 0
        else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) scales shape \(scales.shape) is incompatible with [\(expectedRows), groups] and logical input \(expectedInputFeatures)"
            )
        }
        guard biases.shape == scales.shape else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) biases shape \(biases.shape) does not match scales shape \(scales.shape)"
            )
        }
        let groupSize = expectedInputFeatures / scales.shape[1]
        if let expectedGroupSize, groupSize != expectedGroupSize {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) group size \(groupSize) does not match configured group size \(expectedGroupSize)"
            )
        }
        if let expectedBits, bits != expectedBits {
            throw MLXFastError.invalidInput(
                "quantized tensor \(name) bit width \(bits) does not match configured bit width \(expectedBits)"
            )
        }
        return (groupSize: groupSize, bits: bits)
    }

    // MARK: - Helpers

    private func validateShape(_ actualShape: [Int], expectedShape: [Int]?, tensorName: String) throws {
        guard let expectedShape else {
            return
        }
        guard actualShape == expectedShape else {
            throw MLXFastError.invalidInput(
                "tensor \(tensorName) shape \(actualShape) does not match expected shape \(expectedShape)"
            )
        }
    }

    private static func companionName(for baseName: String, suffix: String) -> String {
        guard baseName.hasSuffix(".weight") else {
            return "\(baseName).\(suffix)"
        }
        return String(baseName.dropLast(".weight".count)) + ".\(suffix)"
    }
}
