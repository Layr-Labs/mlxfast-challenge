import CoreFoundation
import Foundation
import MLXFastCore

/// Attention layer type for a single Gemma 4 decoder layer. Gemma 4 alternates
/// several sliding-window layers with one full-attention layer per block
/// (`layer_types` in the source config), matching Gemma 2/3's local/global
/// interleaving but with different per-type head dimensions.
public enum Gemma4LayerType: String, Equatable {
    case sliding = "sliding_attention"
    case full = "full_attention"
}

/// Per-attention-type RoPE parameters. Sliding layers use standard rotary
/// embeddings over the full head dimension; full-attention layers use a
/// "proportional" partial rotation (see `Gemma4ProportionalRoPE`) computed
/// with a much larger theta.
public struct Gemma4RopeSpec: Equatable {
    public let theta: Double
    public let type: String
    public let partialRotaryFactor: Double

    public init(theta: Double, type: String, partialRotaryFactor: Double = 1.0) {
        self.theta = theta
        self.type = type
        self.partialRotaryFactor = partialRotaryFactor
    }
}

/// Text-tower configuration for Gemma 4 (dense, 31B, 4-bit). Parsed from the
/// config.json written by `SwiftTransform` into the transformed weights
/// directory -- the transform controls this schema directly, so it mirrors
/// the source Hugging Face `text_config` fields the runtime actually needs
/// rather than the full multimodal config.
public struct Gemma4Config: Equatable {
    public let modelType: String
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let numGlobalKeyValueHeads: Int
    public let headDim: Int
    public let globalHeadDim: Int
    public let rmsNormEps: Double
    public let hiddenActivation: String
    public let maxPositionEmbeddings: Int
    public let attentionBias: Bool
    public let attentionDropout: Double
    public let attentionKEqV: Bool
    public let slidingWindow: Int
    public let layerTypes: [Gemma4LayerType]
    public let finalLogitSoftcapping: Double
    public let tieWordEmbeddings: Bool
    public let slidingRope: Gemma4RopeSpec
    public let fullRope: Gemma4RopeSpec
    public let quantizationGroupSize: Int
    public let quantizationBits: Int

    public init(
        modelType: String,
        vocabSize: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        numGlobalKeyValueHeads: Int,
        headDim: Int,
        globalHeadDim: Int,
        rmsNormEps: Double,
        hiddenActivation: String,
        maxPositionEmbeddings: Int,
        attentionBias: Bool,
        attentionDropout: Double,
        attentionKEqV: Bool,
        slidingWindow: Int,
        layerTypes: [Gemma4LayerType],
        finalLogitSoftcapping: Double,
        tieWordEmbeddings: Bool,
        slidingRope: Gemma4RopeSpec,
        fullRope: Gemma4RopeSpec,
        quantizationGroupSize: Int,
        quantizationBits: Int
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.numGlobalKeyValueHeads = numGlobalKeyValueHeads
        self.headDim = headDim
        self.globalHeadDim = globalHeadDim
        self.rmsNormEps = rmsNormEps
        self.hiddenActivation = hiddenActivation
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
        self.attentionKEqV = attentionKEqV
        self.slidingWindow = slidingWindow
        self.layerTypes = layerTypes
        self.finalLogitSoftcapping = finalLogitSoftcapping
        self.tieWordEmbeddings = tieWordEmbeddings
        self.slidingRope = slidingRope
        self.fullRope = fullRope
        self.quantizationGroupSize = quantizationGroupSize
        self.quantizationBits = quantizationBits
    }

    public func headDim(for layerType: Gemma4LayerType) -> Int {
        layerType == .full ? globalHeadDim : headDim
    }

    public func numKeyValueHeads(for layerType: Gemma4LayerType) -> Int {
        (attentionKEqV && layerType == .full) ? numGlobalKeyValueHeads : numKeyValueHeads
    }

    public func usesKEqV(for layerType: Gemma4LayerType) -> Bool {
        attentionKEqV && layerType == .full
    }

    public func rope(for layerType: Gemma4LayerType) -> Gemma4RopeSpec {
        layerType == .full ? fullRope : slidingRope
    }

    public static func load(from weightsPath: String) throws -> Gemma4Config {
        let path = URL(fileURLWithPath: weightsPath).appendingPathComponent("config.json")
        try requireFile(path.path, description: "transformed weights config")

        let data = try Data(contentsOf: path)
        let object = try JSONSerialization.jsonObject(with: data)
        guard let root = object as? [String: Any] else {
            throw MLXFastError.invalidInput("config.json must be a JSON object")
        }

        let numHiddenLayers = try intField("num_hidden_layers", root: root, defaultValue: MLXFastConstants.numHiddenLayers)
        guard numHiddenLayers == MLXFastConstants.numHiddenLayers else {
            throw MLXFastError.invalidInput(
                "Gemma 4 config invariant check failed: num_hidden_layers=\(numHiddenLayers) expected \(MLXFastConstants.numHiddenLayers)"
            )
        }
        let ropeParameters = root["rope_parameters"] as? [String: Any]
        let slidingRopeObject = ropeParameters?["sliding_attention"] as? [String: Any] ?? [:]
        let fullRopeObject = ropeParameters?["full_attention"] as? [String: Any] ?? [:]
        let quantization = (root["quantization"] as? [String: Any]) ?? (root["quantization_config"] as? [String: Any]) ?? [:]

        let config = Gemma4Config(
            modelType: try stringField("model_type", root: root, defaultValue: "gemma4_text"),
            vocabSize: try intField("vocab_size", root: root, defaultValue: MLXFastConstants.vocabSize),
            hiddenSize: try intField("hidden_size", root: root, defaultValue: MLXFastConstants.hiddenSize),
            intermediateSize: try intField("intermediate_size", root: root, defaultValue: MLXFastConstants.intermediateSize),
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: try intField("num_attention_heads", root: root, defaultValue: MLXFastConstants.attentionHeads),
            numKeyValueHeads: try intField("num_key_value_heads", root: root, defaultValue: 16),
            numGlobalKeyValueHeads: try intField("num_global_key_value_heads", root: root, defaultValue: 4),
            headDim: try intField("head_dim", root: root, defaultValue: 256),
            globalHeadDim: try intField("global_head_dim", root: root, defaultValue: 512),
            rmsNormEps: try doubleField("rms_norm_eps", root: root, defaultValue: 1e-6),
            hiddenActivation: try stringField("hidden_activation", root: root, defaultValue: "gelu_pytorch_tanh"),
            maxPositionEmbeddings: try intField("max_position_embeddings", root: root, defaultValue: 262_144),
            attentionBias: try boolField("attention_bias", root: root, defaultValue: false),
            attentionDropout: try doubleField("attention_dropout", root: root, defaultValue: 0.0),
            attentionKEqV: try boolField("attention_k_eq_v", root: root, defaultValue: true),
            slidingWindow: try intField("sliding_window", root: root, defaultValue: 1024),
            layerTypes: try layerTypesField("layer_types", root: root, layerCount: numHiddenLayers),
            finalLogitSoftcapping: try doubleField("final_logit_softcapping", root: root, defaultValue: 30.0),
            tieWordEmbeddings: try boolField("tie_word_embeddings", root: root, defaultValue: true),
            slidingRope: Gemma4RopeSpec(
                theta: try doubleField("rope_theta", root: slidingRopeObject, defaultValue: 10_000.0),
                type: try stringField("rope_type", root: slidingRopeObject, defaultValue: "default"),
                partialRotaryFactor: try doubleField("partial_rotary_factor", root: slidingRopeObject, defaultValue: 1.0)
            ),
            fullRope: Gemma4RopeSpec(
                theta: try doubleField("rope_theta", root: fullRopeObject, defaultValue: 1_000_000.0),
                type: try stringField("rope_type", root: fullRopeObject, defaultValue: "proportional"),
                partialRotaryFactor: try doubleField("partial_rotary_factor", root: fullRopeObject, defaultValue: 0.25)
            ),
            quantizationGroupSize: try intField("group_size", root: quantization, defaultValue: 64),
            quantizationBits: try intField("bits", root: quantization, defaultValue: 4)
        )
        try config.validateFrozenInvariants()
        try config.validateStructuralValues()
        return config
    }

    public func validateFrozenInvariants() throws {
        let expected: [(String, Int, Int)] = [
            ("vocab_size", vocabSize, MLXFastConstants.vocabSize),
            ("hidden_size", hiddenSize, MLXFastConstants.hiddenSize),
            ("intermediate_size", intermediateSize, MLXFastConstants.intermediateSize),
            ("num_hidden_layers", numHiddenLayers, MLXFastConstants.numHiddenLayers),
            ("num_attention_heads", numAttentionHeads, MLXFastConstants.attentionHeads),
        ]
        let errors = expected.compactMap { name, actual, expected in
            actual == expected ? nil : "\(name)=\(actual) expected \(expected)"
        }
        let layerTypeErrors: [String] = layerTypes.count == numHiddenLayers
            ? []
            : ["layer_types count=\(layerTypes.count) expected \(numHiddenLayers)"]
        let allErrors = errors + layerTypeErrors
        if !allErrors.isEmpty {
            throw MLXFastError.invalidInput(
                "Gemma 4 config invariant check failed: \(allErrors.joined(separator: ", "))"
            )
        }
    }

    public func validateStructuralValues() throws {
        guard numAttentionHeads > 0, numKeyValueHeads > 0, numGlobalKeyValueHeads > 0 else {
            throw MLXFastError.invalidInput("Gemma 4 attention and KV head counts must be positive")
        }
        guard headDim > 0, headDim.isMultiple(of: 2),
              globalHeadDim > 0, globalHeadDim.isMultiple(of: 2)
        else {
            throw MLXFastError.invalidInput("Gemma 4 attention head dimensions must be positive and even")
        }
        let projectionDimensions = [
            numAttentionHeads.multipliedReportingOverflow(by: headDim),
            numAttentionHeads.multipliedReportingOverflow(by: globalHeadDim),
            numKeyValueHeads.multipliedReportingOverflow(by: headDim),
            numGlobalKeyValueHeads.multipliedReportingOverflow(by: globalHeadDim),
        ]
        guard projectionDimensions.allSatisfy({ !$0.overflow && $0.partialValue > 0 }) else {
            throw MLXFastError.invalidInput("Gemma 4 attention projection dimensions overflow Int")
        }
        guard numAttentionHeads.isMultiple(of: numKeyValueHeads),
              numAttentionHeads.isMultiple(of: numGlobalKeyValueHeads)
        else {
            throw MLXFastError.invalidInput("Gemma 4 attention heads must be divisible by both KV head counts")
        }
        guard maxPositionEmbeddings > 0,
              slidingWindow > 0,
              slidingWindow <= maxPositionEmbeddings
        else {
            throw MLXFastError.invalidInput(
                "Gemma 4 sliding window must be positive and no larger than max_position_embeddings"
            )
        }
        guard rmsNormEps.isFinite, rmsNormEps > 0 else {
            throw MLXFastError.invalidInput("Gemma 4 rms_norm_eps must be finite and positive")
        }
        guard attentionDropout.isFinite, attentionDropout >= 0, attentionDropout < 1 else {
            throw MLXFastError.invalidInput("Gemma 4 attention_dropout must be finite and in 0..<1")
        }
        guard finalLogitSoftcapping.isFinite, finalLogitSoftcapping > 0 else {
            throw MLXFastError.invalidInput("Gemma 4 final_logit_softcapping must be finite and positive")
        }
        try validateRopeSpec(slidingRope, dims: headDim, label: "sliding_attention")
        try validateRopeSpec(fullRope, dims: globalHeadDim, label: "full_attention")
        guard quantizationGroupSize > 0,
              hiddenSize.isMultiple(of: quantizationGroupSize),
              intermediateSize.isMultiple(of: quantizationGroupSize)
        else {
            throw MLXFastError.invalidInput(
                "Gemma 4 quantization group_size must divide hidden and intermediate dimensions"
            )
        }
        guard [2, 4, 8].contains(quantizationBits) else {
            throw MLXFastError.invalidInput("Gemma 4 quantization bits must be 2, 4, or 8")
        }
    }

    private func validateRopeSpec(_ spec: Gemma4RopeSpec, dims: Int, label: String) throws {
        guard spec.theta.isFinite, spec.theta > 0 else {
            throw MLXFastError.invalidInput("Gemma 4 \(label) rope_theta must be finite and positive")
        }
        guard spec.partialRotaryFactor.isFinite,
              spec.partialRotaryFactor > 0,
              spec.partialRotaryFactor <= 1,
              spec.partialRotaryFactor * Double(dims) >= 2
        else {
            throw MLXFastError.invalidInput(
                "Gemma 4 \(label) partial_rotary_factor must rotate at least one pair and no more than the head"
            )
        }
    }
}

private func fieldValue(_ key: String, root: [String: Any]) -> Any? {
    guard let value = root[key], !(value is NSNull) else {
        return nil
    }
    return value
}

private func intField(_ key: String, root: [String: Any], defaultValue: Int) throws -> Int {
    guard let value = fieldValue(key, root: root) else {
        return defaultValue
    }
    return try parseInt(value, field: key)
}

private func doubleField(_ key: String, root: [String: Any], defaultValue: Double) throws -> Double {
    guard let value = fieldValue(key, root: root) else {
        return defaultValue
    }
    return try parseDouble(value, field: key)
}

private func boolField(_ key: String, root: [String: Any], defaultValue: Bool) throws -> Bool {
    guard let value = fieldValue(key, root: root) else {
        return defaultValue
    }
    guard let number = value as? NSNumber,
          CFGetTypeID(number) == CFBooleanGetTypeID()
    else {
        throw MLXFastError.invalidInput("config field \(key) must be a boolean")
    }
    return number.boolValue
}

private func stringField(_ key: String, root: [String: Any], defaultValue: String) throws -> String {
    guard let value = fieldValue(key, root: root) else {
        return defaultValue
    }
    guard let string = value as? String else {
        throw MLXFastError.invalidInput("config field \(key) must be a string")
    }
    return string
}

private func layerTypesField(_ key: String, root: [String: Any], layerCount: Int) throws -> [Gemma4LayerType] {
    guard let value = fieldValue(key, root: root) else {
        // Default Gemma 4 pattern: five sliding layers, then one full-attention layer.
        return (0..<layerCount).map { $0 % 6 == 5 ? .full : .sliding }
    }
    guard let rawValues = value as? [String] else {
        throw MLXFastError.invalidInput("config field \(key) must be a string array")
    }
    return try rawValues.map { raw in
        guard let type = Gemma4LayerType(rawValue: raw) else {
            throw MLXFastError.invalidInput("config field \(key) contains unsupported layer type \(raw)")
        }
        return type
    }
}

private func parseInt(_ value: Any, field: String) throws -> Int {
    guard let number = value as? NSNumber,
          CFGetTypeID(number) != CFBooleanGetTypeID(),
          !CFNumberIsFloatType(number),
          let integer = Int(number.stringValue)
    else {
        throw MLXFastError.invalidInput("config field \(field) must be a finite integer in Int range")
    }
    return integer
}

private func parseDouble(_ value: Any, field: String) throws -> Double {
    guard let number = value as? NSNumber,
          CFGetTypeID(number) != CFBooleanGetTypeID()
    else {
        throw MLXFastError.invalidInput("config field \(field) must be a finite number")
    }
    let double = number.doubleValue
    guard double.isFinite else {
        throw MLXFastError.invalidInput("config field \(field) must be a finite number")
    }
    return double
}
