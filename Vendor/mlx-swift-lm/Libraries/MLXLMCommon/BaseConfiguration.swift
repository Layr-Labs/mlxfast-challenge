// Copyright © 2025 Apple Inc.

import Foundation
import MLX

public struct BaseConfiguration: Codable, Sendable {

    public let modelType: String

    public struct Quantization: Codable, Sendable, Equatable {

        public init(groupSize: Int, bits: Int) {
            self.groupSize = groupSize
            self.bits = bits
        }

        public let groupSize: Int

        public let bits: Int

        private var _mode: QuantizationMode? = nil

        public var mode: QuantizationMode { _mode ?? .affine }

        public var asTuple: (Int, Int, QuantizationMode) { (groupSize, bits, mode) }

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits = "bits"
            case _mode = "mode"
        }
    }

    public enum QuantizationOption: Sendable {
        case skip
        case quantize(Quantization)
    }

    public struct PerLayerQuantization: Sendable {
        public var quantization: Quantization? = nil

        public var perLayerQuantization: [String: QuantizationOption]

        public init(
            quantization: BaseConfiguration.Quantization? = nil,
            perLayerQuantization: [String: BaseConfiguration.QuantizationOption]
        ) {
            self.quantization = quantization
            self.perLayerQuantization = perLayerQuantization
        }

        public func quantization(layer: String) -> Quantization? {
            if let perLayer = perLayerQuantization[layer] {
                switch perLayer {
                case .skip:
                    return nil
                case .quantize(let quantization):
                    return quantization
                }
            } else {
                return quantization
            }
        }
    }

    struct QuantizationContainer: Codable, Sendable {
        var quantization: Quantization
        var perLayerQuantization: PerLayerQuantization

        internal struct _DictionaryCodingKey: CodingKey {
            internal let stringValue: String
            internal let intValue: Int?

            internal init(stringValue: String) {
                self.stringValue = stringValue
                self.intValue = Int(stringValue)
            }

            internal init(intValue: Int) {
                self.stringValue = "\(intValue)"
                self.intValue = intValue
            }
        }

        init(from decoder: any Decoder) throws {
            self.quantization = try Quantization(from: decoder)

            var perLayerQuantization = [String: QuantizationOption]()
            let container = try decoder.container(keyedBy: _DictionaryCodingKey.self)
            for key in container.allKeys {
                switch key.stringValue {
                case Quantization.CodingKeys.groupSize.rawValue: continue
                case Quantization.CodingKeys.bits.rawValue: continue
                case Quantization.CodingKeys._mode.rawValue: continue

                case "quant_method", "linear_class", "quantization_mode": continue

                default:
                    if let f = try? container.decode(Bool.self, forKey: key) {
                        if !f {
                            perLayerQuantization[key.stringValue] = .skip
                        }
                    } else {
                        perLayerQuantization[key.stringValue] = .quantize(
                            try container.decode(Quantization.self, forKey: key))
                    }
                }
            }
            self.perLayerQuantization = PerLayerQuantization(
                quantization: quantization, perLayerQuantization: perLayerQuantization)
        }

        func encode(to encoder: any Encoder) throws {
            try quantization.encode(to: encoder)

            var container = encoder.container(keyedBy: _DictionaryCodingKey.self)
            for (key, value) in perLayerQuantization.perLayerQuantization {
                switch value {
                case .skip:
                    try container.encode(false, forKey: .init(stringValue: key))
                case .quantize(let q):
                    try container.encode(q, forKey: .init(stringValue: key))
                }
            }
        }
    }

    var quantizationContainer: QuantizationContainer?

    public var eosTokenIds: IntOrIntArray?

    @available(*, deprecated, message: "Please use perLayerQuantization instead")
    public var quantization: Quantization? {
        quantizationContainer?.quantization
    }

    public var perLayerQuantization: PerLayerQuantization? {
        quantizationContainer?.perLayerQuantization
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case quantizationContainer = "quantization"
        case eosTokenIds = "eos_token_id"
    }
}
