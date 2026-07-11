import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

func gemma4LastTokenRange(sequenceLength: Int) -> Range<Int>? {
    sequenceLength > 1 ? (sequenceLength - 1)..<sequenceLength : nil
}

func gemma4LastTokenHidden(_ hidden: MLXArray) -> MLXArray {
    guard let range = gemma4LastTokenRange(sequenceLength: hidden.dim(1)) else {
        return hidden
    }
    return hidden[0..., range, 0...]
}

/// The pinned mlx-swift-lm Gemma 4 trunk with a last-token-only language head.
/// Every benchmark consumer reads only the final sequence row, so projecting
/// earlier rows to the 262k vocabulary is dead work.
public final class Gemma4RuntimeModel: Module, LanguageModel {
    @ModuleInfo(key: "model") var model: Gemma4TextModelInner

    private let config: Gemma4TextConfiguration

    private static let logitSoftcap: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { logits, cap in
            tanh(logits / cap) * cap
        }
        let enabled: Bool
        if let raw = ProcessInfo.processInfo.environment["MLX_COMPILED_DECODE"] {
            enabled = ["1", "true", "yes", "on"].contains(raw.lowercased())
        } else {
            enabled = true
        }
        return enabled ? compile(shapeless: true, body) : body
    }()

    public init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self._model.wrappedValue = Gemma4TextModelInner(config)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let fullHidden = model(inputs, cache: cache)
        let hidden = gemma4LastTokenHidden(fullHidden)
        let cap = MLXArray(config.finalLogitSoftcapping)
        let logits = Self.logitSoftcap(model.embedTokens.asLinear(hidden), cap)
        return logits
    }

    public func prepare(
        _ input: LMInput,
        cache _: [KVCache],
        windowSize _: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    public func newCache(parameters _: GenerateParameters?) -> [KVCache] {
        let firstSharedLayer = config.numHiddenLayers - config.numKvSharedLayers
        return (0..<firstSharedLayer).map { layerIndex in
            if config.layerTypes[layerIndex] == Gemma4LayerType.full.rawValue {
                StandardKVCache()
            } else {
                RotatingKVCache(maxSize: config.slidingWindow, keep: 0)
            }
        }
    }
}
