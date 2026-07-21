import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

private let gemma4MTPExactFourFusedArgmaxEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_MTP_EXACT_FOUR_FUSED_ARGMAX"
    ] else {
        return true
    }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

let gemma4MTPExactPairFusedArgmaxEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_MTP_EXACT_PAIR_FUSED_ARGMAX"
    ] else {
        return true
    }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

func gemma4LastTokenRange(sequenceLength: Int) -> Range<Int>? {
    sequenceLength > 1 ? (sequenceLength - 1)..<sequenceLength : nil
}

func gemma4LastTokenHidden(_ hidden: MLXArray) -> MLXArray {
    guard let range = gemma4LastTokenRange(sequenceLength: hidden.dim(1)) else {
        return hidden
    }
    return hidden[0..., range, 0...]
}

/// Scored runtime model: last-token vocabulary head plus optional fused-MLP
/// engine built from the loaded quantized modules.
public final class Gemma4RuntimeModel: Module, LanguageModel {
    @ModuleInfo(key: "model") var model: Gemma4TextModelInner

    public let configuration: Gemma4TextConfiguration
    private var fastEngine: Gemma4FastEngine?

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
        self.configuration = config
        self._model.wrappedValue = Gemma4TextModelInner(config)
        super.init()
    }

    /// Build the fused-MLP engine after weights are loaded and quantized.
    func prepareFastEngine(
        indexedMetadata: [String: IndexedAffineMetadata],
        tiedHeadPacked13Metadata: Gemma4TiedHeadPacked13Metadata?
    ) throws {
        fastEngine = try Gemma4FastEngine(
            model: self,
            indexedMetadata: indexedMetadata,
            tiedHeadPacked13Metadata: tiedHeadPacked13Metadata
        )
    }

    func fastMTPForward(
        _ inputs: MLXArray,
        cache: [KVCache]
    ) -> Gemma4MTPForward? {
        fastEngine?.forwardForMTP(inputs, cache: cache)
    }

    var supportsExactMTPPair: Bool {
        fastEngine?.supportsExactMTPPair == true
    }

    func exactMTPPair(
        _ inputs: MLXArray,
        cache: [KVCache]
    ) -> Gemma4MTPForward? {
        guard let fastEngine,
              fastEngine.canRunExactMTPPair(cache: cache)
        else {
            return nil
        }
        return fastEngine.exactMTPPair(inputs, cache: cache)
    }

    func exactMTPPairTokens(
        _ inputs: MLXArray,
        cache: [KVCache]
    ) -> Gemma4MTPTokenForward? {
        guard let fastEngine,
              fastEngine.canRunExactMTPPair(cache: cache)
        else {
            return nil
        }
        if gemma4MTPExactPairFusedArgmaxEnabled {
            return fastEngine.exactMTPPairTokens(inputs, cache: cache)
        }
        let output = fastEngine.exactMTPPair(inputs, cache: cache)
        return Gemma4MTPTokenForward(
            tokenIDs: output.logits.asType(.float32).argMax(axis: -1),
            lastHidden: output.lastHidden,
            capturedSharedKV: output.capturedSharedKV
        )
    }

    func exactMTPFour(
        _ inputs: MLXArray,
        cache: [KVCache]
    ) -> Gemma4MTPForward? {
        guard let fastEngine,
              fastEngine.canRunExactMTPFour(cache: cache)
        else {
            return nil
        }
        return fastEngine.exactMTPFour(inputs, cache: cache)
    }

    func exactMTPFourTokens(
        _ inputs: MLXArray,
        cache: [KVCache]
    ) -> Gemma4MTPTokenForward? {
        guard let fastEngine,
              fastEngine.canRunExactMTPFour(cache: cache)
        else {
            return nil
        }
        if gemma4MTPExactFourFusedArgmaxEnabled {
            return fastEngine.exactMTPFourTokens(inputs, cache: cache)
        }
        let output = fastEngine.exactMTPFour(inputs, cache: cache)
        return Gemma4MTPTokenForward(
            tokenIDs: output.logits.asType(.float32).argMax(axis: -1),
            lastHidden: output.lastHidden,
            capturedSharedKV: output.capturedSharedKV
        )
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        if let fastEngine {
            return fastEngine(inputs, cache: cache)
        }
        let fullHidden = model(inputs, cache: cache)
        let hidden = gemma4LastTokenHidden(fullHidden)
        let cap = MLXArray(configuration.finalLogitSoftcapping)
        return Self.logitSoftcap(model.embedTokens.asLinear(hidden), cap)
    }

    public func prepare(
        _ input: LMInput,
        cache _: [KVCache],
        windowSize _: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    public func newCache(parameters _: GenerateParameters?) -> [KVCache] {
        let firstSharedLayer = configuration.numHiddenLayers - configuration.numKvSharedLayers
        let useCombinedCache = gemma4CombinedKVCacheEnabled()
        return (0..<firstSharedLayer).map { layerIndex in
            if configuration.layerTypes[layerIndex] == Gemma4LayerType.full.rawValue {
                if useCombinedCache {
                    Gemma4CombinedKVCache(fullStep: 256)
                } else {
                    StandardKVCache()
                }
            } else {
                if useCombinedCache {
                    Gemma4CombinedKVCache(
                        rotatingMaxSize: configuration.slidingWindow,
                        keep: 0,
                        step: 256
                    )
                } else {
                    RotatingKVCache(maxSize: configuration.slidingWindow, keep: 0)
                }
            }
        }
    }

}
