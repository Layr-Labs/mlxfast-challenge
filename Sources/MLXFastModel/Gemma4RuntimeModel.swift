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
        packedIndexMetadata: [String: Gemma4PackedQKVIndexMetadata] = [:],
        coTiledAttentionPayloads: [String: Gemma4CoTiledAttentionPayload] = [:],
        tiedHeadPacked13Metadata: Gemma4TiedHeadPacked13Metadata?,
        tiedHeadCoTiledPayload: Gemma4TiedHeadCoTiledPayload? = nil
    ) throws {
        fastEngine = try Gemma4FastEngine(
            model: self,
            indexedMetadata: indexedMetadata,
            packedIndexMetadata: packedIndexMetadata,
            coTiledAttentionPayloads: coTiledAttentionPayloads,
            tiedHeadPacked13Metadata: tiedHeadPacked13Metadata,
            tiedHeadCoTiledPayload: tiedHeadCoTiledPayload
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
