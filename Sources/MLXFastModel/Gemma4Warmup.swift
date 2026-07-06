import Foundation
import MLX
import MLXFastCore

/// Constructor-time warmup for the full-size checkpoint.
///
/// The runtime worker builds its weight cache before the benchmark protocol
/// handshake, so everything here runs outside every scored window. Warmup
/// exercises the hot Metal kernels at decode (M=1) and prefill (M=512)
/// shapes with synthetic zero tensors so pipeline-state creation and MLX
/// kernel-cache population happen now, then runs one throwaway single-token
/// forward with real weights to warm the full decode path. Synthetic inputs
/// are all-zero and outputs are discarded, so this is prompt-independent and
/// cannot affect model output.
enum Gemma4Warmup {
    static func run(weightCache: Gemma4RuntimeWeightCache) {
        warmKernels(config: weightCache.config)
        runThrowawayDecodeForward(weightCache: weightCache)
    }

    private static func warmKernels(config: Gemma4Config) {
        let hidden = config.hiddenSize
        guard hidden % 64 == 0 else { return }
        let rows = 512
        let affineWeight = zeros([rows, hidden / 8], dtype: .uint32)
        let affineScales = zeros([rows, hidden / 64], dtype: .bfloat16)
        let affineBiases = zeros([rows, hidden / 64], dtype: .bfloat16)

        for m in [1, 512] {
            let x = zeros([1, m, hidden], dtype: .bfloat16)
            eval(quantizedMM(
                x, affineWeight,
                scales: affineScales, biases: affineBiases,
                transpose: true, groupSize: 64, bits: 4, mode: .affine
            ))
        }

        for layerType in [Gemma4LayerType.sliding, .full] {
            let heads = config.numAttentionHeads
            let headDim = config.headDim(for: layerType)
            let kvHeads = config.numKeyValueHeads(for: layerType)
            let window = layerType == .sliding ? max(config.slidingWindow, 1) : nil
            for (queryLength, keyLength) in [(1, (window ?? 0) + 1), (512, 512)] {
                guard
                    let mask = try? Gemma4MaskCache.causal(
                        queryLength: queryLength,
                        keyLength: keyLength,
                        queryOffset: keyLength - queryLength,
                        keyOffset: 0,
                        windowSize: window
                    )
                else { continue }
                let q = zeros([1, heads, queryLength, headDim], dtype: .bfloat16)
                let kv = zeros([1, kvHeads, keyLength, headDim], dtype: .bfloat16)
                eval(MLXFast.scaledDotProductAttention(
                    queries: q, keys: kv, values: kv,
                    scale: 1.0,
                    mask: mask
                ))
            }
        }

        eval(MLXFast.rmsNorm(
            zeros([512, hidden], dtype: .bfloat16),
            weight: zeros([hidden], dtype: .bfloat16),
            eps: 1e-6
        ))
        eval(zeros([config.vocabSize], dtype: .bfloat16).argMax())
    }

    private static func runThrowawayDecodeForward(weightCache: Gemma4RuntimeWeightCache) {
        let cache = Gemma4ModelCache(config: weightCache.config)
        let inputIDs = MLXArray([Int32(0)]).reshaped([1, 1])
        guard let logits = try? Gemma4Model.logits(
            inputIDs: inputIDs,
            weightCache: weightCache,
            cache: cache,
            positionOffset: 0
        ) else { return }
        eval(logits)
    }
}
