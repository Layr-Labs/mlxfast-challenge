import Foundation
import MLX
import MLXFastCore

/// Constructor-time warmup for full-size checkpoints.
///
/// The runtime worker builds its weight cache before the benchmark protocol
/// handshake, so everything here runs outside every scored window. Warmup
/// (1) evaluates the derived weight views (transposes, exact widenings,
/// packed reshapes) so their buffers exist before the first scored forward,
/// (2) exercises the hot Metal kernels at decode (M=1) and prefill (M=512)
/// shapes with synthetic zero tensors so pipeline-state creation and MLX
/// kernel-cache population happen now, and (3) runs one throwaway
/// single-token forward with real weights to warm the full decode path.
/// Synthetic inputs are all-zero and outputs are discarded, so this is
/// prompt-independent and cannot affect model output.
enum DeepSeekWarmup {
    static func run(weightCache: DeepSeekRuntimeWeightCache) {
        let config = weightCache.config
        evalDerivedWeights(weightCache: weightCache)
        warmKernels(config: config)
        runThrowawayDecodeForward(weightCache: weightCache)
    }

    private static func evalDerivedWeights(weightCache: DeepSeekRuntimeWeightCache) {
        if let model = try? weightCache.modelWeights() {
            var arrays = collect(model.embedTokens) + collect(model.lmHead)
            arrays.append(model.finalNorm)
            arrays += collect(model.headHyperConnection)
            eval(arrays)
        }
        for layerIndex in 0..<weightCache.config.numHiddenLayers {
            var arrays: [MLXArray] = []
            if let block = try? weightCache.blockWeights(layerIndex: layerIndex) {
                arrays.append(block.attentionNorm)
                arrays.append(block.feedForwardNorm)
                arrays += collect(block.attentionHyperConnection)
                arrays += collect(block.feedForwardHyperConnection)
            }
            if let moe = try? weightCache.moeWeights(layerIndex: layerIndex) {
                arrays.append(moe.gateTransposed)
                if let bias = moe.correctionBias { arrays.append(bias) }
                if let tokenToExpert = moe.tokenToExpert { arrays.append(tokenToExpert) }
                arrays += collect(moe.sharedExperts)
            }
            let ratio = weightCache.config.compressRatios[layerIndex]
            if ratio == 0 {
                if let attention = try? weightCache.localAttentionWeights(layerIndex: layerIndex) {
                    arrays += collect(attention)
                }
            } else if let compressed = try? weightCache.compressedAttentionWeights(layerIndex: layerIndex) {
                arrays += collect(compressed.attention)
                arrays += collect(compressed.compressor)
                if let indexer = compressed.indexer {
                    arrays += collect(indexer.wqB) + collect(indexer.weightsProj)
                    arrays += collect(indexer.compressor)
                }
            }
            if !arrays.isEmpty {
                eval(arrays)
            }
        }
    }

    private static func warmKernels(config: DeepSeekConfig) {
        let hidden = config.hiddenSize
        guard hidden % 64 == 0 else { return }
        let rows = 512
        let affineWeight = zeros([rows, hidden / 8], dtype: .uint32)
        let affineScales = zeros([rows, hidden / 64], dtype: .bfloat16)
        let affineBiases = zeros([rows, hidden / 64], dtype: .bfloat16)
        let mxfp4Scales = zeros([rows, hidden / 32], dtype: .uint8)

        for m in [1, 512] {
            let x = zeros([1, m, hidden], dtype: .bfloat16)
            eval(quantizedMM(
                x, affineWeight,
                scales: affineScales, biases: affineBiases,
                transpose: true, groupSize: 64, bits: 4, mode: .affine
            ))
            eval(quantizedMM(
                x, affineWeight,
                scales: mxfp4Scales, biases: nil,
                transpose: true, groupSize: 32, bits: 4, mode: .mxfp4
            ))
        }

        let heads = config.numAttentionHeads
        let headDim = config.headDim
        let window = max(config.slidingWindow, 1)
        for (queryLength, keyLength) in [(1, window + 1), (512, 512)] {
            guard
                let mask = try? DeepSeekMaskCache.causal(
                    queryLength: queryLength,
                    keyLength: keyLength,
                    queryOffset: keyLength - queryLength,
                    keyOffset: 0,
                    windowSize: window
                )
            else { continue }
            let q = zeros([1, heads, queryLength, headDim], dtype: .bfloat16)
            let kv = zeros([1, 1, keyLength, headDim], dtype: .bfloat16)
            eval(MLXFast.scaledDotProductAttention(
                queries: q, keys: kv, values: kv,
                scale: Float(pow(Double(headDim), -0.5)),
                mask: mask,
                sinks: zeros([heads], dtype: .bfloat16)
            ))
        }

        eval(MLXFast.rmsNorm(
            zeros([512, hidden], dtype: .bfloat16),
            weight: zeros([hidden], dtype: .bfloat16),
            eps: 1e-6
        ))
        eval(zeros([config.vocabSize], dtype: .bfloat16).argMax())
        let gateLogits = zeros([1, config.routedExperts], dtype: .float32)
        eval(softmax(gateLogits, axis: -1, precise: true))
        if config.expertsPerToken > 0, config.expertsPerToken <= config.routedExperts {
            eval(argPartition(gateLogits, kth: config.expertsPerToken - 1, axis: -1))
        }
    }

    private static func runThrowawayDecodeForward(weightCache: DeepSeekRuntimeWeightCache) {
        let cache = DeepSeekModelCache(config: weightCache.config)
        let inputIDs = MLXArray([Int32(0)]).reshaped([1, 1])
        guard let logits = try? DeepSeekModel.logits(
            inputIDs: inputIDs,
            weightCache: weightCache,
            cache: cache,
            positionOffset: 0
        ) else { return }
        eval(logits)
    }

    private static func collect(_ weight: DeepSeekLinearWeight) -> [MLXArray] {
        var arrays = [weight.weight]
        if let scales = weight.scales { arrays.append(scales) }
        if let biases = weight.biases { arrays.append(biases) }
        return arrays
    }

    private static func collect(_ mlp: DeepSeekMLPWeights) -> [MLXArray] {
        collect(mlp.gate) + collect(mlp.up) + collect(mlp.down)
    }

    private static func collect(_ hyperConnection: DeepSeekHyperConnectionWeights) -> [MLXArray] {
        [hyperConnection.fnTransposedF32, hyperConnection.baseF32, hyperConnection.scaleF32]
    }

    private static func collect(_ head: DeepSeekHeadHyperConnectionWeights) -> [MLXArray] {
        [head.fnTransposedF32, head.baseF32, head.scaleF32]
    }

    private static func collect(_ attention: DeepSeekLocalAttentionWeights) -> [MLXArray] {
        var arrays = collect(attention.wqA) + collect(attention.wqB)
            + collect(attention.wkv) + collect(attention.woA) + collect(attention.woB)
        arrays.append(attention.qNorm)
        arrays.append(attention.kvNorm)
        if let bias = attention.woBBias { arrays.append(bias) }
        if let sink = attention.attentionSink { arrays.append(sink) }
        return arrays
    }

    private static func collect(_ compressor: DeepSeekCompressorWeights) -> [MLXArray] {
        collect(compressor.wkv) + collect(compressor.wgate) + [compressor.ape, compressor.norm]
    }
}
