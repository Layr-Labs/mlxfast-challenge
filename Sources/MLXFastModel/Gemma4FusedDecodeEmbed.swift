import Foundation
import MLX
import MLXFastCore
import MLXNN

private let gemma4FusedDecodeEmbedFeatureEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_FUSED_DECODE_EMBED"
    ] else {
        return true
    }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

@inline(__always)
func gemma4FusedDecodeEmbedEnabled() -> Bool {
    gemma4FusedDecodeEmbedFeatureEnabled
}

/// Fused decode-time embedding lookup. One custom kernel replaces the stock
/// five-dispatch chain — packed-weight gather, scales gather, biases gather,
/// affine dequantize, and the embed-scale scalar multiply — for the
/// single-token decode embedding step.
///
/// The kernel is purely elementwise. Each thread unpacks one packed byte into
/// two adjacent outputs with the same `scale * d + bias` bfloat expression
/// the vendored `affine_dequantize` kernel uses (byte-wise 4-bit unpack in
/// little-endian nibble order, group index `oindex / 64`), then applies the
/// same bfloat multiply the stock chain's binary `dequantized * embedScale`
/// dispatch performs; the scalar is pre-converted to bf16 exactly like the
/// stock operator's `asMLXArray(dtype: lhs.dtype)` conversion. Fast math is
/// disabled for every MLX Metal compile, so the identical expression sequence
/// rounds bit-identically; exhaustive all-vocab raw-bit equivalence is
/// covered by `Gemma4FusedDecodeEmbedTests`.
struct Gemma4FusedDecodeEmbed {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let embedScale: MLXArray
    let dims: Int
    let groups: Int
    let words: Int
    private let kernel: MLXFast.MLXFastKernel

    init?(_ embedTokens: Embedding, embedScale: Float) {
        guard gemma4FusedDecodeEmbedEnabled(),
              let quantized = embedTokens as? QuantizedEmbedding,
              quantized.groupSize == 64,
              quantized.bits == 4,
              quantized.mode == .affine,
              let biases = quantized.biases,
              quantized.weight.dtype == .uint32,
              quantized.weight.ndim == 2,
              quantized.scales.dtype == .bfloat16,
              biases.dtype == .bfloat16,
              quantized.scales.shape == biases.shape
        else { return nil }
        self.init(
            weight: quantized.weight,
            scales: quantized.scales,
            biases: biases,
            embedScale: embedScale
        )
    }

    /// Component-array form; also the entry point the exhaustive bit
    /// equivalence tests use with synthetic weights.
    init?(
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        embedScale: Float
    ) {
        guard weight.dtype == .uint32,
              weight.ndim == 2,
              scales.dtype == .bfloat16,
              biases.dtype == .bfloat16,
              scales.shape == biases.shape
        else { return nil }
        let vocabulary = weight.dim(0)
        let words = weight.dim(1)
        let groups = scales.dim(1)
        let dims = words * 8
        guard scales.dim(0) == vocabulary,
              groups * 64 == dims
        else { return nil }
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.embedScale = MLXArray(embedScale).asType(.bfloat16)
        self.dims = dims
        self.groups = groups
        self.words = words
        self.kernel = MLXFast.metalKernel(
            name: "gemma4_fused_decode_embed_d\(dims)",
            inputNames: [
                "token_ids", "packed_w", "w_scales", "w_biases", "embed_scale",
            ],
            outputNames: ["output"],
            source: """
                constexpr uint kDims = \(dims);
                constexpr uint kGroups = \(groups);
                constexpr uint kWords = \(words);

                const uint byte_index = thread_position_in_grid.x;
                const uint row = thread_position_in_grid.y;
                const int64_t token = static_cast<int64_t>(token_ids[row]);
                const device uint8_t* w_row =
                    reinterpret_cast<const device uint8_t*>(packed_w)
                    + token * (kWords * 4);
                const uint val = w_row[byte_index];
                const uint oindex = byte_index * 2;
                const uint gindex = oindex / 64;
                const bfloat scale = w_scales[token * kGroups + gindex];
                const bfloat bias = w_biases[token * kGroups + gindex];
                device bfloat* out_row = output + row * kDims;

                const uint8_t d0 = val & 0x0f;
                out_row[oindex] = (scale * d0 + bias) * embed_scale;
                const uint8_t d1 = (val >> 4) & 0x0f;
                out_row[oindex + 1] = (scale * d1 + bias) * embed_scale;
                """,
            header: "using namespace metal;",
            ensureRowContiguous: true
        )
        eval(self.embedScale)
    }

    /// `inputs` must be int32 token ids with shape `[1, L]`; the decode call
    /// site only routes `[1, 1]` here. Returns `inputs.shape + [dims]` bf16,
    /// raw-bit identical to `embedTokens(inputs) * embedScale`.
    func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        precondition(inputs.dtype == .int32 && inputs.ndim == 2)
        let rows = inputs.size
        let outputs = kernel(
            [inputs, weight, scales, biases, embedScale],
            grid: (dims / 2, rows, 1),
            threadGroup: (min(dims / 2, 256), 1, 1),
            outputShapes: [[rows, dims]],
            outputDTypes: [.bfloat16]
        )
        return outputs[0].reshaped(inputs.shape + [-1])
    }
}
