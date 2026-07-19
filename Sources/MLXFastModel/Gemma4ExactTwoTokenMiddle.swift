import Foundation
import MLX

private let gemma4MTPExactFourAttentionEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_MTP_EXACT_FOUR_ATTENTION"
    ] else {
        return true
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// Runs both sliding-attention queries in one causal vector-attention dispatch
/// while both rows use the one-pass MLX kernel. Its causal branch skips the
/// draft key for token zero, preserving the promoted one-query arithmetic and
/// reduction order. Full D=512 attention stays serialized because its fallback
/// matmuls can change reduction order when the key length changes. The final
/// sliding pre-wrap pair also stays serialized because MLX switches D=256 to
/// its two-pass kernel at 1,024 keys while token zero's reference does not.
func gemma4ExactTwoTokenAttention(
    queries: MLXArray,
    keysBeforeDraft: MLXArray,
    valuesBeforeDraft: MLXArray,
    keysWithDraft: MLXArray,
    valuesWithDraft: MLXArray,
    scale: Float
) -> MLXArray {
    precondition(queries.ndim == 4 && queries.dim(0) == 1 && queries.dim(2) == 2)
    precondition(keysBeforeDraft.ndim == 4 && valuesBeforeDraft.ndim == 4)
    precondition(keysWithDraft.ndim == 4 && valuesWithDraft.ndim == 4)
    precondition(keysBeforeDraft.shape == valuesBeforeDraft.shape)
    precondition(keysWithDraft.shape == valuesWithDraft.shape)
    precondition(keysBeforeDraft.dim(0) == 1 && keysWithDraft.dim(0) == 1)
    precondition(keysBeforeDraft.dim(1) == keysWithDraft.dim(1))
    precondition(keysBeforeDraft.dim(3) == queries.dim(3))
    precondition(keysWithDraft.dim(3) == queries.dim(3))
    precondition(keysWithDraft.dim(2) == keysBeforeDraft.dim(2) + 1)

    if queries.dim(3) == 256 && keysWithDraft.dim(2) < 1_024 {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keysWithDraft,
            values: valuesWithDraft,
            scale: scale,
            mask: .causal
        )
    }

    let attention0 = MLXFast.scaledDotProductAttention(
        queries: queries[0..., 0..., 0..<1, 0...],
        keys: keysBeforeDraft,
        values: valuesBeforeDraft,
        scale: scale,
        mask: .none
    )
    let attention1 = MLXFast.scaledDotProductAttention(
        queries: queries[0..., 0..., 1..<2, 0...],
        keys: keysWithDraft,
        values: valuesWithDraft,
        scale: scale,
        mask: .none
    )
    return concatenated([attention0, attention1], axis: 2)
}

/// Four serial-causal attention rows. Sliding D=256 uses one native causal
/// vector dispatch while it remains on the one-pass kernel; other geometries
/// compose the proven exact-two primitive to preserve established arithmetic.
func gemma4ExactFourTokenAttention(
    queries: MLXArray,
    keysThroughRow0: MLXArray,
    valuesThroughRow0: MLXArray,
    keysThroughRow1: MLXArray,
    valuesThroughRow1: MLXArray,
    keysThroughRow2: MLXArray,
    valuesThroughRow2: MLXArray,
    keysThroughRow3: MLXArray,
    valuesThroughRow3: MLXArray,
    scale: Float
) -> MLXArray {
    precondition(queries.ndim == 4 && queries.dim(2) == 4)
    precondition(keysThroughRow0.shape == valuesThroughRow0.shape)
    precondition(keysThroughRow1.shape == valuesThroughRow1.shape)
    precondition(keysThroughRow2.shape == valuesThroughRow2.shape)
    precondition(keysThroughRow3.shape == valuesThroughRow3.shape)
    precondition(keysThroughRow1.dim(2) == keysThroughRow0.dim(2) + 1)
    precondition(keysThroughRow2.dim(2) == keysThroughRow1.dim(2) + 1)
    precondition(keysThroughRow3.dim(2) == keysThroughRow2.dim(2) + 1)

    // The D=256 vector kernel evaluates each query row independently. One
    // four-query causal launch therefore preserves the serial row arithmetic
    // while eliminating the second exact-two dispatch and concatenation. Stay
    // below the 1,024-key switch where MLX changes to its two-pass algorithm.
    if gemma4MTPExactFourAttentionEnabled,
       queries.dim(3) == 256,
       keysThroughRow3.dim(2) < 1_024
    {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keysThroughRow3,
            values: valuesThroughRow3,
            scale: scale,
            mask: .causal
        )
    }

    let first = gemma4ExactTwoTokenAttention(
        queries: queries[0..., 0..., 0..<2, 0...],
        keysBeforeDraft: keysThroughRow0,
        valuesBeforeDraft: valuesThroughRow0,
        keysWithDraft: keysThroughRow1,
        valuesWithDraft: valuesThroughRow1,
        scale: scale
    )
    let second = gemma4ExactTwoTokenAttention(
        queries: queries[0..., 0..., 2..<4, 0...],
        keysBeforeDraft: keysThroughRow2,
        valuesBeforeDraft: valuesThroughRow2,
        keysWithDraft: keysThroughRow3,
        valuesWithDraft: valuesThroughRow3,
        scale: scale
    )
    return concatenated([first, second], axis: 2)
}
