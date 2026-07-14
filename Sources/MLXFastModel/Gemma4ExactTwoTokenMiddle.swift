import MLX

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
