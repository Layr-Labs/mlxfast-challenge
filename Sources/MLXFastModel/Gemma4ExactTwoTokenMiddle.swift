import MLX

/// Runs the promoted one-query attention primitive independently for each
/// speculative row. Token zero excludes the draft K/V row; token one includes
/// it. This preserves the single-token reduction path instead of selecting a
/// numerically different batched attention implementation.
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

    let query0 = queries[0..., 0..., 0..<1, 0...]
    let query1 = queries[0..., 0..., 1..<2, 0...]
    let attention0 = MLXFast.scaledDotProductAttention(
        queries: query0,
        keys: keysBeforeDraft,
        values: valuesBeforeDraft,
        scale: scale,
        mask: .none
    )
    let attention1 = MLXFast.scaledDotProductAttention(
        queries: query1,
        keys: keysWithDraft,
        values: valuesWithDraft,
        scale: scale,
        mask: .none
    )
    return concatenated([attention0, attention1], axis: 2)
}
