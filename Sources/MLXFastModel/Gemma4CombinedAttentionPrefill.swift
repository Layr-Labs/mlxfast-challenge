import MLX

private func supportsCombinedAttentionProjection(
    _ projection: FastQuantizedProjection,
    inputWidth: Int?,
    metadataWidth: Int?
) -> Bool {
    guard let biases = projection.biases,
          projection.groupSize == 64,
          projection.bits == 4,
          projection.weight.dtype == .uint32,
          projection.weight.ndim == 2,
          projection.scales.dtype == .bfloat16,
          projection.scales.ndim == 2,
          biases.dtype == .bfloat16,
          biases.shape == projection.scales.shape,
          projection.weight.dim(0) == projection.scales.dim(0),
          projection.weight.dim(1) * 8 == projection.scales.dim(1) * projection.groupSize
    else {
        return false
    }
    return inputWidth.map { projection.weight.dim(1) == $0 } ?? true
        && metadataWidth.map { projection.scales.dim(1) == $0 } ?? true
}

/// A prefill-only affine projection whose output rows are Q, K, then optional
/// V. The source projections stay alive on `Gemma4FastLayer` for decode and
/// rollback; these input-independent concatenations are materialized once in
/// untimed model construction.
struct CombinedAttentionPrefillProjection: @unchecked Sendable {
    private let combined: FastQuantizedProjection
    private let q: FastQuantizedProjection
    private let k: FastQuantizedProjection
    private let v: FastQuantizedProjection?
    private let qWidth: Int
    private let kWidth: Int
    private let verifyBits: Bool

    init?(
        q: FastQuantizedProjection,
        k: FastQuantizedProjection,
        v: FastQuantizedProjection?,
        verifyBits: Bool
    ) {
        guard supportsCombinedAttentionProjection(
                  q, inputWidth: nil, metadataWidth: nil),
              supportsCombinedAttentionProjection(
                  k,
                  inputWidth: q.weight.dim(1),
                  metadataWidth: q.scales.dim(1)),
              k.groupSize == q.groupSize,
              k.bits == q.bits,
              v.map({
                  supportsCombinedAttentionProjection(
                      $0,
                      inputWidth: q.weight.dim(1),
                      metadataWidth: q.scales.dim(1))
                      && $0.groupSize == q.groupSize
                      && $0.bits == q.bits
                      && $0.weight.dim(0) == k.weight.dim(0)
              }) ?? true,
              let qBiases = q.biases,
              let kBiases = k.biases
        else {
            return nil
        }

        var weights = [q.weight, k.weight]
        var scales = [q.scales, k.scales]
        var biases = [qBiases, kBiases]
        if let v, let vBiases = v.biases {
            weights.append(v.weight)
            scales.append(v.scales)
            biases.append(vBiases)
        }

        let combinedWeight = concatenated(weights, axis: 0)
        let combinedScales = concatenated(scales, axis: 0)
        let combinedBiases = concatenated(biases, axis: 0)
        eval(combinedWeight, combinedScales, combinedBiases)

        self.combined = FastQuantizedProjection(
            weight: combinedWeight,
            scales: combinedScales,
            biases: combinedBiases,
            groupSize: q.groupSize,
            bits: q.bits
        )
        self.q = q
        self.k = k
        self.v = v
        self.qWidth = q.weight.dim(0)
        self.kWidth = k.weight.dim(0)
        self.verifyBits = verifyBits
    }

    func callAsFunction(
        _ input: MLXArray
    ) -> (queries: MLXArray, keys: MLXArray, values: MLXArray?) {
        precondition(input.dtype == .bfloat16)
        precondition(input.ndim == 3 && input.dim(0) == 1 && input.dim(1) > 1)
        precondition(input.dim(2) == combined.weight.dim(1) * 8)

        let projected = combined(input)
        let qEnd = qWidth
        let kEnd = qEnd + kWidth
        let queries = projected[.ellipsis, 0..<qEnd]
        let keys = projected[.ellipsis, qEnd..<kEnd]
        let values = v.map { projection in
            projected[.ellipsis, kEnd..<(kEnd + projection.weight.dim(0))]
        }

        if verifyBits {
            var comparisons = [
                arrayEqual(
                    queries.view(dtype: .uint16),
                    q(input).view(dtype: .uint16)),
                arrayEqual(
                    keys.view(dtype: .uint16),
                    k(input).view(dtype: .uint16)),
            ]
            if let v, let values {
                comparisons.append(arrayEqual(
                    values.view(dtype: .uint16),
                    v(input).view(dtype: .uint16)))
            }
            eval(comparisons)
            precondition(
                comparisons.allSatisfy { $0.item(Bool.self) },
                "combined attention prefill raw projection differs from separate QMMs"
            )
        }

        return (queries, keys, values)
    }
}
