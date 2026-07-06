import Foundation
import MLX
import MLXFastCore

/// Rotary position embedding for a single Gemma 4 attention layer type.
///
/// Sliding-window layers use a standard, full-head-dimension rotation
/// (`rope_type: "default"`). Full-attention layers use a "proportional"
/// partial rotation: frequencies are computed as if rotating the entire head
/// dimension, but only the first `partialRotaryFactor` fraction of each half
/// is actually rotated (ported from the mlx-vlm `ProportionalRoPE`
/// reference implementation for `gemma4`).
public final class Gemma4RoPE {
    private enum Kind {
        case standard(dims: Int, base: Double)
        case proportional(dims: Int, rotatedDims: Int, freqs: MLXArray)
    }

    private let kind: Kind

    public init(dims: Int, spec: Gemma4RopeSpec) throws {
        guard dims > 0, dims % 2 == 0 else {
            throw MLXFastError.invalidInput("Gemma 4 RoPE head dimension must be positive and even")
        }
        guard spec.theta > 0 else {
            throw MLXFastError.invalidInput("Gemma 4 RoPE base must be positive")
        }

        switch spec.type {
        case "default", "":
            self.kind = .standard(dims: dims, base: spec.theta)
        case "proportional":
            let scaled = spec.partialRotaryFactor * Double(dims)
            let rotaryPairs = Int(scaled.rounded(.down)) / 2
            let rotatedDims = 2 * rotaryPairs
            guard rotatedDims > 0, rotatedDims <= dims else {
                throw MLXFastError.invalidInput(
                    "Gemma 4 proportional RoPE rotated dimension \(rotatedDims) is invalid for head dimension \(dims)"
                )
            }
            var freqValues = [Float]()
            freqValues.reserveCapacity(rotaryPairs)
            for index in 0..<rotaryPairs {
                let exponent = Double(2 * index) / Double(dims)
                freqValues.append(Float(pow(spec.theta, exponent)))
            }
            self.kind = .proportional(
                dims: dims,
                rotatedDims: rotatedDims,
                freqs: MLXArray(freqValues)
            )
        default:
            throw MLXFastError.invalidInput("unsupported Gemma 4 RoPE type \(spec.type)")
        }
    }

    public func applied(to x: MLXArray, offset: Int) -> MLXArray {
        switch kind {
        case let .standard(dims, base):
            return MLXFast.RoPE(
                x,
                dimensions: dims,
                traditional: false,
                base: Float(base),
                scale: 1.0,
                offset: offset
            )
        case let .proportional(dims, rotatedDims, freqs):
            return Self.proportionalApply(
                x,
                dims: dims,
                rotatedDims: rotatedDims,
                freqs: freqs,
                offset: offset
            )
        }
    }

    private static func proportionalApply(
        _ x: MLXArray,
        dims: Int,
        rotatedDims: Int,
        freqs: MLXArray,
        offset: Int
    ) -> MLXArray {
        let half = dims / 2
        let rotaryPairs = rotatedDims / 2

        let left = x[.ellipsis, 0..<half]
        let right = x[.ellipsis, half..<dims]

        let rotatedInput = concatenated(
            [left[.ellipsis, 0..<rotaryPairs], right[.ellipsis, 0..<rotaryPairs]],
            axis: -1
        )
        let roped = MLXFast.RoPE(
            rotatedInput,
            dimensions: rotatedDims,
            traditional: false,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: freqs
        )

        let newLeft = concatenated(
            [roped[.ellipsis, 0..<rotaryPairs], left[.ellipsis, rotaryPairs..<half]],
            axis: -1
        )
        let newRight = concatenated(
            [roped[.ellipsis, rotaryPairs..<rotatedDims], right[.ellipsis, rotaryPairs..<half]],
            axis: -1
        )
        return concatenated([newLeft, newRight], axis: -1)
    }
}

/// Process-wide cache of `Gemma4RoPE` instances keyed by their construction
/// parameters, mirroring the rationale for caching elsewhere in this
/// package: constructing one per layer per forward would recompute and
/// re-upload the (tiny) frequency array on every step.
public enum Gemma4RoPECache {
    private struct Key: Hashable {
        let dims: Int
        let theta: Double
        let type: String
        let partialRotaryFactor: Double
    }

    private static let cache = LockedCache<Key, Gemma4RoPE>()

    public static func shared(dims: Int, spec: Gemma4RopeSpec) throws -> Gemma4RoPE {
        let key = Key(
            dims: dims,
            theta: spec.theta,
            type: spec.type,
            partialRotaryFactor: spec.partialRotaryFactor
        )
        return try cache.value(for: key) {
            try Gemma4RoPE(dims: dims, spec: spec)
        }
    }
}
