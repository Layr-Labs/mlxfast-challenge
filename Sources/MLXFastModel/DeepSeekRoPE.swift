import Foundation
import MLX
import MLXFastCore

public final class DeepSeekRoPE {
    public let rotaryDimensions: Int
    public let base: Double
    public let maxPositionEmbeddings: Int
    public let freqScale: Int

    private let baseFrequencies: [Float]
    /// Lock-guarded memo: instances are shared process-wide through
    /// DeepSeekRoPECache, and parallel unit tests exercise the same instance
    /// from multiple threads. A bare dictionary here corrupts under that
    /// concurrent mutation; the model runtime itself stays single-threaded,
    /// so the uncontended lock costs nothing on the hot path.
    private let frequencyCache = LockedCache<FrequencyCacheKey, MLXArray>()

    public init(
        rotaryDimensions: Int,
        base: Double,
        scaling: DeepSeekRopeScaling? = nil,
        maxPositionEmbeddings: Int,
        freqScale: Int = 1
    ) throws {
        guard rotaryDimensions > 0, rotaryDimensions % 2 == 0 else {
            throw MLXFastError.invalidInput("DeepSeek RoPE dimensions must be positive and even")
        }
        guard base > 0 else {
            throw MLXFastError.invalidInput("DeepSeek RoPE base must be positive")
        }
        guard freqScale > 0 else {
            throw MLXFastError.invalidInput("DeepSeek RoPE freqScale must be positive")
        }

        self.rotaryDimensions = rotaryDimensions
        self.base = base
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.freqScale = freqScale
        self.baseFrequencies = try Self.makeBaseFrequencies(
            rotaryDimensions: rotaryDimensions,
            base: base,
            scaling: scaling
        )
    }

    public convenience init(
        config: DeepSeekConfig,
        base: Double? = nil,
        scaling: DeepSeekRopeScaling? = nil,
        freqScale: Int = 1
    ) throws {
        try self.init(
            rotaryDimensions: config.qkRopeHeadDim,
            base: base ?? config.ropeTheta,
            scaling: scaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            freqScale: freqScale
        )
    }

    public static func makeBaseFrequencies(
        rotaryDimensions: Int,
        base: Double,
        scaling: DeepSeekRopeScaling? = nil
    ) throws -> [Float] {
        guard rotaryDimensions > 0, rotaryDimensions % 2 == 0 else {
            throw MLXFastError.invalidInput("DeepSeek RoPE dimensions must be positive and even")
        }
        guard base > 0 else {
            throw MLXFastError.invalidInput("DeepSeek RoPE base must be positive")
        }

        let halfDimensions = rotaryDimensions / 2
        var inverseFrequencies = (0..<halfDimensions).map { index in
            1.0 / pow(base, Double(index) / Double(halfDimensions))
        }

        let ropeType = scaling?.type ?? scaling?.ropeType
        switch ropeType {
        case nil, "default":
            break
        case "yarn", "deepseek_yarn":
            guard let factor = scaling?.factor, factor > 0 else {
                throw MLXFastError.invalidInput("DeepSeek YaRN RoPE requires a positive rope_scaling.factor")
            }
            guard let originalMaxPositionEmbeddings = scaling?.originalMaxPositionEmbeddings,
                  originalMaxPositionEmbeddings > 0
            else {
                throw MLXFastError.invalidInput(
                    "DeepSeek YaRN RoPE requires rope_scaling.original_max_position_embeddings"
                )
            }

            let betaFast = scaling?.betaFast ?? 32.0
            let betaSlow = scaling?.betaSlow ?? 1.0

            func correctionDimension(_ rotations: Double) -> Double {
                Double(rotaryDimensions)
                    * log(Double(originalMaxPositionEmbeddings) / (rotations * 2.0 * Double.pi))
                    / (2.0 * log(base))
            }

            let low = max(floor(correctionDimension(betaFast)), 0.0)
            var high = min(ceil(correctionDimension(betaSlow)), Double(rotaryDimensions - 1))
            if low == high {
                high += 0.001
            }

            for index in inverseFrequencies.indices {
                let ramp = (Double(index) - low) / (high - low)
                let smooth = 1.0 - min(max(ramp, 0.0), 1.0)
                let inverse = inverseFrequencies[index]
                inverseFrequencies[index] = inverse / factor * (1.0 - smooth) + inverse * smooth
            }
        case let unsupported?:
            throw MLXFastError.invalidInput("Unsupported DeepSeek RoPE type: \(unsupported)")
        }

        return inverseFrequencies.map { Float(1.0 / $0) }
    }

    public func frequencies(headDimension: Int, inverse: Bool = false) throws -> [Float] {
        guard headDimension > 0, headDimension % 2 == 0 else {
            throw MLXFastError.invalidInput("DeepSeek RoPE head dimension must be positive and even")
        }
        guard headDimension >= rotaryDimensions else {
            throw MLXFastError.invalidInput(
                "DeepSeek RoPE head dimension \(headDimension) is smaller than rotary dimension \(rotaryDimensions)"
            )
        }

        let noPositionPairs = (headDimension - rotaryDimensions) / 2
        var frequencies = Array(repeating: Float.infinity, count: noPositionPairs) + baseFrequencies
        if freqScale != 1 {
            let scale = Float(freqScale)
            frequencies = frequencies.map { $0 / scale }
        }
        if inverse {
            frequencies = frequencies.map { -$0 }
        }
        return frequencies
    }

    public func applied(to x: MLXArray, offset: Int = 0, inverse: Bool = false) throws -> MLXArray {
        let headDimension = try lastDimension(of: x)
        let freqs = try frequencyArray(headDimension: headDimension, inverse: inverse)
        let adjustedOffset = freqScale == 1 ? offset : offset / freqScale
        return MLXFast.RoPE(
            x,
            dimensions: headDimension,
            traditional: true,
            base: nil,
            scale: 1.0,
            offset: adjustedOffset,
            freqs: freqs
        )
    }

    public func applied(to x: MLXArray, offset: MLXArray, inverse: Bool = false) throws -> MLXArray {
        let headDimension = try lastDimension(of: x)
        let freqs = try frequencyArray(headDimension: headDimension, inverse: inverse)
        let adjustedOffset = freqScale == 1 ? offset : offset.floorDivide(MLXArray(Int32(freqScale)))
        return MLXFast.RoPE(
            x,
            dimensions: headDimension,
            traditional: true,
            base: nil,
            scale: 1.0,
            offset: adjustedOffset,
            freqs: freqs
        )
    }

    private func lastDimension(of array: MLXArray) throws -> Int {
        guard let headDimension = array.shape.last else {
            throw MLXFastError.invalidInput("DeepSeek RoPE input must have at least one dimension")
        }
        return headDimension
    }

    private func frequencyArray(headDimension: Int, inverse: Bool) throws -> MLXArray {
        let key = FrequencyCacheKey(headDimension: headDimension, inverse: inverse)
        return try frequencyCache.value(for: key) {
            let values = try frequencies(headDimension: headDimension, inverse: inverse)
            return MLXArray(values, [values.count])
        }
    }
}

private struct FrequencyCacheKey: Hashable {
    let headDimension: Int
    let inverse: Bool
}
