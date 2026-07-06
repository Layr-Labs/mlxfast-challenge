import MLX

/// A dense linear weight, either quantized (affine 4-bit, U32 codes + BF16
/// scales/biases -- matching the Gemma 4 4-bit checkpoint) or plain.
public struct Gemma4LinearWeight {
    public let weight: MLXArray
    public let scales: MLXArray?
    public let biases: MLXArray?
    public let logicalShape: [Int]
    public let groupSize: Int
    public let bits: Int

    public init(_ weight: MLXArray) {
        self.init(
            weight: weight,
            scales: nil,
            biases: nil,
            logicalShape: weight.shape,
            groupSize: 0,
            bits: 0
        )
    }

    public init(
        weight: MLXArray,
        scales: MLXArray?,
        biases: MLXArray?,
        logicalShape: [Int],
        groupSize: Int,
        bits: Int
    ) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.logicalShape = logicalShape
        self.groupSize = groupSize
        self.bits = bits
    }

    public var isQuantized: Bool {
        scales != nil
    }

    public var shape: [Int] {
        logicalShape
    }
}
