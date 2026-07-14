import MLX
import MLXFastCore

/// Availability of a trained Gemma 4 assistant for speculative block
/// generation. The shipped challenge checkpoint contains only the base target
/// text tower, so the first experimental track intentionally exposes this as
/// unavailable instead of silently binding an assistant trained for an IT
/// target.
public struct Gemma4MTPAssistantAvailability: Equatable, Sendable {
    public let isAvailable: Bool
    public let reason: String

    public init(isAvailable: Bool, reason: String) {
        self.isAvailable = isAvailable
        self.reason = reason
    }

    public static let shippedCheckpoint = Gemma4MTPAssistantAvailability(
        isAvailable: false,
        reason:
            "assistant weights unavailable/incompatible: the shipped Gemma 4 31B base checkpoint "
            + "contains no drafter weights, and the known public Gemma 4 assistant targets are IT models"
    )

    public func requireAvailable() throws {
        guard isAvailable else {
            throw MLXFastError.invalidInput(reason)
        }
    }
}

/// Model-side replacement point for an experimental target-confirmed block
/// generator.
///
/// Implementations may return fewer tokens than requested, but must return a
/// nonempty block and leave `cache` advanced by exactly the returned token
/// count. A future speculative implementation must verify every emitted token
/// with the target and explicitly roll back rejected KV state before returning.
public protocol Gemma4TargetBlockGenerating {
    var implementationName: String { get }

    func generateBlock(
        previousToken: Int,
        maxBlockSize: Int,
        cache: Gemma4ModelCache,
        positionOffset: Int,
        weightCache: Gemma4RuntimeWeightCache
    ) throws -> [Int]
}

/// Pure serial autoregressive block semantics shared by the runtime
/// implementation and lightweight tests.
public enum Gemma4TargetBlockGeneration {
    public static func generateSerialBlock(
        previousToken: Int,
        maxBlockSize: Int,
        positionOffset: Int,
        nextToken: (_ inputToken: Int, _ positionOffset: Int) throws -> Int
    ) throws -> [Int] {
        guard previousToken >= 0, previousToken < MLXFastConstants.vocabSize else {
            throw MLXFastError.invalidInput(
                "experimental block input token is outside the Gemma 4 vocabulary"
            )
        }
        guard maxBlockSize > 0,
              maxBlockSize <= MLXFastConstants.experimentalMTPMaxBlockSize
        else {
            throw MLXFastError.invalidInput(
                "experimental block size must be in 1...\(MLXFastConstants.experimentalMTPMaxBlockSize)"
            )
        }
        guard positionOffset >= 0 else {
            throw MLXFastError.invalidInput(
                "experimental block position offset must be non-negative"
            )
        }

        var inputToken = previousToken
        var generated: [Int] = []
        generated.reserveCapacity(maxBlockSize)
        for index in 0..<maxBlockSize {
            let (stepOffset, overflow) = positionOffset.addingReportingOverflow(index)
            guard !overflow else {
                throw MLXFastError.invalidInput(
                    "experimental block position offset overflows Int"
                )
            }
            let token = try nextToken(inputToken, stepOffset)
            guard token >= 0, token < MLXFastConstants.vocabSize else {
                throw MLXFastError.invalidInput(
                    "experimental block generator returned a token outside the Gemma 4 vocabulary"
                )
            }
            generated.append(token)
            inputToken = token
        }
        return generated
    }
}

/// Target-only fallback used by the first experimental block track.
///
/// This performs one ordinary target forward per returned token. It proves the
/// block protocol and cache contract but is not MTP and is not expected to
/// improve throughput.
public struct Gemma4SerialTargetBlockGenerator: Gemma4TargetBlockGenerating {
    public let implementationName = "serial_target_fallback"

    public init() {}

    public func generateBlock(
        previousToken: Int,
        maxBlockSize: Int,
        cache: Gemma4ModelCache,
        positionOffset: Int,
        weightCache: Gemma4RuntimeWeightCache
    ) throws -> [Int] {
        try Gemma4TargetBlockGeneration.generateSerialBlock(
            previousToken: previousToken,
            maxBlockSize: maxBlockSize,
            positionOffset: positionOffset
        ) { inputToken, stepOffset in
            let logits = try Gemma4Model.logits(
                inputIDs: MLXArray([Int32(inputToken)], [1, 1]),
                weightCache: weightCache,
                cache: cache,
                positionOffset: stepOffset
            )
            return try greedyToken(from: logits)
        }
    }

    private func greedyToken(from logits: MLXArray) throws -> Int {
        guard let vocabularySize = logits.shape.last, vocabularySize > 0 else {
            throw MLXFastError.invalidInput(
                "experimental block logits must have a non-empty vocabulary dimension"
            )
        }
        let rows = logits.reshaped([-1, vocabularySize])
        return Int(rows[-1].argMax().item(Int32.self))
    }
}
