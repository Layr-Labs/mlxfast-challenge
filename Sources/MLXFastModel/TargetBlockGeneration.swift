import MLXFastCore

/// Availability of a trained assistant for speculative block generation.
/// The serial challenge checkpoint contains only the target text tower, so the
/// experimental probe exposes this as unavailable instead of silently binding
/// separately provisioned assistant weights.
public struct MTPAssistantAvailability: Equatable, Sendable {
    public let isAvailable: Bool
    public let reason: String

    public init(isAvailable: Bool, reason: String) {
        self.isAvailable = isAvailable
        self.reason = reason
    }

    public static let shippedCheckpoint = MTPAssistantAvailability(
        isAvailable: false,
        reason:
            "assistant weights unavailable/incompatible: the shipped base checkpoint "
            + "contains no drafter weights"
    )

    public func requireAvailable() throws {
        guard isAvailable else {
            throw MLXFastError.invalidInput(reason)
        }
    }
}

/// Pure serial autoregressive block semantics shared by the runtime
/// implementation and lightweight tests.
public enum TargetBlockGeneration {
    public static func generateSerialBlock(
        previousToken: Int,
        maxBlockSize: Int,
        positionOffset: Int,
        nextToken: (_ inputToken: Int, _ positionOffset: Int) throws -> Int
    ) throws -> [Int] {
        guard previousToken >= 0, previousToken < MLXFastConstants.vocabSize else {
            throw MLXFastError.invalidInput(
                "experimental block input token is outside the model vocabulary"
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
                    "experimental block generator returned a token outside the model vocabulary"
                )
            }
            generated.append(token)
            inputToken = token
        }
        return generated
    }
}
