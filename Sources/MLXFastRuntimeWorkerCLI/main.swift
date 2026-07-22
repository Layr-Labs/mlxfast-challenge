import Darwin
import Foundation
import MLXFastCore
import MLXFastModel
import MLXFastRuntimeWorkerSupport

let exitCode = ParticipantWorkerCLI.run(
    arguments: Array(CommandLine.arguments.dropFirst())
)
exit(Int32(exitCode))

private enum ParticipantWorkerCLI {
    static func run(arguments: [String]) -> Int {
        do {
            guard let command = arguments.first else {
                printUsage()
                return 0
            }
            if ["help", "--help", "-h"].contains(command) {
                printUsage()
                return 0
            }
            if Array(arguments.dropFirst()) == ["--help"]
                || Array(arguments.dropFirst()) == ["-h"]
            {
                printUsage()
                return 0
            }
            let options = try WorkerOptions(
                Array(arguments.dropFirst())
            )
            switch command {
            case "runtime-worker":
                try options.requireOnly(
                    values: ["--weights"],
                    flags: []
                )
                let weightsPath = options.value(
                    for: "--weights",
                    default: ProcessInfo.processInfo.environment[
                        "MLXFAST_WEIGHTS_PATH"
                    ] ?? MLXFastConstants.defaultWeightsPath
                )
                try LagunaRuntime.runWorker(weightsPath: weightsPath)

            case "preflight":
                try options.requireOnly(
                    values: ["--weights"],
                    flags: []
                )
                let weightsPath = options.value(
                    for: "--weights",
                    default: ProcessInfo.processInfo.environment[
                        "MLXFAST_WEIGHTS_PATH"
                    ] ?? MLXFastConstants.defaultWeightsPath
                )
                try LagunaRuntime.runPreflightWorker(
                    weightsPath: weightsPath
                )

            case "mtp-runtime-worker":
                try options.requireOnly(
                    values: [
                        "--weights",
                        "--assistant",
                        "--contract",
                        "--target-verification",
                    ],
                    flags: ["--require-trained-assistant"]
                )
                let weightsPath = options.value(for: "--weights")
                let assistantPath = options.value(for: "--assistant")
                let contractPath = options.value(for: "--contract")
                guard !weightsPath.isEmpty,
                      !assistantPath.isEmpty,
                      !contractPath.isEmpty
                else {
                    throw MLXFastError.invalidInput(
                        "mtp-runtime-worker requires weights, assistant, and contract paths"
                    )
                }
                let verificationValue = options.value(
                    for: "--target-verification",
                    default: MTPVerificationMode.exactPair.rawValue
                ).lowercased()
                guard let verificationMode = MTPVerificationMode(
                    rawValue: verificationValue
                ) else {
                    throw MLXFastError.invalidInput(
                        "--target-verification must be exact-pair or serial"
                    )
                }
                try LagunaRuntime.runExperimentalMTPWorker(
                    targetWeightsPath: weightsPath,
                    assistantPath: assistantPath,
                    contractPath: contractPath,
                    verificationMode: verificationMode,
                    requireTrainedAssistant: options.hasFlag(
                        "--require-trained-assistant"
                    )
                )

            default:
                throw MLXFastError.invalidInput(
                    "unknown participant worker command '\(command)'"
                )
            }
            return 0
        } catch {
            fputs("mlxfast-runtime-worker: \(error)\n", stderr)
            return 1
        }
    }

    private static func printUsage() {
        print(
            """
            Usage:
              mlxfast-runtime-worker runtime-worker [--weights PATH]
              mlxfast-runtime-worker preflight [--weights PATH]
              mlxfast-runtime-worker mtp-runtime-worker --weights PATH --assistant PATH --contract PATH [--target-verification exact-pair|serial] [--require-trained-assistant]

            Participant-side MLX runtime worker for mlxfast-swift.
            """
        )
    }
}

private struct WorkerOptions {
    private let values: [String: String]
    private let flags: Set<String>

    init(_ arguments: [String]) throws {
        var values: [String: String] = [:]
        var flags = Set<String>()
        var index = 0
        while index < arguments.count {
            let option = arguments[index]
            guard option.hasPrefix("--") else {
                throw MLXFastError.invalidInput(
                    "unexpected participant worker argument '\(option)'"
                )
            }
            if option == "--require-trained-assistant" {
                guard flags.insert(option).inserted else {
                    throw MLXFastError.invalidInput(
                        "duplicate participant worker option \(option)"
                    )
                }
                index += 1
                continue
            }
            guard values[option] == nil else {
                throw MLXFastError.invalidInput(
                    "duplicate participant worker option \(option)"
                )
            }
            guard index + 1 < arguments.count else {
                throw MLXFastError.invalidInput(
                    "participant worker option \(option) requires a value"
                )
            }
            values[option] = arguments[index + 1]
            index += 2
        }
        self.values = values
        self.flags = flags
    }

    func value(for option: String, default defaultValue: String = "") -> String {
        values[option] ?? defaultValue
    }

    func hasFlag(_ option: String) -> Bool {
        flags.contains(option)
    }

    func requireOnly(values allowedValues: Set<String>, flags allowedFlags: Set<String>) throws {
        let unexpectedValues = Set(values.keys).subtracting(allowedValues)
        let unexpectedFlags = flags.subtracting(allowedFlags)
        if let unexpected = unexpectedValues.union(unexpectedFlags).sorted().first {
            throw MLXFastError.invalidInput(
                "unexpected participant worker option \(unexpected)"
            )
        }
    }
}
