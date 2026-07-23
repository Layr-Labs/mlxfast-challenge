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
                    values: ["--weights"]
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
                    values: ["--weights"]
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

            Participant-side MLX runtime worker for mlxfast-swift.
            """
        )
    }
}

private struct WorkerOptions {
    private let values: [String: String]

    init(_ arguments: [String]) throws {
        var values: [String: String] = [:]
        var index = 0
        while index < arguments.count {
            let option = arguments[index]
            guard option.hasPrefix("--") else {
                throw MLXFastError.invalidInput(
                    "unexpected participant worker argument '\(option)'"
                )
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
    }

    func value(for option: String, default defaultValue: String = "") -> String {
        values[option] ?? defaultValue
    }

    func requireOnly(values allowedValues: Set<String>) throws {
        if let unexpected = Set(values.keys).subtracting(allowedValues).sorted().first {
            throw MLXFastError.invalidInput(
                "unexpected participant worker option \(unexpected)"
            )
        }
    }
}
