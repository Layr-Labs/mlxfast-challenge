import Darwin
import Foundation
import MLXFastCore
import MLXFastHarness
import MLXFastTransform
import Tokenizers

let exitCode = MLXFastCLI.run(arguments: Array(CommandLine.arguments.dropFirst()))
exit(Int32(exitCode))

private enum MLXFastCLI {
    static func run(arguments: [String]) -> Int {
        guard let command = arguments.first, command != "help", command != "--help", command != "-h" else {
            printUsage()
            return 0
        }

        let options = ParsedOptions(Array(arguments.dropFirst()))

        do {
            switch command {
            case "transform":
                try runTransform(options)
                return 0
            case "verify-transform":
                try runVerifyTransform(options)
                return 0
            case "correctness":
                return try runCorrectness(options)
            case "correctness-trace":
                try runCorrectnessTrace(options)
                return 0
            case "preflight":
                try runPreflight(options)
                return 0
            case "benchmark":
                try runBenchmark(options)
                return 0
            case "attach-gpqa-gates":
                try runAttachGPQAGates(options)
                return 0
            case "attach-free-run-gate":
                try runAttachFreeRunGate(options)
                return 0
            case "generate-golden":
                try runGenerateGolden(options)
                return 0
            case "generate-gpqa-answers":
                try runGenerateGPQAAnswers(options)
                return 0
            case "runtime-worker":
                try runRuntimeWorker(options)
                return 0
            case "checkpoint-shards":
                try runCheckpointShards(options)
                return 0
            default:
                fputs("mlxfast-swift: unknown command '\(command)'\n\n", stderr)
                printUsage()
                return 2
            }
        } catch {
            fputs("mlxfast-swift: \(error)\n", stderr)
            return 1
        }
    }

    private static func runTransform(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--reference", "--output"])
        let referencePath = options.value(
            for: "--reference",
            default: environmentValue(
                "MLXFAST_REFERENCE_DIR",
                fallback: MLXFastConstants.defaultReferencePath
            )
        )
        let outputPath = options.value(
            for: "--output",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let report = try SwiftTransform.run(
            TransformOptions(referencePath: referencePath, outputPath: outputPath)
        )
        print("reference: \(report.referencePath)")
        print("output: \(report.outputPath)")
        print("dense tensors: \(report.denseTensorCount) across \(report.denseShardCount) shard(s)")
        print("config: \(report.configPath)")
        print("index: \(report.indexPath)")
    }

    private static func runVerifyTransform(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--reference", "--weights", "--tmp-parent", "--max-bytes"])
        let referencePath = options.value(
            for: "--reference",
            default: environmentValue(
                "MLXFAST_REFERENCE_DIR",
                fallback: MLXFastConstants.defaultReferencePath
            )
        )
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let temporaryParentPath = options.value(for: "--tmp-parent", default: "")
        let maxBytesRaw = options.value(
            for: "--max-bytes",
            default: environmentValue(
                "MLXFAST_MAX_WEIGHTS_BYTES",
                fallback: "\(MLXFastConstants.defaultMaxTransformedWeightsBytes)"
            )
        )
        let maxByteCount = try parseTransformedWeightsByteLimit(
            raw: maxBytesRaw,
            defaultByteCount: MLXFastConstants.defaultMaxTransformedWeightsBytes,
            optionLabel: "--max-bytes"
        )
        let report = try TransformVerifier.verify(
            TransformVerificationOptions(
                referencePath: referencePath,
                weightsPath: weightsPath,
                temporaryParentPath: temporaryParentPath.isEmpty ? nil : temporaryParentPath,
                maxByteCount: maxByteCount
            )
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(report)
        FileHandle.standardOutput.write(data)
        print("")
    }

    private static func runCorrectness(_ options: ParsedOptions) throws -> Int {
        try options.validate(valueOptions: ["--weights", "--golden"])
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let goldenPath = options.value(
            for: "--golden",
            default: environmentValue(
                "MLXFAST_CORRECTNESS_GOLDEN_PATH",
                fallback: defaultCorrectnessGoldenPath()
            )
        )
        let report = try GemmaRuntime.runCorrectness(
            CorrectnessOptions(
                weightsPath: weightsPath,
                goldenPath: goldenPath
            ),
            worker: try runtimeWorkerOptions(blockedGoldenPath: goldenPath)
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(report)
        FileHandle.standardOutput.write(data)
        print("")
        return report.passed ? 0 : 1
    }

    private static func runCorrectnessTrace(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--weights", "--golden", "--case", "--step", "--top-k"])
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let goldenPath = options.value(
            for: "--golden",
            default: environmentValue(
                "MLXFAST_CORRECTNESS_GOLDEN_PATH",
                fallback: defaultCorrectnessGoldenPath()
            )
        )
        let stepRaw = options.value(for: "--step", default: "")
        guard let step = Int(stepRaw), step >= 0 else {
            throw MLXFastError.invalidInput("correctness-trace requires --step N with N >= 0")
        }
        let topKRaw = options.value(for: "--top-k", default: "8")
        guard let topK = Int(topKRaw), topK > 0 else {
            throw MLXFastError.invalidInput("--top-k must be a positive integer")
        }
        let caseName = options.value(for: "--case", default: "")
        let report = try GemmaRuntime.traceCorrectness(
            CorrectnessTraceOptions(
                weightsPath: weightsPath,
                goldenPath: goldenPath,
                caseName: caseName.isEmpty ? nil : caseName,
                step: step,
                topK: topK
            )
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(report)
        FileHandle.standardOutput.write(data)
        print("")
    }

    private static func runPreflight(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--weights", "--golden"])
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let goldenPath = options.value(
            for: "--golden",
            default: environmentValue(
                "MLXFAST_CORRECTNESS_GOLDEN_PATH",
                fallback: MLXFastConstants.defaultGoldenPath
            )
        )
        let report = try BenchmarkPreflight.check(
            weightsPath: weightsPath,
            goldenPath: goldenPath
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(report)
        FileHandle.standardOutput.write(data)
        print("")
    }

    private static func runBenchmark(_ options: ParsedOptions) throws {
        try options.validate(
            valueOptions: ["--weights", "--golden", "--score-path"],
            flagOptions: ["--local-submit", "--local-iterate"]
        )
        let localSubmit = options.hasFlag("--local-submit")
        let localIterate = options.hasFlag("--local-iterate")
        guard !(localSubmit && localIterate) else {
            throw MLXFastError.invalidInput("--local-submit and --local-iterate cannot be used together")
        }
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let goldenPath = options.value(
            for: "--golden",
            default: environmentValue(
                "MLXFAST_CORRECTNESS_GOLDEN_PATH",
                fallback: localSubmit
                    ? MLXFastConstants.defaultPublicLocalSubmitGoldenPath
                    : localIterate
                        ? MLXFastConstants.defaultPublicCorrectnessGoldenPath
                        : MLXFastConstants.defaultGoldenPath
            )
        )
        let scorePath = options.value(
            for: "--score-path",
            default: environmentValue(
                "MLXFAST_SCORE_PATH",
                fallback: localIterate
                    ? MLXFastConstants.defaultLocalIterateScorePath
                    : MLXFastConstants.defaultScorePath
            )
        )
        if localSubmit || localIterate {
            let decodeSteps = localSubmit
                ? MLXFastConstants.localSubmitBenchmarkDecodeSteps
                : MLXFastConstants.localIterateBenchmarkDecodeSteps
            let timingRepeats = localSubmit ? MLXFastConstants.localSubmitBenchmarkRepeats : 1
            let modeName = localSubmit ? "local-submit" : "local-iterate"
            let runtime = localSubmit ? "swift-local-submit" : "swift-local-iterate"
            let payload = GemmaRuntime.localIterate(
                LocalIterateOptions(
                    weightsPath: weightsPath,
                    goldenPath: goldenPath,
                    benchmarkDecodeSteps: decodeSteps,
                    timingRepeats: timingRepeats,
                    modeName: modeName,
                    runtime: runtime
                ),
                // Local edit loop: stream the worker's stderr live so debug
                // prints in submitted model code are visible while iterating.
                // runtimeWorkerOptions forces this off on official runs.
                worker: try runtimeWorkerOptions(
                    blockedGoldenPath: goldenPath,
                    forwardsWorkerStderr: true
                )
            )
            try writeScorePayload(payload, to: scorePath)
            try emitScorePayloadToStdout(payload)
            return
        }
        let semanticOutputPath = environmentValue("MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH", fallback: "")
        let semanticCaseCount = try parsePositiveInt(
            environmentValue(
                "MLXFAST_SEMANTIC_GPQA_CASE_COUNT",
                fallback: "\(MLXFastConstants.semanticGPQACaseCount)"
            ),
            optionName: "MLXFAST_SEMANTIC_GPQA_CASE_COUNT"
        )
        let semanticMaxNewTokens = try parsePositiveInt(
            environmentValue(
                "MLXFAST_SEMANTIC_GPQA_MAX_NEW_TOKENS",
                fallback: "\(MLXFastConstants.semanticGPQAMaxNewTokens)"
            ),
            optionName: "MLXFAST_SEMANTIC_GPQA_MAX_NEW_TOKENS"
        )
        if !semanticOutputPath.isEmpty {
            try requirePrivateOutputPath(semanticOutputPath, description: "semantic GPQA answer output")
        }
        // Lets a run skip the base teacher-forced case (still runs behavior/GPQA/TTFT/
        // timing) when that case is verified in a separate phase of the ranked
        // pipeline. Defaults to the full official window. See the comment on
        // BenchmarkPreflight/validateBenchmarkOptions for why 0 is accepted: the
        // harness never treats a steps=0 run as self-certifying correctness; only the
        // trusted pipeline that assembles the final score may do that.
        let correctnessSteps = try parseNonNegativeInt(
            environmentValue(
                "MLXFAST_BENCHMARK_CORRECTNESS_STEPS",
                fallback: "\(MLXFastConstants.correctnessSteps)"
            ),
            optionName: "MLXFAST_BENCHMARK_CORRECTNESS_STEPS"
        )
        // Phase controls for the single-machine ranked pipeline: benchmark.yml's
        // gates pass runs with CHECK_GATES=1 SKIP_TIMED=1 (base case + hidden gates,
        // no timing), and the timed measurement runs later through measure-job's
        // own ./benchmark.sh --official invocation. Both default to the original
        // everything-in-one-run behavior.
        let checkGates = environmentValue("MLXFAST_BENCHMARK_CHECK_GATES", fallback: "1") != "0"
        let skipTimedBenchmark = environmentValue("MLXFAST_BENCHMARK_SKIP_TIMED", fallback: "0") == "1"
        let payload = GemmaRuntime.benchmark(
            BenchmarkOptions(
                weightsPath: weightsPath,
                goldenPath: goldenPath,
                correctnessSteps: correctnessSteps,
                semanticGPQAOutputPath: semanticOutputPath.isEmpty ? nil : semanticOutputPath,
                semanticGPQATokenizerPath: weightsPath,
                semanticGPQACaseCount: semanticCaseCount,
                semanticGPQAMaxNewTokens: semanticMaxNewTokens,
                checkGates: checkGates,
                skipTimedBenchmark: skipTimedBenchmark
            ),
            worker: try runtimeWorkerOptions(blockedGoldenPath: goldenPath)
        )
        try writeScorePayload(payload, to: scorePath)
        try emitScorePayloadToStdout(payload)
        fputs("wrote \(scorePath)\n", stderr)
    }

    // Emits the in-memory payload, not a re-read of scorePath: the benchmark
    // process links the editable submission modules and runs unsandboxed, so a
    // file it wrote to scorePath could be tampered with (e.g. via an atexit
    // handler) between the write above and this call. Serializing the value
    // already held in memory means stdout reflects exactly what this trusted
    // process computed, independent of anything written to disk afterward.
    // benchmark.sh captures this stdout, after the process has fully exited, as
    // the sole source of truth for score.json.
    private static func emitScorePayloadToStdout(_ payload: ScorePayload) throws {
        // benchmark.sh seals score.json from THIS stdout, so it -- not the
        // writeScorePayload file it discards -- is the published per-machine
        // artifact. Coarsen the diagnostic analog fields here too, or the
        // timing/memory covert-channel coarsening applied in writeScorePayload
        // (and the combined score) is bypassed on the sealed path.
        let publishedPayload = ScorePayload(
            score: payload.score,
            passed: payload.passed,
            metrics: payload.metrics.withCoarsenedPublicDiagnostics()
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(publishedPayload)
        FileHandle.standardOutput.write(data)
        if data.last != 0x0a { print("") }
    }

    private static func runAttachGPQAGates(_ options: ParsedOptions) throws {
        try options.validate(
            valueOptions: ["--golden", "--gpqa", "--tokenizer", "--output", "--case-count", "--max-new-tokens"]
        )
        let goldenPath = options.value(
            for: "--golden",
            default: environmentValue(
                "MLXFAST_CORRECTNESS_GOLDEN_PATH",
                fallback: MLXFastConstants.defaultGoldenPath
            )
        )
        let gpqaPath = options.value(
            for: "--gpqa",
            default: environmentValue("MLXFAST_GPQA_REFERENCE_PATH", fallback: "")
        )
        guard !gpqaPath.isEmpty else {
            throw MLXFastError.invalidInput("attach-gpqa-gates requires --gpqa or MLXFAST_GPQA_REFERENCE_PATH")
        }
        let tokenizerPath = options.value(
            for: "--tokenizer",
            default: environmentValue("MLXFAST_TOKENIZER_PATH", fallback: MLXFastConstants.defaultWeightsPath)
        )
        let outputPath = options.value(for: "--output", default: goldenPath)
        let caseCount = try parsePositiveInt(
            options.value(for: "--case-count", default: "\(MLXFastConstants.correctnessGPQACaseCount)"),
            optionName: "--case-count"
        )
        let maxNewTokens = try parsePositiveInt(
            options.value(for: "--max-new-tokens", default: "\(MLXFastConstants.correctnessGPQAMaxNewTokens)"),
            optionName: "--max-new-tokens"
        )
        guard maxNewTokens <= MLXFastConstants.correctnessMaxBehaviorSteps else {
            throw MLXFastError.invalidInput(
                "--max-new-tokens must be <= \(MLXFastConstants.correctnessMaxBehaviorSteps)"
            )
        }

        try requireFile(goldenPath, description: "correctness golden file")
        try requireFile(gpqaPath, description: "GPQA reference cases file")
        try requireFile(
            URL(fileURLWithPath: tokenizerPath).appendingPathComponent("tokenizer.json").path,
            description: "tokenizer.json"
        )
        try requireFile(
            URL(fileURLWithPath: tokenizerPath).appendingPathComponent("tokenizer_config.json").path,
            description: "tokenizer_config.json"
        )

        let tokenizer = try loadLocalTokenizer(at: tokenizerPath)
        let goldenData = try Data(contentsOf: URL(fileURLWithPath: goldenPath))
        let golden = try JSONDecoder().decode(GoldenDocument.self, from: goldenData)
        let gpqaData = try Data(contentsOf: URL(fileURLWithPath: gpqaPath))
        let gpqa = try JSONDecoder().decode(GPQAReferenceDocument.self, from: gpqaData)
        var behaviorCases: [GoldenBehaviorCase] = []
        var skippedOverBudgetGPQACases = 0
        for testCase in gpqa.cases {
            guard behaviorCases.count < caseCount else {
                break
            }
            if let behaviorCase = try buildGPQABehaviorCaseIfWithinPromptBudget(
                testCase,
                tokenizer: tokenizer,
                maxNewTokens: maxNewTokens
            ) {
                behaviorCases.append(behaviorCase)
            } else {
                skippedOverBudgetGPQACases += 1
            }
        }
        guard behaviorCases.count == caseCount else {
            throw MLXFastError.invalidInput(
                "GPQA reference produced \(behaviorCases.count) token-budget-valid cases; "
                    + "need \(caseCount); skipped_over_budget=\(skippedOverBudgetGPQACases); "
                    + "max_prompt_tokens=\(MLXFastConstants.correctnessMaxBehaviorPromptTokens)"
            )
        }

        let existingGates = golden.correctnessGates
        let existingBehavior = existingGates?.behaviorCases ?? []
        let mergedGates = GoldenCorrectnessGates(
            anchors: existingGates?.anchors,
            freeRun: existingGates?.freeRun,
            behavior: existingBehavior + behaviorCases
        )
        let merged = GoldenDocument(
            version: golden.version ?? 1,
            cases: golden.cases,
            correctnessGates: mergedGates,
            benchmark: golden.benchmark
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try writeValidatedGoldenDocument(encoder.encode(merged), to: outputPath)
        print(
            "attached GPQA behavior gates cases=\(behaviorCases.count) "
                + "max_new_tokens=\(maxNewTokens) "
                + "skipped_over_budget=\(skippedOverBudgetGPQACases) "
                + "output=\(outputPath)"
        )
    }

    // Operator tool: attach a free-run gate whose greedy continuation covers the
    // timed decode offset range. The 64-step teacher-forced base case only
    // exercises single-token forwards at offsets 512..575, while the timed
    // decode reaches 512..639 -- a submission could special-case a cheaper model
    // path for offsets only the (identifiable) timing worker ever visits and no
    // structural gate would notice. A 512-token free-run case with >= 128
    // generated-and-checked steps makes the unscored correctness gate exercise
    // every timed decode offset with different prompt content, so an
    // offset-gated fast path has to survive correctness too. Run this offline
    // against the baseline reference weights, then upload the regenerated
    // golden through the organizer process (docs/private-benchmark-security.md).
    private static func runAttachFreeRunGate(_ options: ParsedOptions) throws {
        try options.validate(
            valueOptions: [
                "--golden", "--weights", "--output", "--name", "--steps",
                "--case", "--prompt-file", "--tokenizer", "--exact-prefix",
            ],
            flagOptions: ["--allow-partial"]
        )
        let goldenPath = options.value(
            for: "--golden",
            default: environmentValue(
                "MLXFAST_CORRECTNESS_GOLDEN_PATH",
                fallback: MLXFastConstants.defaultGoldenPath
            )
        )
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue("MLXFAST_WEIGHTS_PATH", fallback: MLXFastConstants.defaultWeightsPath)
        )
        let outputPath = options.value(for: "--output", default: goldenPath)
        let caseName = options.value(for: "--name", default: "free-run-decode-offset-coverage")
        let steps = try parsePositiveInt(
            options.value(for: "--steps", default: "\(MLXFastConstants.benchmarkDecodeSteps)"),
            optionName: "--steps"
        )
        guard steps <= MLXFastConstants.correctnessMaxFreeRunSteps else {
            throw MLXFastError.invalidInput(
                "--steps must be <= \(MLXFastConstants.correctnessMaxFreeRunSteps)"
            )
        }
        if steps < MLXFastConstants.benchmarkDecodeSteps {
            // The command exists to cover the timed decode offsets; a partial
            // gate silently leaves the specialization gap open, so fail closed
            // unless the operator explicitly opts in (debugging, staged runs).
            guard options.hasFlag("--allow-partial") else {
                throw MLXFastError.invalidInput(
                    "--steps \(steps) is below benchmarkDecodeSteps "
                        + "\(MLXFastConstants.benchmarkDecodeSteps), so the gate would not cover "
                        + "the full timed decode offset range; pass --allow-partial to write it anyway"
                )
            }
            fputs(
                "attach-free-run-gate: warning: --steps \(steps) is below "
                    + "benchmarkDecodeSteps \(MLXFastConstants.benchmarkDecodeSteps); "
                    + "the gate will NOT cover the full timed decode offset range (--allow-partial)\n",
                stderr
            )
        }
        let exactPrefixRaw = options.value(for: "--exact-prefix", default: "")
        var exactPrefixTokens: Int?
        if !exactPrefixRaw.isEmpty {
            let parsed = try parsePositiveInt(exactPrefixRaw, optionName: "--exact-prefix")
            guard parsed <= steps else {
                throw MLXFastError.invalidInput("--exact-prefix must be <= --steps (\(steps))")
            }
            exactPrefixTokens = parsed
        }

        try requireFile(goldenPath, description: "correctness golden file")
        try requireFile(
            URL(fileURLWithPath: weightsPath).appendingPathComponent("config.json").path,
            description: "weights config.json"
        )
        // Strict-validate the INPUT before any generation or write. --output
        // defaults to the input path, so a malformed input must fail here --
        // never after the original has been replaced on disk.
        _ = try loadGoldenFixture(from: goldenPath)
        let goldenData = try Data(contentsOf: URL(fileURLWithPath: goldenPath))
        let golden = try JSONDecoder().decode(GoldenDocument.self, from: goldenData)

        let requiredPromptTokens = MLXFastConstants.correctnessPromptTokens
        let promptTokens: [Int]
        let promptFile = options.value(for: "--prompt-file", default: "")
        let sourceCaseName = options.value(for: "--case", default: "")
        if !promptFile.isEmpty {
            let tokenizerPath = options.value(for: "--tokenizer", default: weightsPath)
            try requireFile(
                URL(fileURLWithPath: tokenizerPath).appendingPathComponent("tokenizer.json").path,
                description: "tokenizer.json"
            )
            let tokenizer = try loadLocalTokenizer(at: tokenizerPath)
            let promptText = try String(contentsOfFile: promptFile, encoding: .utf8)
            let encoded = tokenizer.encode(text: promptText, addSpecialTokens: false)
            guard encoded.count >= requiredPromptTokens else {
                throw MLXFastError.invalidInput(
                    "--prompt-file tokenized to \(encoded.count) tokens; free-run gates need at least \(requiredPromptTokens)"
                )
            }
            promptTokens = Array(encoded.prefix(requiredPromptTokens))
        } else if !sourceCaseName.isEmpty {
            guard let sourceCase = golden.cases.first(where: { $0.name == sourceCaseName }) else {
                throw MLXFastError.invalidInput("golden does not contain base case \(sourceCaseName)")
            }
            promptTokens = sourceCase.promptTokens
        } else {
            guard let firstCase = golden.cases.first else {
                throw MLXFastError.invalidInput("golden contains no base cases to source a prompt from")
            }
            promptTokens = firstCase.promptTokens
        }
        guard promptTokens.count == requiredPromptTokens else {
            throw MLXFastError.invalidInput(
                "free-run prompt has \(promptTokens.count) tokens; need exactly \(requiredPromptTokens)"
            )
        }

        fputs(
            "attach-free-run-gate: generating \(steps) reference continuation tokens "
                + "(covers decode offsets \(promptTokens.count)..<\(promptTokens.count + steps))\n",
            stderr
        )
        let expectedTokens = try GemmaRuntime.generateGreedyTokens(
            GreedyGenerationOptions(
                weightsPath: weightsPath,
                promptTokens: promptTokens,
                steps: steps
            ),
            worker: try runtimeWorkerOptions(blockedGoldenPath: goldenPath)
        )

        let freeRunCase = GoldenFreeRunCase(
            name: caseName,
            promptTokens: promptTokens,
            expectedTokens: expectedTokens,
            exactPrefixTokens: exactPrefixTokens
        )
        let existingGates = golden.correctnessGates
        let mergedGates = GoldenCorrectnessGates(
            anchors: existingGates?.anchors,
            freeRun: (existingGates?.freeRunCases ?? []) + [freeRunCase],
            behavior: existingGates?.behavior
        )
        let merged = GoldenDocument(
            version: golden.version ?? 1,
            cases: golden.cases,
            correctnessGates: mergedGates,
            benchmark: golden.benchmark
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try writeValidatedGoldenDocument(encoder.encode(merged), to: outputPath)
        print(
            "attached free-run gate name=\(caseName) steps=\(steps) "
                + "decode_offsets=\(promptTokens.count)..<\(promptTokens.count + steps) "
                + "exact_prefix=\(exactPrefixTokens.map(String.init) ?? "full") "
                + "output=\(outputPath)"
        )
    }

    // Operator tool: generate a BASE golden case (the version-1 cases[] shape
    // consumed by `correctness` and the local benchmark modes) from a public
    // prompt text file against the reference weights. This is how the
    // checked-in public fixtures under correctness_prompts/ are produced:
    // tokenize the prompt with the weights-dir tokenizer using the same
    // addSpecialTokens convention as attach-free-run-gate's prompt-file path,
    // keep exactly the required 512 prompt tokens, greedy-generate the
    // requested continuation with the reference model, and write a fixture
    // that passes the strict loader at that step count. Greedy decoding is
    // deterministic, so fixtures generated from the same prompt at different
    // step counts are prefix-identical by construction.
    private static func runGenerateGolden(_ options: ParsedOptions) throws {
        try options.validate(
            valueOptions: ["--prompt-file", "--weights", "--tokenizer", "--output", "--name", "--steps"]
        )
        let promptFile = options.value(for: "--prompt-file", default: "")
        guard !promptFile.isEmpty else {
            throw MLXFastError.invalidInput("generate-golden requires --prompt-file PATH")
        }
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue("MLXFAST_WEIGHTS_PATH", fallback: MLXFastConstants.defaultWeightsPath)
        )
        let tokenizerPath = options.value(for: "--tokenizer", default: weightsPath)
        let outputPath = options.value(for: "--output", default: "")
        guard !outputPath.isEmpty else {
            throw MLXFastError.invalidInput("generate-golden requires --output PATH")
        }
        let caseName = options.value(for: "--name", default: "")
        guard !caseName.isEmpty else {
            throw MLXFastError.invalidInput("generate-golden requires --name NAME")
        }
        let steps = try parsePositiveInt(
            options.value(for: "--steps", default: ""),
            optionName: "--steps"
        )
        guard steps >= MLXFastConstants.correctnessSteps else {
            // The strict fixture loader rejects base cases shorter than the
            // correctness window, so fail before spending any generation time.
            throw MLXFastError.invalidInput(
                "--steps must be >= correctnessSteps \(MLXFastConstants.correctnessSteps)"
            )
        }

        try requireFile(promptFile, description: "golden prompt text file")
        try requireFile(
            URL(fileURLWithPath: tokenizerPath).appendingPathComponent("tokenizer.json").path,
            description: "tokenizer.json"
        )
        try requireFile(
            URL(fileURLWithPath: weightsPath).appendingPathComponent("config.json").path,
            description: "weights config.json"
        )

        let requiredPromptTokens = MLXFastConstants.correctnessPromptTokens
        let tokenizer = try loadLocalTokenizer(at: tokenizerPath)
        let promptText = try String(contentsOfFile: promptFile, encoding: .utf8)
        let encoded = tokenizer.encode(text: promptText, addSpecialTokens: false)
        guard encoded.count >= requiredPromptTokens else {
            throw MLXFastError.invalidInput(
                "--prompt-file tokenized to \(encoded.count) tokens; base golden cases need at least \(requiredPromptTokens)"
            )
        }
        let promptTokens = Array(encoded.prefix(requiredPromptTokens))

        fputs(
            "generate-golden: generating \(steps) reference continuation tokens "
                + "for case \(caseName) (prompt_tokens=\(promptTokens.count))\n",
            stderr
        )
        let expectedTokens = try GemmaRuntime.generateGreedyTokens(
            GreedyGenerationOptions(
                weightsPath: weightsPath,
                promptTokens: promptTokens,
                steps: steps
            ),
            // Block the output path like the attach tools block their input
            // golden: when regenerating an existing fixture in place, the
            // worker running submitted-surface model code must not be able to
            // read the fixture it is being asked to reproduce.
            worker: try runtimeWorkerOptions(blockedGoldenPath: outputPath)
        )

        let document = GoldenDocument(
            version: 1,
            cases: [
                GoldenCase(
                    name: caseName,
                    promptTokens: promptTokens,
                    expectedTokens: expectedTokens
                )
            ],
            benchmark: nil
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try writeValidatedGoldenDocument(encoder.encode(document), to: outputPath)
        // The staging write above validates at the default correctness window;
        // re-validate at the full generated step count so the written fixture
        // provably satisfies the consumer that needs every step (local-submit
        // requires benchmarkDecodeSteps + 1 expected tokens, etc.).
        _ = try loadGoldenFixture(
            from: outputPath,
            requiredSteps: steps,
            requiredPromptTokens: requiredPromptTokens
        )
        print(
            "generated golden case=\(caseName) prompt_tokens=\(promptTokens.count) "
                + "expected_tokens=\(expectedTokens.count) output=\(outputPath)"
        )
    }

    // Writes a merged golden by staging to a temp sibling and proving the
    // result loads through the strict fixture loader BEFORE it can touch the
    // destination. The attach commands default --output to the input golden,
    // so an in-place write followed by a failed validation would destroy the
    // original (typically the private golden) with nothing to roll back to.
    private static func writeValidatedGoldenDocument(_ outputData: Data, to outputPath: String) throws {
        let outputURL = URL(fileURLWithPath: outputPath)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let temporaryURL = outputURL.deletingLastPathComponent()
            .appendingPathComponent(".\(outputURL.lastPathComponent).attach-\(UUID().uuidString).tmp")
        defer {
            try? FileManager.default.removeItem(at: temporaryURL)
        }
        try outputData.write(to: temporaryURL, options: [.atomic])
        _ = try loadGoldenFixture(from: temporaryURL.path)
        if FileManager.default.fileExists(atPath: outputURL.path) {
            _ = try FileManager.default.replaceItemAt(outputURL, withItemAt: temporaryURL)
        } else {
            try FileManager.default.moveItem(at: temporaryURL, to: outputURL)
        }
    }

    private static func runGenerateGPQAAnswers(_ options: ParsedOptions) throws {
        try options.validate(
            valueOptions: ["--gpqa", "--weights", "--tokenizer", "--output", "--case-count", "--max-new-tokens"]
        )
        let gpqaPath = options.value(
            for: "--gpqa",
            default: environmentValue("MLXFAST_GPQA_REFERENCE_PATH", fallback: "")
        )
        guard !gpqaPath.isEmpty else {
            throw MLXFastError.invalidInput("generate-gpqa-answers requires --gpqa or MLXFAST_GPQA_REFERENCE_PATH")
        }
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue("MLXFAST_WEIGHTS_PATH", fallback: MLXFastConstants.defaultWeightsPath)
        )
        let tokenizerPath = options.value(
            for: "--tokenizer",
            default: environmentValue("MLXFAST_TOKENIZER_PATH", fallback: weightsPath)
        )
        let outputPath = options.value(
            for: "--output",
            default: environmentValue("MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH", fallback: "")
        )
        guard !outputPath.isEmpty else {
            throw MLXFastError.invalidInput(
                "generate-gpqa-answers requires --output or MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH"
            )
        }
        try requirePrivateOutputPath(outputPath, description: "semantic GPQA answer output")
        let caseCount = try parsePositiveInt(
            options.value(for: "--case-count", default: "\(MLXFastConstants.semanticGPQACaseCount)"),
            optionName: "--case-count"
        )
        let maxNewTokens = try parsePositiveInt(
            options.value(for: "--max-new-tokens", default: "\(MLXFastConstants.semanticGPQAMaxNewTokens)"),
            optionName: "--max-new-tokens"
        )
        guard maxNewTokens <= MLXFastConstants.correctnessMaxBehaviorSteps else {
            throw MLXFastError.invalidInput(
                "--max-new-tokens must be <= \(MLXFastConstants.correctnessMaxBehaviorSteps)"
            )
        }

        try requireFile(gpqaPath, description: "GPQA reference cases file")
        try requireFile(
            URL(fileURLWithPath: tokenizerPath).appendingPathComponent("tokenizer.json").path,
            description: "tokenizer.json"
        )
        try requireFile(
            URL(fileURLWithPath: tokenizerPath).appendingPathComponent("tokenizer_config.json").path,
            description: "tokenizer_config.json"
        )
        try requireFile(
            URL(fileURLWithPath: weightsPath).appendingPathComponent("config.json").path,
            description: "weights config.json"
        )

        let tokenizer = try loadLocalTokenizer(at: tokenizerPath)
        let data = try Data(contentsOf: URL(fileURLWithPath: gpqaPath))
        let gpqa = try JSONDecoder().decode(GPQAReferenceDocument.self, from: data)
        let worker = try runtimeWorkerOptions(blockedGoldenPath: gpqaPath)

        var answers: [SemanticGPQAAnswerCase] = []
        var skippedOverBudget = 0
        for testCase in gpqa.cases {
            guard answers.count < caseCount else {
                break
            }
            let promptTokens = tokenizer.encode(text: testCase.prompt, addSpecialTokens: false)
            guard !promptTokens.isEmpty else {
                throw MLXFastError.invalidInput("\(testCase.identifier).prompt tokenized to zero tokens")
            }
            guard promptTokens.count <= MLXFastConstants.correctnessMaxBehaviorPromptTokens else {
                skippedOverBudget += 1
                continue
            }

            let generated = try GemmaRuntime.generateGreedyTokens(
                GreedyGenerationOptions(
                    weightsPath: weightsPath,
                    promptTokens: promptTokens,
                    steps: maxNewTokens
                ),
                worker: worker
            )
            let decoded = tokenizer.decode(tokens: generated, skipSpecialTokens: true)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            answers.append(
                SemanticGPQAAnswerCase(
                    id: testCase.identifier,
                    domain: testCase.domain,
                    subdomain: testCase.subdomain,
                    prompt: testCase.prompt,
                    answerKey: testCase.answerKey,
                    referenceAnswer: referenceAnswer(for: testCase),
                    candidateAnswer: decoded,
                    candidateTokens: generated,
                    maxNewTokens: maxNewTokens
                )
            )
            fputs(
                "generate-gpqa-answers: generated \(answers.count)/\(caseCount) "
                    + "tokens=\(generated.count)\n",
                stderr
            )
        }
        guard answers.count == caseCount else {
            throw MLXFastError.invalidInput(
                "GPQA reference produced \(answers.count) token-budget-valid semantic cases; "
                    + "need \(caseCount); skipped_over_budget=\(skippedOverBudget)"
            )
        }

        let document = SemanticGPQAAnswerDocument(
            version: 1,
            cases: answers
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let outputURL = URL(fileURLWithPath: outputPath)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try encoder.encode(document).write(to: outputURL, options: [.atomic])
        print("generated semantic GPQA answer cases=\(answers.count) output=\(outputPath)")
    }

    private static func loadLocalTokenizer(at path: String) throws -> any Tokenizer {
        let modelFolder = URL(fileURLWithPath: path).standardizedFileURL
        return try runBlockingAsync {
            try await AutoTokenizer.from(modelFolder: modelFolder, strict: false)
        }
    }

    private static func requirePrivateOutputPath(_ path: String, description: String) throws {
        let privateDir = environmentValue("MLXFAST_PRIVATE_DIR", fallback: "")
        guard !privateDir.isEmpty else {
            return
        }
        let outputPath = absolutePath(path)
        let privatePath = absolutePath(privateDir)
        guard outputPath.hasPrefix(privatePath + "/") else {
            throw MLXFastError.invalidInput("\(description) must be under MLXFAST_PRIVATE_DIR")
        }
    }

    private static func referenceAnswer(for testCase: GPQAReferenceCase) -> String {
        if let expected = trimmedNonEmpty(testCase.expectedResponse) {
            return expected
        }
        if let accepted = testCase.acceptedResponses?.compactMap({ trimmedNonEmpty($0) }), !accepted.isEmpty {
            return accepted.joined(separator: "\n")
        }
        if let answerKey = trimmedNonEmpty(testCase.answerKey) {
            if let answerText = multipleChoiceAnswerText(in: testCase.prompt, answerKey: answerKey) {
                return "\(answerKey). \(answerText)"
            }
            return "Correct option: \(answerKey)"
        }
        return ""
    }

    private static func multipleChoiceAnswerText(in prompt: String, answerKey: String) -> String? {
        let normalizedKey = answerKey.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        guard normalizedKey.count == 1 else {
            return nil
        }
        for rawLine in prompt.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            for marker in ["\(normalizedKey).", "\(normalizedKey):", "\(normalizedKey))"]
                where line.hasPrefix(marker)
            {
                let start = line.index(line.startIndex, offsetBy: marker.count)
                let value = line[start...].trimmingCharacters(in: .whitespacesAndNewlines)
                if !value.isEmpty {
                    return value
                }
            }
        }
        return nil
    }

    private static func buildGPQABehaviorCaseIfWithinPromptBudget(
        _ testCase: GPQAReferenceCase,
        tokenizer: any Tokenizer,
        maxNewTokens: Int
    ) throws -> GoldenBehaviorCase? {
        let promptTokens = tokenizer.encode(text: testCase.prompt, addSpecialTokens: false)
        guard !promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("\(testCase.identifier).prompt tokenized to zero tokens")
        }
        guard promptTokens.count <= MLXFastConstants.correctnessMaxBehaviorPromptTokens else {
            return nil
        }
        let acceptedSequences = try acceptedReferenceTokenSequences(
            testCase: testCase,
            tokenizer: tokenizer,
            maxNewTokens: maxNewTokens,
            caseName: testCase.identifier
        )
        return GoldenBehaviorCase(
            name: testCase.identifier,
            promptTokens: promptTokens,
            acceptedTokenSequences: acceptedSequences,
            maxNewTokens: maxNewTokens,
            semanticPrompt: testCase.prompt,
            semanticAnswerKey: trimmedNonEmpty(testCase.answerKey),
            semanticReferenceAnswer: referenceAnswer(for: testCase),
            semanticDomain: trimmedNonEmpty(testCase.domain),
            semanticSubdomain: trimmedNonEmpty(testCase.subdomain)
        )
    }

    private static func acceptedReferenceTokenSequences(
        testCase: GPQAReferenceCase,
        tokenizer: any Tokenizer,
        maxNewTokens: Int,
        caseName: String
    ) throws -> [[Int]] {
        if let tokenSequences = testCase.acceptedTokenSequences {
            guard !tokenSequences.isEmpty else {
                throw MLXFastError.invalidInput("\(caseName).accepted_token_sequences must not be empty")
            }
            var acceptedPrefixes: [[Int]] = []
            for (index, sequence) in tokenSequences.enumerated() {
                guard !sequence.isEmpty else {
                    throw MLXFastError.invalidInput(
                        "\(caseName).accepted_token_sequences[\(index)] must not be empty"
                    )
                }
                acceptedPrefixes.append(Array(sequence.prefix(maxNewTokens)))
            }
            return uniqueSortedTokenSequences(acceptedPrefixes)
        }

        guard let acceptedResponses = testCase.acceptedResponses,
              !acceptedResponses.isEmpty
        else {
            throw MLXFastError.invalidInput(
                "\(caseName) requires accepted_token_sequences or accepted_responses generated from the reference model"
            )
        }

        let prefixes = ["", " ", "\n"]
        let suffixes = ["", ".", "\n"]
        var seen = Set<[Int]>()
        var sequences: [[Int]] = []
        for response in acceptedResponses {
            let trimmed = response.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                continue
            }
            for prefix in prefixes {
                for suffix in suffixes {
                    let tokens = tokenizer.encode(text: prefix + trimmed + suffix, addSpecialTokens: false)
                    guard !tokens.isEmpty, tokens.count <= maxNewTokens else {
                        continue
                    }
                    if seen.insert(tokens).inserted {
                        sequences.append(tokens)
                    }
                }
            }
        }
        guard !sequences.isEmpty else {
            throw MLXFastError.invalidInput(
                "\(caseName) accepted_responses have no tokenization within \(maxNewTokens) token(s)"
            )
        }
        return sequences.sorted { lhs, rhs in
            if lhs.count != rhs.count {
                return lhs.count < rhs.count
            }
            return lhs.lexicographicallyPrecedes(rhs)
        }
    }

    private static func uniqueSortedTokenSequences(_ tokenSequences: [[Int]]) -> [[Int]] {
        var seen = Set<[Int]>()
        var sequences: [[Int]] = []
        for sequence in tokenSequences where seen.insert(sequence).inserted {
            sequences.append(sequence)
        }
        return sequences.sorted { lhs, rhs in
            if lhs.count != rhs.count {
                return lhs.count < rhs.count
            }
            return lhs.lexicographicallyPrecedes(rhs)
        }
    }

    private final class AsyncResultBox<T>: @unchecked Sendable {
        var result: Result<T, Error>?
    }

    private static func runBlockingAsync<T>(
        _ body: @escaping @Sendable () async throws -> T
    ) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        let box = AsyncResultBox<T>()
        Task {
            do {
                box.result = .success(try await body())
            } catch {
                box.result = .failure(error)
            }
            semaphore.signal()
        }
        semaphore.wait()
        return try box.result!.get()
    }

    private static func parsePositiveInt(_ rawValue: String, optionName: String) throws -> Int {
        guard let value = Int(rawValue), value > 0 else {
            throw MLXFastError.invalidInput("\(optionName) must be a positive integer")
        }
        return value
    }

    private static func parseNonNegativeInt(_ rawValue: String, optionName: String) throws -> Int {
        guard let value = Int(rawValue), value >= 0 else {
            throw MLXFastError.invalidInput("\(optionName) must be a non-negative integer")
        }
        return value
    }

    private static func runtimeWorkerOptions(
        blockedGoldenPath: String? = nil,
        forwardsWorkerStderr: Bool = false
    ) throws -> RuntimeWorkerOptions? {
        // benchmark.sh's enforce_official_sandbox refuses to run the timed benchmark
        // with the worker or its sandbox disabled on an official run, but the public
        // behavior gate invokes `mlxfast-swift correctness` directly, so the same
        // policy must fail closed here too rather than silently falling back to an
        // unsandboxed (or worker-less) path. MLXFAST_OFFICIAL_BENCHMARK_RUN is set
        // by trusted workflow env and stripped from the worker's own environment by
        // sanitizedRuntimeWorkerEnvironment, so submitted code cannot observe it.
        let officialRun = environmentValue("MLXFAST_OFFICIAL_BENCHMARK_RUN", fallback: "0") == "1"
        let enabled = environmentValue("MLXFAST_USE_RUNTIME_WORKER", fallback: "1")
        guard enabled != "0" && enabled.lowercased() != "false" else {
            if officialRun {
                throw MLXFastError.invalidInput(
                    "official benchmark runs require the runtime worker; unset MLXFAST_USE_RUNTIME_WORKER"
                )
            }
            return nil
        }
        if officialRun, environmentValue("MLXFAST_NO_SANDBOX", fallback: "0") == "1" {
            throw MLXFastError.invalidInput(
                "official benchmark runs require the runtime worker sandbox; unset MLXFAST_NO_SANDBOX"
            )
        }
        let executable = environmentValue(
            "MLXFAST_RUNTIME_WORKER_EXECUTABLE",
            fallback: CommandLine.arguments.first ?? ""
        )
        guard !executable.isEmpty else {
            // Every no-worker exit must be fail-closed on an official run, not
            // just the explicit-disable ones above -- returning nil here means
            // "run the model in-process, no worker, no sandbox".
            if officialRun {
                throw MLXFastError.invalidInput(
                    "official benchmark runs require a runtime worker executable; none was configured or derivable"
                )
            }
            return nil
        }
        let executablePath: String
        if executable.hasPrefix("/") {
            executablePath = executable
        } else {
            executablePath = URL(
                fileURLWithPath: executable,
                relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            ).standardizedFileURL.path
        }
        var sandboxProfile = environmentValue("MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE", fallback: "")
        if sandboxProfile.isEmpty,
           environmentValue("MLXFAST_NO_SANDBOX", fallback: "0") != "1",
           let blockedGoldenPath,
           !blockedGoldenPath.isEmpty
        {
            sandboxProfile = try writeRuntimeWorkerSandboxProfile(
                blockedGoldenPath: blockedGoldenPath,
                allowedExecutablePath: executablePath
            )
        }
        if officialRun, sandboxProfile.isEmpty {
            throw MLXFastError.invalidInput(
                "official benchmark runs require a runtime worker sandbox profile; none was configured or derivable"
            )
        }
        return RuntimeWorkerOptions(
            executablePath: executablePath,
            sandboxProfilePath: sandboxProfile.isEmpty ? nil : sandboxProfile,
            // Fail closed: live worker-stderr forwarding is a local-edit-loop
            // convenience only. Official runs keep today's behavior where
            // worker stderr surfaces solely through the sanitized exit
            // diagnostic, so submitted code cannot stream hidden-prompt
            // content into CI logs.
            forwardsWorkerStderr: forwardsWorkerStderr && !officialRun
        )
    }

    private static func writeRuntimeWorkerSandboxProfile(
        blockedGoldenPath: String,
        allowedExecutablePath: String
    ) throws -> String {
        let sandboxExecutable = "/usr/bin/sandbox-exec"
        guard FileManager.default.isExecutableFile(atPath: sandboxExecutable) else {
            throw MLXFastError.invalidInput("sandbox-exec not found for runtime worker sandbox")
        }
        let profileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxfast-runtime-worker-\(UUID().uuidString).sb")
        let absoluteGoldenPath = absolutePath(blockedGoldenPath)
        let absoluteExecutablePath = absolutePath(allowedExecutablePath)
        var deniedReadRules = [
            "(deny file-read* (literal \"\(seatbeltEscaped(absoluteGoldenPath))\"))",
        ]
        let privateDir = environmentValue("MLXFAST_PRIVATE_DIR", fallback: "")
        if !privateDir.isEmpty {
            deniedReadRules.append(
                "(deny file-read* (subpath \"\(seatbeltEscaped(absolutePath(privateDir)))\"))"
            )
        }
        let profile = """
        (version 1)
        (allow default)
        (deny network*)
        (deny process-fork)
        (deny process-exec*)
        (allow process-exec (literal "\(seatbeltEscaped(absoluteExecutablePath))"))
        (deny file-write*)
        (allow file-write* (literal "/dev/null"))
        \(deniedReadRules.joined(separator: "\n"))
        """
        try profile.write(to: profileURL, atomically: true, encoding: .utf8)
        return profileURL.path
    }

    private static func absolutePath(_ path: String) -> String {
        let url: URL
        if path.hasPrefix("/") {
            url = URL(fileURLWithPath: path)
        } else {
            url = URL(
                fileURLWithPath: path,
                relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            )
        }
        return url.standardizedFileURL.resolvingSymlinksInPath().path
    }

    private static func seatbeltEscaped(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }

    private static func runRuntimeWorker(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--weights"])
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        try GemmaRuntime.runWorker(weightsPath: weightsPath)
    }

    private static func runCheckpointShards(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--index"])
        let indexPath = options.value(for: "--index", default: "")
        guard !indexPath.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint-shards requires --index PATH")
        }
        for shard in try CheckpointIndexTools.safetensorShardNames(from: indexPath) {
            print(shard)
        }
    }

    private static func printUsage() {
        print(
            """
            Usage:
              mlxfast-swift transform [--reference PATH] [--output PATH]
              mlxfast-swift verify-transform [--reference PATH] [--weights PATH] [--tmp-parent PATH] [--max-bytes N]
              mlxfast-swift correctness [--weights PATH] [--golden PATH]
              mlxfast-swift correctness-trace [--weights PATH] [--golden PATH] [--case NAME] --step N [--top-k N]
              mlxfast-swift preflight [--weights PATH] [--golden PATH]
              mlxfast-swift benchmark [--local-submit|--local-iterate] [--weights PATH] [--golden PATH] [--score-path PATH]
              mlxfast-swift attach-gpqa-gates [--golden PATH] --gpqa PATH [--tokenizer PATH] [--output PATH] [--case-count N] [--max-new-tokens N]
              mlxfast-swift attach-free-run-gate [--golden PATH] [--weights PATH] [--output PATH] [--name NAME] [--steps N] [--allow-partial] [--case NAME | --prompt-file PATH [--tokenizer PATH]] [--exact-prefix N]
              mlxfast-swift generate-golden --prompt-file PATH [--weights PATH] [--tokenizer PATH] --output PATH --name NAME --steps N
              mlxfast-swift generate-gpqa-answers --gpqa PATH [--weights PATH] [--tokenizer PATH] --output PATH [--case-count N] [--max-new-tokens N]
              mlxfast-swift checkpoint-shards --index PATH

            Swift-only Gemma 4 31B 4-bit harness entrypoint.
            """
        )
    }

    private static func environmentValue(_ name: String, fallback: String) -> String {
        let value = ProcessInfo.processInfo.environment[name] ?? ""
        return value.isEmpty ? fallback : value
    }

    private static func defaultCorrectnessGoldenPath() -> String {
        if FileManager.default.fileExists(atPath: MLXFastConstants.defaultGoldenPath) {
            return MLXFastConstants.defaultGoldenPath
        }
        let publicPath = environmentValue(
            "MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_PATH",
            fallback: MLXFastConstants.defaultPublicCorrectnessGoldenPath
        )
        if FileManager.default.fileExists(atPath: publicPath) {
            return publicPath
        }
        return MLXFastConstants.defaultGoldenPath
    }

    private static func trimmedNonEmpty(_ value: String?) -> String? {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? nil : trimmed
    }

}

private struct GPQAReferenceDocument: Decodable {
    let cases: [GPQAReferenceCase]
}

private struct GPQAReferenceCase: Decodable {
    let id: String?
    let prompt: String
    let expectedResponse: String?
    let answerKey: String?
    let acceptedTokenSequences: [[Int]]?
    let acceptedResponses: [String]?
    let domain: String?
    let subdomain: String?

    enum CodingKeys: String, CodingKey {
        case id
        case prompt
        case expectedResponse = "expected_response"
        case answerKey = "answer_key"
        case acceptedTokenSequences = "accepted_token_sequences"
        case acceptedResponses = "accepted_responses"
        case domain
        case subdomain
    }

    var identifier: String {
        let trimmed = id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? "gpqa-private" : trimmed
    }

}

private struct SemanticGPQAAnswerDocument: Encodable {
    let version: Int
    let cases: [SemanticGPQAAnswerCase]
}

private struct SemanticGPQAAnswerCase: Encodable {
    let id: String
    let domain: String?
    let subdomain: String?
    let prompt: String
    let answerKey: String?
    let referenceAnswer: String
    let candidateAnswer: String
    let candidateTokens: [Int]
    let maxNewTokens: Int

    enum CodingKeys: String, CodingKey {
        case id
        case domain
        case subdomain
        case prompt
        case answerKey = "answer_key"
        case referenceAnswer = "reference_answer"
        case candidateAnswer = "candidate_answer"
        case candidateTokens = "candidate_tokens"
        case maxNewTokens = "max_new_tokens"
    }
}

private struct ParsedOptions {
    private var values: [String: String] = [:]
    private var flags: Set<String> = []
    private var positionals: [String] = []
    private var duplicates: Set<String> = []

    init(_ arguments: [String]) {
        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            if argument.hasPrefix("--") {
                if let separator = argument.firstIndex(of: "=") {
                    let key = String(argument[..<separator])
                    let value = String(argument[argument.index(after: separator)...])
                    recordOption(key)
                    values[key] = value
                    index += 1
                } else if index + 1 < arguments.count && !arguments[index + 1].hasPrefix("--") {
                    recordOption(argument)
                    values[argument] = arguments[index + 1]
                    index += 2
                } else {
                    recordOption(argument)
                    flags.insert(argument)
                    index += 1
                }
            } else {
                positionals.append(argument)
                index += 1
            }
        }
    }

    private mutating func recordOption(_ name: String) {
        if values[name] != nil || flags.contains(name) {
            duplicates.insert(name)
        }
    }

    func value(for name: String, default defaultValue: String) -> String {
        values[name] ?? defaultValue
    }

    func hasFlag(_ name: String) -> Bool {
        flags.contains(name)
    }

    func validate(
        valueOptions: Set<String>,
        flagOptions: Set<String> = [],
        allowPositionals: Bool = false
    ) throws {
        if let duplicate = duplicates.first {
            throw MLXFastError.invalidInput("duplicate option \(duplicate)")
        }
        for name in values.keys where !valueOptions.contains(name) {
            throw MLXFastError.invalidInput("unknown option \(name)")
        }
        for (name, value) in values where value.isEmpty {
            throw MLXFastError.invalidInput("\(name) requires a non-empty value")
        }
        for flag in flags {
            if valueOptions.contains(flag) {
                throw MLXFastError.invalidInput("\(flag) requires a value")
            }
            if !flagOptions.contains(flag) {
                throw MLXFastError.invalidInput("unknown option \(flag)")
            }
        }
        if !allowPositionals, let positional = positionals.first {
            throw MLXFastError.invalidInput("unexpected argument \(positional)")
        }
    }
}
