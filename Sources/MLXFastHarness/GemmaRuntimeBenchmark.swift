import Foundation
import MLX
import MLXFastCore
import MLXFastModel

// GemmaRuntime is split across GemmaRuntime*.swift for auditability.
// Generated split; behavior identical to the original single file.

extension GemmaRuntime {
    public static func benchmark(
        _ options: BenchmarkOptions,
        worker: RuntimeWorkerOptions? = nil
    ) -> ScorePayload {
        if let worker {
            return benchmarkWithWorker(options, worker: worker)
        }

        let benchmarkStart = DispatchTime.now().uptimeNanoseconds
        let progress = makeBenchmarkProgressReporter(startedAt: benchmarkStart)
        var correctnessReport: CorrectnessReport?
        var benchmarkLoader: Gemma4WeightLoader?
        var transformedWeightsDigest: DirectoryDigest?
        var preflightSeconds = 0.0
        var correctnessSeconds = 0.0
        var timedBenchmarkSeconds = 0.0
        // Overwritten with the golden oracle's per-prompt baselines (if it
        // carries them) once the golden loads; until then failure payloads use
        // the calibrated defaults.
        var baselinePrefillSecondsPerToken = MLXFastConstants.officialBaselinePrefillSecondsPerToken
        var baselineDecodeSecondsPerToken = MLXFastConstants.officialBaselineDecodeSecondsPerToken

        progress(
            "start correctness_steps=\(options.correctnessSteps) "
                + "benchmark_decode_steps=\(options.benchmarkDecodeSteps)"
        )

        func makeFailedScore(
            error: String,
            correctness: CorrectnessReport?,
            passedCorrectness: Bool,
            expertStats explicitExpertStats: ExpertStreamingStats? = nil,
            firstFailingCase explicitFirstFailingCase: String? = nil,
            firstFailingStep explicitFirstFailingStep: Int? = nil,
            expectedToken explicitExpectedToken: Int? = nil,
            actualToken explicitActualToken: Int? = nil,
            weightsDigest: DirectoryDigest? = nil,
            peakRamGB: Double = 0,
            bandwidthGBPerToken: Double = 0,
            decodeSecondsPerToken: Double = 0,
            prefillSecondsPerToken: Double = 0,
            bandwidthSource: String = "",
            gpqaTTFT: GPQATTFTSummary = .zero
        ) -> ScorePayload {
            progress("failed passed_correctness=\(passedCorrectness) error=\(redactedProgressError(error))")
            return failedScore(
                error: error,
                correctness: correctness,
                passedCorrectness: passedCorrectness,
                expertStats: explicitExpertStats,
                firstFailingCase: explicitFirstFailingCase,
                firstFailingStep: explicitFirstFailingStep,
                expectedToken: explicitExpectedToken,
                actualToken: explicitActualToken,
                weightsDigest: weightsDigest,
                benchmarkWallSeconds: secondsSince(benchmarkStart),
                preflightSeconds: preflightSeconds,
                correctnessSeconds: correctnessSeconds,
                timedBenchmarkSeconds: timedBenchmarkSeconds,
                processResidentMemoryGB: currentResidentMemoryGB(),
                peakRamGB: peakRamGB,
                bandwidthGBPerToken: bandwidthGBPerToken,
                decodeSecondsPerToken: decodeSecondsPerToken,
                prefillSecondsPerToken: prefillSecondsPerToken,
                baselinePrefillSecondsPerToken: baselinePrefillSecondsPerToken,
                baselineDecodeSecondsPerToken: baselineDecodeSecondsPerToken,
                bandwidthSource: bandwidthSource,
                gpqaTTFT: gpqaTTFT
            )
        }

        do {
            try validateBenchmarkOptions(options)
            progress("golden load start")
            let golden = try loadGoldenFixture(from: options.goldenPath)
            progress(
                "golden load complete cases=\(golden.totalCorrectnessCaseCount) "
                    + "benchmark_oracle=\(golden.benchmark == nil ? "missing" : "present")"
            )
            guard !benchmarkRequiresRuntimeWorker(golden: golden) else {
                return makeFailedScore(
                    error: "benchmark behavior and GPQA TTFT gates require runtime worker timing",
                    correctness: correctnessReport,
                    passedCorrectness: false,
                    weightsDigest: transformedWeightsDigest
                )
            }
            progress("preflight start")
            let preflightStart = DispatchTime.now().uptimeNanoseconds
            _ = try BenchmarkPreflight.check(
                weightsPath: options.weightsPath,
                goldenPath: options.goldenPath
            )
            preflightSeconds = secondsSince(preflightStart)
            progress("preflight complete seconds=\(formatSeconds(preflightSeconds))")
            progress("weights digest start")
            transformedWeightsDigest = try directoryDigest(
                rootPath: options.weightsPath,
                ignoredRelativePaths: [".benchmark-source.sha256", ".gitkeep"]
            )
            if let transformedWeightsDigest {
                progress(
                    "weights digest complete files=\(transformedWeightsDigest.fileCount) "
                        + "bytes=\(transformedWeightsDigest.byteCount)"
                )
            }
            let config = try Gemma4Config.load(from: options.weightsPath)
            progress("correctness loader start")
            let correctnessLoader = try Gemma4WeightLoader(weightsPath: options.weightsPath)
            let correctnessCache = Gemma4RuntimeWeightCache(loader: correctnessLoader, config: config)
            let correctnessStart = DispatchTime.now().uptimeNanoseconds
            progress("correctness start cases=\(golden.totalCorrectnessCaseCount)")
            let correctness = runLayeredCorrectness(
                golden: golden,
                weightCache: correctnessCache,
                steps: options.correctnessSteps,
                progress: progress
            )
            correctnessSeconds = secondsSince(correctnessStart)
            correctnessReport = correctness
            progress(
                "correctness complete passed=\(correctness.passed) "
                    + "checked_steps=\(correctness.checkedSteps) "
                    + "seconds=\(formatSeconds(correctnessSeconds))"
            )
            guard correctness.passed else {
                return makeFailedScore(
                    error: correctness.error.isEmpty ? "correctness gate failed" : correctness.error,
                    correctness: correctness,
                    passedCorrectness: false,
                    weightsDigest: transformedWeightsDigest
                )
            }

            let runtimeBenchmarkLoader = try Gemma4WeightLoader(weightsPath: options.weightsPath)
            benchmarkLoader = runtimeBenchmarkLoader
            let benchmarkCache = Gemma4RuntimeWeightCache(loader: runtimeBenchmarkLoader, config: config)
            guard let benchmarkGolden = golden.benchmark else {
                throw MLXFastError.invalidInput("benchmark golden file must contain a benchmark oracle")
            }
            let promptPlan = try BenchmarkPrompt.plan(from: benchmarkGolden)
            let pairedBaseline = try PairedBaselineOverride.fromEnvironment()
            baselinePrefillSecondsPerToken = pairedBaseline?.prefillSecondsPerToken
                ?? benchmarkGolden.resolvedBaselinePrefillSecondsPerToken
            baselineDecodeSecondsPerToken = pairedBaseline?.decodeSecondsPerToken
                ?? benchmarkGolden.resolvedBaselineDecodeSecondsPerToken
            progress(
                "benchmark oracle ready prefill_tokens=\(promptPlan.prefillTokens.count) "
                    + "decode_seed_tokens=\(promptPlan.decodeSeedTokens.count) "
                    + "decode_tokens=\(options.benchmarkDecodeSteps) "
                    + "baseline_source=\(baselineSourceLabel(paired: pairedBaseline, golden: benchmarkGolden))"
            )

            Memory.peakMemory = 0
            let timedBenchmarkStart = DispatchTime.now().uptimeNanoseconds
            progress("timed benchmark start")
            let prefillSecondsPerToken = try measurePrefillSecondsPerToken(
                promptTokens: promptPlan.prefillTokens,
                expectedToken: promptPlan.expectedPrefillToken,
                weightCache: benchmarkCache,
                progress: progress
            )
            let decode = try measureDecode(
                seedTokens: promptPlan.decodeSeedTokens,
                expectedSeedToken: promptPlan.expectedDecodeSeedToken,
                expectedTokens: promptPlan.expectedDecodeTokens,
                decodeSteps: options.benchmarkDecodeSteps,
                weightCache: benchmarkCache,
                progress: progress
            )
            timedBenchmarkSeconds = secondsSince(timedBenchmarkStart)
            let peakRamGB = Double(Memory.peakMemory) / Double(1 << 30)
            let score = BenchmarkScore.score(
                decodeSecondsPerToken: decode.secondsPerToken,
                prefillSecondsPerToken: prefillSecondsPerToken,
                baselineDecodeSecondsPerToken: baselineDecodeSecondsPerToken,
                baselinePrefillSecondsPerToken: baselinePrefillSecondsPerToken
            )
            let decodeSpeedup = BenchmarkScore.speedup(
                baselineSecondsPerToken: baselineDecodeSecondsPerToken,
                candidateSecondsPerToken: decode.secondsPerToken
            )
            let prefillSpeedup = BenchmarkScore.speedup(
                baselineSecondsPerToken: baselinePrefillSecondsPerToken,
                candidateSecondsPerToken: prefillSecondsPerToken
            )
            let expertStats = expertStats(from: runtimeBenchmarkLoader)

            guard score.isFinite, score >= 0 else {
                return makeFailedScore(
                    error: "computed score was not finite",
                    correctness: correctnessReport,
                    passedCorrectness: true,
                    expertStats: expertStats,
                    weightsDigest: transformedWeightsDigest
                )
            }
            guard BenchmarkScore.passesSpeedupFloors(
                decodeSpeedup: decodeSpeedup,
                prefillSpeedup: prefillSpeedup
            ) else {
                return makeFailedScore(
                    error: speedupFloorFailureMessage(
                        decodeSpeedup: decodeSpeedup,
                        prefillSpeedup: prefillSpeedup
                    ),
                    correctness: correctnessReport,
                    passedCorrectness: true,
                    expertStats: expertStats,
                    weightsDigest: transformedWeightsDigest,
                    peakRamGB: peakRamGB,
                    bandwidthGBPerToken: decode.bandwidthGBPerToken,
                    decodeSecondsPerToken: decode.secondsPerToken,
                    prefillSecondsPerToken: prefillSecondsPerToken,
                    bandwidthSource: decode.bandwidthSource
                )
            }
            // Acceptance bands vs the resolved baseline B (see AcceptanceBand):
            // prefill +/-5%; decode +2% regression / -5% gain (per-submission decode
            // gain capped at 5% -> larger wins must be chunked). Fail closed on either.
            let prefillBand = AcceptanceBand.check(
                value: prefillSecondsPerToken, reference: baselinePrefillSecondsPerToken,
                upTolerance: MLXFastConstants.prefillBandUpTolerance,
                downTolerance: MLXFastConstants.prefillBandDownTolerance, label: "prefill"
            )
            let decodeBand = AcceptanceBand.check(
                value: decode.secondsPerToken, reference: baselineDecodeSecondsPerToken,
                upTolerance: MLXFastConstants.decodeBandUpTolerance,
                downTolerance: MLXFastConstants.decodeBandDownTolerance, label: "decode"
            )
            guard prefillBand.passed, decodeBand.passed else {
                return makeFailedScore(
                    error: "acceptance band failed: \(prefillBand.passed ? decodeBand.reason : prefillBand.reason)",
                    correctness: correctnessReport,
                    passedCorrectness: true,
                    expertStats: expertStats,
                    weightsDigest: transformedWeightsDigest,
                    peakRamGB: peakRamGB,
                    bandwidthGBPerToken: decode.bandwidthGBPerToken,
                    decodeSecondsPerToken: decode.secondsPerToken,
                    prefillSecondsPerToken: prefillSecondsPerToken,
                    bandwidthSource: decode.bandwidthSource
                )
            }
            progress(
                "complete score=\(formatDouble(score)) "
                    + "decode_speedup=\(formatDouble(decodeSpeedup)) "
                    + "prefill_speedup=\(formatDouble(prefillSpeedup)) "
                    + "wall_seconds=\(formatSeconds(secondsSince(benchmarkStart))) "
                    + "timed_seconds=\(formatSeconds(timedBenchmarkSeconds))"
            )

            return passedScore(
                score: score,
                peakRamGB: peakRamGB,
                bandwidthGBPerToken: decode.bandwidthGBPerToken,
                decodeSecondsPerToken: decode.secondsPerToken,
                prefillSecondsPerToken: prefillSecondsPerToken,
                baselinePrefillSecondsPerToken: baselinePrefillSecondsPerToken,
                baselineDecodeSecondsPerToken: baselineDecodeSecondsPerToken,
                benchmarkWallSeconds: secondsSince(benchmarkStart),
                preflightSeconds: preflightSeconds,
                correctnessSeconds: correctnessSeconds,
                timedBenchmarkSeconds: timedBenchmarkSeconds,
                numLayers: config.numHiddenLayers,
                correctness: correctness,
                expertStats: expertStats,
                bandwidthSource: decode.bandwidthSource,
                weightsDigest: transformedWeightsDigest,
                gpqaTTFT: .zero,
                partialResult: !options.checkGates || options.skipTimedBenchmark
            )
        } catch let mismatch as BenchmarkTokenMismatchError {
            return makeFailedScore(
                error: mismatch.description,
                correctness: correctnessReport,
                passedCorrectness: correctnessReport?.passed == true,
                expertStats: expertStats(from: benchmarkLoader),
                firstFailingCase: "benchmark",
                firstFailingStep: mismatch.step,
                expectedToken: nil,
                actualToken: nil,
                weightsDigest: transformedWeightsDigest
            )
        } catch {
            return makeFailedScore(
                error: "\(error)",
                correctness: correctnessReport,
                passedCorrectness: correctnessReport?.passed == true,
                expertStats: expertStats(from: benchmarkLoader),
                weightsDigest: transformedWeightsDigest
            )
        }
    }

    static func validateBenchmarkOptions(_ options: BenchmarkOptions) throws {
        // 0 is allowed deliberately: it means "skip the base teacher-forced case on this
        // run" (still runs anchors/free-run/behavior/GPQA/TTFT/timing), for a machine that
        // relies on a separate fleet to verify the base case's step range in parallel.
        // Skipping is a caller decision, not a harness one -- the harness never treats a
        // steps=0 run as having actually verified correctness by itself.
        guard options.correctnessSteps >= 0 else {
            throw MLXFastError.invalidInput("benchmark correctness steps must be >= 0")
        }
        guard options.correctnessSteps <= MLXFastConstants.correctnessSteps else {
            throw MLXFastError.invalidInput(
                "benchmark correctness steps \(options.correctnessSteps) exceeds golden length \(MLXFastConstants.correctnessSteps)"
            )
        }
        guard options.benchmarkDecodeSteps > 0 else {
            throw MLXFastError.invalidInput("benchmark decode steps must be positive")
        }
        guard options.benchmarkDecodeSteps <= MLXFastConstants.benchmarkDecodeSteps else {
            throw MLXFastError.invalidInput(
                "benchmark decode steps \(options.benchmarkDecodeSteps) exceeds oracle length \(MLXFastConstants.benchmarkDecodeSteps)"
            )
        }
        guard options.semanticGPQACaseCount > 0 else {
            throw MLXFastError.invalidInput("semantic GPQA case count must be positive")
        }
        guard options.semanticGPQAMaxNewTokens > 0,
              options.semanticGPQAMaxNewTokens <= MLXFastConstants.correctnessMaxBehaviorSteps
        else {
            throw MLXFastError.invalidInput(
                "semantic GPQA max_new_tokens must be in 1...\(MLXFastConstants.correctnessMaxBehaviorSteps)"
            )
        }
        // checkGates false skips anchors/free-run/behavior/GPQA; skipTimedBenchmark
        // true skips the prefill/decode measurement. Both at once would run neither
        // -- a run that checks and times nothing is never a valid machine role.
        guard options.checkGates || !options.skipTimedBenchmark else {
            throw MLXFastError.invalidInput(
                "benchmark checkGates=false and skipTimedBenchmark=true together check and time nothing"
            )
        }
    }

    static func benchmarkRequiresRuntimeWorker(golden: GoldenFixture) -> Bool {
        // Behavior gates include hidden GPQA TTFT. That timing is measured in
        // the trusted parent around sandboxed worker calls. The in-process path
        // cannot produce an equivalent trusted TTFT measurement, so fail closed.
        !(golden.correctnessGates?.behaviorCases.isEmpty ?? true)
    }

    static func semanticGPQACaptureOptions(from options: BenchmarkOptions) throws -> SemanticGPQACaptureOptions? {
        guard let outputPath = trimmedNonEmpty(options.semanticGPQAOutputPath) else {
            return nil
        }
        let tokenizerPath = trimmedNonEmpty(options.semanticGPQATokenizerPath) ?? options.weightsPath
        try requireFile(
            URL(fileURLWithPath: tokenizerPath).appendingPathComponent("tokenizer.json").path,
            description: "semantic GPQA tokenizer.json"
        )
        try requireFile(
            URL(fileURLWithPath: tokenizerPath).appendingPathComponent("tokenizer_config.json").path,
            description: "semantic GPQA tokenizer_config.json"
        )
        return SemanticGPQACaptureOptions(
            outputPath: outputPath,
            tokenizerPath: tokenizerPath,
            caseCount: options.semanticGPQACaseCount,
            maxNewTokens: options.semanticGPQAMaxNewTokens
        )
    }

    static func benchmarkWithWorker(
        _ options: BenchmarkOptions,
        worker workerOptions: RuntimeWorkerOptions
    ) -> ScorePayload {
        let benchmarkStart = DispatchTime.now().uptimeNanoseconds
        let progress = makeBenchmarkProgressReporter(startedAt: benchmarkStart)
        var correctnessReport: CorrectnessReport?
        var transformedWeightsDigest: DirectoryDigest?
        var preflightSeconds = 0.0
        var correctnessSeconds = 0.0
        var timedBenchmarkSeconds = 0.0
        var lastExpertStats = ExpertStreamingStats.zero
        var peakRamGB = 0.0
        // Overwritten with the golden oracle's per-prompt baselines (if it
        // carries them) once the golden loads; until then failure payloads use
        // the calibrated defaults.
        var baselinePrefillSecondsPerToken = MLXFastConstants.officialBaselinePrefillSecondsPerToken
        var baselineDecodeSecondsPerToken = MLXFastConstants.officialBaselineDecodeSecondsPerToken

        progress(
            "start correctness_steps=\(options.correctnessSteps) "
                + "benchmark_decode_steps=\(options.benchmarkDecodeSteps)"
        )

        func makeFailedScore(
            error: String,
            correctness: CorrectnessReport?,
            passedCorrectness: Bool,
            firstFailingCase explicitFirstFailingCase: String? = nil,
            firstFailingStep explicitFirstFailingStep: Int? = nil,
            expectedToken explicitExpectedToken: Int? = nil,
            actualToken explicitActualToken: Int? = nil,
            peakRamGB: Double = 0,
            bandwidthGBPerToken: Double = 0,
            decodeSecondsPerToken: Double = 0,
            prefillSecondsPerToken: Double = 0,
            bandwidthSource: String = "",
            gpqaTTFT: GPQATTFTSummary = .zero
        ) -> ScorePayload {
            progress("failed passed_correctness=\(passedCorrectness) error=\(redactedProgressError(error))")
            return failedScore(
                error: error,
                correctness: correctness,
                passedCorrectness: passedCorrectness,
                expertStats: lastExpertStats,
                firstFailingCase: explicitFirstFailingCase,
                firstFailingStep: explicitFirstFailingStep,
                expectedToken: explicitExpectedToken,
                actualToken: explicitActualToken,
                weightsDigest: transformedWeightsDigest,
                benchmarkWallSeconds: secondsSince(benchmarkStart),
                preflightSeconds: preflightSeconds,
                correctnessSeconds: correctnessSeconds,
                timedBenchmarkSeconds: timedBenchmarkSeconds,
                processResidentMemoryGB: currentResidentMemoryGB(),
                peakRamGB: peakRamGB,
                bandwidthGBPerToken: bandwidthGBPerToken,
                decodeSecondsPerToken: decodeSecondsPerToken,
                prefillSecondsPerToken: prefillSecondsPerToken,
                baselinePrefillSecondsPerToken: baselinePrefillSecondsPerToken,
                baselineDecodeSecondsPerToken: baselineDecodeSecondsPerToken,
                bandwidthSource: bandwidthSource,
                gpqaTTFT: gpqaTTFT
            )
        }

        do {
            try validateBenchmarkOptions(options)
            progress("preflight start")
            let preflightStart = DispatchTime.now().uptimeNanoseconds
            // Model-free preflight: verify required artifacts exist WITHOUT loading
            // config/tensors here. BenchmarkPreflight.check() calls Gemma4Config.load,
            // DenseTensorStore, and Gemma4WeightLoader (all EDITABLE MLXFastModel code),
            // which would execute submitted code in this trusted, unsandboxed parent. The
            // sandboxed runtime worker loads and validates config/dense/expert metadata
            // when it starts; malformed weights make it fail its protocol hello, surfacing
            // as a failed benchmark.
            try checkWorkerBenchmarkInputs(
                weightsPath: options.weightsPath,
                goldenPath: options.goldenPath
            )
            preflightSeconds = secondsSince(preflightStart)
            progress("preflight complete seconds=\(formatSeconds(preflightSeconds))")

            progress("weights digest start")
            transformedWeightsDigest = try directoryDigest(
                rootPath: options.weightsPath,
                ignoredRelativePaths: [".benchmark-source.sha256", ".gitkeep"]
            )
            if let transformedWeightsDigest {
                try enforceTransformedWeightsByteLimit(transformedWeightsDigest.byteCount)
                progress(
                    "weights digest complete files=\(transformedWeightsDigest.fileCount) "
                        + "bytes=\(transformedWeightsDigest.byteCount)"
                )
            }

            progress("golden load start")
            let golden = try loadGoldenFixture(from: options.goldenPath)
            progress(
                "golden load complete cases=\(golden.totalCorrectnessCaseCount) "
                    + "benchmark_oracle=\(golden.benchmark == nil ? "missing" : "present")"
            )

            guard let benchmarkGolden = golden.benchmark else {
                throw MLXFastError.invalidInput("benchmark golden file must contain a benchmark oracle")
            }
            let promptPlan = try BenchmarkPrompt.plan(from: benchmarkGolden)
            let pairedBaseline = try PairedBaselineOverride.fromEnvironment()
            baselinePrefillSecondsPerToken = pairedBaseline?.prefillSecondsPerToken
                ?? benchmarkGolden.resolvedBaselinePrefillSecondsPerToken
            baselineDecodeSecondsPerToken = pairedBaseline?.decodeSecondsPerToken
                ?? benchmarkGolden.resolvedBaselineDecodeSecondsPerToken
            progress(
                "benchmark oracle ready prefill_tokens=\(promptPlan.prefillTokens.count) "
                    + "decode_seed_tokens=\(promptPlan.decodeSeedTokens.count) "
                    + "decode_tokens=\(options.benchmarkDecodeSteps) "
                    + "baseline_source=\(baselineSourceLabel(paired: pairedBaseline, golden: benchmarkGolden))"
            )
            peakRamGB = 0
            lastExpertStats = .zero

            let prefillSecondsPerToken: Double
            let decode: DecodeMeasurement
            let benchmarkPeakRamGB: Double
            let benchmarkExpertStats: ExpertStreamingStats
            if options.skipTimedBenchmark {
                // This machine's role is the anchor/free-run/behavior/GPQA gates
                // only -- a separate "timing-only" machine (checkGates: false)
                // measures the real prefill/decode numbers. Placeholders here use
                // the golden-resolved baseline seconds-per-token exactly
                // (speedup == 1.0, always finite, always clears the floor)
                // rather than 0 or some arbitrary value: 0 would divide-by-zero
                // into +Infinity in BenchmarkScore.speedup, and Double.infinity
                // fails JSON encoding outright. Whatever ships here is
                // overwritten by the real timing-only machine's values when the
                // two are merged before combine-parallel-correctness.sh runs.
                progress("timed benchmark skipped (gates-only machine)")
                prefillSecondsPerToken = baselinePrefillSecondsPerToken
                decode = DecodeMeasurement(
                    secondsPerToken: baselineDecodeSecondsPerToken,
                    bandwidthGBPerToken: 0,
                    bandwidthSource: "skipped_gates_only_machine"
                )
                timedBenchmarkSeconds = 0
                benchmarkPeakRamGB = 0
                benchmarkExpertStats = .zero
            } else {
                let timedBenchmarkStart = DispatchTime.now().uptimeNanoseconds
                progress("timed benchmark start")
                progress("benchmark prefill worker start")
                do {
                    let prefillWorker = try RuntimeWorkerClient(
                        options: workerOptions,
                        weightsPath: options.weightsPath
                    )
                    defer {
                        prefillWorker.close()
                    }
                    prefillSecondsPerToken = try measureWorkerPrefillSecondsPerToken(
                        promptTokens: promptPlan.prefillTokens,
                        expectedToken: promptPlan.expectedPrefillToken,
                        worker: prefillWorker,
                        progress: progress,
                        peakRamGB: &peakRamGB,
                        expertStats: &lastExpertStats
                    )
                }
                progress("benchmark decode worker start")
                do {
                    let decodeWorker = try RuntimeWorkerClient(
                        options: workerOptions,
                        weightsPath: options.weightsPath
                    )
                    defer {
                        decodeWorker.close()
                    }
                    decode = try measureWorkerDecode(
                        seedTokens: promptPlan.decodeSeedTokens,
                        expectedSeedToken: promptPlan.expectedDecodeSeedToken,
                        expectedTokens: promptPlan.expectedDecodeTokens,
                        decodeSteps: options.benchmarkDecodeSteps,
                        worker: decodeWorker,
                        progress: progress,
                        peakRamGB: &peakRamGB,
                        expertStats: &lastExpertStats
                    )
                }
                timedBenchmarkSeconds = secondsSince(timedBenchmarkStart)
                benchmarkPeakRamGB = peakRamGB
                benchmarkExpertStats = lastExpertStats
            }

            // Computed unconditionally from whatever `decode`/prefillSecondsPerToken
            // ended up holding, real or (for a gates-only machine) the baseline
            // placeholder -- the placeholder yields score/speedups of exactly 1.0,
            // which trivially clears the floor checks below, so this guard logic
            // does not need its own skipTimedBenchmark branch.
            let score = BenchmarkScore.score(
                decodeSecondsPerToken: decode.secondsPerToken,
                prefillSecondsPerToken: prefillSecondsPerToken,
                baselineDecodeSecondsPerToken: baselineDecodeSecondsPerToken,
                baselinePrefillSecondsPerToken: baselinePrefillSecondsPerToken
            )
            let decodeSpeedup = BenchmarkScore.speedup(
                baselineSecondsPerToken: baselineDecodeSecondsPerToken,
                candidateSecondsPerToken: decode.secondsPerToken
            )
            let prefillSpeedup = BenchmarkScore.speedup(
                baselineSecondsPerToken: baselinePrefillSecondsPerToken,
                candidateSecondsPerToken: prefillSecondsPerToken
            )

            guard score.isFinite, score >= 0 else {
                return makeFailedScore(
                    error: "computed score was not finite",
                    correctness: correctnessReport,
                    passedCorrectness: false,
                    peakRamGB: benchmarkPeakRamGB,
                    bandwidthGBPerToken: decode.bandwidthGBPerToken,
                    decodeSecondsPerToken: decode.secondsPerToken,
                    prefillSecondsPerToken: prefillSecondsPerToken,
                    bandwidthSource: decode.bandwidthSource
                )
            }
            guard BenchmarkScore.passesSpeedupFloors(
                decodeSpeedup: decodeSpeedup,
                prefillSpeedup: prefillSpeedup
            ) else {
                return makeFailedScore(
                    error: speedupFloorFailureMessage(
                        decodeSpeedup: decodeSpeedup,
                        prefillSpeedup: prefillSpeedup
                    ),
                    correctness: correctnessReport,
                    passedCorrectness: false,
                    peakRamGB: benchmarkPeakRamGB,
                    bandwidthGBPerToken: decode.bandwidthGBPerToken,
                    decodeSecondsPerToken: decode.secondsPerToken,
                    prefillSecondsPerToken: prefillSecondsPerToken,
                    bandwidthSource: decode.bandwidthSource
                )
            }
            // Acceptance bands vs the resolved baseline (see the identical guard on the
            // single-machine path and AcceptanceBand): prefill +/-5%; decode +2%
            // regression / -5% gain (per-submission decode gain capped at 5%).
            let prefillBand = AcceptanceBand.check(
                value: prefillSecondsPerToken, reference: baselinePrefillSecondsPerToken,
                upTolerance: MLXFastConstants.prefillBandUpTolerance,
                downTolerance: MLXFastConstants.prefillBandDownTolerance, label: "prefill"
            )
            let decodeBand = AcceptanceBand.check(
                value: decode.secondsPerToken, reference: baselineDecodeSecondsPerToken,
                upTolerance: MLXFastConstants.decodeBandUpTolerance,
                downTolerance: MLXFastConstants.decodeBandDownTolerance, label: "decode"
            )
            guard prefillBand.passed, decodeBand.passed else {
                return makeFailedScore(
                    error: "acceptance band failed: \(prefillBand.passed ? decodeBand.reason : prefillBand.reason)",
                    correctness: correctnessReport,
                    passedCorrectness: false,
                    peakRamGB: benchmarkPeakRamGB,
                    bandwidthGBPerToken: decode.bandwidthGBPerToken,
                    decodeSecondsPerToken: decode.secondsPerToken,
                    prefillSecondsPerToken: prefillSecondsPerToken,
                    bandwidthSource: decode.bandwidthSource
                )
            }

            let correctnessStart = DispatchTime.now().uptimeNanoseconds
            progress("correctness start cases=\(golden.totalCorrectnessCaseCount)")
            progress("correctness worker start")
            let semanticCapture = try semanticGPQACaptureOptions(from: options)
            let correctnessResult: WorkerLayeredCorrectnessResult
            do {
                let correctnessWorker = try RuntimeWorkerClient(
                    options: workerOptions,
                    weightsPath: options.weightsPath
                )
                defer {
                    correctnessWorker.close()
                }
                correctnessResult = runLayeredCorrectnessWithWorker(
                    golden: golden,
                    worker: correctnessWorker,
                    steps: options.correctnessSteps,
                    checkGates: options.checkGates,
                    semanticCapture: semanticCapture,
                    progress: progress
                )
            }
            // The correctness worker -- the only process that runs editable
            // model code in this phase -- is now closed and reaped by the defer
            // above. Only now is it safe to write the hidden-GPQA capture, which
            // embeds hidden prompt text, answer keys, and reference answers: no
            // submitted code may be alive while that file exists on disk.
            // runLayeredCorrectnessWithWorker only collects and validates the
            // answers (returned in its result); it no longer writes them from
            // inside the live-worker window.
            if let semanticCapture, let semanticAnswers = correctnessResult.semanticGPQAAnswers {
                try writeSemanticGPQAAnswers(semanticAnswers, to: semanticCapture.outputPath)
                progress("correctness semantic GPQA answers written cases=\(semanticAnswers.count)")
            }
            correctnessSeconds = secondsSince(correctnessStart)
            let correctness = correctnessResult.report
            correctnessReport = correctness
            progress(
                "correctness complete passed=\(correctness.passed) "
                    + "checked_steps=\(correctness.checkedSteps) "
                    + "seconds=\(formatSeconds(correctnessSeconds))"
            )
            if correctnessResult.gpqaTTFT.caseCount > 0 {
                let gpqaTTFT = correctnessResult.gpqaTTFT
                progress(
                    "gpqa ttft complete cases=\(gpqaTTFT.caseCount) "
                        + "pass_count=\(gpqaTTFT.passCount) "
                        + "mean_seconds=\(formatSeconds(gpqaTTFT.meanSeconds)) "
                        + "p50_seconds=\(formatSeconds(gpqaTTFT.p50Seconds)) "
                        + "max_seconds=\(formatSeconds(gpqaTTFT.maxSeconds))"
                )
            }
            guard correctness.passed else {
                return makeFailedScore(
                    error: correctness.error.isEmpty ? "correctness gate failed" : correctness.error,
                    correctness: correctness,
                    passedCorrectness: false,
                    peakRamGB: benchmarkPeakRamGB,
                    bandwidthGBPerToken: decode.bandwidthGBPerToken,
                    decodeSecondsPerToken: decode.secondsPerToken,
                    prefillSecondsPerToken: prefillSecondsPerToken,
                    bandwidthSource: decode.bandwidthSource,
                    gpqaTTFT: correctnessResult.gpqaTTFT
                )
            }
            guard correctnessResult.gpqaTTFT.caseCount == 0 || correctnessResult.gpqaTTFT.passed else {
                return makeFailedScore(
                    error: "hidden GPQA TTFT gate failed",
                    correctness: correctness,
                    passedCorrectness: true,
                    peakRamGB: benchmarkPeakRamGB,
                    bandwidthGBPerToken: decode.bandwidthGBPerToken,
                    decodeSecondsPerToken: decode.secondsPerToken,
                    prefillSecondsPerToken: prefillSecondsPerToken,
                    bandwidthSource: decode.bandwidthSource,
                    gpqaTTFT: correctnessResult.gpqaTTFT
                )
            }

            progress(
                "complete score=\(formatDouble(score)) "
                    + "decode_speedup=\(formatDouble(decodeSpeedup)) "
                    + "prefill_speedup=\(formatDouble(prefillSpeedup)) "
                    + "wall_seconds=\(formatSeconds(secondsSince(benchmarkStart))) "
                    + "timed_seconds=\(formatSeconds(timedBenchmarkSeconds))"
            )

            return passedScore(
                score: score,
                peakRamGB: benchmarkPeakRamGB,
                bandwidthGBPerToken: decode.bandwidthGBPerToken,
                decodeSecondsPerToken: decode.secondsPerToken,
                prefillSecondsPerToken: prefillSecondsPerToken,
                baselinePrefillSecondsPerToken: baselinePrefillSecondsPerToken,
                baselineDecodeSecondsPerToken: baselineDecodeSecondsPerToken,
                benchmarkWallSeconds: secondsSince(benchmarkStart),
                preflightSeconds: preflightSeconds,
                correctnessSeconds: correctnessSeconds,
                timedBenchmarkSeconds: timedBenchmarkSeconds,
                numLayers: MLXFastConstants.numHiddenLayers,
                correctness: correctness,
                expertStats: benchmarkExpertStats,
                bandwidthSource: decode.bandwidthSource,
                weightsDigest: transformedWeightsDigest,
                gpqaTTFT: correctnessResult.gpqaTTFT,
                partialResult: !options.checkGates || options.skipTimedBenchmark
            )
        } catch let mismatch as BenchmarkTokenMismatchError {
            return makeFailedScore(
                error: mismatch.description,
                correctness: correctnessReport,
                passedCorrectness: correctnessReport?.passed == true,
                firstFailingCase: "benchmark",
                firstFailingStep: mismatch.step,
                expectedToken: nil,
                actualToken: nil
            )
        } catch {
            return makeFailedScore(
                error: "\(error)",
                correctness: correctnessReport,
                passedCorrectness: correctnessReport?.passed == true
            )
        }
    }

    struct SemanticGPQACaptureOptions {
        let outputPath: String
        let tokenizerPath: String
        let caseCount: Int
        let maxNewTokens: Int
    }

    struct DecodeMeasurement {
        let secondsPerToken: Double
        let bandwidthGBPerToken: Double
        let bandwidthSource: String
    }

    static func measurePrefillSecondsPerToken(
        promptTokens: [Int],
        expectedToken: Int,
        weightCache: Gemma4RuntimeWeightCache,
        progress: ((String) -> Void)? = nil
    ) throws -> Double {
        guard !promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("benchmark prefill prompt must not be empty")
        }

        let totalRuns = MLXFastConstants.benchmarkPrefillWarmupRuns
            + MLXFastConstants.benchmarkPrefillTimedRuns
        var timedElapsed: [Double] = []
        timedElapsed.reserveCapacity(MLXFastConstants.benchmarkPrefillTimedRuns)

        for runIndex in 0..<totalRuns {
            let runLabel = runIndex < MLXFastConstants.benchmarkPrefillWarmupRuns ? "warmup" : "timed"
            let runOrdinal = runIndex < MLXFastConstants.benchmarkPrefillWarmupRuns
                ? runIndex + 1
                : runIndex - MLXFastConstants.benchmarkPrefillWarmupRuns + 1
            let runTotal = runIndex < MLXFastConstants.benchmarkPrefillWarmupRuns
                ? MLXFastConstants.benchmarkPrefillWarmupRuns
                : MLXFastConstants.benchmarkPrefillTimedRuns
            progress?(
                "prefill \(runLabel) \(runOrdinal)/\(runTotal) start "
                    + "prompt_tokens=\(promptTokens.count)"
            )
            let cache = Gemma4ModelCache(config: weightCache.config)
            let start = DispatchTime.now().uptimeNanoseconds
            let logits = try Gemma4Model.logits(
                inputIDs: inputIDsArray(promptTokens),
                weightCache: weightCache,
                cache: cache,
                positionOffset: 0
            )
            eval(logits)
            let token = try GemmaCorrectness.greedyToken(from: logits)
            try requireBenchmarkMatch(
                BenchmarkOutputValidator.comparePrefillToken(
                    expectedToken: expectedToken,
                    actualToken: token
                )
            )
            let elapsed = secondsSince(start)
            Memory.clearCache()
            progress?(
                "prefill \(runLabel) \(runOrdinal)/\(runTotal) complete "
                    + "seconds=\(formatSeconds(elapsed))"
            )

            if runIndex >= MLXFastConstants.benchmarkPrefillWarmupRuns {
                timedElapsed.append(elapsed)
            }
        }

        guard !timedElapsed.isEmpty else {
            throw MLXFastError.invalidInput("benchmark prefill needs at least one timed run")
        }
        let meanElapsed = timedElapsed.reduce(0, +) / Double(timedElapsed.count)
        let secondsPerToken = meanElapsed / Double(promptTokens.count)
        progress?("prefill complete seconds_per_token=\(formatDouble(secondsPerToken))")
        return secondsPerToken
    }

    static func measureWorkerPrefillSecondsPerToken(
        promptTokens: [Int],
        expectedToken: Int,
        worker: RuntimeWorkerClient,
        progress: ((String) -> Void)? = nil,
        peakRamGB: inout Double,
        expertStats: inout ExpertStreamingStats
    ) throws -> Double {
        guard !promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("benchmark prefill prompt must not be empty")
        }

        let totalRuns = MLXFastConstants.benchmarkPrefillWarmupRuns
            + MLXFastConstants.benchmarkPrefillTimedRuns
        var timedElapsed: [Double] = []
        timedElapsed.reserveCapacity(MLXFastConstants.benchmarkPrefillTimedRuns)

        for runIndex in 0..<totalRuns {
            let runLabel = runIndex < MLXFastConstants.benchmarkPrefillWarmupRuns ? "warmup" : "timed"
            let runOrdinal = runIndex < MLXFastConstants.benchmarkPrefillWarmupRuns
                ? runIndex + 1
                : runIndex - MLXFastConstants.benchmarkPrefillWarmupRuns + 1
            let runTotal = runIndex < MLXFastConstants.benchmarkPrefillWarmupRuns
                ? MLXFastConstants.benchmarkPrefillWarmupRuns
                : MLXFastConstants.benchmarkPrefillTimedRuns
            progress?(
                "prefill \(runLabel) \(runOrdinal)/\(runTotal) start "
                    + "prompt_tokens=\(promptTokens.count)"
            )
            // The worker contains submitted model code, so do not trust its
            // reported prefill duration as the score source. The trusted
            // parent measures the full request/response wall time.
            let prefillStart = DispatchTime.now().uptimeNanoseconds
            let response = try worker.prefill(promptTokens: promptTokens)
            let elapsed = secondsSince(prefillStart)
            guard let token = response.token else {
                throw MLXFastError.invalidInput("runtime worker prefill response missing token")
            }
            try requireBenchmarkMatch(
                BenchmarkOutputValidator.comparePrefillToken(
                    expectedToken: expectedToken,
                    actualToken: token
                )
            )
            let diagnostics = try worker.phaseDiagnostics()
            expertStats = diagnostics.expertStats ?? expertStats
            peakRamGB = max(peakRamGB, diagnostics.peakRamGB ?? 0)
            progress?(
                "prefill \(runLabel) \(runOrdinal)/\(runTotal) complete "
                    + "seconds=\(formatSeconds(elapsed))"
            )

            if runIndex >= MLXFastConstants.benchmarkPrefillWarmupRuns {
                timedElapsed.append(elapsed)
            }
        }

        guard !timedElapsed.isEmpty else {
            throw MLXFastError.invalidInput("benchmark prefill needs at least one timed run")
        }
        let meanElapsed = timedElapsed.reduce(0, +) / Double(timedElapsed.count)
        let secondsPerToken = meanElapsed / Double(promptTokens.count)
        progress?("prefill complete seconds_per_token=\(formatDouble(secondsPerToken))")
        return secondsPerToken
    }

    static func measureDecode(
        seedTokens: [Int],
        expectedSeedToken: Int,
        expectedTokens: [Int],
        decodeSteps: Int = MLXFastConstants.benchmarkDecodeSteps,
        weightCache: Gemma4RuntimeWeightCache,
        progress: ((String) -> Void)? = nil
    ) throws -> DecodeMeasurement {
        guard !seedTokens.isEmpty else {
            throw MLXFastError.invalidInput("benchmark decode seed must not be empty")
        }
        guard expectedTokens.count >= decodeSteps else {
            throw MLXFastError.invalidInput(
                "benchmark decode oracle has \(expectedTokens.count) tokens; need at least \(decodeSteps)"
            )
        }
        let timingPlan = try DecodeTimingPlan(
            seedTokenCount: seedTokens.count,
            decodeSteps: decodeSteps
        )

        // Start the scored decode phase before the seed prefill. Otherwise
        // submitted model code can hide speculative work for future decode steps
        // in setup that is not charged to the score.
        //
        // Exactly one whole-prompt (seed) forward runs in this phase, with NO
        // preceding warmup pass. A second identical whole-prompt forward (the
        // warmup this used to run) let submitted model code memoize one pass and
        // serve the other from that memo, collapsing two charged forwards into
        // one and inflating decode_speedup. The trusted harness cannot force
        // editable code to recompute a forward it issues, so the defense is to
        // never issue two identical forwards in the timed window. (Mirrors the
        // runtime worker's decode_begin; this in-process path is only used when
        // the worker is disabled, which official runs forbid.)
        let decodePhaseStart = DispatchTime.now().uptimeNanoseconds
        progress?("decode measured start tokens=\(timingPlan.decodeSteps) includes_seed_prefill=true")
        progress?("decode seed prefill start seed_tokens=\(seedTokens.count)")
        let cache = Gemma4ModelCache(config: weightCache.config)
        var logits = try Gemma4Model.logits(
            inputIDs: inputIDsArray(seedTokens),
            weightCache: weightCache,
            cache: cache,
            positionOffset: 0
        )
        var token = try GemmaCorrectness.greedyToken(from: logits)
        try requireBenchmarkMatch(
            BenchmarkOutputValidator.compareDecodeSeedToken(
                expectedToken: expectedSeedToken,
                actualToken: token
            )
        )
        cache.materializeCachedState()
        progress?("decode seed prefill complete")

        var actualTokens: [Int] = []
        actualTokens.reserveCapacity(timingPlan.decodeSteps)
        for decodedStep in 0..<timingPlan.decodeSteps {
            let inputToken = decodedStep == 0 ? expectedSeedToken : expectedTokens[decodedStep - 1]
            logits = try Gemma4Model.logits(
                inputIDs: inputIDsArray([inputToken]),
                weightCache: weightCache,
                cache: cache,
                positionOffset: try timingPlan.positionOffset(forDecodedStep: decodedStep)
            )
            token = try GemmaCorrectness.greedyToken(from: logits)
            actualTokens.append(token)
            let expectedToken = expectedTokens[decodedStep]
            if token != expectedToken {
                throw BenchmarkTokenMismatchError(
                    comparison: BenchmarkTokenComparison(
                        passed: false,
                        label: "benchmark decode token",
                        step: decodedStep,
                        expectedToken: expectedToken,
                        actualToken: token
                    )
                )
            }
            reportProgress(
                step: decodedStep + 1,
                total: timingPlan.decodeSteps,
                intervalSteps: 64,
                progress: { step, total in
                    progress?("decode measured generated \(step)/\(total) tokens")
                }
            )
        }

        let elapsed = secondsSince(decodePhaseStart)
        progress?(
            "decode measured complete seconds=\(formatSeconds(elapsed)) "
                + "seconds_per_token=\(formatDouble(elapsed / Double(timingPlan.decodeSteps))) "
                + "bandwidth_gb_per_token=0 bandwidth_source=\(bandwidthSource)"
        )
        return DecodeMeasurement(
            secondsPerToken: elapsed / Double(timingPlan.decodeSteps),
            bandwidthGBPerToken: 0,
            bandwidthSource: bandwidthSource
        )
    }

    static func measureWorkerDecode(
        seedTokens: [Int],
        expectedSeedToken: Int,
        expectedTokens: [Int],
        decodeSteps: Int = MLXFastConstants.benchmarkDecodeSteps,
        worker: RuntimeWorkerClient,
        progress: ((String) -> Void)? = nil,
        peakRamGB: inout Double,
        expertStats: inout ExpertStreamingStats
    ) throws -> DecodeMeasurement {
        guard !seedTokens.isEmpty else {
            throw MLXFastError.invalidInput("benchmark decode seed must not be empty")
        }
        guard expectedTokens.count >= decodeSteps else {
            throw MLXFastError.invalidInput(
                "benchmark decode oracle has \(expectedTokens.count) tokens; need at least \(decodeSteps)"
            )
        }
        // The worker contains submitted model code, so do not trust
        // worker-reported per-step timing as the score source. The trusted
        // parent measures decode_begin plus checked decode steps as one phase.
        let decodePhaseStart = DispatchTime.now().uptimeNanoseconds
        progress?("decode measured start tokens=\(decodeSteps) includes_seed_prefill=true")
        let beginResponse = try worker.beginDecode(seedTokens: seedTokens)
        guard let seedToken = beginResponse.seedToken else {
            throw MLXFastError.invalidInput("runtime worker decode_begin response missing seed token")
        }
        try requireBenchmarkMatch(
            BenchmarkOutputValidator.compareDecodeSeedToken(
                expectedToken: expectedSeedToken,
                actualToken: seedToken
            )
        )

        var actualTokens: [Int] = []
        actualTokens.reserveCapacity(decodeSteps)
        for decodedStep in 0..<decodeSteps {
            let inputToken = decodedStep == 0 ? expectedSeedToken : expectedTokens[decodedStep - 1]
            let response = try worker.decodeStep(inputToken: inputToken)
            guard let token = response.token else {
                throw MLXFastError.invalidInput("runtime worker decode_step response missing token")
            }
            actualTokens.append(token)
            let expectedToken = expectedTokens[decodedStep]
            if token != expectedToken {
                throw BenchmarkTokenMismatchError(
                    comparison: BenchmarkTokenComparison(
                        passed: false,
                        label: "benchmark decode token",
                        step: decodedStep,
                        expectedToken: expectedToken,
                        actualToken: token
                    )
                )
            }
            reportProgress(
                step: decodedStep + 1,
                total: decodeSteps,
                intervalSteps: 64,
                progress: { step, total in
                    progress?("decode measured generated \(step)/\(total) tokens")
                }
            )
        }

        let measuredSeconds = secondsSince(decodePhaseStart)
        let diagnostics = try worker.phaseDiagnostics()
        expertStats = diagnostics.expertStats ?? expertStats
        peakRamGB = max(peakRamGB, diagnostics.peakRamGB ?? 0)
        let secondsPerToken = measuredSeconds / Double(decodeSteps)
        let actualTokensComparison = BenchmarkOutputValidator.compareDecodeTokens(
            expectedTokens: Array(expectedTokens.prefix(decodeSteps)),
            actualTokens: actualTokens
        )
        try requireBenchmarkMatch(actualTokensComparison)
        progress?(
            "decode measured complete seconds=\(formatSeconds(measuredSeconds)) "
                + "seconds_per_token=\(formatDouble(secondsPerToken)) "
                + "bandwidth_gb_per_token=0 bandwidth_source=\(bandwidthSource)"
        )
        return DecodeMeasurement(
            secondsPerToken: secondsPerToken,
            bandwidthGBPerToken: 0,
            bandwidthSource: bandwidthSource
        )
    }

    /// Bandwidth diagnostic source label for the RAM-resident dense runtime.
    /// The whole model lives in unified memory after untimed init, so there
    /// is no expert-streaming byte counter to report: this is a fixed,
    /// non-ranking audit field (see `AGENTS.md`/`CLAUDE.md`).
    static let bandwidthSource = "ram_resident_model"

    /// The dense, RAM-resident runtime has no expert-cache/streaming machinery
    /// (see `AGENTS.md`/`CLAUDE.md`); this always returns the zero struct so
    /// the score schema keeps its `expert_*` fields (all zero) unchanged.
    static func expertStats(from weightCache: Gemma4RuntimeWeightCache) -> ExpertStreamingStats {
        .zero
    }

    static func expertStats(from loader: Gemma4WeightLoader?) -> ExpertStreamingStats {
        .zero
    }

    static func passedScore(
        score: Double,
        peakRamGB: Double,
        bandwidthGBPerToken: Double,
        decodeSecondsPerToken: Double,
        prefillSecondsPerToken: Double,
        baselinePrefillSecondsPerToken: Double = MLXFastConstants.officialBaselinePrefillSecondsPerToken,
        baselineDecodeSecondsPerToken: Double = MLXFastConstants.officialBaselineDecodeSecondsPerToken,
        benchmarkWallSeconds: Double,
        preflightSeconds: Double,
        correctnessSeconds: Double,
        timedBenchmarkSeconds: Double,
        numLayers: Int,
        correctness: CorrectnessReport,
        expertStats: ExpertStreamingStats,
        bandwidthSource: String,
        weightsDigest: DirectoryDigest?,
        gpqaTTFT: GPQATTFTSummary,
        partialResult: Bool = false
    ) -> ScorePayload {
        ScorePayload(
            score: score,
            passed: true,
            metrics: ScoreMetrics(
                peakRamGB: peakRamGB,
                bandwidthGBPerToken: bandwidthGBPerToken,
                decodeSecondsPerToken: decodeSecondsPerToken,
                prefillSecondsPerToken: prefillSecondsPerToken,
                baselineDecodeSecondsPerToken: baselineDecodeSecondsPerToken,
                baselinePrefillSecondsPerToken: baselinePrefillSecondsPerToken,
                benchmarkWallSeconds: benchmarkWallSeconds,
                preflightSeconds: preflightSeconds,
                correctnessSeconds: correctnessSeconds,
                timedBenchmarkSeconds: timedBenchmarkSeconds,
                gpqaTTFTPassed: gpqaTTFT.passed,
                gpqaTTFTPassCount: gpqaTTFT.passCount,
                gpqaTTFTCaseCount: gpqaTTFT.caseCount,
                gpqaTTFTSeconds: gpqaTTFT.meanSeconds,
                gpqaTTFTP50Seconds: gpqaTTFT.p50Seconds,
                gpqaTTFTMaxSeconds: gpqaTTFT.maxSeconds,
                gpqaTTFTSource: gpqaTTFT.source,
                processResidentMemoryGB: currentResidentMemoryGB(),
                passedCorrectness: true,
                numLayers: numLayers,
                checkedSteps: correctness.checkedSteps,
                caseCount: correctness.caseCount,
                expertCacheHits: expertStats.cacheHits,
                expertCacheMisses: expertStats.cacheMisses,
                expertCacheEvictions: expertStats.cacheEvictions,
                expertBytesRead: expertStats.bytesRead,
                expertReadSeconds: expertStats.readSeconds,
                expertPeakCachedTensors: expertStats.peakCachedTensors,
                expertHitRate: expertStats.hitRate,
                firstFailingLayer: nil,
                firstFailingCase: nil,
                firstFailingStep: nil,
                expectedToken: nil,
                actualToken: nil,
                maxAbsDiff: 0,
                goldenHash: correctness.goldenHash,
                bandwidthSource: bandwidthSource,
                error: "",
                commit: commitIdentifier(),
                timestamp: ISO8601DateFormatter().string(from: Date()),
                harnessHash: harnessHash(),
                weightsHash: weightsDigest?.sha256 ?? "",
                weightsByteCount: weightsDigest?.byteCount ?? 0,
                weightsFileCount: weightsDigest?.fileCount ?? 0,
                runtime: "swift",
                partialResult: partialResult
            )
        )
    }

    static func failedScore(
        error: String,
        correctness: CorrectnessReport?,
        passedCorrectness: Bool,
        expertStats explicitExpertStats: ExpertStreamingStats? = nil,
        firstFailingCase explicitFirstFailingCase: String? = nil,
        firstFailingStep explicitFirstFailingStep: Int? = nil,
        expectedToken explicitExpectedToken: Int? = nil,
        actualToken explicitActualToken: Int? = nil,
        weightsDigest: DirectoryDigest? = nil,
        benchmarkWallSeconds: Double = 0,
        preflightSeconds: Double = 0,
        correctnessSeconds: Double = 0,
        timedBenchmarkSeconds: Double = 0,
        processResidentMemoryGB: Double = 0,
        peakRamGB: Double = 0,
        bandwidthGBPerToken: Double = 0,
        decodeSecondsPerToken: Double = 0,
        prefillSecondsPerToken: Double = 0,
        baselinePrefillSecondsPerToken: Double = MLXFastConstants.officialBaselinePrefillSecondsPerToken,
        baselineDecodeSecondsPerToken: Double = MLXFastConstants.officialBaselineDecodeSecondsPerToken,
        bandwidthSource: String = "",
        gpqaTTFT: GPQATTFTSummary = .zero,
        runtime: String = "swift"
    ) -> ScorePayload {
        let expertStats = explicitExpertStats ?? correctness?.expertStreamingStats ?? .zero
        return ScorePayload(
            score: nil,
            passed: false,
            metrics: ScoreMetrics(
                peakRamGB: peakRamGB,
                bandwidthGBPerToken: bandwidthGBPerToken,
                decodeSecondsPerToken: decodeSecondsPerToken,
                prefillSecondsPerToken: prefillSecondsPerToken,
                baselineDecodeSecondsPerToken: baselineDecodeSecondsPerToken,
                baselinePrefillSecondsPerToken: baselinePrefillSecondsPerToken,
                benchmarkWallSeconds: benchmarkWallSeconds,
                preflightSeconds: preflightSeconds,
                correctnessSeconds: correctnessSeconds,
                timedBenchmarkSeconds: timedBenchmarkSeconds,
                gpqaTTFTPassed: gpqaTTFT.passed,
                gpqaTTFTPassCount: gpqaTTFT.passCount,
                gpqaTTFTCaseCount: gpqaTTFT.caseCount,
                gpqaTTFTSeconds: gpqaTTFT.meanSeconds,
                gpqaTTFTP50Seconds: gpqaTTFT.p50Seconds,
                gpqaTTFTMaxSeconds: gpqaTTFT.maxSeconds,
                gpqaTTFTSource: gpqaTTFT.source,
                processResidentMemoryGB: processResidentMemoryGB,
                passedCorrectness: passedCorrectness,
                numLayers: MLXFastConstants.numHiddenLayers,
                checkedSteps: correctness?.checkedSteps ?? 0,
                caseCount: correctness?.caseCount ?? 0,
                expertCacheHits: expertStats.cacheHits,
                expertCacheMisses: expertStats.cacheMisses,
                expertCacheEvictions: expertStats.cacheEvictions,
                expertBytesRead: expertStats.bytesRead,
                expertReadSeconds: expertStats.readSeconds,
                expertPeakCachedTensors: expertStats.peakCachedTensors,
                expertHitRate: expertStats.hitRate,
                firstFailingLayer: nil,
                firstFailingCase: explicitFirstFailingCase ?? correctness?.firstFailingCase,
                firstFailingStep: explicitFirstFailingStep ?? correctness?.firstFailingStep,
                expectedToken: explicitExpectedToken,
                actualToken: explicitActualToken,
                maxAbsDiff: 0,
                goldenHash: correctness?.goldenHash ?? "",
                bandwidthSource: bandwidthSource,
                error: error,
                commit: commitIdentifier(),
                timestamp: ISO8601DateFormatter().string(from: Date()),
                harnessHash: harnessHash(),
                weightsHash: weightsDigest?.sha256 ?? "",
                weightsByteCount: weightsDigest?.byteCount ?? 0,
                weightsFileCount: weightsDigest?.fileCount ?? 0,
                runtime: runtime
            )
        )
    }

    static func baselineSourceLabel(
        paired: PairedBaselineOverride?,
        golden: BenchmarkGolden
    ) -> String {
        if paired != nil {
            return "paired_env"
        }
        return golden.baselineDecodeSecondsPerToken == nil ? "constants" : "golden"
    }

    static func speedupFloorFailureMessage(
        decodeSpeedup: Double,
        prefillSpeedup: Double
    ) -> String {
        "performance floor failed: decode_speedup=\(formatDouble(decodeSpeedup)) "
            + "floor=\(formatDouble(MLXFastConstants.scoreDecodeSpeedupFloor)) "
            + "prefill_speedup=\(formatDouble(prefillSpeedup)) "
            + "floor=\(formatDouble(MLXFastConstants.scorePrefillSpeedupFloor))"
    }

}
