import Darwin
import Foundation
@testable import MLXFastHarness
import Testing

@Suite(.serialized)
struct BenchmarkSafetyTests {
    @Test
    func localThermalGateRunsAtEveryPhaseBoundary() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let helper = root.appendingPathComponent("cool-gate-helper.sh")
        let log = root.appendingPathComponent("phases.log")
        try """
        #!/bin/sh
        printf '%s\\n' "$MLXFAST_LOCAL_COOL_GATE_PHASE" >> "$MLXFAST_COOL_GATE_TEST_LOG"
        """.write(to: helper, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: helper.path
        )

        let previousHelper = ProcessInfo.processInfo.environment["MLXFAST_LOCAL_COOL_GATE_HELPER"]
        let previousLog = ProcessInfo.processInfo.environment["MLXFAST_COOL_GATE_TEST_LOG"]
        setenv("MLXFAST_LOCAL_COOL_GATE_HELPER", helper.path, 1)
        setenv("MLXFAST_COOL_GATE_TEST_LOG", log.path, 1)
        defer {
            restoreEnvironment("MLXFAST_LOCAL_COOL_GATE_HELPER", value: previousHelper)
            restoreEnvironment("MLXFAST_COOL_GATE_TEST_LOG", value: previousLog)
        }

        try GemmaRuntime.runLocalPhaseCoolGate(phase: "prefill")
        try GemmaRuntime.runLocalPhaseCoolGate(phase: "decode")

        #expect(try String(contentsOf: log, encoding: .utf8) == "prefill\ndecode\n")
    }

    @Test
    func benchmarkScriptPreservesCompactJSONIntegrityMetrics() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        try """
        #!/bin/sh
        printf '%s\\n' '{"score":null,"passed":true,"metrics":{"weights_hash":"compact-hash","weights_file_count":7,"weights_byte_count":11}}'
        """.write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: ["MLXFAST_SKIP_TRANSFORM": "1"]
        )

        #expect(result.status == 0)
        let object = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: fixture.integrity))
                as? [String: Any]
        )
        #expect(object["weights_sha256"] as? String == "compact-hash")
        #expect(object["weights_file_count"] as? Int == 7)
        #expect(object["weights_byte_count"] as? Int == 11)
    }

    @Test(arguments: [".", ".."])
    func benchmarkScriptRejectsWeightsPathsThatCanEraseWorkspace(weightsPath: String) throws {
        let fixture = try makeBenchmarkScriptFixture(nestedWorkingDirectory: true)
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let sentinel = fixture.root.appendingPathComponent("do-not-delete.txt")
        try "sentinel".write(to: sentinel, atomically: true, encoding: .utf8)
        let invocation = fixture.root.appendingPathComponent("swift-invoked")
        try """
        #!/bin/sh
        touch "\(invocation.path)"
        exit 99
        """.write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_WEIGHTS_PATH": weightsPath,
                "MLXFAST_FORCE_TRANSFORM": "1",
            ]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("refusing"))
        #expect(FileManager.default.fileExists(atPath: sentinel.path))
        #expect(!FileManager.default.fileExists(atPath: invocation.path))
    }

    @Test
    func benchmarkScriptRejectsTrackedSourcesAsWeightsPath() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        try initializeGitIndex(at: fixture.workingDirectory)
        let trackedSource = fixture.workingDirectory
            .appendingPathComponent("Sources/MLXFastCore/Fixture.swift")
        let originalSource = try Data(contentsOf: trackedSource)
        let invocation = fixture.root.appendingPathComponent("swift-invoked")
        try "#!/bin/sh\ntouch \"\(invocation.path)\"\nexit 99\n".write(
            to: fixture.swift,
            atomically: true,
            encoding: .utf8
        )
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_WEIGHTS_PATH": trackedSource.deletingLastPathComponent().path,
                "MLXFAST_FORCE_TRANSFORM": "1",
            ]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("contains tracked file Sources/MLXFastCore/Fixture.swift"))
        #expect(try Data(contentsOf: trackedSource) == originalSource)
        #expect(!FileManager.default.fileExists(atPath: invocation.path))
    }

    @Test
    func benchmarkScriptRejectsTrackedScorePath() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        try initializeGitIndex(at: fixture.workingDirectory)
        let trackedSource = fixture.workingDirectory
            .appendingPathComponent("Sources/MLXFastCore/Fixture.swift")
        let originalSource = try Data(contentsOf: trackedSource)
        let invocation = fixture.root.appendingPathComponent("swift-invoked")
        try "#!/bin/sh\ntouch \"\(invocation.path)\"\nexit 99\n".write(
            to: fixture.swift,
            atomically: true,
            encoding: .utf8
        )
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_SCORE_PATH": trackedSource.path,
                "MLXFAST_SKIP_TRANSFORM": "1",
            ]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("would overwrite tracked file Sources/MLXFastCore/Fixture.swift"))
        #expect(try Data(contentsOf: trackedSource) == originalSource)
        #expect(!FileManager.default.fileExists(atPath: invocation.path))
    }

    @Test
    func benchmarkScriptRejectsWorkspaceAsVerificationTemporaryDirectory() throws {
        let fixture = try makeBenchmarkScriptFixture(nestedWorkingDirectory: true)
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let sentinel = fixture.workingDirectory.appendingPathComponent("do-not-delete.txt")
        try "sentinel".write(to: sentinel, atomically: true, encoding: .utf8)
        try "#!/bin/sh\nexit 99\n".write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_SKIP_TRANSFORM": "1",
                "MLXFAST_VERIFY_TRANSFORM": "1",
                "MLXFAST_VERIFY_TRANSFORM_TMP_PARENT": ".",
            ]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("refusing to clear unsafe transform verification temporary path"))
        #expect(FileManager.default.fileExists(atPath: sentinel.path))
    }

    @Test
    func benchmarkScriptStagesTransformInItsOnlySandboxWritableDirectory() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let scripts = fixture.workingDirectory.appendingPathComponent(".github/scripts")
        try FileManager.default.createDirectory(at: scripts, withIntermediateDirectories: true)
        let sandboxLog = fixture.root.appendingPathComponent("sandbox-writable-path.txt")
        let transformOutputLog = fixture.root.appendingPathComponent("transform-output-path.txt")
        let runtimeProfile = fixture.root.appendingPathComponent("runtime-worker.sb")
        try "(version 1)\n(allow default)\n".write(
            to: runtimeProfile,
            atomically: true,
            encoding: .utf8
        )

        let runOffline = scripts.appendingPathComponent("run-offline.sh")
        try """
        #!/bin/sh
        set -eu
        printf '%s\n' "${MLXFAST_OFFLINE_WRITABLE_PATHS:-}" > "${MLXFAST_SANDBOX_LOG:?}"
        exec "$@"
        """.write(to: runOffline, atomically: true, encoding: .utf8)
        try makeExecutable(runOffline)

        try """
        #!/bin/sh
        set -eu
        if [ "${1:-}" = transform ]; then
          shift
          output=""
          while [ "$#" -gt 0 ]; do
            case "$1" in
              --output)
                output="$2"
                shift 2
                ;;
              *)
                shift
                ;;
            esac
          done
          test -n "$output"
          printf '%s\n' "$output" > "$MLXFAST_TRANSFORM_OUTPUT_LOG"
          mkdir -p "$output"
          printf '%s\n' transformed > "$output/config.json"
          exit 0
        fi
        printf '%s\n' '{"score":null,"passed":true,"metrics":{"weights_hash":"hash","weights_file_count":1,"weights_byte_count":12}}'
        """.write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_FORCE_TRANSFORM": "1",
                "MLXFAST_NO_SANDBOX": "0",
                "MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE": runtimeProfile.path,
                "MLXFAST_SANDBOX_LOG": sandboxLog.path,
                "MLXFAST_TRANSFORM_OUTPUT_LOG": transformOutputLog.path,
            ]
        )

        #expect(result.status == 0)
        let writablePath = try String(contentsOf: sandboxLog, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let transformOutputPath = try String(contentsOf: transformOutputLog, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let writableURL = URL(fileURLWithPath: writablePath)
        let writableParent = canonicalExistingPath(
            writableURL.deletingLastPathComponent().path
        )
        let expectedParent = canonicalExistingPath(
            fixture.weights.deletingLastPathComponent().path
        )
        #expect(writableParent == expectedParent)
        #expect(writableURL.lastPathComponent.hasPrefix(".weights.mlxfast-transform."))
        #expect(transformOutputPath == writableURL.appendingPathComponent("weights").path)
        #expect(!FileManager.default.fileExists(atPath: writablePath))
        #expect(
            try String(
                contentsOf: fixture.weights.appendingPathComponent("config.json"),
                encoding: .utf8
            ) == "transformed\n"
        )
    }

    @Test
    func benchmarkScriptPropagatesUnsandboxedTransformVerificationFailure() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let benchmarkInvocation = fixture.root.appendingPathComponent("benchmark-invoked")
        try """
        #!/bin/sh
        if [ "${1:-}" = verify-transform ]; then
          exit 37
        fi
        touch "\(benchmarkInvocation.path)"
        exit 99
        """.write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_SKIP_TRANSFORM": "1",
                "MLXFAST_VERIFY_TRANSFORM": "1",
            ]
        )

        #expect(result.status == 37)
        #expect(!FileManager.default.fileExists(atPath: benchmarkInvocation.path))
    }

    @Test
    func benchmarkScriptRejectsScoreInsideSymlinkedReferenceDirectory() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let referenceAlias = fixture.root.appendingPathComponent("reference-alias")
        try FileManager.default.createSymbolicLink(
            at: referenceAlias,
            withDestinationURL: fixture.reference
        )
        let config = fixture.reference.appendingPathComponent("config.json")
        let original = try Data(contentsOf: config)
        let invocation = fixture.root.appendingPathComponent("swift-invoked")
        try "#!/bin/sh\ntouch \"\(invocation.path)\"\nexit 99\n".write(
            to: fixture.swift,
            atomically: true,
            encoding: .utf8
        )
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_REFERENCE_DIR": referenceAlias.path,
                "MLXFAST_SCORE_PATH": config.path,
                "MLXFAST_SKIP_TRANSFORM": "1",
            ]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("overlaps protected path"))
        #expect(try Data(contentsOf: config) == original)
        #expect(!FileManager.default.fileExists(atPath: invocation.path))
    }

    @Test
    func benchmarkScriptFailsWhenTransformSourceGitInspectionFails() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let fakeBin = fixture.root.appendingPathComponent("bin")
        try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)
        let fakeGit = fakeBin.appendingPathComponent("git")
        try """
        #!/bin/sh
        case " $* " in
          *" rev-parse --is-inside-work-tree "*) exit 0 ;;
          *" --cached --others --exclude-standard "*) exit 19 ;;
          *) exit 0 ;;
        esac
        """.write(to: fakeGit, atomically: true, encoding: .utf8)
        try makeExecutable(fakeGit)
        let invocation = fixture.root.appendingPathComponent("swift-invoked")
        try "#!/bin/sh\ntouch \"\(invocation.path)\"\nexit 99\n".write(
            to: fixture.swift,
            atomically: true,
            encoding: .utf8
        )
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_SKIP_TRANSFORM": "1",
                "PATH": "\(fakeBin.path):/usr/bin:/bin",
            ]
        )

        #expect(result.status == 19)
        #expect(!FileManager.default.fileExists(atPath: invocation.path))
    }

    @Test
    func benchmarkScriptRejectsConflictingLocalModesBeforeInvokingSwift() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let invocation = fixture.root.appendingPathComponent("swift-invoked")
        try "#!/bin/sh\ntouch \"\(invocation.path)\"\n".write(
            to: fixture.swift,
            atomically: true,
            encoding: .utf8
        )
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate", "--local-submit"]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("--local-iterate and --local-submit cannot be used together"))
        #expect(!FileManager.default.fileExists(atPath: invocation.path))
    }

    @Test
    func benchmarkScriptCleansOwnedTemporaryFilesWithoutMaskingFailureStatus() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let temporaryDirectory = fixture.root.appendingPathComponent("tmp")
        let fakeBin = fixture.root.appendingPathComponent("bin")
        try FileManager.default.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)
        let sandboxExec = fakeBin.appendingPathComponent("sandbox-exec")
        try "#!/bin/sh\nexit 99\n".write(
            to: sandboxExec,
            atomically: true,
            encoding: .utf8
        )
        try makeExecutable(sandboxExec)
        try "#!/bin/sh\nexit 37\n".write(
            to: fixture.swift,
            atomically: true,
            encoding: .utf8
        )
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_SKIP_TRANSFORM": "1",
                "MLXFAST_NO_SANDBOX": "0",
                "PATH": "\(fakeBin.path):/usr/bin:/bin",
                "TMPDIR": temporaryDirectory.path,
            ]
        )

        #expect(result.status == 37)
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: temporaryDirectory.path)
        #expect(!leftovers.contains { $0.hasPrefix("mlxfast-runtime-worker.") })
        #expect(!leftovers.contains { $0.hasPrefix("mlxfast-score.") })
    }

    @Test
    func scoredWorkerResponsesExcludeDiagnostics() throws {
        let worker = try String(
            contentsOfFile: "Sources/MLXFastHarness/GemmaRuntimeWorker.swift",
            encoding: .utf8
        )
        let benchmark = try String(
            contentsOfFile: "Sources/MLXFastHarness/GemmaRuntimeBenchmark.swift",
            encoding: .utf8
        )
        let local = try String(
            contentsOfFile: "Sources/MLXFastHarness/GemmaRuntimeLocalIterate.swift",
            encoding: .utf8
        )
        let support = try String(
            contentsOfFile: "Sources/MLXFastHarness/GemmaRuntimeSupport.swift",
            encoding: .utf8
        )
        let decodeStep = try sourceSlice(worker, from: "case \"decode_step\":", to: "case \"phase_diagnostics\":")
        #expect(!decodeStep.contains("currentResidentMemoryGB"))
        #expect(!decodeStep.contains("expertStats"))
        #expect(!worker.contains("let seconds: Double?"))
        #expect(worker.contains("case \"phase_diagnostics\":"))
        #expect(!worker.contains("peakRamGB: currentResidentMemoryGB()"))

        let measuredDecode = try sourceSlice(
            benchmark,
            from: "static func measureWorkerDecode(",
            to: "static func submissionValidationDelayMilliseconds()"
        )
        let timerEnd = try #require(measuredDecode.range(of: "let measuredSeconds = secondsSince(decodePhaseStart)"))
        let diagnostics = try #require(measuredDecode.range(of: "worker.phaseDiagnostics()"))
        #expect(timerEnd.lowerBound < diagnostics.lowerBound)
        let diagnosticResponse = try sourceSlice(
            worker,
            from: "case \"phase_diagnostics\":",
            to: "default:"
        )
        #expect(diagnosticResponse.contains("peakResidentMemoryGB()"))
        #expect(!diagnosticResponse.contains("currentResidentMemoryGB()"))
        #expect(support.contains("Double(info.resident_size_max)"))

        let localMeasuredTime = try sourceSlice(
            local,
            from: "let timingWallSeconds = secondsSince(timingWallStart)",
            to: "correctnessReport = timing.correctness"
        )
        #expect(localMeasuredTime.contains("timedSeconds = timing.prefillSecondsPerToken"))
        #expect(localMeasuredTime.contains("localCase.promptTokens.count * options.timingRepeats"))
        #expect(localMeasuredTime.contains("timing.decode.secondsPerToken"))
        #expect(localMeasuredTime.contains("options.benchmarkDecodeSteps * options.timingRepeats"))
        #expect(localMeasuredTime.contains("correctnessSeconds = timedSeconds"))
        #expect(!localMeasuredTime.contains("timedSeconds = secondsSince"))
        #expect(!local.contains("let warmupCache = Gemma4ModelCache"))
        #expect(local.components(separatedBy: "runLocalPhaseCoolGate(phase:").count == 5)
    }
}

private struct BenchmarkScriptFixture {
    let root: URL
    let workingDirectory: URL
    let weights: URL
    let reference: URL
    let golden: URL
    let swift: URL
    let score: URL
    let integrity: URL
}

private func makeBenchmarkScriptFixture(
    nestedWorkingDirectory: Bool = false
) throws -> BenchmarkScriptFixture {
    let root = try makeTemporaryDirectory()
    let workingDirectory = nestedWorkingDirectory ? root.appendingPathComponent("run") : root
    let weights = root.appendingPathComponent("weights")
    let reference = root.appendingPathComponent("reference")
    let golden = root.appendingPathComponent("golden.json")
    let swift = root.appendingPathComponent("mlxfast-swift")
    let score = root.appendingPathComponent("score.json")
    let integrity = root.appendingPathComponent("integrity.json")
    try FileManager.default.createDirectory(at: workingDirectory, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    let coreSources = workingDirectory.appendingPathComponent("Sources/MLXFastCore")
    let transformSources = workingDirectory.appendingPathComponent("Sources/MLXFastTransform")
    try FileManager.default.createDirectory(at: coreSources, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: transformSources, withIntermediateDirectories: true)
    try "// fixture".write(
        to: workingDirectory.appendingPathComponent("Package.swift"),
        atomically: true,
        encoding: .utf8
    )
    try "{}".write(
        to: workingDirectory.appendingPathComponent("Package.resolved"),
        atomically: true,
        encoding: .utf8
    )
    try "// fixture".write(
        to: coreSources.appendingPathComponent("Fixture.swift"),
        atomically: true,
        encoding: .utf8
    )
    try "// fixture".write(
        to: transformSources.appendingPathComponent("Fixture.swift"),
        atomically: true,
        encoding: .utf8
    )
    try "{}".write(to: weights.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
    try "{}".write(to: reference.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
    try "{}".write(to: golden, atomically: true, encoding: .utf8)
    return BenchmarkScriptFixture(
        root: root,
        workingDirectory: workingDirectory,
        weights: weights,
        reference: reference,
        golden: golden,
        swift: swift,
        score: score,
        integrity: integrity
    )
}

private func runBenchmarkScript(
    fixture: BenchmarkScriptFixture,
    arguments: [String],
    environment: [String: String] = [:]
) throws -> (status: Int32, stderr: String) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = [
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("benchmark.sh").path,
    ] + arguments
    process.currentDirectoryURL = fixture.workingDirectory
    process.environment = ProcessInfo.processInfo.environment.merging([
        "MLXFAST_NO_SANDBOX": "1",
        "MLXFAST_LOCAL_COOL_GATE": "0",
        "MLXFAST_SWIFT_BIN": fixture.swift.path,
        "MLXFAST_WEIGHTS_PATH": fixture.weights.path,
        "MLXFAST_REFERENCE_DIR": fixture.reference.path,
        "MLXFAST_CORRECTNESS_GOLDEN_PATH": fixture.golden.path,
        "MLXFAST_SCORE_PATH": fixture.score.path,
        "MLXFAST_INTEGRITY_PATH": fixture.integrity.path,
    ].merging(environment) { _, new in new }) { _, new in new }
    let stderr = Pipe()
    process.standardError = stderr
    try process.run()
    let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return (
        process.terminationStatus,
        String(data: stderrData, encoding: .utf8) ?? ""
    )
}

private func makeTemporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func makeExecutable(_ url: URL) throws {
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
}

private func canonicalExistingPath(_ path: String) -> String {
    path.withCString { source in
        var buffer = [CChar](repeating: 0, count: Int(PATH_MAX))
        guard realpath(source, &buffer) != nil else {
            return path
        }
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }
}

private func initializeGitIndex(at directory: URL) throws {
    try runGit(["init", "-q"], at: directory)
    try runGit(["add", "--", "Package.swift", "Package.resolved", "Sources"], at: directory)
}

private func runGit(_ arguments: [String], at directory: URL) throws {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
    process.arguments = arguments
    process.currentDirectoryURL = directory
    let errorPipe = Pipe()
    process.standardError = errorPipe
    try process.run()
    let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        throw NSError(
            domain: "BenchmarkSafetyTests.git",
            code: Int(process.terminationStatus),
            userInfo: [
                NSLocalizedDescriptionKey: String(data: errorData, encoding: .utf8) ?? "git failed",
            ]
        )
    }
}

private func restoreEnvironment(_ name: String, value: String?) {
    if let value {
        setenv(name, value, 1)
    } else {
        unsetenv(name)
    }
}

private func sourceSlice(_ source: String, from start: String, to end: String) throws -> String {
    let startRange = try #require(source.range(of: start))
    let endRange = try #require(source.range(of: end, range: startRange.upperBound..<source.endIndex))
    return String(source[startRange.lowerBound..<endRange.lowerBound])
}
