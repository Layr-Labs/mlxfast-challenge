import Darwin
import CryptoKit
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
    func benchmarkScriptPreservesUnmanagedExternalWeightsDirectory() throws {
        let fixture = try makeBenchmarkScriptFixture(nestedWorkingDirectory: true)
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let sentinel = fixture.weights.appendingPathComponent("do-not-delete.txt")
        try "sentinel".write(to: sentinel, atomically: true, encoding: .utf8)
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
            environment: ["MLXFAST_FORCE_TRANSFORM": "1"]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("existing directories outside the workspace must be MLXFast-managed"))
        #expect(try String(contentsOf: sentinel, encoding: .utf8) == "sentinel")
        #expect(!FileManager.default.fileExists(atPath: invocation.path))
    }

    @Test
    func benchmarkScriptCanInstallIntoEmptyExternalWeightsDirectory() throws {
        let fixture = try makeBenchmarkScriptFixture(nestedWorkingDirectory: true)
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        try FileManager.default.removeItem(
            at: fixture.weights.appendingPathComponent("config.json")
        )
        try """
        #!/bin/sh
        set -eu
        if [ "${1:-}" = transform ]; then
          shift
          output=""
          while [ "$#" -gt 0 ]; do
            if [ "$1" = --output ]; then
              output="$2"
              break
            fi
            shift
          done
          test -n "$output"
          mkdir -p "$output"
          printf '%s\n' '{"transformed":true}' > "$output/config.json"
          printf '%s\n' '{"weight_map":{"tensor":"model.safetensors"}}' > "$output/model.safetensors.index.json"
          printf '\\100\\000\\000\\000\\000\\000\\000\\000' > "$output/model.safetensors"
          printf '%s' '{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}      ' >> "$output/model.safetensors"
          printf '\\001' >> "$output/model.safetensors"
          exit 0
        fi
        printf '%s\n' '{"score":null,"passed":true,"metrics":{"weights_hash":"hash","weights_file_count":1,"weights_byte_count":12}}'
        """.write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: ["MLXFAST_FORCE_TRANSFORM": "1"]
        )

        #expect(result.status == 0)
        #expect(
            try String(
                contentsOf: fixture.weights.appendingPathComponent("config.json"),
                encoding: .utf8
            ) == "{\"transformed\":true}\n"
        )
        #expect(
            FileManager.default.fileExists(
                atPath: fixture.weights.appendingPathComponent(".benchmark-source.sha256").path
            )
        )
    }

    @Test
    func benchmarkScriptRejectsReservedTransformMetadataSymlink() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let sentinel = fixture.root.appendingPathComponent("do-not-touch.txt")
        try "sentinel".write(to: sentinel, atomically: true, encoding: .utf8)
        try """
        #!/bin/sh
        set -eu
        if [ "${1:-}" = transform ]; then
          shift
          output=""
          while [ "$#" -gt 0 ]; do
            if [ "$1" = --output ]; then
              output="$2"
              break
            fi
            shift
          done
          test -n "$output"
          mkdir -p "$output"
          printf '%s\n' transformed > "$output/config.json"
          ln -s "\(sentinel.path)" "$output/.benchmark-source.sha256"
          exit 0
        fi
        exit 99
        """.write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: ["MLXFAST_FORCE_TRANSFORM": "1"]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("created reserved .benchmark-source.sha256 path"))
        #expect(try String(contentsOf: sentinel, encoding: .utf8) == "sentinel")
    }

    @Test
    func benchmarkScriptPreservesWeightsWhenTransformOutputIsIncomplete() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let originalConfig = try Data(
            contentsOf: fixture.weights.appendingPathComponent("config.json")
        )
        try """
        #!/bin/sh
        set -eu
        shift
        output=""
        while [ "$#" -gt 0 ]; do
          if [ "$1" = --output ]; then
            output="$2"
            break
          fi
          shift
        done
        test -n "$output"
        mkdir -p "$output"
        printf '%s\n' incomplete > "$output/config.json"
        """.write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: ["MLXFAST_FORCE_TRANSFORM": "1"]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("missing regular config/index files"))
        #expect(
            try Data(contentsOf: fixture.weights.appendingPathComponent("config.json"))
                == originalConfig
        )
    }

    @Test
    func benchmarkScriptValidatesStagedJSONIndexAndSafetensorsBeforeReplacement() throws {
        let cases = [
            ("malformed-config", "config.json is not a JSON object"),
            ("missing-shard", "index references missing shard"),
            ("invalid-shard", "safetensors shard is too small"),
            ("unsupported-dtype", "safetensors shard has an invalid header"),
            ("byte-span-mismatch", "safetensors shard has an invalid header"),
            ("overlapping-ranges", "safetensors shard has an invalid header"),
        ]
        for (caseName, expectedError) in cases {
            let fixture = try makeBenchmarkScriptFixture()
            defer { try? FileManager.default.removeItem(at: fixture.root) }
            let originalConfig = try Data(
                contentsOf: fixture.weights.appendingPathComponent("config.json")
            )
            try """
            #!/bin/sh
            set -eu
            shift
            output=""
            while [ "$#" -gt 0 ]; do
              if [ "$1" = --output ]; then
                output="$2"
                break
              fi
              shift
            done
            test -n "$output"
            mkdir -p "$output"
            printf '%s\n' '{}' > "$output/config.json"
            printf '%s\n' '{"weight_map":{"tensor":"model.safetensors"}}' > "$output/model.safetensors.index.json"
            printf '\\100\\000\\000\\000\\000\\000\\000\\000' > "$output/model.safetensors"
            printf '%s' '{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}      ' >> "$output/model.safetensors"
            printf '\\001' >> "$output/model.safetensors"
            case "$MLXFAST_TEST_BAD_TRANSFORM" in
              malformed-config)
                printf '%s\n' 'not-json' > "$output/config.json"
                ;;
              missing-shard)
                printf '%s\n' '{"weight_map":{"tensor":"missing.safetensors"}}' > "$output/model.safetensors.index.json"
                ;;
              invalid-shard)
                printf '%s\n' invalid > "$output/model.safetensors"
                ;;
              unsupported-dtype)
                printf '\\100\\000\\000\\000\\000\\000\\000\\000' > "$output/model.safetensors"
                printf '%s' '{"tensor":{"dtype":"XX","shape":[1],"data_offsets":[0,1]}}      ' >> "$output/model.safetensors"
                printf '\\001' >> "$output/model.safetensors"
                ;;
              byte-span-mismatch)
                printf '\\100\\000\\000\\000\\000\\000\\000\\000' > "$output/model.safetensors"
                printf '%s' '{"tensor":{"dtype":"U8","shape":[2],"data_offsets":[0,1]}}      ' >> "$output/model.safetensors"
                printf '\\001' >> "$output/model.safetensors"
                ;;
              overlapping-ranges)
                printf '%s\n' '{"weight_map":{"a":"model.safetensors","b":"model.safetensors"}}' > "$output/model.safetensors.index.json"
                printf '\\160\\000\\000\\000\\000\\000\\000\\000' > "$output/model.safetensors"
                printf '%s' '{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}       ' >> "$output/model.safetensors"
                printf '\\001' >> "$output/model.safetensors"
                ;;
            esac
            """.write(to: fixture.swift, atomically: true, encoding: .utf8)
            try makeExecutable(fixture.swift)

            let result = try runBenchmarkScript(
                fixture: fixture,
                arguments: ["--local-iterate"],
                environment: [
                    "MLXFAST_FORCE_TRANSFORM": "1",
                    "MLXFAST_TEST_BAD_TRANSFORM": caseName,
                ]
            )

            #expect(result.status != 0, "case \(caseName)")
            #expect(result.stderr.contains(expectedError), "case \(caseName)")
            #expect(
                try Data(contentsOf: fixture.weights.appendingPathComponent("config.json"))
                    == originalConfig,
                "case \(caseName)"
            )
        }
    }

    @Test
    func benchmarkScriptRejectsReservedTransformRollbackSymlink() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let sentinelDirectory = fixture.root.appendingPathComponent("rollback-target")
        try FileManager.default.createDirectory(
            at: sentinelDirectory,
            withIntermediateDirectories: true
        )
        let sentinel = sentinelDirectory.appendingPathComponent("do-not-touch.txt")
        try "sentinel".write(to: sentinel, atomically: true, encoding: .utf8)
        let originalConfig = try Data(
            contentsOf: fixture.weights.appendingPathComponent("config.json")
        )
        try """
        #!/bin/sh
        set -eu
        shift
        output=""
        while [ "$#" -gt 0 ]; do
          if [ "$1" = --output ]; then
            output="$2"
            break
          fi
          shift
        done
        test -n "$output"
        mkdir -p "$output"
        printf '%s\n' '{"transformed":true}' > "$output/config.json"
        printf '%s\n' '{"weight_map":{"tensor":"model.safetensors"}}' > "$output/model.safetensors.index.json"
        printf '\\100\\000\\000\\000\\000\\000\\000\\000' > "$output/model.safetensors"
        printf '%s' '{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}      ' >> "$output/model.safetensors"
        printf '\\001' >> "$output/model.safetensors"
        ln -s "\(sentinelDirectory.path)" "$output/../previous-weights"
        """.write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: ["MLXFAST_FORCE_TRANSFORM": "1"]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("created reserved rollback path"))
        #expect(try String(contentsOf: sentinel, encoding: .utf8) == "sentinel")
        #expect(
            try Data(contentsOf: fixture.weights.appendingPathComponent("config.json"))
                == originalConfig
        )
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
    func benchmarkScriptPreservesVerificationTemporaryRootContents() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let scratchRoot = fixture.root.appendingPathComponent("verification-root")
        try FileManager.default.createDirectory(at: scratchRoot, withIntermediateDirectories: true)
        let sentinel = scratchRoot.appendingPathComponent("unrelated.txt")
        try "sentinel".write(to: sentinel, atomically: true, encoding: .utf8)
        try """
        #!/bin/sh
        if [ "${1:-}" = verify-transform ]; then
          test -d "${7:-}"
          exit 0
        fi
        printf '%s\n' '{"score":null,"passed":true,"metrics":{"weights_hash":"hash","weights_file_count":1,"weights_byte_count":12}}'
        """.write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_SKIP_TRANSFORM": "1",
                "MLXFAST_VERIFY_TRANSFORM": "1",
                "MLXFAST_VERIFY_TRANSFORM_TMP_PARENT": scratchRoot.path,
            ]
        )

        #expect(result.status == 0)
        #expect(try String(contentsOf: sentinel, encoding: .utf8) == "sentinel")
        #expect(try FileManager.default.contentsOfDirectory(atPath: scratchRoot.path) == ["unrelated.txt"])
    }

    @Test
    func artifactStagingRejectsTraversalWithoutDeletingDestination() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let artifactRoot = root.appendingPathComponent("mlxfast-artifacts-run")
        let victim = root.appendingPathComponent("victim")
        try FileManager.default.createDirectory(at: artifactRoot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: victim, withIntermediateDirectories: true)
        let sentinel = victim.appendingPathComponent("do-not-delete.txt")
        try "sentinel".write(to: sentinel, atomically: true, encoding: .utf8)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".github/scripts/stage-benchmark-artifacts.sh").path,
            artifactRoot.appendingPathComponent("../victim").path,
            sentinel.path,
        ]
        process.environment = ProcessInfo.processInfo.environment.merging([
            "RUNNER_TEMP": root.path,
        ]) { _, new in new }
        let stderr = Pipe()
        process.standardError = stderr
        try process.run()
        let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()

        #expect(process.terminationStatus != 0)
        #expect((String(data: stderrData, encoding: .utf8) ?? "").contains("artifact destination"))
        #expect(try String(contentsOf: sentinel, encoding: .utf8) == "sentinel")
    }

    @Test
    func artifactStagingRejectsSymlinkedRunRoot() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let outside = root.appendingPathComponent("outside")
        let upload = outside.appendingPathComponent("benchmark-results")
        try FileManager.default.createDirectory(at: upload, withIntermediateDirectories: true)
        let sentinel = upload.appendingPathComponent("do-not-delete.txt")
        try "sentinel".write(to: sentinel, atomically: true, encoding: .utf8)
        let runRoot = root.appendingPathComponent("mlxfast-artifacts-run")
        try FileManager.default.createSymbolicLink(at: runRoot, withDestinationURL: outside)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".github/scripts/stage-benchmark-artifacts.sh").path,
            runRoot.appendingPathComponent("benchmark-results").path,
            "sentinel.txt=\(sentinel.path)",
        ]
        process.environment = ProcessInfo.processInfo.environment.merging([
            "RUNNER_TEMP": root.path,
        ]) { _, new in new }
        let stderr = Pipe()
        process.standardError = stderr
        try process.run()
        let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()

        #expect(process.terminationStatus != 0)
        #expect((String(data: stderrData, encoding: .utf8) ?? "").contains("must not be a symlink"))
        #expect(try String(contentsOf: sentinel, encoding: .utf8) == "sentinel")
    }

    @Test
    func artifactStagingCreatesOwnedRunRootAndCopiesFiles() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("score-source.json")
        try "{}".write(to: source, atomically: true, encoding: .utf8)
        let destination = root
            .appendingPathComponent("mlxfast-artifacts-run")
            .appendingPathComponent("benchmark-results")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".github/scripts/stage-benchmark-artifacts.sh").path,
            destination.path,
            "score.json=\(source.path)",
        ]
        process.environment = ProcessInfo.processInfo.environment.merging([
            "RUNNER_TEMP": root.path,
        ]) { _, new in new }
        try process.run()
        process.waitUntilExit()

        #expect(process.terminationStatus == 0)
        #expect(
            try String(
                contentsOf: destination.appendingPathComponent("score.json"),
                encoding: .utf8
            ) == "{}"
        )
    }

    @Test
    func offlineRunnerRemovesProfilesAndPreservesCommandFailure() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let bin = root.appendingPathComponent("bin")
        let temporaryDirectory = root.appendingPathComponent("tmp")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: true
        )
        let curl = bin.appendingPathComponent("curl")
        try "#!/bin/sh\nexit 99\n".write(to: curl, atomically: true, encoding: .utf8)
        try makeExecutable(curl)
        let sandboxExec = bin.appendingPathComponent("sandbox-exec")
        try """
        #!/bin/sh
        set -eu
        test "$1" = -f
        shift 2
        case "$1" in
          */curl) exit 1 ;;
        esac
        exec "$@"
        """.write(to: sandboxExec, atomically: true, encoding: .utf8)
        try makeExecutable(sandboxExec)
        let command = root.appendingPathComponent("command.sh")
        try "#!/bin/sh\nexit 37\n".write(to: command, atomically: true, encoding: .utf8)
        try makeExecutable(command)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".github/scripts/run-offline.sh").path,
            command.path,
        ]
        process.environment = ProcessInfo.processInfo.environment.merging([
            "PATH": "\(bin.path):/usr/bin:/bin",
            "TMPDIR": temporaryDirectory.path,
        ]) { _, new in new }
        try process.run()
        process.waitUntilExit()

        #expect(process.terminationStatus == 37)
        let leftovers = try FileManager.default.contentsOfDirectory(
            atPath: temporaryDirectory.path
        )
        #expect(!leftovers.contains { $0.hasPrefix("mlxfast-offline.") })
    }

    @Test
    func offlineRunnerForwardsTerminationAndRemovesProfiles() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let bin = root.appendingPathComponent("bin")
        let temporaryDirectory = root.appendingPathComponent("tmp")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: true
        )
        let curl = bin.appendingPathComponent("curl")
        try "#!/bin/sh\nexit 99\n".write(to: curl, atomically: true, encoding: .utf8)
        try makeExecutable(curl)
        let sandboxExec = bin.appendingPathComponent("sandbox-exec")
        try """
        #!/bin/sh
        set -eu
        test "$1" = -f
        shift 2
        case "$1" in
          */curl) exit 1 ;;
        esac
        exec "$@"
        """.write(to: sandboxExec, atomically: true, encoding: .utf8)
        try makeExecutable(sandboxExec)
        let childPIDPath = root.appendingPathComponent("child.pid")
        let terminatedPath = root.appendingPathComponent("child-terminated")
        let command = root.appendingPathComponent("command.sh")
        try """
        #!/bin/sh
        trap 'printf terminated > "\(terminatedPath.path)"; exit 0' TERM
        printf '%s\n' "$$" > "\(childPIDPath.path)"
        while :; do :; done
        """.write(to: command, atomically: true, encoding: .utf8)
        try makeExecutable(command)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".github/scripts/run-offline.sh").path,
            command.path,
        ]
        process.environment = ProcessInfo.processInfo.environment.merging([
            "PATH": "\(bin.path):/usr/bin:/bin",
            "TMPDIR": temporaryDirectory.path,
        ]) { _, new in new }
        try process.run()
        for _ in 0..<200 where !FileManager.default.fileExists(atPath: childPIDPath.path) {
            Thread.sleep(forTimeInterval: 0.01)
        }
        let childPID = try #require(
            Int32(String(contentsOf: childPIDPath, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines))
        )

        process.terminate()
        process.waitUntilExit()

        #expect(FileManager.default.fileExists(atPath: terminatedPath.path))
        #expect(kill(childPID, 0) != 0)
        let leftovers = try FileManager.default.contentsOfDirectory(
            atPath: temporaryDirectory.path
        )
        #expect(!leftovers.contains { $0.hasPrefix("mlxfast-offline.") })
    }

    @Test
    func pairedTimingOverlayRejectsWeightsChangedAfterGates() throws {
        let result = try runPairedTimingOverlay(
            gatesHarnessHash: String(repeating: "c", count: 64),
            candidateHarnessHash: String(repeating: "c", count: 64),
            gatesWeightsHash: String(repeating: "a", count: 64),
            candidateWeightsHash: String(repeating: "b", count: 64)
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("candidate timing score failed the pre-merge checks"))
    }

    @Test
    func pairedTimingOverlayRejectsHarnessChangedAfterGates() throws {
        let weightsHash = String(repeating: "c", count: 64)
        let result = try runPairedTimingOverlay(
            gatesHarnessHash: String(repeating: "a", count: 64),
            candidateHarnessHash: String(repeating: "b", count: 64),
            gatesWeightsHash: weightsHash,
            candidateWeightsHash: weightsHash
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("candidate timing score failed the pre-merge checks"))
    }

    @Test
    func pairedTimingOverlayAcceptsMatchingHarnessAndWeightsIdentity() throws {
        let harnessHash = String(repeating: "c", count: 64)
        let weightsHash = String(repeating: "d", count: 64)
        let result = try runPairedTimingOverlay(
            gatesHarnessHash: harnessHash,
            candidateHarnessHash: harnessHash,
            gatesWeightsHash: weightsHash,
            candidateWeightsHash: weightsHash
        )

        #expect(result.status == 0)
        #expect(result.score?["passed"] as? Bool == true)
        #expect(result.score?["score"] as? Double == 1.0)
    }

    @Test
    func finalArtifactValidatorRejectsScoreIntegrityIdentityMismatches() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let score = root.appendingPathComponent("score.json")
        let integrity = root.appendingPathComponent("benchmark-integrity.json")
        let golden = root.appendingPathComponent("correctness_golden.json")
        let goldenHash = String(repeating: "a", count: 64)
        let weightsHash = String(repeating: "b", count: 64)
        let harnessHash = String(repeating: "c", count: 64)
        let commit = String(repeating: "d", count: 40)

        let metrics: [String: Any] = [
            "actual_token": NSNull(), "bandwidth_gb_per_token": 0,
            "bandwidth_source": "ram_resident_model", "baseline_decode_seconds_per_token": 1,
            "baseline_prefill_seconds_per_token": 1, "benchmark_wall_seconds": 3,
            "case_count": 1, "checked_steps": 64, "commit": commit,
            "correctness_seconds": 1, "decode_seconds_per_token": 1,
            "decode_speedup": 1,
            "decode_speedup_floor": NSDecimalNumber(string: "0.95"), "error": "",
            "expected_token": NSNull(), "expert_bytes_read": 0,
            "expert_cache_evictions": 0, "expert_cache_hits": 0,
            "expert_cache_misses": 0, "expert_hit_rate": 0,
            "expert_peak_cached_tensors": 0, "expert_read_seconds": 0,
            "first_failing_case": NSNull(), "first_failing_layer": NSNull(),
            "first_failing_step": NSNull(), "golden_hash": goldenHash,
            "gpqa_ttft_case_count": 1, "gpqa_ttft_max_seconds": 1,
            "gpqa_ttft_p50_seconds": 1, "gpqa_ttft_pass_count": 1,
            "gpqa_ttft_passed": true, "gpqa_ttft_seconds": 1,
            "gpqa_ttft_source": "hidden_gpqa_first_token", "harness_hash": harnessHash,
            "max_abs_diff": 0, "num_layers": 60, "partial_result": false,
            "passed_correctness": true, "peak_ram_gb": 1,
            "passed_decode_speedup_floor": true, "passed_prefill_speedup_floor": true,
            "prefill_seconds_per_token": 1, "prefill_speedup": 1,
            "prefill_speedup_floor": NSDecimalNumber(string: "0.95"), "preflight_seconds": 1,
            "process_resident_memory_gb": 1, "runtime": "swift",
            "semantic_gpqa_case_count": 1, "semantic_gpqa_model": "fixture",
            "semantic_gpqa_pass_count": 1, "semantic_gpqa_passed": true,
            "timed_benchmark_seconds": 1, "timestamp": "2026-01-01T00:00:00Z",
            "weights_byte_count": 1, "weights_file_count": 1,
            "weights_hash": weightsHash,
        ]
        try JSONSerialization.data(withJSONObject: [
            "metrics": metrics, "passed": true, "score": 1,
        ]).write(to: score)
        let scoreHash = SHA256.hash(data: try Data(contentsOf: score))
            .map { String(format: "%02x", $0) }.joined()
        try "\(scoreHash)  score.json\n".write(
            to: URL(fileURLWithPath: score.path + ".sha256"), atomically: true, encoding: .utf8
        )
        try "{}".write(to: golden, atomically: true, encoding: .utf8)
        try "\(goldenHash)  correctness_golden.json\n".write(
            to: URL(fileURLWithPath: golden.path + ".sha256"), atomically: true, encoding: .utf8
        )
        try "2\n".write(
            to: URL(fileURLWithPath: golden.path + ".bytes"), atomically: true, encoding: .utf8
        )

        let baseIntegrity: [String: Any] = [
            "golden_path": "[private]", "golden_sha256": goldenHash,
            "score_path": "score.json", "score_sha256": scoreHash,
            "transform_source_sha256": String(repeating: "e", count: 64),
            "weights_byte_count": 1, "weights_file_count": 1,
            "weights_path": "weights", "weights_sha256": weightsHash,
        ]
        try JSONSerialization.data(withJSONObject: baseIntegrity).write(to: integrity)
        let accepted = try runFinalArtifactValidator(root: root, golden: golden, commit: commit)
        #expect(accepted.status == 0, Comment(rawValue: accepted.stderr))

        let mismatches: [(String, Any)] = [
            ("weights_sha256", String(repeating: "f", count: 64)),
            ("weights_file_count", 2),
            ("weights_byte_count", 2),
            ("golden_sha256", String(repeating: "f", count: 64)),
        ]
        for (field, value) in mismatches {
            var changed = baseIntegrity
            changed[field] = value
            try JSONSerialization.data(withJSONObject: changed).write(to: integrity)
            #expect(
                try runFinalArtifactValidator(root: root, golden: golden, commit: commit).status != 0,
                "validator accepted mismatched \(field)"
            )
        }
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
          printf '%s\n' '{"transformed":true}' > "$output/config.json"
          printf '%s\n' '{"weight_map":{"tensor":"model.safetensors"}}' > "$output/model.safetensors.index.json"
          printf '\\100\\000\\000\\000\\000\\000\\000\\000' > "$output/model.safetensors"
          printf '%s' '{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}      ' >> "$output/model.safetensors"
          printf '\\001' >> "$output/model.safetensors"
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
            ) == "{\"transformed\":true}\n"
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
    let metallib: URL
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
    let metallib = root.appendingPathComponent("mlx.metallib")
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
    try "fixture metallib".write(to: metallib, atomically: true, encoding: .utf8)
    return BenchmarkScriptFixture(
        root: root,
        workingDirectory: workingDirectory,
        weights: weights,
        reference: reference,
        golden: golden,
        swift: swift,
        metallib: metallib,
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
        "MLXFAST_MLX_METALLIB": fixture.metallib.path,
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

private func runPairedTimingOverlay(
    gatesHarnessHash: String,
    candidateHarnessHash: String,
    gatesWeightsHash: String,
    candidateWeightsHash: String
) throws -> (status: Int32, stderr: String, score: [String: Any]?) {
    let root = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let gates = root.appendingPathComponent("score.json")
    let candidate = root.appendingPathComponent("candidate.json")
    let results = root.appendingPathComponent("results.json")
    let integrity = root.appendingPathComponent("integrity.json")
    let commit = String(repeating: "c", count: 40)

    let commonExpertMetrics: [String: Any] = [
        "expert_bytes_read": 0,
        "expert_cache_hits": 0,
        "expert_cache_misses": 0,
        "expert_cache_evictions": 0,
        "expert_read_seconds": 0,
        "expert_peak_cached_tensors": 0,
    ]
    var gatesMetrics = commonExpertMetrics
    gatesMetrics.merge([
        "harness_hash": gatesHarnessHash,
        "weights_hash": gatesWeightsHash,
        "weights_file_count": 1,
        "weights_byte_count": 1,
        "benchmark_wall_seconds": 1,
        "peak_ram_gb": 1,
        "process_resident_memory_gb": 1,
        "partial_result": true,
    ]) { _, new in new }
    var candidateMetrics = commonExpertMetrics
    candidateMetrics.merge([
        "commit": commit,
        "first_failing_case": NSNull(),
        "first_failing_step": NSNull(),
        "expected_token": NSNull(),
        "actual_token": NSNull(),
        "bandwidth_source": "ram_resident_model",
        "bandwidth_gb_per_token": 0,
        "timed_benchmark_seconds": 1,
        "benchmark_wall_seconds": 1,
        "peak_ram_gb": 1,
        "process_resident_memory_gb": 1,
        "harness_hash": candidateHarnessHash,
        "weights_hash": candidateWeightsHash,
        "weights_file_count": 1,
        "weights_byte_count": 1,
    ]) { _, new in new }

    try JSONSerialization.data(withJSONObject: [
        "passed": true,
        "score": NSNull(),
        "metrics": gatesMetrics,
    ]).write(to: gates)
    try JSONSerialization.data(withJSONObject: [
        "passed": true,
        "score": NSNull(),
        "metrics": candidateMetrics,
    ]).write(to: candidate)
    try JSONSerialization.data(withJSONObject: [
        "mode": "paired",
        "paired": ["decode_speedup": 1, "prefill_speedup": 1],
        "candidate": [
            "decode_seconds_per_token": 1,
            "prefill_seconds_per_token": 1,
            "verdict": "ACCEPT",
        ],
        "baseline": [
            "decode_seconds_per_token": 1,
            "prefill_seconds_per_token": 1,
            "verdict": "ACCEPT",
        ],
    ]).write(to: results)
    try Data("{}".utf8).write(to: integrity)

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = [
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".github/scripts/overlay-paired-timing.sh").path,
    ]
    process.environment = ProcessInfo.processInfo.environment.merging([
        "MLXFAST_SCORE_PATH": gates.path,
        "MLXFAST_CANDIDATE_SCORE_PATH": candidate.path,
        "MLXFAST_MEASURE_RESULTS_PATH": results.path,
        "MLXFAST_INTEGRITY_PATH": integrity.path,
        "MLXFAST_EXPECTED_COMMIT": commit,
        "MLXFAST_DECODE_SPEEDUP_FLOOR": "0.95",
        "MLXFAST_PREFILL_SPEEDUP_FLOOR": "0.95",
    ]) { _, new in new }
    let stderr = Pipe()
    process.standardError = stderr
    try process.run()
    let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()

    let scoreObject = try? JSONSerialization.jsonObject(with: Data(contentsOf: gates))
        as? [String: Any]
    return (
        process.terminationStatus,
        String(data: stderrData, encoding: .utf8) ?? "",
        scoreObject
    )
}

private func runFinalArtifactValidator(
    root: URL,
    golden: URL,
    commit: String
) throws -> (status: Int32, stderr: String) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = [
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".github/scripts/validate-benchmark-artifacts.sh").path,
    ]
    process.currentDirectoryURL = root
    process.environment = ProcessInfo.processInfo.environment.merging([
        "MLXFAST_SCORE_PATH": "score.json",
        "MLXFAST_INTEGRITY_PATH": "benchmark-integrity.json",
        "MLXFAST_CORRECTNESS_GOLDEN_PATH": golden.path,
        "MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256": String(repeating: "a", count: 64),
        "MLXFAST_EXPECTED_CORRECTNESS_STEPS": "64",
        "MLXFAST_EXPECTED_CORRECTNESS_CASES": "1",
        "MLXFAST_EXPECTED_CORRECTNESS_CHECKED_STEPS": "64",
        "MLXFAST_GPQA_TTFT_CASE_COUNT": "1",
        "MLXFAST_SEMANTIC_GPQA_CASE_COUNT": "1",
        "MLXFAST_SEMANTIC_GPQA_MIN_PASS": "1",
        "MLXFAST_SEMANTIC_GPQA_REQUIRED": "1",
        "MLXFAST_EXPECTED_COMMIT": commit,
    ]) { _, new in new }
    process.standardOutput = FileHandle.nullDevice
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
