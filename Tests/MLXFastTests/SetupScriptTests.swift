import Foundation
import Testing

@Test
func setupScriptCoordinatesCacheAndMetallibState() throws {
    let setup = try String(contentsOfFile: "setup.sh", encoding: .utf8)
    let metallibBuilder = try String(
        contentsOfFile: "tools/build-mlx-metallib.sh",
        encoding: .utf8
    )

    #expect(setup.contains("CANONICAL_REFERENCE_LOCK_BASE="))
    #expect(setup.contains("REFERENCE_CACHE_MUTATION_LOCK_DIR=\"${MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR:-${CANONICAL_REFERENCE_LOCK_BASE}.mlxfast-setup.lock}\""))
    #expect(setup.contains("METALLIB_BUILD_STATE=\"not_started\""))
    #expect(!setup.contains("existing-stamp-only"))
    #expect(setup.contains("acquire_reference_cache_mutation_lock"))
    #expect(setup.contains("recover_stale_reference_cache_mutation_lock"))
    #expect(setup.contains("mktemp \"${lock_path}.tmp.XXXXXX\""))
    #expect(setup.contains("metal_toolchain_identifier"))
    #expect(setup.contains("export TOOLCHAINS=\"${TOOLCHAINS:-${identifier}}\""))
    #expect(metallibBuilder.contains("-DMLX_BUILD_GGUF=OFF"))
    #expect(metallibBuilder.contains("export CLANG_MODULE_CACHE_PATH="))
    #expect(metallibBuilder.contains("HOME=\"${METAL_COMPILER_HOME}\" \"${CMAKE_BIN}\""))
}

@Test
func setupStartsMetallibBuildOnlyOnceInSynchronousAndParallelModes() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    for parallel in ["0", "1"] {
        let modeRoot = root.appendingPathComponent("mode-\(parallel)")
        try FileManager.default.createDirectory(at: modeRoot, withIntermediateDirectories: true)
        let result = try runSetupBash(
            """
            eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
            build_mlx_metallib() {
              printf 'build\n' >> "${BUILD_LOG}"
              sleep 0.1
              mkdir -p "$(dirname "${MLX_METALLIB}")"
              : > "${MLX_METALLIB}"
            }

            # These repeated calls model the warm-cache verification, fresh
            # download, and repair call sites that can all request the build.
            start_mlx_metallib_build
            start_mlx_metallib_build
            start_mlx_metallib_build
            wait_for_mlx_metallib_build
            start_mlx_metallib_build
            wait_for_mlx_metallib_build

            [[ "${METALLIB_BUILD_STATE}" == "completed" ]]
            [[ "$(wc -l < "${BUILD_LOG}" | tr -d ' ')" == "1" ]]
            """,
            environment: [
                "REPO_ROOT": FileManager.default.currentDirectoryPath,
                "BUILD_LOG": modeRoot.appendingPathComponent("build.log").path,
                "MLXFAST_MLX_METALLIB": modeRoot.appendingPathComponent("mlx.metallib").path,
                "MLXFAST_SETUP_PARALLEL_METALLIB": parallel,
            ]
        )
        #expect(result.status == 0, "parallel=\(parallel): \(result.stderr)")
    }
}

@Test
func setupRetainsFailedMetallibStateInSynchronousAndParallelModes() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    for parallel in ["0", "1"] {
        let result = try runSetupBash(
            """
            eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
            build_mlx_metallib() { return 19; }

            start_status=0
            start_mlx_metallib_build || start_status=$?
            wait_status=0
            wait_for_mlx_metallib_build || wait_status=$?
            retry_status=0
            start_mlx_metallib_build || retry_status=$?

            if [[ "${SETUP_PARALLEL_METALLIB}" == "1" ]]; then
              [[ "${start_status}" == "0" ]]
            else
              [[ "${start_status}" != "0" ]]
            fi
            [[ "${wait_status}" != "0" ]]
            [[ "${retry_status}" != "0" ]]
            [[ "${METALLIB_BUILD_STATE}" == "failed" ]]
            [[ -z "${METALLIB_BUILD_PID}" ]]
            """,
            environment: [
                "REPO_ROOT": FileManager.default.currentDirectoryPath,
                "MLXFAST_MLX_METALLIB": root.appendingPathComponent("missing-\(parallel).metallib").path,
                "MLXFAST_SETUP_PARALLEL_METALLIB": parallel,
            ]
        )
        #expect(result.status == 0, "parallel=\(parallel): \(result.stderr)")
    }
}

@Test
func setupSerializesSharedReferenceCacheMutationAndRechecksAfterWaiting() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try "{}".write(
        to: referenceDir.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        verify_reference_weights() {
          if [[ ! -d "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]; then
            printf 'unlocked verification\n' >> "${UNLOCKED_VERIFY_LOG}"
          fi
          [[ -f "$1/ready" ]]
        }
        ensure_reference_compat_link() { :; }
        download_reference_weights_locked() {
          printf 'mutation\n' >> "${MUTATION_LOG}"
          sleep 1.2
          : > "$1/ready"
        }

        download_reference_weights "${REFERENCE_DIR}" &
        first_pid=$!
        sleep 0.1
        download_reference_weights "${REFERENCE_DIR}" &
        second_pid=$!
        wait "${first_pid}"
        wait "${second_pid}"

        [[ "$(wc -l < "${MUTATION_LOG}" | tr -d ' ')" == "1" ]]
        [[ ! -e "${UNLOCKED_VERIFY_LOG}" ]]
        [[ ! -e "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MUTATION_LOG": root.appendingPathComponent("mutations.log").path,
            "UNLOCKED_VERIFY_LOG": root.appendingPathComponent("unlocked-verification.log").path,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache.lock").path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS": "5",
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
    #expect(result.stdout.contains("waiting for reference cache mutation lock"))
    #expect(result.stdout.contains("reference weights became ready while waiting"))
}

@Test
func setupCompatibilityLinkFailurePropagatesThroughConditionalCaller() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    let blockedParent = root.appendingPathComponent("not-a-directory")
    let compatibilityLink = blockedParent.appendingPathComponent("reference-link")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try "blocker".write(to: blockedParent, atomically: true, encoding: .utf8)

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        link_status=0
        ensure_reference_compat_link "${REFERENCE_DIR}" || link_status=$?
        [[ "${link_status}" != "0" ]]
        [[ ! -e "${REFERENCE_COMPAT_LINK}" && ! -L "${REFERENCE_COMPAT_LINK}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_COMPAT_LINK": compatibilityLink.path,
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
    #expect(!result.stdout.contains("setup.sh: linked"))
}

@Test
func setupAcceptsLegacyReferenceDirectoryAsItsOwnCompatibilityPath() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        cd "${TEST_ROOT}"
        mkdir -p "${REFERENCE_DIR}"
        ensure_reference_compat_link "${REFERENCE_DIR}"
        [[ -d "${REFERENCE_DIR}" ]]
        [[ ! -L "${REFERENCE_DIR}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "TEST_ROOT": root.path,
            "MLXFAST_REFERENCE_DIR": "reference_weights/gemma-4-31b-4bit",
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
    #expect(!result.stdout.contains("setup.sh: linked"))
}

@Test
func setupRecoversMutationLockOwnedByDeadLocalProcess() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let lockDir = root.appendingPathComponent("cache.lock")

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        mkdir -p "${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
        printf 'pid=99999999 host=%s started_at=2000-01-01T00:00:00Z\n' "$(hostname)" \
          > "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"

        acquire_reference_cache_mutation_lock
        [[ "${REFERENCE_CACHE_MUTATION_LOCK_HELD}" == "1" ]]
        grep -q "^pid=$$ " "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"
        release_reference_cache_mutation_lock
        [[ ! -e "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": lockDir.path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS": "2",
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS": "60",
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
    #expect(result.stdout.contains("recovered stale reference cache mutation lock"))
}

@Test
func setupRejectsZeroMutationLockStaleThreshold() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        lock_status=0
        acquire_reference_cache_mutation_lock || lock_status=$?
        [[ "${lock_status}" != "0" ]]
        [[ ! -e "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache.lock").path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS": "0",
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
    #expect(result.stderr.contains("must be a positive integer"))
}

@Test
func setupUsesOneCanonicalMutationLockForReferenceSymlinkAliases() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference", isDirectory: true)
    let referenceAlias = root.appendingPathComponent("reference-alias", isDirectory: true)
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try FileManager.default.createSymbolicLink(
        at: referenceAlias,
        withDestinationURL: referenceDir
    )

    func mutationLockPath(reference: URL) throws -> SetupBashResult {
        try runSetupBash(
            """
            unset MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR
            eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
            printf '%s\n' "${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
            """,
            environment: [
                "REPO_ROOT": FileManager.default.currentDirectoryPath,
                "MLXFAST_REFERENCE_DIR": reference.path,
            ]
        )
    }

    let direct = try mutationLockPath(reference: referenceDir)
    let aliased = try mutationLockPath(reference: referenceAlias)
    #expect(direct.status == 0, "stderr: \(direct.stderr)")
    #expect(aliased.status == 0, "stderr: \(aliased.stderr)")
    let directPath = direct.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
    let aliasedPath = aliased.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
    #expect(directPath == aliasedPath)
    #expect(directPath.hasSuffix("/reference.mlxfast-setup.lock"))
}

@Test
func setupReleasesReferenceCacheMutationLockWithoutOverwritingOperationFailure() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try "{}".write(
        to: referenceDir.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        verify_reference_weights() { return 1; }
        download_reference_weights_locked() { return 37; }

        operation_status=0
        download_reference_weights "${REFERENCE_DIR}" || operation_status=$?
        [[ "${operation_status}" == "37" ]]
        [[ ! -e "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache.lock").path,
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
}

@Test
func setupReferenceCacheStampUsesUniqueAtomicTemporaryFiles() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    let fixture = referenceDir.appendingPathComponent("config.json")
    try "fixture".write(to: fixture, atomically: true, encoding: .utf8)

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        fixture_hash="$(shasum -a 256 "${REFERENCE_DIR}/config.json" | awk '{print $1}')"
        fixture_size="$(wc -c < "${REFERENCE_DIR}/config.json" | tr -d ' ')"
        printf '%s %s config.json\n' "${fixture_hash}" "${fixture_size}" > "${REFERENCE_MANIFEST_PATH}"

        write_reference_cache_lock "${REFERENCE_DIR}" &
        first_pid=$!
        write_reference_cache_lock "${REFERENCE_DIR}" &
        second_pid=$!
        wait "${first_pid}"
        wait "${second_pid}"

        reference_cache_lock_is_current "${REFERENCE_DIR}"
        stamp_dir="$(dirname "${REFERENCE_CACHE_LOCK_PATH}")"
        ! compgen -G "${stamp_dir}/.mlxfast-reference-cache.lock.tmp.*" >/dev/null
        ! compgen -G "${stamp_dir}/.mlxfast-reference-cache.lock.files.*" >/dev/null
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_CACHE_LOCK_PATH": referenceDir.appendingPathComponent(".mlxfast-reference-cache.lock").path,
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
}

@Test
func setupFallsBackWhenPrimaryReferenceDownloadFails() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let fakeBin = root.appendingPathComponent("bin")
    try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("curl"),
        contents: """
        #!/usr/bin/env bash
        set -euo pipefail
        output=""
        url=""
        while [[ "$#" -gt 0 ]]; do
          case "$1" in
            --output) output="$2"; shift 2 ;;
            http*) url="$1"; shift ;;
            *) shift ;;
          esac
        done
        printf '%s\n' "${url}" >> "${CURL_LOG}"
        if [[ "${url}" == https://primary.invalid/* ]]; then
          exit 28
        fi
        printf 'fallback fixture' > "${output}"
        """
    )

    let output = root.appendingPathComponent("model.bin")
    let curlLog = root.appendingPathComponent("curl.log")
    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        download_reference_file "model.bin" "${OUTPUT_PATH}"
        [[ "$(cat "${OUTPUT_PATH}")" == "fallback fixture" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "PATH": "\(fakeBin.path):/usr/bin:/bin:/usr/sbin:/sbin",
            "CURL_LOG": curlLog.path,
            "OUTPUT_PATH": output.path,
            "MLXFAST_REFERENCE_BASE_URL": "https://primary.invalid",
            "MLXFAST_REFERENCE_FALLBACK_BASE_URL": "https://fallback.invalid",
            "MLXFAST_REFERENCE_HASH_VERIFY": "0",
            "MLXFAST_REFERENCE_DOWNLOAD_PROGRESS_SECONDS": "1",
            "MLXFAST_REFERENCE_DOWNLOAD_STALL_SECONDS": "1",
            "MLXFAST_REFERENCE_DOWNLOAD_MIN_BYTES_PER_SECOND": "1",
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
    let requests = try String(contentsOf: curlLog, encoding: .utf8)
    #expect(requests.contains("https://primary.invalid/model.bin"))
    #expect(requests.contains("https://fallback.invalid/model.bin"))
    #expect(result.stdout.contains("trying fallback source"))
}

@Test
func setupDownloadContractReportsProgressAndDetectsStalls() throws {
    let setup = try String(contentsOfFile: "setup.sh", encoding: .utf8)

    #expect(setup.contains("MLXFAST_REFERENCE_DOWNLOAD_PROGRESS_SECONDS"))
    #expect(setup.contains("MLXFAST_REFERENCE_DOWNLOAD_STALL_SECONDS"))
    #expect(setup.contains("MLXFAST_REFERENCE_DOWNLOAD_MIN_BYTES_PER_SECOND"))
    #expect(setup.contains("--speed-limit"))
    #expect(setup.contains("--speed-time"))
    // The parallel shard phase has exactly one progress reporter: the
    // aggregate heartbeat, which also carries the transfer rate and ETA.
    #expect(setup.contains("still downloading safetensors shard(s)"))
    #expect(setup.contains("rate=%.1f MiB/s eta=%ds"))
}

@Test
func setupParallelShardDownloadsEmitProgress() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let fakeBin = root.appendingPathComponent("bin")
    let output = root.appendingPathComponent("output")
    try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("curl"),
        contents: """
        #!/usr/bin/env bash
        set -euo pipefail
        output=""
        while [[ "$#" -gt 0 ]]; do
          case "$1" in
            --output) output="$2"; shift 2 ;;
            *) shift ;;
          esac
        done
        for chunk in 1 2 3 4; do
          printf 'chunk-%s\n' "${chunk}" >> "${output}"
          sleep 0.45
        done
        """
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        download_reference_shards "${OUTPUT_DIR}" file-a.safetensors file-b.safetensors
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "PATH": "\(fakeBin.path):/usr/bin:/bin:/usr/sbin:/sbin",
            "OUTPUT_DIR": output.path,
            "MLXFAST_REFERENCE_BASE_URL": "https://primary.invalid",
            "MLXFAST_REFERENCE_FALLBACK_BASE_URL": "",
            "MLXFAST_REFERENCE_HASH_VERIFY": "0",
            "MLXFAST_REFERENCE_DOWNLOAD_JOBS": "2",
            "MLXFAST_REFERENCE_DOWNLOAD_PROGRESS_SECONDS": "1",
            "MLXFAST_REFERENCE_DOWNLOAD_STALL_SECONDS": "5",
            "MLXFAST_REFERENCE_DOWNLOAD_MIN_BYTES_PER_SECOND": "1",
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
    #expect(result.stdout.contains("still downloading safetensors shard(s)"))
    #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent("file-a.safetensors").path))
    #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent("file-b.safetensors").path))
}

@Test
func setupRejectsCommandLineToolsWithFullXcodeInstructions() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let fakeBin = root.appendingPathComponent("bin")
    try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)

    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("swift"),
        contents: """
        #!/usr/bin/env bash
        exit 0
        """
    )
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("xcodebuild"),
        contents: """
        #!/usr/bin/env bash
        echo "xcode-select: error: tool 'xcodebuild' requires Xcode, but active developer directory '/Library/Developer/CommandLineTools' is a command line tools instance" >&2
        exit 1
        """
    )
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("xcode-select"),
        contents: """
        #!/usr/bin/env bash
        printf '/Library/Developer/CommandLineTools\n'
        """
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        ensure_swift_toolchain
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "PATH": "\(fakeBin.path):/usr/bin:/bin:/usr/sbin:/sbin",
        ]
    )

    #expect(result.status != 0)
    #expect(result.stderr.contains("full Xcode is required"))
    #expect(result.stderr.contains("Command Line Tools alone"))
    #expect(result.stderr.contains("sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"))
}

@Test
func setupUsesDownloadedMetalToolchainIdentifierForCompilerProbe() throws {
    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        xcodebuild() {
          printf 'Toolchain Identifier: com.apple.dt.toolchain.Metal.1\n'
        }
        xcrun() {
          [[ "${TOOLCHAINS:-}" == "com.apple.dt.toolchain.Metal.1" ]]
        }
        unset TOOLCHAINS
        metal_compiler_is_available
        [[ "${TOOLCHAINS}" == "com.apple.dt.toolchain.Metal.1" ]]
        """,
        environment: ["REPO_ROOT": FileManager.default.currentDirectoryPath]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
}

@Test
func metallibBuilderRejectsAmbiguousCMakeOutputs() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let fakeBin = root.appendingPathComponent("bin")
    let checkout = root.appendingPathComponent("mlx-swift")
    for path in [
        "Source/Cmlx/mlx",
        "Source/Cmlx/metal-cpp",
        "Source/Cmlx/json",
        "Source/Cmlx/fmt",
    ] {
        try FileManager.default.createDirectory(
            at: checkout.appendingPathComponent(path),
            withIntermediateDirectories: true
        )
    }
    try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)

    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("cmake"),
        contents: """
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ " $* " == *" --build "* ]]; then
          mkdir -p "${MLXFAST_MLX_METAL_BUILD_DIR}/first" "${MLXFAST_MLX_METAL_BUILD_DIR}/second"
          printf 'first' > "${MLXFAST_MLX_METAL_BUILD_DIR}/first/mlx.metallib"
          printf 'second' > "${MLXFAST_MLX_METAL_BUILD_DIR}/second/mlx.metallib"
        fi
        """
    )
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("xcodebuild"),
        contents: """
        #!/usr/bin/env bash
        printf 'Toolchain Identifier: com.apple.dt.toolchain.Metal.1\n'
        """
    )
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("xcrun"),
        contents: """
        #!/usr/bin/env bash
        exit 0
        """
    )

    let output = root.appendingPathComponent("output/mlx.metallib")
    let result = try runSetupBash(
        """
        "${REPO_ROOT}/tools/build-mlx-metallib.sh"
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "PATH": "\(fakeBin.path):/usr/bin:/bin",
            "MLXFAST_CMAKE_BIN": fakeBin.appendingPathComponent("cmake").path,
            "MLXFAST_MLX_SWIFT_CHECKOUT": checkout.path,
            "MLXFAST_MLX_METAL_BUILD_DIR": root.appendingPathComponent("metal-build").path,
            "MLXFAST_MLX_METALLIB": output.path,
        ]
    )

    #expect(result.status != 0)
    #expect(result.stderr.contains("produced multiple mlx.metallib files"))
    #expect(!FileManager.default.fileExists(atPath: output.path))
}

private struct SetupBashResult {
    let status: Int32
    let stdout: String
    let stderr: String
}

private func runSetupBash(
    _ script: String,
    environment overrides: [String: String] = [:]
) throws -> SetupBashResult {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["-c", script]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    var environment = ProcessInfo.processInfo.environment
    for (key, value) in overrides {
        environment[key] = value
    }
    process.environment = environment
    let stdout = Pipe()
    let stderr = Pipe()
    process.standardOutput = stdout
    process.standardError = stderr
    try process.run()
    process.waitUntilExit()
    let stdoutData = stdout.fileHandleForReading.readDataToEndOfFile()
    let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()
    return SetupBashResult(
        status: process.terminationStatus,
        stdout: String(data: stdoutData, encoding: .utf8) ?? "",
        stderr: String(data: stderrData, encoding: .utf8) ?? ""
    )
}

private func setupTemporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        "mlxfast-setup-script-tests-\(UUID().uuidString)"
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func writeSetupExecutable(at url: URL, contents: String) throws {
    try contents.write(to: url, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: url.path
    )
}
