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
    #expect(setup.contains("export TOOLCHAINS=\"${identifier}\""))
    #expect(setup.contains("RUNTIME_WORKER_BIN=\"${MLXFAST_RUNTIME_WORKER_EXECUTABLE:-$(dirname \"${SWIFT_BIN}\")/mlxfast-runtime-worker}\""))
    #expect(setup.contains("MLX_METALLIB=\"${MLXFAST_MLX_METALLIB:-$(dirname \"${RUNTIME_WORKER_BIN}\")/mlx.metallib}\""))
    #expect(setup.contains("swift build -c release --product mlxfast-swift"))
    #expect(setup.contains("swift build -c release --product mlxfast-runtime-worker"))
    #expect(setup.contains("participant runtime worker missing at ${RUNTIME_WORKER_BIN}"))
    #expect(setup.contains("TODO(security): Give the trusted CLI and participant worker independent"))
    #expect(metallibBuilder.contains("-DMLX_BUILD_GGUF=OFF"))
    #expect(metallibBuilder.contains("export CLANG_MODULE_CACHE_PATH"))
    #expect(metallibBuilder.contains("CLANG_MODULE_CACHE_PATH=\"$(repository_path"))
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
func setupFailureCleanupTerminatesMetallibProcessGroup() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        build_mlx_metallib() {
          sleep 30 &
          child_pid=$!
          printf '%s\n' "${child_pid}" > "${CHILD_PID_PATH}"
          wait "${child_pid}"
        }

        start_mlx_metallib_build
        for attempt in 1 2 3 4 5 6 7 8 9 10; do
          [[ -s "${CHILD_PID_PATH}" ]] && break
          sleep 0.05
        done
        [[ -s "${CHILD_PID_PATH}" ]]

        set +e
        false
        cleanup_background_builds
        cleanup_status=$?
        set -e
        [[ "${cleanup_status}" != "0" ]]
        child_pid="$(cat "${CHILD_PID_PATH}")"
        ! kill -0 "${child_pid}" 2>/dev/null
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "CHILD_PID_PATH": root.appendingPathComponent("child.pid").path,
            "MLXFAST_MLX_METALLIB": root.appendingPathComponent("mlx.metallib").path,
            "MLXFAST_SETUP_PARALLEL_METALLIB": "1",
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
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
        printf 'token=dead-token pid=99999999 host=%s started_at=2000-01-01T00:00:00Z\n' "$(hostname)" \
          > "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"

        acquire_reference_cache_mutation_lock
        [[ "${REFERENCE_CACHE_MUTATION_LOCK_HELD}" == "1" ]]
        grep -q "token=${REFERENCE_CACHE_MUTATION_LOCK_TOKEN} pid=$$ " \
          "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"
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
func setupAllowsOnlyOneCompetingStaleLockRecoverer() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        mkdir -p "${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
        printf 'token=dead-token pid=99999999 host=%s started_at=2000-01-01T00:00:00Z\n' \
          "$(hostname)" > "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"
        : > "${RECOVERY_LOG}"

        pids=()
        for ordinal in 1 2 3 4 5 6 7 8; do
          (
            if recover_stale_reference_cache_mutation_lock; then
              printf '%s\n' "${ordinal}" >> "${RECOVERY_LOG}"
            fi
          ) &
          pids+=("$!")
        done
        for pid in "${pids[@]}"; do
          wait "${pid}" || true
        done

        [[ "$(wc -l < "${RECOVERY_LOG}" | tr -d ' ')" == "1" ]]
        [[ ! -e "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]
        acquire_reference_cache_mutation_lock
        grep -q "token=${REFERENCE_CACHE_MUTATION_LOCK_TOKEN}" \
          "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"
        release_reference_cache_mutation_lock
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "RECOVERY_LOG": root.appendingPathComponent("recoveries.log").path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache.lock").path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS": "2",
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupStalledLockCreatorCannotPublishIntoReplacementLock() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        ln() {
          local slot=""
          local status=0
          if mkdir "${FIRST_LINK_SLOT}" 2>/dev/null; then
            slot=first
            : > "${FIRST_STALLED}"
            while [[ ! -f "${RESUME_FIRST}" ]]; do sleep 0.05; done
          elif mkdir "${SECOND_LINK_SLOT}" 2>/dev/null; then
            slot=second
            : > "${SECOND_STALLED}"
            while [[ ! -f "${RESUME_SECOND}" ]]; do sleep 0.05; done
          fi
          command ln "$@" || status=$?
          if [[ "${slot}" == "first" ]]; then
            : > "${FIRST_ATTEMPTED}"
          elif [[ "${slot}" == "second" ]]; then
            : > "${SECOND_ATTEMPTED}"
          fi
          return "${status}"
        }

        (status=0
         if acquire_reference_cache_mutation_lock; then
           cat "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner" > "${FIRST_OWNER}"
           : > "${FIRST_ACQUIRED}"
           while [[ ! -f "${RELEASE_FIRST}" ]]; do sleep 0.05; done
           release_reference_cache_mutation_lock || status=$?
         else
           status=$?
         fi
         printf '%s\n' "${status}" > "${FIRST_STATUS}") &
        first_pid=$!
        while [[ ! -f "${FIRST_STALLED}" ]]; do sleep 0.05; done

        (status=0
         if acquire_reference_cache_mutation_lock; then
           cat "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner" > "${SECOND_OWNER}"
           : > "${SECOND_ACQUIRED}"
           while [[ ! -f "${RELEASE_SECOND}" ]]; do sleep 0.05; done
           release_reference_cache_mutation_lock || status=$?
         else
           status=$?
         fi
         printf '%s\n' "${status}" > "${SECOND_STATUS}") &
        second_pid=$!
        while [[ ! -f "${SECOND_STALLED}" ]]; do sleep 0.05; done

        : > "${RESUME_FIRST}"
        while [[ ! -f "${FIRST_ACQUIRED}" ]] && kill -0 "${first_pid}" 2>/dev/null; do
          sleep 0.05
        done
        if [[ ! -f "${FIRST_ACQUIRED}" ]]; then
          : > "${RESUME_SECOND}"
          : > "${RELEASE_FIRST}"
          : > "${RELEASE_SECOND}"
          wait "${first_pid}" || true
          wait "${second_pid}" || true
          exit 1
        fi
        : > "${RESUME_SECOND}"
        while [[ ! -f "${SECOND_ATTEMPTED}" ]]; do sleep 0.05; done

        : > "${RELEASE_FIRST}"
        wait "${first_pid}"
        [[ "$(cat "${FIRST_STATUS}")" == "0" ]]
        while [[ ! -f "${SECOND_ACQUIRED}" ]] && kill -0 "${second_pid}" 2>/dev/null; do
          sleep 0.05
        done
        [[ -f "${SECOND_ACQUIRED}" ]]
        ! cmp -s "${FIRST_OWNER}" "${SECOND_OWNER}"
        : > "${RELEASE_SECOND}"
        wait "${second_pid}"
        [[ "$(cat "${SECOND_STATUS}")" == "0" ]]
        [[ ! -e "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]
        [[ -z "$(find "$(dirname "${REFERENCE_CACHE_MUTATION_LOCK_DIR}")" \
          -maxdepth 1 -name "$(basename "${REFERENCE_CACHE_MUTATION_LOCK_DIR}").generation.*" -print -quit)" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache.lock").path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS": "1",
            "FIRST_LINK_SLOT": root.appendingPathComponent("first-link-slot").path,
            "SECOND_LINK_SLOT": root.appendingPathComponent("second-link-slot").path,
            "FIRST_STALLED": root.appendingPathComponent("first-stalled").path,
            "SECOND_STALLED": root.appendingPathComponent("second-stalled").path,
            "RESUME_FIRST": root.appendingPathComponent("resume-first").path,
            "RESUME_SECOND": root.appendingPathComponent("resume-second").path,
            "FIRST_ATTEMPTED": root.appendingPathComponent("first-attempted").path,
            "SECOND_ATTEMPTED": root.appendingPathComponent("second-attempted").path,
            "FIRST_STATUS": root.appendingPathComponent("first-status").path,
            "SECOND_STATUS": root.appendingPathComponent("second-status").path,
            "FIRST_OWNER": root.appendingPathComponent("first-owner").path,
            "SECOND_OWNER": root.appendingPathComponent("second-owner").path,
            "FIRST_ACQUIRED": root.appendingPathComponent("first-acquired").path,
            "SECOND_ACQUIRED": root.appendingPathComponent("second-acquired").path,
            "RELEASE_FIRST": root.appendingPathComponent("release-first").path,
            "RELEASE_SECOND": root.appendingPathComponent("release-second").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupReclaimsOnlyDeadOrBrokenAtomicLockGenerations() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        host_hash="$(printf '%s' "$(hostname)" | shasum -a 256 | awk '{print substr($1, 1, 16)}')"
        dead_pid=999999
        while kill -0 "${dead_pid}" 2>/dev/null; do dead_pid=$((dead_pid + 1)); done
        dead_generation="${REFERENCE_CACHE_MUTATION_LOCK_DIR}.generation.dead"
        live_generation="${REFERENCE_CACHE_MUTATION_LOCK_DIR}.generation.live"
        dead_initializing="${REFERENCE_CACHE_MUTATION_LOCK_DIR}.initializing.${host_hash}.${dead_pid}.dead"
        live_initializing="${REFERENCE_CACHE_MUTATION_LOCK_DIR}.initializing.${host_hash}.$$.live"
        mkdir "${dead_generation}" "${live_generation}" \
          "${dead_initializing}" "${live_initializing}"
        printf 'token=dead pid=%s host=%s started_at=now\n' "${dead_pid}" "$(hostname)" \
          > "${dead_generation}/owner"
        printf 'token=live pid=%s host=%s started_at=now\n' "$$" "$(hostname)" \
          > "${live_generation}/owner"
        broken_target="${REFERENCE_CACHE_MUTATION_LOCK_DIR}.generation.broken"
        ln -s -h -- "${broken_target}" "${REFERENCE_CACHE_MUTATION_LOCK_DIR}"

        acquire_reference_cache_mutation_lock
        [[ ! -e "${dead_generation}" ]]
        [[ ! -e "${dead_initializing}" ]]
        [[ -d "${live_generation}" ]]
        [[ -d "${live_initializing}" ]]
        release_reference_cache_mutation_lock
        [[ ! -e "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" && ! -L "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]
        rm -rf "${live_generation}" "${live_initializing}"
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache.lock").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupRecoversAgedMalformedMutationLockOwner() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        mkdir -p "${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
        : > "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"
        touch -t 200001010000.00 "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"

        acquire_reference_cache_mutation_lock
        [[ "${REFERENCE_CACHE_MUTATION_LOCK_HELD}" == "1" ]]
        grep -q "token=${REFERENCE_CACHE_MUTATION_LOCK_TOKEN}" \
          "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"
        release_reference_cache_mutation_lock
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache.lock").path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS": "2",
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS": "1",
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupRefusesToReleaseMutationLockWithDifferentOwnerToken() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        acquire_reference_cache_mutation_lock
        original_token="${REFERENCE_CACHE_MUTATION_LOCK_TOKEN}"
        printf 'token=other-token pid=%s host=%s started_at=now\n' "$$" "$(hostname)" \
          > "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"

        release_status=0
        release_reference_cache_mutation_lock || release_status=$?
        [[ "${release_status}" != "0" ]]
        [[ -d "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]

        printf 'token=%s pid=%s host=%s started_at=now\n' \
          "${original_token}" "$$" "$(hostname)" > "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"
        release_reference_cache_mutation_lock
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache.lock").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
    #expect(result.stderr.contains("owned by another process"))
}

@Test
func setupMutationLockWaiterRetriesWhenHolderReleasesDuringPublishAttempt() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        holder_generation="${REFERENCE_CACHE_MUTATION_LOCK_DIR}.generation.holder"
        mkdir "${holder_generation}"
        printf 'token=holder pid=%s host=%s started_at=now\n' "$$" "$(hostname)" \
          > "${holder_generation}/owner"
        ln -s -h -- "${holder_generation}" "${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
        lock_name="$(basename "${REFERENCE_CACHE_MUTATION_LOCK_DIR}")"

        # Deterministically land a concurrent holder's release inside the
        # waiter's race window: the waiter's first removal of its own
        # unpublished generation directory happens between its failed publish
        # attempt and the obstruction check, so dropping the published lock
        # there reproduces the transient absent-path state that used to be
        # misclassified as a fatal non-directory obstruction.
        rm() {
          if [[ ! -e "${RELEASE_SENTINEL}" && "${1:-}" == "-rf" \
              && "$(basename "${2:-none}")" == "${lock_name}.generation."* ]]; then
            : > "${RELEASE_SENTINEL}"
            command rm -f "${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
            command rm -rf "${holder_generation}"
          fi
          command rm "$@"
        }

        acquire_reference_cache_mutation_lock
        [[ "${REFERENCE_CACHE_MUTATION_LOCK_HELD}" == "1" ]]
        [[ -e "${RELEASE_SENTINEL}" ]]
        release_reference_cache_mutation_lock
        [[ ! -e "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" && ! -L "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "RELEASE_SENTINEL": root.appendingPathComponent("released").path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache-mutation.lock").path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS": "10",
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
    #expect(result.stdout.contains("waiting for reference cache mutation lock"))
    #expect(result.stdout.contains("acquired reference cache mutation lock"))
    #expect(!result.stderr.contains("is not a lock directory"))
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

        (acquire_reference_cache_mutation_lock
         verify_reference_manifest "${REFERENCE_DIR}"
         write_reference_cache_lock "${REFERENCE_DIR}"
         release_reference_cache_mutation_lock) &
        first_pid=$!
        (acquire_reference_cache_mutation_lock
         verify_reference_manifest "${REFERENCE_DIR}"
         write_reference_cache_lock "${REFERENCE_DIR}"
         release_reference_cache_mutation_lock) &
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
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache-mutation.lock").path,
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
    // The heartbeat's byte counter includes in-flight .partial transfers, so
    // progress moves from the first transferred byte instead of printing 0%
    // until a whole shard completes and is renamed.
    #expect(setup.contains("elif [[ -f \"${file_path}.partial\" ]]; then"))
}

@Test
func setupDownloadHeartbeatCountsInProgressPartialBytes() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let output = root.appendingPathComponent("output")
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)

    // Shard A is mid-transfer (curl still writing the .partial); shard B has
    // completed (renamed to its final name) and also has a stale leftover
    // .partial, which must NOT be double counted. Expected: 7 + 5 = 12.
    try Data(repeating: 0x61, count: 7).write(
        to: output.appendingPathComponent("file-a.safetensors.partial")
    )
    try Data(repeating: 0x62, count: 5).write(
        to: output.appendingPathComponent("file-b.safetensors")
    )
    try Data(repeating: 0x63, count: 9).write(
        to: output.appendingPathComponent("file-b.safetensors.partial")
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        reference_download_on_disk_bytes "${OUTPUT_DIR}" file-a.safetensors file-b.safetensors
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "OUTPUT_DIR": output.path,
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
    let lastLine = result.stdout
        .split(separator: "\n", omittingEmptySubsequences: true)
        .last
        .map(String.init)
    #expect(lastLine == "12", "stdout: \(result.stdout)")
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
func setupReferenceCacheStampRejectsSameSizeImmediateMutation() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try "fixture".write(
        to: referenceDir.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        fixture_hash="$(shasum -a 256 "${REFERENCE_DIR}/config.json" | awk '{print $1}')"
        printf '%s 7 config.json\n' "${fixture_hash}" > "${REFERENCE_MANIFEST_PATH}"
        verify_reference_manifest "${REFERENCE_DIR}"
        write_reference_cache_lock "${REFERENCE_DIR}"
        reference_cache_lock_is_current "${REFERENCE_DIR}"

        printf 'changed' > "${REFERENCE_DIR}/config.json"
        if reference_cache_lock_is_current "${REFERENCE_DIR}"; then
          echo "cache stamp accepted changed content" >&2
          exit 1
        fi
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_CACHE_LOCK_PATH": referenceDir.appendingPathComponent(".mlxfast-reference-cache.lock").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupReferenceCacheStampRejectsMutationAfterHashBeforePublication() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try "fixture".write(
        to: referenceDir.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        fixture_hash="$(shasum -a 256 "${REFERENCE_DIR}/config.json" | awk '{print $1}')"
        printf '%s 7 config.json\n' "${fixture_hash}" > "${REFERENCE_MANIFEST_PATH}"
        verify_reference_manifest "${REFERENCE_DIR}"

        printf 'changed' > "${REFERENCE_DIR}/config.json"
        if write_reference_cache_lock "${REFERENCE_DIR}"; then
          echo "cache stamp accepted bytes changed after hash verification" >&2
          exit 1
        fi
        [[ ! -e "${REFERENCE_CACHE_LOCK_PATH}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_CACHE_LOCK_PATH": root.appendingPathComponent("cache.stamp").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupReferenceCacheStampRejectsManifestSwapAfterVerification() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try "fixture".write(
        to: referenceDir.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        fixture_hash="$(shasum -a 256 "${REFERENCE_DIR}/config.json" | awk '{print $1}')"
        printf '%s 7 config.json\n' "${fixture_hash}" > "${REFERENCE_MANIFEST_PATH}"
        verify_reference_manifest "${REFERENCE_DIR}"

        printf '%s 7 config.json\n' \
          '0000000000000000000000000000000000000000000000000000000000000000' \
          > "${REFERENCE_MANIFEST_PATH}"
        if write_reference_cache_lock "${REFERENCE_DIR}"; then
          echo "cache stamp accepted a different manifest" >&2
          exit 1
        fi
        [[ ! -e "${REFERENCE_CACHE_LOCK_PATH}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_CACHE_LOCK_PATH": root.appendingPathComponent("cache.stamp").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupReferenceCacheStampRejectsManifestSwapDuringFastValidation() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try "fixture".write(
        to: referenceDir.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        fixture_hash="$(printf fixture | shasum -a 256 | awk '{print $1}')"
        printf '%s 7 config.json\n' "${fixture_hash}" > "${REFERENCE_MANIFEST_PATH}"
        printf '%s 7 config.json\n' \
          '0000000000000000000000000000000000000000000000000000000000000000' \
          > "${MANIFEST_B}"
        verify_reference_manifest "${REFERENCE_DIR}"
        write_reference_cache_lock "${REFERENCE_DIR}"

        file_metadata_signature() {
          if mkdir "${SWAP_ONCE}" 2>/dev/null; then
            cp "${MANIFEST_B}" "${REFERENCE_MANIFEST_PATH}"
          fi
          stat -f '%d:%i:%z:%Fm:%Fc:%v' "$1" 2>/dev/null \
            || stat -c '%d:%i:%s:%Y:%Z:0' "$1"
        }
        if reference_cache_lock_is_current "${REFERENCE_DIR}"; then
          echo "cache stamp accepted a manifest swapped during validation" >&2
          exit 1
        fi
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_CACHE_LOCK_PATH": root.appendingPathComponent("cache.stamp").path,
            "MANIFEST_B": root.appendingPathComponent("manifest-b.sha256").path,
            "SWAP_ONCE": root.appendingPathComponent("swap-once").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
    #expect(result.stderr.contains("manifest changed while the cache lock was being validated"))
}

@Test
func setupReferenceVerificationRejectsManifestABA() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try "fixture".write(
        to: referenceDir.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        fixture_hash="$(printf 'fixture' | shasum -a 256 | awk '{print $1}')"
        changed_hash="$(printf 'changed' | shasum -a 256 | awk '{print $1}')"
        printf '%s 7 config.json\n' "${fixture_hash}" > "${REFERENCE_MANIFEST_PATH}"
        cp "${REFERENCE_MANIFEST_PATH}" "${MANIFEST_A}"

        eval "$(declare -f create_reference_verified_signatures_path \
          | sed '1s/create_reference_verified_signatures_path/create_reference_verified_signatures_path_original/')"
        create_reference_verified_signatures_path() {
          create_reference_verified_signatures_path_original
          printf '%s 7 config.json\n' "${changed_hash}" > "${REFERENCE_MANIFEST_PATH}"
          printf 'changed' > "${REFERENCE_DIR}/config.json"
        }
        file_metadata_signature() {
          count=0
          [[ -f "${SIGNATURE_COUNT}" ]] && count="$(cat "${SIGNATURE_COUNT}")"
          count=$((count + 1))
          printf '%s\n' "${count}" > "${SIGNATURE_COUNT}"
          if [[ "${count}" == "2" ]]; then
            cp "${MANIFEST_A}" "${REFERENCE_MANIFEST_PATH}"
          fi
          stat -f '%d:%i:%z:%Fm:%Fc:%v' "$1"
        }

        if verify_reference_manifest "${REFERENCE_DIR}" \
            && write_reference_cache_lock "${REFERENCE_DIR}"; then
          echo "cache stamp accepted A/B/A manifest verification" >&2
          exit 1
        fi
        [[ ! -e "${REFERENCE_CACHE_LOCK_PATH}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_CACHE_LOCK_PATH": root.appendingPathComponent("cache.stamp").path,
            "MANIFEST_A": root.appendingPathComponent("manifest-a.sha256").path,
            "SIGNATURE_COUNT": root.appendingPathComponent("signature-count").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupVerifiedDownloadMarkerRejectsManifestABA() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try "fixture".write(
        to: referenceDir.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        fixture_hash="$(printf 'fixture' | shasum -a 256 | awk '{print $1}')"
        changed_hash="$(printf 'changed' | shasum -a 256 | awk '{print $1}')"
        printf '%s 7 config.json\n' "${fixture_hash}" > "${REFERENCE_MANIFEST_PATH}"
        cp "${REFERENCE_MANIFEST_PATH}" "${MANIFEST_A}"
        printf '%s 7 config.json\n' "${changed_hash}" > "${REFERENCE_MANIFEST_PATH}"
        printf 'changed' > "${REFERENCE_DIR}/config.json"

        file_metadata_signature() {
          count=0
          [[ -f "${SIGNATURE_COUNT}" ]] && count="$(cat "${SIGNATURE_COUNT}")"
          count=$((count + 1))
          printf '%s\n' "${count}" > "${SIGNATURE_COUNT}"
          if [[ "${count}" == "2" ]]; then
            cp "${MANIFEST_A}" "${REFERENCE_MANIFEST_PATH}"
          fi
          stat -f '%d:%i:%z:%Fm:%Fc:%v' "$1"
        }

        marker="${REFERENCE_DIR}/config.json.complete"
        if reference_file_is_current config.json "${REFERENCE_DIR}/config.json" \
            && mark_reference_file_complete \
              config.json "${REFERENCE_DIR}/config.json" "${marker}" \
              "${REFERENCE_VERIFIED_CONTENT_IDENTITY}" \
              "${REFERENCE_VERIFIED_EXPECTED_HASH}" \
              "${REFERENCE_VERIFIED_EXPECTED_SIZE}" \
              "${REFERENCE_VERIFIED_FILE_MANIFEST_HASH}"; then
          echo "verified marker accepted A/B/A manifest verification" >&2
          exit 1
        fi
        [[ ! -e "${marker}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_CACHE_LOCK_PATH": root.appendingPathComponent("cache.stamp").path,
            "MANIFEST_A": root.appendingPathComponent("manifest-a.sha256").path,
            "SIGNATURE_COUNT": root.appendingPathComponent("signature-count").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupReferenceCacheStampDoesNotModifyReadOnlyCheckpoint() throws {
    let root = try setupTemporaryDirectory()
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    let fixture = referenceDir.appendingPathComponent("config.json")
    try "fixture".write(to: fixture, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o444],
        ofItemAtPath: fixture.path
    )
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o555],
        ofItemAtPath: referenceDir.path
    )
    defer {
        try? FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: referenceDir.path
        )
        try? FileManager.default.setAttributes(
            [.posixPermissions: 0o644],
            ofItemAtPath: fixture.path
        )
        try? FileManager.default.removeItem(at: root)
    }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        fixture_hash="$(shasum -a 256 "${REFERENCE_DIR}/config.json" | awk '{print $1}')"
        printf '%s 7 config.json\n' "${fixture_hash}" > "${REFERENCE_MANIFEST_PATH}"

        before_signature="$(file_metadata_signature "${REFERENCE_DIR}/config.json")"
        verify_reference_manifest "${REFERENCE_DIR}"
        write_reference_cache_lock "${REFERENCE_DIR}"
        after_signature="$(file_metadata_signature "${REFERENCE_DIR}/config.json")"

        [[ "${before_signature}" == "${after_signature}" ]]
        reference_cache_lock_is_current "${REFERENCE_DIR}"
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_CACHE_LOCK_PATH": root.appendingPathComponent("cache.stamp").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupDownloadPublishesVerifiedPartialAtomically() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let published = root.appendingPathComponent("config.json")
    try "old-data".write(to: published, atomically: true, encoding: .utf8)

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        new_hash="$(printf 'new-data' | shasum -a 256 | awk '{print $1}')"
        printf '%s 8 config.json\n' "${new_hash}" > "${REFERENCE_MANIFEST_PATH}"
        curl() {
          output=""
          while [[ "$#" -gt 0 ]]; do
            if [[ "$1" == "--output" ]]; then
              output="$2"
              shift 2
            else
              shift
            fi
          done
          [[ "${output}" == "${PUBLISHED_PATH}.partial" ]]
          [[ "$(cat "${PUBLISHED_PATH}")" == "old-data" ]]
          printf 'new-data' > "${output}"
        }

        download_reference_file config.json "${PUBLISHED_PATH}"
        [[ "$(cat "${PUBLISHED_PATH}")" == "new-data" ]]
        [[ ! -e "${PUBLISHED_PATH}.partial" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "PUBLISHED_PATH": published.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_BASE_URL": "https://invalid.example",
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupFailedDownloadPreservesPreviouslyPublishedFile() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let published = root.appendingPathComponent("config.json")
    try "old-data".write(to: published, atomically: true, encoding: .utf8)

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        new_hash="$(printf 'new-data' | shasum -a 256 | awk '{print $1}')"
        printf '%s 8 config.json\n' "${new_hash}" > "${REFERENCE_MANIFEST_PATH}"
        curl() {
          output=""
          while [[ "$#" -gt 0 ]]; do
            if [[ "$1" == "--output" ]]; then
              output="$2"
              shift 2
            else
              shift
            fi
          done
          printf 'partial' > "${output}"
          return 22
        }

        download_status=0
        download_reference_file config.json "${PUBLISHED_PATH}" || download_status=$?
        [[ "${download_status}" != "0" ]]
        [[ "$(cat "${PUBLISHED_PATH}")" == "old-data" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "PUBLISHED_PATH": published.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_BASE_URL": "https://invalid.example",
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
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
func setupDetectedMetalToolchainOverridesUnrelatedToolchainsEnvironment() throws {
    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        xcodebuild() {
          printf 'Toolchain Identifier: com.apple.dt.toolchain.Metal.1\n'
        }
        xcrun() {
          [[ "${TOOLCHAINS:-}" == "com.apple.dt.toolchain.Metal.1" ]]
        }
        export TOOLCHAINS="org.swift.unrelated"
        metal_compiler_is_available
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
        printf '%s\n' "$*" >> "${CMAKE_ARGUMENT_LOG}"
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
        [[ "${TOOLCHAINS:-}" == "com.apple.dt.toolchain.Metal.1" ]]
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
            "CMAKE_ARGUMENT_LOG": root.appendingPathComponent("cmake-arguments.log").path,
            "TOOLCHAINS": "org.swift.unrelated",
        ]
    )

    #expect(result.status != 0)
    #expect(result.stderr.contains("produced multiple mlx.metallib files"))
    #expect(!FileManager.default.fileExists(atPath: output.path))
    let arguments = try String(
        contentsOf: root.appendingPathComponent("cmake-arguments.log"),
        encoding: .utf8
    )
    #expect(
        arguments.contains(
            "/mlx-swift/Source/Cmlx/metal-cpp"
        )
    )
    #expect(!arguments.contains("\(FileManager.default.currentDirectoryPath)/\(checkout.path)"))
}

@Test
func metallibBuilderSerializesConcurrentProcesses() throws {
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
          if ! mkdir "${ACTIVE_BUILD_DIR}" 2>/dev/null; then
            : > "${OVERLAP_LOG}"
          else
            sleep 0.3
            rmdir "${ACTIVE_BUILD_DIR}"
          fi
          mkdir -p "${MLXFAST_MLX_METAL_BUILD_DIR}/result"
          printf 'metallib' > "${MLXFAST_MLX_METAL_BUILD_DIR}/result/mlx.metallib"
        fi
        """
    )
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("xcodebuild"),
        contents: "#!/usr/bin/env bash\nprintf 'Toolchain Identifier: metal.test\\n'\n"
    )
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("xcrun"),
        contents: "#!/usr/bin/env bash\nexit 0\n"
    )

    let output = root.appendingPathComponent("output/mlx.metallib")
    let result = try runSetupBash(
        """
        "${REPO_ROOT}/tools/build-mlx-metallib.sh" &
        first_pid=$!
        "${REPO_ROOT}/tools/build-mlx-metallib.sh" &
        second_pid=$!
        wait "${first_pid}"
        wait "${second_pid}"
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "PATH": "\(fakeBin.path):/usr/bin:/bin",
            "MLXFAST_CMAKE_BIN": fakeBin.appendingPathComponent("cmake").path,
            "MLXFAST_MLX_SWIFT_CHECKOUT": checkout.path,
            "MLXFAST_MLX_METAL_BUILD_DIR": root.appendingPathComponent("metal-build").path,
            "MLXFAST_MLX_METALLIB": output.path,
            "ACTIVE_BUILD_DIR": root.appendingPathComponent("active-build").path,
            "OVERLAP_LOG": root.appendingPathComponent("overlap").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
    #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("overlap").path))
    #expect(try String(contentsOf: output, encoding: .utf8) == "metallib")
    #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("metal-build.mlxfast-build.lock").path))
}

@Test
func metallibBuilderStalledLockCreatorCannotPublishIntoReplacementLock() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed \
          -e 's|^ROOT_DIR=.*|ROOT_DIR="${REPO_ROOT}"|' \
          -e '/^METAL_TOOLCHAIN_IDENTIFIER=/,$d' \
          "${REPO_ROOT}/tools/build-mlx-metallib.sh")"
        ln() {
          local slot=""
          local status=0
          if mkdir "${FIRST_LINK_SLOT}" 2>/dev/null; then
            slot=first
            : > "${FIRST_STALLED}"
            while [[ ! -f "${RESUME_FIRST}" ]]; do sleep 0.05; done
          elif mkdir "${SECOND_LINK_SLOT}" 2>/dev/null; then
            slot=second
            : > "${SECOND_STALLED}"
            while [[ ! -f "${RESUME_SECOND}" ]]; do sleep 0.05; done
          fi
          command ln "$@" || status=$?
          if [[ "${slot}" == "first" ]]; then
            : > "${FIRST_ATTEMPTED}"
          elif [[ "${slot}" == "second" ]]; then
            : > "${SECOND_ATTEMPTED}"
          fi
          return "${status}"
        }

        (status=0
         if acquire_build_lock; then
           cat "${BUILD_LOCK_DIR}/owner" > "${FIRST_OWNER}"
           : > "${FIRST_ACQUIRED}"
           while [[ ! -f "${RELEASE_FIRST}" ]]; do sleep 0.05; done
           release_build_lock || status=$?
         else
           status=$?
         fi
         printf '%s\n' "${status}" > "${FIRST_STATUS}") &
        first_pid=$!
        while [[ ! -f "${FIRST_STALLED}" ]]; do sleep 0.05; done

        (status=0
         if acquire_build_lock; then
           cat "${BUILD_LOCK_DIR}/owner" > "${SECOND_OWNER}"
           : > "${SECOND_ACQUIRED}"
           while [[ ! -f "${RELEASE_SECOND}" ]]; do sleep 0.05; done
           release_build_lock || status=$?
         else
           status=$?
         fi
         printf '%s\n' "${status}" > "${SECOND_STATUS}") &
        second_pid=$!
        while [[ ! -f "${SECOND_STALLED}" ]]; do sleep 0.05; done

        : > "${RESUME_FIRST}"
        while [[ ! -f "${FIRST_ACQUIRED}" ]] && kill -0 "${first_pid}" 2>/dev/null; do
          sleep 0.05
        done
        if [[ ! -f "${FIRST_ACQUIRED}" ]]; then
          : > "${RESUME_SECOND}"
          : > "${RELEASE_FIRST}"
          : > "${RELEASE_SECOND}"
          wait "${first_pid}" || true
          wait "${second_pid}" || true
          exit 1
        fi
        : > "${RESUME_SECOND}"
        while [[ ! -f "${SECOND_ATTEMPTED}" ]]; do sleep 0.05; done

        : > "${RELEASE_FIRST}"
        wait "${first_pid}"
        [[ "$(cat "${FIRST_STATUS}")" == "0" ]]
        while [[ ! -f "${SECOND_ACQUIRED}" ]] && kill -0 "${second_pid}" 2>/dev/null; do
          sleep 0.05
        done
        [[ -f "${SECOND_ACQUIRED}" ]]
        ! cmp -s "${FIRST_OWNER}" "${SECOND_OWNER}"
        : > "${RELEASE_SECOND}"
        wait "${second_pid}"
        [[ "$(cat "${SECOND_STATUS}")" == "0" ]]
        [[ ! -e "${BUILD_LOCK_DIR}" ]]
        [[ -z "$(find "$(dirname "${BUILD_LOCK_DIR}")" \
          -maxdepth 1 -name "$(basename "${BUILD_LOCK_DIR}").generation.*" -print -quit)" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_MLX_METAL_BUILD_DIR": root.appendingPathComponent("metal-build").path,
            "MLXFAST_MLX_METAL_BUILD_LOCK_DIR": root.appendingPathComponent("metal-build.lock").path,
            "MLXFAST_MLX_METAL_BUILD_LOCK_STALE_SECONDS": "1",
            "MLXFAST_MLX_METALLIB": root.appendingPathComponent("mlx.metallib").path,
            "MLXFAST_CLANG_MODULE_CACHE": root.appendingPathComponent("clang-cache").path,
            "MLXFAST_METAL_COMPILER_HOME": root.appendingPathComponent("metal-home").path,
            "FIRST_LINK_SLOT": root.appendingPathComponent("first-link-slot").path,
            "SECOND_LINK_SLOT": root.appendingPathComponent("second-link-slot").path,
            "FIRST_STALLED": root.appendingPathComponent("first-stalled").path,
            "SECOND_STALLED": root.appendingPathComponent("second-stalled").path,
            "RESUME_FIRST": root.appendingPathComponent("resume-first").path,
            "RESUME_SECOND": root.appendingPathComponent("resume-second").path,
            "FIRST_ATTEMPTED": root.appendingPathComponent("first-attempted").path,
            "SECOND_ATTEMPTED": root.appendingPathComponent("second-attempted").path,
            "FIRST_STATUS": root.appendingPathComponent("first-status").path,
            "SECOND_STATUS": root.appendingPathComponent("second-status").path,
            "FIRST_OWNER": root.appendingPathComponent("first-owner").path,
            "SECOND_OWNER": root.appendingPathComponent("second-owner").path,
            "FIRST_ACQUIRED": root.appendingPathComponent("first-acquired").path,
            "SECOND_ACQUIRED": root.appendingPathComponent("second-acquired").path,
            "RELEASE_FIRST": root.appendingPathComponent("release-first").path,
            "RELEASE_SECOND": root.appendingPathComponent("release-second").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func metallibBuilderReclaimsOnlyDeadOrBrokenAtomicLockGenerations() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed \
          -e 's|^ROOT_DIR=.*|ROOT_DIR="${REPO_ROOT}"|' \
          -e '/^METAL_TOOLCHAIN_IDENTIFIER=/,$d' \
          "${REPO_ROOT}/tools/build-mlx-metallib.sh")"
        host_hash="$(printf '%s' "$(hostname)" | shasum -a 256 | awk '{print substr($1, 1, 16)}')"
        dead_pid=999999
        while kill -0 "${dead_pid}" 2>/dev/null; do dead_pid=$((dead_pid + 1)); done
        dead_generation="${BUILD_LOCK_DIR}.generation.dead"
        live_generation="${BUILD_LOCK_DIR}.generation.live"
        dead_initializing="${BUILD_LOCK_DIR}.initializing.${host_hash}.${dead_pid}.dead"
        live_initializing="${BUILD_LOCK_DIR}.initializing.${host_hash}.$$.live"
        mkdir "${dead_generation}" "${live_generation}" \
          "${dead_initializing}" "${live_initializing}"
        printf 'token=dead pid=%s host=%s started_at=now\n' "${dead_pid}" "$(hostname)" \
          > "${dead_generation}/owner"
        printf 'token=live pid=%s host=%s started_at=now\n' "$$" "$(hostname)" \
          > "${live_generation}/owner"
        broken_target="${BUILD_LOCK_DIR}.generation.broken"
        ln -s -h -- "${broken_target}" "${BUILD_LOCK_DIR}"

        acquire_build_lock
        [[ ! -e "${dead_generation}" ]]
        [[ ! -e "${dead_initializing}" ]]
        [[ -d "${live_generation}" ]]
        [[ -d "${live_initializing}" ]]
        release_build_lock
        [[ ! -e "${BUILD_LOCK_DIR}" && ! -L "${BUILD_LOCK_DIR}" ]]
        rm -rf "${live_generation}" "${live_initializing}"
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_MLX_METAL_BUILD_DIR": root.appendingPathComponent("metal-build").path,
            "MLXFAST_MLX_METAL_BUILD_LOCK_DIR": root.appendingPathComponent("metal-build.lock").path,
            "MLXFAST_MLX_METALLIB": root.appendingPathComponent("mlx.metallib").path,
            "MLXFAST_CLANG_MODULE_CACHE": root.appendingPathComponent("clang-cache").path,
            "MLXFAST_METAL_COMPILER_HOME": root.appendingPathComponent("metal-home").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func metallibBuilderPreservesPublishedOutputWhenCopyFails() throws {
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
          mkdir -p "${MLXFAST_MLX_METAL_BUILD_DIR}/result"
          printf 'replacement' > "${MLXFAST_MLX_METAL_BUILD_DIR}/result/mlx.metallib"
        fi
        """
    )
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("xcodebuild"),
        contents: "#!/usr/bin/env bash\nprintf 'Toolchain Identifier: metal.test\\n'\n"
    )
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("xcrun"),
        contents: "#!/usr/bin/env bash\nexit 0\n"
    )
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("cp"),
        contents: "#!/usr/bin/env bash\nprintf 'partial' > \"$2\"\nexit 23\n"
    )

    let output = root.appendingPathComponent("output/mlx.metallib")
    try FileManager.default.createDirectory(
        at: output.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try "original".write(to: output, atomically: true, encoding: .utf8)
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
    #expect(try String(contentsOf: output, encoding: .utf8) == "original")
    let outputFiles = try FileManager.default.contentsOfDirectory(
        atPath: output.deletingLastPathComponent().path
    )
    #expect(outputFiles == ["mlx.metallib"])

    let directoryOutput = root.appendingPathComponent("directory-output/mlx.metallib")
    try FileManager.default.createDirectory(
        at: directoryOutput,
        withIntermediateDirectories: true
    )
    let sentinel = directoryOutput.appendingPathComponent("sentinel")
    try "preserve".write(to: sentinel, atomically: true, encoding: .utf8)
    let directoryResult = try runSetupBash(
        """
        "${REPO_ROOT}/tools/build-mlx-metallib.sh"
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "PATH": "\(fakeBin.path):/usr/bin:/bin",
            "MLXFAST_CMAKE_BIN": fakeBin.appendingPathComponent("cmake").path,
            "MLXFAST_MLX_SWIFT_CHECKOUT": checkout.path,
            "MLXFAST_MLX_METAL_BUILD_DIR": root.appendingPathComponent("metal-build").path,
            "MLXFAST_MLX_METALLIB": directoryOutput.path,
        ]
    )

    #expect(directoryResult.status != 0)
    #expect(try String(contentsOf: sentinel, encoding: .utf8) == "preserve")
    #expect(
        try FileManager.default.contentsOfDirectory(atPath: directoryOutput.path)
            == ["sentinel"]
    )
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
    for key in environment.keys.filter({ $0.hasPrefix("MLXFAST_") }) {
        environment.removeValue(forKey: key)
    }
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
