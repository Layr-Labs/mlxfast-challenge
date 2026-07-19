import Foundation
import Testing

// Structural guards for the ranked-pipeline isolation hardening: the
// gates->timed allowlist scrub, between-phase bench-process reaping, R2
// object-key scoping, hardened git against the untrusted worktree, and the
// conditional-until-live PF egress preflight. These pin the workflow wiring so
// a refactor cannot silently reopen the closed gaps.
@Suite
struct RankedWorkflowIsolationTests {
    private func workflow() throws -> String {
        try String(contentsOfFile: ".github/workflows/serial-benchmark.yml", encoding: .utf8)
    }

    private func stepBody(_ workflow: String, from stepName: String, to nextStepName: String) throws -> String {
        let start = try #require(workflow.range(of: stepName), "missing step \(stepName)")
        let end = try #require(
            workflow.range(of: nextStepName, range: start.upperBound..<workflow.endIndex),
            "missing step \(nextStepName)"
        )
        return String(workflow[start.lowerBound..<end.lowerBound])
    }

    // F5: the private R2 object keys must not sit at job level (ambient to every
    // step, including bench phases); they belong only to the trusted download
    // step.
    @Test
    func r2ObjectKeysAreScopedToTheTrustedDownloadStepOnly() throws {
        let workflow = try workflow()

        // Assert against the actual `KEY: value` assignment (the leak vector),
        // not a bare mention -- the job-level comment intentionally names the
        // keys to explain why they are absent.
        let stepsMarker = try #require(workflow.range(of: "\n    steps:"))
        let jobHeader = String(workflow[workflow.startIndex..<stepsMarker.lowerBound])
        #expect(!jobHeader.contains(
            "MLXFAST_CORRECTNESS_GOLDEN_R2_PATH: correctness_prompts/golden_prompt_benchmark_transcription_gate_english_512_256-gemma.json"
        ))
        #expect(!jobHeader.contains(
            "MLXFAST_GPQA_R2_PATH: correctness_prompts/gpqa_reference_cases-gemma.json"
        ))
        #expect(!jobHeader.contains(
            "MLXFAST_TIMED_DECODE_PROMPT_R2_PATH: correctness_prompts/timed_decode_lowsim_prose_v1.txt"
        ))

        let prepareStep = try stepBody(
            workflow,
            from: "- name: Prepare hidden correctness golden",
            to: "- name: Attach GPQA gates and verify augmented golden"
        )
        #expect(prepareStep.contains(
            "MLXFAST_CORRECTNESS_GOLDEN_R2_PATH: correctness_prompts/golden_prompt_benchmark_transcription_gate_english_512_256-gemma.json"
        ))
        #expect(prepareStep.contains(
            "MLXFAST_GPQA_R2_PATH: correctness_prompts/gpqa_reference_cases-gemma.json"
        ))
        let prepareTimedPromptStep = try stepBody(
            workflow,
            from: "- name: Prepare hidden timed decode prompt",
            to: "- name: Wait for quiescence before timing"
        )
        #expect(prepareTimedPromptStep.contains(
            "MLXFAST_TIMED_DECODE_PROMPT_R2_PATH: correctness_prompts/timed_decode_lowsim_prose_v1.txt"
        ))
    }

    // F6: git plumbing against the untrusted submission worktree and the bench
    // workspace runs through the hardened wrapper, and no bare `git -C <ws>`
    // call survives.
    @Test
    func untrustedWorktreeGitCallsAreHardened() throws {
        let workflow = try workflow()
        let wrapper = try String(
            contentsOfFile: ".github/scripts/hardened-git.sh",
            encoding: .utf8
        )

        #expect(wrapper.contains("env -i"))
        #expect(wrapper.contains("GIT_CONFIG_GLOBAL=/dev/null"))
        #expect(wrapper.contains("GIT_CONFIG_SYSTEM=/dev/null"))
        #expect(wrapper.contains("GIT_CONFIG_NOSYSTEM=1"))
        #expect(wrapper.contains("GIT_TERMINAL_PROMPT=0"))
        #expect(wrapper.contains("HOME=/var/empty"))
        #expect(wrapper.contains("-c core.fsmonitor=false"))
        #expect(wrapper.contains("-c core.hooksPath=/dev/null"))
        #expect(wrapper.contains("-c core.pager=cat"))
        #expect(wrapper.contains("-c protocol.ext.allow=never"))

        // Submission-worktree calls go through the wrapper...
        #expect(workflow.contains("\"${hardened_git}\" -C .mlxfast-submission-src rev-parse HEAD"))
        #expect(workflow.contains("\"${hardened_git}\" -C .mlxfast-submission-src cat-file -e"))
        #expect(workflow.contains("\"${hardened_git}\" -C .mlxfast-submission-src merge-base"))
        #expect(workflow.contains("\"${hardened_git}\" fetch --no-tags .mlxfast-submission-src"))
        // ...as do the bench-workspace calls.
        #expect(workflow.contains(".github/scripts/hardened-git.sh -C \"${MLXFAST_JOB_WS}\" cat-file -e"))
        #expect(workflow.contains(".github/scripts/hardened-git.sh -C \"${MLXFAST_JOB_WS}\" update-ref"))

        // No bare (unhardened) git against either untrusted tree remains.
        #expect(!workflow.contains("git -C .mlxfast-submission-src"))
        #expect(!workflow.contains("git -C \"${MLXFAST_JOB_WS}\""))
        #expect(!workflow.contains(" git fetch --no-tags .mlxfast-submission-src"))
    }

    // F1: the pre-hidden snapshot is taken before hidden material enters the
    // workspace, and the scrub is an allowlist restore (not a golden-name
    // denylist) that also re-verifies weights/.
    @Test
    func gatesToTimedScrubIsAnAllowlistWithWeightsReverify() throws {
        let workflow = try workflow()

        let snapshotRange = try #require(workflow.range(of: "- name: Snapshot post-transform workspace"))
        let prepareGoldenRange = try #require(workflow.range(of: "- name: Prepare hidden correctness golden"))
        let scrubRange = try #require(workflow.range(of: "- name: Scrub hidden material from bench workspace"))
        let quiescenceRange = try #require(workflow.range(of: "- name: Wait for quiescence before timing"))

        // Snapshot precedes any hidden material; scrub precedes timing.
        #expect(snapshotRange.lowerBound < prepareGoldenRange.lowerBound)
        #expect(prepareGoldenRange.lowerBound < scrubRange.lowerBound)
        #expect(scrubRange.lowerBound < quiescenceRange.lowerBound)

        let snapshotStep = try stepBody(
            workflow,
            from: "- name: Snapshot post-transform workspace",
            to: "- name: Prepare hidden correctness golden"
        )
        #expect(snapshotStep.contains("post-transform-manifest.nul"))
        #expect(snapshotStep.contains("-type f -o -type l"))
        #expect(snapshotStep.contains("-print0"))
        #expect(snapshotStep.contains("sort -z"))
        // The bench-owned TMPDIR scratch is purged THROUGH the bridge before
        // the manifest is recorded (the runner uid cannot unlink inside
        // bench-owned dirs), and only bench-private Permission denied subtrees
        // are tolerated by the manifest walk -- anything else fails closed.
        #expect(snapshotStep.contains(
            "find \"${MJOB_WS}/tmp\" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +"
        ))
        #expect(snapshotStep.contains("post-transform-find.err"))
        #expect(snapshotStep.contains("only bench-private Permission denied subtrees are tolerated"))

        let scrubStep = try stepBody(
            workflow,
            from: "- name: Scrub hidden material from bench workspace",
            to: "- name: Reap lingering bench processes before timing"
        )
        // Explicit hidden-file removal is kept.
        #expect(scrubStep.contains("rm -f \"${MLXFAST_JOB_WS}/correctness_golden_ranked.json\""))
        #expect(scrubStep.contains("rm -rf \"${MLXFAST_JOB_WS}/private\""))
        // Gates-phase bench TMPDIR scratch is purged through the bridge too,
        // so 0700 bench scratch cannot bridge hidden-phase bytes into timing.
        #expect(scrubStep.contains(
            "find \"${MJOB_WS}/tmp\" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +"
        ))
        // Allowlist diff against the pre-hidden snapshot, fail-closed.
        #expect(scrubStep.contains("post-transform-manifest.nul"))
        #expect(scrubStep.contains("grep -zqxF -- \"${rel}\" \"${manifest}\""))
        #expect(scrubStep.contains("./weights/*|./.build/*|./.build-worker/*|./correctness_prompts/*"))
        #expect(scrubStep.contains("./bench_oracle*|./private/*"))
        #expect(scrubStep.contains("allowlist scrub could not remove gates-phase workspace file"))
        // weights/ re-verified against the pinned post-transform hash.
        #expect(scrubStep.contains(".github/scripts/hash-weights-directory.sh \"${MLXFAST_JOB_WS}/weights\""))
        #expect(scrubStep.contains("transformed weights changed before timing"))
    }

    // F2: bench-uid stragglers are reaped between the untrusted phases, through
    // the bench-exec bridge, using the trusted reaper script.
    @Test
    func benchProcessesAreReapedBetweenPhases() throws {
        let workflow = try workflow()
        let reaper = try String(
            contentsOfFile: ".github/scripts/reap-bench-processes.sh",
            encoding: .utf8
        )

        let reapBeforeHiddenRange = try #require(
            workflow.range(of: "- name: Reap lingering bench processes before hidden material")
        )
        let prepareGoldenRange = try #require(workflow.range(of: "- name: Prepare hidden correctness golden"))
        let scrubRange = try #require(workflow.range(of: "- name: Scrub hidden material from bench workspace"))
        let reapBeforeTimingRange = try #require(
            workflow.range(of: "- name: Reap lingering bench processes before timing")
        )
        let quiescenceRange = try #require(workflow.range(of: "- name: Wait for quiescence before timing"))

        // Reap #1 before hidden material is placed; reap #2 between the gates
        // scrub and the timed measurement.
        #expect(reapBeforeHiddenRange.lowerBound < prepareGoldenRange.lowerBound)
        #expect(scrubRange.lowerBound < reapBeforeTimingRange.lowerBound)
        #expect(reapBeforeTimingRange.lowerBound < quiescenceRange.lowerBound)

        // Both invocations cross the bench-exec bridge and run the trusted
        // reaper from the workspace copy.
        let reapCount = workflow.components(
            separatedBy: "/bin/bash \"${MLXFAST_JOB_WS}/.github/scripts/reap-bench-processes.sh\""
        ).count - 1
        #expect(reapCount == 2)

        // The reaper refuses to signal a privileged uid, protects its own tree,
        // and escalates TERM -> KILL.
        #expect(reaper.contains("bench_uid=\"$(id -u)\""))
        #expect(reaper.contains("-lt 500"))
        #expect(reaper.contains("refusing to reap privileged/system uid"))
        #expect(reaper.contains("protected_pids"))
        #expect(reaper.contains("reap_signal TERM"))
        #expect(reaper.contains("reap_signal KILL"))
        #expect(reaper.contains("awk -v u=\"${bench_uid}\" '$3 == u { print $1, $2 }'"))
    }

    // F7: the bench egress preflight runs before any submitted code (before the
    // first bench build) and is CONDITIONAL-UNTIL-LIVE: it verifies (passes)
    // when the bench probe is blocked, fails closed when egress is open while
    // the PF lockdown is provisioned (config-detected or explicitly required),
    // and stays dormant with a warning while the lockdown has not shipped. It
    // must never fail merely because the staged PF block is still deferred.
    @Test
    func benchEgressPreflightIsConditionalUntilPFLockdownShips() throws {
        let workflow = try workflow()

        let egressRange = try #require(workflow.range(of: "- name: Assert bench network egress lockdown"))
        let prepareWorkspaceRange = try #require(workflow.range(of: "- name: Prepare bench workspace"))
        let buildRange = try #require(workflow.range(of: "- name: Build trusted CLI in bench sandbox"))

        // After the workspace exists (bench-exec needs it) but before the first
        // submission-built-code execution (build).
        #expect(prepareWorkspaceRange.lowerBound < egressRange.lowerBound)
        #expect(egressRange.lowerBound < buildRange.lowerBound)

        let egressStep = try stepBody(
            workflow,
            from: "- name: Assert bench network egress lockdown",
            to: "- name: Build trusted CLI in bench sandbox"
        )
        // Functional probe AS bench through the bridge, with a trusted-uid
        // control probe so "box offline" is never mistaken for "PF live".
        #expect(egressStep.contains("sudo -n \"${MLXFAST_BENCH_EXEC}\" \"${MLXFAST_JOB_WS}\" /bin/bash -c \"${probe}\""))
        #expect(egressStep.contains("trusted-uid egress probe failed"))
        // Verified branch: a blocked bench probe passes (and starts verifying
        // automatically once the operator ships the PF block).
        #expect(egressStep.contains(
            "benchmark: bench network egress lockdown verified (bench-uid egress blocked)"
        ))
        // Enforcing branch: open egress with the lockdown provisioned fails
        // closed. Provisioning is detected from the world-readable PF config
        // or forced by the explicit override.
        #expect(egressStep.contains("/etc/pf.conf /etc/pf.anchors/*"))
        #expect(egressStep.contains("MLXFAST_REQUIRE_BENCH_EGRESS_LOCKDOWN"))
        #expect(egressStep.contains("bench network egress is NOT locked down"))
        #expect(egressStep.contains("refusing to run submitted code"))
        // Dormant branch: no provisioned lockdown -> warn, never abort.
        #expect(egressStep.contains("PF bench-egress lockdown not yet active on this box"))
        #expect(egressStep.contains("It will enforce automatically once the operator ships the PF lockdown."))
        // The probe never reports "blocked" vacuously: a missing curl is a
        // distinct no-verdict exit, and the probe pins the absolute path.
        #expect(egressStep.contains("[ -x /usr/bin/curl ] || exit 2"))
    }
}
