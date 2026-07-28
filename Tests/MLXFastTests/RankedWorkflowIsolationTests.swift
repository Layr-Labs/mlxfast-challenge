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
        try String(contentsOfFile: ".github/workflows/benchmark.yml", encoding: .utf8)
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
            "MLXFAST_CORRECTNESS_GOLDEN_R2_PATH: correctness_prompts/laguna-xs-2.1-serial-v2/"
        ))
        #expect(!jobHeader.contains(
            "MLXFAST_GPQA_R2_PATH: correctness_prompts/laguna-xs-2.1-serial-v2/"
        ))
        #expect(!jobHeader.contains(
            "MLXFAST_TIMED_DECODE_PROMPT_R2_PATH: correctness_prompts/laguna-xs-2.1-serial-v2/"
        ))

        let prepareStep = try stepBody(
            workflow,
            from: "- name: Prepare hidden correctness golden",
            to: "- name: Attach GPQA gates and verify augmented golden"
        )
        #expect(prepareStep.contains(
            "MLXFAST_CORRECTNESS_GOLDEN_R2_PATH: correctness_prompts/laguna-xs-2.1-serial-v2/hidden-correctness-golden-94239d59b435eb8f370c82bcf8c86822d1bbc1094e3650aeff3abc5558137023.json"
        ))
        #expect(prepareStep.contains(
            "MLXFAST_GPQA_R2_PATH: correctness_prompts/laguna-xs-2.1-serial-v2/gpqa-reference-cases-6e149e58908d086091ed7e7980e9f8beca7deb4d04063035d68d23e70ee1726f.json"
        ))
        let prepareTimedPromptStep = try stepBody(
            workflow,
            from: "- name: Prepare hidden timed decode prompt",
            to: "- name: Wait for quiescence before timing"
        )
        #expect(prepareTimedPromptStep.contains(
            "MLXFAST_TIMED_DECODE_PROMPT_R2_PATH: correctness_prompts/laguna-xs-2.1-serial-v2/timed-decode-prompt-0b67162cbea948f380e693398b19ba797892b5100cd9e0e415a87e900ac79e03.txt"
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

        // Submission-worktree calls go through the wrapper... (the former
        // merge-base ancestry probe is gone: the modifiable-surface check
        // diffs the candidate tree against the current trusted main tip
        // directly, so no ancestry plumbing runs against the untrusted
        // worktree at all.)
        #expect(workflow.contains("\"${hardened_git}\" -C .mlxfast-submission-src rev-parse HEAD"))
        #expect(workflow.contains("\"${hardened_git}\" -C .mlxfast-submission-src cat-file -e"))
        #expect(workflow.contains("\"${hardened_git}\" fetch --no-tags .mlxfast-submission-src"))
        // ...as do the bench-workspace calls.
        #expect(workflow.contains(".github/scripts/hardened-git.sh -C \"${MLXFAST_JOB_WS}\" cat-file -e"))
        #expect(workflow.contains(".github/scripts/hardened-git.sh -C \"${MLXFAST_JOB_WS}\" update-ref"))

        // No bare (unhardened) git against either untrusted tree remains.
        #expect(!workflow.contains("git -C .mlxfast-submission-src"))
        #expect(!workflow.contains("git -C \"${MLXFAST_JOB_WS}\""))
        #expect(!workflow.contains(" git fetch --no-tags .mlxfast-submission-src"))

        // The doctrine also binds the scripts the workflow runs WITH the
        // untrusted submission checkout as their working directory:
        // enforce-modifiable-surface.sh and run-submission-static-review.sh
        // read commits/blobs from that repository, so their git calls go
        // through the wrapper too -- resolved next to the script itself (the
        // trusted checkout's copy), never one inside the submission worktree.
        let enforceSurface = try String(
            contentsOfFile: ".github/scripts/enforce-modifiable-surface.sh",
            encoding: .utf8
        )
        #expect(enforceSurface.contains(
            "SCRIPT_DIR=\"$(cd -- \"$(dirname -- \"${BASH_SOURCE[0]}\")\" >/dev/null && pwd -P)\""
        ))
        #expect(enforceSurface.contains("HARDENED_GIT=\"${SCRIPT_DIR}/hardened-git.sh\""))
        #expect(enforceSurface.contains(
            "allowed=\"$(\"${HARDENED_GIT}\" show \"${BASE_SHA}:benchmark.json\" | jq -r '.editablePaths[]')\""
        ))
        #expect(enforceSurface.contains(
            "changed=\"$(\"${HARDENED_GIT}\" diff --name-only \"${BASE_SHA}\" \"${HEAD_SHA}\")\""
        ))

        let staticReview = try String(
            contentsOfFile: ".github/scripts/run-submission-static-review.sh",
            encoding: .utf8
        )
        #expect(staticReview.contains("HARDENED_GIT=\"${SCRIPT_DIR}/hardened-git.sh\""))
        // Every plumbing family the script uses is wrapper-routed.
        #expect(staticReview.contains("\"${HARDENED_GIT}\" rev-parse --is-inside-work-tree"))
        #expect(staticReview.contains("\"${HARDENED_GIT}\" rev-parse --verify --quiet"))
        #expect(staticReview.contains("\"$(\"${HARDENED_GIT}\" rev-parse HEAD)\""))
        #expect(staticReview.contains("\"${HARDENED_GIT}\" show \"${review_base}:${CONTRACT_PATH}\""))
        #expect(staticReview.contains("\"${HARDENED_GIT}\" diff --name-only -z"))
        #expect(staticReview.contains("\"${HARDENED_GIT}\" cat-file -e"))
        #expect(staticReview.contains("\"${HARDENED_GIT}\" cat-file -s"))
        #expect(staticReview.contains(
            "\"${HARDENED_GIT}\" diff \"${review_base}\" \"${review_head}\" -- \"${editable_paths[@]}\""
        ))

        // No bare git invocation survives in either script: every non-comment
        // line mentioning `git ` must be the binary-presence check or a
        // diagnostic string (wrapper calls spell the uppercase HARDENED_GIT
        // variable, so they never match `git ` at all).
        let allowedBareGitFragments = [
            "command -v git",
            "did git merge-base fail",
            "not a git work tree",
        ]
        for script in [enforceSurface, staticReview] {
            for rawLine in script.components(separatedBy: "\n") {
                let line = rawLine.trimmingCharacters(in: .whitespaces)
                if line.hasPrefix("#") { continue }
                guard line.contains("git ") else { continue }
                #expect(
                    allowedBareGitFragments.contains(where: { line.contains($0) }),
                    "unhardened git call: \(line)"
                )
            }
        }
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
