import Foundation
import Testing

// Structural guards for the DISTINCT MTP ranked workflow draft
// (.github/workflows/mtp-benchmark.yml): it must stay inert until the
// operator deliberately enables the track on trusted main, it must remain
// fully separate from the serial ranked pipeline (its own track id,
// workspace, artifacts, and baseline), and the serial pipeline must remain
// MTP-free. These pin the wiring so a refactor cannot silently enable
// scoring or blend the two tracks.
@Suite
struct MTPWorkflowIsolationTests {
    private func mtpWorkflow() throws -> String {
        try String(contentsOfFile: ".github/workflows/mtp-benchmark.yml", encoding: .utf8)
    }

    private func serialWorkflow() throws -> String {
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

    // The MTP workflow is dispatch-only (self-hosted safety) and runs only on
    // the allowlisted ranked branch namespaces, enforced by its own pinned
    // trusted-workflow script — never the serial track's.
    @Test
    func mtpWorkflowIsDispatchOnlyWithItsOwnTrustedPin() throws {
        let workflow = try mtpWorkflow()
        #expect(workflow.contains("workflow_dispatch:"))
        #expect(!workflow.contains("pull_request"))
        #expect(!workflow.contains("\n  push:"))
        #expect(workflow.contains(".github/scripts/enforce-trusted-mtp-benchmark-workflow.sh"))

        let script = try String(
            contentsOfFile: ".github/scripts/enforce-trusted-mtp-benchmark-workflow.sh",
            encoding: .utf8
        )
        #expect(script.contains("WORKFLOW_PATH=\".github/workflows/mtp-benchmark.yml\""))
        #expect(script.contains("workflow_dispatch"))
        #expect(script.contains("refs/heads/main|refs/heads/submissions/*|refs/heads/baseline/*|refs/heads/yukon/baseline/*"))

        // The serial pipeline's pin still points at benchmark.yml: neither
        // track can dispatch the other's workflow against its private
        // material.
        let serialScript = try String(
            contentsOfFile: ".github/scripts/enforce-trusted-benchmark-workflow.sh",
            encoding: .utf8
        )
        #expect(serialScript.contains("WORKFLOW_PATH=\".github/workflows/benchmark.yml\""))
    }

    // THE INERT GATE: the enablement step fails closed while the trusted
    // contract fixture carries official_scoring_enabled=false or
    // publication_allowed=false, while any hidden-golden pin is empty, or
    // without the explicit operator dispatch interlock — and it is ordered
    // before any secret use, submitted-code handling, or bench execution.
    @Test
    func mtpWorkflowIsInertUntilOperatorEnablement() throws {
        let workflow = try mtpWorkflow()

        let enablement = try #require(workflow.range(of: "- name: Enforce MTP track enablement"))
        let submissionCheckout = try #require(workflow.range(of: "- name: Checkout submitted editable paths"))
        let privateMaterial = try #require(workflow.range(of: "- name: Check private material present"))
        let workspace = try #require(workflow.range(of: "- name: Prepare bench workspace"))
        let build = try #require(workflow.range(of: "- name: Build harness in bench sandbox"))
        #expect(enablement.lowerBound < submissionCheckout.lowerBound)
        #expect(enablement.lowerBound < privateMaterial.lowerBound)
        #expect(enablement.lowerBound < workspace.lowerBound)
        #expect(enablement.lowerBound < build.lowerBound)

        let gate = try stepBody(
            workflow,
            from: "- name: Enforce MTP track enablement",
            to: "- name: MTP host preflight"
        )
        #expect(gate.contains("jq -r '.official_scoring_enabled'"))
        #expect(gate.contains("jq -r '.reference_baseline.publication_allowed'"))
        #expect(gate.contains("!= \"true\""))
        #expect(gate.contains("deliberately inert until the operator completes"))
        #expect(gate.contains("confirm_track_enabled=true"))
        #expect(gate.contains("MLXFAST_MTP_CORRECTNESS_GOLDEN_SHA256"))
        #expect(gate.contains("MLXFAST_MTP_BENCH_GOLDEN_SHA256"))
        #expect(gate.contains("is empty; freeze the IT-target goldens"))

        // The pins ship EMPTY in this draft: the operator fills them in the
        // same trusted commit that flips the contract fixture.
        #expect(workflow.contains("MLXFAST_MTP_CORRECTNESS_GOLDEN_SHA256: \"\""))
        #expect(workflow.contains("MLXFAST_MTP_CORRECTNESS_GOLDEN_BYTES: \"\""))
        #expect(workflow.contains("MLXFAST_MTP_BENCH_GOLDEN_SHA256: \"\""))
        #expect(workflow.contains("MLXFAST_MTP_BENCH_GOLDEN_BYTES: \"\""))

        // The dispatch interlock defaults to false.
        let confirmInput = try stepBody(
            workflow,
            from: "confirm_track_enabled:",
            to: "run_benchmark:"
        )
        #expect(confirmInput.contains("default: false"))

        // And the trusted contract the gate reads is still disabled (the
        // ExperimentalMTPTests contract test pins this too; asserting here
        // keeps workflow + fixture in one reviewable invariant).
        let contract = try Data(contentsOf: URL(fileURLWithPath: "fixtures/gemma_4_31b_it_mtp_track.json"))
        let json = try #require(try JSONSerialization.jsonObject(with: contract) as? [String: Any])
        #expect(json["official_scoring_enabled"] as? Bool == false)
        let baseline = try #require(json["reference_baseline"] as? [String: Any])
        #expect(baseline["publication_allowed"] as? Bool == false)
    }

    // The two ranked tracks never share mutable identity: distinct workspace,
    // concurrency group, artifact names, baseline tree, calibration, and
    // score file. The MTP score carries its own track id and never writes
    // score.json.
    @Test
    func mtpTrackIdentityIsFullySeparateFromSerialTrack() throws {
        let workflow = try mtpWorkflow()

        #expect(workflow.contains("MLXFAST_MTP_TRACK_ID: gemma4-31b-it-mtp-v1"))
        #expect(workflow.contains("MLXFAST_JOB_WS: /Users/Shared/bench-jobs/mtp-ranked-current"))
        #expect(workflow.contains("group: mlxfast-mtp-ranked-"))
        #expect(workflow.contains("MLXFAST_MTP_BASELINE_WS: /opt/bench-runner/mtp-baseline/current"))
        #expect(workflow.contains("MLXFAST_MTP_BASELINE_CALIBRATION: /opt/bench-runner/state/mtp-baseline-calibration.json"))
        #expect(workflow.contains("name: mtp-benchmark-results-"))
        #expect(workflow.contains("name: mtp-correctness-results-"))
        #expect(workflow.contains("name: mtp-benchmark-audit-"))
        #expect(workflow.contains("track_id: $track"))
        #expect(workflow.contains("> mtp-score.json"))

        // Never the serial track's identity: the serial workspace, the
        // serial baseline, the serial artifact namespaces, or the serial
        // score file name.
        #expect(!workflow.contains("/Users/Shared/bench-jobs/ranked-current"))
        #expect(!workflow.contains("/opt/bench-runner/baseline/current"))
        #expect(!workflow.contains("name: benchmark-results-"))
        #expect(!workflow.contains("state/baseline-calibration.json"))
        #expect(!workflow.contains("> score.json"))

        // The MTP measurement is owned by the MTP measure wrapper, not the
        // serial track's measure-job.sh.
        #expect(workflow.contains("MLXFAST_MTP_MEASURE_JOB: /opt/bench-runner/measure-mtp-job.sh"))
        #expect(!workflow.contains("MLXFAST_MEASURE_JOB: /opt/bench-runner/measure-job.sh"))

        // Trusted caches are restore-only here: only the serial pipeline's
        // main-ref dispatches may populate the trusted cache namespaces.
        #expect(workflow.contains("actions/cache/restore@"))
        #expect(!workflow.contains("actions/cache/save@"))
    }

    // The serial ranked pipeline stays MTP-free: enabling, disabling, or
    // editing the MTP track must never touch the serial score path.
    @Test
    func serialRankedPipelineRemainsMTPFree() throws {
        let serial = try serialWorkflow()
        #expect(!serial.contains("mtp"))
        #expect(!serial.contains("MTP"))
        #expect(!serial.contains("MLXFAST_MTP_"))
        #expect(!serial.contains("gemma4-31b-it"))
    }

    // Submission dispatches run the static review under the MTP track policy
    // and overlay editable paths over trusted main, exactly like the serial
    // pipeline's submission handling.
    @Test
    func mtpSubmissionsAreReviewedUnderTheMTPTrackPolicy() throws {
        let workflow = try mtpWorkflow()
        let review = try stepBody(
            workflow,
            from: "- name: Review submitted code for benchmark bypasses (MTP policy)",
            to: "- name: Overlay submitted editable paths"
        )
        #expect(review.contains("MLXFAST_SUBMISSION_TRACK_ID=\"${MLXFAST_MTP_TRACK_ID}\""))
        #expect(review.contains("run-submission-static-review.sh"))
        #expect(workflow.contains(".github/scripts/enforce-modifiable-surface.sh"))
        #expect(workflow.contains(".github/scripts/overlay-editable-paths.sh"))

        // The review script accepts the MTP track id (pinned by the review
        // script's own allowlist).
        let reviewScript = try String(
            contentsOfFile: ".github/scripts/run-submission-static-review.sh",
            encoding: .utf8
        )
        #expect(reviewScript.contains("serial|gemma4-31b-it-mtp-v1"))
    }

    // Decode-only paired scoring with the hard gates: bit-exact parity across
    // every timed run, a minimum accepted pair count, and the 1.0 floor. The
    // hidden benchmark golden reaches the workspace only inside measure-job's
    // timed sides, and the correctness golden is scrubbed before timing.
    @Test
    func mtpScoringEnforcesBitExactGateMinimumPairsAndFloor() throws {
        let workflow = try mtpWorkflow()

        #expect(workflow.contains("MLXFAST_MTP_DECODE_SPEEDUP_FLOOR: \"1.0\""))
        #expect(workflow.contains("MLXFAST_MTP_MIN_ACCEPTED_PAIRS: \"3\""))

        let score = try stepBody(
            workflow,
            from: "- name: Compute MTP score and enforce floor",
            to: "- name: Check MTP benchmark artifact paths"
        )
        #expect(score.contains(".parity_all_ok == true"))
        #expect(score.contains(".accepted_pair_count >= $min_pairs"))
        #expect(score.contains("is below the ${MLXFAST_MTP_DECODE_SPEEDUP_FLOOR} floor"))
        #expect(score.contains("mtp_decode_speedup_mean"))

        // Ordering: correctness gate -> scrub -> reap -> quiescence ->
        // harness re-verify -> timed measurement -> score.
        let gates = try #require(workflow.range(of: "- name: MTP correctness and parity gate (untimed)"))
        let scrub = try #require(workflow.range(of: "- name: Scrub hidden material from bench workspace"))
        let reap = try #require(workflow.range(of: "- name: Reap lingering bench processes before timing"))
        let quiesce = try #require(workflow.range(of: "- name: Wait for quiescence before timing"))
        let verify = try #require(workflow.range(of: "- name: Verify trusted harness before timing"))
        let timed = try #require(workflow.range(of: "- name: Timed paired MTP benchmark (measure-mtp-job)"))
        let scoreStep = try #require(workflow.range(of: "- name: Compute MTP score and enforce floor"))
        #expect(gates.lowerBound < scrub.lowerBound)
        #expect(scrub.lowerBound < reap.lowerBound)
        #expect(reap.lowerBound < quiesce.lowerBound)
        #expect(quiesce.lowerBound < verify.lowerBound)
        #expect(verify.lowerBound < timed.lowerBound)
        #expect(timed.lowerBound < scoreStep.lowerBound)

        // The untimed gate seals stdout in the trusted shell and requires
        // every returned token to match the parent-owned serial oracle.
        let gateBody = try stepBody(
            workflow,
            from: "- name: MTP correctness and parity gate (untimed)",
            to: "- name: Stage MTP correctness artifacts"
        )
        #expect(gateBody.contains(".all_tokens_matched == true"))
        #expect(gateBody.contains(".uses_trained_drafter == true"))
        #expect(gateBody.contains(".official_score_produced == false"))
        #expect(gateBody.contains("BENCH_GOLDEN_PATH"))
        #expect(gateBody.contains("--require-trained-assistant"))

        // The R2 object keys are scoped to the one trusted download step,
        // never job-level env (same discipline as the serial pipeline).
        let stepsMarker = try #require(workflow.range(of: "\n    steps:"))
        let jobHeader = String(workflow[workflow.startIndex..<stepsMarker.lowerBound])
        #expect(!jobHeader.contains("MLXFAST_MTP_CORRECTNESS_GOLDEN_R2_PATH:"))
        #expect(!jobHeader.contains("MLXFAST_MTP_BENCH_GOLDEN_R2_PATH:"))
        let download = try stepBody(
            workflow,
            from: "- name: Prepare hidden MTP goldens",
            to: "- name: Verify trusted harness before MTP correctness gate"
        )
        #expect(download.contains("MLXFAST_MTP_CORRECTNESS_GOLDEN_R2_PATH:"))
        #expect(download.contains("MLXFAST_MTP_BENCH_GOLDEN_R2_PATH:"))

        // The scrub removes the hidden correctness golden and re-verifies the
        // transformed MTP weights before any timed worker spawns.
        let scrubBody = try stepBody(
            workflow,
            from: "- name: Scrub hidden material from bench workspace",
            to: "- name: Reap lingering bench processes before timing"
        )
        #expect(scrubBody.contains("rm -f \"${MLXFAST_JOB_WS}/.mtp-ranked-src/mtp_correctness_golden.json\""))
        #expect(scrubBody.contains("hash-weights-directory.sh \"${MLXFAST_JOB_WS}/mtp-weights\""))
        #expect(scrubBody.contains("transformed MTP weights changed before timing"))
    }
}
