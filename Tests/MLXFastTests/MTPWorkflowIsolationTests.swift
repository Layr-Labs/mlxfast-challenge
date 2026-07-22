import Foundation
import Testing

// Structural guards for the DEFAULT MTP ranked workflow
// (.github/workflows/benchmark.yml): it remains fail-closed on the trusted
// contract and hidden pins, accepts Yukon's input-less workflow dispatch, and
// stays isolated from the archived serial workflow and manifest.
@Suite
struct MTPWorkflowIsolationTests {
    private func mtpWorkflow() throws -> String {
        try String(contentsOfFile: ".github/workflows/benchmark.yml", encoding: .utf8)
    }

    private func serialWorkflow() throws -> String {
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

    // The MTP workflow is dispatch-only (self-hosted safety) and runs only on
    // the allowlisted ranked branch namespaces, enforced by its own pinned
    // trusted-workflow script — never the serial track's.
    @Test
    func mtpWorkflowIsDispatchOnlyWithItsOwnTrustedPin() throws {
        let workflow = try mtpWorkflow()
        #expect(workflow.contains("workflow_dispatch:"))
        #expect(!workflow.contains("pull_request"))
        #expect(!workflow.contains("\n  push:"))
        #expect(workflow.contains(".github/scripts/enforce-trusted-benchmark-workflow.sh"))

        let script = try String(
            contentsOfFile: ".github/scripts/enforce-trusted-benchmark-workflow.sh",
            encoding: .utf8
        )
        #expect(script.contains("WORKFLOW_PATH=\".github/workflows/benchmark.yml\""))
        #expect(script.contains("workflow_dispatch"))
        #expect(script.contains("refs/heads/main|refs/heads/submissions/*|refs/heads/baseline/*|refs/heads/yukon/baseline/*"))

        // The serial pipeline's pin points at serial-benchmark.yml: neither
        // track can dispatch the other's workflow against its private
        // material.
        let serialScript = try String(
            contentsOfFile: ".github/scripts/enforce-trusted-serial-benchmark-workflow.sh",
            encoding: .utf8
        )
        #expect(serialScript.contains("WORKFLOW_PATH=\".github/workflows/serial-benchmark.yml\""))
    }

    // THE ENABLEMENT GATE (Laguna pre-go-live shape): the gate is standing
    // and fail-closed — it refuses a disabled contract, an emptied pin, or
    // a dispatch without the explicit operator interlock — and it stays
    // ordered before any secret use, submitted-code handling, or bench
    // execution. The Laguna trusted contract ships DISABLED until the
    // hidden goldens and paired baseline are regenerated on m5-bench
    // (docs/mtp-track-golive-runbook.md), so the workflow is deliberately
    // inert while the gate structure stays pinned.
    @Test
    func mtpTrackGateStandsFailClosedWithPinnedGoldensPendingLagunaGoLive() throws {
        let workflow = try mtpWorkflow()

        let enablement = try #require(workflow.range(of: "- name: Enforce MTP track enablement"))
        let submissionCheckout = try #require(workflow.range(of: "- name: Checkout submitted editable paths"))
        let privateMaterial = try #require(workflow.range(of: "- name: Check private material present"))
        let workspace = try #require(workflow.range(of: "- name: Prepare bench workspace"))
        let build = try #require(workflow.range(of: "- name: Build trusted CLI in bench sandbox"))
        let workerBuild = try #require(workflow.range(of: "- name: Build participant worker in bench sandbox"))
        #expect(enablement.lowerBound < submissionCheckout.lowerBound)
        #expect(enablement.lowerBound < privateMaterial.lowerBound)
        #expect(enablement.lowerBound < workspace.lowerBound)
        #expect(enablement.lowerBound < build.lowerBound)
        #expect(build.lowerBound < workerBuild.lowerBound)
        #expect(workflow.contains(
            "/usr/bin/swift build -c release --force-resolved-versions --product mlxfast-swift"
        ))
        #expect(workflow.contains(
            "/usr/bin/swift build -c release --force-resolved-versions --scratch-path .build-worker --product mlxfast-runtime-worker"
        ))
        #expect(workflow.contains(
            "test -x \"${MLXFAST_JOB_WS}/.build-worker/release/mlxfast-runtime-worker\""
        ))
        #expect(workflow.contains("exec .build/release/mlxfast-swift transform"))
        #expect(workflow.contains("exec .build/release/mlxfast-swift mtp-benchmark"))
        #expect(!workflow.contains("mlxfast-runtime-worker transform"))
        #expect(!workflow.contains("mlxfast-runtime-worker mtp-benchmark"))

        let gate = try stepBody(
            workflow,
            from: "- name: Enforce MTP track enablement",
            to: "- name: MTP host preflight"
        )
        #expect(gate.contains("jq -r '.official_scoring_enabled'"))
        #expect(gate.contains("jq -r '.reference_baseline.publication_allowed'"))
        #expect(gate.contains("!= \"true\""))
        #expect(gate.contains("confirm_track_enabled=true"))
        #expect(gate.contains("MLXFAST_MTP_CORRECTNESS_GOLDEN_SHA256"))
        #expect(gate.contains("MLXFAST_MTP_BENCH_GOLDEN_SHA256"))
        #expect(gate.contains("is empty; freeze the IT-target goldens"))

        // The pins still carry the frozen 2026-07-15 Gemma-era golden
        // identities. TODO(operator): they are regenerated for Laguna on
        // m5-bench together with the workflow env re-pin (go-live runbook
        // step B); until then a dispatch fails closed on the disabled
        // contract before any golden is fetched.
        #expect(workflow.contains(
            "MLXFAST_MTP_CORRECTNESS_GOLDEN_SHA256: d4fff3fe015123c395be68cb19401709ec288e44a04e52c2ea36c32220356862"
        ))
        #expect(workflow.contains("MLXFAST_MTP_CORRECTNESS_GOLDEN_BYTES: \"14683\""))
        #expect(workflow.contains(
            "MLXFAST_MTP_BENCH_GOLDEN_SHA256: a472d6e40c3bada936fa39535753fec609d836e709db03274222a4a303ae6f58"
        ))
        #expect(workflow.contains("MLXFAST_MTP_BENCH_GOLDEN_BYTES: \"14982\""))

        // Yukon dispatches runner.workflow with no custom inputs. The live
        // default therefore opts in by default while the trusted contract,
        // publication flag, and hidden pins remain fail-closed controls.
        let confirmInput = try stepBody(
            workflow,
            from: "confirm_track_enabled:",
            to: "run_benchmark:"
        )
        #expect(confirmInput.contains("default: true"))

        // And the trusted contract the gate reads is DISABLED: the Laguna
        // re-pin ships fail-closed until the m5-bench go-live regenerates
        // the hidden goldens, weight manifests, and paired baseline (the
        // ExperimentalMTPTests contract test pins this too; asserting here
        // keeps workflow + fixture in one reviewable invariant).
        // TODO(operator): the enablement commit flips these to
        // true/true/"established".
        let contract = try Data(contentsOf: URL(fileURLWithPath: "fixtures/laguna_xs_2_1_mtp_track.json"))
        let json = try #require(try JSONSerialization.jsonObject(with: contract) as? [String: Any])
        #expect(json["official_scoring_enabled"] as? Bool == false)
        let baseline = try #require(json["reference_baseline"] as? [String: Any])
        #expect(baseline["publication_allowed"] as? Bool == false)
        #expect(baseline["status"] as? String == "pending_m5_rebaseline")
    }

    // The two ranked tracks never share mutable runtime identity: distinct
    // workspace, concurrency group, baseline tree, and calibration. The
    // default MTP score uses Yukon's conventional score.json path and carries
    // its track id inside the payload.
    @Test
    func mtpTrackIdentityIsFullySeparateFromSerialTrack() throws {
        let workflow = try mtpWorkflow()

        // TODO(operator): the workflow env still carries the retired
        // gemma4-31b-it-mtp-v1 id; it rotates to laguna-xs-2.1-mtp-v1 with
        // the .github workflow re-pin (go-live runbook), which keeps the
        // workflow inert against the Laguna contract's track_id until then.
        #expect(workflow.contains("MLXFAST_MTP_TRACK_ID: gemma4-31b-it-mtp-v1"))
        #expect(workflow.contains("MLXFAST_JOB_WS: /Users/Shared/bench-jobs/mtp-ranked-current"))
        #expect(workflow.contains("group: mlxfast-mtp-ranked-"))
        #expect(workflow.contains("MLXFAST_MTP_BASELINE_WS: /opt/bench-runner/mtp-baseline/current"))
        #expect(workflow.contains("MLXFAST_MTP_BASELINE_CALIBRATION: /opt/bench-runner/state/mtp-baseline-calibration.json"))
        #expect(workflow.contains("name: benchmark-results-"))
        #expect(workflow.contains("name: mtp-correctness-results-"))
        #expect(workflow.contains("name: mtp-benchmark-audit-"))
        #expect(workflow.contains("track_id: $track"))
        #expect(workflow.contains("> score.json"))

        // Never the serial track's identity: the serial workspace, the
        // serial baseline, the serial artifact namespaces, or the serial
        // score file name.
        #expect(!workflow.contains("/Users/Shared/bench-jobs/ranked-current"))
        #expect(!workflow.contains("/opt/bench-runner/baseline/current"))
        #expect(!workflow.contains("state/baseline-calibration.json"))

        // The MTP measurement is owned by the MTP measure wrapper, not the
        // serial track's measure-job.sh.
        #expect(workflow.contains("MLXFAST_MTP_MEASURE_JOB: /opt/bench-runner/measure-mtp-job.sh"))
        #expect(!workflow.contains("MLXFAST_MEASURE_JOB: /opt/bench-runner/measure-job.sh"))

        // Trusted caches are restore-only here: only the serial pipeline's
        // main-ref dispatches may populate the trusted cache namespaces.
        #expect(workflow.contains("actions/cache/restore@"))
        #expect(!workflow.contains("actions/cache/save@"))

        // benchmark.json is Yukon's authoritative default registration.
        let registration = try Data(contentsOf: URL(fileURLWithPath: "benchmark.json"))
        let track = try #require(try JSONSerialization.jsonObject(with: registration) as? [String: Any])
        #expect(track["trackId"] as? String == "laguna-xs-2.1-mtp-v1")
        #expect(track["name"] as? String == "mlxfast-challenge-dev-mtp")
        let runner = try #require(track["runner"] as? [String: Any])
        #expect(runner["workflow"] as? String == "benchmark.yml")
        #expect(track["scorePath"] as? String == "score.json")
        #expect(
            track["benchmarkCommand"] as? [String]
                == ["bash", "-c", "MLXFAST_SCORE_PATH=score.json ./benchmark-mtp.sh --local-iterate"]
        )
        #expect(
            track["setupCommand"] as? [String]
                == ["bash", "-c", "MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh && ./setup-mtp.sh"]
        )
        let leaderboard = try #require(track["leaderboard"] as? [String: Any])
        #expect(leaderboard["namespace"] as? String == "laguna-xs-2.1-mtp-v1")
        #expect(leaderboard["separateFromSerialTrack"] as? Bool == true)
        #expect(
            registration
                == (try Data(contentsOf: URL(fileURLWithPath: "benchmark.mtp.json"))),
            "benchmark.mtp.json must remain a byte-identical compatibility alias"
        )

        let serialConfig = try String(contentsOfFile: "benchmark.serial.json", encoding: .utf8)
        #expect(serialConfig.contains("\"workflow\": \"serial-benchmark.yml\""))
        #expect(serialConfig.contains("\"scorePath\": \"score.json\""))
        #expect(!serialConfig.contains("mtp"))
    }

    @Test
    func defaultMTPLocalRunnerProducesYukonScorePayload() throws {
        let path = "benchmark-mtp.sh"
        let source = try String(contentsOfFile: path, encoding: .utf8)
        let attributes = try FileManager.default.attributesOfItem(atPath: path)
        let permissions = try #require(attributes[.posixPermissions] as? NSNumber).intValue

        #expect(permissions & 0o111 != 0)
        #expect(source.contains("mtp-probe"))
        #expect(source.contains("mtp-benchmark"))
        #expect(source.contains("--require-trained-assistant"))
        #expect(source.contains(".all_tokens_matched == true"))
        #expect(source.contains("score: $score"))
        #expect(source.contains("official_score: false"))
    }

    // The serial ranked pipeline stays MTP-free: enabling, disabling, or
    // editing the MTP track must never touch the serial score path.
    @Test
    func serialRankedPipelineRemainsMTPFree() throws {
        let serial = try serialWorkflow()
        #expect(!serial.contains("mtp"))
        #expect(!serial.contains("MTP"))
        #expect(!serial.contains("MLXFAST_MTP_"))
        // Neither the retired nor the current MTP track namespace may leak
        // into the serial pipeline ("laguna-xs-2.1" alone stays legal there:
        // the serial track shares the Laguna target model).
        #expect(!serial.contains("gemma4-31b-it"))
        #expect(!serial.contains("laguna-xs-2.1-mtp"))
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

    // Decode-only paired scoring per the adopted 2026-07-14 decision memo:
    // 512-token decode window, ratio-of-means aggregation computed in the
    // trusted shell from the sealed per-side means, floor 1.0 on that
    // aggregate, minimum 3 / target 4 accepted pairs, and the hard bit-exact
    // parity gate across every timed run. The hidden benchmark golden
    // reaches the workspace only inside measure-job's timed sides, and the
    // correctness golden is scrubbed before timing.
    @Test
    func mtpScoringEnforcesBitExactGateMinimumPairsAndFloor() throws {
        let workflow = try mtpWorkflow()

        #expect(workflow.contains("MLXFAST_MTP_DECODE_TOKENS: \"512\""))
        #expect(workflow.contains("MLXFAST_MTP_DECODE_SPEEDUP_FLOOR: \"1.0\""))
        #expect(workflow.contains("MLXFAST_MTP_MIN_ACCEPTED_PAIRS: \"3\""))
        #expect(workflow.contains("MLXFAST_MTP_TARGET_PAIRS: \"4\""))
        #expect(workflow.contains("--target-pairs \"${MLXFAST_MTP_TARGET_PAIRS}\""))

        let score = try stepBody(
            workflow,
            from: "- name: Compute MTP score and enforce floor",
            to: "- name: Check MTP benchmark artifact paths"
        )
        #expect(score.contains(".parity_all_ok == true"))
        #expect(score.contains(".accepted_pair_count >= $min_pairs"))
        #expect(score.contains("is below the ${MLXFAST_MTP_DECODE_SPEEDUP_FLOOR} floor"))
        // Ratio-of-means is computed in the trusted shell from the sealed
        // per-side seconds/token means, never read from a pre-aggregated
        // speedup field.
        #expect(score.contains(
            ".aggregate.baseline_serial_seconds_per_token_mean / .aggregate.candidate_mtp_seconds_per_token_mean"
        ))
        #expect(score.contains("aggregation: \"ratio_of_means\""))
        #expect(score.contains("mtp_decode_speedup_ratio_of_means"))
        #expect(!score.contains("score: $results[0].aggregate.mtp_decode_speedup_mean"))

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
