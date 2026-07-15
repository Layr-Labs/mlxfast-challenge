# MTP Track Go-Live Runbook (Operator Only)

Operator procedure for taking the Gemma 4 31B-IT MTP track
(`gemma4-31b-it-mtp-v1`) from its current merged-but-inert state to a live
ranked track. Nothing in this document is participant-executable: every step
is operator-gated, and the track's scoring stays disabled until the final
enablement commit described in step 4.

Box naming used throughout (no hostnames in this public doc):

- **Box 1 — dev box.** Drained from ranked serving. Holds the only current
  copies of the IT target + assistant caches (`setup-mtp.sh` layout under the
  dev user's `~/.cache/mlxfast/gemma4-31b-it-mtp-v1/`). The first-block
  stall was root-caused and fixed here; the post-fix validation matrix runs
  here too.
- **Box 2 — ranked serving box.** Serves live ranked serial-decode jobs
  (`benchmark.yml`). Protected surface is integrity-manifest-signed; any
  intentional change requires a manifest re-sign or the janitor quarantines
  the box. Do not touch it except inside step C's provisioning window.

Current state (updated by the enablement change this document rides in):

- The MTP track code, contract, manifests, and provisioner are merged behind
  explicit `mtp-probe` / `mtp-benchmark` commands; serial `benchmark` cannot
  reach them.
- The hidden IT-target goldens are FROZEN (captured and self-validated on
  drained M5 silicon per step B; pins carried in `mtp-benchmark.yml`,
  pending box-2 replay confirmation).
- The enablement change flips `fixtures/gemma_4_31b_it_mtp_track.json` to
  `official_scoring_enabled=true` /
  `reference_baseline.publication_allowed=true`, fills the golden pins, and
  adopts the 2026-07-14 scoring decisions; until it MERGES, `main` stays
  inert (flags false, pins empty). The workflow's enablement gate remains
  standing either way: it fails closed on a reverted contract, an emptied
  pin, or a dispatch without `confirm_track_enabled=true`.
- Box 2 still has no MTP caches, baseline tree, calibration, or
  `measure-mtp-job.sh`; provisioning (step C) is a merge PRECONDITION of the
  enablement change.
- The serial ranked pipeline (`benchmark.yml`) is untouched by all of this
  and must remain untouched by every step below.

## 0. Go-live sequence (the one-page version)

Strictly ordered; each step is a precondition of the next:

1. **Stall fix validated at the merged tip.** The "sporadic" single-block
   stall was root-caused as a deterministic block-0 Metal first-touch after
   the trusted allocator clear and is FIXED on `main`
   (`warmWorkingSetAfterAllocatorReset`, see docs/experimental-mtp-track.md
   "First-block stall: root cause and fix"; dev-box max_block dropped from
   ~1.6 s to <= 79 ms with token parity preserved). Before calibration, run
   one fresh 12-pair thermal-gated matrix at the merged go-live ref showing
   per-category paired CV <= 5% and max_block ~p50 (no block-0 outlier).
   *Hard dependency: runbook step B's floor calibration is meaningless on
   stall-contaminated data — calibrate only at a ref that contains the fix,
   and expect the residual ~1.5 s one-time seed-prefill first-touch
   variation, which the multi-pair alternating protocol averages out.*
2. **Step B** — capture + freeze the hidden IT-target goldens and the serial
   oracle on M5 silicon; upload to R2; record pins. (Capture may run in
   parallel with step 1's validation matrix — goldens are token sequences,
   not timings — but calibration cannot.)
3. **Step C** — provision box 2: target + assistant caches, pinned MTP
   baseline tree, `measure-mtp-job.sh`, golden replay verification, then the
   floor-calibration sessions (B's methodology, clean post-stall-fix data),
   then the manifest re-sign.
4. **Step A/enablement** — one trusted commit on `main` flips the contract
   fixture, fills the golden pins in `mtp-benchmark.yml`, and updates the
   tests that pin the disabled state. Register the
   `gemma4-31b-it-mtp-v1` leaderboard namespace with the orchestrator.
5. **Validate** — correctness-only dispatch, then a full timed main-ref
   dispatch, then a reference self-measurement expecting ~1.2-1.3x with the
   1.0 floor cleared, then (only then) open the track to submissions.

Rollback at any point: revert the enablement commit (step 4) and the
workflow is inert again; nothing on box 2 needs to be removed to stop
scoring.

---

## B. Freeze hidden IT-target goldens + calibrate the decode floor (PLAN — do not execute yet)

### B.0 Preconditions

- [ ] Box 1 free (no MTP experiment or validation matrix running).
- [ ] Repo synced to the tip that will become the go-live ref — it must
      contain the merged #616 squash including the block-0 stall fix
      (`warmWorkingSetAfterAllocatorReset`); `swift build -c release` clean.
- [ ] `./setup-mtp.sh --verify-only` passes on box 1 (both caches byte-exact
      against `fixtures/mtp_gemma_4_31b_it_4bit.sha256` and
      `fixtures/mtp_gemma_4_31b_it_assistant_qat4bit.sha256`).
- [ ] For the calibration part only: sequence step 1's post-fix validation
      matrix is complete (CV <= 5%, no block-0 outlier). Golden capture
      (B.1-B.3) may run before that; **floor calibration (B.6) must not.**

### B.1 What gets frozen (inventory)

| Artifact | Contents | Where it lives | In repo? |
|---|---|---|---|
| Hidden MTP **correctness** golden | private prompt, 512-token seed, 1 seed token + 512 greedy IT-target decode tokens | R2 object `correctness_prompts/mtp_correctness_golden_gemma4_31b_it-v1.json` | Never (hidden; pins only) |
| Hidden MTP **benchmark** golden | different private prompt, 512-token seed, 1 + 512 greedy decode tokens (captures headroom above the 128-token ranked denominator) | R2 object `correctness_prompts/mtp_benchmark_golden_gemma4_31b_it-v1.json` | Never (hidden; pins only) |
| Their SHA256 + byte pins | 4 values | `mtp-benchmark.yml` env (`MLXFAST_MTP_{CORRECTNESS,BENCH}_GOLDEN_{SHA256,BYTES}`) | Yes, in the enablement commit |
| Public IT fixture (optional, participant UX) | public prompt + 129-token golden for local `mtp-probe`/`mtp-benchmark` iteration | `correctness_prompts/` | Yes (optional, non-blocking) |
| Serial oracle | not a separate artifact: the goldens ARE the serial oracle (greedy K=1 continuations of the pinned IT target); the trusted parent validates every returned token against them | — | — |

Prompt selection: two distinct private prompts, not derived from the public
fixtures, not GPQA (no GPQA gate in MTP v1), each tokenizing to >= 512
tokens with the IT tokenizer, mixed genre (one prose-like, one
code/structured) so acceptance-rate behavior is exercised differently in the
correctness pass vs the timed pass. Store the prompt texts in the operator
private store alongside the R2 upload, like the serial hidden prompts.

### B.2 Capture (box 1, or box 2 inside the C window)

```bash
# repo at the go-live ref
./setup.sh                       # toolchain check; base cache untouched
./setup-mtp.sh --verify-only
eval "$(./setup-mtp.sh --print-paths)"
swift build -c release

# transform the IT target (NOT the serial base checkpoint)
.build/release/mlxfast-swift transform \
  --reference "${MLXFAST_MTP_TARGET_DIR}" \
  --output mtp-weights

# capture 1 seed + 512 decode tokens per prompt (steps = decode + 1)
.build/release/mlxfast-swift generate-golden \
  --prompt-file /private/operator/mtp_correctness_prompt.txt \
  --weights mtp-weights \
  --tokenizer "${MLXFAST_MTP_TARGET_DIR}" \
  --output /private/operator/mtp_correctness_golden_gemma4_31b_it-v1.json \
  --name gemma4-31b-it-mtp-correctness-v1 \
  --steps 513

.build/release/mlxfast-swift generate-golden \
  --prompt-file /private/operator/mtp_benchmark_prompt.txt \
  --weights mtp-weights \
  --tokenizer "${MLXFAST_MTP_TARGET_DIR}" \
  --output /private/operator/mtp_benchmark_golden_gemma4_31b_it-v1.json \
  --name gemma4-31b-it-mtp-benchmark-v1 \
  --steps 513
```

### B.3 Self-validation before freezing (all must pass)

```bash
# 1) serial K=1 control replays each golden token-for-token
.build/release/mlxfast-swift mtp-probe \
  --weights mtp-weights --golden <each golden> \
  --block-size 4 --tokens 512          # expect all_tokens_matched=true

# 2) trained exact-pair path replays each golden token-for-token
.build/release/mlxfast-swift mtp-benchmark \
  --target-source "${MLXFAST_MTP_TARGET_DIR}" \
  --weights mtp-weights \
  --assistant "${MLXFAST_MTP_ASSISTANT_DIR}" \
  --contract fixtures/gemma_4_31b_it_mtp_track.json \
  --golden <each golden> \
  --block-size 4 --tokens 512 \
  --target-verification exact-pair --require-trained-assistant

# 3) the four model-backed parity gates (artifact, pair bit-for-bit,
#    forced acceptance seams, deep growth/wrap) — commands as documented in
#    docs/experimental-mtp-track.md "Local/operator workflow"
```

### B.4 Freeze + upload

```bash
shasum -a 256 mtp_*_golden_gemma4_31b_it-v1.json
wc -c        mtp_*_golden_gemma4_31b_it-v1.json   # record all four pins

# upload with the benchmark-private-prompts R2 credentials (same bucket and
# key discipline as the serial hidden golden; aws CLI v2 is on both boxes)
aws s3 cp mtp_correctness_golden_gemma4_31b_it-v1.json \
  "s3://<bucket>/correctness_prompts/mtp_correctness_golden_gemma4_31b_it-v1.json" \
  --endpoint-url "${R2_BUCKET_ENDPOINT}"
aws s3 cp mtp_benchmark_golden_gemma4_31b_it-v1.json \
  "s3://<bucket>/correctness_prompts/mtp_benchmark_golden_gemma4_31b_it-v1.json" \
  --endpoint-url "${R2_BUCKET_ENDPOINT}"
```

Keep offline copies in the operator private store; delete every on-box copy
outside the runner-private paths. The four pins go into
`mtp-benchmark.yml` in the enablement commit (step 4), not before.

### B.5 Cross-box replay verification (mandatory if captured on box 1)

Box 1 and box 2 run different macOS builds; greedy near-tie argmaxes can in
principle diverge across stacks even on same-generation silicon. During the
C provisioning window, replay B.3's checks 1-2 on box 2 against the frozen
goldens. If any token diverges, the ranked box is the source of truth:
regenerate on box 2, re-pin, re-upload, and re-run B.3 there.

### B.6 Floor calibration (box 2, only at a ref containing the stall fix — hard dependency)

**Calibrate only at a ref that contains the block-0 stall fix
(`warmWorkingSetAfterAllocatorReset`, in the merged #616 squash) and only
after sequence step 1's post-fix validation matrix is complete.** The
pre-fix stall inflated MTP-side seconds/token (paired CV observed up to
14%); bands and headroom derived from contaminated sessions would be
simultaneously too loose (accepting drifted baselines) and too pessimistic
(understating reference headroom). Post-fix, expect the residual bounded
one-time seed-prefill first-touch (~1.5 s run-to-run variation, a Metal
residency effect); the multi-pair alternating protocol averages it out,
which is one more reason single-pair numbers are never publishable.

The 1.0 decode floor itself is **fixed by the track contract**, not
calibrated: `score = mtp_decode_speedup >= 1.0` or the run does not rank.
What calibration establishes is (a) the pinned serial denominator and its
sanity band, and (b) proof that the unmodified reference clears the floor
with margin.

Method (mirrors the serial track's baseline-calibration discipline):

1. Run **5 fresh thermal-gated paired sessions** (minimum 3) on box 2 via
   `measure-mtp-job.sh` in operator mode, each session >= 3 accepted pairs,
   alternating serial/MTP order within a session, fresh worker processes per
   phase, 40C gate before every timed phase, 2 Hz telemetry:
   - serial side: `mtp-probe --tokens 128` from the pinned MTP baseline tree;
   - MTP side: `mtp-benchmark --tokens 128 --target-verification exact-pair`
     from the same pinned tree (reference measuring reference);
   - golden: the frozen hidden benchmark golden.
2. Acceptance per session: every token matched; no throttled/telemetry-void
   samples; post-fix stall criterion `max_block_request_seconds <= 4 x
   p50_block_request_seconds`; session paired CV <= 5%.
3. Author `/opt/bench-runner/state/mtp-baseline-calibration.json`
   (runner:runner 0644):
   - `serial_decode_seconds_per_token_mean` (the denominator anchor) with a
     **0.95-1.05 sanity band** (serial K=1 decode is the stable metric, CV
     ~0.1% class on this hardware);
   - `reference_mtp_decode_seconds_per_token_mean` and the observed
     reference speedup distribution (mean/median/min across sessions) with
     a provisional band (widen only with recorded justification);
   - pinned baseline tree ref + binary sha256; session ids; date; stall-fix
     commit it was measured on.
4. Go/no-go: reference mean speedup must be >= 1.15x and every session min
   pair >= 1.0x. If not, the floor's headroom is too thin to open the track
   — escalate rather than lowering the floor.
5. Re-sign the box manifest after writing the calibration (it is a
   content-pinned protected file), confirm `janitor --audit-only` clean, and
   mirror the calibration + session artifacts to the operator DR repo.

---

## C. Provision the IT target + assistant on the serving box (PLAN — OPERATOR-GATED, execute only at go-live)

Box 2 serves live ranked serial jobs. Everything below is designed to be
invisible to that service: no serial-track file is touched, heavy work is
scheduled around ranked jobs, and every protected-surface change ends with a
manifest re-sign before the next job's janitor audit.

### C.0 Non-disruption rules (read first)

- **Never touch:** `/opt/bench-runner/baseline/current` (serial pinned
  baseline), `/opt/bench-runner/state/baseline-calibration.json`,
  `/opt/bench-runner/measure-job.sh`, `/opt/bench/*`, the supervisor
  config/plist, sudoers, PF config, or anything `benchmark.yml` consumes.
- **Schedule around ranked jobs.** Ranked jobs run one at a time via the
  ephemeral supervisor. Before each heavy sub-step (rsync of 18.4 GB,
  full-tree hashing, the baseline build, calibration sessions), check the
  supervisor/runner logs for an in-flight job and wait for it to finish.
  Competing CPU/disk/GPU load during a candidate's timed phase can burn its
  one gated retry (`measure-job` preflight rejects load >= 2.0 / GPU util >=
  10%). Organizationally hold new ranked dispatches during the window
  instead of stopping the supervisor; the calibration sessions (B.6) are the
  only sub-step that NEEDS an idle box, because they use the GPU themselves.
- **Manifest hygiene.** Any change under a manifest-audited path quarantines
  the box at the next janitor run unless re-signed. Batch protected-surface
  changes, then immediately: `sudo /opt/bench/gen-manifest.sh` followed by
  `sudo /opt/bench/janitor.sh --audit-only` (must report clean) — before the
  next ranked job can start.
- **Ownership posture mirrors the serial reference cache:** runner-owned,
  world/bench-readable, never bench-writable. The runtime additionally
  revalidates artifact hashes before and after model load, but operator
  read-only ownership is a stated requirement of the track's TOCTOU stance.

### C.1 Preflight

```bash
test ! -e /opt/bench/quarantine.flag
sudo /opt/bench/janitor.sh --audit-only        # clean before starting
df -h /opt /Users/Shared                       # need ~45 GB free:
#   18.4 GB target + 0.27 GB assistant + ~17.2 GB baseline mtp-weights + slack
```

### C.2 Create the runner-owned MTP cache

```bash
sudo install -d -o runner -g staff -m 0755 \
  /opt/bench-runner/cache/mtp/gemma4-31b-it-mtp-v1/target \
  /opt/bench-runner/cache/mtp/gemma4-31b-it-mtp-v1/assistant
```

These are the exact paths `mtp-benchmark.yml` pins as
`MLXFAST_MTP_TARGET_DIR` / `MLXFAST_MTP_ASSISTANT_DIR`.

### C.3 Stage the artifacts (prefer box-to-box copy; verify identically)

Option 1 (preferred — no 18 GB internet download, no HF dependency): rsync
from box 1's verified dev cache over the tailnet into a staging dir owned by
the staging user, then move into place as runner.

Option 2: fresh download on box 2 as the `runner` user:

```bash
sudo -u runner MLXFAST_MTP_CACHE_ROOT=/opt/bench-runner/cache/mtp/gemma4-31b-it-mtp-v1 \
  ./setup-mtp.sh
```

Either way, verification is what makes them equivalent — every byte against
the checked-in manifests, strict flat inventory, no symlinks/hardlinks:

```bash
sudo -u runner \
  MLXFAST_MTP_TARGET_DIR=/opt/bench-runner/cache/mtp/gemma4-31b-it-mtp-v1/target \
  MLXFAST_MTP_ASSISTANT_DIR=/opt/bench-runner/cache/mtp/gemma4-31b-it-mtp-v1/assistant \
  ./setup-mtp.sh --verify-only
sudo chown -R runner /opt/bench-runner/cache/mtp
sudo find /opt/bench-runner/cache/mtp -type d -exec chmod 0755 {} +
sudo find /opt/bench-runner/cache/mtp -type f -exec chmod 0644 {} +
```

### C.4 Build the pinned MTP baseline tree (the serial-denominator side)

Same pattern as the serial pinned baseline: a frozen, prebuilt,
runner-owned, bench-read-only challenge-repo checkout; jobs measure a
throwaway APFS clone, never the pinned tree.

```bash
# as runner, at the go-live ref <SHA>
git clone <challenge-repo> /Users/Shared/bench-jobs/mtp-baseline/<SHA>
cd /Users/Shared/bench-jobs/mtp-baseline/<SHA>
git checkout <SHA>
swift build -c release && tools/build-mlx-metallib.sh
.build/release/mlxfast-swift transform \
  --reference /opt/bench-runner/cache/mtp/gemma4-31b-it-mtp-v1/target \
  --output mtp-weights
sudo ln -sfn /Users/Shared/bench-jobs/mtp-baseline/<SHA> /opt/bench-runner/mtp-baseline/current
# runner-owned, no bench ACLs (read-only to bench), like the serial baseline
```

### C.5 Install `measure-mtp-job.sh` (box-owned timing wrapper)

Author from `measure-job.sh` as the base (same fixed, readonly thermal
contract) and install to `/opt/bench-runner/measure-mtp-job.sh`
(root:wheel 0755). Required behavior, which `mtp-benchmark.yml` consumes:

- args: `--candidate WS --baseline WS --golden PATH --contract PATH
  --tokens N --block-size K --min-pairs N --target-pairs N --tag T
  --out DIR` (the adopted ranked configuration is `--tokens 512
  --min-pairs 3 --target-pairs 4`);
- per accepted pair, alternating order across pairs: baseline side runs
  `mtp-probe --tokens N` from an APFS clone of the pinned baseline tree;
  candidate side runs `mtp-benchmark --target-verification exact-pair
  --require-trained-assistant --tokens N` from the candidate workspace —
  both through the bench-exec bridge, each phase behind per-phase
  quiescence + reap, the 40C gate (900 s ceiling), 2 Hz macmon telemetry
  with the throttle/steady-sample rules, one gated retry; it attempts
  `--target-pairs` pairs and succeeds with at least `--min-pairs` accepted;
- exports `BENCH_GOLDEN_PATH=<installed golden path>` per side so the
  rendered worker Seatbelt profile denies the hidden benchmark golden to
  the worker (the parent legitimately reads it as the oracle); installs the
  golden 0444 per side and scrubs it after;
- requires `all_tokens_matched=true` from every run (a single divergent
  token invalidates the pair and the job);
- stall guardrail (adopted): reject a run whose
  `max_block_request_seconds > 4 x p50_block_request_seconds` as
  measurement-invalid, with ONE gated retry — the same rejection-and-retry
  class as throttle rejection;
- checks the serial-side seconds/token against the calibration band in
  `/opt/bench-runner/state/mtp-baseline-calibration.json`;
- seals `results.json` (the workflow computes the published ratio-of-means
  score in the trusted shell from the two aggregate per-side means; the
  per-pair `speedup` values and their median/min are diagnostics):

```json
{
  "track_id": "gemma4-31b-it-mtp-v1",
  "mode": "mtp-paired",
  "accepted_pair_count": 4,
  "parity_all_ok": true,
  "pairs": [ { "serial_seconds_per_token": 0.0, "mtp_seconds_per_token": 0.0, "speedup": 0.0, "order": "serial-first" } ],
  "aggregate": {
    "baseline_serial_seconds_per_token_mean": 0.0,
    "candidate_mtp_seconds_per_token_mean": 0.0,
    "mtp_decode_speedup_median": 0.0,
    "mtp_decode_speedup_min": 0.0
  },
  "telemetry": { "max_gpu_temp": 0.0, "min_steady_freq_mhz": 0 }
}
```

Mirror the script to the operator DR repo before installing.

### C.6 Golden replay + calibration on box 2

Run B.5 (replay verification) and then B.6 (floor-calibration sessions —
only at a ref containing the block-0 stall fix, after the post-fix
validation matrix). These are the only GPU-heavy sub-steps; the box must be
idle for them.

### C.7 Re-sign the manifest and prove serving is unaffected

```bash
sudo /opt/bench/gen-manifest.sh
sudo /opt/bench/janitor.sh --audit-only     # must report clean, twice
```

If `manifest-lib.sh`'s protected-path scope must be widened to cover
`measure-mtp-job.sh`, the MTP calibration file, and the MTP baseline
binary/weights digest, that edit is itself a protected-surface change: apply
it and re-sign in the same batch, and mirror it to the DR repo.

Then dispatch one ordinary **serial** ranked run (baseline namespace or
main) and confirm: janitor clean, serial score in its calibration band,
no timing drift — proof the provisioning did not perturb the serving path.

---

## Step 4 (enablement — the operator flip) and validation

The enablement change is ASSEMBLED AS A DRAFT PR (one trusted commit,
reviewed like any ranking-contract change) and merges only after the box-2
preconditions in C are complete. It contains:

1. `fixtures/gemma_4_31b_it_mtp_track.json`:
   `official_scoring_enabled: true`; `reference_baseline` -> `status:
   "established"`, `publication_allowed: true`; the `proposed_scoring`
   block updated to the ADOPTED 2026-07-14 decisions (512-token ranked
   decode window, ratio-of-means aggregation, floor on the aggregate,
   minimum 3 / target 4 pairs, 4x-p50 stall guardrail with one gated
   retry). The harness contract validator
   (`validateExperimentalMTPContract`) pins the enabled identity in the
   same commit. The calibrated serial-denominator stats live in the box's
   `mtp-baseline-calibration.json` (C.5/B.6), not the fixture.
2. `.github/workflows/mtp-benchmark.yml`: the four golden pins filled from
   the freeze record (`MLXFAST_MTP_CORRECTNESS_GOLDEN_SHA256/BYTES`,
   `MLXFAST_MTP_BENCH_GOLDEN_SHA256/BYTES` — pending box-2 replay
   confirmation; re-pin if box 2 diverges), `MLXFAST_MTP_DECODE_TOKENS=512`,
   `MLXFAST_MTP_TARGET_PAIRS=4`, and the ratio-of-means score computation.
3. The tests that deliberately pinned the disabled state updated to pin the
   enabled state instead
   (`ExperimentalMTPTests.trainedMTPContractPinsMatchedITPairAndDependency`,
   `MTPWorkflowIsolationTests`); `swift test` green;
   `serialRankedPipelineRemainsMTPFree` unchanged and passing.
4. The leaderboard registration record `benchmark.mtp.json` (track id
   `gemma4-31b-it-mtp-v1`, workflow `mtp-benchmark.yml`, score artifact
   `mtp-benchmark-results-<run_id>/mtp-score.json`, score field `score`,
   verdict `passed`, floor semantics "failed below 1.0"). The operator
   registers this namespace with the orchestrator/Yukon backend at merge
   time; the serial `benchmark.json` and its leaderboard ingestion are
   untouched.

Validation ladder after merge (each step must pass before the next):

1. Dispatch `mtp-benchmark.yml` on `main`, `confirm_track_enabled=true`,
   `run_benchmark=false` — correctness/parity gate only.
2. Same on `main` with `run_benchmark=true` — full timed run; expect
   reference-vs-reference score ~1.2-1.3x, floor cleared, artifacts sealed,
   janitor clean.
3. Dispatch a `baseline/*` namespace run for the leaderboard ingestion
   smoke test.
4. Only then announce and accept `submissions/*` dispatches on the track.

## Operator decisions — status

Decided by the 2026-07-14 scoring-decision memo (adopted in the enablement
change):

1. **Ranked decode denominator: 512 tokens** (`MLXFAST_MTP_DECODE_TOKENS`;
   the goldens carry 512 decode tokens, the contract's `decode_tokens: 128`
   remains the CLI compatibility default with `maximum_decode_tokens: 512`).
2. **Aggregation: ratio-of-means** — the published score is
   `mean(serial s/tok) / mean(mtp s/tok)` over the accepted pairs, computed
   in the trusted shell from the sealed per-side means; per-pair
   median/min stay as diagnostics.
3. **Stall guardrail: `max_block > 4 x p50` rejects the run** as
   measurement-invalid with one gated retry (owned by `measure-mtp-job.sh`).
4. **Pairs: minimum 3 accepted, target 4**, alternating order; floor 1.0 on
   the aggregate.
5. **License review: complete** — Gemma 4 models are Apache-2.0 with the
   Gemma terms; the repo carries `LICENSE`, `THIRD_PARTY_NOTICES.md`, and a
   README attribution note (license-hygiene PR).

Still open (non-blocking for the enablement draft):

6. **Track memory contract:** observed peak 47.5 GiB RSS; publish a
   participant-facing local minimum (64 GiB practical) and decide whether
   the box enforces a cap.
7. **Second-box MTP serving:** if box 1 later serves the MTP track too, it
   needs its OWN calibration file and baseline tree (never copy
   calibrations across boxes — same rule as the serial track).
8. **Public IT fixture:** whether to check in a public IT-target golden for
   participant local iteration at go-live or later.
