# Benchmark Window Freeze

This document is the frozen definition of the **timed benchmark window** -- the
exact work the official runner charges to the prefill and decode scores -- and
the protocol for changing it. It exists because the ranked score is paired
against a **pinned reference baseline** provisioned on the official M5 box and
calibrated at real cost. Any change to the charged work makes that baseline
(and the cached local-mode constants) mean a different thing, which forces a
new baseline for every axis that moved.

Treat a re-baseline as expensive and rare. The goal of this freeze is to make
the current calibration the last forced one: decide every window knob here, pin
it with `Tests/MLXFastTests/BenchmarkWindowFreezeTests.swift`, and pin any later
security defense that must add charged work as part of this contract.

## The soundness invariant

Every forward the timed window charges MUST be:

1. **Output-validated against an oracle.** Unvalidated charged work can be
   under-computed for free by editable model code, because nothing forces it to
   produce a correct result. (The removed decode "warmup" forward discarded its
   token, so it was reclaimable -- that is why removing it, not keeping it, was
   the correct fix.)
2. **Never an identical repeat of another charged forward in the same worker
   process.** Two identical charged forwards let editable code memoize one and
   serve the other, collapsing two charged forwards into one with no real
   speedup. Prefill, decode, and correctness each run in their own worker
   process, so no model-owned memo persists across phases.
3. **Begin at a trusted MLX free-buffer boundary.** Worker initialization is
   unscored and executes editable model code, so the first request handler for
   every new forward sequence performs a trusted 6 GiB MLX free-buffer cache reset:
   set `Memory.cacheLimit` to the trusted phase-start value, call
   `Memory.clearCache()`, and fail closed unless `Memory.cacheMemory == 0`. The
   pinned MLX API defines `clearCache()` as synchronously deallocating all
   cached (free) buffers; live model weights and KV state are active memory, so
   exact zero is safe here. The reset pins the phase-start state only -- it is
   not an enforced cap for the rest of the phase: editable code may change
   `Memory.cacheLimit` again inside the charged window, and any allocation that
   follows is charged like all other work. The substantive defense is the
   free-buffer clear, which stops unscored initialization from subsidizing the
   first charged forward. The fail-closed zero check also makes "no MLX
   allocator activity in flight across a request boundary" part of the
   submission contract: background work that repopulates the cache at a
   sequence boundary fails the run.

The current window satisfies all three: one validated seed prefill plus 128
validated single-token decode steps (decode axis), and one validated cold
prefill forward (prefill axis). The trusted parent starts each scored timer
before sending the phase-begin request, so the allocator reset is charged. Any
future window change must preserve these invariants.

## Frozen window definition

Charged work per axis. Changing any of these is a **baseline-affecting** change
(see protocol below).

Prefill axis (`measureWorkerPrefillSecondsPerToken`):

- `benchmarkPrefillPromptTokens = 512` -- prompt length of the single timed forward.
- `benchmarkPrefillWarmupRuns = 0` -- no warmup; the one timed run is cold.
- `benchmarkPrefillTimedRuns = 1` -- exactly one measured, validated forward.
- The `prefill` request applies the trusted phase-start allocator reset
  (cache-limit set plus free-buffer clear) before constructing its model cache
  or computing logits.

Decode axis (`measureWorkerDecode` / worker `decode_begin` + `decode_step`):

- `benchmarkDecodeSeedTokens = 512` -- the seed prefill, charged to the decode
  window (so future-token work cannot hide in an unscored seed phase).
- `benchmarkDecodeSteps = 128` -- validated single-token teacher-forced steps.
- Exactly one whole-prompt seed forward in `decode_begin`; the per-step forwards
  are single-token and input-dependent.
- `decode_begin` applies the trusted phase-start allocator reset (cache-limit
  set plus free-buffer clear) before constructing the decode cache or computing
  seed logits.

Measurement authority (not a constant, but part of the frozen contract):

- The trusted parent measures wall time with its own clock across the whole
  phase. Worker-reported `seconds` are diagnostic only and never feed the score.
- The parent starts the prefill/decode timer before `worker.prefill` /
  `worker.beginDecode`, respectively. The trusted allocator reset therefore
  belongs to the charged window, not unscored worker initialization.
- No allocator reset runs in `correctness_step` or `decode_step`; those handlers
  retain legitimate within-sequence KV and intermediate-buffer reuse, and their
  work remains charged to the enclosing correctness/decode request sequence.

## Frozen ranking contract

Changing these does not require a re-baseline, but it does change how a fixed
baseline maps to a published score, so it is frozen here too and pinned by the
same test:

- `scoreDecodeWeight = 0.75`, `scorePrefillWeight = 0.25`.
- `scoreDecodeSpeedupFloor = 0.95`, `scorePrefillSpeedupFloor = 0.95`.
- `prefillBandUpTolerance = 0.05`, `prefillBandDownTolerance = 0.05`.
- `decodeBandUpTolerance = 0.02`, `decodeBandDownTolerance = 0.05`.

Acceptance bands (see `AcceptanceBand`,
`docs/thermal-variance-investigation.md`): prefill and decode are single noisy
measurements, each gated against the same-session paired baseline `B` measured
on the same silicon (which cancels host-speed differences). After the speedup
floors, each axis's measured value must land within
`[B * (1 - downTolerance), B * (1 + upTolerance)]`: it fails if the value
exceeds `B * (1 + upTolerance)` (a real slowdown / regression) or drops below
`B * (1 - downTolerance)` (an improvement too large to trust in one
submission, or a suspiciously lucky-fast reading).

- **Prefill: +/-5% symmetric.** Prefill is not a real optimization axis here, so
  it is a health gate -- both a regression and a lucky-fast reading past 5% fail.
- **Decode: +2% regression / -5% gain.** Decode is the axis the score rewards, so
  the up (regression) side is tight at +2%; the down (gain) side caps a *single
  submission's* decode improvement at 5%. Larger wins are still welcome -- they
  must be **chunked** across submissions so each step stays inside the band and is
  independently verifiable. The cap is per-submission, not cumulative.

`B`'s robustness and the per-axis tolerances are ranking-contract decisions, so
they are operator-owned and pinned here.

## Cached local-mode constants

The `officialBaseline*` constants in `Sources/MLXFastCore/Constants.swift` are
a **cached calibration retained for local-mode estimates and the gates pass's
skip-timed placeholder timing only** -- the ranked score denominator is the
live paired baseline measured on the M5 box (next section), never these
numbers. They were last calibrated 2026-07-12 as the mean of the on-box M5
paired-baseline timings published by 12 consecutive successful ranked runs
(29179374395 through 29197772284; decode CV 0.08%, prefill CV 0.17%),
replacing the retired VM-era 2026-07-07 values (decode 0.1336139485703125 /
prefill 0.010605031949609375) that inflated local estimates ~3.7x. They are
kept so `--local-iterate` / `--local-submit` can print a directional speedup
without a second local build:

- `officialBaselineDecodeSecondsPerToken = 0.04405625764973958`
- `officialBaselinePrefillSecondsPerToken = 0.0016216554767252605`

If either number here disagrees with `Sources/MLXFastCore/Constants.swift`, the
freeze test fails on purpose -- the doc and the code must move together.

### Paired baseline on the single M5 box

The ranked runner is one self-hosted Apple M5 Max (label `m5-bench`). The
paired baseline is a **pinned reference tree provisioned on that box**
(`/opt/bench-runner/baseline/current`), and the timed measurement is owned by
the on-box `measure-job` (trusted, runner-provisioned, fixed thermal
contract). Per ranked run, after all correctness/gates work, a hidden-material
scrub, and a quiescence wait, measure-job runs the pinned **baseline tree
first, then the candidate workspace**, each as a full `./benchmark.sh
--official` in fresh worker processes, each starting only once the GPU is
below the fixed 40C gate (up to a 900s cooldown), each under 2 Hz telemetry.
A measurement is rejected -- with one gated retry -- on GPU throttling under
load, missing telemetry, or token mismatches. Because both sides run back to
back on the same silicon behind the same gate, the paired ratio cancels
common-mode host drift exactly like the old separate-VM pairing did
(validated on this box at paired score 1.0507 with decode CV 0.13%).

- The baseline sample is additionally sanity-checked against the box's
  recorded calibration (`/opt/bench-runner/state/baseline-calibration.json`),
  so a broken baseline build or a sick host fails the run instead of
  repricing every speedup.
- The baseline's own floor verdict is ignored; token mismatches or harness
  errors in the baseline run fail the whole run.
- The speedup floors and the `decode^0.75 * prefill^0.25` score are applied to
  the paired ratio in the trusted shell
  (`.github/scripts/overlay-paired-timing.sh`), with the same constants the
  harness pins.
- Changing the pinned baseline tree or its calibration is a ranking-contract
  change: it redefines what "1.0x" means. Update the tree, its calibration,
  this doc, and the pinned guard tests together (operator procedure: RUNBOOK).

The harness's own paired-override plumbing
(`MLXFAST_PAIRED_BASELINE_{PREFILL,DECODE}_SECONDS_PER_TOKEN`) is still
shipped and armed: a paired override outranks golden-carried baselines, which
outrank the constants; the pair is fail-closed (both together, finite,
positive) and both variables are stripped from the sandboxed worker
environment.

### Per-prompt baselines in the golden oracle

The constants are the **fallback**, not the only source. A golden's
`benchmark` oracle may carry its own calibration:

```json
"benchmark": {
  "...": "...",
  "baseline_prefill_seconds_per_token": 0.0101,
  "baseline_decode_seconds_per_token": 0.1317
}
```

Rules, enforced at golden load and by the freeze tests:

- Both fields must be present together or absent together, finite and positive.
  A half-calibrated oracle would silently mix two calibration regimes.
- When present, scored speedups, floors, the published `baseline_*` metrics,
  and the gates pass's placeholder timing all resolve from the golden.
  When absent, everything resolves from the constants above (the pre-pool
  behavior; all public fixtures carry none).
- Carrying baselines in the golden is what makes prompt-pool rotation rankable:
  each pool prompt ships its own officially measured calibration, so rotating
  prompts of different intrinsic difficulty keeps speedups comparable. It does
  NOT change the charged window -- adding a pool prompt is baseline work for
  that prompt only, never a re-baseline of the window itself.
- On the ranked path the live paired baseline outranks both the golden-carried
  values and the constants, so pool prompts no longer need pre-calibrated
  baselines there -- the baseline is measured live against whichever prompt is
  active.

## Re-baseline protocol (how to change the window)

1. Make the window change and update the constants above in
   `Sources/MLXFastCore/Constants.swift`.
2. Rebuild and re-measure the pinned baseline tree on the official M5 box with
   the reference model, all gates green, and refresh its recorded calibration
   (operator procedure: RUNBOOK).
3. Update `officialBaseline*SecondsPerToken` and the values quoted in this doc,
   `README.md`, and `TASK.md`.
4. Update the pinned literals in `BenchmarkWindowFreezeTests.swift` in the same
   change. The test is designed to fail until you do, so a window edit cannot
   land while silently reusing a stale baseline.

## Constant-runtime holdout: prompt pool + rotation

The ranked job runs one timed prompt to stay inside the runner time budget, so a
Kaggle-style public/private split run side by side would double the timed
runtime. The constant-runtime equivalent is a **pre-calibrated pool of
interchangeable prompts, rotated one at a time**:

- Every pool prompt MUST have identical shape: 512 prefill tokens, 512 decode
  seed tokens, 128 decode steps. Identical shape means each rotation slot
  consumes the same runtime and its per-token baseline is directly comparable to
  the others.
- Calibrate the entire pool in one baseline session (all prompts measured under
  the same toolchain, runner, and thermal state). Never add a pool member later
  without a batched calibration -- a late addition is another baseline.
- Each pool prompt's golden carries its own measured calibration via the
  per-prompt baseline fields described above, so the harness scores every
  rotation slot against that prompt's own baseline automatically.
- At ranking time, select one pool prompt per run and rotate which one is active.
  Contestants get feedback on whichever prompt they were scored against but
  cannot select-overfit a fixed trajectory across repeated submissions.

Rotating within a same-shape, pre-calibrated pool is runtime-neutral and does
not re-open the freeze: the window definition is unchanged, only the prompt
content rotates.

## Defenses that live outside this repo

These fight submission-selection bench-maxing (trying many variants and keeping
whatever scored best on one hidden prompt). They are leaderboard/orchestrator
policy, not harness code, and they do not affect the window or the baseline:

- Cap scored submissions per contestant per period.
- Keep the latest submission's score, not the best-ever (removes
  submit-until-lucky variance harvesting on the single cold prefill run and the
  single decode trajectory).
- Private holdout / rotation selection for final ranking (see pool above).
- Audit frontier-promoted submissions before they count; treat the LLM static
  review as a backup gate, never the sole guarantee for a structural invariant.

## Already implemented (context)

- Feedback coarsening: diagnostic real-valued score fields are published rounded
  to `publicDiagnosticSignificantFigures` significant figures to shrink the
  timing/memory covert channel. Ranking fields stay precise on purpose.
- The submission static review is taught to catch measurement-structure
  exploitation (input-keyed logits/KV memoization that can only hit when the
  harness repeats an identical forward).
