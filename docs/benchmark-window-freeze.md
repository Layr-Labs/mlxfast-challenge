# Benchmark Window Freeze

This document is the frozen definition of the **timed benchmark window** -- the
exact work the official runner charges to the prefill and decode scores -- and
the protocol for changing it. It exists because the official baseline
(`officialBaselineDecodeSecondsPerToken` /
`officialBaselinePrefillSecondsPerToken`) is measured on the Blacksmith runner
at real cost. Any change to the charged work makes the recorded baseline mean a
different thing, which forces a new baseline run for every axis that moved.

Treat a re-baseline as expensive and rare. The goal of this freeze is to make
the current calibration the last forced one: decide every window knob here, pin
it with `Tests/MLXFastTests/BenchmarkWindowFreezeTests.swift`, and afterwards add
only defenses that do not touch the charged work.

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
   process, so no memo persists across phases.

The current window satisfies both: one validated seed prefill plus 128 validated
single-token decode steps (decode axis), and one validated cold prefill forward
(prefill axis). Any future window change must preserve this invariant.

## Frozen window definition

Charged work per axis. Changing any of these is a **baseline-affecting** change
(see protocol below).

Prefill axis (`measureWorkerPrefillSecondsPerToken`):

- `benchmarkPrefillPromptTokens = 512` -- prompt length of the single timed forward.
- `benchmarkPrefillWarmupRuns = 0` -- no warmup; the one timed run is cold.
- `benchmarkPrefillTimedRuns = 1` -- exactly one measured, validated forward.

Decode axis (`measureWorkerDecode` / worker `decode_begin` + `decode_step`):

- `benchmarkDecodeSeedTokens = 512` -- the seed prefill, charged to the decode
  window (so future-token work cannot hide in an unscored seed phase).
- `benchmarkDecodeSteps = 128` -- validated single-token teacher-forced steps.
- Exactly one whole-prompt seed forward in `decode_begin`; the per-step forwards
  are single-token and input-dependent.

Measurement authority (not a constant, but part of the frozen contract):

- The trusted parent measures wall time with its own clock across the whole
  phase. Worker-reported `seconds` are diagnostic only and never feed the score.

## Frozen ranking contract

Changing these does not require a re-baseline, but it does change how a fixed
baseline maps to a published score, so it is frozen here too and pinned by the
same test:

- `scoreDecodeWeight = 0.75`, `scorePrefillWeight = 0.25`.
- `scoreDecodeSpeedupFloor = 0.95`, `scorePrefillSpeedupFloor = 0.95`.

## Current calibrated baseline

Measured on the baseline reference under the current (single-seed) harness:

- `officialBaselineDecodeSecondsPerToken = 3.6366560638046876`
  (recalibrated after the `decode_begin` warmup-forward removal).
- `officialBaselinePrefillSecondsPerToken = 0.17330563175390626`
  (prefill path unchanged; within single-shot noise of the prior value).

If either number here disagrees with `Sources/MLXFastCore/Constants.swift`, the
freeze test fails on purpose -- the doc and the code must move together.

## Re-baseline protocol (how to change the window)

1. Make the window change and update the constants above in
   `Sources/MLXFastCore/Constants.swift`.
2. Re-measure the affected axis (or both) on the official Blacksmith runner with
   the baseline reference model, all gates green.
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
