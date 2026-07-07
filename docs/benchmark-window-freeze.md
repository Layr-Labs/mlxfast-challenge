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
- `prefillBandUpTolerance = 0.02`, `prefillBandDownTolerance = 0.05`.

Prefill acceptance band (see `PrefillBand`,
`docs/thermal-variance-investigation.md`): prefill is a single noisy cold forward,
measured once per run. After the speedup floors, the run's measured prefill is
gated against the resolved baseline reference `B` (paired baseline, else
golden-carried, else `officialBaselinePrefillSecondsPerToken`): the run fails if
prefill exceeds `B * (1 + prefillBandUpTolerance)` (a real slowdown) or drops
below `B * (1 - prefillBandDownTolerance)` (a suspiciously lucky-fast reading).
This treats prefill as a tight health gate around the reference rather than an
optimization axis, and it caps a legitimate prefill improvement at ~5%; it is a
ranking-contract change, so `B`'s robustness (ideally a drop-outlier-averaged
calibration on the ranked runner) and the tolerances are an operator decision.

## Current calibrated baseline

Measured on the Gemma 4 31B 4-bit baseline reference (main commit
`eff7e7f2c85a5a6cef11110442ba4624a6ab3986`, the migration merge and the
timing machine's pinned paired-baseline ref) under the current (single-seed)
harness, on the ranked Blacksmith runner class (`blacksmith-12vcpu-macos-26`)
via `gemma-baseline-timing-probe` run 28809531890 on 2026-07-06:

- `officialBaselineDecodeSecondsPerToken = 0.131727461265625`
- `officialBaselinePrefillSecondsPerToken = 0.01010573933984375`

These supersede the DeepSeek V4 Flash-era values (decode
3.6366560638046876 / prefill 0.17330563175390626), which were measured
against the previous challenge model and could not price Gemma inference.

If either number here disagrees with `Sources/MLXFastCore/Constants.swift`, the
freeze test fails on purpose -- the doc and the code must move together.

### Per-prompt baselines in the golden oracle

These constants are the **fallback**, not the only source. A golden's
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
  and the gates-only machine's placeholder timing all resolve from the golden.
  When absent, everything resolves from the constants above (the pre-pool
  behavior; all public fixtures carry none).
- Carrying baselines in the golden is what makes prompt-pool rotation rankable:
  each pool prompt ships its own officially measured calibration, so rotating
  prompts of different intrinsic difficulty keeps speedups comparable. It does
  NOT change the charged window -- adding a pool prompt is baseline work for
  that prompt only, never a re-baseline of the window itself.

### Paired baseline measurement (official timing machine)

Fixed constants compare a live single sample against a number measured on a
different physical host at a different hour. Measured fleet drift on identical
code across one day: prefill 0.163 -> 0.190 seconds/token (~10%+), decode ~4% --
enough to flip floor verdicts and swing scores for reasons unrelated to the
submission. Official ranked runs therefore measure the baseline live:

- The timing machine checks out the **pinned paired-baseline ref** (trusted
  workflow content; submissions cannot repoint it), builds it, transforms its
  own weights, and runs its timed benchmark against the same hidden golden in
  the same session, minutes before the candidate.
- The measured seconds-per-token are passed to the candidate benchmark through
  `MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN` /
  `MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN`. Resolution precedence in
  the harness is **paired override, then golden-carried baselines, then the
  constants**. The pair is fail-closed (both together, finite, positive) and
  both variables are stripped from the sandboxed worker environment.
- A **sanity band** guards the pairing: the baseline sample must fall within
  [0.66x, 1.5x] of the calibrated constants on both axes, so a pathological VM
  or broken baseline build fails the run instead of repricing every speedup.
- The baseline's own floor verdict is ignored (it is measured against its
  checked-in constants and legitimately fails on a slow-fleet session); token
  mismatches or harness errors in the baseline run fail the whole run.
- Changing the pinned ref is a ranking-contract change: it redefines what
  "1.0x" means. Update the ref, this doc, and the pinned guard test together.

With pairing active, the constants keep three roles: local-mode scoring (no
second build available locally), the gates-only machine's placeholder timing,
and the sanity-band anchor. Pool prompts no longer need pre-calibrated
baselines on the official path -- the baseline is measured live against
whichever prompt is active, which also removes the batched pool-calibration
session as a hard prerequisite for rotation.

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
