# MLXFast Challenge Agent Guide

This repository is the Swift-only Gemma 4 31B 4-bit dense inference
optimization challenge.
Use this file as the working contract for coding agents and participants.

## Goal

Optimize Gemma 4 31B 4-bit (text tower only) inference on Apple Silicon without
changing the observable model behavior required by the correctness gates.

The official score rewards faster prefill and decode:

```text
decode_speedup = baseline_decode_sec_per_token / decode_sec_per_token
prefill_speedup = baseline_prefill_sec_per_token / prefill_sec_per_token
score = decode_speedup^0.75 * prefill_speedup^0.25
```

Higher is better. The ranked score is paired: `baseline_*` is the pinned
reference implementation, measured on the same machine in the same session as
the submission, so unmodified reference code scores about `1.0`. Decode is
weighted more heavily because it dominates interactive generation. Both decode
and prefill must also stay within the configured 0.95 speedup floors.

## Official Hardware

Ranked benchmark runs execute through GitHub Actions on a single self-hosted
Apple M5 Max machine with 128 GB of unified memory. The runner label
configured in `.github/` is the source of truth; today that is:

```text
m5-bench
```

The box is operator-supervised: each ranked job runs on a fresh ephemeral
runner registration, every invocation of submitted code (build, transform,
correctness, benchmark) executes sandboxed, and the machine's protected
surface is integrity-audited between jobs — drift quarantines the box instead
of publishing a score. The whole ranked pipeline runs serially on that one
machine: build and transform, the public correctness gate, the hidden
64-step base case plus behavior/GPQA gates, the semantic judge, and — last,
after a quiescence wait — the timed prefill/decode measurement, which starts
each timed phase only once the GPU has cooled below a fixed 40C gate and
rejects throttled or telemetry-invalid measurements. See
`.github/workflows/benchmark.yml` for the exact step order.

Because the candidate and the pinned reference are measured back to back on
the same silicon behind the same thermal gate, the paired speedup ratio
cancels host drift; the score is that ratio, not a comparison against a
stored constant. Gemma 4 31B 4-bit is a dense model: the text tower is about
17 GB in 4-bit, fully RAM-resident on the ranked box — the runtime loads
every text-tower tensor once during untimed initialization and keeps it
resident for the whole process lifetime. There is no weight streaming of any
kind, no expert cache, and no disk I/O on the scored prefill/decode path.
Optimization effort should go into compute — attention kernels
(sliding-window vs. full-attention dispatch, GQA, partial-rotary RoPE),
quantized matmul dispatch, KV-cache handling, memory layout, and MLX
scheduling — not disk I/O.

Local machines need enough unified memory to hold the ~17 GB text tower plus
KV cache and activation buffers; about 36 GB is a practical local minimum.
The ranked box has more headroom than that, but memory-hungry strategies
tuned against a different machine still have to survive the paired
measurement on the M5, and a kernel or layout strategy that helps on one
Apple Silicon generation can move differently there — always rely on the
official benchmark for ranking.

## What You May Optimize

The submitted editable surface is defined in `benchmark.json`:

```text
Sources/MLXFastModel/
Sources/MLXFastTransform/
```

Focus on:

- Reducing scored prefill and decode seconds per token.
- Optimizing kernels and hot-path MLX operations used by attention (both the
 sliding-window and full-attention layer types), the gated MLP, KV-cache
 handling, and dense weight materialization.
- Reducing model execution work on the hot path: MLX ops, synchronization,
 materialization, copies, and cache misses.
- Improving how RAM-resident dense weight bytes become MLXArrays (quantized
 linear construction, fewer copies, lazier Data-to-Metal conversions).
- Making the offline transform produce better runtime metadata or compact
 transformed artifacts.
- Improving prefill and decode execution inside the Swift/MLX model path.

The model is Gemma 4 31B, dense, 4-bit, text tower only (vision/audio are out
of scope and are never loaded). The frozen reference checkpoint is about
18.4 GB across 4 safetensors shards. `setup.sh` stores it in a repo-local
Hugging Face-style cache by default and verifies it against the pinned
manifest. The transformed `weights/` tree holds only the text-tower tensors
(everything under the source checkpoint's `language_model.` prefix) plus a
runtime-authored `config.json`; it is an overlay/runtime artifact, not a
second full copy of the model. Aim to keep generated transformed weights under
20 GB (the default cap is 25 GiB).

## What Not To Change

Do not spend time modifying files outside `editablePaths` for a submission.
They are trusted harness/operator code and are not packaged by submit:

- `Sources/MLXFastCore/`
- `Sources/MLXFastHarness/`
- `Sources/MLXFastCLI/`
- `.github/`, scripts, tests, docs, `benchmark.json`
- `weights/`, reference checkpoints, scores, golden files, local caches

Do not try to hardcode hidden prompts, hidden token IDs, GPQA answers, timing
shortcuts, protocol injection, network access, or filesystem exfiltration. The
official runner uses private artifacts, sandboxed runtime workers, artifact
validation, trusted workflow code, and static review gates. Hidden prompts and
goldens are not part of the public repo or submission payload.

Python is not part of the challenge runtime. Setup, transform, correctness, and
benchmark run through the Swift package. Account login, clone, and submission
use the Yukon CLI (`mlxfast`).

## Correctness Gates

Correctness is a hard gate. Passing locally is necessary but not sufficient for
ranking.

The public local gate uses checked-in prompt/golden fixtures under
`correctness_prompts/`. These fixtures were generated on the ranked M5
hardware from the reference implementation — the Layr-Labs `mlx-swift-lm`
Gemma 4 text tower this package builds against: the prompt text is tokenized
with the Gemma 4 tokenizer and the expected tokens are greedy continuations
captured with `mlxfast-swift generate-golden`; see `TASK.md`. The hidden
artifacts used by ranked runs were regenerated the same way from the same
reference on the same hardware.

Know this before debugging a local failure: greedy argmaxes with near-tie
logits can diverge across Apple Silicon generations, so on non-M5 machines
the local public gate may fail for a perfectly correct build. Local modes are
directional; the ranked M5 runner is the source of truth for both correctness
and timing.

The official correctness stack includes:

- Teacher-forced token checks on 512-token prompt cases.
- Hidden behavior checks, including GPQA-style prompts.
- Short exact-token GPQA prefix checks from the private reference fixture.
- Semantic GPQA judging through a private judge path.
- TTFT guardrails for hidden GPQA first-token behavior.
- Benchmark oracle checks for the timed prefill/decode prompt.

The source of truth for current token counts, gate thresholds, and baseline
constants is `Sources/MLXFastCore/Constants.swift`.

## Timing And Score Measurement

The official benchmark measures:

- Prefill seconds per token.
- Decode seconds per token.
- Weighted score from prefill and decode speedups.
- Pass/fail component speed floors.

The timed measurement runs last in the ranked job, after all correctness and
gate work, behind a fixed 40C GPU thermal gate with telemetry-validated
acceptance; the pinned reference is measured the same way on the same box in
the same session, and speedups (and the 0.95 floors) are computed from that
paired ratio. The `officialBaseline*` constants in
`Sources/MLXFastCore/Constants.swift` feed local-mode estimates only; they
are not the ranked denominator.

Diagnostic fields such as memory and read timings are recorded for audit and
future guardrails, but are not the primary score unless the benchmark contract
changes. There is no expert/weight-streaming bandwidth to report for this
dense model: `bandwidth_gb_per_token` is always `0` with
`bandwidth_source=ram_resident_model`. Do not optimize for that diagnostic
field as a standalone target; optimize changes that reduce the measured
prefill and decode timings.

The benchmark charges decode setup to the decode measurement so model code
cannot hide future-token work in an unscored seed-prefill phase.

## Local Workflow

Before optimizing, sync to the latest challenge tip and record a same-machine
local baseline. Do not compare your changes against a stale branch or an old
local run:

```bash
git fetch origin main
git switch main
git pull --ff-only
./setup.sh
./benchmark.sh --local-iterate
cp score.local-iterate.json score.local-iterate.baseline.json
```

Create your working branch from that synced commit, or rebase/merge your
existing branch onto `origin/main` before trusting local timings. Every
`./benchmark.sh --local-iterate` result should be interpreted as performance on
top of the latest synced base commit measured on the same local machine, with
the same toolchain, model cache, power state, and thermal conditions. If the
base commit changes, rerun the local baseline before deciding whether an
optimization is faster.

Start with:

```bash
./setup.sh
```

This checks the local Swift/Xcode toolchain, builds the Swift harness and MLX
Metal library, downloads or verifies the Gemma 4 31B 4-bit reference
checkpoint, and prepares the local cache. If the repo disk is too small, put the
reference cache on a larger volume and set `MLXFAST_REFERENCE_CACHE_DIR` or
`MLXFAST_REFERENCE_DIR`.

Common commands:

```bash
swift test
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test
swift build -c release
.build/release/mlxfast-swift transform --output weights
.build/release/mlxfast-swift correctness --weights weights
./benchmark.sh --local-iterate
./benchmark.sh --local-submit
./benchmark.sh --official
```

`./benchmark.sh --local-iterate` is the fast local edit-loop signal.
Use it to compare the current working tree against the latest-tip baseline you
recorded above, not against a result from an older branch.
`./benchmark.sh --local-submit` is the Yukon pre-submit gate and is intended to
be longer and closer to the official path; like `--local-iterate` it publishes
only the estimated local score (never the official ranked score).
`./benchmark.sh --official` is the full ranked entrypoint and requires the
hidden golden artifacts provisioned on the official runner. A bare
`./benchmark.sh` defaults to `--local-iterate`. Remember the public fixtures
are M5-generated: a local correctness failure on non-M5 hardware can be a
near-tie argmax divergence rather than a real bug (see Correctness Gates).

## Notes For Autonomous Agents

Operational contract for coding agents iterating in this repo. These
behaviors are expected, not bugs:

- **The local GPU cool-down gate waits; let it.** `./benchmark.sh
 --local-iterate` and `--local-submit` wait for the GPU to cool below 40C
 before starting the timed run (read via `macmon`), printing a progress
 line roughly every 10 seconds while waiting. A benchmark invocation that
 pauses on "waiting for GPU to cool down" is working, not hung — do not
 kill it or treat the wait as a failure. If the GPU stays hot and is not
 trending down, the gate aborts with a non-zero exit after about 3
 minutes; that abort means "something else is loading the GPU — free it
 up and retry," not "the code change is wrong." (A hard 900-second
 ceiling applies even while the GPU is still slowly cooling.) If `macmon`
 is not installed the gate warns and skips; `./setup.sh` installs it (or
 `brew install macmon`). The gate mirrors the ranked runner's fixed
 40C / 1600 MHz / 900 s thermal contract, which is operator-owned and
 non-overridable.
- **Measurement discipline.** Trust benchmark numbers only from a cool,
 quiescent machine. Back-to-back runs heat the GPU and throttle it; a
 2-3 minute cool-down between local runs is normal and is exactly what
 the gate enforces. Do not fight the gate to iterate faster:
 `MLXFAST_LOCAL_COOL_GATE=0` is for debugging only and produces
 hot-start timings that are not comparable to gated ones. The ranked
 score is a paired speedup versus the on-box pinned reference measured
 in the same session (~1.0 for unmodified code, with decode variance
 around 0.1% CV in validation runs); the `officialBaseline*`
 constants feed local-mode estimates only, so treat local absolute
 numbers and local score estimates as directional.
- **A local gate failure on non-M5 hardware may not be your bug.** The
 public goldens are M5-generated greedy continuations of the
 mlx-swift-lm reference; near-tie argmaxes can diverge on other Apple
 Silicon generations even for correct code. Before treating a local
 public-gate failure as a regression, check whether unmodified `main`
 fails at the same token position on your machine; the ranked M5 runner
 is the source of truth.
- **Know the runnable surface.** Only `Sources/MLXFastModel/` and
 `Sources/MLXFastTransform/` ship in a submission (`benchmark.json`
 `editablePaths`); changes anywhere else will not upload even if they
 help locally. `./benchmark.sh --official` requires the hidden goldens
 and private oracle provisioned on the official runner and is not
 runnable locally — use `--local-iterate` for the edit loop and
 `--local-submit` as the pre-submit gate.
- **One ranked machine, one queue.** Ranked runs execute serially on the
 single M5 runner: one job at a time by construction, and duplicate
 dispatches queue behind the in-flight run instead of cancelling it.
 Expect queueing delays behind other submissions, and do not dispatch
 multiple ranked runs in parallel expecting concurrent results.

## Swift Tooling

Use the Swift toolchain that `./setup.sh` validates. `sourcekit-lsp` is the
standard Swift language server and is usually installed with Xcode or the Swift
toolchain. Point your editor at the repository root so SourceKit-LSP can read
`Package.swift` and resolve the SwiftPM targets.

Useful local tooling commands:

```bash
swift package resolve
swift build -c release
swift test
sourcekit-lsp
xcode-select -p
xcrun --find sourcekit-lsp
```

For editor agents, prefer SourceKit-LSP symbol navigation and diagnostics over
string-only edits when changing Swift model code. Use `swift test` for cheap
contract checks, and use `MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test` when a
change touches MLX runtime behavior and the machine can run those tests.

## Submission Workflow

Use Yukon/Darkbloom submit commands through the Yukon CLI:

```bash
export PATH="${HOME}/.local/bin:${PATH}"
mlxfast login <api-key> --api <url>
mlxfast clone <benchmark-id-or-name>
mlxfast submit --model "<model name>" --note "describe optimization"
mlxfast submissions
```

Submit packages only `editablePaths`. It rejects generated artifacts, symlinks,
local scores, reference checkpoints, and source changes outside the editable
surface. Live submit first runs the configured local pre-submit benchmark, then
uploads the editable-path archive for official validation.

## Practical Optimization Ideas

Good submissions are likely to improve one or more of:

- Attention kernel dispatch: sliding-window vs. full-attention masking, GQA
 head-group broadcasting, and the full-attention layers' partial-rotary
 ("proportional") RoPE.
- Quantized matmul dispatch for the affine 4-bit (group size 64) dense
 projections: fewer dequantize/copy steps, better batching across the gated
 MLP's `gate_proj`/`up_proj`, and reuse of derived weight views.
- KV cache handling: the sliding-window cache only ever needs the last 1024
 positions; a tighter ring-buffer implementation can reduce both memory and
 copy overhead relative to the straightforward baseline.
- Dense weight loading and reuse: eager preparation at init, warm kernels
 before the first scored forward, and avoiding redundant Data-to-Metal
 conversions.
- MLX operation scheduling and synchronization.
- Transform metadata that lets runtime skip work safely.

Be careful with optimizations that only help a single public prompt or a single
machine. The hidden correctness and benchmark prompts are different from the
public local fixtures, and official scoring happens on the single self-hosted
M5 runner.

## Avoid These Wrong Strategies

Do not assume the benchmark machine has the same memory budget as your local
Mac. The ranked box is one Apple M5 Max with 128 GB of unified memory; the
~17 GB text tower is comfortably RAM-resident there, but do not treat that
headroom as an invitation for memory-hungry strategies tuned on a different
machine — KV cache, buffers, and caches still compete, and what is fast on
your Apple Silicon generation can move differently on the M5. Unlike a large
MoE checkpoint there is no meaningfully different "streaming fallback" regime
here to mistune against.

Do not specialize for the public correctness prompt. Optimizations should be
prompt-independent and model-general for Gemma 4. Hidden correctness, GPQA,
and benchmark prompts are different from the public fixtures.

Do not treat local-only environment overrides as proof of a valid improvement.
Examples include disabling the sandbox, skipping transform without verifying
the produced `weights/`, pointing at a user-specific reference path, or tuning
with settings that are not part of the official benchmark contract. Those can
be useful for debugging one machine, but they do not establish a rankable
optimization.

Do not draw conclusions from a tiny local iterate run alone. Short local modes
are smoke tests for speed and correctness direction. They are not substitutes
for the official hidden benchmark, and they are especially weak for testing
sequence-length-dependent optimizations (e.g. attention kernel changes) since
they may not exercise the same sequence lengths or memory pressure as the
ranked run.

## Before Submitting

Run at least:

```bash
swift test
./setup.sh
.build/release/mlxfast-swift correctness --weights weights
./benchmark.sh --local-submit
```

If the local correctness gate fails, the official benchmark will not rank the
submission. If local performance improves but correctness is fragile, prefer a
more conservative optimization.

Do not add caches or memos keyed on a request's input tokens whose only
possible hit is the benchmark harness repeating an identical computation — for
example, memoizing a whole-prompt forward's logits or KV state so a repeated
identical forward can skip the work. Bit-identical output does not make this
legitimate. The benchmark measures single-pass inference: optimizations must
save work that recurs in single-pass production inference (one prefill, then
decode, per prompt), not work that only exists in the measurement protocol.
The harness never legitimately issues the same whole-prompt forward twice to
one worker process; any such repetition is a harness bug, never a contract to
rely on. Input-independent caching (weights, dequantized tensors, RoPE/mask
tables keyed on shapes and offsets) and within-request KV reuse remain fine.
Submissions in this category fail the static review as bypass behavior.
