# MLXFast Challenge Agent Guide

This repository is the Swift-only DeepSeek V4 Flash 4-bit optimization challenge.
Use this file as the working contract for coding agents and participants.

## Goal

Optimize DeepSeek V4 Flash 4-bit inference on Apple Silicon without changing the
observable model behavior required by the correctness gates.

The official score rewards faster prefill and decode:

```text
decode_speedup = baseline_decode_sec_per_token / decode_sec_per_token
prefill_speedup = baseline_prefill_sec_per_token / prefill_sec_per_token
score = decode_speedup^0.75 * prefill_speedup^0.25
```

Higher is better. Baseline is about `1.0` on the official runner. Decode is
weighted more heavily because it dominates interactive generation. Both decode
and prefill must also stay within the configured 0.95 speedup floors.

## Official Hardware

Ranked benchmark runs execute through GitHub Actions on:

```text
blacksmith-12vcpu-macos-26
```

Treat this as the source of truth for performance. The ranked run is calibrated
for a Blacksmith Apple Silicon M4 Pro runner with 48 GB unified memory. Local
M2, M3, M4, or M5 machines are useful for iteration, but local speedups are only
directional. A kernel, cache, or streaming strategy that helps on one Apple
Silicon generation can move differently on the official runner, so always rely
on the official benchmark for ranking. Do not design for a 64 GB, 96 GB, or
128 GB local machine unless the same approach still fits the 48 GB official
runner budget.

## What You May Optimize

The submitted editable surface is defined in `benchmark.json`:

```text
Sources/MLXFastModel/
Sources/MLXFastTransform/
```

Focus on:

- Reducing scored prefill and decode seconds per token.
- Optimizing kernels and hot-path MLX operations used by attention, MoE, dense
  projections, KV-cache handling, and expert materialization.
- Reducing model execution work on the hot path: MLX ops, synchronization,
  materialization, copies, routing overhead, and cache misses.
- Improving expert streaming, caching, prefetching, and layout only when it
  shows up as lower scored prefill/decode latency.
- Making the offline transform produce better runtime metadata or compact
  transformed artifacts.
- Improving prefill and decode execution inside the Swift/MLX model path.

The model is DeepSeek V4 Flash 4-bit. The frozen reference checkpoint is about
141 GiB. `setup.sh` stores it in a repo-local Hugging Face-style cache by
default and verifies it against the pinned manifest. The transformed `weights/`
tree is an overlay/runtime artifact, not a second full copy of the model.
Aim to keep generated transformed weights under 20 GB.

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
`correctness_prompts/`. The official benchmark uses private artifacts supplied
by the organizer.

The official correctness stack includes:

- Teacher-forced token checks on 512-token prompt cases.
- Hidden behavior checks, including GPQA-style prompts.
- Short exact-token GPQA prefix checks from the private reference fixture.
- Semantic GPQA judging through a private judge path.
- TTFT guardrails for hidden GPQA first-token behavior.
- Benchmark oracle checks for the timed prefill/decode prompt.

The source of truth for current token counts and baseline constants is
`Sources/MLXFastCore/Constants.swift`.

## Timing And Score Measurement

The official benchmark measures:

- Prefill seconds per token.
- Decode seconds per token.
- Weighted score from prefill and decode speedups.
- Pass/fail component speed floors.

Diagnostic fields such as expert bytes read, memory, read timings, and
bandwidth are recorded for audit and future guardrails, but are not the primary
score unless the benchmark contract changes. Do not optimize for raw SSD speed
as a standalone target; optimize changes that reduce the measured prefill and
decode timings.

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
Metal library, downloads or verifies the DeepSeek V4 Flash 4-bit reference
checkpoint, and prepares the local cache. If the repo disk is too small, put the
reference cache on a larger SSD and set `MLXFAST_REFERENCE_CACHE_DIR` or
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
./benchmark.sh
```

`./benchmark.sh --local-iterate` is the fast local edit-loop signal.
Use it to compare the current working tree against the latest-tip baseline you
recorded above, not against a result from an older branch.
`./benchmark.sh --local-submit` is the Yukon pre-submit gate and is intended to
be longer and closer to the official path while still producing `score: null`.
`./benchmark.sh` is the full benchmark entrypoint and requires the required
hidden golden artifacts for ranked scoring.

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

- Expert tensor layout that reduces blocking work in measured prefill/decode.
- Per-layer or cross-step expert cache policy that fits the 48 GB runner.
- Predictive expert prefetch that lowers measured latency without depending on
  hidden prompts.
- MoE routing and dispatch overhead on the hot path.
- Dense/shared weight loading and reuse.
- KV cache handling and attention hot paths.
- MLX operation scheduling and synchronization.
- Transform metadata that lets runtime skip work safely.

Be careful with optimizations that only help a single public prompt or a single
machine. The hidden correctness and benchmark prompts are different from the
public local fixtures, and official scoring happens on the Blacksmith runner.

## Avoid These Wrong Strategies

Do not assume the benchmark machine has the same memory budget as your local
Mac. In particular, do not build a solution that relies on keeping thousands of
expert `MLXArray`s resident because it happens to fit on a high-memory local
machine. The challenge is about making DeepSeek V4 Flash fast under the official
runner contract, not about replacing SSD streaming with an unbounded in-memory
expert cache.

Avoid double-caching and cache bypasses that make diagnostics misleading. If
you add a cache, account for its memory use, eviction behavior, and interaction
with `ExpertSlotBank`; do not simply disable the existing byte cache or report
fake read/cache metrics to make the run look better.

Do not copy strategies from files outside this checkout or from parent
directories unless they are part of the public challenge repository. A
participant submission is judged from the submitted editable paths in this repo,
and relying on local-only source trees makes the implementation non-reproducible
for reviewers and the official runner.

Do not specialize for the public correctness prompt. Optimizations should be
prompt-independent and model-general for DeepSeek V4 Flash. Hidden correctness,
GPQA, and benchmark prompts are different from the public fixtures.

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

Do not treat local-only environment overrides as proof of a valid improvement.
Examples include disabling the sandbox, skipping transform without verifying
the produced `weights/`, pointing at a user-specific reference path, or tuning
with a large cache size that is not part of the official benchmark contract.
Those can be useful for debugging one machine, but they do not establish a
rankable optimization.

Do not draw conclusions from a tiny local iterate run alone. Short local modes
are smoke tests for speed and correctness direction. They are not substitutes
for the official hidden benchmark, and they are especially weak for testing
cache strategies because they may not exercise the same expert routing,
sequence length, or memory pressure as the ranked run.

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
