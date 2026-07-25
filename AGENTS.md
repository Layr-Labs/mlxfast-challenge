# MLXFast Challenge Agent Guide

This repository is the Swift-only Poolside Laguna XS 2.1 NVFP4 MoE inference
optimization challenge.
Use this file as the working contract for coding agents and participants.

## Goal

Optimize Poolside Laguna XS 2.1 NVFP4 (text tower only) inference on Apple
Silicon without
changing the observable model behavior required by the correctness gates.

The default — and only — ranked track is `laguna-xs-2.1-serial-v2`, the
serial (one token per decode request) track. It rewards faster prefill and
decode against a paired on-box baseline measured in the same session:

```text
score = decode_speedup^0.75 * prefill_speedup^0.25
```

Higher is better. Each speedup is the pinned baseline's seconds/token divided
by the candidate's for that phase, both measured on the same machine behind
the same thermal gate. Both component floors are `0.95`, hard, and every
checked token must match the golden. A two-sided acceptance band applies on
top of the floors and is tighter than them: measured against the pinned
calibration reference, `decode_speedup` must land in `[0.980, 1.053]` and
`prefill_speedup` in `[0.952, 1.053]`, so a single submission's gain is
capped at about 5% and larger wins must be chunked across submissions (the
published paired speedup is floors-only and can land slightly outside the
window; see "Timing And Score Measurement" below).
`benchmark.json` registers the serial
track and `.github/workflows/benchmark.yml` is the serial ranked pipeline.

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
of publishing a score. The pipeline verifies the pre-provisioned reference
checkpoint against the pinned manifest, builds and transforms, runs the
public drift tripwire and the hidden correctness/gates pass, then runs the
timed paired measurement LAST. Every timed phase starts only once the GPU has
cooled below a fixed 40C gate and rejects throttled or telemetry-invalid
measurements. See `.github/workflows/benchmark.yml` for the exact step order.

Because the candidate and the pinned on-box baseline are measured back to
back on the same silicon behind the same thermal gate, the paired speedup
ratio cancels host drift; the score is that ratio, not a comparison against a
stored constant. Poolside Laguna XS 2.1 NVFP4 is a fine-grained MoE model (256 routed
experts plus one shared expert per sparse layer, 8 experts per token,
per-head gating; layer 0 is a dense MLP): the text tower is about
21.6 GB in Poolside NVFP4, fully RAM-resident on the ranked box — the runtime loads
every text-tower tensor, including all experts, once during untimed
initialization and keeps it
resident for the whole process lifetime. There is no weight streaming of any
kind, no expert cache, and no disk I/O on the scored prefill/decode path.
Optimization effort should go into compute — attention kernels
(sliding-window vs. full-attention dispatch, GQA, YaRN partial-rotary RoPE on
full-attention layers), quantized matmul and MoE gather-GEMM dispatch,
KV-cache handling, memory layout, and MLX
scheduling — not disk I/O.

Local work needs enough unified memory for the ~21.6 GB model plus KV state
and buffers; roughly 40 GiB is the practical local minimum.
Machines below 64 GiB automatically use a low-memory startup profile:
compiled decode is disabled, the MLX
allocator cache is capped at 6 GiB, and free warmup buffers are cleared before
the worker protocol starts. The profile prints a one-line stderr notice when
it engages and only fills in feature flags you have not set yourself —
explicitly exported `DARKBLOOM_*`/`MLXFAST_*`/`MLX_*` values always win.
Local `./benchmark.sh --local-iterate`/`--local-submit` runs whose profile
downgraded ranked-path flags this way also end with a ranked-parity warning
in the final summary listing exactly which flags were disabled and how to
opt in. Set
`DARKBLOOM_STARTUP_MEMORY_PROFILE=full|low|auto` to override the automatic
selection. Note the low profile does not exercise compiled decode locally,
so verify decode-path-sensitive changes on a 64 GiB+ machine or
rely on the ranked run. The 128 GB ranked runner keeps the full profile.
The ranked box has more headroom than that, but memory-hungry strategies
tuned against a different machine still have to survive the paired
measurement on the M5, and a kernel or layout strategy that helps on one
Apple Silicon generation can move differently there — always rely on the
official benchmark for ranking.

## What You May Optimize

The submitted editable surface is defined by `editablePaths` in
`benchmark.json` — that list (currently 97 entries) is the source of truth.
It covers four groups:

```text
Sources/MLXFastModel/ Laguna runtime glue, custom kernels, decode path
Sources/MLXFastTransform/ offline weight transform
Vendor/mlx-swift-lm/ the Laguna model files + MLXLMCommon plumbing
Vendor/mlx-swift/ the MLX Metal kernels Laguna dispatches
```

The vendored model surface is `Libraries/MLXLLM/Models/Laguna.swift` plus
the `MLXLMCommon` files it uses directly (MoE/attention dispatch helpers,
KV caches, RoPE utilities and application, compiled decode, evaluation
plumbing; the exact file list is in `benchmark.json`).

Know which of those vendored files the scored path actually executes.
`Laguna.swift` is a reference implementation, not the scored forward pass:
the benchmark runs the runtime port in
`Sources/MLXFastModel/LagunaRuntimeModel.swift`, and the vendored model is
kept as the behavior oracle for the gated upstream-equivalence cross-check
(`Sources/MLXFastModel/LagunaUpstreamEquivalence.swift`), so edits to
`Laguna.swift` do not change scored timings unless your runtime calls into
it. The vendored `MLXLMCommon` helpers ARE executed by the runtime —
`SwitchLayers.swift` (MoE expert gather/dispatch), `AttentionUtils.swift`
(attention-with-cache-update dispatch), `KVCache.swift` (standard and
rotating caches), `RoPEUtils.swift` / `RoPEApplication.swift`, and
`CompiledDecode.swift` with its compilable cache variants — as are the
vendored `Vendor/mlx-swift` kernels described below.

The vendored kernel surface is the kernel families the Laguna forward pass
actually dispatches — SDPA (`scaled_dot_product_attention.metal`,
`sdpa_vector.h`, and `steel/attn/` with its `steel_attention*.cpp` twins:
Laguna's head dim is 128, so the fused steel attention kernels are
dispatchable at prefill), quantized matmul (the NVFP4 `fp_quantized*` kernels
and the shared `quantized*` infrastructure, including the `_nax`
variants), the MoE gather GEMM (`steel_gemm_gather*.cpp`), `steel/gemm/`,
`gemv`, `rope`, `rms_norm`, `softmax`, `sort`, `reduce`, `copy`, elementwise
(`unary`/`binary`/`ternary`), `arg_reduce`, and gather indexing
— in both forms the build uses: the AOT `.metal`/`.h` sources under
`Vendor/mlx-swift/.../backend/metal/kernels/` and their JIT twins under
`Vendor/mlx-swift/Source/Cmlx/mlx-generated/*.cpp`.

Know how a kernel edit becomes the running kernel. The vendored MLX Swift
package builds in JIT mode: families with an `mlx-generated/*.cpp` twin
(quantized incl. fp_quantized, steel/gemm incl. the gather GEMM, steel/attn,
gemv, softmax, sort, reduce, copy, elementwise, gather) are
compiled at runtime from the C++ source strings embedded in those files, so
for them the twin is the runtime-effective source — editing only the
`.h`/`.metal` form does not change what runs; edit the pair together.
Families without a twin (RoPE, RMSNorm, the SDPA vector kernel,
`arg_reduce`) are served ahead-of-time from `mlx.metallib`, built from the
vendored `.metal` sources by `tools/build-mlx-metallib.sh` (invoked by
`./setup.sh`; rerun either after editing an AOT `.metal`/`.h` file). `_nax`
names are the M5-generation kernel variants; the ranked M5 box selects
them, so tune the `_nax` twin as well as the plain one. Then test with
`./benchmark.sh --local-iterate`, which rebuilds both binaries for you
whenever a build input is newer than them. A bare `swift build -c release`
is **not** enough on its own: without `--scratch-path .build-worker` it
writes `.build/release`, while the scored binary is
`.build-worker/release/mlxfast-runtime-worker`.

All participant model and kernel code — `MLXFastModel` plus the vendored
forks — compiles into the sandboxed `mlxfast-runtime-worker` binary. The
trusted `mlxfast-swift` binary (timing, gates, scoring) links no MLX,
model, or kernel code and drives the worker over a JSON protocol; your
hot-path code runs only inside the worker.

Focus on:

- Reducing scored prefill and decode seconds per token.
- Optimizing the vendored Metal kernels on the prefill and decode paths.
- Optimizing kernels and hot-path MLX operations used by attention (both the
 sliding-window and full-attention layer types), the MoE MLP (routing,
 expert gather GEMM, shared expert), KV-cache
 handling, and weight materialization.
- Reducing model execution work on the hot path: MLX ops, synchronization,
 materialization, copies, and cache misses.
- Improving how RAM-resident weight bytes become MLXArrays (quantized
 linear construction, fewer copies, lazier Data-to-Metal conversions).
- Making the offline transform produce better runtime metadata or compact
 transformed artifacts.
- Improving prefill and single-token decode inside the Swift/MLX model path.

The target is Poolside Laguna XS 2.1, MoE, NVFP4 (untied embeddings, vocab
100352), text tower only (the empty `vision_config` is out of scope and never
loaded). The frozen reference checkpoint is about
21.6 GB across 5 safetensors shards, verified against the pinned manifest.
The transformed `weights/` tree holds the source's text-only `model.*` /
`lm_head.*` tensors plus a
runtime-authored `config.json`; it is an overlay/runtime artifact, not a
second physical copy of the model on APFS. Aim to keep generated transformed
weights under 22 GB (the default cap is 25 GiB).

## What Not To Change

Do not spend time modifying files outside `editablePaths` for a submission.
They are trusted harness/operator code and are not packaged by submit:

- `Sources/MLXFastCore/`, `Sources/MLXFastCLI/`,
 `Sources/MLXFastTrustedHarness/`, `Sources/MLXFastHarness/`, and
 `Sources/MLXFastRuntimeWorkerCLI/`
- `Package.swift` and `Package.resolved` — the dependency graph is frozen
 (the vendored forks are consumed as pinned local path dependencies)
- Everything in `Vendor/` not listed in `editablePaths`: other model
 families, shared model-factory/tokenizer plumbing, and kernels Laguna
 does not dispatch
- `.github/`, scripts, tests, docs, and `benchmark.json`
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

Correctness is a hard gate. The ranked M5 runner is the authority.

The official correctness stack includes:

- The public drift tripwire (one 64-step teacher-forced check against the
 checked-in public fixture, run before any hidden material enters the
 workspace).
- The hidden full-length teacher-forced base case (512-token prompts) plus
 hidden anchor, free-run, and GPQA behavior gates.
- The GPQA TTFT guardrail and the semantic GPQA judge (Anthropic).
- The timed phase's own token acceptance: the trusted measure wrapper
 generates a per-binary oracle for the hidden timed prompt and rejects any
 mismatched output.

Note the public checked-in fixtures are M5-generated; a near-tie argmax can
diverge on other Apple Silicon generations even for correct code.

## Timing And Score Measurement

The official benchmark measures prefill and decode seconds/token for the
candidate and the pinned on-box baseline in the same session, then publishes
the weighted paired speedup `decode_speedup^0.75 * prefill_speedup^0.25`.
Both component floors are `0.95`, hard.

A two-sided acceptance band applies on top of the floors
(`MLXFastConstants.{prefill,decode}Band{Up,Down}Tolerance`): measured
seconds/token must land within +5%/-5% of the pinned calibration reference
for prefill and +2%/-5% for decode, i.e. speedup against that pinned
reference in `[0.980, 1.053]` for decode and `[0.952, 1.053]` for prefill.
The reference is the cached ranked-box calibration
(`MLXFastConstants.officialBaseline{Prefill,Decode}SecondsPerToken`, or
golden-supplied baseline fields carrying the same calibration), not the
same-session paired baseline: `overlay-paired-timing.sh` applies only the
0.95 floors to the published paired ratio, so the published
`decode_speedup`/`prefill_speedup` can land slightly outside the band
window when the session baseline drifts from the pinned calibration mean.
The down side deliberately caps a single submission's gain at about 5% —
larger wins are either lucky measurements or too big to trust in one shot
and must be chunked across submissions; the cap is per submission, not
cumulative. The band is frozen
policy (`docs/benchmark-window-freeze.md`), it is NOT evaluated by
`--local-iterate` / `--local-submit`, and a ranked run that trips it fails
with failure category `acceptance_band_failed`. Local modes print a warning
when their estimate is more than 5% faster than the pinned calibration
reference (the "chunk it" direction only — the slow edge is already covered
by the floors, and a two-sided local check would fire on every run on
hardware slower than the ranked box), but they never fail on it.

The timed measurement runs last in the ranked job, after all correctness and
gate work and after every hidden byte is scrubbed from the bench workspace,
behind a fixed 40C GPU thermal gate with telemetry-validated acceptance
(throttled samples reject the measurement, with one gated retry). The timed
window is frozen: a 512-token prefill prompt and a teacher-forced decode pass
(512-token seed, 128 decode steps) — see `docs/benchmark-window-freeze.md`.

Diagnostic fields such as memory and read timings are recorded for audit and
future guardrails, but are not the primary score unless the benchmark contract
changes. There is no expert/weight-streaming bandwidth to report — every
routed expert is RAM-resident: `bandwidth_gb_per_token` is always `0` with
`bandwidth_source=ram_resident_model`. Do not optimize for that diagnostic
field as a standalone target; optimize changes that reduce the measured
prefill and decode times.

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
cp score.json score.baseline.json
```

Create your working branch from that synced commit, or rebase/merge your
existing branch onto `origin/main` before trusting local timings. Every
`./benchmark.sh --local-iterate` result should be interpreted as performance
on top of the latest synced base commit measured on the same local machine,
with the same toolchain, model cache, power state, and thermal conditions. If
the base commit changes, rerun the local baseline before deciding whether an
optimization is faster.

Start with:

```bash
./setup.sh
```

This checks the local Swift/Xcode toolchain, builds the Swift harness and MLX
Metal library, then downloads and verifies the pinned Laguna XS 2.1 reference
checkpoint (use `MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1` while the checked-in weight
manifests are still entry-less placeholders, or when the checkpoint is
provisioned externally).

Common commands:

```bash
swift test --force-resolved-versions
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions
swift build -c release --force-resolved-versions
tools/build-mlx-metallib.sh
./benchmark.sh --local-iterate
./benchmark.sh --local-submit
```

Pass `--force-resolved-versions` on every direct `swift build` / `swift
test`: the dependency graph is frozen, and a bare invocation can silently
rewrite `Package.resolved` (SwiftPM re-resolves the ranged transitive pins
on toolchain drift), after which `./setup.sh` and `./benchmark.sh` refuse
to run until you restore it with `git checkout -- Package.resolved`. The
flag makes SwiftPM fail closed instead.

`./benchmark.sh --local-iterate` is the fast local edit-loop signal.
Use it to compare the current working tree against the latest-tip baseline you
recorded above, not against a result from an older branch.
`./benchmark.sh --local-submit` is the recommended manual pre-submit check
(`mlxfast submit` does not run it for you) and is intended to be longer and
closer to the official path; like `--local-iterate` it publishes only a
local estimated score (never the official ranked score). The hidden M5
goldens remain the fidelity authority.

## Notes For Autonomous Agents

Operational contract for coding agents iterating in this repo. These
behaviors are expected, not bugs:

- **Cool-down gate.** `./benchmark.sh
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
- **Optional fan boost for a stalled local cool-down.** If the local gate
 sits hot for ~60 seconds with no cooling progress, it offers — once per
 run, interactive terminal only — to force the Mac's fans to a hard-coded
 70% of their maximum speed via `tools/fan-control.sh`. Fan targets are
 SMC keys that macOS only lets root write, so accepting the offer
 triggers sudo's own password prompt; the scripts never read, store, or
 log the password, and the cached credential is dropped (`sudo -k`)
 right after the write. `./benchmark.sh --fan-speed-normal` removes the
 70% override and returns the fans to macOS's automatic curve (no pinned
 RPM). Manual control uses the same helper:
 `tools/fan-control.sh boost|normal|status` (needs an `smc` CLI, e.g.
 from smcFanControl; fanless Macs are refused cleanly).
- **Measurement discipline.** Trust timing numbers only from a cool,
 quiescent machine. Back-to-back runs heat the GPU and throttle it; a
 2-3 minute cool-down between local runs is normal. The wrapper
 enforces this automatically. Do not fight the gate to iterate faster:
 `MLXFAST_LOCAL_COOL_GATE=0` is for debugging only and produces
 hot-start timings that are not comparable to gated ones. The ranked
 score is a paired speedup versus the on-box pinned baseline measured
 in the same session. Treat local scores as directional.
- **A local gate failure on non-M5 hardware may not be your bug.** The
 public goldens are M5-generated greedy continuations of the
 mlx-swift-lm reference; near-tie argmaxes can diverge on other Apple
 Silicon generations even for correct code. Before treating a local
 public-gate failure as a regression, check whether unmodified `main`
 fails at the same token position on your machine; the ranked M5 runner
 is the source of truth. If it does, rerun with
 `MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT=1` so the local mode still publishes
 its timing estimate instead of a `score: null` the Yukon CLI rejects.
 The override is local-only and does not hide the divergence: the score
 keeps `passed_correctness: false`, records the diverging tokens, and
 explains itself in `metrics.error`. Never use it to paper over a real
 regression -- if unmodified `main` passes on your machine, the mismatch
 is yours.
- **Know the runnable surface.** Only the `benchmark.json` `editablePaths`
 entries ship in a submission: `Sources/MLXFastModel/`,
 `Sources/MLXFastTransform/`, the vendored Laguna model and `MLXLMCommon`
 files, and the listed vendored kernel sources (both AOT `.metal`/`.h`
 and JIT `mlx-generated/*.cpp` forms); changes anywhere else will not
 upload even if they help locally. Official ranking requires hidden
 organizer goldens and is not runnable locally — use
 `./benchmark.sh --local-iterate` for the edit loop and
 `--local-submit` as the recommended pre-submit check.
- **One local run at a time; the memory guard is protecting your RAM.** The
 ~21.6 GB RAM-resident text tower means two simultaneous model residencies
 (an overlapping second local run, or a new run started while an orphaned
 model-holding worker from an aborted run lingers) can out-of-memory a
 local machine. A single worker is separately protected by the automatic
 low-memory startup profile described above. Local modes take a per-user
 run lock and refuse to start while a model-holding mlxfast process is
 still alive, printing the offending pid/rss/command list. Read that list
 before reacting: a
 ppid of 1 is usually an orphan from an aborted run (verify, then
 `kill <pid>`); a live ppid is usually a legitimately concurrent run
 (wait for it). The guard warns and aborts -- it never kills anything
 itself. Aborted local runs now reap their own worker on INT/TERM/EXIT
 and the worker exits if its parent dies, so the guard should fire
 rarely. Know its scope: the lock lives in `benchmark.sh`, so direct
 `mlxfast-swift` model commands (`correctness`, `correctness-trace`,
 `generate-golden`, and `generate-gpqa-answers`) take no lock
 and do not check for other runs -- run one model-holding command at a
 time, never concurrently with a local benchmark or with each other.
 (`swift test` never loads the real model and is safe alongside.)
 `MLXFAST_LOCAL_RUN_GUARD=0` disables the guard for harness debugging
 only -- never set it to resolve contention; wait for the other run
 instead. The ranked --official path is unaffected.
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
swift build -c release --force-resolved-versions
swift test --force-resolved-versions
sourcekit-lsp
xcode-select -p
xcrun --find sourcekit-lsp
```

Avoid bare `swift package resolve` / `swift package update`: they can
rewrite the frozen `Package.resolved` (there is no fail-closed flag for
`resolve`), and `./setup.sh` plus the flagged builds already resolve
fail-closed. If `Package.resolved` ever shows as modified, restore it with
`git checkout -- Package.resolved`.

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
surface. `mlxfast submit` uploads the editable-path archive directly for
official validation; it does not run a local benchmark first, and no local run
blocks the upload. Run `./benchmark.sh --local-submit` yourself before
submitting — the official M5 run is the gate that ranks the submission.

## Practical Optimization Ideas

Good submissions are likely to improve one or more of:

- Kernel-level optimization inside the vendored Metal sources. Prioritize
 kernels reached by the timed prefill and decode phases. For the JIT
 families, edit the `mlx-generated/*.cpp` twin — that string is what
 compiles at runtime.
- Attention kernel dispatch: sliding-window vs. full-attention masking, GQA
 head-group broadcasting (8 KV heads at head_dim 128), and the
 full-attention layers' YaRN partial-rotary (0.5) RoPE.
- NVFP4 matmul and MoE dispatch for the group-16 routed/shared expert
 projections: fewer dequantize/copy steps, better expert gather-GEMM
 batching across routed experts, the BF16 per-layer router gates, the
 shared expert, and reuse of derived weight views.
- KV cache handling: the sliding-window cache only ever needs the last 512
 positions; a tighter ring-buffer implementation can reduce both memory and
 copy overhead relative to the straightforward baseline.
- Weight loading and reuse: eager preparation at init, warm kernels
 before the first scored forward, and avoiding redundant Data-to-Metal
 conversions.
- MLX operation scheduling and synchronization.
- Transform metadata that lets runtime skip work safely.

Be careful with optimizations that only help a single public prompt or a single
machine. The hidden correctness and benchmark prompts are different from the
public local fixtures, and official scoring happens on the single self-hosted
M5 runner. Kernel edits are bound by the same correctness gates as model
edits: keep them prompt-independent and model-general for Laguna, and be
conservative with numeric reassociation — a changed accumulation order can
flip near-tie greedy argmaxes on the M5 and fail the exact-token gates.

## Avoid These Wrong Strategies

Do not assume the benchmark machine has the same memory budget as your local
Mac. The ranked box is one Apple M5 Max with 128 GB of unified memory; the
~21.6 GB text tower is comfortably RAM-resident there, but do not treat that
headroom as an invitation for memory-hungry strategies tuned on a different
machine — KV cache, buffers, and caches still compete, and what is fast on
your Apple Silicon generation can move differently on the M5. Although Laguna
is an MoE checkpoint, every expert is RAM-resident: there is no
"streaming fallback" regime here to mistune against.

Do not specialize for the public correctness prompt. Optimizations should be
prompt-independent and model-general for Laguna. Hidden correctness, GPQA,
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
swift test --force-resolved-versions
./setup.sh
./benchmark.sh --local-submit
```

If local correctness fails, check the non-M5 near-tie caveat above before
assuming a regression; if performance improves but correctness is fragile,
prefer a more conservative optimization. The official benchmark will not rank
a submission that fails the hidden gates.

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

### Serial Non-Speculative Track Rules (default)

This is the default Laguna XS 2.1 serial track, `laguna-xs-2.1-serial-v2`,
registered by `benchmark.json`. Each model
invocation may compute logits and KV rows only for tokens supplied in that
invocation, and must advance logical and physical KV position by exactly the
supplied input length. A one-token decode request therefore advances exactly
one position and leaves no pending future token, logits, or KV state for a
later request.

Prompt-lookup decoding; n-gram, suffix, or token-history drafting;
same-target lookahead; and any other selection or evaluation of an unsupplied
future token are excluded. So are two-, three-, or more-row target-model paths
used to verify a draft from a one-token request, cross-request future-logit/KV
buffers, deferred cache rows, and commit, rollback, recommit, or discard
markers for those rows. Generic, bit-exact, or production-useful
implementations are still excluded under this track. Pre-hello or
initialization warmup of an excluded speculative pipeline is also excluded.

Ordinary within-request KV reuse, current-token-only decode, and
input-independent weight, dequantization, kernel, mask, or RoPE caches remain
allowed. Multi-row kernels remain allowed when every row is backed by a token
supplied in that same invocation, such as prefill. Organizer-provided MTP or
other speculative decoding would require a separate explicit track with a
trusted variable-length block protocol, correctness contract, and score; no
such track currently exists.
