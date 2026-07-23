# mlxfast — Poolside Laguna XS 2.1

A benchmark arena for compute-optimal LLM inference on Apple Silicon.
Run Poolside Laguna XS 2.1 NVFP4, keep its exact greedy output, and make
prefill and serial decode faster.

See [TASK.md](TASK.md) for the full problem statement, scoring formula, and
approach space. (The former MTP speculative-decoding track is retired; the
serial track described here is the default and only ranked track.)

The Poolside checkpoint is registered as the new contract
`laguna-xs-2.1-serial-v2`. The earlier `laguna-xs-2.1-serial-v1` contract used
the materially different mlx-community affine checkpoint and remains visible
but frozen. V2 starts with a fresh leaderboard frontier; v1 submissions and
scores are never silently compared with or migrated into v2.

## Quickstart

```bash
# Build the Swift/Metal runtime and download the reference checkpoint.
./setup.sh

# Fast local edit-loop signal (correctness smoke + local timing estimate).
./benchmark.sh --local-iterate

# Longer local pre-submit signal.
./benchmark.sh --local-submit
```

Full model setup needs a moderate local SSD. The reference checkpoint is
`poolside/Laguna-XS-2.1-NVFP4-mlx` at revision
`841778bda563a36104dd521e37d99218e46f4f25`, with 5 safetensors shards
and 21,568,905,520 bytes across all 13 files. `setup.sh` downloads it from
the public organizer R2 mirror
(`https://ds4.darkbloom.ai/laguna-xs-2.1-nvfp4-mlx`) by default, with the
exact pinned Hugging Face revision as fallback and up to 3 shard
downloads in parallel (`MLXFAST_REFERENCE_DOWNLOAD_JOBS`), into a shared
Hugging Face-style cache under
`~/.cache/huggingface/hub/models--poolside--Laguna-XS-2.1-NVFP4-mlx/snapshots/841778bda563a36104dd521e37d99218e46f4f25/`
(in `$HOME` by default so parallel clones reuse one checkpoint).
It verifies cached files against
`fixtures/reference_laguna_xs_2_1_nvfp4_mlx.sha256`
and redownloads only files that are missing, truncated, or hash-mismatched.
A compatibility symlink is created at
`reference_weights/laguna-xs-2.1-nvfp4-mlx`
for older commands, but current setup and CI pass the canonical cache directory
to transform explicitly. The downloader uses resumable `curl` requests, prints
numbered shard progress with elapsed time, and checks for at least 40 GiB free
by default. After a full SHA-256 verification, setup writes
`.mlxfast-reference-cache.lock` next to the checkpoint; later setup runs use
cheap size/mtime checks against that lock and skip the full hash pass
when the cache is unchanged. Use
`MLXFAST_REFERENCE_CACHE_DIR=/Volumes/ssd/hf-cache/.../snapshots/<revision>` or
`MLXFAST_REFERENCE_DIR=/Volumes/ssd/laguna-xs-2.1-nvfp4-mlx` to point at a larger
volume, or `MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh` when the checkpoint will
be supplied separately. If you use a custom cache path, either copy the exact
transform command printed by `setup.sh` or set `MLXFAST_REFERENCE_DIR` before
running `transform` or `benchmark.sh`. The Swift CLI also honors
`MLXFAST_REFERENCE_DIR`, `MLXFAST_WEIGHTS_PATH`,
`MLXFAST_CORRECTNESS_GOLDEN_PATH`, and `MLXFAST_SCORE_PATH` as defaults;
explicit CLI flags take precedence. For `benchmark.sh`, use those `MLXFAST_*`
environment variables for path overrides; pass `--weights`, `--golden`, and
`--score-path` only to `.build/release/mlxfast-swift benchmark` directly. Set
`MLXFAST_REFERENCE_BASE_URL` to use another HTTP checkpoint prefix serving
the same manifest-pinned Poolside files,
and `MLXFAST_REFERENCE_AUTH_HEADER` to pass an auth
header to a private checkpoint endpoint. Run `./setup.sh --help`
for the full local setup knobs.

> **Correctness fixtures are M5-generated.** The checked-in goldens can hit
> near-tie argmax differences on other Apple Silicon generations; the ranked
> M5 result is authoritative.

### Ranked workflow

Yukon dispatches `.github/workflows/benchmark.yml`, the serial ranked
pipeline. Unlike local `setup.sh`, ranked M5 jobs never download a checkpoint:
they verify the pre-provisioned Poolside cache against the pinned manifest,
build and transform submitted code in the sandbox, run
the public drift tripwire, then the hidden teacher-forced base case plus the
anchor/free-run/behavior/GPQA gates and the semantic GPQA judge.

Timing runs last, behind the fixed 40C thermal gate: the trusted on-box
measure-job runs the pinned baseline tree and the candidate back to back on
the same silicon over a hidden 512-token evaluation prompt, and the paired
ratio cancels host drift. The published score is:

```text
score = decode_speedup^0.75 * prefill_speedup^0.25
```

Both speedup floors are `0.95`, hard; a token mismatch, throttled sample, or
invalid telemetry fails the run. `score.json` publishes the paired speedups
and floor verdicts. See
[`docs/private-benchmark-security.md`](docs/private-benchmark-security.md)
for isolation details and
[`docs/benchmark-window-freeze.md`](docs/benchmark-window-freeze.md) for the
frozen timed window.

## Why this challenge exists

Poolside Laguna XS 2.1 is a fine-grained MoE text model (256 routed experts
plus one shared expert per sparse layer, 8 experts per token, per-head
gating). The pinned Poolside export is already text-only (`model.*` / `lm_head.*`
tensors). Its sparse routed/shared expert projections are NVFP4 4-bit
group-16; attention, embeddings, the untied lm_head, layer-0 dense MLP, and
routers remain BF16. The 13-file checkpoint is exactly 21,568,905,520 bytes,
small enough to load
entirely into unified memory once at process startup on the official runner
(a self-hosted Apple M5 Max with 128 GB of unified memory, runner label
`m5-bench`). There is no weight streaming: the model is RAM-resident before
scored prefill or decode.

The optimized runtime also has alternate combined/co-tiled weight layouts that
are profitable on the 128 GB ranked machine but would duplicate roughly
14.5 GiB of active model data on a 40 GiB Mac. At process startup, machines
with less than 64 GiB therefore select a low-memory profile automatically:
large persistent co-tiles and compiled decode are disabled, the MLX allocator
cache is capped at 6 GiB, and free warmup buffers are released before the
worker begins serving requests. The profile announces itself on stderr, never
overrides feature flags you exported explicitly, and can be forced either way
with `DARKBLOOM_STARTUP_MEMORY_PROFILE=full|low`. This changes local speed
only; the 128 GB ranked runner keeps the full optimized profile.

That does not mean there is nothing left to optimize. Attention alternates
three sliding-window layers (512-token window, 64 heads) with one
full-attention layer per block of four (48 heads, YaRN rotary with a 0.5
partial-rotary factor; sliding layers use plain RoPE at theta 10000). All
layers are GQA with 8 KV heads at head_dim 128. Layer 0 has a dense MLP
(intermediate 8192); every other layer routes tokens through 8 of 256
experts plus a shared expert (per-head gating, MoE intermediate 512).
Only routed/shared expert projections are NVFP4-quantized; the whole forward
pass runs through MLX's kernel scheduler on every decode step. Kernel selection,
quantized matmul
dispatch, MoE expert gathering, KV-cache handling, attention masking, and MLX
graph/scheduling
overhead are all optimisation targets — and so are the vendored MLX Metal
kernels themselves, which are part of the editable surface (see "The
modifiable surface" below). The generated `weights/` tree is
expected to stay small: it is a runtime artifact overlay on top of the frozen
reference checkpoint (a straight text-tensor subset plus a runtime-authored
`config.json`), not a second full model copy. Submissions may change the
Swift transform, the Swift runtime, and the vendored Laguna model and
kernel sources, as long as the generated runnable artifacts pass the hidden
correctness and benchmark checks.

## The modifiable surface

Unlike typical inference benchmarks, the entire model execution pipeline is
in scope — including the vendored Laguna model code and the MLX Metal
kernels it runs on. The authoritative list is `editablePaths` in
`benchmark.json` (currently 91 entries), in four groups:

| Path | What it controls |
|---|---|
| `Sources/MLXFastModel/` | Laguna XS 2.1 runtime: weight loading, attention, MoE MLP, KV caches, prefill/decode execution. **Primary target.** |
| `Sources/MLXFastTransform/` | Offline reference-checkpoint transform into benchmark-ready `weights/`. |
| `Vendor/mlx-swift-lm/Libraries/` (listed files) | The vendored Laguna model implementation (`MLXLLM/Models/Laguna.swift`) plus the `MLXLMCommon` plumbing it uses directly (MoE/attention dispatch helpers, KV caches, RoPE utilities/application, compiled decode, evaluation). |
| `Vendor/mlx-swift/Source/Cmlx/` (listed files) | The MLX Metal kernels Laguna dispatches — SDPA (`steel/attn`, `sdpa_vector`), NVFP4 `fp_quantized` matmul plus shared quantized dispatch (incl. `_nax`), MoE gather GEMM (`steel_gemm_gather*`), `steel/gemm`, `gemv`, `rope`, `rms_norm`, `softmax`, `sort`, `reduce`, `copy`, elementwise, `arg_reduce`, gather indexing — as AOT `.metal`/`.h` sources and their JIT `mlx-generated/*.cpp` twins. |

Two build forms matter for kernel edits, because the vendored MLX package
builds in JIT mode. Families with an `mlx-generated/*.cpp` twin (quantized
incl. `fp_quantized`, steel/gemm incl. the gather GEMM, steel/attn, gemv,
softmax, sort, reduce, copy, elementwise, gather) are compiled at
runtime from the C++ source strings embedded in those files — the twin is
the runtime-effective source, so edit it (and keep the readable
`.metal`/`.h` pair in sync). RoPE, RMSNorm, the SDPA vector kernel, and
`arg_reduce` load ahead-of-time from `mlx.metallib`, which
`tools/build-mlx-metallib.sh` (run by `./setup.sh`) compiles from the
vendored `.metal` sources — rerun it after editing those. `_nax` names are
the M5-generation kernel variants the ranked runner selects. After a kernel
edit: `swift build -c release` (plus the metallib rebuild for AOT edits),
then `./benchmark.sh --local-iterate`. Prioritize kernels reached by the
timed prefill and decode phases.

Participant model and kernel code — `MLXFastModel` plus the vendored forks
— builds into the sandboxed `mlxfast-runtime-worker` binary. The trusted
`mlxfast-swift` binary owns correctness, scoring, timing, and provenance,
links no MLX, model, or kernel code, and drives the worker over a JSON
protocol. `Package.swift`/`Package.resolved` and the dependency graph are
frozen, and the rest of the vendored forks (other model families, shared
factory/tokenizer plumbing, and kernels Laguna does not dispatch) stay
non-editable. Kernel changes are bound by the same
hidden correctness gates as model changes: keep them prompt-independent and
model-general, and be conservative with numeric reassociation, which can
flip near-tie greedy argmaxes on the M5.

The repository is Swift-only (no Python): setup, transform, correctness,
and benchmark all run through the Swift package, plus the
`tools/build-mlx-metallib.sh` step for the vendored AOT Metal sources.

Submissions are made with the **Yukon CLI (`mlxfast`)**, a separate tool that
manages your account and uploads across all Yukon benchmarks. The
`mlxfast-swift` binary runs the benchmark domain only (transform, correctness,
benchmark, preflight, verify-transform) and no longer logs in or uploads.

The `mlxfast` CLI is installed by the external Yukon installer from your
challenge onboarding instructions, not by this repository or `./setup.sh`. If
`mlxfast` is not found after installing it, the installer's bin directory
(typically `~/.local/bin`) is not on your PATH; activate it with:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

`./setup.sh` checks for `mlxfast` at the end of setup and prints this same
remediation (with the detected directory) when the CLI is installed but not
activated on PATH. For the current shell only, the first line below exposes
the CLI's default install directory without editing your shell rc:

```bash
export PATH="${HOME}/.local/bin:${PATH}"
mlxfast login <api-key> --api https://yukon-api.fly.dev
mlxfast clone <benchmark-id-or-name>     # fresh checkout; an existing repo auto-links by its git remote
mlxfast submit --model "Claude Opus 4.8" \
  --note "Optimized quantized matmul dispatch for the sliding-window layers."
mlxfast submissions
```

`mlxfast submit` reads `benchmark.json` and uploads only the paths listed in
`editablePaths` as `submission.tar.gz`, POSTed to Yukon with
`Authorization: Bearer <api-key>` and an idempotency key. Generated `weights/`,
reference checkpoints, golden files, and local scores live outside
`editablePaths` and are never uploaded; the backend re-enforces the editable
surface server-side after upload. `--model` is required and is recorded for the
leaderboard. `MLXFAST_API_URL` / `MLXFAST_API_TOKEN` (or the `YUKON_*`
equivalents) configure the endpoint and token for scripted runs.
`mlxfast submit` uploads directly: it does not run the contract
`preSubmitCommand` (`./benchmark.sh --local-submit`), and no local run blocks
the upload — the official M5 run is the gate. Run
`./benchmark.sh --local-submit` yourself before submitting: it runs the
public correctness fixture and a longer local timing pass, writes
`score.json`, and catches obvious correctness or speed regressions before
they spend official runner time.

## Local Commands

Use these modes for local development:

| Command | Purpose | What it checks | Output |
|---|---|---|---|
| `./benchmark.sh --local-iterate` | Fast directional edit loop. | Public-fixture correctness (64 teacher-forced steps) plus a short local timing pass. | `score.json` with a local estimated score. |
| `./benchmark.sh --local-submit` | Longer pre-submit signal. | Same correctness over a longer 1023-step decode timing pass. | `score.json` with a local estimated score. |

Both modes transform the reference checkpoint if needed, run the checked-in
public correctness fixture, and time prefill and decode locally. Local scores
are estimates for direction only; the official paired score exists only on
the ranked M5 runner, measured against the pinned on-box baseline.

## Scoring

```text
score = decode_speedup^0.75 * prefill_speedup^0.25
```

Higher is better. Each speedup is the pinned baseline's seconds/token divided
by the candidate's for that phase, measured on the same M5 in the same
session behind the same fixed 40C thermal gate and telemetry acceptance. The
timed window is frozen (512-token prefill prompt; 512-token decode seed with
128 teacher-forced decode steps — see
`docs/benchmark-window-freeze.md`), and the component floors are hard:

```text
decode_speedup >= 0.95
prefill_speedup >= 0.95
```

Correctness is a hard gate: the full 64-step teacher-forced base case, the
hidden anchor/free-run/behavior/GPQA gates, GPQA TTFT, the semantic GPQA
judge, and the public drift tripwire must all pass, or the score is null.
The whole model is RAM-resident with no weight streaming
(`bandwidth_gb_per_token=0`). RAM and phase-timing metrics are still reported
for operator review and future guardrails; they are not primary score factors.
See TASK.md for the full correctness specification.

## Architecture

```
Sources/
  MLXFastCLI/                trusted CLI entrypoint (mlxfast-swift)
  MLXFastCore/               score.json, golden cases, shared contracts
  MLXFastTransform/          editable Swift offline weight transform
  MLXFastModel/              editable Poolside Laguna XS 2.1 NVFP4 Swift runtime
  MLXFastTrustedHarness/     trusted correctness, golden, and benchmark runner
  MLXFastHarness/            worker-side runtime support (builds into the worker)
  MLXFastRuntimeWorkerCLI/   sandboxed participant worker (mlxfast-runtime-worker)
Vendor/
  mlx-swift/                 pinned MLX fork; the listed kernel sources are editable
  mlx-swift-lm/              pinned mlx-swift-lm fork; the Laguna model files are editable
weights/                     transformed weights (harness loads from here)
  config.json                 runtime-authored text-tower config
  model.safetensors.index.json
~/.cache/huggingface/hub/... canonical frozen Poolside NVFP4 reference cache
reference_weights/...        compatibility symlink to the reference cache
correctness_prompts/         Laguna-tokenized public correctness prompt and checked-in golden
correctness_golden.json      hidden benchmark correctness cases
score.json                   written after each benchmark run
```

The runtime loads every text-tower tensor from `weights/` into unified memory
once at process init and keeps them resident for the process lifetime; there
is no streaming path and no dependency on the frozen reference checkpoint at
runtime (only the offline transform reads the reference checkpoint).

The standard preflight/benchmark path enforces a default 25 GiB cap on the
generated `weights/` tree before correctness or timing runs (the text tower is
about 21.6 GB, inside that cap). Change it with
`MLXFAST_MAX_WEIGHTS_BYTES`; use `0`, `none`, or `unlimited` only for organizer
debugging. For stricter organizer-side provenance, set
`MLXFAST_VERIFY_TRANSFORM=1` when running `benchmark.sh`. That re-runs the
submitted Swift transform into a clean temporary directory and fails unless
`weights/` is byte-equal to that fresh run. This checks determinism and stale
files; it does not require the baseline `weights/` layout. `verify-transform`
uses the same default cap and can also be changed with
`mlxfast-swift verify-transform --max-bytes N`.

### Correctness fixtures

The public correctness prompt and golden live in `correctness_prompts/`.
These fixtures are generated on the ranked M5 hardware against the Poolside
Laguna NVFP4 reference: the prompt text is tokenized with the Laguna tokenizer
(512 prompt tokens) and the expected tokens are greedy reference
continuations captured with `mlxfast-swift generate-golden` (256 tokens for
local-iterate, 1024 for local-submit). Private prompt manifests and hidden
benchmark golden files are not committed or generated by the benchmark
workflow. In private benchmark CI, the normal path downloads precomputed,
content-addressed objects below
`correctness_prompts/laguna-xs-2.1-serial-v2/`: the hidden golden,
GPQA reference, and timed prompt each embed their SHA-256 in the object key
and are independently pinned by SHA-256 and byte count. The workflow merges
the GPQA reference into the local golden as 5 hidden behavior checks. Generate
final hidden benchmark goldens outside the public repository and upload the
resulting files to those protected private R2 paths. `benchmark.yml` keeps
raw hidden material in a runner-only private directory, not the repository
workspace, scrubs every hidden byte out of the bench workspace before the
timed measurement, and uploads only hash and byte-count sidecars. The
semantic GPQA answer and judge result files are also kept under the private
runner directory and are not uploaded.

The older participant-facing Swift `make-golden` generator has been removed
from the public harness; the last commit on this branch containing it is
`bcc9438fabf95a9b371d5749dd64f2f5ccc60fd5`. Golden generation is operator work
(the `generate-golden` capture tool described above): benchmark CI consumes
precomputed, pin-verified correctness fixtures and never regenerates them,
while the ranked timed run's benchmark oracle is self-generated per submitted
binary by the trusted on-box measure-job (see
[`docs/private-benchmark-security.md`](docs/private-benchmark-security.md)).

Each base correctness prompt must contain exactly 512 token IDs. The benchmark
prompt must contain at least 512 token IDs. The precomputed golden file stores
exact expected tokens for each 512-token correctness prompt continuation, the
512-token prefill check, the 512-token decode seed, and at least 128 tokens for
the timed decode window. During correctness, the harness checks the first 64
public continuation positions by default, plus hidden
behavior gates in official benchmark runs. It checks those continuation
positions teacher-forced: after each accepted step it feeds the
golden previous token back into the model. This keeps the gate stable across
Apple GPU/software differences by preventing one earlier mismatch from
cascading into unrelated later-token failures. A token is accepted only when it
matches the expected token, except for a true top-logit tie within the tiny
`1e-6` logit tolerance used by the harness.

Private fixtures can also include a `correctness_gates` object with hidden
anchor logits, short free-run prefixes, and answer-token behavior checks.
Those gates are additive: public local correctness still works with the
checked-in fixture, while official benchmark fixtures can cover more adversarial
behavior without exposing prompt or answer data. Behavior checks compare
accepted answer prefixes against up to `max_new_tokens` generated tokens, which
lets hidden GPQA questions require only a one-letter answer while tolerating
tokenizer whitespace variants.

## License and attribution

This repository's harness code is licensed per [LICENSE](LICENSE). The
Poolside Laguna model the harness downloads and benchmarks
(Laguna XS 2.1 NVFP4, © Poolside) is licensed OpenMDW-1.1 with terms at
<https://huggingface.co/poolside/Laguna-XS-2.1>; no model weights are
distributed in this repository. Full third-party attribution — models,
linked Swift packages, and the Apache-2.0 text — is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Requirements

- Apple Silicon Mac with enough unified memory for the ~21.6 GB RAM-resident
  model plus KV cache and buffers (roughly 40 GiB practical minimum; the
  ranked runner is a single self-hosted Apple M5 Max with 128 GB, so local
  timings — and, on non-M5 machines, near-tie greedy tokens — are
  directional only)
- macOS Sequoia or later
- Swift 6 through Xcode or Xcode Command Line Tools
- Xcode Metal Toolchain for `mlx.metallib`; `./setup.sh` tries
  `xcodebuild -downloadComponent MetalToolchain`, but users with only Command
  Line Tools may need full Xcode installed, opened once, and licensed with
  `sudo xcodebuild -license accept`
- CMake, installed by `./setup.sh` via Homebrew when missing and used by `tools/build-mlx-metallib.sh` to build `mlx.metallib`
