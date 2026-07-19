# mlxfast — Gemma 4 31B-IT MTP

A benchmark arena for compute-optimal LLM inference on Apple Silicon.
Run Gemma 4 31B-IT with Google's matched trained assistant, verify every
returned token against the target model, and make block decode faster.

See [TASK.md](TASK.md) for the full problem statement, scoring formula, and
approach space.

## Quickstart

```bash
# Build the Swift/Metal runtime without downloading the legacy base checkpoint.
MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh

# Download and verify the pinned 31B-IT target and trained assistant.
./setup-mtp.sh

# Directional local MTP-vs-serial control (64 decoded tokens).
./benchmark-mtp.sh --local-iterate

# Longer local pre-submit signal (128 decoded tokens).
./benchmark-mtp.sh --local-submit
```

The local wrapper generates a temporary serial oracle from the current
candidate, then requires the trained-assistant path to match it exactly. That
is useful for parity and speed direction but is not an official correctness
oracle. Ranked runs use organizer-pinned M5 goldens from the frozen 31B-IT
target; one divergent token fails the run.

### Legacy serial local setup

The archived serial track (`benchmark.serial.json` and
`serial-benchmark.yml`) still uses the following base checkpoint and local
commands. Full model setup needs a moderate local SSD. The reference
checkpoint is
`mlx-community/gemma-4-31b-4bit`, with 4 safetensors shards totaling about
18.4 GB. `setup.sh` downloads the checkpoint from the Darkbloom R2 mirror
(`https://ds4.darkbloom.ai/gemma-4-31b-4bit`) by default, with up to 3 shard
downloads in parallel (`MLXFAST_REFERENCE_DOWNLOAD_JOBS`), into a shared
Hugging Face-style cache under
`~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-4bit/snapshots/main/`
(in `$HOME` by default so parallel clones reuse one checkpoint).
It verifies cached files against `fixtures/reference_gemma_4_31b_4bit.sha256`
and redownloads only files that are missing, truncated, or hash-mismatched. A
compatibility symlink is created at `reference_weights/gemma-4-31b-4bit`
for older commands, but current setup and CI pass the canonical cache directory
to transform explicitly. The downloader uses resumable `curl` requests, prints
numbered shard progress with elapsed time, and checks for at least 40 GiB free
by default. After a full SHA-256 verification, setup writes
`.mlxfast-reference-cache.lock` next to the checkpoint; later setup runs use
cheap size/mtime checks against that lock and skip the full hash pass
when the cache is unchanged. Use
`MLXFAST_REFERENCE_CACHE_DIR=/Volumes/ssd/hf-cache/.../snapshots/main` or
`MLXFAST_REFERENCE_DIR=/Volumes/ssd/gemma-4-31b-4bit` to point at a larger
volume, or `MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh` when the checkpoint will
be supplied separately. If you use a custom cache path, either copy the exact
transform command printed by `setup.sh` or set `MLXFAST_REFERENCE_DIR` before
running `transform` or `benchmark.sh`. The Swift CLI also honors
`MLXFAST_REFERENCE_DIR`, `MLXFAST_WEIGHTS_PATH`,
`MLXFAST_CORRECTNESS_GOLDEN_PATH`, and `MLXFAST_SCORE_PATH` as defaults;
explicit CLI flags take precedence. For `benchmark.sh`, use those `MLXFAST_*`
environment variables for path overrides; pass `--weights`, `--golden`, and
`--score-path` only to `.build/release/mlxfast-swift benchmark` directly. Set
`MLXFAST_REFERENCE_BASE_URL` to use
another HTTP checkpoint prefix (for example
`https://huggingface.co/mlx-community/gemma-4-31b-4bit/resolve/main` to fetch
from Hugging Face instead of the R2 mirror; both serve the same
manifest-pinned files), and `MLXFAST_REFERENCE_AUTH_HEADER` to pass an auth
header to a private checkpoint endpoint. Run `./setup.sh --help`
for the full local setup knobs.

> **Correctness fixtures are M5-generated.** The archived serial track's
> checked-in goldens can hit near-tie argmax differences on other Apple
> Silicon generations; the ranked M5 result is authoritative.

### Ranked MTP workflow

Yukon dispatches `.github/workflows/benchmark.yml`, the default MTP workflow.
It validates the pinned 31B-IT target and organizer-owned QAT assistant,
builds and transforms submitted code in the sandbox, and replays a hidden
512-token correctness golden through trained-assistant block decode. The
trusted parent accepts only a target-confirmed prefix and compares every
returned token with its serial K=1 oracle.

Timing runs last. At least three alternating candidate/reference pairs run
behind the fixed 40C thermal gate. The published decode-only score is:

```text
score = mean(serial_K1_seconds_per_token) / mean(MTP_seconds_per_token)
```

The hard floor is `1.0`; parity failure, invalid cache rollback, throttling,
or invalid telemetry fails the run. `score.json` carries
`track_id=gemma4-31b-it-mtp-v1`. See
[`docs/experimental-mtp-track.md`](docs/experimental-mtp-track.md) for the
protocol and [`docs/private-benchmark-security.md`](docs/private-benchmark-security.md)
for isolation details.

## Why this challenge exists

Gemma 4 31B is a dense (non-MoE) text+vision model; only the text tower is in
scope here. In 4-bit the text tower is about 17 GB, small enough to load
entirely into unified memory once at process startup on the official runner
(a self-hosted Apple M5 Max with 128 GB of unified memory, runner label
`m5-bench`). MTP keeps the target, assistant, target KV, shared K/V, and
verification activations resident together; 64 GiB is the practical local
minimum. The archived serial track can still run on roughly 36 GiB. There is
no weight streaming: the target is RAM-resident before scored decode.

The optimized runtime also has alternate combined/co-tiled weight layouts that
are profitable on the 128 GB ranked machine but would duplicate roughly
14.5 GiB of active model data on a 36 GB Mac. At process startup, machines
with less than 64 GiB therefore select a low-memory profile automatically:
large persistent co-tiles and compiled decode are disabled, the MLX allocator
cache is capped at 6 GiB, and free warmup buffers are released before the
worker begins serving requests. The profile announces itself on stderr, never
overrides feature flags you exported explicitly, and can be forced either way
with `DARKBLOOM_STARTUP_MEMORY_PROFILE=full|low`. This changes local speed
only; the 128 GB ranked runner keeps the full optimized profile.

That does not mean there is nothing left to optimize. Attention alternates
five sliding-window layers (1024-token window, GQA with 16 KV heads) with one
full-attention layer per block (GQA with 4 KV heads, a shared K/V projection,
and a partial-rotation "proportional" RoPE), every projection is affine 4-bit
quantized (group size 64), and the whole forward pass runs through MLX's
kernel scheduler on every decode step. Kernel selection, quantized matmul
dispatch, KV-cache handling, attention masking, and MLX graph/scheduling
overhead are all optimisation targets — and so are the vendored MLX Metal
kernels themselves, which are now part of the editable surface (see "The
modifiable surface" below). The generated `weights/` tree is
expected to stay small: it is a runtime artifact overlay on top of the frozen
reference checkpoint (a straight text-tensor subset plus a runtime-authored
`config.json`), not a second full model copy. Submissions may change the
Swift transform, the Swift runtime, and the vendored Gemma 4 model and
kernel sources, as long as the generated runnable artifacts pass the hidden
correctness and benchmark checks.

## The modifiable surface

Unlike typical inference benchmarks, the entire model execution pipeline is
in scope — including the vendored Gemma 4 model code and the MLX Metal
kernels it runs on. The authoritative list is `editablePaths` in
`benchmark.json` (currently 66 entries), in four groups:

| Path | What it controls |
|---|---|
| `Sources/MLXFastModel/` | Gemma 4 31B-IT target runtime, trained-assistant block session, exact-pair verification, MLX Swift array bridge, attention, and KV-cache logic. **Primary target.** |
| `Sources/MLXFastTransform/` | Offline target transform into benchmark-ready `mtp-weights/`. |
| `Vendor/mlx-swift-lm/Libraries/` (listed files) | The vendored Gemma 4 model implementation (`MLXLLM/Models/Gemma4Text.swift`, `Gemma4MTP.swift`, `Gemma4MTPTarget.swift`) plus the `MLXLMCommon` plumbing it uses directly (KV caches, RoPE utilities/application, compiled decode, evaluation). |
| `Vendor/mlx-swift/Source/Cmlx/` (listed files) | The MLX Metal kernels Gemma 4 dispatches — SDPA/`sdpa_vector`, affine-quantized matmul (incl. `_nax`), `steel/gemm`, `gemv`, `rope`, `rms_norm`, `softmax`, `copy`, elementwise, `arg_reduce`, gather indexing — as AOT `.metal`/`.h` sources and their JIT `mlx-generated/*.cpp` twins. |

Two build forms matter for kernel edits, because the vendored MLX package
builds in JIT mode. Families with an `mlx-generated/*.cpp` twin (quantized,
steel/gemm, gemv, softmax, copy, elementwise, gather) are compiled at
runtime from the C++ source strings embedded in those files — the twin is
the runtime-effective source, so edit it (and keep the readable
`.metal`/`.h` pair in sync). RoPE, RMSNorm, the SDPA vector kernel, and
`arg_reduce` load ahead-of-time from `mlx.metallib`, which
`tools/build-mlx-metallib.sh` (run by `./setup.sh`) compiles from the
vendored `.metal` sources — rerun it after editing those. `_nax` names are
the M5-generation kernel variants the ranked runner selects. After a kernel
edit: `swift build -c release` (plus the metallib rebuild for AOT edits),
then `./benchmark-mtp.sh --local-iterate`. Prioritize kernels reached by
assistant drafting, target block verification, and exact-pair decode.

Participant model and kernel code — `MLXFastModel` plus the vendored forks
— builds into the sandboxed `mlxfast-runtime-worker` binary. The trusted
`mlxfast-swift` binary owns correctness, scoring, timing, and provenance,
links no MLX, model, or kernel code, and drives the worker over a JSON
protocol. `Package.swift`/`Package.resolved` and the dependency graph are
frozen, and the rest of the vendored forks (other model families, shared
factory/tokenizer plumbing, kernels Gemma 4 does not dispatch such as
`steel/attn`) stay non-editable. Kernel changes are bound by the same
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
`preSubmitCommand` (`./benchmark-mtp.sh --local-submit`), and no local run blocks
the upload — the official M5 run is the gate. Run
`./benchmark-mtp.sh --local-submit` yourself before submitting: it compares
trained-assistant block decode with the candidate's serial K=1 control, writes
`score.json`, and catches obvious parity or speed regressions before they
spend official runner time.

## Local Commands

Use these MTP modes for local development:

| Command | Purpose | What it checks | Output |
|---|---|---|---|
| `./benchmark-mtp.sh --local-iterate` | Fast directional edit loop. | Candidate-local serial oracle, 64 exact tokens, trained assistant, exact-pair verification. | `score.json` with local MTP speedup. |
| `./benchmark-mtp.sh --local-submit` | Longer pre-submit signal. | Same checks over 128 exact tokens. | `score.json` with local MTP speedup. |

Both modes transform the pinned IT target, generate a temporary oracle from
the current candidate, run serial K=1 and trained-assistant exact-pair decode,
and publish their ratio. The generated oracle deliberately cannot establish
official target fidelity; only the hidden M5 golden can do that. The archived
serial local modes remain available through `benchmark.serial.json` and
`./benchmark.sh`.

## Scoring

```text
score = mean(serial_K1_seconds_per_token) / mean(MTP_seconds_per_token)
```

Higher is better. The denominator is the pinned serial K=1 target baseline
measured on the same M5 in the same session, behind the same thermal gates.
The score is a ratio of means over accepted alternating pairs; at least three
pairs must survive parity, thermal, and telemetry checks. The component floor
is `score >= 1.0`.

The trusted parent owns the timer and divides wall time by its configured
decode count. Seed prefill is charged to decode. Each block may return at most
four tokens, and every token must match the parent-owned serial oracle before
it is committed. The whole target is RAM-resident with no weight streaming.
`bandwidth_gb_per_token=0`. RAM and phase-timing metrics are still reported
for operator review and future guardrails; they are not primary score factors.
Correctness is a hard gate. See TASK.md for the full correctness specification.
The ranked correctness replay runs before timing in a fresh worker. The
benchmark golden is distinct from the correctness golden and is installed
only by the trusted measurement wrapper. The score payload publishes aggregate
pair counts, parity status, serial and MTP seconds/token means, speedup
statistics, and the transformed-weights digest; prompts, token IDs, and
per-case hidden outputs remain private.

## Architecture

```
Sources/
  MLXFastCLI/                trusted CLI entrypoint (mlxfast-swift)
  MLXFastCore/               score.json, golden cases, shared contracts
  MLXFastTransform/          editable Swift offline weight transform
  MLXFastModel/              editable Gemma 4 31B 4-bit Swift runtime
  MLXFastTrustedHarness/     trusted correctness, golden, and benchmark runner
  MLXFastHarness/            worker-side runtime support (builds into the worker)
  MLXFastRuntimeWorkerCLI/   sandboxed participant worker (mlxfast-runtime-worker)
Vendor/
  mlx-swift/                 pinned MLX fork; the listed kernel sources are editable
  mlx-swift-lm/              pinned mlx-swift-lm fork; the Gemma 4 model files are editable
weights/                     transformed weights (harness loads from here)
  config.json                 runtime-authored text-tower config
  model.safetensors.index.json
~/.cache/huggingface/hub/... canonical frozen 4-bit reference checkpoint cache
reference_weights/...        compatibility symlink to the reference cache
correctness_prompts/         public correctness prompt and checked-in Gemma-generated golden
correctness_golden.json      hidden benchmark correctness cases
score.json                   written after each benchmark run
```

The runtime loads every text-tower tensor from `weights/` into unified memory
once at process init and keeps them resident for the process lifetime; there
is no streaming path and no dependency on the frozen reference checkpoint at
runtime (only the offline transform reads the reference checkpoint).

The standard preflight/benchmark path enforces a default 25 GiB cap on the
generated `weights/` tree before correctness or timing runs (the text tower is
about 17 GB, comfortably inside that cap). Change it with
`MLXFAST_MAX_WEIGHTS_BYTES`; use `0`, `none`, or `unlimited` only for organizer
debugging. For stricter organizer-side provenance, set
`MLXFAST_VERIFY_TRANSFORM=1` when running `benchmark.sh`. That re-runs the
submitted Swift transform into a clean temporary directory and fails unless
`weights/` is byte-equal to that fresh run. This checks determinism and stale
files; it does not require the baseline `weights/` layout. `verify-transform`
uses the same default cap and can also be changed with
`mlxfast-swift verify-transform --max-bytes N`.

### Legacy serial fixtures

The archived serial track's public correctness prompt and golden live in
`correctness_prompts/`.
These fixtures are generated on the ranked M5 hardware against the Gemma 4
31B 4-bit reference: the prompt text is tokenized with the Gemma tokenizer
(512 prompt tokens) and the expected tokens are greedy reference
continuations captured with `mlxfast-swift generate-golden` (256 tokens for
local-iterate, 1024 for local-submit). Private prompt manifests and hidden
benchmark golden files are not committed or generated by the benchmark
workflow. In private benchmark CI, the normal path downloads the precomputed
`correctness_prompts/golden_prompt_benchmark_transcription_gate_english_512_256-gemma.json`
object from R2, then downloads
`correctness_prompts/gpqa_reference_cases-gemma.json` and merges it into the
local golden as 5 hidden GPQA behavior checks. Generate
final hidden benchmark goldens outside the public repository and upload the
resulting file to the protected private R2 path. `serial-benchmark.yml` keeps
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

This repository's harness code is licensed per [LICENSE](LICENSE). The Gemma 4
models the harness downloads and benchmarks (Gemma 4 31B, Gemma 4 31B-IT, and
the Gemma 4 31B-IT assistant, © Google DeepMind, plus the mlx-community 4-bit
conversions) are Apache-2.0 with usage terms at
<https://ai.google.dev/gemma/docs/gemma_4_license>; no model weights are
distributed in this repository. Full third-party attribution — models,
linked Swift packages, and the Apache-2.0 text — is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Requirements

- Apple Silicon Mac, 36 GB+ unified memory (enough to hold the ~17 GB text
  tower plus KV cache and buffers; the ranked runner is a single self-hosted
  Apple M5 Max with 128 GB, so local timings — and, on non-M5 machines,
  near-tie greedy tokens — are directional only)
- macOS Sequoia or later
- Swift 6 through Xcode or Xcode Command Line Tools
- Xcode Metal Toolchain for `mlx.metallib`; `./setup.sh` tries
  `xcodebuild -downloadComponent MetalToolchain`, but users with only Command
  Line Tools may need full Xcode installed, opened once, and licensed with
  `sudo xcodebuild -license accept`
- CMake, installed by `./setup.sh` via Homebrew when missing and used by `tools/build-mlx-metallib.sh` to build `mlx.metallib`
