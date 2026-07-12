# mlxfast — Gemma 4 31B 4-bit

A benchmark arena for compute-optimal LLM inference on Apple Silicon.
Run Gemma 4 31B (dense, 4-bit, text tower only) fully RAM-resident — and beat
the baseline score.

See [TASK.md](TASK.md) for the full problem statement, scoring formula, and
approach space.

## Quickstart

```bash
# Check local tools, build the Swift harness/MLX metallib, and fetch weights if needed
./setup.sh

# Transform the reference checkpoint into weights/. setup.sh prints this
# command with the exact reference path it used.
MLXFAST_OFFLINE_WRITABLE_PATHS="${PWD}/weights" \
  .github/scripts/run-offline.sh .build/release/mlxfast-swift transform \
  --reference "${HOME}/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-4bit/snapshots/main" \
  --output weights

# Run the checked-in public correctness gate.
.build/release/mlxfast-swift correctness --weights weights

# Fast edit-loop signal: uses the public 512-token correctness prompt, checks
# the prefill next token plus 16 teacher-forced decode tokens, writes
# score.local-iterate.json, and prints it to stdout.
./benchmark.sh --local-iterate

# Run the Darkbloom-compatible ranked benchmark entrypoint (official runner
# only: requires the organizer-supplied hidden correctness_golden.json).
# A bare ./benchmark.sh defaults to --local-iterate.
./benchmark.sh --official

# Local submit check used by Yukon before upload: runs the public 512-token
# prompt through a longer checked timing window, writes score.json with the
# estimated local score (never an official ranked score), and prints it to
# stdout.
./benchmark.sh --local-submit

# Or call the Swift CLI directly
.build/release/mlxfast-swift correctness --weights weights
.build/release/mlxfast-swift preflight
.build/release/mlxfast-swift benchmark --local-iterate
.build/release/mlxfast-swift benchmark --score-path score.json
.build/release/mlxfast-swift benchmark --local-submit --score-path score.json

# If required model artifacts are missing, the benchmark emits a valid failed
# score.json instead of a ranked score.
```

> **Correctness fixtures are M5-generated.** The checked-in prompt/golden
> fixtures under `correctness_prompts/` were generated on the ranked M5
> hardware against the Gemma 4 31B 4-bit reference implementation
> (`mlxfast-swift generate-golden`): the prompt text is tokenized with the
> Gemma tokenizer and the expected tokens are greedy continuations from the
> reference forward pass. On other Apple Silicon generations, near-tie greedy
> argmaxes can diverge, so the local public gate may fail for a correct
> build — local modes are directional; the ranked M5 runner is the source of
> truth. The private/hidden artifacts used by ranked runs were regenerated
> the same way by the organizer; see [TASK.md](TASK.md).

The benchmark writes `score.json` in the format consumed by Darkbloom.
`score.json` is a generated local output and is not tracked. Public
correctness-only workflow runs use the checked-in
`correctness_prompts/public_longcopy_gate_english_512_256.json` golden and
matching prompt text. Official benchmark runs use a hidden
`correctness_golden.json` supplied by the benchmark operator, or a harness path
set with `MLXFAST_CORRECTNESS_GOLDEN_PATH=/path/to/correctness_golden.json`.
`benchmark.sh` also writes `score.json.sha256` and `benchmark-integrity.json`,
which record the score file hash, golden hash, transformed `weights/` hash, and
transform source hash for run auditing.

Full model setup needs a moderate local SSD. The reference checkpoint is
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

For manual GitHub Actions benchmark runs, dispatch `benchmark.yml` on the
trusted repository workflow. Ranked runs execute as one serial job on the
single self-hosted M5 runner (label `m5-bench`); the reference checkpoint is
pre-provisioned on that box and hash-verified against the pinned manifest
every run, so the workflow never downloads it. The job order is: guard steps
(quarantine/trusted-context/editable-surface/static-review checks), build and
transform in the bench sandbox, the public correctness gate against the
checked-in `correctness_prompts/public_longcopy_gate_english_512_256.json`
fixture, the hidden correctness golden (fetched from R2 and pin-verified),
the full 64-step hidden base case plus anchor/free-run/behavior/GPQA gates in
one harness pass, the semantic GPQA judge, a scrub of all hidden material
from the bench workspace, a quiescence wait plus a fixed 40C GPU thermal
cool-gate, the paired timed measurement (pinned reference baseline then
candidate, back to back on the same silicon), and finally the timing overlay
that seals the score. Dispatching with `run_benchmark=false` (the default is
`true`) runs only the public-fixture correctness gate.
Full benchmark runs require the precomputed hidden R2 object
`correctness_prompts/golden_prompt_benchmark_transcription_gate_english_512_256-gemma.json`.
Full benchmark runs also require the private R2
`correctness_prompts/gpqa_reference_cases-gemma.json` object. The workflow
tokenizes 5 token-budget-valid hidden GPQA multiple-choice prompts locally and
attaches them as short-answer behavior gates before correctness runs. Each
private GPQA case must include precomputed reference
`accepted_token_sequences` or `accepted_responses`; GPQA answer keys are
metadata, not an exact-token oracle, and the benchmark workflow never
regenerates this reference object.
The official workflow checks the first generated GPQA answer token for each
case, using the stable prefix of any longer precomputed reference sequence.
During that hidden behavior correctness pass, it also records TTFT by timing
prompt prefill through the first greedy answer token. The uploaded score records
only aggregate TTFT counts and timings; generated first-token IDs, accepted
token IDs, prompts, and answers stay out of GitHub logs and artifacts.
The same correctness pass captures short hidden GPQA continuations and sends
only those private answer bundles to Claude for a semantic pass/fail judge. This
requires the `ORG_ANTHROPIC_API_KEY` repository secret. The score artifact records
only aggregate semantic counts and the judge model name; prompts, references,
candidate answers, and judge text stay in the private runner directory.
The private workflow keeps semantic GPQA wired as a required gate, with a
pass-count threshold calibrated against the unmodified Gemma 4 31B 4-bit
baseline rather than against better-than-baseline GPQA answer quality; see
`MLXFastConstants.semanticGPQAMinPassCount` for the current threshold and its
calibration provenance.
A full benchmark run fails without the hidden R2 objects and Anthropic key
configured; it will not use a committed prompt, committed golden, or Actions
cache fallback for ranked scoring. Final hidden goldens come from protected
storage. Private R2 golden downloads use the `R2_ACCESS_KEY_ID`,
`R2_BUCKET_ENDPOINT`, and `R2_SECRET_ACCESS_KEY` secrets. See
[`docs/private-benchmark-security.md`](docs/private-benchmark-security.md) for
the private prompt and artifact handling model.

## Why this challenge exists

Gemma 4 31B is a dense (non-MoE) text+vision model; only the text tower is in
scope here. In 4-bit the text tower is about 17 GB, small enough to load
entirely into unified memory once at process startup on the official runner
(a single self-hosted Apple M5 Max with 128 GB of unified memory, runner
label `m5-bench`) and on any local machine with roughly 36 GB or more. There
is no weight streaming of any kind: the whole model is RAM-resident before
the first scored forward pass runs.

That does not mean there is nothing left to optimize. Attention alternates
five sliding-window layers (1024-token window, GQA with 16 KV heads) with one
full-attention layer per block (GQA with 4 KV heads, a shared K/V projection,
and a partial-rotation "proportional" RoPE), every projection is affine 4-bit
quantized (group size 64), and the whole forward pass runs through MLX's
kernel scheduler on every decode step. Kernel selection, quantized matmul
dispatch, KV-cache handling, attention masking, and MLX graph/scheduling
overhead are all optimisation targets. The generated `weights/` tree is
expected to stay small: it is a runtime artifact overlay on top of the frozen
reference checkpoint (a straight text-tensor subset plus a runtime-authored
`config.json`), not a second full model copy. Submissions may change both the
Swift transform and Swift runtime, as long as the generated runnable artifacts
pass the hidden correctness and benchmark checks.

## The modifiable surface

Unlike typical inference benchmarks, the entire model execution pipeline is
in scope. Submissions should focus on the Swift targets listed in
`benchmark.json`:

| Path | What it controls |
|---|---|
| `Sources/MLXFastModel/` | Gemma 4 31B 4-bit runtime, MLX Swift array bridge, dense weight loading, attention/KV-cache/decode/prefill logic. **Primary target.** |
| `Sources/MLXFastTransform/` | Offline weight transform from frozen reference safetensors into benchmark-ready `weights/`. |

The repository is Swift-only: setup, transform, correctness, and benchmark all
run through the Swift package. Correctness, scoring, timing, and provenance are
trusted harness code outside `editablePaths`; only the model and transform
targets are contestant-editable.

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
Before uploading, Yukon runs the contract `preSubmitCommand`, which is
`./benchmark.sh --local-submit` for this benchmark. That local-submit pass is
the local submit gate: it uses the public/local oracle, writes and prints
`score.json`, and stops obviously broken or slower changes before they spend
official runner time.

## Local Commands

Use these two benchmark modes for local development:

| Command | Purpose | What it checks | Output |
|---|---|---|---|
| `./benchmark.sh --local-iterate` | Fast edit-loop signal, usually under 2 minutes after setup. | Public 512-token prompt, standalone prefill next-token check, decode seed-prefill check, and 16 teacher-forced decode checks. | `score.local-iterate.json` with the estimated local `score`. |
| `./benchmark.sh --local-submit` | Yukon pre-submit gate, intended to be about 10 minutes after setup. | Same public prompt, standalone prefill next-token check, decode seed-prefill check, and 1023 teacher-forced decode checks from a longer public fixture. | `score.json` with the estimated local `score`. |

When setup never produced a usable `mlx.metallib` (fresh checkout, or a
partial/failed setup), `benchmark.sh` exits immediately with guidance to run
`mlxfast setup` -- the Yukon CLI subcommand that runs this repository's
`setupCommand` from `benchmark.json`, which is `./setup.sh` -- or to run
`./setup.sh` directly; both are equivalent. Wrapper CLIs that drive
`benchmark.sh` under a different command name can set `MLXFAST_CLI_COMMAND` so
that guidance names their command; the variable only changes the printed
message, never behavior.

Both local modes publish `score` as the local ESTIMATE
(`decode_speedup^0.75 * prefill_speedup^0.25` against the pinned
`officialBaseline*` constants), so the Yukon participant CLI (`mlxfast run`),
which requires a finite numeric `score` at the contract `scorePath`, can
consume local runs; `metrics.runtime`
(`swift-local-iterate`/`swift-local-submit`) marks the payload as local.
Neither local mode produces an official leaderboard score. Official ranking
still runs the hidden benchmark oracle and hidden correctness gates on the
trusted runner, and only that ranked run's paired measurement produces the
official score.

Both local modes stream live numbers to stderr while they run, so you do not
have to wait for the final JSON: the official baseline constants up front,
prefill seconds-per-token and speedup the moment the measured prefill forward
finishes, the decode seed-prefill charge, and a running line per checked decode
token with the last-step latency, an ETA for the remaining decode tokens,
projected charged decode seconds-per-token, projected decode speedup, and a
projected score under the official formula. There is no weight streaming in
this model, so there are no expert-bandwidth/cache-hit-rate live fields to
report; `bandwidth_gb_per_token` is always `0` with
`bandwidth_source=ram_resident_model`. Long silent phases (weights digest,
prefill forward, decode seed prefill) print a heartbeat every 10 seconds, and
the first teacher-forced token mismatch is reported immediately (token values
stay in the score JSON only).

Local modes also forward the runtime worker's stderr live with an
`mlxfast-worker:` prefix, so debug prints added to model code under
`Sources/MLXFastModel/` are visible while iterating instead of disappearing
into the worker pipe. Lines that look like token comparisons are redacted the
same way as harness error output, and official runs keep the old behavior
where worker stderr surfaces only through the sanitized exit diagnostic.

After the score JSON is sealed, `benchmark.sh` prints a compact summary with
both speedups and the estimated score, and -- when a same-machine snapshot
exists at `score.local-iterate.baseline.json` (record one with
`cp score.local-iterate.json score.local-iterate.baseline.json` after a run on
the synced tip) -- deltas against that baseline so a change reads as faster or
slower at a glance. When that snapshot exists, the baseline numbers to beat are
also printed at the start of the run, so the live projected score has a target
from the first second.

## Scoring

```
decode_speedup = baseline_decode_sec_per_token / decode_sec_per_token
prefill_speedup = baseline_prefill_sec_per_token / prefill_sec_per_token
score = decode_speedup^0.75 * prefill_speedup^0.25
```

Higher score is better. The ranked score is **paired**: `baseline_*` is the
pinned reference implementation, measured on the same M5 box in the same
session as the candidate, each behind the same thermal gate, so an unmodified
reference implementation scores about `1.0` and the ratio cancels host drift.
Decode is weighted more heavily because it dominates interactive generation,
while prefill still matters for prompt processing.
Both phases must also stay within 5% of the paired baseline:

```
decode_speedup >= 0.95
prefill_speedup >= 0.95
```

The floor prevents a submission from sacrificing one serving phase badly to
improve the other. The baseline timings used in each run are emitted in that
run's `score.json`. The `MLXFastConstants.officialBaseline*` constants feed
**local-mode estimates only** (no second build is available locally); the
ranked denominator is the on-box measured reference (see
`docs/benchmark-window-freeze.md` for the measurement contract). On the
local-mode constants, the floors correspond to decode at most
`0.14064626165296054` seconds/token and prefill at most
`0.011163191525904606` seconds/token.
For scoring, decode is trusted parent wall-clock time for decode setup plus the
checked decode-token window, not worker-reported per-step time. That charges
prompt-specific seed prefill to the decode phase so submitted model code cannot
hide speculative decode work before the timer starts.
The whole model is RAM-resident with no weight streaming, so the harness
records `bandwidth_source=ram_resident_model` and reports
`bandwidth_gb_per_token=0`. RAM and phase-timing metrics are still reported
for operator review and future guardrails; they are not primary score factors.
Correctness is a hard gate. See TASK.md for the full correctness specification.
On the ranked pipeline the correctness and gates pass (the hidden 64-step base
case plus the GPQA behavior checks) runs first and the timed measurement runs
last, in fresh worker processes behind a quiescence wait and a fixed 40C GPU
thermal gate, so the correctness pass can neither warm nor perturb the
measured model path.
Public local correctness uses the checked-in correctness fixture. When a local
edit-loop signal is enough, `--local-iterate` uses that public 512-token prompt,
times standalone prefill separately, then times decode including seed prefill
plus 16 teacher-forced decode tokens, writes `score.local-iterate.json`, prints
it, and publishes `score` as the estimated local score (never a ranked
benchmark score). The submit hook `--local-submit` uses the same public prompt
with a longer 1024-token
fixture: it times the same standalone prefill and decode-seed phases plus 1023
teacher-forced decode tokens in one continuous trajectory, and writes
`score.json` with the same estimated local `score`; official ranking still
requires the hidden benchmark oracle on the trusted runner.
The score payload includes the baseline timings used for scoring (the
same-session measured reference on ranked runs, the calibrated constants in
local modes), computed speedups, wall-clock phase timings, final process RSS,
the (always-zero) expert streaming counters kept for schema stability, and
transformed-weights digest.

## Architecture

```
Sources/
  MLXFastCLI/                Swift command-line entrypoint
  MLXFastCore/                score.json, golden cases, shared contracts
  MLXFastTransform/           Swift offline weight transform
  MLXFastModel/                editable Gemma 4 31B 4-bit Swift runtime
  MLXFastHarness/             trusted correctness, golden, and benchmark runner
weights/                     transformed weights (harness loads from here)
  config.json                 runtime-authored text-tower config
  model.safetensors.index.json
~/.cache/huggingface/hub/... canonical frozen 4-bit reference checkpoint cache
reference_weights/...        compatibility symlink to the reference cache
correctness_prompts/         public correctness prompt and checked-in Gemma-generated golden
correctness_golden.json      hidden benchmark correctness cases and token oracle
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

The public correctness-only prompt and golden live in `correctness_prompts/`.
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
resulting file to the protected private R2 path. The benchmark workflow keeps
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
