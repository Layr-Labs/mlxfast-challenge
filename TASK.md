# mlxfast — Poolside Laguna XS 2.1 MTP Swift Challenge

Optimize trained-assistant block decode for Poolside Laguna XS 2.1 (MoE text
tower, 4-bit target) on Apple Silicon while preserving the target model's
exact serial greedy output.

## Default ranked contract

`benchmark.json` and `.github/workflows/benchmark.yml` define the default
Yukon track, `laguna-xs-2.1-mtp-v1`. The organizer supplies a pinned Laguna
target, Poolside's matched DFlash speculator (as the assistant), two hidden
M5-generated goldens, and a pinned serial K=1 baseline.

The trusted parent drives:

1. `mtp_decode_begin` with the supplied seed.
2. `mtp_decode_block` with only the last committed token and maximum block
   size 4.
3. Target verification of every proposed token before commitment.
4. Exact comparison of every returned token with the hidden serial oracle.

A one-token divergence fails the run. The parent owns the timer and the
logical token count; seed prefill is charged to decode. Ranked timing uses at
least three accepted alternating serial/MTP pairs behind the 40C thermal gate:

```text
score = mean(serial_K1_seconds_per_token) / mean(MTP_seconds_per_token)
floor = 1.0
```

Run `MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh`, `./setup-mtp.sh`, then
`./benchmark-mtp.sh --local-iterate` or `--local-submit`. Local runs generate
a candidate-local serial oracle, so they check parity and speed direction but
not official target fidelity. Only the hidden M5 goldens are authoritative.

The serial non-speculative challenge remains available explicitly through
`benchmark.serial.json`, `.github/workflows/serial-benchmark.yml`, and
`./benchmark.sh`; it is no longer Yukon's default.

## Model Artifacts

The default MTP target and assistant are provisioned by `setup-mtp.sh` and
pinned by `fixtures/laguna_xs_2_1_mtp_track.json`. The base-checkpoint layout
below documents the archived serial track.

By default, `setup.sh` stores the frozen reference checkpoint in a shared
Hugging Face-style cache under your home directory (so parallel clones reuse
one checkpoint):

```text
~/.cache/huggingface/hub/models--mlx-community--Laguna-XS-2.1-4bit/snapshots/main/
```

It also creates this compatibility symlink unless the path already exists:

```text
reference_weights/laguna-xs-2.1-4bit/
```

By default `setup.sh` downloads `mlx-community/Laguna-XS-2.1-4bit` from the
pinned Hugging Face revision with resumable `curl` requests (no organizer
mirror exists yet; one may be added later). It checks cached files
against the pinned SHA256 manifest and redownloads only missing, truncated, or
hash-mismatched files. (The checked-in Laguna weight manifests are entry-less
placeholders until the operator regenerates them on m5-bench; setup fails
closed on an entry-less manifest.) The safetensors payload is about 18.8 GB across 4
shards; `setup.sh` requires 40 GiB free by default before starting. After a
full verification, setup writes `.mlxfast-reference-cache.lock`; later setup
runs use cheap size/mtime checks from that lock and skip the full checkpoint
hash pass when the cache is unchanged. Set
`MLXFAST_REFERENCE_CACHE_DIR` or `MLXFAST_REFERENCE_DIR` to a different local or
mounted volume when needed, or set
`MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1` when the checkpoint is provisioned externally.

The Swift transform writes benchmark-ready weights here:

```text
weights/
  config.json
  model.safetensors.index.json
  model-0000N-of-0000M.safetensors
  tokenizer.json
  tokenizer_config.json
```

The generated `weights/` tree is a compact runtime artifact set, not a second
full copy of the checkpoint: it holds only the text-tower tensors (the
`language_model.` prefix in the source checkpoint), plus a runtime-authored
`config.json` (the Laguna geometry fields the runtime needs,
plus the checkpoint's quantization metadata). There is no expert
streaming manifest -- the whole model, including every routed expert, is
loaded fully into RAM at
init; there is no weight streaming of any kind. Submissions may adjust this
overlay by changing both `Sources/MLXFastTransform/` and
`Sources/MLXFastModel/`; correctness and benchmark results are the authority,
not byte equality with the baseline layout.

The public correctness-only prompt and golden are committed under
`correctness_prompts/` so participants can run a local correctness smoke test
(the checked-in fixtures are still legacy Gemma-generated and must be
regenerated on m5-bench with the Laguna tokenizer, vocab 100352, before they
gate Laguna code). The official correctness golden
is supplied by the benchmark operator and is intentionally not committed to
the public repo:

```text
correctness_golden.json
```

Use `MLXFAST_CORRECTNESS_GOLDEN_PATH=/path/to/correctness_golden.json` when the
file is provisioned outside the repository root.
Benchmark CI consumes the checked-in public golden for correctness-only runs and
downloads the private precomputed correctness golden from protected storage for
full benchmark runs. The timed phase separately downloads a pinned private
evaluation prompt after the correctness scrub. The trusted box-side measurement
wrapper generates and caches a benchmark token oracle for that prompt per binary
and validates all charged outputs against it. Private prompt manifests, the
timed prompt, and hidden correctness goldens are not committed to the public
repository. Organizers regenerate correctness fixtures and rotate the timed
target through the controlled operator process.

## Editable Surface

The active editable surface is Swift-only and is defined by `benchmark.json`:

| Path | Scope |
|---|---|
| `Sources/MLXFastModel/` | Laguna XS 2.1 4-bit model implementation: attention (sliding-window + full, GQA, YaRN partial-rotary RoPE on full-attention layers), MoE MLP (256 routed experts + shared expert, per-head gating), RMSNorm, KV caches, weight loading, and prefill/decode execution. |
| `Sources/MLXFastTransform/` | Offline safetensors transform (text-tensor selection, config/tokenizer emission). |

`Sources/MLXFastCore/`, `Sources/MLXFastHarness/`,
`Sources/MLXFastCLI/`, scripts, tests, `benchmark.json`, generated
`weights/`, reference checkpoints, golden fixtures, and local scores are
harness/operator files, not submission surface. Correctness, scoring, timing,
golden generation, benchmark-oracle validation, and provenance checks live in
that trusted harness layer.

Account and submission management — login, clone, submit, and listing
submissions — are handled by the **Yukon CLI (`mlxfast`)**, not by
`mlxfast-swift`. The Swift binary now runs the benchmark domain only (transform,
correctness, benchmark, preflight, verify-transform); it no longer logs in or
uploads. The CLI installer defaults to `~/.local/bin`, so expose that directory
in the current shell before using it. Submit with:

```bash
export PATH="${HOME}/.local/bin:${PATH}"
mlxfast login <api-key> --api <url>
mlxfast clone <benchmark-id-or-name>     # fresh checkout; an existing repo auto-links by its git remote
mlxfast submit --model "<model name>" --note "..."
mlxfast submissions
```

`mlxfast submit` reads `benchmark.json` and uploads only `editablePaths` as a
gzip tar archive with bearer-token auth; the backend applies it to the frozen
benchmark checkout and re-enforces the editable surface server-side before
running hidden validation. `--model` is required and is recorded for the
leaderboard; pass `--note-file PATH` or `--claimed-score N` as needed.
The benchmark contract also declares a local `preSubmitCommand`:
`./benchmark-mtp.sh --local-submit`. `mlxfast submit` does not run it — the upload
goes directly to official validation, and no local run blocks it. Running that
command yourself before submitting is the recommended local parity and timing
check, without running the official hidden golden.

`mlxfast-swift verify-transform` is an organizer/debug check for deterministic
transform output. It re-runs the submitted transform and compares the generated
`weights/` tree against that fresh run. It is not a baseline-layout requirement.
The normal preflight/benchmark path also rejects generated `weights/` above the
default 25 GiB transformed-output cap before correctness or timing runs (the
text tower is about 18.8 GB, comfortably under that cap).
Override it with `MLXFAST_MAX_WEIGHTS_BYTES`; `verify-transform` additionally
accepts `--max-bytes`.

There is no Python harness path.

## Correctness Gates

The default track replays a hidden 512-token MTP correctness golden, requires
the trained assistant and exact-pair target verification, and checks every
returned token against the parent-owned serial K=1 oracle. It also validates
block size, accepted-prefix accounting, cache offsets, and rollback. Any
failure makes the submission ineligible before timing.

### Archived serial gate

Correctness is a hard gate. Each base golden case contains exactly 512 prompt
token IDs and at least 64 expected continuation token IDs. The harness checks
the first 64 continuation positions teacher-forced with temperature-zero
behavior: after each accepted step it feeds the golden previous token back into
the model. The first mismatch records only the case, step, expected token, and
actual token in the failed report.

The gate is intended as a first-stage filter: an implementation that fails it is
not eligible for the longer benchmark.

Private golden fixtures may add hidden `correctness_gates` on top of the base
teacher-forced cases:

- `anchors`: one-token checks at selected hidden contexts. These can require an
  exact expected token, explicit accepted tokens, or a bounded top-logit rank
  and delta for near-tie hardware cases.
- `free_run`: short greedy continuations whose exact prefix must match. These
  catch bugs that only appear when the model consumes its own generated tokens.
- `behavior`: GPQA-style or instruction-following prompts whose answer is
  checked exactly against precomputed accepted answer token sequences. Each
  accepted answer sequence must have at most `max_new_tokens` tokens; shorter
  sequences are matched as exact prefixes of the generated answer.

Full benchmark CI adds one more private layer after the correctness and gates
pass (and before the timed measurement, which runs last on the ranked
pipeline): it captures short answers for hidden GPQA cases and asks a Claude
judge whether each candidate is semantically equivalent to the private
reference answer. That semantic gate is pass/fail only and does not affect
the timing score; its pass-count threshold is baseline-calibrated (see
`MLXFastConstants.semanticGPQAMinPassCount`). The uploaded score records only
aggregate semantic counts and the judge model name.

The same hidden GPQA cases are also used for a TTFT guardrail: during the
hidden behavior correctness pass, the workflow times prompt prefill through
the first greedy answer token and verifies that the first token is accepted for
that case. The uploaded score records only
aggregate TTFT pass counts and timing statistics; first-token values and
accepted token sets are not logged or artifacted.

These layers keep the official gate mostly deterministic and token-based while
adding a small semantic backstop against implementations that pass the exact
prefix but damage answer meaning. The benchmark operator should keep private
prompts, accepted answer sequences, reference answers, and judge transcripts
outside the public repository.

The gate intentionally does not port a hidden-state comparison layer. The
benchmark contract cares about the externally observable text-to-text Laguna
output path, and hidden-state tensors are easier to make ambiguous around
normalization than token-level or logit-anchor checks.

VLM/image and audio inputs remain out of scope. Only the Laguna text tower
and organizer-provided DFlash MTP assistant execute.

### Default MTP decode rule

Each `mtp_decode_block` request supplies only the last committed token and a
trusted maximum block length. The participant may draft and verify within that
block using the pinned assistant and target, but may return only the
target-confirmed prefix. Logical and physical cache positions advance exactly
by the returned token count. The trusted parent independently checks every
token against its serial oracle and rejects mismatches, extra rows, invalid
rollback, or protocol drift.

Prompt lookup, token-history drafting, a different assistant, hidden-prompt
specialization, and precomputed future outputs remain forbidden. Kernel edits
are allowed when they are input-general for Laguna and matched across AOT
sources and JIT twins.

### Archived serial non-speculative rule

In `benchmark.serial.json`, each model
invocation may compute logits and KV rows only for tokens supplied in that
invocation, and must advance logical and physical KV position by exactly the
supplied input length. A one-token decode request therefore advances exactly
one position and leaves no pending future token, logits, or KV state for a
later request.

This excludes prompt-lookup decoding; n-gram, suffix, or other token-history
drafting; same-target lookahead; and any other mechanism that selects or
evaluates an unsupplied future token. It also excludes two-, three-, or
more-row target-model execution used to verify a draft from a one-token
request, plus cross-request future-logit/KV buffering, deferred cache rows,
and commit, rollback, recommit, or discard markers for such rows. These
mechanisms remain excluded when they are generic, bit-exact, or useful in
production. Warming an excluded speculative pipeline before the worker
protocol hello or during model initialization does not make it eligible.

Ordinary within-request KV reuse remains allowed, as do current-token-only
decode execution and input-independent caches for weights, dequantized
tensors, kernels, masks, or RoPE tables. Multi-row kernels are allowed when
every row corresponds to a token actually supplied in that same invocation,
such as ordinary prefill; the prohibited case is using extra rows to compute
or verify future tokens for a serial one-token request.

Those restrictions apply only to the explicit serial workflow. They do not
prohibit the organizer-defined block protocol in the default MTP track.

## Score

```text
score = mean(serial_K1_seconds_per_token) / mean(MTP_seconds_per_token)
```

Higher is better. Both sides run on the same M5 in the same session, with
alternating order and the same fixed thermal/telemetry acceptance. The hard
component floor is:

```text
score >= 1.0
```

A run below the floor or with any token mismatch is ineligible.
`score.json` also carries pair counts, parity, MTP acceptance diagnostics,
seconds/token means, and transformed-weight identity.

## Useful Commands

```bash
swift test
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test
swift build -c release
MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh
./setup-mtp.sh
./benchmark-mtp.sh --local-iterate
./benchmark-mtp.sh --local-submit

# Submitting is done with the Yukon CLI (mlxfast), not mlxfast-swift:
mlxfast clone <benchmark-id-or-name>
mlxfast submit --model "<model name>" --note "..."
mlxfast submissions
```
