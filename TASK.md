# mlxfast — Gemma 4 31B 4-bit Swift Challenge

Optimize Gemma 4 31B (dense, text tower, 4-bit) inference on Apple Silicon
while preserving exact greedy output for the supplied correctness prompts.

## Contract

Submissions are evaluated through the Swift harness:

```bash
./setup.sh
./benchmark.sh --official
```

(`--official` requires the organizer-provisioned hidden oracle; a bare
`./benchmark.sh` defaults to the public `--local-iterate` mode.)

The benchmark entrypoint:

1. Builds `mlxfast-swift` when needed.
2. Runs the Swift transform if `weights/` is missing, the recorded transform
   source hash changed, or `MLXFAST_FORCE_TRANSFORM=1`.
3. Runs the correctness gate against `correctness_golden.json`.
4. Validates the benchmark prefill/decode tokens against the hidden benchmark
   oracle in `correctness_golden.json`.
5. Measures prefill latency, 128-token checked decode latency, and MLX peak
   memory.
6. Writes `score.json` in the Darkbloom-compatible schema, plus
   `score.json.sha256` and `benchmark-integrity.json` audit sidecars.

If required artifacts are missing, the harness writes a failed `score.json`
rather than producing a ranked score.

After transform, local users can run the checked-in public correctness gate with
`.build/release/mlxfast-swift correctness --weights weights`. For benchmark
iteration, `./benchmark.sh --local-iterate` uses that public 512-token prompt,
checks the prefill next token plus 16 teacher-forced decode tokens, and writes
the measured 512-token prefill and 16 one-token decode timings to
`score.local-iterate.json`. That file's `score` is the local ESTIMATE
(`decode_speedup^0.75 * prefill_speedup^0.25` vs the pinned `officialBaseline*`
constants; `metrics.runtime` marks the local mode) so the Yukon participant CLI
(`mlxfast run`), which requires a finite numeric `score`, can read it — it is a
directional edit-loop signal, not a ranked score.
For submit-loop iteration, `./benchmark.sh --local-submit` uses the same public
512-token prompt as a longer pre-submit benchmark. It checks the prefill next
token plus 1023 teacher-forced decode tokens from a longer public fixture, and
writes and prints `score.json` with the same estimated local `score`; it is a
directional local signal, not the official ranking run. Only the ranked
runner's paired measurement produces the official score.

> **M5-generated correctness fixtures.** `correctness_prompts/` contains
> prompt/golden fixtures generated on the ranked M5 hardware against the
> Gemma 4 31B 4-bit reference implementation (the Layr-Labs `mlx-swift-lm`
> Gemma 4 text tower this package builds against). The 512-token prompts were
> retokenized from the checked-in prompt text with the Gemma tokenizer, and
> the expected continuation tokens were captured from the reference model's
> greedy forward pass with `mlxfast-swift generate-golden` (256 expected
> tokens for the local-iterate fixture, 1024 for the local-submit fixture;
> the shorter fixture is a greedy prefix of the longer one by construction).
> The hidden/private artifacts used by ranked runs (R2 golden, GPQA
> references) were regenerated the same way through the organizer process.
> One caveat for local work: near-tie greedy argmaxes can diverge across
> Apple Silicon generations, so on non-M5 machines the local public gate may
> fail for a correct build — local modes are directional, and the ranked M5
> runner is the source of truth.

## Model Artifacts

By default, `setup.sh` stores the frozen reference checkpoint in a shared
Hugging Face-style cache under your home directory (so parallel clones reuse
one checkpoint):

```text
~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-4bit/snapshots/main/
```

It also creates this compatibility symlink unless the path already exists:

```text
reference_weights/gemma-4-31b-4bit/
```

By default `setup.sh` downloads `mlx-community/gemma-4-31b-4bit` from the
Darkbloom R2 mirror with resumable `curl` requests, falling back to the public
Hugging Face source on failed or stalled transfers. It checks cached files
against the pinned SHA256 manifest and redownloads only missing, truncated, or
hash-mismatched files. The safetensors payload is about 18.4 GB across 4
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
`language_model.` prefix in the source checkpoint), with every
vision/audio/multimodal-projector tensor dropped, plus a runtime-authored
`config.json` (the flattened Gemma 4 `text_config` fields the runtime needs,
plus the checkpoint's quantization metadata). There is no expert manifest --
the whole model is one flat set of dense tensors loaded fully into RAM at
init; there is no weight streaming of any kind. Submissions may adjust this
overlay by changing both `Sources/MLXFastTransform/` and
`Sources/MLXFastModel/`; correctness and benchmark results are the authority,
not byte equality with the baseline layout.

The public correctness-only prompt and golden are committed under
`correctness_prompts/` so participants can run a local correctness smoke test
(Gemma-generated; see the fixture note above). The official correctness golden
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
| `Sources/MLXFastModel/` | Gemma 4 31B 4-bit model implementation: attention (sliding-window + full, GQA, partial-rotary RoPE), gated MLP, RMSNorm, KV caches, dense weight loading, and prefill/decode execution. |
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
`./benchmark.sh --local-submit`. `mlxfast submit` does not run it — the upload
goes directly to official validation, and no local run blocks it. Running that
command yourself before submitting is the recommended roughly 10-minute local
correctness and timing check, without running the full official hidden
benchmark.

`mlxfast-swift verify-transform` is an organizer/debug check for deterministic
transform output. It re-runs the submitted transform and compares the generated
`weights/` tree against that fresh run. It is not a baseline-layout requirement.
The normal preflight/benchmark path also rejects generated `weights/` above the
default 25 GiB transformed-output cap before correctness or timing runs (the
text tower is about 17 GB, comfortably under that cap).
Override it with `MLXFAST_MAX_WEIGHTS_BYTES`; `verify-transform` additionally
accepts `--max-bytes`.

There is no Python harness path.

## Correctness Gate

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
benchmark contract cares about the externally observable text-to-text Gemma 4
output path, and hidden-state tensors are easier to make ambiguous around
normalization/softcapping than token-level or logit-anchor checks.

VLM/image inputs, audio inputs, and speculative/MTP draft decoding are also
out of scope for this challenge. Only the Gemma 4 text tower is in scope; the
vision tower is never loaded or executed. These should only be added if the
official benchmark contract changes to score those paths.

### Serial non-speculative decode rule

**Controlling rule:** In the current serial non-speculative track, each model
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

Organizer-provided MTP or other speculative decoding belongs in a separate,
explicitly declared track with a trusted variable-length block protocol,
separate correctness rules, and separate scoring. It is not an optimization
within this track.

An unranked prototype of that separate track is documented in
`docs/experimental-mtp-track.md`. Its track ID is
`gemma4-31b-it-mtp-v1`: it pairs an organizer-pinned Gemma 4 31B-IT target
with Google's matched organizer-provisioned assistant, permits only
target-verified blocks of at most four positions, and retains no authority to
publish a score until a separate M5 reference baseline is established. The
trusted parent owns the oracle, timer, and denominator; the participant
surface is the drafting/verification strategy inside
`Sources/MLXFastModel/` (the block session and the bit-exact exact-pair
verification kernels), measured decode-only against a same-session paired
serial reference with a hard bit-exact token gate. This prototype does not
relax any rule above for the current serial track.

The gates-only hidden golden retains a benchmark section because the common
harness schema requires it, but the ranked timed phase uses a separate
self-generated oracle derived from the private timed target. The benchmark
validates the greedy token after the fixed 512-token prefill prompt, the greedy
token after the fixed 512-token decode seed, and all 128 tokens produced inside
the timed decode window before accepting a score.

## Score

```text
decode_speedup = baseline_decode_sec_per_token / decode_sec_per_token
prefill_speedup = baseline_prefill_sec_per_token / prefill_sec_per_token
score = decode_speedup^0.75 * prefill_speedup^0.25
```

Higher is better. The ranked score is paired: the baseline is the pinned
Gemma 4 31B 4-bit reference implementation, measured on the same self-hosted
M5 box in the same session as the candidate (each timed phase behind the same
fixed 40C thermal gate, with telemetry-validated acceptance), so an
unmodified reference scores about `1.0` and host drift cancels out of the
ratio. Decode is weighted more heavily because it dominates interactive
generation, while prefill still contributes to the ranked score.
The official run also enforces component floors:

```text
decode_speedup >= 0.95
prefill_speedup >= 0.95
```

On ranked runs both floors are priced against that live same-session paired
baseline. A run below either floor fails eligibility even if the weighted
score would otherwise be above baseline. The
`MLXFastConstants.officialBaseline*` constants feed local-mode estimates only
(see `docs/benchmark-window-freeze.md` for the measurement contract); on
those local-mode constants, the floors correspond to at most
`0.04637500805235746` seconds/token for decode and
`0.0017070057649739585` seconds/token for prefill.

The whole model is RAM-resident with no weight streaming, so
`bandwidth_gb_per_token` is always `0`, reported with
`bandwidth_source=ram_resident_model`. RAM and phase-timing metrics are
diagnostics and guardrail candidates, not primary score factors.
`score.json` also carries audit-only wall-clock phase timings, final process RSS,
zeroed expert-streaming counters (kept for score-schema stability), and
transformed-weights digest fields. These values help operators review runs but
do not change the score formula.

## Useful Commands

```bash
swift test
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test
swift build -c release
MLXFAST_OFFLINE_WRITABLE_PATHS="${PWD}/weights" .github/scripts/run-offline.sh .build/release/mlxfast-swift transform --output weights
.build/release/mlxfast-swift correctness --weights weights
.build/release/mlxfast-swift preflight
.build/release/mlxfast-swift benchmark --local-iterate
.build/release/mlxfast-swift benchmark --score-path score.json
.build/release/mlxfast-swift benchmark --local-submit --score-path score.json
.build/release/mlxfast-swift verify-transform

# Submitting is done with the Yukon CLI (mlxfast), not mlxfast-swift:
mlxfast clone <benchmark-id-or-name>
mlxfast submit --model "<model name>" --note "..."
mlxfast submissions
```
