# Poolside Laguna XS 2.1 MTP Track (RETIRED)

> **RETIRED (2026-07-21).** The MTP track was dropped before going live for
> Laguna. This document intentionally records the then-current
> `mlx-community` affine target and `laguna-xs-2.1-serial-v1` namespace. The
> current and only ranked track is the independently versioned Poolside
> NVFP4 contract, `laguna-xs-2.1-serial-v2`; `benchmark.json` and
> `.github/workflows/benchmark.yml` are authoritative. The MTP
> manifests (`benchmark.mtp.json` and the MTP-shaped `benchmark.json`), the
> contract fixture (`fixtures/laguna_xs_2_1_mtp_track.json`), the MTP weight
> manifests, and the local scripts (`setup-mtp.sh`, `benchmark-mtp.sh`) have
> been deleted. The `mtp-*` CLI commands remain in the Swift tree as
> unranked experimental code only. This document is preserved as a
> historical design record: file, manifest, and workflow references below
> describe the repository as it was when the track was live and are
> deliberately not rewritten.

## Status and default routing (historical)

This was the default official ranked track, with track ID
`laguna-xs-2.1-mtp-v1`. Yukon read `benchmark.json`, dispatched
`.github/workflows/benchmark.yml` (then the MTP pipeline), and ingested
`score.json`. Official scoring
was decode-only paired speedup against the pinned serial K=1 target baseline.
The base-model serial challenge was archived at the time as
`benchmark.serial.json` and `.github/workflows/serial-benchmark.yml`; it has
since become the default again under the canonical names `benchmark.json`
and `benchmark.yml`.

The two experimental commands are:

- `mtp-probe`: PR #424's serial target-only block control.
- `mtp-benchmark`: a real trained-assistant MTP path that refuses to run
  without `--require-trained-assistant`.

The `mtp-*` commands remain protocol-explicit; no environment variable can
turn the archived serial `benchmark` command into MTP.

## Compatibility conclusion

The current challenge target is:

- Runtime checkpoint:
  `mlx-community/Laguna-XS-2.1-4bit@c42e0a8f8d504ceacde015a535dcb286d65c8799`
  (MLX affine 4-bit, group size 64; the per-layer MoE router gates are
  8-bit; keeps the `language_model.` tensor prefix)
- Declared upstream base:
  `poolside/Laguna-XS-2.1@c405648833500615a2efde76886b8aed4fb9324e`
- Checked-in byte manifest:
  `fixtures/reference_laguna_xs_2_1_4bit.sha256`

Poolside's public DFlash speculator is explicitly paired by its model card
and the target's `generation_config` speculative block with this target:

- Target:
  `poolside/Laguna-XS-2.1@c405648833500615a2efde76886b8aed4fb9324e`
  (model_type `laguna`: hidden 2048, 40 layers, 48 full-attention /
  64 sliding-attention heads with 8 KV heads at head_dim 128, vocab 100352,
  untied embeddings, sliding_window 512 with a 3:1 sliding:full layer
  pattern — full attention at layers 0, 4, 8, ..., 36 — YaRN partial-rotary
  0.5 on full-attention layers and plain RoPE theta 10000 on sliding layers,
  MoE with 256 routed experts + 1 shared expert, 8 experts per token,
  moe_intermediate 512, shared_expert_intermediate 512, per-head gating,
  and a dense-MLP layer 0 with intermediate 8192)
- Assistant (draft):
  `poolside/Laguna-XS-2.1-DFlash@5c36361aab23c8ed3afbd079c10c426b677bc607`
  (5-layer DFlash Eagle-style speculator, BF16 upstream
  `model.safetensors` = 924,135,848 bytes; dflash_config: block_size 16,
  mask_token_id 12, num_target_layers 40, target_layer_ids [1,13,25,33,39],
  causal; draft_vocab_size 100352; aux hidden-state layer ids
  [2,14,26,34,40]. The draft is downloaded as the BF16 upstream and
  converted to MLX affine 4-bit, group size 64, at setup.)

The DFlash speculator is trained against this exact target, so the track
binds the pinned pair directly; there is no separate base-vs-IT pairing
question for Laguna.

The MLX target conversion is pinned independently:

- `mlx-community/Laguna-XS-2.1-4bit@c42e0a8f8d504ceacde015a535dcb286d65c8799`
- Declared base model: `poolside/Laguna-XS-2.1`

The target and DFlash model cards declare OpenMDW-1.1 with terms at
<https://huggingface.co/poolside/Laguna-XS-2.1>. Operators must still review
the license terms before enabling a public ranked track.

## Provenance contract

`fixtures/laguna_xs_2_1_mtp_track.json` binds:

- track identity and disabled-scoring state;
- target and assistant model IDs plus immutable repository revisions;
- the exact pinned `mlx-swift-lm` revision
  `bc1c0ee67d15798343be17c9f8f61f7c0d977149`;
- frozen Laguna target and DFlash assistant architecture fields (including
  the MoE fields: 256 experts, 8 per token, moe_intermediate 512, shared
  expert 512, dense-MLP layer 0);
- SHA256 manifest identities and artifact byte budgets;
- block size four, a compatibility default of 128 parent-counted decode
  tokens, and a hard trusted-parent cap of 1,536 (ranked timed decode stays
  fixed at 512);
- the pending-M5-rebaseline status.

Byte manifests:

- `fixtures/mtp_laguna_xs_2_1_4bit.sha256`
- `fixtures/mtp_laguna_xs_2_1_dflash.sha256`

Both manifests are entry-less placeholder headers, and the manifest
self-hashes and assistant byte pins in the contract fixture are TODO markers,
until they are regenerated on m5-bench with the real weights (hashes are
never pre-filled off-box; setup fails closed on an entry-less manifest).

The assistant source inventory is exactly two regular, single-link files:

- `config.json`: bytes pinned on m5-bench (the upstream file is 963 bytes)
- `model.safetensors`: bytes pinned on m5-bench (the BF16 upstream payload
  is 924,135,848 bytes per Hugging Face LFS metadata; setup converts it to
  MLX affine 4-bit, group size 64, on-box)

Extra files,
symlinks, hardlinks, size mismatches, hash mismatches, incompatible config
fields (including the runtime affine 4-bit group-64 quantization contract),
or a total above the pinned maximum fail before model load.

The target source totals 18,829,720,326 bytes per Hugging Face API metadata
at the pinned revision (confirm on m5-bench when the manifest is
regenerated). The transformed
text-only target is capped at 20 GiB. The assistant remains an
organizer-provisioned read-only sidecar and is not copied into participant
submission artifacts.

`setup-mtp.sh` is the only MTP provisioner. It uses separate target and
assistant cache directories, immutable revision URLs, resumable `.partial`
downloads, exact size/SHA256 verification, and strict flat inventories. It
never changes `setup.sh`'s base checkpoint.

`mtp-benchmark` also requires `--target-source` and revalidates that complete
18.8 GB source inventory in the trusted parent. This prevents an accidental
mispairing of a different checkpoint with the DFlash assistant. In an
official pipeline, the trusted
transform step must consume that same validated source; returned-token parity
then binds the transformed runtime to the Laguna oracle.

No weights are committed to this repository.

## Pinned MLX APIs and model integration

Note: the Swift symbol names in this section and below (`Gemma4*`,
`GemmaRuntimeMTP*`) are the pre-port implementation names. The Laguna/DFlash
Swift port replaces them with Laguna equivalents
(`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Laguna.swift`,
`LagunaMTP.swift`, `LagunaMTPTarget.swift`, and the corresponding
`Sources/MLXFastModel/` session types); the roles and guarantees described
here carry over unchanged.

The Package.resolved revision exposes:

- `Gemma4MTPTarget`
- `Gemma4MTPForward`
- `Gemma4SharedKV`
- `Gemma4AssistantConfiguration`
- `Gemma4AssistantDraftModel.load(from:)`
- assistant `bind(target:)`
- target `forwardForMTP`
- target `rollbackSpeculativeCache`
- `Gemma4MTPTokenIterator` and `runGemma4MTPRounds`

`Gemma4RuntimeModel` conforms to `Gemma4MTPTarget`. Its MTP-only forward uses
the pinned library trunk's pre-norm/shared-KV capture hook. The ordinary
serial entry point is unchanged. The fast engine has a separate exact-pair
entry point that is unreachable from serial `benchmark`.

`Gemma4TrainedMTPBlockSession` persists, for one request:

- the target KV caches;
- the last committed target bonus token;
- the target pre-norm hidden state;
- the last full-attention and sliding-attention shared K/V snapshots;
- a host mirror of the target cache offset.

Target verification is selected explicitly:

- `--target-verification exact-pair` is the default trained-MTP path.
- `--target-verification serial` is the K=1 control.

An exact-pair session fails during construction if the target does not have
all required packed metadata and kernels. It never catches a numerical
mismatch and silently reruns that pair serially.

For configured block size `K` in `2...4`, one round:

1. Uses the trained assistant to draft `K - 1` tokens from the current target
   token, target hidden state, and shared K/V.
2. Verifies two target rows at a time when a pair is available. K=4 composes
   two exact pairs; K=3 uses one pair plus an exact K=1 bonus tail; K=2 uses one
   pair; K=1 is target-only.
3. Compares each draft to the corresponding target argmax in order and starts
   no later segment after the first rejection.
4. Emits only the target-confirmed prefix plus the target token at the first
   rejection (between one and `K` tokens).
5. If row zero of a pair rejects, removes row one's physical cache position
   from every layer and slices the shared-KV view before returning. If row one
   rejects, row zero's accepted input remains committed.
6. Persists hidden/shared-KV state at the committed target position.
7. Checks every target cache offset against the host mirror.

The exact-pair kernels share packed weight traversal while preserving each
row's K=1 accumulation and reduction order. They cover sliding Q/K/V, full
Q/K, attention output, the MLP path, layer boundaries,
and the vocabulary head. RMS normalization remains row-serial.
(This paragraph describes the pre-port Gemma pair kernels; the Laguna port
re-derives the same row-exactness guarantees for head_dim 128 attention,
the untied lm_head, and the MoE expert path.) Row two therefore sees row
one's K/V exactly as it
does in serial decoding.

The expanded M5 pilot proved why an ordinary mathematical K-row target
forward is insufficient: retained layer-1 K/V diverged after the first
full-acceptance block and a public prose continuation flipped argmax at decode
step 48, despite identical logical offsets. The exact-pair path is not that
ordinary batched path. Its regression gate compares both pair rows against two
K=1 forwards using exact logits, pre-norm hidden bits, every layer's physical
K/V state, and rollback state at multiple offsets.

Zero, partial, and full acceptance use the same charged path. A deterministic
serial tail is used only when a block shape requires one row or a rotating
cache reaches geometry where a prefix-preserving pair cannot be formed. These
rows are counted in `serial_verification_row_count`; engine ineligibility
fails closed rather than becoming a timed fallback.

Any exception after a block begins poisons the worker session. The worker
cannot continue from ambiguous cache state.

## Trusted block protocol

The trained worker accepts only:

- `mtp_decode_begin`: the target seed prompt.
- `mtp_decode_block`: request ID, last parent-committed token, and maximum block
  size.

Block requests contain no expected token, future oracle token, accepted count,
worker duration, score, denominator, prompt hash, or continuation history.
Request IDs must be monotonic. Block responses carry a nonce and a nonempty
token block only. The separate post-phase diagnostic response can report
memory, selected verification mode, exact-pair segments, pair rollbacks, and
serial geometry/tail rows; none has timing or score authority. Unknown fields
such as `accepted_count`, `token_count`, `seconds`, or `future_tokens` are
rejected by the parent decoder.

The trusted parent:

1. Validates target/assistant artifacts without invoking editable model code.
2. Starts its timer before the prompt-bearing begin request.
3. Checks the target-produced seed token.
4. Sends only the last committed token on each block request.
5. Rejects missing, empty, oversized, out-of-vocabulary, overrun, or
   mismatching blocks.
6. Checks every returned token against an independent serial target oracle.
7. Stops only after the explicit trusted total (default 128, maximum 512).
8. Divides its own wall time by that parent-owned total.

All seed prefill, drafting, target verification, rejection, rollback, IPC, and
final-tail work is charged. Worker-reported timing and acceptance have no
authority.

The MTP worker receives target, assistant, and contract paths as explicit
arguments. Its environment uses the existing allowlist sanitizer. Its Seatbelt
profile denies network, process creation, and all writes except `/dev/null`.
It validates all artifacts before loading and revalidates them after loading,
closing the practical replace-during-load window. On the official runner,
operator-owned read-only cache permissions remain required.

## Static-review policy

`run-submission-static-review.sh` defaults to `serial`, preserving the current
ban on all speculation. Trusted rollout code may set:

```text
MLXFAST_SUBMISSION_TRACK_ID=laguna-xs-2.1-mtp-v1
```

The MTP policy permits only organizer-assistant, target-verified block
speculation and its within-request commit/rollback state. It still rejects:

- prompt, suffix, n-gram, history, or prompt-hash lookup;
- known fixed-token-total or call-count specialization;
- participant-provided, replaced, tampered, or extra assistants;
- unverified drafter tokens or skipped target verification;
- fake timing, token counts, acceptance, artifact data, or scores;
- stale target logits;
- logical or physical KV state beyond the committed prefix;
- hidden future-token/logit/KV buffering outside the protocol;
- prompt-dependent warmup or precompute before the trusted timer.

Static review is a gate, not a proof. The trusted parent's independent oracle
and fixed timing/count contract remain the substantive runtime controls.

## Memory and artifact budget

Known on-disk bytes:

- target source: 18,829,720,326 bytes per Hugging Face API metadata at the
  pinned revision (confirm on m5-bench);
- assistant draft download: 924,135,848 bytes BF16 per Hugging Face LFS
  metadata (converted to MLX affine 4-bit at setup; the converted runtime
  form is smaller);
- combined source: about 18.4 GiB.

The assistant is the 5-layer DFlash Eagle-style speculator, stored at
runtime as affine 4-bit
(group size 64) packed U32 weights with scales/biases after the on-box
setup conversion. The old-target M5 matrix measured up to 47.6 GiB peak process RSS,
31.2 GiB MLX active memory, 3.4 GiB MLX cache memory, and 34.1 GiB MLX allocator
peak; expect the Laguna numbers to be re-measured during the M5 rebaseline.
Runtime peak is higher than file size because target and assistant
weights, target KV, shared K/V, verification logits/activations, and allocator
state coexist. This fits the 128 GiB runner but exceeds the base track's
practical 36 GiB local budget; a public track needs its own documented memory
minimum and host-enforced cap.

## Participant surface and track design

The MTP track is designed so participants can make block decode faster
without being able to weaken the correctness or measurement guarantees.

### Trusted surface (operator-owned, not submittable)

These components define the measurement and cannot be changed by a
submission; they live outside `benchmark.json` `editablePaths`:

- `Sources/MLXFastHarness/GemmaRuntimeMTP.swift`: the trusted parent. It
  owns the timer (started before the prompt-bearing begin request), the
  serial-oracle validation of every returned token, the configured decode
  total used as the fixed denominator, block-size bounds, and the bounded
  diagnostics accounting.
- `Sources/MLXFastHarness/GemmaRuntimeMTPWorker.swift`: the sandboxed worker
  protocol shell. It validates artifacts before and after model load,
  enforces the strict `mtp_decode_begin`/`mtp_decode_block` request schema,
  monotonic request IDs, in-vocabulary bounded blocks, the cache-offset
  ledger, and session poisoning after any failure.
- `Sources/MLXFastHarness/GemmaRuntimeMTPProvenance.swift` (pre-port name),
  `fixtures/laguna_xs_2_1_mtp_track.json`, and the SHA256 manifests: the
  pinned target/assistant identity and architecture contract.
- `Sources/MLXFastCLI/main.swift`: track dispatch and the worker Seatbelt
  sandbox profile.
- `setup-mtp.sh` and the organizer cache: artifact provisioning.
- `.github/scripts/run-submission-static-review.sh`: the MTP-track
  static-review policy.

### Editable surface (what participants optimize)

Submissions change only `Sources/MLXFastModel/` (and
`Sources/MLXFastTransform/`), which contains the whole speculative strategy:

- `Gemma4MTPRuntime.swift`: `Gemma4TrainedMTPBlockSession` — the drafting
  loop, verification composition (exact pairs, serial tails), acceptance
  handling, commit/rollback, and per-request state reuse.
- `Gemma4ExactTwo*.swift` and the fast-engine pair path: the bit-exact
  multi-row verification kernels. Participants may improve dispatch, extend
  coverage (for example a four-row exact composition), or replace the
  strategy entirely.
- Everything the serial track already allows: quantized matmul dispatch,
  attention restructuring, KV-cache handling, weight layout, scheduling.

Improvement directions intentionally left open: higher drafter acceptance,
fewer target dispatches per committed token, reducing the one-time
seed-prefill first-touch cost, cheaper drafter execution, and deeper kernel
fusion — all subject to the same oracle. (The first-block stall itself is
already fixed; see "First-block stall: root cause and fix".)

### Why participants cannot cheat the measurement

- The parent validates every returned token against a serial oracle golden
  generated from the pinned reference; one divergent token fails the run.
  Bit-exactness is therefore a hard gate, not a convention: a submission
  that "wins" by degrading output cannot score.
- The parent owns wall time and divides by its own configured decode total.
  Worker-reported timing, acceptance, and counters have no score authority,
  are bounded before use, and must satisfy the physical row-accounting
  equation.
- Blocks are capped at four positions; oversized, empty, out-of-vocabulary,
  or overrun blocks are rejected; request IDs are monotonic; the worker sees
  no future oracle tokens (requests carry only the last committed token).
- The worker process receives the prompt only inside the timed window, runs
  sandboxed (no network, no writes, no process spawning), and artifact
  hashes are revalidated after model load, so assistant substitution and
  pre-timer prompt work fail closed.
- What the parent cannot prove (internal use of the assistant, real target
  verification, physical rollback correctness when outputs still match) is
  covered by the MTP static-review policy, hidden prompt-independent tests,
  the cache-offset ledger, and the runtime parity gates that any kernel
  change must rerun.

### Proposed scoring (pending operator calibration)

The contract's `proposed_scoring` block records the adopted ranked form:

```text
mtp_decode_speedup = paired_serial_decode_sec_per_token / mtp_decode_sec_per_token
score = mtp_decode_speedup          (decode-only; floor >= 1.0)
```

- The denominator authority is the trusted parent: wall time divided by the
  parent-configured decode total.
- The paired serial reference is the trusted serial K=1 target decode
  (the `mtp-probe` path) over the same golden, measured in the same session
  behind the same thermal gate on the same box — the same pairing discipline
  the serial track uses, so host drift cancels.
- Decode-only: the MTP protocol charges seed prefill inside the decode
  measurement, so there is no separately scored prefill component.
- The floor is 1.0: an MTP submission that is not actually faster than
  serial decode does not rank. The unmodified reference implementation
  measures about 1.2-1.3x on the M5, so the track starts with visible
  headroom rather than a hard-to-move 1.0.
- Scores publish only from at least three accepted thermal-gated pairs per
  session with alternating candidate/reference order; single-pair scores
  are not publishable because a bounded one-time seed-prefill first-touch
  (~1.5s run-to-run) still needs multi-pair averaging (see "First-block
  stall: root cause and fix").
- The leaderboard namespace is `laguna-xs-2.1-mtp-v1`, fully separate from
  the archived serial configuration; `official_scoring_enabled` flips to
  true only after the hidden Laguna-target goldens and M5 floors are frozen
  (see the go-live runbook).

## Scoring and rebaseline contract

### M5 parity incident and fix

The expanded public/synthetic pilot found a deterministic prose mismatch at
decode step 48. Safe differential replay established:

- the failing block was round 13 at target offset 560;
- it rejected at its first draft position;
- every logical target cache offset was 561 after that position;
- an independent serial shadow cache had the same offsets;
- real and serial physical K/V first diverged at layer 1 immediately after the
  first K-row full-acceptance block (global decode step 4);
- fast-vs-library target and combined-vs-split cache variants all failed at the
  same step.

The root cause was shape-dependent floating-point target execution, not prompt
lookup, assistant incompatibility, host offset arithmetic, or a missing trim.
An ordinary batched K-row target graph is mathematically causal but is not
bit-identical to K serial target graphs on this runtime. The first fix
verified supplied target rows using the exact serial K=1 graph. The current
default replaces the ordinary batched graph with dedicated exact-pair kernels
that keep each row's K=1 accumulation order while sharing packed weight
traversal; the serial K=1 verifier is retained as the explicit control mode.
Runtime-gated parity tests preserve the failing seam: exact logits, pre-norm
hidden bits, and every layer's physical K/V bytes are compared against two
serial forwards at multiple offsets, across forced zero/partial/full and
second-segment acceptance, and across the deep growth/wrap boundaries.

### Prior K=1-control public/synthetic matrix

The corrected serial-equivalent verifier completed 12 thermal-gated M5 pairs:
three each for copy (N=255), prose (N=256), code (N=257), and reasoning
(N=256). All target tokens matched. The matrix exercised full, partial, and
zero draft acceptance plus one-token tails. Mean paired candidate/serial
speedups were 0.741x copy, 0.798x prose, 0.860x code, and 0.883x reasoning;
after the complete fixed K=1 warmup, a fresh copy triplet measured 0.860x.
That K=1 verifier remains the explicit correctness control; these numbers do
not describe the exact-pair implementation.

Per-category paired CV was initially 7.8% copy, 1.9% prose, 4.0% code, and
1.1% reasoning. Phase diagnostics showed copy's variation came from seed and
first-block onset (up to 1.77s) while median block latency and thermal/frequency
telemetry stayed stable. The fixed BOS warmup had not guaranteed all four exact
target rows were compiled when the assistant rejected early. Explicitly warming
four deterministic K=1 target rows reduced a fresh three-pair copy run to 3.2%
paired CV without retaining allocator buffers or using prompt-specific work.
Across accepted runs, maximum temperature was 56.1C and minimum steady GPU
frequency was 1604 MHz.

### Exact-pair public/synthetic matrices

The exact-pair verifier ran the same balanced thermal-gated protocol
(paired `mtp-probe` serial control versus `mtp-benchmark`, K=4, alternating
order, 512-token seeds, N=255/256/257) with every token matching the serial
oracle and every cache-offset invariant passing:

- First matrix, 12 pairs (3 per category): mean paired speedup 1.34x copy,
  1.27x prose, 1.33x code, 1.26x reasoning; overall mean 1.30x,
  ratio-of-means 1.28x.
- Extended matrix, 20 pairs (5 per category) after acceptance-independent
  pair warmup: mean 1.32x copy, 1.37x prose, 1.33x code, 1.30x reasoning;
  overall mean 1.33x, ratio-of-means 1.31x; slowest single pair 1.04x.
- Serial-control seconds per token were unchanged from the pre-exact-pair
  baseline (about 0.03425 s/token), so the generalized boundary kernels did
  not regress the K=1 path.

Every category mean and median clears 1.10x by a wide margin. The remaining
honesty gap is run-to-run spread: per-category paired CV ranged 2.4-13.1%
(first matrix) and 6.3-14.0% (extended matrix). The spread comes from
sporadic single-block stalls up to ~1.7s on the MTP side while p50 block
latency stays at ~75ms and telemetry stays clean (max loaded temperature
58.6C, minimum steady frequency 1612 MHz, peak RSS 47.5 GiB); no outliers
were deleted. The stalls are not explained by kernel compilation (fixed
warmup precedes the timer), thermal throttling, or frequency drops, and they
appear in a minority of fresh processes. Until their source is isolated, the
paired-CV<=5% pilot criterion is met only per-run-set, not universally, and
scores from single pairs must not be quoted. (This "sporadic ~1.7s stall" was
later root-caused as a deterministic block-0 first-touch after the trusted
allocator clear and fixed; see "First-block stall: root cause and fix" below.
The residual variance is a bounded one-time seed-prefill first-touch.)

### Committed-tip revalidation matrix (commit e71f2a9)

The consolidated committed tip (separate MTP-only pair boundary kernels,
serial K=1 boundary kernels restored byte-identical to the base; audit
counter hardening) was rerun on the freed dev box (`m5-max-128gb-1`, drained
from ranked serving) as a fresh 12-pair thermal-gated matrix, K=4,
alternating order, 512-token seeds, N=255/256/257:

- All four model-backed parity gates passed against the pinned IT target and
  assistant (artifact validation, basic pair bit-for-bit, forced
  zero/partial/full/second-segment acceptance seams, deep growth/wrap).
- Every returned token matched the serial oracle across all 24 runs
  (`parity_all_ok=true`), and every cache-offset invariant held.
- Mean paired decode speedup: 1.29x copy, 1.22x prose, 1.23x code,
  1.23x reasoning; overall mean 1.24x, median 1.29x, slowest single pair
  1.03x, fastest 1.42x.
- Serial-control seconds per token stayed at ~0.03425 s/token, confirming the
  boundary-kernel split did not regress the K=1 path.
- Telemetry clean: max loaded temperature 56.1C, minimum steady frequency
  1612 MHz, peak process RSS 47.5 GiB.

### First-block stall: root cause and fix

The "sporadic ~1.7s single-block stall" was root-caused on the dev box and
fixed. Per-block parent-side tracing showed it was not sporadic: it is
deterministic on **block 0**, the first timed decode block, with a magnitude
that varies 0.6-1.6s run to run (which is why per-run `max_block` sometimes
looked like a mild 0.15-0.6s and sometimes 1.6s). Every later block is a flat
~75ms.

Localization:

- An in-process driver (real drafter, real exact-pair verify, identical
  warmup, but no IPC and no worker sandbox) never stalled: 396 blocks across
  6 fresh processes held max 130ms once and ~77ms otherwise, flat allocator
  memory. That ruled out block compute, the allocator growth path, kernel
  JIT, and shape warmup coverage.
- The real parent+worker path stalled only on block 0 (569-1592ms across
  5 runs). Toggling the worker Seatbelt sandbox on/off made no difference,
  ruling out the sandbox.
- The one thing the worker does that the in-process driver did not is
  `resetRuntimeWorkerAllocatorForPhaseStart()` at `mtp_decode_begin`, the
  trusted anti-subsidy control that `clearCache()`es every free buffer the
  untimed warmup allocated. The worker-internal split confirmed block 0's
  cost is in the exact-pair `verify` `eval` (515-888ms) with host-side
  bookkeeping ~0ms and the allocator free-cache dropping right before it: the
  first timed exact-pair forward pays a one-time Metal buffer first-touch for
  the whole 60-layer working set the clear had freed.

Fix (`Gemma4TrainedMTPBlockSession.warmWorkingSetAfterAllocatorReset`, called
by the worker after the trusted clear and before `begin`): re-touch the
drafter and exact-pair working set on throwaway BOS state so those buffers are
resident in the free pool before the first real block. It is charged (inside
the timed window), input-independent (runs before any seed is applied, on
throwaway caches, never touching the real session cache/hidden/shared-KV or
any committed token), and only warms the allocator — the pinned pipelines were
already compiled during untimed init. The trusted `clearCache()` is unchanged;
the serial path and the #586 boundary are untouched. It skips the redundant
512-token prefill warm because `begin`'s charged seed prefill reallocates that
same working set immediately after.

Result on the dev box (real `mtp-benchmark`, sandbox on): block 0 drops from
0.6-1.6s to ~74-79ms, `max_block` from ~1.6s to <=79ms, every block uniform,
token parity preserved (`all_tokens_matched=true`) in all runs, and mean
decode spt improved slightly (the redundant prefill was removed). The one-time
first-touch cost is not eliminated — it cannot be while the trusted clear
stands — but it is relocated into the already-charged seed-prefill phase and
made bounded, so the per-block decode distribution is now trustworthy for
single-pass MTP timing. A residual run-to-run variation of ~1.5s in that
one-time seed-prefill first-touch remains (a genuine Metal residency effect,
not thermal/frequency, and present with or without inter-run cooldown); the
multi-pair alternating-order protocol below averages it out, so single-pair
totals still must not be quoted from one pair.

Phase 1 emits diagnostic JSON with no `score` or `speedup`.

Before publication, organizers must:

1. Freeze an IT-target public and hidden correctness set and serial oracle.
2. Freeze a trusted MTP reference implementation at the same target,
   assistant, block size, prompt, token count, sandbox, and thermal policy.
3. Measure candidate and reference back-to-back on the ranked box.
4. Adopt the decode-only paired score in the contract's `proposed_scoring`
   (or override it), then calibrate the component floor from fresh gated
   sessions.
5. Add MTP-specific behavior and parity gates.
6. Enable the track only through a distinct workflow/track ID, and provision
   the target + assistant on the ranked box (they currently live only on the
   dev box cache).
7. The first-block stall is fixed (see "First-block stall: root cause and
   fix"). A bounded one-time seed-prefill first-touch (~1.5s run-to-run)
   remains, so still require the multi-pair alternating-order protocol above
   before quoting single-pair scores.

Items 1–7 were completed for the previous (Gemma) pin and must be repeated
for the Laguna re-pin: the hidden goldens, serial oracle, floor calibration,
weight manifests, and box provisioning all must be regenerated on m5-bench
with the real Laguna weights before `official_scoring_enabled` flips back to
true. The default workflow remains
`.github/workflows/benchmark.yml`. The
historical operator procedure is retained in
`docs/mtp-track-golive-runbook.md`.

The archived serial configuration remains separate from MTP results.

## Local/operator workflow

Provisioning (large download; do not run merely to compile the prototype):

```bash
MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh
./setup-mtp.sh
eval "$(./setup-mtp.sh --print-paths)"

.build/release/mlxfast-swift transform \
  --reference "${MLXFAST_MTP_TARGET_DIR}" \
  --output mtp-weights
```

Generate a target-matched public oracle on the M5 from a prompt of at least
512 tokens:

```bash
.build/release/mlxfast-swift generate-golden \
  --prompt-file correctness_prompts/public_longcopy_gate_english.txt \
  --weights mtp-weights \
  --tokenizer "${MLXFAST_MTP_TARGET_DIR}" \
  --output /tmp/laguna-xs-2.1-mtp-public.json \
  --name laguna-xs-2.1-mtp-public \
  --steps 129
```

Run the serial block control first:

```bash
.build/release/mlxfast-swift mtp-probe \
  --weights mtp-weights \
  --golden /tmp/laguna-xs-2.1-mtp-public.json \
  --block-size 4 \
  --tokens 128
```

Then run the trained-assistant prototype:

```bash
.build/release/mlxfast-swift mtp-benchmark \
  --target-source "${MLXFAST_MTP_TARGET_DIR}" \
  --weights mtp-weights \
  --assistant "${MLXFAST_MTP_ASSISTANT_DIR}" \
  --contract fixtures/laguna_xs_2_1_mtp_track.json \
  --golden /tmp/laguna-xs-2.1-mtp-public.json \
  --block-size 4 \
  --tokens 128 \
  --target-verification exact-pair \
  --require-trained-assistant
```

`--tokens` defaults to 128 and accepts `1...1536` (the trusted-parent cap;
ranked timed decode stays fixed at 512); the selected golden must
contain one seed token plus the requested number of decode tokens. The parent
owns this total, validates every token, and uses it as the fixed denominator.
`--block-size` accepts `2...4`. A one-position block is used only internally
when the fixed parent-owned decode total leaves a final target-only tail; the
trained-assistant command rejects configured `K=1` because it would never
draft.

For the opt-in model-backed validation, run all four runtime gates — the
artifact gate, the basic pair-parity gate, the forced acceptance-seam gate,
and the deep growth/wrap gate:

```bash
MLXFAST_RUN_MTP_RUNTIME_TESTS=1 \
MLXFAST_MTP_WEIGHTS_PATH="${PWD}/mtp-weights" \
MLXFAST_MTP_TARGET_DIR="${MLXFAST_MTP_TARGET_DIR}" \
MLXFAST_MTP_ASSISTANT_DIR="${MLXFAST_MTP_ASSISTANT_DIR}" \
swift test -c release --filter trainedMTPArtifactValidationRuntimeGate

MLXFAST_RUN_MTP_EXACT_PAIR_TESTS=1 \
MLXFAST_MTP_WEIGHTS_PATH="${PWD}/mtp-weights" \
MLXFAST_MTP_PAIR_GOLDEN_PATH=/tmp/laguna-xs-2.1-mtp-public.json \
swift test --filter exactPairRuntimeMatchesTwoSerialRowsBitForBit

MLXFAST_RUN_MTP_EXACT_PAIR_TESTS=1 \
MLXFAST_MTP_WEIGHTS_PATH="${PWD}/mtp-weights" \
MLXFAST_MTP_ASSISTANT_DIR="${MLXFAST_MTP_ASSISTANT_DIR}" \
MLXFAST_MTP_PAIR_GOLDEN_PATH=/tmp/laguna-xs-2.1-mtp-public.json \
swift test --filter exactPairSessionForcedAcceptanceSeamsMatchSerial

MLXFAST_RUN_MTP_EXACT_PAIR_TESTS=1 \
MLXFAST_RUN_MTP_DEEP_OFFSET_TESTS=1 \
MLXFAST_MTP_WEIGHTS_PATH="${PWD}/mtp-weights" \
swift test --filter exactPairDeepOffsetsGrowthAndWrapMatchSerial
```

Do not claim model-backed success unless all of these commands run with the
pinned weights and an IT-target oracle. CI cannot run them (no weights on
hosted runners); any official MTP workflow must run the full set on the
benchmark host before a kernel change ships.

## Rollout blockers and residual risks

- Public/synthetic IT-target M5 goldens have been exercised remotely, but no
  IT hidden behavior suite is checked in.
- No paired MTP reference baseline or component floors exist.
- The exact-pair kernels amortize dense weight traversal for two rows at a
  time; they are proven bit-identical by runtime gates, not by construction.
  Any future kernel change must rerun the exact-pair parity gates before it
  can ship, and the serial K=1 mode remains the fail-closed control.
- Decodes whose sliding offset crosses the sliding-window wrap (512
  positions for Laguna) lose
  pair eligibility and serialize the remaining rows; with a 512-token window
  the wrap occurs earlier than the old 1,024-window pin, so the Laguna port
  must re-verify wrap-seam behavior on m5-bench.
- The path-based upstream loader cannot atomically bind open descriptors;
  before/after hash validation plus read-only operator ownership mitigates
  TOCTOU, but official provisioning must enforce ownership and permissions.
- The parent can prove returned-token parity, bounds, nonce/ID behavior, wall
  time, and denominator. It cannot directly prove that editable model code
  internally used the assistant, ran target verification, or rolled every
  physical buffer correctly when outputs still match. Static review,
  prompt-independent hidden tests, cache-offset checks, memory telemetry, and
  manual frontier audit are still required.
- Seatbelt does not prove absence of already-running background GPU work.
  Fresh workers, allocator reset, no prompt before the timer, process reaping,
  and operator telemetry are required; active-allocation accounting remains a
  rollout item.
