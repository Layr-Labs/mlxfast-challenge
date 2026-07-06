# Private Benchmark Security

The private correctness prompts and private golden data must be treated as
trusted harness material. They are not part of the contestant modifiable
surface.

## Required GitHub setup

Store private prompt/golden download credentials only as secrets on the
`benchmark-private-prompts` GitHub Environment. Do not store them as
repository-wide or organization-wide secrets.

Configure the `benchmark-private-prompts` Environment with:

- Deployment branches limited to `main` and `submissions/*` (the refs the
  benchmark orchestrator dispatches). Do not grant fork access.
- Required reviewers for private benchmark runs.
- R2 prompt manifest credentials:
  - `R2_ACCESS_KEY_ID`
  - `R2_BUCKET_ENDPOINT`
  - `R2_SECRET_ACCESS_KEY`

Normal private benchmark runs download the precomputed
`correctness_prompts/golden_prompt_benchmark_transcription_gate_english_512_256.json`
object from the private R2 bucket. Full private benchmark runs also download
`correctness_prompts/gpqa_reference_cases.json` from the same private bucket
and merge it into the local golden as 5 hidden multiple-choice behavior gates.

> **Gemma migration status:** both R2 objects above still hold
> DeepSeek-tokenized content and must be regenerated from the Gemma 4 31B
> 4-bit reference before ranked runs can pass. The secret/credential model in
> this document is unchanged by the migration. See
> `docs/gemma-migration-r2-checklist.md` for the full regeneration checklist.


Each GPQA case must carry accepted reference-model output tokens or responses;
the GPQA answer key alone is not used as an exact-token correctness oracle.
The benchmark workflow treats that file as immutable trusted input and never
regenerates or uploads it. Update it only through an organizer-controlled
offline process, then upload the reviewed JSON to private R2.
The private prompt manifest is only an organizer input for regenerating the
golden outside the benchmark workflow. It should not be written into the
repository workspace, uploaded, or cached. The workflow writes downloaded
private files only under `$RUNNER_TEMP` and uploads only golden hash and
byte-count sidecars after a deny-list check rejects prompt, golden, GPQA, model,
symlink, and oversized artifact paths.

Hidden behavioral correctness cases should use short accepted token prefixes
captured from the official reference model on the official runner. The GPQA
gate checks one generated token per case, which avoids cross-machine drift after
the first answer token while still checking behavior across all 5 hidden
questions. Longer precomputed reference sequences may be kept in private R2; the
workflow uses their stable prefix. During the hidden behavior correctness pass,
the workflow also measures hidden GPQA TTFT from prompt prefill through the
first greedy answer token and fails the run if that first token is not accepted.
Only aggregate TTFT fields are written to `score.json`; they do not contain
prompt text, expected token IDs, generated token IDs, accepted token sets, or
per-case prompt lengths.

The semantic GPQA judge runs after candidate answers are written into the
private runner directory. Only aggregate semantic pass counts and the judge
model name are patched into `score.json`; prompts, references, candidate
answers, and judge transcripts remain private and are covered by artifact
deny-list checks.

The benchmark workflow verifies at runtime (see
`enforce-trusted-benchmark-workflow.sh`) that it runs in this repository via a
`workflow_dispatch` event. It benchmarks whatever ref it is dispatched on.

Because the workflow runs the dispatched ref's own workflow file, the real
boundary is the combination of:

- the benchmark orchestrator (Yukon eigenbot) being the only creator of
  `submissions/*` branches, built from remotely validated `editablePaths` so
  their non-`editablePaths` files match `main`;
- the `Enforce modifiable surface` step re-verifying at runtime that a
  `submissions/*` branch changes only `editablePaths` relative to `main`; and
- restricting who can push `submissions/*` branches and dispatch the workflow.

The `benchmark-private-prompts` Environment deployment-branch policy and required
reviewers remain the gate on private-secret access.

## Parallel job topology

When dispatched with `run_benchmark=true`, the trust boundary above applies
independently to 5 privileged jobs instead of one. Each of the following
separately declares `environment: benchmark-private-prompts` and separately
re-runs the trusted-workflow check, the submission-branch static review, the
modifiable-surface enforcement, and the sandbox probe: `correctness-slice-1`,
`correctness-slice-2`, `correctness-slice-3` (each running the reusable
`benchmark-correctness-slice.yml`), and `benchmark-timing`, `benchmark-gates`
(both running the reusable `benchmark-timing-or-gates.yml`). A sixth job,
`combine`, has no `environment:` gate and no private-secret access — it only
downloads the already-computed, already-authenticated outputs the 5 privileged
jobs uploaded (after each independently validated its own content, see
`benchmark-correctness-slice.yml`'s "Validate correctness slice artifacts" and
`benchmark-timing-or-gates.yml`'s "Validate intermediate benchmark artifact"),
merges them, and re-verifies the combined result before it may be staged or
uploaded. A `validate-slice-ranges` job runs before the three correctness-slice
jobs with no checkout and no secrets at all — it only validates the numeric
range inputs. Dispatching with the default `run_benchmark=false` instead runs
everything on the single `correctness-only` job described elsewhere in this
document. Any future change to this topology that drops the environment gate,
the trusted-workflow check, the static/modifiable-surface guards, or the
per-job content-validation gate from one of the 5 privileged jobs — without an
equivalent guard elsewhere — reopens the exact channels this document
describes.

## Submission flow

The benchmark orchestrator (Yukon eigenbot) creates a `submissions/*` branch that
differs from `main` only in `benchmark.json` `editablePaths`, then dispatches
`benchmark.yml` on that branch. The workflow benchmarks the checked-out branch
directly.

On `submissions/*` branches the workflow additionally:

- runs the static cheat review over the editable files the submission changed
  versus its merge-base with `main` (`run-submission-static-review.sh`;
  unchanged editable files are byte-identical to trusted `main` content and are
  not sent to the judge — only a no-base local/manual run reviews the whole
  editable surface),
- enforces the modifiable surface against `main` (`enforce-modifiable-surface.sh`),
- suppresses submitted correctness/benchmark process logs, and
- uploads correctness artifacts only after validation succeeds.

Maintainers can also dispatch the workflow on `main` (baseline) or a dev branch;
those runs skip the submission-only guards.

## Output policy

For `submissions/*` runs:

- Correctness and benchmark process logs are redirected to private runner temp
  files and are not uploaded.
- The workflow prints only fixed heartbeat lines while submitted code is
  running.
- `score.json`, `benchmark-integrity.json`, and golden hash/byte sidecars are
  uploaded only after strict schema and hash validation succeeds.
- Correctness traces are disabled for full benchmark runs.
- Timed benchmark model execution runs in a child worker process that is denied
  network access and direct reads of the private golden path. Submitted model
  code necessarily sees hidden prompt tokens and teacher-forced previous tokens
  while doing inference, but it does not receive the golden file path. The
  trusted harness validates the resulting `score.json` before upload.

This prevents submitted code from using GitHub logs or uploaded artifacts as a
direct prompt-exfiltration channel.

## Residual channel

Submitted code still participates in inference on hidden prompt tokens. Any
public feedback from that run, including pass/fail, score, timing, or repeated
submission attempts, is a possible low-bandwidth covert channel. The workflow
hardening above blocks direct extraction paths, but competition policy should
still limit repeated private benchmark attempts and avoid exposing per-case
failure details for hidden cases.

No prompt manifest or generated correctness golden should be committed to the
public repository.
