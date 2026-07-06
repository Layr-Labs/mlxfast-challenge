# Gemma Migration: R2 / Private Artifact Checklist

The DeepSeek V4 Flash to Gemma 4 31B 4-bit migration changed the model,
tokenizer, and layer count, so every private artifact that embeds prompt
tokens, expected tokens, or model-derived calibration is now
model-mismatched. This file is the consolidated operator checklist; each
in-tree consumption site carries a matching greppable marker:

```bash
rg -n "TODO\(gemma-r2\)|TODO\(gemma-golden\)"
```

Nothing here changes the R2 secret/env plumbing (`R2_ACCESS_KEY_ID`,
`R2_BUCKET_ENDPOINT`, `R2_SECRET_ACCESS_KEY` on the
`benchmark-private-prompts` environment) — that structure is intact and
correct. Only the *contents* of the private objects (and the pins that
verify them) must be regenerated.

## 1. Hidden correctness/benchmark golden (R2) — `TODO(gemma-r2)`

- **Object:** `correctness_prompts/golden_prompt_benchmark_transcription_gate_english_512_256.json`
- **Contents to regenerate:** 512-token base prompt retokenized with the
  Gemma tokenizer; 64 teacher-forced expected continuation tokens from the
  trusted Gemma 4 31B 4-bit reference; hidden `correctness_gates` (anchors,
  free-run prefixes, behavior cases); and the benchmark oracle (prefill
  next token, 512-token decode seed next token, 128 timed decode tokens;
  optionally per-prompt `baseline_*_seconds_per_token` calibration).
- **Consumed by:**
  - `.github/workflows/benchmark-correctness-slice.yml` — "Download and
    verify hidden correctness golden" (all three slice machines, raw
    pre-GPQA form).
  - `.github/workflows/benchmark-timing-or-gates.yml` — "Prepare
    correctness golden" (timing and gates machines; gates additionally
    augments it with GPQA gates via `attach-gpqa-gates`).
  - `.github/workflows/benchmark.yml` — the `correctness-only` job declares
    the object path/pins in its env (defaults; the job itself runs on the
    public golden).
- **Pins to update after upload:** `MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256`
  (currently `8306702...bdbf067`) and `MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_BYTES`
  (currently `26110`) in `benchmark.yml` and
  `benchmark-correctness-slice.yml`.

## 2. Hidden GPQA reference cases (R2) — `TODO(gemma-r2)`

- **Object:** `correctness_prompts/gpqa_reference_cases.json`
- **Contents to regenerate:** 5 token-budget-valid GPQA multiple-choice
  prompts with `accepted_token_sequences` / `accepted_responses` captured
  from the Gemma reference model (Gemma tokenizer, `max_new_tokens=10`),
  plus the private reference answers the semantic judge compares against.
- **Consumed by:** `.github/workflows/benchmark-timing-or-gates.yml` —
  "Prepare correctness golden" (`attach-gpqa-gates`), which drives the
  hidden GPQA behavior gates, the TTFT guardrail, and the semantic-GPQA
  answer capture judged by `run-semantic-gpqa-gate.sh`.
- **No hash pin:** the augmented golden's hash/bytes are computed at run
  time, so no workflow constant needs updating for this object itself.
- **Recalibrate:** the semantic-GPQA 3/5 threshold
  (`MLXFAST_SEMANTIC_GPQA_MIN_PASS`) was calibrated against the previous
  model's baseline answer quality; confirm it against an unmodified Gemma
  baseline once the new cases exist.

## 3. Private prompt manifest (organizer-side, not workflow-consumed)

- The manifest of hidden prompt sources used to regenerate goldens offline
  (see `docs/private-benchmark-security.md`). It is never downloaded by the
  workflows, but the organizer's offline regeneration pipeline must switch
  to the Gemma tokenizer/reference before producing items 1 and 2.

## 4. Public fixtures (checked in) — DONE

- `correctness_prompts/public_longcopy_gate_english_512_256.json` and
  `correctness_prompts/public_longcopy_gate_english_512_1024.json` have been
  regenerated against the Gemma 4 31B 4-bit reference with
  `mlxfast-swift generate-golden` (Gemma-tokenized 512-token prompt, greedy
  reference continuations; the 256 fixture is a greedy prefix of the 1024
  one). The pins that verify them were moved in the same change:
  - `MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_SHA256` / `..._BYTES` in
    `.github/workflows/benchmark.yml` (`correctness-only` env).
  - `public_golden_sha256` in the "Public behavior gate" step of
    `.github/workflows/benchmark-timing-or-gates.yml`.

## 5. Paired-baseline ref and calibrated constants — DONE

- The timing machine's pinned paired-baseline ref now points at the Gemma
  migration merge (`eff7e7f2c85a5a6cef11110442ba4624a6ab3986`), and the
  calibrated constants were re-measured against that exact commit on the
  official Blacksmith runner class (`gemma-baseline-timing-probe` run
  28809531890, 2026-07-06; a dispatch-only, secret-free timing fan-out of
  unmodified `main` over the full official 128-step timing path). The same
  change updated `officialBaselinePrefillSecondsPerToken` /
  `officialBaselineDecodeSecondsPerToken` in
  `Sources/MLXFastCore/Constants.swift`, the `MLXFAST_PAIRED_SANITY_PREFILL`
  / `MLXFAST_PAIRED_SANITY_DECODE` anchors in
  `benchmark-timing-or-gates.yml`, `docs/benchmark-window-freeze.md`,
  `README.md`, `TASK.md`, and
  `Tests/MLXFastTests/BenchmarkWindowFreezeTests.swift`.
- Note: the ranked timing job's own end-to-end verification (a green
  paired-baseline step inside `benchmark.yml`) still depends on items 1 and
  2 above — the paired step measures the reference against the hidden R2
  golden, which is still DeepSeek-tokenized.
