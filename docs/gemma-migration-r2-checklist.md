# Gemma Migration: R2 / Private Artifact Checklist

The DeepSeek V4 Flash to Gemma 4 31B 4-bit migration changed the model,
tokenizer, and layer count, so every private artifact that embeds prompt
tokens, expected tokens, or model-derived calibration is now
model-mismatched. This file is the consolidated operator checklist; each
in-tree consumption site carries a matching greppable marker:

```bash
rg -n "TODO\(gemma-r2\)|TODO\(gemma-golden\)|TODO\(gemma-baseline\)"
```

Nothing here changes the R2 secret/env plumbing (`R2_ACCESS_KEY_ID`,
`R2_BUCKET_ENDPOINT`, `R2_SECRET_ACCESS_KEY` on the
`benchmark-private-prompts` environment) — that structure is intact and
correct. Only the *contents* of the private objects (and the pins that
verify them) must be regenerated.

## 1. Hidden correctness/benchmark golden (R2) — regenerated, pending upload

- **Object:** `correctness_prompts/golden_prompt_benchmark_transcription_gate_english_512_256-gemma.json`
- **Regenerated contents:** 512-token base prompt retokenized with the
  Gemma tokenizer; 256 teacher-forced expected continuation tokens from the
  trusted Gemma 4 31B 4-bit reference; a `correctness_gates.free_run` gate
  covering the full timed decode offset range (`attach-free-run-gate`
  defaults, 128 steps); and the benchmark oracle (prefill next token,
  512-token decode seed next token, 256 timed decode tokens). Per-prompt
  `baseline_*_seconds_per_token` calibration is intentionally omitted (the
  previous object also omitted it); scoring falls back to the calibrated
  constants until the Gemma baseline recalibration lands.
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
- **Pins:** `MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256`
  (now `a593138...e92cb25`) and `MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_BYTES`
  (now `38155`) in `benchmark.yml` and
  `benchmark-correctness-slice.yml` match the regenerated object; they take
  effect once the object is uploaded to R2.

## 2. Hidden GPQA reference cases (R2) — regenerated, pending upload

- **Object:** `correctness_prompts/gpqa_reference_cases-gemma.json`
- **Regenerated contents:** the 9 GPQA multiple-choice prompt cases (prompt
  text, answer keys, domains unchanged) with each case's
  `accepted_token_sequences` replaced by the first greedy answer token
  captured from the Gemma 4 31B 4-bit reference (Gemma tokenizer,
  `max_new_tokens=10` generation), preserving the previous artifact's
  one-token-sequence shape; the semantic judge's reference answers continue
  to derive from the answer keys.
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

## 5. Paired-baseline ref and calibrated constants — `TODO(gemma-baseline)`

- The timing machine's pinned paired-baseline ref
  (`7e2191c...` in `benchmark-timing-or-gates.yml`) still builds the
  DeepSeek-era reference implementation and cannot run the Gemma
  checkpoint. After the migration lands on `main`:
  1. Run a trusted Gemma baseline on the official Blacksmith hardware.
  2. Recalibrate `officialBaselinePrefillSecondsPerToken` /
     `officialBaselineDecodeSecondsPerToken` in
     `Sources/MLXFastCore/Constants.swift` (currently stale placeholders).
  3. Repoint the paired-baseline `ref:` to the Gemma baseline commit and
     update `MLXFAST_PAIRED_SANITY_PREFILL` / `MLXFAST_PAIRED_SANITY_DECODE`
     to the new constants.
  4. Update `docs/benchmark-window-freeze.md`, `README.md`, `TASK.md`, and
     `Tests/MLXFastTests/BenchmarkWindowFreezeTests.swift` in the same
     change (the freeze tests fail until doc and code agree).
