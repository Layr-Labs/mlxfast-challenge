#!/usr/bin/env bash
# Emit a REDACTED failure record for a failed benchmark job, safe for public
# artifact upload.
#
# A failing score.json must never be uploaded raw: its free-text error and
# first_failing/expected/actual token fields can carry hidden-golden token
# values, and errors thrown inside the sandboxed worker can carry arbitrary
# submitted-code-controlled text (an exfiltration channel for anything the
# model code observed). Until now the failing score.json simply died with the
# ephemeral runner, so a floor failure, a decode-oracle token mismatch, and a
# harness crash all surfaced identically as "benchmark_failed" with no way to
# tell them apart afterwards.
#
# This trusted script closes that observability gap by deriving only:
#   - a failure CATEGORY matched against harness-authored error prefixes,
#   - harness-authored booleans (passed flags, partial_result),
#   - already-coarsened wall seconds,
#   - the first failing decode step floored to a coarse bucket of 32.
# The raw error string is never printed, uploaded, or partially quoted.
set -euo pipefail

out="${1:?usage: redact-benchmark-failure.sh OUTPUT_PATH}"
score_path="${2:-score.json}"

category="no_score"
step_bucket="null"
passed="null"
passed_correctness="null"
passed_decode_floor="null"
passed_prefill_floor="null"
partial_result="null"
benchmark_wall_seconds="null"
timed_benchmark_seconds="null"

if [[ -s "${score_path}" ]]; then
  if ! jq -e 'type == "object" and (.metrics | type == "object")' "${score_path}" >/dev/null 2>&1; then
    category="invalid_score"
  else
    # Plain field access (no `//` fallback): jq's alternative operator treats
    # false as empty, which would silently rewrite `false` booleans to the
    # fallback. Missing keys already come back as literal `null`.
    #
    # Every field below is additionally coerced to its expected scalar type
    # (booleans -> boolean-or-null, numbers -> number-or-null) before it is
    # embedded in the redacted record. score.json here is the trusted sealed
    # score, so these fields are already well-typed on a legitimate run and the
    # coercion is a no-op; the guard is defense in depth so that a
    # non-scalar/attacker-shaped value (object, array, or string) in any of
    # these fields can never be passed through --argjson verbatim into the
    # uploaded artifact as a covert exfiltration channel. The raw error string
    # is still never emitted; it is only matched against fixed prefixes below.
    error_text="$(jq -r 'if (.metrics.error | type) == "string" then .metrics.error else "" end' "${score_path}")"
    passed="$(jq 'if (.passed | type) == "boolean" then .passed else null end' "${score_path}")"
    passed_correctness="$(jq 'if (.metrics.passed_correctness | type) == "boolean" then .metrics.passed_correctness else null end' "${score_path}")"
    passed_decode_floor="$(jq 'if (.metrics.passed_decode_speedup_floor | type) == "boolean" then .metrics.passed_decode_speedup_floor else null end' "${score_path}")"
    passed_prefill_floor="$(jq 'if (.metrics.passed_prefill_speedup_floor | type) == "boolean" then .metrics.passed_prefill_speedup_floor else null end' "${score_path}")"
    partial_result="$(jq 'if (.metrics.partial_result | type) == "boolean" then .metrics.partial_result else null end' "${score_path}")"
    benchmark_wall_seconds="$(jq 'if (.metrics.benchmark_wall_seconds | type) == "number" then .metrics.benchmark_wall_seconds else null end' "${score_path}")"
    timed_benchmark_seconds="$(jq 'if (.metrics.timed_benchmark_seconds | type) == "number" then .metrics.timed_benchmark_seconds else null end' "${score_path}")"

    # Categories are matched ONLY against fixed prefixes that the trusted
    # harness itself authors (GemmaRuntimeBenchmark). Anything else --
    # including error text that originated inside the sandboxed worker -- is
    # deliberately collapsed to an opaque category.
    if [[ "${error_text}" == "performance floor failed"* ]]; then
      category="floor_failed"
    elif [[ "${error_text}" == "benchmark prefill token mismatch"* ]]; then
      category="prefill_token_mismatch"
    elif [[ "${error_text}" == "benchmark decode seed token mismatch"* ]]; then
      category="decode_seed_token_mismatch"
    elif [[ "${error_text}" == "benchmark decode token mismatch"* ]]; then
      category="decode_token_mismatch"
      first_failing_step="$(jq -r '.metrics.first_failing_step // ""' "${score_path}")"
      if [[ "${first_failing_step}" =~ ^[0-9]+$ ]]; then
        step_bucket="$((first_failing_step / 32 * 32))"
      fi
    # The expert-streaming byte-read plausibility category
    # ("seed_read_implausible") was removed with the dense Gemma migration:
    # the harness no longer authors that error prefix because there is no
    # streaming path to meter. Its threat (hiding timed work outside the
    # measured window) is covered by the single-seed timed decode protocol
    # and the submission static review.
    elif [[ "${passed_correctness}" == "false" ]]; then
      category="correctness_failed"
    elif [[ -n "${error_text}" ]]; then
      category="runtime_error"
    else
      # score.json says passed with no error, yet the job failed: a
      # validation/staging step tripped, not the benchmark itself.
      category="score_present_validation_failed"
    fi
  fi
fi

jq -n \
  --arg category "${category}" \
  --arg mode "${MLXFAST_BENCHMARK_MODE:-unknown}" \
  --argjson step_bucket "${step_bucket}" \
  --argjson passed "${passed}" \
  --argjson passed_correctness "${passed_correctness}" \
  --argjson passed_decode_speedup_floor "${passed_decode_floor}" \
  --argjson passed_prefill_speedup_floor "${passed_prefill_floor}" \
  --argjson partial_result "${partial_result}" \
  --argjson benchmark_wall_seconds "${benchmark_wall_seconds}" \
  --argjson timed_benchmark_seconds "${timed_benchmark_seconds}" \
  '{
    failure_category: $category,
    mode: $mode,
    first_failing_step_bucket: $step_bucket,
    passed: $passed,
    passed_correctness: $passed_correctness,
    passed_decode_speedup_floor: $passed_decode_speedup_floor,
    passed_prefill_speedup_floor: $passed_prefill_speedup_floor,
    partial_result: $partial_result,
    benchmark_wall_seconds: $benchmark_wall_seconds,
    timed_benchmark_seconds: $timed_benchmark_seconds
  }' > "${out}"

echo "benchmark-failure: category=${category} mode=${MLXFAST_BENCHMARK_MODE:-unknown} wall_seconds=${benchmark_wall_seconds}"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "### Benchmark failure (redacted)"
    echo ""
    echo '```json'
    cat "${out}"
    echo '```'
  } >> "${GITHUB_STEP_SUMMARY}"
fi
