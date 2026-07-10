#!/usr/bin/env bash
# Validate score and integrity artifacts before upload.
set -euo pipefail

SCORE_PATH="${MLXFAST_SCORE_PATH:-score.json}"
INTEGRITY_PATH="${MLXFAST_INTEGRITY_PATH:-benchmark-integrity.json}"
GOLDEN_PATH="${MLXFAST_CORRECTNESS_GOLDEN_PATH:-correctness_golden.json}"

require_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "::error file=${path}::required benchmark artifact is missing or empty" >&2
    exit 1
  fi
}

require_file "${SCORE_PATH}"
require_file "${SCORE_PATH}.sha256"
require_file "${INTEGRITY_PATH}"
require_file "${GOLDEN_PATH}.sha256"
require_file "${GOLDEN_PATH}.bytes"

: "${MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256:?MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256 is required}"
: "${MLXFAST_EXPECTED_CORRECTNESS_STEPS:?MLXFAST_EXPECTED_CORRECTNESS_STEPS is required}"
: "${MLXFAST_EXPECTED_CORRECTNESS_CASES:?MLXFAST_EXPECTED_CORRECTNESS_CASES is required}"
: "${MLXFAST_EXPECTED_CORRECTNESS_CHECKED_STEPS:?MLXFAST_EXPECTED_CORRECTNESS_CHECKED_STEPS is required}"
: "${MLXFAST_GPQA_TTFT_CASE_COUNT:?MLXFAST_GPQA_TTFT_CASE_COUNT is required}"
: "${MLXFAST_SEMANTIC_GPQA_CASE_COUNT:?MLXFAST_SEMANTIC_GPQA_CASE_COUNT is required}"
: "${MLXFAST_SEMANTIC_GPQA_MIN_PASS:?MLXFAST_SEMANTIC_GPQA_MIN_PASS is required}"
: "${MLXFAST_EXPECTED_COMMIT:?MLXFAST_EXPECTED_COMMIT is required}"
if [[ ! "${MLXFAST_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "::error::MLXFAST_EXPECTED_COMMIT must be a full lowercase commit SHA" >&2
  exit 1
fi

case "${MLXFAST_SEMANTIC_GPQA_REQUIRED:-1}" in
  1|true|TRUE|yes|YES)
    semantic_required=1
    ;;
  0|false|FALSE|no|NO|"")
    semantic_required=0
    ;;
  *)
    echo "::error::MLXFAST_SEMANTIC_GPQA_REQUIRED must be boolean-like" >&2
    exit 1
    ;;
esac

shasum -a 256 -c "${SCORE_PATH}.sha256"

score_hash="$(shasum -a 256 "${SCORE_PATH}" | awk '{print $1}')"
integrity_score_hash="$(jq -r '.score_sha256 // empty' "${INTEGRITY_PATH}")"
if [[ "${integrity_score_hash}" != "${score_hash}" ]]; then
  echo "::error file=${INTEGRITY_PATH}::integrity score hash does not match ${SCORE_PATH}" >&2
  exit 1
fi

golden_hash="$(awk '{print $1}' "${GOLDEN_PATH}.sha256")"
if [[ "${golden_hash}" != "${MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256}" ]]; then
  echo "::error file=${GOLDEN_PATH}.sha256::golden hash mismatch" >&2
  exit 1
fi

jq -e \
  --arg golden_hash "${MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256}" \
  --argjson checked_steps "${MLXFAST_EXPECTED_CORRECTNESS_CHECKED_STEPS}" \
  --argjson correctness_cases "${MLXFAST_EXPECTED_CORRECTNESS_CASES}" \
  --argjson ttft_cases "${MLXFAST_GPQA_TTFT_CASE_COUNT}" \
  --argjson semantic_cases "${MLXFAST_SEMANTIC_GPQA_CASE_COUNT}" \
  --argjson semantic_min_pass "${MLXFAST_SEMANTIC_GPQA_MIN_PASS}" \
  --argjson semantic_required "${semantic_required}" \
  --arg expected_commit "${MLXFAST_EXPECTED_COMMIT}" \
  '
  def same_keys($expected):
    (keys_unsorted | sort) == ($expected | sort);

  .metrics.commit as $commit
  | same_keys(["metrics", "passed", "score"])
  and (.metrics | same_keys([
    "actual_token",
    "bandwidth_gb_per_token",
    "bandwidth_source",
    "baseline_decode_seconds_per_token",
    "baseline_prefill_seconds_per_token",
    "benchmark_wall_seconds",
    "case_count",
    "checked_steps",
    "commit",
    "correctness_seconds",
    "decode_seconds_per_token",
    "decode_speedup",
    "decode_speedup_floor",
    "error",
    "expected_token",
    "expert_bytes_read",
    "expert_cache_evictions",
    "expert_cache_hits",
    "expert_cache_misses",
    "expert_hit_rate",
    "expert_peak_cached_tensors",
    "expert_read_seconds",
    "first_failing_case",
    "first_failing_layer",
    "first_failing_step",
    "golden_hash",
    "gpqa_ttft_case_count",
    "gpqa_ttft_max_seconds",
    "gpqa_ttft_p50_seconds",
    "gpqa_ttft_pass_count",
    "gpqa_ttft_passed",
    "gpqa_ttft_seconds",
    "gpqa_ttft_source",
    "harness_hash",
    "max_abs_diff",
    "num_layers",
    "partial_result",
    "passed_correctness",
    "peak_ram_gb",
    "passed_decode_speedup_floor",
    "passed_prefill_speedup_floor",
    "prefill_seconds_per_token",
    "prefill_speedup",
    "prefill_speedup_floor",
    "preflight_seconds",
    "process_resident_memory_gb",
    "runtime",
    "semantic_gpqa_case_count",
    "semantic_gpqa_model",
    "semantic_gpqa_pass_count",
    "semantic_gpqa_passed",
    "timed_benchmark_seconds",
    "timestamp",
    "weights_byte_count",
    "weights_file_count",
    "weights_hash"
  ]))
  and .passed == true
  and (.score | type == "number")
  and (.score > 0)
  and (.metrics.partial_result == false)
  and (.metrics.passed_correctness == true)
  and (.metrics.checked_steps == $checked_steps)
  and (.metrics.case_count == $correctness_cases)
  and (.metrics.gpqa_ttft_passed == true)
  and (.metrics.gpqa_ttft_case_count == $ttft_cases)
  and (.metrics.gpqa_ttft_pass_count == .metrics.gpqa_ttft_case_count)
  and (.metrics.gpqa_ttft_seconds | type == "number")
  and (.metrics.gpqa_ttft_seconds > 0)
  and (.metrics.gpqa_ttft_p50_seconds | type == "number")
  and (.metrics.gpqa_ttft_p50_seconds > 0)
  and (.metrics.gpqa_ttft_max_seconds | type == "number")
  and (.metrics.gpqa_ttft_max_seconds >= .metrics.gpqa_ttft_p50_seconds)
  and (.metrics.gpqa_ttft_source == "hidden_gpqa_first_token")
  and (.metrics.semantic_gpqa_passed | type == "boolean")
  and (.metrics.semantic_gpqa_case_count == $semantic_cases)
  and (.metrics.semantic_gpqa_pass_count | type == "number")
  and (.metrics.semantic_gpqa_pass_count <= .metrics.semantic_gpqa_case_count)
  and (if $semantic_required == 1 then
    (.metrics.semantic_gpqa_passed == true)
    and (.metrics.semantic_gpqa_pass_count >= $semantic_min_pass)
  else true end)
  and (.metrics.semantic_gpqa_model | type == "string")
  and (.metrics.semantic_gpqa_model | length > 0)
  and (.metrics.num_layers == 60)
  and (.metrics.golden_hash == $golden_hash)
  and (.metrics.first_failing_case == null)
  and (.metrics.first_failing_layer == null)
  and (.metrics.first_failing_step == null)
  and (.metrics.expected_token == null)
  and (.metrics.actual_token == null)
  and (.metrics.decode_seconds_per_token | type == "number")
  and (.metrics.decode_seconds_per_token > 0)
  and (.metrics.prefill_seconds_per_token | type == "number")
  and (.metrics.prefill_seconds_per_token > 0)
  and (.metrics.baseline_decode_seconds_per_token | type == "number")
  and (.metrics.baseline_decode_seconds_per_token > 0)
  and (.metrics.baseline_prefill_seconds_per_token | type == "number")
  and (.metrics.baseline_prefill_seconds_per_token > 0)
  and (.metrics.decode_speedup | type == "number")
  and (.metrics.decode_speedup > 0)
  and (.metrics.decode_speedup_floor | type == "number")
  and (.metrics.decode_speedup_floor == 0.95)
  and (.metrics.passed_decode_speedup_floor == true)
  and (.metrics.decode_speedup >= .metrics.decode_speedup_floor)
  and (.metrics.prefill_speedup | type == "number")
  and (.metrics.prefill_speedup > 0)
  and (.metrics.prefill_speedup_floor | type == "number")
  and (.metrics.prefill_speedup_floor == 0.95)
  and (.metrics.passed_prefill_speedup_floor == true)
  and (.metrics.prefill_speedup >= .metrics.prefill_speedup_floor)
  and (.metrics.correctness_seconds | type == "number")
  and (.metrics.correctness_seconds > 0)
  and (.metrics.timed_benchmark_seconds | type == "number")
  and (.metrics.timed_benchmark_seconds > 0)
  and (.metrics.benchmark_wall_seconds | type == "number")
  and (.metrics.benchmark_wall_seconds >= .metrics.timed_benchmark_seconds)
  and (.metrics.weights_hash | test("^[0-9a-f]{64}$"))
  and (.metrics.weights_file_count | type == "number")
  and (.metrics.weights_file_count > 0)
  and (.metrics.weights_byte_count | type == "number")
  and (.metrics.weights_byte_count > 0)
  and (.metrics.bandwidth_source == "ram_resident_model")
  and (.metrics.error == "")
  and ($commit | test("^[0-9a-f]{7,40}$"))
  and ($expected_commit | startswith($commit))
  and (.metrics.harness_hash | test("^[0-9a-f]{64}$"))
  and (.metrics.timestamp | test("^[0-9]{4}-[0-9]{2}-[0-9]{2}T"))
  and (.metrics.runtime == "swift")
  ' "${SCORE_PATH}" >/dev/null

jq -e '
  def same_keys($expected):
    (keys_unsorted | sort) == ($expected | sort);

  same_keys([
    "golden_path",
    "golden_sha256",
    "score_path",
    "score_sha256",
    "transform_source_sha256",
    "weights_byte_count",
    "weights_file_count",
    "weights_path",
    "weights_sha256"
  ])
  and (.score_sha256 | test("^[0-9a-f]{64}$"))
  and (.golden_sha256 | test("^[0-9a-f]{64}$"))
  and (.weights_sha256 | test("^[0-9a-f]{64}$"))
  and (.transform_source_sha256 | test("^[0-9a-f]{64}$"))
  and (.weights_file_count | type == "number")
  and (.weights_file_count > 0)
  and (.weights_byte_count | type == "number")
  and (.weights_byte_count > 0)
  ' "${INTEGRITY_PATH}" >/dev/null

echo "benchmark: validated score and integrity artifacts"
