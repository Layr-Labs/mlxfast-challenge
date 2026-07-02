#!/usr/bin/env bash
# Combine results from a 4-machine parallel run into one authoritative score.json:
#   machine1/ - ran `mlxfast-swift benchmark` with MLXFAST_BENCHMARK_CORRECTNESS_STEPS=0
#               (GPQA cases + GPQA TTFT + timed prefill/decode; skips its own base
#               teacher-forced correctness case)
#   machine2/ .. machineN/ - each ran `mlxfast-swift correctness --base-case-only
#               --step-range START-END --step-range-output step-range.json`, covering a
#               disjoint slice of the base correctness case's step window. --base-case-only
#               is required: without it, checked_steps also includes anchors/free-run/
#               behavior/GPQA from the golden file, which is not comparable across
#               machines and would corrupt the coverage check below.
#
# This script is the only place the real, combined correctness verdict is asserted.
# Machine 1's own score.json reports passed_correctness based on nothing but GPQA/TTFT/
# floor checks -- it must never be published as-is. This script ANDs in the base-case
# verdict from every correctness-only machine before anything gets staged or uploaded.
set -euo pipefail

: "${MLXFAST_EXPECTED_CORRECTNESS_STEPS:?MLXFAST_EXPECTED_CORRECTNESS_STEPS is required}"
: "${MLXFAST_COMBINED_SCORE_PATH:=score.combined.json}"

MACHINE1_DIR="${MLXFAST_MACHINE1_DIR:-machine1}"
# read -a splits on whitespace without also glob-expanding the result, unlike an
# unquoted array assignment.
read -ra CORRECTNESS_MACHINE_DIRS <<< "${MLXFAST_CORRECTNESS_MACHINE_DIRS:-machine2 machine3 machine4}"

require_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "::error file=${path}::required combiner input is missing or empty" >&2
    exit 1
  fi
}

require_file "${MACHINE1_DIR}/score.json"
require_file "${MACHINE1_DIR}/weights.sha256"

expected_weights_hash="$(cat "${MACHINE1_DIR}/weights.sha256")"

# --- Tripwire: every machine must have independently transformed byte-identical
# weights/. Since each machine downloads and transforms on its own (no cross-machine
# artifact handoff), a mismatch here means the checkpoint version, submitted commit, or
# transform itself diverged between machines -- treat that as fatal, not a warning.
for dir in "${MACHINE1_DIR}" "${CORRECTNESS_MACHINE_DIRS[@]}"; do
  require_file "${dir}/weights.sha256"
  actual="$(cat "${dir}/weights.sha256")"
  if [[ "${actual}" != "${expected_weights_hash}" ]]; then
    echo "::error file=${dir}/weights.sha256::weights hash mismatch across machines" >&2
    echo "expected (from ${MACHINE1_DIR})=${expected_weights_hash}" >&2
    echo "actual (from ${dir})=${actual}" >&2
    exit 1
  fi
done
echo "combine-parallel-correctness: weights hash agrees across all machines: ${expected_weights_hash}"

# --- Combine the base correctness case's step-range slices. Each machine reports its
# ASSIGNED range (step-range.json, written unconditionally before the check runs) plus
# its own pass/fail/checked_steps (correctness-report.json). The assigned range is what
# coverage is checked against, not checked_steps -- a real failure legitimately stops
# early and reports fewer checked_steps than its assigned slice, and that must not be
# confused with a range that was never assigned to anyone.
total_checked_steps=0
total_base_case_seconds=0
base_case_passed=true
first_failing_step=""
first_failing_dir=""
ranges_file="$(mktemp)"
trap 'rm -f "${ranges_file}"' EXIT

for dir in "${CORRECTNESS_MACHINE_DIRS[@]}"; do
  require_file "${dir}/correctness-report.json"
  require_file "${dir}/step-range.json"
  report="${dir}/correctness-report.json"

  passed="$(jq -r '.passed' "${report}")"
  checked_steps="$(jq -r '.checked_steps' "${report}")"
  failing_step="$(jq -r '.first_failing_step // empty' "${report}")"
  range_start="$(jq -r '.step_range_start' "${dir}/step-range.json")"
  range_end="$(jq -r '.step_range_end' "${dir}/step-range.json")"

  # The serial run's correctness_seconds covered base case + gates together;
  # machine1's own value now only covers gates, so each slice reports its
  # base-case wall seconds in a sidecar and the total is folded back into
  # metrics.correctness_seconds below. Optional (the public probe workflow
  # predates it), but when present it must be a non-negative integer -- a
  # malformed sidecar is a real bug, not something to silently zero.
  if [[ -s "${dir}/slice-timing.json" ]]; then
    slice_seconds="$(jq -r '.slice_seconds' "${dir}/slice-timing.json")"
    if ! [[ "${slice_seconds}" =~ ^[0-9]+$ ]]; then
      echo "::error file=${dir}/slice-timing.json::slice_seconds must be a non-negative integer, got \"${slice_seconds}\"" >&2
      exit 1
    fi
    total_base_case_seconds=$((total_base_case_seconds + slice_seconds))
  fi

  jq -n --arg dir "${dir}" --argjson start "${range_start}" --argjson end "${range_end}" \
    '{dir: $dir, start: $start, end: $end}' >> "${ranges_file}"

  total_checked_steps=$((total_checked_steps + checked_steps))

  if [[ "${passed}" != "true" ]]; then
    base_case_passed=false
    if [[ -n "${failing_step}" ]] \
      && { [[ -z "${first_failing_step}" ]] || (( failing_step < first_failing_step )); }; then
      first_failing_step="${failing_step}"
      first_failing_dir="${dir}"
    fi
    echo "combine-parallel-correctness: ${dir} FAILED at step ${failing_step:-unknown} (assigned [${range_start}, ${range_end}))" >&2
  else
    # A passing slice that didn't actually cover its whole assigned range is a real
    # bug (e.g. --base-case-only omitted so checked_steps includes gate steps, or a
    # mismatched --step-range/--step-range-output pairing) -- catch it here rather
    # than let it silently corrupt the coverage check below.
    assigned_width=$((range_end - range_start))
    if [[ "${checked_steps}" -ne "${assigned_width}" ]]; then
      echo "::error file=${report}::${dir} passed but checked_steps=${checked_steps} does not match its assigned range width ${assigned_width} ([${range_start}, ${range_end}))" >&2
      echo "this usually means --base-case-only was omitted (checked_steps includes gate steps) or --step-range/--step-range-output disagree" >&2
      base_case_passed=false
      first_failing_dir="${dir} (checked_steps/range mismatch)"
    else
      echo "combine-parallel-correctness: ${dir} passed, range=[${range_start}, ${range_end}) checked_steps=${checked_steps}"
    fi
  fi
done

# --- Range coverage check: the ASSIGNED ranges (not checked_steps) must partition
# [0, EXPECTED) with no gaps or overlaps. Summing checked_steps alone cannot catch
# this -- two machines both covering [0, 32) and nobody covering [32, 64) still sums
# to 64 and would otherwise report passed=true despite half the window never having
# been assigned to anyone. Only enforced when every machine's own slice passed; a
# real failure already fails the run regardless of coverage.
if [[ "${base_case_passed}" == "true" ]]; then
  coverage_ok="$(jq -s -r --argjson expected "${MLXFAST_EXPECTED_CORRECTNESS_STEPS}" '
    sort_by(.start) as $sorted
    | ($sorted | length) as $n
    | if $n == 0 then "false"
      elif $sorted[0].start != 0 then "false"
      elif $sorted[$n - 1].end != $expected then "false"
      else ([range(0; $n - 1) | ($sorted[.].end == $sorted[. + 1].start)] | all) | tostring
      end
  ' "${ranges_file}")"
  if [[ "${coverage_ok}" != "true" ]]; then
    echo "::error::assigned step ranges do not exactly partition [0, ${MLXFAST_EXPECTED_CORRECTNESS_STEPS}) -- a gap or overlap exists" >&2
    jq -s -r 'sort_by(.start)[] | "  \(.dir): [\(.start), \(.end))"' "${ranges_file}" >&2
    echo "fix the per-machine --step-range assignment; do not adjust this check" >&2
    base_case_passed=false
    first_failing_dir="range-coverage-check"
  fi
fi

# --- Assemble the final score.json: take machine1's payload (score, GPQA, TTFT, speedup
# floors) and override the correctness fields with the real, combined verdict.
# machine1's own metrics.checked_steps only covers what it actually checked itself
# (anchors/free-run/behavior -- it skipped the base case), so the combined total adds
# the base-case step count on top of that rather than replacing it. correctness_seconds
# gets the same treatment: machine1's value covers gates only, and adding the slices'
# base-case wall seconds restores the serial run's semantics (base case + gates total).
jq -e \
  --argjson base_case_passed "${base_case_passed}" \
  --argjson base_case_checked_steps "${total_checked_steps}" \
  --argjson base_case_seconds "${total_base_case_seconds}" \
  --arg first_failing_step "${first_failing_step}" \
  --arg first_failing_dir "${first_failing_dir}" \
  '
  (if $first_failing_step == "" then null else ($first_failing_step | tonumber) end) as $ffs
  | (.metrics.passed_correctness and $base_case_passed) as $combined_passed_correctness
  | .metrics.passed_correctness = $combined_passed_correctness
  | .metrics.checked_steps = (.metrics.checked_steps + $base_case_checked_steps)
  | .metrics.correctness_seconds = (.metrics.correctness_seconds + $base_case_seconds)
  | .metrics.first_failing_case = (if $base_case_passed then .metrics.first_failing_case else "base_correctness_case (\($first_failing_dir))" end)
  | .metrics.first_failing_step = (if $base_case_passed then .metrics.first_failing_step else $ffs end)
  | .passed = (.passed and $combined_passed_correctness)
  | if .passed then . else (.score = null) end
  ' "${MACHINE1_DIR}/score.json" > "${MLXFAST_COMBINED_SCORE_PATH}"

# Coarsen the diagnostic (non-ranking) real-valued fields in the FINAL published
# score to the same 2 significant figures the Swift harness applies to per-machine
# scores (see roundedToSignificantFigures / withCoarsenedPublicDiagnostics). The
# per-machine artifacts are already coarse, but this combined file is reassembled
# here by jq -- and the merge RE-DERIVES some fields at full precision
# (expert_hit_rate = hits/(hits+misses)) or reintroduces it (correctness_seconds =
# gates + integer slice seconds, expert_read_seconds = gates + timing) -- so
# without this pass the artifact the leaderboard bot publishes would still leak a
# fine-grained timing/memory covert channel that submitted model code can drive.
# Ranking- and floor-critical fields (decode/prefill seconds-per-token, speedups,
# floors, baselines, score) are left untouched so scoring and the downstream
# artifact validator are unaffected. Rounding is monotone, so validator ordering
# checks (wall >= timed, ttft_max >= p50) still hold; re-clamped for safety.
# FIGURES kept in one place; mirrors MLXFastConstants.publicDiagnosticSignificantFigures.
round_tmp="$(mktemp)"
jq '
  def sigfig($n): if (type != "number") then .
    elif . == 0 then 0
    else (fabs | log10 | floor) as $m | pow(10; ($n - 1 - $m)) as $f | (((. * $f) | round) / $f)
    end;
  # Only touch keys that are present and numeric, so partial/legacy inputs
  # (e.g. the combiner probe fixtures) are not mutated or crashed.
  .metrics |= (
    reduce ([
      "peak_ram_gb", "bandwidth_gb_per_token", "preflight_seconds", "correctness_seconds",
      "timed_benchmark_seconds", "benchmark_wall_seconds", "gpqa_ttft_seconds",
      "gpqa_ttft_p50_seconds", "gpqa_ttft_max_seconds", "process_resident_memory_gb",
      "expert_read_seconds", "expert_hit_rate"
    ] | .[]) as $k (.; if (.[$k] | type) == "number" then .[$k] |= sigfig(2) else . end)
    # Re-clamp the ordering pairs the artifact validator asserts.
    | (if (.benchmark_wall_seconds | type) == "number" and (.timed_benchmark_seconds | type) == "number"
        then .benchmark_wall_seconds = ([.benchmark_wall_seconds, .timed_benchmark_seconds] | max) else . end)
    | (if (.gpqa_ttft_max_seconds | type) == "number" and (.gpqa_ttft_p50_seconds | type) == "number"
        then .gpqa_ttft_max_seconds = ([.gpqa_ttft_max_seconds, .gpqa_ttft_p50_seconds] | max) else . end)
  )
  ' "${MLXFAST_COMBINED_SCORE_PATH}" > "${round_tmp}"
mv "${round_tmp}" "${MLXFAST_COMBINED_SCORE_PATH}"

combined_passed="$(jq -r '.passed' "${MLXFAST_COMBINED_SCORE_PATH}")"
echo "combine-parallel-correctness: wrote ${MLXFAST_COMBINED_SCORE_PATH} passed=${combined_passed}"

if [[ "${combined_passed}" != "true" ]]; then
  exit 1
fi
