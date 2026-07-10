#!/usr/bin/env bash
# Run the Swift benchmark and emit the benchmark.json scorePath.
set -euo pipefail

# --- Local-mode GPU cool-down gate (--local-iterate / --local-submit) --------
# The ranked runner starts each timed phase only after the GPU has cooled
# below a fixed 40C thermal gate. Local modes expose this trusted shell helper
# to the Swift harness, which invokes it immediately before each measured
# prefill and decode phase. Local-mode only: the ranked/official path keeps its
# operator-side gate.
#
# Knobs (local debugging only; see run_local_cool_gate):
#   MLXFAST_LOCAL_COOL_GATE=0   disable the gate (timings taken hot are not
#                               comparable to gated runs)
#   MLXFAST_MACMON_BIN=...      explicit macmon binary path
#   MLXFAST_GPU_TEMP_CMD=...    testing/portability seam: shell command whose
#                               stdout is the current GPU temperature in C
readonly COOL_GATE_TEMP_C=40             # start timing only at/below this GPU temp (C); same 40C as the ranked gate
readonly COOL_GATE_POLL_SECONDS=10       # temperature poll / progress-notification interval
readonly COOL_GATE_ABORT_SECONDS=180     # minimum total wait before a stalled cool-down aborts
readonly COOL_GATE_STALL_SECONDS=90      # abort once no new minimum temp has been seen for this long
readonly COOL_GATE_MAX_WAIT_SECONDS=900  # hard wait ceiling even while temp is still (slowly) falling; matches the ranked COOL_TIMEOUT
readonly COOL_GATE_PROGRESS_EPSILON_C=0.25  # a new minimum must drop at least this much to count as progress (sensor jitter is not progress)

LOCAL_ITERATE=0
LOCAL_SUBMIT=0
OFFICIAL=0
LOCAL_COOL_GATE_ONLY=0
# Arguments forwarded to `mlxfast-swift benchmark`. --official is a shell-level
# mode selector only, so it is filtered out here; the Swift CLI does not know it.
FORWARD_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --weights|--weights=*|--golden|--golden=*|--score-path|--score-path=*)
      echo "benchmark.sh: use MLXFAST_WEIGHTS_PATH, MLXFAST_CORRECTNESS_GOLDEN_PATH, or MLXFAST_SCORE_PATH for shell path overrides" >&2
      echo "benchmark.sh: pass --weights/--golden/--score-path only to .build/release/mlxfast-swift benchmark" >&2
      exit 1
      ;;
    --official)
      OFFICIAL=1
      continue
      ;;
    --local-cool-gate-only)
      LOCAL_COOL_GATE_ONLY=1
      continue
      ;;
  esac
  if [[ "${arg}" == "--local-iterate" ]]; then
    LOCAL_ITERATE=1
  fi
  if [[ "${arg}" == "--local-submit" ]]; then
    LOCAL_SUBMIT=1
  fi
  FORWARD_ARGS+=("${arg}")
done

if [[ "${OFFICIAL}" == "1" && ( "${LOCAL_ITERATE}" == "1" || "${LOCAL_SUBMIT}" == "1" ) ]]; then
  echo "benchmark.sh: --official cannot be combined with --local-iterate/--local-submit" >&2
  exit 1
fi

if [[ "${LOCAL_ITERATE}" == "1" && "${LOCAL_SUBMIT}" == "1" ]]; then
  echo "benchmark.sh: --local-iterate and --local-submit cannot be used together" >&2
  exit 1
fi

# Bare invocations default to the participant-friendly local edit loop. The
# ranked full benchmark must be requested explicitly: with --official, by the
# trusted workflow env (MLXFAST_OFFICIAL_BENCHMARK_RUN=1 -- also inherited by
# the pinned paired-baseline checkout's own benchmark.sh), or implicitly by an
# operator pointing MLXFAST_CORRECTNESS_GOLDEN_PATH at a provisioned oracle.
if [[ "${LOCAL_COOL_GATE_ONLY}" == "0" && "${LOCAL_ITERATE}" == "0" && "${LOCAL_SUBMIT}" == "0" && "${OFFICIAL}" == "0" ]]; then
  if [[ "${MLXFAST_OFFICIAL_BENCHMARK_RUN:-0}" == "1" || -n "${MLXFAST_CORRECTNESS_GOLDEN_PATH:-}" ]]; then
    OFFICIAL=1
  else
    echo "benchmark.sh: no mode given; defaulting to --local-iterate (use --official for the ranked entrypoint, which requires the private oracle)"
    LOCAL_ITERATE=1
    FORWARD_ARGS+=("--local-iterate")
  fi
fi

if [[ "${LOCAL_ITERATE}" == "1" && -z "${MLXFAST_SCORE_PATH:-}" ]]; then
  SCORE_PATH="score.local-iterate.json"
else
  SCORE_PATH="${MLXFAST_SCORE_PATH:-score.json}"
fi
WEIGHTS_PATH="${MLXFAST_WEIGHTS_PATH:-weights}"
if [[ -z "${MLXFAST_CORRECTNESS_GOLDEN_PATH:-}" && "${LOCAL_SUBMIT}" == "1" ]]; then
  GOLDEN_PATH="correctness_prompts/public_longcopy_gate_english_512_1024.json"
elif [[ -z "${MLXFAST_CORRECTNESS_GOLDEN_PATH:-}" && "${LOCAL_ITERATE}" == "1" ]]; then
  GOLDEN_PATH="correctness_prompts/public_longcopy_gate_english_512_256.json"
else
  GOLDEN_PATH="${MLXFAST_CORRECTNESS_GOLDEN_PATH:-correctness_golden.json}"
fi

# Fail fast with actionable guidance when the golden fixture is missing,
# BEFORE any build/transform work runs. The official mode needs the private
# oracle, which is never in the public repo -- participants who reached it by
# accident used to burn minutes on the transform and then hit a raw
# file-not-found error from the Swift harness.
if [[ "${LOCAL_COOL_GATE_ONLY}" == "0" && ! -f "${GOLDEN_PATH}" ]]; then
  if [[ "${LOCAL_ITERATE}" == "1" || "${LOCAL_SUBMIT}" == "1" || -n "${MLXFAST_CORRECTNESS_GOLDEN_PATH:-}" ]]; then
    echo "benchmark.sh: correctness golden not found at ${GOLDEN_PATH}" >&2
    echo "benchmark.sh: if you overrode MLXFAST_CORRECTNESS_GOLDEN_PATH, check the path;" >&2
    echo "benchmark.sh: otherwise re-sync the repo (the public fixtures live in correctness_prompts/)." >&2
  else
    cat >&2 <<'EOF'
benchmark.sh: correctness_golden.json is missing.

--official is the RANKED entrypoint: it requires the private benchmark
oracle, which is provisioned only on the official runner and is not part of
the public repository.

For local development use one of the local modes, which run against the
public fixtures checked into correctness_prompts/ (a bare ./benchmark.sh
defaults to --local-iterate):

  ./benchmark.sh --local-iterate   # fast edit-loop signal (~2 minutes)
  ./benchmark.sh --local-submit    # pre-submit gate (longer decode window)

(Operators with a provisioned private oracle: set
MLXFAST_CORRECTNESS_GOLDEN_PATH=/path/to/correctness_golden.json.)
EOF
  fi
  exit 1
fi
# Resolve the reference checkpoint the same way setup.sh does. benchmark.sh used
# to hard-default to the reference_weights/ compatibility symlink; when that
# symlink is stale, dangling, or points at a non-directory, the Swift transform
# fails with ENOTDIR ("Not a directory"). Prefer an explicit MLXFAST_REFERENCE_DIR;
# else use reference_weights/ only when it actually holds a checkpoint, resolved to
# its real target so the transform never opens a symlinked directory; else fall
# back to the Hugging Face cache setup.sh downloads into.
REFERENCE_MODEL_REPO="${MLXFAST_REFERENCE_MODEL_REPO:-mlx-community/gemma-4-31b-4bit}"
REFERENCE_REVISION="${MLXFAST_REFERENCE_REVISION:-main}"
REFERENCE_DEFAULT_DIR="reference_weights/gemma-4-31b-4bit"
REFERENCE_HF_HOME="${MLXFAST_HF_HOME:-${HF_HOME:-${HOME:-${PWD}}/.cache/huggingface}}"
REFERENCE_HF_HUB_CACHE="${MLXFAST_HF_HUB_CACHE:-${HF_HUB_CACHE:-${REFERENCE_HF_HOME}/hub}}"
REFERENCE_CACHE_DIR="${MLXFAST_REFERENCE_CACHE_DIR:-${REFERENCE_HF_HUB_CACHE}/models--${REFERENCE_MODEL_REPO//\//--}/snapshots/${REFERENCE_REVISION//\//--}}"
if [[ -n "${MLXFAST_REFERENCE_DIR:-}" ]]; then
  REFERENCE_PATH="${MLXFAST_REFERENCE_DIR}"
elif [[ -f "${REFERENCE_DEFAULT_DIR}/config.json" ]]; then
  REFERENCE_PATH="$(cd -P "${REFERENCE_DEFAULT_DIR}" 2>/dev/null && pwd -P)" \
    || REFERENCE_PATH="${REFERENCE_DEFAULT_DIR}"
elif [[ -f "${REFERENCE_CACHE_DIR}/config.json" ]]; then
  REFERENCE_PATH="${REFERENCE_CACHE_DIR}"
else
  REFERENCE_PATH="${REFERENCE_DEFAULT_DIR}"
fi
SWIFT_BIN="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
MLX_METALLIB="${MLXFAST_MLX_METALLIB:-$(dirname "${SWIFT_BIN}")/mlx.metallib}"
SANDBOX_PROFILE="${MLXFAST_SANDBOX_PROFILE:-tools/deny-network.sb}"
SOURCE_HASH_PATH="${WEIGHTS_PATH}/.benchmark-source.sha256"
if [[ "${LOCAL_ITERATE}" == "1" && -z "${MLXFAST_INTEGRITY_PATH:-}" ]]; then
  INTEGRITY_PATH="benchmark-integrity.local-iterate.json"
else
  INTEGRITY_PATH="${MLXFAST_INTEGRITY_PATH:-benchmark-integrity.json}"
fi
USE_RUNTIME_WORKER="${MLXFAST_USE_RUNTIME_WORKER:-1}"

report_local_iterate_git_base() {
  if [[ "${LOCAL_ITERATE}" != "1" ]]; then
    return 0
  fi
  if ! command -v git >/dev/null 2>&1; then
    return 0
  fi
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    return 0
  fi

  local head_sha
  local origin_main_sha
  head_sha="$(git rev-parse --short=12 HEAD 2>/dev/null || true)"
  origin_main_sha="$(git rev-parse --short=12 --verify origin/main 2>/dev/null || true)"

  if [[ -n "${head_sha}" && -n "${origin_main_sha}" ]]; then
    echo "benchmark.sh: local-iterate git_head=${head_sha} origin_main=${origin_main_sha}"
    if ! git merge-base --is-ancestor origin/main HEAD >/dev/null 2>&1; then
      cat >&2 <<EOF
benchmark.sh: warning: HEAD does not contain the currently fetched origin/main.
benchmark.sh: run 'git fetch origin main', rebase or branch from the latest tip,
benchmark.sh: then rerun './benchmark.sh --local-iterate' before trusting local speedups.
EOF
    fi
  else
    cat >&2 <<EOF
benchmark.sh: warning: could not find origin/main for local-iterate baseline context.
benchmark.sh: run 'git fetch origin main' and measure the latest tip locally before comparing changes.
EOF
  fi
}

# Local modes print the same-machine baseline snapshot (when one exists) BEFORE
# the run starts, so the live per-token numbers streaming from the Swift
# harness can be compared against a target from the first second instead of
# only in the final summary. Diagnostic only: any failure here must never fail
# the benchmark run.
report_local_baseline_context() {
  if [[ "${LOCAL_ITERATE}" != "1" && "${LOCAL_SUBMIT}" != "1" ]]; then
    return 0
  fi
  local baseline_path="${SCORE_PATH%.json}.baseline.json"
  if [[ ! -f "${baseline_path}" ]]; then
    return 0
  fi
  local context
  context="$(jq -r '
    def r3: . * 1000 | round / 1000;
    def r6: . * 1000000 | round / 1000000;
    (.metrics // {}) as $m
    | ($m.prefill_seconds_per_token // 0) as $p
    | ($m.decode_seconds_per_token // 0) as $d
    | select($p > 0 and $d > 0)
    | ($m.prefill_speedup // 0) as $ps
    | ($m.decode_speedup // 0) as $ds
    | (if $ps > 0 and $ds > 0 then pow($ds; 0.75) * pow($ps; 0.25) else 0 end) as $est
    | "prefill \($p | r6) s/token, decode \($d | r6) s/token"
      + (if $est > 0 then ", est score \($est | r3)" else "" end)
  ' "${baseline_path}" 2>/dev/null || true)"
  if [[ -n "${context}" ]]; then
    echo "benchmark.sh: local baseline to beat (${baseline_path}): ${context}" >&2
  fi
  return 0
}

# Local modes end with a compact human-readable summary on stderr so the score
# does not have to be dug out of the JSON payload. The estimated score uses the
# official formula against the official baseline constants carried inside the
# score payload; local modes still publish score: null because only the ranked
# runner produces an official score. When a same-machine baseline snapshot
# exists next to the score file (the documented
# `cp score.local-iterate.json score.local-iterate.baseline.json` workflow),
# the summary also prints deltas against it. Diagnostic only: any failure here
# must never fail the benchmark run.
report_local_score_summary() {
  if [[ "${LOCAL_ITERATE}" != "1" && "${LOCAL_SUBMIT}" != "1" ]]; then
    return 0
  fi
  local mode_name="local-submit"
  if [[ "${LOCAL_ITERATE}" == "1" ]]; then
    mode_name="local-iterate"
  fi

  local summary
  summary="$(jq -r '
    def r3: . * 1000 | round / 1000;
    def r6: . * 1000000 | round / 1000000;
    (.metrics // {}) as $m
    | ($m.prefill_seconds_per_token // 0) as $p
    | ($m.decode_seconds_per_token // 0) as $d
    | select($p > 0 and $d > 0)
    | ($m.prefill_speedup // 0) as $ps
    | ($m.decode_speedup // 0) as $ds
    | (if $ps > 0 and $ds > 0 then pow($ds; 0.75) * pow($ps; 0.25) else 0 end) as $est
    | "  prefill \($p | r6) s/token  speedup \($ps | r3)x\n"
      + "  decode  \($d | r6) s/token  speedup \($ds | r3)x"
      + (if $est > 0
         then "\n  est score \($est | r3) (decode_speedup^0.75 * prefill_speedup^0.25; official score comes from the ranked runner)"
         else "" end)
  ' "${SCORE_PATH}" 2>/dev/null || true)"
  if [[ -z "${summary}" ]]; then
    return 0
  fi
  {
    echo "benchmark.sh: ${mode_name} summary"
    printf '%s\n' "${summary}"
  } >&2

  local baseline_path="${SCORE_PATH%.json}.baseline.json"
  if [[ ! -f "${baseline_path}" ]]; then
    if [[ "${LOCAL_ITERATE}" == "1" ]]; then
      echo "benchmark.sh: no local baseline at ${baseline_path}; run 'cp ${SCORE_PATH} ${baseline_path}' to compare future runs" >&2
    fi
    return 0
  fi
  local compare
  compare="$(jq -r -n --slurpfile cur "${SCORE_PATH}" --slurpfile base "${baseline_path}" '
    def r1: . * 10 | round / 10;
    def r3: . * 1000 | round / 1000;
    def r6: . * 1000000 | round / 1000000;
    def sign: if . >= 0 then "+" else "" end;
    ($cur[0].metrics // {}) as $c
    | ($base[0].metrics // {}) as $b
    | ($c.prefill_seconds_per_token // 0) as $cp
    | ($c.decode_seconds_per_token // 0) as $cd
    | ($b.prefill_seconds_per_token // 0) as $bp
    | ($b.decode_seconds_per_token // 0) as $bd
    | select($cp > 0 and $cd > 0 and $bp > 0 and $bd > 0)
    | (($cp - $bp) / $bp * 100) as $pdelta
    | (($cd - $bd) / $bd * 100) as $ddelta
    | ($c.prefill_speedup // 0) as $cps
    | ($c.decode_speedup // 0) as $cds
    | ($b.prefill_speedup // 0) as $bps
    | ($b.decode_speedup // 0) as $bds
    | (if $cps > 0 and $cds > 0 then pow($cds; 0.75) * pow($cps; 0.25) else 0 end) as $cest
    | (if $bps > 0 and $bds > 0 then pow($bds; 0.75) * pow($bps; 0.25) else 0 end) as $best
    | "    prefill \($bp | r6) -> \($cp | r6) s/token (\($pdelta | sign)\($pdelta | r1)%)\n"
      + "    decode  \($bd | r6) -> \($cd | r6) s/token (\($ddelta | sign)\($ddelta | r1)%)"
      + (if $cest > 0 and $best > 0
         then (((($cest - $best) / $best) * 100) as $edelta
           | "\n    est score \($best | r3) -> \($cest | r3) (\($edelta | sign)\($edelta | r1)%)")
         else "" end)
  ' 2>/dev/null || true)"
  if [[ -z "${compare}" ]]; then
    return 0
  fi
  {
    echo "benchmark.sh: vs ${baseline_path} (negative s/token deltas = faster)"
    printf '%s\n' "${compare}"
  } >&2
}

find_macmon() {
  # Prefer an explicit override, then PATH, then the usual install locations
  # (Homebrew, and the ~/bin drop used on the ranked boxes).
  local candidate
  if [[ -n "${MLXFAST_MACMON_BIN:-}" ]]; then
    if [[ -x "${MLXFAST_MACMON_BIN}" ]]; then
      printf '%s\n' "${MLXFAST_MACMON_BIN}"
      return 0
    fi
    echo "benchmark.sh: MLXFAST_MACMON_BIN is set but not executable: ${MLXFAST_MACMON_BIN}" >&2
    return 1
  fi
  if candidate="$(command -v macmon 2>/dev/null)"; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  for candidate in /opt/homebrew/bin/macmon /usr/local/bin/macmon "${HOME}/bin/macmon"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

# Prints the current GPU temperature in C (one line), or nothing on failure.
# MLXFAST_GPU_TEMP_CMD is a documented testing/portability seam: any shell
# command whose stdout is a plain Celsius number can stand in for macmon.
COOL_GATE_MACMON_BIN=""
local_gpu_temp() {
  if [[ -n "${MLXFAST_GPU_TEMP_CMD:-}" ]]; then
    bash -c "${MLXFAST_GPU_TEMP_CMD}" 2>/dev/null | head -n 1 | tr -d '[:space:]'
    return 0
  fi
  "${COOL_GATE_MACMON_BIN}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty' 2>/dev/null
}

format_temp_c() {
  awk -v t="$1" 'BEGIN { printf "%.1f", t }'
}

# Block the timed local run until the GPU has cooled to the gate temperature,
# mirroring the ranked runner's thermal gate. Missing-tool policy: warn loudly
# and SKIP (never hard-fail) -- a participant without macmon still gets a
# working local benchmark, just without the thermal guarantee; setup.sh
# installs/instructs about macmon so the tool being present is the normal
# case. Abort policy: if the GPU is hot and NOT trending down, something else
# is loading it and waiting longer will not help -- exit non-zero with an
# actionable message so scripted loops stop instead of measuring a loaded GPU.
run_local_cool_gate() {
  if [[ "${LOCAL_ITERATE}" != "1" && "${LOCAL_SUBMIT}" != "1" ]]; then
    return 0
  fi
  if [[ "${MLXFAST_LOCAL_COOL_GATE:-1}" == "0" ]]; then
    echo "benchmark.sh: warning: local GPU cool-down gate disabled (MLXFAST_LOCAL_COOL_GATE=0); hot-start timings are not comparable to gated runs" >&2
    return 0
  fi

  if [[ -z "${MLXFAST_GPU_TEMP_CMD:-}" ]]; then
    if ! COOL_GATE_MACMON_BIN="$(find_macmon)"; then
      cat >&2 <<EOF
benchmark.sh: warning: skipping the GPU cool-down gate: no GPU temperature reader found.
benchmark.sh: the ranked runner only starts timed runs below a ${COOL_GATE_TEMP_C}C GPU thermal gate;
benchmark.sh: without the same gate, hot back-to-back local runs can look slower than
benchmark.sh: they are. Install macmon (rerunning ./setup.sh does this for you):
benchmark.sh:   brew install macmon
benchmark.sh: or set MLXFAST_MACMON_BIN=/path/to/macmon.
EOF
      return 0
    fi
  fi

  local temp waited=0 min_temp="" last_progress_waited=0 bad_samples=0
  while :; do
    temp="$(local_gpu_temp || true)"
    if [[ ! "${temp}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
      # Tolerate a couple of flaky reads, then skip the gate rather than hang
      # a participant behind a broken sensor path.
      bad_samples=$((bad_samples + 1))
      if [[ "${bad_samples}" -ge 3 ]]; then
        echo "benchmark.sh: warning: skipping the GPU cool-down gate: temperature reader returned no usable sample (reader: ${MLXFAST_GPU_TEMP_CMD:-${COOL_GATE_MACMON_BIN}})" >&2
        return 0
      fi
      sleep 2
      continue
    fi
    bad_samples=0

    if awk -v t="${temp}" -v gate="${COOL_GATE_TEMP_C}" 'BEGIN { exit !(t <= gate) }'; then
      echo "benchmark.sh: GPU cool-down gate passed (current $(format_temp_c "${temp}")C, target <=${COOL_GATE_TEMP_C}C, waited ${waited}s)" >&2
      return 0
    fi

    # Progress heuristic: track the minimum temperature seen; only a new
    # minimum at least COOL_GATE_PROGRESS_EPSILON_C below the previous one
    # counts as progress, so sensor jitter around a plateau does not look
    # like cooling.
    if [[ -z "${min_temp}" ]] \
        || awk -v t="${temp}" -v m="${min_temp}" -v e="${COOL_GATE_PROGRESS_EPSILON_C}" 'BEGIN { exit !(t <= m - e) }'; then
      min_temp="${temp}"
      last_progress_waited="${waited}"
    fi

    # Abort when BOTH the total wait exceeded the abort floor AND no progress
    # has been made recently: still hot and not decreasing means an external
    # GPU load, and more waiting will not fix that.
    if [[ "${waited}" -ge "${COOL_GATE_ABORT_SECONDS}" \
        && "$((waited - last_progress_waited))" -ge "${COOL_GATE_STALL_SECONDS}" ]]; then
      cat >&2 <<EOF
benchmark.sh: ERROR: GPU is hot and not cooling down (current $(format_temp_c "${temp}")C, min seen $(format_temp_c "${min_temp}")C, target <=${COOL_GATE_TEMP_C}C, waited ${waited}s).
benchmark.sh: something else appears to be loading the GPU. Close GPU-heavy
benchmark.sh: processes (other benchmarks, ML jobs, games, video encodes),
benchmark.sh: let the machine cool, and rerun. To debug without the gate, set
benchmark.sh: MLXFAST_LOCAL_COOL_GATE=0 (hot-start timings are not comparable).
EOF
      exit 1
    fi
    # Hard ceiling: even a slowly-cooling GPU should not stall the edit loop
    # for more than the ranked runner's own cool timeout.
    if [[ "${waited}" -ge "${COOL_GATE_MAX_WAIT_SECONDS}" ]]; then
      echo "benchmark.sh: ERROR: GPU did not reach ${COOL_GATE_TEMP_C}C within ${COOL_GATE_MAX_WAIT_SECONDS}s (current $(format_temp_c "${temp}")C); reduce GPU load or ambient heat and rerun" >&2
      exit 1
    fi

    echo "benchmark.sh: waiting for GPU to cool down before timing (current $(format_temp_c "${temp}")C, target <=${COOL_GATE_TEMP_C}C, waited ${waited}s)..." >&2
    sleep "${COOL_GATE_POLL_SECONDS}"
    waited=$((waited + COOL_GATE_POLL_SECONDS))
  done
}

absolute_path() {
  local path="$1"
  local dir
  local base
  dir="$(dirname "${path}")"
  base="$(basename "${path}")"
  if [[ "${dir}" = "." ]]; then
    printf '%s/%s\n' "$(pwd -P)" "${base}"
  else
    (cd -P "${dir}" 2>/dev/null && printf '%s/%s\n' "$(pwd -P)" "${base}") || printf '%s\n' "${path}"
  fi
}

canonical_directory_path() {
  local path="$1"
  (cd -P "${path}" 2>/dev/null && pwd -P)
}

path_contains() {
  local parent="$1"
  local child="$2"
  [[ "${child}" == "${parent}" || "${child}" == "${parent}/"* ]]
}

safe_clear_directory_path() {
  local path="$1"
  local label="$2"
  shift 2

  if [[ -z "${path}" ]]; then
    echo "benchmark.sh: refusing to clear empty ${label}" >&2
    return 1
  fi

  local target
  local workspace
  if [[ -d "${path}" ]]; then
    target="$(canonical_directory_path "${path}")" || {
      echo "benchmark.sh: could not resolve ${label} '${path}'" >&2
      return 1
    }
  elif [[ -e "${path}" || -L "${path}" ]]; then
    echo "benchmark.sh: refusing to clear ${label} '${path}'; it is not a directory" >&2
    return 1
  else
    local parent
    parent="$(dirname "${path}")"
    if [[ ! -d "${parent}" ]]; then
      echo "benchmark.sh: parent directory for ${label} does not exist: ${parent}" >&2
      return 1
    fi
    parent="$(canonical_directory_path "${parent}")" || {
      echo "benchmark.sh: could not resolve parent directory for ${label} '${path}'" >&2
      return 1
    }
    target="${parent}/$(basename "${path}")"
  fi
  workspace="$(pwd -P)"
  if [[ "${target}" == "/" ]] || path_contains "${target}" "${workspace}"; then
    echo "benchmark.sh: refusing to clear unsafe ${label} '${path}' (resolved to ${target})" >&2
    return 1
  fi

  if path_contains "${workspace}" "${target}"; then
    local relative_target="${target#"${workspace}/"}"
    if [[ "${relative_target}" == ".git" || "${relative_target}" == .git/* ]]; then
      echo "benchmark.sh: refusing to clear Git metadata path '${path}'" >&2
      return 1
    fi
    if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      local tracked_path
      local tracked_paths_file
      local unsafe_tracked_path=""
      tracked_paths_file="$(mktemp "${TMPDIR:-/tmp}/mlxfast-tracked-paths.XXXXXX")" || return 1
      if ! git ls-files -z -- "${relative_target}" > "${tracked_paths_file}"; then
        rm -f "${tracked_paths_file}"
        echo "benchmark.sh: could not inspect tracked files under ${label} '${path}'" >&2
        return 1
      fi
      while IFS= read -r -d '' tracked_path; do
        if [[ "${label}" == "weights path" && "${tracked_path}" == "${relative_target}/.gitkeep" ]]; then
          continue
        fi
        unsafe_tracked_path="${tracked_path}"
        break
      done < "${tracked_paths_file}"
      rm -f "${tracked_paths_file}"
      if [[ -n "${unsafe_tracked_path}" ]]; then
        echo "benchmark.sh: refusing to clear ${label} '${path}'; it contains tracked file ${unsafe_tracked_path}" >&2
        return 1
      fi
    fi
  fi

  local protected
  local protected_path
  for protected in "$@"; do
    [[ -n "${protected}" && -d "${protected}" ]] || continue
    protected_path="$(canonical_directory_path "${protected}")" || continue
    if path_contains "${target}" "${protected_path}" || path_contains "${protected_path}" "${target}"; then
      echo "benchmark.sh: refusing to clear ${label} '${path}'; it overlaps protected path ${protected_path}" >&2
      return 1
    fi
  done

  if [[ ! -d "${target}" ]]; then
    mkdir "${target}" || return 1
  fi

  printf '%s\n' "${target}"
}

assert_safe_output_file_path() {
  local path="$1"
  local label="$2"
  shift 2
  if [[ -z "${path}" || "${path}" == "/" || -d "${path}" || -L "${path}" ]]; then
    echo "benchmark.sh: refusing unsafe ${label} '${path}'" >&2
    return 1
  fi
  case "/${path}/" in
    *"/.git/"*)
      echo "benchmark.sh: refusing ${label} inside Git metadata: ${path}" >&2
      return 1
      ;;
  esac

  local parent
  local target
  local workspace
  parent="$(dirname "${path}")"
  if [[ ! -d "${parent}" ]]; then
    echo "benchmark.sh: parent directory for ${label} does not exist: ${parent}" >&2
    return 1
  fi
  parent="$(canonical_directory_path "${parent}")" || return 1
  target="${parent}/$(basename "${path}")"
  workspace="$(pwd -P)"

  if path_contains "${workspace}" "${target}"; then
    local relative_target="${target#"${workspace}/"}"
    if [[ "${relative_target}" == ".git" || "${relative_target}" == .git/* ]]; then
      echo "benchmark.sh: refusing ${label} inside Git metadata: ${path}" >&2
      return 1
    fi
    if command -v git >/dev/null 2>&1 \
        && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      local tracked_output
      if ! tracked_output="$(git ls-files -- "${relative_target}")"; then
        echo "benchmark.sh: could not inspect ${label} against tracked files: ${relative_target}" >&2
        return 1
      fi
      if [[ -n "${tracked_output}" ]]; then
        echo "benchmark.sh: refusing ${label} that would overwrite tracked file ${relative_target}" >&2
        return 1
      fi
    fi
  fi

  local protected
  local protected_path
  for protected in "$@"; do
    [[ -n "${protected}" ]] || continue
    if [[ -d "${protected}" ]]; then
      protected_path="$(canonical_directory_path "${protected}")" || return 1
    else
      protected_path="$(absolute_path "${protected}")"
    fi
    if [[ "${target}" == "${protected_path}" ]] \
        || { [[ -d "${protected}" ]] && path_contains "${protected_path}" "${target}"; }; then
      echo "benchmark.sh: refusing ${label} that overlaps protected path ${protected_path}" >&2
      return 1
    fi
  done
}

RUNTIME_WORKER_SANDBOX_PROFILE_OWNED=""
TRANSFORM_STAGING_PARENT_OWNED=""
score_stdout=""
cleanup_benchmark_temporaries() {
  local status="$?"
  if [[ -n "${RUNTIME_WORKER_SANDBOX_PROFILE_OWNED}" ]]; then
    rm -f "${RUNTIME_WORKER_SANDBOX_PROFILE_OWNED}" || true
  fi
  if [[ -n "${score_stdout}" ]]; then
    rm -f "${score_stdout}" || true
  fi
  if [[ -n "${TRANSFORM_STAGING_PARENT_OWNED}" ]]; then
    rm -rf "${TRANSFORM_STAGING_PARENT_OWNED}" || true
  fi
  return "${status}"
}

sandbox_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

enforce_official_sandbox() {
  if [[ "${MLXFAST_OFFICIAL_BENCHMARK_RUN:-0}" != "1" ]]; then
    return 0
  fi
  if [[ "${MLXFAST_NO_SANDBOX:-0}" == "1" ]]; then
    echo "benchmark.sh: official GitHub benchmark runs must not set MLXFAST_NO_SANDBOX=1" >&2
    exit 1
  fi
  if [[ "${USE_RUNTIME_WORKER}" != "1" ]]; then
    echo "benchmark.sh: official GitHub benchmark runs must use the runtime worker sandbox" >&2
    exit 1
  fi
}

write_runtime_worker_sandbox_profile() {
  if [[ "${USE_RUNTIME_WORKER}" != "1" || "${MLXFAST_NO_SANDBOX:-0}" == "1" ]]; then
    return 0
  fi
  if [[ -n "${MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE:-}" ]]; then
    return 0
  fi
  if ! command -v sandbox-exec >/dev/null 2>&1; then
    echo "benchmark.sh: sandbox-exec not found for runtime worker sandbox" >&2
    exit 1
  fi

  local profile
  local golden_absolute
  local private_dir_absolute
  local swift_absolute
  profile="$(mktemp "${TMPDIR:-/tmp}/mlxfast-runtime-worker.XXXXXX")"
  RUNTIME_WORKER_SANDBOX_PROFILE_OWNED="${profile}"
  golden_absolute="$(absolute_path "${GOLDEN_PATH}")"
  swift_absolute="$(absolute_path "${SWIFT_BIN}")"
  {
    cat <<EOF
(version 1)
(allow default)
(deny network*)
(deny process-fork)
(deny process-exec*)
(allow process-exec (literal "$(sandbox_escape "${swift_absolute}")"))
(deny file-write*)
(allow file-write* (literal "/dev/null"))
(deny file-read* (literal "$(sandbox_escape "${golden_absolute}")"))
EOF
    if [[ -n "${MLXFAST_PRIVATE_DIR:-}" ]]; then
      private_dir_absolute="$(absolute_path "${MLXFAST_PRIVATE_DIR}")"
      cat <<EOF
(deny file-read* (subpath "$(sandbox_escape "${private_dir_absolute}")"))
(deny file-write* (subpath "$(sandbox_escape "${private_dir_absolute}")"))
EOF
    fi
  } > "${profile}"
  export MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE="${profile}"
}

run_offline_writable_command() {
  local writable_paths="$1"
  local status
  shift
  if [[ "${MLXFAST_IN_SANDBOX:-0}" == "1" || "${MLXFAST_NO_SANDBOX:-0}" == "1" ]]; then
    "$@"
    status="$?"
    return "${status}"
  fi
  MLXFAST_OFFLINE_WRITABLE_PATHS="${writable_paths}" .github/scripts/run-offline.sh "$@"
}

source_hash() {
  # This hash gates regeneration of weights/. Keep it limited to the transform
  # target and shared core code so runtime/model-only edits stay fast locally.
  local paths=(
    "Package.swift"
    "Package.resolved"
    "Sources/MLXFastCore"
    "Sources/MLXFastTransform"
  )
  local hash_status

  if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git ls-files --cached --others --exclude-standard -z "${paths[@]}" \
      | while IFS= read -r -d '' path; do
      if [[ -f "${path}" ]]; then
        printf '%s\0' "${path}"
        shasum -a 256 "${path}"
      else
        printf '%s\0MISSING\0' "${path}"
      fi
    done | shasum -a 256 | awk '{print $1}'
    hash_status="$?"
    return "${hash_status}"
  fi

  find "${paths[@]}" -type f 2>/dev/null | LC_ALL=C sort | while IFS= read -r path; do
    printf '%s\0' "${path}"
    shasum -a 256 "${path}"
  done | shasum -a 256 | awk '{print $1}'
}

install_transformed_weights() {
  local staged_weights="$1"
  local safe_weights_path
  local previous_weights
  local had_previous=0
  safe_weights_path="$(safe_clear_directory_path "${WEIGHTS_PATH}" "weights path" "${REFERENCE_PATH}")" || return 1
  previous_weights="${TRANSFORM_STAGING_PARENT_OWNED}/previous-weights"
  touch "${staged_weights}/.gitkeep" || return 1

  if [[ -e "${safe_weights_path}" || -L "${safe_weights_path}" ]]; then
    mv "${safe_weights_path}" "${previous_weights}" || return 1
    had_previous=1
  fi
  if ! mv "${staged_weights}" "${safe_weights_path}"; then
    if [[ "${had_previous}" == "1" ]]; then
      mv "${previous_weights}" "${safe_weights_path}" || {
        echo "benchmark.sh: failed to restore previous weights from ${previous_weights}" >&2
        return 1
      }
    fi
    return 1
  fi
  if [[ "${had_previous}" == "1" ]]; then
    rm -rf "${previous_weights}" || return 1
  fi
}

if [[ "${LOCAL_COOL_GATE_ONLY}" == "1" ]]; then
  LOCAL_ITERATE=1
  run_local_cool_gate
  exit 0
fi

trap cleanup_benchmark_temporaries EXIT

assert_safe_output_file_path \
  "${SCORE_PATH}" "score path" \
  "${GOLDEN_PATH}" "${INTEGRITY_PATH}" "${WEIGHTS_PATH}" "${REFERENCE_PATH}" || exit 1
assert_safe_output_file_path \
  "${SCORE_PATH}.sha256" "score checksum path" \
  "${GOLDEN_PATH}" "${SCORE_PATH}" "${INTEGRITY_PATH}" "${WEIGHTS_PATH}" "${REFERENCE_PATH}" || exit 1
assert_safe_output_file_path \
  "${INTEGRITY_PATH}" "integrity path" \
  "${GOLDEN_PATH}" "${SCORE_PATH}" "${SCORE_PATH}.sha256" "${WEIGHTS_PATH}" "${REFERENCE_PATH}" || exit 1

enforce_official_sandbox

if [[ "${MLXFAST_IN_SANDBOX:-0}" != "1" && ! -x "${SWIFT_BIN}" ]]; then
  echo "benchmark.sh: Swift release binary missing; building"
  mkdir -p .build/clang-module-cache
  export CLANG_MODULE_CACHE_PATH="${CLANG_MODULE_CACHE_PATH:-${PWD}/.build/clang-module-cache}"
  swift build -c release
fi

# When the runtime worker is disabled, sandbox this whole script so model code
# cannot use the network. With the worker enabled, do not sandbox the parent:
# Blacksmith rejects nested sandbox-exec. Submitted transform runs through
# run-offline.sh below, and submitted model execution runs in the worker sandbox.
if [[ "${USE_RUNTIME_WORKER}" != "1" && "${MLXFAST_IN_SANDBOX:-0}" != "1" && "${MLXFAST_NO_SANDBOX:-0}" != "1" ]]; then
  if ! command -v sandbox-exec >/dev/null 2>&1; then
    echo "benchmark.sh: sandbox-exec not found (the benchmark requires macOS)." >&2
    echo "Set MLXFAST_NO_SANDBOX=1 to skip the offline sandbox; scores" >&2
    echo "produced that way are not comparable to sandboxed runs." >&2
    exit 1
  fi
  if sandbox-exec -f "${SANDBOX_PROFILE}" \
      curl -fsS --max-time 10 https://example.com -o /dev/null 2>/dev/null; then
    echo "benchmark.sh: sandbox-exec did not block network access; refusing to run" >&2
    exit 1
  fi
  echo "benchmark.sh: network egress is blocked; re-running inside the sandbox"
  # Re-exec with the RESOLVED mode (not the raw "$@"): a bare invocation that
  # defaulted to --local-iterate above must not re-default (and re-print the
  # notice) in the sandboxed child. The ${arr[@]+...} idiom keeps the empty
  # array expansion safe under set -u on macOS's bash 3.2.
  RESOLVED_ARGS=(${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"})
  if [[ "${OFFICIAL}" == "1" ]]; then
    RESOLVED_ARGS+=("--official")
  fi
  exec sandbox-exec -f "${SANDBOX_PROFILE}" env \
    MLXFAST_IN_SANDBOX=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 \
    HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
    "$0" ${RESOLVED_ARGS[@]+"${RESOLVED_ARGS[@]}"}
fi

enforce_official_sandbox
report_local_iterate_git_base

if [[ ! -x "${SWIFT_BIN}" ]]; then
  echo "benchmark.sh: Swift release binary missing at ${SWIFT_BIN}" >&2
  exit 1
fi

if [[ ! -f "${MLX_METALLIB}" ]]; then
  echo "benchmark.sh: MLX metallib missing at ${MLX_METALLIB}; run ./setup.sh before ranked benchmark runs" >&2
fi

write_runtime_worker_sandbox_profile
export MLXFAST_USE_RUNTIME_WORKER="${USE_RUNTIME_WORKER}"
MLXFAST_RUNTIME_WORKER_EXECUTABLE="$(absolute_path "${SWIFT_BIN}")"
export MLXFAST_RUNTIME_WORKER_EXECUTABLE
export MLXFAST_REFERENCE_DIR="${REFERENCE_PATH}"
if [[ "${LOCAL_ITERATE}" == "1" || "${LOCAL_SUBMIT}" == "1" ]]; then
  if [[ -z "${MLXFAST_LOCAL_COOL_GATE_HELPER:-}" ]]; then
    MLXFAST_LOCAL_COOL_GATE_HELPER="$(absolute_path "${BASH_SOURCE[0]}")"
  fi
  export MLXFAST_LOCAL_COOL_GATE_HELPER
fi

safe_clear_directory_path "${WEIGHTS_PATH}" "weights path" "${REFERENCE_PATH}" >/dev/null || exit 1
wanted_hash="$(source_hash)"
current_hash="$(cat "${SOURCE_HASH_PATH}" 2>/dev/null || true)"

if [[ "${MLXFAST_SKIP_TRANSFORM:-0}" == "1" ]]; then
  if [[ ! -f "${WEIGHTS_PATH}/config.json" ]]; then
    echo "benchmark.sh: MLXFAST_SKIP_TRANSFORM=1 but ${WEIGHTS_PATH}/config.json is missing" >&2
    exit 1
  fi
  echo "benchmark.sh: reusing ${WEIGHTS_PATH}/ because MLXFAST_SKIP_TRANSFORM=1"
elif [[ "${MLXFAST_FORCE_TRANSFORM:-0}" == "1" || ! -f "${WEIGHTS_PATH}/config.json" || "${current_hash}" != "${wanted_hash}" ]]; then
  if [[ -f "${REFERENCE_PATH}/config.json" ]]; then
    echo "benchmark.sh: regenerating weights with Swift transform"
    safe_weights_path="$(safe_clear_directory_path "${WEIGHTS_PATH}" "weights path" "${REFERENCE_PATH}")" || exit 1
    TRANSFORM_STAGING_PARENT_OWNED="$(mktemp -d \
      "$(dirname "${safe_weights_path}")/.$(basename "${safe_weights_path}").mlxfast-transform.XXXXXX")"
    staged_weights="${TRANSFORM_STAGING_PARENT_OWNED}/weights"
    run_offline_writable_command "${TRANSFORM_STAGING_PARENT_OWNED}" \
      "${SWIFT_BIN}" transform --reference "${REFERENCE_PATH}" --output "${staged_weights}"
    if [[ ! -f "${staged_weights}/config.json" ]]; then
      echo "benchmark.sh: Swift transform did not produce ${staged_weights}/config.json" >&2
      exit 1
    fi
    install_transformed_weights "${staged_weights}"
    rm -rf "${TRANSFORM_STAGING_PARENT_OWNED}"
    TRANSFORM_STAGING_PARENT_OWNED=""
    printf '%s\n' "${wanted_hash}" > "${SOURCE_HASH_PATH}"
  else
    cat >&2 <<EOF
benchmark.sh: reference weights not found at ${REFERENCE_PATH}, needed to regenerate weights/.
Run ./setup.sh, or set MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 only after placing the reference checkpoint there.
(If you expected cached weights/, the transform source hash did not match.)
EOF
    exit 1
  fi
else
  echo "benchmark.sh: reusing ${WEIGHTS_PATH}/ for unchanged transform source"
fi

if [[ "${MLXFAST_VERIFY_TRANSFORM:-0}" == "1" ]]; then
  if [[ ! -f "${REFERENCE_PATH}/config.json" ]]; then
    echo "benchmark.sh: MLXFAST_VERIFY_TRANSFORM=1 requires reference weights at ${REFERENCE_PATH}" >&2
    exit 1
  fi
  VERIFY_TRANSFORM_TMP_PARENT="${MLXFAST_VERIFY_TRANSFORM_TMP_PARENT:-.mlxfast-transform-verify}"
  VERIFY_TRANSFORM_TMP_PARENT="$(safe_clear_directory_path \
    "${VERIFY_TRANSFORM_TMP_PARENT}" \
    "transform verification temporary path" \
    "${REFERENCE_PATH}" \
    "${WEIGHTS_PATH}")" || exit 1
  find "${VERIFY_TRANSFORM_TMP_PARENT}" -mindepth 1 -exec rm -rf {} +
  echo "benchmark.sh: verifying weights match a fresh run of the submitted Swift transform"
  if run_offline_writable_command "$(absolute_path "${VERIFY_TRANSFORM_TMP_PARENT}")" \
    "${SWIFT_BIN}" verify-transform \
    --reference "${REFERENCE_PATH}" \
    --weights "${WEIGHTS_PATH}" \
    --tmp-parent "${VERIFY_TRANSFORM_TMP_PARENT}"; then
    rm -rf "${VERIFY_TRANSFORM_TMP_PARENT}"
  else
    status="$?"
    rm -rf "${VERIFY_TRANSFORM_TMP_PARENT}"
    exit "${status}"
  fi
fi

rm -f "${SCORE_PATH}"

# The Swift benchmark process links the editable model code paths, so any
# score.json it leaves on disk is untrusted: submitted code running in that
# unsandboxed process could overwrite the file (e.g. at exit) after the harness
# wrote it. Capture the trusted score payload from the process stdout and, only
# AFTER it has fully exited, re-materialize score.json from that payload here in
# the trusted shell -- discarding any in-process tamper of the on-disk file.
if ! command -v jq >/dev/null 2>&1; then
  echo "benchmark.sh: jq is required to seal score.json from the benchmark process stdout" >&2
  exit 1
fi
score_stdout="$(mktemp "${TMPDIR:-/tmp}/mlxfast-score.XXXXXX")"

report_local_baseline_context

"${SWIFT_BIN}" benchmark \
  --weights "${WEIGHTS_PATH}" \
  --golden "${GOLDEN_PATH}" \
  --score-path "${SCORE_PATH}" \
  ${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"} > "${score_stdout}"

# Require exactly one JSON object shaped like a score payload; empty, non-JSON,
# or multiple concatenated objects (an injected extra write) fail closed rather
# than sealing an attacker-controlled or malformed score.
if [[ "$(jq -s 'length' "${score_stdout}" 2>/dev/null)" != "1" ]] \
    || ! jq -e '(.passed | type == "boolean") and has("score") and (.metrics | type == "object")' \
        "${score_stdout}" >/dev/null 2>&1; then
  echo "benchmark.sh: benchmark did not emit a single valid score payload on stdout" >&2
  exit 1
fi
rm -f "${SCORE_PATH}"
cp "${score_stdout}" "${SCORE_PATH}"

if [[ ! -s "${SCORE_PATH}" ]]; then
  echo "benchmark.sh: benchmark did not produce ${SCORE_PATH}" >&2
  exit 1
fi

# stdout was redirected to capture the trusted payload above, so local modes
# must explicitly replay it to the console to keep their existing behavior of
# showing the score there.
if [[ "${LOCAL_ITERATE}" == "1" || "${LOCAL_SUBMIT}" == "1" ]]; then
  cat "${SCORE_PATH}"
  report_local_score_summary
fi

score_hash="$(shasum -a 256 "${SCORE_PATH}" | awk '{print $1}')"
printf '%s  %s\n' "${score_hash}" "${SCORE_PATH}" > "${SCORE_PATH}.sha256"

if ! score_metrics="$(jq -er '
    .metrics
    | select((.weights_hash | type) == "string" and (.weights_hash | length) > 0)
    | select((.weights_file_count | type) == "number" and .weights_file_count >= 0 and (.weights_file_count | floor) == .weights_file_count)
    | select((.weights_byte_count | type) == "number" and .weights_byte_count >= 0 and (.weights_byte_count | floor) == .weights_byte_count)
    | [.weights_hash, (.weights_file_count | tostring), (.weights_byte_count | tostring)]
    | @tsv
  ' "${SCORE_PATH}")"; then
  echo "benchmark.sh: score payload has invalid weights integrity metrics" >&2
  exit 1
fi
IFS=$'\t' read -r weights_hash weights_file_count weights_byte_count <<< "${score_metrics}"
golden_hash=""
if [[ -f "${GOLDEN_PATH}" ]]; then
  golden_hash="$(shasum -a 256 "${GOLDEN_PATH}" | awk '{print $1}')"
fi

jq -n \
  --arg score_path "${SCORE_PATH}" \
  --arg score_sha256 "${score_hash}" \
  --arg weights_path "${WEIGHTS_PATH}" \
  --arg weights_sha256 "${weights_hash}" \
  --argjson weights_file_count "${weights_file_count}" \
  --argjson weights_byte_count "${weights_byte_count}" \
  --arg golden_sha256 "${golden_hash}" \
  --arg transform_source_sha256 "${wanted_hash}" \
  '{
    score_path: $score_path,
    score_sha256: $score_sha256,
    weights_path: $weights_path,
    weights_sha256: $weights_sha256,
    weights_file_count: $weights_file_count,
    weights_byte_count: $weights_byte_count,
    golden_path: "[private]",
    golden_sha256: $golden_sha256,
    transform_source_sha256: $transform_source_sha256
  }' > "${INTEGRITY_PATH}"

if ! jq -e '.passed == true' "${SCORE_PATH}" >/dev/null; then
  echo "benchmark.sh: benchmark produced a failing score; see ${SCORE_PATH}" >&2
  exit 1
fi
