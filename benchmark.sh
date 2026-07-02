#!/usr/bin/env bash
# Run the Swift benchmark and emit the benchmark.json scorePath.
set -euo pipefail

LOCAL_ITERATE=0
LOCAL_SUBMIT=0
for arg in "$@"; do
  case "${arg}" in
    --weights|--weights=*|--golden|--golden=*|--score-path|--score-path=*)
      echo "benchmark.sh: use MLXFAST_WEIGHTS_PATH, MLXFAST_CORRECTNESS_GOLDEN_PATH, or MLXFAST_SCORE_PATH for shell path overrides" >&2
      echo "benchmark.sh: pass --weights/--golden/--score-path only to .build/release/mlxfast-swift benchmark" >&2
      exit 1
      ;;
  esac
  if [[ "${arg}" == "--local-iterate" ]]; then
    LOCAL_ITERATE=1
  fi
  if [[ "${arg}" == "--local-submit" ]]; then
    LOCAL_SUBMIT=1
  fi
done

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
REFERENCE_PATH="${MLXFAST_REFERENCE_DIR:-reference_weights/DeepSeek-V4-Flash-4bit}"
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

json_string() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

json_number_or_null() {
  local value="$1"
  if [[ -z "${value}" ]]; then
    printf 'null'
  else
    printf '%s' "${value}"
  fi
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

run_offline_command() {
  if [[ "${MLXFAST_IN_SANDBOX:-0}" == "1" || "${MLXFAST_NO_SANDBOX:-0}" == "1" ]]; then
    "$@"
    return 0
  fi
  .github/scripts/run-offline.sh "$@"
}

run_offline_writable_command() {
  local writable_paths="$1"
  shift
  if [[ "${MLXFAST_IN_SANDBOX:-0}" == "1" || "${MLXFAST_NO_SANDBOX:-0}" == "1" ]]; then
    "$@"
    return 0
  fi
  MLXFAST_OFFLINE_WRITABLE_PATHS="${writable_paths}" .github/scripts/run-offline.sh "$@"
}

score_metric_string() {
  local key="$1"
  sed -n "s/.*\"${key}\" : \"\\([^\"]*\\)\".*/\\1/p" "${SCORE_PATH}" | head -n 1
}

score_metric_number() {
  local key="$1"
  sed -n "s/.*\"${key}\" : \\([0-9][0-9]*\\).*/\\1/p" "${SCORE_PATH}" | head -n 1
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

  if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git ls-files -z "${paths[@]}" | while IFS= read -r -d '' path; do
      if [[ -f "${path}" ]]; then
        printf '%s\0' "${path}"
        shasum -a 256 "${path}"
      else
        printf '%s\0MISSING\0' "${path}"
      fi
    done | shasum -a 256 | awk '{print $1}'
    return 0
  fi

  find "${paths[@]}" -type f 2>/dev/null | LC_ALL=C sort | while IFS= read -r path; do
    printf '%s\0' "${path}"
    shasum -a 256 "${path}"
  done | shasum -a 256 | awk '{print $1}'
}

clear_weights_dir() {
  case "${WEIGHTS_PATH}" in
    ""|"/")
      echo "benchmark.sh: refusing to clear unsafe weights path '${WEIGHTS_PATH}'" >&2
      exit 1
      ;;
  esac
  mkdir -p "${WEIGHTS_PATH}"
  find "${WEIGHTS_PATH}" -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
}

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
  exec sandbox-exec -f "${SANDBOX_PROFILE}" env \
    MLXFAST_IN_SANDBOX=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 \
    HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
    "$0" "$@"
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
export MLXFAST_RUNTIME_WORKER_EXECUTABLE="$(absolute_path "${SWIFT_BIN}")"
export MLXFAST_REFERENCE_DIR="${REFERENCE_PATH}"

mkdir -p "${WEIGHTS_PATH}"
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
    clear_weights_dir
    run_offline_writable_command "$(absolute_path "${WEIGHTS_PATH}")" \
      "${SWIFT_BIN}" transform --reference "${REFERENCE_PATH}" --output "${WEIGHTS_PATH}"
    if [[ ! -f "${WEIGHTS_PATH}/config.json" ]]; then
      echo "benchmark.sh: Swift transform did not produce ${WEIGHTS_PATH}/config.json" >&2
      exit 1
    fi
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
  case "${VERIFY_TRANSFORM_TMP_PARENT}" in
    ""|"/")
      echo "benchmark.sh: refusing unsafe transform verification tmp parent '${VERIFY_TRANSFORM_TMP_PARENT}'" >&2
      exit 1
      ;;
  esac
  rm -rf "${VERIFY_TRANSFORM_TMP_PARENT}"
  mkdir -p "${VERIFY_TRANSFORM_TMP_PARENT}"
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
trap 'rm -f "${score_stdout}"' EXIT

"${SWIFT_BIN}" benchmark \
  --weights "${WEIGHTS_PATH}" \
  --golden "${GOLDEN_PATH}" \
  --score-path "${SCORE_PATH}" \
  "$@" > "${score_stdout}"

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
fi

score_hash="$(shasum -a 256 "${SCORE_PATH}" | awk '{print $1}')"
printf '%s  %s\n' "${score_hash}" "${SCORE_PATH}" > "${SCORE_PATH}.sha256"

weights_hash="$(score_metric_string weights_hash)"
weights_file_count="$(score_metric_number weights_file_count)"
weights_byte_count="$(score_metric_number weights_byte_count)"
golden_hash=""
if [[ -f "${GOLDEN_PATH}" ]]; then
  golden_hash="$(shasum -a 256 "${GOLDEN_PATH}" | awk '{print $1}')"
fi

cat > "${INTEGRITY_PATH}" <<EOF
{
  "score_path": "$(json_string "${SCORE_PATH}")",
  "score_sha256": "$(json_string "${score_hash}")",
  "weights_path": "$(json_string "${WEIGHTS_PATH}")",
  "weights_sha256": "$(json_string "${weights_hash}")",
  "weights_file_count": $(json_number_or_null "${weights_file_count}"),
  "weights_byte_count": $(json_number_or_null "${weights_byte_count}"),
  "golden_path": "[private]",
  "golden_sha256": "$(json_string "${golden_hash}")",
  "transform_source_sha256": "$(json_string "${wanted_hash}")"
}
EOF

if grep -Eq '"passed"[[:space:]]*:[[:space:]]*false' "${SCORE_PATH}"; then
  echo "benchmark.sh: benchmark produced a failing score; see ${SCORE_PATH}" >&2
  exit 1
fi
