#!/usr/bin/env bash
# Provision the separate organizer-pinned Gemma 4 31B-IT MTP experiment.
#
# This script is intentionally independent from setup.sh: invoking the normal
# setup path must never switch the base challenge to the IT target or download
# an assistant sidecar.
set -euo pipefail
umask 022

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
TARGET_MODEL_ID="mlx-community/gemma-4-31b-it-4bit"
TARGET_REVISION="696d436c404745a59f30e4939a658162b0a9e57f"
ASSISTANT_MODEL_ID="google/gemma-4-31B-it-assistant"
ASSISTANT_REVISION="6c9152a7639e1f87626e4d4fd4dd9f3e20c9f3fb"
TARGET_MANIFEST="${ROOT_DIR}/fixtures/mtp_gemma_4_31b_it_4bit.sha256"
ASSISTANT_MANIFEST="${ROOT_DIR}/fixtures/mtp_gemma_4_31b_it_assistant_bf16.sha256"

DEFAULT_CACHE_ROOT="${HOME}/.cache/mlxfast/gemma4-31b-it-mtp-v1"
CACHE_ROOT="${MLXFAST_MTP_CACHE_ROOT:-${DEFAULT_CACHE_ROOT}}"
TARGET_DIR="${MLXFAST_MTP_TARGET_DIR:-${CACHE_ROOT}/target}"
ASSISTANT_DIR="${MLXFAST_MTP_ASSISTANT_DIR:-${CACHE_ROOT}/assistant}"
TARGET_BASE_URL="https://huggingface.co/${TARGET_MODEL_ID}/resolve/${TARGET_REVISION}"
ASSISTANT_BASE_URL="https://huggingface.co/${ASSISTANT_MODEL_ID}/resolve/${ASSISTANT_REVISION}"

VERIFY_ONLY=0
PROVISION_TARGET=1
PROVISION_ASSISTANT=1
PRINT_PATHS=0

usage() {
  cat <<EOF
Usage: ./setup-mtp.sh [--verify-only] [--target-only|--assistant-only] [--print-paths]

Provision the explicit experimental Gemma 4 31B-IT target and its matched,
organizer-pinned MTP assistant. Downloads are resumable and every byte is
checked against the checked-in SHA256/size manifests.

Cache paths:
  target:    ${TARGET_DIR}
  assistant: ${ASSISTANT_DIR}

Overrides:
  MLXFAST_MTP_CACHE_ROOT
  MLXFAST_MTP_TARGET_DIR
  MLXFAST_MTP_ASSISTANT_DIR

This command does not alter or replace the normal base-track reference cache.
EOF
}

while (( "$#" > 0 )); do
  case "$1" in
    --verify-only)
      VERIFY_ONLY=1
      ;;
    --target-only)
      PROVISION_TARGET=1
      PROVISION_ASSISTANT=0
      ;;
    --assistant-only)
      PROVISION_TARGET=0
      PROVISION_ASSISTANT=1
      ;;
    --print-paths)
      PRINT_PATHS=1
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "setup-mtp.sh: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "${PRINT_PATHS}" == "1" ]]; then
  printf 'MLXFAST_MTP_TARGET_DIR=%q\n' "${TARGET_DIR}"
  printf 'MLXFAST_MTP_ASSISTANT_DIR=%q\n' "${ASSISTANT_DIR}"
  exit 0
fi

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "setup-mtp.sh: required tool not found: $1" >&2
    exit 1
  fi
}

require_tool curl
require_tool shasum
require_tool stat

single_link_count() {
  stat -f '%l' "$1" 2>/dev/null || stat -c '%h' "$1"
}

validate_directory() {
  local path="$1"
  local label="$2"
  if [[ -L "${path}" ]]; then
    echo "setup-mtp.sh: ${label} must not be a symlink: ${path}" >&2
    return 1
  fi
  if [[ -e "${path}" && ! -d "${path}" ]]; then
    echo "setup-mtp.sh: ${label} is not a directory: ${path}" >&2
    return 1
  fi
  mkdir -p "${path}"
  if [[ -L "${path}" || ! -d "${path}" ]]; then
    echo "setup-mtp.sh: failed to create a non-symlink ${label}: ${path}" >&2
    return 1
  fi
}

validate_manifest() {
  local manifest="$1"
  local label="$2"
  local line
  local hash
  local size
  local relative
  local extra
  local entries=0

  if [[ ! -f "${manifest}" || -L "${manifest}" ]]; then
    echo "setup-mtp.sh: ${label} manifest is missing or unsafe: ${manifest}" >&2
    return 1
  fi
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    read -r hash size relative extra <<< "${line}"
    if [[ -n "${extra:-}" || ! "${hash:-}" =~ ^[0-9a-f]{64}$ \
        || ! "${size:-}" =~ ^[1-9][0-9]*$ || -z "${relative:-}" ]]; then
      echo "setup-mtp.sh: malformed ${label} manifest line: ${line}" >&2
      return 1
    fi
    if [[ "${relative}" == /* || "${relative}" == *"/"* || "${relative}" == *"\\"* \
        || "${relative}" == "." || "${relative}" == ".." ]]; then
      echo "setup-mtp.sh: unsafe ${label} manifest path: ${relative}" >&2
      return 1
    fi
    entries=$((entries + 1))
  done < "${manifest}"
  if (( entries == 0 )); then
    echo "setup-mtp.sh: ${label} manifest contains no files" >&2
    return 1
  fi
}

verify_file() {
  local path="$1"
  local expected_hash="$2"
  local expected_size="$3"
  local label="$4"
  local actual_size
  local actual_hash
  local before_signature
  local after_signature

  if [[ ! -f "${path}" || -L "${path}" ]]; then
    return 1
  fi
  if [[ "$(single_link_count "${path}")" != "1" ]]; then
    echo "setup-mtp.sh: ${label} must not be hardlinked: ${path}" >&2
    return 1
  fi
  actual_size="$(wc -c < "${path}" | tr -d ' ')"
  [[ "${actual_size}" == "${expected_size}" ]] || return 1
  before_signature="$(stat -f '%d:%i:%z:%m:%c' "${path}" 2>/dev/null \
    || stat -c '%d:%i:%s:%Y:%Z' "${path}")"
  actual_hash="$(shasum -a 256 "${path}" | awk '{print $1}')"
  after_signature="$(stat -f '%d:%i:%z:%m:%c' "${path}" 2>/dev/null \
    || stat -c '%d:%i:%s:%Y:%Z' "${path}")"
  if [[ "${before_signature}" != "${after_signature}" ]]; then
    echo "setup-mtp.sh: ${label} changed while it was being verified" >&2
    return 1
  fi
  [[ "${actual_hash}" == "${expected_hash}" ]]
}

download_file() {
  local base_url="$1"
  local relative="$2"
  local output="$3"
  local expected_hash="$4"
  local expected_size="$5"
  local label="$6"
  local partial="${output}.partial"

  if verify_file "${output}" "${expected_hash}" "${expected_size}" "${label}"; then
    echo "setup-mtp.sh: using verified ${label}"
    return 0
  fi
  if [[ "${VERIFY_ONLY}" == "1" ]]; then
    echo "setup-mtp.sh: ${label} is missing or does not match its pinned hash" >&2
    return 1
  fi
  if [[ -L "${partial}" ]]; then
    echo "setup-mtp.sh: refusing symlink partial download: ${partial}" >&2
    return 1
  fi
  if [[ -e "${partial}" && "$(single_link_count "${partial}")" != "1" ]]; then
    echo "setup-mtp.sh: refusing hardlinked partial download: ${partial}" >&2
    return 1
  fi

  echo "setup-mtp.sh: downloading ${label}"
  if ! curl --fail --location --retry 5 --retry-all-errors --retry-delay 2 \
      --continue-at - --output "${partial}" "${base_url}/${relative}"; then
    echo "setup-mtp.sh: resume failed for ${label}; retrying from byte zero" >&2
    rm -f "${partial}"
    curl --fail --location --retry 5 --retry-all-errors --retry-delay 2 \
      --output "${partial}" "${base_url}/${relative}"
  fi
  if ! verify_file "${partial}" "${expected_hash}" "${expected_size}" "${label}"; then
    echo "setup-mtp.sh: downloaded ${label} failed size/SHA256 verification" >&2
    rm -f "${partial}"
    return 1
  fi
  mv -f "${partial}" "${output}"
  verify_file "${output}" "${expected_hash}" "${expected_size}" "${label}"
}

provision_manifest() {
  local manifest="$1"
  local destination="$2"
  local base_url="$3"
  local label="$4"
  local line
  local hash
  local size
  local relative
  local extra
  local expected_files=()

  validate_manifest "${manifest}" "${label}"
  validate_directory "${destination}" "${label} cache"

  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    read -r hash size relative extra <<< "${line}"
    expected_files+=("${relative}")
    download_file \
      "${base_url}" "${relative}" "${destination}/${relative}" \
      "${hash}" "${size}" "${label} ${relative}"
  done < "${manifest}"

  while IFS= read -r existing; do
    local name
    name="$(basename "${existing}")"
    if [[ "${name}" == *.partial ]]; then
      continue
    fi
    local found=0
    local expected
    for expected in "${expected_files[@]}"; do
      if [[ "${name}" == "${expected}" ]]; then
        found=1
        break
      fi
    done
    if [[ "${found}" != "1" ]]; then
      echo "setup-mtp.sh: unexpected file in ${label} cache: ${existing}" >&2
      return 1
    fi
  done < <(find "${destination}" -mindepth 1 -maxdepth 1 -print)

  echo "setup-mtp.sh: verified ${label} at ${destination}"
}

if [[ "${PROVISION_TARGET}" == "1" ]]; then
  provision_manifest \
    "${TARGET_MANIFEST}" "${TARGET_DIR}" "${TARGET_BASE_URL}" \
    "Gemma 4 31B-IT target"
fi
if [[ "${PROVISION_ASSISTANT}" == "1" ]]; then
  provision_manifest \
    "${ASSISTANT_MANIFEST}" "${ASSISTANT_DIR}" "${ASSISTANT_BASE_URL}" \
    "Gemma 4 31B-IT assistant"
fi

cat <<EOF
setup-mtp.sh: MTP artifacts ready
  target:    ${TARGET_DIR}
  assistant: ${ASSISTANT_DIR}
  source bytes: target=18444420181 assistant=939044876
  official score: disabled until a paired M5 rebaseline is established
EOF
