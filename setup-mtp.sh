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
ASSISTANT_MODEL_ID="mlx-community/gemma-4-31B-it-qat-assistant-4bit"
ASSISTANT_REVISION="5234fd588403c9b68f3bd20a140b7e61700cb7e2"
TARGET_MANIFEST="${ROOT_DIR}/fixtures/mtp_gemma_4_31b_it_4bit.sha256"
ASSISTANT_MANIFEST="${ROOT_DIR}/fixtures/mtp_gemma_4_31b_it_assistant_qat4bit.sha256"

DEFAULT_CACHE_ROOT="${HOME}/.cache/mlxfast/gemma4-31b-it-mtp-v1"
CACHE_ROOT="${MLXFAST_MTP_CACHE_ROOT:-${DEFAULT_CACHE_ROOT}}"
TARGET_DIR="${MLXFAST_MTP_TARGET_DIR:-${CACHE_ROOT}/target}"
ASSISTANT_DIR="${MLXFAST_MTP_ASSISTANT_DIR:-${CACHE_ROOT}/assistant}"

# The target checkpoint is not mirrored yet. Keep Hugging Face primary and the
# fallback slot empty so a future flat mirror only requires changing/overriding
# the primary URL (and optionally retaining this Hugging Face URL as fallback).
DEFAULT_TARGET_BASE_URL="https://huggingface.co/${TARGET_MODEL_ID}/resolve/${TARGET_REVISION}"
DEFAULT_TARGET_FALLBACK_BASE_URL=""
TARGET_BASE_URL="${MLXFAST_MTP_TARGET_BASE_URL:-${DEFAULT_TARGET_BASE_URL}}"
if [[ -n "${MLXFAST_MTP_TARGET_FALLBACK_BASE_URL+x}" ]]; then
  TARGET_FALLBACK_BASE_URL="${MLXFAST_MTP_TARGET_FALLBACK_BASE_URL}"
elif [[ "${TARGET_BASE_URL}" == "${DEFAULT_TARGET_BASE_URL}" ]]; then
  TARGET_FALLBACK_BASE_URL="${DEFAULT_TARGET_FALLBACK_BASE_URL}"
else
  TARGET_FALLBACK_BASE_URL=""
fi

# The assistant is mirrored as flat, manifest-pinned files in Darkbloom R2.
# Stalled or failed mirror downloads fall back to the pinned Hugging Face
# revision. An explicitly overridden primary has no implicit fallback.
DEFAULT_ASSISTANT_BASE_URL="https://ds4.darkbloom.ai/gemma-4-31B-it-qat-assistant-4bit"
DEFAULT_ASSISTANT_FALLBACK_BASE_URL="https://huggingface.co/${ASSISTANT_MODEL_ID}/resolve/${ASSISTANT_REVISION}"
ASSISTANT_BASE_URL="${MLXFAST_MTP_ASSISTANT_BASE_URL:-${DEFAULT_ASSISTANT_BASE_URL}}"
if [[ -n "${MLXFAST_MTP_ASSISTANT_FALLBACK_BASE_URL+x}" ]]; then
  ASSISTANT_FALLBACK_BASE_URL="${MLXFAST_MTP_ASSISTANT_FALLBACK_BASE_URL}"
elif [[ "${ASSISTANT_BASE_URL}" == "${DEFAULT_ASSISTANT_BASE_URL}" ]]; then
  ASSISTANT_FALLBACK_BASE_URL="${DEFAULT_ASSISTANT_FALLBACK_BASE_URL}"
else
  ASSISTANT_FALLBACK_BASE_URL=""
fi

MTP_APPEND_DOWNLOAD_QUERY="${MLXFAST_MTP_APPEND_DOWNLOAD_QUERY:-auto}"
MTP_DOWNLOAD_STALL_SECONDS="${MLXFAST_MTP_DOWNLOAD_STALL_SECONDS:-120}"
MTP_DOWNLOAD_MIN_BYTES_PER_SECOND="${MLXFAST_MTP_DOWNLOAD_MIN_BYTES_PER_SECOND:-1048576}"

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
  MLXFAST_MTP_TARGET_BASE_URL
  MLXFAST_MTP_TARGET_FALLBACK_BASE_URL
  MLXFAST_MTP_ASSISTANT_BASE_URL
  MLXFAST_MTP_ASSISTANT_FALLBACK_BASE_URL
  MLXFAST_MTP_APPEND_DOWNLOAD_QUERY
  MLXFAST_MTP_DOWNLOAD_STALL_SECONDS
  MLXFAST_MTP_DOWNLOAD_MIN_BYTES_PER_SECOND

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

validate_download_settings() {
  local name
  local value

  for name in \
    MTP_DOWNLOAD_STALL_SECONDS \
    MTP_DOWNLOAD_MIN_BYTES_PER_SECOND; do
    value="${!name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
      echo "setup-mtp.sh: ${name} must be a positive integer" >&2
      return 1
    fi
  done
}

download_url_for_file() {
  local url="$1"
  local append_query=0
  local separator="?"

  case "${MTP_APPEND_DOWNLOAD_QUERY}" in
    1|true|TRUE|yes|YES)
      append_query=1
      ;;
    0|false|FALSE|no|NO)
      append_query=0
      ;;
    auto|"")
      if [[ "${url}" == https://huggingface.co/* || "${url}" == http://huggingface.co/* ]]; then
        append_query=1
      fi
      ;;
    *)
      echo "setup-mtp.sh: MLXFAST_MTP_APPEND_DOWNLOAD_QUERY must be auto, true, or false" >&2
      return 1
      ;;
  esac

  if [[ "${append_query}" == "1" ]]; then
    if [[ "${url}" == *\?* ]]; then
      separator="&"
    fi
    url="${url}${separator}download=true"
  fi

  printf '%s\n' "${url}"
}

download_file() {
  local base_url="$1"
  local fallback_base_url="$2"
  local relative="$3"
  local output="$4"
  local expected_hash="$5"
  local expected_size="$6"
  local label="$7"
  local partial="${output}.partial"
  local source_base_url
  local url
  local attempt
  local curl_status
  local source_index=0
  local source_count
  local base_urls=("${base_url}")

  if [[ -n "${fallback_base_url}" && "${fallback_base_url}" != "${base_url}" ]]; then
    base_urls+=("${fallback_base_url}")
  fi
  source_count="${#base_urls[@]}"

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

  for source_base_url in "${base_urls[@]}"; do
    source_index=$((source_index + 1))
    url="${source_base_url%/}/${relative}"
    if ! url="$(download_url_for_file "${url}")"; then
      return 1
    fi

    if [[ "${source_index}" -gt 1 ]]; then
      echo "setup-mtp.sh: trying fallback source for ${label}: ${source_base_url}"
    fi

    attempt=1
    while [[ "${attempt}" -le 2 ]]; do
      if [[ "${attempt}" == "1" ]]; then
        echo "setup-mtp.sh: downloading ${label}"
      else
        echo "setup-mtp.sh: redownloading ${label} from scratch after hash verification failed"
        rm -f "${partial}"
      fi

      curl_status=0
      curl \
        --fail \
        --location \
        --retry 5 \
        --retry-all-errors \
        --retry-delay 2 \
        --continue-at - \
        --speed-limit "${MTP_DOWNLOAD_MIN_BYTES_PER_SECOND}" \
        --speed-time "${MTP_DOWNLOAD_STALL_SECONDS}" \
        --output "${partial}" \
        "${url}" || curl_status=$?
      if [[ "${curl_status}" != "0" ]]; then
        echo "setup-mtp.sh: ${label} source failed or stalled (status=${curl_status}, source=${source_base_url})" >&2
        break
      fi

      if verify_file "${partial}" "${expected_hash}" "${expected_size}" "${label}"; then
        mv -f "${partial}" "${output}"
        verify_file "${output}" "${expected_hash}" "${expected_size}" "${label}"
        return
      fi

      attempt=$((attempt + 1))
    done

    if [[ "${source_index}" -lt "${source_count}" && "${curl_status}" == "0" ]]; then
      rm -f "${partial}"
    fi
  done

  echo "setup-mtp.sh: failed to download verified ${label}" >&2
  return 1
}

provision_manifest() {
  local manifest="$1"
  local destination="$2"
  local base_url="$3"
  local fallback_base_url="$4"
  local label="$5"
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
      "${base_url}" "${fallback_base_url}" "${relative}" "${destination}/${relative}" \
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

validate_download_settings

if [[ "${PROVISION_TARGET}" == "1" ]]; then
  provision_manifest \
    "${TARGET_MANIFEST}" "${TARGET_DIR}" \
    "${TARGET_BASE_URL}" "${TARGET_FALLBACK_BASE_URL}" \
    "Gemma 4 31B-IT target"
fi
if [[ "${PROVISION_ASSISTANT}" == "1" ]]; then
  provision_manifest \
    "${ASSISTANT_MANIFEST}" "${ASSISTANT_DIR}" \
    "${ASSISTANT_BASE_URL}" "${ASSISTANT_FALLBACK_BASE_URL}" \
    "Gemma 4 31B-IT assistant"
fi

cat <<EOF
setup-mtp.sh: MTP artifacts ready
  target:    ${TARGET_DIR}
  assistant: ${ASSISTANT_DIR}
  source bytes: target=18444420181 assistant=264144321
  official score: disabled until a paired M5 rebaseline is established
EOF
