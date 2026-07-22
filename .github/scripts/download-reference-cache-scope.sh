#!/usr/bin/env bash
# Download a small, manifest-verified subset of the Laguna XS 2.1 4-bit
# reference checkpoint for cache experiments. The full checkpoint path
# intentionally stays in setup.sh so full downloads keep the same
# lock/verification behavior.
set -euo pipefail

SCOPE="${1:-}"
REFERENCE_DIR="${MLXFAST_REFERENCE_DIR:-.cache/huggingface/hub/models--mlx-community--Laguna-XS-2.1-4bit/snapshots/main}"
REFERENCE_BASE_URL="${MLXFAST_REFERENCE_BASE_URL:-https://huggingface.co/mlx-community/Laguna-XS-2.1-4bit/resolve/main}"
REFERENCE_AUTH_HEADER="${MLXFAST_REFERENCE_AUTH_HEADER:-}"
REFERENCE_APPEND_DOWNLOAD_QUERY="${MLXFAST_REFERENCE_APPEND_DOWNLOAD_QUERY:-auto}"
REFERENCE_MANIFEST_PATH="${MLXFAST_REFERENCE_MANIFEST_PATH:-fixtures/reference_laguna_xs_2_1_4bit.sha256}"

usage() {
  cat >&2 <<EOF
usage: download-reference-cache-scope.sh metadata|first-shard|two-shards

Downloads and verifies only the selected reference checkpoint files.
Use setup.sh for full-checkpoint downloads.
EOF
}

case "${SCOPE}" in
  metadata|first-shard|two-shards) ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ ! -f "${REFERENCE_MANIFEST_PATH}" ]]; then
  echo "cache-probe: reference manifest missing at ${REFERENCE_MANIFEST_PATH}" >&2
  exit 1
fi

download_url_for_file() {
  local url="$1"
  local append_query=0
  local separator="?"

  case "${REFERENCE_APPEND_DOWNLOAD_QUERY}" in
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
      echo "cache-probe: MLXFAST_REFERENCE_APPEND_DOWNLOAD_QUERY must be auto, true, or false" >&2
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

selected_files() {
  local shard_limit=0
  case "${SCOPE}" in
    metadata) shard_limit=0 ;;
    first-shard) shard_limit=1 ;;
    two-shards) shard_limit=2 ;;
  esac

  awk -v shard_limit="${shard_limit}" '
    NF >= 3 && $1 !~ /^#/ {
      if ($3 ~ /\.safetensors$/) {
        if (shards < shard_limit) {
          print $3
          shards++
        }
      } else {
        print $3
      }
    }
  ' "${REFERENCE_MANIFEST_PATH}"
}

manifest_entry() {
  local relative_path="$1"
  awk -v path="${relative_path}" '
    NF >= 3 && $1 !~ /^#/ && $3 == path {
      print $1 " " $2
      found=1
      exit
    }
    END {
      if (!found) {
        exit 1
      }
    }
  ' "${REFERENCE_MANIFEST_PATH}"
}

file_is_current() {
  local relative_path="$1"
  local output_path="$2"
  local expected_hash
  local expected_size
  local actual_hash
  local actual_size

  read -r expected_hash expected_size <<< "$(manifest_entry "${relative_path}")"
  [[ -f "${output_path}" ]] || return 1
  actual_size="$(wc -c < "${output_path}" | tr -d ' ')"
  [[ "${actual_size}" == "${expected_size}" ]] || return 1
  actual_hash="$(shasum -a 256 "${output_path}" | awk '{print $1}')"
  [[ "${actual_hash}" == "${expected_hash}" ]]
}

download_file() {
  local relative_path="$1"
  local output_path="${REFERENCE_DIR}/${relative_path}"
  local url
  local attempt=1

  if file_is_current "${relative_path}" "${output_path}"; then
    echo "cache-probe: using cached ${relative_path}"
    return 0
  fi

  url="$(download_url_for_file "${REFERENCE_BASE_URL%/}/${relative_path}")"
  mkdir -p "$(dirname "${output_path}")"

  while [[ "${attempt}" -le 2 ]]; do
    if [[ "${attempt}" == "1" ]]; then
      echo "cache-probe: downloading ${relative_path}"
    else
      echo "cache-probe: redownloading ${relative_path} after verification failed"
      rm -f "${output_path}"
    fi

    if [[ -n "${REFERENCE_AUTH_HEADER}" ]]; then
      curl \
        --fail \
        --location \
        --retry 5 \
        --retry-all-errors \
        --retry-delay 2 \
        --continue-at - \
        --silent \
        --show-error \
        -H "${REFERENCE_AUTH_HEADER}" \
        --output "${output_path}" \
        "${url}"
    else
      curl \
        --fail \
        --location \
        --retry 5 \
        --retry-all-errors \
        --retry-delay 2 \
        --continue-at - \
        --silent \
        --show-error \
        --output "${output_path}" \
        "${url}"
    fi

    if file_is_current "${relative_path}" "${output_path}"; then
      echo "cache-probe: verified ${relative_path}"
      return 0
    fi

    attempt=$((attempt + 1))
  done

  echo "cache-probe: failed to download verified ${relative_path}" >&2
  return 1
}

downloaded=0
while IFS= read -r file; do
  [[ -n "${file}" ]] || continue
  download_file "${file}"
  downloaded=$((downloaded + 1))
done < <(selected_files)

echo "cache-probe: downloaded ${downloaded} file(s) for scope ${SCOPE}"
