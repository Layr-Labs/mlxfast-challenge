#!/usr/bin/env bash
# Bootstrap system tools and build the Swift-only Gemma 4 harness.
set -euo pipefail

REFERENCE_MODEL_REPO="${MLXFAST_REFERENCE_MODEL_REPO:-mlx-community/gemma-4-31b-4bit}"
REFERENCE_REVISION="${MLXFAST_REFERENCE_REVISION:-main}"
REFERENCE_CACHE_REPO_DIR="models--${REFERENCE_MODEL_REPO//\//--}"
REFERENCE_CACHE_REVISION_DIR="${REFERENCE_REVISION//\//--}"
# No organizer-hosted mirror exists yet for this checkpoint; default straight
# to the Darkbloom R2 mirror (mlx-challenge bucket), which serves the same
# manifest-pinned files as the public Hugging Face repo with higher parallel
# throughput; set MLXFAST_REFERENCE_BASE_URL to
# https://huggingface.co/mlx-community/gemma-4-31b-4bit/resolve/main to fall
# back to Hugging Face. download_url_for_file appends
# "?download=true" automatically for huggingface.co URLs so that resolves to
# the raw LFS/Xet bytes.
DEFAULT_REFERENCE_BASE_URL="https://ds4.darkbloom.ai/gemma-4-31b-4bit"
REFERENCE_BASE_URL="${MLXFAST_REFERENCE_BASE_URL:-${DEFAULT_REFERENCE_BASE_URL}}"
REFERENCE_AUTH_HEADER="${MLXFAST_REFERENCE_AUTH_HEADER:-}"
REFERENCE_APPEND_DOWNLOAD_QUERY="${MLXFAST_REFERENCE_APPEND_DOWNLOAD_QUERY:-auto}"
REFERENCE_MANIFEST_PATH="${MLXFAST_REFERENCE_MANIFEST_PATH:-fixtures/reference_gemma_4_31b_4bit.sha256}"
REFERENCE_HASH_VERIFY="${MLXFAST_REFERENCE_HASH_VERIFY:-1}"
REFERENCE_POST_DOWNLOAD_FULL_VERIFY="${MLXFAST_REFERENCE_POST_DOWNLOAD_FULL_VERIFY:-1}"
REFERENCE_MIN_FREE_GIB="${MLXFAST_REFERENCE_MIN_FREE_GIB:-40}"
# Default 3 parallel shard downloads: the R2 mirror sustains full throughput at
# 3 streams, and higher fan-out mostly adds contention on home connections.
# Override with MLXFAST_REFERENCE_DOWNLOAD_JOBS.
REFERENCE_DOWNLOAD_JOBS="${MLXFAST_REFERENCE_DOWNLOAD_JOBS:-3}"
SETUP_PARALLEL_METALLIB="${MLXFAST_SETUP_PARALLEL_METALLIB:-${MLXFAST_SETUP_PARALLEL_BUILD:-1}}"
SWIFT_BIN="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
MLX_METALLIB="${MLXFAST_MLX_METALLIB:-$(dirname "${SWIFT_BIN}")/mlx.metallib}"
DEFAULT_REFERENCE_DIR="reference_weights/gemma-4-31b-4bit"
DEFAULT_HF_HOME="${MLXFAST_HF_HOME:-${HF_HOME:-${HOME:-${PWD}}/.cache/huggingface}}"
DEFAULT_HF_HUB_CACHE="${MLXFAST_HF_HUB_CACHE:-${HF_HUB_CACHE:-${DEFAULT_HF_HOME}/hub}}"
REFERENCE_CACHE_DIR="${MLXFAST_REFERENCE_CACHE_DIR:-${DEFAULT_HF_HUB_CACHE}/${REFERENCE_CACHE_REPO_DIR}/snapshots/${REFERENCE_CACHE_REVISION_DIR}}"
if [[ -n "${MLXFAST_REFERENCE_DIR:-}" ]]; then
  REFERENCE_DIR="${MLXFAST_REFERENCE_DIR}"
elif [[ -e "${DEFAULT_REFERENCE_DIR}" && ! -L "${DEFAULT_REFERENCE_DIR}" ]]; then
  REFERENCE_DIR="${DEFAULT_REFERENCE_DIR}"
else
  REFERENCE_DIR="${REFERENCE_CACHE_DIR}"
fi
REFERENCE_COMPAT_LINK="${MLXFAST_REFERENCE_COMPAT_LINK:-${DEFAULT_REFERENCE_DIR}}"
# This is a verification stamp, retained under its original environment-variable
# name for compatibility. It is not the inter-process mutation lock below.
REFERENCE_CACHE_LOCK_PATH="${MLXFAST_REFERENCE_CACHE_LOCK_PATH:-${REFERENCE_DIR}/.mlxfast-reference-cache.lock}"
if [[ -d "${REFERENCE_DIR}" ]]; then
  CANONICAL_REFERENCE_LOCK_BASE="$(cd -P "${REFERENCE_DIR}" && pwd -P)"
else
  REFERENCE_LOCK_PARENT="$(dirname "${REFERENCE_DIR}")"
  REFERENCE_LOCK_NAME="$(basename "${REFERENCE_DIR}")"
  CANONICAL_REFERENCE_LOCK_PARENT="$(cd -P "${REFERENCE_LOCK_PARENT}" 2>/dev/null && pwd -P)" \
    || CANONICAL_REFERENCE_LOCK_PARENT="${REFERENCE_LOCK_PARENT}"
  CANONICAL_REFERENCE_LOCK_BASE="${CANONICAL_REFERENCE_LOCK_PARENT}/${REFERENCE_LOCK_NAME}"
fi
REFERENCE_CACHE_MUTATION_LOCK_DIR="${MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR:-${CANONICAL_REFERENCE_LOCK_BASE}.mlxfast-setup.lock}"
REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS="${MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS:-1800}"
REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS="${MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS:-60}"
SETUP_STARTED_SECONDS="${SECONDS}"
METALLIB_BUILD_PID=""
METALLIB_BUILD_STATE="not_started"
REFERENCE_CACHE_MUTATION_LOCK_HELD=0
REFERENCE_REQUIRED_METADATA_FILES=(
  "config.json"
  "model.safetensors.index.json"
)
REFERENCE_OPTIONAL_METADATA_FILES=(
  "README.md"
  "generation_config.json"
  "processor_config.json"
  "tokenizer.json"
  "tokenizer_config.json"
)

print_help() {
  cat <<EOF
Usage: ./setup.sh

Checks the local macOS/Apple Silicon toolchain, builds the Swift harness,
builds mlx.metallib, and downloads the Gemma 4 31B 4-bit reference
checkpoint when it is not already present.

Important environment variables:
  MLXFAST_REFERENCE_DIR              Reference checkpoint directory.
                                     Default: ${REFERENCE_DIR}
  MLXFAST_REFERENCE_CACHE_DIR        Shared Hugging Face-style cache path (under
                                     $HOME by default, so parallel clones reuse
                                     one checkpoint) used for new downloads when
                                     MLXFAST_REFERENCE_DIR is not set.
                                     Default: ${REFERENCE_CACHE_DIR}
  MLXFAST_REFERENCE_BASE_URL         HTTP prefix for checkpoint files.
                                     Default: ${DEFAULT_REFERENCE_BASE_URL}
  MLXFAST_REFERENCE_MANIFEST_PATH    SHA256 manifest for the reference files.
                                     Default: ${REFERENCE_MANIFEST_PATH}
  MLXFAST_REFERENCE_CACHE_LOCK_PATH  Local stamp proving the checkpoint was
                                     fully verified by this manifest.
                                     Default: ${REFERENCE_CACHE_LOCK_PATH}
  MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR
                                     Inter-process lock directory serializing
                                     shared-cache downloads and repairs.
                                     Default: ${REFERENCE_CACHE_MUTATION_LOCK_DIR}
  MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS
                                     Maximum time to wait for another setup.
                                     Default: ${REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS}
  MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS
                                     Age for reclaiming ownerless lock dirs.
                                     Default: ${REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS}
  MLXFAST_REFERENCE_DOWNLOAD_JOBS    Parallel safetensors downloads.
                                     Default: ${REFERENCE_DOWNLOAD_JOBS}
  MLXFAST_REFERENCE_MIN_FREE_GIB     Required free space before download.
                                     Default: ${REFERENCE_MIN_FREE_GIB}
  MLXFAST_REFERENCE_HASH_VERIFY=0    Skip reference SHA256 verification.
  MLXFAST_REFERENCE_POST_DOWNLOAD_FULL_VERIFY=0
                                     Skip the second full-checkpoint SHA256 pass
                                     after all downloaded files were already
                                     verified by size and hash. CI-only speedup.
  MLXFAST_SETUP_PARALLEL_METALLIB=0  Disable overlapping the Metal library build
                                     with reference checkpoint download.
                                     MLXFAST_SETUP_PARALLEL_BUILD is accepted
                                     as a deprecated alias.
  MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1    Build tools only; do not download weights.
  MLXFAST_SKIP_MLX_METALLIB=1        Skip mlx.metallib build.
  MLXFAST_SKIP_MACMON_INSTALL=1      Do not install macmon (the GPU temperature
                                     reader benchmark.sh's local cool-down gate
                                     uses; the gate is skipped when absent).

After setup:
  MLXFAST_OFFLINE_WRITABLE_PATHS="${PWD}/weights" .github/scripts/run-offline.sh .build/release/mlxfast-swift transform --output weights
  ./benchmark.sh
EOF
}

if [[ "$#" -gt 0 ]]; then
  case "$1" in
    -h|--help|help)
      print_help
      exit 0
      ;;
    *)
      echo "setup.sh: unknown argument '$1'" >&2
      echo "Run ./setup.sh --help for usage." >&2
      exit 2
      ;;
  esac
fi

format_duration() {
  local total_seconds="${1:-0}"
  printf '%02d:%02d:%02d' \
    $((total_seconds / 3600)) \
    $(((total_seconds % 3600) / 60)) \
    $((total_seconds % 60))
}

path_size_gib() {
  local path="$1"
  local size_kib

  if [[ ! -e "${path}" ]]; then
    printf '0.0'
    return 0
  fi

  size_kib="$(du -sk "${path}" 2>/dev/null | awk '{print $1}')"
  if [[ -z "${size_kib}" ]]; then
    printf 'unknown'
    return 0
  fi

  awk -v kib="${size_kib}" 'BEGIN { printf "%.1f", kib / 1024 / 1024 }'
}

print_setup_summary() {
  local reference_status="${1:-ready}"
  local elapsed="$((SECONDS - SETUP_STARTED_SECONDS))"
  local reference_line
  local metallib_line

  if [[ "${reference_status}" == "skipped" ]]; then
    reference_line="skipped (${REFERENCE_DIR})"
  elif [[ -f "${REFERENCE_DIR}/config.json" ]]; then
    reference_line="${REFERENCE_DIR} ($(path_size_gib "${REFERENCE_DIR}") GiB)"
  else
    reference_line="missing (${REFERENCE_DIR})"
  fi

  if [[ "${MLXFAST_SKIP_MLX_METALLIB:-0}" == "1" ]]; then
    metallib_line="skipped (${MLX_METALLIB})"
  else
    metallib_line="${MLX_METALLIB}"
  fi

  cat <<EOF
setup.sh: setup complete elapsed=$(format_duration "${elapsed}")
setup.sh: summary
  binary: ${SWIFT_BIN}
  mlx.metallib: ${metallib_line}
  reference checkpoint: ${reference_line}
  next:
    MLXFAST_OFFLINE_WRITABLE_PATHS="${PWD}/weights" .github/scripts/run-offline.sh ${SWIFT_BIN} transform --reference "${REFERENCE_DIR}" --output weights
    ${SWIFT_BIN} correctness --weights weights
    ./benchmark.sh  # requires organizer-supplied correctness_golden.json
EOF
}

load_homebrew_shellenv() {
  local candidate
  local candidates=()

  if [[ -n "${HOMEBREW_PREFIX:-}" ]]; then
    candidates+=("${HOMEBREW_PREFIX}/bin/brew")
  fi
  candidates+=(
    "/opt/homebrew/bin/brew"
    "/usr/local/bin/brew"
    "${HOME}/.linuxbrew/bin/brew"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      eval "$("${candidate}" shellenv)"
      return 0
    fi
  done

  return 1
}

ensure_homebrew() {
  if command -v brew >/dev/null 2>&1; then
    return 0
  fi

  if load_homebrew_shellenv && command -v brew >/dev/null 2>&1; then
    return 0
  fi

  if [[ "${MLXFAST_SKIP_HOMEBREW_INSTALL:-0}" == "1" ]]; then
    echo "setup.sh: Homebrew is not installed and MLXFAST_SKIP_HOMEBREW_INSTALL=1" >&2
    return 1
  fi

  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "setup.sh: automatic Homebrew installation is only supported on macOS" >&2
    return 1
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo "setup.sh: curl is required to install Homebrew" >&2
    return 1
  fi

  echo "setup.sh: Homebrew not found; installing Homebrew"
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

  if ! load_homebrew_shellenv || ! command -v brew >/dev/null 2>&1; then
    echo "setup.sh: Homebrew installation finished, but brew is still not on PATH" >&2
    echo "setup.sh: open a new shell or run Homebrew's shellenv command, then retry" >&2
    return 1
  fi
}

find_cmake() {
  local candidate
  if [[ -n "${MLXFAST_CMAKE_BIN:-}" ]]; then
    if [[ -x "${MLXFAST_CMAKE_BIN}" ]]; then
      printf '%s\n' "${MLXFAST_CMAKE_BIN}"
      return 0
    fi
    echo "setup.sh: MLXFAST_CMAKE_BIN is set but not executable: ${MLXFAST_CMAKE_BIN}" >&2
    return 1
  fi

  if candidate="$(command -v cmake 2>/dev/null)"; then
    printf '%s\n' "${candidate}"
    return 0
  fi

  for candidate in /opt/homebrew/bin/cmake /usr/local/bin/cmake; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

ensure_cmake() {
  if find_cmake >/dev/null; then
    return 0
  fi

  if [[ "${MLXFAST_SKIP_CMAKE_INSTALL:-0}" == "1" ]]; then
    echo "setup.sh: cmake is not installed and MLXFAST_SKIP_CMAKE_INSTALL=1" >&2
    return 1
  fi

  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "setup.sh: automatic cmake installation is only supported on macOS" >&2
    return 1
  fi

  ensure_homebrew
  echo "setup.sh: installing cmake with Homebrew"
  brew install cmake

  if ! find_cmake >/dev/null; then
    echo "setup.sh: cmake installation finished, but cmake was not found" >&2
    return 1
  fi
}

find_macmon() {
  # Same lookup order as benchmark.sh's local cool-down gate: explicit
  # override, PATH, then the usual install locations.
  local candidate
  if [[ -n "${MLXFAST_MACMON_BIN:-}" ]]; then
    if [[ -x "${MLXFAST_MACMON_BIN}" ]]; then
      printf '%s\n' "${MLXFAST_MACMON_BIN}"
      return 0
    fi
    echo "setup.sh: MLXFAST_MACMON_BIN is set but not executable: ${MLXFAST_MACMON_BIN}" >&2
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

ensure_macmon() {
  # macmon (https://github.com/vladkens/macmon) is the unprivileged GPU
  # temperature reader benchmark.sh's local cool-down gate uses to mirror the
  # ranked runner's 40C thermal gate. It is NOT required to run the benchmark:
  # if it stays missing, benchmark.sh warns and skips the gate, so a failed
  # install must never fail setup.
  local macmon_path
  if macmon_path="$(find_macmon)"; then
    echo "setup.sh: macmon found at ${macmon_path} (GPU cool-down gate enabled for local benchmark modes)"
    return 0
  fi

  # Auto-install only when Homebrew is already available: unlike cmake (a hard
  # build requirement that justifies bootstrapping Homebrew), macmon is
  # optional, so a missing brew only downgrades this to instructions.
  if [[ "${MLXFAST_SKIP_MACMON_INSTALL:-0}" != "1" && "$(uname -s)" == "Darwin" ]]; then
    if command -v brew >/dev/null 2>&1 || load_homebrew_shellenv; then
      echo "setup.sh: installing macmon with Homebrew (GPU temperature reader for the local benchmark cool-down gate)"
      brew install macmon || true
    fi
    if macmon_path="$(find_macmon)"; then
      echo "setup.sh: macmon installed at ${macmon_path}"
      return 0
    fi
  fi

  cat >&2 <<'EOF'
setup.sh: warning: macmon is not installed.

benchmark.sh's local modes (--local-iterate / --local-submit) use macmon to
wait for the GPU to cool below 40C before timing, mirroring the ranked
runner's thermal gate. Without it the gate is skipped and hot back-to-back
runs can report misleading timings. Install it with:

  brew install macmon

or download a release from https://github.com/vladkens/macmon and put it on
PATH (or set MLXFAST_MACMON_BIN=/path/to/macmon).

EOF
  return 0
}

ensure_swift_toolchain() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "setup.sh: this Swift harness targets macOS on Apple Silicon" >&2
    exit 1
  fi

  if [[ "$(uname -m)" != "arm64" ]]; then
    echo "setup.sh: this Swift harness requires Apple Silicon (arm64)" >&2
    exit 1
  fi

  if ! command -v swift >/dev/null 2>&1; then
    echo "setup.sh: swift was not found; install Xcode command line tools with xcode-select --install" >&2
    exit 1
  fi

  if ! command -v xcodebuild >/dev/null 2>&1; then
    echo "setup.sh: xcodebuild was not found; install Xcode" >&2
    exit 1
  fi

  if ! xcodebuild -version >/dev/null 2>&1; then
    cat >&2 <<EOF
setup.sh: xcodebuild is installed but not usable.

Open Xcode once, select its command line tools, and accept the license, then retry:

  sudo xcodebuild -license accept

EOF
    exit 1
  fi
}

metal_toolchain_identifier() {
  xcodebuild -showComponent MetalToolchain 2>/dev/null \
    | awk -F': ' '/Toolchain Identifier/ { print $2; exit }' \
    || true
}

configure_metal_toolchain_environment() {
  local identifier
  identifier="${MLXFAST_METAL_TOOLCHAIN_IDENTIFIER:-$(metal_toolchain_identifier)}"
  if [[ -n "${identifier}" ]]; then
    export TOOLCHAINS="${TOOLCHAINS:-${identifier}}"
  fi
}

metal_compiler_is_available() {
  configure_metal_toolchain_environment
  xcrun -sdk macosx metal -v >/dev/null 2>&1
}

ensure_metal_toolchain() {
  if metal_compiler_is_available; then
    return 0
  fi

  if [[ "${MLXFAST_SKIP_METAL_TOOLCHAIN_INSTALL:-0}" == "1" ]]; then
    cat >&2 <<EOF
setup.sh: Xcode's Metal Toolchain is not installed and MLXFAST_SKIP_METAL_TOOLCHAIN_INSTALL=1.

Install full Xcode from the App Store or Apple Developer, open it once, select
its command line tools, accept the license, then retry:

  sudo xcodebuild -license accept
  xcodebuild -downloadComponent MetalToolchain

If you only installed the Command Line Tools and this still fails, install full
Xcode; the MLX Metal runtime needs Apple's Metal compiler toolchain.

EOF
    return 1
  fi

  echo "setup.sh: installing Xcode Metal Toolchain"
  if ! xcodebuild -downloadComponent MetalToolchain; then
    cat >&2 <<EOF
setup.sh: failed to install Xcode's Metal Toolchain.

Install full Xcode from the App Store or Apple Developer, open it once, select
its command line tools, accept the license, then retry:

  sudo xcodebuild -license accept
  xcodebuild -downloadComponent MetalToolchain

If you only installed the Command Line Tools and this still fails, install full
Xcode; the MLX Metal runtime needs Apple's Metal compiler toolchain.

EOF
    return 1
  fi

  if ! metal_compiler_is_available; then
    echo "setup.sh: Metal Toolchain installation finished, but xcrun still cannot execute metal" >&2
    return 1
  fi
}

ensure_reference_space() {
  local directory="$1"
  local available_kib
  local required_kib

  if ! [[ "${REFERENCE_MIN_FREE_GIB}" =~ ^[0-9]+$ ]]; then
    echo "setup.sh: MLXFAST_REFERENCE_MIN_FREE_GIB must be an integer" >&2
    return 1
  fi

  available_kib="$(df -Pk "${directory}" | awk 'NR == 2 {print $4}')"
  required_kib=$((REFERENCE_MIN_FREE_GIB * 1024 * 1024))
  if [[ -z "${available_kib}" || "${available_kib}" -lt "${required_kib}" ]]; then
    cat >&2 <<EOF
setup.sh: not enough free disk space for ${REFERENCE_MODEL_REPO}.

Need at least ${REFERENCE_MIN_FREE_GIB} GiB free under ${directory}; available is $((available_kib / 1024 / 1024)) GiB.
Set MLXFAST_REFERENCE_DIR to a larger SSD, or set MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1
and place/mount the checkpoint manually.

EOF
    return 1
  fi
}

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
      echo "setup.sh: MLXFAST_REFERENCE_APPEND_DOWNLOAD_QUERY must be auto, true, or false" >&2
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

reference_hash_verification_enabled() {
  case "${REFERENCE_HASH_VERIFY}" in
    0|false|FALSE|no|NO)
      return 1
      ;;
    1|true|TRUE|yes|YES)
      return 0
      ;;
    *)
      echo "setup.sh: MLXFAST_REFERENCE_HASH_VERIFY must be 0 or 1" >&2
      return 2
      ;;
  esac
}

reference_post_download_full_verify_enabled() {
  case "${REFERENCE_POST_DOWNLOAD_FULL_VERIFY}" in
    0|false|FALSE|no|NO)
      return 1
      ;;
    1|true|TRUE|yes|YES)
      return 0
      ;;
    *)
      echo "setup.sh: MLXFAST_REFERENCE_POST_DOWNLOAD_FULL_VERIFY must be 0 or 1" >&2
      return 2
      ;;
  esac
}

reference_manifest_entry() {
  local relative_path="$1"
  local line
  local expected_hash
  local expected_size
  local manifest_path
  local extra

  [[ -f "${REFERENCE_MANIFEST_PATH}" ]] || return 1
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    read -r expected_hash expected_size manifest_path extra <<< "${line}"
    if [[ "${manifest_path}" == "${relative_path}" ]]; then
      printf '%s %s\n' "${expected_hash}" "${expected_size}"
      return 0
    fi
  done < "${REFERENCE_MANIFEST_PATH}"

  return 1
}

reference_file_is_current() {
  local relative_path="$1"
  local output_path="$2"
  local label="${3:-${relative_path}}"
  local manifest_entry
  local expected_hash
  local expected_size
  local actual_size
  local actual_hash
  local hash_status

  if reference_hash_verification_enabled; then
    hash_status=0
  else
    hash_status="$?"
  fi
  if [[ "${hash_status}" == "1" ]]; then
    [[ -s "${output_path}" ]]
    return $?
  elif [[ "${hash_status}" != "0" ]]; then
    return 1
  fi

  if [[ ! -f "${REFERENCE_MANIFEST_PATH}" ]]; then
    echo "setup.sh: reference manifest missing at ${REFERENCE_MANIFEST_PATH}" >&2
    return 1
  fi
  if ! manifest_entry="$(reference_manifest_entry "${relative_path}")"; then
    echo "setup.sh: reference manifest has no entry for ${relative_path}" >&2
    return 1
  fi
  read -r expected_hash expected_size <<< "${manifest_entry}"

  if [[ ! -f "${output_path}" ]]; then
    return 1
  fi

  actual_size="$(wc -c < "${output_path}" | tr -d ' ')"
  if [[ "${actual_size}" != "${expected_size}" ]]; then
    echo "setup.sh: cached ${label} size mismatch: expected ${expected_size}, got ${actual_size}"
    return 1
  fi

  actual_hash="$(shasum -a 256 "${output_path}" | awk '{print $1}')"
  if [[ "${actual_hash}" != "${expected_hash}" ]]; then
    echo "setup.sh: cached ${label} sha256 mismatch"
    echo "setup.sh: expected ${expected_hash}"
    echo "setup.sh: actual   ${actual_hash}"
    return 1
  fi

  return 0
}

reference_manifest_hash() {
  if [[ ! -f "${REFERENCE_MANIFEST_PATH}" ]]; then
    echo "setup.sh: reference manifest missing at ${REFERENCE_MANIFEST_PATH}" >&2
    return 1
  fi
  shasum -a 256 "${REFERENCE_MANIFEST_PATH}" | awk '{print $1}'
}

reference_manifest_totals() {
  local line
  local expected_hash
  local expected_size
  local relative_path
  local extra
  local file_count=0
  local byte_count=0

  if [[ ! -f "${REFERENCE_MANIFEST_PATH}" ]]; then
    echo "setup.sh: reference manifest missing at ${REFERENCE_MANIFEST_PATH}" >&2
    return 1
  fi

  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    read -r expected_hash expected_size relative_path extra <<< "${line}"
    if [[ -n "${extra:-}" || -z "${expected_hash:-}" || -z "${expected_size:-}" || -z "${relative_path:-}" ]]; then
      return 1
    fi
    if [[ ! "${expected_hash}" =~ ^[0-9a-f]{64}$ || ! "${expected_size}" =~ ^[0-9]+$ ]]; then
      return 1
    fi
    if [[ "${relative_path}" == /* || "${relative_path}" == *\\* ]]; then
      return 1
    fi
    case "/${relative_path}/" in
      *"/../"*|*"/./"*) return 1 ;;
    esac

    file_count=$((file_count + 1))
    byte_count=$((byte_count + expected_size))
  done < "${REFERENCE_MANIFEST_PATH}"

  [[ "${file_count}" -gt 0 ]] || return 1
  printf '%s %s\n' "${file_count}" "${byte_count}"
}

file_mtime_seconds() {
  local file_path="$1"
  stat -f '%m' "${file_path}" 2>/dev/null || stat -c '%Y' "${file_path}"
}

reference_cache_lock_is_current() {
  local reference_dir="$1"
  local lock_path="${REFERENCE_CACHE_LOCK_PATH}"
  local hash_status
  local expected_manifest_hash
  local expected_manifest_totals
  local manifest_file_count
  local manifest_byte_count
  local line
  local key
  local value
  local in_files=0
  local version=""
  local model_repo=""
  local revision=""
  local manifest_hash=""
  local lock_file_count=""
  local lock_byte_count=""
  local actual_file_count=0
  local actual_byte_count=0
  local relative_path
  local expected_size
  local expected_mtime
  local extra
  local file_path
  local actual_size
  local actual_mtime
  local manifest_entry
  local manifest_expected_hash
  local manifest_expected_size

  if reference_hash_verification_enabled; then
    hash_status=0
  else
    hash_status="$?"
  fi
  if [[ "${hash_status}" != "0" ]]; then
    return 1
  fi

  [[ -f "${lock_path}" ]] || return 1
  if ! expected_manifest_hash="$(reference_manifest_hash)"; then
    return 1
  fi
  if ! expected_manifest_totals="$(reference_manifest_totals)"; then
    return 1
  fi
  read -r manifest_file_count manifest_byte_count <<< "${expected_manifest_totals}"

  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" ]] && continue
    if [[ "${line}" == "--files--" ]]; then
      in_files=1
      continue
    fi

    if [[ "${in_files}" == "0" ]]; then
      key="${line%%=*}"
      value="${line#*=}"
      case "${key}" in
        version) version="${value}" ;;
        model_repo) model_repo="${value}" ;;
        revision) revision="${value}" ;;
        manifest_sha256) manifest_hash="${value}" ;;
        file_count) lock_file_count="${value}" ;;
        byte_count) lock_byte_count="${value}" ;;
      esac
      continue
    fi

    IFS=$'\t' read -r relative_path expected_size expected_mtime extra <<< "${line}"
    if [[ -n "${extra:-}" || -z "${relative_path:-}" || -z "${expected_size:-}" || -z "${expected_mtime:-}" ]]; then
      return 1
    fi
    if [[ "${relative_path}" == /* || "${relative_path}" == *\\* ]]; then
      return 1
    fi
    [[ "${expected_size}" =~ ^[0-9]+$ ]] || return 1
    [[ "${expected_mtime}" =~ ^[0-9]+$ ]] || return 1
    case "/${relative_path}/" in
      *"/../"*|*"/./"*) return 1 ;;
    esac
    if ! manifest_entry="$(reference_manifest_entry "${relative_path}")"; then
      return 1
    fi
    read -r manifest_expected_hash manifest_expected_size <<< "${manifest_entry}"
    [[ -n "${manifest_expected_hash}" ]] || return 1
    [[ "${expected_size}" == "${manifest_expected_size}" ]] || return 1

    file_path="${reference_dir}/${relative_path}"
    [[ -f "${file_path}" ]] || return 1
    actual_size="$(wc -c < "${file_path}" | tr -d ' ')"
    [[ "${actual_size}" == "${expected_size}" ]] || return 1
    if ! actual_mtime="$(file_mtime_seconds "${file_path}")"; then
      return 1
    fi
    [[ "${actual_mtime}" == "${expected_mtime}" ]] || return 1

    actual_file_count=$((actual_file_count + 1))
    actual_byte_count=$((actual_byte_count + actual_size))
  done < "${lock_path}"

  [[ "${version}" == "1" ]] || return 1
  [[ "${model_repo}" == "${REFERENCE_MODEL_REPO}" ]] || return 1
  [[ "${revision}" == "${REFERENCE_REVISION}" ]] || return 1
  [[ "${manifest_hash}" == "${expected_manifest_hash}" ]] || return 1
  [[ "${lock_file_count}" =~ ^[0-9]+$ ]] || return 1
  [[ "${lock_byte_count}" =~ ^[0-9]+$ ]] || return 1
  [[ "${lock_file_count}" == "${manifest_file_count}" ]] || return 1
  [[ "${lock_byte_count}" == "${manifest_byte_count}" ]] || return 1
  [[ "${actual_file_count}" == "${lock_file_count}" ]] || return 1
  [[ "${actual_byte_count}" == "${lock_byte_count}" ]] || return 1

  echo "setup.sh: trusted reference cache lock at ${lock_path}; skipping full SHA256 verification"
  return 0
}

write_reference_cache_lock() {
  local reference_dir="$1"
  local lock_path="${REFERENCE_CACHE_LOCK_PATH}"
  local temp_path
  local manifest_hash
  local line
  local expected_hash
  local expected_size
  local relative_path
  local extra
  local file_path
  local actual_size
  local actual_mtime
  local file_count=0
  local byte_count=0
  local files_path

  if ! manifest_hash="$(reference_manifest_hash)"; then
    return 1
  fi

  mkdir -p "$(dirname "${lock_path}")"
  if ! temp_path="$(mktemp "${lock_path}.tmp.XXXXXX")"; then
    echo "setup.sh: failed to create temporary reference cache stamp" >&2
    return 1
  fi
  if ! files_path="$(mktemp "${lock_path}.files.XXXXXX")"; then
    rm -f "${temp_path}"
    echo "setup.sh: failed to create temporary reference cache file list" >&2
    return 1
  fi
  : > "${files_path}"
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    read -r expected_hash expected_size relative_path extra <<< "${line}"
    if [[ -n "${extra:-}" || -z "${expected_hash:-}" || -z "${expected_size:-}" || -z "${relative_path:-}" ]]; then
      rm -f "${temp_path}" "${files_path}"
      return 1
    fi
    file_path="${reference_dir}/${relative_path}"
    [[ -f "${file_path}" ]] || {
      rm -f "${temp_path}" "${files_path}"
      return 1
    }
    actual_size="$(wc -c < "${file_path}" | tr -d ' ')"
    if ! actual_mtime="$(file_mtime_seconds "${file_path}")"; then
      rm -f "${temp_path}" "${files_path}"
      return 1
    fi
    printf '%s\t%s\t%s\n' "${relative_path}" "${actual_size}" "${actual_mtime}" >> "${files_path}"
    file_count=$((file_count + 1))
    byte_count=$((byte_count + actual_size))
  done < "${REFERENCE_MANIFEST_PATH}"

  if [[ "${file_count}" -eq 0 ]]; then
    rm -f "${temp_path}" "${files_path}"
    return 1
  fi

  if ! {
    printf 'version=1\n'
    printf 'model_repo=%s\n' "${REFERENCE_MODEL_REPO}"
    printf 'revision=%s\n' "${REFERENCE_REVISION}"
    printf 'manifest_sha256=%s\n' "${manifest_hash}"
    printf 'file_count=%s\n' "${file_count}"
    printf 'byte_count=%s\n' "${byte_count}"
    printf 'verified_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf '%s\n' '--files--'
    cat "${files_path}"
  } > "${temp_path}"; then
    rm -f "${temp_path}" "${files_path}"
    return 1
  fi
  rm -f "${files_path}"
  if ! mv "${temp_path}" "${lock_path}"; then
    rm -f "${temp_path}"
    return 1
  fi
  echo "setup.sh: wrote reference cache lock ${lock_path}"
}

download_reference_file() {
  local file="$1"
  local output_path="$2"
  local label="${3:-${file}}"
  local marker_path="${output_path}.complete"
  local url="${REFERENCE_BASE_URL%/}/${file}"
  local started_seconds
  local attempt=1

  if reference_file_is_current "${file}" "${output_path}" "${label}"; then
    echo "setup.sh: using cached ${label}"
    touch "${marker_path}" || return 1
    return 0
  fi
  rm -f "${marker_path}" || return 1

  if ! url="$(download_url_for_file "${url}")"; then
    return 1
  fi

  mkdir -p "$(dirname "${output_path}")" || return 1
  started_seconds="${SECONDS}"
  while [[ "${attempt}" -le 2 ]]; do
    if [[ "${attempt}" == "1" ]]; then
      echo "setup.sh: downloading ${label}"
    else
      echo "setup.sh: redownloading ${label} from scratch after hash verification failed"
      rm -f "${output_path}" "${marker_path}" || return 1
    fi

    if [[ -n "${REFERENCE_AUTH_HEADER}" ]]; then
      if ! curl \
        --fail \
        --location \
        --retry 5 \
        --retry-all-errors \
        --retry-delay 2 \
        --continue-at - \
        -H "${REFERENCE_AUTH_HEADER}" \
        --output "${output_path}" \
        "${url}"; then
        return 1
      fi
    else
      if ! curl \
        --fail \
        --location \
        --retry 5 \
        --retry-all-errors \
        --retry-delay 2 \
        --continue-at - \
        --output "${output_path}" \
        "${url}"; then
        return 1
      fi
    fi
    touch "${marker_path}" || return 1

    if reference_file_is_current "${file}" "${output_path}" "${label}"; then
      echo "setup.sh: downloaded ${label} elapsed=$(format_duration "$((SECONDS - started_seconds))")"
      return 0
    fi

    attempt=$((attempt + 1))
  done

  echo "setup.sh: failed to download verified ${label}" >&2
  return 1
}

download_optional_reference_file() {
  local file="$1"
  local output_path="$2"

  if download_reference_file "${file}" "${output_path}"; then
    return 0
  fi

  rm -f "${output_path}" "${output_path}.complete"
  echo "setup.sh: optional metadata ${file} was not available; continuing"
}

download_reference_shards() {
  local output_dir="$1"
  shift
  local jobs="${REFERENCE_DOWNLOAD_JOBS}"
  local total="$#"
  local started_seconds="${SECONDS}"

  if ! [[ "${jobs}" =~ ^[1-9][0-9]*$ ]]; then
    echo "setup.sh: MLXFAST_REFERENCE_DOWNLOAD_JOBS must be a positive integer" >&2
    return 1
  fi

  if [[ "${jobs}" == "1" || "$#" -le 1 ]]; then
    local file
    local ordinal=0
    echo "setup.sh: downloading ${total} safetensors shard(s) with 1 parallel job"
    for file in "$@"; do
      ordinal=$((ordinal + 1))
      download_reference_file "${file}" "${output_dir}/${file}" "shard ${ordinal}/${total}: ${file}" || return 1
    done
    echo "setup.sh: downloaded ${total}/${total} safetensors shard(s) elapsed=$(format_duration "$((SECONDS - started_seconds))")"
    return 0
  fi

  echo "setup.sh: downloading ${total} safetensors shard(s) with ${jobs} parallel job(s)"
  export REFERENCE_BASE_URL
  export REFERENCE_AUTH_HEADER
  export REFERENCE_APPEND_DOWNLOAD_QUERY
  export REFERENCE_HASH_VERIFY
  local ordinal=0
  local manifest_entry
  local expected_hash
  local expected_size
  # The embedded program expands these variables in each child Bash process.
  # shellcheck disable=SC2016
  if ! for file in "$@"; do
      ordinal=$((ordinal + 1))
      expected_hash=""
      expected_size=""
      if manifest_entry="$(reference_manifest_entry "${file}")"; then
        read -r expected_hash expected_size <<< "${manifest_entry}"
      fi
      printf "%s|%s|%s|%s\0" "${ordinal}" "${file}" "${expected_hash}" "${expected_size}"
    done | xargs -0 -I{} -P "${jobs}" bash -c '
    set -euo pipefail
    record="$1"
    output_dir="$2"
    total="$3"
    ordinal="${record%%|*}"
    remainder="${record#*|}"
    file="${remainder%%|*}"
    remainder="${remainder#*|}"
    expected_hash="${remainder%%|*}"
    expected_size="${remainder#*|}"
    output_path="${output_dir}/${file}"
    marker_path="${output_path}.complete"
    url="${REFERENCE_BASE_URL%/}/${file}"
    started_seconds="${SECONDS}"
    attempt=1

    download_url_for_file() {
      local url="$1"
      local append_query=0
      local separator="?"

      case "${REFERENCE_APPEND_DOWNLOAD_QUERY:-auto}" in
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
          echo "setup.sh: MLXFAST_REFERENCE_APPEND_DOWNLOAD_QUERY must be auto, true, or false" >&2
          return 1
          ;;
      esac

      if [[ "${append_query}" == "1" ]]; then
        if [[ "${url}" == *\?* ]]; then
          separator="&"
        fi
        url="${url}${separator}download=true"
      fi

      printf "%s\n" "${url}"
    }

    reference_file_is_current() {
      local actual_size
      local actual_hash

      case "${REFERENCE_HASH_VERIFY:-1}" in
        0|false|FALSE|no|NO)
          [[ -s "${output_path}" ]]
          return $?
          ;;
        1|true|TRUE|yes|YES)
          ;;
        *)
          echo "setup.sh: MLXFAST_REFERENCE_HASH_VERIFY must be 0 or 1" >&2
          return 1
          ;;
      esac

      if [[ -z "${expected_hash}" || -z "${expected_size}" ]]; then
        echo "setup.sh: reference manifest has no entry for ${file}" >&2
        return 1
      fi
      if [[ ! -f "${output_path}" ]]; then
        return 1
      fi

      actual_size="$(wc -c < "${output_path}" | tr -d " ")"
      if [[ "${actual_size}" != "${expected_size}" ]]; then
        echo "setup.sh: cached shard ${ordinal}/${total}: ${file} size mismatch: expected ${expected_size}, got ${actual_size}"
        return 1
      fi
      actual_hash="$(shasum -a 256 "${output_path}" | awk "{print \$1}")"
      if [[ "${actual_hash}" != "${expected_hash}" ]]; then
        echo "setup.sh: cached shard ${ordinal}/${total}: ${file} sha256 mismatch"
        echo "setup.sh: expected ${expected_hash}"
        echo "setup.sh: actual   ${actual_hash}"
        return 1
      fi

      return 0
    }

    if reference_file_is_current; then
      echo "setup.sh: using cached shard ${ordinal}/${total}: ${file}"
      touch "${marker_path}"
      exit 0
    fi
    rm -f "${marker_path}"

    url="$(download_url_for_file "${url}")"

    mkdir -p "$(dirname "${output_path}")"
    while [[ "${attempt}" -le 2 ]]; do
      if [[ "${attempt}" == "1" ]]; then
        echo "setup.sh: downloading shard ${ordinal}/${total}: ${file}"
      else
        echo "setup.sh: redownloading shard ${ordinal}/${total}: ${file} from scratch after hash verification failed"
        rm -f "${output_path}" "${marker_path}"
      fi

      if [[ -n "${REFERENCE_AUTH_HEADER:-}" ]]; then
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
      touch "${marker_path}"

      if reference_file_is_current; then
        echo "setup.sh: downloaded shard ${ordinal}/${total}: ${file} elapsed=$((SECONDS - started_seconds))s"
        exit 0
      fi

      attempt=$((attempt + 1))
    done

    echo "setup.sh: failed to download verified shard ${ordinal}/${total}: ${file}" >&2
    exit 1
  ' _ {} "${output_dir}" "${total}"; then
    return 1
  fi
  echo "setup.sh: downloaded ${total}/${total} safetensors shard(s) elapsed=$(format_duration "$((SECONDS - started_seconds))")"
}

list_reference_shards() {
  local index_path="$1"

  ensure_swift_harness_ready || return 1

  "${SWIFT_BIN}" checkpoint-shards --index "${index_path}"
}

verify_reference_weights() {
  local reference_dir="$1"
  local index_path="${reference_dir}/model.safetensors.index.json"
  local shard_list
  local file
  local shard_files=()
  local missing=0

  if [[ ! -f "${reference_dir}/config.json" ]]; then
    echo "setup.sh: reference checkpoint is missing config.json at ${reference_dir}" >&2
    return 1
  fi
  if [[ ! -f "${index_path}" ]]; then
    echo "setup.sh: reference checkpoint is missing model.safetensors.index.json at ${reference_dir}" >&2
    return 1
  fi

  if ! shard_list="$(list_reference_shards "${index_path}")"; then
    return 1
  fi
  start_mlx_metallib_build || return 1
  while IFS= read -r file; do
    if [[ -n "${file}" ]]; then
      shard_files+=("${file}")
    fi
  done <<< "${shard_list}"
  if [[ "${#shard_files[@]}" -eq 0 ]]; then
    echo "setup.sh: checkpoint index did not list any safetensors shards" >&2
    return 1
  fi

  for file in "${shard_files[@]}"; do
    if [[ ! -s "${reference_dir}/${file}" ]]; then
      echo "setup.sh: reference checkpoint is missing shard ${file} at ${reference_dir}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" != "0" ]]; then
    return 1
  fi

  if reference_cache_lock_is_current "${reference_dir}"; then
    echo "setup.sh: verified reference checkpoint at ${reference_dir} (${#shard_files[@]} safetensors shard(s))"
    return 0
  fi

  if ! verify_reference_manifest "${reference_dir}"; then
    rm -f "${REFERENCE_CACHE_LOCK_PATH}"
    return 1
  fi
  if ! write_reference_cache_lock "${reference_dir}"; then
    return 1
  fi
  echo "setup.sh: verified reference checkpoint at ${reference_dir} (${#shard_files[@]} safetensors shard(s))"
}

verify_reference_weights_after_verified_download() {
  local reference_dir="$1"
  local index_path="${reference_dir}/model.safetensors.index.json"
  local hash_status
  local shard_list
  local file
  local shard_files=()
  local missing=0

  if reference_hash_verification_enabled; then
    hash_status=0
  else
    hash_status="$?"
  fi
  if [[ "${hash_status}" != "0" ]]; then
    echo "setup.sh: cannot skip post-download full verification unless MLXFAST_REFERENCE_HASH_VERIFY=1" >&2
    return 1
  fi

  if [[ ! -f "${reference_dir}/config.json" ]]; then
    echo "setup.sh: reference checkpoint is missing config.json at ${reference_dir}" >&2
    return 1
  fi
  if [[ ! -f "${index_path}" ]]; then
    echo "setup.sh: reference checkpoint is missing model.safetensors.index.json at ${reference_dir}" >&2
    return 1
  fi

  if ! shard_list="$(list_reference_shards "${index_path}")"; then
    return 1
  fi
  start_mlx_metallib_build || return 1
  while IFS= read -r file; do
    if [[ -n "${file}" ]]; then
      shard_files+=("${file}")
    fi
  done <<< "${shard_list}"
  if [[ "${#shard_files[@]}" -eq 0 ]]; then
    echo "setup.sh: checkpoint index did not list any safetensors shards" >&2
    return 1
  fi

  for file in "${shard_files[@]}"; do
    if [[ ! -s "${reference_dir}/${file}" ]]; then
      echo "setup.sh: reference checkpoint is missing shard ${file} at ${reference_dir}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" != "0" ]]; then
    return 1
  fi

  if ! verify_reference_manifest_sizes "${reference_dir}"; then
    rm -f "${REFERENCE_CACHE_LOCK_PATH}"
    return 1
  fi
  if ! write_reference_cache_lock "${reference_dir}"; then
    return 1
  fi
  echo "setup.sh: verified reference checkpoint at ${reference_dir} (${#shard_files[@]} safetensors shard(s)); skipped second SHA256 pass after verified downloads"
}

verify_reference_manifest_sizes() {
  local reference_dir="$1"
  local line
  local expected_hash
  local expected_size
  local relative_path
  local extra
  local file_path
  local actual_size
  local checked=0

  if [[ ! -f "${REFERENCE_MANIFEST_PATH}" ]]; then
    echo "setup.sh: reference manifest missing at ${REFERENCE_MANIFEST_PATH}" >&2
    return 1
  fi

  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    read -r expected_hash expected_size relative_path extra <<< "${line}"
    if [[ -n "${extra:-}" || -z "${expected_hash:-}" || -z "${expected_size:-}" || -z "${relative_path:-}" ]]; then
      echo "setup.sh: malformed reference manifest line: ${line}" >&2
      return 1
    fi
    if [[ ! "${expected_hash}" =~ ^[0-9a-f]{64}$ || ! "${expected_size}" =~ ^[0-9]+$ ]]; then
      echo "setup.sh: malformed reference manifest line: ${line}" >&2
      return 1
    fi
    if [[ "${relative_path}" == /* || "${relative_path}" == *\\* ]]; then
      echo "setup.sh: unsafe reference manifest path: ${relative_path}" >&2
      return 1
    fi
    case "/${relative_path}/" in
      *"/../"*|*"/./"*)
        echo "setup.sh: unsafe reference manifest path: ${relative_path}" >&2
        return 1
        ;;
    esac

    file_path="${reference_dir}/${relative_path}"
    if [[ ! -f "${file_path}" ]]; then
      echo "setup.sh: reference checkpoint is missing manifest file ${relative_path}" >&2
      return 1
    fi
    actual_size="$(wc -c < "${file_path}" | tr -d ' ')"
    if [[ "${actual_size}" != "${expected_size}" ]]; then
      echo "setup.sh: reference file ${relative_path} size mismatch: expected ${expected_size}, got ${actual_size}" >&2
      return 1
    fi
    checked=$((checked + 1))
  done < "${REFERENCE_MANIFEST_PATH}"

  if [[ "${checked}" -eq 0 ]]; then
    echo "setup.sh: reference manifest contained no files: ${REFERENCE_MANIFEST_PATH}" >&2
    return 1
  fi
  echo "setup.sh: verified ${checked} reference file size(s)"
}

setup_parallel_metallib_enabled() {
  case "${SETUP_PARALLEL_METALLIB}" in
    0|false|FALSE|no|NO)
      return 1
      ;;
    1|true|TRUE|yes|YES)
      return 0
      ;;
    *)
      echo "setup.sh: MLXFAST_SETUP_PARALLEL_METALLIB must be 0 or 1" >&2
      return 2
      ;;
  esac
}

recover_stale_reference_cache_mutation_lock() {
  local lock_dir="${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
  local owner_path="${lock_dir}/owner"
  local owner_line=""
  local owner_pid=""
  local owner_host=""
  local field
  local stale=0

  if [[ -f "${owner_path}" ]]; then
    owner_line="$(cat "${owner_path}" 2>/dev/null || true)"
    for field in ${owner_line}; do
      case "${field}" in
        pid=*) owner_pid="${field#pid=}" ;;
        host=*) owner_host="${field#host=}" ;;
      esac
    done
    if [[ "${owner_host}" == "$(hostname)" \
        && "${owner_pid}" =~ ^[1-9][0-9]*$ ]] \
        && ! kill -0 "${owner_pid}" >/dev/null 2>&1; then
      stale=1
    fi
  else
    local lock_mtime
    local now
    if lock_mtime="$(file_mtime_seconds "${lock_dir}" 2>/dev/null)"; then
      now="$(date +%s)"
      if (( now - lock_mtime >= REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS )); then
        stale=1
      fi
    fi
  fi

  if [[ "${stale}" != "1" ]]; then
    return 1
  fi
  rm -f "${owner_path}" || return 1
  if rmdir "${lock_dir}" 2>/dev/null; then
    echo "setup.sh: recovered stale reference cache mutation lock ${lock_dir}"
    return 0
  fi
  return 1
}

acquire_reference_cache_mutation_lock() {
  local lock_dir="${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
  local started_seconds="${SECONDS}"
  local announced_wait=0

  if ! [[ "${REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "setup.sh: MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS must be a non-negative integer" >&2
    return 1
  fi
  if ! [[ "${REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "setup.sh: MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS must be a positive integer" >&2
    return 1
  fi
  mkdir -p "$(dirname "${lock_dir}")" || return 1

  while ! mkdir "${lock_dir}" 2>/dev/null; do
    if [[ ! -d "${lock_dir}" ]]; then
      echo "setup.sh: reference cache mutation lock path exists and is not a directory: ${lock_dir}" >&2
      return 1
    fi
    if recover_stale_reference_cache_mutation_lock; then
      continue
    fi
    if [[ "${announced_wait}" == "0" ]]; then
      echo "setup.sh: waiting for reference cache mutation lock ${lock_dir}"
      announced_wait=1
    fi
    if (( SECONDS - started_seconds >= REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS )); then
      echo "setup.sh: timed out waiting for reference cache mutation lock ${lock_dir}" >&2
      if [[ -f "${lock_dir}/owner" ]]; then
        echo "setup.sh: lock owner: $(cat "${lock_dir}/owner")" >&2
      fi
      return 1
    fi
    sleep 1
  done

  REFERENCE_CACHE_MUTATION_LOCK_HELD=1
  if ! printf 'pid=%s host=%s started_at=%s\n' \
      "$$" "$(hostname)" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "${lock_dir}/owner"; then
    rm -f "${lock_dir}/owner"
    rmdir "${lock_dir}" 2>/dev/null || true
    REFERENCE_CACHE_MUTATION_LOCK_HELD=0
    echo "setup.sh: failed to record reference cache mutation lock owner" >&2
    return 1
  fi
  echo "setup.sh: acquired reference cache mutation lock ${lock_dir}"
}

release_reference_cache_mutation_lock() {
  local lock_dir="${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
  if [[ "${REFERENCE_CACHE_MUTATION_LOCK_HELD}" != "1" ]]; then
    return 0
  fi

  rm -f "${lock_dir}/owner"
  if ! rmdir "${lock_dir}"; then
    echo "setup.sh: failed to release reference cache mutation lock ${lock_dir}" >&2
    return 1
  fi
  REFERENCE_CACHE_MUTATION_LOCK_HELD=0
  echo "setup.sh: released reference cache mutation lock ${lock_dir}"
}

cleanup_background_builds() {
  local status="$?"
  if [[ "${status}" != "0" ]]; then
    if [[ -n "${METALLIB_BUILD_PID}" ]] && kill -0 "${METALLIB_BUILD_PID}" >/dev/null 2>&1; then
      kill "${METALLIB_BUILD_PID}" >/dev/null 2>&1 || true
    fi
  fi
  if [[ -n "${METALLIB_BUILD_PID}" ]]; then
    wait "${METALLIB_BUILD_PID}" >/dev/null 2>&1 || true
  fi
  if ! release_reference_cache_mutation_lock; then
    if [[ "${status}" == "0" ]]; then
      status=1
    fi
  fi
  return "${status}"
}

build_swift_harness() {
  echo "setup.sh: building Swift harness"
  mkdir -p .build/clang-module-cache
  export CLANG_MODULE_CACHE_PATH="${CLANG_MODULE_CACHE_PATH:-${PWD}/.build/clang-module-cache}"
  swift build -c release
  if [[ ! -x "${SWIFT_BIN}" ]]; then
    echo "setup.sh: Swift binary missing at ${SWIFT_BIN}; build failed or MLXFAST_SWIFT_BIN is wrong" >&2
    return 1
  fi
}

ensure_swift_harness_ready() {
  if [[ ! -x "${SWIFT_BIN}" ]]; then
    echo "setup.sh: Swift binary missing at ${SWIFT_BIN}; build failed or MLXFAST_SWIFT_BIN is wrong" >&2
    return 1
  fi
}

build_mlx_metallib() {
  if [[ "${MLXFAST_SKIP_MLX_METALLIB:-0}" == "1" ]]; then
    echo "setup.sh: skipping mlx.metallib build"
    return 0
  fi
  ensure_cmake || return 1
  ensure_metal_toolchain || return 1
  echo "setup.sh: building mlx.metallib for MLX Swift runtime"
  tools/build-mlx-metallib.sh || return 1
}

start_mlx_metallib_build() {
  local parallel_status
  if [[ "${MLXFAST_SKIP_MLX_METALLIB:-0}" == "1" ]]; then
    METALLIB_BUILD_STATE="completed"
    return 0
  fi
  case "${METALLIB_BUILD_STATE}" in
    running|completed)
      return 0
      ;;
    failed)
      echo "setup.sh: mlx.metallib build already failed" >&2
      return 1
      ;;
    not_started)
      ;;
    *)
      echo "setup.sh: invalid mlx.metallib build state: ${METALLIB_BUILD_STATE}" >&2
      return 1
      ;;
  esac
  if setup_parallel_metallib_enabled; then
    METALLIB_BUILD_STATE="running"
    build_mlx_metallib &
    METALLIB_BUILD_PID="$!"
    echo "setup.sh: mlx.metallib build running in background pid=${METALLIB_BUILD_PID}"
  else
    parallel_status="$?"
    if [[ "${parallel_status}" != "1" ]]; then
      return "${parallel_status}"
    fi
    METALLIB_BUILD_STATE="running"
    if ! build_mlx_metallib; then
      METALLIB_BUILD_STATE="failed"
      return 1
    fi
    METALLIB_BUILD_STATE="completed"
  fi
}

wait_for_mlx_metallib_build() {
  if [[ "${MLXFAST_SKIP_MLX_METALLIB:-0}" == "1" ]]; then
    return 0
  fi
  case "${METALLIB_BUILD_STATE}" in
    running)
      if [[ -z "${METALLIB_BUILD_PID}" ]]; then
        echo "setup.sh: mlx.metallib build is running without a background pid" >&2
        METALLIB_BUILD_STATE="failed"
        return 1
      fi
      echo "setup.sh: waiting for mlx.metallib build"
      if ! wait "${METALLIB_BUILD_PID}"; then
        METALLIB_BUILD_PID=""
        METALLIB_BUILD_STATE="failed"
        echo "setup.sh: mlx.metallib build failed" >&2
        return 1
      fi
      METALLIB_BUILD_PID=""
      METALLIB_BUILD_STATE="completed"
      ;;
    completed)
      ;;
    failed)
      echo "setup.sh: mlx.metallib build failed" >&2
      return 1
      ;;
    not_started)
      echo "setup.sh: mlx.metallib build was not started" >&2
      return 1
      ;;
    *)
      echo "setup.sh: invalid mlx.metallib build state: ${METALLIB_BUILD_STATE}" >&2
      return 1
      ;;
  esac
  if [[ ! -f "${MLX_METALLIB}" ]]; then
    echo "setup.sh: mlx.metallib missing at ${MLX_METALLIB}" >&2
    return 1
  fi
}

verify_reference_manifest() {
  local reference_dir="$1"
  local line
  local expected_hash
  local expected_size
  local relative_path
  local extra
  local file_path
  local actual_size
  local actual_hash
  local checked=0

  case "${REFERENCE_HASH_VERIFY}" in
    0|false|FALSE|no|NO)
      echo "setup.sh: skipping reference SHA256 verification"
      return 0
      ;;
    1|true|TRUE|yes|YES)
      ;;
    *)
      echo "setup.sh: MLXFAST_REFERENCE_HASH_VERIFY must be 0 or 1" >&2
      return 1
      ;;
  esac

  if [[ ! -f "${REFERENCE_MANIFEST_PATH}" ]]; then
    echo "setup.sh: reference manifest missing at ${REFERENCE_MANIFEST_PATH}" >&2
    return 1
  fi

  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    read -r expected_hash expected_size relative_path extra <<< "${line}"
    if [[ -n "${extra:-}" || -z "${expected_hash:-}" || -z "${expected_size:-}" || -z "${relative_path:-}" ]]; then
      echo "setup.sh: malformed reference manifest line: ${line}" >&2
      return 1
    fi
    if [[ ! "${expected_hash}" =~ ^[0-9a-f]{64}$ || ! "${expected_size}" =~ ^[0-9]+$ ]]; then
      echo "setup.sh: malformed reference manifest line: ${line}" >&2
      return 1
    fi
    if [[ "${relative_path}" == /* || "${relative_path}" == *\\* ]]; then
      echo "setup.sh: unsafe reference manifest path: ${relative_path}" >&2
      return 1
    fi
    case "/${relative_path}/" in
      *"/../"*|*"/./"*)
        echo "setup.sh: unsafe reference manifest path: ${relative_path}" >&2
        return 1
        ;;
    esac

    file_path="${reference_dir}/${relative_path}"
    if [[ ! -f "${file_path}" ]]; then
      echo "setup.sh: reference checkpoint is missing manifest file ${relative_path}" >&2
      return 1
    fi
    actual_size="$(wc -c < "${file_path}" | tr -d ' ')"
    if [[ "${actual_size}" != "${expected_size}" ]]; then
      echo "setup.sh: reference file ${relative_path} size mismatch: expected ${expected_size}, got ${actual_size}" >&2
      return 1
    fi
    actual_hash="$(shasum -a 256 "${file_path}" | awk '{print $1}')"
    if [[ "${actual_hash}" != "${expected_hash}" ]]; then
      echo "setup.sh: reference file ${relative_path} sha256 mismatch" >&2
      echo "setup.sh: expected ${expected_hash}" >&2
      echo "setup.sh: actual   ${actual_hash}" >&2
      return 1
    fi
    checked=$((checked + 1))
  done < "${REFERENCE_MANIFEST_PATH}"

  if [[ "${checked}" -eq 0 ]]; then
    echo "setup.sh: reference manifest contained no files: ${REFERENCE_MANIFEST_PATH}" >&2
    return 1
  fi
  echo "setup.sh: verified ${checked} reference file hash(es)"
}

ensure_reference_compat_link() {
  local reference_dir="$1"
  local link_path="${REFERENCE_COMPAT_LINK}"
  local link_absolute
  local link_parent
  local link_name
  local link_target

  reference_dir="$(cd -P "${reference_dir}" 2>/dev/null && pwd -P)" || return 1
  if [[ -d "${link_path}" ]]; then
    link_absolute="$(cd -P "${link_path}" 2>/dev/null && pwd -P)" || return 1
  else
    link_parent="$(dirname "${link_path}")"
    link_name="$(basename "${link_path}")"
    mkdir -p "${link_parent}" || return 1
    link_parent="$(cd -P "${link_parent}" 2>/dev/null && pwd -P)" || return 1
    link_absolute="${link_parent}/${link_name}"
  fi

  if [[ "${reference_dir}" == "${link_absolute}" ]]; then
    return 0
  fi

  if [[ -L "${link_path}" ]]; then
    link_target="$(readlink "${link_path}")" || return 1
    if [[ "${link_target}" == "${reference_dir}" ]]; then
      return 0
    fi
    rm -f "${link_path}" || return 1
  elif [[ -e "${link_path}" ]]; then
    cat >&2 <<EOF
setup.sh: compatibility reference path exists and is not a symlink: ${link_path}

Move it aside, set MLXFAST_REFERENCE_DIR to that checkpoint directly, or set
MLXFAST_REFERENCE_COMPAT_LINK to another path. Leaving a stale compatibility
path in place would make later transform commands read the wrong checkpoint.

EOF
    return 1
  fi

  link_parent="$(dirname "${link_path}")"
  mkdir -p "${link_parent}" || return 1
  ln -s "${reference_dir}" "${link_path}" || return 1
  echo "setup.sh: linked ${link_path} -> ${reference_dir}"
}

download_reference_weights_locked() {
  local reference_dir="$1"
  local parent_dir
  local file
  local index_path
  local post_verify_status
  local shard_list
  local shard_files=()

  parent_dir="$(dirname "${reference_dir}")"
  ensure_reference_space "${parent_dir}" || return 1
  mkdir -p "${reference_dir}" || return 1

  echo "setup.sh: downloading ${REFERENCE_MODEL_REPO} from ${REFERENCE_BASE_URL}"
  echo "setup.sh: reference cache path ${reference_dir}"
  for file in "${REFERENCE_REQUIRED_METADATA_FILES[@]}"; do
    download_reference_file "${file}" "${reference_dir}/${file}" || return 1
  done
  for file in "${REFERENCE_OPTIONAL_METADATA_FILES[@]}"; do
    download_optional_reference_file "${file}" "${reference_dir}/${file}"
  done

  if [[ ! -f "${reference_dir}/config.json" ]]; then
    echo "setup.sh: downloaded checkpoint is missing config.json" >&2
    return 1
  fi
  index_path="${reference_dir}/model.safetensors.index.json"
  if [[ ! -f "${index_path}" ]]; then
    echo "setup.sh: downloaded checkpoint is missing model.safetensors.index.json" >&2
    return 1
  fi

  if ! shard_list="$(list_reference_shards "${index_path}")"; then
    return 1
  fi
  start_mlx_metallib_build || return 1
  while IFS= read -r file; do
    if [[ -n "${file}" ]]; then
      shard_files+=("${file}")
    fi
  done <<< "${shard_list}"
  if [[ "${#shard_files[@]}" -eq 0 ]]; then
    echo "setup.sh: checkpoint index did not list any safetensors shards" >&2
    return 1
  fi

  echo "setup.sh: checkpoint index lists ${#shard_files[@]} safetensors shard(s)"
  download_reference_shards "${reference_dir}" "${shard_files[@]}" || return 1
  if reference_post_download_full_verify_enabled; then
    if ! verify_reference_weights "${reference_dir}"; then
      return 1
    fi
  else
    post_verify_status="$?"
    if [[ "${post_verify_status}" == "1" ]]; then
      if ! verify_reference_weights_after_verified_download "${reference_dir}"; then
        return 1
      fi
    else
      return 1
    fi
  fi

  find "${reference_dir}" -name "*.complete" -type f -delete || return 1
  ensure_reference_compat_link "${reference_dir}" || return 1
  echo "setup.sh: reference weights ready at ${reference_dir}"
}

download_reference_weights() {
  local reference_dir="$1"
  local parent_dir
  local operation_status=0
  local release_status=0

  if [[ -e "${reference_dir}" && ! -d "${reference_dir}" ]]; then
    cat >&2 <<EOF
setup.sh: ${reference_dir} exists but is not a directory.

Move it aside or set MLXFAST_REFERENCE_DIR to a checkpoint directory.

EOF
    return 1
  fi

  parent_dir="$(dirname "${reference_dir}")"
  mkdir -p "${parent_dir}" || return 1
  acquire_reference_cache_mutation_lock || return 1

  # Every cache read is checked under the same mutex as repair so a setup using
  # another manifest or path alias cannot mutate files during validation.
  if [[ -f "${reference_dir}/config.json" ]] && verify_reference_weights "${reference_dir}"; then
    echo "setup.sh: reference weights became ready while waiting at ${reference_dir}"
    ensure_reference_compat_link "${reference_dir}" || operation_status=$?
  else
    if [[ -f "${reference_dir}/config.json" ]]; then
      echo "setup.sh: reference cache at ${reference_dir} is incomplete or stale; repairing changed files"
    fi
    download_reference_weights_locked "${reference_dir}" || operation_status=$?
  fi

  release_reference_cache_mutation_lock || release_status=$?
  if [[ "${operation_status}" == "0" && "${release_status}" != "0" ]]; then
    operation_status="${release_status}"
  fi
  return "${operation_status}"
}

ensure_swift_toolchain
ensure_macmon
trap cleanup_background_builds EXIT

if [[ "${MLXFAST_SKIP_WEIGHTS_DOWNLOAD:-0}" == "1" || "${SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
  build_swift_harness
  start_mlx_metallib_build
  wait_for_mlx_metallib_build
  echo "setup.sh: skipping reference weight download"
  print_setup_summary "skipped"
  exit 0
fi

build_swift_harness
start_mlx_metallib_build
download_reference_weights "${REFERENCE_DIR}"
wait_for_mlx_metallib_build
print_setup_summary "ready"
