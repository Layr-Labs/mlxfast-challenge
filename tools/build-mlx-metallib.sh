#!/usr/bin/env bash
# Build mlx.metallib for the SwiftPM-linked MLX runtime and place it next to
# the mlxfast-swift executable, where Cmlx searches first.
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd)
cd "${ROOT_DIR}"

BUILD_CONFIGURATION="${MLXFAST_SWIFT_CONFIGURATION:-release}"
SWIFT_BIN="${MLXFAST_SWIFT_BIN:-.build/${BUILD_CONFIGURATION}/mlxfast-swift}"
OUTPUT_PATH="${MLXFAST_MLX_METALLIB:-$(dirname "${SWIFT_BIN}")/mlx.metallib}"

MLX_SWIFT_CHECKOUT="${MLXFAST_MLX_SWIFT_CHECKOUT:-.build/checkouts/mlx-swift}"
MLX_SOURCE="${MLX_SWIFT_CHECKOUT}/Source/Cmlx/mlx"
METAL_CPP_SOURCE="${MLX_SWIFT_CHECKOUT}/Source/Cmlx/metal-cpp"
JSON_SOURCE="${MLX_SWIFT_CHECKOUT}/Source/Cmlx/json"
FMT_SOURCE="${MLX_SWIFT_CHECKOUT}/Source/Cmlx/fmt"
CMAKE_BUILD_DIR="${MLXFAST_MLX_METAL_BUILD_DIR:-.build/mlx-metal}"
export CLANG_MODULE_CACHE_PATH="${CLANG_MODULE_CACHE_PATH:-${ROOT_DIR}/.build/clang-module-cache}"
mkdir -p "${CLANG_MODULE_CACHE_PATH}"
METAL_COMPILER_HOME="${MLXFAST_METAL_COMPILER_HOME:-${CMAKE_BUILD_DIR}/.mlxfast-home}"
mkdir -p "${METAL_COMPILER_HOME}"

find_cmake() {
  local candidate
  if [[ -n "${MLXFAST_CMAKE_BIN:-}" ]]; then
    if [[ -x "${MLXFAST_CMAKE_BIN}" ]]; then
      printf '%s\n' "${MLXFAST_CMAKE_BIN}"
      return 0
    fi
    echo "build-mlx-metallib.sh: MLXFAST_CMAKE_BIN is set but not executable: ${MLXFAST_CMAKE_BIN}" >&2
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

metal_toolchain_identifier() {
  xcodebuild -showComponent MetalToolchain 2>/dev/null \
    | awk -F': ' '/Toolchain Identifier/ { print $2; exit }' \
    || true
}

CMAKE_BIN="$(find_cmake)" || {
  cat >&2 <<EOF
build-mlx-metallib.sh: cmake was not found.

Install CMake, then retry:

  brew install cmake

EOF
  exit 1
}

METAL_TOOLCHAIN_IDENTIFIER="${MLXFAST_METAL_TOOLCHAIN_IDENTIFIER:-$(metal_toolchain_identifier)}"

if [[ -n "${METAL_TOOLCHAIN_IDENTIFIER}" ]]; then
  export TOOLCHAINS="${TOOLCHAINS:-${METAL_TOOLCHAIN_IDENTIFIER}}"
fi

if ! xcrun -sdk macosx metal -v >/dev/null 2>&1; then
  if [[ -n "${METAL_TOOLCHAIN_IDENTIFIER}" ]]; then
    echo "build-mlx-metallib.sh: found Metal Toolchain ${METAL_TOOLCHAIN_IDENTIFIER}, but xcrun could not execute metal" >&2
  fi
  cat >&2 <<EOF
build-mlx-metallib.sh: Xcode's Metal Toolchain is not installed.

Install it, then retry:

  xcodebuild -downloadComponent MetalToolchain

EOF
  exit 1
fi

if [[ ! -d "${MLX_SOURCE}" ]]; then
  echo "build-mlx-metallib.sh: resolving Swift package dependencies"
  swift package resolve
fi

for required_path in "${MLX_SOURCE}" "${METAL_CPP_SOURCE}" "${JSON_SOURCE}" "${FMT_SOURCE}"; do
  if [[ ! -d "${required_path}" ]]; then
    echo "build-mlx-metallib.sh: required MLX Swift source path is missing: ${required_path}" >&2
    exit 1
  fi
done

JOBS="${MLXFAST_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

HOME="${METAL_COMPILER_HOME}" "${CMAKE_BIN}" \
  -S "${MLX_SOURCE}" \
  -B "${CMAKE_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_BUILD_GGUF=OFF \
  -DFETCHCONTENT_SOURCE_DIR_METAL_CPP="${ROOT_DIR}/${METAL_CPP_SOURCE}" \
  -DFETCHCONTENT_SOURCE_DIR_JSON="${ROOT_DIR}/${JSON_SOURCE}" \
  -DFETCHCONTENT_SOURCE_DIR_FMT="${ROOT_DIR}/${FMT_SOURCE}"

HOME="${METAL_COMPILER_HOME}" "${CMAKE_BIN}" \
  --build "${CMAKE_BUILD_DIR}" --target mlx-metallib --parallel "${JOBS}"

METALLIB_PATHS=()
while IFS= read -r path; do
  [[ -n "${path}" ]] && METALLIB_PATHS+=("${path}")
done < <(find "${CMAKE_BUILD_DIR}" -type f -name mlx.metallib -print | LC_ALL=C sort)

if [[ "${#METALLIB_PATHS[@]}" -eq 0 ]]; then
  echo "build-mlx-metallib.sh: CMake finished but no mlx.metallib was found under ${CMAKE_BUILD_DIR}" >&2
  exit 1
fi
if [[ "${#METALLIB_PATHS[@]}" -ne 1 ]]; then
  echo "build-mlx-metallib.sh: CMake produced multiple mlx.metallib files under ${CMAKE_BUILD_DIR}:" >&2
  printf '  %s\n' "${METALLIB_PATHS[@]}" >&2
  exit 1
fi
METALLIB_PATH="${METALLIB_PATHS[0]}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"
cp "${METALLIB_PATH}" "${OUTPUT_PATH}"
echo "build-mlx-metallib.sh: wrote ${OUTPUT_PATH}"

if [[ "${BUILD_CONFIGURATION}" == "debug" ]]; then
  while IFS= read -r test_binary_dir; do
    cp "${METALLIB_PATH}" "${test_binary_dir}/mlx.metallib"
    echo "build-mlx-metallib.sh: wrote ${test_binary_dir}/mlx.metallib"
  done < <(find .build -path "*.xctest/Contents/MacOS" -type d)
fi
