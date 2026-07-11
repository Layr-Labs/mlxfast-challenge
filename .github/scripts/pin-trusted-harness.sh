#!/usr/bin/env bash
# Capture or verify a content pin over the trusted harness artifacts that drive
# the scored phases: the benchmark driver (benchmark.sh), the built CLI binary
# (.build/release/mlxfast-swift), and the MLX metal library
# (.build/release/mlx.metallib).
#
# Why: the submitted transform runs AFTER the trusted build with write access
# to the bench workspace, and the correctness/gates and timed phases re-run
# ./benchmark.sh and the binary FROM that same workspace with no rebuild in
# between. A replaced driver or binary could therefore forge a self-consistent
# passing score or exfiltrate hidden material (a TOCTOU on the workspace).
# benchmark.sh is additionally ACL-locked against bench writes, but the built
# binary and metallib are bench-owned build outputs that ACLs cannot protect,
# so their integrity anchor is this content pin: capture it right after the
# trusted-controlled build, then re-verify immediately before every phase that
# executes the binary as bench. Any mismatch fails the run closed.
#
# The pin file lives under the 0700 runner-only private dir, so the bench uid
# can neither read nor rewrite it.
set -euo pipefail

MODE="${1:?usage: pin-trusted-harness.sh <write|verify> WORKSPACE PIN_FILE}"
WORKSPACE="${2:?workspace path is required}"
PIN_FILE="${3:?pin file path is required}"

# The trusted driver + built artifacts every scored phase depends on.
PINNED_RELATIVE_PATHS=(
  "benchmark.sh"
  ".build/release/mlxfast-swift"
  ".build/release/mlx.metallib"
)

compute_pin() {
  local rel path
  for rel in "${PINNED_RELATIVE_PATHS[@]}"; do
    path="${WORKSPACE}/${rel}"
    if [[ -L "${path}" || ! -f "${path}" ]]; then
      echo "::error::trusted harness artifact missing or not a regular file: ${rel}" >&2
      return 1
    fi
    printf '%s  %s\n' "$(shasum -a 256 "${path}" | awk '{print $1}')" "${rel}"
  done
}

case "${MODE}" in
  write)
    tmp="$(mktemp "${PIN_FILE}.XXXXXX")"
    if ! compute_pin > "${tmp}"; then
      rm -f "${tmp}"
      exit 1
    fi
    mv "${tmp}" "${PIN_FILE}"
    chmod 600 "${PIN_FILE}"
    echo "benchmark: pinned trusted harness artifacts -> ${PIN_FILE}"
    ;;
  verify)
    if [[ ! -s "${PIN_FILE}" ]]; then
      echo "::error::trusted harness pin file missing: ${PIN_FILE}" >&2
      exit 1
    fi
    if ! current="$(compute_pin)"; then
      exit 1
    fi
    if [[ "${current}" != "$(cat "${PIN_FILE}")" ]]; then
      echo "::error::trusted harness artifacts changed since the trusted build (possible in-workspace tamper of benchmark.sh or the CLI binary)" >&2
      exit 1
    fi
    echo "benchmark: trusted harness artifacts verified against the post-build pin"
    ;;
  *)
    echo "::error::unknown mode ${MODE}; expected write or verify" >&2
    exit 2
    ;;
esac
