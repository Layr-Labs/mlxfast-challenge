#!/usr/bin/env bash
# Ensure private benchmark material is only used by this repository's benchmark
# workflow file on trusted main. Candidate commits are passed to that workflow
# as data and cannot select the workflow ref.
set -euo pipefail

readonly TRUSTED_REPOSITORY="Layr-Labs/mlxfast-challenge-dev"
readonly WORKFLOW_PATH=".github/workflows/benchmark.yml"
readonly TRUSTED_REF="refs/heads/main"

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_REF:?GITHUB_REF is required}"
: "${GITHUB_WORKFLOW_REF:?GITHUB_WORKFLOW_REF is required}"
: "${GITHUB_EVENT_NAME:?GITHUB_EVENT_NAME is required}"

expected_workflow_ref="${TRUSTED_REPOSITORY}/${WORKFLOW_PATH}@${TRUSTED_REF}"

if [[ "${GITHUB_REPOSITORY}" != "${TRUSTED_REPOSITORY}" ]]; then
  echo "::error::private benchmark workflow must run in ${TRUSTED_REPOSITORY}, not ${GITHUB_REPOSITORY}" >&2
  exit 1
fi

if [[ "${GITHUB_EVENT_NAME}" != "workflow_dispatch" ]]; then
  echo "::error::private benchmark workflow only supports workflow_dispatch" >&2
  exit 1
fi

if [[ "${GITHUB_REF}" != "${TRUSTED_REF}" ]]; then
  echo "::error::private benchmark workflow must run from ${TRUSTED_REF}; current ref is ${GITHUB_REF}" >&2
  exit 1
fi

if [[ "${GITHUB_WORKFLOW_REF}" != "${expected_workflow_ref}" ]]; then
  echo "::error::unexpected workflow ref ${GITHUB_WORKFLOW_REF}" >&2
  echo "::error::expected ${expected_workflow_ref}" >&2
  exit 1
fi

echo "benchmark: trusted workflow verified ${GITHUB_WORKFLOW_REF}"
