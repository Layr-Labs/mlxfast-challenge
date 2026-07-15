#!/usr/bin/env bash
# Ensure private MTP-track benchmark material is only used by this repository's
# MTP benchmark workflow on an explicitly permitted branch namespace. Sibling
# of enforce-trusted-benchmark-workflow.sh (which pins the SERIAL ranked
# workflow); the two tracks stay pinned to their own distinct workflow files.
set -euo pipefail

readonly TRUSTED_REPOSITORY="Layr-Labs/mlxfast-challenge-dev"
readonly WORKFLOW_PATH=".github/workflows/mtp-benchmark.yml"

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_REF:?GITHUB_REF is required}"
: "${GITHUB_WORKFLOW_REF:?GITHUB_WORKFLOW_REF is required}"
: "${GITHUB_EVENT_NAME:?GITHUB_EVENT_NAME is required}"

if [[ "${GITHUB_REPOSITORY}" != "${TRUSTED_REPOSITORY}" ]]; then
  echo "::error::private MTP benchmark workflow must run in ${TRUSTED_REPOSITORY}, not ${GITHUB_REPOSITORY}" >&2
  exit 1
fi

if [[ "${GITHUB_EVENT_NAME}" != "workflow_dispatch" ]]; then
  echo "::error::private MTP benchmark workflow only supports workflow_dispatch" >&2
  exit 1
fi

case "${GITHUB_REF}" in
  refs/heads/main|refs/heads/submissions/*|refs/heads/baseline/*|refs/heads/yukon/baseline/*)
    ;;
  *)
    echo "::error::private MTP benchmark workflow ref is not allowed: ${GITHUB_REF}" >&2
    echo "::error::allowed branches are main, submissions/*, baseline/*, and yukon/baseline/*" >&2
    exit 1
    ;;
esac

expected_workflow_ref="${TRUSTED_REPOSITORY}/${WORKFLOW_PATH}@${GITHUB_REF}"
if [[ "${GITHUB_WORKFLOW_REF}" != "${expected_workflow_ref}" ]]; then
  echo "::error::unexpected workflow ref ${GITHUB_WORKFLOW_REF}" >&2
  echo "::error::expected ${expected_workflow_ref}" >&2
  exit 1
fi

echo "mtp-benchmark: trusted workflow verified ${GITHUB_WORKFLOW_REF}"
