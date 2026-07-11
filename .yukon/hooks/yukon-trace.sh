#!/usr/bin/env sh
# mlxfast agent trace hook — shared wrapper (Claude Code, Cursor, Codex).
#
# Adapted from each agent's official hook docs + the most-referenced open-source integrations,
# kept intentionally thin (all logic lives in the CLI):
#   - Claude Code hooks (Stop/SessionEnd): https://code.claude.com/docs/en/hooks
#     technique from Langfuse Claude-Observability-Plugin (MIT):
#     https://github.com/langfuse/Claude-Observability-Plugin
#   - Cursor hooks: https://cursor.com/docs/hooks
#     pattern from naoufalelh/cursor-langfuse (MIT)
#   - Codex hooks: https://developers.openai.com/codex/hooks (Stop hook, JSON on stdin)
#
# Changed vs those references:
#   - forwards to mlxfast via `mlxfast trace hook <agent>` instead of Langfuse;
#   - parsing/redaction/upload live in the CLI (this stays a one-liner);
#   - credentials come from the CLI config (~/.config), never written into the repo;
#   - silent-fail so it can never block or slow the agent.
agent="${1:-unknown}"
event="${2:-}"
shift 2>/dev/null || true
shift 2>/dev/null || true
# Read the hook event from stdin now, before we background the upload (this wrapper exits
# immediately, closing its stdin).
payload="$(cat 2>/dev/null || true)"
# This script lives at <repo>/.yukon/hooks/yukon-trace.sh; resolve the repo root for the CLI.
# `unset CDPATH` in the subshell keeps a caller-exported CDPATH from perturbing
# `cd` (same intent as the older `CDPATH= cd` prefix, without tripping SC1007).
repo="$(unset CDPATH; cd -- "$(dirname -- "$0")/../.." 2>/dev/null && pwd)" || repo="$PWD"
# Upload in a detached background process so the hook returns instantly: it never adds latency to a
# turn and is never "cancelled" for outliving the agent's shutdown grace. Fired on every user
# message + turn end, so frequent uploads keep the server's merged transcript current.
( cd "$repo" 2>/dev/null && printf '%s' "$payload" | mlxfast trace hook "$agent" "$@" >/dev/null 2>&1 & ) >/dev/null 2>&1 || true
# Codex Stop/SubagentStop expect JSON on stdout when they exit 0; UserPromptSubmit must print
# nothing (its stdout is injected into the prompt as developer context). Exit 0 = success for all.
[ "$agent" = "codex" ] && [ "$event" != "UserPromptSubmit" ] && printf '{"continue": true}\n'
exit 0
