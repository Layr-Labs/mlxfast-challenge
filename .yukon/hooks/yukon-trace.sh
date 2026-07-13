#!/usr/bin/env sh
# yukon-trace-config-v3
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
#   - parsing, session state, redaction, and upload live in the CLI;
#   - credentials come from the CLI config (~/.config), never written into the repo;
#   - capture is best-effort; session registration is synchronous.
agent="${1:-unknown}"
kind="${2:-capture}"
detail="${3:-}"
payload="$(cat 2>/dev/null || true)"
# This script lives at <repo>/.yukon/hooks/yukon-trace.sh; resolve the repo root for the CLI.
repo="$(CDPATH= cd -- "$(dirname -- "$0")/../.." 2>/dev/null && pwd)" || repo="$PWD"
if [ "$kind" = "session" ]; then
  cd "$repo" 2>/dev/null || exit 1
  printf '%s' "$payload" | mlxfast trace session "$agent"
  exit $?
fi
( cd "$repo" 2>/dev/null && printf '%s' "$payload" | mlxfast trace hook "$agent" >/dev/null 2>&1 & ) >/dev/null 2>&1 || true
# Codex Stop/SubagentStop expect JSON on stdout when they exit 0; UserPromptSubmit must print
# nothing (its stdout is injected into the prompt as developer context). Exit 0 = success for all.
[ "$agent" = "codex" ] && [ "$detail" != "UserPromptSubmit" ] && printf '{"continue": true}\n'
exit 0
