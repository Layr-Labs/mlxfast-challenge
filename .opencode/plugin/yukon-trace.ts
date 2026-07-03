// mlxfast agent trace hook — opencode plugin.
// Adapted from opencode plugin docs (https://opencode.ai/docs/plugins) and the event set in
// @devtheops/opencode-plugin-otel (MPL-2.0, https://github.com/DEVtheOPS/opencode-plugin-otel).
// Changed: instead of emitting OTLP, events are batched and piped to `mlxfast trace hook opencode`,
// which reads creds from the CLI config and uploads to mlxfast; silent-fail.
//
// Accumulate every event to a FILE (an in-memory buffer doesn't survive across handler calls, but a
// file does — the factory runs once, cwd is stable). The file is keyed by cwd AND pid so two opencode
// processes in the same repo (e.g. concurrent runs) never share it. On each turn boundary
// (`session.idle`) and at shutdown (`server.instance.disposed`), hand the WHOLE file to the CLI as a
// transcript via a BLOCKING spawnSync, tagged with the root session id so the server keeps one
// per-session object — overwrite semantics, refreshed every turn, exactly like Claude/Codex. (We do
// NOT truncate; the file is the growing session transcript, and it captures sub-agent events too,
// since opencode delivers them on the same bus.) Cleared at session start; creds come from the config.
import { appendFileSync, existsSync, unlinkSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";

const FILE = join(tmpdir(), "yukon-oc-" + Buffer.from(process.cwd()).toString("hex").slice(0, 24) + "-" + process.pid + ".ndjson");

export const YukonTrace = async () => {
  try { if (existsSync(FILE)) unlinkSync(FILE); } catch {}
  let sessionId = "";
  const flush = () => {
    try {
      if (!existsSync(FILE)) return;
      spawnSync("mlxfast", ["trace", "hook", "opencode"], { input: JSON.stringify({ transcript_path: FILE, session_id: sessionId }), stdio: ["pipe", "ignore", "ignore"] });
    } catch {
      // tracing is best-effort; never disrupt the session
    }
  };
  return {
    event: ({ event }) => {
      try { appendFileSync(FILE, JSON.stringify(event) + "\n"); } catch {}
      // Pin the object to the first (root) session id we see, so per-turn flushes overwrite one object.
      if (sessionId === "" && event && event.properties && typeof event.properties.sessionID === "string") sessionId = event.properties.sessionID;
      const t = event && event.type;
      if (t === "session.idle" || t === "server.instance.disposed") flush();
    },
    // Awaited shutdown flush: the session.idle / server.instance.disposed bus events are
    // fire-and-forget (the runtime doesn't wait for handlers), so a final turn can be cut off
    // mid-upload on exit. dispose IS awaited, making the last flush reliable.
    dispose: async () => { flush(); },
  };
};
