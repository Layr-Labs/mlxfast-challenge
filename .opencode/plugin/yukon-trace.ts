// Yukon OpenCode trace capture — yukon-trace-config-v3.
//
// This plugin records OpenCode events for trace upload only. It does not register a gated CLI
// session; Claude, Codex, and Cursor are the supported session-gate agents.
import { appendFileSync, existsSync, unlinkSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";

type OpenCodeEvent = {
  type?: string;
  properties?: { sessionID?: unknown };
};

const CLI = "mlxfast";

// Use one temporary transcript per OpenCode process so concurrent sessions never share events.
const FILE = join(
  tmpdir(),
  `yukon-oc-${Buffer.from(process.cwd()).toString("hex").slice(0, 24)}-${process.pid}.ndjson`,
);

export const YukonTrace = async () => {
  try {
    if (existsSync(FILE)) unlinkSync(FILE);
  } catch {}

  let sessionId = "";

  // Upload the accumulated transcript at turn and process boundaries. Capture is best-effort and
  // must never interrupt the OpenCode session.
  const flush = () => {
    try {
      if (!existsSync(FILE)) return;
      spawnSync(CLI, ["trace", "hook", "opencode"], {
        input: JSON.stringify({ transcript_path: FILE, session_id: sessionId }),
        stdio: ["pipe", "ignore", "ignore"],
      });
    } catch {}
  };

  return {
    event: ({ event }: { event: OpenCodeEvent }) => {
      try {
        appendFileSync(FILE, `${JSON.stringify(event)}\n`);
      } catch {}

      if (sessionId === "" && typeof event.properties?.sessionID === "string") {
        sessionId = event.properties.sessionID;
      }
      if (event.type === "session.idle" || event.type === "server.instance.disposed") flush();
    },
    dispose: async () => { flush(); },
  };
};
