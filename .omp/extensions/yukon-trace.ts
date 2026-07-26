// Yukon OMP trace transport — yukon-omp-trace-extension-v1.
//
// OMP sessions are native append-only JSONL files. This extension forwards only low-frequency
// lifecycle metadata; Yukon owns transcript reading, redaction, checkpoints, retry, and upload.
import { spawn } from "node:child_process";

const WRAPPER = ".yukon/hooks/yukon-trace.sh";
const ROOT_SESSION_ID = "YUKON_OMP_SESSION_ID";

type OmpContext = {
  cwd: string;
  sessionManager: {
    getHeader(): { id: string } | null;
    getSessionFile(): string | undefined;
  };
};

type OmpExtension = {
  pi: {
    settings: {
      getShellConfig(): { env: Record<string, string> };
    };
  };
  on(
    event: "session_start" | "before_agent_start" | "agent_end" | "session_shutdown",
    handler: (event: unknown, context: OmpContext) => Promise<void>,
  ): void;
};

export default function YukonTrace(pi: OmpExtension): void {
  let parentSessionId: string | undefined;

  pi.on("session_start", async (_event, context) => {
    const session = currentSession(context);
    if (session === null) return;
    parentSessionId = process.env[ROOT_SESSION_ID];
    if (parentSessionId === undefined) {
      setRootSessionId(pi, session.sessionId);
      await forward(context, "session", "sessionStart");
    }
    await forward(context, "capture", "sessionStart", parentSessionId);
  });
  pi.on("before_agent_start", async (_event, context) => {
    await forward(context, "capture", "beforeAgentStart", parentSessionId);
  });
  pi.on("agent_end", async (_event, context) => {
    await forward(context, "capture", "agentEnd", parentSessionId);
  });
  pi.on("session_shutdown", async (_event, context) => {
    await forward(context, "capture", "sessionShutdown", parentSessionId);
    const session = currentSession(context);
    if (session === null) return;
    if (parentSessionId === undefined) clearRootSessionId(pi, session.sessionId);
  });
}

function currentSession(context: OmpContext): { sessionId: string; transcriptPath: string } | null {
  const header = context.sessionManager.getHeader();
  const transcriptPath = context.sessionManager.getSessionFile();
  return header === null || transcriptPath === undefined
    ? null
    : { sessionId: header.id, transcriptPath };
}

function setRootSessionId(pi: OmpExtension, sessionId: string): void {
  const shellEnv = pi.pi.settings.getShellConfig().env;
  process.env[ROOT_SESSION_ID] = sessionId;
  shellEnv[ROOT_SESSION_ID] = sessionId;
}

function clearRootSessionId(pi: OmpExtension, sessionId: string): void {
  const shellEnv = pi.pi.settings.getShellConfig().env;
  if (process.env[ROOT_SESSION_ID] === sessionId) delete process.env[ROOT_SESSION_ID];
  if (shellEnv[ROOT_SESSION_ID] === sessionId) delete shellEnv[ROOT_SESSION_ID];
}

async function forward(
  context: OmpContext,
  kind: "session" | "capture",
  hookEventName: string,
  parentSessionId?: string,
): Promise<void> {
  try {
    const session = currentSession(context);
    if (session === null) return;
    const payload = JSON.stringify({
      hook_event_name: hookEventName,
      session_id: session.sessionId,
      transcript_path: session.transcriptPath,
      parent_session_id: parentSessionId,
    });
    await runWrapper(context.cwd, kind, payload);
  } catch {
    // Trace transport must never alter OMP's event processing.
  }
}

function runWrapper(cwd: string, kind: "session" | "capture", payload: string): Promise<void> {
  return new Promise((resolve) => {
    const child = spawn("sh", [WRAPPER, "omp", kind], {
      cwd,
      stdio: ["pipe", "ignore", "ignore"],
    });
    child.on("error", () => resolve());
    child.on("close", () => resolve());
    child.stdin.on("error", () => {});
    child.stdin.end(payload);
  });
}
