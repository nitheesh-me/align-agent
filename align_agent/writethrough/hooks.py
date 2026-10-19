"""Native hook adapters: `align-agent-hook <agent> [event]` reads one hook event as JSON on stdin and
writes the agent's own reply format on stdout. Field names and reply formats follow each agent's
documentation as of October 2026 (see integrations/README.md for the sources and configs).

    agent      events handled                                    after a tool            before a shell command
    claude     SessionStart, PreToolUse, PostToolUse,             additionalContext       permissionDecision deny
               PostToolUseFailure
    codex      SessionStart, PostCompact, PreToolUse, PostToolUse additionalContext       permissionDecision deny
    gemini     SessionStart, PreCompress, BeforeTool, AfterTool   additionalContext       decision deny
    cursor     sessionStart, preCompact, beforeShellExecution,    additional_context      permission deny + agent_message
               beforeReadFile, afterFileEdit, postToolUse,
               postToolUseFailure
    copilot    sessionStart, preCompact, preToolUse, postToolUse, additionalContext       permissionDecision deny
               postToolUseFailure (event given as an argument)
    cline      PreToolUse, PostToolUse (file hooks; event given   contextModification     (none: cancelling ends the task)
               as an argument)

Agents without a hook that can add text after a tool (Goose, Antigravity) use the MCP server
(align-agent-mcp) instead. Every adapter fails open: an error is logged and the agent continues.
"""

import json
import os
import re
import sys
import time

try:
    from .claude_code import handle as claude_handle
    from .engine import Session, default_state_dir
except Exception as e:  # noqa: BLE001  a broken install must not block the agent: main logs it and fails open
    IMPORT_ERROR = e
else:
    IMPORT_ERROR = None

# apply_patch headers; `*** Move to:` names a rename's new path
PATCH_FILE = re.compile(r"(?m)^\*\*\* (?:(?:Update|Add|Delete) File|Move to): (.+?)\r?$")


def _out_text(resp):
    if isinstance(resp, dict):
        for k in ("output", "stdout", "llmContent", "textResultForLlm", "result"):
            if resp.get(k):
                v = resp[k]
                return v if isinstance(v, str) else json.dumps(v)
        return json.dumps(resp)
    if isinstance(resp, str):
        try:
            return _out_text(json.loads(resp))
        except ValueError:
            return resp
    return str(resp or "")


def patched_files(text):
    """The files an apply_patch text adds, updates, deletes or moves to."""
    return [f.strip() for f in PATCH_FILE.findall(text or "") if f.strip()]


def read_lines(lo, hi):
    """A read range from a first line and a last line, either of which may be missing (None: whole file)."""
    if lo is None and hi is None:
        return None
    lo = int(lo or 1)
    return (lo, int(hi) if hi is not None else lo + 1999)


def _args(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except ValueError:
            return {"command": a}
    return a or {}


# ---------------------------------------------------------------- Codex CLI

def codex(p, event=None):
    """Claude Code's schema, but unknown output fields fail the hook, there is no read tool (reads
    go through the shell), edits arrive as apply_patch text, and added context over about 2,500
    tokens is spilled to a file (raise it with additionalContextLimit on the handler)."""
    event = event or p.get("hook_event_name")
    root = p.get("cwd") or os.getcwd()
    with Session(root, p.get("session_id") or "codex", budget=6000) as s:
        if event in ("SessionStart", "PostCompact"):
            s.start(fresh=event == "PostCompact" or p.get("source") != "resume")
            return None
        tool, inp = p.get("tool_name") or "", _args(p.get("tool_input"))
        if event == "PreToolUse":
            reason = s.before(inp.get("command", "")) if tool == "Bash" else None
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                           "permissionDecisionReason": reason}} if reason else None
        if event != "PostToolUse":
            return None
        command = inp.get("command", "") if tool == "Bash" else ""
        authored = []
        if tool in ("apply_patch", "Edit", "Write"):
            text = inp.get("command") or inp.get("patch") or ""
            authored = patched_files(text) or [inp.get("file_path") or inp.get("path")]
        ctx = s.after(command, _out_text(p.get("tool_response")) if command else "", authored=[a for a in authored if a])
        return {"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": ctx}} if ctx else None


# ---------------------------------------------------------------- Gemini CLI

GEMINI_EDIT = {"write_file", "replace", "edit"}
# read_many_files prints each file under a `--- <path> ---` header
GEMINI_MANY = re.compile(r"(?m)^--- (.+?) ---\r?$")


def gemini(p, event=None):
    """BeforeTool/AfterTool. read_file takes start_line/end_line (1-based, inclusive); older versions
    took a 0-based offset and a limit. Added context is appended to the tool's result inside
    <hook_context>; a deny reaches the model as 'Tool execution blocked: <reason>'."""
    event = event or p.get("hook_event_name")
    root = os.environ.get("GEMINI_PROJECT_DIR") or p.get("cwd") or os.getcwd()
    with Session(root, p.get("session_id") or "gemini") as s:
        if event in ("SessionStart", "PreCompress"):
            s.start(fresh=event == "PreCompress" or p.get("source") not in ("resume",))
            return None
        tool, inp = p.get("tool_name") or "", _args(p.get("tool_input"))
        if event == "BeforeTool":
            reason = s.before(inp.get("command", "")) if tool == "run_shell_command" else None
            return {"decision": "deny", "reason": reason} if reason else None
        if event != "AfterTool":
            return None
        command = inp.get("command", "") if tool == "run_shell_command" else ""
        read = rng = None
        if tool == "read_file":
            read = inp.get("file_path") or inp.get("absolute_path")
            if inp.get("start_line") is not None or inp.get("end_line") is not None:
                rng = read_lines(inp.get("start_line"), inp.get("end_line"))
            elif inp.get("offset") is not None or inp.get("limit") is not None:
                lo = int(inp.get("offset") or 0) + 1
                rng = (lo, lo + int(inp.get("limit") or 2000) - 1)
        elif tool == "read_many_files":
            for f in GEMINI_MANY.findall(_out_text(p.get("tool_response"))):
                if f.strip() != "End of content":
                    s.read(f.strip())
        authored = inp.get("file_path") if tool in GEMINI_EDIT else None
        ctx = s.after(command, _out_text(p.get("tool_response")) if command else "", read=read, read_range=rng,
                      authored=authored)
        return {"hookSpecificOutput": {"additionalContext": ctx}} if ctx else None


# ---------------------------------------------------------------- Cursor

def cursor(p, event=None):
    """beforeShellExecution refuses with agent_message (the model reads it; user_message is for the
    person). afterShellExecution cannot add text, so the diff goes through postToolUse. A permission
    hook must always print valid JSON, or Cursor blocks the action."""
    event = event or p.get("hook_event_name")
    roots = p.get("workspace_roots") or []
    root = (roots[0] if roots else None) or p.get("cwd") or os.getcwd()
    key = p.get("conversation_id") or p.get("session_id") or "cursor"
    # a subagent has its own context, so its own views: key by its id when the event carries one
    agent = p.get("subagent_id")
    with Session(root, re.sub(r"[^A-Za-z0-9_.-]", "_", f"{key}-{agent}" if agent else key)) as s:
        if event in ("sessionStart", "preCompact", "subagentStart"):
            s.start(fresh=event != "sessionStart" or p.get("source") != "resume")
            # subagentStart is a permission hook: it must answer, or Cursor may block the subagent
            return {"permission": "allow"} if event == "subagentStart" else {}
        if event == "beforeShellExecution":
            reason = s.before(p.get("command", ""))
            if reason:
                return {"permission": "deny", "agent_message": reason,
                        "user_message": "ALIGN: line numbers in this command are out of date; the agent was shown the current lines."}
            return {"permission": "allow"}
        if event == "beforeReadFile":
            # the view is recorded from postToolUse, which knows the lines a Read asked for
            return {"permission": "allow"}
        if event == "afterFileEdit":
            s.authored(p.get("file_path"))
            return {}
        if event not in ("postToolUse", "postToolUseFailure"):
            return {}
        tool, inp = p.get("tool_name") or "", _args(p.get("tool_input"))
        command = inp.get("command", "") if tool == "Shell" else ""
        path = inp.get("file_path") or inp.get("path") or inp.get("target_file")
        out = p.get("tool_output") if p.get("tool_output") is not None else p.get("error_message")
        rng = None
        if tool == "Read" and (inp.get("offset") or inp.get("limit")):
            lo = int(inp.get("offset") or 1)
            rng = (lo, lo + int(inp.get("limit") or 2000) - 1)
        ctx = s.after(command, _out_text(out) if command else "", read=path if tool == "Read" else None, read_range=rng,
                      authored=path if tool in ("Write", "Edit", "Delete") else None)
        return {"additional_context": ctx} if ctx else {}


# ---------------------------------------------------------------- GitHub Copilot CLI

COPILOT_SHELL = {"bash", "Bash", "powershell", "PowerShell"}


def copilot(p, event=None):
    """camelCase events: {sessionId, cwd, toolName, toolArgs, toolResult}. postToolUse fires only on
    success, so register postToolUseFailure too. Added context is capped at 10 KB. A subagent's
    events carry its agentId; it has its own context, so its own views."""
    root = p.get("cwd") or os.getcwd()
    key = p.get("sessionId") or p.get("session_id") or "copilot"
    agent = p.get("agentId") or p.get("agent_id")
    with Session(root, re.sub(r"[^A-Za-z0-9_.-]", "_", f"{key}-{agent}" if agent else key)) as s:
        if event in ("sessionStart", "preCompact"):
            s.start(fresh=event == "preCompact" or p.get("source") != "resume")
            return None
        tool, inp = p.get("toolName") or p.get("tool_name") or "", _args(p.get("toolArgs") or p.get("tool_input"))
        if event == "preToolUse":
            reason = s.before(inp.get("command", "")) if tool in COPILOT_SHELL else None
            return {"permissionDecision": "deny", "permissionDecisionReason": reason} if reason else None
        if event not in ("postToolUse", "postToolUseFailure"):
            return None
        command = inp.get("command", "") if tool in COPILOT_SHELL else ""
        path = inp.get("path") or inp.get("file_path")
        rng = None
        vr = inp.get("view_range")
        if tool in ("view", "Read") and isinstance(vr, list) and vr:
            hi = vr[1] if len(vr) > 1 and int(vr[1]) != -1 else 10 ** 9  # -1: to the end of the file
            rng = (int(vr[0]), int(hi))
        ctx = s.after(command, _out_text(p.get("toolResult") or p.get("tool_response")) if command else "",
                      read=path if tool in ("view", "Read") else None, read_range=rng,
                      authored=path if tool in ("edit", "create", "Edit", "Write") else None)
        return {"additionalContext": ctx} if ctx else None


# ---------------------------------------------------------------- Cline

def cline(p, event=None):
    """File hooks. PostToolUse's contextModification reaches the model as <hook_context> after the
    tool results. PreToolUse can only cancel the whole task, so the line-number guard is not used."""
    event = event or ("PostToolUse" if "postToolUse" in p else "PreToolUse" if "preToolUse" in p else None)
    roots = p.get("workspaceRoots") or []
    root = (roots[0] if roots else None) or os.getcwd()
    with Session(root, p.get("taskId") or "cline") as s:
        if event != "PostToolUse":
            return {"cancel": False}
        post = p.get("postToolUse") or {}
        tool, prm = post.get("toolName") or "", post.get("parameters") or {}
        commands = prm.get("commands") or ([prm["command"]] if prm.get("command") else [])
        if isinstance(commands, str):
            commands = [commands]
        if tool in ("read_file", "read_files"):
            files = prm.get("files") if isinstance(prm.get("files"), list) else [prm]
            for f in files:
                if isinstance(f, dict) and f.get("path"):
                    rng = read_lines(f.get("start_line"), f.get("end_line"))
                    s.read(f["path"], *(rng or (None, None)))
        authored = None
        if tool == "apply_patch":
            authored = patched_files(prm.get("input")) or ([prm["path"]] if prm.get("path") else None)
        elif tool in ("editor", "write_to_file", "replace_in_file"):
            authored = prm.get("path")
        ctx = s.after(" && ".join(commands), _out_text(post.get("result")) if commands else "", authored=authored)
        return {"cancel": False, "contextModification": ctx} if ctx else {"cancel": False}


AGENTS = {"claude": lambda p, e=None: claude_handle(p), "codex": codex, "gemini": gemini, "cursor": cursor,
          "copilot": copilot, "cline": cline}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in AGENTS:
        sys.stderr.write("usage: align-agent-hook {" + ",".join(AGENTS) + "} [event] < event.json\n")
        return 2
    agent = argv[0]
    event = argv[1] if len(argv) > 1 else None
    try:
        if IMPORT_ERROR is not None:
            raise IMPORT_ERROR
        payload = json.loads(sys.stdin.buffer.read())  # UTF-8 whatever the locale (Windows)
        event = event or payload.get("hook_event_name") or payload.get("hookName")
        reply = AGENTS[agent](payload, event)
    except Exception as e:  # noqa: BLE001  fail open
        state = default_state_dir() if IMPORT_ERROR is None else (os.environ.get("ALIGN_STATE") or os.path.expanduser("~/.cache/align-agent"))
        log = os.path.join(state, "errors.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {agent} {type(e).__name__}: {e}\n")
        # Cursor blocks an action when a permission hook prints no valid JSON
        reply = {"permission": "allow"} if agent == "cursor" and str(event or "").startswith("before") else None
    if reply is not None:
        json.dump(reply, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
