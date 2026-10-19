"""Claude Code session logs and Codex CLI rollouts as steps for the replay.

Both are JSON Lines files written by the agent itself while a developer works,
so `align-agent analyze` can audit one's own past sessions. Each file is one
trajectory. The adapters map the agents' native tools onto the step schema of
the RQ1 audit (scripts/audit/extract.py), the same schema the published
formats use:

  Claude Code   Bash          shell parser (scripts/audit/shellparse.py), cwd from the log line
                Read          a view of the lines the result shows (numbered, `cat -n` style)
                Edit          an anchored edit on quoted text (str_replace); MultiEdit, one per edit. A
                              result that shows the edited lines or states that the model's copy is
                              current is an editor echo, as in SWE-agent and OpenHands; one that says
                              the file had other changes on disk is not
                Write         a whole-file write; its text is the agent's own, so it is also a view
                NotebookEdit  a write the replay tracks but does not score (cell anchors are not modelled)
                attachments   `edited_text_file` (Claude Code's notice that a read file changed) and
                              `file` (an @-mentioned file) are views; hook output is appended to the
                              tool result it belongs to, so a write-through block counts as a view
                sidechains    lines with the other isSidechain value than the file's own agent, and the
                              session's subagents/*.jsonl transcripts, share the workspace but not the
                              context: their writes enter the timeline (by timestamp, between the
                              session's own turns) as writes the session did not see, and nothing they
                              viewed counts as seen. A subagent transcript given as input is analyzed
                              as its own trajectory.
  Codex CLI     shell, shell_command, exec_command, local_shell_call, container.exec
                              shell parser, cwd from `workdir` or the turn's cwd
                apply_patch   (custom tool, function, or `apply_patch <<EOF` / ["apply_patch", patch]
                              as a command): `*** Update File` hunks are anchored edits, `*** Add File`
                              a whole-file write, `*** Delete File` a delete, `*** Move to` a move
                exec          (code mode): commands passed to tools.exec_command as string literals,
                              directly or through a mapped array; other scripts are counted, not parsed

One composition unit is one model response: an assistant message's tool calls
in Claude Code, consecutive calls without an output between them in Codex. A
context compaction starts a new context (`conv`), since the agent no longer
holds what it viewed before it. Everything the adapters do not map is counted
in the coverage record.
"""

import json
import ntpath
import os
import posixpath
import re
import shlex
import sys
from collections import Counter

try:
    import shellparse as sp  # the RQ1 audit's parser, put on sys.path by core
except ImportError:  # imported before core: same lookup as core._audit_dir()
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.abspath(os.path.join(_here, "..", "..", "scripts", "audit"))
    sys.path.insert(0, _repo if os.path.isdir(_repo) else os.path.join(_here, "_audit"))
    import shellparse as sp
from extract import shell_outcome, tok  # noqa: E402

SNIFF = 64

CLAUDE_TYPES = {"user", "assistant", "system", "summary", "attachment", "file-history-snapshot"}
CODEX_TYPES = {"session_meta", "turn_context", "response_item", "event_msg", "compacted"}
RESPONSE_TYPES = {"message", "function_call", "function_call_output", "reasoning", "local_shell_call",
                  "custom_tool_call", "custom_tool_call_output", "web_search_call"}


def records(path):
    """(record, None) per line of a JSON Lines file, or (None, reason) for a line that is not a JSON object."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if not ln.strip():
                continue
            try:
                d = json.loads(ln)
            except ValueError:
                yield None, "malformed"
                continue
            yield (d, None) if isinstance(d, dict) else (None, "not_an_object")


def shape(d):
    """'claudecode', 'codex' or None for one record."""
    t = d.get("type")
    if "sessionId" in d or ("parentUuid" in d and "uuid" in d) or (t == "summary" and "leafUuid" in d):
        return "claudecode"
    if "payload" in d and t in CODEX_TYPES:
        return "codex"
    if "record_type" in d or ("payload" not in d and t in RESPONSE_TYPES):
        return "codex"  # rollouts written before records carried a payload
    return None


def sniff(path):
    """The session format of a .jsonl file from the shapes of its first lines, or None."""
    votes, n = Counter(), 0
    for d, _ in records(path):
        n += 1
        if d is not None:
            s = shape(d)
            if s:
                votes[s] += 1
        if n >= SNIFF:
            break
    return votes.most_common(1)[0][0] if votes else None


def _text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(_text(x) for x in c)
    if isinstance(c, dict):
        if isinstance(c.get("text"), str):
            return c["text"]
        if c.get("type") in ("image", "document"):
            return f"[{c['type']}]"
        if "content" in c:
            return _text(c["content"])
    return ""


HOME_OF = re.compile(r"^(/Users/[^/]+|/home/[^/]+|/root)(?:/|$)")
TILDE = re.compile(r"(^|[\s=:;|&(<>'\"])~(?=/|$|[\s;|&)'\"])", re.M)


def expand_home(script, cwd):
    """Local sessions run under the user's home, which the audit's parser (written for containers)
    takes to be /root: expand an unquoted-path `~` to the home that holds the session's cwd."""
    m = HOME_OF.match(cwd or "")
    if not m or "~" not in script:
        return script
    return TILDE.sub(lambda t: t.group(1) + m.group(1), script)


ANSI_ESC = {"n": "\n", "t": "\t", "r": "\r", "a": "\a", "b": "\b", "f": "\f", "v": "\v", "e": "\x1b", "E": "\x1b",
            "\\": "\\", "'": "'", '"': '"', "?": "?"}


def _ansi_c(body):
    out, i = [], 0
    while i < len(body):
        ch = body[i]
        if ch != "\\" or i + 1 == len(body):
            out.append(ch)
            i += 1
            continue
        nx = body[i + 1]
        if nx in ANSI_ESC:
            out.append(ANSI_ESC[nx])
            i += 2
        elif nx == "x":
            m = re.match(r"[0-9A-Fa-f]{1,2}", body[i + 2:])
            out.append(chr(int(m.group(0), 16)) if m else "\\x")
            i += 2 + (len(m.group(0)) if m else 0)
        elif nx in "01234567":
            m = re.match(r"[0-7]{1,3}", body[i + 1:])
            out.append(chr(int(m.group(0), 8)))
            i += 1 + len(m.group(0))
        else:
            out.append("\\" + nx)
            i += 2
    return "".join(out)


def ansi_c_unquote(script):
    """Rewrite bash ANSI-C strings ($'...') as plain single-quoted strings with the escapes decoded,
    which the audit's parser reads; quotes and comments elsewhere are left as they are."""
    if "$'" not in script:
        return script
    out, i, q = [], 0, None
    while i < len(script):
        ch = script[i]
        if q == "'":
            out.append(ch)
            q = None if ch == "'" else q
            i += 1
            continue
        if q == '"':
            out.append(ch)
            if ch == "\\" and i + 1 < len(script):
                out.append(script[i + 1])
                i += 2
                continue
            q = None if ch == '"' else q
            i += 1
            continue
        if ch == "\\" and i + 1 < len(script):
            out.append(script[i:i + 2])
            i += 2
            continue
        if ch == "$" and script[i + 1:i + 2] == "'":
            j = i + 2
            while j < len(script) and script[j] != "'":
                j += 2 if script[j] == "\\" else 1
            if j >= len(script):
                return script  # unterminated: leave it to the parser
            out.append(shlex.quote(_ansi_c(script[i + 2:j])))
            i = j + 1
            continue
        if ch in "'\"":
            q = ch
        out.append(ch)
        i += 1
    return "".join(out)


def shell_text(script, cwd):
    """A local session's command as the audit's parser should read it."""
    return ansi_c_unquote(expand_home(script, cwd))


def _step(ops, action, obs, rc, cwd, tool, unit, conv, ts=None, thought=""):
    return {"i": 0, "unit": unit, "conv": conv, "bench": None, "tool": tool, "action": action,
            "thought": thought[-1500:], "obs": obs, "rc": rc, "cwd": cwd, "ran": True, "ops": ops,
            "tok_a": tok(action), "tok_o": tok(obs), "ts": ts}


OTHER_WRITERS = ("sidechain:", "subagent:", "parent:", "peer:")


def unseen(ops, who):
    """Another agent's writes, as the session sees them: changes to files it did not watch happen.
    An anchored edit keeps its anchor (old and new text), so the classifier can compare texts; its
    kind becomes `foreign`, which the replay applies as a write but never scores as the session's edit."""
    out = []
    for o in ops:
        if o["op"] != "W" or o.get("landed") is False:
            continue
        if o["kind"] in ("restore", "delete", "touch"):
            out.append(dict(o, mech=f"{who}:{o.get('mech')}"))
        elif o["kind"] == "anchored" and o.get("anchor"):
            out.append(dict(o, kind="foreign", mech=f"{who}:{o.get('mech')}"))
        else:
            out.append({"op": "W", "path": o["path"], "kind": "replace", "mech": f"{who}:{o.get('mech')}",
                        "landed": o.get("landed"), "ev": o.get("ev"), "cmd_i": o.get("cmd_i", 0),
                        "guard": o.get("guard"), "verb": o.get("verb")})
    return out


def merge(own, others):
    """Insert the other agents' write steps between the session's own units, by timestamp."""
    others = sorted((s for s in others if s["ops"]), key=lambda s: s["ts"] or "")
    out, j = [], 0
    prev = None
    for s in own:
        if s["unit"] != prev and s.get("ts"):
            while j < len(others) and (others[j]["ts"] or "") < s["ts"]:
                out.append(dict(others[j], conv=s["conv"]))
                j += 1
        out.append(s)
        prev = s["unit"]
    conv = own[-1]["conv"] if own else 0
    out += [dict(s, conv=conv) for s in others[j:]]
    return out


def renumber(steps):
    """Sequential step indices and composition units (a unit is a run of steps with one key)."""
    unit, prev = -1, object()
    for i, s in enumerate(steps):
        if s["unit"] != prev:
            unit += 1
            prev = s["unit"]
        s["i"], s["unit"] = i, unit
    return steps


# ------------------------------------------------------------------ apply_patch

def parse_apply_patch(body, cwd):
    """Per-file operations of an apply_patch envelope: [(action, path, hunks, added text, move_to)]."""
    files, cur, hunk = [], None, None
    for ln in (body or "").split("\n"):
        m = re.match(r"^\*\*\* (Update|Add|Delete) File: (.+?)\s*$", ln)
        if m:
            cur = {"action": m.group(1), "path": sp.resolve(m.group(2), cwd), "hunks": [], "added": [], "move": None}
            files.append(cur)
            hunk = None
            continue
        m = re.match(r"^\*\*\* Move to: (.+?)\s*$", ln)
        if m and cur is not None:
            cur["move"] = sp.resolve(m.group(1), cwd)
            continue
        if ln.startswith("***") or cur is None:
            continue  # Begin/End Patch, End of File
        if cur["action"] == "Add":
            if ln.startswith("+"):
                cur["added"].append(ln[1:])
            continue
        if ln.startswith("@@"):
            hunk = {"old_start": None, "new_start": None, "lines": [], "header": ln[2:].strip()}
            cur["hunks"].append(hunk)
            continue
        if ln[:1] in (" ", "+", "-") or ln == "":
            if hunk is None:
                hunk = {"old_start": None, "new_start": None, "lines": [], "header": ""}
                cur["hunks"].append(hunk)
            hunk["lines"].append(ln if ln else " ")
    for f in files:  # a trailing blank line of the envelope is not context
        for h in f["hunks"]:
            while h["lines"] and h["lines"][-1] == " ":
                h["lines"].pop()
    return files


def patch_ops(body, cwd):
    ops = []
    for f in parse_apply_patch(body, cwd):
        p = f["path"]
        if f["action"] == "Add":
            ops.append({"op": "W", "path": p, "kind": "overwrite", "mech": "applypatch-add",
                        "text": "\n".join(f["added"]) + "\n"})
        elif f["action"] == "Delete":
            ops.append({"op": "W", "path": p, "kind": "delete", "mech": "applypatch-delete"})
        else:
            if f["hunks"]:
                ops.append({"op": "W", "path": p, "kind": "anchored", "mech": "applypatch",
                            "anchor": {"type": "patch", "hunks": f["hunks"]}})
            if f["move"]:
                ops.append({"op": "W", "path": f["move"], "kind": "replace", "mech": "applypatch-move", "src": p})
                ops.append({"op": "W", "path": p, "kind": "delete", "mech": "applypatch-move"})
    for o in ops:
        o.update(cmd_i=0, guard=";", verb="apply_patch")
    return ops


PATCH_CONTENT_BAD = re.compile(r"Failed to find (?:expected lines|context)|verification failed")
PATCH_OK = re.compile(r"^Success\.|Updated the following files|^Done!", re.M)
PATCH_BAD = re.compile(r"verification failed|Failed to find|[Ii]nvalid patch|[Ii]nvalid hunk|error:|No such file|rejected|"
                       r"patch rejected|Failed to (?:read|write|apply)")


def patch_outcome(ops, obs, success=None):
    if success is not None:
        landed, ev = bool(success), "apply_patch reported " + ("success" if success else "failure")
    elif PATCH_BAD.search(obs or "") and not PATCH_OK.search(obs or ""):
        landed, ev = False, "apply_patch failed"
    elif PATCH_OK.search(obs or ""):
        landed, ev = True, "apply_patch reported success"
    else:
        landed, ev = None, "apply_patch result unclear"
    # a patch whose context lines were not in the file ran against the agent's view and failed: scored
    content_failure = landed is False and bool(PATCH_CONTENT_BAD.search(obs or ""))
    for o in ops:
        o["ran"] = landed is not False or content_failure
        o["landed"], o["ev"] = landed, ev
        if landed is False:
            o["ev"] = "patch failed: " + ev


HEREDOC_PATCH = re.compile(r"^\s*(?:cd\s+(\"[^\"]+\"|'[^']+'|\S+)\s*&&\s*)?apply_patch\s*<<-?\s*(['\"]?)(\w+)\2[^\n]*\n(.*?)\n\3\s*$", re.S)


def shell_patch(script, cwd):
    """(patch body, cwd) when a shell command is only an apply_patch invocation (Codex runs these itself)."""
    m = HEREDOC_PATCH.match(script or "")
    if m:
        d = m.group(1)
        return m.group(4), (sp.resolve(d.strip("\"'"), cwd) if d else cwd)
    if (script or "").lstrip().startswith("apply_patch"):
        try:
            argv = shlex.split(script)
        except ValueError:
            return None, cwd
        if len(argv) == 2 and argv[0] == "apply_patch":
            return argv[1], cwd
    return None, cwd


# ------------------------------------------------------------------ Claude Code

NOT_RUN = re.compile(r"tool use was rejected|doesn't want to proceed|Permission to use \S+ (?:has been )?denied|"
                     r"[Bb]locked by (?:a )?hook|PreToolUse:\S* hook (?:error|blocking)|Interrupted by user|Request interrupted")
CC_EXIT = re.compile(r"^Exit code (-?\d+)")
# Claude Code wraps every refused call in <tool_use_error>. For an edit, a refusal about the file's text
# (anchor missing or ambiguous, file changed since read) means the edit ran against a view: it is kept as
# a failed edit and scored. Any other refusal (input validation, a cancelled parallel call) did not run.
TOOL_USE_ERROR = "<tool_use_error>"
EDIT_REJECTED = re.compile(r"String to replace not found|not found in file|Found \d+ matches of the string to replace|"
                           r"modified since read")
# Claude Code's own staleness signals: a refusal, or a note on an edit that applied anyway
CC_STALE_REFUSED = re.compile(r"modified since read")
CC_STALE_NOTED = re.compile(r"modified on disk since you last read it")
CC_EDIT_OK = re.compile(r"has been updated successfully|All occurrences were successfully replaced")
CC_BG_ID = re.compile(r"running in background with ID: ([\w-]+)")


def _cc_edit_ev(text):
    first = (text or "").strip().split("\n")[0][:160]
    if re.search(r"String to replace not found|not found in file", text or ""):
        return "editor rejected: old_string did not appear verbatim (" + first + ")"
    if re.search(r"Found \d+ matches of the string to replace", text or ""):
        return "editor rejected: Multiple occurrences of old_string (" + first + ")"
    if re.search(r"modified since read", text or ""):
        return "refused by Claude Code: file modified since read"
    if re.search(r"has not been read yet", text or ""):
        return "refused by Claude Code: file not read yet"
    return "editor rejected: " + first


def _cc_ops(call, cwd, known, cov):
    """Ops of one Claude Code tool call with its result."""
    name, inp = call["name"], call["input"] if isinstance(call["input"], dict) else {}
    res, err, tur = call.get("result"), call.get("is_error"), call.get("tur")
    if res is not None and err:
        head = res[:2000]
        content_refusal = name in ("Edit", "MultiEdit") and EDIT_REJECTED.search(head)
        if NOT_RUN.search(head) or (TOOL_USE_ERROR in head and not content_refusal):
            cov["call_not_run"] += 1
            return []
    path = inp.get("file_path") or inp.get("notebook_path")
    p = sp.resolve(os.path.expanduser(path), cwd) if isinstance(path, str) and path else None
    if name == "Bash":
        cmd = inp.get("command") or ""
        if not isinstance(cmd, str) or not cmd:
            return []
        ops, _, info = sp.extract(shell_text(cmd, cwd), cwd, known, persist_cd=False)
        cov["unparsed"] += bool(info.get("unparsed"))
        m = CC_EXIT.match(res or "")
        rc = int(m.group(1)) if m else (None if res is None or (tur or {}).get("interrupted") else (1 if err else 0))
        call["rc"] = rc
        if ops:
            shell_outcome(ops, res or "", rc, cmd)
        return ops
    if p is None:
        return []
    if name == "Read":
        if err or res is None:
            return []
        f = (tur or {}).get("file") if isinstance(tur, dict) else None
        a, b, whole = None, None, False
        if isinstance(f, dict) and isinstance(f.get("startLine"), int) and isinstance(f.get("numLines"), int):
            a, n, total = f["startLine"], f["numLines"], f.get("totalLines")
            b = a + max(n, 1) - 1
            whole = a <= 1 and isinstance(total, int) and n >= total
        else:
            off, lim = inp.get("offset"), inp.get("limit")
            a = int(off) if isinstance(off, (int, float)) and off > 0 else 1
            whole = not off and not lim
            b = a + int(lim) - 1 if isinstance(lim, (int, float)) and lim > 0 else 10 ** 9
        view = {"range": [1, 10 ** 9] if whole else [a, b], "numbered": True}
        return [{"op": "R", "path": p, "kind": "full" if whole else "range", "mech": "editor-view", "view": view,
                 "landed": True, "cmd_i": 0, "verb": "Read"}]
    if name in ("Edit", "MultiEdit"):
        edits = [inp] if name == "Edit" else [e for e in inp.get("edits") or [] if isinstance(e, dict)]
        if isinstance(tur, dict) and tur.get("userModified"):
            # the user changed the proposed edit before accepting it: the result holds what was applied
            cov["user_modified"] += 1
            if name == "Edit" and isinstance(tur.get("oldString"), str):
                edits = [{"old_string": tur["oldString"], "new_string": tur.get("newString") or "",
                          "replace_all": tur.get("replaceAll", inp.get("replace_all"))}]
            elif isinstance(tur.get("edits"), list):
                edits = [e for e in tur["edits"] if isinstance(e, dict)] or edits
        if res is None:
            landed, ev = None, "no result on record"
        elif err:
            landed, ev = False, _cc_edit_ev(res)
        else:
            landed, ev = True, "editor confirmed"
        harness = None
        if landed is False and CC_STALE_REFUSED.search(res or ""):
            harness = "refused-stale"
        elif landed and CC_STALE_NOTED.search(res or ""):
            harness = "noted-stale"
        ops = [{"op": "W", "path": p, "kind": "anchored", "mech": "editor",
                "anchor": {"type": "str_replace", "old": e.get("old_string"), "new": e.get("new_string") or "",
                           "all": bool(e.get("replace_all"))},
                "landed": landed, "ev": ev, "cmd_i": 0, "verb": name, **({"harness": harness} if harness else {})}
               for e in edits]
        # Claude Code shows a snippet of the edited file (older versions) or states that the model's copy
        # is current (newer ones; when other changes were on disk it says so instead): an editor echo.
        # It refuses an edit to a file changed since it was read, so a plain success (the only answer of
        # versions before the "file state is current" suffix) also means the model's copy is current.
        if landed and harness is None and ("result of running `cat -n`" in (res or "")
                                           or "file state is current in your context" in (res or "")
                                           or CC_EDIT_OK.search(res or "")):
            ops.append({"op": "R", "path": p, "kind": "echo", "mech": "editor-echo", "view": {"numbered": True},
                        "landed": True, "cmd_i": 0, "verb": name})
        return ops
    if name == "Write":
        text = inp.get("content")
        if isinstance(tur, dict) and tur.get("userModified") and isinstance(tur.get("content"), str):
            cov["user_modified"] += 1
            text = tur["content"]
        if not isinstance(text, str):
            return []
        landed = None if res is None else not err
        if landed is not False:
            known[p] = text  # a script it wrote and then runs is analysed like a heredoc script
        return [{"op": "W", "path": p, "kind": "overwrite", "mech": "editor-create", "text": text, "landed": landed,
                 "ev": "editor confirmed" if landed else ("editor rejected" if landed is False else "no result on record"),
                 "cmd_i": 0, "verb": "Write"}]
    if name == "NotebookEdit":
        cov["unsupported:NotebookEdit"] += 1
        if err:
            return []
        return [{"op": "W", "path": p, "kind": "replace", "mech": "notebook-edit", "landed": None if res is None else True,
                 "cmd_i": 0, "verb": name}]
    return []


def _cc_read(path, cwd_arg, cov, prefix=""):
    """One pass over a Claude Code transcript: (calls and notices in order, session meta)."""
    items, by_id, bg = [], {}, {}
    meta = {"cwd": None, "models": Counter(), "session": None}
    own_side, conv, compacting = None, 0, False
    for d, bad in records(path):
        if d is None:
            cov[f"{prefix}skipped_line:{bad}"] += 1
            continue
        typ = d.get("type")
        meta["session"] = meta["session"] or d.get("sessionId")
        if meta["cwd"] is None and isinstance(d.get("cwd"), str):
            meta["cwd"] = d["cwd"]
        cwd = cwd_arg or d.get("cwd") or meta["cwd"] or "/"
        side = bool(d.get("isSidechain"))
        if typ == "assistant":
            if own_side is None:
                own_side = side
            m = d.get("message") if isinstance(d.get("message"), dict) else {}
            if m.get("model") and m["model"] != "<synthetic>":
                meta["models"][m["model"]] += 1
            unit = ("s" if side != own_side else "m", m.get("id") or d.get("uuid"))
            compacting = False
            content = m.get("content") if isinstance(m.get("content"), list) else []
            thought = " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
            for b in content:
                if not isinstance(b, dict) or b.get("type") != "tool_use":
                    continue
                c = {"kind": "call", "name": b.get("name"), "input": b.get("input"), "id": b.get("id"), "unit": unit,
                     "conv": conv, "cwd": cwd, "ts": d.get("timestamp"), "side": side != own_side, "thought": thought,
                     "result": None, "is_error": None, "tur": None, "ts_done": None, "extra": "",
                     "bg": b.get("name") == "Bash" and isinstance(b.get("input"), dict)
                     and bool(b["input"].get("run_in_background")), "bg_done": False}
                cov[f"{prefix}tool:{c['name']}"] += 1
                items.append(c)
                if c["id"]:
                    by_id[c["id"]] = c
        elif typ == "user":
            m = d.get("message") if isinstance(d.get("message"), dict) else {}
            if d.get("isCompactSummary") and not compacting:
                conv += 1
            content = m.get("content")
            if bg and "<task-notification>" in _text(content):
                # a background command finished: its writes land now, not when it was launched
                for tid in re.findall(r"<task-id>([\w-]+)</task-id>", _text(content)):
                    c = bg.pop(tid, None)
                    if c is not None:
                        c["bg_done"] = True
                        items.append({"kind": "bgdone", "call": c, "conv": conv, "ts": d.get("timestamp"), "cwd": cwd})
                        cov[f"{prefix}background_completed"] += 1
            results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"] \
                if isinstance(content, list) else []
            for b in results:
                c = by_id.get(b.get("tool_use_id"))
                if c is None:
                    cov[f"{prefix}result_without_call"] += 1
                    continue
                c["result"], c["is_error"], c["ts_done"] = _text(b.get("content")), bool(b.get("is_error")), d.get("timestamp")
                if c.get("bg") and not c["is_error"]:
                    m_bg = CC_BG_ID.search(c["result"] or "")
                    if m_bg:
                        bg[m_bg.group(1)] = c
                tur = d.get("toolUseResult")
                if len(results) == 1 and isinstance(tur, dict):
                    keep = ("file", "interrupted", "type")
                    if tur.get("userModified"):
                        keep += ("userModified", "oldString", "newString", "replaceAll", "edits", "content")
                    c["tur"] = {k: tur[k] for k in keep if k in tur}
        elif typ == "attachment":
            a = d.get("attachment") if isinstance(d.get("attachment"), dict) else {}
            at = a.get("type")
            if at in ("edited_text_file", "file") and isinstance(a.get("filename"), str):
                body = a.get("snippet") if at == "edited_text_file" else a.get("content")
                if isinstance(body, dict):
                    body = (body.get("file") or {}).get("content") if isinstance(body.get("file"), dict) else _text(body)
                items.append({"kind": "notice", "type": at, "path": a["filename"], "obs": body if isinstance(body, str) else "",
                              "conv": conv, "cwd": cwd, "ts": d.get("timestamp"), "side": side != own_side and own_side is not None})
                cov[f"{prefix}notice:{at}"] += 1
            elif at in ("hook_additional_context", "hook_success", "hook_system_message") and a.get("toolUseID") in by_id:
                extra = a.get("content") if at != "hook_success" else a.get("stdout")
                by_id[a["toolUseID"]]["extra"] += "\n" + _text(extra)
                cov[f"{prefix}hook_output"] += 1
            else:
                cov[f"{prefix}attachment:{at}"] += 1
        elif typ == "system" and d.get("subtype") == "compact_boundary":
            conv += 1
            compacting = True
            cov[f"{prefix}compactions"] += 1
        else:
            cov[f"{prefix}line:{typ}"] += 1
    return items, meta


def _cc_steps(items, cwd_arg, cov, prefix=""):
    """Steps in order; notices and sidechain steps wait for the end of the unit they arrive in."""
    steps, known, held, cur = [], {}, [], None
    for k, it in enumerate(items):
        if it["kind"] == "bgdone":
            c = it["call"]
            w = c.pop("bg_writes", None)
            if w:
                w = [dict(o, no_score=True, no_apply=False) for o in w]
                held.append(_step(unseen(w, "sidechain") if c["side"] else w, c["input"].get("command") or "", "", None,
                                  cwd_arg or c["cwd"], "sidechain" if c["side"] else "shell-bg", ("b", k), it["conv"], it["ts"]))
            continue
        if it["kind"] == "notice":
            p = sp.resolve(it["path"], cwd_arg or it["cwd"])
            ops = [] if it["side"] else [{"op": "R", "path": p, "kind": "full" if it["type"] == "file" else "lines",
                                          "mech": "cc-" + it["type"], "view": {"notice": it["type"]}, "landed": True,
                                          "cmd_i": 0, "verb": "notice"}]
            held.append(_step(ops, f"[{it['type']}] {it['path']}", it["obs"] or "", None, it["cwd"], "notice",
                              ("n", k), it["conv"], it["ts"]))
            continue
        if it["unit"] != cur:
            steps += held
            held, cur = [], it["unit"]
        cwd = cwd_arg or it["cwd"]
        ops = _cc_ops(it, cwd, known, cov)
        name, inp = it["name"], it["input"]
        if it.get("bg") and ops:
            # a background command's output goes to a file, not to the model: no views. Its writes land when
            # it completes; without a completion on record they stay at launch, marked as of unknown landing
            writes = [o for o in ops if o["op"] == "W"]
            if it.get("bg_done"):
                # scored here, where the agent composed them; applied at the completion
                it["bg_writes"], ops = writes, [dict(o, no_apply=True) for o in writes]
            else:
                ops = [dict(o, landed=None, ev="background command; completion not on record", bg=True) for o in writes]
        action = inp.get("command") if name == "Bash" and isinstance(inp, dict) else \
            f"{name} {json.dumps(inp, ensure_ascii=False)[:2000]}"
        obs = (it["result"] or "") + it["extra"]
        if it["side"]:
            cov[f"{prefix}sidechain_calls"] += 1
            held.append(_step(unseen(ops, "sidechain"), action or "", "", it.get("rc"), cwd, "sidechain",
                              it["unit"], it["conv"], it["ts_done"] or it["ts"]))
            continue
        steps.append(_step(ops, action or "", obs, it.get("rc"), cwd, name if name != "Bash" else "shell",
                           it["unit"], it["conv"], it["ts"], it["thought"]))
        steps[-1]["ts_done"] = it["ts_done"]
    steps += held
    _drop_unlanded_background(steps, cov, prefix)
    return steps


def _drop_unlanded_background(steps, cov, prefix=""):
    """A background write with no completion on record had not landed when a later edit of the same file
    came back with the model's copy current (Claude Code refuses or notes an edit to a file changed since
    it was read): it is still scored where it was composed, but not applied at launch."""
    for i, s in enumerate(steps):
        for o in [o for o in s["ops"] if o.get("bg")]:
            for s2 in steps[i + 1:]:
                if any(e["op"] == "R" and e.get("kind") == "echo" and e["path"] == o["path"] for e in s2["ops"]):
                    o["no_apply"] = True
                    cov[f"{prefix}background_write_not_landed"] += 1
                    break


def session_file(path):
    """The main transcript of the session a Claude Code transcript belongs to: itself, or for
    <session>/subagents/[...]/agent-x.jsonl, <session>.jsonl."""
    parts = os.path.normpath(path).split(os.sep)
    if "subagents" in parts[:-1]:
        i = len(parts) - 1 - parts[::-1].index("subagents")
        return os.sep.join(parts[:i]) + ".jsonl"
    return path


_WRITE_CACHE = {}


def _write_steps(path, cwd, who, tag):
    """Another transcript's own writes, as steps for merge(); cached per process (one session's
    transcripts are read once each however many of its subagents are analyzed)."""
    try:
        key = (path, cwd, os.path.getmtime(path))
    except OSError:
        return []
    if key not in _WRITE_CACHE:
        if len(_WRITE_CACHE) > 64:
            _WRITE_CACHE.clear()
        sitems, _ = _cc_read(path, cwd, Counter())
        _WRITE_CACHE[key] = [s for s in _cc_steps(sitems, cwd, Counter()) if s["tool"] not in ("notice", "sidechain")]
    out = []
    for s in _WRITE_CACHE[key]:
        w = unseen(s["ops"], who)
        if w:
            out.append(_step(w, s["action"], "", s["rc"], s["cwd"], who, ("a", tag, s["unit"]), 0, s.get("ts_done") or s["ts"]))
    return out


def adapt_claudecode(path, cwd=None):
    """(steps, meta, coverage) for one Claude Code transcript and the transcripts that share its
    workspace: a session's subagents, or for a subagent, its session and sibling subagents."""
    cov = Counter()
    items, meta = _cc_read(path, cwd, cov)
    steps = _cc_steps(items, cwd, cov)
    main = session_file(path)
    subdir = os.path.join(os.path.splitext(main)[0], "subagents")
    others = []
    if main != path and os.path.isfile(main):
        others += _write_steps(main, cwd, "parent", "parent")
        cov["parent_write_steps"] += len(others)
    if os.path.isdir(subdir):
        for root, _, names in sorted(os.walk(subdir)):
            for n in sorted(names):
                f = os.path.join(root, n)
                if not n.endswith(".jsonl") or os.path.abspath(f) == os.path.abspath(path):
                    continue
                w = _write_steps(f, cwd, "subagent" if main == path else "peer", n)
                others += w
                cov["subagent_transcripts" if main == path else "sibling_transcripts"] += 1
                cov["subagent_write_steps" if main == path else "sibling_write_steps"] += len(w)
    if others:
        steps = merge(steps, others)
    meta["model"] = meta["models"].most_common(1)[0][0] if meta["models"] else ""
    meta["models"] = dict(meta["models"])
    return renumber(steps), meta, cov


# ------------------------------------------------------------------ Codex CLI

SHELLS = {"bash", "sh", "zsh", "dash"}
CX_RC = re.compile(r"^(?:Exit code: |Process exited with code )(-?\d+)", re.M)


def _cmd_text(command):
    """The shell script of a Codex command (["bash", "-lc", s], a plain string, or an argv)."""
    if isinstance(command, str):
        return command
    if isinstance(command, list) and all(isinstance(x, str) for x in command) and command:
        if len(command) >= 3 and posixpath.basename(command[0]) in SHELLS and command[1] in ("-lc", "-c", "-l", "-ic"):
            return command[-1]
        if command[0] == "apply_patch" and len(command) == 2:
            return "apply_patch " + shlex.quote(command[1])
        return shlex.join(command)
    return ""


def _cx_output(o):
    """(text, return code) of a Codex tool output in any of its encodings."""
    if isinstance(o, list):
        o = "\n".join(_text(x) for x in o)
    if isinstance(o, dict):
        o = json.dumps(o)
    if not isinstance(o, str):
        return "", None
    rc = None
    try:
        d = json.loads(o)
    except ValueError:
        d = None
    if isinstance(d, dict) and "output" in d:
        md = d.get("metadata") if isinstance(d.get("metadata"), dict) else {}
        rc = md.get("exit_code")
        o = d["output"] if isinstance(d["output"], str) else _text(d["output"])
    m = CX_RC.search(o[:400])
    if m:
        rc = int(m.group(1))
    if re.match(r"^(?:Exit code|Chunk ID|Wall time|Process |Script completed)", o) and "\nOutput:\n" in o[:600]:
        o = o.split("\nOutput:\n", 1)[1]
    return o, rc


JS_LIT = r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`(?:\\.|[^`\\])*`'


def _js_str(tok, src=""):
    """The value of a JS string token (double, single or backtick quoted, or a name bound to one
    earlier in `src`); None when it is computed (a template with ${...}, a call, a concatenation)."""
    tok = tok.strip()
    if re.fullmatch(r"[A-Za-z_$][\w$]*", tok):
        m = None
        for m in re.finditer(r"(?:const|let|var)\s+" + re.escape(tok) + r"\s*=\s*(" + JS_LIT + r")\s*;?", src, re.S):
            pass
        return _js_str(m.group(1)) if m else None
    if len(tok) < 2 or tok[0] != tok[-1] or tok[0] not in "\"'`":
        return None
    q, body = tok[0], tok[1:-1]
    if q == "`" and "${" in body:
        return None
    if q == '"':
        try:
            return json.loads(tok)
        except ValueError:
            return None
    out, i = [], 0
    esc = {"n": "\n", "t": "\t", "r": "\r", "0": "\0"}
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            out.append(esc.get(body[i + 1], body[i + 1]))
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _exec_commands(src):
    """(commands, workdir or None, parsed) of a code-mode `exec` script."""
    cmds, parsed = [], True
    lit = r'"(?:\\.|[^"\\])*"'
    for m in re.finditer(r"tools\.exec_command\(\s*\{(.*?)\}\s*\)", src, re.S):
        body = m.group(1)
        k = re.search(r'(?:^|[,{\s])["\']?cmd["\']?\s*:\s*(' + lit + ")", body)
        wd = re.search(r'["\']?workdir["\']?\s*:\s*(' + lit + ")", body)
        wd = json.loads(wd.group(1)) if wd else None
        if k:
            cmds.append((json.loads(k.group(1)), wd))
            continue
        short = re.match(r"\s*cmd\s*(?:,|$)", body)
        arr = re.search(r"(\w+)\.map\(\s*\(?\s*cmd\s*\)?\s*=>\s*tools\.exec_command\(\s*\{" + re.escape(body), src)
        if short and arr:
            dm = re.search(r"(?:const|let|var)\s+" + re.escape(arr.group(1)) + r"\s*=\s*\[(.*?)\]\s*;", src, re.S)
            if dm and "`" not in dm.group(1):
                cmds += [(json.loads(x), wd) for x in re.findall(lit, dm.group(1))]
                continue
        parsed = False
    for m in re.finditer(r"tools\.apply_patch\(\s*(" + JS_LIT + r"|[A-Za-z_$][\w$]*)\s*\)", src, re.S):
        body = _js_str(m.group(1), src[:m.start()])
        if body is None:
            parsed = False
            continue
        cmds.append(("apply_patch " + shlex.quote(body), None))
    if not cmds and "tools." in src:
        parsed = False
    return cmds, parsed


BSD_SED_ERR = re.compile(r'^sed: (?:\d+: "|-[ei]: |[^:\n]+: (?:No such file|in-place editing only))', re.M)


def bsd_sed_outcome(ops, obs):
    """BSD/macOS sed reports errors as `sed: 1: "script": ...`, which the audit's (GNU) rules do not
    read: with one sed write in the command, that write did not land."""
    if not BSD_SED_ERR.search(obs or ""):
        return
    sed_w = [o for o in ops if o.get("mech") == "sed" and o["op"] == "W" and o.get("ran")]
    if len({o["cmd_i"] for o in sed_w}) == 1:
        for o in sed_w:
            if o.get("landed") is True:
                o["landed"], o["ev"] = False, "sed error (BSD, single sed unit)"


CUT_MARK = re.compile(r"…\d+ tokens truncated…")


def split_exec_output(src, cmds, out):
    """Per-command (output, exit code) of a code-mode script that ran `cmds` through
    `ARR.map(cmd => tools.exec_command(...))` and printed `text(OUT.map((r, i) => `...${r.output}`)
    .join(SEP))`, found by locating each command's rendered header in order. Codex may prepend a
    note or cut the middle of a long output; a command whose header is missing gets (None, None).
    None when the script prints its results any other way."""
    m = re.search(r"text\(\s*(\w+)\.map\(\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*=>\s*`((?:\\.|[^`\\])*)`\s*\)\s*"
                  r"\.join\(\s*(" + JS_LIT + r")\s*\)\s*\)", src, re.S)
    if not m or not out:
        return None
    r_, i_, tpl, sep = m.group(2), m.group(3), m.group(4), _js_str(m.group(5))
    arr = re.search(r"(?:const|let|var)\s+(\w+)\s*=\s*\[", src)
    pieces = re.split(r"(\$\{[^}]*\})", tpl)
    outs = [k for k, pc in enumerate(pieces) if pc.replace(" ", "") == "${%s.output}" % r_]
    if sep is None or len(outs) != 1 or any(pc.strip() for pc in pieces[outs[0] + 1:]):
        return None  # only templates that end with the output can be split by their headers

    def head(k):
        rx = ""
        for pc in pieces[:outs[0]]:
            if not pc.startswith("${"):
                rx += re.escape(_js_str("`" + pc + "`") or "")
                continue
            e = pc[2:-1].replace(" ", "")
            im = re.fullmatch(re.escape(i_) + r"(?:\+(\d+))?", e)
            if im:
                rx += str(k + int(im.group(1) or 0))
            elif e in (f"{r_}.exit_code", f"{r_}.exitCode"):
                rx += r"(?P<rc>-?\d+|null|undefined)"
            elif arr and e == f"{arr.group(1)}[{i_}]":
                rx += re.escape(cmds[k])
            else:
                return None
        return rx
    heads = [head(k) for k in range(len(cmds))]
    if any(h is None or not h for h in heads):
        return None
    found, pos = [], 0
    for k, h in enumerate(heads):
        hm = re.compile(r"(?:^|(?<=\n))" + h, re.M).search(out, pos)
        found.append(hm)
        if hm:
            pos = hm.end()
    if not any(found):
        return None
    # Codex shortens a long output by cutting its middle at one marker ("…N tokens truncated…"):
    # the command before the cut keeps its head, and the last command whose header the cut removed
    # keeps the tail that runs up to the next header.
    cut = CUT_MARK.search(out)
    res = []
    for k, hm in enumerate(found):
        if not hm:
            res.append((None, None))
            continue
        nxt = next((f for f in found[k + 1:] if f), None)
        end = nxt.start() if nxt else len(out)
        body = out[hm.end():end]
        if k + 1 < len(found) and found[k + 1] is None:
            # the headers after this one were cut: keep the head up to the marker, if it is here
            body = out[hm.end():cut.start()] if cut and hm.end() <= cut.start() < end else None
        elif nxt and body.endswith(sep):
            body = body[:-len(sep)]
        rc = hm.groupdict().get("rc")
        res.append((body, int(rc) if rc and rc.lstrip("-").isdigit() else None))
    if cut:
        nxt_k = next((k for k, f in enumerate(found) if f and f.start() >= cut.end()), None)
        last = (nxt_k if nxt_k is not None else len(found)) - 1
        if last >= 0 and found[last] is None:
            end = found[nxt_k].start() if nxt_k is not None else len(out)
            tail = out[cut.end():end]
            res[last] = (tail[:-len(sep)] if nxt_k is not None and tail.endswith(sep) else tail, None)
    return res


def adapt_codex(path, cwd=None):
    """(steps, meta, coverage) for one Codex CLI rollout."""
    cov = Counter()
    meta = {"cwd": None, "models": Counter(), "session": None}
    calls, by_id, events = [], {}, {}
    cm_patches = []  # code-mode apply_patch calls, matched in order to their patch_apply_end events
    conv, unit, after_output = 0, 0, True
    turn_cwd = cwd
    for d, bad in records(path):
        if d is None:
            cov[f"skipped_line:{bad}"] += 1
            continue
        typ = d.get("type")
        if "payload" in d and isinstance(d["payload"], dict):
            p = d["payload"]
        elif typ in RESPONSE_TYPES:
            typ, p = "response_item", d
        elif "record_type" in d:
            cov["line:state"] += 1
            continue
        else:
            if "instructions" in d and "id" in d:
                meta["session"] = meta["session"] or d.get("id")
            cov[f"line:{typ}"] += 1
            continue
        if typ == "session_meta":
            meta["session"] = meta["session"] or p.get("id")
            if isinstance(p.get("cwd"), str):
                meta["cwd"] = meta["cwd"] or p["cwd"]
                turn_cwd = cwd or p["cwd"]
            continue
        if typ == "turn_context":
            if isinstance(p.get("cwd"), str):
                meta["cwd"] = meta["cwd"] or p["cwd"]
                turn_cwd = cwd or p["cwd"]
            if p.get("model"):
                meta["models"][p["model"]] += 1
            continue
        if typ == "compacted":
            conv += 1
            cov["compactions"] += 1
            continue
        if typ == "event_msg":
            et = p.get("type")
            if et == "exec_command_end" and p.get("call_id"):
                events[p["call_id"]] = {"rc": p.get("exit_code")}
            elif et == "patch_apply_end" and p.get("call_id"):
                if p["call_id"] not in by_id and cm_patches:
                    cm_patches.pop(0)["success"] = p.get("success")
                else:
                    events[p["call_id"]] = {"success": p.get("success")}
            cov[f"event:{et}"] += 1
            continue
        if typ != "response_item":
            cov[f"line:{typ}"] += 1
            continue
        pt = p.get("type")
        if pt in ("function_call", "custom_tool_call", "local_shell_call"):
            if after_output:
                unit += 1
                after_output = False
            name = p.get("name") or pt
            args = p.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"input": args}
            args = args if isinstance(args, dict) else {}
            wd = args.get("workdir") or args.get("working_directory")
            script, patch = None, None
            if pt == "local_shell_call":
                act = p.get("action") if isinstance(p.get("action"), dict) else {}
                script, wd = _cmd_text(act.get("command")), act.get("working_directory") or wd
            elif name in ("shell", "container.exec", "shell_command", "local_shell"):
                script = _cmd_text(args.get("command"))
            elif name == "exec_command":
                script = _cmd_text(args.get("cmd") or args.get("command"))
            elif name == "apply_patch":
                patch = p.get("input") if pt == "custom_tool_call" else (args.get("input") or args.get("patch"))
            elif name == "exec" and pt == "custom_tool_call" and isinstance(p.get("input"), str):
                found, parsed = _exec_commands(p["input"])
                cov["exec_scripts"] += 1
                cov["exec_scripts_unparsed"] += not parsed
                c0 = None
                for k, (cmd, cwd_k) in enumerate(found):
                    c = {"name": "exec_command", "script": cmd, "patch": None, "wd": cwd_k, "cwd": turn_cwd,
                         "unit": unit, "conv": conv, "id": p.get("call_id") if k == 0 else None, "out": None,
                         "rc": None, "action": cmd, "ts": d.get("timestamp")}
                    calls.append(c)
                    c0 = c0 or c
                    if cmd.startswith("apply_patch "):
                        cm_patches.append(c)
                cov["tool:exec"] += 1
                if c0 is not None and p.get("call_id"):
                    by_id[p["call_id"]] = c0
                    group = [c for c in calls[len(calls) - len(found):] if not c["script"].startswith("apply_patch ")]
                    if len(group) > 1 and len(group) == len(found):
                        c0["group"], c0["src"] = group, p["input"]
                continue
            cov[f"tool:{name}"] += 1
            if script is None and patch is None:
                continue
            c = {"name": name, "script": script, "patch": patch, "wd": wd, "cwd": turn_cwd, "unit": unit, "conv": conv,
                 "id": p.get("call_id"), "out": None, "rc": None, "ts": d.get("timestamp"),
                 "action": script if script is not None else "apply_patch\n" + (patch or "")}
            calls.append(c)
            if c["id"]:
                by_id[c["id"]] = c
        elif pt in ("function_call_output", "custom_tool_call_output"):
            after_output = True
            c = by_id.get(p.get("call_id"))
            if c is None:
                cov["output_without_call"] += 1
                continue
            c["out"], c["rc"] = _cx_output(p.get("output"))
            if c.get("group"):
                split = split_exec_output(c["src"], [g["script"] for g in c["group"]], c["out"])
                cov["exec_outputs_split" if split else "exec_outputs_unsplit"] += 1
                if split:
                    cov["exec_outputs_cut"] += bool(CUT_MARK.search(c["out"]))
                    cov["exec_commands_without_output"] += sum(o is None for o, _ in split)
                    cov["exec_commands_split"] += len(split)
                    for g, (o, rc) in zip(c["group"], split):
                        g["out"], g["rc"] = (o or "", rc) if o is not None else ("", None)
                        g["cut"] = o is None
        elif pt == "message":
            if p.get("role") == "user":
                after_output = True
            cov[f"message:{p.get('role')}"] += 1
        else:
            cov[f"item:{pt}"] += 1
    steps, known = [], {}
    for c in calls:
        cwd_c = sp.resolve(c["wd"], c["cwd"] or "/") if isinstance(c["wd"], str) and c["wd"] else (c["cwd"] or "/")
        ev = events.get(c["id"]) or {}
        obs = c["out"] or ""
        rc = ev.get("rc", c["rc"])
        if rc is None and c["script"] and obs:
            # code-mode output carries no exit code; a tool's own error line ("sed: 1: ...") marks a failure
            prog = posixpath.basename((c["script"].split() or [""])[0])
            if prog and re.search(r"^" + re.escape(prog) + r": ", obs, re.M):
                rc = 1
                cov["rc_from_error_line"] += 1
        body, pcwd = (c["patch"], cwd_c) if c["patch"] is not None else shell_patch(c["script"], cwd_c)
        if body is not None:
            ops = patch_ops(body, pcwd)
            patch_outcome(ops, obs, ev.get("success", c.get("success")))
            tool = "apply_patch"
            cov["apply_patch"] += 1
        else:
            ops, _, info = sp.extract(shell_text(c["script"], cwd_c), cwd_c, known, persist_cd=False)
            cov["unparsed"] += bool(info.get("unparsed"))
            if ops:
                shell_outcome(ops, obs, rc, c["script"])
                bsd_sed_outcome(ops, obs)
            tool = "shell"
        steps.append(_step(ops, c["action"], obs, rc, cwd_c, tool, (c["conv"], c["unit"]), c["conv"], c["ts"]))
    meta["model"] = meta["models"].most_common(1)[0][0] if meta["models"] else ""
    meta["models"] = dict(meta["models"])
    return renumber(steps), meta, cov


WIN_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


def _win_path(p):
    """A Windows path with `/` separators: C:\\a\\b.py, C:/a/b.py and Git Bash's /c/a/b.py alike."""
    m = re.match(r"^/([A-Za-z])(/|$)", p)
    if m:
        p = m.group(1).upper() + ":/" + p[3:]
    return ntpath.normpath(p).replace("\\", "/")


def root_of(steps, root):
    """Strip the session's working directory from paths (extract.canonicalize, with the root the log gives).
    A session run on Windows logs backslashed, drive-lettered paths in any case: they are compared as such."""
    if not root:
        return None
    win = bool(WIN_PATH.match(root))
    root = _win_path(root) if win else posixpath.normpath(root)
    key = (lambda x: x.lower()) if win else (lambda x: x)

    def rel(p):
        if not p:
            return p
        p = _win_path(p) if win else p
        if key(p) == key(root):
            return "."
        return p[len(root) + 1:] if key(p).startswith(key(root) + "/") else p

    for s in steps:
        for o in s["ops"]:
            o["abs"] = o["path"]
            o["path"] = rel(o["path"])
            if o.get("src"):
                o["src"] = rel(o["src"])
    return root
