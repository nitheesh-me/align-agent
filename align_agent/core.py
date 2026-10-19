"""Load a trajectory in any supported format, replay it, and classify its stale edits.

The replay and the classifier are the RQ1 audit's (scripts/audit/): the
adapters turn each format into steps of file operations (views, anchored
edits, whole-file writes) parsed from the commands and tool calls, the replay
binds every view to the file version it showed, and an anchored edit to a file
whose last view is outdated is a stale edit. The classifier then asks whether
the edit went wrong because of it.
"""

import json
import os
import sys
from collections import Counter


def _audit_dir():
    """The repository's scorer when run from a checkout, else the copy build.sh vendored."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", "scripts", "audit"))
    return repo if os.path.isdir(repo) else os.path.join(here, "_audit")


sys.path.insert(0, _audit_dir())

import deepswe as _atif  # noqa: E402
from . import sessions as _sess  # noqa: E402
import extract as _ex  # noqa: E402
import tarsur as _tc  # noqa: E402
from materialize import classify  # noqa: E402
from score import Replay, cp_ci  # noqa: E402

FORMATS = {
    "mini": "mini-SWE-agent (text actions, <returncode> observations)",
    "sweagent": "SWE-agent .traj",
    "openhands": "OpenHands completion log or event stream",
    "terminus": "Terminus 2 episodes (prompt/response)",
    "atif": "Harbor ATIF (e.g. DeepSWE leaderboard trials)",
    "toolcalls": "OpenAI-style tool-call messages with a bash tool",
    "claudecode": "Claude Code session logs (~/.claude/projects/*/*.jsonl)",
    "codex": "Codex CLI session rollouts (~/.codex/sessions)",
}
AGENT = {"mini": "mini-SWE-agent", "sweagent": "SWE-agent", "openhands": "OpenHands", "terminus": "Terminus2",
         "atif": "mini-SWE-agent (ATIF)", "toolcalls": "tool calls", "claudecode": "Claude Code", "codex": "Codex"}
# Session logs record their own working directory; "/" is used only when one does not.
DEFAULT_CWD = {"mini": "/app", "atif": "/testbed", "toolcalls": "/testbed", "claudecode": "/", "codex": "/"}
SESSIONS = {"claudecode": _sess.adapt_claudecode, "codex": _sess.adapt_codex}


def detect(raw):
    """The format of a parsed trajectory, or None."""
    if isinstance(raw, dict):
        steps = raw.get("steps")
        if isinstance(steps, list) and (str(raw.get("schema_version", "")).upper().startswith("ATIF")
                                        or any(isinstance(s, dict) and "source" in s for s in steps[:20])):
            return "atif"
        if isinstance(raw.get("trajectory"), list):
            return "sweagent"
        msgs = raw.get("messages")
        if isinstance(msgs, list):
            if any(isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls") for m in msgs):
                return "toolcalls"
            return "mini"
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        k = set(raw[0])
        if {"prompt", "response"} <= k and "messages" not in k:
            return "terminus"
        return "openhands"
    return None


def load(path, fmt=None, cwd=None):
    """(trajectory record for the replay, format, adapter coverage) for one file.

    A JSON Lines file (a Claude Code or Codex session) is detected from the
    shapes of its first lines; any other file is one JSON document."""
    if fmt in SESSIONS or (fmt is None and path.endswith(".jsonl")):
        return _load_session(path, fmt or _sess.sniff(path), cwd)
    with open(path, encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError:
            if fmt is None and _sess.sniff(path):
                return _load_session(path, _sess.sniff(path), cwd)
            raise
    fmt = fmt or detect(raw)
    if fmt not in FORMATS:
        raise ValueError(f"{path}: unrecognised trajectory format")
    tid = os.path.splitext(os.path.basename(path))[0]
    cwd = cwd or (raw.get("container_workdir") if isinstance(raw, dict) else None) or DEFAULT_CWD.get(fmt)
    item = {"traj_id": tid, "agent": AGENT[fmt], "model": "", "step_count": -1, "cwd": cwd}
    meta, cov = {}, {}
    if fmt == "mini":
        steps, meta = _ex.extract_mini(raw, item)
    elif fmt == "sweagent":
        steps, meta = _ex.extract_swe(raw, item)
    elif fmt == "terminus":
        steps, meta = _ex.extract_t2(raw, item)
    elif fmt == "openhands":
        if any(isinstance(x, dict) and "response" in x for x in raw):
            steps, meta = _ex.extract_oh_log(raw, item)
        else:
            steps, meta = _ex.extract_oh_events(raw, item)
    elif fmt == "atif":
        steps, cov = _atif.adapt(raw, cwd=cwd)
    else:
        steps, cov = _tc.adapt(raw, cwd=cwd)
    agent = {"sweagent": "SWE-agent", "openhands": "OpenHands", "terminus": "Terminus2"}.get(fmt, "mini-SWE-agent")
    t = {"traj_id": tid, "agent": agent, "model": "", "steps": steps, "final_diff": meta.get("final_diff")}
    t["root"] = _ex.canonicalize(steps)
    return t, fmt, dict(cov)


def _load_session(path, fmt, cwd):
    if fmt not in SESSIONS:
        raise ValueError(f"{path}: unrecognised trajectory format")
    steps, meta, cov = SESSIONS[fmt](path, cwd=cwd)
    tid = os.path.splitext(os.path.basename(path))[0]
    t = {"traj_id": tid, "agent": AGENT[fmt], "model": meta.get("model") or "", "steps": steps, "final_diff": None,
         "session": meta.get("session")}
    t["root"] = _sess.root_of(steps, cwd or meta.get("cwd"))
    return t, fmt, dict(cov)


OTHER_WRITERS = _sess.OTHER_WRITERS


def writer_of(op):
    """'self' for the agent's own write, 'other' for one another agent (parent, subagent, sibling) made."""
    return "other" if str(op.get("mech") or "").startswith(OTHER_WRITERS) else "self"


def analyze(t):
    """Opportunities, stale edits and their classes for one loaded trajectory."""
    r = Replay(t).run()
    steps = t["steps"]
    stale = []
    hazard_at = {(h["step"], h["op"]) for h in r.hazards}
    # Claude Code's own verdicts on edits the replay scored: stale (refused, or applied with a note)
    oracle = Counter()
    for o in r.opps:
        flag = steps[o["step"]]["ops"][o["op"]].get("harness")
        if flag:
            oracle[flag] += 1
            oracle[flag + (":flagged" if (o["step"], o["op"]) in hazard_at else ":missed")] += 1
    for h in r.hazards:
        cls, why, _ = classify(h, t)
        lv = h.get("last_view")
        e = steps[h["step"]]
        wops = [steps[w[0]]["ops"][w[1]] for w in h.get("W") or [] if 0 <= w[1] < len(steps[w[0]]["ops"])]
        writers = sorted({writer_of(o) for o in wops}) or ["self"]
        stale.append({
            "step": h["step"], "path": h["path"], "mechanism": h.get("mech"),
            "held_view_step": lv[0] if lv else None,
            "view_channel": ("shell" if (steps[lv[0]].get("tool") or "shell") == "shell" else "tool") if lv else None,
            "unseen_writes": [w[0] for w in h.get("W") or []],
            # whose writes made the view stale: only the agent's own (it did not re-read after them), or
            # at least one write by another agent
            "writer": "other" if "other" in writers else "self",
            "class": cls, "reason": why,
            "went_wrong": cls.startswith("MATERIALIZED"),
            "command": (e.get("action") or "")[:400],
        })
    return {"trajectory": t["traj_id"], "agent": t["agent"], "model": t.get("model") or "", "steps": len(steps),
            "session": t.get("session"), "harness_oracle": dict(oracle),
            "anchored_edits": sum(1 for s in steps for o in s["ops"] if o["op"] == "W" and o.get("kind") == "anchored"
                                  and not o.get("no_score")),
            "opportunities": len(r.opps), "stale_edits": len(stale), "stale": stale}


def session_key(r):
    """The session a result belongs to (a session's subagent transcripts share it)."""
    f = r.get("file")
    if f and r.get("format") == "claudecode":
        return os.path.abspath(_sess.session_file(f))
    return f or r.get("session") or r["trajectory"]  # a Codex rollout is one session


MIN_CLUSTERS = 10  # fewer sessions than this: too few to resample, the binomial interval is given instead


def cluster_ci(results, a=0.05, reps=2000, seed=0):
    """Percentile bootstrap of the pooled stale rate, resampling whole sessions: the edits of one
    session are not independent trials. Deterministic (fixed seed), standard library only. With fewer
    than MIN_CLUSTERS sessions, the Clopper-Pearson interval of the pooled counts."""
    import random
    groups = {}
    for r in results:
        s, o = groups.get(session_key(r), (0, 0))
        groups[session_key(r)] = (s + r["stale_edits"], o + r["opportunities"])
    cells = [v for v in groups.values() if v[1]]
    if not cells:
        return (None, None)
    if len(cells) < MIN_CLUSTERS:
        return cp_ci(sum(s for s, _ in cells), sum(o for _, o in cells), a)
    rng, n, rates = random.Random(seed), len(cells), []
    for _ in range(reps):
        k = o = 0
        for _ in range(n):
            s_, o_ = cells[rng.randrange(n)]
            k += s_
            o += o_
        rates.append(k / o)
    rates.sort()
    return (round(rates[int(a / 2 * reps)], 4), round(rates[min(reps - 1, int((1 - a / 2) * reps))], 4))


def summarize(results):
    opps = sum(r["opportunities"] for r in results)
    stale = sum(r["stale_edits"] for r in results)
    classes = Counter(s["class"] for r in results for s in r["stale"])
    lo, hi = cp_ci(stale, opps) if opps else (None, None)
    clo, chi = cluster_ci(results) if opps else (None, None)
    cov = Counter()
    oracle = Counter()
    for r in results:
        cov.update(r.get("coverage") or {})
        oracle.update(r.get("harness_oracle") or {})
    writers = Counter(s.get("writer", "self") for r in results for s in r["stale"])
    sessions = len({session_key(r) for r in results})
    return {"trajectories": len(results), "sessions": sessions, "opportunities": opps, "stale_edits": stale,
            "stale_rate": round(stale / opps, 4) if opps else None, "ci95": [clo, chi], "ci95_binomial": [lo, hi],
            "ci95_method": "bootstrap over sessions" if sessions >= MIN_CLUSTERS else "binomial (too few sessions to resample)",
            "stale_after_own_writes": writers["self"], "stale_after_other_writes": writers["other"],
            "stale_rate_other_writes": round(writers["other"] / opps, 4) if opps else None,
            "caught_by_harness": classes.get("CAUGHT-BY-HARNESS", 0),
            "harness_oracle": dict(oracle),
            "went_wrong": sum(v for k, v in classes.items() if k.startswith("MATERIALIZED")),
            "classes": dict(classes.most_common()),
            "view_channels": dict(Counter(s["view_channel"] for r in results for s in r["stale"])),
            "coverage": dict(cov)}
