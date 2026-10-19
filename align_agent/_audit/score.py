"""Independent stale-edit scorer (version-level and region-level). Usage: python3 scripts/audit/score.py

Reads docs/audit/rq1/work/extracted.jsonl (from extract.py) and replays each
trajectory. Definitions (the RQ1 structural definition, re-implemented):

  content id   every landed write gives the file a new content id; a cp/mv
               gives the destination the source's id; a git restore returns
               the base id 0; an editor undo returns the id before the last
               editor write.
  view         content reaching the agent: a read op, an editor echo, or a
               file quoted in the output (grep -rn / compiler `path:N:` lines,
               traceback frames, diff headers). A directory listing is not a
               view. Views count from the next composition unit on.
  opportunity  the first anchored edit to file f in a composition unit, when
               the agent had seen some content of f before that unit.
  hazard       an opportunity where f's content id at the edit is not among
               the ids the agent had been shown (or had authored) before it.

Variants: without editor echoes; without output-derived views; landed=None
treated as failed; per-step (not per-unit) composition; and a region-level
variant in which a view refreshes only the lines it showed.
Writes docs/audit/rq1/work/score_summary.json and hazards.jsonl.
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import WORK, short_model  # noqa: E402

OUT_TRACE = re.compile(r'File "([^"<>\n]+)", line (\d+)')
OUT_LOC = re.compile(r"(?m)^\s*(?:-->\s+)?((?:/|\./|[\w.\-]+/)?(?:[\w.\-]+/)*[\w.\-]+\.[A-Za-z0-9]{1,8}):(\d+)[:\s-]")
OUT_DIFF = re.compile(r"(?m)^(?:diff --git a/(\S+) b/|\+\+\+ b/(\S+))")
# Blocks the write-through hook (src/harness) appends to a command's output.
# Each shows the file's current state for its region, so it is a view.
OUT_HOOK = re.compile(r"(?m)^\[(?:TAIL DIFF: File '([^'\n]+)'|VIEWPORT \(lines \d+-\d+ of \d+\): ([^\]\n]+))")


def canon_path(p, cwd, root, known=None):
    """Canonical spelling of a path quoted in tool output.

    Compilers, test runners and diff headers quote paths relative to the
    project root (cargo, pytest, git), not to the shell's cwd; tracebacks
    quote absolute paths. Candidates are tried in order and the first one
    naming a file the trajectory touches elsewhere wins: root-relative,
    cwd-relative, then a unique known path ending with the quoted path."""
    import posixpath
    if not p:
        return None
    p = p.strip()
    if p.startswith("/proc/self/cwd/"):
        p = posixpath.join(cwd or "/", p[len("/proc/self/cwd/"):])
    if p.startswith(("a/", "b/")) and not p.startswith("/"):
        p = p[2:]
    if p.startswith("./"):
        p = p[2:]

    def strip(ap):
        if root and (ap == root or ap.startswith(root + "/")):
            return ap[len(root) + 1:]
        return ap

    if p.startswith("/"):
        cands = [strip(posixpath.normpath(p))]
    else:
        cands = [posixpath.normpath(p),
                 strip(posixpath.normpath(posixpath.join(cwd or "/", p)))]
    if known:
        for c in cands:
            if c in known:
                return c
        tail = [k for k in known if k.endswith("/" + cands[0])]
        if len(tail) == 1:
            return tail[0]
    return cands[-1] if p.startswith("/") else cands[1]


def output_views(step, root, known=None):
    """Files quoted in the step's output, with the line numbers quoted."""
    obs = step.get("obs") or ""
    if len(obs) > 400000:
        obs = obs[:400000]
    found = defaultdict(set)
    for m in OUT_TRACE.finditer(obs):
        found[m.group(1)].add(int(m.group(2)))
    for m in OUT_LOC.finditer(obs):
        found[m.group(1)].add(int(m.group(2)))
    for m in OUT_DIFF.finditer(obs):
        found[m.group(1) or m.group(2)].add(0)
    for m in OUT_HOOK.finditer(obs):
        found[m.group(1) or m.group(2)].add(0)
    out = {}
    for p, lines in found.items():
        if p.startswith(("/usr/", "/opt/", "/root/.", "<")) or "site-packages" in p or "/lib/python" in p:
            continue
        c = canon_path(p, step.get("cwd"), root, known)
        if c:
            out[c] = lines
    return out


def known_paths(traj):
    return {o["path"] for s in traj["steps"] for o in s["ops"]}


class Replay:
    def __init__(self, traj, echo=True, diag=True, none_is_landed=True, per_step=False):
        self.t = traj
        self.echo, self.diag, self.none_landed, self.per_step = echo, diag, none_is_landed, per_step
        self.next_id = 1
        self.cur = {}          # (conv, path) -> content id
        self.seen = defaultdict(set)
        self.editor_hist = defaultdict(list)
        self.modified = set()
        self.views = defaultdict(list)   # (conv,path) -> [(step_i, cid, kind, op_index)]
        self.writes = defaultdict(list)  # (conv,path) -> [(step_i, op_index, cid_before, cid_after, landed)]
        self.opps = []
        self.hazards = []
        self.intra = []  # later anchored edits in the same unit after a landed write to the file
        self.not_executed = []  # anchored edit commands that never ran (not found, syntax error)
        self.known = known_paths(traj)

    def cid(self, k):
        return self.cur.get(k, 0)

    def new(self):
        self.next_id += 1
        return self.next_id

    def run(self):
        steps = self.t["steps"]
        root = self.t.get("root")
        groups = []
        for s in steps:
            key = (s["conv"], s["i"]) if self.per_step else (s["conv"], s["unit"])
            if groups and groups[-1][0] == key:
                groups[-1][1].append(s)
            else:
                groups.append((key, [s]))
        for (conv, unit), members in groups:
            held = {k: set(v) for k, v in self.seen.items() if k[0] == conv}
            scored = set()
            written_in_unit = set()
            pending_views = []
            for s in members:
                for oi, o in enumerate(s["ops"]):
                    k = (conv, o["path"])
                    if o["op"] == "R":
                        if o.get("landed") is False or o.get("kind") == "listing":
                            continue
                        if o.get("kind") == "echo" and not self.echo:
                            continue
                        pending_views.append((k, self.cid(k), s["i"], o.get("kind"), oi))
                        continue
                    landed = o.get("landed")
                    lands = landed is True or (landed is None and self.none_landed)
                    if o["path"] == "*":
                        if lands and o["kind"] == "restore":
                            for kk in list(self.modified):
                                if kk[0] == conv:
                                    before = self.cid(kk)
                                    self.cur[kk] = 0
                                    self.writes[kk].append((s["i"], oi, before, 0, landed, "restore*", unit))
                        continue
                    if o["kind"] == "anchored" and o.get("ran") is False:
                        self.not_executed.append({"step": s["i"], "op": oi, "path": o["path"], "ev": o.get("ev")})
                    elif o["kind"] == "anchored" and not o.get("no_score"):
                        if k not in scored:
                            scored.add(k)
                            if k in held:
                                c = self.cid(k)
                                haz = c not in held[k]
                                lv = self.views[k][-1] if self.views[k] else None
                                lv_pos = (lv[0], 10 ** 6 if lv[3] == -1 else lv[3]) if lv else (-1, -1)
                                W = [w for w in self.writes[k] if lv_pos < (w[0], w[1]) < (s["i"], oi)]
                                rec = {"step": s["i"], "op": oi, "path": o["path"], "conv": conv, "unit": unit,
                                       "bench": s.get("bench"), "hazard": haz, "landed": landed,
                                       "mech": o.get("mech"), "at_risk": any(w[3] != w[2] for w in self.writes[k]),
                                       "last_view": lv, "W": W, "n_views_before": len(self.views[k])}
                                self.opps.append(rec)
                                if haz:
                                    self.hazards.append(rec)
                        elif k in written_in_unit:
                            prior = [w for w in self.writes[k] if w[6] == unit]
                            self.intra.append({"step": s["i"], "op": oi, "path": o["path"], "conv": conv, "unit": unit,
                                               "bench": s.get("bench"), "landed": landed, "mech": o.get("mech"),
                                               "prior_same_unit_writes": prior})
                    if not lands or o.get("no_apply"):
                        continue  # no_apply: scored where it was composed, applied later (a background command)
                    before = self.cid(k)
                    kind = o["kind"]
                    if kind == "restore":
                        if k in self.modified:
                            self.cur[k] = 0
                    elif kind == "touch":
                        continue
                    elif kind == "replace" and o.get("src"):
                        sk = (conv, o["src"])
                        self.cur[k] = self.cid(sk) if sk in self.cur or sk in self.seen else self.new()
                        if self.cur[k] == 0:
                            self.cur[k] = ("base", o["src"])
                    elif kind == "undo":
                        hist = self.editor_hist[k]
                        self.cur[k] = hist.pop() if hist else self.new()
                    else:
                        self.cur[k] = self.new()
                    if o.get("mech", "").startswith("editor"):
                        self.editor_hist[k].append(before)
                    self.modified.add(k)
                    written_in_unit.add(k)
                    self.writes[k].append((s["i"], oi, before, self.cid(k), landed, kind, unit))
                    if kind == "overwrite":
                        # the agent authored every byte: a current view
                        held.setdefault(k, set()).add(self.cur[k])
                        pending_views.append((k, self.cur[k], s["i"], "authored", oi))
                if self.diag:
                    for p, lines in output_views(s, root, self.known).items():
                        k = (conv, p)
                        pending_views.append((k, self.cid(k), s["i"], "diag", -1))
            for k, c, si, kind, oi in pending_views:
                self.seen[k].add(c)
                self.views[k].append((si, c, kind, oi, unit))
        return self


def _binom_tail(k, n, p, upper):
    """P(X >= k) if upper else P(X <= k), X ~ Binomial(n, p), summed in log space (standard library only)."""
    import math
    if p <= 0.0:
        return 1.0 if (upper and k <= 0) or (not upper) else 0.0
    if p >= 1.0:
        return 1.0 if (upper) or k >= n else 0.0
    lp, lq, lg = math.log(p), math.log1p(-p), math.lgamma(n + 1)
    rng = range(k, n + 1) if upper else range(0, k + 1)
    terms = [lg - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq for i in rng]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def cp_ci(k, n, a=0.05):
    """Clopper-Pearson interval, found by bisection on the exact binomial tails (no scipy needed)."""
    if n == 0:
        return (None, None)

    def solve(f, target, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if (f(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    lo = 0.0 if k == 0 else solve(lambda p: _binom_tail(k, n, p, True), a / 2, True)
    hi = 1.0 if k == n else solve(lambda p: _binom_tail(k, n, p, False), a / 2, False)
    return (round(lo, 4), round(hi, 4))


def cell(t):
    fw = t["agent"]
    if fw == "OpenHands":
        fw = "OpenHands"
    return fw, short_model(t["model"])


def main():
    variants = {
        "primary": dict(),
        "no_echo": dict(echo=False),
        "no_diag": dict(diag=False),
        "none_as_failed": dict(none_is_landed=False),
        "per_step": dict(per_step=True),
    }
    tallies = {v: defaultdict(Counter) for v in variants}
    per_traj = []
    hz_out = open(os.path.join(WORK, "hazards.jsonl"), "w", encoding="utf-8")
    opp_out = open(os.path.join(WORK, "opportunities.jsonl"), "w", encoding="utf-8")
    intra_out = open(os.path.join(WORK, "intra_unit.jsonl"), "w", encoding="utf-8")
    with open(os.path.join(WORK, "extracted.jsonl"), encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if t.get("missing"):
                continue
            c = cell(t)
            row = {"traj_id": t["traj_id"], "agent": t["agent"], "model": short_model(t["model"]),
                   "layout": t.get("layout"), "steps": t["n_steps"], "n_conv": t.get("n_conv")}
            for v, kw in variants.items():
                r = Replay(t, **kw).run()
                tal = tallies[v][c]
                tal["trajectories"] += 1
                tal["steps"] += t["n_steps"]
                tal["opportunities"] += len(r.opps)
                tal["hazards"] += len(r.hazards)
                tal["at_risk_opportunities"] += sum(1 for o in r.opps if o["at_risk"])
                tal["zero_opp_trajectories"] += not r.opps
                tal["hazard_trajectories"] += bool(r.hazards)
                tal["intra_unit_repeat_edits"] += len(r.intra)
                tal["anchored_not_executed"] += len(r.not_executed)
                row[f"{v}_opps"] = len(r.opps)
                row[f"{v}_haz"] = len(r.hazards)
                if v == "primary":
                    meta = dict(traj_id=t["traj_id"], agent=t["agent"], model=short_model(t["model"]),
                                layout=t.get("layout"))
                    for o in r.opps:
                        opp_out.write(json.dumps(dict(o, **meta), default=str) + "\n")
                    for h in r.hazards:
                        hz_out.write(json.dumps(dict(h, **meta), default=str) + "\n")
                    for x in r.intra:
                        intra_out.write(json.dumps(dict(x, **meta), default=str) + "\n")
                    row["intra"] = len(r.intra)
            ops = Counter()
            for s in t["steps"]:
                for o in s["ops"]:
                    ops[(o["op"], o["kind"])] += 1
            row["ops"] = {f"{a}:{b}": n for (a, b), n in ops.items()}
            row["zero_op_steps"] = sum(1 for s in t["steps"] if not s["ops"])
            per_traj.append(row)
    hz_out.close()
    opp_out.close()
    intra_out.close()
    summary = {}
    for v, cells in tallies.items():
        summary[v] = {}
        for (fw, m), tal in sorted(cells.items()):
            n, k = tal["opportunities"], tal["hazards"]
            summary[v][f"{fw}|{m}"] = dict(tal, rate=round(k / n, 4) if n else None, ci=cp_ci(k, n))
        # framework totals
        fwt = defaultdict(Counter)
        for (fw, m), tal in cells.items():
            fwt[fw].update(tal)
        for fw, tal in fwt.items():
            n, k = tal["opportunities"], tal["hazards"]
            summary[v][f"{fw}|ALL"] = dict(tal, rate=round(k / n, 4) if n else None, ci=cp_ci(k, n))
    with open(os.path.join(WORK, "score_summary.json"), "w") as f:
        json.dump({"summary": summary, "per_trajectory": per_traj}, f, indent=1)
    for key, v in summary["primary"].items():
        print(f"{key:40s} traj {v['trajectories']:4d} steps {v['steps']:6d} opp {v['opportunities']:5d} "
              f"haz {v['hazards']:5d} rate {v['rate']} ci {v['ci']} zeroOpp {v['zero_opp_trajectories']}")
    print("--- variants (hazards/opportunities by framework)")
    for var in variants:
        print(var, {k: (v["hazards"], v["opportunities"]) for k, v in summary[var].items() if k.endswith("|ALL")})


if __name__ == "__main__":
    main()
