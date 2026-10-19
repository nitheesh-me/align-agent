"""Automated materialization classifier for structural hazards (refined protocol). Usage: python3 scripts/audit/materialize.py

For each hazard E (an anchored edit composed against a view that is no longer
current) on file f, with W = the writes to f between the agent's last view of
f and E, it rebuilds what can be known of:
  V   the text the agent last saw of f (all views of f's last-seen version)
  B   the believed state: V plus the agent's own later writes as commanded
      (minus any earlier write that E simply re-issues)
  R   the real state: B unless a write in W failed or did not do what was
      commanded (no exact oracle exists: OpenHands' logged old_content is
      the old_str argument, not the file)
and assigns the first class that fires:
  1 SPURIOUS           no write in W changed f (static no-op or failed)
  2 FAILED-UNRELATED   E did nothing for reasons unrelated to content, or its
                       anchor was not in any text of f the agent had seen or written
  2b FAILED-OWN-WRITE  E's content anchor was removed by a write in W exactly as
                       the agent commanded it: E fails on B and on R alike
                       (named bucket; not materialized under the definition)
  3 MATERIALIZED       R and B give different results in E's region:
                       M1 loud failure, M2 silent no-op, M3 drift/double application
  4 BENIGN-FORWARDED   overlap or drift exists but E is right against R
                       (anchors on W's new text, adjusted or runtime-resolved lines)
  5 BENIGN-DISJOINT    E's anchor shares no text with W's changes; no drift
  6 INDETERMINATE      anchor or W not extractable, or the static prediction is
                       contradicted by what the trajectory later shows
Also classifies intra-unit repeated edits (a later edit in the same action after
an earlier write to the same file) and computes the region-level sensitivity
for all opportunities. Writes work/materialization.jsonl and
work/materialization_summary.json.
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import WORK, dump  # noqa: E402
from excerpts import items as all_items, load_trajs  # noqa: E402
from score import output_views  # noqa: E402

OTHER_WRITERS = ("sidechain:", "subagent:", "parent:", "peer:")  # mech prefixes of writes other agents made
NUM_LINE = re.compile(r"^\s*(\d+)\t(.*)$")
NUM_COLON = re.compile(r"^(\d+)[:-](.*)$")


# ---------------------------------------------------------------- text helpers

def norm(s):
    return re.sub(r"[ \t]+", " ", (s or "")).strip()


def literal_key(pat, bre=True):
    """Longest literal fragment of a regex pattern (for searching text)."""
    if not pat:
        return ""
    frags, cur, i = [], [], 0
    while i < len(pat):
        c = pat[i]
        if c == "\\" and i + 1 < len(pat):
            n = pat[i + 1]
            if n in "nt":
                cur.append("\n" if n == "n" else "\t")
            elif n in "sSwWdDbB<>`'":
                frags.append("".join(cur))
                cur = []
            elif bre and n in "(){}|+?":
                frags.append("".join(cur))
                cur = []
            else:
                cur.append(n)
            i += 2
            continue
        if c in ".[^$" or (not bre and c in "()|+?{}"):
            if c == "[":
                j = pat.find("]", i + 2)
                i = j + 1 if j > 0 else len(pat)
            else:
                i += 1
            frags.append("".join(cur))
            cur = []
            continue
        if c in "*" or (not bre and c in "+?"):
            if cur:
                cur.pop()
            frags.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    frags.append("".join(cur))
    frags = [f for f in frags if len(f.strip()) >= 3]
    return max(frags, key=len) if frags else ""


def shown_lines(step, path):
    """{line number: text} the step's output showed for path, when numbered."""
    obs = step.get("obs") or ""
    out = {}
    base = path.rsplit("/", 1)[-1]
    for ln in obs.split("\n"):
        m = NUM_LINE.match(ln)
        if m:
            out[int(m.group(1))] = m.group(2)
            continue
        m = re.match(r"^(\S+?):(\d+):(.*)$", ln)
        if m and m.group(1).endswith(base):
            out[int(m.group(2))] = m.group(3)
            continue
    return out


def view_text(t, step_i):
    s = t["steps"][step_i]
    obs = s.get("obs") or ""
    if not obs and t["agent"] == "Terminus2":
        same = [x for x in t["steps"] if x.get("episode") == s.get("episode") and x.get("obs")]
        obs = same[-1]["obs"] if same else ""
    return obs


def strip_numbers(text):
    """Text with cat -n / nl / grep -n prefixes removed, line by line."""
    out = []
    for ln in (text or "").split("\n"):
        m = NUM_LINE.match(ln)
        if m:
            out.append(m.group(2))
            continue
        m = re.match(r"^(?:\S+?:)?\d+[:-](.*)$", ln)
        out.append(m.group(1) if m else ln)
    return "\n".join(out)


def contains(hay, needle):
    if not needle or not hay:
        return False
    if needle in hay:
        return True
    return norm(needle) in norm(hay) if len(norm(needle)) >= 4 else False


def lines_of(s):
    return [x.strip() for x in (s or "").split("\n") if len(x.strip()) >= 4]


def overlaps(a, b):
    """Two texts share a line (or one contains the other)."""
    if not a or not b:
        return False
    if contains(a, b) or contains(b, a):
        return True
    la, lb = set(lines_of(a)), set(lines_of(b))
    return bool(la & lb)


_COMMENT = re.compile(r"^(//|#|/\*|\*|--|;|<!--)")


def substantive_lines(text):
    """Lines that identify a block of code: 12+ characters after folding
    whitespace and sed's escapes, with a word character, and not a comment.
    Braces, `pub(crate)` or a shared doc-comment line do not make two blocks
    the same."""
    out = set()
    for ln in (text or "").split("\n"):
        x = norm(re.sub(r"\\(.)", r"\1", ln.replace("\\t", " ").replace("\\n", " ")))
        if len(x) >= 12 and re.search(r"\w", x) and not _COMMENT.match(x):
            out.add(x)
    return out


def removed_later(j, wops, added):
    """Whether a write after wops[j] took back what wops[j] added: a restore of
    the file, or a write that removes that text."""
    for _, op, dw, w in wops[j + 1:]:
        if w[4] is False:
            continue
        if op.get("kind") in ("restore", "restore*"):
            return True
        if any(overlaps(r, a) for r in dw["removed_only"] for a in added):
            return True
    return False


def sed_cmd(op):
    """(command, pattern) of a single-command sed edit, else (None, None)."""
    a = op.get("anchor") or {}
    cmds = a.get("cmds") or []
    if a.get("type") != "sed" or len(cmds) != 1:
        return None, None
    return cmds[0].get("cmd"), cmds[0].get("pat")


# ---------------------------------------------------------------- anchors and write effects

def dynamic_address(raw_scripts):
    return any(re.match(r"^\s*\$\{?[A-Za-z_]", s or "") or re.search(r"\$\{?[A-Za-z_]\w*\}?[aicd]", s or "")
               for s in raw_scripts or [])


def _describe(op):
    """Effects of a write op as commanded: removed texts, added texts, line
    changes [(line or None, delta)], anchors for E: content keys and line numbers."""
    a = op.get("anchor") or {}
    t = a.get("type")
    d = {"removed": [], "added": [], "lines": [], "keys": [], "line_anchors": [], "opaque": False,
         "static_noop": False, "dynamic": False, "exact": False}
    kind = op.get("kind")
    if kind in ("append",):
        d["added"].append(op.get("text") or "")
        return d
    if kind in ("replace", "rewrite", "overwrite_unseen", "restore", "undo", "delete", "restore*"):
        d["opaque"] = True
        return d
    if kind == "overwrite":
        d["opaque"] = True
        return d
    if t == "str_replace":
        old, new = a.get("old") or "", a.get("new") or ""
        d.update(removed=[old], added=[new], keys=[old], exact=True)
        d["lines"].append((None, new.count("\n") - old.count("\n")))
        d["static_noop"] = old == new
    elif t == "insert":
        new = a.get("new") or ""
        try:
            ln = int(a.get("line"))
        except (TypeError, ValueError):
            ln = None
        d["added"].append(new)
        d["lines"].append((ln, new.count("\n") + 1))
        if ln is not None:
            d["line_anchors"].append(ln)
    elif t == "lines":  # SWE-agent 0.x edit N:M
        new = a.get("new") or ""
        st, en = a.get("start"), a.get("end")
        d["added"].append(new)
        d["line_anchors"] += [st, en]
        d["lines"].append((st, (new.count("\n") + 1) - (en - st + 1)))
    elif t == "py":
        for r in a.get("replaces") or []:
            d["removed"].append(r["old"])
            d["added"].append(r["new"])
            d["keys"].append(r["old"])
            d["lines"].append((None, r["new"].count("\n") - r["old"].count("\n")))
            if r["old"] == r["new"]:
                d["static_noop"] = True
        for r in a.get("regex") or []:
            k = literal_key(r.get("pat") or "", bre=False)
            if k:
                d["removed"].append(k)
                d["keys"].append(k)
            if r.get("rep"):
                d["added"].append(re.sub(r"\\\d|\\g<\w+>", "", r["rep"]))
            d["lines"].append((None, None))
        for f in a.get("finds") or []:
            d["keys"].append(f)
        if a.get("line_ops"):
            d["lines"].append((None, None))
            d["opaque"] = not (a.get("replaces") or a.get("regex") or a.get("finds"))
        d["exact"] = bool(a.get("replaces"))
    elif t in ("sed", "perl"):
        raw = a.get("raw") or [a.get("script") or ""]
        if dynamic_address(raw):
            d["dynamic"] = True
        cmds = a.get("cmds") or [{"cmd": "s", "addr": [], "pat": x.get("pat"), "rep": x.get("rep")} for x in a.get("subs") or []]
        bre = not a.get("E") and t == "sed"
        for c in cmds:
            addr = c.get("addr") or []
            nums = [x["line"] for x in addr if isinstance(x.get("line"), int)]
            res = [x["re"] for x in addr if x.get("re")]
            for r_ in res:
                k = literal_key(r_, bre)
                if k:
                    d["keys"].append(k)
            if nums and not d["dynamic"]:
                d["line_anchors"] += nums
            cmd = c.get("cmd")
            if cmd == "s":
                pat, rep = c.get("pat") or "", c.get("rep") or ""
                k = literal_key(pat, bre)
                if k:
                    d["removed"].append(k)
                    d["keys"].append(k)
                rep_lit = re.sub(r"\\[0-9&]|&", "", rep).replace("\\n", "\n").replace("\\/", "/")
                d["added"].append(rep_lit)
                nl_in = "\\n" in pat
                if nl_in and not a.get("z") and not any(x.get("cmd") in ("N", "D", "P") for x in cmds) and t == "sed":
                    d["static_noop"] = True  # a newline cannot occur inside sed's pattern space
                if pat and rep and pat == rep:
                    d["static_noop"] = True
                delta = rep_lit.count("\n") - pat.count("\\n")
                d["lines"].append((nums[0] if nums else None, delta if not (nl_in and d["static_noop"]) else 0))
            elif cmd in ("a", "i"):
                d["added"].append(c.get("text") or "")
                pos = (nums[0] + (1 if cmd == "a" else 0)) if nums else None
                d["lines"].append((pos, (c.get("text") or "").count("\n") + 1))
            elif cmd == "c":
                d["added"].append(c.get("text") or "")
                if len(nums) == 2:
                    d["lines"].append((nums[0], (c.get("text") or "").count("\n") + 1 - (nums[1] - nums[0] + 1)))
                else:
                    d["lines"].append((nums[0] if nums else None, None if res else 0))
            elif cmd == "d":
                rel = [x["rel"] for x in addr if isinstance(x.get("rel"), int)]
                if rel:
                    d["lines"].append((nums[0] if nums else None, -(rel[0] + 1)))
                elif len(nums) == 2:
                    d["lines"].append((nums[0], -(nums[1] - nums[0] + 1)))
                elif len(nums) == 1:
                    d["lines"].append((nums[0], -1))
                else:
                    d["lines"].append((None, None))
                for r_ in res:
                    d["removed"].append(literal_key(r_, bre))
            elif cmd in ("r", "R"):
                # inserts a file the command names: its length is not known here
                d["lines"].append(((nums[0] + 1) if nums else None, None))
            elif cmd in ("?", "y", "{", "}", "p", "q"):
                if cmd == "?":
                    d["opaque"] = True
    elif t == "patch":
        for h in a.get("hunks") or []:
            old = "\n".join(ln[1:] for ln in h["lines"] if ln[:1] in (" ", "-"))
            new = "\n".join(ln[1:] for ln in h["lines"] if ln[:1] in (" ", "+"))
            rem = "\n".join(ln[1:] for ln in h["lines"] if ln[:1] == "-")
            add = "\n".join(ln[1:] for ln in h["lines"] if ln[:1] == "+")
            if rem:
                d["removed"].append(rem)
            if add:
                d["added"].append(add)
            d["keys"].append(old)
            d["lines"].append((h.get("old_start"), new.count("\n") - old.count("\n")))
        d["exact"] = True
    elif t in ("ed", "awk"):
        d["opaque"] = True
    else:
        d["opaque"] = True
    d["removed"] = [x for x in d["removed"] if x and x.strip()]
    d["added"] = [x for x in d["added"] if x is not None]
    d["keys"] = [x for x in d["keys"] if x and len(x.strip()) >= 3]
    # lines that a write only removes / only adds (context shared by the old
    # and new text is neither)
    rem_l = set(lines_of("\n".join(d["removed"])))
    add_l = set(lines_of("\n".join(d["added"])))
    d["removed_only"] = [x for x in d["removed"] if len(lines_of(x)) <= 1] + ["\n".join(sorted(rem_l - add_l))]
    d["added_only"] = [x for x in d["added"] if len(lines_of(x)) <= 1] + ["\n".join(sorted(add_l - rem_l))]
    d["removed_only"] = [x for x in d["removed_only"] if x.strip()]
    d["added_only"] = [x for x in d["added_only"] if x.strip()]
    return d


_LIT = re.compile(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', re.S)


def describe(op, action=None):
    """Effects of a write op; for Python scripts, the long string literals in
    the script are also taken as text the write may add (their exact use is
    not modelled)."""
    d = _describe(op)
    d.setdefault("removed_only", [x for x in d["removed"] if x and x.strip()])
    d.setdefault("added_only", [x for x in d["added"] if x and x.strip()])
    if op.get("mech") == "python" and action:
        lits = [a or b for a, b in _LIT.findall(action)]
        lits = [x.encode("utf-8", "replace").decode("unicode_escape", "replace") if "\\n" in x else x for x in lits]
        d["script_literals"] = [x for x in lits if len(x.strip()) >= 20]
        if d["opaque"] or op.get("kind") in ("rewrite", "overwrite_unseen"):
            d["added_only"] = d["added_only"] + d["script_literals"]
    return d


def later_evidence(t, item, de, horizon=10):
    """What later views of the file show about E: 'applied' when E's new text
    appears, 'not_applied' when the anchor text is still shown and the new
    text is not, else 'unknown'. Stops at the next write to the file."""
    steps = t["steps"]
    new_texts = [x for x in de.get("added_only") or [] if len(x.strip()) >= 6]
    key = max(de["keys"], key=len) if de["keys"] else ""
    s_e = steps[item["step"]]
    e = s_e["ops"][item["op"]]
    if e.get("mech") in ("editor",) and e.get("landed") is True:
        return "applied (editor confirmed)"
    for s in steps[item["step"] + 1:item["step"] + 1 + horizon]:
        wrote = any(o["op"] == "W" and o["path"] == item["path"] and o.get("landed") is not False for o in s["ops"])
        viewed = any(o["op"] == "R" and o["path"] == item["path"] and o.get("landed") is not False for o in s["ops"])
        if viewed:
            txt = strip_numbers(view_text(t, s["i"]))
            if any(contains(txt, n) for n in new_texts):
                return "applied (later view shows the new text)"
            if key and contains(txt, key) and new_texts and not any(contains(txt, n) for n in new_texts):
                return "not_applied (later view still shows the anchor)"
        if wrote:
            break
    fd = t.get("final_diff") or ""
    if fd and new_texts:
        if any(n.strip() in fd for n in new_texts):
            return "applied (final diff has the new text)"
    return "unknown"


# ---------------------------------------------------------------- classification

def same_command(a, b):
    return norm(a) == norm(b) and len(norm(a)) > 0


def classify(item, t):
    """(class, reason, evidence) for one hazard. Claude Code's own staleness signals, where the log has
    them, override the static rules: a refused edit was caught (CAUGHT-BY-HARNESS, not a failure the
    agent suffered), and an edit it applied while noting other changes on disk was stale but harmless."""
    e = t["steps"][item["step"]]["ops"][item["op"]]
    if e.get("harness") == "refused-stale":
        return "CAUGHT-BY-HARNESS", "Claude Code refused the edit: the file was modified since the agent read it", \
            {"e_mech": e.get("mech"), "e_landed": e.get("landed"), "e_ev": e.get("ev")}
    cls, why, ev = _classify(item, t)
    if e.get("harness") == "noted-stale" and not cls.startswith("MATERIALIZED"):
        return "BENIGN-HARNESS-NOTED", "Claude Code applied the edit and noted other changes on disk (static: " + cls + ")", ev
    return cls, why, ev


def _classify(item, t):
    steps = t["steps"]
    s_e = steps[item["step"]]
    e = s_e["ops"][item["op"]]
    de = describe(e, s_e.get("action"))
    W = [w for w in (item.get("W") or item.get("prior_same_unit_writes") or [])]
    wops = []
    for w in W:
        s_w = steps[w[0]]
        op = s_w["ops"][w[1]] if 0 <= w[1] < len(s_w["ops"]) else {"kind": "restore*", "mech": "git"}
        wops.append((s_w, op, describe(op, s_w.get("action")), w))
    lv = item.get("last_view")
    V_text, V_lines = "", {}
    if lv:
        # all views of the last-seen version
        vsteps = [lv[0]]
        V_text = strip_numbers(view_text(t, lv[0]))
        V_lines = shown_lines(steps[lv[0]], item["path"])
    K_text = V_text
    paths = {item["path"]}
    for s in steps[:item["step"]]:
        for o in s["ops"]:
            if o["op"] == "W" and o["path"] == item["path"] and o.get("src"):
                paths.add(o["src"])  # content copied or moved in from another file
    for s in steps[:item["step"]]:
        for o in s["ops"]:
            if o["op"] == "R" and o["path"] in paths and o.get("landed") is not False:
                K_text += "\n" + strip_numbers(view_text(t, s["i"]))
            if o["op"] == "W" and o["path"] in paths and o.get("kind") in ("overwrite", "append") and o.get("text"):
                K_text += "\n" + o["text"]
    ev = {"e_mech": e.get("mech"), "e_landed": e.get("landed"), "e_ev": e.get("ev"),
          "n_W": len(W), "W_kinds": [op.get("kind") + ":" + str(op.get("mech")) for _, op, _, _ in wops],
          "W_landed": [w[4] for w in W], "keys": [k[:80] for k in de["keys"][:3]],
          "line_anchors": de["line_anchors"][:4]}
    # --- 1 SPURIOUS: no write in W made a real change
    real_change = []
    for s_w, op, dw, w in wops:
        if w[4] is False:
            continue
        orc = op.get("oracle") or {}
        if orc.get("old_content") is not None and orc.get("new_content") is not None and orc["old_content"] == orc["new_content"]:
            continue
        if dw["static_noop"]:
            continue
        if op.get("kind") == "restore*" or (op.get("kind") == "restore" and op.get("mech") == "git"):
            real_change.append((s_w, op, dw, w))
            continue
        # a substitution whose pattern the agent had seen nowhere, followed by
        # a view that still shows the pattern: it did not match
        real_change.append((s_w, op, dw, w))
    if not real_change:
        return "SPURIOUS", "no write after the last view changed the file (static no-op or failed)", ev
    # (No OpenHands oracle: in the event streams extras.old_content is the
    # edit's old_str argument, not the file content; see oh_oracle_check.py.)
    # --- 2 FAILED-UNRELATED: E failed for non-content reasons
    e_ev = (e.get("ev") or "")
    anc = e.get("anchor") or {}
    if anc.get("type") == "str_replace" and e.get("landed") is False and anc.get("old"):
        raw = anc["old"]
        if raw != raw.strip("\n") and not contains(K_text, raw) and contains(K_text, raw.strip("\n")):
            return "FAILED-UNRELATED", "editor rejected an old_str whose only mismatch is leading/trailing newlines", ev
    if e.get("landed") is False and not re.search(r"did not appear verbatim|No replacement|patch failed|FAILED|Multiple occurrences", e_ev + " " + (s_e.get("obs") or "")[:2000]):
        return "FAILED-UNRELATED", f"edit failed for a reason unrelated to content ({e_ev[:60]})", ev
    # --- E cannot change anything (s/X/X/)
    if de["static_noop"]:
        return "FAILED-UNRELATED", "the edit replaces text with itself", ev
    # --- double application: E re-issues an earlier write that landed
    e_inserts = any(c.get("cmd") in ("a", "i") for c in ((e.get("anchor") or {}).get("cmds") or [])) or \
        (e.get("anchor") or {}).get("type") in ("insert",) or e.get("kind") == "append"
    for s_w, op, dw, w in real_change:
        same = same_command(json.dumps(op.get("anchor"), sort_keys=True), json.dumps(e.get("anchor"), sort_keys=True)) \
            and op.get("mech") == e.get("mech")
        if same and removed_later(wops.index((s_w, op, dw, w)), wops, dw["added_only"]):
            continue  # restored or removed before E re-issued it
        if same:
            ev["reissue_of_step"] = s_w["i"]
            if e_inserts:
                return "MATERIALIZED-M3", f"re-issues the insertion made at step {s_w['i']}, which had landed: double application", ev
            return "BENIGN-FORWARDED", f"re-issues the substitution made at step {s_w['i']}; on the real file it is a no-op with the same final text", ev
    # --- double application by a different command: E adds text an unseen
    # write already added, without replacing that text
    # The duplicate must be the same block (most of E's substantive lines) put
    # at the same place (E's anchor is W's anchor, or text W wrote), and still
    # in the file when E ran.
    e_sub = substantive_lines("\n".join(de["added_only"]))
    for s_w, op, dw, w in real_change:
        w_new = [x for x in dw["added_only"] if len(norm(x)) >= 12]
        w_sub = substantive_lines("\n".join(dw["added_only"]))
        same_block = bool(e_sub) and len(e_sub & w_sub) >= 0.8 * len(e_sub)
        w_text = "\n".join(dw["added"] + dw["added_only"])
        same_place = any(overlaps(a, b) for a in de["keys"] for b in dw["keys"]) or \
            any(contains(w_text, a) for a in de["keys"]) or (dw["opaque"] and not dw["keys"])
        replaces_it = any(overlaps(r, y) for r in de["removed_only"] for y in w_new)
        if same_block and same_place and e_inserts and not replaces_it and e.get("landed") is not False \
                and not removed_later(wops.index((s_w, op, dw, w)), wops, dw["added_only"]):
            ev["duplicate_of_step"] = s_w["i"]
            return "MATERIALIZED-M3", f"adds text the unseen write at step {s_w['i']} had already added: double application", ev
    # --- the unseen write restored or copied known content (cp from a
    # backup the agent made, git checkout) and E was composed for it
    last = real_change[-1]
    if last[1].get("kind") in ("replace", "restore", "restore*") and (last[1].get("src") or last[1].get("kind") != "replace"):
        same_unit = last[3][6] == item.get("unit")
        if same_unit or e.get("landed") is True or (de["keys"] and contains(K_text, max(de["keys"], key=len))):
            return "BENIGN-FORWARDED", ("composed together with the restore in the same action" if same_unit
                                        else "edit applies to the restored content the agent knew"), ev
    # --- dynamic (runtime-resolved) anchors
    if de["dynamic"]:
        return "BENIGN-FORWARDED", "line address computed at run time from the current file", ev
    # --- line drift. A write that left the file exactly as the agent last saw
    # it (a git checkout back to a viewed committed version, a revert) resets
    # the line numbering: only writes after the last such one can shift it.
    drift_writes = real_change
    lv_cid = lv[1] if lv else None
    back = [j for j, (_, _, _, w) in enumerate(real_change) if w[3] == lv_cid]
    if back:
        drift_writes = real_change[back[-1] + 1:]
    if de["line_anchors"] and not de["keys"]:
        n = min(x for x in de["line_anchors"] if isinstance(x, int))
        e_cmd, e_pat = sed_cmd(e)
        if n == 1 and (e_cmd == "i" or (e_cmd == "s" and e_pat == "^")):
            return "BENIGN-FORWARDED", "inserts at the top of the file, which no change above can move", ev
        deltas, unknown, above = 0, False, []
        for _, op, dw, _ in drift_writes:
            if dw["opaque"]:
                unknown = True
            for pos, dl in dw["lines"]:
                if dl is None:
                    if pos is None or pos <= n:
                        unknown = True
                elif dl != 0:
                    if pos is None:
                        unknown = True
                    elif dl < 0 and pos == n and e_cmd == "i":
                        continue  # E inserts where a block was deleted: a replacement
                    elif pos <= n:
                        deltas += dl
                        above.append((pos, dl, sed_cmd(op)[0]))
        ev["line_delta_above"] = deltas
        # The agent's own run of line inserts, each placed right after the
        # previous one (133a, 134a, 135a): it accounted for every shift.
        e_pos = de["lines"][0][0] if de["lines"] else None
        if deltas and not unknown and e_cmd in ("a", "i") and e_pos is not None and \
                all(c in ("a", "i") and dl > 0 for _, dl, c in above) and \
                all(above[k][0] == above[k - 1][0] + above[k - 1][1] for k in range(1, len(above))) and \
                e_pos == above[-1][0] + above[-1][1]:
            return "BENIGN-FORWARDED", "continues the agent's own run of line inserts, each placed after the last", ev
        if deltas != 0 and not unknown:
            return "MATERIALIZED-M3", f"line anchor {n} after a {deltas:+d}-line change above it (V numbering)", ev
        if unknown:
            return "INDETERMINATE", "line anchor; an unseen write changed line counts at an unknown position", ev
        return "BENIGN-DISJOINT", "line anchor; no line-count change above it", ev
    if not de["keys"]:
        return "INDETERMINATE", "edit anchor not extractable", ev
    a = max(de["keys"], key=len)
    removed_hit = [(s_w, dw) for s_w, _, dw, _ in real_change for r in dw["removed_only"] if overlaps(a, r)]
    added_hit = [(s_w, dw) for s_w, _, dw, _ in real_change for x in dw["added_only"] if x and overlaps(a, x)]
    ev["later"] = later_evidence(t, item, de)
    opaque = any(dw["opaque"] for _, _, dw, _ in real_change)
    in_K = contains(K_text, a) or bool(added_hit)
    ev["anchor_in_known_text"] = in_K
    loud = e.get("landed") is False
    # line anchors mixed with content (e.g. sed 'Ns/foo/bar/'): drift check with content
    if de["line_anchors"]:
        n = min(x for x in de["line_anchors"] if isinstance(x, int))
        deltas = sum(dl for _, _, dw, _ in drift_writes for pos, dl in dw["lines"] if dl and pos is not None and pos <= n)
        if deltas:
            vline = V_lines.get(n)
            if vline is not None and a in vline:
                return "MATERIALIZED-M3", f"line {n} holds the anchor only in the agent's last view; {deltas:+d} lines above moved it", ev
            return "BENIGN-FORWARDED", f"line {n} with a {deltas:+d} shift above; anchor not on the old line (adjusted)", ev
    if added_hit and not removed_hit:
        return "BENIGN-FORWARDED", f"anchor includes text the agent itself wrote at step {added_hit[0][0]['i']}", ev
    if removed_hit and not added_hit:
        if ev["later"].startswith("applied"):
            return "INDETERMINATE", "anchor overlaps text an unseen write removed, yet the edit's new text appears later", ev
        s_r = removed_hit[0][0]
        foreign = [o for o in s_r["ops"] if o["op"] == "W" and o["path"] == item["path"]
                   and str(o.get("mech") or "").startswith(OTHER_WRITERS)]
        if foreign:
            # another agent removed the anchor: the agent could not have known, so the failure is the stale view's
            who = foreign[0]["mech"].split(":")[0]
            if loud:
                return "MATERIALIZED-M1", f"anchor removed by a {who} write at step {s_r['i']} the agent never saw; the edit was rejected", ev
            return "MATERIALIZED-M2", f"anchor removed by a {who} write at step {s_r['i']} the agent never saw; the edit silently matched nothing", ev
        if loud:
            return "FAILED-OWN-WRITE", f"anchor removed by the agent's own write at step {removed_hit[0][0]['i']}; the edit was rejected", ev
        return "FAILED-OWN-WRITE", f"anchor removed by the agent's own write at step {removed_hit[0][0]['i']}; the edit silently matched nothing", ev
    if removed_hit and added_hit:
        return "INDETERMINATE", "anchor overlaps both removed and added text of unseen writes", ev
    if loud and re.search(r"did not appear verbatim|No replacement|patch failed|FAILED", e_ev + (s_e.get("obs") or "")[:2000]):
        if opaque:
            return "INDETERMINATE", "edit rejected after an unseen whole-file write", ev
        if not in_K:
            return "FAILED-UNRELATED", "edit rejected; its anchor was not in any text the agent had seen", ev
        return "INDETERMINATE", "edit rejected although the anchor was seen and no unseen write touched it", ev
    if opaque:
        return "INDETERMINATE", "an unseen whole-file write (copy, script rewrite, restore) with unknown content" + \
            ("; the edit's new text appears later" if ev["later"].startswith("applied") else ""), ev
    if ev["later"].startswith("not_applied"):
        return "INDETERMINATE", "no overlap with the unseen writes found, but a later view shows the edit did not apply", ev
    return "BENIGN-DISJOINT", "anchor shares no text with the unseen writes", ev


def main():
    its = all_items()
    trajs = load_trajs({it["traj_id"] for it in its})
    out = open(os.path.join(WORK, "materialization.jsonl"), "w", encoding="utf-8")
    counts = defaultdict(Counter)
    for it in its:
        t = trajs[it["traj_id"]]
        cls, why, ev = classify(it, t)
        rec = {k: it[k] for k in ("id", "traj_id", "agent", "model", "step", "op", "path", "kind_of_item", "bench")}
        rec.update(cls=cls, why=why, evidence=ev)
        out.write(json.dumps(rec, default=str) + "\n")
        counts[(it["kind_of_item"], it["agent"])][cls] += 1
        counts[(it["kind_of_item"], "ALL")][cls] += 1
    out.close()
    summary = {f"{k[0]}|{k[1]}": dict(v) for k, v in counts.items()}
    for k, v in sorted(summary.items()):
        print(k, dict(sorted(v.items())))
    dump(summary, "materialization_summary.json")


if __name__ == "__main__":
    main()
