"""Stale-edit analysis of synthetic Claude Code and Codex transcripts (python3 -m pytest tests, or python3 -m unittest)."""

import itertools
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from align_agent.core import analyze, cluster_ci, load, summarize  # noqa: E402

CWD = "/home/u/proj"
F = CWD + "/a.py"
TEXT = "import os\n\ndef f():\n    x = 1\n    return x\n\ndef g():\n    y = 2\n    return y\n"
CURRENT = " (file state is current in your context \u2014 no need to Read it back)"
CLOCK = itertools.count()


def ts():
    return "2026-01-01T00:%02d:%02dZ" % divmod(next(CLOCK), 60)


def numbered(text, start=1):
    return "\n".join("%6d\t%s" % (i + start, ln) for i, ln in enumerate(text.split("\n")))


class Transcript:
    """Claude Code JSONL lines: one assistant message per call, then its tool result."""

    def __init__(self, sid="s1", side=False):
        self.lines, self.sid, self.side, self.n = [], sid, side, 0

    def call(self, name, inp, result, is_error=False, tur=None):
        self.n += 1
        mid, tid = "m%d" % self.n, "tu%d" % self.n
        base = {"sessionId": self.sid, "cwd": CWD, "isSidechain": self.side}
        self.lines.append(dict(base, type="assistant", uuid=mid, parentUuid=None, timestamp=ts(), message={
            "id": mid, "role": "assistant", "model": "claude-x",
            "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}]}))
        d = dict(base, type="user", uuid=mid + "r", parentUuid=mid, timestamp=ts(), message={
            "role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": result, "is_error": is_error}]})
        if tur is not None:
            d["toolUseResult"] = tur
        self.lines.append(d)
        return self

    def notify(self, text):
        self.lines.append({"sessionId": self.sid, "cwd": CWD, "isSidechain": self.side, "type": "user", "uuid": "n%d" % len(self.lines),
                           "parentUuid": None, "timestamp": ts(), "message": {"role": "user", "content": text}})
        return self

    def read(self, text=TEXT):
        n = text.count("\n")
        return self.call("Read", {"file_path": F}, numbered(text),
                         tur={"type": "text", "file": {"filePath": F, "content": text, "numLines": n, "startLine": 1, "totalLines": n}})

    def edit(self, old, new, result=None, is_error=False):
        return self.call("Edit", {"file_path": F, "old_string": old, "new_string": new},
                         result if result is not None else "The file %s has been updated successfully.%s" % (F, CURRENT), is_error)

    def bash(self, cmd, out="", **kw):
        return self.call("Bash", dict(command=cmd, **kw), out)

    def write(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in self.lines)
        return path


def py_replace(a, b):
    return "python3 - <<'EOF'\nfrom pathlib import Path\np=Path('a.py')\ns=p.read_text()\ns=s.replace(%r,%r)\np.write_text(s)\nEOF" % (a, b)


def run(path):
    t, fmt, cov = load(path)
    r = analyze(t)
    r.update(file=path, format=fmt, coverage=cov)
    return t, r


class Analyzer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, *p):
        return os.path.join(self.d, *p)

    def test_own_unread_write_is_split_from_other_writes(self):
        _, r = run(Transcript().read().bash(py_replace("x = 1", "x = 10")).bash(py_replace("y = 2", "y = 20")).write(self.path("own.jsonl")))
        self.assertEqual(r["stale_edits"], 1)
        self.assertEqual(r["stale"][0]["writer"], "self")
        s = summarize([r])
        self.assertEqual((s["stale_after_own_writes"], s["stale_after_other_writes"]), (1, 0))

    def test_rejected_edit_is_scored(self):
        _, r = run(Transcript().read().bash("sed -i 's/x = 1/x = 5/' a.py")
                   .edit("x = 1", "x = 10", "<tool_use_error>String to replace not found in file.\nString: x = 1</tool_use_error>", True)
                   .write(self.path("rej.jsonl")))
        self.assertEqual(r["opportunities"], 2)
        self.assertEqual([s["class"] for s in r["stale"]], ["FAILED-OWN-WRITE"])

    def test_modified_since_read_is_caught_by_harness(self):
        _, r = run(Transcript().read().bash("sed -i 's/x = 1/x = 5/' a.py")
                   .edit("y = 2", "y = 20", "<tool_use_error>File has been modified since read, either by the user or by a "
                         "linter. Read it again before attempting to write it.</tool_use_error>", True)
                   .write(self.path("caught.jsonl")))
        self.assertEqual([s["class"] for s in r["stale"]], ["CAUGHT-BY-HARNESS"])
        self.assertFalse(r["stale"][0]["went_wrong"])
        self.assertEqual(summarize([r])["caught_by_harness"], 1)
        self.assertEqual(r["harness_oracle"].get("refused-stale:flagged"), 1)

    def test_non_content_refusal_still_not_run(self):
        _, r = run(Transcript().read().edit("x = 1", "x = 10", "<tool_use_error>InputValidationError: bad input</tool_use_error>", True)
                   .write(self.path("invalid.jsonl")))
        self.assertEqual(r["opportunities"], 0)
        self.assertEqual(r["coverage"].get("call_not_run"), 1)

    def test_noted_stale_edit_is_benign(self):
        note = ("The file %s has been updated successfully. (note: the file had been modified on disk since you last read it "
                "\u2014 the edit applied cleanly, but the file contains other changes not in your context.)" % F)
        _, r = run(Transcript().read().bash("sed -i 's/x = 1/x = 5/' a.py").edit("y = 2", "y = 20", note).write(self.path("noted.jsonl")))
        self.assertEqual([s["class"] for s in r["stale"]], ["BENIGN-HARNESS-NOTED"])

    def test_plain_success_is_an_echo(self):
        plain = "The file %s has been updated successfully." % F
        _, r = run(Transcript().read().edit("x = 1", "x = 10", plain).edit("y = 2", "y = 20", plain).write(self.path("plain.jsonl")))
        self.assertEqual(r["stale_edits"], 0)

    def test_python_membership_check_is_not_an_anchor(self):
        script = "python3 - <<'EOF'\np='a.py'\ns=open(p).read()\nif 'import os' in s:\n    print('ok')\nopen(p,'w').write(s.upper())\nEOF"
        t, r = run(Transcript().read().bash(script, "ok").write(self.path("rewrite.jsonl")))
        self.assertEqual([o["kind"] for s in t["steps"] for o in s["ops"] if o["op"] == "W"], ["rewrite"])
        self.assertEqual(r["anchored_edits"], 0)

    def test_python_splice_at_found_text_is_an_anchor(self):
        script = "python3 - <<'EOF'\ns=open('a.py').read()\ni=s.find('def g')\ns=s[:i]+'# g\\n'+s[i:]\nopen('a.py','w').write(s)\nEOF"
        t, _ = run(Transcript().read().bash(script).write(self.path("splice.jsonl")))
        self.assertEqual([o["kind"] for s in t["steps"] for o in s["ops"] if o["op"] == "W"], ["anchored"])

    def test_python_print_is_a_view(self):
        _, r = run(Transcript().read().bash(py_replace("x = 1", "x = 10"))
                   .bash("python3 -c \"print(open('a.py').read())\"", TEXT.replace("x = 1", "x = 10"))
                   .bash(py_replace("y = 2", "y = 20")).write(self.path("pyprint.jsonl")))
        self.assertEqual(r["stale_edits"], 0)

    def test_python_print_to_dev_null_is_not_a_view(self):
        _, r = run(Transcript().read().bash(py_replace("x = 1", "x = 10"))
                   .bash("python3 -c \"print(open('a.py').read())\" > /dev/null")
                   .bash(py_replace("y = 2", "y = 20")).write(self.path("pynull.jsonl")))
        self.assertEqual(r["stale_edits"], 1)

    def subagent_session(self):
        """Parent reads a.py, launches two subagents; a edits x, then b (which read before a's edit) edits y."""
        parent, a, b = Transcript(), Transcript(side=True), Transcript(side=True)
        parent.read()
        parent.call("Agent", {"prompt": "edit", "subagent_type": "general-purpose"}, "done")
        b.read()
        a.read().edit("x = 1", "x = 10")
        b.edit("return y", "return y + 1")
        parent.edit("return x", "return x + 1")
        main = parent.write(self.path("proj", "sess1.jsonl"))
        pa = a.write(self.path("proj", "sess1", "subagents", "agent-a.jsonl"))
        pb = b.write(self.path("proj", "sess1", "subagents", "agent-b.jsonl"))
        return main, pa, pb

    def test_parent_classifies_subagent_edit_by_its_text(self):
        main, _, _ = self.subagent_session()
        _, r = run(main)
        self.assertEqual(len(r["stale"]), 1)
        self.assertEqual(r["stale"][0]["writer"], "other")
        self.assertNotEqual(r["stale"][0]["class"], "INDETERMINATE")

    def test_sibling_subagent_sees_peer_write(self):
        _, _, pb = self.subagent_session()
        _, r = run(pb)
        self.assertEqual(r["stale_edits"], 1)
        self.assertEqual(r["stale"][0]["writer"], "other")

    def test_peer_removing_anchor_materializes(self):
        parent, a, b = Transcript(), Transcript(side=True), Transcript(side=True)
        parent.read().call("Agent", {"prompt": "x"}, "done")
        b.read()
        a.read().edit("y = 2", "y = 3")
        b.edit("y = 2", "y = 20", "<tool_use_error>String to replace not found in file.\nString: y = 2</tool_use_error>", True)
        parent.write(self.path("p2", "s.jsonl"))
        a.write(self.path("p2", "s", "subagents", "agent-a.jsonl"))
        _, r = run(b.write(self.path("p2", "s", "subagents", "agent-b.jsonl")))
        self.assertEqual([s["class"] for s in r["stale"]], ["MATERIALIZED-M1"])

    def test_background_write_lands_at_completion(self):
        tr = Transcript().read()
        tr.bash("sleep 5; sed -i 's/x = 1/x = 5/' a.py", "Command running in background with ID: b1. Output is being written to: /tmp/x",
                run_in_background=True)
        tr.edit("y = 2", "y = 20")
        tr.notify("<task-notification>\n<task-id>b1</task-id>\n<status>completed</status>\n</task-notification>")
        tr.edit("return y", "return y + 1")
        _, r = run(tr.write(self.path("bg.jsonl")))
        self.assertEqual(r["coverage"].get("background_completed"), 1)
        self.assertEqual([s["step"] for s in r["stale"]], [4])  # only the edit after the completion

    def test_background_write_without_completion_does_not_land_before_current_edit(self):
        tr = Transcript().read()
        tr.bash("sleep 5; sed -i 's/x = 1/x = 5/' a.py", "Command running in background with ID: b2.", run_in_background=True)
        tr.edit("y = 2", "y = 20")
        _, r = run(tr.write(self.path("bg2.jsonl")))
        self.assertEqual(r["stale_edits"], 0)
        self.assertEqual(r["coverage"].get("background_write_not_landed"), 1)

    def test_sessions_and_cluster_ci(self):
        main, pa, pb = self.subagent_session()
        rs = [run(p)[1] for p in (main, pa, pb)]
        s = summarize(rs)
        self.assertEqual((s["trajectories"], s["sessions"]), (3, 1))
        self.assertEqual(s["ci95"], list(s["ci95_binomial"]))  # one session: no clusters to resample
        many = [dict(r, file=self.path("q", "s%d.jsonl" % i), format="claudecode", stale_edits=i % 2, opportunities=2)
                for i, r in enumerate(rs * 10)]
        self.assertEqual(cluster_ci(many), cluster_ci(many))  # deterministic
        lo, hi = cluster_ci(many)
        self.assertTrue(lo <= 0.25 <= hi)
        # all-or-nothing sessions: resampling sessions gives a much wider interval than 300 independent edits
        clustered = [{"file": self.path("c", "s%d.jsonl" % i), "format": "claudecode", "trajectory": str(i),
                      "stale_edits": 10 * (i % 2), "opportunities": 10, "stale": []} for i in range(30)]
        s = summarize(clustered)
        self.assertEqual(s["ci95_method"], "bootstrap over sessions")
        self.assertGreater(s["ci95"][1] - s["ci95"][0], 2 * (s["ci95_binomial"][1] - s["ci95_binomial"][0]))

    def test_top_groups_subagents_with_their_session(self):
        from align_agent.cli import _where
        main, pa, pb = self.subagent_session()
        rs = [run(p)[1] for p in (main, pa, pb)]
        for root in (self.d, self.path("proj")):
            out = "\n".join(_where(rs, [root], 5))
            self.assertIn("proj", out.split("Folders")[1].split("Sessions")[0])
            sessions = out.split("Sessions with most stale edits")[1].splitlines()[1:]
            self.assertEqual(len(sessions), 1)
            self.assertIn("sess1.jsonl", sessions[0])

    def codex_failed_patch(self):
        lines = [{"type": "session_meta", "payload": {"id": "c1", "cwd": CWD}},
                 {"type": "turn_context", "payload": {"cwd": CWD, "model": "gpt"}}]

        def call(kind, i, body, out):
            name = "exec_command" if kind == "fc" else "apply_patch"
            if kind == "fc":
                lines.append({"type": "response_item", "payload": {"type": "function_call", "name": name, "arguments": json.dumps(body), "call_id": i}})
                lines.append({"type": "response_item", "payload": {"type": "function_call_output", "call_id": i, "output": out}})
            else:
                lines.append({"type": "response_item", "payload": {"type": "custom_tool_call", "name": name, "input": body, "call_id": i}})
                lines.append({"type": "response_item", "payload": {"type": "custom_tool_call_output", "call_id": i, "output": out}})
        call("fc", "c1", {"cmd": "cat a.py"}, "Exit code: 0\nOutput:\nx = 1\ny = 2\n")
        call("cp", "c2", "*** Begin Patch\n*** Update File: a.py\n@@\n-x = 1\n+x = 10\n*** End Patch", "Success. Updated the following files:\nM a.py")
        call("cp", "c3", "*** Begin Patch\n*** Update File: a.py\n@@\n-x = 1\n+x = 11\n*** End Patch", "Failed to find expected lines in a.py")
        p = self.path("cx.jsonl")
        with open(p, "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in lines)
        return run(p)[1]

    def test_codex_failed_patch_is_scored(self):
        r = self.codex_failed_patch()
        self.assertEqual(r["opportunities"], 2)
        self.assertEqual(r["stale_edits"], 1)

    def test_anchor_is_not_found_inside_a_longer_token(self):
        # the patch's `x = 1` was removed by the agent's own `x = 10` patch: `x = 1` is not in `x = 10`
        r = self.codex_failed_patch()
        self.assertEqual(r["stale"][0]["class"], "FAILED-OWN-WRITE")

    def test_windows_session_paths_are_made_relative(self):
        from align_agent.sessions import root_of
        steps = [{"ops": [{"path": p} for p in ("C:\\Users\\Me\\proj\\src\\a.py", "c:/users/me/proj/b.py",
                                                  "/c/Users/Me/proj/c.py", "src\\d.py", "D:\\other\\e.py")]}]
        self.assertEqual(root_of(steps, "C:\\Users\\Me\\proj"), "C:/Users/Me/proj")
        self.assertEqual([o["path"] for o in steps[0]["ops"]], ["src/a.py", "b.py", "c.py", "src/d.py", "D:/other/e.py"])
        posix = [{"ops": [{"path": "/home/u/p/a.py"}, {"path": "/home/u/px/a.py"}]}]
        root_of(posix, "/home/u/p/")
        self.assertEqual([o["path"] for o in posix[0]["ops"]], ["a.py", "/home/u/px/a.py"])


if __name__ == "__main__":
    unittest.main()
