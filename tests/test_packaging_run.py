"""Regression tests for the packaging, audit.sh, run and vendored-harness fixes.

Run from the repository root: python3 -m unittest discover -s tests
Standard library only; the mini-swe-agent tests are skipped when it is not installed.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "align_agent", "_vendor")
for p in (ROOT, VENDOR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.harness import dependency_notice as dn  # noqa: E402
from src.harness.tail_diff import tail_diff_blocks  # noqa: E402
from src.harness.viewport import snapshot_workspace  # noqa: E402
from src.parsers.file_events import shell_file_events, split_heredocs  # noqa: E402

COPIES = [os.path.join(ROOT, *p) for p in (("align_agent", "_vendor", "src"),
                                           ("align_agent", "writethrough", "_lib", "wt_src"),
                                           ("claude-plugin", "scripts", "align_wt", "_lib", "wt_src"))]


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def _git(ws, *args):
    subprocess.run(["git", "-C", ws, "-c", "user.email=t@t", "-c", "user.name=t", *args],
                   check=True, capture_output=True)


class Copies(unittest.TestCase):
    def test_shared_modules_identical(self):
        for rel in ("harness/tail_diff.py", "harness/viewport.py", "harness/dependency_notice.py",
                    "parsers/file_events.py"):
            texts = set()
            for c in COPIES:
                with open(os.path.join(c, rel)) as f:
                    texts.add(f.read())
            self.assertEqual(len(texts), 1, rel)


class DependencyNotice(unittest.TestCase):
    BEFORE = "def compute(a):\n    return a\n"

    def notice(self, after, git=True):
        with tempfile.TemporaryDirectory() as ws:
            _write(os.path.join(ws, "lib.py"), self.BEFORE)
            _write(os.path.join(ws, "use.py"), "from lib import compute\nprint(compute(1))\n")
            if git:
                _git(ws, "init", "-q")
                _git(ws, "add", ".")
                _git(ws, "commit", "-qm", "i")
            _write(os.path.join(ws, "lib.py"), after)
            return dn.dependency_notice(ws, ["lib.py"], {"lib.py": self.BEFORE})[0]

    def test_compatible_changes_not_reported(self):
        for after in ("def compute(a, verbose=False):\n    return a\n",
                      "def compute(*args, **kw):\n    return args[0]\n",
                      "import sys\nif sys.version_info >= (3, 8):\n    def compute(a):\n        return a\n"
                      "else:\n    def compute(a):\n        return a\n",
                      "try:\n    from fast import compute\nexcept ImportError:\n    def compute(a):\n        return a\n"):
            self.assertEqual(self.notice(after), "", after)

    def test_breaking_changes_reported(self):
        for after in ("def compute(a, b):\n    return a\n", "def calc(a):\n    return a\n",
                      "def f():\n    def compute(a):\n        return a\n"):
            self.assertIn("use.py:2", self.notice(after), after)

    def test_outside_git(self):
        self.assertIn("use.py:2", self.notice("def calc(a):\n    return a\n", git=False))

    def test_budget_spent_skips_git(self):
        dn._deadline = 0.0
        try:
            self.assertEqual(dn._git_grep(ROOT, ["-l", "-e", "x"]), [])
            self.assertIsNone(dn.committed(ROOT, "pyproject.toml"))
        finally:
            dn._deadline = float("inf")


class TailDiff(unittest.TestCase):
    def test_empty_held_file_is_a_version(self):
        from src.harness.freshness_hook import FreshnessHook
        with tempfile.TemporaryDirectory() as ws:
            _write(os.path.join(ws, "e.py"), "")
            h = FreshnessHook(ws)
            h.register_read(1, 1, "e.py", "returncode: 0\n<output>\n</output>")
            _write(os.path.join(ws, "e.py"), "x = 1\n")
            text = tail_diff_blocks(ws, ["e.py"], h, current_step=2)[0]
            self.assertIn("x = 1", text)
            self.assertNotIn("returncode", text)


class Viewport(unittest.TestCase):
    def test_nested_build_dir_is_source(self):
        with tempfile.TemporaryDirectory() as ws:
            for d in ("build", os.path.join("src", "pkg", "build"), os.path.join("src", "__pycache__")):
                os.makedirs(os.path.join(ws, d))
                _write(os.path.join(ws, d, "f.py"), "")
            snap = snapshot_workspace(ws, skip_dirs={"build", "__pycache__"})
            self.assertEqual(sorted(snap), [os.path.join("src", "pkg", "build", "f.py")])


class FileEvents(unittest.TestCase):
    def test_heredoc_operator_in_quotes_is_text(self):
        ev = shell_file_events("grep -rn '<<EOF' scripts/\nsed -i 's/a/b/' a.py\ncat a.py", cwd="")
        self.assertIn(("W", "a.py", "subst"), ev)
        ev = shell_file_events("python3 -c \"\nx = 1 << shift\nopen('a.py','w').write('y')\n\"", cwd="")
        self.assertTrue(any(e[:2] == ("W", "a.py") for e in ev))

    def test_heredoc_delimiter_exact(self):
        self.assertEqual(split_heredocs("cat > a <<EOF\nl\n  EOF\nEOF\nls")[1], ["l\n  EOF"])
        self.assertEqual(split_heredocs("cat > a <<-EOF\n\tx\n\tEOF\nls")[1], ["\tx"])

    def test_cp_target_directory(self):
        self.assertEqual(shell_file_events("cp -t dir a.py b.py", cwd=""),
                         [("W", "dir/a.py", "copy:a.py"), ("W", "dir/b.py", "copy:b.py")])

    def test_branch_switch_is_not_restore(self):
        for c in ("git checkout main", "git checkout -b feature"):
            self.assertEqual(shell_file_events(c, cwd=""), [("W", "*", "replace")], c)
        self.assertEqual(shell_file_events("git checkout -- a.py", cwd=""), [("W", "a.py", "restore")])


@unittest.skipUnless(os.name == "posix", "audit.sh runs under bash")
class AuditScript(unittest.TestCase):
    SCRIPT = os.path.join(ROOT, "claude-plugin", "scripts", "audit.sh")

    def run_with(self, stub):
        with tempfile.TemporaryDirectory() as tmp:
            home = os.path.join(tmp, "home with space")
            os.makedirs(os.path.join(home, ".claude", "projects"))
            bindir = os.path.join(tmp, "bin")
            os.makedirs(bindir)
            for c in ("bash", "sed", "head", "dirname", "tail", "tr", "mkdir", "basename", "pwd"):
                os.symlink(subprocess.run(["sh", "-c", f"command -v {c}"], capture_output=True, text=True).stdout.strip(),
                           os.path.join(bindir, c))
            with open(os.path.join(bindir, stub), "w") as f:
                f.write(f'#!/bin/sh\necho "{stub}: $*"\n')
            os.chmod(os.path.join(bindir, stub), 0o755)
            return subprocess.run(["bash", self.SCRIPT], env={"HOME": home, "PATH": bindir},
                                  capture_output=True, text=True).stdout

    def test_pinned_to_plugin_version(self):
        with open(os.path.join(ROOT, "claude-plugin", ".claude-plugin", "plugin.json")) as f:
            ver = json.load(f)["version"]
        self.assertIn(f"uvx: --quiet align-agent@{ver} analyze", self.run_with("uvx"))
        self.assertIn(f"pipx: run --spec align-agent=={ver} align-agent analyze", self.run_with("pipx"))

    def test_cache_dir_created_and_home_with_space(self):
        out = self.run_with("align-agent")
        self.assertIn("align-agent: analyze", out)
        self.assertIn("home with space/.claude/projects", out)
        self.assertNotIn("failed", out)


class Run(unittest.TestCase):
    def ns(self, **kw):
        base = dict(image=None, no_sandbox=False, api_key_env="ALIGN_TEST_KEY")
        base.update(kw)
        return argparse.Namespace(**base)

    @unittest.skipIf(sys.platform == "darwin", "sandbox-exec exists on macOS")
    def test_refuses_local_sandbox_off_macos_and_scrubs_key(self):
        from align_agent.run import cmd_run
        os.environ["ALIGN_TEST_KEY"] = "secret"
        self.assertEqual(cmd_run(self.ns()), 2)
        self.assertNotIn("ALIGN_TEST_KEY", os.environ)

    @unittest.skipUnless(importlib.util.find_spec("minisweagent"), "needs mini-swe-agent")
    def test_sandbox_profile_outside_workspace(self):
        from src.harness.mini_real import write_sandbox_profile
        with tempfile.TemporaryDirectory() as ws:
            path = write_sandbox_profile(ws)
            self.assertFalse(os.path.realpath(path).startswith(os.path.realpath(ws)))
            self.assertFalse(os.access(path, os.W_OK) and os.stat(path).st_mode & 0o222)
            with open(path) as f:
                self.assertIn(f'(deny file-write* (subpath "{os.path.dirname(os.path.realpath(path))}"))', f.read())


if __name__ == "__main__":
    unittest.main()
