"""Write-through engine tests (stdlib only): `python3 -m pytest tests` or `python3 tests/test_engine.py`.

Each test drives align_wt as the Claude Code plugin ships it, in a throwaway git workspace.
"""

import json
import ntpath
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "claude-plugin", "scripts")
sys.path.insert(0, SCRIPTS)

import align_wt.engine as E  # noqa: E402
from align_wt import claude_code, protocol, syntax  # noqa: E402
from align_wt.engine import Session  # noqa: E402
from wt_src.harness.lean import redundant_writes  # noqa: E402
from wt_src.harness.tail_diff import format_line_numbered_diff, opcodes, split_lines  # noqa: E402


def lines(n, pre="L"):
    return "".join(f"{pre}{i}\n" for i in range(1, n + 1))


class Workspace:
    def __init__(self, files, git=True):
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.realpath(os.path.join(self.tmp, "ws"))
        self.state = os.path.join(self.tmp, "state")
        os.makedirs(self.root)
        for p, text in files.items():
            self.write(p, text)
        if git:
            subprocess.run(["git", "-C", self.root, "init", "-q"], check=True)

    def write(self, p, text):
        full = os.path.join(self.root, p)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        old = os.stat(full).st_mtime_ns if os.path.exists(full) else 0
        with open(full, "w" if isinstance(text, str) else "wb", **({"newline": ""} if isinstance(text, str) else {})) as f:
            f.write(text)
        st = os.stat(full)
        os.utime(full, ns=(st.st_atime_ns, max(st.st_mtime_ns, old + 1000)))

    def session(self, **kw):
        return Session(self.root, "s", self.state, **kw)

    def start(self, **kw):
        with self.session(**kw) as s:
            s.start()

    def hook(self, payload):
        os.environ["ALIGN_STATE"] = self.state
        os.environ["CLAUDE_PROJECT_DIR"] = self.root
        payload.setdefault("session_id", "s")
        return claude_code.handle(payload)

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def denied(reply):
    return bool(reply) and reply["hookSpecificOutput"].get("permissionDecision") == "deny"


# ---------------------------------------------------------------- line numbers

def test_split_lines_counts_newlines_only():
    assert split_lines("a\n\fb\nc\n") == ["a", "\fb", "c"]
    assert split_lines("a\u2028b\n") == ["a\u2028b"]
    assert split_lines("a\nb", keepends=True) == ["a\n", "b"]
    assert split_lines("") == []


def test_form_feed_does_not_shift_numbers():
    old = "x1\n\f\n" + lines(10, "x")[6:]
    new = old.replace("x9\n", "CHANGED\n")
    diff = format_line_numbered_diff(old, new, "a.py")
    want = split_lines(new).index("CHANGED") + 1  # as `sed -n Np` counts
    assert f"{want:4d} + CHANGED" in diff


def test_opcodes_reconstruct_and_bounded():
    import random
    random.seed(3)
    for _ in range(200):
        a = [random.choice("abcde") for _ in range(random.randint(0, 40))]
        b = list(a)
        for _ in range(random.randint(0, 4)):
            k = random.randint(0, len(b))
            b[k:k + random.randint(0, 3)] = [random.choice("abxy") for _ in range(random.randint(0, 3))]
        out = []
        for tag, i1, i2, j1, j2 in opcodes(a, b):
            if tag == "equal":
                assert a[i1:i2] == b[j1:j2]
            out += b[j1:j2]
        assert out == b
    pool = ["    },", "    {", '      "version": "1.0.0",', '      "dev": true,', "      ]"]
    old = [pool[i * 7919 % 5] for i in range(60000)]
    new = list(old)
    for i in range(0, 60000, 200):
        new[i] = f'"resolved": "x{i}"'
    t = time.monotonic()
    E.hunks_and_shifts("\n".join(old), "\n".join(new))
    assert time.monotonic() - t < 10


def test_large_lockfile_change_is_fast():
    w = Workspace({"package-lock.json": "".join(f'  "pkg{i % 50}": {{"version": "1.0.0"}},\n' for i in range(30000))})
    try:
        w.start()
        with w.session() as s:
            s.read("package-lock.json")
        with open(os.path.join(w.root, "package-lock.json")) as f:
            w.write("package-lock.json", f.read().replace("1.0.0", "2.0.0", 40))
        t = time.monotonic()
        with w.session() as s:
            out = s.after("npm install", "")
        assert time.monotonic() - t < 10 and "package-lock.json" in out
    finally:
        w.close()


# ---------------------------------------------------------------- workspace scan

def test_scan_skips_home_and_bounds_walk():
    w = Workspace({}, git=False)
    try:
        old = os.environ.get("HOME")
        os.environ["HOME"] = w.root
        try:
            assert E.list_files(w.root) == []
        finally:
            if old is None:
                del os.environ["HOME"]
            else:
                os.environ["HOME"] = old
        for i in range(30):
            w.write(f"d{i}/f.txt", "x\n")
        keep = E.WALK_FILES
        E.WALK_FILES = 10
        try:
            assert len(E.list_files(w.root)) < 30
        finally:
            E.WALK_FILES = keep
    finally:
        w.close()


def test_build_dir_below_top_is_source():
    w = Workspace({"src/build/wheel.py": "x = 1\n", "build/out.txt": "o\n"}, git=False)
    try:
        files = E.list_files(w.root)
        assert "src/build/wheel.py" in files and "build/out.txt" not in files
    finally:
        w.close()


def test_gitignored_file_that_was_read_is_diffed():
    w = Workspace({".gitignore": "local.yaml\n", "local.yaml": lines(10)})
    try:
        w.start()
        with w.session() as s:
            s.after(read="local.yaml")
        w.write("local.yaml", lines(10).replace("L5\n", "CHANGED\n"))
        with w.session() as s:
            assert "CHANGED" in s.after("make", "")
    finally:
        w.close()


# ---------------------------------------------------------------- shell directory and paths

def test_guard_follows_payload_cwd_and_cd():
    w = Workspace({"a.py": lines(60, "root"), "sub/a.py": lines(60, "sub")})
    try:
        w.hook({"hook_event_name": "SessionStart", "source": "startup"})
        w.hook({"hook_event_name": "PostToolUse", "tool_name": "Read",
                "tool_input": {"file_path": f"{w.root}/sub/a.py", "offset": 1, "limit": 10}, "tool_response": {}})
        r = w.hook({"hook_event_name": "PreToolUse", "cwd": f"{w.root}/sub", "tool_name": "Bash",
                    "tool_input": {"command": "sed -i '5d' a.py"}})
        assert not denied(r)  # line 5 of sub/a.py was seen
        r = w.hook({"hook_event_name": "PreToolUse", "tool_name": "Bash",
                    "tool_input": {"command": "cd sub && sed -i '40d' a.py"}})
        assert denied(r) and "sub/a.py" in r["hookSpecificOutput"]["permissionDecisionReason"]
        r = w.hook({"hook_event_name": "PreToolUse", "tool_name": "Bash",
                    "tool_input": {"command": "(cd sub && true); sed -i '3d' a.py"}})
        assert not denied(r)  # a.py at the root was never seen: not a stale view
    finally:
        w.close()


def test_absolute_path_in_shell_counts_as_read():
    w = Workspace({"a.py": lines(10)})
    try:
        w.start()
        with w.session() as s:
            s.after(f"cat {w.root}/a.py", lines(10))
        w.write("a.py", lines(10).replace("L5\n", "X5\n"))
        with w.session() as s:
            assert "X5" in s.after("make", "")
    finally:
        w.close()


def test_rel_path_windows_emulated():
    saved = (E.os.path, E.os.sep, E.os.name)
    try:
        E.os.path, E.os.sep, E.os.name = ntpath, "\\", "nt"
        ntpath_realpath = ntpath.realpath
        ntpath.realpath = lambda p, **k: ntpath.normpath(p)
        try:
            assert E.rel_path("C:\\proj", "C:\\proj\\src\\a.py") == "src/a.py"
            assert E.rel_path("C:\\proj", "/c/proj/src/a.py") == "src/a.py"
            assert E.rel_path("C:\\proj", "D:\\other\\a.py") is None
        finally:
            ntpath.realpath = ntpath_realpath
    finally:
        E.os.path, E.os.sep, E.os.name = saved
    assert E.rel_path("/r", "..notes.txt") == "..notes.txt" and E.rel_path("/r", "../x") is None


# ---------------------------------------------------------------- what counts as seen

def test_read_without_range_shows_2000_lines():
    w = Workspace({"a.py": lines(3000)})
    try:
        w.hook({"hook_event_name": "SessionStart", "source": "startup"})
        w.hook({"hook_event_name": "PostToolUse", "tool_name": "Read", "tool_input": {"file_path": f"{w.root}/a.py"},
                "tool_response": {}})
        assert denied(w.hook({"hook_event_name": "PreToolUse", "tool_name": "Bash",
                              "tool_input": {"command": "sed -i '2500d' a.py"}}))
        assert not denied(w.hook({"hook_event_name": "PreToolUse", "tool_name": "Bash",
                                  "tool_input": {"command": "sed -i '1500d' a.py"}}))
    finally:
        w.close()


def test_shell_read_regions_per_command():
    w = Workspace({"a.py": lines(300), "b.py": lines(300), "lib/util.py": lines(50), "tests/util.py": lines(50)})
    try:
        r = dict(E.shell_reads(w.root, "head -n 200 a.py; tail -n 5 b.py", ""))
        assert r == {"a.py": [[1, 200]], "b.py": []}
        assert dict(E.shell_reads(w.root, "grep -n L2 a.py | sed -n 1,3p", "")) == {"a.py": []}
        assert E.shell_reads(w.root, "nl a.py", "")[0][1] == []
        assert E.shell_reads(w.root, "nl -ba a.py", "")[0][1] is None
        assert E.shell_reads(w.root, "sed -n '10,20p' a.py", "")[0][1] == [[10, 20]]
        out = "lib/util.py:3:L3\ntests/util.py:40:L40\n"
        assert dict(E.shell_reads(w.root, "grep -n L lib/util.py tests/util.py", out))["lib/util.py"] == [[3, 3]]
    finally:
        w.close()


def test_partial_read_still_reports_change_elsewhere():
    w = Workspace({"a.py": lines(300)})
    try:
        w.start()
        with w.session() as s:
            s.after(read="a.py")
        w.write("a.py", lines(300).replace("L250\n", "CHANGED250\n"))
        with w.session() as s:
            assert "CHANGED250" in s.after(read="a.py", read_range=(1, 20))
    finally:
        w.close()


def test_write_tool_marks_whole_file_seen():
    w = Workspace({})
    try:
        w.hook({"hook_event_name": "SessionStart", "source": "startup"})
        w.write("a.py", lines(20))
        w.hook({"hook_event_name": "PostToolUse", "tool_name": "Write",
                "tool_input": {"file_path": f"{w.root}/a.py", "content": lines(20)}, "tool_response": {}})
        assert not denied(w.hook({"hook_event_name": "PreToolUse", "tool_name": "Bash",
                                  "tool_input": {"command": "sed -i '5d' a.py"}}))
    finally:
        w.close()


def test_guard_ignores_redirect_targets():
    assert [f for f, *_ in E.sed_line_edits("sed -i '5d' a.py 2>/dev/null")] == ["a.py"]
    assert [f for f, *_ in E.sed_line_edits("sed -i '5d' a.py >/dev/null 2>&1")] == ["a.py"]


# ---------------------------------------------------------------- lean rule

def test_lean_only_skips_what_the_model_already_has():
    cases = {
        "cat > a.py <<'EOF'\nx=1\nEOF": {"a.py"},
        "cat > a.py <<'EOF'\nx=1\nEOF\nblack a.py": set(),
        'printf "x" > a.py && npx prettier --write a.py': set(),
        "echo hi | python gen.py > a.py": set(),
        "cat > a.py <<EOF\n$(sed -n 1p b.py)\nEOF": set(),
        "sed -i s/a/b/ a.py && cat a.py": {"a.py"},
        "sed -i s/a/b/ a.py && head -3 a.py": set(),
        "sed -i s/a/b/ a.py; cat a.py | head -2": set(),
        "sed -i s/a/b/ a.py && nl a.py": set(),
    }
    for command, want in cases.items():
        assert redundant_writes(command, "", ["a.py"]) == want, command


# ---------------------------------------------------------------- storage and adapters

def test_crlf_round_trips_on_windows_emulated():
    w = Workspace({"a.txt": "".join(f"line {i}\r\n" for i in range(1, 41))})
    real_open = open

    def windows_open(file, mode="r", *a, **kw):  # text mode translates \n to \r\n on write, as on Windows
        if "b" not in mode and "w" in mode and kw.get("newline") is None:
            kw["newline"] = "\r\n"
        return real_open(file, mode, *a, **kw)
    E.open = windows_open
    try:
        w.start()
        with w.session() as s:
            s.read("a.txt")
        w.write("a.txt", "".join(f"line {i}\r\n" for i in range(1, 41)).replace("line 20\r\n", "LINE TWENTY\r\n"))
        with w.session() as s:
            out = s.after("make", "")
        assert "  20 + LINE TWENTY" in out and len(out) < 400
    finally:
        del E.open
        w.close()


def test_session_id_cannot_leave_state_dir():
    assert E.safe_id("../../x") == ".._.._x" and E.safe_id("..") == "_.." and E.safe_id(None) == "default"


def test_secret_names():
    for name in (".git-credentials", "prod.tfvars", "kubeconfig", ".env.local"):
        assert E.is_secret(name), name
    assert not E.is_secret("prod.tfvars.example")


def test_syntax_note_names_old_interpreter():
    msg = syntax.broke("x.py", "a = 1\n", "a = (\n")
    assert msg and (sys.version_info >= (3, 12) or "checked with Python" in msg)


def test_hooks_read_utf8_stdin_under_any_locale():
    w = Workspace({"café.py": lines(5)})
    try:
        env = dict(os.environ, PYTHONIOENCODING="cp1252", ALIGN_STATE=w.state, CLAUDE_PROJECT_DIR=w.root)
        payload = {"hook_event_name": "PostToolUse", "session_id": "s", "tool_name": "Read",
                   "tool_input": {"file_path": f"{w.root}/café.py"}, "tool_response": {}}
        hook = os.path.join(SCRIPTS, "hook.py")
        subprocess.run([sys.executable, "-I", hook], input=json.dumps(payload, ensure_ascii=False).encode(), env=env, check=True)
        sessions = os.path.join(w.state, "sessions", "s", "ledger.json")
        with open(sessions, encoding="utf-8") as f:
            assert "café.py" in json.load(f)
    finally:
        w.close()


def test_protocol_fails_open():
    r = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {SCRIPTS!r}); "
                        "from align_wt.protocol import main; sys.exit(main())"],
                       input=b'{"event": "read"}', capture_output=True, env=dict(os.environ, ALIGN_STATE=tempfile.mkdtemp()))
    assert r.returncode == 0 and r.stdout == b""



# ---------------------------------------------------------------- guard addresses, literal writes, locking

def test_sed_addresses_beyond_n_and_n_m():
    w = Workspace({"a.py": lines(40)})
    try:
        assert [(lo, hi) for _, lo, hi, _ in E.sed_line_edits("sed -i '3,+2d;10~2d;0~4d;5{N;d};7,$d' a.py", w.root)] == \
            [(3, 5), (10, None), (4, None), (5, 6), (7, None)]
        w.start()
        with w.session() as s:
            s.read("a.py", 1, 10)
        with w.session() as s:
            assert s.before("sed -i '3,+2d' a.py") is None  # lines 3-5: seen
            assert s.before("sed -i '8,$d' a.py")  # to the end: lines 11-40 not seen
    finally:
        w.close()


def test_guard_sees_find_exec_xargs_and_gawk_options():
    w = Workspace({"a.py": lines(40)})
    try:
        edits = lambda c: [(f, lo, hi) for f, lo, hi, _ in E.sed_line_edits(c, w.root) + E.script_line_edits(c, w.root)]
        assert edits("find a.py -exec sed -i '30d' {} +") == [("a.py", 30, 30)]
        assert edits("find . -name a.py -exec sed -i '30d' {} \\;") == []  # which files: unknown
        assert edits("git ls-files | xargs sed -i '30d'") == []
        assert edits("xargs -I{} sed -i '30d' {} a.py < list") == [("a.py", 30, 30)]
        assert edits("gawk -i inplace -v n=1 'NR==30{next}1' a.py") == [("a.py", 30, 30)]
        assert edits("gawk -i inplace -f prog.awk a.py") == []
    finally:
        w.close()


def test_write_is_literal_only_for_text_typed_in_the_command():
    from wt_src.parsers.file_events import shell_file_events
    kinds = lambda c: [k for op, _, k in shell_file_events(c) if op == "W"]
    assert kinds("echo hi > a.py") == ["overwrite"]
    assert kinds("echo hi | python gen.py > a.py") == ["replace"]  # gen.py's output, not the echo
    assert kinds('echo "$(cat b.py)" > a.py') == ["replace"]
    assert kinds("cat > a.py <<EOF\nx = $y\nEOF") == ["replace"]  # the shell expands $y
    assert kinds("cat > a.py <<'EOF'\nx = $y\nEOF") == ["overwrite"]
    assert kinds("echo x | tee a.py") == ["overwrite"]


def test_failed_save_still_releases_the_lock():
    w = Workspace({"a.py": "x\n"})
    try:
        s = w.session()
        lock = s.state._lock
        try:
            with s:
                s.state.ledger = {"a.py": {"sha": object()}}  # not JSON: save fails
        except TypeError:
            pass
        assert lock.closed
        with w.session() as s2:  # not left locked
            s2.start()
    finally:
        w.close()


def test_windows_lock_wait_stays_inside_hook_timeout():
    import inspect
    src = inspect.getsource(E.lock_file)
    tries = int(re.search(r"range\((\d+)\)", src).group(1))
    assert tries * 0.1 <= 25


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if failed else 0)
