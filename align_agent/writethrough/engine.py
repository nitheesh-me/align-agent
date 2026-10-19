"""Agent-agnostic write-through engine: keep a coding agent's view of its files current.

An agent edits files through what it last saw of them, and its own edits (a `sed -i`, a
script, a formatter) make those views out of date without telling it. The engine keeps, per
session, what the model last saw of each file and which line numbers it has seen as they are
now. An adapter tells it three things, and appends what it returns to what the model sees:

    s.read(path, lo, hi)          the model was shown a file (or lines lo..hi of it)
    s.authored(path)              the model changed a file with its own edit tool
    s.before(command) -> refusal  a shell command is about to run (None: let it run)
    s.after(command, output)      any tool call finished: text to append (may be empty)

`after` returns the net diff of each file the model saw that changed since (with current line
numbers; text from src/harness/tail_diff.py), a note per stretch of lines that moved without
being shown, and the remaining uses in other files of any definition a change removed or
re-signatured (src/harness/dependency_notice.py). `before` refuses once a `sed -i` addressed by
line number whose lines the model has not seen numbered as they are now, and the refusal shows
them. Nothing is rewritten in what the model saw before, so the prompt cache is kept.

Adapters: claude_code (hooks), protocol (a JSON command for any hook system), mcp_server (an
MCP shell tool for any MCP client). State: <state_dir>/sessions/<session>/, see State.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time

try:
    import fcntl  # POSIX
except ImportError:  # Windows
    fcntl = None
    import msvcrt

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(HERE, "_lib") not in sys.path:
    sys.path.insert(0, os.path.join(HERE, "_lib"))

from wt_src.harness.dependency_notice import dependency_notice  # noqa: E402
from . import syntax  # noqa: E402
from wt_src.harness.lean import redundant_writes  # noqa: E402
from wt_src.metrics.replay import _live_events  # noqa: E402
from wt_src.harness.tail_diff import format_line_numbered_diff, line_number_note, opcodes, split_lines  # noqa: E402
from wt_src.parsers.file_events import _strip_prefixes, _units_with_cwd, command_units  # noqa: E402

BUDGET = 9000        # Claude Code caps hook context at 10,000 characters; others are similar
PER_FILE = 3000
MAX_BYTES = 2_000_000
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
TOP_SKIP_DIRS = {"dist", "build"}  # build output at the top only: src/pip/build/ is source
WALK_FILES = 20_000  # without git, the walk stops here (or after WALK_SECONDS) instead of outlasting the hook
WALK_SECONDS = 5.0
DIFF_SECONDS = 15.0  # files left once the blocks have taken this long are named, not diffed (hook timeout 30 s)
CONTEXT = 3          # context lines around a hunk, as in src/harness/tail_diff.py
SHIFTS = os.environ.get("ALIGN_SHIFTS", "1") != "0"
GUARD = os.environ.get("ALIGN_GUARD", "1") != "0"
KEEP_DAYS = float(os.environ.get("ALIGN_KEEP_DAYS", "7"))  # sessions idle longer are deleted
# files whose text is never stored or diffed: they may hold credentials
SECRET = re.compile(r"""(?ix)^(?:
    \.env(?:\..+)? | .+\.(?:pem|key|p12|pfx|jks|keystore|secret|kdbx) | id_(?:rsa|dsa|ecdsa|ed25519)(?:\..+)?
  | \.(?:netrc|npmrc|pypirc|pgpass|git-credentials) | credentials(?:\..+)? | secrets?(?:\..+)?
  | .+\.tfvars(?:\.json)? | kubeconfig(?:\..+)? )$""")
NOT_SECRET = re.compile(r"(?i)\.(?:example|sample|template|dist)$|\.(?:py|js|ts|go|rs|java|rb|md)$")


def is_secret(rel):
    """A file ALIGN never copies: keys, .env files, credential stores (by name)."""
    name = os.path.basename(rel)
    return bool(SECRET.match(name)) and not NOT_SECRET.search(name)


# ---------------------------------------------------------------- state

def default_state_dir():
    return (os.environ.get("ALIGN_STATE") or os.environ.get("CLAUDE_PLUGIN_DATA")
            or os.path.expanduser("~/.cache/align-agent"))


def lock_file(f):
    if fcntl:
        fcntl.flock(f, fcntl.LOCK_EX)
        return
    f.seek(0)
    for _ in range(200):  # msvcrt.locking does not wait; retry for 20 s, inside the hook's 30 s
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            time.sleep(0.1)
    raise OSError("ALIGN: session state is locked")


def unlock_file(f):
    if fcntl:
        fcntl.flock(f, fcntl.LOCK_UN)
        return
    f.seek(0)
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass


class State:
    """snapshot.json: path -> [size, mtime_ns] of each workspace file at the last call;
    ledger.json: path -> {"sha", "regions", "base"}: the version the model last saw, the line ranges it
    has seen numbered as they are now, and the version of its last whole-file view; written.json: files the model changed;
    echoed.json: files the host's own editor wrote last, whose later shell changes the host reports itself; blobs/<sha>: what it saw."""

    def __init__(self, session_id, state_dir=None):
        sessions = os.path.join(state_dir or default_state_dir(), "sessions")
        self.dir = os.path.join(sessions, safe_id(session_id))
        os.makedirs(os.path.join(self.dir, "blobs"), mode=0o700, exist_ok=True)
        prune_sessions(sessions, keep=self.dir)
        self._lock = open(os.path.join(self.dir, ".lock"), "a+")
        try:
            lock_file(self._lock)  # parallel tool calls run their hooks in parallel
        except BaseException:
            self._lock.close()  # still locked by another call: the adapter logs it and lets the tool run
            raise
        self.snapshot = self._load("snapshot.json", None)
        self.ledger = self._load("ledger.json", {})
        self.written = set(self._load("written.json", []))
        self.echoed = set(self._load("echoed.json", []))

    def _load(self, name, default):
        try:
            with open(os.path.join(self.dir, name)) as f:
                return json.load(f)
        except (OSError, ValueError):
            return default

    def save(self):
        for name, obj in (("snapshot.json", self.snapshot), ("ledger.json", self.ledger), ("written.json", sorted(self.written)), ("echoed.json", sorted(self.echoed))):
            tmp = os.path.join(self.dir, name + ".tmp")
            with open(tmp, "w") as f:
                json.dump(obj, f)
            os.replace(tmp, os.path.join(self.dir, name))
        self.drop_unused_blobs()

    def release(self):
        """Unlock and close the session lock; safe to call more than once."""
        if self._lock.closed:
            return
        try:
            unlock_file(self._lock)
        finally:
            self._lock.close()

    def drop_unused_blobs(self):
        """Delete stored file copies the ledger no longer points to (old versions, cleared views)."""
        used = {v for e in self.ledger.values() for v in (e.get("sha"), e.get("base")) if v}
        bdir = os.path.join(self.dir, "blobs")
        try:
            names = os.listdir(bdir)
        except OSError:
            return
        for name in names:
            if name not in used:
                try:
                    os.remove(os.path.join(bdir, name))
                except OSError:
                    pass

    def put_blob(self, text):
        sha = hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()
        path = os.path.join(self.dir, "blobs", sha)
        if not os.path.exists(path):
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # readable by the user only
            with open(fd, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:  # CRLF stays CRLF
                f.write(text)
        return sha

    def blob(self, sha):
        try:
            with open(os.path.join(self.dir, "blobs", sha), encoding="utf-8", errors="surrogateescape", newline="") as f:
                return f.read()
        except OSError:
            return None


def safe_id(session_id):
    """A session id usable as one directory name under sessions/ (no separators, not . or ..)."""
    sid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(session_id or "default"))
    return "_" + sid if sid in (".", "..") else sid


def prune_sessions(sessions, keep=None):
    """Delete session state (with its stored file copies) idle for more than KEEP_DAYS."""
    cutoff = time.time() - KEEP_DAYS * 86400
    try:
        names = os.listdir(sessions)
    except OSError:
        return
    for name in names:
        d = os.path.join(sessions, name)
        if d == keep or not os.path.isdir(d):
            continue
        try:
            last = max(os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d))
        except (OSError, ValueError):
            continue
        if last < cutoff:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- workspace

def list_files(root):
    """Workspace files, relative to root with "/" separators: git's view when it is a repository,
    else a bounded walk. The home directory or the filesystem root is not a workspace: nothing."""
    real = os.path.realpath(root)
    if real in (os.path.realpath(os.path.expanduser("~")), os.path.realpath(os.path.abspath(os.sep))):
        return []
    try:
        out = subprocess.run(["git", "-C", root, "ls-files", "-co", "--exclude-standard", "-z"],
                             capture_output=True, timeout=10)
        if out.returncode == 0:
            return [p for p in out.stdout.decode("utf-8", "surrogateescape").split("\0") if p]
    except (OSError, subprocess.SubprocessError):
        pass
    files, deadline = [], time.monotonic() + WALK_SECONDS
    for d, dirs, names in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not (d == root and x in TOP_SKIP_DIRS)]
        files += [os.path.relpath(os.path.join(d, n), root).replace(os.sep, "/") for n in names]
        if len(files) >= WALK_FILES or time.monotonic() > deadline:
            break
    return files


def scan(root, also=()):
    """path -> [size, mtime_ns] of the workspace files, and of `also` (files the model has seen,
    which may be ignored by git or past the walk's limit)."""
    snap = {}
    for rel in set(list_files(root)) | set(also):
        try:
            st = os.stat(os.path.join(root, rel))
        except OSError:
            continue
        snap[rel] = [st.st_size, st.st_mtime_ns]
    return snap


def read_text(root, rel):
    path = os.path.join(root, rel)
    try:
        if os.path.getsize(path) > MAX_BYTES:
            return None
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None  # binary
    return data.decode("utf-8", "surrogateescape")


def rel_path(root, path, base=None):
    """`path` relative to root, with "/" separators as git lists it, or None outside the root.
    A relative path is taken from `base` (the shell's directory, absolute or relative to the root)."""
    if not path:
        return None
    if os.name == "nt":
        m = re.match(r"^/([a-zA-Z])(?=/|$)", path)  # Git Bash: /c/Users/... is C:/Users/...
        if m:
            path = m.group(1) + ":" + (path[2:] or "/")
    full = os.path.realpath(path if os.path.isabs(path) else os.path.join(root, base or "", path))
    try:
        r = os.path.relpath(full, os.path.realpath(root))
    except ValueError:  # another drive
        return None
    if r == os.pardir or r.startswith(os.pardir + os.sep):  # not `..notes.txt`
        return None
    return r.replace(os.sep, "/")


# ---------------------------------------------------------------- line numbers

def merge(ranges):
    out = []
    for lo, hi in sorted((int(a), int(b)) for a, b in ranges if int(b) >= int(a)):
        if out and lo <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def covered(ranges, lo, hi):
    return all(any(a <= x <= b for a, b in ranges) for x in range(lo, hi + 1))


def remap(old, new, known):
    """Line ranges (1-based, in `new`'s numbering) the model still knows by number after old -> new:
    lines it knew whose number did not change."""
    keep = []
    for tag, i1, i2, j1, j2 in opcodes(split_lines(old), split_lines(new)):
        if tag == "equal" and i1 == j1:
            keep += [[max(a, i1 + 1), min(b, i2)] for a, b in known if max(a, i1 + 1) <= min(b, i2)]
    return merge(keep)


def hunks_and_shifts(old, new):
    """The live line ranges a diff block prints, and the moved stretches it does not print."""
    a, b = split_lines(old), split_lines(new)
    shown, shifts = [], []
    ops = opcodes(a, b)  # the same opcodes the printed diff uses
    for tag, i1, i2, j1, j2 in ops:
        if tag != "equal":
            shown.append([max(0, j1 - CONTEXT) + 1, min(len(b), j2 + CONTEXT)])
    shown = merge(shown)
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal" and j1 != i1 and not covered(shown, j1 + 1, j2):
            shifts.append((i1 + 1, i2, j1 + 1, j2, j1 - i1))
    return shown, shifts


# ---------------------------------------------------------------- blocks

def clip_block(text, limit):
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit("\n", 1)[0]
    more = text[len(cut):].count("\n")
    return cut + f"\n... {more} more changed lines not shown; re-read the file before a line-anchored edit."


def blocks_for(state, root, changed, skip, command="", output="", budget=None):
    """The write-through text for the seen files among `changed`, within the budget."""
    budget = budget or BUDGET
    seen = [p for p in sorted(changed) if p in state.ledger and p not in skip]
    if command and seen:
        try:
            seen = [p for p in seen if p not in redundant_writes(command, output, seen)]
        except Exception:  # noqa: BLE001  the lean rule is an optimisation; never fail the hook on it
            pass
    parts, before, shown = [], {}, {}
    used, deadline = 0, time.monotonic() + DIFF_SECONDS
    for p in seen:
        if time.monotonic() > deadline:
            parts.append(f"[TAIL DIFF: File '{p}' also changed; re-read it before a line-anchored edit]")
            shown[p] = []
            continue
        old = state.blob(state.ledger[p]["sha"])
        new = read_text(root, p)
        if old is None:
            continue
        if new is None:
            text = f"[TAIL DIFF: File '{p}' was deleted, or is no longer text, since your last view]"
        else:
            text = format_line_numbered_diff(old, new, p)
        if not text:
            continue
        before[p] = old
        room = min(PER_FILE, budget - used)
        if room < 200:
            parts.append(f"[TAIL DIFF: File '{p}' also changed; re-read it before a line-anchored edit]")
            shown[p] = []
            continue
        hunks, _ = hunks_and_shifts(old, new or "")
        if SHIFTS and new is not None:
            base = state.blob(state.ledger[p].get("base") or "") if state.ledger[p].get("base") else None
            note = line_number_note(p, base if base is not None else old, new)
            if note:
                text += "\n" + note
        clipped = clip_block(text, room)
        shown[p] = hunks if clipped == text else []  # a cut block shows only part of its hunks
        broken = None
        if new is not None:
            try:
                broken = syntax.broke(p, old, new)
            except Exception:  # noqa: BLE001  a check must never fail the hook
                broken = None
        if broken:
            clipped += f"\n[SYNTAX: '{p}' no longer parses ({broken}); it parsed before this call]"
        parts.append(clipped)
        used += len(clipped)
    if before:
        try:
            notice, _ = dependency_notice(root, list(before), before, viewed=set(state.ledger), written=state.written)
        except Exception:  # noqa: BLE001
            notice = ""
        if notice and used + len(notice) <= budget:
            parts.append(notice)
            used += len(notice)
    unseen = sorted(p for p in changed if p not in state.ledger and p not in skip)
    if unseen and parts:
        names = ", ".join(unseen[:8]) + (f" and {len(unseen) - 8} more" if len(unseen) > 8 else "")
        parts.append(f"[Also changed, not viewed: {names}]")
    return "\n\n".join(parts), shown


def line_count(text):
    return text.count("\n") + (0 if text.endswith("\n") or not text else 1)


def see(state, root, rel, regions=None, after_write=False):
    """Record that the model now knows `rel` as it is on disk.

    `regions` are line ranges shown with their current numbers (None: the whole file). After a
    write the model learned of (an Edit it made, a diff block), the lines it knew keep their
    numbers only where the write did not move them."""
    text = read_text(root, rel) if not is_secret(rel) else None
    if text is None:  # gone, binary, too large, or a secrets file: nothing stored, no diffs later
        state.ledger.pop(rel, None)
        return
    sha = state.put_blob(text)
    n = line_count(text)
    shown = [[1, max(n, 1)]] if regions is None else [list(r) for r in regions]
    old = state.ledger.get(rel)
    keep = []
    if old and old["sha"] == sha:
        keep = old.get("regions") or []
    elif old and after_write:
        prev = state.blob(old["sha"])
        if prev is not None:
            keep = remap(prev, text, old.get("regions") or [])
    # base: the version the model last saw whole; line-number notes count from it
    base = sha if (regions is None and not after_write) else ((old or {}).get("base") or (old or {}).get("sha"))
    state.ledger[rel] = {"sha": sha, "regions": merge(keep + shown), "base": base}


# ---------------------------------------------------------------- reads through the shell

ROOT_MARK = "_ALIGN_ROOT_"  # stands in for the workspace's absolute path while the replay parser reads a command
NUMBERED = re.compile(r"(?m)^(?:([^\n:]+):)?(\d+)[:-]")
SED_RANGE = re.compile(r"^(\d+)(?:\s*,\s*(\d+))?p$")


def placed_units(command, base=""):
    """(argv, unit, directory, piped on) for each simple command of `command` run from `base`:
    a `cd` moves the commands after it; the directory is None where it cannot be known."""
    units = command_units(command)
    placed, _ = _units_with_cwd(units, base)
    out = []
    for k, (u, d) in enumerate(placed):
        piped = any(v["pipe"] == u["pipe"] and v.get("depth", 0) == u.get("depth", 0) for v in units[k + 1:])
        out.append((_strip_prefixes(u["argv"]), u, d, piped))
    return out


def _head_lines(argv):
    """Lines `head` prints (-n N, -nN, -N, --lines=N); None for bytes (-c)."""
    for i, w in enumerate(argv[1:], 1):
        if w == "-n" and i + 1 < len(argv) and argv[i + 1].isdigit():
            return int(argv[i + 1])
        m = re.match(r"^(?:-n|--lines=|-)(\d+)$", w)
        if m:
            return int(m.group(1))
        if w.startswith(("-c", "--bytes")):
            return None
    return 10


def _regions(root, rel, argv, u, d, piped, output):
    """The line ranges a reader showed of `rel` with their numbers: None for the whole file, [] for
    content without known numbers."""
    tool = os.path.basename(argv[0])
    if piped or any(op in (">", ">>", ">|", "&>") for op, _ in u["redirs"]):
        return []  # the output went on to another command or a file
    if tool in ("cat", "bat", "batcat"):
        return None
    if tool == "nl":  # plain nl does not number blank lines
        return None if "-ba" in argv or ("-b" in argv and "a" in argv) else []
    if tool in ("sed", "gsed") and "-n" in argv:
        script = next((w for w in argv[1:] if not w.startswith("-")), "")
        m = SED_RANGE.match(script.strip())
        return [[int(m.group(1)), int(m.group(2) or m.group(1))]] if m else []
    if tool == "head":
        n = _head_lines(argv)
        return [[1, n]] if n else []
    if tool in ("grep", "egrep", "fgrep", "rg", "ag") and any(
            w == "--line-number" or re.match(r"^-[A-Za-z]*n[A-Za-z]*$", w) for w in argv[1:]):
        files = [w for w in argv[1:] if not w.startswith("-")][1:]  # the first word is the pattern
        nums = []
        for f, n in NUMBERED.findall(output):
            if (f and rel_path(root, f, d) == rel) or (not f and len(files) == 1):
                nums.append(int(n))
        return merge([[n, n] for n in nums])
    return []


def shell_reads(root, command, output, base=None):
    """(path, line ranges or None for the whole file) for each file a shell command printed.

    Which files: the replay's command parser (src/metrics/replay.py), with the workspace's absolute
    path spelled relative so it keeps them. Which lines: from the command that read the file, if
    its output reached the model directly: `cat`/`bat`/`nl -ba` the whole file, `sed -n 'A,Bp'`
    lines A-B, `head -n N` lines 1-N, `grep -n`/`rg -n` the numbered lines of that file in the
    output; anything else (tail, awk, a pipe, `git diff`) leaves the version seen but no line
    number known."""
    base = base or root
    spellings = sorted({root, os.path.realpath(root), root.replace(os.sep, "/")}, key=len, reverse=True)
    local = command
    for s in spellings:
        local = local.replace(s.rstrip("/\\") + "/", ROOT_MARK + "/").replace(s.rstrip("/\\") + os.sep, ROOT_MARK + "/")
    try:
        events = _live_events({"command": local, "observation": output, "writes_detected": []}, {})
    except Exception:  # noqa: BLE001
        return []
    try:
        units = placed_units(command, base)
    except Exception:  # noqa: BLE001
        units = []
    out = []
    for op, path, kind, landed in events:
        if op != "R" or not landed:
            continue
        if path.startswith(ROOT_MARK + "/"):
            rel = rel_path(root, os.path.join(root, path[len(ROOT_MARK) + 1:]))
        else:
            rel = rel_path(root, path, base)
        if not rel or not os.path.isfile(os.path.join(root, rel)):
            continue
        reader = next(((argv, u, d, piped) for argv, u, d, piped in units if argv and d is not None
                       and any(rel_path(root, w, d) == rel for w in argv[1:] if not w.startswith("-"))), None)
        out.append((rel, _regions(root, rel, *reader, output) if reader else []))
    return out


# ---------------------------------------------------------------- guard

# N, N,M  N,$  N,+M  N,~M  N~step, then the command (or a `{` block)
SED_LINE = re.compile(r"^\s*(\d+)\s*(?:,\s*(\d+|\$|\+\d+|~\d+)|~\s*(\d+))?\s*!?\s*([acdisyrR{])")
XARGS_VALUE = {"-I", "-i", "-n", "-L", "-l", "-P", "-d", "-E", "-e", "-s", "-a", "--max-args", "--max-procs",
               "--delimiter", "--arg-file", "--replace"}


def _sed_commands(script):
    """A sed script's commands, split at `;` and newlines outside `{ }` blocks."""
    out, cur, depth = [], "", 0
    for c in script:
        if c in ";\n" and depth == 0:
            out.append(cur)
            cur = ""
            continue
        depth += (c == "{") - (c == "}")
        cur += c
    return out + [cur]


def _sed_range(cmd):
    """(first line, last line or None for the end of the file) a sed command addresses by number."""
    m = SED_LINE.match(cmd)
    if not m:
        return None
    lo, to, step = int(m.group(1)), m.group(2), m.group(3)
    if step is not None:  # first~step: every step-th line from the first (0~4: line 4) to the end
        step = int(step)
        return (lo or step, None) if step else (lo, lo)
    if to is None:
        hi = lo
    elif to == "$":
        return lo, None
    elif to.startswith("+"):
        hi = lo + int(to[1:])
    elif to.startswith("~"):  # up to the next multiple of M
        k = int(to[1:])
        hi = -(-lo // k) * k if k else lo
    else:
        hi = max(lo, int(to))
    if m.group(4) == "{":  # each `N` in the block pulls in one more line
        hi += len(re.findall(r"(?:^|[;{\s])N(?=\s*(?:[;}]|$))", cmd[m.end():]))
    return lo, hi


def _unwrap(words, d):
    """The command `find ... -exec CMD {} +` or `xargs CMD` runs, as words, with `{}` replaced by
    the one file find names (when it names a file) and dropped otherwise: files from stdin are unknown."""
    tool = os.path.basename(words[0]) if words else ""
    target = None
    if tool == "find":
        k = next((i for i, w in enumerate(words) if w in ("-exec", "-execdir", "-ok", "-okdir")), None)
        if k is None:
            return words
        paths = [w for w in words[1:k] if not w.startswith(("-", "(", "!"))]
        paths = paths[:next((i for i, w in enumerate(words[1:k]) if w.startswith(("-", "(", "!"))), len(paths))]
        if len(paths) == 1 and d is not None and os.path.isfile(os.path.join(d, paths[0])):
            target = paths[0]
        words = words[k + 1:]
        words = words[:next((i for i, w in enumerate(words) if w in (";", "+")), len(words))]
    elif tool == "xargs":
        i = 1
        while i < len(words) and words[i].startswith("-"):
            i += 2 if words[i] in XARGS_VALUE else 1
        words = words[i:]
    else:
        return words
    return [target if w == "{}" else w for w in words if w != "{}" or target]


def sed_line_edits(command, base=""):
    """(file, first line, last line, directory) for each `sed -i` script command addressed by line number."""
    out = []
    for words, u, d, _ in placed_units(command, base):
        while words and (words[0] in ("sudo", "command") or "=" in words[0]):
            words = words[1:]
        words = _unwrap(words, d)
        if not words or os.path.basename(words[0]) not in ("sed", "gsed"):
            continue
        inplace, scripts, files, explicit, i = False, [], [], "-e" in words or "--expression" in words, 1
        while i < len(words):
            w = words[i]
            if w in ("-e", "--expression") and i + 1 < len(words):
                scripts.append(words[i + 1]); i += 2; continue
            if w.startswith("-i") or w.startswith("--in-place"):
                inplace = True
                if w == "-i" and i + 1 < len(words) and words[i + 1] == "":
                    i += 2  # BSD/macOS sed: -i '' (an empty backup suffix)
                    continue
            elif w.startswith("-") and len(w) > 1:
                pass
            elif not scripts and not explicit:
                scripts.append(w)
            else:
                files.append(w)
            i += 1
        if not inplace:
            continue
        for script in scripts:
            for cmd in _sed_commands(script):
                r = _sed_range(cmd)
                if r:
                    out += [(f, r[0], r[1], d) for f in files]
    return out


PERL_LINE = re.compile(r"\$\.\s*(==|>=|<=|>|<|!=)\s*(\d+)")
AWK_LINE = re.compile(r"\bNR\s*(==|>=|<=|>|<|!=)\s*(\d+)")
ED_CMD = re.compile(r"^(?:-s?c\s*)?(\d+)\s*(?:,\s*(\d+|\$)\s*)?[acdijmst](?![a-z])")


def _bounds(pairs):
    """Line range from `$. == 10`, `NR >= 10 && NR <= 14` style comparisons (None if unbounded)."""
    lo, hi = None, None
    for op, n in pairs:
        n = int(n)
        if op == "==":
            lo, hi = n if lo is None else min(lo, n), n if hi is None else max(hi, n)
        elif op in (">=", ">"):
            lo = n + (op == ">")
        elif op in ("<=", "<"):
            hi = n - (op == "<")
        elif op == "!=":  # everything but line n: the line is what the edit aims at
            lo, hi = n, n
    if lo is None and hi is not None:
        lo = 1
    if lo is not None and hi is None:
        hi = lo
    return (lo, hi) if lo is not None and hi >= lo else None


def script_line_edits(command, base=""):
    """(file, first line, last line, directory) for in-place edits addressed by line number through
    `perl -i` (`$.`), `awk -i inplace` (`NR`) and `ed`/`ex` scripts given inline."""
    out = []
    for words, u, d, _ in placed_units(command, base):
        while words and (words[0] in ("sudo", "command") or "=" in words[0]):
            words = words[1:]
        words = _unwrap(words, d)
        if not words:
            continue
        tool = os.path.basename(words[0])
        if tool == "perl" and any(re.match(r"^-[a-hj-zA-HJ-LN-Z]*i", w) for w in words[1:]):  # -i, -pi, -pie; not -M, -I
            code = " ".join(w for i, w in enumerate(words[1:], 1) if words[i - 1] in ("-e", "-pe", "-ne", "-lne", "-lpe")
                            or (words[i - 1].startswith("-") and words[i - 1].rstrip("i").endswith("e")))
            r = _bounds(PERL_LINE.findall(code))
            files = [w for w in words[1:] if not w.startswith("-") and w != code and w not in code]
        elif tool in ("awk", "gawk") and "inplace" in words:
            prog, files, i = None, [], 1
            while i < len(words):
                w = words[i]
                if w in ("-v", "-i", "-l", "-F", "--assign", "--include", "--load", "--field-separator"):
                    i += 2  # the option's value: a variable, `inplace`, a separator
                    continue
                if w in ("-f", "--file"):
                    prog, i = prog or "", i + 2  # the program is in a file: its lines are unknown
                    continue
                if w in ("-e", "--source") and i + 1 < len(words):
                    prog, i = (prog or "") + words[i + 1] + "\n", i + 2
                    continue
                if not w.startswith("-"):
                    if prog is None:
                        prog = w
                    elif not re.match(r"^[A-Za-z_]\w*=", w):  # `var=value` operands are assignments
                        files.append(w)
                i += 1
            r = _bounds(AWK_LINE.findall(prog or ""))
        elif tool in ("ed", "ex") or (tool in ("vim", "vi") and "-es" in words):
            # the script comes inline: printf '...' | ed f, ed f <<< $'...', ex -sc '...' f
            pieces = re.split(r"\\n|\n|\||;", command)
            pieces += [words[i + 1] for i, w in enumerate(words[:-1]) if w in ("-c", "-sc")]
            pieces += [w[1:] for w in words if w.startswith("+") and w[1:2].isdigit()]
            nums = []
            for piece in pieces:
                m = ED_CMD.match(re.split(r"['\"]", piece)[-1].strip())
                if m:
                    a = int(m.group(1))
                    nums.append((a, a if not m.group(2) or m.group(2) == "$" else int(m.group(2))))
            r = (min(a for a, _ in nums), max(b for _, b in nums)) if nums else None
            files = [w for i, w in enumerate(words[1:], 1) if not w.startswith(("-", "+", "<")) and "\\n" not in w
                     and "|" not in w and words[i - 1] not in ("<<", "<<<", "<<-")][-1:]
        else:
            continue
        if r:
            out += [(f, r[0], r[1], d) for f in files]
    return out


def guard(state, root, command, base=None):
    """A refusal for a line-numbered edit whose lines the model has not seen numbered as they are now."""
    base = base or root
    for f, lo, hi, d in sed_line_edits(command, base) + script_line_edits(command, base):
        if d is None:
            continue  # after a `cd` we cannot follow: the file is not known
        rel = rel_path(root, f, d)
        entry = state.ledger.get(rel) if rel else None
        if not entry:
            continue  # a file the model never saw: not a stale view
        text = read_text(root, rel)
        if text is None:
            continue
        sha = hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()
        known = entry.get("regions") or [] if entry["sha"] == sha else []
        lines = split_lines(text)
        hi = max(lo, len(lines)) if hi is None else hi  # `N,$`, `N~step`: to the end of the file
        if covered(known, lo, hi):
            continue
        a, b = max(1, lo - 5), min(len(lines), hi + 5, lo + 80)
        body = "\n".join(f"{k:6d}\t{lines[k - 1]}" for k in range(a, b + 1))
        see(state, root, rel, [[a, b]])  # the refusal shows them: a retry with these numbers passes
        why = (f"'{rel}' changed since you last saw lines {lo}-{hi} numbered" if entry["sha"] != sha
               else f"you have not seen lines {lo}-{hi} of '{rel}' with their line numbers")
        return (f"[STALE LINE NUMBERS] {why}, so they "
                f"may not be the lines you mean. Here they are as the file stands now (lines {a}-{b}):\n{body}\n"
                f"Reissue the command with these line numbers if this is what you meant to change.")
    return None


# ---------------------------------------------------------------- session API

class Session:
    """One agent session's write-through state. Use as a context manager: the state is locked
    while open (parallel tool calls run their hooks in parallel) and saved on exit."""

    def __init__(self, root, session_id="default", state_dir=None, budget=None, host_echoes=False, read_limit=None):
        self.root = os.path.realpath(root)
        # host_echoes: the host re-sends the changed lines of a file its own editor wrote last when
        # a shell command changes it (Claude Code does), so a block for that change would repeat it
        self.host_echoes = host_echoes
        # read_limit: lines the host's read tool shows when given no range (Claude Code: 2000)
        self.read_limit = read_limit
        self.state = State(session_id, state_dir)
        self.budget = budget or BUDGET  # characters an agent accepts per tool call
        if self.state.snapshot is None:
            # no session-start hook (Cline) or the first call after install: changes are
            # counted from now on
            try:
                self.state.snapshot = scan(self.root)
            except BaseException:
                self.state.release()
                raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.state.save()
        finally:
            self.state.release()  # a failed save must not leave the session locked
        return False

    def _base(self, cwd):
        """The shell's directory: `cwd` when the host reports it, else the workspace root."""
        return os.path.realpath(os.path.join(self.root, cwd)) if cwd else self.root

    def start(self, fresh=True):
        """A session (re)starts. fresh: the model holds none of its earlier views (a new session,
        a cleared or compacted context)."""
        if fresh or self.state.snapshot is None:
            self.state.ledger, self.state.written, self.state.echoed = {}, set(), set()
        self.state.snapshot = scan(self.root, self.state.ledger)

    def _range(self, rel, lo, hi):
        """The lines a read showed: [lo, hi], or None for the whole file. With no range, a host
        read tool that stops at read_limit lines shows only those of a longer file."""
        if lo is not None:
            return [[int(lo), int(hi if hi is not None else lo)]]
        if self.read_limit:
            text = read_text(self.root, rel)
            if text is not None and line_count(text) > self.read_limit:
                return [[1, self.read_limit]]
        return None

    def read(self, path, lo=None, hi=None):
        rel = rel_path(self.root, path)
        if rel:
            see(self.state, self.root, rel, self._range(rel, lo, hi))
        return rel

    def authored(self, path, whole=False):
        """The model changed `path` with its own edit tool; whole: it wrote all of it (a Write)."""
        rel = rel_path(self.root, path)
        if rel:
            if whole:
                see(self.state, self.root, rel)
            else:
                see(self.state, self.root, rel, [], after_write=True)
            self.state.written.add(rel)
            if self.host_echoes:
                self.state.echoed.add(rel)
        return rel

    def before(self, command, cwd=None):
        return guard(self.state, self.root, command, self._base(cwd)) if GUARD and command else None

    def after(self, command="", output="", read=None, read_range=None, authored=None, cwd=None, wrote=None):
        """Text to append after a tool call. `read` / `authored` name a file this very call showed
        or changed through the agent's own tools, so it is not reported back; `wrote`, a file it
        wrote whole. `cwd` is the shell's directory, for the command's relative paths."""
        base = self._base(cwd)
        now = scan(self.root, self.state.ledger)
        prev = self.state.snapshot if self.state.snapshot is not None else now
        changed = {p for p in set(now) | set(prev) if now.get(p) != prev.get(p)}
        changed = {p for p in changed if not (p in self.state.ledger and (t := read_text(self.root, p)) is not None
                                             and hashlib.sha256(t.encode("utf-8", "surrogateescape")).hexdigest()
                                             == self.state.ledger[p]["sha"])}  # same content: not a change
        skip, partial = set(), None
        if read:
            rel = rel_path(self.root, read)
            rng = self._range(rel, *(read_range or (None, None))) if rel else None
            if rng is None:
                skip.add(self.read(read))  # shown whole: nothing to add
            else:
                partial = (rel, rng)  # what changed outside the range is still reported, then it counts as seen
        for a in ([authored] if isinstance(authored, str) else authored or []):
            skip.add(self.authored(a))
        for a in ([wrote] if isinstance(wrote, str) else wrote or []):
            skip.add(self.authored(a, whole=True))
        reads = shell_reads(self.root, command, output, base) if command else []
        skip |= {p for p, regions in reads if regions is None}  # printed whole: nothing to add
        if command and self.host_echoes:
            for p in changed & self.state.echoed:  # the host reports this change; the shell is now the last writer
                see(self.state, self.root, p, [], after_write=True)
                skip.add(p)
            self.state.echoed -= changed
        text, shown = blocks_for(self.state, self.root, changed, skip, command or "", output if command else "", self.budget)
        for p, hunks in shown.items():
            see(self.state, self.root, p, hunks, after_write=True)
            self.state.written.add(p)
        if partial:
            see(self.state, self.root, *partial)
        for p, regions in reads:  # what the command printed is now what the model saw
            see(self.state, self.root, p, regions)
        self.state.snapshot = now
        return text
