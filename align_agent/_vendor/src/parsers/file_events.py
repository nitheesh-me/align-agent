"""Ordered file events in a shell command, for the stale-act replay.

The stale-act rate asks one question of every edit: when the agent issued it,
had the file changed since the agent last saw it? Answering that needs three
things the older extractors did not give. Order: `cat f; sed -i ... f` and
`sed -i ... f; cat f` differ in exactly the way that matters. What was *shown*:
`cp a b` names a without showing it, `grep -l` prints file names and no lines,
and neither is a view. What *kind* of write: an edit located by existing text
or line numbers depends on the view it was composed against, while a heredoc
that rewrites the whole file does not, and a `git checkout` restores content
the agent never typed.

An event is (op, path, kind):
    op    "R"  content of `path` was printed to the agent
          "W"  `path` was changed
    kind  for R: "view", or "echo" for the lines an editor prints back
                 after its own edit
          for W: "anchored"   located by existing text or lines (sed -i,
                              perl -i, awk inplace, patch, a script that
                              opens the file for writing)
                 "overwrite"  whole content replaced by text the command
                              itself carries (heredoc, echo, printf)
                 "replace"    whole content replaced by text the agent does
                              not see (`cmd > f`, cp of several files)
                 "copy:SRC"   whole content replaced by SRC's (cp/mv/install
                              of one file onto another)
                 "append", "delete"
                 "restore"    git checkout/restore/stash/reset; path "*"
                              when the command names no path
"""

import os
import re
import shlex
from typing import Dict, List, Optional, Tuple

FileEvent = Tuple[str, str, str]

ALL_PATHS = "*"
# Workspace mount points agents write absolute paths under; stripping one of
# these gives the workspace-relative path. Shared by the live harness and the
# replay so the two canonicalize identically.
CONTAINER_ROOTS = ("/testbed/", "/app/", "/workspace/", "/root/repo/", "/home/")

_HEREDOC_OP_RE = re.compile(r"""<<(-?)(?!<)\s*(['"]?)([A-Za-z_][A-Za-z0-9_]*)\2""")
_SEPARATORS = {";", ";;", "&&", "||", "&", "(", ")"}
_SEPARATOR_CHARS = set("();|&\n")
_REDIRECT_CHARS = set("<>&")

_VIEW_VERBS = {"cat", "head", "tail", "nl", "less", "more", "bat", "view"}
_GREP_VERBS = {"grep", "rg", "egrep", "fgrep", "ack", "ag"}
# Flags under which grep prints names or counts and never a line of content.
_GREP_NO_CONTENT = {"-l", "-L", "-c", "-q", "--files-with-matches",
                    "--files-without-match", "--count", "--quiet", "--silent"}
_INTERPRETERS = ("python", "perl", "ruby", "node")
_LITERAL_SOURCES = {"echo", "printf"}
_WRAPPERS = {"sudo", "command", "exec", "nohup", "time"}

_SED_SCRIPT_RE = re.compile(r"^[0-9,~$+]*\s*[a-zA-Z]([/,;|!].*)?$|^[/$^].*[/]|^[0-9,$]+[a-zA-Z]")
_DIFF_TARGET_RE = re.compile(r"^\+\+\+\s+(?:b/)?(\S+)", re.M)
_PY_OPEN_LIT = re.compile(r"""open\(\s*[rbfu]?(['"])(?P<p>[^'"\n]+)\1\s*,\s*(?:mode\s*=\s*)?[rbfu]?(['"])(?P<m>[^'"\n]*)\3""")
_PY_ASSIGN = re.compile(r"""(?P<v>\b[A-Za-z_]\w*)\s*=\s*(?:(?:pathlib\.)?Path\(\s*)?[rbfu]?(['"])(?P<p>[^'"\n]+)\2""")
_PY_OPEN_VAR = re.compile(r"""open\(\s*(?P<v>[A-Za-z_]\w*)\s*,\s*(?:mode\s*=\s*)?[rbfu]?(['"])(?P<m>[^'"\n]*)\2""")
_PY_PATH_LIT_WRITE = re.compile(r"""Path\(\s*[rbfu]?(['"])(?P<p>[^'"\n]+)\1\s*\)\s*\.write_(?:text|bytes)\(""")
_PY_VAR_WRITE = re.compile(r"""(?P<v>\b[A-Za-z_]\w*)\s*\.write_(?:text|bytes)\(""")


# Files conventionally named without an extension; any other bare word is
# taken not to be a path, so these must be listed.
COPY_PREFIX = "copy:"


def copy_source(kind: str) -> Optional[str]:
    """The source path a "copy:SRC" write carries, or None."""
    return kind[len(COPY_PREFIX):] if kind.startswith(COPY_PREFIX) else None


def map_kind_path(kind: str, f) -> str:
    """`kind` with the source path of a copy rewritten by `f`, as the
    target's path is, so the two stay in the same spelling."""
    src = copy_source(kind)
    return kind if src is None else COPY_PREFIX + f(src)


_KNOWN_NAMES = {
    "Makefile", "makefile", "GNUmakefile", "Dockerfile", "Containerfile", "configure", "Rakefile",
    "Gemfile", "Procfile", "Vagrantfile", "Jenkinsfile", "Pipfile", "Justfile", "Brewfile",
    "SConstruct", "SConscript", "Snakefile", "BUILD", "WORKSPACE", "Kconfig", "Kbuild",
    "LICENSE", "COPYING", "README", "CHANGELOG", "NOTICE", "AUTHORS", "INSTALL",
}
_DOTFILE = re.compile(r"\.[A-Za-z][\w.-]*")


def plausible_path(tok: str) -> bool:
    """Whether a shell word can be a file path rather than an expression."""
    if not tok or tok.startswith("-"):
        return False
    if tok in ("/dev/null", "/dev/stdout", "/dev/stderr", "null", ".", ".."):
        return False
    if tok.startswith("/dev/") or tok.startswith("/proc/"):
        return False
    if any(ch in tok for ch in "{}$*?!()<>\"'`=\n"):
        return False
    if tok.isdigit():
        return False
    base = tok.rsplit("/", 1)[-1]
    return ("/" in tok or bool(re.search(r"\.[A-Za-z0-9_]{1,6}$", tok)) or base in _KNOWN_NAMES
            or base.split(".", 1)[0] in _KNOWN_NAMES or bool(_DOTFILE.fullmatch(base)))


def clean(tok: str) -> str:
    tok = tok.strip("'\"` ,;")
    # /proc/self/cwd/ is the process's working directory.
    if tok.startswith("/proc/self/cwd/"):
        tok = tok[len("/proc/self/cwd/"):]
    return tok[2:] if tok.startswith("./") else tok


def split_heredocs(cmd: str) -> Tuple[str, List[str]]:
    """Shell text with heredoc bodies removed, and the bodies in the order
    their `<<` operators appear. Every heredoc is removed, not only the first:
    lines of a second body otherwise read as commands."""
    lines = (cmd or "").split("\n")
    out, bodies, i = [], [], 0
    quote = ""  # the quote open at the end of the previous line: `python -c "..."` spans lines
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        ops, quote = _heredoc_ops(line, quote)
        for strip_tabs, delim in ops:
            body = []
            # bash ends a body only at a line that is exactly the delimiter (`<<-`: after tabs)
            while i < len(lines) and (lines[i].lstrip("\t") if strip_tabs else lines[i]).rstrip("\r") != delim:
                body.append(lines[i])
                i += 1
            i += 1
            bodies.append("\n".join(body))
    return "\n".join(out), bodies


def _heredoc_ops(line: str, quote: str) -> Tuple[List[Tuple[bool, str]], str]:
    """The `<<` / `<<-` operators of one line that are outside quotes, as
    (strips tabs, delimiter), and the quote still open at its end. A `<<`
    inside a grep pattern or a python -c string is text, not a heredoc."""
    ops, k = [], 0
    while k < len(line):
        c = line[k]
        if quote == "'":
            quote = "" if c == "'" else quote
        elif c == "\\":
            k += 1
        elif quote == '"':
            quote = "" if c == '"' else quote
        elif c in "'\"":
            quote = c
        elif c == "#" and (k == 0 or line[k - 1] in " \t;&|("):
            break  # a comment
        elif line.startswith("<<<", k):
            k += 2
        elif line.startswith("<<", k):
            m = _HEREDOC_OP_RE.match(line, k)
            if m:
                ops.append((m.group(1) == "-", m.group(3)))
                k = m.end() - 1
        k += 1
    return ops, quote


def command_units(cmd: str) -> List[Dict]:
    """Simple commands in order, each with its argv, its redirections, its
    heredoc body, and the pipeline it belongs to.

    Tokenising respects quoting, so a `>` inside a grep pattern is part of
    the pattern and not a redirection; newlines separate commands as `;`
    does."""
    shell, bodies = split_heredocs(cmd)
    # A backslash-newline continues a line; join before tokenising.
    shell = shell.replace("\\\n", " ")
    try:
        lexer = shlex.shlex(shell, posix=True, punctuation_chars="();<>|&\n")
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return []

    units: List[Dict] = []
    pipe = 0
    depth = 0
    body_i = 0
    pending: Optional[str] = None

    def new() -> Dict:
        return {"argv": [], "redirs": [], "heredoc": None, "pipe": pipe, "depth": depth}

    cur = new()
    for t in tokens:
        if pending is not None:
            if pending.startswith("<<") and pending != "<<<":
                cur["heredoc"] = bodies[body_i] if body_i < len(bodies) else ""
                body_i += 1
            else:
                cur["redirs"].append((pending, t))
            pending = None
            continue
        if t == "|":
            if cur["argv"] or cur["redirs"]:
                units.append(cur)
            cur = new()
        elif t in _SEPARATORS or (t and set(t) <= _SEPARATOR_CHARS):
            # Punctuation arrives grouped (`);`, `))`), so any all-separator
            # token separates; its parentheses open and close subshells.
            if cur["argv"] or cur["redirs"]:
                units.append(cur)
            pipe += 1
            depth = max(0, depth + t.count("(") - t.count(")"))
            cur = new()
        elif set(t) <= _REDIRECT_CHARS and ("<" in t or ">" in t):
            # `2>` and `2>&1` arrive as the word "2" and then the operator; a
            # lone digit right before a redirect is its file descriptor.
            if cur["argv"] and len(cur["argv"][-1]) == 1 and cur["argv"][-1].isdigit():
                t = cur["argv"].pop() + t
            pending = t
        else:
            cur["argv"].append(t)
    if cur["argv"] or cur["redirs"]:
        units.append(cur)
    return units


def script_writes(script: str) -> List[str]:
    """Files a Python-style script opens for writing, by literal or by a
    variable bound to a literal."""
    if not script:
        return []
    names: Dict[str, str] = {}
    for m in _PY_ASSIGN.finditer(script):
        names[m.group("v")] = m.group("p")
    out: List[str] = []
    for m in _PY_OPEN_LIT.finditer(script):
        if any(c in m.group("m") for c in "wa+"):
            out.append(m.group("p"))
    for m in _PY_OPEN_VAR.finditer(script):
        if any(c in m.group("m") for c in "wa+") and m.group("v") in names:
            out.append(names[m.group("v")])
    for m in _PY_PATH_LIT_WRITE.finditer(script):
        out.append(m.group("p"))
    for m in _PY_VAR_WRITE.finditer(script):
        if m.group("v") in names:
            out.append(names[m.group("v")])
    return [clean(p) for p in dict.fromkeys(out) if plausible_path(clean(p))]


_PY_READ_LIT = re.compile(r"""open\(\s*[rbfu]?(['"])(?P<p>[^'"\n]+)\1\s*(?:\)|,\s*(?:mode\s*=\s*)?[rbfu]?(['"])(?P<m>[^'"\n]*)\3)""")
_PY_READ_VAR = re.compile(r"""open\(\s*(?P<v>[A-Za-z_]\w*)\s*(?:\)|,\s*(?:mode\s*=\s*)?[rbfu]?(['"])(?P<m>[^'"\n]*)\2)""")
_PY_PATH_LIT_READ = re.compile(r"""Path\(\s*[rbfu]?(['"])(?P<p>[^'"\n]+)\1\s*\)\s*\.read_(?:text|bytes)\(""")
_PY_VAR_READ = re.compile(r"""(?P<v>\b[A-Za-z_]\w*)\s*\.read_(?:text|bytes)\(""")


def script_reads(script: str) -> List[str]:
    """Files a Python-style script opens for reading, by literal or by a
    variable bound to a literal."""
    if not script:
        return []
    names = {m.group("v"): m.group("p") for m in _PY_ASSIGN.finditer(script)}
    out: List[str] = []
    for m in _PY_READ_LIT.finditer(script):
        if not m.group("m") or ("r" in m.group("m") or "+" in m.group("m")):
            out.append(m.group("p"))
    for m in _PY_READ_VAR.finditer(script):
        if (not m.group("m") or "r" in m.group("m") or "+" in m.group("m")) and m.group("v") in names:
            out.append(names[m.group("v")])
    out += [m.group("p") for m in _PY_PATH_LIT_READ.finditer(script)]
    out += [names[m.group("v")] for m in _PY_VAR_READ.finditer(script) if m.group("v") in names]
    return [clean(p) for p in dict.fromkeys(out) if plausible_path(clean(p))]


def script_write_events(script: str) -> List[Tuple[str, str]]:
    """(path, kind) for each file a script writes: an anchored edit when the
    script also reads that file, a replacement when it only writes it."""
    read = set(script_reads(script))
    return [(p, "anchored" if p in read else "replace") for p in script_writes(script)]


def patch_targets(body: Optional[str]) -> List[str]:
    if not body:
        return []
    return [p for p in dict.fromkeys(_DIFF_TARGET_RE.findall(body))
            if p != "/dev/null" and plausible_path(p)]


def _operands_after_scripts(argv: List[str], script_flags: Tuple[str, ...]) -> Tuple[List[str], bool]:
    """Operands of argv, dropping the argument of each script flag (-e X).
    The second value says whether a script flag was present."""
    ops, skip, seen = [], False, False
    for a in argv[1:]:
        if skip:
            skip = False
            continue
        if a in script_flags:
            skip, seen = True, True
            continue
        if a.startswith("-"):
            continue
        ops.append(a)
    return ops, seen


Rw = Tuple[List[str], List[Tuple[str, str]]]


def _is_short_flag_with(f: str, letters: str) -> bool:
    return len(f) > 1 and f[0] == "-" and f[1] != "-" and bool(set(f[1:]) & set(letters))


def _view(argv, flags, ops, u, literal) -> Rw:
    return ops + [t for op, t in u["redirs"] if op == "<"], []


# One `s` command, optionally line-addressed: `s/a/b/g`, `12s|x|y|`, `$s/a/b/`.
# A backslash is matched only as the start of an escape pair: with `.` also
# allowed to match it, a script full of `\t\n` that fails to parse backtracked
# exponentially (minutes on one vuls-8659668 Qwen step).
_SUBST_CMD = re.compile(r"""^\s*(?:\d+(?:,\d+)?|\$)?\s*s(?P<d>[^\w\s\\])(?:\\.|(?!(?P=d))[^\\])*(?P=d)"""
                        r"""(?:\\.|(?!(?P=d))[^\\])*(?P=d)[gIip0-9]*\s*$""", re.S)


def is_substitution(script: str) -> bool:
    """Whether a sed/perl script only substitutes: every command is `s///`."""
    parts = [x for x in re.split(r"[;\n]", script or "") if x.strip()]
    return bool(parts) and all(_SUBST_CMD.match(x) for x in parts)


def _sed_scripts(argv: List[str], files: List[str], had_e: bool) -> List[str]:
    out, i = [], 1
    while i < len(argv):
        if argv[i] in ("-e", "--expression") and i + 1 < len(argv):
            out.append(argv[i + 1]); i += 2
        elif argv[i] in ("-f",):
            return []  # a script file: contents unknown
        else:
            i += 1
    return out if had_e else files[:1]


def _sed(argv, flags, ops, u, literal) -> Rw:
    inplace = any(f.startswith("--in-place") or _is_short_flag_with(f, "i") for f in flags)
    files, had_e = _operands_after_scripts(argv, ("-e", "-f", "--expression"))
    scripts = _sed_scripts(argv, files, had_e)
    if not had_e and files and (_SED_SCRIPT_RE.match(files[0]) or not plausible_path(files[0])):
        files = files[1:]
    # "subst" is an anchored edit whose effect the agent chose exactly; it
    # scores as "anchored" unless replay(known_substitutions=True).
    kind = "subst" if scripts and all(is_substitution(x) for x in scripts) else "anchored"
    return ([], [(f, kind) for f in files]) if inplace else (files, [])


_GREP_ARG_FLAGS = ("-e", "-f", "--regexp", "-A", "-B", "-C", "-m", "-D", "-d",
                   "--after-context", "--before-context", "--context",
                   "--max-count", "--devices", "--directories")


def _grep(argv, flags, ops, u, literal) -> Rw:
    if any(f in _GREP_NO_CONTENT or _is_short_flag_with(f, "lLcq") for f in flags):
        return [], []
    files, _ = _operands_after_scripts(argv, _GREP_ARG_FLAGS)
    had_e = any(f in ("-e", "-f", "--regexp") for f in flags)
    return (files if had_e else files[1:]), []


def _awk(argv, flags, ops, u, literal) -> Rw:
    inplace = "inplace" in argv
    files, _ = _operands_after_scripts(argv, ("-f", "-v", "-i"))
    had_f = "-f" in argv
    if not had_f and files:
        files = files[1:]
    return ([], [(f, "anchored") for f in files]) if inplace else (files, [])


def _perl_files(argv: List[str]) -> List[str]:
    """Operands of a perl command line, skipping the -e script. The script
    follows any short-flag cluster ending in e or E (-e, -pe, -0777pe)."""
    files, skip = [], False
    for a in argv[1:]:
        if skip:
            skip = False
        elif a.startswith("-") and not a.startswith("--"):
            skip = a[-1] in "eE"
        else:
            files.append(a)
    return files


def _perl(argv, flags, ops, u, literal) -> Rw:
    files = _perl_files(argv)
    if any(f.startswith("-i") or _is_short_flag_with(f, "i") for f in flags):
        exprs = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "-e" or (a.startswith("-") and a.endswith("e") and "e" in a[1:])]
        kind = "subst" if exprs and all(is_substitution(x) for x in exprs) else "anchored"
        return [], [(f, kind) for f in files]
    if any(_is_short_flag_with(f, "pn") for f in flags):
        # `perl -pe` / `-ne` prints the files it reads: a view, not a write.
        return files, []
    return _interpreter(argv, flags, ops, u, literal)


def _copy(argv, flags, ops, u, literal) -> Rw:
    targets = [o for o in ops if plausible_path(clean(o))]
    into = next((argv[k + 1] for k, a in enumerate(argv[:-1]) if a in ("-t", "--target-directory")), None) \
        or next((a.split("=", 1)[1] for a in argv if a.startswith("--target-directory=")), None)
    if into:  # cp -t DIR SRC...: each source lands in DIR under its own name
        srcs = [t for t in targets if t != into]
        kind = lambda s: f"{COPY_PREFIX}{clean(s)}"
        writes = [(f"{into.rstrip('/')}/{clean(s).split('/')[-1]}", kind(s)) for s in srcs]
        if argv[0].split("/")[-1] == "mv":
            writes += [(s, "delete") for s in srcs]
        return [], writes
    if len(targets) < 2:
        return [], []
    # One source onto one target carries the source's content; several go
    # into a directory, and no target has a single source.
    kind = f"{COPY_PREFIX}{clean(targets[0])}" if len(targets) == 2 else "replace"
    writes = [(targets[-1], kind)]
    if argv[0].split("/")[-1] == "mv":
        writes += [(t, "delete") for t in targets[:-1]]
    return [], writes


def _rm(argv, flags, ops, u, literal) -> Rw:
    return [], [(o, "delete") for o in ops]


def _tee(argv, flags, ops, u, literal) -> Rw:
    if "-a" in flags or "--append" in flags:
        kind = "append"
    else:
        kind = "overwrite" if literal else "replace"
    return [], [(o, kind) for o in ops]


def _truncate(argv, flags, ops, u, literal) -> Rw:
    return [], [(o, "replace") for o in ops]


def _patch(argv, flags, ops, u, literal) -> Rw:
    named = [o for o in ops if plausible_path(clean(o)) and not o.endswith((".patch", ".diff"))]
    return [], [(p, "anchored") for p in patch_targets(u["heredoc"]) + named]


def _git(argv, flags, ops, u, literal) -> Rw:
    if not ops:
        return [], []
    sub, rest = ops[0], ops[1:]
    after_dd = argv[argv.index("--") + 1:] if "--" in argv else None
    if sub == "apply":
        return [], [(p, "anchored") for p in patch_targets(u["heredoc"])]
    if sub == "checkout" and after_dd is None and (
            any(f in argv for f in ("-b", "-B", "--orphan", "-")) or not any(plausible_path(clean(r)) or r == "." for r in rest)):
        # `git checkout -b feature`, `git checkout main`: a branch switch. Files it changes
        # show up in the snapshot diff; it is not a restore of everything written.
        return [], [(ALL_PATHS, "replace")]
    if sub in ("checkout", "restore"):
        named = after_dd if after_dd is not None else [r for r in rest if plausible_path(clean(r)) or r == "."]
        if "." in named or not named:
            return [], [(ALL_PATHS, "restore")]
        return [], [(p, "restore") for p in named]
    if sub == "stash" and (not rest or rest[0] in ("push", "save")):
        return [], [(ALL_PATHS, "restore")]
    if sub == "stash" and rest and rest[0] in ("pop", "apply"):
        # Re-applies stashed changes: new content, not the committed one.
        return [], [(ALL_PATHS, "replace")]
    if sub == "reset" and "--hard" in flags:
        return [], [(ALL_PATHS, "restore")]
    if sub in ("diff", "show"):
        # Named paths are views; the files a bare diff prints are read from
        # its output (see output_views).
        named = after_dd if after_dd is not None else [r for r in rest if plausible_path(clean(r))]
        return named, []
    return [], []


def _interpreter(argv, flags, ops, u, literal) -> Rw:
    script = u["heredoc"]
    if script is None:
        for flag in ("-c", "-e"):
            if flag in argv and argv.index(flag) + 1 < len(argv):
                script = argv[argv.index(flag) + 1]
                break
    return [], script_write_events(script or "")


_HANDLERS = {verb: _view for verb in _VIEW_VERBS}
_HANDLERS.update({verb: _grep for verb in _GREP_VERBS})
_HANDLERS.update({"sed": _sed, "awk": _awk, "gawk": _awk, "perl": _perl,
                  "cp": _copy, "mv": _copy, "install": _copy, "rm": _rm,
                  "tee": _tee, "truncate": _truncate, "patch": _patch, "git": _git})


def _strip_prefixes(argv: List[str]) -> List[str]:
    """Drop VAR=value assignments and wrappers such as sudo and timeout N."""
    argv = [a for a in argv if a != ""]
    while argv and (re.match(r"^[A-Za-z_]\w*=", argv[0]) or argv[0] in _WRAPPERS):
        argv = argv[1:]
    if argv and argv[0] == "timeout" and len(argv) > 2:
        argv = argv[2:]
    return argv


def _unit_events(u: Dict, literal_pipe: bool) -> List[FileEvent]:
    # An empty word is BSD sed's backup suffix (`sed -i '' ...`); left in, it
    # shifts the script into the file position.
    argv = _strip_prefixes(u["argv"])
    verb = argv[0].split("/")[-1] if argv else ""
    flags = [a for a in argv[1:] if a.startswith("-")]
    ops = [a for a in argv[1:] if not a.startswith("-")]
    literal = literal_pipe or (verb == "cat" and u["heredoc"] is not None)
    handler = _HANDLERS.get(verb) or (_interpreter if verb.startswith(_INTERPRETERS) else None)
    reads, writes = handler(argv, flags, ops, u, literal) if handler else ([], [])

    for op, target in u["redirs"]:
        op = op.lstrip("0123456789")
        if target.startswith("/dev/"):
            continue
        if op in (">", "&>", ">|"):
            writes.append((target, "overwrite" if literal else "replace"))
        elif op in (">>", "&>>"):
            writes.append((target, "append"))

    events: List[FileEvent] = []
    for r in reads:
        r = clean(r)
        if plausible_path(r):
            events.append(("R", r, "view"))
    for w, kind in writes:
        w = w if w == ALL_PATHS else clean(w)
        if w == ALL_PATHS or plausible_path(w):
            events.append(("W", w, kind))
    return events


# A pipeline ending in one of these prints a count or a digest, never the
# content its earlier stages read.
_SINKS = {"wc", "md5sum", "sha1sum", "sha256sum", "cksum", "xxd", "od"}


def _stdout_captured(u: Dict) -> bool:
    """Whether a unit's standard output goes to a file or /dev/null rather
    than to the terminal the agent reads."""
    for op, target in u["redirs"]:
        base = op.lstrip("0123456789")
        fd = op[:len(op) - len(base)] or "1"
        if base in ("&>", "&>>"):
            return True
        if fd == "1" and base in (">", ">>", ">|") and target not in ("/dev/stdout", "/dev/stderr", "/dev/tty"):
            return True
    return False


def _swallowed_pipes(units: List[Dict]) -> set:
    last: Dict[int, Dict] = {}
    for u in units:
        last[u["pipe"]] = u
    out = set()
    for pipe, u in last.items():
        argv = _strip_prefixes(u["argv"])
        verb = argv[0].split("/")[-1] if argv else ""
        flags = [a for a in argv[1:] if a.startswith("-")]
        if verb in _SINKS or (verb in _GREP_VERBS and any(
                f in _GREP_NO_CONTENT or _is_short_flag_with(f, "lLcq") for f in flags)):
            out.add(pipe)
    return out


def resolve_path(cwd: Optional[str], p: str) -> str:
    """`p` as named from working directory `cwd`. An unknown (None) or
    repository-root ('') working directory leaves it as written."""
    if not cwd or p == ALL_PATHS or p.startswith(("/", "~", "$")):
        return p
    return os.path.normpath(os.path.join(cwd, p))


def _cd_target(cwd: Optional[str], argv: List[str]) -> Optional[str]:
    """The working directory after `cd`/`pushd` from `cwd`; None when it
    cannot be known (home, a variable, `-`, above the repository root)."""
    if argv[0] == "popd":
        return None
    args = [a for a in argv[1:] if a == "-" or not a.startswith("-")]
    if not args:
        return None
    t = clean(args[0])
    if t.startswith(("~", "$")) or t == "-":
        return None
    if t.startswith("/"):
        return os.path.normpath(t)
    if cwd is None:
        return None
    new = os.path.normpath(os.path.join(cwd, t)) if cwd else os.path.normpath(t)
    if new == ".":
        return ""
    return None if new == ".." or new.startswith("../") else new


def _units_with_cwd(units: List[Dict], cwd: Optional[str]):
    """Each unit with the working directory it runs in, and the working
    directory the command leaves at top level. A `cd` applies to the units
    after it at its own subshell depth and deeper, until that subshell ends."""
    at = {0: cwd}
    placed = []
    for u in units:
        d = u.get("depth", 0)
        for k in [k for k in at if k > d]:
            del at[k]
        if d not in at:
            at[d] = at[max(k for k in at if k < d)] if any(k < d for k in at) else cwd
        argv = _strip_prefixes(u["argv"])
        placed.append((u, at[d]))
        if argv and argv[0].split("/")[-1] in ("cd", "pushd", "popd"):
            at[d] = _cd_target(at[d], [argv[0].split("/")[-1]] + argv[1:])
    return placed, at.get(0, cwd)


def command_cwd(cmd: str, cwd: Optional[str]) -> Optional[str]:
    """The working directory a command leaves behind, starting from `cwd`."""
    return _units_with_cwd(command_units(cmd), cwd)[1]


_SCRIPT_SUFFIXES = (".py", ".pl", ".rb", ".js")


def _run_script(u: Dict) -> Optional[str]:
    """The script file an interpreter unit runs (`python fix.py ...`)."""
    argv = _strip_prefixes(u["argv"])
    if not argv or not argv[0].split("/")[-1].startswith(_INTERPRETERS) or u["heredoc"] is not None:
        return None
    if any(a in ("-c", "-e", "-m") for a in argv[1:]):
        return None
    arg = next((a for a in argv[1:] if not a.startswith("-")), None)
    return clean(arg) if arg and clean(arg).endswith(_SCRIPT_SUFFIXES) else None


def _events_with_verbs(cmd: str, cwd: Optional[str] = None,
                       scripts: Optional[Dict[str, str]] = None) -> List[Tuple[str, str, str, str]]:
    """`scripts`, when given, maps script files the agent created (by a
    heredoc) to their text, and is updated in place: running one of them
    writes the files its text writes."""
    units = command_units(cmd)
    literal_pipes = set()
    for u in units:
        first = u["argv"][0].split("/")[-1] if u["argv"] else ""
        if first in _LITERAL_SOURCES or (first == "cat" and u["heredoc"] is not None):
            literal_pipes.add(u["pipe"])
    swallowed = _swallowed_pipes(units)
    # The last unit of each pipeline whose output goes to a file: what it and
    # every unit before it in the pipeline read never reaches the terminal.
    captured = {u["pipe"]: i for i, u in enumerate(units) if _stdout_captured(u)}
    placed, _ = _units_with_cwd(units, cwd)
    composed: set = set()
    out = []
    for i, (u, here) in enumerate(placed):
        argv = _strip_prefixes(u["argv"])
        verb = argv[0].split("/")[-1] if argv else ""
        if scripts is not None:
            run = _run_script(u)
            if run is not None and resolve_path(here, run) in scripts:
                path = resolve_path(here, run)
                if path in composed:
                    # Written and run in this one action: an edit composed now.
                    for p, kind in script_write_events(scripts[path]):
                        out.append(("W", resolve_path(here, clean(p)), kind, verb))
                else:
                    # A saved script is a program: its writes change the files,
                    # but running it is not an edit composed in this action
                    # against the agent's view of them.
                    for p in script_writes(scripts[path]):
                        out.append(("W", resolve_path(here, clean(p)), "replace", verb))
            if u["heredoc"] is not None:
                for op, target in u["redirs"]:
                    if op.lstrip("0123456789") in (">", ">|") and clean(target).endswith(_SCRIPT_SUFFIXES):
                        scripts[resolve_path(here, clean(target))] = u["heredoc"]
                        composed.add(resolve_path(here, clean(target)))
        for op, p, kind in _unit_events(u, u["pipe"] in literal_pipes):
            p = resolve_path(here, p)
            kind = map_kind_path(kind, lambda x: resolve_path(here, x))
            # Content piped into wc or a digest never reaches the agent.
            if op == "R" and u["pipe"] in swallowed and not _last_in_pipe(units, u):
                continue
            if op == "R" and captured.get(u["pipe"], -1) >= i:
                continue
            out.append((op, p, kind, verb))
    return out


def _last_in_pipe(units: List[Dict], u: Dict) -> bool:
    same = [x for x in units if x["pipe"] == u["pipe"]]
    return same[-1] is u


def shell_file_events(cmd: str, cwd: Optional[str] = None,
                      scripts: Optional[Dict[str, str]] = None) -> List[FileEvent]:
    """(op, path, kind) for every file a shell command shows or changes, in
    the order it does so. Relative paths are resolved against `cwd` and any
    `cd` the command makes; with no `cwd` they stay as written."""
    return [(op, p, kind) for op, p, kind, _ in _events_with_verbs(cmd, cwd, scripts)]


# bash names the offending line and quotes it; dash gives only its number.
# Both run every complete command before that line, so the cut is at the
# line, not at the start of the command.
_BASH_SYNTAX = re.compile(r"syntax error near unexpected token[^\n]*\n[^\n]*?line (\d+): `(.*)'")
_DASH_SYNTAX = re.compile(r"(?m)^\S*sh: (\d+): Syntax error:")
_EOF_ERROR = re.compile(r"unexpected EOF while looking for matching|Unterminated quoted string|end of file unexpected")
_FORMAT_ERROR = re.compile(r"Format error:|Please always provide EXACTLY ONE action")
_SED_FAILED = re.compile(r"^sed: ", re.M)
_TRACEBACK = re.compile(r"Traceback \(most recent call last\)")
_SCRIPT_SYNTAX = re.compile(r"^(?:SyntaxError|IndentationError|TabError): ", re.M)
_NOT_FOUND = re.compile(r"(?:python\d?(?:\.\d+)?|perl|ruby|node): (?:command )?not found")


def _line_start(cmd: str, line_no: int) -> int:
    """Character offset where 1-based line `line_no` begins."""
    pos = 0
    for _ in range(max(0, line_no - 1)):
        nxt = cmd.find("\n", pos)
        if nxt < 0:
            return len(cmd)
        pos = nxt + 1
    return pos


def executed_prefix(cmd: str, observation: str) -> str:
    """The part of a command the shell actually ran.

    A syntax error stops the shell at the offending line, after it has run
    every complete command before it; an unterminated quote swallows the rest
    of the text, and we then take nothing to have run."""
    obs = observation or ""
    if _FORMAT_ERROR.search(obs):
        return ""
    m = _BASH_SYNTAX.search(obs)
    if m:
        text = m.group(2).strip()
        idx = cmd.find(text) if text else -1
        if idx >= 0:
            return cmd[:cmd.rfind("\n", 0, idx) + 1] if "\n" in cmd[:idx] else ""
        return cmd[:_line_start(cmd, int(m.group(1)))]
    m = _DASH_SYNTAX.search(obs)
    if m:
        return cmd[:_line_start(cmd, int(m.group(1)))]
    if _EOF_ERROR.search(obs):
        return ""
    return cmd


def shell_events_with_outcome(cmd: str, observation: str = "", returncode: Optional[int] = None,
                              cwd: Optional[str] = None,
                              scripts: Optional[Dict[str, str]] = None) -> List[Tuple[str, str, str, bool]]:
    """Events with a `landed` flag read from the command's own output.

    Only the part of the command the shell ran yields events. A sed that
    printed an error changed nothing, and neither did a script that raised
    and exited nonzero. Other writes are taken to have landed; a sed whose
    pattern matched nothing exits cleanly and prints nothing, and is
    indistinguishable here."""
    obs = observation or ""
    cmd = executed_prefix(cmd, obs)
    sed_failed = bool(_SED_FAILED.search(obs))
    # A script that raised, did not parse, or whose interpreter was missing
    # wrote nothing. A heredoc cut short gives a SyntaxError with no
    # Traceback header.
    never_ran = (bool(_SCRIPT_SYNTAX.search(obs)) and not _TRACEBACK.search(obs) and returncode not in (0, None)) \
        or returncode in (126, 127) or bool(_NOT_FOUND.search(obs))
    script_failed = bool(_TRACEBACK.search(obs)) and returncode not in (0, None)
    out = []
    for op, p, kind, verb in _events_with_verbs(cmd, cwd, scripts):
        landed = True
        if op == "W" and verb == "sed" and sed_failed:
            landed = False
        if verb.startswith(_INTERPRETERS) and never_ran:
            continue  # the script never executed: no edit was attempted
        if op == "W" and verb.startswith(_INTERPRETERS) and script_failed:
            landed = False
        out.append((op, p, kind, landed))
    return out


# Error lines that name the file an operation failed on.
_NAMED_FAILURE = re.compile(
    r"sed: can't read (?P<a>[^:\n]+):"
    r"|cannot create (?P<b>[^:\n]+):"
    r"|(?:^|[\s'\"`])(?P<c>[^\s'\"`:]+)['\"`]?: (?:No such file or directory|Permission denied|"
    r"Is a directory|Read-only file system)", re.M)
_SED_EXPRESSION_ERROR = re.compile(r"^sed: -e expression #\d+, char \d+: ", re.M)


def _same_file(a: str, b: str) -> bool:
    a, b = clean(a), clean(b)
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def terminal_events_with_outcome(cmd: str, output: str = "", cwd: Optional[str] = None,
                                 scripts: Optional[Dict[str, str]] = None) -> List[Tuple[str, str, str, bool]]:
    """Events for keystrokes typed into a terminal, `landed` read from what
    the terminal then showed.

    Unlike a subprocess's output, a pane capture carries no exit status, may
    have lost its top lines, and can follow several commands. So a write is
    taken not to have landed only when an error line names its file, or when
    a sed expression error follows the only sed write typed."""
    output = output or ""
    failed = {m.group("a") or m.group("b") or m.group("c") for m in _NAMED_FAILURE.finditer(output)}
    events = _events_with_verbs(cmd, cwd, scripts)
    sed_writes = [i for i, (op, _, _, verb) in enumerate(events) if op == "W" and verb == "sed"]
    lone_sed_failed = len(sed_writes) == 1 and bool(_SED_EXPRESSION_ERROR.search(output))
    out = []
    for i, (op, p, kind, verb) in enumerate(events):
        landed = True
        if op == "W" and (any(_same_file(p, f) for f in failed) or (lone_sed_failed and i == sed_writes[0])):
            landed = False
        out.append((op, p, kind, landed))
    return out


# Output that quotes lines of a file: a traceback frame, a compiler or test
# diagnostic, a grep hit, a diff header. Each is a view of that file, seen
# after the command ran.
_OUT_TRACEBACK = re.compile(r'File "([^"<>\n]+)", line \d+')
_OUT_LOCATED = re.compile(r"(?m)^\s*(?:-->\s+)?(/?(?:[\w.\-]+/)*[\w.\-]+\.[A-Za-z]{1,6}):(\d+)[:\s]")
_OUT_DIFF = re.compile(r"(?m)^(?:diff --git a/(\S+) b/|\+\+\+ b/(\S+))")
_OUT_VIEWPORT = re.compile(r"\[VIEWPORT[^:]*:\s*([^\]\n]+)\]")
_OUT_TAIL_DIFF = re.compile(r"\[TAIL DIFF: File '([^'\n]+)'")


def output_views(observation: str, cwd: Optional[str] = None) -> List[str]:
    """Files whose content the command's output quoted.

    A traceback frame or a `file:line` diagnostic names its file relative to
    where the process ran, `cwd`; a git diff header and a harness viewport
    name it relative to the repository root."""
    obs = observation or ""
    here: List[str] = _OUT_TRACEBACK.findall(obs) + [m[0] for m in _OUT_LOCATED.findall(obs)]
    rooted: List[str] = ([a or b for a, b in _OUT_DIFF.findall(obs)] + [m.strip() for m in _OUT_VIEWPORT.findall(obs)]
                         + _OUT_TAIL_DIFF.findall(obs))
    found = [resolve_path(cwd, clean(p)) for p in here if plausible_path(clean(p))]
    found += [clean(p) for p in rooted if plausible_path(clean(p))]
    return list(dict.fromkeys(found))


def runs_interpreter(cmd: str) -> bool:
    """Whether the command runs a script interpreter at all."""
    return any((u["argv"][0].split("/")[-1] if u["argv"] else "").startswith(_INTERPRETERS)
               for u in command_units(cmd))
