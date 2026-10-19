"""Independent shell-command file-operation extractor for the RQ1 audit. Usage: imported by scripts/audit/extract.py (python3 scripts/audit/shellparse.py runs self-tests).

Written from scratch for the audit; it does not import src/parsers. Given one
shell command string and the working directory it starts in, it returns the
file operations the command performs, in order, each with:

  op      "R" (content reaches the agent) or "W" (file changed)
  path    absolute path after cwd resolution (canonicalised later)
  kind    R: "full" | "range" | "lines" | "listing"
          W: "anchored" | "overwrite" | "replace" | "append" | "delete" |
             "restore" | "touch"
  mech    the utility (sed, perl, python, patch, cp, redirect, ...)
  anchor  for anchored writes: what locates the edit (see below)
  view    for reads: {"range": [a, b]} when the shown lines are known
  cmd_i   index of the simple command inside the string
  guard   the connector before this simple command ("&&", "||", ";", "|")

Anchors:
  {"type": "sed", "cmds": [...]}      parsed sed script (s///, d, i, a, c with addresses)
  {"type": "perl", "subs": [...]}     s/// substitutions of perl -pe/-i
  {"type": "py", "replaces": [...], "regex": [...], "line_ops": bool, "reads_self": bool}
  {"type": "patch", "hunks": [...]}   unified or apply_patch hunks
  {"type": "ed", "script": str}
  {"type": "awk", "program": str}

A read whose stdout goes to a file or into a counting/digest sink is not a
view. cwd changes (`cd`) are tracked through the command and returned.
"""

import posixpath
import re
import shlex
from typing import Dict, List, Optional, Tuple

HEREDOC_RE = re.compile(r"""(?<!<)<<(-?)(?!<)\s*(\\?)(['"]?)([A-Za-z_][A-Za-z0-9_.-]*)\3""")
SEPS = {";", ";;", "&&", "||", "&", "\n"}
REDIR_TOKENS = {">", ">>", "<", "&>", "&>>", ">|", "2>", "2>>", "1>", "1>>", ">&", "2>&1", "<>"}
VIEW_VERBS = {"cat", "nl", "head", "tail", "less", "more", "bat", "batcat", "tac"}
GREP_VERBS = {"grep", "egrep", "fgrep", "rg", "ag", "ack", "zgrep"}
SINKS = {"wc", "md5sum", "sha1sum", "sha256sum", "cksum", "xxd", "od", "hexdump", "file", "stat", "true"}
LITERAL_SOURCES = {"echo", "printf"}
INTERP_PREFIX = ("python", "pypy")
WRAPPERS = {"sudo", "command", "exec", "nohup", "time", "env", "xargs", "nice", "stdbuf"}


def split_heredocs(cmd: str) -> Tuple[str, List[str]]:
    """Remove heredoc bodies; replace each operator by a placeholder word."""
    lines = (cmd or "").split("\n")
    out, bodies, i = [], [], 0
    while i < len(lines):
        line = lines[i]
        i += 1
        ops = list(HEREDOC_RE.finditer(line))
        if not ops:
            out.append(line)
            continue
        new_line = line
        for m in reversed(ops):
            k = len(bodies) + (len(ops) - 1 - ops.index(m))
            new_line = new_line[:m.start()] + f" __HEREDOC_{k}__ " + new_line[m.end():]
        out.append(new_line)
        for m in ops:
            strip_tabs = m.group(1) == "-"
            delim = m.group(4)
            body = []
            while i < len(lines):
                ln = lines[i]
                cmp = ln.lstrip("\t") if strip_tabs else ln
                if cmp.strip() == delim:
                    i += 1
                    break
                body.append(ln)
                i += 1
            bodies.append("\n".join(body))
    return "\n".join(out), bodies


def mask_heredocs(cmd: str) -> str:
    """The command text with every heredoc body blanked (line count kept),
    so positions of shell words can be searched without hitting payloads."""
    lines = (cmd or "").split("\n")
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        for m in HEREDOC_RE.finditer(line):
            strip_tabs = m.group(1) == "-"
            delim = m.group(4)
            while i < len(lines):
                ln = lines[i]
                cmp = ln.lstrip("\t") if strip_tabs else ln
                out.append(" " * len(ln))  # same length: offsets stay valid
                i += 1
                if cmp.strip() == delim:
                    break
    return "\n".join(out)


_OPS_RE = re.compile(r"\|\||&&|;;|&>>|>>|<<<|<<|&>|>&|<&|>\||[();<>|&\n]")
_PUNCT = set("();<>|&\n")


def tokenize(shell: str) -> List[str]:
    """shlex words; runs of punctuation (which shlex groups, e.g. ')\n')
    are split back into single shell operators."""
    shell = shell.replace("\\\n", " ")
    try:
        lx = shlex.shlex(shell, posix=True, punctuation_chars="();<>|&\n")
        lx.whitespace = " \t\r"
        lx.whitespace_split = True
        lx.commenters = "#"
        raw = list(lx)
    except ValueError:
        return []
    out = []
    for t in raw:
        if t and set(t) <= _PUNCT and len(t) > 1 and t not in ("&&", "||", ";;", ">>", "<<", "&>", ">&", "<&", ">|", "&>>", "<<<"):
            out.extend(_OPS_RE.findall(t))
        else:
            out.append(t)
    return out


def simple_commands(cmd: str) -> List[Dict]:
    """Simple commands with argv, redirections, stdin heredoc body, pipeline
    membership and the connector that precedes each one."""
    shell, bodies = split_heredocs(cmd)
    toks = tokenize(shell)
    if not toks and shell.strip():
        # Unbalanced quoting somewhere: tokenise line by line and keep the
        # lines that parse (a stray apostrophe in one line must not hide the
        # rest of the command).
        toks = []
        for line in shell.split("\n"):
            lt = tokenize(line)
            if lt:
                toks.extend(lt + ["\n"])
        if not toks:
            return [{"argv": [], "redirs": [], "heredoc": None, "pipe": 0, "guard": ";", "depth": 0,
                     "unparsed": True}]
    units, pipe, depth = [], 0, 0
    guard = ";"

    def new():
        return {"argv": [], "redirs": [], "heredoc": None, "pipe": pipe, "guard": guard, "depth": depth}

    cur = new()
    pending = None
    for t in toks:
        m = re.fullmatch(r"__HEREDOC_(\d+)__", t)
        if m:
            k = int(m.group(1))
            cur["heredoc"] = bodies[k] if k < len(bodies) else ""
            continue
        if pending is not None:
            cur["redirs"].append((pending, t))
            pending = None
            continue
        if t == "|":
            if cur["argv"] or cur["redirs"]:
                units.append(cur)
            guard = "|"
            cur = new()
            continue
        if t in ("(", ")"):
            if cur["argv"] or cur["redirs"]:
                units.append(cur)
            depth += 1 if t == "(" else -1
            units.append({"argv": [t], "redirs": [], "heredoc": None, "pipe": pipe, "guard": guard,
                          "depth": depth, "paren": True})
            pipe += 1
            guard = ";"
            cur = new()
            continue
        if t in SEPS or (set(t) <= set(";&|\n") and "\n" in t):
            if cur["argv"] or cur["redirs"]:
                units.append(cur)
            pipe += 1
            tt = t.replace("\n", "")
            guard = tt if tt in ("&&", "||") else ";"
            cur = new()
            continue
        if set(t) <= set("<>&") and ("<" in t or ">" in t):
            pending = t
            continue
        if re.fullmatch(r"\d?>&\d", t) or t in ("2>&1", ">&2", "1>&2"):
            continue
        cur["argv"].append(t)
    if cur["argv"] or cur["redirs"]:
        units.append(cur)
    # Fold "2" fd prefixes: tokenizer yields "2" then ">" for "2>"; drop a bare
    # digit argv word that is followed by a redirection of the same unit.
    for u in units:
        if u["argv"] and u["redirs"] and re.fullmatch(r"[0-9]", u["argv"][-1]):
            u["argv"] = u["argv"][:-1]
    return units


def strip_prefixes(argv: List[str]) -> List[str]:
    argv = [a for a in argv if a != ""]
    while argv and (re.match(r"^[A-Za-z_]\w*=", argv[0]) or argv[0] in WRAPPERS):
        if argv[0] == "env":
            argv = argv[1:]
            while argv and (argv[0].startswith("-") or re.match(r"^[A-Za-z_]\w*=", argv[0])):
                argv = argv[1:]
            continue
        argv = argv[1:]
    if argv and argv[0] == "timeout":
        rest = argv[1:]
        while rest and rest[0].startswith("-"):
            rest = rest[1:]
        argv = rest[1:] if rest else []
    return argv


def path_like(tok: str) -> bool:
    if not tok or tok.startswith("-") or tok in (".", "..", "/", "~"):
        return False
    if tok.startswith(("/dev/", "http://", "https://")) or tok == "/dev/null":
        return False
    if tok.startswith("/proc/") and not tok.startswith("/proc/self/cwd/"):
        return False
    if any(c in tok for c in "{}$*?!()<>\"'`=\n|;"):
        return False
    if tok.isdigit():
        return False
    return True


def resolve(path: str, cwd: str) -> str:
    if path.startswith("~"):
        path = "/root" + path[1:]
    if path.startswith("/proc/self/cwd/"):
        # the shell's working directory, spelled through procfs
        path = posixpath.join(cwd or "/", path[len("/proc/self/cwd/"):])
    p = path if path.startswith("/") else posixpath.join(cwd or "/", path)
    return posixpath.normpath(p)


# --- sed ---------------------------------------------------------------

def _read_delimited(s: str, i: int, d: str) -> Tuple[str, int]:
    """Read text up to the next unescaped delimiter d starting at s[i]."""
    out = []
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            if s[i + 1] == d:
                out.append(d)
            else:
                out.append(c + s[i + 1])
            i += 2
            continue
        if c == d:
            return "".join(out), i + 1
        out.append(c)
        i += 1
    return "".join(out), -1


def parse_sed_script(script: str) -> List[Dict]:
    """Parse a sed script into commands with addresses. Unknown syntax gives
    an entry with cmd '?'."""
    cmds = []
    i, n = 0, len(script)
    while i < n:
        while i < n and script[i] in " \t\n;":
            i += 1
        if i >= n:
            break
        addr = []
        for k in range(2):
            if k == 1 and i + 1 < n and script[i] in "+~" and script[i + 1].isdigit():
                # GNU extension: addr,+N (N lines after) or addr,~N (to a multiple of N)
                j = i + 1
                while j < n and script[j].isdigit():
                    j += 1
                addr.append({"rel" if script[i] == "+" else "mult": int(script[i + 1:j])})
                i = j
                break
            if i < n and script[i].isdigit():
                j = i
                while j < n and (script[j].isdigit()):
                    j += 1
                addr.append({"line": int(script[i:j])})
                i = j
            elif i < n and script[i] == "$":
                addr.append({"line": "$"})
                i += 1
            elif i < n and script[i] in "/\\":
                d = "/"
                if script[i] == "\\" and i + 1 < n:
                    d = script[i + 1]
                    i += 1
                pat, j = _read_delimited(script, i + 1, d)
                if j < 0:
                    return cmds + [{"cmd": "?", "raw": script[i:]}]
                addr.append({"re": pat})
                i = j
                while i < n and script[i] in "I":
                    i += 1
            else:
                break
            if i < n and script[i] == ",":
                i += 1
                continue
            break
        while i < n and script[i] in " \t":
            i += 1
        if i < n and script[i] == "!":
            i += 1
        if i >= n:
            cmds.append({"addr": addr, "cmd": "p"})
            break
        c = script[i]
        if c == "s" and i + 1 < n:
            d = script[i + 1]
            pat, j = _read_delimited(script, i + 2, d)
            if j < 0:
                return cmds + [{"cmd": "?", "raw": script[i:]}]
            rep, k = _read_delimited(script, j, d)
            if k < 0:
                return cmds + [{"cmd": "?", "raw": script[i:]}]
            m = re.match(r"[gpiIeEmM0-9w]*", script[k:])
            flags = m.group(0) if m else ""
            cmds.append({"addr": addr, "cmd": "s", "pat": pat, "rep": rep, "flags": flags})
            i = k + len(flags)
        elif c in "aic":
            j = i + 1
            backslash = False
            if j < n and script[j] == "\\":
                j += 1
                backslash = True
                if j < n and script[j] == "\n":
                    j += 1
            while j < n and script[j] in " \t" and not backslash:
                j += 1
            # GNU one-liner: text runs to end of line (or end of script)
            end = script.find("\n", j)
            text_lines = []
            if end < 0:
                text_lines.append(script[j:])
                i = n
            else:
                text_lines.append(script[j:end])
                i = end + 1
                # continuation lines ending with backslash
                while text_lines[-1].endswith("\\") and i < n:
                    text_lines[-1] = text_lines[-1][:-1]
                    end = script.find("\n", i)
                    if end < 0:
                        text_lines.append(script[i:])
                        i = n
                    else:
                        text_lines.append(script[i:end])
                        i = end + 1
            text = "\n".join(text_lines).replace("\\n", "\n")
            cmds.append({"addr": addr, "cmd": c, "text": text})
        elif c in "dpPDNnqQ=hHgGxlz":
            cmds.append({"addr": addr, "cmd": c})
            i += 1
        elif c == "y" and i + 1 < n:
            d = script[i + 1]
            a, j = _read_delimited(script, i + 2, d)
            b, k = _read_delimited(script, j, d) if j > 0 else ("", -1)
            cmds.append({"addr": addr, "cmd": "y", "a": a, "b": b})
            i = k if k > 0 else n
        elif c in "{}":
            cmds.append({"addr": addr, "cmd": c})
            i += 1
        elif c in "rwRW":
            end = script.find("\n", i)
            cmds.append({"addr": addr, "cmd": c, "file": script[i + 1:end if end > 0 else n].strip()})
            i = end + 1 if end > 0 else n
        else:
            cmds.append({"cmd": "?", "raw": script[i:]})
            break
    return cmds


def _sed_parts(argv: List[str]):
    """(script_texts, files, inplace, quiet, has_z, has_E)."""
    scripts, files = [], []
    inplace = quiet = has_z = has_E = False
    i = 1
    had_e = False
    while i < len(argv):
        a = argv[i]
        if a in ("-e", "--expression") and i + 1 < len(argv):
            scripts.append(argv[i + 1])
            had_e = True
            i += 2
            continue
        if a.startswith("--expression="):
            scripts.append(a.split("=", 1)[1])
            had_e = True
        elif a == "-f" and i + 1 < len(argv):
            scripts.append("__FILE__:" + argv[i + 1])
            had_e = True
            i += 2
            continue
        elif a.startswith("--in-place") or (a.startswith("-") and not a.startswith("--") and "i" in a[1:].split("=")[0][:1] + ("i" if re.match(r"^-[a-zA-Z]*i", a) else "")):
            inplace = True
            if re.match(r"^-[a-zA-Z]*n", a):
                quiet = True
            if "E" in a or "r" in a[1:2]:
                has_E = True
        elif a in ("-n", "--quiet", "--silent") or re.match(r"^-[a-zA-Z]*n[a-zA-Z]*$", a):
            quiet = True
            if "i" in a:
                inplace = True
            if "E" in a or "r" in a:
                has_E = True
        elif a in ("-z", "--null-data") or re.match(r"^-[a-zA-Z]*z", a):
            has_z = True
        elif a in ("-E", "-r", "--regexp-extended"):
            has_E = True
        elif a.startswith("-") and len(a) > 1:
            if re.match(r"^-[a-zA-Z]+$", a):
                if "i" in a:
                    inplace = True
                if "n" in a:
                    quiet = True
                if "E" in a or "r" in a:
                    has_E = True
                if "z" in a:
                    has_z = True
        elif a == "" and inplace:
            pass  # BSD backup suffix
        else:
            if not had_e and not scripts:
                scripts.append(a)
            else:
                files.append(a)
        i += 1
    return scripts, files, inplace, quiet, has_z, has_E


def sed_view_range(cmds: List[Dict]) -> Optional[Tuple[int, int]]:
    """For `sed -n 'a,bp'` (or 'Np'), the printed line range."""
    ps = [c for c in cmds if c.get("cmd") == "p"]
    if len(ps) != 1 or len(cmds) > 2:
        return None
    addr = ps[0].get("addr") or []
    if len(addr) == 1 and isinstance(addr[0].get("line"), int):
        return addr[0]["line"], addr[0]["line"]
    if len(addr) == 2 and isinstance(addr[0].get("line"), int):
        if isinstance(addr[1].get("rel"), int):
            return addr[0]["line"], addr[0]["line"] + addr[1]["rel"]
        b = addr[1].get("line")
        return addr[0]["line"], (b if isinstance(b, int) else 10 ** 9)
    return None


# --- python scripts ------------------------------------------------------

_PY_STR = r"""(?:[rbfuRBFU]{0,2})(?:'''(?:.|\n)*?'''|\"\"\"(?:.|\n)*?\"\"\"|'(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*")"""
_PY_REPLACE = re.compile(r"\.replace\(\s*(" + _PY_STR + r")\s*,\s*(" + _PY_STR + r")")
_PY_RESUB = re.compile(r"re\.sub[n]?\(\s*(" + _PY_STR + r")\s*,\s*(" + _PY_STR + r")?")
_PY_OPEN = re.compile(r"""open\(\s*(?P<arg>[^,)]+?)\s*(?:,\s*(?:mode\s*=\s*)?(?P<mode>[rbfu]?['"][^'"]*['"]))?\s*[,)]""")
_PY_ASSIGN = re.compile(r"""(?P<v>\b[A-Za-z_]\w*)\s*=\s*(?:(?:pathlib\.)?Path\(\s*)?(?P<s>[rbfu]?(['"])(?:[^'"\n]+)\3)""")
_PY_PATHWRITE = re.compile(r"""(?:(?:pathlib\.)?Path\(\s*(?P<lit>[rbfu]?(['"])[^'"\n]+\2)\s*\)|(?P<v>\b[A-Za-z_]\w*))\s*\.write_(?:text|bytes)\(""")
_PY_PATHREAD = re.compile(r"""(?:(?:pathlib\.)?Path\(\s*(?P<lit>[rbfu]?(['"])[^'"\n]+\2)\s*\)|(?P<v>\b[A-Za-z_]\w*))\s*\.read_(?:text|bytes)\(""")
_PY_LINEOPS = re.compile(r"\w*lines\[\s*[\w+\- ]+\]\s*=|\.insert\(\s*[\w+\- ]+,|del\s+\w+\[|\w+\[\s*\d+\s*:\s*\d+\s*\]|splitlines\(\)\[")
# line operations that change lines (assign, insert, delete, slice-assign); indexing or slicing alone only reads
_PY_LINEEDIT = re.compile(r"\w*lines\[\s*[\w+\- ]+\]\s*=(?!=)|\.insert\(\s*[\w+\- ]+,|del\s+\w+\[|\w+\[[^\]\n]*:[^\]\n]*\]\s*=(?!=)")
# text spliced around a found position: s[:i] + new + s[j:]
_PY_SPLICE = re.compile(r"\w+\[\s*:\s*[\w.()+\- ]+\]\s*\+")
_PY_FIND = re.compile(r"\.(?:find|index|rfind|rindex|startswith|count)\(\s*(" + _PY_STR + r")|(" + _PY_STR + r")\s+(?:not\s+)?in\s+\w+")


def py_literal(tok: str) -> Optional[str]:
    try:
        import ast
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = ast.literal_eval(tok)
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        return v if isinstance(v, str) else None
    except Exception:
        return None


def python_script_ops(script: str, cwd: str, strict: bool = False) -> Tuple[List[Dict], List[str]]:
    """(write ops, read paths) of a Python script, by literal or variable."""
    if not script:
        return [], []
    names: Dict[str, str] = {}
    for m in _PY_ASSIGN.finditer(script):
        lit = py_literal(m.group("s"))
        if lit:
            names[m.group("v")] = lit

    def arg_path(arg: str) -> Optional[str]:
        arg = arg.strip()
        lit = py_literal(arg)
        if lit is not None:
            return lit
        m = re.fullmatch(r"(?:pathlib\.)?Path\(\s*(" + _PY_STR + r")\s*\)", arg)
        if m:
            return py_literal(m.group(1))
        if re.fullmatch(r"[A-Za-z_]\w*", arg):
            return names.get(arg)
        return None

    writes, reads = [], []
    wline: Dict[str, int] = {}

    def line_of(pos: int) -> int:
        return script.count("\n", 0, pos) + 1

    for m in _PY_OPEN.finditer(script):
        p = arg_path(m.group("arg"))
        if not p:
            continue
        mode = py_literal(m.group("mode")) if m.group("mode") else "r"
        mode = mode or "r"
        if any(c in mode for c in "wa+x"):
            writes.append((p, "a" in mode))
            wline[p] = min(wline.get(p, 10 ** 9), line_of(m.start()))
        else:
            reads.append(p)
    for m in _PY_PATHWRITE.finditer(script):
        p = py_literal(m.group("lit")) if m.group("lit") else names.get(m.group("v") or "")
        if p:
            writes.append((p, False))
            wline[p] = min(wline.get(p, 10 ** 9), line_of(m.start()))
    for m in _PY_PATHREAD.finditer(script):
        p = py_literal(m.group("lit")) if m.group("lit") else names.get(m.group("v") or "")
        if p:
            reads.append(p)
    replaces = []
    for m in _PY_REPLACE.finditer(script):
        a, b = py_literal(m.group(1)), py_literal(m.group(2))
        if a is not None and b is not None:
            replaces.append({"old": a, "new": b})
    regex = []
    for m in _PY_RESUB.finditer(script):
        a = py_literal(m.group(1))
        b = py_literal(m.group(2)) if m.group(2) else None
        if a is not None:
            regex.append({"pat": a, "rep": b})
    # compiled patterns: pat = re.compile(LIT) ... pat.sub(REP, text)
    compiled = {}
    for m in re.finditer(r"(\b[A-Za-z_]\w*)\s*=\s*re\.compile\(\s*(" + _PY_STR + r")", script):
        lit = py_literal(m.group(2))
        if lit is not None:
            compiled[m.group(1)] = lit
    for m in re.finditer(r"(\b[A-Za-z_]\w*)\.subn?\(\s*(" + _PY_STR + r")?", script):
        if m.group(1) in compiled:
            regex.append({"pat": compiled[m.group(1)], "rep": py_literal(m.group(2)) if m.group(2) else None})
    line_ops = bool(_PY_LINEOPS.search(script))
    line_edit = bool(_PY_LINEEDIT.search(script))
    splice = bool(_PY_SPLICE.search(script))
    finds = []
    for m in _PY_FIND.finditer(script):
        lit = py_literal(m.group(1) or m.group(2) or "")
        if lit and len(lit) >= 3:
            finds.append(lit)
    ops = []
    read_set = {resolve(r, cwd) for r in reads if path_like(r)}
    for p, append in dict.fromkeys(writes):
        if not path_like(p):
            continue
        ap = resolve(p, cwd)
        first_write_line = wline.get(p)
        self_read = ap in read_set
        if append:
            ops.append({"op": "W", "path": ap, "kind": "append", "mech": "python",
                        "anchor": None, "write_line": first_write_line})
        elif self_read and (replaces or regex or (not strict and (line_edit or (finds and splice)))):
            # anchored: a literal substitution, a line edit, or text spliced at a found position. A script
            # that only tests for a literal (`if 'x' in s`) or slices to read, then writes, is a rewrite
            ops.append({"op": "W", "path": ap, "kind": "anchored", "mech": "python",
                        "anchor": {"type": "py", "replaces": replaces, "regex": regex, "finds": finds[:20],
                                   "line_ops": line_ops, "reads_self": True},
                        "write_line": first_write_line})
        elif self_read:
            # read-transform-write with no textual anchor (json.load/dump, a
            # program updating its data file): a whole-file rewrite
            ops.append({"op": "W", "path": ap, "kind": "rewrite", "mech": "python",
                        "anchor": None, "write_line": first_write_line})
        else:
            ops.append({"op": "W", "path": ap, "kind": "overwrite_unseen", "mech": "python",
                        "anchor": None, "write_line": first_write_line})
    return ops, sorted(read_set)


_PY_READEXPR = r"(?:open\(\s*(?P<a{n}>[^,)]+?)\s*(?:,\s*[rbfu]?['\"]r?t?['\"])?\s*\)\.read\(\)|(?:(?:pathlib\.)?Path\(\s*(?P<l{n}>" + _PY_STR + r")\s*\)|(?P<v{n}>\b[A-Za-z_]\w*))\.read_text\(\))"
_PY_PRINT_READ = re.compile(r"(?:\bprint|sys\.stdout\.write)\(\s*" + _PY_READEXPR.replace("{n}", "1") + r"\s*[,)]")
_PY_VAR_READ = re.compile(r"(?P<var>\b[A-Za-z_]\w*)\s*=\s*" + _PY_READEXPR.replace("{n}", "2") + r"\s*$", re.M)


def python_printed(script: str, cwd: str) -> List[str]:
    """Files whose whole text a Python script prints unchanged: print(open(p).read()),
    print(Path(p).read_text()), or s = open(p).read() ... print(s) with s never reassigned."""
    if not script or "print" not in script and "stdout" not in script:
        return []
    names: Dict[str, str] = {}
    for m in _PY_ASSIGN.finditer(script):
        lit = py_literal(m.group("s"))
        if lit:
            names[m.group("v")] = lit

    def target(m, n):
        a, lit, v = m.group(f"a{n}"), m.group(f"l{n}"), m.group(f"v{n}")
        if a is not None:
            a = a.strip()
            return py_literal(a) if py_literal(a) is not None else names.get(a)
        if lit is not None:
            return py_literal(lit)
        return names.get(v or "")

    out = [target(m, 1) for m in _PY_PRINT_READ.finditer(script)]
    for m in _PY_VAR_READ.finditer(script):
        var = m.group("var")
        if len(re.findall(r"\b" + re.escape(var) + r"\s*(?:[+\-*]?=)(?!=)", script)) == 1 \
                and re.search(r"(?:\bprint|sys\.stdout\.write)\(\s*" + re.escape(var) + r"\s*[,)]", script):
            out.append(target(m, 2))
    return [resolve(p, cwd) for p in dict.fromkeys(out) if p and path_like(p)]


# --- patches ---------------------------------------------------------------

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_unified(body: str, cwd: str, strip: int = 1) -> List[Dict]:
    """Per-file hunks of a unified diff (or OpenAI apply_patch envelope)."""
    out: Dict[str, Dict] = {}
    cur = None
    hunk = None
    lines = (body or "").split("\n")
    for ln in lines:
        if ln.startswith("+++ "):
            p = ln[4:].split("\t")[0].strip()
            if p == "/dev/null":
                cur = None
                continue
            parts = p.split("/")
            p = "/".join(parts[strip:]) if strip and len(parts) > strip and p.startswith(("a/", "b/")) else p
            cur = out.setdefault(resolve(p, cwd), {"hunks": []})
            hunk = None
        elif ln.startswith("*** Update File:") or ln.startswith("*** Add File:"):
            p = ln.split(":", 1)[1].strip()
            cur = out.setdefault(resolve(p, cwd), {"hunks": [], "envelope": ln.split(":")[0][4:]})
            hunk = None
        elif ln.startswith("--- ") or ln.startswith("diff ") or ln.startswith("index ") or ln.startswith("*** "):
            continue
        elif cur is not None and (ln.startswith("@@")):
            m = _HUNK_RE.match(ln)
            hunk = {"old_start": int(m.group(1)) if m else None, "new_start": int(m.group(3)) if m else None,
                    "lines": []}
            cur["hunks"].append(hunk)
        elif cur is not None and ln[:1] in (" ", "+", "-"):
            if hunk is None:
                hunk = {"old_start": None, "new_start": None, "lines": []}
                cur["hunks"].append(hunk)
            hunk["lines"].append(ln)
    return [{"path": p, **v} for p, v in out.items()]


def hunk_texts(h: Dict) -> Tuple[str, str]:
    old = [ln[1:] for ln in h["lines"] if ln[:1] in (" ", "-")]
    new = [ln[1:] for ln in h["lines"] if ln[:1] in (" ", "+")]
    return "\n".join(old), "\n".join(new)


# --- main extractor ---------------------------------------------------------

def _grep_parts(argv):
    flags, operands = [], []
    i = 1
    arg_flags = {"-e", "-f", "--regexp", "-A", "-B", "-C", "-m", "--max-count", "--include", "--exclude",
                 "--exclude-dir", "-g", "--glob", "-t", "--type", "-d", "-D"}
    pats = []
    while i < len(argv):
        a = argv[i]
        if a in arg_flags and i + 1 < len(argv):
            if a in ("-e", "--regexp"):
                pats.append(argv[i + 1])
            flags.append(a)
            i += 2
            continue
        if a.startswith("-") and len(a) > 1:
            flags.append(a)
        else:
            operands.append(a)
        i += 1
    if not pats and operands:
        pats.append(operands[0])
        operands = operands[1:]
    no_content = any(f in ("-l", "-L", "-c", "-q", "--files-with-matches", "--files-without-match", "--count",
                           "--quiet", "--silent") or (re.match(r"^-[a-zA-Z]+$", f) and set(f[1:]) & set("lLcq"))
                     for f in flags)
    numbered = any(f in ("-n", "--line-number") or (re.match(r"^-[a-zA-Z]+$", f) and "n" in f) for f in flags)
    return operands, no_content, numbered, pats


def head_tail_n(argv) -> Optional[int]:
    for i, a in enumerate(argv[1:], 1):
        m = re.fullmatch(r"-n\s*(\d+)|-(\d+)|--lines=(\d+)", a)
        if m:
            return int(next(g for g in m.groups() if g))
        if a == "-n" and i + 1 < len(argv) and re.fullmatch(r"\+?\d+", argv[i + 1]):
            return int(argv[i + 1].lstrip("+"))
    return None


_DEPTH = [0]


def extract(cmd: str, cwd: str, known_files: Optional[Dict[str, str]] = None,
            persist_cd: bool = True) -> Tuple[List[Dict], str, Dict]:
    """File operations of one shell command. Returns (ops, cwd_after, info).

    known_files maps absolute path -> text for files whose content the agent
    wrote itself (heredoc scripts), so `python3 fix.py` can be analysed."""
    known_files = known_files if known_files is not None else {}
    units = simple_commands(cmd)
    info = {"unparsed": any(u.get("unparsed") for u in units), "n_units": len(units), "verbs": []}
    ops: List[Dict] = []
    cwd0 = cwd
    stack = []
    # which pipes end in a sink or a no-content grep
    last_in_pipe: Dict[int, Dict] = {}
    for u in units:
        if not u.get("paren"):
            last_in_pipe[u["pipe"]] = u
    pipe_members: Dict[int, List[Dict]] = {}
    for u in units:
        pipe_members.setdefault(u["pipe"], []).append(u)

    def pipe_swallows(u) -> bool:
        """Content produced by u does not reach the agent."""
        members = pipe_members.get(u["pipe"], [])
        idx = members.index(u)
        downstream = members[idx + 1:]
        # redirected stdout of u itself
        for op, t in u["redirs"]:
            if op in (">", ">>", ">|", "&>", "&>>", "1>", "1>>") and t not in ("/dev/stderr",):
                return True
        for v in downstream:
            argv = strip_prefixes(v["argv"])
            verb = argv[0].split("/")[-1] if argv else ""
            if verb in SINKS:
                return True
            if verb in GREP_VERBS:
                _, no_content, _, _ = _grep_parts(argv)
                if no_content:
                    return True
            if verb in ("tee",) and all(t.startswith("/dev/") for t in argv[1:] if not t.startswith("-")) is False:
                # tee writes a copy but still prints: content reaches the agent
                pass
        if downstream:
            last = downstream[-1]
            for op, t in last["redirs"]:
                if op in (">", ">>", ">|", "&>", "&>>", "1>", "1>>") and t != "/dev/stderr":
                    return True
        return False

    def filtered(u) -> bool:
        """Content passes through a filter (grep/head/tail/sed/awk) downstream."""
        members = pipe_members.get(u["pipe"], [])
        idx = members.index(u)
        for v in members[idx + 1:]:
            argv = strip_prefixes(v["argv"])
            verb = argv[0].split("/")[-1] if argv else ""
            if verb in GREP_VERBS or verb in ("head", "tail", "sed", "awk", "cut", "sort", "uniq", "tr", "less", "more"):
                return True
        return False

    literal_pipes = set()
    for u in units:
        argv = strip_prefixes(u["argv"])
        verb = argv[0].split("/")[-1] if argv else ""
        if verb in LITERAL_SOURCES or (verb == "cat" and u["heredoc"] is not None and
                                       not [a for a in argv[1:] if not a.startswith("-")]):
            literal_pipes.add(u["pipe"])

    for ui, u in enumerate(units):
        if u.get("paren"):
            if u["argv"][0] == "(":
                stack.append(cwd)
            elif stack:
                cwd = stack.pop()
            continue
        argv = strip_prefixes(u["argv"])
        verb = argv[0].split("/")[-1] if argv else ""
        info["verbs"].append(verb)
        guard = u["guard"]
        literal = u["pipe"] in literal_pipes
        new_ops: List[Dict] = []
        operands = [a for a in argv[1:] if not a.startswith("-")]

        if verb == "cd":
            tgt = operands[0] if operands else "~"
            if tgt == "-" or any(c in tgt for c in "$`*?{}"):
                continue  # previous or computed directory: unknown, keep cwd
            cwd = resolve(tgt, cwd)
            continue
        if verb in ("pushd",) and operands:
            stack.append(cwd)
            cwd = resolve(operands[0], cwd)
            continue
        if verb == "popd" and stack:
            cwd = stack.pop()
            continue

        swallowed = pipe_swallows(u)
        if verb in VIEW_VERBS:
            files = [a for a in operands if path_like(a)]
            if verb in ("head", "tail"):
                files = [a for a in argv[1:] if path_like(a) and not re.fullmatch(r"\+?\d+", a)]
                # drop the numeric argument of -n
                cleaned = []
                skip = False
                for a in argv[1:]:
                    if skip:
                        skip = False
                        continue
                    if a in ("-n", "-c", "--lines", "--bytes"):
                        skip = True
                        continue
                    if not a.startswith("-") and path_like(a):
                        cleaned.append(a)
                files = cleaned
            for op_, t in u["redirs"]:
                if op_ == "<":
                    files.append(t)
            for f in files:
                if swallowed:
                    continue
                view = {}
                if verb == "head":
                    n = head_tail_n(argv) or 10
                    view = {"range": [1, n]}
                elif verb == "tail":
                    view = {"tail": head_tail_n(argv) or 10}
                elif verb in ("cat", "nl", "bat", "batcat", "less", "more", "tac"):
                    view = {"range": [1, 10 ** 9]}
                    if verb == "nl" or "-n" in argv:
                        view["numbered"] = True
                kind = "full" if view.get("range") == [1, 10 ** 9] else "range"
                if filtered(u):
                    kind, view = "lines", {"filtered": True}
                new_ops.append({"op": "R", "path": resolve(f, cwd), "kind": kind, "mech": verb, "view": view})
        elif verb in GREP_VERBS:
            files, no_content, numbered, pats = _grep_parts(argv)
            recursive = any(re.match(r"^-[a-zA-Z]*[rR]", f) or f in ("--recursive",) for f in argv[1:]
                            if f.startswith("-")) or verb in ("rg", "ag")
            if not no_content and not swallowed:
                for f in files:
                    if path_like(f):
                        new_ops.append({"op": "R", "path": resolve(f, cwd), "kind": "lines", "mech": verb,
                                        "view": {"grep": pats[:1], "numbered": numbered,
                                                 "recursive": recursive}})
        elif verb == "sed":
            scripts, files, inplace, quiet, has_z, has_E = _sed_parts(argv)
            files = [f for f in files if path_like(f)] + [t for o, t in u["redirs"] if o == "<"]
            cmds = []
            for s in scripts:
                if s.startswith("__FILE__:"):
                    cmds.append({"cmd": "?", "raw": s})
                else:
                    cmds.extend(parse_sed_script(s))
            if inplace:
                for f in files:
                    new_ops.append({"op": "W", "path": resolve(f, cwd), "kind": "anchored", "mech": "sed",
                                    "anchor": {"type": "sed", "cmds": cmds, "z": has_z, "E": has_E, "raw": scripts}})
            elif not swallowed:
                rng = sed_view_range(cmds) if quiet else None
                for f in files:
                    if rng:
                        view = {"range": list(rng)}
                        kind = "range"
                    elif not quiet and not any(c.get("cmd") in ("d",) for c in cmds):
                        view, kind = {"transformed": True}, "lines"
                    else:
                        view, kind = {"sed": True}, "lines"
                    if filtered(u):
                        kind, view = "lines", {"filtered": True, **view}
                    new_ops.append({"op": "R", "path": resolve(f, cwd), "kind": kind, "mech": "sed", "view": view})
        elif verb in ("awk", "gawk", "mawk"):
            inplace = "inplace" in argv
            prog, files = None, []
            skip = False
            for i, a in enumerate(argv[1:], 1):
                if skip:
                    skip = False
                    continue
                if a in ("-f", "-v", "-i", "-F"):
                    skip = True
                    if a == "-f" and i + 1 < len(argv):
                        prog = "__FILE__"
                    continue
                if a.startswith("-"):
                    continue
                if prog is None:
                    prog = a
                else:
                    files.append(a)
            files = [f for f in files if path_like(f)]
            if inplace:
                for f in files:
                    new_ops.append({"op": "W", "path": resolve(f, cwd), "kind": "anchored", "mech": "awk",
                                    "anchor": {"type": "awk", "program": prog}})
            elif not swallowed:
                for f in files:
                    new_ops.append({"op": "R", "path": resolve(f, cwd), "kind": "lines", "mech": "awk",
                                    "view": {"awk": prog}})
        elif verb == "perl":
            files, script, skip, inplace, printing = [], None, False, False, False
            for a in argv[1:]:
                if skip:
                    script = a
                    skip = False
                    continue
                if a.startswith("-") and not a.startswith("--"):
                    if "i" in a[1:]:
                        inplace = True
                    if set(a[1:]) & set("pn"):
                        printing = True
                    if a[-1] in "eE":
                        skip = True
                    continue
                files.append(a)
            if script is None and files:
                script, files = files[0], files[1:]
            files = [f for f in files if path_like(f)]
            subs = []
            for m in re.finditer(r"s(.)(.*?)(?<!\\)\1(.*?)(?<!\\)\1([a-z]*)", script or "", re.S):
                subs.append({"pat": m.group(2), "rep": m.group(3), "flags": m.group(4)})
            if inplace:
                for f in files:
                    new_ops.append({"op": "W", "path": resolve(f, cwd), "kind": "anchored", "mech": "perl",
                                    "anchor": {"type": "perl", "subs": subs, "script": script}})
            elif printing and not swallowed:
                for f in files:
                    new_ops.append({"op": "R", "path": resolve(f, cwd), "kind": "lines", "mech": "perl", "view": {}})
            elif script:
                w, r = python_script_ops("", cwd)  # perl scripts: literal open() rarely used; skip
        elif verb.startswith(INTERP_PREFIX) or verb in ("ipython",):
            script = None
            src = None
            if "-c" in argv and argv.index("-c") + 1 < len(argv):
                script = argv[argv.index("-c") + 1]
                src = "-c"
            elif u["heredoc"] is not None:
                script = u["heredoc"]
                src = "heredoc"
            else:
                for a in argv[1:]:
                    if a.startswith("-"):
                        if a == "-m":
                            break
                        continue
                    ap = resolve(a, cwd)
                    if ap in known_files:
                        script = known_files[ap]
                        src = "file:" + ap
                    break
            if script is not None:
                # a program run by path (not an inline edit script) counts as an
                # anchored edit only when it performs literal text substitution
                w, r = python_script_ops(script, cwd, strict=bool(src and src.startswith("file:")))
                for o in w:
                    o["script_src"] = src
                new_ops.extend(w)
                if not swallowed:
                    # a file the script prints whole, without writing it, reaches the agent like `cat`
                    written = {o["path"] for o in w}
                    for f in python_printed(script, cwd):
                        if f not in written:
                            new_ops.append({"op": "R", "path": f, "kind": "lines" if filtered(u) else "full",
                                            "mech": "python-print", "view": {}})
        elif verb in ("cp", "mv", "install"):
            targets = [a for a in operands if path_like(a)]
            if len(targets) >= 2:
                dst = resolve(targets[-1], cwd)
                for s in targets[:-1]:
                    sp = resolve(s, cwd)
                    d_ = dst
                    new_ops.append({"op": "W", "path": d_, "kind": "replace", "mech": verb, "src": sp,
                                    "dst_may_be_dir": True})
                    if verb == "mv":
                        new_ops.append({"op": "W", "path": sp, "kind": "delete", "mech": "mv"})
        elif verb == "rm":
            for a in operands:
                if path_like(a):
                    new_ops.append({"op": "W", "path": resolve(a, cwd), "kind": "delete", "mech": "rm"})
        elif verb == "touch":
            for a in operands:
                if path_like(a):
                    new_ops.append({"op": "W", "path": resolve(a, cwd), "kind": "touch", "mech": "touch"})
        elif verb == "tee":
            app = "-a" in argv or "--append" in argv
            for a in operands:
                if path_like(a):
                    new_ops.append({"op": "W", "path": resolve(a, cwd),
                                    "kind": "append" if app else ("overwrite" if literal else "replace"),
                                    "mech": "tee", "text": _literal_text(units, u) if literal else None})
        elif verb == "truncate":
            for a in operands:
                if path_like(a) and not re.fullmatch(r"\d+", a):
                    new_ops.append({"op": "W", "path": resolve(a, cwd), "kind": "replace", "mech": "truncate"})
        elif verb in ("patch",):
            body = u["heredoc"]
            strip = 1
            for a in argv[1:]:
                m = re.fullmatch(r"-p(\d+)", a)
                if m:
                    strip = int(m.group(1))
            named = [a for a in operands if path_like(a) and not a.endswith((".patch", ".diff"))]
            pfile = None
            for op_, t in u["redirs"]:
                if op_ == "<":
                    pfile = resolve(t, cwd)
            for a in argv[1:]:
                if a.startswith("-i") and len(a) > 2:
                    pfile = resolve(a[2:], cwd)
            if body is None and pfile and pfile in known_files:
                body = known_files[pfile]
            files = parse_unified(body or "", cwd, strip)
            if files:
                for fobj in files:
                    new_ops.append({"op": "W", "path": fobj["path"], "kind": "anchored", "mech": "patch",
                                    "anchor": {"type": "patch", "hunks": fobj["hunks"]}})
            for a in named:
                new_ops.append({"op": "W", "path": resolve(a, cwd), "kind": "anchored", "mech": "patch",
                                "anchor": {"type": "patch", "hunks": []}})
        elif verb in ("applypatch", "apply_patch"):
            files = parse_unified(u["heredoc"] or (argv[1] if len(argv) > 1 else ""), cwd, 0)
            for fobj in files:
                new_ops.append({"op": "W", "path": fobj["path"], "kind": "anchored", "mech": "applypatch",
                                "anchor": {"type": "patch", "hunks": fobj["hunks"]}})
        elif verb in ("ed", "ex"):
            files = [a for a in operands if path_like(a)]
            for f in files:
                new_ops.append({"op": "W", "path": resolve(f, cwd), "kind": "anchored", "mech": "ed",
                                "anchor": {"type": "ed", "script": u["heredoc"] or ""}})
        elif verb == "git" and operands:
            sub = operands[0]
            rest = operands[1:]
            after_dd = argv[argv.index("--") + 1:] if "--" in argv else None
            if sub == "apply":
                body = u["heredoc"]
                pf = [a for a in rest if path_like(a)]
                if body is None and pf and resolve(pf[0], cwd) in known_files:
                    body = known_files[resolve(pf[0], cwd)]
                for fobj in parse_unified(body or "", cwd, 1):
                    new_ops.append({"op": "W", "path": fobj["path"], "kind": "anchored", "mech": "git-apply",
                                    "anchor": {"type": "patch", "hunks": fobj["hunks"]}})
            elif sub in ("checkout", "restore"):
                named = after_dd if after_dd is not None else [r for r in rest if path_like(r) or r == "."]
                if "." in named or not named:
                    if sub == "checkout" and rest and not after_dd and not any(path_like(r) for r in rest):
                        new_ops.append({"op": "W", "path": "*", "kind": "restore", "mech": "git-checkout-branch"})
                    else:
                        new_ops.append({"op": "W", "path": "*", "kind": "restore", "mech": "git"})
                else:
                    for p in named:
                        new_ops.append({"op": "W", "path": resolve(p, cwd), "kind": "restore", "mech": "git"})
            elif sub == "stash" and (not rest or rest[0] in ("push", "save")):
                new_ops.append({"op": "W", "path": "*", "kind": "restore", "mech": "git-stash"})
            elif sub == "stash" and rest and rest[0] in ("pop", "apply"):
                new_ops.append({"op": "W", "path": "*", "kind": "replace", "mech": "git-stash-pop"})
            elif sub == "reset" and "--hard" in argv:
                new_ops.append({"op": "W", "path": "*", "kind": "restore", "mech": "git-reset"})
            elif sub in ("diff", "show") and not swallowed:
                named = after_dd if after_dd is not None else [r for r in rest if path_like(r) and "/" in r or
                                                                re.search(r"\.[A-Za-z]{1,6}$", r or "")]
                for p in named:
                    new_ops.append({"op": "R", "path": resolve(p, cwd), "kind": "lines", "mech": "git-diff",
                                    "view": {"diff": True}})
        elif verb == "str_replace_editor":
            pass  # handled by the SWE-agent adapter
        elif verb in ("bash", "sh", "source", ".") and _DEPTH[0] < 2:
            # a shell script the agent wrote earlier in the trajectory: its
            # commands are the agent's commands
            script = None
            if "-c" in argv and argv.index("-c") + 1 < len(argv):
                script = argv[argv.index("-c") + 1]
            else:
                for a in argv[1:]:
                    if not a.startswith("-"):
                        script = known_files.get(resolve(a, cwd))
                        break
                if script is None and u["heredoc"] is not None:
                    script = u["heredoc"]
            if script:
                _DEPTH[0] += 1
                try:
                    sub_ops, _, _ = extract(script, cwd, known_files, persist_cd=False)
                finally:
                    _DEPTH[0] -= 1
                for so in sub_ops:
                    so["via_script"] = True
                new_ops.extend(sub_ops)

        # output redirections of this unit
        for op_, t in u["redirs"]:
            if not path_like(t) and not t.startswith("/tmp"):
                if not t.startswith("/"):
                    continue
            if t.startswith("/dev/"):
                continue
            if op_ in (">", ">|", "&>", "1>"):
                text = _literal_text(units, u) if literal else None
                new_ops.append({"op": "W", "path": resolve(t, cwd), "kind": "overwrite" if literal else "replace",
                                "mech": "redirect", "text": text,
                                "derived_from": [o["path"] for o in new_ops if o["op"] == "R" or o.get("_src")]})
            elif op_ in (">>", "&>>", "1>>"):
                new_ops.append({"op": "W", "path": resolve(t, cwd), "kind": "append", "mech": "redirect",
                                "text": _literal_text(units, u) if literal else None})
        # remember heredoc-created files (scripts, patches)
        if u["heredoc"] is not None:
            for o in new_ops:
                if o["op"] == "W" and o["kind"] == "overwrite" and o["mech"] in ("redirect", "tee"):
                    known_files[o["path"]] = u["heredoc"]
        for o in new_ops:
            o["cmd_i"] = ui
            o["guard"] = guard
            if o.get("via_script"):
                # keep the inner command's verb (e.g. `ed` inside `bash x.sh`),
                # so "ed: command not found" marks it failed
                o["outer_verb"] = verb
                if o.get("verb") is None:
                    o["verb"] = verb
            else:
                o["verb"] = verb
        ops.extend(new_ops)
    if not persist_cd:
        cwd = cwd0
    return ops, cwd, info


def _literal_text(units, u) -> Optional[str]:
    """The literal text flowing into u's pipe (echo/printf args or a heredoc)."""
    for v in units:
        if v["pipe"] != u["pipe"]:
            continue
        argv = strip_prefixes(v["argv"])
        verb = argv[0].split("/")[-1] if argv else ""
        if verb == "cat" and v["heredoc"] is not None:
            return v["heredoc"] + "\n"
        if verb == "echo":
            args = [a for a in argv[1:] if a not in ("-e", "-n", "-E")]
            return " ".join(args) + ("" if "-n" in argv else "\n")
        if verb == "printf":
            return " ".join(argv[1:2]).replace("\\n", "\n")
        if v["heredoc"] is not None and verb in ("tee",):
            return v["heredoc"] + "\n"
    return None


def _selftest():
    cases = [
        ("cd src && sed -i 's/a/b/' x.py", "/repo"),
        ("sed -n '10,20p' a.py", "/repo"),
        ("sed 's/a/b/' a.py > /tmp/o", "/repo"),
        ("cat <<'EOF' > fix.py\nimport re\np='a.py'\ns=open(p).read()\nopen(p,'w').write(s.replace('x = 1','x = 2'))\nEOF\npython3 fix.py", "/repo"),
        ("grep -n foo a.py | head", "/repo"),
        ("grep -c foo a.py", "/repo"),
        ("nl -ba a.py | sed -n '1,40p'", "/repo"),
        ("sed -i '42d;45s/foo/bar/' Makefile", "/repo"),
        ("sed -i '/pattern/a\\    new line' a.py", "/repo"),
        ("perl -0pi -e 's/foo\\n/bar/' a.py", "/repo"),
        ("python3 - <<'PY'\nfrom pathlib import Path\np=Path('a.py')\ns=p.read_text()\np.write_text(s.replace('a','b'))\nPY", "/repo"),
        ("applypatch << 'PATCH'\n*** Begin Patch\n*** Update File: a.py\n@@\n-x = 1\n+x = 2\n*** End Patch\nPATCH", "/repo"),
        ("(cd sub && cat f.txt) && cat g.txt", "/repo"),
        ("head -n 50 a.py", "/repo"),
        ("echo 'x' >> a.py && cat a.py | wc -l", "/repo"),
        ("git checkout -- a.py b.py", "/repo"),
        ("cp a.py a.py.bak && sed -i 's/x/y/' a.py", "/repo"),
    ]
    kf: Dict[str, str] = {}
    for c, cwd in cases:
        ops, cwd2, info = extract(c, cwd, kf)
        print(repr(c[:70]))
        for o in ops:
            print("    ", {k: v for k, v in o.items() if k not in ("guard",)})
        print("     cwd->", cwd2)


if __name__ == "__main__":
    _selftest()
