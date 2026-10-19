"""Dependency write-through: after a write, name the uses it left stale elsewhere.

File-level write-through shows the agent the file it just wrote. It cannot
show that another file, whose bytes did not change, still uses a name the
write removed or a function whose parameters it changed: the agent's view of
that other file matches the disk and is still wrong. This module finds those
uses and formats a short, append-only notice for the step's own output.

Definitions are found with per-language line scanners rather than a parser
(no tree-sitter in the harness). A scanner reads only declarations at module
level and members one level inside a module-level class, interface, object or
struct; anything nested deeper is a local and is ignored. A name is reported
when the write removed its definition or made its old call sites invalid, and
a word match for it remains outside the written file.
"""

import os
import re
import functools
import subprocess
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

# 2: a name other files have always defined (another type's method) is no
# longer treated as moved; its uses in the written file's directory are listed.
# 3: JS/TS class members, object methods and `X.y = function`; function-body
# locals are no longer definitions (JS/TS, Python, Go); a name the agent added
# in this run is followed only into files it wrote; uses are ranked by whether
# their file imports the written file or names the owner; a parameter change
# is one to the parameter names, and a Go return count or field type counts.
NOTICE_VERSION = 3
MAX_SYMBOLS = 4
MAX_CANDIDATES = 12
MAX_USES_PER_SYMBOL = 4
MAX_CHARS = 2_000
MIN_NAME = 3
# Above this many uses a word match says little: only files tied to the
# written file (same directory, importing it, naming the owner) are listed.
MAX_HITS = 40
MAX_DEFINING = 25
JOIN_LINES = 12
# Seconds for all git calls of one notice: hooks are killed at about 30 s, and a
# timed-out search must not take the rest of the write-through with it.
TIME_BUDGET = 10.0
_deadline = float("inf")


def _timeout(cap: float = 20.0) -> Optional[float]:
    """Seconds left for one git call, or None when the notice's budget is spent."""
    left = _deadline - time.monotonic()
    return min(cap, left) if left > 0.05 else None

Key = Tuple[str, str]  # (owner, name); owner is "" at module level
# What a declaration promises its users: ("fn", parameters, what it returns),
# ("field", "", its type) or ("name", "", "") for anything else. Returns and
# field types are read for Go only, where a changed one breaks the build.
Sig = Tuple[str, str, str]
NAME: Sig = ("name", "", "")

# Names too generic to report: a word match on them says nothing.
STOP = {"init", "main", "test", "setup", "self", "this", "config", "data", "name", "value", "type",
        "error", "result", "default", "options", "props", "state", "__init__", "__main__", "run", "get",
        "set", "new", "len", "String", "Error", "Close", "Name", "start", "end", "key", "index", "item",
        "render", "toString", "constructor", "log", "logger"}
_JS_KEYWORDS = set("if for while switch catch with function return typeof await new else do super "
                   "constructor delete void throw case yield import export in of try finally".split())


def _balanced(lines: List[str], i: int) -> str:
    """Line i, joined with the following lines until its parentheses balance."""
    text = lines[i].strip()
    j = i
    while text.count("(") > text.count(")") and j + 1 < len(lines) and j - i < JOIN_LINES:
        j += 1
        text += " " + lines[j].strip()
    return text


def _params(text: str, open_at: int) -> str:
    """The parameter text of the call or signature whose '(' is at open_at."""
    depth = 0
    for k in range(open_at, len(text)):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                return text[open_at + 1:k].strip()
    return text[open_at + 1:].strip()


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


# --- Python -------------------------------------------------------------

_PY_DEF = re.compile(r"(?:async\s+)?def\s+(\w+)\s*\(")
_PY_CLASS = re.compile(r"class\s+(\w+)")
_PY_ASSIGN = re.compile(r"([A-Za-z_]\w*)\s*(?::[^=]+)?=(?!=)")


def _scan_python(lines: List[str]) -> Dict[Key, Sig]:
    out: Dict[Key, Sig] = {}
    owner, member_indent = "", None
    for i, line in enumerate(lines):
        body = line.strip()
        if not body or body.startswith(("#", "@")):
            continue
        ind = _indent(line)
        if ind == 0:
            owner, member_indent = "", None
            m = _PY_CLASS.match(body)
            if m:
                owner = m.group(1)
                out.setdefault(("", owner), NAME)
                continue
        elif owner and member_indent is None:
            member_indent = ind
        if ind == 0 or (owner and ind == member_indent):
            m = _PY_DEF.match(body)
            if m:
                joined = _balanced(lines, i)
                out.setdefault((owner if ind else "", m.group(1)), ("fn", _params(joined, joined.index("(")), ""))
                continue
            m = _PY_ASSIGN.match(body)  # a module constant or a class attribute
            if m:
                out.setdefault((owner if ind else "", m.group(1)), NAME)
    return out


# --- Go -----------------------------------------------------------------

_GO_FUNC = re.compile(r"func\s+(?:\(\s*\w*\s*\*?\s*(\w+)[^)]*\)\s*)?(\w+)\s*(?:\[[^\]]*\])?\s*\(")
_GO_TYPE = re.compile(r"type\s+(\w+)(?:\s+(struct|interface)\s*\{\s*$)?")
_GO_VAR = re.compile(r"(?:var|const)\s+(\w+)")
_GO_BLOCK = re.compile(r"(?:var|const)\s*\(\s*$")
_GO_FIELD = re.compile(r"\t([A-Z]\w*)\s+([\w\*\[\]\.{}]+)\s*(?:`[^`]*`)?\s*(?://.*)?$")
_GO_IFACE_METHOD = re.compile(r"\t(\w+)\s*\(")
_GO_ENTRY = re.compile(r"\t(\w+)\b")


def _go_fn(joined: str, open_at: int) -> Sig:
    params = _params(joined, open_at)
    rest = joined[open_at + len(params) + 2:] if joined[open_at + 1:].startswith(params) else ""
    close = joined.find(")", open_at + 1 + len(params)) if not rest else -1
    if close >= 0:
        rest = joined[close + 1:]
    return ("fn", params, rest.split("{")[0].strip())


def _scan_go(lines: List[str]) -> Dict[Key, Sig]:
    out: Dict[Key, Sig] = {}
    block, owner = "", ""  # block: "", "struct", "interface" or "vars"
    for i, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        if line[0] not in " \t":
            block, owner = "", ""
            m = _GO_FUNC.match(line)
            if m:
                out.setdefault((m.group(1) or "", m.group(2)), _go_fn(_balanced(lines, i), m.end() - 1))
                continue
            m = _GO_TYPE.match(line)
            if m:
                out.setdefault(("", m.group(1)), NAME)
                block, owner = (m.group(2) or ""), m.group(1)
                continue
            if _GO_BLOCK.match(line):
                block = "vars"
                continue
            m = _GO_VAR.match(line)
            if m:
                out.setdefault(("", m.group(1)), NAME)
            continue
        if block == "struct":
            m = _GO_FIELD.match(line)
            if m:
                out.setdefault((owner, m.group(1)), ("field", "", m.group(2)))
        elif block == "interface":
            m = _GO_IFACE_METHOD.match(line)
            if m:
                joined = _balanced(lines, i)
                out.setdefault((owner, m.group(1)), _go_fn(joined, joined.index("(")))
        elif block == "vars":
            m = _GO_ENTRY.match(line)
            if m and line.startswith("\t") and not line.startswith("\t\t"):
                out.setdefault(("", m.group(1)), NAME)
    return out


# --- JavaScript / TypeScript -------------------------------------------

_MODS = r"(?:(?:public|private|protected|static|readonly|override|abstract|async|declare)\s+)*"
_NAME = r"(#?[A-Za-z_$][\w$]*)"
_ARROW = (r"(?:async\s+)?(?:function\b|(?:<[^>()]*>\s*)?\((?:[^()]|\([^()]*\))*\)\s*(?::(?:[^=]|=>)+?)?=>"
          r"|[A-Za-z_$][\w$]*\s*=>)")
_JS_FUNC = re.compile(r"(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:async\s+)?function\s*\*?\s*(\w+)\s*(?:<[^>()]*>)?\s*\(")
_JS_CLASS = re.compile(r"(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?(?:class|interface)\s+(\w+)")
_JS_TYPE = re.compile(r"(?:export\s+)?(?:declare\s+)?(?:const\s+)?(?:type|enum|namespace)\s+(\w+)")
_JS_VAR = re.compile(r"(?:export\s+)?(?:declare\s+)?(?:const|let|var)\s+(\w+)\s*[=:]")
_JS_OBJECT = re.compile(r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::[^=]+)?=\s*\{\s*$")
_JS_EXPORTS = re.compile(r"(?:module\.exports|exports\.\w+|export\s+default)\s*=?\s*\{\s*$")
# A function that wraps the whole module: its body is the module's top level.
_JS_WRAPPER = re.compile(r"(?:module\.exports\s*=\s*(?:async\s+)?function|define\s*\(|;?\(\s*(?:async\s+)?function|;?\(\s*\(\s*\)\s*=>)")
_JS_MEMBER = re.compile(_MODS + r"(?:(?:get|set)\s+(?=[#\w$]))?\*?\s*" + _NAME + r"\s*\??\s*(?:<[^>()]*>)?\s*\(")
_JS_FIELD_FN = re.compile(_MODS + _NAME + r"\s*[?!]?\s*(?::(?:[^=]|=>)+?)?=\s*" + _ARROW)
_JS_KEY_FN = re.compile(r"['\"]?([A-Za-z_$][\w$]*)['\"]?\s*:\s*" + _ARROW)
_JS_ASSIGN_FN = re.compile(r"((?:[A-Za-z_$][\w$]*\.)+)" + _NAME + r"\s*=\s*" + _ARROW)
_JS_DECORATOR = re.compile(r"(?:@[\w.$]+(?:\([^)]*\))?\s+)+")
_JS_COMMENT = ("//", "/*", "*")


def _js_unit(lines: List[str]) -> int:
    """The file's indentation step: its smallest indent on a code line."""
    found = [_indent(ln) for ln in lines if ln.strip() and ln[0] in " \t" and not ln.strip().startswith(_JS_COMMENT)]
    return min(found) if found else 1


def _fn_params(joined: str, m: "re.Match") -> str:
    at = joined.find("(", m.start())
    return _params(joined, at) if at >= 0 else ""


_JS_ES_MODULE = re.compile(r"(?:export|import)\b")
_JS_EXPORT_LIST = re.compile(r"export\s*(?:default\s+([\w$]+)\s*;?\s*$|\{([^}]*)\})")
_JS_REQUIRE = re.compile(r"(?:const|let|var)\s+[\w${},\s:]+=\s*require\s*\(")


def _js_private(lines: List[str], out: Dict[Key, Sig], declared: Dict[str, bool]) -> Set[str]:
    """Module-level names of an ES module that it does not export: no other
    file can refer to them. A CommonJS or AMD file exports by assignment, which
    a line scanner cannot follow, so nothing is dropped there."""
    if not any(_JS_ES_MODULE.match(ln) for ln in lines):
        return set()
    exported = {name for name, flag in declared.items() if flag}
    for ln in lines:
        m = _JS_EXPORT_LIST.match(ln)
        if m:
            exported |= {part.split(" as ")[0].strip() for part in (m.group(1) or m.group(2) or "").split(",")}
    return {name for name in declared if name not in exported}


def _scan_js(lines: List[str]) -> Dict[Key, Sig]:
    out: Dict[Key, Sig] = {}
    declared: Dict[str, bool] = {}  # module-level name -> declared with `export`
    unit = _js_unit(lines)
    wrapped = any(_JS_WRAPPER.match(ln) for ln in lines)
    owner, in_owner, member_indent = "", False, None
    for i, line in enumerate(lines):
        body = line.strip()
        if not body or body.startswith(_JS_COMMENT):
            continue
        ind = _indent(line)
        if ind == 0:
            if body.startswith("}"):
                owner, in_owner, member_indent = "", False, None
                continue
            if _JS_REQUIRE.match(body):
                continue  # an import, whatever it is called here
            m = _JS_CLASS.match(body)
            if m:
                owner, in_owner, member_indent = m.group(1), True, None
                out.setdefault(("", owner), NAME)
                declared.setdefault(owner, body.startswith("export"))
                continue
            m = _JS_OBJECT.match(body)
            if m:
                owner, in_owner, member_indent = m.group(1), True, None
                out.setdefault(("", owner), NAME)
                declared.setdefault(owner, body.startswith("export"))
                continue
            if _JS_EXPORTS.match(body):
                owner, in_owner, member_indent = "", True, None
                continue
            m = _JS_FUNC.match(body)
            if m:
                joined = _balanced(lines, i)
                out.setdefault(("", m.group(1)), ("fn", _fn_params(joined, m), ""))
                declared.setdefault(m.group(1), body.startswith("export"))
                continue
            m = _JS_TYPE.match(body) or _JS_VAR.match(body)
            if m:
                declared.setdefault(m.group(1), body.startswith("export"))
                joined = _balanced(lines, i)
                fn = _JS_FIELD_FN.match(re.sub(r"^(?:export\s+)?(?:const|let|var)\s+", "", joined))
                out.setdefault(("", m.group(1)), ("fn", _fn_params(joined, fn), "") if fn else NAME)
                continue
        if in_owner and member_indent is None and ind > 0:
            member_indent = ind
        if in_owner and ind == member_indent:
            joined = _JS_DECORATOR.sub("", _balanced(lines, i), count=1)
            m = _JS_FIELD_FN.match(joined) or _JS_KEY_FN.match(joined) or _JS_MEMBER.match(joined)
            if m and m.group(1) not in _JS_KEYWORDS:
                at = joined.find("(", m.end(1))
                out.setdefault((owner, m.group(1).lstrip("#")), ("fn", _params(joined, at) if at >= 0 else "", ""))
            continue
        if ind <= unit and not in_owner:
            joined = _balanced(lines, i)
            m = _JS_ASSIGN_FN.match(joined)
            if m:
                path = [p for p in m.group(1).rstrip(".").split(".") if p not in ("prototype", "module", "exports")]
                if path[:1] != ["this"]:
                    at = joined.find("(", m.end(2))
                    out.setdefault((path[-1] if path else "", m.group(2)), ("fn", _params(joined, at) if at >= 0 else "", ""))
                continue
            if wrapped and ind == unit:
                m = _JS_FUNC.match(body)
                if m:
                    out.setdefault(("", m.group(1)), ("fn", _fn_params(joined, m), ""))
    private = _js_private(lines, out, declared)
    return {k: v for k, v in out.items() if k[1] not in private and k[0] not in private}


SCANNERS = {".py": _scan_python, ".go": _scan_go, ".js": _scan_js, ".jsx": _scan_js, ".ts": _scan_js,
            ".tsx": _scan_js, ".mjs": _scan_js, ".cjs": _scan_js}


def scan(text: Optional[str], path: str) -> Dict[Key, Sig]:
    """(owner, name) -> signature, for the declarations other files can refer to."""
    scanner = SCANNERS.get(os.path.splitext(path)[1])
    if not scanner or text is None:
        return {}
    return {k: v for k, v in scanner(text.splitlines()).items() if len(k[1]) >= MIN_NAME and k[1] not in STOP}


def definitions(text: Optional[str], path: str) -> Dict[str, str]:
    """Defined name -> its parameter text ('' for non-callables)."""
    out: Dict[str, str] = {}
    for (_, name), sig in scan(text, path).items():
        out.setdefault(name, sig[1])
    return out


def receivers(text: Optional[str], path: str) -> Dict[str, str]:
    """Member name -> the class, object or receiver type that owns it."""
    out: Dict[str, str] = {}
    for (owner, name) in scan(text, path):
        if owner:
            out.setdefault(name, owner)
    return out


def _flat(text: str) -> str:
    for _ in range(4):  # nested generics and brackets
        text = re.sub(r"<[^<>]*>|\[[^\[\]]*\]|\([^()]*\)|\{[^{}]*\}", "", text)
    return text


def _param_names(params: str) -> List[str]:
    """Parameter names in order. A changed type or default is not a changed
    name; an added, dropped, renamed or reordered parameter is."""
    names = []
    for part in _flat(params).split(","):
        part = part.strip()
        m = re.match(r"(?:\*{1,2}|\.\.\.)?\s*([A-Za-z_$][\w$]*)", part)
        if part and part not in ("self", "cls"):
            names.append(m.group(1) if m else part)
    return names


def _compatible(params: str, new_params: str) -> bool:
    """Whether every old call still binds: the old parameters come first, in order, and
    each added one has a default or is variadic (`*args`, `**kw`, a bare `*`)."""
    old, new = _param_names(params), _param_names(new_params)
    parts = [p.strip() for p in _flat(new_params).split(",") if p.strip() and p.strip() not in ("self", "cls")]
    if parts and all(p.startswith("*") for p in parts):
        return True  # (*args, **kw) takes any call
    return new[:len(old)] == old and all("=" in p or p.startswith(("*", "...")) for p in parts[len(old):])


_PY_GUARD = re.compile(r"(?:if|elif|else|try|except|finally|with)\b")


def _py_still_defined(text: Optional[str], name: str) -> bool:
    """Whether Python `text` still defines module-level `name` inside an if/try/with
    block (a version or import fallback), which the line scanner does not read."""
    lines = (text or "").split("\n")
    n = re.escape(name)
    defines = re.compile(rf"(?:async\s+)?def\s+{n}\s*\(|class\s+{n}\b|{n}\s*(?::[^=]+)?=(?!=)"
                         rf"|from\s+\S+\s+import\s+.*\b{n}\b|import\s+.*\b{n}\b")
    for i, line in enumerate(lines):
        ind = _indent(line)
        if not ind or not defines.match(line.strip()):
            continue
        # the block opener this line sits in, all the way out to column 0
        openers = []
        for prev in reversed(lines[:i]):
            if prev.strip() and not prev.lstrip().startswith("#") and _indent(prev) < ind:
                openers.append(prev.strip())
                ind = _indent(prev)
                if ind == 0:
                    break
        if openers and all(_PY_GUARD.match(o) for o in openers):
            return True
    return False


def _returns(result: str) -> int:
    """How many values a Go function returns."""
    result = result.strip()
    if not result:
        return 0
    return _flat(result[1:-1]).count(",") + 1 if result.startswith("(") else 1


def changes(before: Optional[str], after: Optional[str], path: str) -> List[Tuple[str, str, str]]:
    """(owner, name, how) for declarations the write removed or changed under
    their users: a parameter added, dropped, renamed or reordered; for Go, a
    different number of return values or a field's type. A rename shows up as
    the removal of the old name. Module-level names come before members."""
    old, new = scan(before, path), scan(after, path)
    out = [(o, n, "removed") for (o, n) in old if (o, n) not in new
           and not (path.endswith(".py") and not o and _py_still_defined(after, n))]
    for key in old:
        if key not in new or old[key] == new[key]:
            continue
        (kind, params, result), (new_kind, new_params, new_result) = old[key], new[key]
        if kind == new_kind == "fn" and _param_names(params) != _param_names(new_params) \
                and not _compatible(params, new_params):
            out.append((*key, f"parameters ({params}) -> ({new_params})"))
        elif kind == new_kind == "fn" and _returns(result) != _returns(new_result):
            out.append((*key, f"returns {result or 'nothing'} -> {new_result or 'nothing'}"))
        elif kind == new_kind == "field" and result != new_result:
            out.append((*key, f"type {result} -> {new_result}"))
    return sorted(out, key=lambda c: bool(c[0]))


def changed_symbols(before: Optional[str], after: Optional[str], path: str) -> List[Tuple[str, str]]:
    """(name, how), without the owner."""
    return [(name, how) for _, name, how in changes(before, after, path)]


def _defines(line: str, path: str, name: str) -> bool:
    """Whether one line, read without its file, declares `name`."""
    body = line.strip()
    ext = os.path.splitext(path)[1]
    n = re.escape(name)
    if ext == ".py":
        return bool(re.match(rf"(?:async\s+)?def\s+{n}\s*\(|class\s+{n}\b|{n}\s*(?::[^=]+)?=(?!=)", body))
    if ext == ".go":
        return bool(re.match(rf"func\s+(?:\([^)]*\)\s*)?{n}\s*[\[(]|type\s+{n}\b|(?:var|const)\s+{n}\b", body))
    if ext not in SCANNERS:
        return False
    body = _JS_DECORATOR.sub("", body, count=1)
    for pat in (_JS_FUNC, _JS_CLASS, _JS_TYPE, _JS_VAR):
        m = pat.match(body)
        if m and m.group(1) == name:
            return True
    m = _JS_ASSIGN_FN.match(body)
    if m and m.group(2) == name:
        return True
    m = _JS_FIELD_FN.match(body) or _JS_KEY_FN.match(body)
    if m and m.group(1).lstrip("#") == name:
        return True
    m = _JS_MEMBER.match(body)
    return bool(m and m.group(1).lstrip("#") == name and body.endswith("{"))


# Not searched: what the harness itself writes, backups an agent leaves beside
# a file, and prose. A stale name there breaks nothing.
NOT_SEARCHED = ["patch.txt", ".materialized", "*.bak", "*.backup", "*.orig", "*.rej", "*.md", "*.rst", "*.txt",
                "*.log", "*.lock", "*.map", "*.snap", "*.min.js"]


# The provisioned environment and what the agent's HOME accumulates in the
# workspace (run_phase2_experiment.HARNESS_PREFIXES), plus vendored packages:
# untracked, so `--untracked` would search them.
NOT_SEARCHED_DIRS = [".venv-task", ".cache", ".local", ".config", ".npm", "node_modules", "vendor"]


@functools.lru_cache(maxsize=64)
def _in_git(workspace: str) -> bool:
    try:
        return subprocess.run(["git", "-C", workspace, "rev-parse", "--is-inside-work-tree"],
                              capture_output=True, timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _git_grep(workspace: str, args: List[str]) -> List[str]:
    timeout = _timeout()
    if timeout is None:
        return []
    # outside a repository git grep searches the directory itself (--no-index)
    scope = "--untracked" if _in_git(workspace) else "--no-index"
    try:
        res = subprocess.run(["git", "-C", workspace, "grep", "-I", scope, *args, "--", ".",
                              *[f":(exclude,glob)**/{pat}" for pat in NOT_SEARCHED],
                              *[f":(exclude,glob){d}/**" for d in NOT_SEARCHED_DIRS],
                              *[f":(exclude,glob)**/{d}/**" for d in ("node_modules", "vendor")]],
                             capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return res.stdout.splitlines()


# Where a name declared in one language can be used: that language's sources,
# and the templates rendered from them.
_JS_FAMILY = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".svelte", ".tpl", ".html", ".hbs", ".ejs"}
FAMILIES = {".py": {".py", ".pyi", ".html", ".j2", ".jinja"}, ".go": {".go", ".tmpl"},
            **{ext: _JS_FAMILY for ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")}}
# A backslash only as an escape pair (a `.` that could also take it backtracks
# exponentially on a long line of escapes).
_STRING = re.compile(r"(['\"`])(?:\\.|(?!\1)[^\\])*\1")
_COMMENT = {".py": re.compile(r"#.*$")}
_SLASH_COMMENT = re.compile(r"//.*$")


def _in_code(text: str, path: str, name: str) -> bool:
    """Whether a source line names `name` outside its strings and comments.
    A template is text throughout, so any mention counts there."""
    if os.path.splitext(path)[1] not in SCANNERS:
        return True
    bare = _COMMENT.get(os.path.splitext(path)[1], _SLASH_COMMENT).sub("", _STRING.sub("", text))
    return bool(re.search(rf"\b{re.escape(name)}\b", bare))


def _grep(workspace: str, name: str, rel: str) -> List[Tuple[str, int, str]]:
    """Lines outside `rel`, in its language family, that use the word `name`."""
    family = FAMILIES.get(os.path.splitext(rel)[1], set())
    hits = []
    for line in _git_grep(workspace, ["-n", "-w", "-e", name]):
        path, _, rest = line.partition(":")
        num, _, text = rest.partition(":")
        if num.isdigit() and path != rel and os.path.splitext(path)[1] in family and _in_code(text, path, name):
            hits.append((path, int(num), text))
    return hits


def _importers(workspace: str, rel: str) -> Set[str]:
    """Files with an import-like line naming the written file's module."""
    stem = os.path.splitext(os.path.basename(rel))[0]
    if rel.endswith(".go") or not stem:
        return set()  # Go imports a directory, not a file
    # One line that has both the module's name and an import word; --and keeps
    # this to fixed strings, which every git build matches the same way.
    return set(_git_grep(workspace, ["-l", "-w", "-F", "-e", stem, "--and", "(", "-e", "import", "-e", "require",
                                     "-e", "from", "-e", "include", ")"]))


def dependency_notice(workspace: str, writes: Iterable[str], before: Dict[str, str],
                      viewed: Optional[Set[str]] = None,
                      written: Optional[Set[str]] = None) -> Tuple[str, List[Dict[str, object]]]:
    """The notice text, and a record of what it reported, for the trace.

    `before` maps each written file to its content before this write;
    `written` is every file the agent has written in this run.
    """
    global _deadline
    _deadline = time.monotonic() + TIME_BUDGET
    viewed = viewed or set()
    written = set(written or ()) | set(writes)
    blocks: List[str] = []
    record: List[Dict[str, object]] = []
    reported: Dict[str, Set[str]] = {}  # name -> files its uses are in
    used = 0
    for rel in writes:
        if rel not in before:
            continue
        try:
            with open(os.path.join(workspace, rel), "r", encoding="utf-8", errors="replace") as f:
                after = f.read()
        except OSError:
            after = ""
        changed = changes(before[rel], after, rel)[:MAX_CANDIDATES]
        if not changed:
            continue
        was = scan(before[rel], rel)
        at_head = scan(committed(workspace, rel), rel)
        here = os.path.dirname(rel)
        importers: Optional[Set[str]] = None
        for owner, name, how in changed:
            if len(record) >= MAX_SYMBOLS:
                break
            if name in reported:
                continue  # said once, under another owner
            hits = _grep(workspace, name, rel)
            uses = [h for h in hits if not _defines(h[2], h[0], name)]  # a definition is not a use
            if rel.endswith(".go") and not (name[:1].isupper() and (not owner or owner[:1].isupper())):
                uses = [h for h in uses if os.path.dirname(h[0]) == here]  # unexported: its package only
            # Files that declare the name now, split by whether they did at the last commit.
            new_homes, always = [], set()
            for p in sorted({p for p, _, t in hits if _defines(t, p, name)})[:MAX_DEFINING]:
                if name in definitions(_read(workspace, p), p):
                    (always.add if name in definitions(committed(workspace, p), p) else new_homes.append)(p)
            # A file that declares the name itself uses its own.
            uses = [h for h in uses if h[0] not in always and h[0] not in new_homes]
            qualified: Set[Tuple[str, int]] = set()
            if owner:
                # A member is used through a receiver (`x.name`) or, for a Go
                # field, set by key (`name:`); a bare word is something else.
                key_form = rf"|\b{re.escape(name)}\s*:" if rel.endswith(".go") else ""
                as_member = re.compile(rf"\.\s*{re.escape(name)}\b{key_form}")
                uses = [h for h in uses if as_member.search(h[2])]
                # `Owner.name`, or `module.name` for a module object named after its file.
                stem_q = os.path.splitext(os.path.basename(rel))[0]
                exact = re.compile(rf"\b(?:{re.escape(owner)}|{re.escape(stem_q)})\s*\.\s*{re.escape(name)}\b",
                                   re.IGNORECASE)
                qualified = {(p, n) for p, n, t in uses if exact.search(t)}
            if importers is None:
                importers = _importers(workspace, rel)
            owner_files = set(_git_grep(workspace, ["-l", "-w", "-e", owner])) if owner else set()
            related = {p for p, _, _ in uses if os.path.dirname(p) == here or p in importers or p in owner_files}
            label = f"{owner}.{name}" if owner else name
            note = ""
            if (owner, name) not in at_head and _in_git(workspace):
                # The agent added this name during the run: only files it has
                # written since can use it. (Outside git there is no record of
                # what the run added.)
                uses = [h for h in uses if h[0] in written]
            if how == "removed" and new_homes:
                home = new_homes[0]
                same_owner = not owner or receivers(_read(workspace, home), home).get(name) == owner
                if same_owner and rel.endswith(".go") and os.path.dirname(home) == here:
                    continue  # still in the package: every use resolves
                if same_owner:
                    # The definition moved: uses that still reach it through
                    # the old module are the stale ones.
                    uses = [h for h in uses if h[0] in importers] if not rel.endswith(".go") else \
                        [h for h in uses if f"{os.path.basename(here)}.{name}" in h[2]]
                    how = f"moved to {home}"
                else:
                    note = f" (a `{name}` is now defined in {home})"
            elif always:
                # Other files have always defined this name (another type's
                # method, say): a word match cannot tell whose use it is. A
                # member is reported where it is named with its owner or in
                # the written file's directory; any other name, in files tied
                # to the written one.
                if owner and rel.endswith(".go"):
                    uses = [h for h in uses if (h[0], h[1]) in qualified or os.path.dirname(h[0]) == here]
                elif owner:  # a directory is not a unit outside Go: the file must name the owner
                    uses = [h for h in uses if (h[0], h[1]) in qualified or h[0] in owner_files]
                else:
                    uses = [h for h in uses if h[0] in related]
                note = f" (other files define a `{name}` too; check whose uses these are)"
            if len(uses) > MAX_HITS:
                uses = [h for h in uses if (h[0], h[1]) in qualified or os.path.dirname(h[0]) == here]
            if how == "removed" and owner in reported and {h[0] for h in uses} <= reported[owner]:
                continue  # its owner's removal, already reported, points at the same files
            if not uses:
                continue
            stem = os.path.splitext(os.path.basename(rel))[0]
            called = re.compile(rf"\b{re.escape(name)}\s*(?:<[^<>()]*>)?\s*\(")
            is_fn = was.get((owner, name), NAME)[0] == "fn"
            uses.sort(key=lambda h: (is_fn and not called.search(h[2]),  # a function is used by calling it
                                     (h[0], h[1]) not in qualified,
                                     h[0] not in importers and h[0] not in owner_files,
                                     not os.path.basename(h[0]).startswith(stem), h[0] not in viewed, h[0], h[1]))
            shown = uses[:MAX_USES_PER_SYMBOL]
            lines = [f"{p}:{n}: {t.strip()[:120]}" for p, n, t in shown]
            more = f"\n  ... and {len(uses) - len(shown)} more" if len(uses) > len(shown) else ""
            block = f"{rel}: `{label}` {how}{note}; still used at:\n  " + "\n  ".join(lines) + more
            if used + len(block) > MAX_CHARS:
                break
            blocks.append(block)
            reported[name] = {p for p, _, _ in uses}
            used += len(block)
            record.append({"file": rel, "symbol": name, "owner": owner, "change": how + note, "uses": len(uses),
                           "shown": [f"{p}:{n}" for p, n, _ in shown],
                           "use_files": sorted({p for p, _, _ in uses})[:20]})
    if not blocks:
        return "", []
    return "\n\n[DEPENDENCY NOTICE: uses in other files that this write may have left stale]\n" + \
        "\n".join(blocks), record


def _read(workspace: str, rel: str) -> Optional[str]:
    try:
        with open(os.path.join(workspace, rel), "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def committed(workspace: str, rel: str) -> Optional[str]:
    """The file as committed, the baseline for a file first written unseen."""
    timeout = _timeout()
    if timeout is None:
        return None
    try:
        res = subprocess.run(["git", "-C", workspace, "show", f"HEAD:{rel}"], capture_output=True,
                             text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return res.stdout if res.returncode == 0 else None
