"""The write-through viewport fix, decoupled from any one agent scaffold.

This is the entire mechanism behind Arm 9 (write-through viewport): after a
step writes a file, compute a small line-numbered window centered on the
edited region and hand it back as text to append to that step's tool output.
Unlike retroactive prompt-rewriting arms, this never touches earlier turns --
the caller appends the returned text to the *current* step's own output,
before it enters the message history, so the transcript stays append-only
and prefix caches stay valid.

Nothing here is specific to mini-SWE-agent or to any particular action
syntax. The three inputs are generic: a workspace directory, the relative
paths a step wrote (however the caller detected that -- a filesystem
snapshot diff is the recommended method: `os.walk` the workspace before and
after the step, compare (mtime_ns, size) per file, and diff the two maps.
Command-text parsing cannot keep up with the real range of ways a step can
write a file: sed -i, an editor tool, a heredoc, a Python script that opens
the file itself), and the file's content immediately before the step, so the
edited region can be located.

Minimal wiring pattern for a new scaffold:

    before = snapshot_workspace(workspace_dir)
    before_contents = {p: read(p) for p in referenced_paths}
    ... run the step ...
    after = snapshot_workspace(workspace_dir)
    writes = sorted(p for p, meta in after.items() if before.get(p) != meta)
    if writes:
        step_output += viewport_echo(workspace_dir, writes, freshness_hook, before_contents)

`freshness_hook` is optional: pass `None` (or any object with an empty
`.bindings` list) if the caller doesn't already track per-file observation
history. Its only use here is as a fallback source for a file's prior
content when the caller didn't capture `before_contents` for a path that
turns out to have been written -- with no fallback available, the viewport
simply centers on the file's midpoint instead of the edited region.
"""

import os
import subprocess
from typing import Dict, List, Optional, Protocol, Tuple

from .tail_diff import split_lines

VIEWPORT_HALF = 25  # lines above and below the edit centre
# Bounds (harness v11). The viewport is appended after the observation template
# has folded the command's output, so nothing else limits it. Without them,
# `npm test` writing coverage files (one 2.85M-character line) and
# `git checkout <rev> -- src/` each produced a single observation of millions
# of characters. The total matches the template's 10,000-character fold.
VIEWPORT_MAX_FILES = 3
VIEWPORT_MAX_LINE = 200
VIEWPORT_MAX_CHARS = 10_000


# never source at any depth; every other skip_dirs entry is skipped at the workspace root only
SKIP_ANYWHERE = {".git", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache", ".tox"}


def snapshot_workspace(workspace_dir: str, skip_dirs: Optional[set] = None) -> Dict[str, Tuple[int, int]]:
    """Map every workspace file to (mtime_ns, size).

    Used to detect writes by diffing before/after a step. Inferring writes
    from command text cannot keep up with the shell surface agents actually
    use (sed -i, python scripts that open(...,'w'), patch, cp, tee, editors),
    and missing a write means a stale binding is never invalidated, which
    silently zeroes out the measurement.
    """
    skip = skip_dirs or {".git", "__pycache__", "node_modules", ".egg-info"}
    snap: Dict[str, Tuple[int, int]] = {}
    for root, dirs, files in os.walk(workspace_dir):
        # build/, dist/, venv/ ... are build trees at the top only: deeper, a directory
        # with that name is source (pip's src/pip/_internal/operations/build/)
        top = os.path.samefile(root, workspace_dir)
        dirs[:] = [d for d in dirs if not (d in SKIP_ANYWHERE or (top and d in skip))]
        for name in files:
            full = os.path.join(root, name)
            try:
                st = os.stat(full)
            except OSError:
                continue
            snap[os.path.relpath(full, workspace_dir)] = (st.st_mtime_ns, st.st_size)
    return snap


def diff_snapshots(before: Dict[str, Tuple[int, int]], after: Dict[str, Tuple[int, int]]) -> List[str]:
    """Files created or modified between two snapshots."""
    return sorted(p for p, meta in after.items() if before.get(p) != meta)


class _BindingsSource(Protocol):
    bindings: List[object]  # each binding needs .file_path and .file_content


def _is_binary(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\0" in f.read(8192)
    except OSError:
        return True


def _ignored(workspace_dir: str, paths: List[str]) -> set:
    """The paths git ignores in this workspace: generated output (coverage,
    build artefacts, dependencies), not file state the agent acts on. Empty
    when the workspace is not a git repository."""
    if not paths:
        return set()
    try:
        res = subprocess.run(["git", "-C", workspace_dir, "check-ignore", "--stdin"], input="\n".join(paths),
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return set()
    return set(res.stdout.split("\n")) if res.returncode in (0, 1) else set()


def _clip(line: str) -> str:
    body = line.rstrip("\n")
    if len(body) <= VIEWPORT_MAX_LINE:
        return line
    tail = "\n" if line.endswith("\n") else ""
    return f"{body[:VIEWPORT_MAX_LINE]} ...[{len(body) - VIEWPORT_MAX_LINE} more characters]{tail}"


def viewport_blocks(
    workspace_dir: str,
    writes: List[str],
    freshness_hook: Optional[_BindingsSource] = None,
    before_contents: Optional[Dict[str, str]] = None,
) -> Tuple[str, List[str]]:
    """The viewport text for a step's writes, and the files it shows.

    A line-numbered window onto each just-written file, centered on the edit.
    `writes` are workspace-relative paths a step wrote (see module docstring
    for how to detect these). `before_contents` maps a subset of those paths
    to their content immediately before the step, used to locate the edited
    region; a path without an entry there falls back to the most recent
    binding in `freshness_hook.bindings` for that path, then to the file's
    midpoint if neither is available.

    At most VIEWPORT_MAX_FILES files are shown, lines are clipped to
    VIEWPORT_MAX_LINE characters and the whole to VIEWPORT_MAX_CHARS; files
    the agent had read come first, and the rest are named in a closing line.
    Binary files are skipped. Only the returned files may be registered as
    seen: a written file the bounds left out was not shown to the agent.
    """
    if not writes:
        return "", []

    bindings = getattr(freshness_hook, "bindings", None) or []
    ignored = _ignored(workspace_dir, list(writes))
    writes = [w for w in writes if w not in ignored]
    known = set(before_contents or ()) | {getattr(b, "file_path", None) for b in bindings}
    blocks: List[str] = []
    shown: List[str] = []
    left_out: List[str] = []
    used = 0
    for rel_path in sorted(writes, key=lambda p: (p not in known, p)):
        abs_path = os.path.join(workspace_dir, rel_path)
        if not os.path.isfile(abs_path) or _is_binary(abs_path):
            continue
        if len(shown) >= VIEWPORT_MAX_FILES:
            left_out.append(rel_path)
            continue
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace", newline="") as f:
                lines = split_lines(f.read(), keepends=True)
        except OSError:
            continue

        total = len(lines)
        # Find the centre of the edit region by diffing against the
        # before-snapshot content. Fall back to the file midpoint.
        centre = total // 2
        old_content = None
        if before_contents and rel_path in before_contents:
            old_content = before_contents[rel_path]
        else:
            for b in reversed(bindings):
                if getattr(b, "file_path", None) == rel_path and getattr(b, "file_content", None) is not None:
                    old_content = b.file_content
                    break

        if old_content is not None:
            old_lines = split_lines(old_content, keepends=True)
            first_diff = None
            for i, (a, b) in enumerate(zip(old_lines, lines)):
                if a != b:
                    first_diff = i
                    break
            if first_diff is None and len(old_lines) != total:
                first_diff = min(len(old_lines), total)
            if first_diff is not None:
                centre = first_diff

        half = VIEWPORT_HALF
        if total <= half * 2:
            start, end = 0, total
        else:
            start = max(0, centre - half)
            end = min(total, start + half * 2)
            if end - start < half * 2:
                start = max(0, end - half * 2)

        viewport = "".join(f"{i}\t{_clip(line)}" for i, line in enumerate(lines[start:end], start=start + 1))
        block = f"[VIEWPORT (lines {start + 1}-{end} of {total}): {rel_path}]\n{viewport}"
        if used + len(block) > VIEWPORT_MAX_CHARS:
            left_out.append(rel_path)
            continue
        blocks.append(block)
        shown.append(rel_path)
        used += len(block)

    if left_out:
        names = ", ".join(left_out[:10]) + (", ..." if len(left_out) > 10 else "")
        blocks.append(f"[{len(left_out)} more written files not shown: {names}]")
    if not blocks:
        return "", []
    return "\n\n" + "\n\n".join(blocks), shown


def viewport_echo(
    workspace_dir: str,
    writes: List[str],
    freshness_hook: Optional[_BindingsSource] = None,
    before_contents: Optional[Dict[str, str]] = None,
) -> str:
    """The text of `viewport_blocks` alone."""
    return viewport_blocks(workspace_dir, writes, freshness_hook, before_contents)[0]
