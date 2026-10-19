"""The tail-diff coherence mechanism, decoupled from any one agent scaffold.

Unlike full-file JIT Tail Refresh (Arm 7), which dumps the entire file (2k-4.2k
lines, $O(N)$ tokens) and rewrites earlier turns in the history (annihilating
the KV-cache prefix), this mechanism:
1. Is strictly append-only: past conversation turns are never modified,
   preserving 100% of the prefix KV-cache in local engines (vLLM, SGLang) and APIs.
2. Computes the net unified diff or line-numbered diff hunk between what the
   agent last saw (held view) and the current live file on disk ($O(\\Delta)$ tokens).
3. Naturally captures multiple disjoint hunks across large files (solving the
   single-window limitation of Arm 9 Write-Through Viewport, which misses edits
   outside a +/- 25 line window).
4. Handles sequential conflicting overwrites by collapsing intermediate mutations
   into the exact net diff from the agent's held view to live disk.
"""

import copy
import difflib
import os
from typing import Any, Dict, List, Optional, Protocol, Tuple

TAIL_DIFF_CONTEXT = 3
TAIL_DIFF_MAX_FILES = 3
TAIL_DIFF_MAX_LINE = 200
TAIL_DIFF_MAX_CHARS = 10_000
EXACT_LINES = 2000    # a changed stretch up to this many lines is diffed exactly (autojunk off)
MATCH_LINES = 60_000  # a longer one is shown as one replaced stretch: SequenceMatcher is near-quadratic


class _BindingsSource(Protocol):
    bindings: List[object]  # each binding needs .file_path and .file_content


def _is_binary(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\0" in f.read(8192)
    except OSError:
        return True


def _clip(line: str) -> str:
    tail = "\n" if line.endswith("\n") else ""
    body = line.rstrip("\n").rstrip("\r")  # a CRLF file's \r is not text the model can use
    if len(body) <= TAIL_DIFF_MAX_LINE:
        return body + tail
    return f"{body[:TAIL_DIFF_MAX_LINE]} ...[{len(body) - TAIL_DIFF_MAX_LINE} more characters]{tail}"


def split_lines(text: str, keepends: bool = False) -> List[str]:
    """Lines as sed, grep -n and editors number them: split at "\\n" only. str.splitlines also
    splits at \\r, \\f, \\v, \\x1c-\\x1e, \\x85, U+2028 and U+2029, which shifts every later number."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if keepends:
        lines = [l + "\n" for l in lines]
        if lines and not text.endswith("\n"):
            lines[-1] = lines[-1][:-1]
    return lines


def opcodes(a: List[str], b: List[str]) -> List[Tuple[str, int, int, int, int]]:
    """difflib opcodes for a -> b in bounded time: the common head and tail are matched directly,
    and only the changed middle goes through SequenceMatcher (one replaced stretch if it is huge)."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    j = 0
    while j < n - i and a[len(a) - 1 - j] == b[len(b) - 1 - j]:
        j += 1
    ma, mb = a[i:len(a) - j], b[i:len(b) - j]
    ops = [("equal", 0, i, 0, i)] if i else []
    if not ma and mb:
        ops.append(("insert", i, i, i, i + len(mb)))
    elif ma and not mb:
        ops.append(("delete", i, i + len(ma), i, i))
    elif ma and mb:
        if max(len(ma), len(mb)) > MATCH_LINES:
            mid = [("replace", 0, len(ma), 0, len(mb))]
        else:
            mid = difflib.SequenceMatcher(None, ma, mb, autojunk=max(len(ma), len(mb)) > EXACT_LINES).get_opcodes()
        ops += [(t, i1 + i, i2 + i, j1 + i, j2 + i) for t, i1, i2, j1, j2 in mid]
    if j:
        ops.append(("equal", len(a) - j, len(a), len(b) - j, len(b)))
    return ops


def _held(binding) -> Optional[str]:
    """The content a binding holds; an empty file is content too, not a reason to use raw_content."""
    content = getattr(binding, "file_content", None)
    return content if content is not None else getattr(binding, "raw_content", None)


def format_line_numbered_diff(
    held_content: str,
    disk_content: str,
    filename: str,
    context: int = TAIL_DIFF_CONTEXT,
    step_seen: Optional[int] = None,
    current_step: Optional[int] = None,
) -> str:
    """Format a diff between held content and live disk with explicit live line numbers."""
    held_lines = split_lines(held_content)
    disk_lines = split_lines(disk_content)

    step_info = f" (held at Step {step_seen})" if step_seen is not None else ""
    curr_info = f" (Step {current_step})" if current_step is not None else ""
    
    blocks: List[str] = []
    for tag, alo, ahi, blo, bhi in opcodes(held_lines, disk_lines):
        if tag == "equal":
            continue

        c_alo = max(0, alo - context)
        c_ahi = min(len(held_lines), ahi + context)
        c_blo = max(0, blo - context)
        c_bhi = min(len(disk_lines), bhi + context)

        hunk_header = (
            f"@@ Held Lines {c_alo + 1}-{c_ahi}{step_info} "
            f"-> Live Disk Lines {c_blo + 1}-{c_bhi}{curr_info} @@"
        )
        lines_out = [hunk_header]

        # Context lines before change (using current live disk line numbers)
        for lno in range(c_blo, blo):
            lines_out.append(f"{lno + 1:4d}   {_clip(disk_lines[lno])}")

        # Removed lines from held view
        for lno in range(alo, ahi):
            lines_out.append(f"     - {_clip(held_lines[lno])}")

        # Added / modified lines on disk with current line numbers
        for lno in range(blo, bhi):
            lines_out.append(f"{lno + 1:4d} + {_clip(disk_lines[lno])}")

        # Context lines after change
        for lno in range(bhi, c_bhi):
            lines_out.append(f"{lno + 1:4d}   {_clip(disk_lines[lno])}")

        blocks.append("\n".join(lines_out))

    if not blocks:
        return ""

    return (
        f"[TAIL DIFF: File '{filename}' mutated on disk vs your held view]:\n"
        + "\n\n".join(blocks)
    )


def tail_diff_blocks(
    workspace_dir: str,
    files: List[str],
    freshness_hook: Optional[_BindingsSource] = None,
    before_contents: Optional[Dict[str, str]] = None,
    context_lines: int = TAIL_DIFF_CONTEXT,
    line_numbered: bool = True,
    current_step: Optional[int] = None,
) -> Tuple[str, List[str]]:
    """Compute tail-diff blocks for the specified files against held views.

    The held view is the latest binding from a step before `current_step`: a
    read in the step that wrote the file was registered with the file as it
    stood after the write, and diffing against it would show nothing. With no
    earlier view, the content before this step is the baseline.

    Returns the formatted diff text and the list of files successfully shown.
    """
    if not files:
        return "", []

    bindings = getattr(freshness_hook, "bindings", None) or []
    from .viewport import _ignored  # generated output is not file state (harness v12)
    ignored = _ignored(workspace_dir, list(files))
    files = [f for f in files if f not in ignored]
    known = set(before_contents or ()) | {getattr(b, "file_path", None) for b in bindings}
    blocks: List[str] = []
    shown: List[str] = []
    used_chars = 0

    for rel_path in sorted(files, key=lambda p: (p not in known, p)):
        abs_path = os.path.join(workspace_dir, rel_path) if not os.path.isabs(rel_path) else rel_path
        if not os.path.isfile(abs_path) or _is_binary(abs_path):
            continue
        if len(shown) >= TAIL_DIFF_MAX_FILES:
            break

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace", newline="") as f:
                disk_content = f.read()
        except OSError:
            continue

        # Locate held content and step ID
        held_content = None
        step_seen = None
        for b in reversed(bindings):
            if getattr(b, "file_path", None) != rel_path:
                continue
            if current_step is not None and getattr(b, "step_id", None) is not None and b.step_id >= current_step:
                continue  # this step's own read: the file after the write
            held_content = _held(b)
            step_seen = getattr(b, "step_id", None)
            break

        if held_content is None and before_contents and rel_path in before_contents:
            held_content = before_contents[rel_path]

        if held_content is None:
            # No prior view exists: agent never read this file
            continue

        if held_content == disk_content:
            # File matches held view: no staleness
            continue

        if line_numbered:
            diff_text = format_line_numbered_diff(
                held_content, disk_content, rel_path, context=context_lines, step_seen=step_seen
            )
        else:
            diff_lines = list(difflib.unified_diff(
                split_lines(held_content, keepends=True),
                split_lines(disk_content, keepends=True),
                fromfile=f"{rel_path} (held at Step {step_seen})",
                tofile=f"{rel_path} (current on disk)",
                n=context_lines,
            ))
            diff_text = f"[TAIL DIFF: File '{rel_path}']\n" + "".join(_clip(l) for l in diff_lines)

        if not diff_text:
            continue

        if used_chars + len(diff_text) > TAIL_DIFF_MAX_CHARS:
            # cut at a line end, so the last line shown keeps its whole line number
            diff_text = diff_text[: TAIL_DIFF_MAX_CHARS - used_chars].rsplit("\n", 1)[0] + "\n...[truncated]"

        blocks.append(diff_text)
        shown.append(rel_path)
        used_chars += len(diff_text)
        if used_chars >= TAIL_DIFF_MAX_CHARS:
            break

    if not blocks:
        return "", []

    return "\n\n" + "\n\n".join(blocks), shown


def apply_tail_diff(
    messages: List[Dict[str, Any]],
    bindings: List[object],
    workspace_root: str = ".",
    line_numbered: bool = True,
) -> List[Dict[str, Any]]:
    """Transform context by appending net diffs of stale files to prompt tail.

    Crucially:
    - messages[:-1] is NEVER touched, guaranteeing 100% prefix KV-cache reuse.
    - Only files that are currently stale are diffed against the agent's held view.
    - The diff is appended to the tail message.
    """
    transformed = copy.deepcopy(messages)
    if not transformed:
        return transformed

    # Find distinct stale files
    latest_by_file: Dict[str, object] = {}
    for b in bindings:
        if getattr(b, "is_invalid", False):
            latest_by_file[getattr(b, "file_path", "")] = b

    if not latest_by_file:
        return transformed

    diff_blocks: List[str] = []
    for f_path, binding in latest_by_file.items():
        abs_p = os.path.join(workspace_root, f_path) if not os.path.isabs(f_path) else f_path
        if os.path.exists(abs_p) and os.path.isfile(abs_p):
            try:
                with open(abs_p, "r", encoding="utf-8", errors="replace", newline="") as f:
                    disk_content = f.read()
                held_content = _held(binding)
                if held_content is None or held_content == disk_content:
                    continue

                step_seen = getattr(binding, "step_id", None)
                inval_step = getattr(binding, "invalidation_step", None)

                if line_numbered:
                    diff_block = format_line_numbered_diff(
                        held_content=held_content,
                        disk_content=disk_content,
                        filename=f_path,
                        context=TAIL_DIFF_CONTEXT,
                        step_seen=step_seen,
                        current_step=inval_step,
                    )
                else:
                    diff_lines = list(difflib.unified_diff(
                        split_lines(held_content, keepends=True),
                        split_lines(disk_content, keepends=True),
                        fromfile=f"{f_path} (held at Step {step_seen})",
                        tofile=f"{f_path} (current on disk at Step {inval_step})",
                        n=TAIL_DIFF_CONTEXT,
                    ))
                    diff_block = f"[TAIL DIFF: File '{f_path}']\n" + "".join(_clip(l) for l in diff_lines)

                if diff_block:
                    diff_blocks.append(diff_block)
            except Exception:
                pass

    if diff_blocks:
        forwarding_text = "\n\n" + "\n\n".join(diff_blocks)
        if transformed[-1].get("role") in ["user", "tool"]:
            transformed[-1]["content"] += forwarding_text
        else:
            transformed.append({"role": "user", "content": forwarding_text.strip()})

    return transformed


def line_number_note(path, base, new, limit=4):
    """Every stretch of the model's last whole-file view of `path` that has moved since, with its
    offset. Counted from that one view, so offsets from several writes add up instead of each being
    stated against the previous block (which, in a 60-run test, led agents to apply a shift twice or
    miss one; see docs/results/shift_test_anthropic_claude-sonnet-5.md)."""
    moved = [(i1 + 1, i2, j1 + 1, j2, j1 - i1) for tag, i1, i2, j1, j2 in
             opcodes(split_lines(base), split_lines(new))
             if tag == "equal" and j1 != i1]
    if not moved:
        return ""
    parts = [f"lines {a}-{b} are now {c}-{d} ({o:+d})" for a, b, c, d, o in moved[:limit]]
    if len(moved) > limit:
        parts.append(f"and {len(moved) - limit} more moved stretches")
    return f"[LINE NUMBERS: since your last full view of '{path}': " + "; ".join(parts) + "]"
