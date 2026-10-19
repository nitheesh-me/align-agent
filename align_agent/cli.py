"""ALIGN: find edits a coding agent made against a stale view of a file (analyze), or run an agent with a hook that prevents them (run)."""

import argparse
import glob
import json
import os
import sys
import time

from . import __version__
from .core import FORMATS, analyze, load, session_key, summarize
from .run import add_parser as add_run

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")


def example_files():
    """The trajectories bundled with the package (two runs of one SWE-Bench Pro task)."""
    return sorted(glob.glob(os.path.join(EXAMPLES, "*.json")))


def _files(paths):
    for p in paths:
        if os.path.isdir(p):
            # Claude Code keeps an agent-*.meta.json beside each subagent transcript: not a trajectory.
            yield from sorted(f for ext in ("*.json", "*.jsonl") for f in glob.glob(os.path.join(p, "**", ext), recursive=True)
                              if not f.endswith(".meta.json"))
        else:
            yield p


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _one(job):
    """Analyze one file; (result, None) or (None, reason it was skipped)."""
    f, fmt, cwd = job
    try:
        t, fmt, cov = load(f, fmt, cwd)
    except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        return None, (f, str(e)[:120])
    r = analyze(t)
    r.update(file=f, format=fmt, coverage=cov)
    return r, None


def _analyzed(args):
    """Results in input order. Trajectories are independent, so --jobs N
    spreads them over N processes; the output does not depend on N."""
    jobs = [(f, args.format, args.cwd) for f in _files(args.paths)]
    if args.jobs <= 1 or len(jobs) < 2:
        yield from map(_one, jobs)
        return
    from multiprocessing import get_context
    with get_context("spawn").Pool(args.jobs) as pool:
        yield from pool.imap(_one, jobs, chunksize=8)


def _where(results, roots, top):
    """Lines saying which folders and files hold the stale edits (shown on this machine only: they name folders)."""
    rows = [r for r in results if r["opportunities"]]
    if not rows or not top:
        return []

    def rel(f):
        for root in roots:
            if os.path.isdir(root) and os.path.abspath(f).startswith(os.path.abspath(root) + os.sep):
                return os.path.relpath(f, root)
        return os.path.join(os.path.basename(os.path.dirname(f)), os.path.basename(f))

    def line(name, stale, opps):
        return f"  {stale:5d} of {opps:5d} edits ({100 * stale / opps:3.0f}%)  {name[-70:]}"

    def folder(r):
        # a Claude Code session lives in its project folder (with its subagents under <session>/), whatever
        # root was given; other formats group by the first folder under the root
        if r.get("format") == "claudecode":
            return os.path.basename(os.path.dirname(session_key(r)))
        return rel(r["file"]).split(os.sep)[0]
    folders, sessions = {}, {}
    for r in rows:
        for d, key in ((folders, folder(r)), (sessions, session_key(r))):
            s, o = d.get(key, (0, 0))
            d[key] = (s + r["stale_edits"], o + r["opportunities"])
    per_file = sorted(rows, key=lambda r: -r["stale_edits"])
    total = sum(r["stale_edits"] for r in rows)
    held = sum(r["stale_edits"] for r in per_file[:10])
    out = [f"\nWhere they are: {sum(1 for r in rows if not r['stale_edits'])} of {len(rows)} files with anchored edits have no stale edit"
           + (f"; the 10 files with most hold {100 * held / total:.0f}% of the stale edits." if total else "."),
           f"Folders with most stale edits (folder names are printed here only, not in the summary line above):"]
    out += [line(k, s, o) for k, (s, o) in sorted(folders.items(), key=lambda kv: -kv[1][0])[:top] if s]
    out += ["Sessions with most stale edits (each with its subagents):"]
    out += [line(rel(k), s, o)
            for k, (s, o) in sorted(sessions.items(), key=lambda kv: -kv[1][0])[:top] if s]
    return out


def cmd_examples(args):
    for f in example_files():
        print(f)
    return 0


def cmd_analyze(args):
    if args.examples:
        args.paths = list(args.paths) + example_files()
    if not args.paths:
        print("align-agent analyze: give trajectory files or directories, or --examples", file=sys.stderr)
        return 2
    results, skipped, t0 = [], [], time.time()
    for r, skip in _analyzed(args):
        if skip:
            skipped.append(skip)
            continue
        results.append(r)
        if not args.quiet:
            print(f"{r['file']}  [{r['format']}]  {r['steps']} steps, {r['opportunities']} anchored edits to viewed files, {r['stale_edits']} stale")
            for s in r["stale"]:
                flag = "WRONG " if s["went_wrong"] else "      "
                print(f"  {flag}step {s['step']:>4}  {s['path']}  view from step {s['held_view_step']} ({s['view_channel']}), "
                      f"unseen writes at {s['unseen_writes'][:6]}  {s['class']}: {s['reason']}")
    summary = summarize(results)
    summary["seconds"] = round(time.time() - t0, 2)
    summary["skipped"] = len(skipped)
    print(f"\n{summary['trajectories']} transcripts in {summary['sessions']} sessions, {summary['opportunities']} anchored edits to viewed files, "
          f"{summary['stale_edits']} stale ({_pct(summary['stale_rate'])}, 95% CI {_pct(summary['ci95'][0])}-{_pct(summary['ci95'][1])}"
          f"{' by session' if summary['ci95_method'].startswith('bootstrap') else ''}): "
          f"{summary['stale_after_own_writes']} after the agent's own unread writes, "
          f"{summary['stale_after_other_writes']} after writes it did not make. "
          f"{summary['went_wrong']} went wrong because of it"
          + (f", {summary['caught_by_harness']} caught by Claude Code before they applied" if summary["caught_by_harness"] else "")
          + f". Views behind stale edits: {summary['view_channels']}. {summary['seconds']}s.")
    for ln in _where(results, args.paths, args.top):
        print(ln)
    if skipped:
        print(f"skipped {len(skipped)} files that are not trajectories in a known format", file=sys.stderr)
    for f, e in skipped[:5]:
        print(f"  {f}: {e}", file=sys.stderr)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"summary": summary, "trajectories": results}, fh, indent=1)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="align-agent", description=__doc__)
    ap.add_argument("--version", action="version", version=f"align-agent {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="find stale edits in trajectories")
    a.add_argument("paths", nargs="*", help="trajectory files or directories (searched for *.json and *.jsonl)")
    a.add_argument("--examples", action="store_true", help="also analyze the example trajectories bundled with the package")
    a.add_argument("--format", choices=sorted(FORMATS), help="skip format detection")
    a.add_argument("--cwd", help="the agent's working directory, for resolving relative paths")
    a.add_argument("--json", help="write per-edit results here")
    a.add_argument("--quiet", action="store_true", help="print the summary only")
    a.add_argument("--top", type=int, default=0, metavar="N", help="after the summary, list the N folders and N files with most stale edits")
    a.add_argument("-j", "--jobs", type=int, default=1, help="analyze this many files in parallel (default 1)")
    a.set_defaults(fn=cmd_analyze)
    e = sub.add_parser("examples", help="print the paths of the bundled example trajectories")
    e.set_defaults(fn=cmd_examples)
    add_run(sub)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
