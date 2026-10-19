"""align-agent run: mini-SWE-agent with the write-through hook, on a local repository.

After every command the hook hashes the workspace, finds the files the command
changed, and appends to that command's own output either the changed region
(viewport) or the net diff since the agent last saw the file (diff, lean). The
dependency arm also lists the remaining uses, in other files, of any definition
the write removed or re-signatured. Nothing earlier in the conversation is
rewritten, so the prompt prefix stays cacheable. The trajectory is written in
mini-SWE-agent's format, so `align-agent analyze` reads it back.
"""

import json
import os
import sys
import time


def _vendor():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    root = repo if os.path.isdir(os.path.join(repo, "src", "harness")) else os.path.join(here, "_vendor")
    if root not in sys.path:
        sys.path.insert(0, root)


ARMS = {"control": "control", "viewport": "write_through_viewport", "diff": "write_through_diff",
        "lean": "write_through_diff_lean", "dependency": "dependency_write_through"}


def cmd_run(args):
    # the key goes to the model endpoint only: LocalEnvironment runs every command with
    # os.environ merged in (and serializes it), so the agent could otherwise print it
    key = os.environ.pop(args.api_key_env, "") if args.api_key_env else ""
    if not args.image and not args.no_sandbox and sys.platform != "darwin":
        print("align-agent run: without --image, commands run under macOS sandbox-exec, which this "
              f"system ({sys.platform}) does not have. Pass --image IMAGE, or --no-sandbox inside a container.",
              file=sys.stderr)
        return 2
    _vendor()
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    from src.harness.arms import InterventionArm
    from src.harness.mini_real import TextChatModel, build_agent

    class Model(TextChatModel):
        def _endpoint(self):
            h = {"Content-Type": "application/json"}
            if key:
                h["Authorization"] = f"Bearer {key}"
            return f"{self.config.api_base.rstrip('/')}/chat/completions", h

    model = Model(model_name=args.model, api_base=args.base_url, temperature=args.temperature,
                  max_tokens=args.max_tokens, context_window=args.context_window)
    task = open(args.task).read() if os.path.exists(args.task) else args.task
    agent = build_agent(arm=InterventionArm(ARMS[args.arm]), workspace_dir=args.repo, model=model,
                        step_limit=args.steps, image=args.image, container_cwd=args.container_cwd,
                        sandboxed=not args.no_sandbox)
    t0 = time.time()
    info = {"exit_status": "KeyboardInterrupt", "submission": ""}
    try:
        info = agent.run(task=task)
    except Exception as e:  # the trajectory so far is still worth keeping
        info = {"exit_status": type(e).__name__, "submission": "", "error": str(e)[:500]}
    finally:  # also on Ctrl-C
        exit_status, result = info.get("exit_status", ""), info.get("submission", "")
        out = {"tool": "align-agent run", "arm": ARMS[args.arm], "model": args.model, "container_workdir":
               args.container_cwd if args.image else os.path.abspath(args.repo), "exit_status": exit_status,
               "result": result, "seconds": round(time.time() - t0, 1), "steps": agent.step_records,
               "messages": agent.messages}
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1, default=str)
        blocks = sum(1 for s in agent.step_records if s.get("writes_detected"))
        print(f"{exit_status}: {len(agent.step_records)} steps, {blocks} with a write; trajectory in {args.out}")
    return 0 if exit_status == "Submitted" else 1  # mini-swe-agent's status when the agent submits


def add_parser(sub):
    r = sub.add_parser("run", help="run mini-SWE-agent with the write-through hook on a repository")
    r.add_argument("--repo", required=True, help="the repository the agent edits (modified in place)")
    r.add_argument("--task", required=True, help="task statement, or a file holding it")
    r.add_argument("--arm", choices=sorted(ARMS), default="diff", help="what the hook appends (control: nothing)")
    r.add_argument("--model", required=True, help="model name at the endpoint")
    r.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    r.add_argument("--api-key-env", default="OPENAI_API_KEY", help="environment variable holding the API key")
    r.add_argument("--image", help="run commands in this Docker image with the repo mounted (recommended)")
    r.add_argument("--container-cwd", default="/testbed", help="where the repo is mounted in --image")
    r.add_argument("--no-sandbox", action="store_true", help="run commands directly (inside a container only)")
    r.add_argument("--steps", type=int, default=100)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--max-tokens", type=int, default=4096)
    r.add_argument("--context-window", type=int, default=None)
    r.add_argument("--out", default="trajectory.json")
    r.set_defaults(fn=cmd_run)
