"""The native hook adapters, driven with each agent's documented event payloads.

Run from the repository root: python3 -m unittest discover tests (or pytest). Each test works in a
fresh temporary workspace with its own state directory; nothing outside it is read or written.
"""

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from align_agent.writethrough import claude_code, hooks, mcp_server  # noqa: E402

DEMO = "".join(f"v{i} = {i}\n" for i in range(1, 41))


class Workspace(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.ws = os.path.join(self.dir, "ws")
        os.makedirs(self.ws)
        self.write("demo.py", DEMO)
        env = mock.patch.dict(os.environ, {"ALIGN_STATE": os.path.join(self.dir, "state")})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("CLAUDE_PROJECT_DIR", None)
        os.environ.pop("GEMINI_PROJECT_DIR", None)
        self.addCleanup(shutil.rmtree, self.dir, True)

    def write(self, name, text):
        with open(os.path.join(self.ws, name), "w") as f:
            f.write(text)

    def path(self, name="demo.py"):
        return os.path.join(self.ws, name)

    def sh(self, cmd):
        subprocess.run(cmd, shell=True, cwd=self.ws, check=True)


class Claude(Workspace):
    def call(self, **p):
        return claude_code.handle({"session_id": "c", "cwd": self.ws, **p})

    def test_failed_tool_still_reports_its_changes(self):
        self.call(hook_event_name="SessionStart", source="startup")
        self.call(hook_event_name="PostToolUse", tool_name="Read", tool_input={"file_path": self.path()}, tool_response={})
        self.sh("sed -i '1i import os' demo.py")
        r = self.call(hook_event_name="PostToolUseFailure", tool_name="Bash",
                      tool_input={"command": "sed -i '1i import os' demo.py && false"}, error="Exit code 1")
        self.assertEqual(r["hookSpecificOutput"]["hookEventName"], "PostToolUseFailure")
        self.assertIn("import os", r["hookSpecificOutput"]["additionalContext"])


    def test_failed_read_records_no_view(self):
        self.call(hook_event_name="SessionStart", source="startup")
        self.call(hook_event_name="PostToolUseFailure", tool_name="Read", tool_input={"file_path": self.path()},
                  error="File content exceeds maximum allowed tokens")
        self.sh("sed -i 's/v5 = 5/v5 = 55/' demo.py")
        r = self.call(hook_event_name="PostToolUse", tool_name="Bash", tool_input={"command": "true"}, tool_response={})
        self.assertIsNone(r)  # never shown: no diff against a view the model does not have


class Codex(Workspace):
    def test_move_to_and_crlf_patch_paths(self):
        text = "*** Begin Patch\r\n*** Update File: a.py\r\n*** Move to: b.py\r\n@@\r\n-x\r\n+y\r\n*** End Patch\r\n"
        self.assertEqual(hooks.patched_files(text), ["a.py", "b.py"])


class Gemini(Workspace):
    def call(self, **p):
        return hooks.gemini({"session_id": "g", "cwd": self.ws, **p})

    def test_start_end_line_read_is_partial(self):
        self.call(hook_event_name="SessionStart", source="startup")
        self.call(hook_event_name="AfterTool", tool_name="read_file",
                  tool_input={"file_path": self.path(), "start_line": 1, "end_line": 5}, tool_response={})
        r = self.call(hook_event_name="BeforeTool", tool_name="run_shell_command", tool_input={"command": "sed -i '30d' demo.py"})
        self.assertEqual(r["decision"], "deny")
        r = self.call(hook_event_name="BeforeTool", tool_name="run_shell_command", tool_input={"command": "sed -i '3d' demo.py"})
        self.assertIsNone(r)

    def test_legacy_offset_limit(self):
        self.call(hook_event_name="SessionStart", source="startup")
        self.call(hook_event_name="AfterTool", tool_name="read_file",
                  tool_input={"file_path": self.path(), "offset": 0, "limit": 5}, tool_response={})
        r = self.call(hook_event_name="BeforeTool", tool_name="run_shell_command", tool_input={"command": "sed -i '30d' demo.py"})
        self.assertEqual(r["decision"], "deny")

    def test_read_many_files_records_each_file(self):
        self.write("b.py", "b = 1\n")
        self.call(hook_event_name="SessionStart", source="startup")
        out = f"--- {self.path()} ---\n\n{DEMO}\n--- {self.path('b.py')} ---\n\nb = 1\n\n--- End of content ---"
        self.call(hook_event_name="AfterTool", tool_name="read_many_files", tool_input={"include": ["*.py"]},
                  tool_response={"llmContent": out})
        self.sh("sed -i 's/b = 1/b = 2/' b.py")
        r = self.call(hook_event_name="AfterTool", tool_name="run_shell_command",
                      tool_input={"command": "sed -i 's/b = 1/b = 2/' b.py"}, tool_response={"llmContent": ""})
        self.assertIn("b = 2", r["hookSpecificOutput"]["additionalContext"])


class Cursor(Workspace):
    def call(self, **p):
        return hooks.cursor({"conversation_id": "k", "workspace_roots": [self.ws], **p})

    def test_own_delete_not_reported(self):
        self.call(hook_event_name="sessionStart")
        self.call(hook_event_name="postToolUse", tool_name="Read", tool_input={"file_path": self.path()}, tool_output="")
        os.remove(self.path())
        r = self.call(hook_event_name="postToolUse", tool_name="Delete", tool_input={"file_path": self.path()}, tool_output="")
        self.assertEqual(r, {})

    def test_subagent_has_its_own_views(self):
        self.call(hook_event_name="sessionStart")
        self.call(hook_event_name="postToolUse", tool_name="Read", tool_input={"file_path": self.path()}, tool_output="")
        self.assertEqual(self.call(hook_event_name="subagentStart", subagent_id="sub1"), {"permission": "allow"})
        self.sh("sed -i 's/v5 = 5/v5 = 55/' demo.py")
        # the subagent never read demo.py: the change is not reported to it as a diff
        r = self.call(hook_event_name="postToolUse", subagent_id="sub1", tool_name="Shell",
                      tool_input={"command": "true"}, tool_output="")
        self.assertNotIn("TAIL DIFF", r.get("additional_context", ""))
        # the parent read it: the parent is told
        r = self.call(hook_event_name="postToolUse", tool_name="Shell", tool_input={"command": "true"}, tool_output="")
        self.assertIn("v5 = 55", r["additional_context"])

    def test_read_range_and_before_read_file(self):
        self.call(hook_event_name="sessionStart")
        self.assertEqual(self.call(hook_event_name="beforeReadFile", file_path=self.path()), {"permission": "allow"})
        self.call(hook_event_name="postToolUse", tool_name="Read",
                  tool_input={"file_path": self.path(), "offset": 1, "limit": 5}, tool_output="")
        r = self.call(hook_event_name="beforeShellExecution", command="sed -i '30d' demo.py")
        self.assertEqual(r["permission"], "deny")

    def test_failure_event_reports_changes(self):
        self.call(hook_event_name="sessionStart")
        self.call(hook_event_name="postToolUse", tool_name="Read", tool_input={"file_path": self.path()}, tool_output="")
        self.sh("sed -i '1i import os' demo.py")
        r = self.call(hook_event_name="postToolUseFailure", tool_name="Shell",
                      tool_input={"command": "sed -i '1i import os' demo.py && false"}, error_message="exit 1")
        self.assertIn("import os", r["additional_context"])

    def test_compaction_forgets_views(self):
        self.call(hook_event_name="sessionStart")
        self.call(hook_event_name="postToolUse", tool_name="Read", tool_input={"file_path": self.path()}, tool_output="")
        self.call(hook_event_name="preCompact")
        self.sh("sed -i '1i import os' demo.py")
        r = self.call(hook_event_name="postToolUse", tool_name="Shell", tool_input={"command": "true"}, tool_output="")
        self.assertEqual(r, {})


class Copilot(Workspace):
    def call(self, event, **p):
        return hooks.copilot({"sessionId": "p", "cwd": self.ws, **p}, event)

    def test_resume_keeps_views(self):
        self.call("sessionStart", source="startup")
        self.call("postToolUse", toolName="view", toolArgs={"path": self.path()}, toolResult={})
        self.call("sessionStart", source="resume")
        self.sh("sed -i '1i import os' demo.py")
        r = self.call("postToolUse", toolName="bash", toolArgs={"command": "true"}, toolResult={})
        self.assertIn("import os", r["additionalContext"])

    def test_view_range_is_partial(self):
        self.call("sessionStart", source="startup")
        self.call("postToolUse", toolName="view", toolArgs={"path": self.path(), "view_range": [1, 5]}, toolResult={})
        r = self.call("preToolUse", toolName="bash", toolArgs={"command": "sed -i '30d' demo.py"})
        self.assertEqual(r["permissionDecision"], "deny")

    def test_powershell_is_a_shell(self):
        self.call("sessionStart", source="startup")
        self.call("postToolUse", toolName="view", toolArgs={"path": self.path(), "view_range": [1, 5]}, toolResult={})
        r = self.call("preToolUse", toolName="powershell", toolArgs={"command": "sed -i '30d' demo.py"})
        self.assertEqual(r["permissionDecision"], "deny")

    def test_subagent_has_its_own_views(self):
        self.call("sessionStart", source="startup")
        self.call("postToolUse", toolName="view", toolArgs={"path": self.path()}, toolResult={}, agentId="sub1")
        r = self.call("preToolUse", toolName="bash", toolArgs={"command": "sed -i '30d' demo.py"})
        self.assertIsNone(r)  # the parent never saw demo.py: not a stale view
        r = self.call("preToolUse", toolName="bash", toolArgs={"command": "sed -i '30d' demo.py"}, agentId="sub1")
        self.assertIsNone(r)  # the subagent saw it whole


class Cline(Workspace):
    def call(self, **post):
        return hooks.cline({"taskId": "t", "workspaceRoots": [self.ws], "postToolUse": post}, "PostToolUse")

    def test_own_apply_patch_not_reported(self):
        self.call(toolName="read_file", parameters={"path": self.path()}, result="")
        self.sh("sed -i 's/v3 = 3/v3 = 33/' demo.py")
        patch = "*** Begin Patch\n*** Update File: demo.py\n@@\n-v3 = 3\n+v3 = 33\n*** End Patch"
        self.assertEqual(self.call(toolName="apply_patch", parameters={"input": patch}, result=""), {"cancel": False})

    def test_read_files_records_every_file(self):
        self.write("b.py", "b = 1\n")
        self.call(toolName="read_files", parameters={"files": [{"path": self.path(), "start_line": 1, "end_line": 5},
                                                                {"path": self.path("b.py")}]}, result="")
        self.sh("sed -i 's/b = 1/b = 2/' b.py")
        r = self.call(toolName="execute_command", parameters={"command": "sed -i 's/b = 1/b = 2/' b.py"}, result="")
        self.assertIn("b = 2", r["contextModification"])


class Main(Workspace):
    def test_import_error_fails_open(self):
        with mock.patch.object(hooks, "IMPORT_ERROR", ImportError("boom")), \
                mock.patch.object(sys, "stdin", io.StringIO("{}")), mock.patch.object(sys, "stdout", io.StringIO()) as out:
            self.assertEqual(hooks.main(["copilot", "preToolUse"]), 0)
            self.assertEqual(out.getvalue(), "")
        with open(os.path.join(os.environ["ALIGN_STATE"], "errors.log")) as f:
            self.assertIn("boom", f.read())


class MCP(Workspace):
    def test_background_command_returns(self):
        s = mcp_server.Server(root=self.ws, session="m")
        t = time.time()
        out, err = s.run_command("sleep 5 & echo started", timeout=3)
        self.assertLess(time.time() - t, 2.5)
        self.assertIn("started", out)
        self.assertFalse(err)

    def test_timeout_stops_the_command(self):
        s = mcp_server.Server(root=self.ws, session="m")
        out, err = s.run_command("echo begun; sleep 30", timeout=1)
        self.assertTrue(err)
        self.assertIn("begun", out)
        self.assertIn("timed out", out)

    def test_protocol_version(self):
        s = mcp_server.Server(root=self.ws, session="m")
        init = lambda v: s.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": v}})
        self.assertEqual(init("2099-01-01")["result"]["protocolVersion"], mcp_server.PROTOCOL)
        self.assertEqual(init("2025-03-26")["result"]["protocolVersion"], "2025-03-26")


@unittest.skipIf(os.name == "nt", "POSIX sh launcher")
@unittest.skipIf(os.name == "nt", "POSIX sh and symlinks; Windows runs hook.sh under Git Bash in CI instead")
class Launcher(Workspace):
    def run_hook(self, path_dirs):
        hook = os.path.join(ROOT, "claude-plugin", "scripts", "hook.sh")
        env = dict(os.environ, PATH=os.pathsep.join(path_dirs), CLAUDE_PLUGIN_DATA=os.path.join(self.dir, "state"))
        payload = json.dumps({"hook_event_name": "SessionStart", "session_id": "l", "cwd": self.ws, "source": "startup"})
        return subprocess.run(["/bin/sh", hook], input=payload, capture_output=True, text=True, env=env, timeout=30)

    def test_no_python_exits_quietly(self):
        bindir = os.path.join(self.dir, "bin")
        os.makedirs(bindir)
        for tool in ("dirname", "cat", "uname"):
            os.symlink(shutil.which(tool), os.path.join(bindir, tool))
        r = self.run_hook([bindir])
        self.assertEqual((r.returncode, r.stdout, r.stderr), (0, "", ""))

    def test_store_stub_is_skipped(self):
        bindir = os.path.join(self.dir, "bin")
        os.makedirs(bindir)
        for tool in ("dirname", "cat", "uname"):
            os.symlink(shutil.which(tool), os.path.join(bindir, tool))
        with open(os.path.join(bindir, "python3"), "w") as f:
            f.write("#!/bin/sh\necho 'Python was not found; run without arguments to install from the Microsoft Store'\nexit 9009\n")
        os.chmod(os.path.join(bindir, "python3"), 0o755)
        os.symlink(sys.executable, os.path.join(bindir, "python"))
        r = self.run_hook([bindir])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("Microsoft Store", r.stdout + r.stderr)
        self.assertTrue(os.path.isdir(os.path.join(self.dir, "state", "sessions", "l")))  # hook.py ran via `python`

    def test_interpreter_is_cached_and_reprobed(self):
        bindir = os.path.join(self.dir, "bin")
        os.makedirs(bindir)
        for tool in ("dirname", "cat", "uname", "mkdir"):
            os.symlink(shutil.which(tool), os.path.join(bindir, tool))
        os.symlink(sys.executable, os.path.join(bindir, "python3"))
        cache = os.path.join(self.dir, "state", "python")
        self.assertEqual(self.run_hook([bindir]).returncode, 0)
        with open(cache) as f:
            self.assertEqual(f.read().splitlines()[0], os.path.join(bindir, "python3"))
        with open(cache, "w") as f:
            f.write("/nonexistent/python3\n\n")  # uninstalled since: probe again, do not fail
        r = self.run_hook([bindir])
        self.assertEqual((r.returncode, r.stderr), (0, ""))
        with open(cache) as f:
            self.assertEqual(f.read().splitlines()[0], os.path.join(bindir, "python3"))


if __name__ == "__main__":
    unittest.main()

