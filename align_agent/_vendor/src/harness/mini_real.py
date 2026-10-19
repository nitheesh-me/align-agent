"""The published mini-swe-agent, instrumented with the freshness hook.

Earlier runs used a local re-implementation of the mini-swe-agent loop. Its
system prompt only asked for commands inside bash fences, so models answered
with entire imagined sessions: gemma-4-26b-a4b emitted 20 to 36 blocks per
turn, invented the output of each, and ran to the 4096-token cap (60 to 113
seconds a turn). The re-implementation executed all of those blocks as one
script, which means it ran edits the model had decided on without ever seeing
a real observation.

The published agent does not behave that way. It requires exactly one command
per turn and rejects anything else with a format error, and under its prompt
the same model answers with one action in 131 to 323 tokens (5 to 8 seconds a
turn). Whatever the re-implementation measured, it was not mini-swe-agent, so
this module runs the installed package (v2.4.6) with its own prompts, parser,
control flow and limits, and adds only what the experiment needs:

  * observation bindings, registered from the same filesystem-validated
    read/write detection used everywhere else in this repository;
  * the per-arm transformation of the context that is *sent* to the model,
    leaving the recorded trajectory untouched.

Deviations from stock mini-swe-agent, all forced by running locally rather
than in the SWE-bench Docker image, are listed in MINI_DEVIATIONS below.
"""

import json
import logging
import os
import re
import shlex
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel

from minisweagent import package_dir, __version__ as MINI_VERSION
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded
from minisweagent.models.utils.actions_text import (
    format_observation_messages,
    parse_regex_actions,
)

from src.metrics.corruption_oracles import check_ast_parseability

from .arms import (
    ARM_REMEDIATION,
    PROMPT_ONLY_POLICY,
    WRITE_THROUGH_ARMS,
    ArmRemediation,
    InterventionArm,
    apply_intervention_arm,
    synced_context_view,
    LEAN_ARMS,
    SYNC_HEADER,
    SYNC_WIDE_MAX_CHARS,
    SYNC_WIDE_MAX_FILE_CHARS,
)
from .completion import post_completion
from .dependency_notice import NOTICE_VERSION, committed, dependency_notice
from .lean import redundant_writes
from .tail_diff import line_number_note
from .miniswe_scaffold import MiniSWEScaffold

logger = logging.getLogger("mini_real")

# Tokens a chat template adds per turn beyond '<|im_start|>role\n...<|im_end|>',
# allowed for when the server cannot render the template itself.
TEMPLATE_TOKENS_PER_MESSAGE = 8
# A request that overflows the window is retried with half the output budget,
# at most this many times, when the prompt alone still fits.
MAX_BUDGET_RETRIES = 2
_WINDOW_RE = re.compile(r"maximum context length is (\d+) tokens")
_INPUT_RE = re.compile(r"(\d+) input tokens|\((\d+) (?:of text input|in the messages)")


def smaller_budget(error: str, max_tokens: int) -> Optional[int]:
    """The max_tokens to retry with after a context-length error, or None when
    the prompt alone does not fit and no output budget can help."""
    if "Context length exceeded" not in error:
        return None
    window, prompt = _WINDOW_RE.search(error), _INPUT_RE.search(error)
    if window and prompt and int(prompt.group(1) or prompt.group(2)) >= int(window.group(1)) - 1024:
        return None
    smaller = max_tokens // 2
    return smaller if smaller >= 512 else None

# Text-based (backticks) variant: one ```mswea_bash_command block per turn.
# This is the format the RQ1 trajectory corpus was produced in, and the one
# local servers support without tool-calling support.
MINI_CONFIG_PATH = package_dir / "config" / "benchmarks" / "swebench_backticks.yaml"

# Deviations when running in the benchmark's own image, which is the default.
MINI_DEVIATIONS = [
    "the cell's workspace is bind-mounted over /testbed so instrumentation can watch it from the host",
    "container limited to 4 CPUs and 8GB, and images run emulated on arm64",
    "cost_limit disabled: neither backend reports usable per-call cost",
    "max_tokens capped per response, which stock mini-swe-agent leaves unset",
    "step_limit set per run (stock: 250), bounded by local compute",
]

# Deviations when running locally instead, which is the fallback path.
LOCAL_DEVIATIONS = MINI_DEVIATIONS[2:] + [
    "environment: LocalEnvironment, no SWE-bench image",
    "working directory: per-task workspace clone instead of /testbed",
    "PYTHONPATH: workspace first, so imports resolve to the checkout under test",
    "build and numeric thread pools capped at one, so a single command cannot take the host",
    "HOME points at the workspace, so agent writes stay inside the cell",
    "commands run under sandbox-exec: reads unrestricted, writes confined to the cell",
]

DOCKER_BIN = os.environ.get("SGA_DOCKER", "/usr/local/bin/docker" if os.path.exists("/usr/local/bin/docker") else "/Applications/Docker.app/Contents/Resources/bin/docker")
CONTAINER_CWD = "/testbed"


def ensure_docker_on_path(docker_bin: str = DOCKER_BIN) -> None:
    """Put Docker Desktop's bin directory on PATH.

    The CLI is not on PATH on this machine, and more importantly neither is
    docker-credential-desktop, which docker shells out to for any registry
    access: without it every pull fails with "error getting credentials".
    """
    bindir = os.path.dirname(docker_bin)
    if bindir and bindir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")


ensure_docker_on_path()

def load_mini_config(path=MINI_CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mini_model_defaults() -> Dict[str, str]:
    """Prompt-side settings the published config defines for the model.

    These are not cosmetic. The observation template folds any output over
    10,000 characters into head, tail and an elided-character count, which is
    what keeps a `cat` of a large module from tripling the context; the format
    error template distinguishes a malformed answer from one truncated by the
    token limit and tells the model to be briefer. Running without them once
    cost a cell three token-limit truncations in a row, each reported to the
    model as a generic format error it could not act on.
    """
    model_cfg = load_mini_config().get("model", {})
    return {
        key: model_cfg[key]
        for key in ("observation_template", "format_error_template", "action_regex")
        if key in model_cfg
    }


class TextModelConfig(BaseModel):
    """Settings for a text-action (non-toolcall) chat backend."""

    model_name: str
    api_base: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 360
    # Context window the prompt is budgeted against. None means 1M for Gemini
    # and 32K for every other backbone, gateway models included: they are
    # held to the local Qwen's window, not their own. Compaction applies to
    # any window of 64K or less, so all of those share one compaction policy.
    context_window: Optional[int] = None
    # Mark the request for the provider's prefix cache (gateway backends). It
    # changes what a request costs, never what the model is sent as text.
    prompt_cache: bool = False
    action_regex: str = r"```(?:mswea_bash_command|bash|sh|shell)\s*\n(.*?)\n```"
    format_error_template: str = (
        "Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions."
    )
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    multimodal_regex: str = ""


class TextChatModel:
    """Minimal mini-swe-agent Model backed by an OpenAI-compatible endpoint.

    mini-swe-agent ships litellm-based models; this one exists because the two
    backends used here need what litellm does not give: a bearer token that is
    refreshed mid-run for the cloud gateway, and the retry policy that keeps a
    dropped request from costing a whole trajectory.
    """

    def __init__(self, **kwargs):
        self.config = TextModelConfig(**kwargs)
        self.n_calls = 0
        self.last_finish_reason = ""
        # Requests retried with a smaller output budget after a context-length
        # error; recorded per cell.
        self.budget_retries = 0
        # Indices of observation messages the last request shortened for the
        # context budget. The staleness metrics read the recorded observation,
        # so a trace has to say when the model was sent less than that.
        self.last_compacted: List[int] = []

    # --- request construction -------------------------------------------------

    def _endpoint(self) -> Tuple[str, Dict[str, str]]:
        return f"{self.config.api_base}/chat/completions", {"Content-Type": "application/json"}

    def _extra_payload(self) -> Dict[str, Any]:
        return {}

    def _tokenize(self, base: str, body: Dict[str, Any]) -> int:
        req = urllib.request.Request(f"{base}/tokenize", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return int(json.loads(resp.read().decode("utf-8"))["count"])

    def _prompt_token_count(self, messages: List[Dict[str, Any]]) -> int:
        """Prompt tokens as the server will count them.

        vLLM's /tokenize renders `messages` with the model's own chat template
        and the template options the completion request sends. The hand-built
        '<|im_start|>' string used before is not that template, and budgeting
        max_tokens against it overflowed the window in 37 of 63 Qwen SWE-Bench
        Pro cells. That string remains a fallback for a server that cannot
        render chat, with an allowance per message; a character estimate is
        the last resort.
        """
        if self.config.api_base and ("localhost" in self.config.api_base or "127.0.0.1" in self.config.api_base):
            base = self.config.api_base.replace("/v1", "")
            chat: Dict[str, Any] = {
                "model": self.config.model_name,
                "messages": [{"role": m.get("role", ""), "content": m.get("content", "")} for m in messages],
                "add_generation_prompt": True,
            }
            template_kwargs = self._extra_payload().get("chat_template_kwargs")
            if template_kwargs:
                chat["chat_template_kwargs"] = template_kwargs
            try:
                return self._tokenize(base, chat)
            except Exception as exc:
                logger.debug("chat tokenization unavailable, counting a rendered prompt: %s", exc)
            try:
                parts = ["<|im_start|>" + m.get("role", "") + "\n" + m.get("content", "") + "<|im_end|>" for m in messages]
                count = self._tokenize(base, {"model": self.config.model_name, "prompt": "\n".join(parts)})
                return count + len(messages) * TEMPLATE_TOKENS_PER_MESSAGE
            except Exception as exc:
                logger.debug("vLLM tokenize endpoint unavailable, falling back to estimate: %s", exc)
        # Highly conservative fallback: 3.0 chars/token ensures we never underestimate prompt length
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        return int(prompt_chars / 3.0) + len(messages) * 4 + 64

    # Through protocol v13 the task statement was cut like any old observation,
    # so a task over 8,000 characters lost its middle from the second request
    # on. It is never compacted now; the switch exists only so an analysis can
    # rebuild the requests an older run was sent.
    compacts_task = False

    def prompt_window(self) -> int:
        """The window prompts are budgeted against, in tokens."""
        is_gemini = "gemini" in self.config.model_name.lower()
        return self.config.context_window or (1048576 if is_gemini else 32768)

    def _payload(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        is_gemini = "gemini" in self.config.model_name.lower()
        max_model_len = self.prompt_window()
        # Compaction follows the window, not the model: a small window needs
        # it whichever backbone is behind it. Thresholds scale with the window
        # and are exactly 24000 / 28000 at 32K, as before.
        compact = max_model_len <= 65536
        phase2_at = max_model_len * 24000 // 32768
        phase3_at = max_model_len * 28000 // 32768
        compacted = set()

        # Set Context Budgets and Multi-Phase Compaction:
        # Phase 1: Auto-compact older assistant chain-of-thought blocks (<think>...</think>)
        # and truncate huge tool observations in older turns to avoid context overflow.
        cleaned_messages = []
        for i, m in enumerate(messages):
            m_copy = dict(m)
            content = str(m.get("content", ""))
            if m.get("role") == "assistant" and i < len(messages) - 2:
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                elif "<think>" in content:
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                m_copy["content"] = content
            elif compact and (i >= 2 or self.compacts_task) and m.get("role") in ("user", "tool") \
                    and i < len(messages) - 1 and len(content) > 8000:
                half = 4000
                m_copy["content"] = content[:half] + f"\n\n[... {len(content) - 8000} characters elided for context budget ...]\n\n" + content[-half:]
                compacted.add(i)
            cleaned_messages.append(m_copy)

        prompt_tokens = self._prompt_token_count(cleaned_messages)

        # Phase 2: If prompt approaches 32k context boundary, aggressively compact older observations
        if compact and prompt_tokens > phase2_at:
            for i in range(2, max(2, len(cleaned_messages) - 4)):
                m = cleaned_messages[i]
                if m.get("role") in ("user", "tool"):
                    c = str(m.get("content", ""))
                    if len(c) > 1000:
                        m["content"] = c[:400] + f"\n\n[... {len(c) - 800} chars compacted ...]\n\n" + c[-400:]
                        compacted.add(i)
            prompt_tokens = self._prompt_token_count(cleaned_messages)

        # Phase 3: Final safety compaction if still near the limit
        if compact and prompt_tokens > phase3_at:
            for i in range(2, max(2, len(cleaned_messages) - 2)):
                m = cleaned_messages[i]
                c = str(m.get("content", ""))
                if len(c) > 400:
                    m["content"] = c[:200] + "\n[... compacted ...]\n" + c[-200:]
                    if m.get("role") in ("user", "tool"):
                        compacted.add(i)
            prompt_tokens = self._prompt_token_count(cleaned_messages)

        self.last_compacted = sorted(compacted)
        configured_max = self.config.max_tokens or (32768 if is_gemini else 16384)
        # The count is the server's own for a local model; the margin covers
        # what it cannot see (a template change, a stop sequence) and grows
        # with the prompt. It binds only near the window.
        safety_margin = max(256, prompt_tokens // 50)
        remaining = max_model_len - prompt_tokens - safety_margin
        effective_max = max(128, min(configured_max, remaining))
        if prompt_tokens + effective_max > max_model_len:
            effective_max = max(64, max_model_len - prompt_tokens - 32)

        return {
            "model": self.config.model_name,
            "messages": [{"role": m["role"], "content": m.get("content", "")} for m in cleaned_messages],
            "temperature": self.config.temperature,
            "max_tokens": effective_max,
            **self._extra_payload(),
            **kwargs,
        }

    # --- Model protocol -------------------------------------------------------

    def query(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        url, headers = self._endpoint()
        payload = self._payload(messages, **kwargs)
        for attempt in range(MAX_BUDGET_RETRIES + 1):
            try:
                data = post_completion(url, payload, headers, self.config.timeout)
                break
            except RuntimeError as exc:
                smaller = smaller_budget(str(exc), int(payload.get("max_tokens") or 0))
                if smaller is None or attempt == MAX_BUDGET_RETRIES:
                    raise
                logger.warning("Context length exceeded with max_tokens=%s; retrying with %d",
                               payload.get("max_tokens"), smaller)
                payload = {**payload, "max_tokens": smaller}
                self.budget_retries += 1
        self.n_calls += 1
        choice = data["choices"][0]
        message = choice.get("message", {})
        # Reasoning models leave `content` empty while thinking; a turn that
        # spent its whole budget reasoning still has to be parsed for an action.
        content = (
            message.get("content")
            or message.get("reasoning_content")
            or message.get("reasoning")
            or ""
        )
        finish_reason = choice.get("finish_reason") or ""
        self.last_finish_reason = finish_reason
        if finish_reason in ("length", "max_tokens"):
            # Check if the model already emitted a valid, complete action block, or an unclosed block
            content_to_check = content
            if "```" in content_to_check and content_to_check.count("```") % 2 == 1:
                content_to_check += "\n```"
            try:
                actions = parse_regex_actions(
                    content_to_check,
                    action_regex=self.config.action_regex,
                    format_error_template=self.config.format_error_template,
                    template_kwargs={"finish_reason": finish_reason},
                )
                if actions:
                    logger.info("Model hit token cap (%r) but successfully emitted action from unclosed block: continuing trajectory.", finish_reason)
                    return {
                        "role": "assistant",
                        "content": content_to_check,
                        "extra": {
                            "actions": actions,
                            "cost": 0.0,
                            "finish_reason": finish_reason,
                            "response": data,
                        },
                    }
            except Exception:
                pass
            logger.warning("Model hit token cap (%r) without complete action: raising FormatError to retry concisely.", finish_reason)
            raise FormatError(
                self.format_message(
                    role="user",
                    content=(
                        "Your previous response was truncated by the token limit before emitting an action.\n"
                        "Please be concise. Provide ONLY the next shell command inside code fences:\n"
                        "```bash\n<command>\n```"
                    ),
                    extra={"cost": 0.0, "response": data, "finish_reason": finish_reason},
                )
            )
        try:
            actions = parse_regex_actions(
                content,
                action_regex=self.config.action_regex,
                format_error_template=self.config.format_error_template,
                template_kwargs={"finish_reason": finish_reason},
            )
        except FormatError as e:
            # The spec requires the response and cost to survive a format error.
            e.messages[0]["extra"].update({"cost": 0.0, "response": data})
            raise
        return {
            "role": "assistant",
            "content": content,
            "extra": {
                "actions": actions,
                "cost": 0.0,
                "finish_reason": finish_reason,
                "usage": data.get("usage", {}),
                "timestamp": time.time(),
            },
        }

    def format_message(self, **kwargs) -> Dict[str, Any]:
        return dict(kwargs)

    def format_observation_messages(
        self, message: Dict[str, Any], outputs: List[Dict[str, Any]], template_vars: Optional[dict] = None
    ) -> List[Dict[str, Any]]:
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> Dict[str, Any]:
        return self.config.model_dump() | kwargs

    def serialize(self) -> Dict[str, Any]:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }


class LMStudioModel(TextChatModel):
    """Local LM Studio server.

    Sampling follows Qwen's published guidance for precise coding tasks rather
    than the greedy decoding used for the cloud backbone. Measured on one
    astropy prompt, greedy decoding spent 3029 reasoning tokens against 2341
    under these settings, and Qwen's general-task preset (temperature 1.0 with
    presence_penalty 1.5) spent 4132, so the presets are not interchangeable.
    Running a model outside the regime its authors specify and then reporting
    the resulting truncations as properties of the model is the error this
    avoids; the cost is that local cells are sampled rather than greedy, which
    the paper records.

    These are sent per request because the LM Studio GUI's own defaults for
    this model (top_k 100, repeat penalty 1.1) do not match the guidance, and
    per-request parameters override them.
    """

    QWEN_CODING_SAMPLING = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 0.0,
    }

    def _extra_payload(self) -> Dict[str, Any]:
        return {
            "cache_prompt": True,
            "chat_template_kwargs": {"preserve_thinking": True},
            "extra_body": {"chat_template_kwargs": {"preserve_thinking": True}},
            **self.QWEN_CODING_SAMPLING,
        }


class PostQodeModel(TextChatModel):
    """Cloud gateway whose bearer token is refreshed during long runs."""

    def __init__(self, auth, reasoning_effort: str = "minimal", **kwargs):
        super().__init__(**kwargs)
        self._auth = auth
        self._reasoning_effort = reasoning_effort

    def _endpoint(self) -> Tuple[str, Dict[str, str]]:
        headers = self._auth.request_headers()
        self._sent_token = headers["Authorization"].split(" ", 1)[-1]
        return self._auth.completions_url, headers

    def _payload(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        payload = super()._payload(messages, **kwargs)
        if self.config.prompt_cache:
            # Anthropic-style breakpoints, passed through by the gateway: one
            # on the system prompt and one on the newest message. The next
            # request extends this one, so its prefix up to here is read from
            # the cache and only the new turn is written. That holds only
            # while the request is append-only, which compaction under a
            # small window is not.
            # The synced-context arm's block changes every step, so its
            # breakpoint goes before the block, which is sent as a second part.
            sent = payload["messages"]
            for i in {0, len(sent) - 1}:
                text = str(sent[i]["content"])
                cut = text.find("\n\n" + SYNC_HEADER) if i else -1
                parts = [{"type": "text", "text": text[:cut] if cut >= 0 else text, "cache_control": {"type": "ephemeral"}}]
                if cut >= 0:
                    parts.append({"type": "text", "text": text[cut:]})
                sent[i] = {**sent[i], "content": parts}
        return payload

    def query(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        # Runner processes share one credentials file; a refresh in one revokes
        # the token the others hold, so a 401 is retried once with the token
        # now on disk (or a fresh one).
        try:
            return super().query(messages, **kwargs)
        except urllib.error.HTTPError as exc:
            if exc.code != 401:
                raise
            logger.warning("PostQode rejected the access token; reloading credentials and retrying once")
            self._auth.recover(self._sent_token)
            return super().query(messages, **kwargs)

    def _extra_payload(self) -> Dict[str, Any]:
        if not self._reasoning_effort:
            return {}
        return {"reasoning": {"effort": self._reasoning_effort}}


SANDBOX_PROFILE_NAME = "align-agent.sb"

# Everything is readable, as in the SWE-bench image, but writes are confined
# to the cell. Without this an agent looking for a test runner ran
# `find / -name pytest`, found an unrelated virtualenv belonging to the user,
# and pip-installed into it: host state changed because of a measurement.
SANDBOX_PROFILE = """(version 1)
(allow default)
(deny file-write*)
(allow file-write*
  (subpath "{workspace}")
  (subpath "/private/tmp")
  (subpath "/private/var/tmp")
  (subpath "/private/var/folders")
  (subpath "/dev"))
(deny file-write* (subpath "{profile_dir}"))
"""


class SandboxedLocalEnvironment(LocalEnvironment):
    """LocalEnvironment whose commands cannot write outside the workspace.

    Docker is the faithful option and is what stock mini-swe-agent uses for
    SWE-bench, but the published images are x86_64 and would run emulated
    here. This keeps execution native, and keeps the freshness instrumentation
    working directly against the workspace, while removing the agent's ability
    to modify anything else on the machine.
    """

    def __init__(self, *, sandbox_profile: str, **kwargs):
        super().__init__(**kwargs)
        self.sandbox_profile = sandbox_profile

    def execute(self, action: Dict[str, Any], cwd: str = "", *, timeout: Optional[int] = None):
        command = action.get("command", "")
        wrapped = dict(action)
        wrapped["command"] = (
            f"/usr/bin/sandbox-exec -f {shlex.quote(self.sandbox_profile)} "
            f"/bin/bash -c {shlex.quote(command)}"
        )
        return super().execute(wrapped, cwd, timeout=timeout)


def write_sandbox_profile(workspace_dir: str) -> str:
    """Write the cell's sandbox profile and return its path.

    sandbox-exec re-reads the profile for every command, so it lives outside the
    workspace, read-only, in a directory the profile itself denies writes to (the
    temporary directory is otherwise writable): the agent cannot loosen it."""
    profile_dir = os.path.realpath(tempfile.mkdtemp(prefix="align-agent-sandbox-"))
    path = os.path.join(profile_dir, SANDBOX_PROFILE_NAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write(SANDBOX_PROFILE.format(workspace=os.path.realpath(workspace_dir), profile_dir=profile_dir))
    os.chmod(path, 0o444)
    return path


class DryRunModel(TextChatModel):
    """Offline stand-in that replays a fixed action script.

    Exercises the whole pipeline (workspace provisioning, parsing, execution,
    binding, invalidation, arm transformation, trace writing) without a server.
    The script deliberately reads a file, edits it, and reads it again, so a
    dry run that reports zero invalidations means the instrumentation is
    broken rather than the model being quiet.
    """

    SCRIPT = [
        "printf 'def f():\\n    return 1\\n' > probe.py && ls probe.py",
        "cat probe.py",
        "python3 - <<'PY'\nimport pathlib\np = pathlib.Path('probe.py')\np.write_text(p.read_text().replace('return 1', 'return 2'))\nPY",
        "cat probe.py",
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat probe.py",
    ]

    def __init__(self, model_name: str = "dry-run", **kwargs):
        super().__init__(model_name=model_name, **kwargs)

    def query(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        command = self.SCRIPT[min(self.n_calls, len(self.SCRIPT) - 1)]
        self.n_calls += 1
        content = f"THOUGHT: scripted dry-run step {self.n_calls}\n\n```mswea_bash_command\n{command}\n```"
        actions = parse_regex_actions(
            content,
            action_regex=self.config.action_regex,
            format_error_template=self.config.format_error_template,
        )
        return {
            "role": "assistant",
            "content": content,
            "extra": {"actions": actions, "cost": 0.0, "finish_reason": "stop", "timestamp": time.time()},
        }


class FreshnessAgent(DefaultAgent):
    """mini-swe-agent with observation-freshness instrumentation.

    Two hooks, both narrow:

    `query` sends the arm's view of the history to the model. The recorded
    trajectory keeps the unmodified messages, so what the agent saw and what
    the trace says it saw stay separable.

    `execute_actions` brackets the command with workspace snapshots, binds the
    files it read to the observation message that carries their contents, and
    invalidates bindings for the files it wrote. That is the measurement.
    """

    def __init__(self, model, env, *, arm: InterventionArm, workspace_dir: str,
                 use_mtime_cache: bool = True, **kwargs):
        super().__init__(model, env, **kwargs)
        self.arm = arm
        self.workspace_dir = os.path.abspath(workspace_dir)
        # The scaffold is used here purely as the instrumentation toolbox
        # (snapshots, path normalisation, operation ordering, freshness hook),
        # not as an agent loop: the loop is mini-swe-agent's.
        self.instr = MiniSWEScaffold(
            arm=arm, workspace_dir=self.workspace_dir, use_mtime_cache=use_mtime_cache
        )
        self.step_records: List[Dict[str, Any]] = []
        self.n_format_errors = 0
        # Files whose current contents the last prompt carried (synced-context arm).
        self._synced_shown: List[str] = []
        # Dependency write-through: each written file as this harness last left it,
        # the baseline its next write's definitions are compared with.
        self._dep_cache: Dict[str, str] = {}
        # Lean diff: scripts the agent saved, so a later run of one is read as the edit it makes.
        self._lean_scripts: Dict[str, str] = {}
        # Line-number note: each file as the agent last read it, the view its line numbers come from.
        self._line_base: Dict[str, str] = {}

    def _disk_text(self, rel: str) -> Optional[str]:
        try:
            with open(os.path.join(self.workspace_dir, rel), "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return None

    # --- context transformation ----------------------------------------------

    def _arm_view(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        plain = [{k: v for k, v in m.items() if k != "extra"} for m in messages]
        if self.arm == InterventionArm.ARM_11_SYNCED_CONTEXT:
            wide = hasattr(self.model, "prompt_window") and self.model.prompt_window() > 65536
            view, self._synced_shown = synced_context_view(
                plain, self.instr.freshness_hook.bindings, self.workspace_dir,
                **({"max_chars": SYNC_WIDE_MAX_CHARS, "max_file_chars": SYNC_WIDE_MAX_FILE_CHARS} if wide else {}))
            last = str(view[-1].get("content", "")) if view else ""
            cut = last.find(SYNC_HEADER)
            self._synced_chars = len(last) - cut if cut >= 0 else 0
            return view
        return apply_intervention_arm(
            messages=plain,
            bindings=self.instr.freshness_hook.bindings,
            arm=self.arm,
            max_tokens=self.instr.max_tokens,
            current_tokens=sum(len(str(m.get("content", ""))) // 4 for m in plain),
            workspace_root=self.workspace_dir,
        )

    def query(self) -> Dict[str, Any]:
        original = self.messages
        self.messages = self._arm_view(original)
        try:
            message = super().query()
        except FormatError:
            self.n_format_errors += 1
            raise
        finally:
            # super().query() appended the reply to the transformed copy; the
            # real history is restored and the reply added to it instead.
            self.messages = original
        self.add_messages(message)
        return message

    # --- measurement ----------------------------------------------------------

    def _syntax_error_in(self, writes: List[str]) -> bool:
        """Whether any Python file this step wrote is now unparseable.

        Checked immediately after the command, since a later step may repair
        the file and a post-hoc pass over the final tree would miss it.
        """
        for rel in writes:
            if not rel.endswith(".py"):
                continue
            path = rel if os.path.isabs(rel) else os.path.join(self.workspace_dir, rel)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError:
                continue
            if not check_ast_parseability(content, rel).is_valid:
                return True
        return False

    def _bound_matches_disk(self) -> Dict[str, bool]:
        """Per file, whether the agent's most recent view still matches disk.

        Compared by content hash against the live file, so a write that landed
        outside the harness's view still shows up as a mismatch.
        """
        hook = self.instr.freshness_hook
        latest: Dict[str, Any] = {}
        for binding in hook.bindings:
            latest[binding.file_path] = binding
        out: Dict[str, bool] = {}
        for path, binding in latest.items():
            current = hook.compute_file_hash(path)
            out[path] = bool(current) and current == binding.content_hash
        return out

    def _context_staleness(self) -> Tuple[int, int]:
        """Bytes of stale observation still visible, and bytes of bound observations.

        Visibility follows the arm: an arm that replaces stale content shows
        none of it, which is what separates "the hazard did not happen" from
        "the hazard happened and the arm repaired it".
        """
        remediation = ARM_REMEDIATION.get(self.arm, ArmRemediation())
        bound_bytes = 0
        stale_bytes = 0
        for binding in self.instr.freshness_hook.bindings:
            bound_bytes += len(binding.raw_content or "")
            if binding.is_invalid and not remediation.withdraws:
                stale_bytes += len(binding.raw_content or "")
            elif binding.is_invalid and remediation.partial:
                stale_bytes += len(binding.raw_content or "") // 2
        return stale_bytes, bound_bytes

    def _append_to_observation(self, observations: List[Dict[str, Any]], block: str) -> None:
        """Append a hook block to the step's observation, once.

        `add_messages` returns the dicts it put in the history, so the
        observation and `self.messages[-1]` are one object. Through protocol
        v13 the block was appended to each name and reached the model twice.
        """
        if not observations:
            return  # a submitting step: nothing further is sent to the model
        obs = observations[-1]
        targets = [obs]
        if self.messages and self.messages[-1] is not obs and self.messages[-1].get("role") in ("user", "tool"):
            targets.append(self.messages[-1])  # a base agent that copies its messages
        for m in targets:
            m["content"] = str(m.get("content", "")) + block
        extra = obs.get("extra")
        if isinstance(extra, dict) and "raw_output" in extra:
            extra["raw_output"] = str(extra["raw_output"]) + block

    def execute_actions(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        actions = message.get("extra", {}).get("actions", [])
        command = "\n".join(a.get("command", "") for a in actions)
        before = self.instr.snapshot_workspace()
        before_contents: Dict[str, str] = {}
        if self.arm in WRITE_THROUGH_ARMS:
            referenced = self.instr.resolve_referenced_files(command)
            for r_file in referenced:
                abs_p = os.path.join(self.workspace_dir, r_file)
                if os.path.isfile(abs_p):
                    try:
                        with open(abs_p, "r", encoding="utf-8", errors="replace") as f:
                            before_contents[r_file] = f.read()
                    except OSError:
                        pass

        # Whether the agent's view of each bound file still matches disk at the
        # moment it acts. Inferring this after the fact only catches the edits
        # that visibly failed; recorded here it also catches the ones that
        # succeeded against a file the agent had not looked at since changing
        # it, which is the case the paper is about.
        bound_matches_disk = self._bound_matches_disk()
        visible_stale_bytes, context_bytes = self._context_staleness()
        interrupted: Optional[BaseException] = None
        observations: List[Dict[str, Any]] = []
        try:
            observations = super().execute_actions(message)
        except InterruptAgentFlow as exc:
            # Submission raises out of env.execute before the observation is
            # recorded; the step still happened and is still measured.
            interrupted = exc
        # Two different texts, deliberately. `output` is what the command
        # actually printed and goes into the trace for auditing; `obs_text` is
        # what the published observation template put in the context, which is
        # elided above 10,000 characters. Bindings must hold the second: the
        # volume-matched arm evicts an amount of context equal to what the
        # masking arm removes, and measuring that against the unelided output
        # would evict several times too much.
        output = ""
        obs_text = ""
        if observations:
            output = str(observations[-1].get("extra", {}).get("raw_output", ""))
            obs_text = str(observations[-1].get("content", ""))
            output = output or obs_text
        elif interrupted is not None and getattr(interrupted, "messages", None):
            output = str(interrupted.messages[0].get("content", ""))
            obs_text = output

        self.instr.step_counter += 1
        step_id = self.instr.step_counter
        # Bindings point at the observation message that carries the file
        # contents. A submitted turn has no observation message, so the index
        # is deliberately past the end and no arm can rewrite anything for it.
        msg_obs_idx = len(self.messages) - 1 if observations else len(self.messages)
        reads, writes = self.instr.observe_step(
            command=command, output=obs_text, msg_obs_idx=msg_obs_idx,
            before=before, live_run=True, step_id=step_id,
        )

        # Arm 9 (Write-Through Viewport): Append viewport echo and refresh binding
        lean_skipped: List[str] = []
        if self.arm in LEAN_ARMS and writes:
            # No block for a write that tells the agent nothing new; its
            # binding is still refreshed, so the next diff of that file starts
            # from what the agent now knows.
            lean_skipped = sorted(redundant_writes(command, obs_text, writes, self._lean_scripts))
            for w_file in lean_skipped:
                self.instr.freshness_hook.register_read(msg_obs_idx, step_id, w_file, obs_text)
        shown_writes = [w for w in writes if w not in lean_skipped]
        if self.arm in WRITE_THROUGH_ARMS and shown_writes:
            viewport, shown = self.instr._viewport_blocks(shown_writes, before_contents=before_contents)
            if viewport and self.arm == InterventionArm.ARM_14_WRITE_THROUGH_DIFF_LINES:
                for w_file in shown:
                    base = self._line_base.get(w_file, before_contents.get(w_file))
                    now = self._disk_text(w_file)
                    note = line_number_note(w_file, base, now) if base is not None and now is not None else ""
                    if note:
                        viewport += "\n" + note
            if viewport:
                output += viewport
                obs_text += viewport
                self._append_to_observation(observations, viewport)
                for w_file in shown:
                    self.instr.freshness_hook.register_read(msg_obs_idx, step_id, w_file, obs_text)

        if self.arm == InterventionArm.ARM_14_WRITE_THROUGH_DIFF_LINES:
            for r_file in reads:  # the agent's line numbers now come from this read
                now = self._disk_text(r_file)
                if now is not None:
                    self._line_base[r_file] = now

        dep_record: List[Dict[str, Any]] = []
        if self.arm == InterventionArm.ARM_12_DEPENDENCY_WRITE_THROUGH and writes:
            before_defs = {}
            for w in writes:
                base = next((b for b in (before_contents.get(w), self._dep_cache.get(w)) if b is not None), None)
                base = base if base is not None else committed(self.workspace_dir, w)  # an empty file is a version
                if base is not None:
                    before_defs[w] = base
            viewed = {b.file_path for b in self.instr.freshness_hook.bindings}
            notice, dep_record = dependency_notice(self.workspace_dir, writes, before_defs, viewed,
                                                   written=set(self._dep_cache) | set(writes))
            if notice:
                output += notice
                obs_text += notice
                self._append_to_observation(observations, notice)
            for w in writes:
                try:
                    with open(os.path.join(self.workspace_dir, w), "r", encoding="utf-8", errors="replace") as f:
                        self._dep_cache[w] = f.read()
                except OSError:
                    self._dep_cache.pop(w, None)

        stale = self.instr.freshness_hook.get_stale_bindings()
        syntax_error = self._syntax_error_in(writes)
        self.step_records.append({
            "step_id": step_id,
            "action": message.get("content", ""),
            "command": command,
            "observation": output,
            "returncode": (observations[-1].get("extra", {}).get("returncode") if observations else None),
            "invalid_observations_count": len(stale),
            "stale_bindings": [b.file_path for b in stale],
            "reads_detected": list(reads),
            "writes_detected": list(writes),
            "context_messages_count": len(self.messages),
            "finish_reason": message.get("extra", {}).get("finish_reason", ""),
            "syntax_error": syntax_error,
            # State of the agent's beliefs as this step began.
            "bound_matches_disk": bound_matches_disk,
            "visible_stale_bytes": visible_stale_bytes,
            "context_bytes": context_bytes,
            # Observation messages the request that produced this action sent
            # shortened; empty when the model saw the full history.
            "compacted_msg_indices": list(getattr(self.model, "last_compacted", []) or []),
            # Synced-context arm: files the prompt that produced this action carried.
            "synced_files": list(self._synced_shown),
            "synced_chars": getattr(self, "_synced_chars", 0),
            # Dependency write-through: the stale uses its notice reported.
            "dependency_notice": dep_record,
            "dependency_notice_version": NOTICE_VERSION,
            # Lean diff: written files that got no block.
            "lean_skipped": lean_skipped,
        })
        if interrupted is not None:
            raise interrupted
        return observations


def build_agent(
    *,
    arm: InterventionArm,
    workspace_dir: str,
    model,
    step_limit: int,
    image: Optional[str] = None,
    command_timeout: int = 120,
    wall_time_limit_seconds: int = 1800,
    use_mtime_cache: bool = True,
    container_cpus: str = "4",
    container_memory: str = "8g",
    container_cwd: str = CONTAINER_CWD,
    container_network: Optional[str] = None,
    sandboxed: bool = True,
) -> FreshnessAgent:
    """Assemble the published agent over a task workspace.

    With `image`, the cell runs inside the instance's SWE-bench container and
    the workspace is bind-mounted over /testbed. That is the environment the
    benchmark defines, and it is also what the stock prompt describes, so the
    templates need no rewriting. Instrumentation still watches the workspace
    from the host, because the bind mount is the same files.

    Without `image`, execution is local and sandboxed, which is the fallback
    for a machine with no docker. `sandboxed=False` drops the macOS sandbox
    profile, for runs inside a container that is itself the sandbox.
    """
    cfg = load_mini_config()
    workspace_dir = os.path.abspath(workspace_dir)
    system_template = cfg["agent"]["system_template"]
    instance_template = cfg["agent"]["instance_template"]
    env_vars = dict(cfg["environment"].get("env", {}))

    if image:
        if container_cwd != CONTAINER_CWD:
            # The stock templates name /testbed; SWE-Bench Pro images keep the
            # repository at /app, so the prompt names where the files are.
            system_template = system_template.replace(CONTAINER_CWD, container_cwd)
            instance_template = instance_template.replace(CONTAINER_CWD, container_cwd)
        env = DockerEnvironment(
            image=image,
            cwd=container_cwd,
            timeout=command_timeout,
            env=env_vars,
            executable=DOCKER_BIN,
            run_args=[
                "--rm",
                "--platform", "linux/amd64",
                # One cell cannot take the machine: a previous sweep hung the
                # host when two astropy builds ran at once.
                f"--cpus={container_cpus}",
                f"--memory={container_memory}",
                "-v", f"{workspace_dir}:{container_cwd}",
            ] + (["--network", container_network] if container_network else []),
        )
    else:
        # The stock templates name /testbed, which exists only in the image.
        system_template = system_template.replace(CONTAINER_CWD, workspace_dir)
        instance_template = instance_template.replace(CONTAINER_CWD, workspace_dir)
        env_vars.update({
            "MAKEFLAGS": "-j1",
            "MAX_JOBS": "1",
            "NPY_NUM_BUILD_JOBS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONPATH": workspace_dir,
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": workspace_dir,
        })
        venv_bin = os.path.join(workspace_dir, ".venv-task", "bin")
        if os.path.isdir(venv_bin):
            env_vars["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
            env_vars["VIRTUAL_ENV"] = os.path.dirname(venv_bin)
        if sandboxed:
            if sys.platform != "darwin":
                raise RuntimeError("the local sandbox uses macOS sandbox-exec; pass an image, or "
                                   "sandboxed=False inside a container")
            env = SandboxedLocalEnvironment(
                sandbox_profile=write_sandbox_profile(workspace_dir),
                cwd=workspace_dir,
                timeout=command_timeout,
                env=env_vars,
            )
        else:
            env = LocalEnvironment(cwd=workspace_dir, timeout=command_timeout, env=env_vars)

    if arm == InterventionArm.ARM_0_PROMPT_ONLY:
        system_template = system_template + PROMPT_ONLY_POLICY

    return FreshnessAgent(
        model,
        env,
        arm=arm,
        workspace_dir=workspace_dir,
        use_mtime_cache=use_mtime_cache,
        system_template=system_template,
        instance_template=instance_template,
        step_limit=step_limit,
        cost_limit=0.0,
        wall_time_limit_seconds=wall_time_limit_seconds,
        max_consecutive_format_errors=cfg["agent"].get("max_consecutive_format_errors", 3),
    )
