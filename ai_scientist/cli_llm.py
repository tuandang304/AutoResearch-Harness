"""
Use locally installed, already logged-in coding-agent CLIs as LLM backends.

Supported providers (model string = "<provider>/<model>"):

- ``claude-code/<model>``  -> Claude Code  (``claude -p``), e.g. ``claude-code/opus``
- ``codex/<model>``        -> OpenAI Codex (``codex exec``), e.g. ``codex/gpt-5.6-luna``
- ``antigravity/<model>``  -> Google Antigravity CLI (``agy -p``), e.g. ``antigravity/gemini-3.1-pro-high``

Use ``<provider>/default`` to let the CLI pick the model from its own config.

The CLIs are run as plain text/vision completers: each call runs in an empty
scratch directory with tools disabled (or restricted as far as the CLI allows),
and function calling is emulated by asking for JSON that matches the schema.
Sampling parameters such as temperature are not exposed by the CLIs and are
ignored.

``CLIClient`` mimics the small part of the OpenAI client that this codebase
uses (``client.chat.completions.create(model=..., messages=..., n=...)``),
so the existing call sites keep working.
"""

import base64
import json
import logging
import os
import queue
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from types import SimpleNamespace
from collections import deque
from typing import Any

logger = logging.getLogger("ai-scientist")

CLI_PROVIDERS = ("claude-code", "codex", "antigravity")

# Suggested model names per provider; any "<provider>/<name>" string is accepted.
CLI_MODELS = [
    "claude-code/default",
    "claude-code/fable",
    "claude-code/opus",
    "claude-code/sonnet",
    "claude-code/haiku",
    "codex/default",
    "antigravity/default",
    "antigravity/gemini-3.1-pro-high",
    "antigravity/gemini-3.1-pro-low",
    "antigravity/gemini-3.8-flash-high",
    "antigravity/gemini-3.8-flash-medium",
    "antigravity/claude-opus-4-6-thinking",
    "antigravity/claude-sonnet-4-6",
]

# Models used by `launch_scientist_bfts.py --provider <name>`: "big" for code
# generation and the final write-up, "small" for feedback, summaries,
# citations, plot aggregation and review.
PROVIDER_PRESETS = {
    "claude-code": {"big": "claude-code/opus", "small": "claude-code/sonnet"},
    "codex": {"big": "codex/default", "small": "codex/default"},
    "antigravity": {
        "big": "antigravity/gemini-3.1-pro-high",
        "small": "antigravity/gemini-3.8-flash-medium",
    },
}

# Per-call wall clock limit and retry policy (overridable via env vars).
CLI_TIMEOUT = int(os.environ.get("AI_SCIENTIST_CLI_TIMEOUT", "1200"))
CLI_RETRIES = int(os.environ.get("AI_SCIENTIST_CLI_RETRIES", "3"))
CLI_STARTUP_TIMEOUT = int(os.environ.get("AI_SCIENTIST_CLI_STARTUP_TIMEOUT", "90"))

_NO_TOOLS_NOTE = (
    "You are being used as a plain text-completion model inside an automated "
    "research pipeline. Answer directly in your reply. Do not run commands, "
    "do not edit or create files, and do not browse the web."
)


class CLIError(RuntimeError):
    pass


def is_cli_model(model: str | None) -> bool:
    return bool(model) and model.split("/", 1)[0] in CLI_PROVIDERS and "/" in model


def split_cli_model(model: str) -> tuple[str, str | None]:
    if not is_cli_model(model):
        raise ValueError(f"Unsupported CLI model: {model!r}")
    provider, name = model.split("/", 1)
    return provider, (None if name in ("", "default") else name)


class _ModelList(list):
    """A list of model names that also accepts any "<cli-provider>/<model>" string.

    Used for argparse ``choices=`` so CLI models don't need to be enumerated.
    """

    def __contains__(self, item):
        return super().__contains__(item) or is_cli_model(item)


def with_cli_models(models: list[str]) -> list[str]:
    return _ModelList(models + [m for m in CLI_MODELS if m not in models])


# --------------------------------------------------------------------------- #
# Scratch space
# --------------------------------------------------------------------------- #


def _scratch_root(provider: str) -> str:
    """Directory where per-call scratch dirs are created.

    The Antigravity CLI ships as a snap, which can only read files inside its
    own snap home, so its scratch dir lives there (and needs a read_file allow
    rule, see scripts/setup_cli_providers.sh).
    """
    env = os.environ.get("AI_SCIENTIST_CLI_TMP")
    if env:
        return env
    if provider == "antigravity":
        snap_home = os.path.expanduser("~/snap/antigravity-cli/common")
        if os.path.isdir(snap_home):
            return os.path.join(snap_home, "ai_scientist_tmp")
    return os.path.join(tempfile.gettempdir(), "ai_scientist_cli")


def _image_bytes(url: str) -> tuple[bytes, str]:
    """Decode an OpenAI-style image_url (data URI or local path)."""
    if url.startswith("data:"):
        header, data = url.split(",", 1)
        media_type = header[5:].split(";")[0] or "image/jpeg"
        return base64.b64decode(data), media_type
    path = url[len("file://") :] if url.startswith("file://") else url
    with open(path, "rb") as f:
        data = f.read()
    ext = os.path.splitext(path)[1].lower()
    media_type = {".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}.get(
        ext, "image/jpeg"
    )
    return data, media_type


def _sniff_media_type(data: bytes, fallback: str) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return fallback


# --------------------------------------------------------------------------- #
# Message flattening
# --------------------------------------------------------------------------- #


def _split_content(content: Any) -> tuple[str, list[str]]:
    """Return (text, image_urls) for an OpenAI/Anthropic-style message content."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    texts, images = [], []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
        elif part.get("type") == "text":
            texts.append(part.get("text", ""))
        elif part.get("type") == "image_url":
            iu = part["image_url"]
            images.append(iu["url"] if isinstance(iu, dict) else iu)
        elif part.get("type") == "image" and part.get("source", {}).get("type") == "base64":
            src = part["source"]
            images.append(f"data:{src['media_type']};base64,{src['data']}")
    return "\n".join(texts), images


def flatten_messages(messages: list[dict]) -> tuple[str, str, list[str]]:
    """Collapse a chat history into (system_prompt, user_prompt, image_urls).

    The CLIs take a single prompt per invocation, so earlier turns are
    rendered as a transcript in front of the latest user message.
    """
    system_parts, turns, images = [], [], []
    for m in messages:
        text, imgs = _split_content(m.get("content"))
        images.extend(imgs)
        if m.get("role") == "system":
            system_parts.append(text)
        else:
            turns.append((m.get("role", "user"), text))

    if not turns:
        turns = [("user", "\n\n".join(system_parts))]
        system_parts = []

    *history, (last_role, last_text) = turns
    if history:
        transcript = "\n\n".join(
            f"<{role}_turn>\n{text}\n</{role}_turn>" for role, text in history
        )
        user_prompt = (
            "Below is the conversation so far, followed by the latest message "
            "from the user. Continue the conversation by replying to the latest "
            "message only.\n\n"
            f"<conversation_history>\n{transcript}\n</conversation_history>\n\n"
            f"<latest_{last_role}_message>\n{last_text}\n</latest_{last_role}_message>"
        )
    else:
        user_prompt = last_text
    return "\n\n".join(p for p in system_parts if p), user_prompt, images


# --------------------------------------------------------------------------- #
# Provider runners: each returns (text, usage_dict)
# --------------------------------------------------------------------------- #


def _run(cmd: list[str], cwd: str, stdin: str | None) -> subprocess.CompletedProcess:
    proc = None
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
        out, err = proc.communicate(stdin, timeout=CLI_TIMEOUT)
        return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    except FileNotFoundError:
        raise CLIError(
            f"'{cmd[0]}' was not found on PATH. Install it and log in first "
            f"(see README: 'LLM providers')."
        )
    except subprocess.TimeoutExpired:
        raise CLIError(f"{cmd[0]} timed out after {CLI_TIMEOUT}s")
    finally:
        if proc is not None:
            _stop_process_group(proc)


def _stop_process_group(proc):
    # Kill descendants even if the CLI parent already exited.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=5)


def _run_streaming(cmd: list[str], cwd: str, stdin: str, is_final) -> list[str]:
    """Run a CLI that emits NDJSON events; return its stdout lines.

    Kills the process if it prints nothing within CLI_STARTUP_TIMEOUT (agy
    occasionally hangs at startup) and returns as soon as ``is_final(event)``
    is true, without waiting for the process to exit on its own.
    """
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, errors="replace", start_new_session=True,
        )
    except FileNotFoundError:
        raise CLIError(
            f"'{cmd[0]}' was not found on PATH. Install it and log in first "
            f"(see README: 'LLM providers')."
        )
    lines: "queue.Queue[str | None]" = queue.Queue()
    stderr = deque(maxlen=100)

    def pump_stdout():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def send_stdin():
        # A CLI that never reads stdin must not block the timeout loop.
        try:
            proc.stdin.write(stdin)
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass

    threads = [threading.Thread(target=pump_stdout, daemon=True),
               threading.Thread(target=lambda: stderr.extend(proc.stderr), daemon=True),
               threading.Thread(target=send_stdin, daemon=True)]
    start = time.monotonic()
    for thread in threads:
        thread.start()

    out: list[str] = []
    output_size = 0
    try:
        while True:
            limit = CLI_TIMEOUT if out else CLI_STARTUP_TIMEOUT
            remaining = min(limit, CLI_TIMEOUT) - (time.monotonic() - start)
            if remaining <= 0:
                what = "timed out" if out else "produced no output (hung at startup)"
                raise CLIError(f"{cmd[0]} {what} after {limit}s")
            try:
                line = lines.get(timeout=min(remaining, 5))
            except queue.Empty:
                continue
            if line is None:
                break
            out.append(line)
            output_size += len(line)
            if output_size > 32 * 1024 * 1024:
                raise CLIError(f"{cmd[0]} exceeded the 32 MiB output limit")
            try:
                event = json.loads(line)
                if isinstance(event, dict) and is_final(event):
                    break
            except json.JSONDecodeError:
                pass
    finally:
        _stop_process_group(proc)
        for thread in threads:
            thread.join(timeout=1)
        proc.stdout.close()
        proc.stderr.close()
    if not out and stderr:
        out.append(json.dumps({"stderr": "".join(stderr)[-2000:]}))
    return out


def _run_claude_code(model, system, prompt, images, workdir):
    content = [{"type": "text", "text": prompt}]
    for url in images:
        data, media_type = _image_bytes(url)
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _sniff_media_type(data, media_type),
                    "data": base64.b64encode(data).decode(),
                },
            }
        )
    stdin = json.dumps(
        {"type": "user", "message": {"role": "user", "content": content}}
    )
    cmd = [
        "claude", "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
        "--tools", "",
        "--safe-mode",
        "--no-session-persistence",
        "--system-prompt", (system + "\n\n" if system else "") + _NO_TOOLS_NOTE,
    ]
    if model:
        cmd += ["--model", model]
    lines = _run_streaming(cmd, workdir, stdin + "\n", lambda e: e.get("type") == "result")

    result = None
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            result = event
    if result is None or result.get("is_error") or result.get("subtype") != "success":
        detail = (result or {}).get("result") or "".join(lines)[-2000:]
        raise CLIError(f"claude failed: {detail}")
    u = result.get("usage", {})
    usage = {
        "prompt": u.get("input_tokens", 0)
        + u.get("cache_read_input_tokens", 0)
        + u.get("cache_creation_input_tokens", 0),
        "completion": u.get("output_tokens", 0),
        "cached": u.get("cache_read_input_tokens", 0),
        "reasoning": u.get("output_tokens_details", {}).get("thinking_tokens", 0),
    }
    return result.get("result", ""), usage


def _run_codex(model, system, prompt, images, workdir):
    image_args = []
    for i, url in enumerate(images):
        data, media_type = _image_bytes(url)
        ext = _sniff_media_type(data, media_type).split("/")[-1].replace("jpeg", "jpg")
        path = os.path.join(workdir, f"image_{i}.{ext}")
        with open(path, "wb") as f:
            f.write(data)
        image_args += ["-i", path]

    full_prompt = f"{_NO_TOOLS_NOTE}\n\n"
    if system:
        full_prompt += f"<system_instructions>\n{system}\n</system_instructions>\n\n"
    full_prompt += prompt
    if images:
        full_prompt += f"\n\n({len(images)} image(s) are attached to this message.)"

    last_msg = os.path.join(workdir, "last_message.txt")
    cmd = [
        "codex", "exec",
        "--json",
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox", "read-only",
        "-C", workdir,
        "-o", last_msg,
        *image_args,
    ]
    if model:
        cmd += ["-m", model]
    cmd.append("-")  # read the prompt from stdin
    proc = _run(cmd, workdir, full_prompt)

    usage = {"prompt": 0, "completion": 0, "cached": 0, "reasoning": 0}
    error = None
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed":
            u = event.get("usage", {})
            usage = {
                "prompt": u.get("input_tokens", 0),
                "completion": u.get("output_tokens", 0),
                "cached": u.get("cached_input_tokens", 0),
                "reasoning": u.get("reasoning_output_tokens", 0),
            }
        elif event.get("type") in ("error", "turn.failed"):
            error = event
    text = ""
    if os.path.exists(last_msg):
        with open(last_msg) as f:
            text = f.read()
    if proc.returncode != 0 or error or not text.strip():
        raise CLIError(
            f"codex exited with code {proc.returncode}: "
            f"{json.dumps(error) if error else proc.stderr[-2000:]}"
        )
    return text, usage


def _run_antigravity(model, system, prompt, images, workdir):
    full_prompt = f"{_NO_TOOLS_NOTE}\n\n"
    if system:
        full_prompt += f"<system_instructions>\n{system}\n</system_instructions>\n\n"
    full_prompt += prompt
    if images:
        # agy's stream-json input only accepts text blocks, so images are
        # written to the scratch dir and opened with its view_file tool.
        paths = []
        for i, url in enumerate(images):
            data, media_type = _image_bytes(url)
            ext = _sniff_media_type(data, media_type).split("/")[-1].replace("jpeg", "jpg")
            path = os.path.join(workdir, f"image_{i}.{ext}")
            with open(path, "wb") as f:
                f.write(data)
            paths.append(path)
        full_prompt += (
            "\n\nThis message comes with the following attached images, in order. "
            "Open each of them with your view_file tool (this is the only tool you "
            "may use), then answer:\n" + "\n".join(f"- {p}" for p in paths)
        )

    stdin = json.dumps({"event": "user", "message": {"content": full_prompt}})
    cmd = ["agy", "-p=", "--input-format", "stream-json", "--output-format", "stream-json"]
    if model:
        cmd += ["--model", model]
    lines = _run_streaming(cmd, workdir, stdin + "\n", lambda e: e.get("event") == "result")

    result = None
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "result":
            result = event.get("result", {})
    if not result or result.get("status") != "SUCCESS" or not result.get("response", "").strip():
        detail = (result or {}).get("error") or (result or {}).get("denied_actions")
        raise CLIError(
            f"agy failed: {detail or ''.join(lines)[-2000:]}"
            + (
                "\nIf images were denied, run scripts/setup_cli_providers.sh to allow "
                "agy to read its scratch directory."
                if (result or {}).get("denied_actions")
                else ""
            )
        )
    u = result.get("usage", {})
    usage = {
        "prompt": u.get("input_tokens", 0),
        "completion": u.get("output_tokens", 0),
        "cached": u.get("cache_read_tokens", 0),
        "reasoning": u.get("thinking_tokens", 0),
    }
    return result["response"], usage


_RUNNERS = {
    "claude-code": _run_claude_code,
    "codex": _run_codex,
    "antigravity": _run_antigravity,
}


def complete(model: str, messages: list[dict]) -> tuple[str, dict]:
    """Run one completion through the CLI named by ``model``."""
    provider, name = split_cli_model(model)
    if CLI_RETRIES < 1 or CLI_TIMEOUT <= 0 or CLI_STARTUP_TIMEOUT <= 0:
        raise CLIError("CLI retries and timeouts must be positive")
    system, prompt, images = flatten_messages(messages)
    root = _scratch_root(provider)
    os.makedirs(root, exist_ok=True)

    last_err = None
    for attempt in range(CLI_RETRIES):
        workdir = tempfile.mkdtemp(prefix=f"{provider}_", dir=root)
        try:
            text, usage = _RUNNERS[provider](name, system, prompt, images, workdir)
            if not isinstance(text, str) or not text.strip():
                raise CLIError(f"{provider} returned an empty completion")
            return text, usage
        except CLIError as e:
            last_err = e
            if "was not found on PATH" in str(e):
                raise
            # startup hangs are not rate limits: retry right away
            wait = 2 if "hung at startup" in str(e) else 15 * (3**attempt)
            logger.warning(
                f"{model} call failed (attempt {attempt + 1}/{CLI_RETRIES}): {e}. "
                f"Retrying in {wait}s"
            )
            if attempt + 1 < CLI_RETRIES:
                time.sleep(wait)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    raise last_err


# --------------------------------------------------------------------------- #
# JSON / function-call emulation
# --------------------------------------------------------------------------- #


def json_instruction(name: str, description: str, schema: dict) -> str:
    return (
        f"\n\n# Required output format\n"
        f"Respond by calling the function `{name}` ({description}). Since function "
        f"calling is not available, output ONLY a single JSON object with the "
        f"function arguments inside a ```json code block, with no other text. "
        f"The JSON must validate against this JSON Schema:\n"
        f"```json\n{json.dumps(schema, indent=2)}\n```"
    )


def parse_json_output(text: str) -> Any:
    """Extract the first JSON object from a model reply."""
    candidates = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates.append(text)
    for c in candidates:
        c = c.strip()
        start, end = c.find("{"), c.rfind("}")
        if start == -1 or end <= start:
            continue
        try:
            return json.loads(c[start : end + 1])
        except json.JSONDecodeError:
            try:
                return json.loads(re.sub(r"[\x00-\x1F\x7F]", "", c[start : end + 1]))
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found in the model output")


# --------------------------------------------------------------------------- #
# OpenAI-compatible shim
# --------------------------------------------------------------------------- #


class _Completions:
    def __init__(self, client: "CLIClient"):
        self._client = client

    def create(self, model=None, messages=None, n=1, **_ignored):
        model = model or self._client.model
        choices, totals = [], {"prompt": 0, "completion": 0, "cached": 0, "reasoning": 0}
        for i in range(n or 1):
            text, usage = complete(model, messages or [])
            for k in totals:
                totals[k] += usage.get(k, 0)
            choices.append(
                SimpleNamespace(
                    index=i,
                    message=SimpleNamespace(role="assistant", content=text, tool_calls=None),
                    finish_reason="stop",
                )
            )
        return SimpleNamespace(
            id=f"cli-{uuid.uuid4().hex}",
            model=model,
            created=int(time.time()),
            choices=choices,
            system_fingerprint=None,
            usage=SimpleNamespace(
                prompt_tokens=totals["prompt"],
                completion_tokens=totals["completion"],
                total_tokens=totals["prompt"] + totals["completion"],
                completion_tokens_details=SimpleNamespace(reasoning_tokens=totals["reasoning"]),
                prompt_tokens_details=SimpleNamespace(cached_tokens=totals["cached"]),
            ),
        )


class CLIClient:
    """Minimal stand-in for ``openai.OpenAI`` that routes to a local CLI."""

    def __init__(self, model: str):
        if not is_cli_model(model):
            raise ValueError(f"{model} is not a CLI model")
        self.model = model
        self.provider = model.split("/", 1)[0]
        self.chat = SimpleNamespace(completions=_Completions(self))

    def __repr__(self):
        return f"CLIClient({self.model!r})"
