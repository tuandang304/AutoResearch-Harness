"""
Drop-in replacement for ``Interpreter`` that runs the experiment code on a
remote GPU box (e.g. Google Colab running ai_scientist/remote/colab_server.py).

The worker's workspace directory is uploaded before every run and the remote
workspace is mirrored back afterwards, so the rest of the tree search (which
reads ``working/*.npy`` and plots from local disk) works unchanged.

Connection settings are read, in order of precedence, from
``AI_SCIENTIST_REMOTE_URL`` / ``AI_SCIENTIST_REMOTE_TOKEN`` or from the JSON file
``remote_executor.json`` ({"url": ..., "token": ...}) in the repo root (or the
path in ``AI_SCIENTIST_REMOTE_CONFIG``). They are re-read on every connection
attempt, so after restarting Colab you only need to update that file; running
experiments wait for the executor to come back.
"""

import argparse
import base64
import json
import logging
import os
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import requests

from .interpreter import ExecutionResult, Interpreter
from ai_scientist.remote.workspace import extract_workspace, pack_workspace, safe_path

logger = logging.getLogger("ai-scientist")

POLL_SECONDS = 5
REQUEST_TIMEOUT = 90  # below Cloudflare's ~100s quick-tunnel response limit


class RemoteExecutorUnavailable(RuntimeError):
    pass


def _config_path() -> Path:
    if os.environ.get("AI_SCIENTIST_REMOTE_CONFIG"):
        return Path(os.environ["AI_SCIENTIST_REMOTE_CONFIG"])
    root = os.environ.get("AI_SCIENTIST_ROOT") or Path(__file__).resolve().parents[2]
    return Path(root) / "remote_executor.json"


def load_remote_config() -> tuple[str, str]:
    url = os.environ.get("AI_SCIENTIST_REMOTE_URL")
    token = os.environ.get("AI_SCIENTIST_REMOTE_TOKEN")
    if not (url and token):
        path = _config_path()
        if path.exists():
            try:
                cfg = json.loads(path.read_text())
                if not isinstance(cfg, dict):
                    raise ValueError("expected a JSON object")
            except (OSError, ValueError) as exc:
                raise RemoteExecutorUnavailable(
                    f"Invalid remote configuration at {path}"
                ) from exc
            url, token = url or cfg.get("url"), token or cfg.get("token")
    if not (url and token):
        raise RemoteExecutorUnavailable(
            "Remote executor is not configured. Start notebooks/colab_gpu_executor.ipynb "
            f"on Colab and paste the printed JSON into {_config_path()} "
            "(or set AI_SCIENTIST_REMOTE_URL and AI_SCIENTIST_REMOTE_TOKEN)."
        )
    if not isinstance(url, str) or not isinstance(token, str):
        raise RemoteExecutorUnavailable("Remote url and token must be strings")
    parts = urlsplit(url)
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise RemoteExecutorUnavailable(
            "Remote URL must be an HTTP(S) endpoint without credentials, query or fragment"
        )
    if parts.scheme == "http" and parts.hostname not in (
        "localhost",
        "127.0.0.1",
        "::1",
    ):
        raise RemoteExecutorUnavailable(
            "Remote endpoints require HTTPS; HTTP is supported only on localhost"
        )
    if any(c in token for c in "\r\n"):
        raise RemoteExecutorUnavailable("Invalid remote token")
    return url.rstrip("/"), token


class RemoteInterpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
        env_vars: dict[str, str] | None = None,
        max_file_mb: float = 100,
        wait_minutes: float = 60,
    ):
        self.working_dir = Path(working_dir).resolve()
        if not self.working_dir.is_dir():
            raise ValueError(f"Working directory {self.working_dir} does not exist")
        if timeout <= 0 or max_file_mb <= 0 or wait_minutes < 0:
            raise ValueError(
                "Timeout and file limit must be positive; wait_minutes must be nonnegative"
            )
        if (
            Path(agent_file_name).name != agent_file_name
            or agent_file_name in ("", ".", "..")
            or "\\" in agent_file_name
        ):
            raise ValueError("agent_file_name must be a plain filename")
        self.timeout = timeout
        self.agent_file_name = agent_file_name
        self.env_vars = dict(env_vars or {})
        self.max_file_mb = max_file_mb
        self.wait_minutes = wait_minutes
        self.process = None  # interface compatibility with Interpreter
        self._job_id = None

    # ------------------------------------------------------------------ http

    def _request(
        self, method: str, path: str, *, _deadline=None, **kwargs
    ) -> requests.Response:
        """HTTP request that waits (up to wait_minutes) for the executor to be reachable."""
        deadline = time.monotonic() + self.wait_minutes * 60
        if _deadline is not None:
            deadline = min(deadline, _deadline)
        warned = False
        while True:
            try:
                url, token = load_remote_config()
                resp = requests.request(
                    method,
                    url + path,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=min(REQUEST_TIMEOUT, max(0.1, deadline - time.monotonic())),
                    allow_redirects=False,
                    **kwargs,
                )
                if resp.status_code in (401, 403):
                    raise RemoteExecutorUnavailable(
                        "Remote executor rejected the token; check remote_executor.json"
                    )
                # 502/503/504/530: tunnel up but Colab side gone or restarting
                if 300 <= resp.status_code < 400:
                    raise RemoteExecutorUnavailable(
                        "Remote endpoint redirected; update remote_executor.json"
                    )
                if resp.status_code < 500 and resp.status_code != 429:
                    if warned:
                        logger.warning("Remote executor is reachable again")
                    return resp
                err = f"HTTP {resp.status_code}"
            except requests.RequestException as e:
                err = str(e)
            if time.monotonic() >= deadline:
                raise RemoteExecutorUnavailable(
                    f"Remote executor unreachable for {self.wait_minutes} min ({err}). "
                    f"Restart the Colab notebook and update {_config_path()}."
                )
            if not warned:
                logger.warning(
                    f"Remote executor unreachable ({err}); waiting up to "
                    f"{self.wait_minutes} min. If Colab restarted, update {_config_path()}."
                )
                warned = True
            time.sleep(min(15, max(0, deadline - time.monotonic())))

    # ------------------------------------------------------------- workspace

    def _pack_workspace(self) -> tuple[str, set[str]]:
        data, sent, skipped = pack_workspace(self.working_dir, self.max_file_mb)
        for rel in skipped:
            logger.warning("Not uploading %s: larger than %s MB", rel, self.max_file_mb)
        return base64.b64encode(data).decode(), sent

    def _apply_workspace(self, data: bytes, sent: set[str], skipped: list[str]) -> None:
        received = extract_workspace(data, self.working_dir)
        # files the remote code deleted (e.g. os.remove) disappear locally too
        for rel in sent - received - set(skipped):
            safe_path(self.working_dir, rel).unlink(missing_ok=True)
        for rel in skipped:
            logger.warning(
                f"Remote file {rel} is larger than {self.max_file_mb} MB; left on Colab"
            )

    # ------------------------------------------------------------ interface

    def run(self, code: str, reset_session=True) -> ExecutionResult:
        try:
            return self._run(code)
        except BaseException:
            self.cleanup_session()
            raise

    def _run(self, code: str) -> ExecutionResult:
        # Keep a local copy of the executed file, like Interpreter does.
        safe_path(self.working_dir, self.agent_file_name).write_text(code)
        tgz_b64, sent = self._pack_workspace()
        spec = {
            "job_id": uuid.uuid4().hex,
            "code": code,
            "timeout": self.timeout,
            "agent_file_name": self.agent_file_name,
            "env_vars": {k: v for k, v in self.env_vars.items() if v is not None},
            "max_file_mb": self.max_file_mb,
            "workspace_tgz_b64": tgz_b64,
        }

        for _attempt in range(3):
            # Keep the ID when retrying a lost POST response: the server deduplicates it.
            self._job_id = spec["job_id"]
            resp = self._request("POST", "/jobs", json=spec)
            resp.raise_for_status()
            self._job_id = resp.json()["job_id"]
            logger.info(f"Submitted remote job {self._job_id} for {self.working_dir}")

            status = None
            deadline = time.monotonic() + self.timeout + self.wait_minutes * 60 + 60
            while True:
                if time.monotonic() >= deadline:
                    raise RemoteExecutorUnavailable(
                        "Remote job exceeded its execution and queue/wait budget"
                    )
                time.sleep(POLL_SECONDS)
                resp = self._request("GET", f"/jobs/{self._job_id}", _deadline=deadline)
                if resp.status_code == 404:  # executor restarted and lost the job
                    logger.warning(f"Remote job {self._job_id} was lost; resubmitting")
                    # A restarted server may retain directories but lose its registry.
                    spec["job_id"] = uuid.uuid4().hex
                    break
                resp.raise_for_status()
                status = resp.json()
                if status["status"] == "done":
                    break
            if status is None or status.get("status") != "done":
                continue

            resp = self._request("GET", f"/jobs/{self._job_id}/workspace")
            resp.raise_for_status()
            skipped = json.loads(resp.headers.get("X-Skipped-Files", "[]"))
            self._apply_workspace(resp.content, sent, skipped)
            if skipped:
                logger.warning(
                    "Remote job %s retained for manual retrieval of skipped files (server TTL applies)",
                    self._job_id,
                )
                self._job_id = None
            else:
                self.cleanup_session()

            r = status["result"]
            return ExecutionResult(
                term_out=r["term_out"],
                exec_time=r["exec_time"],
                exc_type=r["exc_type"],
                exc_info=r["exc_info"],
                exc_stack=(
                    [tuple(s) for s in r["exc_stack"]]
                    if r["exc_stack"]
                    else r["exc_stack"]
                ),
            )
        raise RemoteExecutorUnavailable(
            "Remote job was lost 3 times in a row; giving up"
        )

    def cleanup_session(self):
        if self._job_id is not None:
            try:
                url, token = load_remote_config()
                requests.delete(
                    f"{url}/jobs/{self._job_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=5,
                    allow_redirects=False,
                )
            except Exception:
                pass
            self._job_id = None


def make_interpreter(cfg, working_dir, **kwargs):
    """Interpreter for experiment code, chosen by ``cfg.exec.backend`` (local | colab)."""
    common = dict(
        working_dir=working_dir,
        timeout=cfg.exec.timeout,
        format_tb_ipython=cfg.exec.format_tb_ipython,
        agent_file_name=cfg.exec.agent_file_name,
        **kwargs,
    )
    backend = cfg.exec.get("backend", "local")
    if backend == "local":
        return Interpreter(**common)
    if backend in ("colab", "remote"):
        return RemoteInterpreter(
            **common,
            max_file_mb=cfg.exec.get("remote_max_file_mb", 100),
            wait_minutes=cfg.exec.get("remote_wait_minutes", 60),
        )
    raise ValueError(f"Unknown exec.backend {backend!r} (expected 'local' or 'colab')")


def _check():
    """Smoke test: run a small CUDA program on the remote executor."""
    import tempfile

    url, _ = load_remote_config()
    print(f"Remote executor: {url}")
    with tempfile.TemporaryDirectory() as d:
        health = RemoteInterpreter(d, wait_minutes=0.1)._request("GET", "/health")
        health.raise_for_status()
        print("Health:", health.json())
        os.makedirs(os.path.join(d, "working"))
        interp = RemoteInterpreter(d, timeout=120, wait_minutes=0.5)
        res = interp.run(
            "import os, torch\n"
            "print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())\n"
            "if torch.cuda.is_available(): print('device:', torch.cuda.get_device_name(0))\n"
            "x = torch.randn(2048, 2048, device='cuda' if torch.cuda.is_available() else 'cpu')\n"
            "print('matmul ok:', float((x @ x).sum()) != 0)\n"
            "open(os.path.join('working', 'hello.txt'), 'w').write('from colab')\n"
        )
        print("".join(res.term_out))
        synced = os.path.exists(os.path.join(d, "working", "hello.txt"))
        print("exc_type:", res.exc_type, "| workspace synced back:", synced)
        return res.exc_type is None and synced


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote executor utilities")
    parser.add_argument(
        "--check",
        action="store_true",
        help="run a GPU smoke test on the remote executor",
    )
    if parser.parse_args().check:
        raise SystemExit(0 if _check() else 1)
    parser.print_help()
