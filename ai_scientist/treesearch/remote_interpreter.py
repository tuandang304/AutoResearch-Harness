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
import io
import json
import logging
import os
import tarfile
import time
from pathlib import Path

import requests

from .interpreter import ExecutionResult, Interpreter

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
            cfg = json.loads(path.read_text())
            url, token = url or cfg.get("url"), token or cfg.get("token")
    if not (url and token):
        raise RemoteExecutorUnavailable(
            "Remote executor is not configured. Start notebooks/colab_gpu_executor.ipynb "
            f"on Colab and paste the printed JSON into {_config_path()} "
            "(or set AI_SCIENTIST_REMOTE_URL and AI_SCIENTIST_REMOTE_TOKEN)."
        )
    return url.rstrip("/"), token


class RemoteInterpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
        env_vars: dict[str, str] = {},
        max_file_mb: float = 100,
        wait_minutes: float = 60,
    ):
        self.working_dir = Path(working_dir).resolve()
        assert self.working_dir.exists(), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.agent_file_name = agent_file_name
        self.env_vars = env_vars
        self.max_file_mb = max_file_mb
        self.wait_minutes = wait_minutes
        self.process = None  # interface compatibility with Interpreter
        self._job_id = None

    # ------------------------------------------------------------------ http

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """HTTP request that waits (up to wait_minutes) for the executor to be reachable."""
        deadline = time.time() + self.wait_minutes * 60
        warned = False
        while True:
            try:
                url, token = load_remote_config()
                resp = requests.request(
                    method,
                    url + path,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=REQUEST_TIMEOUT,
                    **kwargs,
                )
                if resp.status_code == 401:
                    raise RemoteExecutorUnavailable(
                        "Remote executor rejected the token; check remote_executor.json"
                    )
                # 502/503/504/530: tunnel up but Colab side gone or restarting
                if resp.status_code < 500:
                    if warned:
                        logger.warning("Remote executor is reachable again")
                    return resp
                err = f"HTTP {resp.status_code}"
            except requests.RequestException as e:
                err = str(e)
            if time.time() > deadline:
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
            time.sleep(15)

    # ------------------------------------------------------------- workspace

    def _pack_workspace(self) -> tuple[str, set[str]]:
        limit = self.max_file_mb * 1024 * 1024
        buf, sent = io.BytesIO(), set()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for path in self.working_dir.rglob("*"):
                if not path.is_file() or path.is_symlink():
                    continue
                rel = path.relative_to(self.working_dir).as_posix()
                if path.stat().st_size > limit:
                    logger.warning(f"Not uploading {rel}: larger than {self.max_file_mb} MB")
                    continue
                tar.add(path, arcname=rel)
                sent.add(rel)
        return base64.b64encode(buf.getvalue()).decode(), sent

    def _apply_workspace(self, data: bytes, sent: set[str], skipped: list[str]) -> None:
        received = set()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            members = tar.getmembers()
            received = {m.name for m in members if m.isfile()}
            try:
                tar.extractall(self.working_dir, filter="data")
            except TypeError:
                tar.extractall(self.working_dir)
        # files the remote code deleted (e.g. os.remove) disappear locally too
        for rel in sent - received - set(skipped):
            (self.working_dir / rel).unlink(missing_ok=True)
        for rel in skipped:
            logger.warning(f"Remote file {rel} is larger than {self.max_file_mb} MB; left on Colab")

    # ------------------------------------------------------------ interface

    def run(self, code: str, reset_session=True) -> ExecutionResult:
        # Keep a local copy of the executed file, like Interpreter does.
        (self.working_dir / self.agent_file_name).write_text(code)
        tgz_b64, sent = self._pack_workspace()
        spec = {
            "code": code,
            "timeout": self.timeout,
            "agent_file_name": self.agent_file_name,
            "env_vars": {k: v for k, v in self.env_vars.items() if v is not None},
            "max_file_mb": self.max_file_mb,
            "workspace_tgz_b64": tgz_b64,
        }

        for _attempt in range(3):
            resp = self._request("POST", "/jobs", json=spec)
            resp.raise_for_status()
            self._job_id = resp.json()["job_id"]
            logger.info(f"Submitted remote job {self._job_id} for {self.working_dir}")

            status = None
            while True:
                time.sleep(POLL_SECONDS)
                resp = self._request("GET", f"/jobs/{self._job_id}")
                if resp.status_code == 404:  # executor restarted and lost the job
                    logger.warning(f"Remote job {self._job_id} was lost; resubmitting")
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
            self._request("DELETE", f"/jobs/{self._job_id}")
            self._job_id = None

            r = status["result"]
            return ExecutionResult(
                term_out=r["term_out"],
                exec_time=r["exec_time"],
                exc_type=r["exc_type"],
                exc_info=r["exc_info"],
                exc_stack=[tuple(s) for s in r["exc_stack"]] if r["exc_stack"] else r["exc_stack"],
            )
        raise RemoteExecutorUnavailable("Remote job was lost 3 times in a row; giving up")

    def cleanup_session(self):
        if self._job_id is not None:
            try:
                url, token = load_remote_config()
                requests.delete(
                    f"{url}/jobs/{self._job_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=REQUEST_TIMEOUT,
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
    health = RemoteInterpreter(tempfile.mkdtemp(), wait_minutes=0.1)._request("GET", "/health")
    print("Health:", health.json())
    with tempfile.TemporaryDirectory() as d:
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
    parser.add_argument("--check", action="store_true", help="run a GPU smoke test on the remote executor")
    if parser.parse_args().check:
        raise SystemExit(0 if _check() else 1)
    parser.print_help()
