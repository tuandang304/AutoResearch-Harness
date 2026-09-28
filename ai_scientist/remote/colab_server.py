"""
Remote code-execution server for running AutoResearch-Harness experiments on a GPU box
(e.g. Google Colab) while the tree search itself runs on your local machine.

Standard library only, so it can be dropped into a Colab runtime as a single
file (notebooks/colab_gpu_executor.ipynb embeds it).

    AISCI_TOKEN=<secret> python colab_server.py --port 8765

Protocol (all endpoints require ``Authorization: Bearer <token>``):

    GET    /health                 -> GPU / queue info
    POST   /jobs                   -> {"job_id"}; body: {code, timeout, agent_file_name,
                                      env_vars, max_file_mb, workspace_tgz_b64}
    GET    /jobs/<id>              -> {"status": queued|running|done, "result": {...}}
    GET    /jobs/<id>/workspace    -> tar.gz of the job workspace after the run
    DELETE /jobs/<id>              -> cancel (if running) and delete the job

Each job runs in a fresh copy of the uploaded workspace, in its own process
group, with the same semantics as ai_scientist/treesearch/interpreter.py
(output capture, exception summary, SIGINT on timeout, then SIGKILL).
"""

import argparse
import base64
import binascii
import hashlib
import hmac
import json
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if __package__:
    from .workspace import MAX_ARCHIVE_BYTES, extract_workspace, pack_workspace as pack_files
else:  # standalone files embedded in the Colab notebook
    from workspace import MAX_ARCHIVE_BYTES, extract_workspace, pack_workspace as pack_files

JOBS_ROOT = os.environ.get("AISCI_JOBS_ROOT", "/content/aisci_jobs")
MAX_CONCURRENT = int(os.environ.get("AISCI_MAX_CONCURRENT", "2"))
MAX_OUTPUT_CHARS = 2_000_000
JOB_TTL_SECONDS = 6 * 3600
EXC_FILE = ".aisci_exc.json"
MAX_REQUEST_BYTES = 2 * MAX_ARCHIVE_BYTES
MAX_JOBS = 64


def validate_spec(spec):
    if not isinstance(spec, dict) or not isinstance(spec.get("code"), str):
        raise ValueError("code must be a string")
    for key, default, maximum in (("timeout", 3600, 86400), ("max_file_mb", 100, 512)):
        value = spec.setdefault(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError(f"{key} must be positive and at most {maximum}")
    name = spec.setdefault("agent_file_name", "runfile.py")
    if not isinstance(name, str) or not name or name in (".", "..", EXC_FILE) or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError("agent_file_name must be a plain filename")
    env = spec.setdefault("env_vars", {})
    if not isinstance(env, dict) or any(not isinstance(k, str) or not k or "=" in k or "\x00" in k or not isinstance(v, str) or "\x00" in v for k, v in env.items()):
        raise ValueError("env_vars must contain string environment names and values")
    if "job_id" in spec and (not isinstance(spec["job_id"], str) or len(spec["job_id"]) != 32 or any(c not in "0123456789abcdef" for c in spec["job_id"])):
        raise ValueError("job_id must be a 32-character lowercase hexadecimal ID")
    if not isinstance(spec.get("workspace_tgz_b64", ""), str):
        raise ValueError("workspace_tgz_b64 must be a base64 string")

RUNNER_SOURCE = r'''
import json, os, sys, traceback
fname = sys.argv[1]
exc_file = sys.argv[2]
sys.path.append(os.getcwd())
with open(fname) as f:
    code = f.read()
try:
    exec(compile(code, fname, "exec"), {"__name__": "__main__", "__file__": fname})
except BaseException as e:
    tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
    tb_str = "".join(l for l in tb_lines if "aisci_runner" not in l and "importlib" not in l)
    tb_str = tb_str.replace(os.path.join(os.getcwd(), fname), fname)
    sys.stderr.write(tb_str)
    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))
    stack = [(t.filename, t.lineno, t.name, t.line) for t in traceback.extract_tb(e.__traceback__)]
    name = e.__class__.__name__
    if name == "KeyboardInterrupt":
        name = "TimeoutError"
    with open(exc_file, "w") as f:
        json.dump({"exc_type": name, "exc_info": exc_info, "exc_stack": stack}, f)
    sys.exit(1)
'''


def naturaldelta(seconds: float) -> str:
    try:
        import humanize

        return humanize.naturaldelta(seconds)
    except ImportError:
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds} seconds"
        if seconds < 3600:
            return f"{round(seconds / 60)} minutes"
        return f"{seconds / 3600:.1f} hours"


def gpu_info() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return "no GPU"


class Job:
    def __init__(self, spec: dict):
        self.id = spec.get("job_id") or uuid.uuid4().hex
        self.spec = spec
        self.status = "queued"
        self.result = None
        self.proc: subprocess.Popen | None = None
        self.cancelled = False
        self.created = time.time()
        self.finished = None
        self.lock = threading.Lock()
        self.dir = os.path.join(JOBS_ROOT, self.id)
        self.ws = os.path.join(self.dir, "ws")


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
QUEUE: "queue.Queue[Job]" = queue.Queue()


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    return text[:half] + "\n\n[... output truncated by remote executor ...]\n\n" + text[-half:]


def run_job(job: Job) -> None:
    spec = job.spec
    agent_file = spec.get("agent_file_name", "runfile.py")
    timeout = spec.get("timeout")
    with open(os.path.join(job.ws, agent_file), "w") as f:
        f.write(spec["code"])
    runner = os.path.join(job.dir, "aisci_runner.py")
    with open(runner, "w") as f:
        f.write(RUNNER_SOURCE)

    env = os.environ.copy()
    # The control-plane secret is never passed to generated experiment code.
    env.pop("AISCI_TOKEN", None)
    env.update({k: str(v) for k, v in (spec.get("env_vars") or {}).items()})
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLBACKEND", "Agg")

    start = time.monotonic()
    output = tempfile.TemporaryFile()
    with job.lock:
        if job.cancelled:
            output.close()
            return
        job.proc = subprocess.Popen(
        [sys.executable, "-u", runner, agent_file, EXC_FILE],
        cwd=job.ws, env=env, stdout=output, stderr=subprocess.STDOUT,
        text=True, errors="replace", start_new_session=True,
        )
    timed_out = False
    try:
        job.proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(job.proc.pid, signal.SIGINT)
            job.proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(job.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            job.proc.wait()
    finally:
        # Also reap descendants left behind by a normally exited parent.
        try:
            os.killpg(job.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        size = output.seek(0, os.SEEK_END)
        output.seek(0)
        raw = output.read(MAX_OUTPUT_CHARS)
        if size > MAX_OUTPUT_CHARS:
            output.seek(-MAX_OUTPUT_CHARS // 2, os.SEEK_END)
            raw = raw[:MAX_OUTPUT_CHARS // 2] + b"\n[... output truncated ...]\n" + output.read()
        output.close()
        out = raw.decode("utf-8", errors="replace")
    exec_time = time.monotonic() - start

    exc_type = exc_info = exc_stack = None
    exc_path = os.path.join(job.ws, EXC_FILE)
    if os.path.exists(exc_path):
        with open(exc_path) as f:
            exc = json.load(f)
        os.remove(exc_path)
        exc_type, exc_info, exc_stack = exc["exc_type"], exc["exc_info"], exc["exc_stack"]
    elif job.proc.returncode not in (0, None):
        # killed by a signal or crashed without a Python exception (e.g. OOM kill)
        exc_type = "RuntimeError"
        exc_info = {"args": [f"process exited with code {job.proc.returncode}"]}
        exc_stack = []
    if timed_out or job.cancelled:
        exc_type = "TimeoutError"
        exc_info = exc_info or {}
        exc_stack = exc_stack or []
        exec_time = timeout or exec_time

    term_out = [_truncate(out or "")]
    if exc_type == "TimeoutError":
        term_out.append(
            f"TimeoutError: Execution exceeded the time limit of {naturaldelta(timeout or 0)}"
        )
    else:
        term_out.append(
            f"Execution time: {naturaldelta(exec_time)} seconds "
            f"(time limit is {naturaldelta(timeout or 0)})."
        )
    job.result = {
        "term_out": term_out,
        "exec_time": exec_time,
        "exc_type": exc_type,
        "exc_info": exc_info,
        "exc_stack": exc_stack,
    }


def worker_loop() -> None:
    while True:
        job = QUEUE.get()
        try:
            if job.cancelled:
                continue
            job.status = "running"
            run_job(job)
        except Exception:
            job.result = {
                "term_out": ["Remote executor error:\n" + traceback.format_exc()],
                "exec_time": 0.0,
                "exc_type": "RemoteExecutorError",
                "exc_info": {},
                "exc_stack": [],
            }
        finally:
            job.status = "done"
            job.finished = time.time()
            if job.cancelled:
                shutil.rmtree(job.dir, ignore_errors=True)
            QUEUE.task_done()


def janitor_loop() -> None:
    while True:
        time.sleep(600)
        now = time.time()
        with JOBS_LOCK:
            stale = [j for j in JOBS.values() if j.status == "done" and now - (j.finished or now) > JOB_TTL_SECONDS]
            for j in stale:
                JOBS.pop(j.id, None)
        for j in stale:
            shutil.rmtree(j.dir, ignore_errors=True)


def pack_workspace(ws: str, max_file_mb: float) -> tuple[bytes, list[str]]:
    data, _, skipped = pack_files(ws, max_file_mb)
    return data, skipped


class Handler(BaseHTTPRequestHandler):
    server_version = "AutoResearchHarnessRemote/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _send(self, code: int, body, content_type="application/json", headers=None):
        data = json.dumps(body).encode() if content_type == "application/json" else body
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        if hmac.compare_digest(self.headers.get("Authorization", "").encode(), f"Bearer {self.server.token}".encode()):
            return True
        self._send(401, {"error": "unauthorized"})
        return False

    def _job(self, job_id: str) -> Job | None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None:
            self._send(404, {"error": "unknown job"})
        return job

    def do_GET(self):
        if not self._authorized():
            return
        parts = self.path.strip("/").split("/")
        if parts == ["health"]:
            with JOBS_LOCK:
                states = [j.status for j in JOBS.values()]
            self._send(200, {
                "ok": True,
                "protocol_version": 2,
                "gpu": gpu_info(),
                "python": sys.version.split()[0],
                "max_concurrent": MAX_CONCURRENT,
                "running": states.count("running"),
                "queued": states.count("queued"),
            })
        elif len(parts) == 2 and parts[0] == "jobs":
            job = self._job(parts[1])
            if job:
                self._send(200, {"status": job.status, "result": job.result, "created": job.created})
        elif len(parts) == 3 and parts[0] == "jobs" and parts[2] == "workspace":
            job = self._job(parts[1])
            if job is None:
                return
            if job.status != "done":
                return self._send(409, {"error": "job not finished"})
            try:
                data, skipped = pack_workspace(job.ws, job.spec.get("max_file_mb", 100))
            except (ValueError, OSError) as exc:
                return self._send(422, {"error": str(exc)})
            self._send(200, data, "application/gzip", {"X-Skipped-Files": json.dumps(skipped)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._authorized():
            return
        if self.path.strip("/") != "jobs":
            return self._send(404, {"error": "not found"})
        job = None
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_REQUEST_BYTES:
                return self._send(413, {"error": "Request body is empty or exceeds size limit"})
            self.connection.settimeout(30)
            body = self.rfile.read(length)
            spec = json.loads(body)
            validate_spec(spec)
            fingerprint = hashlib.sha256(body).hexdigest()
            with JOBS_LOCK:
                existing = JOBS.get(spec.get("job_id"))
                if existing:
                    if existing.fingerprint != fingerprint:
                        return self._send(409, {"error": "job_id already used for a different request"})
                    return self._send(200, {"job_id": existing.id})
                if len(JOBS) >= MAX_JOBS:
                    return self._send(429, {"error": "Executor job capacity reached"})
                tgz = base64.b64decode(spec.pop("workspace_tgz_b64", ""), validate=True)
                job = Job(spec)
                job.fingerprint = fingerprint
                os.makedirs(job.ws, exist_ok=False)
                if tgz:
                    extract_workspace(tgz, job.ws)
                JOBS[job.id] = job
                QUEUE.put(job)
        except (ValueError, binascii.Error, tarfile.TarError, OSError) as exc:
            if job is not None:
                shutil.rmtree(job.dir, ignore_errors=True)
            return self._send(400, {"error": str(exc)})
        self._send(200, {"job_id": job.id})

    def do_DELETE(self):
        if not self._authorized():
            return
        parts = self.path.strip("/").split("/")
        if len(parts) != 2 or parts[0] != "jobs":
            return self._send(404, {"error": "not found"})
        with JOBS_LOCK:
            job = JOBS.pop(parts[1], None)
        if job is None:
            return self._send(404, {"error": "unknown job"})
        with job.lock:
            job.cancelled = True
            if job.proc is not None:
                try:
                    os.killpg(job.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if job.status == "done":
                shutil.rmtree(job.dir, ignore_errors=True)
        self._send(200, {"deleted": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    token = os.environ.get("AISCI_TOKEN")
    if not token:
        sys.exit("Set AISCI_TOKEN to a random secret before starting the server.")
    if MAX_CONCURRENT < 1:
        sys.exit("AISCI_MAX_CONCURRENT must be at least 1")
    os.makedirs(JOBS_ROOT, exist_ok=True)
    for _ in range(MAX_CONCURRENT):
        threading.Thread(target=worker_loop, daemon=True).start()
    threading.Thread(target=janitor_loop, daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.token = token
    print(f"AutoResearch-Harness remote executor on {args.host}:{args.port} "
          f"(max {MAX_CONCURRENT} concurrent jobs, GPU: {gpu_info()})", flush=True)
    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with JOBS_LOCK:
            for job in JOBS.values():
                with job.lock:
                    job.cancelled = True
                    if job.proc is not None:
                        try:
                            os.killpg(job.proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
        server.server_close()


if __name__ == "__main__":
    main()
