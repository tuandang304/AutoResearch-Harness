"""Provision and release Colab runtimes with the ``colab`` CLI, one per compute tier.

The existing HTTP executor (ai_scientist/remote/colab_server.py) is started on a
CLI-created VM, exposed through a Cloudflare quick tunnel and recorded in an
executor file that RemoteInterpreter reads. A run owns a *pool*: one state file
and at most one session per tier (cpu, T4, L4, A100, ...).

With ``selection: llm`` the code-writing model declares the tier each script
needs on its first line (``# COMPUTE: T4  # reason``), choosing from the
configured tiers with their memory, rate and remaining budget in the prompt.
Undeclared follow-up code (metric parsing, plotting) stays on the node's tier.

Compute is spent only while it is needed:

* each tier's VM is created lazily, when the first job for it arrives;
* the launcher releases the pool as soon as the tree search ends;
* a detached watchdog stops each VM after its idle limit, stops all of them at
  the run's compute-unit budget, and when the controller process disappears;
* ``max_sessions`` bounds concurrent VMs; an idle one is stopped to make room;
* a tier whose VMs are all overloaded (no free slot, or GPU memory >= 85%) gets
  another replica, up to ``max_replicas`` and within a budget reserve;
* an unavailable tier falls back to a pricier one only within max_upgrade_ratio,
  otherwise to cheaper GPU tiers;
* a CUDA out-of-memory failure is re-run once: alone on the same tier when the
  VM was shared, otherwise on the next larger tier.

The CLI records every executed cell and its output in its local history, so the
executor token is uploaded as a file and never appears in code or output.

    python -m autoresearch.colab_runtime up --gpu T4
    python -m autoresearch.colab_runtime status
    python -m autoresearch.colab_runtime down --session <name>
"""

import argparse
from collections import namedtuple
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

import requests

from ai_scientist.treesearch.remote_interpreter import (
    RemoteExecutorUnavailable,
    _config_path,
)

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / ".state" / "colab"
GPUS = ("T4", "L4", "G4", "A100", "H100")
TIER_NAMES = ("cpu",) + GPUS
# nvidia-smi names checked after allocation; the CLI silently maps unknown
# --gpu values to A100, and a wrong accelerator is a cost error.
GPU_NAME_MARKERS = {"T4": "T4", "L4": "L4", "A100": "A100", "H100": "H100"}
DEFAULT_PACKAGES = [
    "datasets", "humanize", "timm", "albumentations", "bayesian-optimization",
    "torch_geometric", "statsmodels", "seaborn",
]
# Rates are compute units per hour: cpu measured on this account; GPU values
# from a third-party measurement (mccormickml.com, March 2026), not Google
# pricing. They order tiers and inform the model; budgets use `colab usage`.
DEFAULT_TIERS = {
    "cpu": {"rate": 0.08, "memory_gb": None,
            "use": "no GPU: numpy/pandas/scikit-learn, tiny PyTorch models, data preparation"},
    "T4": {"rate": 1.19, "memory_gb": 15,
           "use": "small CNNs/MLPs/RNNs, transformers up to ~100M parameters, fp16"},
    "L4": {"rate": 1.71, "memory_gb": 22,
           "use": "mid-size models or batches that exceed a T4, bf16, roughly 2x T4 speed"},
    "A100": {"rate": 5.40, "memory_gb": 40, "idle_stop_minutes": 10,
             "use": "large models (hundreds of millions of parameters) or long training "
                    "that cannot fit the time limit on L4"},
}
DEFAULTS = {
    "auto_provision": False,
    "selection": "llm",
    "gpu": "T4",
    "tiers": DEFAULT_TIERS,
    "max_sessions": 2,
    "max_replicas": 2,
    "max_upgrade_ratio": 1.5,
    "escalate_on_oom": True,
    "high_mem": False,
    "idle_stop_minutes": 30,
    "max_compute_units": 30,
    "min_balance": 5,
    "max_concurrent": None,
    "packages": DEFAULT_PACKAGES,
}
PORT = 8765
CLI_TIMEOUT = 300
BOOTSTRAP_TIMEOUT = 600
HEALTH_SECONDS = 180  # quick tunnels can take minutes to accept traffic
WATCH_SECONDS = 30
USAGE_SECONDS = 300
ROOM_WAIT_SECONDS = 1800
UNAVAILABLE_SECONDS = 1800  # entitlement/quota refusals
TRANSIENT_SECONDS = 180  # capacity refusals (HTTP 5xx), which usually clear quickly
OVERLOAD_MEMORY = 0.85  # a VM whose GPU memory is this full cannot take another job
SCALE_OUT_RESERVE_HOURS = 0.5  # a new replica needs this much budget at its tier's rate
_SECRET = re.compile(r"(token=)[^\s&'\"]+", re.I)
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_BOX = re.compile(r"[\u2500-\u257f]+")
_TRANSIENT = re.compile(r"Service Unavailable|\b50[234]\b|capacity|try again|Unavailable", re.I)
_COMPUTE = re.compile(r"^[ \t]*#[ \t]*COMPUTE:[ \t]*([A-Za-z0-9]+)[ \t]*(?:[#:-][ \t]*(.*))?$",
                      re.M | re.I)
_OOM = re.compile(r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED")

STATE_ENV = "AUTORESEARCH_COLAB_STATE"
COMPUTE_LOG_ENV = "AUTORESEARCH_COMPUTE_LOG"


# shared: the VM already had running/queued jobs when this one was placed.
Placement = namedtuple("Placement", "tier key path notes shared")


class TierUnavailable(RemoteExecutorUnavailable):
    """One tier could not be allocated; cheaper tiers may still work."""


class NoTierAvailable(RemoteExecutorUnavailable):
    """No candidate tier could be allocated right now; retrying later may work."""


def _tier_of(key):
    """Replica keys are the tier name, then T4.2, T4.3, ... for scale-out VMs."""
    return key.split(".")[0]


def _replica_index(key):
    return int(key.split(".")[1]) if "." in key else 1


def gpu_memory_fraction(health):
    """Used/total GPU memory from the executor's nvidia-smi line, if reported."""
    match = re.search(r"(\d+)\s*MiB\s*,\s*(\d+)\s*MiB", str(health.get("gpu", "")))
    if not match or not int(match.group(2)):
        return None
    return int(match.group(1)) / int(match.group(2))


def overloaded(health):
    """No free job slot, or GPU memory too full for another concurrent job."""
    busy = health.get("running", 0) + health.get("queued", 0)
    if busy >= max(1, health.get("max_concurrent", 1)):
        return True
    fraction = gpu_memory_fraction(health)
    return bool(health.get("running") and fraction is not None and fraction >= OVERLOAD_MEMORY)


def _load(health):
    return (health.get("running", 0) + health.get("queued", 0)) / max(1, health.get("max_concurrent", 1))


def _tier_name(value):
    if not isinstance(value, str):
        return None
    return next((t for t in TIER_NAMES if t.lower() == value.strip().lower()), None)


def validate_colab_settings(settings):
    """Validate exec.colab without touching the CLI (safe for dry runs)."""
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError("exec.colab must be a mapping")
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown exec.colab settings: {', '.join(sorted(unknown))}")
    merged = {**DEFAULTS, **settings}
    for key in ("auto_provision", "high_mem", "escalate_on_oom"):
        if not isinstance(merged[key], bool):
            raise ValueError(f"exec.colab.{key} must be a boolean")
    if merged["selection"] not in ("llm", "fixed"):
        raise ValueError("exec.colab.selection must be llm or fixed")
    gpus = merged["gpu"] if isinstance(merged["gpu"], list) else [merged["gpu"]]
    if not gpus or any(g not in TIER_NAMES for g in gpus):
        raise ValueError(
            f"exec.colab.gpu must be one of {', '.join(TIER_NAMES)}, or a list of them"
        )
    merged["gpu"] = gpus

    def number(label, value, nullable=False, positive=False):
        if value is None and nullable:
            return
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or (positive and value == 0)):
            raise ValueError(f"exec.colab.{label} must be a {'positive' if positive else 'nonnegative'} "
                             f"number{' or null' if nullable else ''}")

    tiers = merged["tiers"]
    if not isinstance(tiers, dict) or not tiers:
        raise ValueError("exec.colab.tiers must be a nonempty mapping")
    ordered = {}
    for name, spec in tiers.items():
        if name not in TIER_NAMES:
            raise ValueError(f"exec.colab.tiers: unknown tier {name!r} (use {', '.join(TIER_NAMES)})")
        if not isinstance(spec, dict) or set(spec) - {"rate", "memory_gb", "use", "idle_stop_minutes"}:
            raise ValueError(f"exec.colab.tiers.{name} takes rate, memory_gb, use, idle_stop_minutes")
        number(f"tiers.{name}.rate", spec.get("rate"), positive=True)
        number(f"tiers.{name}.memory_gb", spec.get("memory_gb"), nullable=True)
        number(f"tiers.{name}.idle_stop_minutes", spec.get("idle_stop_minutes"), nullable=True)
        if not isinstance(spec.get("use", ""), str):
            raise ValueError(f"exec.colab.tiers.{name}.use must be a string")
        ordered[name] = dict(spec)
    merged["tiers"] = dict(sorted(ordered.items(), key=lambda item: item[1]["rate"]))
    if merged["selection"] == "llm" and gpus[0] not in merged["tiers"]:
        raise ValueError("exec.colab.gpu (the default tier) must be listed in exec.colab.tiers")
    for key, nullable in (("idle_stop_minutes", True), ("max_compute_units", True),
                          ("min_balance", False)):
        number(key, merged[key], nullable=nullable)
    number("max_upgrade_ratio", merged["max_upgrade_ratio"], positive=True)
    if merged["max_upgrade_ratio"] < 1:
        raise ValueError("exec.colab.max_upgrade_ratio must be at least 1 (1 = never pay more)")
    for key in ("max_concurrent", "max_sessions", "max_replicas"):
        value = merged[key]
        if (value is not None or key == "max_sessions") and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"exec.colab.{key} must be a positive integer"
                             + (" or null" if key == "max_concurrent" else ""))
    packages = merged["packages"]
    if not isinstance(packages, list) or not all(
        isinstance(p, str) and re.fullmatch(r"[A-Za-z0-9_.\-\[\],<>=!~]+", p) for p in packages
    ):
        raise ValueError("exec.colab.packages must be a list of pip requirement strings")
    return merged


# ---------------------------------------------------------------- tier policy


def parse_compute(code):
    """Return (tier or None, reason) from the first ``# COMPUTE:`` line."""
    match = _COMPUTE.search(code or "")
    if not match:
        return None, ""
    return _tier_name(match.group(1)), (match.group(2) or "").strip()[:200]


def is_oom(result):
    text = "".join(result.term_out or []) if result.term_out else ""
    return result.exc_type in ("OutOfMemoryError",) or bool(_OOM.search(text))


def candidates(settings, requested):
    """Tiers to try, in order, when the requested one cannot be allocated.

    First pricier tiers within max_upgrade_ratio of the requested rate (memory
    safe, bounded cost; e.g. T4 -> L4 when T4 capacity is exhausted), then
    cheaper GPU tiers. Never cpu for a GPU request.
    """
    if settings["selection"] == "fixed":
        return list(settings["gpu"])
    tiers = settings["tiers"]
    names = list(tiers)
    if requested not in tiers:
        requested = settings["gpu"][0]
    if requested == "cpu":
        return ["cpu"]
    index, limit = names.index(requested), tiers[requested]["rate"] * settings["max_upgrade_ratio"]
    upgrades = [t for t in names[index + 1:] if tiers[t]["rate"] <= limit]
    cheaper = [t for t in reversed(names[:index]) if t != "cpu"]
    return [requested] + upgrades + cheaper


def next_tier(settings, tier):
    """The next larger GPU tier for an out-of-memory re-run, if any."""
    if settings["selection"] == "fixed" or not settings["escalate_on_oom"]:
        return None
    tiers = list(settings["tiers"])
    if tier not in tiers or tier == "cpu":
        return None
    larger = tiers[tiers.index(tier) + 1:]
    return larger[0] if larger else None


def compute_instructions():
    """Prompt section for LLM tier selection, or None when it does not apply."""
    path = os.environ.get(STATE_ENV)
    if not path or not Path(path).is_file():
        return None
    state = json.loads(Path(path).read_text())
    settings = state["settings"]
    if settings["selection"] != "llm":
        return None
    lines = [
        "This script runs on a Colab runtime chosen by you. Declare it on the FIRST line of the code:",
        "  # COMPUTE: <tier>  # <one-line reason: model size, data size, expected run time>",
        "Choose the CHEAPEST tier that fits the model/batch in memory and finishes within the time limit.",
        "Most baselines, tuning sweeps and ablations of small models belong on cpu or T4; "
        "request A100 or larger only when the workload genuinely needs it.",
        "If the previous run ran out of GPU memory or hit the time limit, a larger tier is "
        "reasonable; if it finished quickly with memory to spare, choose a cheaper one. "
        "Do not change the experiment's scientific content just to fit a tier.",
        "Available tiers (cheapest first; rate in compute units per hour):",
    ]
    for name, spec in settings["tiers"].items():
        measured = state.get("sessions", {}).get(name, {}).get("measured_rate")
        rate = measured if measured else spec["rate"]
        memory = f"{spec['memory_gb']} GB GPU memory" if spec.get("memory_gb") else "no GPU"
        lines.append(f"  - {name}: {memory}, ~{rate:g} units/hour. {spec.get('use', '')}".rstrip())
    cap = settings["max_compute_units"]
    if cap is not None:
        remaining = max(0.0, cap - state.get("spent_units", 0.0))
        lines.append(f"Remaining run budget: {remaining:.1f} of {cap:g} compute units.")
    if settings["max_replicas"] > 1:
        lines.append(
            "If the chosen tier's VM is busy with other jobs, the harness may start another VM of "
            f"the same tier (up to {settings['max_replicas']}); do not pick a larger tier just "
            "to avoid waiting.")
    lines.append(f"Without a COMPUTE line the script runs on {settings['gpu'][0]}.")
    return {"Compute selection": lines}


def log_compute(record):
    path = os.environ.get(COMPUTE_LOG_ENV)
    if path:
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")


def _cli_error(text):
    """The informative line of a CLI failure (it prints Rich tracebacks)."""
    lines = [line for line in redact_lines(text) if line]
    wanted = [l for l in lines if re.search(r"error|unavailable|denied|quota|not found", l, re.I)]
    return (wanted or lines or ["no output"])[-1][-300:]


def redact_lines(text):
    return [redact(line) for line in _ANSI.sub("", text or "").splitlines()]


def redact(text):
    text = _BOX.sub(" ", _ANSI.sub("", text or ""))
    return re.sub(r"\s+", " ", _SECRET.sub(r"\1<redacted>", text)).strip()


def _write_private(path, value):
    """Atomic 0600 JSON write; readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ColabCLI:
    def __init__(self, binary=None):
        self.binary = binary or os.environ.get("AUTORESEARCH_COLAB_CLI", "colab")

    def run(self, *args, timeout=CLI_TIMEOUT, check=True, stdin_text=None):
        if shutil.which(self.binary) is None:
            raise RemoteExecutorUnavailable(
                f"Colab CLI {self.binary!r} not found; install it with "
                "`uv tool install google-colab-cli`"
            )
        env = dict(os.environ, NO_COLOR="1", TERM="dumb", COLUMNS="200")
        try:
            done = subprocess.run(
                [self.binary, *args], capture_output=True, text=True, timeout=timeout,
                env=env, **({"input": stdin_text} if stdin_text is not None
                            else {"stdin": subprocess.DEVNULL}),
            )
        except subprocess.TimeoutExpired as exc:
            raise RemoteExecutorUnavailable(
                f"colab {args[0]} did not finish within {timeout}s"
            ) from exc
        out = _ANSI.sub("", done.stdout)
        err = _ANSI.sub("", done.stderr)
        if check and done.returncode != 0:
            raise RemoteExecutorUnavailable(
                f"colab {args[0]} failed ({done.returncode}): {redact(err or out)[-800:]}"
            )
        return done.returncode, out, err

    def usage(self):
        """Return (balance, rate per hour) from `colab usage`."""
        _, out, err = self.run("usage", timeout=60)
        balance = re.search(r"balance:\s*([\d.]+)", out + err)
        rate = re.search(r"rate:\s*([\d.]+)", out + err)
        if not balance:
            raise RemoteExecutorUnavailable(
                f"Could not read the compute-unit balance: {redact(out + err)[-400:]}"
            )
        return float(balance.group(1)), float(rate.group(1)) if rate else 0.0

    def session_exists(self, name):
        _, out, err = self.run("status", "-s", name, timeout=60, check=False)
        text = out + err
        return f"[{name}]" in text and "not found" not in text.lower()

    def stop(self, name):
        self.run("stop", "-s", name, timeout=120, check=False)


def bootstrap_script(max_concurrent, packages):
    """Kernel code that (re)starts the server if needed and always a fresh tunnel.

    A live server keeps running jobs; only the tunnel is replaced. The token is
    read from an uploaded file so it never enters the CLI history.
    """
    return f'''
import json, os, re, signal, subprocess, sys, time, urllib.request
PORT = {PORT}
BASE = "/content/aisci"
os.makedirs(BASE, exist_ok=True)
for name in ("colab_server.py", "workspace.py"):
    if os.path.exists("/content/" + name):
        os.replace("/content/" + name, os.path.join(BASE, name))
token_file = "/content/.aisci_token"
token = open(token_file).read().strip()
os.chmod(token_file, 0o600)

def alive(pidfile):
    try:
        pid = int(open(pidfile).read())
        os.kill(pid, 0)
        return pid
    except Exception:
        return None

def healthy():
    req = urllib.request.Request(f"http://127.0.0.1:{{PORT}}/health",
                                 headers={{"Authorization": "Bearer " + token}})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.load(response).get("protocol_version") == 2
    except Exception:
        return False

gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                     shell=True, capture_output=True, text=True)
print("AISCI_GPU=" + (gpu.stdout.strip().splitlines() or ["none"])[0], flush=True)

server_pid = os.path.join(BASE, "server.pid")
if not (alive(server_pid) and healthy()):
    if alive(server_pid):
        os.killpg(alive(server_pid), signal.SIGTERM)
        time.sleep(3)
    packages = {packages!r}
    if packages and not os.path.exists(os.path.join(BASE, ".packages")):
        subprocess.run([sys.executable, "-m", "pip", "-q", "install", *packages], check=True)
        open(os.path.join(BASE, ".packages"), "w").write("\\n".join(packages))
    env = dict(os.environ, AISCI_TOKEN=token, AISCI_MAX_CONCURRENT="{max_concurrent}")
    server = subprocess.Popen(
        [sys.executable, os.path.join(BASE, "colab_server.py"), "--port", str(PORT)],
        stdout=open(os.path.join(BASE, "server.log"), "a"), stderr=subprocess.STDOUT,
        env=env, start_new_session=True, cwd=BASE,
    )
    open(server_pid, "w").write(str(server.pid))
    for _ in range(30):
        time.sleep(1)
        if healthy():
            break
    else:
        print(open(os.path.join(BASE, "server.log")).read()[-2000:])
        raise RuntimeError("executor server did not become healthy")
    print("AISCI_SERVER=started", flush=True)
else:
    print("AISCI_SERVER=reused", flush=True)
os.remove(token_file)

tunnel_pid = os.path.join(BASE, "tunnel.pid")
if alive(tunnel_pid):
    os.killpg(alive(tunnel_pid), signal.SIGTERM)
binary = os.path.join(BASE, "cloudflared")
if not os.path.exists(binary):
    urllib.request.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        binary,
    )
    os.chmod(binary, 0o755)
log = os.path.join(BASE, "tunnel.log")
tunnel = subprocess.Popen(
    [binary, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{{PORT}}"],
    stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True,
)
open(tunnel_pid, "w").write(str(tunnel.pid))
url = None
for _ in range(120):
    time.sleep(1)
    text = open(log).read()
    match = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", text)
    url = url or (match and match.group(0))
    # The URL is printed before the edge accepts traffic; wait for registration.
    if url and "Registered tunnel connection" in text:
        break
if url is None:
    print(open(log).read()[-2000:])
    raise RuntimeError("cloudflared did not print a tunnel URL")
print("AISCI_TUNNEL=" + ("registered" if "Registered tunnel connection" in text else "pending"))
print("AISCI_URL=" + url, flush=True)
'''




class ColabPool:
    """The sessions of one run (one per tier), described by a state file."""

    def __init__(self, state_path, cli=None):
        self.state_path = Path(state_path)
        self.cli = cli or ColabCLI()

    # ------------------------------------------------------------ state

    @classmethod
    def create(cls, settings, *, executor_path=None, owner_pid=None, max_concurrent=1,
               session=None, cli=None):
        settings = validate_colab_settings(settings)
        name = session or f"ar-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,56}", name):
            raise ValueError("Colab session names use letters, digits, - and _ only")
        state_path = STATE_DIR / f"{name}.json"
        _write_private(state_path, {
            "pool": name,
            "settings": settings,
            "max_concurrent": settings["max_concurrent"] or max_concurrent,
            "executor_override": str(Path(executor_path).resolve()) if executor_path else None,
            "owner_pid": owner_pid,
            "released": False,
            "stop_reason": None,
            "spent_units": 0.0,
            "created_at": time.time(),
            "sessions": {},
        })
        return cls(state_path, cli)

    def load(self):
        return json.loads(self.state_path.read_text())

    def save(self, state):
        _write_private(self.state_path, state)

    @contextlib.contextmanager
    def locked(self, name="state", blocking=True):
        """flock per name; not re-entrant, so never nest the same name."""
        path = self.state_path.with_name(f"{self.state_path.stem}.{name}.lock")
        with open(path, "a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    @contextlib.contextmanager
    def mutate(self):
        with self.locked():
            state = self.load()
            yield state
            self.save(state)

    def _entry(self, state, key):
        entry = state["sessions"].setdefault(key, {})
        if not entry:
            tier, index = _tier_of(key), _replica_index(key)
            fixed = state["settings"]["selection"] == "fixed"
            name = state["pool"] if fixed else f"{state['pool']}-{tier.lower()}"
            name += f"-{index}" if index > 1 else ""
            entry.update(
                tier=tier,
                session=name,
                executor_path=state["executor_override"]
                or str(self.state_path.parent / f"{name}.executor.json"),
                up=False, gpu=None, provisions=0, stop_reason=None, unavailable_until=0,
            )
        return entry

    # ------------------------------------------------------------ health

    def health(self, entry, timeout=10):
        try:
            executor = json.loads(Path(entry["executor_path"]).read_text())
            response = requests.get(
                executor["url"].rstrip("/") + "/health",
                headers={"Authorization": f"Bearer {executor['token']}"},
                timeout=timeout, allow_redirects=False,
            )
            if response.status_code == 200:
                return response.json()
        except (OSError, ValueError, KeyError, requests.RequestException):
            pass
        return None

    def _busy(self, entry):
        health = self.health(entry)
        return bool(health and health.get("running", 0) + health.get("queued", 0))

    # ------------------------------------------------------------ lifecycle

    def preflight(self):
        """Fail fast on missing CLI/auth or a low balance; allocates nothing."""
        return self._check_balance(self.load()["settings"])[0]

    def _check_balance(self, settings):
        balance, rate = self.cli.usage()
        if balance < settings["min_balance"]:
            raise RemoteExecutorUnavailable(
                f"Colab balance {balance:.2f} is below exec.colab.min_balance "
                f"{settings['min_balance']}; not provisioning"
            )
        return balance, rate

    @staticmethod
    def _check_pool(state):
        if state["released"]:
            raise RemoteExecutorUnavailable(
                f"Colab pool {state['pool']} was released ({state['stop_reason']})"
            )
        if state["stop_reason"] == "budget":
            raise RemoteExecutorUnavailable(
                f"Colab budget of {state['settings']['max_compute_units']} compute units "
                f"is spent (pool {state['pool']})"
            )

    def acquire(self, requested=None, exclusive=False):
        """Place a job on a tier (see candidates) and replica; returns a Placement.

        exclusive: only an idle replica will do (re-running an OOM caused by
        sharing a GPU); a new replica is started when none is idle.
        """
        state = self.load()
        self._check_pool(state)
        notes = []
        for tier in candidates(state["settings"], requested):
            try:
                with self.locked(f"place-{tier}"):  # one scale-out decision at a time
                    key, path, shared = self._place(tier, notes, exclusive)
                return Placement(tier, key, path, notes, shared)
            except TierUnavailable as exc:
                notes.append(str(exc))
        raise NoTierAvailable("No Colab tier could be allocated: " + "; ".join(notes))

    def _place(self, tier, notes, exclusive=False):
        """Least-loaded healthy replica with room; else start another one if allowed."""
        state = self.load()
        keys = sorted((k for k, e in state["sessions"].items() if e.get("tier", k) == tier),
                      key=_replica_index) or [tier]
        up = {k: self.health(state["sessions"][k]) for k in keys
              if state["sessions"].get(k, {}).get("up")}
        if not up:
            return keys[0], self.ensure(keys[0]), False
        free = [k for k, h in up.items() if h and (_load(h) == 0 if exclusive else not overloaded(h))]
        if free:
            key = min(free, key=lambda k: _load(up[k]))
            return key, self.ensure(key), _load(up[key]) > 0
        unreachable = [k for k, h in up.items() if h is None]
        if unreachable:  # reconnect before paying for another VM
            return unreachable[0], self.ensure(unreachable[0]), False
        blocker = self._scale_blocker(state, tier, len(up))
        if blocker is None:
            index = next(i for i in range(2, 1000) if f"{tier}.{i}" not in up)
            key = f"{tier}.{index}"
            try:
                path = self.ensure(key, wait_for_room=False)
                notes.append(f"every {tier} VM was {'in use' if exclusive else 'busy'}; "
                             f"started {tier} replica {index}")
                return key, path, False
            except TierUnavailable as exc:
                blocker = str(exc)
        key = min(up, key=lambda k: _load(up[k]))
        notes.append(f"every {tier} VM was busy; queued on {key} ({blocker})")
        return key, self.ensure(key), True

    def _scale_blocker(self, state, tier, replicas_up):
        """Why another replica of this tier may not start, or None."""
        settings = state["settings"]
        if state["executor_override"]:
            return "a standalone executor file serves one VM"
        if replicas_up >= settings["max_replicas"]:
            return f"max_replicas={settings['max_replicas']}"
        others = [e for e in state["sessions"].values()
                  if e.get("up") and e.get("tier") != tier]
        if replicas_up + len(others) >= settings["max_sessions"] and all(
                self._busy(e) for e in others):
            return f"max_sessions={settings['max_sessions']} VMs are all busy"
        cap = settings["max_compute_units"]
        rate = settings["tiers"].get(tier, {}).get("rate", 0)
        if cap is not None and cap - state.get("spent_units", 0.0) < rate * SCALE_OUT_RESERVE_HOURS:
            return "remaining budget below the scale-out reserve"
        return None

    def ensure(self, tier, wait_for_room=True):
        """Make one tier's executor reachable; returns its executor file path."""
        with self.locked(f"tier-{tier}"):
            with self.mutate() as state:
                self._check_pool(state)
                entry = dict(self._entry(state, tier))
            if entry["up"] and self.health(entry):
                return entry["executor_path"]
            if entry.get("unavailable_until", 0) > time.time():
                raise TierUnavailable(f"{tier} was unavailable recently ({entry['stop_reason']})")
            if self.cli.session_exists(entry["session"]):
                self._bootstrap(tier, reuse_token=True)
            else:
                self._make_room(tier, wait=wait_for_room)
                self._provision(tier)
                self._bootstrap(tier, reuse_token=False)
            with self.mutate() as state:
                self._entry(state, tier).update(up=True, stop_reason=None, last_busy=time.time())
            return entry["executor_path"]

    def _make_room(self, tier, wait=True):
        """Enforce max_sessions by stopping the priciest idle session, else wait
        (or, for an optional scale-out, give up at once)."""
        deadline = time.time() + ROOM_WAIT_SECONDS
        while True:
            state = self.load()
            others = {t: e for t, e in state["sessions"].items() if t != tier and e.get("up")}
            if len(others) < state["settings"]["max_sessions"]:
                return
            rates = state["settings"]["tiers"]
            for victim in sorted(others, key=lambda t: -rates.get(_tier_of(t), {}).get("rate", 0)):
                if not self._busy(others[victim]) and self.stop_tier(victim, "make-room",
                                                                    blocking=False):
                    break
            else:
                if not wait:
                    raise TierUnavailable(
                        f"max_sessions={state['settings']['max_sessions']} VMs are all busy")
                if time.time() >= deadline:
                    raise RemoteExecutorUnavailable(
                        f"exec.colab.max_sessions={state['settings']['max_sessions']} sessions "
                        f"stayed busy for {ROOM_WAIT_SECONDS // 60} min; cannot start {tier}"
                    )
                time.sleep(15)

    def _provision(self, key):
        tier = _tier_of(key)
        state = self.load()
        settings, entry = state["settings"], self._entry(state, key)
        balance, before = self._check_balance(settings)
        args = ["new", "-s", entry["session"]]
        if tier != "cpu":
            args += ["--gpu", tier]
        if settings["high_mem"]:
            args.append("--high-mem")
        for attempt in range(2):
            code, out, err = self.cli.run(*args, check=False)
            if code == 0 and "READY" in (out + err).upper():
                break
            self.cli.stop(entry["session"])
            transient = bool(_TRANSIENT.search(out + err))
            if transient and attempt == 0:
                time.sleep(20)
                continue
            with self.mutate() as fresh:
                self._entry(fresh, key).update(
                    stop_reason="capacity" if transient else "allocation-failed",
                    unavailable_until=time.time()
                    + (TRANSIENT_SECONDS if transient else UNAVAILABLE_SECONDS))
            raise TierUnavailable(f"{tier}: {_cli_error(err or out)}")
        try:
            after = self.cli.usage()[1]
        except RemoteExecutorUnavailable:
            after = None
        with self.mutate() as fresh:
            fresh.setdefault("start_balance", balance)
            current = self._entry(fresh, key)
            current.update(provisions=current["provisions"] + 1, provisioned_at=time.time())
            if after is not None and after > before:
                current["measured_rate"] = round(after - before, 3)

    def _bootstrap(self, tier, reuse_token):
        """Start (or reuse) the server and a tunnel; one tunnel restart if unreachable."""
        for attempt in range(2):
            url = self._bootstrap_once(tier, reuse_token or attempt > 0)
            entry = self._entry(self.load(), tier)
            deadline = time.time() + HEALTH_SECONDS
            while time.time() < deadline:
                if self.health(entry):
                    return
                time.sleep(5)
            note = f"{url} unreachable after {HEALTH_SECONDS}s"
        _, out, _ = self.cli.run(
            "exec", "-s", entry["session"], "--timeout", "30", timeout=90, check=False,
            stdin_text="print(open('/content/aisci/tunnel.log').read()[-1500:])")
        self._halt(tier, "tunnel-unreachable")
        raise RemoteExecutorUnavailable(
            f"Executor on {entry['session']} started but {note}; VM stopped. "
            f"Tunnel log: {redact(out)[-800:]}")

    def _bootstrap_once(self, key, reuse_token):
        tier = _tier_of(key)
        state = self.load()
        entry = self._entry(state, key)
        name, executor_path = entry["session"], Path(entry["executor_path"])
        token = None
        if reuse_token:
            with contextlib.suppress(OSError, ValueError, KeyError):
                token = json.loads(executor_path.read_text())["token"]
        token = token or secrets.token_urlsafe(32)
        with tempfile.TemporaryDirectory() as tmp:
            token_file = Path(tmp) / "token"
            token_file.write_text(token)
            token_file.chmod(0o600)
            script = Path(tmp) / "bootstrap.py"
            script.write_text(bootstrap_script(state["max_concurrent"],
                                               state["settings"]["packages"]))
            remote = ROOT / "ai_scientist" / "remote"
            for local, target in ((remote / "colab_server.py", "/content/colab_server.py"),
                                  (remote / "workspace.py", "/content/workspace.py"),
                                  (token_file, "/content/.aisci_token")):
                self.cli.run("upload", "-s", name, str(local), target)
            _, out, err = self.cli.run("exec", "-s", name, "-f", str(script),
                                       "--timeout", str(BOOTSTRAP_TIMEOUT),
                                       timeout=BOOTSTRAP_TIMEOUT + 60)
        fields = dict(re.findall(r"^AISCI_(\w+)=(.*)$", out, re.M))
        if "URL" not in fields:
            # `colab exec` exits 0 on remote exceptions; the marker is authoritative.
            self._halt(key, "bootstrap-failed")
            raise RemoteExecutorUnavailable(
                f"Executor bootstrap failed on {name}: {redact(err or out).strip()[-1200:]}"
            )
        gpu = fields.get("GPU", "none").strip()
        marker = GPU_NAME_MARKERS.get(tier)
        if (tier != "cpu" and gpu == "none") or (marker and marker not in gpu):
            self._halt(key, "wrong-gpu", unavailable=True)
            raise TierUnavailable(f"requested {tier} but Colab allocated {gpu!r}; session stopped")
        _write_private(executor_path, {"url": fields["URL"].strip(), "token": token})
        with self.mutate() as fresh:
            self._entry(fresh, key)["gpu"] = gpu
        return fields["URL"].strip()

    def _halt(self, tier, reason, unavailable=False):
        """Stop a tier's VM and drop its dead credentials, so the next job
        reaches ensure() at once instead of retrying a stale URL.
        Callers hold the tier lock (or use stop_tier)."""
        state = self.load()
        entry = self._entry(state, tier)
        self.cli.stop(entry["session"])
        Path(entry["executor_path"]).unlink(missing_ok=True)
        with self.mutate() as fresh:
            current = self._entry(fresh, tier)
            current.update(up=False, stop_reason=reason)
            if unavailable:
                current["unavailable_until"] = time.time() + UNAVAILABLE_SECONDS

    def stop_tier(self, tier, reason, blocking=True):
        with self.locked(f"tier-{tier}", blocking=blocking) as acquired:
            if not acquired:
                return False
            self._halt(tier, reason)
            return True

    def stop(self, reason, release=False):
        for tier in list(self.load()["sessions"]):
            self.stop_tier(tier, reason)
        with self.mutate() as state:
            state["stop_reason"] = reason
            state["released"] = state["released"] or release

    # ------------------------------------------------------------ watchdog

    def watch(self, poll=WATCH_SECONDS, usage_every=USAGE_SECONDS, once=False):
        """Stop VMs on idle (per tier), budget, release or controller exit.

        An unreachable executor counts as idle: a controller that still needs
        it re-establishes the tunnel through ensure() long before the limit.
        """
        last_usage = None
        while True:
            state = self.load()
            settings, owner = state["settings"], state.get("owner_pid")
            if state["released"]:
                return "released"
            if owner and not _pid_alive(owner):
                self.stop("owner-exited", release=True)
                return "owner-exited"
            now = time.time()
            running = [t for t, e in state["sessions"].items() if e.get("up")]
            for tier in running:
                entry = state["sessions"][tier]
                last_busy = entry.get("last_busy") or entry.get("provisioned_at") or now
                if self._busy(entry):
                    last_busy = now
                    with self.mutate() as fresh:
                        self._entry(fresh, tier)["last_busy"] = now
                idle = settings["tiers"].get(_tier_of(tier), {}).get("idle_stop_minutes")
                idle = settings["idle_stop_minutes"] if idle is None else idle
                if idle is not None and now - last_busy >= idle * 60:
                    self.stop_tier(tier, "idle", blocking=False)
            if running and (last_usage is None or now - last_usage >= usage_every):
                self._account(now - last_usage if last_usage else 0)
                last_usage = now
            if not owner and not any(e.get("up") for e in self.load()["sessions"].values()):
                return "idle" if running else (state["stop_reason"] or "stopped")
            if once:
                return None
            time.sleep(poll)

    def _account(self, interval):
        """Budget check from balance delta, or rate x interval when the balance lags.

        The balance is account-wide, so other Colab use in the same account
        counts against this run's cap.
        """
        try:
            balance, rate = self.cli.usage()
        except RemoteExecutorUnavailable:
            return
        with self.mutate() as state:
            cap = state["settings"]["max_compute_units"]
            estimated = state.get("spent_units", 0.0) + rate * interval / 3600
            delta = state.get("start_balance", balance) - balance
            state.update(spent_units=round(max(estimated, delta), 4), rate_per_hour=rate,
                         balance=balance)
            spent = state["spent_units"]
        if cap is not None and spent >= cap:
            self.stop("budget")

    def summary(self):
        """Cost/provenance record for the run directory; contains no token."""
        state = self.load()
        return {
            "pool": state["pool"],
            "selection": state["settings"]["selection"],
            "stop_reason": state["stop_reason"],
            "spent_units": state.get("spent_units"),
            "start_balance": state.get("start_balance"),
            "balance": state.get("balance"),
            "sessions": {
                replica: {key: entry.get(key) for key in (
                    "tier", "session", "gpu", "provisions", "stop_reason", "measured_rate")}
                for replica, entry in state["sessions"].items()
            },
        }


# ---------------------------------------------------------------- launcher API


def spawn_watchdog(pool):
    """Detached so it survives a killed controller and can still stop the VMs."""
    with open(pool.state_path.with_suffix(".log"), "a") as log:
        return subprocess.Popen(
            [sys.executable, "-m", "autoresearch.colab_runtime", "watchdog",
             "--state", str(pool.state_path)],
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, cwd=ROOT,
        )


def start_managed(settings, max_concurrent, owner_pid=None):
    """Create run state, check CLI/balance and start the watchdog; no VM yet."""
    pool = ColabPool.create(settings, owner_pid=owner_pid or os.getpid(),
                            max_concurrent=max_concurrent)
    pool.preflight()
    spawn_watchdog(pool)
    return pool


def managed_pool():
    path = os.environ.get(STATE_ENV)
    return ColabPool(path) if path else None


def release_from_env(reason):
    pool = managed_pool()
    if pool and pool.state_path.exists() and not pool.load()["released"]:
        pool.stop(reason, release=True)


def tier_rate(pool, tier):
    state = pool.load()
    measured = state["sessions"].get(tier, {}).get("measured_rate")
    return measured or state["settings"]["tiers"].get(tier, {}).get("rate")


# ---------------------------------------------------------------- command line


def _pools():
    if not STATE_DIR.is_dir():
        return
    for path in sorted(STATE_DIR.glob("*.json")):
        if path.name.endswith(".executor.json"):
            continue
        with contextlib.suppress(OSError, ValueError, KeyError):
            state = json.loads(path.read_text())
            if "sessions" in state:
                yield ColabPool(path), state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    up = sub.add_parser("up", help="provision one runtime and write the executor file")
    up.add_argument("--gpu", action="append", help="T4, L4, G4, A100, H100 or cpu; repeat for fallbacks")
    up.add_argument("--session")
    up.add_argument("--high-mem", action="store_true")
    up.add_argument("--max-concurrent", type=int, default=1)
    up.add_argument("--idle-stop-minutes", type=float, default=DEFAULTS["idle_stop_minutes"])
    up.add_argument("--max-compute-units", type=float, default=DEFAULTS["max_compute_units"])
    up.add_argument("--executor-config", type=Path, default=None,
                    help=f"default: {_config_path()}")
    up.add_argument("--no-packages", action="store_true")
    up.add_argument("--no-watchdog", action="store_true",
                    help="keep the VM until `down` (idle and budget stops disabled)")
    down = sub.add_parser("down", help="stop managed sessions")
    down.add_argument("--session", help="pool or session name; default: every pool that is up")
    sub.add_parser("status", help="balance, rate and managed sessions")
    watchdog = sub.add_parser("watchdog", help=argparse.SUPPRESS)
    watchdog.add_argument("--state", required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "watchdog":
            reason = ColabPool(args.state).watch()
            print(f"{time.strftime('%H:%M:%S')} watchdog exit: {reason}", flush=True)
            return 0
        if args.command == "up":
            settings = {
                "selection": "fixed", "gpu": args.gpu or [DEFAULTS["gpu"]],
                "high_mem": args.high_mem, "idle_stop_minutes": args.idle_stop_minutes,
                "max_compute_units": args.max_compute_units,
                "packages": [] if args.no_packages else DEFAULT_PACKAGES,
            }
            pool = ColabPool.create(
                settings, executor_path=args.executor_config or _config_path(),
                max_concurrent=args.max_concurrent, session=args.session)
            pool.preflight()
            tier, key, path, notes, _ = pool.acquire()
            entry = pool.load()["sessions"][key]
            for note in notes:
                print(f"Skipped: {note}")
            print(f"Session {entry['session']} on {tier} ({entry['gpu']})")
            print(f"Executor file: {path} (token not shown)")
            if not args.no_watchdog:
                spawn_watchdog(pool)
                print(f"Watchdog: idle stop {args.idle_stop_minutes} min, "
                      f"budget {args.max_compute_units} units")
            print(f"Stop with: python -m autoresearch.colab_runtime down --session {entry['session']}")
            return 0
        if args.command == "down":
            found = False
            for pool, state in _pools():
                names = {state["pool"], *(e["session"] for e in state["sessions"].values())}
                if args.session and args.session not in names:
                    continue
                if not args.session and state["released"] and not any(
                        e.get("up") for e in state["sessions"].values()):
                    continue
                pool.stop("manual", release=True)
                print(f"Stopped {state['pool']}")
                found = True
            if args.session and not found:
                ColabCLI().stop(args.session)
                print(f"Stopped unmanaged session {args.session}")
            return 0
        balance, rate = ColabCLI().usage()
        print(f"Balance: {balance:.2f} compute units; current rate {rate:.2f}/hr")
        for pool, state in _pools():
            if state["released"] and not any(e.get("up") for e in state["sessions"].values()):
                continue
            print(f"{state['pool']}: spent~{state.get('spent_units', 0):.2f} "
                  f"reason={state['stop_reason']}")
            for tier, entry in state["sessions"].items():
                health = pool.health(entry) if entry.get("up") else None
                detail = (f"running={health['running']} queued={health['queued']}"
                          if health else "unreachable" if entry.get("up") else "stopped")
                print(f"  {tier}: {entry['session']} gpu={entry['gpu']} {detail} "
                      f"reason={entry['stop_reason']}")
        return 0
    except RemoteExecutorUnavailable as exc:
        print(f"colab_runtime: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
