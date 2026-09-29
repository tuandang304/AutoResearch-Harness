"""Offline stand-in for the `colab` CLI used by tests/test_colab_runtime.py.

State lives in $FAKE_COLAB_DIR. `exec` does not run the uploaded bootstrap;
it starts the real executor server on localhost with the uploaded token and
prints the same AISCI_* markers, so the controller path is exercised end to end.
Like the real CLI, `exec` exits 0 even when the remote code fails.
"""

import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HOME = Path(os.environ["FAKE_COLAB_DIR"])
STATE = HOME / "state.json"
GPU_NAMES = {"cpu": "none", "T4": "Tesla T4, 15360 MiB", "L4": "NVIDIA L4, 23034 MiB",
             "A100": "NVIDIA A100-SXM4-40GB, 40960 MiB", "H100": "NVIDIA H100 80GB HBM3, 81559 MiB"}
RATES = {"cpu": 0.08, "T4": 1.19, "L4": 1.71, "A100": 5.4, "H100": 9.0}


def load():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"sessions": {}, "balance": 100.0, "rate": 0.0}


def save(state):
    STATE.write_text(json.dumps(state))


def option(args, name):
    return args[args.index(name) + 1] if name in args else None


def main(args):
    with open(HOME / "calls.log", "a") as log:
        log.write(" ".join(args) + "\n")
    state = load()
    command, session = args[0], option(args, "-s")
    sessions = state["sessions"]
    if command == "usage":
        print(f"Current balance: {state['balance']:.2f} compute units")
        print(f"Usage rate: {state['rate']:.2f}/hr")
        return 0
    if command == "new":
        gpu = option(args, "--gpu") or "cpu"
        if gpu in os.environ.get("FAKE_COLAB_FAIL_GPU", "").split(","):
            print(f"[colab] Failed: 400 Client Error for url: https://x?token=SECRET{gpu}",
                  file=sys.stderr)
            return 1
        busy = os.environ.get("FAKE_COLAB_BUSY_GPU", "").split(",")  # "T4" or "T4:1" (once)
        for item in busy:
            name, _, times = item.partition(":")
            if name == gpu and state.setdefault("busy_count", 0) < int(times or 10**9):
                state["busy_count"] += 1
                save(state)
                print("ColabRequestError: Failed to issue request POST https://x/assign"
                      f"?accelerator={gpu}: Service Unavailable", file=sys.stderr)
                return 1
        sessions[session] = {"gpu": gpu}
        state["rate"] = round(state["rate"] + RATES.get(gpu, 1.0), 4)
        (HOME / "vm" / session).mkdir(parents=True, exist_ok=True)
        save(state)
        print(f"[colab] Creating session '{session}'...\n[colab] Session READY.", file=sys.stderr)
        return 0
    if command == "status":
        if (HOME / "status_fails").exists():  # e.g. DNS failure on the controller
            print("ConnectionError: Temporary failure in name resolution", file=sys.stderr)
            return 1
        if session in sessions:
            print(f"[{session}] m-s-fake | Hardware: {sessions[session]['gpu']} | Status: IDLE")
        else:
            print(f"[colab] Session '{session}' not found.")
        return 0
    if command == "upload":
        local, remote = args[-2], args[-1]
        shutil.copy(local, HOME / "vm" / session / Path(remote).name)
        return 0
    if command == "exec":
        vm = HOME / "vm" / session
        if "-f" not in args:  # diagnostics snippet on stdin
            print("fake tunnel log: failed to dial to edge with quic")
            return 0
        gpu = sessions[session]["gpu"]
        print("AISCI_GPU=" + os.environ.get("FAKE_COLAB_GPU_NAME_" + gpu, GPU_NAMES.get(gpu, "none")),
              flush=True)
        if os.environ.get("FAKE_COLAB_BOOT_FAIL"):
            print("RuntimeError: cloudflared did not print a tunnel URL", file=sys.stderr)
            return 0
        token = (vm / ".aisci_token").read_text().strip()
        (vm / ".aisci_token").unlink()
        info = sessions[session]
        alive = False
        if info.get("server_pid"):
            try:
                os.kill(info["server_pid"], 0)
                alive = info.get("token") == token
            except ProcessLookupError:
                pass
        if not alive:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            server = subprocess.Popen(
                [sys.executable, str(vm / "colab_server.py"), "--port", str(port)],
                env=dict(os.environ, AISCI_TOKEN=token,
                         AISCI_MAX_CONCURRENT=os.environ.get("FAKE_COLAB_SLOTS", "1"),
                         AISCI_JOBS_ROOT=str(vm / "jobs"), FAKE_TIER=info["gpu"],
                         FAKE_SESSION=session),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
            )
            info.update(server_pid=server.pid, port=port, token=token)
            save(state)
            for _ in range(100):
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                    break
                except OSError:
                    time.sleep(0.05)
        port = 9 if os.environ.get("FAKE_COLAB_BAD_URL") else info["port"]  # discard port
        print(f"AISCI_URL=http://127.0.0.1:{port}", flush=True)
        return 0
    if command == "stop":
        info = sessions.pop(session, None)
        if info:
            state["rate"] = max(0.0, round(state["rate"] - RATES.get(info["gpu"], 1.0), 4))
        if info and info.get("server_pid"):
            try:
                os.killpg(info["server_pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
        save(state)
        print(f"[colab] Session '{session}' not found." if info is None
              else "[colab] Session terminated.", file=sys.stderr)
        return 0
    print(f"unsupported fake command {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
