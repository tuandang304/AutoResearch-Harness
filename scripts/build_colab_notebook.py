"""Generate notebooks/colab_gpu_executor.ipynb with the remote executor server embedded.

Re-run after editing ai_scientist/remote/colab_server.py:

    python scripts/build_colab_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = (ROOT / "ai_scientist" / "remote" / "colab_server.py").read_text()
WORKSPACE = (ROOT / "ai_scientist" / "remote" / "workspace.py").read_text()
OUT = ROOT / "notebooks" / "colab_gpu_executor.ipynb"


def md(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.strip("\n").splitlines(True),
    }


def code(src):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.strip("\n").splitlines(True),
    }


cells = [
    md("""
# AutoResearch-Harness — Colab GPU executor

This notebook turns a Colab GPU runtime into the place where AutoResearch-Harness
runs its **experiment code**. The tree search, all LLM calls (API keys or the
Claude Code / Codex / Antigravity CLIs) and the paper write-up stay on your
own machine.

1. `Runtime → Change runtime type → GPU`, then **Run all**.
2. Copy the JSON printed by step 4 into `remote_executor.json` in your local
   AutoResearch-Harness folder.
3. Locally: `python -m ai_scientist.treesearch.remote_interpreter --check`, then
   launch with `--exec_backend colab`.
4. Keep this tab open. The monitoring cell prints status; it does not prevent
   runtime timeouts or guarantee continued GPU access.
   If the runtime restarts, run all cells again and update `remote_executor.json`.
   Running experiments wait up to `exec.remote_wait_minutes` for it to come back.

**Colab usage policy:** check the current [Colab FAQ](https://research.google.com/colaboratory/faq.html)
and your plan's restrictions before running a remotely controlled worker.

**Security:** anyone with the URL *and* the token can run code on this runtime.
Don't share them. A new token is generated every time step 4 runs.
"""),
    md("## 1. Check the GPU"),
    code("!nvidia-smi"),
    md("## 2. Write the executor server"),
    code("%%writefile /content/colab_server.py\n" + SERVER),
    code("%%writefile /content/workspace.py\n" + WORKSPACE),
    md("""
## 3. Install packages that experiments commonly use

Colab already ships PyTorch, NumPy, pandas, scikit-learn, xgboost, lightgbm,
matplotlib and transformers.
"""),
    code(
        "!pip -q install datasets humanize timm albumentations "
        "bayesian-optimization torch_geometric statsmodels seaborn"
    ),
    md("""
## 4. Start the executor and a Cloudflare quick tunnel

`MAX_CONCURRENT` = how many experiments may share the GPU at once. Keep
`agent.num_workers` in `bfts_config.yaml` equal to it. 1–2 suits a T4 (16 GB);
an L4 or A100 can take 3–4.
"""),
    code(r"""
import json, os, re, secrets, subprocess, time, urllib.request

MAX_CONCURRENT = 1
PORT = 8765

for p in ("server", "tunnel"):
    try:
        globals()[p].terminate()
        globals()[p].wait(timeout=10)
    except subprocess.TimeoutExpired:
        globals()[p].kill()
        globals()[p].wait()
    except Exception:
        pass

TOKEN = secrets.token_urlsafe(32)
env = dict(os.environ, AISCI_TOKEN=TOKEN, AISCI_MAX_CONCURRENT=str(MAX_CONCURRENT))
server = subprocess.Popen(
    ["python", "/content/colab_server.py", "--port", str(PORT)],
    stdout=open("/content/server.log", "w"), stderr=subprocess.STDOUT, env=env,
)

if not os.path.exists("/content/cloudflared"):
    urllib.request.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        "/content/cloudflared",
    )
    os.chmod("/content/cloudflared", 0o755)
tunnel = subprocess.Popen(
    ["/content/cloudflared", "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
    stdout=open("/content/tunnel.log", "w"), stderr=subprocess.STDOUT,
)

url = None
for _ in range(90):
    time.sleep(1)
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open("/content/tunnel.log").read())
    if m:
        url = m.group(0)
        break
if url is None:
    print(open("/content/tunnel.log").read()[-3000:])
    raise RuntimeError("cloudflared did not print a tunnel URL")

time.sleep(3)
print(open("/content/server.log").read())
if server.poll() is not None or tunnel.poll() is not None:
    raise RuntimeError("Server or tunnel exited; inspect /content/server.log and /content/tunnel.log")
req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health",
                             headers={"Authorization": f"Bearer {TOKEN}"})
with urllib.request.urlopen(req, timeout=15) as response:
    assert json.load(response)["protocol_version"] == 2
print("Paste this into remote_executor.json on your machine:\n")
print(json.dumps({"url": url, "token": TOKEN}, indent=2))
"""),
    md("""
## 5. Monitor the executor

Leave this cell running. It prints the queue and GPU status every minute.
Interrupting it does **not** stop the executor.
"""),
    code(r"""
import json, time, urllib.request

def health():
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health",
                                 headers={"Authorization": f"Bearer {TOKEN}"})
    return json.load(urllib.request.urlopen(req, timeout=10))

while True:
    try:
        h = health()
        print(time.strftime("%H:%M:%S"), f"running={h['running']} queued={h['queued']} gpu={h['gpu']}", flush=True)
    except Exception as e:
        print(time.strftime("%H:%M:%S"), "executor not responding:", e, flush=True)
    if server.poll() is not None or tunnel.poll() is not None:
        print("server or tunnel exited; re-run step 4")
        print(open("/content/server.log").read()[-2000:])
        break
    time.sleep(60)
"""),
    md("""
## 6. Stop the executor and tunnel

Interrupt the monitoring cell, then run this cell manually. It stops the
executor, its experiments and the tunnel. It does not delete completed artifacts.
"""),
    code(r"""
for name in ("server", "tunnel"):
    process = globals().get(name)
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
print("Executor and tunnel stopped.")
"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

OUT.parent.mkdir(exist_ok=True)
for index, cell in enumerate(cells):
    cell["id"] = f"autoresearch-{index:02d}"
OUT.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {OUT.relative_to(ROOT)}")
