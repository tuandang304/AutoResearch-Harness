# AutoResearch-Harness

AutoResearch-Harness is a personal development version of an automated research
pipeline. From a short topic description it generates research ideas, runs
experiments with an agentic tree search, aggregates the results into figures,
writes a LaTeX paper and reviews it.

This version adds two things:

- **No API keys needed.** Every LLM call can go through a coding-agent CLI that
  you are already logged in to: **Claude Code**, **Codex** or **Antigravity**.
  Classic API keys still work.
- **GPU work runs on Google Colab.** The experiment code runs on a Colab GPU,
  while the tree search, the LLM calls and the write-up stay on your machine.
  The local machine needs neither a GPU nor PyTorch.

> **Caution:** this system executes code written by an LLM. That code can install
> packages, access the network or spawn processes. Run it in an environment you
> are comfortable giving to an untrusted program, such as a VM, a container or a
> Colab runtime.

## Contents

1. [How it works](#how-it-works)
2. [Installation](#installation)
3. [LLM providers](#llm-providers)
4. [Running experiments on Google Colab](#running-experiments-on-google-colab)
5. [Quick start](#quick-start)
6. [Generating research ideas](#generating-research-ideas)
7. [Running the pipeline](#running-the-pipeline)
8. [Troubleshooting](#troubleshooting)
9. [Project layout](#project-layout)
10. [Origin, license and required disclosure](#origin-license-and-required-disclosure)

## How it works

```
 topic.md ──▶ ideation ──▶ ideas.json ──▶ tree search (4 stages) ──▶ plot aggregation ──▶ write-up ──▶ review
              (LLM + Semantic Scholar)     │ draft → debug/improve                         (LaTeX PDF)
                                          │ ├ stage 1  initial implementation
                                          │ ├ stage 2  baseline tuning
                                          │ ├ stage 3  creative research
                                          │ └ stage 4  ablation studies
                                          └─ each node: LLM writes code → code runs (local or Colab GPU)
                                             → LLM checks output, parses metrics, plots, reviews plots
```

```
 your machine                                     Google Colab (GPU)
 ────────────                                     ──────────────────
 launch_scientist_bfts.py                         notebooks/colab_gpu_executor.ipynb
  ├─ LLM calls (claude / codex / agy / APIs)       └─ colab_server.py
  └─ RemoteInterpreter ── HTTPS (cloudflared) ──▶     runs the experiment code on the GPU
       uploads workspace, polls, syncs results ◀──    returns output + working/ files
```

## Installation

Linux, Python 3.11. [uv](https://docs.astral.sh/uv/) is used below, but any
virtualenv tool works.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# LaTeX + PDF tools for the write-up stage (Ubuntu/Debian)
sudo apt install texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra \
                 poppler-utils chktex
```

PyTorch is only needed locally if you run experiments on your own GPU
(`exec.backend: local`), e.g. `uv pip install torch torchvision`.

## LLM providers

A model is named by a string. It can be a CLI model (`<provider>/<model>`) or an
API model (e.g. `gpt-4o-2024-11-20`). Providers can be mixed freely across stages.

### Coding-agent CLIs (recommended)

| Provider | CLI | Example model names | Log in with |
|---|---|---|---|
| Claude Code | `claude` | `claude-code/opus`, `claude-code/sonnet`, `claude-code/fable`, `claude-code/haiku` | `claude` |
| Codex | `codex` | `codex/default` (model from `~/.codex/config.toml`), `codex/<model>` | `codex login` |
| Antigravity | `agy` | `antigravity/gemini-3.1-pro-high`, `antigravity/gemini-3.8-flash-medium` (list: `agy models`) | `agy` |

`<provider>/default` uses whatever model the CLI is configured with. These names
work everywhere a model is accepted: `--model` for ideation, the `--model_*`
flags, and the `model:` fields in `bfts_config.yaml`, including vision
(`vlm_feedback`).

Check the CLIs once:

```bash
bash scripts/setup_cli_providers.sh
```

The script checks that each CLI is installed and logged in, and checks the
LaTeX tools. It also lets Antigravity read the plots it must review, by adding a
`read_file` rule scoped to the harness's scratch directory. `agy` is usually a
snap, so that rule goes in
`~/snap/antigravity-cli/common/.gemini/antigravity-cli/settings.json`.

**Using one provider for everything:** `launch_scientist_bfts.py --provider <name>`
sets every stage's model, including the tree-search models in `bfts_config.yaml`:

| `--provider` | code generation and write-up | feedback, VLM, summaries, citations, review |
|---|---|---|
| `claude-code` | `claude-code/opus` | `claude-code/sonnet` |
| `codex` | `codex/default` | `codex/default` |
| `antigravity` | `antigravity/gemini-3.1-pro-high` | `antigravity/gemini-3.8-flash-medium` |

Explicit `--model_*` flags take precedence. The presets live in `PROVIDER_PRESETS`
in `ai_scientist/cli_llm.py`.

**How it works** (`ai_scientist/cli_llm.py`, `ai_scientist/treesearch/backend/backend_cli.py`):
- Each call runs the CLI non-interactively (`claude -p`, `codex exec`, `agy -p`)
  in an empty scratch directory. Claude Code runs with all tools disabled and
  `--safe-mode`. Codex runs in its `read-only` sandbox. Antigravity may only read
  its scratch directory.
- Function calling, which the tree search depends on, is emulated. The model is
  asked for JSON matching the schema, the JSON is validated, and on failure it is
  re-requested with the error.
- Multi-turn histories are sent as a transcript, and images as attachments.
- Temperature and max-tokens settings are ignored because the CLIs don't expose
  them. Token counts are still written to `token_tracker.json`, with cost 0.

| Environment variable | Default | Meaning |
|---|---|---|
| `AI_SCIENTIST_CLI_TIMEOUT` | 1200 | seconds per call |
| `AI_SCIENTIST_CLI_RETRIES` | 3 | attempts per call (backoff 15 s, 45 s) |
| `AI_SCIENTIST_CLI_STARTUP_TIMEOUT` | 90 | kill and retry a CLI that prints nothing (`agy` sometimes hangs at startup) |
| `AI_SCIENTIST_CLI_TMP` | per provider | scratch directory for CLI calls |

**Limits:**
- A full run makes several hundred calls, some in parallel (`agent.num_workers`),
  so it can use up a subscription's usage limits.
- Codex and Antigravity add about 12k tokens of agent system prompt to every call.
- Antigravity is the slowest provider, at about 30 s per call.

### API keys (optional)

| Models | Environment variables |
|---|---|
| OpenAI (`gpt-*`, `o1*`, `o3*`) | `OPENAI_API_KEY` |
| Anthropic API (`claude-*`) | `ANTHROPIC_API_KEY` |
| Claude via AWS Bedrock (`bedrock/...`, and the default `agent.code.model`) | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME` (`pip install anthropic[bedrock]`) |
| Claude via Vertex AI (`vertex_ai/...`) | Google Cloud credentials |
| Gemini API (`gemini-*`) | `GEMINI_API_KEY` |
| DeepSeek / OpenRouter / HuggingFace / Ollama | `DEEPSEEK_API_KEY` / `OPENROUTER_API_KEY` / `HUGGINGFACE_API_KEY` / local Ollama |

### Semantic Scholar (literature search)

Ideation and citation gathering query Semantic Scholar. Without a key you may
hit rate limits. With one, set `S2_API_KEY`
([request a key](https://www.semanticscholar.org/product/api)).

## Running experiments on Google Colab

1. In Colab, open `notebooks/colab_gpu_executor.ipynb` (File → Upload notebook),
   select a GPU runtime and choose **Run all**. It starts the executor behind a
   Cloudflare quick tunnel and prints `{"url": ..., "token": ...}`.
2. Save that JSON as `remote_executor.json` in the repository root. It is
   gitignored. Alternatively, set `AI_SCIENTIST_REMOTE_URL` and
   `AI_SCIENTIST_REMOTE_TOKEN`.
3. Check the connection and GPU:
   ```bash
   python -m ai_scientist.treesearch.remote_interpreter --check
   ```
4. Run with `--exec_backend colab`, or set `exec.backend: colab` in `bfts_config.yaml`.

| `bfts_config.yaml` | Default | Meaning |
|---|---|---|
| `exec.backend` | `local` | `local` or `colab` |
| `exec.remote_max_file_mb` | 100 | files larger than this (e.g. checkpoints) stay on Colab |
| `exec.remote_wait_minutes` | 60 | how long running experiments wait for Colab after a disconnect |

**Notes:**
- Set `agent.num_workers` to the notebook's `MAX_CONCURRENT` (default 2),
  because all workers share the one GPU. Extra jobs queue up.
- Before each run, the worker's workspace is uploaded to a fresh directory on
  Colab. After the run, the remote workspace is mirrored back.
- Seed aggregation, plot aggregation and the write-up always run locally.
- **Disconnects:** re-run the notebook and update `remote_executor.json`. The
  file is re-read on every request, so the run continues without a restart.
  Colab sessions last at most 12 h on the free tier and 24 h on Pro+.
- **Colab policy:** the free tier disallows "remote control" and "running
  distributed computing workers", and this setup drives a Colab runtime from
  your machine. Use a paid plan (Pro, Pro+ or pay-as-you-go compute units),
  where those restrictions are lifted.
- **Security:** anyone with the tunnel URL and token can run code on the
  runtime. The token is regenerated each time the notebook's start cell runs.
- **Monitoring:** the notebook's last cell prints `running=… queued=… gpu=…`
  every minute. If it stops, the runtime has died.
- After editing `ai_scientist/remote/colab_server.py`, regenerate the notebook
  with `python scripts/build_colab_notebook.py`.

## Quick start

```bash
source .venv/bin/activate
bash scripts/setup_cli_providers.sh                      # once

# 1. ideas from a topic description
python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file ai_scientist/ideas/my_topic.md \
  --model claude-code/opus --max-num-generations 10 --num-reflections 3

# 2. experiments on Colab + paper, all LLM calls via Claude Code
python launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/my_topic.json --idea_idx 0 \
  --provider claude-code --exec_backend colab
```

## Generating research ideas

1. Write a topic description in Markdown with `Title`, `Keywords`, `TL;DR` and
   `Abstract` sections. See `ai_scientist/ideas/i_cant_believe_its_not_better.md`.
   To steer the experiments towards smaller models or datasets that fit your
   GPU, say so in this file.
2. Run `ai_scientist/perform_ideation_temp_free.py`:
   - `--workshop-file`: the topic file
   - `--model`: any model name from [LLM providers](#llm-providers)
   - `--max-num-generations`: number of ideas to generate
   - `--num-reflections`: refinement rounds per idea
3. The ideas are written next to the topic file, as a JSON file with the same
   name (e.g. `my_topic.json`). Each idea has a hypothesis, the planned
   experiments and related work, and was checked for novelty against Semantic
   Scholar.

## Running the pipeline

`launch_scientist_bfts.py` runs one idea through experiments, plot aggregation,
write-up and review.

| Flag | Meaning |
|---|---|
| `--load_ideas`, `--idea_idx` | idea file and which idea to run |
| `--provider` | `claude-code`, `codex` or `antigravity` for all models |
| `--exec_backend` | `local` or `colab` |
| `--model_writeup`, `--model_writeup_small`, `--model_citation`, `--model_agg_plots`, `--model_review` | per-stage models (override `--provider`) |
| `--writeup-type` | `icbinb` (4 pages, default) or `normal` (8 pages) |
| `--num_cite_rounds` | citation-gathering rounds (default 20) |
| `--load_code` | seed the experiments with `<ideas>.py` next to the idea file |
| `--skip_writeup`, `--skip_review` | stop after experiments / after the write-up |
| `--attempt_id` | distinguishes parallel attempts of the same idea |

Tree-search settings live in `bfts_config.yaml`:

- `agent.num_workers`: nodes expanded in parallel. With Colab, match `MAX_CONCURRENT`.
- `agent.stages.stage{1..4}_max_iters`: iteration budget per stage.
- `agent.multi_seed_eval.num_seeds`: seeds used to re-run the best node (1–3).
- `agent.search.num_drafts`, `max_debug_depth`, `debug_prob`: number of initial
  drafts and how hard to try fixing buggy nodes.
- `agent.code/feedback/vlm_feedback/summary/select_node.model`: models per role.
  `--provider` writes all of these.
- `exec.timeout`: time limit per experiment run, in seconds.

`k_fold_validation`, `expose_prediction` and `data_preview` are currently unused.

**Outputs** go to `experiments/<date>_<idea>_attempt_<id>/`:
- `logs/0-run/unified_tree_viz.html`: interactive view of the search tree
- `logs/0-run/experiment_results/`: code, data and plots of each successful node
- `figures/`: aggregated figures used in the paper
- `<date>_<idea>_attempt_<id>.pdf`: the paper (`*_reflection*.pdf` are
  intermediate drafts; the review uses the final one)
- `review_text.txt`, `review_img_cap_ref.json`: automated review
- `token_tracker.json`: LLM usage per model

## Troubleshooting

- **No PDF or review was produced.** Success depends on the model and the
  difficulty of the idea. Check `logs/0-run/` to see whether any node reached a
  working implementation, and use stronger models for `agent.code`.
- **CUDA out of memory on Colab.** Lower `MAX_CONCURRENT` (and
  `agent.num_workers`), choose a bigger GPU, or ask for smaller models in the
  topic description.
- **"Remote executor is not configured / unreachable".** Start or re-run the
  notebook, update `remote_executor.json`, and check with
  `python -m ai_scientist.treesearch.remote_interpreter --check`.
- **A CLI provider fails.** Run `bash scripts/setup_cli_providers.sh`. A usage
  limit reached on your subscription appears as repeated failures in the log.
  Switch the provider or wait.
- **Semantic Scholar errors or rate limits.** Set `S2_API_KEY`, or skip
  citations with fewer `--num_cite_rounds`.

## Project layout

```
launch_scientist_bfts.py          end-to-end pipeline for one idea
bfts_config.yaml                  tree-search, model and execution settings
ai_scientist/
  perform_ideation_temp_free.py   idea generation
  perform_plotting.py             figure aggregation
  perform_icbinb_writeup.py       4-page write-up (default); perform_writeup.py: 8-page
  perform_llm_review.py, perform_vlm_review.py   automated review
  llm.py, vlm.py                  model clients (API and CLI)
  cli_llm.py                      Claude Code / Codex / Antigravity adapter
  treesearch/                     agentic tree search (stages, journal, interpreter)
    backend/backend_cli.py        CLI backend with function-call emulation
    remote_interpreter.py         runs experiment code on the Colab executor
  remote/colab_server.py          executor server that runs on Colab
notebooks/colab_gpu_executor.ipynb   generated by scripts/build_colab_notebook.py
scripts/setup_cli_providers.sh    CLI login and permission check
```

The Python package (`ai_scientist`) and the environment variables
(`AI_SCIENTIST_*`) keep their original names so that imports and existing
configs keep working.

## Origin, license and required disclosure

AutoResearch-Harness is a derivative of
[SakanaAI/AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2), and its
tree search builds on [AIDE](https://github.com/WecoAI/aideml). It is
distributed under **The AI Scientist Source Code License** (see `LICENSE`, which
must be included in full with any copy of this code). The license's use
restrictions (Section 3.2) apply to this version as well.

**Mandatory disclosure:** any manuscript, paper or technical report produced
with this code must state, prominently (e.g. in the abstract or a
Disclosure/Methods section), that it was machine-generated using The AI
Scientist, for example:

> "This manuscript was autonomously generated using [The AI Scientist](https://github.com/SakanaAI/AI-Scientist)."
