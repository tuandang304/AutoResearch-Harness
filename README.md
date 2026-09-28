# AutoResearch-Harness

Turn a research question into experiments, plots, a paper draft and automated
review. AutoResearch-Harness coordinates the work on your laptop and can send
generated experiment code to a remote GPU runtime.

- **Bring your coding assistant:** Claude Code, Codex or Antigravity through a
  locally installed, authenticated CLI. API integrations are also available.
- **Choose where experiments run:** local execution or a Colab executor.
- **Inspect each run:** resolved configuration, experiment code, artifacts, a
  search-tree viewer, terminal status and token usage.
- **Validate before spending:** an offline dry run checks inputs and shows
  exactly which models and execution backend will be used.

This is an active development project. Generated code, scientific claims and
paper drafts require human inspection. A successful run is not evidence that
a finding is novel or correct.

## Start here

### Orchestrator–worker model routing

The default and UAV smoke configurations use **GPT-6-Astra** for orchestration
and **GPT-6-Sol** for workers, with `codex_reasoning_effort: low` for both.
The launcher passes this to every Codex subprocess as
`-c 'model_reasoning_effort="low"'`, including manuscript and review calls.
This setting is scoped to the run; it does not modify your personal Codex config.

Use `--orchestrator-model codex/gpt-6-astra --worker-model PROVIDER/MODEL`
with your normal run command. Replace `PROVIDER/MODEL` with an authenticated,
lower-cost model available to your account; workers need image support for plot
review. `--dry-run` shows the resolved assignments without making model calls.

The orchestrator makes stage decisions, proposes tuning and ablation experiments,
selects promising results, and writes and reviews the final paper. Workers generate
and debug code, extract metrics, create and inspect plots, summarize results, and
assist with citations and writing. The manager delegates through the existing
experiment task prompts and receives execution results and structured feedback.
This reuses the existing worker processes without adding a planning call to every
routine operation. Colab training configuration is independent of model routing.

For YAML configuration, set `agent.orchestrator: {model: codex/gpt-6-astra, temp: 0.2}`
and assign worker models to `agent.code`, `agent.feedback`, `agent.vlm_feedback`,
`agent.summary`, and `report`. Node selection follows the orchestrator. CLI role
flags such as `--model_review` still override paper defaults. Existing configs keep
their previous behavior. Setting only `--worker-model` retains the original code
model as orchestrator. Worker failures use existing bounded retry/debug paths;
there is no automatic escalation to the expensive model. Actual savings depend
on provider billing and retries; no cost reduction has yet been benchmarked.

For a small GPU pipeline check, see the [low-light UAV T4 smoke test](docs/uav-lowlight-smoke.md).

Linux and Python 3.11 are the local development baseline.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# No model calls, training, authentication or output files:
python -m autoresearch --dry-run --provider codex --exec_backend colab
```

The included [example idea](examples/ideas.json) uses small synthetic data and is
intended to exercise the pipeline. It is not a publishable research proposal.
Use [examples/topic.md](examples/topic.md) as a starting point for your own topic.

For paper compilation on Ubuntu/Debian:

```bash
sudo apt install texlive-latex-extra texlive-fonts-recommended \
  texlive-bibtex-extra poppler-utils chktex
```

Install PyTorch and experiment-specific packages in the environment that runs
the experiments. Local orchestration does not require PyTorch when using Colab,
although generated local plotting code may request additional packages.

## Models

| Provider | Selector | Code / paper | Feedback / review |
|---|---|---|---|
| Claude Code | `--provider claude-code` | `claude-code/opus` | `claude-code/sonnet` |
| Codex | `--provider codex` | `codex/default` | `codex/default` |
| Antigravity | `--provider antigravity` | `antigravity/default` | `antigravity/default` |

Install and authenticate the selected CLI first. A name such as
`codex/default` uses that CLI's configured model. Any supported model can be
selected with `<provider>/<model>`; availability depends on your account.

The repository configuration defaults to Claude Code and one local worker.
A provider override applies to all tree-search roles and the paper pipeline.
Explicit `--model_*` flags override the corresponding paper-stage model.
Without a provider override, the paper writer inherits `agent.code.model` and
the other paper stages inherit `agent.feedback.model`.

```bash
# Check installed tools without making model calls:
bash scripts/setup_cli_providers.sh --no-test

# Also make small authenticated test calls:
bash scripts/setup_cli_providers.sh
```

The setup script merges a scoped image-read permission into Antigravity's
settings. For snap installations it uses the snap home. CLI permissions still
depend on the installed CLI and your settings; this adapter is not a security
sandbox. Claude runs with tools disabled, Codex requests a read-only sandbox,
and Antigravity receives instructions to use its file viewer for images.

A full research run can make hundreds of calls and consume subscription or API
allowances. CLI sampling controls differ from API controls; temperature and
max-token arguments are not enforced by these adapters. No particular
latency, price or subscription entitlement is guaranteed.

Legacy API clients remain in `ai_scientist/llm.py`, `vlm.py` and the tree-search
backends. Their model coverage differs across stages; test your chosen
configuration before a long run. Common credentials are `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY` and AWS credentials for Bedrock.
Literature search uses Semantic Scholar; `S2_API_KEY` is optional.

## Experiments on Colab

1. Upload [notebooks/colab_gpu_executor.ipynb](notebooks/colab_gpu_executor.ipynb)
   to Colab and select a GPU runtime.
2. Run the setup and start cells. Copy the printed URL/token JSON into
   `remote_executor.json` in the repository root. This file is gitignored.
3. Check the connection:
   ```bash
   python -m ai_scientist.treesearch.remote_interpreter --check
   ```
4. Start with one worker and the example idea:
   ```bash
   python -m autoresearch --provider claude-code --exec_backend colab \
     --num-workers 1 --load_ideas examples/ideas.json --skip_writeup
   ```

To include the paper and review, omit `--skip_writeup`. Plot aggregation still
runs locally when write-up is skipped. The GPU check prints whether CUDA is
available; a successful CPU fallback alone does not establish GPU availability.

The notebook contains a status monitor and a manual stop cell. Interrupt the
monitor before running the stop cell. Monitoring does not prevent Colab
timeouts. Check the current [Colab usage restrictions](https://research.google.com/colaboratory/faq.html)
and your plan before using a remotely controlled worker.

Generated experiments execute with the runtime user's permissions. Use a
dedicated runtime without sensitive mounted storage. Workspace-transfer
validation is not isolation for the code itself.

See [remote execution](docs/remote-execution.md) for reconnects, retained
checkpoints, transfer limits and protocol details.

## Research workflow

1. Describe a hypothesis, compute budget, data and evaluation criteria in a
   topic file.
2. Generate candidate ideas:
   ```bash
   python -m ai_scientist.perform_ideation_temp_free \
     --workshop-file examples/topic.md --model claude-code/sonnet \
     --max-num-generations 3 --num-reflections 2
   ```
3. Inspect the resulting JSON and select an idea. Literature-search output
   should be checked; it does not certify novelty.
4. Validate your settings, then run:
   ```bash
   python -m autoresearch --load_ideas examples/topic.json --idea_idx 0 \
     --provider claude-code --exec_backend colab --dry-run
   # Remove --dry-run when ready to execute.
   ```

The search progresses through implementation, baseline tuning, research
variants and ablations. Each node contains generated code, execution results,
metric analysis and plot feedback. The final stages aggregate plots, write a
LaTeX manuscript and request automated review.

## Configuration

`bfts_config.yaml` is the default. Use `--config path/to/config.yaml` for a
separate configuration; per-run changes do not modify that source file.

| Option | Purpose |
|---|---|
| `--dry-run` | Validate and print configuration without a live run |
| `--provider` | Select a CLI provider for every model role |
| `--exec_backend local\|colab` | Select experiment execution |
| `--num-workers` | Override concurrent tree-search workers |
| `--output-dir` | Store runs in another directory |
| `--load_ideas`, `--idea_idx` | Select a JSON idea |
| `--model_writeup`, `--model_writeup_small`, `--model_citation`, `--model_agg_plots`, `--model_review` | Override paper-stage models |
| `--writeup-type normal\|icbinb` | Eight-page or four-page format |
| `--skip_writeup` | Experiments and plot aggregation only |
| `--skip_review` | Skip automated paper review |
| `--num_cite_rounds`, `--writeup-retries` | Citation and write-up budgets |
| `--load_code` | Include the Python file beside the ideas JSON |

Important YAML settings include `agent.stages.stage*_max_iters`,
`agent.multi_seed_eval.num_seeds`, `agent.search` and `exec.timeout`.
Keep worker concurrency appropriate for the remote GPU; the executor queues
excess jobs. Seeds and workers are independent counts.

CLI controls: `AI_SCIENTIST_CLI_TIMEOUT` (1200 seconds),
`AI_SCIENTIST_CLI_STARTUP_TIMEOUT` (90 seconds for streaming providers),
`AI_SCIENTIST_CLI_RETRIES` (3) and `AI_SCIENTIST_CLI_TMP` (scratch directory).
Environment variable values must be positive where applicable.

## Outputs and failures

Each run creates a unique directory under `experiments/` or `--output-dir`:

- `idea.json`, `idea.md`, `bfts_config.yaml`: inputs and resolved settings.
- `run_status.json`: current stage and completed, failed or interrupted status.
- `logs/0-run/`: experiment artifacts, summaries and the HTML search tree.
- `figures/`, `latex/`, `*.pdf`: figures and manuscript outputs when produced.
- `review_text.txt`, `review_img_cap_ref.json`: automated review.
- `usage.jsonl`: append-only token ledger shared by worker processes.
- `token_tracker.json`: aggregate token usage and available price estimates.

An unknown price is `null`, not zero. Price entries are estimates, not billing
records. Interaction details remain process-local; the token ledger aggregates
usage across processes.

Failures return a nonzero exit code and preserve outputs. Ctrl+C records
`interrupted` and cleans up descendant processes. Existing unrelated Python
sessions are not scanned or terminated. Status files are diagnostic; automatic
resume of the full research pipeline is not implemented.

## Development

```bash
python -m unittest discover -s tests -v
python scripts/build_colab_notebook.py
git diff --check
```

Tests use fake model responses and a real localhost executor; no CLI login,
subscription calls, CUDA or Colab account is required. GitHub Actions runs the
suite on Python 3.11 and 3.12 and checks the embedded notebook stays current.

See [development notes](docs/development.md) for architecture, test coverage
and remaining limitations. `python launch_scientist_bfts.py` remains supported;
`ai_scientist` imports and `AI_SCIENTIST_*` settings are compatibility interfaces.

## License and disclosure

Distribution and use are subject to the complete [LICENSE](LICENSE).
Component provenance is recorded in [NOTICE.md](NOTICE.md).

Machine-generated manuscripts and reports must prominently disclose that
fact. For example: “This manuscript was produced with AutoResearch-Harness
using AI-generated text, code and analysis, and was reviewed by its authors.”
Only claim human review when it actually occurred.
