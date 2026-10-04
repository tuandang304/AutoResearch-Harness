# AutoResearch-Harness

A reusable orchestrator–worker pipeline for experiments, analysis, paper drafts
and automated review. Run orchestration locally and training locally or on Colab.
Each research project owns its inputs and outputs.

## Layout

```text
autoresearch/                 Entry point, configuration and model strategy
ai_scientist/                 Research engine, adapters, execution and writing
configs/default.yaml         Shared defaults
configs/llm.yaml             Exact model profiles, instructions and fallback policy
projects/
  _template/                 Copyable structure for a new study
  regularization/            Runnable illustrative study
    brief.md                 Question and constraints
    ideas.json               Pipeline input
    runs/                    Created on execution; gitignored
  uav-lowlight/              Research brief and preserved local archive/
  time-series/               Another example research brief
docs/                        Architecture, project workflow, remote execution
notebooks/                   Colab GPU executor
scripts/                     Setup and notebook generation
tests/                       Offline regressions and fixtures
AGENTS.md                    Coding-agent instructions
```

The `ai_scientist` package and root launcher keep their names for import and
integration compatibility. Generated outputs are separate from source code.

## Setup

Use Python 3.11 on Linux:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
bash scripts/setup_cli_providers.sh --no-test
```

Authenticate the selected model CLI before a live run. Paper compilation needs
`texlive-latex-extra`, `texlive-fonts-recommended`, `texlive-bibtex-extra`,
`poppler-utils` and `chktex`. Install PyTorch and training dependencies in the
execution environment; remote training does not require local PyTorch.

## Run a project

```bash
# Validate without model calls, training or generated files.
python -m autoresearch --project projects/regularization --dry-run

# Experiments, figures, manuscript and review.
python -m autoresearch --project projects/regularization --exec_backend colab
```

The regularization example uses synthetic data; it is not a verified scientific
contribution. Other sample folders contain planning briefs. Add reviewed
ideas.json inputs before running them. To create a study, copy projects/_template/
and follow [project workflow](docs/projects.md).

Configuration precedence: explicit `--config`, project `config.yaml`, then
`configs/default.yaml`. Project configuration is a **complete file**, not an
overlay. CLI overrides are applied afterward. Without `--project`, the launcher
selects projects/regularization. Always select your study for real research.

## Models

Orchestrators fall back in order: **claude-code/claude-opus-5-5**, **codex/gpt-6-astra**,
then **antigravity/claude-opus-5-5-high**, all at **high effort**. Workers are
**claude-code/claude-sonnet-5-5** and **codex/gpt-6.1-sol** at **medium effort**, plus
**antigravity/claude-opus-5-5-high** at **high effort**. Exact profiles, task instructions
and quota groups live in [configs/llm.yaml](configs/llm.yaml).

The orchestrator selects coding/plotting workers by task fit. Sonnet handles scoped
implementation and writing; Sol handles extraction and summaries; Antigravity Opus
handles deeper analysis and visual review. These assignments require evaluation.
Write-up and final review remain pinned to Claude Code Opus. See
[routing](docs/llm-routing.md) for precedence, bounded fallback and verification.

Three experiment sub-agents run concurrently by default (`--num-workers` overrides
this; local GPU availability can cap it). Independent routed completion batches,
such as ensemble reviews, use up to three parallel sub-agents via
`routing.max_parallel_subagents`. Shared account leases still cap concurrent model
calls at two per quota group. Dependent research stages remain sequential.

The orchestrator decides stages, proposes tuning/ablations, selects results, and
writes/reviews the final paper. Workers implement/debug code, extract metrics,
generate/review plots, summarize results and assist writing.

Override assignments with `--orchestrator-model` and `--worker-model`. Paper-role
flags (`--model_writeup`, `--model_review`, `--model_citation`,
`--model_writeup_small`, `--model_agg_plots`) take precedence for their stages.
`--llm-config PATH` selects a policy; projects may provide their own `llm.yaml`.
Concrete model overrides bypass pool routing. Vision roles need image-capable models.
Effort is passed per call without changing personal CLI settings.
CLI temperature/max-token arguments are
not enforced. Actual costs depend on provider billing and retries.

`--provider claude-code`, `--provider codex`, and `--provider antigravity` select
provider presets. Explicit role flags take precedence. The Codex preset selects
its CLI default instead of the pinned pair. API adapters also remain available;
coverage differs between stages.

## Colab

**Automatic (Colab CLI).** Install and log in to the
[Colab CLI](https://pypi.org/project/google-colab-cli/) (`colab usage` should show
your balance), then set `exec.colab.auto_provision: true` in the project config
and launch with `--exec_backend colab`. The code model declares a tier for each
script (`# COMPUTE: T4`) from the configured menu (cpu, T4, L4, A100 by
default). Each tier's VM starts on its first job. VMs stop when the tree search
ends, after their idle limit, at `max_compute_units`, or if the controller
dies. When every VM of a tier is overloaded, another one of that tier starts
(`max_replicas`, `max_sessions`). Out-of-memory failures are re-run once: alone if
the GPU was shared, otherwise on the next tier. Choices and
estimated units are logged in the run's `compute.jsonl`. For a standalone
executor: `python -m autoresearch.colab_runtime up --gpu T4` (then `status` /
`down`).

**Manual (notebook).**

1. Open [the executor notebook](notebooks/colab_gpu_executor.ipynb) in a GPU runtime.
2. Run setup/start cells. Save the printed URL/token JSON in root-level
   `remote_executor.json` (gitignored).
3. Run `python -m ai_scientist.treesearch.remote_interpreter --check`.
4. Launch your project with `--exec_backend colab --num-workers 1`.

Keep tokens private. Generated code runs with runtime-user permissions. The
notebook monitor does not prevent runtime expiry. See
[remote execution](docs/remote-execution.md) for transfers and reconnects.

## Outputs and controls

Runs live in `projects/<study>/runs/<timestamp>_<idea>_attempt_<id>/` unless
`--output-dir` is supplied. A run retains input snapshots, resolved
`bfts_config.yaml`, status, experiment artifacts under `logs/0-run/`, token usage,
and generated figures, LaTeX, PDFs and reviews when their stages finish.
Routed runs also retain a policy snapshot and `routing.jsonl` model provenance.
Failures preserve outputs. Automatic full-run resume is not implemented.

- `--load_ideas PATH --idea_idx N`: select alternate ideas or a candidate.
- `--experiments-only`: stop after experiments.
- `--skip_writeup`: experiments and aggregate plots only.
- `--skip_review`: omit automated manuscript review.
- `--writeup-type icbinb|normal`: four- or eight-page manuscript.
- `--num_cite_rounds`, `--writeup-retries`: bound writing work.
- `--load_code`: include the Python file beside the ideas JSON with the same stem.

Set stage limits, seeds, concurrency and timeouts before execution. Review
generated code and scientific claims; synthetic results do not establish
real-world effectiveness. Usage estimates are not billing records.

## Development

```bash
python -m unittest discover -s tests -v
python scripts/build_colab_notebook.py
git diff --check
```

Tests mock models and use a localhost executor without paid calls or Colab.
Read [AGENTS.md](AGENTS.md) and [architecture](docs/development.md) before changes.

## License

Retain [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md), including applicable
attribution and manuscript disclosure obligations. Disclose generated text,
code and analysis; only claim human review when it actually occurred.
