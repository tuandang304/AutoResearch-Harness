# Low-light UAV detection: T4 smoke test

This profile tests the research infrastructure, not detector quality. Its data
are synthetic aerial-style rectangles, not real UAV imagery. A real-data study
is a separate run after this test passes.

## Profile

- Explicit `codex/gpt-6-sol` for every model role. A live text call succeeded
  during setup. End-to-end results must be checked in the run journal.
- Colab execution, one worker, stage one only, up to two search attempts and
  one additional seed evaluation. Each remote script has a 600-second timeout;
  reconnection waits up to two minutes. Each full worker node has a separate
  1,800-second `agent.worker_timeout`; a timeout fails the run rather than
  silently retrying. These are not a total wall-clock or
  token-cost cap: model calls, plots and reviews take additional time.
- Requested workload: 128/32/32 synthetic train/validation/test images at
  128x128, one tiny detector, batch 8, two epochs, no downloads.
- CUDA is required in the generated training script. Inspect its logs for
  the actual GPU name, CUDA tensors, finite losses and allocated memory.
- The generated training script saves loss curves and a prediction montage on
  Colab, and the executor transfers them back alongside the metrics. Full-mode
  analysis can run locally with `exec.local_postprocessing: true`.
- `agent.smoke_test: true` validates returned numpy artifacts deterministically,
  retains the generated plots/code/execution metadata, and skips LLM-based
  metric parsing, plot regeneration, visual review and seed aggregation. The
  generated experiment and one seeded repeat still execute on Colab.

## Run

From the repository root, first validate without spending model calls:

```bash
.venv/bin/python -m autoresearch \
  --config configs/uav_lowlight_t4_smoke.yaml \
  --load_ideas examples/uav_lowlight_smoke.json \
  --experiments-only --dry-run
```

Open `notebooks/colab_gpu_executor.ipynb` in your Colab T4 GPU runtime and run
its setup/start cells. Store the printed URL/token JSON in the gitignored
`remote_executor.json` at the repository root. Do not commit or publish it.
See [remote execution](remote-execution.md) for security and runtime guidance.

```bash
.venv/bin/python -m ai_scientist.treesearch.remote_interpreter --check
.venv/bin/python -m autoresearch \
  --config configs/uav_lowlight_t4_smoke.yaml \
  --load_ideas examples/uav_lowlight_smoke.json \
  --experiments-only
```

Do not pass `--provider codex`: that overrides the explicit model with your
CLI default. The connection check now requires CUDA and tests artifact transfer.
Confirm CUDA/T4 training separately in the experiment logs.

## Acceptance checklist

1. A working generated implementation runs remotely on the T4, with finite
   training losses and recorded GPU metadata; no CPU training fallback.
2. `experiment_data.npy`, prediction montage and loss plots return locally.
3. Returned numpy artifacts load and contain finite training losses and measured
   precision/recall. Zero recall is acceptable. Inspect the saved detector code
   for actual bounding-box matching; the smoke validator does not certify
   scientific correctness. LLM visual review is excluded from this profile.
4. The run stops after stage one and records a completed `run_status.json`.
   Inspect the journal for failed seed/plot nodes too: top-level completion
   alone does not certify every artifact or check passed.

Outputs live under `experiments/*uav_lowlight_t4_smoke*`. Review the saved code
to confirm it followed the requested workload: those training limits are
instructions to the coding model, whereas stage/worker/execution limits are
enforced by the harness. Partial workspaces survive failure and interruption,
including `experiment_code.py` and `execution_result.json` once training returns.
Each worker attempt has its own directory. A previous live run verified T4
training but exposed a whole-node timeout; it was not an end-to-end pass.

## Verified run

On 2026-09-28, `2026-09-28_20-43-14_420838_uav_lowlight_t4_smoke_attempt_0`
completed stage one and its seed repeat on a Tesla T4. Both runs used 64 clean
and 64 low-light training images, two epochs, batch size 8, and fixed dataset
splits; model seeds were 42 and 0. Remote execution took 9.05 and 8.60 seconds.
Each returned a 405,813-byte data artifact, loss curves, prediction montage,
experiment code, and execution metadata. Training losses were finite and the
final run status was `completed`. The full run took about seven minutes,
including code generation, a format retry and journal saving. Subsequent
changes remove LLM ranking and summary calls from smoke journal saving; those
changes passed regression tests but were not separately timed on Colab.

This verifies the bounded smoke profile. Full research stages and manuscript
generation need their own integration run. Standard analysis was separately
verified as described below. The smoke success metric records execution/artifact validation, not
detection accuracy. Actual detector precision/recall remain in the data artifact.

## Validate standard analysis without retraining

The standard analysis path was separately verified on the saved smoke artifact
in `experiments/uav_lowlight_analysis_check_2`: execution review, generated metric
parser, structured metric extraction, generated plots, GPT-6-Sol image review
and dataset identification all passed. All 12 extracted metrics matched the
saved values within the parser's printed precision, and three valid PNG plots
were generated. Later research stages and manuscript generation remain untested.

To repeat this check, supply an experiment result directory containing
`experiment_data.npy`, `experiment_code.py`, and `execution_result.json`:

```bash
.venv/bin/python -m autoresearch.validate_analysis \
  experiments/<run>/logs/0-run/experiment_results/<experiment> \
  --config configs/uav_lowlight_t4_smoke.yaml \
  --output experiments/<new-analysis-check>
```

The command copies the saved data into a new directory and runs analysis locally.
It uses the configured model and requires CLI quota, but no new GPU training.
Each step retains generated code and results; `analysis_status.json` records
completion or the failed step. Execution review now treats low detection scores
as a result to explain, while actual execution or measurement errors remain bugs.
