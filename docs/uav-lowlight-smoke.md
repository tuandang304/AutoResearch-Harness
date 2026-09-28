# Low-light UAV detection: T4 smoke test

This profile tests the research infrastructure, not detector quality. Its data
are synthetic aerial-style rectangles, not real UAV imagery. A real-data study
is a separate run after this test passes.

## Profile

- Explicit `codex/gpt-6-sol` for every model role. A live text call succeeded
  during setup; image/structured-output behavior with this model is not yet
  verified end to end.
- Colab execution, one worker, stage one only, up to two search attempts and
  one additional seed evaluation. Each remote script has a 600-second timeout;
  reconnection waits up to two minutes. These are not a total wall-clock or
  token-cost cap: model calls, plots and reviews take additional time.
- Requested workload: 128/32/32 synthetic train/validation/test images at
  128x128, one tiny detector, batch 8, two epochs, no downloads.
- CUDA is required in the generated training script. Inspect its logs for
  the actual GPU name, CUDA tensors, finite losses and allocated memory.
- Stage-level plotting and visual feedback remain enabled. Final cross-stage
  plotting, manuscript generation and paper review are skipped.

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
CLI default. The connection check alone is not proof of GPU training; it can
fall back to CPU. Confirm CUDA/T4 in the experiment logs.

## Acceptance checklist

1. A working generated implementation runs remotely on the T4, with finite
   training losses and recorded GPU metadata; no CPU training fallback.
2. `experiment_data.npy`, prediction montage and loss plots return locally.
3. Metric extraction and visual feedback complete, with precision/recall
   computed from actual bounding boxes. Zero recall is acceptable; invented
   scores, classifier accuracy presented as detection quality, or NaNs are not.
4. The run stops after stage one and records a completed `run_status.json`.
   Inspect the journal for failed seed/plot nodes too: top-level completion
   alone does not certify every artifact or check passed.

Outputs live under `experiments/*uav_lowlight_t4_smoke*`. Review the saved code
to confirm it followed the requested workload: those training limits are
instructions to the coding model, whereas stage/worker/execution limits are
enforced by the harness. No real GPU run has been verified during preparation.
