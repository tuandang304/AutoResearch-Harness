# Coding-agent guide

Read README.md, docs/projects.md and docs/development.md first. The entry point
is `.venv/bin/python -m autoresearch`; launch_scientist_bfts.py implements it.
Keep ai_scientist imports and AI_SCIENTIST_* compatibility settings intact.

## Ownership and paths

- Shared engine: ai_scientist/. Public configuration/routing: autoresearch/.
- Shared defaults: configs/default.yaml. Keep study-specific prompts out of it.
- Studies: projects/<slug>/brief.md, reviewed ideas.json, optional full config.yaml.
- Outputs: project runs/, archive/, artifacts/; datasets: data/. All are ignored.
- Use --project to keep outputs with their study. Explicit --config, --load_ideas
  and --output-dir override project paths. Do not edit historical run snapshots.
- Never print remote_executor.json tokens or commit credentials/notebook outputs.
- Archived UAV results are in projects/uav-lowlight/archive/experiments. Embedded
  absolute paths may be stale; historical records are not resumable jobs.

## Models and execution

Read docs/llm-routing.md. configs/llm.yaml pins claude-opus-5-5 medium as
orchestrator. Workers: gpt-6-sol, gpt-6-luna, gpt-6-astra, claude-sonnet-5-5,
claude-opus-5-5 low;
gemini-3.8-flash high. Opus workers are last-resort, sharing orchestrator quota.
Do not silently substitute IDs or effort. Policy binds before provider/model overrides.
autoresearch/llm_router.py owns assignment, shared cooldowns and per-call provenance.
cli_llm.py translates per-call effort; legacy Codex env settings remain supported.
Policy loading/dry runs must never touch SQLite or call a model. Live policy is
snapshotted per run; SQLite availability is shared across processes, not hosts.
Model strengths are cited hypotheses, not measured rankings. Keep live probes
opt-in; never claim a model's self-description verifies its serving identity.
Propagate RouterUnavailable without turning it into an experimental result.

Colab training uses ai_scientist/remote/ and remote_interpreter.py. Local
postprocessing is optional. Keep exp_name: run because artifact consumers rely
on logs/0-run. Pipeline completion does not establish research validity.

## Verification

Inspect git status and preserve user changes. First check path/model edits with
`.venv/bin/python -m autoresearch --project projects/regularization --dry-run`.
It must not create outputs or make model calls. For engine changes, run
`.venv/bin/python -m unittest discover -s tests -v` and `git diff --check`.
Rebuild the notebook with `.venv/bin/python scripts/build_colab_notebook.py`
after editing its generator or remote modules. Offline tests need no GPU or login.

Do not delete results for tidiness. Preserve LICENSE and NOTICE.md and their
disclosure requirements. Paper claims must match recorded experiments.
Automatic full-pipeline resume is not implemented.
