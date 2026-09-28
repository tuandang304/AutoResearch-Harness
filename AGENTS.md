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

Orchestrator: codex/gpt-6-astra. Worker: codex/gpt-6-sol. Both low effort.
The launcher scopes AUTORESEARCH_CODEX_REASONING_EFFORT to each run and children.
Do not silently substitute models. Provider presets precede explicit role flags.
Routing lives in autoresearch/llm_strategy.py, agent_manager.py and parallel_agent.py.
Workers use agent.code/feedback; final writing/review inherit the orchestrator.

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
