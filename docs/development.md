# Development

The public entry point is `python -m autoresearch`. The legacy launcher filename
and `ai_scientist` package remain compatible with existing scripts.

## Components

| Path | Responsibility |
|---|---|
| `autoresearch/config.py` | Validate ideas/config and resolve provider overrides |
| `autoresearch/llm_router.py` | Profile validation, task assignment, shared quota state and audit |
| `configs/llm.yaml` | Exact model IDs, effort, capability declarations and task instructions |
| `autoresearch/llm_strategy.py` | Orchestrator/worker role assignments |
| `configs/default.yaml` | Topic-independent runtime defaults |
| `projects/` | Independent study inputs and ignored run artifacts |
| `launch_scientist_bfts.py` | Orchestrate one run, persist status, clean up children |
| `ai_scientist/cli_llm.py` | CLI text/vision completions, process lifecycle and retries |
| `ai_scientist/treesearch/backend/` | Tree-search model routing and structured responses |
| `ai_scientist/treesearch/` | Experiment search, journals, local/remote interpreters |
| `ai_scientist/remote/` | HTTP executor and bounded workspace-transfer primitives |
| `autoresearch/colab_runtime.py` | Colab CLI tier pool: model-selected compute, provisioning, watchdog, release |
| `ai_scientist/utils/token_tracker.py` | SDK usage normalization and worker ledger |
| `scripts/build_colab_notebook.py` | Deterministic notebook with both executor modules |
| `tests/` | Offline regressions and localhost integration tests |

## Verification

Run `python -m unittest discover -s tests -v` from the repository root. Tests
exercise real subprocess timeouts and a localhost HTTP executor, but mock model
responses and paper-generation calls. They cover batch-history accounting,
optional SDK usage fields, cross-process usage, dry runs, PDF selection, config
errors, normal write-up dispatch, failure status, journal serialization,
archive traversal/link rejection, authentication, duplicate submissions,
workspace synchronization, exceptions, timeouts and checkpoint retention.

Rebuild the notebook after changing either remote module:

```bash
python scripts/build_colab_notebook.py
git diff --check
```

The notebook stores no outputs or credentials. Its code cells embed the server
and workspace helper so a fresh Colab session does not need to clone the repo.
GitHub Actions checks the suite on Python 3.11 and 3.12 and rejects stale
notebook embeddings.

## Current limitations

- A live Colab GPU run and a complete paper-generation run are not covered by
  the offline suite. Authenticated CLI availability is account/version dependent.
  Colab CLI provisioning is tested offline against `tests/fixtures/fake_colab.py`;
  its bootstrap and tier routing were checked live on CPU and T4 runtimes with
  CLI 0.7.4. L4/A100 routing, OOM re-runs and replica scale-out are covered
  offline only.
- The execution service and local interpreter run arbitrary generated code with
  user permissions. Workspace validation does not isolate that code.
- Full pipeline resume after controller shutdown is not implemented. Journals,
  configuration and status are retained for inspection.
- Individual files above the transfer limit stay on the remote runtime; runtime
  loss destroys them unless downloaded separately.
- Token totals cover recorded calls across processes. Interaction details remain
  process-local; CLI billing cannot be inferred from token counts alone.
- API model support differs between the text, vision and tree-search layers.
  These adapters have not all been migrated to a single provider interface.
  Profile routing currently targets the three CLI adapters, not API backends.
- The LaTeX templates and summary consumers expect the standard `run` experiment
  name. Use `--output-dir` for an alternate output location.

## Study inputs and search fixes

Stages 2 and 4 now see the research idea and experiment plan when proposing and
implementing tuning and ablations. Projects can override stage goals, ship support
files and restrict seed evaluation (see [project workflow](projects.md)). Sub-stages
keep their main stage number, receive the carried-over best node and share the
stage's node budget; summaries take one journal per main stage. Nodes that save no
`.npy` data are debugged instead of stalling. Failed plotting code is retried. The
best-node prompt includes analysis and plot feedback. The remote runner executes
scripts as a real `__main__` module, so process pools can pickle their functions.
A Colab status call that fails (for example a DNS error) no longer provisions a
replacement VM; the job waits and reconnects to the existing one.

Seed results now reach the paper. The research summary takes the node that was
seed-evaluated, not a fresh best-node choice that can pick a sibling. Plotting and
writing receive the per-seed results and the seed aggregate, without repeating the
code. A seed rerun of accepted code is marked buggy only when it crashes or its data
or metrics fail to parse, not when the reviewer dislikes the result (for example a
failed pre-registered guard); the review prompt says such failures are outcomes.
The plot aggregator must save PNG figures, the only format the paper writer uses. Without a Semantic Scholar key (`S2_API_KEY`), citations come from OpenAlex,
because unauthenticated Semantic Scholar searches are mostly rate limited;
`OPENALEX_MAILTO` optionally adds a contact address to OpenAlex requests.

The orchestrator role may use an ordered `fallback` selection (Opus, Astra, Argon in
`configs/llm.yaml`). It is not sticky, does not skip a busy profile, and cannot use
`orchestrator` selection itself.

## Changes in this development update

See [model routing](llm-routing.md) for profile selection, live verification,
model-strength sources, bounded failover and actual-model usage attribution.

Project selection resolves inputs and output directories before execution. See
[project workflow](projects.md) for precedence and historical migration details.
Project config files replace defaults rather than overlaying them. Dry runs must
remain side-effect free. Test fixtures belong under tests/fixtures, not projects/.

The user-facing documentation and search-tree title use AutoResearch-Harness.
Historical provenance is separate from product documentation. The launcher now
validates inputs, offers an offline dry run, supports output/config paths,
records terminal status, handles missing PDFs, and dispatches the eight-page
writer with supported arguments. Cleanup is limited to newly created children.

CLI streaming timeouts include blocked input pipes. Remote jobs have idempotent
submission, input limits, restricted archive extraction and explicit artifact
retention. Seed workers receive independent inputs, and restoring a journal
node no longer modifies its source dictionary. Shared usage accounting and
regression tests make these behaviors reviewable.
