# Model profiles and quota-aware routing

`configs/llm.yaml` is the source of truth for model IDs, effort, instructions and
worker pools. Shared experiment configuration opts in through `llm_config`.
Study files can provide `projects/<study>/llm.yaml` without copying training settings.

## Selection and precedence

Policy path: `--llm-config` > `AUTORESEARCH_LLM_CONFIG` > project `llm.yaml` >
experiment config `llm_config` (repository-relative). Policy bindings precede
`--provider`, then explicit orchestrator/worker overrides; explicit paper flags win.
An explicit concrete model bypasses the pool, including its profile effort and
instructions. Legacy configs without a policy continue to work.

```bash
python -m autoresearch --project projects/regularization --dry-run
python -m autoresearch --project projects/regularization --llm-config configs/llm.yaml
```

Dry runs validate and show the effective policy but make no model calls or state
files. Live runs copy it to `llm_policy.yaml` inside the run directory before any
model calls. Children inherit that snapshot. Editing the source policy affects
the next run, not an in-progress run.

## Profiles and research basis

The following assignments are **our task-fit hypotheses**, not a measured ranking
of these models in this harness. Provider benchmark claims do not establish
performance at the selected effort or with this completion-only CLI interface.

| Profile | Exact ID | Effort | Initial assignment |
|---|---|---|---|
| opus | claude-code/claude-opus-5-5 | high | Primary orchestration, manuscript decisions, final review |
| astra_orchestrator | codex/gpt-6-astra | high | Cross-component reasoning and orchestration fallback |
| antigravity_opus_orchestrator | antigravity/claude-opus-5-5-high | high | Research synthesis and final orchestration fallback |
| sonnet | claude-code/claude-sonnet-5-5 | medium | Scoped coding, iterative fixes and writing |
| sol | codex/gpt-6.1-sol | medium | Reproducible implementation, extraction, summaries and citation assistance |
| antigravity_opus | antigravity/claude-opus-5-5-high | high | Difficult debugging, numerical analysis, visual and evidence review |

The 2026-10-04 policy uses the exact IDs and efforts requested by the user. Task
assignments are engineering hypotheses informed by the earlier
[Anthropic Sonnet guidance](https://www.anthropic.com/claude-sonnet-5-5),
[OpenAI model selection guidance](https://developers.openai.com/api/docs/guides/model-selection)
and [Opus prompting guidance](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5-5).
Those sources do not verify these exact endpoint IDs or compare these effort settings.
Live access probes on 2026-10-04 found two unavailable IDs, which were then
replaced and re-probed successfully (see the results below).
No quality benchmark was run. Historical probes apply only to their recorded IDs
and efforts.

Claude Code profiles share `claude-main`; Codex profiles share `codex-main`;
Antigravity profiles share `antigravity-main`. The two Antigravity Opus roles
share the same model and quota. Separate CLI accounts are assumed to have
independent quotas; change groups if account limits are actually shared.

## How assignment works

For `code` and `plotting`, a separate orchestrator call selects one allowed worker
using the task and profile strengths. The selection must pass JSON/allowlist
validation. It does not execute tools or alter the experiment plan. Images are
not sent to the selector; the worker receives the full original image content.
Long task context is truncated for selection only, not for execution.

The orchestrator role is an ordered `fallback`: every call starts with Opus, moves
to Astra (high) only when Opus is in cooldown or its call fails, and to
Antigravity Opus (high) after Astra. A busy but available profile is waited for rather than skipped, and
there is no stickiness, so Opus is used again as soon as its cooldown ends.
`writeup` and `review` stay pinned to Opus. Each call records the profile that
answered in `routing.jsonl`.

Routine feedback, summaries, citation assistance and vision review use configured
priority pools without paying for another selection call. Switching stays within
each role's capability-filtered pool. A successful fallback is sticky for that
role/preference; an unavailable profile is reconsidered after its cooldown expires.
Changing the policy produces a new stickiness namespace.

Profile instructions accompany every routed call. The orchestrator receives the
worker roster and assignment guidance. Native worker CLI tools are not generally
enabled: source generation is distinct from experiment execution; Antigravity
uses its image-reading tool for attachments. Provider agentic tool benchmarks
therefore cannot be directly transferred to this harness.

## Failure handling and limits

- Pinned roles never silently substitute a model. The orchestrator's ordered
  fallback is explicit policy, not substitution.
- Temporary rate limits honor an exposed `Retry-After`; otherwise use the configured
  cooldown. Explicit account-wide limits block the group; model-local/ambiguous
  rate limits block only the affected profile.
- Quota exhaustion without a known reset blocks the group until explicitly
  rechecked/reset. It does not busy-loop across models sharing the account.
- A reset time printed by the CLI ("try again at 1:30 AM") becomes the cooldown.
- A provider safety-filter refusal ("safeguards flagged this message") moves that
  call to the next candidate without a cooldown; pinned roles still fail.
- Transient errors have bounded attempts. Authentication, configuration, unknown
  errors and serving-model mismatches are surfaced, not disguised as quota limits.
- Output-format repairs remain bounded in the caller. They are not quota events.
- SQLite leases cap concurrency per group across local processes. Expired/dead
  process leases can be reclaimed. This is not a distributed multi-host scheduler.
- With `reserve_orchestrator_capacity: true`, workers sharing the primary
  orchestrator's account are ordered after independent providers. The shipped
  policy sets it to false so task-specific preferences take precedence. Leases
  limit concurrent requests but do not reserve a token budget.
- `max_parallel_subagents` bounds independent routed completion batches (1–16,
  shipped value 3; omitted means 1). Each child routes separately, with copied
  prompts and ordered results. Shared quota leases still apply across threads and
  local processes. On failure queued work is cancelled, active children finish
  cleanup, and the original failure propagates. Concrete overrides remain serial.
- Experiment sub-agents use the existing process pool, with `agent.num_workers`
  defaulting to 3 (`--num-workers` overrides it). Local GPU count can cap this.
  This is separate from completion concurrency; dependent stages stay sequential.

The maximum wait bounds idle routing waits, not CLI execution time. Each worker
completion has an attempt cap; an optional selector call has its own pinned-call
cap. Schema/code-format repair loops add calls, so use stage and worker timeouts
as the outer execution bounds. Exhaustion retains outputs but does not implement
automatic full-pipeline resume.

Inspect or explicitly clear a cooldown after verifying account availability:

```bash
python -m autoresearch.llm_router
python -m autoresearch.llm_router --reset-quota codex-main
```

State lives in ignored `.state/llm-router.sqlite`; override with
`AUTORESEARCH_ROUTER_STATE`. Do not share it between unrelated machines/accounts.

## Credentials and provenance

Keep authenticated CLI sessions in their native credential stores. `.env` is
optional for exported configuration/API keys, is ignored, and is **not** loaded
automatically. Do not put access tokens in YAML or commit `.state`.

`routing.jsonl` records profile, concrete requested model, effort, timestamps,
attempts, coarse failure/switch information, policy hash and successful-call usage.
It does not store raw prompts or raw CLI errors. Serving-model metadata is recorded
when exposed; absent metadata means accepted request, not proven serving identity.
Different reported IDs fail rather than silently substituting.
The ordinary usage ledger attributes successful calls to the selected concrete
model, including selector calls and heterogeneous batches. Failed calls may incur
unreported provider usage. Token counts are not subscription billing records.

## Verification

```bash
# Offline preflight: policy schema, CLI binaries on PATH, role coverage. No model calls.
python scripts/verify_llm_profiles.py [--llm-config projects/<study>/llm.yaml]
# Explicit live tests: provider usage is incurred; no automatic model fallback.
python scripts/verify_llm_profiles.py --live --vision --output .state/model-verification.json
# Add/retest one model without repeating the entire pool:
python scripts/verify_llm_profiles.py --live --vision --model claude-code/claude-sonnet-5-5 --output .state/model-verification.json
# Offline tests never authenticate or call providers.
python -m unittest discover -s tests -v
```

Live runs print JSON on stdout and a per-model and per-role summary on stderr.
A role is `ok` when every capable candidate passed, `degraded` when only some did,
`unavailable` when none did and `untested` when candidates were not probed (for
example, the vision role without `--vision`). The JSON adds `role_coverage`.

Text/JSON/image probes test access and interface correctness only. They do not
measure scientific reasoning quality, model ranking, or full-pipeline completion.
Preserve the sanitized results with the exact CLI versions and test date when
using the harness in a paper. Repeat after CLI/account/model changes.

## Live verification on 2026-10-04

Explicit live probes used Claude Code 2.1.289 and Codex CLI 0.160.0. Each unique
model/effort pair received text, JSON and generated-image checks. The two
Antigravity Opus profiles use an identical pair and were probed once.

| Requested model | Effort | Text | JSON | Image |
|---|---|---|---|---|
| claude-code/claude-opus-5-5 | high | Pass | Pass | Pass on retry |
| codex/gpt-6-astra | high | Pass | Pass | Pass |
| codex/gpt-6.1-sol | medium | Pass | Pass | Pass |
| claude-code/sonnet-5-5 | medium | Unavailable | Unavailable | Unavailable |
| antigravity/claude-opus-5.5 | high | Unavailable | Unavailable | Unavailable |

Claude Code Opus initially timed out on the image probe at 120 seconds; a targeted
rerun passed all three checks. Claude Code metadata matched its requested model.
Codex accepted the requests but exposed no serving-model metadata.

The configuration was **not fully live-validated** at that point: Sonnet returned `model_not_found`,
and Antigravity rejected the dotted Opus ID. The user's exact requested IDs remain
in the policy. `agy models` listed `claude-opus-5-5-high` for high-effort Opus;
that alternative was not substituted or live-probed. Configuration errors are
terminal in the router, so reaching either unavailable profile can stop a run.

A real two-child `router/summary` completion batch also passed, with both children
using `codex/gpt-6.1-sol` and returning valid JSON. The 166-test offline suite passed.
The documented regularization project was absent; the test project dry run with
`configs/default.yaml` passed instead. These checks establish interface behavior,
not research quality or full-pipeline completion.

Ignored local evidence: `.state/model-verification-2026-10-04.json` and
`.state/live-routing-2026-10-04/`. No credentials or generated research outputs
are included in this documentation update.

### Model ID update on 2026-10-04

At the user's request the failing IDs were replaced: `sonnet` now uses
`claude-code/claude-sonnet-5-5` and both Antigravity Opus profiles use
`antigravity/claude-opus-5-5-high` (the ID listed by `agy models`); efforts are
unchanged. Targeted live probes (Claude Code 2.1.289, Antigravity CLI 1.2.11):

| Requested model | Effort | Text | JSON | Image | Identity evidence |
|---|---|---|---|---|---|
| claude-code/claude-sonnet-5-5 | medium | Pass | Pass | Pass | Metadata matched |
| antigravity/claude-opus-5-5-high | high | Pass | Pass | Pass | Request accepted only |

With the earlier passes for Opus, Astra and Sol, every configured model/effort
pair has now passed access probes. This is still interface verification, not a
quality benchmark. Ignored evidence: `.state/model-verification-ids-*.json`.

## Historical verification and policy changes

The following entries describe previous policies, not the current defaults.

### Local verification on 2026-09-29

Installed clients checked: Claude Code 2.1.284, Codex CLI 0.158.0,
Antigravity CLI snap 1.2.11 (revision 23).

All seven configured model/effort profiles passed text, JSON and simple image
checks (21 checks): Opus medium/low, Sonnet low, Astra/Sol/Luna low, Flash high.
Opus initially reported a weekly quota rejection; after the indicated 13:00
Asia/Ho_Chi_Minh reset, a targeted rerun passed. Claude responses exposed matching
model metadata. Codex and Antigravity accepted the exact requested IDs but did
not expose serving-model metadata in these probes.

A separate routed call passed: Opus selected Luna for a bounded extraction,
the worker returned the expected JSON, and both models were attributed separately.

### Claude-first worker routing on 2026-10-01

Sonnet is now first in every worker pool, and the orchestrator is told to prefer it.
Opus workers stay last because they use the Opus quota that the pinned write-up and
review roles need. `reserve_orchestrator_capacity` is false, since it would otherwise
move Sonnet behind the other providers. `max_concurrency_per_quota_group` is 2,
so an orchestrator call can run beside one Sonnet worker call; this applies to every
quota group. More work on `claude-main` raises the chance of hitting its weekly limit.
When it is hit, the orchestrator falls back to Astra, workers fall back to Sol,
Flash and Astra, and the pinned write-up and review stop with `RouterUnavailable`
(outputs retained) until the limit resets.
Router tests pin their own pool order so routing mechanics do not depend on this preference.

### Profile update on 2026-10-01

The luna profile (`gpt-6-luna`) was removed. Its feedback, summary, writing and
code slots now start with Sol. The orchestrator gained two fallbacks: `astra_orchestrator`
(`gpt-6-astra`, medium, with orchestrator instructions) and `argon` (`gemini-4-argon`
via Antigravity, medium). Argon is unreleased: no live probe has accepted it, no
strengths are documented, and it declares only text/JSON until a vision probe
passes. If it is reached before release, its call fails as a configuration or
unknown error and the router raises `RouterUnavailable`. Its effort and capabilities
are placeholders to revisit after release. Verify it with
`scripts/verify_llm_profiles.py --live --model antigravity/gemini-4-argon` once it is available.
Astra's fallback shares `codex-main` with the GPT workers, so
`reserve_orchestrator_capacity` protects only the Opus account.

### Sol profile update on 2026-09-30

The sol profile now uses `gpt-6.1-sol` (low effort). Codex CLI 0.158.0 rejected
this ID for ChatGPT-account logins; Codex CLI 0.159.2 accepts it. The text, JSON and
image checks passed (`.state/model-verification-gpt-6.1-sol.json`). As with the other
Codex models, the probes show only that the request was accepted, not which model
served it. Run snapshots made before this date still record `gpt-6-sol`.
This is interface verification, not a full research run or a quality benchmark.
Sanitized local evidence is in `.state/model-verification.json`; run-specific
routing evidence is under `.state/routing-check-*/`. These artifacts are ignored
by Git and are not credentials.
