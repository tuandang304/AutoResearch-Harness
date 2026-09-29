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
| opus | claude-opus-5-5 | medium | Orchestration, manuscript decisions and final review |
| opus_worker | claude-opus-5-5 | low | Last-resort bounded worker |
| astra | gpt-6-astra | low | Integration and difficult debugging |
| sol | gpt-6-sol | low | Routine implementation and writing |
| luna | gpt-6-luna | low | Extraction, small edits, summaries |
| flash | gemini-3.8-flash | high | Numerical/visual analysis and multi-step work |

Research checked 2026-09-29:

- [OpenAI model selection](https://developers.openai.com/api/docs/guides/model-selection)
  distinguishes efficient scoped work, general-purpose judgment, and broader
  demanding tasks. Low effort is not evidence of top benchmark performance.
- [Anthropic Opus prompting guidance](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5-5)
  supports medium as a starting point and emphasizes concrete completion criteria.
- [Google Flash guidance](https://ai.google.dev/gemini-api/docs/latest-model?hl=en)
  describes high thinking for complex reasoning and multi-step tasks; it can use
  more tokens, so high effort should not be assumed cheapest.

Exact IDs are explicit; `opus`, `default`, and display-name guesses are not used
in profiles. All GPT profiles conservatively share `codex-main`, and both Opus
profiles share `claude-main`. These groupings are local account assumptions,
not a claim about every provider's limit structure. Adjust only after confirming
which credentials and limits are independent.

## How assignment works

For `code` and `plotting`, a separate pinned Opus call selects one allowed worker
using the task and profile strengths. The selection must pass JSON/allowlist
validation. It does not execute tools or alter the experiment plan. Images are
not sent to the selector; the worker receives the full original image content.
Long task context is truncated for selection only, not for execution.

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

- Pinned roles never silently substitute a model.
- Temporary rate limits honor an exposed `Retry-After`; otherwise use the configured
  cooldown. Explicit account-wide limits block the group; model-local/ambiguous
  rate limits block only the affected profile.
- Quota exhaustion without a known reset blocks the group until explicitly
  rechecked/reset. It does not busy-loop across models sharing the account.
- Transient errors have bounded attempts. Authentication, configuration, unknown
  errors and serving-model mismatches are surfaced, not disguised as quota limits.
- Output-format repairs remain bounded in the caller. They are not quota events.
- SQLite leases cap concurrency per group across local processes. Expired/dead
  process leases can be reclaimed. This is not a distributed multi-host scheduler.
- Opus workers are ordered last to reduce competition with orchestration. This
  is not a guaranteed token reservation: the CLI exposes no reliable remaining
  quota budget, and an already-running worker cannot be preempted safely.

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
# Explicit live tests: provider usage is incurred; no automatic model fallback.
python scripts/verify_llm_profiles.py --live
# Offline tests never authenticate or call providers.
python -m unittest discover -s tests -v
```

Text/JSON/image probes test access and interface correctness only. They do not
measure scientific reasoning quality, model ranking, or full-pipeline completion.
Preserve the sanitized results with the exact CLI versions and test date when
using the harness in a paper. Repeat after CLI/account/model changes.
