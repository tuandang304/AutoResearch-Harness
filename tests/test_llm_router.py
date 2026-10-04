"""Offline routing regressions: real policy/state, mocked concrete CLI calls."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from ai_scientist.cli_llm import CLIError
from autoresearch.config import ROOT, load_run_config
from autoresearch.llm_router import (
    REQUIRED_ROLES, RouterUnavailable, State, classify_error, complete_routed,
    load_policy, retry_after,
)
from launch_scientist_bfts import main


class RouterTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.path = self.directory / "llm.yaml"
        self.db = self.directory / "state" / "router.sqlite"
        self.log = self.directory / "routing.jsonl"
        self.usage_log = self.directory / "usage.jsonl"
        # Freeze the richer legacy topology to exercise shared models/accounts and
        # fallback mechanics independently of changes to the production roster.
        self.policy = load_policy(ROOT / "tests/fixtures/llm_router.yaml")
        workers = ["sol", "flash", "astra", "sonnet", "opus_worker"]
        for role in ("feedback", "summary", "citation", "writing", "plotting"):
            self.policy["roles"][role]["candidates"] = list(workers)
        self.policy["roles"]["code"]["candidates"] = ["sol", "astra", "flash", "sonnet", "opus_worker"]
        self.policy["roles"]["vision"]["candidates"] = ["flash", "sol", "astra", "sonnet", "opus_worker"]
        self.policy["routing"].update(max_wait_seconds=0, transient_cooldown_seconds=1,
                                      max_concurrency_per_quota_group=1, reserve_orchestrator_capacity=True)
        self.save_policy()
        env = patch.dict(os.environ, {
            "AUTORESEARCH_LLM_CONFIG": str(self.path),
            "AUTORESEARCH_ROUTER_STATE": str(self.db),
            "AUTORESEARCH_ROUTING_LOG": str(self.log),
            "AUTORESEARCH_USAGE_LOG": str(self.usage_log),
        })
        env.start()
        self.addCleanup(env.stop)
        from ai_scientist.utils.token_tracker import TokenTracker
        self.tracker = TokenTracker()
        tracker = patch("ai_scientist.utils.token_tracker.token_tracker", self.tracker)
        tracker.start()
        self.addCleanup(tracker.stop)
        concrete = patch("ai_scientist.cli_llm.complete", return_value=("done", {"prompt": 2, "completion": 3}))
        self.complete = concrete.start()
        self.addCleanup(concrete.stop)
        sleep = patch("autoresearch.llm_router.time.sleep", side_effect=AssertionError("zero-wait routing must not sleep"))
        sleep.start()
        self.addCleanup(sleep.stop)

    def save_policy(self):
        self.path.write_text(yaml.safe_dump(self.policy))

    def route(self, role="feedback"):
        return complete_routed("router/" + role, [{"role": "user", "content": "bounded offline task"}])

    def models(self):
        return [call.args[0] for call in self.complete.call_args_list]

    def test_state_shared_leases_release_expiry_and_group_isolation(self):
        first, second = State(self.db), State(self.db)
        with patch("autoresearch.llm_router.time.time", return_value=100):
            lease = first.acquire("sol", "codex", 1, 10)
            self.assertIsNotNone(lease)
            self.assertIsNone(second.acquire("astra", "codex", 1, 10))
            other = second.acquire("flash", "google", 1, 10)
            self.assertIsNotNone(other)
            first.release(lease)
            replacement = second.acquire("astra", "codex", 1, 10)
            self.assertIsNotNone(replacement)
        with patch("autoresearch.llm_router.time.time", return_value=111):
            self.assertIsNotNone(first.acquire("sol", "codex", 1, 10))
            self.assertIsNotNone(first.acquire("flash", "google", 1, 10))

    def test_state_shared_cooldowns_and_persistent_quota_blocks(self):
        first, second = State(self.db), State(self.db)
        with patch("autoresearch.llm_router.time.time", return_value=100):
            first.block("profile:sol", 5, "rate_limit")
            self.assertIsNone(second.acquire("sol", "codex", 1, 10))
            lease = second.acquire("astra", "codex", 1, 10)
            self.assertIsNotNone(lease)
            second.release(lease)
            first.block("quota:codex", None, "quota")
        with patch("autoresearch.llm_router.time.time", return_value=100000):
            self.assertIsNone(second.acquire("sol", "codex", 1, 10))
            self.assertIsNone(second.acquire("astra", "codex", 1, 10))
            self.assertIsNotNone(second.acquire("sol", "different-account", 1, 10))

    def test_state_recovers_dead_process_lease(self):
        state = State(self.db)
        state.acquire("sol", "codex", 1, 100)
        with patch("autoresearch.llm_router.os.kill", side_effect=ProcessLookupError):
            self.assertIsNotNone(State(self.db).acquire("astra", "codex", 1, 100))

    def test_cooldown_cannot_be_shortened_by_concurrent_failure(self):
        state = State(self.db)
        with patch("autoresearch.llm_router.time.time", return_value=100):
            state.block("quota:claude-main", 300, "quota")
            state.block("quota:claude-main", 10, "rate_limit")
        with patch("autoresearch.llm_router.time.time", return_value=200):
            self.assertFalse(state.available("sonnet", "claude-main"))
            state.block("quota:claude-main", None, "quota")
            state.block("quota:claude-main", 10, "rate_limit")
        with patch("autoresearch.llm_router.time.time", return_value=10000):
            self.assertFalse(state.available("opus", "claude-main"))

    def test_claude_known_weekly_reset_is_honored(self):
        error = CLIError("provider rejected request", metadata={
            "error_code": "rate_limit",
            "rate_limit": {"status": "rejected", "rateLimitType": "seven_day", "resetsAt": "400"},
        })
        self.assertEqual(classify_error(error), ("quota", True))
        with patch("autoresearch.llm_router.time.time", return_value=100):
            self.assertEqual(retry_after(error), 300)
            self.complete.side_effect = error
            with self.assertRaises(RouterUnavailable):
                self.route("orchestrator")
        with patch("autoresearch.llm_router.time.time", return_value=399):
            self.assertFalse(State(self.db).available("sonnet", "claude-main"))
        with patch("autoresearch.llm_router.time.time", return_value=401):
            self.assertTrue(State(self.db).available("sonnet", "claude-main"))

    def test_production_policy_has_exact_models_efforts_and_task_assignments(self):
        policy = load_policy(ROOT / "configs/llm.yaml")
        expected = {
            "opus": ("claude-code", "claude-opus-5-5", "high"),
            "astra_orchestrator": ("codex", "gpt-6-astra", "high"),
            "antigravity_opus_orchestrator": ("antigravity", "claude-opus-5-5-high", "high"),
            "sonnet": ("claude-code", "claude-sonnet-5-5", "medium"),
            "sol": ("codex", "gpt-6.1-sol", "medium"),
            "antigravity_opus": ("antigravity", "claude-opus-5-5-high", "high"),
        }
        self.assertEqual({name: tuple(profile[key] for key in ("provider", "model", "effort"))
                          for name, profile in policy["profiles"].items()}, expected)
        self.assertFalse(policy["routing"]["reserve_orchestrator_capacity"])
        self.assertEqual(policy["routing"]["max_parallel_subagents"], 3)
        assignments = {
            "orchestrator": ("fallback", ["opus", "astra_orchestrator", "antigravity_opus_orchestrator"]),
            "writeup": ("pinned", ["opus"]),
            "review": ("pinned", ["opus"]),
            "code": ("orchestrator", ["sonnet", "sol", "antigravity_opus"]),
            "plotting": ("orchestrator", ["sonnet", "sol", "antigravity_opus"]),
            "feedback": ("sticky_priority", ["antigravity_opus", "sonnet", "sol"]),
            "vision": ("sticky_priority", ["antigravity_opus", "sonnet", "sol"]),
            "summary": ("sticky_priority", ["sol", "sonnet", "antigravity_opus"]),
            "citation": ("sticky_priority", ["sol", "sonnet", "antigravity_opus"]),
            "writing": ("sticky_priority", ["sonnet", "sol", "antigravity_opus"]),
        }
        self.assertEqual({role: (settings["selection"], settings["candidates"])
                          for role, settings in policy["roles"].items()}, assignments)

    def test_production_routing_dispatches_task_specific_models_and_efforts(self):
        self.policy = load_policy(ROOT / "configs/llm.yaml")
        self.policy["routing"]["max_wait_seconds"] = 0
        self.save_policy()
        for role, model, effort in (
            ("feedback", "antigravity/claude-opus-5-5-high", "high"),
            ("summary", "codex/gpt-6.1-sol", "medium"),
            ("writing", "claude-code/claude-sonnet-5-5", "medium"),
        ):
            with self.subTest(role=role):
                self.complete.reset_mock()
                self.route(role)
                self.assertEqual(self.models(), [model])
                self.assertEqual(self.complete.call_args.kwargs["effort"], effort)

    def test_production_orchestrator_uses_high_effort_fallbacks(self):
        self.policy = load_policy(ROOT / "configs/llm.yaml")
        self.policy["routing"]["max_wait_seconds"] = 0
        self.save_policy()
        self.complete.side_effect = [CLIError("quota exhausted"), CLIError("quota exhausted"), ("done", {})]
        self.assertEqual(self.route("orchestrator")[0], "done")
        self.assertEqual(self.models(), ["claude-code/claude-opus-5-5", "codex/gpt-6-astra",
                                        "antigravity/claude-opus-5-5-high"])
        self.assertEqual([call.kwargs["effort"] for call in self.complete.call_args_list], ["high"] * 3)

    def test_production_selector_dispatches_requested_worker_and_effort(self):
        self.policy = load_policy(ROOT / "configs/llm.yaml")
        self.policy["routing"]["max_wait_seconds"] = 0
        self.save_policy()
        self.complete.side_effect = [
            (json.dumps({"profile": "antigravity_opus", "reason": "complex numerical analysis"}), {}),
            ("done", {}),
        ]
        self.assertEqual(self.route("code")[0], "done")
        self.assertEqual(self.models(), ["claude-code/claude-opus-5-5", "antigravity/claude-opus-5-5-high"])
        self.assertEqual([call.kwargs["effort"] for call in self.complete.call_args_list], ["high", "high"])

    def test_production_antigravity_aliases_share_limits_but_claude_code_is_independent(self):
        self.policy = load_policy(ROOT / "configs/llm.yaml")
        self.policy["routing"]["max_wait_seconds"] = 0
        self.save_policy()
        self.complete.side_effect = [CLIError("429 model rate limit"), ("done", {})]
        self.assertEqual(self.route("feedback")[0], "done")
        self.assertEqual(self.models(), ["antigravity/claude-opus-5-5-high", "claude-code/claude-sonnet-5-5"])
        state = State(self.db)
        for name in ("antigravity_opus", "antigravity_opus_orchestrator"):
            self.assertFalse(state.available(name, self.policy["profiles"][name]["quota_group"]))
        self.assertTrue(state.available("opus", self.policy["profiles"]["opus"]["quota_group"]))

    def test_parallel_subagent_limit_is_optional_bounded_integer_without_state(self):
        self.policy["routing"].pop("max_parallel_subagents", None)
        self.save_policy()
        self.assertEqual(load_policy(self.path)["routing"].get("max_parallel_subagents", 1), 1)
        for value in (1, 3, 16):
            with self.subTest(value=value):
                self.policy["routing"]["max_parallel_subagents"] = value
                self.save_policy()
                self.assertEqual(load_policy(self.path)["routing"]["max_parallel_subagents"], value)
        for value in (0, 17, -1, True, 2.5, "3", None):
            with self.subTest(value=value):
                self.policy["routing"]["max_parallel_subagents"] = value
                self.save_policy()
                with self.assertRaisesRegex(ValueError, "max_parallel_subagents"):
                    load_policy(self.path)
        self.assertFalse(self.db.exists())
        self.complete.assert_not_called()

    def test_sonnet_is_available_in_worker_pools_not_pinned_roles(self):
        sonnet = self.policy["profiles"]["sonnet"]
        self.assertEqual((sonnet["model"], sonnet["effort"]), ("claude-sonnet-5-5", "low"))
        self.assertEqual(sonnet["quota_group"], self.policy["profiles"]["opus"]["quota_group"])
        for role, settings in self.policy["roles"].items():
            if settings["selection"] == "pinned" or role == "orchestrator":
                self.assertNotIn("sonnet", settings["candidates"])
            else:
                self.assertIn("sonnet", settings["candidates"])

    def test_different_effort_profiles_share_model_local_limit(self):
        self.complete.side_effect = CLIError("429 model rate limit")
        with self.assertRaises(RouterUnavailable):
            self.route("orchestrator")
        state = State(self.db)
        self.assertFalse(state.available("opus", "claude-main"))
        self.assertFalse(state.available("opus_worker", "claude-main"))
        self.assertTrue(state.available("sonnet", "claude-main"))

    def test_error_classification(self):
        cases = {
            "401 unauthorized": ("authentication", True),
            "login required": ("authentication", True),
            "insufficient_quota": ("quota", True),
            "weekly limit reached": ("quota", True),
            "429 rate limit": ("rate_limit", False),
            "429 organization rate limit": ("rate_limit", True),
            "unsupported effort": ("configuration", False),
            "unknown model": ("configuration", False),
            "not found on PATH": ("configuration", False),
            "connection timed out": ("transient", False),
            "503 overloaded": ("transient", False),
            "unexpected response": ("unknown", False),
            "API Error: Sonnet 5.5's safeguards flagged this message": ("refusal", False),
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(classify_error(CLIError(message)), expected)
        self.assertEqual(retry_after(CLIError("429 Retry-After: 12")), 12)
        self.assertIsNone(retry_after(CLIError("weekly limit reached")))
        codex = retry_after(CLIError("You've hit your usage limit. ... or try again at 1:30 AM."))
        self.assertTrue(0 < codex <= 24 * 3600 + 60)
        self.assertEqual(classify_error(CLIError("You've hit your usage limit.")), ("quota", True))

    def test_safety_refusal_falls_back_without_cooldown(self):
        self.complete.side_effect = [CLIError("API Error: safeguards flagged this message"), ("fallback", {})]
        text, _ = self.route()
        self.assertEqual(text, "fallback")
        self.assertEqual(self.models(), ["codex/gpt-6.1-sol", "antigravity/gemini-3.8-flash"])
        self.assertTrue(State(self.db).available("sol", "codex-main"))

    def test_exact_profile_and_effort_and_lease_release(self):
        text, usage = self.route()
        self.assertEqual(text, "done")
        self.assertEqual(self.models(), ["codex/gpt-6.1-sol"])
        self.assertEqual(self.complete.call_args.kwargs, {"effort": "low", "max_attempts": 1})
        self.assertEqual(usage["model"], "codex/gpt-6.1-sol")
        self.assertEqual(usage["profile"], "sol")
        self.assertEqual(usage["effort"], "low")
        state = State(self.db)
        self.assertIsNotNone(state.acquire("sol", "codex-main", 1, 10))

    def test_profile_rate_limit_falls_back_without_blocking_its_account(self):
        self.complete.side_effect = [CLIError("429 Retry-After: 30"), ("fallback", {})]
        self.assertEqual(self.route()[0], "fallback")
        self.assertEqual(self.models(), ["codex/gpt-6.1-sol", "antigravity/gemini-3.8-flash"])
        self.assertIsNone(State(self.db).acquire("sol", "codex-main", 1, 10))

        self.assertTrue(State(self.db).available("astra", "codex-main"))
    def test_shared_quota_skips_all_profiles_in_account_and_is_sticky(self):
        self.complete.side_effect = [CLIError("quota exhausted"), ("fallback", {}), ("again", {})]
        self.assertEqual(self.route()[0], "fallback")
        self.assertEqual(self.route()[0], "again")
        self.assertEqual(self.models(), ["codex/gpt-6.1-sol", "antigravity/gemini-3.8-flash", "antigravity/gemini-3.8-flash"])
        self.assertEqual(self.complete.call_args.kwargs["effort"], "high")
        state = State(self.db)
        for profile in ("sol", "astra"):
            self.assertIsNone(state.acquire(profile, "codex-main", 1, 10))

    def test_shared_rate_limit_falls_back_across_accounts(self):
        self.complete.side_effect = [CLIError("429 account rate limit Retry-After: 30"), ("fallback", {})]
        self.route()
        self.assertEqual(self.models(), ["codex/gpt-6.1-sol", "antigravity/gemini-3.8-flash"])

    def test_transient_failure_falls_back_without_waiting(self):
        self.complete.side_effect = [CLIError("connection timeout"), ("fallback", {})]
        self.route()
        self.assertEqual(self.models(), ["codex/gpt-6.1-sol", "antigravity/gemini-3.8-flash"])

    def test_auth_configuration_and_unknown_errors_do_not_substitute(self):
        for message in ("authentication failed", "unknown model", "unexpected response"):
            with self.subTest(message=message):
                self.complete.reset_mock()
                self.complete.side_effect = CLIError(message)
                with self.assertRaises(RouterUnavailable):
                    self.route()
                self.assertEqual(self.models(), ["codex/gpt-6.1-sol"])
                state = State(self.db)
                lease = state.acquire("sol", "codex-main", 1, 10)
                self.assertIsNotNone(lease)
                state.release(lease)

    def test_attempt_budget_is_enforced(self):
        self.policy["routing"]["max_total_attempts_per_call"] = 2
        self.save_policy()
        self.complete.side_effect = CLIError("503 overloaded")
        with self.assertRaises(RouterUnavailable):
            self.route()
        self.assertEqual(self.models(), ["codex/gpt-6.1-sol", "antigravity/gemini-3.8-flash"])

    def test_pinned_roles_never_substitute_when_quota_exhausted(self):
        self.complete.side_effect = CLIError("quota exhausted")
        for role in ("writeup", "review"):
            with self.subTest(role=role), self.assertRaises(RouterUnavailable):
                self.route(role)
        self.assertEqual(self.models(), ["claude-code/claude-opus-5-5"])

    def test_orchestrator_falls_back_in_order_and_returns_to_opus(self):
        self.complete.side_effect = [CLIError("quota exhausted"), ("astra", {}), CLIError("quota exhausted"), ("argon", {})]
        self.assertEqual(self.route("orchestrator")[0], "astra")
        self.assertEqual(self.models(), ["claude-code/claude-opus-5-5", "codex/gpt-6-astra"])
        self.assertEqual([call.kwargs["effort"] for call in self.complete.call_args_list], ["medium", "medium"])
        self.assertEqual(self.route("orchestrator")[0], "argon")
        self.assertEqual(self.models()[2:], ["codex/gpt-6-astra", "antigravity/gemini-4-argon"])
        # Not sticky: once Opus is available again it is tried first.
        with State(self.db).db() as db:
            db.execute("DELETE FROM cooldowns")
        self.complete.reset_mock()
        self.complete.side_effect = None
        self.route("orchestrator")
        self.assertEqual(self.models(), ["claude-code/claude-opus-5-5"])

    def test_orchestrator_waits_for_busy_opus_instead_of_falling_back(self):
        lease = State(self.db).acquire("sonnet", "claude-main", 1, 100)
        self.assertIsNotNone(lease)
        with self.assertRaises(RouterUnavailable):
            self.route("orchestrator")
        self.complete.assert_not_called()

    def test_orchestrator_cannot_be_orchestrator_selected(self):
        self.policy["roles"]["orchestrator"]["selection"] = "orchestrator"
        self.save_policy()
        with self.assertRaisesRegex(ValueError, "Orchestrator must be"):
            load_policy(self.path)

    def test_selector_can_only_choose_allowed_profile(self):
        for decision in ({"profile": "opus", "reason": "not a worker"},
                         {"profile": "sol", "reason": ""}, {"profile": "invented", "reason": "x"}):
            with self.subTest(decision=decision):
                self.complete.reset_mock()
                self.complete.return_value = (json.dumps(decision), {})
                with self.assertRaisesRegex(RouterUnavailable, "allowlist"):
                    self.route("code")
                self.assertEqual(self.models(), ["claude-code/claude-opus-5-5"])

    def test_selector_valid_assignment_dispatches_exact_worker(self):
        self.complete.side_effect = [(json.dumps({"profile": "astra", "reason": "integration"}), {}), ("done", {})]
        self.assertEqual(self.route("code")[0], "done")
        self.assertEqual(self.models(), ["claude-code/claude-opus-5-5", "codex/gpt-6-astra"])
        self.assertEqual([call.kwargs["effort"] for call in self.complete.call_args_list], ["medium", "low"])

    def test_reported_model_accepts_bare_and_qualified_exact_identity(self):
        for reported in ("gpt-6.1-sol", "codex/gpt-6.1-sol"):
            with self.subTest(reported=reported):
                self.complete.reset_mock()
                self.complete.return_value = ("verified", {"reported_model": reported})
                text, usage = self.route()
                self.assertEqual(text, "verified")
                self.assertEqual(usage["model"], "codex/gpt-6.1-sol")
                self.assertEqual(usage["reported_model"], reported)
                self.assertEqual(self.models(), ["codex/gpt-6.1-sol"])

    def test_reported_model_mismatch_fails_without_substitution_and_releases_lease(self):
        for reported in ("gpt-6-sol", "codex/gpt-6-sol", "antigravity/gpt-6.1-sol"):
            with self.subTest(reported=reported):
                self.complete.reset_mock()
                self.complete.return_value = ("must not escape", {"reported_model": reported})
                with self.assertRaisesRegex(RouterUnavailable, "configuration"):
                    self.route()
                self.assertEqual(self.models(), ["codex/gpt-6.1-sol"])
                state = State(self.db)
                lease = state.acquire("sol", "codex-main", 1, 10)
                self.assertIsNotNone(lease)
                state.release(lease)
        events = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual([event["event"] for event in events], ["failed"] * 3)
        self.assertTrue(all(event["error_class"] == "configuration" for event in events))

    def test_available_observes_shared_and_profile_cooldown_expiry(self):
        state, reader = State(self.db), State(self.db)
        with patch("autoresearch.llm_router.time.time", return_value=100):
            state.block("profile:sol", 5, "rate_limit")
            self.assertFalse(reader.available("sol", "codex-main"))
            self.assertTrue(reader.available("astra", "codex-main"))
            state.block("quota:google-main", None, "quota")
            self.assertFalse(reader.available("flash", "google-main"))
        with patch("autoresearch.llm_router.time.time", return_value=106):
            self.assertTrue(reader.available("sol", "codex-main"))
            self.assertFalse(reader.available("flash", "google-main"))

    def test_selector_allowlist_excludes_cooled_down_profiles(self):
        State(self.db).block("quota:codex-main", None, "quota")
        self.complete.return_value = (json.dumps({"profile": "sol", "reason": "ignore cooldown"}), {})
        with self.assertRaisesRegex(RouterUnavailable, "allowlist"):
            self.route("code")
        self.assertEqual(self.models(), ["claude-code/claude-opus-5-5"])
        prompt = self.complete.call_args.args[1][-1]["content"]
        choices = json.loads(prompt.split("Allowed profiles: ", 1)[1].split("\n<task_data>", 1)[0])
        self.assertEqual(set(choices), {"flash", "sonnet", "opus_worker"})

    def test_single_available_worker_bypasses_selector(self):
        state = State(self.db)
        state.block("quota:codex-main", None, "quota")
        state.block("quota:claude-main", None, "quota")
        self.assertEqual(self.route("code")[0], "done")
        self.assertEqual(self.models(), ["antigravity/gemini-3.8-flash"])

    def test_no_available_worker_makes_no_calls(self):
        state = State(self.db)
        for group in ("codex-main", "claude-main", "google-main"):
            state.block("quota:" + group, None, "quota")
        with self.assertRaises(RouterUnavailable):
            self.route("code")
        self.complete.assert_not_called()

    def test_backend_attributes_selector_and_fallback_tokens_once_to_concrete_models(self):
        from ai_scientist.treesearch.backend import backend_cli

        selector_usage = {"prompt": 11, "completion": 7, "reasoning": 2, "cached": 3}
        worker_usage = {"prompt": 23, "completion": 13, "reasoning": 5, "cached": 7}
        self.complete.side_effect = [
            (json.dumps({"profile": "sol", "reason": "bounded task"}), selector_usage),
            CLIError("quota exhausted"),
            ("done", {**worker_usage, "reported_model": "gemini-3.8-flash"}),
        ]
        with patch.object(backend_cli, "complete", side_effect=complete_routed), patch.object(backend_cli, "token_tracker", self.tracker):
            output, _, incoming, outgoing, metadata = backend_cli.query(None, "offline task", model="router/code")
        self.assertEqual((output, incoming, outgoing), ("done", 23, 13))
        self.assertEqual(metadata, {"model": "antigravity/gemini-3.8-flash"})
        self.assertEqual(self.models(), ["claude-code/claude-opus-5-5", "codex/gpt-6.1-sol", "antigravity/gemini-3.8-flash"])
        self.assertEqual(dict(self.tracker.token_counts), {
            "claude-code/claude-opus-5-5": selector_usage,
            "antigravity/gemini-3.8-flash": worker_usage,
        })
        events = [json.loads(line) for line in self.usage_log.read_text().splitlines()]
        self.assertEqual(events, [
            {"model": "claude-code/claude-opus-5-5", **selector_usage},
            {"model": "antigravity/gemini-3.8-flash", **worker_usage},
        ])

    def test_policy_requires_all_roles_and_pinned_single_candidate(self):
        original = copy.deepcopy(self.policy)
        for role in REQUIRED_ROLES:
            with self.subTest(role=role):
                self.policy = copy.deepcopy(original)
                del self.policy["roles"][role]
                self.save_policy()
                with self.assertRaisesRegex(ValueError, "Missing LLM roles"):
                    load_policy(self.path)
        self.policy = copy.deepcopy(original)
        self.policy["roles"]["review"]["candidates"].append("sol")
        self.save_policy()
        with self.assertRaisesRegex(ValueError, "Pinned role"):
            load_policy(self.path)
        self.assertFalse(self.db.exists())

    def test_real_launcher_binding_and_dry_run_create_no_state_or_outputs(self):
        output = io.StringIO()
        before = dict(os.environ)
        files_before = set(self.directory.rglob("*"))
        with contextlib.redirect_stdout(output), patch("autoresearch.llm_router.State", side_effect=AssertionError("dry run opened state")):
            status = main(["--project", str(ROOT / "tests/fixtures/project"),
                           "--output-dir", str(self.directory / "runs"), "--dry-run"])
        self.assertEqual(status, 0)
        resolved = json.loads(output.getvalue())
        self.assertEqual(resolved["llm_policy"], self.policy)
        self.assertEqual(resolved["config"]["agent"]["vlm_feedback"]["model"], "router/vision")
        self.assertEqual(resolved["config"]["report"]["model"], "router/summary")
        self.assertEqual(resolved["models"]["model_review"], "router/review")
        self.assertEqual(resolved["models"]["model_writeup"], "router/writeup")
        self.assertEqual(resolved["llm_policy"]["profiles"]["opus"]["effort"], "medium")
        self.assertEqual(set(self.directory.rglob("*")), files_before)
        self.assertEqual(dict(os.environ), before)
        self.complete.assert_not_called()
        config = load_run_config(ROOT / "configs/default.yaml", provider="codex", worker="codex/exact", orchestrator="codex/boss")
        self.assertEqual(config["agent"]["code"]["model"], "codex/exact")
        self.assertEqual(config["agent"]["orchestrator"]["model"], "codex/boss")
        self.assertFalse(self.db.exists())
