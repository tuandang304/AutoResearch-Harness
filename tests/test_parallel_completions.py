"""Offline checks for bounded completion sub-agents; no provider processes."""

from copy import deepcopy
import threading
import unittest
from unittest.mock import patch

from ai_scientist import cli_llm
from autoresearch.llm_router import RouterUnavailable


class ParallelCompletionsTests(unittest.TestCase):
    def test_parallel_order_input_isolation_and_usage(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "task"}]}]
        lock = threading.Lock()
        first_pair = threading.Barrier(2, timeout=5)
        second_done = threading.Event()
        active = peak = issued = 0
        completed = []

        def copy_for_child(value):
            nonlocal issued
            copied = deepcopy(value)
            copied[0]["child_index"] = issued
            issued += 1
            return copied

        def complete(model, child_messages):
            nonlocal active, peak
            index = child_messages[0]["child_index"]
            with lock:
                active += 1
                peak = max(peak, active)
            if index < 2:
                first_pair.wait()
            if index == 0:
                self.assertTrue(second_done.wait(5))
            child_messages[0]["content"][0]["text"] = str(index)
            with lock:
                completed.append(index)
                active -= 1
            if index == 1:
                second_done.set()
            return str(index), {"model": f"codex/child-{index}", "prompt": index + 1,
                                "completion": 2, "reasoning": 1, "cached": index}

        with patch("autoresearch.llm_router.load_policy", return_value={
            "routing": {"max_parallel_subagents": 2}
        }), patch.object(cli_llm, "complete", side_effect=complete), patch.object(
            cli_llm, "deepcopy", side_effect=copy_for_child
        ):
            response = cli_llm.CLIClient("router/writing").chat.completions.create(
                messages=messages, n=5
            )

        self.assertEqual(peak, 2)
        self.assertLess(completed.index(1), completed.index(0))
        self.assertEqual([choice.index for choice in response.choices], list(range(5)))
        self.assertEqual([choice.message.content for choice in response.choices], list(map(str, range(5))))
        self.assertEqual([choice.model for choice in response.choices], [f"codex/child-{i}" for i in range(5)])
        self.assertEqual([choice.usage["prompt"] for choice in response.choices], [1, 2, 3, 4, 5])
        self.assertEqual(response.model, "router/mixed")
        self.assertEqual(response.usage.prompt_tokens, 15)
        self.assertEqual(response.usage.completion_tokens, 10)
        self.assertEqual(response.usage.prompt_tokens_details.cached_tokens, 10)
        self.assertEqual(response.usage.completion_tokens_details.reasoning_tokens, 5)
        self.assertEqual(messages, [{"role": "user", "content": [{"type": "text", "text": "task"}]}])

    def test_router_failure_propagates_without_launching_rest_of_batch(self):
        first_pair = threading.Barrier(2, timeout=5)
        failure = RouterUnavailable("quota exhausted")

        def fail(model, messages):
            first_pair.wait()
            raise failure

        with patch("autoresearch.llm_router.load_policy", return_value={
            "routing": {"max_parallel_subagents": 2}
        }), patch.object(cli_llm, "complete", side_effect=fail) as complete:
            with self.assertRaises(RouterUnavailable) as caught:
                cli_llm.CLIClient("router/review").chat.completions.create(messages=[], n=8)
        self.assertIs(caught.exception, failure)
        self.assertEqual(complete.call_count, 2)

    def test_legacy_policy_and_disabled_parallelism_are_serial(self):
        caller = threading.get_ident()
        for routing in ({}, {"max_parallel_subagents": 1}):
            with self.subTest(routing=routing), patch("autoresearch.llm_router.load_policy", return_value={
                "routing": routing
            }), patch.object(cli_llm, "complete", side_effect=lambda *args: (str(threading.get_ident()), {})):
                response = cli_llm.CLIClient("router/writing").chat.completions.create(n=3)
            self.assertEqual([choice.message.content for choice in response.choices], [str(caller)] * 3)

    def test_concrete_overrides_and_single_calls_do_not_load_policy(self):
        caller = threading.get_ident()
        for model, count in (("codex/exact", 3), ("router/writing", 1), ("router/writing", 0)):
            with self.subTest(model=model, count=count), patch("autoresearch.llm_router.load_policy") as load, patch.object(
                cli_llm, "complete", side_effect=lambda *args: (str(threading.get_ident()), {})
            ):
                response = cli_llm.CLIClient(model).chat.completions.create(n=count)
                load.assert_not_called()
            self.assertEqual([choice.message.content for choice in response.choices], [str(caller)] * (count or 1))


if __name__ == "__main__":
    unittest.main()
