import os
import unittest
from unittest.mock import patch

from ai_scientist.cli_llm import CLIClient
from ai_scientist.utils.token_tracker import record_response, token_tracker


class RoutedAccountingTests(unittest.TestCase):
    def test_heterogeneous_batch_is_counted_per_model_once(self):
        token_tracker.reset()
        with patch.dict(os.environ, {}, clear=True), patch("ai_scientist.cli_llm.complete", side_effect=[
            ("a", {"model": "codex/gpt-6-luna", "prompt": 5, "completion": 2}),
            ("b", {"model": "antigravity/gemini-3.8-flash", "prompt": 7, "completion": 3}),
        ]):
            result = CLIClient("router/feedback").chat.completions.create(messages=[], n=2)
            record_response(result, model="router/feedback")
        self.assertEqual(set(token_tracker.token_counts), {
            "codex/gpt-6-luna", "antigravity/gemini-3.8-flash"})
        self.assertEqual(token_tracker.token_counts["codex/gpt-6-luna"]["prompt"], 5)
        self.assertEqual(token_tracker.token_counts["antigravity/gemini-3.8-flash"]["completion"], 3)
        token_tracker.reset()

    def test_tree_backend_uses_concrete_model_after_fallback(self):
        from ai_scientist.treesearch.backend import backend_cli
        token_tracker.reset()
        with patch.dict(os.environ, {}, clear=True), patch.object(backend_cli, "complete", return_value=(
            "ok", {"model": "codex/gpt-6-sol", "prompt": 8, "completion": 2}
        )):
            result = backend_cli.query("instructions", "task", model="router/code")
        self.assertEqual(result[-1]["model"], "codex/gpt-6-sol")
        self.assertEqual(set(token_tracker.token_counts), {"codex/gpt-6-sol"})
        token_tracker.reset()
