import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from ai_scientist import cli_llm
from ai_scientist.llm import get_batch_responses_from_llm
from ai_scientist.utils.token_tracker import (
    record_response,
    token_tracker,
    track_token_usage,
)


class CLITests(unittest.TestCase):
    def test_codex_low_effort_is_explicit(self):
        events = ['{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}', '{"type":"turn.completed","usage":{}}']
        with tempfile.TemporaryDirectory() as cwd, patch.dict(
            os.environ, {"AUTORESEARCH_CODEX_REASONING_EFFORT": "low"}
        ), patch.object(cli_llm, "_run_streaming", return_value=events) as run:
            for model in ("gpt-6-astra", "gpt-6-sol"):
                cli_llm._run_codex(model, "", "ok", [], cwd)
                cmd = run.call_args.args[0]
                self.assertEqual(cmd[cmd.index("-m") + 1], model)
                self.assertEqual(cmd[cmd.index("-c") + 1], 'model_reasoning_effort="low"')

    def test_api_retries_are_bounded_and_preserve_error(self):
        from ai_scientist.treesearch.backend.utils import backoff_create
        from unittest.mock import Mock

        request = Mock(side_effect=ConnectionError("unreachable"))
        with patch("ai_scientist.treesearch.backend.utils.time.sleep"):
            with self.assertRaisesRegex(ConnectionError, "unreachable"):
                backoff_create(request, (ConnectionError,))
        self.assertEqual(request.call_count, 6)

    def test_codex_stream_result_and_usage(self):
        events = [
            '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}\n',
            '{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":3}}\n',
        ]
        with tempfile.TemporaryDirectory() as cwd, patch.object(
            cli_llm, "_run_streaming", return_value=events
        ):
            text, usage = cli_llm._run_codex(None, "system", "prompt", [], cwd)
            self.assertEqual(text, "ok")
            self.assertEqual(usage["prompt"], 12)

    def test_structured_output_retry_and_validation(self):
        from ai_scientist.treesearch.backend.backend_cli import query

        replies = [("not JSON", {"prompt": 2}), ('{"ok":true}', {"prompt": 3})]
        schema = {
            "name": "submit",
            "parameters": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
        }
        with patch(
            "ai_scientist.treesearch.backend.backend_cli.complete", side_effect=replies
        ) as complete:
            result = query("task", None, func_spec=schema, model="codex/default")
            self.assertEqual(result[0], {"ok": True})
            self.assertEqual(result[2], 5)
            self.assertEqual(complete.call_count, 2)

    def test_startup_timeout_includes_blocked_stdin(self):
        with tempfile.TemporaryDirectory() as cwd, patch.object(
            cli_llm, "CLI_STARTUP_TIMEOUT", 0.2
        ):
            start = time.monotonic()
            with self.assertRaisesRegex(cli_llm.CLIError, "hung at startup"):
                cli_llm._run_streaming(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    cwd,
                    "x" * 2_000_000,
                    lambda e: False,
                )
            self.assertLess(time.monotonic() - start, 3)

    def test_result_returns_without_waiting_for_cli_exit(self):
        with tempfile.TemporaryDirectory() as cwd:
            start = time.monotonic()
            output = cli_llm._run_streaming(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    'import time; print(\'{"type":"result","result":"ok"}\'); time.sleep(30)',
                ],
                cwd,
                "",
                lambda e: e.get("type") == "result",
            )
            self.assertIn('"result":"ok"', "".join(output))
            self.assertLess(time.monotonic() - start, 3)

    def test_batch_preserves_history_and_counts_once(self):
        token_tracker.reset()
        with patch.object(
            cli_llm, "complete", return_value=("42", {"prompt": 10, "completion": 2})
        ):
            history = [
                {"role": "user", "content": "remember 42"},
                {"role": "assistant", "content": "ok"},
            ]
            answers, histories = get_batch_responses_from_llm(
                "which number?",
                cli_llm.CLIClient("codex/default"),
                "codex/default",
                "Be terse",
                msg_history=history,
                n_responses=2,
            )
            self.assertEqual(answers, ["42", "42"])
            self.assertEqual(histories[0][:2], history)
            self.assertEqual(
                token_tracker.get_summary()["codex/default"]["tokens"]["prompt"], 20
            )

    def test_optional_usage_fields_and_positional_calls(self):
        token_tracker.reset()

        @track_token_usage
        def request(prompt, model):
            return NS(
                model=model,
                usage=NS(prompt_tokens=4, completion_tokens=2),
                choices=[NS(message=NS(content="ok"))],
            )

        request("hi", "fake")
        self.assertEqual(token_tracker.get_summary()["fake"]["tokens"]["prompt"], 4)
        self.assertIsNone(token_tracker.get_summary()["fake"]["cost (USD)"])
        record_response(
            NS(
                model="claude-test",
                usage=NS(input_tokens=3, output_tokens=2, cache_read_input_tokens=5),
                content=[NS(text="hi")],
            )
        )
        self.assertEqual(
            token_tracker.get_summary()["claude-test"]["tokens"]["prompt"], 8
        )

    def test_worker_usage_ledger(self):
        import subprocess

        with tempfile.TemporaryDirectory() as directory:
            ledger = str(Path(directory) / "usage.jsonl")
            with patch.dict(os.environ, AUTORESEARCH_USAGE_LOG=ledger):
                child = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from ai_scientist.utils.token_tracker import token_tracker; token_tracker.add_tokens('worker', 7, 3, 1, 0)",
                    ],
                    check=True,
                )
                self.assertEqual(child.returncode, 0)
                self.assertEqual(
                    token_tracker.get_summary()["worker"]["tokens"]["prompt"], 7
                )


if __name__ == "__main__":
    unittest.main()
