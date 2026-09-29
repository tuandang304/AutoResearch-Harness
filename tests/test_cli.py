import os
import json
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
    def test_claude_error_metadata_and_safe_logging(self):
        events = [
            {"type": "system", "subtype": "init", "model": "claude-opus-5-5"},
            {"type": "assistant", "error": "rate_limit", "errors": ["token=secret-value https://example.org/?token=private"]},
            {"type": "result", "subtype": "success", "is_error": True},
        ]
        with tempfile.TemporaryDirectory() as cwd, patch.object(
            cli_llm, "_run_streaming", return_value=list(map(json.dumps, events))
        ):
            with self.assertRaises(cli_llm.CLIError) as caught:
                cli_llm._run_claude_code("claude-opus-5-5", "", "hi", [], cwd, effort="medium")
        metadata = caught.exception.metadata
        self.assertEqual(metadata["error_code"], "rate_limit")
        self.assertEqual(metadata["reported_model"], "claude-opus-5-5")
        self.assertEqual(metadata["initialized_model"], "claude-opus-5-5")
        self.assertNotIn("secret-value", json.dumps(metadata))
        self.assertNotIn("private", json.dumps(metadata))
        from unittest.mock import Mock
        with patch.dict(cli_llm._RUNNERS, codex=Mock(side_effect=cli_llm.CLIError("SECRET PROMPT"))), self.assertLogs("ai-scientist") as logs:
            with self.assertRaises(cli_llm.CLIError):
                cli_llm.complete("codex/test", [], max_attempts=1)
        self.assertNotIn("SECRET PROMPT", str(logs.output))

    def test_verification_identity_matches_and_mismatches(self):
        from scripts.verify_llm_profiles import identity_evidence, probe

        for reported in ("gpt-6-sol", "codex/gpt-6-sol"):
            self.assertEqual(identity_evidence("codex/gpt-6-sol", reported), "exact_metadata_match")
        self.assertEqual(identity_evidence("codex/gpt-6-sol", None), "request_accepted_only")
        with patch.object(cli_llm, "complete", return_value=("PROFILE_OK", {"reported_model": "gpt-6-luna"})):
            records = probe(("codex/gpt-6-sol", "low"))
        self.assertEqual(records[0]["status"], "model_mismatch")

    def test_verifier_refuses_unowned_artifact_before_calls(self):
        from scripts.verify_llm_profiles import main

        with tempfile.TemporaryDirectory() as cwd:
            target = Path(cwd) / "existing.json"
            target.write_text('{"user_owned": true}')
            with patch.object(sys, "argv", ["verify", "--live", "--output", str(target)]), patch.object(cli_llm, "complete") as complete, patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    main()
            complete.assert_not_called()
            self.assertEqual(target.read_text(), '{"user_owned": true}')

    def test_router_dispatch_and_batch_attribution(self):
        from unittest.mock import Mock

        routed = Mock(return_value=("ok", {"model": "codex/gpt-6-sol"}))
        with patch.dict(sys.modules, {"autoresearch.llm_router": NS(complete_routed=routed)}):
            self.assertTrue(cli_llm.is_cli_model("router/worker"))
            self.assertNotIn("router", cli_llm.CLI_PROVIDERS)
            client = cli_llm.CLIClient("router/worker")
            response = client.chat.completions.create(messages=[])
            self.assertEqual(response.model, "codex/gpt-6-sol")
            routed.assert_called_once_with("router/worker", [])
        with patch.object(cli_llm, "complete", side_effect=[
            ("a", {"model": "codex/gpt-6-sol", "prompt": 2}),
            ("b", {"model": "claude-code/claude-opus-5-5", "prompt": 3}),
        ]):
            response = client.chat.completions.create(messages=[], n=2)
        self.assertEqual(response.model, "router/mixed")
        self.assertEqual(response.usage.prompt_tokens, 5)
        self.assertEqual([c.usage["prompt"] for c in response.choices], [2, 3])
        self.assertEqual([c.model for c in response.choices], ["codex/gpt-6-sol", "claude-code/claude-opus-5-5"])

    def test_per_call_effort_flags_and_metadata(self):
        cases = [
            ("claude-code", cli_llm._run_claude_code, [
                {"type": "assistant", "message": {"model": "reported-id"}},
                {"type": "result", "subtype": "success", "result": "ok"},
            ]),
            ("codex", cli_llm._run_codex, [
                {"type": "session.created", "model": "reported-id"},
                {"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}},
                {"type": "turn.completed", "usage": {}},
            ]),
            ("antigravity", cli_llm._run_antigravity, [
                {"event": "result", "result": {"status": "SUCCESS", "response": "ok", "model": "reported-id"}},
            ]),
        ]
        with tempfile.TemporaryDirectory() as cwd, patch.dict(
            os.environ, {"AUTORESEARCH_CODEX_REASONING_EFFORT": "high"}
        ):
            for provider, runner, events in cases:
                with self.subTest(provider=provider), patch.object(
                    cli_llm, "_run_streaming", return_value=list(map(json.dumps, events))
                ) as run:
                    for effort in ("low", "medium"):
                        _, usage = runner("selected-id", "", "ok", [], cwd, effort=effort)
                        cmd = run.call_args.args[0]
                        flag = "-c" if provider == "codex" else "--effort"
                        expected = f'model_reasoning_effort="{effort}"' if provider == "codex" else effort
                        self.assertEqual(cmd[cmd.index(flag) + 1], expected)
                        self.assertEqual(usage["reported_model"], "reported-id")
                    with self.assertRaises(ValueError):
                        runner("selected-id", "", "ok", [], cwd, effort="invalid")
                    if provider != "codex":
                        runner("selected-id", "", "ok", [], cwd)
                        self.assertNotIn("--effort", run.call_args.args[0])
                    self.assertEqual(os.environ["AUTORESEARCH_CODEX_REASONING_EFFORT"], "high")

    def test_complete_effort_attempts_and_exact_model(self):
        from unittest.mock import Mock

        runner = Mock(side_effect=[cli_llm.CLIError("startup"), ("ok", {"reported_model": "actual"})])
        with patch.dict(cli_llm._RUNNERS, codex=runner), patch.object(cli_llm.time, "sleep"):
            _, usage = cli_llm.complete("codex/exact-id", [], effort="low", max_attempts=2)
        self.assertEqual(runner.call_count, 2)
        self.assertEqual(runner.call_args.kwargs, {"effort": "low"})
        self.assertEqual(usage["model"], "codex/exact-id")
        self.assertEqual(usage["reported_model"], "actual")
        runner.reset_mock(side_effect=True)
        runner.side_effect = cli_llm.CLIError("failure")
        with patch.dict(cli_llm._RUNNERS, codex=runner), patch.object(cli_llm.time, "sleep") as sleep:
            with self.assertRaises(cli_llm.CLIError):
                cli_llm.complete("codex/exact-id", [], max_attempts=1)
            self.assertEqual(runner.call_count, 1)
            sleep.assert_not_called()
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                cli_llm.complete("codex/exact-id", [], max_attempts=invalid)

    def test_reported_model_never_reads_generated_text(self):
        self.assertIsNone(cli_llm._reported_model({"result": "I am model X"}))
        self.assertEqual(cli_llm._reported_model({"modelUsage": {"real-id": {}}}), "real-id")

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
