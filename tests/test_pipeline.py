from contextlib import ExitStack, redirect_stdout, redirect_stderr
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from autoresearch.config import ROOT, idea_slug, load_run_config, validate_config
from launch_scientist_bfts import main, find_pdf_path_for_review


class PipelineTests(unittest.TestCase):
    def test_smoke_configuration_and_stage_limit(self):
        from types import SimpleNamespace
        from ai_scientist.treesearch.agent_manager import AgentManager

        config = load_run_config(ROOT / "configs/uav_lowlight_t4_smoke.yaml")
        self.assertEqual(config["agent"]["code"]["model"], "codex/gpt-6-sol")
        self.assertEqual(config["exec"]["backend"], "colab")
        manager = AgentManager.__new__(AgentManager)
        manager.cfg = SimpleNamespace(agent=SimpleNamespace(max_stages=1))
        stage = SimpleNamespace(name="1_initial_implementation_1_preliminary", stage_number=1, max_iterations=2)
        self.assertIsNone(manager._create_next_main_stage(stage, None))
        manager.journals = {stage.name: SimpleNamespace(nodes=[1, 2], good_nodes=[2])}
        self.assertEqual(manager._check_stage_completion(stage), (True, "Found working implementation"))
        for value in (0, 5, True, 1.5):
            invalid = copy.deepcopy(config)
            invalid["agent"]["max_stages"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_config(invalid)

    def test_experiments_only_skips_final_plotting_and_paper(self):
        def experiments(config_path):
            config = yaml.safe_load(Path(config_path).read_text())
            path = Path(config["log_dir"]) / "0-run" / "experiment_results"
            path.mkdir(parents=True)
            (path / "result.txt").write_text("fixture")

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch("ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.perform_experiments_bfts", side_effect=experiments))
            plots = stack.enter_context(patch("ai_scientist.perform_plotting.aggregate_plots"))
            writer = stack.enter_context(patch("ai_scientist.perform_writeup.perform_writeup"))
            self.assertEqual(main(["--experiments-only", "--output-dir", directory]), 0)
            plots.assert_not_called()
            writer.assert_not_called()

    def test_dry_run_resolves_all_roles_and_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as directory, io.StringIO() as output, redirect_stdout(
            output
        ):
            result = main(
                [
                    "--provider",
                    "codex",
                    "--exec_backend",
                    "colab",
                    "--dry-run",
                    "--output-dir",
                    directory,
                    "--model_review",
                    "claude-code/sonnet",
                ]
            )
            resolved = json.loads(output.getvalue())
            self.assertEqual(result, 0)
            self.assertEqual(resolved["models"]["model_review"], "claude-code/sonnet")
            self.assertEqual(
                resolved["config"]["agent"]["summary"]["model"], "codex/default"
            )
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_validation_rejects_bad_workers_paths_and_timeouts(self):
        original = load_run_config(ROOT / "bfts_config.yaml")
        for section, key, value in (
            ("agent", "num_workers", 0),
            ("exec", "timeout", float("nan")),
            ("exec", "agent_file_name", "../escape.py"),
            ("exec", "backend", "unknown"),
        ):
            config = copy.deepcopy(original)
            config[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(config)
        self.assertNotIn("/", idea_slug("../../outside/path"))

    def test_missing_idea_and_out_of_range_index(self):
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--dry-run", "--idea_idx", "-1"]), 1)
            self.assertEqual(
                main(["--dry-run", "--load_ideas", "/no-such-ideas.json"]), 1
            )

    def test_pdf_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(find_pdf_path_for_review(root))
            for name in (
                "paper.pdf",
                "paper_reflection2.pdf",
                "paper_reflection10.pdf",
                "paper_reflection_final.pdf",
            ):
                (root / name).touch()
                self.assertEqual(find_pdf_path_for_review(root), str(root / name))

    def test_external_output_and_nonmutating_node_restore(self):
        from ai_scientist.treesearch.journal import Node

        with tempfile.TemporaryDirectory() as directory:
            node = Node(code="print(1)", plan="test")
            node.exp_results_dir = directory
            data = node.to_dict()
            before = copy.deepcopy(data)
            Node.from_dict(data)
            self.assertEqual(data, before)
            self.assertEqual(data["exp_results_dir"], directory)

    def test_normal_writeup_and_failure_status_without_model_calls(self):
        def experiments(config_path):
            config = yaml.safe_load(Path(config_path).read_text())
            path = Path(config["log_dir"]) / "0-run" / "experiment_results"
            path.mkdir(parents=True)
            (path / "result.txt").write_text("test output")

        # Assert the actual writeup signature through autospec: no citations_text.
        for writeup_result, expected in ((True, "completed"), (False, "failed")):
            with self.subTest(
                status=expected
            ), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(redirect_stderr(io.StringIO()))
                stack.enter_context(
                    patch(
                        "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.perform_experiments_bfts",
                        side_effect=experiments,
                    )
                )
                stack.enter_context(
                    patch("ai_scientist.perform_plotting.aggregate_plots")
                )
                writeup = stack.enter_context(
                    patch(
                        "ai_scientist.perform_writeup.perform_writeup",
                        autospec=True,
                        return_value=writeup_result,
                    )
                )
                citations = stack.enter_context(
                    patch("ai_scientist.perform_icbinb_writeup.gather_citations")
                )
                code = main(
                    [
                        "--output-dir",
                        directory,
                        "--writeup-type",
                        "normal",
                        "--skip_review",
                        "--writeup-retries",
                        "1",
                    ]
                )
                self.assertEqual(code, 0 if writeup_result else 1)
                self.assertEqual(writeup.call_count, 1)
                citations.assert_not_called()
                state_file = next(Path(directory).glob("*/run_status.json"))
                self.assertEqual(json.loads(state_file.read_text())["status"], expected)
                self.assertTrue((state_file.parent / "token_tracker.json").exists())


if __name__ == "__main__":
    unittest.main()
