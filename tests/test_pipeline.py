from contextlib import ExitStack, redirect_stdout, redirect_stderr
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import yaml

from autoresearch.config import ROOT, idea_slug, load_run_config, validate_config
from launch_scientist_bfts import main, find_pdf_path_for_review

FIXTURE_PROJECT = ROOT / "tests" / "fixtures" / "project"


class PipelineTests(unittest.TestCase):
    def test_project_paths_and_overrides_do_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "ideas.json").write_text('[{"Name":"independent-study"}]')
            config = load_run_config(ROOT / "configs/default.yaml")
            config["exec"]["timeout"] = 123
            (project / "config.yaml").write_text(yaml.safe_dump(config))
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(["--project", str(project), "--dry-run"])
            self.assertEqual(status, 0)
            resolved = json.loads(output.getvalue())
            self.assertEqual(resolved["idea"], "independent-study")
            self.assertEqual(resolved["output_dir"], str(project / "runs"))
            self.assertEqual(resolved["config"]["exec"]["timeout"], 123)
            self.assertFalse((project / "runs").exists())
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(["--project", str(project), "--dry-run",
                               "--config", str(ROOT / "configs/default.yaml"),
                               "--output-dir", str(project / "custom")])
            self.assertEqual(status, 0)
            resolved = json.loads(output.getvalue())
            self.assertEqual(resolved["output_dir"], str(project / "custom"))
            self.assertNotEqual(resolved["config"]["exec"]["timeout"], 123)
            self.assertFalse((project / "custom").exists())

    def test_code_only_model_reply_does_not_waste_format_retries(self):
        from ai_scientist.treesearch.parallel_agent import MinimalAgent
        from types import SimpleNamespace
        cfg = SimpleNamespace(agent=SimpleNamespace(code=SimpleNamespace(model="codex/gpt-6-sol", temp=0.2)))
        agent = MinimalAgent(task_desc="smoke", cfg=cfg)
        with patch("ai_scientist.treesearch.parallel_agent.query", return_value="```python\nprint('ok')\n```") as call:
            plan, code = agent.plan_and_code_query({})
            self.assertTrue(plan)
            self.assertIn("print", code)
            call.assert_called_once()
        with patch("ai_scientist.treesearch.parallel_agent.query", return_value="No executable code available.") as call:
            with self.assertRaisesRegex(ValueError, "executable Python"):
                agent.plan_and_code_query({}, retries=2)
            self.assertEqual(call.call_count, 2)

    def test_worker_timeout_fails_instead_of_silently_retrying(self):
        from ai_scientist.treesearch.parallel_agent import ParallelAgent
        from omegaconf import OmegaConf
        agent = ParallelAgent.__new__(ParallelAgent)
        agent.cfg = OmegaConf.create({"agent": {"summary": None}})
        agent._select_parallel_nodes = Mock(return_value=[None])
        agent.journal = Mock()
        agent.gpu_manager = None
        agent.stage_name = "1_initial_implementation_1_preliminary"
        agent.best_stage1_node = agent.best_stage2_node = agent.best_stage3_node = None
        agent.task_desc = "test"
        agent.evaluation_metrics = "test"
        agent.timeout = 1
        agent.executor = Mock()
        future = agent.executor.submit.return_value
        future.result.side_effect = TimeoutError()
        with self.assertRaisesRegex(TimeoutError, "uncounted retries"):
            agent.step(None)
        future.cancel.assert_called_once()
        agent.executor.submit.assert_called_once()

    def test_shutdown_retains_process_handles(self):
        from ai_scientist.treesearch.parallel_agent import ParallelAgent
        agent = ParallelAgent.__new__(ParallelAgent)
        agent._is_shutdown = False
        agent.gpu_manager = None
        agent.executor = Mock()
        process = Mock()
        agent.executor._processes = {1: process}
        agent.executor.shutdown.side_effect = lambda **kw: setattr(agent.executor, "_processes", None)
        agent.cleanup()
        process.terminate.assert_called_once()

    def test_deterministic_smoke_validation_retains_artifacts(self):
        from ai_scientist.treesearch.interpreter import ExecutionResult
        from ai_scientist.treesearch.journal import Node
        from ai_scientist.treesearch.parallel_agent import _finish_smoke_node
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspaces" / "0-run" / "worker"
            working = workspace / "working"
            working.mkdir(parents=True)
            np = __import__("numpy")
            np.save(working / "experiment_data.npy", {"loss": np.array([1.0])})
            (working / "plot.png").write_bytes(b"png")
            (workspace / "experiment_code.py").write_text("print('ok')")
            (workspace / "execution_result.json").write_text("{}")
            cfg = SimpleNamespace(workspace_dir=root / "workspaces" / "0-run")
            node = Node(code="print('ok')", plan="smoke")
            result = _finish_smoke_node(
                node, ExecutionResult(["ok"], 0.1, None), working, workspace, cfg
            )
            self.assertFalse(result["is_buggy"])
            self.assertFalse(result["is_buggy_plots"])
            from ai_scientist.treesearch.journal import Journal
            journal = Journal()
            journal.append(Node.from_dict(result, journal))
            self.assertEqual(len(journal.good_nodes), 1)
            seed_node = Node(code="seed", plan="repeat", is_seed_node=True)
            seed_node.is_buggy = seed_node.is_buggy_plots = False
            seed_node.metric = journal.nodes[0].metric
            journal.append(seed_node)
            from types import SimpleNamespace
            with patch("ai_scientist.treesearch.journal.query") as query:
                selected = journal.get_best_node(
                    cfg=SimpleNamespace(agent={"smoke_test": True})
                )
                self.assertEqual(selected.id, result["id"])
                query.assert_not_called()
            saved = root / "workspaces" / "logs" / "0-run" / "experiment_results"
            self.assertTrue(list(saved.rglob("experiment_data.npy")))
            np.save(working / "experiment_data.npy", {"loss": np.array([float("nan")])})
            failed = _finish_smoke_node(
                Node(code="print('ok')", plan="smoke"),
                ExecutionResult(["ok"], 0.1, None), working, workspace, cfg
            )
            self.assertTrue(failed["is_buggy"])
            self.assertIn("non-finite", failed["analysis"])

    def test_smoke_validation_normalizes_torch_version_without_torch(self):
        import sys
        import types
        import numpy as np
        from ai_scientist.treesearch.interpreter import ExecutionResult
        from ai_scientist.treesearch.journal import Node
        from ai_scientist.treesearch.parallel_agent import _finish_smoke_node
        from types import SimpleNamespace

        torch_stub = types.ModuleType("torch")
        version_stub = types.ModuleType("torch.torch_version")
        TorchVersion = type("TorchVersion", (str,), {"__module__": "torch.torch_version"})
        version_stub.TorchVersion = TorchVersion
        torch_stub.torch_version = version_stub
        previous = {name: sys.modules.get(name) for name in ("torch", "torch.torch_version")}
        sys.modules["torch"], sys.modules["torch.torch_version"] = torch_stub, version_stub
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                workspace = root / "workspaces" / "0-run" / "worker"
                working = workspace / "working"
                working.mkdir(parents=True)
                np.save(working / "experiment_data.npy", {"torch": TorchVersion("2.0")})
                (working / "plot.png").write_bytes(b"png")
                (workspace / "execution_result.json").write_text("{}")
                result = _finish_smoke_node(
                    Node(code="x", plan="x"), ExecutionResult([], 0.1, None),
                    working, workspace, SimpleNamespace(workspace_dir=root / "workspaces" / "0-run")
                )
                self.assertFalse(result["is_buggy"])
                del sys.modules["torch"], sys.modules["torch.torch_version"]
                loaded = np.load(working / "experiment_data.npy", allow_pickle=True).item()
                self.assertIs(type(loaded["torch"]), str)
        finally:
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_smoke_configuration_and_stage_limit(self):
        from types import SimpleNamespace
        from ai_scientist.treesearch.agent_manager import AgentManager

        config = load_run_config(ROOT / "tests/fixtures/pipeline.yaml")
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
            self.assertEqual(main(["--project", str(FIXTURE_PROJECT), "--experiments-only", "--output-dir", directory]), 0)
            plots.assert_not_called()
            writer.assert_not_called()

    def test_dry_run_resolves_all_roles_and_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as directory, io.StringIO() as output, redirect_stdout(
            output
        ):
            result = main(
                [
                    "--project",
                    str(FIXTURE_PROJECT),
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
        original = load_run_config(ROOT / "configs/default.yaml")
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
            self.assertEqual(main(["--project", str(FIXTURE_PROJECT), "--dry-run", "--idea_idx", "-1"]), 1)
            self.assertEqual(
                main(["--project", str(FIXTURE_PROJECT), "--dry-run", "--load_ideas", "/no-such-ideas.json"]), 1
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
                        "--project",
                        str(FIXTURE_PROJECT),
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
