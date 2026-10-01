"""Offline regressions: routing exhaustion must stop scientific work."""
from concurrent.futures import Future, ProcessPoolExecutor
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from autoresearch.llm_router import RouterUnavailable
from ai_scientist.treesearch.agent_manager import AgentManager
from ai_scientist.treesearch.parallel_agent import MinimalAgent, ParallelAgent
from ai_scientist.treesearch.interpreter import ExecutionResult
from ai_scientist.treesearch.journal import Journal, Node


def exhausted_worker():
    raise RouterUnavailable("quota exhausted")


class RoutingFailureTests(unittest.TestCase):
    def setUp(self):
        self.failure = RouterUnavailable("quota exhausted")
        self.cfg = OmegaConf.create({
            "agent": {"feedback": {"model": "router/feedback", "temp": 0},
                      "vlm_feedback": {"model": "router/vision", "temp": 0},
                      "smoke_test": False, "multi_seed_eval": {"num_seeds": 1}},
            "exec": {"local_postprocessing": False},
        })

    def test_manager_decisions_propagate_exhaustion(self):
        manager = AgentManager.__new__(AgentManager)
        manager.cfg = self.cfg
        stage = SimpleNamespace(name="2_test", stage_number=2, max_iterations=10,
                                description="test", goals=["test"])
        journal = Mock()
        journal.nodes = [object(), object()]
        journal.get_best_node.return_value = SimpleNamespace(datasets_successfully_tested=[])
        manager.journals = {stage.name: journal}
        manager._parse_vlm_feedback = Mock(return_value="feedback")
        manager._gather_stage_metrics = Mock(return_value={"total_nodes": 2, "good_nodes": 1, "best_metric": None})
        manager._identify_issues = Mock(return_value=[])
        manager._analyze_progress = Mock(return_value={"convergence_status": "unknown", "recent_changes": []})
        calls = [
            lambda: manager._check_substage_completion(stage, journal),
            lambda: manager._check_stage_completion(stage),
            lambda: manager._generate_substage_goal("test", journal),
            lambda: manager._get_response("test"),
            lambda: manager._evaluate_stage_progression(stage, {}),
        ]
        for index, call in enumerate(calls):
            with self.subTest(handler=index), patch("ai_scientist.treesearch.agent_manager.query", side_effect=self.failure) as query:
                with self.assertRaises(RouterUnavailable) as caught:
                    call()
                self.assertIs(caught.exception, self.failure)
                query.assert_called_once()

    def test_ordinary_manager_error_keeps_legacy_fallback(self):
        manager = AgentManager.__new__(AgentManager)
        manager.cfg = self.cfg
        with patch("ai_scientist.treesearch.agent_manager.query", side_effect=ValueError("bad response")):
            self.assertEqual(manager._get_response("test")["name"], "fallback_stage")

    def test_citation_helpers_do_not_hide_routing_exhaustion(self):
        from ai_scientist import perform_writeup, perform_icbinb_writeup
        for module in (perform_writeup, perform_icbinb_writeup):
            with self.subTest(module=module.__name__), patch.object(
                module, "get_response_from_llm", side_effect=self.failure
            ) as request:
                with self.assertRaises(RouterUnavailable):
                    module.get_citation_addition(Mock(), "router/citation", ("results", ""), 0, 1, "idea")
                request.assert_called_once()

    def test_citation_collection_stops_without_uncounted_round_retries(self):
        from ai_scientist import perform_icbinb_writeup as module
        with tempfile.TemporaryDirectory() as directory, patch.object(module, "load_idea_text", return_value="idea"), \
                patch.object(module, "load_exp_summaries", return_value={}), \
                patch.object(module, "filter_experiment_summaries", return_value={}), \
                patch.object(module, "create_client", return_value=(Mock(), "router/citation")), \
                patch.object(module, "get_citation_addition", side_effect=self.failure) as request:
            with self.assertRaises(RouterUnavailable):
                module.gather_citations(directory, num_cite_rounds=3)
            request.assert_called_once()

    def test_aggregation_does_not_return_success_when_model_unavailable(self):
        from ai_scientist import perform_plotting as module
        with tempfile.TemporaryDirectory() as directory, patch.object(module, "load_idea_text", return_value="idea"), \
                patch.object(module, "load_exp_summaries", return_value={}), \
                patch.object(module, "filter_experiment_summaries", return_value={}), \
                patch.object(module, "create_client", return_value=(Mock(), "router/plotting")), \
                patch.object(module, "get_response_from_llm", side_effect=self.failure) as request:
            with self.assertRaises(RouterUnavailable):
                module.aggregate_plots(directory, model="router/plotting")
            request.assert_called_once()

    def test_plot_selection_and_seed_aggregation_propagate(self):
        worker = MinimalAgent("test", self.cfg)
        node = Node(code="pass", plan="test")
        node.plot_paths = [f"plot{i}.png" for i in range(11)]
        with patch("ai_scientist.treesearch.parallel_agent.query", side_effect=self.failure):
            with self.assertRaises(RouterUnavailable):
                worker._analyze_plots_with_vlm(node)
        agent = ParallelAgent.__new__(ParallelAgent)
        agent._aggregate_seed_eval_results = Mock(side_effect=self.failure)
        with self.assertRaises(RouterUnavailable):
            agent._run_plot_aggregation(node, [node])

    def test_future_failures_stop_step_and_seed_evaluation(self):
        agent = ParallelAgent.__new__(ParallelAgent)
        agent.cfg = self.cfg
        agent.journal = Journal()
        agent.journal.generate_summary = Mock(return_value="summary")
        agent._select_parallel_nodes = Mock(return_value=[None])
        agent.gpu_manager = None
        agent.task_desc = "test"
        agent.evaluation_metrics = "accuracy"
        agent.stage_name = "1_test"
        agent.timeout = 10
        agent.best_stage1_node = agent.best_stage2_node = agent.best_stage3_node = None
        for seeds in (False, True):
            with self.subTest(seeds=seeds):
                future = Future()
                future.set_exception(self.failure)
                agent.executor = Mock()
                agent.executor.submit.return_value = future
                with self.assertRaises(RouterUnavailable) as caught:
                    if seeds:
                        agent._run_multi_seed_evaluation(Node(code="pass", plan="test"))
                    else:
                        agent.step(Mock())
                self.assertIs(caught.exception, self.failure)
                agent.executor.submit.assert_called_once()
                self.assertEqual(agent.journal.nodes, [])

    def test_exception_survives_process_boundary(self):
        with ProcessPoolExecutor(max_workers=1) as executor:
            with self.assertRaisesRegex(RouterUnavailable, "quota exhausted"):
                executor.submit(exhausted_worker).result(timeout=10)

    def test_worker_postprocessing_failures_preserve_artifacts(self):
        for phase in ("metrics", "plotting", "vision"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as root:
                self.cfg.workspace_dir = str(Path(root) / "run")
                worker = Mock()
                child = Node(code="pass", plan="test")
                child.is_buggy = False
                worker._draft.return_value = child
                worker.plan_and_code_query.return_value = ("parse", "pass")
                worker._generate_plotting_code.return_value = "pass"
                interpreter = Mock()

                def execute(code, reset):
                    working = next(Path(root).glob("run/process_*/working"))
                    (working / "data.npy").write_bytes(b"retained experiment data")
                    if phase == "vision":
                        (working / "plot.png").write_bytes(b"plot")
                    return ExecutionResult(["ok"], 1.0, None)

                interpreter.run.side_effect = execute
                # Metric parsing succeeds so the plotting handlers are reached.
                parsed = {"valid_metrics_received": True, "metric_names": [{
                    "metric_name": "accuracy", "lower_is_better": False, "description": "test",
                    "data": [{"dataset_name": "d", "final_value": 1.0, "best_value": 1.0}]}]}
                if phase == "plotting":
                    worker._generate_plotting_code.side_effect = self.failure
                if phase == "vision":
                    worker._analyze_plots_with_vlm.side_effect = self.failure
                with patch.dict(os.environ), patch("ai_scientist.treesearch.parallel_agent.MinimalAgent", return_value=worker), patch("ai_scientist.treesearch.remote_interpreter.make_interpreter", return_value=interpreter), patch("ai_scientist.treesearch.parallel_agent.query", side_effect=self.failure if phase == "metrics" else [parsed]) as query:
                    with self.assertRaises(RouterUnavailable) as caught:
                        ParallelAgent._process_node_wrapper(None, "test", self.cfg)
                self.assertIs(caught.exception, self.failure)
                self.assertEqual(len(list(Path(root).glob("run/process_*/execution_result.json"))), 1)
                self.assertEqual(len(list(Path(root).glob("run/process_*/experiment_code.py"))), 1)
                self.assertTrue(list(Path(root).rglob("data.npy")))
                self.assertGreaterEqual(interpreter.cleanup_session.call_count, 1)
                # main run, metric parsing, then plotting (vision phase only)
                self.assertEqual(interpreter.run.call_count, {"metrics": 2, "plotting": 2, "vision": 3}[phase])
                if phase == "metrics":
                    query.assert_called_once()


    def test_seed_rerun_keeps_results_the_reviewer_flags(self):
        """A seed rerun is dropped only for a crash, not for a disliked result."""
        for exc_type, expected_buggy in ((None, False), ("RuntimeError", True)):
            with self.subTest(exc_type=exc_type), tempfile.TemporaryDirectory() as root:
                self.cfg.workspace_dir = str(Path(root) / "run")
                parent = Node(code="pass", plan="test")
                parent.parse_metrics_code, parent.plot_code = "pass", "pass"
                worker = Mock()
                worker._generate_seed_node.side_effect = lambda p: Node(code=p.code, plan="seed")

                def review(node, exec_result, workspace):
                    node.exc_type = exc_type
                    node.analysis = "The day_test guard failed."
                    node.is_buggy = True

                worker.parse_exec_result.side_effect = review
                interpreter = Mock()

                def execute(code, reset):
                    working = next(Path(root).glob("run/process_*/working"))
                    (working / "experiment_data.npy").write_bytes(b"data")
                    return ExecutionResult(["ok"], 1.0, None)

                interpreter.run.side_effect = execute
                parsed = {"valid_metrics_received": True, "metric_names": [{
                    "metric_name": "AP", "lower_is_better": False, "description": "test",
                    "data": [{"dataset_name": "d", "final_value": 0.1, "best_value": 0.1}]}]}
                with patch.dict(os.environ), \
                        patch("ai_scientist.treesearch.parallel_agent.MinimalAgent", return_value=worker), \
                        patch("ai_scientist.treesearch.remote_interpreter.make_interpreter", return_value=interpreter), \
                        patch("ai_scientist.treesearch.parallel_agent.query", return_value=parsed):
                    result = ParallelAgent._process_node_wrapper(
                        parent.to_dict(), "test", self.cfg, seed_eval=True)
                self.assertEqual(result["is_buggy"], expected_buggy)
                self.assertIn("The day_test guard failed.", result["analysis"])


    def test_stage4_nodes_receive_stage3_baseline_results(self):
        from ai_scientist.treesearch import parallel_agent as module
        with tempfile.TemporaryDirectory() as root:
            results, workspace = Path(root) / "results", Path(root) / "ws"
            results.mkdir(); workspace.mkdir()
            (results / "experiment_data.npy").write_bytes(b"x")
            (results / "states_A.npy").write_bytes(b"y")
            (results / "plot.png").write_bytes(b"z")
            names = module._copy_baseline_results(str(results), workspace)
            self.assertEqual(names, ["experiment_data.npy", "states_A.npy"])
            self.assertEqual(sorted(p.name for p in (workspace / "parent_results").iterdir()), names)
            self.assertEqual(module._copy_baseline_results(None, workspace), [])
            self.assertEqual(module._copy_baseline_results(str(Path(root) / "missing"), workspace), [])
        cfg = OmegaConf.merge(self.cfg, {"exec": {"timeout": 3600, "backend": "local"}})
        worker = MinimalAgent("test", cfg)
        idea = SimpleNamespace(name="no cast", description="drop the colour cast")
        with patch.object(MinimalAgent, "plan_and_code_query", return_value=("plan", "code")) as query:
            worker._generate_ablation_node(Node(code="pass", plan="base"), idea, ["experiment_data.npy"])
        instructions = query.call_args.args[0]["Instructions"]
        self.assertIn("./parent_results/", instructions["Baseline results"])

    def test_step_passes_each_stage_plot_code_to_its_own_parameter(self):
        agent = ParallelAgent.__new__(ParallelAgent)
        agent.cfg = self.cfg
        agent.journal = Journal()
        agent.journal.generate_summary = Mock(return_value="summary")
        parent = Node(code="3", plan="p")
        parent.is_buggy = False
        agent._select_parallel_nodes = Mock(return_value=[parent])
        agent.gpu_manager = None
        agent.task_desc, agent.evaluation_metrics, agent.timeout = "test", "AP", 10
        agent.stage_name = "4_ablation"
        agent._ablation_state = {"completed_ablations": set()}
        agent._generate_ablation_idea = Mock(return_value=SimpleNamespace(name="x", description="y"))
        agent.best_stage1_node, agent.best_stage2_node = Node(code="1", plan="p"), Node(code="2", plan="p")
        agent.best_stage3_node = Node(code="3", plan="p")
        for stage, node in zip("123", (agent.best_stage1_node, agent.best_stage2_node, agent.best_stage3_node)):
            node.plot_code = f"plot{stage}"
        agent.best_stage3_node.exp_results_dir = "logs/0-run/experiment_results/x"
        future = Future()
        future.set_exception(self.failure)
        agent.executor = Mock()
        agent.executor.submit.return_value = future
        with self.assertRaises(RouterUnavailable):
            agent.step(Mock())
        kwargs = agent.executor.submit.call_args.kwargs
        self.assertEqual([kwargs[f"best_stage{i}_plot_code"] for i in "123"], ["plot1", "plot2", "plot3"])
        self.assertEqual(kwargs["baseline_results_dir"], "logs/0-run/experiment_results/x")


    def test_reused_evaluated_node_skips_rerunning_seed_zero(self):
        agent = ParallelAgent.__new__(ParallelAgent)
        agent.cfg = OmegaConf.merge(self.cfg, {"agent": {"multi_seed_eval": {
            "num_seeds": 3, "reuse_evaluated_node": True}}})
        agent.journal, agent.gpu_manager, agent.timeout = Journal(), None, 10
        agent.task_desc, agent.evaluation_metrics, agent.stage_name = "test", "AP", "3_test"
        future = Future()
        future.set_exception(self.failure)
        agent.executor = Mock()
        agent.executor.submit.return_value = future
        with self.assertRaises(RouterUnavailable):
            agent._run_multi_seed_evaluation(Node(code="pass", plan="test"))
        codes = [call.args[1]["code"] for call in agent.executor.submit.call_args_list]
        self.assertEqual(len(codes), 2)
        self.assertIn("AUTORESEARCH_SEED'] = '1'", codes[0])
        self.assertIn("AUTORESEARCH_SEED'] = '2'", codes[1])


if __name__ == "__main__":
    unittest.main()
