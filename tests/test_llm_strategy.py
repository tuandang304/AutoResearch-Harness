import contextlib
import io
import json
import unittest

from omegaconf import OmegaConf

from autoresearch.config import ROOT, load_run_config
from autoresearch.llm_strategy import decision_model
from launch_scientist_bfts import main


class StrategyTests(unittest.TestCase):
    def test_worker_override_preserves_original_as_orchestrator(self):
        config = load_run_config(
            ROOT / "configs/uav_lowlight_t4_smoke.yaml", worker="codex/test-worker"
        )
        agent = OmegaConf.create(config["agent"])
        self.assertEqual(decision_model(agent).model, "codex/gpt-6-astra")
        self.assertEqual(agent.select_node.model, "codex/gpt-6-astra")
        for role in ("code", "feedback", "vlm_feedback", "summary"):
            self.assertEqual(agent[role].model, "codex/test-worker")

    def test_legacy_routing(self):
        agent = OmegaConf.create({"code": {"model": "coder"}, "feedback": {"model": "reviewer"}})
        self.assertEqual(decision_model(agent).model, "reviewer")
        self.assertEqual(decision_model(agent, "code").model, "coder")

    def test_dry_run_resolves_paper_roles_and_explicit_override(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main([
                "--config", str(ROOT / "configs/uav_lowlight_t4_smoke.yaml"),
                "--dry-run", "--orchestrator-model", "codex/gpt-6-sol",
                "--worker-model", "codex/test-worker",
                "--model_citation", "codex/test-citation",
            ])
        self.assertEqual(status, 0)
        resolved = json.loads(output.getvalue())
        self.assertEqual(resolved["models"]["model_writeup"], "codex/gpt-6-sol")
        self.assertEqual(resolved["models"]["model_review"], "codex/gpt-6-sol")
        self.assertEqual(resolved["models"]["model_agg_plots"], "codex/test-worker")
        self.assertEqual(resolved["models"]["model_citation"], "codex/test-citation")

    def test_empty_orchestrator_rejected(self):
        with self.assertRaises(ValueError):
            load_run_config(ROOT / "bfts_config.yaml", orchestrator=" ")
