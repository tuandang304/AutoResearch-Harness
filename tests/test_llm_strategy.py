import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

from omegaconf import OmegaConf

from autoresearch.config import ROOT, load_run_config
from autoresearch.llm_strategy import decision_model
from launch_scientist_bfts import main, snapshot_llm_policy


class StrategyTests(unittest.TestCase):
    def test_worker_override_preserves_original_as_orchestrator(self):
        config = load_run_config(
            ROOT / "tests/fixtures/pipeline.yaml", worker="codex/test-worker"
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
                "--config", str(ROOT / "tests/fixtures/pipeline.yaml"),
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
            load_run_config(ROOT / "configs/default.yaml", orchestrator=" ")


class RouterLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.config = self.project / "config.yaml"
        self.config.write_text((ROOT / "tests/fixtures/pipeline.yaml").read_text())
        (self.project / "ideas.json").write_text('[{"Name": "router-test"}]')
        self.policy = {"roles": {"orchestrator": {"model": "codex/test", "effort": "low"}}}

        def bind(config, policy):
            for role in ("orchestrator", "code", "feedback", "summary", "vlm_feedback", "select_node"):
                config["agent"].setdefault(role, {"temp": 0.2})["model"] = "router/" + role
            config["report"]["model"] = "router/report"

        self.router = SimpleNamespace(load_policy=Mock(return_value=self.policy), bind_roles=Mock(side_effect=bind))
        self.addCleanup(patch.stopall)
        patch.dict("sys.modules", {"autoresearch.llm_router": self.router}).start()
        patch.dict(os.environ).start()
        for key in ("AUTORESEARCH_LLM_CONFIG", "AUTORESEARCH_ROUTER_STATE", "AUTORESEARCH_ROUTING_LOG"):
            os.environ.pop(key, None)

    def dry_run(self, *flags):
        output = io.StringIO()
        before = dict(os.environ)
        with contextlib.redirect_stdout(output), patch("launch_scientist_bfts.run_pipeline") as run:
            status = main(["--project", str(self.project), "--dry-run", *flags])
        self.assertEqual(status, 0)
        run.assert_not_called()
        self.assertEqual(dict(os.environ), before)
        self.assertFalse((self.project / "runs").exists())
        return json.loads(output.getvalue())

    def test_policy_selection_precedence_and_single_binding(self):
        config = yaml.safe_load(self.config.read_text())
        config["llm_config"] = "configs/selected.yaml"
        self.config.write_text(yaml.safe_dump(config))
        self.dry_run()
        self.router.load_policy.assert_called_once_with(ROOT / "configs/selected.yaml")
        (self.project / "llm.yaml").touch()
        self.dry_run()
        self.router.load_policy.assert_called_with(self.project / "llm.yaml")
        os.environ["AUTORESEARCH_LLM_CONFIG"] = str(self.project / "env.yaml")
        self.dry_run()
        self.router.load_policy.assert_called_with(self.project / "env.yaml")
        resolved = self.dry_run("--llm-config", str(self.project / "explicit.yaml"))
        self.router.load_policy.assert_called_with(self.project / "explicit.yaml")
        self.assertEqual(self.router.load_policy.call_count, 4)
        self.assertEqual(self.router.bind_roles.call_count, 4)
        self.assertEqual(resolved["llm_policy"], self.policy)
        self.assertEqual(resolved["config"]["llm_config"], str(self.project / "explicit.yaml"))
        self.assertEqual(resolved["models"]["model_review"], "router/review")
        self.assertEqual(resolved["models"]["model_writeup"], "router/writeup")
        self.assertEqual(resolved["models"]["model_citation"], "router/citation")

    def test_overrides_follow_binding(self):
        (self.project / "llm.yaml").touch()
        resolved = self.dry_run("--provider", "codex", "--orchestrator-model", "codex/boss",
                                "--worker-model", "codex/helper", "--model_citation", "codex/citer")
        self.assertEqual(resolved["config"]["agent"]["code"]["model"], "codex/helper")
        self.assertEqual(resolved["models"]["model_writeup"], "codex/boss")
        self.assertEqual(resolved["models"]["model_review"], "codex/boss")
        self.assertEqual(resolved["models"]["model_agg_plots"], "codex/helper")
        self.assertEqual(resolved["models"]["model_citation"], "codex/citer")
        resolved = self.dry_run("--worker-model", "codex/helper")
        self.assertEqual(resolved["models"]["model_review"], "router/review")
        self.assertEqual(resolved["models"]["model_writeup_small"], "codex/helper")
        resolved = self.dry_run("--orchestrator-model", "codex/boss")
        self.assertEqual(resolved["models"]["model_review"], "codex/boss")
        self.assertEqual(resolved["models"]["model_citation"], "router/citation")
        resolved = self.dry_run("--provider", "codex")
        self.assertTrue(all(not model.startswith("router/") for model in resolved["models"].values()))

    def test_snapshot_and_environment_restored_after_success_and_failure(self):
        (self.project / "llm.yaml").touch()
        for fail in (False, True):
            for inherited in (False, True):
                with self.subTest(fail=fail, inherited=inherited), patch.dict(os.environ):
                    if inherited:
                        for key in ("AUTORESEARCH_ROUTER_STATE", "AUTORESEARCH_ROUTING_LOG", "AUTORESEARCH_LLM_CONFIG"):
                            os.environ[key] = str(self.project / key)
                    before = dict(os.environ)

                    def run(args, config_idea, config):
                        args.run_dir = self.project / f"run-{fail}-{inherited}"
                        args.run_dir.mkdir()
                        snapshot_llm_policy(args, config)
                        snapshot = args.run_dir / "llm_policy.yaml"
                        self.assertEqual(yaml.safe_load(snapshot.read_text()), self.policy)
                        self.assertEqual(config["llm_config"], str(snapshot))
                        self.assertEqual(os.environ["AUTORESEARCH_LLM_CONFIG"], str(snapshot))
                        self.assertEqual(os.environ["AUTORESEARCH_ROUTING_LOG"], str(args.run_dir / "routing.jsonl"))
                        expected = before.get("AUTORESEARCH_ROUTER_STATE", str(ROOT / ".state/llm-router.sqlite"))
                        self.assertEqual(os.environ["AUTORESEARCH_ROUTER_STATE"], expected)
                        if fail:
                            raise RuntimeError("offline failure")

                    with patch("launch_scientist_bfts.run_pipeline", side_effect=run), patch("launch_scientist_bfts.save_token_tracker"), contextlib.redirect_stderr(io.StringIO()):
                        status = main(["--project", str(self.project), "--exec_backend", "local"])
                    self.assertEqual(status, int(fail))
                    self.assertEqual(dict(os.environ), before)

    def test_invalid_policy_fails_without_outputs(self):
        self.router.load_policy.side_effect = ValueError("invalid policy")
        with contextlib.redirect_stderr(io.StringIO()), patch("launch_scientist_bfts.run_pipeline") as run:
            self.assertEqual(main(["--project", str(self.project), "--llm-config", "bad.yaml", "--dry-run"]), 1)
        run.assert_not_called()
        self.assertFalse((self.project / "runs").exists())
