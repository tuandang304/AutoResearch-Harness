"""Study-specific inputs: stage goals, experiment plan, seed stages, support files."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from omegaconf import OmegaConf
import yaml

from autoresearch.config import ROOT, load_run_config, validate_config
from ai_scientist.treesearch.agent_manager import AgentManager
from ai_scientist.treesearch.parallel_agent import MinimalAgent

FIXTURE = ROOT / "tests" / "fixtures" / "pipeline.yaml"
IDEA = {
    "Title": "t", "Abstract": "a", "Short Hypothesis": "h",
    "Experiments": ["Stage 2: tune epochs only.", "Stage 4 ablations: no noise."],
    "Risk Factors and Limitations": ["r"],
}


def manager(idea):
    cfg = OmegaConf.create({"agent": {"search": {"num_drafts": 1}, "stages": {}, "steps": 2}})
    with tempfile.TemporaryDirectory() as root:
        return AgentManager(json.dumps(idea), cfg, Path(root))


class StageInputTests(unittest.TestCase):
    def test_stage_goals_override_generic_goals(self):
        goals = manager({**IDEA, "Stage Goals": {"2": "Use VisDrone only."}}).main_stage_goals
        self.assertEqual(goals[2], "Use VisDrone only.")
        self.assertIn("HuggingFace", goals[3])  # stages without an override keep defaults
        for bad in ({"5": "x"}, {"2": ""}, ["x"]):
            with self.assertRaises(ValueError):
                manager({**IDEA, "Stage Goals": bad})

    def test_every_stage_sees_the_experiment_plan(self):
        m = manager(IDEA)
        for name in ("1_initial_implementation_1_preliminary", "2_baseline_tuning_1_first",
                     "3_creative_research_1_first", "4_ablation_studies_1_first"):
            desc = m._curate_task_desc(SimpleNamespace(name=name))
            self.assertIn("Stage 4 ablations: no noise.", desc)
            self.assertEqual("Risk Factors" in desc, name.startswith("4_"))


class ConfigTests(unittest.TestCase):
    def base(self):
        return yaml.safe_load(FIXTURE.read_text())

    def test_seed_stages_are_validated(self):
        config = self.base()
        config["agent"]["multi_seed_eval"]["stages"] = [3]
        validate_config(config)
        for bad in ([5], "3", [True]):
            config["agent"]["multi_seed_eval"]["stages"] = bad
            with self.assertRaisesRegex(ValueError, "multi_seed_eval.stages"):
                validate_config(copy.deepcopy(config))

    def test_support_files_resolve_next_to_the_config_and_are_checked(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "helper.py").write_text("X = 1\n")
            config = self.base()
            config["exec"]["support_files"] = ["helper.py"]
            path = Path(root) / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            loaded = load_run_config(path)
            self.assertEqual(loaded["exec"]["support_files"],
                             [str((Path(root) / "helper.py").resolve())])
            config["exec"]["support_files"] = ["missing.py"]
            path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "not a file"):
                load_run_config(path)

    def test_code_prompts_name_support_files(self):
        cfg = SimpleNamespace(exec=SimpleNamespace(backend="local", support_files=["/x/uavlib.py"]))
        self.assertIn("uavlib.py", MinimalAgent("task", cfg)._prompt_compute["Support files"])
        cfg.exec.support_files = None
        self.assertEqual(MinimalAgent("task", cfg)._prompt_compute, {})


if __name__ == "__main__":
    unittest.main()
