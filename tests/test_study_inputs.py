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
from ai_scientist.treesearch.agent_manager import AgentManager, Stage
from ai_scientist.treesearch.journal import Journal, Node
from ai_scientist.treesearch.log_summarization import journals_by_main_stage
from ai_scientist.treesearch.parallel_agent import MinimalAgent

FIXTURE = ROOT / "tests" / "fixtures" / "pipeline.yaml"
IDEA = {
    "Title": "t", "Abstract": "a", "Short Hypothesis": "h",
    "Experiments": ["Stage 2: tune epochs only.", "Stage 4 ablations: no noise."],
    "Risk Factors and Limitations": ["r"],
}


def manager(idea):
    cfg = OmegaConf.create({"agent": {"search": {"num_drafts": 1}, "stages": {}, "steps": 2,
                                      "max_stages": 4}})
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


class SubStageTests(unittest.TestCase):
    def test_sub_stages_keep_their_stage_number_and_share_the_budget(self):
        m = manager(IDEA)
        m.cfg.agent.stages = {"stage3_max_iters": 4}
        first = Stage(name="3_creative_research_1_first_attempt", description="",
                      goals="g", max_iterations=4, num_drafts=0, stage_number=3)
        m.stages = [first]
        m.journals = {first.name: Journal(nodes=[Node(code="", plan=""), Node(code="", plan="")])}
        m._generate_substage_goal = lambda goal, journal: ("more", "second")
        second = m._create_next_substage(first, m.journals[first.name], "")
        self.assertEqual((second.stage_number, second.max_iterations), (3, 3))
        m.stages.append(second)
        m.journals[second.name] = Journal(nodes=[Node(code="", plan="") for _ in range(3)])
        self.assertIsNone(m._create_next_substage(second, m.journals[second.name], ""))
        after = m._create_next_main_stage(second, m.journals[second.name])
        self.assertEqual((after.name.split("_")[0], after.stage_number), ("4", 4))

    def test_summaries_take_one_journal_per_main_stage(self):
        a, b, c, d = (Journal(nodes=[Node(code="", plan=str(i))]) for i in range(4))
        chosen = journals_by_main_stage([("1_x_1_a", a), ("3_y_1_a", b), ("3_y_2_b", c),
                                         ("4_z_1_a", d), ("4_z_2_b", a)])
        self.assertEqual(chosen[0], ("1_x_1_a", a))
        self.assertIsNone(chosen[1])
        self.assertEqual(chosen[2], ("3_y_2_b", c))
        self.assertEqual(len(chosen[3][1].nodes), 2)


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
