"""Validate inputs before creating workspaces or making model calls."""

import copy
import json
import math
from pathlib import Path
import re

import yaml
from ai_scientist.cli_llm import PROVIDER_PRESETS
from autoresearch.llm_strategy import apply_strategy

ROOT = Path(__file__).resolve().parents[1]


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for section in ("agent", "exec", "report"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Configuration needs a {section} mapping")
    agent, execution = config["agent"], config["exec"]
    effort = config.get("codex_reasoning_effort")
    if effort is not None and effort not in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
        raise ValueError("Invalid codex_reasoning_effort")
    max_stages = agent.get("max_stages", 4)
    if isinstance(max_stages, bool) or not isinstance(max_stages, int) or not 1 <= max_stages <= 4:
        raise ValueError("agent.max_stages must be an integer between 1 and 4")
    if agent.get("type") != "parallel":
        raise ValueError("agent.type must be parallel (the supported implementation)")
    for section in ("search", "multi_seed_eval", "stages"):
        if not isinstance(agent.get(section), dict):
            raise ValueError(f"agent.{section} must be a mapping")
    counts = {
        "worker_timeout": agent.get("worker_timeout", 7200),
        "num_workers": agent.get("num_workers"),
        "steps": agent.get("steps"),
        "multi_seed_eval.num_seeds": agent["multi_seed_eval"].get("num_seeds"),
        "search.num_drafts": agent["search"].get("num_drafts"),
        **{f"stages.{k}": v for k, v in agent["stages"].items()},
    }
    for label, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"agent.{label} must be a positive integer")
    if not isinstance(execution.get("local_postprocessing", False), bool):
        raise ValueError("exec.local_postprocessing must be a boolean")
    if not isinstance(agent.get("smoke_test", False), bool):
        raise ValueError("agent.smoke_test must be a boolean")
    depth = agent["search"].get("max_debug_depth")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError("agent.search.max_debug_depth must be a nonnegative integer")
    for key, default in (
        ("timeout", 3600),
        ("remote_max_file_mb", 100),
        ("remote_wait_minutes", 60),
    ):
        value = execution.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (value == 0 and key != "remote_wait_minutes")
        ):
            raise ValueError(
                f"exec.{key} must be finite and {'nonnegative' if key == 'remote_wait_minutes' else 'positive'}"
            )
    if execution.get("backend", "local") not in ("local", "colab"):
        raise ValueError("exec.backend must be local or colab")
    if execution.get("backend") == "colab" and (
        execution.get("timeout", 3600) > 86400
        or execution.get("remote_max_file_mb", 100) > 512
    ):
        raise ValueError(
            "Colab timeout is limited to 86400 seconds and remote_max_file_mb to 512"
        )
    if config.get("exp_name") != "run":
        raise ValueError(
            "exp_name must be run for the paper pipeline; use --output-dir to choose a location"
        )
    filename = execution.get("agent_file_name", "runfile.py")
    if not isinstance(filename, str) or not re.fullmatch(r"[\w.-]+\.py", filename):
        raise ValueError(
            "exec.agent_file_name must be a Python filename without directories"
        )
    probability = agent["search"].get("debug_prob")
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not 0 <= probability <= 1
    ):
        raise ValueError("agent.search.debug_prob must be between 0 and 1")
    for role in ("code", "feedback", "vlm_feedback", "summary", "select_node", "orchestrator"):
        setting = agent.get(role)
        if role in ("summary", "select_node", "orchestrator") and setting is None:
            continue
        if (
            not isinstance(setting, dict)
            or not isinstance(setting.get("model"), str)
            or not setting["model"].strip()
        ):
            raise ValueError(f"agent.{role}.model must be a nonempty string")
    if (
        not isinstance(config["report"].get("model"), str)
        or not config["report"]["model"].strip()
    ):
        raise ValueError("report.model must be a nonempty string")
    return config


def load_run_config(path, provider=None, backend=None, workers=None, orchestrator=None, worker=None):
    with Path(path).open() as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    config = copy.deepcopy(config)
    if provider:
        preset = PROVIDER_PRESETS[provider]
        agent = config.setdefault("agent", {})
        if agent.get("orchestrator"):
            agent["orchestrator"]["model"] = preset["big"]
        for role in ("code", "feedback", "vlm_feedback", "summary", "select_node"):
            setting = agent.setdefault(role, {"temp": 0.3})
            setting["model"] = preset["big" if role == "code" else "small"]
        config.setdefault("report", {})["model"] = preset["small"]
    if backend:
        config.setdefault("exec", {})["backend"] = backend
    if workers is not None:
        config.setdefault("agent", {})["num_workers"] = workers
    return validate_config(apply_strategy(config, orchestrator, worker))


def load_idea(path, index):
    with Path(path).open() as source:
        ideas = json.load(source)
    if not isinstance(ideas, list) or not ideas:
        raise ValueError("Ideas file must contain a nonempty JSON array")
    if not 0 <= index < len(ideas):
        raise ValueError(f"idea_idx must be between 0 and {len(ideas) - 1}")
    idea = ideas[index]
    if (
        not isinstance(idea, dict)
        or not isinstance(idea.get("Name"), str)
        or not idea["Name"].strip()
    ):
        raise ValueError("Selected idea must be an object with a nonempty Name")
    return idea


def idea_slug(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:100] or "research"
