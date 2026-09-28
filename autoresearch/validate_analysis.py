"""Check the standard LLM analysis path against an existing experiment artifact."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

def validate(source, config, output):
    from omegaconf import OmegaConf
    from ai_scientist.treesearch.backend import query
    from ai_scientist.treesearch.interpreter import ExecutionResult, Interpreter
    from ai_scientist.treesearch.journal import Node
    from ai_scientist.treesearch.parallel_agent import MinimalAgent, metric_parse_spec

    source = Path(source).resolve()
    output = Path(output).resolve()
    for filename in ("experiment_data.npy", "experiment_code.py", "execution_result.json"):
        if not (source / filename).is_file():
            raise ValueError(f"Missing saved artifact: {source / filename}")
    output.mkdir(parents=True, exist_ok=False)
    working = output / "working"
    working.mkdir()
    shutil.copy2(source / "experiment_data.npy", working / "experiment_data.npy")
    cfg = OmegaConf.load(config)
    cfg.exec.local_postprocessing = True
    agent = MinimalAgent(
        task_desc=("Infrastructure smoke test only: two training epochs, fixed confidence threshold, "
                   "tiny synthetic low-light UAV detector. Poor scores and zero recall are acceptable. "
                   "Review execution and measurement correctness within these constraints."),
        cfg=cfg, stage_name="1_initial_implementation_1_preliminary",
    )
    node = Node(code=(source / "experiment_code.py").read_text(), plan="Saved T4 experiment")
    execution = ExecutionResult.from_dict(json.loads((source / "execution_result.json").read_text()))
    interpreter = Interpreter(output, timeout=120, agent_file_name="analysis.py")
    status = {"status": "running", "source": str(source), "checks": {}}
    previous_log = os.environ.get("AUTORESEARCH_USAGE_LOG")
    os.environ["AUTORESEARCH_USAGE_LOG"] = str(output / "usage.jsonl")

    def save():
        (output / "analysis_status.json").write_text(json.dumps(status, indent=2))

    def execute(code, name):
        (output / f"{name}.py").write_text(code)
        result = interpreter.run(code, True)
        interpreter.cleanup_session()
        (output / f"{name}_execution.json").write_text(json.dumps(asdict(result), indent=2))
        if result.exc_type:
            raise RuntimeError(f"{name} failed: {result.exc_type}: {''.join(result.term_out)}")
        return result

    save()
    try:
        print("Reviewing saved execution", flush=True)
        agent.parse_exec_result(node, execution, str(working))
        if node.is_buggy:
            raise RuntimeError(f"Execution review rejected experiment: {node.analysis}")
        status["checks"]["execution_review"] = "passed"
        save()

        print("Generating and executing metric parser", flush=True)
        _, code = agent.plan_and_code_query({
            "Task": "Write a brief plan followed by one fenced Python program. Load working/experiment_data.npy with numpy allow_pickle=True. Print final train/validation losses and final clean/dark test precision, recall, TP, FP, FN for every dataset. Use only numpy and standard library. Execute at global scope; do not retrain or generate plots.",
            "Original experiment": node.code,
        })
        result = execute(code, "metric_parser")
        if not any(line.strip() for line in result.term_out):
            raise RuntimeError("Metric parser produced no output")
        parsed = query(
            system_message={"Task": "Extract measured final metrics only.", "Output": result.term_out},
            user_message=None, func_spec=metric_parse_spec,
            model=cfg.agent.feedback.model, temperature=cfg.agent.feedback.temp,
        )
        if not parsed["valid_metrics_received"] or not parsed["metric_names"]:
            raise RuntimeError("No valid metrics extracted")
        (output / "metrics.json").write_text(json.dumps(parsed, indent=2))
        status["checks"]["metric_parser"] = "passed"
        save()

        print("Generating and executing plotting code", flush=True)
        code = agent._generate_plotting_code(node, str(working))
        execute(code, "plotting")
        node.plot_paths = [str(p.resolve()) for p in sorted(working.glob("*.png"))]
        if not node.plot_paths:
            raise RuntimeError("Plotting code produced no PNG artifacts")
        status["checks"]["plot_generation"] = "passed"
        save()

        print("Reviewing plots with the configured vision model", flush=True)
        agent._analyze_plots_with_vlm(node)
        if node.is_buggy_plots:
            raise RuntimeError("Visual review rejected generated plots")
        (output / "visual_review.json").write_text(json.dumps({
            "analyses": node.plot_analyses, "summary": node.vlm_feedback_summary,
            "datasets": node.datasets_successfully_tested,
        }, indent=2))
        status["checks"]["visual_review"] = "passed"
        status["status"] = "completed"
    except BaseException as exc:
        status["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        status["error"] = str(exc)
        raise
    finally:
        interpreter.cleanup_session()
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
        if previous_log is None:
            os.environ.pop("AUTORESEARCH_USAGE_LOG", None)
        else:
            os.environ["AUTORESEARCH_USAGE_LOG"] = previous_log
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Saved experiment result directory")
    parser.add_argument("--config", type=Path, default=Path("configs/uav_lowlight_t4_smoke.yaml"))
    parser.add_argument("--output", type=Path, required=True, help="New directory for analysis checks")
    args = parser.parse_args()
    print(json.dumps(validate(args.source, args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
