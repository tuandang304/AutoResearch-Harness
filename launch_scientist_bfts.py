import os.path as osp
import json
import argparse
import shutil
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import tempfile

from autoresearch.config import ROOT, load_run_config, load_idea, idea_slug
from ai_scientist.cli_llm import PROVIDER_PRESETS
from contextlib import contextmanager


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def save_token_tracker(idea_dir):
    from ai_scientist.utils.token_tracker import token_tracker
    write_json(Path(idea_dir) / "token_tracker.json", token_tracker.get_summary())
    write_json(Path(idea_dir) / "token_tracker_interactions.json", token_tracker.get_interactions())


# Defaults for the per-stage model flags when --provider is not given.
DEFAULT_MODELS = {
    "model_agg_plots": "o3-mini-2025-01-31",
    "model_writeup": "o1-preview-2024-09-12",
    "model_citation": "gpt-4o-2024-11-20",
    "model_writeup_small": "gpt-4o-2024-05-13",
    "model_review": "gpt-4o-2024-11-20",
}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="AutoResearch-Harness: experiments, papers and review")
    parser.add_argument("--config", type=Path, default=ROOT / "bfts_config.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments")
    parser.add_argument("--num-workers", type=int, help="Override agent.num_workers")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the resolved run without model calls or experiments")
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=sorted(PROVIDER_PRESETS),
        help="Use a local, logged-in CLI (Claude Code, Codex or Antigravity) for every "
        "LLM call instead of API keys. Explicit --model_* flags still take precedence.",
    )
    parser.add_argument(
        "--exec_backend",
        type=str,
        default=None,
        choices=["local", "colab"],
        help="Where experiment code runs (overrides exec.backend in bfts_config.yaml). "
        "'colab' uses the remote GPU executor from notebooks/colab_gpu_executor.ipynb.",
    )
    parser.add_argument(
        "--writeup-type",
        type=str,
        default="icbinb",
        choices=["normal", "icbinb"],
        help="Type of writeup to generate (normal=8 page, icbinb=4 page)",
    )
    parser.add_argument(
        "--load_ideas",
        type=str,
        default=str(ROOT / "examples" / "ideas.json"),
        help="Path to a JSON file containing pregenerated ideas",
    )
    parser.add_argument(
        "--load_code",
        action="store_true",
        help="If set, load a Python file with same name as ideas file but .py extension",
    )
    parser.add_argument(
        "--idea_idx",
        type=int,
        default=0,
        help="Index of the idea to run",
    )
    parser.add_argument(
        "--add_dataset_ref",
        action="store_true",
        help="If set, add a HF dataset reference to the idea",
    )
    parser.add_argument(
        "--writeup-retries",
        type=int,
        default=3,
        help="Number of writeup attempts to try",
    )
    parser.add_argument(
        "--attempt_id",
        type=int,
        default=0,
        help="Attempt ID, used to distinguish same idea in different attempts in parallel runs",
    )
    parser.add_argument(
        "--model_agg_plots",
        type=str,
        default=None,
        help="Model to use for plot aggregation",
    )
    parser.add_argument(
        "--model_writeup",
        type=str,
        default=None,
        help="Model to use for writeup",
    )
    parser.add_argument(
        "--model_citation",
        type=str,
        default=None,
        help="Model to use for citation gathering",
    )
    parser.add_argument(
        "--num_cite_rounds",
        type=int,
        default=20,
        help="Number of citation rounds to perform",
    )
    parser.add_argument(
        "--model_writeup_small",
        type=str,
        default=None,
        help="Smaller model to use for writeup",
    )
    parser.add_argument(
        "--model_review",
        type=str,
        default=None,
        help="Model to use for review main text and captions",
    )
    parser.add_argument(
        "--skip_writeup",
        action="store_true",
        help="If set, skip the writeup process",
    )
    parser.add_argument(
        "--skip_review",
        action="store_true",
        help="If set, skip the review process",
    )
    args = parser.parse_args(argv)
    if args.writeup_retries < 1 or args.num_cite_rounds < 0 or args.attempt_id < 0:
        parser.error("writeup-retries must be positive; citation rounds and attempt_id must be nonnegative")
    return args


def get_available_gpus(gpu_ids=None):
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    try:
        import torch
    except ImportError:  # not needed locally when experiments run on Colab
        return []
    return list(range(torch.cuda.device_count()))


def find_pdf_path_for_review(idea_dir):
    """Select final, latest numbered reflection, or base PDF deterministically."""
    files = list(Path(idea_dir).glob("*.pdf"))
    if not files:
        return None

    def rank(path):
        match = re.search(r"reflection[_.]?(\d+)", path.stem)
        return ("final" in path.stem.lower(), int(match[1]) if match else -1, path.stat().st_mtime_ns, path.name)

    return str(max(files, key=rank))


@contextmanager
def redirect_stdout_stderr_to_file(log_file_path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log = open(log_file_path, "a")
    sys.stdout = log
    sys.stderr = log
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log.close()


def run_pipeline(args, idea, config):
    from ai_scientist.llm import create_client
    from ai_scientist.cli_llm import PROVIDER_PRESETS
    
    from contextlib import contextmanager
    from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
        perform_experiments_bfts,
    )
    from ai_scientist.treesearch.bfts_utils import (
        idea_to_markdown,
        edit_bfts_config_file,
        apply_run_overrides,
    )
    from ai_scientist.perform_plotting import aggregate_plots
    from ai_scientist.perform_writeup import perform_writeup
    from ai_scientist.perform_icbinb_writeup import (
        perform_writeup as perform_icbinb_writeup,
        gather_citations,
    )
    from ai_scientist.perform_llm_review import perform_review, load_paper
    from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
    from ai_scientist.utils.token_tracker import token_tracker
    
    os.environ["AI_SCIENTIST_ROOT"] = os.path.dirname(os.path.abspath(__file__))
    print(f"Set AI_SCIENTIST_ROOT to {os.environ['AI_SCIENTIST_ROOT']}")

    # Check available GPUs and adjust parallel processes if necessary
    available_gpus = get_available_gpus()
    print(f"Using GPUs: {available_gpus}")

    ideas = [idea]
    args.idea_idx = 0

    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    idea_dir = str(args.output_dir / f"{date}_{idea_slug(idea['Name'])}_attempt_{args.attempt_id}")
    args.run_dir = Path(idea_dir)
    print(f"Results will be saved in {idea_dir}")
    os.makedirs(idea_dir, exist_ok=False)
    os.environ["AUTORESEARCH_USAGE_LOG"] = str(Path(idea_dir) / "usage.jsonl")

    # Convert idea json to markdown file
    idea_path_md = osp.join(idea_dir, "idea.md")

    # If load_code is True, get the Python file with same name as JSON
    code = None
    if args.load_code:
        code_path = args.load_ideas.rsplit(".", 1)[0] + ".py"
        if os.path.exists(code_path):
            with open(code_path, "r") as f:
                code = f.read()
        else:
            print(f"Warning: Code file {code_path} not found")
    else:
        code_path = None

    idea_to_markdown(ideas[args.idea_idx], idea_path_md, code_path)

    dataset_ref_code = None
    if args.add_dataset_ref:
        dataset_ref_path = "hf_dataset_reference.py"
        if os.path.exists(dataset_ref_path):
            with open(dataset_ref_path, "r") as f:
                dataset_ref_code = f.read()
        else:
            print(f"Warning: Dataset reference file {dataset_ref_path} not found")
            dataset_ref_code = None

    if dataset_ref_code is not None and code is not None:
        added_code = dataset_ref_code + "\n" + code
    elif dataset_ref_code is not None and code is None:
        added_code = dataset_ref_code
    elif dataset_ref_code is None and code is not None:
        added_code = code
    else:
        added_code = None

    print(added_code)

    # Add code to idea json if it was loaded
    if added_code is not None:
        ideas[args.idea_idx]["Code"] = added_code

    # Store raw idea json
    idea_path_json = osp.join(idea_dir, "idea.json")
    with open(idea_path_json, "w") as f:
        json.dump(ideas[args.idea_idx], f, indent=4)

    config_path = args.config
    idea_config_path = edit_bfts_config_file(
        config_path,
        idea_dir,
        idea_path_json,
    )
    if args.provider or args.exec_backend:
        preset = PROVIDER_PRESETS.get(args.provider, {})
        apply_run_overrides(
            idea_config_path,
            big_model=preset.get("big"),
            small_model=preset.get("small"),
            exec_backend=args.exec_backend,
        )

    import yaml
    with open(idea_config_path) as source:
        run_config = yaml.safe_load(source)
    config.update({key: run_config[key] for key in ("desc_file", "workspace_dir", "data_dir", "log_dir")})
    with open(idea_config_path, "w") as dest:
        yaml.safe_dump(config, dest)
    update_run_status(args, "running", stage="experiments")
    perform_experiments_bfts(idea_config_path)
    experiment_results_dir = osp.join(idea_dir, "logs/0-run/experiment_results")
    if not os.path.isdir(experiment_results_dir) or not os.listdir(experiment_results_dir):
        raise RuntimeError("No experiment results were produced; inspect the tree-search logs")
    if os.path.exists(experiment_results_dir):
        shutil.copytree(
            experiment_results_dir,
            osp.join(idea_dir, "experiment_results"),
            dirs_exist_ok=True,
        )

    update_run_status(args, "running", stage="plotting")
    aggregate_plots(base_folder=idea_dir, model=args.model_agg_plots)

    shutil.rmtree(osp.join(idea_dir, "experiment_results"))

    save_token_tracker(idea_dir)

    if not args.skip_writeup:
        writeup_success = False
        update_run_status(args, "running", stage="writeup")
        citations_text = None
        if args.writeup_type == "icbinb":
            citations_text = gather_citations(
                idea_dir,
                num_cite_rounds=args.num_cite_rounds,
                small_model=args.model_citation,
            )
        for attempt in range(args.writeup_retries):
            print(f"Writeup attempt {attempt+1} of {args.writeup_retries}")
            if args.writeup_type == "normal":
                writeup_success = perform_writeup(
                    base_folder=idea_dir,
                    small_model=args.model_writeup_small,
                    big_model=args.model_writeup,
                    page_limit=8,
                    num_cite_rounds=args.num_cite_rounds,
                )
            else:
                writeup_success = perform_icbinb_writeup(
                    base_folder=idea_dir,
                    small_model=args.model_writeup_small,
                    big_model=args.model_writeup,
                    page_limit=4,
                    citations_text=citations_text,
                )
            if writeup_success:
                break

        if not writeup_success:
            raise RuntimeError("Writeup did not complete successfully after all retries")

    save_token_tracker(idea_dir)

    if not args.skip_review and not args.skip_writeup:
        update_run_status(args, "running", stage="review")
        # Perform paper review if the paper exists
        pdf_path = find_pdf_path_for_review(idea_dir)
        if pdf_path is None:
            raise RuntimeError("Writeup reported success but produced no PDF")
        if os.path.exists(pdf_path):
            print("Paper found at: ", pdf_path)
            paper_content = load_paper(pdf_path)
            client, client_model = create_client(args.model_review)
            review_text = perform_review(paper_content, client_model, client)
            review_img_cap_ref = perform_imgs_cap_ref_review(
                client, client_model, pdf_path
            )
            with open(osp.join(idea_dir, "review_text.txt"), "w") as f:
                f.write(json.dumps(review_text, indent=4))
            with open(osp.join(idea_dir, "review_img_cap_ref.json"), "w") as f:
                json.dump(review_img_cap_ref, f, indent=4)
            print("Paper review completed.")

    return idea_dir


def write_json(path, value):
    """Atomically write metadata, including after interruptions."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def update_run_status(args, status, **details):
    if getattr(args, "run_dir", None) is not None:
        path = args.run_dir / "run_status.json"
        state = json.loads(path.read_text()) if path.exists() else {"started_at": datetime.now(timezone.utc).isoformat()}
        state.update(status=status, **details)
        if status != "running":
            state["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(path, state)


def cleanup_children(previous):
    """Stop only descendants created by this run; never scan other processes."""
    import psutil
    children = [child for child in psutil.Process().children(recursive=True) if child not in previous]
    for child in children:
        try:
            child.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(children, timeout=3)
    for child in alive:
        try:
            child.kill()
        except psutil.Error:
            pass


def main(argv=None):
    args = parse_arguments(argv)
    previous = None
    previous_cwd = Path.cwd()
    previous_usage_log = os.environ.get("AUTORESEARCH_USAGE_LOG")
    try:
        args.load_ideas = str(Path(args.load_ideas).resolve())
        args.config = args.config.resolve()
        args.output_dir = args.output_dir.resolve()
        config = load_run_config(args.config, args.provider, args.exec_backend, args.num_workers)
        idea = load_idea(args.load_ideas, args.idea_idx)
        for name in DEFAULT_MODELS:
            if getattr(args, name) is None:
                role = "code" if name == "model_writeup" else "feedback"
                setattr(args, name, config["agent"][role]["model"])
        if args.load_code and not Path(args.load_ideas).with_suffix(".py").is_file():
            raise ValueError("--load_code needs a Python file next to the ideas JSON")
        if args.add_dataset_ref and not (ROOT / "hf_dataset_reference.py").is_file():
            raise ValueError("--add_dataset_ref needs hf_dataset_reference.py in the repository")
        if args.dry_run:
            print(json.dumps({"idea": idea["Name"], "output_dir": str(args.output_dir), "config": config,
                              "models": {k: v for k, v in vars(args).items() if k.startswith("model_")}}, indent=2))
            return 0
        os.environ["AI_SCIENTIST_ROOT"] = str(ROOT)
        if config["exec"].get("backend") == "colab":
            from ai_scientist.treesearch.remote_interpreter import load_remote_config
            load_remote_config()
        import psutil
        previous = set(psutil.Process().children(recursive=True))
        os.chdir(ROOT)  # legacy templates use repository-relative paths
        run_pipeline(args, idea, config)
        update_run_status(args, "completed")
        return 0
    except KeyboardInterrupt:
        update_run_status(args, "interrupted")
        print("Run interrupted; outputs retained.", file=sys.stderr)
        return 130
    except Exception as exc:
        update_run_status(args, "failed", error=str(exc))
        print(f"AutoResearch-Harness: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            if getattr(args, "run_dir", None) is not None:
                save_token_tracker(args.run_dir)
        finally:
            if previous is not None:
                cleanup_children(previous)
            os.chdir(previous_cwd)
            if previous_usage_log is None:
                os.environ.pop("AUTORESEARCH_USAGE_LOG", None)
            else:
                os.environ["AUTORESEARCH_USAGE_LOG"] = previous_usage_log


if __name__ == "__main__":
    raise SystemExit(main())
