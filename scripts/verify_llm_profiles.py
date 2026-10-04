"""Validate the configured LLM profiles.

Without --live: offline preflight only (policy schema, CLI binaries on PATH and
role coverage). It never calls a model or opens router state.
With --live: opt-in CLI probes; emit metadata only, never raw replies or errors.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import hashlib
import io
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai_scientist import cli_llm
from autoresearch.llm_router import load_policy


DEFAULT_POLICY = ROOT / "configs/llm.yaml"
# Executables invoked by the ai_scientist.cli_llm runners.
CLI_BINARIES = {"claude-code": "claude", "codex": "codex", "antigravity": "agy"}
ROLE_MARKS = {"ok": "PASS", "degraded": "WARN", "untested": "SKIP", "unavailable": "FAIL"}


def policy_profiles(policy):
    """Unique (provider/model, effort) pairs; identical pairs are probed once."""
    return list(dict.fromkeys(
        (f"{profile['provider']}/{profile['model']}", profile["effort"])
        for profile in policy["profiles"].values()
    ))


PROFILES = policy_profiles(load_policy(DEFAULT_POLICY))


def role_requirements(role):
    """Mirror complete_routed: text+json always; the vision role receives images."""
    return {"text", "json", "vision"} if role == "vision" else {"text", "json"}


def summarize_roles(policy, candidate_state):
    """Per-role coverage. candidate_state(alias, required) returns "ok" or a reason.

    ok: every capable candidate works; degraded: only some do (the router must
    fall back or the selector may pick a failing worker); untested: nothing works
    yet but some candidates were not probed; unavailable: no candidate works.
    """
    summary = {}
    for role, setting in policy["roles"].items():
        required = role_requirements(role)
        states = {}
        for alias in setting["candidates"]:
            if not required <= set(policy["profiles"][alias]["capabilities"]):
                states[alias] = "not_capable"
            else:
                states[alias] = candidate_state(alias, required)
        capable = [s for s in states.values() if s != "not_capable"]
        working = [s for s in capable if s == "ok"]
        failed = [s for s in capable if s not in ("ok", "untested")]
        if capable and len(working) == len(capable):
            status = "ok"
        elif working and failed:
            status = "degraded"
        elif "untested" in capable:
            status = "untested"
        else:
            status = "unavailable"
        summary[role] = {"selection": setting["selection"], "status": status, "candidates": states}
    return summary


def preflight_state(policy, which=None):
    """Offline: a candidate is "ok" when its CLI is on PATH. No model is called."""
    which = which or shutil.which
    found = {provider: which(binary) is not None for provider, binary in CLI_BINARIES.items()}

    def state(alias, required):
        return "ok" if found[policy["profiles"][alias]["provider"]] else "missing_cli"
    return found, state


def live_state(policy, results):
    """A candidate is "ok" only when every probe its role needs passed."""
    checks = {}
    for record in results:
        checks.setdefault((record["model"], record["effort"]), {})[record["check"]] = record["status"]

    def state(alias, required):
        profile = policy["profiles"][alias]
        got = checks.get((f"{profile['provider']}/{profile['model']}", profile["effort"]), {})
        needed = ["text", "structured"] + (["vision"] if "vision" in required else [])
        failed = [got[c] for c in needed if c in got and got[c] != "passed"]
        if failed:
            return failed[0]
        return "ok" if all(c in got for c in needed) else "untested"
    return state


def render_roles(roles, header, stream):
    print(header, file=stream)
    width = max(map(len, roles))
    for role, info in roles.items():
        candidates = ", ".join(f"{alias}={state}" for alias, state in info["candidates"].items())
        print(f"  [{ROLE_MARKS[info['status']]}] {role:<{width}} ({info['selection']}): {candidates}", file=stream)


def render_profiles(results, stream):
    """Statuses are allowlisted categories, so they are safe to print."""
    rows = {}
    for record in results:
        rows.setdefault((record["model"], record["effort"]), {})[record["check"]] = record["status"]
    print("Model probes:", file=stream)
    for (model, effort), checks in sorted(rows.items()):
        cells = "  ".join(f"{kind}={checks.get(kind, '-')}" for kind in ("text", "structured", "vision"))
        print(f"  {model} ({effort}): {cells}", file=stream)


def error_category(error):
    """Allowlisted categories only: exception strings may contain credentials."""
    detail = (str(error) + " " + str(getattr(error, "metadata", {}).get("error_code", ""))).lower()
    for category, markers in (
        ("startup_timeout", ("hung at startup",)),
        ("timeout", ("timed out",)),
        ("missing_cli", ("was not found on path",)),
        ("authentication", ("not logged in", "authentication", "unauthorized")),
        ("model_unavailable", ("model not found", "unknown model", "invalid model", "does not exist", "not available", "not have access")),
        ("rate_limit", ("rate limit", "rate_limit",)),
        ("unsupported_option", ("unknown option", "unrecognized argument",)),
    ):
        if any(marker in detail for marker in markers):
            return category
    return "cli_error"


def identity_evidence(model, reported):
    if reported is None:
        return "request_accepted_only"
    if reported in (model, model.split("/", 1)[1]):
        return "exact_metadata_match"
    return "model_mismatch"


def vision_asset():
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (360, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 40, 100, 120), fill="red")
    draw.ellipse((140, 40, 220, 120), fill="green")
    draw.polygon([(290, 35), (245, 125), (335, 125)], fill="blue")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    return "data:image/png;base64," + base64.b64encode(data).decode(), {
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": 360, "height": 160,
        "expected": {"shape": "circle", "color": "green"},
        "generator": "Pillow: red square left, green circle center, blue triangle right",
    }


def probe(profile, image_url=None):
    model, effort = profile
    records = []
    checks = [
        ("text", "Reply with exactly: PROFILE_OK"),
        ("structured", 'Return only a JSON object with keys "ok" (true) and "sum" (7, the sum of 3 and 4).'),
    ]
    if image_url:
        checks.append(("vision", [
            {"type": "text", "text": 'Identify the middle shape in the attached image. Return only JSON with keys "shape" and "color", using lowercase words.'},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]))
    for kind, prompt in checks:
        start = time.monotonic()
        record = {"model": model, "effort": effort, "check": kind,
                  "tested_at": datetime.now(timezone.utc).isoformat()}
        for attempt in range(2):
            record["attempts"] = attempt + 1
            try:
                text, usage = cli_llm.complete(
                    model, [{"role": "user", "content": prompt}],
                    effort=effort, max_attempts=1,
                )
                if kind == "text":
                    valid = text.strip() == "PROFILE_OK"
                elif kind == "structured":
                    valid = cli_llm.parse_json_output(text) == {"ok": True, "sum": 7}
                else:
                    valid = cli_llm.parse_json_output(text) == {"shape": "circle", "color": "green"}
                reported = usage.get("reported_model")
                # Only simple model IDs may reach stdout; never emit arbitrary metadata.
                safe = isinstance(reported, str) and re.fullmatch(r"[A-Za-z0-9._:/-]{1,160}", reported)
                identity = identity_evidence(model, reported)
                record.update(
                    status="model_mismatch" if identity == "model_mismatch" else "passed" if valid else "invalid_response",
                    reported_model=reported if safe else None,
                    identity_evidence=identity,
                    usage={key: usage.get(key, 0) for key in ("prompt", "completion", "cached", "reasoning")},
                )
                break
            except cli_llm.CLIError as error:
                startup = "hung at startup" in str(error)
                if startup and attempt == 0:
                    continue
                record.update(status=error_category(error))
                if error.metadata:
                    record["error_metadata"] = error.metadata
                break
            except ValueError:
                record.update(status="invalid_response")
                break
        record["elapsed_seconds"] = round(time.monotonic() - start, 2)
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm-config", type=Path,
                        help="Policy to validate (default: $AUTORESEARCH_LLM_CONFIG, else configs/llm.yaml)")
    parser.add_argument("--live", action="store_true", help="Authorize logged-in CLI model calls")
    parser.add_argument("--vision", action="store_true", help="Include a generated shape/color image probe")
    parser.add_argument("--output", type=Path, help="Explicit path for sanitized JSON evidence")
    parser.add_argument("--provider", choices=cli_llm.CLI_PROVIDERS, help="Probe only this provider")
    parser.add_argument("--model", help="Probe only one exact provider/model ID from the policy")
    args = parser.parse_args()
    policy_path = args.llm_config or Path(os.environ.get("AUTORESEARCH_LLM_CONFIG") or DEFAULT_POLICY)
    try:
        policy = load_policy(policy_path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(f"Invalid LLM policy {policy_path}: {error}")
    all_profiles = policy_profiles(policy)
    model_ids = sorted({p[0] for p in all_profiles})
    if args.model and args.model not in model_ids:
        parser.error(f"--model must be one of: {', '.join(model_ids)}")
    if not args.live:
        if args.output or args.vision:
            parser.error("--output and --vision record live probes; add --live (may incur provider usage)")
        found, state = preflight_state(policy)
        roles = summarize_roles(policy, state)
        print(f"Offline preflight of {policy_path} (no model calls).")
        for provider in sorted({p["provider"] for p in policy["profiles"].values()}):
            status = "found" if found[provider] else "MISSING from PATH"
            print(f"  CLI {CLI_BINARIES[provider]:<6} ({provider}): {status}")
        render_roles(roles, "Role coverage (ok = CLI installed; models not yet tested):", sys.stdout)
        print("Run with --live [--vision] to send real test prompts (incurs provider usage).")
        return 0 if all(r["status"] != "unavailable" for r in roles.values()) else 1
    owner = "scripts/verify_llm_profiles.py:v1"
    existing = {}
    if args.output and args.output.exists():
        try:
            existing = json.loads(args.output.read_text())
        except (ValueError, OSError):
            parser.error("Refusing to overwrite an existing non-test artifact")
        if not isinstance(existing, dict) or existing.get("artifact_owner") != owner:
            parser.error("Refusing to overwrite an artifact not owned by this verifier")
    # Process-local timeout configuration, fixed before starting parallel calls.
    cli_llm.CLI_TIMEOUT = 120
    cli_llm.CLI_STARTUP_TIMEOUT = 45
    logging.getLogger("ai-scientist").disabled = True
    image_url, asset = vision_asset() if args.vision else (None, None)
    profiles = [p for p in all_profiles if (not args.provider or p[0].split("/", 1)[0] == args.provider)
                and (not args.model or p[0] == args.model)]
    if not profiles:
        parser.error("Provider and model filters select no profiles")
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = [record for records in pool.map(lambda p: probe(p, image_url), profiles) for record in records]
    if (args.provider or args.model) and existing:
        # A targeted rerun replaces this provider only, retaining prior evidence.
        tested = {p[0] for p in profiles}
        retained = [r for r in existing.get("results", []) if r["model"] not in tested]
        results = results + retained
    payload = {
        "artifact_owner": owner,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "timeout_seconds": 120, "startup_retry_limit": 1,
        "CLAUDECODE_present": bool(os.environ.get("CLAUDECODE")),
        "vision_asset": asset, "results": results,
        "policy_path": str(policy_path),
        "role_coverage": summarize_roles(policy, live_state(policy, results)),
    }
    serialized = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)
    # Human summary on stderr keeps stdout machine-readable JSON.
    render_profiles(results, sys.stderr)
    render_roles(payload["role_coverage"], "Role coverage (access/interface only, not quality):", sys.stderr)
    return 0 if all(r["status"] == "passed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
