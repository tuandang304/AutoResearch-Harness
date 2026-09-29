"""Opt-in live CLI probes; emit metadata only, never raw replies or errors."""

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
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_scientist import cli_llm
from autoresearch.llm_router import load_policy


PROFILES = list(dict.fromkeys(
    (f"{profile['provider']}/{profile['model']}", profile["effort"])
    for profile in load_policy(Path(__file__).resolve().parents[1] / "configs/llm.yaml")["profiles"].values()
))


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize logged-in CLI model calls")
    parser.add_argument("--vision", action="store_true", help="Include a generated shape/color image probe")
    parser.add_argument("--output", type=Path, help="Explicit path for sanitized JSON evidence")
    parser.add_argument("--provider", choices=cli_llm.CLI_PROVIDERS, help="Probe only this provider")
    parser.add_argument("--model", choices=sorted({p[0] for p in PROFILES}), help="Probe only one exact provider/model ID")
    args = parser.parse_args()
    if not args.live:
        parser.error("Live verification requires --live (may incur provider usage).")
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
    profiles = [p for p in PROFILES if (not args.provider or p[0].split("/", 1)[0] == args.provider)
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
    }
    serialized = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)
    return 0 if all(r["status"] == "passed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
