"""Tree-search backend for the local coding-agent CLIs (see ai_scientist/cli_llm.py)."""

import logging
import time

import jsonschema

from ai_scientist.cli_llm import (
    CLIClient,
    complete,
    json_instruction,
    parse_json_output,
)
from .utils import FunctionSpec, OutputType
from ai_scientist.utils.token_tracker import token_tracker

logger = logging.getLogger("ai-scientist")

# How often to re-ask when the reply doesn't contain valid function-call JSON.
JSON_RETRIES = 3


def get_ai_client(model: str, **_kwargs) -> CLIClient:
    return CLIClient(model)


def _func_spec_parts(func_spec) -> tuple[str, str, dict]:
    # Some call sites pass a plain OpenAI-style function dict instead of a FunctionSpec.
    if isinstance(func_spec, FunctionSpec):
        return func_spec.name, func_spec.description, func_spec.json_schema
    return func_spec["name"], func_spec.get("description", ""), func_spec["parameters"]


def query(
    system_message: str | None,
    user_message: str | list | None,
    func_spec: FunctionSpec | dict | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    model = model_kwargs["model"]

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        # user_message is either markdown text or a list of multimodal parts
        messages.append({"role": "user", "content": user_message})
    if not messages:
        raise ValueError("query() needs a system or user message")

    if func_spec is not None:
        name, description, schema = _func_spec_parts(func_spec)
        instruction = json_instruction(name, description, schema)
        last = messages[-1]
        if isinstance(last["content"], str):
            last["content"] += instruction
        else:
            last["content"] = list(last["content"]) + [
                {"type": "text", "text": instruction}
            ]

    t0 = time.time()
    in_tokens = out_tokens = 0
    feedback = None
    for attempt in range(JSON_RETRIES if func_spec is not None else 1):
        attempt_messages = messages
        if feedback:
            attempt_messages = messages + [{"role": "user", "content": feedback}]
        text, usage = complete(model, attempt_messages)
        token_tracker.add_tokens(
            model,
            usage.get("prompt", 0),
            usage.get("completion", 0),
            usage.get("reasoning", 0),
            usage.get("cached", 0),
        )
        in_tokens += usage.get("prompt", 0)
        out_tokens += usage.get("completion", 0)

        if func_spec is None:
            output = text
            break
        try:
            output = parse_json_output(text)
            jsonschema.validate(output, schema)
            break
        except (ValueError, jsonschema.ValidationError) as e:
            err = e.message if isinstance(e, jsonschema.ValidationError) else str(e)
            logger.warning(f"{model}: invalid function-call JSON ({err}), retrying")
            feedback = (
                f"Your previous reply was:\n{text}\n\nIt was rejected because: {err}\n"
                "Reply again with ONLY the corrected JSON object in a ```json block."
            )
    else:
        raise ValueError(
            f"{model} did not return valid JSON for `{name}`: {text[:500]}"
        )

    return output, time.time() - t0, in_tokens, out_tokens, {"model": model}
