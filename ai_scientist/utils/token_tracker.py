from functools import wraps
from typing import Dict, Optional, List
from collections import defaultdict
import asyncio
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import fcntl
import logging


class TokenTracker:
    def __init__(self):
        """
        Token counts for prompt, completion, reasoning, and cached.
        Reasoning tokens are included in completion tokens.
        Cached tokens are included in prompt tokens.
        Also tracks prompts, responses, and timestamps.
        We assume we get these from the LLM response, and we don't count
        the tokens by ourselves.
        """
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)

        self.MODEL_PRICES = {
            "gpt-4o-2024-11-20": {
                "prompt": 2.5 / 1000000,  # $2.50 per 1M tokens
                "cached": 1.25 / 1000000,  # $1.25 per 1M tokens
                "completion": 10 / 1000000,  # $10.00 per 1M tokens
            },
            "gpt-4o-2024-08-06": {
                "prompt": 2.5 / 1000000,  # $2.50 per 1M tokens
                "cached": 1.25 / 1000000,  # $1.25 per 1M tokens
                "completion": 10 / 1000000,  # $10.00 per 1M tokens
            },
            "gpt-4o-2024-05-13": {  # this ver does not support cached tokens
                "prompt": 5.0 / 1000000,  # $5.00 per 1M tokens
                "completion": 15 / 1000000,  # $15.00 per 1M tokens
            },
            "gpt-4o-mini-2024-07-18": {
                "prompt": 0.15 / 1000000,  # $0.15 per 1M tokens
                "cached": 0.075 / 1000000,  # $0.075 per 1M tokens
                "completion": 0.6 / 1000000,  # $0.60 per 1M tokens
            },
            "o1-2024-12-17": {
                "prompt": 15 / 1000000,  # $15.00 per 1M tokens
                "cached": 7.5 / 1000000,  # $7.50 per 1M tokens
                "completion": 60 / 1000000,  # $60.00 per 1M tokens
            },
            "o1-preview-2024-09-12": {
                "prompt": 15 / 1000000,  # $15.00 per 1M tokens
                "cached": 7.5 / 1000000,  # $7.50 per 1M tokens
                "completion": 60 / 1000000,  # $60.00 per 1M tokens
            },
            "o3-mini-2025-01-31": {
                "prompt": 1.1 / 1000000,  # $1.10 per 1M tokens
                "cached": 0.55 / 1000000,  # $0.55 per 1M tokens
                "completion": 4.4 / 1000000,  # $4.40 per 1M tokens
            },
        }

    def add_tokens(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        cached_tokens: int,
    ):
        self.token_counts[model]["prompt"] += prompt_tokens
        self.token_counts[model]["completion"] += completion_tokens
        self.token_counts[model]["reasoning"] += reasoning_tokens
        self.token_counts[model]["cached"] += cached_tokens
        ledger = os.environ.get("AUTORESEARCH_USAGE_LOG")
        if ledger:
            event = {"model": model, "prompt": prompt_tokens, "completion": completion_tokens,
                     "reasoning": reasoning_tokens, "cached": cached_tokens}
            # Workers share an append-only ledger; process-local counters otherwise disappear.
            with open(ledger, "a") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                stream.write(json.dumps(event) + "\n")
                stream.flush()
                fcntl.flock(stream, fcntl.LOCK_UN)


    def add_interaction(
        self,
        model: str,
        system_message: str,
        prompt: str,
        response: str,
        timestamp: datetime,
    ):
        """Record a single interaction with the model."""
        self.interactions[model].append(
            {
                "system_message": system_message,
                "prompt": prompt,
                "response": response,
                "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
            }
        )

    def get_interactions(self, model: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Get all interactions, optionally filtered by model."""
        if model:
            return {model: self.interactions[model]}
        return dict(self.interactions)

    def reset(self):
        """Reset all token counts and interactions."""
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)
        # self._encoders = {}

    def calculate_cost(self, model: str) -> float:
        """Calculate the cost for a specific model based on token usage."""
        if model not in self.MODEL_PRICES:
            logging.warning(f"Price information not available for model {model}")
            return None

        prices = self.MODEL_PRICES[model]
        tokens = self.token_counts[model]

        # Calculate cost for prompt and completion tokens
        if "cached" in prices:
            prompt_cost = (tokens["prompt"] - tokens["cached"]) * prices["prompt"]
            cached_cost = tokens["cached"] * prices["cached"]
        else:
            prompt_cost = tokens["prompt"] * prices["prompt"]
            cached_cost = 0
        completion_cost = tokens["completion"] * prices["completion"]

        return prompt_cost + cached_cost + completion_cost

    def get_summary(self):
        """Summarize all processes when a shared usage ledger is configured."""
        ledger = os.environ.get("AUTORESEARCH_USAGE_LOG")
        counts = self.token_counts
        if ledger and Path(ledger).exists():
            counts = defaultdict(lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0})
            with open(ledger) as stream:
                fcntl.flock(stream, fcntl.LOCK_SH)
                for line in stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for key in ("prompt", "completion", "reasoning", "cached"):
                        counts[event["model"]][key] += event.get(key, 0)
                fcntl.flock(stream, fcntl.LOCK_UN)
        summary = {}
        for model, tokens in counts.items():
            prices = self.MODEL_PRICES.get(model)
            cost = None
            if prices:
                cached = min(tokens["prompt"], tokens["cached"]) if "cached" in prices else 0
                cost = ((tokens["prompt"] - cached) * prices["prompt"] +
                        cached * prices.get("cached", 0) + tokens["completion"] * prices["completion"])
            summary[model] = {"tokens": tokens.copy(), "cost (USD)": cost}
        return summary


# Global token tracker instance
token_tracker = TokenTracker()


def record_response(result, model=None, prompt=None, system_message=None):
    """Record SDK responses without assuming a provider's optional usage fields."""
    usage = getattr(result, "usage", None)
    if usage is None:
        return  # high-level helpers return (text, history), already counted below
    model = getattr(result, "model", None) or model or "unknown"
    anthropic_usage = hasattr(usage, "input_tokens")
    if anthropic_usage:
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        prompt_tokens = (usage.input_tokens or 0) + cached + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
        completion = usage.output_tokens or 0
        reasoning = 0
        text = "\n".join(getattr(part, "text", "") for part in getattr(result, "content", []))
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
        text = "\n".join(getattr(c.message, "content", "") or "" for c in getattr(result, "choices", []))
    token_tracker.add_tokens(model, prompt_tokens, completion, reasoning, cached)
    token_tracker.add_interaction(model, system_message, prompt, text,
                                  getattr(result, "created", None) or datetime.now(timezone.utc).isoformat())


def track_token_usage(func):
    signature = inspect.signature(func)

    def record(args, kwargs, result):
        bound = signature.bind_partial(*args, **kwargs).arguments
        record_response(result, model=bound.get("model"), prompt=bound.get("prompt"),
                        system_message=bound.get("system_message"))

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        record(args, kwargs, result)
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        record(args, kwargs, result)
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
