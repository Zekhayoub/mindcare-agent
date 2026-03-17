"""Carbon footprint calculation for MindCare Agent queries."""

import logging
from typing import Optional

import tiktoken

from src.config import CONFIG

logger = logging.getLogger(__name__)

_CARBON_CONFIG = CONFIG["carbon"]
_CO2_FACTORS: dict[str, float] = _CARBON_CONFIG["co2_factors"]
_SYSTEM_PROMPT_TOKENS: int = _CARBON_CONFIG["system_prompt_tokens"]
_TOOL_TOKENS_PER_CALL: int = _CARBON_CONFIG["tool_tokens_per_call"]
_ITERATION_OVERHEAD: int = _CARBON_CONFIG["iteration_overhead_tokens"]
_UNCERTAINTY_PCT: int = _CARBON_CONFIG["uncertainty_pct"]

# Approximation: cl100k_base (OpenAI) is used as a proxy for Mistral
# token counts. Mistral does not expose a public client-side tokenizer.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def calculate_co2(
    text_input: str,
    text_output: str,
    mode: str = "agent",
    intermediate_steps: Optional[list] = None,
    model: str = "mistral-large-latest",
) -> dict:
    """Calculate the CO2 footprint of a single query.

    Tokenizes input/output, adds overhead for system prompt, tool
    calls and agent iterations, then applies the appropriate CO2
    factor per token.

    Args:
        text_input: User message.
        text_output: Generated response.
        mode: "eco" for local processing, "agent" for cloud LLM.
        intermediate_steps: Agent tool call steps (used to estimate
            tool token overhead). None for eco mode.
        model: LLM model identifier for CO2 factor lookup.

    Returns:
        Dictionary with keys: total_co2, input_tokens, output_tokens,
        system_tokens, tool_tokens, iterations, total_tokens,
        uncertainty, source, mode, model. On error, includes an
        "error" key instead.
    """
    try:
        input_tokens = len(_ENCODING.encode(text_input))
        output_tokens = len(_ENCODING.encode(text_output))

        system_tokens = _SYSTEM_PROMPT_TOKENS if mode == "agent" else 0

        iterations = len(intermediate_steps) if intermediate_steps else 1
        tool_tokens = 0
        if mode == "agent" and intermediate_steps:
            tool_tokens = len(intermediate_steps) * _TOOL_TOKENS_PER_CALL
        iteration_overhead = (iterations - 1) * _ITERATION_OVERHEAD if iterations > 1 else 0

        total_tokens = input_tokens + output_tokens + system_tokens + tool_tokens + iteration_overhead

        model_key = _resolve_model_key(model)
        co2_per_token = _CO2_FACTORS.get(model_key, _CO2_FACTORS["mistral_large"])
        total_co2 = total_tokens * co2_per_token

        if mode == "eco":
            total_co2 += _CO2_FACTORS.get("eco_cpu", 0.00004)

        uncertainty = round(total_co2 * _UNCERTAINTY_PCT / 100, 6)

        source = (
            f"Mistral LCA ({model}), agent granularity"
            if mode == "agent"
            else "CodeCarbon estimate (local CPU + tokens)"
        )

        return {
            "total_co2": round(total_co2, 5),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "system_tokens": system_tokens,
            "tool_tokens": tool_tokens,
            "iterations": iterations,
            "total_tokens": total_tokens,
            "uncertainty": uncertainty,
            "source": source,
            "mode": mode,
            "model": model,
        }

    except Exception as exc:
        logger.error("CO2 calculation failed: %s", exc)
        return {
            "total_co2": 0.0,
            "uncertainty": 0.0,
            "source": "N/A",
            "error": str(exc),
        }


def _resolve_model_key(model: str) -> str:
    """Map a model identifier to a CO2 factor key.

    Args:
        model: Full model name (e.g. "mistral-large-latest").

    Returns:
        Key matching an entry in the co2_factors config.
    """
    model_lower = model.lower()
    if "large" in model_lower:
        return "mistral_large"
    if "medium" in model_lower:
        return "mistral_medium"
    if "small" in model_lower:
        return "mistral_small"
    return "mistral_large"