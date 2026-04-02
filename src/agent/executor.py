"""Agent invocation with response validation.

Handles agent execution, output validation, and error recovery.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def invoke_agent(
    executor,
    user_input: str,
    chat_history: list,
    user_location: str = "Unknown",
    pre_emotion: str = "unknown",
    pre_confidence: float = 0.0,
    pre_intensity: str = "moderate",
    pre_context: str = "general",
) -> dict:
    """Invoke the agent with pre-analyzed context.

    Args:
        executor: AgentExecutor from build_agent().
        user_input: Raw user message.
        chat_history: List of LangChain message objects.
        user_location: Detected or entered city name.
        pre_emotion: Pre-classified emotion label.
        pre_confidence: Classifier confidence score.
        pre_intensity: Estimated intensity level.
        pre_context: Detected situational context.

    Returns:
        Dictionary with "output", "intermediate_steps", and
        optionally "quality_warning" or "error".
    """
    from src.agent.utils import format_chat_history

    chat_history_str = format_chat_history(chat_history)

    try:
        response = executor.invoke({
            "input": user_input,
            "chat_history": chat_history_str,
            "user_location": user_location,
            "pre_analyzed_emotion": pre_emotion,
            "pre_analyzed_confidence": f"{pre_confidence:.2f}",
            "pre_analyzed_intensity": pre_intensity,
            "pre_analyzed_context": pre_context,
        })

        output = response.get("output", "")
        steps = response.get("intermediate_steps", [])

        is_valid, reason = _validate_response(output, steps)
        if not is_valid and len(output) < 50:
            return {
                "output": (
                    f"I understand you're feeling {pre_emotion}. "
                    "Would you like to tell me more about what's happening?"
                ),
                "intermediate_steps": steps,
                "quality_warning": reason,
            }

        return response

    except Exception as exc:
        logger.error("Agent invocation failed: %s", exc)
        return {
            "output": (
                f"I'm experiencing a technical issue, but I'm here for you. "
                f"You mentioned feeling {pre_emotion}. "
                "Would you like to try rephrasing, or would you prefer "
                "some general support?"
            ),
            "intermediate_steps": [],
            "error": str(exc),
        }


def handle_parsing_error(error: str) -> str:
    """Provide a user-friendly message on agent parsing errors.

    The original version returns a generic message. This will be
    improved in a later commit to extract useful content from
    malformed ReAct output.
    """
    logger.warning("Agent parsing error: %s", error)
    return "I encountered a processing error. Let me try a different approach."


def _validate_response(output: str, intermediate_steps: list) -> tuple[bool, str]:
    """Check if the agent response meets minimum quality standards.

    Args:
        output: Agent's final answer text.
        intermediate_steps: Tool call steps from the agent.

    Returns:
        Tuple of (is_valid, reason).
    """
    if not output or len(output.strip()) < 20:
        return False, "Response too short"

    if not intermediate_steps and len(output.strip()) < 100:
        return False, "No tools were used"

    useful_keywords = [
        "advice", "technique", "suggest", "help", "support", "feel", "emotion",
    ]
    if not any(kw in output.lower() for kw in useful_keywords):
        return False, "Response lacks useful content"

    return True, "Response quality acceptable"