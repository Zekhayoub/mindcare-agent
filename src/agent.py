"""LangChain ReAct agent with tool definitions and lazy initialization."""

import logging
import re
from typing import Optional

from src.config import CONFIG

logger = logging.getLogger(__name__)

_AGENT_CONFIG = CONFIG["agent"]


def build_agent(tools_instance, config: Optional[dict] = None):
    """Build and return a configured AgentExecutor.

    The agent is not created at import time — only when this
    function is explicitly called. This avoids crashes if the
    API key is missing or LangChain is not installed.

    Args:
        tools_instance: A MindCareTools instance to bind tools to.
        config: Configuration dictionary. Defaults to global CONFIG.

    Returns:
        A LangChain AgentExecutor ready to invoke.

    Raises:
        ValueError: If MISTRAL_API_KEY is not set.
    """
    import os

    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import tool
    from langchain_core.prompts import PromptTemplate
    from langchain_mistralai import ChatMistralAI

    config = config or CONFIG
    agent_cfg = config["agent"]

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY not found in environment variables. "
            "Copy .env.example to .env and add your key."
        )

    llm = ChatMistralAI(
        api_key=api_key,
        model=agent_cfg["model_name"],
        temperature=agent_cfg["temperature"],
    )

    # ------------------------------------------------------------------
    # Tool definitions (bound to tools_instance via closure)
    # ------------------------------------------------------------------

    @tool
    def analyze_emotion(text: str) -> str:
        """Useful to analyze the user's emotion and confidence level."""
        result = tools_instance.classify_emotion(text)
        return str(result)

    @tool
    def get_advice(emotion: str, intensity: str = "moderate", context: str = "general") -> str:
        """Useful to get psychological advice based on an emotion, intensity, and context."""
        advice, note = tools_instance.get_advice(emotion, intensity=intensity, context=context)
        return f"Advice: {advice}\nNote: {note}"

    @tool
    def get_activity(emotion: str, location: str = "Brussels") -> str:
        """Useful to find a location-based activity matching the emotion."""
        result = tools_instance.get_activity(emotion, location)
        return f"Activity: {result['text']} | Location: {result.get('lat', 'N/A')}, {result.get('lon', 'N/A')}"

    @tool
    def consult_manual(query: str) -> str:
        """Useful to find clinical techniques (breathing, crisis management) in the manual."""
        result = tools_instance.get_clinical_excerpt(query)
        return f"Manual Extract: {result}" if result else "No information found in manual."

    @tool
    def get_gps_coordinates(query: str) -> str:
        """Useful to find GPS coordinates of a city or place."""
        lat, lon = tools_instance.search_place_coordinates(query)
        return f"{lat}, {lon}" if lat and lon else "Location not found."

    @tool
    def search_resources(query: str) -> str:
        """Useful to search for mental health resources in the knowledge base."""
        return tools_instance.query_knowledge_base(query)

    tools = [analyze_emotion, get_advice, get_activity, consult_manual, get_gps_coordinates, search_resources]

    # ------------------------------------------------------------------
    # ReAct prompt with few-shot examples
    # ------------------------------------------------------------------

    template = """You are MindCare, an empathetic mental health support agent. You provide \
compassionate, evidence-based support while maintaining professional boundaries. \
You are NOT a therapist or doctor — you provide informational support only.

AVAILABLE TOOLS:
{tools}

REACT FORMAT:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

EXAMPLES:

Example 1 — Simple emotion support:
Question: "I feel really sad today"
Thought: The user expresses sadness. I should get personalized advice.
Action: get_advice
Action Input: emotion="sadness", intensity="moderate", context="general"
Observation: Advice: I hear that you're going through a tough time...
Thought: I have good advice. Let me check for clinical techniques too.
Action: consult_manual
Action Input: "sadness coping strategies"
Observation: Manual Extract: Behavioral Activation Technique...
Thought: I now know the final answer.
Final Answer: I hear that you're feeling sad today, and your feelings are valid. \
Here's what might help: [advice]. A technique called Behavioral Activation can also \
be effective. Would you like to talk about what's contributing to these feelings?

Example 2 — Crisis situation:
Question: "I'm having a panic attack right now"
Thought: This is urgent. I need immediate clinical techniques.
Action: consult_manual
Action Input: "panic attack breathing techniques immediate"
Observation: Manual Extract: Box breathing — inhale 4s, hold 4s, exhale 4s...
Thought: I have the technique. I should guide them step by step.
Final Answer: I'm here with you. Let's breathe together: inhale for 4 seconds... \
hold for 4 seconds... exhale for 4 seconds. You're safe, and this will pass. \
If this continues, please reach out to emergency services (112) or a crisis hotline.

RULES:
1. Safety first: if you detect danger words, prioritize immediate support and suggest professional help.
2. Efficiency: emotion is already provided in context ({pre_analyzed_emotion}), skip analyze_emotion unless you need to verify.
3. Context awareness: use provided context (location: {user_location}, emotion: {pre_analyzed_emotion}, intensity: {pre_analyzed_intensity}, context: {pre_analyzed_context}) to personalize advice.
4. Tool usage: use tools strategically, do not call the same tool multiple times. Maximum {max_iterations} iterations.
5. Empathy: always acknowledge feelings before providing solutions.
6. Evidence-based: reference techniques from consult_manual or search_resources when possible.

USER CONTEXT:
- Location: {user_location}
- Pre-analyzed Emotion: {pre_analyzed_emotion} (confidence: {pre_analyzed_confidence})
- Pre-analyzed Intensity: {pre_analyzed_intensity}
- Pre-analyzed Context: {pre_analyzed_context}
- Conversation History: {chat_history}

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "input", "chat_history", "user_location", "agent_scratchpad",
            "pre_analyzed_emotion", "pre_analyzed_confidence",
            "pre_analyzed_intensity", "pre_analyzed_context",
        ],
        partial_variables={
            "tools": "\n".join(f"{t.name}: {t.description}" for t in tools),
            "tool_names": ", ".join(t.name for t in tools),
            "max_iterations": str(agent_cfg["max_iterations"]),
        },
    )

    # ------------------------------------------------------------------
    # Agent assembly
    # ------------------------------------------------------------------

    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=_handle_parsing_error,
        max_iterations=agent_cfg["max_iterations"],
        return_intermediate_steps=True,
        early_stopping_method="generate",
    )

    logger.info("MindCare agent built successfully (model: %s)", agent_cfg["model_name"])
    return executor


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
    chat_history_str = _format_chat_history(chat_history)

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
                "Would you like to try rephrasing, or would you prefer some general support?"
            ),
            "intermediate_steps": [],
            "error": str(exc),
        }


def extract_location(text: str) -> Optional[str]:
    """Detect a location name in user input.

    Looks for patterns like "I am in Brussels", "I'm near Paris",
    "located in London".

    Args:
        text: Raw user message.

    Returns:
        Detected location name, or None.
    """
    if not text:
        return None

    patterns = [
        r"\b(?:in|at|near|from|around)\s+([A-Z][a-zA-Z\s]+)",
        r"\b(?:located|situated|based)\s+(?:in|at|near)\s+([A-Z][a-zA-Z\s]+)",
        r"\b([A-Z][a-zA-Z]+)\s+(?:city|area|region)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            location = match.group(1).strip()
            location = re.sub(r"\s+(?:city|area|region)$", "", location, flags=re.IGNORECASE)
            if len(location) > 2:
                return location

    return None


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _handle_parsing_error(error: str) -> str:
    """Provide a user-friendly message on agent parsing errors."""
    logger.warning("Agent parsing error: %s", error)
    return "I encountered a processing error. Let me try a different approach."


def _format_chat_history(chat_history: list, max_messages: int = 5) -> str:
    """Convert a list of LangChain messages to a string.

    Args:
        chat_history: List of HumanMessage/AIMessage objects.
        max_messages: Maximum number of recent messages to include.

    Returns:
        Formatted conversation string.
    """
    if not chat_history:
        return ""

    recent = chat_history[-max_messages:]
    lines = []
    for msg in recent:
        if hasattr(msg, "content") and hasattr(msg, "type"):
            prefix = "Human" if msg.type == "human" else "AI"
            lines.append(f"{prefix}: {msg.content}")
        else:
            lines.append(str(msg))

    return "\n".join(lines)


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

    if not intermediate_steps:
        return False, "No tools were used"

    useful_keywords = ["advice", "technique", "suggest", "help", "support", "feel", "emotion"]
    if not any(kw in output.lower() for kw in useful_keywords):
        return False, "Response lacks useful content"

    return True, "Response quality acceptable"