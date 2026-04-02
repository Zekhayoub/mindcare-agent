"""LangChain ReAct agent construction with tool definitions.

The agent is not created at import time — only when build_agent()
is explicitly called. This avoids crashes if the API key is missing.
"""

import logging
from typing import Optional

from src.config import CONFIG

logger = logging.getLogger(__name__)


def build_agent(tools_instance, config: Optional[dict] = None):
    """Build and return a configured AgentExecutor.

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
        if isinstance(result, dict):
            return f"Manual Extract: {result['content']}\nSource: {result['source']}"
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

    # All 6 tools included — analyze_emotion will be conditionally
    # excluded in a later commit when confidence is high
    tools = [analyze_emotion, get_advice, get_activity, consult_manual,
             get_gps_coordinates, search_resources]

    # ------------------------------------------------------------------
    # ReAct prompt (will be extracted to prompts.py in Part 5)
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

    from src.agent.executor import handle_parsing_error

    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=handle_parsing_error,
        max_iterations=agent_cfg["max_iterations"],
        return_intermediate_steps=True,
        early_stopping_method="generate",
    )

    logger.info("MindCare agent built successfully (model: %s)", agent_cfg["model_name"])
    return executor