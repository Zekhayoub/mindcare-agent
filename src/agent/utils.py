"""Agent utilities: location extraction, chat formatting.

Location detection currently uses regex patterns — will be
replaced by spaCy NER.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def extract_location(text: str) -> Optional[str]:
    """Detect a location name in user input using regex.

    Looks for patterns like "I am in Brussels", "I'm near Paris",
    "located in London".

    Known issue: "I feel sad in general" detects "General" as a city.
    This will be fixed by replacing regex with spaCy NER.

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
            location = re.sub(
                r"\s+(?:city|area|region)$", "", location, flags=re.IGNORECASE
            )
            if len(location) > 2:
                return location

    return None


def format_chat_history(chat_history: list, max_messages: int = 5) -> str:
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

