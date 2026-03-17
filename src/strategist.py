"""ECO/AGENT routing logic for MindCare queries."""

import logging
import re
from typing import Optional

from src.config import CONFIG

logger = logging.getLogger(__name__)


class MindCareStrategist:
    """Routes user queries to ECO (local) or AGENT (cloud) mode.

    Decision is based on safety keywords, query complexity,
    classifier confidence, and emotion type. Designed to maximize
    local processing for lower carbon footprint while ensuring
    quality on complex or ambiguous queries.

    Args:
        config: Configuration dictionary. Defaults to global CONFIG.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        config = config or CONFIG
        strategist_cfg = config["strategist"]

        self.danger_words: list[str] = strategist_cfg["danger_words"]
        self.question_words: list[str] = strategist_cfg["question_words"]
        self.safe_emotions: list[str] = strategist_cfg["safe_emotions"]
        self.threshold: float = config["ml"]["confidence_threshold"]

    def decide_strategy(self, user_input: str, ml_analysis: dict) -> tuple[str, str]:
        """Determine whether to use ECO or AGENT mode.

        Applies rules in priority order: safety, complexity,
        confidence, emotion recognition, emotion type.

        Args:
            user_input: Raw user message.
            ml_analysis: Output from MindCareTools.classify_emotion().

        Returns:
            Tuple of (mode, reason) where mode is "ECO" or "AGENT".
        """
        text_lower = user_input.lower()
        words = set(re.sub(r"[^\w\s?]", "", text_lower).split())
        emotion = ml_analysis.get("emotion", "unknown").lower()
        confidence = ml_analysis.get("confidence", 0.0)

        # Rule 1: Safety — always escalate danger signals
        for word in self.danger_words:
            if word in words:
                return "AGENT", f"Safety trigger detected: '{word}'"

        # Rule 2: Complexity — questions need LLM reasoning
        for q_word in self.question_words:
            if q_word == "?" and "?" in text_lower:
                return "AGENT", "Question mark detected"
            if q_word != "?" and q_word in words:
                return "AGENT", "Complex question detected"

        # Rule 3: Confidence — below threshold means uncertain classification
        if confidence < self.threshold:
            return "AGENT", f"Low confidence ({confidence:.0%})"

        # Rule 4: Unknown emotion — classifier could not decide
        if emotion == "unknown":
            return "AGENT", "Emotion not recognized"

        # Rule 5: Emotion type — some emotions need nuanced handling
        if emotion not in self.safe_emotions:
            return "AGENT", f"'{emotion}' requires nuanced handling"

        # All checks passed — safe for local processing
        return "ECO", "Simple request (Green AI)"