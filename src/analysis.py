"""Context detection, intensity estimation, and emotion scoring utilities."""

import logging

from src.config import CONFIG

logger = logging.getLogger(__name__)

_CONTEXT_KEYWORDS: dict[str, list[str]] = CONFIG.get("context_keywords", {})
_INTENSITY_KEYWORDS: dict[str, list[str]] = CONFIG.get("intensity_keywords", {})


def detect_context(text: str) -> str:
    """Detect the emotional context from user input.

    Scans the text for keyword matches against predefined context
    categories (work, relationship, academic, etc.) and returns
    the category with the highest match count.

    Args:
        text: Raw user message.

    Returns:
        Context label (e.g. "work", "academic", "general").
    """
    text_lower = (text or "").lower()
    if not text_lower:
        return "general"

    context_scores: dict[str, int] = {}
    for context, keywords in _CONTEXT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            context_scores[context] = score

    if context_scores:
        return max(context_scores, key=context_scores.get)
    return "general"


def determine_intensity(confidence: float, text: str = "") -> str:
    """Estimate emotion intensity from classifier confidence and text cues.

    Combines the ML confidence score with keyword analysis to
    produce a three-level intensity label.

    Args:
        confidence: Classifier confidence score (0.0 to 1.0).
        text: Raw user message for keyword analysis.

    Returns:
        Intensity label: "mild", "moderate", or "severe".
    """
    confidence = max(0.0, min(1.0, confidence))
    text_lower = (text or "").lower()

    severe_keywords = _INTENSITY_KEYWORDS.get("severe", [])
    mild_keywords = _INTENSITY_KEYWORDS.get("mild", [])

    has_severe = any(kw in text_lower for kw in severe_keywords)
    has_mild = any(kw in text_lower for kw in mild_keywords)

    if has_severe or confidence > 0.85:
        return "severe"
    if has_mild or confidence < 0.5:
        return "mild"
    return "moderate"


def get_emotion_score(emotion_name: str) -> float:
    """Map an emotion label to a numerical sentiment score.

    Used for the emotion timeline in the dashboard.

    Args:
        emotion_name: Emotion label (e.g. "joy", "sadness").

    Returns:
        Score between -1.0 (negative) and 1.0 (positive).
    """
    mapping = {
        "joy": 1.0,
        "love": 1.0,
        "surprise": 0.5,
        "unknown": 0.0,
        "fear": -0.5,
        "sadness": -1.0,
        "anger": -1.0,
    }
    return mapping.get(emotion_name.lower(), 0.0)