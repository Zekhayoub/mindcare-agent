"""
Routing signals for the MindCare Strategist.

Each signal extracts a feature from the user input and conversation
context, returning a float between 0.0 (strongly ECO) and 1.0
(strongly AGENT).

Signals are independent, testable, and composable. The Scorer
aggregates them with configurable weights.
"""

from __future__ import annotations

import re
import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from src.config import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    """Output of a single signal evaluation."""

    name: str
    score: float  # 0.0 = ECO, 1.0 = AGENT
    confidence: float  # How confident is this signal in its own score
    reason: str  # Human-readable explanation for logging
    is_veto: bool = False  # If True, overrides all other signals

    def __post_init__(self) -> None:
        self.score = max(0.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class ConversationContext:
    """Accumulated context from the conversation history."""

    message_count: int = 0
    emotion_history: list[str] = field(default_factory=list)
    confidence_history: list[float] = field(default_factory=list)
    mode_history: list[str] = field(default_factory=list)


class BaseSignal(ABC):
    """Abstract base class for all routing signals."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this signal."""

    @property
    @abstractmethod
    def weight(self) -> float:
        """Default weight for this signal in the scoring function."""

    @abstractmethod
    def evaluate(
        self,
        text: str,
        classifier_output: Optional[dict] = None,
        context: Optional[ConversationContext] = None,
    ) -> SignalResult:
        """Evaluate the signal and return a result."""


class SafetySignal(BaseSignal):
    """Detects crisis indicators in user input.

    Uses three layers of detection:
    1. Explicit multi-word patterns ("kill myself", "want to die")
    2. Implicit regex patterns for indirect expressions
    3. Negation awareness ("I'm NOT suicidal" does not trigger)

    This signal has VETO power: if triggered, routing goes to AGENT
    regardless of all other signals.
    """

    @property
    def name(self) -> str:
        return "safety"

    @property
    def weight(self) -> float:
        return 1.0

    def __init__(self) -> None:
        scoring_cfg = CONFIG.get("scoring", {})
        patterns_cfg = scoring_cfg.get("safety_patterns", {})

        # Multi-word explicit patterns
        self._explicit_patterns: list[str] = patterns_cfg.get("explicit", [])

        # Regex implicit patterns
        self._implicit_patterns: list[re.Pattern] = []
        for pattern_str in patterns_cfg.get("implicit", []):
            try:
                self._implicit_patterns.append(
                    re.compile(pattern_str, re.IGNORECASE)
                )
            except re.error:
                logger.warning("Invalid safety pattern: %s", pattern_str)

        # Benign contexts that should NOT trigger safety
        self._benign_contexts: list[str] = patterns_cfg.get("benign_contexts", [])

    def _is_benign_context(self, text_lower: str) -> bool:
        """Check if the text contains a benign context for 'help' etc."""
        return any(ctx in text_lower for ctx in self._benign_contexts)

    def _is_negated(self, text: str, match_start: int) -> bool:
        """Check if a match is preceded by a negation within 40 chars."""
        preceding = text[max(0, match_start - 40): match_start]
        negation_pattern = re.compile(
            r"\b(not|never|no longer|don'?t|doesn'?t|isn'?t|aren'?t|wasn'?t)\b",
            re.IGNORECASE,
        )
        return bool(negation_pattern.search(preceding))

    def evaluate(
        self,
        text: str,
        classifier_output: Optional[dict] = None,
        context: Optional[ConversationContext] = None,
    ) -> SignalResult:
        text_lower = text.lower()
        triggers_found: list[str] = []

        # Skip if benign context detected
        if self._is_benign_context(text_lower):
            return SignalResult(
                name=self.name,
                score=0.0,
                confidence=0.85,
                reason="Benign context detected — safety not triggered",
            )

        # Check explicit multi-word patterns
        for pattern in self._explicit_patterns:
            idx = text_lower.find(pattern)
            if idx != -1 and not self._is_negated(text_lower, idx):
                triggers_found.append(f"explicit: '{pattern}'")

        # Check implicit regex patterns
        for regex in self._implicit_patterns:
            match = regex.search(text_lower)
            if match and not self._is_negated(text_lower, match.start()):
                triggers_found.append(f"implicit: '{match.group()}'")

        if triggers_found:
            return SignalResult(
                name=self.name,
                score=1.0,
                confidence=0.95,
                reason=f"Safety triggers: {', '.join(triggers_found)}",
                is_veto=True,
            )

        return SignalResult(
            name=self.name,
            score=0.0,
            confidence=0.9,
            reason="No safety triggers detected",
        )


class ConfidenceSignal(BaseSignal):
    """Converts classifier confidence into a continuous routing signal.

    Uses an inverse sigmoid that creates a smooth transition:
        95% confidence → score 0.05 (very ECO)
        80% confidence → score 0.15 (mostly ECO)
        60% confidence → score 0.50 (uncertain — grey zone)
        40% confidence → score 0.85 (mostly AGENT)
        20% confidence → score 0.95 (very AGENT)

    The midpoint and steepness are configurable in config.yaml.
    """

    @property
    def name(self) -> str:
        return "confidence"

    @property
    def weight(self) -> float:
        scoring_cfg = CONFIG.get("scoring", {})
        return scoring_cfg.get("weights", {}).get("confidence", 0.30)

    def __init__(self) -> None:
        scoring_cfg = CONFIG.get("scoring", {})
        conf_cfg = scoring_cfg.get("confidence_signal", {})
        self._midpoint = conf_cfg.get("midpoint", 0.60)
        self._steepness = conf_cfg.get("steepness", 10.0)

    def _sigmoid_inverse(self, confidence: float) -> float:
        """Map confidence to routing score using inverse sigmoid."""
        x = (self._midpoint - confidence) * self._steepness
        return 1.0 / (1.0 + math.exp(-x))

    def evaluate(
        self,
        text: str,
        classifier_output: Optional[dict] = None,
        context: Optional[ConversationContext] = None,
    ) -> SignalResult:
        if classifier_output is None:
            return SignalResult(
                name=self.name,
                score=0.8,
                confidence=0.3,
                reason="No classifier output — defaulting toward AGENT",
            )

        clf_confidence = classifier_output.get("confidence", 0.5)
        emotion = classifier_output.get("emotion", "unknown")
        score = self._sigmoid_inverse(clf_confidence)

        return SignalResult(
            name=self.name,
            score=score,
            confidence=0.85,
            reason=(
                f"Classifier: {emotion} at {clf_confidence:.1%} "
                f"→ routing score {score:.2f}"
            ),
        )
    
