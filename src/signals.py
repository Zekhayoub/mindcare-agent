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
    
class ComplexitySignal(BaseSignal):
    """Analyzes linguistic complexity to determine if LLM reasoning is needed.

    Multi-factor analysis:
    - Question type (open-ended vs closed)
    - Message length (longer = more context to process)
    - Temporal references ("since last month" = narrative context)
    - Conditional language ("what if" = hypothetical reasoning)
    - Compound topics (multiple conjunctions)
    - Negation complexity ("not unhappy" = nuanced)

    Each factor produces a score, weighted and summed.
    """

    OPEN_ENDED_PATTERNS: list[re.Pattern] = [
        re.compile(r"\b(why|how come|what should|what can|how do|how can)\b", re.I),
        re.compile(r"\b(explain|help me understand|tell me about)\b", re.I),
    ]

    CLOSED_QUESTION_PATTERNS: list[re.Pattern] = [
        re.compile(r"\b(are you|is it|do you|can I|will you)\b.*\?", re.I),
    ]

    TEMPORAL_PATTERNS: list[re.Pattern] = [
        re.compile(r"\b(since|for the past|for \d+|last \w+|recently|lately)\b", re.I),
        re.compile(r"\b(months?|years?|weeks?|days?) (ago|now|later)\b", re.I),
    ]

    CONDITIONAL_PATTERNS: list[re.Pattern] = [
        re.compile(r"\b(if|what if|suppose|assuming|would|could|might)\b", re.I),
    ]

    COMPOUND_CONJUNCTIONS: list[re.Pattern] = [
        re.compile(r"\b(but also|and also|moreover|furthermore|on the other hand)\b", re.I),
        re.compile(r"\b(however|although|even though|despite)\b", re.I),
    ]

    NEGATION_COMPLEXITY: list[re.Pattern] = [
        re.compile(r"\bnot\s+\w+\s+but\b", re.I),
        re.compile(r"\b(not|never)\s+(really|quite|exactly|entirely)\b", re.I),
        re.compile(r"\bdon'?t\s+(know|think|feel|believe)\s+(if|that|whether)\b", re.I),
    ]

    @property
    def name(self) -> str:
        return "complexity"

    @property
    def weight(self) -> float:
        scoring_cfg = CONFIG.get("scoring", {})
        return scoring_cfg.get("weights", {}).get("complexity", 0.25)

    def evaluate(
        self,
        text: str,
        classifier_output: Optional[dict] = None,
        context: Optional[ConversationContext] = None,
    ) -> SignalResult:
        factors: dict[str, float] = {}

        # Length factor
        word_count = len(text.split())
        factors["length"] = min(1.0, word_count / 50.0)

        # Question complexity — open-ended vs closed
        is_open = any(p.search(text) for p in self.OPEN_ENDED_PATTERNS)
        is_closed = any(p.search(text) for p in self.CLOSED_QUESTION_PATTERNS)
        has_question_mark = "?" in text

        if is_open:
            factors["question"] = 0.8
        elif is_closed and has_question_mark:
            factors["question"] = 0.2  # Closed questions don't need LLM
        elif has_question_mark:
            factors["question"] = 0.4
        else:
            factors["question"] = 0.0

        # Temporal context
        factors["temporal"] = 0.6 if any(p.search(text) for p in self.TEMPORAL_PATTERNS) else 0.0

        # Conditional language
        factors["conditional"] = 0.5 if any(p.search(text) for p in self.CONDITIONAL_PATTERNS) else 0.0

        # Compound topics
        compound_count = sum(1 for p in self.COMPOUND_CONJUNCTIONS if p.search(text))
        factors["compound"] = min(1.0, compound_count * 0.4)

        # Negation complexity
        negation_count = sum(1 for p in self.NEGATION_COMPLEXITY if p.search(text))
        factors["negation"] = min(1.0, negation_count * 0.5)

        # Weighted sum
        factor_weights = CONFIG.get("scoring", {}).get("complexity_factors", {
            "length": 0.10, "question": 0.30, "temporal": 0.15,
            "conditional": 0.15, "compound": 0.15, "negation": 0.15,
        })
        score = sum(factors.get(k, 0) * factor_weights.get(k, 0) for k in factor_weights)

        active = [k for k, v in factors.items() if v > 0.3]

        return SignalResult(
            name=self.name,
            score=score,
            confidence=0.75,
            reason=f"Complexity {score:.2f} — active: {active or ['none']}",
        )



class SentimentShiftSignal(BaseSignal):
    """Detects abrupt emotional transitions in the conversation.

    A user expressing joy 2 messages ago and deep sadness now
    represents a concerning shift that warrants LLM attention,
    even if the classifier is confident about the current emotion.

    Uses a valence mapping and exponential decay on recent history.
    Also detects sustained negative patterns (3+ consecutive
    negative emotions).
    """

    @property
    def name(self) -> str:
        return "sentiment_shift"

    @property
    def weight(self) -> float:
        scoring_cfg = CONFIG.get("scoring", {})
        return scoring_cfg.get("weights", {}).get("sentiment_shift", 0.25)

    def __init__(self) -> None:
        scoring_cfg = CONFIG.get("scoring", {})
        self._valence: dict[str, float] = scoring_cfg.get("emotion_valence", {
            "sadness": -0.8, "anger": -0.6, "fear": -0.7,
            "surprise": 0.0, "love": 0.8, "joy": 0.9,
        })

    def _compute_shift(self, history: list[str], current: str) -> float:
        """Compute emotional shift magnitude from recent history."""
        if len(history) < 1:
            return 0.0

        current_valence = self._valence.get(current, 0.0)

        shifts: list[float] = []
        for i, emotion in enumerate(reversed(history[-3:])):
            past_valence = self._valence.get(emotion, 0.0)
            delta = abs(current_valence - past_valence)
            recency_weight = 1.0 / (i + 1)
            shifts.append(delta * recency_weight)

        return min(1.0, max(shifts) / 1.5) if shifts else 0.0

    def evaluate(
        self,
        text: str,
        classifier_output: Optional[dict] = None,
        context: Optional[ConversationContext] = None,
    ) -> SignalResult:
        if context is None or classifier_output is None:
            return SignalResult(
                name=self.name,
                score=0.0,
                confidence=0.5,
                reason="Insufficient context for shift detection",
            )

        current_emotion = classifier_output.get("emotion", "unknown")

        # Guard: unknown emotion → neutral score
        if current_emotion == "unknown":
            return SignalResult(
                name=self.name,
                score=0.0,
                confidence=0.4,
                reason="Unknown emotion — cannot compute shift",
            )

        shift = self._compute_shift(context.emotion_history, current_emotion)
        reason_suffix = ""

        # Sustained negative pattern (3+ consecutive negative emotions)
        if len(context.emotion_history) >= 3:
            recent = context.emotion_history[-3:]
            if all(self._valence.get(e, 0.0) < -0.5 for e in recent):
                shift = max(shift, 0.6)
                reason_suffix = " + sustained negative pattern"

        return SignalResult(
            name=self.name,
            score=shift,
            confidence=0.7 if len(context.emotion_history) >= 2 else 0.4,
            reason=f"Sentiment shift: {shift:.2f}{reason_suffix}",
        )
    

    