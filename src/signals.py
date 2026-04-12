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


        