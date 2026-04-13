"""
Weighted Scorer — aggregates routing signals into a final decision.

The scorer collects SignalResults from all signals, checks for vetoes,
and computes a weighted sum to determine the routing mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.signals import (
    BaseSignal,
    ConversationContext,
    SignalResult,
    SafetySignal,
    ConfidenceSignal,
    ComplexitySignal,
    SentimentShiftSignal,
)
from src.config import CONFIG

logger = logging.getLogger(__name__)


class RoutingMode(str, Enum):
    """The three possible routing outcomes."""
    ECO = "eco"
    HYBRID = "hybrid"
    AGENT = "agent"


@dataclass
class RoutingDecision:
    """Complete routing decision with full audit trail."""
    mode: RoutingMode
    score: float
    signals: list[SignalResult]
    reason: str
    vetoed_by: Optional[str] = None

    @property
    def is_vetoed(self) -> bool:
        return self.vetoed_by is not None

    def to_dict(self) -> dict:
        """Serialize for JSON structured logging."""
        return {
            "mode": self.mode.value,
            "score": round(self.score, 4),
            "vetoed_by": self.vetoed_by,
            "reason": self.reason,
            "signals": [
                {
                    "name": s.name,
                    "score": round(s.score, 4),
                    "confidence": round(s.confidence, 4),
                    "reason": s.reason,
                    "is_veto": s.is_veto,
                }
                for s in self.signals
            ],
        }


@dataclass
class ScorerConfig:
    """Configuration for the routing scorer."""
    low_threshold: float = 0.35
    high_threshold: float = 0.60

    def __post_init__(self) -> None:
        if self.low_threshold >= self.high_threshold:
            raise ValueError(
                f"low_threshold ({self.low_threshold}) must be < "
                f"high_threshold ({self.high_threshold})"
            )


class WeightedScorer:
    """Aggregates multiple routing signals into a single decision.

    First version: simple weighted sum without confidence weighting.
    """

    def __init__(
        self,
        config: Optional[ScorerConfig] = None,
        signals: Optional[list[BaseSignal]] = None,
    ) -> None:
        scoring_cfg = CONFIG.get("scoring", {})
        self._config = config or ScorerConfig(
            low_threshold=scoring_cfg.get("low_threshold", 0.35),
            high_threshold=scoring_cfg.get("high_threshold", 0.60),
        )
        self._signals = signals or [
            SafetySignal(),
            ConfidenceSignal(),
            ComplexitySignal(),
            SentimentShiftSignal(),
        ]

    def _compute_weighted_score(self, results: list[SignalResult]) -> float:
        """Simple weighted sum — does not factor in signal confidence."""
        non_veto = [r for r in results if not r.is_veto]
        if not non_veto:
            return 0.0

        total_weight = sum(s.weight for s in self._signals if s.name != "safety")
        if total_weight == 0:
            return 0.5

        weighted_sum = 0.0
        for result in non_veto:
            signal_weight = next(
                (s.weight for s in self._signals if s.name == result.name),
                0.1,
            )
            weighted_sum += result.score * signal_weight

        return weighted_sum / total_weight

    def _classify_score(self, score: float) -> RoutingMode:
        """Map continuous score to discrete routing mode."""
        if score < self._config.low_threshold:
            return RoutingMode.ECO
        elif score > self._config.high_threshold:
            return RoutingMode.AGENT
        else:
            return RoutingMode.HYBRID

    def score(
        self,
        text: str,
        classifier_output: Optional[dict] = None,
        context: Optional[ConversationContext] = None,
    ) -> RoutingDecision:
        """Evaluate all signals and produce a routing decision."""
        results: list[SignalResult] = []

        for signal in self._signals:
            try:
                result = signal.evaluate(text, classifier_output, context)
                results.append(result)
            except Exception as e:
                logger.warning("Signal %s failed: %s", signal.name, e)
                results.append(SignalResult(
                    name=signal.name, score=0.5, confidence=0.1,
                    reason=f"Signal evaluation failed: {e}",
                ))

        # Check for veto
        veto_signals = [r for r in results if r.is_veto and r.score > 0.5]
        if veto_signals:
            veto = veto_signals[0]
            return RoutingDecision(
                mode=RoutingMode.AGENT, score=1.0, signals=results,
                reason=f"VETO by {veto.name}: {veto.reason}",
                vetoed_by=veto.name,
            )

        final_score = self._compute_weighted_score(results)
        mode = self._classify_score(final_score)

        top = sorted(results, key=lambda r: r.score, reverse=True)[:2]
        reason = (
            f"Score {final_score:.3f} → {mode.value.upper()} "
            f"(top: {', '.join(f'{r.name}={r.score:.2f}' for r in top)})"
        )

        return RoutingDecision(
            mode=mode, score=final_score, signals=results, reason=reason,
        )
    

    