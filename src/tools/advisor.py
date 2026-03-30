"""CSV advice database with multi-level fallback chain.

Retrieves personalized advice based on emotion, intensity, and context.
Falls back through three levels of specificity:
    1. emotion + intensity + context (most specific)
    2. emotion + intensity
    3. emotion only (most general)
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class Advisor:
    """Retrieve personalized advice from the CSV database.

    The advice database contains 68 curated entries with emotion,
    intensity, context, technique, and citation columns.

    Args:
        advice_path: Path to the CSV file.
    """

    def __init__(self, advice_path: Path) -> None:
        self.advice_df = pd.DataFrame()
        self._load(advice_path)

    def _load(self, path: Path) -> None:
        """Load the CSV advice database."""
        if not path.exists():
            logger.warning("Advice database not found: %s", path)
            return

        try:
            self.advice_df = pd.read_csv(path)
            self.advice_df["emotion"] = (
                self.advice_df["emotion"].str.strip().str.lower()
            )
            logger.info("Advice database loaded (%d rows)", len(self.advice_df))
        except Exception as exc:
            logger.error("Failed to load advice database: %s", exc)

    @property
    def is_loaded(self) -> bool:
        """Check if the advice database has entries."""
        return not self.advice_df.empty

    def get_advice(
        self,
        emotion: str,
        intensity: str = "moderate",
        context: str = "general",
        confidence: Optional[float] = None,
    ) -> tuple[str, str]:
        """Retrieve personalized advice with fallback chain.

        Tries (emotion + intensity + context) first, then
        (emotion + intensity), then (emotion only).

        Args:
            emotion: Detected emotion label.
            intensity: Emotion intensity ("mild", "moderate", "severe").
            context: Situational context ("work", "academic", etc.).
            confidence: Classifier confidence, used to adjust intensity.

        Returns:
            Tuple of (advice text, notes/metadata).
        """
        if emotion.lower() == "unknown":
            return (
                "I can see something is on your mind. "
                "Could you tell me more about how you are feeling?",
                "N/A",
            )

        if not self.is_loaded:
            return "I'm here to support you.", "Database unavailable"

        emotion_lower = emotion.lower()

        # Adjust intensity based on confidence if not explicitly set
        if confidence is not None and intensity == "moderate":
            if confidence < 0.5:
                intensity = "mild"
            elif confidence > 0.85:
                intensity = "severe"

        has_enriched = (
            "intensity" in self.advice_df.columns
            and "context" in self.advice_df.columns
        )

        if has_enriched:
            # Level 1: emotion + intensity + context
            row = self._query(emotion_lower, intensity, context)
            if row is not None:
                return self._format(row)

            # Level 2: emotion + intensity
            row = self._query(emotion_lower, intensity)
            if row is not None:
                return self._format(row)

        # Level 3: emotion only
        matches = self.advice_df[self.advice_df["emotion"] == emotion_lower]
        if not matches.empty:
            row = (
                matches.sample(n=1).iloc[0]
                if len(matches) > 1
                else matches.iloc[0]
            )
            return str(row["advice"]), str(row["notes"])

        # Empathetic fallback (not robotic)
        return (
            f"I hear that you're experiencing {emotion}. "
            "Your feelings are valid, and I'm here to listen. "
            "Would you like to share more about what's going on?",
            "General — no specific advice found for this combination",
        )

    def _query(
        self,
        emotion: str,
        intensity: str,
        context: Optional[str] = None,
    ) -> Optional[pd.Series]:
        """Query the advice dataframe with optional context filter."""
        mask = (
            (self.advice_df["emotion"] == emotion)
            & (self.advice_df["intensity"] == intensity)
        )
        if context:
            mask = mask & (self.advice_df["context"] == context)

        filtered = self.advice_df[mask]
        if filtered.empty:
            return None
        return (
            filtered.sample(n=1).iloc[0]
            if len(filtered) > 1
            else filtered.iloc[0]
        )

    def _format(self, row: pd.Series) -> tuple[str, str]:
        """Format an advice row into (advice, notes) with extras."""
        advice = str(row["advice"])
        notes = str(row["notes"])

        if "technique" in row.index and pd.notna(row["technique"]):
            notes += f" | Technique: {row['technique']}"
        if "citation" in row.index and pd.notna(row["citation"]):
            notes += f" | Source: {row['citation']}"

        return advice, notes
    

