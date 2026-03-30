"""ML emotion classifier using Logistic Regression + TF-IDF.

Loads a pre-trained model and vectorizer from disk. Returns the
primary emotion, confidence score, ambiguity flag, and secondary
emotions above a configurable threshold.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from src.config import CONFIG

logger = logging.getLogger(__name__)

_ML_CONFIG = CONFIG["ml"]
_LABEL_MAP: dict[int, str] = {int(k): v for k, v in _ML_CONFIG["label_map"].items()}
_SECONDARY_THRESHOLD: float = _ML_CONFIG["secondary_emotion_threshold"]


class EmotionClassifier:
    """Classify emotions from text using a pre-trained ML pipeline.

    The classifier uses TF-IDF vectorization followed by Logistic
    Regression. It was selected over SVM (F1=0.909) and Naive Bayes
    (F1=0.862) for the best accuracy-to-cost ratio (F1=0.914).
    Model selection tracked with MLflow (see notebook 2).

    Args:
        model_dir: Path to directory containing the .pkl files.
    """

    def __init__(self, model_dir: Path) -> None:
        self.model = None
        self.vectorizer = None
        self._load(model_dir)

    def _load(self, model_dir: Path) -> None:
        """Load model and vectorizer from disk."""
        model_path = model_dir / "LogisticRegression.pkl"
        vectorizer_path = model_dir / "tfidf_vectorizer.pkl"

        if not model_path.exists() or not vectorizer_path.exists():
            logger.warning("ML models not found in %s", model_dir)
            return

        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            logger.info("ML models loaded from %s", model_dir)
        except Exception as exc:
            logger.error("Failed to load ML models: %s", exc)

    @property
    def is_loaded(self) -> bool:
        """Check if the model is ready for inference."""
        return self.model is not None and self.vectorizer is not None

    def classify(self, text: str) -> dict:
        """Classify the primary emotion in a text message.

        Args:
            text: Raw user message.

        Returns:
            Dictionary with keys: emotion, confidence, is_ambiguous,
            secondary_emotions. Returns an error dict if model is
            unavailable.
        """
        if not self.is_loaded:
            return {
                "error": "ML model not loaded",
                "emotion": "unknown",
                "confidence": 0.0,
            }

        vec_text = self.vectorizer.transform([text])
        probas = self.model.predict_proba(vec_text)[0]

        # Production-safe validation (assert is disabled with python -O)
        if np.any(np.isnan(probas)):
            raise ValueError("NaN detected in prediction probabilities")
        if np.any(np.isinf(probas)):
            raise ValueError("Inf detected in prediction probabilities")

        max_proba = float(np.max(probas))
        pred_index = int(np.argmax(probas))
        primary_emotion = _LABEL_MAP.get(pred_index, "unknown")

        secondary_emotions = {}
        for index, score in enumerate(probas):
            label = _LABEL_MAP.get(index, "unknown")
            if label != primary_emotion and score >= _SECONDARY_THRESHOLD:
                secondary_emotions[label] = round(float(score), 3)

        return {
            "emotion": primary_emotion,
            "confidence": round(max_proba, 2),
            "is_ambiguous": max_proba < 0.35,
            "secondary_emotions": secondary_emotions,
        }
    
    