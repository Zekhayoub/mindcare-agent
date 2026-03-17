"""MindCare tools: emotion classifier, advice database, RAG, and geolocation."""

import logging
import re
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import requests

from src.config import CONFIG, PROJECT_ROOT

logger = logging.getLogger(__name__)

_PATHS = CONFIG["paths"]
_ML_CONFIG = CONFIG["ml"]
_GEO_CONFIG = CONFIG["geolocation"]
_LABEL_MAP: dict[int, str] = {int(k): v for k, v in _ML_CONFIG["label_map"].items()}
_SECONDARY_THRESHOLD: float = _ML_CONFIG["secondary_emotion_threshold"]


class MindCareTools:
    """Core toolbox for the MindCare agent.

    Loads ML models, advice database, and RAG vectorstore on init.
    All methods are safe to call even if a resource failed to load
    (graceful degradation with fallback responses).

    Args:
        config: Configuration dictionary. Defaults to the global CONFIG.
        root: Project root path. Defaults to PROJECT_ROOT.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        root: Optional[Path] = None,
    ) -> None:
        config = config or CONFIG
        root = root or PROJECT_ROOT

        self.model = None
        self.vectorizer = None
        self.advice_df = pd.DataFrame()
        self.vector_db = None

        self._load_ml_models(root / _PATHS["model_dir"])
        self._load_advice_db(root / _PATHS["advice_db"])
        self._load_vectorstore(root / _PATHS["vectorstore_dir"])

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_ml_models(self, model_dir: Path) -> None:
        """Load the emotion classifier and TF-IDF vectorizer."""
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

    def _load_advice_db(self, path: Path) -> None:
        """Load the CSV advice database."""
        if not path.exists():
            logger.warning("Advice database not found: %s", path)
            return

        try:
            self.advice_df = pd.read_csv(path)
            self.advice_df["emotion"] = self.advice_df["emotion"].str.strip().str.lower()
            logger.info("Advice database loaded (%d rows)", len(self.advice_df))
        except Exception as exc:
            logger.error("Failed to load advice database: %s", exc)

    def _load_vectorstore(self, vs_path: Path) -> None:
        """Load the FAISS vectorstore for RAG retrieval.

        FAISS and LangChain imports are deferred so the rest of the
        class still works if these packages are not installed.
        """
        import os

        api_key = os.getenv("MISTRAL_API_KEY")
        if not vs_path.exists() or not api_key:
            logger.warning("RAG vectorstore not loaded (path or API key missing)")
            return

        try:
            from langchain_community.vectorstores import FAISS
            from langchain_mistralai import MistralAIEmbeddings

            embeddings = MistralAIEmbeddings(api_key=api_key, model="mistral-embed")
            # Safe: vectorstore is built locally from trusted source documents (see notebook 3)
            self.vector_db = FAISS.load_local(
                str(vs_path), embeddings, allow_dangerous_deserialization=True
            )
            logger.info("RAG vectorstore loaded from %s", vs_path)
        except Exception as exc:
            logger.error("Failed to load vectorstore: %s", exc)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def classify_emotion(self, text: str) -> dict:
        """Classify the primary emotion in a text message.

        Args:
            text: Raw user message.

        Returns:
            Dictionary with keys: emotion, confidence, is_ambiguous,
            secondary_emotions. Returns an error dict if model is
            unavailable.
        """
        if self.model is None:
            return {"error": "ML model not loaded", "emotion": "unknown", "confidence": 0.0}

        vec_text = self.vectorizer.transform([text])
        probas = self.model.predict_proba(vec_text)[0]

        assert not np.any(np.isnan(probas)), "NaN detected in prediction probabilities"
        assert not np.any(np.isinf(probas)), "Inf detected in prediction probabilities"

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

    def get_advice(
        self,
        emotion: str,
        intensity: str = "moderate",
        context: str = "general",
        confidence: Optional[float] = None,
    ) -> tuple[str, str]:
        """Retrieve personalized advice from the CSV database.

        Uses a fallback chain: (emotion + intensity + context) then
        (emotion + intensity) then (emotion only).

        Args:
            emotion: Detected emotion label.
            intensity: Emotion intensity ("mild", "moderate", "severe").
            context: Situational context ("work", "academic", etc.).
            confidence: Classifier confidence, used to adjust intensity
                if not explicitly provided.

        Returns:
            Tuple of (advice text, notes/metadata).
        """
        if emotion.lower() == "unknown":
            return "Could you tell me more about how you are feeling?", "N/A"
        if self.advice_df.empty:
            return "I'm here to support you.", "Database unavailable"

        emotion_lower = emotion.lower()

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
            row = self._query_advice(emotion_lower, intensity, context)
            if row is not None:
                return self._format_advice(row)

            row = self._query_advice(emotion_lower, intensity)
            if row is not None:
                return self._format_advice(row)

        matches = self.advice_df[self.advice_df["emotion"] == emotion_lower]
        if not matches.empty:
            row = matches.sample(n=1).iloc[0] if len(matches) > 1 else matches.iloc[0]
            return str(row["advice"]), str(row["notes"])

        return f"I hear you are feeling {emotion}. I'm here for you.", "General"

    def get_activity(self, emotion: str, user_location: str) -> dict:
        """Suggest a location-based activity for the detected emotion.

        Fully local — no LLM call. Uses Nominatim for geocoding
        with config-based fallback locations.

        Args:
            emotion: Detected emotion label.
            user_location: City or area name.

        Returns:
            Dictionary with keys: text, lat, lon.
        """
        emotion_key = (emotion or "").lower()
        user_loc = user_location or "Brussels"

        fallback_query = "City Center"
        if emotion_key in ("sadness", "fear"):
            fallback_query = "Park"
        elif emotion_key in ("joy", "love"):
            fallback_query = "Plaza"
        elif emotion_key == "anger":
            fallback_query = "Gym"

        lat, lon = self.search_place_coordinates(fallback_query, user_loc)
        if lat and lon:
            return {"text": f"{fallback_query} ({user_loc})", "lat": lat, "lon": lon}

        fallback = _GEO_CONFIG["fallback_locations"].get(emotion_key, {})
        if fallback:
            return {
                "text": fallback.get("name", "Local activity"),
                "lat": fallback.get("lat"),
                "lon": fallback.get("lon"),
            }

        return {"text": "Location unavailable", "lat": None, "lon": None}

    def get_clinical_excerpt(self, query: str) -> Optional[str]:
        """Retrieve a clinical excerpt from the RAG vectorstore.

        Args:
            query: Search query (emotion or coping strategy).

        Returns:
            Relevant text excerpt (max 500 chars), or None if RAG
            is unavailable.
        """
        if self.vector_db is None:
            return None

        try:
            enriched_query = (
                f"techniques for managing {query} emotion coping strategies"
            )
            results = self.vector_db.similarity_search(enriched_query, k=1)

            if not results:
                return None

            content = results[0].page_content
            if len(content) > 500:
                truncated = content[:500]
                last_period = truncated.rfind(".")
                content = truncated[:last_period + 1] if last_period > 400 else truncated + "..."

            return content

        except Exception as exc:
            logger.error("RAG excerpt retrieval failed: %s", exc)
            return None

    def query_knowledge_base(self, query: str) -> str:
        """Search the RAG knowledge base for detailed information.

        Args:
            query: Search query.

        Returns:
            Concatenated content from the top 2 matching chunks,
            or an error message if RAG is unavailable.
        """
        if self.vector_db is None:
            return "Knowledge base unavailable."

        try:
            results = self.vector_db.similarity_search(query, k=2)
            return "\n".join(doc.page_content for doc in results)
        except Exception as exc:
            logger.error("Knowledge base query failed: %s", exc)
            return f"Knowledge base error: {exc}"

    def search_place_coordinates(
        self, place_name: str, city_name: Optional[str] = None
    ) -> tuple[Optional[float], Optional[float]]:
        """Geocode a place name using the Nominatim API.

        Respects Nominatim rate limit (1 req/s) by sleeping between
        retry attempts.

        Args:
            place_name: Name of the place to search.
            city_name: Optional city for context.

        Returns:
            Tuple of (latitude, longitude) or (None, None) on failure.
        """
        headers = {"User-Agent": "MindCareAgent/1.0"}
        timeout = _GEO_CONFIG.get("timeout_s", 5)
        base_url = _GEO_CONFIG.get("nominatim_url", "https://nominatim.openstreetmap.org/search")

        queries = [f"{place_name}, {city_name}", place_name] if city_name else [place_name]

        for i, query in enumerate(queries):
            if i > 0:
                time.sleep(1)
            try:
                response = requests.get(
                    base_url,
                    params={"q": query, "format": "json", "limit": 1},
                    headers=headers,
                    timeout=timeout,
                )
                if response.ok and response.json():
                    data = response.json()[0]
                    return float(data["lat"]), float(data["lon"])
            except Exception as exc:
                logger.warning("Geocoding failed for '%s': %s", query, exc)
                continue

        return None, None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _query_advice(
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
        return filtered.sample(n=1).iloc[0] if len(filtered) > 1 else filtered.iloc[0]

    def _format_advice(self, row: pd.Series) -> tuple[str, str]:
        """Format an advice row into (advice, notes) with optional extras."""
        advice = str(row["advice"])
        notes = str(row["notes"])

        if "technique" in row.index and pd.notna(row["technique"]):
            notes += f" | Technique: {row['technique']}"
        if "citation" in row.index and pd.notna(row["citation"]):
            notes += f" | Source: {row['citation']}"

        return advice, notes