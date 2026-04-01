"""MindCare tools — modular toolbox for the agent.

Sub-modules:
    classifier: ML emotion classification (LogReg + TF-IDF).
    advisor: CSV advice database with fallback chain.
    rag: FAISS vectorstore retrieval with auto-build.
    geolocation: Nominatim geocoding and activity suggestion.

The MindCareTools class is a facade that delegates to sub-modules.
External code (app.py, agent.py) imports from here — the internal
structure is transparent.

    from src.tools import MindCareTools
    tools = MindCareTools()
    tools.classify_emotion("I feel sad")  # → classifier.classify()
"""

import logging
from pathlib import Path
from typing import Optional

from src.config import CONFIG, PROJECT_ROOT
from src.tools.classifier import EmotionClassifier
from src.tools.advisor import Advisor
from src.tools.rag import RAGRetriever
from src.tools.geolocation import GeoLocator

logger = logging.getLogger(__name__)

_PATHS = CONFIG["paths"]


class MindCareTools:
    """Facade that aggregates all tool sub-modules.

    Maintains backward compatibility — existing code continues
    to call tools_instance.classify_emotion(), get_advice(), etc.

    Args:
        config: Configuration dictionary. Defaults to global CONFIG.
        root: Project root path. Defaults to PROJECT_ROOT.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        root: Optional[Path] = None,
    ) -> None:
        config = config or CONFIG
        root = root or PROJECT_ROOT

        self._classifier = EmotionClassifier(root / _PATHS["model_dir"])
        self._advisor = Advisor(root / _PATHS["advice_db"])
        self._rag = RAGRetriever(
            vectorstore_dir=root / _PATHS["vectorstore_dir"],
            source_dir=root / _PATHS["source_dir"],
        )
        self._geo = GeoLocator()

    # --- Classifier delegation ---

    def classify_emotion(self, text: str) -> dict:
        """Classify the primary emotion in a text message."""
        return self._classifier.classify(text)

    # --- Advisor delegation ---

    def get_advice(
        self,
        emotion: str,
        intensity: str = "moderate",
        context: str = "general",
        confidence: Optional[float] = None,
    ) -> tuple[str, str]:
        """Retrieve personalized advice from the CSV database."""
        return self._advisor.get_advice(emotion, intensity, context, confidence)

    # --- RAG delegation ---

    def get_clinical_excerpt(
        self,
        query: str,
        context: str = "general",
    ) -> Optional[dict]:
        """Retrieve a clinical excerpt with source metadata."""
        return self._rag.get_clinical_excerpt(query, context)

    def query_knowledge_base(self, query: str) -> str:
        """Search the RAG knowledge base for detailed information."""
        return self._rag.query_knowledge_base(query)

    # --- Geolocation delegation ---

    def get_activity(self, emotion: str, user_location: str) -> dict:
        """Suggest a location-based activity for the detected emotion."""
        return self._geo.get_activity(emotion, user_location)

    def search_place_coordinates(
        self, place_name: str, city_name: Optional[str] = None
    ) -> tuple[Optional[float], Optional[float]]:
        """Geocode a place name using the Nominatim API."""
        return self._geo.search_place_coordinates(place_name, city_name)
    

    