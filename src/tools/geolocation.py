"""Geolocation and activity suggestion using Nominatim API.

Suggests location-based activities based on detected emotion.
Fully local logic — no LLM call. Uses OpenStreetMap Nominatim
for geocoding with config-based fallback locations.
"""

import logging
import time
from typing import Optional

import requests

from src.config import CONFIG

logger = logging.getLogger(__name__)

_GEO_CONFIG = CONFIG["geolocation"]


class GeoLocator:
    """Suggest activities and geocode locations.

    Activity mapping:
        sadness/fear → Park (nature, calm)
        joy/love → Plaza (social, celebration)
        anger → Gym (physical outlet)
        other → City Center (neutral)

    Args:
        config: Geolocation configuration section.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._config = config or _GEO_CONFIG

    def get_activity(self, emotion: str, user_location: str) -> dict:
        """Suggest a location-based activity for the detected emotion.

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

        fallback = self._config["fallback_locations"].get(emotion_key, {})
        if fallback:
            return {
                "text": fallback.get("name", "Local activity"),
                "lat": fallback.get("lat"),
                "lon": fallback.get("lon"),
            }

        return {"text": "Location unavailable", "lat": None, "lon": None}

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
        timeout = self._config.get("timeout_s", 5)
        base_url = self._config.get(
            "nominatim_url", "https://nominatim.openstreetmap.org/search"
        )

        queries = (
            [f"{place_name}, {city_name}", place_name]
            if city_name
            else [place_name]
        )

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
    


    