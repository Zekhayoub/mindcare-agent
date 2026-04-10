"""Agent utilities: location extraction, chat formatting.

Location detection currently uses regex patterns — will be
replaced by spaCy NER.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

import spacy

# Load spaCy model once at module level
try:
    _nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy NER model loaded (en_core_web_sm)")
except OSError:
    _nlp = None
    logger.warning(
        "spaCy model 'en_core_web_sm' not found. "
        "Run: python -m spacy download en_core_web_sm"
    )

# False positives: common words that spaCy sometimes tags as GPE
_GPE_FALSE_POSITIVES = {
    "general", "particular", "real", "local", "personal",
    "mental", "physical", "emotional", "social", "overall",
}

def extract_location(text: str) -> Optional[str]:
    """Detect a location name in user input using regex.

    Looks for patterns like "I am in Brussels", "I'm near Paris",
    "located in London".

    Known issue: "I feel sad in general" detects "General" as a city.
    This will be fixed by replacing regex with spaCy NER.

    Args:
        text: Raw user message.

    Returns:
        Detected location name, or None.
    """
    if not text:
        return None

    patterns = [
        r"\b(?:in|at|near|from|around)\s+([A-Z][a-zA-Z\s]+)",
        r"\b(?:located|situated|based)\s+(?:in|at|near)\s+([A-Z][a-zA-Z\s]+)",
        r"\b([A-Z][a-zA-Z]+)\s+(?:city|area|region)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            location = match.group(1).strip()
            location = re.sub(
                r"\s+(?:city|area|region)$", "", location, flags=re.IGNORECASE
            )
            if len(location) > 2:
                return location

    return None

def extract_entities(text: str) -> dict:
    """Extract named entities from user input using spaCy NER.

    IMPORTANT: spaCy needs original text (with capitalization) to
    detect GPE and PERSON entities accurately. The ML classifier
    should receive lowercased text separately. Do NOT lowercase
    the input before passing it to this function.

    The calling code should:
        1. Call extract_entities(original_text) for NER
        2. Call classifier.classify(original_text) — the classifier
           handles lowercasing internally via TF-IDF

    Returns location (GPE) for activity recommendation and
    person names (PERSON) for PII detection.

    Falls back to regex if spaCy model is not loaded.

    Args:
        text: Raw user message.

    Returns:
        Dictionary with keys:
            - location: detected city/country name or None
            - persons: list of detected person names
    """
    if _nlp is None:
        # Fallback to legacy regex if spaCy not available
        return {
            "location": extract_location(text),
            "persons": [],
        }

    doc = _nlp(text)

    # Extract first GPE, filtering known false positives
    location = None
    for ent in doc.ents:
        if ent.label_ == "GPE" and ent.text.lower() not in _GPE_FALSE_POSITIVES:
            location = ent.text
            break

    # Extract all PERSON entities for PII detection
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    return {
        "location": location,
        "persons": persons,
    }


def mask_pii(text: str, entities: Optional[dict] = None) -> str:
    """Mask personally identifiable information in text.

    Replaces detected PERSON names with <PERSON> placeholder.
    The original text is never sent to the LLM — only the masked
    version. Location (GPE) is stored separately in the LangGraph
    state for activity recommendation.

    Note: This is a lightweight PII detection using spaCy NER.

    Args:
        text: Original user message.
        entities: Pre-extracted entities from extract_entities().
            If None, extracts them automatically.

    Returns:
        Text with PERSON names replaced by <PERSON>.
    """
    if entities is None:
        entities = extract_entities(text)

    masked = text
    # Replace longest names first to avoid partial replacements
    for person in sorted(entities.get("persons", []), key=len, reverse=True):
        masked = masked.replace(person, "<PERSON>")

    if entities.get("persons"):
        logger.info(
            "PII masked: %d person name(s) replaced", len(entities["persons"])
        )

    return masked


def format_chat_history(chat_history: list, max_messages: int = 5) -> str:
    """Convert a list of LangChain messages to a string.

    Args:
        chat_history: List of HumanMessage/AIMessage objects.
        max_messages: Maximum number of recent messages to include.

    Returns:
        Formatted conversation string.
    """
    if not chat_history:
        return ""

    recent = chat_history[-max_messages:]
    lines = []
    for msg in recent:
        if hasattr(msg, "content") and hasattr(msg, "type"):
            prefix = "Human" if msg.type == "human" else "AI"
            lines.append(f"{prefix}: {msg.content}")
        else:
            lines.append(str(msg))

    return "\n".join(lines)

