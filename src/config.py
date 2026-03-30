"""Configuration loader for MindCare Agent.

Loads config.yaml and validates:
    - Required sections exist (ml, agent, carbon, scoring)
    - Safety regex patterns are valid (fail-fast on malformed patterns)
    - Scoring thresholds are consistent (low < high)
    - Signal weights are between 0 and 1

No Pydantic — validation is explicit and lightweight.
"""

import logging
import re
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

# Sections that MUST exist in config.yaml
_REQUIRED_SECTIONS = ["paths", "ml", "agent", "carbon", "scoring"]


def _validate_config(config: dict) -> None:
    """Validate configuration schema and constraints.

    Checks:
        1. All required sections are present.
        2. Scoring thresholds are consistent (low < high).
        3. Signal weights are in [0, 1].
        4. Safety regex patterns compile without errors.

    Args:
        config: Parsed YAML configuration.

    Raises:
        ValueError: If any validation check fails.
    """
    # 1. Required sections
    for section in _REQUIRED_SECTIONS:
        if section not in config:
            raise ValueError(
                f"Missing required config section: '{section}'. "
                f"Check config/config.yaml."
            )

    # 2. Scoring thresholds
    scoring = config.get("scoring", {})
    low = scoring.get("low_threshold", 0.35)
    high = scoring.get("high_threshold", 0.60)
    if low >= high:
        raise ValueError(
            f"scoring.low_threshold ({low}) must be < "
            f"scoring.high_threshold ({high})"
        )

    # 3. Signal weights
    weights = scoring.get("weights", {})
    for signal_name, weight in weights.items():
        if not 0.0 <= weight <= 1.0:
            raise ValueError(
                f"Signal weight '{signal_name}' = {weight} is out of range [0, 1]"
            )

    # 4. Safety regex patterns
    patterns = scoring.get("safety_patterns", {})
    for pattern_str in patterns.get("implicit", []):
        try:
            re.compile(pattern_str, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(
                f"Invalid safety regex pattern: '{pattern_str}' — {exc}"
            ) from exc

    logger.info("Configuration validated successfully")


def load_config(path: Path = _CONFIG_PATH) -> dict:
    """Load and validate the YAML configuration file.

    Args:
        path: Path to the config file. Defaults to config/config.yaml
            resolved from the project root.

    Returns:
        Validated configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config fails validation.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _validate_config(config)
    logger.info("Configuration loaded from %s", path)
    return config


CONFIG = load_config()
PROJECT_ROOT = Path(__file__).resolve().parent.parent