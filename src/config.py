"""Configuration loader for MindCare Agent."""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    """Load the YAML configuration file.

    Args:
        path: Path to the config file. Defaults to config/config.yaml
            resolved from the project root.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Configuration loaded from %s", path)
    return config


CONFIG = load_config()
PROJECT_ROOT = Path(__file__).resolve().parent.parent