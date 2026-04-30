"""
config.py
Loads and provides access to configs/config.yaml.
"""

import yaml
from pathlib import Path

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"


def load_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> dict:
    """Load YAML config file and return as nested dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
