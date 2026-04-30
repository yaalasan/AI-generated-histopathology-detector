"""
tests/test_model.py
Basic sanity checks - no dataset required.
Run with: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from src.config import load_config
from src.model import build_model


@pytest.fixture
def cfg():
    return load_config("configs/config.yaml")


def test_model_builds(cfg):
    model = build_model(cfg)
    assert model is not None


def test_model_output_shape(cfg):
    model = build_model(cfg)
    model.eval()
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 2), f"Expected (2,2), got {out.shape}"


def test_only_fc_trainable(cfg):
    """When freeze_backbone=True, only the final FC layer should be trainable."""
    cfg["model"]["freeze_backbone"] = True
    model = build_model(cfg)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all("fc" in n for n in trainable), \
        f"Non-FC params are trainable: {[n for n in trainable if 'fc' not in n]}"


def test_config_loads(cfg):
    assert "data" in cfg
    assert "training" in cfg
    assert "model" in cfg
