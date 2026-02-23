"""Shared fixtures for FLUX.1 Krea tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def test_config():
    """Create a FluxConfig with test-friendly defaults."""
    from config import FluxConfig

    config = FluxConfig()
    config.generation.width = 512
    config.generation.height = 512
    config.generation.steps = 4
    config.optimization_level = "none"
    return config


@pytest.fixture
def mock_diffusers_pipeline(monkeypatch):
    """Mock diffusers.FluxPipeline.from_pretrained to avoid loading the 24GB model."""
    mock_pipe = MagicMock()
    mock_pipe.to = MagicMock(return_value=mock_pipe)
    mock_pipe.enable_attention_slicing = MagicMock()
    mock_pipe.enable_vae_tiling = MagicMock()
    mock_pipe.enable_vae_slicing = MagicMock()
    mock_pipe.enable_model_cpu_offload = MagicMock()

    # Mock __call__ to return a fake image result
    from PIL import Image

    fake_image = Image.new("RGB", (512, 512), color="red")
    mock_result = MagicMock()
    mock_result.images = [fake_image]
    mock_pipe.return_value = mock_result

    mock_from_pretrained = MagicMock(return_value=mock_pipe)

    import diffusers

    monkeypatch.setattr(diffusers, "FluxPipeline", MagicMock(from_pretrained=mock_from_pretrained))

    return mock_pipe, mock_from_pretrained


requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available",
)
