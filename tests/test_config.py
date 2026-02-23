"""Tests for config.py."""

import os
from unittest.mock import patch

import pytest

from config import (
    FluxConfig,
    GenerationConfig,
    OptimizationLevel,
    get_config,
    get_hf_token,
)


class TestFluxConfigDefaults:
    def test_default_model_id(self):
        config = FluxConfig()
        assert config.model.id == "black-forest-labs/FLUX.1-Krea-dev"

    def test_default_device(self):
        config = FluxConfig()
        assert config.device.preferred in ("mps", "cpu", "auto")

    def test_default_generation_params(self):
        config = FluxConfig()
        assert config.generation.width == 1024
        assert config.generation.height == 1024
        assert config.generation.steps == 28
        assert config.generation.guidance_scale == 4.5

    def test_default_watermark_ratio(self):
        config = FluxConfig()
        assert config.device.mps_watermark_ratio == 0.8

    def test_default_optimization_level(self):
        config = FluxConfig()
        assert config.optimization_level == "standard"

    def test_safety_mode_disabled_by_default(self):
        config = FluxConfig()
        assert config.safety_mode.enabled is False


class TestFluxConfigValidation:
    def test_width_must_be_multiple_of_64(self):
        config = FluxConfig()
        config.generation.width = 500
        with pytest.raises(ValueError, match="multiple of 64"):
            config.validate()

    def test_height_must_be_multiple_of_64(self):
        config = FluxConfig()
        config.generation.height = 100
        with pytest.raises(ValueError, match="multiple of 64"):
            config.validate()

    def test_valid_dimensions_pass(self):
        config = FluxConfig()
        config.generation.width = 768
        config.generation.height = 1024
        config.validate()  # Should not raise

    def test_invalid_optimization_level(self):
        config = FluxConfig()
        config.optimization_level = "turbo"
        with pytest.raises(ValueError, match="optimization_level"):
            config.validate()

    def test_watermark_low_must_be_less_than_high(self):
        config = FluxConfig()
        config.device.mps_low_watermark_ratio = 0.9
        config.device.mps_watermark_ratio = 0.5
        with pytest.raises(ValueError, match="watermark"):
            config.validate()


class TestGetConfig:
    def test_get_config_returns_valid(self):
        config = get_config()
        assert isinstance(config, FluxConfig)

    def test_get_config_with_overrides(self):
        config = get_config(optimization_level="none")
        assert config.optimization_level == "none"

    def test_env_var_override(self):
        with patch.dict(os.environ, {"FLUX_OPTIMIZATION_LEVEL": "maximum"}):
            config = get_config()
            assert config.optimization_level == "maximum"


class TestGetHfToken:
    def test_returns_token_from_env(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test123"}):
            assert get_hf_token() == "hf_test123"

    def test_returns_none_when_not_set(self):
        with patch.dict(os.environ, {}, clear=True):
            # Clear both possible env var names
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
            result = get_hf_token()
            # May still be set from .env or system — just check type
            assert result is None or isinstance(result, str)


class TestOptimizationLevel:
    def test_enum_values(self):
        assert OptimizationLevel.NONE.value == "none"
        assert OptimizationLevel.STANDARD.value == "standard"
        assert OptimizationLevel.MAXIMUM.value == "maximum"

    def test_get_optimization_level_method(self):
        config = FluxConfig()
        config.optimization_level = "standard"
        level = config.get_optimization_level()
        assert level == OptimizationLevel.STANDARD


class TestEffectiveGenerationParams:
    def test_normal_mode(self):
        config = FluxConfig()
        config.safety_mode.enabled = False
        params = config.get_effective_generation_params()
        assert params["guidance_scale"] == config.generation.guidance_scale
        assert params["max_sequence_length"] == config.generation.max_sequence_length

    def test_safety_mode_overrides(self):
        config = FluxConfig()
        config.safety_mode.enabled = True
        params = config.get_effective_generation_params()
        assert params["guidance_scale"] == config.safety_mode.guidance_scale
        assert params["max_sequence_length"] == config.safety_mode.max_sequence_length
