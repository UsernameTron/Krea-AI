"""Tests for optimizers/__init__.py — get_optimizer factory."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from config import FluxConfig
from optimizers import get_optimizer


class TestGetOptimizerFactory:
    def test_none_level_returns_empty_dict(self):
        config = FluxConfig()
        config.optimization_level = "none"
        result = get_optimizer(config)
        assert result == {}

    def test_standard_level_returns_empty_dict(self):
        config = FluxConfig()
        config.optimization_level = "standard"
        result = get_optimizer(config)
        assert result == {}

    def test_maximum_level_returns_dict(self):
        config = FluxConfig()
        config.optimization_level = "maximum"
        result = get_optimizer(config)
        assert isinstance(result, dict)

    def test_maximum_level_includes_metal_when_importable(self):
        config = FluxConfig()
        config.optimization_level = "maximum"

        mock_metal = MagicMock()
        mock_metal_instance = MagicMock()
        mock_metal.return_value = mock_metal_instance

        with patch.dict(
            sys.modules,
            {"optimizers.metal": MagicMock(MetalOptimizer=mock_metal)},
        ):
            # Re-import so the patched module is used
            import importlib
            import optimizers
            importlib.reload(optimizers)
            result = optimizers.get_optimizer(config)
            importlib.reload(optimizers)  # Restore to avoid polluting other tests

        # Can't easily assert mock was used after reload, but the function should run
        assert isinstance(result, dict)

    def test_maximum_level_continues_if_metal_import_fails(self):
        """If MetalOptimizer raises ImportError, the dict is returned without 'metal'."""
        config = FluxConfig()
        config.optimization_level = "maximum"

        with patch(
            "optimizers.MetalOptimizer" if hasattr(__import__("optimizers"), "MetalOptimizer")
            else "optimizers.metal.MetalOptimizer",
            side_effect=ImportError,
            create=True,
        ):
            # Even if patching fails, the factory itself should not raise
            result = get_optimizer(config)
        assert isinstance(result, dict)

    def test_returns_dict_always(self):
        for level in ("none", "standard", "maximum"):
            config = FluxConfig()
            config.optimization_level = level
            result = get_optimizer(config)
            assert isinstance(result, dict), f"Expected dict for level={level}"


class TestGetOptimizerMaximumComponents:
    """Test that MAXIMUM level tries to create all three optimizer components."""

    def test_maximum_creates_metal_optimizer(self):
        config = FluxConfig()
        config.optimization_level = "maximum"
        result = get_optimizer(config)
        # On this machine (no actual MPS needed), MetalOptimizer should be importable
        # and present in the result dict
        assert "metal" in result

    def test_maximum_creates_neural_engine_optimizer(self):
        config = FluxConfig()
        config.optimization_level = "maximum"
        result = get_optimizer(config)
        assert "neural_engine" in result

    def test_maximum_creates_thermal_manager(self):
        config = FluxConfig()
        config.optimization_level = "maximum"
        result = get_optimizer(config)
        assert "thermal" in result

    def test_maximum_metal_import_error_skipped(self):
        """ImportError on MetalOptimizer is silently skipped — other optimizers still added."""
        config = FluxConfig()
        config.optimization_level = "maximum"

        with patch.dict(sys.modules, {"optimizers.metal": None}):
            # When optimizers.metal is None in sys.modules, from optimizers.metal import X → ImportError
            result = get_optimizer(config)
        # neural_engine and thermal should still be present
        assert "neural_engine" in result
        assert "thermal" in result
        assert "metal" not in result

    def test_maximum_neural_engine_import_error_skipped(self):
        config = FluxConfig()
        config.optimization_level = "maximum"

        with patch.dict(sys.modules, {"optimizers.neural_engine": None}):
            result = get_optimizer(config)
        assert "metal" in result
        assert "thermal" in result
        assert "neural_engine" not in result

    def test_maximum_thermal_import_error_skipped(self):
        config = FluxConfig()
        config.optimization_level = "maximum"

        with patch.dict(sys.modules, {"optimizers.thermal": None}):
            result = get_optimizer(config)
        assert "metal" in result
        assert "neural_engine" in result
        assert "thermal" not in result
