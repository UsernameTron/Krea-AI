"""Tests for optimizers/neural_engine.py — NeuralEngineOptimizer."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import FluxConfig
from optimizers.neural_engine import NeuralEngineOptimizer


class TestNeuralEngineOptimizerInit:
    def test_init_creates_cache_dir(self, tmp_path):
        config = FluxConfig()
        cache = str(tmp_path / "ne_cache")
        opt = NeuralEngineOptimizer(config, cache_dir=cache)
        assert Path(cache).exists()

    def test_init_stores_config(self):
        config = FluxConfig()
        opt = NeuralEngineOptimizer(config)
        assert opt.config is config

    def test_init_lazy_coreml_check(self):
        config = FluxConfig()
        opt = NeuralEngineOptimizer(config)
        assert opt._coreml_available is None

    def test_init_compiled_count_zero(self):
        config = FluxConfig()
        opt = NeuralEngineOptimizer(config)
        assert opt._compiled_count == 0


class TestIsAvailable:
    def test_is_available_returns_false_when_coremltools_missing(self):
        config = FluxConfig()
        opt = NeuralEngineOptimizer(config)
        with patch.dict(sys.modules, {"coremltools": None}):
            result = opt.is_available
        assert result is False

    def test_is_available_returns_true_when_coremltools_present(self):
        config = FluxConfig()
        opt = NeuralEngineOptimizer(config)

        mock_coreml = MagicMock()
        with patch.dict(sys.modules, {"coremltools": mock_coreml}):
            # Reset lazy cache to force re-check
            opt._coreml_available = None
            result = opt.is_available
        assert result is True

    def test_is_available_cached_after_first_call(self):
        config = FluxConfig()
        opt = NeuralEngineOptimizer(config)

        call_count = [0]
        original_check = opt._check_coreml

        def counting_check():
            call_count[0] += 1
            return False

        opt._check_coreml = counting_check

        # Call twice
        _ = opt.is_available
        _ = opt.is_available

        assert call_count[0] == 1

    def test_is_available_False_when_coremltools_raises(self):
        config = FluxConfig()
        opt = NeuralEngineOptimizer(config)
        opt._coreml_available = None

        with patch.dict(sys.modules, {"coremltools": None}):
            result = opt.is_available
        assert result is False


class TestCheckCoreml:
    def test_returns_false_when_coremltools_not_installed(self):
        config = FluxConfig()
        opt = NeuralEngineOptimizer(config)
        with patch.dict(sys.modules, {"coremltools": None}):
            result = opt._check_coreml()
        assert result is False

    def test_returns_true_when_coremltools_installed(self):
        config = FluxConfig()
        opt = NeuralEngineOptimizer(config)

        mock_ct = MagicMock()
        with patch.dict(sys.modules, {"coremltools": mock_ct}):
            result = opt._check_coreml()
        assert result is True


class TestOptimizePipeline:
    def test_returns_pipeline_when_coreml_unavailable(self):
        config = FluxConfig()
        opt = NeuralEngineOptimizer(config)
        opt._coreml_available = False  # Skip lazy check

        mock_pipeline = MagicMock()
        result = opt.optimize_pipeline(mock_pipeline)
        assert result is mock_pipeline

    def test_returns_pipeline_unchanged_when_coreml_available(self):
        """optimize_pipeline is a no-op even when CoreML is available (not yet applied)."""
        config = FluxConfig()
        opt = NeuralEngineOptimizer(config)
        opt._coreml_available = True

        mock_pipeline = MagicMock()
        result = opt.optimize_pipeline(mock_pipeline)
        assert result is mock_pipeline

    def test_does_not_raise(self):
        config = FluxConfig()
        opt = NeuralEngineOptimizer(config)
        opt._coreml_available = True

        mock_pipeline = MagicMock()
        opt.optimize_pipeline(mock_pipeline)  # Should not raise


class TestGetStats:
    def test_returns_dict(self):
        config = FluxConfig()
        opt = NeuralEngineOptimizer(config)
        opt._coreml_available = False
        result = opt.get_stats()
        assert isinstance(result, dict)

    def test_contains_neural_engine_available(self):
        config = FluxConfig()
        opt = NeuralEngineOptimizer(config)
        opt._coreml_available = False
        result = opt.get_stats()
        assert "neural_engine_available" in result

    def test_contains_compiled_models(self):
        config = FluxConfig()
        opt = NeuralEngineOptimizer(config)
        opt._coreml_available = False
        result = opt.get_stats()
        assert "compiled_models" in result
        assert result["compiled_models"] == 0

    def test_contains_cache_dir(self, tmp_path):
        config = FluxConfig()
        cache = str(tmp_path / "ne_cache2")
        opt = NeuralEngineOptimizer(config, cache_dir=cache)
        opt._coreml_available = False
        result = opt.get_stats()
        assert "cache_dir" in result
        assert cache in result["cache_dir"]

    def test_acceleration_ratio_is_none(self):
        config = FluxConfig()
        opt = NeuralEngineOptimizer(config)
        opt._coreml_available = False
        result = opt.get_stats()
        assert result["acceleration_ratio"] is None

    def test_neural_engine_available_false_when_coreml_missing(self):
        config = FluxConfig()
        opt = NeuralEngineOptimizer(config)
        with patch.dict(sys.modules, {"coremltools": None}):
            opt._coreml_available = None  # Force re-check
            result = opt.get_stats()
        assert result["neural_engine_available"] is False

    def test_contains_status_field(self):
        config = FluxConfig()
        opt = NeuralEngineOptimizer(config)
        opt._coreml_available = False
        result = opt.get_stats()
        assert "status" in result
        assert isinstance(result["status"], str)
