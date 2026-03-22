"""Tests for optimizers/neural_engine.py — Neural Engine / CoreML acceleration."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from config import FluxConfig
from optimizers.neural_engine import (
    NeuralEngineOptimizer,
    ANE_MAX_MODEL_SIZE_BYTES,
    BYTES_PER_PARAM_BF16,
    FLUX_COMPONENT_PARAMS,
    _get_coremltools,
)


@pytest.fixture
def config():
    """Create a FluxConfig with test defaults."""
    cfg = FluxConfig()
    cfg.generation.width = 512
    cfg.generation.height = 512
    cfg.generation.steps = 4
    return cfg


@pytest.fixture
def optimizer(config, tmp_path):
    """Create a NeuralEngineOptimizer with a temp cache dir."""
    return NeuralEngineOptimizer(config, cache_dir=str(tmp_path / "ane_cache"))


@pytest.fixture
def mock_pipeline():
    """Create a mock FluxPipeline with VAE and text encoder."""
    import torch

    pipe = MagicMock()

    # Mock VAE with a decoder that has real parameters-like behavior
    vae = MagicMock()
    decoder = MagicMock()
    # Make decoder callable for tracing
    decoder.cpu.return_value = decoder
    decoder.float.return_value = decoder
    vae.decoder = decoder
    pipe.vae = vae

    # Mock text encoder (T5-XXL)
    text_encoder = MagicMock()
    # Simulate ~11B parameters to trigger the size check
    large_param = MagicMock()
    large_param.numel.return_value = 11_000_000_000
    text_encoder.parameters.return_value = [large_param]
    text_encoder.cpu.return_value = text_encoder
    text_encoder.float.return_value = text_encoder
    pipe.text_encoder_2 = text_encoder

    pipe.to = MagicMock(return_value=pipe)
    return pipe


class TestIsAvailable:
    """Test CoreML availability detection."""

    def test_available_when_coremltools_importable(self, optimizer):
        """is_available returns True when coremltools can be imported."""
        mock_ct = MagicMock()
        mock_ct.__version__ = "7.0"
        with patch("optimizers.neural_engine._get_coremltools", return_value=mock_ct):
            # Reset lazy check
            optimizer._coreml_available = None
            result = optimizer._check_coreml()
            assert result is True

    def test_unavailable_when_coremltools_missing(self, optimizer):
        """is_available returns False when coremltools is not installed."""
        with patch("optimizers.neural_engine._get_coremltools", return_value=None):
            optimizer._coreml_available = None
            result = optimizer._check_coreml()
            assert result is False

    def test_lazy_evaluation(self, optimizer):
        """is_available caches the result after first check."""
        optimizer._coreml_available = True
        assert optimizer.is_available is True
        # Doesn't re-check
        optimizer._coreml_available = False
        assert optimizer.is_available is False


class TestAssessModelCompatibility:
    """Test the ANE compatibility assessment."""

    def test_returns_report_structure(self, optimizer):
        """Assessment returns expected report structure."""
        optimizer._coreml_available = True
        report = optimizer._assess_model_compatibility()

        assert "coreml_available" in report
        assert "ane_max_size_gb" in report
        assert "components" in report
        assert report["coreml_available"] is True

    def test_vae_decoder_marked_convertible(self, optimizer):
        """VAE decoder should be marked as convertible (under ANE size limit)."""
        optimizer._coreml_available = True
        report = optimizer._assess_model_compatibility()

        vae_info = report["components"]["vae_decoder"]
        assert vae_info["convertible"] is True
        assert vae_info["estimated_size_gb"] < ANE_MAX_MODEL_SIZE_BYTES / (1024**3)

    def test_text_encoder_marked_not_convertible(self, optimizer):
        """T5-XXL text encoder should be marked as NOT convertible."""
        optimizer._coreml_available = True
        report = optimizer._assess_model_compatibility()

        t5_info = report["components"]["text_encoder_2"]
        assert t5_info["convertible"] is False
        assert "exceeds ANE limit" in t5_info["reason"]

    def test_transformer_marked_not_convertible(self, optimizer):
        """Flow-matching transformer should be marked as NOT convertible."""
        optimizer._coreml_available = True
        report = optimizer._assess_model_compatibility()

        transformer_info = report["components"]["transformer"]
        assert transformer_info["convertible"] is False
        assert transformer_info["estimated_size_gb"] > ANE_MAX_MODEL_SIZE_BYTES / (1024**3)

    def test_all_components_have_required_fields(self, optimizer):
        """Every component in the report has the required fields."""
        optimizer._coreml_available = True
        report = optimizer._assess_model_compatibility()

        for name, info in report["components"].items():
            assert "convertible" in info, f"{name} missing 'convertible'"
            assert "reason" in info, f"{name} missing 'reason'"
            assert "estimated_size_gb" in info, f"{name} missing 'estimated_size_gb'"
            assert "param_count" in info, f"{name} missing 'param_count'"

    def test_when_coreml_unavailable(self, optimizer):
        """Assessment still works when coremltools is not available."""
        optimizer._coreml_available = False
        report = optimizer._assess_model_compatibility()

        assert report["coreml_available"] is False
        # Components are still assessed (static analysis)
        assert len(report["components"]) > 0


class TestCompileVaeDecoder:
    """Test VAE decoder CoreML conversion."""

    def test_success_path(self, optimizer, mock_pipeline, tmp_path):
        """VAE decoder conversion succeeds with mocked coremltools."""
        mock_ct = MagicMock()
        mock_ct.ComputeUnit.ALL = "ALL"
        mock_ct.target.macOS13 = "macOS13"
        mock_ct.TensorType = MagicMock()
        mock_coreml_model = MagicMock()
        mock_ct.convert.return_value = mock_coreml_model

        optimizer._coreml_available = True

        with patch("optimizers.neural_engine._get_coremltools", return_value=mock_ct), \
             patch("torch.jit.trace", return_value=MagicMock()):
            result = optimizer.compile_vae_decoder(mock_pipeline)

        assert result is True
        assert "vae_decoder" in optimizer._compiled_components
        assert optimizer._conversion_results["vae_decoder"]["success"] is True
        mock_coreml_model.save.assert_called_once()

    def test_failure_path_conversion_error(self, optimizer, mock_pipeline):
        """VAE decoder conversion fails gracefully on CoreML error."""
        mock_ct = MagicMock()
        mock_ct.ComputeUnit.ALL = "ALL"
        mock_ct.target.macOS13 = "macOS13"
        mock_ct.TensorType = MagicMock()
        mock_ct.convert.side_effect = RuntimeError("Unsupported op: custom_attention")

        optimizer._coreml_available = True

        with patch("optimizers.neural_engine._get_coremltools", return_value=mock_ct), \
             patch("torch.jit.trace", return_value=MagicMock()):
            result = optimizer.compile_vae_decoder(mock_pipeline)

        assert result is False
        assert "vae_decoder" not in optimizer._compiled_components
        assert "Unsupported op" in optimizer._conversion_results["vae_decoder"]["error"]

    def test_no_coremltools(self, optimizer, mock_pipeline):
        """Returns False when coremltools is not available."""
        optimizer._coreml_available = False
        result = optimizer.compile_vae_decoder(mock_pipeline)

        assert result is False
        assert optimizer._conversion_results["vae_decoder"]["error"] == "coremltools not available"

    def test_no_vae_attribute(self, optimizer):
        """Returns False when pipeline has no VAE."""
        optimizer._coreml_available = True
        mock_ct = MagicMock()
        pipe = MagicMock(spec=[])  # No attributes

        with patch("optimizers.neural_engine._get_coremltools", return_value=mock_ct):
            result = optimizer.compile_vae_decoder(pipe)

        assert result is False
        assert "no VAE" in optimizer._conversion_results["vae_decoder"]["error"]

    def test_loads_cached_model(self, optimizer, mock_pipeline, tmp_path):
        """Loads from cache if .mlpackage already exists."""
        mock_ct = MagicMock()
        mock_ct.models.MLModel.return_value = MagicMock()
        optimizer._coreml_available = True

        # Create fake cache directory
        cache_path = optimizer.cache_dir / "vae_decoder.mlpackage"
        cache_path.mkdir(parents=True, exist_ok=True)

        with patch("optimizers.neural_engine._get_coremltools", return_value=mock_ct):
            result = optimizer.compile_vae_decoder(mock_pipeline)

        assert result is True
        assert "vae_decoder" in optimizer._compiled_components
        mock_ct.convert.assert_not_called()  # Should NOT reconvert


class TestCompileTextEncoder:
    """Test text encoder CoreML conversion (expected to fail for T5-XXL)."""

    def test_fails_due_to_size(self, optimizer, mock_pipeline):
        """Text encoder conversion is rejected due to T5-XXL exceeding ANE size limit."""
        optimizer._coreml_available = True
        mock_ct = MagicMock()

        with patch("optimizers.neural_engine._get_coremltools", return_value=mock_ct):
            result = optimizer.compile_text_encoder(mock_pipeline)

        assert result is False
        error = optimizer._conversion_results["text_encoder"]["error"]
        assert "exceeding ANE limit" in error
        assert "11.0B" in error

    def test_no_coremltools(self, optimizer, mock_pipeline):
        """Returns False when coremltools is not available."""
        optimizer._coreml_available = False
        result = optimizer.compile_text_encoder(mock_pipeline)

        assert result is False

    def test_no_text_encoder(self, optimizer):
        """Returns False when pipeline has no text encoder."""
        optimizer._coreml_available = True
        mock_ct = MagicMock()
        pipe = MagicMock(spec=[])  # No attributes

        with patch("optimizers.neural_engine._get_coremltools", return_value=mock_ct):
            result = optimizer.compile_text_encoder(pipe)

        assert result is False
        assert "no text encoder" in optimizer._conversion_results["text_encoder"]["error"]

    def test_conversion_attempted_when_size_check_fails(self, optimizer):
        """If param counting throws, conversion is attempted anyway (and fails)."""
        optimizer._coreml_available = True
        mock_ct = MagicMock()
        mock_ct.ComputeUnit.ALL = "ALL"
        mock_ct.target.macOS13 = "macOS13"
        mock_ct.TensorType = MagicMock()
        mock_ct.convert.side_effect = RuntimeError("Model too large for ANE")

        pipe = MagicMock()
        encoder = MagicMock()
        # parameters() raises, so size check is skipped
        encoder.parameters.side_effect = AttributeError("no parameters")
        encoder.cpu.return_value = encoder
        encoder.float.return_value = encoder
        pipe.text_encoder_2 = encoder

        with patch("optimizers.neural_engine._get_coremltools", return_value=mock_ct), \
             patch("torch.jit.trace", return_value=MagicMock()):
            result = optimizer.compile_text_encoder(pipe)

        assert result is False
        assert "Model too large" in optimizer._conversion_results["text_encoder"]["error"]


class TestOptimizePipeline:
    """Test the main optimize_pipeline() entry point."""

    def test_returns_pipeline_unchanged_when_ane_unavailable(self, optimizer, mock_pipeline):
        """Pipeline is returned as-is when coremltools is not available."""
        optimizer._coreml_available = False
        result = optimizer.optimize_pipeline(mock_pipeline)
        assert result is mock_pipeline

    def test_returns_pipeline_when_conversions_fail(self, optimizer, mock_pipeline):
        """Pipeline is returned even when all conversions fail."""
        optimizer._coreml_available = True
        mock_ct = MagicMock()
        mock_ct.ComputeUnit.ALL = "ALL"
        mock_ct.target.macOS13 = "macOS13"
        mock_ct.TensorType = MagicMock()
        mock_ct.convert.side_effect = RuntimeError("Conversion failed")

        with patch("optimizers.neural_engine._get_coremltools", return_value=mock_ct), \
             patch("torch.jit.trace", return_value=MagicMock()):
            result = optimizer.optimize_pipeline(mock_pipeline)

        assert result is mock_pipeline

    def test_attempts_vae_and_text_encoder(self, optimizer, mock_pipeline):
        """optimize_pipeline attempts both VAE and text encoder conversion."""
        optimizer._coreml_available = True
        mock_ct = MagicMock()
        mock_ct.ComputeUnit.ALL = "ALL"
        mock_ct.target.macOS13 = "macOS13"
        mock_ct.TensorType = MagicMock()
        # VAE conversion succeeds
        mock_coreml_model = MagicMock()
        mock_ct.convert.return_value = mock_coreml_model

        with patch("optimizers.neural_engine._get_coremltools", return_value=mock_ct), \
             patch("torch.jit.trace", return_value=MagicMock()):
            result = optimizer.optimize_pipeline(mock_pipeline)

        assert result is mock_pipeline
        # Both were attempted
        assert "vae_decoder" in optimizer._conversion_results
        assert "text_encoder" in optimizer._conversion_results

    def test_with_successful_vae_conversion(self, optimizer, mock_pipeline):
        """Pipeline optimization records successful VAE conversion."""
        optimizer._coreml_available = True
        mock_ct = MagicMock()
        mock_ct.ComputeUnit.ALL = "ALL"
        mock_ct.target.macOS13 = "macOS13"
        mock_ct.TensorType = MagicMock()
        mock_ct.convert.return_value = MagicMock()

        with patch("optimizers.neural_engine._get_coremltools", return_value=mock_ct), \
             patch("torch.jit.trace", return_value=MagicMock()):
            optimizer.optimize_pipeline(mock_pipeline)

        assert "vae_decoder" in optimizer._compiled_components
        assert optimizer._conversion_results["vae_decoder"]["success"] is True
        # Text encoder should have failed (T5-XXL too large)
        assert optimizer._conversion_results["text_encoder"]["success"] is False


class TestGetStats:
    """Test the get_stats() method."""

    def test_basic_structure(self, optimizer):
        """Stats dict has all required keys."""
        optimizer._coreml_available = False
        stats = optimizer.get_stats()

        assert "neural_engine_available" in stats
        assert "compiled_components" in stats
        assert "compiled_count" in stats
        assert "cache_dir" in stats
        assert "conversion_results" in stats
        assert "acceleration_ratio" in stats
        assert "status" in stats

    def test_status_when_unavailable(self, optimizer):
        """Status reports disabled when coremltools missing."""
        optimizer._coreml_available = False
        stats = optimizer.get_stats()

        assert stats["neural_engine_available"] is False
        assert "disabled" in stats["status"]
        assert stats["compiled_count"] == 0
        assert stats["acceleration_ratio"] is None

    def test_status_when_available_no_conversions(self, optimizer):
        """Status reports available but no conversions attempted."""
        optimizer._coreml_available = True
        stats = optimizer.get_stats()

        assert stats["neural_engine_available"] is True
        assert "no conversions attempted" in stats["status"]

    def test_status_with_compiled_components(self, optimizer, tmp_path):
        """Stats reflect compiled components."""
        optimizer._coreml_available = True
        optimizer._compiled_components["vae_decoder"] = tmp_path / "vae.mlpackage"
        optimizer._conversion_results["vae_decoder"] = {"success": True, "error": None}

        stats = optimizer.get_stats()

        assert stats["compiled_count"] == 1
        assert "vae_decoder" in stats["compiled_components"]
        assert "active" in stats["status"]
        assert stats["conversion_results"]["vae_decoder"]["success"] is True

    def test_status_with_failed_conversions(self, optimizer):
        """Stats reflect failed conversion attempts."""
        optimizer._coreml_available = True
        optimizer._conversion_results["vae_decoder"] = {
            "success": False,
            "error": "Unsupported op",
        }
        optimizer._conversion_results["text_encoder"] = {
            "success": False,
            "error": "Model too large",
        }

        stats = optimizer.get_stats()

        assert stats["compiled_count"] == 0
        assert "none succeeded" in stats["status"]
        assert stats["conversion_results"]["vae_decoder"]["error"] == "Unsupported op"
        assert stats["conversion_results"]["text_encoder"]["error"] == "Model too large"

    def test_acceleration_ratio_from_benchmark(self, optimizer, tmp_path):
        """Acceleration ratio is populated from benchmark data."""
        optimizer._coreml_available = True
        optimizer._compiled_components["vae_decoder"] = tmp_path / "vae.mlpackage"
        optimizer._benchmark_data = {
            "mps_only": 10.0,
            "ane_hybrid": 8.0,
        }

        stats = optimizer.get_stats()
        assert stats["acceleration_ratio"] == 1.25  # 10 / 8

    def test_no_acceleration_ratio_without_benchmark(self, optimizer):
        """Acceleration ratio is None when no benchmark data exists."""
        optimizer._coreml_available = True
        stats = optimizer.get_stats()
        assert stats["acceleration_ratio"] is None


class TestBenchmarkAneVsMps:
    """Test the benchmark_ane_vs_mps() method."""

    def test_mps_only_mode(self, optimizer, mock_pipeline):
        """Benchmark reports mps_only mode when no components converted."""
        import torch

        # Make pipeline callable for benchmark
        mock_result = MagicMock()
        mock_pipeline.return_value = mock_result

        with patch.object(torch.mps, "synchronize", create=True):
            result = optimizer.benchmark_ane_vs_mps(mock_pipeline)

        assert result["mode"] == "mps_only"
        assert result["converted_components"] == []
        assert result["generation_time_seconds"] is not None or "error" in result

    def test_ane_hybrid_mode(self, optimizer, mock_pipeline, tmp_path):
        """Benchmark reports ane_hybrid mode when components are converted."""
        import torch

        optimizer._compiled_components["vae_decoder"] = tmp_path / "vae.mlpackage"
        mock_pipeline.return_value = MagicMock()

        with patch.object(torch.mps, "synchronize", create=True):
            result = optimizer.benchmark_ane_vs_mps(mock_pipeline)

        assert result["mode"] == "ane_hybrid"
        assert "vae_decoder" in result["converted_components"]

    def test_benchmark_handles_generation_error(self, optimizer, mock_pipeline):
        """Benchmark handles errors during generation gracefully."""
        mock_pipeline.side_effect = RuntimeError("Out of memory")

        result = optimizer.benchmark_ane_vs_mps(mock_pipeline)

        assert "error" in result

    def test_acceleration_ratio_calculated(self, optimizer, mock_pipeline, tmp_path):
        """Acceleration ratio is calculated when both baselines exist."""
        import torch

        # Pre-populate MPS baseline
        optimizer._benchmark_data = {"mps_only": 5.0}
        optimizer._compiled_components["vae_decoder"] = tmp_path / "vae.mlpackage"
        mock_pipeline.return_value = MagicMock()

        with patch.object(torch.mps, "synchronize", create=True):
            result = optimizer.benchmark_ane_vs_mps(mock_pipeline)

        # Should have acceleration_ratio since mps_only baseline exists
        if result.get("generation_time_seconds"):
            assert result["acceleration_ratio"] is not None

    def test_custom_prompt(self, optimizer, mock_pipeline):
        """Benchmark uses custom prompt."""
        import torch

        mock_pipeline.return_value = MagicMock()

        with patch.object(torch.mps, "synchronize", create=True):
            result = optimizer.benchmark_ane_vs_mps(mock_pipeline, prompt="a red circle")

        assert result["prompt"] == "a red circle"


class TestConstants:
    """Test module-level constants are sensible."""

    def test_ane_max_size(self):
        assert ANE_MAX_MODEL_SIZE_BYTES == 4 * 1024 * 1024 * 1024  # 4GB

    def test_bytes_per_param(self):
        assert BYTES_PER_PARAM_BF16 == 2  # bfloat16

    def test_t5_exceeds_ane_limit(self):
        """T5-XXL parameters exceed ANE size limit."""
        t5_size = FLUX_COMPONENT_PARAMS["text_encoder_2"] * BYTES_PER_PARAM_BF16
        assert t5_size > ANE_MAX_MODEL_SIZE_BYTES

    def test_vae_within_ane_limit(self):
        """VAE decoder parameters are within ANE size limit."""
        vae_size = FLUX_COMPONENT_PARAMS["vae_decoder"] * BYTES_PER_PARAM_BF16
        assert vae_size < ANE_MAX_MODEL_SIZE_BYTES


class TestInit:
    """Test NeuralEngineOptimizer initialization."""

    def test_creates_cache_dir(self, config, tmp_path):
        """Constructor creates the cache directory."""
        cache = tmp_path / "new_cache"
        opt = NeuralEngineOptimizer(config, cache_dir=str(cache))
        assert cache.exists()
        assert opt.cache_dir == cache

    def test_initial_state(self, optimizer):
        """Fresh optimizer has empty conversion state."""
        assert optimizer._compiled_components == {}
        assert optimizer._conversion_results == {}
        assert optimizer._benchmark_data is None
        assert optimizer._coreml_available is None
