"""Tests for pipeline.py."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import FluxConfig, OptimizationLevel, get_config


class TestFluxKreaPipelineInit:
    def test_init_none_level(self, test_config):
        from pipeline import FluxKreaPipeline

        test_config.optimization_level = "none"
        p = FluxKreaPipeline(test_config)
        assert p.is_loaded is False

    def test_init_standard_level(self, test_config):
        from pipeline import FluxKreaPipeline

        test_config.optimization_level = "standard"
        p = FluxKreaPipeline(test_config)
        assert p.is_loaded is False

    def test_init_maximum_level(self, test_config):
        from pipeline import FluxKreaPipeline

        test_config.optimization_level = "maximum"
        p = FluxKreaPipeline(test_config)
        assert p.is_loaded is False


class TestDeviceDetection:
    def test_detect_device_returns_torch_device(self, test_config):
        from pipeline import FluxKreaPipeline

        p = FluxKreaPipeline(test_config)
        device = p._detect_device()
        assert str(device) in ("mps", "cuda", "cpu")

    def test_cpu_fallback(self, test_config):
        from pipeline import FluxKreaPipeline

        test_config.device.preferred = "cpu"
        p = FluxKreaPipeline(test_config)
        device = p._detect_device()
        assert str(device) == "cpu"


class TestMemoryCleanup:
    def test_cleanup_does_not_raise(self, test_config):
        from pipeline import FluxKreaPipeline

        p = FluxKreaPipeline(test_config)
        p._cleanup_memory()  # Should not raise even without MPS


class TestPipelineWithMock:
    def test_load_calls_from_pretrained(self, test_config, mock_diffusers_pipeline):
        from pipeline import FluxKreaPipeline

        mock_pipe, mock_from_pretrained = mock_diffusers_pipeline
        test_config.optimization_level = "none"

        p = FluxKreaPipeline(test_config)
        p.load()

        mock_from_pretrained.assert_called_once()
        assert p.is_loaded is True

    def test_generate_returns_image_and_metrics(self, test_config, mock_diffusers_pipeline):
        from pipeline import FluxKreaPipeline

        mock_pipe, _ = mock_diffusers_pipeline
        test_config.optimization_level = "none"

        p = FluxKreaPipeline(test_config)
        p.load()

        image, metrics = p.generate(prompt="test", width=512, height=512, num_inference_steps=4)

        assert image is not None
        assert "generation_time" in metrics
        assert "time_per_step" in metrics
        assert "device" in metrics

    def test_unload_clears_pipeline(self, test_config, mock_diffusers_pipeline):
        from pipeline import FluxKreaPipeline

        test_config.optimization_level = "none"
        p = FluxKreaPipeline(test_config)
        p.load()
        assert p.is_loaded is True

        p.unload()
        assert p.is_loaded is False


class TestGetSystemInfo:
    def test_returns_dict(self, test_config):
        from pipeline import FluxKreaPipeline

        p = FluxKreaPipeline(test_config)
        info = p.get_system_info()
        assert isinstance(info, dict)
        assert "pytorch_version" in info
        assert "is_loaded" in info


class TestLoadFallbackChain:
    """Test the MAXIMUM -> STANDARD -> NONE fallback chain in load()."""

    def test_maximum_falls_back_to_standard_on_error(self, test_config):
        from pipeline import FluxKreaPipeline

        test_config.optimization_level = "maximum"
        p = FluxKreaPipeline(test_config)

        call_levels = []

        def fake_load_at_level(level, progress_callback=None):
            call_levels.append(level)
            if level == OptimizationLevel.MAXIMUM:
                raise RuntimeError("max failed")
            # STANDARD and NONE succeed

        with patch.object(p, "_load_at_level", side_effect=fake_load_at_level):
            p.load()

        assert OptimizationLevel.MAXIMUM in call_levels
        assert OptimizationLevel.STANDARD in call_levels
        assert p.is_loaded is True

    def test_maximum_falls_back_to_none_when_standard_also_fails(self, test_config):
        from pipeline import FluxKreaPipeline

        test_config.optimization_level = "maximum"
        p = FluxKreaPipeline(test_config)

        call_levels = []

        def fake_load_at_level(level, progress_callback=None):
            call_levels.append(level)
            if level in (OptimizationLevel.MAXIMUM, OptimizationLevel.STANDARD):
                raise RuntimeError("both failed")

        with patch.object(p, "_load_at_level", side_effect=fake_load_at_level):
            p.load()

        assert OptimizationLevel.NONE in call_levels
        assert p.is_loaded is True

    def test_standard_falls_back_to_none_on_error(self, test_config):
        from pipeline import FluxKreaPipeline

        test_config.optimization_level = "standard"
        p = FluxKreaPipeline(test_config)

        call_levels = []

        def fake_load_at_level(level, progress_callback=None):
            call_levels.append(level)
            if level == OptimizationLevel.STANDARD:
                raise RuntimeError("std failed")

        with patch.object(p, "_load_at_level", side_effect=fake_load_at_level):
            p.load()

        assert OptimizationLevel.NONE in call_levels
        assert p.is_loaded is True

    def test_none_level_raises_on_error(self, test_config):
        """If NONE level itself fails, the error must propagate."""
        from pipeline import FluxKreaPipeline

        test_config.optimization_level = "none"
        p = FluxKreaPipeline(test_config)

        with patch.object(
            p, "_load_at_level", side_effect=RuntimeError("none also failed")
        ):
            with pytest.raises(RuntimeError, match="none also failed"):
                p.load()

    def test_progress_callback_called_at_start_and_end(self, test_config):
        from pipeline import FluxKreaPipeline

        test_config.optimization_level = "none"
        p = FluxKreaPipeline(test_config)
        progress_calls = []

        with patch.object(p, "_load_at_level"):
            p.load(progress_callback=lambda f, m: progress_calls.append((f, m)))

        fractions = [f for f, _ in progress_calls]
        assert 0.05 in fractions  # Initial call
        assert 1.0 in fractions   # Final call


class TestLoadAtLevel:
    def test_none_level_calls_cpu_offload(self, test_config, mock_diffusers_pipeline):
        from pipeline import FluxKreaPipeline

        test_config.optimization_level = "none"
        mock_pipe, _ = mock_diffusers_pipeline
        p = FluxKreaPipeline(test_config)

        with patch.object(p, "_apply_cpu_offload") as mock_offload:
            p._load_at_level(OptimizationLevel.NONE)

        mock_offload.assert_called_once()

    def test_standard_level_calls_standard_optimizations(self, test_config, mock_diffusers_pipeline):
        from pipeline import FluxKreaPipeline

        test_config.optimization_level = "standard"
        mock_pipe, _ = mock_diffusers_pipeline
        p = FluxKreaPipeline(test_config)
        p.device = MagicMock()
        p.device.type = "cpu"

        with patch.object(p, "_apply_standard_optimizations") as mock_std:
            p._load_at_level(OptimizationLevel.STANDARD)

        mock_std.assert_called_once()

    def test_maximum_level_calls_both_standard_and_maximum(self, test_config, mock_diffusers_pipeline):
        from pipeline import FluxKreaPipeline

        mock_pipe, _ = mock_diffusers_pipeline
        p = FluxKreaPipeline(test_config)
        p.device = MagicMock()
        p.device.type = "cpu"

        with patch.object(p, "_apply_standard_optimizations") as mock_std:
            with patch.object(p, "_apply_maximum_optimizations") as mock_max:
                p._load_at_level(OptimizationLevel.MAXIMUM)

        mock_std.assert_called_once()
        mock_max.assert_called_once()

    def test_progress_callback_called_inside_load_at_level(self, test_config, mock_diffusers_pipeline):
        from pipeline import FluxKreaPipeline

        mock_pipe, _ = mock_diffusers_pipeline
        p = FluxKreaPipeline(test_config)
        p.device = MagicMock()
        p.device.type = "cpu"
        calls = []

        with patch.object(p, "_apply_standard_optimizations"):
            p._load_at_level(
                OptimizationLevel.STANDARD,
                progress_callback=lambda f, m: calls.append(f),
            )

        assert len(calls) >= 2  # At least 0.1 and 0.4


class TestApplyCpuOffload:
    def test_enable_model_cpu_offload_called(self, test_config, mock_diffusers_pipeline):
        from pipeline import FluxKreaPipeline

        mock_pipe, _ = mock_diffusers_pipeline
        test_config.optimization_level = "none"
        p = FluxKreaPipeline(test_config)
        p.pipeline = mock_pipe
        p._apply_cpu_offload()
        mock_pipe.enable_model_cpu_offload.assert_called_once()

    def test_falls_back_to_sequential_offload(self, test_config, mock_diffusers_pipeline):
        from pipeline import FluxKreaPipeline

        mock_pipe, _ = mock_diffusers_pipeline
        mock_pipe.enable_model_cpu_offload.side_effect = RuntimeError("not available")
        p = FluxKreaPipeline(test_config)
        p.pipeline = mock_pipe
        p._apply_cpu_offload()
        mock_pipe.enable_sequential_cpu_offload.assert_called_once()

    def test_silent_when_both_offload_methods_fail(self, test_config, mock_diffusers_pipeline):
        from pipeline import FluxKreaPipeline

        mock_pipe, _ = mock_diffusers_pipeline
        mock_pipe.enable_model_cpu_offload.side_effect = RuntimeError("no")
        mock_pipe.enable_sequential_cpu_offload.side_effect = RuntimeError("no")
        p = FluxKreaPipeline(test_config)
        p.pipeline = mock_pipe
        p._apply_cpu_offload()  # Must not raise


class TestApplyStandardOptimizations:
    def _make_pipeline(self, test_config, mock_diffusers_pipeline):
        from pipeline import FluxKreaPipeline

        mock_pipe, _ = mock_diffusers_pipeline
        mock_pipe.to.return_value = mock_pipe
        p = FluxKreaPipeline(test_config)
        p.device = MagicMock()
        p.device.type = "cpu"
        p.pipeline = mock_pipe
        return p, mock_pipe

    def test_moves_pipeline_to_device(self, test_config, mock_diffusers_pipeline):
        p, mock_pipe = self._make_pipeline(test_config, mock_diffusers_pipeline)
        p._apply_standard_optimizations()
        mock_pipe.to.assert_called_once_with(p.device)

    def test_enables_attention_slicing_auto_when_not_safety_mode(self, test_config, mock_diffusers_pipeline):
        p, mock_pipe = self._make_pipeline(test_config, mock_diffusers_pipeline)
        test_config.safety_mode.enabled = False
        p._apply_standard_optimizations()
        mock_pipe.enable_attention_slicing.assert_called_once_with("auto")

    def test_enables_conservative_attention_slicing_in_safety_mode(self, test_config, mock_diffusers_pipeline):
        p, mock_pipe = self._make_pipeline(test_config, mock_diffusers_pipeline)
        test_config.safety_mode.enabled = True
        p._apply_standard_optimizations()
        mock_pipe.enable_attention_slicing.assert_called_once_with(
            test_config.safety_mode.attention_slice_size
        )

    def test_enables_vae_tiling_when_not_safety_disabled(self, test_config, mock_diffusers_pipeline):
        p, mock_pipe = self._make_pipeline(test_config, mock_diffusers_pipeline)
        test_config.safety_mode.enabled = False
        p._apply_standard_optimizations()
        mock_pipe.enable_vae_tiling.assert_called()

    def test_skips_vae_tiling_in_safety_mode(self, test_config, mock_diffusers_pipeline):
        p, mock_pipe = self._make_pipeline(test_config, mock_diffusers_pipeline)
        test_config.safety_mode.enabled = True
        test_config.safety_mode.disable_vae_tiling = True
        p._apply_standard_optimizations()
        mock_pipe.enable_vae_tiling.assert_not_called()


class TestApplyMaximumOptimizations:
    def test_creates_metal_optimizer(self, test_config):
        from pipeline import FluxKreaPipeline

        p = FluxKreaPipeline(test_config)
        p.pipeline = MagicMock()
        p.pipeline.to.return_value = p.pipeline

        mock_metal_cls = MagicMock()
        mock_metal_instance = MagicMock()
        mock_metal_instance.optimize_pipeline.return_value = p.pipeline
        mock_metal_cls.return_value = mock_metal_instance

        with patch("optimizers.metal.MetalOptimizer", mock_metal_cls):
            p._apply_maximum_optimizations()

        mock_metal_cls.assert_called_once_with(test_config)

    def test_creates_thermal_manager_and_starts_monitoring(self, test_config):
        from pipeline import FluxKreaPipeline

        p = FluxKreaPipeline(test_config)
        p.pipeline = MagicMock()
        p.pipeline.to.return_value = p.pipeline

        mock_thermal_cls = MagicMock()
        mock_thermal_instance = MagicMock()
        mock_thermal_cls.return_value = mock_thermal_instance

        with patch("optimizers.thermal.ThermalManager", mock_thermal_cls):
            p._apply_maximum_optimizations()

        mock_thermal_instance.start_monitoring.assert_called_once()

    def test_silently_skips_metal_on_import_error(self, test_config):
        from pipeline import FluxKreaPipeline

        p = FluxKreaPipeline(test_config)
        p.pipeline = MagicMock()

        with patch.dict("sys.modules", {"optimizers.metal": None}):
            with patch.dict("sys.modules", {"optimizers.thermal": None}):
                p._apply_maximum_optimizations()  # Must not raise


class TestSaveImage:
    def test_saves_to_explicit_path(self, test_config):
        from pipeline import FluxKreaPipeline

        p = FluxKreaPipeline(test_config)
        mock_image = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "test.png")
            result = p.save_image(mock_image, "a cat", output_path=out_path)

        mock_image.save.assert_called_once()
        assert result == Path(out_path)

    def test_auto_names_file_from_prompt_and_timestamp(self, test_config):
        from pipeline import FluxKreaPipeline

        p = FluxKreaPipeline(test_config)
        mock_image = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            test_config.output.directory = tmpdir
            result = p.save_image(mock_image, "a beautiful sunset over the ocean")

        mock_image.save.assert_called_once()
        assert isinstance(result, Path)

    def test_creates_output_directory_if_missing(self, test_config):
        from pipeline import FluxKreaPipeline

        p = FluxKreaPipeline(test_config)
        mock_image = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_subdir"
            test_config.output.directory = str(new_dir)
            p.save_image(mock_image, "test prompt")
            # Assert while still inside TemporaryDirectory context
            assert new_dir.exists()


class TestGenerateWithThermal:
    def test_thermal_throttling_reduces_steps(self, test_config, mock_diffusers_pipeline):
        from pipeline import FluxKreaPipeline

        mock_pipe, _ = mock_diffusers_pipeline
        test_config.optimization_level = "none"
        p = FluxKreaPipeline(test_config)
        p.load()

        # Set up thermal manager with throttling profile
        mock_thermal = MagicMock()
        mock_profile = MagicMock()
        mock_profile.inference_steps_scale = 0.5  # Throttle to 50%
        mock_thermal.get_current_profile.return_value = mock_profile
        p._thermal_manager = mock_thermal

        image, metrics = p.generate(
            prompt="test", width=512, height=512, num_inference_steps=10
        )

        # Steps should be throttled (max(10*0.5, 4) = 5)
        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["num_inference_steps"] == 5

    def test_generate_without_thermal_manager_uses_original_steps(
        self, test_config, mock_diffusers_pipeline
    ):
        from pipeline import FluxKreaPipeline

        mock_pipe, _ = mock_diffusers_pipeline
        test_config.optimization_level = "none"
        p = FluxKreaPipeline(test_config)
        p.load()
        assert p._thermal_manager is None

        image, metrics = p.generate(
            prompt="test", width=512, height=512, num_inference_steps=8
        )

        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["num_inference_steps"] == 8

    def test_thermal_throttling_minimum_4_steps(self, test_config, mock_diffusers_pipeline):
        from pipeline import FluxKreaPipeline

        mock_pipe, _ = mock_diffusers_pipeline
        test_config.optimization_level = "none"
        p = FluxKreaPipeline(test_config)
        p.load()

        mock_thermal = MagicMock()
        mock_profile = MagicMock()
        mock_profile.inference_steps_scale = 0.1  # Extreme throttle
        mock_thermal.get_current_profile.return_value = mock_profile
        p._thermal_manager = mock_thermal

        _, metrics = p.generate(
            prompt="test", width=512, height=512, num_inference_steps=10
        )
        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["num_inference_steps"] >= 4


class TestUnloadWithThermal:
    def test_unload_stops_thermal_manager(self, test_config, mock_diffusers_pipeline):
        from pipeline import FluxKreaPipeline

        mock_pipe, _ = mock_diffusers_pipeline
        test_config.optimization_level = "none"
        p = FluxKreaPipeline(test_config)
        p.load()

        mock_thermal = MagicMock()
        p._thermal_manager = mock_thermal

        p.unload()

        mock_thermal.stop_monitoring.assert_called_once()
        assert p._thermal_manager is None

    def test_unload_handles_thermal_stop_exception(self, test_config, mock_diffusers_pipeline):
        from pipeline import FluxKreaPipeline

        mock_pipe, _ = mock_diffusers_pipeline
        test_config.optimization_level = "none"
        p = FluxKreaPipeline(test_config)
        p.load()

        mock_thermal = MagicMock()
        mock_thermal.stop_monitoring.side_effect = RuntimeError("stop failed")
        p._thermal_manager = mock_thermal

        p.unload()  # Must not raise
        assert p.is_loaded is False
