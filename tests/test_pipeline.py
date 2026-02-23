"""Tests for pipeline.py."""

import pytest

from config import FluxConfig, get_config


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
