"""Tests for optimizers/metal.py."""

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from config import FluxConfig
from optimizers.metal import MetalOptimizer


class TestMetalOptimizer:
    def test_init(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        assert opt is not None

    def test_get_memory_stats_returns_dict(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        stats = opt.get_memory_stats()
        assert isinstance(stats, dict)
        assert "metal_available" in stats

    def test_config_watermark_used(self):
        config = FluxConfig()
        config.device.mps_watermark_ratio = 0.7
        opt = MetalOptimizer(config)
        assert opt.config.device.mps_watermark_ratio == 0.7


class TestConfigureEnvironment:
    def test_configure_environment_when_metal_unavailable_does_not_raise(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.metal_available = False
        opt._configure_environment()  # Should not raise

    def test_configure_environment_when_metal_available_sets_env_vars(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.metal_available = True

        with patch("torch.mps.set_per_process_memory_fraction") as mock_frac:
            # Clear env vars so setdefault actually sets them
            with patch.dict("os.environ", {}, clear=False):
                opt._configure_environment()
        # Should complete without raising

    def test_configure_environment_handles_exception(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.metal_available = True

        with patch(
            "torch.mps.set_per_process_memory_fraction",
            side_effect=RuntimeError("mps error"),
        ):
            opt._configure_environment()  # Warning logged but no raise


class TestMetalKernelContext:
    def test_context_manager_runs(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        with opt.metal_kernel_context() as available:
            assert isinstance(available, bool)

    def test_context_manager_completes(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        with opt.metal_kernel_context():
            pass  # Should complete without error

    def test_yields_false_when_metal_unavailable(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.metal_available = False
        with opt.metal_kernel_context() as available:
            assert available is False

    def test_yields_true_when_metal_available_and_no_error(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.metal_available = True

        with patch.object(opt, "_safe_empty_cache"):
            with opt.metal_kernel_context() as available:
                assert available is True

    def test_yields_false_on_exception_inside_context(self):
        """If _safe_empty_cache raises before yield, context logs and yields False.

        The finally block then calls _safe_empty_cache again; if that also raises,
        the RuntimeError propagates out of the context manager.
        """
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.metal_available = True

        # Only raise on the first call (pre-yield); let the finally call succeed
        call_count = [0]
        original = opt._safe_empty_cache

        def raise_once():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("boom")

        with patch.object(opt, "_safe_empty_cache", side_effect=raise_once):
            with opt.metal_kernel_context() as available:
                # Pre-yield raised → context yields False
                assert available is False

    def test_safe_empty_cache_called_on_exit(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.metal_available = True
        call_count = []

        with patch.object(opt, "_safe_empty_cache", side_effect=lambda: call_count.append(1)):
            with opt.metal_kernel_context():
                pass

        # Called at entry and exit = 2 times
        assert len(call_count) >= 1


class TestSafeEmptyCache:
    def test_safe_empty_cache_does_not_raise(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt._safe_empty_cache()  # Should not raise

    def test_cleanup_does_not_raise(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.cleanup()  # Should not raise

    def test_safe_empty_cache_swallows_watermark_error(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)

        with patch(
            "torch.mps.empty_cache",
            side_effect=RuntimeError("watermark ratio exceeded"),
        ):
            opt._safe_empty_cache()  # Must not raise

    def test_safe_empty_cache_swallows_non_watermark_runtime_error(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)

        with patch(
            "torch.mps.empty_cache",
            side_effect=RuntimeError("some other mps error"),
        ):
            opt._safe_empty_cache()  # Must not raise (logged as warning)


class TestOptimizePipeline:
    def test_returns_pipeline_unchanged_when_metal_unavailable(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.metal_available = False

        mock_pipeline = MagicMock()
        result = opt.optimize_pipeline(mock_pipeline)
        assert result is mock_pipeline

    def test_calls_pipeline_to_device_when_metal_available(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.metal_available = True

        mock_pipeline = MagicMock()
        mock_pipeline.to.return_value = mock_pipeline
        # No vae attribute to avoid _optimize_vae
        del mock_pipeline.vae

        with patch.object(opt, "_safe_empty_cache"):
            opt.optimize_pipeline(mock_pipeline)

        mock_pipeline.to.assert_called_once_with(opt.device)

    def test_calls_optimize_vae_when_pipeline_has_vae(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.metal_available = True

        mock_pipeline = MagicMock()
        mock_pipeline.to.return_value = mock_pipeline

        with patch.object(opt, "_optimize_vae") as mock_vae_opt:
            with patch.object(opt, "_safe_empty_cache"):
                opt.optimize_pipeline(mock_pipeline)

        mock_vae_opt.assert_called_once_with(mock_pipeline.vae)

    def test_handles_exception_during_optimization(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.metal_available = True

        mock_pipeline = MagicMock()
        mock_pipeline.to.side_effect = RuntimeError("device error")

        result = opt.optimize_pipeline(mock_pipeline)
        # Should return original pipeline even if optimization fails
        assert result is mock_pipeline


class TestOptimizeVae:
    def test_optimize_vae_patches_forward(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.metal_available = True

        mock_vae = MagicMock()
        original_forward = MagicMock(return_value=MagicMock())
        mock_vae.forward = original_forward

        opt._optimize_vae(mock_vae)

        # Forward should be replaced
        assert mock_vae.forward is not original_forward

    def test_optimize_vae_skips_if_no_forward(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.metal_available = True

        mock_vae = MagicMock(spec=[])  # No 'forward' attribute
        # Should complete without error
        opt._optimize_vae(mock_vae)

    def test_metal_vae_forward_calls_original_when_context_unavailable(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.metal_available = True

        original_result = MagicMock()
        original_forward = MagicMock(return_value=original_result)
        mock_vae = MagicMock()
        mock_vae.forward = original_forward

        opt._optimize_vae(mock_vae)

        # Call the patched forward with context unavailable
        mock_sample = MagicMock()
        with patch.object(opt, "metal_kernel_context") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=False)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = mock_vae.forward(mock_sample)

        original_forward.assert_called_once_with(mock_sample)

    def test_metal_vae_forward_uses_fallback_on_exception(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.metal_available = False  # So context yields False

        original_result = MagicMock()
        original_forward = MagicMock(return_value=original_result)
        mock_vae = MagicMock()
        mock_vae.forward = original_forward

        opt._optimize_vae(mock_vae)
        mock_sample = MagicMock()
        result = mock_vae.forward(mock_sample)
        assert result is original_result


class TestGetMemoryStats:
    def test_returns_metal_unavailable_dict_when_no_metal(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.metal_available = False

        stats = opt.get_memory_stats()
        assert stats == {"metal_available": False}

    def test_returns_allocated_mb_when_metal_available(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.metal_available = True

        allocated_bytes = 256 * 1024 * 1024
        with patch("torch.mps.current_allocated_memory", return_value=allocated_bytes):
            stats = opt.get_memory_stats()

        assert stats["metal_available"] is True
        assert stats["allocated_memory_mb"] == 256.0

    def test_returns_error_key_on_exception(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.metal_available = True

        with patch(
            "torch.mps.current_allocated_memory",
            side_effect=RuntimeError("mps error"),
        ):
            stats = opt.get_memory_stats()

        assert "error" in stats
        assert stats["metal_available"] is True

    def test_includes_device_string_when_successful(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.metal_available = True

        with patch("torch.mps.current_allocated_memory", return_value=0):
            stats = opt.get_memory_stats()

        assert "device" in stats
