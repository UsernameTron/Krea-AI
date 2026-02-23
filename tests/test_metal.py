"""Tests for optimizers/metal.py."""

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


class TestSafeEmptyCache:
    def test_safe_empty_cache_does_not_raise(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt._safe_empty_cache()  # Should not raise

    def test_cleanup_does_not_raise(self):
        config = FluxConfig()
        opt = MetalOptimizer(config)
        opt.cleanup()  # Should not raise
