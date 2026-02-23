"""
Metal Performance Shaders optimization for FLUX.1 Krea on Apple Silicon.

Provides:
- MPS environment configuration with correct watermark ratios
- Metal kernel context manager with safe cache management
- Pipeline component optimization (device placement + dtype)
- VAE optimization with float16 autocast
- Memory statistics

No-op monkey patches from the original flux_metal_kernels.py have been removed.
The attention/MLP forward wrappers were not actually optimizing anything.
"""

import gc
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch
import torch.backends.mps

from config import FluxConfig

logger = logging.getLogger(__name__)


class MetalOptimizer:
    """Metal Performance Shaders optimizer for FLUX.1 Krea pipeline."""

    def __init__(self, config: FluxConfig):
        self.config = config
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.metal_available = torch.backends.mps.is_available()
        self._configure_environment()

    def _configure_environment(self):
        """Configure MPS environment with correct watermark ratios."""
        if not self.metal_available:
            logger.warning("Metal Performance Shaders not available")
            return

        # Use config values — ensures LOW < HIGH (validated by config)
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(self.config.device.mps_watermark_ratio))
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(self.config.device.mps_low_watermark_ratio))
        os.environ.setdefault("PYTORCH_MPS_PREFER_FAST_ALLOC", "1" if self.config.device.mps_prefer_fast_alloc else "0")
        os.environ.setdefault("PYTORCH_MPS_ALLOCATOR_POLICY", self.config.device.mps_allocator_policy)
        os.environ.setdefault("PYTORCH_MPS_MEMORY_FRACTION", str(self.config.device.mps_memory_fraction))

        try:
            if hasattr(torch.mps, "set_per_process_memory_fraction"):
                torch.mps.set_per_process_memory_fraction(self.config.device.mps_memory_fraction)
            logger.info("Metal environment configured")
        except Exception as e:
            logger.warning("Metal configuration warning: %s", e)

    @contextmanager
    def metal_kernel_context(self):
        """Context manager for Metal operations with safe cache management."""
        if not self.metal_available:
            yield False
            return

        try:
            self._safe_empty_cache()
            yield True
        except Exception as e:
            logger.error("Metal kernel context error: %s", e)
            yield False
        finally:
            self._safe_empty_cache()

    def _safe_empty_cache(self):
        """Empty MPS cache, catching both high and low watermark ratio errors."""
        try:
            torch.mps.empty_cache()
        except RuntimeError as e:
            msg = str(e).lower()
            if "watermark" in msg:
                logger.debug("MPS cache clear skipped (watermark ratio): %s", e)
            else:
                logger.warning("MPS cache clear failed: %s", e)

    def optimize_pipeline(self, pipeline) -> Any:
        """Optimize FLUX pipeline components for Metal execution.

        Applies device placement and VAE float16 autocast.
        Does NOT monkey-patch attention/MLP forward methods.
        """
        if not self.metal_available:
            logger.info("Metal not available, returning pipeline unchanged")
            return pipeline

        optimizations = []

        try:
            # Ensure pipeline is on MPS device
            pipeline = pipeline.to(self.device)
            optimizations.append("device_transfer")

            # VAE optimization with autocast
            if hasattr(pipeline, "vae"):
                self._optimize_vae(pipeline.vae)
                optimizations.append("vae_autocast")

            logger.info("Metal optimizations applied: %s", optimizations)

        except Exception as e:
            logger.error("Metal pipeline optimization failed: %s", e)

        return pipeline

    def _optimize_vae(self, vae):
        """Apply Metal autocast to VAE forward pass for better performance."""
        if not hasattr(vae, "forward"):
            return

        original_forward = vae.forward

        def metal_vae_forward(sample, *args, **kwargs):
            with self.metal_kernel_context() as available:
                if not available:
                    return original_forward(sample, *args, **kwargs)
                try:
                    sample = sample.to(self.device, non_blocking=True)
                    with torch.autocast(device_type="mps", dtype=torch.float16):
                        return original_forward(sample, *args, **kwargs)
                except Exception as e:
                    logger.warning("Metal VAE optimization failed, using fallback: %s", e)
                    return original_forward(sample, *args, **kwargs)

        vae.forward = metal_vae_forward

    # TODO: Implement real Metal-accelerated attention when PyTorch MPS
    # supports custom kernels. The original monkey-patches in flux_metal_kernels.py
    # were no-ops — they wrapped the original forward without changes.

    # TODO: Implement real Metal-accelerated MLP when PyTorch MPS
    # supports custom kernels. Same issue as attention above.

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get Metal memory statistics."""
        if not self.metal_available:
            return {"metal_available": False}

        try:
            allocated = 0
            if hasattr(torch.mps, "current_allocated_memory"):
                allocated = torch.mps.current_allocated_memory()

            return {
                "metal_available": True,
                "allocated_memory_mb": allocated / (1024 * 1024),
                "device": str(self.device),
            }
        except Exception as e:
            return {"metal_available": True, "error": str(e)}

    def cleanup(self):
        """Clean up Metal resources."""
        self._safe_empty_cache()
        gc.collect()
