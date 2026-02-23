"""
Unified FLUX.1 Krea Pipeline.
Replaces: flux_krea_official.py, flux_krea_m4_optimized.py,
          maximum_performance_pipeline.py, flux_interactive.py, and 5 other variants.
"""

import gc
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.backends.mps

from config import (
    FluxConfig,
    OptimizationLevel,
    apply_mps_environment,
    get_config,
    get_hf_token,
)

logger = logging.getLogger(__name__)


class FluxKreaPipeline:
    """Unified FLUX.1 Krea pipeline with configurable optimization levels.

    Optimization levels:
        NONE:     Baseline diffusers FluxPipeline, no extras.
        STANDARD: MPS device + attention slicing + VAE tiling + CPU offload.
        MAXIMUM:  All of STANDARD + Metal kernel optimizations + thermal management.

    Fallback chain: MAXIMUM -> STANDARD -> NONE (with warnings).
    """

    def __init__(self, config: Optional[FluxConfig] = None):
        self.config = config or get_config()
        self.pipeline = None
        self.is_loaded = False
        self.device = None
        self.dtype = self._get_dtype()
        self._metal_optimizer = None
        self._thermal_manager = None
        self._load_time = 0.0

    def _get_dtype(self) -> torch.dtype:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.config.model.dtype, torch.bfloat16)

    def _detect_device(self) -> torch.device:
        """Determine the best available device."""
        preferred = self.config.device.preferred

        if preferred == "mps" or preferred == "auto":
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device("mps")
        if preferred == "cuda" or preferred == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")

        if preferred not in ("cpu",) and preferred != "auto":
            logger.warning("Preferred device '%s' not available, falling back to CPU", preferred)

        return torch.device("cpu")

    def _configure_environment(self):
        """Set MPS environment variables and thread counts from config."""
        apply_mps_environment(self.config)

        # Thread configuration
        torch.set_num_threads(self.config.device.cpu_threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(self.config.device.cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(self.config.device.cpu_threads))

        # Disable unnecessary warnings
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")

    def load(self, progress_callback: Optional[Callable[[float, str], None]] = None):
        """Load the pipeline with optimizations based on configured level.

        Args:
            progress_callback: Optional (progress_fraction, message) callback for UI.
        """
        self._configure_environment()
        self.device = self._detect_device()
        level = self.config.get_optimization_level()

        logger.info("Loading FLUX.1 Krea pipeline: device=%s, level=%s", self.device, level.value)
        if progress_callback:
            progress_callback(0.05, "Initializing pipeline...")

        start_time = time.time()

        try:
            self._load_at_level(level, progress_callback)
        except Exception as e:
            if level == OptimizationLevel.MAXIMUM:
                logger.warning("MAXIMUM optimization failed (%s), falling back to STANDARD", e)
                try:
                    self._load_at_level(OptimizationLevel.STANDARD, progress_callback)
                except Exception as e2:
                    logger.warning("STANDARD optimization failed (%s), falling back to NONE", e2)
                    self._load_at_level(OptimizationLevel.NONE, progress_callback)
            elif level == OptimizationLevel.STANDARD:
                logger.warning("STANDARD optimization failed (%s), falling back to NONE", e)
                self._load_at_level(OptimizationLevel.NONE, progress_callback)
            else:
                raise

        self._load_time = time.time() - start_time
        self.is_loaded = True
        logger.info("Pipeline loaded in %.1fs", self._load_time)

        if progress_callback:
            progress_callback(1.0, f"Pipeline loaded in {self._load_time:.1f}s")

    def _load_at_level(self, level: OptimizationLevel,
                       progress_callback: Optional[Callable] = None):
        """Load pipeline at a specific optimization level."""
        from diffusers import FluxPipeline

        if progress_callback:
            progress_callback(0.1, f"Loading model ({level.value} optimization)...")

        self.pipeline = FluxPipeline.from_pretrained(
            self.config.model.id,
            torch_dtype=self.dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )

        if level == OptimizationLevel.NONE:
            # Baseline: just CPU offload for memory safety
            self._apply_cpu_offload()
            return

        if progress_callback:
            progress_callback(0.4, "Moving to device...")

        if level == OptimizationLevel.STANDARD:
            self._apply_standard_optimizations()

        elif level == OptimizationLevel.MAXIMUM:
            self._apply_standard_optimizations()
            self._apply_maximum_optimizations()

        if progress_callback:
            progress_callback(0.8, "Finalizing optimizations...")

    def _apply_cpu_offload(self):
        """Apply CPU offload for memory management."""
        try:
            self.pipeline.enable_model_cpu_offload()
            logger.info("Model CPU offload enabled")
        except Exception:
            try:
                self.pipeline.enable_sequential_cpu_offload()
                logger.info("Sequential CPU offload enabled (fallback)")
            except Exception as e:
                logger.warning("CPU offload not available: %s", e)

    def _apply_standard_optimizations(self):
        """STANDARD level: device placement + attention + VAE + offload."""
        # Move to device first
        self.pipeline = self.pipeline.to(self.device)

        # Attention slicing
        safety = self.config.safety_mode
        if safety.enabled:
            # Conservative slicing to prevent black images
            self.pipeline.enable_attention_slicing(safety.attention_slice_size)
            logger.info("Conservative attention slicing enabled (size=%d)", safety.attention_slice_size)
        else:
            self.pipeline.enable_attention_slicing("auto")
            logger.info("Attention slicing enabled (auto)")

        # Attention processor
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0
            if hasattr(self.pipeline, "transformer"):
                self.pipeline.transformer.set_attn_processor(AttnProcessor2_0())
                logger.info("AttnProcessor2_0 set on transformer")
        except Exception as e:
            logger.debug("Could not set attention processor: %s", e)

        # VAE optimizations (skip if safety mode disables them)
        if not (safety.enabled and safety.disable_vae_tiling):
            if hasattr(self.pipeline, "enable_vae_tiling"):
                self.pipeline.enable_vae_tiling()
                logger.info("VAE tiling enabled")
            if hasattr(self.pipeline, "enable_vae_slicing"):
                self.pipeline.enable_vae_slicing()
                logger.info("VAE slicing enabled")
        else:
            logger.info("VAE tiling/slicing disabled (safety mode)")

        # MPS-specific settings
        if self.device.type == "mps":
            try:
                torch.mps.set_per_process_memory_fraction(self.config.device.mps_memory_fraction)
            except Exception:
                pass

    def _apply_maximum_optimizations(self):
        """MAXIMUM level: Metal kernels + thermal management on top of STANDARD."""
        # Metal kernel optimizations
        try:
            from optimizers.metal import MetalOptimizer
            self._metal_optimizer = MetalOptimizer(self.config)
            self.pipeline = self._metal_optimizer.optimize_pipeline(self.pipeline)
            logger.info("Metal kernel optimizations applied")
        except ImportError:
            logger.info("Metal optimizer not available, skipping")
        except Exception as e:
            logger.warning("Metal optimization failed: %s", e)

        # Thermal management
        try:
            from optimizers.thermal import ThermalManager
            self._thermal_manager = ThermalManager(self.config)
            self._thermal_manager.start_monitoring()
            logger.info("Thermal management started")
        except ImportError:
            logger.info("Thermal manager not available, skipping")
        except Exception as e:
            logger.warning("Thermal management failed to start: %s", e)

    @contextmanager
    def _inference_context(self):
        """Context manager for inference with proper cleanup."""
        try:
            with torch.inference_mode():
                yield
        finally:
            self._cleanup_memory()

    def _cleanup_memory(self):
        """Clean up device memory safely."""
        if self.device and self.device.type == "mps":
            try:
                torch.mps.empty_cache()
            except RuntimeError as e:
                # Catch watermark ratio errors that plague MPS
                if "watermark" in str(e).lower():
                    logger.debug("MPS cache clear skipped (watermark ratio): %s", e)
                else:
                    logger.warning("MPS cache clear failed: %s", e)
        elif self.device and self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def generate(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate.
            width: Image width (multiple of 64). Uses config default if None.
            height: Image height (multiple of 64). Uses config default if None.
            guidance_scale: Classifier-free guidance scale. Uses config default if None.
            num_inference_steps: Number of denoising steps. Uses config default if None.
            max_sequence_length: Max prompt token length. Uses config default if None.
            seed: Random seed for reproducibility.
            progress_callback: Optional (fraction, message) callback.

        Returns:
            PIL.Image.Image: The generated image.
        """
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        # Apply config defaults (with safety mode overrides)
        params = self.config.get_effective_generation_params()
        width = width or params["width"]
        height = height or params["height"]
        guidance_scale = guidance_scale if guidance_scale is not None else params["guidance_scale"]
        num_inference_steps = num_inference_steps or params["steps"]
        max_sequence_length = max_sequence_length or params["max_sequence_length"]

        # Apply thermal throttling if active
        if self._thermal_manager:
            try:
                profile = self._thermal_manager.get_current_profile()
                if profile and profile.inference_steps_scale < 1.0:
                    adjusted = int(num_inference_steps * profile.inference_steps_scale)
                    logger.info("Thermal throttling: steps %d -> %d", num_inference_steps, adjusted)
                    num_inference_steps = max(adjusted, 4)
            except Exception:
                pass

        logger.info(
            "Generating: %dx%d, %d steps, guidance=%.1f, seq_len=%d, seed=%s",
            width, height, num_inference_steps, guidance_scale, max_sequence_length,
            seed if seed is not None else "random",
        )

        if progress_callback:
            progress_callback(0.05, "Preparing generation...")

        # Clear cache before generation
        self._cleanup_memory()

        start_time = time.time()

        with self._inference_context():
            # Generator for reproducibility — always create on CPU for MPS compatibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device="cpu").manual_seed(seed)

            result = self.pipeline(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
                generator=generator,
                return_dict=True,
            )

        generation_time = time.time() - start_time
        logger.info("Generation complete in %.1fs (%.2fs/step)", generation_time, generation_time / num_inference_steps)

        if progress_callback:
            progress_callback(1.0, f"Done in {generation_time:.1f}s")

        image = result.images[0]
        return image, {
            "generation_time": generation_time,
            "time_per_step": generation_time / num_inference_steps,
            "width": width,
            "height": height,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "device": str(self.device),
        }

    def save_image(self, image, prompt: str, output_path: Optional[str] = None) -> Path:
        """Save a generated image with auto-naming if no path specified."""
        if output_path:
            path = Path(output_path)
        else:
            output_dir = Path(self.config.output.directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename from pattern
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            slug = "_".join(
                "".join(c if c.isalnum() or c.isspace() else "" for c in prompt).split()[:5]
            )
            filename = (
                self.config.output.filename_pattern
                .replace("{timestamp}", timestamp)
                .replace("{prompt_slug}", slug or "image")
            )
            path = output_dir / filename

        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        logger.info("Saved: %s", path)
        return path

    def unload(self):
        """Unload pipeline and free memory."""
        if self._thermal_manager:
            try:
                self._thermal_manager.stop_monitoring()
            except Exception:
                pass
            self._thermal_manager = None

        self._metal_optimizer = None

        if self.pipeline:
            del self.pipeline
            self.pipeline = None

        self.is_loaded = False
        self._cleanup_memory()
        logger.info("Pipeline unloaded")

    def get_system_info(self) -> dict:
        """Return system information for display."""
        info = {
            "pytorch_version": torch.__version__,
            "device": str(self.device) if self.device else "not initialized",
            "dtype": str(self.dtype),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "hf_token_set": get_hf_token() is not None,
            "model_id": self.config.model.id,
            "optimization_level": self.config.optimization_level,
            "is_loaded": self.is_loaded,
            "load_time": self._load_time,
        }

        try:
            import psutil
            mem = psutil.virtual_memory()
            info["system_memory_gb"] = mem.total // (1024 ** 3)
            info["memory_used_percent"] = mem.percent
        except ImportError:
            pass

        if self.device and self.device.type == "mps" and self.is_loaded:
            try:
                info["mps_allocated_mb"] = torch.mps.current_allocated_memory() // (1024 ** 2)
            except Exception:
                pass

        return info
