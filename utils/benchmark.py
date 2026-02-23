"""
Benchmark runner for FLUX.1 Krea pipeline.

Measures generation performance at different settings.
"""

import logging
import time
from typing import Dict, List, Optional

from config import FluxConfig, get_config
from pipeline import FluxKreaPipeline

logger = logging.getLogger(__name__)


def run_benchmark(
    config: Optional[FluxConfig] = None,
    steps_list: Optional[List[int]] = None,
    quick: bool = False,
) -> Dict:
    """Run generation benchmark at different step counts.

    Args:
        config: FluxConfig to use. Defaults to get_config().
        steps_list: List of step counts to test. Defaults to [4, 10, 20].
        quick: If True, use smaller image size and fewer steps.

    Returns:
        Dict of benchmark results keyed by step count.
    """
    config = config or get_config()

    if steps_list is None:
        steps_list = [4, 10] if quick else [4, 10, 20]

    width = 512 if quick else config.generation.width
    height = 512 if quick else config.generation.height

    pipeline = FluxKreaPipeline(config)
    pipeline.load()

    prompt = "a cute cat sitting in a garden, photorealistic"
    results = {}

    for steps in steps_list:
        logger.info("Benchmarking %d steps at %dx%d...", steps, width, height)
        try:
            image, metrics = pipeline.generate(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                seed=42,
            )
            results[steps] = {
                "total_time": metrics["generation_time"],
                "time_per_step": metrics["time_per_step"],
                "pixels_per_second": (width * height) / metrics["generation_time"],
            }
            logger.info(
                "%d steps: %.1fs total, %.2fs/step",
                steps, metrics["generation_time"], metrics["time_per_step"],
            )
        except Exception as e:
            results[steps] = {"error": str(e)}
            logger.error("%d steps failed: %s", steps, e)

    pipeline.unload()
    return results
