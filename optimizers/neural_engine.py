"""
Neural Engine acceleration for FLUX.1 Krea on Apple Silicon.

Status: Experimental / mostly stubbed out.

The original flux_neural_engine_accelerator.py compiled models with CoreML
but never actually applied them (comments said "don't replace in-place to
avoid unintended regressions"). This module preserves the class structure
with honest documentation about what works and what doesn't.

What works:
- CoreML availability detection (lazy, on first use)
- Basic model compilation API

What doesn't work yet:
- Compiled models are not applied to the pipeline
- Text encoder optimization returns the original unchanged
- No actual performance measurements (previously hardcoded 2.5x)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from config import FluxConfig

logger = logging.getLogger(__name__)


class NeuralEngineOptimizer:
    """Neural Engine optimizer for FLUX.1 Krea pipeline.

    Currently experimental — CoreML compilation works but compiled
    models are not applied to the inference pipeline.
    """

    def __init__(self, config: FluxConfig, cache_dir: str = "./neural_engine_cache"):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._coreml_available: Optional[bool] = None  # Lazy check
        self._compiled_count = 0

    @property
    def is_available(self) -> bool:
        """Check CoreML availability lazily on first access."""
        if self._coreml_available is None:
            self._coreml_available = self._check_coreml()
        return self._coreml_available

    def _check_coreml(self) -> bool:
        """Check if coremltools is installed and functional."""
        try:
            import coremltools  # noqa: F401
            logger.info("CoreML tools available for Neural Engine")
            return True
        except ImportError:
            logger.info("CoreML tools not installed — Neural Engine optimization disabled")
            return False

    def optimize_pipeline(self, pipeline) -> Any:
        """Apply Neural Engine optimizations to pipeline.

        Currently a no-op: CoreML compilation is available but compiled
        models are not applied to avoid stability regressions.
        """
        if not self.is_available:
            return pipeline

        # TODO: Apply compiled CoreML models once stability is verified.
        # The compilation works, but replacing attention/MLP forward methods
        # with CoreML inference caused regressions in the original codebase.
        logger.info("Neural Engine optimizer initialized (compilation available, not applied)")
        return pipeline

    def get_stats(self) -> Dict[str, Any]:
        """Get Neural Engine status."""
        return {
            "neural_engine_available": self.is_available,
            "compiled_models": self._compiled_count,
            "cache_dir": str(self.cache_dir),
            "acceleration_ratio": None,  # No measured data available
            "status": "experimental — compilation works but not applied to pipeline",
        }
