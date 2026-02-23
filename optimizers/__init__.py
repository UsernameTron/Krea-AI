"""
Optimization modules for FLUX.1 Krea pipeline.

Provides Metal kernel optimization, Neural Engine acceleration,
and thermal performance management for Apple Silicon.
"""

from config import FluxConfig, OptimizationLevel


def get_optimizer(config: FluxConfig):
    """Factory function to get the appropriate optimizer set for the config.

    Returns a dict of available optimizers keyed by name.
    """
    optimizers = {}
    level = config.get_optimization_level()

    if level == OptimizationLevel.MAXIMUM:
        try:
            from optimizers.metal import MetalOptimizer
            optimizers["metal"] = MetalOptimizer(config)
        except ImportError:
            pass

        try:
            from optimizers.neural_engine import NeuralEngineOptimizer
            optimizers["neural_engine"] = NeuralEngineOptimizer(config)
        except ImportError:
            pass

        try:
            from optimizers.thermal import ThermalManager
            optimizers["thermal"] = ThermalManager(config)
        except ImportError:
            pass

    return optimizers
