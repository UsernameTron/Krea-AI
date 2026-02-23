"""
Performance monitor for FLUX.1 Krea.

Provides system info and resource usage tracking.
"""

import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)


def get_system_info() -> Dict:
    """Get system information relevant to FLUX.1 Krea."""
    info = {
        "pytorch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }

    try:
        import psutil
        mem = psutil.virtual_memory()
        info["system_memory_gb"] = mem.total // (1024 ** 3)
        info["memory_used_percent"] = mem.percent
        info["cpu_count"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    except ImportError:
        pass

    if torch.backends.mps.is_available():
        try:
            info["mps_allocated_mb"] = torch.mps.current_allocated_memory() // (1024 ** 2)
        except Exception:
            pass

    return info
