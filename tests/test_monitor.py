"""Tests for utils/monitor.py — get_system_info()."""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestGetSystemInfo:
    def test_returns_dict(self):
        from utils.monitor import get_system_info

        result = get_system_info()
        assert isinstance(result, dict)

    def test_always_contains_pytorch_version(self):
        from utils.monitor import get_system_info

        result = get_system_info()
        assert "pytorch_version" in result
        assert isinstance(result["pytorch_version"], str)

    def test_always_contains_mps_available(self):
        from utils.monitor import get_system_info

        result = get_system_info()
        assert "mps_available" in result
        assert isinstance(result["mps_available"], bool)

    def test_always_contains_cuda_available(self):
        from utils.monitor import get_system_info

        result = get_system_info()
        assert "cuda_available" in result
        assert isinstance(result["cuda_available"], bool)

    def test_mps_available_reflects_torch(self):
        from utils.monitor import get_system_info

        with patch("torch.backends.mps.is_available", return_value=True):
            result = get_system_info()
        assert result["mps_available"] is True

    def test_mps_unavailable_reflects_torch(self):
        from utils.monitor import get_system_info

        with patch("torch.backends.mps.is_available", return_value=False):
            result = get_system_info()
        assert result["mps_available"] is False

    def test_psutil_memory_included_when_available(self):
        from utils.monitor import get_system_info

        mock_mem = MagicMock()
        mock_mem.total = 32 * 1024 ** 3  # 32 GB
        mock_mem.percent = 60.0

        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = mock_mem
        mock_psutil.cpu_count.return_value = 8

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            result = get_system_info()

        assert result["system_memory_gb"] == 32
        assert result["memory_used_percent"] == 60.0

    def test_psutil_cpu_count_included_when_available(self):
        from utils.monitor import get_system_info

        mock_mem = MagicMock()
        mock_mem.total = 16 * 1024 ** 3
        mock_mem.percent = 30.0

        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = mock_mem
        mock_psutil.cpu_count.side_effect = lambda logical: 8 if logical else 4

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            result = get_system_info()

        assert result["cpu_count"] == 8
        assert result["cpu_count_physical"] == 4

    def test_psutil_not_present_does_not_raise(self):
        from utils.monitor import get_system_info

        with patch.dict(sys.modules, {"psutil": None}):
            result = get_system_info()
        # Should succeed without psutil keys
        assert isinstance(result, dict)
        assert "pytorch_version" in result

    def test_mps_allocated_included_when_mps_available(self):
        from utils.monitor import get_system_info

        allocated_bytes = 256 * 1024 ** 2  # 256 MB
        with patch("torch.backends.mps.is_available", return_value=True):
            with patch("torch.mps.current_allocated_memory", return_value=allocated_bytes):
                result = get_system_info()

        assert result["mps_allocated_mb"] == 256

    def test_mps_allocated_not_included_when_mps_unavailable(self):
        from utils.monitor import get_system_info

        with patch("torch.backends.mps.is_available", return_value=False):
            result = get_system_info()

        assert "mps_allocated_mb" not in result

    def test_mps_allocated_error_does_not_raise(self):
        from utils.monitor import get_system_info

        with patch("torch.backends.mps.is_available", return_value=True):
            with patch("torch.mps.current_allocated_memory", side_effect=RuntimeError("mps error")):
                result = get_system_info()

        # Should not raise, mps_allocated_mb simply not set
        assert isinstance(result, dict)
        assert "mps_available" in result
