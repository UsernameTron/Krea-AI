"""Tests for optimizers/thermal.py."""

import pytest

from config import FluxConfig
from optimizers.thermal import (
    PerformanceProfile,
    ThermalManager,
    ThermalMetrics,
    ThermalState,
)


class TestThermalState:
    def test_enum_values(self):
        assert ThermalState.OPTIMAL.value == "optimal"
        assert ThermalState.WARM.value == "warm"
        assert ThermalState.HOT.value == "hot"
        assert ThermalState.CRITICAL.value == "critical"


class TestThermalManager:
    def test_init(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        assert tm is not None

    def test_classify_none_temperature(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        state = tm._classify_temperature(None)
        assert state == ThermalState.OPTIMAL

    def test_classify_optimal(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        state = tm._classify_temperature(50.0)
        assert state == ThermalState.OPTIMAL

    def test_classify_warm(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        state = tm._classify_temperature(75.0)
        assert state == ThermalState.WARM

    def test_classify_hot(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        state = tm._classify_temperature(85.0)
        assert state == ThermalState.HOT

    def test_classify_critical(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        state = tm._classify_temperature(95.0)
        assert state == ThermalState.CRITICAL

    def test_profile_for_optimal(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        profile = tm._profile_for_state(ThermalState.OPTIMAL)
        assert profile.inference_steps_scale == 1.0
        assert profile.max_gpu_utilization == 1.0

    def test_profile_for_critical(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        profile = tm._profile_for_state(ThermalState.CRITICAL)
        assert profile.inference_steps_scale < 1.0
        assert profile.max_cpu_threads == 4

    def test_get_status(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        status = tm.get_status()
        assert isinstance(status, dict)
        assert "thermal_state" in status
        assert "monitoring" in status
        assert status["monitoring"] is False

    def test_get_current_metrics(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        metrics = tm.get_current_metrics()
        assert isinstance(metrics, ThermalMetrics)

    def test_get_current_profile(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        profile = tm.get_current_profile()
        assert isinstance(profile, PerformanceProfile)

    def test_callback_registration(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        called = []
        tm.add_callback(lambda p: called.append(p))
        assert len(tm._callbacks) == 1


class TestPerformanceProfile:
    def test_defaults(self):
        p = PerformanceProfile()
        assert p.max_cpu_threads == 8
        assert p.max_gpu_utilization == 1.0
        assert p.memory_fraction == 0.9
        assert p.inference_steps_scale == 1.0


class TestThermalMetrics:
    def test_defaults(self):
        m = ThermalMetrics()
        assert m.cpu_temp is None
        assert m.gpu_temp is None
        assert m.thermal_state == ThermalState.OPTIMAL
        assert m.is_estimated is False
