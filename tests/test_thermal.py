"""Tests for optimizers/thermal.py."""

import subprocess
from unittest.mock import MagicMock, patch

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


class TestStartStopMonitoring:
    def test_start_monitoring_sets_flag(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        with patch.object(tm, "_monitor_loop"):
            tm.start_monitoring(interval=999)
        assert tm._monitoring is True

    def test_start_monitoring_idempotent(self):
        """Calling start_monitoring when already monitoring must not spawn a second thread."""
        config = FluxConfig()
        tm = ThermalManager(config)
        tm._monitoring = True  # Pre-set as if already started
        with patch("optimizers.thermal.threading.Thread") as mock_thread_cls:
            tm.start_monitoring()
        mock_thread_cls.assert_not_called()

    def test_stop_monitoring_clears_flag(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        tm._monitoring = True
        mock_thread = MagicMock()
        tm._thread = mock_thread
        tm.stop_monitoring()
        assert tm._monitoring is False
        assert tm._thread is None

    def test_stop_monitoring_joins_thread(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        tm._monitoring = True
        mock_thread = MagicMock()
        tm._thread = mock_thread
        tm.stop_monitoring()
        mock_thread.join.assert_called_once_with(timeout=3.0)

    def test_stop_monitoring_no_thread_does_not_raise(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        tm._monitoring = True
        tm._thread = None
        tm.stop_monitoring()  # Should complete without error


class TestMonitorLoop:
    def _run_one_iteration(self, tm, temp_value=55.0, callback=None):
        """Helper: run _monitor_loop for exactly one iteration then stop."""
        if callback:
            tm.add_callback(callback)

        call_count = [0]

        def read_temp_once():
            call_count[0] += 1
            tm._monitoring = False  # Stop after first read
            return temp_value

        with patch.object(tm, "_read_temperature", side_effect=read_temp_once):
            with patch("optimizers.thermal.time.sleep"):
                tm._monitoring = True
                tm._monitor_loop(interval=0)

        return call_count[0]

    def test_monitor_loop_updates_metrics(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        self._run_one_iteration(tm, temp_value=55.0)
        assert tm._current_metrics.cpu_temp == 55.0

    def test_monitor_loop_sets_is_estimated_false_when_temp_known(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        self._run_one_iteration(tm, temp_value=55.0)
        assert tm._current_metrics.is_estimated is False

    def test_monitor_loop_sets_is_estimated_true_when_temp_none(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        self._run_one_iteration(tm, temp_value=None)
        assert tm._current_metrics.is_estimated is True

    def test_monitor_loop_invokes_callback_on_profile_change(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        # Force a profile change by making current profile differ from critical
        tm._current_profile = PerformanceProfile(
            max_cpu_threads=999, inference_steps_scale=0.1
        )
        called = []
        self._run_one_iteration(tm, temp_value=95.0, callback=lambda p: called.append(p))
        assert len(called) == 1

    def test_monitor_loop_callback_exception_does_not_stop_loop(self):
        """Callback errors must be swallowed, not crash the monitor."""
        config = FluxConfig()
        tm = ThermalManager(config)
        tm._current_profile = PerformanceProfile(
            max_cpu_threads=999, inference_steps_scale=0.1
        )
        tm.add_callback(lambda p: (_ for _ in ()).throw(RuntimeError("cb error")))
        # _run_one_iteration should complete without raising
        self._run_one_iteration(tm, temp_value=95.0)

    def test_monitor_loop_handles_read_exception(self):
        """If _read_temperature raises, the loop should log and continue."""
        config = FluxConfig()
        tm = ThermalManager(config)
        call_count = [0]

        def raise_once():
            call_count[0] += 1
            tm._monitoring = False
            raise RuntimeError("read failed")

        with patch.object(tm, "_read_temperature", side_effect=raise_once):
            with patch("optimizers.thermal.time.sleep"):
                tm._monitoring = True
                tm._monitor_loop(interval=0)  # Must not propagate exception


class TestReadTemperature:
    def test_returns_sysctl_temp_if_available(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        with patch.object(tm, "_try_sysctl", return_value=65.0):
            with patch.object(tm, "_try_psutil", return_value=70.0):
                result = tm._read_temperature()
        # sysctl takes priority
        assert result == 65.0

    def test_falls_back_to_psutil_when_sysctl_returns_none(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        with patch.object(tm, "_try_sysctl", return_value=None):
            with patch.object(tm, "_try_psutil", return_value=70.0):
                result = tm._read_temperature()
        assert result == 70.0

    def test_falls_back_to_powermetrics_when_both_none(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        with patch.object(tm, "_try_sysctl", return_value=None):
            with patch.object(tm, "_try_psutil", return_value=None):
                with patch.object(tm, "_try_powermetrics", return_value=72.0):
                    result = tm._read_temperature()
        assert result == 72.0

    def test_returns_none_when_all_methods_fail(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        with patch.object(tm, "_try_sysctl", return_value=None):
            with patch.object(tm, "_try_psutil", return_value=None):
                with patch.object(tm, "_try_powermetrics", return_value=None):
                    result = tm._read_temperature()
        assert result is None


class TestTrySysctl:
    def _make_result(self, returncode=0, stdout="3\n"):
        r = MagicMock()
        r.returncode = returncode
        r.stdout = stdout
        return r

    def test_returns_temperature_on_success(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        with patch("subprocess.run", return_value=self._make_result(stdout="3\n")):
            result = tm._try_sysctl()
        # 35.0 + 3 * 5.0 = 50.0
        assert result == 50.0

    def test_returns_none_on_nonzero_returncode(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        with patch("subprocess.run", return_value=self._make_result(returncode=1, stdout="")):
            result = tm._try_sysctl()
        assert result is None

    def test_returns_none_on_empty_stdout(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        with patch("subprocess.run", return_value=self._make_result(stdout="")):
            result = tm._try_sysctl()
        assert result is None

    def test_returns_none_on_subprocess_error(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        with patch("subprocess.run", side_effect=subprocess.SubprocessError("timeout")):
            result = tm._try_sysctl()
        assert result is None

    def test_returns_none_on_value_error(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        with patch("subprocess.run", return_value=self._make_result(stdout="not_a_number\n")):
            result = tm._try_sysctl()
        assert result is None


class TestTryPsutil:
    def test_returns_temperature_when_sensors_available(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        mock_sensor = MagicMock()
        mock_sensor.current = 68.5
        mock_psutil = MagicMock()
        mock_psutil.sensors_temperatures.return_value = {"coretemp": [mock_sensor]}
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            result = tm._try_psutil()
        assert result == 68.5

    def test_returns_none_when_sensors_temperatures_missing(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        mock_psutil = MagicMock(spec=[])  # No sensors_temperatures attribute
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            result = tm._try_psutil()
        assert result is None

    def test_returns_none_when_known_keys_missing_from_sensors(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        mock_psutil = MagicMock()
        mock_psutil.sensors_temperatures.return_value = {"unknown_sensor": []}
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            result = tm._try_psutil()
        assert result is None

    def test_returns_none_on_import_error(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        with patch.dict("sys.modules", {"psutil": None}):
            result = tm._try_psutil()
        assert result is None


class TestTryPowermetrics:
    def _make_result(self, returncode=0, stdout=""):
        r = MagicMock()
        r.returncode = returncode
        r.stdout = stdout
        return r

    def test_returns_temperature_on_success(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        output = "CPU die temperature: 72.3 C\n"
        with patch("subprocess.run", return_value=self._make_result(stdout=output)):
            result = tm._try_powermetrics()
        assert result == 72.3

    def test_returns_none_on_nonzero_returncode(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        with patch("subprocess.run", return_value=self._make_result(returncode=1)):
            result = tm._try_powermetrics()
        assert result is None

    def test_returns_none_when_cpu_die_line_missing(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        with patch("subprocess.run", return_value=self._make_result(stdout="nothing useful\n")):
            result = tm._try_powermetrics()
        assert result is None

    def test_returns_none_on_subprocess_error(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        with patch("subprocess.run", side_effect=subprocess.SubprocessError("sudo fail")):
            result = tm._try_powermetrics()
        assert result is None


class TestProfileDiffers:
    def test_identical_profiles_return_false(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        a = PerformanceProfile(max_cpu_threads=8, inference_steps_scale=1.0)
        b = PerformanceProfile(max_cpu_threads=8, inference_steps_scale=1.0)
        assert tm._profile_differs(a, b) is False

    def test_different_cpu_threads_returns_true(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        a = PerformanceProfile(max_cpu_threads=8, inference_steps_scale=1.0)
        b = PerformanceProfile(max_cpu_threads=4, inference_steps_scale=1.0)
        assert tm._profile_differs(a, b) is True

    def test_different_inference_scale_returns_true(self):
        config = FluxConfig()
        tm = ThermalManager(config)
        a = PerformanceProfile(max_cpu_threads=8, inference_steps_scale=1.0)
        b = PerformanceProfile(max_cpu_threads=8, inference_steps_scale=0.8)
        assert tm._profile_differs(a, b) is True

    def test_tiny_inference_scale_difference_returns_false(self):
        """Differences smaller than 0.01 tolerance are treated as equal."""
        config = FluxConfig()
        tm = ThermalManager(config)
        a = PerformanceProfile(max_cpu_threads=8, inference_steps_scale=1.0)
        b = PerformanceProfile(max_cpu_threads=8, inference_steps_scale=1.005)
        assert tm._profile_differs(a, b) is False
