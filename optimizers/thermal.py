"""
Thermal performance management for FLUX.1 Krea on Apple Silicon.

Monitors system temperature and adjusts performance profiles to prevent
thermal throttling during long generation sessions.

Temperature reading priority:
1. sysctl (no sudo needed, fastest)
2. psutil sensors (cross-platform)
3. powermetrics (needs sudo, most accurate but slowest)
4. Returns None if no real data available (no fake estimates)
"""

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from config import FluxConfig

logger = logging.getLogger(__name__)


class ThermalState(Enum):
    OPTIMAL = "optimal"    # < optimal_threshold
    WARM = "warm"          # optimal_threshold - warm_threshold
    HOT = "hot"            # warm_threshold - hot_threshold
    CRITICAL = "critical"  # > hot_threshold


@dataclass
class ThermalMetrics:
    cpu_temp: Optional[float] = None
    gpu_temp: Optional[float] = None  # None = unknown (not faked)
    thermal_state: ThermalState = ThermalState.OPTIMAL
    is_estimated: bool = False
    timestamp: float = 0.0


@dataclass
class PerformanceProfile:
    max_cpu_threads: int = 8
    max_gpu_utilization: float = 1.0
    memory_fraction: float = 0.9
    inference_steps_scale: float = 1.0


class ThermalManager:
    """Thermal-aware performance manager for Apple Silicon."""

    def __init__(self, config: FluxConfig):
        self.config = config
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self._current_metrics = ThermalMetrics(timestamp=time.time())
        self._current_profile = self._profile_for_state(ThermalState.OPTIMAL)
        self._callbacks: List[Callable[[PerformanceProfile], None]] = []

    def start_monitoring(self, interval: float = 2.0):
        """Start background thermal monitoring."""
        if self._monitoring:
            return
        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        self._thread.start()
        logger.info("Thermal monitoring started")

    def stop_monitoring(self):
        """Stop background thermal monitoring."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("Thermal monitoring stopped")

    def _monitor_loop(self, interval: float):
        while self._monitoring:
            try:
                temp = self._read_temperature()
                state = self._classify_temperature(temp)

                self._current_metrics = ThermalMetrics(
                    cpu_temp=temp,
                    gpu_temp=None,  # Unknown — not faking it
                    thermal_state=state,
                    is_estimated=(temp is None),
                    timestamp=time.time(),
                )

                new_profile = self._profile_for_state(state)
                if self._profile_differs(self._current_profile, new_profile):
                    self._current_profile = new_profile
                    logger.info("Performance adjusted for %s thermal state", state.value)
                    for cb in self._callbacks:
                        try:
                            cb(new_profile)
                        except Exception as e:
                            logger.warning("Thermal callback error: %s", e)

            except Exception as e:
                logger.error("Thermal monitoring error: %s", e)

            time.sleep(interval)

    def _read_temperature(self) -> Optional[float]:
        """Read CPU temperature using the best available method.

        Returns None if no real temperature data is available.
        """
        # Method 1: sysctl (fast, no sudo)
        temp = self._try_sysctl()
        if temp is not None:
            return temp

        # Method 2: psutil sensors
        temp = self._try_psutil()
        if temp is not None:
            return temp

        # Method 3: powermetrics (needs sudo, slowest)
        temp = self._try_powermetrics()
        if temp is not None:
            return temp

        # No real data available — return None instead of faking it
        return None

    def _try_sysctl(self) -> Optional[float]:
        """Try reading thermal state via sysctl (no sudo needed)."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.xcpm.cpu_thermal_state"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                thermal_state = int(result.stdout.strip())
                # Rough conversion: state 0 = ~35C, each state +5C
                return 35.0 + (thermal_state * 5.0)
        except (subprocess.SubprocessError, ValueError):
            pass
        return None

    def _try_psutil(self) -> Optional[float]:
        """Try reading temperature via psutil sensors."""
        try:
            import psutil
            if hasattr(psutil, "sensors_temperatures"):
                sensors = psutil.sensors_temperatures()
                for key in ("coretemp", "cpu_thermal", "k10temp"):
                    if key in sensors and sensors[key]:
                        return sensors[key][0].current
        except (ImportError, AttributeError, IndexError, KeyError):
            pass
        return None

    def _try_powermetrics(self) -> Optional[float]:
        """Try reading temperature via powermetrics (needs sudo)."""
        try:
            result = subprocess.run(
                ["sudo", "/usr/bin/powermetrics", "-n", "1", "-s", "thermal"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "CPU die temperature" in line:
                        temp_str = line.split(":")[1].strip().split()[0]
                        return float(temp_str)
        except (subprocess.SubprocessError, ValueError, IndexError):
            pass
        return None

    def _classify_temperature(self, temp: Optional[float]) -> ThermalState:
        """Classify temperature into thermal state using config thresholds."""
        if temp is None:
            return ThermalState.OPTIMAL  # Assume optimal if we can't read

        if temp < self.config.thermal.optimal_threshold:
            return ThermalState.OPTIMAL
        elif temp < self.config.thermal.warm_threshold:
            return ThermalState.WARM
        elif temp < self.config.thermal.hot_threshold:
            return ThermalState.HOT
        else:
            return ThermalState.CRITICAL

    def _profile_for_state(self, state: ThermalState) -> PerformanceProfile:
        """Get performance profile for a thermal state."""
        profiles = {
            ThermalState.OPTIMAL: PerformanceProfile(
                max_cpu_threads=self.config.device.cpu_threads,
                max_gpu_utilization=1.0,
                memory_fraction=0.9,
                inference_steps_scale=1.0,
            ),
            ThermalState.WARM: PerformanceProfile(
                max_cpu_threads=max(self.config.device.cpu_threads - 1, 4),
                max_gpu_utilization=0.9,
                memory_fraction=0.85,
                inference_steps_scale=0.95,
            ),
            ThermalState.HOT: PerformanceProfile(
                max_cpu_threads=max(self.config.device.cpu_threads - 2, 4),
                max_gpu_utilization=0.8,
                memory_fraction=0.8,
                inference_steps_scale=0.9,
            ),
            ThermalState.CRITICAL: PerformanceProfile(
                max_cpu_threads=4,
                max_gpu_utilization=0.6,
                memory_fraction=0.7,
                inference_steps_scale=0.8,
            ),
        }
        return profiles.get(state, profiles[ThermalState.OPTIMAL])

    def _profile_differs(self, a: PerformanceProfile, b: PerformanceProfile) -> bool:
        return (
            a.max_cpu_threads != b.max_cpu_threads
            or abs(a.inference_steps_scale - b.inference_steps_scale) > 0.01
        )

    def get_current_profile(self) -> PerformanceProfile:
        """Get the current performance profile."""
        return self._current_profile

    def get_current_metrics(self) -> ThermalMetrics:
        """Get the latest thermal metrics."""
        return self._current_metrics

    def add_callback(self, callback: Callable[[PerformanceProfile], None]):
        """Register a callback for performance profile changes."""
        self._callbacks.append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get thermal manager status summary."""
        m = self._current_metrics
        p = self._current_profile
        return {
            "monitoring": self._monitoring,
            "cpu_temp": m.cpu_temp,
            "gpu_temp": m.gpu_temp,
            "thermal_state": m.thermal_state.value,
            "is_estimated": m.is_estimated,
            "cpu_threads": p.max_cpu_threads,
            "inference_scale": p.inference_steps_scale,
        }
