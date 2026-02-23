"""
Centralized configuration for FLUX.1 Krea.
Priority: CLI args > environment variables > config.yaml > defaults
"""

import os
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml


class OptimizationLevel(Enum):
    NONE = "none"          # Baseline diffusers FluxPipeline
    STANDARD = "standard"  # MPS + attention slicing + VAE tiling + CPU offload
    MAXIMUM = "maximum"    # All of STANDARD + Metal kernels + thermal management


class ThermalMode(Enum):
    MAXIMUM = "maximum"      # Full performance, ignore thermal limits
    BALANCED = "balanced"    # Balance performance and thermals
    EFFICIENT = "efficient"  # Prioritize thermal management
    ADAPTIVE = "adaptive"    # Adaptive management based on temperature


@dataclass
class ModelConfig:
    id: str = "black-forest-labs/FLUX.1-Krea-dev"
    dtype: str = "bfloat16"


@dataclass
class DeviceConfig:
    preferred: str = "mps"  # mps, cpu, auto
    mps_watermark_ratio: float = 0.8
    mps_low_watermark_ratio: float = 0.6
    mps_memory_fraction: float = 0.85
    mps_allocator_policy: str = "expandable_segments"
    mps_prefer_fast_alloc: bool = True
    mps_enable_fallback: bool = True
    cpu_threads: int = 8  # M4 Pro: 8 performance cores


@dataclass
class GenerationConfig:
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance_scale: float = 4.5
    max_sequence_length: int = 256


@dataclass
class SafetyModeConfig:
    """Conservative settings to prevent black images on MPS."""
    enabled: bool = False
    guidance_scale: float = 4.0
    max_sequence_length: int = 128
    disable_vae_tiling: bool = True
    attention_slice_size: int = 1


@dataclass
class ThermalConfig:
    mode: str = "adaptive"
    optimal_threshold: int = 70
    warm_threshold: int = 80
    hot_threshold: int = 90


@dataclass
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False


@dataclass
class OutputConfig:
    directory: str = "./outputs"
    filename_pattern: str = "flux_{timestamp}_{prompt_slug}.png"


@dataclass
class FluxConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    safety_mode: SafetyModeConfig = field(default_factory=SafetyModeConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    web: WebConfig = field(default_factory=WebConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    optimization_level: str = "standard"

    def validate(self):
        """Validate configuration values."""
        errors = []

        # Resolution must be multiples of 64
        if self.generation.width % 64 != 0:
            errors.append(f"width must be a multiple of 64, got {self.generation.width}")
        if self.generation.height % 64 != 0:
            errors.append(f"height must be a multiple of 64, got {self.generation.height}")

        # Positive values
        if self.generation.steps <= 0:
            errors.append(f"steps must be > 0, got {self.generation.steps}")
        if self.generation.guidance_scale <= 0:
            errors.append(f"guidance_scale must be > 0, got {self.generation.guidance_scale}")
        if self.generation.max_sequence_length <= 0:
            errors.append(f"max_sequence_length must be > 0, got {self.generation.max_sequence_length}")

        # Watermark ratio range
        if not 0.0 <= self.device.mps_watermark_ratio <= 1.0:
            errors.append(f"mps_watermark_ratio must be between 0.0 and 1.0, got {self.device.mps_watermark_ratio}")
        if not 0.0 <= self.device.mps_low_watermark_ratio <= 1.0:
            errors.append(f"mps_low_watermark_ratio must be between 0.0 and 1.0, got {self.device.mps_low_watermark_ratio}")
        if self.device.mps_low_watermark_ratio >= self.device.mps_watermark_ratio:
            errors.append(
                f"mps_low_watermark_ratio ({self.device.mps_low_watermark_ratio}) "
                f"must be less than mps_watermark_ratio ({self.device.mps_watermark_ratio})"
            )

        # Memory fraction range
        if not 0.0 < self.device.mps_memory_fraction <= 1.0:
            errors.append(f"mps_memory_fraction must be between 0.0 and 1.0, got {self.device.mps_memory_fraction}")

        # Thermal thresholds ascending
        if not (self.thermal.optimal_threshold < self.thermal.warm_threshold < self.thermal.hot_threshold):
            errors.append("Thermal thresholds must be ascending: optimal < warm < hot")

        # Optimization level valid
        try:
            OptimizationLevel(self.optimization_level)
        except ValueError:
            errors.append(f"optimization_level must be one of {[e.value for e in OptimizationLevel]}, got {self.optimization_level}")

        if errors:
            raise ValueError("Configuration validation failed:\n  " + "\n  ".join(errors))

    def get_optimization_level(self) -> OptimizationLevel:
        return OptimizationLevel(self.optimization_level)

    def get_thermal_mode(self) -> ThermalMode:
        return ThermalMode(self.thermal.mode)

    def get_effective_generation_params(self) -> dict:
        """Return generation params, applying safety mode overrides if enabled."""
        params = {
            "width": self.generation.width,
            "height": self.generation.height,
            "steps": self.generation.steps,
            "guidance_scale": self.generation.guidance_scale,
            "max_sequence_length": self.generation.max_sequence_length,
        }
        if self.safety_mode.enabled:
            params["guidance_scale"] = self.safety_mode.guidance_scale
            params["max_sequence_length"] = self.safety_mode.max_sequence_length
        return params


def _apply_yaml(config: FluxConfig, yaml_data: dict):
    """Apply YAML config data to a FluxConfig instance."""
    section_map = {
        "model": config.model,
        "device": config.device,
        "generation": config.generation,
        "safety_mode": config.safety_mode,
        "thermal": config.thermal,
        "web": config.web,
        "output": config.output,
    }
    for section_name, section_obj in section_map.items():
        if section_name in yaml_data and isinstance(yaml_data[section_name], dict):
            for key, value in yaml_data[section_name].items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)

    if "optimization_level" in yaml_data:
        config.optimization_level = yaml_data["optimization_level"]


def _apply_env_vars(config: FluxConfig):
    """Override config with FLUX_ prefixed environment variables."""
    env_map = {
        "FLUX_MODEL_ID": ("model", "id"),
        "FLUX_MODEL_DTYPE": ("model", "dtype"),
        "FLUX_DEVICE_PREFERRED": ("device", "preferred"),
        "FLUX_DEVICE_MPS_WATERMARK_RATIO": ("device", "mps_watermark_ratio", float),
        "FLUX_DEVICE_MPS_MEMORY_FRACTION": ("device", "mps_memory_fraction", float),
        "FLUX_DEVICE_CPU_THREADS": ("device", "cpu_threads", int),
        "FLUX_GENERATION_WIDTH": ("generation", "width", int),
        "FLUX_GENERATION_HEIGHT": ("generation", "height", int),
        "FLUX_GENERATION_STEPS": ("generation", "steps", int),
        "FLUX_GENERATION_GUIDANCE_SCALE": ("generation", "guidance_scale", float),
        "FLUX_GENERATION_MAX_SEQUENCE_LENGTH": ("generation", "max_sequence_length", int),
        "FLUX_SAFETY_MODE": ("safety_mode", "enabled", lambda v: v.lower() in ("1", "true", "yes")),
        "FLUX_THERMAL_MODE": ("thermal", "mode"),
        "FLUX_WEB_HOST": ("web", "host"),
        "FLUX_WEB_PORT": ("web", "port", int),
        "FLUX_OUTPUT_DIRECTORY": ("output", "directory"),
        "FLUX_OPTIMIZATION_LEVEL": None,  # handled specially
    }

    section_map = {
        "model": config.model,
        "device": config.device,
        "generation": config.generation,
        "safety_mode": config.safety_mode,
        "thermal": config.thermal,
        "web": config.web,
        "output": config.output,
    }

    for env_key, mapping in env_map.items():
        value = os.environ.get(env_key)
        if value is None:
            continue

        if mapping is None:
            # Special case for top-level fields
            if env_key == "FLUX_OPTIMIZATION_LEVEL":
                config.optimization_level = value
            continue

        section_name = mapping[0]
        attr_name = mapping[1]
        converter = mapping[2] if len(mapping) > 2 else str

        section_obj = section_map[section_name]
        try:
            setattr(section_obj, attr_name, converter(value))
        except (ValueError, TypeError):
            pass  # Skip invalid env var values silently


def _apply_overrides(config: FluxConfig, overrides: dict):
    """Apply CLI argument overrides to config."""
    section_map = {
        "model": config.model,
        "device": config.device,
        "generation": config.generation,
        "safety_mode": config.safety_mode,
        "thermal": config.thermal,
        "web": config.web,
        "output": config.output,
    }

    for key, value in overrides.items():
        if value is None:
            continue

        # Handle top-level fields
        if key == "optimization_level":
            config.optimization_level = value
            continue

        # Handle dotted keys like "generation.width"
        if "." in key:
            section_name, attr_name = key.split(".", 1)
            if section_name in section_map:
                section_obj = section_map[section_name]
                if hasattr(section_obj, attr_name):
                    setattr(section_obj, attr_name, value)
            continue

        # Handle flat keys — search all sections
        for section_obj in section_map.values():
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)
                break


def get_config(config_path: Optional[str] = None, **overrides) -> FluxConfig:
    """
    Load configuration with priority: overrides > env vars > config.yaml > defaults.

    Args:
        config_path: Path to config.yaml. If None, searches current directory.
        **overrides: CLI argument overrides (e.g., width=512, steps=20).

    Returns:
        Validated FluxConfig instance.
    """
    config = FluxConfig()

    # Load from YAML if available
    if config_path is None:
        # Search in project directory
        candidates = [
            Path(__file__).parent / "config.yaml",
            Path.cwd() / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = str(candidate)
                break

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f)
            if yaml_data:
                _apply_yaml(config, yaml_data)

    # Override with environment variables
    _apply_env_vars(config)

    # Override with CLI args
    if overrides:
        _apply_overrides(config, overrides)

    # Validate
    config.validate()

    return config


def apply_mps_environment(config: FluxConfig):
    """Apply MPS-related environment variables from config.

    Must be called before importing torch or creating pipelines.
    """
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(config.device.mps_watermark_ratio))
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(config.device.mps_low_watermark_ratio))
    os.environ.setdefault("PYTORCH_MPS_MEMORY_FRACTION", str(config.device.mps_memory_fraction))
    os.environ.setdefault("PYTORCH_MPS_ALLOCATOR_POLICY", config.device.mps_allocator_policy)
    if config.device.mps_prefer_fast_alloc:
        os.environ.setdefault("PYTORCH_MPS_PREFER_FAST_ALLOC", "1")
    if config.device.mps_enable_fallback:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment. Never hardcoded."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
