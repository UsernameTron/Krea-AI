# FLUX.1 Krea AI — Claude Code Improvement Prompt

> **Copy everything below this line and paste into Claude Code.**

---

## Context

You are working on a FLUX.1 Krea AI text-to-image generation application located at `/Users/cpconnor/Krea AI/flux-krea/`. This is a local implementation of the FLUX.1 Krea [dev] 12B parameter diffusion model, optimized for Apple Silicon M4 Pro with Metal Performance Shaders, Neural Engine acceleration, and thermal management.

**Tech Stack:** PyTorch 2.7.1+, diffusers 0.34.0+, Gradio 5.0.0+, CoreML 8.0+, Python 3.10.13

## Current State & Problems

The codebase has significant technical debt from iterative development. There are **20+ Python files** with massive duplication:

- **4 pipeline implementations** doing essentially the same thing: `flux_krea_official.py`, `flux_krea_m4_optimized.py`, `maximum_performance_pipeline.py`, `flux_interactive.py`
- **3 web UI variants**: `flux_web_ui_official.py`, `flux_web_m4_optimized.py`, `flux_web_debug.py`
- **6 shell launchers** with hardcoded `HF_TOKEN='YOUR_HF_TOKEN_HERE'` placeholders
- **No centralized configuration** — settings scattered across .env, shell scripts, Python files, and magic numbers everywhere
- **No tests whatsoever**
- **Incomplete error handling** — MPS watermark ratio errors in `flux_metal_kernels.py`, inconsistent fallback chains
- **README.md has conflicting information** between original and Apple Silicon sections

## Improvement Plan

Execute these phases in order. After each phase, verify nothing is broken before proceeding.

---

### Phase 1: Unified Configuration System

Create a single source of truth for all configuration.

**Create `config.py`:**
```python
"""
Centralized configuration for FLUX.1 Krea.
Priority: CLI args > environment variables > config.yaml > defaults
"""
```

Requirements:
- Single `FluxConfig` dataclass with all settings: model path, device preferences, MPS settings (watermark ratios, memory fractions, allocator policy), generation defaults (width, height, steps, guidance_scale), thermal thresholds, web UI settings (host, port), output directory
- Load from `config.yaml` if present, override with environment variables (prefixed `FLUX_`), override with CLI args
- HF_TOKEN loaded from environment only — never hardcoded anywhere
- All magic numbers currently scattered across files centralized here with descriptive names
- Validation on load (e.g., width/height must be multiples of 64, steps > 0, guidance_scale > 0)

**Create `config.yaml` template:**
```yaml
model:
  id: "black-forest-labs/FLUX.1-Krea-dev"
  dtype: "bfloat16"
  
device:
  preferred: "mps"  # mps, cpu, auto
  mps_watermark_ratio: 0.8
  mps_memory_fraction: 0.85
  mps_allocator_policy: "expandable_segments"
  cpu_threads: 8  # M4 Pro: 8 performance cores

generation:
  width: 1024
  height: 1024
  steps: 28
  guidance_scale: 4.5
  max_sequence_length: 256

thermal:
  mode: "adaptive"  # maximum, balanced, efficient, adaptive
  optimal_threshold: 70
  warm_threshold: 80
  hot_threshold: 90

web:
  host: "127.0.0.1"
  port: 7860
  share: false

output:
  directory: "./outputs"
  filename_pattern: "flux_{timestamp}_{prompt_slug}.png"
```

**Remove all hardcoded settings** from every Python file and shell script. Replace with `from config import get_config`.

---

### Phase 2: Consolidated Pipeline Architecture

Merge all 4 pipeline variants into a single, configurable pipeline.

**Create `pipeline.py`:**

```python
"""
Unified FLUX.1 Krea Pipeline
Replaces: flux_krea_official.py, flux_krea_m4_optimized.py, 
          maximum_performance_pipeline.py, flux_interactive.py
"""
```

Requirements:
- Single `FluxKreaPipeline` class that accepts a `FluxConfig`
- Optimization levels as an enum: `NONE`, `STANDARD`, `MAXIMUM`
  - `NONE`: Baseline diffusers FluxPipeline, no extras
  - `STANDARD`: MPS device, attention slicing, VAE tiling, CPU offload
  - `MAXIMUM`: All of STANDARD + Metal kernel optimizations + Neural Engine + thermal management
- Clean `load()` method that applies optimizations progressively based on level
- `generate()` method with proper `torch.inference_mode()` context, MPS cache management, and comprehensive error handling
- Fallback chain: if MAXIMUM optimization fails during load, fall back to STANDARD with a warning; if STANDARD fails, fall back to NONE
- Memory cleanup method that safely handles MPS cache clearing (catch the watermark ratio errors that currently plague `flux_metal_kernels.py`)
- Progress callback support for web UI integration
- Proper generator handling for seed reproducibility on both MPS and CPU devices

**Key fixes to incorporate:**
- The `_cleanup_memory()` method must wrap `torch.mps.empty_cache()` in try/except that catches RuntimeError with "low watermark ratio" 
- `enable_model_cpu_offload()` and `enable_sequential_cpu_offload()` conflict with `.to(device)` — pick one strategy, don't try both
- `max_sequence_length=128` in `flux_interactive.py` prevents black images but limits prompt quality. Default to 256, make configurable
- `guidance_scale=4.0` vs `4.5` inconsistency — standardize to 4.5 (Krea default)
- VAE tiling/slicing can cause black images on MPS in some PyTorch versions. Make these toggleable via config, default ON but with clear documentation

**Delete after consolidation:** `flux_krea_official.py`, `flux_krea_m4_optimized.py`, `maximum_performance_pipeline.py`, `flux_interactive.py`, `flux_performance_fix.py` (if exists)

---

### Phase 3: Consolidated Optimization Modules

The three optimization modules (`flux_metal_kernels.py`, `flux_neural_engine_accelerator.py`, `thermal_performance_manager.py`) are well-structured but have issues.

**Refactor `flux_metal_kernels.py` → `optimizers/metal.py`:**
- Fix the watermark ratio error: `_configure_metal_environment()` sets both HIGH and LOW watermark ratios, but the LOW must be less than HIGH. Current code sets HIGH=0.8 and LOW=0.6 which is correct, but `torch.mps.set_per_process_memory_fraction(0.85)` can conflict. Pick one approach.
- The `_manual_metal_attention()` method has a bug: it processes `query` shape as `(batch_size, seq_len, head_dim)` but FLUX attention tensors are 4D `(batch, heads, seq_len, head_dim)`. Fix the dimension handling.
- `optimize_conv_operations()` hardcodes `stride=1, padding=0` which won't work for all convolutions in the pipeline. Accept these as parameters from the actual conv layer.
- The monkey-patching in `_replace_attention_with_metal()` is a no-op — it wraps the original forward but doesn't actually change behavior. Either implement real Metal optimization or remove the pretense.
- Remove `_fallback_attention()` that just raises — if you don't want fallback, don't have the method.

**Refactor `flux_neural_engine_accelerator.py` → `optimizers/neural_engine.py`:**
- `_create_test_model()` tries to create and convert a model on init, which will fail if coremltools isn't installed. Make this lazy — check availability only when first needed.
- `_create_neural_engine_text_encoder()` is a stub that returns the original encoder. Either implement it or mark the entire text encoder optimization as TODO and don't claim "2.5-3x" performance boost.
- The `_optimize_transformer_blocks()` method compiles attention and MLP but explicitly doesn't replace them (comments say "not altering in-place to avoid unintended regressions"). This means Neural Engine optimization is effectively disabled. Document this honestly.
- Clean up the hash-based cache keys — `hash(str(module))` is not reliable across Python sessions.

**Refactor `thermal_performance_manager.py` → `optimizers/thermal.py`:**
- `_get_apple_silicon_temperature()` calls `sudo powermetrics` which requires root. This will fail for most users. Move it to last resort, not first attempt.
- The temperature estimation fallback using CPU percentage + random variation is misleading — it presents fake data as real temperature. Either label it clearly as "estimated" or use `subprocess.run(["sysctl", "hw.sensors.cpu_thermal"])` which doesn't need sudo.
- GPU temperature estimation as `cpu_temp + 5.0` and memory as `cpu_temp - 10.0` are completely made up. Remove these or label them as rough estimates.
- The `_apply_light/moderate/aggressive_throttling()` methods create new PerformanceProfile objects but nothing actually consumes these to change pipeline behavior. Wire them into the pipeline or remove.

**Create `optimizers/__init__.py`** that provides a clean `get_optimizer(config)` factory function.

---

### Phase 4: Unified Web Interface

Merge all 3 web UIs into a single Gradio interface.

**Create `app.py`:**

Requirements:
- Single Gradio Blocks interface with all features from the 3 variants
- Settings panel with optimization level selector (None/Standard/Maximum)
- Real-time system info display (device, memory, thermal state if available)
- Progress tracking that actually shows step-by-step diffusion progress (not just 3 fake progress stages like current implementation)
- Gallery view showing previous generations in the session
- Performance metrics display after each generation (time, memory, device used)
- Error display that shows actionable solutions (the current `flux_interactive.py` error messages are good — port those)
- Remove the hard-dependency on background thread loading — make it optional and handle the race condition properly (current code has a `while self.is_loading: sleep(0.5)` busy-wait)

**Delete after consolidation:** `flux_web_ui_official.py`, `flux_web_m4_optimized.py`, `flux_web_debug.py`

---

### Phase 5: Unified Entry Points

**Create `main.py`:**
```
Usage:
  python main.py generate --prompt "..." [--width 1024] [--height 1024] [--steps 28] [--seed 42]
  python main.py web [--port 7860]
  python main.py benchmark [--quick]
  python main.py info  # Show system info, device capabilities, model status
```

- Uses `argparse` with subcommands
- Imports from consolidated `pipeline.py`, `app.py`, `config.py`
- `info` subcommand shows: PyTorch version, MPS availability, memory, estimated model size, HF_TOKEN status, model download status

**Create single `launch.sh`:**
```bash
#!/bin/bash
cd "$(dirname "$0")"
source .env 2>/dev/null || true
python main.py "$@"
```

**Delete:** All 6 existing shell launchers (`quick_launch.sh`, `launch_optimized.sh`, `launch_memory_safe.sh`, `memory_optimizer_m4_pro.sh`, `restart_flux.sh`, `create_desktop_shortcut.sh`)

---

### Phase 6: Documentation Rewrite

**Rewrite `README.md`** with this structure:
1. What this is (one paragraph)
2. Requirements (macOS with Apple Silicon, Python 3.10+, PyTorch with MPS support, HuggingFace account with model access)
3. Setup (step by step: clone, create venv, install deps, get HF token, request model access, configure)
4. Usage (CLI generate, web UI, benchmark)
5. Configuration (config.yaml reference)
6. Optimization levels explained
7. Troubleshooting (black images, memory errors, MPS errors, model access errors — consolidate from all current error messages)
8. Architecture overview (brief, with file listing)

**Update `CLAUDE.md`** to reflect the new consolidated architecture.

**Delete:** `FLUX_KREA_SETUP.md`, `SETUP_TOKEN.md` (fold into README)

---

### Phase 7: Testing Foundation

**Create `tests/` directory with:**

- `tests/test_config.py`: Config loading, validation, environment variable override, defaults
- `tests/test_pipeline.py`: Pipeline initialization at each optimization level, generate with mock model (don't actually load the 24GB model in tests), MPS fallback to CPU, memory cleanup
- `tests/test_metal.py`: Metal kernel context manager, watermark ratio error handling, memory stats
- `tests/test_thermal.py`: Thermal state determination, throttling profile calculation
- `tests/conftest.py`: Fixtures for mock pipeline, test config, MPS availability check

Use `pytest`. Tests should be runnable without the actual FLUX model downloaded (mock the diffusers pipeline). Mark tests that require MPS with `@pytest.mark.skipif(not torch.backends.mps.is_available())`.

---

### Phase 8: Dependency Cleanup

**Pin dependencies in `requirements.txt`:**
```
torch>=2.7.1
diffusers>=0.34.0
transformers>=4.44.0
gradio>=5.0.0
safetensors>=0.4.0
psutil>=5.9.0
pyyaml>=6.0
Pillow>=10.0.0
```

**Create `requirements-optional.txt`:**
```
coremltools>=8.0  # Neural Engine acceleration
mlx>=0.0.3  # MLX framework (alternative backend)
```

**Create `.env.example`:**
```bash
# Required: HuggingFace token with read access
# Get yours at: https://huggingface.co/settings/tokens
HF_TOKEN=hf_your_token_here

# Optional: Override config.yaml settings
# FLUX_DEVICE_PREFERRED=mps
# FLUX_GENERATION_STEPS=28
```

---

### Final Directory Structure

After all phases, the project should look like:

```
flux-krea/
├── main.py                    # CLI entry point
├── app.py                     # Web UI (Gradio)
├── pipeline.py                # Unified pipeline
├── config.py                  # Configuration system
├── config.yaml                # Default configuration
├── optimizers/
│   ├── __init__.py           # Optimizer factory
│   ├── metal.py              # Metal Performance Shaders
│   ├── neural_engine.py      # CoreML/Neural Engine
│   └── thermal.py            # Thermal management
├── utils/
│   ├── __init__.py
│   ├── benchmark.py          # Benchmark runner (cleaned up)
│   └── monitor.py            # Performance monitor (cleaned up)
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_pipeline.py
│   ├── test_metal.py
│   └── test_thermal.py
├── models/                    # Downloaded model files
│   └── FLUX.1-Krea-dev/
├── outputs/                   # Generated images
├── launch.sh                  # Single launcher
├── requirements.txt           # Pinned dependencies
├── requirements-optional.txt  # Optional dependencies
├── .env.example              # Environment template
├── .python-version           # 3.10.13
├── README.md                 # Complete documentation
├── CLAUDE.md                 # AI assistant guidance
└── .gitignore
```

**Files to delete** (moved into consolidated modules):
- `flux_krea_official.py`
- `flux_krea_m4_optimized.py`
- `maximum_performance_pipeline.py`
- `flux_interactive.py`
- `flux_performance_fix.py`
- `flux_web_ui_official.py`
- `flux_web_m4_optimized.py`
- `flux_web_debug.py`
- `flux_metal_kernels.py` (→ optimizers/metal.py)
- `flux_neural_engine_accelerator.py` (→ optimizers/neural_engine.py)
- `thermal_performance_manager.py` (→ optimizers/thermal.py)
- `benchmark_runner.py` (→ utils/benchmark.py)
- `monitor_performance.py` (→ utils/monitor.py)
- `validate_optimizations.py`
- `quick_launch.sh`
- `launch_optimized.sh`
- `launch_memory_safe.sh`
- `memory_optimizer_m4_pro.sh`
- `restart_flux.sh`
- `create_desktop_shortcut.sh`
- `FLUX_KREA_SETUP.md`
- `SETUP_TOKEN.md`

---

## Critical Implementation Notes

1. **Never hardcode HF_TOKEN anywhere.** Always load from environment.
2. **MPS cache clearing is fragile.** Always wrap `torch.mps.empty_cache()` in try/except catching RuntimeError.
3. **Don't call both `.to(device)` and `enable_model_cpu_offload()`** — they conflict. Use CPU offload OR explicit device placement, not both.
4. **bfloat16 is correct for Apple Silicon** — don't change to float16, it causes black images on some MPS configurations.
5. **Test on CPU fallback** — everything must work if MPS is unavailable.
6. **Preserve the black image prevention logic** from `flux_interactive.py` (conservative guidance_scale=4.0, max_sequence_length=128 as optional safety mode).
7. **Back up everything before starting.** Copy the entire `flux-krea/` directory first.

## Execution Order

1. Back up the entire project directory
2. Phase 1: Configuration system (foundation for everything else)
3. Phase 2: Pipeline consolidation (core functionality)
4. Phase 3: Optimizer refactoring (supporting modules)
5. Phase 4: Web UI consolidation (user-facing)
6. Phase 5: Entry points (clean interface)
7. Phase 6: Documentation (explain the new architecture)
8. Phase 7: Tests (validate everything works)
9. Phase 8: Dependencies (lock it down)
10. Final validation: run `python main.py info`, `python main.py generate --prompt "test" --steps 5`, and `python -m pytest tests/`
