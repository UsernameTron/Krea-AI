# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

FLUX-Krea is a PyTorch-based text-to-image generation application using the FLUX.1 Krea [dev] 12B parameter model, optimized for Apple Silicon (MPS). The codebase was consolidated from ~40 files into a clean modular architecture.

## Common Commands

```bash
# System info (no model load)
python main.py info

# Generate an image
python main.py generate -p "a cute cat" --seed 42

# Launch web UI
python main.py web

# Run benchmarks
python main.py benchmark --quick

# Run tests
python -m pytest tests/ -v

# Single launcher (loads .env)
./launch.sh info
```

## File Structure

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point — argparse subcommands: generate, web, benchmark, info |
| `app.py` | Gradio web UI — FluxWebApp with background loading, timeout, debug log |
| `pipeline.py` | FluxKreaPipeline — unified pipeline with optimization levels and fallback chain |
| `config.py` | FluxConfig dataclass — loads from config.yaml + env vars + CLI overrides |
| `config.yaml` | Default settings — generation params, device config, thresholds |
| `optimizers/metal.py` | MetalOptimizer — MPS environment, VAE autocast, memory stats |
| `optimizers/neural_engine.py` | NeuralEngineOptimizer — CoreML stub (not yet functional) |
| `optimizers/thermal.py` | ThermalManager — background temp monitoring, performance profiles |
| `utils/benchmark.py` | run_benchmark() — tests generation at different step counts |
| `utils/monitor.py` | get_system_info() — PyTorch, MPS, memory, CPU info |
| `launch.sh` | Shell launcher — loads .env and delegates to main.py |

## Key Architecture Decisions

- **Single pipeline** with 3 optimization levels: none, standard, maximum
- **Fallback chain**: maximum -> standard -> none (with warnings)
- **Config priority**: CLI args > env vars (FLUX_ prefix) > config.yaml > defaults
- **MPS watermark ratio**: 0.8 (never 0.0 — causes cache errors)
- **bfloat16**: Required dtype for Apple Silicon (float16 causes black images)
- **Safety mode**: max_sequence_length=128 + guidance_scale=4.0 prevents black images
- **No fake data**: Thermal/Neural Engine report None when real data unavailable

## Device Support

Primary: Apple Silicon via MPS (Metal Performance Shaders)
Fallback: CPU (automatic if MPS unavailable)
Model: `black-forest-labs/FLUX.1-Krea-dev` from Hugging Face Hub

## Environment

- Python 3.10.13 (.python-version)
- PyTorch 2.x with MPS
- HF_TOKEN required (env var only, never hardcoded)
- `.env` sets MPS tuning parameters
