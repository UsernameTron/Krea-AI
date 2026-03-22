# DevOps Handoff — FLUX-Krea

## Project Summary

FLUX-Krea is a text-to-image generation app using the FLUX.1 Krea [dev] 12B model, optimized for Apple Silicon (MPS). Provides CLI, web UI (Gradio), and benchmark interfaces.

## Environment Requirements

- Python 3.10.13
- PyTorch 2.x with MPS support
- Apple Silicon Mac (primary target)
- `HF_TOKEN` environment variable (Hugging Face access token)

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# System info (no model load)
python main.py info

# Generate image
python main.py generate -p "a cute cat" --seed 42

# Web UI
python main.py web

# Benchmark
python main.py benchmark --quick

# Via launcher (loads .env)
./launch.sh info
```

## Configuration

- `config.yaml` — default generation parameters, device config, thresholds
- `.env` — MPS tuning parameters, HF_TOKEN
- CLI args override env vars override config.yaml

## Security Notes

- `HF_TOKEN` must be set via environment variable only — never hardcoded
- `.env` file is gitignored

## Deployment Maturity

Local development tool. No production deployment pipeline.

## Known Tech Debt

- Neural Engine optimizer (`optimizers/neural_engine.py`) is a CoreML stub, not yet functional
- CPU fallback is automatic but untested at scale
