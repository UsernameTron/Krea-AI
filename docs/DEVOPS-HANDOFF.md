# DevOps Handoff — FLUX-Krea

## Project Summary

FLUX-Krea is a text-to-image generation application using the FLUX.1 Krea [dev] 12B parameter model. It is optimized for Apple Silicon (MPS/Metal) but falls back to CPU. The CLI entry point supports image generation, a Gradio web UI, benchmarking, and system info commands.

## Environment Requirements

| Requirement | Version / Notes |
|---|---|
| Python | 3.10.13 (see `.python-version`) |
| PyTorch | 2.x with MPS support |
| OS | macOS (Apple Silicon preferred) |
| RAM | ≥16 GB recommended (model is 12B parameters) |
| HF_TOKEN | Required — Hugging Face access token for model download |

### Python Dependencies

```bash
pip install -r requirements.txt
```

Key packages: `torch`, `diffusers`, `transformers`, `gradio`, `psutil`, `pyyaml`

## How to Run

```bash
# Load environment variables and run via launcher
./launch.sh info          # System info (no model load)
./launch.sh generate -p "a cute cat" --seed 42
./launch.sh web           # Gradio UI at http://localhost:7860
./launch.sh benchmark --quick

# Or run directly (manually set HF_TOKEN)
export HF_TOKEN=hf_your_token_here
python main.py info
python main.py generate -p "a photo of a dog"
python main.py web --port 7860
python main.py benchmark
```

## Configuration Reference

Configuration is loaded in priority order: CLI args > env vars (`FLUX_` prefix) > `config.yaml` > defaults.

Key settings in `config.yaml`:

| Setting | Default | Notes |
|---|---|---|
| `generation.width` | 1024 | Image width in pixels |
| `generation.height` | 1024 | Image height in pixels |
| `generation.num_inference_steps` | 28 | Inference step count |
| `generation.guidance_scale` | 4.5 | CFG guidance scale |
| `device.mps_watermark_ratio` | 0.8 | MPS memory limit (never set to 0.0) |
| `optimization_level` | standard | Options: none, standard, maximum |

Environment variable overrides use the `FLUX_` prefix, e.g.:
- `FLUX_OPTIMIZATION_LEVEL=maximum`
- `FLUX_GENERATION_WIDTH=512`

## Running Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=. --cov-report=term-missing
```

Current coverage: **96% overall** (246 tests). All modules ≥80%.

## Security Notes

- `HF_TOKEN` must be set as an environment variable — never hardcoded or committed
- `.env` file is gitignored; use it for local MPS tuning parameters
- MPS watermark ratio must remain ≥0.1 (0.0 causes cache errors)
- bfloat16 dtype is required on Apple Silicon (float16 produces black images)

## Deployment Maturity

**Development / Local Use Only.** This project is not configured for production deployment. There is no containerization, no API server, no authentication layer, and no CI/CD pipeline.

The Gradio web UI (`python main.py web`) is intended for local use only. Enabling `--share` exposes a public Gradio tunnel — use with caution.

## Known Technical Debt

- `optimizers/neural_engine.py` is a stub — CoreML integration is not yet functional
- No streaming or async generation support in the web UI
- Thermal monitoring uses `sysctl` and `powermetrics` — requires macOS; no Linux equivalent
- Model is loaded in full precision by default; quantization is not implemented
- No model caching between CLI invocations (model reloads on each `generate` call)
