# DevOps Handoff — FLUX-Krea

## Project Summary

FLUX-Krea is a text-to-image generation app using the FLUX.1 Krea [dev] 12B model, optimized for Apple Silicon (MPS). Provides CLI, web UI (Gradio), benchmark, and profiling interfaces.

## Environment Requirements

- Python 3.10.13
- PyTorch 2.x with MPS support
- Apple Silicon Mac (primary target)
- `HF_TOKEN` environment variable (Hugging Face access token)
- Optional: `coremltools` for Neural Engine acceleration

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

# Profile generation (stage-by-stage timing)
python main.py profile --prompt "a cute cat" --steps 20

# Via launcher (loads .env)
./launch.sh info
```

## Testing

```bash
# Run full suite (299 tests, 96% coverage)
python -m pytest tests/ -v -m "not requires_mps"

# With coverage report
python -m pytest tests/ -m "not requires_mps" --cov=. --cov-report=term-missing

# Lint
ruff check .

# Type check
mypy . --ignore-missing-imports
```

## CI/CD

GitHub Actions runs on push to `main` and pull requests:

| Job | What it checks |
|-----|----------------|
| `test` | pytest with `not requires_mps` marker (no Apple hardware needed) |
| `lint` | ruff check (E, F, W rules) |
| `typecheck` | mypy with `--ignore-missing-imports` |

Config files: `.github/workflows/ci.yml`, `ruff.toml`, `pyproject.toml`

## Configuration

- `config.yaml` — default generation parameters, device config, thresholds
- `.env` — MPS tuning parameters, HF_TOKEN
- CLI args override env vars override config.yaml

## Security Notes

- `HF_TOKEN` must be set via environment variable only — never hardcoded
- `.env` file is gitignored
- No secrets in CI — tests mock all external dependencies

## Deployment Maturity

Local development tool with CI/CD quality gates. No production deployment pipeline.

## Known Tech Debt

- Neural Engine VAE decoder conversion is implemented but conversion success depends on coremltools version and chip generation
- T5-XXL text encoder (~11B params) exceeds ANE's ~4GB limit — documented limitation, not a bug
- CPU fallback is automatic but untested at scale
