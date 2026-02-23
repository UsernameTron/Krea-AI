# FLUX-Krea

![banner](assets/banner.jpg)

---

This is the official repository for `FLUX.1 Krea [dev]` (AKA `flux-krea`).

The code in this repository and the weights hosted on Huggingface are the open version of [Krea 1](https://www.krea.ai/krea-1), our first image model trained in collaboration with [Black Forest Labs](https://bfl.ai/) to offer superior aesthetic control and image quality.

## Requirements

- Python 3.10+
- PyTorch 2.x with MPS support (Apple Silicon) or CUDA
- ~24 GB storage for model weights (downloaded on first run)
- Hugging Face account with access to [FLUX.1-Krea-dev](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev)

## Setup

### 1. Clone and install

```bash
git clone https://github.com/krea-ai/flux-krea.git
cd flux-krea
pip install -r requirements.txt
```

### 2. Configure Hugging Face token

```bash
# Get your token from https://huggingface.co/settings/tokens
export HF_TOKEN="hf_your_token_here"
```

You must also [request access](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev) to the model repository.

### 3. Verify setup

```bash
python main.py info
```

## Usage

### Command Line

```bash
# Generate an image
python main.py generate --prompt "a cute cat sitting in a garden" --seed 42

# Custom resolution and steps
python main.py generate -p "mountain landscape at sunset" -W 1280 -H 768 -s 32

# Enable safety mode (prevents black images)
python main.py generate -p "portrait of a woman" --safety
```

### Web UI

```bash
# Launch Gradio interface
python main.py web

# Custom port
python main.py web --port 8080

# Create a public link
python main.py web --share
```

### Benchmark

```bash
python main.py benchmark
python main.py benchmark --quick
```

### Single launcher (loads .env automatically)

```bash
./launch.sh info
./launch.sh web
./launch.sh generate -p "your prompt"
```

### Jupyter Notebook

See `inference.ipynb` for an interactive example.

### Live Demo

Generate on [krea.ai](https://www.krea.ai/apps/image/flux-krea)

## Configuration

Settings are loaded from `config.yaml` and can be overridden with environment variables (prefixed `FLUX_`) or CLI arguments.

Key defaults:

| Setting | Default | Notes |
|---------|---------|-------|
| Resolution | 1024x1024 | Supports 512-1536, multiples of 64 |
| Steps | 28 | Recommended 28-32 |
| Guidance | 4.5 | Recommended 3.5-5.0 |
| Device | mps (auto) | Falls back to CPU if MPS unavailable |
| Optimization | standard | none, standard, or maximum |

### Optimization Levels

- **none** — Baseline diffusers FluxPipeline, no optimizations
- **standard** — MPS config + attention slicing + VAE tiling + CPU offload
- **maximum** — All of standard + Metal kernel optimizations + thermal management

### Safety Mode

If you encounter black images, enable safety mode which uses conservative settings (`max_sequence_length=128`, `guidance_scale=4.0`):

```bash
python main.py generate -p "your prompt" --safety
```

Or toggle it in the web UI settings panel.

## Troubleshooting

**Black images:** Enable safety mode, or reduce `max_sequence_length` to 128 and `guidance_scale` to 4.0.

**"Gated repo" error:** You need to [request access](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev) and set `HF_TOKEN`.

**Out of memory:** Reduce resolution to 768x768, reduce steps, or use `standard` optimization level with CPU offload.

**MPS watermark errors:** The `.env` file sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8`. Don't set this to 0.0.

## Architecture

```
flux-krea/
├── main.py              # CLI entry point (generate, web, benchmark, info)
├── app.py               # Gradio web interface
├── pipeline.py          # Unified FluxKreaPipeline
├── config.py            # FluxConfig + loader (YAML, env vars, CLI)
├── config.yaml          # Default configuration
├── optimizers/
│   ├── metal.py         # Metal Performance Shaders optimization
│   ├── neural_engine.py # CoreML/Neural Engine (stub)
│   └── thermal.py       # Thermal monitoring and throttling
├── utils/
│   ├── benchmark.py     # Benchmark runner
│   └── monitor.py       # System info
├── tests/               # pytest test suite
├── launch.sh            # Single launcher script
├── .env                 # MPS environment settings
└── inference.ipynb      # Jupyter notebook example
```

## How was it made?

Krea 1 was created as a research collaboration between [Krea](https://www.krea.ai) and [Black Forest Labs](https://bfl.ai).

`FLUX.1 Krea [dev]` is a 12B param. rectified-flow model _distilled_ from Krea 1. This model is a CFG-distilled model and fully compatible with the [FLUX.1 [dev]](https://github.com/black-forest-labs/flux) architecture.

For more details on the development of this model, [read our technical blog post](https://krea.ai/blog/flux-krea-open-source-release).

### Citation

```bib
@misc{flux1kreadev2025,
    author={Sangwu Lee, Titus Ebbecke, Erwann Millon, Will Beddow, Le Zhuo, Iker García-Ferrero, Liam Esparraguera, Mihai Petrescu, Gian Saß, Gabriel Menezes, Victor Perez},
    title={FLUX.1 Krea [dev]},
    year={2025},
    howpublished={\url{https://github.com/krea-ai/flux-krea}},
}
```
