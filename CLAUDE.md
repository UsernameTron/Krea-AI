# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLUX-Krea is a PyTorch-based implementation of FLUX.1 Krea [dev], a 12B parameter text-to-image model that's a CFG-distilled version of Krea 1. The repository provides inference code for generating images from text prompts using the model.

## Common Commands

### Environment Setup
```bash
# Using pip
pip install -r requirements.txt

# Using uv (recommended)
uv sync
```

### Running Inference
```bash
# Basic image generation
python inference.py --prompt "a cute cat" --seed 42

# With custom parameters
python inference.py --prompt "your prompt" --width 1280 --height 1024 --guidance 4.5 --num-steps 28 --seed 42 --output my_image.png
```

### Jupyter Notebook
```bash
jupyter notebook inference.ipynb
```

## Architecture Overview

The codebase follows a modular structure centered around the FLUX diffusion model:

### Core Components

1. **Model Architecture** (`src/flux/model.py`)
   - `Flux`: Main transformer-based model with double and single stream blocks
   - `FluxParams`: Configuration dataclass for model parameters
   - Uses mixed attention mechanisms with both image and text streams

2. **Pipeline** (`src/flux/pipeline.py`)
   - `Pipeline`: Base class for inference workflow
   - `Sampler`: Simplified sampler that takes pre-loaded models
   - Handles the complete generation process from noise to final image

3. **Model Loading** (`src/flux/util.py`)
   - `load_flow_model()`: Loads the main FLUX model from HuggingFace
   - `load_ae()`: Loads the autoencoder for image encoding/decoding
   - `load_clip()` and `load_t5()`: Load text encoders
   - All models are loaded with bfloat16 precision by default

4. **Modules** (`src/flux/modules/`)
   - `layers.py`: Core attention and MLP layers
   - `autoencoder.py`: VAE for image encoding/decoding
   - `conditioner.py`: Text conditioning modules

### Key Configuration
- Model: "flux-krea-dev" (12B parameters)
- Default resolution: 1024x1024 (supports 1024-1280px range)
- Recommended steps: 28-32
- Recommended guidance: 3.5-5.0
- Uses HuggingFace Hub for model downloads

### Data Flow
1. Text prompt → CLIP/T5 embeddings
2. Random noise generation
3. Denoising process through FLUX model
4. VAE decoding to final image
5. Post-processing and saving

## Model Weights
The model automatically downloads weights from `black-forest-labs/FLUX.1-Krea-dev` on HuggingFace Hub. Local paths can be set via environment variables `FLUX` and `AE`.

## Device Support
Supports both CUDA and CPU inference, with CUDA being the default and recommended option for performance.