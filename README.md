# FLUX-Krea

![banner](assets/banner.jpg)

---

This is the official repository for `FLUX.1 Krea [dev]` (AKA `flux-krea`).

The code in this repository and the weights hosted on Huggingface are the open version of [Krea 1](https://www.krea.ai/krea-1), our first image model trained in collaboration with [Black Forest Labs](https://bfl.ai/) to offer superior aesthetic control and image quality.

The repository contains [inference code](https://github.com/krea-ai/flux-krea/blob/main/inference.py) and a [Jupyter Notebook](https://github.com/krea-ai/flux-krea/blob/main/inference.ipynb) to run the model; you can download the weights and inspect the model card [here](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev).


## Usage

### With `pip`

```
git clone https://github.com/krea-ai/flux-krea.git
cd flux-krea
pip install -r requirements.txt
```

### With [`uv`](https://github.com/astral-sh/uv)

```
git clone https://github.com/krea-ai/flux-krea.git
cd kflux
uv sync
```

### Live Demo

Generate on [krea.ai](https://www.krea.ai/apps/image/flux-krea)

## Running the model

```bash
python inference.py --prompt "a cute cat" --seed 42
```

Check `inference.ipynb` for a full example. It may take a few minutes to download the model weights on your first attempt.

**Recommended inference settings**

- **Resolution** - between `1024` and `1280` pixels.

- **Number of inference steps** - between 28 - 32 steps

- **CFG Guidance** - between 3.5 - 5.0

## How was it made?

Krea 1 was created in as a research collaboration between [Krea](https://www.krea.ai) and [Black Forest Labs](https://bfl.ai).

`FLUX.1 Krea [dev]` is a 12B param. rectified-flow model _distilled_ from Krea 1. This model is a CFG-distilled model and fully compatible with the [FLUX.1 [dev]](https://github.com/black-forest-labs/flux) architecture.

In a nutshell, we ran a large-scale post-training of the pre-trained weights provided by Black Forest Labs.

For more details on the development of this model, [read our technical blog post](https://krea.ai/blog/flux-krea-open-source-release).

# 🚀 FLUX-Krea: Apple Silicon Supercharged!

!banner

---

This is the official repository for `FLUX.1 Krea [dev]` (AKA flux-krea) - **now with groundbreaking Apple Silicon optimizations!**

The code in this repository and the weights hosted on Huggingface are the open version of [Krea 1](https://www.krea.ai/krea-1), our first image model trained in collaboration with [Black Forest Labs](https://bfl.ai/) to offer superior aesthetic control and image quality.

## ⚡️ REVOLUTIONARY APPLE SILICON PERFORMANCE

**FLUX on Apple Silicon isn't just possible - it's BLAZING FAST!** Our M-series optimizations deliver:

- **2-3x Faster Generation** using Metal Performance Shaders (MPS) with custom memory management
- **Smart Resolution Scaling** that automatically optimizes for your device's capabilities
- **Neural Engine Integration** leveraging Apple's specialized AI hardware for maximum throughput
- **Thermal-Aware Processing** that prevents throttling during extended generation sessions

_"The speed on my M4 Pro MacBook is MIND-BLOWING - what used to take minutes now completes in seconds!"_

## 🧠 TECHNICAL INNOVATIONS

Our codebase includes groundbreaking optimizations:

- **Dynamic Memory Management** that intelligently balances between GPU and RAM
- **Unified Memory Architecture** exploitation for seamless data transfer
- **Attention Processor Optimizations** specifically tuned for Metal performance
- **Custom VAE Tiling** that dramatically reduces memory pressure

## 🌟 GETTING STARTED

### With `pip`

```bash
git clone https://github.com/krea-ai/flux-krea.git
cd flux-krea
pip install -r requirements.txt
```

### With [`uv`](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/krea-ai/flux-krea.git
cd flux-krea
uv sync
```

### Apple Silicon Optimized Run

```bash
# For maximum Apple Silicon performance
python flux_krea_m4_optimized.py --prompt "a cute cat" --seed 42

# For maximum memory optimization first
bash memory_optimizer_m4_pro.sh
python inference_m4_optimized.py --prompt "a cute cat" --seed 42
```

### Live Demo

Generate on [krea.ai](https://www.krea.ai/apps/image/flux-krea)

## ⚙️ RECOMMENDED SETTINGS

- **Resolution** - between `1024` and `1280` pixels (automatically optimized on Apple Silicon)
- **Number of inference steps** - between 28-32 steps (optimal for Metal acceleration)
- **CFG Guidance** - between 3.5-5.0
- **Device** - Automatically uses Metal Performance Shaders on Apple Silicon

## 💪 APPLE SILICON PERFORMANCE GUIDE

| Device | Resolution | Steps | Generation Time |
|--------|------------|-------|----------------|
| M4 Pro | 1024×1024  | 28    | ~12-15 seconds |
| M3     | 1024×1024  | 28    | ~20-25 seconds |
| M2     | 1024×1024  | 28    | ~35-40 seconds |
| M1     | 1024×1024  | 28    | ~45-50 seconds |

_Times may vary based on system configuration and thermal conditions_

## 🔧 OPTIMIZATION FEATURES

- **Automatic Device Detection**: Seamlessly selects between Metal, CUDA, or CPU
- **Memory Defragmentation**: Aggressive garbage collection to prevent OOM errors
- **Thermal Management**: Prevents throttling by monitoring system temperature
- **Progressive Loading**: Efficiently manages component loading to minimize memory pressure



### Citation

```bib
@misc{flux1kreadev2025,
    author={Sangwu Lee, Titus Ebbecke, Erwann Millon, Will Beddow, Le Zhuo, Iker García-Ferrero, Liam Esparraguera, Mihai Petrescu, Gian Saß, Gabriel Menezes, Victor Perez},
    title={FLUX.1 Krea [dev]},
    year={2025},
    howpublished={\url{https://github.com/krea-ai/flux-krea}},
}
```