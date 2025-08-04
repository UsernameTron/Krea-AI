#!/bin/bash
"""
Launch FLUX.1 Krea Interactive with Maximum Performance Optimizations
Enables Metal VAE, optimal MPS settings, and comprehensive optimizations
"""

echo "🚀 FLUX.1 Krea [dev] - Interactive Studio (Performance Optimized)"
echo "================================================================="

# Set optimal MPS environment variables
export PYTORCH_MPS_MEMORY_FRACTION=0.95
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6
export PYTORCH_MPS_ALLOCATOR_POLICY=expandable_segments
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Enable Metal VAE optimization
export KREA_ENABLE_METAL_VAE=1

# Diffusers optimizations
export DIFFUSERS_ATTENTION_SLICING=8
export DIFFUSERS_VAE_TILING=true
export DIFFUSERS_ENABLE_MEMORY_EFFICIENT_ATTENTION=true

# OMP threading for M4 Pro (8P + 4E cores)
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

echo "🔧 Performance Environment Variables Set:"
echo "   PYTORCH_MPS_MEMORY_FRACTION: $PYTORCH_MPS_MEMORY_FRACTION"
echo "   KREA_ENABLE_METAL_VAE: $KREA_ENABLE_METAL_VAE"
echo "   OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

# Check HF token
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN not set. Run authentication validation:"
    echo "   ./auth_validation.sh"
    echo ""
fi

# Launch interactive studio
echo "🎨 Launching Interactive Studio..."
python3 flux_interactive.py "$@"