#!/bin/bash
"""
Launch FLUX.1 Krea with Apple Silicon M4 Pro Optimizations
"""

echo "🍎 Launching FLUX.1 Krea - Apple Silicon M4 Pro Optimized"
echo "========================================================"

# Kill existing processes
pkill -f "python.*flux" 2>/dev/null
lsof -ti:7860 | xargs kill -9 2>/dev/null

# Set Apple Silicon optimizations
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_PREFER_FAST_ALLOC=1
export PYTORCH_MPS_ALLOCATOR_POLICY=page
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Set HuggingFace token
export HF_TOKEN="your_huggingface_token_here"

echo "🚀 Starting optimized web interface..."
echo "🌐 Available at: http://localhost:7860"
echo "⚡ Apple Silicon optimizations active"

python flux_web_m4_optimized.py