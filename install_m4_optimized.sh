#!/bin/bash
"""
Install FLUX.1 Krea with Apple Silicon M4 Pro Optimizations
"""

echo "🍎 Installing FLUX.1 Krea - Apple Silicon M4 Pro Optimized"
echo "=========================================================="

# Verify we're on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "❌ This script is for Apple Silicon Macs only"
    exit 1
fi

# Update pip and install wheel
echo "📦 Updating pip and core packages..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with Metal Performance Shaders support
echo "⚡ Installing PyTorch with Metal Performance Shaders..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Install optimized requirements
echo "🔧 Installing optimized dependencies..."
pip install -r requirements_flux_krea.txt

# Install MLX for Neural Engine acceleration (optional)
echo "🧠 Installing MLX for Neural Engine acceleration..."
pip install mlx mlx-lm

# Verify Metal Performance Shaders availability
echo "✅ Verifying Apple Silicon optimizations..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    print('✅ Metal Performance Shaders ready!')
else:
    print('❌ MPS not available - check PyTorch installation')
"

echo ""
echo "🎉 Installation complete!"
echo "🚀 To run optimized version:"
echo "   python flux_krea_m4_optimized.py --prompt 'your prompt here'"
echo "🌐 To run optimized web interface:"
echo "   python flux_web_m4_optimized.py"