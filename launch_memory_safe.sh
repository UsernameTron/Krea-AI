#!/bin/bash
"""
Launch FLUX.1 Krea Interactive with Memory-Safe Settings
Optimized for 25GB MPS memory limit
"""

echo "💾 FLUX.1 Krea [dev] - Memory-Safe Interactive Studio"
echo "===================================================="

# Memory-safe MPS environment variables for 25GB limit
export PYTORCH_MPS_MEMORY_FRACTION=0.75
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Disable watermark limit
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
export PYTORCH_MPS_ALLOCATOR_POLICY=expandable_segments
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Conservative threading for M4 Pro
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "💾 Memory-Safe Environment Variables Set:"
echo "   PYTORCH_MPS_MEMORY_FRACTION: $PYTORCH_MPS_MEMORY_FRACTION"
echo "   PYTORCH_MPS_HIGH_WATERMARK_RATIO: $PYTORCH_MPS_HIGH_WATERMARK_RATIO"
echo "   OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

# Check available memory
echo "🔍 System Memory Check:"
memory_gb=$(sysctl hw.memsize | awk '{print int($2/1024/1024/1024)}')
echo "   Total System Memory: ${memory_gb}GB"

if [ $memory_gb -lt 32 ]; then
    echo "   ⚠️  Limited memory detected - using conservative settings"
else
    echo "   ✅ Sufficient memory for standard generation"
fi

# Check HF token
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "⚠️  HF_TOKEN not set. Please set it:"
    echo "   export HF_TOKEN=your_token_here"
    echo "   Or run: ./auth_validation.sh"
    echo ""
fi

# Launch memory-conservative studio
echo "🎨 Launching Memory-Safe Interactive Studio..."
python3 flux_interactive_conservative.py "$@"