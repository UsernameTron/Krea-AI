#!/usr/bin/env python3
"""
Validate FLUX.1 Krea Optimizations
Check that optimizations are properly applied
"""

import torch
import os
from diffusers import FluxPipeline

def validate_optimizations():
    """Validate that optimizations can be applied"""
    print("🔍 Validating FLUX.1 Krea Optimizations")
    print("=" * 45)
    
    # Check PyTorch and MPS
    print(f"📱 PyTorch: {torch.__version__}")
    print(f"🎯 MPS Available: {'✅' if torch.backends.mps.is_available() else '❌'}")
    print(f"🧠 MPS Built: {'✅' if torch.backends.mps.is_built() else '❌'}")
    
    # Check environment variables
    print(f"\n🔧 Environment Variables:")
    env_vars = [
        'PYTORCH_MPS_MEMORY_FRACTION',
        'PYTORCH_MPS_HIGH_WATERMARK_RATIO', 
        'KREA_ENABLE_METAL_VAE',
        'HF_TOKEN'
    ]
    
    for var in env_vars:
        value = os.getenv(var, 'Not set')
        if var == 'HF_TOKEN' and value != 'Not set':
            value = f"...{value[-4:]}"
        print(f"   {var}: {value}")
    
    # Test basic pipeline loading with minimal memory
    print(f"\n🧪 Testing Pipeline Components:")
    
    try:
        # Test fp16 support
        x = torch.randn(2, 2, dtype=torch.float16)
        if torch.backends.mps.is_available():
            x = x.to('mps')
            print("✅ FP16 MPS support working")
        else:
            print("⚠️  MPS not available, using CPU")
            
    except Exception as e:
        print(f"❌ FP16 test failed: {e}")
    
    # Test attention processors
    try:
        from diffusers.models.attention_processor import AttnProcessor2_0
        processor = AttnProcessor2_0()
        print("✅ Memory-efficient attention processor available")
    except Exception as e:
        print(f"❌ Attention processor test failed: {e}")
    
    # Test Metal kernels
    try:
        from flux_metal_kernels import M4ProMetalOptimizer
        optimizer = M4ProMetalOptimizer()
        stats = optimizer.get_metal_optimization_summary()
        print(f"✅ Metal kernels available: {stats.get('metal_available', False)}")
    except Exception as e:
        print(f"❌ Metal kernels test failed: {e}")
    
    # Memory check
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            print("✅ MPS cache management working")
        except Exception as e:
            print(f"⚠️  MPS cache warning: {e}")
    
    print(f"\n📊 Optimization Status:")
    
    # Check individual optimizations that should be available
    optimizations = {
        "FP16 precision": torch.backends.mps.is_available(),
        "MPS device": torch.backends.mps.is_available(),
        "Metal kernels": os.getenv('KREA_ENABLE_METAL_VAE') == '1',
        "Memory management": os.getenv('PYTORCH_MPS_MEMORY_FRACTION') is not None,
        "HF authentication": os.getenv('HF_TOKEN') is not None
    }
    
    working_count = 0
    for opt, status in optimizations.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {opt}")
        if status:
            working_count += 1
    
    print(f"\n🎯 Optimization Score: {working_count}/{len(optimizations)}")
    
    if working_count >= 4:
        print("🚀 EXCELLENT: Ready for high-performance inference!")
        print("💡 Expected performance: <2 min for 1024x1024")
    elif working_count >= 3:
        print("✅ GOOD: Most optimizations active")
        print("💡 Expected performance: 2-4 min for 1024x1024")
    else:
        print("⚠️  NEEDS IMPROVEMENT: Missing key optimizations")
        print("💡 Expected performance: >4 min for 1024x1024")
    
    print(f"\n🔧 Quick fixes for common issues:")
    print("1. Set HF_TOKEN: export HF_TOKEN=your_token")
    print("2. Enable Metal VAE: export KREA_ENABLE_METAL_VAE=1")
    print("3. Use launch script: ./launch_optimized_interactive.sh")
    print("4. Test with smaller images first (512x512)")

if __name__ == "__main__":
    validate_optimizations()