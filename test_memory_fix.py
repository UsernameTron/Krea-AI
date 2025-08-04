#!/usr/bin/env python3
"""
Test Memory Fix for FLUX.1 Krea Pipeline Loading
Quick validation of memory settings
"""

import torch
import os
from diffusers import FluxPipeline

# Set memory-safe environment
os.environ['PYTORCH_MPS_MEMORY_FRACTION'] = '0.8'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable limit
os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.6'

def test_pipeline_loading():
    """Test if pipeline can load without memory errors"""
    print("🧪 Testing FLUX.1 Krea Pipeline Loading with Memory Fixes")
    print("=" * 60)
    
    print(f"📱 PyTorch: {torch.__version__}")
    print(f"🎯 MPS Available: {'✅' if torch.backends.mps.is_available() else '❌'}")
    print(f"💾 Memory Fraction: {os.environ.get('PYTORCH_MPS_MEMORY_FRACTION')}")
    print(f"🔧 High Watermark: {os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO')}")
    
    try:
        print(f"\n📥 Loading pipeline...")
        
        # Clear cache before loading
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Load pipeline with memory fixes
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Krea-dev",
            torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        
        print("✅ Pipeline loaded successfully!")
        
        # Move to MPS
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "mps":
            torch.mps.empty_cache()
        
        pipeline = pipeline.to(device)
        print(f"✅ Pipeline moved to {device}")
        
        # Test optimizations
        try:
            pipeline.enable_model_cpu_offload()
            print("✅ CPU offload enabled")
        except:
            print("⚠️  CPU offload not available")
        
        try:
            pipeline.enable_attention_slicing("auto")
            print("✅ Attention slicing enabled")
        except:
            print("⚠️  Attention slicing not available")
        
        try:
            pipeline.enable_vae_slicing()
            print("✅ VAE slicing enabled")
        except:
            print("⚠️  VAE slicing not available")
        
        try:
            pipeline.enable_vae_tiling()
            print("✅ VAE tiling enabled")  
        except:
            print("⚠️  VAE tiling not available")
        
        print(f"\n🎉 SUCCESS: Pipeline loaded and optimized!")
        print(f"💡 Memory fixes resolved the 25GB MPS limit issue")
        
        # Cleanup
        del pipeline
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ FAILED: {error_msg}")
        
        if "out of memory" in error_msg.lower():
            print(f"\n💾 Memory issue still present:")
            print(f"1. Try: export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0")
            print(f"2. Try: export PYTORCH_MPS_MEMORY_FRACTION=0.7") 
            print(f"3. Close other applications")
        
        return False

if __name__ == "__main__":
    success = test_pipeline_loading()
    if success:
        print(f"\n🚀 Ready for high-performance generation!")
    else:
        print(f"\n🔧 Additional memory tuning needed")