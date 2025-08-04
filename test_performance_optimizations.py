#!/usr/bin/env python3
"""
Test FLUX.1 Krea Performance Optimizations
Quick validation of speed improvements
"""

import torch
import time
import os
from diffusers import FluxPipeline

# Set optimal environment for testing
os.environ['PYTORCH_MPS_MEMORY_FRACTION'] = '0.85'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable for testing
os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.6'
os.environ['KREA_ENABLE_METAL_VAE'] = '1'

def test_optimized_performance():
    """Test optimized pipeline performance"""
    print("🧪 Testing Optimized FLUX.1 Krea Performance")
    print("=" * 50)
    
    # Load optimized pipeline  
    print("📥 Loading optimized pipeline...")
    start_time = time.time()
    
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Krea-dev",
        torch_dtype=torch.float16,  # fp16 for speed
        use_safetensors=True,
        low_cpu_mem_usage=True
    )
    
    # Move to MPS
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    # Apply optimizations
    optimizations = []
    
    try:
        pipeline.enable_model_cpu_offload()
        optimizations.append("CPU offload")
    except:
        optimizations.append("Default memory")
    
    try:
        pipeline.enable_attention_slicing("auto")
        optimizations.append("Attention slicing")
    except:
        pass
    
    try:
        pipeline.enable_vae_slicing()
        optimizations.append("VAE slicing")
    except:
        pass
        
    try:
        pipeline.enable_vae_tiling()  
        optimizations.append("VAE tiling")
    except:
        pass
    
    # Try Metal VAE optimization
    try:
        from flux_metal_kernels import M4ProMetalOptimizer
        metal_optimizer = M4ProMetalOptimizer()
        pipeline = metal_optimizer.optimize_flux_pipeline_components(pipeline)
        optimizations.append("Metal VAE")
    except:
        pass
    
    load_time = time.time() - start_time
    print(f"✅ Pipeline loaded in {load_time:.1f}s")
    
    print("🔧 Optimizations applied:")
    for opt in optimizations:
        print(f"   ✅ {opt}")
    
    # Test generation - Quick settings (512x512, 20 steps)
    print(f"\n🖼️  Testing generation (512x512, 20 steps)...")
    prompt = "a cute cat sitting on a rainbow"
    
    gen_start = time.time()
    
    with torch.inference_mode():
        result = pipeline(
            prompt,
            height=512,
            width=512, 
            guidance_scale=4.5,
            num_inference_steps=20,
            max_sequence_length=256
        )
    
    gen_time = time.time() - gen_start
    
    # Save result
    output_path = "performance_test_optimized.png"
    result.images[0].save(output_path)
    
    print(f"✅ Generation completed!")
    print(f"⏱️  Generation time: {gen_time:.1f} seconds")
    print(f"🚀 Speed: {(512 * 512) / gen_time:.0f} pixels/second") 
    print(f"💾 Saved: {output_path}")
    
    # Performance targets
    print(f"\n📊 Performance Analysis:")
    if gen_time < 30:
        print(f"   🎉 EXCELLENT: {gen_time:.1f}s (target: <30s)")
    elif gen_time < 60:
        print(f"   ✅ GOOD: {gen_time:.1f}s (target: <60s)")
    else:
        print(f"   ⚠️  SLOW: {gen_time:.1f}s (expected: <60s)")
        print(f"   💡 Check optimizations are working properly")
    
    # Memory stats
    if device == "mps":
        try:
            allocated = torch.mps.current_allocated_memory() / (1024**3)
            print(f"   💾 MPS Memory: {allocated:.1f} GB")
        except:
            pass
    
    return gen_time

if __name__ == "__main__":
    try:
        test_time = test_optimized_performance()
        print(f"\n🎯 Performance test completed in {test_time:.1f} seconds")
        
        if test_time < 30:
            print("🚀 Ready for production use!")
        else:
            print("💡 Consider further optimizations")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check HF_TOKEN is set")
        print("2. Verify model access permissions")  
        print("3. Run: ./auth_validation.sh")