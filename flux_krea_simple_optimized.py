#!/usr/bin/env python3
"""
FLUX.1 Krea [dev] - Simple Apple Silicon M4 Pro Optimized Implementation
"""

import torch
import argparse
import time
import gc
import os
from pathlib import Path
from diffusers import FluxPipeline
import psutil

def configure_apple_silicon():
    """Configure optimizations for Apple Silicon"""
    # Set optimal thread counts for M4 Pro (8 performance cores)
    torch.set_num_threads(8)
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    
    # Disable unnecessary optimizations that may cause issues
    os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    print("🍎 Apple Silicon M4 Pro optimizations configured")

def main():
    parser = argparse.ArgumentParser(description='FLUX.1 Krea Apple Silicon Optimized')
    parser.add_argument('--prompt', required=True, help='Text prompt for image generation')
    parser.add_argument('--output', default='simple-optimized.png', help='Output filename')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--guidance', type=float, default=4.5, help='Guidance scale')
    parser.add_argument('--steps', type=int, default=28, help='Number of inference steps')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    args = parser.parse_args()
    
    print("🍎 FLUX.1 Krea [dev] - Apple Silicon M4 Pro Simple Optimized")
    print("=" * 62)
    print(f"Cores: 8 Performance cores optimized")
    print(f"Memory: {psutil.virtual_memory().total // (1024**3)} GB unified memory")
    print(f"Prompt: {args.prompt}")
    
    # Configure Apple Silicon optimizations
    configure_apple_silicon()
    
    start_time = time.time()
    
    try:
        print("\n📥 Loading FLUX.1 Krea [dev] with Apple Silicon optimizations...")
        
        # Load pipeline with Apple Silicon friendly settings
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Krea-dev",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        
        # Enable memory optimizations
        if hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing("auto")
            print("✅ Attention slicing enabled")
        
        if hasattr(pipeline, 'enable_vae_tiling'):
            pipeline.enable_vae_tiling()
            print("✅ VAE tiling enabled")
        
        if hasattr(pipeline, 'enable_vae_slicing'):
            pipeline.enable_vae_slicing()
            print("✅ VAE slicing enabled")
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f} seconds")
        
        print(f"\n🖼️  Generating {args.width}x{args.height} image...")
        generation_start = time.time()
        
        # Set up generator for reproducible results
        generator = None
        if args.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(args.seed)
        
        # Generate with optimized inference mode
        with torch.inference_mode():
            result = pipeline(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                generator=generator,
                return_dict=True
            )
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        # Save image
        output_path = Path(args.output)
        result.images[0].save(output_path)
        
        print(f"🎉 Generation complete!")
        print(f"⏱️  Load time: {load_time:.1f}s | Generation: {generation_time:.1f}s | Total: {total_time:.1f}s")
        print(f"💾 Saved: {output_path.absolute()}")
        
        # Memory cleanup
        del pipeline
        gc.collect()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Apple Silicon Troubleshooting:")
        print("1. Check available memory with Activity Monitor")
        print("2. Try reducing image size: --width 768 --height 768")
        print("3. Try reducing steps: --steps 20")

if __name__ == "__main__":
    main()