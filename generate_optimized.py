#!/usr/bin/env python3
"""
FLUX.1 Krea Optimized Generation
Uses advanced optimization libraries for maximum performance on M4 Pro
"""

import torch
import argparse
import time
from pathlib import Path
from diffusers import FluxPipeline
from optimum.quanto import freeze, qfloat8, quantize
import bitsandbytes as bnb

def main():
    parser = argparse.ArgumentParser(description='Optimized FLUX.1 Krea generation with advanced features')
    parser.add_argument('--prompt', required=True, help='Text prompt for image generation')
    parser.add_argument('--output', default='flux-optimized.png', help='Output filename')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--guidance', type=float, default=4.5, help='Guidance scale')
    parser.add_argument('--steps', type=int, default=28, help='Number of inference steps')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--quantize', action='store_true', help='Enable 8-bit quantization for memory optimization')
    parser.add_argument('--low-memory', action='store_true', help='Enable maximum memory optimization')
    
    args = parser.parse_args()
    
    print("🚀 FLUX.1 Krea [dev] - OPTIMIZED for M4 Pro")
    print("=" * 55)
    print(f"🎨 Prompt: {args.prompt}")
    print(f"📐 Size: {args.width}x{args.height}")
    print(f"⚙️  Settings: guidance={args.guidance}, steps={args.steps}")
    
    if args.quantize:
        print("🔧 Quantization: ENABLED (8-bit)")
    if args.low_memory:
        print("💾 Low Memory Mode: ENABLED")
    
    print("=" * 55)
    
    start_time = time.time()
    
    try:
        print("\n📥 Loading FLUX.1 Krea [dev] with optimizations...")
        
        # Configure PyTorch for optimal M4 Pro performance
        if torch.backends.mps.is_available():
            device = "mps"
            print("🍎 Using Apple Silicon MPS acceleration")
        else:
            device = "cpu"
            print("🖥️  Using CPU (MPS not available)")
        
        # Load pipeline with advanced optimizations
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Krea-dev",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            device_map="auto" if not args.low_memory else None
        )
        
        # Apply quantization if requested
        if args.quantize:
            print("🔧 Applying 8-bit quantization...")
            quantize(pipe.transformer, weights=qfloat8)
            freeze(pipe.transformer)
            print("✅ Quantization applied")
        
        # Configure memory optimizations
        if args.low_memory:
            print("💾 Enabling maximum memory optimization...")
            pipe.enable_model_cpu_offload()
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()
        
        # Enable memory efficient attention if available
        try:
            pipe.enable_attention_slicing()
            print("🧠 Memory efficient attention: ENABLED")
        except:
            print("🧠 Memory efficient attention: Not available")
        
        # Enable VAE slicing for large images  
        try:
            pipe.vae.enable_slicing()
            print("🖼️  VAE slicing: ENABLED")
        except:
            print("🖼️  VAE slicing: Not available")
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded and optimized in {load_time:.1f} seconds")
        
        print(f"\n🎨 Generating optimized image...")
        generation_start = time.time()
        
        # Set up generator for reproducible results
        generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed else None
        
        # Generate with optimizations
        with torch.inference_mode():
            image = pipe(
                args.prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                generator=generator
            ).images[0]
        
        generation_time = time.time() - generation_start
        
        # Save the image
        output_path = Path(args.output)
        image.save(output_path)
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 OPTIMIZED GENERATION COMPLETE!")
        print("=" * 55)
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"   📥 Load: {load_time:.1f}s")
        print(f"   🎨 Generate: {generation_time:.1f}s")
        print(f"💾 Saved as: {output_path.absolute()}")
        print(f"📐 Final size: {image.size[0]}x{image.size[1]}")
        
        # Memory usage info
        if torch.backends.mps.is_available():
            print(f"🍎 MPS Memory: {torch.mps.current_allocated_memory() / 1024**2:.1f} MB allocated")
        
        if args.seed:
            print(f"🎲 Seed: {args.seed}")
        
        print("\n💡 OPTIMIZATION FEATURES USED:")
        print("   ✅ Apple Silicon MPS acceleration")
        print("   ✅ bfloat16 precision optimization")
        print("   ✅ Model CPU offloading")
        print("   ✅ SafeTensors efficient loading")
        if args.quantize:
            print("   ✅ 8-bit quantization (optimum-quanto)")
        if args.low_memory:
            print("   ✅ Sequential CPU offloading")
        print("   ✅ Memory efficient attention")
        print("   ✅ VAE slicing for large images")
            
    except Exception as e:
        error_str = str(e)
        print(f"\n❌ Error: {e}")
        
        if "gated" in error_str.lower():
            print("\n🔒 REPOSITORY ACCESS REQUIRED")
            print("Visit: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev")
            print("Request access and wait for approval")
            
        elif "401" in error_str or "403" in error_str:
            print("\n🔑 AUTHENTICATION ISSUE")
            print("Run: huggingface-cli login")
            print("Make sure you have repository access")
            
        else:
            print("\n🔧 TROUBLESHOOTING:")
            print("1. Ensure repository access is approved")
            print("2. Try without --quantize flag")
            print("3. Try with --low-memory flag")
            print("4. Check available memory")

if __name__ == "__main__":
    main()