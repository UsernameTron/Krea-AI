#!/usr/bin/env python3
"""
FLUX.1 Krea [dev] - Official Diffusers Implementation
Based on the official HuggingFace documentation
"""

import torch
import argparse
import time
from pathlib import Path
from diffusers import FluxPipeline

def main():
    parser = argparse.ArgumentParser(description='FLUX.1 Krea [dev] Official Generation')
    parser.add_argument('--prompt', required=True, help='Text prompt for image generation')
    parser.add_argument('--output', default='flux-krea-official.png', help='Output filename')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--guidance', type=float, default=4.5, help='Guidance scale (3.5-5.0 recommended)')
    parser.add_argument('--steps', type=int, default=28, help='Number of inference steps (28-32 recommended)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    print("🎨 FLUX.1 Krea [dev] - Official Implementation")
    print("=" * 50)
    print(f"Prompt: {args.prompt}")
    print(f"Size: {args.width}x{args.height}")
    print(f"Guidance: {args.guidance}, Steps: {args.steps}")
    
    start_time = time.time()
    
    try:
        print("\n📥 Loading FLUX.1 Krea [dev] from HuggingFace...")
        
        # Official implementation from HuggingFace docs
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Krea-dev", 
            torch_dtype=torch.bfloat16
        )
        
        # Enable memory optimizations for Apple Silicon
        try:
            pipe.enable_model_cpu_offload()
        except Exception as e:
            print(f"Note: CPU offload not available: {e}")
            # Alternative: use sequential CPU offload
            try:
                pipe.enable_sequential_cpu_offload()
            except:
                print("Using default memory management")
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f} seconds")
        
        print(f"\n🖼️  Generating image...")
        generation_start = time.time()
        
        # Set up generator for reproducible results if seed provided
        generator = torch.Generator().manual_seed(args.seed) if args.seed else None
        
        # Generate image using official parameters
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
        
        print(f"🎉 Generation complete!")
        print(f"⏱️  Total time: {total_time:.1f}s (Load: {load_time:.1f}s + Gen: {generation_time:.1f}s)")
        print(f"💾 Saved as: {output_path.absolute()}")
        print(f"📐 Size: {image.size[0]}x{image.size[1]}")
        
        if args.seed:
            print(f"🎲 Seed used: {args.seed}")
            
    except Exception as e:
        error_str = str(e)
        print(f"\n❌ Error: {e}")
        
        if "gated" in error_str.lower() or "access" in error_str.lower():
            print("\n🔒 REPOSITORY ACCESS REQUIRED")
            print("1. Visit: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev")
            print("2. Click 'Request access to this repository'")
            print("3. Accept the license agreement")
            print("4. Wait for approval (usually within hours)")
            
        elif "401" in error_str or "403" in error_str:
            print("\n🔑 AUTHENTICATION ISSUE")
            print("1. Run: huggingface-cli login")
            print("2. Enter your HuggingFace token")
            print("3. Make sure you've requested access to the model")
            
        else:
            print("\n🔧 TROUBLESHOOTING STEPS")
            print("1. Update diffusers: pip install -U diffusers")
            print("2. Check internet connection")
            print("3. Clear cache: rm -rf ~/.cache/huggingface")
            print("4. Restart and try again")

if __name__ == "__main__":
    main()