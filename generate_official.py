#!/usr/bin/env python3
"""
FLUX.1 Krea generation using the official method from the documentation
Based on: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev
"""

import torch
import argparse
import time
from pathlib import Path
from diffusers import FluxPipeline

def main():
    parser = argparse.ArgumentParser(description='Generate images with FLUX.1 Krea using official method')
    parser.add_argument('--prompt', required=True, help='Text prompt for image generation')
    parser.add_argument('--output', default='flux-krea-official.png', help='Output filename')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--guidance', type=float, default=4.5, help='Guidance scale')
    parser.add_argument('--steps', type=int, default=28, help='Number of inference steps')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    args = parser.parse_args()
    
    print("🎨 FLUX.1 Krea [dev] - Official Implementation")
    print("=" * 50)
    print(f"Prompt: {args.prompt}")
    print(f"Size: {args.width}x{args.height}")
    print(f"Guidance: {args.guidance}, Steps: {args.steps}")
    
    start_time = time.time()
    
    try:
        print("\n📥 Loading FLUX.1 Krea [dev] model...")
        
        # Official implementation from HuggingFace docs
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Krea-dev", 
            torch_dtype=torch.bfloat16
        )
        
        # Enable CPU offloading to save VRAM/RAM
        pipe.enable_model_cpu_offload()
        
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
        
        if "gated" in error_str.lower():
            print("\n🔒 REPOSITORY ACCESS REQUIRED")
            print("1. Visit: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev")
            print("2. Click 'Request access to this repository'")
            print("3. Accept the license agreement")
            print("4. Wait for approval (usually within hours)")
            
        elif "401" in error_str or "403" in error_str:
            print("\n🔑 AUTHENTICATION ISSUE")
            print("1. Check your HuggingFace token is valid")
            print("2. Run: huggingface-cli login")
            print("3. Make sure you've requested access to the model")
            
        elif "connection" in error_str.lower() or "network" in error_str.lower():
            print("\n🌐 NETWORK ISSUE")
            print("1. Check your internet connection")
            print("2. Try again in a few minutes")
            print("3. Consider using a VPN if in a restricted region")
            
        else:
            print("\n🔧 TROUBLESHOOTING STEPS")
            print("1. Clear cache: rm -rf ~/.cache/huggingface")
            print("2. Update diffusers: pip install -U diffusers")
            print("3. Check available disk space")
            print("4. Restart Python and try again")

if __name__ == "__main__":
    main()