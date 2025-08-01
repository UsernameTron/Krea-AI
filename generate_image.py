#!/usr/bin/env python3
"""
Optimized FLUX.1 Krea image generation for Apple Silicon
This script is configured specifically for M4 Pro performance characteristics
"""

import argparse
import torch
import time
from diffusers import FluxPipeline
from pathlib import Path
import os

def main():
    parser = argparse.ArgumentParser(description='Generate high-quality images with FLUX.1 Krea')
    parser.add_argument('--prompt', required=True, help='Detailed text description of the image you want')
    parser.add_argument('--output', default='flux_generation.png', help='Name for the output image file')
    parser.add_argument('--steps', type=int, default=28, help='Number of denoising steps (28-32 recommended)')
    parser.add_argument('--guidance', type=float, default=4.0, help='How closely to follow the prompt (3.5-5.0 range)')
    parser.add_argument('--width', type=int, default=1024, help='Image width in pixels')
    parser.add_argument('--height', type=int, default=1024, help='Image height in pixels')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    print("🎨 Loading FLUX.1 Krea model...")
    print("   This takes a moment as we load 12 billion parameters into memory")
    
    start_time = time.time()
    
    # Check if we have a local model directory
    local_model_path = "./models/FLUX.1-Krea-dev"
    if os.path.exists(local_model_path) and os.path.isdir(local_model_path):
        print(f"   Using local model from: {local_model_path}")
        model_id = local_model_path
        local_files_only = True
    else:
        print("   Downloading model from Hugging Face Hub...")
        model_id = "black-forest-labs/FLUX.1-Krea-dev"
        local_files_only = False
    
    try:
        # Initialize the pipeline with optimizations for Apple Silicon
        pipeline = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,   # Efficient precision for M4 Pro
            local_files_only=local_files_only
        )
        
        # Enable CPU offloading to manage memory efficiently on your 48GB system
        pipeline.enable_model_cpu_offload()
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f} seconds")
        
    except Exception as e:
        if "gated repo" in str(e).lower() or "access to model" in str(e).lower():
            print("❌ Model access required!")
            print("   Please visit: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev")
            print("   and request access to the repository.")
            print("   After approval, run this script again.")
            return
        else:
            print(f"❌ Error loading model: {e}")
            return
    
    print(f"🖼️  Generating: '{args.prompt}'")
    print(f"   Resolution: {args.width}x{args.height}")
    print(f"   Steps: {args.steps}, Guidance: {args.guidance}")
    
    generation_start = time.time()
    
    try:
        # Generate the image with your specified parameters
        result = pipeline(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=torch.Generator().manual_seed(args.seed) if args.seed else None
        )
        
        generation_time = time.time() - generation_start
        
        # Save the generated image
        image = result.images[0]
        output_path = Path(args.output)
        image.save(output_path)
        
        print(f"🎉 Generation complete in {generation_time:.1f} seconds")
        print(f"💾 Image saved as: {output_path.absolute()}")
        
        # Display some helpful information about the generation
        if args.seed:
            print(f"🎲 Used seed: {args.seed} (use this seed again for identical results)")
            
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        print("   This might be due to memory constraints or model access issues.")

if __name__ == "__main__":
    main()