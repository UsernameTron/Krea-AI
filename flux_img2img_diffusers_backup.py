#!/usr/bin/env python3
"""
FLUX.1 Krea Image-to-Image transformation with Apple Silicon optimizations
Based on FluxImg2ImgPipeline from the official documentation
"""

import argparse
import torch
from PIL import Image
from diffusers import FluxImg2ImgPipeline
from diffusers.utils import load_image

def main():
    parser = argparse.ArgumentParser(description='FLUX Image-to-Image Generation')
    parser.add_argument('--prompt', required=True, help='Transformation prompt')
    parser.add_argument('--input_image', required=True, help='Input image path or URL')
    parser.add_argument('--output', default='flux_img2img.png', help='Output filename')
    parser.add_argument('--strength', type=float, default=0.6, help='Transformation strength (0-1)')
    parser.add_argument('--steps', type=int, default=28, help='Inference steps')
    parser.add_argument('--guidance', type=float, default=7.0, help='Guidance scale')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    args = parser.parse_args()
    
    print("🔄 Loading FLUX Image-to-Image Pipeline...")
    
    # Load the specialized img2img pipeline with optimizations
    pipeline = FluxImg2ImgPipeline.from_pretrained(
        "./models/FLUX.1-Krea-dev",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    
    # Apply the memory optimizations documented for Apple Silicon
    pipeline.enable_model_cpu_offload()
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    
    print("📷 Loading input image...")
    
    # Load input image (supports URLs and local files as per documentation)
    if args.input_image.startswith(('http://', 'https://')):
        input_image = load_image(args.input_image)
    else:
        input_image = Image.open(args.input_image)
    
    # Ensure image is in RGB mode
    input_image = input_image.convert('RGB')
    
    print(f"🎨 Transforming image with prompt: '{args.prompt}'")
    print(f"   Strength: {args.strength}, Steps: {args.steps}, Guidance: {args.guidance}")
    
    # Generate with the documented parameters
    result = pipeline(
        prompt=args.prompt,
        image=input_image,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=torch.Generator().manual_seed(args.seed) if args.seed else None
    )
    
    # Save the result
    result.images[0].save(args.output)
    print(f"✅ Transformed image saved as: {args.output}")

if __name__ == "__main__":
    main()