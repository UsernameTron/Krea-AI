#\!/usr/bin/env python3
"""
FLUX.1 Krea Inpainting with advanced mask handling
Implements FluxInpaintPipeline with the documented optimizations
"""

import argparse
import torch
from PIL import Image, ImageOps
from diffusers import FluxInpaintPipeline
from diffusers.utils import load_image
import numpy as np

def prepare_mask(mask_image):
    """Prepare mask image according to FLUX requirements"""
    # Convert to grayscale if needed
    if mask_image.mode != 'L':
        mask_image = mask_image.convert('L')
    
    # Ensure mask is binary (white areas will be inpainted)
    mask_array = np.array(mask_image)
    mask_array = (mask_array > 128).astype(np.uint8) * 255
    
    return Image.fromarray(mask_array, mode='L')

def main():
    parser = argparse.ArgumentParser(description='FLUX Inpainting')
    parser.add_argument('--prompt', required=True, help='Inpainting prompt')
    parser.add_argument('--image', required=True, help='Input image path or URL')
    parser.add_argument('--mask', required=True, help='Mask image path or URL')
    parser.add_argument('--output', default='flux_inpaint.png', help='Output filename')
    parser.add_argument('--strength', type=float, default=0.8, help='Inpainting strength')
    parser.add_argument('--steps', type=int, default=28, help='Inference steps')
    parser.add_argument('--guidance', type=float, default=7.0, help='Guidance scale')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    args = parser.parse_args()
    
    print("🎭 Loading FLUX Inpainting Pipeline...")
    
    # Load inpainting pipeline with Apple Silicon optimizations
    pipeline = FluxInpaintPipeline.from_pretrained(
        "./models/FLUX.1-Krea-dev",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    
    # Apply comprehensive optimizations as documented
    pipeline.enable_model_cpu_offload()
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    
    print("📷 Loading images...")
    
    # Load and prepare images
    if args.image.startswith(('http://', 'https://')):
        source_image = load_image(args.image)
    else:
        source_image = Image.open(args.image)
    
    if args.mask.startswith(('http://', 'https://')):
        mask_image = load_image(args.mask)
    else:
        mask_image = Image.open(args.mask)
    
    # Ensure images are properly formatted
    source_image = source_image.convert('RGB')
    mask_image = prepare_mask(mask_image)
    
    # Resize mask to match source image if needed
    if mask_image.size != source_image.size:
        mask_image = mask_image.resize(source_image.size, Image.LANCZOS)
    
    print(f"🎨 Inpainting with prompt: '{args.prompt}'")
    print(f"   Image size: {source_image.size}")
    print(f"   Strength: {args.strength}, Steps: {args.steps}")
    
    # Perform inpainting with documented parameters
    result = pipeline(
        prompt=args.prompt,
        image=source_image,
        mask_image=mask_image,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=torch.Generator().manual_seed(args.seed) if args.seed else None
    )
    
    # Save result
    result.images[0].save(args.output)
    print(f"✅ Inpainted image saved as: {args.output}")

if __name__ == "__main__":
    main()