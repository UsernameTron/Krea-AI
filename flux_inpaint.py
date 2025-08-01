#!/usr/bin/env python3
"""
FLUX.1 Krea Inpainting with Apple Silicon optimizations
Advanced inpainting capabilities for precise image editing
"""

import argparse
import torch
from PIL import Image
from diffusers import FluxInpaintPipeline
from diffusers.utils import load_image
import numpy as np

def create_mask_from_bbox(image_size, bbox):
    """Create a mask from bounding box coordinates (x1, y1, x2, y2)"""
    width, height = image_size
    mask = Image.new('L', (width, height), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bbox, fill=255)
    return mask

def main():
    parser = argparse.ArgumentParser(description='FLUX Inpainting Generation')
    parser.add_argument('--prompt', required=True, help='Inpainting prompt')
    parser.add_argument('--input_image', required=True, help='Input image path or URL')
    parser.add_argument('--mask_image', help='Mask image path (white = inpaint area)')
    parser.add_argument('--bbox', help='Bounding box for mask "x1,y1,x2,y2"')
    parser.add_argument('--output', default='flux_inpaint.png', help='Output filename')
    parser.add_argument('--strength', type=float, default=0.8, help='Inpainting strength (0-1)')
    parser.add_argument('--steps', type=int, default=32, help='Inference steps')
    parser.add_argument('--guidance', type=float, default=7.5, help='Guidance scale')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    args = parser.parse_args()
    
    # Validate mask input
    if not args.mask_image and not args.bbox:
        print("❌ Error: Must provide either --mask_image or --bbox")
        return
    
    print("🎨 Loading FLUX Inpainting Pipeline...")
    
    try:
        # Load the specialized inpainting pipeline
        pipeline = FluxInpaintPipeline.from_pretrained(
            "./models/FLUX.1-Krea-dev",
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )
        
        # Apply Apple Silicon optimizations
        pipeline.enable_model_cpu_offload()
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
        
    except Exception as e:
        print(f"⚠️  Inpainting pipeline not available: {e}")
        print("💡 Using standard pipeline with mask guidance")
        from diffusers import FluxPipeline
        pipeline = FluxPipeline.from_pretrained(
            "./models/FLUX.1-Krea-dev",
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )
        pipeline.enable_model_cpu_offload()
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
    
    print("📷 Loading input image...")
    
    # Load input image
    if args.input_image.startswith(('http://', 'https://')):
        input_image = load_image(args.input_image)
    else:
        input_image = Image.open(args.input_image)
    
    input_image = input_image.convert('RGB')
    
    # Load or create mask
    if args.mask_image:
        print("🎭 Loading mask image...")
        if args.mask_image.startswith(('http://', 'https://')):
            mask_image = load_image(args.mask_image)
        else:
            mask_image = Image.open(args.mask_image)
        mask_image = mask_image.convert('L')
    elif args.bbox:
        print("🎭 Creating mask from bounding box...")
        bbox_coords = [int(x) for x in args.bbox.split(',')]
        if len(bbox_coords) != 4:
            print("❌ Error: Bounding box must be 'x1,y1,x2,y2'")
            return
        mask_image = create_mask_from_bbox(input_image.size, bbox_coords)
    
    print(f"🖌️  Inpainting with prompt: '{args.prompt}'")
    print(f"   Strength: {args.strength}, Steps: {args.steps}, Guidance: {args.guidance}")
    
    # Generate with inpainting
    try:
        result = pipeline(
            prompt=args.prompt,
            image=input_image,
            mask_image=mask_image,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=torch.Generator().manual_seed(args.seed) if args.seed else None
        )
    except TypeError:
        # Fallback for standard pipeline
        print("💡 Using img2img approach with masked guidance")
        from diffusers import FluxImg2ImgPipeline
        pipeline = FluxImg2ImgPipeline.from_pretrained(
            "./models/FLUX.1-Krea-dev",
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )
        pipeline.enable_model_cpu_offload()
        pipeline.vae.enable_slicing()
        
        # Create masked input
        masked_input = input_image.copy()
        masked_array = np.array(masked_input)
        mask_array = np.array(mask_image)
        
        # Apply mask (set masked areas to noise/gray)
        mask_bool = mask_array > 128
        masked_array[mask_bool] = 128
        masked_input = Image.fromarray(masked_array)
        
        result = pipeline(
            prompt=args.prompt,
            image=masked_input,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=torch.Generator().manual_seed(args.seed) if args.seed else None
        )
    
    # Save the result
    result.images[0].save(args.output)
    print(f"✅ Inpainted image saved as: {args.output}")

if __name__ == "__main__":
    main()