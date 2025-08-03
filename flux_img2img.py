#!/usr/bin/env python3
"""
Working FLUX Image-to-Image transformation
"""

import argparse
import torch
from PIL import Image
from diffusers import FluxImg2ImgPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--output', default='img2img_result.png')
    parser.add_argument('--strength', type=float, default=0.6)
    parser.add_argument('--steps', type=int, default=28)
    
    args = parser.parse_args()
    
    print("🔄 Loading Image-to-Image pipeline...")
    
    pipeline = FluxImg2ImgPipeline.from_pretrained(
        "./models/FLUX.1-Krea-dev",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    
    pipeline.enable_model_cpu_offload()
    pipeline.vae.enable_slicing()
    
    input_image = Image.open(args.input).convert('RGB')
    
    result = pipeline(
        prompt=args.prompt,
        image=input_image,
        strength=args.strength,
        num_inference_steps=args.steps
    )
    
    result.images[0].save(args.output)
    print(f"✅ Saved: {args.output}")

if __name__ == "__main__":
    main()