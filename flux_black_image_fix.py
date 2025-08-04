#!/usr/bin/env python3
'''
FLUX.1 Krea Black Image Fix
Use this version if you're getting black images
'''

import torch
import os
from diffusers import FluxPipeline
import argparse

# Environment fixes for black images
os.environ['PYTORCH_MPS_MEMORY_FRACTION'] = '0.8'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def main():
    parser = argparse.ArgumentParser(description='FLUX.1 Krea - Black Image Fix')
    parser.add_argument('--prompt', required=True, help='Text prompt')
    parser.add_argument('--output', default='fixed_output.png', help='Output filename')
    parser.add_argument('--width', type=int, default=512, help='Width (512 recommended)')
    parser.add_argument('--height', type=int, default=512, help='Height (512 recommended)')
    parser.add_argument('--steps', type=int, default=20, help='Steps (20 recommended)')
    parser.add_argument('--guidance', type=float, default=4.0, help='Guidance scale')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    args = parser.parse_args()
    
    print("🔧 FLUX.1 Krea - Black Image Fix Mode")
    print("=" * 40)
    
    # Clear cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    print("📥 Loading pipeline with black image fixes...")
    
    # Use bfloat16 for better MPS stability
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Krea-dev",
        torch_dtype=torch.bfloat16,  # Key fix: bfloat16 instead of float16
        use_safetensors=True,
        low_cpu_mem_usage=True
    )
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    # Conservative optimizations to prevent black images
    try:
        pipeline.enable_model_cpu_offload()
        print("✅ CPU offload enabled")
    except:
        pass
    
    try:
        pipeline.enable_attention_slicing(1)  # Minimal slicing
        print("✅ Conservative attention slicing enabled")
    except:
        pass
    
    # Skip VAE optimizations that can cause black images
    print("ℹ️  Using conservative VAE settings")
    
    print(f"\n🖼️  Generating: {args.prompt}")
    print(f"📐 Size: {args.width}x{args.height}")
    print(f"🎯 Steps: {args.steps}, Guidance: {args.guidance}")
    
    # Clear cache before generation
    if device == "mps":
        torch.mps.empty_cache()
    
    # Generate with conservative settings
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed else None
    
    with torch.inference_mode():
        result = pipeline(
            args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            generator=generator,
            max_sequence_length=128
        )
    
    # Save result
    result.images[0].save(args.output)
    print(f"✅ Image saved: {args.output}")
    
    # Verify image is not black
    import numpy as np
    from PIL import Image
    
    img = Image.open(args.output)
    img_array = np.array(img)
    mean_brightness = np.mean(img_array)
    
    print(f"📊 Image brightness: {mean_brightness:.1f}/255")
    
    if mean_brightness < 10:
        print("⚠️  Warning: Image appears to be black/very dark")
        print("💡 Try: Increase guidance scale or reduce image size")
    else:
        print("✅ Image generated successfully!")

if __name__ == "__main__":
    main()
