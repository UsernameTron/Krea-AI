#!/usr/bin/env python3
"""
FLUX.1 Krea Image Generation optimized for Apple Silicon
Using local models with compilation fixes
"""

import torch
import click
import os
import time

# Disable problematic torch compilation features for Apple Silicon
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

# Import after setting environment variables
from src.flux.util import load_ae, load_flow_model, load_clip, load_t5
from src.flux.pipeline import Sampler
from PIL import Image

@click.command()
@click.option("--prompt", required=True, help="Generation prompt")
@click.option("--output", default="flux_apple_output.png", help="Output filename")
@click.option("--width", default=1024, help="Image width")
@click.option("--height", default=1024, help="Image height")
@click.option("--num_steps", default=28, help="Number of inference steps")
@click.option("--guidance", default=4.5, help="Guidance scale")
@click.option("--seed", default=None, type=int, help="Random seed")
def generate(prompt, output, width, height, num_steps, guidance, seed):
    """Generate images using FLUX.1 Krea optimized for Apple Silicon"""
    
    print("🍎 FLUX.1 Krea Generation for Apple Silicon")
    print("=" * 50)
    
    # Determine best device for Apple Silicon
    if torch.backends.mps.is_available():
        device = "mps"
        print("📱 Using Apple Metal Performance Shaders (MPS)")
    else:
        device = "cpu"
        print("💻 Using CPU")
    
    torch_dtype = torch.bfloat16
    
    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        print(f"🎲 Seed: {seed}")
    
    try:
        print("⏳ Loading models from local files...")
        start_time = time.time()
        
        # Load models with Apple Silicon optimizations
        print("📥 Loading autoencoder...")
        ae = load_ae("flux-krea-dev", device=device)
        
        print("📥 Loading CLIP...")
        clip = load_clip(device=device)
        
        print("📥 Loading T5...")
        t5 = load_t5(device=device)
        
        print("📥 Loading FLUX model...")
        model = load_flow_model("flux-krea-dev", device=device)
        
        # Move models to device with proper dtype
        ae = ae.to(device=device, dtype=torch_dtype)
        model = model.to(device=device, dtype=torch_dtype)
        
        load_time = time.time() - start_time
        print(f"✅ Models loaded in {load_time:.1f}s")
        
        # Create sampler with Apple Silicon optimizations
        sampler = Sampler(clip=clip, t5=t5, ae=ae, model=model, device=device, dtype=torch_dtype)
        
        print(f"🎨 Generating: '{prompt}'")
        print(f"   Size: {width}x{height}")
        print(f"   Steps: {num_steps}, Guidance: {guidance}")
        
        # Generate image with memory management for Apple Silicon
        start_time = time.time()
        
        # Use inference mode and memory management
        with torch.inference_mode():
            if device == "mps":
                # Enable memory efficient operations for MPS
                torch.mps.empty_cache()
            
            image = sampler(
                prompt=prompt,
                width=width,
                height=height,
                guidance=guidance,
                num_steps=num_steps,
                seed=seed,
            )
            
            if device == "mps":
                torch.mps.empty_cache()
        
        generation_time = time.time() - start_time
        print(f"⚡ Generated in {generation_time:.1f}s")
        
        # Save image
        image.save(output)
        print(f"💾 Saved to: {output}")
        
        # Display image info
        print(f"📏 Image size: {image.size}")
        print(f"🎯 Total time: {load_time + generation_time:.1f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    generate()