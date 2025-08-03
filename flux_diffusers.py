#!/usr/bin/env python3
"""
FLUX.1 Krea Image Generation using Diffusers Pipeline
Optimized for Apple Silicon with local model support
"""

import torch
import click
from diffusers import FluxPipeline
from PIL import Image
import os
import time

def get_local_model_path():
    """Get the model path - use HuggingFace with local caching"""
    return "black-forest-labs/FLUX.1-schnell"

@click.command()
@click.option("--prompt", required=True, help="Generation prompt")
@click.option("--output", default="flux_diffusers_output.png", help="Output filename")
@click.option("--width", default=1024, help="Image width")
@click.option("--height", default=1024, help="Image height")
@click.option("--num_steps", default=28, help="Number of inference steps")
@click.option("--guidance", default=4.5, help="Guidance scale")
@click.option("--seed", default=None, type=int, help="Random seed")
@click.option("--device", default="auto", help="Device to use (auto, cpu, mps, cuda)")
def generate(prompt, output, width, height, num_steps, guidance, seed, device):
    """Generate images using FLUX.1 Krea with Diffusers pipeline"""
    
    print("🚀 FLUX.1 Krea Generation with Diffusers Pipeline")
    print("=" * 50)
    
    # Determine device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    print(f"📱 Using device: {device}")
    
    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        print(f"🎲 Seed: {seed}")
    
    # Get model path (local or HuggingFace)
    model_path = get_local_model_path()
    if model_path.startswith("./"):
        print(f"📁 Using local models: {model_path}")
    else:
        print(f"🌐 Using HuggingFace models: {model_path}")
    
    try:
        print("⏳ Loading FLUX pipeline...")
        start_time = time.time()
        
        # Load pipeline with optimizations for Apple Silicon
        pipeline = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="balanced" if device != "cpu" else None
        )
        
        # Apply Apple Silicon optimizations
        if device == "mps":
            pipeline = pipeline.to("mps")
            # Enable memory efficient attention
            pipeline.enable_attention_slicing()
            pipeline.enable_vae_slicing()
        elif device == "cpu":
            pipeline = pipeline.to("cpu")
            # CPU optimizations
            pipeline.enable_attention_slicing()
            pipeline.enable_vae_slicing()
        else:
            pipeline = pipeline.to(device)
        
        load_time = time.time() - start_time
        print(f"✅ Pipeline loaded in {load_time:.1f}s")
        
        print(f"🎨 Generating: '{prompt}'")
        print(f"   Size: {width}x{height}")
        print(f"   Steps: {num_steps}, Guidance: {guidance}")
        
        # Generate image
        start_time = time.time()
        
        with torch.inference_mode():
            image = pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                generator=torch.Generator(device=device).manual_seed(seed) if seed else None
            ).images[0]
        
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