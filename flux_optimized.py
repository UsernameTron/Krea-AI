#!/usr/bin/env python3
"""
Optimized FLUX.1 Krea generation for Apple Silicon M4 Pro
Uses working optimization libraries only
"""

import argparse
import torch
import time
import psutil
from pathlib import Path
from diffusers import FluxPipeline

class OptimizedFluxGenerator:
    def __init__(self, model_path="./models/FLUX.1-Krea-dev", memory_mode="balanced"):
        self.model_path = model_path
        self.memory_mode = memory_mode
        self.pipeline = None
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load pipeline with Apple Silicon optimizations"""
        print(f"🚀 Loading FLUX.1 Krea ({self.memory_mode} mode)...")
        
        self.pipeline = FluxPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            use_safetensors=True,
            variant="fp16"
        )
        
        # Apply memory optimizations based on mode
        if self.memory_mode == "speed":
            # Minimal optimizations for maximum speed
            pass
        elif self.memory_mode == "balanced":
            # Recommended for M4 Pro with 48GB RAM
            self.pipeline.enable_model_cpu_offload()
            self.pipeline.vae.enable_slicing()
        else:  # memory mode
            # Maximum memory efficiency
            self.pipeline.enable_sequential_cpu_offload()
            self.pipeline.vae.enable_slicing()
            self.pipeline.vae.enable_tiling()
        
        print(f"✅ Pipeline loaded - Memory: {psutil.virtual_memory().percent:.1f}% used")
    
    def generate(self, prompt, **kwargs):
        """Generate image with optimized parameters"""
        params = {
            "prompt": prompt,
            "height": kwargs.get("height", 1024),
            "width": kwargs.get("width", 1024),
            "num_inference_steps": kwargs.get("steps", 28),
            "guidance_scale": kwargs.get("guidance", 4.0),
            "max_sequence_length": 512,
            "generator": torch.Generator().manual_seed(kwargs.get("seed")) if kwargs.get("seed") else None
        }
        
        print(f"🎨 Generating: '{prompt[:50]}...'")
        start_time = time.time()
        
        result = self.pipeline(**params)
        
        generation_time = time.time() - start_time
        print(f"✅ Complete in {generation_time:.1f}s")
        
        return result.images[0]

def main():
    parser = argparse.ArgumentParser(description='Optimized FLUX Generation')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--output', default='flux_optimized.png')
    parser.add_argument('--memory', choices=['speed', 'balanced', 'memory'], default='balanced')
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=28)
    parser.add_argument('--guidance', type=float, default=4.0)
    parser.add_argument('--seed', type=int)
    
    args = parser.parse_args()
    
    generator = OptimizedFluxGenerator(memory_mode=args.memory)
    
    image = generator.generate(
        args.prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed
    )
    
    image.save(args.output)
    print(f"💾 Saved: {args.output}")

if __name__ == "__main__":
    main()