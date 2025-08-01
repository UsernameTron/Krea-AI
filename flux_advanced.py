#!/usr/bin/env python3
"""
Advanced FLUX.1 Krea generation with full optimization suite for Apple Silicon
This script implements all performance optimizations documented in the official API
"""

import argparse
import torch
import time
import psutil
import os
from pathlib import Path
from diffusers import FluxPipeline
from diffusers.hooks import apply_group_offloading
import warnings

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class FluxAdvancedGenerator:
    def __init__(self, model_path="./models/FLUX.1-Krea-dev", optimization_level="balanced"):
        """
        Initialize the advanced FLUX generator with configurable optimizations
        
        optimization_level options:
        - "speed": Maximum speed, uses more memory
        - "balanced": Good balance of speed and memory efficiency  
        - "memory": Maximum memory efficiency, slower but works on lower-end hardware
        """
        self.model_path = model_path
        self.optimization_level = optimization_level
        self.pipeline = None
        self.load_pipeline()
    
    def load_pipeline(self):
        """Load and optimize the pipeline based on your M4 Pro's capabilities"""
        print("🚀 Initializing Advanced FLUX.1 Krea Pipeline...")
        print(f"   Optimization Level: {self.optimization_level}")
        
        start_time = time.time()
        
        # Load pipeline with optimal settings for Apple Silicon
        self.pipeline = FluxPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,  # Optimal precision for M4 Pro
            local_files_only=True,
            use_safetensors=True  # Faster loading
        )
        
        # Apply optimization based on selected level
        if self.optimization_level == "speed":
            self._apply_speed_optimizations()
        elif self.optimization_level == "balanced":
            self._apply_balanced_optimizations()
        else:  # memory
            self._apply_memory_optimizations()
        
        load_time = time.time() - start_time
        print(f"✅ Pipeline loaded and optimized in {load_time:.1f} seconds")
        self._print_memory_usage()
    
    def _apply_speed_optimizations(self):
        """Maximum performance optimizations - uses more memory but fastest generation"""
        print("   Applying speed optimizations...")
        # Enable attention slicing for better performance on Apple Silicon
        self.pipeline.enable_attention_slicing("max")
        # Enable VAE slicing for memory efficiency without major speed impact
        self.pipeline.vae.enable_slicing()
    
    def _apply_balanced_optimizations(self):
        """Balanced optimizations - recommended for M4 Pro with 48GB RAM"""
        print("   Applying balanced optimizations...")
        
        # CPU offloading for memory management
        self.pipeline.enable_model_cpu_offload()
        
        # Enable VAE optimizations
        self.pipeline.vae.enable_slicing()
        self.pipeline.vae.enable_tiling()
        
        # Apply group offloading for intelligent memory management
        self._setup_group_offloading()
    
    def _apply_memory_optimizations(self):
        """Maximum memory efficiency - slower but works with limited resources"""
        print("   Applying memory-focused optimizations...")
        
        # Sequential CPU offloading - most memory efficient
        self.pipeline.enable_sequential_cpu_offload()
        
        # Enable all VAE optimizations
        self.pipeline.vae.enable_slicing()
        self.pipeline.vae.enable_tiling()
        
        # Apply comprehensive group offloading
        self._setup_group_offloading()
    
    def _setup_group_offloading(self):
        """Implement the advanced group offloading from the documentation"""
        print("   Configuring intelligent group offloading...")
        
        # Apply group offloading to transformer (the main compute component)
        apply_group_offloading(
            self.pipeline.transformer,
            offload_type="leaf_level",
            offload_device=torch.device("cpu"),
            onload_device=torch.device("mps"),  # Apple Silicon GPU
            use_stream=True  # Overlap data transfer with computation
        )
        
        # Apply to text encoders for comprehensive optimization
        for encoder_name in ["text_encoder", "text_encoder_2"]:
            if hasattr(self.pipeline, encoder_name):
                encoder = getattr(self.pipeline, encoder_name)
                apply_group_offloading(
                    encoder,
                    offload_type="leaf_level",
                    offload_device=torch.device("cpu"),
                    onload_device=torch.device("mps"),
                    use_stream=True
                )
    
    def _print_memory_usage(self):
        """Display current memory usage to help understand resource utilization"""
        memory = psutil.virtual_memory()
        print(f"   Memory Usage: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent:.1f}%)")
    
    def generate_image(self, prompt, **kwargs):
        """Generate image with intelligent parameter optimization"""
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")
        
        # Set optimal defaults based on the documentation
        generation_params = {
            "prompt": prompt,
            "height": kwargs.get("height", 1024),
            "width": kwargs.get("width", 1024),
            "num_inference_steps": kwargs.get("steps", 28),  # Optimal for guidance-distilled model
            "guidance_scale": kwargs.get("guidance", 4.0),   # Recommended range 3.5-5.0
            "max_sequence_length": kwargs.get("max_seq_len", 512),
            "generator": torch.Generator().manual_seed(kwargs.get("seed")) if kwargs.get("seed") else None
        }
        
        print(f"🎨 Generating: '{prompt[:60]}{'...' if len(prompt) > 60 else ''}'")
        print(f"   Resolution: {generation_params['width']}x{generation_params['height']}")
        print(f"   Steps: {generation_params['num_inference_steps']}, Guidance: {generation_params['guidance_scale']}")
        
        start_time = time.time()
        
        # Generate with error handling
        try:
            result = self.pipeline(**generation_params)
            image = result.images[0]
            
            generation_time = time.time() - start_time
            print(f"✨ Generation completed in {generation_time:.1f} seconds")
            
            return image
            
        except Exception as e:
            print(f"❌ Generation failed: {str(e)}")
            print("💡 Try reducing resolution or enabling memory optimizations")
            raise

def main():
    parser = argparse.ArgumentParser(description='Advanced FLUX.1 Krea Generation')
    parser.add_argument('--prompt', required=True, help='Generation prompt')
    parser.add_argument('--output', default='flux_advanced.png', help='Output filename')
    parser.add_argument('--optimization', choices=['speed', 'balanced', 'memory'], 
                       default='balanced', help='Optimization level')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--steps', type=int, default=28, help='Inference steps')
    parser.add_argument('--guidance', type=float, default=4.0, help='Guidance scale')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize the advanced generator
    generator = FluxAdvancedGenerator(optimization_level=args.optimization)
    
    # Generate the image
    image = generator.generate_image(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed
    )
    
    # Save the result
    image.save(args.output)
    print(f"💾 Image saved as: {args.output}")
    
    # Display final memory usage
    generator._print_memory_usage()

if __name__ == "__main__":
    main()