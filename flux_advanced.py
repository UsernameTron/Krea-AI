#!/usr/bin/env python3
"""
FLUX.1 Krea Advanced Generation System
Comprehensive text-to-image generation with optimization profiles
"""

import argparse
import torch
import time
from pathlib import Path
from src.flux.util import load_ae, load_clip, load_t5, load_flow_model
from src.flux.pipeline import Sampler
from flux_exceptions import (
    ModelNotFoundError, DeviceError, InsufficientMemoryError, 
    GenerationError, handle_common_errors
)
from flux_model_validator import ModelValidator, validate_before_loading

class FluxAdvancedGenerator:
    def __init__(self, optimization_level="balanced"):
        self.optimization_level = optimization_level
        self.pipeline = None
        
    @handle_common_errors
    # @validate_before_loading("flux-krea-dev")  # Disabled - models are in HF cache
    def load_pipeline(self):
        """Load FLUX pipeline with specified optimizations"""
        print(f"🚀 Loading FLUX.1 Krea with '{self.optimization_level}' optimization...")
        
        torch_dtype = torch.bfloat16
        
        # Smart device detection with validation
        try:
            if torch.cuda.is_available():
                device = "cuda"
                # Test CUDA device
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon GPU
            else:
                device = "cpu"
                print("⚠️  Using CPU - generation will be slower")
        except Exception as e:
            raise DeviceError("auto", f"Device detection failed: {e}")
        
        print(f"Using device: {device}")
        
        # Load models with proper error handling
        try:
            print("Loading autoencoder...")
            ae = load_ae("flux-krea-dev", device=device)
            print("Loading CLIP...")
            clip = load_clip(device=device)
            print("Loading T5...")
            t5 = load_t5(device=device)  
            print("Loading FLUX model...")
            model = load_flow_model("flux-krea-dev", device=device)
        except FileNotFoundError:
            raise ModelNotFoundError("flux-krea-dev", "./models/FLUX.1-Krea-dev")
        except torch.cuda.OutOfMemoryError:
            raise InsufficientMemoryError("GPU memory insufficient for model loading")
        
        # Apply optimization-specific model placement
        if self.optimization_level == "speed":
            # Prioritize speed - keep everything on primary device
            print("⚡ Speed optimization: All models on primary device")
            target_device = device
            model = model.to(device=target_device, dtype=torch_dtype)
            ae = ae.to(device=target_device, dtype=torch_dtype)
            clip = clip.to(device=target_device, dtype=torch_dtype)
            t5 = t5.to(device=target_device, dtype=torch_dtype)
            
        elif self.optimization_level == "memory":
            # Memory-efficient: Strategic offloading
            print("🧠 Memory optimization: Strategic model offloading")
            # Keep smaller, frequently used models on device
            ae = ae.to(device=device, dtype=torch_dtype)  # Small, used for encoding/decoding
            # Offload large models that can be moved as needed
            model = model.to(device="cpu", dtype=torch_dtype)  # Large model
            clip = clip.to(device="cpu", dtype=torch_dtype)   # Can be offloaded
            t5 = t5.to(device="cpu", dtype=torch_dtype)       # Large text encoder
            
        else:  # balanced
            # Balanced: Smart placement based on device capabilities
            print("⚖️  Balanced optimization: Smart model placement")
            if device == "cuda":
                # GPU available - keep most on GPU, offload T5 (largest)
                model = model.to(device=device, dtype=torch_dtype)
                ae = ae.to(device=device, dtype=torch_dtype)
                clip = clip.to(device=device, dtype=torch_dtype)
                t5 = t5.to(device="cpu", dtype=torch_dtype)  # Offload largest
            else:
                # CPU or MPS - keep all on same device for consistency
                model = model.to(device=device, dtype=torch_dtype)
                ae = ae.to(device=device, dtype=torch_dtype)
                clip = clip.to(device=device, dtype=torch_dtype)
                t5 = t5.to(device=device, dtype=torch_dtype)
        
        # Create sampler
        self.pipeline = Sampler(
            model=model,
            ae=ae,
            clip=clip,
            t5=t5,
            device=device,
            dtype=torch_dtype,
        )
        
        print(f"✅ Pipeline loaded with {self.optimization_level} optimizations")
    
    def generate_image(self, prompt, **kwargs):
        """Generate image with advanced parameters"""
        if self.pipeline is None:
            self.load_pipeline()
            
        # Default parameters optimized for FLUX.1 Krea
        height = kwargs.get("height", 1024)
        width = kwargs.get("width", 1024)
        steps = kwargs.get("steps", 28)
        guidance = kwargs.get("guidance", 4.0)
        seed = kwargs.get("seed", 42)
        
        print(f"🎨 Generating: '{prompt[:50]}...'")
        print(f"   Size: {width}x{height}")
        print(f"   Steps: {steps}, Guidance: {guidance}")
        
        start_time = time.time()
        result = self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_steps=steps,
            guidance=guidance,
            seed=seed
        )
        generation_time = time.time() - start_time
        
        print(f"✅ Generated in {generation_time:.1f}s")
        return result

def main():
    parser = argparse.ArgumentParser(description='FLUX.1 Krea Advanced Generation')
    parser.add_argument('--prompt', required=True, help='Generation prompt')
    parser.add_argument('--output', default='flux_advanced.png', help='Output filename')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--steps', type=int, default=28, help='Inference steps')
    parser.add_argument('--guidance', type=float, default=4.0, help='Guidance scale')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--optimization', choices=['speed', 'balanced', 'memory'], 
                       default='balanced', help='Optimization level')
    
    args = parser.parse_args()
    
    generator = FluxAdvancedGenerator(optimization_level=args.optimization)
    
    image = generator.generate_image(
        args.prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed
    )
    
    # Save result
    output_path = Path(args.output)
    image.save(output_path)
    print(f"💾 Saved: {output_path}")

if __name__ == "__main__":
    main()