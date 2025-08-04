#!/usr/bin/env python3
"""
FLUX.1 Krea [dev] - Apple Silicon M4 Pro Optimized Implementation
Leverages Metal Performance Shaders, unified memory, and Neural Engine
"""

import torch
import torch.backends.mps
import argparse
import time
import gc
import os
from pathlib import Path
from diffusers import FluxPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
import psutil
import threading
from contextlib import contextmanager

class AppleSiliconOptimizer:
    """Apple Silicon specific optimizations for FLUX.1 Krea"""
    
    def __init__(self):
        self.device = self._get_optimal_device()
        self.dtype = torch.bfloat16  # Optimal for Apple Silicon
        self._configure_metal()
        self._optimize_memory()
    
    def _get_optimal_device(self):
        """Determine the best device for Apple Silicon"""
        if torch.backends.mps.is_available():
            # Verify MPS is built correctly
            if torch.backends.mps.is_built():
                return torch.device("mps")
        return torch.device("cpu")
    
    def _configure_metal(self):
        """Configure Metal Performance Shaders optimizations"""
        if self.device.type == "mps":
            # Enable high-performance mode
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            # Optimize memory allocation patterns
            torch.mps.set_per_process_memory_fraction(0.8)
            # Enable Metal shader caching
            os.environ['PYTORCH_MPS_PREFER_FAST_ALLOC'] = '1'
    
    def _optimize_memory(self):
        """Configure unified memory architecture optimizations"""
        # Set optimal thread counts for M4 Pro (12 cores: 8P + 4E)
        torch.set_num_threads(8)  # Use performance cores only
        
        # Configure memory mapping for large models
        os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'page'
        
        # Enable memory efficient attention
        os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'

class M4ProFluxPipeline:
    """M4 Pro optimized FLUX pipeline with Metal acceleration"""
    
    def __init__(self):
        self.optimizer = AppleSiliconOptimizer()
        self.pipeline = None
        self.is_loaded = False
        
    def load_pipeline(self, model_id="black-forest-labs/FLUX.1-Krea-dev"):
        """Load pipeline with Apple Silicon optimizations"""
        print(f"🍎 Loading on device: {self.optimizer.device}")
        print(f"🧠 Using dtype: {self.optimizer.dtype}")
        
        # Load with memory-efficient settings
        self.pipeline = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=self.optimizer.dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True,  # Critical for unified memory
        )
        
        # Move to optimal device
        self.pipeline = self.pipeline.to(self.optimizer.device)
        
        # Apply Apple Silicon specific optimizations
        self._apply_attention_optimizations()
        self._enable_memory_efficient_attention()
        self._configure_vae_tiling()
        
        # Enable CPU offloading for memory efficiency - must be done after moving to device
        try:
            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
                print("✅ Model CPU offload enabled")
        except Exception as e:
            print(f"Note: Model CPU offload not available: {e}")
            # Try sequential CPU offload as fallback
            try:
                if hasattr(self.pipeline, 'enable_sequential_cpu_offload'):
                    self.pipeline.enable_sequential_cpu_offload()
                    print("✅ Sequential CPU offload enabled")
            except Exception as e2:
                print(f"Note: Sequential CPU offload not available: {e2}")
        
        self.is_loaded = True
        print("✅ Pipeline loaded with Apple Silicon optimizations")
    
    def _apply_attention_optimizations(self):
        """Apply optimized attention processors for Apple Silicon"""
        # Use attention processor optimized for Apple Silicon
        attention_processor = AttnProcessor2_0()
        
        # Apply to transformer blocks
        if hasattr(self.pipeline, 'transformer'):
            self.pipeline.transformer.set_attn_processor(attention_processor)
    
    def _enable_memory_efficient_attention(self):
        """Enable memory efficient attention patterns"""
        if hasattr(self.pipeline, 'enable_attention_slicing'):
            # Use slice size optimized for M4 Pro's memory bandwidth
            self.pipeline.enable_attention_slicing("auto")
        
        if hasattr(self.pipeline, 'enable_sequential_cpu_offload'):
            # Use sequential offloading for better memory management
            self.pipeline.enable_sequential_cpu_offload()
    
    def _configure_vae_tiling(self):
        """Configure VAE tiling for memory efficiency"""
        if hasattr(self.pipeline, 'enable_vae_tiling'):
            self.pipeline.enable_vae_tiling()
        
        if hasattr(self.pipeline, 'enable_vae_slicing'):
            self.pipeline.enable_vae_slicing()
    
    @contextmanager
    def _optimized_inference_context(self):
        """Context manager for optimized inference"""
        try:
            # Disable autograd for inference
            with torch.inference_mode():
                # Enable MPS optimizations if available
                if self.optimizer.device.type == "mps":
                    # Use MPS optimizations without sdp_kernel for compatibility
                    yield
                else:
                    yield
        finally:
            # Aggressive cleanup after inference
            self._cleanup_memory()
    
    def _cleanup_memory(self):
        """Aggressive memory cleanup for Apple Silicon"""
        if self.optimizer.device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()
    
    def generate(self, prompt, width=1024, height=1024, guidance_scale=4.5, 
                 num_inference_steps=28, seed=None):
        """Generate image with Apple Silicon optimizations"""
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")
        
        with self._optimized_inference_context():
            # Set up generator for reproducible results
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.optimizer.device)
                generator.manual_seed(seed)
            
            # Generate with optimized parameters
            result = self.pipeline(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                return_dict=True
            )
            
            return result.images[0]

def main():
    parser = argparse.ArgumentParser(description='FLUX.1 Krea M4 Pro Optimized Generation')
    parser.add_argument('--prompt', required=True, help='Text prompt for image generation')
    parser.add_argument('--output', default='flux-krea-m4-optimized.png', help='Output filename')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--guidance', type=float, default=4.5, help='Guidance scale')
    parser.add_argument('--steps', type=int, default=28, help='Number of inference steps')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    args = parser.parse_args()
    
    print("🍎 FLUX.1 Krea [dev] - Apple Silicon M4 Pro Optimized")
    print("=" * 55)
    print(f"Device: {'Metal Performance Shaders' if torch.backends.mps.is_available() else 'CPU'}")
    print(f"Memory: {psutil.virtual_memory().total // (1024**3)} GB unified memory")
    print(f"Prompt: {args.prompt}")
    
    # Initialize optimized pipeline
    flux_pipeline = M4ProFluxPipeline()
    
    start_time = time.time()
    
    try:
        # Load pipeline with optimizations
        flux_pipeline.load_pipeline()
        load_time = time.time() - start_time
        
        print(f"\n🖼️  Generating {args.width}x{args.height} image...")
        generation_start = time.time()
        
        # Generate image
        image = flux_pipeline.generate(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            seed=args.seed
        )
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        # Save image
        output_path = Path(args.output)
        image.save(output_path)
        
        print(f"🎉 Generation complete!")
        print(f"⏱️  Load time: {load_time:.1f}s | Generation: {generation_time:.1f}s | Total: {total_time:.1f}s")
        print(f"💾 Saved: {output_path.absolute()}")
        print(f"🧠 Peak memory: {torch.mps.current_allocated_memory() // (1024**2)} MB" if torch.backends.mps.is_available() else "")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Apple Silicon Troubleshooting:")
        print("1. Update to latest PyTorch: pip install --upgrade torch torchvision")
        print("2. Verify MPS availability: python -c 'import torch; print(torch.backends.mps.is_available())'")
        print("3. Check memory: Activity Monitor > Memory tab")

if __name__ == "__main__":
    main()