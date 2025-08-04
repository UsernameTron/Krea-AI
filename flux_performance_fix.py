#!/usr/bin/env python3
"""
FLUX.1 Krea Performance Fix for M4 Pro
Addresses critical performance bottlenecks in pipeline loading and generation
"""

import torch
import time
import os
import gc
from pathlib import Path
from diffusers import FluxPipeline
import argparse
from typing import Optional

class FluxPerformanceFix:
    """High-performance FLUX implementation with critical fixes"""
    
    def __init__(self):
        self.device = self._setup_optimal_device()
        self.dtype = torch.bfloat16  # Optimal for M4 Pro
        self.pipeline = None
        
        # Apply critical performance fixes
        self._apply_performance_fixes()
    
    def _setup_optimal_device(self) -> str:
        """Setup optimal device with proper checks"""
        if torch.backends.mps.is_available():
            print("✅ MPS (Metal Performance Shaders) detected")
            return "mps"
        elif torch.cuda.is_available():
            print("✅ CUDA detected")
            return "cuda"
        else:
            print("⚠️  Using CPU - performance will be limited")
            return "cpu"
    
    def _apply_performance_fixes(self):
        """Apply critical performance optimizations"""
        print("🔧 Applying M4 Pro performance fixes...")
        
        # Critical M4 Pro environment variables
        env_vars = {
            "PYTORCH_MPS_MEMORY_FRACTION": "0.95",
            "PYTORCH_MPS_ALLOCATOR_POLICY": "expandable_segments", 
            "PYTORCH_MPS_PREFER_FAST_ALLOC": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
            "OMP_NUM_THREADS": "12",
            "MKL_NUM_THREADS": "12",
            "TOKENIZERS_PARALLELISM": "false",  # Prevents tokenizer warnings
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",  # Reduces console clutter
            # HF_TOKEN should be set in environment before running
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        # PyTorch optimization settings
        torch.set_num_threads(12)
        
        if self.device == "mps":
            torch.backends.mps.allow_fp16_reduced_precision = True
            torch.backends.mps.allow_tf32 = True
        
        # Memory optimization
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print("✅ Performance fixes applied")
    
    def load_pipeline_optimized(self, model_id: str = "black-forest-labs/FLUX.1-Krea-dev"):
        """Load pipeline with aggressive optimizations"""
        print(f"📥 Loading {model_id} with performance optimizations...")
        start_time = time.time()
        
        try:
            # Load with optimized settings
            self.pipeline = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                use_safetensors=True,
            )
            
            print(f"📦 Moving pipeline to {self.device}...")
            self.pipeline = self.pipeline.to(self.device)
            
            # Critical memory optimizations
            if self.device == "mps":
                self._apply_mps_optimizations()
            elif self.device == "cuda":  
                self._apply_cuda_optimizations()
            else:
                self._apply_cpu_optimizations()
            
            load_time = time.time() - start_time
            print(f"✅ Pipeline loaded in {load_time:.1f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline loading failed: {e}")
            return False
    
    def _apply_mps_optimizations(self):
        """Apply MPS-specific optimizations"""
        try:
            # Enable attention slicing for memory efficiency
            self.pipeline.enable_attention_slicing(1)
            print("✅ Attention slicing enabled")
            
            # Enable VAE slicing
            if hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()
                print("✅ VAE slicing enabled")
            
            # Sequential CPU offload for large models
            try:
                self.pipeline.enable_sequential_cpu_offload()
                print("✅ Sequential CPU offload enabled")
            except:
                print("ℹ️  Sequential CPU offload not available")
            
        except Exception as e:
            print(f"⚠️  MPS optimization warning: {e}")
    
    def _apply_cuda_optimizations(self):
        """Apply CUDA-specific optimizations"""
        try:
            self.pipeline.enable_model_cpu_offload()
            self.pipeline.enable_attention_slicing(1)
            if hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()
            print("✅ CUDA optimizations applied")
        except Exception as e:
            print(f"⚠️  CUDA optimization warning: {e}")
    
    def _apply_cpu_optimizations(self):
        """Apply CPU-specific optimizations"""
        try:
            self.pipeline.enable_attention_slicing("max")
            if hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()
            print("✅ CPU optimizations applied")
        except Exception as e:
            print(f"⚠️  CPU optimization warning: {e}")
    
    def generate_optimized(self, prompt: str, width: int = 1024, height: int = 1024,
                          guidance_scale: float = 4.5, num_inference_steps: int = 20,
                          seed: Optional[int] = None) -> tuple:
        """Generate image with performance optimizations"""
        
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline_optimized() first.")
        
        print(f"🖼️  Generating with optimizations...")
        print(f"   Prompt: {prompt}")
        print(f"   Size: {width}x{height}")
        print(f"   Steps: {num_inference_steps}")
        print(f"   Device: {self.device}")
        
        # Clear cache before generation
        self._clear_cache()
        
        start_time = time.time()
        
        try:
            # Setup generator
            generator = None
            if seed is not None:
                if self.device == "mps":
                    generator = torch.Generator().manual_seed(seed)
                else:
                    generator = torch.Generator(device=self.device).manual_seed(seed)
                print(f"   Seed: {seed}")
            
            # Generate with optimizations
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    max_sequence_length=256,  # Optimize sequence length
                )
            
            generation_time = time.time() - start_time
            
            print(f"✅ Generation complete!")
            print(f"⏱️  Time: {generation_time:.1f}s")
            print(f"🚀 Speed: {generation_time/num_inference_steps:.1f}s/step")
            print(f"🖥️  Pixels/sec: {(width*height)/generation_time:.0f}")
            
            return result.images[0], generation_time
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            raise
    
    def _clear_cache(self):
        """Clear device cache"""
        if self.device == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
    
    def benchmark(self, steps_list: list = [4, 10, 20]) -> dict:
        """Run performance benchmark"""
        print("\n🏃 Running Performance Benchmark")
        print("=" * 40)
        
        results = {}
        test_prompt = "a cute cat sitting in a garden"
        
        for steps in steps_list:
            print(f"\n🔄 Testing {steps} steps...")
            
            try:
                start_time = time.time()
                image, gen_time = self.generate_optimized(
                    prompt=test_prompt,
                    width=512,  # Smaller for benchmark
                    height=512,
                    num_inference_steps=steps,
                    seed=42
                )
                
                step_time = gen_time / steps
                pixels_per_sec = (512 * 512) / gen_time
                
                results[f"{steps}_steps"] = {
                    "total_time": gen_time,
                    "time_per_step": step_time,
                    "pixels_per_second": pixels_per_sec
                }
                
                print(f"✅ {steps} steps: {gen_time:.1f}s ({step_time:.1f}s/step)")
                
            except Exception as e:
                print(f"❌ {steps} steps failed: {e}")
                results[f"{steps}_steps"] = {"error": str(e)}
        
        return results
    
    def cleanup(self):
        """Cleanup resources"""
        self._clear_cache()
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        print("✅ Resources cleaned up")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='FLUX.1 Krea Performance Fix')
    parser.add_argument('--prompt', default='a cute cat', help='Text prompt')
    parser.add_argument('--output', default='performance_fix_output.png', help='Output filename')
    parser.add_argument('--width', type=int, default=1024, help='Width')
    parser.add_argument('--height', type=int, default=1024, help='Height')  
    parser.add_argument('--steps', type=int, default=20, help='Inference steps')
    parser.add_argument('--guidance', type=float, default=4.5, help='Guidance scale')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    
    args = parser.parse_args()
    
    print("🚀 FLUX.1 Krea Performance Fix")
    print("=" * 35)
    
    # Initialize performance-fixed pipeline
    flux = FluxPerformanceFix()
    
    try:
        # Load pipeline
        if not flux.load_pipeline_optimized():
            print("❌ Failed to load pipeline")
            return 1
        
        if args.benchmark:
            # Run benchmark
            results = flux.benchmark()
            
            print(f"\n📊 BENCHMARK RESULTS")
            print("=" * 25)
            for test, result in results.items():
                if "error" not in result:
                    print(f"{test}: {result['time_per_step']:.1f}s/step")
                else:
                    print(f"{test}: Failed")
        else:
            # Single generation
            image, gen_time = flux.generate_optimized(
                prompt=args.prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                seed=args.seed
            )
            
            # Save image
            output_path = Path(args.output)
            image.save(output_path)
            print(f"💾 Saved: {output_path.absolute()}")
            
            # Performance comparison
            print(f"\n📈 PERFORMANCE COMPARISON")
            print("=" * 30)
            print(f"Your system: {gen_time/args.steps:.1f}s/step")
            print(f"Previous system: ~182.70s/step")
            print(f"Improvement: {182.70/(gen_time/args.steps):.1f}x faster!")
        
    except KeyboardInterrupt:
        print("\n⏸️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    finally:
        flux.cleanup()
    
    return 0

if __name__ == "__main__":
    exit(main())