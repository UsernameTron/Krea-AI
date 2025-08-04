#!/usr/bin/env python3
"""
FLUX.1 Krea [dev] - M4 Pro Optimized Inference Script
High-performance inference optimized for Apple Silicon M4 Pro
"""

import os
import torch
import argparse
import time
import psutil
from pathlib import Path
from typing import Optional

# Check for required modules
try:
    from src.flux.util import load_ae, load_clip, load_t5, load_flow_model
    from src.flux.pipeline import Sampler
    FLUX_MODULES_AVAILABLE = True
except ImportError:
    print("⚠️  FLUX modules not found. Falling back to diffusers implementation.")
    FLUX_MODULES_AVAILABLE = False
    from diffusers import FluxPipeline

class M4ProFluxInference:
    """Optimized FLUX inference for Apple Silicon M4 Pro"""
    
    def __init__(self, use_official_modules: bool = True):
        self.use_official_modules = use_official_modules and FLUX_MODULES_AVAILABLE
        self.device = self._get_optimal_device()
        self.dtype = torch.bfloat16
        
        # Performance tracking
        self.load_time = 0.0
        self.generation_time = 0.0
        
        # Model components
        self.model = None
        self.ae = None
        self.clip = None
        self.t5 = None
        self.sampler = None
        self.pipeline = None
        
        self._apply_m4_optimizations()
    
    def _get_optimal_device(self) -> str:
        """Get optimal device for M4 Pro"""
        if torch.backends.mps.is_available():
            print("✅ Metal Performance Shaders (MPS) available")
            return "mps"
        elif torch.cuda.is_available():
            print("✅ CUDA available")
            return "cuda"
        else:
            print("⚠️  Using CPU (consider installing PyTorch with MPS support)")
            return "cpu"
    
    def _apply_m4_optimizations(self):
        """Apply M4 Pro specific optimizations"""
        print("🔧 Applying M4 Pro optimizations...")
        
        # Set optimal environment variables
        os.environ["PYTORCH_MPS_MEMORY_FRACTION"] = "0.95"
        os.environ["PYTORCH_MPS_ALLOCATOR_POLICY"] = "expandable_segments"
        os.environ["PYTORCH_MPS_PREFER_FAST_ALLOC"] = "1"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["OMP_NUM_THREADS"] = "12"  # 8P + 4E cores
        os.environ["MKL_NUM_THREADS"] = "12"
        # HF_TOKEN should be set in environment before running
        
        # Optimize PyTorch for Apple Silicon
        if self.device == "mps":
            # Enable MPS optimizations
            torch.backends.mps.allow_fp16_reduced_precision = True
            torch.backends.mps.allow_tf32 = True
            
        # Set optimal thread count for M4 Pro
        torch.set_num_threads(12)
        
        print("✅ M4 Pro optimizations applied")
    
    def _apply_vae_optimizations(self):
        """Apply VAE memory optimizations"""
        if self.ae is None:
            return
            
        try:
            # Enable VAE slicing for memory efficiency
            if hasattr(self.ae, 'enable_slicing'):
                self.ae.enable_slicing()
                print("    ✅ VAE slicing enabled")
            elif hasattr(self.ae, 'enable_vae_slicing'):
                self.ae.enable_vae_slicing()
                print("    ✅ VAE slicing enabled")
                
            # Enable VAE tiling for very large images
            if hasattr(self.ae, 'enable_tiling'):
                self.ae.enable_tiling()
                print("    ✅ VAE tiling enabled")
            elif hasattr(self.ae, 'enable_vae_tiling'):
                self.ae.enable_vae_tiling()
                print("    ✅ VAE tiling enabled")
                
        except Exception as e:
            print(f"    ⚠️  VAE optimization warning: {e}")
    
    def _apply_attention_optimizations(self):
        """Apply attention memory optimizations"""
        if self.model is None:
            return
            
        try:
            # Enable attention slicing for memory efficiency
            if hasattr(self.model, 'enable_attention_slicing'):
                self.model.enable_attention_slicing("auto")
                print("    ✅ Attention slicing enabled")
            elif hasattr(self.model, 'enable_sliced_attention'):
                self.model.enable_sliced_attention()
                print("    ✅ Attention slicing enabled")
                
            # Enable memory efficient attention if available
            if hasattr(self.model, 'enable_memory_efficient_attention'):
                self.model.enable_memory_efficient_attention()
                print("    ✅ Memory efficient attention enabled")
                
        except Exception as e:
            print(f"    ⚠️  Attention optimization warning: {e}")
    
    def load_models(self, model_name: str = "flux-krea-dev"):
        """Load models with M4 Pro optimizations"""
        print(f"📥 Loading {model_name} with M4 Pro optimizations...")
        start_time = time.time()
        
        if self.use_official_modules:
            self._load_flux_modules(model_name)
        else:
            self._load_diffusers_pipeline(model_name)
        
        self.load_time = time.time() - start_time
        print(f"✅ Models loaded in {self.load_time:.1f} seconds")
    
    def _load_flux_modules(self, model_name: str):
        """Load FLUX modules (original implementation)"""
        print("Using FLUX original modules...")
        
        try:
            # Load models to CPU first, then move to device
            print("  Loading flow model...")
            self.model = load_flow_model(model_name, device="cpu")
            
            print("  Loading autoencoder...")
            self.ae = load_ae(model_name)
            
            print("  Loading CLIP...")
            self.clip = load_clip()
            
            print("  Loading T5...")
            self.t5 = load_t5()
            
            # Move models to device with proper dtype
            print(f"  Moving models to {self.device}...")
            self.ae = self.ae.to(device=self.device, dtype=self.dtype)
            self.clip = self.clip.to(device=self.device, dtype=self.dtype)
            self.t5 = self.t5.to(device=self.device, dtype=self.dtype)
            self.model = self.model.to(device=self.device, dtype=self.dtype)
            
            # Apply memory optimizations to autoencoder
            print("  Applying VAE optimizations...")
            self._apply_vae_optimizations()
            
            # Apply attention optimizations to main model
            print("  Applying attention optimizations...")
            self._apply_attention_optimizations()
            
            # Create sampler
            self.sampler = Sampler(
                model=self.model,
                ae=self.ae,
                clip=self.clip,
                t5=self.t5,
                device=self.device,
                dtype=self.dtype,
            )
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["gated", "access denied", "401", "403", "unauthorized", "token"]):
                print(f"\n❌ HuggingFace Access Error: {e}")
                print("\n🔑 Authentication Required:")
                print("1. Get a HuggingFace token: https://huggingface.co/settings/tokens")
                print("2. Request model access: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev")
                print("3. Set token: export HF_TOKEN=your_token_here")
                print("4. Or use: huggingface-cli login")
            raise
    
    def _load_diffusers_pipeline(self, model_name: str):
        """Load diffusers pipeline (fallback)"""
        print("Using diffusers pipeline...")
        
        full_model_name = f"black-forest-labs/FLUX.1-Krea-dev"
        
        self.pipeline = FluxPipeline.from_pretrained(
            full_model_name,
            torch_dtype=self.dtype
        )
        
        # Move to device
        if self.device != "cpu":
            self.pipeline = self.pipeline.to(self.device)
        
        # Apply memory optimizations
        try:
            self.pipeline.enable_model_cpu_offload()
        except Exception:
            try:
                self.pipeline.enable_sequential_cpu_offload()
            except Exception as e:
                print(f"Note: CPU offload not available: {e}")
        
        # Enable memory-efficient attention and VAE optimizations
        try:
            self.pipeline.enable_attention_slicing("auto")
        except Exception:
            pass
            
        try:
            self.pipeline.enable_vae_slicing()
        except Exception:
            pass
            
        try:
            self.pipeline.enable_vae_tiling()
        except Exception:
            pass
    
    def generate_image(self, prompt: str, width: int = 1024, height: int = 1024,
                      guidance: float = 4.5, num_steps: int = 28, 
                      seed: Optional[int] = None):
        """Generate image with M4 Pro optimizations"""
        
        if not (self.sampler or self.pipeline):
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        print(f"🖼️  Generating image...")
        print(f"  Prompt: {prompt}")
        print(f"  Size: {width}x{height}")
        print(f"  Steps: {num_steps}, Guidance: {guidance}")
        print(f"  Device: {self.device}")
        
        # Clear MPS cache before generation
        if self.device == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()
        
        start_time = time.time()
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            print(f"  Seed: {seed}")
        
        try:
            if self.use_official_modules:
                # Use FLUX sampler
                image = self.sampler(
                    prompt=prompt,
                    width=width,
                    height=height,
                    guidance=guidance,
                    num_steps=num_steps,
                    seed=seed,
                )
            else:
                # Use diffusers pipeline
                generator = torch.Generator(device=self.device).manual_seed(seed) if seed else None
                
                image = self.pipeline(
                    prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance,
                    num_inference_steps=num_steps,
                    generator=generator
                ).images[0]
            
            self.generation_time = time.time() - start_time
            
            print(f"✅ Generation complete in {self.generation_time:.1f} seconds")
            print(f"🚀 Speed: {(width * height) / self.generation_time:.0f} pixels/second")
            
            return image
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            raise
    
    def save_image(self, image, output_path: str):
        """Save generated image"""
        try:
            save_path = Path(output_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Handle different image types
            if hasattr(image, 'save'):
                image.save(save_path)
            else:
                # Convert tensor to PIL if needed
                from PIL import Image
                if torch.is_tensor(image):
                    # Convert from tensor to PIL
                    image_array = (image.cpu().numpy() * 255).astype('uint8')
                    if len(image_array.shape) == 4:
                        image_array = image_array[0]  # Remove batch dimension
                    if image_array.shape[0] in [1, 3]:  # CHW format
                        image_array = image_array.transpose(1, 2, 0)
                    image = Image.fromarray(image_array)
                    image.save(save_path)
                else:
                    image.save(save_path)
            
            print(f"💾 Image saved: {save_path.absolute()}")
            return save_path
            
        except Exception as e:
            print(f"❌ Failed to save image: {e}")
            raise
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        return {
            "device": self.device,
            "dtype": str(self.dtype),
            "load_time": self.load_time,
            "generation_time": self.generation_time,
            "total_time": self.load_time + self.generation_time,
            "memory_used_gb": psutil.Process().memory_info().rss / (1024**3),
            "system_memory_percent": psutil.virtual_memory().percent
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        
        print("✅ Resources cleaned up")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='FLUX.1 Krea M4 Pro Optimized Inference')
    parser.add_argument('--prompt', required=True, help='Text prompt for image generation')
    parser.add_argument('--output', default='m4_optimized_output.png', help='Output filename')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--guidance', type=float, default=4.5, help='Guidance scale')
    parser.add_argument('--steps', type=int, default=28, help='Number of inference steps')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    parser.add_argument('--use-diffusers', action='store_true', 
                       help='Force use of diffusers pipeline instead of FLUX modules')
    
    args = parser.parse_args()
    
    print("🚀 FLUX.1 Krea [dev] - M4 Pro Optimized Inference")
    print("=" * 55)
    
    # System info
    print(f"💻 System: Apple Silicon M4 Pro")
    print(f"🧠 CPU cores: {psutil.cpu_count()} ({psutil.cpu_count(logical=False)} physical)")
    print(f"💾 Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    try:
        # Initialize inference engine
        use_modules = not args.use_diffusers
        inference = M4ProFluxInference(use_official_modules=use_modules)
        
        # Load models
        inference.load_models()
        
        # Generate image
        image = inference.generate_image(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            num_steps=args.steps,
            seed=args.seed
        )
        
        # Save image
        inference.save_image(image, args.output)
        
        # Show performance stats
        stats = inference.get_performance_stats()
        print(f"\n📊 Performance Statistics:")
        print(f"  Device: {stats['device']}")
        print(f"  Load time: {stats['load_time']:.1f}s")
        print(f"  Generation time: {stats['generation_time']:.1f}s")
        print(f"  Total time: {stats['total_time']:.1f}s")
        print(f"  Memory used: {stats['memory_used_gb']:.1f} GB")
        print(f"  System memory: {stats['system_memory_percent']:.1f}%")
        
        # Cleanup
        inference.cleanup()
        
        print(f"\n🎉 Success! Image generated and saved to {args.output}")
        
    except KeyboardInterrupt:
        print("\n⏸️  Generation interrupted by user")
    except Exception as e:
        error_msg = str(e).lower()
        
        # Enhanced HuggingFace authentication error handling
        if any(keyword in error_msg for keyword in ["gated", "access denied", "repository is gated"]):
            print(f"\n❌ Model Access Restricted: {e}")
            print("\n🔒 This model requires explicit access approval:")
            print("1. Visit: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev")
            print("2. Click 'Request access to this model' and wait for approval")
            print("3. Once approved, set your token: export HF_TOKEN=your_token_here")
            print("4. Get your token at: https://huggingface.co/settings/tokens")
            print("5. Alternative: Run 'huggingface-cli login' and follow prompts")
        elif any(keyword in error_msg for keyword in ["401", "403", "unauthorized", "authentication", "token"]):
            print(f"\n❌ Authentication Error: {e}")
            print("\n🔑 HuggingFace token required:")
            print("1. Create account at: https://huggingface.co/join")
            print("2. Generate token at: https://huggingface.co/settings/tokens")
            print("3. Set environment variable: export HF_TOKEN=your_token_here")
            print("4. Alternative: Run 'huggingface-cli login'")
            print("5. Ensure token has 'read' permissions")
        elif any(keyword in error_msg for keyword in ["connection", "network", "timeout", "dns"]):
            print(f"\n❌ Network Error: {e}")
            print("\n🌐 Connection troubleshooting:")
            print("1. Check internet connection")
            print("2. Try again in a few minutes (server may be busy)")
            print("3. Check if HuggingFace is accessible: https://status.huggingface.co")
            print("4. Try using a VPN if access is restricted in your region")
        elif any(keyword in error_msg for keyword in ["memory", "out of memory", "oom", "cuda out of memory"]):
            print(f"\n❌ Memory Error: {e}")
            print("\n💾 Memory optimization suggestions:")
            print("1. Reduce image size: --width 512 --height 512")
            print("2. Reduce inference steps: --steps 20")
            print("3. Use CPU offloading: --use-diffusers (enables automatic offloading)")
            print("4. Close other applications to free memory")
            print("5. For M4 Pro: Ensure you have at least 16GB unified memory")
        elif any(keyword in error_msg for keyword in ["module", "import", "no module named"]):
            print(f"\n❌ Missing Dependencies: {e}")
            print("\n📦 Installation required:")
            print("1. Install requirements: pip install -r requirements_flux_krea.txt")
            print("2. Or use uv: uv sync")
            print("3. Ensure you're in the correct environment")
            print("4. Try: pip install --upgrade torch torchvision torchaudio")
        else:
            print(f"\n❌ Unexpected Error: {e}")
            print("\n🔍 General troubleshooting:")
            print("1. Check that all requirements are installed")
            print("2. Verify HuggingFace token is set correctly")
            print("3. Ensure sufficient system resources")
            print("4. Try the --use-diffusers flag for fallback implementation")
        
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())