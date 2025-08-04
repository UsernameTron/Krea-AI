#!/usr/bin/env python3
"""
FLUX.1 Krea [dev] - Interactive Command Line Interface
User-friendly interactive version for desktop launcher
"""

import torch
import argparse
import time
import signal
import sys
import os
from pathlib import Path
from diffusers import FluxPipeline

# Set optimal MPS environment variables
os.environ.setdefault('PYTORCH_MPS_MEMORY_FRACTION', '0.95')
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.7')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.6')
os.environ.setdefault('PYTORCH_MPS_ALLOCATOR_POLICY', 'expandable_segments')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Generation timed out")

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')

def print_banner():
    """Print FLUX banner"""
    print("🎨" + "=" * 60 + "🎨")
    print("       FLUX.1 Krea [dev] - Interactive Studio")
    print("     🛡️  Timeout Protected  •  🚀 Ready to Generate")
    print("🎨" + "=" * 60 + "🎨")
    print()

def get_user_input():
    """Get generation parameters from user"""
    print("📝 Generation Settings:")
    print("-" * 30)
    
    # Prompt
    while True:
        prompt = input("🖼️  Enter your prompt: ").strip()
        if prompt:
            break
        print("❌ Please enter a prompt!")
    
    # Quick settings or custom
    print("\n⚙️  Choose settings:")
    print("1. 🚀 Quick (768x768, 20 steps) - ~2 minutes")
    print("2. 🎯 Standard (1024x1024, 28 steps) - ~4 minutes") 
    print("3. 🎨 High Quality (1280x1024, 32 steps) - ~6 minutes")
    print("4. ⚙️  Custom settings")
    
    while True:
        choice = input("\nSelect option (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            break
        print("❌ Please enter 1, 2, 3, or 4!")
    
    if choice == '1':
        width, height, steps = 768, 768, 20
    elif choice == '2':
        width, height, steps = 1024, 1024, 28
    elif choice == '3':
        width, height, steps = 1280, 1024, 32
    else:
        # Custom settings
        print("\n⚙️  Custom Settings:")
        width = int(input("   Width (512-1280): ") or "1024")
        height = int(input("   Height (512-1280): ") or "1024")
        steps = int(input("   Steps (10-50): ") or "28")
    
    # Seed
    seed_input = input(f"\n🎲 Seed (press Enter for random): ").strip()
    seed = int(seed_input) if seed_input else None
    
    # Timeout
    timeout_input = input("⏰ Timeout minutes (default 5): ").strip()
    timeout = int(timeout_input) if timeout_input else 5
    
    return {
        'prompt': prompt,
        'width': width,
        'height': height,
        'steps': steps,
        'seed': seed,
        'timeout': timeout
    }

def _apply_pipeline_optimizations(pipeline):
    """Apply comprehensive pipeline optimizations"""
    optimizations_applied = []
    
    # Apply memory optimizations
    try:
        pipeline.enable_model_cpu_offload()
        optimizations_applied.append("CPU offload")
    except:
        try:
            pipeline.enable_sequential_cpu_offload()
            optimizations_applied.append("Sequential CPU offload")
        except:
            optimizations_applied.append("Default memory management")
    
    # Enable attention optimizations
    try:
        pipeline.enable_attention_slicing("auto")
        optimizations_applied.append("Attention slicing")
    except:
        pass
    
    # Enable memory-efficient attention if xformers not available
    try:
        # Try PyTorch native memory efficient attention
        if hasattr(pipeline, 'unet') and hasattr(pipeline.unet, 'set_attn_processor'):
            from diffusers.models.attention_processor import AttnProcessor2_0
            pipeline.unet.set_attn_processor(AttnProcessor2_0())
            optimizations_applied.append("Memory-efficient attention")
    except Exception as e:
        pass
    
    # Enable VAE optimizations
    try:
        pipeline.enable_vae_slicing()
        optimizations_applied.append("VAE slicing")
    except:
        pass
        
    try:
        pipeline.enable_vae_tiling()
        optimizations_applied.append("VAE tiling")
    except:
        pass
    
    # Print applied optimizations
    for opt in optimizations_applied:
        print(f"✅ {opt} enabled")

def generate_filename(prompt):
    """Generate descriptive filename from prompt"""
    # Clean prompt for filename
    clean_prompt = ''.join(c if c.isalnum() or c.isspace() else '' for c in prompt)
    clean_prompt = '_'.join(clean_prompt.split()[:5])  # First 5 words
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"flux_{clean_prompt}_{timestamp}.png"

def main():
    # Clear screen and show banner
    clear_screen()
    print_banner()
    
    # Check environment
    print("🔍 Environment Check:")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   MPS Available: {'✅' if torch.backends.mps.is_available() else '❌'}")
    
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print(f"   HF Token: ✅ Set (...{hf_token[-4:]})")
    else:
        print("   HF Token: ⚠️  Not set (may cause issues)")
    
    print()
    
    try:
        # Get user parameters
        params = get_user_input()
        
        print(f"\n📋 Generation Summary:")
        print(f"   Prompt: {params['prompt']}")
        print(f"   Size: {params['width']}x{params['height']}")
        print(f"   Steps: {params['steps']}")
        print(f"   Seed: {params['seed'] or 'Random'}")
        print(f"   Timeout: {params['timeout']} minutes")
        
        input(f"\n✅ Press Enter to start generation (or Ctrl+C to cancel)...")
        
        print(f"\n📥 Loading FLUX.1 Krea [dev] pipeline...")
        load_start = time.time()
        
        # Load pipeline with optimizations
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Krea-dev",
            torch_dtype=torch.float16,  # Use fp16 for 2x speed boost
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        
        # Move to MPS device with optimizations
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        pipeline = pipeline.to(device)
        
        # Apply comprehensive optimizations
        _apply_pipeline_optimizations(pipeline)
        
        # Enable Metal VAE if configured
        if os.getenv('KREA_ENABLE_METAL_VAE') == '1':
            try:
                from flux_metal_kernels import M4ProMetalOptimizer
                metal_optimizer = M4ProMetalOptimizer()
                pipeline = metal_optimizer.optimize_flux_pipeline_components(pipeline)
                print("✅ Metal-optimized VAE active")
            except Exception as e:
                print(f"⚠️  Metal VAE optimization failed: {e}")
        
        load_time = time.time() - load_start
        print(f"✅ Pipeline loaded in {load_time:.1f} seconds")
        
        print(f"\n🖼️  Generating image...")
        print(f"⏰ Timeout: {params['timeout']} minutes")
        print("🛑 Press Ctrl+C to cancel if needed")
        
        generation_start = time.time()
        
        # Set up timeout protection
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(params['timeout'] * 60)
        
        try:
            # Set up generator
            generator = torch.Generator().manual_seed(params['seed']) if params['seed'] else None
            
            # Generate with timeout protection
            with torch.inference_mode():
                result = pipeline(
                    params['prompt'],
                    height=params['height'],
                    width=params['width'],
                    guidance_scale=4.5,
                    num_inference_steps=params['steps'],
                    generator=generator,
                    max_sequence_length=256,
                    return_dict=True
                )
            
            # Clear timeout
            signal.alarm(0)
            
            generation_time = time.time() - generation_start
            
            # Save image with descriptive filename
            output_path = Path(generate_filename(params['prompt']))
            result.images[0].save(output_path)
            
            print(f"\n🎉 Generation Complete!")
            print(f"⏱️  Total time: {generation_time:.1f} seconds")
            print(f"💾 Saved as: {output_path.absolute()}")
            print(f"📐 Size: {params['width']}x{params['height']}")
            
            # Ask if user wants to generate another
            print(f"\n🔄 Generate another image?")
            another = input("Press Enter for yes, or 'q' to quit: ").strip().lower()
            
            if another != 'q':
                # Cleanup and restart
                del pipeline
                import gc
                gc.collect()
                if torch.backends.mps.is_available():
                    try:
                        torch.mps.empty_cache()
                    except:
                        pass
                
                # Restart the process
                print("\n" + "🔄" * 20)
                main()
            else:
                print("\n👋 Thanks for using FLUX.1 Krea Studio!")
                
        except TimeoutError:
            signal.alarm(0)
            print(f"\n⏰ Generation timed out after {params['timeout']} minutes!")
            print("💡 Try reducing image size or steps for faster generation")
            
    except KeyboardInterrupt:
        print(f"\n⏸️  Generation cancelled by user")
        signal.alarm(0)
        
    except Exception as e:
        signal.alarm(0)
        error_msg = str(e)
        print(f"\n❌ Error: {error_msg}")
        
        if "gated" in error_msg.lower():
            print("\n🔒 SOLUTION - Repository Access Required:")
            print("1. Visit: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev")
            print("2. Click 'Request access to this repository'")
            print("3. Accept license agreement")
            print("4. Set HF_TOKEN environment variable")
            
        elif "401" in error_msg or "403" in error_msg:
            print("\n🔑 SOLUTION - HuggingFace Token:")
            print("1. Go to: https://huggingface.co/settings/tokens")
            print("2. Create token with 'Read' permissions")
            print("3. Export: export HF_TOKEN='your_token_here'")

if __name__ == "__main__":
    main()