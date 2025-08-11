#!/usr/bin/env python3
"""
FLUX.1 Krea [dev] - Memory-Conservative Interactive Version
Optimized for systems with tight MPS memory limits (25GB)
"""

import torch
import argparse
import time
import signal
import sys
import os
from pathlib import Path
from diffusers import FluxPipeline

# Set very conservative MPS environment variables
os.environ.setdefault('PYTORCH_MPS_MEMORY_FRACTION', '0.75')  # Conservative 75%
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')  # Disable limit
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.5')
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
    print("   FLUX.1 Krea [dev] - Memory-Conservative Studio")
    print("     🛡️  Timeout Protected  •  💾 Memory Optimized")
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
    
    # Memory-conservative settings only
    print("\n⚙️  Choose settings (Memory-Conservative):")
    print("1. 🚀 Quick (512x512, 20 steps) - ~1.5 minutes, ~8GB")
    print("2. 🎯 Standard (768x768, 25 steps) - ~3 minutes, ~15GB") 
    print("3. 🎨 Large (1024x1024, 28 steps) - ~5 minutes, ~20GB")
    print("4. ⚙️  Custom settings")
    
    while True:
        choice = input("\nSelect option (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            break
        print("❌ Please enter 1, 2, 3, or 4!")
    
    if choice == '1':
        width, height, steps = 512, 512, 20
    elif choice == '2':
        width, height, steps = 768, 768, 25
    elif choice == '3':
        width, height, steps = 1024, 1024, 28
    else:
        # Custom settings with memory warnings
        print("\n⚙️  Custom Settings (Memory Limited):")
        width = int(input("   Width (512-1024 recommended): ") or "768")
        height = int(input("   Height (512-1024 recommended): ") or "768")
        steps = int(input("   Steps (15-30 recommended): ") or "25")
        
        # Memory warning for large sizes
        if width * height > 1024 * 1024:
            print("⚠️  Large size may cause memory issues!")
    
    # Seed
    seed_input = input(f"\n🎲 Seed (press Enter for random): ").strip()
    seed = int(seed_input) if seed_input else None
    
    # Timeout
    timeout_input = input("⏰ Timeout minutes (default 10): ").strip()
    timeout = int(timeout_input) if timeout_input else 10
    
    return {
        'prompt': prompt,
        'width': width,
        'height': height,
        'steps': steps,
        'seed': seed,
        'timeout': timeout
    }

def _apply_conservative_optimizations(pipeline):
    """Apply memory-conservative optimizations"""
    optimizations_applied = []
    
    # Aggressive memory management
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # Enable CPU offloading (essential for memory conservation)
    try:
        pipeline.enable_model_cpu_offload()
        optimizations_applied.append("CPU offload")
    except:
        try:
            pipeline.enable_sequential_cpu_offload()
            optimizations_applied.append("Sequential CPU offload")
        except Exception as e:
            print(f"⚠️  Warning: Could not enable CPU offload: {e}")
            optimizations_applied.append("No CPU offload")
    
    # Enable all memory-saving features
    try:
        pipeline.enable_attention_slicing("max")  # Maximum slicing
        optimizations_applied.append("Max attention slicing")
    except:
        optimizations_applied.append("No attention slicing")
    
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
    
    # Final cache clear
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def generate_filename(prompt):
    """Generate descriptive filename from prompt"""
    # Clean prompt for filename
    clean_prompt = ''.join(c if c.isalnum() or c.isspace() else '' for c in prompt)
    clean_prompt = '_'.join(clean_prompt.split()[:5])  # First 5 words
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"flux_conservative_{clean_prompt}_{timestamp}.png"

def main():
    # Clear screen and show banner
    clear_screen()
    print_banner()
    
    # Check environment
    print("🔍 Environment Check:")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   MPS Available: {'✅' if torch.backends.mps.is_available() else '❌'}")
    
    # Memory info
    if torch.backends.mps.is_available():
        print(f"   MPS Memory Fraction: {os.environ.get('PYTORCH_MPS_MEMORY_FRACTION', 'default')}")
    
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
        
        print(f"\n📥 Loading FLUX.1 Krea [dev] pipeline (Memory-Conservative)...")
        load_start = time.time()
        
        # Aggressive cache clearing
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        # Load pipeline with maximum memory conservation
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Krea-dev",
            torch_dtype=torch.float16,  # fp16 for memory efficiency
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        
        # Apply conservative optimizations BEFORE moving to device
        _apply_conservative_optimizations(pipeline)
        
        load_time = time.time() - load_start
        print(f"✅ Pipeline loaded in {load_time:.1f} seconds")
        
        print(f"\n🖼️  Generating image (Memory-Conservative Mode)...")
        print(f"⏰ Timeout: {params['timeout']} minutes")
        print("🛑 Press Ctrl+C to cancel if needed")
        
        generation_start = time.time()
        
        # Set up timeout protection
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(params['timeout'] * 60)
        
        try:
            # Set up generator
            generator = torch.Generator().manual_seed(params['seed']) if params['seed'] else None
            
            # Clear cache before generation
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # Generate with timeout protection and memory management
            with torch.inference_mode():
                result = pipeline(
                    params['prompt'],
                    height=params['height'],
                    width=params['width'],
                    guidance_scale=3.5,  # Lower guidance for memory
                    num_inference_steps=params['steps'],
                    generator=generator,
                    max_sequence_length=256,  # Extended token limit
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
            
            # Memory cleanup
            del result
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
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
        
        if "out of memory" in error_msg.lower():
            print("\n💾 MEMORY SOLUTION:")
            print("1. Try smaller image size (512x512)")
            print("2. Reduce steps (15-20)")
            print("3. Close other applications")
            print("4. Use: export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0")
            
        elif "gated" in error_msg.lower():
            print("\n🔒 SOLUTION - Repository Access Required:")
            print("1. Visit: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev")
            print("2. Click 'Request access to this repository'")
            print("3. Accept license agreement")
            print("4. Set HF_TOKEN environment variable")

if __name__ == "__main__":
    main()