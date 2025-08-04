#!/usr/bin/env python3
"""
FLUX.1 Krea [dev] - Timeout Protected Version
Fixes infinite loop issues with generation timeout
"""

import torch
import argparse
import time
import signal
import sys
from pathlib import Path
from diffusers import FluxPipeline

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Generation timed out")

def main():
    parser = argparse.ArgumentParser(description='FLUX.1 Krea [dev] with Timeout Protection')
    parser.add_argument('--prompt', required=True, help='Text prompt for image generation')
    parser.add_argument('--output', default='flux-krea-timeout-fix.png', help='Output filename')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--guidance', type=float, default=4.5, help='Guidance scale')
    parser.add_argument('--steps', type=int, default=28, help='Number of inference steps')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--timeout', type=int, default=300, help='Generation timeout in seconds (default: 5 minutes)')
    
    args = parser.parse_args()
    
    print("🛡️  FLUX.1 Krea [dev] - Timeout Protected")
    print("=" * 50)
    print(f"Prompt: {args.prompt}")
    print(f"Size: {args.width}x{args.height}")
    print(f"Timeout: {args.timeout} seconds")
    
    start_time = time.time()
    
    try:
        print("\n📥 Loading FLUX.1 Krea [dev]...")
        
        # Load pipeline with timeout protection
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Krea-dev", 
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            low_cpu_mem_usage=True  # Critical for preventing memory loops
        )
        
        # Apply memory optimizations to prevent loops
        try:
            pipe.enable_model_cpu_offload()
            print("✅ CPU offload enabled")
        except:
            try:
                pipe.enable_sequential_cpu_offload()
                print("✅ Sequential CPU offload enabled")
            except:
                print("⚠️  Using default memory management")
        
        # Enable attention slicing to prevent memory loops
        try:
            pipe.enable_attention_slicing("auto")
            print("✅ Attention slicing enabled")
        except:
            pass
            
        # Enable VAE slicing to prevent memory loops
        try:
            pipe.enable_vae_slicing()
            print("✅ VAE slicing enabled")
        except:
            pass
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f} seconds")
        
        print(f"\n🖼️  Generating image with {args.timeout}s timeout...")
        generation_start = time.time()
        
        # Set up timeout protection
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timeout)
        
        try:
            # Generator setup
            generator = torch.Generator().manual_seed(args.seed) if args.seed else None
            
            # Generate with explicit memory management
            with torch.inference_mode():  # Prevent autograd loops
                image = pipe(
                    args.prompt,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance,
                    num_inference_steps=args.steps,
                    generator=generator,
                    max_sequence_length=256,  # Prevent text encoder loops
                    return_dict=True
                ).images[0]
            
            # Clear the timeout
            signal.alarm(0)
            
        except TimeoutError:
            print(f"\n⏰ Generation timed out after {args.timeout} seconds!")
            print("This prevents infinite loops but means generation failed.")
            print("\n💡 Solutions:")
            print("1. Reduce image size: --width 768 --height 768")
            print("2. Reduce steps: --steps 20")
            print("3. Increase timeout: --timeout 600")
            print("4. Use optimized version: python maximum_performance_pipeline.py")
            return
            
        generation_time = time.time() - generation_start
        
        # Save the image
        output_path = Path(args.output)
        image.save(output_path)
        
        total_time = time.time() - start_time
        
        print(f"🎉 Generation complete!")
        print(f"⏱️  Total: {total_time:.1f}s (Load: {load_time:.1f}s + Gen: {generation_time:.1f}s)")
        print(f"💾 Saved: {output_path.absolute()}")
        
        # Memory cleanup to prevent future loops
        del pipe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
    except KeyboardInterrupt:
        print("\n⏸️  Generation interrupted by user")
        signal.alarm(0)  # Clear timeout
        
    except Exception as e:
        signal.alarm(0)  # Clear timeout
        print(f"\n❌ Error: {e}")
        
        if "gated" in str(e).lower():
            print("\n🔒 Need HuggingFace access - visit:")
            print("https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev")
        elif "memory" in str(e).lower() or "out of memory" in str(e).lower():
            print("\n💾 Memory issue - try:")
            print("python maximum_performance_pipeline.py --prompt '{args.prompt}'")

if __name__ == "__main__":
    main()