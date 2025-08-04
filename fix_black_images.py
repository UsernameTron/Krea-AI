#!/usr/bin/env python3
"""
Fix Black Image Issue for FLUX.1 Krea on MPS
Addresses VAE decoding and numerical stability problems
"""

import torch
import os
from diffusers import FluxPipeline
from pathlib import Path

# Conservative settings to prevent black images
os.environ['PYTORCH_MPS_MEMORY_FRACTION'] = '0.8'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def fix_black_images_test():
    """Test fixes for black image generation"""
    print("🔍 Diagnosing and Fixing Black Image Issue")
    print("=" * 50)
    
    # Clear cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    print("📥 Loading pipeline with black image fixes...")
    
    # Load pipeline with specific fixes for black images
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Krea-dev",
        torch_dtype=torch.bfloat16,  # Use bfloat16 instead of float16 for MPS stability
        use_safetensors=True,
        low_cpu_mem_usage=True
    )
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    # Apply conservative optimizations (avoid aggressive slicing that can cause black images)
    try:
        pipeline.enable_model_cpu_offload()
        print("✅ CPU offload enabled")
    except:
        print("⚠️  CPU offload not available")
    
    # Be more conservative with slicing to avoid black images
    try:
        # Use less aggressive slicing
        pipeline.enable_attention_slicing(1)  # Minimal slicing
        print("✅ Conservative attention slicing enabled")
    except:
        print("⚠️  Attention slicing not available")
    
    # VAE slicing can sometimes cause black images - test without it first
    print("ℹ️  Skipping VAE slicing to prevent black images")
    
    print(f"\n🖼️  Testing generation with black image fixes...")
    
    try:
        # Clear cache before generation
        if device == "mps":
            torch.mps.empty_cache()
        
        # Test with a simple prompt and conservative settings
        prompt = "a cute cat"
        
        # Use more conservative generation parameters
        with torch.inference_mode():
            result = pipeline(
                prompt,
                height=512,
                width=512,
                guidance_scale=4.0,  # Conservative guidance
                num_inference_steps=20,  # Fewer steps for testing
                generator=torch.Generator(device=device).manual_seed(42),
                max_sequence_length=128  # Reduced sequence length
            )
        
        # Save and verify the result
        output_path = "black_image_fix_test.png"
        result.images[0].save(output_path)
        
        # Check if image is actually generated (not black)
        import numpy as np
        from PIL import Image
        
        img = Image.open(output_path)
        img_array = np.array(img)
        
        # Check if image is not all black/dark
        mean_brightness = np.mean(img_array)
        
        print(f"✅ Image generated: {output_path}")
        print(f"📊 Mean brightness: {mean_brightness:.1f} (0=black, 255=white)")
        
        if mean_brightness < 10:
            print("❌ Image is still black/very dark")
            return False
        elif mean_brightness < 50:
            print("⚠️  Image is quite dark but has some content")
            return True
        else:
            print("✅ Image has good brightness and content")
            return True
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

def create_black_image_fix_script():
    """Create a script with the fixes for black images"""
    
    fix_script = """#!/usr/bin/env python3
'''
FLUX.1 Krea Black Image Fix
Use this version if you're getting black images
'''

import torch
import os
from diffusers import FluxPipeline
import argparse

# Environment fixes for black images
os.environ['PYTORCH_MPS_MEMORY_FRACTION'] = '0.8'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def main():
    parser = argparse.ArgumentParser(description='FLUX.1 Krea - Black Image Fix')
    parser.add_argument('--prompt', required=True, help='Text prompt')
    parser.add_argument('--output', default='fixed_output.png', help='Output filename')
    parser.add_argument('--width', type=int, default=512, help='Width (512 recommended)')
    parser.add_argument('--height', type=int, default=512, help='Height (512 recommended)')
    parser.add_argument('--steps', type=int, default=20, help='Steps (20 recommended)')
    parser.add_argument('--guidance', type=float, default=4.0, help='Guidance scale')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    args = parser.parse_args()
    
    print("🔧 FLUX.1 Krea - Black Image Fix Mode")
    print("=" * 40)
    
    # Clear cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    print("📥 Loading pipeline with black image fixes...")
    
    # Use bfloat16 for better MPS stability
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Krea-dev",
        torch_dtype=torch.bfloat16,  # Key fix: bfloat16 instead of float16
        use_safetensors=True,
        low_cpu_mem_usage=True
    )
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    # Conservative optimizations to prevent black images
    try:
        pipeline.enable_model_cpu_offload()
        print("✅ CPU offload enabled")
    except:
        pass
    
    try:
        pipeline.enable_attention_slicing(1)  # Minimal slicing
        print("✅ Conservative attention slicing enabled")
    except:
        pass
    
    # Skip VAE optimizations that can cause black images
    print("ℹ️  Using conservative VAE settings")
    
    print(f"\\n🖼️  Generating: {args.prompt}")
    print(f"📐 Size: {args.width}x{args.height}")
    print(f"🎯 Steps: {args.steps}, Guidance: {args.guidance}")
    
    # Clear cache before generation
    if device == "mps":
        torch.mps.empty_cache()
    
    # Generate with conservative settings
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed else None
    
    with torch.inference_mode():
        result = pipeline(
            args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            generator=generator,
            max_sequence_length=128
        )
    
    # Save result
    result.images[0].save(args.output)
    print(f"✅ Image saved: {args.output}")
    
    # Verify image is not black
    import numpy as np
    from PIL import Image
    
    img = Image.open(args.output)
    img_array = np.array(img)
    mean_brightness = np.mean(img_array)
    
    print(f"📊 Image brightness: {mean_brightness:.1f}/255")
    
    if mean_brightness < 10:
        print("⚠️  Warning: Image appears to be black/very dark")
        print("💡 Try: Increase guidance scale or reduce image size")
    else:
        print("✅ Image generated successfully!")

if __name__ == "__main__":
    main()
"""
    
    with open("flux_black_image_fix.py", "w") as f:
        f.write(fix_script)
    
    os.chmod("flux_black_image_fix.py", 0o755)
    print("✅ Created: flux_black_image_fix.py")

if __name__ == "__main__":
    print("🔧 FLUX.1 Krea Black Image Diagnostic & Fix")
    print("=" * 45)
    
    # Test the fixes  
    success = fix_black_images_test()
    
    if success:
        print(f"\n✅ Black image fix successful!")
        print(f"💡 The key fixes are:")
        print(f"   1. Use torch.bfloat16 instead of torch.float16")
        print(f"   2. Conservative attention slicing (1 instead of 'auto')")
        print(f"   3. Skip aggressive VAE optimizations")
        print(f"   4. Conservative generation parameters")
    else:
        print(f"\n❌ Still getting black images")
        print(f"💡 Additional things to try:")
        print(f"   1. Use CPU mode: device='cpu'")
        print(f"   2. Try different guidance scales (2.0-6.0)")
        print(f"   3. Increase inference steps (25-30)")
        print(f"   4. Try different seeds")
    
    # Create the fix script regardless
    create_black_image_fix_script()
    
    print(f"\n🎯 Usage:")
    print(f"python flux_black_image_fix.py --prompt 'a cute cat' --output test.png")