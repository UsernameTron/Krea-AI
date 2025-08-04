#!/usr/bin/env python3
"""
Test FLUX pipeline loading to diagnose issues
"""

import os
import torch
from diffusers import FluxPipeline

print("🔍 FLUX Pipeline Loading Test")
print("=" * 40)

# Check environment
print(f"Python: {torch.__version__}")
print(f"HF_TOKEN set: {'Yes' if os.getenv('HF_TOKEN') else 'No'}")
print(f"MPS available: {torch.backends.mps.is_available()}")

try:
    print("\n📥 Attempting to load FLUX.1 Krea [dev]...")
    
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Krea-dev",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        low_cpu_mem_usage=True
    )
    
    print("✅ Pipeline loaded successfully!")
    print("🧪 Testing basic functionality...")
    
    # Quick test
    test_prompt = "a red apple"
    print(f"🖼️  Testing with: '{test_prompt}'")
    
    with torch.inference_mode():
        image = pipeline(
            test_prompt,
            height=512,
            width=512,
            num_inference_steps=10,  # Very few steps for quick test
            guidance_scale=4.5
        ).images[0]
    
    print("✅ Generation test successful!")
    image.save("pipeline_test.png")
    print("💾 Saved test image: pipeline_test.png")
    
except Exception as e:
    print(f"❌ Error: {e}")
    
    if "gated" in str(e).lower():
        print("\n🔒 SOLUTION - Repository Access:")
        print("1. Visit: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev")
        print("2. Click 'Request access to this repository'")
        print("3. Accept license agreement")
        print("4. Wait for approval (usually within hours)")
        
    elif "401" in str(e) or "403" in str(e):
        print("\n🔑 SOLUTION - HuggingFace Token:")
        print("1. Go to: https://huggingface.co/settings/tokens")
        print("2. Create a new token with 'Read' permissions")
        print("3. Export it: export HF_TOKEN='your_token_here'")
        print("4. Restart this script")
        
    else:
        print(f"\n❓ Unknown error. Full details:")
        import traceback
        traceback.print_exc()