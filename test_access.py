#!/usr/bin/env python3
"""
Test script to verify model access and troubleshoot issues
"""

import os
import torch
from diffusers import FluxPipeline
from huggingface_hub import HfApi

def test_model_access():
    print("🔍 FLUX.1 Krea Access Test")
    print("="*40)
    
    # Test 1: Repository access
    print("\n1. Testing repository access...")
    try:
        api = HfApi()
        repo_info = api.repo_info('black-forest-labs/FLUX.1-Krea-dev')
        print(f"✅ Repository accessible: {repo_info.id}")
    except Exception as e:
        print(f"❌ Repository access failed: {e}")
        return
    
    # Test 2: File listing
    print("\n2. Testing file access...")
    try:
        files = api.list_repo_files('black-forest-labs/FLUX.1-Krea-dev')
        print(f"✅ Found {len(files)} files")
        
        # Look for essential files
        essential_files = ['model_index.json', 'scheduler/scheduler_config.json']
        found_files = []
        missing_files = []
        
        for essential in essential_files:
            if essential in files:
                found_files.append(essential)
            else:
                missing_files.append(essential)
        
        if found_files:
            print(f"✅ Found essential files: {found_files}")
        if missing_files:
            print(f"⚠️  Missing essential files: {missing_files}")
            
    except Exception as e:
        print(f"❌ File access failed: {e}")
        return
    
    # Test 3: Direct diffusers loading
    print("\n3. Testing diffusers model loading...")
    try:
        print("   Attempting to load model directly...")
        pipeline = FluxPipeline.from_pretrained(
            'black-forest-labs/FLUX.1-Krea-dev',
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        )
        print("✅ Model loaded successfully!")
        
        # Test a simple generation
        print("   Testing image generation...")
        image = pipeline(
            "a simple test image",
            num_inference_steps=4,  # Very fast test
            height=512,
            width=512
        ).images[0]
        
        image.save("access_test.png")
        print("✅ Test generation successful! Saved as access_test.png")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        
        # Additional diagnostics
        if "404" in str(e):
            print("   → Model or files not found")
        elif "403" in str(e) or "unauthorized" in str(e).lower():
            print("   → Authentication/permission issue")
        elif "connection" in str(e).lower():
            print("   → Network connectivity issue")
        else:
            print("   → Unknown error, possibly model configuration issue")
    
    print("\n" + "="*40)
    print("Test complete!")

if __name__ == "__main__":
    test_model_access()