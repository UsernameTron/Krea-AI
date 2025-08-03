#!/usr/bin/env python3
"""
Simple test script to verify local models are working
"""

import torch
import os

# Disable torch compile to avoid MPS issues
torch._dynamo.config.suppress_errors = True
os.environ['TORCHDYNAMO_DISABLE'] = '1'

from src.flux.util import load_ae, load_flow_model, load_clip, load_t5

def test_local_models():
    device = "cpu"  # Use CPU to avoid MPS compilation issues
    
    print("🧪 Testing local model loading...")
    
    try:
        print("📥 Loading autoencoder...")
        ae = load_ae("flux-krea-dev", device=device)
        print("✅ Autoencoder loaded successfully")
        
        print("📥 Loading CLIP...")
        clip = load_clip(device=device)
        print("✅ CLIP loaded successfully")
        
        print("📥 Loading T5...")
        t5 = load_t5(device=device)
        print("✅ T5 loaded successfully")
        
        print("📥 Loading FLUX model...")
        model = load_flow_model("flux-krea-dev", device=device)
        print("✅ FLUX model loaded successfully")
        
        print("\n🎉 All models loaded successfully from local files!")
        print("✅ Local model setup is working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_local_models()
    if success:
        print("\n🏆 SUCCESS: Local models are configured and loading properly!")
        print("The 'Model not found' error has been resolved.")
    else:
        print("\n💥 FAILED: There are still issues with model loading.")