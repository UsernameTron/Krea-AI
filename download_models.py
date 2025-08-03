#!/usr/bin/env python3
"""
Download FLUX model files locally
"""

import os
from huggingface_hub import hf_hub_download

def download_flux_models():
    """Download FLUX model files to local directory"""
    model_dir = "./models/FLUX.1-Krea-dev"
    os.makedirs(model_dir, exist_ok=True)
    
    repo_id = "black-forest-labs/FLUX.1-Krea-dev"
    
    files_to_download = [
        "flux1-krea-dev.safetensors",
        "ae.safetensors"
    ]
    
    for filename in files_to_download:
        local_path = os.path.join(model_dir, filename)
        
        if os.path.exists(local_path):
            print(f"✅ {filename} already exists locally")
            continue
            
        try:
            print(f"📥 Downloading {filename}...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=model_dir,
                local_dir_use_symlinks=False
            )
            print(f"✅ Downloaded {filename} to {downloaded_path}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")

if __name__ == "__main__":
    download_flux_models()