#!/usr/bin/env python3
"""
FLUX.1 Krea Model Validation Utilities
Validates model availability and provides helpful error messages
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download, repo_exists
from flux_exceptions import ModelNotFoundError, AuthenticationError
from flux_auth import auth_manager

class ModelValidator:
    """Validates FLUX model availability and provides setup guidance"""
    
    def __init__(self):
        self.local_model_dir = Path("./models/FLUX.1-Krea-dev")
        self.hf_repo_id = "black-forest-labs/FLUX.1-Krea-dev"
        self.required_files = [
            "flux1-krea-dev.safetensors",
            "ae.safetensors"
        ]
    
    def check_local_models(self):
        """Check if models exist locally or in HF cache"""
        
        # Check local directory first
        if self.local_model_dir.exists():
            missing_files = []
            for file in self.required_files:
                if not (self.local_model_dir / file).exists():
                    missing_files.append(file)
            
            if not missing_files:
                return True, "Local models found"
        
        # Check HuggingFace cache
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        cache_patterns = [
            "models--black-forest-labs--FLUX.1-Krea-dev",
            "*FLUX.1-Krea-dev*"
        ]
        
        for pattern in cache_patterns:
            cache_dirs = list(hf_cache.glob(pattern))
            for cache_dir in cache_dirs:
                snapshots_dir = cache_dir / "snapshots"
                if snapshots_dir.exists():
                    for snapshot in snapshots_dir.iterdir():
                        if snapshot.is_dir():
                            # Check if required files exist in this snapshot
                            has_all_files = True
                            for file in self.required_files:
                                if not (snapshot / file).exists():
                                    has_all_files = False
                                    break
                            
                            if has_all_files:
                                return True, f"Models found in HuggingFace cache: {snapshot}"
        
        return False, f"Missing model files: {self.required_files}"
    
    def check_hf_availability(self):
        """Check if models are available on HuggingFace"""
        try:
            if repo_exists(self.hf_repo_id):
                return True, "HuggingFace repository accessible"
            else:
                return False, "HuggingFace repository not found"
        except Exception as e:
            return False, f"HuggingFace check failed: {e}"
    
    def check_hf_authentication(self):
        """Check HuggingFace authentication status using auth manager"""
        if not auth_manager.is_authenticated():
            return False, "HuggingFace authentication required"
        
        # Check model-specific access
        access_ok, access_msg = auth_manager.check_model_access(self.hf_repo_id)
        return access_ok, access_msg
    
    def validate_model_setup(self, model_name="flux-krea-dev"):
        """Comprehensive model validation with helpful guidance"""
        print(f"🔍 Validating FLUX model setup for '{model_name}'...")
        
        # Check local models first
        local_ok, local_msg = self.check_local_models()
        print(f"📁 Local models: {'✅' if local_ok else '❌'} {local_msg}")
        
        if local_ok:
            return True, "Models ready"
        
        # Check HuggingFace availability
        hf_ok, hf_msg = self.check_hf_availability()
        print(f"🌐 HuggingFace repo: {'✅' if hf_ok else '❌'} {hf_msg}")
        
        if not hf_ok:
            error_msg = f"Models not available locally or on HuggingFace: {hf_msg}"
            return False, error_msg
        
        # Check authentication
        auth_ok, auth_msg = self.check_hf_authentication()
        print(f"🔑 Authentication: {'✅' if auth_ok else '❌'} {auth_msg}")
        
        if not auth_ok:
            guidance = self._get_auth_guidance()
            return False, f"{auth_msg}\n\n{guidance}"
        
        return True, "Models can be downloaded from HuggingFace"
    
    def _get_auth_guidance(self):
        """Provide authentication setup guidance"""
        return """
💡 To set up HuggingFace authentication:

1. Get a token at: https://huggingface.co/settings/tokens
2. Ensure token has 'Public Gated Repositories' permission
3. Login using one of these methods:

   Method 1 - Command line:
   huggingface-cli login --token YOUR_TOKEN

   Method 2 - Environment variable:
   export HUGGINGFACE_HUB_TOKEN="YOUR_TOKEN"

   Method 3 - In Python:
   from huggingface_hub import login
   login("YOUR_TOKEN")
"""
    
    def get_model_download_guidance(self):
        """Provide model download guidance"""
        return f"""
📥 To download FLUX.1 Krea models:

1. Ensure HuggingFace authentication is set up
2. Models will auto-download on first use, or manually run:

   python -c "
   from huggingface_hub import snapshot_download
   snapshot_download('{self.hf_repo_id}', local_dir='./models/FLUX.1-Krea-dev')
   "

3. Models will be saved to: {self.local_model_dir}
4. Total download size: ~24GB
"""

def validate_before_loading(model_name="flux-krea-dev"):
    """Decorator to validate models before loading"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            validator = ModelValidator()
            is_valid, message = validator.validate_model_setup(model_name)
            
            if not is_valid:
                if "authentication" in message.lower():
                    raise AuthenticationError(message)
                else:
                    raise ModelNotFoundError(model_name, message)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator