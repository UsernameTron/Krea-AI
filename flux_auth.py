#!/usr/bin/env python3
"""
FLUX.1 Krea Authentication Manager
Standardized HuggingFace authentication handling
"""

import os
from huggingface_hub import login, whoami, HfApi
from flux_exceptions import AuthenticationError

class FluxAuthManager:
    """Manages HuggingFace authentication for FLUX models"""
    
    def __init__(self):
        self.api = HfApi()
        self._is_authenticated = None
        self._user_info = None
    
    def is_authenticated(self):
        """Check if user is authenticated with HuggingFace"""
        if self._is_authenticated is None:
            try:
                self._user_info = whoami()
                self._is_authenticated = True
                print(f"✅ Authenticated as: {self._user_info.get('name', 'Unknown')}")
            except Exception:
                self._is_authenticated = False
        
        return self._is_authenticated
    
    def login_with_token(self, token):
        """Login with HuggingFace token"""
        try:
            login(token=token, add_to_git_credential=False)
            self._is_authenticated = None  # Reset cache
            if self.is_authenticated():
                return True, "Successfully authenticated with HuggingFace"
            else:
                return False, "Authentication failed - token may be invalid"
        except Exception as e:
            return False, f"Authentication error: {str(e)}"
    
    def auto_authenticate(self):
        """Try to authenticate using available methods"""
        # Method 1: Check if already authenticated
        if self.is_authenticated():
            return True, f"Already authenticated as {self._user_info.get('name', 'user')}"
        
        # Method 2: Try environment variable
        token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
        if token:
            success, message = self.login_with_token(token)
            if success:
                return True, f"Authenticated using environment token: {message}"
        
        # Method 3: Check for stored token
        try:
            # This will use any stored token from previous login
            if self.is_authenticated():
                return True, "Using stored authentication token"
        except:
            pass
        
        return False, "No valid authentication found"
    
    def get_auth_instructions(self):
        """Get user-friendly authentication instructions"""
        return """
🔑 HuggingFace Authentication Required

To access FLUX.1 Krea models, you need to authenticate with HuggingFace:

📋 Steps:
1. Go to: https://huggingface.co/settings/tokens
2. Create a new token with 'Read' permissions
3. Enable 'Public Gated Repositories' if available
4. Authenticate using one of these methods:

   🖥️  Command Line:
   huggingface-cli login --token YOUR_TOKEN

   🐍 Python Code:
   from huggingface_hub import login
   login("YOUR_TOKEN")

   🌍 Environment Variable:
   export HUGGINGFACE_HUB_TOKEN="YOUR_TOKEN"

5. Restart the FLUX application

💡 The token will be saved securely for future use.
"""
    
    def check_model_access(self, repo_id="black-forest-labs/FLUX.1-Krea-dev"):
        """Check if user has access to specific model"""
        if not self.is_authenticated():
            return False, "Not authenticated with HuggingFace"
        
        try:
            # Try to get repo info
            repo_info = self.api.repo_info(repo_id)
            return True, f"Access confirmed to {repo_id}"
        except Exception as e:
            if "401" in str(e) or "403" in str(e):
                return False, f"Access denied to {repo_id}. Please check token permissions."
            return False, f"Error checking access: {str(e)}"
    
    def ensure_authenticated(self):
        """Ensure authentication or raise error with guidance"""
        success, message = self.auto_authenticate()
        if not success:
            instructions = self.get_auth_instructions()
            raise AuthenticationError(f"{message}\n\n{instructions}")
        
        # Check model access
        access_ok, access_msg = self.check_model_access()
        if not access_ok:
            raise AuthenticationError(f"Model access failed: {access_msg}")
        
        return True

# Global auth manager instance
auth_manager = FluxAuthManager()

def ensure_authenticated():
    """Decorator to ensure authentication before function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            auth_manager.ensure_authenticated()
            return func(*args, **kwargs)
        return wrapper
    return decorator

def check_auth_status():
    """Quick authentication status check"""
    return auth_manager.is_authenticated()

def setup_auth_interactive():
    """Interactive authentication setup"""
    print("🔑 FLUX.1 Krea Authentication Setup")
    print("=" * 40)
    
    # Check current status
    if auth_manager.is_authenticated():
        user_info = auth_manager._user_info
        print(f"✅ Already authenticated as: {user_info.get('name', 'Unknown')}")
        
        # Check model access
        access_ok, access_msg = auth_manager.check_model_access()
        print(f"🎯 Model access: {'✅' if access_ok else '❌'} {access_msg}")
        
        if access_ok:
            print("🎉 Authentication setup complete!")
            return True
    else:
        print("❌ Not currently authenticated")
    
    # Try auto-authentication
    success, message = auth_manager.auto_authenticate()
    if success:
        print(f"✅ {message}")
        return True
    
    # Show instructions
    print(auth_manager.get_auth_instructions())
    return False

if __name__ == "__main__":
    setup_auth_interactive()