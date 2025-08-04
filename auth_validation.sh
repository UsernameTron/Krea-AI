#!/bin/bash
# HuggingFace Authentication Validation Script for FLUX.1 Krea
# Validates HF_TOKEN setup and model access

echo "🔑 HuggingFace Authentication Validation for FLUX.1 Krea"
echo "=========================================================="

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN environment variable is not set"
    echo ""
    echo "To fix this:"
    echo "1. Go to https://huggingface.co/settings/tokens"
    echo "2. Create a new token with 'Read' permissions"
    echo "3. Set it in your environment:"
    echo "   export HF_TOKEN=your_token_here"
    echo "4. Add to your shell profile (.bashrc, .zshrc, etc.):"
    echo "   echo 'export HF_TOKEN=your_token_here' >> ~/.bashrc"
    echo ""
    exit 1
fi

echo "✅ HF_TOKEN environment variable is set"

# Validate token format
if [[ ! $HF_TOKEN =~ ^hf_[a-zA-Z0-9]{34}$ ]]; then
    echo "⚠️  HF_TOKEN format appears invalid (should start with 'hf_' and be 37 characters)"
    echo "   Current token: ${HF_TOKEN:0:10}..."
    echo ""
fi

# Test authentication with HuggingFace Hub
echo "🔍 Testing HuggingFace Hub authentication..."

# Use Python to test the authentication
python3 -c "
import sys
try:
    from huggingface_hub import HfApi
    api = HfApi()
    user_info = api.whoami()
    print(f'✅ Successfully authenticated as: {user_info[\"name\"]}')
except Exception as e:
    print(f'❌ Authentication failed: {e}')
    print('')
    print('To fix this:')
    print('1. Verify your token at: https://huggingface.co/settings/tokens')
    print('2. Make sure the token has \"Read\" permissions')
    print('3. Reset your token if needed: export HF_TOKEN=new_token')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

# Test access to FLUX.1 Krea model
echo "🎯 Testing access to FLUX.1 Krea model..."

python3 -c "
import sys
try:
    from huggingface_hub import HfApi
    api = HfApi()
    
    model_id = 'black-forest-labs/FLUX.1-Krea-dev'
    try:
        model_info = api.model_info(model_id)
        print(f'✅ Successfully accessed model: {model_id}')
        print(f'   Model size: ~{model_info.safetensors[\"total\"] // (1024**3)}GB' if hasattr(model_info, 'safetensors') and model_info.safetensors else '')
    except Exception as e:
        error_str = str(e).lower()
        if 'gated' in error_str or 'access' in error_str:
            print(f'❌ Model access denied: {e}')
            print('')
            print('To fix this:')
            print(f'1. Go to https://huggingface.co/{model_id}')
            print('2. Click \"Request access to this model\"')
            print('3. Fill out the access request form')
            print('4. Wait for approval (usually within 24 hours)')
            print('5. Re-run this script after approval')
        else:
            print(f'❌ Error accessing model: {e}')
        sys.exit(1)
        
except ImportError:
    print('❌ huggingface_hub not installed')
    print('   Install with: pip install huggingface_hub')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

# Check available disk space
echo "💾 Checking disk space for model download..."

available_space=$(df . | tail -1 | awk '{print $4}')
available_gb=$((available_space / 1024 / 1024))

if [ $available_gb -lt 25 ]; then
    echo "⚠️  Low disk space: ${available_gb}GB available"
    echo "   FLUX.1 Krea requires ~24GB for download"
    echo "   Consider freeing up space before running inference"
else
    echo "✅ Sufficient disk space: ${available_gb}GB available"
fi

# Test huggingface-cli login status
echo "🔧 Checking huggingface-cli login status..."

if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli whoami >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ huggingface-cli is properly logged in"
    else
        echo "⚠️  huggingface-cli not logged in"
        echo "   Run: huggingface-cli login"
        echo "   Or set HF_TOKEN environment variable (already done)"
    fi
else
    echo "ℹ️  huggingface-cli not installed (optional)"
    echo "   Install with: pip install huggingface_hub[cli]"
fi

echo ""
echo "🎉 Authentication validation complete!"
echo ""
echo "✅ You're ready to run FLUX.1 Krea inference"
echo "   Try: python inference_m4_optimized.py --prompt 'a cute cat' --seed 42"
echo ""