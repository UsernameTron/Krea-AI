#!/bin/bash
# Quick fix for HuggingFace token authentication

echo "🔧 FLUX.1 Krea HuggingFace Token Fix"
echo "=================================="

# Set the correct token (replace with your actual token)
export HF_TOKEN='YOUR_HF_TOKEN_HERE'

# Verify token is set
echo "✅ HF_TOKEN set to: ${HF_TOKEN:0:6}...${HF_TOKEN: -6}"

# Test authentication
echo "🔍 Testing HuggingFace authentication..."
python3 -c "
import os
from huggingface_hub import HfApi
try:
    api = HfApi(token=os.environ.get('HF_TOKEN'))
    user = api.whoami()
    print(f'✅ Authentication successful!')
    print(f'   Username: {user.get(\"name\", \"Unknown\")}')
except Exception as e:
    print(f'❌ Authentication failed: {e}')
    print('🔧 Solutions:')
    print('   1. Check token is correct')
    print('   2. Visit: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev')
    print('   3. Click \"Request access to this repository\"')
    print('   4. Accept license agreement')
"

echo ""
echo "🚀 To run FLUX with proper authentication:"
echo "source fix_hf_token.sh"
echo "python flux_performance_fix.py --prompt 'a cute cat' --steps 20"
echo ""
echo "Or for interactive mode:"
echo "python flux_interactive.py"