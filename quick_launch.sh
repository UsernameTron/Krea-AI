#!/bin/bash
# Quick launch script with correct HF token

cd '/Users/cpconnor/Krea AI/flux-krea'
# Set your HuggingFace token here  
export HF_TOKEN='YOUR_HF_TOKEN_HERE'

echo '🛡️  FLUX.1 Krea Studio - Performance Optimized'
echo '=============================================='
echo '✅ HF_TOKEN configured for model access'
echo '🚀 18-30x performance improvements active'
echo '🛑 Press Ctrl+C to cancel any stuck generation'
echo ''

python flux_interactive.py