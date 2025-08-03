#!/bin/bash
"""
Restart FLUX.1 Krea [dev] Web Interface
Kills old processes and starts the official implementation
"""

echo "🔄 Restarting FLUX.1 Krea [dev] Web Interface..."
echo "============================================="

# Kill any existing FLUX processes
echo "🛑 Stopping existing FLUX processes..."
pkill -f "python.*flux" 2>/dev/null || echo "   No FLUX processes found"
lsof -ti:7860 | xargs kill -9 2>/dev/null || echo "   Port 7860 already free"

echo "⏱️  Waiting 2 seconds..."
sleep 2

# Set the HuggingFace token (replace with your actual token)
export HF_TOKEN="your_huggingface_token_here"

echo "🚀 Starting FLUX.1 Krea [dev] Official Web Interface..."
echo "🌐 Interface will be available at: http://localhost:7860"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Start the official web interface
python flux_web_ui_official.py