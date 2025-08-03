#!/bin/bash
"""
FLUX.1 Krea Web GUI Desktop Launcher
Launches the web interface and opens browser automatically
"""

# Set the project directory
PROJECT_DIR="/Users/cpconnor/Krea AI/flux-krea"

# Function to show status messages
show_status() {
    echo "🎨 FLUX.1 Krea Web GUI Launcher"
    echo "================================"
    echo "$1"
}

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    show_status "❌ Error: Project directory not found at $PROJECT_DIR"
    echo "Please update the PROJECT_DIR path in this script"
    read -p "Press Enter to exit..."
    exit 1
fi

# Change to project directory
cd "$PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d "flux_env" ]; then
    show_status "❌ Error: Virtual environment 'flux_env' not found"
    echo "Please run the setup first or check the installation"
    read -p "Press Enter to exit..."
    exit 1
fi

show_status "🚀 Starting FLUX.1 Krea Web Interface..."
echo "📁 Project Directory: $PROJECT_DIR"
echo "🐍 Activating virtual environment..."

# Activate virtual environment and launch
source flux_env/bin/activate

# Check if required files exist
if [ ! -f "flux_web_ui.py" ]; then
    show_status "❌ Error: flux_web_ui.py not found"
    echo "Please ensure all files are properly installed"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "🌐 Starting web server..."
echo "📝 The interface will open in your default browser"
echo "🔗 URL: http://localhost:7860"
echo ""
echo "⚠️  First generation may take time to load models"
echo "🛑 Press Ctrl+C in this window to stop the server"
echo ""

# Wait a moment then open browser
sleep 3 && open "http://localhost:7860" &

# Launch the web interface
python launch_web_ui.py

echo ""
echo "👋 FLUX.1 Krea Web GUI stopped"
read -p "Press Enter to close..."