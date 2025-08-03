#!/bin/bash
"""
Create Desktop Shortcut for FLUX.1 Krea Web GUI
This script creates a clickable desktop application
"""

# Define paths
PROJECT_DIR="/Users/cpconnor/Krea AI/flux-krea"
DESKTOP_DIR="$HOME/Desktop"
APP_NAME="FLUX Krea Studio"
APP_PATH="$DESKTOP_DIR/$APP_NAME.app"

echo "🎨 Creating Desktop Shortcut for FLUX.1 Krea Web GUI"
echo "=================================================="

# Create the .app bundle structure
mkdir -p "$APP_PATH/Contents/MacOS"
mkdir -p "$APP_PATH/Contents/Resources"

# Create the Info.plist file
cat > "$APP_PATH/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>FLUX Krea Studio</string>
    <key>CFBundleIdentifier</key>
    <string>com.flux.krea.studio</string>
    <key>CFBundleName</key>
    <string>FLUX Krea Studio</string>
    <key>CFBundleDisplayName</key>
    <string>FLUX Krea Studio</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>LSUIElement</key>
    <false/>
    <key>CFBundleIconFile</key>
    <string>icon.icns</string>
</dict>
</plist>
EOF

# Create the main executable script
cat > "$APP_PATH/Contents/MacOS/FLUX Krea Studio" << 'EOF'
#!/bin/bash

# Set the project directory (update this path if needed)
PROJECT_DIR="/Users/cpconnor/Krea AI/flux-krea"

# Function to show dialog
show_dialog() {
    osascript -e "display dialog \"$1\" with title \"FLUX Krea Studio\" buttons {\"OK\"} default button \"OK\""
}

# Function to show notification
show_notification() {
    osascript -e "display notification \"$1\" with title \"FLUX Krea Studio\""
}

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    show_dialog "Error: Project directory not found at $PROJECT_DIR. Please update the path in the application."
    exit 1
fi

# Change to project directory
cd "$PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d "flux_env" ]; then
    show_dialog "Error: Virtual environment 'flux_env' not found. Please run the setup first."
    exit 1
fi

# Show starting notification
show_notification "Starting FLUX Krea Studio..."

# Open Terminal and run the launcher
osascript << 'APPLESCRIPT'
tell application "Terminal"
    activate
    set currentTab to do script "cd '/Users/cpconnor/Krea AI/flux-krea' && source flux_env/bin/activate && echo '🎨 FLUX.1 Krea Studio Starting...' && echo '🌐 Web interface will open at: http://localhost:7860' && echo '⚠️  First generation may take time to load models' && echo '🛑 Press Ctrl+C to stop the server' && echo '' && sleep 3 && open 'http://localhost:7860' && python launch_web_ui.py"
    set custom title of currentTab to "FLUX Krea Studio"
end tell
APPLESCRIPT

EOF

# Make the executable script runnable
chmod +x "$APP_PATH/Contents/MacOS/FLUX Krea Studio"

# Create a simple icon (using emoji as fallback)
# For a proper icon, you would need an .icns file
cat > "$APP_PATH/Contents/Resources/icon.icns" << 'EOF'
# Placeholder for icon - in a real application, this would be a proper .icns file
# You can replace this with a custom icon file
EOF

echo "✅ Desktop shortcut created successfully!"
echo "📍 Location: $APP_PATH"
echo ""
echo "🎯 To use the shortcut:"
echo "   1. Double-click 'FLUX Krea Studio' on your desktop"
echo "   2. Terminal will open and start the web interface"
echo "   3. Your browser will automatically open to the interface"
echo "   4. Close Terminal window to stop the server"
echo ""
echo "💡 If you need to update the project path:"
echo "   Right-click the app → Show Package Contents → Contents/MacOS/FLUX Krea Studio"
echo "   Edit the PROJECT_DIR variable at the top of the file"

# Make the app executable
chmod -R 755 "$APP_PATH"

echo ""
echo "🎉 FLUX Krea Studio desktop shortcut is ready!"