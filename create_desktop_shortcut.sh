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

echo "🛡️  Creating Desktop Shortcut for FLUX.1 Krea [dev] - Timeout Protected"
echo "====================================================================="

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

# Create the main executable script with timeout protection
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

# Function to show choice dialog
show_choice_dialog() {
    choice=$(osascript -e "display dialog \"Choose FLUX.1 Krea mode:\" with title \"FLUX Krea Studio\" buttons {\"Cancel\", \"Web UI (Protected)\", \"Command Line (Protected)\", \"Maximum Performance\"} default button \"Web UI (Protected)\"" 2>/dev/null)
    echo "$choice"
}

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    show_dialog "Error: Project directory not found at $PROJECT_DIR. Please update the path in the application."
    exit 1
fi

# Change to project directory
cd "$PROJECT_DIR"

# Check if timeout-protected script exists (primary choice)
if [ ! -f "flux_krea_timeout_fix.py" ]; then
    show_dialog "Error: Timeout-protected FLUX script not found. Please make sure flux_krea_timeout_fix.py exists."
    exit 1
fi

# Show version choice dialog
choice_result=$(show_choice_dialog)

# Determine which version to run based on user choice
if [[ "$choice_result" == *"Maximum Performance"* ]]; then
    if [ ! -f "maximum_performance_pipeline.py" ]; then
        show_dialog "Maximum Performance version not found. Using Web UI Protected instead."
        script_choice="web"
    else
        script_choice="maximum"
    fi
elif [[ "$choice_result" == *"Command Line (Protected)"* ]]; then
    script_choice="timeout"
elif [[ "$choice_result" == *"Cancel"* ]]; then
    exit 0
else
    script_choice="web"
fi

# Show starting notification and set launch parameters
if [ "$script_choice" = "maximum" ]; then
    show_notification "Starting FLUX Krea Studio - Maximum Performance Mode..."
    launch_script="maximum_performance_pipeline.py"
    mode_description="Maximum Performance (M4 Pro Optimized)"
    launch_command="echo 'python $launch_script --prompt \"a cute cat\" --steps 20' && exec bash"
elif [ "$script_choice" = "timeout" ]; then
    show_notification "Starting FLUX Krea Studio - Command Line Protected..."
    launch_script="flux_krea_timeout_fix.py"
    mode_description="Command Line Protected (Anti-Infinite Loop)"
    launch_command="echo 'python $launch_script --prompt \"a cute cat\" --steps 20 --timeout 300' && exec bash"
else
    show_notification "Starting FLUX Krea Studio - Web UI Protected..."
    launch_script="flux_web_timeout_protected.py"
    mode_description="Web UI Protected (Anti-Infinite Loop)"
    launch_command="python $launch_script"
fi

# Open Terminal and run the selected FLUX Krea version
osascript << APPLESCRIPT
tell application "Terminal"
    activate
    set currentTab to do script "cd '$PROJECT_DIR' && export HF_TOKEN='YOUR_HF_TOKEN_HERE' && echo '🛡️  FLUX.1 Krea Studio - $mode_description' && echo '========================================' && echo '⚠️  First generation: 2-5 minutes (24GB model)' && echo '⚠️  HF_TOKEN required for model access' && echo '🛑 Press Ctrl+C to cancel any stuck generation' && echo '⏰ Timeout protection: 5 minutes per generation' && echo '' && $launch_command"
    set custom title of currentTab to "FLUX Krea Studio - $mode_description"
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
echo "🛡️  NEW FEATURES - Timeout Protection:"
echo "   ✅ Prevents infinite loops (like the 971% CPU issue)"
echo "   ✅ 5-minute timeout per generation"
echo "   ✅ Choice between Timeout Protected & Maximum Performance modes"
echo "   ✅ Better error handling and memory management"
echo ""
echo "🎯 To use the shortcut:"
echo "   1. Double-click 'FLUX Krea Studio' on your desktop"
echo "   2. Choose version: 'Timeout Protected' (recommended) or 'Maximum Performance'"
echo "   3. Terminal opens with the selected mode ready"
echo "   4. Run: python [script] --prompt 'your prompt' --steps 20"
echo "   5. Generation will timeout after 5 minutes if stuck"
echo ""
echo "🚨 INFINITE LOOP PREVENTION:"
echo "   • Timeout Protected mode prevents 900%+ CPU usage"
echo "   • Automatic memory cleanup after generation"
echo "   • Press Ctrl+C to cancel stuck generations"
echo ""
echo "💡 If you need to update the project path:"
echo "   Right-click the app → Show Package Contents → Contents/MacOS/FLUX Krea Studio"
echo "   Edit the PROJECT_DIR variable at the top of the file"

# Make the app executable
chmod -R 755 "$APP_PATH"

echo ""
echo "🎉 FLUX Krea Studio desktop shortcut is ready!"