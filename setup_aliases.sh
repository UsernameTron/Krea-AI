#!/bin/bash
"""
Setup Command Line Aliases for FLUX.1 Krea
Adds convenient shortcuts to your shell profile
"""

PROJECT_DIR="/Users/cpconnor/Krea AI/flux-krea"
SHELL_PROFILE=""

# Detect shell and profile file
if [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_PROFILE="$HOME/.zshrc"
elif [[ "$SHELL" == *"bash"* ]]; then
    SHELL_PROFILE="$HOME/.bash_profile"
    # Fallback to .bashrc if .bash_profile doesn't exist
    if [ ! -f "$SHELL_PROFILE" ]; then
        SHELL_PROFILE="$HOME/.bashrc"
    fi
else
    echo "⚠️  Unknown shell: $SHELL"
    echo "Please manually add aliases to your shell profile"
    exit 1
fi

echo "🔧 Setting up FLUX.1 Krea Command Line Aliases"
echo "============================================="
echo "Shell: $SHELL"
echo "Profile: $SHELL_PROFILE"
echo ""

# Create aliases
ALIASES="
# FLUX.1 Krea Aliases - Added by setup_aliases.sh
alias flux-web='cd \"$PROJECT_DIR\" && source flux_env/bin/activate && python launch_web_ui.py'
alias flux-cli='cd \"$PROJECT_DIR\" && source flux_env/bin/activate && python flux_advanced.py'
alias flux-bench='cd \"$PROJECT_DIR\" && source flux_env/bin/activate && python flux_benchmark.py'
alias flux-inpaint='cd \"$PROJECT_DIR\" && source flux_env/bin/activate && python flux_inpaint.py'
alias flux-workflow='cd \"$PROJECT_DIR\" && source flux_env/bin/activate && python flux_workflow.py --interactive'
alias flux-dir='cd \"$PROJECT_DIR\"'
"

# Check if aliases already exist
if grep -q "FLUX.1 Krea Aliases" "$SHELL_PROFILE" 2>/dev/null; then
    echo "⚠️  FLUX aliases already exist in $SHELL_PROFILE"
    echo "Would you like to update them? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        # Remove old aliases and add new ones
        sed -i.bak '/# FLUX.1 Krea Aliases/,/^$/d' "$SHELL_PROFILE"
        echo "$ALIASES" >> "$SHELL_PROFILE"
        echo "✅ Aliases updated successfully!"
    else
        echo "❌ Aliases not updated"
        exit 0
    fi
else
    # Add aliases to profile
    echo "$ALIASES" >> "$SHELL_PROFILE"
    echo "✅ Aliases added successfully!"
fi

echo ""
echo "🎯 Available Commands (after restarting terminal or running 'source $SHELL_PROFILE'):"
echo "   flux-web       - Start web interface"
echo "   flux-cli       - Run CLI generation (add --help for options)"
echo "   flux-bench     - Run performance benchmark"
echo "   flux-inpaint   - Run inpainting tool (add --help for options)"
echo "   flux-workflow  - Start interactive workflow"
echo "   flux-dir       - Navigate to project directory"
echo ""
echo "💡 To activate aliases now, run:"
echo "   source $SHELL_PROFILE"
echo ""
echo "🎉 FLUX.1 Krea aliases setup complete!"