#!/usr/bin/env python3
"""
FLUX.1 Krea Web UI Launcher
Simple launcher with system checks and setup
"""

import os
import sys
import subprocess
from pathlib import Path

def check_environment():
    """Check if environment is properly set up"""
    print("🔍 Checking FLUX.1 Krea Web UI Environment...")
    
    # Check if we're in the right directory
    if not Path("flux_web_ui.py").exists():
        print("❌ Error: flux_web_ui.py not found")
        print("   Please run this script from the flux-krea directory")
        return False
    
    # Check if virtual environment is activated
    if "flux_env" not in str(sys.executable):
        print("⚠️  Virtual environment not detected")
        print("   Attempting to activate flux_env...")
        return "need_venv"
    
    # Check required modules
    try:
        import gradio
        import torch
        from flux_advanced import FluxAdvancedGenerator
        print("✅ All dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def launch_interface():
    """Launch the web interface"""
    print("\n🚀 Starting FLUX.1 Krea Web Interface...")
    print("🌐 Web interface will be available at: http://localhost:7860")
    print("📁 Generated images will be saved to: web_outputs/")
    print("⚠️  Note: First generation may take time to load models")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 60)
    
    # Import and launch
    from flux_web_ui import create_web_interface
    
    interface = create_web_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )

def main():
    """Main launcher function"""
    print("🎨 FLUX.1 Krea Web UI Launcher")
    print("=" * 40)
    
    env_status = check_environment()
    
    if env_status == "need_venv":
        print("\n💡 To run with virtual environment, use:")
        print("   source flux_env/bin/activate")
        print("   python launch_web_ui.py")
        return
    
    if env_status != True:
        print("\n❌ Environment check failed. Please fix the issues above.")
        return
    
    try:
        launch_interface()
    except KeyboardInterrupt:
        print("\n\n👋 FLUX.1 Krea Web UI stopped")
    except Exception as e:
        print(f"\n❌ Error starting web interface: {e}")
        print("\n💡 Try running: python flux_web_ui.py directly for more details")

if __name__ == "__main__":
    main()