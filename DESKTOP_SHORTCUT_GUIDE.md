# 🖥️ FLUX.1 Krea Desktop Shortcut Guide

## ✅ Desktop Shortcut Created!

A **"FLUX Krea Studio"** application has been created on your desktop that provides one-click access to the web interface.

## 🎯 How to Use the Desktop Shortcut

### Method 1: Desktop App (Recommended)
1. **Double-click** `FLUX Krea Studio.app` on your desktop
2. Terminal will automatically open and start the web server
3. Your default browser will open to `http://localhost:7860`
4. Start generating images immediately!
5. **To stop**: Close the Terminal window

### Method 2: Command Line Aliases
Run the alias setup to get convenient terminal commands:
```bash
cd "/Users/cpconnor/Krea AI/flux-krea"
./setup_aliases.sh
```

After setup, you can use these commands from anywhere:
- `flux-web` - Start web interface
- `flux-cli` - Use command line tools  
- `flux-bench` - Run performance tests
- `flux-inpaint` - Inpainting tool
- `flux-workflow` - Interactive workflow
- `flux-dir` - Navigate to project folder

## 🔧 Customization Options

### Update Project Path
If you move the project folder:
1. Right-click `FLUX Krea Studio.app` → **Show Package Contents**
2. Navigate to `Contents/MacOS/FLUX Krea Studio`
3. Edit the `PROJECT_DIR` variable at the top of the file
4. Save and close

### Custom Icon
To add a custom icon:
1. Find a `.icns` icon file
2. Right-click `FLUX Krea Studio.app` → **Show Package Contents**  
3. Replace `Contents/Resources/icon.icns` with your icon
4. Restart Finder or log out/in to see the new icon

### Change Port or Settings
Edit `flux_web_ui.py` to modify:
- Server port (default: 7860)
- Network access settings
- Default generation parameters
- Interface theme and styling

## 🚀 Quick Start Workflow

1. **Launch**: Double-click desktop shortcut
2. **Wait**: First launch takes 2-3 minutes to load models
3. **Create**: 
   - Enter detailed prompt
   - Adjust settings as needed
   - Click "Generate Image"
4. **View**: Generated images appear in the interface
5. **Save**: Images auto-save to `web_outputs/` folder

## 💡 Tips for Best Experience

### Performance:
- First generation is slower (model loading)
- Use "Balanced" optimization for most systems
- Close other memory-intensive apps during generation

### Quality:
- Use detailed, specific prompts
- 28-32 steps for high quality
- Guidance 4.0-7.0 for balanced results
- 1024x1024 resolution works well

### Workflow:
- Keep browser tab open for multiple generations
- Use variations tab to explore concepts
- Check history tab to review past generations
- Generated images are automatically saved

## 🛠️ Troubleshooting

### Shortcut Won't Launch:
- Check that the project folder still exists
- Verify virtual environment is set up
- Try running `launch_flux_web.sh` directly

### Web Interface Won't Load:
- Wait 2-3 minutes for initial model loading
- Check Terminal for error messages  
- Try restarting the shortcut
- Ensure port 7860 isn't in use

### Permission Issues:
```bash
# Fix permissions if needed
chmod -R 755 "/Users/cpconnor/Desktop/FLUX Krea Studio.app"
```

## 📁 File Locations

- **Desktop Shortcut**: `/Users/cpconnor/Desktop/FLUX Krea Studio.app`
- **Project Directory**: `/Users/cpconnor/Krea AI/flux-krea/`
- **Generated Images**: `/Users/cpconnor/Krea AI/flux-krea/web_outputs/`
- **Shell Launcher**: `/Users/cpconnor/Krea AI/flux-krea/launch_flux_web.sh`

---

## 🎉 Ready to Create!

Your FLUX.1 Krea Studio is now ready for easy desktop access. Double-click the desktop shortcut and start generating amazing images with AI!

**Need help?** Check the `WEB_UI_GUIDE.md` for detailed interface instructions.