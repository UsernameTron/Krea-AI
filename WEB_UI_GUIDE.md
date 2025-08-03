# 🎨 FLUX.1 Krea Web Interface Guide

## 🌟 Overview

The FLUX.1 Krea Web Interface provides a comprehensive, user-friendly web-based GUI for all FLUX capabilities using Gradio. Access advanced AI image generation through your browser with intuitive controls and real-time feedback.

## 🚀 Quick Start

### Method 1: Using the Launcher (Recommended)
```bash
# Activate environment and start
source flux_env/bin/activate
python launch_web_ui.py
```

### Method 2: Direct Launch
```bash
# Activate environment and start directly
source flux_env/bin/activate  
python flux_web_ui.py
```

### Access the Interface
- Open your browser to: **http://localhost:7860**
- The interface will be available locally
- Generated images are saved to `web_outputs/` directory

## 🎛️ Interface Features

### 🎨 Text-to-Image Tab
**Generate images from text descriptions**

- **Prompt Input**: Detailed text description of desired image
- **Resolution Controls**: Width/Height (512-2048px)
- **Generation Settings**:
  - Steps: 1-50 (recommended: 20-35)
  - Guidance: 1.0-20.0 (recommended: 3.5-7.0)
  - Seed: Specific number or -1 for random
- **Optimization Levels**:
  - **Speed**: Maximum performance, higher memory usage
  - **Balanced**: Good balance of speed and memory efficiency  
  - **Memory**: Memory-efficient, slower but stable

### 🖌️ Inpainting Tab
**Edit images by painting over areas to replace**

- **Upload Source Image**: The image you want to edit
- **Upload Mask Image**: White areas will be inpainted
- **Inpainting Controls**:
  - Strength: 0.1-1.0 (how much to change masked area)
  - Steps: 1-50 (quality vs speed)
  - Guidance: 1.0-20.0 (prompt following strength)

### 🔄 Variations Tab
**Create multiple variations of the same concept**

- **Base Prompt**: The core description for variations
- **Number of Variations**: 2-8 different versions
- **Base Seed**: Starting seed for consistent variations
- **Quality Settings**: Same controls as text-to-image

### 📊 History & Settings Tab
**View generation history and system information**

- **Recent Generations**: Last 10 generations with details
- **System Settings**: Optimization level explanations
- **Usage Tips**: Best practices for optimal results

## 💡 Usage Tips

### For Best Results:
1. **Detailed Prompts**: Use specific, descriptive language
2. **Optimal Settings**: 
   - Steps: 28-32 for high quality
   - Guidance: 4.0-7.0 for balanced results
   - Resolution: 1024x1024 works well
3. **Seed Usage**: 
   - Use same seed to reproduce exact results
   - Use -1 for random generation
4. **Optimization Selection**:
   - **Speed**: If you have plenty of RAM (32GB+)
   - **Balanced**: Recommended for most systems
   - **Memory**: For systems with limited RAM

### Prompt Writing Tips:
```
Good: "A serene mountain landscape with crystal clear lake, golden hour lighting, photorealistic, highly detailed"

Better: "A breathtaking mountain landscape at golden hour, pristine alpine lake reflecting snow-capped peaks, warm orange and pink sky, cinematic composition, ultra-detailed, professional photography"
```

### Inpainting Tips:
1. **Mask Preparation**: 
   - White areas = what gets replaced
   - Black areas = what stays unchanged
   - Use clean, well-defined masks
2. **Strength Settings**:
   - 0.3-0.5: Subtle changes
   - 0.6-0.8: Moderate changes (recommended)
   - 0.9-1.0: Complete replacement

## 🛠️ Technical Details

### System Requirements:
- **RAM**: 16GB+ recommended (8GB minimum with memory optimization)
- **Storage**: 15GB+ for models
- **Python**: 3.8+ with virtual environment
- **Browser**: Modern browser with JavaScript enabled

### Performance Expectations:
- **First Generation**: 2-5 minutes (model loading)
- **Subsequent Generations**: 30-120 seconds (depends on settings)
- **Inpainting**: Similar to text-to-image
- **Variations**: Proportional to number requested

### File Management:
- **Generated Images**: Saved to `web_outputs/` directory
- **Naming Convention**: 
  - Text-to-image: `txt2img_YYYYMMDD_HHMMSS.png`
  - Inpainting: `inpaint_YYYYMMDD_HHMMSS.png`
  - Variations: `variation_01_YYYYMMDD_HHMMSS.png`

## 🐛 Troubleshooting

### Common Issues:

**Interface won't start:**
```bash
# Check environment
source flux_env/bin/activate
pip install gradio

# Try direct launch
python flux_web_ui.py
```

**Generation fails:**
- Check if models are downloaded
- Try lower resolution or fewer steps
- Switch to "memory" optimization
- Restart the interface

**Slow generation:**
- Use "speed" optimization if you have RAM
- Reduce image resolution
- Lower the number of steps

**Memory errors:**
- Switch to "memory" optimization
- Close other applications
- Reduce batch size for variations

### Error Messages:
- **"Models not found"**: Run model download first
- **"CUDA not available"**: Normal on CPU-only systems
- **"Out of memory"**: Use memory optimization or smaller images

## 🔧 Advanced Configuration

### Custom Settings:
Edit `flux_web_ui.py` to modify:
- Default values for sliders
- Interface theme and styling
- Port number (default: 7860)
- Output directory location

### Network Access:
To allow access from other devices:
```python
interface.launch(
    server_name="0.0.0.0",  # Allow external access
    server_port=7860,
    share=True  # Create public link (optional)
)
```

### Integration with Other Tools:
The web interface can be extended to include:
- Batch processing workflows
- Custom model loading
- API endpoints for external integration
- Advanced image editing tools

## 📝 Version History

### v1.0 Features:
- ✅ Text-to-image generation
- ✅ Inpainting capabilities  
- ✅ Variation generation
- ✅ Multiple optimization levels
- ✅ Generation history tracking
- ✅ Responsive web interface
- ✅ Real-time progress updates
- ✅ Automatic image saving

---

**🎉 Enjoy creating with FLUX.1 Krea Web Interface!**

For additional help or issues, check the main project documentation.