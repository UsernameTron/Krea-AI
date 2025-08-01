# 🎨 FLUX.1 Krea Usage Guide

Your M4 Pro MacBook is now set up as a powerful AI image generation workstation. You have **two approaches** to choose from:

## 🚀 Quick Start Options

### Option A: API Generation (Recommended for Quick Start)
**✅ Advantages: No model download, instant start, consistent performance**

1. **Already configured!** Your HF_TOKEN is set up
2. **Generate immediately:**
```bash
source flux_env/bin/activate
python generate_api.py --prompt "a majestic snow-capped mountain reflected in a crystal clear lake at golden hour"
```

### Option B: Local Generation (Full Control)
**✅ Advantages: Complete privacy, no API costs, offline capability**

1. **Request Model Access:**
   Visit: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev
   Click "Request access" and agree to license terms

2. **Generate locally:**
```bash
source flux_env/bin/activate
python generate_image.py --prompt "your amazing prompt here"
```

## 📝 Available Scripts

### 🚀 **NEW: Advanced Generation Scripts**

#### `flux_advanced.py` - **FLAGSHIP** Professional-Grade Advanced Generation
**Features enterprise-level optimizations with intelligent group offloading for maximum M4 Pro performance**

```bash
# Balanced mode (recommended for M4 Pro)
python flux_advanced.py --prompt "your amazing prompt"

# Speed mode (maximum performance, uses more memory)
python flux_advanced.py --prompt "your prompt" --optimization speed

# Memory mode (maximum efficiency, perfect for complex scenes)
python flux_advanced.py --prompt "your prompt" --optimization memory

# High-resolution with intelligent optimization
python flux_advanced.py --prompt "detailed landscape" --width 1280 --height 1024 --optimization balanced
```

**🏆 Advanced Features:**
- 🧠 **Group Offloading**: Intelligent memory management with leaf-level control
- 🔄 **Stream Processing**: Overlaps data transfer with computation
- 🍎 **Apple Silicon MPS**: Native M4 Pro neural engine acceleration
- 💾 **Three Optimization Modes**: Speed, balanced, and memory efficiency
- 📊 **Performance Monitoring**: Real-time memory usage tracking

#### `generate_optimized.py` - Advanced Optimized Generation
**Features professional-grade optimizations for maximum M4 Pro performance**

```bash
# Basic optimized generation (recommended)
python generate_optimized.py --prompt "your amazing prompt"

# With 8-bit quantization (50% less memory usage)
python generate_optimized.py --prompt "your prompt" --quantize

# Maximum memory efficiency for large images
python generate_optimized.py --prompt "your prompt" --quantize --low-memory

# High-resolution optimized generation
python generate_optimized.py --prompt "detailed landscape" --width 1280 --height 1024 --quantize
```

**Advanced Features:**
- 🔧 **8-bit Quantization**: Reduce memory usage by ~50%
- 🍎 **Apple Silicon MPS**: Native M4 Pro acceleration
- 💾 **Memory Management**: Intelligent CPU offloading
- 🧠 **Attention Optimization**: Memory efficient processing
- 🖼️ **VAE Slicing**: Handle large images efficiently

### API-Based Scripts (Instant, No Download Required)

#### `generate_api.py` - Fast API Generation
**Uses Hugging Face Inference API - perfect for quick results**

```bash
# Basic usage
python generate_api.py --prompt "your amazing prompt here"

# With custom output filename
python generate_api.py --prompt "a cyberpunk cityscape with neon lights" --output my_api_image.png
```

#### `batch_generate_api.py` - API Batch Variations
**Generate multiple variations quickly via API**

```bash
# Generate 4 variations via API
python batch_generate_api.py "a serene japanese garden with cherry blossoms" 4

# Generate 8 variations
python batch_generate_api.py "abstract digital art with flowing colors" 8
```

### Local Generation Scripts (Full Control, Privacy)

#### `generate_image.py` - Local Generation Script
**Optimized specifically for your M4 Pro with 48GB RAM**

```bash
# Basic usage (requires model access approval)
python generate_image.py --prompt "your amazing prompt here"

# Advanced usage with all parameters
python generate_image.py \
  --prompt "a cyberpunk cityscape with neon lights" \
  --width 1280 \
  --height 1024 \
  --steps 28 \
  --guidance 4.0 \
  --seed 42 \
  --output my_masterpiece.png
```

**Parameters:**
- `--prompt`: Your creative description (required)
- `--width/--height`: Image dimensions (default: 1024x1024)
- `--steps`: Quality vs speed (28-32 recommended)
- `--guidance`: How closely to follow prompt (3.5-5.0 range)
- `--seed`: For reproducible results
- `--output`: Filename for your image

#### `batch_generate.py` - Local Batch Variations
**Perfect for exploring different interpretations locally**

```bash
# Generate 4 variations locally
python batch_generate.py "a serene japanese garden with cherry blossoms" 4
```

### `monitor_generation.py` - Performance Analysis
**Understand how your M4 Pro handles the workload**

```bash
# Monitor a single generation
python monitor_generation.py "a futuristic space station orbiting earth"

# Monitor with custom parameters
python monitor_generation.py "cyberpunk street scene" --steps 32 --guidance 4.5
```

### `test_advanced_optimizations.py` - **NEW** Comprehensive Optimization Testing
**Test and compare all optimization levels to find the best configuration for your workflow**

```bash
# Run comprehensive test suite
python test_advanced_optimizations.py

# This will test speed, balanced, and memory modes and generate a detailed performance report
```

**📊 Testing Features:**
- Performance benchmarking across all optimization levels
- Memory usage analysis and recommendations
- Automated test image generation for validation
- JSON report generation with detailed metrics
- Personalized recommendations for your M4 Pro configuration

## 🎨 **NEW: Specialized FLUX Workflows**

### `flux_img2img.py` - **Professional Image-to-Image Transformation**
**Advanced image transformation with Apple Silicon optimizations**

```bash
# Transform an existing image
python flux_img2img.py --prompt "turn this into a cyberpunk scene" --input_image photo.jpg

# Fine-tune transformation strength
python flux_img2img.py \
  --prompt "make this image look like an oil painting" \
  --input_image landscape.jpg \
  --strength 0.7 \
  --steps 32 \
  --guidance 7.0
```

**🔧 Features:**
- Supports both local files and URLs as input
- Configurable transformation strength (0.0-1.0)
- Apple Silicon memory optimizations
- Automatic RGB conversion for compatibility

### `flux_inpaint.py` - **Advanced Inpainting & Object Replacement**
**Precise image editing with mask-based inpainting**

```bash
# Inpaint using a mask image
python flux_inpaint.py \
  --prompt "a beautiful garden" \
  --input_image room.jpg \
  --mask_image mask.png

# Inpaint using bounding box coordinates
python flux_inpaint.py \
  --prompt "a modern fireplace" \
  --input_image living_room.jpg \
  --bbox "100,150,300,400"
```

**🎭 Features:**
- Mask-based or bounding box inpainting
- Automatic fallback to img2img if inpainting unavailable
- Support for URLs and local files
- Professional inpainting strength controls

### `flux_batch_workflow.py` - **Professional Batch Processing Manager**
**Enterprise-grade batch processing for creative workflows**

```bash
# Process multiple prompt variations
python flux_batch_workflow.py \
  --mode variations \
  --prompts_file example_prompts.json \
  --output_dir my_batch

# Explore artistic styles for a subject
python flux_batch_workflow.py \
  --mode styles \
  --prompt "a majestic lion" \
  --styles_file artistic_styles.json

# Parameter sweep optimization
python flux_batch_workflow.py \
  --mode sweep \
  --prompt "futuristic cityscape" \
  --output_dir parameter_test
```

**🏭 Professional Features:**
- **Three Workflow Modes**: Variations, style exploration, parameter sweeps
- **JSON Configuration**: Use provided example files or create custom workflows
- **Intelligent Naming**: Automatic filename generation with safe characters
- **Progress Tracking**: Real-time progress with success/failure reporting
- **Comprehensive Reports**: Detailed JSON reports with batch statistics
- **Memory Management**: Uses advanced optimization levels throughout

**📁 Included Configuration Files:**
- `example_prompts.json`: 10 diverse, high-quality prompts for testing
- `artistic_styles.json`: 15 professional artistic styles and techniques

## ⚖️ API vs Local: Which Should You Choose?

### API Generation (Recommended for Most Users)
**✅ Advantages:**
- **Instant start** - No 22GB model download
- **Consistent speed** - ~3-5 seconds per image
- **No waiting** - No repository access approval needed
- **Lower memory usage** - Doesn't use your Mac's RAM for model
- **Always latest** - API uses the most current model version

**❌ Considerations:**
- Requires internet connection
- Uses your Hugging Face API quota
- Less control over generation parameters
- Images processed in the cloud

### Local Generation (For Power Users)
**✅ Advantages:**
- **Complete privacy** - Everything stays on your Mac
- **Full control** - All parameters available (steps, guidance, etc.)
- **No API costs** - Unlimited generations once downloaded
- **Offline capable** - Works without internet
- **Optimized for M4 Pro** - Takes advantage of your 48GB RAM

**❌ Considerations:**
- 22GB model download required
- Needs repository access approval
- 3-5 minute cold start (model loading)
- Uses significant local memory

### 💡 Recommended Workflow
1. **Start with API** for quick experimentation and prompt testing
2. **Switch to local** when you find prompts you love and want to refine
3. **Use local for batches** when you need many variations
4. **Use API for sharing** when you want to quickly show others

## 🎯 Pro Tips for Best Results

### Prompt Engineering
- **Be specific**: "a red sports car" → "a sleek Ferrari 488 GTB in rosso corsa red, parked on a winding mountain road at sunset"
- **Include style**: "...in the style of Studio Ghibli animation"
- **Add technical details**: "...shot with a 85mm lens, shallow depth of field, cinematic lighting"

### Optimal Settings for Your M4 Pro
- **Resolution**: 1024x1024 to 1280x1024 (sweet spot for your hardware)
- **Steps**: 28-32 (good quality/speed balance)
- **Guidance**: 3.5-5.0 (higher = more prompt adherence)

### Performance Optimization
- Your 48GB RAM allows for larger batch sizes
- The M4 Pro neural engine accelerates certain operations
- First generation takes longer (model loading), subsequent ones are faster

## 📊 Understanding Your System Performance

### Expected Generation Times
- **First generation**: 3-5 minutes (includes model loading)
- **Subsequent generations**: 1.5-3 minutes
- **Batch generations**: ~2 minutes per image

### Memory Usage
- **Model loading**: ~22GB
- **Peak generation**: ~30-35GB
- **Your advantage**: 48GB allows comfortable operation without swapping

## 🛠️ Troubleshooting

### Common Issues

**"Access to model is restricted"**
- Solution: Request access at the HuggingFace model page
- Wait for approval (usually 5 minutes to 2 hours)

**"Out of memory" errors**
- Reduce image resolution: `--width 896 --height 896`
- Restart Python to clear memory: exit and `source flux_env/bin/activate`

**Slow generations**
- Check Activity Monitor for other memory-intensive apps
- Close unnecessary applications
- Ensure your MacBook is plugged in (performance throttling)

**Network issues during model download**
- The model will auto-download on first use (~22GB)
- Ensure stable internet connection
- Download can resume if interrupted

### Performance Monitoring
Use the monitoring script to understand your system:
```bash
python monitor_generation.py "test prompt"
```

Watch for:
- CPU usage (should peak around 80-90%)
- Memory usage (should stay under 80% of your 48GB)
- Generation time trends

## 🎨 Creative Workflows

### Style Exploration
```bash
# Try different artistic styles
python batch_generate.py "a lighthouse on a cliff, impressionist painting style" 3
python batch_generate.py "a lighthouse on a cliff, photorealistic" 3
python batch_generate.py "a lighthouse on a cliff, anime art style" 3
```

### Resolution Testing
```bash
# Test different aspect ratios
python generate_image.py --prompt "panoramic landscape" --width 1280 --height 720
python generate_image.py --prompt "portrait photo" --width 768 --height 1024
```

### Seed Experiments
```bash
# Compare different seeds with same prompt
python generate_image.py --prompt "futuristic cityscape" --seed 42
python generate_image.py --prompt "futuristic cityscape" --seed 123
python generate_image.py --prompt "futuristic cityscape" --seed 456
```

## 📈 Next Steps

1. **Experiment with prompting**: Try different styles, subjects, and technical specifications
2. **Monitor your system**: Use the monitoring script to understand performance patterns
3. **Create workflows**: Develop personal processes for different types of projects
4. **Share results**: Your local setup gives you complete privacy for your creative work

## 🆘 Getting Help

If you encounter issues:
1. Check the `installation_summary.txt` for your setup details
2. Ensure your virtual environment is activated
3. Monitor system resources during generation
4. Consider restarting if memory usage seems high

Your M4 Pro MacBook with 48GB RAM is exceptionally well-suited for AI image generation. Enjoy creating!