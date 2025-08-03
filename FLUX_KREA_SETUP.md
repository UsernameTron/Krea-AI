# 🎨 FLUX.1 Krea [dev] Setup Guide

## ✅ Step-by-Step Setup

### 1. **Request Model Access (REQUIRED)**
🔒 The model is gated and requires approval:

1. Go to: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev
2. Click **"Request access to this repository"**
3. Accept the **FluxDev Non-Commercial License Agreement**
4. Wait for approval (usually 5 minutes to 2 hours)

### 2. **Authentication Setup**
```bash
# Install huggingface CLI if not already installed
pip install huggingface-hub

# Login with your token
huggingface-cli login
```

Get your token from: https://huggingface.co/settings/tokens

### 3. **Install Requirements**
```bash
# Update to latest diffusers (critical!)
pip install -U diffusers

# Install other requirements
pip install -r requirements_flux_krea.txt
```

### 4. **Set Your HuggingFace Token**
```bash
export HF_TOKEN='your_huggingface_token_here'
```

### 5. **Test the Installation**
```bash
python flux_krea_official.py --prompt "A frog holding a sign that says hello world"
```

## 🎯 **Key Differences from Your Current Code**

### ❌ **What's Not Working in Your Code:**
1. **Custom FLUX Implementation**: Your `src/flux/` folder uses an older/incompatible architecture
2. **Model Loading**: Your code tries to load local files that don't exist
3. **Mixed Approaches**: You're mixing custom implementation with diffusers

### ✅ **Official Approach:**
- Uses `diffusers.FluxPipeline` directly
- Loads from HuggingFace hub automatically
- Much simpler and more reliable
- Based on the official documentation

## 🚀 **Usage Examples**

### Basic Generation
```bash
python flux_krea_official.py --prompt "a serene mountain landscape at golden hour"
```

### Advanced Parameters
```bash
python flux_krea_official.py \
  --prompt "photorealistic portrait of a wise old wizard" \
  --width 1280 \
  --height 1024 \
  --guidance 5.0 \
  --steps 32 \
  --seed 42 \
  --output wizard_portrait.png
```

### Web Interface
```bash
python flux_web_ui_official.py
```
Then open: http://localhost:7860

### Recommended Settings (from docs)
- **Resolution**: 1024-1280 pixels
- **Steps**: 28-32 
- **Guidance**: 3.5-5.0
- **Format**: Use detailed, specific prompts

## 🍎 **Apple Silicon Optimizations**

The official code automatically:
- Uses `torch.bfloat16` for efficiency
- Enables `model_cpu_offload()` for memory management
- Works with MPS (Metal Performance Shaders)

## 🐛 **Troubleshooting**

### "Access Denied" Error
- ❌ You haven't requested repository access
- ✅ Visit the model page and request access

### "401 Unauthorized" Error  
- ❌ Not authenticated with HuggingFace
- ✅ Run `huggingface-cli login`

### "Model Loading" Errors
- ❌ Old diffusers version
- ✅ Run `pip install -U diffusers`

### Memory Issues
- ❌ Model is very large (~24GB)
- ✅ Code uses `enable_model_cpu_offload()` automatically

## 📊 **Expected Performance (M4 Pro)**
- **First run**: 3-5 minutes (includes model download)
- **Subsequent runs**: 1-3 minutes per image
- **Memory usage**: ~20-30GB RAM during generation
- **Model size**: ~24GB downloaded

## 🎉 **Success Indicators**
When working correctly, you should see:
```
🎨 FLUX.1 Krea [dev] - Official Implementation
📥 Loading FLUX.1 Krea [dev] from HuggingFace...
✅ Model loaded in X.X seconds
🖼️  Generating image...
🎉 Generation complete!
💾 Saved as: your_image.png
```

---

This approach replaces your complex custom implementation with the official, supported method that's guaranteed to work with FLUX.1 Krea [dev].