#!/usr/bin/env python3
"""
FLUX.1 Krea Web Interface - Official Implementation
Uses the official diffusers FluxPipeline for reliability
"""

import gradio as gr
import torch
import time
import os
from pathlib import Path
from datetime import datetime
from diffusers import FluxPipeline

class FluxKreaWebUI:
    def __init__(self):
        self.pipeline = None
        self.output_dir = Path("web_outputs")
        self.output_dir.mkdir(exist_ok=True)
        self.load_pipeline()
    
    def load_pipeline(self):
        """Load the FLUX.1 Krea pipeline"""
        print("🚀 Loading FLUX.1 Krea [dev] pipeline...")
        
        # Check for HuggingFace token
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            print("❌ HF_TOKEN environment variable not set!")
            print("💡 Set your token: export HF_TOKEN='your_token_here'")
            self.pipeline = None
            return
            
        try:
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Krea-dev",
                torch_dtype=torch.bfloat16,
                token=hf_token
            )
            # Enable memory optimization for Apple Silicon
            try:
                self.pipeline.enable_model_cpu_offload()
            except Exception as e:
                print(f"Note: CPU offload not available: {e}")
                # Alternative: use sequential CPU offload
                try:
                    self.pipeline.enable_sequential_cpu_offload()
                except:
                    print("Using default memory management")
            print("✅ Pipeline loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load pipeline: {e}")
            if "gated" in str(e).lower() or "access" in str(e).lower():
                print("💡 Need model access? Visit: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev")
            elif "401" in str(e) or "authentication" in str(e).lower():
                print("💡 Run: huggingface-cli login")
            self.pipeline = None
    
    def generate_image(self, prompt, width, height, guidance, steps, seed, progress=gr.Progress()):
        """Generate image using FLUX.1 Krea"""
        if not self.pipeline:
            return None, "❌ Pipeline not loaded. Please check model access and authentication."
        
        if not prompt.strip():
            return None, "Please enter a prompt"
        
        try:
            progress(0.1, desc="Preparing generation...")
            
            # Setup generator for reproducible results
            generator = torch.Generator().manual_seed(seed) if seed != -1 else None
            
            progress(0.3, desc="Generating image...")
            start_time = time.time()
            
            # Generate using official FluxPipeline
            result = self.pipeline(
                prompt,
                height=height,
                width=width,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=generator
            )
            
            generation_time = time.time() - start_time
            image = result.images[0]
            
            progress(0.9, desc="Saving image...")
            
            # Save with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"flux_krea_{timestamp}.png"
            output_path = self.output_dir / filename
            image.save(output_path)
            
            progress(1.0, desc="Complete!")
            
            info = f"✅ Generated in {generation_time:.1f}s | {width}x{height} | Steps: {steps} | Guidance: {guidance}"
            if seed != -1:
                info += f" | Seed: {seed}"
            
            return image, info
            
        except Exception as e:
            error_msg = f"❌ Generation failed: {str(e)}"
            if "gated" in str(e).lower() or "access" in str(e).lower():
                error_msg += "\n💡 Need model access? Visit: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev"
            elif "401" in str(e) or "authentication" in str(e).lower():
                error_msg += "\n💡 Run: huggingface-cli login"
            return None, error_msg

def create_interface():
    """Create the Gradio interface"""
    flux_ui = FluxKreaWebUI()
    
    with gr.Blocks(title="FLUX.1 Krea Studio", theme=gr.themes.Soft()) as interface:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #1e3a8a;">🎨 FLUX.1 Krea Studio</h1>
            <p style="color: #64748b; font-size: 18px;">Official Implementation - Photorealistic AI Art Generation</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe your image in detail... e.g., 'A serene mountain landscape at golden hour, photorealistic, highly detailed'",
                    lines=3
                )
                
                with gr.Row():
                    width_slider = gr.Slider(
                        minimum=512, maximum=1280, value=1024, step=64,
                        label="Width (1024-1280 recommended)"
                    )
                    height_slider = gr.Slider(
                        minimum=512, maximum=1280, value=1024, step=64,
                        label="Height (1024-1280 recommended)"
                    )
                
                with gr.Row():
                    guidance_slider = gr.Slider(
                        minimum=1.0, maximum=10.0, value=4.5, step=0.5,
                        label="Guidance Scale (3.5-5.0 recommended)"
                    )
                    steps_slider = gr.Slider(
                        minimum=1, maximum=50, value=28, step=1,
                        label="Steps (28-32 recommended)"
                    )
                
                seed_input = gr.Number(
                    label="Seed (-1 for random)", 
                    value=-1, 
                    precision=0
                )
                
                generate_btn = gr.Button(
                    "🚀 Generate Image", 
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Generated Image", 
                    height=600,
                    type="pil"
                )
                output_info = gr.Textbox(
                    label="Generation Info", 
                    lines=3, 
                    interactive=False
                )
        
        # Tips section
        gr.HTML("""
        <div style="margin-top: 20px; padding: 15px; background: #f8fafc; border-radius: 8px;">
            <h3>💡 Tips for Best Results:</h3>
            <ul>
                <li><strong>Detailed Prompts:</strong> Use specific, descriptive language</li>
                <li><strong>Photography Terms:</strong> Include "photorealistic", "highly detailed", "professional photography"</li>
                <li><strong>Lighting:</strong> Specify lighting conditions like "golden hour", "soft lighting", "dramatic shadows"</li>
                <li><strong>Style:</strong> FLUX.1 Krea excels at realistic, aesthetic photography</li>
            </ul>
        </div>
        """)
        
        # Connect the interface
        generate_btn.click(
            flux_ui.generate_image,
            inputs=[
                prompt_input, width_slider, height_slider, 
                guidance_slider, steps_slider, seed_input
            ],
            outputs=[output_image, output_info]
        )
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #64748b;">
            <p>FLUX.1 Krea [dev] - Photorealistic AI Image Generation</p>
            <p>Generated images saved to: web_outputs/</p>
        </div>
        """)
    
    return interface

if __name__ == "__main__":
    print("🚀 Starting FLUX.1 Krea Web Interface...")
    print("🌐 Interface will be available at: http://localhost:7860")
    
    # Check for HuggingFace token before starting
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("\n❌ ERROR: HuggingFace token not found!")
        print("💡 Please run:")
        print("   export HF_TOKEN='your_huggingface_token_here'")
        print("   python flux_web_ui_official.py")
        exit(1)
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )