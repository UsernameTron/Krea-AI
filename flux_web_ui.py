#!/usr/bin/env python3
"""
FLUX.1 Krea Web Interface
Comprehensive Gradio-based web UI for all FLUX capabilities
"""

import gradio as gr
import torch
import time
import os
import threading
import uuid
from pathlib import Path
from PIL import Image
import numpy as np
from datetime import datetime
import json
from collections import defaultdict

from flux_advanced import FluxAdvancedGenerator
from flux_exceptions import FluxBaseException
try:
    from diffusers import FluxInpaintPipeline
    from diffusers.utils import load_image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

class FluxWebUI:
    def __init__(self):
        # Thread-safe model management
        self._lock = threading.Lock()
        self._generators = {}  # optimization_level -> generator
        self._inpaint_pipeline = None
        
        # Per-session state management
        self._session_histories = defaultdict(list)
        self._active_sessions = set()
        
        # Output management
        self.output_dir = Path("web_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        print("🌐 FLUX Web UI initialized with concurrent user support")
    
    def get_session_id(self, request: gr.Request = None):
        """Get or create session ID for user"""
        if request and hasattr(request, 'session_hash'):
            session_id = request.session_hash
        else:
            session_id = str(uuid.uuid4())
        
        self._active_sessions.add(session_id)
        return session_id
    
    def get_generator(self, optimization_level="balanced"):
        """Thread-safe generator management"""
        with self._lock:
            if optimization_level not in self._generators:
                print(f"🔄 Creating new generator for optimization: {optimization_level}")
                self._generators[optimization_level] = FluxAdvancedGenerator(optimization_level=optimization_level)
            return self._generators[optimization_level]
    
    def text_to_image(self, prompt, width, height, steps, guidance, seed, optimization, progress=gr.Progress(), request: gr.Request = None):
        """Generate image from text prompt with session management"""
        session_id = self.get_session_id(request)
        
        if not prompt.strip():
            return None, "Please enter a prompt"
        
        try:
            progress(0.1, desc="Initializing generator...")
            generator = self.get_generator(optimization)
            
            progress(0.3, desc="Loading models...")
            
            # Generate image
            progress(0.5, desc="Generating image...")
            start_time = time.time()
            
            image = generator.generate_image(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                seed=seed if seed != -1 else None
            )
            
            generation_time = time.time() - start_time
            progress(0.9, desc="Saving image...")
            
            # Save image with session info
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_short = session_id[:8] if session_id else "unknown"
            filename = f"txt2img_{session_short}_{timestamp}.png"
            output_path = self.output_dir / filename
            image.save(output_path)
            
            # Add to session history
            history_entry = {
                "type": "text-to-image",
                "prompt": prompt,
                "parameters": {
                    "width": width, "height": height, "steps": steps,
                    "guidance": guidance, "seed": seed, "optimization": optimization
                },
                "time": generation_time,
                "output": str(output_path),
                "timestamp": datetime.now().isoformat()
            }
            self._session_histories[session_id].append(history_entry)
            
            progress(1.0, desc="Complete!")
            
            info = f"✅ Generated in {generation_time:.1f}s | Size: {width}x{height} | Steps: {steps} | Guidance: {guidance}"
            if seed != -1:
                info += f" | Seed: {seed}"
            
            return image, info
            
        except FluxBaseException as e:
            error_msg = f"❌ FLUX Error: {str(e)}"
            if hasattr(e, 'guidance'):
                error_msg += f"\n💡 {e.guidance}"
            return None, error_msg
        except Exception as e:
            return None, f"❌ Unexpected Error: {str(e)}"
    
    def inpaint_image(self, prompt, source_image, mask_image, strength, steps, guidance, seed, progress=gr.Progress()):
        """Inpaint image with mask"""
        if not prompt.strip():
            return None, "Please enter a prompt"
        
        if source_image is None:
            return None, "Please upload a source image"
        
        if mask_image is None:
            return None, "Please upload a mask image"
        
        try:
            progress(0.1, desc="Initializing inpainting pipeline...")
            
            if self.inpaint_pipeline is None:
                self.inpaint_pipeline = FluxInpaintPipeline.from_pretrained(
                    "./models/FLUX.1-Krea-dev",
                    torch_dtype=torch.bfloat16,
                    local_files_only=True
                )
                self.inpaint_pipeline.enable_model_cpu_offload()
                self.inpaint_pipeline.vae.enable_slicing()
                self.inpaint_pipeline.vae.enable_tiling()
            
            progress(0.3, desc="Processing images...")
            
            # Process images
            source = Image.fromarray(source_image).convert('RGB')
            mask = Image.fromarray(mask_image).convert('L')
            
            # Resize mask to match source
            if mask.size != source.size:
                mask = mask.resize(source.size, Image.LANCZOS)
            
            progress(0.5, desc="Inpainting...")
            start_time = time.time()
            
            result = self.inpaint_pipeline(
                prompt=prompt,
                image=source,
                mask_image=mask,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=torch.Generator().manual_seed(seed) if seed != -1 else None
            )
            
            generation_time = time.time() - start_time
            image = result.images[0]
            
            progress(0.9, desc="Saving result...")
            
            # Save image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"inpaint_{timestamp}.png"
            output_path = self.output_dir / filename
            image.save(output_path)
            
            progress(1.0, desc="Complete!")
            
            info = f"✅ Inpainted in {generation_time:.1f}s | Strength: {strength} | Steps: {steps}"
            return image, info
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def create_variations(self, base_prompt, num_variations, base_seed, width, height, steps, guidance, optimization, progress=gr.Progress()):
        """Create multiple variations of a prompt"""
        if not base_prompt.strip():
            return [], "Please enter a base prompt"
        
        try:
            generator = self.get_generator(optimization)
            variations = []
            
            for i in range(num_variations):
                progress((i + 1) / num_variations, desc=f"Generating variation {i+1}/{num_variations}...")
                
                # Create variation prompt
                variation_prompt = f"{base_prompt}, variation {i+1}, unique perspective"
                
                # Use sequential seeds
                seed = base_seed + i if base_seed != -1 else None
                
                image = generator.generate_image(
                    prompt=variation_prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance=guidance,
                    seed=seed
                )
                
                # Save variation
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"variation_{i+1:02d}_{timestamp}.png"
                output_path = self.output_dir / filename
                image.save(output_path)
                
                variations.append(image)
            
            info = f"✅ Generated {num_variations} variations"
            return variations, info
            
        except Exception as e:
            return [], f"❌ Error: {str(e)}"
    
    def get_generation_history(self, request: gr.Request = None):
        """Get formatted generation history for current session"""
        session_id = self.get_session_id(request) if request else "default"
        session_history = self._session_histories.get(session_id, [])
        
        if not session_history:
            return "No generations yet for this session"
        
        history_text = f"Session History ({len(session_history)} generations):\n\n"
        for i, entry in enumerate(reversed(session_history[-10:])):
            history_text += f"{i+1}. {entry['type'].title()}\n"
            history_text += f"   Prompt: {entry['prompt'][:60]}...\n"
            history_text += f"   Time: {entry['time']:.1f}s\n"
            if 'parameters' in entry:
                params = entry['parameters']
                history_text += f"   Settings: {params['width']}x{params['height']}, {params['steps']} steps\n"
            history_text += "\n"
        
        return history_text

def create_web_interface():
    """Create the complete Gradio interface"""
    flux_ui = FluxWebUI()
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .tab-nav {
        background: linear-gradient(45deg, #1e3a8a, #3b82f6);
    }
    .generate-btn {
        background: linear-gradient(45deg, #059669, #10b981) !important;
        border: none !important;
        color: white !important;
    }
    """
    
    with gr.Blocks(css=css, title="FLUX.1 Krea Studio", theme=gr.themes.Soft()) as interface:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #1e3a8a; margin-bottom: 10px;">🎨 FLUX.1 Krea Studio</h1>
            <p style="color: #64748b; font-size: 18px;">Advanced AI Image Generation & Editing Suite</p>
        </div>
        """)
        
        with gr.Tabs():
            # Text-to-Image Tab
            with gr.Tab("🎨 Text-to-Image", elem_id="txt2img"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_input = gr.Textbox(
                            label="Prompt", 
                            placeholder="Describe your image in detail...",
                            lines=3
                        )
                        
                        with gr.Row():
                            width_slider = gr.Slider(512, 2048, value=1024, step=64, label="Width")
                            height_slider = gr.Slider(512, 2048, value=1024, step=64, label="Height")
                        
                        with gr.Row():
                            steps_slider = gr.Slider(1, 50, value=28, step=1, label="Steps")
                            guidance_slider = gr.Slider(1.0, 20.0, value=4.0, step=0.5, label="Guidance")
                        
                        with gr.Row():
                            seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                            optimization_dropdown = gr.Dropdown(
                                choices=["speed", "balanced", "memory"], 
                                value="balanced", 
                                label="Optimization"
                            )
                        
                        generate_btn = gr.Button("🚀 Generate Image", variant="primary", elem_classes="generate-btn")
                    
                    with gr.Column(scale=1):
                        output_image = gr.Image(label="Generated Image", height=500)
                        output_info = gr.Textbox(label="Generation Info", lines=2, interactive=False)
                
                generate_btn.click(
                    flux_ui.text_to_image,
                    inputs=[prompt_input, width_slider, height_slider, steps_slider, guidance_slider, seed_input, optimization_dropdown],
                    outputs=[output_image, output_info]
                )
            
            # Inpainting Tab
            with gr.Tab("🖌️ Inpainting", elem_id="inpaint"):
                with gr.Row():
                    with gr.Column(scale=1):
                        inpaint_prompt = gr.Textbox(
                            label="Inpainting Prompt",
                            placeholder="Describe what should replace the masked area...",
                            lines=2
                        )
                        
                        source_upload = gr.Image(label="Source Image", type="numpy")
                        mask_upload = gr.Image(label="Mask Image (white = inpaint area)", type="numpy")
                        
                        with gr.Row():
                            strength_slider = gr.Slider(0.1, 1.0, value=0.8, step=0.1, label="Strength")
                            inpaint_steps = gr.Slider(1, 50, value=28, step=1, label="Steps")
                        
                        with gr.Row():
                            inpaint_guidance = gr.Slider(1.0, 20.0, value=7.0, step=0.5, label="Guidance")
                            inpaint_seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                        
                        inpaint_btn = gr.Button("🖌️ Inpaint", variant="primary")
                    
                    with gr.Column(scale=1):
                        inpaint_output = gr.Image(label="Inpainted Image", height=500)
                        inpaint_info = gr.Textbox(label="Inpainting Info", lines=2, interactive=False)
                
                inpaint_btn.click(
                    flux_ui.inpaint_image,
                    inputs=[inpaint_prompt, source_upload, mask_upload, strength_slider, inpaint_steps, inpaint_guidance, inpaint_seed],
                    outputs=[inpaint_output, inpaint_info]
                )
            
            # Variations Tab  
            with gr.Tab("🔄 Variations", elem_id="variations"):
                with gr.Row():
                    with gr.Column(scale=1):
                        var_prompt = gr.Textbox(
                            label="Base Prompt",
                            placeholder="Base description for variations...",
                            lines=2
                        )
                        
                        with gr.Row():
                            num_variations = gr.Slider(2, 8, value=4, step=1, label="Number of Variations")
                            var_seed = gr.Number(label="Base Seed (-1 for random)", value=-1, precision=0)
                        
                        with gr.Row():
                            var_width = gr.Slider(512, 2048, value=1024, step=64, label="Width")
                            var_height = gr.Slider(512, 2048, value=1024, step=64, label="Height")
                        
                        with gr.Row():
                            var_steps = gr.Slider(1, 50, value=28, step=1, label="Steps")
                            var_guidance = gr.Slider(1.0, 20.0, value=4.0, step=0.5, label="Guidance")
                        
                        var_optimization = gr.Dropdown(
                            choices=["speed", "balanced", "memory"], 
                            value="balanced", 
                            label="Optimization"
                        )
                        
                        variations_btn = gr.Button("🔄 Generate Variations", variant="primary")
                    
                    with gr.Column(scale=1):
                        variations_gallery = gr.Gallery(
                            label="Generated Variations", 
                            columns=2, 
                            rows=2, 
                            height=500,
                            object_fit="contain"
                        )
                        variations_info = gr.Textbox(label="Variations Info", lines=1, interactive=False)
                
                variations_btn.click(
                    flux_ui.create_variations,
                    inputs=[var_prompt, num_variations, var_seed, var_width, var_height, var_steps, var_guidance, var_optimization],
                    outputs=[variations_gallery, variations_info]
                )
            
            # History & Settings Tab
            with gr.Tab("📊 History & Settings", elem_id="settings"):
                with gr.Row():
                    with gr.Column():
                        gr.HTML("<h3>🔧 System Settings</h3>")
                        
                        gr.HTML("""
                        <div style="padding: 15px; background: #f8fafc; border-radius: 8px; margin: 10px 0;">
                            <h4>🎛️ Optimization Levels:</h4>
                            <ul>
                                <li><strong>Speed:</strong> Maximum performance, higher memory usage</li>
                                <li><strong>Balanced:</strong> Good balance of speed and memory efficiency</li>
                                <li><strong>Memory:</strong> Memory-efficient, slower but stable</li>
                            </ul>
                        </div>
                        """)
                        
                        gr.HTML("""
                        <div style="padding: 15px; background: #f0f9ff; border-radius: 8px; margin: 10px 0;">
                            <h4>💡 Tips:</h4>
                            <ul>
                                <li>Use detailed prompts for better results</li>
                                <li>Steps 20-35 work well for most images</li>
                                <li>Guidance 3.5-7.0 gives good prompt following</li>
                                <li>Use same seed for reproducible results</li>
                            </ul>
                        </div>
                        """)
                    
                    with gr.Column():
                        gr.HTML("<h3>📈 Generation History</h3>")
                        history_display = gr.Textbox(
                            label="Recent Generations",
                            lines=15,
                            interactive=False,
                            value=flux_ui.get_generation_history()
                        )
                        
                        refresh_history_btn = gr.Button("🔄 Refresh History")
                        
                        refresh_history_btn.click(
                            flux_ui.get_generation_history,
                            outputs=[history_display]
                        )
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #64748b;">
            <p>FLUX.1 Krea Studio - Advanced AI Image Generation</p>
            <p>Generated images are saved to the 'web_outputs' directory</p>
        </div>
        """)
    
    return interface

if __name__ == "__main__":
    print("🚀 Starting FLUX.1 Krea Web Interface...")
    print("🌐 The interface will be available at: http://localhost:7860")
    print("📁 Generated images will be saved to: web_outputs/")
    
    interface = create_web_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        debug=True
    )