#!/usr/bin/env python3
"""
FLUX.1 Krea Web Interface - Apple Silicon M4 Pro Optimized
"""

import gradio as gr
import torch
import time
import os
import threading
from pathlib import Path
from datetime import datetime
from flux_krea_m4_optimized import M4ProFluxPipeline
import psutil

class OptimizedFluxWebUI:
    def __init__(self):
        self.pipeline = M4ProFluxPipeline()
        self.output_dir = Path("web_outputs")
        self.output_dir.mkdir(exist_ok=True)
        self.is_loading = False
        
        # Pre-load pipeline in background thread for faster startup
        self.load_thread = threading.Thread(target=self._background_load)
        self.load_thread.daemon = True
        self.load_thread.start()
    
    def _background_load(self):
        """Load pipeline in background thread"""
        self.is_loading = True
        try:
            print("🍎 Pre-loading FLUX.1 Krea with Apple Silicon optimizations...")
            self.pipeline.load_pipeline()
            print("✅ Background loading complete!")
        except Exception as e:
            print(f"❌ Background loading failed: {e}")
        finally:
            self.is_loading = False
    
    def generate_image(self, prompt, width, height, guidance, steps, seed, progress=gr.Progress()):
        """Generate image with progress tracking"""
        # Wait for background loading if still in progress
        while self.is_loading:
            progress(0.05, desc="Loading model in background...")
            time.sleep(0.5)
        
        if not self.pipeline.is_loaded:
            return None, "❌ Pipeline failed to load. Check console for errors."
        
        if not prompt.strip():
            return None, "Please enter a prompt"
        
        try:
            progress(0.1, desc="Preparing Apple Silicon optimized generation...")
            
            start_time = time.time()
            
            # Monitor memory usage
            memory_before = psutil.virtual_memory().percent
            
            progress(0.3, desc="Generating with Metal Performance Shaders...")
            
            # Generate image
            image = self.pipeline.generate(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=guidance,
                num_inference_steps=steps,
                seed=seed if seed != -1 else None
            )
            
            generation_time = time.time() - start_time
            memory_after = psutil.virtual_memory().percent
            
            progress(0.9, desc="Saving optimized output...")
            
            # Save with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"flux_krea_m4_{timestamp}.png"
            output_path = self.output_dir / filename
            image.save(output_path)
            
            progress(1.0, desc="Complete!")
            
            # Performance metrics
            device_info = "Metal Performance Shaders" if torch.backends.mps.is_available() else "CPU"
            memory_peak = torch.mps.current_allocated_memory() // (1024**2) if torch.backends.mps.is_available() else 0
            
            info = f"✅ Generated in {generation_time:.1f}s | {device_info}"
            info += f"\n📐 Size: {width}x{height} | Steps: {steps} | Guidance: {guidance}"
            info += f"\n🧠 Peak GPU memory: {memory_peak} MB | System memory: {memory_before}% → {memory_after}%"
            if seed != -1:
                info += f"\n🎲 Seed: {seed}"
            
            return image, info
            
        except Exception as e:
            error_msg = f"❌ Generation failed: {str(e)}"
            return None, error_msg

def create_interface():
    """Create optimized Gradio interface"""
    flux_ui = OptimizedFluxWebUI()
    
    # Custom CSS for Apple-like interface
    css = """
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    }
    .apple-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(title="FLUX.1 Krea M4 Pro Studio", theme=gr.themes.Soft(), css=css) as interface:
        
        gr.HTML("""
        <div class="apple-header">
            <h1>🍎 FLUX.1 Krea Studio</h1>
            <p style="font-size: 18px; opacity: 0.9;">Apple Silicon M4 Pro Optimized • Metal Performance Shaders</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="✨ Prompt",
                    placeholder="Professional portrait photography, golden hour lighting, highly detailed, photorealistic...",
                    lines=4
                )
                
                with gr.Accordion("🎛️ Generation Settings", open=False):
                    with gr.Row():
                        width_slider = gr.Slider(
                            minimum=512, maximum=1280, value=1024, step=64,
                            label="Width (optimized for M4 Pro: 1024-1280)"
                        )
                        height_slider = gr.Slider(
                            minimum=512, maximum=1280, value=1024, step=64,
                            label="Height (optimized for M4 Pro: 1024-1280)"
                        )
                    
                    with gr.Row():
                        guidance_slider = gr.Slider(
                            minimum=1.0, maximum=10.0, value=4.5, step=0.1,
                            label="Guidance Scale (3.5-5.0 recommended)"
                        )
                        steps_slider = gr.Slider(
                            minimum=20, maximum=40, value=28, step=1,
                            label="Steps (28-32 optimal for M4 Pro)"
                        )
                    
                    seed_input = gr.Number(
                        label="🎲 Seed (-1 for random)", 
                        value=-1, 
                        precision=0
                    )
                
                generate_btn = gr.Button(
                    "🚀 Generate with Apple Silicon", 
                    variant="primary",
                    size="lg"
                )
                
                # System info
                device_info = "Metal Performance Shaders" if torch.backends.mps.is_available() else "CPU"
                memory_gb = psutil.virtual_memory().total // (1024**3)
                gr.HTML(f"""
                <div style="margin-top: 15px; padding: 12px; background: #f0f9ff; border-radius: 8px; font-size: 14px;">
                    <strong>🍎 System:</strong> {device_info}<br>
                    <strong>💾 Unified Memory:</strong> {memory_gb} GB<br>
                    <strong>⚡ Optimization:</strong> Apple Silicon Native
                </div>
                """)
            
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Generated Image", 
                    height=600,
                    type="pil"
                )
                output_info = gr.Textbox(
                    label="Performance Metrics", 
                    lines=4, 
                    interactive=False
                )
        
        # Apple Silicon optimization tips
        gr.HTML("""
        <div style="margin-top: 20px; padding: 15px; background: #fafafa; border-radius: 12px;">
            <h3>🍎 Apple Silicon M4 Pro Optimizations Active:</h3>
            <ul style="margin: 10px 0;">
                <li><strong>Metal Performance Shaders:</strong> GPU acceleration via Metal API</li>
                <li><strong>Unified Memory:</strong> Optimized for Apple's shared memory architecture</li>
                <li><strong>Neural Engine:</strong> Leveraging dedicated ML acceleration (when supported)</li>
                <li><strong>Memory Efficient Attention:</strong> Reduced memory footprint for larger images</li>
                <li><strong>Attention Slicing:</strong> Breaking large attention operations into smaller chunks</li>
                <li><strong>VAE Tiling:</strong> Processing images in tiles to reduce memory usage</li>
            </ul>
        </div>
        """)
        
        # Connect interface
        generate_btn.click(
            flux_ui.generate_image,
            inputs=[
                prompt_input, width_slider, height_slider, 
                guidance_slider, steps_slider, seed_input
            ],
            outputs=[output_image, output_info]
        )
    
    return interface

if __name__ == "__main__":
    print("🍎 Starting FLUX.1 Krea Web Interface - Apple Silicon M4 Pro Optimized")
    print("🌐 Interface will be available at: http://localhost:7860")
    
    interface = create_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )