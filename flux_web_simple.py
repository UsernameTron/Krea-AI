#!/usr/bin/env python3
"""
FLUX.1 Krea [dev] - Simple Reliable Web UI
Simplified version that loads pipeline on startup and keeps it ready
"""

import gradio as gr
import torch
import time
from pathlib import Path
from diffusers import FluxPipeline
import threading
import gc

class SimpleFluxUI:
    def __init__(self):
        self.pipeline = None
        self.load_status = "🔄 Starting pipeline load..."
        self.generation_lock = threading.Lock()
        
        # Start loading pipeline immediately on startup
        self.load_thread = threading.Thread(target=self._load_pipeline_background, daemon=True)
        self.load_thread.start()
        
    def _load_pipeline_background(self):
        """Load pipeline in background thread"""
        try:
            print("📥 Loading FLUX.1 Krea [dev] in background...")
            self.load_status = "📥 Loading FLUX.1 Krea [dev] (this takes 2-3 minutes)..."
            
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Krea-dev",
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                low_cpu_mem_usage=True
            )
            
            # Apply optimizations
            try:
                self.pipeline.enable_model_cpu_offload()
                print("✅ CPU offload enabled")
            except:
                print("⚠️  Using default memory management")
            
            try:
                self.pipeline.enable_attention_slicing("auto")
                print("✅ Attention slicing enabled")
            except:
                pass
                
            try:
                self.pipeline.enable_vae_slicing()
                print("✅ VAE slicing enabled")
            except:
                pass
            
            self.load_status = "✅ FLUX.1 Krea pipeline ready for generation!"
            print("🎉 Pipeline loaded and ready!")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Pipeline load error: {error_msg}")
            
            if "gated" in error_msg.lower():
                self.load_status = """❌ Repository Access Required:
Visit: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev
Request access and set HF_TOKEN environment variable"""
            else:
                self.load_status = f"❌ Error loading pipeline: {error_msg}"
    
    def get_load_status(self):
        """Get current loading status"""
        return self.load_status
    
    def generate_image(self, prompt, width, height, steps, guidance, seed):
        """Generate image with the pre-loaded pipeline"""
        
        if self.pipeline is None:
            return None, "❌ Pipeline not ready yet. Please wait for loading to complete."
        
        if not prompt.strip():
            return None, "❌ Please enter a prompt"
        
        # Prevent concurrent generations
        if not self.generation_lock.acquire(blocking=False):
            return None, "⚠️  Another generation is in progress. Please wait."
        
        try:
            print(f"🖼️  Generating: '{prompt[:50]}...'")
            start_time = time.time()
            
            # Generator setup
            generator = torch.Generator().manual_seed(seed) if seed > 0 else None
            
            # Generate image
            with torch.inference_mode():
                result = self.pipeline(
                    prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    generator=generator,
                    max_sequence_length=256,
                    return_dict=True
                )
            
            generation_time = time.time() - start_time
            
            # Memory cleanup
            gc.collect()
            if torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except:
                    pass
            
            success_msg = f"✅ Generated in {generation_time:.1f}s | {width}x{height} | {steps} steps"
            print(success_msg)
            
            return result.images[0], success_msg
            
        except Exception as e:
            error_msg = f"❌ Generation error: {str(e)}"
            print(error_msg)
            return None, error_msg
            
        finally:
            self.generation_lock.release()

# Global UI instance - starts loading immediately
flux_ui = SimpleFluxUI()

def create_interface():
    """Create simple Gradio interface"""
    
    with gr.Blocks(title="FLUX.1 Krea - Simple", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🎨 FLUX.1 Krea [dev] - Simple Studio
        **Auto-Loading** • **Pre-loaded Pipeline** • **Ready to Generate**
        """)
        
        # Status display
        status_display = gr.Textbox(
            label="Pipeline Status", 
            value="🔄 Loading pipeline in background...",
            interactive=False,
            lines=2
        )
        
        # Refresh status button
        refresh_btn = gr.Button("🔄 Refresh Status")
        
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your image description here...",
                    lines=3,
                    value="a cute cat sitting in a garden"  # Default prompt for easy testing
                )
                
                with gr.Row():
                    width = gr.Slider(512, 1280, 1024, step=64, label="Width")
                    height = gr.Slider(512, 1280, 1024, step=64, label="Height")
                
                with gr.Row():
                    steps = gr.Slider(10, 50, 20, step=1, label="Steps")  # Default 20 for faster generation
                    guidance = gr.Slider(1.0, 10.0, 4.5, step=0.1, label="Guidance Scale")
                
                with gr.Row():
                    seed = gr.Number(label="Seed (0 for random)", value=42, precision=0)
                
                generate_btn = gr.Button("🖼️ Generate Image", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                output_image = gr.Image(label="Generated Image", type="pil")
                generation_status = gr.Textbox(label="Generation Status", interactive=False)
        
        # Event handlers
        refresh_btn.click(
            fn=flux_ui.get_load_status,
            outputs=status_display
        )
        
        generate_btn.click(
            fn=flux_ui.generate_image,
            inputs=[prompt, width, height, steps, guidance, seed],
            outputs=[output_image, generation_status]
        )
        
        # Auto-refresh status every 5 seconds during loading
        interface.load(
            fn=flux_ui.get_load_status,
            outputs=status_display,
            every=5
        )
        
        gr.Markdown("""
        ### 🚀 Ready to Use:
        1. **Wait** for "Pipeline ready" status above (2-3 minutes first time)
        2. **Enter** your prompt or use the default "a cute cat sitting in a garden"
        3. **Click** "Generate Image" 
        4. **Wait** for generation (1-3 minutes depending on settings)
        
        ### 💡 Fast Generation Tips:
        - Use **20 steps** instead of 28 for faster results
        - Try **768x768** for quicker generation
        - Default prompt is ready to test immediately
        """)
    
    return interface

if __name__ == "__main__":
    print("🎨 Starting Simple FLUX.1 Krea Studio")
    print("=" * 40)
    print("🔄 Pipeline loading in background...")
    print("🌐 Web Interface: http://localhost:7860")
    print("⏳ Wait 2-3 minutes for pipeline to load")
    print("🎯 Default prompt ready for quick testing")
    print()
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        quiet=False
    )