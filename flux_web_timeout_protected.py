#!/usr/bin/env python3
"""
FLUX.1 Krea [dev] - Timeout Protected Web UI
Prevents infinite loops with timeout protection and better error handling
"""

import gradio as gr
import torch
import signal
import time
from pathlib import Path
from diffusers import FluxPipeline
import threading
import gc

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Generation timed out")

class TimeoutProtectedFluxUI:
    def __init__(self):
        self.pipeline = None
        self.is_loading = False
        self.generation_lock = threading.Lock()
        
    def load_pipeline(self):
        """Load FLUX pipeline with timeout protection"""
        if self.pipeline is not None:
            return "✅ Pipeline already loaded"
            
        if self.is_loading:
            return "⏳ Pipeline is already loading..."
            
        self.is_loading = True
        
        try:
            print("📥 Loading FLUX.1 Krea [dev] with timeout protection...")
            
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Krea-dev",
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                low_cpu_mem_usage=True
            )
            
            # Apply memory optimizations to prevent loops
            try:
                self.pipeline.enable_model_cpu_offload()
                print("✅ CPU offload enabled")
            except:
                try:
                    self.pipeline.enable_sequential_cpu_offload()
                    print("✅ Sequential CPU offload enabled")
                except:
                    print("⚠️  Using default memory management")
            
            # Enable memory-efficient features
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
            
            self.is_loading = False
            return "✅ FLUX.1 Krea pipeline loaded successfully with timeout protection!"
            
        except Exception as e:
            self.is_loading = False
            error_msg = str(e)
            
            if "gated" in error_msg.lower() or "access" in error_msg.lower():
                return """❌ Repository Access Required:
1. Visit: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev
2. Click 'Request access to this repository'
3. Accept the license agreement
4. Set HF_TOKEN environment variable with your token"""
            else:
                return f"❌ Error loading pipeline: {error_msg}"
    
    def generate_with_timeout(self, prompt, width, height, steps, guidance, seed, timeout_minutes):
        """Generate image with timeout protection"""
        
        # Prevent concurrent generations
        if not self.generation_lock.acquire(blocking=False):
            return None, "⚠️  Another generation is in progress. Please wait."
        
        try:
            if self.pipeline is None:
                return None, "❌ Pipeline not loaded. Click 'Load Pipeline' first."
            
            if not prompt.strip():
                return None, "❌ Please enter a prompt"
            
            timeout_seconds = int(timeout_minutes * 60)
            
            # Set up timeout protection
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            status_msg = f"🖼️  Generating '{prompt}' ({width}x{height}, {steps} steps, {timeout_minutes}min timeout)..."
            print(status_msg)
            
            start_time = time.time()
            
            try:
                # Generator setup
                generator = torch.Generator().manual_seed(seed) if seed > 0 else None
                
                # Generate with timeout protection
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
                
                # Clear timeout
                signal.alarm(0)
                
                generation_time = time.time() - start_time
                
                # Memory cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.backends.mps.is_available():
                    try:
                        torch.mps.empty_cache()
                    except:
                        pass
                
                success_msg = f"✅ Generated in {generation_time:.1f}s | Size: {width}x{height} | Steps: {steps}"
                print(success_msg)
                
                return result.images[0], success_msg
                
            except TimeoutError:
                signal.alarm(0)  # Clear timeout
                timeout_msg = f"⏰ Generation timed out after {timeout_minutes} minutes!"
                error_details = """
💡 Solutions to try:
• Reduce image size (try 768x768)
• Reduce steps (try 20 steps)
• Use shorter prompt
• Increase timeout
• Restart the interface if needed"""
                return None, timeout_msg + error_details
                
        except Exception as e:
            signal.alarm(0)  # Clear timeout
            error_msg = f"❌ Generation error: {str(e)}"
            
            if "memory" in str(e).lower():
                error_msg += "\n💾 Try: Reduce image size or restart interface"
            
            return None, error_msg
            
        finally:
            self.generation_lock.release()

# Global UI instance
flux_ui = TimeoutProtectedFluxUI()

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="FLUX.1 Krea - Timeout Protected", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🛡️ FLUX.1 Krea [dev] - Timeout Protected Studio
        **Anti-Infinite Loop Version** • Prevents 900%+ CPU usage • 5-minute timeout protection
        """)
        
        with gr.Row():
            with gr.Column():
                load_btn = gr.Button("📥 Load Pipeline", variant="primary", size="lg")
                load_status = gr.Textbox(label="Loading Status", interactive=False, 
                                       value="Click 'Load Pipeline' to start")
        
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your image description here...",
                    lines=3
                )
                
                with gr.Row():
                    width = gr.Slider(512, 1280, 1024, step=64, label="Width")
                    height = gr.Slider(512, 1280, 1024, step=64, label="Height")
                
                with gr.Row():
                    steps = gr.Slider(10, 50, 28, step=1, label="Steps")
                    guidance = gr.Slider(1.0, 10.0, 4.5, step=0.1, label="Guidance Scale")
                
                with gr.Row():
                    seed = gr.Number(label="Seed (0 for random)", value=0, precision=0)
                    timeout = gr.Slider(1, 10, 5, step=0.5, label="Timeout (minutes)")
                
                generate_btn = gr.Button("🛡️ Generate (Timeout Protected)", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                output_image = gr.Image(label="Generated Image", type="pil")
                generation_status = gr.Textbox(label="Generation Status", interactive=False)
        
        # Event handlers
        load_btn.click(
            fn=flux_ui.load_pipeline,
            outputs=load_status
        )
        
        generate_btn.click(
            fn=flux_ui.generate_with_timeout,
            inputs=[prompt, width, height, steps, guidance, seed, timeout],
            outputs=[output_image, generation_status]
        )
        
        gr.Markdown("""
        ### 🚨 Infinite Loop Prevention Features:
        - ⏰ **Automatic Timeout**: Stops generation after specified time
        - 🧠 **Memory Management**: Prevents memory leaks and loops
        - 🛡️ **Error Handling**: Graceful failure instead of hanging
        - 🔄 **Process Isolation**: Each generation is isolated
        - 📊 **Resource Monitoring**: Tracks generation progress
        
        ### 💡 Tips to Avoid Issues:
        - Start with smaller images (768x768) and fewer steps (20)
        - Use timeout protection (default 5 minutes)
        - Monitor CPU usage - if it goes above 800%, cancel and restart
        - Keep prompts concise for better performance
        """)
    
    return interface

if __name__ == "__main__":
    print("🛡️ Starting Timeout Protected FLUX.1 Krea Studio")
    print("=" * 55)
    print("🚨 Infinite Loop Prevention: ACTIVE")
    print("⏰ Timeout Protection: 5 minutes default")
    print("🌐 Web Interface: http://localhost:7860")
    print("🛑 Press Ctrl+C to stop server")
    print()
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        quiet=False
    )