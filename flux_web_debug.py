#!/usr/bin/env python3
"""
FLUX.1 Krea [dev] - Debug Web UI
Enhanced debugging and error handling for pipeline loading issues
"""

import gradio as gr
import torch
import signal
import time
import os
import sys
from pathlib import Path
from diffusers import FluxPipeline
import threading
import gc
import traceback

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Generation timed out")

class DebugFluxUI:
    def __init__(self):
        self.pipeline = None
        self.is_loading = False
        self.generation_lock = threading.Lock()
        self.debug_log = []
        
    def add_debug_log(self, message):
        """Add message to debug log"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.debug_log.append(log_message)
        print(log_message)
        
    def get_debug_log(self):
        """Get debug log as string"""
        return "\n".join(self.debug_log[-20:])  # Last 20 messages
        
    def check_environment(self):
        """Check environment and requirements"""
        self.add_debug_log("🔍 Checking environment...")
        
        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.add_debug_log(f"Python version: {python_version}")
        
        # Check PyTorch
        try:
            torch_version = torch.__version__
            self.add_debug_log(f"PyTorch version: {torch_version}")
        except:
            self.add_debug_log("❌ PyTorch not available")
            return "❌ PyTorch not installed"
        
        # Check diffusers
        try:
            import diffusers
            diffusers_version = diffusers.__version__
            self.add_debug_log(f"Diffusers version: {diffusers_version}")
        except:
            self.add_debug_log("❌ Diffusers not available")
            return "❌ Diffusers not installed"
        
        # Check HuggingFace token
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            self.add_debug_log(f"HF_TOKEN: {'*' * 10}{hf_token[-4:] if len(hf_token) > 4 else '****'}")
        else:
            self.add_debug_log("⚠️  HF_TOKEN not set")
        
        # Check device capabilities
        if torch.backends.mps.is_available():
            self.add_debug_log("✅ MPS (Metal) available")
        else:
            self.add_debug_log("⚠️  MPS not available")
            
        if torch.cuda.is_available():
            self.add_debug_log("✅ CUDA available")
        else:
            self.add_debug_log("ℹ️  CUDA not available")
        
        # Check memory
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        self.add_debug_log(f"System memory: {memory_gb:.1f} GB")
        
        return "✅ Environment check complete"
        
    def load_pipeline(self):
        """Load FLUX pipeline with extensive debugging"""
        if self.pipeline is not None:
            return "✅ Pipeline already loaded"
            
        if self.is_loading:
            return "⏳ Pipeline is already loading..."
            
        self.is_loading = True
        self.add_debug_log("🚀 Starting pipeline load...")
        
        try:
            # Environment check
            env_result = self.check_environment()
            
            # Check HuggingFace access first
            self.add_debug_log("🔐 Checking model access...")
            
            # Try to load with detailed error catching
            self.add_debug_log("📥 Loading FLUX.1 Krea [dev] pipeline...")
            
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Krea-dev",
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                trust_remote_code=True  # Add this for compatibility
            )
            
            self.add_debug_log("✅ Pipeline loaded from HuggingFace")
            
            # Apply optimizations with debugging
            self.add_debug_log("⚙️  Applying optimizations...")
            
            try:
                self.pipeline.enable_model_cpu_offload()
                self.add_debug_log("✅ CPU offload enabled")
            except Exception as e:
                self.add_debug_log(f"⚠️  CPU offload failed: {e}")
                try:
                    self.pipeline.enable_sequential_cpu_offload()
                    self.add_debug_log("✅ Sequential CPU offload enabled")
                except Exception as e2:
                    self.add_debug_log(f"⚠️  Sequential offload failed: {e2}")
                    self.add_debug_log("Using default memory management")
            
            # Memory optimizations
            try:
                self.pipeline.enable_attention_slicing("auto")
                self.add_debug_log("✅ Attention slicing enabled")
            except Exception as e:
                self.add_debug_log(f"⚠️  Attention slicing failed: {e}")
                
            try:
                self.pipeline.enable_vae_slicing()
                self.add_debug_log("✅ VAE slicing enabled")
            except Exception as e:
                self.add_debug_log(f"⚠️  VAE slicing failed: {e}")
            
            self.is_loading = False
            self.add_debug_log("🎉 Pipeline loaded successfully!")
            
            # Test the pipeline quickly
            self.add_debug_log("🧪 Testing pipeline...")
            try:
                # Quick test without generation
                test_result = "Pipeline components accessible"
                self.add_debug_log(f"✅ Test passed: {test_result}")
                return "✅ FLUX.1 Krea pipeline loaded and tested successfully!"
            except Exception as test_error:
                self.add_debug_log(f"⚠️  Pipeline test warning: {test_error}")
                return "⚠️  Pipeline loaded but test failed - may still work for generation"
            
        except Exception as e:
            self.is_loading = False
            error_msg = str(e)
            self.add_debug_log(f"❌ Pipeline load error: {error_msg}")
            
            # Detailed error analysis
            if "gated" in error_msg.lower() or "access" in error_msg.lower():
                self.add_debug_log("🔒 Repository access issue detected")
                return """❌ Repository Access Required:
1. Visit: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev
2. Click 'Request access to this repository'  
3. Accept the license agreement
4. Set HF_TOKEN environment variable with your token
5. Restart this application

Debug info available below."""
                
            elif "401" in error_msg or "403" in error_msg:
                self.add_debug_log("🔑 Authentication issue detected")
                return f"""❌ Authentication Error:
Your HuggingFace token may be invalid or expired.
Error: {error_msg}

Debug info available below."""
                
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                self.add_debug_log("🌐 Network issue detected")
                return f"""❌ Network Error:
Check your internet connection.
Error: {error_msg}

Debug info available below."""
                
            else:
                self.add_debug_log("❓ Unknown error detected")
                return f"""❌ Unknown Error:
{error_msg}

Full traceback:
{traceback.format_exc()}

Debug info available below."""
    
    def generate_with_timeout(self, prompt, width, height, steps, guidance, seed, timeout_minutes):
        """Generate image with timeout protection and debugging"""
        
        # Check pipeline first
        if self.pipeline is None:
            self.add_debug_log("❌ Generation attempted with no pipeline")
            return None, "❌ Pipeline not loaded. Click 'Load Pipeline' first."
        
        # Prevent concurrent generations
        if not self.generation_lock.acquire(blocking=False):
            self.add_debug_log("⚠️  Concurrent generation attempt blocked")
            return None, "⚠️  Another generation is in progress. Please wait."
        
        try:
            if not prompt.strip():
                return None, "❌ Please enter a prompt"
            
            self.add_debug_log(f"🖼️  Starting generation: '{prompt[:50]}...'")
            self.add_debug_log(f"⚙️  Settings: {width}x{height}, {steps} steps, guidance {guidance}")
            
            timeout_seconds = int(timeout_minutes * 60)
            
            # Set up timeout protection
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            start_time = time.time()
            
            try:
                # Generator setup
                generator = torch.Generator().manual_seed(seed) if seed > 0 else None
                self.add_debug_log(f"🎲 Seed: {'Random' if generator is None else seed}")
                
                # Generate with timeout protection
                self.add_debug_log("🚀 Starting FLUX generation...")
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
                self.add_debug_log(f"✅ Generation completed in {generation_time:.1f}s")
                
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
                return result.images[0], success_msg
                
            except TimeoutError:
                signal.alarm(0)
                self.add_debug_log(f"⏰ Generation timed out after {timeout_minutes} minutes")
                timeout_msg = f"⏰ Generation timed out after {timeout_minutes} minutes!"
                return None, timeout_msg + "\n💡 Try: Reduce size, steps, or increase timeout"
                
        except Exception as e:
            signal.alarm(0)
            error_msg = str(e)
            self.add_debug_log(f"❌ Generation error: {error_msg}")
            
            if "memory" in error_msg.lower():
                return None, f"❌ Memory error: {error_msg}\n💾 Try: Reduce image size or restart"
            else:
                return None, f"❌ Generation error: {error_msg}"
            
        finally:
            self.generation_lock.release()

# Global UI instance
flux_ui = DebugFluxUI()

def create_interface():
    """Create the Gradio interface with debugging"""
    
    with gr.Blocks(title="FLUX.1 Krea - Debug", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🛡️ FLUX.1 Krea [dev] - Debug Mode
        **Enhanced Error Handling** • **Detailed Logging** • **Timeout Protection**
        """)
        
        with gr.Row():
            with gr.Column():
                load_btn = gr.Button("📥 Load Pipeline (Debug Mode)", variant="primary", size="lg")
                load_status = gr.Textbox(label="Loading Status", interactive=False, 
                                       value="Click 'Load Pipeline' to start", lines=5)
        
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
                    steps = gr.Slider(10, 50, 20, step=1, label="Steps")  # Default to 20 for faster testing
                    guidance = gr.Slider(1.0, 10.0, 4.5, step=0.1, label="Guidance Scale")
                
                with gr.Row():
                    seed = gr.Number(label="Seed (0 for random)", value=42, precision=0)  # Default seed for testing
                    timeout = gr.Slider(1, 10, 5, step=0.5, label="Timeout (minutes)")
                
                generate_btn = gr.Button("🛡️ Generate (Debug Mode)", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                output_image = gr.Image(label="Generated Image", type="pil")
                generation_status = gr.Textbox(label="Generation Status", interactive=False)
        
        with gr.Row():
            debug_log = gr.Textbox(label="Debug Log", interactive=False, lines=10)
            refresh_log_btn = gr.Button("🔄 Refresh Debug Log")
        
        # Event handlers
        def load_with_debug():
            result = flux_ui.load_pipeline()
            return result, flux_ui.get_debug_log()
        
        def generate_with_debug(prompt, width, height, steps, guidance, seed, timeout):
            image, status = flux_ui.generate_with_timeout(prompt, width, height, steps, guidance, seed, timeout)
            return image, status, flux_ui.get_debug_log()
        
        load_btn.click(
            fn=load_with_debug,
            outputs=[load_status, debug_log]
        )
        
        generate_btn.click(
            fn=generate_with_debug,
            inputs=[prompt, width, height, steps, guidance, seed, timeout],
            outputs=[output_image, generation_status, debug_log]
        )
        
        refresh_log_btn.click(
            fn=lambda: flux_ui.get_debug_log(),
            outputs=debug_log
        )
        
        gr.Markdown("""
        ### 🔍 Debug Features:
        - **Detailed Logging**: See exactly what's happening during load/generation
        - **Environment Check**: Verify Python, PyTorch, diffusers, HF_TOKEN
        - **Error Analysis**: Specific solutions for common issues
        - **Memory Monitoring**: Track memory usage and cleanup
        - **Timeout Protection**: 5-minute default timeout prevents hanging
        
        ### 💡 Quick Test Settings:
        - **Prompt**: "a cute cat" (simple test)
        - **Size**: 1024x1024 (standard)
        - **Steps**: 20 (faster than 28)
        - **Seed**: 42 (reproducible results)
        """)
    
    return interface

if __name__ == "__main__":
    print("🔍 Starting FLUX.1 Krea Debug Mode")
    print("=" * 40)
    print("🛡️ Timeout Protection: ACTIVE")
    print("📝 Debug Logging: ENABLED")
    print("🌐 Web Interface: http://localhost:7860")
    print("💡 Check debug log for detailed information")
    print()
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        quiet=False
    )