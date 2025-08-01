import time
import torch
import psutil
from diffusers import FluxPipeline

def quick_test():
    print("🧪 Quick performance test...")
    
    pipeline = FluxPipeline.from_pretrained(
        "./models/FLUX.1-Krea-dev",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    
    pipeline.enable_model_cpu_offload()
    pipeline.vae.enable_slicing()
    
    start_time = time.time()
    result = pipeline(
        "a simple test image",
        num_inference_steps=10,  # Quick test
        height=512,
        width=512
    )
    
    duration = time.time() - start_time
    memory_usage = psutil.virtual_memory().percent
    
    result.images[0].save("quick_test.png")
    
    print(f"✅ Test complete: {duration:.1f}s, Memory: {memory_usage:.1f}%")

if __name__ == "__main__":
    quick_test()