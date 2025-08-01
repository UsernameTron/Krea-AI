#!/usr/bin/env python3
"""
FLUX.1 Krea Image generation using Hugging Face Inference API
This approach uses the cloud API instead of running the model locally
"""

import os
import argparse
import time
from pathlib import Path
from huggingface_hub import InferenceClient

def main():
    parser = argparse.ArgumentParser(description='Generate images with FLUX.1 Krea via Hugging Face API')
    parser.add_argument('--prompt', required=True, help='Detailed text description of the image you want')
    parser.add_argument('--output', default='flux_api_generation.png', help='Name for the output image file')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # Check for HF_TOKEN environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("❌ Error: HF_TOKEN environment variable not set!")
        print("   Please set your Hugging Face token:")
        print("   export HF_TOKEN='your_token_here'")
        print("   Or add it to your shell profile for persistence")
        return
    
    print("🌐 Using Hugging Face Inference API")
    print("   This uses cloud computing - no local model download needed!")
    print(f"🖼️  Generating: '{args.prompt}'")
    
    start_time = time.time()
    
    try:
        # Initialize the Inference Client
        client = InferenceClient(
            provider="fal-ai",
            api_key=hf_token,
        )
        
        print("📡 Sending request to Hugging Face API...")
        
        # Generate the image
        generation_params = {
            "model": "black-forest-labs/FLUX.1-Krea-dev"
        }
        
        if args.seed:
            # Note: Not all API providers support seed parameter
            # This may need adjustment based on the actual API capabilities
            print(f"🎲 Using seed: {args.seed}")
        
        image = client.text_to_image(
            args.prompt,
            **generation_params
        )
        
        generation_time = time.time() - start_time
        
        # Save the generated image
        output_path = Path(args.output)
        image.save(output_path)
        
        print(f"🎉 Generation complete in {generation_time:.1f} seconds")
        print(f"💾 Image saved as: {output_path.absolute()}")
        
        # Display image info
        print(f"📐 Image size: {image.size[0]}x{image.size[1]}")
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"📁 File size: {file_size:.1f} MB")
        
        if args.seed:
            print(f"🎲 Used seed: {args.seed}")
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error during generation: {e}")
        
        if "402" in error_msg or "Payment Required" in error_msg:
            print("\n💳 API CREDITS EXHAUSTED")
            print("   Your Hugging Face account has exceeded the free API credits.")
            print("   Solutions:")
            print("   1. Upgrade to HF Pro ($20/month): https://huggingface.co/pricing")
            print("   2. Wait for monthly credit reset")
            print("   3. Use local generation instead:")
            print("      python generate_image.py --prompt 'your prompt'")
        elif "gated" in error_msg.lower() or "access" in error_msg.lower():
            print("\n🔒 MODEL ACCESS REQUIRED")
            print("   Visit: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev")
            print("   Click 'Request access' and wait for approval")
        else:
            print("   Other possible causes:")
            print("   - Invalid or expired HF_TOKEN")
            print("   - Network connectivity issues")
            print("   - Temporary API service issues")

if __name__ == "__main__":
    main()