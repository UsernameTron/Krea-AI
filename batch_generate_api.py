#!/usr/bin/env python3
"""
Batch generation script using Hugging Face Inference API
Perfect for quickly exploring variations without local compute overhead
"""

import os
import time
from pathlib import Path
from huggingface_hub import InferenceClient
import sys

def generate_variations_api(base_prompt, num_variations=4):
    """Generate multiple variations using the Hugging Face API"""
    
    # Check for HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("❌ Error: HF_TOKEN environment variable not set!")
        print("   Please set your Hugging Face token:")
        print("   export HF_TOKEN='your_token_here'")
        return False
    
    print(f"🌐 Generating {num_variations} variations via Hugging Face API")
    print(f"📝 Prompt: '{base_prompt}'")
    print("="*60)
    
    # Initialize client
    client = InferenceClient(
        provider="fal-ai",
        api_key=hf_token,
    )
    
    successful_generations = 0
    total_time = 0
    
    for i in range(num_variations):
        output_name = f"api_variation_{i+1:02d}.png"
        
        print(f"\n📸 Creating variation {i+1}/{num_variations}")
        
        try:
            start_time = time.time()
            
            # Generate image via API
            image = client.text_to_image(
                base_prompt,
                model="black-forest-labs/FLUX.1-Krea-dev"
            )
            
            generation_time = time.time() - start_time
            total_time += generation_time
            
            # Save the image
            output_path = Path(output_name)
            image.save(output_path)
            
            successful_generations += 1
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            
            print(f"✅ Variation {i+1} completed in {generation_time:.1f}s")
            print(f"   Size: {image.size[0]}x{image.size[1]}, File: {file_size:.1f}MB")
            
        except Exception as e:
            print(f"❌ Variation {i+1} failed: {e}")
        
        # Brief pause to be respectful to the API
        if i < num_variations - 1:
            time.sleep(1)
    
    print(f"\n{'='*60}")
    print(f"✨ Batch generation complete!")
    print(f"📊 Success rate: {successful_generations}/{num_variations} variations")
    print(f"⏱️  Total time: {total_time:.1f}s, Average: {total_time/max(successful_generations,1):.1f}s per image")
    
    if successful_generations > 0:
        print("🖼️  Generated files:")
        existing_files = []
        for i in range(num_variations):
            filename = f"api_variation_{i+1:02d}.png"
            if Path(filename).exists():
                existing_files.append(filename)
        
        if existing_files:
            for filename in existing_files:
                file_path = Path(filename)
                file_size = file_path.stat().st_size / (1024 * 1024)
                print(f"   {filename} ({file_size:.1f}MB)")
    
    return successful_generations > 0

def main():
    if len(sys.argv) < 2:
        print("🌐 FLUX.1 Krea API Batch Generator")
        print("Usage: python batch_generate_api.py 'your prompt here' [number_of_variations]")
        print("\nExample:")
        print("python batch_generate_api.py 'a serene mountain landscape at sunset' 4")
        print("\nAdvantages of API approach:")
        print("✅ No local model download (saves 22GB)")
        print("✅ No waiting for repository access")
        print("✅ Faster cold start (no model loading)")
        print("✅ Consistent generation times")
        print("✅ Lower memory usage on your Mac")
        sys.exit(1)
    
    prompt = sys.argv[1]
    num_vars = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    
    if num_vars < 1 or num_vars > 10:
        print("❌ Number of variations must be between 1 and 10 for API usage")
        sys.exit(1)
    
    print("🚀 FLUX.1 Krea API Batch Generator")
    print("=" * 50)
    print(f"Approach: Hugging Face Inference API")
    print(f"Prompt: {prompt}")
    print(f"Variations: {num_vars}")
    print("=" * 50)
    
    success = generate_variations_api(prompt, num_vars)
    
    if success:
        print("\n💡 Next steps:")
        print("   - Review your generated variations")
        print("   - Use the best ones as inspiration for further refinement")
        print("   - Try different prompting techniques")
    else:
        print("\n🔧 Troubleshooting:")
        print("   - Check your HF_TOKEN is set correctly")
        print("   - Verify internet connection")
        print("   - Try with a simpler prompt")

if __name__ == "__main__":
    main()