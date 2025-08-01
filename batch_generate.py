#!/usr/bin/env python3
"""
Batch generation script for exploring variations of a concept
Useful for creative exploration and finding the perfect image
"""

import subprocess
import sys
import time
import os

def generate_variations(base_prompt, num_variations=4):
    """Generate multiple variations of a prompt with different seeds"""
    
    print(f"🎨 Generating {num_variations} variations of: '{base_prompt}'")
    
    successful_generations = 0
    
    for i in range(num_variations):
        seed = 100 + i  # Sequential seeds for consistent but varied results
        output_name = f"variation_{i+1:02d}.png"
        
        print(f"\n📸 Creating variation {i+1}/{num_variations}")
        
        cmd = [
            "python", "generate_image.py",
            "--prompt", base_prompt,
            "--seed", str(seed),
            "--output", output_name
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=False, text=True)
            if result.returncode == 0:
                successful_generations += 1
                print(f"✅ Variation {i+1} completed successfully")
            else:
                print(f"❌ Variation {i+1} failed")
        except Exception as e:
            print(f"❌ Error generating variation {i+1}: {e}")
        
        time.sleep(1)  # Brief pause between generations
    
    print(f"\n✨ Batch generation complete!")
    print(f"📊 Success rate: {successful_generations}/{num_variations} variations")
    
    if successful_generations > 0:
        print("🖼️  Generated files:")
        # List all variation files that exist
        existing_files = []
        for i in range(num_variations):
            filename = f"variation_{i+1:02d}.png"
            if os.path.exists(filename):
                existing_files.append(filename)
        
        if existing_files:
            subprocess.run(["ls", "-la"] + existing_files)
        else:
            print("   No variation files found")

def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_generate.py 'your prompt here' [number_of_variations]")
        print("\nExample:")
        print("python batch_generate.py 'a serene mountain landscape at sunset' 4")
        sys.exit(1)
    
    prompt = sys.argv[1]
    num_vars = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    
    if num_vars < 1 or num_vars > 20:
        print("❌ Number of variations must be between 1 and 20")
        sys.exit(1)
    
    print("🚀 FLUX.1 Krea Batch Generator")
    print("=" * 40)
    print(f"Prompt: {prompt}")
    print(f"Variations: {num_vars}")
    print("=" * 40)
    
    # Check if the virtual environment is activated
    if "flux_env" not in sys.executable:
        print("⚠️  Warning: Virtual environment may not be activated")
        print("   Run: source flux_env/bin/activate")
    
    generate_variations(prompt, num_vars)

if __name__ == "__main__":
    main()