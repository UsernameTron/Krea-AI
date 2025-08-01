#\!/usr/bin/env python3
"""
FLUX.1 Krea Complete Workflow System
Integrates all documented features into a unified creative workflow
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
import json
from datetime import datetime
import time

class FluxWorkflow:
    def __init__(self, optimization_level="balanced"):
        self.optimization_level = optimization_level
        self.session_log = []
        self.output_directory = Path("flux_outputs")
        self.output_directory.mkdir(exist_ok=True)
    
    def log_generation(self, workflow_type, params, output_path, generation_time):
        """Log generation details for workflow tracking"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "workflow_type": workflow_type,
            "parameters": params,
            "output_path": str(output_path),
            "generation_time": generation_time,
            "optimization_level": self.optimization_level
        }
        self.session_log.append(log_entry)
    
    def save_session_log(self):
        """Save session log for workflow analysis"""
        log_path = self.output_directory / f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, 'w') as f:
            json.dump(self.session_log, f, indent=2)
        print(f"📝 Session log saved: {log_path}")
    
    def text_to_image(self, prompt, **kwargs):
        """Standard text-to-image generation workflow"""
        from flux_advanced import FluxAdvancedGenerator
        
        generator = FluxAdvancedGenerator(optimization_level=self.optimization_level)
        
        start_time = time.time()
        image = generator.generate_image(prompt, **kwargs)
        generation_time = time.time() - start_time
        
        # Save with systematic naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = self.output_directory / f"txt2img_{timestamp}.png"
        image.save(output_path)
        
        self.log_generation("text_to_image", {"prompt": prompt, **kwargs}, output_path, generation_time)
        return output_path
    
    def create_variations(self, base_prompt, num_variations=4):
        """Create multiple variations of a concept"""
        print(f"🎨 Creating {num_variations} variations of: '{base_prompt}'")
        
        variations = []
        for i in range(num_variations):
            # Add subtle variation prompts
            variation_prompt = f"{base_prompt}, variation {i+1}, unique perspective"
            output_path = self.text_to_image(
                variation_prompt,
                seed=100 + i,  # Consistent but different seeds
                output=f"variation_{i+1:02d}.png"
            )
            variations.append(output_path)
        
        return variations
    
    def interactive_workflow(self):
        """Interactive workflow for creative exploration"""
        print("🎨 Welcome to FLUX.1 Krea Interactive Workflow!")
        print("Available commands:")
        print("  1. Generate image")
        print("  2. Create variations") 
        print("  3. Image-to-image transformation")
        print("  4. Inpainting")
        print("  5. Exit")
        
        while True:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                prompt = input("Enter your prompt: ")
                output_path = self.text_to_image(prompt)
                print(f"✅ Generated: {output_path}")
            
            elif choice == "2":
                base_prompt = input("Enter base prompt for variations: ")
                num_vars = int(input("Number of variations (default 4): ") or "4")
                variations = self.create_variations(base_prompt, num_vars)
                print(f"✅ Created {len(variations)} variations")
            
            elif choice == "3":
                print("Image-to-image transformation")
                # Implementation would use flux_img2img.py
                print("Use: python flux_img2img.py --prompt 'your prompt' --input_image path/to/image.jpg")
            
            elif choice == "4":
                print("Inpainting workflow")
                # Implementation would use flux_inpaint.py  
                print("Use: python flux_inpaint.py --prompt 'your prompt' --image path/to/image.jpg --mask path/to/mask.jpg")
            
            elif choice == "5":
                self.save_session_log()
                print("👋 Goodbye!")
                break
            
            else:
                print("Invalid choice. Please enter 1-5.")

def main():
    parser = argparse.ArgumentParser(description='FLUX.1 Krea Complete Workflow')
    parser.add_argument('--interactive', action='store_true', help='Start interactive mode')
    parser.add_argument('--optimization', choices=['speed', 'balanced', 'memory'], 
                       default='balanced', help='Optimization level')
    
    args = parser.parse_args()
    
    workflow = FluxWorkflow(optimization_level=args.optimization)
    
    if args.interactive:
        workflow.interactive_workflow()
    else:
        print("Use --interactive for the complete workflow experience")
        print("Or use individual scripts: flux_advanced.py, flux_img2img.py, flux_inpaint.py")

if __name__ == "__main__":
    main()