#!/usr/bin/env python3
"""
FLUX.1 Krea Batch Workflow Manager
Professional batch processing with advanced optimizations for creative workflows
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import torch
from diffusers import FluxPipeline
from flux_advanced import FluxAdvancedGenerator

class FluxBatchWorkflow:
    def __init__(self, optimization_level="balanced", output_dir="batch_output"):
        self.optimization_level = optimization_level
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.generator = FluxAdvancedGenerator(optimization_level=optimization_level)
        
    def process_prompt_variations(self, base_prompt, variations, **kwargs):
        """Process multiple variations of a prompt"""
        results = []
        
        print(f"🎨 Processing {len(variations)} prompt variations...")
        
        for i, variation in enumerate(variations, 1):
            print(f"\n📝 Variation {i}/{len(variations)}: {variation}")
            
            try:
                image = self.generator.generate_image(variation, **kwargs)
                
                # Generate filename
                safe_prompt = "".join(c if c.isalnum() or c in "._- " else "" for c in variation)[:50]
                filename = f"variation_{i:02d}_{safe_prompt}.png"
                output_path = self.output_dir / filename
                
                image.save(output_path)
                
                results.append({
                    "variation": i,
                    "prompt": variation,
                    "filename": filename,
                    "success": True
                })
                
                print(f"✅ Saved: {filename}")
                
            except Exception as e:
                print(f"❌ Failed: {str(e)}")
                results.append({
                    "variation": i,
                    "prompt": variation,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def process_parameter_sweep(self, prompt, parameter_ranges, **base_kwargs):
        """Process a prompt with different parameter combinations"""
        results = []
        
        print(f"🔬 Parameter sweep for: '{prompt}'")
        
        # Generate all combinations
        combinations = []
        for param_name, values in parameter_ranges.items():
            if not combinations:
                combinations = [{param_name: v} for v in values]
            else:
                new_combinations = []
                for combo in combinations:
                    for value in values:
                        new_combo = combo.copy()
                        new_combo[param_name] = value
                        new_combinations.append(new_combo)
                combinations = new_combinations
        
        print(f"📊 Testing {len(combinations)} parameter combinations...")
        
        for i, params in enumerate(combinations, 1):
            print(f"\n🧪 Combination {i}/{len(combinations)}: {params}")
            
            try:
                # Merge parameters
                generation_params = {**base_kwargs, **params}
                
                image = self.generator.generate_image(prompt, **generation_params)
                
                # Generate descriptive filename
                param_str = "_".join([f"{k}{v}" for k, v in params.items()])
                filename = f"sweep_{i:02d}_{param_str}.png"
                output_path = self.output_dir / filename
                
                image.save(output_path)
                
                results.append({
                    "combination": i,
                    "parameters": params,
                    "filename": filename,
                    "success": True
                })
                
                print(f"✅ Saved: {filename}")
                
            except Exception as e:
                print(f"❌ Failed: {str(e)}")
                results.append({
                    "combination": i,
                    "parameters": params,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def process_style_exploration(self, subject, styles, **kwargs):
        """Explore different artistic styles for a subject"""
        style_prompts = [f"{subject}, {style}" for style in styles]
        return self.process_prompt_variations(style_prompts, style_prompts, **kwargs)
    
    def save_batch_report(self, results, batch_type="batch"):
        """Save detailed batch processing report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"{batch_type}_report_{timestamp}.json"
        
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]
        
        report = {
            "timestamp": timestamp,
            "batch_type": batch_type,
            "optimization_level": self.optimization_level,
            "summary": {
                "total": len(results),
                "successful": len(successful),
                "failed": len(failed)
            },
            "results": results
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Batch report saved: {report_file}")
        print(f"✅ Success rate: {len(successful)}/{len(results)} ({100*len(successful)/len(results):.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='FLUX Batch Workflow Manager')
    parser.add_argument('--mode', choices=['variations', 'sweep', 'styles'], required=True,
                       help='Batch processing mode')
    parser.add_argument('--prompt', help='Base prompt (required for sweep and styles modes)')
    parser.add_argument('--prompts_file', help='JSON file with prompt variations')
    parser.add_argument('--styles_file', help='JSON file with style variations')
    parser.add_argument('--output_dir', default='batch_output', help='Output directory')
    parser.add_argument('--optimization', choices=['speed', 'balanced', 'memory'], 
                       default='balanced', help='Optimization level')
    
    # Generation parameters
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=28)
    parser.add_argument('--guidance', type=float, default=4.0)
    
    args = parser.parse_args()
    
    # Initialize workflow manager
    workflow = FluxBatchWorkflow(
        optimization_level=args.optimization,
        output_dir=args.output_dir
    )
    
    # Common generation parameters
    gen_params = {
        'width': args.width,
        'height': args.height,
        'steps': args.steps,
        'guidance': args.guidance
    }
    
    if args.mode == 'variations':
        if args.prompts_file:
            with open(args.prompts_file, 'r') as f:
                prompts_data = json.load(f)
            variations = prompts_data.get('prompts', [])
        else:
            print("❌ Variations mode requires --prompts_file")
            return
        
        results = workflow.process_prompt_variations(args.prompt or "", variations, **gen_params)
        workflow.save_batch_report(results, "variations")
        
    elif args.mode == 'sweep':
        if not args.prompt:
            print("❌ Sweep mode requires --prompt")
            return
        
        # Example parameter sweep
        parameter_ranges = {
            'guidance': [3.0, 4.0, 5.0],
            'steps': [20, 28, 35]
        }
        
        results = workflow.process_parameter_sweep(args.prompt, parameter_ranges, **gen_params)
        workflow.save_batch_report(results, "parameter_sweep")
        
    elif args.mode == 'styles':
        if not args.prompt:
            print("❌ Styles mode requires --prompt")
            return
        
        if args.styles_file:
            with open(args.styles_file, 'r') as f:
                styles_data = json.load(f)
            styles = styles_data.get('styles', [])
        else:
            # Default artistic styles
            styles = [
                "photorealistic, highly detailed",
                "digital art, vibrant colors",
                "oil painting, impressionist style",
                "watercolor painting, soft brushstrokes",
                "anime art style, studio ghibli",
                "cyberpunk aesthetic, neon lighting",
                "vintage photography, film grain",
                "pencil sketch, detailed linework"
            ]
        
        results = workflow.process_style_exploration(args.prompt, styles, **gen_params)
        workflow.save_batch_report(results, "style_exploration")
    
    print(f"\n🎉 Batch processing complete! Check {args.output_dir} for results.")

if __name__ == "__main__":
    main()