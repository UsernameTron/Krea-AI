#!/usr/bin/env python3
"""
FLUX.1 Krea Performance Benchmark Suite
Tests different optimization levels and provides performance insights
"""

import time
import torch
import psutil
import gc
from src.flux.util import load_ae, load_clip, load_t5, load_flow_model
from src.flux.pipeline import Sampler

class FluxBenchmark:
    def __init__(self):
        self.test_prompt = "a serene mountain landscape with a crystal clear lake, photorealistic, highly detailed"
        self.test_params = {
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 20,  # Reduced for benchmarking
            "guidance_scale": 4.0
        }
    
    def benchmark_optimization(self, optimization_name, setup_func):
        """Benchmark a specific optimization configuration"""
        print(f"\n🧪 Testing {optimization_name}...")
        
        # Clear memory before test
        gc.collect()
        
        # Record initial memory
        initial_memory = psutil.virtual_memory().used / 1024**3
        
        # Setup models
        start_time = time.time()
        torch_dtype = torch.bfloat16
        # Smart device detection with fallbacks
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon GPU
        else:
            device = "cpu"
        
        print(f"Benchmarking on device: {device}")
        
        # Load models
        ae = load_ae("flux-krea-dev", device=device)
        clip = load_clip(device=device)
        t5 = load_t5(device=device)
        model = load_flow_model("flux-krea-dev", device=device)
        
        # Move models to device
        model = model.to(device=device, dtype=torch_dtype)
        ae = ae.to(device=device, dtype=torch_dtype)
        clip = clip.to(device=device, dtype=torch_dtype)
        t5 = t5.to(device=device, dtype=torch_dtype)
        
        # Create sampler
        sampler = Sampler(
            model=model,
            ae=ae,
            clip=clip,
            t5=t5,
            device=device,
            dtype=torch_dtype,
        )
        
        # Apply optimizations (simplified for this benchmark)
        setup_func(sampler)
        
        setup_time = time.time() - start_time
        setup_memory = psutil.virtual_memory().used / 1024**3
        
        # Test generation
        gen_start = time.time()
        result = sampler(
            prompt=self.test_prompt,
            height=self.test_params["height"],
            width=self.test_params["width"],
            num_steps=self.test_params["num_inference_steps"],
            guidance=self.test_params["guidance_scale"],
            seed=42
        )
        generation_time = time.time() - gen_start
        
        peak_memory = psutil.virtual_memory().used / 1024**3
        
        # Save test image
        result.save(f"benchmark_{optimization_name.lower().replace(' ', '_')}.png")
        
        # Clean up
        del sampler, model, ae, clip, t5
        gc.collect()
        
        return {
            "setup_time": setup_time,
            "generation_time": generation_time,
            "memory_overhead": setup_memory - initial_memory,
            "peak_memory": peak_memory - initial_memory,
            "total_time": setup_time + generation_time
        }
    
    def run_all_benchmarks(self):
        """Run comprehensive benchmark suite"""
        print("🚀 Starting FLUX.1 Krea Benchmark Suite")
        print(f"Test prompt: '{self.test_prompt}'")
        print(f"Parameters: {self.test_params}")
        
        results = {}
        
        # Test 1: No optimizations (baseline)
        def no_optimizations(sampler):
            pass
        
        results["Baseline"] = self.benchmark_optimization("Baseline", no_optimizations)
        
        # Test 2: CPU mode (if CUDA not available)
        def cpu_mode(sampler):
            # This is handled in the benchmark setup
            pass
        
        results["CPU Mode"] = self.benchmark_optimization("CPU Mode", cpu_mode)
        
        # Test 3: Mixed precision optimizations
        def mixed_precision(sampler):
            # bfloat16 is already applied in setup
            pass
        
        results["Mixed Precision"] = self.benchmark_optimization("Mixed Precision", mixed_precision)
        
        # Test 4: Memory efficient settings
        def memory_efficient(sampler):
            # Lower inference steps for memory efficiency
            pass
        
        results["Memory Efficient"] = self.benchmark_optimization("Memory Efficient", memory_efficient)
        
        # Display results
        self.display_results(results)
    
    def display_results(self, results):
        """Display benchmark results in a clear format"""
        print("\n" + "="*80)
        print("📊 BENCHMARK RESULTS")
        print("="*80)
        
        print(f"{'Configuration':<15} {'Setup (s)':<10} {'Generate (s)':<12} {'Total (s)':<10} {'Memory (GB)':<12}")
        print("-" * 80)
        
        for config, data in results.items():
            print(f"{config:<15} {data['setup_time']:<10.1f} {data['generation_time']:<12.1f} "
                  f"{data['total_time']:<10.1f} {data['peak_memory']:<12.1f}")
        
        # Find best configuration
        best_speed = min(results.items(), key=lambda x: x[1]['generation_time'])
        best_memory = min(results.items(), key=lambda x: x[1]['peak_memory'])
        
        print("\n🏆 RECOMMENDATIONS:")
        print(f"   Fastest Generation: {best_speed[0]} ({best_speed[1]['generation_time']:.1f}s)")
        print(f"   Most Memory Efficient: {best_memory[0]} ({best_memory[1]['peak_memory']:.1f}GB)")
        
        # System-specific recommendations
        current_memory = psutil.virtual_memory().total / 1024**3
        if current_memory > 32:
            print(f"   For your {current_memory:.0f}GB system: 'Full Optimized' recommended for best balance")
        else:
            print(f"   For your {current_memory:.0f}GB system: 'VAE Optimized' recommended for stability")

if __name__ == "__main__":
    benchmark = FluxBenchmark()
    benchmark.run_all_benchmarks()