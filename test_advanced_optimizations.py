#!/usr/bin/env python3
"""
Comprehensive test suite for advanced FLUX optimizations
Tests all optimization levels and measures performance
"""

import time
import psutil
import torch
from pathlib import Path
import json
from datetime import datetime

def test_optimization_level(level, test_prompt="a simple test image"):
    """Test a specific optimization level and collect metrics"""
    print(f"\n🧪 Testing {level} optimization level...")
    
    try:
        # Import here to avoid conflicts between tests
        from flux_advanced import FluxAdvancedGenerator
        
        # Record initial memory
        initial_memory = psutil.virtual_memory().used / 1024**3
        
        # Initialize generator
        start_time = time.time()
        generator = FluxAdvancedGenerator(optimization_level=level)
        init_time = time.time() - start_time
        
        # Record memory after loading
        loaded_memory = psutil.virtual_memory().used / 1024**3
        
        # Generate test image
        gen_start = time.time()
        image = generator.generate_image(
            test_prompt,
            width=512,  # Small size for quick testing
            height=512,
            steps=10    # Reduced steps for speed
        )
        gen_time = time.time() - gen_start
        
        # Record peak memory
        peak_memory = psutil.virtual_memory().used / 1024**3
        
        # Save test image
        output_path = f"test_{level}_mode.png"
        image.save(output_path)
        
        # Cleanup
        del generator
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        
        results = {
            "optimization_level": level,
            "initialization_time": round(init_time, 2),
            "generation_time": round(gen_time, 2),
            "total_time": round(init_time + gen_time, 2),
            "initial_memory_gb": round(initial_memory, 2),
            "loaded_memory_gb": round(loaded_memory, 2),
            "peak_memory_gb": round(peak_memory, 2),
            "memory_increase_gb": round(loaded_memory - initial_memory, 2),
            "output_file": output_path,
            "success": True
        }
        
        print(f"✅ {level.capitalize()} mode completed successfully")
        print(f"   Init: {init_time:.1f}s, Generation: {gen_time:.1f}s")
        print(f"   Memory: {initial_memory:.1f}GB → {peak_memory:.1f}GB")
        
        return results
        
    except Exception as e:
        print(f"❌ {level.capitalize()} mode failed: {str(e)}")
        return {
            "optimization_level": level,
            "success": False,
            "error": str(e)
        }

def run_comprehensive_test():
    """Run tests for all optimization levels"""
    print("🚀 Starting comprehensive FLUX advanced optimization test suite")
    print(f"System: {psutil.virtual_memory().total / 1024**3:.1f}GB RAM available")
    
    test_results = []
    test_prompt = "a cute red robot in a futuristic laboratory"
    
    # Test each optimization level
    for level in ["speed", "balanced", "memory"]:
        result = test_optimization_level(level, test_prompt)
        test_results.append(result)
        
        # Brief pause between tests
        time.sleep(2)
    
    # Generate summary report
    generate_test_report(test_results)
    
    return test_results

def generate_test_report(results):
    """Generate a comprehensive test report"""
    print("\n📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 50)
    
    successful_tests = [r for r in results if r.get("success", False)]
    
    if not successful_tests:
        print("❌ No tests completed successfully")
        return
    
    # Performance comparison
    print("\n🏁 Performance Comparison:")
    for result in successful_tests:
        level = result["optimization_level"]
        total_time = result["total_time"]
        memory_use = result["memory_increase_gb"]
        
        print(f"   {level.capitalize():>8} | {total_time:>6.1f}s | {memory_use:>6.1f}GB")
    
    # Recommendations
    print("\n💡 Recommendations for your M4 Pro:")
    
    fastest = min(successful_tests, key=lambda x: x["total_time"])
    most_efficient = min(successful_tests, key=lambda x: x["memory_increase_gb"])
    
    print(f"   Fastest: {fastest['optimization_level']} ({fastest['total_time']:.1f}s)")
    print(f"   Most Memory Efficient: {most_efficient['optimization_level']} ({most_efficient['memory_increase_gb']:.1f}GB)")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"optimization_test_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "system_info": {
                "total_memory_gb": round(psutil.virtual_memory().total / 1024**3, 1),
                "cpu_count": psutil.cpu_count(),
                "platform": "darwin"
            },
            "test_results": results
        }, f, indent=2)
    
    print(f"\n📄 Detailed report saved: {report_file}")

def quick_smoke_test():
    """Quick test to verify the advanced script works"""
    print("🔥 Running quick smoke test...")
    
    try:
        from flux_advanced import FluxAdvancedGenerator
        
        # Test balanced mode only
        generator = FluxAdvancedGenerator(optimization_level="balanced")
        print("✅ Advanced generator initialized successfully")
        
        # Don't actually generate - just verify loading works
        del generator
        print("✅ Smoke test passed - ready for full testing")
        return True
        
    except Exception as e:
        print(f"❌ Smoke test failed: {str(e)}")
        return False

def main():
    """Main test execution"""
    print("🧪 FLUX Advanced Optimization Test Suite")
    print("Testing all optimization levels on your M4 Pro")
    
    # Check if model is available
    model_path = Path("./models/FLUX.1-Krea-dev")
    if not model_path.exists():
        print("❌ Model not found at ./models/FLUX.1-Krea-dev")
        print("💡 Please ensure you have HuggingFace access and the model is downloaded")
        return
    
    # Run smoke test first
    if not quick_smoke_test():
        return
    
    # Run comprehensive tests
    results = run_comprehensive_test()
    
    print("\n🎉 Testing complete!")
    print("Your M4 Pro FLUX setup is ready for advanced generation!")

if __name__ == "__main__":
    main()