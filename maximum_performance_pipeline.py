#!/usr/bin/env python3
"""
FLUX.1 Krea Maximum Performance Pipeline for Apple Silicon M4 Pro
Integrates all optimizations: Neural Engine + Metal + Async + Thermal Management
"""

import torch
import argparse
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import psutil

# Import our optimization modules
from flux_neural_engine_accelerator import M4ProNeuralEngineOptimizer
from flux_metal_kernels import M4ProMetalOptimizer
from flux_async_pipeline_m4 import AsyncFluxPipeline
from thermal_performance_manager import ThermalPerformanceManager, PerformanceMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaximumPerformanceFluxPipeline:
    """Maximum performance FLUX pipeline with all M4 Pro optimizations"""
    
    def __init__(self, model_id: str = "black-forest-labs/FLUX.1-Krea-dev"):
        self.model_id = model_id
        
        # Initialize optimization components
        self.neural_engine_optimizer = M4ProNeuralEngineOptimizer()
        self.metal_optimizer = M4ProMetalOptimizer()
        self.async_pipeline = AsyncFluxPipeline(model_id)
        self.thermal_manager = ThermalPerformanceManager(PerformanceMode.ADAPTIVE)
        
        # Performance tracking
        self.generation_stats = {
            "total_generations": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "best_time": float('inf'),
            "optimizations_active": []
        }
        
        self.is_initialized = False
        
        # Setup thermal performance callback
        self.thermal_manager.add_performance_callback(self._thermal_performance_callback)
    
    async def initialize(self):
        """Initialize all optimization systems"""
        if self.is_initialized:
            return
        
        print("🚀 Initializing MAXIMUM Performance FLUX.1 Krea Pipeline")
        print("=" * 65)
        
        # Start thermal management
        self.thermal_manager.start()
        print("✅ Thermal performance manager active")
        
        # Initialize async pipeline
        await self.async_pipeline.initialize()
        print("✅ Async pipeline initialized")
        
        # Apply optimizations to pipeline once loaded
        await self._apply_all_optimizations()
        
        self.is_initialized = True
        print("🎉 MAXIMUM Performance pipeline ready!")
        
        # Show optimization summary
        await self._show_optimization_summary()
    
    async def _apply_all_optimizations(self):
        """Apply all available optimizations to the pipeline with proper error isolation"""
        optimizations_applied = []
        optimization_errors = []
        
        try:
            # Get the loaded pipeline
            pipeline = self.async_pipeline._get_pipeline()
            if pipeline is None:
                logger.error("Pipeline not loaded, aborting optimization application")
                self.generation_stats["optimizations_active"] = []
                self.generation_stats["optimization_errors"] = ["Pipeline not loaded"]
                return
            
            # Apply Neural Engine optimizations with isolation
            try:
                logger.info("Applying Neural Engine optimizations...")
                optimized_pipeline = self.neural_engine_optimizer.optimize_pipeline_components(pipeline)
                if optimized_pipeline is not None:
                    optimizations_applied.append("Neural Engine")
                    print("✅ Neural Engine optimizations applied")
                else:
                    optimization_errors.append("Neural Engine: No optimizations applied")
                    logger.info("ℹ️  Neural Engine optimizations skipped (not available)")
            except Exception as e:
                error_msg = f"Neural Engine optimization failed: {str(e)[:100]}..."
                optimization_errors.append(error_msg)
                logger.warning(error_msg)
            
            # Apply Metal optimizations with isolation
            try:
                logger.info("Applying Metal Performance Shaders optimizations...")
                metal_result = self.metal_optimizer.optimize_flux_pipeline_components(pipeline)
                if metal_result is not None:
                    optimizations_applied.append("Metal Performance Shaders")
                    print("✅ Metal Performance Shaders optimizations applied")
                else:
                    optimization_errors.append("Metal: Optimization returned None")
                    logger.info("ℹ️  Metal optimizations skipped (not available)")
            except Exception as e:
                error_msg = f"Metal optimization failed: {str(e)[:100]}..."
                optimization_errors.append(error_msg)
                logger.warning(error_msg)
            
            # Apply async optimizations with isolation
            try:
                logger.info("Configuring async pipeline optimizations...")
                # Async optimizations are already applied during pipeline initialization
                optimizations_applied.append("Async Pipeline")
                print("✅ Async pipeline optimizations applied")
            except Exception as e:
                error_msg = f"Async optimization failed: {str(e)[:100]}..."
                optimization_errors.append(error_msg)
                logger.warning(error_msg)
            
            # Apply thermal management with isolation
            try:
                logger.info("Configuring thermal management...")
                # Thermal management is already running
                if self.thermal_manager.monitor.monitoring_active:
                    optimizations_applied.append("Thermal Management")
                    print("✅ Thermal management optimizations applied")
                else:
                    optimization_errors.append("Thermal: Monitoring not active")
            except Exception as e:
                error_msg = f"Thermal management failed: {str(e)[:100]}..."
                optimization_errors.append(error_msg)
                logger.warning(error_msg)
            
            # Update statistics
            self.generation_stats["optimizations_active"] = optimizations_applied
            self.generation_stats["optimization_errors"] = optimization_errors
            
            # Log summary
            if optimizations_applied:
                logger.info(f"✅ Successfully applied {len(optimizations_applied)} optimizations: {', '.join(optimizations_applied)}")
            else:
                logger.warning("⚠️  No optimizations were successfully applied")
            
            if optimization_errors:
                logger.info(f"ℹ️  {len(optimization_errors)} optimization(s) skipped or failed")
            
        except Exception as e:
            error_msg = f"Critical error in optimization application: {e}"
            logger.error(error_msg)
            self.generation_stats["optimizations_active"] = optimizations_applied
            self.generation_stats["optimization_errors"] = optimization_errors + [error_msg]
    
    def _thermal_performance_callback(self, profile):
        """Handle thermal performance profile changes"""
        logger.info(f"🌡️  Thermal adjustment: {profile.max_cpu_threads} threads, "
                   f"{profile.max_gpu_utilization:.1%} GPU utilization")
    
    async def generate_maximum_performance(self, prompt: str, width: int = 1024, height: int = 1024,
                                         guidance_scale: float = 4.5, num_inference_steps: int = 28,
                                         seed: Optional[int] = None, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate image with maximum performance optimizations"""
        
        if not self.is_initialized:
            await self.initialize()
        
        print(f"\n🖼️  Generating with MAXIMUM Performance")
        print(f"📝 Prompt: {prompt}")
        print(f"📏 Size: {width}x{height}")
        print(f"⚙️  Steps: {num_inference_steps}, Guidance: {guidance_scale}")
        
        start_time = time.time()
        
        try:
            # Adjust parameters based on thermal profile
            current_profile = self.thermal_manager.get_current_profile()
            adjusted_steps = int(num_inference_steps * current_profile.inference_steps_scale)
            
            if adjusted_steps != num_inference_steps:
                print(f"🌡️  Thermal adjustment: {num_inference_steps} → {adjusted_steps} steps")
            
            # Submit async generation
            request_id = await self.async_pipeline.generate_async(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=adjusted_steps,
                seed=seed
            )
            
            print(f"📋 Generation request: {request_id}")
            print("⏳ Processing with maximum optimizations...")
            
            # Wait for result
            result = await self.async_pipeline.get_result(request_id)
            
            if result.error:
                print(f"❌ Generation failed: {result.error}")
                return {"success": False, "error": result.error}
            
            generation_time = time.time() - start_time
            
            # Save image if path provided
            if output_path and result.image:
                save_path = Path(output_path)
                result.image.save(save_path)
                print(f"💾 Saved: {save_path.absolute()}")
            
            # Update performance stats
            self._update_performance_stats(generation_time)
            
            # Show results
            print(f"🎉 Generation complete!")
            print(f"⏱️  Time: {generation_time:.1f}s")
            print(f"🚀 Speed: {(width * height) / generation_time:.0f} pixels/second")
            
            return {
                "success": True,
                "generation_time": generation_time,
                "pixels_per_second": (width * height) / generation_time,
                "request_id": request_id,
                "optimizations_used": self.generation_stats["optimizations_active"],
                "thermal_state": self.thermal_manager.monitor.get_current_metrics().thermal_state.value
            }
            
        except Exception as e:
            error_msg = f"Maximum performance generation failed: {e}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _update_performance_stats(self, generation_time: float):
        """Update performance statistics"""
        self.generation_stats["total_generations"] += 1
        self.generation_stats["total_time"] += generation_time
        self.generation_stats["average_time"] = (
            self.generation_stats["total_time"] / self.generation_stats["total_generations"]
        )
        
        if generation_time < self.generation_stats["best_time"]:
            self.generation_stats["best_time"] = generation_time
    
    async def _show_optimization_summary(self):
        """Show comprehensive optimization summary"""
        print("\n🔍 OPTIMIZATION SUMMARY")
        print("=" * 40)
        
        # Neural Engine status
        ne_summary = self.neural_engine_optimizer.get_optimization_summary()
        print(f"🧠 Neural Engine: {'✅ Active' if ne_summary['neural_engine_enabled'] else '❌ Unavailable'}")
        if ne_summary['neural_engine_enabled']:
            print(f"   Utilization: ~{ne_summary['estimated_utilization']:.0f}%")
            print(f"   Performance boost: {ne_summary['performance_boost']}")
        
        # Metal status
        metal_summary = self.metal_optimizer.get_metal_optimization_summary()
        print(f"⚡ Metal GPU: {'✅ Active' if metal_summary.get('metal_available', False) else '❌ Unavailable'}")
        if metal_summary.get('metal_available', False):
            print(f"   GPU utilization: {metal_summary['estimated_gpu_utilization']}")
            print(f"   Memory bandwidth: {metal_summary['estimated_bandwidth_utilization']}")
            print(f"   Performance boost: {metal_summary['performance_boost']}")
        
        # Async pipeline status
        async_summary = self.async_pipeline.get_performance_summary()
        print(f"🔄 Async Pipeline: ✅ Active")
        print(f"   Performance cores: {async_summary['performance_cores_used']}")
        print(f"   Efficiency cores: {async_summary['efficiency_cores_used']}")
        print(f"   Max concurrent: {async_summary['max_concurrent_generations']}")
        
        # Thermal management status
        thermal_summary = self.thermal_manager.get_status_summary()
        print(f"🌡️  Thermal Management: ✅ Active ({thermal_summary['performance_mode']})")
        print(f"   Current temp: {thermal_summary['thermal_status'].get('current_temp', 0):.1f}°C")
        print(f"   CPU threads: {thermal_summary['current_profile']['cpu_threads']}")
        
        # System info
        print(f"\n💻 System Information:")
        print(f"   CPU: Apple M4 Pro (8P + 4E cores)")
        print(f"   GPU: 20-core GPU")
        print(f"   Neural Engine: 16-core")
        print(f"   Memory: {psutil.virtual_memory().total // (1024**3)} GB unified")
        print(f"   Memory bandwidth: 273 GB/s")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        thermal_summary = self.thermal_manager.get_status_summary()
        async_summary = self.async_pipeline.get_performance_summary()
        
        return {
            "performance_stats": self.generation_stats,
            "thermal_status": thermal_summary,
            "async_pipeline": async_summary,
            "system_utilization": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_gb": psutil.virtual_memory().available / (1024**3)
            },
            "optimizations_summary": {
                "neural_engine": self.neural_engine_optimizer.get_optimization_summary(),
                "metal_gpu": self.metal_optimizer.get_metal_optimization_summary()
            }
        }
    
    async def cleanup(self):
        """Cleanup all resources"""
        self.thermal_manager.stop()
        print("✅ Maximum performance pipeline shutdown complete")

async def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='FLUX.1 Krea Maximum Performance Pipeline')
    parser.add_argument('--prompt', required=True, help='Text prompt for image generation')
    parser.add_argument('--output', default='maximum-performance-output.png', help='Output filename')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--guidance', type=float, default=4.5, help='Guidance scale')
    parser.add_argument('--steps', type=int, default=28, help='Number of inference steps')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    args = parser.parse_args()
    
    # Initialize maximum performance pipeline
    pipeline = MaximumPerformanceFluxPipeline()
    
    try:
        if args.benchmark:
            # Run benchmark
            print("🏃 Running Maximum Performance Benchmark")
            print("=" * 50)
            
            test_prompts = [
                "a cute cat sitting in a garden",
                "a majestic mountain landscape at sunset",
                "a futuristic city with flying cars"
            ]
            
            results = []
            for i, prompt in enumerate(test_prompts):
                print(f"\n🔄 Benchmark {i+1}/3: {prompt}")
                result = await pipeline.generate_maximum_performance(
                    prompt=prompt,
                    width=args.width,
                    height=args.height,
                    guidance_scale=args.guidance,
                    num_inference_steps=args.steps,
                    seed=42,
                    output_path=f"benchmark_{i+1}.png"
                )
                results.append(result)
            
            # Show benchmark results
            print(f"\n📊 BENCHMARK RESULTS")
            print("=" * 30)
            for i, result in enumerate(results):
                if result["success"]:
                    print(f"Test {i+1}: {result['generation_time']:.1f}s "
                          f"({result['pixels_per_second']:.0f} pixels/s)")
                else:
                    print(f"Test {i+1}: Failed - {result['error']}")
            
            avg_time = sum(r["generation_time"] for r in results if r["success"]) / len([r for r in results if r["success"]])
            print(f"\nAverage: {avg_time:.1f}s")
            
        else:
            # Single generation
            result = await pipeline.generate_maximum_performance(
                prompt=args.prompt,
                width=args.width,
                height=args.height,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                seed=args.seed,
                output_path=args.output
            )
            
            if not result["success"]:
                print(f"❌ Generation failed: {result['error']}")
            
        # Show final performance report
        report = pipeline.get_performance_report()
        print(f"\n📈 FINAL PERFORMANCE REPORT")
        print("=" * 40)
        print(f"Total generations: {report['performance_stats']['total_generations']}")
        print(f"Average time: {report['performance_stats']['average_time']:.1f}s")
        print(f"Best time: {report['performance_stats']['best_time']:.1f}s")
        print(f"Active optimizations: {', '.join(report['performance_stats']['optimizations_active'])}")
        
    except KeyboardInterrupt:
        print("\n⏸️  Generation interrupted")
    finally:
        await pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(main())