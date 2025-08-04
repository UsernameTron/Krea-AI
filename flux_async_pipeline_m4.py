#!/usr/bin/env python3
"""
FLUX.1 Krea Async Pipeline for Apple Silicon M4 Pro
Asynchronous processing with maximum core utilization
"""

import torch
import asyncio
import concurrent.futures
import threading
import time
import logging
from typing import Optional, Dict, Any, List, Callable, Awaitable
from pathlib import Path
from contextlib import asynccontextmanager
import multiprocessing as mp
from dataclasses import dataclass
import queue

logger = logging.getLogger(__name__)

@dataclass
class AsyncGenerationRequest:
    """Async generation request structure"""
    prompt: str
    width: int = 1024
    height: int = 1024
    guidance_scale: float = 4.5
    num_inference_steps: int = 28
    seed: Optional[int] = None
    request_id: str = ""
    priority: int = 1

@dataclass
class AsyncGenerationResult:
    """Async generation result structure"""
    request_id: str
    image: Optional[Any] = None
    generation_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

class M4ProAsyncScheduler:
    """Async scheduler optimized for M4 Pro architecture"""
    
    def __init__(self, max_concurrent_generations: int = 2):
        self.max_concurrent_generations = max_concurrent_generations
        self.performance_cores = 8  # M4 Pro performance cores
        self.efficiency_cores = 4   # M4 Pro efficiency cores
        self.generation_queue = asyncio.Queue()
        self.active_generations = {}
        self.performance_stats = {}
        
        # Configure thread pools for different task types
        self.cpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.performance_cores, 
            thread_name_prefix="flux_cpu"
        )
        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.efficiency_cores,
            thread_name_prefix="flux_io"
        )
        
    async def submit_generation(self, request: AsyncGenerationRequest) -> str:
        """Submit async generation request"""
        request.request_id = f"gen_{int(time.time()*1000)}_{id(request)}"
        await self.generation_queue.put(request)
        logger.info(f"📥 Queued generation: {request.request_id}")
        return request.request_id
    
    async def process_generation_queue(self, pipeline_loader: Callable):
        """Process generation queue asynchronously"""
        while True:
            try:
                # Wait for generation request
                request = await self.generation_queue.get()
                
                # Check if we can process (respect concurrency limits)
                if len(self.active_generations) >= self.max_concurrent_generations:
                    await asyncio.sleep(0.1)
                    await self.generation_queue.put(request)  # Re-queue
                    continue
                
                # Process generation asynchronously
                task = asyncio.create_task(
                    self._process_single_generation(request, pipeline_loader)
                )
                self.active_generations[request.request_id] = task
                
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_single_generation(self, request: AsyncGenerationRequest, 
                                       pipeline_loader: Callable) -> AsyncGenerationResult:
        """Process single generation request"""
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting generation: {request.request_id}")
            
            # Load pipeline asynchronously (if needed)
            pipeline = await self._load_pipeline_async(pipeline_loader)
            
            # Generate image asynchronously
            image = await self._generate_image_async(pipeline, request)
            
            generation_time = time.time() - start_time
            
            result = AsyncGenerationResult(
                request_id=request.request_id,
                image=image,
                generation_time=generation_time,
                metadata={
                    "prompt": request.prompt,
                    "dimensions": f"{request.width}x{request.height}",
                    "steps": request.num_inference_steps,
                    "guidance": request.guidance_scale
                }
            )
            
            logger.info(f"✅ Completed generation: {request.request_id} in {generation_time:.1f}s")
            return result
            
        except Exception as e:
            error_msg = f"Generation failed: {e}"
            logger.error(f"❌ {error_msg} for {request.request_id}")
            
            result = AsyncGenerationResult(
                request_id=request.request_id,
                error=error_msg,
                generation_time=time.time() - start_time
            )
            return result
            
        finally:
            # Remove from active generations
            self.active_generations.pop(request.request_id, None)
    
    async def _load_pipeline_async(self, pipeline_loader: Callable):
        """Load pipeline asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.cpu_executor, pipeline_loader)
    
    async def _generate_image_async(self, pipeline, request: AsyncGenerationRequest):
        """Generate image asynchronously"""
        loop = asyncio.get_event_loop()
        
        def generate_sync():
            # Set up generator for reproducible results
            generator = None
            if request.seed is not None:
                generator = torch.Generator()
                generator.manual_seed(request.seed)
            
            # Generate with optimized inference mode
            with torch.inference_mode():
                result = pipeline(
                    prompt=request.prompt,
                    height=request.height,
                    width=request.width,
                    guidance_scale=request.guidance_scale,
                    num_inference_steps=request.num_inference_steps,
                    generator=generator,
                    return_dict=True
                )
            return result.images[0]
        
        return await loop.run_in_executor(self.cpu_executor, generate_sync)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue and processing status"""
        return {
            "queue_size": self.generation_queue.qsize(),
            "active_generations": len(self.active_generations),
            "max_concurrent": self.max_concurrent_generations,
            "active_request_ids": list(self.active_generations.keys())
        }

class AsyncFluxPipeline:
    """Async FLUX pipeline with M4 Pro optimizations"""
    
    def __init__(self, model_id: str = "black-forest-labs/FLUX.1-Krea-dev"):
        self.model_id = model_id
        self.pipeline = None
        self.scheduler = M4ProAsyncScheduler()
        self.is_loaded = False
        self.load_lock = asyncio.Lock()
        
        # Performance monitoring
        self.generation_count = 0
        self.total_generation_time = 0.0
        self.average_generation_time = 0.0
    
    async def initialize(self):
        """Initialize async pipeline"""
        await self._load_pipeline()
        
        # Start queue processor
        asyncio.create_task(
            self.scheduler.process_generation_queue(self._get_pipeline)
        )
        
        logger.info("✅ Async FLUX pipeline initialized")
    
    async def _load_pipeline(self):
        """Load pipeline asynchronously"""
        async with self.load_lock:
            if self.is_loaded:
                return
            
            logger.info("📥 Loading FLUX pipeline asynchronously...")
            
            def load_sync():
                from diffusers import FluxPipeline
                
                pipeline = FluxPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    low_cpu_mem_usage=True
                )
                
                # Apply optimizations
                if hasattr(pipeline, 'enable_attention_slicing'):
                    pipeline.enable_attention_slicing("auto")
                
                if hasattr(pipeline, 'enable_vae_tiling'):
                    pipeline.enable_vae_tiling()
                
                if hasattr(pipeline, 'enable_vae_slicing'):
                    pipeline.enable_vae_slicing()
                
                return pipeline
            
            loop = asyncio.get_event_loop()
            self.pipeline = await loop.run_in_executor(
                self.scheduler.cpu_executor, load_sync
            )
            
            self.is_loaded = True
            logger.info("✅ Pipeline loaded asynchronously")
    
    def _get_pipeline(self):
        """Get loaded pipeline (for scheduler)"""
        return self.pipeline
    
    async def generate_async(self, prompt: str, width: int = 1024, height: int = 1024,
                           guidance_scale: float = 4.5, num_inference_steps: int = 28,
                           seed: Optional[int] = None, priority: int = 1) -> str:
        """Submit async generation request"""
        if not self.is_loaded:
            await self.initialize()
        
        request = AsyncGenerationRequest(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            priority=priority
        )
        
        request_id = await self.scheduler.submit_generation(request)
        return request_id
    
    async def get_result(self, request_id: str, timeout: float = 300.0) -> AsyncGenerationResult:
        """Get generation result by request ID"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if generation is complete
            if request_id in self.scheduler.active_generations:
                task = self.scheduler.active_generations[request_id]
                if task.done():
                    result = await task
                    self._update_performance_stats(result)
                    return result
            
            await asyncio.sleep(0.1)
        
        # Timeout
        return AsyncGenerationResult(
            request_id=request_id,
            error="Generation timeout"
        )
    
    def _update_performance_stats(self, result: AsyncGenerationResult):
        """Update performance statistics"""
        if result.error is None and result.generation_time > 0:
            self.generation_count += 1
            self.total_generation_time += result.generation_time
            self.average_generation_time = self.total_generation_time / self.generation_count
    
    async def generate_batch_async(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate multiple images asynchronously"""
        request_ids = []
        for prompt in prompts:
            request_id = await self.generate_async(prompt, **kwargs)
            request_ids.append(request_id)
        
        return request_ids
    
    async def wait_for_batch(self, request_ids: List[str], 
                           timeout: float = 300.0) -> List[AsyncGenerationResult]:
        """Wait for batch of generations to complete"""
        results = []
        
        # Use asyncio.gather for concurrent waiting
        tasks = [self.get_result(req_id, timeout) for req_id in request_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(AsyncGenerationResult(
                    request_id=request_ids[i],
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        queue_status = self.scheduler.get_queue_status()
        
        return {
            "pipeline_loaded": self.is_loaded,
            "total_generations": self.generation_count,
            "average_generation_time": f"{self.average_generation_time:.1f}s",
            "queue_status": queue_status,
            "performance_cores_used": self.scheduler.performance_cores,
            "efficiency_cores_used": self.scheduler.efficiency_cores,
            "max_concurrent_generations": self.scheduler.max_concurrent_generations
        }

# Usage example and testing
async def test_async_pipeline():
    """Test async pipeline functionality"""
    print("🚀 Testing Async FLUX Pipeline for M4 Pro")
    print("=" * 50)
    
    pipeline = AsyncFluxPipeline()
    await pipeline.initialize()
    
    # Submit multiple generations
    prompts = [
        "a cute cat sitting in a garden",
        "a majestic mountain landscape",
        "a futuristic city at sunset"
    ]
    
    print(f"\n📝 Submitting {len(prompts)} async generations...")
    request_ids = await pipeline.generate_batch_async(prompts, seed=42)
    
    print(f"📋 Request IDs: {request_ids}")
    
    # Wait for results
    print("\n⏳ Waiting for generations to complete...")
    results = await pipeline.wait_for_batch(request_ids)
    
    # Show results
    for i, result in enumerate(results):
        if result.error:
            print(f"❌ Generation {i+1}: {result.error}")
        else:
            print(f"✅ Generation {i+1}: {result.generation_time:.1f}s")
    
    # Performance summary
    summary = pipeline.get_performance_summary()
    print(f"\n📊 Performance Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_async_pipeline())