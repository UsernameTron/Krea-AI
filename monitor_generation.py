#!/usr/bin/env python3
"""
System monitoring during FLUX generation
Helps you understand how your M4 Pro handles the workload
"""

import psutil
import time
import subprocess
import sys
from threading import Thread
import os

class SystemMonitor:
    def __init__(self):
        self.monitoring = False
        self.peak_cpu = 0
        self.peak_memory = 0
        
    def start_monitoring(self):
        self.monitoring = True
        self.peak_cpu = 0
        self.peak_memory = 0
        monitor_thread = Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
    def stop_monitoring(self):
        self.monitoring = False
        
    def get_stats(self):
        """Return final statistics"""
        return {
            'peak_cpu': self.peak_cpu,
            'peak_memory': self.peak_memory
        }
        
    def _monitor_loop(self):
        print("\n📊 System Resource Monitor")
        print("=" * 60)
        print("CPU Usage | Memory Usage    | GPU Temp | Disk I/O")
        print("-" * 60)
        
        while self.monitoring:
            try:
                # Get system stats
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Track peaks
                self.peak_cpu = max(self.peak_cpu, cpu_percent)
                self.peak_memory = max(self.peak_memory, memory.percent)
                
                # Get disk I/O if available
                try:
                    disk_io = psutil.disk_io_counters()
                    disk_read = disk_io.read_bytes / (1024 * 1024)  # MB
                    disk_write = disk_io.write_bytes / (1024 * 1024)  # MB
                    disk_info = f"R:{disk_read:.0f}MB W:{disk_write:.0f}MB"
                except:
                    disk_info = "N/A"
                
                # Try to get GPU temperature (Apple Silicon specific)
                try:
                    # This is a placeholder - actual GPU temp monitoring on M4 Pro
                    # would require specialized tools or system calls
                    gpu_temp = "N/A"
                except:
                    gpu_temp = "N/A"
                
                # Display current stats
                print(f"\r{cpu_percent:6.1f}%   | "
                      f"{memory.used/1024**3:5.1f}GB/{memory.total/1024**3:5.1f}GB ({memory.percent:5.1f}%) | "
                      f"{gpu_temp:8s} | {disk_info:20s}", 
                      end="", flush=True)
                
                time.sleep(2)
                
            except Exception as e:
                print(f"\r❌ Monitoring error: {e}", end="", flush=True)
                time.sleep(2)

def main():
    if len(sys.argv) < 2:
        print("Usage: python monitor_generation.py 'your prompt here' [additional args]")
        print("\nExample:")
        print("python monitor_generation.py 'a cyberpunk cityscape' --steps 30 --guidance 4.5")
        sys.exit(1)
    
    prompt = sys.argv[1]
    additional_args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    print("🔍 FLUX.1 Krea Generation Monitor")
    print("=" * 50)
    print(f"Prompt: {prompt}")
    print(f"Additional args: {' '.join(additional_args)}")
    print("=" * 50)
    
    # Check if the virtual environment is activated
    if "flux_env" not in sys.executable:
        print("⚠️  Warning: Virtual environment may not be activated")
        print("   Run: source flux_env/bin/activate")
    
    # Start monitoring
    monitor = SystemMonitor()
    monitor.start_monitoring()
    
    start_time = time.time()
    
    try:
        # Build command
        cmd = ["python", "generate_image.py", "--prompt", prompt] + additional_args
        
        print(f"🚀 Starting generation with monitoring...")
        print(f"Command: {' '.join(cmd)}")
        
        # Run generation while monitoring
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        total_time = time.time() - start_time
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Get final stats
        stats = monitor.get_stats()
        
        print("\n\n" + "=" * 60)
        print("📈 GENERATION STATISTICS")
        print("=" * 60)
        print(f"Total Time:     {total_time:.1f} seconds")
        print(f"Peak CPU:       {stats['peak_cpu']:.1f}%")
        print(f"Peak Memory:    {stats['peak_memory']:.1f}%")
        print(f"Exit Code:      {result.returncode}")
        
        if result.returncode == 0:
            print("✅ Generation completed successfully!")
            
            # Try to show the generated image info
            output_files = [f for f in os.listdir('.') if f.endswith('.png') and 
                          (f.startswith('flux_generation') or f.startswith('monitored_generation'))]
            if output_files:
                latest_file = max(output_files, key=os.path.getmtime)
                file_size = os.path.getsize(latest_file) / (1024 * 1024)  # MB
                print(f"📁 Output file:  {latest_file} ({file_size:.1f} MB)")
        else:
            print("❌ Generation failed!")
            
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        print("\n\n⚠️  Generation interrupted by user")
    except Exception as e:
        monitor.stop_monitoring()
        print(f"\n\n❌ Error during monitored generation: {e}")
    
    print("\n🏁 Monitoring complete!")

if __name__ == "__main__":
    main()