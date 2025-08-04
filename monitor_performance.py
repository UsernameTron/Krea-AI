#!/usr/bin/env python3
"""
Real-time performance monitoring for Apple Silicon optimization
"""

import torch
import psutil
import time
import threading
from datetime import datetime

class M4ProMonitor:
    def __init__(self):
        self.monitoring = False
        self.stats = []
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        print("🔍 Started Apple Silicon performance monitoring")
    
    def stop_monitoring(self):
        """Stop monitoring and show summary"""
        self.monitoring = False
        self._show_summary()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            stats = {
                'timestamp': datetime.now(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            }
            
            if torch.backends.mps.is_available():
                stats['mps_allocated_mb'] = torch.mps.current_allocated_memory() / (1024**2)
                stats['mps_cached_mb'] = torch.mps.current_cached_memory() / (1024**2)
            
            self.stats.append(stats)
            time.sleep(1)
    
    def _show_summary(self):
        """Show performance summary"""
        if not self.stats:
            return
        
        print("\n🍎 Apple Silicon Performance Summary")
        print("=" * 40)
        
        avg_cpu = sum(s['cpu_percent'] for s in self.stats) / len(self.stats)
        max_memory = max(s['memory_percent'] for s in self.stats)
        min_available = min(s['memory_available_gb'] for s in self.stats)
        
        print(f"Average CPU usage: {avg_cpu:.1f}%")
        print(f"Peak memory usage: {max_memory:.1f}%")
        print(f"Minimum available memory: {min_available:.1f} GB")
        
        if torch.backends.mps.is_available() and any('mps_allocated_mb' in s for s in self.stats):
            max_mps = max(s.get('mps_allocated_mb', 0) for s in self.stats)
            print(f"Peak MPS memory: {max_mps:.1f} MB")

# Usage example
if __name__ == "__main__":
    monitor = M4ProMonitor()
    monitor.start_monitoring()
    
    try:
        input("Press Enter to stop monitoring...")
    except KeyboardInterrupt:
        pass
    
    monitor.stop_monitoring()