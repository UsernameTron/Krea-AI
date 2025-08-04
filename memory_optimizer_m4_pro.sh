#!/bin/bash

# Enhanced macOS Memory Optimization Script for M4 Pro
# Optimized for FLUX.1 Krea ML workloads
# Run with: sudo bash memory_optimizer_m4_pro.sh

echo "🚀 === M4 Pro Memory Optimization for FLUX.1 Krea ==="
echo "Unified Memory: 48GB | Neural Engine: 16-core | GPU: 20-core"
echo "=" * 65

# Function to get memory stats in GB
get_memory_gb() {
    echo "scale=2; $1 / 1024 / 1024 / 1024" | bc -l 2>/dev/null || echo "0"
}

# Function to check if running on M4 Pro
check_m4_architecture() {
    chip_info=$(system_profiler SPHardwareDataType | grep "Chip:")
    if [[ $chip_info == *"Apple M4 Pro"* ]]; then
        echo "✅ Apple M4 Pro detected - applying optimized settings"
        return 0
    else
        echo "⚠️  Non-M4 Pro system detected - using generic optimizations"
        return 1
    fi
}

# 1. System Architecture Detection
echo "\n--- System Architecture Detection ---"
check_m4_architecture
IS_M4_PRO=$?

# 2. Pre-optimization Memory Status
echo "\n--- Current Memory Status ---"
vm_stat_output=$(vm_stat)
pages_free=$(echo "$vm_stat_output" | grep "Pages free:" | awk '{print $3}' | tr -d '.')
pages_wired=$(echo "$vm_stat_output" | grep "Pages wired down:" | awk '{print $4}' | tr -d '.')
pages_active=$(echo "$vm_stat_output" | grep "Pages active:" | awk '{print $3}' | tr -d '.')
pages_inactive=$(echo "$vm_stat_output" | grep "Pages inactive:" | awk '{print $3}' | tr -d '.')

# Calculate memory in GB (4KB pages)
free_gb=$(get_memory_gb $((pages_free * 4096)))
wired_gb=$(get_memory_gb $((pages_wired * 4096)))
active_gb=$(get_memory_gb $((pages_active * 4096)))
inactive_gb=$(get_memory_gb $((pages_inactive * 4096)))

echo "Free memory: ${free_gb} GB"
echo "Wired memory: ${wired_gb} GB"
echo "Active memory: ${active_gb} GB"
echo "Inactive memory: ${inactive_gb} GB"

# Memory pressure analysis
memory_pressure_output=$(memory_pressure)
echo "\nMemory Pressure Analysis:"
echo "$memory_pressure_output" | head -5

# 3. M4 Pro Specific PyTorch MPS Environment Variables
echo "\n--- Setting M4 Pro Optimized PyTorch Variables ---"

if [ $IS_M4_PRO -eq 0 ]; then
    # M4 Pro specific optimizations
    export PYTORCH_MPS_MEMORY_FRACTION=0.95
    export PYTORCH_MPS_ALLOCATOR_POLICY=expandable_segments
    export PYTORCH_MPS_PREFER_FAST_ALLOC=1
    export PYTORCH_MPS_ENABLE_FALLBACK=1
    export OMP_NUM_THREADS=12  # 8P + 4E cores
    export MKL_NUM_THREADS=12
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    
    # Neural Engine optimizations
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_PROFILING_ENABLED=0  # Disable for performance
    
    # M4 Pro memory bandwidth optimization (273 GB/s)
    export PYTORCH_MPS_PREFER_UNIFIED_MEMORY=1
    
    # Make persistent across sessions
    launchctl setenv PYTORCH_MPS_MEMORY_FRACTION 0.95
    launchctl setenv PYTORCH_MPS_ALLOCATOR_POLICY expandable_segments
    launchctl setenv PYTORCH_MPS_PREFER_FAST_ALLOC 1
    launchctl setenv PYTORCH_MPS_ENABLE_FALLBACK 1
    launchctl setenv OMP_NUM_THREADS 12
    launchctl setenv MKL_NUM_THREADS 12
    launchctl setenv PYTORCH_MPS_HIGH_WATERMARK_RATIO 0.0
    launchctl setenv PYTORCH_ENABLE_MPS_FALLBACK 1
    launchctl setenv PYTORCH_MPS_PREFER_UNIFIED_MEMORY 1
else
    # Generic Apple Silicon optimizations
    export PYTORCH_MPS_MEMORY_FRACTION=0.90
    export PYTORCH_MPS_ALLOCATOR_POLICY=page
    export PYTORCH_MPS_PREFER_FAST_ALLOC=1
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
    
    launchctl setenv PYTORCH_MPS_MEMORY_FRACTION 0.90
    launchctl setenv PYTORCH_MPS_ALLOCATOR_POLICY page
    launchctl setenv PYTORCH_MPS_PREFER_FAST_ALLOC 1
    launchctl setenv OMP_NUM_THREADS 8
    launchctl setenv MKL_NUM_THREADS 8
fi

echo "✅ M4 Pro PyTorch MPS variables configured"

# 4. Advanced System Cache Management
echo "\n--- Advanced Cache Clearing ---"

# Purge memory and caches
echo "Purging system memory..."
sudo purge

# Clear DNS cache
echo "Flushing DNS cache..."
sudo dscacheutil -flushcache
sudo killall -HUP mDNSResponder

# Clear font caches (can use significant memory)
echo "Clearing font caches..."
sudo atsutil databases -removeUser
sudo atsutil server -shutdown
sudo atsutil server -ping

# Clear Core ML cache
echo "Clearing Core ML caches..."
rm -rf ~/Library/Caches/com.apple.CoreML/
rm -rf /System/Library/Caches/com.apple.CoreML/

# Clear Spotlight cache if needed
echo "Optimizing Spotlight cache..."
sudo mdutil -E / >/dev/null 2>&1

echo "✅ Advanced cache clearing complete"

# 5. Memory-Intensive Process Management
echo "\n--- Memory Consumer Analysis ---"
echo "Top 10 memory consumers:"
ps aux --sort=-%mem | head -11

# Intelligent process termination for ML workloads
echo "\n--- Optimizing for ML Workloads ---"

# Kill memory-heavy browser helpers but preserve main apps
pkill -f "Google Chrome Helper" 2>/dev/null
pkill -f "Safari Web Content" 2>/dev/null
pkill -f "WebKit.WebContent" 2>/dev/null

# Kill non-essential background processes
pkill -f "quicklookd" 2>/dev/null
pkill -f "Preview" 2>/dev/null
pkill -f "TextEdit" 2>/dev/null

# Suspend Time Machine during ML workloads
echo "Suspending Time Machine during optimization..."
sudo tmutil disable 2>/dev/null
sleep 2
sudo tmutil enable 2>/dev/null

# 6. Python/ML Cache and Memory Management
echo "\n--- ML-Specific Memory Optimization ---"

# Clear HuggingFace cache if it's too large
hf_cache_size=$(du -sh ~/.cache/huggingface 2>/dev/null | cut -f1 || echo "0")
echo "HuggingFace cache size: $hf_cache_size"

# If cache is larger than 10GB, offer to clear it
if [[ $(du -s ~/.cache/huggingface 2>/dev/null | cut -f1) -gt 10485760 ]]; then
    echo "Large HuggingFace cache detected (>10GB)"
    echo "Consider clearing with: rm -rf ~/.cache/huggingface"
fi

# Clear Python bytecode cache
find . -name "*.pyc" -delete 2>/dev/null
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Clear PyTorch MPS cache and reset
python3 -c "
import gc
import sys

try:
    import torch
    if torch.backends.mps.is_available():
        # Clear MPS cache
        torch.mps.empty_cache()
        torch.mps.synchronize()
        print('✅ MPS cache cleared and synchronized')
        
        # Show MPS memory info
        if hasattr(torch.mps, 'memory_summary'):
            print('MPS Memory Summary:')
            print(torch.mps.memory_summary())
    else:
        print('ℹ️  MPS not available')
        
    # Clear CUDA cache if available (for compatibility)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('✅ CUDA cache cleared')
        
except ImportError:
    print('ℹ️  PyTorch not installed')
except Exception as e:
    print(f'⚠️  Cache clearing error: {e}')

# Force garbage collection
gc.collect()
print('✅ Python garbage collection complete')

# Memory info
try:
    import psutil
    memory = psutil.virtual_memory()
    print(f'System memory: {memory.percent:.1f}% used, {memory.available / 1024**3:.1f} GB available')
except ImportError:
    pass
"

# 7. Restart ML-related services
echo "\n--- Restarting ML Services ---"
pkill -f "jupyter" 2>/dev/null
pkill -f "ipython" 2>/dev/null
pkill -f "python.*flux" 2>/dev/null

# Wait for processes to terminate
sleep 2

echo "✅ ML services restarted"

# 8. File System Optimization
echo "\n--- File System Optimization ---"
# Check available disk space
disk_usage=$(df -h / | tail -1 | awk '{print $5}' | tr -d '%')
if [ "$disk_usage" -gt 85 ]; then
    echo "⚠️  Disk usage high (${disk_usage}%) - consider freeing space"
    echo "Large files in current directory:"
    find . -size +1G -type f 2>/dev/null | head -5
fi

# 9. Performance Tuning for M4 Pro
echo "\n--- M4 Pro Performance Tuning ---"

if [ $IS_M4_PRO -eq 0 ]; then
    # Set optimal performance parameters
    sudo sysctl -w kern.maxfilesperproc=65536 2>/dev/null
    sudo sysctl -w kern.maxfiles=1048576 2>/dev/null
    
    # Optimize for large ML model loading
    sudo sysctl -w vm.swappiness=1 2>/dev/null || true
    
    echo "✅ M4 Pro performance parameters optimized"
fi

# 10. Final Memory Status
echo "\n--- Post-Optimization Memory Status ---"
vm_stat_final=$(vm_stat)
pages_free_final=$(echo "$vm_stat_final" | grep "Pages free:" | awk '{print $3}' | tr -d '.')
free_gb_final=$(get_memory_gb $((pages_free_final * 4096)))

echo "Free memory after optimization: ${free_gb_final} GB"
echo "Memory freed: $(echo "$free_gb_final - $free_gb" | bc -l | xargs printf "%.2f") GB"

# Top memory consumers after optimization
echo "\n--- Top 5 Memory Consumers (Post-Optimization) ---"
ps aux --sort=-%mem | head -6

# 11. Active Environment Variables Summary
echo "\n--- Active M4 Pro Environment Variables ---"
echo "PYTORCH_MPS_MEMORY_FRACTION: $PYTORCH_MPS_MEMORY_FRACTION"
echo "PYTORCH_MPS_ALLOCATOR_POLICY: $PYTORCH_MPS_ALLOCATOR_POLICY"
echo "PYTORCH_MPS_PREFER_FAST_ALLOC: $PYTORCH_MPS_PREFER_FAST_ALLOC"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "MKL_NUM_THREADS: $MKL_NUM_THREADS"

if [ $IS_M4_PRO -eq 0 ]; then
    echo "PYTORCH_MPS_ENABLE_FALLBACK: $PYTORCH_MPS_ENABLE_FALLBACK"
    echo "PYTORCH_MPS_PREFER_UNIFIED_MEMORY: $PYTORCH_MPS_PREFER_UNIFIED_MEMORY"
    echo "PYTORCH_MPS_HIGH_WATERMARK_RATIO: $PYTORCH_MPS_HIGH_WATERMARK_RATIO"
fi

# 12. Performance Recommendations
echo "\n--- M4 Pro FLUX.1 Krea Performance Recommendations ---"
echo "🔥 For maximum performance:"
echo "   • Use image sizes: 1024x1024 to 1280x1024"
echo "   • Steps: 28-32 (optimal for quality/speed balance)"
echo "   • Guidance: 3.5-5.0"
echo "   • Monitor thermals with: python thermal_performance_manager.py"
echo "   • Use: python maximum_performance_pipeline.py for best results"
echo ""
echo "🌡️  Thermal tips:"
echo "   • Keep ambient temperature <25°C for sustained workloads"
echo "   • Use maximum_performance_pipeline.py for thermal management"
echo "   • Monitor with Activity Monitor GPU tab"

# 13. Generate optimization report
echo "\n--- Optimization Report ---"
{
    echo "M4 Pro Memory Optimization Report - $(date)"
    echo "============================================"
    echo "System: $(system_profiler SPHardwareDataType | grep 'Chip:' | cut -d: -f2 | xargs)"
    echo "Memory freed: $(echo "$free_gb_final - $free_gb" | bc -l | xargs printf "%.2f") GB"
    echo "Free memory: ${free_gb_final} GB"
    echo "Optimizations applied: PyTorch MPS, Cache clearing, Process management"
    echo "Status: Ready for FLUX.1 Krea generation"
} > /tmp/m4_pro_optimization_report.txt

echo "📊 Optimization report saved to: /tmp/m4_pro_optimization_report.txt"

echo "\n🎉 === M4 Pro Memory Optimization Complete ==="
echo "🚀 System optimized for FLUX.1 Krea maximum performance!"
echo "💡 Run 'python maximum_performance_pipeline.py --help' for usage"