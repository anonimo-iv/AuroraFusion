#!/usr/bin/env python3
"""
Intel PyTorch Extension (IPEX) GPU vs CPU Comparison
"""

import time
import numpy as np
from diffusers import StableDiffusionPipeline
import torch
import intel_extension_for_pytorch as ipex
import gc

# Configuration
MODEL_ID = "stabilityai/stable-diffusion-2-1"
PROMPT = "A majestic lion wearing a crown, digital art, highly detailed"
NUM_INFERENCE_STEPS = 20
NUM_WARMUP_RUNS = 2
NUM_BENCHMARK_RUNS = 5
IMAGE_SIZE = 512

def check_devices():
    """Check available compute devices"""
    print("Available compute devices:")
    print(f"- CPU: {torch.cpu.is_available()}")
    print(f"- CUDA: {torch.cuda.is_available()}")
    print(f"- XPU: {torch.xpu.is_available()}")

    if torch.xpu.is_available():
        return "xpu"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def benchmark_cpu():
    """Test 1: CPU Baseline"""
    print("\n" + "="*60)
    print("TEST 1: CPU Baseline")
    print("="*60)
    
    # Load model
    start_load = time.time()
    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_ID)
    pipeline.to("cpu")
    pipeline.safety_checker = None  # Disable for benchmarking
    load_time = time.time() - start_load
    print(f"Model load time: {load_time:.2f}s")
    
    # Warmup
    print("\nWarming up...")
    for i in range(NUM_WARMUP_RUNS):
        _ = pipeline(PROMPT, num_inference_steps=NUM_INFERENCE_STEPS).images[0]
        print(f"  Warmup {i+1}/{NUM_WARMUP_RUNS} complete")
    
    # Benchmark
    print("\nBenchmarking...")
    times = []
    for i in range(NUM_BENCHMARK_RUNS):
        start = time.time()
        image = pipeline(PROMPT, num_inference_steps=NUM_INFERENCE_STEPS).images[0]
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")
    
    if times:
        image.save("cpu_output.png")
    
    del pipeline
    gc.collect()
    return times

def benchmark_gpu_baseline(device):
    """Test 2: GPU Baseline without IPEX"""
    print("\n" + "="*60)
    print("TEST 2: GPU Baseline (without IPEX)")
    print("="*60)
    
    # Load model
    start_load = time.time()
    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_ID)
    pipeline.to(device)
    pipeline.safety_checker = None
    load_time = time.time() - start_load
    print(f"Model load time: {load_time:.2f}s")
    
    # Warmup
    print("\nWarming up...")
    for i in range(NUM_WARMUP_RUNS):
        _ = pipeline(PROMPT, num_inference_steps=NUM_INFERENCE_STEPS).images[0]
        print(f"  Warmup {i+1}/{NUM_WARMUP_RUNS} complete")
    
    # Benchmark
    print("\nBenchmarking...")
    times = []
    for i in range(NUM_BENCHMARK_RUNS):
        start = time.time()
        image = pipeline(PROMPT, num_inference_steps=NUM_INFERENCE_STEPS).images[0]
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")
    
    if times:
        image.save("gpu_baseline_output.png")
    
    del pipeline
    gc.collect()
    return times

def benchmark_gpu_ipex(device):
    """Test 3: GPU with IPEX Optimizations"""
    print("\n" + "="*60)
    print("TEST 3: GPU with IPEX Optimizations")
    print("="*60)
    
    # Load model
    start_load = time.time()
    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_ID)
    pipeline.to(device)
    pipeline.safety_checker = None
    
    # Apply IPEX optimizations
    pipeline.unet = ipex.optimize(pipeline.unet)
    pipeline.vae = ipex.optimize(pipeline.vae)
    pipeline.text_encoder = ipex.optimize(pipeline.text_encoder)
    
    load_time = time.time() - start_load
    print(f"Model load + optimization time: {load_time:.2f}s")
    
    # Warmup
    print("\nWarming up...")
    for i in range(NUM_WARMUP_RUNS):
        _ = pipeline(PROMPT, num_inference_steps=NUM_INFERENCE_STEPS).images[0]
        print(f"  Warmup {i+1}/{NUM_WARMUP_RUNS} complete")
    
    # Benchmark
    print("\nBenchmarking...")
    times = []
    for i in range(NUM_BENCHMARK_RUNS):
        # torch.xpu.synchronize()
        start = time.time()
        image = pipeline(PROMPT, num_inference_steps=NUM_INFERENCE_STEPS).images[0]
        # torch.xpu.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")
    
    if times:
        image.save("gpu_ipex_output.png")
    
    del pipeline
    gc.collect()
    return times

def main():
    """Main function"""
    print("IPEX Performance Analysis")
    print("="*60)
    
    try:
        # Device setup
        device = check_devices()
        print(f"\nUsing device: {device.upper()}")
        
        # Clear caches
        torch.xpu.empty_cache() if device == "xpu" else torch.cuda.empty_cache()
        
        # Run benchmarks
        cpu_times = benchmark_cpu()
        gpu_baseline_times = benchmark_gpu_baseline(device)
        gpu_ipex_times = benchmark_gpu_ipex(device)
        
        # Results analysis
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        # CPU Results
        if cpu_times:
            avg_cpu = np.mean(cpu_times)
            std_cpu = np.std(cpu_times)
            print(f"\n1. CPU Baseline:")
            print(f"   - Average inference: {avg_cpu:.2f}s ± {std_cpu:.2f}s")
            print(f"   - Throughput: {NUM_INFERENCE_STEPS/avg_cpu:.2f} steps/s")
        
        # GPU Baseline Results
        if gpu_baseline_times:
            avg_gpu_base = np.mean(gpu_baseline_times)
            std_gpu_base = np.std(gpu_baseline_times)
            print(f"\n2. GPU Baseline:")
            print(f"   - Average inference: {avg_gpu_base:.2f}s ± {std_gpu_base:.2f}s")
            print(f"   - Throughput: {NUM_INFERENCE_STEPS/avg_gpu_base:.2f} steps/s")
        
        # GPU IPEX Results
        if gpu_ipex_times:
            avg_gpu_ipex = np.mean(gpu_ipex_times)
            std_gpu_ipex = np.std(gpu_ipex_times)
            print(f"\n3. GPU with IPEX:")
            print(f"   - Average inference: {avg_gpu_ipex:.2f}s ± {std_gpu_ipex:.2f}s")
            print(f"   - Throughput: {NUM_INFERENCE_STEPS/avg_gpu_ipex:.2f} steps/s")
        
        # Comparison
        if cpu_times and gpu_ipex_times:
            speedup = avg_cpu / avg_gpu_ipex
            print(f"\nGPU IPEX Speedup vs CPU: {speedup:.2f}x")
        
        if gpu_baseline_times and gpu_ipex_times:
            ipex_gain = avg_gpu_base / avg_gpu_ipex
            print(f"IPEX Gain vs GPU Baseline: {ipex_gain:.2f}x")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()