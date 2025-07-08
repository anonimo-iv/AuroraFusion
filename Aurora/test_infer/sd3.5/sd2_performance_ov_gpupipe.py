#!/usr/bin/env python3
"""
OpenVINO Heterogeneous Pipeline Parallelism Test
Testing compiled models with pipeline parallelism across multiple GPUs
"""

import time
import numpy as np
from pathlib import Path
from optimum.intel import OVStableDiffusionPipeline
import openvino as ov
import gc
import os
import torch
from typing import Dict, List, Tuple

# Configuration
MODEL_ID = "stabilityai/stable-diffusion-2-1"
MODEL_DIR = "ov_model_hetero"
PROMPT = "A majestic lion wearing a crown, digital art, highly detailed"
NUM_INFERENCE_STEPS = 20
NUM_WARMUP_RUNS = 2
NUM_BENCHMARK_RUNS = 5
IMAGE_SIZE = 512

class HeterogeneousPipeline:
    """Custom pipeline wrapper for heterogeneous execution"""
    
    def __init__(self, model_dir: str, device_mapping: Dict[str, str]):
        self.model_dir = Path(model_dir)
        self.device_mapping = device_mapping
        self.core = ov.Core()
        self.compiled_models = {}
        
    def load_and_compile_components(self):
        """Load and compile individual components to different devices"""
        print("\nLoading and compiling components:")
        
        # Component paths
        components = {
            "text_encoder": "text_encoder/openvino_model.xml",
            "unet": "unet/openvino_model.xml", 
            "vae_decoder": "vae_decoder/openvino_model.xml",
            "vae_encoder": "vae_encoder/openvino_model.xml"
        }
        
        compile_times = {}
        
        for component_name, model_path in components.items():
            full_path = self.model_dir / model_path
            if not full_path.exists():
                print(f"  ⚠️  {component_name} not found at {full_path}")
                continue
                
            device = self.device_mapping.get(component_name, "GPU")
            
            print(f"  Loading {component_name}...")
            start = time.time()
            
            # Read model
            model = self.core.read_model(str(full_path))
            
            # Configure for specific device
            config = {
                "PERFORMANCE_HINT": "LATENCY",
                "GPU_ENABLE_SDPA_OPTIMIZATION": "YES"
            }
            
            # Compile to specific device
            compiled_model = self.core.compile_model(model, device, config)
            self.compiled_models[component_name] = compiled_model
            
            compile_time = time.time() - start
            compile_times[component_name] = compile_time
            
            print(f"    ✓ Compiled to {device} in {compile_time:.2f}s")
        
        return compile_times

def free_gpu_memory():
    """Free GPU memory aggressively"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()

def detect_gpus():
    """Detect available GPU devices"""
    core = ov.Core()
    devices = core.available_devices
    print(f"All available devices: {devices}")
    
    gpu_devices = [d for d in devices if d.startswith("GPU")]
    
    if not gpu_devices:
        raise RuntimeError("No GPU devices found!")
    
    print(f"\nDetected {len(gpu_devices)} GPU device(s):")
    for gpu in gpu_devices:
        print(f"  - {gpu}")
        # Try to get device properties
        try:
            device_name = core.get_property(gpu, "FULL_DEVICE_NAME")
            print(f"    Name: {device_name}")
            # Try to get memory info if available
            try:
                memory_info = core.get_property(gpu, "GPU_DEVICE_TOTAL_MEM_SIZE")
                print(f"    Memory: {memory_info / (1024**3):.1f} GB")
            except:
                pass
        except:
            pass
    
    return gpu_devices

def setup_and_convert():
    """Setup and convert model if needed"""
    # Convert model if needed
    if not Path(MODEL_DIR).exists():
        print(f"\nConverting {MODEL_ID} to OpenVINO format...")
        pipeline = OVStableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            export=True,
            compile=False
        )
        pipeline.save_pretrained(MODEL_DIR)
        del pipeline
        gc.collect()
        print(f"✓ Model converted and saved to {MODEL_DIR}")

def clear_gpu_cache(cache_dir: str = None):
    """Clear GPU cache to avoid serialization issues"""
    import shutil
    cache_paths = []
    
    if cache_dir:
        cache_paths.append(Path(cache_dir))
    else:
        # Clear all possible cache directories
        cache_paths = [
            Path("cache_GPU.0"),
            Path("cache_GPU.1"),
            Path("cache_hetero"),
            Path("model_cache"),
            Path("cache_batch_GPU.0"),
            Path("cache_batch_GPU.1")
        ]
    
    for cache_path in cache_paths:
        if cache_path.exists():
            try:
                shutil.rmtree(cache_path)
                print(f"  Cleared cache: {cache_path}")
            except Exception as e:
                print(f"  Warning: Could not clear {cache_path}: {e}")

def benchmark_single_gpu(gpu_device: str):
    """Benchmark with all components on a single GPU"""
    print(f"\n{'='*60}")
    print(f"TEST: Single GPU Pipeline ({gpu_device})")
    print(f"{'='*60}")
    
    # Clear cache to avoid serialization issues
    cache_dir = f"cache_{gpu_device.replace('.', '_').replace(':', '_')}"
    print("Clearing cache to avoid serialization issues...")
    clear_gpu_cache(cache_dir)
    
    # Try different configurations if the default fails
    configs_to_try = [
        {
            "name": "With cache and optimizations",
            "ov_config": {
                "PERFORMANCE_HINT": "LATENCY",
                "CACHE_DIR": cache_dir,
                "GPU_ENABLE_SDPA_OPTIMIZATION": "YES"
            }
        },
        {
            "name": "Without cache",
            "ov_config": {
                "PERFORMANCE_HINT": "LATENCY",
                "GPU_ENABLE_SDPA_OPTIMIZATION": "YES"
            }
        },
        {
            "name": "Minimal config",
            "ov_config": {
                "PERFORMANCE_HINT": "LATENCY"
            }
        }
    ]
    
    pipeline = None
    load_time = 0
    
    for config in configs_to_try:
        try:
            print(f"\nTrying configuration: {config['name']}...")
            print(f"Loading pipeline with device={gpu_device}...")
            start = time.time()
            
            pipeline = OVStableDiffusionPipeline.from_pretrained(
                MODEL_DIR,
                device=gpu_device,
                compile=True,
                ov_config=config['ov_config']
            )
            
            load_time = time.time() - start
            print(f"✓ Success! Total load + compile time: {load_time:.2f}s")
            break
            
        except Exception as e:
            print(f"✗ Failed with error: {str(e)[:200]}...")
            if pipeline:
                del pipeline
                pipeline = None
            gc.collect()
            continue
    
    if pipeline is None:
        print("\n✗ All configurations failed. Skipping this test.")
        return [], 0
    
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
        image.save(f"single_gpu_{gpu_device.replace(':', '_')}_output.png")
    
    del pipeline
    gc.collect()
    
    return times, load_time

def benchmark_multi_gpu_hetero(gpu_devices: List[str]):
    """Benchmark with HETERO plugin for multi-GPU execution"""
    print(f"\n{'='*60}")
    print("TEST: Multi-GPU with HETERO Plugin")
    print(f"{'='*60}")
    
    # Create HETERO device string
    hetero_device = "HETERO:" + ",".join(gpu_devices)
    print(f"Using heterogeneous device: {hetero_device}")
    
    # Load and compile
    print("Loading pipeline with HETERO device...")
    start = time.time()
    
    try:
        pipeline = OVStableDiffusionPipeline.from_pretrained(
            MODEL_DIR,
            device=hetero_device,
            compile=True,
            ov_config={
                "PERFORMANCE_HINT": "THROUGHPUT",
                "CACHE_DIR": "cache_hetero"
            }
        )
        
        load_time = time.time() - start
        print(f"Total load + compile time: {load_time:.2f}s")
        
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
            image.save("multi_gpu_hetero_output.png")
        
        del pipeline
        gc.collect()
        
        return times, load_time
        
    except Exception as e:
        print(f"Error with HETERO device: {e}")
        return [], 0

def benchmark_pipeline_parallel(gpu_devices: List[str]):
    """Benchmark with manual pipeline parallelism across GPUs"""
    print(f"\n{'='*60}")
    print("TEST: Pipeline Parallelism (Component Distribution)")
    print(f"{'='*60}")
    
    if len(gpu_devices) == 1:
        print("Only one GPU available, using different configurations on same GPU")
        device_mapping = {
            "text_encoder": gpu_devices[0],
            "unet": gpu_devices[0],
            "vae_decoder": gpu_devices[0],
            "vae_encoder": gpu_devices[0]
        }
    else:
        # Distribute components across GPUs
        print("Distributing components across GPUs:")
        device_mapping = {
            "text_encoder": gpu_devices[0],
            "unet": gpu_devices[1] if len(gpu_devices) > 1 else gpu_devices[0],
            "vae_decoder": gpu_devices[2] if len(gpu_devices) > 2 else gpu_devices[0],
            "vae_encoder": gpu_devices[3] if len(gpu_devices) > 3 else gpu_devices[1] if len(gpu_devices) > 1 else gpu_devices[0]
        }
    
    for component, device in device_mapping.items():
        print(f"  {component} -> {device}")
    
    # For this demo, we'll use the standard pipeline but with device affinity hints
    # In a real implementation, you'd create a custom pipeline that routes components
    print("\nNote: Full pipeline parallelism requires custom pipeline implementation")
    print("Using standard pipeline with device hints for demonstration")
    
    # Create heterogeneous pipeline
    hetero_pipeline = HeterogeneousPipeline(MODEL_DIR, device_mapping)
    compile_times = hetero_pipeline.load_and_compile_components()
    
    print(f"\nComponent compilation times:")
    for component, time_taken in compile_times.items():
        print(f"  {component}: {time_taken:.2f}s")
    
    return compile_times

def benchmark_batch_parallel(gpu_devices: List[str]):
    """Benchmark batch parallelism across GPUs"""
    print(f"\n{'='*60}")
    print("TEST: Batch Parallelism Across GPUs")
    print(f"{'='*60}")
    
    if len(gpu_devices) < 2:
        print("Batch parallelism requires at least 2 GPUs")
        return [], 0
    
    print(f"Creating {len(gpu_devices)} pipeline instances...")
    pipelines = []
    
    # Create one pipeline per GPU
    for i, gpu in enumerate(gpu_devices):
        print(f"\nLoading pipeline {i+1} on {gpu}...")
        start = time.time()
        
        pipeline = OVStableDiffusionPipeline.from_pretrained(
            MODEL_DIR,
            device=gpu,
            compile=True,
            ov_config={
                "PERFORMANCE_HINT": "THROUGHPUT",
                "CACHE_DIR": f"cache_batch_{gpu}"
            }
        )
        
        load_time = time.time() - start
        print(f"  Load time: {load_time:.2f}s")
        pipelines.append(pipeline)
    
    # Test parallel batch processing
    batch_prompts = [
        "A majestic lion wearing a crown, digital art",
        "A cyberpunk city at night, neon lights",
        "A serene mountain landscape at sunset",
        "An underwater coral reef, vibrant colors"
    ][:len(pipelines)]
    
    print(f"\nProcessing {len(batch_prompts)} prompts in parallel...")
    
    # Simulate parallel processing (in real implementation, use threading/multiprocessing)
    start = time.time()
    images = []
    
    for i, (pipeline, prompt) in enumerate(zip(pipelines, batch_prompts)):
        print(f"  GPU {i}: Processing '{prompt[:30]}...'")
        image = pipeline(prompt, num_inference_steps=NUM_INFERENCE_STEPS).images[0]
        images.append(image)
        image.save(f"batch_parallel_gpu{i}_output.png")
    
    total_time = time.time() - start
    print(f"\nTotal batch processing time: {total_time:.2f}s")
    print(f"Average per image: {total_time/len(images):.2f}s")
    
    # Cleanup
    for pipeline in pipelines:
        del pipeline
    gc.collect()
    
    return [total_time], 0

def main():
    """Main function"""
    print("OpenVINO Heterogeneous Pipeline Parallelism Test")
    print("Testing compiled models with multi-GPU configurations")
    print("="*60)
    
    # Environment variable options for troubleshooting
    print("\nChecking environment settings...")
    if os.environ.get("OV_GPU_CACHE_MODEL", "") != "0":
        print("  Note: GPU model caching is enabled. Set OV_GPU_CACHE_MODEL=0 to disable if experiencing issues.")
    
    try:
        # Clear all caches before starting
        print("\nClearing all cache directories...")
        clear_gpu_cache()
        
        # Setup
        setup_and_convert()
        gpu_devices = detect_gpus()
        
        # Free memory before starting tests
        print("\nFreeing GPU memory before tests...")
        free_gpu_memory()
        
        results = {}
        
        # Test 1: Single GPU baseline
        for i, gpu in enumerate(gpu_devices):
            print(f"\nTesting GPU {i+1}/{len(gpu_devices)}...")
            
            # Free memory before each test
            free_gpu_memory()
            
            try:
                times, load_time = benchmark_single_gpu(gpu)
                if times:
                    results[f"single_{gpu}"] = {
                        "times": times,
                        "load_time": load_time,
                        "avg": np.mean(times),
                        "std": np.std(times)
                    }
            
            except Exception as e:
                print(f"  Error testing {gpu}: {e}")
                continue
            break
        
        # Test 2: HETERO plugin (if multiple GPUs)
        if len(gpu_devices) > 1:
            print("\nTesting HETERO configuration...")
            free_gpu_memory()
            
            try:
                times, load_time = benchmark_multi_gpu_hetero(gpu_devices)
                if times:
                    results["hetero"] = {
                        "times": times,
                        "load_time": load_time,
                        "avg": np.mean(times),
                        "std": np.std(times)
                    }
            except Exception as e:
                print(f"  Error with HETERO: {e}")
        
        # Test 3: Pipeline parallelism
        print("\nTesting pipeline parallelism concept...")
        free_gpu_memory()
        
        try:
            compile_times = benchmark_pipeline_parallel(gpu_devices)
        except Exception as e:
            print(f"  Error with pipeline parallelism: {e}")
        
        # Test 4: Batch parallelism (if multiple GPUs)
        if len(gpu_devices) > 1:
            print("\nTesting batch parallelism...")
            free_gpu_memory()
            
            try:
                times, _ = benchmark_batch_parallel(gpu_devices)
                if times:
                    results["batch_parallel"] = {
                        "times": times,
                        "avg": times[0] if times else 0
                    }
            except Exception as e:
                print(f"  Error with batch parallelism: {e}")
        
        # Results summary
        if results:
            print("\n" + "="*60)
            print("PERFORMANCE SUMMARY")
            print("="*60)
            
            for config_name, result in results.items():
                print(f"\n{config_name}:")
                if "load_time" in result:
                    print(f"  Load time: {result['load_time']:.2f}s")
                if "avg" in result:
                    print(f"  Average inference: {result['avg']:.2f}s")
                    if "std" in result and result["std"] > 0:
                        print(f"  Std deviation: {result['std']:.2f}s")
                    print(f"  Throughput: {NUM_INFERENCE_STEPS/result['avg']:.2f} steps/s")
        else:
            print("\n⚠️  No successful test runs completed.")
        
        # Analysis
        print("\n" + "="*60)
        print("TROUBLESHOOTING TIPS:")
        print("="*60)
        print("\nIf experiencing GPU memory or serialization errors:")
        print("1. Set environment variable: export OV_GPU_CACHE_MODEL=0")
        print("2. Reduce batch size or inference steps")
        print("3. Clear system GPU memory: sudo nvidia-smi --gpu-reset")
        print("4. Update GPU drivers and OpenVINO to latest versions")
        print("5. Try running with single component at a time")
        print("6. Check available GPU memory with nvidia-smi")
        
        print("\nFor Intel GPUs:")
        print("1. Ensure Intel GPU drivers are up to date")
        print("2. Check clinfo for OpenCL support")
        print("3. Set export OCL_ICD_VENDORS=/etc/OpenCL/vendors/")
        
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "="*60)
        print("DEBUGGING INFORMATION:")
        print("="*60)
        print(f"OpenVINO version: {ov.__version__}")
        print(f"Python version: {os.sys.version}")
        print(f"Working directory: {os.getcwd()}")
        print(f"Model directory exists: {Path(MODEL_DIR).exists()}")
        
        # Check OpenVINO installation
        try:
            core = ov.Core()
            print(f"OpenVINO Core initialized: ✓")
            print(f"Available devices: {core.available_devices}")
        except Exception as core_error:
            print(f"OpenVINO Core initialization failed: {core_error}")

if __name__ == "__main__":
    main()