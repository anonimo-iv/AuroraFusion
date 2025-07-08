#!/usr/bin/env python3
"""
Multi-GPU Stable Diffusion 3.5 Inference Script for Aurora Supercomputer
Optimized for Intel Data Center GPU Max 1550 with OpenVINO GenAI

Features:
- Automatic GPU/tile discovery for Aurora's 6x Max 1550 GPUs (12 visible devices)  
- Dynamic load balancing across all available compute units
- Batch processing with intelligent scheduling
- Performance monitoring and statistics
- Support for T5 text encoder and Flash LoRA (when available)
- Configurable model paths and inference parameters

Note on T5 Support:
- T5TextEncoder may not be available in all openvino_genai versions
- The script automatically detects and disables T5 if not available
- Use --no-t5 flag to explicitly disable T5 encoder

Note on Aurora GPU Configuration:
- Each Intel Max 1550 GPU presents its tiles as separate GPU devices
- Typical Aurora node: 6 GPUs x 2 tiles = 12 GPU devices (GPU.0 - GPU.11)
- No hierarchical GPU.X.Y naming - each tile is a direct GPU.X device

Usage:
  python openvino_sd3.5.py              # Run full inference demo
  python openvino_sd3.5.py --diagnose   # Show GPU device diagnostics
  python openvino_sd3.5.py --test       # Quick test of GPU discovery only
  python openvino_sd3.5.py --test-single # Test single GPU pipeline creation
  python openvino_sd3.5.py --no-t5      # Run without T5 text encoder
"""

import os
import sys
import time
import json
import asyncio
import logging
import threading
import queue
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from PIL import Image
import itertools

# Aurora-specific imports
import openvino as ov
import openvino_genai as ov_genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AuroraConfig:
    """Configuration for Aurora supercomputer setup"""
    node_gpus: int = 6  # Intel Max 1550 GPUs per Aurora node
    tiles_per_gpu: int = 2  # Tiles per Intel Max 1550 (each presented as separate GPU device)
    total_tile_devices: int = 12  # Total GPU devices visible (6 GPUs * 2 tiles)
    memory_per_gpu: int = 128  # GB per Intel Max 1550
    target_batch_size: int = 2  # Optimal batch size per tile
    max_nodes: int = 10624  # Total Aurora compute nodes
    
@dataclass
class ModelConfig:
    """SD3.5 model configuration"""
    base_path: Path = Path("/lus/flare/projects/hp-ptycho/binkma/models/sd3.5")
    use_t5: bool = True
    use_flash_lora: bool = False
    weight_compression: bool = True
    model_variant: str = "tensorart/stable-diffusion-3.5-medium-turbo"
    
    def __post_init__(self):
        """Check if T5 is actually available and adjust configuration"""
        if self.use_t5:
            if hasattr(ov_genai, 'T5TextEncoder'):
                logger.info("T5TextEncoder is available in openvino_genai")
            else:
                logger.warning("T5TextEncoder not available in openvino_genai, disabling T5 support")
                self.use_t5 = False
    
@dataclass
class InferenceRequest:
    """Individual inference request"""
    request_id: str
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 28
    guidance_scale: float = 5.0
    seed: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher priority processed first

class TileDevice:
    """Represents an individual GPU tile on Aurora"""
    def __init__(self, device_id: str, tile_index: int, gpu_index: int):
        self.device_id = device_id
        self.tile_index = tile_index
        self.gpu_index = gpu_index
        self.pipeline = None
        self.is_busy = False
        self.total_inferences = 0
        self.total_time = 0.0
        self.lock = threading.Lock()
        self.last_used = time.time()
        
    def get_utilization(self) -> float:
        """Get average inference time for this tile"""
        with self.lock:
            if self.total_inferences == 0:
                return 0.0
            return self.total_time / self.total_inferences
            
    def get_throughput(self) -> float:
        """Get images per second for this tile"""
        with self.lock:
            if self.total_time == 0:
                return 0.0
            return self.total_inferences / self.total_time

class SD35_Aurora_MultiGPU:
    """
    Multi-GPU Stable Diffusion 3.5 pipeline optimized for Aurora supercomputer
    using OpenVINO GenAI
    """
    
    def __init__(self, 
                 model_config: ModelConfig,
                 aurora_config: AuroraConfig,
                 max_tiles: Optional[int] = None,
                 enable_profiling: bool = False,
                 test_mode: bool = False):
        """
        Initialize the multi-GPU SD3.5 pipeline
        
        Args:
            model_config: Model configuration
            aurora_config: Aurora-specific configuration
            max_tiles: Maximum number of tiles to use (None = use all available)
            enable_profiling: Enable detailed performance profiling
            test_mode: If True, only test GPU discovery without loading models
        """
        self.model_config = model_config
        self.aurora_config = aurora_config
        self.enable_profiling = enable_profiling
        self.test_mode = test_mode
        
        # Initialize OpenVINO
        self.core = ov.Core()
        self.tiles = []
        
        # Request queues with priority support
        self.request_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()
        
        # Worker management
        self.worker_threads = []
        self.stop_workers = False
        
        # Performance tracking
        self.start_time = time.time()
        self.total_requests = 0
        self.failed_requests = 0
        self.counter = itertools.count()
        
        # Initialize system
        self._setup_environment()
        
        # Verify OpenVINO GenAI components
        components = verify_openvino_genai()
        if components is None:
            raise RuntimeError("OpenVINO GenAI not properly installed")
            
        self._discover_tiles(max_tiles)
        self._setup_model_cache()
        
        if not self.test_mode:
            self._start_workers()
            logger.info(f"Initialized SD3.5 pipeline with {len(self.tiles)} GPU tiles")
        else:
            logger.info(f"Test mode: Discovered {len(self.tiles)} GPU tiles, skipping model loading")
        
    def _setup_environment(self):
        """Configure Aurora-specific environment settings"""
        # Set optimal thread counts for Aurora
        os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() // self.aurora_config.node_gpus)
        os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
        
        # Enable GPU-specific optimizations
        os.environ['GPU_MAX_THREADS_PER_BLOCK'] = '1024'
        os.environ['GPU_MAX_WORKGROUP_SIZE'] = '1024'
        
    def _discover_tiles(self, max_tiles: Optional[int]):
        """Discover available Intel GPU devices on Aurora
        
        Note: Aurora presents each tile as an independent GPU device (GPU.0 through GPU.11),
        rather than using a hierarchical GPU.X.Y naming scheme. This simplifies discovery.
        """
        # available_devices = self.core.available_devices
        # available_devices = [d for d in available_devices if d.startswith('GPU')]
        available_devices = [f"GPU.{i}" for i in range(self.aurora_config.total_tile_devices)]

        
        # Filter GPU devices - each GPU.X is already a tile/compute unit
        gpu_devices = []
        for device in available_devices:
            if device.startswith('GPU'):
                gpu_devices.append(device)
        
        # Sort GPU devices properly (handle GPU.0, GPU.1, ..., GPU.10, GPU.11)
        def get_gpu_number(device_name):
            """Extract GPU number from device name for proper sorting"""
            try:
                if '.' in device_name:
                    return int(device_name.split('.')[-1])
                elif ':' in device_name:
                    return int(device_name.split(':')[-1])
                else:
                    # Plain GPU device
                    return 0
            except (ValueError, IndexError):
                return 0
        
        gpu_devices.sort(key=get_gpu_number)
        
        logger.info(f"Found {len(gpu_devices)} GPU devices: {gpu_devices}")
        
        discovered_tiles = []
        
        # Treat each GPU device as an independent compute unit
        for idx, gpu_device in enumerate(gpu_devices):
            try:
                # Test device availability and get properties
                props = self.core.get_property(gpu_device, "FULL_DEVICE_NAME")
                
                # Extract GPU index from device name (e.g., GPU.0 -> 0, GPU:0 -> 0)
                try:
                    if '.' in gpu_device:
                        gpu_number = int(gpu_device.split('.')[-1])
                    elif ':' in gpu_device:
                        gpu_number = int(gpu_device.split(':')[-1])
                    else:
                        gpu_number = idx
                except (ValueError, IndexError):
                    gpu_number = idx
                
                tile = TileDevice(
                    device_id=gpu_device,
                    tile_index=gpu_number,  # Use the GPU number as tile index
                    gpu_index=gpu_number // self.aurora_config.tiles_per_gpu  # Physical GPU index
                )
                discovered_tiles.append(tile)
                
                logger.info(f"Discovered GPU device: {gpu_device} - {props}")
                
            except Exception as e:
                logger.warning(f"Could not access GPU device {gpu_device}: {e}")
                continue
        
        # Limit to requested number of devices
        if max_tiles and max_tiles < len(discovered_tiles):
            discovered_tiles = discovered_tiles[:max_tiles]
            logger.info(f"Limiting to {max_tiles} GPU devices")
            
        self.tiles = discovered_tiles
        
        if not self.tiles:
            raise RuntimeError("No GPU devices discovered!")
            
    def _setup_model_cache(self):
        """Setup model caching for faster loading"""
        cache_dir = Path("./aurora_sd35_cache")
        cache_dir.mkdir(exist_ok=True)
        
        # Configure each GPU device
        for tile in self.tiles:
            try:
                # Create cache subdirectory for this device
                device_cache_dir = cache_dir / f"gpu_{tile.tile_index}"
                device_cache_dir.mkdir(exist_ok=True)
                
                # Try to set cache directory - may not be supported on all devices
                try:
                    self.core.set_property(tile.device_id, {
                        "CACHE_DIR": str(device_cache_dir)
                    })
                except:
                    # Try global GPU cache setting as fallback
                    try:
                        self.core.set_property("GPU", {
                            "CACHE_DIR": str(cache_dir)
                        })
                    except:
                        pass
                
                # Aurora-specific optimizations - set what we can
                optimization_props = {
                    "PERFORMANCE_HINT": "THROUGHPUT",
                    "NUM_STREAMS": "2",
                    "GPU_THROUGHPUT_STREAMS": "2",
                    "INFERENCE_PRECISION_HINT": "f16"
                }
                
                for prop, value in optimization_props.items():
                    try:
                        self.core.set_property(tile.device_id, {prop: value})
                    except:
                        # Some properties might not be supported
                        pass
                
                logger.debug(f"Configured optimizations for {tile.device_id}")
                
            except Exception as e:
                logger.warning(f"Could not fully configure {tile.device_id}: {e}")
                
    def _init_pipeline_on_tile(self, tile: TileDevice) -> bool:
        """Initialize SD3.5 pipeline on specific tile
        
        This method follows the exact approach from the working script,
        using the stable_diffusion_3 factory method with the correct parameters.
        """
        try:
            logger.info(f"Loading pipeline on tile {tile.device_id}")
            
            # Build model paths
            model_paths = {
                "transformer": self.model_config.base_path / "transformer",
                "vae": self.model_config.base_path / "vae_decoder",
                "text_encoder": self.model_config.base_path / "text_encoder",
                "text_encoder_2": self.model_config.base_path / "text_encoder_2",
                "scheduler": self.model_config.base_path / "scheduler",
            }
            
            # Only add text_encoder_3 if T5 is available and requested
            t5_available = hasattr(ov_genai, 'T5TextEncoder')
            if self.model_config.use_t5 and t5_available:
                model_paths["text_encoder_3"] = self.model_config.base_path / "text_encoder_3"
            
            # Verify all paths exist
            for name, path in model_paths.items():
                if not path.exists():
                    logger.warning(f"Model component not found: {path}")
                    if name == "text_encoder_3":
                        # T5 is optional, continue without it
                        del model_paths["text_encoder_3"]
                        logger.info("Continuing without T5 encoder")
                    else:
                        # Other components are required
                        raise FileNotFoundError(f"Required model component not found: {path}")
            
            # Initialize components following the working script's pattern
            scheduler = ov_genai.Scheduler.from_config(
                model_paths["scheduler"] / "scheduler_config.json"
            )
            
            text_encoder = ov_genai.CLIPTextModelWithProjection(
                model_paths["text_encoder"], 
                tile.device_id
            )
            
            text_encoder_2 = ov_genai.CLIPTextModelWithProjection(
                model_paths["text_encoder_2"], 
                tile.device_id
            )
            
            transformer = ov_genai.SD3Transformer2DModel(
                model_paths["transformer"], 
                tile.device_id
            )
            
            vae = ov_genai.AutoencoderKL(
                model_paths["vae"], 
                device=tile.device_id
            )
            
            # Create pipeline based on available components
            if "text_encoder_3" in model_paths and t5_available:
                try:
                    text_encoder_3 = ov_genai.T5TextEncoder(
                        model_paths["text_encoder_3"], 
                        tile.device_id
                    )
                    
                    # Try different method signatures for T5 pipeline
                    try:
                        # Try with positional arguments
                        tile.pipeline = ov_genai.Text2ImagePipeline.stable_diffusion_3(
                            scheduler,
                            text_encoder,
                            text_encoder_2,
                            text_encoder_3,
                            transformer,
                            vae
                        )
                        logger.info(f"Created pipeline with T5 encoder on {tile.device_id}")
                    except TypeError:
                        # Try with keyword arguments
                        tile.pipeline = ov_genai.Text2ImagePipeline.stable_diffusion_3(
                            scheduler=scheduler,
                            text_encoder=text_encoder,
                            text_encoder_2=text_encoder_2,
                            text_encoder_3=text_encoder_3,
                            transformer=transformer,
                            vae_decoder=vae
                        )
                        logger.info(f"Created pipeline with T5 encoder on {tile.device_id} (kwargs)")
                except Exception as e:
                    logger.warning(f"Failed to initialize T5 encoder: {e}")
                    # Fallback to pipeline without T5
                    try:
                        tile.pipeline = ov_genai.Text2ImagePipeline.stable_diffusion_3(
                            scheduler,
                            text_encoder,
                            text_encoder_2,
                            transformer,
                            vae
                        )
                        logger.info(f"Created pipeline without T5 on {tile.device_id}")
                    except TypeError:
                        # Try with keyword arguments
                        tile.pipeline = ov_genai.Text2ImagePipeline.stable_diffusion_3(
                            scheduler=scheduler,
                            text_encoder=text_encoder,
                            text_encoder_2=text_encoder_2,
                            transformer=transformer,
                            vae_decoder=vae
                        )
                        logger.info(f"Created pipeline without T5 on {tile.device_id} (kwargs)")
            else:
                # Create pipeline without T5
                tile.pipeline = ov_genai.Text2ImagePipeline.stable_diffusion_3(
                    scheduler,
                    text_encoder,
                    text_encoder_2,
                    transformer,
                    vae
                )
                logger.info(f"Created pipeline without T5 on {tile.device_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load pipeline on {tile.device_id}: {e}")
            return False
            
    def _start_workers(self):
        """Start worker threads for each tile"""
        successful_tiles = []
        
        # Initialize pipelines on tiles
        for tile in self.tiles:
            if self._init_pipeline_on_tile(tile):
                successful_tiles.append(tile)
            else:
                logger.warning(f"Skipping tile {tile.device_id} due to initialization failure")
        
        self.tiles = successful_tiles
        
        # Start worker threads
        for i, tile in enumerate(self.tiles):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(tile,),
                name=f"TileWorker_{tile.gpu_index}_{tile.tile_index}"
            )
            worker.start()
            self.worker_threads.append(worker)
            
        logger.info(f"Started {len(self.worker_threads)} worker threads")
        
    def _worker_loop(self, tile: TileDevice):
        """Main worker loop for processing inference requests"""
        while not self.stop_workers:
            try:
                # Get next request with priority (lower number = higher priority)
                priority,_, request = self.request_queue.get(timeout=1.0)
                
                if request is None:  # Shutdown signal
                    logger.info(f"Worker {tile.device_id} received shutdown signal")
                    self.request_queue.task_done()  # 
                    # logger.info(f"{tile.device_id}: Received shutdown signal")   
                    break
                    
                # Mark tile as busy
                with tile.lock:
                    tile.is_busy = True
                    tile.last_used = time.time()
                
                # Process request
                start_time = time.time()
                result = self._process_single_request(tile, request)
                end_time = time.time()
                
                inference_time = end_time - start_time
                
                # Update statistics
                with tile.lock:
                    tile.total_inferences += 1
                    tile.total_time += inference_time
                    tile.is_busy = False
                
                # Return result
                self.result_queue.put({
                    'request_id': request.request_id,
                    'image': result['image'],
                    'inference_time': inference_time,
                    'tile_id': tile.device_id,
                    'seed_used': result.get('seed', request.seed),
                    'error': None
                })
                
                self.request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in worker loop for {tile.device_id}: {e}")
                
                # Return error result
                if 'request' in locals():
                    self.result_queue.put({
                        'request_id': request.request_id,
                        'image': None,
                        'inference_time': 0,
                        'tile_id': tile.device_id,
                        'error': str(e)
                    })
                    self.failed_requests += 1
                    self.request_queue.task_done()
                
                with tile.lock:
                    tile.is_busy = False
                    
    def _process_single_request(self, tile: TileDevice, request: InferenceRequest) -> Dict[str, Any]:
        """Process a single inference request on specified tile"""
        try:
            # Set random seed if specified
            if request.seed is not None:
                # OpenVINO GenAI handles seeding internally
                seed = request.seed
            else:
                seed = np.random.randint(0, 2**32 - 1)
            
            # Log if profiling enabled
            if self.enable_profiling:
                logger.debug(f"Processing {request.request_id} on {tile.device_id}: '{request.prompt[:50]}...'")
            
            # Prepare generation parameters
            generation_params = {
                "prompt": request.prompt,
                "num_inference_steps": request.num_inference_steps,
                "height": request.height,
                "width": request.width,
            }
            
            # Handle guidance scale and negative prompt
            if self.model_config.use_flash_lora:
                generation_params["guidance_scale"] = 0
            else:
                generation_params["guidance_scale"] = request.guidance_scale
                # Only include negative_prompt when guidance_scale > 1.0
                if request.guidance_scale > 1.0 and request.negative_prompt:
                    generation_params["negative_prompt"] = request.negative_prompt
            
            # Generate image
            result = tile.pipeline.generate(**generation_params)
            
            # Convert result to PIL Image - handle various output formats
            image = None
            
            # Try different methods to extract the image
            if hasattr(result, 'images') and result.images:
                image = result.images[0]
            elif isinstance(result, list) and len(result) > 0:
                image = result[0]
            elif hasattr(result, 'data'):
                # Handle tensor output
                try:
                    image_array = np.array(result.data).reshape(result.shape)
                    
                    # Remove batch dimension if present
                    if len(image_array.shape) == 4:
                        image_array = image_array[0]
                    
                    # Normalize to 0-255 range
                    if image_array.max() <= 1.0:
                        image_array = (image_array * 255).astype(np.uint8)
                    else:
                        image_array = image_array.astype(np.uint8)
                    
                    # Ensure proper channel ordering (H, W, C)
                    if len(image_array.shape) == 3 and image_array.shape[0] in [3, 4]:
                        image_array = np.transpose(image_array, (1, 2, 0))
                    
                    image = Image.fromarray(image_array)
                except Exception as e:
                    logger.error(f"Failed to convert tensor to image: {e}")
                    raise
            
            # If image is already a PIL Image, use it directly
            if image is not None and not isinstance(image, Image.Image):
                # Try to convert to PIL Image if it's not already
                try:
                    if hasattr(image, '__array__'):
                        # It's array-like
                        image_array = np.array(image)
                        if image_array.max() <= 1.0:
                            image_array = (image_array * 255).astype(np.uint8)
                        image = Image.fromarray(image_array)
                except Exception as e:
                    logger.error(f"Failed to convert to PIL Image: {e}")
                    raise
            
            if image is None:
                raise ValueError("Failed to extract image from pipeline result")
            
            return {
                'image': image,
                'seed': seed
            }
            
        except Exception as e:
            logger.error(f"Error processing request {request.request_id}: {e}")
            raise
            
    def generate_batch(self, 
                      requests: List[InferenceRequest], 
                      timeout: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate images for multiple requests using all available tiles
        
        Args:
            requests: List of inference requests
            timeout: Maximum time to wait for all results (None = no timeout)
            
        Returns:
            Dictionary mapping request_id to result dict
        """
        start_time = time.time()
        logger.info(f"Starting batch generation for {len(requests)} requests across {len(self.tiles)} tiles")
        
        # Add requests to priority queue
        for request in requests:
            # Priority queue uses (priority, item) tuples
            # Lower priority number = higher priority
            self.request_queue.put((request.priority,next(self.counter), request))
        
        self.request_queue.join()
        # Collect results
        results = {}
        completed = 0
        
        while completed < len(requests):
            try:
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    logger.warning(f"Batch generation timeout after {completed}/{len(requests)} completed")
                    break
                
                # Get result with timeout
                wait_time = 1.0 if timeout is None else min(1.0, timeout - (time.time() - start_time))
                result = self.result_queue.get(timeout=wait_time)
                
                results[result['request_id']] = result
                completed += 1
                
                # Log progress
                if completed % 10 == 0 or completed == len(requests):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    logger.info(f"Progress: {completed}/{len(requests)} "
                              f"({rate:.2f} images/sec)")
                
            except queue.Empty:
                if timeout and (time.time() - start_time) > timeout:
                    break
                continue
        
        # Update total requests counter
        self.total_requests += len(requests)
        
        # Log summary
        total_time = time.time() - start_time
        successful = sum(1 for r in results.values() if r['error'] is None)
        logger.info(f"Batch generation completed: {successful}/{len(requests)} successful "
                   f"in {total_time:.2f}s ({successful/total_time:.2f} images/sec)")
        
        return results
        
    def generate_single(self, 
                       prompt: str,
                       **kwargs) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """
        Convenience method to generate a single image
        
        Args:
            prompt: Text prompt
            **kwargs: Additional parameters (negative_prompt, width, height, etc.)
            
        Returns:
            Tuple of (image, metadata dict)
        """
        request = InferenceRequest(
            request_id=f"single_{int(time.time()*1000)}",
            prompt=prompt,
            **kwargs
        )
        
        results = self.generate_batch([request])
        
        if request.request_id in results:
            result = results[request.request_id]
            return result['image'], result
        else:
            return None, {'error': 'Generation failed'}
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the pipeline"""
        stats = {
            'uptime': time.time() - self.start_time,
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.total_requests - self.failed_requests) / max(1, self.total_requests),
            'tiles': {}
        }
        
        total_inferences = 0
        total_time = 0
        
        for tile in self.tiles:
            logger.info(f"Accessing stats for tile {tile.device_id}")
            # with tile.lock:
            # logger.info(f"Acquired lock for tile {tile.device_id}")
            tile_stats = {
                'gpu_index': tile.gpu_index,
                'tile_index': tile.tile_index,
                'total_inferences': tile.total_inferences,
                'total_time': tile.total_time,
                'avg_inference_time': tile.get_utilization(),
                'throughput': tile.get_throughput(),
                'is_busy': tile.is_busy,
                'idle_time': time.time() - tile.last_used if not tile.is_busy else 0
            }
            stats['tiles'][tile.device_id] = tile_stats
            
            total_inferences += tile.total_inferences
            total_time += tile.total_time
        
        # Overall statistics
        stats['total_inferences'] = total_inferences
        stats['overall_throughput'] = total_inferences / max(1, stats['uptime'])
        stats['average_utilization'] = total_time / (max(1, stats['uptime']) * len(self.tiles))
        
        return stats
        
    def save_statistics(self, filepath: str):
        """Save statistics to JSON file"""
        stats = self.get_statistics()
        stats['timestamp'] = datetime.now().isoformat()
        stats['model_config'] = {
            'base_path': str(self.model_config.base_path),
            'use_t5': self.model_config.use_t5,
            'use_flash_lora': self.model_config.use_flash_lora,
            'model_variant': self.model_config.model_variant
        }
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
            
    def shutdown(self):
        """Gracefully shutdown the pipeline"""
        logger.info("Shutting down multi-GPU pipeline...")
        
        if not self.test_mode and self.worker_threads:
            # Signal workers to stop
            self.stop_workers = True
            
            # Add shutdown signals to queue
            for _ in self.worker_threads:
                self.request_queue.put((0, next(self.counter), None))  # High priority shutdown
            
            # Wait for workers to finish
            for worker in self.worker_threads:
                worker.join(timeout=10)
            
            # Save final statistics
            self.save_statistics("aurora_sd35_final_stats.json")
        
        logger.info("Multi-GPU pipeline shutdown complete")


def verify_openvino_genai():
    """Verify OpenVINO GenAI installation and available components"""
    try:
        import openvino_genai as ov_genai
        components = {
            'CLIPTextModelWithProjection': hasattr(ov_genai, 'CLIPTextModelWithProjection'),
            'SD3Transformer2DModel': hasattr(ov_genai, 'SD3Transformer2DModel'),
            'AutoencoderKL': hasattr(ov_genai, 'AutoencoderKL'),
            'Scheduler': hasattr(ov_genai, 'Scheduler'),
            'Text2ImagePipeline': hasattr(ov_genai, 'Text2ImagePipeline'),
            'T5TextEncoder': hasattr(ov_genai, 'T5TextEncoder')
        }
        
        missing = [k for k, v in components.items() if not v]
        if missing:
            logger.warning(f"Missing OpenVINO GenAI components: {missing}")
            
        return components
    except ImportError:
        logger.error("OpenVINO GenAI not installed!")
        return None


def diagnose_gpu_devices():
    """Utility function to diagnose GPU device configuration on Aurora"""
    core = ov.Core()
    devices = core.available_devices
    
    print("="*60)
    print("OpenVINO GPU Device Diagnostics")
    print("="*60)
    print(f"OpenVINO version: {ov.__version__}")
    
    # Try to get openvino_genai version
    try:
        genai_version = ov_genai.__version__ if hasattr(ov_genai, '__version__') else "Unknown"
        print(f"OpenVINO GenAI version: {genai_version}")
    except:
        print("OpenVINO GenAI version: Unable to determine")
    
    # Check openvino_genai components
    print("\nOpenVINO GenAI components:")
    genai_components = [
        'CLIPTextModelWithProjection',
        'SD3Transformer2DModel', 
        'AutoencoderKL',
        'Scheduler',
        'Text2ImagePipeline',
        'T5TextEncoder'  # May not be available
    ]
    
    for component in genai_components:
        if hasattr(ov_genai, component):
            print(f"  ✓ {component}")
        else:
            print(f"  ✗ {component} (not available)")
    
    print(f"\nAvailable devices: {devices}")
    print()
    
    gpu_devices = [d for d in devices if d.startswith('GPU')]
    print(f"Found {len(gpu_devices)} GPU devices:")
    
    for device in sorted(gpu_devices):
        print(f"\n{device}:")
        try:
            props = [
                "FULL_DEVICE_NAME",
                "DEVICE_TYPE", 
                "DEVICE_ARCHITECTURE",
                "GPU_EXECUTION_UNITS_COUNT",
                "GPU_FREQUENCY",
                "CACHE_DIR",
                "AVAILABLE_DEVICES"
            ]
            
            for prop in props:
                try:
                    value = core.get_property(device, prop)
                    print(f"  {prop}: {value}")
                except:
                    pass
                    
        except Exception as e:
            print(f"  Error getting properties: {e}")
    
    print("\n" + "="*60)


def main():
    """Example usage of the Aurora SD3.5 multi-GPU pipeline"""
    
    # Check for command line flags
    use_t5 = "--no-t5" not in sys.argv
    
    # Optional: Run GPU diagnostics first
    if "--diagnose" in sys.argv:
        diagnose_gpu_devices()
        return
    
    # Configure for Aurora
    aurora_config = AuroraConfig(
        node_gpus=3,
        tiles_per_gpu=2,  # Each Max 1550 presents 2 tiles as separate GPU devices
        total_tile_devices=6,  # Total visible GPU.X devices
        target_batch_size=2
    )
    
    # Model configuration
    model_config = ModelConfig(
        base_path=Path("/lus/flare/projects/hp-ptycho/binkma/models/sd3.5"),
        use_t5=use_t5,  # Can be disabled via --no-t5 flag
        use_flash_lora=False,
        model_variant="tensorart/stable-diffusion-3.5-medium-turbo"
    )
    
    # Check if T5 was disabled
    if not model_config.use_t5:
        logger.info("Running without T5 text encoder")
    
    # Initialize pipeline
    # Use None to use all available GPU devices, or specify a number
    pipeline = SD35_Aurora_MultiGPU(
        model_config=model_config,
        aurora_config=aurora_config,
        max_tiles=None,  # Use all available GPU devices
        enable_profiling=True
    )
    
    try:
        # Example 1: Generate single image
        logger.info("Generating single image...")
        image, metadata = pipeline.generate_single(
            prompt="A majestic aurora borealis over a futuristic Chicago skyline",
            negative_prompt="blurry, low quality",
            width=1024,
            height=1024,
            num_inference_steps=28,
            guidance_scale=7.5
        )
        
        if image:
            image.save("aurora_single_test.png")
            logger.info(f"Single image generated in {metadata['inference_time']:.2f}s")
        
        # Example 2: Batch generation
        logger.info("\nGenerating batch of images...")
        
        test_prompts = [
            "A cyberpunk version of Argonne National Laboratory",
            "Quantum computers arranged in a beautiful pattern",
            "Data flowing through fiber optic cables like rivers of light",
            "A supercomputer datacenter with ethereal blue lighting",
            "Visualization of parallel computing processes as abstract art",
            "Northern lights dancing over a field of server racks",
            "A futuristic command center monitoring global computations",
            "Binary code transforming into butterflies",
            "A digital forest where trees are made of circuit boards",
            "Time-lapse of AI learning visualized as growing crystals",
            "A surreal landscape where data streams form waterfalls",
            "A cityscape where buildings are made of glowing data",
        ]
        
        # Create requests with different priorities
        requests = []
        for i, prompt in enumerate(test_prompts):
            request = InferenceRequest(
                request_id=f"batch_{i:03d}",
                prompt=prompt,
                negative_prompt="ugly, blurry, low quality, distorted",
                width=768,
                height=768,
                num_inference_steps=20,
                guidance_scale=5.0,
                seed=42 + i,
                priority=i % 3  # Vary priorities
            )
            requests.append(request)
        
        # Generate batch
        results = pipeline.generate_batch(requests, timeout=300)  # 5 minute timeout
        
        # Save results
        output_dir = Path("./aurora_output")
        output_dir.mkdir(exist_ok=True)
        
        for request_id, result in results.items():
            if result['error'] is None:
                result['image'].save(output_dir / f"{request_id}.png")
                logger.info(f"{request_id}: Generated in {result['inference_time']:.2f}s on {result['tile_id']}")
            else:
                logger.error(f"{request_id}: Failed - {result['error']}")
        
        logger.info(f"Batch generation completed: {len(results)} results processed")
        # Print statistics
        stats = pipeline.get_statistics()
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*60)
        logger.info(f"Total uptime: {stats['uptime']:.2f}s")
        logger.info(f"Total requests: {stats['total_requests']}")
        logger.info(f"Success rate: {stats['success_rate']*100:.1f}%")
        logger.info(f"Overall throughput: {stats['overall_throughput']:.2f} images/second")
        logger.info(f"Average utilization: {stats['average_utilization']*100:.1f}%")
        
        logger.info("\nPER-TILE STATISTICS:")
        for tile_id, tile_stats in stats['tiles'].items():
            logger.info(f"\n{tile_id} (GPU {tile_stats['gpu_index']}, Tile {tile_stats['tile_index']}):")
            logger.info(f"  Inferences: {tile_stats['total_inferences']}")
            logger.info(f"  Avg time: {tile_stats['avg_inference_time']:.2f}s")
            logger.info(f"  Throughput: {tile_stats['throughput']:.2f} img/s")
            
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        # Cleanup
        pipeline.shutdown()


def test_single_gpu_pipeline():
    """Test creating a simple single-GPU pipeline"""
    logger.info("Testing single GPU pipeline creation...")
    
    try:
        # Model paths
        base_path = Path("/lus/flare/projects/hp-ptycho/binkma/models/sd3.5")
        
        # Initialize components
        scheduler = ov_genai.Scheduler.from_config(base_path / "scheduler/scheduler_config.json")
        text_encoder = ov_genai.CLIPTextModelWithProjection(base_path / "text_encoder", "GPU.0")
        text_encoder_2 = ov_genai.CLIPTextModelWithProjection(base_path / "text_encoder_2", "GPU.0")
        transformer = ov_genai.SD3Transformer2DModel(base_path / "transformer", "GPU.0")
        vae = ov_genai.AutoencoderKL(base_path / "vae_decoder", device="GPU.0")
        
        # Create pipeline
        pipeline = ov_genai.Text2ImagePipeline.stable_diffusion_3(
            scheduler, text_encoder, text_encoder_2, transformer, vae
        )
        
        logger.info("Successfully created single GPU pipeline!")
        
        # Test generation
        logger.info("Testing image generation...")
        result = pipeline.generate(
            prompt="A test image",
            num_inference_steps=4,
            height=512,
            width=512,
            guidance_scale=1.0
        )
        
        logger.info("Generation successful!")
        return True
        
    except Exception as e:
        logger.error(f"Single GPU pipeline test failed: {e}")
        return False


if __name__ == "__main__":
    # Add command line options
    if len(sys.argv) > 1:
        if "--diagnose" in sys.argv:
            diagnose_gpu_devices()
            sys.exit(0)
        elif "--test" in sys.argv:
            # Quick test mode - just verify GPU discovery
            logger.info("Running in test mode - GPU discovery only")
            aurora_config = AuroraConfig()
            test_pipeline = SD35_Aurora_MultiGPU(
                model_config=ModelConfig(use_t5=False),  # Disable T5 for test
                aurora_config=aurora_config,
                max_tiles=2,  # Just test with 2 devices
                enable_profiling=False,
                test_mode=True  # Skip model loading
            )
            stats = test_pipeline.get_statistics()
            logger.info(f"Successfully initialized with {len(test_pipeline.tiles)} GPU devices")
            test_pipeline.shutdown()
            sys.exit(0)
        elif "--test-single" in sys.argv:
            # Test single GPU pipeline
            success = test_single_gpu_pipeline()
            sys.exit(0 if success else 1)
        elif "--help" in sys.argv or "-h" in sys.argv:
            print(__doc__)
            sys.exit(0)
    
    main()