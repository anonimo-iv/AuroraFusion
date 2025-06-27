#!/usr/bin/env python
import argparse
import logging
import math
import os
import gc
import random
import shutil
import warnings
from pathlib import Path
import json
import socket
import numpy as np
import torch
import torch.distributed as dist
import transformers
from torch.utils.data.distributed import DistributedSampler
from huggingface_hub import create_repo, upload_folder
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers.modeling_outputs import BaseModelOutputWithPooling
import deepspeed
from mpi4py import MPI
from torch.profiler import profile, record_function, ProfilerActivity

# Intel GPU specific imports
try:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch as torch_ccl
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False
    print("Intel Extension for PyTorch not installed. Performance may be suboptimal.")

# XPU-specific autocast
from torch.amp import autocast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from diffusers.utils import check_min_version, is_wandb_available

if is_wandb_available():
    import wandb

check_min_version("0.34.0.dev0")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Unconditional SD3 Training Script with DeepSpeed on Aurora")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Model variant (e.g. fp16)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name from HuggingFace hub",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Dataset config name",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Directory containing training images",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default=None,
        help="Directory containing validation images",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="Image column name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd3-unconditional",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Center crop images",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="Randomly flip images",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total training steps (overrides num_train_epochs if set)",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save checkpoint every X steps",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max checkpoints to keep",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Data loader workers",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="Logit mean"
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="Logit std"
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Mode scale",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="Run validation every X steps",
    )
    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Precondition outputs",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Push to Hub")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    # DeepSpeed config file (this is the key parameter)
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default=None,
        required=True,
        help="Path to DeepSpeed config JSON file",
    )
    
    # Intel GPU specific
    parser.add_argument(
        "--use_ipex_optimize",
        action="store_true",
        default=True,
        help="Use IPEX optimization for Intel GPU",
    )

    # Parse known args to allow DeepSpeed to add its own arguments
    args, unknown = parser.parse_known_args(input_args)

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    return args


def get_weight_dtype_from_config(ds_config_path):
    """Extract weight dtype from DeepSpeed config"""
    with open(ds_config_path, 'r') as f:
        config = json.load(f)
    
    if config.get("fp16", {}).get("enabled", False):
        return torch.float16
    elif config.get("bf16", {}).get("enabled", False):
        return torch.bfloat16
    else:
        return torch.float32

class UnconditionalImageDataset(Dataset):
    def __init__(
        self,
        data_root,
        size=512,
        center_crop=False,
        is_validation=False,
        args=None
    ):
        self.size = size
        self.center_crop = center_crop
        self.is_validation = is_validation
        self.args = args
        
        if os.path.isdir(data_root):
            self.data_root = Path(data_root)
            if not self.data_root.exists():
                raise ValueError(f"Data directory {data_root} does not exist.")
                
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            self.image_paths = []
            for ext in image_extensions:
                self.image_paths.extend(sorted(list(self.data_root.glob(ext))))
            
            if len(self.image_paths) == 0:
                raise ValueError(f"No images found in {data_root}. Supported formats: {', '.join(image_extensions)}")
            
            logger.info(f"Found {len(self.image_paths)} images in {data_root}")
        
        elif args and args.dataset_name is not None:
            from datasets import load_dataset
            try:
                dataset = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    cache_dir=args.cache_dir,
                    split="train" if not is_validation else "validation"
                )
                self.images = dataset[args.image_column]
                logger.info(f"Loaded {len(self.images)} examples from dataset {args.dataset_name}")
            except Exception as e:
                logger.error(f"Error loading dataset: {e}")
                raise
        else:
            raise ValueError("Invalid data source")
        
        # Define transforms
        if is_validation:
            self.transform = transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            transform_list = [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            ]
            if args and args.random_flip:
                transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            self.transform = transforms.Compose(transform_list)

    def __len__(self):
        if hasattr(self, 'images'):
            return len(self.images)
        return len(self.image_paths)
    
    def __getitem__(self, index):
        try:
            if hasattr(self, 'images'):
                image = self.images[index]
                if not isinstance(image, Image.Image):
                    if isinstance(image, str):
                        image = Image.open(image).convert("RGB")
                    elif isinstance(image, np.ndarray):
                        image = Image.fromarray(image).convert("RGB")
                    else:
                        raise ValueError(f"Unsupported image type: {type(image)}")
            else:
                image = Image.open(self.image_paths[index]).convert("RGB")
            
            image = exif_transpose(image)
            image = self.transform(image)
            
            return {
                "pixel_values": image
            }
        except Exception as e:
            logger.error(f"Error loading item {index}: {e}")
            # Return dummy image
            dummy_image = torch.zeros(3, self.size, self.size)
            return {
                "pixel_values": dummy_image
            }

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    return {"pixel_values": pixel_values}

def set_seed(seed, rank=0):
    """Set random seed for reproducibility"""
    actual_seed = seed + rank  # Different seed per rank
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    torch.xpu.manual_seed_all(actual_seed)

def log_validation(transformer, args, rank, epoch, vae, noise_scheduler, weight_dtype):
    """Generate validation images on XPU"""
    if rank != 0:  # Only run on main process
        return []
        
    logger.info("Running unconditional validation...")
    original_rng_state = torch.get_rng_state()
    original_xpu_rng_state = torch.xpu.get_rng_state()

    try:
        device = torch.device(f'xpu:{torch.xpu.current_device()}')
        
        with torch.no_grad():
            seed = args.seed + epoch if args.seed is not None else None
            if seed is not None:
                torch.manual_seed(seed)
                torch.xpu.manual_seed_all(seed)

            generator = torch.Generator(device=device)
            if seed is not None:
                generator.manual_seed(seed)

            # Create dummy components for unconditional generation
            class DummyTokenizer:
                def __init__(self):
                    self.model_max_length = 77
                    
                def __call__(self, text=None, return_tensors="pt", padding=True, truncation=True, **kwargs):
                    batch_size = 1 if text is None else len(text) if isinstance(text, list) else 1
                    return type('', (), {
                        'input_ids': torch.zeros((batch_size, self.model_max_length), dtype=torch.long),
                        'attention_mask': torch.ones((batch_size, self.model_max_length), dtype=torch.long),
                        'to': lambda self, device: self
                    })()

            class DummyTextEncoder:
                def __init__(self):
                    self.config = type('', (), {'hidden_size': 4096})()
                    self.dtype = weight_dtype
                    self.device = device
                    
                def __call__(self, input_ids=None, attention_mask=None, **kwargs):
                    batch_size = 1
                    seq_len = 77
                    
                    return type('', (), {
                        'last_hidden_state': torch.zeros((batch_size, seq_len, self.config.hidden_size), dtype=self.dtype, device=self.device),
                        'hidden_states': [torch.zeros((batch_size, seq_len, self.config.hidden_size), dtype=self.dtype, device=self.device) for _ in range(3)],
                        'pooler_output': torch.zeros((batch_size, 2048), dtype=self.dtype, device=self.device)
                    })()
            
            # Create pipeline
            dummy_tokenizer = DummyTokenizer()
            dummy_text_encoder = DummyTextEncoder()
            pipeline = StableDiffusion3Pipeline(
                vae=vae,
                transformer=transformer,
                scheduler=noise_scheduler,
                tokenizer=dummy_tokenizer,
                text_encoder=dummy_text_encoder,
                tokenizer_2=dummy_tokenizer,
                text_encoder_2=dummy_text_encoder,
                tokenizer_3=dummy_tokenizer,
                text_encoder_3=dummy_text_encoder,
            )
            pipeline = pipeline.to(device)
            pipeline.set_progress_bar_config(disable=True)
            
            # Generate images
            images = []
            for i in range(args.num_validation_images):
                try:
                    mixed_precision_enabled = weight_dtype != torch.float32
                    with autocast(device_type='xpu', dtype=weight_dtype, enabled=mixed_precision_enabled):
                        image = pipeline(
                            prompt="", 
                            generator=generator,
                            num_inference_steps=20,
                            guidance_scale=7.0
                        ).images[0]
                    images.append(image)
                except Exception as e:
                    logger.error(f"Error generating image {i+1}: {e}")
                    images.append(None)

            return images

    finally:
        torch.set_rng_state(original_rng_state)
        torch.xpu.set_rng_state(original_xpu_rng_state)
        if 'pipeline' in locals():
            del pipeline
        free_memory()
        torch.xpu.empty_cache()

def run_validation(model_engine, vae, noise_scheduler, val_dataloader, args, weight_dtype, device):
    """Run validation loop on XPU"""
    logger.info("Running validation...")
    model_engine.eval()
    total_val_loss = 0.0
    num_batches = 0
    
    mixed_precision_enabled = weight_dtype != torch.float32
    
    with torch.no_grad():
        for step, val_batch in enumerate(val_dataloader):
            pixel_values = val_batch["pixel_values"].to(device=device, dtype=weight_dtype)
            with autocast(device_type='xpu', dtype=weight_dtype, enabled=mixed_precision_enabled):
                # Encode images to latents
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                noise = torch.randn_like(latents).to(dtype=latents.dtype, device=latents.device)
                bsz = latents.shape[0]
                
                # Sample timesteps
                u = compute_density_for_timestep_sampling(
                    args.weighting_scheme,
                    bsz,
                    args.logit_mean,
                    args.logit_std,
                    args.mode_scale
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(latents.device, dtype=weight_dtype)
                
                # Add noise
                sigmas = noise_scheduler.sigmas[indices].view(-1, 1, 1, 1).to(latents.device)
                noise = noise.to(latents.device)
                noisy_latents = (1 - sigmas) * latents + sigmas * noise
                    
                batch_size = noisy_latents.shape[0]
                dummy_pooled = torch.zeros(
                    batch_size, 2048, 
                    device=device, 
                    dtype=weight_dtype
                )
                dummy_encoder = torch.zeros(
                    batch_size, 77, 4096, 
                    device=device, 
                    dtype=weight_dtype
                )
                
                # Model prediction
                model_pred = model_engine.module(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=dummy_encoder,
                    pooled_projections=dummy_pooled,
                    return_dict=False
                )[0]
                
                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_latents
                
                weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas)
                target = noise - latents if not args.precondition_outputs else latents
                loss = torch.mean((weighting * (model_pred - target) ** 2).mean())
                total_val_loss += loss.item()
                num_batches += 1
    
    avg_val_loss = total_val_loss / max(num_batches, 1) 
    model_engine.train()
    return avg_val_loss

def setup_distributed():
    """Initialize distributed training for Aurora with MPI and oneCCL"""
    # Aurora uses MPI for distributed initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    # Aurora-specific environment variables
    local_rank_env_vars = ['PALS_LOCAL_RANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 
                           'MPI_LOCALRANKID', 'LOCAL_RANK']
    local_rank = -1
    for var in local_rank_env_vars:
        if var in os.environ:
            local_rank = int(os.environ[var])
            break
    
    if local_rank == -1:
        # Fallback to calculating local rank
        local_rank = rank % torch.xpu.device_count()
    
    # Set environment variables for torch.distributed
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)
    
    # Get master address
    master_addr = socket.gethostname() if rank == 0 else None
    master_addr = comm.bcast(master_addr, root=0)

    torch.xpu.set_device(local_rank)
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    print(f"DDP: Hi from rank {rank} of {world_size} with local rank {local_rank}. {master_addr}")
    # Set XPU-specific environment variables for optimization

    os.environ['CCL_BACKEND'] = 'native'
    os.environ['CCL_ATL_TRANSPORT'] = 'ofi'
    os.environ['FI_PROVIDER'] = 'cxi'

    os.environ['CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD'] = '5000'
    
    # Critical for multi-XPU on Aurora
    os.environ['CCL_ZE_IPC_EXCHANGE'] = 'sockets'
    os.environ['CCL_ZE_ENABLE'] = '1'
    os.environ['CCL_LOG_LEVEL'] = 'info'


    os.environ['IPEX_XPU_ONEDNN_LAYOUT'] = '1'
    os.environ['IPEX_OFFLINE_COMPILER'] = '1'

    # Prevent build-for-1-device issues
    os.environ['SYCL_CACHE_PERSISTENT'] = '1'
    os.environ['SYCL_DEVICE_FILTER'] = 'level_zero:*'
    os.environ['SYCL_PI_LEVEL_ZERO_PROGRAM_BUILD_TRACK'] = '2'
    
    # if world_size > 1:
        # Initialize with oneCCL backend
        # dist.init_process_group(backend='ccl', rank=rank, world_size=world_size)
    if world_size > 1:
        dist.init_process_group(backend='ccl',init_method='env://', rank=int(rank), world_size=int(world_size))
        
    
    logger.info(f"Initialized distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    
    return rank, world_size, local_rank


def main():
    args = parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Setup device
    device = torch.device(f'xpu:{local_rank}')
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if rank == 0 else logging.WARNING,
    )
    
    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed, rank)
    
    # Create output directory
    if rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(Path(args.output_dir).name, exist_ok=True).repo_id

    # Get weight dtype from DeepSpeed config
    weight_dtype = get_weight_dtype_from_config(args.deepspeed_config)
    logger.info(f"Using weight dtype: {weight_dtype}")

    # Load models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", variant=args.variant
    )

    # Load and optimize VAE with IPEX
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="vae", 
        variant=args.variant,
        torch_dtype=weight_dtype  
    )
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    # Apply IPEX optimization if available
    if IPEX_AVAILABLE and args.use_ipex_optimize:
        logger.info("Applying IPEX optimization to VAE...")
        vae = ipex.optimize(vae.eval(), dtype=weight_dtype)
    
    
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    # Enable XPU optimization
    if IPEX_AVAILABLE:
        # torch.xpu.optimize_for_training()
        logger.info("Applying IPEX optimization to transformer...")
        transformer = ipex.optimize(transformer, dtype=weight_dtype)
    
    # Create datasets
    train_dataset = UnconditionalImageDataset(
        args.train_data_dir,
        size=args.resolution,
        center_crop=args.center_crop,
        args=args
    )
    
    if args.val_data_dir:
        val_dataset = UnconditionalImageDataset(
            args.val_data_dir,
            size=args.resolution,
            center_crop=args.center_crop,
            is_validation=True,
            args=args
        )
    else:
        # Split training dataset
        total_size = len(train_dataset)
        val_size = int(total_size * 0.1)
        indices = list(range(total_size))
        random.shuffle(indices)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        from torch.utils.data import Subset
        val_dataset = Subset(train_dataset, val_indices)
        train_dataset = Subset(train_dataset, train_indices)
        
        if rank == 0:
            logger.info(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation")

    # Create data loaders
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        set_seed(worker_seed)

    # Get batch size from DeepSpeed config
    with open(args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)
    train_batch_size = ds_config.get("train_micro_batch_size_per_gpu", ds_config.get("train_batch_size", 1) // world_size)
    gradient_accumulation_steps = ds_config.get("gradient_accumulation_steps", 1)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=False  # XPU doesn't support pin_memory well
    )
    
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=False
    )

    # Initialize DeepSpeed
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=transformer,
        config=args.deepspeed_config,
        model_parameters=transformer.parameters(),
        dist_init_required=False  # We already initialized distributed
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # Initialize tracking
    if rank == 0 and is_wandb_available():
        wandb.init(project="sd3-unconditional-aurora", config=vars(args))
    
    # Training loop
    global_step = 0
    best_val_loss = float("inf")
    
    if args.resume_from_checkpoint:
        if os.path.isfile(args.resume_from_checkpoint):
            checkpoint_path = args.resume_from_checkpoint
        else:
            checkpoint_path = os.path.join(args.resume_from_checkpoint, "checkpoint")
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        _, client_state = model_engine.load_checkpoint(checkpoint_path)
        if client_state is not None and 'global_step' in client_state:
            global_step = client_state['global_step']

    if rank == 0:
        progress_bar = tqdm(
            total=args.max_train_steps,
            desc="Training progress",
            initial=global_step,
            dynamic_ncols=True
        )

    mixed_precision_enabled = weight_dtype != torch.float32
    

    profiler_output_dir = "/lus/flare/projects/hp-ptycho/binkma/profiler/sd3.5m"
    for epoch in range(args.num_train_epochs):
        model_engine.train()
        train_sampler.set_epoch(epoch)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            schedule=torch.profiler.schedule(
                wait=2,  
                warmup=1,
                active=2,  
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for step, batch in enumerate(train_dataloader):
                with record_function("training_step"):
                    with autocast(device_type='xpu', dtype=weight_dtype, enabled=mixed_precision_enabled):
                        with record_function("data_loading"):
                            pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)

                        # Encode images to latents
                        with record_function("vae_encoding"):
                            with torch.no_grad():
                                latents = vae.encode(pixel_values).latent_dist.sample()
                                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                        
                        with record_function("noise_preparation"):
                            noise = torch.randn_like(latents).to(dtype=latents.dtype, device=latents.device)
                            bsz = latents.shape[0]
                            
                            # Sample timesteps
                            u = compute_density_for_timestep_sampling(
                                args.weighting_scheme,
                                bsz,
                                args.logit_mean,
                                args.logit_std,
                                args.mode_scale
                            )
                            indices = (u * noise_scheduler.config.num_train_timesteps).long()
                            timesteps = noise_scheduler.timesteps[indices].to(latents.device, dtype=weight_dtype)
                            
                            # Add noise
                            sigmas = noise_scheduler.sigmas[indices].view(-1, 1, 1, 1).to(latents.device)
                            noise = noise.to(latents.device)
                            noisy_latents = (1 - sigmas) * latents + sigmas * noise
                        
                        with record_function("dummy_inputs_creation"):
                            batch_size = noisy_latents.shape[0]
                            dummy_pooled = torch.zeros(
                                batch_size,
                                2048,
                                device=noisy_latents.device,
                                dtype=weight_dtype
                            )
                            dummy_encoder = torch.zeros(
                                batch_size,
                                77,
                                4096,
                                device=noisy_latents.device,
                                dtype=weight_dtype
                            )
                        
                        # Model prediction (unconditional)
                        with record_function("model_forward"):
                            model_pred = model_engine(
                                hidden_states=noisy_latents,
                                timestep=timesteps,
                                encoder_hidden_states=dummy_encoder,
                                pooled_projections=dummy_pooled,
                                return_dict=False
                            )[0]
                        with record_function("loss_computation"):
                            if args.precondition_outputs:
                                model_pred = model_pred * (-sigmas) + noisy_latents
                            
                            # Calculate loss
                            weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas)
                            target = noise - latents if not args.precondition_outputs else latents
                            loss = torch.mean((weighting * (model_pred - target) ** 2).mean())
                        
                    # Backward pass with DeepSpeed
                    with record_function("backward_pass"):
                        model_engine.backward(loss)
                    with record_function("optimizer_step"):
                        model_engine.step()
                
                    if rank == 0:
                        if global_step % 10 == 0:  # Log every 10 steps
                            logger.info(f"Step {global_step}: Loss = {loss.item():.4f}")
                        progress_bar.update(1)
                
                    global_step += 1
                    prof.step()
                    # Checkpointing
                    if global_step % args.checkpointing_steps == 0 and global_step > 0:
                        if rank == 0:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            os.makedirs(save_path, exist_ok=True)
                            
                        # Save checkpoint with client state
                        client_state = {
                            'global_step': global_step,
                            'epoch': epoch,
                            'best_val_loss': best_val_loss
                        }
                        model_engine.save_checkpoint(args.output_dir, tag=f"checkpoint-{global_step}", 
                                                    client_state=client_state)
                        
                        if rank == 0:
                            logger.info(f"Saved checkpoint at step {global_step}")
                        
                        # Manage checkpoint limits
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint-")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                                
                                if len(checkpoints) > args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                                    removing_checkpoints = checkpoints[:num_to_remove]
                                    
                                    logger.info(f"Removing {len(removing_checkpoints)} checkpoints to maintain limit of {args.checkpoints_total_limit}")
                                    
                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint_path = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint_path, ignore_errors=True)
                    
                    # Validation
                    if global_step % args.validation_steps == 0 and global_step > 0:
                        val_sampler.set_epoch(epoch)
                        val_loss = run_validation(
                            model_engine=model_engine,
                            vae=vae,
                            noise_scheduler=noise_scheduler,
                            val_dataloader=val_dataloader,
                            args=args,
                            weight_dtype=weight_dtype,
                            device=device
                        )
                    
                        if rank == 0:
                            logger.info(f"Step {global_step}: Validation Loss = {val_loss:.4f}")

                        model_engine.train()
                    
                    if global_step >= args.max_train_steps:
                        break
            
            if global_step >= args.max_train_steps:
                break

    # Final save
    if rank == 0:
        logger.info("Finalizing training...")
        progress_bar.close()

    # Save final model
    client_state = {
        'global_step': global_step,
        'epoch': epoch,
        'best_val_loss': best_val_loss
    }
    model_engine.save_checkpoint(args.output_dir, tag="final_model", client_state=client_state)

    if rank == 0:
        logger.info("Training completed. Final model saved.")
        
        # Convert DeepSpeed checkpoint to regular PyTorch checkpoint if needed
        try:
            from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
            
            # Convert the best model
            best_model_path = os.path.join(args.output_dir, "best_model")
            if os.path.exists(best_model_path):
                logger.info("Converting best model checkpoint to fp32...")
                convert_zero_checkpoint_to_fp32_state_dict(
                    best_model_path,
                    os.path.join(args.output_dir, "best_model_fp32.bin"),
                    tag="best_model"
                )
            
            # Convert the final model
            logger.info("Converting final model checkpoint to fp32...")
            convert_zero_checkpoint_to_fp32_state_dict(
                args.output_dir,
                os.path.join(args.output_dir, "final_model_fp32.bin"),
                tag="final_model"
            )
        except Exception as e:
            logger.warning(f"Could not convert DeepSpeed checkpoint: {e}")

    # Push to hub if requested
    if args.push_to_hub and rank == 0:
        logger.info("Pushing to hub...")
        upload_folder(
            repo_id=repo_id,
            folder_path=args.output_dir,
            commit_message="End of training",
            ignore_patterns=["step_*", "epoch_*", "checkpoint-*"],
        )

    # Cleanup
    prof.export_chrome_trace(os.path.join(profiler_output_dir, f"train_trace_rank_{rank}.json"))

    torch.xpu.synchronize()
    torch.xpu.empty_cache()

    if dist.is_initialized():
        dist.destroy_process_group()
    gc.collect()
    

if __name__ == "__main__":
   main()