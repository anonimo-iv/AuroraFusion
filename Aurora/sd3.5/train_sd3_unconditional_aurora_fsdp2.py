#!/usr/bin/env python
import argparse
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path
import json
import socket
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, CPUOffloadPolicy
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.checkpoint import (
    save_state_dict,
    load_state_dict,
    FileSystemReader,
    FileSystemWriter,
)
from torch.amp import GradScaler, autocast
import transformers
from torch.utils.data.distributed import DistributedSampler
from huggingface_hub import create_repo, upload_folder
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers.modeling_outputs import BaseModelOutputWithPooling
from mpi4py import MPI

# Intel GPU specific imports
try:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch as torch_ccl
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False
    print("Intel Extension for PyTorch not installed. Performance may be suboptimal.")

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
    parser = argparse.ArgumentParser(description="Unconditional SD3 Training Script with FSDP2 on Aurora")
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
        default="sd3-unconditional-fsdp2",
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
        default=100,
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
    
    # FSDP specific arguments
    parser.add_argument(
        "--fsdp_cpu_offload",
        action="store_true",
        help="Enable CPU offloading for FSDP2",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Training batch size per GPU",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=0.01,
        help="Adam weight decay",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Adam epsilon",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Learning rate warmup steps",
    )
    
    # Intel GPU specific
    parser.add_argument(
        "--use_ipex_optimize",
        action="store_true",
        default=True,
        help="Use IPEX optimization for Intel GPU",
    )

    args = parser.parse_args(input_args)

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    return args


def get_fsdp2_mixed_precision(dtype_str):
    """Configure FSDP2 mixed precision for XPU"""
    if dtype_str == "fp16":
        dtype = torch.float16
    elif dtype_str == "bf16":
        dtype = torch.bfloat16
    else:
        return None
    
    return MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )


def get_weight_dtype(mixed_precision):
    """Get weight dtype from mixed precision string"""
    if mixed_precision == "fp16":
        return torch.float16
    elif mixed_precision == "bf16":
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
                transformer=transformer,  # FSDP2 doesn't need unwrapping
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


def run_validation(transformer, vae, noise_scheduler, val_dataloader, args, weight_dtype, device):
    """Run validation loop on XPU"""
    logger.info("Running validation...")
    transformer.eval()
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
                model_pred = transformer(
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
    transformer.train()
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
    
    if world_size > 1:
        dist.init_process_group(backend='ccl', init_method='env://', rank=int(rank), world_size=int(world_size))
        
    logger.info(f"Initialized distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    
    return rank, world_size, local_rank


def save_checkpoint_fsdp2(transformer, optimizer, lr_scheduler, global_step, epoch, args, best_val_loss):
    """Save FSDP2 checkpoint using torch.distributed.checkpoint"""
    save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare state dict
    state_dict = {
        "model": transformer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }
    
    # Save using distributed checkpoint
    writer = FileSystemWriter(save_dir)
    save_state_dict(
        state_dict,
        writer,
        no_dist=False,  # Use distributed saving
    )
    
    if dist.get_rank() == 0:
        logger.info(f"Saved checkpoint at step {global_step}")
    
    dist.barrier()


def load_checkpoint_fsdp2(transformer, optimizer, lr_scheduler, checkpoint_path):
    """Load FSDP2 checkpoint"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Prepare state dict structure
    state_dict = {
        "model": transformer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": torch.tensor(0),
        "global_step": torch.tensor(0),
        "best_val_loss": torch.tensor(float("inf")),
    }
    
    # Load using distributed checkpoint
    reader = FileSystemReader(checkpoint_path)
    load_state_dict(
        state_dict,
        reader,
        no_dist=False,
    )
    
    # Load states
    transformer.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
    
    training_state = {
        'global_step': state_dict["global_step"].item(),
        'epoch': state_dict["epoch"].item(),
        'best_val_loss': state_dict["best_val_loss"].item(),
    }
    
    return training_state

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


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

    # Get weight dtype
    weight_dtype = get_weight_dtype(args.mixed_precision)
    logger.info(f"Using weight dtype: {weight_dtype}")

    # Load models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Load transformer
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="transformer", 
        variant=args.variant,
        torch_dtype=weight_dtype
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
    
    # ========== FSDP2 Configuration ==========
    # Configure mixed precision
    mixed_precision_policy = get_fsdp2_mixed_precision(args.mixed_precision)
    
    # Configure CPU offload
    offload_policy = CPUOffloadPolicy(offload_params=True) if args.fsdp_cpu_offload else None
    
    # Apply FSDP2 to the entire transformer model
    logger.info("Applying FSDP2 to transformer model...")
    fully_shard(
        transformer,
        mp_policy=mixed_precision_policy,
        offload_policy=offload_policy,
        reshard_after_forward=True,  # Important for memory efficiency
    )
    
    # Create optimizer AFTER applying FSDP2
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
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


    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=False  # XPU doesn't support pin_memory well
    )
    
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=False
    )

    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # Create learning rate scheduler
    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    # Initialize tracking
    if rank == 0 and is_wandb_available():
        wandb.init(project="sd3-unconditional-aurora-fsdp2", config=vars(args))
    
    # Training loop
    global_step = 0
    best_val_loss = float("inf")
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        if os.path.isfile(args.resume_from_checkpoint):
            checkpoint_path = args.resume_from_checkpoint
        else:
            checkpoint_path = args.resume_from_checkpoint
        
        training_state = load_checkpoint_fsdp2(transformer, optimizer, lr_scheduler, checkpoint_path)
        if training_state:
            global_step = training_state.get('global_step', 0)
            best_val_loss = training_state.get('best_val_loss', float("inf"))

    if rank == 0:
        progress_bar = tqdm(
            total=args.max_train_steps,
            desc="Training progress",
            initial=global_step,
            dynamic_ncols=True
        )

    mixed_precision_enabled = weight_dtype != torch.float32
    
    # Create gradient scaler for mixed precision (standard GradScaler for FSDP2)
    scaler = GradScaler(enabled=(args.mixed_precision == "fp16"))

    for epoch in range(args.num_train_epochs):
        transformer.train()
        train_sampler.set_epoch(epoch)
        
        for step, batch in enumerate(train_dataloader):
            with autocast(device_type='xpu', dtype=weight_dtype, enabled=mixed_precision_enabled):
                pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)

                # Encode images to latents
                with torch.no_grad():
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
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=dummy_encoder,
                    pooled_projections=dummy_pooled,
                    return_dict=False
                )[0]
                
                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_latents
                
                # Calculate loss
                weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas)
                target = noise - latents if not args.precondition_outputs else latents
                loss = torch.mean((weighting * (model_pred - target) ** 2).mean())
                loss = loss / args.gradient_accumulation_steps
                
            # Backward pass
            if args.mixed_precision == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.mixed_precision == "fp16":
                    scaler.unscale_(optimizer)
                
                # Gradient clipping
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                
                if args.mixed_precision == "fp16":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                lr_scheduler.step()
                optimizer.zero_grad()
           
                if rank == 0:
                    if global_step % 10 == 0:  # Log every 10 steps
                        logger.info(f"Step {global_step}: Loss = {loss.item() * args.gradient_accumulation_steps:.4f}")
                    progress_bar.update(1)
           
                global_step += 1
            
                # Checkpointing
                if global_step % args.checkpointing_steps == 0 and global_step > 0:
                    save_checkpoint_fsdp2(transformer, optimizer, lr_scheduler, global_step, epoch, args, best_val_loss)
                    
                    # Manage checkpoint limits
                    if rank == 0 and args.checkpoints_total_limit is not None:
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
                        transformer=transformer,
                        vae=vae,
                        noise_scheduler=noise_scheduler,
                        val_dataloader=val_dataloader,
                        args=args,
                        weight_dtype=weight_dtype,
                        device=device
                    )
                   
                    if rank == 0:
                        logger.info(f"Step {global_step}: Validation Loss = {val_loss:.4f}")
                        
                        # Save best model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_checkpoint_fsdp2(transformer, optimizer, lr_scheduler, global_step, epoch, args, best_val_loss)
                            # Rename to best model
                            best_path = os.path.join(args.output_dir, "best_model")
                            if os.path.exists(best_path):
                                shutil.rmtree(best_path)
                            shutil.copytree(os.path.join(args.output_dir, f"checkpoint-{global_step}"), best_path)

                    transformer.train()
                
                if global_step >= args.max_train_steps:
                    break
        
        if global_step >= args.max_train_steps:
            break

    # Final save
    if rank == 0:
        logger.info("Finalizing training...")
        progress_bar.close()

    # Save final model
    save_checkpoint_fsdp2(transformer, optimizer, lr_scheduler, global_step, epoch, args, best_val_loss)
    
    # Rename to final model
    if rank == 0:
        final_path = os.path.join(args.output_dir, "final_model")
        if os.path.exists(final_path):
            shutil.rmtree(final_path)
        shutil.copytree(os.path.join(args.output_dir, f"checkpoint-{global_step}"), final_path)
        logger.info("Training completed. Final model saved.")

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
    if dist.is_initialized():
        dist.destroy_process_group()

    torch.xpu.empty_cache()


if __name__ == "__main__":
   main()