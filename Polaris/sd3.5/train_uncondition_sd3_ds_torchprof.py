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
import numpy as np
import torch
import torch.distributed as dist
import transformers
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from huggingface_hub import create_repo, upload_folder
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers.modeling_outputs import BaseModelOutputWithPooling
import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler.profiler import ProfilerAction
import torch.cuda.nvtx as nvtx
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb

check_min_version("0.34.0.dev0")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Unconditional SD3 Training Script with DeepSpeed")
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
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Training batch size"
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10,
        help="Total training steps",
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="LR warmup steps"
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
        default=10,
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
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=["AdamW", "FusedAdam", "DeepSpeedCPUAdam"],
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="Adam beta1"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="Adam beta2"
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Adam weight decay")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Adam epsilon",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to Hub")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Mixed precision",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    # DeepSpeed specific arguments
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default=None,
        help="Path to DeepSpeed config file",
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="DeepSpeed ZeRO stage",
    )
    parser.add_argument(
        "--offload_optimizer",
        action="store_true",
        help="Offload optimizer states to CPU",
    )
    parser.add_argument(
        "--offload_param",
        action="store_true",
        help="Offload parameters to CPU",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    return args

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def create_deepspeed_config(args):
    """Create DeepSpeed configuration"""
    if args.deepspeed_config and os.path.exists(args.deepspeed_config):
        with open(args.deepspeed_config, 'r') as f:
            return json.load(f)
    
    # Default DeepSpeed configuration
    config = {
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clipping": args.max_grad_norm,
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": args.zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "optimizer": {
            "type": args.optimizer,
            "params": {
                "lr": args.learning_rate,
                "betas": [args.adam_beta1, args.adam_beta2],
                "eps": args.adam_epsilon,
                "weight_decay": args.adam_weight_decay
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": args.lr_warmup_steps
            }
        }
    }
    
    # Add CPU offloading if requested
    if args.offload_optimizer:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }
    
    if args.offload_param:
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True
        }
    
    # Add mixed precision settings
    if args.mixed_precision == "fp16":
        config["fp16"] = {
            "enabled": True,
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    elif args.mixed_precision == "bf16":
        config["bf16"] = {"enabled": True}
    
    return config

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

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def log_validation(transformer, args, rank, epoch, vae, noise_scheduler):
    """Generate validation images"""
    if rank != 0:  # Only run on main process
        return []
        
    logger.info("Running unconditional validation...")
    original_rng_state = torch.get_rng_state()
    original_cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    try:
        device = torch.cuda.current_device()
        dtype = torch.float32
        if hasattr(args, 'mixed_precision'):
            if args.mixed_precision == "bf16":
                dtype = torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float32
            elif args.mixed_precision == "fp16":
                dtype = torch.float16
        
        with torch.no_grad():
            seed = args.seed + epoch if args.seed is not None else None
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            generator = torch.Generator(device=device)
            if seed is not None:
                generator.manual_seed(seed)

            # Create dummy components for unconditional generation
            class DummyBatchEncoding:
                def __init__(self, input_ids, attention_mask):
                    self.input_ids = input_ids
                    self.attention_mask = attention_mask
                    
                def to(self, device):
                    self.input_ids = self.input_ids.to(device)
                    self.attention_mask = self.attention_mask.to(device)
                    return self

            class DummyTokenizer:
                def __init__(self):
                    self.model_max_length = 77
                    
                def __call__(self, text=None, return_tensors="pt", padding=True, truncation=True, **kwargs):
                    batch_size = 1 if text is None else len(text) if isinstance(text, list) else 1
                    return DummyBatchEncoding(
                        input_ids=torch.zeros((batch_size, self.model_max_length), dtype=torch.long),
                        attention_mask=torch.ones((batch_size, self.model_max_length), dtype=torch.long)
                    )

            class DummyTextEncoderOutput(BaseModelOutputWithPooling):
                pass

            class DummyTextEncoder:
                def __init__(self):
                    self.config = type('', (), {})()
                    self.config.hidden_size = 4096
                    self.dtype = dtype
                    self.device = device
                    
                def __call__(self, input_ids=None, attention_mask=None, **kwargs):
                    if isinstance(input_ids, DummyBatchEncoding):
                        batch_size = input_ids.input_ids.shape[0]
                        seq_len = input_ids.input_ids.shape[1]
                    else:
                        batch_size = input_ids.shape[0] if input_ids is not None else 1
                        seq_len = input_ids.shape[1] if input_ids is not None else 77

                    last_hidden_state = torch.zeros(
                        (batch_size, seq_len, self.config.hidden_size),
                        dtype=self.dtype,
                        device=self.device
                    )

                    hidden_state = [
                        torch.zeros(
                            (batch_size, seq_len, self.config.hidden_size),
                            dtype=self.dtype,
                            device=self.device
                        ) for _ in range(3)
                    ]

                    pooled_output = torch.zeros(
                        (batch_size, 2048),
                        dtype=self.dtype,
                        device=self.device
                    )
                    return DummyTextEncoderOutput(
                        last_hidden_state=last_hidden_state,
                        hidden_states=hidden_state,
                        pooler_output=pooled_output
                    )
            
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
                    with torch.autocast(device_type='cuda', dtype=dtype):
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
        if original_cuda_rng_state is not None:
            torch.cuda.set_rng_state(original_cuda_rng_state)
        if 'pipeline' in locals():
            del pipeline
        free_memory()
        torch.cuda.empty_cache()

def run_validation_dummy():
    torch.cuda.synchronize()
    return 0.0

def run_validation(model_engine, vae, noise_scheduler, val_dataloader, args, weight_dtype):
    """Run validation loop"""
    logger.info("Running validation...")
    model_engine.eval()
    total_val_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for step, val_batch in enumerate(val_dataloader):
            pixel_values = val_batch["pixel_values"].to(device=torch.cuda.current_device(), dtype=weight_dtype)
            with torch.autocast(device_type='cuda', dtype=weight_dtype):
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
                    device=torch.cuda.current_device(), 
                    dtype=weight_dtype
                )
                dummy_encoder = torch.zeros(
                    batch_size, 77, 4096, 
                    device=torch.cuda.current_device(), 
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
    return avg_val_loss

def main():
    args = parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if rank == 0 else logging.WARNING,
    )
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create output directory
    if rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(Path(args.output_dir).name, exist_ok=True).repo_id

    # Load models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", variant=args.variant
    )
    
    # Set weight dtype
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="vae", 
        variant=args.variant,
        torch_dtype=weight_dtype  
    )
    vae.requires_grad_(False)
    vae.eval()
    vae.to(torch.cuda.current_device(), dtype=weight_dtype)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
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

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )
    
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        worker_init_fn=worker_init_fn
        # pin_memory=True
    )

    # Create DeepSpeed config
    ds_config = create_deepspeed_config(args)
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=transformer,
        config=ds_config,
        model_parameters=transformer.parameters()
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # Initialize tracking
    if rank == 0 and is_wandb_available():
        wandb.init(project="sd3-unconditional", config=vars(args))
    
    # Training loop
    
    best_val_loss = float("inf")
    
    if args.resume_from_checkpoint:
        if os.path.isfile(args.resume_from_checkpoint):
            checkpoint_path = args.resume_from_checkpoint
        else:
            checkpoint_path = os.path.join(args.resume_from_checkpoint, "checkpoint")
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        model_engine.load_checkpoint(checkpoint_path)

    if rank == 0:
        progress_bar = tqdm(
            total=args.max_train_steps,
            desc="Training progress",
            dynamic_ncols=True
        )


    profiler_output_dir= '/grand/hp-ptycho/binkma/profiler/sd3.5m'


    global_step = 0

    for epoch in range(args.num_train_epochs):
        model_engine.train()
        train_sampler.set_epoch(epoch)

        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=3, warmup=1, active=2, repeat=1),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_output_dir, worker_name=f"rank_{rank}"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            warmup_iters = 1
            for step, batch in enumerate(train_dataloader):
                with record_function("training_step"):
                    # if step == warmup_iters: torch.cuda.cudart().cudaProfilerStart()
                    if step >= warmup_iters: torch.cuda.nvtx.range_push("iteration{}".format(step))

                    with torch.autocast(device_type='cuda', dtype=weight_dtype):
                        
                        with record_function("data_loading"):
                            if step >= warmup_iters: torch.cuda.nvtx.range_push("data_loading")
                            pixel_values = batch["pixel_values"].to(device=torch.cuda.current_device(), dtype=weight_dtype)
                            if step >= warmup_iters: torch.cuda.nvtx.range_pop()

                        # Encode images to latents
                        with record_function("vae_encoding"):
                            # with torch.no_grad():
                            if step >= warmup_iters: torch.cuda.nvtx.range_push("vae_encoding")
                            latents = vae.encode(pixel_values).latent_dist.sample()
                            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                            if step >= warmup_iters: torch.cuda.nvtx.range_pop()
                        
                        with record_function("noise_preparation"):
                            if step >= warmup_iters: torch.cuda.nvtx.range_push("noise_preparation")
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
                            if step >= warmup_iters: torch.cuda.nvtx.range_pop()
                        
                        with record_function("dummy_inputs_creation"):
                            if step >= warmup_iters: torch.cuda.nvtx.range_push("dummy_inputs_creation")
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
                            if step >= warmup_iters: torch.cuda.nvtx.range_pop()
                            
                        # Model prediction (unconditional)
                        with record_function("model_forward"):
                            if step >= warmup_iters: torch.cuda.nvtx.range_push("model_forward")
                            model_pred = model_engine(
                                hidden_states=noisy_latents,
                                timestep=timesteps,
                                encoder_hidden_states=dummy_encoder,
                                pooled_projections=dummy_pooled,
                                return_dict=False
                            )[0]
                            if step >= warmup_iters: torch.cuda.nvtx.range_pop()
                        
                        with record_function("loss_computation"):
                            if step >= warmup_iters: torch.cuda.nvtx.range_push("loss_computation")
                            if args.precondition_outputs:
                                model_pred = model_pred * (-sigmas) + noisy_latents
                            
                            # Calculate loss
                            weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas)
                            target = noise - latents if not args.precondition_outputs else latents
                            loss = torch.mean((weighting * (model_pred - target) ** 2).mean())
                            if step >= warmup_iters: torch.cuda.nvtx.range_pop()
                        
                        # Backward pass with DeepSpeed
                    with record_function("backward_pass"):
                        if step >= warmup_iters: torch.cuda.nvtx.range_push("backward_pass")
                        model_engine.backward(loss)
                        if step >= warmup_iters: torch.cuda.nvtx.range_pop()
                    
                    with record_function("optimizer_step"):
                        if step >= warmup_iters: torch.cuda.nvtx.range_push("optimizer_step")
                        model_engine.step()
                        if step >= warmup_iters: torch.cuda.nvtx.range_pop()
                    prof.step()

                    if step >= warmup_iters: torch.cuda.nvtx.range_pop()
                    if rank == 0:
                        logger.info(f"Step {global_step}: Loss = {loss.item()}")
                        progress_bar.update(1)

                    global_step += 1
                    if global_step > 20:
                        # torch.cuda.cudart().cudaProfilerStop()
                        break
                        # Checkpointing
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        model_engine.save_checkpoint(save_path)
                        logger.info(f"Saved checkpoint {global_step}")
                        
                        # Validation
                    if global_step % args.validation_steps == 0:
                        val_sampler.set_epoch(epoch)
                        # if deepspeed.comm.get_rank() == 0:
                        val_loss = run_validation(
                            model_engine=model_engine,
                            vae=vae,
                            noise_scheduler=noise_scheduler,
                            val_dataloader=val_dataloader,
                            args=args,
                            weight_dtype=weight_dtype,
                        )
                        logger.info(f"Step {global_step}: Validation Loss = {val_loss:.4f}")
                            
                        # if val_loss < best_val_loss:
                        # best_val_loss = val_loss
                        save_path = os.path.join(args.output_dir, "best_model")
                        model_engine.save_checkpoint(save_path)
                        logger.info(f"Saved best model with val loss {val_loss:.4f}")
                            # if deepspeed.comm.get_rank() == 0:
                            #     from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict,get_fp32_state_dict_from_zero_checkpoint
                            #     state_dict = get_fp32_state_dict_from_zero_checkpoint(args.output_dir)
                            #     torch.save(state_dict, f"{args.output_dir}/pytorch_model.bin")
                        
                # model_engine.train()
            if global_step >= args.max_train_steps:
                break
        # torch.cuda.cudart().cudaProfilerStop()
        prof.export_chrome_trace(os.path.join(profiler_output_dir, f"train_rank_{rank}.json"))
    # torch.cuda.cudart().cudaProfilerStop()
    # Final save
    if rank == 0:
        logger.info("Finalizing training...")
        progress_bar.close()
    final_save_path = os.path.join(args.output_dir, "final_model")
    model_engine.save_checkpoint(final_save_path)
    logger.info("Training completed. Final model saved.")
        
    # Push to hub if requested
    # if args.push_to_hub:
    #     upload_folder(
    #         repo_id=repo_id,
    #         folder_path=args.output_dir,
    #         commit_message="End of training",
    #         ignore_patterns=["step_*", "epoch_*"],
    #     )
    
    

if __name__ == "__main__":
    main()