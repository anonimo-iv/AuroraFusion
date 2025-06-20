#!/usr/bin/env python
import argparse
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers.modeling_outputs import BaseModelOutputWithPooling
from torch.utils.data.distributed import DistributedSampler
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
logger = get_logger(__name__)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Unconditional SD3 Training Script")
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
        choices=["AdamW", "prodigy"],
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
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

class UnconditionalImageDataset(Dataset):
    def __init__(
        self,
        data_root,
        size=512,
        center_crop=False,
        is_validation=False
    ):
        self.size = size
        self.center_crop = center_crop
        self.is_validation = is_validation
        
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
        
        elif args.dataset_name is not None:
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
        #
        if is_validation:
            self.transform = transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

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
            # 返回虚拟图像
            dummy_image = torch.zeros(3, self.size, self.size)
            return {
                "pixel_values": dummy_image
            }

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    
    return {
        "pixel_values": pixel_values
    }

def load_text_encoders(class_one, class_two, class_three):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three

def log_validation(transformer, args, accelerator, epoch, vae, noise_scheduler):
    logger.info("Running unconditional validation...")
    # 
    original_rng_state = torch.get_rng_state()
    original_cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    try:
        # 
        device = accelerator.device
        dtype = torch.float32
        if hasattr(args, 'mixed_precision'):
            if args.mixed_precision == "bf16":
                dtype = torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float32
            elif args.mixed_precision == "fp16":
                dtype = torch.float16
        # 
        with torch.no_grad(), torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
            seed = args.seed + epoch if args.seed is not None else None
            if seed is not None:
                torch.manual_seed(seed)
                if device.type == 'cuda':
                    torch.cuda.manual_seed_all(seed)

            # 
            generator = torch.Generator(device=device)
            if seed is not None:
                generator.manual_seed(seed)

            # 
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
            # 
            dummy_tokenizer = DummyTokenizer()
            dummy_text_encoder = DummyTextEncoder()
            pipeline = StableDiffusion3Pipeline(
                vae=vae,
                transformer=accelerator.unwrap_model(transformer),
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
            # 
            images = []
            for i in range(args.num_validation_images):
                try:
                    with torch.autocast(device.type, dtype=dtype):
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

def run_validation(transformer, vae, noise_scheduler, val_dataloader, accelerator, args, weight_dtype):
    logger.info("Running validation...")
    transformer.eval()
    total_val_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for step,val_batch in enumerate(val_dataloader):
            pixel_values = val_batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
            # Accelerator autocast
            with accelerator.autocast():
                # noisy_latents = (1 - sigmas) * latents + sigmas * noise
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                noise = torch.randn_like(latents).to(dtype=latents.dtype, device=latents.device)
                bsz = latents.shape[0]
                # timestep
                u = compute_density_for_timestep_sampling(
                    args.weighting_scheme,
                    bsz,
                    args.logit_mean,
                    args.logit_std,
                    args.mode_scale
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(latents.device, dtype=weight_dtype)
                # add noise
                sigmas = noise_scheduler.sigmas[indices].view(-1, 1, 1, 1).to(latents.device)
                noise = noise.to(latents.device)
                noisy_latents = (1 - sigmas) * latents + sigmas * noise
                    
                batch_size = noisy_latents.shape[0]
                dummy_pooled = torch.zeros(
                    batch_size, 2048, 
                    device=accelerator.device, 
                    dtype=weight_dtype
                )
                dummy_encoder = torch.zeros(
                    batch_size, 77, 4096, 
                    device=accelerator.device, 
                    dtype=weight_dtype
                )
                # model prediction
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
    # 
    avg_val_loss = total_val_loss / max(num_batches, 1) 
    return avg_val_loss


def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    if args.seed is not None:
        set_seed(args.seed) 
        accelerator.set_seed(args.seed)  # Use accelerator's set_seed

    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(Path(args.output_dir).name, exist_ok=True).repo_id

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", variant=args.variant
    )
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

        vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="vae", 
        variant=args.variant,
        torch_dtype=weight_dtype  
    )
    vae.requires_grad_(False)

    
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    
    # prepare VAE and noise scheduler
    train_dataset = UnconditionalImageDataset(
        args.train_data_dir,
        size=args.resolution,
        center_crop=args.center_crop
    )
    if args.val_data_dir:  # if validation data directory is provided
        val_dataset = UnconditionalImageDataset(
            args.val_data_dir,
            size=args.resolution,
            center_crop=args.center_crop,
            is_validation=True
        )
    else:  # split training dataset into train and validation
        total_size = len(train_dataset)
        val_size = int(total_size * 0.1)  # 10% 
        indices = list(range(total_size))
        random.shuffle(indices)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        from torch.utils.data import Subset
        val_dataset = Subset(train_dataset, val_indices)
        train_dataset = Subset(train_dataset, train_indices)
        
        logger.info(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation")

    

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        set_seed(worker_seed)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        # num_workers=args.dataloader_num_workers,
        num_workers=0,  #
        worker_init_fn=worker_init_fn
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        worker_init_fn=worker_init_fn
    )

    params_to_optimize = [
        {"params": transformer.parameters(), "lr": args.learning_rate},
    ]
    
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # prepare accelerator
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    if accelerator.is_main_process:
        accelerator.init_trackers("sd3-unconditional", config=vars(args))
    
    step_progress = tqdm(
        total=args.max_train_steps,
        desc="Training progress",
        dynamic_ncols=True
    )

    # training loop
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                with torch.autocast(device_type=accelerator.device.type, dtype=weight_dtype):
                    # print(f"Processing step {step}")
                    pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)

                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                    noise = torch.randn_like(latents).to(dtype=latents.dtype, device=latents.device)
                    bsz = latents.shape[0]
                    # timestep
                    u = compute_density_for_timestep_sampling(
                        args.weighting_scheme,
                        bsz,
                        args.logit_mean,
                        args.logit_std,
                        args.mode_scale
                    )
                    indices = (u * noise_scheduler.config.num_train_timesteps).long()
                    timesteps = noise_scheduler.timesteps[indices].to(latents.device, dtype=weight_dtype)
                    # add noise
                    sigmas = noise_scheduler.sigmas[indices].view(-1, 1, 1, 1).to(latents.device)
                    noise = noise.to(latents.device)
                    noisy_latents = (1 - sigmas) * latents + sigmas * noise
                    
                    batch_size = noisy_latents.shape[0]
                    dummy_pooled = torch.zeros(
                        batch_size,
                        2048,  # SD3 typically uses 2048 for pooled projections
                        device=noisy_latents.device,
                        dtype=weight_dtype
                    )
                    dummy_encoder = torch.zeros(
                        batch_size,
                        77,  # Sequence length
                        4096,  # Hidden dimension
                        device=noisy_latents.device,
                        dtype=weight_dtype
                    )
                    # Model prediction (unconditional)
                    model_pred = transformer(
                        hidden_states=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=dummy_encoder,  # Critical for unconditional
                        pooled_projections=dummy_pooled,
                        return_dict=False
                    )[0]  # Gets the prediction tensor
                    
                    if args.precondition_outputs:
                        model_pred = model_pred * (-sigmas) + noisy_latents
                    
                    # 
                    weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas)
                    # calculate target
                    target = noise - latents if not args.precondition_outputs else latents
                    
                    # calculate loss
                    loss = torch.mean((weighting * (model_pred - target) ** 2).mean())
                    logger.info(f"Step {global_step}: Loss = {loss.item()}")
                    # backward pass
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                global_step += 1
                step_progress.update(1)
                
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved checkpoint {global_step}")
                if global_step % args.validation_steps == 0:
                    if accelerator.is_main_process:
                        val_loss = run_validation(
                            transformer=accelerator.unwrap_model(transformer),
                            vae=vae,
                            noise_scheduler=noise_scheduler,
                            val_dataloader=val_dataloader,
                            accelerator=accelerator,
                            args=args,
                            weight_dtype=weight_dtype
                        )
                        accelerator.log({"val_loss": val_loss}, step=global_step)
                        logger.info(f"Step {global_step}: Validation Loss = {val_loss:.4f}")

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_path = os.path.join(args.output_dir, "best_model")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved best model with val loss {val_loss:.4f}")

            if global_step >= args.max_train_steps:
                break
        
    if args.push_to_hub:
        upload_folder(
            repo_id=repo_id,
            folder_path=args.output_dir,
            commit_message="End of training",
            ignore_patterns=["step_*", "epoch_*"],
        )
    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)