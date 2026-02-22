"""
Text-to-Image training script using flow matching.
Based on projects/rae/train.py with text conditioning support.
"""

import os
import sys
sys.path.append(".")

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import math
import argparse
import logging
import json
from pathlib import Path
from copy import deepcopy
from glob import glob
from time import time
from collections import defaultdict
from PIL import Image
from typing import Optional
import io
import traceback

# WebDataset for large-scale training
try:
    import webdataset as wds
    HAS_WEBDATASET = True
except ImportError:
    HAS_WEBDATASET = False
    print("Warning: webdataset not installed. Install with: pip install webdataset")

# Import utilities
from models.rae.utils.ddp_utils import setup_ddp, cleanup_ddp as cleanup, create_logger, requires_grad, update_ema
from models.rae.utils.image_utils import center_crop_arr
from models.rae.utils.vae_utils import (
    load_vae,
    reconstruct_from_latent_with_diffusion,
    load_latent_stats,
    LatentNormalizer,
    get_normalize_fn,
)
from models.rae.utils.argparse_utils import get_encoder_config

# Import model
from model import DiTwDDTHead_T2I, DiT_T2I_XXL_2, DiT_T2I_L_2, DiT_T2I_B_2


#################################################################################
#                              Text Encoder                                     #
#################################################################################

class TextEncoder:
    """
    Text encoder wrapper using Qwen3 or other models.
    The text encoder is frozen during training.
    """
    def __init__(
        self, 
        model_path: str,
        max_length: int = 128,
        device: torch.device = None,
    ):
        from transformers import AutoModel, AutoTokenizer
        
        self.device = device or torch.device("cuda")
        self.max_length = max_length
        
        # Load tokenizer and model
        print(f"Loading text encoder from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        # Freeze the text encoder
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get hidden size
        self.hidden_size = self.model.config.hidden_size
        print(f"Text encoder loaded. Hidden size: {self.hidden_size}")
        
        # Cache for unconditional embedding
        self._uncond_embedding = None
    
    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """
        Encode a list of text prompts.
        
        Args:
            texts: List of text strings
        Returns:
            Text embeddings [B, max_length, hidden_size]
        """
        tokenized = self.tokenizer(
            texts, 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length", 
            return_tensors="pt"
        )
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get last hidden state
        hidden_states = outputs.last_hidden_state  # [B, L, D]
        return hidden_states.float()
    
    @torch.no_grad()
    def get_uncond_embedding(self, batch_size: int = 1) -> torch.Tensor:
        """Get unconditional embedding (empty string)."""
        if self._uncond_embedding is None:
            self._uncond_embedding = self.encode([""])
        return self._uncond_embedding.repeat(batch_size, 1, 1)


#################################################################################
#                           Text-Image Dataset                                  #
#################################################################################

class TextImageDataset(Dataset):
    """
    Dataset for text-image pairs.
    
    Expected directory structure:
    - data_path/
        - images/
            - 000000.jpg
            - 000001.jpg
            - ...
        - metadata.jsonl (each line: {"image": "000000.jpg", "caption": "..."})
    
    Or ImageFolder style with captions in separate files:
    - data_path/
        - class_name/
            - image.jpg
            - image.txt (containing caption)
    """
    def __init__(
        self,
        data_path: str,
        image_size: int = 256,
        caption_key: str = "caption",
        image_key: str = "image",
    ):
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.caption_key = caption_key
        self.image_key = image_key
        
        # Setup transform
        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t * 2.0 - 1.0),  # [-1, 1]
        ])
        
        # Load metadata
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def _load_samples(self) -> list:
        samples = []
        
        # Check for metadata.jsonl
        metadata_file = self.data_path / "metadata.jsonl"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                for line in f:
                    item = json.loads(line.strip())
                    image_path = self.data_path / "images" / item[self.image_key]
                    if image_path.exists():
                        samples.append({
                            "image_path": str(image_path),
                            "caption": item[self.caption_key],
                        })
            return samples
        
        # Fallback: ImageFolder style with .txt files
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            for img_path in self.data_path.rglob(ext):
                txt_path = img_path.with_suffix(".txt")
                if txt_path.exists():
                    with open(txt_path, "r") as f:
                        caption = f.read().strip()
                    samples.append({
                        "image_path": str(img_path),
                        "caption": caption,
                    })
        
        # Fallback: Just images with empty captions (for unconditional pretraining)
        if len(samples) == 0:
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                for img_path in self.data_path.rglob(ext):
                    samples.append({
                        "image_path": str(img_path),
                        "caption": "",  # Empty caption
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)
        
        caption = sample["caption"]
        
        return image, caption


class ImageNetWithCaptions(Dataset):
    """
    ImageNet dataset with class-based captions.
    Uses class names as captions for simplicity.
    """
    def __init__(
        self,
        data_path: str,
        image_size: int = 256,
        class_to_caption_file: Optional[str] = None,
    ):
        from torchvision.datasets import ImageFolder
        
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t * 2.0 - 1.0),
        ])
        
        self.dataset = ImageFolder(data_path, transform=self.transform)
        
        # Load class to caption mapping
        if class_to_caption_file and os.path.exists(class_to_caption_file):
            with open(class_to_caption_file, "r") as f:
                self.class_to_caption = json.load(f)
        else:
            # Use folder names as captions
            self.class_to_caption = {
                idx: name.replace("_", " ") 
                for name, idx in self.dataset.class_to_idx.items()
            }
        
        print(f"Loaded ImageNet with {len(self.dataset)} images")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, class_idx = self.dataset[idx]
        caption = self.class_to_caption.get(class_idx, f"class {class_idx}")
        # Create a simple prompt
        caption = f"a photo of {caption}"
        return image, caption


#################################################################################
#                            WebDataset Support                                 #
#################################################################################

def create_webdataset(
    tar_paths: list[str],
    image_size: int = 256,
    batch_size: int = 32,
    num_workers: int = 4,
    image_key: str = "jpg",
    caption_key: str = "txt",
    shuffle_buffer: int = 5000,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 0,
    epoch_length: int = None,
):
    """
    Create a WebDataset dataloader from tar files.
    
    Args:
        tar_paths: List of tar file paths, can be:
            - Single tar file: ["/path/to/data.tar"]
            - Multiple tar files: ["/path/to/data-{000..099}.tar"]
            - Glob pattern: ["/path/to/data/*.tar"]
        image_size: Target image size
        batch_size: Batch size per GPU
        num_workers: Number of workers
        image_key: Key for image in tar (e.g., "jpg", "png", "webp")
        caption_key: Key for caption in tar (e.g., "txt", "json", "caption")
        shuffle_buffer: Size of shuffle buffer
        world_size: Total number of GPUs
        rank: Current GPU rank
        seed: Random seed
        epoch_length: Number of samples per epoch (for infinite datasets)
    
    Returns:
        WebDataset dataloader
    
    Expected tar structure:
        sample_0001.jpg
        sample_0001.txt (or .json with "caption" field)
        sample_0002.jpg
        sample_0002.txt
        ...
    """
    if not HAS_WEBDATASET:
        raise ImportError("webdataset is required. Install with: pip install webdataset")
    
    # Expand tar paths
    expanded_paths = []
    for path in tar_paths:
        if '*' in path:
            # Glob pattern
            expanded = sorted(glob(path))
            expanded_paths.extend(expanded)
        elif '{' in path and '..' in path:
            # Brace expansion pattern like data-{000..099}.tar
            expanded_paths.append(path)
        else:
            expanded_paths.append(path)
    
    if not expanded_paths:
        raise ValueError(f"No tar files found for paths: {tar_paths}")
    
    print(f"[Rank {rank}] WebDataset: Found {len(expanded_paths)} tar file(s)")
    
    # Image transform
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),
    ])
    
    def preprocess_sample(sample):
        """Preprocess a single sample from webdataset."""
        # Get image
        img_data = None
        for key in ["jpg", "jpeg", "png", "webp", image_key]:
            if key in sample:
                img_data = sample[key]
                break
        
        if img_data is None:
            raise ValueError(f"No image found in sample. Keys: {list(sample.keys())}")
        
        # Decode image
        if isinstance(img_data, bytes):
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
        elif isinstance(img_data, Image.Image):
            img = img_data.convert("RGB")
        else:
            img = img_data
        
        # Apply transform
        img = transform(img)
        
        # Get caption
        caption = ""
        if caption_key in sample:
            caption_data = sample[caption_key]
            if isinstance(caption_data, bytes):
                caption = caption_data.decode("utf-8").strip()
            elif isinstance(caption_data, str):
                caption = caption_data.strip()
            elif isinstance(caption_data, dict):
                caption = caption_data.get("caption", caption_data.get("text", ""))
        elif "json" in sample:
            # Try parsing JSON for caption
            try:
                json_data = sample["json"]
                if isinstance(json_data, bytes):
                    json_data = json.loads(json_data.decode("utf-8"))
                elif isinstance(json_data, str):
                    json_data = json.loads(json_data)
                caption = json_data.get("caption", json_data.get("text", ""))
            except:
                pass
        
        return img, caption
    
    # Build dataset pipeline
    if len(expanded_paths) == 1 and '{' not in expanded_paths[0]:
        urls = expanded_paths[0]
    else:
        urls = expanded_paths
    
    dataset = (
        wds.WebDataset(
            urls,
            resampled=True,  # Enable infinite resampling
            shardshuffle=True,
            nodesplitter=wds.split_by_node,  # Automatic node splitting
        )
        .shuffle(shuffle_buffer)
        .decode("pil")
        .map(preprocess_sample)
    )
    
    # Set epoch length if specified
    if epoch_length:
        dataset = dataset.with_epoch(epoch_length // world_size)
    
    # Create dataloader
    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Add batching features
    loader = loader.unbatched().shuffle(1000).batched(batch_size)
    
    return loader


def get_webdataset_loader(
    tar_paths: list[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    world_size: int,
    rank: int,
    seed: int = 0,
    image_key: str = "jpg",
    caption_key: str = "txt",
    shuffle_buffer: int = 5000,
    epoch_length: int = None,
):
    """
    Convenience function to create WebDataset loader.
    
    Args:
        tar_paths: Can be:
            - String with path to single tar or pattern
            - List of tar file paths
            - Path to directory containing tar files
    """
    # Handle different input types
    if isinstance(tar_paths, str):
        if os.path.isdir(tar_paths):
            # Directory of tar files
            tar_paths = sorted(glob(os.path.join(tar_paths, "*.tar")))
        else:
            tar_paths = [tar_paths]
    
    return create_webdataset(
        tar_paths=tar_paths,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        image_key=image_key,
        caption_key=caption_key,
        shuffle_buffer=shuffle_buffer,
        world_size=world_size,
        rank=rank,
        seed=seed,
        epoch_length=epoch_length,
    )


#################################################################################
#                              Training Utils                                   #
#################################################################################

def sample_timesteps(
    batch_size: int,
    device: torch.device,
    time_shift: float = 1.0,
    noise_schedule: str = "uniform",
    log_norm_mean: float = 0.0,
    log_norm_std: float = 1.0,
):
    """Sample timesteps for flow matching training."""
    if noise_schedule == "uniform":
        t = torch.rand(batch_size, device=device)
    elif noise_schedule == "log_norm":
        u = torch.randn(batch_size, device=device) * log_norm_std + log_norm_mean
        t = torch.sigmoid(u)
    else:
        raise ValueError(f"Unknown noise_schedule: {noise_schedule}")
    
    # Apply time shift
    t = (time_shift * t) / (1.0 + (time_shift - 1.0) * t)
    t = t.clamp(1e-5, 1.0 - 1e-5)
    return t


def compute_train_loss(
    model,
    x_latent: torch.Tensor,
    y_embed: torch.Tensor,
    time_shift: float = 1.0,
    noise_schedule: str = "uniform",
    log_norm_mean: float = 0.0,
    log_norm_std: float = 1.0,
):
    """Compute flow matching training loss."""
    B = x_latent.size(0)
    device = x_latent.device
    
    noise = torch.randn_like(x_latent)
    
    t = sample_timesteps(
        batch_size=B,
        device=device,
        time_shift=time_shift,
        noise_schedule=noise_schedule,
        log_norm_mean=log_norm_mean,
        log_norm_std=log_norm_std,
    )
    
    t_broadcast = t.view(B, 1, 1, 1)
    
    # Linear interpolation: x_t = t * noise + (1 - t) * x_0
    x_t = t_broadcast * noise + (1.0 - t_broadcast) * x_latent
    
    # Model predicts x_0
    model_output = model(x_t, t, y_embed)
    
    # MSE loss with reweighting
    loss = F.mse_loss(model_output, x_latent, reduction="none")
    
    # Reweight by 1/t^2 (clamped)
    reweight_scale = torch.clamp_max(1.0 / (t**2 + 1e-8), 10)
    loss = loss * reweight_scale.view(B, 1, 1, 1)
    loss = loss.mean()
    
    return loss, model_output, noise, t


@torch.no_grad()
def sample_latent_linear(
    model,
    text_embeddings: torch.Tensor,
    latent_shape: tuple,
    device: torch.device,
    time_shift: float,
    steps: int = 50,
    init_noise: Optional[torch.Tensor] = None,
    cfg_scale: float = 7.5,
    uncond_embeddings: Optional[torch.Tensor] = None,
    cfg_interval: tuple = (0.0, 1.0),
):
    """
    Sample latent using linear flow with x-prediction.
    
    Args:
        model: DiT model
        text_embeddings: Conditional text embeddings [B, L, D]
        latent_shape: (C, H, W)
        device: Device
        time_shift: Time shift factor
        steps: Number of sampling steps
        init_noise: Optional initial noise
        cfg_scale: CFG scale (set to 1.0 to disable)
        uncond_embeddings: Unconditional embeddings for CFG
        cfg_interval: Time interval for applying CFG
    """
    model_inner = model.module if hasattr(model, "module") else model
    model_inner.eval()
    
    batch_size = text_embeddings.shape[0]
    C, H, W = latent_shape
    
    if init_noise is None:
        x = torch.randn((batch_size, C, H, W), device=device, dtype=torch.float32)
    else:
        x = init_noise.to(device=device, dtype=torch.float32)
    
    shift = float(time_shift)
    use_cfg = cfg_scale > 1.0 and uncond_embeddings is not None
    
    def flow_shift(t_lin: torch.Tensor) -> torch.Tensor:
        t = (shift * t_lin) / (1.0 + (shift - 1.0) * t_lin)
        return t.clamp(0.0, 1.0 - 1e-6)
    
    for i in range(steps, 0, -1):
        t_lin = torch.full((batch_size,), i / steps, device=device, dtype=torch.float32)
        t_next_lin = torch.full((batch_size,), (i - 1) / steps, device=device, dtype=torch.float32)
        
        t = flow_shift(t_lin)
        t_next = flow_shift(t_next_lin)
        
        if use_cfg:
            t_scalar = t[0].item()
            cfg_t_min, cfg_t_max = cfg_interval
            
            if cfg_t_min <= t_scalar <= cfg_t_max:
                # Conditional prediction
                x0_cond = model_inner(x, t, text_embeddings)
                # Unconditional prediction
                x0_uncond = model_inner(x, t, uncond_embeddings)
                # CFG
                x0_hat = x0_uncond + cfg_scale * (x0_cond - x0_uncond)
            else:
                x0_hat = model_inner(x, t, text_embeddings)
        else:
            x0_hat = model_inner(x, t, text_embeddings)
        
        # Compute eps from x_t = t*eps + (1-t)*x0
        t_scalar = t.view(batch_size, 1, 1, 1)
        eps_hat = (x - (1.0 - t_scalar) * x0_hat) / (t_scalar + 1e-8)
        
        # Update to next time
        t_next_scalar = t_next.view(batch_size, 1, 1, 1)
        x = t_next_scalar * eps_hat + (1.0 - t_next_scalar) * x0_hat
    
    return x


#################################################################################
#                              Main Training                                    #
#################################################################################

def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires at least one GPU.")
    
    # DDP setup
    rank, local_rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank)
    
    # Batch size
    if args.global_batch_size % (world_size * args.grad_accum_steps) != 0:
        raise ValueError("Global batch size must be divisible by world_size * grad_accum_steps.")
    micro_batch_size = args.global_batch_size // (world_size * args.grad_accum_steps)
    
    # Seed
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}")
    
    # 获取 encoder 配置
    encoder_config = get_encoder_config(args.encoder_type)
    
    # Latent size: 优先使用命令行参数，否则使用 encoder 默认配置
    if args.latent_channels is not None:
        latent_channels = args.latent_channels
    else:
        latent_channels = encoder_config["latent_channels"]
    
    patch_size = encoder_config.get("patch_size", 16)
    latent_hw = args.image_size // patch_size
    latent_size = (latent_channels, latent_hw, latent_hw)
    
    # Time shift
    shift_dim = latent_channels * latent_hw * latent_hw
    time_shift = math.sqrt(shift_dim / args.time_shift_base)
    
    if rank == 0:
        print(f"Encoder type: {args.encoder_type}")
        print(f"Latent channels: {latent_channels} (from {'args' if args.latent_channels else 'encoder_config'})")
        print(f"Latent size: {latent_size}, time_shift: {time_shift:.4f}")
    
    # Setup directories
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_dir = os.path.join(args.results_dir, "experiment")
        os.makedirs(experiment_dir, exist_ok=True)
        checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger = create_logger(experiment_dir, rank=0)
        logger.info(f"Experiment directory: {experiment_dir}")
        
        tb_dir = os.path.join(experiment_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
    else:
        logger = create_logger(None, rank=rank)
        writer = None
        experiment_dir = None
        checkpoint_dir = None
    
    # Load VAE
    # 获取 decoder block channels (如果有)
    dec_block_out_channels = encoder_config.get("dec_block_out_channels", (768, 512, 256, 128, 64))
    
    vae_model_params = {
        "encoder_type": args.encoder_type,
        "image_size": args.image_size,
        "patch_size": patch_size,
        "out_channels": 3,
        "latent_channels": latent_channels,
        "target_latent_channels": None,
        "spatial_downsample_factor": patch_size,
        "lora_rank": 0 if args.no_lora else args.lora_rank,
        "lora_alpha": 0 if args.no_lora else args.lora_alpha,
        "decoder_dropout": 0.0,
        "gradient_checkpointing": False,
        "denormalize_decoder_output": args.denormalize_decoder_output,
        "skip_to_moments": args.skip_to_moments,
        "dec_block_out_channels": dec_block_out_channels,
        "dec_layers_per_block": 3,
    }
    
    if args.encoder_type == "dinov3" or args.encoder_type == "dinov3_vitl":
        vae_model_params["dinov3_model_dir"] = args.dinov3_dir
    elif args.encoder_type == "dinov2":
        vae_model_params["dinov2_model_name"] = args.dinov2_model_name
    elif args.encoder_type == "siglip2":
        vae_model_params["siglip2_model_name"] = encoder_config.get("default_model_name", "google/siglip2-base-patch16-256")
    
    vae = load_vae(
        args.vae_ckpt,
        device,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        model_params=vae_model_params,
        skip_to_moments=args.skip_to_moments,
        use_ema=args.vae_use_ema,
    )
    
    # Get normalization function
    if args.latent_stats_path:
        latent_stats = load_latent_stats(args.latent_stats_path, device=device, verbose=(rank == 0))
        normalize_fn = LatentNormalizer(latent_stats, per_channel=args.per_channel_norm)
    else:
        normalize_fn = get_normalize_fn(args.encoder_type)
    
    if rank == 0:
        logger.info(f"Loaded VAE: {args.vae_ckpt}")
    
    # Load text encoder
    text_encoder = TextEncoder(
        model_path=args.text_encoder_path,
        max_length=args.txt_max_length,
        device=device,
    )
    
    if rank == 0:
        logger.info(f"Loaded text encoder: {args.text_encoder_path}")
        logger.info(f"Text encoder hidden size: {text_encoder.hidden_size}")
    
    # Create model
    model_kwargs = {
        "input_size": latent_hw,
        "in_channels": latent_channels,
        "txt_embed_dim": text_encoder.hidden_size,
        "txt_max_length": args.txt_max_length,
        "num_text_refine_blocks": args.num_text_refine_blocks,
        "use_qknorm": args.use_qknorm,
        "use_swiglu": True,
        "use_rope": True,
        "use_rmsnorm": True,
        "wo_shift": False,
        "use_pos_embed": True,
        "cfg_dropout_prob": args.cfg_prob,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    
    if args.model_size == "XXL":
        model = DiT_T2I_XXL_2(**model_kwargs).to(device, dtype=torch.bfloat16)
    elif args.model_size == "L":
        model = DiT_T2I_L_2(**model_kwargs).to(device, dtype=torch.bfloat16)
    elif args.model_size == "B":
        model = DiT_T2I_B_2(**model_kwargs).to(device, dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
    
    # EMA 放在 CPU 上以节省 GPU 显存
    ema = deepcopy(model).to("cpu", dtype=torch.float32)
    requires_grad(ema, False)
    
    # Load checkpoint if provided (robust loading)
    train_steps = 0
    opt_state = None
    sched_state = None
    
    def try_load_checkpoint(ckpt_path, model, ema, rank, logger):
        """
        Attempt to load checkpoint with robust error handling.
        Returns: (success, train_steps, opt_state, sched_state)
        """
        if not ckpt_path:
            return False, 0, None, None
        
        # Check if file exists
        if not os.path.exists(ckpt_path):
            if rank == 0:
                logger.warning(f"Checkpoint not found: {ckpt_path}, training from scratch")
            return False, 0, None, None
        
        try:
            if rank == 0:
                logger.info(f"Attempting to load checkpoint: {ckpt_path}")
            
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            
            # Load model state
            model_loaded = False
            if "model" in checkpoint:
                try:
                    # 将 checkpoint 参数转换为 bf16 以匹配模型
                    model_state = {k: v.to(torch.bfloat16) if v.is_floating_point() else v 
                                   for k, v in checkpoint["model"].items()}
                    missing, unexpected = model.load_state_dict(model_state, strict=False)
                    model_loaded = True
                    if rank == 0:
                        if missing:
                            logger.warning(f"Missing keys in model: {len(missing)} keys")
                        if unexpected:
                            logger.warning(f"Unexpected keys in model: {len(unexpected)} keys")
                except Exception as e:
                    if rank == 0:
                        logger.warning(f"Failed to load model state: {e}")
            
            # Load EMA state (EMA 在 CPU 上，fp32)
            ema_loaded = False
            if "ema" in checkpoint:
                try:
                    # EMA 保持 fp32
                    ema_state = {k: v.to(torch.float32) if v.is_floating_point() else v 
                                 for k, v in checkpoint["ema"].items()}
                    missing, unexpected = ema.load_state_dict(ema_state, strict=False)
                    ema_loaded = True
                    if rank == 0:
                        if missing:
                            logger.warning(f"Missing keys in EMA: {len(missing)} keys")
                        if unexpected:
                            logger.warning(f"Unexpected keys in EMA: {len(unexpected)} keys")
                except Exception as e:
                    if rank == 0:
                        logger.warning(f"Failed to load EMA state: {e}")
                    # If EMA load fails, copy from model (转换为 fp32)
                    if model_loaded:
                        model_state_fp32 = {k: v.to(torch.float32).cpu() if v.is_floating_point() else v.cpu()
                                            for k, v in model.state_dict().items()}
                        ema.load_state_dict(model_state_fp32)
                        ema_loaded = True
                        if rank == 0:
                            logger.info("Initialized EMA from model state (converted to fp32 on CPU)")
            
            # Get optimizer state (optional)
            opt_state = checkpoint.get("opt", None)
            sched_state = checkpoint.get("scheduler", None)
            train_steps = int(checkpoint.get("train_steps", 0))
            
            # Validate train_steps
            if train_steps < 0:
                if rank == 0:
                    logger.warning(f"Invalid train_steps {train_steps}, resetting to 0")
                train_steps = 0
            
            if model_loaded or ema_loaded:
                if rank == 0:
                    logger.info(f"Successfully resumed from {ckpt_path}")
                    logger.info(f"  - Model loaded: {model_loaded}")
                    logger.info(f"  - EMA loaded: {ema_loaded}")
                    logger.info(f"  - Train steps: {train_steps}")
                    logger.info(f"  - Optimizer state: {'available' if opt_state else 'not available'}")
                return True, train_steps, opt_state, sched_state
            else:
                if rank == 0:
                    logger.warning(f"No model/EMA state found in checkpoint, training from scratch")
                return False, 0, None, None
                
        except Exception as e:
            if rank == 0:
                logger.warning(f"Failed to load checkpoint: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                logger.info("Training from scratch...")
            return False, 0, None, None
    
    # Try to resume from checkpoint
    ckpt_loaded, train_steps, opt_state, sched_state = try_load_checkpoint(
        args.ckpt, model, ema, rank, logger
    )
    
    # If no explicit checkpoint, try to find latest checkpoint in results_dir
    if not ckpt_loaded and args.auto_resume:
        latest_ckpt = os.path.join(args.results_dir, "checkpoints", "latest.pt")
        if os.path.exists(latest_ckpt):
            if rank == 0:
                logger.info(f"Auto-resuming from latest checkpoint: {latest_ckpt}")
            ckpt_loaded, train_steps, opt_state, sched_state = try_load_checkpoint(
                latest_ckpt, model, ema, rank, logger
            )
    
    if not ckpt_loaded and rank == 0:
        logger.info("Starting training from scratch")
    
    model_param_count = sum(p.numel() for p in model.parameters())
    if rank == 0:
        logger.info(f"Model parameters: {model_param_count / 1e6:.2f}M")
        logger.info(f"Model dtype: {next(model.parameters()).dtype}")
        logger.info(f"EMA device: {next(ema.parameters()).device}, dtype: {next(ema.parameters()).dtype}")
        logger.info(f"Gradient checkpointing: {args.gradient_checkpointing}")
    
    # torch.compile
    if args.compile:
        if rank == 0:
            logger.info(f"Compiling model with torch.compile (mode={args.compile_mode})...")
        model = torch.compile(model, mode=args.compile_mode)
    
    # DDP
    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=False)
    
    # Optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    if opt_state:
        opt.load_state_dict(opt_state)
    
    # Dataset
    use_webdataset = args.dataset_type == "webdataset"
    
    if use_webdataset:
        # WebDataset for large-scale tar-based datasets
        if not HAS_WEBDATASET:
            raise ImportError("webdataset is required for dataset_type='webdataset'. "
                            "Install with: pip install webdataset")
        
        loader = get_webdataset_loader(
            tar_paths=args.data_path,
            image_size=args.image_size,
            batch_size=micro_batch_size,
            num_workers=args.num_workers,
            world_size=world_size,
            rank=rank,
            seed=args.global_seed,
            image_key=args.wds_image_key,
            caption_key=args.wds_caption_key,
            shuffle_buffer=args.wds_shuffle_buffer,
            epoch_length=args.wds_epoch_length,
        )
        
        if rank == 0:
            logger.info(f"WebDataset: {args.data_path}")
            logger.info(f"Micro batch size: {micro_batch_size}, Global batch size: {args.global_batch_size}")
    else:
        # Standard PyTorch Dataset
        if args.dataset_type == "imagenet":
            dataset = ImageNetWithCaptions(
                args.data_path,
                image_size=args.image_size,
                class_to_caption_file=args.class_to_caption_file,
            )
        else:
            dataset = TextImageDataset(
                args.data_path,
                image_size=args.image_size,
            )
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.global_seed,
        )
        
        loader = DataLoader(
            dataset,
            batch_size=micro_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        if rank == 0:
            logger.info(f"Dataset: {len(dataset)} samples")
            logger.info(f"Micro batch size: {micro_batch_size}, Global batch size: {args.global_batch_size}")
    
    # Initialize EMA
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()
    
    # AMP
    use_amp = args.precision == "bf16"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = GradScaler(enabled=False)  # bf16 doesn't need scaler
    
    # Training loop
    log_steps = 0
    running_loss = 0.0
    start_time = time()
    step_times = []  # 记录每个 step 的时间
    
    if rank == 0:
        logger.info(f"Training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Set epoch for sampler (not needed for webdataset)
        if not use_webdataset:
            sampler.set_epoch(epoch)
        if rank == 0:
            print(f"Beginning epoch {epoch}...")
        
        opt.zero_grad()
        accum_counter = 0
        step_loss_accum = 0.0
        
        for x_img, captions in loader:
            step_start_time = time()  # 记录 step 开始时间
            x_img = x_img.to(device, non_blocking=True)
            
            # Encode images to latent
            with torch.no_grad(), autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                x_latent, _ = vae.encode(x_img)
            
            # Encode text
            with torch.no_grad():
                # CFG: randomly drop captions
                if args.cfg_prob > 0.0:
                    drop_mask = torch.rand(len(captions)) < args.cfg_prob
                    captions_train = [
                        "" if drop else cap 
                        for cap, drop in zip(captions, drop_mask)
                    ]
                else:
                    captions_train = list(captions)
                
                y_embed = text_encoder.encode(captions_train)
            
            # Forward pass
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                loss_tensor, pred_latent, noise, t_sample = compute_train_loss(
                    model=model,
                    x_latent=x_latent,
                    y_embed=y_embed,
                    time_shift=time_shift,
                    noise_schedule=args.noise_schedule,
                    log_norm_mean=args.log_norm_mean,
                    log_norm_std=args.log_norm_std,
                )
            
            # NaN check
            is_nan_local = 0 if torch.isfinite(loss_tensor) else 1
            is_nan = torch.tensor(is_nan_local, device=device)
            dist.all_reduce(is_nan, op=dist.ReduceOp.SUM)
            
            if is_nan.item() > 0:
                if rank == 0:
                    logger.warning(f"[step {train_steps}] NaN detected! Skipping...")
                opt.zero_grad(set_to_none=True)
                accum_counter = 0
                step_loss_accum = 0.0
                dist.barrier()
                continue
            
            # Backward
            step_loss_accum += loss_tensor.item()
            loss_tensor = loss_tensor / args.grad_accum_steps
            loss_tensor.backward()
            accum_counter += 1
            
            if accum_counter < args.grad_accum_steps:
                continue
            
            # Optimizer step
            if args.clip_grad > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            else:
                grad_norm = torch.tensor(0.0, device=device)
            
            opt.step()
            update_ema(ema, model.module, decay=args.ema_decay)
            opt.zero_grad()
            
            running_loss += step_loss_accum / args.grad_accum_steps
            log_steps += 1
            train_steps += 1
            accum_counter = 0
            step_loss_accum = 0.0
            
            # 记录 step 时间
            torch.cuda.synchronize()  # 确保 GPU 操作完成
            step_time = time() - step_start_time
            step_times.append(step_time)
            
            # 每 10 个 step 打印一次平均时间
            if rank == 0 and train_steps % 10 == 0:
                avg_step_time = sum(step_times[-100:]) / len(step_times[-100:])  # 最近 100 步平均
                print(f"[step {train_steps}] Avg step time: {avg_step_time*1000:.1f}ms ({1/avg_step_time:.2f} steps/sec)")
            
            # Visualization (排除可视化时间，不计入 steps/sec)
            if rank == 0 and train_steps % args.vis_every == 0:
                vis_start = time()  # 记录可视化开始时间
                with torch.no_grad():
                    # Sample with EMA model - 临时移到 GPU
                    ema.to(device, dtype=torch.bfloat16)

                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                        sample_caption = captions[0] if captions[0] else "a beautiful landscape"
                        y_sample = text_encoder.encode([sample_caption])
                        y_uncond = text_encoder.get_uncond_embedding(1)

                        z0_hat = sample_latent_linear(
                            model=ema,
                            text_embeddings=y_sample,
                            latent_shape=latent_size,
                            device=device,
                            time_shift=time_shift,
                            steps=50,
                            cfg_scale=args.cfg_scale,
                            uncond_embeddings=y_uncond,
                            cfg_interval=(args.cfg_interval_low, args.cfg_interval_high),
                        )

                    # 可选：采样完成后移回 CPU 以节省显存
                    if not args.ema_stay_on_gpu:
                        ema.to("cpu", dtype=torch.float32)
                        torch.cuda.empty_cache()
                    
                    # Decode to image
                    img_gen = reconstruct_from_latent_with_diffusion(
                        vae=vae,
                        latent_z=z0_hat.float(),
                        image_shape=torch.Size([1, 3, args.image_size, args.image_size]),
                        diffusion_steps=args.vae_diffusion_steps,
                        decoder_type=args.decoder_type,
                        encoder_type=args.encoder_type,
                        denormalize_fn_override=normalize_fn if args.latent_stats_path else None,
                    )
                    
                    img_gen_01 = (img_gen + 1.0) / 2.0
                    vis_dir = os.path.join(experiment_dir, "samples")
                    os.makedirs(vis_dir, exist_ok=True)
                    save_image(img_gen_01, os.path.join(vis_dir, f"gen_step_{train_steps:07d}.png"))
                    
                    # Save caption
                    with open(os.path.join(vis_dir, f"gen_step_{train_steps:07d}.txt"), "w") as f:
                        f.write(sample_caption)
                
                vis_time = time() - vis_start
                start_time += vis_time  # 把可视化时间加回 start_time，这样就不会算入 steps/sec
                logger.info(f"[step {train_steps}] Saved sample image ({vis_time:.1f}s). Caption: {sample_caption[:50]}...")
            
            # Logging
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / max(1, log_steps), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                
                if rank == 0:
                    print(
                        f"(step={train_steps:07d}) "
                        f"Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}, "
                        f"Grad Norm: {float(grad_norm):.4f}"
                    )
                    if writer:
                        writer.add_scalar("train/loss", avg_loss, global_step=train_steps)
                        writer.add_scalar("train/steps_per_sec", steps_per_sec, global_step=train_steps)
                        writer.add_scalar("train/grad_norm", float(grad_norm), global_step=train_steps)
                
                running_loss = 0.0
                log_steps = 0
                start_time = time()
            
            # Checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                ckpt_start = time()
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "train_steps": train_steps,
                        "args": vars(args),
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint: {checkpoint_path}")
                dist.barrier()
                start_time += time() - ckpt_start  # 排除 checkpoint 保存时间
            
            # Latest checkpoint
            if train_steps % 1000 == 0 and train_steps > 0:
                latest_start = time()
                if rank == 0:
                    snapshot = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "train_steps": train_steps,
                        "args": vars(args),
                    }
                    torch.save(snapshot, f"{checkpoint_dir}/latest.pt")
                    logger.info(f"Updated latest checkpoint")
                dist.barrier()
                start_time += time() - latest_start  # 排除保存时间
    
    if rank == 0:
        logger.info("Training complete!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to data. For webdataset, can be path to tar files or directory containing tars")
    parser.add_argument("--dataset-type", type=str, default="imagenet", 
                        choices=["imagenet", "custom", "webdataset"],
                        help="Dataset type: imagenet, custom (text-image pairs), or webdataset (tar files)")
    parser.add_argument("--class-to-caption-file", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    
    # WebDataset options
    parser.add_argument("--wds-image-key", type=str, default="jpg",
                        help="Key for image in webdataset tar (jpg, png, webp)")
    parser.add_argument("--wds-caption-key", type=str, default="txt",
                        help="Key for caption in webdataset tar (txt, json, caption)")
    parser.add_argument("--wds-shuffle-buffer", type=int, default=5000,
                        help="Shuffle buffer size for webdataset")
    parser.add_argument("--wds-epoch-length", type=int, default=None,
                        help="Number of samples per epoch for webdataset (for infinite datasets)")
    
    # Model
    parser.add_argument("--model-size", type=str, default="XXL", choices=["XXL", "L", "B"])
    parser.add_argument("--latent-channels", type=int, default=None, 
                        help="Latent channels. If not specified, auto-detect from encoder_type.")
    parser.add_argument("--num-text-refine-blocks", type=int, default=4)
    parser.add_argument("--use-qknorm", action="store_true")
    
    # Text encoder
    parser.add_argument("--text-encoder-path", type=str, required=True)
    parser.add_argument("--txt-max-length", type=int, default=128)
    
    # VAE
    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--vae-use-ema", action="store_true")
    parser.add_argument("--encoder-type", type=str, default="dinov3", 
                        choices=["dinov3", "dinov3_vitl", "dinov2", "siglip2"],
                        help="Encoder type. latent_channels will be auto-detected if not specified.")
    parser.add_argument("--decoder-type", type=str, default="cnn_decoder")
    parser.add_argument("--dinov3-dir", type=str, default="/cpfs01/huangxu/models/dinov3")
    parser.add_argument("--dinov2-model-name", type=str, default="facebook/dinov2-with-registers-base")
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=256)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--skip-to-moments", action="store_true")
    parser.add_argument("--denormalize-decoder-output", action="store_true")
    parser.add_argument("--vae-diffusion-steps", type=int, default=50)
    parser.add_argument("--latent-stats-path", type=str, default=None)
    parser.add_argument("--per-channel-norm", action="store_true")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--ema-stay-on-gpu", action="store_true",
                        help="Keep EMA on GPU during visualization sampling (default moves EMA back to CPU)")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing for DiT blocks to save memory")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=0)
    
    # Flow matching
    parser.add_argument("--time-shift-base", type=int, default=4096)
    parser.add_argument("--noise-schedule", type=str, default="uniform", choices=["uniform", "log_norm"])
    parser.add_argument("--log-norm-mean", type=float, default=0.0)
    parser.add_argument("--log-norm-std", type=float, default=1.0)
    
    # CFG
    parser.add_argument("--cfg-prob", type=float, default=0.1, help="Probability of dropping caption during training")
    parser.add_argument("--cfg-scale", type=float, default=7.5, help="CFG scale for sampling")
    parser.add_argument("--cfg-interval-low", type=float, default=0.0)
    parser.add_argument("--cfg-interval-high", type=float, default=1.0)
    
    # Logging
    parser.add_argument("--results-dir", type=str, default="results_t2i")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--vis-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--ckpt", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--auto-resume", action="store_true", 
                        help="Auto resume from latest.pt if available")
    
    # Compile
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead")
    
    args = parser.parse_args()
    main(args)
