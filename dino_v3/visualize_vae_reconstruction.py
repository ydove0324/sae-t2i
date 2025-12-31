#!/usr/bin/env python3
"""
Compare reconstruction quality between trained VAE models (with distributed inference support):
1. DINO-Encoder + CNN-Decoder (train_dino-vae_decoder-only.sh)
2. DINO-Encoder + Diffusion-Decoder (train_dinov3_diffusion_decoder.sh)
3. CNN-Encoder + CNN-Decoder (train_cnn-vae.sh)

At least one checkpoint must be provided.

Usage (single GPU):
    python compare_vae_reconstruction.py \
        --dino_cnn_ckpt /path/to/dino_cnn/ema_vae.pth \
        --output_dir ./comparison_results

Usage (distributed, e.g., 8 GPUs):
    TORCHRUN compare_vae_reconstruction.py \
        --dino_cnn_ckpt /path/to/dino_cnn/ema_vae.pth \
        --output_dir ./comparison_results
"""

import argparse
import os
from datetime import timedelta
from typing import Optional, List, Tuple, Dict

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for distributed
import matplotlib.pyplot as plt

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.config import create_object, create_dataset
from common.distributed import (
    get_device,
    get_global_rank,
    get_local_rank,
    get_world_size,
    init_torch,
)
from common.fs import download
from common.precision import init_precision
from omegaconf import OmegaConf
from data.video.transforms.clamp import Clamp
from projects.video_vae_v3.helper import collate_fn_with_filter


def parse_args():
    parser = argparse.ArgumentParser(description="Compare VAE reconstruction quality (distributed)")
    
    # Checkpoint paths (all optional, but at least one required)
    parser.add_argument(
        "--dino_cnn_ckpt",
        type=str,
        default=None,
        help="Path to DINO-Encoder + CNN-Decoder VAE checkpoint (ema_vae.pth or vae.pth)",
    )
    parser.add_argument(
        "--dino_diffusion_ckpt",
        type=str,
        default=None,
        help="Path to DINO-Encoder + Diffusion-Decoder VAE checkpoint (ema_vae.pth or vae.pth)",
    )
    parser.add_argument(
        "--cnn_cnn_ckpt",
        type=str,
        default=None,
        help="Path to CNN-Encoder + CNN-Decoder VAE checkpoint (ema_vae.pth or vae.pth)",
    )
    
    # Data settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco",
        choices=["coco", "imagenet"],
        help="Dataset to use for visualization (default: coco, which auto-downloads from HDFS)",
    )
    parser.add_argument(
        "--imagenet_path",
        type=str,
        default="/opt/tiger/imagenet1k",
        help="Path to ImageNet dataset root (only used if --dataset=imagenet)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Image resolution for evaluation",
    )
    
    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./comparison_results",
        help="Directory to save comparison results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per GPU (total samples = batch_size × num_gpus)",
    )
    
    # Diffusion decoder settings
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=25,
        help="Number of diffusion sampling steps (default: 25)",
    )
    parser.add_argument(
        "--use_full_diffusion_sampling",
        action="store_true",
        default=True,
        help="Use full diffusion sampling loop (default: True). If False, uses single-step training forward.",
    )
    parser.add_argument(
        "--use_training_forward",
        action="store_true",
        default=False,
        help="Use training-style single-step forward (matches validation during training). Overrides --use_full_diffusion_sampling.",
    )
    
    # Precision settings
    parser.add_argument(
        "--precision",
        type=str,
        default="tf32",
        choices=["fp32", "fp16", "bf16", "tf32"],
        help="Precision for inference",
    )
    
    args = parser.parse_args()
    
    # Validate that at least one checkpoint is provided
    if not any([args.dino_cnn_ckpt, args.dino_diffusion_ckpt, args.cnn_cnn_ckpt]):
        parser.error("At least one checkpoint must be provided (--dino_cnn_ckpt, --dino_diffusion_ckpt, or --cnn_cnn_ckpt)")
    
    return args


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    """Load checkpoint from local path or HDFS."""
    if checkpoint_path.startswith("hdfs://"):
        checkpoint_path = download(checkpoint_path)
    return torch.load(checkpoint_path, map_location=device)


def load_dino_vae_model(checkpoint_path: str, device: torch.device):
    """
    Load DINO-Encoder based VAE model (for both CNN and Diffusion decoder).
    Uses the same model config as train_image.yaml.
    """
    # Model config for DINO-based VAE (matching models/dino_v3/s16_c1280.yaml)
    model_config = OmegaConf.create({
        "__object__": {
            "path": "models.dino_v3.image_vae_dinov3_encode",
            "name": "AutoencoderKL",
            "args": "as_params",
        },
        "enc_block_out_channels": [128, 256, 384, 512, 768],
        "dec_block_out_channels": [1280, 1024, 512, 256, 128],
        "enc_layers_per_block": 2,
        "dec_layers_per_block": 3,
        "in_channels": 3,
        "latent_channels": 1280,
        "out_channels": 3,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
        "spatial_downsample_factor": 16,
        "variational": False,
        "running_mode": "dec",
        "noise_tau": 0.0,  # Disable noise at inference
        "denormalize_decoder_output": True,
    })
    
    # Create model using common.config.create_object
    vae = create_object(model_config)
    vae.to(device)
    vae.eval()
    
    # Load checkpoint
    state_dict = load_checkpoint(checkpoint_path, device)
    vae.load_state_dict(state_dict, strict=False)
    
    if get_global_rank() == 0:
        print(f"[Rank 0] Loaded DINO VAE from: {checkpoint_path}")
    
    return vae


def load_cnn_vae_model(checkpoint_path: str, device: torch.device):
    """
    Load CNN-Encoder + CNN-Decoder VAE model.
    Uses the same model config as train_cnn-vae.sh (models/video_vae_v4/s16_c48.yaml).
    """
    # Model config for pure CNN VAE
    model_config = OmegaConf.create({
        "__object__": {
            "path": "models.video_vae_v4.modules.image_vae",
            "name": "AutoencoderKL",
            "args": "as_params",
        },
        "enc_block_out_channels": [128, 256, 384, 512, 768],
        "dec_block_out_channels": [768, 512, 384, 256, 128],
        "enc_layers_per_block": 2,
        "dec_layers_per_block": 3,
        "in_channels": 3,
        "latent_channels": 48,
        "out_channels": 3,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
        "spatial_downsample_factor": 16,
        "variational": True,
    })
    
    # Create model using common.config.create_object
    vae = create_object(model_config)
    vae.to(device)
    vae.eval()
    
    # Load checkpoint
    state_dict = load_checkpoint(checkpoint_path, device)
    vae.load_state_dict(state_dict, strict=False)
    
    if get_global_rank() == 0:
        print(f"[Rank 0] Loaded CNN VAE from: {checkpoint_path}")
    
    return vae


def create_image_transform(resolution: int):
    """
    Create image transform for preprocessing.
    Matches the transform used in projects/video_vae_v3/train_image.py.
    """
    return Compose([
        Resize(resolution, interpolation=InterpolationMode.BILINEAR, antialias=True),
        CenterCrop(resolution),
        ToTensor(),
        Clamp(0, 1),
        Normalize(mean=0.5, std=0.5),
    ])


def create_dataloader(dataset_name: str, imagenet_path: str, transform, batch_size: int, seed: int = 42):
    """
    Create distributed dataloader for evaluation.
    Uses the same approach as projects/video_vae_v3/eval_image.py.
    
    Args:
        dataset_name: "coco" or "imagenet"
        imagenet_path: Path to ImageNet (only used if dataset_name="imagenet")
        transform: Image transform
        batch_size: Batch size per GPU
        seed: Random seed
    """
    if dataset_name == "coco":
        # COCO 2017 supports auto-download from HDFS
        if get_global_rank() == 0:
            print(f"[Rank 0] Loading COCO 2017 dataset (auto-download from HDFS if needed)...")
        
        dataset = create_dataset(
            "data.image.configs.benchmark.coco_2017",
            image_transform=transform,
            seed=seed,
        )
        if get_global_rank() == 0:
            print(f"[Rank 0] Loaded COCO 2017 dataset")
    else:
        # ImageNet requires local path
        if get_global_rank() == 0:
            print(f"[Rank 0] Loading ImageNet dataset from: {imagenet_path}")
        
        try:
            dataset = create_dataset(
                "data.image.configs.benchmark.imagenet",
                image_transform=transform,
                seed=seed,
            )
            if get_global_rank() == 0:
                print(f"[Rank 0] Loaded ImageNet dataset from config")
        except Exception as e:
            if get_global_rank() == 0:
                print(f"[Rank 0] Failed to load from config: {e}")
                print(f"[Rank 0] Trying direct path: {imagenet_path}")
            
            from torchvision.datasets import ImageFolder
            val_path = os.path.join(imagenet_path, "val")
            
            if not os.path.isdir(val_path):
                raise ValueError(f"ImageNet val path not found: {val_path}")
            
            # Wrap ImageFolder to return dict format
            class ImageNetDataset(ImageFolder):
                def __getitem__(self, index):
                    img, label = super().__getitem__(index)
                    return {"image": img, "class_label": label}
            
            dataset = ImageNetDataset(val_path, transform=transform)
    
    # Use DistributedSampler for multi-GPU
    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        shuffle=False,
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collate_fn_with_filter,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        pin_memory_device=str(get_device()),
    )
    
    return dataloader, dataset


@torch.no_grad()
def reconstruct_cnn_decoder(vae, images: torch.Tensor) -> torch.Tensor:
    """Reconstruct images using CNN decoder (works for both DINO-CNN and CNN-CNN)."""
    z, _ = vae.encode(images)
    recon = vae.decode(z).sample
    return recon


@torch.no_grad()
def reconstruct_diffusion_decoder_training_forward(vae, images: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct images using Diffusion decoder with TRAINING-STYLE single-step forward.
    
    This matches how the model is evaluated during training (train_image.py evaluation_loop):
    - vae(images) calls encode -> diffusion_decode_training
    - diffusion_decode_training does single-step denoising from a random timestep
    - Output is processed by _denormalize_output (if enabled)
    
    Note: This does NOT use the full diffusion sampling loop.
    The reconstruction quality will be lower than full sampling.
    
    The output range is approximately [-1.2, 1.3], which should be clipped to [-1, 1]
    then converted to [0, 1] for display using the standard denormalize_for_display.
    """
    # Use the same forward method as training validation
    output = vae(images)
    recon = output.sample
    
    # The output from _denormalize_output is in approximately [-1.2, 1.3] range
    # We clip to [-1, 1] first (matching eval_image.py line 152)
    recon = recon.clip(-1, 1)
    
    return recon


@torch.no_grad()
def reconstruct_diffusion_decoder(vae, images: torch.Tensor, sample_steps: int = 25) -> torch.Tensor:
    """
    Reconstruct images using Diffusion decoder with FULL diffusion sampling loop.
    
    This is the proper inference method that runs the complete denoising process:
    1. Encode images with DINO encoder to get latent z
    2. Use z to get multi-scale context features from compressor
    3. Run full DDIM sampling loop for `sample_steps` iterations
    4. Apply denormalization
    
    Args:
        vae: The AutoencoderKL model with diffusion decoder
        images: Input images in [-1, 1] range, shape [B, C, H, W]
        sample_steps: Number of diffusion sampling steps (more steps = better quality but slower)
    
    Returns:
        Reconstructed images in [-1, 1] range
    """
    # 1. Encode with DINO encoder
    z, _ = vae.encode(images)
    
    # 2. Apply post_quant_conv if exists
    z = vae.post_quant_conv(z) if vae.post_quant_conv is not None else z
    
    # 3. Get context from z (using diffusion decoder's compressor)
    # The compressor.decode() returns multi-scale features that condition the UNet
    context = vae.diffusion_decoder.get_context(z)
    # Reverse the context order to match what forward() does
    corrected_context = list(reversed(context[:]))
    
    # 4. Set up diffusion sampling schedule
    diffusion = vae.diffusion_decoder.diffusion
    diffusion.set_sample_schedule(sample_steps, images.device)
    
    # 5. Run full diffusion sampling loop
    b, c, h, w = images.shape
    shape = (b, c, h, w)
    
    # Start from random Gaussian noise (standard DDPM/DDIM)
    init_noise = torch.randn(shape, device=images.device, dtype=images.dtype)
    
    recon = diffusion.p_sample_loop(
        shape=shape,
        context=corrected_context,
        clip_denoised=True,
        init=init_noise,  # Start from random Gaussian noise
        eta=0,  # Deterministic DDIM (eta=0)
    )
    
    # 6. Apply denormalization (convert from ImageNet normalization to [-1, 1])
    recon = vae._denormalize_output(recon)
    recon = recon.clip(-1, 1)
    
    return recon


def denormalize_for_display(images: torch.Tensor) -> torch.Tensor:
    """Convert from [-1, 1] to [0, 1] for display."""
    return (images.clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1)


def gather_tensors(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensors from all ranks to rank 0."""
    if get_world_size() == 1:
        return tensor
    
    # Get tensor sizes from all ranks
    local_size = torch.tensor([tensor.shape[0]], device=tensor.device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=tensor.device) for _ in range(get_world_size())]
    dist.all_gather(all_sizes, local_size)
    
    max_size = max([s.item() for s in all_sizes])
    
    # Pad tensor to max size
    if tensor.shape[0] < max_size:
        padding = torch.zeros(max_size - tensor.shape[0], *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, padding], dim=0)
    
    # Gather all tensors
    gathered = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(gathered, tensor)
    
    # Concatenate and remove padding
    result = []
    for i, (g, s) in enumerate(zip(gathered, all_sizes)):
        result.append(g[:s.item()])
    
    return torch.cat(result, dim=0)


def create_comparison_grid(
    original: torch.Tensor,
    recons: List[Tuple[str, torch.Tensor]],
) -> torch.Tensor:
    """Create a comparison grid with all available reconstructions."""
    batch_size = original.shape[0]
    num_cols = 1 + len(recons)
    
    original_display = denormalize_for_display(original)
    
    comparison = []
    for i in range(batch_size):
        comparison.append(original_display[i])
        for name, recon in recons:
            # Recon is in [-1, 1], convert to [0, 1]
            comparison.append(denormalize_for_display(recon[i]))
    
    comparison = torch.stack(comparison)
    grid = make_grid(comparison, nrow=num_cols, padding=4, normalize=False, pad_value=1.0)
    
    return grid


def create_detailed_figure(
    original: torch.Tensor,
    recons: List[Tuple[str, torch.Tensor]],
    save_path: str,
    num_display: int = 8,
):
    """Create a detailed matplotlib figure for visualization."""
    num_display = min(num_display, original.shape[0])
    num_cols = 1 + len(recons)
    
    original_display = denormalize_for_display(original[:num_display])
    
    fig, axes = plt.subplots(num_display, num_cols, figsize=(4 * num_cols, 4 * num_display))
    
    if num_display == 1:
        axes = axes.reshape(1, -1)
    
    col_titles = ["Original"] + [name for name, _ in recons]
    
    for i in range(num_display):
        axes[i, 0].imshow(original_display[i].permute(1, 2, 0).cpu().numpy())
        if i == 0:
            axes[i, 0].set_title(col_titles[0], fontsize=14, fontweight='bold')
        axes[i, 0].axis("off")
        
        for j, (name, recon) in enumerate(recons):
            # Recon is in [-1, 1], convert to [0, 1]
            recon_display = denormalize_for_display(recon[i])
            axes[i, j + 1].imshow(recon_display.permute(1, 2, 0).cpu().numpy())
            if i == 0:
                axes[i, j + 1].set_title(col_titles[j + 1], fontsize=14, fontweight='bold')
            axes[i, j + 1].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Rank 0] Saved detailed figure to: {save_path}")


def main():
    args = parse_args()
    
    # Initialize distributed environment
    init_torch(timeout=timedelta(seconds=3600), cudnn_benchmark=False)
    
    # Initialize precision (required for autocast)
    precision_config = OmegaConf.create({"training": {"precision": "tf32"}})
    init_precision(precision_config)
    
    device = get_device()
    rank = get_global_rank()
    world_size = get_world_size()
    
    if rank == 0:
        print(f"Running with {world_size} GPU(s)")
        print(f"Using device: {device}")
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Synchronize before loading models
    dist.barrier()
    
    # Track which models are loaded
    models: Dict[str, Tuple[nn.Module, bool]] = {}  # name -> (vae, is_diffusion)
    
    # Load models based on provided checkpoints
    if args.dino_cnn_ckpt:
        if rank == 0:
            print("\n" + "=" * 60)
            print("Loading DINO-Encoder + CNN-Decoder VAE...")
            print("=" * 60)
        models["DINO + CNN"] = (load_dino_vae_model(args.dino_cnn_ckpt, device), False)
    
    if args.dino_diffusion_ckpt:
        if rank == 0:
            print("\n" + "=" * 60)
            print("Loading DINO-Encoder + Diffusion-Decoder VAE...")
            print("=" * 60)
        models["DINO + Diffusion"] = (load_dino_vae_model(args.dino_diffusion_ckpt, device), True)
    
    if args.cnn_cnn_ckpt:
        if rank == 0:
            print("\n" + "=" * 60)
            print("Loading CNN-Encoder + CNN-Decoder VAE...")
            print("=" * 60)
        models["CNN + CNN"] = (load_cnn_vae_model(args.cnn_cnn_ckpt, device), False)
    
    if rank == 0:
        print(f"\nLoaded {len(models)} model(s): {list(models.keys())}")
    
    # Synchronize after loading models
    dist.barrier()
    
    # Create dataloader
    if rank == 0:
        print("\n" + "=" * 60)
        print(f"Loading {args.dataset.upper()} dataset...")
        print("=" * 60)
    
    transform = create_image_transform(args.resolution)
    dataloader, dataset = create_dataloader(
        args.dataset, args.imagenet_path, transform, args.batch_size
    )
    
    if rank == 0:
        print(f"[Rank 0] Dataset size: {len(dataset)}, batch_size per GPU: {args.batch_size}")
    
    # Process one batch per GPU
    local_recons = {name: None for name in models.keys()}
    
    # Determine diffusion inference mode
    use_full_sampling = args.use_full_diffusion_sampling and not args.use_training_forward
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("Running reconstruction...")
        print("=" * 60)
        if args.dino_diffusion_ckpt:
            if use_full_sampling:
                print(f"[Diffusion] Using FULL sampling loop with {args.diffusion_steps} steps")
            else:
                print(f"[Diffusion] Using TRAINING-STYLE single-step forward (matches training validation)")
    
    # Get first batch from dataloader
    batch = next(iter(dataloader))
    local_images = batch["image"].to(device)
    
    if rank == 0:
        print(f"[Rank 0] Processing {local_images.shape[0]} samples per GPU, total = {local_images.shape[0] * world_size}")
    
    with torch.inference_mode():
        for name, (vae, is_diffusion) in tqdm(models.items(), disable=rank != 0, desc="Reconstructing"):
            if is_diffusion:
                if use_full_sampling:
                    # Use full diffusion sampling loop (proper inference)
                    recon = reconstruct_diffusion_decoder(vae, local_images, sample_steps=args.diffusion_steps)
                else:
                    # Use training-style single-step forward (matches training validation)
                    recon = reconstruct_diffusion_decoder_training_forward(vae, local_images)
            else:
                recon = reconstruct_cnn_decoder(vae, local_images)
            local_recons[name] = recon.cpu()
    
    local_images = local_images.cpu()
    
    # Gather results from all ranks
    if rank == 0:
        print("\nGathering results from all ranks...")
    
    dist.barrier()
    
    all_images = gather_tensors(local_images.to(device)).cpu()
    all_recons = {}
    for name in models.keys():
        all_recons[name] = gather_tensors(local_recons[name].to(device)).cpu()
    
    # Only rank 0 saves results
    if rank == 0:
        print(f"\nTotal samples collected: {all_images.shape[0]} (batch_size={args.batch_size} × {world_size} GPUs)")
        
        # Prepare recons list for visualization
        recons_list = [(name, all_recons[name]) for name in models.keys()]
        
        # Save visualizations
        print("\n" + "=" * 60)
        print("Saving visualizations...")
        print("=" * 60)
        
        # Save grid image
        grid = create_comparison_grid(all_images, recons_list)
        grid_path = os.path.join(args.output_dir, "comparison_grid.png")
        save_image(grid, grid_path)
        print(f"Saved comparison grid to: {grid_path}")
        
        # Save detailed matplotlib figure
        figure_path = os.path.join(args.output_dir, "comparison_detailed.png")
        create_detailed_figure(
            all_images,
            recons_list,
            figure_path,
            num_display=min(8, len(all_images)),
        )
        
        # Save individual reconstructions
        individual_dir = os.path.join(args.output_dir, "individual")
        os.makedirs(individual_dir, exist_ok=True)
        
        for i in range(min(8, len(all_images))):
            orig_path = os.path.join(individual_dir, f"{i:03d}_original.png")
            save_image(denormalize_for_display(all_images[i]), orig_path)
            
            for name, recon in recons_list:
                safe_name = name.lower().replace(" ", "_").replace("+", "_")
                recon_path = os.path.join(individual_dir, f"{i:03d}_{safe_name}_recon.png")
                # Recon is in [-1, 1], convert to [0, 1]
                save_image(denormalize_for_display(recon[i]), recon_path)
        
        print(f"Saved individual images to: {individual_dir}")
        
        # Save info file
        info_path = os.path.join(args.output_dir, "info.txt")
        with open(info_path, "w") as f:
            f.write("VAE Reconstruction Comparison\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: {args.dataset.upper()}\n")
            f.write(f"World size (GPUs): {world_size}\n")
            f.write(f"Batch size per GPU: {args.batch_size}\n")
            f.write(f"Total samples: {all_images.shape[0]}\n")
            f.write(f"Resolution: {args.resolution}\n")
            f.write(f"\nModels compared:\n")
            for name in models.keys():
                f.write(f"  - {name}\n")
            if args.dino_diffusion_ckpt:
                f.write(f"\nDiffusion Decoder Settings:\n")
                if use_full_sampling:
                    f.write(f"  Inference mode: FULL diffusion sampling loop\n")
                    f.write(f"  Sampling steps: {args.diffusion_steps}\n")
                else:
                    f.write(f"  Inference mode: TRAINING-STYLE single-step forward\n")
                    f.write(f"  Note: This matches how the model is validated during training.\n")
                    f.write(f"        For better reconstruction, use --use_full_diffusion_sampling\n")
            f.write(f"\nCheckpoints:\n")
            if args.dino_cnn_ckpt:
                f.write(f"  DINO+CNN: {args.dino_cnn_ckpt}\n")
            if args.dino_diffusion_ckpt:
                f.write(f"  DINO+Diffusion: {args.dino_diffusion_ckpt}\n")
            if args.cnn_cnn_ckpt:
                f.write(f"  CNN+CNN: {args.cnn_cnn_ckpt}\n")
        
        print(f"Saved info to: {info_path}")
        
        print("\n" + "=" * 60)
        print("Comparison complete!")
        print(f"Results saved to: {args.output_dir}")
        print("=" * 60)
    
    # Final synchronization
    dist.barrier()


if __name__ == "__main__":
    main()
