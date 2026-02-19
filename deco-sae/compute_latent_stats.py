"""
Compute mean and std of DECO-SAE latent features on ImageNet.
Useful for DiT training normalization.

Usage:
torchrun --nproc_per_node=8 deco-sae/compute_latent_stats.py \
    --config deco-sae/dinov2_base_sae_vit_decoder.yaml \
    --ckpt results_sae/xxx/step_xxx.pth \
    --data-path /path/to/imagenet/train \
    --output-dir results_sae/latent_stats
"""

import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

os.environ["TORCH_HOME"] = "/cpfs01/huangxu/.cache/torch"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config_utils import load_config
from model import DecoSAE
from models.rae.utils.ddp_utils import cleanup_ddp, setup_ddp
from models.rae.utils.image_utils import center_crop_arr


# ==========================================
#         Running Statistics
# ==========================================

class RunningStats:
    """
    Welford's online algorithm for computing running mean and variance.
    Numerically stable for large datasets.
    """
    def __init__(self, shape, device='cpu'):
        self.n = 0
        self.mean = torch.zeros(shape, device=device, dtype=torch.float64)
        self.M2 = torch.zeros(shape, device=device, dtype=torch.float64)
    
    def update_batch(self, x: torch.Tensor):
        """
        Batch update for efficiency.
        x: [B, C, H, W]
        """
        if x.dim() == 4:
            B, C, H, W = x.shape
            batch_mean = x.mean(dim=[0, 2, 3]).double()  # [C]
            batch_var = x.var(dim=[0, 2, 3], unbiased=False).double()  # [C]
            batch_n = B * H * W
        elif x.dim() == 3:
            # x: [B, N, C] -> [B, C, H, W] where N = H * W
            B, N, C = x.shape
            batch_mean = x.mean(dim=[0, 1]).double()  # [C]
            batch_var = x.var(dim=[0, 1], unbiased=False).double()  # [C]
            batch_n = B * N
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")
        
        if self.n == 0:
            self.mean = batch_mean
            self.M2 = batch_var * batch_n
            self.n = batch_n
        else:
            delta = batch_mean - self.mean
            total_n = self.n + batch_n
            self.mean = self.mean + delta * batch_n / total_n
            self.M2 = self.M2 + batch_var * batch_n + delta ** 2 * self.n * batch_n / total_n
            self.n = total_n
    
    def get_mean(self):
        return self.mean.float()
    
    def get_std(self):
        if self.n < 2:
            return torch.zeros_like(self.mean).float()
        variance = self.M2 / self.n
        return torch.sqrt(variance).float()
    
    def get_var(self):
        if self.n < 2:
            return torch.zeros_like(self.mean).float()
        return (self.M2 / self.n).float()


# ==========================================
#              Model Building
# ==========================================

def build_deco_sae(sae_cfg, device: torch.device) -> DecoSAE:
    """Build DecoSAE model from config."""
    return DecoSAE(
        encoder_type=sae_cfg.encoder.type,
        dinov3_model_dir=sae_cfg.encoder.dinov3_model_dir,
        siglip2_model_name=sae_cfg.encoder.siglip2_model_name,
        dinov2_model_name=sae_cfg.encoder.dinov2_model_name,
        image_size=sae_cfg.data.image_size,
        in_channels=3,
        out_channels=3,
        hidden_size=sae_cfg.model.hidden_size,
        hidden_size_x=sae_cfg.model.hidden_size_x,
        decoder_type=getattr(sae_cfg.model, "decoder_type", "vit_decoder"),
        num_decoder_blocks=sae_cfg.model.num_decoder_blocks,
        nerf_max_freqs=sae_cfg.model.nerf_max_freqs,
        flow_steps=sae_cfg.model.flow_steps,
        time_dim=sae_cfg.model.time_dim,
        vit_decoder_hidden_size=getattr(sae_cfg.model, "vit_decoder_hidden_size", 1024),
        vit_decoder_num_layers=getattr(sae_cfg.model, "vit_decoder_num_layers", 24),
        vit_decoder_num_heads=getattr(sae_cfg.model, "vit_decoder_num_heads", 16),
        vit_decoder_intermediate_size=getattr(sae_cfg.model, "vit_decoder_intermediate_size", 4096),
        vit_decoder_dropout=getattr(sae_cfg.model, "vit_decoder_dropout", 0.0),
        gradient_checkpointing=False,
        enable_hf_branch=sae_cfg.model.enable_hf_branch,
        hf_dim=sae_cfg.model.hf_dim,
        hf_encoder_config_path=getattr(sae_cfg.model, "hf_encoder_config_path", None),
        hf_dropout_prob=0.0,
        hf_noise_std=0.0,
        hf_loss_weight=sae_cfg.model.hf_loss_weight,
        recon_l2_weight=sae_cfg.loss.recon_l2_weight,
        recon_l1_weight=sae_cfg.loss.recon_l1_weight,
        recon_lpips_weight=sae_cfg.loss.recon_lpips_weight,
        recon_gan_weight=sae_cfg.loss.recon_gan_weight,
        lora_rank=sae_cfg.model.lora_rank,
        lora_alpha=sae_cfg.model.lora_alpha,
        lora_dropout=sae_cfg.model.lora_dropout,
        enable_lora=sae_cfg.model.enable_lora,
        target_latent_channels=sae_cfg.model.target_latent_channels,
        variational=sae_cfg.model.variational,
        kl_weight=sae_cfg.model.kl_weight,
        skip_to_moments=sae_cfg.model.skip_to_moments,
        noise_tau=0.0,
        random_masking_channel_ratio=0.0,
        denormalize_decoder_output=sae_cfg.model.denormalize_decoder_output,
    ).to(device)


def load_sae_checkpoint(model: DecoSAE, ckpt_path: str, device: torch.device, verbose=True):
    """Load DECO-SAE checkpoint."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in payload:
        state_dict = payload["model"]
    elif "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    step = payload.get("step", 0) if isinstance(payload, dict) else 0
    if verbose:
        print(f"Loaded SAE checkpoint from {ckpt_path} (step={step})")
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    return step


@torch.no_grad()
def encode_with_deco_sae(sae_model: DecoSAE, x_img: torch.Tensor):
    """
    Encode images using DECO-SAE.
    Returns:
        latent: [B, C, H, W] fused latent (semantic + HF)
        enc_cond: [B, N, C] fused encoding
    """
    sae_model.eval()
    
    # Get semantic latent
    enc = sae_model._infer_latent(x_img)
    z = enc.latent  # [B, semantic_channels, H_p, W_p]
    
    # Get fused encoding (semantic + HF)
    enc_cond = sae_model.encode(z, x_img=x_img, force_drop_hf=False)  # [B, N, C]
    
    # Reshape to spatial format [B, C, H, W]
    B, N, C = enc_cond.shape
    H = W = int(math.sqrt(N))
    latent = enc_cond.transpose(1, 2).contiguous().view(B, C, H, W)
    
    return latent, enc_cond


# ==========================================
#                   Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Compute latent statistics for DECO-SAE")
    
    # Data
    parser.add_argument("--data-path", type=str, required=True, help="Path to ImageNet train set.")
    parser.add_argument("--num-samples", type=int, default=50000, help="Number of samples to use.")
    
    # Model
    parser.add_argument("--config", type=str, required=True, help="DECO-SAE config path")
    parser.add_argument("--ckpt", type=str, required=True, help="DECO-SAE checkpoint path")
    
    # Processing
    parser.add_argument("--batch-size", type=int, default=64)
    
    # Output
    parser.add_argument("--output-dir", type=str, default="results_sae/latent_stats")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    # 1. Initialize DDP
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # Set seed
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    # Load config
    sae_cfg = load_config(args.config)
    image_size = sae_cfg.data.image_size

    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print("=" * 60)
        print(" Computing Latent Statistics for DECO-SAE")
        print("=" * 60)
        print(f" Config:      {args.config}")
        print(f" Checkpoint:  {args.ckpt}")
        print(f" Image Size:  {image_size}")
        print(f" Num Samples: {args.num_samples}")
        print(f" GPUs:        {world_size}")
        print("=" * 60)
    
    dist.barrier()

    # 2. Load DECO-SAE
    if rank == 0:
        print("\nLoading DECO-SAE...")
    
    sae_model = build_deco_sae(sae_cfg, device)
    load_sae_checkpoint(sae_model, args.ckpt, device, verbose=(rank == 0))
    sae_model.eval()
    
    semantic_channels = sae_model.semantic_channels
    hf_dim = sae_model.hf_dim
    total_channels = semantic_channels + hf_dim
    
    if rank == 0:
        print(f"Semantic channels: {semantic_channels}, HF dim: {hf_dim}, Total: {total_channels}")
    
    # 3. Load dataset
    if rank == 0:
        print("\nLoading dataset...")
    
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),
    ])
    
    dataset = ImageFolder(args.data_path, transform=transform)
    
    if rank == 0:
        print(f"Total dataset size: {len(dataset)}")
    
    # Random subset
    if args.num_samples < len(dataset):
        torch.manual_seed(args.seed)
        indices = torch.randperm(len(dataset))[:args.num_samples].tolist()
        dataset = Subset(dataset, indices)
    
    if rank == 0:
        print(f"Using {len(dataset)} samples")
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    # 4. Compute statistics
    if rank == 0:
        print("\nComputing statistics...")
    
    # Initialize running stats for full latent
    stats_full = RunningStats(shape=(total_channels,), device=device)
    # Separate stats for semantic and HF
    stats_semantic = RunningStats(shape=(semantic_channels,), device=device)
    stats_hf = RunningStats(shape=(hf_dim,), device=device)
    
    # Track min/max
    global_min = torch.full((total_channels,), float('inf'), device=device)
    global_max = torch.full((total_channels,), float('-inf'), device=device)
    
    with torch.no_grad():
        if rank == 0:
            iterator = tqdm(loader, desc="Processing")
        else:
            iterator = loader
        
        for x, _ in iterator:
            x = x.to(device)
            
            # Encode
            latent, enc_cond = encode_with_deco_sae(sae_model, x)  # [B, C, H, W], [B, N, C]
            
            # Update full stats
            stats_full.update_batch(latent)
            
            # Update separate stats
            semantic_part = latent[:, :semantic_channels, :, :]
            hf_part = latent[:, semantic_channels:, :, :]
            stats_semantic.update_batch(semantic_part)
            stats_hf.update_batch(hf_part)
            
            # Update min/max
            batch_min = latent.amin(dim=[0, 2, 3])
            batch_max = latent.amax(dim=[0, 2, 3])
            global_min = torch.minimum(global_min, batch_min)
            global_max = torch.maximum(global_max, batch_max)
    
    # 5. Synchronize across GPUs
    if rank == 0:
        print("\nSynchronizing across GPUs...")
    
    def sync_stats(stats):
        """Synchronize RunningStats across all ranks."""
        local_n = torch.tensor([stats.n], device=device, dtype=torch.float64)
        local_mean = stats.mean.clone()
        local_M2 = stats.M2.clone()
        
        all_n = [torch.zeros_like(local_n) for _ in range(world_size)]
        all_mean = [torch.zeros_like(local_mean) for _ in range(world_size)]
        all_M2 = [torch.zeros_like(local_M2) for _ in range(world_size)]
        
        dist.all_gather(all_n, local_n)
        dist.all_gather(all_mean, local_mean)
        dist.all_gather(all_M2, local_M2)
        
        combined_n = all_n[0].item()
        combined_mean = all_mean[0].clone()
        combined_M2 = all_M2[0].clone()
        
        for i in range(1, world_size):
            n_a = combined_n
            n_b = all_n[i].item()
            if n_b == 0:
                continue
            
            mean_a = combined_mean
            mean_b = all_mean[i]
            M2_a = combined_M2
            M2_b = all_M2[i]
            
            combined_n = n_a + n_b
            delta = mean_b - mean_a
            combined_mean = mean_a + delta * n_b / combined_n
            combined_M2 = M2_a + M2_b + delta ** 2 * n_a * n_b / combined_n
        
        final_mean = combined_mean.float()
        final_var = (combined_M2 / combined_n).float()
        final_std = torch.sqrt(final_var)
        
        return final_mean, final_std, final_var, combined_n
    
    # Sync all stats
    full_mean, full_std, full_var, total_n = sync_stats(stats_full)
    semantic_mean, semantic_std, _, _ = sync_stats(stats_semantic)
    hf_mean, hf_std, _, _ = sync_stats(stats_hf)
    
    # Sync min/max
    dist.all_reduce(global_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
    
    # 6. Save results (rank 0 only)
    if rank == 0:
        spatial_size = int(math.sqrt(total_n / args.num_samples))
        
        print("\n" + "=" * 60)
        print(" Statistics Summary")
        print("=" * 60)
        print(f" Total samples processed: {args.num_samples}")
        print(f" Total tokens processed:  {int(total_n)}")
        print(f" Total channels:          {total_channels}")
        print(f"   - Semantic:            {semantic_channels}")
        print(f"   - HF:                  {hf_dim}")
        print(f" Spatial size:            {spatial_size}x{spatial_size}")
        print("-" * 60)
        print(" Full Latent (Semantic + HF):")
        print(f"   Mean range: [{full_mean.min().item():.6f}, {full_mean.max().item():.6f}]")
        print(f"   Std range:  [{full_std.min().item():.6f}, {full_std.max().item():.6f}]")
        print(f"   Mean of mean: {full_mean.mean().item():.6f}")
        print(f"   Mean of std:  {full_std.mean().item():.6f}")
        print("-" * 60)
        print(" Semantic Part:")
        print(f"   Mean range: [{semantic_mean.min().item():.6f}, {semantic_mean.max().item():.6f}]")
        print(f"   Std range:  [{semantic_std.min().item():.6f}, {semantic_std.max().item():.6f}]")
        print(f"   Mean of mean: {semantic_mean.mean().item():.6f}")
        print(f"   Mean of std:  {semantic_std.mean().item():.6f}")
        print("-" * 60)
        print(" HF Part:")
        print(f"   Mean range: [{hf_mean.min().item():.6f}, {hf_mean.max().item():.6f}]")
        print(f"   Std range:  [{hf_std.min().item():.6f}, {hf_std.max().item():.6f}]")
        print(f"   Mean of mean: {hf_mean.mean().item():.6f}")
        print(f"   Mean of std:  {hf_std.mean().item():.6f}")
        print("-" * 60)
        print(f" Global min: {global_min.min().item():.6f}")
        print(f" Global max: {global_max.max().item():.6f}")
        print("=" * 60)
        
        # Save as numpy
        np.savez(
            os.path.join(args.output_dir, "latent_stats.npz"),
            # Full latent
            mean=full_mean.cpu().numpy(),
            std=full_std.cpu().numpy(),
            var=full_var.cpu().numpy(),
            # Semantic
            semantic_mean=semantic_mean.cpu().numpy(),
            semantic_std=semantic_std.cpu().numpy(),
            # HF
            hf_mean=hf_mean.cpu().numpy(),
            hf_std=hf_std.cpu().numpy(),
            # Min/Max
            min=global_min.cpu().numpy(),
            max=global_max.cpu().numpy(),
            # Meta
            num_samples=args.num_samples,
            num_tokens=int(total_n),
            semantic_channels=semantic_channels,
            hf_dim=hf_dim,
        )
        print(f"\nSaved: {os.path.join(args.output_dir, 'latent_stats.npz')}")
        
        # Save as YAML
        stats_dict = {
            "config": args.config,
            "checkpoint": args.ckpt,
            "num_samples": args.num_samples,
            "image_size": image_size,
            "spatial_size": spatial_size,
            "total_channels": total_channels,
            "semantic_channels": semantic_channels,
            "hf_dim": hf_dim,
            # Full latent stats
            "full": {
                "mean": full_mean.cpu().tolist(),
                "std": full_std.cpu().tolist(),
                "mean_of_mean": float(full_mean.mean().item()),
                "mean_of_std": float(full_std.mean().item()),
            },
            # Semantic stats
            "semantic": {
                "mean": semantic_mean.cpu().tolist(),
                "std": semantic_std.cpu().tolist(),
                "mean_of_mean": float(semantic_mean.mean().item()),
                "mean_of_std": float(semantic_std.mean().item()),
            },
            # HF stats
            "hf": {
                "mean": hf_mean.cpu().tolist(),
                "std": hf_std.cpu().tolist(),
                "mean_of_mean": float(hf_mean.mean().item()),
                "mean_of_std": float(hf_std.mean().item()),
            },
            # Global range
            "global_min": float(global_min.min().item()),
            "global_max": float(global_max.max().item()),
        }
        
        yaml_path = os.path.join(args.output_dir, "latent_stats.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(stats_dict, f, default_flow_style=False, allow_unicode=True)
        print(f"Saved: {yaml_path}")
        
        # Save as PyTorch tensors
        torch.save({
            "mean": full_mean.cpu(),
            "std": full_std.cpu(),
            "var": full_var.cpu(),
            "semantic_mean": semantic_mean.cpu(),
            "semantic_std": semantic_std.cpu(),
            "hf_mean": hf_mean.cpu(),
            "hf_std": hf_std.cpu(),
            "min": global_min.cpu(),
            "max": global_max.cpu(),
        }, os.path.join(args.output_dir, "latent_stats.pt"))
        print(f"Saved: {os.path.join(args.output_dir, 'latent_stats.pt')}")
        
        print("\n" + "=" * 60)
        print(" Normalization Suggestion for DiT Training")
        print("=" * 60)
        print(f" Full latent normalization:")
        print(f"   z_norm = (z - {full_mean.mean().item():.4f}) / {full_std.mean().item():.4f}")
        print("-" * 60)
        print(f" Per-part normalization:")
        print(f"   semantic: (z[:, :{semantic_channels}] - {semantic_mean.mean().item():.4f}) / {semantic_std.mean().item():.4f}")
        print(f"   hf:       (z[:, {semantic_channels}:] - {hf_mean.mean().item():.4f}) / {hf_std.mean().item():.4f}")
        print("=" * 60)
    
    dist.barrier()
    cleanup_ddp()
    
    if rank == 0:
        print("\nDone!")


if __name__ == "__main__":
    main()
