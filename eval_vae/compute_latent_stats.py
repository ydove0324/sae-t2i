# compute_latent_stats.py
# Compute mean and std of VAE encoder latent features on ImageNet
# Useful for DiT training normalization

import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import yaml

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# 设置 torch 缓存路径 (可选)
os.environ["TORCH_HOME"] = "/cpfs01/huangxu/.cache/torch"

# 添加当前目录以导入项目模块
sys.path.append(".")

# 导入 VAE 工具函数
from models.rae.utils.vae_utils import load_vae, get_normalize_fn

# 导入统一工具模块
from models.rae.utils.ddp_utils import setup_ddp, cleanup_ddp
from models.rae.utils.image_utils import center_crop_arr
from models.rae.utils.argparse_utils import (
    add_encoder_args, add_lora_args, add_vit_decoder_args, get_encoder_config,
)


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
        self.M2 = torch.zeros(shape, device=device, dtype=torch.float64)  # Sum of squared differences
    
    def update(self, x: torch.Tensor):
        """
        Update running statistics with a batch of samples.
        x: [B, C, H, W] or [B, C]
        """
        # Flatten spatial dimensions if present
        if x.dim() == 4:
            # x: [B, C, H, W] -> compute per-channel stats
            B, C, H, W = x.shape
            x = x.permute(1, 0, 2, 3).reshape(C, -1)  # [C, B*H*W]
            n_samples = x.shape[1]
        elif x.dim() == 2:
            # x: [B, C]
            x = x.T  # [C, B]
            n_samples = x.shape[1]
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")
        
        x = x.double()
        
        for i in range(n_samples):
            self.n += 1
            delta = x[:, i] - self.mean
            self.mean += delta / self.n
            delta2 = x[:, i] - self.mean
            self.M2 += delta * delta2
    
    def update_batch(self, x: torch.Tensor):
        """
        Batch update for efficiency.
        x: [B, C, H, W]
        """
        if x.dim() == 4:
            B, C, H, W = x.shape
            # Compute batch mean and var per channel
            batch_mean = x.mean(dim=[0, 2, 3]).double()  # [C]
            batch_var = x.var(dim=[0, 2, 3], unbiased=False).double()  # [C]
            batch_n = B * H * W
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")
        
        if self.n == 0:
            self.mean = batch_mean
            self.M2 = batch_var * batch_n
            self.n = batch_n
        else:
            # Parallel algorithm for combining statistics
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
#                   Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Compute latent statistics for VAE encoder")
    
    # Data
    parser.add_argument("--data-path", type=str, required=True, help="Path to ImageNet train set.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=50000, help="Number of samples to use.")
    
    # VAE (使用 argparse_utils)
    parser.add_argument("--vae-ckpt", type=str, required=True, help="Path to VAE checkpoint.")
    add_encoder_args(parser)     # --encoder-type, --dinov3-dir, --siglip2-model-name, --dinov2-model-name
    add_lora_args(parser)        # --lora-rank, --lora-alpha, --lora-dropout, --no-lora
    add_vit_decoder_args(parser) # --vit-hidden-size, --vit-num-layers, --vit-num-heads, --vit-intermediate-size
    
    # Decoder type
    parser.add_argument("--decoder-type", type=str, default="cnn_decoder",
                        choices=["cnn_decoder", "vit_decoder"])
    
    # Processing
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--use-lora", action="store_true", default=True, 
                        help="Use LoRA-enabled encoder features.")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="results/latent_stats")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    # 1. Initialize DDP
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # Set seed
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print("=" * 60)
        print(" Computing Latent Statistics for VAE Encoder")
        print("=" * 60)
        print(f" Encoder:     {args.encoder_type}")
        print(f" Decoder:     {args.decoder_type}")
        print(f" VAE Ckpt:    {args.vae_ckpt}")
        print(f" Num Samples: {args.num_samples}")
        print(f" Use LoRA:    {args.use_lora}")
        print(f" GPUs:        {world_size}")
        print("=" * 60)
    
    dist.barrier()

    # 2. Load VAE
    if rank == 0:
        print("\nLoading VAE...")
    
    # 使用 get_encoder_config 获取 encoder 配置
    encoder_config = get_encoder_config(args.encoder_type)
    latent_channels = encoder_config["latent_channels"]
    dec_block_out_channels = encoder_config["dec_block_out_channels"]
    default_patch_size = encoder_config["patch_size"]
    
    vae_model_params = {
        "encoder_type": args.encoder_type,
        "image_size": args.image_size,
        "patch_size": default_patch_size,
        "out_channels": 3,
        "latent_channels": latent_channels,
        "target_latent_channels": None,
        "spatial_downsample_factor": default_patch_size,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "decoder_dropout": 0.0,
        "gradient_checkpointing": False,
        "denormalize_decoder_output": False,
    }
    
    # 根据 decoder_type 添加特定参数
    if args.decoder_type == "cnn_decoder":
        vae_model_params["dec_block_out_channels"] = dec_block_out_channels
        vae_model_params["dec_layers_per_block"] = 3
    elif args.decoder_type == "vit_decoder":
        vae_model_params["vit_decoder_hidden_size"] = args.vit_hidden_size
        vae_model_params["vit_decoder_num_layers"] = args.vit_num_layers
        vae_model_params["vit_decoder_num_heads"] = args.vit_num_heads
        vae_model_params["vit_decoder_intermediate_size"] = args.vit_intermediate_size
    
    # 根据 encoder_type 添加模型路径参数
    if args.encoder_type == "dinov3" or args.encoder_type == "dinov3_vitl":
        vae_model_params["dinov3_model_dir"] = args.dinov3_dir
    elif args.encoder_type == "siglip2":
        vae_model_params["siglip2_model_name"] = args.siglip2_model_name
    elif args.encoder_type == "dinov2":
        vae_model_params["dinov2_model_name"] = args.dinov2_model_name
    
    vae = load_vae(
        args.vae_ckpt,
        device,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        model_params=vae_model_params,
        verbose=(rank == 0),
    )
    vae.eval()
    
    # 3. Load dataset
    if rank == 0:
        print("\nLoading dataset...")
    
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),  # [-1, 1]
    ])
    
    dataset = ImageFolder(args.data_path, transform=transform)
    
    if rank == 0:
        print(f"Total dataset size: {len(dataset)}")
    
    # Random subset
    if args.num_samples < len(dataset):
        # Use same random indices across all ranks
        torch.manual_seed(args.seed)  # Same seed for all ranks
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
    
    # Initialize running stats
    # Note: For DINOv2, 256x256 input is resized to 224x224, then patch_size=14 -> 16x16 patches
    # For other encoders (dinov3/siglip2), 256x256 input with patch_size=16 -> 16x16 patches
    if args.encoder_type == "dinov2":
        # DINOv2: resize 256 -> 224, then patch_size=14
        effective_image_size = 224 if args.image_size == 256 else args.image_size
        encoder_patch_size = 14
    else:
        effective_image_size = args.image_size
        encoder_patch_size = 16
    spatial_size = effective_image_size // encoder_patch_size
    stats = RunningStats(shape=(latent_channels,), device=device)
    
    # Also track min/max for debugging
    global_min = torch.full((latent_channels,), float('inf'), device=device)
    global_max = torch.full((latent_channels,), float('-inf'), device=device)
    
    local_count = 0
    
    with torch.no_grad():
        if rank == 0:
            iterator = tqdm(loader, desc="Processing")
        else:
            iterator = loader
        
        for x, _ in iterator:
            x = x.to(device)
            
            # Extract features
            feat = vae.encode_features(x, use_lora=args.use_lora)  # [B, C, H, W]
            
            # Update running statistics
            stats.update_batch(feat)
            
            # Update min/max
            batch_min = feat.amin(dim=[0, 2, 3])
            batch_max = feat.amax(dim=[0, 2, 3])
            global_min = torch.minimum(global_min, batch_min)
            global_max = torch.maximum(global_max, batch_max)
            
            local_count += x.size(0)
    
    # 5. Synchronize across GPUs
    if rank == 0:
        print("\nSynchronizing across GPUs...")
    
    # Gather statistics from all ranks
    # For mean: weighted average
    # For variance: use parallel variance formula
    
    local_n = torch.tensor([stats.n], device=device, dtype=torch.float64)
    local_mean = stats.mean.clone()
    local_M2 = stats.M2.clone()
    
    # All-reduce sum of n
    total_n = local_n.clone()
    dist.all_reduce(total_n, op=dist.ReduceOp.SUM)
    
    # Gather all means and M2 (need all-gather for proper combination)
    all_n = [torch.zeros_like(local_n) for _ in range(world_size)]
    all_mean = [torch.zeros_like(local_mean) for _ in range(world_size)]
    all_M2 = [torch.zeros_like(local_M2) for _ in range(world_size)]
    
    dist.all_gather(all_n, local_n)
    dist.all_gather(all_mean, local_mean)
    dist.all_gather(all_M2, local_M2)
    
    # Combine using parallel algorithm
    combined_n = all_n[0].item()
    combined_mean = all_mean[0].clone()
    combined_M2 = all_M2[0].clone()
    
    for i in range(1, world_size):
        n_a = combined_n
        n_b = all_n[i].item()
        mean_a = combined_mean
        mean_b = all_mean[i]
        M2_a = combined_M2
        M2_b = all_M2[i]
        
        if n_b == 0:
            continue
        
        combined_n = n_a + n_b
        delta = mean_b - mean_a
        combined_mean = mean_a + delta * n_b / combined_n
        combined_M2 = M2_a + M2_b + delta ** 2 * n_a * n_b / combined_n
    
    # Compute final mean and std
    final_mean = combined_mean.float()
    final_var = (combined_M2 / combined_n).float()
    final_std = torch.sqrt(final_var)
    
    # All-reduce min/max
    dist.all_reduce(global_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
    
    # 6. Save results (rank 0 only)
    if rank == 0:
        print("\n" + "=" * 60)
        print(" Statistics Summary")
        print("=" * 60)
        print(f" Total samples processed: {int(combined_n / (spatial_size ** 2))}")
        print(f" Total tokens processed:  {int(combined_n)}")
        print(f" Latent channels:         {latent_channels}")
        print(f" Spatial size:            {spatial_size}x{spatial_size}")
        print("-" * 60)
        print(f" Mean range: [{final_mean.min().item():.6f}, {final_mean.max().item():.6f}]")
        print(f" Std range:  [{final_std.min().item():.6f}, {final_std.max().item():.6f}]")
        print(f" Min value:  {global_min.min().item():.6f}")
        print(f" Max value:  {global_max.max().item():.6f}")
        print("=" * 60)
        
        # Save as numpy
        np.savez(
            os.path.join(args.output_dir, "latent_stats.npz"),
            mean=final_mean.cpu().numpy(),
            std=final_std.cpu().numpy(),
            var=final_var.cpu().numpy(),
            min=global_min.cpu().numpy(),
            max=global_max.cpu().numpy(),
            num_samples=int(combined_n / (spatial_size ** 2)),
            num_tokens=int(combined_n),
        )
        print(f"\nSaved: {os.path.join(args.output_dir, 'latent_stats.npz')}")
        
        # Save as YAML (easier to read and use in configs)
        stats_dict = {
            "encoder_type": args.encoder_type,
            "decoder_type": args.decoder_type,
            "vae_checkpoint": args.vae_ckpt,
            "use_lora": args.use_lora,
            "num_samples": int(combined_n / (spatial_size ** 2)),
            "latent_channels": latent_channels,
            "spatial_size": spatial_size,
            "image_size": args.image_size,
            # Channel-wise statistics (as lists for YAML)
            "mean": final_mean.cpu().tolist(),
            "std": final_std.cpu().tolist(),
            # Summary statistics (scalars)
            "mean_of_mean": float(final_mean.mean().item()),
            "mean_of_std": float(final_std.mean().item()),
            "global_min": float(global_min.min().item()),
            "global_max": float(global_max.max().item()),
            # For easy normalization: shift and scale factors
            "shift_factor": float(final_mean.mean().item()),  # ema_shift_factor style
            "scale_factor": float(final_std.mean().item()),   # ema_scale_factor style
        }
        
        yaml_path = os.path.join(args.output_dir, "latent_stats.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(stats_dict, f, default_flow_style=False, allow_unicode=True)
        print(f"Saved: {yaml_path}")
        
        # Save as PyTorch tensors (for direct loading)
        torch.save({
            "mean": final_mean.cpu(),
            "std": final_std.cpu(),
            "var": final_var.cpu(),
            "min": global_min.cpu(),
            "max": global_max.cpu(),
        }, os.path.join(args.output_dir, "latent_stats.pt"))
        print(f"Saved: {os.path.join(args.output_dir, 'latent_stats.pt')}")
        
        # Print normalization suggestion
        print("\n" + "=" * 60)
        print(" Normalization Suggestion for DiT Training")
        print("=" * 60)
        print(f" To normalize latents: z_norm = (z - mean) / std")
        print(f" To denormalize:       z = z_norm * std + mean")
        print("-" * 60)
        print(f" Simple scalar normalization:")
        print(f"   shift_factor = {stats_dict['shift_factor']:.6f}")
        print(f"   scale_factor = {stats_dict['scale_factor']:.6f}")
        print("=" * 60)
    
    dist.barrier()
    cleanup_ddp()
    
    if rank == 0:
        print("\nDone!")


if __name__ == "__main__":
    main()
