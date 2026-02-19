"""
Test script for DiT on DECO-SAE latent space.
Supports masking HF branch during decoding to evaluate its impact on generation quality.

Usage:
python deco-sae/test_dit.py \
    --config configs/dit/deco_dinov2_base_dit_xl.yaml \
    --sae-config deco-sae/dinov2_base_sae_vit_decoder.yaml \
    --sae-ckpt results_sae/xxx/step_xxx.pth \
    --dit-ckpt results_dit/xxx/checkpoints/xxx.pt \
    --fid-ref-path VIRTUAL_imagenet256_labeled.npz \
    --output-dir results_dit/test_hf_mask \
    --mask-hf  # Enable HF masking
"""

import argparse
import math
import os
import shutil
import sys
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import save_image

# FID imports
try:
    from pytorch_fid import fid_score
    HAS_FID = True
except ImportError:
    HAS_FID = False
    print("Warning: 'pytorch-fid' not found. FID calculation will be skipped.")

# Inception Score and Precision/Recall imports
try:
    from torch_fidelity import calculate_metrics
    HAS_TORCH_FIDELITY = True
except ImportError:
    HAS_TORCH_FIDELITY = False
    print("Warning: 'torch-fidelity' not found. IS and Recall calculations will be skipped.")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config_utils import load_config
from model import DecoSAE
from models.rae.utils.ddp_utils import cleanup_ddp, setup_ddp
from models.rae.utils.model_utils import instantiate_from_config
from models.rae.utils.train_utils import parse_configs


#################################################################################
#                              Model Building                                   #
#################################################################################

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
        gradient_checkpointing=getattr(sae_cfg.model, "gradient_checkpointing", False),
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


def load_sae_checkpoint(model: DecoSAE, ckpt_path: str, device: torch.device):
    """Load DECO-SAE checkpoint."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")

    # Load to CPU first to avoid OOM when multiple processes load to same GPU
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in payload:
        state_dict = payload["model"]
    elif "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    step = payload.get("step", 0) if isinstance(payload, dict) else 0
    print(f"Loaded SAE checkpoint from {ckpt_path} (step={step})")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    return step


def load_dit_checkpoint(model, ckpt_path: str, device: torch.device):
    """Load DiT checkpoint (EMA weights)."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"DiT checkpoint not found: {ckpt_path}")

    # Load to CPU first to avoid OOM when multiple processes load to same GPU
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Prefer EMA weights
    if "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"], strict=False)
        print(f"Loaded EMA weights from {ckpt_path}")
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
        print(f"Loaded model weights from {ckpt_path}")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded weights from {ckpt_path}")
    
    train_steps = checkpoint.get("train_steps", 0) if isinstance(checkpoint, dict) else 0
    print(f"DiT checkpoint train_steps: {train_steps}")
    return train_steps


#################################################################################
#                              Sampling Functions                               #
#################################################################################

@torch.no_grad()
def sample_latent(
    model,
    batch_size: int,
    latent_shape: tuple,
    device: torch.device,
    y: torch.Tensor,
    time_shift: float,
    steps: int = 50,
    use_cfg: bool = False,
    cfg_scale: float = 3.0,
    null_class: int = 1000,
):
    """Sample latent z0 using Euler method."""
    model.eval()

    C, H, W = latent_shape
    x = torch.randn((batch_size, C, H, W), device=device, dtype=torch.float32)

    shift = float(time_shift)

    def flow_shift(t_lin: torch.Tensor) -> torch.Tensor:
        t = (shift * t_lin) / (1.0 + (shift - 1.0) * t_lin)
        return t.clamp(0.0, 1.0 - 1e-6)

    for i in range(steps, 0, -1):
        t_lin = torch.full((batch_size,), i / steps, device=device, dtype=torch.float32)
        t_next_lin = torch.full((batch_size,), (i - 1) / steps, device=device, dtype=torch.float32)

        t = flow_shift(t_lin)
        t_next = flow_shift(t_next_lin)

        if not use_cfg:
            x0_hat = model(x, t, y=y)
        else:
            y_cond = y
            y_uncond = torch.full_like(y, null_class)
            x0_hat_cond = model(x, t, y=y_cond)
            x0_hat_uncond = model(x, t, y=y_uncond)
            x0_hat = x0_hat_uncond + cfg_scale * (x0_hat_cond - x0_hat_uncond)

        t_scalar = t.view(batch_size, 1, 1, 1)
        eps_hat = (x - (1.0 - t_scalar) * x0_hat) / (t_scalar + 1e-8)

        t_next_scalar = t_next.view(batch_size, 1, 1, 1)
        x = t_next_scalar * eps_hat + (1.0 - t_next_scalar) * x0_hat

    return x


@torch.no_grad()
def decode_with_deco_sae(
    sae_model: DecoSAE,
    latent: torch.Tensor,
    mask_hf: bool = False,
    hf_mask_mode: str = "zero",
) -> torch.Tensor:
    """
    Decode latent using DECO-SAE.
    
    Args:
        sae_model: DECO-SAE model
        latent: [B, C, H, W] fused latent (semantic + HF)
        mask_hf: If True, mask the HF branch
        hf_mask_mode: How to mask HF:
            - "zero": Set HF to zero
            - "noise": Replace HF with Gaussian noise
            - "mean": Replace HF with mean value
            
    Returns: [B, 3, image_size, image_size] reconstructed images
    """
    sae_model.eval()
    
    B, C, H, W = latent.shape
    semantic_channels = sae_model.semantic_channels
    hf_dim = sae_model.hf_dim
    
    # Reshape to [B, N, C]
    enc_cond = latent.view(B, C, H * W).transpose(1, 2).contiguous()  # [B, N, C]
    
    if mask_hf and hf_dim > 0:
        # Split into semantic and HF
        s_sem = enc_cond[:, :, :semantic_channels]  # [B, N, semantic_channels]
        s_hf = enc_cond[:, :, semantic_channels:]   # [B, N, hf_dim]
        
        # Mask HF based on mode
        if hf_mask_mode == "zero":
            s_hf_masked = torch.zeros_like(s_hf)
        elif hf_mask_mode == "noise":
            std = s_hf.std()
            mean = s_hf.mean()
            s_hf_masked = torch.randn_like(s_hf) * std + mean
        elif hf_mask_mode == "mean":
            s_hf_masked = s_hf.mean(dim=(0, 1), keepdim=True).expand_as(s_hf)
        else:
            raise ValueError(f"Unknown hf_mask_mode: {hf_mask_mode}")
        
        # Recombine
        enc_cond = torch.cat([s_sem, s_hf_masked], dim=-1)
    
    # Generate image
    img = sae_model.generate(enc_cond)
    return img.clamp(-1.0, 1.0)


#################################################################################
#                              FID Evaluation                                   #
#################################################################################

def create_flat_temp_dir(source_root, temp_dir):
    """Flatten the directory structure for pytorch-fid."""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    count = 0
    for root, dirs, files in os.walk(source_root):
        if os.path.abspath(root) == os.path.abspath(temp_dir):
            continue
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.abspath(os.path.join(root, file))
                parent_name = os.path.basename(root)
                new_name = f"{parent_name}_{file}"
                dst_path = os.path.join(temp_dir, new_name)
                try:
                    os.symlink(src_path, dst_path)
                    count += 1
                except OSError:
                    shutil.copy(src_path, dst_path)
                    count += 1
    return count


@torch.no_grad()
def generate_samples_and_compute_fid(
    dit_model,
    sae_model: DecoSAE,
    output_dir: str,
    fid_ref_path: str,
    samples_per_class: int,
    batch_size: int,
    latent_shape: tuple,
    time_shift: float,
    device: torch.device,
    rank: int,
    world_size: int,
    mask_hf: bool = False,
    hf_mask_mode: str = "zero",
    use_cfg: bool = False,
    cfg_scale: float = 3.0,
    num_classes: int = 1000,
    sample_steps: int = 50,
    ref_images_path: str = None,
):
    """
    Generate samples and compute FID/IS/Precision/Recall (distributed version).
    
    Args:
        dit_model: DiT model
        sae_model: DECO-SAE model for decoding
        output_dir: Directory to save generated samples
        fid_ref_path: Path to reference FID statistics (.npz)
        samples_per_class: Number of samples per class
        batch_size: Batch size for generation
        latent_shape: Shape of latent (C, H, W)
        time_shift: Time shift for flow matching
        device: Device to use
        rank: Current process rank
        world_size: Total number of processes
        mask_hf: Whether to mask HF branch during decoding
        hf_mask_mode: How to mask HF ("zero", "noise", "mean")
        use_cfg: Whether to use classifier-free guidance
        cfg_scale: CFG scale
        num_classes: Number of classes
        sample_steps: Number of sampling steps
        ref_images_path: Path to reference images for Precision/Recall
    """
    # Create output directories
    suffix = f"_mask_hf_{hf_mask_mode}" if mask_hf else "_no_mask"
    if use_cfg:
        suffix += f"_cfg{cfg_scale}"
    
    eval_dir = os.path.join(output_dir, f"samples{suffix}")
    flat_dir = os.path.join(output_dir, f"samples{suffix}_flat")
    
    if rank == 0:
        os.makedirs(eval_dir, exist_ok=True)
    
    # Barrier to ensure directory exists before other ranks proceed
    if world_size > 1:
        dist.barrier()
    
    if rank == 0:
        print(f"========== Starting Distributed Generation ==========")
        print(f"Output directory: {eval_dir}")
        print(f"Mask HF: {mask_hf}, Mode: {hf_mask_mode}")
        print(f"CFG: {use_cfg}, Scale: {cfg_scale}")
        print(f"Samples per class: {samples_per_class}, Total: {samples_per_class * num_classes}")
        print(f"World size: {world_size}")
    
    dit_model.eval()
    sae_model.eval()
    
    # Distribute classes across ranks
    all_classes = list(range(num_classes))
    my_classes = all_classes[rank::world_size]
    
    if rank == 0:
        print(f"Each rank handles ~{len(my_classes)} classes")
    
    total_generated = 0
    
    for idx, cls_id in enumerate(my_classes):
        class_dir = os.path.join(eval_dir, str(cls_id))
        os.makedirs(class_dir, exist_ok=True)
        
        generated_count = 0
        while generated_count < samples_per_class:
            curr_batch = min(batch_size, samples_per_class - generated_count)
            
            y = torch.full((curr_batch,), cls_id, device=device, dtype=torch.long)
            
            # Sample latent
            z_sample = sample_latent(
                model=dit_model,
                batch_size=curr_batch,
                latent_shape=latent_shape,
                device=device,
                y=y,
                time_shift=time_shift,
                steps=sample_steps,
                use_cfg=use_cfg,
                cfg_scale=cfg_scale,
            )
            
            # Decode with optional HF masking
            imgs = decode_with_deco_sae(
                sae_model, 
                z_sample.float(),
                mask_hf=mask_hf,
                hf_mask_mode=hf_mask_mode,
            )
            
            # Convert to [0, 1] range
            imgs = (imgs + 1.0) / 2.0
            imgs = imgs.clamp(0, 1)
            
            # Save images
            for i in range(curr_batch):
                img_idx = generated_count + i
                save_image(imgs[i], os.path.join(class_dir, f"{img_idx}.png"))
            
            generated_count += curr_batch
        
        total_generated += generated_count
        
        if (idx + 1) % 50 == 0:
            print(f"[Rank {rank}] Generated {idx + 1}/{len(my_classes)} classes, {total_generated} images")
    
    print(f"[Rank {rank}] Generation finished. Total images on this rank: {total_generated}")
    
    # Synchronize all ranks before computing metrics
    if world_size > 1:
        dist.barrier()
    
    # Only rank 0 computes metrics
    results = {}
    if rank == 0:
        print("\nAll ranks finished generation. Computing metrics on rank 0...")
        
        print("Flattening directory for evaluation...")
        num_imgs = create_flat_temp_dir(eval_dir, flat_dir)
        print(f"Total images for evaluation: {num_imgs}")
        
        if num_imgs < 100:
            print(f"Warning: Too few images ({num_imgs}) for valid metrics.")
        else:
            # FID calculation
            if HAS_FID and fid_ref_path is not None and os.path.exists(fid_ref_path):
                print(f"Calculating FID using {num_imgs} images...")
                try:
                    fid_value = fid_score.calculate_fid_given_paths(
                        paths=[flat_dir, fid_ref_path],
                        batch_size=50,
                        device=device,
                        dims=2048,
                        num_workers=8
                    )
                    results["FID"] = fid_value
                    print(f"FID: {fid_value:.4f}")
                except Exception as e:
                    print(f"FID calculation failed: {e}")
            
            # IS, Precision, Recall calculation
            if HAS_TORCH_FIDELITY:
                print(f"Calculating IS using {num_imgs} images...")
                try:
                    is_metrics = calculate_metrics(
                        input1=flat_dir,
                        cuda=True,
                        isc=True,
                        isc_splits=10,
                        verbose=False,
                    )
                    is_value = is_metrics.get('inception_score_mean', None)
                    is_std = is_metrics.get('inception_score_std', None)
                    if is_value is not None:
                        results["IS"] = is_value
                        results["IS_std"] = is_std
                        print(f"IS: {is_value:.4f} ± {is_std:.4f}")
                    
                    # Precision and Recall
                    if ref_images_path is not None and os.path.exists(ref_images_path):
                        print(f"Calculating Precision/Recall with reference: {ref_images_path}")
                        pr_metrics = calculate_metrics(
                            input1=flat_dir,
                            input2=ref_images_path,
                            cuda=True,
                            prc=True,
                            verbose=False,
                        )
                        precision_value = pr_metrics.get('precision', None)
                        recall_value = pr_metrics.get('recall', None)
                        if precision_value is not None:
                            results["Precision"] = precision_value
                            print(f"Precision: {precision_value:.4f}")
                        if recall_value is not None:
                            results["Recall"] = recall_value
                            print(f"Recall: {recall_value:.4f}")
                except Exception as e:
                    print(f"torch-fidelity metrics calculation failed: {e}")
        
        # Clean up flat directory
        if os.path.exists(flat_dir):
            shutil.rmtree(flat_dir)
        
        # Save results to file
        results_file = os.path.join(output_dir, f"metrics{suffix}.txt")
        with open(results_file, "w") as f:
            f.write(f"Mask HF: {mask_hf}\n")
            f.write(f"HF Mask Mode: {hf_mask_mode}\n")
            f.write(f"CFG: {use_cfg}, Scale: {cfg_scale}\n")
            f.write(f"Samples per class: {samples_per_class}\n")
            f.write(f"Total images: {samples_per_class * num_classes}\n")
            f.write(f"World size: {world_size}\n")
            f.write(f"\nMetrics:\n")
            for k, v in results.items():
                f.write(f"  {k}: {v:.4f}\n")
        print(f"Results saved to {results_file}")
    
    # Final barrier before returning
    if world_size > 1:
        dist.barrier()
    
    return results


#################################################################################
#                              Main Function                                    #
#################################################################################

def main():
    parser = argparse.ArgumentParser(description="Test DiT with HF masking (Distributed)")
    parser.add_argument("--config", type=str, required=True, help="DiT config path")
    parser.add_argument("--sae-config", type=str, required=True, help="DECO-SAE config path")
    parser.add_argument("--sae-ckpt", type=str, required=True, help="DECO-SAE checkpoint path")
    parser.add_argument("--dit-ckpt", type=str, required=True, help="DiT checkpoint path")
    parser.add_argument("--output-dir", type=str, default="results_dit/test_hf_mask", help="Output directory")
    
    # HF masking options
    parser.add_argument("--mask-hf", action="store_true", help="Mask HF branch during decoding")
    parser.add_argument("--hf-mask-mode", type=str, default="zero", choices=["zero", "noise", "mean"],
                        help="How to mask HF: zero, noise, or mean")
    
    # CFG options
    parser.add_argument("--use-cfg", action="store_true", help="Use classifier-free guidance")
    parser.add_argument("--cfg-scale", type=float, default=1.5, help="CFG scale")
    
    # Generation options
    parser.add_argument("--samples-per-class", type=int, default=50, help="Samples per class")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for generation")
    parser.add_argument("--sample-steps", type=int, default=50, help="Number of sampling steps")
    
    # FID options
    parser.add_argument("--fid-ref-path", type=str, default="VIRTUAL_imagenet256_labeled.npz",
                        help="Path to reference FID statistics (.npz)")
    parser.add_argument("--ref-images-path", type=str, default=None,
                        help="Path to reference images for Precision/Recall")
    
    # Other options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Initialize distributed environment
    rank, local_rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank)
    
    # Set seed (different for each rank to ensure diverse samples)
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if rank == 0:
        print(f"Distributed Test: rank={rank}, world_size={world_size}")
        print(f"Using device: {device}")
    
    # Create output directory (only rank 0)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse DiT config
    (
        _,
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config,
    ) = parse_configs(args.config)
    
    def to_dict(cfg_section):
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)
    
    misc = to_dict(misc_config)
    
    # Get latent size and time shift
    latent_size_cfg = misc.get("latent_size", [832, 16, 16])
    latent_size = tuple(latent_size_cfg)
    
    shift_dim = latent_size[0] * latent_size[1] * latent_size[2]
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_shift = math.sqrt(shift_dim / shift_base)
    
    num_classes = int(misc.get("num_classes", 1000))
    
    if rank == 0:
        print(f"Latent size: {latent_size}, time_shift: {time_shift:.4f}")
    
    # Load DECO-SAE
    if rank == 0:
        print(f"\nLoading DECO-SAE from {args.sae_ckpt}...")
    sae_cfg = load_config(args.sae_config)
    sae_model = build_deco_sae(sae_cfg, device)
    load_sae_checkpoint(sae_model, args.sae_ckpt, device)
    sae_model.eval()
    if rank == 0:
        print(f"DECO-SAE loaded. Semantic channels: {sae_model.semantic_channels}, HF dim: {sae_model.hf_dim}")
    
    # Build and load DiT model
    if rank == 0:
        print(f"\nLoading DiT from {args.dit_ckpt}...")
    dit_model = instantiate_from_config(model_config).to(device)
    load_dit_checkpoint(dit_model, args.dit_ckpt, device)
    dit_model.eval()
    
    if rank == 0:
        dit_param_count = sum(p.numel() for p in dit_model.parameters())
        print(f"DiT Parameters: {dit_param_count / 1e6:.2f}M")
    
    # Synchronize before starting
    if world_size > 1:
        dist.barrier()
    
    # Run evaluation
    if rank == 0:
        print("\n" + "=" * 60)
        print("Starting distributed evaluation...")
        print("=" * 60)
    
    results = generate_samples_and_compute_fid(
        dit_model=dit_model,
        sae_model=sae_model,
        output_dir=args.output_dir,
        fid_ref_path=args.fid_ref_path,
        samples_per_class=args.samples_per_class,
        batch_size=args.batch_size,
        latent_shape=latent_size,
        time_shift=time_shift,
        device=device,
        rank=rank,
        world_size=world_size,
        mask_hf=args.mask_hf,
        hf_mask_mode=args.hf_mask_mode,
        use_cfg=args.use_cfg,
        cfg_scale=args.cfg_scale,
        num_classes=num_classes,
        sample_steps=args.sample_steps,
        ref_images_path=args.ref_images_path,
    )
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("Final Results:")
        print("=" * 60)
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        print("\nDone!")
    
    # Cleanup
    cleanup_ddp()


if __name__ == "__main__":
    main()
