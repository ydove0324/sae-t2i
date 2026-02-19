"""
Test script for DecoSAE: evaluate the impact of different HF branch manipulations.

Manipulations:
1. No change (baseline)
2. Full dropout (all HF tokens zeroed)
3. Random 75% token dropout
4. Replace HF with Gaussian noise
5. Add 0.2 * Gaussian noise to HF
6. Add 0.5 * Gaussian noise to HF

python deco-sae/test_sae.py \
    --config deco-sae/dinov2_base_sae_vit_decoder.yaml \
    --ckpt results_sae/dinov2_base_vit_decoder_hf_dim256_dropout0p4_GAN0p5/step_70000.pth \
    --output test_hf_manipulation.png \
    --num_images 4 \
    --seed 42
"""

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

from config_utils import load_config
from model import DecoSAE

# Add parent directory to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from models.rae.utils.image_utils import center_crop_arr


class FlatImageDataset(torch.utils.data.Dataset):
    """Simple flat image dataset (no subdirectories)."""
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.paths = sorted(
            [
                os.path.join(root, f)
                for f in os.listdir(root)
                if os.path.splitext(f)[1].lower() in self.EXTS
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def build_model(cfg, device: torch.device) -> DecoSAE:
    """Build DecoSAE model from config."""
    return DecoSAE(
        encoder_type=cfg.encoder.type,
        dinov3_model_dir=cfg.encoder.dinov3_model_dir,
        siglip2_model_name=cfg.encoder.siglip2_model_name,
        dinov2_model_name=cfg.encoder.dinov2_model_name,
        image_size=cfg.data.image_size,
        in_channels=3,
        out_channels=3,
        hidden_size=cfg.model.hidden_size,
        hidden_size_x=cfg.model.hidden_size_x,
        decoder_type=getattr(cfg.model, "decoder_type", "flow_matching"),
        num_decoder_blocks=cfg.model.num_decoder_blocks,
        nerf_max_freqs=cfg.model.nerf_max_freqs,
        flow_steps=cfg.model.flow_steps,
        time_dim=cfg.model.time_dim,
        vit_decoder_hidden_size=getattr(cfg.model, "vit_decoder_hidden_size", 1024),
        vit_decoder_num_layers=getattr(cfg.model, "vit_decoder_num_layers", 24),
        vit_decoder_num_heads=getattr(cfg.model, "vit_decoder_num_heads", 16),
        vit_decoder_intermediate_size=getattr(cfg.model, "vit_decoder_intermediate_size", 4096),
        vit_decoder_dropout=getattr(cfg.model, "vit_decoder_dropout", 0.0),
        gradient_checkpointing=getattr(cfg.model, "gradient_checkpointing", False),
        enable_hf_branch=cfg.model.enable_hf_branch,
        hf_dim=cfg.model.hf_dim,
        hf_encoder_config_path=getattr(cfg.model, "hf_encoder_config_path", None),
        hf_dropout_prob=0.0,  # Disable dropout for testing
        hf_noise_std=0.0,     # Disable noise for testing
        hf_loss_weight=cfg.model.hf_loss_weight,
        recon_l2_weight=cfg.loss.recon_l2_weight,
        recon_l1_weight=cfg.loss.recon_l1_weight,
        recon_lpips_weight=cfg.loss.recon_lpips_weight,
        recon_gan_weight=cfg.loss.recon_gan_weight,
        lora_rank=cfg.model.lora_rank,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        enable_lora=cfg.model.enable_lora,
        target_latent_channels=cfg.model.target_latent_channels,
        variational=cfg.model.variational,
        kl_weight=cfg.model.kl_weight,
        skip_to_moments=cfg.model.skip_to_moments,
        noise_tau=0.0,  # Disable noise for testing
        random_masking_channel_ratio=0.0,
        denormalize_decoder_output=cfg.model.denormalize_decoder_output,
    ).to(device)


def load_checkpoint(model: DecoSAE, ckpt_path: str, device: torch.device):
    """Load model checkpoint."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in payload:
        state_dict = payload["model"]
    elif "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    return payload.get("step", 0) if isinstance(payload, dict) else 0


def manipulate_hf_tokens(
    s_hf: torch.Tensor,
    mode: str,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Apply different manipulations to HF tokens.
    
    Args:
        s_hf: [B, N, hf_dim] HF tokens
        mode: manipulation mode
        generator: random generator for reproducibility
        
    Returns:
        Manipulated HF tokens with same shape
    """
    B, N, C = s_hf.shape
    device = s_hf.device
    dtype = s_hf.dtype
    
    if mode == "original":
        # No change
        return s_hf
    
    elif mode == "full_dropout":
        # All HF tokens zeroed
        return torch.zeros_like(s_hf)
    
    elif mode == "random_dropout_75":
        # Randomly drop 75% of tokens (per sample)
        keep_mask = torch.zeros((B, N, 1), device=device, dtype=dtype)
        for b in range(B):
            # Keep 25% of tokens
            num_keep = max(1, int(N * 0.25))
            keep_indices = torch.randperm(N, generator=generator, device=device)[:num_keep]
            keep_mask[b, keep_indices, :] = 1.0
        return s_hf * keep_mask
    
    elif mode == "gaussian_replace":
        # Replace with Gaussian noise (same std as original HF tokens)
        std = s_hf.std()
        mean = s_hf.mean()
        print(f"Gaussian replace: std={std}, mean={mean}")
        noise = torch.randn_like(s_hf) * std * 0.5 + mean
        return noise
    
    elif mode == "gaussian_add_0.2":
        # Add 0.2 * Gaussian noise
        std = s_hf.std()
        noise = torch.randn_like(s_hf) * std * 0.2
        return s_hf + noise
    
    elif mode == "gaussian_add_0.5":
        # Add 0.5 * Gaussian noise
        std = s_hf.std()
        noise = torch.randn_like(s_hf) * std * 0.5
        return s_hf + noise
    
    else:
        raise ValueError(f"Unknown manipulation mode: {mode}")


@torch.no_grad()
def generate_with_manipulation(
    model: DecoSAE,
    x: torch.Tensor,
    mode: str,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Generate reconstruction with HF manipulation.
    
    Args:
        model: DecoSAE model
        x: [B, 3, H, W] input images
        mode: HF manipulation mode
        generator: random generator
        
    Returns:
        [B, 3, H, W] reconstructed images
    """
    model.eval()
    device = x.device
    dtype = x.dtype
    
    # 1. Get semantic latent from DINOv2 (unchanged)
    enc = model._infer_latent(x)
    z = enc.latent  # [B, C, H_p, W_p]
    
    # 2. Extract semantic tokens
    s_sem = model._encoder_semantic(z)  # [B, N, semantic_channels]
    bsz, num_patches, _ = s_sem.shape
    s_h, s_w = z.shape[-2:]
    
    # 3. Extract original HF tokens
    hf_feat = model.hf_encoder(x)
    if hf_feat.shape[-2:] != (s_h, s_w):
        hf_feat = F.interpolate(hf_feat, size=(s_h, s_w), mode="bilinear", align_corners=False)
    s_hf_original = hf_feat.flatten(2).transpose(1, 2).contiguous()  # [B, N, hf_dim]
    
    # 4. Apply manipulation to HF tokens
    s_hf_manipulated = manipulate_hf_tokens(s_hf_original, mode, generator)
    
    # 5. Fuse and generate
    enc_cond = model.fused_norm(torch.cat([s_sem, s_hf_manipulated], dim=-1))
    recon = model.generate(enc_cond)
    
    return recon.clamp(-1.0, 1.0)


def create_visualization(
    original_images: torch.Tensor,
    reconstructions: Dict[str, torch.Tensor],
    output_path: str,
    nrow: int = 4,
):
    """
    Create visualization grid comparing original and different reconstructions.
    
    Layout: Each row shows one image across all manipulations.
    Columns: Original | Baseline | Full Dropout | 75% Dropout | Gaussian Replace | +0.2 Noise | +0.5 Noise
    """
    B = original_images.shape[0]
    
    # Convert from [-1, 1] to [0, 1]
    original_vis = (original_images + 1.0) / 2.0
    
    manipulation_order = [
        "original",
        "full_dropout",
        "random_dropout_75",
        "gaussian_replace",
        "gaussian_add_0.2",
        "gaussian_add_0.5",
    ]
    
    manipulation_names = {
        "original": "Original",
        "full_dropout": "HF Full Drop",
        "random_dropout_75": "HF 75% Drop",
        "gaussian_replace": "HF -> Noise",
        "gaussian_add_0.2": "HF + 0.2*Noise",
        "gaussian_add_0.5": "HF + 0.5*Noise",
    }
    
    # Create list of all images in row-major order
    all_images = []
    for b in range(B):
        # Add original image
        all_images.append(original_vis[b])
        # Add reconstructions
        for mode in manipulation_order:
            recon = (reconstructions[mode][b] + 1.0) / 2.0
            all_images.append(recon)
    
    # Stack all images
    all_images = torch.stack(all_images, dim=0)
    
    # Create grid: B rows, 7 columns (original + 6 manipulations)
    num_cols = 1 + len(manipulation_order)
    grid = make_grid(all_images, nrow=num_cols, padding=4, normalize=False, pad_value=1.0)
    
    # Convert to PIL and save
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    img = Image.fromarray(grid_np)
    
    # Add text labels (create a larger image with header)
    from PIL import ImageDraw, ImageFont
    
    header_height = 40
    new_img = Image.new("RGB", (img.width, img.height + header_height), (255, 255, 255))
    new_img.paste(img, (0, header_height))
    
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # Calculate column width
    col_width = img.width // num_cols
    
    # Draw column labels
    labels = ["GT"] + [manipulation_names[m] for m in manipulation_order]
    for i, label in enumerate(labels):
        x = i * col_width + col_width // 2
        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text((x - text_width // 2, 10), label, fill=(0, 0, 0), font=font)
    
    new_img.save(output_path)
    print(f"Saved visualization to {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Test DecoSAE with different HF manipulations")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="test_hf_manipulation.png", help="Output image path")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    cfg = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # Build model
    model = build_model(cfg, device)
    print(f"Built model with decoder_type={model.decoder_type}")
    
    # Load checkpoint
    step = load_checkpoint(model, args.ckpt, device)
    print(f"Checkpoint step: {step}")
    
    # Build validation dataset
    val_transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, cfg.data.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),
    ])
    
    try:
        val_dataset = ImageFolder(cfg.data.val_path, transform=val_transform)
    except Exception:
        val_dataset = FlatImageDataset(cfg.data.val_path, transform=val_transform)
    print(f"Loaded validation dataset with {len(val_dataset)} images")
    
    # Sample random images
    indices = torch.randperm(len(val_dataset), generator=torch.Generator().manual_seed(args.seed))[:args.num_images]
    images = torch.stack([val_dataset[i][0] for i in indices], dim=0).to(device)
    print(f"Sampled {len(indices)} images for visualization")
    
    # Define manipulation modes
    modes = [
        "original",
        "full_dropout",
        "random_dropout_75",
        "gaussian_replace",
        "gaussian_add_0.2",
        "gaussian_add_0.5",
    ]
    
    # Generate reconstructions for each mode
    reconstructions = {}
    for mode in modes:
        print(f"Generating with mode: {mode}")
        recon = generate_with_manipulation(model, images, mode, generator)
        reconstructions[mode] = recon
    
    # Create visualization
    create_visualization(images, reconstructions, args.output)
    
    # Print some statistics
    print("\n=== Reconstruction Statistics ===")
    for mode in modes:
        recon = reconstructions[mode]
        mse = F.mse_loss(recon, images).item()
        psnr = -10 * math.log10(mse) if mse > 0 else float('inf')
        print(f"{mode:20s}: MSE={mse:.6f}, PSNR={psnr:.2f} dB")


if __name__ == "__main__":
    main()
