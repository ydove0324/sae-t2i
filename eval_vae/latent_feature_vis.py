# latent_feature_vis.py
# Visualize latent features as RGB images (like the paper figure)
# Maps high-dim latent to 3D via PCA, then displays as RGB

import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

# 添加当前目录以导入项目模块
sys.path.append(".")

# 导入 VAE 工具函数
from models.rae.utils.vae_utils import load_vae


# ==========================================
#              Helper Functions
# ==========================================

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def latent_to_rgb(latent: torch.Tensor, pca_model=None, fit_pca=False) -> tuple:
    """
    Convert latent features to RGB image using PCA.
    
    Args:
        latent: [C, H, W] latent feature map
        pca_model: Pre-fitted PCA model (optional)
        fit_pca: Whether to fit PCA on this latent
    
    Returns:
        rgb_image: [H, W, 3] normalized to [0, 1]
        pca_model: Fitted PCA model
    """
    C, H, W = latent.shape
    
    # Reshape to [H*W, C]
    latent_flat = latent.permute(1, 2, 0).reshape(-1, C).cpu().numpy()
    
    if pca_model is None or fit_pca:
        # Fit PCA to reduce to 3 dimensions
        pca_model = PCA(n_components=3)
        latent_3d = pca_model.fit_transform(latent_flat)
    else:
        latent_3d = pca_model.transform(latent_flat)
    
    # Reshape back to [H, W, 3]
    rgb = latent_3d.reshape(H, W, 3)
    
    # Normalize to [0, 1] for visualization
    rgb_min = rgb.min()
    rgb_max = rgb.max()
    rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)
    
    return rgb, pca_model


def latent_to_rgb_global(latent: torch.Tensor, global_min: np.ndarray, global_max: np.ndarray, pca_model) -> np.ndarray:
    """
    Convert latent to RGB using global normalization (for consistent colors across images).
    """
    C, H, W = latent.shape
    latent_flat = latent.permute(1, 2, 0).reshape(-1, C).cpu().numpy()
    latent_3d = pca_model.transform(latent_flat)
    rgb = latent_3d.reshape(H, W, 3)
    
    # Use global min/max for normalization
    rgb = (rgb - global_min) / (global_max - global_min + 1e-8)
    rgb = np.clip(rgb, 0, 1)
    
    return rgb


@torch.no_grad()
def extract_latent(vae, x: torch.Tensor, use_lora: bool = True) -> torch.Tensor:
    """Extract latent features from VAE encoder."""
    feat = vae.encode_features(x, use_lora=use_lora)  # [B, C, H, W]
    return feat


# ==========================================
#                   Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Visualize latent features as RGB images")
    
    # Data
    parser.add_argument("--data-path", type=str, required=True, help="Path to ImageNet validation set.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-images", type=int, default=10, help="Number of images to visualize.")
    
    # VAE
    parser.add_argument("--vae-ckpt", type=str, required=True, help="Path to VAE checkpoint.")
    parser.add_argument("--encoder-type", type=str, default="dinov3", choices=["dinov3", "siglip2"])
    parser.add_argument("--dinov3-dir", type=str, default="/cpfs01/huangxu/models/dinov3")
    parser.add_argument("--siglip2-model-name", type=str, default="google/siglip2-base-patch16-256")
    parser.add_argument("--lora-rank", type=int, default=256)
    parser.add_argument("--lora-alpha", type=int, default=256)
    
    # Visualization options
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--global-norm", action="store_true", 
                        help="Use global normalization for consistent colors across images.")
    parser.add_argument("--upscale", type=int, default=16, 
                        help="Upscale factor for latent visualization (latent is 16x16 for 256 input).")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="results/latent_vis")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print(" Latent Feature Visualization")
    print("=" * 60)
    print(f" Encoder:     {args.encoder_type}")
    print(f" VAE Ckpt:    {args.vae_ckpt}")
    print(f" Use LoRA:    {args.use_lora}")
    print(f" Num Images:  {args.num_images}")
    print("=" * 60)
    
    # 1. Load VAE
    print("\nLoading VAE...")
    
    if args.encoder_type == "dinov3":
        latent_channels = 1280
        dec_block_out_channels = (1280, 1024, 512, 256, 128)
    else:  # siglip2
        latent_channels = 768
        dec_block_out_channels = (768, 512, 256, 128, 64)
    
    vae_model_params = {
        "encoder_type": args.encoder_type,
        "image_size": args.image_size,
        "patch_size": 16,
        "out_channels": 3,
        "latent_channels": latent_channels,
        "target_latent_channels": None,
        "spatial_downsample_factor": 16,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "dec_block_out_channels": dec_block_out_channels,
        "dec_layers_per_block": 3,
        "decoder_dropout": 0.0,
        "gradient_checkpointing": False,
        "denormalize_decoder_output": True,
        "skip_to_moments": True,
        "denormalize_decoder_output": True,
        "skip_to_moments": True,
    }
    
    if args.encoder_type == "dinov3":
        vae_model_params["dinov3_model_dir"] = args.dinov3_dir
    elif args.encoder_type == "siglip2":
        vae_model_params["siglip2_model_name"] = args.siglip2_model_name
    
    vae = load_vae(
        args.vae_ckpt,
        device,
        encoder_type=args.encoder_type,
        decoder_type="cnn_decoder",
        model_params=vae_model_params,
        verbose=True,
        skip_to_moments=True
        skip_to_moments=False
    )
    vae.eval()
    
    # 2. Load dataset
    print("\nLoading dataset...")
    
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),  # [-1, 1]
    ])
    
    dataset = ImageFolder(args.data_path, transform=transform)
    
    # Random sample
    indices = random.sample(range(len(dataset)), min(args.num_images, len(dataset)))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False)
    
    print(f"Selected {len(indices)} images")
    
    # 3. First pass: fit global PCA if needed
    print("\nExtracting latent features...")
    
    all_latents = []
    all_images = []
    
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Extracting"):
            x = x.to(device)
            latent = extract_latent(vae, x, use_lora=args.use_lora)  # [1, C, H, W]
            all_latents.append(latent[0])  # [C, H, W]
            
            # Store original image
            img_np = ((x[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
            all_images.append(img_np)
    
    # Fit global PCA on all latents
    print("\nFitting PCA...")
    
    # Stack all latents and reshape for PCA
    all_latents_stacked = torch.stack(all_latents)  # [N, C, H, W]
    N, C, H, W = all_latents_stacked.shape
    all_latents_flat = all_latents_stacked.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()  # [N*H*W, C]
    
    pca_model = PCA(n_components=3)
    all_latents_3d = pca_model.fit_transform(all_latents_flat)  # [N*H*W, 3]
    
    # Get global min/max for consistent normalization
    global_min = all_latents_3d.min(axis=0)
    global_max = all_latents_3d.max(axis=0)
    
    print(f"  PCA explained variance: {pca_model.explained_variance_ratio_}")
    print(f"  Global range: [{global_min}, {global_max}]")
    
    # 4. Create visualization
    print("\nCreating visualizations...")
    
    # Individual images
    latent_vis_list = []
    for i, latent in enumerate(all_latents):
        if args.global_norm:
            rgb = latent_to_rgb_global(latent, global_min, global_max, pca_model)
        else:
            rgb, _ = latent_to_rgb(latent, pca_model, fit_pca=False)
        
        # Upscale latent visualization to match original image size
        rgb_upscaled = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize(
            (args.image_size, args.image_size), Image.NEAREST
        ))
        latent_vis_list.append(rgb_upscaled)
        
        # Save individual
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        axes[0].imshow(all_images[i])
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(rgb_upscaled)
        axes[1].set_title(f"Latent Feature (PCA→RGB)", fontsize=12)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"latent_vis_{i:03d}.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 5. Create grid visualization (like the paper figure)
    print("\nCreating grid visualization...")
    
    n_cols = min(5, args.num_images)
    n_rows = 2  # Original + Latent
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    
    for i in range(min(n_cols, len(all_images))):
        # Original image
        axes[0, i].imshow(all_images[i])
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel("Image", fontsize=14, rotation=0, labelpad=50, va='center')
        
        # Latent visualization
        axes[1, i].imshow(latent_vis_list[i])
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel(f"{args.encoder_type}", fontsize=14, rotation=0, labelpad=50, va='center')
    
    plt.suptitle("Latent Feature Visualization (PCA → RGB)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "latent_grid.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 6. Create comparison with reconstruction (if decoder available)
    print("\nCreating reconstruction comparison...")
    
    try:
        recon_list = []
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                x = x.to(device)
                # Encode and decode
                out = vae(x)
                recon = out.sample  # [-1, 1]
                recon_np = ((recon[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
                recon_np = np.clip(recon_np, 0, 255)
                recon_list.append(recon_np)
        
        # Grid: Original | Latent | Reconstruction
        n_rows = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        
        row_labels = ["Image", f"Latent\n({args.encoder_type})", "Recon"]
        
        for i in range(min(n_cols, len(all_images))):
            axes[0, i].imshow(all_images[i])
            axes[0, i].axis('off')
            
            axes[1, i].imshow(latent_vis_list[i])
            axes[1, i].axis('off')
            
            axes[2, i].imshow(recon_list[i])
            axes[2, i].axis('off')
        
        # Add row labels
        for row, label in enumerate(row_labels):
            axes[row, 0].set_ylabel(label, fontsize=12, rotation=0, labelpad=60, va='center')
        
        plt.suptitle("Image → Latent → Reconstruction", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "full_comparison.png"), dpi=200, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"  Skipping reconstruction: {e}")
    
    print(f"\n✓ Saved visualizations to {args.output_dir}")
    print("  - latent_vis_XXX.png: Individual image + latent pairs")
    print("  - latent_grid.png: Grid of images and their latent features")
    print("  - full_comparison.png: Image → Latent → Reconstruction")
    print("\nDone!")


if __name__ == "__main__":
    main()
