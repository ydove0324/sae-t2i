# spectrum_analysis.py
# Spectrum analysis for VAE latents using DCT/DFT
# Analyzes frequency characteristics of latent representations

import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

import torch
import matplotlib.pyplot as plt
from scipy.fftpack import dct, dctn
from scipy.fft import fft2, fftshift

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

# 设置 torch 缓存路径 (可选)
os.environ["TORCH_HOME"] = "/cpfs01/huangxu/.cache/torch"

# 添加当前目录以导入项目模块
sys.path.append(".")

# 导入 VAE 工具函数
from models.rae.utils.vae_utils import load_vae


# ==========================================
#              Helper Functions
# ==========================================

def center_crop_arr(pil_image, image_size):
    """Center crop image to target size."""
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


def zigzag_indices(n):
    """
    Generate zigzag scan indices for an n x n matrix.
    Returns list of (row, col) tuples in zigzag order.
    """
    indices = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            # Even sum: go up-right
            for i in range(min(s, n - 1), max(0, s - n + 1) - 1, -1):
                j = s - i
                if 0 <= i < n and 0 <= j < n:
                    indices.append((i, j))
        else:
            # Odd sum: go down-left
            for i in range(max(0, s - n + 1), min(s, n - 1) + 1):
                j = s - i
                if 0 <= i < n and 0 <= j < n:
                    indices.append((i, j))
    return indices


def apply_zigzag(matrix):
    """Apply zigzag scan to a 2D matrix, return 1D array."""
    n = matrix.shape[0]
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"
    indices = zigzag_indices(n)
    return np.array([matrix[i, j] for i, j in indices])


def compute_2d_dct(image):
    """Compute 2D DCT of an image/feature map."""
    return dctn(image, norm='ortho')


def compute_2d_dft(image):
    """Compute 2D DFT of an image/feature map and return magnitude spectrum."""
    f_transform = fft2(image)
    f_shift = fftshift(f_transform)
    magnitude = np.abs(f_shift)
    return magnitude


def normalize_spectrum(spectrum, eps=1e-10):
    """Normalize spectrum to [0, 1] range."""
    spectrum = np.abs(spectrum)
    max_val = np.max(spectrum)
    if max_val > eps:
        return spectrum / max_val
    return spectrum


@torch.no_grad()
def extract_latents(vae, x: torch.Tensor, use_lora: bool = True):
    """Extract latent features from VAE encoder."""
    feat = vae.encode_features(x, use_lora=use_lora)  # [B, C, H, W]
    return feat


@torch.no_grad()
def extract_latents_without_lora(vae, x: torch.Tensor):
    """Extract latent features from VAE encoder without LoRA."""
    feat = vae.encode_features(x, use_lora=False)  # [B, C, H, W]
    return feat


# ==========================================
#           Spectrum Analysis Functions
# ==========================================

def analyze_spectrum_dct(features, method='zigzag'):
    """
    Analyze DCT spectrum of feature maps.
    
    Args:
        features: numpy array of shape [B, C, H, W]
        method: 'zigzag' for zigzag scan, 'radial' for radial average
        
    Returns:
        spectrum: averaged spectrum along zigzag or radial direction
    """
    B, C, H, W = features.shape
    assert H == W, f"Feature maps must be square, got {H}x{W}"
    
    all_spectra = []
    
    for b in range(B):
        for c in range(C):
            # Compute 2D DCT
            dct_coeffs = compute_2d_dct(features[b, c])
            dct_abs = np.abs(dct_coeffs)
            
            if method == 'zigzag':
                # Zigzag scan
                zigzag_spectrum = apply_zigzag(dct_abs)
                all_spectra.append(zigzag_spectrum)
            elif method == 'radial':
                # Radial average
                radial_spectrum = radial_average(dct_abs)
                all_spectra.append(radial_spectrum)
    
    all_spectra = np.array(all_spectra)  # [B*C, N]
    
    # Compute mean and std
    mean_spectrum = np.mean(all_spectra, axis=0)
    std_spectrum = np.std(all_spectra, axis=0)
    
    return mean_spectrum, std_spectrum


def analyze_spectrum_dft(features, method='radial'):
    """
    Analyze DFT magnitude spectrum of feature maps.
    
    Args:
        features: numpy array of shape [B, C, H, W]
        method: 'radial' for radial average
        
    Returns:
        spectrum: averaged spectrum
    """
    B, C, H, W = features.shape
    
    all_spectra = []
    
    for b in range(B):
        for c in range(C):
            # Compute 2D DFT magnitude
            magnitude = compute_2d_dft(features[b, c])
            
            if method == 'radial':
                radial_spectrum = radial_average(magnitude)
                all_spectra.append(radial_spectrum)
            elif method == 'zigzag':
                zigzag_spectrum = apply_zigzag(magnitude)
                all_spectra.append(zigzag_spectrum)
    
    all_spectra = np.array(all_spectra)
    
    mean_spectrum = np.mean(all_spectra, axis=0)
    std_spectrum = np.std(all_spectra, axis=0)
    
    return mean_spectrum, std_spectrum


def radial_average(image):
    """Compute radial average of a 2D image (for frequency analysis)."""
    H, W = image.shape
    center_y, center_x = H // 2, W // 2
    
    # Create distance matrix from center
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
    r = r.astype(int)
    
    # Compute radial average
    max_r = int(np.sqrt(center_y ** 2 + center_x ** 2)) + 1
    radial_mean = np.zeros(max_r)
    radial_count = np.zeros(max_r)
    
    for i in range(H):
        for j in range(W):
            radius = int(r[i, j])
            if radius < max_r:
                radial_mean[radius] += image[i, j]
                radial_count[radius] += 1
    
    # Avoid division by zero
    radial_count[radial_count == 0] = 1
    radial_mean /= radial_count
    
    return radial_mean


def analyze_rgb_spectrum(images):
    """
    Analyze DCT spectrum of RGB images.
    
    Args:
        images: numpy array of shape [B, C, H, W] where C=3 (RGB)
        
    Returns:
        mean_spectrum, std_spectrum
    """
    B, C, H, W = images.shape
    assert H == W, f"Images must be square, got {H}x{W}"
    
    all_spectra = []
    
    for b in range(B):
        for c in range(C):
            # Compute 2D DCT
            dct_coeffs = compute_2d_dct(images[b, c])
            dct_abs = np.abs(dct_coeffs)
            
            # Zigzag scan
            zigzag_spectrum = apply_zigzag(dct_abs)
            all_spectra.append(zigzag_spectrum)
    
    all_spectra = np.array(all_spectra)
    
    mean_spectrum = np.mean(all_spectra, axis=0)
    std_spectrum = np.std(all_spectra, axis=0)
    
    return mean_spectrum, std_spectrum


# ==========================================
#                   Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Spectrum analysis for VAE latents")
    
    # Data
    parser.add_argument("--data-path", type=str, required=True, help="Path to ImageNet validation set.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=500, help="Number of random samples to analyze.")
    
    # VAE
    parser.add_argument("--vae-ckpt", type=str, required=True, help="Path to VAE checkpoint.")
    parser.add_argument("--encoder-type", type=str, default="dinov3", 
                        choices=["dinov3", "dinov3_vitl", "siglip2", "dinov2"])
    parser.add_argument("--decoder-type", type=str, default="cnn_decoder",
                        choices=["cnn_decoder", "vit_decoder"],
                        help="Decoder type: 'cnn_decoder' or 'vit_decoder'")
    parser.add_argument("--dinov3-dir", type=str, default="/cpfs01/huangxu/models/dinov3")
    parser.add_argument("--siglip2-model-name", type=str, default="google/siglip2-base-patch16-256")
    parser.add_argument("--dinov2-model-name", type=str, default="facebook/dinov2-with-registers-base",
                        help="DINOv2 model name from HuggingFace")
    parser.add_argument("--lora-rank", type=int, default=256)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--skip-to-moments", action="store_true",
                        help="Skip the to_moments layer (use raw encoder features)")
    parser.add_argument("--use-ema", action="store_true",
                        help="Load EMA model from checkpoint if available")
    parser.add_argument("--target-latent-channels", type=int, default=None,
                        help="Target latent channels (for channel projection)")
    
    # Normalization
    parser.add_argument("--normalization", type=str, default="none",
                        choices=["none", "layernorm", "channelwise", "global"],
                        help="Latent normalization type: 'none', 'layernorm', 'channelwise', 'global'")
    parser.add_argument("--latent-stats-path", type=str, default=None,
                        help="Path to latent stats .npz file (for channelwise/global normalization)")
    
    # Analysis options
    parser.add_argument("--transform", type=str, default="dct", choices=["dct", "dft", "both"])
    parser.add_argument("--method", type=str, default="zigzag", choices=["zigzag", "radial"])
    parser.add_argument("--compare-lora", action="store_true", 
                        help="Compare latents with and without LoRA (like w/ SE in the figure)")
    parser.add_argument("--compare-rgb", action="store_true", default=True,
                        help="Also analyze RGB image spectrum for comparison")
    parser.add_argument("--batch-size", type=int, default=16)
    
    # Output
    parser.add_argument("--output-dir", type=str, default="results/spectrum_analysis")
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
    print(" Spectrum Analysis for VAE Latents")
    print("=" * 60)
    print(f" Encoder:         {args.encoder_type}")
    print(f" Decoder:         {args.decoder_type}")
    print(f" VAE Ckpt:        {args.vae_ckpt}")
    print(f" Num Samples:     {args.num_samples}")
    print(f" Image Size:      {args.image_size}")
    print(f" Transform:       {args.transform}")
    print(f" Method:          {args.method}")
    print(f" Skip to Moments: {args.skip_to_moments}")
    print(f" Use EMA:         {args.use_ema}")
    print(f" Normalization:   {args.normalization}")
    print(f" Compare LoRA:    {args.compare_lora}")
    print(f" Compare RGB:     {args.compare_rgb}")
    print("=" * 60)
    
    # 1. Load VAE
    print("\nLoading VAE...")
    
    # 根据 encoder_type 设置参数
    if args.encoder_type == "dinov3":
        latent_channels = 1280
        dec_block_out_channels = (1280, 1024, 512, 256, 128)
        patch_size = 16
    elif args.encoder_type == "dinov3_vitl":
        latent_channels = 1024
        dec_block_out_channels = (1024, 768, 512, 256, 128)
        patch_size = 16
    elif args.encoder_type == "siglip2":
        latent_channels = 768
        dec_block_out_channels = (768, 512, 256, 128, 64)
        patch_size = 16
    elif args.encoder_type == "dinov2":
        latent_channels = 768
        dec_block_out_channels = (768, 512, 256, 128, 64)
        patch_size = 16
    else:
        raise ValueError(f"Unknown encoder_type: {args.encoder_type}")
    
    vae_model_params = {
        "encoder_type": args.encoder_type,
        "image_size": args.image_size,
        "patch_size": patch_size,
        "out_channels": 3,
        "latent_channels": latent_channels,
        "target_latent_channels": args.target_latent_channels,
        "spatial_downsample_factor": patch_size,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "dec_block_out_channels": dec_block_out_channels,
        "dec_layers_per_block": 3,
        "decoder_dropout": 0.0,
        "gradient_checkpointing": False,
        "denormalize_decoder_output": False,
    }
    
    # 根据 encoder_type 添加模型路径参数
    if args.encoder_type in ["dinov3", "dinov3_vitl"]:
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
        verbose=True,
        skip_to_moments=args.skip_to_moments,
        use_ema=args.use_ema,
    )
    vae.eval()
    
    # Load latent stats if normalization requires it
    latent_stats = None
    if args.normalization in ["channelwise", "global"] and args.latent_stats_path:
        from models.rae.utils.vae_utils import load_latent_stats, normalize_with_stats
        latent_stats = load_latent_stats(args.latent_stats_path, device=device, verbose=True)
    
    # 2. Load dataset
    print("\nLoading dataset...")
    
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),  # [-1, 1]
    ])
    
    dataset = ImageFolder(args.data_path, transform=transform)
    print(f"Total dataset size: {len(dataset)}")
    
    # Random sample
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Selected {len(subset)} samples for analysis")
    
    # 3. Extract latents and compute spectra
    print("\nExtracting latents and computing spectra...")
    
    all_latents_lora = []
    all_latents_no_lora = []
    all_rgb_images = []
    
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Extracting"):
            x = x.to(device)
            
            # Extract latents with LoRA
            latents_lora = extract_latents(vae, x, use_lora=True)
            
            # Apply normalization if specified
            if args.normalization == "layernorm":
                # LayerNorm across spatial dimensions
                B, C, H, W = latents_lora.shape
                latents_lora = latents_lora.view(B, C, -1)
                latents_lora = torch.nn.functional.layer_norm(latents_lora, [latents_lora.shape[-1]])
                latents_lora = latents_lora.view(B, C, H, W)
            elif args.normalization == "channelwise" and latent_stats is not None:
                from models.rae.utils.vae_utils import normalize_with_stats
                latents_lora = normalize_with_stats(latents_lora, latent_stats, per_channel=True)
            elif args.normalization == "global" and latent_stats is not None:
                from models.rae.utils.vae_utils import normalize_with_stats
                latents_lora = normalize_with_stats(latents_lora, latent_stats, per_channel=False)
            
            all_latents_lora.append(latents_lora.cpu().numpy())
            
            # Extract latents without LoRA (if comparing)
            if args.compare_lora:
                latents_no_lora = extract_latents_without_lora(vae, x)
                # Apply same normalization
                if args.normalization == "layernorm":
                    B, C, H, W = latents_no_lora.shape
                    latents_no_lora = latents_no_lora.view(B, C, -1)
                    latents_no_lora = torch.nn.functional.layer_norm(latents_no_lora, [latents_no_lora.shape[-1]])
                    latents_no_lora = latents_no_lora.view(B, C, H, W)
                elif args.normalization == "channelwise" and latent_stats is not None:
                    from models.rae.utils.vae_utils import normalize_with_stats
                    latents_no_lora = normalize_with_stats(latents_no_lora, latent_stats, per_channel=True)
                elif args.normalization == "global" and latent_stats is not None:
                    from models.rae.utils.vae_utils import normalize_with_stats
                    latents_no_lora = normalize_with_stats(latents_no_lora, latent_stats, per_channel=False)
                all_latents_no_lora.append(latents_no_lora.cpu().numpy())
            
            # Store RGB images (convert back to [0, 1] for analysis)
            if args.compare_rgb:
                rgb = (x.cpu().numpy() + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                all_rgb_images.append(rgb)
    
    # Concatenate
    latents_lora = np.concatenate(all_latents_lora, axis=0)  # [N, C, H, W]
    print(f"Latents shape (with LoRA): {latents_lora.shape}")
    
    if args.compare_lora:
        latents_no_lora = np.concatenate(all_latents_no_lora, axis=0)
        print(f"Latents shape (without LoRA): {latents_no_lora.shape}")
    
    if args.compare_rgb:
        rgb_images = np.concatenate(all_rgb_images, axis=0)
        print(f"RGB images shape: {rgb_images.shape}")
    
    # 4. Compute spectra
    print("\nComputing spectra...")
    
    results = {}
    
    if args.transform in ["dct", "both"]:
        print("  Computing DCT spectrum for latents (with LoRA)...")
        mean_lora, std_lora = analyze_spectrum_dct(latents_lora, method=args.method)
        results["latents_lora_dct"] = (mean_lora, std_lora)
        
        if args.compare_lora:
            print("  Computing DCT spectrum for latents (without LoRA)...")
            mean_no_lora, std_no_lora = analyze_spectrum_dct(latents_no_lora, method=args.method)
            results["latents_no_lora_dct"] = (mean_no_lora, std_no_lora)
        
        if args.compare_rgb:
            print("  Computing DCT spectrum for RGB images...")
            # 需要调整 RGB 图像大小以匹配 latent 的空间尺寸
            # 或者直接在原始分辨率上计算
            mean_rgb, std_rgb = analyze_rgb_spectrum(rgb_images)
            results["rgb_dct"] = (mean_rgb, std_rgb)
    
    if args.transform in ["dft", "both"]:
        print("  Computing DFT spectrum for latents (with LoRA)...")
        mean_lora_dft, std_lora_dft = analyze_spectrum_dft(latents_lora, method=args.method)
        results["latents_lora_dft"] = (mean_lora_dft, std_lora_dft)
        
        if args.compare_lora:
            print("  Computing DFT spectrum for latents (without LoRA)...")
            mean_no_lora_dft, std_no_lora_dft = analyze_spectrum_dft(latents_no_lora, method=args.method)
            results["latents_no_lora_dft"] = (mean_no_lora_dft, std_no_lora_dft)
    
    # 5. Visualization
    print("\nCreating visualizations...")
    
    # Plot style similar to the reference figure
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'figure.figsize': (10, 7),
    })
    
    # DCT Spectrum Plot (main plot like the reference figure)
    if args.transform in ["dct", "both"]:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Determine x-axis length (use minimum length across all spectra)
        min_len = min(len(v[0]) for k, v in results.items() if 'dct' in k)
        x = np.arange(min_len)
        
        # Plot latents with LoRA
        if "latents_lora_dct" in results:
            mean, std = results["latents_lora_dct"]
            mean, std = mean[:min_len], std[:min_len]
            # Normalize
            mean_norm = normalize_spectrum(mean)
            std_norm = std / (np.max(np.abs(mean)) + 1e-10)
            
            ax.semilogy(x, mean_norm, color='purple', linewidth=2, label='Latents; w/ LoRA')
            ax.fill_between(x, 
                           np.clip(mean_norm - std_norm, 1e-5, None),
                           mean_norm + std_norm, 
                           color='purple', alpha=0.2)
        
        # Plot latents without LoRA
        if "latents_no_lora_dct" in results:
            mean, std = results["latents_no_lora_dct"]
            mean, std = mean[:min_len], std[:min_len]
            mean_norm = normalize_spectrum(mean)
            std_norm = std / (np.max(np.abs(mean)) + 1e-10)
            
            ax.semilogy(x, mean_norm, color='magenta', linewidth=2, label='Latents')
            ax.fill_between(x,
                           np.clip(mean_norm - std_norm, 1e-5, None),
                           mean_norm + std_norm,
                           color='magenta', alpha=0.2)
        
        # Plot RGB
        if "rgb_dct" in results:
            mean, std = results["rgb_dct"]
            # RGB might have different length, truncate or interpolate
            rgb_len = len(mean)
            if rgb_len != min_len:
                # Interpolate to match latent spectrum length
                x_rgb = np.linspace(0, min_len - 1, rgb_len)
                mean_interp = np.interp(x, x_rgb, mean)
                std_interp = np.interp(x, x_rgb, std)
                mean, std = mean_interp, std_interp
            else:
                mean, std = mean[:min_len], std[:min_len]
            
            mean_norm = normalize_spectrum(mean)
            std_norm = std / (np.max(np.abs(results["rgb_dct"][0])) + 1e-10)
            
            ax.semilogy(x, mean_norm, color='orange', linewidth=2, label='RGB')
            ax.fill_between(x,
                           np.clip(mean_norm - std_norm, 1e-5, None),
                           mean_norm + std_norm,
                           color='orange', alpha=0.2)
        
        ax.set_xlabel('Zigzag Frequency Index')
        ax.set_ylabel('Normalized Amplitude')
        ax.set_title(f'DCT Spectrum Analysis\n({args.encoder_type})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([1e-4, 2])
        
        plt.tight_layout()
        save_path = os.path.join(args.output_dir, "dct_spectrum.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
    
    # DFT Spectrum Plot (radial)
    if args.transform in ["dft", "both"]:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        if "latents_lora_dft" in results:
            mean, std = results["latents_lora_dft"]
            x = np.arange(len(mean))
            mean_norm = normalize_spectrum(mean)
            
            ax.semilogy(x, mean_norm, color='purple', linewidth=2, label='Latents; w/ LoRA')
            ax.fill_between(x,
                           np.clip(mean_norm - std / (np.max(mean) + 1e-10), 1e-5, None),
                           mean_norm + std / (np.max(mean) + 1e-10),
                           color='purple', alpha=0.2)
        
        if "latents_no_lora_dft" in results:
            mean, std = results["latents_no_lora_dft"]
            mean_norm = normalize_spectrum(mean)
            
            ax.semilogy(x, mean_norm, color='magenta', linewidth=2, label='Latents')
            ax.fill_between(x,
                           np.clip(mean_norm - std / (np.max(mean) + 1e-10), 1e-5, None),
                           mean_norm + std / (np.max(mean) + 1e-10),
                           color='magenta', alpha=0.2)
        
        ax.set_xlabel('Radial Frequency')
        ax.set_ylabel('Normalized Amplitude')
        ax.set_title(f'DFT Magnitude Spectrum Analysis\n({args.encoder_type})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(args.output_dir, "dft_spectrum.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
    
    # 6. Additional visualizations - 2D DCT spectrum heatmap
    print("\nCreating 2D spectrum heatmaps...")
    
    # Average 2D DCT spectrum across all samples and channels
    sample_latent = latents_lora[0, 0]  # [H, W]
    dct_2d = compute_2d_dct(sample_latent)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original latent
    im0 = axes[0].imshow(sample_latent, cmap='viridis')
    axes[0].set_title('Sample Latent (Channel 0)')
    axes[0].set_xlabel('Width')
    axes[0].set_ylabel('Height')
    plt.colorbar(im0, ax=axes[0])
    
    # DCT coefficients
    im1 = axes[1].imshow(np.log10(np.abs(dct_2d) + 1e-10), cmap='hot')
    axes[1].set_title('2D DCT (log scale)')
    axes[1].set_xlabel('Frequency (horizontal)')
    axes[1].set_ylabel('Frequency (vertical)')
    plt.colorbar(im1, ax=axes[1])
    
    # Average 2D DCT across multiple samples
    avg_dct_2d = np.mean([np.abs(compute_2d_dct(latents_lora[i, 0])) 
                          for i in range(min(100, len(latents_lora)))], axis=0)
    im2 = axes[2].imshow(np.log10(avg_dct_2d + 1e-10), cmap='hot')
    axes[2].set_title('Average 2D DCT (100 samples)')
    axes[2].set_xlabel('Frequency (horizontal)')
    axes[2].set_ylabel('Frequency (vertical)')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    save_path = os.path.join(args.output_dir, "dct_2d_heatmap.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    # 7. Save numerical results
    print("\nSaving numerical results...")
    
    save_data = {
        'latents_shape': latents_lora.shape,
        'num_samples': len(latents_lora),
    }
    
    for key, (mean, std) in results.items():
        save_data[f'{key}_mean'] = mean
        save_data[f'{key}_std'] = std
    
    np.savez(os.path.join(args.output_dir, "spectrum_data.npz"), **save_data)
    
    # Save config
    with open(os.path.join(args.output_dir, "config.txt"), "w") as f:
        f.write(f"VAE Checkpoint: {args.vae_ckpt}\n")
        f.write(f"Encoder Type: {args.encoder_type}\n")
        f.write(f"Decoder Type: {args.decoder_type}\n")
        f.write(f"Image Size: {args.image_size}\n")
        f.write(f"Num Samples: {len(latents_lora)}\n")
        f.write(f"Latent Shape: {latents_lora.shape}\n")
        f.write(f"Transform: {args.transform}\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"Skip to Moments: {args.skip_to_moments}\n")
        f.write(f"Use EMA: {args.use_ema}\n")
        f.write(f"Normalization: {args.normalization}\n")
        f.write(f"Latent Stats Path: {args.latent_stats_path}\n")
        f.write(f"Compare LoRA: {args.compare_lora}\n")
        f.write(f"Compare RGB: {args.compare_rgb}\n")
        f.write(f"LoRA Rank: {args.lora_rank}\n")
        f.write(f"LoRA Alpha: {args.lora_alpha}\n")
    
    print("\n" + "=" * 60)
    print(" Done!")
    print(f" Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
