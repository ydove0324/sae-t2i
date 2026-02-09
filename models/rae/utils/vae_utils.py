"""
VAE utility functions for loading and using DINOv3/SigLIP2/DINOv2 VAE models.
Supports both CNN decoder and diffusion decoder.

包含:
- Latent 归一化/反归一化
- VAE 模型加载
- 从 latent 重建图像
- Encoder 配置获取
- 从命令行参数构建 VAE

统一了 train_vae, eval_vae, projects/rae 中的重复代码
"""

import os
import argparse
from typing import Dict, Any, Optional, Tuple, Union, Callable
import numpy as np
import torch
import torch.nn as nn

# ==========================================
#           SAE Normalization (DINOv3)
# ==========================================

EMA_SHIFT_FACTOR = 0.0019670347683131695
EMA_SCALE_FACTOR = 0.247765451669693


def normalize_sae(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor using SAE statistics (DINOv3)."""
    return (tensor - EMA_SHIFT_FACTOR) / EMA_SCALE_FACTOR


def denormalize_sae(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize tensor using SAE statistics (DINOv3)."""
    return tensor * EMA_SCALE_FACTOR + EMA_SHIFT_FACTOR


# ==========================================
#           SigLIP2 Normalization
# ==========================================

SIGLIP2_SHIFT_FACTOR = 0.0
SIGLIP2_SCALE_FACTOR = 0.6689115762710571


def normalize_siglip2(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor using SigLIP2 statistics."""
    return (tensor - SIGLIP2_SHIFT_FACTOR) / SIGLIP2_SCALE_FACTOR


def denormalize_siglip2(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize tensor using SigLIP2 statistics."""
    return tensor * SIGLIP2_SCALE_FACTOR + SIGLIP2_SHIFT_FACTOR


# ==========================================
#           DINOv2 Normalization
# ==========================================

# DINOv2-B (base) normalization statistics
# These are placeholder values - you may need to compute proper statistics
# from your training data similar to how DINOv3/SigLIP2 stats were computed.
DINOV2_SHIFT_FACTOR = 0.0
DINOV2_SCALE_FACTOR = 1.0  # Placeholder - adjust based on actual feature distribution


def normalize_dinov2(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor using DINOv2 statistics."""
    return (tensor - DINOV2_SHIFT_FACTOR) / DINOV2_SCALE_FACTOR


def denormalize_dinov2(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize tensor using DINOv2 statistics."""
    return tensor * DINOV2_SCALE_FACTOR + DINOV2_SHIFT_FACTOR


# ==========================================
#       Generic Normalization Helpers
# ==========================================

def get_normalize_fn(encoder_type: str = "dinov3"):
    """Get normalization function based on encoder type."""
    if encoder_type == "dinov3":
        return normalize_sae
    elif encoder_type == "siglip2":
        return normalize_siglip2
    elif encoder_type == "dinov2":
        return normalize_dinov2
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}. Supported: 'dinov3', 'siglip2', 'dinov2'")


def get_denormalize_fn(encoder_type: str = "dinov3"):
    """Get denormalization function based on encoder type."""
    if encoder_type == "dinov3":
        return denormalize_sae
    elif encoder_type == "siglip2":
        return denormalize_siglip2
    elif encoder_type == "dinov2":
        return denormalize_dinov2
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}. Supported: 'dinov3', 'siglip2', 'dinov2'")


# ==========================================
#     Latent Stats Loading & Normalization
# ==========================================

def load_latent_stats(stats_path: str, device: torch.device = None, verbose: bool = True):
    """
    Load pre-computed latent statistics from .npz file.
    
    Args:
        stats_path: Path to latent_stats.npz file (from compute_latent_stats.py)
        device: Device to load tensors to (default: CPU)
        verbose: Whether to print loading information
    
    Returns:
        dict with keys: 'mean', 'std', 'shift_factor', 'scale_factor'
    """
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Latent stats file not found: {stats_path}")
    
    data = np.load(stats_path)
    
    # Load per-channel mean and std
    mean = torch.from_numpy(data['mean']).float()  # [C]
    std = torch.from_numpy(data['std']).float()    # [C]
    
    # Compute scalar shift/scale factors (mean of means/stds)
    shift_factor = float(mean.mean().item())
    scale_factor = float(std.mean().item())
    
    if device is not None:
        mean = mean.to(device)
        std = std.to(device)
    
    if verbose:
        print(f"[load_latent_stats] Loaded from: {stats_path}")
        print(f"[load_latent_stats] Channels: {len(mean)}, shift_factor: {shift_factor:.6f}, scale_factor: {scale_factor:.6f}")
    
    return {
        'mean': mean,
        'std': std,
        'shift_factor': shift_factor,
        'scale_factor': scale_factor,
    }


def normalize_with_stats(
    tensor: torch.Tensor, 
    stats: dict, 
    per_channel: bool = False
) -> torch.Tensor:
    """
    Normalize tensor using pre-computed latent statistics.
    
    Args:
        tensor: Input tensor [B, C, H, W]
        stats: Dict from load_latent_stats() containing 'mean', 'std', 'shift_factor', 'scale_factor'
        per_channel: If True, use per-channel mean/std. If False, use scalar shift/scale factors.
    
    Returns:
        Normalized tensor
    """
    if per_channel:
        # Per-channel normalization: (x - mean[c]) / std[c]
        mean = stats['mean'].to(tensor.device)  # [C]
        std = stats['std'].to(tensor.device)    # [C]
        # Reshape for broadcasting: [C] -> [1, C, 1, 1]
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
        return (tensor - mean) / (std + 1e-8)
    else:
        # Scalar normalization: (x - shift_factor) / scale_factor
        shift = stats['shift_factor']
        scale = stats['scale_factor']
        return (tensor - shift) / scale


def denormalize_with_stats(
    tensor: torch.Tensor, 
    stats: dict, 
    per_channel: bool = False
) -> torch.Tensor:
    """
    Denormalize tensor using pre-computed latent statistics.
    
    Args:
        tensor: Normalized tensor [B, C, H, W]
        stats: Dict from load_latent_stats() containing 'mean', 'std', 'shift_factor', 'scale_factor'
        per_channel: If True, use per-channel mean/std. If False, use scalar shift/scale factors.
    
    Returns:
        Denormalized tensor
    """
    if per_channel:
        # Per-channel denormalization: x * std[c] + mean[c]
        mean = stats['mean'].to(tensor.device)  # [C]
        std = stats['std'].to(tensor.device)    # [C]
        # Reshape for broadcasting: [C] -> [1, C, 1, 1]
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
        return tensor * std + mean
    else:
        # Scalar denormalization: x * scale_factor + shift_factor
        shift = stats['shift_factor']
        scale = stats['scale_factor']
        return tensor * scale + shift


class LatentNormalizer:
    """
    A helper class that wraps normalization/denormalization with loaded stats.
    Can be used as a drop-in replacement for normalize_fn/denormalize_fn.
    """
    def __init__(self, stats: dict, per_channel: bool = False):
        """
        Args:
            stats: Dict from load_latent_stats()
            per_channel: Whether to use per-channel normalization
        """
        self.stats = stats
        self.per_channel = per_channel
    
    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return normalize_with_stats(tensor, self.stats, self.per_channel)
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return denormalize_with_stats(tensor, self.stats, self.per_channel)
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Default call is normalize."""
        return self.normalize(tensor)


# ==========================================
#           VAE Loading
# ==========================================

def requires_grad(model: nn.Module, flag: bool = True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def load_dinov3_vae(
    vae_checkpoint_path: str,
    device: torch.device,
    decoder_type: str = "diffusion_decoder",
    model_params: dict = None,
    verbose: bool = True,
    use_ema: bool = False,
):
    """
    Build and load DINOv3 AutoencoderKL.
    This is a wrapper for backward compatibility.
    Use load_vae() for more flexibility with different encoder types.
    
    Args:
        vae_checkpoint_path: Path to VAE checkpoint file.
        device: Device to load the model on.
        decoder_type: "diffusion_decoder" or "cnn_decoder".
        model_params: Optional custom model parameters.
        verbose: Whether to print loading information.
        use_ema: If True, load EMA model from checkpoint if available.
    """
    return load_vae(
        vae_checkpoint_path=vae_checkpoint_path,
        device=device,
        encoder_type="dinov3",
        decoder_type=decoder_type,
        model_params=model_params,
        verbose=verbose,
        use_ema=use_ema,
    )


def load_vae(
    vae_checkpoint_path: str,
    device: torch.device,
    encoder_type: str = "dinov3",
    decoder_type: str = "cnn_decoder",
    model_params: dict = None,
    verbose: bool = True,
    skip_to_moments: bool = False,
    use_ema: bool = False,
):
    """
    Build and load VAE with different encoder types (DINOv3, SigLIP2, or DINOv2).

    Args:
        vae_checkpoint_path: Path to VAE checkpoint file.
        device: Device to load the model on.
        encoder_type: "dinov3", "siglip2", or "dinov2".
        decoder_type: "diffusion_decoder" or "cnn_decoder".
        model_params: Optional custom model parameters. If None, uses default based on encoder_type.
        verbose: Whether to print loading information.
        skip_to_moments: If True, ignore missing 'to_moments' keys (for old checkpoints without this layer).
        use_ema: If True, load EMA model from checkpoint if available. Otherwise, load regular model.

    Returns:
        Loaded VAE model in eval mode with frozen parameters.
    """
    # Dynamic import based on decoder type
    if decoder_type in ["cnn_decoder", "vit_decoder"]:
        # Both cnn_decoder and vit_decoder are implemented in cnn_decoder.AutoencoderKL
        try:
            from cnn_decoder import AutoencoderKL
        except ImportError:
            raise ImportError("cnn_decoder module not found, cannot use decoder_type='cnn_decoder' or 'vit_decoder'.")
        if verbose:
            print(f"[load_vae] Using {decoder_type} (cnn_decoder.AutoencoderKL) with encoder_type={encoder_type}.")
    elif decoder_type == "diffusion_decoder":
        try:
            from sae_model import AutoencoderKL
        except ImportError:
            raise ImportError("sae_model module not found, cannot use decoder_type='diffusion_decoder'.")
        if verbose:
            print(f"[load_vae] Using diffusion decoder (sae_model.AutoencoderKL) with encoder_type={encoder_type}.")
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}. Supported: 'cnn_decoder', 'vit_decoder', 'diffusion_decoder'")

    # Default model parameters based on encoder_type
    if model_params is None:
        if encoder_type == "dinov3":
            model_params = {
                "encoder_type": "dinov3",
                "dinov3_model_dir": "/share/project/huangxu/models/dinov3",
                "image_size": 256,
                "patch_size": 16,
                "out_channels": 3,
                "latent_channels": 1280,
                "target_latent_channels": None,
                "spatial_downsample_factor": 16,
                "lora_rank": 256,
                "lora_alpha": 256,
                "dec_block_out_channels": (1280, 1024, 512, 256, 128),
                "dec_layers_per_block": 3,
                "decoder_dropout": 0.0,
                "gradient_checkpointing": False,
                "denormalize_decoder_output": False,
            }
        elif encoder_type == "siglip2":
            model_params = {
                "encoder_type": "siglip2",
                "siglip2_model_name": "google/siglip2-base-patch16-256",
                "image_size": 256,
                "patch_size": 16,
                "out_channels": 3,
                "latent_channels": 768,  # SigLIP2-base hidden size
                "target_latent_channels": None,
                "spatial_downsample_factor": 16,
                "lora_rank": 256,
                "lora_alpha": 256,
                "dec_block_out_channels": (768, 512, 256, 128, 64),  # Adjusted for 768 channels
                "dec_layers_per_block": 3,
                "decoder_dropout": 0.0,
                "gradient_checkpointing": False,
                "denormalize_decoder_output": False,
            }
        elif encoder_type == "dinov2":
            model_params = {
                "encoder_type": "dinov2",
                "dinov2_model_name": "facebook/dinov2-with-registers-base",
                "image_size": 224,  # DINOv2 typically uses 224
                "patch_size": 14,  # DINOv2 uses patch_size=14
                "out_channels": 3,
                "latent_channels": 768,  # DINOv2-base hidden size
                "target_latent_channels": None,
                "spatial_downsample_factor": 14,  # Match patch_size
                "lora_rank": 256,
                "lora_alpha": 256,
                "dec_block_out_channels": (768, 512, 256, 128, 64),  # Adjusted for 768 channels
                "dec_layers_per_block": 3,
                "decoder_dropout": 0.0,
                "gradient_checkpointing": False,
                "denormalize_decoder_output": False,
            }
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Supported: 'dinov3', 'siglip2', 'dinov2'")
    else:
        # Ensure encoder_type is set in model_params
        if "encoder_type" not in model_params:
            model_params["encoder_type"] = encoder_type

    # Ensure skip_to_moments is set in model_params (parameter takes precedence)
    if skip_to_moments:
        model_params["skip_to_moments"] = True
    elif "skip_to_moments" not in model_params:
        model_params["skip_to_moments"] = False

    # Set decoder_type in model_params (for cnn_decoder.AutoencoderKL which supports both cnn and vit)
    if decoder_type in ["cnn_decoder", "vit_decoder"]:
        model_params["decoder_type"] = decoder_type

    vae = AutoencoderKL(**model_params).to(device)

    if verbose:
        print(f"[load_vae] Loading VAE checkpoint from: {vae_checkpoint_path}")

    checkpoint = torch.load(vae_checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    # If use_ema=True, prioritize loading ema_model
    if use_ema and "ema_model" in checkpoint:
        state_dict = checkpoint["ema_model"]
        if verbose:
            print("[load_vae] Loading EMA model from checkpoint (use_ema=True)")
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Warn if use_ema=True but ema_model not found
    if use_ema and "ema_model" not in checkpoint:
        if verbose:
            print("[load_vae] Warning: use_ema=True but checkpoint does not contain 'ema_model'. Loading regular model instead.")

    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    
    if verbose:
        if skip_to_moments:
            print("[load_vae] skip_to_moments=True: to_moments layer not created, using raw encoder features.")
        if missing:
            print(f"[load_vae] Missing keys: {missing}")
        if unexpected:
            print(f"[load_vae] Unexpected keys: {unexpected}")
        print(f"[load_vae] Loaded VAE. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    vae.eval()
    requires_grad(vae, False)
    return vae


# ==========================================
#           Reconstruction
# ==========================================

# ==========================================
#           Encoder 配置
# ==========================================

# 各 encoder 类型的默认配置
ENCODER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "dinov3": {
        "latent_channels": 1280,
        "patch_size": 16,
        "dec_block_out_channels": (1280, 1024, 512, 256, 128),
        "default_model_dir": "/cpfs01/huangxu/models/dinov3",
        "model_key": "dinov3_model_dir",
    },
    "dinov3_vitl": {
        "latent_channels": 1024,
        "patch_size": 16,
        "dec_block_out_channels": (1024, 768, 512, 256, 128),
        "default_model_dir": "/cpfs01/huangxu/models/dinov3",
        "model_key": "dinov3_model_dir",
    },
    "siglip2": {
        "latent_channels": 768,
        "patch_size": 16,
        "dec_block_out_channels": (768, 512, 256, 128, 64),
        "default_model_name": "google/siglip2-base-patch16-256",
        "model_key": "siglip2_model_name",
    },
    "dinov2": {
        "latent_channels": 768,
        "patch_size": 14,
        "dec_block_out_channels": (768, 512, 256, 128, 64),
        "default_model_name": "facebook/dinov2-with-registers-base",
        "model_key": "dinov2_model_name",
    },
}


def get_encoder_config(encoder_type: str) -> Dict[str, Any]:
    """
    获取 encoder 类型的默认配置。
    
    Args:
        encoder_type: encoder 类型 ('dinov3', 'dinov3_vitl', 'siglip2', 'dinov2')
    
    Returns:
        配置字典，包含:
        - latent_channels: 潜在空间通道数
        - patch_size: patch 大小
        - dec_block_out_channels: 默认 decoder block channels
        - default_model_dir/default_model_name: 默认模型路径
        - model_key: 模型路径参数名
    
    Example:
        >>> config = get_encoder_config("dinov3")
        >>> print(config["latent_channels"])
        1280
    """
    if encoder_type not in ENCODER_CONFIGS:
        raise ValueError(
            f"Unknown encoder_type: {encoder_type}. "
            f"Supported: {list(ENCODER_CONFIGS.keys())}"
        )
    return ENCODER_CONFIGS[encoder_type].copy()


def get_latent_channels(encoder_type: str) -> int:
    """获取 encoder 类型的 latent channels。"""
    return get_encoder_config(encoder_type)["latent_channels"]


def get_patch_size(encoder_type: str) -> int:
    """获取 encoder 类型的 patch size。"""
    return get_encoder_config(encoder_type)["patch_size"]


def get_dec_block_out_channels(encoder_type: str) -> Tuple[int, ...]:
    """获取 encoder 类型对应的默认 decoder block channels。"""
    return get_encoder_config(encoder_type)["dec_block_out_channels"]


# ==========================================
#           从参数构建 VAE
# ==========================================

def build_vae_model_params(
    encoder_type: str,
    image_size: int = 256,
    decoder_type: str = "cnn_decoder",
    lora_rank: int = 256,
    lora_alpha: int = 256,
    lora_dropout: float = 0.0,
    # ViT decoder 参数
    vit_hidden_size: int = 1024,
    vit_num_layers: int = 24,
    vit_num_heads: int = 16,
    vit_intermediate_size: int = 4096,
    # 其他参数
    dec_block_out_channels: Tuple[int, ...] = None,
    dec_layers_per_block: int = 3,
    decoder_dropout: float = 0.0,
    gradient_checkpointing: bool = False,
    denormalize_decoder_output: bool = False,
    skip_to_moments: bool = False,
    # 模型路径
    dinov3_dir: str = None,
    siglip2_model_name: str = None,
    dinov2_model_name: str = None,
) -> Dict[str, Any]:
    """
    构建 VAE 模型参数字典。
    
    根据 encoder_type 自动推断 latent_channels、patch_size 等参数，
    简化了多处重复的参数构建逻辑。
    
    Args:
        encoder_type: encoder 类型
        image_size: 输入图像尺寸
        decoder_type: decoder 类型
        lora_rank: LoRA rank (0 表示不使用 LoRA)
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        vit_hidden_size: ViT decoder hidden size
        vit_num_layers: ViT decoder num layers
        vit_num_heads: ViT decoder num heads
        vit_intermediate_size: ViT decoder intermediate size
        dec_block_out_channels: 自定义 decoder block channels (None 使用默认)
        dec_layers_per_block: decoder layers per block
        decoder_dropout: decoder dropout
        gradient_checkpointing: 是否使用 gradient checkpointing
        denormalize_decoder_output: 是否反归一化 decoder 输出
        skip_to_moments: 是否跳过 to_moments 层
        dinov3_dir: DINOv3 模型目录
        siglip2_model_name: SigLIP2 模型名
        dinov2_model_name: DINOv2 模型名
    
    Returns:
        VAE 模型参数字典
    """
    config = get_encoder_config(encoder_type)
    latent_channels = config["latent_channels"]
    patch_size = config["patch_size"]
    
    # 使用默认或自定义的 decoder channels
    if dec_block_out_channels is None:
        dec_block_out_channels = config["dec_block_out_channels"]
    
    # 基础参数
    model_params = {
        "encoder_type": encoder_type,
        "image_size": image_size,
        "patch_size": patch_size,
        "out_channels": 3,
        "latent_channels": latent_channels,
        "target_latent_channels": None,
        "spatial_downsample_factor": patch_size,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "decoder_dropout": decoder_dropout,
        "gradient_checkpointing": gradient_checkpointing,
        "denormalize_decoder_output": denormalize_decoder_output,
        "skip_to_moments": skip_to_moments,
    }
    
    # Decoder 类型特定参数
    if decoder_type in ["cnn_decoder", "vit_decoder"]:
        model_params["decoder_type"] = decoder_type
        
        if decoder_type == "cnn_decoder":
            model_params["dec_block_out_channels"] = dec_block_out_channels
            model_params["dec_layers_per_block"] = dec_layers_per_block
        elif decoder_type == "vit_decoder":
            model_params["vit_decoder_hidden_size"] = vit_hidden_size
            model_params["vit_decoder_num_layers"] = vit_num_layers
            model_params["vit_decoder_num_heads"] = vit_num_heads
            model_params["vit_decoder_intermediate_size"] = vit_intermediate_size
    
    # Encoder 模型路径
    if encoder_type in ["dinov3", "dinov3_vitl"]:
        model_params["dinov3_model_dir"] = (
            dinov3_dir or config.get("default_model_dir", "/cpfs01/huangxu/models/dinov3")
        )
    elif encoder_type == "siglip2":
        model_params["siglip2_model_name"] = (
            siglip2_model_name or config.get("default_model_name", "google/siglip2-base-patch16-256")
        )
    elif encoder_type == "dinov2":
        model_params["dinov2_model_name"] = (
            dinov2_model_name or config.get("default_model_name", "facebook/dinov2-with-registers-base")
        )
    
    return model_params


def build_vae_from_args(
    args: argparse.Namespace,
    device: torch.device,
    verbose: bool = True,
) -> nn.Module:
    """
    从命令行参数构建并加载 VAE 模型。
    
    这是一个便捷函数，整合了参数解析、模型构建和权重加载。
    
    Args:
        args: 解析后的命令行参数，应包含以下属性:
            - vae_ckpt: VAE checkpoint 路径
            - encoder_type: encoder 类型
            - decoder_type: decoder 类型
            - image_size: 图像尺寸
            - lora_rank, lora_alpha: LoRA 参数
            - vit_hidden_size, vit_num_layers 等: ViT decoder 参数
            - skip_to_moments, denormalize_decoder_output 等
        device: 设备
        verbose: 是否打印加载信息
    
    Returns:
        加载好的 VAE 模型
    
    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_all_vae_args(parser)  # from argparse_utils
        >>> args = parser.parse_args()
        >>> vae = build_vae_from_args(args, device)
    """
    # 从 args 获取参数（带默认值）
    encoder_type = getattr(args, 'encoder_type', 'dinov3')
    decoder_type = getattr(args, 'decoder_type', 'cnn_decoder')
    image_size = getattr(args, 'image_size', 256)
    
    # LoRA 参数
    lora_rank = getattr(args, 'lora_rank', 256)
    lora_alpha = getattr(args, 'lora_alpha', 256)
    if getattr(args, 'no_lora', False):
        lora_rank = 0
        lora_alpha = 0
    
    # ViT decoder 参数
    vit_hidden_size = getattr(args, 'vit_hidden_size', 1024)
    vit_num_layers = getattr(args, 'vit_num_layers', 24)
    vit_num_heads = getattr(args, 'vit_num_heads', 16)
    vit_intermediate_size = getattr(args, 'vit_intermediate_size', 4096)
    
    # 其他参数
    skip_to_moments = getattr(args, 'skip_to_moments', False)
    denormalize_decoder_output = getattr(args, 'denormalize_decoder_output', False)
    use_ema = getattr(args, 'vae_use_ema', False) or getattr(args, 'use_ema', False)
    
    # 模型路径
    dinov3_dir = getattr(args, 'dinov3_dir', None)
    siglip2_model_name = getattr(args, 'siglip2_model_name', None)
    dinov2_model_name = getattr(args, 'dinov2_model_name', None)
    
    # 构建模型参数
    model_params = build_vae_model_params(
        encoder_type=encoder_type,
        image_size=image_size,
        decoder_type=decoder_type,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        vit_hidden_size=vit_hidden_size,
        vit_num_layers=vit_num_layers,
        vit_num_heads=vit_num_heads,
        vit_intermediate_size=vit_intermediate_size,
        skip_to_moments=skip_to_moments,
        denormalize_decoder_output=denormalize_decoder_output,
        dinov3_dir=dinov3_dir,
        siglip2_model_name=siglip2_model_name,
        dinov2_model_name=dinov2_model_name,
    )
    
    # 加载 VAE
    vae_ckpt = getattr(args, 'vae_ckpt', None)
    if vae_ckpt is None:
        raise ValueError("vae_ckpt is required but not provided in args")
    
    return load_vae(
        vae_checkpoint_path=vae_ckpt,
        device=device,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        model_params=model_params,
        verbose=verbose,
        skip_to_moments=skip_to_moments,
        use_ema=use_ema,
    )


# ==========================================
#           Reconstruction
# ==========================================

@torch.no_grad()
def reconstruct_from_latent_with_diffusion(
    vae,
    latent_z: torch.Tensor,
    image_shape: torch.Size,
    diffusion_steps: int = 25,
    decoder_type: str = "diffusion_decoder",
    encoder_type: str = "dinov3",
    denormalize_fn_override=None,
) -> torch.Tensor:
    """
    Given latent_z from encoder (or model prediction in the same latent space),
    run the full decoder reconstruction.

    Args:
        vae: The VAE model.
        latent_z: Latent tensor from encoder.
        image_shape: Output image shape (B, C, H, W).
        diffusion_steps: Number of diffusion steps (only for diffusion_decoder).
        decoder_type: "diffusion_decoder" or "cnn_decoder".
        encoder_type: "dinov3", "siglip2", or "dinov2".
        denormalize_fn_override: Optional custom denormalization function. If provided,
            uses this instead of the default encoder_type-based denormalization.
            Can be a callable or a LatentNormalizer instance.

    Returns:
        Reconstructed image tensor in [-1, 1] range.
    """
    device = latent_z.device
    
    # Get denormalize function
    if denormalize_fn_override is not None:
        # Use custom denormalize function
        if isinstance(denormalize_fn_override, LatentNormalizer):
            denormalize_fn = denormalize_fn_override.denormalize
        else:
            denormalize_fn = denormalize_fn_override
    else:
        # Use default encoder_type-based denormalization
        denormalize_fn = get_denormalize_fn(encoder_type)
    
    # VAE is float32, ensure input is also float32
    z = denormalize_fn(latent_z).float()

    if decoder_type == "diffusion_decoder":
        if getattr(vae, "post_quant_conv", None) is not None and vae.post_quant_conv is not None:
            z = vae.post_quant_conv(z)

        context = vae.diffusion_decoder.get_context(z)
        corrected_context = list(reversed(context[:]))

        diffusion = vae.diffusion_decoder.diffusion
        diffusion.set_sample_schedule(diffusion_steps, device)

        init_noise = torch.randn(
            image_shape,
            device=device,
            dtype=torch.float32,
        )

        recon = diffusion.p_sample_loop(
            vae=vae,
            shape=image_shape,
            context=corrected_context,
            clip_denoised=True,
            init_noise=init_noise,
            eta=0.0,
        )
        return recon

    elif decoder_type == "cnn_decoder" or decoder_type == "vit_decoder":
        if not hasattr(vae, "decode"):
            raise AttributeError("VAE does not have decode(); cnn_decoder requires vae.decode(z).")
        out = vae.decode(z)
        recon = out.sample if hasattr(out, "sample") else out
        return recon

    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}")
