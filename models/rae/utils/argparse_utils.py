"""
共享命令行参数定义

包含:
- VAE 相关参数
- LoRA 参数
- Decoder 参数（CNN/ViT）
- 训练参数
- 分布式训练参数

统一了 train_vae, eval_vae, projects/rae 中的重复参数定义
"""

import argparse
from typing import Tuple, Dict, Any


# ==========================================
#           VAE 参数
# ==========================================

def add_vae_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加 VAE 模型相关参数。
    
    Args:
        parser: argparse.ArgumentParser 实例
    
    Returns:
        添加参数后的 parser
    """
    group = parser.add_argument_group("VAE Model")
    
    group.add_argument(
        "--vae-ckpt",
        type=str,
        default=None,
        help="Path to VAE checkpoint file.",
    )
    group.add_argument(
        "--vae-use-ema",
        action="store_true",
        help="Load EMA model from VAE checkpoint if available.",
    )
    
    return parser


def add_encoder_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加 Encoder 相关参数。
    
    Args:
        parser: argparse.ArgumentParser 实例
    
    Returns:
        添加参数后的 parser
    """
    group = parser.add_argument_group("Encoder")
    
    group.add_argument(
        "--encoder-type",
        type=str,
        default="dinov3",
        choices=["dinov3", "dinov3_vitl", "siglip2", "dinov2"],
        help="Encoder type: 'dinov3', 'dinov3_vitl', 'siglip2', or 'dinov2'.",
    )
    group.add_argument(
        "--dinov3-dir",
        type=str,
        default="/cpfs01/huangxu/models/dinov3",
        help="Path to DINOv3 model directory.",
    )
    group.add_argument(
        "--siglip2-model-name",
        type=str,
        default="google/siglip2-base-patch16-256",
        help="SigLIP2 model name from HuggingFace.",
    )
    group.add_argument(
        "--dinov2-model-name",
        type=str,
        default="facebook/dinov2-with-registers-base",
        help="DINOv2 model name from HuggingFace.",
    )
    
    return parser


def add_decoder_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加 Decoder 相关参数。
    
    Args:
        parser: argparse.ArgumentParser 实例
    
    Returns:
        添加参数后的 parser
    """
    group = parser.add_argument_group("Decoder")
    
    group.add_argument(
        "--decoder-type",
        type=str,
        default="cnn_decoder",
        choices=["cnn_decoder", "vit_decoder", "diffusion_decoder"],
        help="Decoder type: 'cnn_decoder', 'vit_decoder', or 'diffusion_decoder'.",
    )
    group.add_argument(
        "--dec-block-out-channels",
        type=str,
        default=None,
        help="Decoder block output channels, comma-separated (e.g., '1280,1024,512,256,128'). "
             "If not specified, auto-detected from encoder type.",
    )
    group.add_argument(
        "--dec-layers-per-block",
        type=int,
        default=3,
        help="Number of layers per decoder block (for CNN decoder).",
    )
    group.add_argument(
        "--vae-diffusion-steps",
        type=int,
        default=25,
        help="Sampling steps for VAE diffusion decoder.",
    )
    
    return parser


def add_vit_decoder_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加 ViT Decoder 相关参数。
    
    Args:
        parser: argparse.ArgumentParser 实例
    
    Returns:
        添加参数后的 parser
    """
    group = parser.add_argument_group("ViT Decoder")
    
    group.add_argument(
        "--vit-hidden-size",
        type=int,
        default=1024,
        help="ViT decoder hidden size (XL: 1024, L: 768, B: 512).",
    )
    group.add_argument(
        "--vit-num-layers",
        type=int,
        default=24,
        help="ViT decoder num layers (XL: 24, L: 16, B: 8).",
    )
    group.add_argument(
        "--vit-num-heads",
        type=int,
        default=16,
        help="ViT decoder num heads (XL: 16, L: 12, B: 8).",
    )
    group.add_argument(
        "--vit-intermediate-size",
        type=int,
        default=4096,
        help="ViT decoder intermediate size (XL: 4096, L: 3072, B: 2048).",
    )
    
    return parser


# ==========================================
#           LoRA 参数
# ==========================================

def add_lora_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加 LoRA 相关参数。
    
    Args:
        parser: argparse.ArgumentParser 实例
    
    Returns:
        添加参数后的 parser
    """
    group = parser.add_argument_group("LoRA")
    
    group.add_argument(
        "--no-lora",
        action="store_true",
        help="Do not use LoRA in VAE encoder (set lora_rank=0).",
    )
    group.add_argument(
        "--lora-rank",
        type=int,
        default=256,
        help="LoRA rank (ignored if --no-lora is set).",
    )
    group.add_argument(
        "--lora-alpha",
        type=int,
        default=256,
        help="LoRA alpha (ignored if --no-lora is set).",
    )
    group.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout rate.",
    )
    
    return parser


# ==========================================
#           训练参数
# ==========================================

def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加通用训练参数。
    
    Args:
        parser: argparse.ArgumentParser 实例
    
    Returns:
        添加参数后的 parser
    """
    group = parser.add_argument_group("Training")
    
    group.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16"],
        default="bf16",
        help="Compute precision for training.",
    )
    group.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Global batch size (across all GPUs).",
    )
    group.add_argument(
        "--global-seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    group.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    group.add_argument(
        "--clip-grad",
        type=float,
        default=1.0,
        help="Gradient clipping value (0 to disable).",
    )
    group.add_argument(
        "--ema-decay",
        type=float,
        default=0.9999,
        help="EMA decay rate.",
    )
    
    return parser


def add_distributed_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加分布式训练参数。
    
    Args:
        parser: argparse.ArgumentParser 实例
    
    Returns:
        添加参数后的 parser
    """
    group = parser.add_argument_group("Distributed")
    
    group.add_argument(
        "--fsdp-size",
        type=int,
        default=1,
        help="FSDP sharding size. 1=disabled (use DDP), >1=enable FSDP.",
    )
    group.add_argument(
        "--ema-cpu",
        action="store_true",
        help="Keep EMA model on CPU to save GPU memory.",
    )
    
    return parser


# ==========================================
#           数据参数
# ==========================================

def add_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加数据相关参数。
    
    Args:
        parser: argparse.ArgumentParser 实例
    
    Returns:
        添加参数后的 parser
    """
    group = parser.add_argument_group("Data")
    
    group.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to training dataset.",
    )
    group.add_argument(
        "--val-path",
        type=str,
        default=None,
        help="Path to validation dataset.",
    )
    group.add_argument(
        "--image-size",
        type=int,
        default=256,
        choices=[224, 256, 384, 512],
        help="Input image resolution.",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size per GPU.",
    )
    group.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers.",
    )
    
    return parser


# ==========================================
#           日志参数
# ==========================================

def add_logging_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加日志相关参数。
    
    Args:
        parser: argparse.ArgumentParser 实例
    
    Returns:
        添加参数后的 parser
    """
    group = parser.add_argument_group("Logging")
    
    group.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results.",
    )
    group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs (alias for --results-dir).",
    )
    group.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log metrics every N steps.",
    )
    group.add_argument(
        "--save-every",
        type=int,
        default=5000,
        help="Save checkpoint every N steps.",
    )
    group.add_argument(
        "--eval-every",
        type=int,
        default=2000,
        help="Run evaluation every N steps.",
    )
    group.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    
    return parser


# ==========================================
#           Latent Stats 参数
# ==========================================

def add_latent_stats_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加 Latent Stats 相关参数（用于 DiT 训练）。
    
    Args:
        parser: argparse.ArgumentParser 实例
    
    Returns:
        添加参数后的 parser
    """
    group = parser.add_argument_group("Latent Stats")
    
    group.add_argument(
        "--latent-stats-path",
        type=str,
        default=None,
        help="Path to pre-computed latent_stats.npz for normalization.",
    )
    group.add_argument(
        "--per-channel-norm",
        action="store_true",
        help="Use per-channel normalization (only with --latent-stats-path).",
    )
    group.add_argument(
        "--skip-to-moments",
        action="store_true",
        help="Skip loading to_moments layer (for old checkpoints).",
    )
    group.add_argument(
        "--denormalize-decoder-output",
        action="store_true",
        help="Denormalize decoder output in VAE.",
    )
    
    return parser


# ==========================================
#           组合函数
# ==========================================

def add_all_vae_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加所有 VAE 相关参数（VAE + Encoder + Decoder + ViT Decoder + LoRA + Latent Stats）。
    
    Args:
        parser: argparse.ArgumentParser 实例
    
    Returns:
        添加参数后的 parser
    """
    add_vae_args(parser)
    add_encoder_args(parser)
    add_decoder_args(parser)
    add_vit_decoder_args(parser)
    add_lora_args(parser)
    add_latent_stats_args(parser)
    return parser


def add_common_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    添加通用训练相关参数（Training + Distributed + Data + Logging）。
    
    Args:
        parser: argparse.ArgumentParser 实例
    
    Returns:
        添加参数后的 parser
    """
    add_training_args(parser)
    add_distributed_args(parser)
    add_data_args(parser)
    add_logging_args(parser)
    return parser


# ==========================================
#           参数后处理
# ==========================================

def process_lora_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    处理 LoRA 相关参数（--no-lora 标志）。
    
    Args:
        args: 解析后的参数
    
    Returns:
        处理后的参数
    """
    if hasattr(args, 'no_lora') and args.no_lora:
        args.lora_rank = 0
        args.lora_alpha = 0
    return args


def process_output_dir_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    处理输出目录参数（--output-dir 是 --results-dir 的别名）。
    
    Args:
        args: 解析后的参数
    
    Returns:
        处理后的参数
    """
    if hasattr(args, 'output_dir') and args.output_dir is not None:
        args.results_dir = args.output_dir
    elif hasattr(args, 'results_dir') and hasattr(args, 'output_dir'):
        args.output_dir = args.results_dir
    return args


def process_dec_channels_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    处理 decoder block channels 参数（从字符串解析为 tuple）。
    
    Args:
        args: 解析后的参数
    
    Returns:
        处理后的参数
    """
    if hasattr(args, 'dec_block_out_channels') and args.dec_block_out_channels is not None:
        if isinstance(args.dec_block_out_channels, str):
            args.dec_block_out_channels = tuple(
                int(x.strip()) for x in args.dec_block_out_channels.split(',')
            )
    return args


def process_all_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    处理所有参数的后处理。
    
    Args:
        args: 解析后的参数
    
    Returns:
        处理后的参数
    """
    args = process_lora_args(args)
    args = process_output_dir_args(args)
    args = process_dec_channels_args(args)
    return args


# ==========================================
#           Encoder 配置获取
# ==========================================

# 这些配置也在 vae_utils.py 中定义，这里提供一个简化的访问方式
ENCODER_CONFIGS = {
    "dinov3": {
        "latent_channels": 1280,
        "patch_size": 16,
        "dec_block_out_channels": (1280, 1024, 512, 256, 128),
        "default_model_dir": "/cpfs01/huangxu/models/dinov3",
    },
    "dinov3_vitl": {
        "latent_channels": 1024,
        "patch_size": 16,
        "dec_block_out_channels": (1024, 768, 512, 256, 128),
        "default_model_dir": "/cpfs01/huangxu/models/dinov3",
    },
    "siglip2": {
        "latent_channels": 768,
        "patch_size": 16,
        "dec_block_out_channels": (768, 512, 256, 128, 64),
        "default_model_name": "google/siglip2-base-patch16-256",
    },
    "dinov2": {
        "latent_channels": 768,
        "patch_size": 14,
        "dec_block_out_channels": (768, 512, 256, 128, 64),
        "default_model_name": "facebook/dinov2-with-registers-base",
    },
}


def get_encoder_config(encoder_type: str) -> Dict[str, Any]:
    """
    获取 encoder 类型的默认配置。
    
    Args:
        encoder_type: encoder 类型
    
    Returns:
        配置字典
    """
    if encoder_type not in ENCODER_CONFIGS:
        raise ValueError(f"Unknown encoder_type: {encoder_type}. "
                         f"Supported: {list(ENCODER_CONFIGS.keys())}")
    return ENCODER_CONFIGS[encoder_type].copy()


def get_latent_channels(encoder_type: str) -> int:
    """
    获取 encoder 类型的 latent channels。
    
    Args:
        encoder_type: encoder 类型
    
    Returns:
        latent channels
    """
    return get_encoder_config(encoder_type)["latent_channels"]


def get_patch_size(encoder_type: str) -> int:
    """
    获取 encoder 类型的 patch size。
    
    Args:
        encoder_type: encoder 类型
    
    Returns:
        patch size
    """
    return get_encoder_config(encoder_type)["patch_size"]


def get_dec_block_out_channels(encoder_type: str) -> Tuple[int, ...]:
    """
    获取 encoder 类型对应的默认 decoder block channels。
    
    Args:
        encoder_type: encoder 类型
    
    Returns:
        decoder block output channels
    """
    return get_encoder_config(encoder_type)["dec_block_out_channels"]
