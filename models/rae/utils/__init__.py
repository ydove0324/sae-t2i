"""
RAE (Representation Autoencoder) 工具模块

提供训练和评估所需的通用工具函数，包括:
- ddp_utils: DDP/FSDP 分布式训练工具
- image_utils: 图像预处理工具
- argparse_utils: 共享命令行参数定义
- vae_utils: VAE 模型加载和使用工具
- metrics_utils: PSNR/LPIPS 等评估指标
- optim_utils: 优化器和调度器工具
- model_utils: 模型实例化工具
- train_utils: 训练配置解析工具

使用示例:
    >>> from models.rae.utils import setup_ddp, cleanup_ddp
    >>> from models.rae.utils import get_train_transform, center_crop_arr
    >>> from models.rae.utils import add_all_vae_args, build_vae_from_args
    >>> from models.rae.utils import calculate_psnr, LPIPSLoss
"""

# DDP/FSDP 工具
from .ddp_utils import (
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    get_rank,
    get_world_size,
    barrier,
    requires_grad,
    unwrap_model,
    get_model_state_dict,
    wrap_model_ddp,
    wrap_model_fsdp,
    update_ema,
    create_logger,
    all_reduce_mean,
    all_reduce_sum,
    broadcast,
)

# 图像处理工具
from .image_utils import (
    center_crop_arr,
    random_crop_arr,
    get_train_transform,
    get_val_transform,
    get_augment_transform,
    tensor_to_pil,
    pil_to_tensor,
    denormalize_tensor,
    normalize_tensor,
    save_tensor_as_image,
    load_image_as_tensor,
)

# 命令行参数工具
from .argparse_utils import (
    add_vae_args,
    add_encoder_args,
    add_decoder_args,
    add_vit_decoder_args,
    add_lora_args,
    add_training_args,
    add_distributed_args,
    add_data_args,
    add_logging_args,
    add_latent_stats_args,
    add_all_vae_args,
    add_common_training_args,
    process_lora_args,
    process_output_dir_args,
    process_dec_channels_args,
    process_all_args,
    get_encoder_config as get_encoder_config_from_argparse,
    get_latent_channels as get_latent_channels_from_argparse,
    get_patch_size as get_patch_size_from_argparse,
    get_dec_block_out_channels as get_dec_block_out_channels_from_argparse,
    ENCODER_CONFIGS,
)

# VAE 工具
from .vae_utils import (
    # 归一化
    normalize_sae,
    denormalize_sae,
    normalize_siglip2,
    denormalize_siglip2,
    normalize_dinov2,
    denormalize_dinov2,
    get_normalize_fn,
    get_denormalize_fn,
    # Latent stats
    load_latent_stats,
    normalize_with_stats,
    denormalize_with_stats,
    LatentNormalizer,
    # Encoder 配置
    get_encoder_config,
    get_latent_channels,
    get_patch_size,
    get_dec_block_out_channels,
    build_vae_model_params,
    build_vae_from_args,
    # VAE 加载
    load_dinov3_vae,
    load_vae,
    # 重建
    reconstruct_from_latent_with_diffusion,
)

# 评估指标工具
from .metrics_utils import (
    calculate_psnr,
    calculate_batch_psnr,
    calculate_psnr_per_sample,
    LPIPSLoss,
    l1_loss,
    l2_loss,
    ReconstructionLoss,
    calculate_fid,
    is_fid_available,
    is_lpips_available,
    HAS_LPIPS,
    HAS_FID,
)

# 其他现有模块的导入（保持向后兼容）
try:
    from .optim_utils import build_optimizer, build_scheduler
except ImportError:
    pass

try:
    from .model_utils import instantiate_from_config
except ImportError:
    pass

try:
    from .train_utils import parse_configs
except ImportError:
    pass

__all__ = [
    # DDP utils
    "setup_ddp",
    "cleanup_ddp",
    "is_main_process",
    "get_rank",
    "get_world_size",
    "barrier",
    "requires_grad",
    "unwrap_model",
    "get_model_state_dict",
    "wrap_model_ddp",
    "wrap_model_fsdp",
    "update_ema",
    "create_logger",
    "all_reduce_mean",
    "all_reduce_sum",
    "broadcast",
    # Image utils
    "center_crop_arr",
    "random_crop_arr",
    "get_train_transform",
    "get_val_transform",
    "get_augment_transform",
    "tensor_to_pil",
    "pil_to_tensor",
    "denormalize_tensor",
    "normalize_tensor",
    "save_tensor_as_image",
    "load_image_as_tensor",
    # Argparse utils
    "add_vae_args",
    "add_encoder_args",
    "add_decoder_args",
    "add_vit_decoder_args",
    "add_lora_args",
    "add_training_args",
    "add_distributed_args",
    "add_data_args",
    "add_logging_args",
    "add_latent_stats_args",
    "add_all_vae_args",
    "add_common_training_args",
    "process_lora_args",
    "process_output_dir_args",
    "process_dec_channels_args",
    "process_all_args",
    "ENCODER_CONFIGS",
    # VAE utils
    "normalize_sae",
    "denormalize_sae",
    "normalize_siglip2",
    "denormalize_siglip2",
    "normalize_dinov2",
    "denormalize_dinov2",
    "get_normalize_fn",
    "get_denormalize_fn",
    "load_latent_stats",
    "normalize_with_stats",
    "denormalize_with_stats",
    "LatentNormalizer",
    "get_encoder_config",
    "get_latent_channels",
    "get_patch_size",
    "get_dec_block_out_channels",
    "build_vae_model_params",
    "build_vae_from_args",
    "load_dinov3_vae",
    "load_vae",
    "reconstruct_from_latent_with_diffusion",
    # Metrics utils
    "calculate_psnr",
    "calculate_batch_psnr",
    "calculate_psnr_per_sample",
    "LPIPSLoss",
    "l1_loss",
    "l2_loss",
    "ReconstructionLoss",
    "calculate_fid",
    "is_fid_available",
    "is_lpips_available",
    "HAS_LPIPS",
    "HAS_FID",
]
