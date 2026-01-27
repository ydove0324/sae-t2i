# config_utils.py
"""
配置加载和解析工具
支持 YAML 配置文件和命令行参数覆盖
"""
import os
import yaml
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class DataConfig:
    train_path: str = "/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/train/"
    val_path: str = "/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/val/"
    image_size: int = 256
    batch_size: int = 16


@dataclass
class DINOv3Config:
    model_dir: str = "/cpfs01/huangxu/models/dinov3"
    hidden_size: int = 1280
    patch_size: int = 16


@dataclass
class SigLIP2Config:
    model_name: str = "google/siglip2-base-patch16-256"
    num_tokens: int = 256
    hidden_size: int = 768


@dataclass
class EncoderConfig:
    type: str = "dinov3"  # "dinov3" 或 "siglip2"
    dinov3: DINOv3Config = field(default_factory=DINOv3Config)
    siglip2: SigLIP2Config = field(default_factory=SigLIP2Config)


@dataclass
class DecoderConfig:
    out_channels: int = 3
    latent_channels: int = 1280
    target_latent_channels: Optional[int] = None
    spatial_downsample_factor: int = 16
    block_out_channels: Tuple[int, ...] = (1280, 1024, 512, 256, 128)
    layers_per_block: int = 3
    dropout: float = 0.0
    gradient_checkpointing: bool = False
    denormalize_output: bool = False


@dataclass
class VAEConfig:
    variational: bool = True
    noise_tau: float = 0.0
    mask_channels: float = 0.0


@dataclass
class LoRAConfig:
    enabled: bool = True
    rank: int = 256
    alpha: int = 256
    dropout: float = 0.0


@dataclass
class VFLossConfig:
    enabled: bool = True
    weight: float = 0.1
    margin_cos: float = 0.1
    margin_dms: float = 0.1
    max_tokens: int = 256
    adaptive_weight: bool = True


@dataclass
class GANLossConfig:
    enabled: bool = True
    weight: float = 0.01
    start_step: int = 0
    max_d_weight: float = 10000.0
    stage_disc_steps: int = 0


@dataclass
class LossConfig:
    l1_weight: float = 1.0
    lpips_weight: float = 0.5
    kl_weight: float = 1e-6
    vf: VFLossConfig = field(default_factory=VFLossConfig)
    gan: GANLossConfig = field(default_factory=GANLossConfig)


@dataclass
class DiscriminatorConfig:
    ndf: int = 64
    n_layers: int = 4
    norm: str = "gn"
    lr: float = 2e-5


@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.5, 0.9)
    weight_decay: float = 0.01


@dataclass
class TrainingConfig:
    max_steps: int = 100000
    precision: str = "bf16"
    fsdp_size: int = 1


@dataclass
class LoggingConfig:
    output_dir: str = "results_vae/default"
    log_every: int = 10
    eval_every: int = 2000
    save_every: int = 5000
    val_max_batches: int = 2000
    debug: bool = False


@dataclass
class CheckpointConfig:
    vae_ckpt: str = ""


@dataclass
class TrainConfig:
    """完整的训练配置"""
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


def dict_to_dataclass(d: dict, cls):
    """递归地将字典转换为 dataclass"""
    if d is None:
        return cls()
    
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    
    for key, value in d.items():
        if key not in field_types:
            continue
        
        field_type = field_types[key]
        
        # 处理嵌套的 dataclass
        if hasattr(field_type, '__dataclass_fields__'):
            kwargs[key] = dict_to_dataclass(value, field_type)
        # 处理 Tuple
        elif hasattr(field_type, '__origin__') and field_type.__origin__ is tuple:
            kwargs[key] = tuple(value) if value else ()
        # 处理 Optional
        elif hasattr(field_type, '__origin__') and field_type.__origin__ is type(None):
            kwargs[key] = value
        else:
            kwargs[key] = value
    
    return cls(**kwargs)


def load_config(config_path: str) -> TrainConfig:
    """从 YAML 文件加载配置"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return dict_to_dataclass(config_dict, TrainConfig)


def config_to_args(config: TrainConfig):
    """将配置转换为类似 argparse 的 namespace"""
    args = argparse.Namespace()
    
    # Data
    args.train_path = config.data.train_path
    args.val_path = config.data.val_path
    args.image_size = config.data.image_size
    args.batch_size = config.data.batch_size
    
    # Encoder
    args.encoder_type = config.encoder.type
    args.dinov3_dir = config.encoder.dinov3.model_dir
    args.siglip2_model_name = config.encoder.siglip2.model_name
    args.siglip2_num_tokens = config.encoder.siglip2.num_tokens
    
    # Decoder
    args.out_channels = config.decoder.out_channels
    args.latent_channels = config.decoder.latent_channels
    args.target_latent_channels = config.decoder.target_latent_channels
    args.spatial_downsample_factor = config.decoder.spatial_downsample_factor
    args.dec_block_out_channels = config.decoder.block_out_channels
    args.dec_layers_per_block = config.decoder.layers_per_block
    args.decoder_dropout = config.decoder.dropout
    args.gradient_checkpointing = config.decoder.gradient_checkpointing
    args.denormalize_output = config.decoder.denormalize_output
    
    # VAE
    args.variational = config.vae.variational
    args.noise_tau = config.vae.noise_tau
    args.mask_channels = config.vae.mask_channels
    
    # LoRA
    args.enable_lora = config.lora.enabled
    args.lora_rank = config.lora.rank
    args.lora_alpha = config.lora.alpha
    args.lora_dropout = config.lora.dropout
    
    # Loss
    args.l1_weight = config.loss.l1_weight
    args.lpips_weight = config.loss.lpips_weight
    args.kl_weight = config.loss.kl_weight
    
    # VF Loss
    args.vf_enabled = config.loss.vf.enabled
    args.vf_weight = config.loss.vf.weight
    args.vf_m1 = config.loss.vf.margin_cos
    args.vf_m2 = config.loss.vf.margin_dms
    args.vf_max_tokens = config.loss.vf.max_tokens
    args.vf_adaptive_weight = config.loss.vf.adaptive_weight
    
    # GAN Loss
    args.gan_enabled = config.loss.gan.enabled
    args.gan_weight = config.loss.gan.weight if config.loss.gan.enabled else 0.0
    args.gan_start_step = config.loss.gan.start_step
    args.max_d_weight = config.loss.gan.max_d_weight
    args.stage_disc_steps = config.loss.gan.stage_disc_steps
    
    # Discriminator
    args.disc_ndf = config.discriminator.ndf
    args.disc_n_layers = config.discriminator.n_layers
    args.disc_norm = config.discriminator.norm
    args.disc_lr = config.discriminator.lr
    
    # Optimizer
    args.lr = config.optimizer.lr
    args.betas = config.optimizer.betas
    args.weight_decay = config.optimizer.weight_decay
    
    # Training
    args.max_steps = config.training.max_steps
    args.precision = config.training.precision
    args.fsdp_size = config.training.fsdp_size
    
    # Logging
    args.output_dir = config.logging.output_dir
    args.log_every = config.logging.log_every
    args.eval_every = config.logging.eval_every
    args.save_every = config.logging.save_every
    args.val_max_batches = config.logging.val_max_batches
    args.debug = config.logging.debug
    
    # Checkpoint
    args.vae_ckpt = config.checkpoint.vae_ckpt
    
    return args


def get_args_parser():
    """创建命令行参数解析器，支持配置文件和命令行覆盖"""
    parser = argparse.ArgumentParser(description="VAE Training with Config")
    
    # 配置文件
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    
    # 以下参数可以用命令行覆盖配置文件
    # Data
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--val-path", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    
    # Encoder
    parser.add_argument("--encoder-type", type=str, default=None,
                        choices=["dinov3", "siglip2"])
    parser.add_argument("--dinov3-dir", type=str, default=None)
    parser.add_argument("--siglip2-model-name", type=str, default=None)
    
    # LoRA
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA")
    
    # Loss weights
    parser.add_argument("--l1-weight", type=float, default=None)
    parser.add_argument("--lpips-weight", type=float, default=None)
    parser.add_argument("--kl-weight", type=float, default=None)
    parser.add_argument("--vf-weight", type=float, default=None)
    parser.add_argument("--vf-m1", type=float, default=None)
    parser.add_argument("--vf-m2", type=float, default=None)
    parser.add_argument("--vf-max-tokens", type=int, default=None)
    parser.add_argument("--no-vf", action="store_true", help="Disable VF loss")
    
    # GAN
    parser.add_argument("--gan-weight", type=float, default=None)
    parser.add_argument("--gan-start-step", type=int, default=None)
    parser.add_argument("--disc-lr", type=float, default=None)
    parser.add_argument("--disc-ndf", type=int, default=None)
    parser.add_argument("--disc-n-layers", type=int, default=None)
    parser.add_argument("--stage-disc-steps", type=int, default=None)
    parser.add_argument("--no-gan", action="store_true", help="Disable GAN")
    
    # Training
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--precision", type=str, default=None, choices=["fp32", "bf16"])
    parser.add_argument("--fsdp-size", type=int, default=None)
    
    # Logging
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--val-max-batches", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    
    # Checkpoint
    parser.add_argument("--vae-ckpt", type=str, default=None)
    
    # VAE
    parser.add_argument("--noise-tau", type=float, default=None)
    parser.add_argument("--mask-channels", type=float, default=None)
    
    return parser


def load_and_merge_config(cli_args=None):
    """
    加载配置：
    1. 如果指定了 --config，从 YAML 加载
    2. 用命令行参数覆盖配置文件中的值
    """
    parser = get_args_parser()
    cli_args = parser.parse_args(cli_args)
    
    # 从配置文件加载基础配置
    if cli_args.config:
        config = load_config(cli_args.config)
    else:
        # 使用默认配置
        config = TrainConfig()
    
    # 转换为 args 格式
    args = config_to_args(config)
    
    # 用命令行参数覆盖
    cli_dict = vars(cli_args)
    args_dict = vars(args)
    
    for key, value in cli_dict.items():
        if key == 'config':
            continue
        if key == 'no_lora' and value:
            args.enable_lora = False
            continue
        if key == 'no_vf' and value:
            args.vf_enabled = False
            args.vf_weight = 0.0
            continue
        if key == 'no_gan' and value:
            args.gan_enabled = False
            args.gan_weight = 0.0
            continue
        if value is not None:
            # 将 cli 的 key (带 -) 转换为 args 的 key (带 _)
            args_key = key.replace('-', '_')
            if args_key in args_dict:
                setattr(args, args_key, value)
    
    return args


def print_config(args):
    """打印配置信息"""
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    
    sections = {
        "Data": ["train_path", "val_path", "image_size", "batch_size"],
        "Encoder": ["encoder_type", "dinov3_dir", "siglip2_model_name"],
        "LoRA": ["enable_lora", "lora_rank", "lora_alpha", "lora_dropout"],
        "Loss": ["l1_weight", "lpips_weight", "kl_weight"],
        "VF Loss": ["vf_enabled", "vf_weight", "vf_m1", "vf_m2", "vf_max_tokens"],
        "GAN": ["gan_enabled", "gan_weight", "gan_start_step", "disc_lr", "stage_disc_steps"],
        "Training": ["lr", "max_steps", "precision", "fsdp_size"],
        "Logging": ["output_dir", "log_every", "eval_every", "save_every", "debug"],
    }
    
    args_dict = vars(args)
    for section_name, keys in sections.items():
        print(f"\n[{section_name}]")
        for key in keys:
            if key in args_dict:
                print(f"  {key}: {args_dict[key]}")
    
    print("=" * 60)


if __name__ == "__main__":
    # 测试配置加载
    args = load_and_merge_config()
    print_config(args)
