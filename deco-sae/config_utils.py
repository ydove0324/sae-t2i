import argparse
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class DataConfig:
    train_path: str = "/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/train/"
    val_path: str = "/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/val/"
    image_size: int = 256
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class EncoderConfig:
    type: str = "dinov2"
    dinov3_model_dir: str = "/cpfs01/huangxu/models/dinov3"
    siglip2_model_name: str = "google/siglip2-base-patch16-256"
    dinov2_model_name: str = "/cpfs01/huangxu/models/dinov2-register-base"


@dataclass
class ModelConfig:
    hidden_size: int = 1152
    num_decoder_blocks: int = 12
    nerf_max_freqs: int = 8
    flow_steps: int = 25
    time_dim: int = 256

    enable_hf_branch: bool = True
    hf_dropout_prob: float = 0.4
    hf_noise_std: float = 0.1
    hf_loss_weight: float = 0.1

    target_latent_channels: Optional[int] = None
    variational: bool = False
    kl_weight: float = 1e-8
    skip_to_moments: bool = True
    noise_tau: float = 0.0
    random_masking_channel_ratio: float = 0.0
    denormalize_decoder_output: bool = False

    lora_rank: int = 0
    lora_alpha: int = 0
    lora_dropout: float = 0.0
    enable_lora: bool = False


@dataclass
class LossConfig:
    recon_l2_weight: float = 1.0
    recon_l1_weight: float = 0.0
    recon_lpips_weight: float = 0.0
    recon_gan_weight: float = 0.0


@dataclass
class DiscriminatorConfig:
    enabled: bool = False
    disc_type: str = "dino"  # "dino" | "patchgan"
    lr: float = 2e-4
    start_step: int = 0
    weight_decay: float = 0.0
    betas0: float = 0.5
    betas1: float = 0.9

    # DinoDisc options
    dino_ckpt_path: str = "./dino_vit_small_patch8_224.pth"
    recipe: str = "S_8"
    ks: int = 3
    norm: str = "bn"
    key_depths: str = "2,5,8,11"
    diffaug_prob: float = 1.0
    diffaug_cutout: float = 0.2

    # PatchGAN options
    ndf: int = 64
    n_layers: int = 4


@dataclass
class OptimizerConfig:
    lr: float = 2e-4
    min_lr: float = 2e-5
    betas0: float = 0.9
    betas1: float = 0.999
    weight_decay: float = 0.0
    warmup_steps: int = 0
    scheduler: str = "cosine"


@dataclass
class TrainingConfig:
    max_steps: int = 100000
    precision: str = "bf16"
    seed: int = 42


@dataclass
class LoggingConfig:
    output_dir: str = "results_sae/default"
    log_every: int = 50
    eval_every: int = 2000
    save_every: int = 5000
    val_max_batches: int = 200


@dataclass
class CheckpointConfig:
    sae_ckpt: str = ""
    strict_load: bool = False


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


def _merge_dataclass(dc_obj, updates: dict):
    if updates is None:
        return dc_obj
    for key, value in updates.items():
        if not hasattr(dc_obj, key):
            continue
        field_value = getattr(dc_obj, key)
        if hasattr(field_value, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(field_value, value)
        else:
            setattr(dc_obj, key, value)
    return dc_obj


def load_config(config_path: str) -> TrainConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = TrainConfig()
    _merge_dataclass(cfg, raw)
    return cfg


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DecoSAE with Flow Matching")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")

    # Common runtime overrides.
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--val-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--scheduler", type=str, default=None, choices=["constant", "cosine"])
    parser.add_argument("--precision", type=str, default=None, choices=["fp32", "bf16"])

    parser.add_argument("--sae-ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def load_and_merge_config(cli_args=None) -> TrainConfig:
    parser = get_args_parser()
    args = parser.parse_args(cli_args)

    cfg = load_config(args.config)

    if args.train_path is not None:
        cfg.data.train_path = args.train_path
    if args.val_path is not None:
        cfg.data.val_path = args.val_path
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.image_size is not None:
        cfg.data.image_size = args.image_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    if args.output_dir is not None:
        cfg.logging.output_dir = args.output_dir
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    if args.lr is not None:
        cfg.optimizer.lr = args.lr
    if args.min_lr is not None:
        cfg.optimizer.min_lr = args.min_lr
    if args.warmup_steps is not None:
        cfg.optimizer.warmup_steps = args.warmup_steps
    if args.scheduler is not None:
        cfg.optimizer.scheduler = args.scheduler
    if args.precision is not None:
        cfg.training.precision = args.precision
    if args.sae_ckpt is not None:
        cfg.checkpoint.sae_ckpt = args.sae_ckpt
    if args.seed is not None:
        cfg.training.seed = args.seed

    return cfg
