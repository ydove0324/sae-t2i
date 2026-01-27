# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.

Features:
- Stage2 flow-matching training in DINOv3 latent space
- prediction_mode: x-pred (predict x0) or v-pred (predict v = eps - x0)
- VAE decoder_type: diffusion_decoder (sae_model.AutoencoderKL) or cnn_decoder (cnn_decoder.AutoencoderKL)
- dataset_type: overfit_single_image or single_class
- periodic visualization (generation + recon/noisy/gt)
"""

import os
import sys
import math
import gc
import argparse
import logging
from glob import glob
from time import time
from copy import deepcopy
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image

from omegaconf import OmegaConf

sys.path.append(".")

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from models.rae.stage2.models import Stage2ModelProtocol
from models.rae.utils.train_utils import parse_configs
from models.rae.utils.model_utils import instantiate_from_config
from models.rae.utils import wandb_utils
from models.rae.utils.optim_utils import build_optimizer, build_scheduler
from models.rae.utils.vae_utils import (
    load_dinov3_vae,
    load_vae,
    normalize_sae,
    denormalize_sae,
    get_normalize_fn,
    get_denormalize_fn,
    reconstruct_from_latent_with_diffusion,
)

# === Datasets ===
from dataset import ImageNetIdxDataset, OverfitSingleImageDataset, SingleClassDataset
from torch.amp import autocast


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    Supports EMA on CPU and model on GPU.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    ema_device = next(ema_model.parameters()).device
    for name, param in model_params.items():
        # 如果 EMA 在 CPU 上，需要先把 model 参数复制到 CPU
        param_data = param.data.to(ema_device) if param.device != ema_device else param.data
        ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)


@torch.no_grad()
def update_ema_fsdp(ema_model, fsdp_model, decay=0.9999):
    """
    Update EMA from FSDP wrapped model.
    Uses summon_full_params to get full parameters from FSDP shards.
    """
    ema_device = next(ema_model.parameters()).device
    
    # 使用 summon_full_params 临时获取完整参数
    with FSDP.summon_full_params(fsdp_model, writeback=False, recurse=True):
        model_inner = fsdp_model.module if hasattr(fsdp_model, 'module') else fsdp_model
        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(model_inner.named_parameters())
        
        for name, param in model_params.items():
            if name in ema_params:
                param_data = param.data.to(ema_device) if param.device != ema_device else param.data
                ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def unwrap_model(model):
    """
    获取 DDP 或 FSDP 包装模型的内部原始模型。
    """
    if isinstance(model, FSDP):
        return model.module
    elif isinstance(model, DDP):
        return model.module
    elif hasattr(model, "module"):
        return model.module
    return model


def get_model_state_dict(model):
    """
    获取模型状态字典，兼容 DDP 和 FSDP。
    对于 FSDP，需要使用 full_state_dict 来收集所有分片。
    """
    if isinstance(model, FSDP):
        # FSDP 需要特殊处理来收集完整的状态字典
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            return model.state_dict()
    elif isinstance(model, DDP):
        return model.module.state_dict()
    elif hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt") if logging_dir is not None else logging.StreamHandler(),
            ],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                         Latent & Diffusion Sampling Logic                     #
#################################################################################

def compute_train_loss(
    model,
    x_latent: torch.Tensor,
    model_kwargs: dict,
    time_shift: float,
    pred_mode: str = "x",
):
    """
    Flow matching on linear path:
      x_t = t * eps + (1 - t) * x0,  t in [0,1]

    pred_mode:
      - "x": model predicts x0
      - "v": model predicts v = dx/dt = eps - x0
    """
    B = x_latent.size(0)
    device = x_latent.device

    noise = torch.randn_like(x_latent)

    # t ~ Uniform[0,1], shifted
    t = torch.rand(B, device=device)
    t = (time_shift * t) / (1.0 + (time_shift - 1.0) * t)
    t_broadcast = t.view(B, 1, 1, 1)

    x_t = t_broadcast * noise + (1.0 - t_broadcast) * x_latent

    model_output = model(x_t, t, **model_kwargs)

    if pred_mode == "x":
        target = x_latent
        loss = F.mse_loss(model_output, target, reduction="none")
        # x-pred reweighting to balance boundaries
        reweight_scale = torch.clamp_max(1.0 / (t**2 + 1e-8), 20.0)
        loss = loss * reweight_scale.view(B, 1, 1, 1)

    elif pred_mode == "v":
        target = noise - x_latent
        loss = F.mse_loss(model_output, target, reduction="none")
        # v-pred usually uses uniform weighting

    else:
        raise ValueError(f"Unknown pred_mode: {pred_mode}")

    loss = loss.mean()
    return loss, model_output, noise, t


@torch.no_grad()
def sample_latent_linear_50_steps(
    model,
    batch_size: int,
    latent_shape: tuple,   # (C,H,W)
    device: torch.device,
    y: torch.Tensor,
    time_shift: float,
    steps: int = 50,
    init_noise: torch.Tensor | None = None,
    pred_mode: str = "x",
):
    """
    Deterministic solver in latent space, stepping t from 1 -> 0.
    Supports both x-pred and v-pred.

    Note: this is an Euler-style integrator for v-pred, and DDIM-like update for x-pred.
    """
    # 对于 FSDP/DDP 模型，直接使用包装后的模型进行推理
    # FSDP 在 forward 时会自动 all-gather 参数
    was_training = model.training
    model.eval()

    C, H, W = latent_shape
    if init_noise is None:
        x = torch.randn((batch_size, C, H, W), device=device, dtype=torch.float32)
    else:
        x = init_noise.to(device=device, dtype=torch.float32)

    shift = float(time_shift)

    def flow_shift(t_lin: torch.Tensor) -> torch.Tensor:
        t = (shift * t_lin) / (1.0 + (shift - 1.0) * t_lin)
        return t.clamp(0.0, 1.0 - 1e-6)

    for i in range(steps, 0, -1):
        t_lin = torch.full((batch_size,), i / steps, device=device, dtype=torch.float32)
        t_next_lin = torch.full((batch_size,), (i - 1) / steps, device=device, dtype=torch.float32)

        t = flow_shift(t_lin)
        t_next = flow_shift(t_next_lin)

        model_out = model(x, t, y=y)

        t_scalar = t.view(batch_size, 1, 1, 1)
        t_next_scalar = t_next.view(batch_size, 1, 1, 1)

        if pred_mode == "x":
            # x-pred DDIM-like step:
            x0_hat = model_out
            eps_hat = (x - (1.0 - t_scalar) * x0_hat) / (t_scalar + 1e-8)
            x = t_next_scalar * eps_hat + (1.0 - t_next_scalar) * x0_hat

        elif pred_mode == "v":
            # v-pred Euler step for ODE dx/dt = v:
            v_pred = model_out
            dt = t_next_scalar - t_scalar
            x = x + v_pred * dt

        else:
            raise ValueError(f"Unknown pred_mode: {pred_mode}")

    # 恢复模型原来的训练状态
    if was_training:
        model.train()

    return x


@torch.no_grad()
def stage2_sample_and_reconstruct(
    model,
    dinov3_vae,
    y: torch.Tensor,
    image_size: int = 256,
    latent_shape: tuple = (1280, 16, 16),
    stage2_steps: int = 50,
    vae_diffusion_steps: int = 25,
    time_shift: float = 1.0,
    device: torch.device | None = None,
    pred_mode: str = "x",
    decoder_type: str = "diffusion_decoder",
    encoder_type: str = "dinov3",
):
    if device is None:
        device = y.device

    B = y.shape[0]
    y = y.to(device)

    z0_hat = sample_latent_linear_50_steps(
        model=model,
        batch_size=B,
        latent_shape=latent_shape,
        device=device,
        y=y,
        time_shift=time_shift,
        steps=stage2_steps,
        pred_mode=pred_mode,
    )

    img = reconstruct_from_latent_with_diffusion(
        vae=dinov3_vae,
        latent_z=z0_hat,
        image_shape=torch.Size([B, 3, image_size, image_size]),
        diffusion_steps=vae_diffusion_steps,
        decoder_type=decoder_type,
        encoder_type=encoder_type,
    )
    return img, z0_hat


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Training currently requires at least one GPU.")

    (
        rae_config,
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

    training_cfg = to_dict(training_config)

    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    clip_grad = float(training_cfg.get("clip_grad", 1.0))
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    epochs = int(training_cfg.get("epochs", 1400))
    num_workers = int(training_cfg.get("num_workers", 4))
    log_every = int(training_cfg.get("log_every", 10))
    ckpt_every = int(training_cfg.get("ckpt_every", 5_000))
    default_seed = int(training_cfg.get("global_seed", 0))
    global_seed = args.global_seed if args.global_seed is not None else default_seed

    if args.global_batch_size is not None:
        global_batch_size = args.global_batch_size
    else:
        global_batch_size = int(training_cfg.get("global_batch_size", 1024))

    if args.image_size % 16 != 0:
        raise ValueError("Image size must be divisible by 16 for the VAE encoder (downsample_factor=16).")

    # ---------------- DDP init ----------------
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()

    if (global_batch_size * args.fsdp_size) % (world_size * grad_accum_steps) != 0:
        raise ValueError("Global batch size must be divisible by world_size * grad_accum_steps.")

    rank = dist.get_rank()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    micro_batch_size = global_batch_size * args.fsdp_size // (world_size * grad_accum_steps)

    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")

    # ---------------- time_shift & latent size ----------------
    # Determine latent_channels based on encoder_type for time_shift calculation
    _latent_c = 1280 if args.encoder_type == "dinov3" else 768  # SigLIP2-base hidden size
    shift_dim = (args.image_size // 16) * (args.image_size // 16) * _latent_c
    shift_base = 4096
    time_shift = math.sqrt(shift_dim / shift_base)

    # ---------------- experiment / logging / wandb ----------------
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*")) - 1

        model_target = str(model_config.get("target", "stage2"))
        model_string_name = model_target.split(".")[-1]
        precision_suffix = f"-{args.precision}" if args.precision == "bf16" else ""

        experiment_name = (
            f"{experiment_index:03d}-{model_string_name}-"
            f"flowmatch-{args.prediction_mode}pred-{args.decoder_type}{precision_suffix}-acc{grad_accum_steps}"
        )

        experiment_dir = os.path.join(args.results_dir, "experiment")
        os.makedirs(experiment_dir, exist_ok=True)
        checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"Prediction Mode: {args.prediction_mode.upper()}")
        logger.info(f"Decoder Type: {args.decoder_type}")

        # TensorBoard
        tb_dir = os.path.join(experiment_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)

        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        experiment_dir = None
        checkpoint_dir = None
        logger = create_logger(None)
        writer = None

    # ---------------- load VAE (DINOv3 or SigLIP2) ----------------
    # Determine latent_channels based on encoder_type
    if args.encoder_type == "dinov3":
        latent_channels = 1280
        default_dec_block_out_channels = (1280, 1024, 512, 256, 128)
    elif args.encoder_type == "siglip2":
        latent_channels = 768  # SigLIP2-base hidden size
        default_dec_block_out_channels = (768, 512, 256, 128, 64)
    else:
        raise ValueError(f"Unknown encoder_type: {args.encoder_type}")
    
    # Build model params for VAE
    vae_model_params = {
        "encoder_type": args.encoder_type,
        "image_size": args.image_size,
        "patch_size": 16,
        "out_channels": 3,
        "latent_channels": latent_channels,
        "target_latent_channels": None,
        "spatial_downsample_factor": 16,
        "lora_rank": 256,
        "lora_alpha": 256,
        "dec_block_out_channels": default_dec_block_out_channels,
        "dec_layers_per_block": 3,
        "decoder_dropout": 0.0,
        "gradient_checkpointing": False,
        "denormalize_decoder_output": False,
    }
    
    # Add encoder-specific params
    if args.encoder_type == "dinov3":
        vae_model_params["dinov3_model_dir"] = args.dinov3_dir
    elif args.encoder_type == "siglip2":
        vae_model_params["siglip2_model_name"] = args.siglip2_model_name
    
    dinov3_vae = load_vae(
        args.vae_ckpt, 
        device, 
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        model_params=vae_model_params,
    )
    
    # Get normalization function based on encoder_type
    normalize_fn = get_normalize_fn(args.encoder_type)
    
    logger.info(f"Encoder type: {args.encoder_type}, Latent channels: {latent_channels}")

    # ---------------- Stage2 model ----------------
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    
    # EMA 可以放在 CPU 上以节省 GPU 显存
    ema_device = torch.device("cpu") if args.ema_cpu else device
    ema = deepcopy(model).to(ema_device)
    requires_grad(ema, False)
    if args.ema_cpu:
        logger.info("EMA model kept on CPU to save GPU memory.")

    opt_state = None
    sched_state = None
    train_steps = 0

    # ---------------- optional resume ----------------
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        opt_state = checkpoint.get("opt")
        sched_state = checkpoint.get("scheduler")
        train_steps = int(checkpoint.get("train_steps", 0))

    model_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {model_param_count/1e6:.2f}M")

    # 根据 fsdp_size 选择 DDP 或 FSDP
    if args.fsdp_size > 1:
        # 使用 FSDP 分割优化器状态，节省显存
        fsdp_mixed_precision = None
        if args.precision == "bf16":
            fsdp_mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=fsdp_mixed_precision,
            device_id=device_idx,
            use_orig_params=True,  # 允许访问原始参数，方便 optimizer
        )
        logger.info(f"Using FSDP with sharding_strategy=FULL_SHARD, fsdp_size={args.fsdp_size}")
    else:
        model = DDP(model, device_ids=[device_idx], gradient_as_bucket_view=False)
        logger.info("Using DDP")

    opt, opt_msg = build_optimizer(model.parameters(), training_cfg)
    if opt_state is not None:
        opt.load_state_dict(opt_state)

    # ---------------- AMP 设置 ----------------
    use_amp = args.precision == "bf16"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    logger.info(f"AMP enabled: {use_amp}, dtype: {amp_dtype}")

    # ---------------- dataset / loader ----------------
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),                       # [0,1]
        transforms.Lambda(lambda t: t * 2.0 - 1.0),  # [-1,1]
    ])

    index_synset_path = "/share/project/datasets/ImageNet/train/index_synset.yaml"

    if args.dataset_type == "overfit_single_image":
        image_path = args.overfit_image_path
        dataset = OverfitSingleImageDataset(
            image_path=image_path,
            length=args.overfit_length,
            label=args.overfit_label,
            transform=transform,
        )
    elif args.dataset_type == "single_class":
        dataset = SingleClassDataset(
            root=args.data_path,
            index_synset_path=index_synset_path,
            target_class_index=args.single_class_index,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=global_seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    logger.info(f"Dataset contains {len(dataset):,} samples")
    loader_batches = len(loader)
    steps_per_epoch = max(1, loader_batches // grad_accum_steps)
    schedl, sched_msg = build_scheduler(opt, steps_per_epoch, training_cfg, sched_state)

    # Init EMA
    use_fsdp = args.fsdp_size > 1
    if use_fsdp:
        update_ema_fsdp(ema, model, decay=0.0)
    else:
        update_ema(ema, unwrap_model(model), decay=0.0)
    model.train()
    ema.eval()

    logger.info(f"Training for {epochs} epochs...")
    # You overwrote log_every to 10 in your first script; keep configurable but default to config value.
    if args.force_log_every is not None:
        log_every = int(args.force_log_every)

    log_steps = 0
    running_loss = 0.0
    start_time = time()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        opt.zero_grad()
        accum_counter = 0
        step_loss_accum = 0.0

        for x_img, y in loader:
            x_img = x_img.to(device)
            y = y.to(device)

            img_gt = (x_img + 1.0) / 2.0
            # print("step",train_steps)
            with torch.no_grad():
                x_latent, _p = dinov3_vae.encode(x_img)
                x_latent = normalize_fn(x_latent)
            model_kwargs = dict(y=y)

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                loss_tensor, pred_latent, noise, t_sample = compute_train_loss(
                    model=model,
                    x_latent=x_latent,
                    model_kwargs=model_kwargs,
                    time_shift=time_shift,
                    pred_mode=args.prediction_mode,
                )

            # NaN check across ranks
            is_nan_local = 0 if torch.isfinite(loss_tensor) else 1
            is_nan = torch.tensor(is_nan_local, device=device)
            dist.all_reduce(is_nan, op=dist.ReduceOp.SUM)

            if is_nan.item() > 0:
                if rank == 0:
                    logger.warning(f"[step {train_steps}] NaN detected! Skipping step.")
                dist.barrier()
                opt.zero_grad(set_to_none=True)
                continue

            step_loss_accum += loss_tensor.item()
            loss_tensor = loss_tensor / grad_accum_steps
            loss_tensor.backward()
            accum_counter += 1

            if accum_counter < grad_accum_steps:
                continue

            if clip_grad > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            else:
                grad_norm = torch.tensor(0.0, device=device)

            opt.step()
            schedl.step()
            if use_fsdp:
                update_ema_fsdp(ema, model, decay=ema_decay)
            else:
                update_ema(ema, unwrap_model(model), decay=ema_decay)
            opt.zero_grad(set_to_none=True)

            running_loss += step_loss_accum / grad_accum_steps
            log_steps += 1
            train_steps += 1
            accum_counter = 0
            step_loss_accum = 0.0

            # ---------------- Visualization ----------------
            # 注意：FSDP 模型的 forward 需要所有 rank 同时执行
            # 所以所有 rank 都要参与采样，只让 rank 0 保存图片
            show_time = args.vis_every
            DEBUG = False
            if (rank // args.fsdp_size == 0) and ((train_steps % show_time == 0) or DEBUG):
                with torch.no_grad():
                    y_sample = y[:1]

                    # 1) Full sampling (generation) - 所有 rank 都执行
                    img_gen, z_gen = stage2_sample_and_reconstruct(
                        model=model,
                        dinov3_vae=dinov3_vae,
                        y=y_sample,
                        image_size=args.image_size,
                        latent_shape=(latent_channels, args.image_size // 16, args.image_size // 16),
                        stage2_steps=args.stage2_steps,
                        vae_diffusion_steps=args.vae_diffusion_steps,
                        time_shift=time_shift,
                        device=device,
                        pred_mode=args.prediction_mode,
                        decoder_type=args.decoder_type,
                        encoder_type=args.encoder_type,
                    )

                    # 只有 rank 0 保存图片
                    if rank == 0:
                        img_gen_01 = (img_gen + 1.0) / 2.0
                        vis_dir = os.path.join(experiment_dir, "samples")
                        os.makedirs(vis_dir, exist_ok=True)
                        save_image(img_gen_01.clamp(0, 1), os.path.join(vis_dir, f"gen_step_{train_steps:07d}.png"))

                        # 2) One-step reconstruction monitor
                        if args.prediction_mode == "x":
                            x0_pred_step = pred_latent[0:1]
                        else:
                            # v = eps - x0  => x0 = eps - v
                            x0_pred_step = noise[0:1] - pred_latent[0:1]

                        recon = reconstruct_from_latent_with_diffusion(
                            vae=dinov3_vae,
                            latent_z=x0_pred_step,
                            image_shape=torch.Size([1, 3, args.image_size, args.image_size]),
                            diffusion_steps=args.vae_diffusion_steps,
                            decoder_type=args.decoder_type,
                            encoder_type=args.encoder_type,
                        )

                        # Original noisy latent reconstruction monitor
                        # x_t = t*eps + (1-t)*x0
                        t0 = t_sample[0].view(1, 1, 1, 1)
                        x_latent_noisy = x_latent[0:1] * (1.0 - t0) + noise[0:1] * t0

                        x_recon_noisy = reconstruct_from_latent_with_diffusion(
                            vae=dinov3_vae,
                            latent_z=x_latent_noisy,
                            image_shape=torch.Size([1, 3, args.image_size, args.image_size]),
                            diffusion_steps=args.vae_diffusion_steps,
                            decoder_type=args.decoder_type,
                            encoder_type=args.encoder_type,
                        )

                        recon_img = (recon + 1.0) / 2.0
                        x_recon_img = (x_recon_noisy + 1.0) / 2.0

                        vis_dir_recon = os.path.join(experiment_dir, "recon")
                        os.makedirs(vis_dir_recon, exist_ok=True)

                        t_val = int(t_sample[0].item() * 1000)
                        save_image(x_recon_img.clamp(0, 1), os.path.join(vis_dir_recon, f"noisy_step_{train_steps:07d}_t{t_val}.png"))
                        save_image(recon_img.clamp(0, 1), os.path.join(vis_dir_recon, f"recon_step_{train_steps:07d}_t{t_val}.png"))
                        save_image(img_gt[0:1].clamp(0, 1), os.path.join(vis_dir_recon, f"gt_step_{train_steps:07d}.png"))

                        logger.info(f"[step {train_steps}] Saved generation/reconstruction images.")

            # ---------------- Logging ----------------
            if train_steps % log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / max(1e-9, (end_time - start_time))

                avg_loss = torch.tensor(running_loss / max(1, log_steps), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size

                if rank == 0:
                    print(
                        f"(step={train_steps:07d}) Mode={args.prediction_mode} Decoder={args.decoder_type} "
                        f"Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f} "
                        f"GradNorm: {float(grad_norm):.4f}"
                    )
                    if writer is not None:
                        writer.add_scalar("train/loss", avg_loss, global_step=train_steps)
                        writer.add_scalar("train/steps_per_sec", steps_per_sec, global_step=train_steps)
                        writer.add_scalar("train/grad_norm", float(grad_norm), global_step=train_steps)

                running_loss = 0.0
                log_steps = 0
                start_time = time()

            # ---------------- Checkpointing ----------------
            if train_steps % ckpt_every == 0 and train_steps > 0:
                model_state = get_model_state_dict(model)
                if rank == 0:
                    checkpoint = {
                        "model": model_state,
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": schedl.state_dict(),
                        "train_steps": train_steps,
                        "args": vars(args),
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            if train_steps % 1000 == 0 and train_steps > 0:
                model_state = get_model_state_dict(model)
                if rank == 0:
                    snapshot = {
                        "model": model_state,
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": schedl.state_dict(),
                        "train_steps": train_steps,
                        "args": vars(args),
                    }
                    latest_path = f"{checkpoint_dir}/latest.pt"
                    torch.save(snapshot, latest_path)
                    logger.info(f"Updated quick resume checkpoint: {latest_path}")
                dist.barrier()

    model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the training dataset root.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256, help="Input image resolution.")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32", help="Compute precision for training.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training.")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed from the config.")
    parser.add_argument("--global-batch-size", type=int, default=None, help="Override training.global_batch_size from the config.")

    # VAE Args
    parser.add_argument(
        "--vae-ckpt",
        type=str,
        default="/share/project/huangxu/sae_hx/diff_decoder/tuned_enc_vae_26000.pth",
        help="VAE checkpoint path",
    )
    parser.add_argument("--vae-diffusion-steps", type=int, default=25, help="Sampling steps for VAE diffusion decoder.")

    # === Encoder type ===
    parser.add_argument(
        "--encoder-type",
        type=str,
        default="dinov3",
        choices=["dinov3", "siglip2"],
        help="Choose encoder type: 'dinov3' or 'siglip2'.",
    )
    parser.add_argument(
        "--dinov3-dir",
        type=str,
        default="/cpfs01/huangxu/models/dinov3",
        help="Path to DINOv3 model directory.",
    )
    parser.add_argument(
        "--siglip2-model-name",
        type=str,
        default="google/siglip2-base-patch16-256",
        help="SigLIP2 model name from HuggingFace.",
    )

    # === Decoder type ===
    parser.add_argument(
        "--decoder-type",
        type=str,
        default="cnn_decoder",
        choices=["diffusion_decoder", "cnn_decoder"],
        help="Choose VAE decoder type: 'diffusion_decoder' (sae_model.AutoencoderKL) or 'cnn_decoder' (cnn_decoder.AutoencoderKL).",
    )

    # Dataset choice
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["overfit_single_image", "single_class"],
        default="single_class",
        help="Choose dataset: 'overfit_single_image' or 'single_class'.",
    )
    # overfit options
    parser.add_argument("--overfit-image-path", type=str, default="/share/project/huangxu/SAE/dinov3_overfit.png")
    parser.add_argument("--overfit-length", type=int, default=1024 * 1024)
    parser.add_argument("--overfit-label", type=int, default=0)
    # single-class options
    parser.add_argument("--single-class-index", type=int, default=552)

    # Prediction Mode
    parser.add_argument(
        "--prediction-mode",
        type=str,
        choices=["x", "v"],
        default="x",
        help="Prediction mode: 'x' (predict x0) or 'v' (predict velocity v=eps-x0).",
    )

    # sampling/vis knobs
    parser.add_argument("--vis-every", type=int, default=50, help="Save visualization every N train steps.")
    parser.add_argument("--stage2-steps", type=int, default=50, help="Stage2 sampling steps for visualization/generation.")
    parser.add_argument("--force-log-every", type=int, default=None, help="Force log_every override (else use config).")

    # FSDP for memory optimization
    parser.add_argument("--fsdp-size", type=int, default=1, help="FSDP sharding size. 1=disabled (use DDP), >1=enable FSDP to shard optimizer states.")
    parser.add_argument("--ema-cpu", action="store_true", help="Keep EMA model on CPU to save GPU memory.")

    args = parser.parse_args()
    main(args)
