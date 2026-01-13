# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""

import os
import torch
from tqdm.std import trange
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging

import sys
sys.path.append(".")
import math
from torch.cuda.amp import autocast
from omegaconf import OmegaConf
# from models.rae.stage1 import RAE
from models.rae.stage2.models import Stage2ModelProtocol
# from models.rae.stage2.transport import create_transport, Sampler
from models.rae.utils.train_utils import parse_configs
from models.rae.utils.model_utils import instantiate_from_config
from models.rae.utils import wandb_utils
from models.rae.utils.optim_utils import build_optimizer, build_scheduler
from sae_model import AutoencoderKL
from dataset import ImageNetIdxDataset,OverfitSingleImageDataset
import gc
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


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


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
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

def normalize_sae(tensor):
    ema_shift_factor = 0.0019670347683131695
    ema_scale_factor = 0.247765451669693
    return (tensor - ema_shift_factor) / ema_scale_factor

def denormalize_sae(tensor):
    ema_shift_factor = 0.0019670347683131695
    ema_scale_factor = 0.247765451669693
    return tensor * ema_scale_factor + ema_shift_factor

#################################################################################
#                                  Training Loop                                #
#################################################################################

@torch.no_grad()
def reconstruct_from_latent_with_diffusion(
    vae: AutoencoderKL,
    latent_z: torch.Tensor,
    image_shape: torch.Size,
    diffusion_steps: int = 25,
) -> torch.Tensor:
    """
    Given latent_z from DINO encoder (or model x-pred in the same latent space),
    run the full diffusion decoder sampling loop to reconstruct images.

    Args:
        vae: AutoencoderKL model with diffusion decoder.
        latent_z: latent representation, [B, C_latent, H_latent, W_latent].
        image_shape: target image shape (B, 3, H, W).
        diffusion_steps: number of diffusion sampling steps.

    Returns:
        Reconstructed images in [-1, 1], shape = image_shape.
    """
    device = latent_z.device
    z = denormalize_sae(latent_z)

    # 如果有 post_quant_conv，则与训练时保持一致
    if getattr(vae, "post_quant_conv", None) is not None and vae.post_quant_conv is not None:
        z = vae.post_quant_conv(z)

    # 用 diffusion decoder 的 compressor 获得多尺度 context
    context = vae.diffusion_decoder.get_context(z)
    # 与 forward 一致：反转 context 顺序
    corrected_context = list(reversed(context[:]))

    diffusion = vae.diffusion_decoder.diffusion
    diffusion.set_sample_schedule(diffusion_steps, device)

    # 从纯高斯噪声开始采样
    init_noise = torch.randn(
        image_shape,
        device=device,
        dtype=latent_z.dtype,
    )

    recon = diffusion.p_sample_loop(
        vae=vae,
        shape=image_shape,
        context=corrected_context,
        clip_denoised=True,
        init_noise=init_noise,
        eta=0.0,  # DDIM deterministic sampling
    )

    return recon


def load_dinov3_vae(
    vae_checkpoint_path: str,
    device: torch.device,
) -> AutoencoderKL:
    """
    构建并加载 DINOv3 AutoencoderKL（带 diffusion decoder）。
    和你 simple_dino_vae_reconstruct 里的参数保持一致。
    """
    model_params = {
        "in_channels": 3,
        "out_channels": 3,
        "enc_block_out_channels": [128, 256, 384, 512, 768],
        "dec_block_out_channels": [1280, 1024, 512, 256, 128],
        "enc_layers_per_block": 2,
        "dec_layers_per_block": 3,
        "latent_channels": 1280,           # DINOv3 ViT-H/16plus output dim
        "use_quant_conv": False,
        "use_post_quant_conv": False,
        "spatial_downsample_factor": 16,   # 例如 256/16 = 16
        "variational": False,              # 训练 stage2 时一般用确定性 latent
        "noise_tau": 0.0,
        "denormalize_decoder_output": True,
        "running_mode": "dec",
        "random_masking_channel_ratio": 0.0,
        "target_latent_channels": None,
        "lpips_weight": 0.1,
    }

    vae = AutoencoderKL(**model_params).to(device)

    print(f"[load_dinov3_vae] Loading VAE checkpoint from: {vae_checkpoint_path}")
    checkpoint = torch.load(vae_checkpoint_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    print(f"[load_dinov3_vae] Loaded VAE. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    vae.eval()
    requires_grad(vae, False)
    return vae


def compute_train_loss(
    model,
    x_latent: torch.Tensor,
    model_kwargs: dict,
    time_shift: float,
):
    B = x_latent.size(0)
    device = x_latent.device

    noise = torch.randn_like(x_latent)

    # t ~ Uniform[0,1], 再做 time_shift 变换
    t = torch.rand(B, device=device)  # [B]
    t = (time_shift * t) / (1.0 + (time_shift - 1.0) * t)

    t_broadcast = t.view(B, 1, 1, 1)

    # Linear path: x_t = t * ε + (1 - t) * x_0
    x_t = t_broadcast * noise + (1.0 - t_broadcast) * x_latent
    # print("x_t.shape",x_t.shape,model_kwargs)
    model_output = model(x_t, t, **model_kwargs)

    # 不 reduce：逐元素 MSE
    loss = F.mse_loss(model_output, x_latent, reduction="none")  # same shape as x_latent

    # reweight_scale 乘上去（按 batch 维广播）
    reweight_scale = torch.clamp_max(1.0 / (t**2 + 1e-8), 20)  # [B]
    # print(reweight_scale)
    loss = loss * reweight_scale.view(B, 1, 1, 1)
    loss = loss.mean()

    return loss, model_output, noise, t



@torch.no_grad()
def sample_latent_linear_50_steps(
    model,                 # Stage2ModelProtocol or DDP-wrapped
    batch_size: int,
    latent_shape: tuple,   # (C, H, W) e.g. (1280,16,16)
    device: torch.device,
    y: torch.Tensor,       # class labels [B]
    time_shift: float,
    steps: int = 50,
    init_noise: torch.Tensor | None = None,
):
    """
    Sample latent z0 using the SAME training parameterization:

    Training:
        eps ~ N(0,I)
        t_lin ~ U(0,1)
        t = (shift * t_lin) / (1 + (shift-1)*t_lin)
        x_t = t * eps + (1 - t) * x0
        model(x_t, t, y) -> x0 (x-pred)

    Inference (deterministic):
        start x_{t=1} = eps
        iterate t: 1 -> 0:
            x0_hat = model(x_t, t, y)
            eps_hat = (x_t - (1-t)*x0_hat)/t
            x_{t_next} = t_next*eps_hat + (1-t_next)*x0_hat
    """
    # unwrap DDP if needed
    model_inner = model.module if hasattr(model, "module") else model
    model_inner.eval()

    C, H, W = latent_shape

    if init_noise is None:
        x = torch.randn((batch_size, C, H, W), device=device, dtype=torch.float32)
    else:
        x = init_noise.to(device=device, dtype=torch.float32)

    shift = float(time_shift)

    def flow_shift(t_lin: torch.Tensor) -> torch.Tensor:
        # SD3-style flow shift
        t = (shift * t_lin) / (1.0 + (shift - 1.0) * t_lin)
        # 和你训练里保持一致：避免 t=1 / t=0 的数值问题
        return t.clamp(0.0, 1.0 - 1e-6)

    # 线性 schedule in t_lin：从 1 -> 0
    # 注意：我们这里用确定性 schedule，不再随机采样 t
    for i in range(steps, 0, -1):
        t_lin = torch.full((batch_size,), i / steps, device=device, dtype=torch.float32)
        t_next_lin = torch.full((batch_size,), (i - 1) / steps, device=device, dtype=torch.float32)

        t = flow_shift(t_lin)           # [B]
        t_next = flow_shift(t_next_lin) # [B]

        # Stage2 forward 用的是 t:[B]，不是 [B,1]
        x0_hat = model_inner(x, t, y=y)  # x-pred: predict x0 in latent space

        # eps_hat from x_t = t*eps + (1-t)*x0
        t_scalar = t.view(batch_size, 1, 1, 1)
        eps_hat = (x - (1.0 - t_scalar) * x0_hat) / (t_scalar + 1e-8)

        # update to next time
        t_next_scalar = t_next.view(batch_size, 1, 1, 1)
        x = t_next_scalar * eps_hat + (1.0 - t_next_scalar) * x0_hat

    # 最后一步 t_next=0，x 就是 z0_hat
    return x

@torch.no_grad()
def stage2_sample_and_reconstruct(
    model,
    dinov3_vae,                 # AutoencoderKL (with diffusion decoder)
    y: torch.Tensor,            # [B]
    image_size: int = 256,
    latent_shape: tuple = (1280, 16, 16),
    stage2_steps: int = 50,
    vae_diffusion_steps: int = 25,
    time_shift: float = 1.0,
    device: torch.device | None = None,
):
    """
    端到端：
      1) Stage2 在 latent 空间采样 z0_hat（50 steps）
      2) 用 dinov3_vae 的 diffusion_decoder 做 diffusion_reconstruct 得到图像
    """
    if device is None:
        device = y.device

    B = y.shape[0]
    y = y.to(device)

    # 1) latent sampling
    z0_hat = sample_latent_linear_50_steps(
        model=model,
        batch_size=B,
        latent_shape=latent_shape,
        device=device,
        y=y,
        time_shift=time_shift,
        steps=stage2_steps,
    )  # [B,1280,16,16]

    # 2) diffusion reconstruct to image
    img = reconstruct_from_latent_with_diffusion(
        vae=dinov3_vae,
        latent_z=z0_hat,
        image_shape=torch.Size([B, 3, image_size, image_size]),
        diffusion_steps=vae_diffusion_steps,
    )  # [-1,1]

    return img, z0_hat


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    """Trains a new SiT model using config-driven hyperparameters + DINO VAE."""
    if not torch.cuda.is_available():
        raise RuntimeError("Training currently requires at least one GPU.")

    (
        rae_config,          # unused now
        model_config,
        transport_config,    # unused
        sampler_config,      # unused
        guidance_config,     # unused
        misc_config,
        training_config,
    ) = parse_configs(args.config)

    def to_dict(cfg_section):
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)

    misc = to_dict(misc_config)
    training_cfg = to_dict(training_config)

    num_classes = int(misc.get("num_classes", 1000))

    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    clip_grad = float(training_cfg.get("clip_grad", 1.0))
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    epochs = int(training_cfg.get("epochs", 1400))
    if args.global_batch_size is not None:
        global_batch_size = args.global_batch_size
    else:
        global_batch_size = int(training_cfg.get("global_batch_size", 1024))
    num_workers = int(training_cfg.get("num_workers", 4))
    log_every = int(training_cfg.get("log_every", 100))
    ckpt_every = int(training_cfg.get("ckpt_every", 5_000))
    default_seed = int(training_cfg.get("global_seed", 0))
    global_seed = args.global_seed if args.global_seed is not None else default_seed

    if grad_accum_steps < 1:
        raise ValueError("Gradient accumulation steps must be >= 1.")
    if args.image_size % 16 != 0:
        raise ValueError("Image size must be divisible by 16 for the VAE encoder (downsample_factor=16).")
    print("test")
    # ---------------- DDP init ----------------
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    print("ddp train!!")
    if global_batch_size % (world_size * grad_accum_steps) != 0:
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

    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    # 为了稳定性，默认不开 autocast，有需要你自己改成 bf16
    autocast_kwargs = dict(dtype=torch.float32, enabled=False)

    # ---------------- time_shift & latent size ----------------
    # DINO latent 尺寸固定 (C=1280, H=W=16)
    latent_size = (1280, (args.image_size // 16), (args.image_size // 16))
    # 你要求：time_shift = sqrt((16 * 16 * 1280) / 4096)
    shift_dim = (args.image_size // 16) * (args.image_size // 16) * 1280
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
            f"flowmatch-xpred{precision_suffix}-acc{grad_accum_steps}"
        )

        experiment_dir = os.path.join(args.results_dir, "experiment")
        os.makedirs(experiment_dir, exist_ok=True)
        checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        # === 新增：TensorBoard SummaryWriter ===
        tb_dir = os.path.join(experiment_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        # ==============================

        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        experiment_dir = None
        checkpoint_dir = None
        logger = create_logger(None)
        writer = None

    # ---------------- load DINO VAE ----------------
    dinov3_vae = load_dinov3_vae(args.vae_ckpt, device)
    print("load vae!!!")
    # ---------------- Stage2 model ----------------
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    # print("model",model)
    print("load diffusion model!!!")

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

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

    model = DDP(model, device_ids=[device_idx], gradient_as_bucket_view=False)

    opt, opt_msg = build_optimizer(model.parameters(), training_cfg)
    if opt_state is not None:
        opt.load_state_dict(opt_state)

    # ---------------- dataset / loader ----------------
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),                       # [0,1]
        transforms.Lambda(lambda t: t * 2.0 - 1.0),  # [-1,1]
    ])
    image_path="/share/project/huangxu/SAE/dinov3_overfit.png"
    dataset = OverfitSingleImageDataset(
        image_path=image_path,
        length=1024*1024,
        label=0,
        transform=transform,
    )
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
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(
        f"Gradient accumulation: steps={grad_accum_steps}, micro batch={micro_batch_size}, "
        f"per-GPU batch={micro_batch_size * grad_accum_steps}, global batch={global_batch_size}"
    )
    logger.info(f"Precision mode: {args.precision}")

    loader_batches = len(loader)
    if loader_batches % grad_accum_steps != 0:
        raise ValueError("Number of loader batches must be divisible by grad_accum_steps when drop_last=True.")
    steps_per_epoch = loader_batches // grad_accum_steps
    if steps_per_epoch <= 0:
        raise ValueError("Gradient accumulation configuration results in zero optimizer steps per epoch.")
    schedl, sched_msg = build_scheduler(opt, steps_per_epoch, training_cfg, sched_state)
    if rank == 0:
        logger.info(f"Training configured for {epochs} epochs, {steps_per_epoch} steps per epoch.")
        logger.info(opt_msg + "\n" + sched_msg)

    # 初始化 EMA
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    log_steps = 0
    running_loss = 0.0
    start_time = time()

    logger.info(f"Training for {epochs} epochs...")
    log_every = 10

    log_steps = 0
    running_loss = 0.0
    start_time = time()
    print("time_shift",time_shift)

    logger.info(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        opt.zero_grad()
        accum_counter = 0
        step_loss_accum = 0.0

        for x_img, y in loader:
            x_img = x_img.to(device)  # [-1,1]
            y = y.to(device)
            # print("x_img",x_img,"y",y)

            # 记录 GT 图像 [0,1] 用于可视化
            img_gt = (x_img + 1.0) / 2.0

            with torch.no_grad():
                x_latent,p = dinov3_vae.encode(x_img)
                x_latent = normalize_sae(x_latent)
                # x_latent = encoder_output.latent  # [B, 1280,16,16]
            # print("x_latent!!!")
            model_kwargs = dict(y=y)

            # with autocast(**autocast_kwargs):
            loss_tensor, pred_latent, noise, t_sample = compute_train_loss(
                model=model,
                x_latent=x_latent,
                model_kwargs=model_kwargs,
                time_shift=time_shift,
            )

            # NaN 检测
            is_nan_local = 0 if torch.isfinite(loss_tensor) else 1
            is_nan = torch.tensor(is_nan_local, device=device)
            dist.all_reduce(is_nan, op=dist.ReduceOp.SUM)

            if is_nan.item() > 0:
                if rank == 0:
                    logger.warning(f"[step {train_steps}] NaN detected across ranks! Auto-resuming from latest.pt ...")
                dist.barrier()

                try:
                    torch.cuda.synchronize()
                    del loss_tensor, x_latent
                    gc.collect()
                    torch.cuda.empty_cache()
                    dist.barrier()

                    latest_path = f"{checkpoint_dir}/latest.pt"
                    checkpoint = torch.load(latest_path, map_location="cpu")

                    if "ema" in checkpoint:
                        ema.load_state_dict(checkpoint["ema"])
                    if "model" in checkpoint:
                        model.module.load_state_dict(checkpoint["model"])
                    if "opt" in checkpoint and checkpoint["opt"] is not None:
                        opt.load_state_dict(checkpoint["opt"])
                    if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
                        schedl.load_state_dict(checkpoint["scheduler"])
                    train_steps = int(checkpoint.get("train_steps", 0))

                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()

                    dist.barrier()
                    if rank == 0:
                        logger.warning(f"All ranks resumed from {latest_path} (train_steps={train_steps})")

                    dist.barrier()
                    continue

                except Exception as e:
                    if rank == 0:
                        logger.error(f"Failed to resume checkpoint: {e}")
                    dist.barrier()
                    cleanup()
                    exit(1)

            step_loss_accum += loss_tensor.item()
            loss_tensor /= grad_accum_steps
            loss_tensor.backward()
            accum_counter += 1

            if accum_counter < grad_accum_steps:
                continue

            if clip_grad > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
            schedl.step()
            update_ema(ema, model.module, decay=ema_decay)
            opt.zero_grad()

            running_loss += step_loss_accum / grad_accum_steps
            log_steps += 1
            train_steps += 1
            accum_counter = 0
            step_loss_accum = 0.0

            # 每 1000 step 重建一张图（只在 rank==0）
            show_time = 1000
            DEBUG=False
            if rank == 0 and ((train_steps % show_time == 0) or DEBUG):
                with torch.no_grad():
                    y_sample = y[:1]  # [1]
                    img_gen, z_gen = stage2_sample_and_reconstruct(
                        model=model,                       # DDP 包装也行
                        dinov3_vae=dinov3_vae,
                        y=y_sample,
                        image_size=args.image_size,
                        latent_shape=(1280,  args.image_size // 16,  args.image_size // 16),
                        stage2_steps=50,
                        vae_diffusion_steps=args.vae_diffusion_steps,
                        time_shift=time_shift,             # 你算出来的那个
                        device=device,
                    )

                    img_gen_01 = (img_gen + 1.0) / 2.0
                    vis_dir = os.path.join(experiment_dir, "samples")
                    os.makedirs(vis_dir, exist_ok=True)
                    save_image(img_gen_01, os.path.join(vis_dir, f"gen_step_{train_steps:07d}.png"))
                    # 用当前 step 的 x-pred latent 做重建
                    latent_sample = pred_latent[0:1]  # [1, 1280,16,16]
                    recon = reconstruct_from_latent_with_diffusion(
                        vae=dinov3_vae,
                        latent_z=latent_sample,
                        image_shape=torch.Size([1, 3, args.image_size, args.image_size]),
                        diffusion_steps=args.vae_diffusion_steps,
                    )  # [-1,1]
                    x_latent_sample=x_latent[0:1] * (1 - t_sample[0]) + noise[0:1] * t_sample[0]
                    x_recon = reconstruct_from_latent_with_diffusion(
                        vae=dinov3_vae,
                        latent_z=x_latent_sample,
                        image_shape=torch.Size([1, 3, args.image_size, args.image_size]),
                        diffusion_steps=args.vae_diffusion_steps,
                    )  # [-1,1]

                    recon_img = (recon + 1.0) / 2.0
                    vis_dir = os.path.join(experiment_dir, "recon")
                    if DEBUG==True:
                        vis_dir = os.path.join(experiment_dir, "recon_debug")
                    os.makedirs(vis_dir, exist_ok=True)
                    x_recon_img = (x_recon + 1.0) / 2.0
                    save_image(
                        x_recon_img,
                        os.path.join(vis_dir,f"noisy_step_{train_steps:07d}_{int(t_sample[0] * 1000)}.png")
                    )
                    save_image(
                        recon_img,
                        os.path.join(vis_dir, f"recon_step_{train_steps:07d}_{int(t_sample[0] * 1000)}.png"),
                    )
                    # 也可以顺便存一张 GT
                    save_image(
                        img_gt[0:1],
                        os.path.join(vis_dir, f"gt_step_{train_steps:07d}_{int(t_sample[0] * 1000)}.png"),
                    )

                logger.info(f"[step {train_steps}] Saved reconstruction and GT image.")

            # === 每 10 step 打印 & TensorBoard 记录 loss ===
            if train_steps % log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size

                if rank == 0:
                    # 控制台打印
                    print(
                        f"(step={train_steps:07d}) "
                        f"Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f} "
                        f"Train Grad Norm {grad_norm:.4f}"
                    )
                    # TensorBoard 记录
                    if writer is not None:
                        writer.add_scalar("train/loss", avg_loss, global_step=train_steps)
                        writer.add_scalar("train/steps_per_sec", steps_per_sec, global_step=train_steps)
                        writer.add_scalar("train/grad_norm", grad_norm, global_step=train_steps)

                    # 若还想继续用 wandb，这里保留
                    # if args.wandb:
                    #     wandb_utils.log(
                    #         {"train loss": avg_loss, "train steps/sec": steps_per_sec},
                    #         step=train_steps,
                    #     )

                running_loss = 0.0
                log_steps = 0
                start_time = time()
            # === end log block ===

            if train_steps % ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": schedl.state_dict(),
                        "train_steps": train_steps,
                        "config_path": args.config,
                        "training_cfg": training_cfg,
                        "cli_overrides": {
                            "data_path": args.data_path,
                            "results_dir": args.results_dir,
                            "image_size": args.image_size,
                            "precision": args.precision,
                            "global_seed": global_seed,
                        },
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            if train_steps % 1000 == 0 and train_steps > 0:
                if rank == 0:
                    snapshot = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": schedl.state_dict(),
                        "train_steps": train_steps,
                        "config_path": args.config,
                        "training_cfg": training_cfg,
                        "cli_overrides": {
                            "data_path": args.data_path,
                            "results_dir": args.results_dir,
                            "image_size": args.image_size,
                            "precision": args.precision,
                            "global_seed": global_seed,
                        },
                    }
                    latest_path = f"{checkpoint_dir}/latest.pt"
                    torch.save(snapshot, latest_path)
                    logger.info(f"Updated quick resume checkpoint: {latest_path}")
                dist.barrier()

        # if accum_counter != 0:
        #     raise RuntimeError("Gradient accumulation counter not zero at epoch end.")

        if train_steps % ckpt_every == 0 and train_steps > 0:
            if rank == 0:
                checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": schedl.state_dict(),
                        "train_steps": train_steps,
                        "config_path": args.config,
                        "training_cfg": training_cfg,
                        "cli_overrides": {
                            "data_path": args.data_path,
                            "results_dir": args.results_dir,
                            "image_size": args.image_size,
                            "precision": args.precision,
                            "global_seed": global_seed,
                        },
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()

            if train_steps % 1000 == 0 and train_steps > 0:
                if rank == 0:
                    snapshot = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": schedl.state_dict(),
                        "train_steps": train_steps,
                        "config_path": args.config,
                        "training_cfg": training_cfg,
                        "cli_overrides": {
                            "data_path": args.data_path,
                            "results_dir": args.results_dir,
                            "image_size": args.image_size,
                            "precision": args.precision,
                            "global_seed": global_seed,
                        },
                    }
                    latest_path = f"{checkpoint_dir}/latest.pt"
                    torch.save(snapshot, latest_path)
                    logger.info(f"Updated quick resume checkpoint: {latest_path}")
                dist.barrier()

        # if accum_counter != 0:
        #     raise RuntimeError("Gradient accumulation counter not zero at epoch end.")

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

    # 新增：VAE 相关参数
    parser.add_argument("--vae-ckpt", type=str, default="/share/project/huangxu/sae_hx/diff_decoder/tuned_enc_vae_26000.pth", help="")
    parser.add_argument("--vae-diffusion-steps", type=int, default=25, help="Sampling steps for VAE diffusion decoder.")

    args = parser.parse_args()
    main(args)