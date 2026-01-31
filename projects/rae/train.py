# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""

import os
import torch
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
from collections import defaultdict
import argparse


#################################################################################
#                              Timing Profiler                                  #
#################################################################################

class TrainingProfiler:
    """
    训练过程时间分析器，用于追踪各主要操作的耗时。
    """
    def __init__(self, enabled=True, sync_cuda=True):
        self.enabled = enabled
        self.sync_cuda = sync_cuda  # 是否在计时前同步CUDA
        self.timers = defaultdict(list)
        self.current_timers = {}
        
    def start(self, name: str):
        """开始计时某个操作"""
        if not self.enabled:
            return
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.current_timers[name] = time()
    
    def stop(self, name: str):
        """停止计时某个操作"""
        if not self.enabled:
            return
        if name not in self.current_timers:
            return
        if self.sync_cuda:
            torch.cuda.synchronize()
        elapsed = time() - self.current_timers[name]
        self.timers[name].append(elapsed)
        del self.current_timers[name]
        return elapsed
    
    def reset(self):
        """重置所有计时器"""
        self.timers.clear()
        self.current_timers.clear()
    
    def get_stats(self, last_n: int = None) -> dict:
        """获取统计信息"""
        stats = {}
        for name, times in self.timers.items():
            if last_n is not None:
                times = times[-last_n:]
            if len(times) > 0:
                stats[name] = {
                    "count": len(times),
                    "total": sum(times),
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                }
        return stats
    
    def print_summary(self, last_n: int = None, logger=None):
        """打印时间分析摘要"""
        stats = self.get_stats(last_n)
        if not stats:
            return
        
        # 按总时间排序
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True)
        total_time = sum(s["total"] for _, s in sorted_stats)
        
        lines = [
            "\n" + "=" * 70,
            f"{'操作名称':<30} {'次数':>8} {'总时间(s)':>12} {'平均(ms)':>12} {'占比':>8}",
            "-" * 70,
        ]
        for name, s in sorted_stats:
            pct = (s["total"] / total_time * 100) if total_time > 0 else 0
            lines.append(
                f"{name:<30} {s['count']:>8} {s['total']:>12.3f} "
                f"{s['mean']*1000:>12.2f} {pct:>7.1f}%"
            )
        lines.append("=" * 70)
        
        msg = "\n".join(lines)
        if logger:
            logger.info(msg)
        else:
            print(msg)
import logging
import shutil  # Added for file operations

import sys
sys.path.append(".")
import math
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from omegaconf import OmegaConf
# from models.rae.stage1 import RAE
from models.rae.stage2.models import Stage2ModelProtocol
# from models.rae.stage2.transport import create_transport, Sampler
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
from torchvision.datasets import ImageFolder
import gc
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import random

# === FID Imports ===
try:
    from pytorch_fid import fid_score
    HAS_FID = True
except ImportError:
    HAS_FID = False
    print("Warning: 'pytorch-fid' not found. FID calculation will be skipped.")

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


#################################################################################
#                             FID Helper Functions                              #
#################################################################################

def create_flat_temp_dir(source_root, temp_dir):
    """
    Flatten the directory structure for pytorch-fid.
    """
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    count = 0
    # Walk through source_root
    for root, dirs, files in os.walk(source_root):
        if os.path.abspath(root) == os.path.abspath(temp_dir):
            continue

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.abspath(os.path.join(root, file))
                # Avoid name collisions
                parent_name = os.path.basename(root)
                new_name = f"{parent_name}_{file}"
                dst_path = os.path.join(temp_dir, new_name)
                
                try:
                    os.symlink(src_path, dst_path)
                    count += 1
                except OSError:
                    shutil.copy(src_path, dst_path)
                    count += 1
    return count

@torch.no_grad()
def run_fid_evaluation(
    model, 
    vae, 
    args, 
    logger, 
    device, 
    step, 
    time_shift, 
    rank, 
    world_size,
    writer=None,
    decoder_type: str = "diffusion_decoder",
    encoder_type: str = "dinov3",
    latent_shape: tuple = (1280, 16, 16),
    use_cfg: bool = False,
    cfg_scale: float = 3.0,
    cfg_interval_low: float = 0.1,
    cfg_interval_high: float = 1.0,
    null_class: int = 1000,
):
    """
    Runs generation and FID calculation.
    """
    if not HAS_FID:
        if rank == 0:
            logger.warning("FID skipped because pytorch-fid is not installed.")
        return

    if args.fid_ref_path is None or not os.path.exists(args.fid_ref_path):
        if rank == 0:
            logger.warning(f"FID skipped because reference path is invalid: {args.fid_ref_path}")
        return

    # Directories
    mode_suffix = "cfg" if use_cfg else "nocfg"
    eval_dir = os.path.join(args.results_dir, "eval_samples", f"step_{step:07d}_{mode_suffix}")
    flat_dir = os.path.join(args.results_dir, "eval_samples", f"step_{step:07d}_{mode_suffix}_flat")
    
    if rank == 0:
        logger.info(f"========== Starting FID Evaluation (Step {step}) ==========")
        os.makedirs(eval_dir, exist_ok=True)
    
    dist.barrier()

    # Distribute classes
    all_classes = list(range(1000))
    my_classes = all_classes[rank::world_size]
    
    # Generation Loop
    model.eval() # Ensure eval mode
    
    total_needed = args.fid_samples_per_class
    
    # Use a progress log on rank 0
    if rank == 0:
        logger.info(f"Generating {total_needed} samples per class for 1000 classes...")

    for cls_id in my_classes:
        class_dir = os.path.join(eval_dir, str(cls_id))
        os.makedirs(class_dir, exist_ok=True)
        
        generated_count = 0
        while generated_count < total_needed:
            curr_batch = min(args.fid_batch_size, total_needed - generated_count)
            
            y = torch.full((curr_batch,), cls_id, device=device, dtype=torch.long)
            
            # FID evaluation always uses FP32 for precision
            with torch.autocast(device_type='cuda', enabled=False):
                # 1. Sample Latent
                z0_hat = sample_latent_linear_50_steps(
                    model=model,
                    batch_size=curr_batch,
                    latent_shape=latent_shape,
                    device=device,
                    y=y,
                    time_shift=time_shift,
                    steps=50,  # Using 50 steps as standard
                    use_cfg=use_cfg,
                    cfg_scale=cfg_scale,
                    cfg_interval_low=cfg_interval_low,
                    cfg_interval_high=cfg_interval_high,
                    null_class=null_class,
                )
                
                # 2. Decode using Diffusion Decoder (Default)
                imgs = reconstruct_from_latent_with_diffusion(
                    vae=vae,
                    latent_z=z0_hat.float(),  # Ensure FP32
                    image_shape=torch.Size([curr_batch, 3, args.image_size, args.image_size]),
                    diffusion_steps=args.vae_diffusion_steps,
                    decoder_type=decoder_type,
                    encoder_type=encoder_type,
                )
            
            # 3. Save
            imgs = (imgs + 1.0) / 2.0
            imgs = imgs.clamp(0, 1)
            
            for i in range(curr_batch):
                img_idx = generated_count + i
                save_image(imgs[i], os.path.join(class_dir, f"{img_idx}.png"))
                
            generated_count += curr_batch

    dist.barrier()
    
    # FID Calculation (Rank 0 only)
    if rank == 0:
        logger.info("Generation finished. Flattening directory for FID...")
        try:
            num_imgs = create_flat_temp_dir(eval_dir, flat_dir)
            if num_imgs < 100:
                logger.warning(f"Too few images ({num_imgs}) for valid FID.")
            else:
                logger.info(f"Calculating FID using {num_imgs} images...")
                fid_value = fid_score.calculate_fid_given_paths(
                    paths=[flat_dir, args.fid_ref_path],
                    batch_size=50,
                    device=device,
                    dims=2048,
                    num_workers=8
                )
                logger.info(f"Step {step} FID: {fid_value:.4f}")
                
                if writer is not None:
                    writer.add_scalar("eval/FID", fid_value, global_step=step)
                    
                # Write to text file
                with open(os.path.join(args.results_dir, "fid_scores.txt"), "a") as f:
                    f.write(f"Step {step} ({mode_suffix}): {fid_value}\n")

        except Exception as e:
            logger.error(f"FID Calculation failed: {e}")
        finally:
            # Clean up to save space
                if os.path.exists(flat_dir):
                    shutil.rmtree(flat_dir)
            # Optional: Clean up the structured eval_dir too if you don't want to keep images
            # shutil.rmtree(eval_dir)

    dist.barrier()
    logger.info(f"========== End FID Evaluation (Step {step}) ==========")


#################################################################################
#                                  Training Loop                                #
#################################################################################

def sample_timesteps(
    batch_size: int,
    device: torch.device,
    noise_schedule: str = "uniform",
    time_shift: float = 1.0,
    log_norm_mean: float = 0.0,
    log_norm_std: float = 1.0,
):
    """
    Sample timesteps for flow matching training.
    
    Args:
        batch_size: Number of samples
        device: Device to use
        noise_schedule: Type of noise schedule
            - "uniform": Uniform sampling with SD3-style time shift
            - "log_norm": Log-normal sampling (SD3 style)
        time_shift: Time shift factor for uniform schedule
        log_norm_mean: Mean for log-normal distribution
        log_norm_std: Std for log-normal distribution
    
    Returns:
        t: Timesteps in [0, 1)
    """
    if noise_schedule == "uniform":
        # Uniform sampling with SD3-style time shift
        t = torch.rand(batch_size, device=device)
    elif noise_schedule == "log_norm":
        # Log-normal sampling (SD3 style)
        # Sample u ~ Normal(mean, std), then t = sigmoid(u)
        u = torch.randn(batch_size, device=device) * log_norm_std + log_norm_mean
        t = torch.sigmoid(u)
    else:
        raise ValueError(f"Unknown noise_schedule: {noise_schedule}")
    t = (time_shift * t) / (1.0 + (time_shift - 1.0) * t)
    # Clamp to avoid numerical issues at t=0 or t=1
    t = t.clamp(1e-5, 1.0 - 1e-5)
    return t


def compute_train_loss(
    model,
    x_latent: torch.Tensor,
    model_kwargs: dict,
    time_shift: float = 1.0,
    noise_schedule: str = "uniform",
    log_norm_mean: float = 0.0,
    log_norm_std: float = 1.0,
):
    B = x_latent.size(0)
    device = x_latent.device

    noise = torch.randn_like(x_latent)

    # Sample timesteps based on noise schedule
    t = sample_timesteps(
        batch_size=B,
        device=device,
        noise_schedule=noise_schedule,
        time_shift=time_shift,
        log_norm_mean=log_norm_mean,
        log_norm_std=log_norm_std,
    )

    t_broadcast = t.view(B, 1, 1, 1)

    # Linear path: x_t = t * ε + (1 - t) * x_0
    x_t = t_broadcast * noise + (1.0 - t_broadcast) * x_latent

    model_output = model(x_t, t, **model_kwargs)

    # 不 reduce：逐元素 MSE
    loss = F.mse_loss(model_output, x_latent, reduction="none")  # same shape as x_latent

    # reweight_scale 乘上去（按 batch 维广播）
    reweight_scale = torch.clamp_max(1.0 / (t**2 + 1e-8), 10)  # [B]
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
    use_cfg: bool = False,
    cfg_scale: float = 3.0,
    cfg_interval_low: float = 0.1,
    cfg_interval_high: float = 1.0,
    null_class: int = 1000,
):
    """
    Sample latent z0 using the SAME training parameterization.
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
    for i in range(steps, 0, -1):
        t_lin = torch.full((batch_size,), i / steps, device=device, dtype=torch.float32)
        t_next_lin = torch.full((batch_size,), (i - 1) / steps, device=device, dtype=torch.float32)

        t = flow_shift(t_lin)           # [B]
        t_next = flow_shift(t_next_lin) # [B]

        # Stage2 forward 用的是 t:[B]，不是 [B,1]
        if not use_cfg:
            # 原来的条件采样
            x0_hat = model_inner(x, t, y=y)  # x-pred: predict x0 in latent space
        else:
            # Classifier-free guidance: 在指定 t 区间内启用 CFG
            y_cond = y
            y_uncond = torch.full_like(y, null_class)

            x0_hat_cond = model_inner(x, t, y=y_cond)
            x0_hat_uncond = model_inner(x, t, y=y_uncond)

            # t 是常数向量，所以取第一个元素判断区间
            t_scalar_val = t[0].item()
            if cfg_interval_low <= t_scalar_val <= cfg_interval_high:
                x0_hat = x0_hat_uncond + cfg_scale * (x0_hat_cond - x0_hat_uncond)
            else:
                x0_hat = x0_hat_uncond

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
    decoder_type: str = "diffusion_decoder",
    encoder_type: str = "dinov3",
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
    )  # [B,C,16,16]

    # 2) diffusion reconstruct to image
    img = reconstruct_from_latent_with_diffusion(
        vae=dinov3_vae,
        latent_z=z0_hat,
        image_shape=torch.Size([B, 3, image_size, image_size]),
        diffusion_steps=vae_diffusion_steps,
        decoder_type=decoder_type,
        encoder_type=encoder_type,
    )  # [-1,1]

    return img, z0_hat


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
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


def main(args):
    """Trains a new SiT model using config-driven hyperparameters + DINO VAE."""
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

    # ---------------- DDP init ----------------
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    if global_batch_size % (world_size * grad_accum_steps) != 0:
        raise ValueError("Global batch size must be divisible by world_size * grad_accum_steps.")
    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)

    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # ---------------- latent size from config ----------------
    # Read latent_size from config misc section
    latent_size_cfg = misc.get("latent_size", None)
    if latent_size_cfg is not None:
        # Config format: [C, H, W]
        latent_size = tuple(latent_size_cfg)
        latent_channels = latent_size[0]
    else:
        # Fallback: determine from encoder_type
        if args.encoder_type == "dinov3":
            latent_channels = 1280
        elif args.encoder_type == "siglip2":
            latent_channels = 768  # SigLIP2-base hidden size
        else:
            raise ValueError(f"Unknown encoder_type: {args.encoder_type}")
        latent_size = (latent_channels, 16, 16)
    
    if rank == 0:
        print(f"Using latent_size from config: {latent_size}")
    
    # ---------------- time_shift ----------------
    shift_dim = latent_size[0] * latent_size[1] * latent_size[2]
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_shift = math.sqrt(shift_dim / shift_base)
    
    # Noise schedule config
    noise_schedule = args.noise_schedule
    log_norm_mean = args.log_norm_mean
    log_norm_std = args.log_norm_std
    if rank == 0:
        print(f"Noise schedule: {noise_schedule}, time_shift: {time_shift:.4f}")

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

        tb_dir = os.path.join(experiment_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)

        # if args.wandb:
        #     entity = os.environ["ENTITY"]
        #     project = os.environ["PROJECT"]
        #     wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        experiment_dir = None
        checkpoint_dir = None
        logger = create_logger(None)
        writer = None

    # ---------------- load VAE (DINOv3 or SigLIP2) ----------------
    # Build model params for VAE
    if args.encoder_type == "dinov3":
        default_dec_block_out_channels = (1280, 1024, 512, 256, 128)
    else:  # siglip2
        default_dec_block_out_channels = (768, 512, 256, 128, 64)
    
    # LoRA rank: 0 means no LoRA layers at all
    effective_lora_rank = 0 if args.no_lora else args.lora_rank
    effective_lora_alpha = 0 if args.no_lora else args.lora_alpha
    
    vae_model_params = {
        "encoder_type": args.encoder_type,
        "image_size": args.image_size,
        "patch_size": 16,
        "out_channels": 3,
        "latent_channels": latent_channels,
        "target_latent_channels": None,
        "spatial_downsample_factor": 16,
        "lora_rank": effective_lora_rank,
        "lora_alpha": effective_lora_alpha,
        "dec_block_out_channels": default_dec_block_out_channels,
        "dec_layers_per_block": 3,
        "decoder_dropout": 0.0,
        "gradient_checkpointing": False,
        "denormalize_decoder_output": args.denormalize_decoder_output,
        "skip_to_moments": args.skip_to_moments,
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
        skip_to_moments=args.skip_to_moments,
    )
    
    # Get normalization function based on encoder_type
    normalize_fn = get_normalize_fn(args.encoder_type)
    
    if rank == 0:
        logger.info(f"Loaded VAE: {args.vae_ckpt} encoder_type={args.encoder_type} decoder_type={args.decoder_type}")

    # ---------------- Stage2 model ----------------
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
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
        if rank == 0:
            logger.info(f"Resumed from {args.ckpt}, train_steps={train_steps}")

    model_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {model_param_count/1e6:.2f}M")

    # torch.compile acceleration (before DDP wrapping)
    if args.compile:
        if rank == 0:
            logger.info(f"Compiling model with torch.compile (mode={args.compile_mode})...")
        model = torch.compile(model, mode=args.compile_mode)
        if rank == 0:
            logger.info("Model compiled successfully.")

    model = DDP(model, device_ids=[device_idx], gradient_as_bucket_view=False)

    opt, opt_msg = build_optimizer(model.parameters(), training_cfg)
    if opt_state is not None:
        opt.load_state_dict(opt_state)


    # ---------------- dataset / loader ----------------
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),                       # [0,1]
        transforms.Lambda(lambda t: t * 2.0 - 1.0),  # [-1,1]
    ])

    dataset = ImageFolder(args.data_path, transform=transform)

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

    logger.info(f"Dataset contains {len(dataset):,} samples ({args.data_path})")
    logger.info(
        f"Gradient accumulation: steps={grad_accum_steps}, micro batch={micro_batch_size}, "
        f"per-GPU batch={micro_batch_size * grad_accum_steps}, global batch={global_batch_size}"
    )
    logger.info(f"Precision mode: {args.precision}")

    loader_batches = len(loader)
    steps_per_epoch = max(1, loader_batches // grad_accum_steps)

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
    
    # 初始化 AMP (混合精度训练)
    use_amp = (args.precision == "bf16")
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))  # bf16不需要scaler
    if rank == 0:
        logger.info(f"AMP enabled: {use_amp}, dtype: {amp_dtype}")
    
    # 初始化时间分析器
    profiler = TrainingProfiler(enabled=args.profile, sync_cuda=True)
    profile_print_every = 100  # 每500步打印一次时间分析

    logger.info(f"Training for {epochs} epochs...")
    # 你原来强制 log_every=10，我保留 args/config 的值（如果你想强制就自己改）
    # log_every = 10

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        epoch_start_time = time()
        if rank == 0:
            print(f"Beginning epoch {epoch}...")

        opt.zero_grad()
        accum_counter = 0
        step_loss_accum = 0.0

        for x_img, y in loader:
            profiler.start("data_to_gpu")
            x_img = x_img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            profiler.stop("data_to_gpu")

            # 实时编码图片到 latent
            profiler.start("vae_encode")
            with torch.no_grad(), autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                x_latent, _p = dinov3_vae.encode(x_img)
                x_latent = normalize_fn(x_latent)
            profiler.stop("vae_encode")

            # CFG label dropout
            NULL_CLASS = num_classes
            if args.cfg_prob > 0.0:
                drop_mask = (torch.rand_like(y.float()) < args.cfg_prob)
                y_train = y.clone()
                y_train[drop_mask] = NULL_CLASS
            else:
                y_train = y

            model_kwargs = dict(y=y_train)

            profiler.start("forward_loss")
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                loss_tensor, pred_latent, noise, t_sample = compute_train_loss(
                    model=model,
                    x_latent=x_latent,
                    model_kwargs=model_kwargs,
                    time_shift=time_shift,
                    noise_schedule=noise_schedule,
                    log_norm_mean=log_norm_mean,
                    log_norm_std=log_norm_std,
                )
            profiler.stop("forward_loss")

            # ==========================
            # NaN 检测 + 自动从 latest.pt 恢复（保留你原逻辑）
            # ==========================
            is_nan_local = 0 if torch.isfinite(loss_tensor) else 1
            is_nan = torch.tensor(is_nan_local, device=device)
            dist.all_reduce(is_nan, op=dist.ReduceOp.SUM)

            if is_nan.item() > 0:
                if rank == 0:
                    logger.warning(f"[step {train_steps}] NaN detected across ranks! Auto-resuming from latest.pt ...")
                dist.barrier()

                try:
                    torch.cuda.synchronize()
                    del loss_tensor
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

                    # Reset training state after resume
                    opt.zero_grad(set_to_none=True)
                    accum_counter = 0
                    step_loss_accum = 0.0
                    model.train()
                    ema.eval()

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

            # backward + accum
            step_loss_accum += loss_tensor.item()
            loss_tensor = loss_tensor / grad_accum_steps
            
            profiler.start("backward")
            if scaler.is_enabled():
                scaler.scale(loss_tensor).backward()
            else:
                loss_tensor.backward()
            profiler.stop("backward")
            accum_counter += 1

            if accum_counter < grad_accum_steps:
                continue

            profiler.start("optimizer_step")
            if scaler.is_enabled():
                scaler.unscale_(opt)
            
            if clip_grad > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            else:
                grad_norm = torch.tensor(0.0, device=device)

            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            schedl.step()
            profiler.stop("optimizer_step")
            
            profiler.start("ema_update")
            update_ema(ema, model.module, decay=ema_decay)
            profiler.stop("ema_update")
            
            opt.zero_grad()

            running_loss += step_loss_accum / grad_accum_steps
            log_steps += 1
            train_steps += 1
            accum_counter = 0
            step_loss_accum = 0.0

            # ==========================
            # 可视化重建（保留你原 eval：gen + recon + noisy + gt）
            # 注意：缓存命中时 dataset 不一定返回 img_gt（img 是 None）
            # 这里我做了一个兼容：如果本 batch 恰好有 img（miss）就存 gt，否则跳过 gt 存图。
            # ==========================
            show_time = 100
            DEBUG = False
            if rank == 0 and ((train_steps % show_time == 0) or DEBUG):
                # Eval visualization always uses FP32 for precision
                with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
                    # 1) 生成一张（EMA）
                    y_sample = y[:1]
                    img_gen, z_gen = stage2_sample_and_reconstruct(
                        model=ema,
                        dinov3_vae=dinov3_vae,
                        y=y_sample,
                        image_size=args.image_size,
                        latent_shape=latent_size,
                        stage2_steps=50,
                        vae_diffusion_steps=args.vae_diffusion_steps,
                        time_shift=time_shift,
                        device=device,
                        decoder_type=args.decoder_type,
                        encoder_type=args.encoder_type,
                    )
                    img_gen_01 = (img_gen + 1.0) / 2.0
                    vis_dir = os.path.join(experiment_dir, "samples")
                    os.makedirs(vis_dir, exist_ok=True)
                    save_image(img_gen_01, os.path.join(vis_dir, f"gen_step_{train_steps:07d}.png"))

                    # 2) 用当前 step 的 x-pred latent 做重建
                    latent_sample = pred_latent[0:1].float()  # Ensure FP32
                    recon = reconstruct_from_latent_with_diffusion(
                        vae=dinov3_vae,
                        latent_z=latent_sample,
                        image_shape=torch.Size([1, 3, args.image_size, args.image_size]),
                        diffusion_steps=args.vae_diffusion_steps,
                        decoder_type=args.decoder_type,
                        encoder_type=args.encoder_type,
                    )

                    # 3) noisy latent 可视化（按你原代码）
                    # x_latent 是 z0，t_sample/noise 是 compute_train_loss 里生成的
                    x_latent_sample = (x_latent[0:1] * (1 - t_sample[0]) + noise[0:1] * t_sample[0]).float()  # Ensure FP32
                    x_recon = reconstruct_from_latent_with_diffusion(
                        vae=dinov3_vae,
                        latent_z=x_latent_sample,
                        image_shape=torch.Size([1, 3, args.image_size, args.image_size]),
                        diffusion_steps=args.vae_diffusion_steps,
                        decoder_type=args.decoder_type,
                        encoder_type=args.encoder_type,
                    )

                    recon_img = (recon + 1.0) / 2.0
                    x_recon_img = (x_recon + 1.0) / 2.0

                    vis_dir = os.path.join(experiment_dir, "recon_debug" if DEBUG else "recon")
                    os.makedirs(vis_dir, exist_ok=True)

                    save_image(
                        x_recon_img,
                        os.path.join(vis_dir, f"noisy_step_{train_steps:07d}_{int(t_sample[0] * 1000)}.png"),
                    )
                    save_image(
                        recon_img,
                        os.path.join(vis_dir, f"recon_step_{train_steps:07d}_{int(t_sample[0] * 1000)}.png"),
                    )

                    # 4) GT：只有当 dataset miss 才会给 img；命中缓存时 img 是 None
                    # img_cpu0 = extra["img_cpu_list"][0]  # Tensor[3,H,W] or None
                    # if img_cpu0 is not None:
                    #     img_gt = (img_cpu0 + 1.0) / 2.0
                    #     save_image(
                    #         img_gt,
                    #         os.path.join(vis_dir, f"gt_step_{train_steps:07d}_{int(t_sample[0] * 1000)}.png"),
                    #     )

                logger.info(f"[step {train_steps}] Saved reconstruction/gen images (gt saved if available).")

            # ==========================
            # FID Evaluation（保留你原逻辑：no-CFG & CFG，decoder_type 保留）
            # ==========================
            if train_steps % args.fid_every == 0 and train_steps > 0:
                if rank == 0:
                    logger.info(f"Step {train_steps}: Starting FID evaluation (no-CFG and CFG)...")

                torch.cuda.empty_cache()

                # 1) no CFG
                run_fid_evaluation(
                    model=ema,
                    vae=dinov3_vae,
                    args=args,
                    logger=logger,
                    device=device,
                    step=train_steps,
                    time_shift=time_shift,
                    rank=rank,
                    world_size=world_size,
                    writer=writer,
                    decoder_type=args.decoder_type,
                    encoder_type=args.encoder_type,
                    latent_shape=latent_size,
                    use_cfg=False,
                    cfg_scale=args.cfg_scale,
                    cfg_interval_low=args.cfg_interval_low,
                    cfg_interval_high=args.cfg_interval_high,
                    null_class=num_classes,
                )

                # 2) CFG
                if train_steps % (args.fid_every * 4) != 0:
                    continue
                run_fid_evaluation(
                    model=ema,
                    vae=dinov3_vae,
                    args=args,
                    logger=logger,
                    device=device,
                    step=train_steps,
                    time_shift=time_shift,
                    rank=rank,
                    world_size=world_size,
                    writer=writer,
                    decoder_type=args.decoder_type,
                    encoder_type=args.encoder_type,
                    latent_shape=latent_size,
                    use_cfg=True,
                    cfg_scale=args.cfg_scale,
                    cfg_interval_low=args.cfg_interval_low,
                    cfg_interval_high=args.cfg_interval_high,
                    null_class=num_classes,
                )

                model.train()
                torch.cuda.empty_cache()

            # ==========================
            # log block（保留）
            # ==========================
            if train_steps % log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / max(1, log_steps), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size

                if rank == 0:
                    print(
                        f"(step={train_steps:07d}) "
                        f"Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f} "
                        f"Train Grad Norm {float(grad_norm):.4f}"
                    )
                    if writer is not None:
                        writer.add_scalar("train/loss", avg_loss, global_step=train_steps)
                        writer.add_scalar("train/steps_per_sec", steps_per_sec, global_step=train_steps)
                        writer.add_scalar("train/grad_norm", float(grad_norm), global_step=train_steps)

                running_loss = 0.0
                log_steps = 0
                start_time = time()
            
            # ==========================
            # 周期性打印时间分析
            # ==========================
            if args.profile and train_steps % profile_print_every == 0 and train_steps > 0 and rank == 0:
                profiler.print_summary(last_n=profile_print_every, logger=logger)
                profiler.reset()

            # ==========================
            # checkpoint 保存（保留 + 修正你原来重复块问题：这里只留一次）
            # ==========================
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
                            "vae_ckpt": args.vae_ckpt,
                            "encoder_type": args.encoder_type,
                            "decoder_type": args.decoder_type,
                        },
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            # ==========================
            # latest 快照（保留）
            # ==========================
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
                            "vae_ckpt": args.vae_ckpt,
                            "encoder_type": args.encoder_type,
                            "decoder_type": args.decoder_type,
                        },
                    }
                    latest_path = f"{checkpoint_dir}/latest.pt"
                    torch.save(snapshot, latest_path)
                    logger.info(f"Updated quick resume checkpoint: {latest_path}")
                dist.barrier()

        # Epoch结束，打印epoch时间
        if rank == 0:
            epoch_time = time() - epoch_start_time
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s ({epoch_time/60:.2f}min)")

    model.eval()
    logger.info("Done!")
    cleanup()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--global-seed", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--precision", type=str, choices=["bf16", "fp32"], default="fp32")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--profile", action="store_true", help="Enable training profiler to analyze time spent in each operation.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for model acceleration.")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"], help="torch.compile mode.")

    # VAE
    parser.add_argument("--vae-ckpt", type=str, required=True)
    
    # Encoder type
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
    
    # LoRA
    parser.add_argument("--no-lora", action="store_true", help="Do not use LoRA in VAE encoder (for loading non-LoRA checkpoints).")
    parser.add_argument("--lora-rank", type=int, default=256, help="LoRA rank (ignored if --no-lora is set).")
    parser.add_argument("--lora-alpha", type=int, default=256, help="LoRA alpha (ignored if --no-lora is set).")
    parser.add_argument("--skip-to-moments", action="store_true", help="Skip loading to_moments layer (for old checkpoints without it).")
    
    # Decoder type
    parser.add_argument("--decoder-type", type=str, choices=["cnn_decoder", "diffusion_decoder"], default="cnn_decoder")

    # VAE diffusion decoder sampling steps (用于 reconstruct / FID decode)
    parser.add_argument("--vae-diffusion-steps", type=int, default=50, help="Sampling steps for VAE diffusion decoder.")

    # Noise schedule for training
    parser.add_argument(
        "--noise-schedule",
        type=str,
        default="uniform",
        choices=["uniform", "log_norm"],
        help="Noise schedule for flow matching training: 'uniform' (with time_shift) or 'log_norm' (SD3 style).",
    )
    parser.add_argument(
        "--log-norm-mean",
        type=float,
        default=0.0,
        help="Mean for log-normal noise schedule (SD3 style).",
    )
    parser.add_argument(
        "--log-norm-std",
        type=float,
        default=1.0,
        help="Std for log-normal noise schedule (SD3 style).",
    )

    # CFG (训练 label dropout)
    parser.add_argument("--cfg-prob", type=float, default=0.1)

    # CFG (采样时 guidance 参数：用于 FID sampling)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--cfg-interval-low", type=float, default=0.1)
    parser.add_argument("--cfg-interval-high", type=float, default=1.0)

    # VAE denormalize
    parser.add_argument("--denormalize-decoder-output", action="store_true", help="Denormalize decoder output in VAE.")

    # FID eval
    parser.add_argument("--fid-every", type=int, default=10000, help="Run FID evaluation every N steps.")
    parser.add_argument("--fid-samples-per-class", type=int, default=50, help="Number of samples per class for FID.")
    parser.add_argument(
        "--fid-ref-path",
        type=str,
        default="VIRTUAL_imagenet256_labeled.npz",
        help="Path to reference .npz stats for FID.",
    )
    parser.add_argument("--fid-batch-size", type=int, default=32, help="Batch size for FID generation.")

    args = parser.parse_args()
    main(args)
