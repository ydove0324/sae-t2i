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
import argparse
import logging
import shutil  # Added for file operations

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
from models.rae.utils.vae_utils import (
    load_dinov3_vae,
    normalize_sae,
    denormalize_sae,
    reconstruct_from_latent_with_diffusion,
)
from dataset import (ImageNetIdxDataset,
    DatasetSpec,
    ImageNetLatentCacheDataset,
    collate_keep_dict
)
import gc
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import random
from wds_dataset import make_latent_loader

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
            
            # 1. Sample Latent
            z0_hat = sample_latent_linear_50_steps(
                model=model,
                batch_size=curr_batch,
                latent_shape=(1280, 16, 16),
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
                latent_z=z0_hat,
                image_shape=torch.Size([curr_batch, 3, args.image_size, args.image_size]),
                diffusion_steps=args.vae_diffusion_steps,
                decoder_type=decoder_type,
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
        decoder_type=decoder_type,
    )  # [-1,1]

    return img, z0_hat


def atomic_torch_save(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def materialize_latents_for_batch(
    batch: dict,
    dinov3_vae,
    device: torch.device,
    cache_dtype: torch.dtype = torch.float16,
):
    """
    输入：collate_keep_dict 输出的 batch dict
    输出：
      x_latent: [B,C,H,W] on GPU float32
      y: [B] on GPU long
      extra: dict (用于你后面可视化的 img_gt 等)
    逻辑：
      - cache hit：CPU latent -> GPU
      - miss：把 img 组成子 batch，在 GPU 上 encode->normalize，然后写回 cache_path
    """
    has_latent = batch["has_latent"]           # [B] bool (CPU)
    labels = batch["label"]                   # [B] long (CPU)
    latents_list = batch["latent"]            # list[Tensor or None] (CPU)
    imgs_list = batch["img"]                  # list[Tensor or None] (CPU)
    cache_paths = batch["cache_path"]         # list[str]
    paths = batch["path"]                     # list[str]

    B = labels.numel()

    # 先收集：命中 latent、miss 的 img
    x_latent_cpu = [None] * B
    miss_indices = []
    miss_imgs = []
    for i in range(B):
        if bool(has_latent[i].item()):
            x_latent_cpu[i] = latents_list[i]  # Tensor[C,h,w] CPU
        else:
            miss_indices.append(i)
            miss_imgs.append(imgs_list[i])     # Tensor[3,H,W] CPU

    # 对 miss 执行 encode 并写回缓存
    if len(miss_indices) > 0:
        x_img_miss = torch.stack(miss_imgs, dim=0).to(device, non_blocking=True)  # [m,3,H,W] GPU
        latent_miss, _p = dinov3_vae.encode(x_img_miss)
        latent_miss = normalize_sae(latent_miss)  # [m,C,h,w]
        latent_cpu = latent_miss.detach().to("cpu")
        if cache_dtype is not None:
            latent_cpu = latent_cpu.to(cache_dtype)

        for j, i in enumerate(miss_indices):
            obj = {
                "latent": latent_cpu[j].contiguous(),
                "label": int(labels[i].item()),
                "path": paths[i],
            }
            atomic_torch_save(obj, cache_paths[i])
            x_latent_cpu[i] = latent_cpu[j]

    # 拼回完整 batch latent -> GPU float32
    x_latent = torch.stack(x_latent_cpu, dim=0).to(device, non_blocking=True).float()
    y = labels.to(device, non_blocking=True)

    # 同时准备 img_gt（只有 miss 才有 img；hit 没有 img，这里返回 None 或者仅对 miss 可视化）
    # 如果你希望可视化时总有 gt，可以把 dataset 改成即使命中也返回 img（会慢一些 I/O）
    extra = {
        "img_cpu_list": imgs_list,  # list[Tensor or None], 用于你想仅在 miss 情况可视化 gt
        "paths": paths,
    }
    return x_latent, y, extra


def main(args):
    """Trains a new SiT model using config-driven hyperparameters + DINO VAE + latent cache."""
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

    # ---------------- time_shift & latent size ----------------
    latent_size = (1280, 16, 16)
    shift_dim = 16 * 16 * 1280
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

    # ---------------- load DINO VAE (keep decoder_type) ----------------
    dinov3_vae = load_dinov3_vae(args.vae_ckpt, device, decoder_type=args.decoder_type)
    if rank == 0:
        logger.info(f"Loaded VAE: {args.vae_ckpt} decoder_type={args.decoder_type}")

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

    model = DDP(model, device_ids=[device_idx], gradient_as_bucket_view=False)

    opt, opt_msg = build_optimizer(model.parameters(), training_cfg)
    if opt_state is not None:
        opt.load_state_dict(opt_state)


    spec = DatasetSpec(
        root=args.data_path,
        index_synset_path=args.index_synset_path,
        image_size=args.image_size,
        vae_ckpt=args.vae_ckpt,           # tag 用这个
        cache_dir=args.cache_dir,
        single_class_index=args.single_class,
        overfit_image=args.overfit_image,
        overfit_length=args.overfit_length,
    )
    dataset = ImageNetLatentCacheDataset(spec)

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
        collate_fn=collate_keep_dict,
        persistent_workers=True if num_workers > 0 else False,
    )


    logger.info(f"Dataset contains {len(dataset):,} samples ({args.data_path})")
    logger.info(
        f"Gradient accumulation: steps={grad_accum_steps}, micro batch={micro_batch_size}, "
        f"per-GPU batch={micro_batch_size * grad_accum_steps}, global batch={global_batch_size}"
    )
    logger.info(f"Precision mode: {args.precision}")
    logger.info(f"Latent cache: cache_dir={args.cache_dir}, cache_dtype={args.cache_dtype}")

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

    # cache dtype
    if args.cache_dtype == "fp16":
        cache_dtype = torch.float16
    elif args.cache_dtype == "bf16":
        cache_dtype = torch.bfloat16
    elif args.cache_dtype == "fp32":
        cache_dtype = torch.float32
    else:
        raise ValueError(f"Unknown cache_dtype: {args.cache_dtype}")
    loader = make_latent_loader(args.wds_urls, micro_batch_size, num_workers) 
    log_steps = 0
    running_loss = 0.0
    start_time = time()

    logger.info(f"Training for {epochs} epochs...")
    # 你原来强制 log_every=10，我保留 args/config 的值（如果你想强制就自己改）
    # log_every = 10

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            print(f"Beginning epoch {epoch}...")

        opt.zero_grad()
        accum_counter = 0
        step_loss_accum = 0.0

        # for batch in loader:

        for x_latent, y in loader:
            x_latent = x_latent.to(device, non_blocking=True)  # [B,1280,16,16] fp32
            y = y.to(device, non_blocking=True)
            # ==========================
            # 关键：命中缓存直接用 latent；miss 才 encode 并写回 pt
            # ==========================
            # CFG label dropout（按你原逻辑）
            NULL_CLASS = num_classes
            if args.cfg_prob > 0.0:
                drop_mask = (torch.rand_like(y.float()) < args.cfg_prob)
                y_train = y.clone()
                y_train[drop_mask] = NULL_CLASS
            else:
                y_train = y

            model_kwargs = dict(y=y_train)

            loss_tensor, pred_latent, noise, t_sample = compute_train_loss(
                model=model,
                x_latent=x_latent,
                model_kwargs=model_kwargs,
                time_shift=time_shift,
            )

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

            # backward + accum
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
            update_ema(ema, model.module, decay=ema_decay)
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
            show_time = 1000
            DEBUG = False
            if rank == 0 and ((train_steps % show_time == 0) or DEBUG):
                with torch.no_grad():
                    # 1) 生成一张（EMA）
                    y_sample = y[:1]
                    img_gen, z_gen = stage2_sample_and_reconstruct(
                        model=ema,
                        dinov3_vae=dinov3_vae,
                        y=y_sample,
                        image_size=args.image_size,
                        latent_shape=(1280, 16, 16),
                        stage2_steps=50,
                        vae_diffusion_steps=args.vae_diffusion_steps,
                        time_shift=time_shift,
                        device=device,
                        decoder_type=args.decoder_type,
                    )
                    img_gen_01 = (img_gen + 1.0) / 2.0
                    vis_dir = os.path.join(experiment_dir, "samples")
                    os.makedirs(vis_dir, exist_ok=True)
                    save_image(img_gen_01, os.path.join(vis_dir, f"gen_step_{train_steps:07d}.png"))

                    # 2) 用当前 step 的 x-pred latent 做重建
                    latent_sample = pred_latent[0:1]
                    recon = reconstruct_from_latent_with_diffusion(
                        vae=dinov3_vae,
                        latent_z=latent_sample,
                        image_shape=torch.Size([1, 3, args.image_size, args.image_size]),
                        diffusion_steps=args.vae_diffusion_steps,
                        decoder_type=args.decoder_type,
                    )

                    # 3) noisy latent 可视化（按你原代码）
                    # x_latent 是 z0，t_sample/noise 是 compute_train_loss 里生成的
                    x_latent_sample = x_latent[0:1] * (1 - t_sample[0]) + noise[0:1] * t_sample[0]
                    x_recon = reconstruct_from_latent_with_diffusion(
                        vae=dinov3_vae,
                        latent_z=x_latent_sample,
                        image_shape=torch.Size([1, 3, args.image_size, args.image_size]),
                        diffusion_steps=args.vae_diffusion_steps,
                        decoder_type=args.decoder_type,
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
                    use_cfg=False,
                    cfg_scale=args.cfg_scale,
                    cfg_interval_low=args.cfg_interval_low,
                    cfg_interval_high=args.cfg_interval_high,
                    null_class=num_classes,
                )

                # 2) CFG
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
                            "decoder_type": args.decoder_type,
                            "cache_dir": args.cache_dir,
                            "cache_dtype": args.cache_dtype,
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
                            "decoder_type": args.decoder_type,
                            "cache_dir": args.cache_dir,
                            "cache_dtype": args.cache_dtype,
                        },
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
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument(
        "--index-synset-path",
        type=str,
        default="/share/project/datasets/ImageNet/train/index_synset.yaml",
    )
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--global-seed", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--precision", type=str, choices=["bf16", "fp32"], default="fp32")
    parser.add_argument("--wds-urls",type=str,default="/share/project/huangxu/SAE/kl500_vae_latent/*.tar")
    # (可选) wandb：你 main 里虽然注释掉了，但建议保留参数防止以后打开时报错
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")

    # VAE + cache
    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True, help="Directory to store latent pt cache")
    parser.add_argument("--cache-dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp32")
    parser.add_argument("--decoder-type", type=str, choices=["cnn_decoder", "diffusion_decoder"], default="cnn_decoder")

    # VAE diffusion decoder sampling steps (用于 reconstruct / FID decode)
    parser.add_argument("--vae-diffusion-steps", type=int, default=50, help="Sampling steps for VAE diffusion decoder.")

    # dataset modes
    parser.add_argument("--single-class", type=int, default=None)
    parser.add_argument("--overfit-image", type=str, default=None)
    parser.add_argument("--overfit-length", type=int, default=1024)

    # CFG (训练 label dropout)
    parser.add_argument("--cfg-prob", type=float, default=0.1)

    # CFG (采样时 guidance 参数：用于 FID sampling)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--cfg-interval-low", type=float, default=0.1)
    parser.add_argument("--cfg-interval-high", type=float, default=1.0)

    # FID eval
    parser.add_argument("--fid-every", type=int, default=10000, help="Run FID evaluation every N steps.")
    parser.add_argument("--fid-samples-per-class", type=int, default=50, help="Number of samples per class for FID.")
    parser.add_argument(
        "--fid-ref-path",
        type=str,
        default="/share/project/huangxu/SAE/VIRTUAL_imagenet256_labeled.npz",
        help="Path to reference .npz stats for FID.",
    )
    parser.add_argument("--fid-batch-size", type=int, default=32, help="Batch size for FID generation.")

    args = parser.parse_args()
    main(args)
