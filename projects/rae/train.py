import os
import torch
# 开启 TF32 加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from time import time
import argparse
import logging
import shutil
import sys
import math
import gc
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from glob import glob

# 添加当前目录
sys.path.append(".")

# === 动态导入模块 ===
# 假设项目结构中存在 dataset.py, models/ 等
from models.rae.utils.train_utils import parse_configs
from models.rae.utils.model_utils import instantiate_from_config
from models.rae.utils.optim_utils import build_optimizer, build_scheduler
from dataset import ImageNetIdxDataset

# === FID ===
try:
    from pytorch_fid import fid_score
    HAS_FID = True
except ImportError:
    HAS_FID = False
    print("Warning: pytorch-fid not found. FID will be skipped.")

# -----------------------------------------------------------------------------
#                               Helper Functions
# -----------------------------------------------------------------------------

def update_ema(ema_model, model, decay=0.9999):
    """标准的 EMA 更新"""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
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

# -----------------------------------------------------------------------------
#                               Model Loading
# -----------------------------------------------------------------------------

def load_vae_model(args, device):
    """
    根据 args.decoder_type 动态加载 CNN Decoder 或 SAE (Diffusion Decoder)
    """
    if args.decoder_type == "cnn_decoder":
        try:
            from cnn_decoder import AutoencoderKL
            if dist.get_rank() == 0: print(">> [Model] Loading CNN Decoder VAE...")
        except ImportError:
            print("Error: cnn_decoder not found.")
            sys.exit(1)
    else:
        try:
            from sae_model import AutoencoderKL
            if dist.get_rank() == 0: print(">> [Model] Loading Diffusion Decoder VAE (SAE)...")
        except ImportError:
            print("Error: sae_model not found.")
            sys.exit(1)

    # 这里的参数根据你的模型配置进行调整
    model_params = {
        "in_channels": 3,
        "out_channels": 3,
        "enc_block_out_channels": [128, 256, 384, 512, 768],
        "dec_block_out_channels": [1280, 1024, 512, 256, 128],
        "enc_layers_per_block": 2,
        "dec_layers_per_block": 3,
        "latent_channels": 1280,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
        "spatial_downsample_factor": 16,
        "variational": False,
        "noise_tau": 0.0,
        "denormalize_decoder_output": True,
        "running_mode": "dec", # SAE specific
        "lpips_weight": 0.1,
    }

    vae = AutoencoderKL(**model_params).to(device)
    
    if dist.get_rank() == 0:
        print(f">> [Model] Loading VAE weights from: {args.vae_ckpt}")
        
    checkpoint = torch.load(args.vae_ckpt, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    vae.load_state_dict(state_dict, strict=False)
    vae.eval()
    requires_grad(vae, False)
    return vae

# -----------------------------------------------------------------------------
#                               Training Loss
# -----------------------------------------------------------------------------

def compute_train_loss(model, x_latent, model_kwargs, time_shift):
    B = x_latent.size(0)
    device = x_latent.device
    noise = torch.randn_like(x_latent)

    # Time sampling (Flow Matching specific)
    t = torch.rand(B, device=device)
    # Time shift trick for better performance at high res
    t = (time_shift * t) / (1.0 + (time_shift - 1.0) * t)
    t_broadcast = t.view(B, 1, 1, 1)

    # Linear Interpolation: x_t = t * x_1 + (1-t) * x_0 ?
    # Standard Flow Matching: x_t = (1 - t) * x_0 + t * x_1 (here noise is x_1)
    # Be careful with definition: usually t=0 is data, t=1 is noise. 
    # The code below implies: x_t = t * noise + (1-t) * data. So t=0 is Data.
    x_t = t_broadcast * noise + (1.0 - t_broadcast) * x_latent

    # Predict Vector Field (velocity)
    model_output = model(x_t, t, **model_kwargs)

    # Target velocity: v_t = x_1 - x_0 = noise - data
    target = noise - x_latent

    loss = F.mse_loss(model_output, target, reduction="none")
    
    # Optional re-weighting
    # reweight_scale = torch.clamp_max(1.0 / (t**2 + 1e-8), 10) # Optional
    # loss = loss * reweight_scale.view(B, 1, 1, 1)
    
    return loss.mean()

# -----------------------------------------------------------------------------
#                               Sampling Logic (with CFG)
# -----------------------------------------------------------------------------

@torch.no_grad()
def sample_latent_cfg(model, batch_size, latent_shape, device, y, cfg_scale, time_shift, steps=50, null_class=1000):
    """
    CFG Sampling Routine for Flow Matching
    """
    model.eval()
    C, H, W = latent_shape
    
    # 1. Start from Noise (t=1)
    x = torch.randn((batch_size, C, H, W), device=device, dtype=torch.float32)
    
    # Null label vector
    y_null = torch.full_like(y, null_class)
    
    shift = float(time_shift)
    def flow_shift(t_lin):
        t = (shift * t_lin) / (1.0 + (shift - 1.0) * t_lin)
        return t.clamp(0.0, 1.0 - 1e-6) # Avoid singularities

    # Euler Solver from t=1 to t=0
    for i in range(steps, 0, -1):
        # Current time and next time
        t_lin = torch.full((batch_size,), i / steps, device=device)
        t_next_lin = torch.full((batch_size,), (i - 1) / steps, device=device)
        t = flow_shift(t_lin)
        t_next = flow_shift(t_next_lin)
        
        if cfg_scale == 1.0:
            # Unconditional / Standard generation
            v_pred = model(x, t, y=y)
        else:
            # Classifier-Free Guidance
            # Batch concatenation for efficiency
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t, t], dim=0)
            y_in = torch.cat([y, y_null], dim=0) # [Conditional, Unconditional]
            
            v_out = model(x_in, t_in, y=y_in)
            v_cond, v_uncond = v_out.chunk(2, dim=0)
            
            # CFG Formula: v = v_uncond + s * (v_cond - v_uncond)
            v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)

        # Euler Update: x_{t-1} = x_t + (t_{next} - t) * v_pred
        dt = t_next - t
        dt = dt.view(batch_size, 1, 1, 1)
        x = x + dt * v_pred
        
    return x # This is Latent (z0)

@torch.no_grad()
def decode_latents(vae, z, args):
    """Unified decoder interface"""
    z = denormalize_sae(z)
    
    if args.decoder_type == "cnn_decoder":
        # Standard VAE Decode
        out = vae.decode(z)
        if hasattr(out, 'sample'): out = out.sample
        return out
    else:
        # SAE (Diffusion) Decode
        # 1. Post Quant (if needed)
        if getattr(vae, "post_quant_conv", None) is not None:
            z = vae.post_quant_conv(z)
        
        # 2. Setup Diffusion
        # Get context (inverted)
        ctx = list(reversed(vae.diffusion_decoder.get_context(z)))
        
        # Set scheduler
        vae.diffusion_decoder.diffusion.set_sample_schedule(args.vae_diffusion_steps, z.device)
        
        # Sample
        noise = torch.randn((z.shape[0], 3, args.image_size, args.image_size), device=z.device)
        recon = vae.diffusion_decoder.diffusion.p_sample_loop(
            vae=vae,
            shape=noise.shape,
            context=ctx,
            clip_denoised=True,
            init_noise=noise,
            eta=0.0
        )
        return recon

# -----------------------------------------------------------------------------
#                               Evaluation Routine
# -----------------------------------------------------------------------------

def run_evaluation(args, model, vae, device, step, logger, writer, time_shift):
    """
    运行评估：生成图片 (CFG=1.0 和 4.0)，计算 FID。
    """
    if not HAS_FID: return
    if dist.get_rank() == 0: logger.info(f"==> Starting Evaluation at Step {step}")

    # 需要生成的总数
    total_samples = args.eval_num_samples # e.g., 5000 or 10000
    batch_size = args.eval_batch_size
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # 每个 GPU 需要生成的数量
    samples_per_gpu = int(math.ceil(total_samples / world_size))
    
    # 随机选择类别 (固定种子以保证每次评估的类别分布相似，如果需要)
    # 或者简单起见，随机生成 0-999
    eval_labels = torch.randint(0, 1000, (samples_per_gpu,), device=device)

    cfg_modes = [1.0, 4.0] # 两种模式

    for cfg in cfg_modes:
        if rank == 0: logger.info(f"    Generating for CFG = {cfg} ...")
        
        # 临时保存路径
        temp_dir = os.path.join(args.results_dir, "eval_temp", f"step_{step}_cfg_{cfg}")
        # 为了避免文件系统冲突，每个 Rank 写到自己的子文件夹，或者文件名带 rank
        os.makedirs(temp_dir, exist_ok=True)
        
        # 1. 生成循环
        model.eval()
        num_batches = int(math.ceil(samples_per_gpu / batch_size))
        
        for i in range(num_batches):
            b_start = i * batch_size
            b_end = min((i + 1) * batch_size, samples_per_gpu)
            curr_batch_size = b_end - b_start
            
            y_batch = eval_labels[b_start:b_end]
            
            # A. Sample Latent
            z_gen = sample_latent_cfg(
                model, curr_batch_size, (1280, 16, 16), device, 
                y_batch, cfg, time_shift, steps=args.eval_steps, null_class=args.null_class
            )
            
            # B. Decode
            img_gen = decode_latents(vae, z_gen, args) # [-1, 1]
            img_gen = (img_gen + 1.0) / 2.0
            img_gen = torch.clamp(img_gen, 0, 1)
            
            # C. Save
            for idx in range(curr_batch_size):
                global_idx = rank * samples_per_gpu + b_start + idx
                save_image(img_gen[idx], os.path.join(temp_dir, f"{global_idx}.png"))
        
        # 等待所有 GPU 生成完毕
        dist.barrier()
        
        # 2. 计算 FID (仅 Rank 0)
        if rank == 0:
            logger.info(f"    Calculating FID for CFG = {cfg} ...")
            try:
                # 调用 pytorch-fid
                fid_value = fid_score.calculate_fid_given_paths(
                    paths=[temp_dir, args.fid_ref_path],
                    batch_size=50,
                    device=device,
                    dims=2048,
                    num_workers=8
                )
                logger.info(f"    >> Step {step} | CFG {cfg} | FID: {fid_value:.4f}")
                
                # Log to TensorBoard
                if writer:
                    writer.add_scalar(f"eval/FID_cfg_{cfg}", fid_value, step)
                
                # Log to txt
                with open(os.path.join(args.results_dir, "fid_history.txt"), "a") as f:
                    f.write(f"Step: {step}, CFG: {cfg}, FID: {fid_value}\n")

            except Exception as e:
                logger.error(f"FID calculation failed: {e}")
            
            # 3. 清理图片 (节省空间)
            shutil.rmtree(temp_dir)
            
        dist.barrier()

    # 恢复训练模式
    model.train()
    torch.cuda.empty_cache() # 清理显存
    if rank == 0: logger.info("==> Evaluation Finished.")


# -----------------------------------------------------------------------------
#                               Main Training
# -----------------------------------------------------------------------------

def main(args):
    # DDP 初始化
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)
    
    seed = (args.global_seed + rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Logging
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        logger = create_logger(args.results_dir)
        writer = SummaryWriter(log_dir=os.path.join(args.results_dir, "tensorboard"))
    else:
        logger = create_logger(None)
        writer = None

    # Configs
    _, model_config, _, _, _, misc_config, training_config = parse_configs(args.config)
    training_cfg = OmegaConf.to_container(training_config, resolve=True)
    
    # 1. Load VAE
    vae = load_vae_model(args, device)
    
    # 2. Load DiT/SiT Model
    model = instantiate_from_config(model_config).to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    
    model = DDP(model, device_ids=[device_idx], gradient_as_bucket_view=False)
    
    # Optimizer
    opt, _ = build_optimizer(model.parameters(), training_cfg)
    
    # 3. Dataset
    transform = transforms.Compose([
        transforms.Lambda(lambda p: center_crop_arr(p, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),
    ])
    
    dataset = ImageNetIdxDataset(
        root=args.data_path, 
        index_synset_path="/share/project/datasets/ImageNet/train/index_synset.yaml", 
        transform=transform
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
    
    # Batch Calculation
    local_batch_size = args.global_batch_size // world_size
    loader = DataLoader(
        dataset, batch_size=local_batch_size, sampler=sampler, 
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    # Scheduler
    grad_accum_steps = args.grad_accum_steps
    steps_per_epoch = len(loader) // grad_accum_steps
    schedl, _ = build_scheduler(opt, steps_per_epoch, training_cfg, None)
    
    # Resume Logic
    train_steps = 0
    if args.ckpt:
        if rank == 0: logger.info(f"Resuming from {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.module.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        train_steps = ckpt["train_steps"]

    # Constants
    shift_dim = 16 * 16 * 1280
    time_shift = math.sqrt(shift_dim / 4096)
    NULL_CLASS = args.null_class # e.g., 1000

    if rank == 0:
        logger.info(f"Start Training. Total Epochs: {args.epochs}")
        logger.info(f"CFG Enabled: {args.use_cfg} | Probability: {args.cfg_prob}")
        logger.info(f"Decoder Type: {args.decoder_type}")

    model.train()
    
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        opt.zero_grad()
        
        for step, (x_img, y) in enumerate(loader):
            x_img = x_img.to(device)
            y = y.to(device) # [B]
            
            # ==========================================
            #           CFG Training Logic
            # ==========================================
            # Correct Element-wise masking
            if args.use_cfg:
                # 生成一个与 Batch 长度相同的随机 mask
                # 概率 < cfg_prob (0.1) 的样本，其 label 设为 NULL_CLASS
                mask = torch.rand(y.shape, device=device) < args.cfg_prob
                y_train = torch.where(mask, torch.tensor(NULL_CLASS, device=device), y)
            else:
                y_train = y
            # ==========================================

            # Encode Latents
            with torch.no_grad():
                if args.decoder_type == "cnn_decoder":
                    x_latent = vae.encode(x_img).latent_dist.sample()
                else:
                    x_latent, _ = vae.encode(x_img)
                x_latent = normalize_sae(x_latent)

            # Forward & Loss
            loss = compute_train_loss(model, x_latent, dict(y=y_train), time_shift)
            
            loss.backward()
            
            # Optimization
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                schedl.step()
                update_ema(ema, model.module)
                opt.zero_grad()
                train_steps += 1
                
                # Logging
                if train_steps % 100 == 0 and rank == 0:
                    lr = opt.param_groups[0]["lr"]
                    logger.info(f"Step {train_steps}: Loss {loss.item():.4f} | LR {lr:.6f}")
                    if writer:
                        writer.add_scalar("train/loss", loss.item(), train_steps)

                # Checkpointing
                if train_steps % 5000 == 0 and rank == 0:
                    save_path = os.path.join(args.results_dir, "checkpoints", f"{train_steps:07d}.pt")
                    torch.save({
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "train_steps": train_steps
                    }, save_path)
                
                dist.barrier()

                # ==========================================
                #        Evaluation (Every 10k Steps)
                # ==========================================
                if train_steps > 0 and train_steps % 10000 == 0:
                    # 使用 EMA 模型进行评估
                    run_evaluation(
                        args, ema, vae, device, train_steps, logger, writer, time_shift
                    )

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 基础配置
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data-path", type=str, required=True, help="ImageNet root")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--ckpt", type=str, default=None)
    
    # 训练超参
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)

    # VAE / Decoder 设置
    parser.add_argument("--vae-ckpt", type=str, required=True)
    parser.add_argument("--decoder-type", type=str, default="cnn_decoder", 
                        choices=["cnn_decoder", "diffusion_decoder"])
    parser.add_argument("--vae-diffusion-steps", type=int, default=25, help="Only for SAE decoder")

    # CFG Training
    parser.add_argument("--use-cfg", action="store_true", help="Enable CFG training")
    parser.add_argument("--cfg-prob", type=float, default=0.1, help="Probability to drop class label")
    parser.add_argument("--null-class", type=int, default=1000, help="Index for null class")

    # Evaluation
    parser.add_argument("--fid-ref-path", type=str, default="/share/project/huangxu/SAE/VIRTUAL_imagenet256_labeled.npz", help="Path to ImageNet .npz for FID")
    parser.add_argument("--eval-num-samples", type=int, default=50000, help="Total images to generate for FID")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-steps", type=int, default=50, help="Sampling steps during eval")

    args = parser.parse_args()
    
    # 安全检查
    if args.fid_ref_path == "" or not os.path.exists(args.fid_ref_path):
        print("Warning: --fid-ref-path not set or invalid. Evaluation will crash if triggered.")

    main(args)