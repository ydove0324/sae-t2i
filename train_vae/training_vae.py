# training_vae.py
import os
import sys
sys.path.append(".")
import argparse
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

# local import
from cnn_decoder import AutoencoderKL

# DinoDisc from local dinodisc.py
from dinodisc import DinoDisc, DiffAug, hinge_d_loss, vanilla_g_loss

# === 导入统一工具模块 ===
from models.rae.utils.ddp_utils import (
    setup_ddp,
    cleanup_ddp,
    create_logger,
    requires_grad,
    unwrap_model,
    get_model_state_dict,
    update_ema,
)
from models.rae.utils.image_utils import center_crop_arr
from models.rae.utils.metrics_utils import LPIPSLoss, calculate_psnr, is_fid_available, is_lpips_available
from models.rae.utils.argparse_utils import get_encoder_config

# === deps ===
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not found. TensorBoard logging will be skipped.")

# FID 可用性检查
HAS_FID = is_fid_available()
if not HAS_FID:
    print("Warning: pytorch-fid not found. rFID calculation will be skipped.")
else:
    from pytorch_fid import fid_score

# LPIPS 可用性检查
if not is_lpips_available():
    print("Warning: lpips not installed. LPIPS loss will not be available.")
    sys.exit(1)


# ==========================
# Learning Rate Scheduler
# ==========================

import math

def get_cosine_schedule_with_warmup(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,  # min_lr / max_lr
):
    """
    Create a cosine annealing scheduler with linear warmup.
    
    Args:
        optimizer: The optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Ratio of min_lr to max_lr (default 0.1 means min_lr = 0.1 * max_lr)
    """
    # 获取优化器的参数组数量，确保为每个参数组创建对应的 lambda 函数
    num_param_groups = len(optimizer.param_groups)
    
    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < warmup_steps:
            lr_factor = float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay phase
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Scale from [0, 1] to [min_lr_ratio, 1]
            lr_factor = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
        
        # 为每个参数组返回相同的学习率因子
        # LambdaLR 会为每个参数组调用这个函数，所以返回单个值即可
        return lr_factor
    
    # 如果优化器有多个参数组，需要为每个参数组提供 lambda 函数
    # 但 LambdaLR 支持单个函数（会被每个参数组调用），所以这样应该可以
    # 如果仍然有问题，可以显式创建函数列表
    if num_param_groups == 1:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # 为每个参数组创建相同的 lambda 函数
        lr_lambdas = [lr_lambda] * num_param_groups
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas)




def calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    params,  # Can be a single Parameter or an iterable of Parameters (e.g., decoder.parameters())
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    """Calculate adaptive weight for GAN loss based on gradient magnitudes.
    
    Args:
        recon_loss: Reconstruction loss tensor
        gan_loss: GAN generator loss tensor
        params: Single parameter or iterable of parameters (e.g., model.decoder.parameters())
        max_d_weight: Maximum value for the adaptive weight
    """
    # Convert to list if it's an iterator/generator
    if hasattr(params, 'parameters'):
        # It's a module, get its parameters
        param_list = list(params.parameters())
    elif isinstance(params, torch.nn.Parameter):
        # Single parameter
        param_list = [params]
    else:
        # Assume it's already an iterable
        param_list = list(params)
    
    # Filter out parameters that don't require grad
    param_list = [p for p in param_list if p.requires_grad]
    
    if len(param_list) == 0:
        return torch.tensor(1.0, device=recon_loss.device)
    
    # Compute gradients for all parameters
    recon_grads = torch.autograd.grad(recon_loss, param_list, retain_graph=True, allow_unused=True)
    gan_grads = torch.autograd.grad(gan_loss, param_list, retain_graph=True, allow_unused=True)
    
    # Compute total gradient norms
    recon_norm_sq = sum(g.pow(2).sum() for g in recon_grads if g is not None)
    gan_norm_sq = sum(g.pow(2).sum() for g in gan_grads if g is not None)
    
    recon_norm = torch.sqrt(recon_norm_sq)
    gan_norm = torch.sqrt(gan_norm_sq)
    
    d_weight = recon_norm / (gan_norm + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()






# ==========================
# Validation
# ==========================

@torch.no_grad()
def run_validation(model, val_loader, device, step, args, logger, use_fsdp=False, tb_writer=None, ema_model=None):
    """
    运行验证。
    
    Args:
        model: 当前模型
        val_loader: 验证数据加载器
        device: 设备
        step: 当前步数
        args: 参数
        logger: 日志记录器
        use_fsdp: 是否使用 FSDP
        tb_writer: TensorBoard writer
        ema_model: EMA 模型（可选，如果提供则使用 EMA 模型进行验证）
    """
    # 如果提供了 EMA 模型，使用 EMA 模型进行验证
    eval_model = ema_model if ema_model is not None else model
    eval_model.eval()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    gt_dir = os.path.join(args.output_dir, f"temp_eval_gt_{step}")
    recon_dir = os.path.join(args.output_dir, f"temp_eval_recon_{step}")

    if rank == 0:
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)
        logger.info(f"Running validation at step {step}...")

    dist.barrier()

    total_psnr = 0.0
    count = 0
    max_batches = args.val_max_batches
    
    # 计算所有 rank 应该迭代的最小 batch 数，确保同步
    # 获取当前 rank 的 val_loader 实际 batch 数
    local_num_batches = len(val_loader)
    local_num_batches_tensor = torch.tensor(local_num_batches, device=device, dtype=torch.long)
    dist.all_reduce(local_num_batches_tensor, op=dist.ReduceOp.MIN)
    min_batches_across_ranks = local_num_batches_tensor.item()
    
    # 实际使用的 batch 数 = min(max_batches, 所有 rank 中最小的 batch 数)
    actual_max_batches = min(max_batches, min_batches_across_ranks)
    
    if rank == 0:
        logger.info(f"Validation: using {actual_max_batches} batches (max_batches={max_batches}, min_across_ranks={min_batches_across_ranks})")

    for i, (x, _) in enumerate(val_loader):
        if i >= actual_max_batches:
            break
        x = x.to(device)

        # Force fp32 precision for validation
        with torch.autocast("cuda", dtype=torch.float32, enabled=False):
            out = eval_model(x)
            recon = out.sample  # [-1,1]
        batch_psnr = calculate_psnr(x, recon)
        total_psnr += batch_psnr.item()
        count += 1

        img_gt = torch.clamp((x + 1.0) / 2.0, 0, 1)
        img_recon = torch.clamp((recon + 1.0) / 2.0, 0, 1)

        for idx in range(x.size(0)):
            fname = f"r{rank}_b{i}_{idx}.png"
            save_image(img_gt[idx], os.path.join(gt_dir, fname))
            save_image(img_recon[idx], os.path.join(recon_dir, fname))

    avg_psnr = torch.tensor(total_psnr / max(count, 1), device=device)
    dist.all_reduce(avg_psnr, op=dist.ReduceOp.SUM)
    avg_psnr = avg_psnr.item() / world_size

    dist.barrier()

    if rank == 0:
        rfid = -1.0
        if HAS_FID:
            try:
                rfid = fid_score.calculate_fid_given_paths(
                    paths=[gt_dir, recon_dir],
                    batch_size=50,
                    device=device,
                    dims=2048,
                    num_workers=4
                )
            except Exception as e:
                logger.error(f"FID Calculation Error: {e}")

        logger.info(f"[Validation Step {step}] PSNR: {avg_psnr:.4f} | rFID: {rfid:.4f}")
        with open(os.path.join(args.output_dir, "metrics.txt"), "a") as f:
            f.write(f"Step: {step}, PSNR: {avg_psnr:.4f}, rFID: {rfid:.4f}\n")

        # TensorBoard logging for validation metrics
        if tb_writer is not None:
            tb_writer.add_scalar("Validation/PSNR", avg_psnr, step)
            if rfid >= 0:  # Only log if FID was successfully calculated
                tb_writer.add_scalar("Validation/rFID", rfid, step)

    # 先同步，确保所有 rank 都完成了 FID 相关操作
    dist.barrier()
    
    # 所有 rank 都尝试删除自己的文件（而不是只让 rank 0 删除整个目录）
    # 这样可以分散 I/O 负载，避免单个 rank 卡住
    # 使用多线程并行删除以加快速度
    if not args.debug:
        import glob
        
        def delete_file(file_path):
            """删除单个文件的辅助函数"""
            try:
                os.remove(file_path)
                return True
            except:
                return False
        
        # 收集所有需要删除的文件
        files_to_delete = []
        for pattern in [f"r{rank}_b*_*.png"]:
            files_to_delete.extend(glob.glob(os.path.join(gt_dir, pattern)))
            files_to_delete.extend(glob.glob(os.path.join(recon_dir, pattern)))
        
        # 使用多线程并行删除
        if files_to_delete:
            num_workers = min(32, len(files_to_delete))  # 最多32个线程
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(delete_file, f) for f in files_to_delete]
                # 等待所有删除操作完成（可选：检查结果）
                for future in as_completed(futures):
                    future.result()  # 获取结果，忽略异常
    
    dist.barrier()
    
    # rank 0 最后清理空目录
    if rank == 0 and not args.debug:
        shutil.rmtree(gt_dir, ignore_errors=True)
        shutil.rmtree(recon_dir, ignore_errors=True)

    dist.barrier()
    eval_model.train()
    if model is not None:
        model.train()


# ==========================
# Main
# ==========================

def load_config_file(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()

    # 配置文件支持
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--val-path", type=str, default="/share/project/datasets/ImageNet/val")
    parser.add_argument("--output-dir", type=str, default="results/dinov3_vae_lora_vf")
    parser.add_argument("--vae-ckpt", type=str, default="", help="optional init checkpoint")
    
    # Encoder 配置
    parser.add_argument("--encoder-type", type=str, default="dinov3", choices=["dinov3", "siglip2", "dinov2"],
                        help="Encoder type: 'dinov3', 'siglip2', or 'dinov2'")
    parser.add_argument("--dinov3-dir", type=str, default="/cpfs01/huangxu/models/dinov3")
    parser.add_argument("--siglip2-model-name", type=str, default="google/siglip2-base-patch16-256",
                        help="SigLIP2 model name from HuggingFace")
    parser.add_argument("--dinov2-model-name", type=str, default="facebook/dinov2-with-registers-base",
                        help="DINOv2 model name from HuggingFace (e.g., 'facebook/dinov2-with-registers-base')")
    
    # Decoder 配置
    parser.add_argument("--decoder-type", type=str, default="cnn_decoder", 
                        choices=["cnn_decoder", "vit_decoder"],
                        help="Decoder type: 'cnn_decoder' or 'vit_decoder'")
    parser.add_argument("--latent-channels", type=int, default=None, 
                        help="Latent channels (auto-detected from encoder if not set)")
    parser.add_argument("--dec-block-out-channels", type=str, default="1280,1024,512,256,128",
                        help="Decoder block output channels, comma-separated")
    # ViT decoder 配置
    parser.add_argument("--vit-hidden-size", type=int, default=1024, help="ViT decoder hidden size")
    parser.add_argument("--vit-num-layers", type=int, default=24, help="ViT decoder number of layers")
    parser.add_argument("--vit-num-heads", type=int, default=16, help="ViT decoder number of heads")
    parser.add_argument("--vit-intermediate-size", type=int, default=4096, help="ViT decoder intermediate size")

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=None, 
                        help="Patch size (auto-detected from encoder if not set)")
    parser.add_argument("--spatial-downsample-factor", type=int, default=None,
                        help="Spatial downsample factor (auto-detected from patch_size if not set)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4, help="Max learning rate")
    parser.add_argument("--min-lr", type=float, default=2e-5, help="Min learning rate for cosine decay")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps (0 to disable)")
    parser.add_argument("--scheduler", type=str, default="constant", choices=["constant", "cosine"],
                        help="Learning rate scheduler: 'constant' or 'cosine'")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument("--max-steps", type=int, default=100000)

    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--lpips-weight", type=float, default=0.5)

    parser.add_argument("--kl-weight", type=float, default=1e-6)

    # VF loss
    parser.add_argument("--vf-weight", type=float, default=0.1)
    parser.add_argument("--vf-m1", type=float, default=0.1)
    parser.add_argument("--vf-m2", type=float, default=0.1)
    parser.add_argument("--vf-max-tokens", type=int, default=32)
    parser.add_argument("--no-vf", action="store_true", help="Disable VF loss")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA")

    # model regularization
    parser.add_argument("--noise-tau", type=float, default=0.0)
    parser.add_argument("--mask-channels", type=float, default=0.0)
    
    # VAE output options
    parser.add_argument("--denormalize-decoder-output", action="store_true",
                        help="Denormalize decoder output in VAE.")
    parser.add_argument("--skip-to-moments", action="store_true",
                        help="Skip loading to_moments layer (for old checkpoints without it).")

    # GAN - DinoDisc
    parser.add_argument("--gan-weight", type=float, default=0.0, help="GAN loss weight (0 to disable)")
    parser.add_argument("--gan-start-step", type=int, default=0, help="Step to start GAN training")
    parser.add_argument("--disc-lr", type=float, default=1e-4, help="Discriminator learning rate")
    parser.add_argument("--max-d-weight", type=float, default=1e4, help="Max adaptive weight for GAN loss")
    # DinoDisc discriminator config
    parser.add_argument("--dino-ckpt-path", type=str, default="./dino_vit_small_patch8_224.pth",
                        help="Path to DINO checkpoint for DinoDisc")
    parser.add_argument("--disc-recipe", type=str, default="S_8", choices=["S_8", "S_16", "B_16"],
                        help="DinoDisc recipe (S_8, S_16, or B_16)")
    parser.add_argument("--disc-ks", type=int, default=3, help="DinoDisc head kernel size")
    parser.add_argument("--disc-norm", type=str, default="bn", choices=["bn", "gn"], help="DinoDisc normalization")
    parser.add_argument("--disc-key-depths", type=str, default="2,5,8,11",
                        help="DinoDisc key depths for feature extraction, comma-separated")
    parser.add_argument("--diffaug-prob", type=float, default=1.0, help="DiffAug probability")
    parser.add_argument("--diffaug-cutout", type=float, default=0.2, help="DiffAug cutout ratio")
    parser.add_argument("--no-gan", action="store_true", help="Disable GAN loss")

    # Stage training mode for GAN
    parser.add_argument("--stage-disc-steps", type=int, default=0,
                        help="Number of steps to train discriminator only (stage_disc mode). "
                             "If 0, train jointly from the start (stage_joint mode).")

    # Stage 1 training: train to_moments + decoder only (no LoRA, no VF loss)
    # This is a separate training stage concept, not related to GAN
    parser.add_argument("--stage1-steps", type=int, default=0,
                        help="Stage 1 steps: train to_moments + decoder only (freeze encoder/LoRA, no VF loss). "
                             "After stage1_steps, transition to Stage 2: enable LoRA + VF loss. "
                             "If 0, skip Stage 1 and train everything from start.")

    # FSDP
    parser.add_argument("--fsdp-size", type=int, default=1,
                        help="FSDP sharding size. 1=disabled (use DDP), >1=enable FSDP to shard optimizer states.")

    # Precision
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="bf16",
                        help="Compute precision for training.")

    # freq
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--val-max-batches", type=int, default=200)
    parser.add_argument("--debug", action="store_true")
    
    # EMA 配置
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument("--use-ema", action="store_true", help="Enable EMA (Exponential Moving Average)")
    parser.add_argument("--val-use-ema", action="store_true", help="Use EMA model for validation")
    
    # Resume 配置
    parser.add_argument("--no-resume-optimizer", action="store_true", 
                        help="Do not resume optimizer state from checkpoint (start with fresh optimizer)")

    args = parser.parse_args()
    
    # 如果指定了配置文件，加载并覆盖默认值
    if args.config:
        config = load_config_file(args.config)
        # 从配置文件更新 args
        if 'data' in config:
            if args.train_path is None and 'train_path' in config['data']:
                args.train_path = config['data']['train_path']
            if 'val_path' in config['data']:
                args.val_path = config['data']['val_path']
            if 'image_size' in config['data']:
                args.image_size = config['data']['image_size']
            if 'batch_size' in config['data']:
                args.batch_size = config['data']['batch_size']
        
        if 'encoder' in config:
            enc_cfg = config['encoder']
            if 'type' in enc_cfg:
                args.encoder_type = enc_cfg['type']
            if 'dinov3' in enc_cfg and 'model_dir' in enc_cfg['dinov3']:
                args.dinov3_dir = enc_cfg['dinov3']['model_dir']
            if 'dinov3' in enc_cfg and 'patch_size' in enc_cfg['dinov3']:
                if args.patch_size is None:
                    args.patch_size = enc_cfg['dinov3']['patch_size']
            if 'siglip2' in enc_cfg and 'model_name' in enc_cfg['siglip2']:
                args.siglip2_model_name = enc_cfg['siglip2']['model_name']
            if 'dinov2' in enc_cfg and 'model_name' in enc_cfg['dinov2']:
                args.dinov2_model_name = enc_cfg['dinov2']['model_name']
            if 'dinov2' in enc_cfg and 'patch_size' in enc_cfg['dinov2']:
                if args.patch_size is None:
                    args.patch_size = enc_cfg['dinov2']['patch_size']
        
        if 'decoder' in config:
            dec_cfg = config['decoder']
            if 'type' in dec_cfg:
                args.decoder_type = dec_cfg['type']
            if 'latent_channels' in dec_cfg:
                args.latent_channels = dec_cfg['latent_channels']
            # CNN decoder 配置
            if 'cnn' in dec_cfg and 'block_out_channels' in dec_cfg['cnn']:
                args.dec_block_out_channels = ','.join(map(str, dec_cfg['cnn']['block_out_channels']))
            elif 'block_out_channels' in dec_cfg:
                args.dec_block_out_channels = ','.join(map(str, dec_cfg['block_out_channels']))
            # ViT decoder 配置
            if 'vit' in dec_cfg:
                vit_cfg = dec_cfg['vit']
                if 'hidden_size' in vit_cfg:
                    args.vit_hidden_size = vit_cfg['hidden_size']
                if 'num_layers' in vit_cfg:
                    args.vit_num_layers = vit_cfg['num_layers']
                if 'num_heads' in vit_cfg:
                    args.vit_num_heads = vit_cfg['num_heads']
                if 'intermediate_size' in vit_cfg:
                    args.vit_intermediate_size = vit_cfg['intermediate_size']
            # spatial_downsample_factor
            if 'spatial_downsample_factor' in dec_cfg:
                if args.spatial_downsample_factor is None:
                    args.spatial_downsample_factor = dec_cfg['spatial_downsample_factor']
        
        if 'lora' in config:
            lora_cfg = config['lora']
            if 'rank' in lora_cfg:
                args.lora_rank = lora_cfg['rank']
            if 'alpha' in lora_cfg:
                args.lora_alpha = lora_cfg['alpha']
            if 'dropout' in lora_cfg:
                args.lora_dropout = lora_cfg['dropout']
            if 'enabled' in lora_cfg and not lora_cfg['enabled']:
                args.no_lora = True
        
        if 'loss' in config:
            loss_cfg = config['loss']
            if 'l1_weight' in loss_cfg:
                args.l1_weight = loss_cfg['l1_weight']
            if 'lpips_weight' in loss_cfg:
                args.lpips_weight = loss_cfg['lpips_weight']
            if 'kl_weight' in loss_cfg:
                args.kl_weight = loss_cfg['kl_weight']
            
            if 'vf' in loss_cfg:
                vf_cfg = loss_cfg['vf']
                if 'weight' in vf_cfg:
                    args.vf_weight = vf_cfg['weight']
                if 'margin_cos' in vf_cfg:
                    args.vf_m1 = vf_cfg['margin_cos']
                if 'margin_dms' in vf_cfg:
                    args.vf_m2 = vf_cfg['margin_dms']
                if 'max_tokens' in vf_cfg:
                    args.vf_max_tokens = vf_cfg['max_tokens']
                if 'enabled' in vf_cfg and not vf_cfg['enabled']:
                    args.no_vf = True
            
            if 'gan' in loss_cfg:
                gan_cfg = loss_cfg['gan']
                if 'weight' in gan_cfg:
                    args.gan_weight = gan_cfg['weight']
                if 'start_step' in gan_cfg:
                    args.gan_start_step = gan_cfg['start_step']
                if 'max_d_weight' in gan_cfg:
                    args.max_d_weight = gan_cfg['max_d_weight']
                if 'stage_disc_steps' in gan_cfg:
                    args.stage_disc_steps = gan_cfg['stage_disc_steps']
                if 'enabled' in gan_cfg and not gan_cfg['enabled']:
                    args.no_gan = True
        
        if 'discriminator' in config:
            disc_cfg = config['discriminator']
            if 'dino_ckpt_path' in disc_cfg:
                args.dino_ckpt_path = disc_cfg['dino_ckpt_path']
            if 'recipe' in disc_cfg:
                args.disc_recipe = disc_cfg['recipe']
            if 'ks' in disc_cfg:
                args.disc_ks = disc_cfg['ks']
            if 'norm' in disc_cfg:
                args.disc_norm = disc_cfg['norm']
            if 'key_depths' in disc_cfg:
                args.disc_key_depths = ','.join(map(str, disc_cfg['key_depths']))
            if 'lr' in disc_cfg:
                args.disc_lr = disc_cfg['lr']
            if 'diffaug_prob' in disc_cfg:
                args.diffaug_prob = disc_cfg['diffaug_prob']
            if 'diffaug_cutout' in disc_cfg:
                args.diffaug_cutout = disc_cfg['diffaug_cutout']
        
        if 'optimizer' in config:
            opt_cfg = config['optimizer']
            if 'lr' in opt_cfg:
                args.lr = opt_cfg['lr']
            if 'min_lr' in opt_cfg:
                args.min_lr = opt_cfg['min_lr']
            if 'warmup_steps' in opt_cfg:
                args.warmup_steps = opt_cfg['warmup_steps']
            if 'scheduler' in opt_cfg:
                args.scheduler = opt_cfg['scheduler']
            if 'weight_decay' in opt_cfg:
                args.weight_decay = opt_cfg['weight_decay']
        
        if 'training' in config:
            train_cfg = config['training']
            if 'max_steps' in train_cfg:
                args.max_steps = train_cfg['max_steps']
            if 'precision' in train_cfg:
                args.precision = train_cfg['precision']
            if 'fsdp_size' in train_cfg:
                args.fsdp_size = train_cfg['fsdp_size']
            # Stage 1: train to_moments + decoder only (no LoRA, no VF loss)
            if 'stage1_steps' in train_cfg:
                args.stage1_steps = train_cfg['stage1_steps']
        
        if 'logging' in config:
            log_cfg = config['logging']
            if 'output_dir' in log_cfg:
                args.output_dir = log_cfg['output_dir']
            if 'log_every' in log_cfg:
                args.log_every = log_cfg['log_every']
            if 'eval_every' in log_cfg:
                args.eval_every = log_cfg['eval_every']
            if 'save_every' in log_cfg:
                args.save_every = log_cfg['save_every']
            if 'val_max_batches' in log_cfg:
                args.val_max_batches = log_cfg['val_max_batches']
            if 'debug' in log_cfg:
                args.debug = log_cfg['debug']
        
        if 'checkpoint' in config:
            ckpt_cfg = config['checkpoint']
            if 'vae_ckpt' in ckpt_cfg and ckpt_cfg['vae_ckpt']:
                args.vae_ckpt = ckpt_cfg['vae_ckpt']
            if 'no_resume_optimizer' in ckpt_cfg:
                args.no_resume_optimizer = ckpt_cfg['no_resume_optimizer']
        
        if 'vae' in config:
            vae_cfg = config['vae']
            if 'noise_tau' in vae_cfg:
                args.noise_tau = vae_cfg['noise_tau']
            if 'mask_channels' in vae_cfg:
                args.mask_channels = vae_cfg['mask_channels']
            if 'denormalize_decoder_output' in vae_cfg:
                args.denormalize_decoder_output = vae_cfg['denormalize_decoder_output']
            if 'skip_to_moments' in vae_cfg:
                args.skip_to_moments = vae_cfg['skip_to_moments']
        
        if 'ema' in config:
            ema_cfg = config['ema']
            if 'enabled' in ema_cfg:
                args.use_ema = ema_cfg['enabled']
            if 'decay' in ema_cfg:
                args.ema_decay = ema_cfg['decay']
            if 'val_use_ema' in ema_cfg:
                args.val_use_ema = ema_cfg['val_use_ema']
    
    # 处理 --no-xxx 参数
    if args.no_lora:
        args.lora_rank = 0
    if args.no_vf:
        args.vf_weight = 0.0
    if args.no_gan:
        args.gan_weight = 0.0
    
    # 检查必需参数
    if args.train_path is None:
        raise ValueError("--train-path is required (or set via config file)")
    
    # 解析 decoder block out channels
    args.dec_block_out_channels = tuple(int(x) for x in args.dec_block_out_channels.split(','))

    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    logger = create_logger(args.output_dir, rank)
    if rank == 0:
        logger.info(f"Starting training: {args}")

    # TensorBoard setup
    tb_writer = None
    if rank == 0 and HAS_TENSORBOARD:
        # 统一放到 tensorboard_logs 目录下，名字为 output_dir 最后一项，方便不同实验对比
        exp_name = os.path.basename(args.output_dir.rstrip('/'))
        tb_log_dir = os.path.join("tensorboard_logs", exp_name)
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logging to: {tb_log_dir}")

    # Precision settings
    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    if rank == 0:
        logger.info(f"AMP enabled: {use_bf16}, dtype: {amp_dtype}")

    # Dataset
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),  # [-1,1]
    ])
    val_transform = transforms.Compose([
        transforms.Lambda(lambda pil: center_crop_arr(pil, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),  # [-1,1]
    ])

    train_dataset = ImageFolder(args.train_path, transform=train_transform)
    
    # 验证集支持两种格式：ImageFolder（有子文件夹分类结构）或纯图片文件夹
    try:
        val_dataset = ImageFolder(args.val_path, transform=val_transform)
    except:
        # root 下面直接是图片，没有子文件夹
        class ImageDataset(Dataset):
            EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
            def __init__(self, root, transform=None):
                self.transform = transform
                self.paths = sorted([os.path.join(root, f) for f in os.listdir(root) 
                                     if os.path.splitext(f)[1].lower() in self.EXTS])
            def __len__(self):
                return len(self.paths)
            def __getitem__(self, idx):
                img = Image.open(self.paths[idx]).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, 0
        val_dataset = ImageDataset(args.val_path, transform=val_transform)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=4, pin_memory=True, drop_last=False)

    # 使用 get_encoder_config 获取 encoder 默认配置
    try:
        encoder_config = get_encoder_config(args.encoder_type)
        default_latent_channels = encoder_config["latent_channels"]
        default_patch_size = encoder_config["patch_size"]
    except ValueError:
        # 未知的 encoder_type，使用默认值
        default_latent_channels = 1280
        default_patch_size = 16
    
    # 根据 encoder_type 设置默认 latent_channels
    if args.latent_channels is None:
        args.latent_channels = default_latent_channels
    
    # 根据 encoder_type 设置默认 patch_size 和 spatial_downsample_factor
    if args.patch_size is None:
        args.patch_size = default_patch_size
    
    if args.spatial_downsample_factor is None:
        args.spatial_downsample_factor = args.patch_size
    
    if rank == 0:
        logger.info(f"Encoder type: {args.encoder_type}")
        logger.info(f"Decoder type: {args.decoder_type}")
        logger.info(f"Image size: {args.image_size}, Patch size: {args.patch_size}, Spatial downsample: {args.spatial_downsample_factor}")
        logger.info(f"Latent channels: {args.latent_channels}")
        if args.decoder_type == "cnn_decoder":
            logger.info(f"Decoder block out channels: {args.dec_block_out_channels}")
        else:
            logger.info(f"ViT decoder: hidden={args.vit_hidden_size}, layers={args.vit_num_layers}, heads={args.vit_num_heads}")
        logger.info(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
        logger.info(f"VF weight: {args.vf_weight}, GAN weight: {args.gan_weight}")

    # Model
    model = AutoencoderKL(
        # Encoder 配置
        encoder_type=args.encoder_type,
        dinov3_model_dir=args.dinov3_dir,
        siglip2_model_name=args.siglip2_model_name,
        dinov2_model_name=args.dinov2_model_name,
        
        image_size=args.image_size,
        patch_size=args.patch_size,
        out_channels=3,

        latent_channels=args.latent_channels,
        target_latent_channels=None,
        spatial_downsample_factor=args.spatial_downsample_factor,

        # Decoder 配置
        decoder_type=args.decoder_type,
        
        # CNN decoder 参数
        dec_block_out_channels=args.dec_block_out_channels,
        dec_layers_per_block=3,
        decoder_dropout=0.0,
        gradient_checkpointing=False,
        
        # ViT decoder 参数
        vit_decoder_hidden_size=args.vit_hidden_size,
        vit_decoder_num_layers=args.vit_num_layers,
        vit_decoder_num_heads=args.vit_num_heads,
        vit_decoder_intermediate_size=args.vit_intermediate_size,

        variational=True,
        kl_weight=args.kl_weight,

        noise_tau=args.noise_tau,
        random_masking_channel_ratio=args.mask_channels,

        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        enable_lora=(args.lora_rank > 0),
        vf_margin_cos=args.vf_m1,
        vf_margin_dms=args.vf_m2,
        vf_max_tokens=args.vf_max_tokens,
        vf_hyper=args.vf_weight,
        vf_use_adaptive_weight=True,
        training_mode="enc_dec",
        denormalize_decoder_output=args.denormalize_decoder_output,
        skip_to_moments=args.skip_to_moments,
    ).to(device)

    # Load optional ckpt
    resume_ckpt = None  # 用于后续恢复 optimizer/discriminator/step
    if args.vae_ckpt and os.path.isfile(args.vae_ckpt):
        if rank == 0:
            logger.info(f"Loading weights from {args.vae_ckpt}")
        resume_ckpt = torch.load(args.vae_ckpt, map_location="cpu")
        if "state_dict" in resume_ckpt:
            state_dict = resume_ckpt["state_dict"]
        elif "model_state_dict" in resume_ckpt:
            state_dict = resume_ckpt["model_state_dict"]
        elif "model" in resume_ckpt:
            state_dict = resume_ckpt["model"]
        else:
            state_dict = resume_ckpt
            resume_ckpt = None  # 如果 ckpt 就是 state_dict，则不尝试恢复其他状态
        
        # Skip to_moments layer if requested (for old checkpoints without it)
        if args.skip_to_moments:
            state_dict = {k: v for k, v in state_dict.items() if "to_moments" not in k}
            if rank == 0:
                logger.info("Skipping to_moments layer loading (--skip-to-moments)")
        
        model.load_state_dict(state_dict, strict=False)

    # Freeze base encoder weights; train LoRA + decoder + moments head (+ optional downsample convs)
    requires_grad(model, False)

    # decoder train
    requires_grad(model.decoder, True)

    # moments head train
    if model.to_moments is not None:
        requires_grad(model.to_moments, True)
    else:
        if rank == 0:
            logger.info("to_moments is None")

    # extra convs
    for conv in model.latent_downsample_layers:
        requires_grad(conv, True)
    if model.channel_downsample_conv is not None:
        requires_grad(model.channel_downsample_conv, True)

    # LoRA params train (match by module types, not names)
    # 获取 encoder backbone (支持多种 encoder 类型)
    encoder_backbone = model.encoder.get_backbone()
    
    step = 0
    epoch = 0
    if resume_ckpt is not None:
        # 恢复 step
        if "step" in resume_ckpt:
            step = resume_ckpt["step"]
            if rank == 0:
                logger.info(f"Resuming from step {step}")
    # Stage 1 时冻结 LoRA，只训练 to_moments + decoder
    # Stage 2 时解冻 LoRA，一起训练
    in_stage1 = args.stage1_steps > 0 and step < args.stage1_steps
    enable_lora_training = not in_stage1 and args.lora_rank > 0
    
    for m in encoder_backbone.modules():
        # LoRALinear/LoRAConv2d contains lora_down/lora_up
        if hasattr(m, "lora_down") and hasattr(m, "lora_up"):
            for p in m.lora_down.parameters():
                p.requires_grad = enable_lora_training
            for p in m.lora_up.parameters():
                p.requires_grad = enable_lora_training
        # keep base frozen
        if hasattr(m, "base") and isinstance(getattr(m, "base"), (nn.Linear, nn.Conv2d)):
            for p in m.base.parameters():
                p.requires_grad = False

    lora_params = [p for n, p in encoder_backbone.named_parameters()
                   if p.requires_grad and (("lora_up" in n) or ("lora_down" in n))]
    
    if rank == 0:
        num_lora_params = sum(p.numel() for p in lora_params)
        logger.info(f"LoRA trainable parameters: {num_lora_params / 1e6:.2f}M")
        if args.stage1_steps > 0:
            logger.info(f"Stage 1 mode: training to_moments + decoder only (no LoRA, no VF loss) for {args.stage1_steps} steps")
    
    model.train()

    # Wrap model with DDP or FSDP
    use_fsdp = args.fsdp_size > 1
    if use_fsdp:
        fsdp_mixed_precision = None
        if use_bf16:
            fsdp_mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=fsdp_mixed_precision,
            device_id=local_rank,
            use_orig_params=True,
        )
        if rank == 0:
            logger.info(f"Using FSDP with sharding_strategy=FULL_SHARD, fsdp_size={args.fsdp_size}")
    else:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        if rank == 0:
            logger.info("Using DDP")

    # Optimizer for generator (VAE)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.5, 0.9),
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler for VAE
    scheduler = None
    if args.scheduler == "cosine":
        min_lr_ratio = args.min_lr / args.lr if args.lr > 0 else 0.1
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=opt,
            warmup_steps=args.warmup_steps,
            total_steps=args.max_steps,
            min_lr_ratio=min_lr_ratio,
        )
        if rank == 0:
            logger.info(f"Using cosine scheduler: warmup={args.warmup_steps}, max_lr={args.lr}, min_lr={args.min_lr}")
    else:
        if rank == 0:
            logger.info(f"Using constant learning rate: {args.lr}")

    loss_lpips = LPIPSLoss(device)

    # ========== Ref Model Setup (for VF loss when no-lora) ==========
    ref_model = None
    if args.lora_rank == 0 and args.vf_weight > 0.0:
        # 当 no-lora 时，创建 ref_model 用于 VF loss 计算
        model_inner = unwrap_model(model)
        ref_model = AutoencoderKL(
            # Encoder 配置
            encoder_type=args.encoder_type,
            dinov3_model_dir=args.dinov3_dir,
            siglip2_model_name=args.siglip2_model_name,
            dinov2_model_name=args.dinov2_model_name,
            
            image_size=args.image_size,
            patch_size=args.patch_size,
            out_channels=3,

            latent_channels=args.latent_channels,
            target_latent_channels=None,
            spatial_downsample_factor=args.spatial_downsample_factor,

            # Decoder 配置
            decoder_type=args.decoder_type,
            
            # CNN decoder 参数
            dec_block_out_channels=args.dec_block_out_channels,
            dec_layers_per_block=3,
            decoder_dropout=0.0,
            gradient_checkpointing=False,
            
            # ViT decoder 参数
            vit_decoder_hidden_size=args.vit_hidden_size,
            vit_decoder_num_layers=args.vit_num_layers,
            vit_decoder_num_heads=args.vit_num_heads,
            vit_decoder_intermediate_size=args.vit_intermediate_size,

            variational=True,
            kl_weight=args.kl_weight,

            noise_tau=args.noise_tau,
            random_masking_channel_ratio=args.mask_channels,

            lora_rank=0,  # ref_model 不使用 LoRA
            lora_alpha=0,
            lora_dropout=0.0,
            enable_lora=False,  # ref_model 不使用 LoRA
            vf_margin_cos=args.vf_m1,
            vf_margin_dms=args.vf_m2,
            vf_max_tokens=args.vf_max_tokens,
            vf_hyper=args.vf_weight,
            vf_use_adaptive_weight=True,
            training_mode="enc_dec",
            denormalize_decoder_output=args.denormalize_decoder_output,
            skip_to_moments=args.skip_to_moments,
        ).to(device)
        
        # 初始化 ref_model 为当前模型的状态（但不包括 LoRA，因为 ref_model 没有 LoRA）
        ref_model.load_state_dict(model_inner.state_dict(), strict=False)
        ref_model.eval()
        requires_grad(ref_model, False)
        
        if rank == 0:
            logger.info("Created ref_model for VF loss (no-lora mode)")

    # ========== EMA Setup ==========
    ema_model = None
    if args.use_ema:
        # 创建 EMA 模型（复制当前模型结构，但不包装 DDP/FSDP）
        model_inner = unwrap_model(model)
        ema_model = AutoencoderKL(
            # Encoder 配置
            encoder_type=args.encoder_type,
            dinov3_model_dir=args.dinov3_dir,
            siglip2_model_name=args.siglip2_model_name,
            dinov2_model_name=args.dinov2_model_name,
            
            image_size=args.image_size,
            patch_size=args.patch_size,
            out_channels=3,

            latent_channels=args.latent_channels,
            target_latent_channels=None,
            spatial_downsample_factor=args.spatial_downsample_factor,

            # Decoder 配置
            decoder_type=args.decoder_type,
            
            # CNN decoder 参数
            dec_block_out_channels=args.dec_block_out_channels,
            dec_layers_per_block=3,
            decoder_dropout=0.0,
            gradient_checkpointing=False,
            
            # ViT decoder 参数
            vit_decoder_hidden_size=args.vit_hidden_size,
            vit_decoder_num_layers=args.vit_num_layers,
            vit_decoder_num_heads=args.vit_num_heads,
            vit_decoder_intermediate_size=args.vit_intermediate_size,

            variational=True,
            kl_weight=args.kl_weight,

            noise_tau=args.noise_tau,
            random_masking_channel_ratio=args.mask_channels,

            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            enable_lora=(args.lora_rank > 0),
            vf_margin_cos=args.vf_m1,
            vf_margin_dms=args.vf_m2,
            vf_max_tokens=args.vf_max_tokens,
            vf_hyper=args.vf_weight,
            vf_use_adaptive_weight=True,
            training_mode="enc_dec",
            denormalize_decoder_output=args.denormalize_decoder_output,
            skip_to_moments=args.skip_to_moments,
        ).to(device)
        
        # 初始化 EMA 模型为当前模型的状态
        ema_model.load_state_dict(model_inner.state_dict(),strict=False)
        ema_model.eval()
        requires_grad(ema_model, False)
        
        if rank == 0:
            logger.info(f"EMA enabled with decay={args.ema_decay}")
            if args.val_use_ema:
                logger.info("Using EMA model for validation")

    # ========== GAN Setup - DinoDisc ==========
    use_gan = args.gan_weight > 0.0
    discriminator = None
    opt_disc = None
    scheduler_disc = None
    diffaug = None

    if use_gan:
        # Parse key_depths
        disc_key_depths = tuple(int(x) for x in args.disc_key_depths.split(','))
        
        # Build DinoDisc discriminator
        discriminator = DinoDisc(
            device=device,
            dino_ckpt_path=args.dino_ckpt_path,
            ks=args.disc_ks,
            key_depths=disc_key_depths,
            norm_type=args.disc_norm,
            using_spec_norm=True,
            norm_eps=1e-6,
            recipe=args.disc_recipe,
        ).to(device)
        
        # DiffAug for discriminator
        diffaug = DiffAug(prob=args.diffaug_prob, cutout=args.diffaug_cutout)

        # Wrap discriminator with DDP or FSDP (only wrap trainable heads)
        if use_fsdp:
            fsdp_mixed_precision_disc = None
            if use_bf16:
                fsdp_mixed_precision_disc = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )
            discriminator = FSDP(
                discriminator,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=fsdp_mixed_precision_disc,
                device_id=local_rank,
                use_orig_params=True,
            )
        else:
            discriminator = DDP(discriminator, device_ids=[local_rank], find_unused_parameters=False)
        discriminator.train()

        opt_disc = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, discriminator.parameters()),
            lr=args.disc_lr,
            betas=(0.5, 0.9),
            weight_decay=args.weight_decay,
        )
        
        # Learning rate scheduler for discriminator (same schedule as VAE)
        scheduler_disc = None
        if args.scheduler == "cosine":
            # Discriminator training starts at gan_start_step
            # Total steps for discriminator = max_steps - gan_start_step
            disc_total_steps = args.max_steps - args.gan_start_step
            disc_min_lr_ratio = args.min_lr / args.disc_lr if args.disc_lr > 0 else 0.1
            scheduler_disc = get_cosine_schedule_with_warmup(
                optimizer=opt_disc,
                warmup_steps=0,  # No warmup for discriminator (starts later)
                total_steps=disc_total_steps,
                min_lr_ratio=disc_min_lr_ratio,
            )

        if rank == 0:
            num_disc_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
            logger.info(f"DinoDisc Discriminator trainable parameters: {num_disc_params / 1e6:.2f}M")
            logger.info(f"DinoDisc recipe: {args.disc_recipe}, key_depths: {disc_key_depths}")
            logger.info(f"DiffAug: {diffaug}")
            logger.info(f"GAN weight: {args.gan_weight}, start step: {args.gan_start_step}")
            logger.info(f"Stage disc steps: {args.stage_disc_steps} (0 = joint training from start)")

    # ========== 从 checkpoint 恢复 optimizer/discriminator/scheduler/step ==========
    if resume_ckpt is not None:
        # 恢复 VAE optimizer
        if "opt" in resume_ckpt and not args.no_resume_optimizer:
            try:
                opt.load_state_dict(resume_ckpt["opt"])
                if rank == 0:
                    logger.info("Loaded VAE optimizer state from checkpoint")
            except Exception as e:
                if rank == 0:
                    logger.warning(f"Failed to load VAE optimizer state: {e}")
        elif args.no_resume_optimizer and rank == 0:
            logger.info("Skipping optimizer resume (--no-resume-optimizer)")
        
        # 恢复 VAE scheduler
        if "scheduler" in resume_ckpt and scheduler is not None and not args.no_resume_optimizer:
            try:
                # 检查调度器状态中的参数组数量是否与当前优化器匹配
                scheduler_state = resume_ckpt["scheduler"]
                if "base_lrs" in scheduler_state:
                    saved_num_groups = len(scheduler_state["base_lrs"])
                    current_num_groups = len(opt.param_groups)
                    if saved_num_groups != current_num_groups:
                        if rank == 0:
                            logger.warning(
                                f"Scheduler param_groups mismatch: saved={saved_num_groups}, "
                                f"current={current_num_groups}. Skipping scheduler restore."
                            )
                    else:
                        scheduler.load_state_dict(scheduler_state)
                        if rank == 0:
                            logger.info("Loaded VAE scheduler state from checkpoint")
                else:
                    scheduler.load_state_dict(scheduler_state)
                    if rank == 0:
                        logger.info("Loaded VAE scheduler state from checkpoint")
            except Exception as e:
                if rank == 0:
                    logger.warning(f"Failed to load VAE scheduler state: {e}")
        
        # 恢复 discriminator 和其 optimizer/scheduler
        if use_gan and discriminator is not None:
            if "discriminator" in resume_ckpt:
                try:
                    disc_state = resume_ckpt["discriminator"]
                    unwrap_model(discriminator).load_state_dict(disc_state)
                    if rank == 0:
                        logger.info("Loaded discriminator state from checkpoint")
                except Exception as e:
                    if rank == 0:
                        logger.warning(f"Failed to load discriminator state: {e}")
            
            if "opt_disc" in resume_ckpt and opt_disc is not None and not args.no_resume_optimizer:
                try:
                    opt_disc.load_state_dict(resume_ckpt["opt_disc"])
                    if rank == 0:
                        logger.info("Loaded discriminator optimizer state from checkpoint")
                except Exception as e:
                    if rank == 0:
                        logger.warning(f"Failed to load discriminator optimizer state: {e}")
            
            if "scheduler_disc" in resume_ckpt and scheduler_disc is not None and not args.no_resume_optimizer:
                try:
                    # 检查调度器状态中的参数组数量是否与当前优化器匹配
                    scheduler_disc_state = resume_ckpt["scheduler_disc"]
                    if "base_lrs" in scheduler_disc_state:
                        saved_num_groups = len(scheduler_disc_state["base_lrs"])
                        current_num_groups = len(opt_disc.param_groups)
                        if saved_num_groups != current_num_groups:
                            if rank == 0:
                                logger.warning(
                                    f"Discriminator scheduler param_groups mismatch: "
                                    f"saved={saved_num_groups}, current={current_num_groups}. "
                                    f"Skipping scheduler restore."
                                )
                        else:
                            scheduler_disc.load_state_dict(scheduler_disc_state)
                            if rank == 0:
                                logger.info("Loaded discriminator scheduler state from checkpoint")
                    else:
                        scheduler_disc.load_state_dict(scheduler_disc_state)
                        if rank == 0:
                            logger.info("Loaded discriminator scheduler state from checkpoint")
                except Exception as e:
                    if rank == 0:
                        logger.warning(f"Failed to load discriminator scheduler state: {e}")
        
        # 恢复 EMA 模型
        # if args.use_ema and ema_model is not None and "ema_model" in resume_ckpt:
        #     try:
        #         ema_model.load_state_dict(resume_ckpt["ema_model"],strict=False)
        #         if rank == 0:
        #             logger.info("Loaded EMA model state from checkpoint")
        #     except Exception as e:
        #         if rank == 0:
        #             logger.warning(f"Failed to load EMA model state: {e}")
        
        # 恢复 ref_model (no-lora 模式)
        if ref_model is not None:
            if "ref_model" in resume_ckpt:
                try:
                    ref_model.load_state_dict(resume_ckpt["ref_model"], strict=False)
                    if rank == 0:
                        logger.info("Loaded ref_model state from checkpoint")
                except Exception as e:
                    if rank == 0:
                        logger.warning(f"Failed to load ref_model state: {e}, initializing from current model")
                    # 如果恢复失败，从当前模型重新初始化
                    model_inner = unwrap_model(model)
                    ref_model.load_state_dict(model_inner.state_dict(), strict=False)
            else:
                # checkpoint 中没有 ref_model，从当前模型初始化
                model_inner = unwrap_model(model)
                ref_model.load_state_dict(model_inner.state_dict(), strict=False)
                if rank == 0:
                    logger.info("Initialized ref_model from current model (checkpoint had no ref_model)")

    # Loop
    if rank == 0:
        logger.info("Start Training Loop...")
    
    # Track stage transition
    lora_enabled_in_training = not (args.stage1_steps > 0 and step < args.stage1_steps)
    stage1_just_ended = False
    
    # Keep track of current lora_params for VF loss (updated during stage transition)
    current_lora_params = lora_params if lora_enabled_in_training else []

    while step < args.max_steps:
        train_sampler.set_epoch(epoch)
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)  # [-1,1]

            # Check if we need to transition from Stage 1 to Stage 2
            # Stage 1: train to_moments + decoder only (no LoRA, no VF loss)
            # Stage 2: train LoRA + to_moments + decoder (with VF loss)
            in_stage1 = args.stage1_steps > 0 and step < args.stage1_steps
            
            # Transition from Stage 1 to Stage 2: enable LoRA training
            if not in_stage1 and not lora_enabled_in_training and args.lora_rank > 0:
                lora_enabled_in_training = True
                stage1_just_ended = True
                # Get the unwrapped encoder backbone (after DDP/FSDP wrapping)
                model_inner_for_lora = unwrap_model(model)
                encoder_backbone_inner = model_inner_for_lora.encoder.get_backbone()
                # Unfreeze LoRA parameters
                for m in encoder_backbone_inner.modules():
                    if hasattr(m, "lora_down") and hasattr(m, "lora_up"):
                        for p in m.lora_down.parameters():
                            p.requires_grad = True
                        for p in m.lora_up.parameters():
                            p.requires_grad = True
                # Update optimizer to include LoRA params
                lora_params = [p for n, p in encoder_backbone_inner.named_parameters()
                               if p.requires_grad and (("lora_up" in n) or ("lora_down" in n))]
                current_lora_params = lora_params  # Update for VF loss computation
                opt.add_param_group({'params': lora_params, 'lr': args.lr})
                
                # Recreate scheduler to match new number of param groups
                # LambdaLR needs to be recreated when param_groups change
                if scheduler is not None:
                    # Save current scheduler step before recreating
                    current_scheduler_step = scheduler.last_epoch + 1 if hasattr(scheduler, 'last_epoch') else step
                    # Recreate scheduler with updated optimizer (now has 2 param groups)
                    min_lr_ratio = args.min_lr / args.lr if args.lr > 0 else 0.1
                    scheduler = get_cosine_schedule_with_warmup(
                        optimizer=opt,
                        warmup_steps=args.warmup_steps,
                        total_steps=args.max_steps,
                        min_lr_ratio=min_lr_ratio,
                    )
                    # Restore scheduler to correct step by setting last_epoch
                    # LambdaLR's last_epoch is -1 initially, so we set it to current_step - 1
                    scheduler.last_epoch = current_scheduler_step - 1
                    # Manually update learning rates to match the step
                    scheduler.step()
                    if rank == 0:
                        logger.info(f"Recreated scheduler at step {current_scheduler_step} to match {len(opt.param_groups)} param groups")
                
                # 同步 EMA 模型（如果启用）
                if args.use_ema and ema_model is not None:
                    # 在 Stage 转换时，确保 EMA 模型与当前模型同步
                    model_inner_for_ema = unwrap_model(model)
                    ema_model.load_state_dict(model_inner_for_ema.state_dict())
                    if rank == 0:
                        logger.info("Synchronized EMA model with current model at Stage transition")
                
                if rank == 0:
                    num_lora_params = sum(p.numel() for p in lora_params)
                    logger.info(f"=== Stage 1 -> Stage 2 Transition at step {step} ===")
                    logger.info(f"Enabled LoRA training: {num_lora_params / 1e6:.2f}M params added")
                    logger.info(f"VF loss now active (weight={args.vf_weight})")
                dist.barrier()

            # Determine if GAN is active this step
            gan_active = use_gan and step >= args.gan_start_step

            # Determine training mode:
            # stage_disc: only train discriminator, generator frozen
            # stage_joint: train both together
            in_stage_disc = gan_active and (step - args.gan_start_step) < args.stage_disc_steps
            in_stage_joint = gan_active and not in_stage_disc

            # ========== Generator (VAE) Forward & Update ==========
            # VAE is always trained with recon/KL/VF loss
            # - stage_disc: train VAE (recon + KL + VF), train discriminator, but NO GAN loss for VAE
            # - stage_joint: train both VAE and discriminator, VAE includes GAN loss

            # Initialize loss values for logging
            loss = torch.tensor(0.0, device=device)
            loss_rec = torch.tensor(0.0, device=device)
            l1_loss = torch.tensor(0.0, device=device)
            lpips_loss = torch.tensor(0.0, device=device)
            kl_loss = torch.tensor(0.0, device=device)
            vf_raw = torch.tensor(0.0, device=device)
            w_adapt = torch.tensor(1.0, device=device)
            loss_vf = torch.tensor(0.0, device=device)
            gan_loss = torch.tensor(0.0, device=device)
            d_weight = torch.tensor(0.0, device=device)

            # Freeze discriminator during VAE update (to avoid computing unnecessary gradients)
            if gan_active and discriminator is not None:
                discriminator.eval()
                requires_grad(discriminator, False)

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_bf16):
                out = model(x)
                recon = out.sample  # [-1,1]
                posterior = out.posterior

                # Recon loss (keep in [-1,1] for LPIPS)
                l1_loss = F.l1_loss(recon, x)
                lpips_loss = loss_lpips(x, recon)
                loss_rec = args.l1_weight * l1_loss + args.lpips_weight * lpips_loss

                # KL
                if posterior is not None:
                    kl_loss = posterior.kl().mean()
                else:
                    kl_loss = torch.tensor(0.0, device=device)

                # VF loss: student vs ref
                # - 如果使用 LoRA: student (LoRA on) vs ref (LoRA off，使用当前模型的 use_lora=False)
                # - 如果 no-lora: student (当前模型) vs ref (ref_model)
                # Stage 1: skip VF loss (only recon + GAN)
                # Stage 2: enable VF loss
                model_inner = unwrap_model(model)
                
                if in_stage1 or args.vf_weight == 0.0:
                    # Stage 1 or VF disabled: no VF loss
                    vf_raw = torch.tensor(0.0, device=device)
                    w_adapt = torch.tensor(0.0, device=device)
                    loss_vf = torch.tensor(0.0, device=device)
                else:
                    # Stage 2: compute VF loss
                    if args.lora_rank > 0:
                        # 使用 LoRA: student (LoRA on) vs ref (LoRA off)
                        z_feat = model_inner.encode_features(x, use_lora=True)  # [B,C,S,S]
                        with torch.no_grad():
                            f_feat = model_inner.encode_features(x, use_lora=False)
                    else:
                        # no-lora: student (当前模型) vs ref (ref_model)
                        z_feat = model_inner.encode_features(x, use_lora=True)  # [B,C,S,S]
                        with torch.no_grad():
                            if ref_model is not None:
                                f_feat = ref_model.encode_features(x, use_lora=True)  # ref_model 没有 LoRA，但接口一致
                            else:
                                # fallback: 如果 ref_model 不存在，使用当前模型的 use_lora=False
                                f_feat = model_inner.encode_features(x, use_lora=False)
                    
                    vf_raw = model_inner.compute_vf_loss(z_feat, f_feat)

                    # 对于 adaptive weight，使用当前模型的可训练参数
                    if args.lora_rank > 0:
                        vf_params = current_lora_params
                    else:
                        # no-lora 时，使用 decoder 参数
                        vf_params = list(model_inner.decoder.parameters())
                    
                    if model_inner.vf_use_adaptive_weight:
                        w_adapt = model_inner.adaptive_weight(loss_rec, vf_raw, vf_params)
                    else:
                        w_adapt = torch.tensor(1.0, device=device)
                    w_adapt = torch.clamp(w_adapt, max=3.0)
                    loss_vf = model_inner.vf_hyper * w_adapt * vf_raw
                
                loss = loss_rec + model_inner.kl_weight * kl_loss + loss_vf

                # GAN Generator loss (only in stage_joint, NOT in stage_disc)
                if in_stage_joint:
                    # Apply DiffAug to reconstruction for generator training
                    recon_aug = diffaug.aug(recon) if diffaug is not None else recon
                    logits_fake = discriminator(recon_aug)
                    gan_loss = vanilla_g_loss(logits_fake)
                    # Compute adaptive weight using entire decoder parameters
                    d_weight = calculate_adaptive_weight(
                        loss_rec + model_inner.kl_weight * kl_loss + loss_vf,
                        gan_loss,
                        model_inner.decoder,  # Pass entire decoder module
                        args.max_d_weight,
                    )
                    loss = loss + args.gan_weight * d_weight * gan_loss

            loss.backward()
            opt.step()
            
            # Update VAE learning rate scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Update EMA model
            if args.use_ema and ema_model is not None:
                update_ema(ema_model, model, decay=args.ema_decay, use_fsdp=use_fsdp)

            # ========== Discriminator Update ==========
            disc_loss = torch.tensor(0.0, device=device)
            logits_real_mean = torch.tensor(0.0, device=device)
            logits_fake_mean = torch.tensor(0.0, device=device)

            if gan_active and discriminator is not None:
                discriminator.train()
                requires_grad(discriminator, True)
                opt_disc.zero_grad(set_to_none=True)

                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_bf16):
                    with torch.no_grad():
                        recon_detach = recon.detach()
                        # Discretize for discriminator input
                        recon_detach = recon_detach.clamp(-1.0, 1.0)
                        recon_detach = torch.round((recon_detach + 1.0) * 127.5) / 127.5 - 1.0
                        # Apply DiffAug to both real and fake images
                        x_aug = diffaug.aug(x) if diffaug is not None else x
                        recon_aug = diffaug.aug(recon_detach) if diffaug is not None else recon_detach

                    logits_fake_d = discriminator(recon_aug)
                    logits_real_d = discriminator(x_aug)
                    disc_loss = hinge_d_loss(logits_real_d, logits_fake_d)
                    logits_real_mean = logits_real_d.detach().mean()
                    logits_fake_mean = logits_fake_d.detach().mean()

                disc_loss.backward()
                opt_disc.step()
                
                # Update discriminator learning rate scheduler
                if scheduler_disc is not None:
                    scheduler_disc.step()
            # print(f"disc_loss: {disc_loss.item()}")
            step += 1

            if step % args.log_every == 0 and rank == 0:
                # Determine stage string
                if in_stage1:
                    stage_str = "stage1"
                elif in_stage_disc:
                    stage_str = "stage_disc"
                elif in_stage_joint:
                    stage_str = "stage_joint"
                else:
                    stage_str = "stage2" if args.stage1_steps > 0 else "no_gan"

                # Get current learning rates
                current_lr = opt.param_groups[0]['lr']
                current_disc_lr = opt_disc.param_groups[0]['lr'] if opt_disc is not None else 0.0

                log_msg = (
                    f"Step {step} [{stage_str}] lr={current_lr:.2e}: total={loss.item():.4f} | rec={loss_rec.item():.4f} "
                    f"(L1={l1_loss.item():.4f}, LPIPS={lpips_loss.item():.4f}) "
                    f"| KL={kl_loss.item():.4f} "
                    f"| VF={vf_raw.item():.4f} w={w_adapt.item():.3f} vf_term={loss_vf.item():.4f}"
                )
                if gan_active:
                    log_msg += (
                        f" | GAN={gan_loss.item():.4f} d_w={d_weight.item():.3f} "
                        f"D_loss={disc_loss.item():.4f} D_real={logits_real_mean.item():.3f} D_fake={logits_fake_mean.item():.3f}"
                        f" disc_lr={current_disc_lr:.2e}"
                    )
                logger.info(log_msg)

                # TensorBoard logging
                if tb_writer is not None:
                    # Total loss
                    tb_writer.add_scalar("Loss/total", loss.item(), step)
                    
                    # Reconstruction losses
                    tb_writer.add_scalar("Loss/rec", loss_rec.item(), step)
                    tb_writer.add_scalar("Loss/L1", l1_loss.item(), step)
                    tb_writer.add_scalar("Loss/LPIPS", lpips_loss.item(), step)
                    
                    # KL loss
                    tb_writer.add_scalar("Loss/KL", kl_loss.item(), step)
                    
                    # VF loss
                    tb_writer.add_scalar("Loss/VF_raw", vf_raw.item(), step)
                    tb_writer.add_scalar("Loss/VF_term", loss_vf.item(), step)
                    tb_writer.add_scalar("Weight/VF_adaptive", w_adapt.item(), step)
                    
                    # Stage indicator (0=stage1, 1=stage2)
                    tb_writer.add_scalar("Training/stage", 0 if in_stage1 else 1, step)
                    tb_writer.add_scalar("Training/lora_enabled", 0 if in_stage1 else 1, step)
                    
                    # GAN losses (when active)
                    if gan_active:
                        tb_writer.add_scalar("Loss/GAN_gen", gan_loss.item(), step)
                        tb_writer.add_scalar("Loss/GAN_disc", disc_loss.item(), step)
                        tb_writer.add_scalar("Weight/GAN_d_weight", d_weight.item(), step)
                        tb_writer.add_scalar("Discriminator/logits_real", logits_real_mean.item(), step)
                        tb_writer.add_scalar("Discriminator/logits_fake", logits_fake_mean.item(), step)

            if step % args.eval_every == 0:
                # 如果启用了 EMA 且设置了使用 EMA 进行验证，则使用 EMA 模型
                val_ema_model = ema_model if (args.use_ema and args.val_use_ema and ema_model is not None) else None
                run_validation(model, val_loader, device, step, args, logger, use_fsdp=use_fsdp, tb_writer=tb_writer, ema_model=val_ema_model)

            if step % args.save_every == 0 and rank == 0:
                save_path = os.path.join(args.output_dir, f"step_{step}.pth")
                model_state = get_model_state_dict(model)
                ckpt_state = {
                    "model": model_state, 
                    "step": step,
                    "opt": opt.state_dict(),  # 保存 VAE optimizer
                }
                # 保存 VAE scheduler
                if scheduler is not None:
                    ckpt_state["scheduler"] = scheduler.state_dict()
                # 保存 EMA 模型
                if args.use_ema and ema_model is not None:
                    ckpt_state["ema_model"] = ema_model.state_dict()
                # 保存 ref_model (no-lora 模式)
                if ref_model is not None:
                    ckpt_state["ref_model"] = ref_model.state_dict()
                if use_gan and discriminator is not None:
                    disc_state = get_model_state_dict(discriminator)
                    ckpt_state["discriminator"] = disc_state
                    if opt_disc is not None:
                        ckpt_state["opt_disc"] = opt_disc.state_dict()
                    if scheduler_disc is not None:
                        ckpt_state["scheduler_disc"] = scheduler_disc.state_dict()
                torch.save(ckpt_state, save_path)
                logger.info(f"Saved checkpoint to {save_path}")

            if step >= args.max_steps:
                break

        epoch += 1

    # Close TensorBoard writer
    if tb_writer is not None:
        tb_writer.close()

    cleanup_ddp()


if __name__ == "__main__":
    main()
