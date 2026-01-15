import os
import sys
import argparse
import math
import shutil
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

# 添加当前目录以导入 cnn_decoder
sys.path.append(".")

# === 导入依赖库 ===
try:
    import lpips
except ImportError:
    print("Please install lpips: pip install lpips")
    sys.exit(1)

try:
    from pytorch_fid import fid_score
    HAS_FID = True
except ImportError:
    HAS_FID = False
    print("Warning: pytorch-fid not found. rFID calculation will be skipped.")

# 尝试导入你的模型定义
try:
    from cnn_decoder import AutoencoderKL, CausalAutoencoderOutput
except ImportError:
    print("Error: cnn_decoder.py not found in current directory.")
    sys.exit(1)

# ==========================================
#              Helper Functions
# ==========================================

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        # 兼容单卡调试
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(0)
        return 0, 0, 1

def cleanup_ddp():
    dist.destroy_process_group()

def create_logger(save_dir, rank):
    logger = logging.getLogger(f"rank{rank}")
    logger.setLevel(logging.INFO)
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        # mode='a' 追加模式，防止意外覆盖日志
        fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='a')
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    else:
        logger.addHandler(logging.NullHandler())
    return logger

# === 数据处理逻辑 (严格对齐 inference) ===
def center_crop_arr(pil_image, image_size):
    """
    Validation 使用此 CenterCrop 逻辑
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

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

# ==========================================
#              Loss & Metrics
# ==========================================

class LPIPSLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # AlexNet backbone, input range [-1, 1]
        self.lpips = lpips.LPIPS(net='alex').to(device)
        self.lpips.eval()
        requires_grad(self.lpips, False)

    def forward(self, x, rec):
        # x, rec: [-1, 1]
        loss = self.lpips(x, rec)
        return loss.mean()

def calculate_psnr(img1, img2):
    """
    img1, img2: [-1, 1] tensors
    """
    img1 = (img1 + 1.0) / 2.0
    img2 = (img2 + 1.0) / 2.0
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# ==========================================
#              Evaluation Loop
# ==========================================

@torch.no_grad()
def run_validation(model, val_loader, device, step, args, logger):
    model.eval()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 临时目录
    gt_dir = os.path.join(args.output_dir, f"temp_eval_gt_{step}")
    recon_dir = os.path.join(args.output_dir, f"temp_eval_recon_{step}")
    
    if rank == 0:
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)
        logger.info(f"Running validation at step {step}...")
    
    dist.barrier()

    total_psnr = 0.0
    count = 0
    
    # 限制验证集数量
    max_batches = 50000 
    
    for i, (x, _) in enumerate(val_loader):
        if i >= max_batches: break
        
        x = x.to(device)
        
        # 直接调用 forward，这会经过 encode -> sample -> decode
        # DDP 自动处理 .module
        out = model(x)
        
        # 根据你提供的 CausalAutoencoderOutput 定义
        recon = out.sample 

        # 计算 PSNR (两者都是 [-1, 1])
        batch_psnr = calculate_psnr(x, recon)
        total_psnr += batch_psnr.item()
        count += 1
        
        # 保存图片用于 FID (转为 [0, 1])
        img_gt = torch.clamp((x + 1.0) / 2.0, 0, 1)
        img_recon = torch.clamp((recon + 1.0) / 2.0, 0, 1)
        
        for idx in range(x.size(0)):
            fname = f"r{rank}_b{i}_{idx}.png"
            save_image(img_gt[idx], os.path.join(gt_dir, fname))
            save_image(img_recon[idx], os.path.join(recon_dir, fname))
    
    # 聚合 PSNR
    avg_psnr = torch.tensor(total_psnr / max(count, 1), device=device)
    dist.all_reduce(avg_psnr, op=dist.ReduceOp.SUM)
    avg_psnr = avg_psnr.item() / world_size
    
    dist.barrier() 

    # 计算 FID (仅 Rank 0)
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
        
        # 记录到文件
        with open(os.path.join(args.output_dir, "metrics.txt"), "a") as f:
            f.write(f"Step: {step}, PSNR: {avg_psnr:.4f}, rFID: {rfid:.4f}\n")

        # 清理临时文件
        if not args.debug:
            shutil.rmtree(gt_dir)
            shutil.rmtree(recon_dir)

    dist.barrier()
    model.train() # 恢复训练模式

# ==========================================
#                 Main
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument("--train-path", type=str, required=True, help="ImageNet Train Dir")
    parser.add_argument("--val-path", type=str, default="/share/project/datasets/ImageNet/val", help="ImageNet Val Dir")
    parser.add_argument("--output-dir", type=str, default="results/cnn_decoder_finetune")
    parser.add_argument("--vae-ckpt", type=str, required=True, help="Pretrained VAE checkpoint")
    
    # 训练超参
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=100000)
    
    # Loss 权重
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--lpips-weight", type=float, default=0.5)
    
    # 频率
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    
    parser.add_argument("--debug", action="store_true", help="Keep temp images")

    args = parser.parse_args()

    # 1. DDP Init
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    logger = create_logger(args.output_dir, rank)
    if rank == 0:
        logger.info(f"Starting training: {args}")

    # 2. Dataset & DataLoader
    # Train: 使用 RandomResizedCrop
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),
    ])
    
    # Val: 严格对齐 Inference 脚本 (CenterCropArr)
    val_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),
    ])

    train_dataset = ImageFolder(args.train_path, transform=train_transform)
    val_dataset = ImageFolder(args.val_path, transform=val_transform)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=4, pin_memory=True, drop_last=False
    )

    # 3. Model Setup
    # 默认 Config
    model_config = {
        "in_channels": 3, "out_channels": 3,
        "enc_block_out_channels": [128, 256, 384, 512, 768],
        "dec_block_out_channels": [1280, 1024, 512, 256, 128],
        "enc_layers_per_block": 2, "dec_layers_per_block": 3,
        "latent_channels": 1280, "spatial_downsample_factor": 16,
        "use_quant_conv": False, "use_post_quant_conv": False,
        # 关键修改：训练时设置为 False，这样输出就是 raw logits ([-1, 1]范围)，
        # 方便与输入 x 计算 L1/LPIPS Loss
        "denormalize_decoder_output": True,
        # 关键修改：设置为 'dec' 模式，模型初始化时会自动冻结 encoder
        "running_mode": "dec"
    }
    
    model = AutoencoderKL(**model_config).to(device)
    
    # Load Checkpoint
    if rank == 0: logger.info(f"Loading weights from {args.vae_ckpt}")
    ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=False)

    # === Freeze Encoder (Double Check) ===
    # AutoencoderKL 已经在 init 里通过 running_mode='dec' 冻结了 dino
    # 这里我们再次确保所有 encoder 相关参数不更新
    requires_grad(model.encoder, False)
    if hasattr(model, 'quant_conv') and model.quant_conv: 
        requires_grad(model.quant_conv, False)
    
    # 确保 Decoder 可训
    requires_grad(model.decoder, True)
    if hasattr(model, 'post_quant_conv') and model.post_quant_conv: 
        requires_grad(model.post_quant_conv, True)

    # 切换模式
    model.eval() # 整体 eval (影响 dropout/bn)
    model.decoder.train() # 只开 decoder train
    
    # DDP Wrapper
    # find_unused_parameters=True 因为我们冻结了 encoder 的部分参数
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # 4. Optimizer & Loss
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,betas=(0.5,0.9))
    loss_lpips = LPIPSLoss(device)

    # 5. Training Loop
    step = 0
    epoch = 0
    
    if rank == 0: logger.info("Start Training Loop...")

    while step < args.max_steps:
        train_sampler.set_epoch(epoch)
        
        for x, _ in train_loader:
            x = x.to(device)
            
            optimizer.zero_grad()
            
            # Forward: 这里的 out 是 CausalAutoencoderOutput
            # 这里的 x 是 [-1, 1]，由于 denormalize_decoder_output=False，
            # out.sample 也是 [-1, 1] 空间，可以直接算 Loss
            out = model(x)
            recon = out.sample
            x = (x + 1.0) / 2.0
            recon = (recon + 1.0) / 2.0

            # Loss Calculation
            l1_loss = F.l1_loss(recon, x)
            p_loss = loss_lpips(x, recon)
            
            loss = args.l1_weight * l1_loss + args.lpips_weight * p_loss
            
            loss.backward()
            optimizer.step()
            
            step += 1
            
            # Log
            if step % args.log_every == 0 and rank == 0:
                logger.info(f"Step {step}: Total={loss.item():.4f} | L1={l1_loss.item():.4f} | LPIPS={p_loss.item():.4f}")
            
            # Evaluation
            if step % args.eval_every == 0:
                run_validation(model, val_loader, device, step, args, logger)
            
            # Save Checkpoint
            if step % args.save_every == 0 and rank == 0:
                save_path = os.path.join(args.output_dir, f"step_{step}.pth")
                torch.save(model.module.state_dict(), save_path)
                logger.info(f"Saved checkpoint to {save_path}")
            
            if step >= args.max_steps:
                break
        
        epoch += 1

    cleanup_ddp()

if __name__ == "__main__":
    main()