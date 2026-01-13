import os
import sys
import argparse
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

# 设置 torch 缓存路径 (可选)
os.environ["TORCH_HOME"] = "/share/project/huangxu/.cache/torch"

# 添加当前目录以导入项目模块
sys.path.append(".")

# 尝试导入 pytorch-fid
try:
    from pytorch_fid import fid_score
    HAS_FID = True
except ImportError:
    HAS_FID = False
    
# ==========================================
#              Helper Functions
# ==========================================

def setup_ddp():
    if "LOCAL_RANK" not in os.environ:
        # Fallback for single GPU run
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def center_crop_arr(pil_image, image_size):
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

# SAE 特有的归一化
def normalize_sae(tensor):
    ema_shift_factor = 0.0019670347683131695
    ema_scale_factor = 0.247765451669693
    return (tensor - ema_shift_factor) / ema_scale_factor

def denormalize_sae(tensor):
    ema_shift_factor = 0.0019670347683131695
    ema_scale_factor = 0.247765451669693
    return tensor * ema_scale_factor + ema_shift_factor

def calculate_batch_psnr(img1, img2):
    """
    img1, img2: Tensor [-1, 1]
    Return: (sum_psnr, count)
    """
    img1 = (img1 + 1.0) / 2.0
    img2 = (img2 + 1.0) / 2.0
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.sum().item(), psnr.shape[0]

# ==========================================
#              Model Loading
# ==========================================

def load_vae(model_type, vae_checkpoint_path, device):
    # 动态导入
    if model_type == "CNN":
        if dist.get_rank() == 0: print(">> Mode: CNN. Importing from cnn_decoder...")
        try:
            from cnn_decoder import AutoencoderKL
        except ImportError:
            print("Error: cnn_decoder.py not found.")
            sys.exit(1)
    else:
        if dist.get_rank() == 0: print(">> Mode: DIFFUSION. Importing from sae_model...")
        try:
            from sae_model import AutoencoderKL
        except ImportError:
            print("Error: sae_model.py not found.")
            sys.exit(1)

    # 这里的参数根据你的模型具体情况调整
    # 如果 CNN 和 SAE 参数不一致，可以用 if model_type == ... 来区分
    model_params = {
        "in_channels": 3,
        "out_channels": 3,
        "enc_block_out_channels": [128, 256, 384, 512, 768],
        "dec_block_out_channels": [1280, 1024, 512, 256, 128],
        "enc_layers_per_block": 2,
        "dec_layers_per_block": 3,
        "latent_channels": 1280,
        "spatial_downsample_factor": 16,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
        "denormalize_decoder_output": True, # 注意：确认 cnn_decoder 是否支持此参数
    }
    
    # SAE 可能需要额外的参数
    if model_type == "DIFFUSION":
        model_params.update({
            "variational": False,
            "running_mode": "dec",
        })

    vae = AutoencoderKL(**model_params).to(device)
    
    if dist.get_rank() == 0:
        print(f"Loading Checkpoint: {vae_checkpoint_path}")
    
    checkpoint = torch.load(vae_checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    vae.load_state_dict(state_dict, strict=False)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
        
    return vae

# ==========================================
#           Reconstruction Strategies
# ==========================================

@torch.no_grad()
def reconstruct_with_cnn(vae, z_raw):
    # CNN Decoder: 直接 decode
    # 注意：某些 VAE 实现返回 (recon, posterior) 或者只是 recon
    out = vae.decode(z_raw)
    if isinstance(out, tuple):
        return out[0]
    return out

@torch.no_grad()
def reconstruct_with_diffusion(vae, z_raw, image_shape, diffusion_steps):
    device = z_raw.device
    
    # SAE Process: Normalize -> (Optional Post Quant) -> Diffusion
    z = normalize_sae(z_raw) 
    latent_in = denormalize_sae(z)

    if getattr(vae, "post_quant_conv", None) is not None:
        latent_in = vae.post_quant_conv(latent_in)

    # 准备 Context
    context = vae.diffusion_decoder.get_context(latent_in)
    corrected_context = list(reversed(context[:]))

    diffusion = vae.diffusion_decoder.diffusion
    diffusion.set_sample_schedule(diffusion_steps, device)

    init_noise = torch.randn(image_shape, device=device, dtype=z.dtype)

    recon = diffusion.p_sample_loop(
        vae=vae,
        shape=image_shape,
        context=corrected_context,
        clip_denoised=True,
        init_noise=init_noise,
        eta=0.0, 
    )
    return recon

# ==========================================
#                   Main
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to ImageNet validation set.")
    parser.add_argument("--vae-ckpt", type=str, required=True, help="Path to VAE checkpoint.")
    parser.add_argument("--type", type=str, default="CNN", choices=["CNN", "DIFFUSION"], help="Reconstruction type.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--diffusion-steps", type=int, default=8, help="Steps for diffusion decoder (only used if type=DIFFUSION).")
    parser.add_argument("--max-images", type=int, default=None, help="Limit total images (across all GPUs).")
    parser.add_argument("--output-dir", type=str, default="results/test_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-images", action="store_true", help="Always keep images.")
    args = parser.parse_args()

    # 1. 初始化 DDP
    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # 2. 准备目录
    gt_dir = os.path.join(args.output_dir, "gt_temp")
    recon_dir = os.path.join(args.output_dir, "recon_temp")
    
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)
        print(f"==========================================")
        print(f" Mode: {args.type}")
        print(f" GPUs: {world_size}")
        print(f" Ckpt: {args.vae_ckpt}")
        print(f"==========================================")
    dist.barrier()

    # 3. 加载模型
    vae = load_vae(args.type, args.vae_ckpt, device)

    # 4. 数据集与 Sampler
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0), 
    ])
    
    # 简单的容错: 如果路径不存在，rank 0 报错
    if not os.path.exists(args.data_path):
        if rank == 0: print(f"Error: Data path {args.data_path} not found.")
        sys.exit(1)

    dataset = ImageFolder(root=args.data_path, transform=transform)

    if args.max_images is not None:
        if rank == 0: print(f"Limiting dataset to {args.max_images} images.")
        indices = list(range(min(len(dataset), args.max_images)))
        dataset = Subset(dataset, indices)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,
        num_workers=4, 
        pin_memory=True
    )

    # 5. 推理循环
    local_psnr_sum = 0.0
    local_count = 0
    
    if rank == 0:
        print("Starting Inference...")
        iterator = tqdm(loader, desc="Processing")
    else:
        iterator = loader

    for i, (x_img, _) in enumerate(iterator):
        x_img = x_img.to(device)
        B = x_img.shape[0]

        with torch.no_grad():
            # Encoder 部分通常是一样的
            z_raw, _ = vae.encode(x_img) 

            # Decoder 部分根据类型分支
            if args.type == "CNN":
                x_recon = reconstruct_with_cnn(vae, z_raw)
            else:
                x_recon = reconstruct_with_diffusion(
                    vae, z_raw, x_img.shape, diffusion_steps=args.diffusion_steps
                )

        # 统计 PSNR
        batch_psnr_sum, batch_n = calculate_batch_psnr(x_img, x_recon)
        local_psnr_sum += batch_psnr_sum
        local_count += batch_n

        # 保存图片用于 FID
        img_gt_save = torch.clamp((x_img + 1.0) / 2.0, 0, 1)
        img_recon_save = torch.clamp((x_recon + 1.0) / 2.0, 0, 1)
        
        for idx in range(B):
            file_name = f"rank{rank}_b{i}_{idx}.png"
            save_image(img_gt_save[idx], os.path.join(gt_dir, file_name))
            save_image(img_recon_save[idx], os.path.join(recon_dir, file_name))

    dist.barrier()

    # 6. 聚合结果
    stats_tensor = torch.tensor([local_psnr_sum, local_count], device=device, dtype=torch.float64)
    dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
    
    global_psnr_sum = stats_tensor[0].item()
    global_count = stats_tensor[1].item()
    global_avg_psnr = global_psnr_sum / max(global_count, 1.0)

    # 7. 计算 Metrics (Rank 0)
    if rank == 0:
        print("\n" + "="*40)
        print(" Calculating Metrics...")
        print("="*40)
        
        rfid_value = -1.0
        if HAS_FID:
            if not os.path.exists(gt_dir) or not os.path.exists(recon_dir):
                print("Warning: Image directories empty, skipping FID.")
            else:
                print("Computing FID...")
                try:
                    rfid_value = fid_score.calculate_fid_given_paths(
                        paths=[gt_dir, recon_dir],
                        batch_size=50,
                        device=device,
                        dims=2048,
                        num_workers=8
                    )
                except Exception as e:
                    print(f"FID Failed: {e}")
        
        print("\n" + "#"*40)
        print(f" FINAL RESULTS ({args.type})")
        print(f" Total Images     : {int(global_count)}")
        if args.type == "DIFFUSION":
            print(f" Diffusion Steps  : {args.diffusion_steps}")
        print(f" PSNR             : {global_avg_psnr:.4f} dB")
        if HAS_FID:
            print(f" rFID             : {rfid_value:.4f}")
        else:
            print(f" rFID             : N/A")
        print("#"*40 + "\n")

        # Save Text
        with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
            f.write(f"Type: {args.type}\n")
            f.write(f"PSNR: {global_avg_psnr:.4f}\n")
            f.write(f"rFID: {rfid_value}\n")
            f.write(f"Count: {global_count}\n")
        
        # Cleanup
        if not args.save_images:
            print("Cleaning up temporary images...")
            if os.path.exists(gt_dir): shutil.rmtree(gt_dir)
            if os.path.exists(recon_dir): shutil.rmtree(recon_dir)
    
    dist.barrier()
    cleanup_ddp()

if __name__ == "__main__":
    main()