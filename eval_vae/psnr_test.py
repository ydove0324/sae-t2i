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
from torch.utils.data import Dataset

# 设置 torch 缓存路径 (可选)

# 添加当前目录以导入项目模块
sys.path.append(".")

# 导入 VAE 工具函数
from models.rae.utils.vae_utils import (
    load_vae,
    normalize_sae,
    denormalize_sae,
    reconstruct_from_latent_with_diffusion,
)

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
#                   Main
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data-path", type=str, required=True, help="Path to ImageNet validation set.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit total images (across all GPUs).")
    
    # VAE model
    parser.add_argument("--vae-ckpt", type=str, required=True, help="Path to VAE checkpoint.")
    parser.add_argument("--encoder-type", type=str, default="dinov3", choices=["dinov3", "dinov3_vitl", "siglip2", "dinov2"], help="Encoder type.")
    parser.add_argument("--dinov3-dir", type=str, default="/cpfs01/huangxu/models/dinov3", help="Path to DINOv3 model directory.")
    parser.add_argument("--siglip2-model-name", type=str, default="google/siglip2-base-patch16-256", help="SigLIP2 model name.")
    parser.add_argument("--dinov2-model-name", type=str, default="facebook/dinov2-with-registers-base", help="DINOv2 model name.")
    parser.add_argument("--lora-rank", type=int, default=256, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=256, help="LoRA alpha.")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA (set lora_rank=0).")
    parser.add_argument("--skip-to-moments", action="store_true", help="Skip to_moments layer (for old checkpoints).")
    parser.add_argument("--denormalize-output", action="store_true", help="Denormalize decoder output.")
    parser.add_argument("--patch-size", type=int, default=None, 
                        help="Patch size (auto-detected from encoder if not set). "
                             "For old checkpoints trained with hardcoded patch_size=16, use --patch-size 16")
    
    # Decoder type
    parser.add_argument("--decoder-type", type=str, default="cnn_decoder", choices=["cnn_decoder", "vit_decoder"], help="Decoder architecture type.")
    
    # ViT decoder params (only used if --decoder-type=vit_decoder)
    parser.add_argument("--vit-hidden-size", type=int, default=1024, help="ViT decoder hidden size (XL: 1024, L: 768, B: 512).")
    parser.add_argument("--vit-num-layers", type=int, default=24, help="ViT decoder num layers (XL: 24, L: 16, B: 8).")
    parser.add_argument("--vit-num-heads", type=int, default=16, help="ViT decoder num heads (XL: 16, L: 12, B: 8).")
    parser.add_argument("--vit-intermediate-size", type=int, default=4096, help="ViT decoder intermediate size (XL: 4096, L: 3072, B: 2048).")
    
    # Reconstruction
    parser.add_argument("--type", type=str, default="CNN", choices=["CNN", "DIFFUSION"], help="Reconstruction type (for diffusion decoder).")
    parser.add_argument("--diffusion-steps", type=int, default=8, help="Steps for diffusion decoder (only used if type=DIFFUSION).")
    
    # Output
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
        print(f" Encoder: {args.encoder_type}")
        print(f" Decoder: {args.decoder_type}")
        if args.decoder_type == "vit_decoder":
            print(f"   ViT hidden_size: {args.vit_hidden_size}")
            print(f"   ViT num_layers: {args.vit_num_layers}")
            print(f"   ViT num_heads: {args.vit_num_heads}")
        print(f" GPUs: {world_size}")
        print(f" Ckpt: {args.vae_ckpt}")
        print(f" LoRA: {'Disabled' if args.no_lora else f'rank={args.lora_rank}, alpha={args.lora_alpha}'}")
        print(f" Skip to_moments: {args.skip_to_moments}")
        print(f" Denormalize output: {args.denormalize_output}")
        print(f"==========================================")
    dist.barrier()

    # 3. 加载模型
    # 确定 decoder_type：优先使用 --decoder-type，但如果是 DIFFUSION 模式则用 diffusion_decoder
    if args.type == "DIFFUSION":
        decoder_type = "diffusion_decoder"
    else:
        decoder_type = args.decoder_type  # "cnn_decoder" or "vit_decoder"
    
    # 根据 encoder_type 设置 latent_channels, dec_block_out_channels, patch_size
    if args.encoder_type == "dinov3":
        latent_channels = 1280
        dec_block_out_channels = (1280, 1024, 512, 256, 128)
        default_patch_size = 16
    elif args.encoder_type == "dinov3_vitl":
        latent_channels = 1024
        dec_block_out_channels = (1024, 768, 512, 256, 128)
        default_patch_size = 16
    elif args.encoder_type == "siglip2":
        latent_channels = 768
        dec_block_out_channels = (768, 512, 256, 128, 64)
        default_patch_size = 16
    elif args.encoder_type == "dinov2":
        latent_channels = 768
        dec_block_out_channels = (768, 512, 256, 128, 64)
        # Note: DINOv2 encoder uses patch_size=14, but old training code used patch_size=16 for decoder
        # If loading old checkpoint trained with hardcoded patch_size=16, use --patch-size 16
        default_patch_size = 14  # Default for new training, but old checkpoints may need 16
    else:
        raise ValueError(f"Unknown encoder_type: {args.encoder_type}")
    
    # Use user-specified patch_size if provided, otherwise use default
    patch_size = args.patch_size if args.patch_size is not None else default_patch_size
    
    # 如果 --no-lora，则设置 lora_rank=0
    lora_rank = 0 if args.no_lora else args.lora_rank
    lora_alpha = 0 if args.no_lora else args.lora_alpha
    
    # 构建 model_params (基础参数)
    vae_model_params = {
        "encoder_type": args.encoder_type,
        "image_size": args.image_size,
        "patch_size": patch_size,
        "out_channels": 3,
        "latent_channels": latent_channels,
        "target_latent_channels": None,
        "spatial_downsample_factor": patch_size,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "decoder_dropout": 0.0,
        "gradient_checkpointing": False,
        "denormalize_decoder_output": args.denormalize_output,
    }
    
    # 根据 decoder_type 添加特定参数
    if decoder_type == "cnn_decoder":
        vae_model_params["dec_block_out_channels"] = dec_block_out_channels
        vae_model_params["dec_layers_per_block"] = 3
    elif decoder_type == "vit_decoder":
        # ViT decoder 参数 (参数名需要匹配 cnn_decoder.AutoencoderKL)
        vae_model_params["vit_decoder_hidden_size"] = args.vit_hidden_size
        vae_model_params["vit_decoder_num_layers"] = args.vit_num_layers
        vae_model_params["vit_decoder_num_heads"] = args.vit_num_heads
        vae_model_params["vit_decoder_intermediate_size"] = args.vit_intermediate_size
    
    # 根据 encoder_type 添加模型路径参数
    if args.encoder_type in ["dinov3", "dinov3_vitl"]:
        vae_model_params["dinov3_model_dir"] = args.dinov3_dir
    elif args.encoder_type == "siglip2":
        vae_model_params["siglip2_model_name"] = args.siglip2_model_name
    elif args.encoder_type == "dinov2":
        vae_model_params["dinov2_model_name"] = args.dinov2_model_name
    
    vae = load_vae(
        args.vae_ckpt,
        device,
        encoder_type=args.encoder_type,
        decoder_type=decoder_type,
        model_params=vae_model_params,
        verbose=(rank == 0),
        skip_to_moments=args.skip_to_moments,
    )

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
    class ImageDataset(Dataset):
        EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
        def __init__(self, root, transform=None):
            self.transform = transform
            self.paths = sorted([
                os.path.join(root, f)
                for f in os.listdir(root)
                if os.path.splitext(f)[1].lower() in self.EXTS
            ])
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            try:
                img = Image.open(self.paths[idx]).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, 0
            except Exception as e:
                print(f"Failed to load image {self.paths[idx]}: {e}")
                # 返回一个全零图像，保证训练不崩
                dummy_img = torch.zeros(3, *([self.transform.transforms[0].function.keywords['image_size']]*2))
                return dummy_img, 0

    try:
        dataset = ImageDataset(root=args.data_path, transform=transform)
    except Exception as e:
        if rank == 0:
            print(f"Error: Failed to create dataset from {args.data_path}: {e}")
        sys.exit(1)

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
            # Encoder
            z_raw, _ = vae.encode(x_img,sample_posterior=False)
            # Normalize for latent space
            z_normalized = normalize_sae(z_raw)

            # Decoder (using unified reconstruct function)
            x_recon = reconstruct_from_latent_with_diffusion(
                vae=vae,
                latent_z=z_normalized,
                image_shape=x_img.shape,
                diffusion_steps=args.diffusion_steps,
                decoder_type=decoder_type,
                encoder_type=args.encoder_type,
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
        print(f" Encoder Type     : {args.encoder_type}")
        print(f" Decoder Type     : {decoder_type}")
        print(f" Total Images     : {int(global_count)}")
        if args.type == "DIFFUSION":
            print(f" Diffusion Steps  : {args.diffusion_steps}")
        if decoder_type == "vit_decoder":
            print(f" ViT Config       : hidden={args.vit_hidden_size}, layers={args.vit_num_layers}")
        print(f" LoRA             : {'Disabled' if args.no_lora else f'rank={args.lora_rank}'}")
        print(f" PSNR             : {global_avg_psnr:.4f} dB")
        if HAS_FID:
            print(f" rFID             : {rfid_value:.4f}")
        else:
            print(f" rFID             : N/A")
        print("#"*40 + "\n")

        # Save Text
        with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
            f.write(f"Type: {args.type}\n")
            f.write(f"Encoder: {args.encoder_type}\n")
            f.write(f"Decoder: {decoder_type}\n")
            if decoder_type == "vit_decoder":
                f.write(f"ViT Config: hidden={args.vit_hidden_size}, layers={args.vit_num_layers}, heads={args.vit_num_heads}\n")
            f.write(f"LoRA: {'Disabled' if args.no_lora else f'rank={args.lora_rank}, alpha={args.lora_alpha}'}\n")
            f.write(f"Skip to_moments: {args.skip_to_moments}\n")
            f.write(f"Denormalize output: {args.denormalize_output}\n")
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