# inference_ddp_fid.py
import os
import math
import argparse
import torch
import torch.distributed as dist
import numpy as np
from torchvision.utils import save_image
from omegaconf import OmegaConf
from tqdm import tqdm
import glob
import traceback
import shutil
import sys

# === 动态导入模型类，避免 import 冲突 ===
# 我们会在 load_dinov3_vae 函数内部根据参数决定使用哪个 AutoencoderKL
try:
    import cnn_decoder
except ImportError:
    cnn_decoder = None

try:
    import sae_model
except ImportError:
    sae_model = None

from models.rae.utils.model_utils import instantiate_from_config

# === FID 库 (需要 pip install pytorch-fid) ===
try:
    from pytorch_fid import fid_score
    HAS_FID = True
except ImportError:
    HAS_FID = False
    print("Warning: 'pytorch-fid' not found. FID calculation will be skipped.")

# ---------------------------
# Helper Functions
# ---------------------------
def normalize_sae(tensor: torch.Tensor) -> torch.Tensor:
    ema_shift_factor = 0.0019670347683131695
    ema_scale_factor = 0.247765451669693
    return (tensor - ema_shift_factor) / ema_scale_factor

def denormalize_sae(tensor: torch.Tensor) -> torch.Tensor:
    ema_shift_factor = 0.0019670347683131695
    ema_scale_factor = 0.247765451669693
    return tensor * ema_scale_factor + ema_shift_factor

def requires_grad(model, flag: bool = False):
    for p in model.parameters():
        p.requires_grad = flag

# ---------------------------
# FID Calculation Helper
# ---------------------------
def create_flat_temp_dir(source_root, temp_dir):
    """
    遍历 source_root 下所有子文件夹，将图片软链接到 temp_dir。
    如果文件名冲突，会自动重命名 (例如 classID_filename.png)。
    """
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    print(f"  [Flattening] Creating symbolic links in {temp_dir} ...")
    
    count = 0
    # 遍历 source_root
    for root, dirs, files in os.walk(source_root):
        # 跳过 temp_dir 自身，防止死循环 (如果 temp_dir 在 source_root 里面)
        if os.path.abspath(root) == os.path.abspath(temp_dir):
            continue

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 原文件绝对路径
                src_path = os.path.abspath(os.path.join(root, file))
                
                # 为了防止不同文件夹里有同名文件 (比如都是 0.png)
                # 我们用 "父文件夹名_文件名" 作为新名字
                parent_name = os.path.basename(root)
                new_name = f"{parent_name}_{file}"
                dst_path = os.path.join(temp_dir, new_name)
                
                # 创建软链接 (symlink)，速度极快，不占空间
                try:
                    os.symlink(src_path, dst_path)
                    count += 1
                except OSError:
                    # 如果系统不支持软链接，回退到复制 (Copy)
                    shutil.copy(src_path, dst_path)
                    count += 1
    
    print(f"  [Flattening] Linked {count} images.")
    return count

# ---------------------------
# FID Calculation (Updated)
# ---------------------------
def run_fid_calculation(args, device):
    """
    使用临时扁平文件夹计算 FID，计算完后自动删除临时文件夹。
    """
    if not HAS_FID:
        print("Error: pytorch-fid not installed.")
        return

    # 1. 定义临时文件夹路径
    # 放在 args.out 下面，名字叫 .temp_fid_flat
    flat_temp_dir = os.path.join(args.out, ".temp_fid_flat")

    print("\n" + "="*50)
    print(f"[Rank 0] Calculating FID...")
    print(f"  Source Subfolders : {args.out}")
    print(f"  Temp Flat Dir     : {flat_temp_dir}")
    print(f"  Reference Stats   : {args.ref_path}")
    
    try:
        # 2. 把图片全部“搬”到临时文件夹
        num_imgs = create_flat_temp_dir(args.out, flat_temp_dir)
        
        if num_imgs < 2:
            print("Error: Too few images found. Skipping FID.")
            return

        # 3. 调用原始的 fid_score 函数
        # 现在我们可以传 path string 了，因为图片都在这一层
        fid_value = fid_score.calculate_fid_given_paths(
            paths=[flat_temp_dir, args.ref_path],  # <--- 传临时文件夹路径
            batch_size=50,
            device=device,
            dims=2048,
            num_workers=8
        )
        
        print(f"\n>>>>> FID Score: {fid_value} <<<<<\n")
        
        # 保存结果
        with open(os.path.join(args.out, "fid_score.txt"), "w") as f:
            f.write(f"FID: {fid_value}\n")
            f.write(f"Ref: {args.ref_path}\n")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during FID calculation: {e}")
    
    finally:
        # 4. 清理现场：删除临时文件夹
        if os.path.exists(flat_temp_dir):
            print(f"  Cleaning up temp dir: {flat_temp_dir}")
            shutil.rmtree(flat_temp_dir)
            
    print("="*50)

# ---------------------------
# Model Loaders
# ---------------------------
def load_dinov3_vae(vae_checkpoint_path: str, device: torch.device, decoder_type: str = "cnn_decoder"):
    """
    根据 decoder_type 动态选择使用 cnn_decoder 还是 sae_model
    """
    if decoder_type == "cnn_decoder":
        if cnn_decoder is None:
            raise ImportError("cnn_decoder module not found.")
        AutoencoderKL = cnn_decoder.AutoencoderKL
        print("[Info] Loading VAE with CNN Decoder...")
    elif decoder_type == "diffusion_decoder":
        if sae_model is None:
            raise ImportError("sae_model module not found.")
        AutoencoderKL = sae_model.AutoencoderKL
        print("[Info] Loading VAE with Diffusion Decoder...")
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}")

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
        "running_mode": "dec",
        "random_masking_channel_ratio": 0.0,
        "target_latent_channels": None,
        "lpips_weight": 0.1,
    }
    
    # 实例化对应的类
    vae = AutoencoderKL(**model_params).to(device)
    
    ckpt = torch.load(vae_checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    
    # strict=False 允许加载时有一些 key 不匹配（例如 SAE 模型可能会多出 diffusion_decoder 相关的 key）
    vae.load_state_dict(state_dict, strict=False)
    vae.eval()
    requires_grad(vae, False)
    return vae

def load_stage2_from_config_and_ckpt(config_path: str, ckpt_path: str, device: torch.device, use_ema: bool = True):
    cfg = OmegaConf.load(config_path)
    model = instantiate_from_config(cfg.stage_2).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    if use_ema and "ema" in ckpt:
        if rank == 0: print(f"[Stage2] Loading EMA weights")
        model.load_state_dict(ckpt["ema"], strict=True)
    elif "model" in ckpt:
        if rank == 0: print(f"[Stage2] Loading standard weights")
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
        
    model.eval()
    requires_grad(model, False)
    return model

# ---------------------------
# Inference Logic
# ---------------------------
@torch.no_grad()
def reconstruct_from_latent_with_diffusion(vae, latent_z, image_shape, diffusion_steps=25, decoder_type="cnn_decoder"):
    """
    根据 decoder_type 切换解码逻辑
    """
    device = latent_z.device
    z = denormalize_sae(latent_z).to(device)

    if decoder_type == "diffusion_decoder":
        # === 使用 Diffusion Decoder 的逻辑 ===
        # 1. Post Quant Conv (如果模型里有)
        if getattr(vae, "post_quant_conv", None) is not None:
            z = vae.post_quant_conv(z)
        
        # 2. 获取 context
        # 注意：这里假设 sae_model.AutoencoderKL 有 diffusion_decoder 属性
        context = vae.diffusion_decoder.get_context(z)
        
        # 3. 翻转 Context (参考原注释)
        corrected_context = list(reversed(context[:]))
        
        # 4. 获取 Diffusion 对象并设置步数
        diffusion = vae.diffusion_decoder.diffusion
        diffusion.set_sample_schedule(diffusion_steps, device)
        
        # 5. 初始化噪声
        init_noise = torch.randn(image_shape, device=device, dtype=latent_z.dtype)
        
        # 6. 采样
        recon = diffusion.p_sample_loop(
            vae=vae, 
            shape=image_shape, 
            context=corrected_context, 
            clip_denoised=True, 
            init_noise=init_noise, 
            eta=0.0
        )
    else:
        # === 使用 CNN Decoder 的逻辑 (默认) ===
        recon = vae.decode(z).sample

    return recon

@torch.no_grad()
def sample_latent_linear_steps(model, batch_size, latent_shape, device, y, time_shift, steps=50):
    model.eval()
    C, H, W = latent_shape
    x = torch.randn((batch_size, C, H, W), device=device, dtype=torch.float32)
    shift = float(time_shift)
    def flow_shift(t_lin):
        t = (shift * t_lin) / (1.0 + (shift - 1.0) * t_lin)
        return t.clamp(0.0, 1.0 - 1e-6)
    
    for i in range(steps, 0, -1):
        t_lin = torch.full((batch_size,), i / steps, device=device, dtype=torch.float32)
        t_next_lin = torch.full((batch_size,), (i - 1) / steps, device=device, dtype=torch.float32)
        t = flow_shift(t_lin)
        t_next = flow_shift(t_next_lin)
        x0_hat = model(x, t, y=y)
        t4 = t.view(batch_size, 1, 1, 1)
        eps_hat = (x - (1.0 - t4) * x0_hat) / (t4 + 1e-8)
        t_next4 = t_next.view(batch_size, 1, 1, 1)
        x = t_next4 * eps_hat + (1.0 - t_next4) * x0_hat
    return x

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    # 必要路径
    parser.add_argument("--config", type=str, default="", help="Config for Stage 2")
    parser.add_argument("--ckpt", type=str, default="", help="Checkpoint for Stage 2")
    parser.add_argument("--vae-ckpt", type=str, default="", help="Checkpoint for VAE")
    parser.add_argument("--out", type=str, default="results/fid_test", help="Output root directory")
    
    # 功能开关
    parser.add_argument("--eval-only", action="store_true", help="Skip generation, only calculate FID using existing images in --out")
    parser.add_argument("--calc-fid", action="store_true", help="Calculate FID after generation (ignored if --eval-only is set, as it is implied)")
    parser.add_argument("--ref-path", type=str, default=None, help="Path to .npz stats or reference image folder")
    
    # Decoder 类型选择
    parser.add_argument("--decoder-type", type=str, default="cnn_decoder", choices=["cnn_decoder", "diffusion_decoder"], 
                        help="Choose VAE decoder type: 'cnn_decoder' (default) or 'diffusion_decoder'")

    # 生成参数
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stage2-steps", type=int, default=50)
    parser.add_argument("--vae-diffusion-steps", type=int, default=25, help="Steps for diffusion decoder (only used if decoder-type is diffusion_decoder)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-ema", action="store_true")
    
    args = parser.parse_args()

    # 1. Initialize Distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # 设置种子
    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # ==========================
    # Branch A: EVAL ONLY MODE
    # ==========================
    if args.eval_only:
        if rank == 0:
            print(f"[Info] Eval-only mode active.")
            print(f"[Info] Skipping model loading and generation.")
            # 直接计算 FID
            run_fid_calculation(args, device)
        
        # 阻断其他进程，防止它们做多余的事，然后退出
        dist.barrier()
        dist.destroy_process_group()
        return

    # ==========================
    # Branch B: GENERATION MODE
    # ==========================
    if rank == 0:
        print(f"==================================================")
        print(f" Start Distributed Inference: World Size = {world_size}")
        print(f" Target: 1000 classes x {args.samples_per_class} samples")
        print(f" Decoder Type: {args.decoder_type}")
        print(f" Output Dir: {args.out}")
        print(f"==================================================")
        os.makedirs(args.out, exist_ok=True)

    # 1. Load Models (仅在需要生成时加载)
    # 简单的参数检查
    if not args.config or not args.ckpt or not args.vae_ckpt:
        if rank == 0: print("Error: --config, --ckpt, and --vae-ckpt are required for generation.")
        dist.destroy_process_group()
        return

    # 传递 decoder_type 给 VAE 加载器
    vae = load_dinov3_vae(args.vae_ckpt, device, decoder_type=args.decoder_type)
    stage2 = load_stage2_from_config_and_ckpt(args.config, args.ckpt, device, args.use_ema)

    # 2. Task Distribution
    all_classes = list(range(1000))
    my_classes = all_classes[rank::world_size]
    time_shift = math.sqrt((16 * 16 * 1280) / 4096)

    # 3. Generation Loop
    iterator = tqdm(my_classes, desc=f"Rank {rank}", position=rank)
    
    for cls_id in iterator:
        total_needed = args.samples_per_class
        generated_count = 0
        
        class_dir = os.path.join(args.out, str(cls_id))
        os.makedirs(class_dir, exist_ok=True)

        # 检查是否已经生成过（可选优化：断点续传）
        # existing_files = len([name for name in os.listdir(class_dir) if name.endswith('.png')])
        # if existing_files >= total_needed: continue 

        while generated_count < total_needed:
            curr_batch = min(args.batch_size, total_needed - generated_count)
            y = torch.full((curr_batch,), cls_id, device=device, dtype=torch.long)
            
            # Stage 2 Sampling
            z0_hat = sample_latent_linear_steps(stage2, curr_batch, (1280, 16, 16), device, y, time_shift, args.stage2_steps)
            
            # VAE Decode (传递 decoder_type 参数)
            imgs = reconstruct_from_latent_with_diffusion(
                vae, 
                z0_hat, 
                torch.Size([curr_batch, 3, 256, 256]), 
                diffusion_steps=args.vae_diffusion_steps,
                decoder_type=args.decoder_type
            )
            
            # Save
            imgs = (imgs + 1.0) / 2.0
            imgs = imgs.clamp(0, 1)

            for i in range(curr_batch):
                img_idx = generated_count + i
                save_path = os.path.join(class_dir, f"{img_idx}.png")
                save_image(imgs[i], save_path)
            
            generated_count += curr_batch

    print(f"[Rank {rank}] Finished generation.")
    
    # 4. Synchronization
    dist.barrier() # 等待所有 GPU 完成生成和写入

    # 5. Calculate FID (仅 Rank 0)
    if rank == 0 and args.calc_fid:
        run_fid_calculation(args, device)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()