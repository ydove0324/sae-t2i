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
def load_dinov3_vae(
    vae_checkpoint_path: str,
    device: torch.device,
    encoder_type: str = "dinov3",
    decoder_type: str = "cnn_decoder",
    dinov3_dir: str = "/cpfs01/huangxu/models/dinov3",
    siglip2_model_name: str = "google/siglip2-base-patch16-256",
    dinov2_model_name: str = "facebook/dinov2-with-registers-base",
    image_size: int = 256,
    patch_size: int = None,
    latent_channels: int = None,
    spatial_downsample_factor: int = None,
    dec_block_out_channels: tuple = None,
    vit_hidden_size: int = 1024,
    vit_num_layers: int = 24,
    vit_num_heads: int = 16,
    vit_intermediate_size: int = 4096,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    skip_to_moments: bool = False,
    denormalize_decoder_output: bool = True,
):
    """
    加载 VAE 模型，支持多种 encoder 和 decoder 类型
    
    Args:
        vae_checkpoint_path: VAE checkpoint 路径
        device: 设备
        encoder_type: "dinov3", "siglip2", "dinov2"
        decoder_type: "cnn_decoder", "vit_decoder", "diffusion_decoder"
        dinov3_dir: DINOv3 模型目录
        siglip2_model_name: SigLIP2 模型名称
        dinov2_model_name: DINOv2 模型名称
        image_size: 图像大小
        patch_size: Patch 大小（如果为 None，会根据 encoder_type 自动设置）
        latent_channels: Latent channels（如果为 None，会根据 encoder_type 自动设置）
        spatial_downsample_factor: 空间下采样因子（如果为 None，会等于 patch_size）
        dec_block_out_channels: CNN decoder 的 block out channels（tuple）
        vit_hidden_size: ViT decoder hidden size
        vit_num_layers: ViT decoder 层数
        vit_num_heads: ViT decoder 头数
        vit_intermediate_size: ViT decoder intermediate size
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        skip_to_moments: 是否跳过加载 to_moments 层
        denormalize_decoder_output: 是否反归一化 decoder 输出
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # 根据 decoder_type 选择模块
    if decoder_type in ["cnn_decoder", "vit_decoder"]:
        if cnn_decoder is None:
            raise ImportError("cnn_decoder module not found.")
        AutoencoderKL = cnn_decoder.AutoencoderKL
        if rank == 0:
            print(f"[Info] Loading VAE with {decoder_type.upper()}...")
    elif decoder_type == "diffusion_decoder":
        if sae_model is None:
            raise ImportError("sae_model module not found.")
        AutoencoderKL = sae_model.AutoencoderKL
        if rank == 0:
            print("[Info] Loading VAE with Diffusion Decoder...")
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}. Supported: 'cnn_decoder', 'vit_decoder', 'diffusion_decoder'")

    # 根据 encoder_type 设置默认值
    if latent_channels is None:
        if encoder_type == "dinov3":
            latent_channels = 1280
        elif encoder_type == "siglip2":
            latent_channels = 768
        elif encoder_type == "dinov2":
            latent_channels = 768
        else:
            latent_channels = 1280
    
    if patch_size is None:
        if encoder_type == "dinov2":
            patch_size = 14
        else:
            patch_size = 16
    
    if spatial_downsample_factor is None:
        spatial_downsample_factor = patch_size
    
    if dec_block_out_channels is None:
        if encoder_type == "dinov2":
            dec_block_out_channels = (768, 512, 256, 128, 64)
        else:
            dec_block_out_channels = (1280, 1024, 512, 256, 128)
    
    # 构建模型参数
    if decoder_type in ["cnn_decoder", "vit_decoder"]:
        # 使用 cnn_decoder.AutoencoderKL
        model_params = {
            "encoder_type": encoder_type,
            "dinov3_model_dir": dinov3_dir,
            "siglip2_model_name": siglip2_model_name,
            "dinov2_model_name": dinov2_model_name,
            "image_size": image_size,
            "patch_size": patch_size,
            "out_channels": 3,
            "latent_channels": latent_channels,
            "target_latent_channels": None,
            "spatial_downsample_factor": spatial_downsample_factor,
            "decoder_type": decoder_type,
            "dec_block_out_channels": dec_block_out_channels,
            "dec_layers_per_block": 3,
            "decoder_dropout": 0.0,
            "gradient_checkpointing": False,
            "vit_decoder_hidden_size": vit_hidden_size,
            "vit_decoder_num_layers": vit_num_layers,
            "vit_decoder_num_heads": vit_num_heads,
            "vit_decoder_intermediate_size": vit_intermediate_size,
            "variational": True,
            "kl_weight": 1e-6,
            "noise_tau": 0.0,
            "random_masking_channel_ratio": 0.0,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "enable_lora": (lora_rank > 0),
            "vf_margin_cos": 0.1,
            "vf_margin_dms": 0.3,
            "vf_max_tokens": 32,
            "vf_hyper": 1.0,
            "vf_use_adaptive_weight": True,
            "vf_weight_clamp": 1e4,
            "training_mode": "enc_dec",
            "denormalize_decoder_output": denormalize_decoder_output,
            "skip_to_moments": skip_to_moments,
        }
    else:
        # 使用 sae_model.AutoencoderKL (diffusion_decoder)
        model_params = {
            "in_channels": 3,
            "out_channels": 3,
            "enc_block_out_channels": [128, 256, 384, 512, 768],
            "dec_block_out_channels": list(dec_block_out_channels),
            "enc_layers_per_block": 2,
            "dec_layers_per_block": 3,
            "latent_channels": latent_channels,
            "use_quant_conv": False,
            "use_post_quant_conv": False,
            "spatial_downsample_factor": spatial_downsample_factor,
            "variational": False,
            "noise_tau": 0.0,
            "denormalize_decoder_output": denormalize_decoder_output,
            "running_mode": "dec",
            "random_masking_channel_ratio": 0.0,
            "target_latent_channels": None,
            "lpips_weight": 0.1,
        }
    
    # 实例化模型
    vae = AutoencoderKL(**model_params).to(device)
    
    # 加载 checkpoint
    ckpt = torch.load(vae_checkpoint_path, map_location="cpu")
    
    # 处理不同的 checkpoint 格式
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    
    # 如果 skip_to_moments=True，跳过 to_moments 相关的 key
    if skip_to_moments:
        state_dict = {k: v for k, v in state_dict.items() if "to_moments" not in k}
        if rank == 0:
            print("[Info] Skipping to_moments layer loading (skip_to_moments=True)")
    
    # 加载权重
    vae.load_state_dict(state_dict, strict=False)
    vae.eval()
    requires_grad(vae, False)
    
    if rank == 0:
        print(f"[Info] VAE loaded: encoder={encoder_type}, decoder={decoder_type}, "
              f"image_size={image_size}, patch_size={patch_size}, "
              f"latent_channels={latent_channels}, spatial_downsample={spatial_downsample_factor}")
    
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
    
    Args:
        vae: VAE 模型
        latent_z: Latent 特征
        image_shape: 输出图像形状
        diffusion_steps: Diffusion 步数（仅用于 diffusion_decoder）
        decoder_type: "cnn_decoder", "vit_decoder", 或 "diffusion_decoder"
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
        # === 使用 CNN Decoder 或 ViT Decoder 的逻辑 ===
        # cnn_decoder.AutoencoderKL 的 decode 方法会根据 decoder_type 自动选择
        # CNN 或 ViT decoder，所以这里统一使用 vae.decode(z).sample
        recon = vae.decode(z).sample

    return recon


@torch.no_grad()
def sample_latent_dopri5(
    model,
    batch_size,
    latent_shape,
    device,
    y,
    time_shift,
    # 这些是 dopri5 的精度参数
    rtol=1e-5,
    atol=1e-5,
    # 防止 t=0 奇点
    eps_time=1e-5,
    # 可选：限制最大内部步数，防止发疯
    max_num_steps=1000,
):
    model.eval()
    C, H, W = latent_shape
    x0 = torch.randn((batch_size, C, H, W), device=device, dtype=torch.float32)

    shift = float(time_shift)

    # 你的 time shift 映射
    def flow_shift(t_lin: torch.Tensor) -> torch.Tensor:
        t = (shift * t_lin) / (1.0 + (shift - 1.0) * t_lin)
        return t

    # 直接在“真实 t”上积分，且 t ∈ [eps, 1-eps]
    t_start = flow_shift(torch.tensor(1.0, device=device))
    t_end   = flow_shift(torch.tensor(0.0, device=device))

    # clamp 避免 t 真到 0 或 1
    t_start = torch.clamp(t_start, min=eps_time, max=1.0 - eps_time)
    t_end   = torch.clamp(t_end,   min=eps_time, max=1.0 - eps_time)

    # 从噪声端 -> 干净端（反向）
    t_span = torch.stack([t_start, t_end]).to(torch.float32)

    def ode_rhs(t_scalar, x):
        # torchdiffeq 会传一个标量 t
        t = torch.full((batch_size,), t_scalar, device=device, dtype=torch.float32)
        # 再次 clamp，双保险
        t = t.clamp(min=eps_time, max=1.0 - eps_time)

        x0_hat = model(x, t, y=y)

        t4 = t.view(batch_size, 1, 1, 1)
        eps_hat = (x - (1.0 - t4) * x0_hat) / (t4 + 1e-8)

        # 对应 x(t) = t*eps + (1-t)*x0  => dx/dt = eps - x0
        return eps_hat - x0_hat

    x_traj = odeint(
        ode_rhs,
        x0,
        t_span,
        method="dopri5",
        rtol=rtol,
        atol=atol,
        options={"max_num_steps": max_num_steps},
    )

    return x_traj[-1]



@torch.no_grad()
def sample_latent_linear_steps(model, batch_size, latent_shape, device, y, time_shift, steps=50):
    """
    Sample latent z0 using the SAME training parameterization.
    Aligned with sample_latent_linear_50_steps in train.py
    """
    # unwrap DDP if needed
    model_inner = model.module if hasattr(model, "module") else model
    model_inner.eval()
    
    C, H, W = latent_shape
    x = torch.randn((batch_size, C, H, W), device=device, dtype=torch.float32)
    shift = float(time_shift)
    
    def flow_shift(t_lin: torch.Tensor) -> torch.Tensor:
        # SD3-style flow shift
        t = (shift * t_lin) / (1.0 + (shift - 1.0) * t_lin)
        # 和训练里保持一致：避免 t=1 / t=0 的数值问题
        return t.clamp(0.0, 1.0 - 1e-6)
    
    # 线性 schedule in t_lin：从 1 -> 0
    for i in range(steps, 0, -1):
        t_lin = torch.full((batch_size,), i / steps, device=device, dtype=torch.float32)
        t_next_lin = torch.full((batch_size,), (i - 1) / steps, device=device, dtype=torch.float32)
        
        t = flow_shift(t_lin)           # [B]
        t_next = flow_shift(t_next_lin) # [B]
        
        # Stage2 forward 用的是 t:[B]，不是 [B,1]
        # 保持与原始代码一致的 autocast 使用（即使 enabled=False）
        with torch.autocast(device_type='cuda', enabled=False, dtype=torch.float32):
            x0_hat = model_inner(x, t, y=y)  # x-pred: predict x0 in latent space
        
        # eps_hat from x_t = t*eps + (1-t)*x0
        t_scalar = t.view(batch_size, 1, 1, 1)
        eps_hat = (x - (1.0 - t_scalar) * x0_hat) / (t_scalar + 1e-8)
        
        # update to next time
        t_next_scalar = t_next.view(batch_size, 1, 1, 1)
        with torch.autocast(device_type='cuda', enabled=False, dtype=torch.float32):
            x = t_next_scalar * eps_hat + (1.0 - t_next_scalar) * x0_hat
    
    # 最后一步 t_next=0，x 就是 z0_hat
    return x

@torch.no_grad()
def sample_latent_heun_steps(
    model,
    batch_size,
    latent_shape,
    device,
    y,
    time_shift,
    steps=50,
    eps_time=1e-5,   # 避免 t=0 奇点
):
    """
    Heun (RK2) sampler on ODE: dx/dt = eps_hat(x,t) - x0_hat(x,t)
    Time runs from 1 -> 0 (noise -> clean). We discretize on t_lin then map by flow_shift.
    """
    model.eval()
    C, H, W = latent_shape
    x = torch.randn((batch_size, C, H, W), device=device, dtype=torch.float32)

    shift = float(time_shift)

    def flow_shift(t_lin: torch.Tensor) -> torch.Tensor:
        t = (shift * t_lin) / (1.0 + (shift - 1.0) * t_lin)
        return t.clamp(eps_time, 1.0 - eps_time)

    def drift(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        compute f(x,t) = eps_hat - x0_hat, where x0_hat = model(x,t)
        """
        x0_hat = model(x, t, y=y)
        t4 = t.view(batch_size, 1, 1, 1)
        eps_hat = (x - (1.0 - t4) * x0_hat) / (t4 + 1e-8)
        return eps_hat - x0_hat

    # integrate from t_lin = 1 -> 0
    for i in range(steps, 0, -1):
        t_lin = torch.full((batch_size,), i / steps, device=device, dtype=torch.float32)
        t_next_lin = torch.full((batch_size,), (i - 1) / steps, device=device, dtype=torch.float32)

        t = flow_shift(t_lin)
        t_next = flow_shift(t_next_lin)

        # dt is negative (since t decreases)
        dt = (t_next - t).view(batch_size, 1, 1, 1)  # shape [B,1,1,1]

        # Heun:
        K1 = drift(x, t)

        # predictor: x_p = x + dt*K1
        x_p = x + dt * K1


        K2 = drift(x_p, t_next)

        # corrector: x_{n+1} = x + dt*(K1+K2)/2
        x = x + dt * 0.5 * (K1 + K2)

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
    
    # Encoder 配置
    parser.add_argument("--encoder-type", type=str, default="dinov3", choices=["dinov3", "siglip2", "dinov2"],
                        help="Encoder type: 'dinov3', 'siglip2', or 'dinov2'")
    parser.add_argument("--dinov3-dir", type=str, default="/cpfs01/huangxu/models/dinov3",
                        help="Path to DINOv3 model directory")
    parser.add_argument("--siglip2-model-name", type=str, default="google/siglip2-base-patch16-256",
                        help="SigLIP2 model name from HuggingFace")
    parser.add_argument("--dinov2-model-name", type=str, default="facebook/dinov2-with-registers-base",
                        help="DINOv2 model name from HuggingFace")
    
    # Decoder 类型选择
    parser.add_argument("--decoder-type", type=str, default="cnn_decoder", 
                        choices=["cnn_decoder", "vit_decoder", "diffusion_decoder"], 
                        help="Choose VAE decoder type: 'cnn_decoder', 'vit_decoder', or 'diffusion_decoder'")
    
    # VAE 模型配置
    parser.add_argument("--image-size", type=int, default=256, help="Image size")
    parser.add_argument("--patch-size", type=int, default=None, 
                        help="Patch size (auto-detected from encoder if not set)")
    parser.add_argument("--latent-channels", type=int, default=None,
                        help="Latent channels (auto-detected from encoder if not set)")
    parser.add_argument("--spatial-downsample-factor", type=int, default=None,
                        help="Spatial downsample factor (auto-detected from patch_size if not set)")
    
    # CNN Decoder 配置
    parser.add_argument("--dec-block-out-channels", type=str, default=None,
                        help="Decoder block output channels, comma-separated (e.g., '1280,1024,512,256,128')")
    
    # ViT Decoder 配置
    parser.add_argument("--vit-hidden-size", type=int, default=1024, help="ViT decoder hidden size")
    parser.add_argument("--vit-num-layers", type=int, default=24, help="ViT decoder number of layers")
    parser.add_argument("--vit-num-heads", type=int, default=16, help="ViT decoder number of heads")
    parser.add_argument("--vit-intermediate-size", type=int, default=4096, help="ViT decoder intermediate size")
    
    # LoRA 配置
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA")
    
    # VAE 选项
    parser.add_argument("--skip-to-moments", action="store_true",
                        help="Skip loading to_moments layer (for old checkpoints without it)")
    parser.add_argument("--denormalize-decoder-output", action="store_true", default=True,
                        help="Denormalize decoder output in VAE (default: True)")
    parser.add_argument("--no-denormalize-decoder-output", dest="denormalize_decoder_output", action="store_false",
                        help="Disable denormalize decoder output")

    # 生成参数
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stage2-steps", type=int, default=50)
    parser.add_argument("--vae-diffusion-steps", type=int, default=25, help="Steps for diffusion decoder (only used if decoder-type is diffusion_decoder)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-ema", action="store_true")
    
    args = parser.parse_args()
    
    # 处理 --no-lora
    if args.no_lora:
        args.lora_rank = 0
    
    # 解析 dec_block_out_channels
    if args.dec_block_out_channels is not None:
        args.dec_block_out_channels = tuple(int(x) for x in args.dec_block_out_channels.split(','))
    else:
        args.dec_block_out_channels = None

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
        print(f" Encoder Type: {args.encoder_type}")
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

    # 传递所有参数给 VAE 加载器
    vae = load_dinov3_vae(
        vae_checkpoint_path=args.vae_ckpt,
        device=device,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        dinov3_dir=args.dinov3_dir,
        siglip2_model_name=args.siglip2_model_name,
        dinov2_model_name=args.dinov2_model_name,
        image_size=args.image_size,
        patch_size=args.patch_size,
        latent_channels=args.latent_channels,
        spatial_downsample_factor=args.spatial_downsample_factor,
        dec_block_out_channels=args.dec_block_out_channels,
        vit_hidden_size=args.vit_hidden_size,
        vit_num_layers=args.vit_num_layers,
        vit_num_heads=args.vit_num_heads,
        vit_intermediate_size=args.vit_intermediate_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        skip_to_moments=args.skip_to_moments,
        denormalize_decoder_output=args.denormalize_decoder_output,
    )
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
            # z0_hat = sample_latent_heun_steps(
            #     stage2,
            #     curr_batch,
            #     (1280, 16, 16),
            #     device,
            #     y,
            #     time_shift,
            #     steps=args.stage2_steps,
            #     eps_time=1e-5,
            # )
            
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