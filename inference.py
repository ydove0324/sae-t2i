# inference_stage2_to_image.py
import os
import math
import argparse
from copy import deepcopy

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

# === 你工程里的依赖（按你训练脚本一致）===
from sae_model import AutoencoderKL
from models.rae.utils.model_utils import instantiate_from_config
from omegaconf import OmegaConf


# ---------------------------
# normalize / denormalize (和训练一致)
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
# 加载 DINOv3 VAE (带 diffusion decoder) - 与训练一致
# ---------------------------
def load_dinov3_vae(vae_checkpoint_path: str, device: torch.device) -> AutoencoderKL:
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

    vae = AutoencoderKL(**model_params).to(device)
    ckpt = torch.load(vae_checkpoint_path, map_location="cpu")

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    print(f"[VAE] loaded. missing={len(missing)} unexpected={len(unexpected)}")

    vae.eval()
    requires_grad(vae, False)
    return vae


# ---------------------------
# latent -> image: 用 VAE diffusion decoder 完整采样回图
# ---------------------------
@torch.no_grad()
def reconstruct_from_latent_with_diffusion(
    vae: AutoencoderKL,
    latent_z: torch.Tensor,
    image_shape: torch.Size,
    diffusion_steps: int = 25,
) -> torch.Tensor:
    device = latent_z.device

    # latent 空间是 normalize 后的，推理前要 denormalize 回去
    z = denormalize_sae(latent_z)

    # 如果有 post_quant_conv，保持与 forward 一致
    if getattr(vae, "post_quant_conv", None) is not None and vae.post_quant_conv is not None:
        z = vae.post_quant_conv(z)

    # diffusion decoder 多尺度 context
    context = vae.diffusion_decoder.get_context(z)
    corrected_context = list(reversed(context[:]))  # 和你训练里一致

    diffusion = vae.diffusion_decoder.diffusion
    diffusion.set_sample_schedule(diffusion_steps, device)

    # init_noise shape: (B, C, H, W)
    init_noise = torch.randn(image_shape, device=device, dtype=latent_z.dtype)

    recon = diffusion.p_sample_loop(
        vae=vae,
        shape=image_shape,
        context=corrected_context,
        clip_denoised=True,
        init_noise=init_noise,
        eta=0.0,  # DDIM deterministic
    )
    return recon  # [-1,1]


# ---------------------------
# Stage2 latent 采样：与你训练参数化一致（flow-shift + x-pred）
# ---------------------------
@torch.no_grad()
def sample_latent_linear_steps(
    model,
    batch_size: int,
    latent_shape: tuple,   # (C,H,W)
    device: torch.device,
    y: torch.Tensor,       # [B]
    time_shift: float,
    steps: int = 50,
    init_noise: torch.Tensor | None = None,  # 允许外部指定 noise
) -> torch.Tensor:
    model.eval()

    C, H, W = latent_shape
    if init_noise is None:
        # 按照 batch_size 生成随机噪声
        x = torch.randn((batch_size, C, H, W), device=device, dtype=torch.float32)
    else:
        x = init_noise.to(device=device, dtype=torch.float32)

    shift = float(time_shift)

    def flow_shift(t_lin: torch.Tensor) -> torch.Tensor:
        t = (shift * t_lin) / (1.0 + (shift - 1.0) * t_lin)
        return t.clamp(0.0, 1.0 - 1e-6)

    # deterministic schedule: t_lin 从 1 -> 0
    for i in range(steps, 0, -1):
        # 扩展维度以匹配 batch_size
        t_lin = torch.full((batch_size,), i / steps, device=device, dtype=torch.float32)
        t_next_lin = torch.full((batch_size,), (i - 1) / steps, device=device, dtype=torch.float32)

        t = flow_shift(t_lin)
        t_next = flow_shift(t_next_lin)

        # x-pred: predict x0
        x0_hat = model(x, t, y=y)

        # eps_hat from x_t = t*eps + (1-t)*x0
        t4 = t.view(batch_size, 1, 1, 1)
        eps_hat = (x - (1.0 - t4) * x0_hat) / (t4 + 1e-8)

        # update
        t_next4 = t_next.view(batch_size, 1, 1, 1)
        x = t_next4 * eps_hat + (1.0 - t_next4) * x0_hat

    return x  # z0_hat


# ---------------------------
# 从 config 加载 Stage2 + checkpoint
# ---------------------------
def load_stage2_from_config_and_ckpt(config_path: str, ckpt_path: str, device: torch.device, use_ema: bool = True):
    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.get("stage_2", None)
    if model_cfg is None:
        raise ValueError("Config does not contain `stage_2` section. Please adapt to your config layout.")

    model = instantiate_from_config(model_cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if use_ema and "ema" in ckpt:
        print("[Stage2] loading EMA weights from checkpoint")
        model.load_state_dict(ckpt["ema"], strict=True)
    elif "model" in ckpt:
        print("[Stage2] loading model weights from checkpoint")
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        try:
            model.load_state_dict(ckpt, strict=True)
            print("[Stage2] loading weights directly from dict")
        except:
            raise ValueError("Checkpoint format not recognized (no `ema` or `model` keys).")

    model.eval()
    requires_grad(model, False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Stage2 config yaml (must contain `stage_2` section).")
    parser.add_argument("--ckpt", type=str, required=True, help="Stage2 checkpoint path (xxx.pt).")
    parser.add_argument("--vae-ckpt", type=str, required=True, help="DINOv3 VAE checkpoint path (xxx.pth).")

    # === 修改处 1: Label 改为 str 以支持 'all' ===
    parser.add_argument("--label", type=str, default="207", help="ImageNet class label [0..999] or 'all'.")
    
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--stage2-steps", type=int, default=50)
    parser.add_argument("--vae-diffusion-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of images to generate per class.")
    parser.add_argument("--out", type=str, default="results", help="Output directory root.")
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights in ckpt if available.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # === 确定要生成的类别列表 ===
    if args.label.lower() == "all":
        target_labels = list(range(1000))
        print(f"Mode: Generating ALL classes (0-999), batch_size per class = {args.batch_size}")
    else:
        target_labels = [int(args.label)]
        print(f"Mode: Generating single class {args.label}, batch_size = {args.batch_size}")

    # === 计算 time_shift ===
    shift_dim = 16 * 16 * 1280
    shift_base = 4096
    time_shift = math.sqrt(shift_dim / shift_base)
    print(f"time_shift = {time_shift}")

    # 1) load models (只加载一次)
    vae = load_dinov3_vae(args.vae_ckpt, device=device)
    stage2 = load_stage2_from_config_and_ckpt(args.config, args.ckpt, device=device, use_ema=args.use_ema)

    # 2) 遍历类别进行生成
    total_classes = len(target_labels)
    
    for i, cls_id in enumerate(target_labels):
        print(f"[{i+1}/{total_classes}] Generating class {cls_id} ...")

        # 构造 label tensor
        y = torch.full((args.batch_size,), cls_id, device=device, dtype=torch.long)

        # 3) sample latent
        z0_hat = sample_latent_linear_steps(
            model=stage2,
            batch_size=args.batch_size,
            latent_shape=(1280, 16, 16),
            device=device,
            y=y,
            time_shift=time_shift,
            steps=args.stage2_steps,
            init_noise=None, 
        )

        # 4) latent -> image via diffusion decoder
        imgs = reconstruct_from_latent_with_diffusion(
            vae=vae,
            latent_z=z0_hat,
            image_shape=torch.Size([args.batch_size, 3, args.image_size, args.image_size]),
            diffusion_steps=args.vae_diffusion_steps,
        )  # shape [B, 3, H, W]

        # 5) save images
        imgs_01 = (imgs + 1.0) / 2.0
        
        # 目录结构：args.out/gen_{cls_id}/
        save_dir = os.path.join(args.out, f"gen_{cls_id}")
        os.makedirs(save_dir, exist_ok=True)

        for idx in range(args.batch_size):
            file_path = os.path.join(save_dir, f"{idx}.png")
            save_image(imgs_01[idx], file_path)
        
        # 可选：手动清理显存（如果显存非常紧张）
        # torch.cuda.empty_cache()

    print(f"Done. Results saved in {args.out}")


if __name__ == "__main__":
    main()