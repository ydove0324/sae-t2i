import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
import sys
import os
import argparse

# 设置 torch cache (保留你的设置)
os.environ["TORCH_HOME"] = "/share/project/huangxu/.cache/torch"

# 添加当前目录
sys.path.append(".")

# ================= 参数解析 =================
parser = argparse.ArgumentParser(description="VAE/SAE Inference Script")
parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--input", type=str, required=True, help="Path to input image")
parser.add_argument("--output", type=str, default="test/recon.png", help="Path to save output image")
parser.add_argument("--type", type=str, default="CNN", choices=["CNN", "DIFFUSION"], 
                    help="Reconstruction type: 'CNN' (Standard VAE) or 'DIFFUSION' (SAE)")
parser.add_argument("--steps", type=int, default=8, help="Diffusion steps (only for SAE)")
args = parser.parse_args()

# ================= 动态导入 =================
# 根据参数决定导入哪个 AutoencoderKL 类
try:
    if args.type == "CNN":
        print(">> Mode: CNN. Importing from cnn_decoder...")
        from cnn_decoder import AutoencoderKL
    else:
        print(">> Mode: DIFFUSION. Importing from sae_model...")
        from sae_model import AutoencoderKL
except ImportError as e:
    print(f"Error: Could not import necessary modules. {e}")
    print(f"Please ensure 'cnn_decoder.py' or 'sae_model.py' exists in {os.getcwd()}")
    sys.exit(1)

# ================= 全局配置 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256

# VAE 模型结构参数 (需确保两套代码都兼容此配置，或者根据 args.type 调整)
MODEL_CONFIG = {
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
    # "denormalize_decoder_output": True, # 注意：有些 CNN VAE 实现可能不支持此参数，如果报错请注释掉
}

# SAE 特有的归一化参数
EMA_SHIFT = 0.0019670347683131695
EMA_SCALE = 0.247765451669693

# ================= 核心函数 =================

def load_vae(ckpt_path):
    print(f"Loading model from {ckpt_path}...")
    # 这里的 AutoencoderKL 指向的是上面动态导入的那个类
    model = AutoencoderKL(**MODEL_CONFIG).to(DEVICE)
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # 加载权重 (strict=False 允许忽略多余或缺失的键，例如 loss 相关的参数)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def preprocess_image(img_path):
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        sys.exit(1)
        
    img = Image.open(img_path).convert("RGB")
    # Center Crop & Resize
    short_side = min(img.size)
    img = TF.center_crop(img, short_side)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    # ToTensor & Normalize [-1, 1]
    x = TF.to_tensor(img) * 2.0 - 1.0
    return x.unsqueeze(0).to(DEVICE)

# --- Strategy 1: CNN Reconstruction ---
@torch.no_grad()
def cnn_reconstruct(vae, x):
    z_raw, _ = vae.encode(x)
    
    # 假设 cnn_decoder 的 decode 返回 (sample, posterior)
    # 如果代码不同，请根据实际情况修改，例如: recon = vae.decode(z_raw)
    out = vae.decode(z_raw)
    
    if isinstance(out, tuple):
        x_recon = out[0]
    else:
        x_recon = out
        
    return x_recon

# --- Strategy 2: Diffusion Reconstruction (SAE) ---
@torch.no_grad()
def diffusion_reconstruct(vae, x, steps=8):
    # 1. Encode
    z_raw, _ = vae.encode(x)
    
    # 2. Latent Normalization (SAE specific)
    z = (z_raw - EMA_SHIFT) / EMA_SCALE
    latent_in = z * EMA_SCALE + EMA_SHIFT 
    
    if getattr(vae, "post_quant_conv", None):
        latent_in = vae.post_quant_conv(latent_in)

    # 3. Diffusion Decode
    diffusion = vae.diffusion_decoder.diffusion
    diffusion.set_sample_schedule(steps, DEVICE)
    
    context = vae.diffusion_decoder.get_context(latent_in)
    # 翻转 context
    corrected_context = list(reversed(context[:]))
    
    init_noise = torch.randn(x.shape, device=DEVICE)
    
    recon = diffusion.p_sample_loop(
        vae=vae,
        shape=x.shape,
        context=corrected_context,
        clip_denoised=True,
        init_noise=init_noise,
        eta=0.0
    )
    return recon

# ================= 主程序 =================

if __name__ == "__main__":
    print(f"Running with Type: {args.type} | Device: {DEVICE}")

    # 1. 加载模型
    vae = load_vae(args.ckpt)

    # 2. 处理图片
    x = preprocess_image(args.input)

    # 3. 执行推理
    print("Running reconstruction...")
    if args.type == "CNN":
        x_recon = cnn_reconstruct(vae, x)
    else:
        x_recon = diffusion_reconstruct(vae, x, steps=args.steps)

    # 4. 保存结果
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # [-1, 1] -> [0, 1]
    save_image((x_recon + 1) / 2, args.output)
    print(f"Done! Saved to {args.output}")