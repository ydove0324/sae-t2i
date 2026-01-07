import torch
import torch.nn as nn
from math import sqrt
from typing import Optional
from transformers import AutoImageProcessor
from models.dino_v3.image_vae_dinov3_encode import AutoencoderKL
import yaml


class SAE(nn.Module):
    def __init__(
        self,
        ckpt_path: str = "/opt/tiger/vfm/weights/dinov3/ema_vae.pth",
        encoder_input_size: int = 256,
        encoder_config_path: str = "/opt/tiger/vfm/models/dinov3-vith16plus-pretrain-lvd1689m",
        noise_tau: float = 0.0,
        reshape_to_2d: bool = True,
        normalization_stat_path: str = "/opt/tiger/vfm/weights/dinov3/config/main.yaml",
        config_path: str = "/opt/tiger/vfm/models/dino_v3/s16_c1280.yaml",
        eps: float = 1e-5,
    ):
        super().__init__()

        # ---- normalization (from DINOv3 processor) ----
        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)

        # ---- sizes ----
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = 16
        assert self.encoder_input_size % self.encoder_patch_size == 0
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2

        # ---- load Autoencoder ----
        # self.vae = AutoencoderKL(
        #     in_channels=3,
        #     out_channels=3,
        #     enc_block_out_channels=(128, 256, 384, 512, 768),
        #     dec_block_out_channels=(1280, 1024, 512, 256, 128),
        #     enc_layers_per_block=2,
        #     dec_layers_per_block=3,
        #     latent_channels=1280,
        #     use_quant_conv=False,
        #     use_post_quant_conv=False,
        #     spatial_downsample_factor=16,
        #     variational=False,
        #     running_mode="enc_dec",
        # )
        # 读取 yaml 配置
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        if "__object__" in cfg:
            del cfg["__object__"]

        self.vae = AutoencoderKL(**cfg)
        state = torch.load(ckpt_path, map_location="cpu")
        self.vae.load_state_dict(state, strict=False)
        self.vae.eval()

        # ---- misc ----
        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d
        self.eps = eps
        if normalization_stat_path is not None:
            with open(normalization_stat_path, "r") as f:
                cfg = yaml.safe_load(f)
            ema_shift_factor = cfg["ae"]["ema_shift_factor"] if "ae" in cfg else 0.0
            ema_scale_factor = cfg["ae"]["ema_scale_factor"] if "ae" in cfg else 1.0
            self.latent_mean = torch.as_tensor(ema_shift_factor)
            self.latent_var  = torch.as_tensor(ema_scale_factor**2)
            self.do_normalization = True
        else:
            self.do_normalization = False

        print(f"SAE: loaded DINOv3 VAE from {ckpt_path}")
    # ----------------------------------------------
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.encoder_input_size or x.shape[-2] != self.encoder_input_size:
            x = nn.functional.interpolate(
                x, size=(self.encoder_input_size, self.encoder_input_size),
                mode="bicubic", align_corners=False, antialias=True
            )
        # x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
        x = 2 * x - 1

        enc_out = self.vae.encode(x)
        h, p = enc_out

        if self.noise_tau > 0:
            sigma = self.noise_tau * torch.rand((h.size(0),) + (1,) * (h.ndim - 1), device=h.device)
            h = h + sigma * torch.randn_like(h)

        # latent 归一化
        if self.do_normalization:
            m = self.latent_mean.to(h.device) if getattr(self, "latent_mean", None) is not None else 0
            v = self.latent_var.to(h.device)  if getattr(self, "latent_var",  None) is not None else 1
            h = (h - m) / torch.sqrt(v + self.eps)

        return h  # [B, 1280, 16, 16]

    # ----------------------------------------------
    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:

        if self.do_normalization:
            m = self.latent_mean.to(z.device) if getattr(self, "latent_mean", None) is not None else 0
            v = self.latent_var.to(z.device)  if getattr(self, "latent_var",  None) is not None else 1
            z = z * torch.sqrt(v + self.eps) + m

        if z.ndim == 3:  # [B, N, C] → [B, C, H, W]
            b, n, c = z.shape
            h = w = int(sqrt(n))
            z = z.transpose(1, 2).reshape(b, c, h, w)

        x_rec = self.vae.decode(z).sample
        # x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
        x_rec = (x_rec + 1) / 2
        return x_rec

    # ----------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec
