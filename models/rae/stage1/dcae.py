import torch
import torch.nn as nn
from math import sqrt
from diffusers import AutoencoderDC


class DCAE(nn.Module):
    def __init__(
        self,
        ckpt: str = "mit-han-lab/dc-ae-f64c128-in-1.0-diffusers", 
        input_size: int = 512,
        reshape_to_2d: bool = True,
    ):
        super().__init__()
        self.ae = AutoencoderDC.from_pretrained(
            ckpt, torch_dtype=torch.float32
        ).eval()

        self.input_size = input_size
        self.reshape_to_2d = reshape_to_2d
        # self.downsample = 64  # f64 → spatial compression ratio
        self.latent_dim = self.ae.config.latent_channels 

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size:
            x = nn.functional.interpolate(
                x, size=(self.input_size, self.input_size),
                mode="bicubic", align_corners=False
            )
        x = 2 * x - 1  # → [-1,1]
        posterior = self.ae.encode(x)
        z = posterior.latent
        if not self.reshape_to_2d:
            b, c, h, w = z.shape
            z = z.view(b, c, h * w).permute(0, 2, 1)
        return z  # [B,128,H/64,W/64]

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 3:
            b, n, c = z.shape
            h = w = int(sqrt(n))
            z = z.permute(0, 2, 1).reshape(b, c, h, w)
        x_rec = self.ae.decode(z).sample
        x_rec = (x_rec + 1) / 2  # → [0,1]
        return x_rec

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
