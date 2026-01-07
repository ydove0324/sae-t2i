import torch
import torch.nn as nn
from math import sqrt
from diffusers import AutoencoderKL


class VAE(nn.Module):
    def __init__(
        self,
        ckpt: str = "stabilityai/sd-vae-ft-mse",
        input_size: int = 256,
        reshape_to_2d: bool = True,
    ):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(ckpt)
        self.vae.eval()

        self.input_size = input_size
        self.reshape_to_2d = reshape_to_2d
        self.downsample = 8  # SD-VAE latent stride
        self.latent_dim = 4

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Expect [0,1] input
        if x.shape[-1] != self.input_size:
            x = nn.functional.interpolate(
                x, size=(self.input_size, self.input_size),
                mode="bicubic", align_corners=False
            )
        x = 2 * x - 1  # to [-1,1]
        posterior = self.vae.encode(x)
        z = posterior.latent_dist.mode()  # deterministic encoding
        if not self.reshape_to_2d:
            b, c, h, w = z.shape
            z = z.view(b, c, h * w).permute(0, 2, 1)
        return z  # [B,4,H/8,W/8]

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 3:
            b, n, c = z.shape
            h = w = int(sqrt(n))
            z = z.permute(0, 2, 1).reshape(b, c, h, w)
        x_rec = self.vae.decode(z).sample
        x_rec = (x_rec + 1) / 2  # to [0,1]
        return x_rec

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
