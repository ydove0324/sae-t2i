from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch import nn
import math

from common.fs import download, exists
from models.dino_v3.modeling_dino_v3 import DINOv3ViTModel

import logging
import numpy as np
from functools import partial
from einops import rearrange

import torch.nn.functional as F

from enum import Enum
from typing import Literal, NamedTuple, Optional
import torch

import torch


class DiagonalGaussianDistribution:
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor):
        self.mean = mean
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self) -> torch.FloatTensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self) -> torch.Tensor:
        return 0.5 * torch.sum(
            self.mean**2 + self.var - 1.0 - self.logvar,
            dim=list(range(1, self.mean.ndim)),
        )


_receptive_field_t = Literal["half", "full"]
_inflation_mode_t = Literal["none", "tail", "replicate"]
_memory_device_t = Optional[Literal["cpu", "same"]]
_norm_type_t = Literal["BatchNorm", "GroupNorm", "SpectralNorm", "InstanceNorm"]


class MemoryState(Enum):
    """
    State[Disabled]:        No memory bank will be enabled.
    State[Initializing]:    The model is handling the first clip, need to reset the memory bank.
    State[Active]:          There has been some data in the memory bank.
    State[Unset]:           Error state, indicating users didn't pass correct memory state in.
    """

    DISABLED = 0
    INITIALIZING = 1
    ACTIVE = 2
    UNSET = 3


class CausalAutoencoderOutput(NamedTuple):
    sample: torch.Tensor
    latent: torch.Tensor
    posterior: Optional[DiagonalGaussianDistribution]
    diffusion_mse_loss: Optional[torch.Tensor]



class CausalEncoderOutput(NamedTuple):
    latent: torch.Tensor
    posterior: Optional[DiagonalGaussianDistribution]


class CausalDecoderOutput(NamedTuple):
    sample: torch.Tensor

class CausalDiffusionDecoderOutput(NamedTuple):
    sample: torch.Tensor
    mse_loss: torch.Tensor



# ---------- Basic building blocks ----------

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Upsample(nn.Module):
    def __init__(self, dim_in, dim_out=None, new_upsample=False):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        if new_upsample:
            # nearest neighbor upsampling followed by conv
            self.up = nn.Upsample(scale_factor=2.0, mode='nearest')
            self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
        else:
            self.up = None
            self.conv = nn.ConvTranspose2d(
                dim_in, dim_out, kernel_size=4, stride=2, padding=1
            )

    def forward(self, x):
        if self.up is not None:
            x = self.up(x)
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class LayerNorm(nn.Module):
    """
    Channel-wise LayerNorm for 4D tensors (b, c, h, w).
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, large_filter=False):
        super().__init__()
        # large_filter is ignored in 2D version for simplicity
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, kernel_size=7 if large_filter else 3, padding=3 if large_filter else 1),
            LayerNorm(dim_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, large_filter=False):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.LeakyReLU(0.2), nn.Linear(time_emb_dim, dim_out))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(dim, dim_out, large_filter)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if time_emb is not None:
            if time_emb.dim() == 1:
                time_emb = time_emb.unsqueeze(-1)
            # time_emb: (b, time_emb_dim)
            h = h + self.mlp(time_emb)[:, :, None, None]

        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=1, dim_head=None):
        super().__init__()
        if dim_head is None:
            dim_head = dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        # spatial attention on flattened spatial dimension
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


# ---------- Gaussian diag distribution ----------

class DiagonalGaussianDistributionDiffusion(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def mode(self):
        return self.mean


# ---------- Compressor (2D VAE-like) ----------

class Compressor(nn.Module):
    """
    2D version of the compressor.
    """
    def __init__(
        self,
        dim=16,
        reverse_dim_mults=(80, 32, 16, 8), # [1280, 512, 256, 128]
        reversed_space_down=(1, 1, 1, 0),
        new_upsample=False,
        channels=3,
        out_channels=16
    ):
        super().__init__()
        self.channels = channels
        out_channels = out_channels
        self.new_upsample = new_upsample
        self.reversed_space_down = reversed_space_down

        # decoder dims
        self.reversed_dims = [*map(lambda m: dim * m, reverse_dim_mults), out_channels]
        self.reversed_in_out = list(zip(self.reversed_dims[:-1], self.reversed_dims[1:]))

        self.build_network()

    @property
    def dtype(self):
        return self.dec[0][0].block1.block[0].weight.dtype # torch.float32

    def build_network(self):
        self.dec = nn.ModuleList([])

        # Decoder: ResnetBlock + upsample
        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            mapping_or_identity = (
                ResnetBlock(dim_in, dim_out)
                if not self.reversed_space_down[ind] and is_last
                else nn.Identity()
            )
            upsample_layer = (
                Upsample(dim_out if not is_last else dim_in, dim_out, self.new_upsample)
                if self.reversed_space_down[ind]
                else mapping_or_identity
            )
            self.dec.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out if not is_last else dim_in),
                        upsample_layer,
                    ]
                )
            )

    def decode(self, z):
        """
        Decode latent z → image.
        """
        x = z.float()
        q = x.to(self.dtype)
        outputs = []
        for (resnet, up) in self.dec:
            q = resnet(q)
            q = up(q)
            outputs.append(q)
        # get the history of outputs for unet context input
        return outputs


# ---------- Unet (2D, image space + multi-scale context) ----------

class Unet(nn.Module):
    def __init__(
        self,
        dim=64,
        out_dim=None,
        dim_mults=(1, 2, 3, 4, 5, 6),
        context_dim=16,
        context_out_channels=16,
        context_dim_mults=(1, 8, 16, 32),
        space_down=(0, 1, 1, 1, 1, 0),
        channels=3,
        with_time_emb=True,
        new_upsample=False,
        embd_type="01",
        condition_times=4,
    ):
        super().__init__()
        self.channels = channels

        # base channels per resolution
        dims = [channels, *map(lambda m: dim * m, dim_mults)] # [3, 64, 128, 192, 256, 320, 384]

        self.space_down = space_down # [0, 1, 1, 1, 1, 0]
        self.reversed_space_down = list(reversed(self.space_down[:-1])) # [1, 1, 1, 1, 0]

        in_out = list(zip(dims[:-1], dims[1:])) # [(3, 64), (64, 128), (128, 192), (192, 256),(256,320),(320,384)]
        self.embd_type = embd_type
        self.condition_times = condition_times

        context_dim = context_dim # 16
        context_out_channels = context_out_channels # 16
        context_dims = [*map(lambda m: context_dim * m, context_dim_mults)] # [16, 128, 256, 512]

        if with_time_emb:
            if embd_type == "01":
                time_dim = dim
                self.time_mlp = nn.Sequential(
                    nn.Linear(1, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
                )
            else:
                raise NotImplementedError
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Encoder (down) path
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            use_context = (not is_last) and (ind < self.condition_times)

            # Input channels to first Resnet at this level:
            #   x channels (dim_in) + context channels (dim_in) if used
            in_channels = dim_in + (context_dims[ind] if use_context else 0)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            in_channels,
                            dim_out,
                            time_dim,
                            True if ind == 0 else False,
                        ),
                        ResnetBlock(dim_out, dim_out, time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if self.space_down[ind] else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_dim)

        # Decoder (up) path
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_dim),
                        ResnetBlock(dim_in, dim_in, time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in, new_upsample=new_upsample) if self.reversed_space_down[ind] else nn.Identity()
                    ]
                )
            )

        out_dim = out_dim or channels
        self.final_conv = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, out_dim, kernel_size=7, padding=3), # 最后一层还是large filter
        )

    @property
    def dtype(self):
        return self.final_conv[1].weight.dtype

    def encode(self, x, t, context):
        """
        context: list of feature maps from Compressor.encode(..., return_features=True)
                 Each context[idx] has channels = dim_in for that level.
        """
        h = []
        for idx, (backbone, backbone2, attn, downsample) in enumerate(self.downs):
            if context is not None and idx < self.condition_times and idx < len(context):
                ctx = context[idx]
                # resize context if spatial size mismatch
                if ctx.shape[2:] != x.shape[2:]:
                    print("trigger resize context")
                    ctx = F.interpolate(ctx, size=x.shape[2:], mode='nearest')
                x = torch.cat([x, ctx], dim=1)

            x = backbone(x, t)
            x = backbone2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        return x, h

    def decode(self, x, h, t):
        device = x.device
        dtype = x.dtype

        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for backbone, backbone2, attn, upsample in self.ups:
            reference = h.pop()
            if x.shape[2:] != reference.shape[2:]:
                x = F.interpolate(x.float(), size=reference.shape[2:], mode='nearest').type_as(x)
            x = torch.cat((x, reference), dim=1)
            x = backbone(x, t)
            x = backbone2(x, t)
            x = attn(x)
            x = upsample(x)

        x = x.to(device).to(dtype)
        x = self.final_conv(x)
        return x

    def forward(self, x, time=None, context=None):
        """
        time: either
            - normalized float timesteps in [0,1] of shape (B, 1), or
            - integer timesteps (B,) which will be normalized here.
        """
        if self.time_mlp is not None and time is not None:
            # Ensure shape (B, 1)
            if time.dim() == 1:
                time = time.unsqueeze(-1)
            time_mlp = self.time_mlp.float()
            t = time_mlp(time.float()).to(self.dtype)
        else:
            t = None

        x = x.to(self.dtype)
        x, h = self.encode(x, t, context)
        return self.decode(x, h, t)


# ---------- Diffusion utilities ----------

def extract(a, t, x_shape):
    a = a.to(t.device)
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn=None,
        context_fn=None,
        ae_fn=None,
        num_timesteps=1024,
        pred_mode="x",
        var_schedule="cosine",
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.context_fn = context_fn  # callable: images -> list of context feats
        self.ae_fn = ae_fn
        self.otherlogs = {}
        self.var_schedule = var_schedule
        self.sample_steps = None
        assert pred_mode in ["noise", "x", "v"]
        self.pred_mode = pred_mode
        to_torch = partial(torch.tensor, dtype=torch.float32)

        train_betas = cosine_beta_schedule(num_timesteps)
        train_alphas = 1.0 - train_betas
        train_alphas_cumprod = np.cumprod(train_alphas, axis=0)
        (num_timesteps,) = train_betas.shape
        self.num_timesteps = int(num_timesteps)

        self.train_snr = to_torch(train_alphas_cumprod / (1 - train_alphas_cumprod))
        self.train_betas = to_torch(train_betas)
        self.train_alphas_cumprod = to_torch(train_alphas_cumprod)
        self.train_sqrt_alphas_cumprod = to_torch(np.sqrt(train_alphas_cumprod))
        self.train_sqrt_one_minus_alphas_cumprod = to_torch(
            np.sqrt(1.0 - train_alphas_cumprod)
        )
        self.train_sqrt_recip_alphas_cumprod = to_torch(
            np.sqrt(1.0 / train_alphas_cumprod)
        )
        self.train_sqrt_recipm1_alphas_cumprod = to_torch(
            np.sqrt(1.0 / train_alphas_cumprod - 1)
        )

    def set_sample_schedule(self, sample_steps, device):
        self.sample_steps = sample_steps
        if sample_steps != 1:
            indice = torch.linspace(0, self.num_timesteps - 1, sample_steps, device=device).long()
        else:
            indice = torch.tensor([self.num_timesteps - 1], device=device).long()
        self.train_alphas_cumprod = self.train_alphas_cumprod.to(device)
        self.train_snr = self.train_snr.to(device)
        self.alphas_cumprod = self.train_alphas_cumprod[indice]
        self.snr = self.train_snr[indice]
        self.index = torch.arange(self.num_timesteps, device=device)[indice]
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod
        self.one_minus_alphas_cumprod_prev = 1.0 - self.alphas_cumprod_prev
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )
        self.sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(
            1.0 - self.alphas_cumprod_prev
        )
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod_prev = torch.sqrt(
            1.0 / self.alphas_cumprod_prev
        )
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1.0 / self.alphas_cumprod - 1
        )
        self.sigma = (
            self.sqrt_one_minus_alphas_cumprod_prev
            / self.sqrt_one_minus_alphas_cumprod
            * torch.sqrt(
                1.0 - self.alphas_cumprod / self.alphas_cumprod_prev
            )
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0)
            / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        if self.training:
            return (
                extract(self.train_sqrt_alphas_cumprod, t, x_start.shape) * noise
                - extract(self.train_sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * x_start
            )
        else:
            return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
                - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * x_start
            )

    def predict_start_from_v(self, x_t, t, v):
        if self.training:
            return (
                extract(self.train_sqrt_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.train_sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
            )
        else:
            return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
            )

    def predict_start_from_noise(self, x_t, t, noise):
        if self.training:
            return (
                extract(self.train_sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.train_sqrt_recipm1_alphas_cumprod, t, x_t.shape)
                * noise
            )
        else:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )

    def ddim(self, x, t, context, clip_denoised, eta=0):
        # time fed to Unet is normalized index[t]/num_timesteps of shape (B, 1)
        if self.denoise_fn.embd_type == "01":
            time_in = self.index[t].float().unsqueeze(-1) / self.num_timesteps
            fx = self.denoise_fn(x, time_in, context=context)
        else:
            fx = self.denoise_fn(x, self.index[t], context=context)

        fx = fx.float()
        if self.pred_mode == "noise":
            x_recon = self.predict_start_from_noise(x, t=t, noise=fx)
        elif self.pred_mode == "x":
            x_recon = fx
        elif self.pred_mode == "v":
            x_recon = self.predict_start_from_v(x, t=t, v=fx)
        else:
            raise ValueError

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        noise = fx if self.pred_mode == "noise" else self.predict_noise_from_start(
            x, t=t, x0=x_recon
        )
        x_next = (
            extract(self.sqrt_alphas_cumprod_prev, t, x.shape) * x_recon
            + torch.sqrt(
                (
                    extract(self.one_minus_alphas_cumprod_prev, t, x.shape)
                    - (eta * extract(self.sigma, t, x.shape)) ** 2
                ).clamp(min=0)
            )
            * noise
            + eta * extract(self.sigma, t, x.shape) * torch.randn_like(noise)
        )
        return x_next

    def p_sample(self, x, t, context, clip_denoised, eta=0):
        return self.ddim(x=x, t=t, context=context, clip_denoised=clip_denoised, eta=eta)

    def p_sample_loop(self, shape, context, clip_denoised=False, init=None, eta=0):
        device = self.alphas_cumprod.device
        b = shape[0]
        img = torch.zeros(shape, device=device) if init is None else init
        for count, i in enumerate(reversed(range(0, self.sample_steps))):
            time = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(
                img,
                time,
                context=context,
                clip_denoised=clip_denoised,
                eta=eta,
            )
        return img

    @torch.no_grad()
    def compress(
        self,
        images,
        sample_steps=10,
        init=None,
        clip_denoised=True,
        eta=0,
    ):
        # context_fn: images -> list of context feature maps
        context = self.context_fn(images)
        self.set_sample_schedule(
            self.num_timesteps if (sample_steps is None) else sample_steps,
            images.device,
        )
        return self.p_sample_loop(
            images.shape,
            context,
            clip_denoised=clip_denoised,
            init=init,
            eta=eta,
        )


# ---------- High-level Tokenizer ----------

class ConditionedDiffusionTokenizer(nn.Module):
    def __init__(
        self,
        # Compressor (VAE) Args
        enc_dim=16,
        reverse_dim_mults=(80, 48, 32, 16, 8),
        reversed_space_down=(1, 1, 1, 1, 0),
        channels=3,
        out_channels=16,
        # Unet (Denoiser) Args
        unet_dim=64,
        unet_dim_mults=(1, 2, 3, 4, 5, 6),
        unet_cond_times=4,
        unet_space_down=(0, 1, 1, 1, 1, 0),
        # Diffusion Args
        beta_schedule="cosine",
        timesteps=1024,
        predict_mode="x",
        sample_steps=3,
        # General
        new_upsample=False,
    ):
        super().__init__()

        # 1. Compressor (2D VAE)
        self.compressor = Compressor(
            dim=enc_dim,
            reverse_dim_mults=reverse_dim_mults,
            reversed_space_down=reversed_space_down,
            channels=channels,
            out_channels=out_channels,
            new_upsample=new_upsample,
        )

        # 2. Unet (Denoiser) in image space
        self.unet = Unet(
            dim=unet_dim,
            out_dim=channels,          # predict image/noise in RGB space
            dim_mults=unet_dim_mults,
            space_down=unet_space_down,
            channels=channels,         # input is image/noisy image
            with_time_emb=True,
            new_upsample=new_upsample,
            condition_times=unet_cond_times,
        )

        # 3. Gaussian Diffusion
        self.diffusion = GaussianDiffusion(
            denoise_fn=self.unet,
            context_fn=self.get_context,  # callable: x -> list of features
            ae_fn=self.compressor,
            num_timesteps=timesteps,
            pred_mode=predict_mode,
            var_schedule=beta_schedule,
        )

        self.sample_steps = sample_steps

    # --- context helper ---

    def get_context(self, x):
        """
        Get multi-scale decoder features from the compressor to condition Unet.

        Returns:
            list of features, length = number of decoder stages
            feats[idx]: [B, C_in_level_idx, H, W]
        """
        # deterministic=True is fine for context
        feats = self.compressor.decode(x)
        return feats

    # --- training forward ---

    def forward(self, x, z, return_recon=False):
        """
        Training forward pass: diffusion loss on images x,
        conditioned on compressor features of x.
        """
        b, c, h, w = x.shape
        device = x.device

        # 1) Get context features using dino v3 embedding
        context = self.get_context(z)
        corrected_context = list(reversed(context[:]))
        # for ind, context_feat in enumerate(corrected_context):
        #     print(f"[diffusion unet] here is the context shape: {corrected_context[ind].shape}")

        # 2) Sample random timesteps
        t = torch.randint(0, self.diffusion.num_timesteps, (b,), device=device).long()

        # 3) Diffuse x to x_t using ground truth image x
        x_start = x
        # print(f"[diffusion unet] here is the x_start shape: {x_start.shape}")
        noise = torch.randn_like(x_start)

        x_noisy = (
            extract(self.diffusion.train_sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.diffusion.train_sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        # print(f"[diffusion unet] here is the x_noisy shape: {x_noisy.shape}")

        # 4) Predict with Unet (time normalized)
        time_in = t.float().unsqueeze(-1) / self.diffusion.num_timesteps
        predicted = self.unet(x_noisy, time=time_in, context=corrected_context)

        # 5) Choose target depending on pred_mode
        if self.diffusion.pred_mode == 'noise':
            target = noise
        elif self.diffusion.pred_mode == 'x':
            target = x_start
        elif self.diffusion.pred_mode == 'v':
            target = self.diffusion.predict_v(x_start, t, noise)
        else:
            raise ValueError(f"Unknown prediction mode {self.diffusion.pred_mode}")

        loss = F.mse_loss(predicted, target)
        log_dict = {'loss': loss.item()}

        if return_recon:
            with torch.no_grad():
                if self.diffusion.pred_mode == 'noise':
                    recon = self.diffusion.predict_start_from_noise(x_noisy, t, predicted)
                elif self.diffusion.pred_mode == 'v':
                    recon = self.diffusion.predict_start_from_v(x_noisy, t, predicted)
                else:
                    recon = predicted
            return loss, log_dict, recon

        return loss, log_dict

    # --- tokenizer API ---

    @torch.no_grad()
    def tokenize(self, x):
        """
        Encodes image to latent z (VAE only).
        """
        z, _ = self.compressor.encode(x, deterministic=True, return_features=False)
        return z

    @torch.no_grad()
    def detokenize(self, z):
        """
        Decodes latent z back to image (VAE decoder only).
        Returns a tensor [B, C, H, W].
        """
        return self.compressor.decode(z)

    # --- sampling / reconstruction ---

    @torch.no_grad()
    def sample(self, x, steps=None, clip_denoised=True):
        """
        Run diffusion sampling to reconstruct x via denoising,
        conditioned on compressor features of x.
        """
        steps = steps or self.sample_steps
        return self.diffusion.compress(
            x,
            sample_steps=steps,
            clip_denoised=clip_denoised,
        )


def mask_channels(tensor, mask_ratio=0.1, channel_dim=1):
    """
    Randomly sets a percentage of channels in the input tensor to zero.
    
    Args:
        tensor (torch.Tensor): The input tensor.
        mask_ratio (float): The fraction of channels to mask (0.0 to 1.0).
        channel_dim (int): The dimension representing the channels.
        
    Returns:
        torch.Tensor: A new tensor with masked channels.
    """
    # Clone to avoid modifying the original tensor in-place
    output = tensor.clone()
    
    # Get the total number of channels
    num_channels = tensor.shape[channel_dim]
    
    # Calculate how many channels to mask
    num_masked = int(num_channels * mask_ratio)
    
    if num_masked == 0:
        return output
    
    # Generate random permutation of channel indices and pick the first 'num_masked'
    mask_indices = torch.randperm(num_channels)[:num_masked]
    
    # Create a simplified index object to access the dynamic channel dimension
    # This creates a slice(None) [equivalent to :] for every dimension
    idx = [slice(None)] * tensor.ndim
    
    # Replace the slice for the channel dimension with our specific indices
    idx[channel_dim] = mask_indices
    
    # Set the selected channels to zero
    output[idx] = 0
    
    return output, mask_indices


# align the normalization for imageNet
class ScalingLayer2D(nn.Module):
    def __init__(self):
        super().__init__()
        # [-1, 1]
        # dataloader 已经做了 mean=0.5, std=0.5 的 normalization
        # 这里再做一次后就可以跟 ImageNet 的 normalization 对齐
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class Encoder2D(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder
        that encodes its input into a latent representation.
    """

    def __init__(self):
        super().__init__()

        # dinov3_hdfs_path = "hdfs://harunava/home/byte_data_seed_vagcp/user/ming.li1/vfm_exp/pretrained_models/dinov3-vith16plus-pretrain-lvd1689m/model.safetensors"
        dinov3_hdfs_path = "hdfs://harunawl/home/byte_data_seed_wl/vgfm/users/xiaojieli/dinov3/dinov3-vith16plus-pretrain-lvd1689m/model.safetensors"
        assert exists(dinov3_hdfs_path), "The dinov3 weights on HDFS is not found."
        download(
            dinov3_hdfs_path,
            dirname="/opt/tiger/vfm/models/dinov3-vith16plus-pretrain-lvd1689m/",
            filename="model.safetensors",
        )

        dinov3_config_hdfs_path = [
            # VA
            # "hdfs://harunava/home/byte_data_seed_vagcp/user/ming.li1/vfm_exp/pretrained_models/dinov3-vith16plus-pretrain-lvd1689m/config.json",
            # "hdfs://harunava/home/byte_data_seed_vagcp/user/ming.li1/vfm_exp/pretrained_models/dinov3-vith16plus-pretrain-lvd1689m/preprocessor_config.json",
            # CN
            "hdfs://harunawl/home/byte_data_seed_wl/vgfm/users/xiaojieli/dinov3/dinov3-vith16plus-pretrain-lvd1689m/config.json",
            "hdfs://harunawl/home/byte_data_seed_wl/vgfm/users/xiaojieli/dinov3/dinov3-vith16plus-pretrain-lvd1689m/preprocessor_config.json",
        ]
        for config_hdfs_path in dinov3_config_hdfs_path:
            assert exists(
                config_hdfs_path
            ), f"The dinov3 config on HDFS is not found: {config_hdfs_path}"
            download(
                config_hdfs_path,
                dirname="/opt/tiger/vfm/models/dinov3-vith16plus-pretrain-lvd1689m/",
                filename=f"{config_hdfs_path.split('/')[-1]}",
            )

        self.dino_v3 = DINOv3ViTModel.from_pretrained(
            pretrained_model_name_or_path="/opt/tiger/vfm/models/dinov3-vith16plus-pretrain-lvd1689m",
            use_safetensors=True,
        )

        self.normalization_layer = ScalingLayer2D()

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `Encoder` class."""
        normalized_sample = self.normalization_layer(sample)

        pred_vit7b = self.dino_v3(pixel_values=normalized_sample)

        return pred_vit7b


class AutoencoderKL(nn.Module):
    r"""
    A VAE model with KL loss for encoding images into latents
    and decoding latent representations into images.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["ResnetBlock2D"]

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        enc_block_out_channels: Tuple[int] = (64,),
        dec_block_out_channels: Tuple[int] = (64,),
        enc_layers_per_block: int = 1,
        dec_layers_per_block: int = 1,
        latent_channels: int = 4,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        gradient_checkpointing: bool = False,
        spatial_downsample_factor: int = 1,
        variational: bool = True,
        latent_as_rep: bool = True,
        noise_tau: float = 0.8,
        denormalize_decoder_output: bool = True,
        running_mode: str = "dec",
        random_masking_channel_ratio: float = 0.0,
        target_latent_channels: Optional[int] = None,  # 新增：目标 latent channel 数，用于 channel 降维
        *args,
        **kwargs,
    ):
        super().__init__()

        assert 2 ** (len(dec_block_out_channels) - 1) == spatial_downsample_factor

        self.spatial_downsample_factor = spatial_downsample_factor
        self.variational = variational
        self.latent_as_rep = latent_as_rep
        self.noise_tau = noise_tau
        self.denormalize_decoder_output = denormalize_decoder_output
        self.random_masking_channel_ratio = random_masking_channel_ratio
        
        # Channel 降维设置：如果 target_latent_channels 未指定或等于 latent_channels，则不做 channel 降维
        self.original_latent_channels = latent_channels
        self.target_latent_channels = target_latent_channels if target_latent_channels is not None else latent_channels
        self.use_channel_downsample = self.target_latent_channels != latent_channels

        print(f"here is the random mask ratio {self.random_masking_channel_ratio}")
        if self.use_channel_downsample:
            print(f"[DINO_VAE] Channel downsample enabled: {latent_channels} -> {self.target_latent_channels}")

        # pass init params to Encoder
        self.encoder = Encoder2D()

        # Channel 降维的 1x1 卷积层
        if self.use_channel_downsample:
            # Encoder 端: 1280 -> target_latent_channels（降维）
            self.channel_downsample_conv = nn.Conv2d(
                latent_channels, self.target_latent_channels, kernel_size=1, stride=1, padding=0
            )
            decoder_in_channels = self.target_latent_channels  # decoder 直接接收降维后的特征
        else:
            self.channel_downsample_conv = None
            decoder_in_channels = latent_channels

        self.diffusion_decoder = ConditionedDiffusionTokenizer()

        # quant_conv 需要根据是否有 channel downsample 来调整
        effective_latent_channels = self.target_latent_channels if self.use_channel_downsample else latent_channels
        self.quant_conv = (
            nn.Conv2d(2 * effective_latent_channels, 2 * effective_latent_channels, 1) if use_quant_conv else None
        )
        self.post_quant_conv = (
            nn.Conv2d(effective_latent_channels, effective_latent_channels, 1) if use_post_quant_conv else None
        )

        self.running_mode = running_mode

        # Additional latent downsampling after DINO's fixed 16x16 grid.
        # Only support total factors of 16 * 2^k (i.e., 16x, 32x, 64x, ...).
        assert (
            self.spatial_downsample_factor % 16 == 0
        ), "spatial_downsample_factor 必须是 16 的整数倍 (16×2^k)"
        extra_factor = self.spatial_downsample_factor // 16
        # extra_factor must be power of two
        assert (
            extra_factor & (extra_factor - 1) == 0
        ), "仅支持 16× 的 2 的幂倍数（例如 16/32/64）"
        extra_steps = 0 if extra_factor == 0 else int(math.log2(extra_factor))

        # latent_downsample_layers 在 channel downsample 之前操作，所以使用原始 latent_channels
        self.latent_downsample_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    latent_channels, latent_channels, kernel_size=3, stride=2, padding=1
                )
                for _ in range(extra_steps)
            ]
        )

        # If running in decoder-only mode, freeze DINOv3 parameters only
        if self.running_mode == "dec":
            print(f"[DINO_VAE] running mode = {self.running_mode} only decoder is trained")
            for p in self.encoder.dino_v3.parameters():
                p.requires_grad = False

    def _noising(self, tensor: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand(
            (tensor.shape[0],) + (1,) * (tensor.dim() - 1),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        noise = noise_sigma * torch.randn_like(tensor)
        return tensor + noise

    def _denormalize_output(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.denormalize_decoder_output:
            return tensor

        # imagenet 的 normalization 参数
        device = tensor.device
        imagenet_mean = torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None].to(device)
        imagenet_std = torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None].to(device)
        return (tensor * imagenet_std + imagenet_mean)

    def forward(self, x: torch.FloatTensor) -> CausalAutoencoderOutput:
        z, p = self.encode(x)
        assert x.size(-2) // z.size(-2) == self.spatial_downsample_factor
        assert x.size(-1) // z.size(-1) == self.spatial_downsample_factor
        # x = self.decode(z).sample
        # print(f"here is the forward input data x shape: {x.shape}")
        diffusion_decode_output = self.diffusion_decode_training(x=x, z=z)
        recon_output = diffusion_decode_output.sample
        x_mse_loss = diffusion_decode_output.mse_loss
        return CausalAutoencoderOutput(recon_output, z, p, x_mse_loss)

    def encode(self, x: torch.FloatTensor) -> CausalEncoderOutput:
        # print(f"here is the input data device: {x.device}")
        # print(f"here is the encoder device: {self.encoder.device}")

        # for name, param in self.encoder.base_model.named_parameters():
        #     print(f"参数 {name} 所在设备: {param.device}")

        if self.running_mode == "dec":
            # print(f"[DINO_VAE] running mode = {self.running_mode} only decoder is trained")
            with torch.no_grad():
                pred_vit7b = self.encoder(x)  # return a baseresponse
        else:
            # print(f"[DINO_VAE] running mode = {self.running_mode} both encoder and decoder are trained")
            pred_vit7b = self.encoder(x)

        cls_len = 1  # cls_token 通常只有1个
        register_len = 4  # 注册token的数量

        # 2. 切片获取 patch_embeddings
        patch_embeddings_restored = pred_vit7b.last_hidden_state[:, cls_len + register_len :, :]

        restored_tensor = patch_embeddings_restored.transpose(1, 2).view(
            patch_embeddings_restored.shape[0], -1, 16, 16
        )

        # Apply optional extra latent downsampling (stride=2 convs) - 空间下采样
        for conv in self.latent_downsample_layers:
            restored_tensor = conv(restored_tensor)

        # Apply channel downsampling if enabled - 通道降维
        if self.channel_downsample_conv is not None:
            restored_tensor = self.channel_downsample_conv(restored_tensor)

        if self.training and self.noise_tau > 0:
            restored_tensor = self._noising(restored_tensor)

        h = self.quant_conv(restored_tensor) if self.quant_conv is not None else restored_tensor
        p = DiagonalGaussianDistribution(h, deterministic=not self.variational)
        # z = p.sample() if self.variational else p.mode()
        # Apply the mask
        if self.random_masking_channel_ratio > 0.0:
            h_masked, indices_zeroed = mask_channels(restored_tensor, mask_ratio=self.random_masking_channel_ratio, channel_dim=1)
            return CausalEncoderOutput(h_masked, p)
        else:
            return CausalEncoderOutput(h, p)

    def diffusion_decode_training(self, x: torch.FloatTensor, z: torch.FloatTensor) -> CausalDiffusionDecoderOutput:
        z = self.post_quant_conv(z) if self.post_quant_conv is not None else z
        recon_mse_loss, log_dict, recon_output = self.diffusion_decoder(x=x, z=z, return_recon=True)
        # print(f"here is the loss value: {log_dict.get('loss')}")
        recon__denormalize_output = self._denormalize_output(recon_output)
        return CausalDiffusionDecoderOutput(recon__denormalize_output, recon_mse_loss)
