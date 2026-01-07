import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np
from functools import partial
from einops import rearrange
from typing import Literal, NamedTuple, Optional, Tuple
import sys
sys.path.append('/share/project/huangxu/sae/SAE')
from models.dino_v3.modeling_dino_v3 import DINOv3ViTModel
from torch.distributed import init_process_group,destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import PngImagePlugin
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision.utils import save_image # Used only in the example usage block
from torch.utils.tensorboard import SummaryWriter  # <<< 新增
import time
PngImagePlugin.MAX_TEXT_CHUNK = 256 * 1024 * 1024  # 256MB
Image.MAX_IMAGE_PIXELS = None  # 比 199,756,800 大就行

import lpips  # <<< 新增：LPIPS 库

# --- Type Definitions (Essential for model structure) ---

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

    # KL divergence is for training, not strictly needed for inference, but kept for completeness
    def kl(self) -> torch.Tensor:
        return 0.5 * torch.sum(
            self.mean**2 + self.var - 1.0 - self.logvar,
            dim=list(range(1, self.mean.ndim)),
        )

# NamedTuples for clearer function outputs
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

# ---------- Basic building blocks for CNN and Transformer components ----------

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
        if new_upsample: # Nearest neighbor upsampling then conv
            self.up = nn.Upsample(scale_factor=2.0, mode='nearest')
            self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
        else: # Transposed convolution for upsampling
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
    Equivalent to nn.GroupNorm(1, dim) but without group dim restriction.
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
        # `large_filter` is used for the first layer in the encoder for ResnetBlock
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
            h = h + self.mlp(time_emb)[:, :, None, None] # Apply time embedding spatially

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
        # Spatial attention on flattened spatial dimension (HW)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2) # Softmax over sequence dimension for q
        k = k.softmax(dim=-1) # Softmax over sequence dimension for k

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v) # (k.T @ v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q) # (context @ q.T).T
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

# ---------- Gaussian diag distribution for Diffusion ----------

class DiagonalGaussianDistributionDiffusion(object):
    """
    A specific Gaussian distribution for the diffusion process, handling mean and logvar.
    """
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

# ---------- Compressor (2D VAE-like Decoder for Context Generation) ----------
# This component acts as a multi-scale feature extractor from the DINO latent.

class Compressor(nn.Module):
    def __init__(
        self,
        dim=16, # Base dimension for the compressor's feature maps
        reverse_dim_mults=(80, 48, 32, 16, 8), # Multipliers for `dim` to get actual channel sizes
        reversed_space_down=(1, 1, 1, 1, 0), # Controls upsampling at each stage
        new_upsample=False,
        channels=3, # Not directly used for DINO latent input, but for output if it were encoder
        out_channels=16 # Final output channels (e.g., if it was encoding an image)
    ):
        super().__init__()
        self.channels = channels
        self.new_upsample = new_upsample
        self.reversed_space_down = reversed_space_down

        # `reverse_dim_mults` defines the channel progression for the decoder path.
        # It's intended to start from a high channel count (e.g., DINO latent 1280) and reduce.
        # If `dim * reverse_dim_mults[0]` is the input channel count (1280 from DINO latent),
        # and `out_channels` is the final desired output channels (e.g., 16 for compressor output),
        # the list is structured to progressively reduce dimensions.
        
        # Example: if dim=16, reverse_dim_mults=(80, 48, 32, 16, 8)
        # channels become: 16*80=1280, 16*48=768, 16*32=512, 16*16=256, 16*8=128, then final out_channels.
        # The first element of `reversed_dims` should align with the latent input channel count.
        self.reversed_dims = [*map(lambda m: dim * m, reverse_dim_mults), out_channels]
        self.reversed_in_out = list(zip(self.reversed_dims[:-1], self.reversed_dims[1:]))

        self.build_network()

    @property
    def dtype(self):
        # Infer dtype from a specific layer's weight
        return self.dec[0][0].block1.block[0].weight.dtype

    def build_network(self):
        self.dec = nn.ModuleList([])

        # Decoder path: ResnetBlock + Upsample to progressively increase resolution
        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            
            # The `mapping_or_identity` handles the case where there's no upsampling at this stage
            mapping_or_identity = (
                ResnetBlock(dim_in, dim_out)
                if not self.reversed_space_down[ind] and is_last # This condition might be complex
                else nn.Identity()
            )
            
            # Decide whether to use an Upsample layer or just an identity/mapping
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
        Decode latent `z` (from DINO) into multi-scale feature maps.
        These feature maps will then be used as context for the Diffusion UNet.
        """
        x = z.float()
        q = x.to(self.dtype)
        outputs = []
        for (resnet, up) in self.dec:
            q = resnet(q)
            q = up(q)
            outputs.append(q) # Store intermediate feature maps for UNet context
        return outputs

# ---------- Unet (2D, image space + multi-scale context) ----------
# This is the denoiser for the diffusion process.

class Unet(nn.Module):
    def __init__(
        self,
        dim=64, # Base dimension for UNet feature maps
        out_dim=None, # Output channels (e.g., 3 for RGB image)
        dim_mults=(1, 2, 3, 4, 5, 6), # Multipliers for `dim` to get UNet channel sizes
        context_dim=16, # Base dimension for context features (from Compressor)
        context_out_channels=16, # Not directly used as input, refers to compressor's final output
        context_dim_mults=(1, 8, 16, 32), # Multipliers for `context_dim`
        space_down=(0, 1, 1, 1, 1, 0), # Controls downsampling/upsampling in UNet
        channels=3, # Input channels (e.g., 3 for noisy RGB image)
        with_time_emb=True,
        new_upsample=False,
        embd_type="01", # Type of time embedding (e.g., "01" for simple linear)
        condition_times=4, # How many initial UNet encoder stages receive context
    ):
        super().__init__()
        self.channels = channels

        # UNet's internal dimensions progression
        dims = [channels, *map(lambda m: dim * m, dim_mults)]

        self.space_down = space_down
        # Reversed downsampling for the upsampling path
        self.reversed_space_down = list(reversed(self.space_down[:-1]))

        # Pairs of (input_dim, output_dim) for UNet stages
        in_out = list(zip(dims[:-1], dims[1:]))
        self.embd_type = embd_type
        self.condition_times = condition_times

        # Context feature dimensions from the Compressor
        context_dims = [*map(lambda m: context_dim * m, context_dim_mults)]

        if with_time_emb:
            if embd_type == "01": # Simple linear time embedding
                time_dim = dim
                self.time_mlp = nn.Sequential(
                    nn.Linear(1, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
                )
            else:
                raise NotImplementedError(f"Time embedding type {embd_type} not implemented.")
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([]) # Encoder (downsampling) path
        self.ups = nn.ModuleList([])   # Decoder (upsampling) path
        num_resolutions = len(in_out)

        # Encoder (downsampling) path
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            use_context = (not is_last) and (ind < self.condition_times)

            # Input channels for this UNet level: original input + concatenated context features
            in_channels = dim_in + (context_dims[ind] if use_context else 0)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            in_channels, # Input channels might be wider due to context
                            dim_out,
                            time_dim,
                            True if ind == 0 else False, # Apply large filter only for the very first block
                        ),
                        ResnetBlock(dim_out, dim_out, time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if self.space_down[ind] else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1] # Latent space dimension after encoding
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_dim)

        # Decoder (upsampling) path
        # It processes in reverse order of the encoder path (skipping the first encoder input)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_dim), # Concatenates with skip connection
                        ResnetBlock(dim_in, dim_in, time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in, new_upsample=new_upsample) if self.reversed_space_down[ind] else nn.Identity()
                    ]
                )
            )

        out_dim = out_dim or channels # Final output channels (e.g., 3 for RGB)
        self.final_conv = nn.Sequential(
            LayerNorm(dim), # Normalization before final output
            nn.Conv2d(dim, out_dim, kernel_size=7, padding=3), # Final convolution, typically large filter
        )

    @property
    def dtype(self):
        return self.final_conv[1].weight.dtype

    def encode(self, x, t, context):
        """Processes input through the UNet encoder path."""
        h = [] # Stores skip connections
        for idx, (backbone, backbone2, attn, downsample) in enumerate(self.downs):
            if context is not None and idx < self.condition_times and idx < len(context):
                ctx = context[idx] # Multi-scale context feature
                # Resize context if its spatial size does not match current `x`
                if ctx.shape[2:] != x.shape[2:]:
                    # print(f"UNet: Resizing context from {ctx.shape[2:]} to {x.shape[2:]}")
                    ctx = F.interpolate(ctx, size=x.shape[2:], mode='nearest')
                x = torch.cat([x, ctx], dim=1) # Concatenate context channels

            x = backbone(x, t)
            x = backbone2(x, t)
            x = attn(x)
            h.append(x) # Store for skip connection
            x = downsample(x)
        x = self.mid_block1(x, t) # Middle block processing
        return x, h

    def decode(self, x, h, t):
        """Processes input through the UNet decoder path."""
        device = x.device
        dtype = x.dtype

        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for backbone, backbone2, attn, upsample in self.ups:
            reference = h.pop() # Retrieve skip connection
            # Ensure skip connection and current feature map have same spatial size
            if x.shape[2:] != reference.shape[2:]:
                x = F.interpolate(x.float(), size=reference.shape[2:], mode='nearest').type_as(x)
            x = torch.cat((x, reference), dim=1) # Concatenate skip connection
            x = backbone(x, t)
            x = backbone2(x, t)
            x = attn(x)
            x = upsample(x)

        x = x.to(device).to(dtype)
        x = self.final_conv(x) # Final convolution to desired output channels
        return x

    def forward(self, x, time=None, context=None):
        """
        Forward pass for the UNet.
        `x`: Noisy input image.
        `time`: Diffusion timestep (normalized).
        `context`: Multi-scale features from the Compressor.
        """
        if self.time_mlp is not None and time is not None:
            if time.dim() == 1: # Ensure time is (B, 1)
                time = time.unsqueeze(-1)
            time_mlp = self.time_mlp.float()
            t = time_mlp(time.float()).to(self.dtype)
        else:
            t = None

        x = x.to(self.dtype)
        x, h = self.encode(x, t, context)
        return self.decode(x, h, t)

# ---------- Diffusion utilities (for GaussianDiffusion) ----------

def extract(a, t, x_shape):
    """Extract values from a 1D tensor `a` at indices `t` and reshape."""
    a = a.to(t.device)
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Generates a cosine beta schedule for diffusion models.
    Smoothly increases beta values.
    """
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

    # ================== New: rectified-flow-style sampling ================== #

    def ddim(self, vae, x, t_idx, t_next_idx, context, clip_denoised: bool):
        """
        One Euler step following Algorithm 2:

            x_pred = net(z, t)
            v_pred = (x_pred - z) / (1 - t)
            z_next = z + (t_next - t) * v_pred

        Args:
            vae: not used here except for可选 debug，可忽略
            x:      current z(t), shape [B, C, H, W]
            t_idx:  当前时间索引（0 .. sample_steps-1）
            t_next_idx: 下一时间索引
            context: multi-scale context from compressor
            clip_denoised: 是否把 x_pred / z_next clamp 到 [-1,1]
        Returns:
            z_next, x_pred
        """
        assert self.pred_mode == "x", "flow-style sampling assumes UNet predicts x"

        b = x.shape[0]
        device = x.device
        dtype = x.dtype

        # ---- 1) 把离散的 t_idx 映射到 [0,1] 上，与训练保持一致 ----
        # set_sample_schedule 里你已经构建了 self.index: 长度 sample_steps，
        # 其中元素是原始 0..num_timesteps-1 的索引
        t_scalar = (self.index[t_idx].float() / self.num_timesteps)          # 标量, in (0,1)
        t_next_scalar = (self.index[t_next_idx].float() / self.num_timesteps)

        # 喂给 UNet 的是 batch-wise 的 (B, 1) time 向量
        if self.denoise_fn.embd_type == "01":
            time_in = t_scalar.expand(b, 1).to(device=device, dtype=torch.float32)
            x_pred = self.denoise_fn(x, time_in, context=context)
        else:
            # 如果将来有别的时间 embedding 形式，这里可以改
            time_idx_full = torch.full((b,), self.index[t_idx],
                                       device=device, dtype=torch.long)
            x_pred = self.denoise_fn(x, time_idx_full, context=context)
        x_pred = x_pred.to(dtype)
        if clip_denoised:
            x_pred = x_pred.clamp(-1.0, 1.0)

        # ---- 2) 计算 v_pred = (x_pred - z) / (1 - t) ----
        # one_minus_t = (1.0 - t_scalar).to(dtype=dtype, device=device)  # 标量
        # 避免除 0
        # one_minus_t = torch.clamp(one_minus_t, min=1e-6)
        v_pred = (x_pred - x) / t_scalar

        # ---- 3) Euler 步进: z_next = z + (t_next - t) * v_pred ----
        dt = (t_next_scalar - t_scalar).to(dtype=dtype, device=device)  # 负数
        z_next = x - dt * v_pred

        

        return z_next, x_pred

    def p_sample(self, vae, x, t_idx, t_next_idx, context, clip_denoised):
        """
        Wrapper: 单步采样
        """
        return self.ddim(
            vae=vae,
            x=x,
            t_idx=t_idx,
            t_next_idx=t_next_idx,
            context=context,
            clip_denoised=clip_denoised,
        )

    def p_sample_loop(self, vae, shape, context, clip_denoised: bool = False,
                      init_noise: torch.Tensor = None, eta: float = 0.0):
        """
        Rectified-flow Euler integration from t=1 -> t=0

        Args:
            vae: 仅用于 debug 保存图像，可选
            shape: 目标图像 shape = [B, C, H, W]
            context: compressor 提供的 multi-scale features
            clip_denoised: 每步是否 clamp z 到 [-1,1]
            init_noise: 初始 z(1)，若 None 则 N(0,1)
            eta: 保留参数，这里不使用
        Returns:
            最终的 z(0)，应接近重建图像 x
        """
        device = self.alphas_cumprod.device  # 任意已经在 set_sample_schedule 中 to(device) 的 tensor
        b = shape[0]

        # z(t=1) ~ N(0, I)
        img = init_noise if init_noise is not None else torch.randn(shape, device=device)

        # 时间索引序列: [0, 1, ..., sample_steps-1] 反过来从大到小走
        time_indices = list(range(self.sample_steps))
        time_indices = list(reversed(time_indices))  # e.g. [K-1, ..., 1, 0]

        for cur_idx, next_idx in zip(time_indices[:-1], time_indices[1:]):
            img, x_pred = self.p_sample(
                vae=vae,
                x=img,
                t_idx=cur_idx,
                t_next_idx=next_idx,
                context=context,
                clip_denoised=clip_denoised,
            )

            # ---- 可选：debug 可视化 ----
            # 这里假设整个 diffusion 都在 [-1,1] 空间里
            # 如果你想看图，用 denormalize_for_display 即可
            # recon_vis = vae._denormalize_output(x_pred)
            # recon_vis = x_pred
            # recon_vis = torch.clamp(recon_vis,-1,1)
            # recon_vis = denormalize_for_display(recon_vis)
            # save_image(recon_vis, f"./reconstruction_output/recon_{cur_idx}.png")

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
        time_shift=3,
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
        self.time_shift = time_shift
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

    def forward(self, x, z, return_recon: bool = False):
        """
        Training forward pass: diffusion loss on images x,
        conditioned on compressor features of x.

        使用最基本的噪声混合：
            x_noisy = t * noise + (1 - t) * x
        其中 t ∈ [0, 1]，并在 t 上做 SD3-style flow shift：
            t' = (shift * t) / (1 + (shift - 1) * t)
        """
        b, c, h, w = x.shape
        device = x.device
        num_steps = self.diffusion.num_timesteps  # 这里只用于 bookkeeping

        # 1) Context features from compressor
        context = self.get_context(z)
        corrected_context = list(reversed(context[:]))

        # 2) 连续线性时间 t_lin ~ U(0, 1)
        t_lin = torch.rand(b, device=device)

        #    SD3 风格的 flow shift：t_flow = (shift * t) / (1 + (shift - 1) * t)
        shift = float(self.time_shift)
        t_flow = (shift * t_lin) / (1.0 + (shift - 1.0) * t_lin)
        t_flow = t_flow.clamp(0.0, 1.0 - 1e-6)

        #    把 t_flow 映射到离散 step index（如果你想在别处用到 step，可以留着）
        t_index = (t_flow * (num_steps - 1)).long()

        # 3) 最基础的噪声注入：x_noisy = t * noise + (1 - t) * x
        x_start = x
        noise = torch.randn_like(x_start)

        #    把标量 t_flow broadcast 成 [B,1,1,1]
        t_scalar = t_flow.view(b, 1, 1, 1)
        x_noisy = t_scalar * noise + (1.0 - t_scalar) * x_start

        # 4) UNet 预测：time 输入用连续的 t_flow
        time_in = t_flow.unsqueeze(-1)  # [B, 1]
        predicted = self.unet(x_noisy, time=time_in, context=corrected_context)

        # 5) 选择 target
        if self.diffusion.pred_mode == "noise":
            target = noise
        elif self.diffusion.pred_mode == "x":
            target = x_start
        elif self.diffusion.pred_mode == "v":
            # 简单 scheduler 下没专门推 v 的公式，先禁掉，避免用错
            raise NotImplementedError(
                "v-prediction not defined for simple linear scheduler; "
                "use pred_mode='noise' or 'x'."
            )
        else:
            raise ValueError(f"Unknown prediction mode {self.diffusion.pred_mode}")

        loss = F.mse_loss(predicted, target)
        # DEBUG = False
        # # print("t",t_lin,"t_flow",t_flow,"mse_loss",loss)
        # if DEBUG:
        #     recon_0 = predicted[0].clamp(-1, 1)
        #     reconstructed_display = denormalize_for_display(recon_0)
        #     save_image(reconstructed_display, f"./debug_output/pred_{int(t_flow[0] * 1000)}.png")

        #     recon_1 = target[0].clamp(-1, 1)
        #     reconstructed_display = denormalize_for_display(recon_1)
        #     save_image(reconstructed_display, f"./debug_output/target_{int(t_flow[0] * 1000)}.png")

        log_dict = {"loss": loss.item()}

        if return_recon:
            with torch.no_grad():
                if self.diffusion.pred_mode == "noise":
                    # 由 x_noisy = t * noise + (1 - t) * x_start 解出 x_start
                    recon = (x_noisy - t_scalar * predicted) / (1.0 - t_scalar + 1e-8)
                elif self.diffusion.pred_mode == "x":
                    recon = predicted
                else:
                    recon = predicted  # 理论上上面已经 NotImplementedError 了
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


# ---------- Image Normalization Layer (for DINO input) ----------

class ScalingLayer2D(nn.Module):
    """
    Applies ImageNet-style normalization to input tensors.
    Assumes input is already in [-1, 1] range and converts to ImageNet's mean/std.
    DINOv3 expects normalized inputs similar to ImageNet.
    """
    def __init__(self):
        super().__init__()
        # These are precomputed values to map [-1, 1] to ImageNet distribution
        # For input `x_norm = (x - 0.5) / 0.5`, this layer does `(x_norm - shift) / scale`
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


# ---------- Encoder2D (DINOv3 based) ----------

class Encoder2D(nn.Module):
    """
    The Encoder part of the VAE, leveraging a pretrained DINOv3 ViT model
    to extract a rich latent representation from an image.
    """
    def __init__(self):
        super().__init__()

        # Define a local path for the DINOv3 model files.
        # This path must contain: model.safetensors, config.json, preprocessor_config.json
        # For this self-contained example, these files will be virtually created if not found.
        dinov3_model_dir = "/share/project/huangxu/models/dinov3"
        
        import os
        # Ensure the directory exists for the placeholder model
        os.makedirs(dinov3_model_dir, exist_ok=True)
        # Create dummy config files if they don't exist, as `from_pretrained` might check
        for fname in ["config.json", "preprocessor_config.json"]:
            fpath = os.path.join(dinov3_model_dir, fname)
            if not os.path.exists(fpath):
                with open(fpath, "w") as f:
                    f.write("{}") # Write an empty JSON object
        
        # Initialize the DINOv3 model (using our placeholder here)
        self.dino_v3 = DINOv3ViTModel.from_pretrained(
            pretrained_model_name_or_path=dinov3_model_dir,
            use_safetensors=True, # Or False, depending on your actual model
        )

        self.normalization_layer = ScalingLayer2D()

    def forward(self, sample: torch.Tensor):
        """
        Forward pass for the Encoder:
        1. Normalize the input image.
        2. Pass through the DINOv3 model.
        """
        normalized_sample = self.normalization_layer(sample)
        pred_vit7b = self.dino_v3(pixel_values=normalized_sample)
        return pred_vit7b


# ---------- AutoencoderKL (The full DINO VAE) ----------

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
        lpips_weight: float = 0.1,  # <<< 新增：LPIPS loss 权重
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

        # <<< 新增：LPIPS 配置
        self.lpips_weight = lpips_weight
        self.lpips = lpips.LPIPS(net='vgg')
        self.lpips.eval()
        for p in self.lpips.parameters():
            p.requires_grad = False

        # <<< 新增：日志字典，用于在训练外部打印 MSE / LPIPS
        self.otherlogs = {}

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

    def _normalize_input(self,tensor:torch.Tensor) -> torch.Tensor:
        if not self.denormalize_decoder_output:
            return tensor

        # imagenet 的 normalization 参数
        device = tensor.device
        imagenet_mean = torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None].to(device)
        imagenet_std = torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None].to(device)
        return (tensor - imagenet_mean) / imagenet_std


    def forward(self, x: torch.FloatTensor) -> CausalAutoencoderOutput:
        z, p = self.encode(x)
        assert x.size(-2) // z.size(-2) == self.spatial_downsample_factor
        assert x.size(-1) // z.size(-1) == self.spatial_downsample_factor

        diffusion_decode_output = self.diffusion_decode_training(x=x, z=z)
        recon_output = diffusion_decode_output.sample
        x_mse_loss = diffusion_decode_output.mse_loss
        return CausalAutoencoderOutput(recon_output, z, p, x_mse_loss)

    def encode(self, x: torch.FloatTensor) -> CausalEncoderOutput:
        h,w = x.shape[-2], x.shape[-1]
        if self.running_mode == "dec":
            with torch.no_grad():
                pred_vit7b = self.encoder(x)  # return a baseresponse
        else:
            pred_vit7b = self.encoder(x)

        cls_len = 1  # cls_token 通常只有1个
        register_len = 4  # 注册token的数量

        # 2. 切片获取 patch_embeddings
        patch_embeddings_restored = pred_vit7b.last_hidden_state[:, cls_len + register_len :, :]

        restored_tensor = patch_embeddings_restored.transpose(1, 2).view(
            patch_embeddings_restored.shape[0], -1, h//16, w//16
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
        p = DiagonalGaussianDistributionDiffusion(parameters=h, deterministic=not self.variational)

        return CausalEncoderOutput(h, p)

    def diffusion_decode_training(self, x: torch.FloatTensor, z: torch.FloatTensor) -> CausalDiffusionDecoderOutput:
        """
        训练时 decoder 的前向过程：
        - 先通过 diffusion_decoder 得到 MSE 与重建
        - 再计算 LPIPS loss，并按比例加权到总 loss 中
        """
        # 1. 量化后卷积
        z = self.post_quant_conv(z) if self.post_quant_conv is not None else z

        # 2. diffusion decoder 计算重建 MSE loss 和 recon_output（[-1,1]）
        recon_mse_loss, log_dict, recon_output = self.diffusion_decoder(
            x=x,
            z=z,
            return_recon=True
        )

        # 3. LPIPS 感知损失（LPIPS 输入期望在 [-1,1]）
        x_lpips = x.clamp(-1, 1)
        recon_lpips = recon_output.clamp(-1, 1)
        lpips_loss = self.lpips(x_lpips, recon_lpips).mean()

        # 4. 总 loss = MSE + λ * LPIPS
        total_loss = recon_mse_loss + self.lpips_weight * lpips_loss
        # print("lpips_loss",lpips_loss)

        # 记录日志（用于外部查看）
        self.otherlogs["mse_loss"] = recon_mse_loss.detach()
        self.otherlogs["lpips_loss"] = lpips_loss.detach()
        self.otherlogs["total_loss"] = total_loss.detach()

        # 5. 用于输出查看的 sample：反归一化到 [0,1] 附近的图像
        recon__denormalize_output = self._denormalize_output(recon_output)

        # 注意：CausalDiffusionDecoderOutput 的第二个字段名叫 mse_loss，
        # 这里实际装的是 total_loss（总损失），训练那边用的就是这个。
        return CausalDiffusionDecoderOutput(recon__denormalize_output, total_loss)




