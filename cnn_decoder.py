# cnn_decoder.py
import math
import contextlib
from typing import NamedTuple, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

# your dinov3 import
import sys
from models.dino_v3.modeling_dino_v3 import DINOv3ViTModel

# SigLIP2 import
try:
    from transformers import SiglipModel
    HAS_SIGLIP = True
except ImportError:
    HAS_SIGLIP = False
    print("Warning: transformers SiglipModel not found. SigLIP2 encoder will not be available.")


# =========================
# Outputs
# =========================

class CausalAutoencoderOutput(NamedTuple):
    sample: torch.Tensor
    latent: torch.Tensor
    posterior: Optional[DiagonalGaussianDistribution]

class CausalEncoderOutput(NamedTuple):
    latent: torch.Tensor
    posterior: Optional[DiagonalGaussianDistribution]

class CausalDecoderOutput(NamedTuple):
    sample: torch.Tensor


# =========================
# LoRA Modules
# =========================

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 32, alpha: int = 32, dropout: float = 0.0, enabled: bool = True):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(r, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = enabled

        self.lora_down = nn.Linear(base.in_features, r, bias=False)
        self.lora_up = nn.Linear(r, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        out = self.base(x)
        if self.enabled and self.r > 0:
            out = out + self.lora_up(self.dropout(self.lora_down(x))) * self.scaling
        return out


class LoRAConv2d(nn.Module):
    """
    LoRA for Conv2d, best for patch-embed conv:
      down: Conv2d(in->r, kernel=k, stride=s)
      up:   Conv2d(r->out, kernel=1)
    """
    def __getattr__(self, name):
        # nn.Module 自己的属性先走默认逻辑
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # 再兜底到 base（比如 padding_mode 等）
        return getattr(self.base, name)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def __init__(self, base: nn.Conv2d, r: int = 32, alpha: int = 32, dropout: float = 0.0, enabled: bool = True):
        super().__init__()
        assert isinstance(base, nn.Conv2d)
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(r, 1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.enabled = enabled

        self.lora_down = nn.Conv2d(
            base.in_channels, r,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=base.groups,
            bias=False,
        )
        self.lora_up = nn.Conv2d(r, base.out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        out = self.base(x)
        if self.enabled and self.r > 0:
            out = out + self.lora_up(self.dropout(self.lora_down(x))) * self.scaling
        return out


def _set_lora_enabled(module: nn.Module, enabled: bool):
    for m in module.modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            m.enabled = enabled


@contextlib.contextmanager
def lora_disabled(module: nn.Module):
    prev = []
    for m in module.modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            prev.append((m, m.enabled))
            m.enabled = False
    try:
        yield
    finally:
        for m, e in prev:
            m.enabled = e


def add_lora_to_dinov3(dino_model: nn.Module, r: int = 32, alpha: int = 32, dropout: float = 0.0):
    """
    Targets:
      - dino_model.embeddings.patch_embeddings (Conv2d)
      - each block.attention.{q_proj,k_proj,v_proj} (Linear)
    """
    # patch embed
    pe = dino_model.embeddings.patch_embeddings
    dino_model.embeddings.patch_embeddings = LoRAConv2d(pe, r=r, alpha=alpha, dropout=dropout, enabled=True)

    # qkv
    for blk in dino_model.layer:
        attn = blk.attention
        attn.q_proj = LoRALinear(attn.q_proj, r=r, alpha=alpha, dropout=dropout, enabled=True)
        attn.k_proj = LoRALinear(attn.k_proj, r=r, alpha=alpha, dropout=dropout, enabled=True)
        attn.v_proj = LoRALinear(attn.v_proj, r=r, alpha=alpha, dropout=dropout, enabled=True)

    return dino_model


def add_lora_to_siglip2(siglip_vision_model: nn.Module, r: int = 32, alpha: int = 32, dropout: float = 0.0):
    """
    为 SigLIP2 VisionModel 添加 LoRA
    Targets:
      - vision_model.embeddings.patch_embedding (Conv2d)
      - each encoder.layers[i].self_attn.{q_proj,k_proj,v_proj} (Linear)
    """
    # patch embed - SigLIP2 使用 patch_embedding
    if hasattr(siglip_vision_model.embeddings, 'patch_embedding'):
        pe = siglip_vision_model.embeddings.patch_embedding
        siglip_vision_model.embeddings.patch_embedding = LoRAConv2d(pe, r=r, alpha=alpha, dropout=dropout, enabled=True)

    # qkv in encoder layers
    for layer in siglip_vision_model.encoder.layers:
        attn = layer.self_attn
        attn.q_proj = LoRALinear(attn.q_proj, r=r, alpha=alpha, dropout=dropout, enabled=True)
        attn.k_proj = LoRALinear(attn.k_proj, r=r, alpha=alpha, dropout=dropout, enabled=True)
        attn.v_proj = LoRALinear(attn.v_proj, r=r, alpha=alpha, dropout=dropout, enabled=True)

    return siglip_vision_model


# =========================
# Feature alignment losses (VF loss)
# =========================

def vf_marginal_cos_loss(z_map: torch.Tensor, f_map: torch.Tensor, m1: float = 0.1, eps: float = 1e-6) -> torch.Tensor:
    """
    z_map,f_map: [B,C,H,W]
    Lmcos = mean ReLU(1 - m1 - cos(z,f))
    """
    z = F.normalize(z_map, dim=1, eps=eps)
    f = F.normalize(f_map, dim=1, eps=eps)
    cos = (z * f).sum(dim=1)  # [B,H,W]
    return F.relu(1.0 - m1 - cos).mean()


def vf_mdms_loss(
    z_map: torch.Tensor,
    f_map: torch.Tensor,
    m2: float = 0.1,
    max_tokens: int = 32,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Pair-wise distance-matrix similarity with random token sampling.
    z_map,f_map: [B,C,H,W]
    - Flatten to tokens N=H*W
    - Randomly sample K=min(N,max_tokens) positions (shared across batch)
    - Compare cosine-sim matrices:
        Lmdms = mean ReLU(|sim_z - sim_f| - m2)
    """
    B, C, H, W = z_map.shape
    N = H * W
    K = min(N, max_tokens)
    if K <= 1:
        return torch.zeros((), device=z_map.device, dtype=z_map.dtype)

    # tokens: [B,N,C]
    z = z_map.flatten(2).transpose(1, 2).contiguous()
    f = f_map.flatten(2).transpose(1, 2).contiguous()

    # sample shared indices for efficiency
    idx = torch.randperm(N, device=z.device)[:K]
    z = z[:, idx, :]
    f = f[:, idx, :]

    z = F.normalize(z, dim=-1, eps=eps)
    f = F.normalize(f, dim=-1, eps=eps)

    sim_z = torch.bmm(z, z.transpose(1, 2))  # [B,K,K]
    sim_f = torch.bmm(f, f.transpose(1, 2))  # [B,K,K]

    diff = (sim_z - sim_f).abs()
    return F.relu(diff - m2).mean()


# =========================
# Utilities
# =========================

def mask_channels(tensor, mask_ratio=0.1, channel_dim=1):
    output = tensor.clone()
    num_channels = tensor.shape[channel_dim]
    num_masked = int(num_channels * mask_ratio)
    if num_masked == 0:
        return output, None
    mask_indices = torch.randperm(num_channels, device=tensor.device)[:num_masked]
    idx = [slice(None)] * tensor.ndim
    idx[channel_dim] = mask_indices
    output[idx] = 0
    return output, mask_indices


# =========================
# Blocks for CNN decoder
# =========================

class ResnetBlock2D(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.nonlinearity = nn.SiLU()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.nonlinearity(self.norm1(x)))
        h = self.conv2(self.dropout(self.nonlinearity(self.norm2(h))))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class Upsample2D(nn.Module):
    """A 2D upsampling layer

    Parameters:
        channels (`int`): number of channels in the inputs and outputs.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels

        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")

        hidden_states = self.conv(hidden_states)

        return hidden_states


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels, out_channels=out_channels, dropout=dropout
                )
            )

        self.resnets = nn.ModuleList(resnets)
        # NOTE: DO NOT USE SEQUENTIAL HERE.

        self.upsampler = Upsample2D(out_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        hidden_states = self.upsampler(hidden_states)

        return hidden_states

class FinalBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, dropout=dropout)
            )

        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.Tensor):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        return hidden_states


class Decoder2D(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder
        that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3): The number of input channels.
        out_channels (`int`, *optional*, defaults to 3): The number of output channels.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
                            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        gradient_checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to switch on gradient checkpointing.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1
        )

        self.up_blocks = nn.ModuleList([])

        # up
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels) - 1):
            prev_output_channel = output_channel
            output_channel = block_out_channels[i]

            up_block = UpDecoderBlock2D(
                num_layers=self.layers_per_block,
                in_channels=prev_output_channel,
                out_channels=output_channel,
            )
            self.up_blocks.append(up_block)

        # final
        self.final_block = FinalBlock2D(
            in_channels=output_channel,
            out_channels=block_out_channels[-1],
            num_layers=self.layers_per_block,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=32, eps=1e-6
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[-1], out_channels, 3, padding=1)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:
            # up
            for up_block in self.up_blocks:
                sample = torch.utils.checkpoint.checkpoint(
                    up_block,
                    sample,
                    use_reentrant=False,
                )

            # final
            sample = torch.utils.checkpoint.checkpoint(
                self.final_block,
                sample,
                use_reentrant=False,
            )

        else:
            # up
            for up_block in self.up_blocks:
                sample = up_block(sample)

            # final
            sample = self.final_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


# =========================
# Normalization for DINO input
# =========================

class ScalingLayer2D(nn.Module):
    """
    input: [-1,1] (after dataset transform t*2-1)
    output: aligned to ImageNet normalization for DINO (approx)
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, x):
        return (x - self.shift) / self.scale


# =========================
# Normalization for SigLIP2 input
# =========================

class SigLIP2ScalingLayer(nn.Module):
    """
    SigLIP2 的输入归一化层
    input: [-1,1] (after dataset transform t*2-1)
    output: aligned to SigLIP2 expected normalization
    """
    def __init__(self):
        super().__init__()
        # SigLIP2 使用类似 CLIP 的归一化
        # mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] 意味着输入就是 [-1,1]
        # 所以实际上不需要额外变换，但为了灵活性保留此层
        self.register_buffer("shift", torch.Tensor([0.0, 0.0, 0.0])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([1.0, 1.0, 1.0])[None, :, None, None])

    def forward(self, x):
        return (x - self.shift) / self.scale


# =========================
# Encoder (DINOv3 + LoRA)
# =========================

class Encoder2D(nn.Module):
    """DINOv3 Encoder with LoRA support"""
    def __init__(self, dinov3_model_dir: str, lora_rank=32, lora_alpha=32, lora_dropout=0.0, enable_lora=True):
        super().__init__()
        self.encoder_type = "dinov3"
        self.dino_v3 = DINOv3ViTModel.from_pretrained(
            pretrained_model_name_or_path=dinov3_model_dir,
            use_safetensors=True,
        )
        add_lora_to_dinov3(self.dino_v3, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        _set_lora_enabled(self.dino_v3, enable_lora)

        self.normalization_layer = ScalingLayer2D()
        
        # 保存配置
        self.hidden_size = self.dino_v3.config.hidden_size
        self.patch_size = self.dino_v3.config.patch_size
        self.num_register_tokens = getattr(self.dino_v3.config, 'num_register_tokens', 4)

    def set_lora_enabled(self, enabled: bool):
        _set_lora_enabled(self.dino_v3, enabled)

    @contextlib.contextmanager
    def lora_disabled(self):
        with lora_disabled(self.dino_v3):
            yield

    def forward(self, x: torch.Tensor):
        x = self.normalization_layer(x)
        return self.dino_v3(pixel_values=x)
    
    def get_backbone(self):
        """获取底层 backbone 用于 LoRA 参数访问"""
        return self.dino_v3


# =========================
# Encoder (SigLIP2 + LoRA)
# =========================

class SigLIP2Encoder2D(nn.Module):
    """SigLIP2 Encoder with LoRA support"""
    def __init__(self, model_name: str, lora_rank=32, lora_alpha=32, lora_dropout=0.0, enable_lora=True):
        super().__init__()
        if not HAS_SIGLIP:
            raise ImportError("SigLIP2 requires transformers with SiglipModel. Please install: pip install transformers>=4.36.0")
        
        self.encoder_type = "siglip2"
        self.model_name = model_name
        
        # 加载 SigLIP2 完整模型，只使用 vision_model
        full_model = SiglipModel.from_pretrained(model_name)
        self.vision_model = full_model.vision_model
        del full_model  # 释放 text model 内存
        
        # 移除 final layernorm 的 affine (类似原版 SigLIP2wNorm)
        self.vision_model.post_layernorm.elementwise_affine = False
        self.vision_model.post_layernorm.weight = None
        self.vision_model.post_layernorm.bias = None
        
        # 添加 LoRA
        add_lora_to_siglip2(self.vision_model, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        _set_lora_enabled(self.vision_model, enable_lora)
        
        self.normalization_layer = SigLIP2ScalingLayer()
        
        # 保存配置
        self.hidden_size = self.vision_model.config.hidden_size
        self.patch_size = self.vision_model.config.patch_size
        self.num_register_tokens = 0  # SigLIP2 没有 register tokens

    def set_lora_enabled(self, enabled: bool):
        _set_lora_enabled(self.vision_model, enabled)

    @contextlib.contextmanager
    def lora_disabled(self):
        with lora_disabled(self.vision_model):
            yield

    def forward(self, x: torch.Tensor):
        x = self.normalization_layer(x)
        outputs = self.vision_model(x, output_hidden_states=True, interpolate_pos_encoding=True)
        return outputs
    
    def get_backbone(self):
        """获取底层 backbone 用于 LoRA 参数访问"""
        return self.vision_model


# =========================
# AutoencoderKL (DINO encoder + CNN decoder)
# =========================

def _grad_norm(loss, params, eps=1e-12):
    grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=True)
    norms = []
    for g in grads:
        if g is not None:
            norms.append(g.detach().norm())
    if len(norms) == 0:
        return torch.tensor(0.0, device=loss.device)
    return torch.norm(torch.stack(norms))  # L2 over per-param norms

class AutoencoderKL(nn.Module):
    def __init__(
        self,
        # Encoder 配置 - 支持多种 encoder
        encoder_type: str = "dinov3",  # "dinov3", "dinov3_vitl" 或 "siglip2"
        dinov3_model_dir: str = "",
        siglip2_model_name: str = "google/siglip2-base-patch16-256",
        
        image_size: int = 256,
        patch_size: int = 16,
        out_channels: int = 3,

        latent_channels: int = 1280,             # encoder hidden size
        target_latent_channels: Optional[int] = None,

        # extra downsample factor (total must be 16 * 2^k)
        spatial_downsample_factor: int = 16,

        # decoder channels, number of upsample stages = len(block_out_channels)-1
        dec_block_out_channels: Tuple[int, ...] = (1280, 1024, 512, 256, 128),
        dec_layers_per_block: int = 2,
        decoder_dropout: float = 0.0,
        gradient_checkpointing: bool = False,

        # VAE behavior
        variational: bool = True,
        kl_weight: float = 1e-6,

        # regularization
        noise_tau: float = 0.0,
        random_masking_channel_ratio: float = 0.0,

        # LoRA
        lora_rank: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        enable_lora: bool = True,

        # VF loss hyper
        vf_margin_cos: float = 0.1,
        vf_margin_dms: float = 0.3,
        vf_max_tokens: int = 32,
        vf_hyper: float = 1.0,
        vf_use_adaptive_weight: bool = True,
        vf_weight_clamp: float = 1e4,
        training_mode: str = "enc_dec",
        denormalize_decoder_output: bool = True,
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.image_size = image_size
        self.patch_size = patch_size
        self.variational = variational
        self.kl_weight = kl_weight
        self.noise_tau = noise_tau
        self.random_masking_channel_ratio = random_masking_channel_ratio

        self.vf_margin_cos = vf_margin_cos
        self.vf_margin_dms = vf_margin_dms
        self.vf_max_tokens = vf_max_tokens
        self.vf_hyper = vf_hyper
        self.vf_use_adaptive_weight = vf_use_adaptive_weight
        self.vf_weight_clamp = vf_weight_clamp

        self.original_latent_channels = latent_channels
        self.target_latent_channels = target_latent_channels if target_latent_channels is not None else latent_channels
        self.use_channel_downsample = (self.target_latent_channels != latent_channels)
        self.denormalize_decoder_output = denormalize_decoder_output

        # 根据 encoder_type 创建不同的 encoder
        if encoder_type == "dinov3" or encoder_type == "dinov3_vitl":
            self.encoder = Encoder2D(
                dinov3_model_dir=dinov3_model_dir,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                enable_lora=enable_lora,
            )
            # 使用 encoder 的实际 hidden_size
            latent_channels = self.encoder.hidden_size
        elif encoder_type == "siglip2":
            self.encoder = SigLIP2Encoder2D(
                model_name=siglip2_model_name,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                enable_lora=enable_lora,
            )
            # 使用 encoder 的实际 hidden_size
            latent_channels = self.encoder.hidden_size
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Supported: 'dinov3', 'dinov3_vitl', 'siglip2'")
        
        # 更新 latent_channels (可能被 encoder 覆盖)
        self.original_latent_channels = latent_channels
        if self.target_latent_channels is None:
            self.target_latent_channels = latent_channels
        self.use_channel_downsample = (self.target_latent_channels != latent_channels)

        # extra spatial downsample after ViT 16x factor
        assert spatial_downsample_factor % 16 == 0, "spatial_downsample_factor must be 16 * 2^k"
        extra_factor = spatial_downsample_factor // 16
        assert extra_factor & (extra_factor - 1) == 0, "only allow 16 * 2^k"
        extra_steps = int(math.log2(extra_factor)) if extra_factor > 1 else 0

        self.latent_downsample_layers = nn.ModuleList([
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=2, padding=1)
            for _ in range(extra_steps)
        ])

        # optional channel downsample
        self.channel_downsample_conv = None
        effective_c = latent_channels
        if self.use_channel_downsample:
            self.channel_downsample_conv = nn.Conv2d(latent_channels, self.target_latent_channels, kernel_size=1, stride=1, padding=0)
            effective_c = self.target_latent_channels

        # VAE moments head: feature -> (mu, logvar)
        self.to_moments = nn.Conv2d(effective_c, 2 * effective_c, kernel_size=1, stride=1, padding=0)

        # Decoder
        self.decoder = Decoder2D(
            in_channels=effective_c,
            out_channels=out_channels,
            block_out_channels=dec_block_out_channels,
            layers_per_block=dec_layers_per_block,
            gradient_checkpointing=gradient_checkpointing,
        )

    def _noising(self, tensor: torch.Tensor) -> torch.Tensor:
        # sigma per-sample
        noise_sigma = self.noise_tau * torch.rand(
            (tensor.shape[0],) + (1,) * (tensor.dim() - 1),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        return tensor + noise_sigma * torch.randn_like(tensor)

    def _extract_patch_map(self, pred) -> torch.Tensor:
        """
        pred.last_hidden_state: [B, 1 + R + N, C] (DINOv3) 或 [B, N, C] (SigLIP2)
        return patch map: [B, C, S, S]
        """
        if self.encoder_type == "dinov3" or self.encoder_type == "dinov3_vitl":
            # DINOv3: 有 CLS token 和 register tokens
            cls_len = 1
            reg_len = self.encoder.num_register_tokens
            tokens = pred.last_hidden_state[:, cls_len + reg_len:, :]  # [B,N,C]
        elif self.encoder_type == "siglip2":
            # SigLIP2: 直接是 patch tokens，无 CLS/register
            tokens = pred.last_hidden_state  # [B,N,C]
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        B, N, C = tokens.shape
        S = int(math.sqrt(N))
        assert S * S == N, f"patch token count N={N} is not square; check image size/patch size."
        feat = tokens.transpose(1, 2).contiguous().view(B, C, S, S)
        return feat

    def get_vf_ref_param(self):
        """获取 VF loss 参考的参数"""
        backbone = self.encoder.get_backbone()
        for n, p in backbone.named_parameters():
            if ("lora_up" in n) or ("lora_down" in n):
                if p.requires_grad:
                    return p
        # fallback: patch embedding weight
        if self.encoder_type == "dinov3" or self.encoder_type == "dinov3_vitl":
            return backbone.embeddings.patch_embeddings.weight
        elif self.encoder_type == "siglip2":
            return backbone.embeddings.patch_embedding.weight
        return None

    def encode_features(self, x: torch.Tensor, use_lora: bool = True) -> torch.Tensor:
        if use_lora:
            pred = self.encoder(x)
        else:
            with torch.no_grad():
                with self.encoder.lora_disabled():
                    pred = self.encoder(x)
        feat = self._extract_patch_map(pred)  # [B,C,S,S] (C=1280)
        return feat

    def compute_vf_loss(self, z_feat: torch.Tensor, f_feat: torch.Tensor) -> torch.Tensor:
        lmcos = vf_marginal_cos_loss(z_feat, f_feat, m1=self.vf_margin_cos)
        lmdms = vf_mdms_loss(z_feat, f_feat, m2=self.vf_margin_dms, max_tokens=self.vf_max_tokens)
        return lmcos + lmdms

    def adaptive_weight(self, loss_rec, loss_vf, params, eps=1e-6):
        n_rec = _grad_norm(loss_rec, params)
        n_vf  = _grad_norm(loss_vf,  params)
        if (n_rec == 0) or (n_vf == 0):
            return torch.tensor(1.0, device=loss_rec.device, dtype=loss_rec.dtype)
        w = (n_rec / (n_vf + eps)).clamp(0.0, self.vf_weight_clamp).detach()
        return w

    def encode(self, x: torch.Tensor,sample_posterior: bool | None = None) -> CausalEncoderOutput:
        feat = self.encode_features(x, use_lora=True)  # [B,1280,S,S]

        # extra downsample
        for conv in self.latent_downsample_layers:
            feat = conv(feat)

        # channel downsample
        if self.channel_downsample_conv is not None:
            feat = self.channel_downsample_conv(feat)

        moments = self.to_moments(feat)  # [B,2C,H,W]

        posterior = DiagonalGaussianDistribution(moments, deterministic=False)

        if sample_posterior is None:
            sample_posterior = self.training

        z = posterior.sample() if sample_posterior else posterior.mode()

        if self.random_masking_channel_ratio > 0.0:
            z, _ = mask_channels(z, mask_ratio=self.random_masking_channel_ratio, channel_dim=1)

        if self.training and self.noise_tau > 0:
            z = self._noising(z)

        return CausalEncoderOutput(latent=z, posterior=posterior)
    

    def _denormalize_output(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.denormalize_decoder_output:
            return tensor

        # imagenet 的 normalization 参数
        device = tensor.device
        imagenet_mean = torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None].to(device)
        imagenet_std = torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None].to(device)
        return (tensor * imagenet_std + imagenet_mean)

    def decode(self, z: torch.Tensor) -> CausalDecoderOutput:
        x = self.decoder(z)
        x = self._denormalize_output(x)
        return CausalDecoderOutput(x)

    def forward(self, x: torch.Tensor) -> CausalAutoencoderOutput:
        enc = self.encode(x)
        dec = self.decode(enc.latent)
        return CausalAutoencoderOutput(dec.sample, enc.latent, enc.posterior)
