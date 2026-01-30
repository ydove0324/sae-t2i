# cnn_decoder.py
import math
import contextlib
from copy import deepcopy
from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

# your dinov3 import
import sys
from models.dino_v3.modeling_dino_v3 import DINOv3ViTModel

# ViT decoder 相关导入
from models.rae.stage1.decoders.utils import ViTMAEConfig, ACT2FN

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
    
    If r <= 0, do NOT add any LoRA layers (return original model unchanged).
    """
    if r <= 0:
        # No LoRA: return original model unchanged
        return dino_model
    
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
    
    If r <= 0, do NOT add any LoRA layers (return original model unchanged).
    """
    if r <= 0:
        # No LoRA: return original model unchanged
        return siglip_vision_model
    
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
# ViT Decoder Components (for ViT-XL decoder)
# =========================

def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """Create 2D sin/cos positional embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class ViTSelfAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}."
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class ViTSelfOutput(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ViTAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class ViTIntermediate(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ViTOutput(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class ViTLayer(nn.Module):
    """Transformer block for ViT decoder."""
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs
        return outputs


class ViTXLDecoder(nn.Module):
    """
    ViT-XL Decoder for AutoencoderKL.
    
    可配置的 ViT decoder，默认使用 XL 配置:
    - hidden_size: 1024
    - num_layers: 24
    - num_heads: 16
    - intermediate_size: 4096
    """
    def __init__(
        self,
        encoder_hidden_size: int = 1280,
        decoder_hidden_size: int = 1024,
        decoder_num_layers: int = 24,
        decoder_num_heads: int = 16,
        decoder_intermediate_size: int = 4096,
        image_size: int = 256,
        patch_size: int = 16,
        out_channels: int = 3,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_patches = (image_size // patch_size) ** 2
        self.gradient_checkpointing = gradient_checkpointing
        
        # Encoder to decoder projection
        self.decoder_embed = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=True)
        
        # Positional embedding (fixed sin-cos)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_hidden_size), requires_grad=False
        )
        
        # Trainable CLS token for decoder
        self.trainable_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))
        
        # Decoder config
        decoder_config = ViTMAEConfig(
            hidden_size=decoder_hidden_size,
            num_hidden_layers=decoder_num_layers,
            num_attention_heads=decoder_num_heads,
            intermediate_size=decoder_intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=out_channels,
            qkv_bias=True,
        )
        
        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList(
            [ViTLayer(decoder_config) for _ in range(decoder_num_layers)]
        )
        
        # Output normalization and projection
        self.decoder_norm = nn.LayerNorm(decoder_hidden_size, eps=decoder_config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            decoder_hidden_size, patch_size ** 2 * out_channels, bias=True
        )
        
        self.decoder_config = decoder_config
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize position embeddings with sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], 
            int(self.num_patches ** 0.5), 
            add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        # Initialize CLS token
        nn.init.normal_(self.trainable_cls_token, std=0.02)
    
    def interpolate_pos_encoding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Interpolate position encodings for different resolutions."""
        embeddings_positions = embeddings.shape[1] - 1
        num_positions = self.decoder_pos_embed.shape[1] - 1

        if embeddings_positions == num_positions:
            return self.decoder_pos_embed

        class_pos_embed = self.decoder_pos_embed[:, 0, :]
        patch_pos_embed = self.decoder_pos_embed[:, 1:, :]
        dim = self.decoder_pos_embed.shape[-1]

        patch_pos_embed = patch_pos_embed.reshape(1, 1, -1, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(1, embeddings_positions / num_positions),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    def interpolate_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate latent to match expected num_patches."""
        b, l, c = x.shape
        if l == self.num_patches:
            return x
        h, w = int(l ** 0.5), int(l ** 0.5)
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        target_size = (int(self.num_patches ** 0.5), int(self.num_patches ** 0.5))
        x = nn.functional.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1).contiguous().view(b, self.num_patches, c)
        return x
    
    def unpatchify(self, patchified_pixel_values: torch.Tensor) -> torch.Tensor:
        """Convert patchified representation back to image."""
        patch_size = self.patch_size
        num_channels = self.out_channels
        num_patches_h = self.image_size // patch_size
        num_patches_w = self.image_size // patch_size
        
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size, num_patches_h, num_patches_w, patch_size, patch_size, num_channels
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size, num_channels, num_patches_h * patch_size, num_patches_w * patch_size
        )
        return pixel_values
    
    def _gradient_checkpointing_func(self, func, *args, **kwargs):
        return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, **kwargs)
    
    def forward(self, hidden_states: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, C, H, W] latent from encoder
            interpolate_pos_encoding: whether to interpolate position encodings
        Returns:
            reconstructed image [B, C, image_size, image_size]
        """
        # Convert from [B, C, H, W] to [B, N, C]
        B, C, H, W = hidden_states.shape
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # [B, N, C]
        
        # Embed tokens
        x = self.decoder_embed(hidden_states)
        
        # Interpolate latent if needed
        x = self.interpolate_latent(x)
        
        # Add CLS token
        cls_token = self.trainable_cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add position embeddings
        if interpolate_pos_encoding:
            decoder_pos_embed = self.interpolate_pos_encoding(x)
        else:
            decoder_pos_embed = self.decoder_pos_embed
        hidden_states = x + decoder_pos_embed
        
        # Apply transformer layers
        for layer_module in self.decoder_layers:
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module,
                    hidden_states,
                    None,
                    False,
                )
            else:
                layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=False)
            hidden_states = layer_outputs[0]
        
        # Normalize and predict
        hidden_states = self.decoder_norm(hidden_states)
        logits = self.decoder_pred(hidden_states)
        
        # Remove CLS token and unpatchify
        logits = logits[:, 1:, :]
        pixel_values = self.unpatchify(logits)
        
        return pixel_values


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

        # Decoder 配置 - 支持 CNN 或 ViT decoder
        decoder_type: str = "cnn_decoder",  # "cnn_decoder" 或 "vit_decoder"
        
        # CNN decoder channels, number of upsample stages = len(block_out_channels)-1
        dec_block_out_channels: Tuple[int, ...] = (1280, 1024, 512, 256, 128),
        dec_layers_per_block: int = 2,
        decoder_dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        
        # ViT decoder 配置 (当 decoder_type="vit_decoder" 时使用)
        vit_decoder_hidden_size: int = 1024,      # XL: 1024, L: 768, B: 512
        vit_decoder_num_layers: int = 24,         # XL: 24, L: 16, B: 8
        vit_decoder_num_heads: int = 16,          # XL: 16, L: 12, B: 8
        vit_decoder_intermediate_size: int = 4096, # XL: 4096, L: 3072, B: 2048

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
        
        # Skip to_moments layer (for old checkpoints without VAE reparameterization)
        skip_to_moments: bool = False,
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
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
        # If skip_to_moments=True, don't create this layer (for old checkpoints)
        self.skip_to_moments = skip_to_moments
        if skip_to_moments:
            self.to_moments = None
        else:
            self.to_moments = nn.Conv2d(effective_c, 2 * effective_c, kernel_size=1, stride=1, padding=0)

        # Decoder - 支持 CNN 或 ViT
        if decoder_type == "cnn_decoder":
            self.decoder = Decoder2D(
                in_channels=effective_c,
                out_channels=out_channels,
                block_out_channels=dec_block_out_channels,
                layers_per_block=dec_layers_per_block,
                gradient_checkpointing=gradient_checkpointing,
            )
        elif decoder_type == "vit_decoder":
            self.decoder = ViTXLDecoder(
                encoder_hidden_size=effective_c,
                decoder_hidden_size=vit_decoder_hidden_size,
                decoder_num_layers=vit_decoder_num_layers,
                decoder_num_heads=vit_decoder_num_heads,
                decoder_intermediate_size=vit_decoder_intermediate_size,
                image_size=image_size,
                patch_size=patch_size,
                out_channels=out_channels,
                dropout=decoder_dropout,
                gradient_checkpointing=gradient_checkpointing,
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}. Supported: 'cnn_decoder', 'vit_decoder'")

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

    def encode(self, x: torch.Tensor, sample_posterior: bool | None = None) -> CausalEncoderOutput:
        feat = self.encode_features(x, use_lora=True)  # [B,1280,S,S]

        # extra downsample
        for conv in self.latent_downsample_layers:
            feat = conv(feat)

        # channel downsample
        if self.channel_downsample_conv is not None:
            feat = self.channel_downsample_conv(feat)

        # Skip to_moments: directly return feature without VAE reparameterization
        if self.to_moments is None:
            # No VAE reparameterization, just return the feature as latent
            if self.random_masking_channel_ratio > 0.0:
                feat, _ = mask_channels(feat, mask_ratio=self.random_masking_channel_ratio, channel_dim=1)
            if self.training and self.noise_tau > 0:
                feat = self._noising(feat)
            return CausalEncoderOutput(latent=feat, posterior=None)

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
