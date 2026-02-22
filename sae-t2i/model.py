"""
Text-to-Image DiT model based on DDT architecture.
Adapted from DiTwDDTHead with text conditioning support.
"""
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Mlp
from typing import *

import sys
sys.path.append(".")
from models.rae.stage2.models.model_utils import (
    VisionRotaryEmbeddingFast, 
    RMSNorm, 
    SwiGLUFFN, 
    GaussianFourierEmbedding, 
    get_2d_sincos_pos_embed
)
from text_encoder import Qwen3TextEncoder


def modulate(x, shift, scale):
    """Standard DiT modulation."""
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift


def DDTModulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Applies per-segment modulation to x.
    """
    if shift is None:
        B, Lx, D = x.shape
        _, L, _ = scale.shape
        if Lx % L != 0:
            raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
        repeat = Lx // L
        if repeat != 1:
            scale = scale.repeat_interleave(repeat, dim=1)
        return x * (1 + scale)
    
    B, Lx, D = x.shape
    _, L, _ = shift.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        shift = shift.repeat_interleave(repeat, dim=1)
        scale = scale.repeat_interleave(repeat, dim=1)
    return x * (1 + scale) + shift


def DDTGate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Applies per-segment gating to x.
    """
    B, Lx, D = x.shape
    _, L, _ = gate.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        gate = gate.repeat_interleave(repeat, dim=1)
    return x * gate


#################################################################################
#                           Text Refinement Modules                             #
#################################################################################

class TextRefineAttention(nn.Module):
    """Self-attention for text token refinement."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Use scaled_dot_product_attention for efficiency
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TextRefineBlock(nn.Module):
    """
    Refines text embeddings with self-attention and FFN.
    Uses AdaLN conditioning from timestep.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = TextRefineAttention(hidden_size, num_heads=num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Text features [B, L_text, D]
            c: Condition (timestep embedding) [B, 1, D] or [B, D]
        """
        if len(c.shape) == 2:
            c = c.unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


#################################################################################
#                        Cross-Attention DiT Block                              #
#################################################################################

class CrossAttention(nn.Module):
    """
    Cross-attention where image queries attend to both image and text KV.
    Image: generates Q, K, V
    Text: generates only K, V
    K = concat(K_img, K_text), V = concat(V_img, V_text)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        use_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        
        # Image: Q, K, V
        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Text: K, V only
        self.kv_y = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, y: torch.Tensor, rope=None) -> torch.Tensor:
        """
        Args:
            x: Image features [B, L_img, D]
            y: Text features [B, L_text, D]
            rope: Rotary position embedding for image tokens
        Returns:
            Updated image features [B, L_img, D]
        """
        B, N, C = x.shape
        
        # Image QKV
        qkv_x = self.qkv_x(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]
        q = self.q_norm(q)
        kx = self.k_norm(kx)
        
        # Apply RoPE to image Q and K
        if rope is not None:
            q = rope(q)
            kx = rope(kx)
        
        # Text KV
        kv_y = self.kv_y(y).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        ky, vy = kv_y[0], kv_y[1]
        ky = self.k_norm(ky)
        
        # Concatenate image and text KV
        k = torch.cat([kx, ky], dim=2)  # [B, H, L_img + L_text, head_dim]
        v = torch.cat([vx, vy], dim=2)
        
        # Attention
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SelfAttention(nn.Module):
    """Standard self-attention for image tokens."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        use_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if rope is not None:
            q = rope(q)
            k = rope(k)
        
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class T2IDDTBlock(nn.Module):
    """
    DiT block with cross-attention for text-to-image.
    Supports both self-attention only and cross-attention modes.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = False,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        use_cross_attn: bool = True,
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        
        # Normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
        
        # Attention layer
        if use_cross_attn:
            self.attn = CrossAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=use_qknorm,
                use_rmsnorm=use_rmsnorm,
            )
        else:
            self.attn = SelfAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=use_qknorm,
                use_rmsnorm=use_rmsnorm,
            )
        
        # MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            def approx_gelu(): return nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )
        
        # AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    def forward(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, feat_rope=None):
        """
        Args:
            x: Image features [B, L_img, D]
            y: Text features [B, L_text, D] (used only if use_cross_attn=True)
            c: Condition [B, L_c, D] or [B, D]
            feat_rope: Rotary position embedding
        """
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # Attention
        normed_x = DDTModulate(self.norm1(x), shift_msa, scale_msa)
        if self.use_cross_attn:
            attn_out = self.attn(normed_x, y, rope=feat_rope)
        else:
            attn_out = self.attn(normed_x, rope=feat_rope)
        x = x + DDTGate(attn_out, gate_msa)
        
        # MLP
        x = x + DDTGate(self.mlp(DDTModulate(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp)
        return x


class DDTFinalLayer(nn.Module):
    """The final layer of DDT."""
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, use_rmsnorm: bool = False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = DDTModulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#                         Text-to-Image DDT Model                               #
#################################################################################

class TextEmbedder(nn.Module):
    """Projects text encoder output to model hidden size."""
    def __init__(self, txt_embed_dim: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(txt_embed_dim, hidden_size, bias=True)
        self.norm = RMSNorm(hidden_size)
    
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(y))


class DiTwDDTHead_T2I(nn.Module):
    """
    Text-to-Image DiT model with DDT head.
    
    Architecture:
    1. Text embedding + positional embedding
    2. TextRefineBlocks to refine text features with timestep conditioning
    3. Encoder blocks with cross-attention (image attends to text)
    4. Decoder blocks
    5. Final layer
    """
    def __init__(
        self,
        input_size: int = 1,
        patch_size: Union[list, int] = 1,
        in_channels: int = 768,
        hidden_size: List[int] = [1152, 2048],
        depth: List[int] = [28, 2],
        num_heads: Union[List[int], int] = [16, 16],
        mlp_ratio: float = 4.0,
        txt_embed_dim: int = 3584,  # Qwen3 hidden size
        txt_max_length: int = 128,
        num_text_refine_blocks: int = 4,
        use_qknorm: bool = False,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        use_pos_embed: bool = True,
        cfg_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.cfg_dropout_prob = cfg_dropout_prob
        
        self.encoder_hidden_size = hidden_size[0]
        self.decoder_hidden_size = hidden_size[1]
        self.num_heads = [num_heads, num_heads] if isinstance(num_heads, int) else list(num_heads)
        self.num_encoder_blocks = depth[0]
        self.num_decoder_blocks = depth[1]
        self.num_blocks = depth[0] + depth[1]
        self.num_text_refine_blocks = num_text_refine_blocks
        self.use_rope = use_rope
        self.txt_max_length = txt_max_length
        
        # Patch size
        if isinstance(patch_size, int) or isinstance(patch_size, float):
            patch_size = [patch_size, patch_size]
        assert len(patch_size) == 2
        self.patch_size = patch_size
        self.s_patch_size = patch_size[0]
        self.x_patch_size = patch_size[1]
        
        s_channel_per_token = in_channels * self.s_patch_size * self.s_patch_size
        x_channel_per_token = in_channels * self.x_patch_size * self.x_patch_size
        self.s_channel_per_token = s_channel_per_token
        self.x_channel_per_token = x_channel_per_token
        
        # Patch embedders
        self.x_embedder = PatchEmbed(input_size, self.x_patch_size, x_channel_per_token, self.decoder_hidden_size, bias=True)
        self.s_embedder = PatchEmbed(input_size, self.s_patch_size, s_channel_per_token, self.encoder_hidden_size, bias=True)
        
        # Projector for encoder -> decoder
        self.s_projector = nn.Linear(
            self.encoder_hidden_size, self.decoder_hidden_size
        ) if self.encoder_hidden_size != self.decoder_hidden_size else nn.Identity()
        
        # Timestep embedding
        self.t_embedder = GaussianFourierEmbedding(self.encoder_hidden_size)
        
        # Text embedding
        self.y_embedder = TextEmbedder(txt_embed_dim, self.encoder_hidden_size)
        self.y_pos_embedding = nn.Parameter(
            torch.randn(1, txt_max_length, self.encoder_hidden_size) * 0.02,
            requires_grad=True
        )
        
        # Final layer
        self.final_layer = DDTFinalLayer(
            self.decoder_hidden_size, 1, x_channel_per_token, use_rmsnorm=use_rmsnorm
        )
        
        # Position embedding
        if use_pos_embed:
            num_patches = self.s_embedder.num_patches
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.encoder_hidden_size), 
                requires_grad=False
            )
            self.x_pos_embed = None
        self.use_pos_embed = use_pos_embed
        
        # RoPE
        enc_num_heads = self.num_heads[0]
        dec_num_heads = self.num_heads[1]
        if self.use_rope:
            enc_half_head_dim = self.encoder_hidden_size // enc_num_heads // 2
            hw_seq_len = int(sqrt(self.s_embedder.num_patches))
            self.enc_feat_rope = VisionRotaryEmbeddingFast(
                dim=enc_half_head_dim,
                pt_seq_len=hw_seq_len,
            )
            dec_half_head_dim = self.decoder_hidden_size // dec_num_heads // 2
            hw_seq_len = int(sqrt(self.x_embedder.num_patches))
            self.dec_feat_rope = VisionRotaryEmbeddingFast(
                dim=dec_half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.enc_feat_rope = None
            self.dec_feat_rope = None
        
        # Text refine blocks
        self.text_refine_blocks = nn.ModuleList([
            TextRefineBlock(self.encoder_hidden_size, enc_num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_text_refine_blocks)
        ])
        
        # Encoder blocks (with cross-attention to text)
        self.encoder_blocks = nn.ModuleList([
            T2IDDTBlock(
                self.encoder_hidden_size,
                enc_num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm,
                use_rmsnorm=use_rmsnorm,
                use_swiglu=use_swiglu,
                wo_shift=wo_shift,
                use_cross_attn=True,  # Encoder uses cross-attention
            ) for _ in range(self.num_encoder_blocks)
        ])
        
        # Decoder blocks (self-attention only, conditioned on s)
        self.decoder_blocks = nn.ModuleList([
            T2IDDTBlock(
                self.decoder_hidden_size,
                dec_num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm,
                use_rmsnorm=use_rmsnorm,
                use_swiglu=use_swiglu,
                wo_shift=wo_shift,
                use_cross_attn=False,  # Decoder uses self-attention
            ) for _ in range(self.num_decoder_blocks)
        ])
        
        self.initialize_weights()

    def initialize_weights(self, xavier_uniform_init: bool = False):
        if xavier_uniform_init:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)
        
        # Initialize patch_embed
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        
        # Initialize text embedder
        nn.init.xavier_uniform_(self.y_embedder.proj.weight)
        nn.init.constant_(self.y_embedder.proj.bias, 0)
        
        # Initialize pos_embed
        if self.use_pos_embed:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.s_embedder.num_patches ** 0.5)
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Zero-out adaLN modulation layers
        for block in self.encoder_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.decoder_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.text_refine_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.x_channel_per_token
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, s=None, mask=None):
        """
        Args:
            x: Noisy latent [B, C, H, W]
            t: Timesteps [B]
            y: Text embeddings from frozen text encoder [B, L_text, txt_embed_dim]
            s: Optional pre-computed encoder output
            mask: Optional attention mask
        Returns:
            Predicted x0 [B, C, H, W]
        """
        B = x.shape[0]
        
        # Timestep embedding
        t_emb = self.t_embedder(t)  # [B, D]
        condition = nn.functional.silu(t_emb)  # [B, D]
        
        # Text embedding + positional embedding
        y = self.y_embedder(y)  # [B, L_text, D]
        y = y + self.y_pos_embedding[:, :y.shape[1], :].to(y.dtype)
        
        # Text refine blocks
        for block in self.text_refine_blocks:
            y = block(y, condition)
        
        # Encoder
        if s is None:
            s = self.s_embedder(x)  # [B, L_img, D]
            if self.use_pos_embed:
                s = s + self.pos_embed
            
            # Encoder blocks with cross-attention to text
            for block in self.encoder_blocks:
                s = block(s, y, condition, feat_rope=self.enc_feat_rope)
            
            # Combine timestep with encoder output
            t_broadcast = t_emb.unsqueeze(1).repeat(1, s.shape[1], 1)
            s = nn.functional.silu(t_broadcast + s)
        
        # Project to decoder dimension
        s = self.s_projector(s)
        
        # Decoder
        x = self.x_embedder(x)
        if self.use_pos_embed and self.x_pos_embed is not None:
            x = x + self.x_pos_embed
        
        for block in self.decoder_blocks:
            x = block(x, y=None, c=s, feat_rope=self.dec_feat_rope)
        
        # Final layer
        x = self.final_layer(x, s)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        y_cond: torch.Tensor,
        y_uncond: torch.Tensor,
        cfg_scale: float = 7.5, 
        cfg_interval: Tuple[float, float] = (0.0, 1.0)
    ):
        """
        Forward with classifier-free guidance.
        
        Args:
            x: Noisy latent [B, C, H, W]
            t: Timesteps [B]
            y_cond: Conditional text embeddings [B, L, D]
            y_uncond: Unconditional text embeddings [B, L, D]
            cfg_scale: Guidance scale
            cfg_interval: Time interval for applying CFG
        """
        # Conditional forward
        x0_cond = self.forward(x, t, y_cond)
        
        # Unconditional forward
        x0_uncond = self.forward(x, t, y_uncond)
        
        # Apply CFG within interval
        guid_t_min, guid_t_max = cfg_interval
        t_expanded = t.view(-1, *[1] * (len(x0_cond.shape) - 1))
        
        x0 = torch.where(
            ((t_expanded >= guid_t_min) & (t_expanded <= guid_t_max)),
            x0_uncond + cfg_scale * (x0_cond - x0_uncond),
            x0_cond
        )
        return x0


#################################################################################
#                              Model Variants                                   #
#################################################################################

def DiT_T2I_XXL_2(**kwargs):
    """DiT-XL/2 for text-to-image."""
    return DiTwDDTHead_T2I(
        patch_size=[1, 1],
        hidden_size=[2048, 2048],
        depth=[32, 4],
        num_heads=[32, 32],
        **kwargs
    )


def DiT_T2I_L_2(**kwargs):
    """DiT-L/2 for text-to-image."""
    return DiTwDDTHead_T2I(
        patch_size=[2, 2],
        hidden_size=[1024, 1024],
        depth=[24, 4],
        num_heads=[16, 16],
        **kwargs
    )


def DiT_T2I_B_2(**kwargs):
    """DiT-B/2 for text-to-image."""
    return DiTwDDTHead_T2I(
        patch_size=[2, 2],
        hidden_size=[768, 768],
        depth=[12, 4],
        num_heads=[12, 12],
        **kwargs
    )


if __name__ == "__main__":
    # Test the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = DiT_T2I_XXL_2(
        input_size=16,
        in_channels=768,
        txt_embed_dim=3584,  # Qwen3 hidden size
        txt_max_length=128,
    ).to(device)
    qwen_encoder = Qwen3TextEncoder(weight_path="Qwen/Qwen3-0.5B")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test forward
    batch_size = 2
    x = torch.randn(batch_size, 768, 16, 16, device=device)
    t = torch.rand(batch_size, device=device)
    y = torch.randn(batch_size, 128, 3584, device=device)  # Text embeddings
    
    with torch.no_grad():
        out = model(x, t, y)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
