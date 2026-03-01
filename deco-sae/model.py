"""
DecoSAE with pretrained vision encoder + configurable decoder.

- Keep original encoder loading path (DINOv3 / DINOv2 / SigLIP2 + optional LoRA).
- Support multiple decoder types: Flow Matching or ViT decoder.
- Add an optional HF (high-frequency) residual branch.
"""

import sys
import os
import math
import json
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block
from diffusers import FluxPipeline

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append("/cpfs01/huangxu/SAE")
from cnn_decoder import (
    Encoder2D,
    SigLIP2Encoder2D,
    DINOv2Encoder2D,
    Qwen3ViTEncoder2D,
    CausalAutoencoderOutput,
    CausalEncoderOutput,
    CausalDecoderOutput,
    vf_marginal_cos_loss,
    vf_mdms_loss,
    _grad_norm,
    mask_channels,
    ViTXLDecoder,  # ViT decoder support
)

try:
    from models.rae.utils.metrics_utils import LPIPSLoss
except ImportError:
    LPIPSLoss = None

try:
    from train_vae.dinodisc import vanilla_g_loss
except ImportError:
    vanilla_g_loss = None


# =========================================================================
# 1. Flow Matching components
# =========================================================================

class TimestepEmbedder(nn.Module):
    """Standard sinusoidal timestep embedding for Flow Matching."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LearnablePatchPosEmbed(nn.Module):
    """Learnable patch-level positional embeddings."""

    def __init__(self, num_patches: int, hidden_size_output: int):
        super().__init__()
        self.num_patches = num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size_output))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, batch_size: int, num_patches: int, dtype: torch.dtype) -> torch.Tensor:
        if num_patches != self.num_patches:
            raise ValueError(
                f"num_patches mismatch: got {num_patches}, expected {self.num_patches}"
            )
        pos = self.pos_embed.to(dtype=dtype).expand(batch_size, -1, -1)
        return pos.reshape(batch_size * num_patches, -1)


class ResBlockAdaLN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        x_norm = self.in_ln(x)
        x_norm = x_norm * (1 + scale) + shift
        return x + gate * self.mlp(x_norm)


class FlowPatchDecoder(nn.Module):
    """
    Flow-matching decoder on flattened patch vectors.

    x_t:      [B*N, C*P^2]
    pos_emb:  [B*N, C]
    c_global: [B*N, C_cond]
    """

    def __init__(
        self,
        in_channels: int = 768,
        model_channels: int = 256,
        out_channels: int = 768,
        cond_channels: int = 1152,
        num_res_blocks: int = 12,
    ):
        super().__init__()
        self.x_proj = nn.Linear(in_channels, model_channels)
        self.cond_proj = nn.Linear(cond_channels, model_channels)
        self.blocks = nn.ModuleList([ResBlockAdaLN(model_channels) for _ in range(num_res_blocks)])
        self.final_norm = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(model_channels, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)

    def forward(self, x_t: torch.Tensor, pos_emb: torch.Tensor, c_global: torch.Tensor) -> torch.Tensor:
        h = self.x_proj(x_t)
        h = h + pos_emb
        c = self.cond_proj(c_global)
        for block in self.blocks:
            h = block(h, c)
        h = self.final_norm(h)
        return self.final_linear(h)


class HFResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels, eps=1e-6)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels, eps=1e-6)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h



class HFEncoder(nn.Module):
    """
    Extract patch-aligned HF tokens from image space.
    Reference: diffusers VAE Encoder with down_blocks + mid_block (attention).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
        patch_size: int = 16,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        block_out_channels: Tuple[int, ...] = (64, 128, 256),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.layers_per_block = layers_per_block

        # Calculate total downsample factor from down_blocks
        # Each DownEncoderBlock2D with add_downsample=True does 2x downsample
        # Last block has add_downsample=False, so N blocks -> (N-1) downsamples
        num_downsample = len(down_block_types) - 1
        self.down_factor = 2 ** num_downsample  # e.g., 4 blocks -> 3 downsample -> 8x
        
        # conv_in
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # down_blocks
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        # mid_block with attention
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        # conv_out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv_in
        h = self.conv_in(x)

        # down_blocks
        for down_block in self.down_blocks:
            h = down_block(h)

        # mid_block
        h = self.mid_block(h)

        # conv_out
        h = self.conv_norm_out(h)
        h = self.conv_act(h)
        h = self.conv_out(h)

        # Final pooling to match patch_size
        # After down_blocks, spatial is reduced by down_factor
        # We need additional pooling to reach patch_size
        current_stride = self.down_factor
        if self.patch_size > current_stride:
            pool_size = self.patch_size // current_stride
            h = F.avg_pool2d(h, kernel_size=pool_size, stride=pool_size)
        
        return h
    
    @classmethod
    def from_config(cls, config_path: str, patch_size: int = 14):
        """Load HFEncoder from a JSON config file."""
        with open(config_path, "r") as f:
            config = json.load(f)
        # Convert lists to tuples for type compatibility
        if "down_block_types" in config:
            config["down_block_types"] = tuple(config["down_block_types"])
        if "block_out_channels" in config:
            config["block_out_channels"] = tuple(config["block_out_channels"])
        config["patch_size"] = patch_size
        return cls(**config)


# =========================================================================
# 2. Main model
# =========================================================================

class DecoSAE(nn.Module):
    def __init__(
        self,
        # ---- Encoder config ----
        encoder_type: str = "dinov3",
        dinov3_model_dir: str = "",
        siglip2_model_name: str = "google/siglip2-base-patch16-256",
        dinov2_model_name: str = "facebook/dinov2-with-registers-base",
        qwen3_vit_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        # ---- Image config ----
        image_size: int = 256,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_size_x: int = 64,
        # ---- Decoder type ----
        decoder_type: str = "flow_matching",  # "flow_matching" or "vit_decoder"
        # ---- Flow decoder config ----
        hidden_size: int = 1152,
        num_decoder_blocks: int = 12,
        nerf_max_freqs: int = 8,
        flow_steps: int = 25,
        time_dim: int = 256,
        # ---- ViT decoder config (when decoder_type="vit_decoder") ----
        vit_decoder_hidden_size: int = 1024,      # XL: 1024, L: 768, B: 512
        vit_decoder_num_layers: int = 24,         # XL: 24, L: 16, B: 8
        vit_decoder_num_heads: int = 16,          # XL: 16, L: 12, B: 8
        vit_decoder_intermediate_size: int = 4096, # XL: 4096, L: 3072, B: 2048
        vit_decoder_dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        # ---- HF branch ----
        enable_hf_branch: bool = True,
        hf_dim: int = 256,
        hf_encoder_config_path: Optional[str] = None,
        hf_dropout_prob: float = 0.4,
        hf_noise_std: float = 0.1,
        hf_noise_alpha_schedule: str = "alpha_one",
        hf_loss_weight: float = 0.1,
        recon_l2_weight: float = 1.0,
        recon_l1_weight: float = 0.0,
        recon_lpips_weight: float = 0.0,
        recon_gan_weight: float = 0.0,
        # ---- LoRA ----
        lora_rank: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        enable_lora: bool = True,
        # ---- Latent / VAE ----
        target_latent_channels: Optional[int] = None,
        variational: bool = False,
        kl_weight: float = 1e-6,
        skip_to_moments: bool = True,
        # ---- Regularization ----
        noise_tau: float = 0.0,
        random_masking_channel_ratio: float = 0.0,
        # ---- VF loss ----
        vf_margin_cos: float = 0.1,
        vf_margin_dms: float = 0.3,
        vf_max_tokens: int = 32,
        vf_hyper: float = 1.0,
        vf_use_adaptive_weight: bool = True,
        vf_weight_clamp: float = 1e4,

        # ---- Output ----
        denormalize_decoder_output: bool = True,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.hidden_size_x = hidden_size_x
        self.variational = variational
        self.kl_weight = kl_weight
        self.noise_tau = noise_tau
        self.random_masking_channel_ratio = random_masking_channel_ratio
        self.denormalize_decoder_output = denormalize_decoder_output
        self.skip_to_moments = skip_to_moments
        self.flow_steps = flow_steps
        self.time_dim = time_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.enable_hf_branch = enable_hf_branch
        self.hf_dropout_prob = hf_dropout_prob
        self.hf_noise_std = hf_noise_std
        self.hf_noise_alpha_schedule = str(hf_noise_alpha_schedule).lower()
        if self.hf_noise_alpha_schedule not in {"alpha_one", "sqrt_1m_sigma2"}:
            raise ValueError(
                "hf_noise_alpha_schedule must be one of {'alpha_one', 'sqrt_1m_sigma2'}, "
                f"got: {hf_noise_alpha_schedule}"
            )
        self.hf_loss_weight = hf_loss_weight
        self.recon_l2_weight = recon_l2_weight
        self.recon_l1_weight = recon_l1_weight
        self.recon_lpips_weight = recon_lpips_weight
        self.recon_gan_weight = recon_gan_weight

        self.vf_margin_cos = vf_margin_cos
        self.vf_margin_dms = vf_margin_dms
        self.vf_max_tokens = vf_max_tokens
        self.vf_hyper = vf_hyper
        self.vf_use_adaptive_weight = vf_use_adaptive_weight
        self.vf_weight_clamp = vf_weight_clamp
        self._lpips_loss_fn: Optional[nn.Module] = None
        
        # Store ViT decoder config
        self.vit_decoder_hidden_size = vit_decoder_hidden_size
        self.vit_decoder_num_layers = vit_decoder_num_layers
        self.vit_decoder_num_heads = vit_decoder_num_heads
        self.vit_decoder_intermediate_size = vit_decoder_intermediate_size
        self.vit_decoder_dropout = vit_decoder_dropout

        # 1) Keep original encoder loading logic.
        if encoder_type in ("dinov3", "dinov3_vitl"):
            self.encoder = Encoder2D(
                dinov3_model_dir=dinov3_model_dir,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                enable_lora=enable_lora,
            )
        elif encoder_type == "siglip2":
            self.encoder = SigLIP2Encoder2D(
                model_name=siglip2_model_name,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                enable_lora=enable_lora,
            )
        elif encoder_type == "dinov2":
            self.encoder = DINOv2Encoder2D(
                model_name=dinov2_model_name,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                enable_lora=enable_lora,
                normalize=True,
            )
        elif encoder_type == "qwen3_vit":
            self.encoder = Qwen3ViTEncoder2D(
                model_name=qwen3_vit_model_name,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                enable_lora=enable_lora,
            )
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}. "
                f"Supported: 'dinov3', 'dinov3_vitl', 'siglip2', 'dinov2', 'qwen3_vit'"
            )

        encoder_hidden_size = self.encoder.hidden_size
        encoder_patch_size = self.encoder.patch_size
        self.decode_patch_size = 16 if encoder_type == "dinov2" else (32 if encoder_type == "qwen3_vit" else encoder_patch_size)
        self.num_patches_h = image_size // self.decode_patch_size
        self.num_patches_w = image_size // self.decode_patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # 2) Optional latent processing.
        self.original_latent_channels = encoder_hidden_size
        self.target_latent_channels = (
            target_latent_channels if target_latent_channels is not None else encoder_hidden_size
        )
        self.use_channel_downsample = self.target_latent_channels != encoder_hidden_size
        if self.use_channel_downsample:
            self.channel_downsample_conv = nn.Conv2d(
                encoder_hidden_size,
                self.target_latent_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.channel_downsample_conv = None

        effective_c = self.target_latent_channels
        self.semantic_channels = effective_c
        if skip_to_moments or not variational:
            self.to_moments = None
        else:
            self.to_moments = nn.Conv2d(effective_c, 2 * effective_c, kernel_size=1)

        # 3) Condition fusion (Semantic + HF -> hidden_size).
        self.hf_dim = hf_dim
        if hf_encoder_config_path is not None:
            self.hf_encoder = HFEncoder.from_config(hf_encoder_config_path, patch_size=self.decode_patch_size)
            # Override hf_dim from config if specified
            self.hf_dim = self.hf_encoder.conv_out.out_channels
        else:
            self.hf_encoder = HFEncoder(
                in_channels=in_channels,
                out_channels=self.hf_dim,
                patch_size=self.decode_patch_size,
            )
        self.fused_norm = nn.LayerNorm(self.semantic_channels + self.hf_dim, elementwise_affine=False, eps=1e-6)
        self.fused_proj = nn.Linear(self.semantic_channels + self.hf_dim, hidden_size, bias=True)

        # 4) Decoder - support Flow Matching or ViT decoder
        if decoder_type == "flow_matching":
            print("Using Flow Matching decoder")
            # Flow matching modules
            self.t_embedder = TimestepEmbedder(time_dim)
            self.coord_embedder = LearnablePatchPosEmbed(self.num_patches, hidden_size_x)
            patch_dim = out_channels * (self.decode_patch_size ** 2)
            self.decoder = FlowPatchDecoder(
                in_channels=patch_dim,
                model_channels=hidden_size_x,
                out_channels=patch_dim,
                cond_channels=hidden_size + time_dim,
                num_res_blocks=num_decoder_blocks,
            )
        elif decoder_type == "vit_decoder":
            print("Using ViT decoder")
            # ViT decoder (directly from fused condition to image)
            # Input: fused condition [B, N, hidden_size] -> [B, hidden_size, H, W]
            self.t_embedder = None
            self.coord_embedder = None
            self.decoder = ViTXLDecoder(
                encoder_hidden_size=hidden_size,
                decoder_hidden_size=vit_decoder_hidden_size,
                decoder_num_layers=vit_decoder_num_layers,
                decoder_num_heads=vit_decoder_num_heads,
                decoder_intermediate_size=vit_decoder_intermediate_size,
                image_size=image_size,
                patch_size=self.decode_patch_size,
                out_channels=out_channels,
                dropout=vit_decoder_dropout,
                gradient_checkpointing=gradient_checkpointing,
            )
        else:
            raise ValueError(
                f"Unknown decoder_type: {decoder_type}. "
                f"Supported: 'flow_matching', 'vit_decoder'"
            )

    # ------------------------------------------------------------------
    # Encoder utilities
    # ------------------------------------------------------------------

    def _extract_patch_tokens(self, pred) -> torch.Tensor:
        if self.encoder_type in ("dinov3", "dinov3_vitl"):
            cls_len = 1
            reg_len = self.encoder.num_register_tokens
            tokens = pred.last_hidden_state[:, cls_len + reg_len :, :]
        elif self.encoder_type == "siglip2":
            tokens = pred.last_hidden_state
        elif self.encoder_type == "dinov2":
            cls_len = 1
            reg_len = self.encoder.num_register_tokens
            tokens = pred.last_hidden_state[:, cls_len + reg_len :, :]
        elif self.encoder_type == "qwen3_vit":
            tokens = pred.last_hidden_state
            if getattr(self.encoder, "has_cls_token", True) and tokens.shape[1] > 1:
                tokens = tokens[:, 1:, :]
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")
        return tokens

    def _tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, channels = tokens.shape
        side = int(math.sqrt(num_tokens))
        assert side * side == num_tokens, f"Token count {num_tokens} is not a perfect square"
        return tokens.transpose(1, 2).contiguous().view(bsz, channels, side, side)

    def encode_features(self, x: torch.Tensor, use_lora: bool = True) -> torch.Tensor:
        if use_lora:
            pred = self.encoder(x)
        else:
            with torch.no_grad():
                with self.encoder.lora_disabled():
                    pred = self.encoder(x)
        return self._extract_patch_tokens(pred)

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def _noising(self, tensor: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand(
            (tensor.shape[0],) + (1,) * (tensor.dim() - 1),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        return tensor + noise_sigma * torch.randn_like(tensor)

    def _encoder_semantic(self, z: torch.Tensor) -> torch.Tensor:
        bsz, _, s_h, s_w = z.shape
        num_patches = s_h * s_w
        return z.flatten(2).transpose(1, 2).contiguous().reshape(bsz, num_patches, self.semantic_channels)

    def _get_patch_coords(self, batch_size: int, num_patches: int, device, dtype) -> torch.Tensor:
        return self.coord_embedder(batch_size, num_patches, dtype)

    def _extract_hf_tokens(
        self,
        x_img: Optional[torch.Tensor],
        bsz: int,
        s_h: int,
        s_w: int,
        num_patches: int,
        device: torch.device,
        dtype: torch.dtype,
        force_drop_hf: bool = False,
    ) -> torch.Tensor:
        if (not self.enable_hf_branch) or (self.hf_dim <= 0):
            return torch.zeros(bsz, num_patches, self.hf_dim, device=device, dtype=dtype)

        if x_img is not None:
            hf_feat = self.hf_encoder(x_img)
            if hf_feat.shape[-2:] != (s_h, s_w):
                hf_feat = F.interpolate(hf_feat, size=(s_h, s_w), mode="bilinear", align_corners=False)
            s_hf = hf_feat.flatten(2).transpose(1, 2).contiguous()
        else:
            s_hf = torch.zeros(bsz, num_patches, self.hf_dim, device=device, dtype=dtype)

        if self.training:
            if force_drop_hf:
                keep_mask = torch.zeros((bsz, 1, 1), device=device, dtype=dtype)
            else:
                keep_prob = max(0.0, min(1.0, 1.0 - self.hf_dropout_prob))
                keep_mask = torch.bernoulli(
                    torch.full((bsz, 1, 1), keep_prob, device=device, dtype=dtype)
                )       # 不是随机 mask 掉 40% 的 token，而是有 40% 的概率将整个样本的所有 HF token 全部 dropout。
            s_hf = s_hf * keep_mask
            if self.hf_noise_std > 0:
                # Diffusion-style perturbation with per-sample sigma in [0, hf_noise_std).
                sigma = torch.rand((bsz, 1, 1), device=device, dtype=dtype) * self.hf_noise_std
                if self.hf_noise_alpha_schedule == "alpha_one":
                    alpha = torch.ones_like(sigma)
                else:
                    alpha = torch.sqrt(torch.clamp(1.0 - sigma ** 2, min=0.0))
                s_hf = s_hf * alpha + torch.randn_like(s_hf) * sigma
        elif force_drop_hf:
            s_hf = torch.zeros_like(s_hf)

        return s_hf

    def _infer_latent(self, x: torch.Tensor, sample_posterior: Optional[bool] = None) -> CausalEncoderOutput:
        tokens = self.encode_features(x, use_lora=True)
        feat = self._tokens_to_spatial(tokens)

        if self.channel_downsample_conv is not None:
            feat = self.channel_downsample_conv(feat)

        if self.to_moments is not None:
            moments = self.to_moments(feat)
            posterior = DiagonalGaussianDistribution(moments, deterministic=False)
            if sample_posterior is None:
                sample_posterior = self.training
            z = posterior.sample() if sample_posterior else posterior.mode()
        else:
            z = feat
            posterior = None

        if self.random_masking_channel_ratio > 0.0:
            z, _ = mask_channels(z, mask_ratio=self.random_masking_channel_ratio, channel_dim=1)
        if self.training and self.noise_tau > 0:
            z = self._noising(z)
        return CausalEncoderOutput(latent=z, posterior=posterior)

    def _patches_to_image(
        self,
        x_patch: torch.Tensor,
        batch_size: int,
        num_patches: int,
        patch_size: int,
    ) -> torch.Tensor:
        patch_dim = x_patch.shape[-1]
        pixels_per_channel = patch_size ** 2
        if patch_dim % pixels_per_channel != 0:
            raise ValueError(
                f"Invalid patch_dim={patch_dim} for patch_size={patch_size}. "
                f"Expected divisible by patch_size**2={pixels_per_channel}."
            )
        x_out = x_patch.reshape(batch_size, num_patches, patch_dim)
        x_out = x_out.transpose(1, 2).contiguous()
        side = int(math.sqrt(num_patches)) * patch_size
        return F.fold(
            x_out,
            output_size=(side, side),
            kernel_size=patch_size,
            stride=patch_size,
        )

    def _compute_reconstruction_loss(
        self,
        pred_patch: torch.Tensor,
        target_patch: torch.Tensor,
        pred_img: torch.Tensor,
        target_img: torch.Tensor,
        lpips_loss_fn: Optional[nn.Module] = None,
        gan_loss: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        l2_loss = F.mse_loss(pred_patch, target_patch)
        loss = self.recon_l2_weight * l2_loss
        
        l1_loss = torch.tensor(0.0, device=pred_patch.device)
        if self.recon_l1_weight > 0:
            l1_loss = F.l1_loss(pred_patch, target_patch)
            loss = loss + self.recon_l1_weight * l1_loss
        
        lpips_loss = torch.tensor(0.0, device=pred_patch.device)
        if self.recon_lpips_weight > 0 and lpips_loss_fn is not None:
            lpips_loss = lpips_loss_fn(target_img, pred_img)
            loss = loss + self.recon_lpips_weight * lpips_loss
        
        if self.recon_gan_weight > 0 and gan_loss is not None:
            loss = loss + self.recon_gan_weight * gan_loss
        
        if return_dict:
            return {
                "loss": loss,
                "l2_loss": l2_loss,
                "l1_loss": l1_loss,
                "lpips_loss": lpips_loss,
                "gan_loss": gan_loss if gan_loss is not None else torch.tensor(0.0, device=pred_patch.device),
            }
        return loss

    def _get_lpips_loss(self, device: torch.device) -> nn.Module:
        if LPIPSLoss is None:
            raise RuntimeError(
                "LPIPSLoss is unavailable. Please install lpips and ensure "
                "models.rae.utils.metrics_utils is importable."
            )
        if self._lpips_loss_fn is None:
            self._lpips_loss_fn = LPIPSLoss(device=device)
        else:
            self._lpips_loss_fn = self._lpips_loss_fn.to(device)
        return self._lpips_loss_fn

    def _compute_gan_generator_loss(
        self,
        pred_img: torch.Tensor,
        discriminator: Optional[nn.Module] = None,
        diffaug: Optional[object] = None,
    ) -> Optional[torch.Tensor]:
        if self.recon_gan_weight <= 0:
            return None
        if discriminator is None:
            return None
        if vanilla_g_loss is None:
            raise RuntimeError(
                "vanilla_g_loss is unavailable. Please ensure train_vae/dinodisc.py is importable."
            )
        recon_for_disc = pred_img.clamp(-1.0, 1.0)
        if diffaug is not None and hasattr(diffaug, "aug"):
            recon_for_disc = diffaug.aug(recon_for_disc)
        logits_fake = discriminator(recon_for_disc)
        return vanilla_g_loss(logits_fake)

    def encode(
        self,
        z: torch.Tensor,
        x_img: Optional[torch.Tensor] = None,
        force_drop_hf: bool = False,
    ) -> torch.Tensor:
        s_sem = self._encoder_semantic(z)
        bsz, num_patches, _ = s_sem.shape
        s_h, s_w = z.shape[-2:]
        s_hf = self._extract_hf_tokens(
            x_img=x_img,
            bsz=bsz,
            s_h=s_h,
            s_w=s_w,
            num_patches=num_patches,
            device=z.device,
            dtype=s_sem.dtype,
            force_drop_hf=force_drop_hf,
        )
        return self.fused_norm(torch.cat([s_sem, s_hf], dim=-1))

    def forward_loss(
        self,
        x_clean: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        lpips_loss_fn: Optional[nn.Module] = None,
        gan_loss: Optional[torch.Tensor] = None,
        discriminator: Optional[nn.Module] = None,
        diffaug: Optional[object] = None,
        return_dict: bool = False,
    ):
        """
        Training loss for both Flow Matching and ViT decoder.

        x_clean: [B, 3, H, W]
        return_dict: if True, return a dict with all loss components
        """
        device = x_clean.device
        dtype = x_clean.dtype
        bsz = x_clean.shape[0]

        if z is None:
            z = self._infer_latent(x_clean).latent

        enc_cond = self.encode(z, x_img=x_clean, force_drop_hf=False)
        dec_cond = self.fused_proj(enc_cond)
        num_patches = dec_cond.shape[1]
        patch_size = self.decode_patch_size

        if self.decoder_type == "flow_matching":
            # Flow matching loss
            x_data_flat = F.unfold(x_clean, kernel_size=patch_size, stride=patch_size)
            x_data_flat = x_data_flat.transpose(1, 2)
            patch_dim = self.out_channels * patch_size ** 2

            t = torch.rand(bsz, device=device, dtype=dtype)
            x_noise_flat = torch.randn_like(x_data_flat)
            t_expand = t.view(bsz, 1, 1)
            x_t = t_expand * x_data_flat + (1 - t_expand) * x_noise_flat
            target_v = x_data_flat - x_noise_flat

            x_t_input = x_t.reshape(-1, patch_dim)
            s_input = dec_cond.reshape(-1, self.hidden_size)
            t_emb = self.t_embedder(t).to(dtype=dtype)
            t_input = t_emb.repeat_interleave(num_patches, dim=0)
            cond_input = torch.cat([s_input, t_input], dim=-1)
            pos_emb = self._get_patch_coords(bsz, num_patches, device, dtype)

            pred_v = self.decoder(x_t_input, pos_emb, cond_input)
            target_v_flat = target_v.reshape(-1, patch_size ** 2 * self.out_channels)
            loss_flow = F.mse_loss(pred_v, target_v_flat)

            # x_1 = x_t + (1 - t) * v
            t_bn = t.repeat_interleave(num_patches).view(-1, 1)
            pred_x1 = x_t_input + (1 - t_bn) * pred_v
            target_x1 = x_data_flat.reshape(-1, patch_size ** 2 * self.out_channels)

            pred_img = self._patches_to_image(pred_x1, bsz, num_patches, patch_size)
            
        elif self.decoder_type == "vit_decoder":
            # ViT decoder loss (direct reconstruction)
            loss_flow = torch.tensor(0.0, device=device, dtype=dtype)
            
            # Convert dec_cond [B, N, C] -> [B, C, H, W]
            side = int(math.sqrt(num_patches))
            dec_cond_spatial = dec_cond.transpose(1, 2).contiguous().view(bsz, self.hidden_size, side, side)
            
            # Decode directly
            pred_img = self.decoder(dec_cond_spatial)
            
            # For reconstruction loss computation, flatten to patch format
            x_data_flat = F.unfold(x_clean, kernel_size=patch_size, stride=patch_size)
            x_data_flat = x_data_flat.transpose(1, 2)
            target_x1 = x_data_flat.reshape(-1, patch_size ** 2 * self.out_channels)
            
            pred_flat = F.unfold(pred_img, kernel_size=patch_size, stride=patch_size)
            pred_flat = pred_flat.transpose(1, 2)
            pred_x1 = pred_flat.reshape(-1, patch_size ** 2 * self.out_channels)
        else:
            raise ValueError(f"Unknown decoder_type: {self.decoder_type}")

        if self.recon_lpips_weight > 0 and lpips_loss_fn is None:
            lpips_loss_fn = self._get_lpips_loss(device=x_clean.device)
        if gan_loss is None:
            gan_loss = self._compute_gan_generator_loss(
                pred_img=pred_img,
                discriminator=discriminator,
                diffaug=diffaug,
            )
        
        if return_dict:
            recon_dict = self._compute_reconstruction_loss(
                pred_patch=pred_x1,
                target_patch=target_x1,
                pred_img=pred_img,
                target_img=x_clean,
                lpips_loss_fn=lpips_loss_fn,
                gan_loss=gan_loss,
                return_dict=True,
            )
            total_loss = loss_flow + recon_dict["loss"]
            return {
                "loss": total_loss,
                "flow_loss": loss_flow,
                "recon_loss": recon_dict["loss"],
                "l2_loss": recon_dict["l2_loss"],
                "l1_loss": recon_dict["l1_loss"],
                "lpips_loss": recon_dict["lpips_loss"],
                "gan_loss": recon_dict["gan_loss"],
            }
        
        loss_recon = self._compute_reconstruction_loss(
            pred_patch=pred_x1,
            target_patch=target_x1,
            pred_img=pred_img,
            target_img=x_clean,
            lpips_loss_fn=lpips_loss_fn,
            gan_loss=gan_loss,
        )
        return loss_flow + loss_recon

    @torch.no_grad()
    def generate(self, enc_cond: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """
        Generate image from encoded condition.
        
        enc_cond: [B, N, semantic_channels + hf_dim]
        return: [B, 3, H, W]
        """
        dec_cond = self.fused_proj(enc_cond)
        device = dec_cond.device
        dtype = dec_cond.dtype
        bsz, num_patches, _ = dec_cond.shape
        patch_size = self.decode_patch_size

        if self.decoder_type == "flow_matching":
            # Flow matching generation
            steps = self.flow_steps if steps is None else steps

            s_input = dec_cond.reshape(-1, self.hidden_size)
            patch_dim = patch_size ** 2 * self.out_channels
            x_t = torch.randn(bsz * num_patches, patch_dim, device=device, dtype=dtype)
            pos_emb = self._get_patch_coords(bsz, num_patches, device, dtype)

            dt = 1.0 / max(steps, 1)
            for i in range(steps):
                t_value = i / max(steps, 1)
                t_tensor = torch.full((bsz,), t_value, device=device, dtype=dtype)
                t_emb = self.t_embedder(t_tensor).to(dtype=dtype)
                t_input = t_emb.repeat_interleave(num_patches, dim=0)
                cond_input = torch.cat([s_input, t_input], dim=-1)
                v_pred = self.decoder(x_t, pos_emb, cond_input)
                x_t = x_t + v_pred * dt

            h = int(math.sqrt(num_patches)) * patch_size
            w = h
            x_out = x_t.reshape(bsz, num_patches, patch_dim).transpose(1, 2).contiguous()
            img = F.fold(
                x_out,
                output_size=(h, w),
                kernel_size=patch_size,
                stride=patch_size,
            )
            
        elif self.decoder_type == "vit_decoder":
            # ViT decoder generation (single forward pass)
            side = int(math.sqrt(num_patches))
            dec_cond_spatial = dec_cond.transpose(1, 2).contiguous().view(bsz, self.hidden_size, side, side)
            img = self.decoder(dec_cond_spatial)
            
        else:
            raise ValueError(f"Unknown decoder_type: {self.decoder_type}")
            
        return img

    def _denormalize_output(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.denormalize_decoder_output:
            return tensor
        device = tensor.device
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
        return tensor * imagenet_std + imagenet_mean

    def forward(self, x: torch.Tensor) -> CausalAutoencoderOutput:
        enc = self._infer_latent(x)
        enc_cond = self.encode(enc.latent, x_img=x, force_drop_hf=False)
        sample = self.generate(enc_cond)
        return CausalAutoencoderOutput(sample, enc.latent, enc.posterior)

    # ------------------------------------------------------------------
    # VF loss utilities (keep same interface)
    # ------------------------------------------------------------------

    # def get_vf_ref_param(self):
    #     backbone = self.encoder.get_backbone()
    #     for n, p in backbone.named_parameters():
    #         if ("lora_up" in n) or ("lora_down" in n):
    #             if p.requires_grad:
    #                 return p
    #     if self.encoder_type in ("dinov3", "dinov3_vitl"):
    #         return backbone.embeddings.patch_embeddings.weight
    #     if self.encoder_type == "siglip2":
    #         return backbone.embeddings.patch_embedding.weight
    #     if self.encoder_type == "dinov2":
    #         return backbone.embeddings.patch_embeddings.projection.weight
    #     return None

    # def compute_vf_loss(self, z_feat: torch.Tensor, f_feat: torch.Tensor) -> torch.Tensor:
    #     lmcos = vf_marginal_cos_loss(z_feat, f_feat, m1=self.vf_margin_cos)
    #     lmdms = vf_mdms_loss(z_feat, f_feat, m2=self.vf_margin_dms, max_tokens=self.vf_max_tokens)
    #     return lmcos + lmdms

    # def adaptive_weight(self, loss_rec, loss_vf, params, eps: float = 1e-6):
    #     n_rec = _grad_norm(loss_rec, params)
    #     n_vf = _grad_norm(loss_vf, params)
    #     if (n_rec == 0) or (n_vf == 0):
    #         return torch.tensor(1.0, device=loss_rec.device, dtype=loss_rec.dtype)
    #     return (n_rec / (n_vf + eps)).clamp(0.0, self.vf_weight_clamp).detach()
