"""
DecoSAE with pretrained vision encoder + Flow Matching patch decoder.

- Keep original encoder loading path (DINOv3 / DINOv2 / SigLIP2 + optional LoRA).
- Replace old pixel decoder with Flow Matching decoder.
- Add an optional HF (high-frequency) residual branch.
"""

import sys
import os
import math
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cnn_decoder import (
    Encoder2D,
    SigLIP2Encoder2D,
    DINOv2Encoder2D,
    CausalAutoencoderOutput,
    CausalEncoderOutput,
    CausalDecoderOutput,
    vf_marginal_cos_loss,
    vf_mdms_loss,
    _grad_norm,
    mask_channels,
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


class NerfEmbedder(nn.Module):
    """Coordinate encoder for patch pixels (no pixel-value input)."""

    def __init__(self, hidden_size_output: int, max_freqs: int = 8):
        super().__init__()
        self.max_freqs = max_freqs
        self.embedder = nn.Sequential(
            nn.Linear(max_freqs ** 2, hidden_size_output, bias=True),
        )

    @lru_cache(maxsize=8)
    def fetch_pos(self, patch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos_x = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)

        freqs = torch.linspace(0, self.max_freqs, self.max_freqs, dtype=dtype, device=device)
        freqs_x = freqs[None, :, None]
        freqs_y = freqs[None, None, :]
        coeffs = (1 + freqs_x * freqs_y) ** -1
        dct_x = torch.cos(pos_x * freqs_x * torch.pi)
        dct_y = torch.cos(pos_y * freqs_y * torch.pi)
        dct = (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs ** 2)
        return dct

    def forward(
        self,
        batch_size: int,
        num_patches: int,
        patch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        dct = self.fetch_pos(patch_size, device, dtype)
        dct = dct.expand(batch_size * num_patches, -1, -1)
        return self.embedder(dct)


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
    Flow-matching decoder, predicts patch-wise velocity field.

    x_t:      [B*N, P^2, 3]
    pos_emb:  [B*N, P^2, C]
    c_global: [B*N, C_cond]
    """

    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 256,
        out_channels: int = 3,
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
        c = self.cond_proj(c_global).unsqueeze(1)
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
    """

    def __init__(self, in_channels: int, out_channels: int, patch_size: int):
        super().__init__()
        hidden = max(out_channels // 2, 32)
        self.patch_size = patch_size
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SiLU(),
        )
        self.proj = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = F.avg_pool2d(h, kernel_size=self.patch_size, stride=self.patch_size)
        return self.proj(h)


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
        # ---- Image config ----
        image_size: int = 256,
        in_channels: int = 3,
        out_channels: int = 3,
        # ---- Flow decoder config ----
        hidden_size: int = 1152,
        num_decoder_blocks: int = 12,
        nerf_max_freqs: int = 8,
        flow_steps: int = 25,
        time_dim: int = 256,
        # ---- HF branch ----
        enable_hf_branch: bool = True,
        hf_dim: int = 256,
        hf_dropout_prob: float = 0.4,
        hf_noise_std: float = 0.1,
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
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.variational = variational
        self.kl_weight = kl_weight
        self.noise_tau = noise_tau
        self.random_masking_channel_ratio = random_masking_channel_ratio
        self.denormalize_decoder_output = denormalize_decoder_output
        self.skip_to_moments = skip_to_moments
        self.flow_steps = flow_steps
        self.time_dim = time_dim
        self.enable_hf_branch = enable_hf_branch
        self.hf_dropout_prob = hf_dropout_prob
        self.hf_noise_std = hf_noise_std
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
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}. "
                f"Supported: 'dinov3', 'dinov3_vitl', 'siglip2', 'dinov2'"
            )

        encoder_hidden_size = self.encoder.hidden_size
        encoder_patch_size = self.encoder.patch_size
        self.decode_patch_size = 16 if encoder_type == "dinov2" else encoder_patch_size
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
        self.hf_encoder = HFEncoder(
            in_channels=in_channels,
            out_channels=self.hf_dim,
            patch_size=self.decode_patch_size,
        )
        self.fused_norm = nn.LayerNorm(self.semantic_channels + self.hf_dim, elementwise_affine=False, eps=1e-6)
        self.fused_proj = nn.Linear(self.semantic_channels + self.hf_dim, hidden_size, bias=True)

        # 4) Flow matching modules.
        self.t_embedder = TimestepEmbedder(time_dim)
        self.coord_embedder = NerfEmbedder(hidden_size, max_freqs=nerf_max_freqs)
        self.decoder = FlowPatchDecoder(
            in_channels=out_channels,
            model_channels=hidden_size,
            out_channels=out_channels,
            cond_channels=hidden_size + time_dim,
            num_res_blocks=num_decoder_blocks,
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
        return self.coord_embedder(batch_size, num_patches, self.decode_patch_size, device, dtype)

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
                )
            s_hf = s_hf * keep_mask
            if self.hf_noise_std > 0:
                s_hf = s_hf + torch.randn_like(s_hf) * self.hf_noise_std
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
        x_out = x_patch.reshape(batch_size, num_patches, self.out_channels * patch_size ** 2)
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
    ) -> torch.Tensor:
        loss = self.recon_l2_weight * F.mse_loss(pred_patch, target_patch)
        if self.recon_l1_weight > 0:
            loss = loss + self.recon_l1_weight * F.l1_loss(pred_patch, target_patch)
        if self.recon_lpips_weight > 0 and lpips_loss_fn is not None:
            loss = loss + self.recon_lpips_weight * lpips_loss_fn(target_img, pred_img)
        if self.recon_gan_weight > 0 and gan_loss is not None:
            loss = loss + self.recon_gan_weight * gan_loss
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
    ) -> torch.Tensor:
        """
        Flow matching training loss.

        x_clean: [B, 3, H, W]
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

        x_data_flat = F.unfold(x_clean, kernel_size=patch_size, stride=patch_size)
        x_data_flat = x_data_flat.transpose(1, 2)
        x_data_flat = x_data_flat.reshape(bsz, num_patches, self.out_channels, patch_size ** 2).permute(0, 1, 3, 2)

        t = torch.rand(bsz, device=device, dtype=dtype)
        x_noise_flat = torch.randn_like(x_data_flat)
        t_expand = t.view(bsz, 1, 1, 1)
        x_t = t_expand * x_data_flat + (1 - t_expand) * x_noise_flat
        target_v = x_data_flat - x_noise_flat

        x_t_input = x_t.reshape(-1, patch_size ** 2, self.out_channels)
        s_input = dec_cond.reshape(-1, self.hidden_size)
        t_emb = self.t_embedder(t).to(dtype=dtype)
        t_input = t_emb.repeat_interleave(num_patches, dim=0)
        cond_input = torch.cat([s_input, t_input], dim=-1)
        pos_emb = self._get_patch_coords(bsz, num_patches, device, dtype)

        pred_v = self.decoder(x_t_input, pos_emb, cond_input)
        target_v_flat = target_v.reshape(-1, patch_size ** 2, self.out_channels)
        loss_flow = F.mse_loss(pred_v, target_v_flat)

        # x_1 = x_t + (1 - t) * v
        t_bn = t.repeat_interleave(num_patches).view(-1, 1, 1)
        pred_x1 = x_t_input + (1 - t_bn) * pred_v
        target_x1 = x_data_flat.reshape(-1, patch_size ** 2, self.out_channels)

        pred_img = self._patches_to_image(pred_x1, bsz, num_patches, patch_size)
        if self.recon_lpips_weight > 0 and lpips_loss_fn is None:
            lpips_loss_fn = self._get_lpips_loss(device=x_clean.device)
        if gan_loss is None:
            gan_loss = self._compute_gan_generator_loss(
                pred_img=pred_img,
                discriminator=discriminator,
                diffaug=diffaug,
            )
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
        enc_cond: [B, N, semantic_channels + hf_dim]
        return: [B, 3, H, W]
        """
        dec_cond = self.fused_proj(enc_cond)
        device = dec_cond.device
        dtype = dec_cond.dtype
        bsz, num_patches, _ = dec_cond.shape
        patch_size = self.decode_patch_size
        steps = self.flow_steps if steps is None else steps

        s_input = dec_cond.reshape(-1, self.hidden_size)
        x_t = torch.randn(bsz * num_patches, patch_size ** 2, self.out_channels, device=device, dtype=dtype)
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
        x_out = x_t.reshape(bsz, num_patches, self.out_channels * patch_size ** 2).transpose(1, 2).contiguous()
        img = F.fold(
            x_out,
            output_size=(h, w),
            kernel_size=patch_size,
            stride=patch_size,
        )
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
