"""
VAE utility functions for loading and using DINOv3 VAE models.
Supports both CNN decoder and diffusion decoder.
"""

import torch
import torch.nn as nn

# ==========================================
#           SAE Normalization
# ==========================================

EMA_SHIFT_FACTOR = 0.0019670347683131695
EMA_SCALE_FACTOR = 0.247765451669693


def normalize_sae(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor using SAE statistics."""
    return (tensor - EMA_SHIFT_FACTOR) / EMA_SCALE_FACTOR


def denormalize_sae(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize tensor using SAE statistics."""
    return tensor * EMA_SCALE_FACTOR + EMA_SHIFT_FACTOR


# ==========================================
#           VAE Loading
# ==========================================

def requires_grad(model: nn.Module, flag: bool = True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def load_dinov3_vae(
    vae_checkpoint_path: str,
    device: torch.device,
    decoder_type: str = "diffusion_decoder",
    model_params: dict = None,
    verbose: bool = True,
):
    """
    Build and load DINOv3 AutoencoderKL.

    Args:
        vae_checkpoint_path: Path to VAE checkpoint file.
        device: Device to load the model on.
        decoder_type: "diffusion_decoder" or "cnn_decoder".
        model_params: Optional custom model parameters. If None, uses default.
        verbose: Whether to print loading information.

    Returns:
        Loaded VAE model in eval mode with frozen parameters.
    """
    # Dynamic import based on decoder type
    if decoder_type == "cnn_decoder":
        try:
            from cnn_decoder import AutoencoderKL
        except ImportError:
            raise ImportError("cnn_decoder module not found, cannot use decoder_type='cnn_decoder'.")
        if verbose:
            print("[load_dinov3_vae] Using CNN decoder (cnn_decoder.AutoencoderKL).")
    elif decoder_type == "diffusion_decoder":
        try:
            from sae_model import AutoencoderKL
        except ImportError:
            raise ImportError("sae_model module not found, cannot use decoder_type='diffusion_decoder'.")
        if verbose:
            print("[load_dinov3_vae] Using diffusion decoder (sae_model.AutoencoderKL).")
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}")

    # Default model parameters
    if model_params is None:
        model_params = {
            "dinov3_model_dir": "/share/project/huangxu/models/dinov3",
            "image_size": 256,
            "patch_size": 16,
            "out_channels": 3,
            "latent_channels": 1280,
            "target_latent_channels": None,
            "spatial_downsample_factor": 16,
            "lora_rank": 256,
            "lora_alpha": 256,
            "dec_block_out_channels": (1280, 1024, 512, 256, 128),
            "dec_layers_per_block": 3,
            "decoder_dropout": 0.0,
            "gradient_checkpointing": False,
            "denormalize_decoder_output": False,
        }

    vae = AutoencoderKL(**model_params).to(device)

    if verbose:
        print(f"[load_dinov3_vae] Loading VAE checkpoint from: {vae_checkpoint_path}")

    checkpoint = torch.load(vae_checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    if verbose:
        if missing:
            print(f"[load_dinov3_vae] Missing keys: {missing}")
        if unexpected:
            print(f"[load_dinov3_vae] Unexpected keys: {unexpected}")
        print(f"[load_dinov3_vae] Loaded VAE. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    vae.eval()
    requires_grad(vae, False)
    return vae


# ==========================================
#           Reconstruction
# ==========================================

@torch.no_grad()
def reconstruct_from_latent_with_diffusion(
    vae,
    latent_z: torch.Tensor,
    image_shape: torch.Size,
    diffusion_steps: int = 25,
    decoder_type: str = "diffusion_decoder",
) -> torch.Tensor:
    """
    Given latent_z from DINO encoder (or model prediction in the same latent space),
    run the full decoder reconstruction.

    Args:
        vae: The VAE model.
        latent_z: Latent tensor from encoder.
        image_shape: Output image shape (B, C, H, W).
        diffusion_steps: Number of diffusion steps (only for diffusion_decoder).
        decoder_type: "diffusion_decoder" or "cnn_decoder".

    Returns:
        Reconstructed image tensor in [-1, 1] range.
    """
    device = latent_z.device
    # VAE is float32, ensure input is also float32
    z = denormalize_sae(latent_z).float()

    if decoder_type == "diffusion_decoder":
        if getattr(vae, "post_quant_conv", None) is not None and vae.post_quant_conv is not None:
            z = vae.post_quant_conv(z)

        context = vae.diffusion_decoder.get_context(z)
        corrected_context = list(reversed(context[:]))

        diffusion = vae.diffusion_decoder.diffusion
        diffusion.set_sample_schedule(diffusion_steps, device)

        init_noise = torch.randn(
            image_shape,
            device=device,
            dtype=torch.float32,
        )

        recon = diffusion.p_sample_loop(
            vae=vae,
            shape=image_shape,
            context=corrected_context,
            clip_denoised=True,
            init_noise=init_noise,
            eta=0.0,
        )
        return recon

    elif decoder_type == "cnn_decoder":
        if not hasattr(vae, "decode"):
            raise AttributeError("VAE does not have decode(); cnn_decoder requires vae.decode(z).")
        out = vae.decode(z)
        recon = out.sample if hasattr(out, "sample") else out
        return recon

    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}")
