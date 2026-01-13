import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch import nn
from typing import Literal, NamedTuple, Optional, Tuple
import math
import sys
sys.path.append('/share/project/huangxu/sae/SAE')
from models.dino_v3.modeling_dino_v3 import DINOv3ViTModel

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


class ResnetBlock2D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer.
            If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
    """

    def __init__(
        self, *, in_channels: int, out_channels: Optional[int] = None, dropout: float = 0.0
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.nonlinearity = nn.SiLU()

        self.norm1 = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = torch.nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden = input_tensor

        hidden = self.norm1(hidden)
        hidden = self.nonlinearity(hidden)
        hidden = self.conv1(hidden)

        hidden = self.norm2(hidden)
        hidden = self.nonlinearity(hidden)
        hidden = self.dropout(hidden)
        hidden = self.conv2(hidden)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden

        return output_tensor


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
        random_masking_channel_ratio: float = 0.1,
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

        # pass init params to Decoder
        self.decoder = Decoder2D(
            in_channels=decoder_in_channels,
            out_channels=out_channels,
            block_out_channels=dec_block_out_channels,
            layers_per_block=dec_layers_per_block,
            gradient_checkpointing=gradient_checkpointing,
        )

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
        x = self.decode(z).sample
        return CausalAutoencoderOutput(x, z, p)

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

    def decode(self, z: torch.FloatTensor) -> CausalDecoderOutput:
        z = self.post_quant_conv(z) if self.post_quant_conv is not None else z
        x = self.decoder(z)
        x = self._denormalize_output(x)
        return CausalDecoderOutput(x)