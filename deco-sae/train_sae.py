import math
import os
import random
import sys
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config_utils import load_and_merge_config
from model import DecoSAE
from models.rae.utils.ddp_utils import cleanup_ddp, create_logger, requires_grad, setup_ddp
from models.rae.utils.image_utils import center_crop_arr
from models.rae.utils.metrics_utils import calculate_psnr
from torchvision.utils import make_grid

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not found. TensorBoard logging will be skipped.")

# Reuse GAN components from train_vae.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_VAE_DIR = os.path.join(ROOT_DIR, "train_vae")
if TRAIN_VAE_DIR not in sys.path:
    sys.path.insert(0, TRAIN_VAE_DIR)

from train_vae.dinodisc import DiffAug, DinoDisc, hinge_d_loss
from train_vae.gan_model import NLayerDiscriminator, d_hinge_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_grad_norm(model) -> float:
    """Compute the gradient norm of a model's parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class FlatImageDataset(Dataset):
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.paths = sorted(
            [
                os.path.join(root, f)
                for f in os.listdir(root)
                if os.path.splitext(f)[1].lower() in self.EXTS
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def build_datasets(image_size: int, train_path: str, val_path: str):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t * 2.0 - 1.0),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: center_crop_arr(img, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t * 2.0 - 1.0),
        ]
    )

    train_dataset = ImageFolder(train_path, transform=train_transform)
    try:
        val_dataset = ImageFolder(val_path, transform=val_transform)
    except Exception:
        val_dataset = FlatImageDataset(val_path, transform=val_transform)
    return train_dataset, val_dataset


def unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def reduce_mean_tensor(t: torch.Tensor) -> torch.Tensor:
    out = t.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out = out / dist.get_world_size()
    return out


@torch.no_grad()
def save_visualization(
    model: torch.nn.Module,
    vis_batch: torch.Tensor,
    step: int,
    output_dir: str,
    device: torch.device,
    nrow: int = 4,
):
    """Save visualization of ground truth and reconstructed images side by side."""
    model.eval()
    x = vis_batch.to(device, non_blocking=True)
    out = model(x)
    recon = out.sample.clamp(-1.0, 1.0)
    
    # Convert from [-1, 1] to [0, 1]
    x_vis = (x + 1.0) / 2.0
    recon_vis = (recon + 1.0) / 2.0
    
    # Interleave GT and recon for side-by-side comparison
    # [gt1, recon1, gt2, recon2, ...]
    batch_size = x.shape[0]
    interleaved = torch.stack([x_vis, recon_vis], dim=1).view(batch_size * 2, *x_vis.shape[1:])
    
    # Create grid: each row shows pairs of (GT, Recon)
    grid = make_grid(interleaved, nrow=nrow * 2, padding=2, normalize=False)
    
    # Convert to PIL and save
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    img = Image.fromarray(grid_np)
    
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    save_path = os.path.join(vis_dir, f"step_{step}.png")
    img.save(save_path)
    
    model.train()
    return save_path


def build_model(cfg, device: torch.device) -> DecoSAE:
    return DecoSAE(
        encoder_type=cfg.encoder.type,
        dinov3_model_dir=cfg.encoder.dinov3_model_dir,
        siglip2_model_name=cfg.encoder.siglip2_model_name,
        dinov2_model_name=cfg.encoder.dinov2_model_name,
        image_size=cfg.data.image_size,
        in_channels=3,
        out_channels=3,
        hidden_size=cfg.model.hidden_size,
        hidden_size_x=cfg.model.hidden_size_x,
        # Decoder type selection
        decoder_type=getattr(cfg.model, "decoder_type", "flow_matching"),
        # Flow matching decoder config
        num_decoder_blocks=cfg.model.num_decoder_blocks,
        nerf_max_freqs=cfg.model.nerf_max_freqs,
        flow_steps=cfg.model.flow_steps,
        time_dim=cfg.model.time_dim,
        # ViT decoder config
        vit_decoder_hidden_size=getattr(cfg.model, "vit_decoder_hidden_size", 1024),
        vit_decoder_num_layers=getattr(cfg.model, "vit_decoder_num_layers", 24),
        vit_decoder_num_heads=getattr(cfg.model, "vit_decoder_num_heads", 16),
        vit_decoder_intermediate_size=getattr(cfg.model, "vit_decoder_intermediate_size", 4096),
        vit_decoder_dropout=getattr(cfg.model, "vit_decoder_dropout", 0.0),
        gradient_checkpointing=getattr(cfg.model, "gradient_checkpointing", False),
        # HF branch
        enable_hf_branch=cfg.model.enable_hf_branch,
        hf_dim=cfg.model.hf_dim,
        hf_encoder_config_path=getattr(cfg.model, "hf_encoder_config_path", None),
        hf_dropout_prob=cfg.model.hf_dropout_prob,
        hf_noise_std=cfg.model.hf_noise_std,
        hf_loss_weight=cfg.model.hf_loss_weight,
        recon_l2_weight=cfg.loss.recon_l2_weight,
        recon_l1_weight=cfg.loss.recon_l1_weight,
        recon_lpips_weight=cfg.loss.recon_lpips_weight,
        recon_gan_weight=cfg.loss.recon_gan_weight,
        lora_rank=cfg.model.lora_rank,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        enable_lora=cfg.model.enable_lora,
        target_latent_channels=cfg.model.target_latent_channels,
        variational=cfg.model.variational,
        kl_weight=cfg.model.kl_weight,
        skip_to_moments=cfg.model.skip_to_moments,
        noise_tau=cfg.model.noise_tau,
        random_masking_channel_ratio=cfg.model.random_masking_channel_ratio,
        denormalize_decoder_output=cfg.model.denormalize_decoder_output,
    ).to(device)


def freeze_for_hf_and_pixel_decoder(model: DecoSAE):
    requires_grad(model, False)
    trainable_modules = [
        model.hf_encoder,
        model.fused_norm,
        model.fused_proj,
        model.decoder,
    ]
    # Flow matching specific modules (may be None for ViT decoder)
    if model.t_embedder is not None:
        trainable_modules.append(model.t_embedder)
    if model.coord_embedder is not None:
        trainable_modules.append(model.coord_embedder)
    
    for module in trainable_modules:
        requires_grad(module, True)


def build_discriminator(cfg, device: torch.device):
    if not cfg.discriminator.enabled:
        return None, None, None

    disc_type = cfg.discriminator.disc_type.lower()
    if disc_type == "dino":
        key_depths = tuple(int(x) for x in cfg.discriminator.key_depths.split(","))
        discriminator = DinoDisc(
            device=device,
            dino_ckpt_path=cfg.discriminator.dino_ckpt_path,
            ks=cfg.discriminator.ks,
            key_depths=key_depths,
            norm_type=cfg.discriminator.norm,
            using_spec_norm=True,
            norm_eps=1e-6,
            recipe=cfg.discriminator.recipe,
        ).to(device)
        disc_loss_fn = hinge_d_loss
    elif disc_type == "patchgan":
        discriminator = NLayerDiscriminator(
            in_channels=3,
            ndf=cfg.discriminator.ndf,
            n_layers=cfg.discriminator.n_layers,
            norm=cfg.discriminator.norm,
        ).to(device)
        disc_loss_fn = d_hinge_loss
    else:
        raise ValueError(f"Unsupported discriminator type: {cfg.discriminator.disc_type}")

    diffaug = DiffAug(
        prob=cfg.discriminator.diffaug_prob,
        cutout=cfg.discriminator.diffaug_cutout,
    )
    return discriminator, diffaug, disc_loss_fn


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_count = 0

    for idx, (x, _) in enumerate(val_loader):
        if idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        out = model(x)
        recon = out.sample
        loss = F.mse_loss(recon, x)
        psnr = calculate_psnr(recon, x)

        bsz = x.shape[0]
        total_loss += loss.item() * bsz
        total_psnr += psnr.item() * bsz
        total_count += bsz

    metrics = torch.tensor([total_loss, total_psnr, float(total_count)], device=device, dtype=torch.float64)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    denom = max(metrics[2].item(), 1.0)
    model.train()
    return metrics[0].item() / denom, metrics[1].item() / denom


def try_load_checkpoint(model: DecoSAE, ckpt_path: str, strict_load: bool, logger):
    if not ckpt_path:
        return 0, None
    if not os.path.isfile(ckpt_path):
        logger.warning(f"Checkpoint not found: {ckpt_path}")
        return 0, None

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in payload:
        state_dict = payload["model"]
    elif "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload

    missing, unexpected = model.load_state_dict(state_dict, strict=strict_load)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)
    step = payload.get("step", 0) if isinstance(payload, dict) else 0

    logger.info(f"Loaded checkpoint from {ckpt_path} (step={step})")
    if not strict_load:
        logger.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    return step, payload


def main():
    cfg = load_and_merge_config()

    rank, local_rank, _ = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    logger = create_logger(cfg.logging.output_dir, rank, name="train_sae")

    set_seed(cfg.training.seed + rank)
    torch.backends.cudnn.benchmark = True

    if cfg.training.precision == "bf16" and (not torch.cuda.is_bf16_supported()):
        raise ValueError("precision=bf16 but current GPU does not support bfloat16.")
    use_bf16 = cfg.training.precision == "bf16"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    train_dataset, val_dataset = build_datasets(
        image_size=cfg.data.image_size,
        train_path=cfg.data.train_path,
        val_path=cfg.data.val_path,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        sampler=val_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(cfg, device=device)
    resume_step, resume_payload = try_load_checkpoint(
        model, cfg.checkpoint.sae_ckpt, cfg.checkpoint.strict_load, logger
    )

    # Freeze semantic encoder; train only HF encoder and pixel decoder path.
    freeze_for_hf_and_pixel_decoder(model)
    model.train()

    if rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable / 1e6:.3f}M / {total / 1e6:.3f}M")

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # TensorBoard setup
    tb_writer = None
    if rank == 0 and HAS_TENSORBOARD:
        exp_name = os.path.basename(cfg.logging.output_dir.rstrip('/'))
        tb_log_dir = os.path.join("tensorboard_logs", exp_name)
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logging to: {tb_log_dir}")

    discriminator, diffaug, disc_loss_fn = build_discriminator(cfg, device=device)
    if discriminator is not None:
        discriminator = DDP(discriminator, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.optimizer.lr,
        betas=(cfg.optimizer.betas0, cfg.optimizer.betas1),
        weight_decay=cfg.optimizer.weight_decay,
    )

    scheduler = None
    if cfg.optimizer.scheduler == "cosine":
        min_lr_ratio = cfg.optimizer.min_lr / cfg.optimizer.lr if cfg.optimizer.lr > 0 else 0.1
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            warmup_steps=cfg.optimizer.warmup_steps,
            total_steps=cfg.training.max_steps,
            min_lr_ratio=min_lr_ratio,
        )

    optimizer_disc = None
    scheduler_disc = None
    if discriminator is not None:
        optimizer_disc = torch.optim.AdamW(
            [p for p in discriminator.parameters() if p.requires_grad],
            lr=cfg.discriminator.lr,
            betas=(cfg.discriminator.betas0, cfg.discriminator.betas1),
            weight_decay=cfg.discriminator.weight_decay,
        )
        if cfg.optimizer.scheduler == "cosine":
            min_lr_ratio_disc = (
                cfg.optimizer.min_lr / cfg.discriminator.lr if cfg.discriminator.lr > 0 else 0.1
            )
            scheduler_disc = get_cosine_schedule_with_warmup(
                optimizer=optimizer_disc,
                warmup_steps=cfg.optimizer.warmup_steps,
                total_steps=cfg.training.max_steps,
                min_lr_ratio=min_lr_ratio_disc,
            )

    if resume_payload is not None:
        if "optimizer" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer"])
        if scheduler is not None and ("scheduler" in resume_payload) and (resume_payload["scheduler"] is not None):
            scheduler.load_state_dict(resume_payload["scheduler"])
        if discriminator is not None and "discriminator" in resume_payload:
            unwrap_ddp(discriminator).load_state_dict(resume_payload["discriminator"], strict=False)
        if optimizer_disc is not None and "optimizer_disc" in resume_payload:
            optimizer_disc.load_state_dict(resume_payload["optimizer_disc"])
        if (
            scheduler_disc is not None
            and ("scheduler_disc" in resume_payload)
            and (resume_payload["scheduler_disc"] is not None)
        ):
            scheduler_disc.load_state_dict(resume_payload["scheduler_disc"])

    step = resume_step
    epoch = 0

    # Create a fixed visualization batch (same across all evals)
    vis_batch = None
    if rank == 0:
        # Use a fixed seed to get the same batch every time
        vis_generator = torch.Generator()
        vis_generator.manual_seed(42)
        vis_indices = torch.randperm(len(val_dataset), generator=vis_generator)[:min(8, len(val_dataset))]
        vis_images = [val_dataset[i][0] for i in vis_indices]
        vis_batch = torch.stack(vis_images, dim=0)
        logger.info(f"Created fixed visualization batch with {vis_batch.shape[0]} images")

    if rank == 0:
        logger.info("Start training DecoSAE with Flow Matching.")

    while step < cfg.training.max_steps:
        train_sampler.set_epoch(epoch)

        for x, _ in train_loader:
            if step >= cfg.training.max_steps:
                break
            x = x.to(device, non_blocking=True)

            gan_active = (
                discriminator is not None
                and cfg.loss.recon_gan_weight > 0
                and step >= cfg.discriminator.start_step
            )

            if gan_active:
                discriminator.eval()
                requires_grad(discriminator, False)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_bf16):
                loss_dict = unwrap_ddp(model).forward_loss(
                    x,
                    discriminator=unwrap_ddp(discriminator) if gan_active else None,
                    diffaug=diffaug if gan_active else None,
                    return_dict=True,
                )
                loss = loss_dict["loss"]
            loss.backward()
            
            # Compute gradient norm before optimizer step
            grad_norm = compute_grad_norm(model)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            disc_loss = torch.tensor(0.0, device=device)
            d_real = torch.tensor(0.0, device=device)
            d_fake = torch.tensor(0.0, device=device)
            disc_grad_norm = 0.0
            if gan_active:
                discriminator.train()
                requires_grad(discriminator, True)
                optimizer_disc.zero_grad(set_to_none=True)
                with torch.no_grad():
                    recon_detach = model(x).sample.detach().clamp(-1.0, 1.0)
                    # Keep discriminator input discretization same as VAE+GAN implementation.
                    recon_detach = torch.round((recon_detach + 1.0) * 127.5) / 127.5 - 1.0
                    x_aug = diffaug.aug(x) if diffaug is not None else x
                    recon_aug = diffaug.aug(recon_detach) if diffaug is not None else recon_detach

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_bf16):
                    logits_fake = discriminator(recon_aug)
                    logits_real = discriminator(x_aug)
                    disc_loss = disc_loss_fn(logits_real, logits_fake)
                    d_real = logits_real.mean().detach()
                    d_fake = logits_fake.mean().detach()
                disc_loss.backward()
                disc_grad_norm = compute_grad_norm(discriminator)
                optimizer_disc.step()
                if scheduler_disc is not None:
                    scheduler_disc.step()

            step += 1

            # Reduce losses across ranks
            reduced_loss = reduce_mean_tensor(loss)
            reduced_flow_loss = reduce_mean_tensor(loss_dict["flow_loss"])
            reduced_recon_loss = reduce_mean_tensor(loss_dict["recon_loss"])
            reduced_l2_loss = reduce_mean_tensor(loss_dict["l2_loss"])
            reduced_l1_loss = reduce_mean_tensor(loss_dict["l1_loss"])
            reduced_lpips_loss = reduce_mean_tensor(loss_dict["lpips_loss"])
            reduced_gan_loss = reduce_mean_tensor(loss_dict["gan_loss"])
            reduced_disc_loss = reduce_mean_tensor(disc_loss)
            reduced_d_real = reduce_mean_tensor(d_real)
            reduced_d_fake = reduce_mean_tensor(d_fake)

            if step % cfg.logging.log_every == 0 and rank == 0:
                lr = optimizer.param_groups[0]["lr"]
                log_msg = (
                    f"Step {step} lr={lr:.2e}: total={reduced_loss.item():.4f} | "
                    f"flow={reduced_flow_loss.item():.4f} recon={reduced_recon_loss.item():.4f} "
                    f"(L2={reduced_l2_loss.item():.4f}, L1={reduced_l1_loss.item():.4f}, "
                    f"LPIPS={reduced_lpips_loss.item():.4f}) | grad_norm={grad_norm:.4f}"
                )
                if gan_active:
                    disc_lr = optimizer_disc.param_groups[0]["lr"]
                    log_msg += (
                        f" | GAN_g={reduced_gan_loss.item():.4f} D_loss={reduced_disc_loss.item():.4f} "
                        f"D_real={reduced_d_real.item():.3f} D_fake={reduced_d_fake.item():.3f} "
                        f"disc_grad_norm={disc_grad_norm:.4f} disc_lr={disc_lr:.2e}"
                    )
                logger.info(log_msg)
                
                # TensorBoard logging
                if tb_writer is not None:
                    # Total loss
                    tb_writer.add_scalar("Loss/total", reduced_loss.item(), step)
                    tb_writer.add_scalar("Loss/flow", reduced_flow_loss.item(), step)
                    tb_writer.add_scalar("Loss/recon", reduced_recon_loss.item(), step)
                    
                    # Reconstruction loss components
                    tb_writer.add_scalar("Loss/L2", reduced_l2_loss.item(), step)
                    tb_writer.add_scalar("Loss/L1", reduced_l1_loss.item(), step)
                    tb_writer.add_scalar("Loss/LPIPS", reduced_lpips_loss.item(), step)
                    
                    # Gradient norms
                    tb_writer.add_scalar("GradNorm/generator", grad_norm, step)
                    
                    # Learning rate
                    tb_writer.add_scalar("LR/generator", lr, step)
                    
                    # GAN losses (when active)
                    if gan_active:
                        tb_writer.add_scalar("Loss/GAN_gen", reduced_gan_loss.item(), step)
                        tb_writer.add_scalar("Loss/GAN_disc", reduced_disc_loss.item(), step)
                        tb_writer.add_scalar("Discriminator/logits_real", reduced_d_real.item(), step)
                        tb_writer.add_scalar("Discriminator/logits_fake", reduced_d_fake.item(), step)
                        tb_writer.add_scalar("GradNorm/discriminator", disc_grad_norm, step)
                        tb_writer.add_scalar("LR/discriminator", disc_lr, step)

            if step % cfg.logging.eval_every == 0:
                val_loss, val_psnr = run_validation(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    max_batches=cfg.logging.val_max_batches,
                )
                if rank == 0:
                    logger.info(f"[Eval {step}] val_loss={val_loss:.6f}, val_psnr={val_psnr:.4f}")
                    if tb_writer is not None:
                        tb_writer.add_scalar("Validation/loss", val_loss, step)
                        tb_writer.add_scalar("Validation/PSNR", val_psnr, step)
                    
                    # Save visualization of fixed batch
                    if vis_batch is not None:
                        vis_path = save_visualization(
                            model=model,
                            vis_batch=vis_batch,
                            step=step,
                            output_dir=cfg.logging.output_dir,
                            device=device,
                            nrow=4,
                        )
                        logger.info(f"Saved visualization to {vis_path}")

            if step % cfg.logging.save_every == 0 and rank == 0:
                os.makedirs(cfg.logging.output_dir, exist_ok=True)
                ckpt_path = os.path.join(cfg.logging.output_dir, f"step_{step}.pth")
                torch.save(
                    {
                        "model": unwrap_ddp(model).state_dict(),
                        "step": step,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "discriminator": unwrap_ddp(discriminator).state_dict()
                        if discriminator is not None
                        else None,
                        "optimizer_disc": optimizer_disc.state_dict() if optimizer_disc is not None else None,
                        "scheduler_disc": scheduler_disc.state_dict() if scheduler_disc is not None else None,
                        "config": cfg,
                    },
                    ckpt_path,
                )
                logger.info(f"Saved checkpoint to {ckpt_path}")

        epoch += 1

    # Close TensorBoard writer
    if tb_writer is not None:
        tb_writer.close()

    cleanup_ddp()


if __name__ == "__main__":
    main()
