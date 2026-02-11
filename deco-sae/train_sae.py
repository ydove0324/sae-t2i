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

# Reuse GAN components from train_vae.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_VAE_DIR = os.path.join(ROOT_DIR, "train_vae")
if TRAIN_VAE_DIR not in sys.path:
    sys.path.insert(0, TRAIN_VAE_DIR)

from dinodisc import DiffAug, DinoDisc, hinge_d_loss
from gan_model import NLayerDiscriminator, d_hinge_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        num_decoder_blocks=cfg.model.num_decoder_blocks,
        nerf_max_freqs=cfg.model.nerf_max_freqs,
        flow_steps=cfg.model.flow_steps,
        time_dim=cfg.model.time_dim,
        enable_hf_branch=cfg.model.enable_hf_branch,
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
        model.t_embedder,
        model.coord_embedder,
        model.decoder,
    ]
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

    payload = torch.load(ckpt_path, map_location="cpu")
    if "model" in payload:
        state_dict = payload["model"]
    elif "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload

    missing, unexpected = model.load_state_dict(state_dict, strict=strict_load)
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
                loss = model.module.forward_loss(
                    x,
                    discriminator=discriminator if gan_active else None,
                    diffaug=diffaug if gan_active else None,
                )
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            disc_loss = torch.tensor(0.0, device=device)
            d_real = torch.tensor(0.0, device=device)
            d_fake = torch.tensor(0.0, device=device)
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
                optimizer_disc.step()
                if scheduler_disc is not None:
                    scheduler_disc.step()

            step += 1

            reduced_loss = reduce_mean_tensor(loss)
            reduced_disc_loss = reduce_mean_tensor(disc_loss)
            reduced_d_real = reduce_mean_tensor(d_real)
            reduced_d_fake = reduce_mean_tensor(d_fake)

            if step % cfg.logging.log_every == 0 and rank == 0:
                lr = optimizer.param_groups[0]["lr"]
                if gan_active:
                    disc_lr = optimizer_disc.param_groups[0]["lr"]
                    logger.info(
                        "Step %d: g_loss=%.6f d_loss=%.6f d_real=%.4f d_fake=%.4f lr=%.3e disc_lr=%.3e"
                        % (
                            step,
                            reduced_loss.item(),
                            reduced_disc_loss.item(),
                            reduced_d_real.item(),
                            reduced_d_fake.item(),
                            lr,
                            disc_lr,
                        )
                    )
                else:
                    logger.info(f"Step {step}: loss={reduced_loss.item():.6f}, lr={lr:.3e}")

            if step % cfg.logging.eval_every == 0:
                val_loss, val_psnr = run_validation(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    max_batches=cfg.logging.val_max_batches,
                )
                if rank == 0:
                    logger.info(f"[Eval {step}] val_loss={val_loss:.6f}, val_psnr={val_psnr:.4f}")

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

    cleanup_ddp()


if __name__ == "__main__":
    main()
