# training_vae.py
import os
import sys
sys.path.append(".")
import argparse
import shutil
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

# local import
from cnn_decoder import AutoencoderKL

# PatchGAN from local gan_model.py
from gan_model import NLayerDiscriminator, d_hinge_loss, g_hinge_loss

# === deps ===
try:
    import lpips
except ImportError:
    print("Please install lpips: pip install lpips")
    sys.exit(1)

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not found. TensorBoard logging will be skipped.")

try:
    from pytorch_fid import fid_score
    HAS_FID = True
except ImportError:
    HAS_FID = False
    print("Warning: pytorch-fid not found. rFID calculation will be skipped.")


# ==========================
# DDP / FSDP helpers
# ==========================

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(0)
        return 0, 0, 1

def cleanup_ddp():
    dist.destroy_process_group()

def create_logger(save_dir, rank):
    logger = logging.getLogger(f"rank{rank}")
    logger.setLevel(logging.INFO)
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='a')
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    else:
        logger.addHandler(logging.NullHandler())
    return logger

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def unwrap_model(model):
    """获取 DDP 或 FSDP 包装模型的内部原始模型。"""
    if isinstance(model, FSDP):
        return model.module
    elif isinstance(model, DDP):
        return model.module
    elif hasattr(model, "module"):
        return model.module
    return model


def get_model_state_dict(model):
    """
    获取模型状态字典，兼容 DDP 和 FSDP。
    对于 FSDP，需要使用 full_state_dict 来收集所有分片。
    """
    if isinstance(model, FSDP):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            return model.state_dict()
    elif isinstance(model, DDP):
        return model.module.state_dict()
    elif hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    """Calculate adaptive weight for GAN loss based on gradient magnitudes."""
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()


# ==========================
# Data helpers
# ==========================

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


# ==========================
# Losses & metrics
# ==========================

class LPIPSLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # LPIPS expects input in [-1, 1]
        self.lpips = lpips.LPIPS(net='vgg',lpips=False,use_dropout=False).to(device)
        self.lpips.eval()
        requires_grad(self.lpips, False)

    def forward(self, x, rec):
        # x, rec: [-1, 1]
        return self.lpips(x, rec).mean()

def calculate_psnr(img1, img2):
    # img1,img2: [-1,1]
    img1 = torch.clamp((img1 + 1.0) / 2.0, 0, 1)
    img2 = torch.clamp((img2 + 1.0) / 2.0, 0, 1)
    mse = torch.mean((img1 - img2) ** 2)
    if mse <= 0:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# ==========================
# Validation
# ==========================

@torch.no_grad()
def run_validation(model, val_loader, device, step, args, logger, use_fsdp=False, tb_writer=None):
    model.eval()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    gt_dir = os.path.join(args.output_dir, f"temp_eval_gt_{step}")
    recon_dir = os.path.join(args.output_dir, f"temp_eval_recon_{step}")

    if rank == 0:
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)
        logger.info(f"Running validation at step {step}...")

    dist.barrier()

    total_psnr = 0.0
    count = 0
    max_batches = args.val_max_batches

    for i, (x, _) in enumerate(val_loader):
        if i >= max_batches:
            break
        x = x.to(device)

        out = model(x)
        recon = out.sample  # [-1,1]
        batch_psnr = calculate_psnr(x, recon)
        total_psnr += batch_psnr.item()
        count += 1

        img_gt = torch.clamp((x + 1.0) / 2.0, 0, 1)
        img_recon = torch.clamp((recon + 1.0) / 2.0, 0, 1)

        for idx in range(x.size(0)):
            fname = f"r{rank}_b{i}_{idx}.png"
            save_image(img_gt[idx], os.path.join(gt_dir, fname))
            save_image(img_recon[idx], os.path.join(recon_dir, fname))

    avg_psnr = torch.tensor(total_psnr / max(count, 1), device=device)
    dist.all_reduce(avg_psnr, op=dist.ReduceOp.SUM)
    avg_psnr = avg_psnr.item() / world_size

    dist.barrier()

    if rank == 0:
        rfid = -1.0
        if HAS_FID:
            try:
                rfid = fid_score.calculate_fid_given_paths(
                    paths=[gt_dir, recon_dir],
                    batch_size=50,
                    device=device,
                    dims=2048,
                    num_workers=4
                )
            except Exception as e:
                logger.error(f"FID Calculation Error: {e}")

        logger.info(f"[Validation Step {step}] PSNR: {avg_psnr:.4f} | rFID: {rfid:.4f}")
        with open(os.path.join(args.output_dir, "metrics.txt"), "a") as f:
            f.write(f"Step: {step}, PSNR: {avg_psnr:.4f}, rFID: {rfid:.4f}\n")

        # TensorBoard logging for validation metrics
        if tb_writer is not None:
            tb_writer.add_scalar("Validation/PSNR", avg_psnr, step)
            if rfid >= 0:  # Only log if FID was successfully calculated
                tb_writer.add_scalar("Validation/rFID", rfid, step)

        if not args.debug:
            shutil.rmtree(gt_dir, ignore_errors=True)
            shutil.rmtree(recon_dir, ignore_errors=True)

    dist.barrier()
    model.train()


# ==========================
# Main
# ==========================

def load_config_file(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()

    # 配置文件支持
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--val-path", type=str, default="/share/project/datasets/ImageNet/val")
    parser.add_argument("--output-dir", type=str, default="results/dinov3_vae_lora_vf")
    parser.add_argument("--vae-ckpt", type=str, default="", help="optional init checkpoint")
    
    # Encoder 配置
    parser.add_argument("--encoder-type", type=str, default="dinov3", choices=["dinov3", "siglip2"],
                        help="Encoder type: 'dinov3' or 'siglip2'")
    parser.add_argument("--dinov3-dir", type=str, default="/cpfs01/huangxu/models/dinov3")
    parser.add_argument("--siglip2-model-name", type=str, default="google/siglip2-base-patch16-256",
                        help="SigLIP2 model name from HuggingFace")
    
    # Decoder 配置
    parser.add_argument("--latent-channels", type=int, default=None, 
                        help="Latent channels (auto-detected from encoder if not set)")
    parser.add_argument("--dec-block-out-channels", type=str, default="1280,1024,512,256,128",
                        help="Decoder block output channels, comma-separated")

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=100000)

    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--lpips-weight", type=float, default=0.5)

    parser.add_argument("--kl-weight", type=float, default=1e-6)

    # VF loss
    parser.add_argument("--vf-weight", type=float, default=0.1)
    parser.add_argument("--vf-m1", type=float, default=0.1)
    parser.add_argument("--vf-m2", type=float, default=0.1)
    parser.add_argument("--vf-max-tokens", type=int, default=32)
    parser.add_argument("--no-vf", action="store_true", help="Disable VF loss")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA")

    # model regularization
    parser.add_argument("--noise-tau", type=float, default=0.0)
    parser.add_argument("--mask-channels", type=float, default=0.0)

    # GAN - PatchGAN
    parser.add_argument("--gan-weight", type=float, default=0.0, help="GAN loss weight (0 to disable)")
    parser.add_argument("--gan-start-step", type=int, default=0, help="Step to start GAN training")
    parser.add_argument("--disc-lr", type=float, default=1e-4, help="Discriminator learning rate")
    parser.add_argument("--max-d-weight", type=float, default=1e4, help="Max adaptive weight for GAN loss")
    # PatchGAN discriminator config
    parser.add_argument("--disc-ndf", type=int, default=64, help="PatchGAN base channels")
    parser.add_argument("--disc-n-layers", type=int, default=4, help="PatchGAN number of layers")
    parser.add_argument("--disc-norm", type=str, default="gn", choices=["in", "bn", "gn"], help="PatchGAN normalization")
    parser.add_argument("--no-gan", action="store_true", help="Disable GAN loss")

    # Stage training mode for GAN
    parser.add_argument("--stage-disc-steps", type=int, default=0,
                        help="Number of steps to train discriminator only (stage_disc mode). "
                             "If 0, train jointly from the start (stage_joint mode).")

    # Stage 1 training: train to_moments + decoder only (no LoRA, no VF loss)
    # This is a separate training stage concept, not related to GAN
    parser.add_argument("--stage1-steps", type=int, default=0,
                        help="Stage 1 steps: train to_moments + decoder only (freeze encoder/LoRA, no VF loss). "
                             "After stage1_steps, transition to Stage 2: enable LoRA + VF loss. "
                             "If 0, skip Stage 1 and train everything from start.")

    # FSDP
    parser.add_argument("--fsdp-size", type=int, default=1,
                        help="FSDP sharding size. 1=disabled (use DDP), >1=enable FSDP to shard optimizer states.")

    # Precision
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="bf16",
                        help="Compute precision for training.")

    # freq
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--val-max-batches", type=int, default=200)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    
    # 如果指定了配置文件，加载并覆盖默认值
    if args.config:
        config = load_config_file(args.config)
        # 从配置文件更新 args
        if 'data' in config:
            if args.train_path is None and 'train_path' in config['data']:
                args.train_path = config['data']['train_path']
            if 'val_path' in config['data']:
                args.val_path = config['data']['val_path']
            if 'image_size' in config['data']:
                args.image_size = config['data']['image_size']
            if 'batch_size' in config['data']:
                args.batch_size = config['data']['batch_size']
        
        if 'encoder' in config:
            enc_cfg = config['encoder']
            if 'type' in enc_cfg:
                args.encoder_type = enc_cfg['type']
            if 'dinov3' in enc_cfg and 'model_dir' in enc_cfg['dinov3']:
                args.dinov3_dir = enc_cfg['dinov3']['model_dir']
            if 'siglip2' in enc_cfg and 'model_name' in enc_cfg['siglip2']:
                args.siglip2_model_name = enc_cfg['siglip2']['model_name']
        
        if 'decoder' in config:
            dec_cfg = config['decoder']
            if 'latent_channels' in dec_cfg:
                args.latent_channels = dec_cfg['latent_channels']
            if 'block_out_channels' in dec_cfg:
                args.dec_block_out_channels = ','.join(map(str, dec_cfg['block_out_channels']))
        
        if 'lora' in config:
            lora_cfg = config['lora']
            if 'rank' in lora_cfg:
                args.lora_rank = lora_cfg['rank']
            if 'alpha' in lora_cfg:
                args.lora_alpha = lora_cfg['alpha']
            if 'dropout' in lora_cfg:
                args.lora_dropout = lora_cfg['dropout']
            if 'enabled' in lora_cfg and not lora_cfg['enabled']:
                args.no_lora = True
        
        if 'loss' in config:
            loss_cfg = config['loss']
            if 'l1_weight' in loss_cfg:
                args.l1_weight = loss_cfg['l1_weight']
            if 'lpips_weight' in loss_cfg:
                args.lpips_weight = loss_cfg['lpips_weight']
            if 'kl_weight' in loss_cfg:
                args.kl_weight = loss_cfg['kl_weight']
            
            if 'vf' in loss_cfg:
                vf_cfg = loss_cfg['vf']
                if 'weight' in vf_cfg:
                    args.vf_weight = vf_cfg['weight']
                if 'margin_cos' in vf_cfg:
                    args.vf_m1 = vf_cfg['margin_cos']
                if 'margin_dms' in vf_cfg:
                    args.vf_m2 = vf_cfg['margin_dms']
                if 'max_tokens' in vf_cfg:
                    args.vf_max_tokens = vf_cfg['max_tokens']
                if 'enabled' in vf_cfg and not vf_cfg['enabled']:
                    args.no_vf = True
            
            if 'gan' in loss_cfg:
                gan_cfg = loss_cfg['gan']
                if 'weight' in gan_cfg:
                    args.gan_weight = gan_cfg['weight']
                if 'start_step' in gan_cfg:
                    args.gan_start_step = gan_cfg['start_step']
                if 'max_d_weight' in gan_cfg:
                    args.max_d_weight = gan_cfg['max_d_weight']
                if 'stage_disc_steps' in gan_cfg:
                    args.stage_disc_steps = gan_cfg['stage_disc_steps']
                if 'enabled' in gan_cfg and not gan_cfg['enabled']:
                    args.no_gan = True
        
        if 'discriminator' in config:
            disc_cfg = config['discriminator']
            if 'ndf' in disc_cfg:
                args.disc_ndf = disc_cfg['ndf']
            if 'n_layers' in disc_cfg:
                args.disc_n_layers = disc_cfg['n_layers']
            if 'norm' in disc_cfg:
                args.disc_norm = disc_cfg['norm']
            if 'lr' in disc_cfg:
                args.disc_lr = disc_cfg['lr']
        
        if 'optimizer' in config:
            opt_cfg = config['optimizer']
            if 'lr' in opt_cfg:
                args.lr = opt_cfg['lr']
        
        if 'training' in config:
            train_cfg = config['training']
            if 'max_steps' in train_cfg:
                args.max_steps = train_cfg['max_steps']
            if 'precision' in train_cfg:
                args.precision = train_cfg['precision']
            if 'fsdp_size' in train_cfg:
                args.fsdp_size = train_cfg['fsdp_size']
            # Stage 1: train to_moments + decoder only (no LoRA, no VF loss)
            if 'stage1_steps' in train_cfg:
                args.stage1_steps = train_cfg['stage1_steps']
        
        if 'logging' in config:
            log_cfg = config['logging']
            if 'output_dir' in log_cfg:
                args.output_dir = log_cfg['output_dir']
            if 'log_every' in log_cfg:
                args.log_every = log_cfg['log_every']
            if 'eval_every' in log_cfg:
                args.eval_every = log_cfg['eval_every']
            if 'save_every' in log_cfg:
                args.save_every = log_cfg['save_every']
            if 'val_max_batches' in log_cfg:
                args.val_max_batches = log_cfg['val_max_batches']
            if 'debug' in log_cfg:
                args.debug = log_cfg['debug']
        
        if 'checkpoint' in config:
            ckpt_cfg = config['checkpoint']
            if 'vae_ckpt' in ckpt_cfg and ckpt_cfg['vae_ckpt']:
                args.vae_ckpt = ckpt_cfg['vae_ckpt']
        
        if 'vae' in config:
            vae_cfg = config['vae']
            if 'noise_tau' in vae_cfg:
                args.noise_tau = vae_cfg['noise_tau']
            if 'mask_channels' in vae_cfg:
                args.mask_channels = vae_cfg['mask_channels']
    
    # 处理 --no-xxx 参数
    if args.no_lora:
        args.lora_rank = 0
    if args.no_vf:
        args.vf_weight = 0.0
    if args.no_gan:
        args.gan_weight = 0.0
    
    # 检查必需参数
    if args.train_path is None:
        raise ValueError("--train-path is required (or set via config file)")
    
    # 解析 decoder block out channels
    args.dec_block_out_channels = tuple(int(x) for x in args.dec_block_out_channels.split(','))

    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    logger = create_logger(args.output_dir, rank)
    if rank == 0:
        logger.info(f"Starting training: {args}")

    # TensorBoard setup
    tb_writer = None
    if rank == 0 and HAS_TENSORBOARD:
        # 统一放到 tensorboard_logs 目录下，名字为 output_dir 最后一项，方便不同实验对比
        exp_name = os.path.basename(args.output_dir.rstrip('/'))
        tb_log_dir = os.path.join("tensorboard_logs", exp_name)
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logging to: {tb_log_dir}")

    # Precision settings
    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    if rank == 0:
        logger.info(f"AMP enabled: {use_bf16}, dtype: {amp_dtype}")

    # Dataset
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),  # [-1,1]
    ])
    val_transform = transforms.Compose([
        transforms.Lambda(lambda pil: center_crop_arr(pil, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),  # [-1,1]
    ])

    train_dataset = ImageFolder(args.train_path, transform=train_transform)
    
    # 验证集支持两种格式：ImageFolder（有子文件夹分类结构）或纯图片文件夹
    try:
        val_dataset = ImageFolder(args.val_path, transform=val_transform)
    except:
        # root 下面直接是图片，没有子文件夹
        class ImageDataset(Dataset):
            EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
            def __init__(self, root, transform=None):
                self.transform = transform
                self.paths = sorted([os.path.join(root, f) for f in os.listdir(root) 
                                     if os.path.splitext(f)[1].lower() in self.EXTS])
            def __len__(self):
                return len(self.paths)
            def __getitem__(self, idx):
                img = Image.open(self.paths[idx]).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, 0
        val_dataset = ImageDataset(args.val_path, transform=val_transform)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=4, pin_memory=True, drop_last=False)

    # 根据 encoder_type 设置默认 latent_channels
    if args.latent_channels is None:
        if args.encoder_type == "dinov3":
            args.latent_channels = 1280
        elif args.encoder_type == "siglip2":
            args.latent_channels = 768  # SigLIP2-base hidden size
    
    if rank == 0:
        logger.info(f"Encoder type: {args.encoder_type}")
        logger.info(f"Latent channels: {args.latent_channels}")
        logger.info(f"Decoder block out channels: {args.dec_block_out_channels}")
        logger.info(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
        logger.info(f"VF weight: {args.vf_weight}, GAN weight: {args.gan_weight}")

    # Model
    model = AutoencoderKL(
        # Encoder 配置
        encoder_type=args.encoder_type,
        dinov3_model_dir=args.dinov3_dir,
        siglip2_model_name=args.siglip2_model_name,
        
        image_size=args.image_size,
        patch_size=16,
        out_channels=3,

        latent_channels=args.latent_channels,
        target_latent_channels=None,
        spatial_downsample_factor=16,

        dec_block_out_channels=args.dec_block_out_channels,
        dec_layers_per_block=3,
        decoder_dropout=0.0,
        gradient_checkpointing=False,

        variational=True,
        kl_weight=args.kl_weight,

        noise_tau=args.noise_tau,
        random_masking_channel_ratio=args.mask_channels,

        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        enable_lora=(args.lora_rank > 0),
        vf_margin_cos=args.vf_m1,
        vf_margin_dms=args.vf_m2,
        vf_max_tokens=args.vf_max_tokens,
        vf_hyper=args.vf_weight,
        vf_use_adaptive_weight=True,
        training_mode="enc_dec",
        denormalize_decoder_output=False,
    ).to(device)

    # Load optional ckpt
    resume_ckpt = None  # 用于后续恢复 optimizer/discriminator/step
    if args.vae_ckpt and os.path.isfile(args.vae_ckpt):
        if rank == 0:
            logger.info(f"Loading weights from {args.vae_ckpt}")
        resume_ckpt = torch.load(args.vae_ckpt, map_location="cpu")
        if "state_dict" in resume_ckpt:
            state_dict = resume_ckpt["state_dict"]
        elif "model_state_dict" in resume_ckpt:
            state_dict = resume_ckpt["model_state_dict"]
        elif "model" in resume_ckpt:
            state_dict = resume_ckpt["model"]
        else:
            state_dict = resume_ckpt
            resume_ckpt = None  # 如果 ckpt 就是 state_dict，则不尝试恢复其他状态
        model.load_state_dict(state_dict, strict=False)

    # Freeze base encoder weights; train LoRA + decoder + moments head (+ optional downsample convs)
    requires_grad(model, False)

    # decoder train
    requires_grad(model.decoder, True)

    # moments head train
    requires_grad(model.to_moments, True)

    # extra convs
    for conv in model.latent_downsample_layers:
        requires_grad(conv, True)
    if model.channel_downsample_conv is not None:
        requires_grad(model.channel_downsample_conv, True)

    # LoRA params train (match by module types, not names)
    # 获取 encoder backbone (支持多种 encoder 类型)
    encoder_backbone = model.encoder.get_backbone()
    
    step = 0
    epoch = 0
    if resume_ckpt is not None:
        # 恢复 step
        if "step" in resume_ckpt:
            step = resume_ckpt["step"]
            if rank == 0:
                logger.info(f"Resuming from step {step}")
    # Stage 1 时冻结 LoRA，只训练 to_moments + decoder
    # Stage 2 时解冻 LoRA，一起训练
    in_stage1 = args.stage1_steps > 0 and step < args.stage1_steps
    enable_lora_training = not in_stage1 and args.lora_rank > 0
    
    for m in encoder_backbone.modules():
        # LoRALinear/LoRAConv2d contains lora_down/lora_up
        if hasattr(m, "lora_down") and hasattr(m, "lora_up"):
            for p in m.lora_down.parameters():
                p.requires_grad = enable_lora_training
            for p in m.lora_up.parameters():
                p.requires_grad = enable_lora_training
        # keep base frozen
        if hasattr(m, "base") and isinstance(getattr(m, "base"), (nn.Linear, nn.Conv2d)):
            for p in m.base.parameters():
                p.requires_grad = False

    lora_params = [p for n, p in encoder_backbone.named_parameters()
                   if p.requires_grad and (("lora_up" in n) or ("lora_down" in n))]
    
    if rank == 0:
        num_lora_params = sum(p.numel() for p in lora_params)
        logger.info(f"LoRA trainable parameters: {num_lora_params / 1e6:.2f}M")
        if args.stage1_steps > 0:
            logger.info(f"Stage 1 mode: training to_moments + decoder only (no LoRA, no VF loss) for {args.stage1_steps} steps")
    
    model.train()

    # Wrap model with DDP or FSDP
    use_fsdp = args.fsdp_size > 1
    if use_fsdp:
        fsdp_mixed_precision = None
        if use_bf16:
            fsdp_mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=fsdp_mixed_precision,
            device_id=local_rank,
            use_orig_params=True,
        )
        if rank == 0:
            logger.info(f"Using FSDP with sharding_strategy=FULL_SHARD, fsdp_size={args.fsdp_size}")
    else:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        if rank == 0:
            logger.info("Using DDP")

    # Optimizer for generator (VAE)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.5, 0.9),
        weight_decay=1e-2,
    )

    loss_lpips = LPIPSLoss(device)

    # ========== GAN Setup - PatchGAN ==========
    use_gan = args.gan_weight > 0.0
    discriminator = None
    opt_disc = None

    if use_gan:
        # Build PatchGAN discriminator
        discriminator = NLayerDiscriminator(
            in_channels=3,
            ndf=args.disc_ndf,
            n_layers=args.disc_n_layers,
            norm=args.disc_norm,
        ).to(device)

        # Wrap discriminator with DDP or FSDP
        if use_fsdp:
            fsdp_mixed_precision_disc = None
            if use_bf16:
                fsdp_mixed_precision_disc = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )
            discriminator = FSDP(
                discriminator,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=fsdp_mixed_precision_disc,
                device_id=local_rank,
                use_orig_params=True,
            )
        else:
            discriminator = DDP(discriminator, device_ids=[local_rank], find_unused_parameters=False)
        discriminator.train()

        opt_disc = torch.optim.AdamW(
            discriminator.parameters(),
            lr=args.disc_lr,
            betas=(0.5, 0.9),
            weight_decay=1e-2,
        )

        if rank == 0:
            num_disc_params = sum(p.numel() for p in discriminator.parameters())
            logger.info(f"PatchGAN Discriminator parameters: {num_disc_params / 1e6:.2f}M")
            logger.info(f"GAN weight: {args.gan_weight}, start step: {args.gan_start_step}")
            logger.info(f"Stage disc steps: {args.stage_disc_steps} (0 = joint training from start)")

    # ========== 从 checkpoint 恢复 optimizer/discriminator/step ==========
    if resume_ckpt is not None:
        # 恢复 VAE optimizer
        if "opt" in resume_ckpt:
            try:
                opt.load_state_dict(resume_ckpt["opt"])
                if rank == 0:
                    logger.info("Loaded VAE optimizer state from checkpoint")
            except Exception as e:
                if rank == 0:
                    logger.warning(f"Failed to load VAE optimizer state: {e}")
        
        # 恢复 discriminator 和其 optimizer
        if use_gan and discriminator is not None:
            if "discriminator" in resume_ckpt:
                try:
                    disc_state = resume_ckpt["discriminator"]
                    unwrap_model(discriminator).load_state_dict(disc_state)
                    if rank == 0:
                        logger.info("Loaded discriminator state from checkpoint")
                except Exception as e:
                    if rank == 0:
                        logger.warning(f"Failed to load discriminator state: {e}")
            
            if "opt_disc" in resume_ckpt and opt_disc is not None:
                try:
                    opt_disc.load_state_dict(resume_ckpt["opt_disc"])
                    if rank == 0:
                        logger.info("Loaded discriminator optimizer state from checkpoint")
                except Exception as e:
                    if rank == 0:
                        logger.warning(f"Failed to load discriminator optimizer state: {e}")

    # Loop
    if rank == 0:
        logger.info("Start Training Loop...")
    
    # Track stage transition
    lora_enabled_in_training = not (args.stage1_steps > 0 and step < args.stage1_steps)
    stage1_just_ended = False
    
    # Keep track of current lora_params for VF loss (updated during stage transition)
    current_lora_params = lora_params if lora_enabled_in_training else []

    while step < args.max_steps:
        train_sampler.set_epoch(epoch)
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)  # [-1,1]

            # Check if we need to transition from Stage 1 to Stage 2
            # Stage 1: train to_moments + decoder only (no LoRA, no VF loss)
            # Stage 2: train LoRA + to_moments + decoder (with VF loss)
            in_stage1 = args.stage1_steps > 0 and step < args.stage1_steps
            
            # Transition from Stage 1 to Stage 2: enable LoRA training
            if not in_stage1 and not lora_enabled_in_training and args.lora_rank > 0:
                lora_enabled_in_training = True
                stage1_just_ended = True
                # Get the unwrapped encoder backbone (after DDP/FSDP wrapping)
                model_inner_for_lora = unwrap_model(model)
                encoder_backbone_inner = model_inner_for_lora.encoder.get_backbone()
                # Unfreeze LoRA parameters
                for m in encoder_backbone_inner.modules():
                    if hasattr(m, "lora_down") and hasattr(m, "lora_up"):
                        for p in m.lora_down.parameters():
                            p.requires_grad = True
                        for p in m.lora_up.parameters():
                            p.requires_grad = True
                # Update optimizer to include LoRA params
                lora_params = [p for n, p in encoder_backbone_inner.named_parameters()
                               if p.requires_grad and (("lora_up" in n) or ("lora_down" in n))]
                current_lora_params = lora_params  # Update for VF loss computation
                opt.add_param_group({'params': lora_params, 'lr': args.lr})
                
                if rank == 0:
                    num_lora_params = sum(p.numel() for p in lora_params)
                    logger.info(f"=== Stage 1 -> Stage 2 Transition at step {step} ===")
                    logger.info(f"Enabled LoRA training: {num_lora_params / 1e6:.2f}M params added")
                    logger.info(f"VF loss now active (weight={args.vf_weight})")
                dist.barrier()

            # Determine if GAN is active this step
            gan_active = use_gan and step >= args.gan_start_step

            # Determine training mode:
            # stage_disc: only train discriminator, generator frozen
            # stage_joint: train both together
            in_stage_disc = gan_active and (step - args.gan_start_step) < args.stage_disc_steps
            in_stage_joint = gan_active and not in_stage_disc

            # ========== Generator (VAE) Forward & Update ==========
            # Determine if we need to update generator
            # - Before GAN starts (not gan_active): train VAE normally
            # - stage_disc: freeze VAE, only train discriminator (no_grad to save memory)
            # - stage_joint: train both VAE and discriminator
            update_generator = (not gan_active) or in_stage_joint

            # Initialize loss values for logging
            loss = torch.tensor(0.0, device=device)
            loss_rec = torch.tensor(0.0, device=device)
            l1_loss = torch.tensor(0.0, device=device)
            lpips_loss = torch.tensor(0.0, device=device)
            kl_loss = torch.tensor(0.0, device=device)
            vf_raw = torch.tensor(0.0, device=device)
            w_adapt = torch.tensor(1.0, device=device)
            loss_vf = torch.tensor(0.0, device=device)
            gan_loss = torch.tensor(0.0, device=device)
            d_weight = torch.tensor(0.0, device=device)

            if in_stage_disc:
                # stage_disc: only forward VAE to get recon, no grad to save memory
                if discriminator is not None:
                    discriminator.eval()
                    requires_grad(discriminator, False)

                with torch.no_grad():
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_bf16):
                        out = model(x)
                        recon = out.sample  # [-1,1]
            else:
                # Normal training or stage_joint: compute full loss with gradients
                if gan_active and discriminator is not None:
                    discriminator.eval()
                    requires_grad(discriminator, False)

                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_bf16):
                    out = model(x)
                    recon = out.sample  # [-1,1]
                    posterior = out.posterior

                    # Recon loss (keep in [-1,1] for LPIPS)
                    l1_loss = F.l1_loss(recon, x)
                    lpips_loss = loss_lpips(x, recon)
                    loss_rec = args.l1_weight * l1_loss + args.lpips_weight * lpips_loss

                    # KL
                    kl_loss = posterior.kl().mean()

                    # VF loss: student (LoRA on) vs ref (LoRA off)
                    # Stage 1: skip VF loss (only recon + GAN)
                    # Stage 2: enable VF loss
                    model_inner = unwrap_model(model)
                    
                    if in_stage1 or args.vf_weight == 0.0:
                        # Stage 1 or VF disabled: no VF loss
                        vf_raw = torch.tensor(0.0, device=device)
                        w_adapt = torch.tensor(0.0, device=device)
                        loss_vf = torch.tensor(0.0, device=device)
                    else:
                        # Stage 2: compute VF loss
                        z_feat = model_inner.encode_features(x, use_lora=True)  # [B,C,S,S]
                        with torch.no_grad():
                            f_feat = model_inner.encode_features(x, use_lora=False)
                        vf_raw = model_inner.compute_vf_loss(z_feat, f_feat)

                        if model_inner.vf_use_adaptive_weight:
                            w_adapt = model_inner.adaptive_weight(loss_rec, vf_raw, current_lora_params)
                        else:
                            w_adapt = torch.tensor(1.0, device=device)
                        w_adapt = torch.clamp(w_adapt, max=3.0)
                        loss_vf = model_inner.vf_hyper * w_adapt * vf_raw
                    
                    loss = loss_rec + model_inner.kl_weight * kl_loss + loss_vf

                    # GAN Generator loss (only in stage_joint)
                    if in_stage_joint:
                        logits_fake = discriminator(recon)
                        gan_loss = g_hinge_loss(logits_fake)
                        # Compute adaptive weight using decoder's last layer
                        last_layer = model_inner.decoder.conv_out.weight
                        d_weight = calculate_adaptive_weight(
                            loss_rec + model_inner.kl_weight * kl_loss + loss_vf,
                            gan_loss,
                            last_layer,
                            args.max_d_weight,
                        )
                        loss = loss + args.gan_weight * d_weight * gan_loss

                loss.backward()
                opt.step()

            # ========== Discriminator Update ==========
            disc_loss = torch.tensor(0.0, device=device)
            logits_real_mean = torch.tensor(0.0, device=device)
            logits_fake_mean = torch.tensor(0.0, device=device)

            if gan_active and discriminator is not None:
                discriminator.train()
                requires_grad(discriminator, True)
                opt_disc.zero_grad(set_to_none=True)

                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_bf16):
                    with torch.no_grad():
                        recon_detach = recon.detach()
                        # Discretize for discriminator input
                        recon_detach = recon_detach.clamp(-1.0, 1.0)
                        recon_detach = torch.round((recon_detach + 1.0) * 127.5) / 127.5 - 1.0

                    logits_fake_d = discriminator(recon_detach)
                    logits_real_d = discriminator(x)
                    disc_loss = d_hinge_loss(logits_real_d, logits_fake_d)
                    logits_real_mean = logits_real_d.detach().mean()
                    logits_fake_mean = logits_fake_d.detach().mean()

                disc_loss.backward()
                opt_disc.step()
            # print(f"disc_loss: {disc_loss.item()}")
            step += 1

            if step % args.log_every == 0 and rank == 0:
                # Determine stage string
                if in_stage1:
                    stage_str = "stage1"
                elif in_stage_disc:
                    stage_str = "stage_disc"
                elif in_stage_joint:
                    stage_str = "stage_joint"
                else:
                    stage_str = "stage2" if args.stage1_steps > 0 else "no_gan"

                log_msg = (
                    f"Step {step} [{stage_str}]: total={loss.item():.4f} | rec={loss_rec.item():.4f} "
                    f"(L1={l1_loss.item():.4f}, LPIPS={lpips_loss.item():.4f}) "
                    f"| KL={kl_loss.item():.4f} "
                    f"| VF={vf_raw.item():.4f} w={w_adapt.item():.3f} vf_term={loss_vf.item():.4f}"
                )
                if gan_active:
                    log_msg += (
                        f" | GAN={gan_loss.item():.4f} d_w={d_weight.item():.3f} "
                        f"D_loss={disc_loss.item():.4f} D_real={logits_real_mean.item():.3f} D_fake={logits_fake_mean.item():.3f}"
                    )
                logger.info(log_msg)

                # TensorBoard logging
                if tb_writer is not None:
                    # Total loss
                    tb_writer.add_scalar("Loss/total", loss.item(), step)
                    
                    # Reconstruction losses
                    tb_writer.add_scalar("Loss/rec", loss_rec.item(), step)
                    tb_writer.add_scalar("Loss/L1", l1_loss.item(), step)
                    tb_writer.add_scalar("Loss/LPIPS", lpips_loss.item(), step)
                    
                    # KL loss
                    tb_writer.add_scalar("Loss/KL", kl_loss.item(), step)
                    
                    # VF loss
                    tb_writer.add_scalar("Loss/VF_raw", vf_raw.item(), step)
                    tb_writer.add_scalar("Loss/VF_term", loss_vf.item(), step)
                    tb_writer.add_scalar("Weight/VF_adaptive", w_adapt.item(), step)
                    
                    # Stage indicator (0=stage1, 1=stage2)
                    tb_writer.add_scalar("Training/stage", 0 if in_stage1 else 1, step)
                    tb_writer.add_scalar("Training/lora_enabled", 0 if in_stage1 else 1, step)
                    
                    # GAN losses (when active)
                    if gan_active:
                        tb_writer.add_scalar("Loss/GAN_gen", gan_loss.item(), step)
                        tb_writer.add_scalar("Loss/GAN_disc", disc_loss.item(), step)
                        tb_writer.add_scalar("Weight/GAN_d_weight", d_weight.item(), step)
                        tb_writer.add_scalar("Discriminator/logits_real", logits_real_mean.item(), step)
                        tb_writer.add_scalar("Discriminator/logits_fake", logits_fake_mean.item(), step)

            if step % args.eval_every == 0:
                run_validation(model, val_loader, device, step, args, logger, use_fsdp=use_fsdp, tb_writer=tb_writer)

            if step % args.save_every == 0 and rank == 0:
                save_path = os.path.join(args.output_dir, f"step_{step}.pth")
                model_state = get_model_state_dict(model)
                ckpt_state = {
                    "model": model_state, 
                    "step": step,
                    "opt": opt.state_dict(),  # 保存 VAE optimizer
                }
                if use_gan and discriminator is not None:
                    disc_state = get_model_state_dict(discriminator)
                    ckpt_state["discriminator"] = disc_state
                    ckpt_state["opt_disc"] = opt_disc.state_dict()
                torch.save(ckpt_state, save_path)
                logger.info(f"Saved checkpoint to {save_path}")

            if step >= args.max_steps:
                break

        epoch += 1

    # Close TensorBoard writer
    if tb_writer is not None:
        tb_writer.close()

    cleanup_ddp()


if __name__ == "__main__":
    main()
