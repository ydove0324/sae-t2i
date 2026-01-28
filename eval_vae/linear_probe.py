# linear_probe.py
# Linear Probing evaluation for VAE encoder on ImageNet classification

import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
# 设置 torch 缓存路径 (可选)
os.environ["TORCH_HOME"] = "/share/project/huangxu/.cache/torch"

# 添加当前目录以导入项目模块
sys.path.append(".")

# 导入 VAE 工具函数
from models.rae.utils.vae_utils import load_vae, get_normalize_fn


# ==========================================
#              Helper Functions
# ==========================================

def setup_ddp():
    if "LOCAL_RANK" not in os.environ:
        # Fallback for single GPU run
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


# ==========================================
#           Linear Classifier
# ==========================================

class LinearClassifier(nn.Module):
    """Linear classifier for feature evaluation."""
    def __init__(self, in_features: int, num_classes: int = 1000):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ==========================================
#           Feature Extraction
# ==========================================

@torch.no_grad()
def extract_features(vae, x: torch.Tensor, pooling: str = "avg") -> torch.Tensor:
    """
    Extract features from VAE encoder.
    
    Args:
        vae: AutoencoderKL model
        x: Input images [B, 3, H, W] in [-1, 1]
        pooling: Feature pooling method
            - "avg": Global average pooling over spatial dims
            - "cls": Use CLS token (DINOv3 only)
            - "flatten": Flatten all features
    
    Returns:
        Features [B, D]
    """
    # Get encoder features (before to_moments)
    feat = vae.encode_features(x, use_lora=True)  # [B, C, H, W]
    
    B, C, H, W = feat.shape
    
    if pooling == "avg":
        # Global average pooling
        features = feat.mean(dim=[2, 3])  # [B, C]
    elif pooling == "flatten":
        # Flatten all spatial features
        features = feat.view(B, -1)  # [B, C*H*W]
    elif pooling == "max":
        # Global max pooling
        features = feat.amax(dim=[2, 3])  # [B, C]
    elif pooling == "avg_max":
        # Concatenate avg and max pooling
        avg_feat = feat.mean(dim=[2, 3])
        max_feat = feat.amax(dim=[2, 3])
        features = torch.cat([avg_feat, max_feat], dim=1)  # [B, 2*C]
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")
    
    return features


def get_feature_dim(vae, pooling: str, image_size: int = 256) -> int:
    """Get the feature dimension for a given pooling method."""
    # Infer from encoder
    latent_channels = vae.original_latent_channels
    spatial_size = image_size // 16  # Assuming 16x downsample
    
    if pooling == "avg":
        return latent_channels
    elif pooling == "max":
        return latent_channels
    elif pooling == "avg_max":
        return latent_channels * 2
    elif pooling == "flatten":
        return latent_channels * spatial_size * spatial_size
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")


# ==========================================
#           Training Functions
# ==========================================

def train_one_epoch(
    vae,
    classifier,
    train_loader,
    optimizer,
    device,
    epoch,
    pooling,
    rank,
):
    """Train linear classifier for one epoch."""
    classifier.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    if rank == 0:
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        iterator = train_loader
    
    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)
        
        # Extract features (frozen encoder)
        with torch.no_grad():
            features = extract_features(vae, x, pooling=pooling)
        
        # Forward through classifier
        logits = classifier(features)
        loss = F.cross_entropy(logits, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_samples += x.size(0)
        
        if rank == 0:
            iterator.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * total_correct / total_samples:.2f}%"
            })
    
    # Aggregate across GPUs
    stats = torch.tensor([total_loss, total_correct, total_samples], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    
    avg_loss = stats[0].item() / stats[2].item()
    accuracy = stats[1].item() / stats[2].item()
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    vae,
    classifier,
    val_loader,
    device,
    pooling,
    rank,
):
    """Evaluate linear classifier on validation set."""
    classifier.eval()
    
    total_correct = 0
    total_correct_top5 = 0
    total_samples = 0
    
    if rank == 0:
        iterator = tqdm(val_loader, desc="Evaluating")
    else:
        iterator = val_loader
    
    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)
        
        # Extract features
        features = extract_features(vae, x, pooling=pooling)
        
        # Forward
        logits = classifier(features)
        
        # Top-1 accuracy
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        
        # Top-5 accuracy
        _, top5_pred = logits.topk(5, dim=1)
        top5_correct = top5_pred.eq(y.view(-1, 1).expand_as(top5_pred))
        total_correct_top5 += top5_correct.any(dim=1).sum().item()
        
        total_samples += x.size(0)
    
    # Aggregate across GPUs
    stats = torch.tensor([total_correct, total_correct_top5, total_samples], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    
    top1_acc = stats[0].item() / stats[2].item()
    top5_acc = stats[1].item() / stats[2].item()
    
    return top1_acc, top5_acc


# ==========================================
#                   Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Linear Probing evaluation for VAE encoder")
    
    # Data
    parser.add_argument("--train-path", type=str, required=True, help="Path to ImageNet train set.")
    parser.add_argument("--val-path", type=str, required=True, help="Path to ImageNet validation set.")
    parser.add_argument("--image-size", type=int, default=256)
    
    # VAE
    parser.add_argument("--vae-ckpt", type=str, required=True, help="Path to VAE checkpoint.")
    parser.add_argument("--encoder-type", type=str, default="dinov3", choices=["dinov3", "dinov3_vitl", "siglip2"])
    parser.add_argument("--dinov3-dir", type=str, default="/cpfs01/huangxu/models/dinov3")
    parser.add_argument("--siglip2-model-name", type=str, default="google/siglip2-base-patch16-256")
    parser.add_argument("--lora-rank", type=int, default=256)
    parser.add_argument("--lora-alpha", type=int, default=256)
    
    # Feature extraction
    parser.add_argument("--pooling", type=str, default="avg", 
                        choices=["avg", "max", "avg_max", "flatten"],
                        help="Feature pooling method.")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size per GPU.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "step"])
    parser.add_argument("--warmup-epochs", type=int, default=5)
    
    # Output
    parser.add_argument("--output-dir", type=str, default="results/linear_probe")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs.")
    parser.add_argument("--seed", type=int, default=42)
    
    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to resume checkpoint.")
    
    args = parser.parse_args()

    # 1. 初始化 DDP
    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # 2. 创建输出目录
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print("=" * 50)
        print(" Linear Probing Evaluation")
        print("=" * 50)
        print(f" Encoder:   {args.encoder_type}")
        print(f" VAE Ckpt:  {args.vae_ckpt}")
        print(f" Pooling:   {args.pooling}")
        print(f" Epochs:    {args.epochs}")
        print(f" LR:        {args.lr}")
        print(f" Batch:     {args.batch_size} x {world_size} GPUs")
        print("=" * 50)
    
    dist.barrier()

    # 3. 加载 VAE (frozen encoder)
    if rank == 0:
        print("Loading VAE...")
    
    # 设置 latent_channels
    if args.encoder_type == "dinov3":
        latent_channels = 1280
        dec_block_out_channels = (1280, 1024, 512, 256, 128)
    elif args.encoder_type == "dinov3_vitl":
        latent_channels = 1024
        dec_block_out_channels = (1024, 768, 512, 256, 128)
    else:  # siglip2
        latent_channels = 768
        dec_block_out_channels = (768, 512, 256, 128, 64)
    
    vae_model_params = {
        "encoder_type": args.encoder_type,
        "image_size": args.image_size,
        "patch_size": 16,
        "out_channels": 3,
        "latent_channels": latent_channels,
        "target_latent_channels": None,
        "spatial_downsample_factor": 16,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "dec_block_out_channels": dec_block_out_channels,
        "dec_layers_per_block": 3,
        "decoder_dropout": 0.0,
        "gradient_checkpointing": False,
        "denormalize_decoder_output": False,
    }
    
    if args.encoder_type == "dinov3" or args.encoder_type == "dinov3_vitl":
        vae_model_params["dinov3_model_dir"] = args.dinov3_dir
    elif args.encoder_type == "siglip2":
        vae_model_params["siglip2_model_name"] = args.siglip2_model_name
    
    vae = load_vae(
        args.vae_ckpt,
        device,
        encoder_type=args.encoder_type,
        decoder_type="cnn_decoder",
        model_params=vae_model_params,
        verbose=(rank == 0),
    )
    
    # Freeze VAE
    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()
    
    # 4. 创建 Linear Classifier
    feature_dim = get_feature_dim(vae, args.pooling, args.image_size)
    if rank == 0:
        print(f"Feature dimension: {feature_dim}")
    
    classifier = LinearClassifier(in_features=feature_dim, num_classes=1000).to(device)
    classifier = DDP(classifier, device_ids=[local_rank])
    
    # 5. 优化器和调度器
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    
    # Cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # 6. 数据集
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),  # [-1, 1]
    ])
    
    val_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),  # [-1, 1]
    ])
    
    train_dataset = ImageFolder(args.train_path, transform=train_transform)
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )
    
    if rank == 0:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
    
    # 7. 恢复训练
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume is not None and os.path.exists(args.resume):
        if rank == 0:
            print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        classifier.module.load_state_dict(ckpt["classifier"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
    
    # 8. 训练循环
    if rank == 0:
        print("\nStarting Training...")
        log_file = open(os.path.join(args.output_dir, "train_log.txt"), "a")
    
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            vae, classifier, train_loader, optimizer, device, epoch, args.pooling, rank
        )
        
        # Evaluate
        val_top1, val_top5 = evaluate(
            vae, classifier, val_loader, device, args.pooling, rank
        )
        
        scheduler.step()
        
        # Log
        if rank == 0:
            lr = optimizer.param_groups[0]["lr"]
            msg = (
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}% | "
                f"Val Top-1: {100*val_top1:.2f}% | Val Top-5: {100*val_top5:.2f}% | "
                f"LR: {lr:.6f}"
            )
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()
            
            # Save best
            if val_top1 > best_acc:
                best_acc = val_top1
                torch.save({
                    "epoch": epoch,
                    "classifier": classifier.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_acc": best_acc,
                    "args": vars(args),
                }, os.path.join(args.output_dir, "best.pth"))
                print(f"  -> New best! Top-1: {100*best_acc:.2f}%")
            
            # Save periodic
            if (epoch + 1) % args.save_every == 0:
                torch.save({
                    "epoch": epoch,
                    "classifier": classifier.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_acc": best_acc,
                    "args": vars(args),
                }, os.path.join(args.output_dir, f"epoch_{epoch:03d}.pth"))
        
        dist.barrier()
    
    # 9. 最终结果
    if rank == 0:
        print("\n" + "=" * 50)
        print(" FINAL RESULTS")
        print("=" * 50)
        print(f" Best Top-1 Accuracy: {100*best_acc:.2f}%")
        print("=" * 50)
        
        log_file.write(f"\nBest Top-1 Accuracy: {100*best_acc:.2f}%\n")
        log_file.close()
        
        # Save final results
        with open(os.path.join(args.output_dir, "final_results.txt"), "w") as f:
            f.write(f"Encoder: {args.encoder_type}\n")
            f.write(f"VAE Checkpoint: {args.vae_ckpt}\n")
            f.write(f"Pooling: {args.pooling}\n")
            f.write(f"Feature Dim: {feature_dim}\n")
            f.write(f"Best Top-1 Accuracy: {100*best_acc:.2f}%\n")
    
    dist.barrier()
    cleanup_ddp()


if __name__ == "__main__":
    main()
