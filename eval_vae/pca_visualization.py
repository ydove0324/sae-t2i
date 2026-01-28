# pca_visualization.py
# PCA visualization for VAE encoder features on ImageNet

import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

# 设置 torch 缓存路径 (可选)
os.environ["TORCH_HOME"] = "/cpfs01/huangxu/.cache/torch"

# 添加当前目录以导入项目模块
sys.path.append(".")

# 导入 VAE 工具函数
from models.rae.utils.vae_utils import load_vae


# ==========================================
#              Helper Functions
# ==========================================

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


@torch.no_grad()
def extract_features(vae, x: torch.Tensor, pooling: str = "avg") -> torch.Tensor:
    """Extract features from VAE encoder."""
    feat = vae.encode_features(x, use_lora=True)  # [B, C, H, W]
    B, C, H, W = feat.shape
    
    if pooling == "avg":
        features = feat.mean(dim=[2, 3])  # [B, C]
    elif pooling == "flatten":
        features = feat.view(B, -1)  # [B, C*H*W]
    elif pooling == "tokens":
        features = feat.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
    else:
        raise ValueError(f"Unknown pooling: {pooling}")
    
    return features


# ==========================================
#           ImageNet class names (subset)
# ==========================================

# 一些常见的 ImageNet 类别名称（用于可视化标签）
IMAGENET_CLASSES = {
    0: "tench", 1: "goldfish", 2: "great white shark",
    207: "golden retriever", 208: "Labrador retriever",
    281: "tabby cat", 282: "tiger cat", 283: "Persian cat",
    291: "lion", 292: "tiger",
    323: "monarch butterfly", 324: "cabbage butterfly",
    386: "African elephant", 387: "Indian elephant",
    417: "balloon", 418: "basketball",
    463: "bucket", 464: "buckle",
    508: "computer keyboard", 509: "computer mouse",
    530: "digital clock", 531: "digital watch",
    621: "laptop", 622: "lawn mower",
    717: "pickup truck", 718: "pier",
    779: "school bus", 780: "scoreboard",
    817: "sports car", 818: "spotlight",
    920: "traffic light", 921: "trailer truck",
    950: "orange", 951: "orange_tree",
    954: "banana", 955: "bar",
}


def get_class_name(class_idx):
    """Get class name or default to index."""
    return IMAGENET_CLASSES.get(class_idx, f"class_{class_idx}")


# ==========================================
#                   Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="PCA visualization for VAE encoder features")
    
    # Data
    parser.add_argument("--data-path", type=str, required=True, help="Path to ImageNet validation set.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of random samples.")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes to visualize.")
    parser.add_argument("--samples-per-class", type=int, default=100, help="Samples per class.")
    
    # VAE
    parser.add_argument("--vae-ckpt", type=str, required=True, help="Path to VAE checkpoint.")
    parser.add_argument("--encoder-type", type=str, default="dinov3", choices=["dinov3", "siglip2"])
    parser.add_argument("--dinov3-dir", type=str, default="/cpfs01/huangxu/models/dinov3")
    parser.add_argument("--siglip2-model-name", type=str, default="google/siglip2-base-patch16-256")
    parser.add_argument("--lora-rank", type=int, default=256)
    parser.add_argument("--lora-alpha", type=int, default=256)
    
    # Feature extraction
    parser.add_argument("--pooling", type=str, default="avg", choices=["avg", "flatten"])
    parser.add_argument("--batch-size", type=int, default=32)
    
    # Visualization
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "tsne", "both"])
    parser.add_argument("--n-components", type=int, default=2, help="PCA components (2 or 3)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="results/pca_visualization")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 50)
    print(" PCA Visualization for VAE Encoder Features")
    print("=" * 50)
    print(f" Encoder:      {args.encoder_type}")
    print(f" VAE Ckpt:     {args.vae_ckpt}")
    print(f" Num Classes:  {args.num_classes}")
    print(f" Samples/Class:{args.samples_per_class}")
    print(f" Method:       {args.method}")
    print("=" * 50)
    
    # 1. Load VAE
    print("\nLoading VAE...")
    
    if args.encoder_type == "dinov3":
        latent_channels = 1280
        dec_block_out_channels = (1280, 1024, 512, 256, 128)
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
    
    if args.encoder_type == "dinov3":
        vae_model_params["dinov3_model_dir"] = args.dinov3_dir
    elif args.encoder_type == "siglip2":
        vae_model_params["siglip2_model_name"] = args.siglip2_model_name
    
    vae = load_vae(
        args.vae_ckpt,
        device,
        encoder_type=args.encoder_type,
        decoder_type="cnn_decoder",
        model_params=vae_model_params,
        verbose=True,
        skip_to_moments=False,
    )
    vae.eval()
    
    # 2. Load dataset
    print("\nLoading dataset...")
    
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),  # [-1, 1]
    ])
    
    dataset = ImageFolder(args.data_path, transform=transform)
    print(f"Total dataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    
    # 3. Select random classes and samples
    print("\nSelecting samples...")
    
    # Get all class indices
    all_classes = list(range(len(dataset.classes)))
    selected_classes = random.sample(all_classes, min(args.num_classes, len(all_classes)))
    
    # Get indices for each selected class
    class_to_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        if label in selected_classes:
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
    
    # Sample from each class
    selected_indices = []
    selected_labels = []
    for class_idx in selected_classes:
        indices = class_to_indices.get(class_idx, [])
        sampled = random.sample(indices, min(args.samples_per_class, len(indices)))
        selected_indices.extend(sampled)
        selected_labels.extend([class_idx] * len(sampled))
    
    print(f"Selected {len(selected_indices)} samples from {len(selected_classes)} classes")
    
    # Create subset
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 4. Extract features
    print("\nExtracting features...")
    
    all_features = []
    all_labels = []
    label_idx = 0
    
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Extracting"):
            x = x.to(device)
            features = extract_features(vae, x, pooling=args.pooling)
            all_features.append(features.cpu().numpy())
            
            batch_size = x.size(0)
            all_labels.extend(selected_labels[label_idx:label_idx + batch_size])
            label_idx += batch_size
    
    features_np = np.concatenate(all_features, axis=0)
    labels_np = np.array(all_labels)
    
    print(f"Feature shape: {features_np.shape}")
    
    # 5. Dimensionality reduction
    print("\nPerforming dimensionality reduction...")
    
    results = {}
    
    if args.method in ["pca", "both"]:
        print("  Running PCA...")
        pca = PCA(n_components=args.n_components)
        features_pca = pca.fit_transform(features_np)
        results["pca"] = features_pca
        
        # Print explained variance
        print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"  Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
        
        # Calculate cumulative explained variance for different numbers of components
        print("\n  Cumulative explained variance analysis:")
        max_components = min(100, features_np.shape[1], features_np.shape[0])
        pca_full = PCA(n_components=max_components)
        pca_full.fit(features_np)
        
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        checkpoints = [5, 10, 32, 64, 100]
        for k in checkpoints:
            if k <= max_components:
                print(f"    Top {k:3d} components: {cumsum[k-1]:.4f} ({cumsum[k-1]*100:.2f}%)")
            else:
                print(f"    Top {k:3d} components: N/A (only {max_components} available)")
    
    if args.method in ["tsne", "both"]:
        print("  Running t-SNE...")
        # For t-SNE, first reduce with PCA if feature dim is high
        if features_np.shape[1] > 50:
            pca_pre = PCA(n_components=50)
            features_pre = pca_pre.fit_transform(features_np)
        else:
            features_pre = features_np
        
        # n_iter was renamed to max_iter in newer sklearn versions
        try:
            tsne = TSNE(n_components=2, perplexity=30, random_state=args.seed, max_iter=1000)
        except TypeError:
            tsne = TSNE(n_components=2, perplexity=30, random_state=args.seed)
        features_tsne = tsne.fit_transform(features_pre)
        results["tsne"] = features_tsne
    
    # 6. Visualization
    print("\nCreating visualizations...")
    
    # Color map
    unique_labels = np.unique(labels_np)
    colors = cm.get_cmap('tab10' if len(unique_labels) <= 10 else 'tab20')(
        np.linspace(0, 1, len(unique_labels))
    )
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Create class name mapping
    class_names = {label: dataset.classes[label] for label in unique_labels}
    
    for method_name, features_reduced in results.items():
        fig, ax = plt.subplots(figsize=(12, 10))
        
        for label in unique_labels:
            mask = labels_np == label
            ax.scatter(
                features_reduced[mask, 0],
                features_reduced[mask, 1],
                c=[label_to_color[label]],
                label=f"{class_names[label]} ({label})",
                alpha=0.6,
                s=30,
            )
        
        ax.set_xlabel(f"{method_name.upper()} Component 1", fontsize=12)
        ax.set_ylabel(f"{method_name.upper()} Component 2", fontsize=12)
        ax.set_title(f"VAE Encoder Features - {method_name.upper()}\n({args.encoder_type}, {args.pooling} pooling)", fontsize=14)
        
        # Legend outside
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        save_path = os.path.join(args.output_dir, f"{method_name}_2d.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
    
    # 3D PCA plot if n_components >= 3
    if args.method in ["pca", "both"] and args.n_components >= 3:
        print("  Creating 3D PCA plot...")
        pca_3d = PCA(n_components=3)
        features_pca_3d = pca_3d.fit_transform(features_np)
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for label in unique_labels:
            mask = labels_np == label
            ax.scatter(
                features_pca_3d[mask, 0],
                features_pca_3d[mask, 1],
                features_pca_3d[mask, 2],
                c=[label_to_color[label]],
                label=f"{class_names[label]}",
                alpha=0.6,
                s=20,
            )
        
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        ax.set_zlabel("PC3", fontsize=10)
        ax.set_title(f"VAE Encoder Features - PCA 3D\n({args.encoder_type})", fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        save_path = os.path.join(args.output_dir, "pca_3d.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
    
    # 7. Save sample images with their features
    print("\nSaving sample images grid...")
    
    # Select a few samples from each class for visualization
    n_samples_per_class = 5
    fig, axes = plt.subplots(len(unique_labels), n_samples_per_class, figsize=(n_samples_per_class * 2, len(unique_labels) * 2))
    
    for i, label in enumerate(unique_labels):
        mask = labels_np == label
        indices = np.where(mask)[0][:n_samples_per_class]
        
        for j, idx in enumerate(indices):
            # Get original image
            orig_idx = selected_indices[idx]
            img_path, _ = dataset.samples[orig_idx]
            img = Image.open(img_path).convert('RGB')
            img = center_crop_arr(img, args.image_size)
            
            ax = axes[i, j] if len(unique_labels) > 1 else axes[j]
            ax.imshow(img)
            ax.axis('off')
            if j == 0:
                ax.set_title(f"{class_names[label][:15]}...", fontsize=8)
    
    plt.suptitle(f"Sample Images from Selected Classes", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(args.output_dir, "sample_images.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    # 8. Save numerical results
    print("\nSaving numerical results...")
    
    np.savez(
        os.path.join(args.output_dir, "features.npz"),
        features=features_np,
        labels=labels_np,
        class_names=[class_names[l] for l in unique_labels],
        unique_labels=unique_labels,
    )
    
    # Save config
    with open(os.path.join(args.output_dir, "config.txt"), "w") as f:
        f.write(f"VAE Checkpoint: {args.vae_ckpt}\n")
        f.write(f"Encoder Type: {args.encoder_type}\n")
        f.write(f"Pooling: {args.pooling}\n")
        f.write(f"Num Classes: {len(unique_labels)}\n")
        f.write(f"Total Samples: {len(features_np)}\n")
        f.write(f"Feature Dimension: {features_np.shape[1]}\n")
        if "pca" in results:
            f.write(f"PCA Explained Variance: {pca.explained_variance_ratio_}\n")
    
    print("\n" + "=" * 50)
    print(" Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
