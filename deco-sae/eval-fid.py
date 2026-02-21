from torch_fidelity import calculate_metrics
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import os
import shutil
import torch
import numpy as np
import argparse

os.environ["TORCH_HOME"] = "/cpfs01/huangxu/.cache/torch"
os.environ["http_proxy"] = "localhost:27890"
os.environ["https_proxy"] = "localhost:27890"

# 解析命令行参数
parser = argparse.ArgumentParser(description="Evaluate FID, IS, Precision, and Recall")
parser.add_argument("--force-flatten", action="store_true", help="强制重新展开目录，即使已存在")
args = parser.parse_args()

# 路径配置
source_dir = "results_dit/deco_dinov2_base_dit_xl_hf_zero_mode_256dim/eval_samples/step_0050000"
flat_dir = "results_dit/deco_dinov2_base_dit_xl_hf_zero_mode_256dim/eval_samples/step_0050000_flat/"
# 使用预计算的 ImageNet 256x256 统计文件（用于 FID）
ref_npz_path = "/cpfs01/huangxu/SAE/VIRTUAL_imagenet256_labeled.npz"
# 256x256 的 ImageNet 验证集（用于 Precision/Recall）
ref_images_256_path = "/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/val_256/"


def create_flat_temp_dir(source_root, temp_dir):
    """展平目录结构，将所有图片放到同一层级"""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    count = 0
    for root, dirs, files in os.walk(source_root):
        if os.path.abspath(root) == os.path.abspath(temp_dir):
            continue
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.abspath(os.path.join(root, file))
                parent_name = os.path.basename(root)
                new_name = f"{parent_name}_{file}"
                dst_path = os.path.join(temp_dir, new_name)
                try:
                    os.symlink(src_path, dst_path)
                    count += 1
                except OSError:
                    shutil.copy(src_path, dst_path)
                    count += 1
    return count


# 1. 展平目录（如果已存在则跳过，除非指定 --force-flatten）
print("=" * 60)
print("Step 1: Checking/Flattening directory structure...")

# 检查 flat_dir 是否已存在且有图片
if os.path.exists(flat_dir) and not args.force_flatten:
    existing_files = [f for f in os.listdir(flat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(existing_files)
    if num_images > 0:
        print(f"  flat_dir already exists with {num_images} images, skipping flatten.")
    else:
        print("  flat_dir exists but empty, re-flattening...")
        num_images = create_flat_temp_dir(source_dir, flat_dir)
        print(f"  Total images: {num_images}")
else:
    if args.force_flatten:
        print("  Force flatten enabled, re-creating flat_dir...")
    else:
        print("  Creating flat_dir...")
    num_images = create_flat_temp_dir(source_dir, flat_dir)
    print(f"  Total images: {num_images}")

print("=" * 60)

# 2. 计算 FID + IS 一起（使用 torch_fidelity，避免重复加载模型）
print("\nStep 2: Calculating FID & IS...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载参考统计（用于打印 mean/std）
print("  Loading reference statistics...")
ref_stats = np.load(ref_npz_path)
ref_mu, ref_sigma = ref_stats['mu'], ref_stats['sigma']

# 使用 pytorch_fid 计算 FID（它支持 npz 文件）
print("  Computing FID...")
fid_value = fid_score.calculate_fid_given_paths(
    paths=[flat_dir, ref_npz_path],
    batch_size=50,
    device=device,
    dims=2048,
    num_workers=8
)

# 使用 torch_fidelity 计算 IS（它会复用 Inception 特征缓存）
print("  Computing IS...")
is_metrics = calculate_metrics(
    input1=flat_dir,
    cuda=True,
    isc=True,           # Inception Score
    isc_splits=10,
    verbose=True,
)

# 3. 打印结果
print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"Source: {source_dir}")
print(f"Reference (FID): {ref_npz_path}")
print(f"Total generated images: {num_images}")
print("-" * 60)

print(f"\nFID (Fréchet Inception Distance): {fid_value:.4f}")
print(f"  Reference features - mean: {ref_mu.mean():.4f}, std: {ref_mu.std():.4f}")

if 'inception_score_mean' in is_metrics:
    print(f"\nIS (Inception Score): {is_metrics['inception_score_mean']:.4f} ± {is_metrics['inception_score_std']:.4f}")
    print("  (IS measures: quality via confident predictions + diversity via class coverage)")

print("=" * 60)

# 4. 计算 Precision/Recall (需要 256x256 的 ImageNet 验证集)
# 先运行 resize_imagenet_val.py 生成 val_256 目录
if os.path.exists(ref_images_256_path):
    print("\nStep 4: Calculating Precision & Recall...")
    pr_metrics = calculate_metrics(
        input1=flat_dir,
        input2=ref_images_256_path,
        cuda=True,
        prc=True,  # Precision and Recall
        verbose=True,
    )
    print(f"\nPrecision: {pr_metrics['precision']:.4f}")
    print(f"Recall: {pr_metrics['recall']:.4f}")
else:
    print(f"\nSkipping Precision/Recall: {ref_images_256_path} not found.")
    print("Run 'python resize_imagenet_val.py' first to generate 256x256 val images.")