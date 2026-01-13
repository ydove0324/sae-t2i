import os
import glob
import numpy as np
import torch
from pytorch_fid.fid_score import calculate_activation_statistics
from pytorch_fid.inception import InceptionV3

# === 配置路径 ===
# 你的 ImageNet 验证集根目录
DATA_ROOT = "/share/project/datasets/imagenet/val/"
# 输出文件名
OUTPUT_FILE = "./imagenet_val_stats.npz"
# 设备
DEVICE = "cuda:0"
BATCH_SIZE = 50
DIMS = 2048
NUM_WORKERS = 8

def main():
    print(f"Searching for images in {DATA_ROOT} recursively...")
    
    # 1. 递归查找所有 .JPEG 图片 (ImageNet 默认格式)
    # 如果你的后缀是 .png 或 .jpg，请修改下面
    files = glob.glob(os.path.join(DATA_ROOT, '**', '*.JPEG'), recursive=True)
    
    # 如果没找到，尝试一下 .jpg 和 .png
    if len(files) == 0:
        files = glob.glob(os.path.join(DATA_ROOT, '**', '*.jpg'), recursive=True)
    if len(files) == 0:
        files = glob.glob(os.path.join(DATA_ROOT, '**', '*.png'), recursive=True)

    print(f"Found {len(files)} images.")
    if len(files) == 0:
        raise ValueError(f"No images found in {DATA_ROOT}. Check path structure.")

    # 2. 加载 InceptionV3 模型
    print("Loading InceptionV3 model...")
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[DIMS]
    model = InceptionV3([block_idx]).to(DEVICE)

    # 3. 计算统计量 (mu, sigma)
    print("Calculating statistics (this may take a while)...")
    mu, sigma = calculate_activation_statistics(
        files, 
        model, 
        batch_size=BATCH_SIZE, 
        dims=DIMS, 
        device=DEVICE,
        num_workers=NUM_WORKERS
    )

    # 4. 保存为 .npz
    print(f"Saving statistics to {OUTPUT_FILE}...")
    np.savez_compressed(OUTPUT_FILE, mu=mu, sigma=sigma)
    print("Done!")

if __name__ == "__main__":
    main()