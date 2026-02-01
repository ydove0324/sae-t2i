#!/usr/bin/env python3
"""
清理 results_vae 下的临时评估目录
- 删除所有 temp_eval_gt_* 目录
- 对于 temp_eval_recon_* 目录，只保留步数为 10000 倍数的，删除其他
"""

import os
import re
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List
import time

# 配置
RESULTS_VAE_DIR = "results_vae"
NUM_WORKERS = 256  # 多线程数量


def get_dir_size(path: Path) -> int:
    """计算目录大小（字节）"""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except (OSError, FileNotFoundError):
                    pass
    except (OSError, FileNotFoundError):
        pass
    return total


def parse_step_from_dirname(dirname: str) -> int:
    """从目录名中提取步数，例如 temp_eval_recon_5000 -> 5000"""
    match = re.search(r'_(\d+)$', dirname)
    if match:
        return int(match.group(1))
    return -1


def should_keep_recon_dir(dirname: str) -> bool:
    """判断是否应该保留 recon 目录（步数是 10000 的倍数）"""
    step = parse_step_from_dirname(dirname)
    if step < 0:
        return False
    return step % 10000 == 0


def delete_directory(dir_path: Path) -> Tuple[bool, int, str]:
    """
    删除目录并返回 (成功, 大小, 路径)
    """
    try:
        size = get_dir_size(dir_path)
        shutil.rmtree(dir_path)
        return True, size, str(dir_path)
    except Exception as e:
        return False, 0, f"{dir_path}: {e}"


def process_experiment_dir(exp_dir: Path) -> Tuple[int, int, List[str]]:
    """
    处理一个实验目录，返回 (删除的目录数, 删除的总大小, 删除的目录列表)
    """
    deleted_count = 0
    deleted_size = 0
    deleted_paths = []
    
    if not exp_dir.is_dir():
        return 0, 0, []
    
    # 查找所有 temp_eval_* 目录
    temp_dirs = []
    for item in exp_dir.iterdir():
        if item.is_dir():
            name = item.name
            if name.startswith("temp_eval_gt_"):
                # 所有 gt 目录都删除
                temp_dirs.append((item, True))  # (path, should_delete)
            elif name.startswith("temp_eval_recon_"):
                # 只保留步数是 10000 倍数的
                if not should_keep_recon_dir(name):
                    temp_dirs.append((item, True))  # 需要删除
                # else: 保留，不添加到删除列表
    if len(temp_dirs) == 0:
        return 0, 0, []
    # 使用线程池删除目录
    with ThreadPoolExecutor(max_workers=min(NUM_WORKERS, len(temp_dirs))) as executor:
        futures = [executor.submit(delete_directory, dir_path) for dir_path, should_del in temp_dirs if should_del]
        
        for future in as_completed(futures):
            success, size, path_or_error = future.result()
            if success:
                deleted_count += 1
                deleted_size += size
                deleted_paths.append(path_or_error)
            else:
                print(f"  Warning: Failed to delete {path_or_error}")
    
    return deleted_count, deleted_size, deleted_paths


def main():
    results_vae_path = Path(RESULTS_VAE_DIR)
    
    if not results_vae_path.exists():
        print(f"Error: {RESULTS_VAE_DIR} does not exist!")
        return
    
    print("=" * 70)
    print(" 清理 results_vae 下的临时评估目录")
    print("=" * 70)
    print(f" 目标目录: {results_vae_path.absolute()}")
    print(f" 删除规则:")
    print(f"   - 所有 temp_eval_gt_* 目录")
    print(f"   - temp_eval_recon_* 目录（只保留步数为 10000 倍数的）")
    print(f" 多线程数: {NUM_WORKERS}")
    print("=" * 70)
    
    # 获取所有实验目录
    experiment_dirs = [d for d in results_vae_path.iterdir() if d.is_dir()]
    
    if not experiment_dirs:
        print("No experiment directories found!")
        return
    
    print(f"\n找到 {len(experiment_dirs)} 个实验目录")
    print("开始处理...\n")
    
    start_time = time.time()
    total_deleted_count = 0
    total_deleted_size = 0
    all_deleted_paths = []
    
    # 处理每个实验目录
    for i, exp_dir in enumerate(experiment_dirs, 1):
        print(f"[{i}/{len(experiment_dirs)}] 处理: {exp_dir.name}")
        deleted_count, deleted_size, deleted_paths = process_experiment_dir(exp_dir)
        
        if deleted_count > 0:
            size_gb = deleted_size / (1024 ** 3)
            print(f"  ✓ 删除了 {deleted_count} 个目录 ({size_gb:.2f} GB)")
            total_deleted_count += deleted_count
            total_deleted_size += deleted_size
            all_deleted_paths.extend(deleted_paths)
        else:
            print(f"  - 无需删除")
    
    elapsed_time = time.time() - start_time
    
    # 报告结果
    print("\n" + "=" * 70)
    print(" 清理完成!")
    print("=" * 70)
    print(f" 总删除目录数: {total_deleted_count}")
    print(f" 总删除大小: {total_deleted_size / (1024 ** 3):.2f} GB")
    print(f" 总删除大小: {total_deleted_size / (1024 ** 2):.2f} MB")
    print(f" 总删除大小: {total_deleted_size / 1024:.2f} KB")
    print(f" 耗时: {elapsed_time:.2f} 秒")
    print("=" * 70)
    
    # 可选：保存删除的目录列表
    if all_deleted_paths:
        log_file = results_vae_path / "cleanup_log.txt"
        with open(log_file, "w") as f:
            f.write(f"清理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"删除目录数: {total_deleted_count}\n")
            f.write(f"删除大小: {total_deleted_size / (1024 ** 3):.2f} GB\n\n")
            f.write("删除的目录列表:\n")
            for path in sorted(all_deleted_paths):
                f.write(f"{path}\n")
        print(f"\n删除日志已保存到: {log_file}")


if __name__ == "__main__":
    main()
