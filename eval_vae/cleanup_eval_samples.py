#!/usr/bin/env python3
"""
清理 result_dit/result_kl500_vae/eval_samples 下的 step 目录
- 删除 step_0xxxxx_xxx 格式的目录
- 删除规则：步数 < 500000 且不是除以 100000 被整除的
- 即保留：步数 >= 500000 的，或步数是 100000 倍数的
"""

import os
import re
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List
import time

# 配置
TARGET_DIR = "result_dit/result_kl500_vae/eval_samples"
NUM_WORKERS = 256  # 多线程数量
MIN_STEP = 500000  # 最小保留步数
STEP_INTERVAL = 100000  # 步数间隔（100000 的倍数）


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
    """从目录名中提取步数，例如 step_0010000_cfg -> 10000"""
    match = re.search(r'step_0*(\d+)_', dirname)
    if match:
        return int(match.group(1))
    return -1


def should_keep_step_dir(dirname: str) -> bool:
    """判断是否应该保留 step 目录
    保留条件：
    - 步数 >= 500000
    - 或步数是 100000 的倍数
    """
    step = parse_step_from_dirname(dirname)
    if step < 0:
        return True  # 无法解析的目录名，保留
    if step >= MIN_STEP:
        return True  # 步数 >= 500000，保留
    if step % STEP_INTERVAL == 0:
        return True  # 是 100000 的倍数，保留
    return False  # 其他情况删除


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


def main():
    target_path = Path(TARGET_DIR)
    
    if not target_path.exists():
        print(f"Error: {TARGET_DIR} does not exist!")
        return
    
    print("=" * 70)
    print(" 清理 eval_samples 下的 step 目录")
    print("=" * 70)
    print(f" 目标目录: {target_path.absolute()}")
    print(f" 删除规则:")
    print(f"   - 删除 step_0xxxxx_xxx 格式的目录")
    print(f"   - 删除条件：步数 < {MIN_STEP} 且不是 {STEP_INTERVAL} 的倍数")
    print(f"   - 保留：步数 >= {MIN_STEP} 的，或步数是 {STEP_INTERVAL} 倍数的")
    print(f" 多线程数: {NUM_WORKERS}")
    print("=" * 70)
    
    # 查找所有 step_* 目录
    step_dirs = []
    for item in target_path.iterdir():
        if item.is_dir() and item.name.startswith("step_"):
            name = item.name
            if not should_keep_step_dir(name):
                step_dirs.append(item)
    
    if not step_dirs:
        print("\n没有需要删除的目录！")
        return
    
    print(f"\n找到 {len(step_dirs)} 个需要删除的目录")
    print("开始处理...\n")
    
    start_time = time.time()
    total_deleted_count = 0
    total_deleted_size = 0
    all_deleted_paths = []
    
    # 使用线程池删除目录
    with ThreadPoolExecutor(max_workers=min(NUM_WORKERS, len(step_dirs))) as executor:
        futures = [executor.submit(delete_directory, dir_path) for dir_path in step_dirs]
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            success, size, path_or_error = future.result()
            if success:
                total_deleted_count += 1
                total_deleted_size += size
                all_deleted_paths.append(path_or_error)
                if completed % 10 == 0:
                    print(f"  进度: {completed}/{len(step_dirs)}")
            else:
                print(f"  Warning: Failed to delete {path_or_error}")
    
    elapsed_time = time.time() - start_time
    
    # 报告结果
    print("\n" + "=" * 70)
    print(" 清理完成!")
    print("=" * 70)
    print(f" 总删除目录数: {total_deleted_count}")
    print(f" 总删除大小: {total_deleted_size / (1024 ** 3):.2f} GB")
    print(f" 总删除大小: {total_deleted_size / (1024 ** 2):.2f} MB")
    print(f" 耗时: {elapsed_time:.2f} 秒")
    print("=" * 70)
    
    # 可选：保存删除的目录列表
    if all_deleted_paths:
        log_file = target_path.parent / "cleanup_eval_samples_log.txt"
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
