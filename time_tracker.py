import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
import copy
import contextlib
import time
from typing import Dict, Optional, Any
from enum import Enum

# 添加计时器相关代码
class TimerDevice(Enum):
    CPU = "cpu"
    CUDA = "cuda"

class Timer:
    def __init__(self, name: str, device: TimerDevice = TimerDevice.CPU, device_sync: bool = False):
        self.name = name
        self.device = device
        self.device_sync = device_sync
        self.start_time = None
        self.end_time = None
        self.elapsed_time = 0.0
    
    def start(self):
        if self.device == TimerDevice.CUDA and self.device_sync:
            torch.cuda.synchronize()
        self.start_time = time.time()
    
    def end(self):
        if self.device == TimerDevice.CUDA and self.device_sync:
            torch.cuda.synchronize()
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

class TimeTracker:
    def __init__(self):
        self._timed_metrics: Dict[str, float] = {}
        self.trackers = []
    
    @contextlib.contextmanager
    def timed(self, name: str, device: TimerDevice = TimerDevice.CPU, device_sync: bool = False):
        """Context manager to track time for a specific operation."""
        timer = Timer(name, device, device_sync)
        timer.start()
        yield timer
        timer.end()
        elapsed_time = timer.elapsed_time
        if name in self._timed_metrics:
            # If the timer name already exists, add the elapsed time to the existing value since a log has not been invoked yet
            self._timed_metrics[name] += elapsed_time
        else:
            self._timed_metrics[name] = elapsed_time
        for tracker in self.trackers:
            tracker._timed_metrics = copy.deepcopy(self._timed_metrics)
    
    def log_times(self, step: int = None):
        """打印所有记录的时间"""
        if not self._timed_metrics:
            return
        
        print(f"\n=== Time Metrics {f'(Step {step})' if step else ''} ===")
        total_time = sum(self._timed_metrics.values())
        
        for name, elapsed_time in self._timed_metrics.items():
            percentage = (elapsed_time / total_time) * 100 if total_time > 0 else 0
            print(f"{name}: {elapsed_time:.4f}s ({percentage:.1f}%)")
        
        print(f"Total: {total_time:.4f}s")
        print("=" * 40)
        
        # 清空记录，准备下一轮
        self._timed_metrics.clear()
    
    def get_times(self) -> Dict[str, float]:
        """获取当前的时间记录"""
        return copy.deepcopy(self._timed_metrics)

# 创建全局计时器实例
time_tracker = TimeTracker()