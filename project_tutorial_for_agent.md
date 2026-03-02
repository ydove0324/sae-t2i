# SAE 项目 Agent 教程

> **重要**: 每次开始新任务时，先阅读并更新此文档。

## 1. 项目概述

这是一个 **Semantic Autoencoder (SAE)** 项目，用于训练图像重建模型。核心架构：
- **Encoder**: 支持多种 Vision Encoder（DINOv3, DINOv2, SigLIP2, Qwen3-ViT）
- **Decoder**: 支持 CNN Decoder 或 ViT Decoder
- **训练框架**: PyTorch DDP 分布式训练

## 2. 关键文件

| 文件 | 说明 |
|------|------|
| `cnn_decoder.py` | 所有 Encoder 类的定义（Encoder2D, DINOv2Encoder2D, SigLIP2Encoder2D, **Qwen3ViTEncoder2D**） |
| `deco-sae/model.py` | DecoSAE 模型主类，包含 forward_loss, encode, decode 等核心逻辑 |
| `deco-sae/train_sae.py` | 训练脚本入口 |
| `deco-sae/*.yaml` | 训练配置文件 |

## 3. Encoder 类型与配置

### 3.1 Qwen3-ViT (4096 dim)

```python
# 输出: [B, 64, 4096] (对于 256x256 图像)
# - 64 tokens = (256/16/2)^2，经过 spatial_merge_size=2
# - 4096 dim = merger 输出维度
# - has_cls_token = False (重要！不含 CLS token)
```

**关键属性**:
- `hidden_size = 4096`
- `patch_size = 16`
- `spatial_merge_size = 2`
- `has_cls_token = False` ← **必须设置，否则会导致 token 数量错误**

### 3.2 DINOv2 (768 dim for base)

```python
# 输出: [B, 256, 768] (对于 256x256 图像，resize 到 224x224)
# - 256 tokens = (224/14)^2 = 16^2
# - 768 dim for base, 1024 for large
# - 有 CLS token 和 4 个 register tokens (需要去掉前 5 个)
```

### 3.3 SigLIP2

```python
# 输出: [B, 256, hidden_size]
# - 无 CLS token，无 register tokens
```

## 4. Debug 模式指南

### 4.1 添加调试代码

```python
# 在可疑位置添加 print
print(f"DEBUG: tokens.shape = {tokens.shape}")
print(f"DEBUG: expected = {expected_value}")

# 或使用 pdb
import pdb; pdb.set_trace()
```

### 4.2 运行单 GPU 测试

```bash
# 单 GPU 快速测试
cd /cpfs01/huangxu/SAE
conda activate sae
export TORCH_HOME="/cpfs01/huangxu/.cache/torch"
torchrun --nproc_per_node=1 deco-sae/train_sae.py --config <config.yaml>
```

### 4.3 Debug 完成后

**重要**: 必须删除所有调试代码（print/pdb）

## 5. 常见 Bug 与解决方案

### Bug 1: Token count is not a perfect square

**错误信息**:
```
AssertionError: Token count 63 is not a perfect square
```

**原因**: `_extract_patch_tokens` 中 `has_cls_token` 默认为 `True`，错误地去掉了第一个 token

**解决**: 在 Encoder 类中设置 `self.has_cls_token = False`

### Bug 2: pixel_values 格式错误 (Qwen3-VL)

**错误**: Qwen3-VL 期望特殊的 `pixel_values` 格式

**解决**: 使用 `_preprocess_images()` 方法将 `[B, C, H, W]` 转换为 `[total_patches, C*T*ps*ps]`

## 6. 配置文件注意事项

### 6.1 Qwen3-ViT 配置

```yaml
encoder:
  type: "qwen3_vit"
  qwen3_vit_model_name: "/cpfs01/huangxu/models/qwen3vl-8b"

model:
  hidden_size: 4096      # merger 输出维度
  hidden_size_x: 4096    # 与 hidden_size 一致
```

### 6.2 DINOv2 配置

```yaml
encoder:
  type: "dinov2"
  dinov2_model_name: "/cpfs01/huangxu/models/dinov2-register-base"

model:
  hidden_size: 768       # base: 768, large: 1024
  hidden_size_x: 768
```

## 7. 显存优化

### Qwen3-VL 显存优化

```python
# 删除 LLM 部分，只保留 visual 模块
# 完整模型: ~16GB
# 仅 visual: ~1.1GB
# 节省: ~15GB

full_model = Qwen3VLForConditionalGeneration.from_pretrained(...)
self.visual = full_model.visual
del full_model.model
del full_model.lm_head
del full_model
torch.cuda.empty_cache()
```

## 8. 单元测试模板

在 `tmp/` 目录中创建测试脚本：

```python
# tmp/test_encoder.py
import torch
from cnn_decoder import Qwen3ViTEncoder2D

encoder = Qwen3ViTEncoder2D(model_name="...").cuda()
dummy = torch.randn(2, 3, 256, 256, device='cuda')

with torch.no_grad():
    out = encoder(dummy)

print(f"Output shape: {out.last_hidden_state.shape}")
assert out.last_hidden_state.shape == (2, 64, 4096)
print("Test passed!")
```

## 9. 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2026-03-02 | 初始创建；添加 Qwen3-ViT 4096 dim encoder；修复 has_cls_token bug |

---

**提醒**: 每次完成任务后，更新此文档的相关章节和更新日志。
