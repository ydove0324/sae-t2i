# Report: qwen3_vit_base_sae_hf0_vit_decoder

## 实验标识
- 配置文件: `deco-sae/qwen3_vit_base_sae_hf0_vit_decoder.yaml`
- 实验名称: `qwen3_vit_4096dim_vit_decoder_hf_dim0`
- 报告日期: 2026-03-02
- 主要基线报告: `exp_report/report_qwen3_vit_base_sae_vit_decoder.md`

## 实验目标（希望得到的结论）
- 在保持主干结构一致的前提下，将 HF 分支关闭（`hf_dim=0`），验证训练流程可正常运行。
- 对比 baseline（`hf_dim=256`）时，重点观察稳定性与重建指标变化趋势。

## Config 快照（关键字段）
```yaml
encoder:
  type: "qwen3_vit"
model:
  decoder_type: "vit_decoder"
  hidden_size: 4096
  hidden_size_x: 4096
  enable_hf_branch: false
  hf_dim: 0
  hf_encoder_config_path: null
  hf_dropout_prob: 0.0
  hf_noise_std: 0.0
  hf_loss_weight: 0.0
loss:
  recon_l1_weight: 1.0
  recon_lpips_weight: 1.0
  recon_gan_weight: 0.5
training:
  max_steps: 120000
  precision: "bf16"
logging:
  output_dir: "results_sae/qwen3_vit_4096dim_vit_decoder_hf_dim0"
```

## 日志与产物路径
- 正式训练日志（目标路径）: `results_sae/qwen3_vit_4096dim_vit_decoder_hf_dim0/log.txt`
- 正式 nohup 日志建议: `log/qwen_vit/qwen3_vit_sae_hf_dim0.log`
- 现有 0dim 最小验证日志（历史）: `results_sae/qwen3_vit_4096dim_vit_decoder_hf_dim0_smoke/log.txt`

## 启动命令（ssh + tmux + nohup）
- ssh 命令：
  - `ssh -p 8029 root@139.224.222.61`
- tmux 会话创建命令：
  - `tmux has-session -t qwen3_vit_sae 2>/dev/null || tmux new-session -d -s qwen3_vit_sae`
- nohup 启动命令：
  - `source /cpfs01/huangxu/miniconda3/bin/activate sae && cd /cpfs01/huangxu/SAE && mkdir -p log/qwen_vit && nohup bash deco-sae/run_vit_zero_dim.sh > log/qwen_vit/qwen3_vit_sae_hf_dim0.log 2>&1 &`
- tmux 下发执行（单条远程命令）：
  - `ssh -p 8029 root@139.224.222.61 "tmux has-session -t qwen3_vit_sae 2>/dev/null || tmux new-session -d -s qwen3_vit_sae; tmux send-keys -t qwen3_vit_sae 'source /cpfs01/huangxu/miniconda3/bin/activate sae && cd /cpfs01/huangxu/SAE && mkdir -p log/qwen_vit && nohup bash deco-sae/run_vit_zero_dim.sh > log/qwen_vit/qwen3_vit_sae_hf_dim0.log 2>&1 &' C-m"`

## 与基线报告的主要改动
- 基于 `report_qwen3_vit_base_sae_vit_decoder.md`。
- 主要改动：
  - `enable_hf_branch: true -> false`
  - `hf_dim: 256 -> 0`
  - `hf_encoder_config_path: "deco-sae/hf_encoder_config_256.json" -> null`
  - HF 相关正则与损失权重置零（`hf_dropout_prob/hf_noise_std/hf_loss_weight`）。
  - 输出目录切到正式命名 `...hf_dim0`（不再使用 smoke 命名作为正式实验）。

## 当前结果（已获取）
- 目前已完成 0dim 最小可运行验证（历史 smoke 日志）：
  - 日志显示成功跑到 `Step 5`，未出现 `hf_dim=0` 初始化或分支相关报错。
  - 该结果用于证明“0dim 可跑通”。
- 正式目录 `results_sae/...hf_dim0/log.txt` 作为后续长期训练记录入口；当前报告待正式训练启动后持续补充指标。

## 阶段性结论
- `hf_dim=0` 配置与代码路径已经具备运行条件。
- 下一阶段需要在正式目录日志上积累足够步数，再与 baseline 做公平对比（如固定 step 的 `val_psnr/val_loss`）。

## 更新记录
- 2026-03-02:
  - 新建 hf0 正式报告。
  - 记录与 baseline 的关键差异。
  - 写入正式日志目标路径与当前可用的 0dim 验证日志来源。
  - 补充远程启动信息：`ssh -p 8029`、tmux 会话 `qwen3_vit_sae`、nohup 启动命令。
  - 已确认远端存在会话 `qwen3_vit_sae`，并生成日志文件 `log/qwen_vit/qwen3_vit_sae_hf_dim0.log`。

