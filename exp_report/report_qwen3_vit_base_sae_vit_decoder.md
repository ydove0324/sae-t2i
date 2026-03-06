# Report: qwen3_vit_base_sae_vit_decoder

## 实验标识
- 配置文件: `deco-sae/qwen3_vit_base_sae_vit_decoder.yaml`
- 实验名称: `qwen3_vit_4096dim_vit_decoder_hf_dim256_dropout0p4_GAN0p5`
- 报告日期: 2026-03-02
- 主要基线报告: 首次建立（无前置同名 report）

## 实验目标（希望得到的结论）
- 验证 `qwen3_vit + vit_decoder + hf_dim=256` 在当前训练设置下可以稳定训练。
- 观察重建损失随训练步数的下降趋势，并在阶段性评估点记录 `val_loss/val_psnr`。
- 需要思考的点，前面的 semantic encoder 像 qwen3-vit 那样过 merger 是否是最优的方案，有没有可能前面直接下采样到 256 x 256, 然后过 dino/siglip 更优呢？ 对比先 512 x 512 得到 token 后再 merge

## Config 快照（关键字段）
```yaml
encoder:
  type: "qwen3_vit"
model:
  decoder_type: "vit_decoder"
  hidden_size: 4096
  hidden_size_x: 4096
  enable_hf_branch: true
  hf_dim: 256
  hf_encoder_config_path: "deco-sae/hf_encoder_config_256.json"
  hf_dropout_prob: 0.4
  hf_noise_std: 0.8
  hf_loss_weight: 0.1
loss:
  recon_l1_weight: 1.0
  recon_lpips_weight: 1.0
  recon_gan_weight: 0.5
training:
  max_steps: 120000
  precision: "bf16"
logging:
  output_dir: "results_sae/qwen3_vit_4096dim_vit_decoder_hf_dim256_dropout0p4_GAN0p5"
```

## 日志与产物路径
- 主训练日志: `results_sae/qwen3_vit_4096dim_vit_decoder_hf_dim256_dropout0p4_GAN0p5/log.txt`
- nohup 日志: `log/qwen_vit/qwen3_vit_sae_hf_256dim.log`
- 可视化目录: `results_sae/qwen3_vit_4096dim_vit_decoder_hf_dim256_dropout0p4_GAN0p5/visualizations`
- checkpoint 示例: `results_sae/qwen3_vit_4096dim_vit_decoder_hf_dim256_dropout0p4_GAN0p5/step_10000.pth`

## 当前结果（基于最新日志摘录）
- 训练持续推进到 `Step 10140`（日志时间约 `2026-03-02 17:24:32`）。
- 在 `Step 10000` 处记录：
  - `total/recon = 4.7688`
  - `Eval: val_loss=0.060863, val_psnr=18.2490`
  - 已保存 `step_10000` 可视化与 checkpoint。

## 阶段性结论
- 该配置在当前环境下训练稳定，loss 在约 `4.7~5.0` 区间波动并伴随可用的验证指标输出。
- 可作为后续 `hf_dim=0` 版本的直接对照基线。

## 更新记录
- 2026-03-02:
  - 新建基线报告。
  - 从 `results_sae/.../log.txt` 与 `log/qwen_vit/...256dim.log` 抽取最新状态并记录。

