#!/usr/bin/env bash
# ==================================================
# DECO-SAE Latent Statistics 计算脚本
# ==================================================

export TORCH_HOME="/cpfs01/huangxu/.cache/torch"

# =================== 配置 ===================
# 256 dim 版本
SAE_CONFIG="deco-sae/dinov2_base_sae_vit_decoder.yaml"
SAE_CKPT="results_sae/dinov2_base_vit_decoder_hf_dim256_dropout0p4_GAN0p5/step_70000.pth"
OUTPUT_DIR="results_sae/dinov2_base_vit_decoder_hf_dim256_dropout0p4_GAN0p5/latent_stats"

# 64 dim 版本 (取消注释使用)
# SAE_CONFIG="deco-sae/dinov2_base_sae_vit_decoder_64dim.yaml"
# SAE_CKPT="results_sae/dinov2_base_vit_decoder_hf_dim64_dropout0p4_GAN0p5/step_100000.pth"
# OUTPUT_DIR="results_sae/dinov2_base_vit_decoder_hf_dim64_dropout0p4_GAN0p5/latent_stats"

DATA_PATH="/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/train/"
NUM_SAMPLES=50000
BATCH_SIZE=64

# =================== 运行 ===================
echo "=========================================="
echo "Computing DECO-SAE Latent Statistics"
echo "=========================================="
echo "Config: $SAE_CONFIG"
echo "Checkpoint: $SAE_CKPT"
echo "Output: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES"
echo "=========================================="

torchrun --nproc_per_node=8 deco-sae/compute_latent_stats.py \
    --config $SAE_CONFIG \
    --ckpt $SAE_CKPT \
    --data-path $DATA_PATH \
    --num-samples $NUM_SAMPLES \
    --batch-size $BATCH_SIZE \
    --output-dir $OUTPUT_DIR
