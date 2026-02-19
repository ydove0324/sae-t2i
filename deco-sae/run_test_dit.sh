#!/usr/bin/env bash
# ==================================================
# DECO-SAE DiT 测试脚本 (单机多卡)
# ==================================================

export TORCH_HOME="/cpfs01/huangxu/.cache/torch"

# =================== 模型配置 ===================
DIT_CONFIG="deco-sae/dit_xl_deco_dinov2.yaml"
SAE_CONFIG="deco-sae/dinov2_base_sae_vit_decoder_64dim.yaml"
SAE_CKPT="results_sae/dinov2_base_vit_decoder_hf_dim64_dropout0p4_GAN0p5/step_100000.pth"
DIT_CKPT="results_dit/deco_dinov2_base_dit_xl_64dim/checkpoints/latest.pt"
OUTPUT_DIR="results_dit/deco_dinov2_base_dit_xl_64dim/test_results"

# FID 评估配置
FID_REF_PATH="/cpfs01/huangxu/SAE/VIRTUAL_imagenet256_labeled.npz"
# REF_IMAGES_PATH="/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/val_256/"  # 用于 Precision/Recall

# 采样配置
SAMPLES_PER_CLASS=50
BATCH_SIZE=64
SAMPLE_STEPS=50

# CFG 配置
USE_CFG=""  # 设置为 "--use-cfg" 启用 CFG
CFG_SCALE=1.5

# HF Mask 配置
MASK_HF=""  # 设置为 "--mask-hf" 启用 HF masking
HF_MASK_MODE="zero"  # zero, noise, mean

# =================== 构建参数 ===================
ARGS="--config $DIT_CONFIG \
    --sae-config $SAE_CONFIG \
    --sae-ckpt $SAE_CKPT \
    --dit-ckpt $DIT_CKPT \
    --output-dir $OUTPUT_DIR \
    --fid-ref-path $FID_REF_PATH \
    --samples-per-class $SAMPLES_PER_CLASS \
    --batch-size $BATCH_SIZE \
    --sample-steps $SAMPLE_STEPS \
    --cfg-scale $CFG_SCALE \
    --hf-mask-mode $HF_MASK_MODE \
    $USE_CFG \
    $MASK_HF"

# 如果设置了 REF_IMAGES_PATH，添加到参数中
if [ -n "$REF_IMAGES_PATH" ]; then
    ARGS="$ARGS --ref-images-path $REF_IMAGES_PATH"
fi

# =================== 运行测试 ===================
echo "=========================================="
echo "Starting DECO-SAE DiT Testing"
echo "=========================================="
echo "DIT_CONFIG: $DIT_CONFIG"
echo "SAE_CONFIG: $SAE_CONFIG"
echo "SAE_CKPT: $SAE_CKPT"
echo "DIT_CKPT: $DIT_CKPT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "=========================================="

torchrun --nproc_per_node=8 deco-sae/test_dit.py $ARGS
