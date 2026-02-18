#!/usr/bin/env bash
# ==================================================
# DECO-SAE DiT 单机多卡训练脚本
# ==================================================

export TORCH_HOME="/cpfs01/huangxu/.cache/torch"

# 配置路径
DIT_CONFIG="deco-sae/dit_xl_deco_dinov2.yaml"
SAE_CONFIG="deco-sae/dinov2_base_sae_vit_decoder.yaml"
SAE_CKPT="results_sae/dinov2_base_vit_decoder_hf_dim256_dropout0p4_GAN0p5/step_70000.pth"
DATA_PATH="/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/train/"
RESULTS_DIR="results_dit/deco_dinov2_base_dit_xl"
RESUME_CKPT="results_dit/deco_dinov2_base_dit_xl/checkpoints/latest.pt"

# FID 评估配置
FID_REF_PATH="/cpfs01/huangxu/VIRTUAL_imagenet256_labeled.npz"
FID_SAMPLES_PER_CLASS=50
FID_BATCH_SIZE=64

torchrun --nproc_per_node=8 deco-sae/train_dit.py \
    --config $DIT_CONFIG \
    --sae-config $SAE_CONFIG \
    --sae-ckpt $SAE_CKPT \
    --data-path $DATA_PATH \
    --results-dir $RESULTS_DIR \
    --precision bf16 \
    --cfg-prob 0.1 \
    --cfg-scale 3.0 \
    --ckpt $RESUME_CKPT \
    --fid-ref-path $FID_REF_PATH \
    --fid-samples-per-class $FID_SAMPLES_PER_CLASS \
    --fid-batch-size $FID_BATCH_SIZE
