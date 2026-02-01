#!/bin/bash

# PCA Visualization script for VAE encoder features
# Usage: bash pca_visualization.sh

# ===== Configuration =====
DATA_PATH="/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/train"
# VAE_CKPT="results_vae/siglip2_default/step_140000.pth"
# VAE_CKPT="/cpfs01/huangxu/models/SAE/vfloss_vae.pth"
# VAE_CKPT="/cpfs01/huangxu/models/SAE/ema_vae.pth"
# VAE_CKPT="results_vae/siglip2_default_v3/step_120000.pth"
VAE_CKPT="results_vae/dinov2_base_vit_decoder_GAN0p75/step_60000.pth"
ENCODER_TYPE="dinov2"  # dinov3 or siglip2
DINOV3_DIR="/cpfs01/huangxu/models/dinov3"
SIGLIP2_MODEL_NAME="/cpfs01/huangxu/models/siglip2"
DINOV2_MODEL_NAME="/cpfs01/huangxu/models/dinov2-register-base"

OUTPUT_DIR="results/pca_visualization_dinov2_base_vit_decoder_GAN0p75_60000"
IMAGE_SIZE=256
NUM_CLASSES=20         # 选择多少个类别
SAMPLES_PER_CLASS=200  # 每个类别采样多少
POOLING="avg"          # avg or flatten
METHOD="both"          # pca, tsne, or both

# LoRA config (should match VAE training)
LORA_RANK=256
LORA_ALPHA=256

# ===== Run =====
echo "Starting PCA Visualization..."
echo "Data: ${DATA_PATH}"
echo "VAE: ${VAE_CKPT}"
echo "Encoder: ${ENCODER_TYPE}"

python eval_vae/pca_visualization.py \
    --data-path ${DATA_PATH} \
    --vae-ckpt ${VAE_CKPT} \
    --encoder-type ${ENCODER_TYPE} \
    --dinov3-dir ${DINOV3_DIR} \
    --siglip2-model-name ${SIGLIP2_MODEL_NAME} \
    --dinov2-model-name ${DINOV2_MODEL_NAME} \
    --image-size ${IMAGE_SIZE} \
    --num-classes ${NUM_CLASSES} \
    --samples-per-class ${SAMPLES_PER_CLASS} \
    --pooling ${POOLING} \
    --method ${METHOD} \
    --lora-rank ${LORA_RANK} \
    --lora-alpha ${LORA_ALPHA} \
    --output-dir ${OUTPUT_DIR} \
    --n-components 10 \

echo "Done! Results saved to ${OUTPUT_DIR}"
