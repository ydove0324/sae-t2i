#!/bin/bash

# Linear Probing evaluation script for VAE encoder
# Usage: bash linear_probe.sh

# ===== Configuration =====
TRAIN_PATH="/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/train"
VAL_PATH="/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/val"
# VAE_CKPT="results_vae/cnn_decoder_finetune_vf_loss0p1_lora_rank256_ganloss0p01_frozen_dinov3_gramloss/step_40000.pth"  # Change to your VAE checkpoint
# VAE_CKPT="results_vae/siglip2_default/step_140000.pth"  # Change to your VAE checkpoint
# VAE_CKPT="/cpfs01/huangxu/models/SAE/models/ema_vae.pth"
VAE_CKPT="results_vae/dinov2_base_vit_decoder_GAN0p75/step_60000.pth"

ENCODER_TYPE="dinov2"  # dinov3, dinov3_vitl, siglip2, or dinov2
DINOV3_DIR="/cpfs01/huangxu/models/dinov3"
SIGLIP2_MODEL_NAME="/cpfs01/huangxu/models/siglip2"
DINOV2_MODEL_NAME="/cpfs01/huangxu/models/dinov2-register-base"

OUTPUT_DIR="results/linear_probe_dinov2_base_vit_decoder_GAN0p75_60000"
IMAGE_SIZE=256
BATCH_SIZE=256
EPOCHS=100
LR=0.1
POOLING="avg"  # avg, max, avg_max, flatten

# LoRA config (should match VAE training)
LORA_RANK=256
LORA_ALPHA=256

# Number of GPUs
NUM_GPUS=8

# ===== Run =====
echo "Starting Linear Probing evaluation..."
echo "Train: ${TRAIN_PATH}"
echo "Val: ${VAL_PATH}"
echo "VAE: ${VAE_CKPT}"
echo "Encoder: ${ENCODER_TYPE}"
echo "GPUs: ${NUM_GPUS}"

torchrun --nproc_per_node=${NUM_GPUS} --master_addr="localhost" --master_port=28372 eval_vae/linear_probe.py \
    --train-path ${TRAIN_PATH} \
    --val-path ${VAL_PATH} \
    --vae-ckpt ${VAE_CKPT} \
    --encoder-type ${ENCODER_TYPE} \
    --dinov3-dir ${DINOV3_DIR} \
    --siglip2-model-name ${SIGLIP2_MODEL_NAME} \
    --dinov2-model-name ${DINOV2_MODEL_NAME} \
    --image-size ${IMAGE_SIZE} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --lora-rank ${LORA_RANK} \
    --lora-alpha ${LORA_ALPHA} \
    --output-dir ${OUTPUT_DIR} \
    --warmup-epochs 5 \
    --save-every 10

echo "Done!"
