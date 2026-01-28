#!/bin/bash

# Linear Probing evaluation script for VAE encoder
# Usage: bash linear_probe.sh

# ===== Configuration =====
TRAIN_PATH="/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/train"
VAL_PATH="/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/val"
# VAE_CKPT="results_vae/cnn_decoder_finetune_vf_loss0p1_lora_rank256_ganloss0p01_frozen_dinov3_gramloss/step_40000.pth"  # Change to your VAE checkpoint
VAE_CKPT="/cpfs01/huangxu/models/SAE/models/ema_vae.pth"  # Change to your VAE checkpoint

ENCODER_TYPE="dinov3"  # dinov3 or siglip2
DINOV3_DIR="/cpfs01/huangxu/models/dinov3"

OUTPUT_DIR="results/linear_probe"
IMAGE_SIZE=256
BATCH_SIZE=256
EPOCHS=100
LR=0.1
POOLING="avg"  # avg, max, avg_max, flatten

# LoRA config (should match VAE training)
LORA_RANK=0
LORA_ALPHA=0

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
    --image-size ${IMAGE_SIZE} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --pooling ${POOLING} \
    --lora-rank ${LORA_RANK} \
    --lora-alpha ${LORA_ALPHA} \
    --output-dir ${OUTPUT_DIR} \
    --warmup-epochs 5 \
    --save-every 10

echo "Done!"
