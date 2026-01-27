#!/bin/bash

# Compute latent statistics for VAE encoder
# Usage: bash compute_latent_stats.sh

# ===== Configuration =====
DATA_PATH="/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/train"
# VAE_CKPT="results_vae/siglip2_default/step_60000.pth"
# VAE_CKPT="results_vae/cnn_decoder_finetune_vf_loss0p1_lora_rank256_ganloss0p01_frozen_dinov3_gramloss/step_40000.pth"
VAE_CKPT="/cpfs01/huangxu/models/SAE/vfloss_vae.pth"

ENCODER_TYPE="dinov3"  # dinov3 or siglip2
DINOV3_DIR="/cpfs01/huangxu/models/dinov3"
SIGLIP2_MODEL="/cpfs01/huangxu/models/siglip2"

OUTPUT_DIR="results/latent_stats_dinov3"
IMAGE_SIZE=256
NUM_SAMPLES=50000  # 50k samples
BATCH_SIZE=64

# LoRA config (should match VAE training)
LORA_RANK=256
LORA_ALPHA=256

# Use LoRA or not (comment out --use-lora for frozen encoder)
USE_LORA="--use-lora"
# USE_LORA="--no-lora"

# Number of GPUs
NUM_GPUS=8

# ===== Run =====
echo "Computing Latent Statistics..."
echo "Data: ${DATA_PATH}"
echo "VAE: ${VAE_CKPT}"
echo "Encoder: ${ENCODER_TYPE}"
echo "Samples: ${NUM_SAMPLES}"
echo "GPUs: ${NUM_GPUS}"

torchrun --nproc_per_node=${NUM_GPUS} --master_addr="localhost" --master_port=28373 eval_vae/compute_latent_stats.py \
    --data-path ${DATA_PATH} \
    --vae-ckpt ${VAE_CKPT} \
    --encoder-type ${ENCODER_TYPE} \
    --dinov3-dir ${DINOV3_DIR} \
    --image-size ${IMAGE_SIZE} \
    --num-samples ${NUM_SAMPLES} \
    --batch-size ${BATCH_SIZE} \
    --lora-rank ${LORA_RANK} \
    --lora-alpha ${LORA_ALPHA} \
    --output-dir ${OUTPUT_DIR} \
    ${USE_LORA}

echo "Done! Results saved to ${OUTPUT_DIR}"
