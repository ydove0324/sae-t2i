#!/bin/bash
#
# Text-to-Image training script
# Usage: bash train.sh
#

# Configuration
DATA_PATH="/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/train"
VAE_CKPT="/cpfs01/huangxu/models/SAE/models/ema_vae.pth"
TEXT_ENCODER_PATH="/cpfs01/huangxu/models/qwen3-1.7b"  # Or local path
RESULTS_DIR="./results_t2i"

# Optional: for custom text-image dataset
# DATA_PATH="/path/to/custom_dataset"
# DATASET_TYPE="custom"

# Training settings
GLOBAL_BATCH_SIZE=256
GRAD_ACCUM=4
NUM_GPUS=8

# Calculate per-GPU batch size
BATCH_PER_GPU=$((GLOBAL_BATCH_SIZE / NUM_GPUS / GRAD_ACCUM))

echo "=== Text-to-Image Training ==="
echo "Data path: ${DATA_PATH}"
echo "VAE checkpoint: ${VAE_CKPT}"
echo "Text encoder: ${TEXT_ENCODER_PATH}"
echo "Global batch size: ${GLOBAL_BATCH_SIZE}"
echo "Per-GPU batch size: ${BATCH_PER_GPU}"
echo "Gradient accumulation: ${GRAD_ACCUM}"
echo "=============================="

# Run training
torchrun --nproc_per_node=${NUM_GPUS} \
    --master_port=29501 \
    sae-t2i/train_t2i.py \
    --data-path ${DATA_PATH} \
    --dataset-type imagenet \
    --vae-ckpt ${VAE_CKPT} \
    --text-encoder-path ${TEXT_ENCODER_PATH} \
    --results-dir ${RESULTS_DIR} \
    --model-size XXL \
    --image-size 256 \
    --txt-max-length 128 \
    --num-text-refine-blocks 4 \
    --encoder-type dinov3 \
    --decoder-type cnn_decoder \
    --dinov3-dir "/cpfs01/huangxu/models/dinov3" \
    --no-lora \
    --vae-diffusion-steps 50 \
    --epochs 100 \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --grad-accum-steps ${GRAD_ACCUM} \
    --lr 1e-4 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --ema-decay 0.9999 \
    --precision bf16 \
    --num-workers 4 \
    --time-shift-base 4096 \
    --noise-schedule uniform \
    --cfg-prob 0.1 \
    --cfg-scale 7.5 \
    --log-every 100 \
    --vis-every 500 \
    --ckpt-every 10000 \
    --skip-to-moments
