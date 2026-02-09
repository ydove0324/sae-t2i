#!/bin/bash

# Compute latent statistics for VAE encoder
# Usage: bash compute_latent_stats.sh

# ===== Configuration =====
DATA_PATH="/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/train"
# VAE_CKPT="results_vae/siglip2_default/step_60000.pth"
# VAE_CKPT="results_vae/cnn_decoder_finetune_vf_loss0p1_lora_rank256_ganloss0p01_frozen_dinov3_gramloss/step_40000.pth"
# VAE_CKPT="/cpfs01/huangxu/models/SAE/vfloss_vae.pth"
VAE_CKPT="results_vae/dinov2_base_vit_decoder_GAN0p75_VF0p75/step_90000.pth"

# Encoder type: dinov3, dinov3_vitl, siglip2, or dinov2
ENCODER_TYPE="dinov2"
DINOV3_DIR="/cpfs01/huangxu/models/dinov3"
SIGLIP2_MODEL="/cpfs01/huangxu/models/siglip2"
DINOV2_MODEL="/cpfs01/huangxu/models/dinov2-register-base"

# Decoder type: cnn_decoder or vit_decoder
DECODER_TYPE="vit_decoder"

# ViT decoder config (only used when DECODER_TYPE="vit_decoder")
# ViT-XL: hidden=1024, layers=24, heads=16, intermediate=4096
# ViT-L:  hidden=768,  layers=16, heads=12, intermediate=3072
# ViT-B:  hidden=512,  layers=8,  heads=8,  intermediate=2048
VIT_HIDDEN_SIZE=1024
VIT_NUM_LAYERS=24
VIT_NUM_HEADS=16
VIT_INTERMEDIATE_SIZE=4096

OUTPUT_DIR="results/latent_stats_dinov2"
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
echo "Decoder: ${DECODER_TYPE}"
echo "Samples: ${NUM_SAMPLES}"
echo "GPUs: ${NUM_GPUS}"

# Build encoder-specific arguments
ENCODER_ARGS=""
if [ "${ENCODER_TYPE}" = "dinov3" ] || [ "${ENCODER_TYPE}" = "dinov3_vitl" ]; then
    ENCODER_ARGS="--dinov3-dir ${DINOV3_DIR}"
elif [ "${ENCODER_TYPE}" = "siglip2" ]; then
    ENCODER_ARGS="--siglip2-model-name ${SIGLIP2_MODEL}"
elif [ "${ENCODER_TYPE}" = "dinov2" ]; then
    ENCODER_ARGS="--dinov2-model-name ${DINOV2_MODEL}"
fi

# Build decoder-specific arguments
DECODER_ARGS=""
if [ "${DECODER_TYPE}" = "vit_decoder" ]; then
    DECODER_ARGS="--vit-hidden-size ${VIT_HIDDEN_SIZE} --vit-num-layers ${VIT_NUM_LAYERS} --vit-num-heads ${VIT_NUM_HEADS} --vit-intermediate-size ${VIT_INTERMEDIATE_SIZE}"
fi

torchrun --nproc_per_node=${NUM_GPUS} --master_addr="localhost" --master_port=28373 eval_vae/compute_latent_stats.py \
    --data-path ${DATA_PATH} \
    --vae-ckpt ${VAE_CKPT} \
    --encoder-type ${ENCODER_TYPE} \
    --decoder-type ${DECODER_TYPE} \
    ${ENCODER_ARGS} \
    --image-size ${IMAGE_SIZE} \
    --num-samples ${NUM_SAMPLES} \
    --batch-size ${BATCH_SIZE} \
    --lora-rank ${LORA_RANK} \
    --lora-alpha ${LORA_ALPHA} \
    --output-dir ${OUTPUT_DIR} \
    --dinov2-model-name ${DINOV2_MODEL} \
    ${DECODER_ARGS} \
    ${USE_LORA}

echo "Done! Results saved to ${OUTPUT_DIR}"
