#!/bin/bash
# ============================================================
# Latent Feature Visualization Script
# Generates RGB visualization of VAE latent features (like paper figures)
# ============================================================

# Change to project root
cd /cpfs01/huangxu/SAE

# ==================== Configuration ====================

# Data paths
DATA_PATH="/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/train"              # ImageNet validation
IMAGE_SIZE=256
NUM_IMAGES=10                                          # Number of images to visualize

# VAE checkpoint
VAE_CKPT="results_vae/siglip2_default/step_10000.pth"
# VAE_CKPT="results_vae/cnn_decoder_finetune_vf_loss0p1_lora_rank256_ganloss0p01_frozen_dinov3_gramloss/step_40000.pth"

# Encoder type: "dinov3" or "siglip2"
ENCODER_TYPE="siglip2"

# LoRA parameters
LORA_RANK=256
LORA_ALPHA=256

# Model paths (adjust based on encoder type)
DINOV3_DIR="/cpfs01/huangxu/models/dinov3"
SIGLIP2_MODEL="/cpfs01/huangxu/models/siglip2"

# Output directory
OUTPUT_DIR="results/latent_feature_vis_siglip2"

# Visualization options
USE_LORA="--use-lora"                                  # Use "--no-lora" to disable
GLOBAL_NORM=""                                         # Use "--global-norm" for consistent colors
UPSCALE=16

# Random seed
SEED=42

# ==================== Run Script ====================

echo "========================================"
echo " Latent Feature Visualization"
echo "========================================"
echo " Encoder:     $ENCODER_TYPE"
echo " VAE Ckpt:    $VAE_CKPT"
echo " Num Images:  $NUM_IMAGES"
echo " Output:      $OUTPUT_DIR"
echo "========================================"

python eval_vae/latent_feature_vis.py \
    --data-path "$DATA_PATH" \
    --image-size $IMAGE_SIZE \
    --num-images $NUM_IMAGES \
    --vae-ckpt "$VAE_CKPT" \
    --encoder-type "$ENCODER_TYPE" \
    --dinov3-dir "$DINOV3_DIR" \
    --siglip2-model-name "$SIGLIP2_MODEL" \
    --lora-rank $LORA_RANK \
    --lora-alpha $LORA_ALPHA \
    --output-dir "$OUTPUT_DIR" \
    --upscale $UPSCALE \
    --seed $SEED \
    $USE_LORA $GLOBAL_NORM

echo "========================================"
echo " Done! Results saved to: $OUTPUT_DIR"
echo "========================================"
