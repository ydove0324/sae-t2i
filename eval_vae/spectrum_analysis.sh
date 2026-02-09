#!/bin/bash

# Spectrum Analysis script for VAE latents
# Usage: bash spectrum_analysis.sh

# ===== Configuration =====
DATA_PATH="/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/train"
# VAE_CKPT="results_vae/dinov2_base_vit_decoder_GAN0p75_VF0p75_tuned_layernorm/step_150000.pth"
VAE_CKPT="results_vae/dinov2_base_vit_decoder_GAN0p75_VF0p75_layernorm/step_95000.pth"
ENCODER_TYPE="dinov2"  # dinov3, dinov3_vitl, siglip2, dinov2
DECODER_TYPE="vit_decoder"  # cnn_decoder or vit_decoder
DINOV3_DIR="/cpfs01/huangxu/models/dinov3"
SIGLIP2_MODEL_NAME="/cpfs01/huangxu/models/siglip2"
DINOV2_MODEL_NAME="/cpfs01/huangxu/models/dinov2-register-base"

OUTPUT_DIR="results/spectrum_analysis_dinov2_base_vit_decoder_GAN0p75_VF0p75_layernorm_95000"
IMAGE_SIZE=256
NUM_SAMPLES=500        # 分析的样本数量
BATCH_SIZE=16

# 频谱分析参数
TRANSFORM="dft"        # dct, dft, or both
METHOD="radial"        # zigzag or radial

# LoRA config (should match VAE training)
LORA_RANK=256
LORA_ALPHA=256

# 高级选项
SKIP_TO_MOMENTS=true  # true or false
USE_EMA=true         # true or false
NORMALIZATION="none"   # none, layernorm, channelwise, global
LATENT_STATS_PATH=""   # 用于 channelwise/global normalization

# 对比选项
COMPARE_LORA=true     # 对比有/无 LoRA
COMPARE_RGB=false     # 对比 RGB 图像频谱

# ===== Build command =====
CMD="python eval_vae/spectrum_analysis.py \
    --data-path ${DATA_PATH} \
    --vae-ckpt ${VAE_CKPT} \
    --encoder-type ${ENCODER_TYPE} \
    --decoder-type ${DECODER_TYPE} \
    --dinov3-dir ${DINOV3_DIR} \
    --siglip2-model-name ${SIGLIP2_MODEL_NAME} \
    --dinov2-model-name ${DINOV2_MODEL_NAME} \
    --image-size ${IMAGE_SIZE} \
    --num-samples ${NUM_SAMPLES} \
    --batch-size ${BATCH_SIZE} \
    --transform ${TRANSFORM} \
    --method ${METHOD} \
    --lora-rank ${LORA_RANK} \
    --lora-alpha ${LORA_ALPHA} \
    --normalization ${NORMALIZATION} \
    --output-dir ${OUTPUT_DIR}"

# 添加可选参数
if [ "${SKIP_TO_MOMENTS}" = "true" ]; then
    CMD="${CMD} --skip-to-moments"
fi

if [ "${USE_EMA}" = "true" ]; then
    CMD="${CMD} --use-ema"
fi

if [ "${COMPARE_LORA}" = "true" ]; then
    CMD="${CMD} --compare-lora"
fi

if [ "${COMPARE_RGB}" = "true" ]; then
    CMD="${CMD} --compare-rgb"
fi

if [ -n "${LATENT_STATS_PATH}" ]; then
    CMD="${CMD} --latent-stats-path ${LATENT_STATS_PATH}"
fi

# ===== Run =====
echo "Starting Spectrum Analysis..."
echo "Data: ${DATA_PATH}"
echo "VAE: ${VAE_CKPT}"
echo "Encoder: ${ENCODER_TYPE}"
echo "Transform: ${TRANSFORM}"
echo "Method: ${METHOD}"
echo ""

eval ${CMD}

echo ""
echo "Done! Results saved to ${OUTPUT_DIR}"
