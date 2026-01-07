# export https_proxy=http://bj-rd-proxy.byted.org:3128
# export http_proxy=http://bj-rd-proxy.byted.org:3128
# export WANDB_BASE_URL=https://api.bandw.top
# export WANDB_KEY="3ff2050be63ca796f027523bc233bc3e70c7668b"
# export ENTITY="fobow"
# export PROJECT="sae-imagenet-noise-1536-e2e-104"
wandb offline

MASTER_ADDR=$ARNOLD_WORKER_0_HOST
MASTER_PORT=$ARNOLD_WORKER_0_PORT
# NNODES=$ARNOLD_WORKER_NUM 
NNODES=1
NODE_RANK=$ARNOLD_ID
NPROC_PER_NODE=${NPROC_PER_NODE:-8}  # 默认 8 个 GPU/节点

echo "[INFO] Launching torchrun with:"
echo "  NNODES=$NNODES"
echo "  NODE_RANK=$NODE_RANK"
echo "  MASTER_ADDR=$MASTER_ADDR"
echo "  MASTER_PORT=$MASTER_PORT"

torchrun  --nnodes=$NNODES --node-rank=$NODE_RANK  --nproc-per-node=$NPROC_PER_NODE \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  projects/rae/train.py \
  --config /share/project/huangxu/SAE/projects/rae/configs/stage2/training/ImageNet256/DiTDH-XL_DINOv3_1536.yaml \
  --data-path /opt/tiger/dataset/ILSVRC2012_img_train \
  --results-dir ./trash \
  --precision fp32 \
  --wandb \
  --global-batch-size 64 \
  --ckpt /opt/tiger/vfm/decoder_only/latest.pt