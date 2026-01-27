#!/bin/bash

# ==================================================
# 使用 YAML 配置文件启动 VAE 训练
# ==================================================

# ================= DDP 环境配置 =================
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}
NUM_GPUS=${NUM_GPUS:-8}

export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_100,mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107
export NCCL_IB_GID_INDEX=7
export TORCH_HOME="/cpfs01/huangxu/.cache/torch"

unset PET_NNODES
unset PET_NODE_RANK
unset PET_MASTER_ADDR
export OMP_NUM_THREADS=4

# ================= 配置文件选择 =================
# 可用的预设配置:
#   - train_vae/configs/default_dinov3.yaml    # DINOv3 encoder (默认)
#   - train_vae/configs/siglip2_encoder.yaml   # SigLIP2 encoder
#   - train_vae/configs/dinov3_no_gan.yaml     # DINOv3 不使用 GAN

CONFIG_FILE="${CONFIG_FILE:-train_vae/configs/default_dinov3.yaml}"

# 可选：从检查点恢复
VAE_CKPT="${VAE_CKPT:-}"

# ================= 启动训练 =================
echo "=========================================="
echo "Using config: $CONFIG_FILE"
echo "=========================================="

CMD="torchrun --nnodes=$NUM_NODES --node_rank=$NODE_RANK \
    --nproc-per-node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_vae/training_vae.py \
    --config $CONFIG_FILE"

# 添加可选的检查点参数
if [ -n "$VAE_CKPT" ]; then
    CMD="$CMD --vae-ckpt $VAE_CKPT"
fi

# 允许通过环境变量覆盖配置
# 例如: EXTRA_ARGS="--lr 2e-4 --batch-size 32" bash train_with_config.sh
if [ -n "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

echo "Running: $CMD"
eval $CMD
