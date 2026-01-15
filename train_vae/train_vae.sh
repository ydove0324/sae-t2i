#!/bin/bash

# ================= DDP 环境配置 =================
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}
NUM_GPUS=${NUM_GPUS:-8}
JOB_ID=${JOB_ID:-"1"}

export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_100,mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107
export NCCL_IB_GID_INDEX=7
export TORCH_HOME="/share/project/huangxu/.cache/torch"

# export NCCL_DEBUG=INFO

unset PET_NNODES
unset PET_NODE_RANK
unset PET_MASTER_ADDR
export OMP_NUM_THREADS=4

# ================= 启动训练 =================
torchrun --nnodes=$NUM_NODES --node_rank=$NODE_RANK \
    --nproc-per-node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_vae/training_vae.py \
    --train-path "/share/project/datasets/ImageNet/train" \
    --val-path "/share/project/datasets/ImageNet/val" \
    --vae-ckpt "/share/project/huangxu/models/SAE/models/ema_vae.pth" \
    --output-dir "results_vae/cnn_decoder_finetune_v2" \
    --batch-size 16 \
    --image-size 256 \
    --lr 1e-5 \
    --l1-weight 1.0 \
    --lpips-weight 1 \
    --max-steps 100000 \
    --eval-every 2000 \
    --save-every 5000 \
    --log-every 100
    # --debug \
    # --full-validation