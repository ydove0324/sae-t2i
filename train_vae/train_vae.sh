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
# torchrun --nnodes=$NUM_NODES --node_rank=$NODE_RANK \
#     --nproc-per-node=$NUM_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
VAE_CKPT="/share/project/huangxu/models/SAE/noise_tau_0.8_from_scratch/noise_tau_0.8_from_scratch/0000485000/models/ema_vae.pth"    # Decoder Only
# VAE_CKPT="/share/project/huangxu/models/SAE/models/ema_vae.pth"     # KL 500
# VAE_CKPT="results_vae/cnn_decoder_finetune_vf_loss0p01_lora_rank256_eval_mode/step_100000.pth"     
torchrun --nnodes=1 --node_rank=0 \
    --nproc-per-node=8 \
    --master_addr="localhost" \
    --master_port=28372 \
    train_vae/training_vae.py \
    --train-path "/share/project/datasets/ImageNet/train" \
    --val-path "/share/project/datasets/ImageNet/val" \
    --vae-ckpt $VAE_CKPT \
    --output-dir "results_vae/cnn_decoder_finetune_vf_loss0p1_lora_rank256_ganloss0p01_frozen_dinov3_lpips0p1" \
    --batch-size 8 \
    --image-size 256 \
    --lr 1e-4 \
    --l1-weight 1.0 \
    --lpips-weight 0.1 \
    --max-steps 100000 \
    --eval-every 2000 \
    --save-every 5000 \
    --log-every 10 \
    --debug \
    --lora-rank 256 \
    --lora-alpha 256 \
    --val-max-batches 2000 \
    --vf-weight 0.1 \
    --stage-disc-steps 0 \
    --gan-weight 0.01 \
    --gan-start-step 0 \
    --kl-weight 1e-6 \
    --disc-lr 4e-5 \
    # --debug \
    # --full-validation

# nohup bash train_vae/train_vae.sh > log/lora_rank256_vfloss_0p1_ganloss0p01_kl1e-8_frozen_dinov3_lpips0p1.log 2>&1 < /dev/null &