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
export TORCH_HOME="/cpfs01/huangxu/.cache/torch"
# export NCCL_DEBUG=INFO
# torchrun
unset PET_NNODES
unset PET_NODE_RANK
unset PET_MASTER_ADDR
export OMP_NUM_THREADS=4

# export MASTER_ADDR="localhost"
# export MASTER_PORT=0
# NNODES=$ARNOLD_WORKER_NUM 
# NNODES=1
# NODE_RANK=0
# NPROC_PER_NODE=8  # 默认 8 个 GPU/节点

# echo "[INFO] Launching torchrun with:"
# echo "  NNODES=$NNODES"
# echo "  NODE_RANK=$NODE_RANK"
# echo "  MASTER_ADDR=$MASTER_ADDR"
# echo "  MASTER_PORT=$MASTER_PORT"
VAE_CKPT="/cpfs01/huangxu/models/SAE/ema_vae.pth"

torchrun --nnodes=$NUM_NODES --node_rank=$NODE_RANK \
  --nproc-per-node=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  projects/rae/train.py \
  --config projects/rae/configs/stage2/training/ImageNet256/DiTDH-XL_DINOv3_1536.yaml \
  --data-path /cpfs01/huangxu/ILSVRC/Data/CLS-LOC/train/ \
  --results-dir ./result_dit/result_kl100_vae \
  --global-batch-size 256 \
  --vae-ckpt $VAE_CKPT \
  --decoder-type cnn_decoder \
  --encoder-type dinov3 \
  --dinov3-dir /cpfs01/huangxu/models/dinov3 \
  --global-seed 1031 \
  --cfg-prob 0.1 \
  --precision bf16 \
  --noise-schedule log_norm \
  --log-norm-mean 0.5 \
  --log-norm-std 1.0  \
  --fid-every 25000 \
  --fid-ref-path VIRTUAL_imagenet256_labeled.npz \
  --fid-batch-size 64 \
  --no-lora \
  --skip-to-moments \
  --ckpt "result_dit/result_kl100_vae/checkpoints/latest.pt" \
  # --ckpt result_dit/result_kl100_vae/checkpoints/latest.pt
#   --ckpt result_dit/result_vfloss_vae/checkpoints/latest.pt
  # --ckpt ./result_cfg_v8/checkpoints/latest.pt