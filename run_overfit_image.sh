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
VAE_CKPT="/share/project/huangxu/models/SAE/diffusion_decoder/kl100/vae.pth"

torchrun --nnodes=1 --node_rank=0 \
  --nproc-per-node=1 \
  --master_addr="localhost" \
  --master_port=19291 \
  projects/rae/train_overfit.py \
  --config /share/project/huangxu/SAE/projects/rae/configs/stage2/training/ImageNet256/DiTDH-XL_DINOv3_1536.yaml \
  --data-path /share/project/datasets/ImageNet/train \
  --vae-ckpt $VAE_CKPT \
  --results-dir ./result_overfit_kl100_v \
  --precision fp32 \
  --image-size 256 \
  --global-batch-size 1 \
  --prediction-mode v \
  --dataset-type overfit_single_image
  # --ckpt /opt/tiger/vfm/decoder_only/latest.pt