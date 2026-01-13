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
torchrun --nnodes=$NUM_NODES --node_rank=$NODE_RANK \
    --nproc-per-node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    inference_ddp.py \
    --config "/share/project/huangxu/SAE/projects/rae/configs/stage2/training/ImageNet256/DiTDH-XL_DINOv3_1536.yaml" \
    --ckpt "/share/project/huangxu/SAE/result_v5/checkpoints/0070000.pt" \
    --vae-ckpt "/share/project/huangxu/models/SAE/models/ema_vae.pth" \
    --out "result_v5/imagenet_50k_70000_diff_cnn_decoder" \
    --samples-per-class 50 \
    --batch-size 16 \
    --stage2-steps 50 \
    --vae-diffusion-steps 10 \
    --use-ema \
    --calc-fid \
    --ref-path "/share/project/huangxu/SAE/VIRTUAL_imagenet256_labeled.npz" \
    --decoder-type "cnn_decoder"
    # --eval-only