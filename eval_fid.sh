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
unset PET_NNODES
unset PET_NODE_RANK
unset PET_MASTER_ADDR
export OMP_NUM_THREADS=4
# VAE_CKPT="/cpfs01/huangxu/models/SAE/models/ema_vae.pth"
VAE_CKPT="results_vae/dinov3_vit_decoder_xl/step_70000.pth"
torchrun --nnodes=$NUM_NODES --node_rank=$NODE_RANK \
    --nproc-per-node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    inference_ddp.py \
    --config "projects/rae/configs/stage2/training/ImageNet256/DiTDH-XL_DINOv3_1536.yaml" \
    --ckpt "result_dit/result_kl500_vae/checkpoints/0600000.pt" \
    --vae-ckpt $VAE_CKPT \
    --out "result_dit/kl500_vit_decoder_GAN0p75_noise_tau_0p8_600000.pt" \
    --samples-per-class 50 \
    --batch-size 32 \
    --stage2-steps 50 \
    --ref-path "./VIRTUAL_imagenet256_labeled.npz" \
    --encoder-type "dinov3" \
    --dinov3-dir "/cpfs01/huangxu/models/dinov3" \
    --decoder-type "vit_decoder" \
    --image-size 256 \
    --skip-to-moments \
    --denormalize-decoder-output \
    --no-lora \
    --calc-fid \
    --use-ema \
    # --eval-only \