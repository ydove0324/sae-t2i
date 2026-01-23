export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_100,mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107
export NCCL_IB_GID_INDEX=7
export TORCH_HOME="/share/project/huangxu/.cache/torch"
unset PET_NNODES
unset PET_NODE_RANK
unset PET_MASTER_ADDR

# VAE_CKPT="/share/project/huangxu/models/SAE/models/ema_vae.pth"
VAE_CKPT="results_vae/cnn_decoder_finetune_vf_loss0p01_lora_rank256_eval_mode_ganloss0p01_kl1e-8/step_60000.pth"

torchrun --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr="localhost" \
  --master_port=12732 \
  eval_vae/psnr_test.py \
  --data-path /share/project/datasets/ImageNet/val \
  --vae-ckpt $VAE_CKPT \
  --batch-size 32 \
  --diffusion-steps 4 \
  --max-images 50000 \
  --output-dir eval_vae/vae_lora_rank256_vfloss_0p01_ganloss0p01_kl1e-8_60000step