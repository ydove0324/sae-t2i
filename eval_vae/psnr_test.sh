export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_100,mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107
export NCCL_IB_GID_INDEX=7
export TORCH_HOME="/cpfs01/huangxu/.cache/torch"
unset PET_NNODES
unset PET_NODE_RANK
unset PET_MASTER_ADDR

# ==========================================
# Example 1: DINOv3 + ViT-XL Decoder (no LoRA)
# ==========================================
# VAE_CKPT="results_vae/dinov3_vit_decoder_xl/step_25000.pth"
# VAE_CKPT="/cpfs01/huangxu/models/SAE/models/ema_vae.pth"
# VAE_CKPT="results_vae/dinov3_vit_decoder_xl_noise_tau_0p8/step_70000.pth"
VAE_CKPT="results_vae/dinov2_base_vit_decoder_GAN0p75/step_60000.pth"

# torchrun --nproc_per_node=8 \
#   --node_rank=0 \
#   --master_addr="localhost" \
#   --master_port=12738 \
#   eval_vae/psnr_test.py \
#   --data-path /cpfs01/huangxu/ILSVRC/Data/CLS-LOC/val \
#   --vae-ckpt $VAE_CKPT \
#   --encoder-type dinov3 \
#   --decoder-type cnn_decoder \
#   --batch-size 64 \
#   --max-images 50000 \
#   --output-dir eval_vae/dinov3_cnn_decoder_kl500 \
#   --denormalize-output \
#   --no-lora \
#   --skip-to-moments

  # --vit-hidden-size 1024 \
  # --vit-num-layers 24 \
  # --vit-num-heads 16 \
  # --vit-intermediate-size 4096 \
# ==========================================
# Example 2: DINOv2 + ViT-XL Decoder (with LoRA)
# ==========================================
# VAE_CKPT="results_vae/dinov2_base_vit_decoder/step_50000.pth"
# 
torchrun --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr="localhost" \
  --master_port=12732 \
  eval_vae/psnr_test.py \
  --data-path /cpfs01/huangxu/ILSVRC/Data/CLS-LOC/val \
  --vae-ckpt $VAE_CKPT \
  --encoder-type dinov2 \
  --dinov2-model-name /cpfs01/huangxu/models/dinov2-register-base \
  --decoder-type vit_decoder \
  --vit-hidden-size 1024 \
  --vit-num-layers 24 \
  --vit-num-heads 16 \
  --vit-intermediate-size 4096 \
  --image-size 224 \
  --patch-size 14 \
  --batch-size 64 \
  --max-images 50000 \
  --output-dir eval_vae/dinov2_vit_decoder_xl \
  --denormalize-output \
  --lora-rank 256 \
  --lora-alpha 256

# ==========================================
# Example 3: DINOv3 + CNN Decoder (original)
# ==========================================
# VAE_CKPT="results_vae/cnn_decoder_finetune/step_60000.pth"
# 
# torchrun --nproc_per_node=8 \
#   --node_rank=0 \
#   --master_addr="localhost" \
#   --master_port=12732 \
#   eval_vae/psnr_test.py \
#   --data-path /cpfs01/huangxu/ILSVRC/Data/CLS-LOC/val \
#   --vae-ckpt $VAE_CKPT \
#   --encoder-type dinov3 \
#   --decoder-type cnn_decoder \
#   --batch-size 64 \
#   --max-images 50000 \
#   --output-dir eval_vae/dinov3_cnn_decoder \
#   --lora-rank 256 \
#   --lora-alpha 256