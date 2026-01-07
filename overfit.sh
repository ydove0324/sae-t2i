export WANDB_KEY="3ff2050be63ca796f027523bc233bc3e70c7668b"
export ENTITY="fobow"
export PROJECT="sae-overfit-img"

# overfit rae 256
# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
#   projects/rae/overfit.py \
#   --config /opt/tiger/vfm/projects/rae/configs/stage2/overfit/ImageNet256/DiT-XL_DINOv2-B.yaml \
#   --data-path /opt/tiger/dataset/overfit \
#   --results-dir results/rae/overfit/ \
#   --precision bf16 \
#   --wandb \
#   --ae rae \
#   --sdown 16 \
#   --KL 0



# overfit dit xl with sae, 256
# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
#   projects/rae/overfit.py \
#   --config /opt/tiger/vfm/projects/rae/configs/stage2/overfit/ImageNet256/DiT-XL_DINOv3.yaml \
#   --data-path /opt/tiger/dataset/overfit \
#   --results-dir results/rae/overfit/ \
#   --precision bf16 \
#   --wandb \
#   --ae sae \
#   --sdown 16 \
#   --KL 0 

# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
#   projects/rae/overfit.py \
#   --config /opt/tiger/vfm/projects/rae/configs/stage2/overfit/ImageNet256/channel/DiT-XL_DINOv3_c384.yaml \
#   --data-path /opt/tiger/dataset/overfit \
#   --results-dir results/rae/overfit/ \
#   --precision bf16 \
#   --wandb \
#   --ae sae \
#   --sdown 16 \
#   --KL 0 \
#   --channel 384

# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
#   projects/rae/overfit.py \
#   --config /opt/tiger/vfm/projects/rae/configs/stage2/overfit/ImageNet256/channel/DiT-XL_DINOv3_c768.yaml \
#   --data-path /opt/tiger/dataset/overfit \
#   --results-dir results/rae/overfit/ \
#   --precision bf16 \
#   --wandb \
#   --ae sae \
#   --sdown 16 \
#   --KL 0 \
#   --channel 768

# CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
#   projects/rae/overfit.py \
#   --config /opt/tiger/vfm/projects/rae/configs/stage2/overfit/ImageNet256/channel/DiT-XL_DINOv3_c1152.yaml \
#   --data-path /opt/tiger/dataset/overfit \
#   --results-dir results/rae/overfit/ \
#   --precision bf16 \
#   --wandb \
#   --ae sae \
#   --sdown 16 \
#   --KL 0 \
#   --channel 1152


# overfit vae 256
# CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
#   projects/rae/overfit.py \
#   --config /opt/tiger/vfm/projects/rae/configs/stage2/overfit/ImageNet256/DiT-XL_SDVAE.yaml \
#   --data-path /opt/tiger/dataset/overfit \
#   --results-dir results/rae/overfit/ \
#   --precision bf16 \
#   --wandb \
#   --ae vae \
#   --sdown 8 \
#   --KL 0




####################### spatial ########################
# CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
#   projects/rae/overfit.py \
#   --config /opt/tiger/vfm/projects/rae/configs/stage2/overfit/ImageNet256/DiT-XL_DINOv3_s32.yaml \
#   --data-path /opt/tiger/dataset/overfit \
#   --results-dir results/rae/overfit/ \
#   --precision bf16 \
#   --wandb \
#   --ae sae \
#   --sdown 32 \
#   --KL 0


# CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
#   projects/rae/overfit.py \
#   --config /opt/tiger/vfm/projects/rae/configs/stage2/overfit/ImageNet256/DiT-XL_DINOv3_s64.yaml \
#   --data-path /opt/tiger/dataset/overfit \
#   --results-dir results/rae/overfit/ \
#   --precision bf16 \
#   --wandb \
#   --ae sae \
#   --sdown 64 \
#   --KL 0


# CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
#   projects/rae/overfit.py \
#   --config /opt/tiger/vfm/projects/rae/configs/stage2/overfit/ImageNet256/DiT-XL_DCAE_s32.yaml \
#   --data-path /opt/tiger/dataset/overfit \
#   --results-dir results/rae/overfit/ \
#   --precision bf16 \
#   --wandb \
#   --ae dcae \
#   --sdown 32 

# CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
#   projects/rae/overfit.py \
#   --config /opt/tiger/vfm/projects/rae/configs/stage2/overfit/ImageNet256/DiT-XL_DCAE_s64.yaml \
#   --data-path /opt/tiger/dataset/overfit \
#   --results-dir results/rae/overfit/ \
#   --precision bf16 \
#   --wandb \
#   --ae dcae \
#   --sdown 64





####### KL

# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
#   projects/rae/overfit.py \
#   --config /opt/tiger/vfm/projects/rae/configs/stage2/overfit/ImageNet256/KL/DiT-XL_DINOv3_KL0.yaml \
#   --data-path /opt/tiger/dataset/overfit \
#   --results-dir results/rae/overfit/ \
#   --precision bf16 \
#   --wandb \
#   --ae sae_kl \
#   --sdown 16 \
#   --KL 0

# CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
#   projects/rae/overfit.py \
#   --config /opt/tiger/vfm/projects/rae/configs/stage2/overfit/ImageNet256/KL/DiT-XL_DINOv3_KL500.yaml \
#   --data-path /opt/tiger/dataset/overfit \
#   --results-dir results/rae/overfit/ \
#   --precision bf16 \
#   --wandb \
#   --ae sae_kl \
#   --sdown 16 \
#   --KL 500












####################### overfit 512
# sae 512
# CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
#   projects/rae/overfit.py \
#   --config /opt/tiger/vfm/projects/rae/configs/stage2/overfit/ImageNet512/DiTDH-XL_DINOv3.yaml \
#   --data-path /opt/tiger/dataset/overfit \
#   --results-dir results/rae/overfit/ \
#   --precision bf16 \
#   --wandb \
#   --image-size 512 \
#   --ae sae


# overfit dit xl with rae 512
# CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
#   projects/rae/overfit.py \
#   --config /opt/tiger/vfm/projects/rae/configs/stage2/overfit/ImageNet512/DiTDH-XL_DINOv2-B.yaml \
#   --data-path /opt/tiger/dataset/overfit \
#   --results-dir results/rae/overfit/ \
#   --precision bf16 \
#   --wandb \
#   --image-size 512 \
#   --ae rae







# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
#   projects/rae/overfit.py \
#   --config /opt/tiger/vfm/projects/rae/configs/stage2/overfit/ImageNet256/widthDiT-XL_DINOv3.yaml \
#   --data-path /opt/tiger/dataset/overfit \
#   --results-dir results/rae/overfit/ \
#   --precision bf16 \
#   --wandb \
#   --ae sae_width \
#   --sdown 16 \
#   --KL 0 \
#   --channel 1152


CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  projects/rae/overfit.py \
  --config /opt/tiger/vfm/projects/rae/configs/stage2/overfit/ImageNet256/widthDiTDH-XL_DINOv3.yaml \
  --data-path /opt/tiger/dataset/overfit \
  --results-dir results/rae/overfit/ \
  --precision bf16 \
  --wandb \
  --ae sae_width \
  --sdown 16 \
  --KL 0 \
  --channel 1280

