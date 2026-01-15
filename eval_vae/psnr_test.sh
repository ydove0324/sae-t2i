export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_100,mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107
export NCCL_IB_GID_INDEX=7
export TORCH_HOME="/share/project/huangxu/.cache/torch"
unset PET_NNODES
unset PET_NODE_RANK
unset PET_MASTER_ADDR

torchrun --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr="localhost" \
  --master_port=12732 \
  eval_vae/psnr_test.py \
  --data-path /share/project/datasets/ImageNet/val \
  --vae-ckpt /share/project/huangxu/SAE/results_vae/cnn_decoder_finetune_v2/step_5000.pth \
  --batch-size 16 \
  --diffusion-steps 4 \
  --max-images 50000 \
  --output-dir eval_vae/vae_eval_test_cnn_5000