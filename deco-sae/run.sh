export TORCH_HOME="/cpfs01/huangxu/.cache/torch"
torchrun --nproc_per_node=8 deco-sae/train_sae.py --config deco-sae/dinov2_base_sae_flow_matching.yaml