export TORCH_HOME="/cpfs01/huangxu/.cache/torch"
cd /cpfs01/huangxu/SAE/
torchrun --nproc_per_node=8 deco-sae/train_sae.py --config deco-sae/qwen3_vit_base_sae_vit_decoder.yaml