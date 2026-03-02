export TORCH_HOME="/cpfs01/huangxu/.cache/torch"
cd /cpfs01/huangxu/SAE/

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29611}"
CONFIG_FILE="${CONFIG_FILE:-deco-sae/qwen3_vit_base_sae_hf0_vit_decoder.yaml}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" deco-sae/train_sae.py --config "${CONFIG_FILE}" "$@"
