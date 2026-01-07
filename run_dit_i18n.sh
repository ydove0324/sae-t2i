sudo apt-get install -y unzip tmux python3-venv
cd /opt/tiger/vfm

mkdir -p /opt/tiger/dataset
## us
hdfs dfs -get hdfs://harunava/home/byte_data_seed_azureb/user/yancheng.zhang/imagenet/imagenet.zip /opt/tiger/dataset
unzip -q /opt/tiger/dataset/imagenet.zip -d /opt/tiger/dataset
## cn
# hdfs dfs -get hdfs://haruna/home/byte_arnold_lq/user/yibairen.byron/imagenet/ILSVRC2012_img_train_full.tar /opt/tiger/dataset
# tar xf /opt/tiger/dataset/ILSVRC2012_img_train_full.tar -C /opt/tiger/dataset

mkdir -p /opt/tiger/vfm/weights/dinov3
hdfs dfs -get \
    hdfs://haruna/home/byte_data_seed/hdd_hldy/user/ming.li1/work_dirs/vfm_exp/ming_dino_vae_exp/1026_from_pt_masked_nosied_sae-e2e_0.1_0.8_denormalize/states/0000110000/models \
    /opt/tiger/vfm/weights/dinov3
hdfs dfs -get \
    hdfs://haruna/home/byte_data_seed/hdd_hldy/user/ming.li1/work_dirs/vfm_exp/ming_dino_vae_exp/1026_from_pt_masked_nosied_sae-e2e_0.1_0.8_denormalize/configs \
    /opt/tiger/vfm/weights/dinov3

python3 -m venv rae_test_1
source rae_test_1/bin/activate

pip install uv
uv pip install torch==2.8.0 torchvision torchaudio --index-url https://bytedpypi.byted.org/simple/
uv pip install timm==0.9.16 accelerate==0.23.0 torchdiffeq==0.2.5 wandb diffusers --index-url https://bytedpypi.byted.org/simple/
uv pip install "numpy<2" transformers einops omegaconf --index-url https://bytedpypi.byted.org/simple/

export https_proxy=http://bj-rd-proxy.byted.org:3128
export http_proxy=http://bj-rd-proxy.byted.org:3128
export WANDB_BASE_URL=https://api.bandw.top
export WANDB_KEY="3ff2050be63ca796f027523bc233bc3e70c7668b"
export ENTITY="fobow"
export PROJECT="sae-imagenet-noise-1536-e2e"
export DATE="20251202"

MASTER_ADDR=$ARNOLD_WORKER_0_HOST
MASTER_PORT=$ARNOLD_WORKER_0_PORT
# NNODES=$ARNOLD_WORKER_NUM 
NNODES=1
NODE_RANK=$ARNOLD_ID
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

export WORK_DIR="/mnt/hdfs/__MERLIN_USER_DIR__/$PROJECT/$DATE/results"

echo "[INFO] Launching torchrun with:"
echo "  NNODES=$NNODES"
echo "  NODE_RANK=$NODE_RANK"
echo "  MASTER_ADDR=$MASTER_ADDR"
echo "  MASTER_PORT=$MASTER_PORT"
echo "  WORK_DIR=$WORK_DIR"

# download a checkpoint to resume
mkdir -p $WORK_DIR/sae/stage2/checkpoints
mkdir -p $WORK_DIR/sae/stage2/experiment
hdfs dfs -get hdfs://harunava/home/byte_data_seed_ord/ssd/user/xiaojieli/sae_1536_batch_1024_vae_noise_e2e/results/sae/stage2/checkpoints/latest.pt $WORK_DIR/sae/stage2/checkpoints/

torchrun  --nnodes=$NNODES --node-rank=$NODE_RANK  --nproc-per-node=$NPROC_PER_NODE \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  projects/rae/train.py \
  --config /opt/tiger/vfm/projects/rae/configs/stage2/training/ImageNet256/DiTDH-XL_DINOv3_1536.yaml \
  --data-path /opt/tiger/dataset/ILSVRC/Data/CLS-LOC/train/ \
  --results-dir $WORK_DIR/sae/stage2 \
  --precision fp32 \
  --wandb \
  --ckpt $WORK_DIR/sae/stage2/checkpoints/latest.pt \
  --global-batch-size 32