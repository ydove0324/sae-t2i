#!/usr/bin/env bash

# ==================================================
# DECO-SAE DiT 多机多卡并行训练脚本
# ==================================================
# 使用方法：
#   1. 配置 WORKER_NODES 和对应的 SSH_PORTS
#   2. 配置节点信息（GPU数量、MASTER地址等）
#   3. 运行: bash deco-sae/run_dit_multi_node.sh

# =================== 节点配置 ===================
# Worker节点列表（按顺序，rank从1开始）
WORKER_NODES=(
    "root@139.224.222.61"  # 示例：机器1
)

# 对应的SSH端口（与WORKER_NODES顺序一致）
SSH_PORTS=(
    8029  # 机器1的SSH端口
)

# 验证配置
if [ ${#WORKER_NODES[@]} -ne ${#SSH_PORTS[@]} ]; then
    echo "Error: WORKER_NODES and SSH_PORTS must have the same length!"
    exit 1
fi

# =================== 训练配置 ===================
NUM_NODES=$((1 + ${#WORKER_NODES[@]}))  # Master + Workers
MASTER_GPUS=8
WORKER_GPUS_LIST=(8)  # 与 WORKER_NODES 数量一致，每个Worker的GPU数

# Master节点配置（当前机器）
# 使用 bond0 高速网络 (200Gbps RoCE RDMA)
MASTER_ADDR="200.45.1.2"  # bond0 高速网络 IP
MASTER_PORT="27532"

# =================== 脚本和模型配置 ===================
TRAIN_SCRIPT="deco-sae/train_dit.py"
DIT_CONFIG="deco-sae/dit_xl_deco_dinov2_per_channel_normalize.yaml"
SAE_CONFIG="deco-sae/dinov2_base_sae_vit_decoder.yaml"
SAE_CKPT="results_sae/dinov2_base_vit_decoder_hf_dim256_dropout0p4_GAN0p5/step_70000.pth"
DATA_PATH="/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/train/"
RESULTS_DIR="results_dit/deco_dinov2_base_dit_xl_per_channel_normalize_256dim"
PRECISION="bf16"
CFG_PROB="0.1"
CFG_SCALE="3.0"

# 可选：断点续训
RESUME_CKPT="results_dit/deco_dinov2_base_dit_xl_per_channel_normalize_256dim/checkpoints/0225000.pt"  # 留空则从头训练，如: "results_dit/deco_dinov2_base_dit_xl/checkpoints/latest.pt"
# RESUME_CKPT=""

# FID 评估配置
FID_REF_PATH="/cpfs01/huangxu/SAE/VIRTUAL_imagenet256_labeled.npz"
FID_SAMPLES_PER_CLASS="50"
FID_BATCH_SIZE="64"
REF_IMAGES_PATH="not-exist"
# REF_IMAGES_PATH="/cpfs01/huangxu/ILSVRC/Data/CLS-LOC/val_256/"
# REF_IMAGES_PATH=""

# =================== 日志配置 ===================
LOG_DIR="log"
LOG_FILE="${LOG_DIR}/deco_dinov2_dit_xl_multi_node_per_channel_normalize.log"
TMUX_SESSION="deco_dit"

# Conda环境配置
CONDA_ENV="sae"
CONDA_PATH="/cpfs01/huangxu/miniconda3/bin/activate"
WORKSPACE_PATH="/cpfs01/huangxu/SAE"

# =================== 环境变量 ===================
# RoCE 网卡配置 (8x 200Gbps Mellanox CX-7)
export NCCL_IB_DISABLE=0  # 启用 RDMA (RoCE)
export NCCL_IB_GID_INDEX=3  # RoCEv2 + IPv4
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7
export NCCL_SOCKET_IFNAME=bond0  # OOB 控制通道走 bond0
export GLOO_SOCKET_IFNAME=bond0
export TORCH_HOME="/cpfs01/huangxu/.cache/torch"

unset PET_NNODES
unset PET_NODE_RANK
unset PET_MASTER_ADDR
export OMP_NUM_THREADS=4

# =================== 创建日志目录 ===================
mkdir -p $LOG_DIR

# =================== 构建训练参数 ===================
TRAIN_ARGS="--config $DIT_CONFIG \
    --sae-config $SAE_CONFIG \
    --sae-ckpt $SAE_CKPT \
    --data-path $DATA_PATH \
    --results-dir $RESULTS_DIR \
    --precision $PRECISION \
    --cfg-prob $CFG_PROB \
    --cfg-scale $CFG_SCALE \
    --ckpt $RESUME_CKPT \
    --fid-ref-path $FID_REF_PATH \
    --fid-samples-per-class $FID_SAMPLES_PER_CLASS \
    --fid-batch-size $FID_BATCH_SIZE \
    --ref-images-path $REF_IMAGES_PATH"

# =================== Master节点启动 ===================
echo "=========================================="
echo "Starting DECO-SAE DiT Multi-Node Training"
echo "=========================================="
echo "Master node: $(hostname)"
echo "Total nodes: $NUM_NODES (1 Master + ${#WORKER_NODES[@]} Workers)"
echo "Master GPUs: $MASTER_GPUS"
echo "Worker GPUs: ${WORKER_GPUS_LIST[@]}"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "DIT_CONFIG: $DIT_CONFIG"
echo "SAE_CONFIG: $SAE_CONFIG"
echo "SAE_CKPT: $SAE_CKPT"
echo "=========================================="

# 设置Master节点环境变量
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# Master节点启动训练
echo ""
echo "Starting training on Master node ($MASTER_GPUS GPUs, rank=0)"

MASTER_CMD="torchrun --nnodes=$NUM_NODES --node_rank=0 \
    --nproc-per-node=$MASTER_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $TRAIN_SCRIPT \
    $TRAIN_ARGS"

echo "Master command: $MASTER_CMD"
nohup $MASTER_CMD >> $LOG_FILE 2>&1 < /dev/null &
MASTER_PID=$!
echo "Master process PID: $MASTER_PID"
echo "Master log: $LOG_FILE"

# =================== Worker节点启动 ===================
echo ""
echo "Starting Workers..."
for i in "${!WORKER_NODES[@]}"; do
    NODE="${WORKER_NODES[$i]}"
    SSH_PORT="${SSH_PORTS[$i]}"
    WORKER_RANK=$((i + 1))  # Worker节点的rank从1开始
    WORKER_GPUS=${WORKER_GPUS_LIST[$i]}
    
    echo ""
    echo "----------------------------------------"
    echo "Worker $WORKER_RANK: $NODE (port: $SSH_PORT)"
    echo "  GPUs: $WORKER_GPUS"
    echo "  Rank: $WORKER_RANK"
    echo "----------------------------------------"
    
    # Worker节点的启动命令（包含环境变量，确保 tmux session 中也能生效）
    WORKER_CMD="source $CONDA_PATH && conda activate $CONDA_ENV && cd $WORKSPACE_PATH && \
export NCCL_IB_DISABLE=0 && \
export NCCL_IB_GID_INDEX=3 && \
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7 && \
export NCCL_SOCKET_IFNAME=bond0 && \
export GLOO_SOCKET_IFNAME=bond0 && \
export TORCH_HOME=/cpfs01/huangxu/.cache/torch && \
export OMP_NUM_THREADS=4 && \
torchrun --nnodes=$NUM_NODES --node_rank=$WORKER_RANK \
        --nproc-per-node=$WORKER_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        $TRAIN_SCRIPT \
        $TRAIN_ARGS"

    # 使用指定的SSH端口连接并启动训练
    ssh -p $SSH_PORT $NODE "bash -c \"
        # Kill existing tmux session if it exists
        tmux kill-session -t '$TMUX_SESSION' 2>/dev/null || true && \
        # Create tmux session and send training command (env vars are inside WORKER_CMD)
        tmux new-session -A -d -s '$TMUX_SESSION' && \
        tmux send-keys -t '$TMUX_SESSION' '$WORKER_CMD' C-m
    \"" &
    
    WORKER_PID=$!
    echo "  Worker $WORKER_RANK SSH PID: $WORKER_PID"
    echo "  Worker $WORKER_RANK tmux session: $TMUX_SESSION"
done

# =================== 等待所有节点 ===================
echo ""
echo "=========================================="
echo "All nodes started. Waiting for completion..."
echo "=========================================="
echo "Master PID: $MASTER_PID"
echo "Monitor logs:"
echo "  Master: tail -f $LOG_FILE"
echo "  Workers: ssh -p <port> <node> 'tmux attach -t $TMUX_SESSION'"
echo ""
echo "To kill all processes:"
echo "  Master: kill $MASTER_PID"
echo "  Workers: ssh -p <port> <node> 'tmux kill-session -t $TMUX_SESSION'"
echo ""

# 等待所有后台任务完成
wait

echo ""
echo "=========================================="
echo "Distributed training completed!"
echo "=========================================="
