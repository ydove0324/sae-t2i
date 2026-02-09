#!/usr/bin/env bash

# ==================================================
# 多机并行训练脚本 - 支持不同SSH端口
# ==================================================
# 使用方法：
#   1. 配置 WORKER_NODES 和对应的 SSH_PORTS
#   2. 配置节点信息（GPU数量、MASTER地址等）
#   3. 运行: bash distributed_train_vae_multi_node.sh

# =================== 节点配置 ===================
# Worker节点列表（按顺序，rank从1开始）
WORKER_NODES=(
    "root@139.224.222.61"  # 示例：机器1
)

# 对应的SSH端口（与WORKER_NODES顺序一致）
SSH_PORTS=(
    8028  # 机器1的SSH端口
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
MASTER_ADDR="192.168.1.124"  # 或手动指定，如 "master-node-0"
MASTER_PORT="27531"
JOB_ID="100"

# =================== 脚本配置 ===================
TRAIN_SCRIPT="train_vae/train_with_config.sh"
LOG_FILE="log/dinov2_base_vae_GAN0p75_VF0p75_bs1024_layernorm.log"
TMUX_SESSION="dinov2_base_vae_multi_node"

# 训练配置文件
CONFIG_FILE="train_vae/configs/dinov2_base_vae.yaml"

# Conda环境配置
CONDA_ENV="sae"  # 或 "video-decode" 等
CONDA_PATH="/cpfs01/huangxu/miniconda3/bin/activate"
WORKSPACE_PATH="/cpfs01/huangxu/SAE"  # 或 "/share/project/huangxu/SAE"

# =================== Master节点启动 ===================
echo "=========================================="
echo "Starting Multi-Node Distributed Training"
echo "=========================================="
echo "Master node: $(hostname)"
echo "Total nodes: $NUM_NODES (1 Master + ${#WORKER_NODES[@]} Workers)"
echo "Master GPUs: $MASTER_GPUS"
echo "Worker GPUs: ${WORKER_GPUS_LIST[@]}"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "CONFIG_FILE: $CONFIG_FILE"
echo "=========================================="

# 设置Master节点环境变量
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NUM_NODES=$NUM_NODES
export NODE_RANK=0
export NUM_GPUS=$MASTER_GPUS
export JOB_ID=$JOB_ID
export CONFIG_FILE=$CONFIG_FILE

# Master节点启动训练
echo ""
echo "Starting training on Master node ($MASTER_GPUS GPUs, rank=0)"
nohup bash $TRAIN_SCRIPT > $LOG_FILE 2>&1 < /dev/null &
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
    
    # 使用指定的SSH端口连接并启动训练
    ssh -p $SSH_PORT $NODE "bash -c \"
        source $CONDA_PATH && \
        conda activate $CONDA_ENV && \
        cd $WORKSPACE_PATH && \
        # Kill existing tmux session if it exists
        tmux kill-session -t '$TMUX_SESSION' 2>/dev/null || true && \
        # Create tmux session
        tmux new-session -A -d -s '$TMUX_SESSION' && \
        # Send commands to set environment variables and start training
        tmux send-keys -t '$TMUX_SESSION' 'export NODE_RANK=$WORKER_RANK && export MASTER_ADDR=$MASTER_ADDR && export MASTER_PORT=$MASTER_PORT && export NUM_NODES=$NUM_NODES && export NUM_GPUS=$WORKER_GPUS && export JOB_ID=$JOB_ID && export CONFIG_FILE=$CONFIG_FILE && source $CONDA_PATH && conda activate $CONDA_ENV && cd $WORKSPACE_PATH && bash $TRAIN_SCRIPT' C-m
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

# 等待所有后台任务完成
wait

echo ""
echo "=========================================="
echo "Distributed training completed!"
echo "=========================================="
