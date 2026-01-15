#!/usr/bin/env bash

# 分布式训练节点配置
WORKER_NODES=(
job-427e70d4-bb80-4be6-87bc-101b3e139f7a-worker-8 
job-427e70d4-bb80-4be6-87bc-101b3e139f7a-worker-9 
job-427e70d4-bb80-4be6-87bc-101b3e139f7a-worker-10 
)

# 节点配置
NUM_NODES=4  # 1个Master + 7个Worker
MASTER_GPUS=8
WORKER_GPUS_LIST=(8 8 8)  # 与 WORKER_NODES 数量一致
MASTER_ADDR="job-427e70d4-bb80-4be6-87bc-101b3e139f7a-master-0"
MASTER_PORT="27519"
JOB_ID="100"

# 训练脚本路径
TRAIN_SCRIPT="train_vae/train_vae.sh"
LOG_FILE="log/dinov3_vae_cnn_decoder_v1.log"
TMUX_SESSION="dinov3_vae_cnn_decoder"

# Master节点启动
echo "Starting training on Master node ($MASTER_GPUS GPUs)"
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NUM_NODES=$NUM_NODES
export NODE_RANK=0
export NUM_GPUS=$MASTER_GPUS
export JOB_ID=$JOB_ID

nohup bash $TRAIN_SCRIPT > $LOG_FILE 2>&1 < /dev/null &
# Worker节点启动
for i in "${!WORKER_NODES[@]}"; do
    NODE="${WORKER_NODES[$i]}"
    WORKER_RANK=$((i + 1))  # Worker节点的rank从1开始
    WORKER_GPUS=${WORKER_GPUS_LIST[$i]}
    
    echo "Local WORKER_RANK=$WORKER_RANK"
    echo "Starting training on $NODE ($WORKER_GPUS GPUs, node_rank=$WORKER_RANK)"
        ssh $NODE 'bash -c "
        source /share/project/huangxu/miniconda3/bin/activate && \
        conda activate video-decode && \
        cd /share/project/huangxu/SAE && \
        # Kill existing tmux session if it exists
        tmux kill-session -t '$TMUX_SESSION' 2>/dev/null || true && \
        # First create tmux session with environment variables
        tmux new-session -A -d -s '$TMUX_SESSION' \
            -e NODE_RANK='"$WORKER_RANK"' \
            -e MASTER_ADDR='"$MASTER_ADDR"' \
            -e MASTER_PORT='"$MASTER_PORT"' \
            -e NUM_NODES='"$NUM_NODES"' \
            -e NUM_GPUS='"$WORKER_GPUS"' \
            -e JOB_ID='"$JOB_ID"' \
        # Then send commands to the session
        tmux send-keys -t '$TMUX_SESSION' \"source /share/project/huangxu/miniconda3/bin/activate && conda activate video-decode && cd /share/project/huangxu/SAE && bash '"$TRAIN_SCRIPT"'\" C-m 
    " ' &
done

# 等待所有节点完成
echo "Waiting for all nodes to complete..."
wait

echo "-------------------- Finished executing distributed training --------------------"


