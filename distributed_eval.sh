#!/usr/bin/env bash

# 分布式训练节点配置
WORKER_NODES=(
job-427e70d4-bb80-4be6-87bc-101b3e139f7a-worker-0 
job-427e70d4-bb80-4be6-87bc-101b3e139f7a-worker-1 
job-427e70d4-bb80-4be6-87bc-101b3e139f7a-worker-2 
job-427e70d4-bb80-4be6-87bc-101b3e139f7a-worker-3 
job-427e70d4-bb80-4be6-87bc-101b3e139f7a-worker-4 
job-427e70d4-bb80-4be6-87bc-101b3e139f7a-worker-5 
job-427e70d4-bb80-4be6-87bc-101b3e139f7a-worker-6 
job-427e70d4-bb80-4be6-87bc-101b3e139f7a-worker-7 
job-427e70d4-bb80-4be6-87bc-101b3e139f7a-worker-8 
job-427e70d4-bb80-4be6-87bc-101b3e139f7a-worker-9 
job-427e70d4-bb80-4be6-87bc-101b3e139f7a-worker-10 
)

# 节点配置
NUM_NODES=12  # 1个Master + 11个Worker
MASTER_GPUS=8
WORKER_GPUS_LIST=(8 8 8 8 8 8 8 8 8 8 8)  # 与 WORKER_NODES 数量一致
MASTER_ADDR="job-427e70d4-bb80-4be6-87bc-101b3e139f7a-master-0"
MASTER_PORT="27529"
JOB_ID="100"

# 训练脚本路径
TRAIN_SCRIPT="./eval_fid.sh"
# 修改：使用日志目录
LOG_DIR="log/dinov3_eval_logs"
TMUX_SESSION="dinov3_vae_eval"

# 创建日志目录
mkdir -p "$LOG_DIR"
echo "Logs will be saved to: $LOG_DIR"

# Master节点启动
echo "Starting training on Master node ($MASTER_GPUS GPUs)"
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NUM_NODES=$NUM_NODES
export NODE_RANK=0
export NUM_GPUS=$MASTER_GPUS
export JOB_ID=$JOB_ID

# 修改：Master日志输出到 master.log
nohup bash $TRAIN_SCRIPT > "${LOG_DIR}/master.log" 2>&1 < /dev/null &

# Worker节点启动
for i in "${!WORKER_NODES[@]}"; do
    NODE="${WORKER_NODES[$i]}"
    WORKER_RANK=$((i + 1))  # Worker节点的rank从1开始
    WORKER_GPUS=${WORKER_GPUS_LIST[$i]}
    
    # 定义当前Worker的日志文件路径
    WORKER_LOG="${LOG_DIR}/worker_${WORKER_RANK}.log"

    echo "Local WORKER_RANK=$WORKER_RANK"
    echo "Starting training on $NODE ($WORKER_GPUS GPUs, node_rank=$WORKER_RANK)"
    echo "Worker log: $WORKER_LOG"
    
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
        # 修改：将执行命令的输出重定向到对应的 worker log 文件
        tmux send-keys -t '$TMUX_SESSION' \"source /share/project/huangxu/miniconda3/bin/activate && conda activate video-decode && cd /share/project/huangxu/SAE && bash '"$TRAIN_SCRIPT"' > '"$WORKER_LOG"' 2>&1\" C-m 
    " ' &
done

# 等待所有节点完成
echo "Waiting for all nodes to complete..."
wait

echo "-------------------- Finished executing distributed training --------------------"