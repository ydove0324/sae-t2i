#!/usr/bin/env bash

# 分布式训练节点配置
WORKER_NODES=(
job-5ca8fa4e-4e2d-4eba-9720-e850d50e118e-master-0
job-5bb88e20-e119-448d-8ee6-13732210c58f-master-0
job-9e1ddf42-fa45-413d-8feb-adcac7c07c99-master-0
# job-b7373458-0c71-4964-8ac8-f59d684a98e7-worker-3
# job-b7373458-0c71-4964-8ac8-f59d684a98e7-worker-4
# job-b7373458-0c71-4964-8ac8-f59d684a98e7-worker-5
# job-b7373458-0c71-4964-8ac8-f59d684a98e7-worker-6
)

# 节点配置
NUM_NODES=4  # 1个Master + 7个Worker
MASTER_GPUS=8
WORKER_GPUS_LIST=(8 8 8 8 8 8 8 8 8 8 8 8 8 8 8)  # 与 WORKER_NODES 数量一致
MASTER_ADDR="job-4d210284-abc8-485e-bcc3-8b48d3e37ab1-master-0"
MASTER_PORT="27535"
JOB_ID="100"

# 训练脚本路径
TRAIN_SCRIPT="./build_wds.sh"
# 修改：使用日志目录
LOG_DIR="log/dinov3_build_wds"
TMUX_SESSION="dinov3_build_wds"

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