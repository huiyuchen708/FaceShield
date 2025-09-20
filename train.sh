#!/bin/bash
TEMP_DIR="./tmp"
mkdir -p $TEMP_DIR
export TMPDIR=$TEMP_DIR
export TEMP=$TEMP_DIR
export TMP=$TEMP_DIR
find $TMPDIR -name "pymp-*" -type d -mtime +1 -exec rm -rf {} \; 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=0,1
NNODES=1
NPROC_PER_NODE=2
MASTER_ADDR=127.0.0.1
MASTER_PORT=29501

# stage-Ⅰ
nohup torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --nnodes=${NNODES} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train_Reconstructor.py \
  "$@" \
  > train_Reconstructor.log 2>&1 &


# # stage-Ⅱ
# nohup torchrun \
#   --nproc_per_node=${NPROC_PER_NODE} \
#   --nnodes=${NNODES} \
#   --master_addr=${MASTER_ADDR} \
#   --master_port=${MASTER_PORT} \
#   train_SeparationExtractor.py \
#   "$@" \
#   > train_SeparationExtractor.log 2>&1 &

PID=$!
echo "The training process has started (DDP), PID: $PID"