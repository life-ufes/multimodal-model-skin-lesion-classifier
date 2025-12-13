#!/bin/bash

set -e

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

LOG_DIR=logs
mkdir -p $LOG_DIR

nohup python3 -u ./src/scripts/benchmark/nas/optimization_train_process_pad_20_llm-as-controller.py \
  > $LOG_DIR/nas_stdout.log \
  2> $LOG_DIR/nas_stderr.log &
