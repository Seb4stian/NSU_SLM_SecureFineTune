#!/bin/bash
# Launch MiniCPM fine-tuning on 4x A100 GPUs
# Usage: bash run_train_minicpm.sh

export HF_HUB_CACHE="/mnt/cache"
export HF_DATASETS_CACHE="/mnt/cache"
export HF_HOME="/mnt/cache"

# Optimize NCCL for A100
export NCCL_P2P_LEVEL=NVL
export TORCH_NCCL_BLOCKING_WAIT=1

cd /home/azureuser/cloudfiles/code/Users/ing.eduardo.castro

echo "Starting MiniCPM-1B fine-tuning on 4x A100 80GB GPUs..."
accelerate launch \
    --num_processes=4 \
    --multi_gpu \
    --mixed_precision=bf16 \
    TrainModel_MiniCPM.py
