#!/bin/bash
# vLLM 启动脚本 - 使用本地模型、离线模式、自动选择最空闲 GPU
# 模型: Qwen3-8B (单卡)
# 用法: ./start_vllm.sh

set -e

# 本地模型路径
MODEL_PATH="/lpai/models/Qwen__Qwen3-8B/25-07-26-0349"

# 自动选择显存最空闲的 1 张 GPU（8B 模型单卡即可）
FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | \
  sort -t',' -k2 -rn | head -n 1 | cut -d',' -f1 | tr -d ' ')
if [ -z "$FREE_GPUS" ]; then
  echo "Warning: nvidia-smi failed, using CUDA_VISIBLE_DEVICES=0"
  FREE_GPUS=0
fi
echo "Selected GPU(s): $FREE_GPUS"

# 启动 vLLM（HF_HUB_OFFLINE=1 使用本地模型，不联网）
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$FREE_GPUS vllm serve \
  --model "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 256 \
  --block-size 16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --trust-remote-code \
  | tee vllm_state.log
