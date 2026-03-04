# HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0,1 vllm serve --model Qwen/Qwen2.5-72B-Instruct --host 0.0.0.0 --port 8000 --max-num-seqs 256 --block-size 16 --tensor-parallel-size 2 --pipeline-parallel-size 1 --gpu-memory-utilization 0.95  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks": 25600}}' --enable-prefix-caching --trust-remote-code --disable-hybrid-kv-cache-manager | tee vllm_state.log

# --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks": <num_cpu_blocks>}}'

# /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct

# --kv-offloading-backend native --kv_offloading_size 8 

  # --model /lpai/models/Qwen__Qwen3-32B/25-07-26-0345 \

FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -n -k2 -r | head -n 2 | cut -d ',' -f 1 | xargs | sed 's/ /,/g')
echo "Selected GPUs: $FREE_GPUS"

# 设定 CPU KV Cache 允许使用的最大内存字节数，这里设置为 50GB (50 * 1024^3 = 53687091200)
CPU_BYTES=53687091200

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$FREE_GPUS vllm serve \
  --model /lpai/models/Qwen__Qwen3-8B/25-07-26-0349 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 256 \
  --block-size 16 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\": 25600, \"cpu_bytes_to_use\": $CPU_BYTES}}" \
  --enable-prefix-caching \
  --trust-remote-code \
  --disable-hybrid-kv-cache-manager | tee vllm_state.log