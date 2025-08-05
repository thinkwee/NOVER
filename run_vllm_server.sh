#!/bin/bash

# Set environment variables to avoid tokenizers parallelism warnings
export TOKENIZERS_PARALLELISM=false

# Check if experiment name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <experiment_name>"
    echo "Available experiments:"
    ls config/*.yaml | grep -v config.yaml | sed 's/config\///' | sed 's/\.yaml//'
    exit 1
fi

# Get config value from config file
get_config() {
    local key="$1"
    local default="$2"
    local config_name="${3:-config}"
    
    python3 simple_config_loader.py --config-name "$config_name" --key "$key" --default "$default" 2>/dev/null
}

# Get config name from command line
CONFIG_NAME="${1:-config}"

# Get config values
MODEL_NAME=$(get_config "model.name_vllm" "YOUR_MODEL_NAME_VLLM" "$CONFIG_NAME")
PORT=$(get_config "vllm.port" "8087" "$CONFIG_NAME")
GPU_MEM_UTIL=$(get_config "vllm.gpu_memory_utilization" "0.85" "$CONFIG_NAME")
TENSOR_PARALLEL_SIZE=$(get_config "gpu.vllm.tensor_parallel_size" "1" "$CONFIG_NAME")
GPU_IDS=$(get_config "gpu.vllm.gpu_ids" "1" "$CONFIG_NAME")

# Set environment variables
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export VLLM_ATTENTION_BACKEND=triton
export MASTER_PORT=29505 
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1

echo "Starting vLLM server for model: $MODEL_NAME"
echo "Using GPU: $GPU_IDS, Port: $PORT, GPU Memory: $GPU_MEM_UTIL"

# 
NCCL_DEBUG=WARN trl vllm-serve --model "$MODEL_NAME" --port "$PORT" --gpu_memory_utilization "$GPU_MEM_UTIL" --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"