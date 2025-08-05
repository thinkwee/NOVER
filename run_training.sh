#!/bin/bash

# Set environment variables to reduce verbose output
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=ERROR
export NCCL_P2P_DISABLE=1
export VLLM_LOG_LEVEL=WARNING
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_VERBOSITY=error
export PYTHONWARNINGS=ignore
export CUDA_LAUNCH_BLOCKING=0
export DEEPSPEED_LOG_LEVEL=ERROR

# Check if experiment name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <experiment_name>"
    echo "Available experiments:"
    ls config/*.yaml | grep -v config.yaml | sed 's/config\///' | sed 's/\.yaml//'
    exit 1
fi

# Get config values
get_config() {
    local key="$1"
    local default="$2"
    local config_name="${3:-config}"
    
    python3 simple_config_loader.py --config-name "$config_name" --key "$key" --default "$default" 2>/dev/null
}

# Get config name from command line
CONFIG_NAME="${1:-config}"

# Get config values
TRAINING_GPU_IDS=$(get_config "gpu.training.gpu_ids" "0" "$CONFIG_NAME")
TRAINING_GPUS=$(get_config "gpu.training.num_gpus" "1" "$CONFIG_NAME")
MAIN_PROCESS_PORT=$(get_config "gpu.training.main_process_port" "28890" "$CONFIG_NAME")
USED_VLLM_PORT=$(get_config "vllm.port" "8087" "$CONFIG_NAME")
CONFIG_VLLM_HOST=$(get_config "vllm.host" "localhost" "$CONFIG_NAME")
INTERMEDIATE_TAG=$(get_config "dataset.intermediate_tag" "think" "$CONFIG_NAME")
FINAL_TAG=$(get_config "dataset.final_tag" "answer" "$CONFIG_NAME")

export CUDA_VISIBLE_DEVICES="$TRAINING_GPU_IDS"
echo "Using GPUs $TRAINING_GPU_IDS for training (num_gpus: $TRAINING_GPUS)"
echo "Training will connect to vLLM server at ${CONFIG_VLLM_HOST}:${USED_VLLM_PORT}"
echo "Using intermediate tag: <${INTERMEDIATE_TAG}>"
echo "Using final tag: <${FINAL_TAG}>"

# Start training
accelerate launch \
    --num_processes=$TRAINING_GPUS \
    --main_process_port $MAIN_PROCESS_PORT \
    main.py --config-name "$CONFIG_NAME" 