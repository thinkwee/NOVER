#!/bin/bash

# Multi-node Multi-GPU Training Script for NOVER
# This script supports both single-node and multi-node distributed training

set -e

# Set environment variables to reduce verbose output
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export VLLM_LOG_LEVEL=WARNING
export TRANSFORMERS_VERBOSITY=warning
export HF_HUB_VERBOSITY=error
export DEEPSPEED_LOG_LEVEL=ERROR
export PYTHONWARNINGS=ignore
export CUDA_LAUNCH_BLOCKING=0

# Check if experiment name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <experiment_name> [launcher_type]"
    echo ""
    echo "Arguments:"
    echo "  experiment_name: Name of the configuration file (without .yaml extension)"
    echo "  launcher_type:   'accelerate' (default) or 'torchrun'"
    echo ""
    echo "Available experiments:"
    ls config/*.yaml | grep -v config.yaml | sed 's/config\///' | sed 's/\.yaml//'
    echo ""
    echo "Examples:"
    echo "  $0 my_exp                    # Single-node training with accelerate"
    echo "  $0 multi_node_example        # Multi-node training with accelerate"  
    echo "  $0 multi_node_example torchrun  # Multi-node training with torchrun"
    exit 1
fi

# Get config values helper function
get_config() {
    local key="$1"
    local default="$2"
    local config_name="${3:-config}"
    
    python3 src/simple_config_loader.py --config-name "$config_name" --key "$key" --default "$default" 2>/dev/null
}

# Parse arguments
CONFIG_NAME="${1:-config}"
LAUNCHER_TYPE="${2:-accelerate}"

echo "=========================================="
echo "NOVER Multi-Node Training Script"
echo "=========================================="
echo "Configuration: $CONFIG_NAME"
echo "Launcher: $LAUNCHER_TYPE"
echo ""

# Get distributed training configuration
NUM_NODES=$(get_config "distributed.num_nodes" "1" "$CONFIG_NAME")
NODE_RANK=$(get_config "distributed.node_rank" "0" "$CONFIG_NAME")
MASTER_ADDR=$(get_config "distributed.master_addr" "localhost" "$CONFIG_NAME")
MASTER_PORT=$(get_config "distributed.master_port" "29500" "$CONFIG_NAME")
BACKEND=$(get_config "distributed.backend" "nccl" "$CONFIG_NAME")
NCCL_SOCKET_IFNAME=$(get_config "distributed.nccl_socket_ifname" "eth0" "$CONFIG_NAME")

# Get GPU configuration
TRAINING_GPU_IDS=$(get_config "gpu.training.gpu_ids" "0" "$CONFIG_NAME")
TRAINING_GPUS=$(get_config "gpu.training.num_gpus" "1" "$CONFIG_NAME")
MAIN_PROCESS_PORT=$(get_config "gpu.training.main_process_port" "28890" "$CONFIG_NAME")

# Get vLLM configuration
USED_VLLM_PORT=$(get_config "vllm.port" "8087" "$CONFIG_NAME")
CONFIG_VLLM_HOST=$(get_config "vllm.host" "localhost" "$CONFIG_NAME")

# Get dataset tags
INTERMEDIATE_TAG=$(get_config "dataset.intermediate_tag" "think" "$CONFIG_NAME")
FINAL_TAG=$(get_config "dataset.final_tag" "answer" "$CONFIG_NAME")

# Set distributed training environment variables
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"
export NODE_RANK="$NODE_RANK"
export WORLD_SIZE=$((NUM_NODES * TRAINING_GPUS))
export LOCAL_RANK=0  # Will be set by launcher

# Set NCCL environment variables for better multi-node performance
export NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=3600
export NCCL_SOCKET_TIMEOUT=3600
export NCCL_IB_TIMEOUT=3600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set GPU visibility
export CUDA_VISIBLE_DEVICES="$TRAINING_GPU_IDS"

echo "Distributed Training Configuration:"
echo "  Total nodes: $NUM_NODES"
echo "  Current node rank: $NODE_RANK"
echo "  Master address: $MASTER_ADDR:$MASTER_PORT"
echo "  Backend: $BACKEND"
echo "  World size: $WORLD_SIZE"
echo "  GPUs per node: $TRAINING_GPUS"
echo "  GPU IDs on this node: $TRAINING_GPU_IDS"
echo "  Network interface: $NCCL_SOCKET_IFNAME"
echo ""
echo "Training Configuration:"
echo "  vLLM server: ${CONFIG_VLLM_HOST}:${USED_VLLM_PORT}"
echo "  Intermediate tag: <${INTERMEDIATE_TAG}>"
echo "  Final tag: <${FINAL_TAG}>"
echo ""

# Wait for user confirmation in multi-node setup
if [ "$NUM_NODES" -gt 1 ]; then
    echo "=========================================="
    echo "Multi-Node Setup Instructions:"
    echo "=========================================="
    echo "1. Make sure vLLM server is running and accessible from all nodes"
    echo "2. Ensure all nodes can communicate on port $MASTER_PORT"
    echo "3. Run this script on all nodes with the same configuration"
    echo "4. Start nodes in order: node 0 (master) first, then others"
    echo ""
    echo "Current node rank: $NODE_RANK"
    if [ "$NODE_RANK" -eq 0 ]; then
        echo "This is the MASTER node (rank 0)"
    else
        echo "This is WORKER node (rank $NODE_RANK)"
        echo "Make sure the master node (rank 0) is already running!"
    fi
    echo ""
    read -p "Press Enter to continue or Ctrl+C to abort..."
fi

# Choose launcher and start training
if [ "$LAUNCHER_TYPE" = "torchrun" ]; then
    echo "Starting training with torchrun..."
    echo "=========================================="
    
    torchrun \
        --nnodes=$NUM_NODES \
        --node_rank=$NODE_RANK \
        --nproc_per_node=$TRAINING_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        -m src.main --config-name "$CONFIG_NAME"
        
elif [ "$LAUNCHER_TYPE" = "accelerate" ]; then
    echo "Starting training with accelerate..."
    echo "=========================================="
    
    # Calculate total number of processes
    TOTAL_PROCESSES=$((NUM_NODES * TRAINING_GPUS))
    
    accelerate launch \
        --num_processes=$TOTAL_PROCESSES \
        --num_machines=$NUM_NODES \
        --machine_rank=$NODE_RANK \
        --main_process_ip=$MASTER_ADDR \
        --main_process_port=$MAIN_PROCESS_PORT \
        --same_network \
        -m src.main --config-name "$CONFIG_NAME"
else
    echo "Error: Unknown launcher type '$LAUNCHER_TYPE'. Use 'accelerate' or 'torchrun'."
    exit 1
fi

echo ""
echo "Training completed!"