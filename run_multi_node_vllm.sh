#!/bin/bash

# Multi-node vLLM Server Script for NOVER
# This script supports running vLLM server across multiple nodes for distributed inference

set -e

# Set environment variables to avoid warnings
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

# Check if experiment name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <experiment_name> [mode]"
    echo ""
    echo "Arguments:"
    echo "  experiment_name: Name of the configuration file (without .yaml extension)"
    echo "  mode:           'single' (default) or 'distributed'"
    echo ""
    echo "Available experiments:"
    ls config/*.yaml | grep -v config.yaml | sed 's/config\///' | sed 's/\.yaml//'
    echo ""
    echo "Modes:"
    echo "  single:       Run vLLM server on current node only"
    echo "  distributed:  Run vLLM server with multi-node tensor parallelism"
    echo ""
    echo "Examples:"
    echo "  $0 my_exp                           # Single-node vLLM server"
    echo "  $0 multi_node_example single       # Single-node vLLM server"
    echo "  $0 multi_node_example distributed  # Multi-node vLLM server"
    exit 1
fi

# Get config value from config file
get_config() {
    local key="$1"
    local default="$2"
    local config_name="${3:-config}"
    
    python3 src/simple_config_loader.py --config-name "$config_name" --key "$key" --default "$default" 2>/dev/null
}

# Parse arguments
CONFIG_NAME="${1:-config}"
MODE="${2:-single}"

echo "=========================================="
echo "NOVER Multi-Node vLLM Server Script"
echo "=========================================="
echo "Configuration: $CONFIG_NAME"
echo "Mode: $MODE"
echo ""

# Get model and vLLM configuration
MODEL_NAME=$(get_config "model.name_vllm" "YOUR_MODEL_NAME_VLLM" "$CONFIG_NAME")
PORT=$(get_config "vllm.port" "8087" "$CONFIG_NAME")
HOST=$(get_config "vllm.host" "localhost" "$CONFIG_NAME")
GPU_MEM_UTIL=$(get_config "vllm.gpu_memory_utilization" "0.85" "$CONFIG_NAME")

# Get GPU configuration
GPU_IDS=$(get_config "gpu.vllm.gpu_ids" "1" "$CONFIG_NAME")
TENSOR_PARALLEL_SIZE=$(get_config "gpu.vllm.tensor_parallel_size" "1" "$CONFIG_NAME")

# Get distributed configuration
NUM_NODES=$(get_config "distributed.num_nodes" "1" "$CONFIG_NAME")
NODE_RANK=$(get_config "distributed.node_rank" "0" "$CONFIG_NAME")
MASTER_ADDR=$(get_config "distributed.master_addr" "localhost" "$CONFIG_NAME")
MASTER_PORT_DIST=$(get_config "distributed.master_port" "29500" "$CONFIG_NAME")
NCCL_SOCKET_IFNAME=$(get_config "distributed.nccl_socket_ifname" "eth0" "$CONFIG_NAME")

# Set environment variables
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export VLLM_ATTENTION_BACKEND=triton

echo "Model Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  GPU Memory Utilization: $GPU_MEM_UTIL"
echo "  GPU IDs: $GPU_IDS"
echo ""

if [ "$MODE" = "distributed" ] && [ "$NUM_NODES" -gt 1 ]; then
    echo "Distributed vLLM Configuration:"
    echo "  Total nodes: $NUM_NODES"
    echo "  Current node rank: $NODE_RANK"
    echo "  Master address: $MASTER_ADDR"
    echo "  Tensor parallel size: $TENSOR_PARALLEL_SIZE"
    echo "  Network interface: $NCCL_SOCKET_IFNAME"
    echo ""
    
    # Set distributed environment variables for vLLM
    export MASTER_ADDR="$MASTER_ADDR"
    export MASTER_PORT="$((MASTER_PORT_DIST + 1))"  # Use different port than training
    export NODE_RANK="$NODE_RANK"
    export WORLD_SIZE="$NUM_NODES"
    export LOCAL_RANK=0
    
    # Set NCCL environment variables
    export NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1
    export NCCL_TIMEOUT=3600
    export CUDA_LAUNCH_BLOCKING=1
    
    echo "Distributed Environment Variables:"
    echo "  MASTER_ADDR: $MASTER_ADDR"
    echo "  MASTER_PORT: $MASTER_PORT"
    echo "  NODE_RANK: $NODE_RANK"
    echo "  WORLD_SIZE: $WORLD_SIZE"
    echo ""
    
    echo "=========================================="
    echo "Multi-Node vLLM Setup Instructions:"
    echo "=========================================="
    echo "1. Run this script on all nodes with the same configuration"
    echo "2. Start nodes in order: node 0 (master) first, then others"
    echo "3. Wait for all nodes to connect before starting training"
    echo ""
    echo "Current node rank: $NODE_RANK"
    if [ "$NODE_RANK" -eq 0 ]; then
        echo "This is the MASTER node (rank 0)"
        echo "Other nodes will connect to this node"
    else
        echo "This is WORKER node (rank $NODE_RANK)"
        echo "This node will connect to master at $MASTER_ADDR"
    fi
    echo ""
    read -p "Press Enter to continue or Ctrl+C to abort..."
    
    # Start distributed vLLM server
    echo "Starting distributed vLLM server..."
    echo "=========================================="
    
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_NAME" \
        --host "$HOST" \
        --port "$PORT" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --distributed-executor-backend ray \
        --worker-use-ray
        
elif [ "$MODE" = "single" ] || [ "$NUM_NODES" -eq 1 ]; then
    echo "Single Node Configuration:"
    echo "  Tensor parallel size: $TENSOR_PARALLEL_SIZE"
    echo ""
    
    # Set simple environment variables for single node
    export MASTER_PORT="29505"  # Different from training port
    export NCCL_P2P_DISABLE=1
    export CUDA_LAUNCH_BLOCKING=1
    
    echo "Starting single-node vLLM server..."
    echo "=========================================="
    
    # Use TRL's vllm-serve command for single node (as in original script)
    NCCL_DEBUG=WARN trl vllm-serve \
        --model "$MODEL_NAME" \
        --port "$PORT" \
        --gpu_memory_utilization "$GPU_MEM_UTIL" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
else
    echo "Error: Invalid mode '$MODE'. Use 'single' or 'distributed'."
    exit 1
fi

echo ""
echo "vLLM server stopped!"