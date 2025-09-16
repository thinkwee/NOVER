#!/bin/bash
#SBATCH --job-name=nover_multi_node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=nover_%j.out
#SBATCH --error=nover_%j.err

# SLURM integration example for NOVER multi-node training
# This script demonstrates how to launch NOVER training in a SLURM-managed cluster

echo "NOVER Multi-Node Training on SLURM"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo ""

# Setup environment
module load cuda/11.8
module load python/3.9
source /path/to/nover/venv/bin/activate

# Change to NOVER directory
cd /path/to/NOVER

# Get SLURM environment information
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NNODES

echo "SLURM Environment:"
echo "  Master node: $MASTER_ADDR"
echo "  Node rank: $NODE_RANK"
echo "  World size: $WORLD_SIZE"
echo ""

# Configuration name (should be created beforehand)
CONFIG_NAME="slurm_multi_node"

# Generate node-specific configurations if they don't exist
if [ "$NODE_RANK" -eq 0 ]; then
    echo "Master node: generating configurations..."
    
    # Create base SLURM configuration if it doesn't exist
    if [ ! -f "config/${CONFIG_NAME}.yaml" ]; then
        cat > "config/${CONFIG_NAME}.yaml" << EOF
defaults:
  - config

project:
  suffix: "slurm_multi_node"
  wandb_project: "nover_slurm"
  save_base_path: "/shared/models/nover"

dataset:
  name: "your_dataset"
  hf_home: "/shared/huggingface"

model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  name_vllm: "Qwen/Qwen2.5-7B-Instruct"

training:
  batch_size: 2
  save_steps: 100
  logging_steps: 10

distributed:
  num_nodes: $SLURM_NNODES
  node_rank: 0  # Will be overridden
  master_addr: "$MASTER_ADDR"
  master_port: "29500"
  nccl_socket_ifname: "ib0"
  backend: "nccl"

gpu:
  training:
    gpu_ids: "0,1,2,3,4,5,6,7"
    num_gpus: 8
  vllm:
    gpu_ids: "0,1"
    tensor_parallel_size: 2

vllm:
  port: 8087
  host: "0.0.0.0"
EOF
    fi
    
    # Generate node-specific configs
    ./multi_node_helper.sh setup "$CONFIG_NAME"
    
    # Start vLLM server on master node
    echo "Starting vLLM server on master node..."
    ./run_multi_node_vllm.sh "${CONFIG_NAME}_node0" single &
    VLLM_PID=$!
    
    # Wait a bit for vLLM to start
    sleep 30
fi

# Synchronize all nodes before starting training
echo "Waiting for all nodes to be ready..."
sleep 60

# Create node-specific config name
NODE_CONFIG="${CONFIG_NAME}_node${NODE_RANK}"

echo "Starting training on node $NODE_RANK with config $NODE_CONFIG..."

# Launch training using torchrun (better for SLURM integration)
srun ./run_multi_node_training.sh "$NODE_CONFIG" torchrun

# Cleanup on master node
if [ "$NODE_RANK" -eq 0 ] && [ -n "$VLLM_PID" ]; then
    echo "Training completed. Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null || true
fi

echo "SLURM job completed!"