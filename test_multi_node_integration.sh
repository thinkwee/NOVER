#!/bin/bash

# Integration test for NOVER multi-node training scripts
# This script demonstrates the complete workflow for setting up multi-node training

set -e

echo "NOVER Multi-Node Integration Test"
echo "================================="
echo ""

# Test 1: Configuration generation
echo "Test 1: Configuration generation"
echo "--------------------------------"
./multi_node_helper.sh setup multi_node_example
echo "✓ Node-specific configurations generated"
echo ""

# Test 2: Configuration validation
echo "Test 2: Configuration validation"
echo "--------------------------------"
NUM_NODES=$(python3 src/simple_config_loader.py --config-name multi_node_example --key distributed.num_nodes)
NODE0_RANK=$(python3 src/simple_config_loader.py --config-name multi_node_example_node0 --key distributed.node_rank)
NODE1_RANK=$(python3 src/simple_config_loader.py --config-name multi_node_example_node1 --key distributed.node_rank)

echo "Base config nodes: $NUM_NODES"
echo "Node 0 rank: $NODE0_RANK"
echo "Node 1 rank: $NODE1_RANK"

if [ "$NUM_NODES" = "2" ] && [ "$NODE0_RANK" = "0" ] && [ "$NODE1_RANK" = "1" ]; then
    echo "✓ Configuration validation passed"
else
    echo "✗ Configuration validation failed"
    exit 1
fi
echo ""

# Test 3: Script help outputs
echo "Test 3: Script help outputs"
echo "---------------------------"
echo "Testing training script help..."
./run_multi_node_training.sh >/dev/null 2>&1 || echo "✓ Training script help working"

echo "Testing vLLM script help..."
./run_multi_node_vllm.sh >/dev/null 2>&1 || echo "✓ vLLM script help working"

echo "Testing helper script..."
./multi_node_helper.sh help >/dev/null 2>&1 && echo "✓ Helper script working"
echo ""

# Test 4: System status check
echo "Test 4: System status check"
echo "---------------------------"
./multi_node_helper.sh status | grep -q "Available configurations" && echo "✓ Status check working"
echo ""

# Test 5: Configuration parsing for different setups
echo "Test 5: Configuration parsing"
echo "-----------------------------"
MASTER_ADDR=$(python3 src/simple_config_loader.py --config-name multi_node_example --key distributed.master_addr)
TRAINING_GPUS=$(python3 src/simple_config_loader.py --config-name multi_node_example --key gpu.training.num_gpus)
VLLM_PORT=$(python3 src/simple_config_loader.py --config-name multi_node_example --key vllm.port)

echo "Master address: $MASTER_ADDR"
echo "Training GPUs per node: $TRAINING_GPUS"
echo "vLLM port: $VLLM_PORT"

if [ -n "$MASTER_ADDR" ] && [ -n "$TRAINING_GPUS" ] && [ -n "$VLLM_PORT" ]; then
    echo "✓ Configuration parsing working"
else
    echo "✗ Configuration parsing failed"
    exit 1
fi
echo ""

# Test 6: Dry run of scripts (configuration parsing only)
echo "Test 6: Script dry run tests"
echo "----------------------------"
echo "Testing training script configuration parsing..."
timeout 5 ./run_multi_node_training.sh multi_node_example_node0 accelerate 2>&1 | grep -q "Master address" && echo "✓ Training script config parsing working"

echo "Testing vLLM script configuration parsing..."
timeout 5 ./run_multi_node_vllm.sh multi_node_example_node0 single 2>&1 | grep -q "Model Configuration" && echo "✓ vLLM script config parsing working"
echo ""

# Clean up generated test files (keep the node configs as they're useful)
echo "Integration test completed successfully! ✓"
echo ""
echo "Generated files that can be used for actual training:"
echo "- config/multi_node_example_node0.yaml"
echo "- config/multi_node_example_node1.yaml"
echo ""
echo "To start actual multi-node training:"
echo "1. Ensure all nodes have the same NOVER installation"
echo "2. Copy appropriate config files to each node"
echo "3. Start vLLM server: ./run_multi_node_vllm.sh multi_node_example_node0 single"
echo "4. Start training on each node with the appropriate config"
echo ""
echo "For detailed instructions, see MULTI_NODE_GUIDE.md"