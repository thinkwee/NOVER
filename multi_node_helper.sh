#!/bin/bash

# NOVER Multi-Node Setup Helper
# This script helps you set up and launch multi-node training

set -e

show_help() {
    echo "NOVER Multi-Node Setup Helper"
    echo "=============================="
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  setup <config_name>    - Generate node-specific config files"
    echo "  start-vllm <config>    - Start vLLM server"
    echo "  start-training <config> [launcher] - Start training (accelerate|torchrun)"
    echo "  status                 - Check system status"
    echo "  help                   - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 setup multi_node_example     # Generate configs for each node"
    echo "  $0 start-vllm multi_node_example"
    echo "  $0 start-training multi_node_example accelerate"
    echo "  $0 status                        # Check GPUs and network"
    echo ""
}

get_config() {
    local key="$1"
    local default="$2"
    local config_name="${3:-config}"
    
    python3 src/simple_config_loader.py --config-name "$config_name" --key "$key" --default "$default" 2>/dev/null
}

setup_configs() {
    local base_config="$1"
    
    if [ ! -f "config/${base_config}.yaml" ]; then
        echo "Error: Configuration file config/${base_config}.yaml not found"
        exit 1
    fi
    
    local num_nodes=$(get_config "distributed.num_nodes" "1" "$base_config")
    local master_addr=$(get_config "distributed.master_addr" "localhost" "$base_config")
    
    echo "Setting up configuration files for $num_nodes nodes..."
    echo "Master address: $master_addr"
    echo ""
    
    for ((i=0; i<num_nodes; i++)); do
        local config_file="config/${base_config}_node${i}.yaml"
        
        cat > "$config_file" << EOF
# Auto-generated node-specific configuration for node $i
defaults:
  - $base_config

distributed:
  node_rank: $i
  master_addr: "$master_addr"
EOF
        
        echo "Created: $config_file"
    done
    
    echo ""
    echo "Configuration files created successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Copy appropriate config files to each node"
    echo "2. Start vLLM server: $0 start-vllm ${base_config}_node0"
    echo "3. Start training on each node:"
    for ((i=0; i<num_nodes; i++)); do
        echo "   Node $i: $0 start-training ${base_config}_node${i}"
    done
}

start_vllm() {
    local config="$1"
    echo "Starting vLLM server with config: $config"
    ./run_multi_node_vllm.sh "$config" single
}

start_training() {
    local config="$1"
    local launcher="${2:-accelerate}"
    
    echo "Starting training with config: $config, launcher: $launcher"
    ./run_multi_node_training.sh "$config" "$launcher"
}

check_status() {
    echo "NOVER Multi-Node System Status"
    echo "=============================="
    echo ""
    
    # Check GPU status
    echo "GPU Status:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    else
        echo "  nvidia-smi not found"
    fi
    echo ""
    
    # Check network interfaces
    echo "Network Interfaces:"
    ip addr show | grep -E "^[0-9]+:|inet " | while read line; do
        echo "  $line"
    done
    echo ""
    
    # Check available ports
    echo "Port Status:"
    for port in 29500 28890 8087; do
        if ss -ln | grep -q ":$port "; then
            echo "  Port $port: IN USE"
        else
            echo "  Port $port: Available"
        fi
    done
    echo ""
    
    # Check Python environment
    echo "Python Environment:"
    if python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null; then
        python3 -c "import accelerate; print(f'  Accelerate: {accelerate.__version__}')" 2>/dev/null || echo "  Accelerate: Not found"
        python3 -c "import transformers; print(f'  Transformers: {transformers.__version__}')" 2>/dev/null || echo "  Transformers: Not found"
        python3 -c "import trl; print(f'  TRL: {trl.__version__}')" 2>/dev/null || echo "  TRL: Not found"
    else
        echo "  PyTorch: Not found"
    fi
    echo ""
    
    # Check NOVER components
    echo "NOVER Components:"
    if [ -f "src/main.py" ]; then
        echo "  Main script: Found"
    else
        echo "  Main script: Not found"
    fi
    
    if [ -f "config/config.yaml" ]; then
        echo "  Base config: Found"
    else
        echo "  Base config: Not found"
    fi
    
    echo ""
    echo "Available configurations:"
    ls config/*.yaml 2>/dev/null | sed 's/config\///; s/\.yaml//' | sed 's/^/  /' || echo "  None found"
}

# Main script logic
case "${1:-help}" in
    setup)
        if [ -z "$2" ]; then
            echo "Error: Please specify a base configuration name"
            echo "Usage: $0 setup <base_config_name>"
            exit 1
        fi
        setup_configs "$2"
        ;;
    start-vllm)
        if [ -z "$2" ]; then
            echo "Error: Please specify a configuration name"
            echo "Usage: $0 start-vllm <config_name>"
            exit 1
        fi
        start_vllm "$2"
        ;;
    start-training)
        if [ -z "$2" ]; then
            echo "Error: Please specify a configuration name"
            echo "Usage: $0 start-training <config_name> [launcher]"
            exit 1
        fi
        start_training "$2" "$3"
        ;;
    status)
        check_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Error: Unknown command '$1'"
        echo ""
        show_help
        exit 1
        ;;
esac