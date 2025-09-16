# Multi-Node Multi-GPU Training Guide for NOVER

This guide provides detailed instructions for setting up and running NOVER training across multiple nodes with multiple GPUs.

## Overview

NOVER now supports distributed training across multiple nodes using either:
- **Accelerate**: HuggingFace's distributed training launcher
- **Torchrun**: PyTorch's native distributed launcher

## Prerequisites

### Hardware Requirements
- Multiple nodes with GPUs
- High-speed network connection between nodes (InfiniBand recommended)
- Shared storage accessible from all nodes (NFS, Lustre, etc.)

### Software Requirements
- Same NOVER environment installed on all nodes
- SSH access between nodes
- Same CUDA and PyTorch versions across all nodes

### Network Configuration
- All nodes must be able to communicate on the specified ports
- Firewall rules allowing traffic on training and vLLM ports
- Consistent network interface names across nodes

## Configuration

### 1. Multi-Node Configuration File

Create a configuration file for your multi-node setup (see `config/multi_node_example.yaml`):

```yaml
# Key multi-node settings
distributed:
  num_nodes: 2                    # Total number of nodes
  node_rank: 0                    # Current node rank (0, 1, 2, ...)
  master_addr: "10.0.0.1"        # IP address of master node
  master_port: "29500"           # Port for distributed communication
  nccl_socket_ifname: "eth0"     # Network interface name
  backend: "nccl"                # Distributed backend

# Shared storage paths (accessible from all nodes)
project:
  save_base_path: "/shared/models"

dataset:
  hf_home: "/shared/huggingface"

# GPU configuration per node
gpu:
  training:
    gpu_ids: "0,1,2,3"          # GPUs for training on this node
    num_gpus: 4                 # Number of training GPUs per node
  vllm:
    gpu_ids: "4,5,6,7"          # GPUs for vLLM on this node  
    tensor_parallel_size: 4     # vLLM tensor parallelism per node
```

### 2. Node-Specific Configuration

Each node needs its own configuration file with the correct `node_rank`:

**Node 0 (Master):**
```yaml
distributed:
  node_rank: 0
  master_addr: "10.0.0.1"  # This node's IP
```

**Node 1 (Worker):**
```yaml
distributed:
  node_rank: 1
  master_addr: "10.0.0.1"  # Master node's IP
```

## Setup Instructions

### 1. Prepare Shared Storage

Ensure all nodes have access to shared directories:
- Model checkpoints directory
- Dataset directory  
- HuggingFace cache directory

```bash
# Example NFS mount on all nodes
sudo mount -t nfs master-node:/shared /shared
```

### 2. Network Configuration

Configure network interfaces and firewall rules:

```bash
# Allow distributed training ports
sudo ufw allow 29500  # Distributed communication
sudo ufw allow 28890  # Accelerate main process
sudo ufw allow 8087   # vLLM server
```

### 3. Environment Setup

Install NOVER on all nodes with identical environments:

```bash
# On each node
git clone https://github.com/thinkwee/NOVER.git
cd NOVER
pip install -r requirements.txt
```

## Running Multi-Node Training

### Option 1: Using Accelerate (Recommended)

**Step 1: Start vLLM server on one or more nodes**
```bash
# On the vLLM server node(s)
./run_multi_node_vllm.sh multi_node_example single
```

**Step 2: Launch training on all nodes**

Start with the master node (rank 0):
```bash
# On master node (rank 0)
./run_multi_node_training.sh multi_node_example accelerate
```

Then start worker nodes:
```bash
# On worker node (rank 1)
./run_multi_node_training.sh multi_node_worker1 accelerate

# On worker node (rank 2) 
./run_multi_node_training.sh multi_node_worker2 accelerate
```

### Option 2: Using Torchrun

**Step 1: Start vLLM server**
```bash
./run_multi_node_vllm.sh multi_node_example single
```

**Step 2: Launch training**
```bash
# On all nodes simultaneously
./run_multi_node_training.sh multi_node_example torchrun
```

## Configuration Examples

### Example 1: 2 Nodes, 8 GPUs Each (16 GPUs Total)

**Node 0 Config (`config/multi_node_2x8.yaml`):**
```yaml
distributed:
  num_nodes: 2
  node_rank: 0
  master_addr: "192.168.1.10"

gpu:
  training:
    gpu_ids: "0,1,2,3,4,5,6,7"
    num_gpus: 8
```

**Node 1 Config (`config/multi_node_2x8_worker.yaml`):**
```yaml
distributed:
  num_nodes: 2
  node_rank: 1
  master_addr: "192.168.1.10"

gpu:
  training:
    gpu_ids: "0,1,2,3,4,5,6,7"
    num_gpus: 8
```

### Example 2: 4 Nodes, 4 GPUs Each (16 GPUs Total)

Configure similar files for each node with `node_rank: 0, 1, 2, 3`.

## Monitoring and Debugging

### 1. Check Network Connectivity
```bash
# Test connectivity between nodes
nc -v <master_ip> 29500
```

### 2. Monitor Training Progress
```bash
# Check logs on master node
tail -f nohup.out

# Monitor GPU usage across nodes
watch -n 1 nvidia-smi
```

### 3. Common Issues and Solutions

**Issue: NCCL timeout errors**
```bash
# Increase timeout values
export NCCL_TIMEOUT=3600
export NCCL_SOCKET_TIMEOUT=3600
```

**Issue: Network interface problems**
```bash
# List available network interfaces
ip addr show

# Update config with correct interface
nccl_socket_ifname: "ib0"  # For InfiniBand
nccl_socket_ifname: "ens3" # For Ethernet
```

**Issue: Port conflicts**
```bash
# Check port availability
netstat -tulpn | grep :29500

# Use different ports in config
master_port: "29501"
```

## Performance Optimization

### 1. Network Optimization
- Use InfiniBand if available
- Disable P2P for better stability: `NCCL_P2P_DISABLE=1`
- Tune network interface: `NCCL_SOCKET_IFNAME=ib0`

### 2. Memory Optimization
- Enable expandable segments: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Adjust batch size per GPU when scaling
- Use gradient checkpointing for large models

### 3. Communication Optimization
- Use NCCL backend for GPU-to-GPU communication
- Optimize AllReduce operations with proper topology

## Scaling Guidelines

### Batch Size Scaling
When scaling from 1 to N nodes:
- Keep total effective batch size constant
- Reduce per-GPU batch size by factor of N
- Adjust learning rate if needed

Example:
- Single node: `batch_size: 8` (8 GPUs) = effective batch size 64
- 2 nodes: `batch_size: 4` (16 GPUs) = effective batch size 64

### Learning Rate Scaling
Consider linear learning rate scaling:
- 1 node: `learning_rate: 1e-5`
- 2 nodes: `learning_rate: 2e-5`
- 4 nodes: `learning_rate: 4e-5`

## Troubleshooting

### 1. Connection Issues
- Verify all nodes can reach master node IP
- Check firewall settings
- Ensure same ports are available on all nodes

### 2. Synchronization Issues
- Verify shared storage is properly mounted
- Check file permissions across nodes
- Ensure clocks are synchronized (use NTP)

### 3. GPU Issues
- Verify CUDA versions match across nodes
- Check GPU visibility with `nvidia-smi`
- Ensure driver versions are compatible

## Advanced Configurations

### Using SLURM

For SLURM-managed clusters, create a submission script:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

# Get node information
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID

# Run training
srun ./run_multi_node_training.sh multi_node_example torchrun
```

### Using Multiple vLLM Servers

For better inference throughput, run vLLM on multiple nodes:

```bash
# Node 0: vLLM server 1
./run_multi_node_vllm.sh multi_node_example single

# Node 1: vLLM server 2 (different port)
# Modify config to use port 8088
./run_multi_node_vllm.sh multi_node_example_alt single
```

Use a load balancer to distribute requests across vLLM servers.

## Best Practices

1. **Start Simple**: Begin with 2 nodes before scaling further
2. **Test Network**: Verify connectivity before starting training
3. **Monitor Resources**: Watch GPU memory and network bandwidth
4. **Use Shared Storage**: Ensure all nodes access same model/data
5. **Synchronize Clocks**: Use NTP to avoid timing issues
6. **Plan Checkpoints**: Save frequently with shared storage
7. **Log Everything**: Keep detailed logs for debugging

## Support

For additional support with multi-node setup:
1. Check the logs on all nodes for error messages
2. Verify network connectivity between nodes
3. Ensure configuration files are correct for each node
4. Monitor resource usage during training

The multi-node setup extends NOVER's capabilities to larger scale training while maintaining the same high-quality reasoning improvements.