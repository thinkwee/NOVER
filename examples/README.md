# NOVER Multi-Node Training Examples

This directory contains practical examples for deploying NOVER multi-node training in different environments.

## Files

### `slurm_multi_node.sh`
Complete SLURM job script for running NOVER training on a multi-node HPC cluster.

**Usage:**
```bash
# Submit to SLURM scheduler
sbatch examples/slurm_multi_node.sh

# Or customize and use as template
cp examples/slurm_multi_node.sh my_job.sh
# Edit my_job.sh with your specific paths and settings
sbatch my_job.sh
```

**Key features:**
- Automatic SLURM environment detection
- Dynamic configuration generation
- Integrated vLLM server management
- Proper cleanup after job completion

### `docker-compose-multi-node.yml`
Docker Compose configuration for containerized multi-node training.

**Usage:**
```bash
# Build and start containers
docker-compose -f examples/docker-compose-multi-node.yml up --build

# Scale to more workers
docker-compose -f examples/docker-compose-multi-node.yml up --scale nover-worker=3
```

**Key features:**
- Isolated container environments
- Shared storage volumes
- GPU allocation per container
- Network configuration for inter-container communication

### `docker_multi_node.yaml`
Configuration template optimized for Docker deployments.

**Key features:**
- Container-friendly network settings
- Shared volume paths
- Resource-optimized batch sizes

## Quick Start

### SLURM Environment
1. Copy and customize the SLURM script:
   ```bash
   cp examples/slurm_multi_node.sh my_training.sh
   ```

2. Edit paths and configuration:
   - Update `/path/to/nover/venv/bin/activate`
   - Update `/path/to/NOVER`
   - Modify SLURM directives for your cluster

3. Submit job:
   ```bash
   sbatch my_training.sh
   ```

### Docker Environment
1. Ensure Docker and docker-compose are installed
2. Build NOVER Docker image (create Dockerfile first)
3. Copy configuration:
   ```bash
   cp examples/docker_multi_node.yaml config/
   ```

4. Start multi-node training:
   ```bash
   docker-compose -f examples/docker-compose-multi-node.yml up
   ```

## Configuration Notes

### SLURM Specific
- Uses `scontrol` to detect master node automatically
- Integrates with SLURM's `$SLURM_*` environment variables
- Automatically sets up distributed training parameters

### Docker Specific
- Uses Docker service names for networking
- Shared volumes for model and data access
- GPU device allocation per container

## Troubleshooting

### SLURM Issues
- Verify module loads work on your cluster
- Check shared storage accessibility
- Ensure SLURM partition supports GPU allocation

### Docker Issues
- Verify NVIDIA Docker runtime is installed
- Check GPU device mappings
- Ensure shared volumes have proper permissions

## Customization

Both examples can be customized for your specific setup:

1. **Resource allocation**: Modify GPU counts, memory, CPU cores
2. **Network configuration**: Adjust network interfaces and ports
3. **Storage paths**: Update shared storage mount points
4. **Model configuration**: Change model sizes and training parameters

For more detailed configuration options, see the main [MULTI_NODE_GUIDE.md](../MULTI_NODE_GUIDE.md).