import os
import torch
import wandb
import yaml
from datetime import datetime
from trl import GRPOConfig
from peft import LoraConfig
from omegaconf import DictConfig

def setup_environment():
    """
    Set up environment variables for distributed training and clear GPU memory.
    """
    os.environ.update({
        "NCCL_DEBUG": "ERROR",
        "NCCL_TIMEOUT": "3600",
        "NCCL_SOCKET_TIMEOUT": "3600",
        "NCCL_IB_TIMEOUT": "3600",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "NCCL_P2P_DISABLE": "1",  
        "NCCL_IB_DISABLE": "0",
        "NCCL_SOCKET_IFNAME": "^docker0,lo",
    })
    torch.cuda.empty_cache()

def get_precision_config():
    """
    Determine the appropriate precision settings based on available hardware.
    Returns a tuple of (precision_type, torch_dtype, vllm_dtype).
    """
    if not torch.cuda.is_available():
        return "fp16", torch.float16, "half"
    
    gpu_props = torch.cuda.get_device_properties(0)
    compute_capability = float(f"{gpu_props.major}.{gpu_props.minor}")
    
    return ("bf16", torch.bfloat16, "bfloat16") if compute_capability >= 8.0 else ("fp16", torch.float16, "half")

def get_output_dir(config: DictConfig, suffix=""):
    """
    Generate the output directory path for saving model checkpoints.
    If resuming training, uses the adapter directory.
    Otherwise, creates a new directory with timestamp.
    """
    if config.model.resume_from_checkpoint and config.model.adapter_dir:
        return config.model.adapter_dir
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_name_short = config.model.name.split('/')[-1]
    base_dir = config.project.save_base_path
    dir_name = f"{model_name_short}_{timestamp}"
    if suffix:
        dir_name += f"_{suffix}"
    return os.path.join(base_dir, dir_name)

def init_wandb(config: DictConfig):
    """
    Initialize Weights & Biases logging with appropriate configuration.
    """
    model_short = config.model.name.split('/')[-1]
    timestamp = datetime.now().strftime('%m%d_%H%M')
    
    if config.model.resume_from_checkpoint:
        run_name = f"{model_short}_r{config.model.latest_checkpoint_step}_{timestamp}"
    else:
        run_name = f"{model_short}_{timestamp}"
    
    if config.project.suffix:
        run_name = f"{config.project.suffix}_{run_name}"
    
    precision_type, torch_dtype, vllm_dtype = get_precision_config()
    
    # Convert ListConfig objects to regular Python lists for JSON serialization
    target_modules = list(config.lora.target_modules) if hasattr(config.lora, 'target_modules') else []
    modules_to_save = list(config.lora.modules_to_save) if hasattr(config.lora, 'modules_to_save') else []
    
    wandb.init(
        project=config.project.wandb_project,
        name=run_name,
        config={
            "model": config.model.name,
            "output_dir": get_output_dir(config, suffix=config.project.suffix),
            "resume_from_checkpoint": config.model.resume_from_checkpoint,
            "latest_checkpoint_step": config.model.latest_checkpoint_step if config.model.resume_from_checkpoint else None,
            "adapter_dir": config.model.adapter_dir if config.model.resume_from_checkpoint else None,
            "batch_size": config.training.batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "epochs": config.training.num_train_epochs,
            "num_iterations": config.training.num_iterations,
            "num_generations": config.training.num_generations,
            "beta": config.training.beta,
            "logging_steps": config.training.logging_steps,
            "save_steps": config.training.save_steps,
            "save_total_limit": config.training.save_total_limit,
            "scale_rewards": config.training.scale_rewards,
            "epsilon": config.training.epsilon,
            "epsilon_high": config.training.epsilon_high,
            "learning_rate": config.training.learning_rate,
            "temperature": config.training.temperature,
            "max_completion_length": config.training.max_completion_length,
            "sync_ref_model": config.training.sync_ref_model,
            "ref_model_mixup_alpha": config.training.ref_model_mixup_alpha,
            "ref_model_sync_steps": config.training.ref_model_sync_steps,
            "use_vllm": config.vllm.use_vllm,
            "vllm_host": config.vllm.host if config.vllm.use_vllm else None,
            "vllm_port": config.vllm.port if config.vllm.use_vllm else None,
            "vllm_gpu_memory_utilization": config.vllm.gpu_memory_utilization,
            "vllm_request_timeout": config.vllm.request_timeout,
            "vllm_mode": config.vllm.mode,
            "vllm_guided_decoding_regex": config.vllm.guided_decoding_regex,
            "vllm_tensor_parallel_size": config.vllm.tensor_parallel_size,
            # Advanced GRPO features
            "loss_type": config.training.loss_type,
            "mask_truncated_completions": config.training.mask_truncated_completions,
            "top_entropy_quantile": config.training.top_entropy_quantile,
            "importance_sampling_level": config.training.importance_sampling_level,
            "use_liger_loss": config.training.use_liger_loss,
            "use_transformers_paged": config.training.use_transformers_paged,
            "reward_weights": {
                "tag_format_reward": config.reward.tag_format_reward_weight,
                "reasoning_reward": config.reward.reasoning_reward_weight,
                "efficiency_reward": config.reward.efficiency_reward_weight
            },
            "peft_config": {
                "r": config.lora.r,
                "lora_alpha": config.lora.lora_alpha,
                "lora_dropout": config.lora.lora_dropout,
                "bias": config.lora.bias,
                "task_type": config.lora.task_type,
                "target_modules": target_modules,
                "modules_to_save": modules_to_save
            },
            "dataset": {
                "name": config.dataset.name,
                "subset": config.dataset.subset,
                "validation_size": config.dataset.validation_size
            }
        }
    )

def get_peft_config(config: DictConfig):
    """Create LoRA configuration from config"""
    # Convert ListConfig objects to regular Python lists
    target_modules = list(config.lora.target_modules) if hasattr(config.lora, 'target_modules') else []
    modules_to_save = list(config.lora.modules_to_save) if hasattr(config.lora, 'modules_to_save') else []
    
    return LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )

def get_training_config(config: DictConfig):
    """
    Create and return the GRPO training configuration object.
    """
    precision_type, torch_dtype, vllm_dtype = get_precision_config()
    
    grpo_config = GRPOConfig(
        # Output settings
        output_dir=get_output_dir(config, suffix=config.project.suffix),
        
        # Batch settings
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        dataloader_drop_last=True,
        
        # Optimization settings
        learning_rate=config.training.learning_rate,
        gradient_checkpointing=True,
        **{precision_type: True},
        
        # Training loop settings
        num_train_epochs=config.training.num_train_epochs,
        num_iterations=config.training.num_iterations,
        beta=config.training.beta,
        
        # Checkpoint and logging
        save_strategy="steps",
        save_steps=config.training.save_steps,
        logging_steps=config.training.logging_steps,
        eval_strategy=config.training.evaluation_strategy,
        eval_steps=config.training.eval_steps,
        save_total_limit=config.training.save_total_limit,
        report_to=["wandb"],
        log_completions=True,
        
        # Model initialization
        model_init_kwargs={
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True
        },
        
        # GRPO specific settings
        num_generations=config.training.num_generations,
        temperature=config.training.temperature,
        epsilon=config.training.epsilon,
        epsilon_high=config.training.epsilon_high,
        max_completion_length=config.training.max_completion_length,
        reward_weights=[
            float(config.reward.tag_format_reward_weight),
            float(config.reward.reasoning_reward_weight),
            float(config.reward.efficiency_reward_weight)
        ],
        scale_rewards=config.training.scale_rewards,
        
        # New GRPO settings in updated TRL version
        loss_type=config.training.loss_type,
        mask_truncated_completions=config.training.mask_truncated_completions,
        top_entropy_quantile=config.training.top_entropy_quantile,
        importance_sampling_level=config.training.importance_sampling_level,
        use_liger_loss=config.training.use_liger_loss,
        use_transformers_paged=config.training.use_transformers_paged,
        
        # Reference model settings
        sync_ref_model=config.training.sync_ref_model,
        ref_model_mixup_alpha=config.training.ref_model_mixup_alpha,
        ref_model_sync_steps=config.training.ref_model_sync_steps,
        
        # VLLM settings
        use_vllm=config.vllm.use_vllm,
        vllm_server_host=config.vllm.host if config.vllm.use_vllm else None,
        vllm_server_port=config.vllm.port if config.vllm.use_vllm else None,
        vllm_server_timeout=config.vllm.request_timeout if config.vllm.use_vllm else None,
        
        # New vLLM settings in updated TRL version
        vllm_mode=config.vllm.mode if config.vllm.use_vllm else "server",
        vllm_guided_decoding_regex=config.vllm.guided_decoding_regex,
        vllm_gpu_memory_utilization=config.vllm.gpu_memory_utilization,
        vllm_tensor_parallel_size=config.vllm.tensor_parallel_size,
    )
    
    return grpo_config

def get_dataset_path(config: DictConfig, dataset_name=None):
    """Return the full path to the dataset."""
    if dataset_name is None:
        dataset_name = config.dataset.name
    return os.path.join(config.dataset.hf_home, "datasets", dataset_name)

def save_config_to_yaml(config: DictConfig):
    """
    Save the complete configuration to a YAML file in the output directory.
    """
    precision_type, torch_dtype, vllm_dtype = get_precision_config()
    
    # Convert ListConfig objects to regular Python lists
    target_modules = list(config.lora.target_modules) if hasattr(config.lora, 'target_modules') else []
    modules_to_save = list(config.lora.modules_to_save) if hasattr(config.lora, 'modules_to_save') else []
    
    config_dict = {
        "model": {
            "name": config.model.name,
            "vllm_name": config.model.name_vllm,
            "output_dir": get_output_dir(config, suffix=config.project.suffix),
            "resume_from_checkpoint": config.model.resume_from_checkpoint,
            "latest_checkpoint_step": config.model.latest_checkpoint_step if config.model.resume_from_checkpoint else None,
            "adapter_dir": config.model.adapter_dir if config.model.resume_from_checkpoint else None,
            "suffix": config.project.suffix,
            "save_base_path": config.project.save_base_path
        },
        "dataset": {
            "name": config.dataset.name,
            "subset": config.dataset.subset,
            "hf_home": config.dataset.hf_home,
            "validation_size": config.dataset.validation_size
        },
        "training": {
            "wandb_project": config.project.wandb_project,
            "batch_size": config.training.batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "num_train_epochs": config.training.num_train_epochs,
            "num_iterations": config.training.num_iterations,
            "beta": config.training.beta,
            "logging_steps": config.training.logging_steps,
            "save_steps": config.training.save_steps,
            "save_total_limit": config.training.save_total_limit,
            "num_generations": config.training.num_generations,
            "scale_rewards": config.training.scale_rewards,
            "epsilon": config.training.epsilon,
            "epsilon_high": config.training.epsilon_high,
            "max_completion_length": config.training.max_completion_length,
            "dataloader_drop_last": True,
            "gradient_checkpointing": True,
            "precision": precision_type,
            "save_strategy": "steps",
            "evaluation_strategy": "steps",
            "eval_steps": config.training.eval_steps,
            "model_init_kwargs": {
                "torch_dtype": str(torch_dtype),
                "low_cpu_mem_usage": True
            },
            "log_completions": True,
            "report_to": ["wandb"],
            "reward_weights": {
                "tag_format_reward": config.reward.tag_format_reward_weight,
                "reasoning_reward": config.reward.reasoning_reward_weight,
                "efficiency_reward": config.reward.efficiency_reward_weight,
                "weights_list": [
                    float(config.reward.tag_format_reward_weight),
                    float(config.reward.reasoning_reward_weight),
                    float(config.reward.efficiency_reward_weight)
                ]
            },
            # New GRPO settings in updated TRL version
            "loss_type": config.training.loss_type,
            "mask_truncated_completions": config.training.mask_truncated_completions,
            "top_entropy_quantile": config.training.top_entropy_quantile,
            "importance_sampling_level": config.training.importance_sampling_level,
            "use_liger_loss": config.training.use_liger_loss,
            "use_transformers_paged": config.training.use_transformers_paged
        },
        "reference_model": {
            "sync_ref_model": config.training.sync_ref_model,
            "ref_model_mixup_alpha": config.training.ref_model_mixup_alpha,
            "ref_model_sync_steps": config.training.ref_model_sync_steps,
        },
        "vllm": {
            "enabled": config.vllm.use_vllm,
            "host": config.vllm.host if config.vllm.use_vllm else None,
            "port": config.vllm.port if config.vllm.use_vllm else None,
            "temperature": config.training.temperature,
            "gpu_memory_utilization": config.vllm.gpu_memory_utilization,
            "request_timeout": config.vllm.request_timeout,
            # New vLLM settings in updated TRL version
            "mode": config.vllm.mode,
            "guided_decoding_regex": config.vllm.guided_decoding_regex,
            "tensor_parallel_size": config.vllm.tensor_parallel_size
        },
        "peft": {
            "r": config.lora.r,
            "lora_alpha": config.lora.lora_alpha,
            "lora_dropout": config.lora.lora_dropout,
            "bias": config.lora.bias,
            "task_type": config.lora.task_type,
            "target_modules": target_modules,
            "modules_to_save": modules_to_save
        },
        "environment": {
            "nccl_debug": "INFO",
            "nccl_timeout": "3600",
            "nccl_socket_timeout": "3600",
            "nccl_ib_timeout": "3600",
            "pytorch_cuda_alloc_conf": "expandable_segments:True",
            "nccl_p2p_disable": "1",
            "nccl_ib_disable": "0",
            "nccl_socket_ifname": "^docker0,lo"
        }
    }
    
    output_dir = get_output_dir(config, suffix=config.project.suffix)
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False) 