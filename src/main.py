#!/usr/bin/env python3
"""
Main training script using Hydra configuration management.
"""

from omegaconf import DictConfig
from .config_manager import (
    setup_environment, init_wandb, get_training_config, get_peft_config, save_config_to_yaml
)
import hydra
from .data_loader import load_dataset
from .trainer import CustomGRPOTrainer
from .reward_functions import tag_format_reward, reasoning_reward, efficiency_reward, rule_based_reward, llm_as_judge_reward
import functools

# Enable progress bars explicitly
try:
    from transformers.utils import logging as transformers_logging
    transformers_logging.enable_progress_bar()
except ImportError:
    pass

def create_reward_factory(config: DictConfig):
    """
    Creates reward functions and weights based on the provided configuration.
    """

    # Map reward types to functions
    reward_function_map = {
        "tag_format": tag_format_reward,
        "reasoning": reasoning_reward,
        "efficiency": efficiency_reward,
        "rule_based": rule_based_reward,
        "llm_as_judge": llm_as_judge_reward,
    }

    reward_funcs = []
    reward_weights = []

    for reward_config in config.reward.functions:
        reward_type = reward_config.type
        weight = reward_config.weight

        if weight == 0:
            print(f"[INFO] Skipping reward function '{reward_type}' as its weight is 0.")
            continue

        if reward_type not in reward_function_map:
            print(f"[WARNING] Unknown reward function type: {reward_type}. Skipping.")
            continue

        base_reward_func = reward_function_map[reward_type]

        # Create a wrapper to pass specific arguments from the config
        if reward_type in ["tag_format", "reasoning", "efficiency"]:
            wrapped_func = functools.partial(
                base_reward_func,
                intermediate_tag=config.dataset.intermediate_tag,
                final_tag=config.dataset.final_tag
            )
        elif reward_type == "llm_as_judge":
            # Pass the entire sub-config for the judge
            judge_params = {k: v for k, v in reward_config.items() if k not in ['type', 'weight']}
            wrapped_func = functools.partial(
                base_reward_func,
                **judge_params
            )
        else:
            # Default case for functions that don't need special params
            wrapped_func = base_reward_func

        # Set a descriptive name for logging purposes
        wrapped_func.__name__ = reward_config.get("name", f"{reward_type.replace('_', ' ').title()} Reward")

        reward_funcs.append(wrapped_func)
        reward_weights.append(weight)

        print(f"[INFO] Initialized reward function: {wrapped_func.__name__} with weight {weight}")

    return reward_funcs, reward_weights

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    """
    Main training function using hydra configuration.
    
    You can override any config value from command line:
    python main.py training.batch_size=8 training.learning_rate=1e-5
    python main.py gpu.training.gpu_ids=0,1 gpu.training.num_gpus=2
    python main.py dataset.intermediate_tag=think dataset.final_tag=answer
    """
    
    # Initialize the configuration system
    setup_environment()
    
    # Print configuration summary
    print(f"Configuration loaded:")
    print(f"  Project: {config.project.suffix}")
    print(f"  Dataset: {config.dataset.name}")
    print(f"  Model: {config.model.name}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Intermediate tag: {config.dataset.intermediate_tag}")
    print(f"  Final tag: {config.dataset.final_tag}")
    
    # Initialize wandb logging
    init_wandb(config)
    
    # Save configuration to YAML
    save_config_to_yaml(config)
    
    # Load dataset using the new config system
    train_dataset, eval_dataset = load_dataset(
        config=config,
        dataset_name=config.dataset.name,
        val_size=config.dataset.validation_size
    )
    
    # Get training config from the new system
    training_config = get_training_config(config)
    
    # Get PEFT config from the new system
    peft_config = get_peft_config(config)
    
    # Create reward functions and weights using the factory
    reward_funcs, reward_weights = create_reward_factory(config)
    
    # Initialize trainer with custom tags from config
    trainer = CustomGRPOTrainer(
        model=config.model.name,
        args=training_config,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
        custom_tags={
            "intermediate_tag": config.dataset.intermediate_tag,
            "final_tag": config.dataset.final_tag
        }
    )
    
    # Resume from checkpoint if specified
    resume_from_checkpoint = config.model.name if config.model.resume_from_checkpoint else None
    
    # Start training
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save the final model
    trainer.save_model()

if __name__ == "__main__":
    main()