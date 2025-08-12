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
from .reward_functions import tag_format_reward, reasoning_reward, efficiency_reward

# Enable progress bars explicitly
try:
    from transformers.utils import logging as transformers_logging
    transformers_logging.enable_progress_bar()
except ImportError:
    pass

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
    
    # Create reward functions with custom tags
    def tag_format_reward_wrapper(completions, **kwargs):
        return tag_format_reward(
            completions, 
            intermediate_tag=config.dataset.intermediate_tag, 
            final_tag=config.dataset.final_tag, 
            **kwargs
        )
    tag_format_reward_wrapper.__name__ = "Tag Format Reward"
    
    def reasoning_reward_wrapper(completions, **kwargs):
        return reasoning_reward(
            completions, 
            intermediate_tag=config.dataset.intermediate_tag, 
            final_tag=config.dataset.final_tag, 
            **kwargs
        )
    reasoning_reward_wrapper.__name__ = "Reasoning Reward"
    
    def efficiency_reward_wrapper(completions, **kwargs):
        return efficiency_reward(
            completions, 
            intermediate_tag=config.dataset.intermediate_tag, 
            final_tag=config.dataset.final_tag, 
            **kwargs
        )
    efficiency_reward_wrapper.__name__ = "Efficiency Reward"
    
    reward_funcs = [
        tag_format_reward_wrapper,
        reasoning_reward_wrapper,
        efficiency_reward_wrapper
    ]
    
    # Initialize trainer with custom tags from config
    trainer = CustomGRPOTrainer(
        model=config.model.name,
        args=training_config,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_funcs,
        custom_config={
            "intermediate_tag": config.dataset.intermediate_tag,
            "final_tag": config.dataset.final_tag,
            "force_chat_template": config.dataset.force_chat_template
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