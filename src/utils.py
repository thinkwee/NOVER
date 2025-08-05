import re
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
import wandb
import numpy as np
from omegaconf import DictConfig
from .config_manager import get_training_config

# Default tag values
DEFAULT_INTERMEDIATE_TAG = "think"
DEFAULT_FINAL_TAG = "answer"

def get_tags_from_config(config: DictConfig = None):
    """
    Get tag values from Hydra config or return defaults.
    
    Args:
        config: Hydra configuration object
        
    Returns:
        tuple: (intermediate_tag, final_tag)
    """
    if config is None:
        return DEFAULT_INTERMEDIATE_TAG, DEFAULT_FINAL_TAG
    
    intermediate_tag = getattr(config.dataset, 'intermediate_tag', DEFAULT_INTERMEDIATE_TAG)
    final_tag = getattr(config.dataset, 'final_tag', DEFAULT_FINAL_TAG)
    
    return intermediate_tag, final_tag

def extract_content(text: str, tag: str = None) -> str:
    """
    Extract content between tags.

    Args:
        text: The text to extract content from
        tag: The tag to extract content from
        
    Returns:
        str: The content between the tags
    """
    if tag is None:
        tag = DEFAULT_FINAL_TAG
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_all_content(text: str, tag: str = None) -> list:
    """
    Extract all content between tags.

    Args:
        text: The text to extract content from
        tag: The tag to extract content from
        
    Returns:
        list: The content between the tags
    """
    if tag is None:
        tag = DEFAULT_INTERMEDIATE_TAG
    pattern = f"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

def selective_log_softmax(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log softmax of the logits for the targets.

    Args:
        logits: The logits to calculate the log softmax of
        targets: The targets to calculate the log softmax of
        
    Returns:
        torch.Tensor: The log softmax of the logits for the targets
    """
    batch_size, seq_len, vocab_size = logits.shape
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    
    log_probs = F.log_softmax(flat_logits, dim=-1)
    target_log_probs = log_probs[torch.arange(flat_targets.size(0)), flat_targets]
    return target_log_probs.reshape(batch_size, seq_len)

def calculate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str = "",
    reasoning: str = "",
    target: str = "",
    use_natural_format: bool = False,
    use_tagged_format: bool = True,
    intermediate_tag: str = None,
    final_tag: str = None,
    config: DictConfig = None
) -> float:
    """
    Calculate the perplexity of the model's output.

    Args:
        model: The model to calculate the perplexity of
        tokenizer: The tokenizer to use for the model
        question: The question to answer
        reasoning: The reasoning to use for the answer
        target: The target answer
        use_natural_format: Whether to use the natural format
        use_tagged_format: Whether to use the tagged format
        intermediate_tag: The intermediate tag to use
        final_tag: The final tag to use
        config: The configuration to use

    Returns:
        float: The perplexity of the model's output
    """

    # Get tags from config if not provided
    if intermediate_tag is None or final_tag is None:
        config_intermediate_tag, config_final_tag = get_tags_from_config(config)
        if intermediate_tag is None:
            intermediate_tag = config_intermediate_tag
        if final_tag is None:
            final_tag = config_final_tag
        
    if use_tagged_format:
        prompt = f"""
Question: {question}

Answer the question and return in the following format:

<{intermediate_tag}>
...
</{intermediate_tag}>

<{final_tag}>
...
</{final_tag}>
"""
        if reasoning:
            prompt = prompt + f"\n<{intermediate_tag}>\n{reasoning}\n</{intermediate_tag}>\n"
            
        target = f"\n<{final_tag}>\n{target}\n</{final_tag}>"
        
        combined = prompt + target
    elif use_natural_format:
        if reasoning:
            prompt = f"For the question: {question}, I think {reasoning}, so the answer is "
        else:
            prompt = f"For the question: {question}, the answer is "
        combined = prompt + target
    else:
        prompt = question + reasoning
        combined = prompt + target

    inputs = tokenizer(combined, return_tensors="pt").to(model.device)
    input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
    
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_len = prompt_tokens.size(-1)

    reasoning_tokens = tokenizer(reasoning, return_tensors="pt").input_ids.to(model.device)
    reasoning_len = reasoning_tokens.size(-1)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        
        targets = input_ids[:, 1:]
        
        target_logits = logits[:, prompt_len-1:, :]
        target_ids = targets[:, prompt_len-1:]
        target_mask = attention_mask[:, 1:][:, prompt_len-1:]
        
        log_probs = selective_log_softmax(target_logits, target_ids)
        
        log_probs = log_probs * target_mask
        nll = -log_probs.sum() / target_mask.sum()

        norm_term = max(1, 1 + np.log10(reasoning_len))
        raw_nll = nll
        nll = nll / norm_term
        
    return torch.exp(nll).item(), torch.exp(raw_nll).item()

def safe_wandb_log(data, step=None):
    """
    Log data to Weights & Biases.

    Args:
        data: The data to log
        step: The step to log the data at
    """
    if wandb.run is None:
        return
    
    fixed_data = {}
    for key, value in data.items():
        if isinstance(key, str) and key.startswith("metrics/") and isinstance(value, str):
            fixed_key = key.replace("metrics/", "text_metrics/")
            fixed_data[fixed_key] = value
        else:
            fixed_data[key] = value
    
    if step is None:
        try:
            current_step = wandb.run.history._step
            step = current_step
        except:
            step = 0
    
    wandb.log(fixed_data)