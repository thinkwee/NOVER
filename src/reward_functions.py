import re
import numpy as np
from typing import List
import torch
from .utils import extract_content, calculate_perplexity, extract_all_content, safe_wandb_log
import pandas as pd
import wandb

_ref_model = None
_ref_tokenizer = None

def set_reference_model(model, tokenizer):
    global _ref_model, _ref_tokenizer
    _ref_model = model
    _ref_tokenizer = tokenizer

def _extract_completion_text(completion):
    """
    Extract text content from completion, handling both string and message formats.
    
    Args:
        completion: Either a string or a list of messages
        
    Returns:
        str: The completion text
    """
    if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
        # Extract content from assistant message
        for message in completion:
            if message.get("role") == "assistant":
                return message.get("content", "")
        # No assistant message found
        return ""
    elif isinstance(completion, str):
        return completion
    else:
        return str(completion)

def tag_format_reward(completions: List[str], intermediate_tag: str = "think", final_tag: str = "answer", **kwargs) -> List[float]:
    """
    Reward function for proper tag formatting.
    
    Args:
        completions: List of completion texts
        intermediate_tag: Tag for intermediate reasoning
        final_tag: Tag for final answer
        **kwargs: Additional arguments
    
    Returns:
        List of reward scores
    """
    rewards = []
    
    tag_details = []
    
    for completion in completions:
        # Convert completion to string if it's in message format
        completion_text = _extract_completion_text(completion)
        
        reward = 0.0
        
        reasoning_content = extract_content(completion_text, intermediate_tag)
        answer_content = extract_content(completion_text, final_tag)
        
        expected = f"<{intermediate_tag}>{reasoning_content}</{intermediate_tag}><{final_tag}>{answer_content}</{final_tag}>"
        
        completion_no_whitespace = re.sub(r'\s+', '', completion_text)
        expected_no_whitespace = re.sub(r'\s+', '', expected)
        
        is_perfect_format = completion_no_whitespace == expected_no_whitespace
        
        if is_perfect_format:
            reward = 1.0
        
        rewards.append(float(reward))
        
        tag_details.append({
            "is_perfect_format": is_perfect_format,
            "reasoning_content": reasoning_content,
            "answer_content": answer_content,
            "reasoning_length": len(reasoning_content),
            "answer_length": len(answer_content)
        })
    
    if wandb.run is not None and "global_step" in kwargs:
        try:
            step = kwargs.get("global_step", 0)
            
            log_dict = {
                "tag_reward/mean": np.mean(rewards),
                "tag_reward/std": np.std(rewards),
                "tag_reward/max": np.max(rewards),
                "tag_reward/min": np.min(rewards),
            }
            
            # More robust parameter handling
            prompts = kwargs.get("prompts", [])
            if isinstance(prompts, list) and len(prompts) > 0:
                question = prompts[0]
            else:
                question = str(prompts) if prompts else ""
                
            if isinstance(question, str) and "Answer the question and return in the following format" in question:
                question = question.split("Answer the question and return in the following format")[0]
                
            reference = kwargs.get("reference", [])
            if isinstance(reference, list) and len(reference) > 0:
                reference_answer = reference[0]
            else:
                reference_answer = str(reference) if reference else ""
            
            table = {
                "step": [str(step)] * len(completions),
                "question": [question for _ in range(len(completions))],
                "completion": [_extract_completion_text(comp) for comp in completions],
                "reference": [reference_answer for _ in range(len(completions))],
                "tag_reward": rewards,
                "is_perfect_format": [detail["is_perfect_format"] for detail in tag_details],
                "reasoning_content": [detail["reasoning_content"] for detail in tag_details],
                "answer_content": [detail["answer_content"] for detail in tag_details],
                "reasoning_length": [detail["reasoning_length"] for detail in tag_details],
                "answer_length": [detail["answer_length"] for detail in tag_details],
            }
            
            num_generations = kwargs.get("num_generations", 1)
            if num_generations > 1 and len(rewards) % num_generations == 0:
                group_ids = []
                for i in range(len(completions)):
                    group_idx = i // num_generations
                    group_ids.append(group_idx + 1)  
                table["group_id"] = group_ids
            
            df = pd.DataFrame(table)

            safe_wandb_log({"Tag Format Rewards": wandb.Table(dataframe=df)}, step=step)
            
            safe_wandb_log(log_dict, step=step)
        except Exception as e:
            print(f"Warning: Wandb logging failed: {str(e)}")
    
    return rewards

def precompute_completion_data(completions, intermediate_tag=None, final_tag=None, **kwargs):
    """
    Precompute completion data for reward functions.

    Args:
        completions: List of completions
        intermediate_tag: The intermediate tag to use
        final_tag: The final tag to use
        **kwargs: Additional keyword arguments
    """
    global _ref_model, _ref_tokenizer
    if _ref_model is None or _ref_tokenizer is None:
        raise ValueError("Reference model or tokenizer not initialized. Call set_reference_model() first.")
    
    if intermediate_tag is None:
        intermediate_tag = "think"
    if final_tag is None:
        final_tag = "answer"
        
    # More robust parameter handling
    prompts = kwargs.get("prompts", [])
    if isinstance(prompts, list) and len(prompts) > 0:
        prompt = prompts[0]
    else:
        prompt = str(prompts) if prompts else ""
    
    reference = kwargs.get("reference", [])
    if isinstance(reference, list) and len(reference) > 0:
        reference_answer = reference[0]
    else:
        reference_answer = str(reference) if reference else ""
    
    question = prompt.split("Answer the question and return in the following format")[0] if "Answer the question and return in the following format" in prompt else prompt
    
    if not reference_answer or reference_answer == "":
        return [{"valid": False}] * len(completions), question, reference_answer, [float(0.5)] * len(completions)
    
    tag_rewards = tag_format_reward(completions, intermediate_tag=intermediate_tag, final_tag=final_tag, **kwargs)
    
    completion_data = []
    
    all_reasonings = []
    all_answers = []
    for completion in completions:
        reasoning = extract_all_content(completion, intermediate_tag)
        answer = extract_content(completion, final_tag)
        all_reasonings.append(reasoning)
        all_answers.append(answer)
    
    with torch.no_grad():
        for i, (reasoning, answer) in enumerate(zip(all_reasonings, all_answers)):
            is_perfect_tag_format = tag_rewards[i] == 1.0
            
            if not is_perfect_tag_format:
                completion_data.append({
                    "valid": False,
                    "reasoning": "",
                    "answer": "",
                    "sentences": [],
                    "length": 0,
                    "full_reasoning_ppl": float('inf'),
                    "completion": _extract_completion_text(completions[i])
                })
                continue
            
            if len(reasoning) == 0:
                completion_data.append({
                    "valid": False,
                    "reasoning": "",
                    "answer": "",
                    "sentences": [],
                    "length": 0,
                    "full_reasoning_ppl": float('inf'),
                    "completion": _extract_completion_text(completions[i])
                })
                continue
                
            reasoning = reasoning[0]
            
            reasoning_length = len(_ref_tokenizer.encode(reasoning))
            
            full_reasoning_ppl, full_reasoning_ppl_nonorm = calculate_perplexity(
                _ref_model, _ref_tokenizer, question=question, reasoning=reasoning, target=reference_answer,
                intermediate_tag=intermediate_tag, final_tag=final_tag
            )
            
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', reasoning) if s.strip()]
            
            completion_data.append({
                "valid": True,
                "reasoning": reasoning,
                "answer": answer,
                "sentences": sentences,
                "length": float(reasoning_length),
                "full_reasoning_ppl": float(full_reasoning_ppl),
                "full_reasoning_ppl_nonorm": float(full_reasoning_ppl_nonorm),
                "completion": _extract_completion_text(completions[i])
            })
    
    return completion_data, question, reference_answer, tag_rewards

def efficiency_reward(completions, **kwargs):
    """
    Reward function for efficiency.

    Args:
        completions: List of completions
        **kwargs: Additional keyword arguments
    """
    # Get tags from kwargs
    intermediate_tag = kwargs.get("intermediate_tag", "think")
    final_tag = kwargs.get("final_tag", "answer")
    
    completion_data, question, reference_answer, _ = precompute_completion_data(completions, **kwargs)
    
    rewards = [0.0] * len(completions)
    
    with torch.no_grad():
        valid_indices = []
        lengths = []
        ppls = []
        for i, data in enumerate(completion_data):
            if data["valid"]:
                valid_indices.append(i)
                lengths.append(float(data["length"]))
                ppls.append(float(data["full_reasoning_ppl"]))
        
        if len(valid_indices) < 2:
            return [float(0.0)] * len(completions)
        
        lengths = np.array(lengths)
        ppls = np.array(ppls)
        
        length_diff = lengths[:, None] < lengths[None, :]
        ppl_diff = ppls[:, None] < ppls[None, :]
        
        better_pairs = length_diff & ppl_diff
        
        np.fill_diagonal(better_pairs, False)
        
        valid_comparisons = np.ones_like(better_pairs, dtype=bool)
        np.fill_diagonal(valid_comparisons, False)
        
        valid_comparison_counts = valid_comparisons.sum(axis=1)
        
        rewards_valid = np.zeros(len(valid_indices))
        for i in range(len(valid_indices)):
            if valid_comparison_counts[i] > 0:
                rewards_valid[i] = better_pairs[i].sum() / valid_comparison_counts[i]
            else:
                rewards_valid[i] = 0.0  
        
        for i, valid_idx in enumerate(valid_indices):
            rewards[valid_idx] = float(rewards_valid[i])
    
    if wandb.run is not None and "global_step" in kwargs:
        try:
            step = kwargs.get("global_step", 0)
            
            log_dict = {
                "efficiency_reward/mean": np.mean(rewards),
                "efficiency_reward/std": np.std(rewards),
                "efficiency_reward/max": np.max(rewards),
                "efficiency_reward/min": np.min(rewards),
            }
            
            table = {
                "step": [str(step)] * len(completions),
                "question": [question for _ in range(len(completions))],
                "completion": [_extract_completion_text(comp) for comp in completions],
                "reference": [reference_answer for _ in range(len(completions))],
                "efficiency_reward": rewards,
            }
            
            reasoning_lengths = []
            reasoning_contents = []
            answer_texts = []
            valid_flags = []
            sentence_counts = []
            ppl_values = []
            better_than_count = []
            valid_comparison_count = []
            
            better_than_counts_raw = {}
            if len(valid_indices) >= 2:
                for i, valid_idx in enumerate(valid_indices):
                    better_than_counts_raw[valid_idx] = int(better_pairs[i].sum())
                    valid_comparison_count.append(int(valid_comparison_counts[i]))
            
            for i, data in enumerate(completion_data):
                valid_flags.append(data["valid"])
                
                if data["valid"]:
                    length = data.get("length", 0)
                    reasoning_lengths.append(float(length))
                    reasoning_content = data.get("reasoning", "")
                    reasoning_contents.append(reasoning_content)
                    sentence_count = len(data.get("sentences", []))
                    sentence_counts.append(sentence_count)
                    ppl_values.append(float(data.get("full_reasoning_ppl", float('inf'))))
                    better_than_count.append(better_than_counts_raw.get(i, 0))
                    if i not in valid_indices:
                        valid_comparison_count.append(0)
                else:
                    reasoning_lengths.append(float(0))
                    reasoning_contents.append("")
                    sentence_counts.append(0)
                    ppl_values.append(float('inf'))
                    better_than_count.append(0)
                    valid_comparison_count.append(0)
                
                answer = extract_content(completions[i], final_tag)
                answer_texts.append(answer)
            
            table["reasoning_length"] = reasoning_lengths
            table["reasoning_content"] = reasoning_contents
            table["answer_content"] = answer_texts
            table["answer_length"] = [len(answer) for answer in answer_texts]
            table["valid_completion"] = valid_flags
            table["sentence_count"] = sentence_counts
            table["ppl"] = ppl_values
            table["better_than_count"] = better_than_count
            table["valid_comparison_count"] = valid_comparison_count
            
            df = pd.DataFrame(table)

            safe_wandb_log({"Efficiency Rewards": wandb.Table(dataframe=df)}, step=step)
            
            safe_wandb_log(log_dict, step=step)
        except Exception as e:
            print(f"Warning: Wandb logging failed: {str(e)}")
    
    return rewards

def reasoning_reward(completions, **kwargs):
    """
    Reward function for reasoning.

    Args:
        completions: List of completions
        **kwargs: Additional keyword arguments
    """
    # Get tags from kwargs
    intermediate_tag = kwargs.get("intermediate_tag", "think")
    final_tag = kwargs.get("final_tag", "answer")
    
    completion_data, question, reference_answer, _ = precompute_completion_data(completions, **kwargs)
    
    ppl_values = []
    ppl_nonorm_values = []
    
    with torch.no_grad():
        for data in completion_data:
            if not data["valid"]:
                ppl_values.append(float('inf'))
                ppl_nonorm_values.append(float('inf'))
            else:
                ppl_values.append(float(data["full_reasoning_ppl"]))
                ppl_nonorm_values.append(float(data["full_reasoning_ppl_nonorm"]))
    
    final_rewards = [0.0] * len(completions)
    
    ranks = [0] * len(completions)
    
    idx_ppl_pairs = [(i, ppl) for i, ppl in enumerate(ppl_values)]
    
    valid_pairs = [(idx, ppl) for idx, ppl in idx_ppl_pairs if not np.isinf(ppl)]
    
    if valid_pairs:
        sorted_pairs = sorted(valid_pairs, key=lambda x: x[1])
        
        valid_count = len(valid_pairs)
        for rank, (idx, _) in enumerate(sorted_pairs):
            ranks[idx] = rank + 1
            final_rewards[idx] = float((valid_count - rank) / valid_count)
        
    else:
        final_rewards = [0.0] * len(completions)
    
    if wandb.run is not None and "global_step" in kwargs:
        try:
            step = kwargs.get("global_step", 0)
            
            log_dict = {
                "reasoning_reward/mean": np.mean(final_rewards),
                "reasoning_reward/std": np.std(final_rewards),
                "reasoning_reward/max": np.max(final_rewards),
                "reasoning_reward/min": np.min(final_rewards),
                "ppl_norm/mean": np.mean([ppl for ppl in ppl_values if not np.isinf(ppl)]) if any(not np.isinf(ppl) for ppl in ppl_values) else 0,
                "ppl_norm/min": np.min([ppl for ppl in ppl_values if not np.isinf(ppl)]) if any(not np.isinf(ppl) for ppl in ppl_values) else 0,
                "ppl_norm/max": np.max([ppl for ppl in ppl_values if not np.isinf(ppl)]) if any(not np.isinf(ppl) for ppl in ppl_values) else 0,
                "ppl_nonorm/mean": np.mean([ppl for ppl in ppl_nonorm_values if not np.isinf(ppl)]) if any(not np.isinf(ppl) for ppl in ppl_nonorm_values) else 0,
                "ppl_nonorm/min": np.min([ppl for ppl in ppl_nonorm_values if not np.isinf(ppl)]) if any(not np.isinf(ppl) for ppl in ppl_nonorm_values) else 0,
                "ppl_nonorm/max": np.max([ppl for ppl in ppl_nonorm_values if not np.isinf(ppl)]) if any(not np.isinf(ppl) for ppl in ppl_nonorm_values) else 0,
            }
            
            safe_wandb_log(log_dict, step=step)
            
            table = {
                "step": [str(step)] * len(completions),
                "question": [question for _ in range(len(completions))],
                "completion": [_extract_completion_text(comp) for comp in completions],
                "reference": [reference_answer for _ in range(len(completions))],
                "ppl_norm": ppl_values,
                "ppl_nonorm": ppl_nonorm_values,
                "ppl_rank": ranks,
                "reasoning_reward": final_rewards,
            }
            
            reasoning_lengths = []
            reasoning_contents = []
            answer_texts = []
            valid_flags = []
            sentence_counts = []
            
            for i, data in enumerate(completion_data):
                valid_flags.append(data["valid"])
                
                if data["valid"]:
                    reasoning_lengths.append(float(data["length"]))
                    reasoning_contents.append(data["reasoning"])
                    sentence_counts.append(len(data.get("sentences", [])))
                else:
                    reasoning_lengths.append(float(0))
                    reasoning_contents.append("")
                    sentence_counts.append(0)
                
                answer = extract_content(completions[i], final_tag)
                answer_texts.append(answer)
            
            table["reasoning_length"] = reasoning_lengths
            table["reasoning_content"] = reasoning_contents
            table["answer_content"] = answer_texts
            table["valid_completion"] = valid_flags
            table["sentence_count"] = sentence_counts
            
            df = pd.DataFrame(table)

            safe_wandb_log({"Reasoning Quality Details": wandb.Table(dataframe=df)}, step=step)
            
        except Exception as e:
            print(f"Warning: Wandb logging failed: {str(e)}")
    
    return final_rewards

def validation_accuracy(completions, **kwargs):
    """
    Reward function for validation accuracy.

    Args:
        completions: List of completions
        **kwargs: Additional keyword arguments
    """
    # Get tags from kwargs
    intermediate_tag = kwargs.get("intermediate_tag", "think")
    final_tag = kwargs.get("final_tag", "answer")
    
    references = kwargs.get("reference", [])
    if not references or len(references) == 0:
        print("[WARNING] No reference answers provided for validation accuracy calculation")
        return [0.0] * len(completions)
    
    import numpy as np
    
    def levenshtein_distance(s1, s2):
        if s1 == s2:
            return 0
        
        if len(s1) == 0:
            return len(s2)
        
        if len(s2) == 0:
            return len(s1)
        
        matrix = np.zeros((len(s1) + 1, len(s2) + 1))
        
        for i in range(len(s1) + 1):
            matrix[i, 0] = i
        
        for j in range(len(s2) + 1):
            matrix[0, j] = j
        
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                if s1[i-1] == s2[j-1]:
                    matrix[i, j] = matrix[i-1, j-1]
                else:
                    matrix[i, j] = min(
                        matrix[i-1, j] + 1,     
                        matrix[i, j-1] + 1,    
                        matrix[i-1, j-1] + 1    
                    )
        
        return matrix[len(s1), len(s2)]
    
    def normalize_distance(s1, s2):
        if not s1 or not s2:
            return 0.0
        
        distance = levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        
        return 1.0 - (distance / max_len)
    
    accuracy_scores = []
    
    for i, completion in enumerate(completions):
        # More robust reference handling
        if isinstance(references, list) and i < len(references):
            ref_answer = references[i]
        elif isinstance(references, list) and len(references) > 0:
            ref_answer = references[0]
        else:
            ref_answer = str(references) if references else ""
        
        answer_content = extract_content(completion, final_tag)
        
        if not answer_content:
            accuracy_scores.append(0.0)
        else:
            answer_content = answer_content.strip()
            ref_answer = ref_answer.strip()
            
            score = normalize_distance(answer_content, ref_answer)
            accuracy_scores.append(float(score))
    
    if wandb.run is not None and "global_step" in kwargs:
        try:
            step = kwargs.get("global_step", 0)
            
            log_dict = {
                "edit_distance_accuracy/mean": np.mean(accuracy_scores),
                "edit_distance_accuracy/std": np.std(accuracy_scores),
                "edit_distance_accuracy/max": np.max(accuracy_scores),
                "edit_distance_accuracy/min": np.min(accuracy_scores),
            }
            
            safe_wandb_log(log_dict, step=step)
            
            # More robust parameter handling
            prompts = kwargs.get("prompts", [])
            if isinstance(prompts, list) and len(prompts) > 0:
                question = prompts[0]
            else:
                question = str(prompts) if prompts else ""
                
            if isinstance(question, str) and "Answer the question and return in the following format" in question:
                question = question.split("Answer the question and return in the following format")[0]
            
            # Prepare reference list for table
            if isinstance(references, list):
                ref_list = references[:len(completions)]
                # Pad if needed
                while len(ref_list) < len(completions):
                    ref_list.append(ref_list[0] if ref_list else "")
            else:
                ref_list = [str(references)] * len(completions)
            
            table = {
                "step": [str(step)] * len(completions),
                "question": [question for _ in range(len(completions))],
                "completion": [_extract_completion_text(comp) for comp in completions],
                "reference": ref_list,
                "extracted_answer": [extract_content(_extract_completion_text(comp), final_tag) for comp in completions],
                "edit_distance_score": accuracy_scores,
            }
            
            df = pd.DataFrame(table)
            
            safe_wandb_log({"Edit Distance Accuracy": wandb.Table(dataframe=df)}, step=step)
            
        except Exception as e:
            print(f"Warning: Wandb logging failed: {str(e)}")
    
    return accuracy_scores
