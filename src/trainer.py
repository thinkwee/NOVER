from trl import GRPOTrainer
import torch
import wandb
from torch import nn
from accelerate.utils import gather_object
from trl.data_utils import is_conversational, maybe_apply_chat_template
from .reward_functions import set_reference_model, validation_accuracy
from typing import Union, Any, List, Optional
from .utils import safe_wandb_log
from trl.trainer.callbacks import SyncRefModelCallback
import copy
import Levenshtein
import statistics
import numpy as np
import pandas as pd
import random
from trl.trainer.callbacks import TrainerCallback
from peft import get_peft_model_state_dict, set_peft_model_state_dict
import os
import json
from datetime import datetime
from collections import deque

class SyncRefLoraModelCallback(TrainerCallback):
    """
    Callback to synchronize LoRA adapter states between reference and policy models.
    
    This callback handles the synchronization of LoRA adapter parameters between
    the reference model and policy model during training. It performs a weighted
    combination of adapter states using the ref_model_mixup_alpha parameter.
    
    The reference model is used for calculating perplexity-based reasoning rewards
    as described in the paper. This callback ensures the reference model's adapter
    state stays synchronized with the policy model's adapter state.
    
    Note: This is a custom implementation for LoRA adapters, as the original
    SyncRefModelCallback in TRL is designed for full model replacement and
    doesn't work properly with LoRA adapters. (see issue https://github.com/huggingface/trl/issues/3108)

    TODO: make a better patch and PR for it.
    
    Args:
        ref_model: The reference model to sync adapter state from
        accelerator: The accelerator instance for distributed training
        policy_model: The policy model to sync adapter state to
    """
    def __init__(self, ref_model, accelerator, policy_model):
        self.accelerator = accelerator
        self.ref_model = ref_model
        self.policy_model = policy_model
        
    def on_save(self, args, state, control, **kwargs):
        try:
            policy_adapter_state = get_peft_model_state_dict(self.policy_model)
            ref_adapter_state = get_peft_model_state_dict(self.ref_model)
            missing_keys = set(ref_adapter_state.keys()) - set(policy_adapter_state.keys())
            extra_keys = set(policy_adapter_state.keys()) - set(ref_adapter_state.keys())
            if missing_keys:
                print(f"[DEBUG] First few missing keys: {list(missing_keys)[:3]}")
            if extra_keys:
                print(f"[DEBUG] First few extra keys: {list(extra_keys)[:3]}")
            alpha = args.ref_model_mixup_alpha

            mixed_adapter_state_dict = {}
            for key in policy_adapter_state:
                mixed_adapter_state_dict[key] = alpha * ref_adapter_state[key] + (1 - alpha) * policy_adapter_state[key]

            set_peft_model_state_dict(self.ref_model, mixed_adapter_state_dict)

            print(f"[INFO] Successfully synced adapter from policy model")
            
        except Exception as e:
            print(f"[ERROR] Failed to sync adapter: {e}")
            import traceback
            print(traceback.format_exc())


class CustomGRPOTrainer(GRPOTrainer):
    """
    Custom GRPO Trainer extended from TRL GRPOTrainer with enhanced functionality.
    
    This trainer extends the base GRPOTrainer with additional features including:
    - Reference model synchronization for LoRA adapters
    - Advanced reward analysis and logging
    - Completion diversity tracking
    - Validation accuracy calculation
    - Detailed reward breakdown visualization
    
    
    Args:
        *args: Arguments passed to the parent GRPOTrainer
        custom_tags (dict, optional): Custom tags for intermediate and final outputs.
            Defaults to {"intermediate_tag": "think", "final_tag": "answer"}
        **kwargs: Keyword arguments passed to the parent GRPOTrainer
    
    Attributes:
        custom_tags (dict): Custom tags configuration
        intermediate_tag (str): Tag for intermediate outputs (default: "think")
        final_tag (str): Tag for final outputs (default: "answer")
        ref_model: Reference model for reward calculation
        diversity_history (list): History of diversity scores
        diversity_steps (list): Training steps corresponding to diversity scores
        validation_results_dir (str): Directory for saving validation results
    """
    def __init__(self, *args, custom_tags=None, **kwargs):
        self.custom_tags = custom_tags or {}
        self.intermediate_tag = self.custom_tags.get("intermediate_tag", "think")
        self.final_tag = self.custom_tags.get("final_tag", "answer")
        
        super().__init__(*args, **kwargs)

        self.remove_callback(SyncRefModelCallback)
        self._setup_reference_model()
        
        set_reference_model(self.ref_model, self.processing_class)
        
        self.current_batch = None
        
        self.diversity_history = []
        self.diversity_steps = []
        
        if self.args.sync_ref_model:
            if hasattr(self.model, 'get_adapter_state_dict'):
                self.add_callback(SyncRefLoraModelCallback(ref_model=self.ref_model, accelerator=self.accelerator, policy_model=self.model))
        
        self.validation_results_dir = os.path.join(self.args.output_dir, "validation_results")
        os.makedirs(self.validation_results_dir, exist_ok=True)
        
        # Extend the _logs dictionary with the six new fields
        self._logs.update({
            "completion_length": deque(maxlen=self.args.generation_batch_size),
            "reasoning_length": deque(maxlen=self.args.generation_batch_size),
            "answer_length": deque(maxlen=self.args.generation_batch_size),
            "full_reasoning_ppl": deque(maxlen=self.args.generation_batch_size),
            "full_reasoning_ppl_nonorm": deque(maxlen=self.args.generation_batch_size),
            "reference": deque(maxlen=self.args.generation_batch_size),
        })
        
    def _setup_reference_model(self):
        
        if not self.args.sync_ref_model:
            print(f"[INFO] No ref sync, ref = base model")
            self.ref_model = self.model.get_base_model()
            set_reference_model(self.ref_model, self.processing_class)
            return
            
        try:
            self.ref_model = copy.deepcopy(self.model)
            
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
            
            set_reference_model(self.ref_model, self.processing_class)
            
        except Exception as e:
            import traceback
            print(traceback.format_exc())

    def _calculate_completion_diversity(self, completions: List[str]) -> float:
        """
        Calculate the diversity of completions using normalized Levenshtein distance.
        
        This method measures how different the completions are from each other by
        computing the average normalized Levenshtein distance between all pairs of
        completions. For large numbers of completions, it uses random sampling to
        maintain computational efficiency.
        
        Args:
            completions (List[str]): List of completion strings to analyze
            
        Returns:
            float: Average normalized Levenshtein distance (0.0 to 1.0).
                  Higher values indicate more diverse completions.
                  0.0 is returned if there are fewer than 2 completions.
        """
        if len(completions) <= 1:
            return 0.0
        
        max_pairs = 30
        total_possible_pairs = (len(completions) * (len(completions) - 1)) // 2
        
        if total_possible_pairs > max_pairs:
            distances = []
            sampled_pairs = set()
            attempts = 0
            max_attempts = max_pairs * 3
            
            while len(sampled_pairs) < max_pairs and attempts < max_attempts:
                i = random.randint(0, len(completions) - 1)
                j = random.randint(0, len(completions) - 1)
                if i != j and (min(i, j), max(i, j)) not in sampled_pairs:
                    sampled_pairs.add((min(i, j), max(i, j)))
                    
                    distance = Levenshtein.distance(completions[i], completions[j])
                    max_len = max(len(completions[i]), len(completions[j]))
                    if max_len > 0:
                        normalized_distance = distance / max_len
                        distances.append(normalized_distance)
                attempts += 1
        else:
            distances = []
            for i in range(len(completions)):
                for j in range(i+1, len(completions)):
                    distance = Levenshtein.distance(completions[i], completions[j])
                    max_len = max(len(completions[i]), len(completions[j]))
                    if max_len > 0:
                        normalized_distance = distance / max_len
                        distances.append(normalized_distance)
                    
        return statistics.mean(distances) if distances else 0.0

    def _log_detailed_rewards_analysis(self, rewards, rewards_per_func, advantages, mean_grouped_rewards, std_grouped_rewards):
        """
        Log detailed analysis of rewards and advantages for debugging and monitoring.
        
        This method provides comprehensive logging of reward components, including:
        - Per-function reward breakdown with weights applied
        - Total rewards and advantages for each generation
        - Group statistics (mean and standard deviation)
        
        The output is formatted as a table showing each generation's performance
        across all reward functions, making it easy to identify which components
        contribute most to the overall reward.
        
        Args:
            rewards (torch.Tensor): Total rewards for each sample
            rewards_per_func (torch.Tensor): Rewards per function for each sample
            advantages (torch.Tensor): Calculated advantages for each sample
            mean_grouped_rewards (torch.Tensor): Mean rewards per group
            std_grouped_rewards (torch.Tensor): Standard deviation of rewards per group
        """
        print("\n" + "-"*50)
        print(f"Step {self.state.global_step} - Rewards & Advantages")
        print("-"*50)
        
        reward_func_names = []
        for reward_func in self.reward_funcs:
            if isinstance(reward_func, nn.Module):
                reward_func_names.append(reward_func.config._name_or_path.split("/")[-1])
            else:
                reward_func_names.append(reward_func.__name__)
        
        rewards_by_group = rewards.view(-1, self.num_generations)
        rewards_per_func_by_group = rewards_per_func.view(-1, self.num_generations, len(self.reward_funcs))
        
        for group_idx in range(rewards_by_group.shape[0]):
            print(f"Group {group_idx+1} | Mean: {mean_grouped_rewards[group_idx*self.num_generations]:.4f} | Std: {std_grouped_rewards[group_idx*self.num_generations]:.4f}")
            print("Gen | " + " | ".join([f"{name}" for name in reward_func_names]) + " | Total | Adv")
            print("-" * 50)
            
            for gen_idx in range(self.num_generations):
                global_idx = group_idx * self.num_generations + gen_idx
                
                reward_components = []
                for func_idx in range(len(self.reward_funcs)):
                    component_value = rewards_per_func[global_idx, func_idx].item()
                    if torch.isnan(torch.tensor(component_value)):
                        reward_components.append("N/A")
                    else:
                        weight = self.reward_weights[func_idx].item()
                        weighted_value = component_value * weight
                        reward_components.append(f"{weighted_value:.2f}")
                
                components_str = " | ".join(reward_components)
                print(f"{gen_idx+1:3d} | {components_str} | {rewards[global_idx]:.2f} | {advantages[global_idx]:.2f}")
            
            print("")

    def _log_diversity_analysis(self, completions_to_log, mode):
        """
        Log diversity analysis and track diversity metrics over time.
        
        This method calculates and logs completion diversity scores, maintains
        a history of diversity metrics, and optionally logs to wandb for
        visualization and tracking.
        
        Args:
            completions_to_log (List[str]): Completions to analyze for diversity
            mode (str): Training mode ("train" or "eval")
        """
        diversity_score = self._calculate_completion_diversity(completions_to_log)
        self._metrics[mode]["diversity_score"].append(diversity_score)
        
        self.diversity_history.append(diversity_score)
        self.diversity_steps.append(self.state.global_step)
        if len(self.diversity_history) > 100:
            self.diversity_history = self.diversity_history[-100:]
            self.diversity_steps = self.diversity_steps[-100:]
        
        print(f"\n[DEBUG] Step {self.state.global_step} - Advanced Metrics")
        print(f"Completion Diversity: {diversity_score:.4f}")
        print("-" * 50)

        if self.accelerator.is_main_process:
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                diversity_analysis = []
                diversity_analysis.append({
                    "step": str(self.state.global_step),
                    "metric_type": "Diversity",
                    "metric_name": "Completion Diversity",
                    "value": diversity_score,
                    "count": len(completions_to_log)
                })
                
                diversity_df = pd.DataFrame(diversity_analysis)
                safe_wandb_log({"Diversity Analysis": wandb.Table(dataframe=diversity_df)}, 
                              step=self.state.global_step)

    def _calculate_validation_accuracy(self, inputs, prompts, completions, reference_text, mode):
        """
        Calculate validation accuracy using edit distance similarity.
        
        This method computes accuracy scores by comparing generated completions
        against reference texts using edit distance similarity. It supports both
        single generation and multiple generation scenarios.
        
        Args:
            inputs (list): Input data containing prompts and other metadata
            prompts (list): Processed prompts
            completions (list): Generated completions
            reference_text (list): Reference texts for comparison
            mode (str): Training mode ("train" or "eval")
            
        Returns:
            tuple: (accuracy_scores, overall_accuracy, overall_max_similarity_at_k)
                - accuracy_scores: List of individual accuracy scores
                - overall_accuracy: Average accuracy across all samples
                - overall_max_similarity_at_k: Max similarity when using multiple generations
        """
        try:
            print(f"\n[INFO] Running validation accuracy calculation at step {self.state.global_step}")
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
            val_kwargs = {key: [example[key] for example in inputs] for key in keys}
            
            val_kwargs["global_step"] = self.state.global_step
            
            if 'reference' in val_kwargs:
                del val_kwargs['reference']
            
            accuracy_scores = validation_accuracy(
                prompts=prompts,
                completions=completions,
                reference=reference_text,
                **val_kwargs
            )
            
            valid_scores = [score for score in accuracy_scores if score is not None]
            overall_accuracy = sum(valid_scores) / max(1, len(valid_scores))
            
            self._metrics[mode]["validation/edit_distance_similarity"].append(overall_accuracy)
            print(f"[INFO] Validation edit distance similarity: {overall_accuracy:.4f}")
            
            if self.num_generations > 1:
                grouped_scores = np.array(accuracy_scores).reshape(-1, self.num_generations)
                
                max_similarity_at_k = [float(max(group)) for group in grouped_scores]
                
                overall_max_similarity_at_k = sum(max_similarity_at_k) / max(1, len(max_similarity_at_k))
                
                self._metrics[mode][f"validation/max_similarity@{self.num_generations}"].append(overall_max_similarity_at_k)
                print(f"[INFO] Validation max similarity@{self.num_generations}: {overall_max_similarity_at_k:.4f}")
            
            return accuracy_scores, overall_accuracy, overall_max_similarity_at_k if self.num_generations > 1 else None
            
        except Exception as e:
            print(f"[ERROR] Error calculating validation accuracy: {e}")
            import traceback
            print(traceback.format_exc())
            return None, None, None

    def _save_validation_results(self, prompts_text, completions_text, reference_text, accuracy_scores, rewards, rewards_per_func, overall_accuracy, overall_max_similarity_at_k):
        """
        Save comprehensive validation results to JSON files.
        
        This method creates detailed validation reports including:
        - Individual sample results with prompts, completions, and references
        - Per-function reward breakdowns
        - Overall accuracy metrics
        - Max similarity metrics for multiple generation scenarios
        
        The results are saved to the validation_results_dir with timestamps
        and step information for easy tracking and analysis.
        
        Args:
            prompts_text (list): Text prompts
            completions_text (list): Generated completion texts
            reference_text (list): Reference texts
            accuracy_scores (list): Individual accuracy scores
            rewards (torch.Tensor): Total rewards for each sample
            rewards_per_func (torch.Tensor): Rewards per function
            overall_accuracy (float): Overall accuracy score
            overall_max_similarity_at_k (float, optional): Max similarity for multiple generations
        """
        if not self.accelerator.is_main_process:
            return
            
        prompts_to_save = gather_object(prompts_text)
        completions_to_save = gather_object(completions_text)
        reference_to_save = gather_object(reference_text)
        accuracy_scores_to_save = gather_object(accuracy_scores)
        rewards_to_save = rewards.tolist()
        
        reward_func_names = []
        reward_raw_values = {}
        
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            
            reward_func_names.append(reward_func_name)
            reward_raw_values[reward_func_name] = rewards_per_func[:, i].tolist()
        
        validation_results = []
        for i, (prompt, completion, reference, accuracy) in enumerate(
            zip(prompts_to_save, completions_to_save, reference_to_save, accuracy_scores_to_save)
        ):
            result = {
                "id": i,
                "prompt": prompt,
                "completion": completion,
                "reference": reference,
                "edit_distance_similarity": float(accuracy) if accuracy is not None else None,
                "total_reward": rewards_to_save[i] if i < len(rewards_to_save) else None,
                "rewards": {}
            }
            
            for reward_name in reward_func_names:
                if i < len(reward_raw_values[reward_name]):
                    raw_value = reward_raw_values[reward_name][i]
                    result["rewards"][reward_name] = float(raw_value) if not torch.isnan(torch.tensor(raw_value)) else None
            
            validation_results.append(result)
        
        results_object = {
            "step": self.state.global_step,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "overall_edit_distance_similarity": overall_accuracy,
                "mean_reward": rewards.mean().item() if len(rewards) > 0 else None
            },
            "reward_functions": reward_func_names,
            "samples": validation_results
        }
        
        if self.num_generations > 1 and overall_max_similarity_at_k is not None:
            results_object["metrics"][f"max_similarity@{self.num_generations}"] = overall_max_similarity_at_k
        
        output_file = os.path.join(
            self.validation_results_dir, 
            f"validation_results_step_{self.state.global_step}.json"
        )
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_object, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Validation results saved to: {output_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save validation results: {e}")

    def _apply_custom_hooks(self, inputs, prompts, completions, completions_text, reference_text, prompts_text,
                           rewards, rewards_per_func, advantages, mean_grouped_rewards, std_grouped_rewards, mode):
        """
        Apply custom hooks for enhanced logging and analysis.
        
        This method orchestrates various custom analysis and logging functions:
        - Validation accuracy calculation (for eval mode with references)
        - Detailed reward analysis logging
        - Diversity analysis and tracking
        
        The hooks are applied based on the current training state and mode,
        providing comprehensive monitoring and debugging capabilities.
        
        Args:
            inputs (list): Input data
            prompts (list): Processed prompts
            completions (list): Generated completions
            completions_text (list): Decoded completion texts
            reference_text (list): Reference texts
            prompts_text (list): Decoded prompt texts
            rewards (torch.Tensor): Total rewards
            rewards_per_func (torch.Tensor): Rewards per function
            advantages (torch.Tensor): Calculated advantages
            mean_grouped_rewards (torch.Tensor): Mean rewards per group
            std_grouped_rewards (torch.Tensor): Standard deviation of rewards per group
            mode (str): Training mode ("train" or "eval")
        """
        
        # calculate validation accuracy
        if mode == "eval" and reference_text:
            accuracy_scores, overall_accuracy, overall_max_similarity_at_k = self._calculate_validation_accuracy(
                inputs, prompts, completions, reference_text, mode
            )
            self._save_validation_results(
                prompts_text, completions_text, reference_text, accuracy_scores, rewards, rewards_per_func,
                overall_accuracy, overall_max_similarity_at_k
            )

        # log detailed rewards analysis
        if self.accelerator.is_main_process and self.state.global_step % self.args.logging_steps == 0:
            self._log_detailed_rewards_analysis(rewards, rewards_per_func, advantages, mean_grouped_rewards, std_grouped_rewards)

        # log diversity analysis
        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            completions_to_log = gather_object(completions_text)
            self._log_diversity_analysis(completions_to_log, mode)

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        
        """
        Override to add custom hooks and populate additional fields for wandb summary table.
        """
        # Call the parent method to get the standard result and populate basic _logs
        result = super()._generate_and_score_completions(inputs)
        
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        reference_text = [maybe_apply_chat_template(example, self.processing_class)["reference"] for example in inputs]
        
        completion_ids = result["completion_ids"]
        
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        
        # Get rewards and advantages from the result (already calculated by parent method)
        # The parent method handles the reward calculation and _logs population
        
        # Apply our custom hooks for additional analysis
        if hasattr(self, '_logs') and len(self._logs["rewards"]) > 0:
            # Get the latest rewards and advantages from _logs
            batch_size = len(prompts)
            rewards_per_func = torch.zeros(batch_size, len(self.reward_funcs), device=device)
            
            for i, name in enumerate(self.reward_func_names):
                if name in self._logs["rewards"]:
                    reward_data = list(self._logs["rewards"][name])
                    if len(reward_data) >= batch_size:
                        latest_rewards = reward_data[-batch_size:]
                        rewards_per_func[:, i] = torch.tensor(latest_rewards, device=device)
            
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards:
                advantages = advantages / (std_grouped_rewards + 1e-4)
            
            # Apply custom hooks
            self._apply_custom_hooks(
                inputs, prompts, completions, completions_text, reference_text, prompts_text,
                rewards, rewards_per_func, advantages, mean_grouped_rewards, std_grouped_rewards, mode
            )
        
        # Populate the new fields for wandb summary table
        self._populate_summary_fields(inputs, completions_text, reference_text)
        
        return result

    def _populate_summary_fields(self, inputs, completions_text, reference_text):
        """
        Populate the six new fields for the wandb summary table:
        - completion_length: token count of the full completion
        - reasoning_length: token count of the reasoning part  
        - answer_length: token count of the answer part
        - full_reasoning_ppl: perplexity of the full reasoning
        - full_reasoning_ppl_nonorm: non-normalized perplexity of the full reasoning
        - reference: reference text for comparison
        
        Args:
            inputs: Input data
            completions_text: List of completion texts
            reference_text: List of reference texts
        """
        
        from .utils import extract_content, calculate_perplexity
        from accelerate.utils import gather_object
        
        # Gather completions and references across all processes
        all_completions = gather_object(completions_text)
        all_references = gather_object(reference_text)
        all_inputs = gather_object(inputs)
        
        completion_lengths = []
        reasoning_lengths = []
        answer_lengths = []
        full_reasoning_ppls = []
        full_reasoning_ppls_nonorm = []
        references = []
        
        for i, completion in enumerate(all_completions):
            # Calculate completion length (token count)
            completion_tokens = self.processing_class.encode(completion, add_special_tokens=False)
            completion_length = len(completion_tokens)
            
            # Extract reasoning and answer content
            reasoning_content = extract_content(completion, self.intermediate_tag)
            answer_content = extract_content(completion, self.final_tag)
            
            # Calculate reasoning and answer lengths (token counts)
            reasoning_length = len(self.processing_class.encode(reasoning_content, add_special_tokens=False)) if reasoning_content else 0
            answer_length = len(self.processing_class.encode(answer_content, add_special_tokens=False)) if answer_content else 0
            
            # Calculate perplexity for reasoning (if we have reference model and reasoning content)
            full_reasoning_ppl = float('inf')
            full_reasoning_ppl_nonorm = float('inf')
            
            if reasoning_content and hasattr(self, 'ref_model') and self.ref_model is not None:
                try:
                    # Get the question/prompt for perplexity calculation
                    question = ""
                    if i < len(all_inputs) and "prompt" in all_inputs[i]:
                        question = all_inputs[i]["prompt"]
                    elif i < len(all_references):
                        # Extract question from reference if available
                        question = str(all_references[i]) if all_references[i] else ""
                    
                    # Get reference answer
                    reference_answer = ""
                    if i < len(all_references):
                        reference_answer = str(all_references[i]) if all_references[i] else ""
                    
                    full_reasoning_ppl, full_reasoning_ppl_nonorm = calculate_perplexity(
                        model=self.ref_model,
                        tokenizer=self.processing_class,
                        question=question,
                        reasoning=reasoning_content,
                        target=reference_answer,
                        intermediate_tag=self.intermediate_tag,
                        final_tag=self.final_tag
                    )
                except Exception as e:
                    print(f"[WARNING] Failed to calculate perplexity for completion {i}: {e}")
                    full_reasoning_ppl = float('inf')
                    full_reasoning_ppl_nonorm = float('inf')
            
            # Get reference text
            reference = ""
            if i < len(all_references):
                reference = str(all_references[i]) if all_references[i] else ""
            
            # Collect all values
            completion_lengths.append(completion_length)
            reasoning_lengths.append(reasoning_length)
            answer_lengths.append(answer_length)
            full_reasoning_ppls.append(float(full_reasoning_ppl))
            full_reasoning_ppls_nonorm.append(float(full_reasoning_ppl_nonorm))
            references.append(reference)
        
        # Extend the _logs with new fields
        self._logs["completion_length"].extend(completion_lengths)
        self._logs["reasoning_length"].extend(reasoning_lengths)
        self._logs["answer_length"].extend(answer_lengths)
        self._logs["full_reasoning_ppl"].extend(full_reasoning_ppls)
        self._logs["full_reasoning_ppl_nonorm"].extend(full_reasoning_ppls_nonorm)
        self._logs["reference"].extend(references)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Override the parent log method to include our custom fields in the wandb completions table.
        We call the parent's log method for most functionality, but override the wandb table creation.
        """
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        
        # Call parent's log method but temporarily disable wandb logging to avoid duplicate tables
        original_report_to = self.args.report_to
        self.args.report_to = []  # Temporarily disable wandb reporting
        
        super().log(logs, start_time)
        
        # Restore original report_to
        self.args.report_to = original_report_to
        
        self._metrics[mode].clear()

        # Handle wandb logging with our extended table
        if self.accelerator.is_main_process and self.log_completions:
            # Print to console using rich if available
            try:
                from trl.trainer.utils import print_prompt_completions_sample
                from rich import is_rich_available
                
                if is_rich_available():
                    print_prompt_completions_sample(
                        self._logs["prompt"],
                        self._logs["completion"],
                        self._logs["rewards"],
                        self._logs["advantages"],
                        self.state.global_step,
                        self.num_completions_to_print,
                    )
            except ImportError:
                pass

            # Log to wandb with our extended table (only if wandb is enabled)
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                # Create the extended table with all fields
                table = {
                    "step": [str(self.state.global_step)] * len(self._logs["prompt"]),
                    "prompt": list(self._logs["prompt"]),
                    "completion": list(self._logs["completion"]),
                    **{k: list(v) for k, v in self._logs["rewards"].items()},
                    "advantage": list(self._logs["advantages"]),
                    # Add our six new fields
                    "completion_length": list(self._logs["completion_length"]),
                    "reasoning_length": list(self._logs["reasoning_length"]), 
                    "answer_length": list(self._logs["answer_length"]),
                    "full_reasoning_ppl": list(self._logs["full_reasoning_ppl"]),
                    "full_reasoning_ppl_nonorm": list(self._logs["full_reasoning_ppl_nonorm"]),
                    "reference": list(self._logs["reference"]),
                }

                if self._logs["image"]:
                    table["image"] = []
                    for img in self._logs["image"]:
                        if img is not None:
                            # Convert images to wandb Image objects for proper visualization
                            table["image"].append(wandb.Image(img))
                        else:
                            table["image"].append(None)

                df = pd.DataFrame(table)
                
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                
                # Log the extended completions table to wandb
                wandb.log({"completions": wandb.Table(dataframe=df)})
                
                # Also log the metrics that the parent would have logged
                wandb.log(logs)