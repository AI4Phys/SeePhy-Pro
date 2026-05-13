# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from typing import Optional

import numpy as np
import torch

from ..protocol import DataProto


def compute_variance(values: list[float]) -> float:
    """Compute variance of a list of values."""
    if len(values) < 2:
        return 0.0
    mean = np.mean(values)
    return np.mean([(x - mean) ** 2 for x in values])


def dynamic_sample_batch(
    batch: DataProto,
    tokenizer,
    target_batch_size: int,
    enable_prompt_sampling: bool = True,
    enable_answer_sampling: bool = True,
    min_variance: float = 100.0,
    max_attempts: int = 10,
) -> Optional[DataProto]:
    """
    Dynamically sample a batch to ensure variance in prompt and/or answer lengths.
    
    Args:
        batch: Input DataProto batch
        tokenizer: Tokenizer to compute answer lengths
        target_batch_size: Target batch size to sample
        enable_prompt_sampling: Whether to ensure prompt length variance
        enable_answer_sampling: Whether to ensure answer length variance
        min_variance: Minimum variance threshold
        max_attempts: Maximum resampling attempts
        
    Returns:
        Resampled DataProto or None if failed
    """
    if len(batch) < target_batch_size:
        return None
    
    # Compute prompt lengths
    prompt_lengths = []
    if enable_prompt_sampling:
        # Try to get prompt lengths from prompts field (from gen_batch_output)
        if "prompts" in batch.batch:
            prompts = batch.batch["prompts"]
            # Sum non-padding tokens to get prompt lengths
            prompt_lengths = (prompts != tokenizer.pad_token_id).sum(dim=-1).cpu().tolist()
        # Fallback: try to compute from response_mask (prompt length = total - response)
        elif "response_mask" in batch.batch and "attention_mask" in batch.batch:
            attention_mask = batch.batch["attention_mask"]
            response_mask = batch.batch["response_mask"]
            # Prompt length = total length - response length
            total_lengths = attention_mask.sum(dim=-1).cpu()
            response_lengths = response_mask.sum(dim=-1).cpu()
            prompt_lengths = (total_lengths - response_lengths).tolist()
        else:
            # Cannot compute prompt lengths, disable prompt sampling
            enable_prompt_sampling = False
    
    # Compute answer lengths from ground_truth or responses
    answer_lengths = []
    if enable_answer_sampling:
        # First try to use ground_truth if available
        if "ground_truth" in batch.non_tensor_batch:
            ground_truths = batch.non_tensor_batch["ground_truth"]
            for gt in ground_truths:
                if isinstance(gt, str):
                    # Tokenize ground truth to get length
                    tokens = tokenizer.encode(gt, add_special_tokens=False)
                    answer_lengths.append(len(tokens))
                else:
                    answer_lengths.append(0)
        # Fallback: use response lengths from response_mask
        elif "response_mask" in batch.batch:
            response_mask = batch.batch["response_mask"]
            answer_lengths = response_mask.sum(dim=-1).cpu().tolist()
        else:
            # Cannot compute answer lengths, disable answer sampling
            enable_answer_sampling = False
    
    # If we don't have enough data, return None
    if len(prompt_lengths) < target_batch_size and enable_prompt_sampling:
        return None
    if len(answer_lengths) < target_batch_size and enable_answer_sampling:
        return None
    
    # Try to sample a subset with sufficient variance
    indices = list(range(len(batch)))
    
    for attempt in range(max_attempts):
        # Randomly sample indices
        sampled_indices = random.sample(indices, min(target_batch_size, len(indices)))
        
        # Check variance requirements
        valid = True
        
        if enable_prompt_sampling:
            sampled_prompt_lengths = [prompt_lengths[i] for i in sampled_indices]
            prompt_var = compute_variance(sampled_prompt_lengths)
            if prompt_var < min_variance:
                valid = False
        
        if enable_answer_sampling and valid:
            sampled_answer_lengths = [answer_lengths[i] for i in sampled_indices]
            answer_var = compute_variance(sampled_answer_lengths)
            if answer_var < min_variance:
                valid = False
        
        if valid:
            # Return sampled batch
            return batch[sampled_indices]
    
    # If we couldn't find a good sample, try a greedy approach
    # Sort by length diversity and select diverse samples
    if enable_prompt_sampling and enable_answer_sampling:
        # Combine prompt and answer lengths for diversity
        combined_scores = [
            (prompt_lengths[i] if i < len(prompt_lengths) else 0) * 0.5 +
            (answer_lengths[i] if i < len(answer_lengths) else 0) * 0.5
            for i in range(len(batch))
        ]
    elif enable_prompt_sampling:
        combined_scores = prompt_lengths
    elif enable_answer_sampling:
        combined_scores = answer_lengths
    else:
        # No sampling enabled, just return random sample
        sampled_indices = random.sample(indices, min(target_batch_size, len(indices)))
        return batch[sampled_indices]
    
    # Greedy selection: select samples that maximize variance
    selected_indices = []
    remaining_indices = set(indices)
    
    # Start with the sample with median score
    sorted_indices = sorted(indices, key=lambda i: combined_scores[i])
    median_idx = sorted_indices[len(sorted_indices) // 2]
    selected_indices.append(median_idx)
    remaining_indices.remove(median_idx)
    
    # Greedily add samples that maximize variance
    while len(selected_indices) < target_batch_size and remaining_indices:
        best_idx = None
        best_variance = -1
        
        for candidate_idx in remaining_indices:
            # Try adding this candidate
            test_indices = selected_indices + [candidate_idx]
            
            if enable_prompt_sampling:
                test_prompt_lengths = [prompt_lengths[i] for i in test_indices]
                prompt_var = compute_variance(test_prompt_lengths)
            else:
                prompt_var = min_variance
            
            if enable_answer_sampling:
                test_answer_lengths = [answer_lengths[i] for i in test_indices]
                answer_var = compute_variance(test_answer_lengths)
            else:
                answer_var = min_variance
            
            # Use minimum variance as the score (we want both to be high)
            combined_var = min(prompt_var, answer_var) if (enable_prompt_sampling and enable_answer_sampling) else (prompt_var if enable_prompt_sampling else answer_var)
            
            if combined_var > best_variance:
                best_variance = combined_var
                best_idx = candidate_idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        else:
            # Fallback: just add random remaining indices
            selected_indices.extend(list(remaining_indices)[:target_batch_size - len(selected_indices)])
            break
    
    return batch[selected_indices[:target_batch_size]]

