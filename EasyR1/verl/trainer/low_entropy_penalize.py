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
"""
Functions for penalizing low entropy tokens to reduce repetitive generation.
"""

from typing import Optional

import numpy as np
import torch

from ..utils import torch_functional as VF


def penalize_low_entropy_importance(
    log_importance_ratio: torch.Tensor,
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    entropy_threshold_ratio: float = 0.25,
    min_consecutive_length: int = 10,
    penalty_factor: float = 0.5,
    **kwargs,
) -> torch.Tensor:
    """Penalize importance score for continuous low entropy tokens (likely repetitive tokens).
    
    This reduces the importance score for low entropy tokens, which will result in smaller
    ratio values, effectively reducing the weighted advantage for these tokens.
    
    Args:
        log_importance_ratio: `(torch.Tensor)`
            shape: (bs, response_length), log importance ratio to be penalized
        log_probs: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        entropy_threshold_ratio: (float)
            Threshold ratio relative to mean entropy. Tokens with entropy below 
            mean_entropy * entropy_threshold_ratio are considered low entropy.
            Default: 0.25
        min_consecutive_length: (int)
            Minimum consecutive length of low entropy tokens to be penalized.
            Default: 10
        penalty_factor: (float)
            Factor to reduce importance score for penalized tokens.
            log_importance_ratio_penalized = log_importance_ratio - penalty_factor * abs(log_importance_ratio)
            This will make exp(log_importance_ratio) smaller, reducing the weighted advantage.
            Default: 0.5
    
    Returns:
        log_importance_ratio: `(torch.Tensor)`
            shape: (bs, response_length), with penalized importance scores for low entropy tokens
    """
    # Compute entropy (negative log prob)
    entropy = -log_probs  # Higher entropy = more uncertainty = better
    
    # Compute mean entropy per sequence (only for valid tokens)
    mean_entropy = VF.masked_mean(entropy, response_mask, dim=-1, eps=1e-8)  # (bs,)
    entropy_threshold = mean_entropy.unsqueeze(-1) * entropy_threshold_ratio  # (bs, response_length)
    
    # Identify low entropy tokens
    is_low_entropy = (entropy < entropy_threshold) & response_mask  # (bs, response_length)
    
    # Detect consecutive low entropy tokens
    penalized_mask = torch.zeros_like(is_low_entropy, dtype=torch.bool)
    
    for i in range(log_importance_ratio.shape[0]):
        seq_low_entropy = is_low_entropy[i].cpu().detach().numpy()
        
        # Find consecutive runs of low entropy tokens
        if seq_low_entropy.sum() > 0:
            # Use numpy to find consecutive runs
            diff = np.diff(np.concatenate(([False], seq_low_entropy, [False])).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            for start, end in zip(starts, ends):
                if end - start >= min_consecutive_length:
                    # Mark this consecutive region for penalty
                    penalized_mask[i, start:end] = True
    
    # Apply penalty: reduce importance score for penalized tokens
    # By subtracting a penalty from log_importance_ratio, we make exp(log_importance_ratio) smaller
    # This effectively reduces the weighted advantage: advantage * ratio
    penalty = torch.where(
        penalized_mask,
        penalty_factor * torch.abs(log_importance_ratio),
        0.0
    )
    log_importance_ratio_penalized = log_importance_ratio - penalty
    
    return log_importance_ratio_penalized


def compute_low_entropy_penalty_metrics(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    entropy_threshold_ratio: float = 0.25,
    min_consecutive_length: int = 10,
) -> dict[str, float]:
    """Compute metrics for low entropy penalty.
    
    Args:
        log_probs: `(torch.Tensor)` shape: (bs, response_length)
        response_mask: `(torch.Tensor)` shape: (bs, response_length)
        entropy_threshold_ratio: Threshold ratio for low entropy detection
        min_consecutive_length: Minimum consecutive length to be penalized
        
    Returns:
        Dictionary with penalty metrics
    """
    entropy = -log_probs
    mean_entropy = VF.masked_mean(entropy, response_mask, dim=-1, eps=1e-8)
    entropy_threshold = mean_entropy.unsqueeze(-1) * entropy_threshold_ratio
    is_low_entropy = (entropy < entropy_threshold) & response_mask
    
    penalized_count = 0
    total_low_entropy = is_low_entropy.sum().item()
    
    for i in range(log_probs.shape[0]):
        seq_low_entropy = is_low_entropy[i].cpu().detach().numpy()
        if seq_low_entropy.sum() > 0:
            diff = np.diff(np.concatenate(([False], seq_low_entropy, [False])).astype(int))
            starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
            penalized_count += sum(end - start for start, end in zip(starts, ends) if end - start >= min_consecutive_length)
    
    return {
        "penalty_ratio": penalized_count / total_low_entropy if total_low_entropy > 0 else 0.0,
    }


def compute_window_penalty_metrics(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    window_size: int = 64,
    window_stride: int = 32,
    min_consecutive_windows: int = 3,
) -> dict[str, float]:
    """Compute metrics for window-based low entropy penalty.
    
    Args:
        log_probs: `(torch.Tensor)` shape: (bs, response_length)
        response_mask: `(torch.Tensor)` shape: (bs, response_length)
        window_size: Size of sliding window
        window_stride: Stride between windows
        min_consecutive_windows: Minimum consecutive low entropy windows to trigger penalty
        
    Returns:
        Dictionary with penalty metrics
    """
    entropy = -log_probs
    mean_entropy = VF.masked_mean(entropy, response_mask, dim=-1, eps=1e-8)
    entropy_threshold = mean_entropy.unsqueeze(-1) * 0.5  # Use 0.5 ratio for window method
    
    penalized_count = 0
    total_tokens = response_mask.sum().item()
    
    for i in range(log_probs.shape[0]):
        seq_entropy = entropy[i].cpu().detach().numpy()
        seq_mask = response_mask[i].cpu().detach().numpy()
        seq_threshold = entropy_threshold[i].cpu().detach().numpy()
        
        seq_length = int(seq_mask.sum())
        if seq_length == 0:
            continue
        
        window_entropies = []
        for start in range(0, seq_length, window_stride):
            end = min(start + window_size, seq_length)
            if end - start < window_size // 2:  # Skip incomplete windows at the end
                break
            window_entropy = np.mean(seq_entropy[start:end])
            window_entropies.append(window_entropy)
        
        if len(window_entropies) < min_consecutive_windows:
            continue
        
        # Check for consecutive low entropy windows
        low_entropy_windows = np.array(window_entropies) < np.mean(seq_entropy[:seq_length]) * 0.5
        for j in range(len(low_entropy_windows) - min_consecutive_windows + 1):
            if np.all(low_entropy_windows[j:j + min_consecutive_windows]):
                # Count tokens in these windows
                start_token = j * window_stride
                end_token = min((j + min_consecutive_windows) * window_stride + window_size, seq_length)
                penalized_count += end_token - start_token
                break
    
    return {
        "penalized_token_count": penalized_count,
        "total_token_count": total_tokens,
        "penalty_ratio": penalized_count / total_tokens if total_tokens > 0 else 0.0,
    }


def penalize_low_entropy_importance_window(
    log_importance_ratio: torch.Tensor,
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    window_size: int = 64,
    window_stride: int = 64,
    min_consecutive_windows: int = 2,
    penalty_factor: float = 0.5,
    **kwargs,
) -> torch.Tensor:
    """Adaptively penalize importance score using sliding window approach.
    
    Algorithm:
    1. Use sliding window of size k with stride s to compute average token entropy for each window
    2. If n consecutive windows all have avg entropy < 0.5 * (avg of all previous windows),
       mark these windows and subsequent windows (that are below previous avg) as "low entropy windows"
    3. Within these low entropy windows, penalize tokens with entropy < avg entropy of low entropy windows
    
    Args:
        log_importance_ratio: `(torch.Tensor)`
            shape: (bs, response_length), log importance ratio to be penalized
        log_probs: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        window_size: (int)
            Size of sliding window. Default: 64
        window_stride: (int)
            Stride between windows. Default: 64 (non-overlapping windows)
        min_consecutive_windows: (int)
            Minimum number of consecutive low entropy windows to trigger penalty. Default: 2
        penalty_factor: (float)
            Factor to reduce importance score for penalized tokens.
            Default: 0.5
    
    Returns:
        log_importance_ratio: `(torch.Tensor)`
            shape: (bs, response_length), with penalized importance scores
    """
    # Compute entropy (negative log prob)
    entropy = -log_probs  # Higher entropy = more uncertainty = better
    
    penalized_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    
    for i in range(log_importance_ratio.shape[0]):
        seq_entropy = entropy[i].cpu().detach().numpy()
        seq_mask = response_mask[i].cpu().detach().numpy()
        
        valid_indices = np.where(seq_mask)[0]
        if len(valid_indices) == 0:
            continue
        
        valid_entropy = seq_entropy[valid_indices]
        seq_length = len(valid_indices)
        
        if seq_length < window_size:
            continue
        
        window_entropies = []
        window_starts = []
        
        # Compute window entropies
        for start in range(0, seq_length, window_stride):
            end = min(start + window_size, seq_length)
            if end - start < window_size // 2:
                break
            window_entropy = np.mean(valid_entropy[start:end])
            window_entropies.append(window_entropy)
            window_starts.append(start)
        
        if len(window_entropies) < min_consecutive_windows:
            continue
        
        # Track average entropy of all previous windows
        previous_avg_entropy = None
        low_entropy_window_start = None
        
        for j, window_entropy in enumerate(window_entropies):
            if previous_avg_entropy is None:
                previous_avg_entropy = window_entropy
                continue
            
            # Check if current window is low entropy (below half of previous average)
            if window_entropy < previous_avg_entropy * 0.5:
                if low_entropy_window_start is None:
                    # Check if we have n consecutive low entropy windows
                    if j >= min_consecutive_windows - 1:
                        # Check previous n-1 windows
                        prev_windows = window_entropies[max(0, j - min_consecutive_windows + 1):j]
                        if len(prev_windows) == min_consecutive_windows - 1:
                            prev_avg = np.mean(prev_windows)
                            if all(w < prev_avg * 0.5 for w in prev_windows):
                                low_entropy_window_start = j - min_consecutive_windows + 1
                
                if low_entropy_window_start is not None:
                    # Mark tokens in this window
                    window_start_idx = window_starts[j]
                    window_end_idx = min(window_starts[j] + window_size, seq_length)
                    
                    # Compute average entropy of low entropy windows
                    low_entropy_window_avg = np.mean(window_entropies[low_entropy_window_start:j+1])
                    
                    # Mark tokens with entropy below low entropy window average
                    for token_idx in range(window_start_idx, window_end_idx):
                        if valid_entropy[token_idx] < low_entropy_window_avg:
                            original_idx = valid_indices[token_idx]
                            penalized_mask[i, original_idx] = True
            else:
                # Reset if we encounter a high entropy window
                low_entropy_window_start = None
            
            # Update previous average
            previous_avg_entropy = np.mean(window_entropies[:j+1])
    
    # Apply penalty: reduce importance score for penalized tokens
    penalty = torch.where(
        penalized_mask,
        penalty_factor * torch.abs(log_importance_ratio),
        0.0
    )
    log_importance_ratio_penalized = log_importance_ratio - penalty
    
    return log_importance_ratio_penalized


def encourage_entropy_for_selected_tokens(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    entropy_threshold_ratio: float = 0.25,
    selection_ratio: float = 0.3,
    random_seed: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    """Encourage higher entropy for randomly selected low entropy tokens.
    
    Algorithm:
    1. Identify low entropy tokens (entropy < mean_entropy * threshold_ratio)
    2. Randomly select a proportion (selection_ratio) of these low entropy tokens
    3. Return an entropy encouragement term (without weight) that will be multiplied
       by weight externally to push these tokens towards higher entropy (lower log_prob)
    
    Args:
        log_probs: `(torch.Tensor)`
            shape: (bs, response_length), current log probabilities
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length), mask for valid tokens
        entropy_threshold_ratio: (float)
            Threshold ratio relative to mean entropy. Tokens with entropy below 
            mean_entropy * entropy_threshold_ratio are considered low entropy.
            Default: 0.25
        selection_ratio: (float)
            Proportion of low entropy tokens to randomly select for entropy encouragement.
            Range: [0.0, 1.0]. Default: 0.3 (30%)
        random_seed: (Optional[int])
            Random seed for reproducibility. If None, uses current time.
    
    Returns:
        entropy_encouragement: `(torch.Tensor)`
            shape: (bs, response_length), unweighted encouragement term.
            Returns log_probs for selected tokens, 0.0 for others.
            Should be multiplied by weight externally before adding to loss.
    """
    # Compute entropy (negative log prob)
    entropy = -log_probs  # Higher entropy = more uncertainty = better
    
    # Compute mean entropy per sequence (only for valid tokens)
    mean_entropy = VF.masked_mean(entropy, response_mask, dim=-1, eps=1e-8)  # (bs,)
    entropy_threshold = mean_entropy.unsqueeze(-1) * entropy_threshold_ratio  # (bs, response_length)
    
    # Identify low entropy tokens
    is_low_entropy = (entropy < entropy_threshold) & response_mask  # (bs, response_length)
    
    # Randomly select a proportion of low entropy tokens
    selected_mask = torch.zeros_like(is_low_entropy, dtype=torch.bool)
    
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    for i in range(log_probs.shape[0]):
        seq_low_entropy = is_low_entropy[i].cpu().detach().numpy()
        low_entropy_indices = np.where(seq_low_entropy)[0]
        
        if len(low_entropy_indices) > 0:
            # Calculate number of tokens to select
            num_to_select = max(1, int(len(low_entropy_indices) * selection_ratio))
            # Randomly select indices
            selected_indices = np.random.choice(
                low_entropy_indices, 
                size=min(num_to_select, len(low_entropy_indices)),
                replace=False
            )
            selected_mask[i, selected_indices] = True
    
    # Create encouragement term: encourage higher entropy (lower log_prob)
    # Return unweighted log_probs for selected tokens (weight will be applied externally)
    # Use detach to avoid double gradient
    entropy_encouragement = torch.where(
        selected_mask,
        log_probs.detach(),  # Return unweighted log_probs
        0.0
    )
    
    return entropy_encouragement


def flip_advantages_for_selected_low_entropy_tokens(
    advantages: torch.Tensor,
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    entropy_threshold_ratio: float = 0.25,
    selection_ratio: float = 0.3,
    random_seed: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    """Penalize positive advantages for randomly selected low entropy tokens.
    
    Algorithm:
    1. Compute token-level entropy from log_probs (entropy = -log_probs)
    2. Identify low entropy tokens (entropy < mean_entropy * threshold_ratio)
    3. Randomly select a proportion (selection_ratio) of these low entropy tokens
    4. Penalize positive advantages of selected tokens by flipping them to negative
       (only positive advantages are penalized, negative ones remain unchanged)
       This injects noise and encourages the model to change these tokens
    
    Args:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length), advantage values
        log_probs: `(torch.Tensor)`
            shape: (bs, response_length), current log probabilities
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length), mask for valid tokens
        entropy_threshold_ratio: (float)
            Threshold ratio relative to mean entropy. Tokens with entropy below 
            mean_entropy * entropy_threshold_ratio are considered low entropy.
            Default: 0.25
        selection_ratio: (float)
            Proportion of low entropy tokens to randomly select for advantage penalization.
            Range: [0.0, 1.0]. Default: 0.3 (30%)
        random_seed: (Optional[int])
            Random seed for reproducibility. If None, uses current time.
    
    Returns:
        modified_advantages: `(torch.Tensor)`
            shape: (bs, response_length), advantages with selected low entropy tokens' 
            positive advantages flipped to negative (negative advantages unchanged)
    """
    # Compute entropy (negative log prob)
    entropy = -log_probs  # Higher entropy = more uncertainty = better
    
    # Compute mean entropy per sequence (only for valid tokens)
    mean_entropy = VF.masked_mean(entropy, response_mask, dim=-1, eps=1e-8)  # (bs,)
    entropy_threshold = mean_entropy.unsqueeze(-1) * entropy_threshold_ratio  # (bs, response_length)
    
    # Identify low entropy tokens
    is_low_entropy = (entropy < entropy_threshold) & response_mask  # (bs, response_length)
    
    # Randomly select a proportion of low entropy tokens
    selected_mask = torch.zeros_like(is_low_entropy, dtype=torch.bool)
    
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    for i in range(log_probs.shape[0]):
        seq_low_entropy = is_low_entropy[i].cpu().detach().numpy()
        low_entropy_indices = np.where(seq_low_entropy)[0]
        
        if len(low_entropy_indices) > 0:
            # Calculate number of tokens to select
            num_to_select = max(1, int(len(low_entropy_indices) * selection_ratio))
            # Randomly select indices
            selected_indices = np.random.choice(
                low_entropy_indices, 
                size=min(num_to_select, len(low_entropy_indices)),
                replace=False
            )
            selected_mask[i, selected_indices] = True
    
    # Penalize advantages for selected tokens: only flip positive advantages to negative
    # This injects noise and encourages the model to change these low entropy tokens
    # Only penalize positive advantages, keep negative ones unchanged
    modified_advantages = torch.where(
        selected_mask & (advantages > 0),
        -advantages,  # Flip positive advantages to negative
        advantages    # Keep original (including negative advantages)
    )
    
    return modified_advantages


def compute_entropy_encouragement_metrics(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    entropy_threshold_ratio: float = 0.25,
    selection_ratio: float = 0.3,
) -> dict[str, float]:
    """Compute metrics for entropy encouragement.
    
    Args:
        log_probs: `(torch.Tensor)` shape: (bs, response_length)
        response_mask: `(torch.Tensor)` shape: (bs, response_length)
        entropy_threshold_ratio: Threshold ratio for low entropy detection
        selection_ratio: Proportion of low entropy tokens selected
        
    Returns:
        Dictionary with encouragement metrics
    """
    entropy = -log_probs
    mean_entropy = VF.masked_mean(entropy, response_mask, dim=-1, eps=1e-8)
    entropy_threshold = mean_entropy.unsqueeze(-1) * entropy_threshold_ratio
    is_low_entropy = (entropy < entropy_threshold) & response_mask
    
    total_low_entropy = is_low_entropy.sum().item()
    total_tokens = response_mask.sum().item()
    
    # Estimate selected tokens (approximate, since random selection)
    estimated_selected = int(total_low_entropy * selection_ratio) if total_low_entropy > 0 else 0
    
    return {
        "low_entropy_count": total_low_entropy,
        "low_entropy_ratio": total_low_entropy / total_tokens if total_tokens > 0 else 0.0,
        "estimated_selected_count": estimated_selected,
        "selection_ratio": selection_ratio,
    }
