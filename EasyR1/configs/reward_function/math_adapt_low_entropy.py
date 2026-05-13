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

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from mathruler.grader import extract_boxed_content, grade_answer


def estimate_token_entropy_from_text(text: str) -> List[float]:
    """Estimate token-level entropy from text using word frequency and repetition.
    
    This estimates entropy based on:
    1. Word frequency in the response (repeated words = lower entropy)
    2. Local context diversity (repetitive patterns = lower entropy)
    
    Args:
        text: Input text string
    
    Returns:
        List of estimated entropy values for each word token
    """
    if not text:
        return []
    
    words = text.split()
    if len(words) == 0:
        return []
    
    # Count word frequencies globally
    word_counts = Counter(words)
    total_words = len(words)
    
    # Calculate base entropy for each word based on frequency
    # More frequent words = lower entropy
    base_entropies = []
    for word in words:
        freq = word_counts[word] / total_words
        # Inverse frequency as entropy proxy (normalized)
        # Rare words have higher entropy, common words have lower
        base_entropy = 1.0 - min(1.0, freq * 10)  # Scale factor to make it more sensitive
        base_entropies.append(base_entropy)
    
    # Adjust based on local context (sliding window)
    window_size = min(10, len(words))
    adjusted_entropies = []
    
    for i in range(len(words)):
        # Get local context
        start = max(0, i - window_size // 2)
        end = min(len(words), i + window_size // 2)
        context = words[start:end]
        
        # Count unique words in context (diversity = higher entropy)
        unique_in_context = len(set(context))
        total_in_context = len(context)
        diversity_ratio = unique_in_context / total_in_context if total_in_context > 0 else 1.0
        
        # Check if current word appears multiple times in context (repetition = lower entropy)
        word_in_context_count = context.count(words[i])
        repetition_penalty = 1.0 - min(0.5, (word_in_context_count - 1) * 0.2)  # Penalize repetition
        
        # Combine base entropy with local context
        adjusted_entropy = base_entropies[i] * diversity_ratio * repetition_penalty
        adjusted_entropies.append(max(0.0, min(1.0, adjusted_entropy)))
    
    return adjusted_entropies


def compute_low_entropy_penalty(
    response: str,
    entropy_threshold: Optional[float] = None,
    max_low_entropy_ratio: float = 0.3,
    token_entropies: Optional[List[float]] = None,
) -> float:
    """Compute penalty score for responses with too many low entropy tokens.
    
    Args:
        response: The model response string
        entropy_threshold: Fixed entropy threshold. Tokens with entropy < threshold are considered low entropy.
                          If provided, percentile_threshold is ignored (mutually exclusive).
        percentile_threshold: Percentile threshold (0.0-1.0). Tokens with entropy < this percentile are low entropy.
                             Only used if entropy_threshold is None (mutually exclusive).
                             E.g., 0.1 means tokens below 10th percentile are low entropy.
        max_low_entropy_ratio: Maximum allowed ratio of low entropy tokens (0.0-1.0).
                               If ratio exceeds this, apply penalty. Default: 0.1
        use_token_entropy: Whether to use provided token_entropies instead of estimating.
        token_entropies: Optional list of token-level entropy values (if available from model).
    
    Note:
        entropy_threshold and percentile_threshold are mutually exclusive.
        If entropy_threshold is provided, it takes precedence and percentile_threshold is ignored.
    
    Returns:
        Excess ratio (0.0 if below threshold, >0.0 if above threshold).
        Represents how much the low entropy ratio exceeds max_low_entropy_ratio.
        E.g., if max_low_entropy_ratio=0.1 and actual ratio=0.3, returns 0.2.
    """
    if not response:
        return 0.0
    
    # Get token-level entropies
    entropies = token_entropies
    if len(entropies) == 0:
        return 0.0
    
    # Determine threshold: use entropy_threshold if provided, otherwise use percentile_threshold
    # They are mutually exclusive
    if entropy_threshold is not None:
        # Use fixed threshold (ignore percentile_threshold)
        threshold = entropy_threshold
    else:
        # Default: use 10th percentile if neither is provided
        sorted_entropies = sorted(entropies)
        threshold_idx = int(len(sorted_entropies) * 0.1)
        threshold = sorted_entropies[threshold_idx] if threshold_idx < len(sorted_entropies) else sorted_entropies[-1]
    
    # Count low entropy tokens
    low_entropy_count = sum(1 for e in entropies if e < threshold)
    total_tokens = len(entropies)
    low_entropy_ratio = low_entropy_count / total_tokens if total_tokens > 0 else 0.0
    
    # Compute excess ratio: how much above the threshold
    # Returns the excess ratio (0.0 if below threshold, >0.0 if above)
    if low_entropy_ratio <= max_low_entropy_ratio:
        excess_ratio = 0.0
    else:
        # Calculate excess ratio: how much above the threshold
        # e.g., if max_low_entropy_ratio=0.1 and low_entropy_ratio=0.3, excess_ratio=0.2
        excess_ratio = low_entropy_ratio - max_low_entropy_ratio
        # Cap at 1.0 - max_low_entropy_ratio to prevent excessive penalty
        excess_ratio = min(excess_ratio, 1.0 - max_low_entropy_ratio)
    
    return excess_ratio


def format_reward(response: str) -> float:
    # 检查新格式：<thinking> </thinking> 标签 + \boxed{}
    pattern = re.compile(r"<thinking>.*</thinking>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.search(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,
    # Low entropy penalty parameters
    entropy_threshold: Optional[float] = None,
    max_low_entropy_ratio: float = 0.1,
    low_entropy_penalty_weight: float = 0.1,
    **kwargs
) -> List[Dict[str, float]]:
    """Compute reward scores with low entropy token penalty.
    
    Args:
        reward_inputs: List of reward input dictionaries, each containing:
            - "response": The model response string
            - "ground_truth": The ground truth answer
            - "token_entropies": (optional) List of token-level entropy values
        format_weight: Weight for format score in overall score
        entropy_threshold: Fixed entropy threshold (if provided, overrides percentile_threshold)
        percentile_threshold: Percentile threshold (0.0-1.0) for determining low entropy tokens.
                            E.g., 0.1 means tokens below 10th percentile are low entropy.
                            Default: 0.1
        max_low_entropy_ratio: Maximum allowed ratio of low entropy tokens (0.0-1.0).
                              If actual ratio exceeds this, penalty is applied. Default: 0.1 (10%)
        low_entropy_penalty_weight: Penalty factor for low entropy tokens (0.0-1.0).
                                   Controls how much to reduce overall reward per unit of excess ratio.
                                   E.g., 0.5 means 50% reduction per unit excess ratio.
                                   Default: 0.1
        use_token_entropy: Whether to use token_entropies from reward_inputs if available.
                          Default: False (estimate from text)
    
    Returns:
        List of score dictionaries, each containing:
            - "overall": Overall reward score (linearly penalized if low entropy ratio exceeds threshold)
            - "format": Format score
            - "accuracy": Accuracy score
            - "low_entropy_penalty": Excess low entropy ratio (0.0 = no excess, >0.0 = excess above threshold)
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        
        # Get token entropies if available
        token_entropies = None
        if "token_entropies" in reward_input:
            token_entropies = reward_input["token_entropies"]
        else:
            print("DEBUG token_entropies not available")
        # Compute low entropy penalty ratio (excess ratio above threshold)
        low_entropy_penalty_ratio = compute_low_entropy_penalty(
            response,
            entropy_threshold=entropy_threshold,
            max_low_entropy_ratio=max_low_entropy_ratio,
            token_entropies=token_entropies,
        )
        
        # Calculate base overall score (without low entropy penalty)
        base_overall = (1 - format_weight) * accuracy_score + format_weight * format_score
        
        # Apply linear penalty to overall reward based on excess low entropy ratio
        # penalty_factor controls how much to reduce the reward per unit of excess ratio
        # e.g., if penalty_factor=1.0 and excess_ratio=0.2, reward is reduced by 20%
        penalty_factor = low_entropy_penalty_weight  # Use this parameter to control penalty strength
        overall = base_overall * (1.0 - penalty_factor * low_entropy_penalty_ratio)
        
        # Ensure overall score is non-negative
        overall = max(0.0, overall)
        
        scores.append(
            {
                "overall": overall,
                "format": format_score,
                "accuracy": accuracy_score,
                "low_entropy_penalty": low_entropy_penalty_ratio,
            }
        )

    return scores

