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
from typing import Any, Dict, List

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    # 检查新格式：<thinking> </thinking> 标签 + \boxed{}
    pattern = re.compile(r"<thinking>.*</thinking>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.search(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def detect_repeat_tokens(response: str, subseq_length: int = 10, max_repeat: int = 2) -> float:
    """Detect if there are repeated token subsequences in the response.
    
    Args:
        response: The model response string
        subseq_length: Length L of the subsequence to check (in tokens)
        max_repeat: Maximum allowed repeat count K. If any subsequence appears more than K times, return 0.0
    
    Returns:
        1.0 if no subsequence repeats more than K times, 0.0 otherwise
    """
    if len(response) < subseq_length:
        return 1.0  # Too short to have repeats
    
    # Split response into tokens (by whitespace)
    # You can modify this to use actual tokenizer if needed
    tokens = response.split()
    
    if len(tokens) < subseq_length:
        return 1.0  # Not enough tokens
    
    # Extract all subsequences of length L
    subsequences = []
    for i in range(len(tokens) - subseq_length + 1):
        subsequence = tuple(tokens[i:i + subseq_length])
        subsequences.append(subsequence)
    
    # Count occurrences of each subsequence
    subsequence_counts = Counter(subsequences)
    
    # Check if any subsequence appears more than K times
    max_count = max(subsequence_counts.values()) if subsequence_counts else 0
    
    if max_count > max_repeat:
        return 0.0
    else:
        return 1.0


def compute_score(
    reward_inputs: List[Dict[str, Any]], 
    format_weight: float = 0.1,
    repeat_subseq_length: int = 20,
    repeat_max_count: int = 3,
    repeat_penalty_weight: float = 0.1,
    **kwargs
) -> List[Dict[str, float]]:
    """Compute reward scores with repeat token detection.
    
    Args:
        reward_inputs: List of reward input dictionaries, each containing:
            - "response": The model response string
            - "ground_truth": The ground truth answer
        format_weight: Weight for format score
        repeat_subseq_length: Length L of subsequence to check for repeats
        repeat_max_count: Maximum allowed repeat count K
        repeat_penalty_weight: Weight for repeat penalty in overall score
    
    Returns:
        List of score dictionaries, each containing:
            - "overall": Overall reward score
            - "format": Format score
            - "accuracy": Accuracy score
            - "repeat": Repeat detection score (1.0 if no excessive repeats, 0.0 otherwise)
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        repeat_score = detect_repeat_tokens(response, repeat_subseq_length, repeat_max_count)
        
        # Calculate weights: accuracy + format + repeat = 1.0
        acc_weight = 1.0 - format_weight - repeat_penalty_weight
        
        scores.append(
            {
                "overall": acc_weight * accuracy_score + format_weight * format_score + repeat_penalty_weight * repeat_score,
                "format": format_score,
                "accuracy": accuracy_score,
                "repeat": repeat_score,
            }
        )

    return scores

