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


def length_reward(
    response_length: int,
    min_ideal_length: int = 2048,
    max_ideal_length: int = 6144,
    max_response_length: int = 8192,
    min_acceptable_length: int = 10,
) -> float:
    """Compute length-based reward with soft punishment for overlong and soft reward for short responses.
    
    Reward curve:
    - If length < min_acceptable_length: reward = 0.0 (too short)
    - If min_acceptable_length <= length < min_ideal_length: linear reward from 0.0 to 1.0 (soft reward for short)
    - If min_ideal_length <= length <= max_ideal_length: reward = 1.0 (ideal range)
    - If max_ideal_length < length <= max_response_length: linear penalty from 1.0 to 0.0 (soft punishment for overlong)
    - If length > max_response_length: reward = 0.0 (hard cutoff)
    
    Args:
        response_length: Length of the response in tokens
        min_acceptable_length: Minimum acceptable response length (reward = 0.0 below this)
        min_ideal_length: Minimum ideal response length (reward reaches 1.0 here)
        max_ideal_length: Maximum ideal response length (reward starts decreasing after this)
        max_response_length: Maximum allowed response length (reward = 0.0 beyond this)
        
    Returns:
        Length reward score in range [0.0, 1.0]
    """
    if response_length < min_acceptable_length:
        # Too short: no reward
        reward = 0.0
    elif response_length < min_ideal_length:
        # Soft reward for short responses: linear from 0.0 to 1.0
        reward_range = min_ideal_length - min_acceptable_length
        if reward_range > 0:
            progress = (response_length - min_acceptable_length) / reward_range
            reward = progress  # Linear from 0.0 to 1.0
        else:
            reward = 1.0 if response_length >= min_ideal_length else 0.0
    elif response_length <= max_ideal_length:
        # Ideal length range: full reward
        reward = 1.0
    elif response_length <= max_response_length:
        # Soft punishment for overlong responses: linear from 1.0 to 0.0
        penalty_range = max_response_length - max_ideal_length
        if penalty_range > 0:
            excess = response_length - max_ideal_length
            reward = 1.0 - (excess / penalty_range)  # Linear from 1.0 to 0.0
        else:
            reward = 1.0
    else:
        # Beyond max_response_length: hard cutoff
        reward = 0.0
    
    # Ensure reward is in [0.0, 1.0]
    return max(0.0, min(1.0, reward))


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,
    min_acceptable_length: int = 10,
    min_ideal_length: int = 2048,
    max_ideal_length: int = 6144,
    max_response_length: int = 8192,
    length_weight: float = 0.1,
    **kwargs
) -> List[Dict[str, float]]:
    """Compute reward scores with length-based soft punishment and reward.
    
    Args:
        reward_inputs: List of reward input dictionaries, each containing:
            - "response": The model response string
            - "ground_truth": The ground truth answer
            - "response_length": Length of the response in tokens
        format_weight: Weight for format reward (default: 0.1)
        min_acceptable_length: Minimum acceptable response length (default: 50)
        min_ideal_length: Minimum ideal response length (default: 100)
        max_ideal_length: Maximum ideal response length (default: 500)
        max_response_length: Maximum allowed response length (default: 4096)
        length_weight: Weight for length reward in overall score (default: 0.1)
        
    Returns:
        List of score dictionaries, each containing:
            - "overall": Overall reward score (0.0 to 1.0)
            - "format": Format reward score (0.0 or 1.0)
            - "accuracy": Accuracy reward score (0.0 or 1.0)
            - "length": Length reward score (0.0 to 1.0)
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        
        # Compute length reward
        response_length = reward_input.get("response_length", 0)
        length_score = length_reward(
            response_length=response_length,
            min_acceptable_length=min_acceptable_length,
            min_ideal_length=min_ideal_length,
            max_ideal_length=max_ideal_length,
            max_response_length=max_response_length,
        )
        
        # Combine scores: ensure overall is in [0, 1]
        # Remaining weight is split between accuracy and format
        remaining_weight = 1.0 - length_weight
        overall = (
            length_weight * length_score +
            remaining_weight * (1 - format_weight) * accuracy_score +
            remaining_weight * format_weight * format_score
        )
        
        scores.append(
            {
                "overall": overall,
                "format": format_score,
                "accuracy": accuracy_score,
                "length": length_score,
            }
        )

    return scores

