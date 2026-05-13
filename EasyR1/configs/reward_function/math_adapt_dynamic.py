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
from typing import Any, Dict, List, Optional

from mathruler.grader import extract_boxed_content, grade_answer

# Global counter to track training steps (incremented per batch)
# Note: This is approximate since each batch call increments by 1
_step_counter = 0


def format_reward(response: str) -> float:
    # 检查新格式：<thinking> </thinking> 标签 + \boxed{}
    pattern = re.compile(r"<thinking>.*</thinking>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.search(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_dynamic_format_weight(
    step: int,
    initial_format_weight: float = 0.1,
    step_interval: int = 50,
    weight_increment: float = 0.01,
    max_format_weight: float = 0.5,
) -> float:
    """Compute dynamic format weight based on training step.
    
    Args:
        step: Current training step
        initial_format_weight: Initial format weight
        step_interval: Step interval for weight increment
        weight_increment: Amount to increment weight per interval
        max_format_weight: Maximum format weight
    
    Returns:
        Current format weight
    """
    # Calculate how many intervals have passed
    intervals_passed = step // step_interval
    
    # Calculate new weight
    new_weight = initial_format_weight + intervals_passed * weight_increment
    
    # Clamp to max
    new_weight = min(new_weight, max_format_weight)
    
    return new_weight


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    initial_format_weight: float = 0.1,
    step_interval: int = 10,
    weight_increment: float = 0.01,
    max_format_weight: float = 0.5,
    current_step: Optional[int] = None,
    reset_counter: bool = False,
    **kwargs
) -> List[Dict[str, float]]:
    """Compute reward scores with dynamically adjusted format weight.
    
    Args:
        reward_inputs: List of reward input dictionaries
        initial_format_weight: Initial format weight (default: 0.1)
        step_interval: Step interval for weight increment (default: 50)
        weight_increment: Amount to increment weight per interval (default: 0.01)
        max_format_weight: Maximum format weight (default: 0.5)
        current_step: Current training step (if provided, uses this instead of internal counter)
        reset_counter: Whether to reset the step counter (default: False)
    
    Returns:
        List of score dictionaries
    """
    global _step_counter
    
    if reset_counter:
        _step_counter = 0
    
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    # Use provided step or internal counter
    step_to_use = current_step if current_step is not None else _step_counter
    
    # Compute current format weight based on step
    current_format_weight = compute_dynamic_format_weight(
        step=step_to_use,
        initial_format_weight=initial_format_weight,
        step_interval=step_interval,
        weight_increment=weight_increment,
        max_format_weight=max_format_weight,
    )
    
    # Increment step counter only if not using provided step
    if current_step is None:
        _step_counter += 1

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - current_format_weight) * accuracy_score + current_format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
                "format_weight": current_format_weight,  # Log current format weight
            }
        )

    return scores

