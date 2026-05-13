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

def soft_overlong_punishment(response_length: int, max_response_length: int, overlong_buffer_length: int):
    """Compute soft punishment for overlong responses.
    Args:
        response_length: Length of the response
        max_response_length: Maximum allowed response length
        overlong_buffer_length: Buffer length before applying punishment
    """
    expected_len = max_response_length - overlong_buffer_length
    if response_length <= expected_len:
        return 1.0
    elif response_length <= max_response_length:
        return (expected_len - response_length) / overlong_buffer_length
    else:
        return 0.0
    

def compute_score(reward_inputs: List[Dict[str, Any]], 
                  max_response_length: int=4096,
                  overlong_buffer_length: int=512,
                  overlong_penalty_weight: float=0.2,
                  format_weight: float = 0.1, 
                  **kwargs) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        overlong_score = soft_overlong_punishment(
            reward_input["response_length"], max_response_length, overlong_buffer_length
        )
        acc_weight = 1 - format_weight - overlong_penalty_weight
        scores.append(
            {
                "overall": acc_weight * accuracy_score + format_weight * format_score + overlong_score * overlong_penalty_weight,
                "format": format_score,
                "accuracy": accuracy_score,
                "overlong": overlong_score,
            }
        )

    return scores