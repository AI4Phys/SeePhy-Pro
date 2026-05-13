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


def keyword_reward(response: str, keywords: List[str]) -> float:
    """
    Calculate caption reward based on keyword presence in response.
    
    Args:
        response: The model response string
        keywords: List of keywords to check for in the response
        
    Returns:
        Float between 0.0 and 1.0 representing the ratio of keywords found
    """
    if not keywords or len(keywords) == 0:
        return 1.0  # If no keywords, return full score
    
    # Normalize response to lowercase for case-insensitive matching
    response_lower = response.lower()
    
    # Count how many keywords appear in the response
    found_count = 0
    for keyword in keywords:
        if not keyword or not keyword.strip():
            continue
        # Case-insensitive substring matching
        keyword_lower = keyword.lower().strip()
        if keyword_lower in response_lower:
            found_count += 1
    
    # Return ratio of found keywords
    return found_count / len(keywords) if len(keywords) > 0 else 1.0


def compute_score(
    reward_inputs: List[Dict[str, Any]], 
    format_weight: float = 0.1,
    keyword_weight: float = 0.1,
    **kwargs
) -> List[Dict[str, float]]:
    """
    Compute reward scores with keyword-based caption reward.
    
    Args:
        reward_inputs: List of reward input dictionaries, each containing:
            - "response": The model response string
            - "ground_truth": The ground truth answer
            - "keyword": (optional) List of keywords to check for in response
        format_weight: Weight for format score (default: 0.1)
        keyword_weight: Weight for keyword/caption reward (default: 0.1)
        **kwargs: Additional keyword arguments
        
    Returns:
        List of score dictionaries containing:
            - "overall": Overall reward score
            - "format": Format score (0.0 or 1.0)
            - "accuracy": Accuracy score (0.0 or 1.0)
            - "keyword": Keyword/caption reward score (0.0 to 1.0)
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        
        # Get keywords from reward_input, default to empty list if not present
        keywords = reward_input.get("keyword", [])
        if keywords is None:
            keywords = []
        # Ensure keywords is a list
        if not isinstance(keywords, list):
            keywords = [keywords] if keywords else []
        
        keyword_score = keyword_reward(response, keywords)
        
        # Calculate overall score with weights
        # Ensure weights sum to 1.0 or less
        remaining_weight = max(0.0, 1.0 - format_weight - keyword_weight)
        overall = (
            remaining_weight * accuracy_score + 
            format_weight * format_score + 
            keyword_weight * keyword_score
        )
        
        scores.append(
            {
                "overall": overall,
                "format": format_score,
                "accuracy": accuracy_score,
                "keyword_score": keyword_score
            }
        )

    return scores

