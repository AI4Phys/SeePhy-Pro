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
from typing import Any


# Metadata
REWARD_NAME = "dapo"
REWARD_TYPE = "batch"


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def extract_answer_from_boxed(response: str) -> str:
    """Extract answer from \\boxed{} tag in the response.
    
    Args:
        response: The model response string
        
    Returns:
        Extracted answer string, or "[INVALID]" if not found
    """
    # Try to find \boxed{...} pattern
    # Handle both \boxed{answer} and \boxed{answer} with nested braces
    patterns = [
        r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",  # Simple nested braces
        r"\\boxed\{([^}]+)\}",  # Fallback: anything until first }
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            # Take the last match (most likely the final answer)
            return matches[-1]
    
    # If no \boxed{} found, try to find any boxed content (case insensitive)
    matches = re.findall(r"(?i)boxed\{([^}]+)\}", response)
    if matches:
        return matches[-1]
    
    return "[INVALID]"


def check_thinking_tag(response: str) -> bool:
    """Check if response contains <thinking> tag.
    
    Args:
        response: The model response string
        
    Returns:
        True if <thinking> tag is present, False otherwise
    """
    return bool(re.search(r"<thinking>", response, re.IGNORECASE))


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Compute accuracy reward based on extracted answer from \\boxed{}.
    
    Args:
        response: The model response string
        ground_truth: The ground truth answer
        
    Returns:
        1.0 if answer matches, -1.0 otherwise
    """
    answer = extract_answer_from_boxed(response)
    if answer == "[INVALID]":
        return -1.0
    
    if normalize_final_answer(answer) == normalize_final_answer(ground_truth):
        return 1.0
    else:
        return -1.0


def soft_overlong_punishment(response_length: int, max_response_length: int, overlong_buffer_length: int):
    """Compute soft punishment for overlong responses.
    
    Args:
        response_length: Length of the response
        max_response_length: Maximum allowed response length
        overlong_buffer_length: Buffer length before applying punishment
        
    Returns:
        Punishment score (0.0 to -1.0)
    """
    expected_len = max_response_length - overlong_buffer_length
    if response_length <= expected_len:
        return 0.0
    elif response_length <= max_response_length:
        return (expected_len - response_length) / overlong_buffer_length
    else:
        return -1.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    max_response_length: int,
    overlong_buffer_length: int,
    overlong_penalty_factor: float,
) -> list[dict[str, float]]:
    """Compute reward scores for a batch of responses.
    
    Args:
        reward_inputs: List of reward input dictionaries, each containing:
            - "response": The model response string
            - "ground_truth": The ground truth answer
            - "response_length": Length of the response
        max_response_length: Maximum allowed response length
        overlong_buffer_length: Buffer length before applying punishment
        overlong_penalty_factor: Factor to multiply overlong punishment
        
    Returns:
        List of score dictionaries, each containing:
            - "overall": Overall reward score
            - "accuracy": Accuracy reward score
            - "overlong": Overlong punishment score
            - "accuracy_normalized": Normalized accuracy (0.0 to 1.0)
    """
    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"][-300:]  # The longest answer in MATH-500 has 159 characters
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        overlong_score = soft_overlong_punishment(
            reward_input["response_length"], max_response_length, overlong_buffer_length
        )
        scores.append(
            {
                "overall": accuracy_score + overlong_score * overlong_penalty_factor,
                "accuracy": accuracy_score,
                "overlong": overlong_score,
                "accuracy_normalized": 0.5 * (accuracy_score + 1.0),
            }
        )

    return scores

