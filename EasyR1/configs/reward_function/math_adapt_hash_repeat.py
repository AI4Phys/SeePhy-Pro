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

try:
    from datasketch import MinHash
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False
    print("Warning: datasketch not installed. MinHash functionality will be disabled. Install with: pip install datasketch")

try:
    import simhash
    HAS_SIMHASH = True
except ImportError:
    HAS_SIMHASH = False
    print("Warning: simhash not installed. SimHash functionality will be disabled. Install with: pip install simhash")


from mathruler.grader import extract_boxed_content, grade_answer


def _compute_minhash(text: str, num_perm: int = 128) -> MinHash:
    """Compute MinHash signature for a text string.
    
    Args:
        text: Input text string
        num_perm: Number of permutations for MinHash (higher = more accurate)
    
    Returns:
        MinHash object
    """
    if not HAS_DATASKETCH:
        return None
    
    m = MinHash(num_perm=num_perm)
    # Use character n-grams (shingles) for better detection
    words = text.split()
    for i in range(len(words) - 2):  # 3-word shingles
        shingle = " ".join(words[i:i+3])
        m.update(shingle.encode('utf-8'))
    return m


def _compute_simhash(text: str) -> int:
    """Compute SimHash signature for a text string.
    
    Args:
        text: Input text string
    
    Returns:
        SimHash integer value
    """
    if not HAS_SIMHASH:
        return None
    
    # Use word-level features
    words = text.split()
    return simhash.Simhash(words).value


def _hamming_distance(hash1: int, hash2: int) -> int:
    """Compute Hamming distance between two SimHash values.
    
    Args:
        hash1: First SimHash value
        hash2: Second SimHash value
    
    Returns:
        Hamming distance (number of differing bits)
    """
    return bin(hash1 ^ hash2).count('1')


def detect_repetition_minhash(
    response: str,
    window_size: int = 50,
    similarity_threshold: float = 0.7,
    num_perm: int = 128,
) -> float:
    """Detect internal repetition using MinHash.
    
    This function splits the response into sliding windows and uses MinHash
    to detect similar windows (indicating repetition).
    
    Args:
        response: The model response string
        window_size: Number of words per window
        similarity_threshold: Jaccard similarity threshold to consider as repetition (0.0-1.0)
        num_perm: Number of permutations for MinHash
    
    Returns:
        Score from 0.0 (high repetition) to 1.0 (no repetition)
    """
    if not HAS_DATASKETCH:
        return 1.0  # If library not available, don't penalize
    
    words = response.split()
    if len(words) < window_size * 2:
        return 1.0  # Too short to have meaningful repetition
    
    # Create sliding windows
    windows = []
    window_hashes = []
    
    for i in range(0, len(words) - window_size + 1, window_size // 2):  # 50% overlap
        window_text = " ".join(words[i:i + window_size])
        windows.append(window_text)
        mh = _compute_minhash(window_text, num_perm)
        if mh is not None:
            window_hashes.append(mh)
        else:
            window_hashes.append(None)
    
    if len(window_hashes) < 2:
        return 1.0
    
    # Compare all pairs of windows
    repetition_count = 0
    total_comparisons = 0
    
    for i in range(len(window_hashes)):
        if window_hashes[i] is None:
            continue
        for j in range(i + 1, len(window_hashes)):
            if window_hashes[j] is None:
                continue
            # Skip adjacent windows (they naturally overlap)
            if abs(i - j) <= 1:
                continue
            
            total_comparisons += 1
            similarity = window_hashes[i].jaccard(window_hashes[j])
            if similarity >= similarity_threshold:
                repetition_count += 1
    
    if total_comparisons == 0:
        return 1.0
    
    repetition_ratio = repetition_count / total_comparisons
    # Return score: 1.0 for no repetition, decreasing linearly to 0.0 for high repetition
    return max(0.0, 1.0 - repetition_ratio * 2.0)  # Penalize more heavily


def detect_repetition_simhash(
    response: str,
    window_size: int = 50,
    hamming_threshold: int = 6,
) -> float:
    """Detect internal repetition using SimHash.
    
    This function splits the response into sliding windows and uses SimHash
    to detect similar windows based on Hamming distance.
    
    Args:
        response: The model response string
        window_size: Number of words per window
        hamming_threshold: Maximum Hamming distance to consider as repetition (lower = stricter)
    
    Returns:
        Score from 0.0 (high repetition) to 1.0 (no repetition)
    """
    if not HAS_SIMHASH:
        return 1.0  # If library not available, don't penalize
    
    words = response.split()
    if len(words) < window_size * 2:
        return 1.0  # Too short to have meaningful repetition
    
    # Create sliding windows
    window_hashes = []
    
    for i in range(0, len(words) - window_size + 1, window_size // 2):  # 50% overlap
        window_text = " ".join(words[i:i + window_size])
        sh = _compute_simhash(window_text)
        if sh is not None:
            window_hashes.append(sh)
        else:
            window_hashes.append(None)
    
    if len(window_hashes) < 2:
        return 1.0
    
    # Compare all pairs of windows
    repetition_count = 0
    total_comparisons = 0
    
    for i in range(len(window_hashes)):
        if window_hashes[i] is None:
            continue
        for j in range(i + 1, len(window_hashes)):
            if window_hashes[j] is None:
                continue
            # Skip adjacent windows (they naturally overlap)
            if abs(i - j) <= 1:
                continue
            
            total_comparisons += 1
            hamming_dist = _hamming_distance(window_hashes[i], window_hashes[j])
            if hamming_dist <= hamming_threshold:
                repetition_count += 1
    
    if total_comparisons == 0:
        return 1.0
    
    repetition_ratio = repetition_count / total_comparisons
    # Return score: 1.0 for no repetition, decreasing linearly to 0.0 for high repetition
    return max(0.0, 1.0 - repetition_ratio * 2.0)  # Penalize more heavily


def format_reward(
    response: str,
    use_minhash: bool = True,
    use_simhash: bool = True,
    minhash_weight: float = 0.5,
    simhash_weight: float = 0.5,
    minhash_window_size: int = 50,
    minhash_similarity_threshold: float = 0.7,
    simhash_window_size: int = 50,
    simhash_hamming_threshold: int = 6,
) -> float:
    """Compute format reward with repetition detection using MinHash and SimHash.
    
    The format reward combines:
    1. Basic format check (thinking tags + boxed answer)
    2. Repetition detection using MinHash and/or SimHash
    
    If repetition is detected, the format score is penalized (multiplied by repetition_score).
    This means: if format is wrong (0), score is 0; if format is correct (1), score depends on repetition.
    High repetition (repetition_score = 0) will reduce format_score to 0.
    
    Args:
        response: The model response string
        use_minhash: Whether to use MinHash for repetition detection
        use_simhash: Whether to use SimHash for repetition detection
        minhash_weight: Weight for MinHash score when both are used
        simhash_weight: Weight for SimHash score when both are used
        minhash_window_size: Window size for MinHash detection
        minhash_similarity_threshold: Similarity threshold for MinHash
        simhash_window_size: Window size for SimHash detection
        simhash_hamming_threshold: Hamming distance threshold for SimHash
    
    Returns:
        Format score from 0.0 to 1.0
    """
    # Basic format check: <thinking> </thinking> 标签 + \boxed{}
    pattern = re.compile(r"<thinking>.*</thinking>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.search(pattern, response)
    basic_format_score = 1.0 if format_match else 0.0
    
    # If basic format is wrong, return 0
    if basic_format_score == 0.0:
        return 0.0
    
    # Compute repetition scores
    repetition_scores = []
    
    if use_minhash:
        minhash_score = detect_repetition_minhash(
            response,
            window_size=minhash_window_size,
            similarity_threshold=minhash_similarity_threshold,
        )
        repetition_scores.append(("minhash", minhash_score, minhash_weight))
    
    if use_simhash:
        simhash_score = detect_repetition_simhash(
            response,
            window_size=simhash_window_size,
            hamming_threshold=simhash_hamming_threshold,
        )
        repetition_scores.append(("simhash", simhash_score, simhash_weight))
    
    if not repetition_scores:
        # If no repetition detection is enabled, return basic format score
        return basic_format_score
    
    # Combine repetition scores (weighted average if both are used)
    total_weight = sum(w for _, _, w in repetition_scores)
    if total_weight == 0:
        repetition_score = 1.0
    else:
        repetition_score = sum(score * weight for _, score, weight in repetition_scores) / total_weight
    
    # Combine basic format and repetition scores
    # If repetition is detected, penalize the format score until it reaches 0
    # format_score = basic_format_score * repetition_score
    # This means: if format is wrong (0), score is 0; if format is correct (1), score depends on repetition
    format_score = basic_format_score * repetition_score
    
    return format_score


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,
    # MinHash parameters
    use_minhash: bool = True,
    minhash_window_size: int = 50,
    minhash_similarity_threshold: float = 0.7,
    minhash_weight: float = 0.5,
    # SimHash parameters
    use_simhash: bool = True,
    simhash_window_size: int = 50,
    simhash_hamming_threshold: int = 6,
    simhash_weight: float = 0.5,
    **kwargs
) -> List[Dict[str, float]]:
    """Compute reward scores with MinHash/SimHash-based repetition detection.
    
    Args:
        reward_inputs: List of reward input dictionaries, each containing:
            - "response": The model response string
            - "ground_truth": The ground truth answer
        format_weight: Weight for format score in overall score
        use_minhash: Whether to use MinHash for repetition detection
        minhash_window_size: Window size (in words) for MinHash detection
        minhash_similarity_threshold: Jaccard similarity threshold (0.0-1.0) for MinHash
        minhash_weight: Weight for MinHash score when both MinHash and SimHash are used
        use_simhash: Whether to use SimHash for repetition detection
        simhash_window_size: Window size (in words) for SimHash detection
        simhash_hamming_threshold: Maximum Hamming distance for SimHash (lower = stricter)
        simhash_weight: Weight for SimHash score when both MinHash and SimHash are used
    
    Note:
        If repetition is detected, the format score is penalized (multiplied by repetition_score).
        High repetition will reduce format_score to 0.
    
    Returns:
        List of score dictionaries, each containing:
            - "overall": Overall reward score
            - "format": Format score (includes repetition penalty)
            - "accuracy": Accuracy score
            - "repetition_score": Repetition detection score (1.0 = no repetition, 0.0 = high repetition)
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        
        # Compute format score with repetition detection
        format_score = format_reward(
            response,
            use_minhash=use_minhash,
            use_simhash=use_simhash,
            minhash_weight=minhash_weight,
            simhash_weight=simhash_weight,
            minhash_window_size=minhash_window_size,
            minhash_similarity_threshold=minhash_similarity_threshold,
            simhash_window_size=simhash_window_size,
            simhash_hamming_threshold=simhash_hamming_threshold,
        )
        
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        
        # Compute standalone repetition score for logging
        repetition_scores = []
        weights = []
        if use_minhash:
            minhash_score = detect_repetition_minhash(
                response,
                window_size=minhash_window_size,
                similarity_threshold=minhash_similarity_threshold,
            )
            repetition_scores.append(minhash_score)
            weights.append(minhash_weight)
        if use_simhash:
            simhash_score = detect_repetition_simhash(
                response,
                window_size=simhash_window_size,
                hamming_threshold=simhash_hamming_threshold,
            )
            repetition_scores.append(simhash_score)
            weights.append(simhash_weight)
        
        if repetition_scores and weights:
            total_weight = sum(weights)
            repetition_score = sum(score * weight for score, weight in zip(repetition_scores, weights)) / total_weight
        else:
            repetition_score = 1.0
        
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
                "repetition_score": repetition_score,
            }
        )

    return scores

