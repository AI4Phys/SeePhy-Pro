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
from typing import Any, Dict, List, Set

from mathruler.grader import extract_boxed_content, grade_answer


def extract_assumptions(response: str) -> List[str]:
    """Extract assumptions from <assumption> </assumption> tags.
    
    Args:
        response: The model response string
        
    Returns:
        List of assumption strings, each in format like "friction: ignored"
    """
    pattern = re.compile(r"<assumption>(.*?)</assumption>", re.DOTALL | re.IGNORECASE)
    match = re.search(pattern, response)
    if not match:
        return []
    
    assumption_text = match.group(1).strip()
    # Split by lines and filter out empty lines
    assumptions = [line.strip() for line in assumption_text.split('\n') if line.strip()]
    # Remove leading dashes or bullets if present
    assumptions = [re.sub(r'^[-•*]\s*', '', assump) for assump in assumptions]
    return assumptions


def extract_assumption_keywords(assumptions: List[str]) -> Set[str]:
    """Extract keywords from assumptions for matching.
    
    Args:
        assumptions: List of assumption strings
        
    Returns:
        Set of keywords extracted from assumptions (full phrases and variations)
    """
    keywords = set()
    for assumption in assumptions:
        # Split by colon to get the key part
        if ':' in assumption:
            key_part = assumption.split(':')[0].strip().lower()
        else:
            key_part = assumption.strip().lower()
        
        # Add the full key phrase (for multi-word concepts)
        keywords.add(key_part)
        
        # Add common variations (English)
        if 'friction' in key_part:
            keywords.update([
                'friction', 'frictional force', 'frictional', 'frict', 
                'friction force', 'friction coefficient', 'coefficient of friction'
            ])
        if 'resistance' in key_part:
            keywords.update([
                'resistance', 'air resistance', 'resist', 'drag', 'drag force',
                'air drag', 'wind resistance'
            ])
        if 'velocity' in key_part or 'speed' in key_part:
            keywords.update([
                'velocity', 'speed', 'v', 'rate', 'velocities', 'speeds',
                'initial velocity', 'final velocity', 'constant velocity',
                'uniform velocity', 'velocity vector'
            ])
        if 'gravity' in key_part or 'gravitational' in key_part:
            keywords.update([
                'gravity', 'gravitational', 'g', 'gravitational force',
                'gravitational acceleration', 'acceleration due to gravity',
                'earth gravity', 'gravity constant'
            ])
        if 'mass' in key_part:
            keywords.update([
                'mass', 'm', 'weight', 'masses', 'total mass',
                'object mass', 'particle mass'
            ])
        if 'force' in key_part:
            keywords.update([
                'force', 'f', 'forces', 'net force', 'applied force',
                'external force', 'internal force', 'resultant force'
            ])
        if 'acceleration' in key_part:
            keywords.update([
                'acceleration', 'a', 'accelerate', 'accelerating',
                'constant acceleration', 'uniform acceleration',
                'acceleration vector'
            ])
        if 'constant' in key_part:
            keywords.update([
                'constant', 'const', 'fixed', 'uniform', 'unchanged',
                'remains constant', 'is constant', 'constant value'
            ])
        if 'temperature' in key_part:
            keywords.update([
                'temperature', 'temp', 't', 'thermal', 'heat',
                'room temperature', 'ambient temperature'
            ])
        if 'pressure' in key_part:
            keywords.update([
                'pressure', 'p', 'atmospheric pressure', 'air pressure',
                'pressure difference'
            ])
        if 'energy' in key_part:
            keywords.update([
                'energy', 'kinetic energy', 'potential energy',
                'mechanical energy', 'total energy', 'energy conservation'
            ])
        if 'momentum' in key_part:
            keywords.update([
                'momentum', 'linear momentum', 'angular momentum',
                'momentum conservation', 'conservation of momentum'
            ])
        if 'collision' in key_part:
            keywords.update([
                'collision', 'elastic collision', 'inelastic collision',
                'collide', 'colliding'
            ])
        if 'ignored' in key_part or 'ignore' in key_part or 'neglect' in key_part:
            keywords.update([
                'ignored', 'ignore', 'neglect', 'neglected', 'assume',
                'assuming', 'assumption', 'negligible', 'can be ignored'
            ])
        
        # Add common variations (Chinese)
        if '摩擦' in key_part or '摩擦力' in key_part:
            keywords.update([
                '摩擦', '摩擦力', '摩擦系数', '摩擦阻力'
            ])
        if '阻力' in key_part or '空气阻力' in key_part:
            keywords.update([
                '阻力', '空气阻力', '风阻', '阻力系数'
            ])
        if '速度' in key_part:
            keywords.update([
                '速度', '速率', '初速度', '末速度', '恒定速度',
                '匀速', '速度矢量'
            ])
        if '重力' in key_part or '引力' in key_part:
            keywords.update([
                '重力', '引力', '万有引力', '重力加速度',
                '地球重力', '重力常数'
            ])
        if '质量' in key_part:
            keywords.update([
                '质量', '总质量', '物体质量', '粒子质量'
            ])
        if '力' in key_part:
            keywords.update([
                '力', '合力', '作用力', '外力', '内力', '净力'
            ])
        if '加速度' in key_part:
            keywords.update([
                '加速度', '匀加速', '恒定加速度', '加速度矢量'
            ])
        if '恒定' in key_part or '不变' in key_part or '均匀' in key_part:
            keywords.update([
                '恒定', '不变', '均匀', '保持不变', '是恒定的'
            ])
        if '温度' in key_part:
            keywords.update([
                '温度', '室温', '环境温度', '热'
            ])
        if '压力' in key_part or '压强' in key_part:
            keywords.update([
                '压力', '压强', '大气压', '气压', '压力差'
            ])
        if '能量' in key_part:
            keywords.update([
                '能量', '动能', '势能', '机械能', '总能量', '能量守恒'
            ])
        if '动量' in key_part:
            keywords.update([
                '动量', '线动量', '角动量', '动量守恒'
            ])
        if '碰撞' in key_part:
            keywords.update([
                '碰撞', '弹性碰撞', '非弹性碰撞'
            ])
        if '忽略' in key_part or '忽视' in key_part or '不计' in key_part:
            keywords.update([
                '忽略', '忽视', '不计', '可忽略', '假设', '假定'
            ])
    
    return keywords


def assumption_consistency_reward(response: str) -> float:
    """Check if thinking section uses the assumptions listed.
    
    Args:
        response: The model response string
        
    Returns:
        Score from 0.0 to 1.0 indicating how well assumptions are used in thinking
    """
    assumptions = extract_assumptions(response)
    if not assumptions:
        return 0.0  # No assumptions listed, cannot check consistency
    
    # Extract thinking section
    thinking_pattern = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)
    thinking_match = re.search(thinking_pattern, response)
    if not thinking_match:
        return 0.0  # No thinking section found
    
    thinking_text = thinking_match.group(1).lower()
    
    # Extract keywords from assumptions
    assumption_keywords = extract_assumption_keywords(assumptions)
    if not assumption_keywords:
        return 0.0
    
    # Count how many assumption keywords appear in thinking
    matched_keywords = set()
    for keyword in assumption_keywords:
        # Use word boundary matching for better accuracy
        if len(keyword) > 2:  # Only match keywords longer than 2 chars
            pattern = r'\b' + re.escape(keyword) + r'\b'
        else:
            pattern = re.escape(keyword)
        
        if re.search(pattern, thinking_text, re.IGNORECASE):
            matched_keywords.add(keyword)
    
    # Calculate score: at least 50% of unique assumption concepts should be mentioned
    # We consider each assumption as a concept
    num_assumptions = len(assumptions)
    if num_assumptions == 0:
        return 0.0
    
    # For each assumption, check if at least one of its keywords appears
    assumptions_used = 0
    for assumption in assumptions:
        assumption_lower = assumption.lower()
        # Extract keywords from this specific assumption
        if ':' in assumption_lower:
            key_part = assumption_lower.split(':')[0].strip()
        else:
            key_part = assumption_lower.strip()
        
        # Check if any word from this assumption appears in thinking
        assumption_words = re.findall(r'\b\w+\b', key_part)
        if any(word in thinking_text for word in assumption_words if len(word) > 2):
            assumptions_used += 1
        # Also check for the full key phrase
        elif key_part in thinking_text:
            assumptions_used += 1
    
    # Score is the ratio of assumptions used
    consistency_score = assumptions_used / num_assumptions
    
    # Bonus: if most keywords are matched, give extra credit
    keyword_match_ratio = len(matched_keywords) / len(assumption_keywords) if assumption_keywords else 0
    # Weighted average: 70% from assumption usage, 30% from keyword matching
    final_score = 0.7 * consistency_score + 0.3 * min(1.0, keyword_match_ratio * 2)
    
    return final_score


def format_reward(response: str) -> float:
    # 检查新格式：<assumption> </assumption> + <thinking> </thinking> 标签 + \boxed{}
    pattern = re.compile(
        r"<assumption>.*?</assumption>.*?<thinking>.*?</thinking>.*?\\boxed\{.*?\}.*",
        re.DOTALL | re.IGNORECASE
    )
    format_match = re.search(pattern, response)
    if not format_match:
        return 0.0
    
    # 检查 assumption 内容格式是否为 -xx: xx\n 格式
    assumption_pattern = re.compile(r"<assumption>(.*?)</assumption>", re.DOTALL | re.IGNORECASE)
    assumption_match = re.search(assumption_pattern, response)
    if not assumption_match:
        return 0.0
    
    assumption_content = assumption_match.group(1).strip()
    if not assumption_content:
        return 0.0
    
    # 检查每行是否以 - 开头，并且包含 : 分隔符
    # 允许空行，但至少需要有一行符合格式
    lines = assumption_content.split('\n')
    valid_lines = 0
    for line in lines:
        line = line.strip()
        if not line:  # 跳过空行
            continue
        # 检查是否以 - 开头，并且包含 : 分隔符
        if line.startswith('-') and ':' in line:
            valid_lines += 1
    
    # 至少需要有一行符合格式
    return 1.0 if valid_lines > 0 else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    reward_inputs: List[Dict[str, Any]], 
    format_weight: float = 0.1,
    assumption_consistency_weight: float = 0.1,
    **kwargs
) -> List[Dict[str, float]]:
    """Compute reward scores with assumption consistency checking.
    
    Args:
        reward_inputs: List of reward input dictionaries
        format_weight: Weight for format score (default: 0.1)
        assumption_consistency_weight: Weight for assumption consistency score (default: 0.1)
        **kwargs: Additional keyword arguments
        
    Returns:
        List of score dictionaries containing:
            - "overall": Overall reward score
            - "format": Format score (0.0 or 1.0)
            - "accuracy": Accuracy score (0.0 or 1.0)
            - "assumption_consistency": Assumption consistency score (0.0 to 1.0)
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        assumption_consistency_score = assumption_consistency_reward(response)
        
        # Calculate overall score: accuracy is most important, then format and assumption consistency
        remaining_weight = 1.0 - format_weight - assumption_consistency_weight
        overall = (
            remaining_weight * accuracy_score + 
            format_weight * format_score + 
            assumption_consistency_weight * assumption_consistency_score
        )
        
        scores.append(
            {
                "overall": overall,
                "format": format_score,
                "accuracy": accuracy_score,
                "assumption_consistency": assumption_consistency_score
            }
        )

    return scores

