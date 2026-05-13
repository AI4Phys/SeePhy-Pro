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

import importlib.util
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class SequentialFunctionRewardManagerMixin:
    reward_fn: SequentialRewardFunction

    def compute_reward_sequential(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]], dict[str, torch.Tensor]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        response_mask = data.batch["response_mask"]
        
        # Extract token entropies if available (from old_log_probs)
        token_entropies_available = "old_log_probs" in data.batch
        
        # Track all reward dimensions (excluding "overall")
        reward_dimensions = set()
        
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            
            reward_input = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }
            
            # Add keyword field if available in non_tensor_batch
            if "keyword" in data.non_tensor_batch:
                reward_input["keyword"] = data.non_tensor_batch["keyword"][i]
            
            # Add token entropies if available
            if token_entropies_available:
                old_log_probs = data.batch["old_log_probs"]  # (batch_size, response_length)
                seq_mask = response_mask[i]  # (response_length,)
                seq_log_probs = old_log_probs[i]  # (response_length,)
                seq_entropies = (-seq_log_probs).cpu().tolist()
                valid_entropies = [
                    seq_entropies[j] 
                    for j in range(cur_response_length) 
                    if seq_mask[j].item()
                ]
                reward_input["token_entropies"] = valid_entropies
            
            score = self.reward_fn(reward_input)
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)
                # Track all dimensions except "overall"
                if key != "overall":
                    reward_dimensions.add(key)
        
        # Create multi-dimensional token-level scores for GDPO
        multi_dim_scores = {}
        if reward_dimensions:
            batch_size = len(data)
            for dim_name in reward_dimensions:
                dim_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
                for i in range(batch_size):
                    cur_response_length = int(response_length[i].item())
                    if dim_name in reward_metrics and i < len(reward_metrics[dim_name]):
                        dim_tensor[i, cur_response_length - 1] = reward_metrics[dim_name][i]
                multi_dim_scores[f"token_level_scores_{dim_name}"] = dim_tensor

        return reward_tensor, reward_metrics, multi_dim_scores


class BatchFunctionRewardManagerMixin:
    reward_fn: BatchRewardFunction

    def compute_reward_batch(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]], dict[str, torch.Tensor]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        response_mask = data.batch["response_mask"]
        
        # Extract token entropies if available (from old_log_probs)
        token_entropies_available = "old_log_probs" in data.batch

        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            
            reward_input = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }
            
            # Add keyword field if available in non_tensor_batch
            if "keyword" in data.non_tensor_batch:
                reward_input["keyword"] = data.non_tensor_batch["keyword"][i]
            
            # Add token entropies if available
            # entropy = -log_prob, so we compute entropy from old_log_probs
            if token_entropies_available:
                old_log_probs = data.batch["old_log_probs"]  # (batch_size, response_length)
                seq_mask = response_mask[i]  # (response_length,)
                
                # Extract valid token entropies (only for response tokens)
                # entropy = -log_prob, higher entropy = more uncertainty
                seq_log_probs = old_log_probs[i]  # (response_length,)
                seq_entropies = (-seq_log_probs).cpu().tolist()  # Convert to entropy
                
                # Filter to only valid tokens (where mask is True)
                valid_entropies = [
                    seq_entropies[j] 
                    for j in range(cur_response_length) 
                    if seq_mask[j].item()
                ]
                
                reward_input["token_entropies"] = valid_entropies
            
            reward_inputs.append(reward_input)

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        
        # Track all reward dimensions (excluding "overall")
        reward_dimensions = set()
        
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)
                # Track all dimensions except "overall"
                if key != "overall":
                    reward_dimensions.add(key)
        
        # Create multi-dimensional token-level scores for GDPO
        multi_dim_scores = {}
        if reward_dimensions:
            batch_size = len(scores)
            for dim_name in reward_dimensions:
                dim_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
                for i in range(batch_size):
                    cur_response_length = int(response_length[i].item())
                    if dim_name in reward_metrics and i < len(reward_metrics[dim_name]):
                        dim_tensor[i, cur_response_length - 1] = reward_metrics[dim_name][i]
                multi_dim_scores[f"token_level_scores_{dim_name}"] = dim_tensor

        return reward_tensor, reward_metrics, multi_dim_scores


class AutoRewardManager(BatchFunctionRewardManagerMixin, SequentialFunctionRewardManagerMixin):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        reward_name = getattr(module, "REWARD_NAME", "unknown")
        reward_type = getattr(module, "REWARD_TYPE", "batch")
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        print(f"Reward name: {reward_name}, reward type: {reward_type}.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.reward_type = reward_type
        self.config = config
        self.tokenizer = tokenizer

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]], dict[str, torch.Tensor]]:
        """Compute reward for a batch of data.
        
        Returns:
            reward_tensor: Token-level reward tensor (batch_size, seq_len) with overall scores
            reward_metrics: Dictionary of reward metrics for all dimensions
            multi_dim_scores: Dictionary containing token-level scores for each dimension (for GDPO)
        """
        if self.reward_type == "batch":
            return self.compute_reward_batch(data)
        elif self.reward_type == "sequential":
            return self.compute_reward_sequential(data)
        else:
            raise ValueError(f"Unsupported reward type: {self.reward_type}.")
