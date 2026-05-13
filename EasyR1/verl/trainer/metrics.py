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

from typing import Any

import numpy as np
import torch

from ..protocol import DataProto
from ..utils import torch_functional as VF


def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    return {key: np.mean(value) for key, value in metrics.items()}


def compute_length_metrics(batch: DataProto) -> dict[str, Any]:
    max_response_length = batch.batch["responses"].size(-1)
    max_prompt_length = batch.batch["attention_mask"].size(-1) - max_response_length

    prompt_length = batch.batch["attention_mask"][:, :-max_response_length].sum(-1).float()
    response_length = batch.batch["attention_mask"][:, -max_response_length:].sum(-1).float()

    return {
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/variance": torch.var(response_length).detach().item(),
        "response_length/clip_ratio": torch.eq(response_length, max_response_length).float().mean().detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/variance": torch.var(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.eq(prompt_length, max_prompt_length).float().mean().detach().item(),
    }


def compute_repetition_rate(tokens: list[int], n: int = 20) -> float:
    """Compute repetition rate using n-gram overlap.
    
    Args:
        tokens: List of token IDs
        n: n-gram size (default: 2 for bigram)
        
    Returns:
        Repetition rate in [0.0, 1.0]
    """
    if len(tokens) < n:
        return 0.0
    
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if len(ngrams) == 0:
        return 0.0
    
    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)
    
    # Repetition rate = 1 - (unique_ngrams / total_ngrams)
    # Higher value means more repetition
    return 1.0 - (unique_ngrams / total_ngrams)


def compute_response_half_metrics(batch: DataProto, k: int = 1000) -> dict[str, Any]:
    """Compute metrics for tokens after the k-th token: entropy and repetition rate.
    
    Args:
        batch: DataProto containing responses, response_mask, and old_log_probs
        k: Starting token index (0-indexed). Only sequences with length > k will be included.
        
    Returns:
        Dictionary with metrics for tokens after the k-th token
    """
    max_response_length = batch.batch["responses"].size(-1)
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()
    responses = batch.batch["responses"]  # (batch_size, response_length)
    
    # Compute response lengths
    response_lengths = response_mask.sum(dim=-1).long()  # (batch_size,)
    
    half_entropies = []
    half_repetition_rates = []
    
    for i in range(len(batch)):
        seq_length = int(response_lengths[i].item())
        # Skip sequences that are shorter than k
        if seq_length <= k:
            continue
        
        # Get tokens after the k-th token
        start_idx = k
        end_idx = seq_length
        
        # Extract mask and tokens after k-th token
        tail_mask = response_mask[i, start_idx:end_idx]
        tail_tokens = responses[i, start_idx:end_idx]
        
        # Filter valid tokens using mask
        valid_tail_tokens = tail_tokens[tail_mask].cpu().tolist()
        
        if len(valid_tail_tokens) == 0:
            continue
        
        # Compute entropy for tokens after k-th token (if old_log_probs available)
        if "old_log_probs" in batch.batch:
            old_log_probs = batch.batch["old_log_probs"]  # (batch_size, response_length)
            tail_log_probs = old_log_probs[i, start_idx:end_idx]
            
            # Compute average entropy (negative log prob) for tokens after k-th token
            if tail_mask.sum() > 0:
                tail_entropy = VF.masked_mean(-tail_log_probs, tail_mask).item()
                half_entropies.append(tail_entropy)
        
        # Compute repetition rate for tokens after k-th token
        if len(valid_tail_tokens) >= 2:
            rep_rate = compute_repetition_rate(valid_tail_tokens, n=20)
            half_repetition_rates.append(rep_rate)
    
    metrics = {}
    
    if half_entropies:
        metrics["response_half/entropy_mean"] = np.mean(half_entropies)
        metrics["response_half/entropy_max"] = np.max(half_entropies)
        metrics["response_half/entropy_min"] = np.min(half_entropies)
    else:
        metrics["response_half/entropy_mean"] = 0.0
        metrics["response_half/entropy_max"] = 0.0
        metrics["response_half/entropy_min"] = 0.0
    
    if half_repetition_rates:
        metrics["response_half/repetition_rate_mean"] = np.mean(half_repetition_rates)
        metrics["response_half/repetition_rate_max"] = np.max(half_repetition_rates)
        metrics["response_half/repetition_rate_min"] = np.min(half_repetition_rates)
    else:
        metrics["response_half/repetition_rate_mean"] = 0.0
        metrics["response_half/repetition_rate_max"] = 0.0
        metrics["response_half/repetition_rate_min"] = 0.0
    
    return metrics


def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> dict[str, Any]:
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].size(-1)
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    return {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        **compute_length_metrics(batch),
        **compute_response_half_metrics(batch, k=batch.meta_info.get("response_half_k", 4096)),
    }


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    num_response_tokens = torch.sum(batch.batch["response_mask"]).item()
    num_overall_tokens = sum(batch.meta_info["global_token_num"])
    num_tokens_of_section = {
        **dict.fromkeys(["gen", "reward"], num_response_tokens),
        **dict.fromkeys(["ref", "old", "values", "adv", "update_critic", "update_actor"], num_overall_tokens),
    }
    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], num_gpus: int) -> dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * num_gpus),
    }
