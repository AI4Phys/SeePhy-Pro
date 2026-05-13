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
"""
Reward Model Worker for neural network-based reward computation.
"""

from collections import defaultdict
from typing import Dict, Tuple

import torch

from ...protocol import DataProto
from ...single_controller.base.decorator import Dispatch, register

# Lazy import to avoid circular dependency
# FSDPWorker will be imported when the class is actually created


def _create_reward_model_worker():
    """Create RewardModelWorker class dynamically to avoid circular import."""
    from ..fsdp_workers import FSDPWorker
    
    class RewardModelWorker(FSDPWorker):
        """
        Reward Model Worker for computing rewards using a neural network model.
        
        This worker inherits from FSDPWorker and supports:
        - FSDP distributed inference
        - Parameter offloading
        - Multi-modal input processing
        - Token-level reward computation
        """

        @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
        def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, list]]:
            """
            Compute reward scores for a batch of sequences.
            
            Args:
                data: DataProto containing prompts and responses with the following keys:
                    - batch["input_ids"]: (batch_size, seq_len) input token ids
                    - batch["attention_mask"]: (batch_size, seq_len) attention mask
                    - batch["response_mask"]: (batch_size, seq_len) mask for response tokens
                    - non_tensor_batch["multi_modal_inputs"]: (optional) multi-modal inputs
            
            Returns:
                reward_tensor: (batch_size, seq_len) token-level reward scores
                reward_metrics: Dictionary containing additional reward metrics
            """
            # Process multi-modal inputs if present
            self._process_multi_modal_inputs(data)
            data = data.to(torch.cuda.current_device())

            # Load model if using parameter offloading
            if self._use_param_offload:
                from ...utils.fsdp_utils import load_fsdp_model
                load_fsdp_model(self.fsdp_module)

            # Prepare model inputs
            with self.ulysses_sharding_manager:
                data = self.ulysses_sharding_manager.preprocess_data(data)

                # Build model inputs
                model_inputs = {
                    "input_ids": data.batch["input_ids"],
                    "attention_mask": data.batch["attention_mask"],
                }

                # Add multi-modal inputs if present
                # Note: Multi-modal inputs are typically processed in the model's forward method
                # The processor should have already converted images/videos to the correct format
                # and they should be in data.batch if the model expects them
                # For most vision-language models, pixel_values are added automatically by the processor

                # Forward pass to compute rewards
                with torch.no_grad():
                    outputs = self.fsdp_module(**model_inputs)

                    # Extract reward scores from model output
                    # The exact format depends on your reward model architecture
                    if hasattr(outputs, "reward_scores"):
                        # If model directly outputs reward scores
                        reward_scores = outputs.reward_scores
                    elif hasattr(outputs, "logits"):
                        # If model outputs logits, extract reward scores
                        # For sequence-level reward: use last token or mean pooling
                        # For token-level reward: use all logits
                        logits = outputs.logits  # (batch_size, seq_len, vocab_size) or (batch_size, seq_len, 1)
                        
                        if logits.dim() == 3:
                            # If logits have vocab dimension, we need to extract reward
                            # Option 1: Use a special reward token (e.g., last token)
                            # Option 2: Mean pooling over sequence
                            # Option 3: Use a learned projection
                            # Here we assume the model outputs (batch_size, seq_len, 1) or we take mean
                            if logits.shape[-1] == 1:
                                reward_scores = logits.squeeze(-1)  # (batch_size, seq_len)
                            else:
                                # If vocab_size > 1, we might need to use a specific token or projection
                                # For now, use mean pooling (you may need to adjust this)
                                reward_scores = logits.mean(dim=-1)  # (batch_size, seq_len)
                        elif logits.dim() == 2:
                            # Already in correct shape (batch_size, seq_len)
                            reward_scores = logits
                        else:
                            raise ValueError(f"Unexpected logits shape: {logits.shape}")
                    elif hasattr(outputs, "last_hidden_state"):
                        # If model outputs hidden states, we need a projection head
                        # This would require a custom model with reward head
                        raise NotImplementedError(
                            "Reward model with hidden states output requires a reward projection head. "
                            "Please ensure your model outputs 'reward_scores' or 'logits'."
                        )
                    else:
                        raise ValueError(
                            f"Reward model output must have 'reward_scores' or 'logits'. "
                            f"Got output keys: {outputs.keys() if hasattr(outputs, 'keys') else type(outputs)}"
                        )

                # Create output DataProto
                output = DataProto.from_dict(tensors={"reward_scores": reward_scores})
                output = self.ulysses_sharding_manager.postprocess_data(output)

            # Offload model if using parameter offloading
            if self._use_param_offload:
                from ...utils.fsdp_utils import offload_fsdp_model
                offload_fsdp_model(self.fsdp_module)

            # Convert to token-level rewards
            reward_tensor = output.batch["reward_scores"].float()

            # Build reward metrics
            reward_metrics = defaultdict(list)
            batch_size = reward_tensor.shape[0]
            response_mask = data.batch.get("response_mask", None)

            # Compute per-sample metrics
            for i in range(batch_size):
                if response_mask is not None:
                    # Only compute average reward for response tokens
                    sample_rewards = reward_tensor[i]  # (seq_len,)
                    sample_mask = response_mask[i]  # (seq_len,)
                    masked_rewards = sample_rewards * sample_mask.float()
                    seq_length = sample_mask.sum().item()
                    
                    if seq_length > 0:
                        avg_reward = masked_rewards.sum().item() / seq_length
                        max_reward = masked_rewards.max().item()
                        min_reward = masked_rewards[sample_mask.bool()].min().item() if sample_mask.sum() > 0 else 0.0
                    else:
                        avg_reward = 0.0
                        max_reward = 0.0
                        min_reward = 0.0
                else:
                    # If no mask, use last token reward
                    avg_reward = reward_tensor[i, -1].item()
                    max_reward = reward_tensor[i].max().item()
                    min_reward = reward_tensor[i].min().item()

                reward_metrics["overall"].append(avg_reward)
                reward_metrics["max"].append(max_reward)
                reward_metrics["min"].append(min_reward)

            output = output.to("cpu")
            return reward_tensor, reward_metrics
    
    return RewardModelWorker


# Create the class when module is imported (but after FSDPWorker can be imported)
# Use a module-level variable to ensure we always return the same class object
# This is critical for Ray's worker class checking
_reward_model_worker_class = None


def get_reward_model_worker():
    """Get RewardModelWorker class, creating it if necessary.
    
    This function ensures we always return the same class object,
    which is important for Ray's worker class checking.
    """
    global _reward_model_worker_class
    if _reward_model_worker_class is None:
        _reward_model_worker_class = _create_reward_model_worker()
    return _reward_model_worker_class


# Create the class at module level to ensure it's a stable object
# This is important for Ray's worker class checking
# We'll try to create it immediately, but if it fails due to circular import,
# we'll create a proxy that initializes it on first use
try:
    RewardModelWorker = get_reward_model_worker()
except (ImportError, AttributeError):
    # If we can't create it yet (circular import), create a placeholder
    # that will be replaced when the class is actually needed
    RewardModelWorker = None
