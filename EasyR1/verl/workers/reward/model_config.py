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
Reward Model config
"""

from dataclasses import dataclass, field

from ..actor.config import FSDPConfig, ModelConfig, OffloadConfig


@dataclass
class RewardModelConfig:
    """Configuration for Reward Model (neural network-based reward)."""
    strategy: str = "fsdp"
    """strategy for reward model, currently only support 'fsdp'"""
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    """FSDP configuration for reward model"""
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    """Offload configuration for reward model"""
    model: ModelConfig = field(default_factory=ModelConfig)
    """Model configuration for reward model"""
    padding_free: bool = True
    """use padding-free inference"""
    dynamic_batching: bool = True
    """enable dynamic batching"""
    ulysses_size: int = 1
    """ulysses sequence parallel size"""
    use_torch_compile: bool = False
    """enable torch compile (usually not needed for inference)"""
    # below are auto keys
    global_batch_size: int = field(default=256, init=False)
    """global batch size (not used for reward model, but required for _init_dist_mesh)"""
    micro_batch_size_per_device_for_update: int = field(default=1, init=False)
    """micro batch size (not used for reward model, but required for _init_dist_mesh)"""

