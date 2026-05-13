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
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16
    max_images_per_sample: Optional[int] = None
    """optional preprocessing-time cap on the number of images in one sample"""
    max_estimated_input_length: Optional[int] = None
    """optional preprocessing-time cap on estimated prompt length (text + estimated multimodal tokens)"""
    estimated_image_token_stride: int = 28
    """estimate image tokens as ceil(sum(width * height) / stride^2) after image preprocessing"""
    dynamic_sampling_response_length: bool = False
    """enable dynamic sampling based on response length threshold per group"""
    dynamic_sampling_response_length_threshold: int = 512
    """minimum response length (in tokens) required for at least one response in a group to keep the group"""
    max_train_samples: Optional[int] = None
    """maximum number of training samples to use. If None, use all samples. If set, randomly sample this many samples from the dataset."""
    num_workers: int = 8
    """number of worker processes for data loading. Set to 0 to disable multiprocessing."""
    train_image_mask_mode: str = "none"
    """image ablation mode: `none`, `all_blank` (PAPO-style patch blackening), `random_blank`, `patch_mask`"""
    train_image_mask_ratio: float = 0.0
    """mask strength: patch blackening prob in `all_blank`, blank prob in `random_blank`, pixel ratio in `patch_mask`"""
    train_image_mask_patch_size: int = 32
    """side length (in pixels) of each image patch when `train_image_mask_mode` uses patch-based masking"""
    train_image_mask_seed: Optional[int] = None
    """optional random seed for deterministic image ablation"""
    train_add_blank_image_for_text: bool = False
    """if true, text-only training samples will be converted to single-image samples with a blank image"""
    train_blank_image_size: int = 512
    """side length (in pixels) of injected blank image for text-only samples"""

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `gdpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    use_multivariate_whitening: bool = False
    """use multivariate whitening for GDPO to decorrelate reward dimensions"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""
    online_response_length_filtering: bool = False
    """use online response length filtering based on group-level response lengths"""
    response_length_filter_low: int = 0
    """filter out groups with average response length < this value if online_response_length_filtering"""
    response_length_filter_high: int = 1000000
    """filter out groups with average response length > this value if online_response_length_filtering"""


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    dynamic_max_try_make_batch: bool = False
    """enable dynamic max_try_make_batch that increases during training"""
    max_try_make_batch_initial: int = 0
    """initial max_try_make_batch value at the start of training (when dynamic_max_try_make_batch is enabled)"""
    max_try_make_batch_final: int = 20
    """final max_try_make_batch value after warmup (when dynamic_max_try_make_batch is enabled)"""
    max_try_make_batch_warmup_steps: int = 1000
    """number of steps to gradually increase max_try_make_batch from initial to final value"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""
    dynamic_max_response_length: bool = False
    """enable dynamic max_response_length reduction during training"""
    dynamic_max_response_length_step_interval: int = 100
    """step interval for reducing max_response_length"""
    dynamic_max_response_length_reduce_tokens: int = 50
    """number of tokens to reduce per interval"""
    dynamic_max_response_length_min: int = 512
    """minimum max_response_length"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # may be not exist
        self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
