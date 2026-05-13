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

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from ..workers.rollout.config import RolloutConfig
from .config import DataConfig


_VAL_NAME_PATTERN = re.compile(r"[^0-9A-Za-z]+")


@dataclass(frozen=True)
class ValidationDataLoader:
    name: str
    data_path: str
    dataloader: StatefulDataLoader


def _normalize_validation_entries(val_files: Any) -> list[str]:
    if isinstance(val_files, str):
        stripped = val_files.strip()
        if not stripped:
            return []

        if stripped.startswith("[") and stripped.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(stripped)
                except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
                    continue
                return _normalize_validation_entries(parsed)

        return [entry.strip() for entry in re.split(r"[\n,]+", stripped) if entry.strip()]

    if isinstance(val_files, (list, tuple)):
        normalized_entries = []
        for entry in val_files:
            if not isinstance(entry, str):
                raise TypeError(f"Validation dataset entries must be strings, but got {type(entry)!r}.")
            normalized_entries.extend(_normalize_validation_entries(entry))
        return normalized_entries

    if val_files is None:
        return []

    raise TypeError(
        "`data.val_files` must be a string, list of strings, or a JSON/Python list literal, "
        f"but got {type(val_files)!r}."
    )


def _default_validation_name(data_path: str, index: int) -> str:
    dataset_name = data_path.split("@", maxsplit=1)[0].rstrip("/").split("/")[-1]
    split_name = data_path.split("@", maxsplit=1)[1] if "@" in data_path else ""
    raw_name = "_".join(part for part in (dataset_name, split_name) if part)
    normalized_name = _VAL_NAME_PATTERN.sub("_", raw_name).strip("_").lower()
    return normalized_name or f"val_{index + 1}"


def parse_validation_datasets(val_files: Any) -> list[tuple[str, str]]:
    validation_entries = _normalize_validation_entries(val_files)
    validation_specs: list[tuple[str, str]] = []
    name_counts: dict[str, int] = {}

    for index, entry in enumerate(validation_entries):
        alias = ""
        data_path = entry
        if "=" in entry:
            maybe_alias, maybe_path = entry.split("=", maxsplit=1)
            if maybe_alias.strip() and maybe_path.strip():
                alias = maybe_alias.strip()
                data_path = maybe_path.strip()

        base_name = (
            _VAL_NAME_PATTERN.sub("_", alias).strip("_").lower()
            if alias
            else _default_validation_name(data_path, index)
        )
        dedup_count = name_counts.get(base_name, 0)
        name_counts[base_name] = dedup_count + 1
        dataset_name = base_name if dedup_count == 0 else f"{base_name}_{dedup_count + 1}"
        validation_specs.append((dataset_name, data_path))

    return validation_specs


def create_dataloader(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin],
    rollout_config: Optional[RolloutConfig] = None,
) -> tuple[StatefulDataLoader, list[ValidationDataLoader]]:
    max_images_per_sample = config.max_images_per_sample
    if max_images_per_sample is None and rollout_config is not None and rollout_config.limit_images > 0:
        max_images_per_sample = rollout_config.limit_images

    max_estimated_input_length = config.max_estimated_input_length
    if max_estimated_input_length is None and rollout_config is not None:
        max_estimated_input_length = rollout_config.max_model_len or (
            config.max_prompt_length + config.max_response_length
        )

    train_dataset = RLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
        max_images_per_sample=max_images_per_sample,
        max_estimated_input_length=max_estimated_input_length,
        estimated_image_token_stride=config.estimated_image_token_stride,
        max_train_samples=config.max_train_samples,
        seed=config.seed,
        image_mask_mode=config.train_image_mask_mode,
        image_mask_ratio=config.train_image_mask_ratio,
        image_mask_patch_size=config.train_image_mask_patch_size,
        image_mask_seed=config.train_image_mask_seed,
        add_blank_image_for_text=config.train_add_blank_image_for_text,
        blank_image_size=config.train_blank_image_size,
    )
    # use sampler for better ckpt resume
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    validation_dataloaders: list[ValidationDataLoader] = []
    for dataset_name, data_path in parse_validation_datasets(config.val_files):
        val_dataset = RLHFDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=config.prompt_key,
            answer_key=config.answer_key,
            image_key=config.image_key,
            video_key=config.video_key,
            image_dir=config.image_dir,
            video_fps=config.video_fps,
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
            max_images_per_sample=max_images_per_sample,
            max_estimated_input_length=max_estimated_input_length,
            estimated_image_token_stride=config.estimated_image_token_stride,
            image_mask_mode="none",
            add_blank_image_for_text=False,
        )

        if config.val_batch_size == -1:
            val_batch_size = len(val_dataset)
        else:
            val_batch_size = config.val_batch_size

        val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )

        assert len(val_dataloader) >= 1
        validation_dataloaders.append(
            ValidationDataLoader(name=dataset_name, data_path=data_path, dataloader=val_dataloader)
        )

    assert len(train_dataloader) >= 1
    if not validation_dataloaders:
        raise ValueError("At least one validation dataset is required. Please set `data.val_files`.")
    print(f"Size of train dataloader: {len(train_dataloader)}")
    for validation_dataloader in validation_dataloaders:
        print(
            f"Size of val dataloader `{validation_dataloader.name}` ({validation_dataloader.data_path}): "
            f"{len(validation_dataloader.dataloader)}"
        )
    return train_dataloader, validation_dataloaders
