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

import math
import os
from collections import Counter, defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF


_PROMPT_FILTER_KEEP_COLUMN = "__prompt_filter_keep__"
_PROMPT_FILTER_REASON_COLUMN = "__prompt_filter_reason__"


def _parse_dataset_spec(data_path: str) -> tuple[str, Optional[str], str]:
    common_splits = {"train", "test", "validation", "val"}
    data_path = data_path.strip()

    if os.path.isdir(data_path) or os.path.isfile(data_path):
        if "@" in data_path:
            local_path, data_split = data_path.rsplit("@", maxsplit=1)
            return local_path, None, data_split
        return data_path, None, "train"

    data_name: Optional[str] = None
    data_split = "train"
    remote_path = data_path

    # Remote dataset syntax:
    # - repo@config
    # - repo#split
    # - repo@config#split
    if "#" in remote_path:
        remote_path, data_split = remote_path.rsplit("#", maxsplit=1)

    if "@" in remote_path:
        maybe_path, maybe_name = remote_path.rsplit("@", maxsplit=1)
        if "#" in data_path or maybe_name not in common_splits:
            remote_path, data_name = maybe_path, maybe_name
        else:
            remote_path, data_split = maybe_path, maybe_name

    return remote_path, data_name, data_split


def _load_remote_dataset_with_fallback(data_path: str, data_name: Optional[str], data_split: str):
    try:
        return load_dataset(data_path, data_name, split=data_split)
    except ValueError as exc:
        # Backward compatibility: some launch scripts historically used `repo@split`
        # for HF datasets that only expose the default config but custom split names.
        if (
            data_name is not None
            and data_split == "train"
            and "BuilderConfig" in str(exc)
            and "not found" in str(exc)
        ):
            return load_dataset(data_path, split=data_name)
        raise


def _normalize_visual_prompt(prompt_str: str, placeholder: str, num_items: int) -> str:
    if num_items <= 0:
        return prompt_str.replace(placeholder, "")

    current_count = prompt_str.count(placeholder)
    if current_count == num_items:
        return prompt_str

    if current_count == 0:
        prefix = "\n".join([placeholder] * num_items)
        return f"{prefix}\n{prompt_str}" if prompt_str else prefix

    if current_count < num_items:
        missing_prefix = "\n".join([placeholder] * (num_items - current_count))
        return f"{missing_prefix}\n{prompt_str}"

    parts = prompt_str.split(placeholder)
    kept_parts = parts[: num_items + 1]
    tail_text = "".join(parts[num_items + 1 :])
    if tail_text:
        kept_parts[-1] = kept_parts[-1] + tail_text
    return placeholder.join(kept_parts)


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def prepare_image_batch(
    images: list[Union[dict[str, Any], ImageObject, str]],
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    *,
    already_preprocessed: bool = False,
) -> list[ImageObject]:
    prepared_images = []
    for image in images:
        if already_preprocessed and isinstance(image, ImageObject):
            if image.mode != "RGB":
                image = image.convert("RGB")
            prepared_images.append(image)
        else:
            prepared_images.append(process_image(image, min_pixels, max_pixels))
    return prepared_images


def process_video(
    video: str, min_pixels: Optional[int], max_pixels: Optional[int], video_fps: float, return_fps: bool = False
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps)


def _blank_image(image: ImageObject) -> ImageObject:
    return Image.new("RGB", image.size, (255, 255, 255))


def _apply_patch_blackening(
    image: ImageObject,
    patch_size: int,
    black_prob: float,
    rng: np.random.Generator,
) -> ImageObject:
    black_prob = float(np.clip(black_prob, 0.0, 1.0))
    if black_prob <= 0.0:
        return image

    arr = np.array(image).copy()
    h, w = arr.shape[:2]
    patch_size = max(int(patch_size), 1)

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            if float(rng.random()) < black_prob:
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                if arr.ndim == 3:
                    arr[y:y_end, x:x_end, :] = 0
                else:
                    arr[y:y_end, x:x_end] = 0

    return Image.fromarray(arr.astype(np.uint8))


def _apply_patch_mask(
    image: ImageObject,
    mask_ratio: float,
    patch_size: int,
    rng: np.random.Generator,
) -> ImageObject:
    mask_ratio = float(np.clip(mask_ratio, 0.0, 1.0))
    if mask_ratio <= 0.0:
        return image
    if mask_ratio >= 1.0:
        return _blank_image(image)

    arr = np.array(image).copy()
    h, w = arr.shape[:2]
    total_pixels = h * w
    target_pixels = int(total_pixels * mask_ratio)
    if target_pixels <= 0:
        return image

    patch_size = max(int(patch_size), 1)
    patch_h = min(h, patch_size)
    patch_w = min(w, patch_size)
    patch_area = max(patch_h * patch_w, 1)
    expected_patch_num = max(int(math.ceil(target_pixels / patch_area)), 1)
    mask = np.zeros((h, w), dtype=bool)

    max_attempts = expected_patch_num * 4
    for _ in range(max_attempts):
        if int(mask.sum()) >= target_pixels:
            break
        x0 = int(rng.integers(0, max(1, w - patch_w + 1)))
        y0 = int(rng.integers(0, max(1, h - patch_h + 1)))
        mask[y0 : y0 + patch_h, x0 : x0 + patch_w] = True

    current_pixels = int(mask.sum())
    if current_pixels < target_pixels:
        remaining = target_pixels - current_pixels
        unmasked_flat = np.flatnonzero(~mask.reshape(-1))
        if len(unmasked_flat) > 0:
            choose = min(remaining, len(unmasked_flat))
            selected = rng.choice(unmasked_flat, size=choose, replace=False)
            mask.reshape(-1)[selected] = True

    arr[mask] = 255
    return Image.fromarray(arr)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        max_images_per_sample: Optional[int] = None,
        max_estimated_input_length: Optional[int] = None,
        estimated_image_token_stride: int = 28,
        max_train_samples: Optional[int] = None,
        seed: int = 1,
        image_mask_mode: str = "none",
        image_mask_ratio: float = 0.0,
        image_mask_patch_size: int = 32,
        image_mask_seed: Optional[int] = None,
        add_blank_image_for_text: bool = False,
        blank_image_size: int = 512,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_images_per_sample = max_images_per_sample
        self.max_estimated_input_length = max_estimated_input_length
        self.estimated_image_token_stride = estimated_image_token_stride
        self.image_mask_mode = image_mask_mode
        self.image_mask_ratio = image_mask_ratio
        self.image_mask_patch_size = image_mask_patch_size
        self.image_mask_seed = image_mask_seed
        self.add_blank_image_for_text = add_blank_image_for_text
        self.blank_image_size = blank_image_size

        valid_image_mask_modes = {"none", "all_blank", "random_blank", "patch_mask"}
        if self.image_mask_mode not in valid_image_mask_modes:
            raise ValueError(
                f"Unknown image_mask_mode `{self.image_mask_mode}`. "
                f"Expected one of: {sorted(valid_image_mask_modes)}."
            )
        if self.max_images_per_sample is not None and self.max_images_per_sample <= 0:
            raise ValueError("max_images_per_sample should be positive when provided.")
        if self.max_estimated_input_length is not None and self.max_estimated_input_length <= 0:
            raise ValueError("max_estimated_input_length should be positive when provided.")
        if self.estimated_image_token_stride <= 0:
            raise ValueError("estimated_image_token_stride should be positive.")

        data_path, data_name, data_split = _parse_dataset_spec(data_path)
        dataset_source = data_path
        if data_name is not None:
            dataset_source = f"{dataset_source}@{data_name}"
        dataset_source = f"{dataset_source}#{data_split}"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = _load_remote_dataset_with_fallback(data_path, data_name, data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            dataset_with_filter_flags = self.dataset.map(
                self._get_prompt_filter_metadata,
                desc="Inspecting prompt lengths",
                num_proc=filter_overlong_prompts_workers,
            )
            reason_counts = Counter(dataset_with_filter_flags[_PROMPT_FILTER_REASON_COLUMN])
            kept_count = reason_counts.get("ok", 0)
            self.dataset = dataset_with_filter_flags.filter(
                lambda keep: keep,
                input_columns=[_PROMPT_FILTER_KEEP_COLUMN],
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )
            self.dataset = self.dataset.remove_columns([
                _PROMPT_FILTER_KEEP_COLUMN,
                _PROMPT_FILTER_REASON_COLUMN,
            ])
            self._log_prompt_filter_summary(
                dataset_source=dataset_source,
                total_count=sum(reason_counts.values()),
                kept_count=kept_count,
                reason_counts=reason_counts,
            )

        # Randomly sample a subset of training data if max_train_samples is specified
        if max_train_samples is not None and max_train_samples > 0:
            dataset_size = len(self.dataset)
            if max_train_samples < dataset_size:
                # Shuffle with seed for reproducibility, then select first max_train_samples
                self.dataset = self.dataset.shuffle(seed=seed).select(range(max_train_samples))
                print(f"Randomly sampled {max_train_samples} samples from {dataset_size} total samples (seed={seed})")
            else:
                print(f"max_train_samples ({max_train_samples}) >= dataset size ({dataset_size}), using all samples")

    def _get_image_mask_rng(self, index: Optional[int], image_idx: int) -> np.random.Generator:
        if self.image_mask_seed is None:
            return np.random.default_rng()
        # Stable per-sample/image randomness when seed is provided.
        if index is None:
            return np.random.default_rng(self.image_mask_seed + image_idx * 97)
        return np.random.default_rng(self.image_mask_seed + index * 10007 + image_idx * 97)

    def _apply_image_ablation(self, image: ImageObject, index: int, image_idx: int) -> ImageObject:
        if self.image_mask_mode == "none":
            return image

        rng = self._get_image_mask_rng(index=index, image_idx=image_idx)
        if self.image_mask_mode == "all_blank":
            black_prob = float(np.clip(self.image_mask_ratio, 0.0, 1.0))
            if black_prob <= 0.0:
                black_prob = 1.0
            return _apply_patch_blackening(
                image=image,
                patch_size=self.image_mask_patch_size,
                black_prob=black_prob,
                rng=rng,
            )
        if self.image_mask_mode == "random_blank":
            prob = float(np.clip(self.image_mask_ratio, 0.0, 1.0))
            return _blank_image(image) if float(rng.random()) < prob else image
        if self.image_mask_mode == "patch_mask":
            return _apply_patch_mask(
                image=image,
                mask_ratio=self.image_mask_ratio,
                patch_size=self.image_mask_patch_size,
                rng=rng,
            )

        return image

    def _maybe_add_blank_image_to_text(self, example: dict[str, Any]) -> dict[str, Any]:
        if not self.add_blank_image_for_text:
            return example
        if self.processor is None:
            return example
        if self._get_media_items(example, self.image_key) or self._get_media_items(example, self.video_key):
            return example

        converted_example = dict(example)
        prompt_str = str(converted_example.get(self.prompt_key, ""))
        if "<image>" not in prompt_str:
            prompt_str = f"<image>\n{prompt_str}" if prompt_str else "<image>"
        blank_size = max(int(self.blank_image_size), 1)
        converted_example[self.prompt_key] = prompt_str
        converted_example[self.image_key] = [Image.new("RGB", (blank_size, blank_size), (255, 255, 255))]
        return converted_example

    def _resolve_images(self, images: list[Any]) -> list[Any]:
        if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):
            return [os.path.join(self.image_dir, image) for image in images]
        return images

    def _get_media_items(self, example: dict[str, Any], key: str) -> list[Any]:
        items = example.get(key)
        if items is None:
            return []
        if isinstance(items, list):
            return items
        return list(items)

    def _prepare_training_images(
        self,
        images: list[Any],
        *,
        index: Optional[int],
    ) -> list[ImageObject]:
        resolved_images = self._resolve_images(images)
        prepared_images = prepare_image_batch(resolved_images, self.min_pixels, self.max_pixels)
        for image_idx, image in enumerate(prepared_images):
            prepared_images[image_idx] = self._apply_image_ablation(image, index=index, image_idx=image_idx)
        return prepared_images

    def _estimate_image_tokens(self, images: list[ImageObject]) -> int:
        stride = self.estimated_image_token_stride
        total_pixels = sum(image.width * image.height for image in images)
        return math.ceil(total_pixels / (stride * stride))

    def _estimate_prompt_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))

    def _log_prompt_filter_summary(
        self,
        *,
        dataset_source: str,
        total_count: int,
        kept_count: int,
        reason_counts: Counter[str],
    ) -> None:
        dropped_count = total_count - kept_count
        summary_parts = [
            f"Prompt filtering summary for `{dataset_source}`:",
            f"kept {kept_count}/{total_count}",
        ]
        if dropped_count:
            summary_parts.append(f"dropped {dropped_count}")
        print(" ".join(summary_parts))
        for reason in sorted(reason_counts):
            if reason == "ok":
                continue
            print(f"  - {reason}: {reason_counts[reason]}")

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            # Detect Chinese characters in Python for more reliable detection
            has_chinese = any(
                '\u4e00' <= char <= '\u9fa5'
                for char in prompt_str
            )
            prompt_str = format_prompt.render(content=prompt_str, has_chinese=has_chinese)
            # print(f"[DEBUG] prompt_str: {prompt_str}")
            # raise
        images = self._get_media_items(example, self.image_key)
        if images:
            prompt_str = _normalize_visual_prompt(prompt_str, "<image>", len(images))
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image", "image": images[i - 1]})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        videos = self._get_media_items(example, self.video_key)
        if videos:
            prompt_str = _normalize_visual_prompt(prompt_str, "<video>", len(videos))
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video", "video": videos[i - 1]})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _get_prompt_filter_metadata(self, example: dict[str, Any]) -> dict[str, Any]:
        example = self._maybe_add_blank_image_to_text(example)
        messages = self._build_messages(example)
        images = self._get_media_items(example, self.image_key)
        if images:
            if self.max_images_per_sample is not None and len(images) > self.max_images_per_sample:
                return {
                    _PROMPT_FILTER_KEEP_COLUMN: False,
                    _PROMPT_FILTER_REASON_COLUMN: "too_many_images",
                }
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            processed_images = self._prepare_training_images(images, index=None)
            if self.max_estimated_input_length is not None:
                estimated_length = self._estimate_prompt_tokens(prompt) + self._estimate_image_tokens(processed_images)
                if estimated_length > self.max_estimated_input_length:
                    return {
                        _PROMPT_FILTER_KEEP_COLUMN: False,
                        _PROMPT_FILTER_REASON_COLUMN: "estimated_total_length",
                    }

            model_inputs = self.processor(
                images=processed_images,
                text=[prompt],
                add_special_tokens=False,
                return_tensors="pt",
            )
            keep = model_inputs["input_ids"].size(-1) <= self.max_prompt_length
            return {
                _PROMPT_FILTER_KEEP_COLUMN: keep,
                _PROMPT_FILTER_REASON_COLUMN: "ok" if keep else "actual_prompt_length",
            }
        videos = self._get_media_items(example, self.video_key)
        if videos:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            if self.max_estimated_input_length is not None:
                if self._estimate_prompt_tokens(prompt) > self.max_estimated_input_length:
                    return {
                        _PROMPT_FILTER_KEEP_COLUMN: False,
                        _PROMPT_FILTER_REASON_COLUMN: "estimated_total_length",
                    }
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            keep = model_inputs["input_ids"].size(-1) <= self.max_prompt_length
            return {
                _PROMPT_FILTER_KEEP_COLUMN: keep,
                _PROMPT_FILTER_REASON_COLUMN: "ok" if keep else "actual_prompt_length",
            }
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            text_token_count = self._estimate_prompt_tokens(prompt)
            if self.max_estimated_input_length is not None and text_token_count > self.max_estimated_input_length:
                return {
                    _PROMPT_FILTER_KEEP_COLUMN: False,
                    _PROMPT_FILTER_REASON_COLUMN: "estimated_total_length",
                }
            keep = text_token_count <= self.max_prompt_length
            return {
                _PROMPT_FILTER_KEEP_COLUMN: keep,
                _PROMPT_FILTER_REASON_COLUMN: "ok" if keep else "actual_prompt_length",
            }

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        return bool(self._get_prompt_filter_metadata(example)[_PROMPT_FILTER_KEEP_COLUMN])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        example = self._maybe_add_blank_image_to_text(example)
        prompt_example = dict(example)
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)
        raw_messages = messages

        images = self._get_media_items(example, self.image_key)
        if images:
            example.pop(self.image_key, None)
            processed_images = self._prepare_training_images(images, index=index)
            # Keep train/ref/critic inputs and rollout/validation chat inputs on the same processed images.
            downstream_images = processed_images
            prompt_example[self.image_key] = downstream_images
            raw_messages = self._build_messages(prompt_example)
            prompt = self.processor.apply_chat_template(raw_messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.processor(images=processed_images, text=[prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {
                "images": downstream_images,
                "images_are_preprocessed": True,
            }
        else:
            example.pop(self.image_key, None)

        videos = self._get_media_items(example, self.video_key)
        if not images and videos:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            example.pop(self.video_key, None)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]
            prompt_example[self.video_key] = videos
            raw_messages = self._build_messages(prompt_example)

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        elif not images:
            example.pop(self.video_key, None)
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {}

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_messages"] = raw_messages
        example["raw_prompt"] = prompt
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        return example
