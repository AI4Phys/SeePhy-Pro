import ast
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger as eval_logger
from PIL import Image
from jinja2 import Template
from datasets.features import features as hf_features

from lmms_eval.models.openai_usage import is_generation_error_response
from lmms_eval.tasks.seephys_pro.seephys_pro_evals import (
    SeephysProEvaluator,
    load_seephys_pro_config,
)

if "List" not in hf_features._FEATURE_TYPES:
    hf_features.register_feature(hf_features.Sequence, "List")

config = load_seephys_pro_config()
seephys_pro_evaluator = SeephysProEvaluator()


def _parse_visual_key_info_list(doc: Dict[str, Any]) -> List[str]:
    raw_value = doc.get("visual key information list")
    if raw_value is None:
        raw_value = doc.get("visual_key_information_list")
    if raw_value is None:
        raw_value = doc.get("visual_key_info")

    if raw_value is None:
        return []

    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]

    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass
        parts = re.split(r"[\n;|]+", text)
        return [part.strip() for part in parts if part.strip()]

    text = str(raw_value).strip()
    return [text] if text else []


def _doc_index(doc: Dict[str, Any]) -> Any:
    return doc.get("index", doc.get("idx", "N/A"))


def _decode_image_from_item(item: Any) -> Image.Image:
    if isinstance(item, Image.Image):
        return item

    if isinstance(item, dict):
        image_path = item.get("path")
        image_bytes = item.get("bytes")

        if isinstance(image_path, str) and image_path:
            return Image.open(image_path)

        if image_bytes:
            if isinstance(image_bytes, memoryview):
                image_bytes = image_bytes.tobytes()
            if not isinstance(image_bytes, (bytes, bytearray)):
                raise TypeError(
                    "Expected image bytes to be bytes-like, "
                    f"but got {type(image_bytes)}"
                )
            return Image.open(BytesIO(image_bytes))

        raise ValueError("Image dict must contain non-empty 'path' or 'bytes'.")

    raise TypeError(f"Unsupported image item type: {type(item)}")


def seephys_pro_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    if "images" not in doc or not doc["images"]:
        eval_logger.warning(
            f"Document index {_doc_index(doc)} has no 'images' field or it is empty."
        )
        return []

    image_list = doc["images"]

    if isinstance(image_list, np.ndarray):
        image_list = image_list.tolist()

    if not isinstance(image_list, list):
        image_list = [image_list]

    processed_images = []
    for i, image in enumerate(image_list):
        try:
            decoded = _decode_image_from_item(image)
        except Exception as e:
            raise TypeError(
                f"Failed to decode image item {i} for index {_doc_index(doc)}: {e}"
            ) from e

        if decoded.mode != "RGB":
            decoded = decoded.convert("RGB")
        processed_images.append(decoded)

    return processed_images


def seephys_pro_doc_to_text(
    doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None
) -> str:
    _ = lmms_eval_specific_kwargs
    question = doc.get("question") or doc.get("problem", "")
    if not isinstance(question, str) or str(question).lower() == "nan":
        question = ""

    # 自动检测语言：检查问题中是否包含中文字符
    def contains_chinese(text: str) -> bool:
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                return True
        return False

    lang = doc.get("language")
    if lang is None:
        # 如果没有 language 字段，自动检测
        lang = "Chinese" if contains_chinese(question) else "English"

    sig_figs = doc.get("sig_figs")

    try:
        if sig_figs and (
            isinstance(sig_figs, (str, int, float)) and not np.isnan(float(sig_figs))
        ):
            sf_str = str(int(float(sig_figs)))
            if lang == "English":
                question += (
                    f" The final answer should retain {sf_str} significant figures."
                )
            else:
                question += f" 最终答案应保留{sf_str}位有效数字。"
    except (ValueError, TypeError):
        pass

    def _normalize_aux_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            text = value.strip()
            if text.lower() in {"", "nan", "none", "null"}:
                return ""
            return text
        if isinstance(value, (list, tuple)):
            parts = [_normalize_aux_text(v) for v in value]
            return "\n".join([part for part in parts if part])
        if isinstance(value, dict):
            parts = []
            for key, val in value.items():
                norm = _normalize_aux_text(val)
                if norm:
                    parts.append(f"{key}: {norm}")
            return "\n".join(parts)
        text = str(value).strip()
        if text.lower() in {"", "nan", "none", "null"}:
            return ""
        return text

    use_aux_hints = os.getenv("SEEPHYS_PRO_USE_AUX_HINTS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if use_aux_hints:
        visual_key_info = _normalize_aux_text(doc.get("visual_key_info"))
        ablated_info = _normalize_aux_text(doc.get("ablated_info"))
        aux_blocks: List[str] = []
        if visual_key_info:
            aux_blocks.append(f"[Visual Key Info]\n{visual_key_info}")
        if ablated_info:
            aux_blocks.append(f"[Ablated Info]\n{ablated_info}")
        if aux_blocks:
            if lang == "English":
                question += "\n\nReference visual text hints:\n" + "\n\n".join(
                    aux_blocks
                )
            else:
                question += "\n\n可参考的视觉文字线索：\n" + "\n\n".join(aux_blocks)

    has_chinese = True if lang == "Chinese" else False
    template_path = os.environ.get("TEMPLATE_PATH", "math_adapt.jinja")

    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            template_content = f.read()
        template = Template(template_content)
        prompt_str = template.render(content=question, has_chinese=has_chinese)
    else:
        if has_chinese:
            post_prompt = (
                r"请给出简洁的推理过程，"
                r"最终答案只放在 <answer> </answer> 中。"
                r"若为选择题，仅给字母（多选连写如 AB）；若为数值题，请给出数值和后面的单位。"
                r"无论是否确定，都必须给出一个非空的 <answer>...</answer>。"
            )
        else:
            post_prompt = (
                r"Provide a Concise reasoning process. "
                r"Put ONLY the final answer inside <answer> </answer>. "
                r"For multiple choice, output just the letter(s) (e.g., AB for multi-select). "
                r"For numeric answers, give the value and the unit behind. "
                r"Always provide a non-empty <answer>...</answer>, even when uncertain."
            )
        prompt_str = question + "\n" + post_prompt

    return prompt_str


def seephys_pro_doc_to_messages(
    doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    text_content = seephys_pro_doc_to_text(doc, lmms_eval_specific_kwargs)
    image_list = seephys_pro_doc_to_visual(doc)

    content: List[Dict[str, Any]] = [{"type": "text", "text": text_content}]
    for img in image_list:
        content.append({"type": "image", "url": img})

    return [{"role": "user", "content": content}]


def seephys_pro_process_results(
    doc: Dict[str, Any], results: List[str]
) -> Dict[str, Any]:
    true_false = 0
    key_info_hit_rate = None
    key_info_judge_raw = None
    key_info_prompt_chars = None
    key_info_pred_chars = None
    key_info_items = None
    generation_error = None
    if isinstance(config, dict):
        config.setdefault("metadata", {})
        env_metric_output_mode = os.getenv("METRIC_OUTPUT_MODE")
        if env_metric_output_mode:
            config["metadata"]["metric_output_mode"] = env_metric_output_mode

    metric_output_mode = config.get("metadata", {}).get("metric_output_mode", "acc")
    if metric_output_mode not in ["acc", "key_info_hit_rate", "both"]:
        eval_logger.warning(
            f"Invalid metric_output_mode={metric_output_mode}, fallback to both."
        )
        metric_output_mode = "both"

    if not results:
        eval_logger.error(
            f"Received empty results list for index {doc.get('index', 'N/A')}."
        )
    else:
        prediction = results[0].strip()
        doc["filtered_prediction"] = str(prediction)
        raw_prediction = doc.get("raw_prediction", "")
        if raw_prediction is None:
            raw_prediction = ""
        doc["raw_prediction"] = str(raw_prediction)
        if raw_prediction and "<answer>" in str(raw_prediction).lower():
            doc["prediction"] = str(raw_prediction)
        else:
            doc["prediction"] = str(prediction)
        doc["answer"] = str(doc.get("answer", ""))

        if is_generation_error_response(doc["prediction"]) or is_generation_error_response(
            doc["raw_prediction"]
        ):
            generation_error = doc["prediction"]
            eval_logger.warning(
                "Document index {} has generation error marker. Directly marked wrong.",
                doc.get("index", "N/A"),
            )

        quick_extract = config.get("metadata", {}).get("quick_extract")
        if quick_extract is None:
            raise ValueError(
                "Config 'metadata.quick_extract' not found in seephys_pro.yaml"
            )

        if generation_error:
            true_false = 0
        else:
            if quick_extract:
                tmp = seephys_pro_evaluator.SeephysPro_process_line(doc)
                true_false = tmp["match"]
            else:
                llm_tmp = seephys_pro_evaluator.SeephysPro_auxeval(doc)
                true_false = llm_tmp["res"]

        if metric_output_mode in ["key_info_hit_rate", "both"]:
            key_info_list = _parse_visual_key_info_list(doc)
            if generation_error:
                key_info_hit_rate = 0.0
            elif not key_info_list:
                key_info_hit_rate = 0.0
            elif quick_extract:
                eval_logger.warning(
                    "quick_extract 模式下不支持 key info judge，返回 0.0"
                )
                key_info_hit_rate = 0.0
            else:
                raw_pred_for_key_info = doc.get("raw_prediction") or prediction
                key_info_eval = seephys_pro_evaluator.SeephysPro_key_info_eval(
                    doc, key_info_list, raw_pred_for_key_info
                )
                key_info_hit_rate = float(key_info_eval.get("res", 0.0))
                key_info_judge_raw = key_info_eval.get("raw")
                key_info_prompt_chars = key_info_eval.get("prompt_chars")
                key_info_pred_chars = key_info_eval.get("pred_chars")
                key_info_items = key_info_eval.get("key_info_items")

    eval_result = {
        "index": doc.get("index", -1),
        "true_false": true_false,
        "level": doc.get("level"),
        "discipline": doc.get("discipline"),
        "domain": doc.get("domain"),
        "field": doc.get("field"),
        "answer_type": doc.get("answer_type", ""),
        "unit": doc.get("unit", ""),
        "error_range": doc.get("error_range", ""),
        "coordinate_gridscape": doc.get("coordinate_gridscape", ""),
        "answer": doc.get("answer", ""),
        "key_info_hit_rate": key_info_hit_rate,
        "key_info_judge_raw": key_info_judge_raw
        if metric_output_mode in ["key_info_hit_rate", "both"]
        else None,
        "key_info_prompt_chars": key_info_prompt_chars
        if metric_output_mode in ["key_info_hit_rate", "both"]
        else None,
        "key_info_pred_chars": key_info_pred_chars
        if metric_output_mode in ["key_info_hit_rate", "both"]
        else None,
        "key_info_items": key_info_items
        if metric_output_mode in ["key_info_hit_rate", "both"]
        else None,
        "generation_error": generation_error,
    }

    metrics: Dict[str, Any] = {}
    if metric_output_mode in ["acc", "both"]:
        metrics["eval_results"] = eval_result
        metrics["eval_results_by_discipline"] = eval_result
    if metric_output_mode in ["key_info_hit_rate", "both"]:
        metrics["key_info_hit_rate"] = (
            key_info_hit_rate if key_info_hit_rate is not None else 0.0
        )

    return metrics


def seephys_pro_aggregate_results(results: List[Dict[str, Any]]) -> float:
    if not results:
        eval_logger.warning("Aggregating empty results list. Returning 0.0")
        return 0.0

    subset_to_eval_samples: Dict[Any, List[Dict[str, Any]]] = {}
    for result in results:
        subset = result.get("discipline", "unknown")
        subset_to_eval_samples.setdefault(subset, []).append(result)

    evaluation_result = {}
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        hit = [x["true_false"] for x in sub_eval_samples]
        acc = float(np.mean(hit)) if hit else 0.0
        evaluation_result[subset] = {"num": len(sub_eval_samples), "acc": round(acc, 5)}

    overall_acc = float(np.mean([x["true_false"] for x in results]))

    printable = {k: v for k, v in evaluation_result.items()}
    printable["Overall"] = {"num": len(results), "acc": round(overall_acc, 5)}
    print(printable)
    return overall_acc


def seephys_pro_aggregate_key_info_hit_rate(results: List[float]) -> float:
    if not results:
        eval_logger.warning("Aggregating empty key info list. Returning 0.0")
        return 0.0
    normalized = [float(x) for x in results]
    return float(sum(normalized) / len(normalized))


def seephys_pro_aggregate_results_by_discipline(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not results:
        eval_logger.warning(
            "Aggregating empty results list. Returning empty breakdown."
        )
        return {}

    subset_to_eval_samples: Dict[Any, List[Dict[str, Any]]] = {}
    for result in results:
        subset = result.get("discipline", "unknown")
        subset_to_eval_samples.setdefault(subset, []).append(result)

    evaluation_result: Dict[str, Dict[str, Any]] = {}
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        hit = [x["true_false"] for x in sub_eval_samples]
        acc = float(np.mean(hit)) if hit else 0.0
        evaluation_result[subset] = {"num": len(sub_eval_samples), "acc": round(acc, 5)}

    overall_acc = float(np.mean([x["true_false"] for x in results]))
    evaluation_result["Overall"] = {"num": len(results), "acc": round(overall_acc, 5)}
    return evaluation_result
