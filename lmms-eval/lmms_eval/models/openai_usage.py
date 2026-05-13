import json
import os
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_OPENAI_PRICING_USD_PER_1M_TOKENS = {
    "gpt-5.4": {"input": 2.50, "output": 15.00},
}

GENERATION_ERROR_PREFIX = "[[LMMS_EVAL_GENERATION_ERROR]]"


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"none", "null"}:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _model_pricing_key(model_version: str) -> str:
    model = str(model_version or "").lower().strip()
    if "/" in model:
        model = model.rsplit("/", 1)[-1]
    return model


def resolve_openai_pricing(
    model_version: str,
    input_price_per_million: Any = None,
    output_price_per_million: Any = None,
) -> Dict[str, Any]:
    input_override = _coerce_float(input_price_per_million)
    output_override = _coerce_float(output_price_per_million)
    pricing_key = _model_pricing_key(model_version)
    default = DEFAULT_OPENAI_PRICING_USD_PER_1M_TOKENS.get(pricing_key, {})

    input_price = input_override
    if input_price is None:
        input_price = default.get("input")

    output_price = output_override
    if output_price is None:
        output_price = default.get("output")

    return {
        "model": model_version,
        "pricing_key": pricing_key,
        "currency": "USD",
        "unit": "1M tokens",
        "input_per_1m_tokens": input_price,
        "output_per_1m_tokens": output_price,
        "source": (
            "model_args"
            if input_override is not None or output_override is not None
            else "default_table"
        ),
    }


def _object_to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return value.dict()
        except Exception:
            pass
    return {}


def extract_usage_dict(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    return _object_to_dict(usage)


def make_generation_error_response(
    reason: str,
    *,
    finish_reason: Any = None,
    max_output_tokens: Any = None,
    output_tokens: Any = None,
) -> str:
    parts = [GENERATION_ERROR_PREFIX, f"reason={reason}"]
    if finish_reason is not None:
        parts.append(f"finish_reason={finish_reason}")
    if max_output_tokens is not None:
        parts.append(f"max_output_tokens={max_output_tokens}")
    if output_tokens is not None:
        parts.append(f"output_tokens={output_tokens}")
    return " ".join(parts)


def is_generation_error_response(text: Any) -> bool:
    return isinstance(text, str) and text.startswith(GENERATION_ERROR_PREFIX)


def _cost(tokens: Optional[int], price_per_million: Optional[float]) -> Optional[float]:
    if tokens is None or price_per_million is None:
        return None
    return (tokens * price_per_million) / 1_000_000


def build_token_usage_record(
    *,
    doc_uuid: str,
    model_version: str,
    pricing: Dict[str, Any],
    usage: Optional[Dict[str, Any]] = None,
    response_text: str = "",
    latency: float = 0.0,
    source: str = "api",
    cache_hit: bool = False,
) -> Dict[str, Any]:
    usage = usage or {}
    prompt_tokens = _coerce_int(
        usage.get("prompt_tokens", usage.get("input_tokens"))
    )
    completion_tokens = _coerce_int(
        usage.get("completion_tokens", usage.get("output_tokens"))
    )
    total_tokens = _coerce_int(usage.get("total_tokens"))

    estimated = False
    if completion_tokens is None and response_text:
        completion_tokens = len(response_text.split())
        estimated = True
        source = f"{source}+estimated_output"

    if total_tokens is None:
        known_tokens = [
            value for value in (prompt_tokens, completion_tokens) if value is not None
        ]
        total_tokens = sum(known_tokens) if known_tokens else None

    input_cost = _cost(prompt_tokens, pricing.get("input_per_1m_tokens"))
    output_cost = _cost(completion_tokens, pricing.get("output_per_1m_tokens"))
    total_cost = None
    if input_cost is not None or output_cost is not None:
        total_cost = (input_cost or 0.0) + (output_cost or 0.0)

    return {
        "doc_uuid": doc_uuid,
        "model": model_version,
        "source": source,
        "cache_hit": cache_hit,
        "estimated": estimated,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
        "latency_seconds": latency,
        "pricing": pricing,
        "raw_usage": usage,
    }


def summarize_token_usage_entries(
    entries: Iterable[Dict[str, Any]],
    pricing: Optional[Dict[str, Any]] = None,
    model_version: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    unique_entries: Dict[str, Dict[str, Any]] = {}
    anonymous_entries: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        doc_uuid = entry.get("doc_uuid")
        if doc_uuid:
            unique_entries[str(doc_uuid)] = entry
        else:
            anonymous_entries.append(entry)

    all_entries = list(unique_entries.values()) + anonymous_entries
    if not all_entries:
        return None

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    input_cost = 0.0
    output_cost = 0.0
    total_cost = 0.0
    missing_usage = 0
    estimated_usage = 0
    cache_hits = 0

    first_pricing = pricing
    first_model = model_version
    for entry in all_entries:
        if first_pricing is None and isinstance(entry.get("pricing"), dict):
            first_pricing = entry["pricing"]
        if first_model is None and entry.get("model"):
            first_model = entry["model"]
        if entry.get("cache_hit"):
            cache_hits += 1
        if entry.get("estimated"):
            estimated_usage += 1

        entry_input_tokens = _coerce_int(entry.get("input_tokens"))
        entry_output_tokens = _coerce_int(entry.get("output_tokens"))
        entry_total_tokens = _coerce_int(entry.get("total_tokens"))

        if entry_input_tokens is None and entry_output_tokens is None:
            missing_usage += 1
            continue

        input_tokens += entry_input_tokens or 0
        output_tokens += entry_output_tokens or 0
        total_tokens += (
            entry_total_tokens
            if entry_total_tokens is not None
            else (entry_input_tokens or 0) + (entry_output_tokens or 0)
        )
        input_cost += _coerce_float(entry.get("input_cost_usd")) or 0.0
        output_cost += _coerce_float(entry.get("output_cost_usd")) or 0.0
        total_cost += _coerce_float(entry.get("total_cost_usd")) or 0.0

    return {
        "model": first_model,
        "pricing": first_pricing,
        "sample_count": len(all_entries),
        "counted_sample_count": len(all_entries) - missing_usage,
        "missing_usage_count": missing_usage,
        "estimated_usage_count": estimated_usage,
        "cache_hit_count": cache_hits,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
    }


def summarize_token_usage_from_samples(samples: Any) -> Optional[Dict[str, Any]]:
    if not samples:
        return None

    entries: List[Dict[str, Any]] = []
    for task_samples in samples.values():
        for sample in task_samples:
            if not isinstance(sample, dict):
                continue
            usage = sample.get("token_usage")
            if isinstance(usage, dict):
                entries.append(usage)
            elif isinstance(usage, list):
                entries.extend(item for item in usage if isinstance(item, dict))
    return summarize_token_usage_entries(entries)


def format_token_usage_summary(summary: Optional[Dict[str, Any]]) -> str:
    if not summary:
        return ""
    pricing = summary.get("pricing") or {}
    input_price = pricing.get("input_per_1m_tokens")
    output_price = pricing.get("output_per_1m_tokens")
    price_text = ""
    if input_price is not None and output_price is not None:
        price_text = (
            f"; rates=${input_price:g}/1M input, ${output_price:g}/1M output"
        )
    missing = summary.get("missing_usage_count") or 0
    missing_text = f"; missing_usage={missing}" if missing else ""
    return (
        "Token usage/cost: "
        f"input={summary.get('input_tokens', 0)} tokens "
        f"(${summary.get('input_cost_usd', 0.0):.6f}), "
        f"output={summary.get('output_tokens', 0)} tokens "
        f"(${summary.get('output_cost_usd', 0.0):.6f}), "
        f"total_tokens={summary.get('total_tokens', 0)}, "
        f"total_cost=${summary.get('total_cost_usd', 0.0):.6f}"
        f"{price_text}{missing_text}"
    )


class OpenAITokenUsageMixin:
    def _max_output_tokens_from_payload(self, payload: Dict[str, Any]) -> Optional[int]:
        if not isinstance(payload, dict):
            return None
        return _coerce_int(
            payload.get("max_completion_tokens", payload.get("max_tokens"))
        )

    def _finish_reason_from_response(self, response: Any) -> Optional[str]:
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        return getattr(choices[0], "finish_reason", None)

    def _message_content_from_response(self, response: Any) -> Optional[str]:
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if content is None:
            return None
        return str(content)

    def _response_text_or_error(self, response: Any, payload: Dict[str, Any]) -> str:
        usage = extract_usage_dict(response)
        output_tokens = _coerce_int(
            usage.get("completion_tokens", usage.get("output_tokens"))
        )
        max_output_tokens = self._max_output_tokens_from_payload(payload)
        finish_reason = self._finish_reason_from_response(response)
        content = self._message_content_from_response(response)

        hit_output_limit = finish_reason in {"length", "max_tokens"}
        if (
            not hit_output_limit
            and max_output_tokens is not None
            and output_tokens is not None
            and output_tokens >= max_output_tokens
        ):
            hit_output_limit = True

        if hit_output_limit:
            return make_generation_error_response(
                "max_output_tokens",
                finish_reason=finish_reason,
                max_output_tokens=max_output_tokens,
                output_tokens=output_tokens,
            )

        if content is None or not content.strip():
            return make_generation_error_response(
                "empty_message_content",
                finish_reason=finish_reason,
                max_output_tokens=max_output_tokens,
                output_tokens=output_tokens,
            )

        return content

    def _cached_usage_error_response(self, doc_uuid: str) -> Optional[str]:
        cached_usage = self.token_usage_cache.get(doc_uuid)
        if not isinstance(cached_usage, dict):
            return None
        return make_generation_error_response(
            "missing_cached_response",
            output_tokens=cached_usage.get("output_tokens"),
        )

    def _init_token_usage_tracking(
        self,
        *,
        input_token_price_per_million: Any = None,
        output_token_price_per_million: Any = None,
    ) -> None:
        self.token_pricing = resolve_openai_pricing(
            self.model_version,
            input_token_price_per_million,
            output_token_price_per_million,
        )
        self.token_usage_by_doc_uuid: Dict[str, Dict[str, Any]] = {}
        self.token_usage_cache: Dict[str, Dict[str, Any]] = {}
        self.token_usage_summary: Dict[str, Any] = {}
        self.token_usage_persistent_file = None

        if getattr(self, "continual_mode", False) and getattr(
            self, "response_persistent_folder", None
        ):
            self.token_usage_persistent_file = os.path.join(
                self.response_persistent_folder,
                f"{self.model_version}_usage.json",
            )
            os.makedirs(
                os.path.dirname(self.token_usage_persistent_file), exist_ok=True
            )
            if os.path.exists(self.token_usage_persistent_file):
                try:
                    with open(
                        self.token_usage_persistent_file, "r", encoding="utf-8"
                    ) as f:
                        usage_cache = json.load(f)
                    if isinstance(usage_cache, dict):
                        self.token_usage_cache = usage_cache
                except Exception:
                    self.token_usage_cache = {}

    def _record_token_usage(
        self,
        *,
        doc_uuid: str,
        response: Any = None,
        response_text: str = "",
        latency: float = 0.0,
        source: str = "api",
        cache_hit: bool = False,
    ) -> Dict[str, Any]:
        if cache_hit:
            cached_usage = self.token_usage_cache.get(doc_uuid)
            if isinstance(cached_usage, dict):
                entry = dict(cached_usage)
                entry["cache_hit"] = True
                entry["source"] = "usage_cache"
            else:
                entry = build_token_usage_record(
                    doc_uuid=doc_uuid,
                    model_version=self.model_version,
                    pricing=self.token_pricing,
                    usage={},
                    response_text="",
                    latency=0.0,
                    source="response_cache_without_usage",
                    cache_hit=True,
                )
        else:
            entry = build_token_usage_record(
                doc_uuid=doc_uuid,
                model_version=self.model_version,
                pricing=self.token_pricing,
                usage=extract_usage_dict(response),
                response_text=response_text,
                latency=latency,
                source=source,
                cache_hit=False,
            )
            self.token_usage_cache[doc_uuid] = entry

        self.token_usage_by_doc_uuid[doc_uuid] = entry
        self._refresh_token_usage_summary()
        return entry

    def _save_token_usage_cache(self) -> None:
        if not self.token_usage_persistent_file:
            return
        os.makedirs(os.path.dirname(self.token_usage_persistent_file), exist_ok=True)
        with open(self.token_usage_persistent_file, "w", encoding="utf-8") as f:
            json.dump(self.token_usage_cache, f, ensure_ascii=False, indent=2)

    def _refresh_token_usage_summary(self) -> Dict[str, Any]:
        summary = summarize_token_usage_entries(
            self.token_usage_by_doc_uuid.values(),
            pricing=self.token_pricing,
            model_version=self.model_version,
        )
        self.token_usage_summary = summary or {}
        return self.token_usage_summary

    def get_token_usage_for_request_args(
        self, request_args: Any
    ) -> Optional[Dict[str, Any]]:
        if not request_args or len(request_args) < 3:
            return None
        doc_id, task, split = request_args[-3:]
        doc_uuid = f"{task}___{split}___{doc_id}"
        return self.token_usage_by_doc_uuid.get(doc_uuid) or self.token_usage_cache.get(
            doc_uuid
        )
