import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional
import threading

import yaml
from loguru import logger as eval_logger
from openai import AzureOpenAI, OpenAI
from sympy import Eq, latex, simplify, sympify
from sympy.parsing.latex import parse_latex

FAIL_MSG = "Failed to obtain answer via API."


def load_seephys_pro_config() -> Dict[str, Any]:
    config_dir = Path(__file__).parent
    template_path = config_dir / "_default_template_seephys_pro.yaml"
    fallback_path = config_dir / "seephys_pro.yaml"

    config_path = template_path if template_path.exists() else fallback_path
    eval_logger.info(f"Loading SeePhys-Pro config from: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_data = f.readlines()
        safe_data = []
        for line in raw_data:
            if "!function" not in line:
                safe_data.append(line)

    config_data = "".join(safe_data)
    if not config_data.strip():
        eval_logger.error(
            f"Config file {config_path} is empty or only contains !function tags."
        )
        return {"metadata": {}}

    try:
        config = yaml.safe_load(config_data)
        if "metadata" not in config:
            eval_logger.warning(
                f"'metadata' block not found in {config_path}. Using empty metadata."
            )
            config["metadata"] = {}
        return config
    except yaml.YAMLError as e:
        eval_logger.error(f"Error parsing YAML config {config_path}: {e}")
        return {"metadata": {}}


config = load_seephys_pro_config()

if "metadata" not in config:
    raise ValueError(
        "Could not load metadata from seephys_pro.yaml. Check file content and permissions."
    )


class SeephysProEvaluator:
    def __init__(self):
        self.judger_model = None
        self.judge_fallback_model = None
        self.headers = None
        self.client = None
        self._client_kind = None
        self._api_key = None
        self._api_url = None
        self._azure_endpoint = None
        self._azure_version = None
        self._client_local = threading.local()
        self._missing_api_key = False

        if not config["metadata"].get("quick_extract", False):
            self.judger_model = os.getenv(
                "JUDGE_MODEL_NAME", config["metadata"].get("eval_model_name")
            )
            self.judge_fallback_model = os.getenv(
                "JUDGE_FALLBACK_MODEL",
                config["metadata"].get("judge_fallback_model", ""),
            )

            # 优先检查 Azure 环境变量
            azure_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            azure_endpoint = os.getenv("AZURE_OPENAI_API_BASE")
            azure_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

            if azure_endpoint and "openai.azure.com" in azure_endpoint:
                if not azure_key:
                    self._missing_api_key = True
                    eval_logger.warning(
                        "SeePhys-Pro Azure judge endpoint is set but no API key was found. "
                        "Judge-based metrics require AZURE_OPENAI_API_KEY or OPENAI_API_KEY."
                    )
                else:
                    eval_logger.info(
                        "Initializing AzureOpenAI client for SeePhys-Pro judge..."
                    )
                    self._client_kind = "azure"
                    self._api_key = azure_key
                    self._azure_endpoint = azure_endpoint
                    self._azure_version = azure_version
            else:
                api_url = os.getenv(
                    "OPENAI_API_BASE",
                    os.getenv("OPENAI_BASE_URL", os.getenv("JUDGE_BASE_URL", "")),
                )
                api_key = os.getenv("OPENAI_API_KEY", os.getenv("JUDGE_API_KEY", ""))
                if not api_key:
                    self._missing_api_key = True
                    eval_logger.warning(
                        "SeePhys-Pro judge API key is not set. Task loading will continue; "
                        "judge-based metrics require OPENAI_API_KEY, JUDGE_API_KEY, or Azure OpenAI settings."
                    )
                else:
                    self._client_kind = "openai"
                    self._api_key = api_key
                    self._api_url = api_url

            # Initialize a client for the main thread; worker threads will lazily create their own.
            if self._client_kind:
                self.client = self._create_client()

    def _create_client(self):
        if self._client_kind == "azure":
            if not self._azure_endpoint:
                return None
            return AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._azure_version,
            )
        if self._client_kind == "openai":
            return OpenAI(api_key=self._api_key, base_url=self._api_url)
        return None

    def _get_client(self):
        client = getattr(self._client_local, "client", None)
        if client is None:
            client = self._create_client()
            self._client_local.client = client
        return client

    def judger_generate(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,  # 太短了的话gpt5之类的推理模型会输出空响应。
        n: int = 1,
        patience: int = 3,
        sleep_time: int = 0,
        return_error: bool = False,
    ) -> str:
        client = self._get_client()
        if not client:
            eval_logger.error(
                "LLM judger client not initialized. Ensure quick_extract=False and API key is set."
            )
            return FAIL_MSG

        messages = [{"role": "user", "content": prompt}]
        is_openrouter = bool(self._api_url and "openrouter.ai" in self._api_url)
        payload = {
            "model": self.judger_model,
            "messages": messages,
            "temperature": temperature,
        }
        if is_openrouter:
            payload["max_tokens"] = max_tokens
        else:
            payload["max_completion_tokens"] = max_tokens
        if self.judger_model:
            lower_model = self.judger_model.lower()
        else:
            lower_model = ""
        is_reasoning_model = any(
            name in lower_model for name in ["o1", "o3", "o4", "gpt-5", "gpt-5.2"]
        )
        if is_reasoning_model:
            payload.pop("temperature", None)
            payload["reasoning_effort"] = "low"
            payload["response_format"] = {"type": "text"}
            if is_openrouter:
                payload["max_tokens"] = max_tokens
                payload.pop("max_completion_tokens", None)
            else:
                payload["max_completion_tokens"] = max_tokens

        last_error = None
        fallback_used = False
        model_fallback_used = False
        while patience > 0:
            patience -= 1
            try:
                response = client.chat.completions.create(**payload)
                content = response.choices[0].message.content
                prediction = content.strip() if isinstance(content, str) else ""
                if prediction:
                    return prediction
                finish_reason = None
                if response.choices and hasattr(response.choices[0], "finish_reason"):
                    finish_reason = response.choices[0].finish_reason
                last_error = f"Empty response; finish_reason={finish_reason}"
                eval_logger.warning(
                    "judge empty response: model={} base_url={} max_tokens={} max_completion_tokens={} prompt_chars={} finish_reason={}",
                    payload.get("model"),
                    self._api_url,
                    payload.get("max_tokens"),
                    payload.get("max_completion_tokens"),
                    len(prompt),
                    finish_reason,
                )
                if is_reasoning_model and not fallback_used:
                    payload.pop("reasoning_effort", None)
                    payload.pop("response_format", None)
                    payload.pop("max_completion_tokens", None)
                    payload["temperature"] = temperature
                    payload["max_tokens"] = max_tokens
                    fallback_used = True
                    if patience == 0:
                        patience = 1
                    continue
                if self.judge_fallback_model and not model_fallback_used:
                    payload["model"] = self.judge_fallback_model
                    model_fallback_used = True
                    if patience == 0:
                        patience = 1
                    continue
            except Exception as e:
                eval_logger.warning(f"LLM judger API call failed: {e}")
                if "Rate limit" not in str(e):
                    eval_logger.error(e)
                last_error = str(e)
                if "Please reduce the length of the messages" in str(e):
                    eval_logger.error("!!Reduce prompt size")
                    new_size = int(len(prompt) * 0.9)
                    prompt = prompt[len(prompt) - new_size :]
                    payload["messages"] = [{"role": "user", "content": prompt}]
                if sleep_time > 0:
                    time.sleep(sleep_time)

        if return_error and last_error:
            return f"{FAIL_MSG} {last_error}"
        return FAIL_MSG

    def get_ICE_scoring(self) -> str:
        return r"""
You are a strict physics answer equivalence judge.

Decide whether [Model Answer] is equivalent to [Standard Answer] for the given [Question].
Return ONLY one character: 1 (equivalent) or 0 (non-equivalent).

Judging rules:
1) Mathematical equivalence counts as equivalent (e.g., algebraic/trigonometric equivalent forms).
2) For numeric answers, enforce required significant figures if the question explicitly asks for them.
3) If unit is required and provided in context, wrong/incompatible unit => 0.
4) For multiple-choice answers, option letters must match exactly after normalization.
5) If [Model Answer] is empty, missing, or clearly irrelevant to the asked quantity => 0.
6) Be conservative: output 1 only when clearly equivalent.

Output format:
- Output ONLY: 0 or 1
- No words, no punctuation, no explanation.

[Question]: A force of 20 N acts on an object of mass 5 kg. What is the acceleration of the object?
[Standard Answer]: 4 m/s²
[Model Answer] : 4
Judgement: 1

[Question]: A projectile is launched at an angle $\\theta$ with initial velocity $v_0$. What is its time of flight before returning to the same height, assuming negligible air resistance and gravitational acceleration $g$?
[Standard Answer]: $$ t = \\frac{{2 v_0 \\sin(\\theta)}}{{g}} $$
[Model Answer] : Extracted Answer: $$ t = \\frac{{2 v_0 \\cos(\\frac{\\pi}{2} - \\theta)}}{{g}} $$
Judgement: 1

[Question]: The position of a particle is given by $x(t) = 3t^2 - 2t + 5$ meters. What is its instantaneous velocity at $t=2$ seconds?
[Standard Answer]: 10 m/s
[Model Answer] : Velocity $v(t) = dx/dt = 6t - 2$. At $t=2s$, $v(2) = 6(2) - 2 = 12 - 2 = 10$. So the velocity is 10 m/s.
Judgement: 1

[Question]: A car travels North at 20 m/s. It then turns and travels East at 20 m/s. What is the magnitude of its change in velocity?
[Standard Answer]: Approximately 28.3 m/s
[Model Answer] : The change in velocity is 0 m/s because the speed is the same.
Judgement: 0

[Question]: An object is thrown horizontally from a height of 20m with an initial speed of 10 m/s. Calculate: (a) the time it takes to hit the ground ($t_g$), and (b) the horizontal distance ($d_x$) it travels before hitting the ground. (Use g = 10 m/s²)
[Standard Answer]: (a) $t_g = 2$ s, (b) $d_x = 20$ m
[Model Answer] : (a) The time to hit the ground $t_g$ is 2 s. (b) The horizontal distance $d_x$ is 10 m.
Judgement: 0

[Question]: An engine performs $1.2 \\times 10^5$ J of work in 2 minutes. What is its average power output in watts?
[Standard Answer]: 1 kW
[Model Answer] : Power = Work / Time = $1.2 \\times 10^5$ J / (2 min * 60 s/min) = $1.2 \\times 10^5$ J / 120 s = 1000 W.
Judgement: 1

[Question]: A resistor has a voltage of 10V across it and a current of 2A flowing through it. What is its resistance and power dissipation?
[Standard Answer]: Resistance R = 5 Ohms , Power P = 20 Watts.
[Model Answer] : The resistance is $R = V/I = 10V / 2A = 5 \Omega$. The power dissipated is $P = VI = 10V \\times 2A = 20W$.
Judgement: 1

[Question]: The displacement of an object in Simple Harmonic Motion (SHM) is given by $x(t) = A \sin(\omega t)$. Determine the equation for its acceleration, $a(t)$.
[Standard Answer]: $$ a(t) = -A\omega^2 \sin(\omega t) $$
[Model Answer] : The acceleration is the second derivative of displacement. $v(t) = A\omega \cos(\omega t)$. $a(t) = A\omega^2 \cos\left(\omega t + \\frac{\pi}{2}\\right)$.
Judgement: 1

[Question]: 给出相对论性粒子总能量 $E$ 的速度展开式（到 $v^4/c^4$ 项）。
[Standard Answer]: $E = mc^2 \left(1 + \frac{v^2}{2c^2} + \frac{3v^4}{8c^4} + \mathcal{O}(v^6/c^6)\right)$
[Model Answer]: $E = \gamma m c^2 = \frac{mc^2}{\sqrt{1 - v^2/c^2}} \approx mc^2 + \frac{1}{2}mv^2 + \frac{3}{8} \frac{mv^4}{c^2}$
Judgement: 1

[Question]: 计算粒子能量 $E$ 穿过势垒 $V_0$ ($E < V_0$) 的透射系数 $T$。
[Standard Answer]: $\ln T \approx \ln 16 + \ln\left(\frac{E}{V_0}\right) + \ln\left(1 - \frac{E}{V_0}\right) - \frac{2d}{\hbar} \sqrt{2m(V_0 - E)}$
[Model Answer]: $T \approx 16 \frac{E}{V_0} \left(1 - \frac{E}{V_0}\right) e^{-2d\sqrt{2m(V_0 - E)}/\hbar}$
Judgement: 1

[Question]: The position of a particle is given by $x(t) = (2t^3 - 3t)$ meters. What is its acceleration at $t=1$ second? The final answer should retain 3 significant figures.
[Standard Answer]: 12.0 m/s²
[Model Answer] : $v(t) = 6t^2 - 3$. $a(t) = 12.1t$. At $t=1s$, $a(1) = 12.1 \\text{ m/s}^2$.
Judgement: 0
---
Now please provide your judgement (0 or 1), DO NOT output explanation:
"""

    def get_key_info_scoring(self) -> str:
        return r"""
You are a strict grader. Given a model answer and a Visual Key Information List, compute a deterministic hit rate.

Rules:
1) Treat each bullet in the Visual Key Information List as one atomic checklist item.
2) Count an item as hit only if the model answer explicitly mentions the same key fact.
   - No inference or paraphrase beyond clear equivalence.
   - If the item contains a symbol/value (e.g., R, OP=3/5 R, theta, q, m), it must be present.
3) Ignore any extra content not in the list.
4) Hit Rate = (hits) / (total items). Output a decimal with up to 3 digits.
5) Only output one line in this exact format:
Hit Rate: <float>

Example:
[Model Answer]: The incline angle is 30 degrees. The block is frictionless.
[Visual Key Information List]:
- The incline angle is labeled as 30 degrees.
- The block is shown without friction symbols.
Hit Rate: 1.0
"""

    def build_seephys_scoring_prompt(self, line: dict, pred: str) -> str:
        query = line.get("question", "")
        gt = line.get("answer", "")

        full_prompt = (
            self.get_ICE_scoring().strip()
            + f"\n[Question]: {query}\n[Standard Answer]: {gt}\n[Model Answer]: {pred}\nJudgement: "
        )
        return full_prompt

    def build_key_info_scoring_prompt(
        self, line: dict, pred: str, key_info_list: list
    ) -> str:
        key_info_text = (
            "\n".join([f"- {item}" for item in key_info_list])
            if key_info_list
            else "- None"
        )
        full_prompt = (
            self.get_key_info_scoring().strip()
            + f"\n[Model Answer]: {pred}\n[Visual Key Information List]:\n{key_info_text}\nHit Rate: "
        )
        return full_prompt

    def _parse_key_info_score(self, response_text: str) -> float:
        if not response_text:
            return 0.0
        matches = re.findall(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", response_text)
        for match in matches:
            try:
                score = float(match)
                if 0.0 <= score <= 1.0:
                    return score
            except Exception:
                continue
        return 0.0

    def _normalize_answer(self, text: str) -> str:
        normalized = str(text).strip()
        normalized = normalized.replace("\n", " ").replace("\t", " ")
        normalized = normalized.replace("\u2212", "-")
        normalized = normalized.replace("，", ",")
        normalized = normalized.replace("：", ":")
        normalized = normalized.replace("\\left", "")
        normalized = normalized.replace("\\right", "")
        normalized = normalized.replace("$", "")
        normalized = re.sub(r"\\text\{(.*?)\}", r"\1", normalized)
        normalized = re.sub(r"\s+", "", normalized)
        if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})*(\.\d+)?", normalized):
            normalized = normalized.replace(",", "")
        normalized = normalized.strip("\"'`。.,;:!?()[]{}")
        return normalized.lower()

    def _normalize_unit(self, unit: str) -> str:
        normalized = str(unit or "")
        normalized = normalized.replace("$", "")
        normalized = re.sub(r"\\(?:mathrm|text)\{(.*?)\}", r"\1", normalized)
        normalized = normalized.replace("\\", "")
        normalized = normalized.replace("{", "").replace("}", "")
        normalized = normalized.replace("·", "*")
        normalized = normalized.replace("μ", "u")
        normalized = normalized.replace("Ω", "ohm")
        normalized = normalized.replace("Ω", "ohm")
        normalized = normalized.replace("ω", "ohm")
        normalized = normalized.lower()
        normalized = normalized.replace("ohms", "ohm")
        normalized = re.sub(r"\s+", "", normalized)
        normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff/*^%_-]", "", normalized)
        return normalized

    def _canonical_option(self, text: str) -> str:
        normalized = self._normalize_answer(text)
        normalized = re.sub(r"[，,;；、/|\s]+", "", normalized)
        if re.fullmatch(r"[a-z]+", normalized):
            dedup_upper = []
            for char in normalized.upper():
                if char not in dedup_upper:
                    dedup_upper.append(char)
            return "".join(sorted(dedup_upper))
        return normalized

    def _extract_numeric_value(self, text: str) -> Optional[float]:
        raw = str(text or "").strip()
        candidates: list[str] = []
        if "=" in raw:
            rhs = raw.rsplit("=", 1)[-1].strip()
            if rhs:
                candidates.append(rhs)
        if "：" in raw or ":" in raw:
            tail = re.split(r"[:：]", raw)[-1].strip()
            if tail:
                candidates.append(tail)
        candidates.append(raw)

        math_blocks = re.findall(r"\$(.*?)\$", raw)
        for block in math_blocks:
            block = block.strip()
            if block:
                candidates.append(block)

        seen: set[str] = set()
        deduped_candidates: list[str] = []
        for item in candidates:
            key = item.strip()
            if key and key not in seen:
                seen.add(key)
                deduped_candidates.append(key)

        for candidate in deduped_candidates:
            normalized = self._normalize_answer(candidate)

            sci_text = candidate.replace(" ", "")
            sci_text = sci_text.replace("×", "*").replace("x", "*")
            sci_match = re.search(
                r"([-+]?\d*\.?\d+)\*?10\^\{?([-+]?\d+)\}?",
                sci_text,
                flags=re.IGNORECASE,
            )
            if sci_match:
                try:
                    base = float(sci_match.group(1))
                    exponent = int(sci_match.group(2))
                    return base * (10**exponent)
                except Exception:
                    pass

            frac_match = re.search(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", candidate)
            if frac_match:
                try:
                    numerator = sympify(frac_match.group(1).replace("^", "**"))
                    denominator = sympify(frac_match.group(2).replace("^", "**"))
                    if denominator != 0:
                        return float((numerator / denominator).evalf())
                except Exception:
                    pass

            if "/" in candidate and re.fullmatch(
                r"\s*[-+]?\d+(?:\.\d+)?\s*/\s*[-+]?\d+(?:\.\d+)?\s*",
                candidate,
            ):
                try:
                    value = sympify(candidate.replace("^", "**"))
                    if getattr(value, "is_number", False):
                        return float(value.evalf())
                except Exception:
                    pass

            plain_match = re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", normalized)
            if plain_match:
                try:
                    return float(normalized)
                except Exception:
                    pass

            try:
                expr = parse_latex(candidate)
                if getattr(expr, "is_number", False):
                    return float(expr.evalf())
            except Exception:
                pass

            try:
                expr_text = candidate.replace("^", "**")
                expr_text = re.sub(r"\\cdot|\\times", "*", expr_text)
                expr_text = expr_text.replace("{", "(").replace("}", ")")
                expr_text = expr_text.replace("\\", "")
                expr = sympify(expr_text)
                if getattr(expr, "is_number", False):
                    return float(expr.evalf())
            except Exception:
                pass

            numbers = re.findall(
                r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?",
                normalized,
            )
            if numbers:
                try:
                    return float(numbers[-1])
                except Exception:
                    continue

        normalized = self._normalize_answer(raw)
        plain_match = re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", normalized)
        if plain_match:
            try:
                return float(normalized)
            except Exception:
                pass
        return None

    def _extract_unit_from_text(self, text: str) -> str:
        raw = str(text or "")
        raw = re.sub(r"<[^>]+>", " ", raw)
        numeric_iter = list(
            re.finditer(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?", raw)
        )
        if numeric_iter:
            tail = raw[numeric_iter[-1].end() :]
        else:
            tail = raw
        return self._normalize_unit(tail)

    def _parse_error_range(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        text = self._normalize_answer(str(value))
        if text in {"", "none", "nan", "null"}:
            return None
        parsed = self._extract_numeric_value(text)
        if parsed is None:
            return None
        return abs(parsed)

    def _option_equal(self, pred: str, ans: str) -> bool:
        pred_canonical = self._canonical_option(pred)
        ans_canonical = self._canonical_option(ans)
        if bool(ans_canonical) and pred_canonical == ans_canonical:
            return True
        if re.fullmatch(r"[A-Z]+", ans_canonical):
            pred_letters = re.findall(r"[A-Za-z]", str(pred))
            if pred_letters:
                dedup_upper = []
                for char in "".join(pred_letters).upper():
                    if char not in dedup_upper:
                        dedup_upper.append(char)
                pred_letter_key = "".join(sorted(dedup_upper))
                return pred_letter_key == ans_canonical
        return False

    def _value_equal(
        self, pred: str, line: dict, raw_prediction_unit: str = ""
    ) -> bool:
        ans = str(line.get("answer", ""))
        gt_unit = self._normalize_unit(str(line.get("unit", "")))
        pred_unit = self._extract_unit_from_text(pred)
        effective_pred_unit = pred_unit or raw_prediction_unit

        if (
            gt_unit
            and effective_pred_unit
            and not (
                effective_pred_unit == gt_unit
                or effective_pred_unit in gt_unit
                or gt_unit in effective_pred_unit
            )
        ):
            return False

        pred_value = self._extract_numeric_value(pred)
        ans_value = self._extract_numeric_value(ans)
        if pred_value is not None and ans_value is not None:
            err = self._parse_error_range(line.get("error_range"))
            if err is not None:
                return abs(pred_value - ans_value) <= err + 1e-12
            return abs(pred_value - ans_value) <= 1e-6
        return self._answers_equal(pred, ans)

    def _extract_answer_candidates(self, response: str) -> list[str]:
        text = str(response or "")
        candidates: list[str] = []
        tag_matches = re.findall(
            r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE
        )
        for item in tag_matches:
            cleaned = item.strip()
            if cleaned:
                candidates.append(cleaned)
        if not candidates:
            stripped = text.strip()
            if stripped:
                candidates.append(stripped)

        deduped: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            key = self._normalize_answer(item)
            if key and key not in seen:
                seen.add(key)
                deduped.append(item.strip())
        return deduped

    def _numeric_equal(self, pred: str, ans: str, tol: float = 1e-6) -> bool:
        pred_num = re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", pred)
        ans_num = re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", ans)
        if not pred_num or not ans_num:
            return False
        try:
            return abs(float(pred) - float(ans)) <= tol
        except Exception:
            return False

    def _answers_equal(self, pred: str, ans: str) -> bool:
        pred_norm = self._normalize_answer(pred)
        ans_norm = self._normalize_answer(ans)
        if pred_norm == ans_norm and pred_norm != "":
            return True
        if self._numeric_equal(pred_norm, ans_norm):
            return True
        try:
            parsed_res = parse_latex(pred_norm)
            parsed_ans = parse_latex(ans_norm)
            return self._quick_compare(parsed_res, parsed_ans)
        except Exception:
            return False

    def _answers_equal_by_type(
        self, pred: str, line: dict, raw_prediction_unit: str = ""
    ) -> bool:
        answer_type = self._normalize_answer(str(line.get("answer_type", "")))
        ans = str(line.get("answer", ""))
        if answer_type == "option":
            return self._option_equal(pred, ans)
        if answer_type == "value":
            return self._value_equal(
                pred, line, raw_prediction_unit=raw_prediction_unit
            )
        return self._answers_equal(pred, ans)

    def _extract_answer_by_rule(self, line: dict) -> str:
        response = str(line.get("prediction", ""))
        candidates = self._extract_answer_candidates(response)
        if candidates:
            return candidates[0]
        return ""

    def _quick_compare(self, response_expr: Any, answer_expr: Any, tol: float = 1e-6):
        if response_expr is None or answer_expr is None:
            return False

        try:
            if response_expr.is_Number and answer_expr.is_Number:
                return abs(float(response_expr) - float(answer_expr)) < tol
            if isinstance(response_expr, Eq) and isinstance(answer_expr, Eq):
                return str(response_expr) == str(answer_expr)
            return simplify(response_expr - answer_expr) == 0
        except Exception:
            return False

    def _post_check(self, line: dict, prefetch: bool = False) -> Any:
        raw_prediction = str(line.get("prediction", ""))
        raw_prediction_unit = self._extract_unit_from_text(raw_prediction)
        candidates = self._extract_answer_candidates(raw_prediction)
        if not candidates:
            return False
        for candidate in candidates:
            if self._answers_equal_by_type(
                candidate, line, raw_prediction_unit=raw_prediction_unit
            ):
                if prefetch:
                    try:
                        return latex(parse_latex(self._normalize_answer(candidate)))
                    except Exception:
                        return candidate
                return True
        return False

    def SeephysPro_auxeval(self, line: dict) -> Dict[str, Any]:
        log = ""
        gt_answer = str(line["answer"])

        extracted_answer = self._extract_answer_by_rule(line)
        if not extracted_answer:
            return dict(
                log="No non-empty <answer> box. Directly marked wrong.",
                res=0,
                extracted="",
            )

        precheck_result = self._post_check(line, prefetch=True)
        if precheck_result is not False:
            return dict(
                log="Prefetch succeed (Symbolic Match)",
                res=1,
                extracted=precheck_result,
            )

        prompt = self.build_seephys_scoring_prompt(line, extracted_answer)
        for i in range(3):  # 一共最多重试3次
            res_str = self.judger_generate(prompt, temperature=1.0, patience=1)

            if FAIL_MSG in res_str:
                log += f"Try {i}: answer and prediction are {gt_answer} and {extracted_answer}, failed to compare.\n"
                continue

            if "Judgement: " in res_str:
                score_str = res_str.split("Judgement: ")[-1]
            elif "Judgement:" in res_str:
                score_str = res_str.split("Judgement:")[-1]
            elif "judgement: " in res_str:
                score_str = res_str.split("judgement: ")[-1]
            elif "judgement:" in res_str:
                score_str = res_str.split("judgement:")[-1]
            else:
                score_str = res_str

            try:
                score_match = re.search(r"\b(0|1)\b", score_str)
                if score_match:
                    score = int(score_match.group(1))
                    log += "Succeed"
                    return dict(log=log, res=score, extracted=extracted_answer)
            except Exception:
                log += f"Try {i}: Failed to parse score from judger output: {res_str}\n"
                continue

        log += "All 3 retries failed.\n"
        return dict(log=log, res=0, extracted=extracted_answer)

    def SeephysPro_key_info_eval(
        self, line: dict, key_info_list: list, pred: str
    ) -> Dict[str, Any]:
        if not key_info_list:
            return dict(log="Empty key info list", res=0.0, raw=None)

        # 按需求禁用 key info 评估输入截断：使用完整模型回答与完整 key info 列表
        trimmed_pred = pred if isinstance(pred, str) else str(pred)
        trimmed_list = key_info_list
        prompt = self.build_key_info_scoring_prompt(line, trimmed_pred, trimmed_list)
        prompt_chars = len(prompt)
        pred_chars = len(trimmed_pred)
        key_info_items = len(trimmed_list)
        retries = int(config.get("metadata", {}).get("key_info_max_retries", 3) or 3)
        temperature = float(
            config.get("metadata", {}).get("key_info_temperature", 0.0) or 0.0
        )
        log = ""
        last_raw = None

        for i in range(retries):
            res_str = self.judger_generate(
                prompt, temperature=temperature, patience=1, return_error=True
            )
            last_raw = res_str
            if FAIL_MSG in res_str:
                log += f"Try {i}: judge failed.\n"
                continue
            score = self._parse_key_info_score(res_str)
            if score >= 0.0:
                return dict(
                    log=log or "Succeed",
                    res=score,
                    raw=res_str,
                    prompt_chars=prompt_chars,
                    pred_chars=pred_chars,
                    key_info_items=key_info_items,
                )
            log += f"Try {i}: parse failed.\n"

        return dict(
            log=log or "All retries failed",
            res=0.0,
            raw=last_raw,
            prompt_chars=prompt_chars,
            pred_chars=pred_chars,
            key_info_items=key_info_items,
        )

    def SeephysPro_process_line(self, line: dict) -> Dict[str, Any]:
        match = self._post_check(line, prefetch=False)
        extracted = self._extract_answer_by_rule(line)

        return {
            "index": line.get("index", -1),
            "match": 1 if match else 0,
            "extracted": extracted,
            "gt": line["answer"],
        }
