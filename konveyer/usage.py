"""Расход DeepSeek по заданию: токены из ответа API и оценка в долларах.

prompt_cache_hit_tokens — часть prompt_tokens, в сумму «N токенов» второй раз не входит.
Цены читаются из [llm] (доллары за 1 млн токенов). Если все три цены нули — сумму не пишем.
"""
from __future__ import annotations

from konveyer.config import Settings
from konveyer.jobs import Job
from konveyer.log import Log

STAGE_RU = {
    "moments": "моменты",
    "translate": "перевод",
    "post": "пост",
    "quality": "проверка",
    "score": "оценка",
    "voice": "озвучка",
    "voice_check": "проверка озвучки",
    "прочее": "прочее",
}


def _as_int(value) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _get(obj, key):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    if hasattr(obj, key):
        value = getattr(obj, key)
        if value is not None:
            return value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict) and key in extra:
        return extra[key]
    return None


def read_usage(usage) -> tuple[int, int, int] | None:
    """(prompt, completion, cache hit) или None, если блока usage нет."""
    if usage is None:
        return None
    prompt = _as_int(_get(usage, "prompt_tokens"))
    completion = _as_int(_get(usage, "completion_tokens"))
    cache = _get(usage, "prompt_cache_hit_tokens")
    if cache is None:
        details = _get(usage, "prompt_tokens_details")
        cache = _get(details, "cached_tokens") if details is not None else None
    return prompt, completion, _as_int(cache)


def estimate_cost(prompt: int, completion: int, cache: int,
                  price_input: float, price_output: float, price_cache_hit: float) -> float | None:
    prices = (price_input, price_output, price_cache_hit)
    if any(p is None or p < 0 for p in prices):
        return None
    if price_input == 0 and price_output == 0 and price_cache_hit == 0:
        return None
    hit = min(max(cache, 0), max(prompt, 0))
    miss = max(prompt, 0) - hit
    return (miss * price_input + hit * price_cache_hit + max(completion, 0) * price_output) / 1_000_000


def format_usd(amount: float) -> str:
    text = f"{amount:.6f}".rstrip("0").rstrip(".")
    return f"≈ ${text}"


def _empty() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "prompt_cache_hit_tokens": 0}


def _add(bucket: dict, prompt: int, completion: int, cache: int) -> None:
    bucket["prompt_tokens"] += prompt
    bucket["completion_tokens"] += completion
    bucket["prompt_cache_hit_tokens"] += cache


def total_tokens(bucket: dict) -> int:
    return int(bucket.get("prompt_tokens") or 0) + int(bucket.get("completion_tokens") or 0)


def summary_line(bucket: dict, settings: Settings) -> str:
    llm = settings.llm
    total = total_tokens(bucket)
    cost = estimate_cost(bucket.get("prompt_tokens") or 0, bucket.get("completion_tokens") or 0,
                         bucket.get("prompt_cache_hit_tokens") or 0,
                         llm.price_input, llm.price_output, llm.price_cache_hit)
    if cost is None:
        return f"DeepSeek: {total} токенов"
    return f"DeepSeek: {total} токенов, {format_usd(cost)}"


def stage_line(stage: str, row: dict) -> str:
    name = STAGE_RU.get(stage, stage)
    return (f"DeepSeek [{name}]: {total_tokens(row)} токенов "
            f"(prompt {row['prompt_tokens']}, completion {row['completion_tokens']}, "
            f"cache hit {row['prompt_cache_hit_tokens']})")


class UsageBook:
    """Копит токены в job.json. В журнал пишет итог по этапам, когда попросят flush."""

    def __init__(self, job: Job, settings: Settings, log: Log):
        self.job, self.settings, self.log = job, settings, log
        self._logged: dict[str, tuple] = {}
        self._total_line = ""

    def add(self, stage: str, prompt: int, completion: int, cache: int) -> None:
        prompt, completion, cache = _as_int(prompt), _as_int(completion), _as_int(cache)
        if prompt == 0 and completion == 0 and cache == 0:
            return
        root = self.job.data.setdefault("deepseek", _empty())
        root.setdefault("stages", {})
        _add(root, prompt, completion, cache)
        row = root["stages"].setdefault(stage, _empty())
        _add(row, prompt, completion, cache)
        cost = estimate_cost(root["prompt_tokens"], root["completion_tokens"],
                             root["prompt_cache_hit_tokens"],
                             self.settings.llm.price_input, self.settings.llm.price_output,
                             self.settings.llm.price_cache_hit)
        root["total_tokens"] = total_tokens(root)
        root["cost_usd"] = None if cost is None else round(cost, 8)
        root["summary"] = summary_line(root, self.settings)
        self.job.save()

    def flush(self, final: bool = False) -> None:
        root = self.job.data.get("deepseek")
        if not root or total_tokens(root) <= 0:
            return
        for stage, row in (root.get("stages") or {}).items():
            marker = (row.get("prompt_tokens"), row.get("completion_tokens"),
                      row.get("prompt_cache_hit_tokens"))
            if marker == self._logged.get(stage) or total_tokens(row) <= 0:
                continue
            self.log.info(stage_line(stage, row))
            self._logged[stage] = marker
        if final:
            line = root.get("summary") or summary_line(root, self.settings)
            if line != self._total_line:
                self.log.info(line)
                self._total_line = line
