"""Перевод реплик на русский через DeepSeek.

Если задан [paths] subs_dir с вашим subs.py — используем его промпт и
translate_texts (пачки по 40 реплик, дозапрос пропущенных, оригинал если не вышло).
Если нет — та же логика здесь.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from konveyer.llm import LLM, extract_json, set_stage
from konveyer.log import Log

FALLBACK_PROMPT = (
    "Ты переводишь субтитры к видео на русский язык. Пиши живым разговорным русским, "
    "без канцелярита. Коротко: реплика читается за пару секунд. Мат и сленг передавай "
    "естественными русскими аналогами. Имена и названия не коверкай. На вход — JSON-объект: "
    "ключ — номер реплики, значение — текст. Реплики могут обрываться на полуфразе — переводи "
    "каждую отдельно, не переноси текст между номерами. Верни ТОЛЬКО JSON-объект с теми же "
    "ключами и переводами в значениях."
)

_subs_cache: dict[str, object] = {}


def load_subs(subs_dir: str):
    """Модуль subs.py из папки Субтитры или None."""
    if subs_dir in _subs_cache:
        return _subs_cache[subs_dir]
    module = None
    path = Path(subs_dir) / "subs.py"
    if path.is_file():
        name = f"konveyer_subs_bridge_{len(_subs_cache)}"
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module          # без этого @dataclass внутри subs.py падает
            spec.loader.exec_module(module)
            if not (hasattr(module, "translate_texts") and hasattr(module, "SYSTEM_PROMPT")):
                module = None
        except Exception:
            module = None
        if module is None:
            sys.modules.pop(name, None)
    _subs_cache[subs_dir] = module
    return module


def _parse(raw: str, expected: list[int]) -> dict[int, str]:
    data = extract_json(raw)
    out = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                key = int(k)
            except (TypeError, ValueError):
                continue
            if key in expected and isinstance(v, str) and v.strip():
                out[key] = v.strip()
    return out


def _internal_translate(texts: list[str], call, log: Log, batch: int = 40,
                        attempts: int = 3) -> list[str]:
    result: dict[int, str] = {}
    numbered = list(enumerate(texts, start=1))
    for i in range(0, len(numbered), batch):
        pending = dict(numbered[i:i + batch])
        for _ in range(attempts):
            try:
                got = _parse(call(pending), list(pending))
            except Exception as e:
                log.info(f"перевод: сбой ответа ({e}), повторяю")
                continue
            result.update(got)
            pending = {k: v for k, v in pending.items() if k not in got}
            if not pending:
                break
        if pending:
            log.warn(f"не перевелись реплики № {', '.join(map(str, pending))} — оставляю оригинал")
            result.update(pending)
    return [result[i] for i in range(1, len(texts) + 1)]


def translate_texts(texts: list[str], llm: LLM, log: Log, subs_dir: str = "") -> list[str]:
    if not texts:
        return []
    subs = load_subs(subs_dir) if subs_dir else None
    prompt = getattr(subs, "SYSTEM_PROMPT", FALLBACK_PROMPT) if subs else FALLBACK_PROMPT

    def call(batch: dict[int, str]) -> str:
        set_stage(llm, "translate")
        payload = json.dumps({str(k): v for k, v in batch.items()}, ensure_ascii=False)
        return llm.chat(prompt, payload, json_mode=True, temperature=1.0)

    if subs:
        log.info("перевожу через логику subs.py")
        return list(subs.translate_texts(texts, call))
    return _internal_translate(texts, call, log)
