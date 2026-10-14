"""DeepSeek через OpenAI-совместимый API. Ключ — только DEEPSEEK_API_KEY из окружения."""
from __future__ import annotations

import json
import re
import time
from typing import Protocol

from konveyer.config import LLMCfg, Settings
from konveyer.envkeys import get_secret, mask
from konveyer.log import Log
from konveyer.usage import UsageBook, read_usage
from konveyer.util import KonveyerError


class LLM(Protocol):
    def chat(self, system: str, user: str, json_mode: bool = True,
             temperature: float | None = None) -> str: ...


def extract_json(raw: str):
    """JSON из ответа модели: снимает ```json ...```, вырезает первый {...} или [...]."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        a, b = text.find(open_ch), text.rfind(close_ch)
        if a != -1 and b > a:
            try:
                return json.loads(text[a:b + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("в ответе модели нет корректного JSON")


class DeepSeek:
    def __init__(self, cfg: LLMCfg, api_key: str, client=None, sleep=time.sleep, book: UsageBook | None = None):
        self.cfg = cfg
        self._sleep = sleep
        self.book = book
        self.stage = "прочее"
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=cfg.base_url, timeout=cfg.timeout)
        self.client = client

    def chat(self, system: str, user: str, json_mode: bool = True,
             temperature: float | None = None) -> str:
        import openai

        kwargs = {"model": self.cfg.model,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                  "temperature": self.cfg.temperature if temperature is None else temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last: Exception | None = None
        for attempt in range(1, max(self.cfg.attempts, 1) + 1):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                self._note_usage(resp)
                return resp.choices[0].message.content or ""
            except openai.AuthenticationError as e:
                raise KonveyerError("DeepSeek отклонил ключ (401). Проверь DEEPSEEK_API_KEY.") from e
            except openai.APIStatusError as e:
                if getattr(e, "status_code", None) == 402:
                    raise KonveyerError("DeepSeek: на балансе нет денег (402).") from e
                last = e
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                last = e
            if attempt < self.cfg.attempts:
                self._sleep(2 ** attempt)
        raise KonveyerError(f"DeepSeek не ответил после {self.cfg.attempts} попыток: {mask(str(last))}")

    def _note_usage(self, resp) -> None:
        if self.book is None:
            return
        try:
            parsed = read_usage(getattr(resp, "usage", None))
        except Exception:
            return
        if not parsed:
            return
        self.book.add(self.stage, *parsed)


def chat_json(llm: LLM, system: str, user: str, attempts: int = 2,
              temperature: float | None = None):
    """Запрос с разбором JSON; если модель вернула мусор — переспросить."""
    err: Exception | None = None
    for _ in range(max(attempts, 1)):
        raw = llm.chat(system, user, json_mode=True, temperature=temperature)
        try:
            return extract_json(raw)
        except ValueError as e:
            err = e
    raise KonveyerError(f"модель вернула не JSON: {err}")


def set_stage(llm, stage: str) -> None:
    if llm is not None and hasattr(llm, "stage"):
        llm.stage = stage


def make_llm(settings: Settings, log: Log, enabled: bool = True, job=None,
             warn_missing: bool = True) -> LLM | None:
    if not enabled:
        return None
    key = get_secret("DEEPSEEK_API_KEY")
    if not key:
        if warn_missing:
            log.warn("DEEPSEEK_API_KEY не задан — работаю без ИИ: моменты эвристикой, "
                     "субтитры без перевода, пост шаблоном")
        return None
    book = UsageBook(job, settings, log) if job is not None else None
    return DeepSeek(settings.llm, key, book=book)
