"""Секреты только из переменных окружения. Значения никогда не печатаются."""
from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_dotenv(path: Path = ENV_FILE) -> None:
    """Подхватить KEY=VALUE из .env в корне проекта. Уже заданные переменные не трогает.
    Load KEY=VALUE pairs from .env in the project root; existing variables win."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and not os.environ.get(key):
            os.environ[key] = value


load_dotenv()

SECRET_NAMES = ("DEEPSEEK_API_KEY", "TG_BOT_TOKEN", "TG_CHAT_ID",
                "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def get_secret(name: str) -> str:
    return os.environ.get(name, "").strip()


def has_secret(name: str) -> bool:
    return bool(get_secret(name))


def mask(text: str) -> str:
    """Заменяет в тексте значения всех известных секретов на ***."""
    if not text:
        return text
    for name in SECRET_NAMES:
        value = get_secret(name)
        if len(value) >= 4 and value in text:
            text = text.replace(value, "***")
    return text
