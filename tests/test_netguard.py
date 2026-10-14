"""Тесты не ходят в Telegram, DeepSeek и Edge, даже если ключи лежат в окружении."""
from __future__ import annotations

import os

import pytest

from conftest import BLOCKED_HOSTS
from konveyer.config import LLMCfg
from konveyer.llm import DeepSeek
from konveyer.log import Log
from konveyer.telegram import make_telegram
from konveyer.util import KonveyerError
from konveyer.voice import EdgeEngine


def test_secrets_are_stripped_before_the_test(monkeypatch):
    assert os.environ.get("TG_BOT_TOKEN", "") == ""
    assert os.environ.get("TG_CHAT_ID", "") == ""
    assert os.environ.get("DEEPSEEK_API_KEY", "") == ""
    monkeypatch.setenv("TG_BOT_TOKEN", "123456:FAKE-not-a-real-token")
    monkeypatch.setenv("TG_CHAT_ID", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake-not-real")
    assert os.environ["TG_BOT_TOKEN"].startswith("123456:")


def test_guard_blocks_telegram_and_deepseek_hosts():
    import socket

    for host in ("api.telegram.org", "api.deepseek.com"):
        BLOCKED_HOSTS.clear()
        with pytest.raises(OSError, match="тест не ходит в сеть"):
            socket.getaddrinfo(host, 443)
        assert host in BLOCKED_HOSTS


def test_telegram_deepseek_and_edge_stay_offline(monkeypatch, settings, tmp_path):
    monkeypatch.setenv("TG_BOT_TOKEN", "123456:FAKE-not-a-real-token")
    monkeypatch.setenv("TG_CHAT_ID", "424242")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake-not-real")
    BLOCKED_HOSTS.clear()

    with pytest.raises(KonveyerError) as tg:
        make_telegram(settings).send_message("привет из теста")
    assert "ответил ошибкой" not in str(tg.value)
    assert any("telegram" in host.lower() for host in BLOCKED_HOSTS)

    BLOCKED_HOSTS.clear()
    try:
        DeepSeek(LLMCfg(attempts=1), "sk-fake-not-real", sleep=lambda s: None).chat("sys", "user")
    except KonveyerError as exc:
        assert "401" not in str(exc)
        assert any("deepseek" in host.lower() for host in BLOCKED_HOSTS)
    except ModuleNotFoundError as exc:
        assert "openai" in str(exc).lower()
    else:
        raise AssertionError("DeepSeek ответил из теста")

    BLOCKED_HOSTS.clear()
    eng = EdgeEngine(settings.voice, Log(None, echo=False), tmp_path, sleeper=lambda s: None)
    try:
        audio = eng.synth(["Привет"])
    except KonveyerError as exc:
        assert "edge-tts" in str(exc)
    else:
        assert audio == [None]
        assert BLOCKED_HOSTS
