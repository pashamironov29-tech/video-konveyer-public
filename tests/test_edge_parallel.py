"""Edge: параллельный синтез, таймаут запроса, выключатель и общий лимит времени (офлайн)."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from konveyer import voice as voicemod
from konveyer.captions import Cue
from konveyer.log import Log
from konveyer.voice import EdgeEngine, VoiceOver, call_with_timeout, eta_text

MIX = 48000


def _patch_mix(monkeypatch):
    monkeypatch.setattr(voicemod, "to_mix",
                        lambda path, tempo, workdir: np.asarray(voicemod.read_wav(path)[0], np.float32))


def _cfg(settings, **kw):
    settings.voice.voice_mode = "voiceover"
    settings.voice.voice_check = False
    settings.voice.expression["enabled"] = False
    for k, v in kw.items():
        setattr(settings.voice, k, v)
    return settings


def test_call_with_timeout_gives_up_on_hang():
    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match="не ответил"):
        call_with_timeout(lambda: time.sleep(5), 0.2)
    assert time.monotonic() - t0 < 2
    assert call_with_timeout(lambda: 7, 0.5) == 7
    with pytest.raises(ValueError):
        call_with_timeout(lambda: (_ for _ in ()).throw(ValueError("x")), 0.5)


def test_eta_text():
    assert eta_text(0, 10, 30) == "" and eta_text(10, 10, 30) == ""
    assert eta_text(5, 10, 120) == ", осталось ~2 мин"


def test_flaky_edge_fails_then_answers(settings, tmp_path):
    _cfg(settings)
    calls = []

    def fn(text, voice, rate, pitch, volume):
        calls.append(voice)
        if len(calls) <= 2:
            raise ConnectionError("обрыв")
        return np.full(MIX // 2, 0.2, np.float32)

    eng = EdgeEngine(settings.voice, Log(None, echo=False), tmp_path, synth_fn=fn,
                     sleeper=lambda s: None)
    audio, _ = eng.synth_sentence("Привет мир.", voicemod.Prosody("+0%", "+0Hz", "+0%", "normal"))
    assert audio is not None and eng.fell_back == "" and len(set(calls)) == 1
    assert not eng.tripped


def test_hanging_edge_times_out_and_breaker_skips_long_waits(settings, tmp_path):
    _cfg(settings, edge_timeout=0.1, edge_breaker=3, edge_rest_sec=30)
    log_file = tmp_path / "job.log"
    slept = []

    def hang(text, voice, rate, pitch, volume):
        time.sleep(3)

    eng = EdgeEngine(settings.voice, Log(log_file, echo=False), tmp_path, synth_fn=hang,
                     sleeper=slept.append)
    eng.allow_fallback = False
    t0 = time.monotonic()
    audio, _ = eng.synth_sentence("Привет мир.", voicemod.Prosody("+0%", "+0Hz", "+0%", "normal"))
    assert audio is None and eng.tripped
    assert time.monotonic() - t0 < 2                      # не 8 попыток × 20 с и не 2 минуты
    assert not any(s >= voicemod.EDGE_PATIENT_CAP for s in slept)   # терпеливых пауз нет
    text = log_file.read_text(encoding="utf-8")
    assert "3 неудач подряд" in text and "отложен" in text
    # следующий запрос сначала отдыхает (одна пауза на всех), потом спрашивает
    slept.clear()
    eng._try_voice("Ещё.", voicemod.Prosody("+0%", "+0Hz", "+0%", "normal"), eng.voice, 1)
    assert slept and slept[0] == pytest.approx(30, abs=1)


def test_budget_over_goes_to_fallback_voice(settings, tmp_path):
    _cfg(settings, edge_budget_min=1)
    now = [0.0]
    heard = []

    def fn(text, voice, rate, pitch, volume):
        heard.append(voice)
        if voice == voicemod.EDGE_VOICES["brian"]:
            raise ConnectionError("молчит")
        return np.full(MIX // 2, 0.2, np.float32)

    log_file = tmp_path / "job.log"
    eng = EdgeEngine(settings.voice, Log(log_file, echo=False), tmp_path, synth_fn=fn,
                     sleeper=lambda s: None, clock=lambda: now[0])
    now[0] = 61.0                                        # лимит в 1 минуту вышел
    audio, _ = eng.synth_sentence("Привет мир.", voicemod.Prosody("+0%", "+0Hz", "+0%", "normal"))
    assert audio is not None and eng.fell_back and eng.fell_back != eng.voice
    assert heard.count(voicemod.EDGE_VOICES["brian"]) == 1   # основной — одна попытка, без ожиданий
    assert "лимит времени озвучки" in log_file.read_text(encoding="utf-8")


def _render(settings, folder, monkeypatch, workers):
    _cfg(settings, edge_workers=workers)
    lock = threading.Lock()
    state = {"now": 0, "max": 0}

    def fn(text, voice, rate, pitch, volume):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return np.full(int(MIX * (0.3 + 0.02 * len(text))), 0.1 + 0.01 * len(text), np.float32)

    folder.mkdir()
    log_file = folder / "job.log"
    eng = EdgeEngine(settings.voice, Log(log_file, echo=False), folder / "edge", synth_fn=fn,
                     sleeper=lambda s: None)
    cues = [Cue(i * 2.0, i * 2.0 + 1.5, f"Фраза номер {'раз два три'[: i + 3]}.") for i in range(8)]
    track = VoiceOver(folder, settings, Log(log_file, echo=False), None, engine=eng).make(
        folder, cues, 18.0, 1)
    wav, _ = voicemod.read_wav(folder / "voice.wav")
    return track, wav, state["max"], log_file.read_text(encoding="utf-8")


def test_parallel_synth_keeps_order_and_result(settings, tmp_path, monkeypatch):
    _patch_mix(monkeypatch)
    one, wav1, max1, _ = _render(settings, tmp_path / "one", monkeypatch, 1)
    four, wav4, max4, log4 = _render(settings, tmp_path / "four", monkeypatch, 4)
    assert max1 == 1 and max4 > 1                        # действительно одновременно
    assert [p.start for p in four.phrases] == [p.start for p in one.phrases]
    assert [p.text for p in four.phrases] == [p.text for p in one.phrases]
    assert np.array_equal(wav1, wav4)                     # тот же звук, тот же порядок
    assert "озвучка в 4 потока" in log4 and "готовы" in log4
    # кэш: второй прогон Edge не спрашивает
    calls = []
    eng = EdgeEngine(settings.voice, Log(None, echo=False), tmp_path / "four" / "edge",
                     synth_fn=lambda *a: calls.append(a), sleeper=lambda s: None)
    VoiceOver(tmp_path / "four", settings, Log(None, echo=False), None, engine=eng).make(
        tmp_path / "four", [Cue(i * 2.0, i * 2.0 + 1.5, f"Фраза номер {'раз два три'[: i + 3]}.")
                            for i in range(8)], 18.0, 1)
    assert calls == []
