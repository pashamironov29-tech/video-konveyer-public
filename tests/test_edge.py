"""Edge TTS: голос по умолчанию, интонация, повторы, без отката на Silero, ускорение rate.

Сеть и пакет edge-tts не вызываются: синтез подменён функцией, doctor — заглушкой.
"""
from __future__ import annotations

import wave

import numpy as np
import pytest

from conftest import FakeLLM
from konveyer import cli
from konveyer.captions import Cue
from konveyer.config import EDGE_VOICES, combine_rates, load_settings, speaker_error
from konveyer.jobs import Job
from konveyer.log import Log
from konveyer.pipeline import settings_for_job
from konveyer.util import KonveyerError
from konveyer.saytext import prepare_for_speech
from konveyer.voice import (EDGE_ATTEMPTS, EDGE_FALLBACK_ATTEMPTS, EDGE_RETRY_CAP, EdgeEngine, Prosody, SileroEngine,
                            VoiceOver, cache_key, clean_text, decode_to_f32, edge_markup,
                            edge_fallback_voices, edge_pieces, edge_plain, edge_retry_pause, edge_segments, edge_service_rate,
                            edge_text_ok, expression_prosody, make_engine, phrase_kind,
                            strip_speech_punct)
from konveyer import voice as voicemod

MIX = 48000
PER = 0.05          # секунд на символ при rate +0%


class NoAudioReceived(Exception):
    pass


def _tone(seconds: float) -> np.ndarray:
    n = max(1, int(MIX * seconds))
    return np.full(n, 0.3, np.float32)


def test_edge_voices_mapping_and_default(settings, tmp_path):
    assert tuple(EDGE_VOICES)[0] == "brian"
    assert EDGE_VOICES["brian"] == "en-US-BrianMultilingualNeural"
    assert EDGE_VOICES["dmitry"] == "ru-RU-DmitryNeural"
    assert EDGE_VOICES["svetlana"] == "ru-RU-SvetlanaNeural"
    assert EDGE_VOICES["ava"] == "en-US-AvaMultilingualNeural"
    assert EDGE_VOICES["emma"] == "en-US-EmmaMultilingualNeural"
    assert EDGE_VOICES["florian"] == "de-DE-FlorianMultilingualNeural"
    assert EDGE_VOICES["seraphina"] == "de-DE-SeraphinaMultilingualNeural"
    assert "andrew" not in EDGE_VOICES and "Andrew" not in EDGE_VOICES
    assert settings.voice.engine == "edge" and settings.voice.speaker == "brian"
    eng = make_engine(settings, Log(None, echo=False), tmp_path)
    assert isinstance(eng, EdgeEngine)
    assert eng.voice == "en-US-BrianMultilingualNeural"
    assert eng.tag.startswith("edge:en-US-BrianMultilingualNeural:+0%:+0Hz:")
    settings.voice.speaker = "aidar"
    with pytest.raises(KonveyerError, match="не подходит движку edge"):
        make_engine(settings, Log(None, echo=False), tmp_path)
    settings.voice.engine = "silero"
    settings.voice.speaker = "brian"
    with pytest.raises(KonveyerError, match="не подходит движку silero"):
        make_engine(settings, Log(None, echo=False), tmp_path)
    settings.voice.speaker = "aidar"
    assert isinstance(make_engine(settings, Log(None, echo=False), tmp_path), SileroEngine)
    assert speaker_error("xtts", "brian") is None


def test_phrase_kind_and_prosody(settings):
    assert phrase_kind("Ты это видел?") == "question"
    assert phrase_kind("Невероятно!") == "exclamation"
    assert phrase_kind("Что?!") == "exclamation"          # последний знак — восклицание
    assert phrase_kind("Ну… ладно.") == "ellipsis"
    assert phrase_kind("Коротко.") == "normal"
    long = "Это длинное спокойное утверждение о том, как всё устроено на самом деле сегодня."
    assert len(long) >= 80 and phrase_kind(long) == "serious"
    assert phrase_kind(long, serious_chars=200) == "normal"

    q = expression_prosody("Как дела?", settings.voice)
    assert q.kind == "question" and q.pitch == "+8Hz" and q.rate == "+0%"
    ex = expression_prosody("Невероятно!", settings.voice)
    assert ex.kind == "exclamation" and ex.pitch == "+12Hz" and ex.rate == "+8%"
    serious = expression_prosody(long, settings.voice)
    assert serious.kind == "serious" and serious.pitch == "-4Hz" and serious.rate == "-5%"
    dots = expression_prosody("Ну… ладно.", settings.voice)
    assert dots.kind == "ellipsis" and dots.rate == "-8%" and dots.volume == "-10%"
    assert expression_prosody("Идём.", settings.voice).rate == "+0%"

    # ускорение складывается с интонацией, а не заменяет её: 1.08 × 1.25 = 1.35
    stacked = expression_prosody("Невероятно!", settings.voice, 1.25)
    assert stacked.rate == "+35%" and stacked.pitch == "+12Hz"
    assert combine_rates("+8%", 1.2) == "+29.6%"
    assert combine_rates("+0%", "-5%", 1.3) == "+23.5%"

    settings.voice.edge_rate = "+10%"
    settings.voice.edge_pitch = "+2Hz"
    shifted = expression_prosody("Как дела?", settings.voice)
    assert shifted.rate == "+10%" and shifted.pitch == "+10Hz"

    settings.voice.expression["enabled"] = False
    flat = expression_prosody("Как дела?", settings.voice)
    assert flat.pitch == "+2Hz" and flat.rate == "+10%"   # база остаётся, интонации нет


def test_edge_markup_keeps_sentence_pauses():
    text = "Ты это видел? Невероятно! Идём дальше. Конец"
    parts = edge_segments(text)
    assert [pause for _, pause in parts] == [300, 300, 300, 0]
    marked = edge_markup(text)
    assert marked.endswith("Конец") and "prosody" not in marked
    assert edge_text_ok(marked)
    assert "<" not in marked and ">" not in marked and "break" not in marked.casefold()
    assert edge_markup("Одна фраза.") == "Одна фраза."
    dots = edge_segments("Ну… ладно.")
    assert dots[0][1] == 500 and edge_text_ok(edge_markup("Ну… ладно."))
    dirty = edge_plain('меньше чем <break time="300ms"/> секунды')
    assert dirty == "меньше чем секунды" and edge_text_ok(dirty)


def test_cache_key_includes_engine_voice_rate_pitch():
    base = "edge:en-US-BrianMultilingualNeural:+0%:+0Hz:abcd"
    a = cache_key(base, "Привет", "+0%|+8Hz|+0%|question")
    b = cache_key(base, "Привет", "+20%|+8Hz|+0%|question")
    c = cache_key("edge:ru-RU-DmitryNeural:+0%:+0Hz:abcd", "Привет", "+0%|+8Hz|+0%|question")
    d = cache_key("edge:en-US-BrianMultilingualNeural:+5%:+0Hz:abcd", "Привет", "+0%|+8Hz|+0%|question")
    e = cache_key("silero:aidar:48000", "Привет")
    assert len({a, b, c, d, e}) == 5


def _engine(settings, tmp_path, fn, sleeps=None):
    return EdgeEngine(settings.voice, Log(None, echo=False), tmp_path, synth_fn=fn,
                      sleeper=(sleeps.append if sleeps is not None else (lambda s: None)))


def test_edge_retries_no_audio_then_succeeds(settings, tmp_path):
    calls = {"n": 0}
    sleeps: list[float] = []

    def fn(text, voice, rate, pitch, volume):
        calls["n"] += 1
        assert voice == "en-US-BrianMultilingualNeural"
        assert edge_text_ok(text) and "prosody" not in text
        if calls["n"] < EDGE_ATTEMPTS:
            raise NoAudioReceived("vpn")
        return _tone(0.4)

    eng = _engine(settings, tmp_path, fn, sleeps)
    out = eng.synth(["Идём дальше вместе"])
    assert calls["n"] == EDGE_ATTEMPTS and len(out[0]) == int(0.4 * MIX)
    assert len(sleeps) == EDGE_ATTEMPTS - 1
    assert all(1.0 <= s <= EDGE_RETRY_CAP for s in sleeps)


PATIENT = int(-(-voicemod.EDGE_PATIENT_SEC // voicemod.EDGE_PATIENT_CAP))


def _all_tries(settings) -> int:
    alts = edge_fallback_voices(settings.voice, EDGE_VOICES["brian"])
    return EDGE_ATTEMPTS + PATIENT + len(alts) * EDGE_FALLBACK_ATTEMPTS


def test_edge_falls_back_to_same_gender_voice(settings, tmp_path):
    """Основной голос молчит все попытки — фраза звучит запасным мужским, а не пропадает."""
    settings.voice.edge_breaker = 0       # считаем попытки старой политики, без выключателя
    log_file = tmp_path / "job.log"
    heard = []

    def fn(text, voice, rate, pitch, volume):
        heard.append(voice)
        if voice == EDGE_VOICES["dmitry"]:
            raise NoAudioReceived("vpn")
        return _tone(0.4)

    eng = EdgeEngine(settings.voice, Log(log_file, echo=False), tmp_path, synth_fn=fn,
                     sleeper=lambda s: None)
    out = eng.synth_with_voices(["Это была ещё одна часть тревоги."], None, [EDGE_VOICES["dmitry"]])
    assert out[0] is not None and len(out[0]) == int(0.4 * MIX)
    same = EDGE_ATTEMPTS + PATIENT          # сначала долго и терпеливо тем же голосом
    assert heard[:same] == [EDGE_VOICES["dmitry"]] * same
    assert heard[same] == "fr-FR-RemyMultilingualNeural"
    assert eng.fell_back == heard[-1]
    alts = edge_fallback_voices(settings.voice, EDGE_VOICES["ava"])
    assert alts[0] == "fr-FR-VivienneMultilingualNeural"
    assert all("Andrew" not in a and "Dariya" not in a for a in alts)
    assert all(a in ("fr-FR-VivienneMultilingualNeural",
                     *(EDGE_VOICES[n] for n in ("emma", "seraphina", "svetlana"))) for a in alts)
    text = log_file.read_text(encoding="utf-8")
    assert "запасным" in text and "без озвучки" not in text


def test_edge_retry_pause_grows_then_caps(monkeypatch):
    def uniform(low, high):
        return (low + high) / 2

    monkeypatch.setattr(voicemod.random, "uniform", uniform)
    pauses = [edge_retry_pause(i) for i in range(EDGE_ATTEMPTS)]
    assert pauses[0] < pauses[2] < pauses[4]
    assert all(1.0 <= p <= EDGE_RETRY_CAP for p in pauses)
    assert pauses[-1] == pauses[-2]


@pytest.mark.parametrize("boom", ["noaudio", "error", "empty"])
def test_edge_silent_does_not_call_silero(settings, tmp_path, monkeypatch, boom):
    """Edge стабильно без звука: Silero не вызывается и aidar не подставляется."""
    settings.voice.edge_breaker = 0       # считаем попытки старой политики, без выключателя
    (tmp_path / "v4_ru.pt").write_bytes(b"x")
    settings.voice.model_path = str(tmp_path / "v4_ru.pt")
    log_file = tmp_path / "job.log"
    calls = {"n": 0}
    silero = {"n": 0}

    class Stub:
        def __init__(self, cfg, log, loader=None):
            silero["n"] += 1
            self.cfg, self.rate = cfg, 24000

        def synth(self, texts):
            silero["n"] += 1
            return [_tone(0.2)]

    monkeypatch.setattr(voicemod, "SileroEngine", Stub)

    def fn(*a):
        calls["n"] += 1
        if boom == "noaudio":
            raise NoAudioReceived("vpn")
        if boom == "error":
            raise RuntimeError("сеть оборвалась")
        return np.zeros(0, np.float32)

    eng = EdgeEngine(settings.voice, Log(log_file, echo=False), tmp_path, synth_fn=fn,
                     sleeper=lambda s: None)
    assert eng.synth(["Привет всем"]) == [None]
    assert calls["n"] == _all_tries(settings) and silero["n"] == 0
    text = log_file.read_text(encoding="utf-8")
    assert text.count("\n") == 1
    assert text.count("фраза без озвучки") == 1
    assert "aidar" not in text.lower() and "silero" not in text.lower() and "откат" not in text.lower()
    assert "Traceback" not in text
    if boom == "noaudio":
        assert "NoAudioReceived" in text
    elif boom == "error":
        assert "RuntimeError" in text
    else:
        assert "пустой звук" in text


def test_edge_without_model_still_skips_phrase(settings, tmp_path, monkeypatch):
    settings.voice.model_path = str(tmp_path / "нет" / "v4_ru.pt")
    log_file = tmp_path / "job.log"
    silero = {"n": 0}

    class Stub:
        def __init__(self, *a, **k):
            silero["n"] += 1

        def synth(self, texts):
            silero["n"] += 1
            return [_tone(0.2)]

    monkeypatch.setattr(voicemod, "SileroEngine", Stub)

    def fn(*a):
        raise NoAudioReceived("vpn")

    eng = EdgeEngine(settings.voice, Log(log_file, echo=False), tmp_path, synth_fn=fn,
                     sleeper=lambda s: None)
    assert eng.synth(["Одна."]) == [None]
    assert silero["n"] == 0
    text = log_file.read_text(encoding="utf-8")
    assert "фраза без озвучки" in text and "aidar" not in text.lower()


def test_missing_edge_package_disables_voice(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(voicemod.importlib.util, "find_spec", lambda name: None)
    eng = EdgeEngine(settings.voice, Log(None, echo=False), tmp_path)
    with pytest.raises(KonveyerError, match="edge-tts"):
        eng.synth(["Привет"])
    log_file = tmp_path / "job.log"
    vo = VoiceOver(tmp_path, settings, Log(log_file, echo=False), None)
    assert vo.make(tmp_path, [Cue(0, 1, "Раз.")], 2.0, 1) is None
    assert vo.make(tmp_path, [Cue(0, 1, "Два.")], 2.0, 2) is None
    assert log_file.read_text(encoding="utf-8").count("озвучка выключена") == 1


def _patch_mix(monkeypatch):
    seen = []

    def fake_to_mix(path, tempo, workdir):
        seen.append(tempo)
        data, _rate = voicemod.read_wav(path)
        if tempo > 1.0001:
            data = data[:max(1, int(len(data) / tempo))]
        return np.asarray(data, np.float32)

    monkeypatch.setattr(voicemod, "to_mix", fake_to_mix)
    return seen


def _seconds_fn(calls):
    def fn(text, voice, rate, pitch, volume):
        from konveyer.config import parse_percent
        calls.append({"text": text, "voice": voice, "rate": rate, "pitch": pitch, "volume": volume})
        factor = 1.0 + parse_percent(rate) / 100.0
        plain = text.split("<break")[0]
        return _tone(len(plain) * PER / factor)
    return fn


def test_edge_rate_fits_and_stacks_with_expression(settings, tmp_path, monkeypatch):
    settings.voice.voice_mode = "fit"
    tempos = _patch_mix(monkeypatch)
    calls = []
    text = "А" * 23 + "!"
    assert len(text) == 24 and phrase_kind(text) == "exclamation"
    eng = EdgeEngine(settings.voice, Log(None, echo=False), tmp_path,
                     synth_fn=_seconds_fn(calls), sleeper=lambda s: None)
    vo = VoiceOver(tmp_path, settings, Log(None, echo=False), None, engine=eng)
    # 24 символа при +8% ≈ 1.111 с, окно 1.0 с → rate +20% (1.08 × 1.111), ниже потолка 1.15
    track = vo.make(tmp_path, [Cue(0.0, 0.2, text)], 1.05, 1)
    ph = track.phrases[0]
    assert ph.status == "tempo" and ph.tempo == 1.0 and ph.kind == "exclamation"
    assert ph.rate == "+20%" and ph.duration <= ph.slot + 0.01
    assert [c["rate"] for c in calls] == ["+8%", "+20%"]
    assert all(c["pitch"] == "+12Hz" for c in calls)
    assert tempos == [1.0]
    assert track.intervals and track.note.startswith("brian")
    saved = __import__("json").loads((tmp_path / "voice.json").read_text(encoding="utf-8"))
    assert saved["phrases"][0]["kind"] == "exclamation"
    assert saved["engine"].startswith("edge:en-US-BrianMultilingualNeural:")


def test_edge_atempo_when_rate_does_not_help(settings, tmp_path, monkeypatch):
    settings.voice.voice_mode = "fit"
    settings.voice.expression["enabled"] = False
    tempos = _patch_mix(monkeypatch)
    calls = []

    def fn(text, voice, rate, pitch, volume):
        calls.append(rate)
        return _tone(1.6)          # rate игнорируем: пересинтез не укоротил фразу

    eng = EdgeEngine(settings.voice, Log(None, echo=False), tmp_path, synth_fn=fn,
                     sleeper=lambda s: None)
    vo = VoiceOver(tmp_path, settings, Log(None, echo=False), None, engine=eng)
    track = vo.make(tmp_path, [Cue(0.0, 0.5, "Коротко.")], 1.45, 1)
    ph = track.phrases[0]
    assert ph.status == "tempo" and ph.tempo == pytest.approx(1.143)
    assert calls[0] == "+0%" and calls[1] != calls[0]
    assert tempos == [pytest.approx(1.143)]
    assert ph.duration <= ph.slot + 0.02


def _whole_words(spoken: str, original: str) -> bool:
    words = original.split()
    kept = spoken.split()
    return bool(kept) and kept == words[:len(kept)] and len(kept) < len(words)


def test_edge_sentence_pause_is_silence_not_tags(settings, tmp_path):
    texts = []

    def fn(text, voice, rate, pitch, volume):
        texts.append(text)
        assert edge_text_ok(text)
        return _tone(0.2)

    eng = _engine(settings, tmp_path, fn)
    out = eng.synth(["Раз. Два."])
    assert texts == ["Раз.", "Два."]
    assert len(out[0]) == int(0.2 * MIX) + int(0.3 * MIX) + int(0.2 * MIX)
    eng.synth(['меньше чем <break time="200ms"/> секунды'])
    assert texts[-1] == "меньше чем секунды"
    assert all(edge_text_ok(t) for t in texts)


def test_edge_keeps_subtitle_and_cuts_without_rewrite(settings, tmp_path, monkeypatch):
    settings.voice.voice_mode = "fit"
    _patch_mix(monkeypatch)
    calls = []
    long = ("Это очень длинная спокойная фраза, которая никак не помещается "
            "в крошечное окно клипа и поэтому её надо сократить целиком.")
    assert len(long) >= 80 and phrase_kind(long) == "serious"
    eng = EdgeEngine(settings.voice, Log(None, echo=False), tmp_path,
                     synth_fn=_seconds_fn(calls), sleeper=lambda s: None)
    llm = FakeLLM()
    log_file = tmp_path / "job.log"
    vo = VoiceOver(tmp_path, settings, Log(log_file, echo=False), llm, engine=eng)
    track = vo.make(tmp_path, [Cue(0.0, 0.2, long)], 1.05, 1)
    ph = track.phrases[0]
    assert ph.status == "cut" and ph.text == long and ph.tempo == 1.0
    assert ph.said == ph.spoken and _whole_words(ph.spoken, long) and ph.dropped
    assert ph.duration <= 1.12
    assert calls[0]["rate"] == "-5%"          # серьёзная интонация, сначала вся строка
    assert calls[0]["text"] == long
    assert all(edge_text_ok(c["text"]) for c in calls)
    assert not any("сокращаешь" in c[0] for c in llm.calls)
    assert "Коротко." not in ph.spoken
    assert "обрезка по границе слова" in log_file.read_text(encoding="utf-8")
    assert ph.dropped in log_file.read_text(encoding="utf-8")

    calls.clear()
    vo = VoiceOver(tmp_path / "b", settings, Log(None, echo=False), None, engine=eng)
    (tmp_path / "b").mkdir()
    track = vo.make(tmp_path / "b", [Cue(0.0, 0.2, long)], 1.05, 1)
    ph = track.phrases[0]
    assert ph.status == "cut" and ph.tempo == 1.0 and ph.duration <= 1.12
    assert ph.text == long and ph.said == ph.spoken and _whole_words(ph.spoken, long)
    assert settings.voice.edge_max_rate == 1.15
    assert ph.rate == "+9.2%"                  # -5% интонации × потолок 1.15, до округления
    assert calls[-1]["rate"] == "+9%"          # в Edge уходит целый процент
    assert "обрезано 1" in track.note


def test_edge_cache_respects_rate_pitch_and_expression(settings, tmp_path, monkeypatch):
    _patch_mix(monkeypatch)
    calls = []
    eng = EdgeEngine(settings.voice, Log(None, echo=False), tmp_path,
                     synth_fn=_seconds_fn(calls), sleeper=lambda s: None)
    cues = [Cue(0.0, 0.5, "Как дела?")]
    VoiceOver(tmp_path, settings, Log(None, echo=False), None, engine=eng).make(tmp_path, cues, 3.0, 1)
    n = len(calls)
    VoiceOver(tmp_path, settings, Log(None, echo=False), None, engine=eng).make(tmp_path, cues, 3.0, 1)
    assert len(calls) == n
    settings.voice.edge_pitch = "+4Hz"
    eng2 = EdgeEngine(settings.voice, Log(None, echo=False), tmp_path,
                      synth_fn=_seconds_fn(calls), sleeper=lambda s: None)
    VoiceOver(tmp_path, settings, Log(None, echo=False), None, engine=eng2).make(tmp_path, cues, 3.0, 1)
    assert len(calls) > n
    assert calls[-1]["pitch"] == "+12Hz"       # +4 Гц базы и +8 Гц вопроса


def test_one_failed_phrase_does_not_drop_the_rest(settings, tmp_path, monkeypatch):
    settings.voice.edge_breaker = 0       # считаем попытки старой политики, без выключателя
    settings.voice.edge_workers = 1       # и без предзапросов в потоках
    _patch_mix(monkeypatch)
    (tmp_path / "v4_ru.pt").write_bytes(b"x")
    settings.voice.model_path = str(tmp_path / "v4_ru.pt")
    log_file = tmp_path / "job.log"
    silero = {"n": 0}
    calls = {"n": 0}

    class Stub:
        def __init__(self, *a, **k):
            silero["n"] += 1

        def synth(self, texts):
            silero["n"] += 1
            return [_tone(0.2)]

    monkeypatch.setattr(voicemod, "SileroEngine", Stub)

    def fn(text, voice, rate, pitch, volume):
        calls["n"] += 1
        if text.startswith("Раз"):
            raise NoAudioReceived("vpn")
        return _tone(0.3)

    eng = EdgeEngine(settings.voice, Log(log_file, echo=False), tmp_path, synth_fn=fn,
                     sleeper=lambda s: None)
    vo = VoiceOver(tmp_path, settings, Log(log_file, echo=False), None, engine=eng)
    # Пауза длиннее границы блока: сбой первой реплики не забирает вторую.
    track = vo.make(tmp_path, [Cue(0.0, 0.4, "Раз."), Cue(1.3, 1.8, "Два слова.")], 3.0, 1)
    assert [p.status for p in track.phrases] == ["fail", "ok"]
    assert len(track.intervals) == 1 and track.note.startswith("brian")
    # первый проход — только основной голос, кусок отложен; в конце — ещё раз и запасные
    assert calls["n"] == EDGE_ATTEMPTS + PATIENT + 1 + _all_tries(settings) and silero["n"] == 0
    with wave.open(str(tmp_path / "voice.wav"), "rb") as w:
        assert w.getframerate() == 48000
    text = log_file.read_text(encoding="utf-8")
    assert text.count("фраза без озвучки") == 1
    assert "aidar" not in text.lower() and "silero" not in text.lower()
    dumped = (tmp_path / "voice.json").read_text(encoding="utf-8")
    assert "aidar" not in dumped.lower() and '"status": "fail"' in dumped


_REJECTED = ("+14.1%", "+14.6%", "+24.5%", "+19.7%", "+2.3%", "+29.2%", "+19.8%")


def test_edge_service_rate_keeps_cap_and_intonation():
    assert edge_service_rate("+29.6%") == "+30%"     # потолок не сползает вниз
    assert edge_service_rate("+30%") == "+30%"
    assert edge_service_rate("+35%") == "+35%"       # восклицание поверх ускорения не зажимается
    assert edge_service_rate("+8%") == "+8%"
    assert edge_service_rate("-5%") == "-5%"
    assert edge_service_rate("+23.5%") == "+24%"     # -5% × 1.3 не стал ни +0%, ни +30%
    assert edge_service_rate("+0%") == "+0%"
    for raw in _REJECTED:
        got = edge_service_rate(raw)
        assert got[0] in "+-" and got.endswith("%") and "." not in got
        assert abs(int(got[:-1]) - float(raw[:-1])) <= 0.5


def test_edge_fractional_rate_is_spoken(settings, tmp_path, monkeypatch):
    """Сервис отвергает +14.1%. В Edge уходит целый процент, фраза озвучена, Silero молчит."""
    seen = []
    silero = {"n": 0}

    class Stub:
        def __init__(self, *a, **k):
            silero["n"] += 1

        def synth(self, texts):
            silero["n"] += 1
            return [_tone(0.2)]

    monkeypatch.setattr(voicemod, "SileroEngine", Stub)

    def fn(text, voice, rate, pitch, volume):
        seen.append((text, rate))
        if "." in rate:
            raise ValueError(f"Invalid rate '{rate}'.")
        return _tone(0.3)

    eng = EdgeEngine(settings.voice, Log(tmp_path / "job.log", echo=False), tmp_path,
                     synth_fn=fn, sleeper=lambda s: None)
    line = "между нами и теми"
    for raw in _REJECTED:
        audio = eng.synth([line], [Prosody(raw, "+0Hz", "+0%", "normal")])[0]
        assert audio is not None and len(audio) == int(0.3 * MIX)
    assert [rate for _, rate in seen] == [edge_service_rate(r) for r in _REJECTED]
    assert all("." not in rate for _, rate in seen)
    assert silero["n"] == 0
    assert all(text == line for text, _ in seen)


def test_edge_speaks_subtitle_line(settings, tmp_path, monkeypatch):
    settings.voice.voice_mode = "fit"
    _patch_mix(monkeypatch)
    calls = []
    line = "Ну, знаешь, я"
    # 1.141 × окно даёт внутренний rate +14.1%, в сервис должен уйти целый процент.
    slot = 1.95
    dur = slot * 1.141

    def fn(text, voice, rate, pitch, volume):
        calls.append({"text": text, "rate": rate})
        if "." in rate:
            raise ValueError(f"Invalid rate '{rate}'.")
        return _tone(dur if rate == "+0%" else dur / 1.14)

    eng = EdgeEngine(settings.voice, Log(None, echo=False), tmp_path, synth_fn=fn,
                     sleeper=lambda s: None)
    llm = FakeLLM()
    track = VoiceOver(tmp_path, settings, Log(None, echo=False), llm, engine=eng).make(
        tmp_path, [Cue(0.0, 1.0, line), Cue(2.0, 3.0, "Вот так.")], 4.0, 1)
    spoken = [c["text"] for c in calls]
    assert line in spoken and "Вот так." in spoken
    assert all("." not in c["rate"] for c in calls)
    assert any(c["rate"] == "+14%" for c in calls)
    assert track.phrases[0].said == line and track.phrases[0].spoken == line
    assert track.phrases[1].said == "Вот так."
    assert not any("сокращаешь" in c[0] for c in llm.calls)
    assert "Коротко." not in spoken


def test_edge_prepares_speech_and_keeps_subtitle(settings, tmp_path, monkeypatch):
    """Edge говорит подготовленный русский текст. Субтитр и смысл строки не сокращаются."""
    _patch_mix(monkeypatch)
    seen = []
    real = prepare_for_speech

    def wrap(text, replacements=None):
        seen.append(text)
        return real(text, replacements)

    monkeypatch.setattr(voicemod, "prepare_for_speech", wrap)
    calls = []
    line = "В 2024 году вышел GPT и API, буква З"
    cue = Cue(0.0, 1.2, line)

    def fn(text, voice, rate, pitch, volume):
        calls.append(text.split("<break")[0].strip())
        return _tone(0.4)

    eng = EdgeEngine(settings.voice, Log(None, echo=False), tmp_path, synth_fn=fn,
                     sleeper=lambda s: None)
    llm = FakeLLM()
    track = VoiceOver(tmp_path, settings, Log(None, echo=False), llm, engine=eng).make(
        tmp_path, [cue], 4.0, 1)
    said = strip_speech_punct(prepare_for_speech(clean_text(line)))
    assert seen == [clean_text(line)]
    assert cue.text == line and track.phrases[0].text == line
    assert track.phrases[0].spoken == line and "," in line
    assert track.phrases[0].said == said and "," not in said
    assert calls == [said]
    assert "2024" not in said and "GPT" not in said and "API" not in said
    assert "джи-пи-ти" in said and "эй-пи-ай" in said
    assert "две тысячи двадцать четвёртом" in said
    assert "зэ" in said and "три" not in said
    assert not any("сокращаешь" in c[0] for c in llm.calls)


def test_voice_cues_match_subtitle_lines(settings):
    from konveyer.layout import geometry
    from konveyer.render import Renderer
    from konveyer.transcript import Segment, Transcript, Word

    words = [Word(0.2, 0.6, "hello"), Word(0.7, 1.1, "there."), Word(1.4, 1.8, "friend")]
    tr = Transcript("en", [Segment(0.2, 1.8, "hello there. friend", words)])
    job = Job.create(settings, "a.mp4")
    renderer = Renderer(job, settings, Log(None, echo=False), tr, {}, FakeLLM())
    geo = geometry("blur", 1920, 1080)
    voice = renderer.voice_cues(words, None)
    ass, note = renderer.captions(words, None, geo)
    assert note == "перевод на русский"
    assert [c.text for c in voice] == [c.text for c in renderer.subtitle_cues(words, None)]
    assert voice and all(c.text in ass for c in voice)

    settings.subs.style = "big"
    voice_big = renderer.voice_cues(words, None)
    ass_big, note_big = renderer.captions(words, None, geo)
    assert note_big.startswith("крупные")
    # Крупные субтитры пишутся заглавными, слова те же.
    folded = ass_big.upper()
    assert voice_big and all(c.text.upper() in folded for c in voice_big)
    assert [c.text for c in voice_big] == [c.text for c in renderer.subtitle_cues(words, None)]


def test_expression_config_and_voice_flat(settings, tmp_path):
    toml = tmp_path / "konveyer.toml"
    toml.write_text(
        '[voice]\nedge_rate = "+5%"\nedge_pitch = "-2Hz"\n'
        '[voice.expression]\nenabled = true\nquestion_pitch = "+3Hz"\npause_ms = 400\n',
        encoding="utf-8")
    s = load_settings(toml, root=tmp_path)
    assert s.voice.edge_rate == "+5%" and s.voice.edge_pitch == "-2Hz"
    assert s.voice.expression["question_pitch"] == "+3Hz"
    assert s.voice.expression["pause_ms"] == 400
    assert s.voice.expression["exclamation_rate"] == "+8%"    # не затёрлось
    assert s.voice.expression["enabled"] is True
    pros = expression_prosody("Как дела?", s.voice)
    assert pros.pitch == "+1Hz" and pros.rate == "+5%"

    p = cli.build_parser()
    opts = cli._job_options(p.parse_args(["redo", "last", "--voice", "--voice-flat"]), settings)
    assert opts["voice_flat"] is True
    job = Job.create(settings, "a.mp4", opts)
    tuned = settings_for_job(settings, job)
    assert tuned.voice.expression["enabled"] is False
    assert settings.voice.expression["enabled"] is True
    assert expression_prosody("Как дела?", tuned.voice).pitch == "+0Hz"


def test_decode_mp3_to_mono_f32(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, what, **kw):
        seen["cmd"] = cmd
        raw = __import__("pathlib").Path(cmd[-1])
        raw.write_bytes(np.array([0.25, -0.5, 0.0], dtype="<f4").tobytes())

    monkeypatch.setattr(voicemod, "run", fake_run)
    monkeypatch.setattr(voicemod, "need_tool", lambda name: name)
    audio = decode_to_f32(tmp_path / "фраза.mp3", tmp_path)
    assert audio.tolist() == pytest.approx([0.25, -0.5, 0.0])
    cmd = seen["cmd"]
    assert cmd[0] == "ffmpeg" and str(tmp_path / "фраза.mp3") in cmd
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-ar") + 1] == "48000"
    assert cmd[cmd.index("-f") + 1] == "f32le"
    assert not (tmp_path / "фраза.f32").exists()


def test_edge_service_probe_without_network():
    import asyncio

    class Up:
        @staticmethod
        async def list_voices():
            return [{"Name": "a"}, {"Name": "b"}]

    class Slow:
        @staticmethod
        async def list_voices():
            await asyncio.sleep(30)
            return []

    assert cli._edge_service(Up, 1) == "сервис отвечает, голосов: 2"
    assert "не ответил" in cli._edge_service(Slow, 0.05)


def test_doctor_edge_row(settings, monkeypatch):
    monkeypatch.setattr(cli, "edge_tts_status",
                        lambda timeout=5.0: (True, "edge-tts 7.0, сервис отвечает (400 голосов)"))
    rows = {name: (ok, note) for name, ok, note in cli.voice_rows(settings)}
    assert rows["Edge TTS"] == (True, "edge-tts 7.0, сервис отвечает (400 голосов)")
    monkeypatch.setattr(cli, "edge_tts_status",
                        lambda timeout=5.0: (True, "edge-tts 7.0, сервис не ответил за 5 с"))
    rows = {name: (ok, note) for name, ok, note in cli.voice_rows(settings)}
    assert rows["Edge TTS"][0] is True and "не ответил" in rows["Edge TTS"][1]
    monkeypatch.setattr(cli, "edge_tts_status",
                        lambda timeout=5.0: (False, "не установлен: py -m pip install edge-tts"))
    rows = {name: (ok, note) for name, ok, note in cli.voice_rows(settings)}
    assert rows["Edge TTS"][0] is False and "edge-tts" in rows["Edge TTS"][1]


def test_edge_long_phrase_is_split_before_other_voice(settings, tmp_path):
    """Dmitry молчит на длинной фразе, а куски отдаёт: голос спикера не меняется."""
    log_file = tmp_path / "job.log"
    heard = []

    def fn(text, voice, rate, pitch, volume):
        heard.append((voice, text))
        if len(text.split()) > 12:
            raise NoAudioReceived("long")
        return _tone(0.1 * len(text.split()))

    long = ("И знаете честно говоря ущерб они нанесли очень в экономическом плане это ну "
            "они уронили несколько серверов ненадолго но экономический ущерб был минимальный.")
    assert all(len(p.split()) <= 12 for p in edge_pieces(long)) and len(edge_pieces(long)) > 1
    eng = EdgeEngine(settings.voice, Log(log_file, echo=False), tmp_path, synth_fn=fn,
                     sleeper=lambda s: None)
    out = eng.synth_with_voices([long], None, [EDGE_VOICES["dmitry"]])
    assert out[0] is not None and len(out[0]) > 0
    assert {v for v, _t in heard} == {EDGE_VOICES["dmitry"]} and eng.fell_back == ""
    assert "по 2 кускам" in log_file.read_text(encoding="utf-8")


def test_deferred_block_recovers_with_main_voice_no_fallback(settings, tmp_path, monkeypatch):
    _patch_mix(monkeypatch)
    # Count attempts per phrase and synthesize in one worker: parallel prefetch made the
    # number of attempts (and so whether the fallback voice kicked in) depend on timing.
    settings.voice.edge_workers = 1
    calls = {}
    heard = []

    def fn(text, voice, rate, pitch, volume):
        calls[text] = calls.get(text, 0) + 1
        heard.append(voice)
        if text.startswith("Раз") and calls[text] <= EDGE_ATTEMPTS + PATIENT:
            raise NoAudioReceived("vpn")
        return _tone(0.3)

    eng = EdgeEngine(settings.voice, Log(None, echo=False), tmp_path, synth_fn=fn,
                     sleeper=lambda s: None)
    vo = VoiceOver(tmp_path, settings, Log(None, echo=False), None, engine=eng)
    track = vo.make(tmp_path, [Cue(0.0, 0.4, "Раз."), Cue(1.3, 1.8, "Два слова.")], 3.0, 1)
    assert [p.status for p in track.phrases] == ["ok", "ok"]
    assert all(p.fallback == "" for p in track.phrases)
    assert set(heard) == {EDGE_VOICES[settings.voice.speaker]}
    assert track.phrases[0].start == 0.0


def test_guard_lists_fallback_voice():
    from konveyer.guard import check_phrases, message
    ph = [{"origin": 135.0, "start": 135.0, "duration": 1.0, "speaker": "S2", "voice": "d",
           "status": "ok", "cue_end": 136.0, "text": "что-то вроде", "dropped": "",
           "fallback": "fr-FR-RemyMultilingualNeural"}]
    text = message(check_phrases(ph, [(135.0, 136.0)]))
    assert "• запасной голос: 2:15" in text
