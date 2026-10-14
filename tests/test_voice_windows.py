"""Гибкие окна озвучки: пауза соседа, сдвиг, обрезка только по целому слову, субтитры следом."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import FakeLLM
from konveyer.captions import Cue, ass_time
from konveyer.jobs import Job
from konveyer.layout import geometry
from konveyer.log import Log
from konveyer.render import Renderer
from konveyer.transcript import Segment, Transcript, Word
from konveyer.voice import (VOICE_BORROW, Phrase, VoiceOver, align_cues_to_voice, fit_words,
                            max_phrase_ends)
from konveyer import voice as voicemod

MIX = 48000


class LenEngine:
    """0.5 с на слово, без ускорения своим rate. Как Silero для сборки окна."""

    name = "silero"
    rate = MIX
    tag = "len"

    def __init__(self):
        self.calls: list[list[str]] = []

    def synth(self, texts):
        self.calls.append(list(texts))
        out = []
        for text in texts:
            n = max(1, int(self.rate * 0.5 * max(len(text.split()), 1)))
            out.append(np.full(n, 0.2, np.float32))
        return out


def _patch_mix(monkeypatch):
    def fake_to_mix(path, tempo, workdir):
        data, _rate = voicemod.read_wav(path)
        if tempo > 1.0001:
            data = data[:max(1, int(len(data) / tempo))]
        return np.asarray(data, np.float32)

    monkeypatch.setattr(voicemod, "to_mix", fake_to_mix)


def test_max_phrase_ends_borrows_only_real_slack():
    ends = max_phrase_ends([0.0, 2.0], [1.0, 0.4], 5.0)
    assert ends == pytest.approx([2.78, 4.98])
    tight = max_phrase_ends([0.0, 1.2], [1.9, 1.9], 3.0)
    assert tight == pytest.approx([1.18, 2.98])


def test_fit_words_keeps_a_whole_word():
    keep, drop = fit_words("раз два три", 1.0, 2.0)
    assert (keep, drop) == ("раз два три", "")
    keep, drop = fit_words("раз два три четыре пять", 2.0, 0.9)
    assert drop and keep.split() == ["раз", "два", "три", "четыре", "пять"][:len(keep.split())]
    assert " ".join(keep.split() + drop.split()) == "раз два три четыре пять"
    assert fit_words("супердлинное", 5.0, 0.2) == ("супердлинное", "")


def test_align_cues_follows_voice_and_spoken_text():
    cues = [Cue(0.2, 1.4, "раз два три"), Cue(1.6, 2.4, "дальше")]
    phrases = [
        Phrase(0.55, 1.0, "раз два три", spoken="раз два", duration=2.0, status="cut",
               dropped="три", origin=0.2),
        Phrase(1.7, 0.8, "дальше", spoken="дальше", duration=0.4, status="ok", origin=1.6),
    ]
    align_cues_to_voice(cues, phrases)
    assert cues[0].start == 0.55 and cues[0].text == "раз два"
    assert cues[1].start == 1.7 and cues[1].text == "дальше"
    assert cues[0].end == cues[1].start


def test_voice_borrows_pause_and_shifts_next(settings, tmp_path, monkeypatch):
    _patch_mix(monkeypatch)
    settings.voice.voice_mode = "fit"
    settings.voice.max_tempo = 1.3
    log_file = tmp_path / "job.log"
    cues = [Cue(0.0, 0.4, "раз два три четыре пять"), Cue(1.2, 1.6, "да")]
    track = VoiceOver(tmp_path, settings, Log(log_file, echo=False), None,
                      engine=LenEngine()).make(tmp_path, cues, 4.0, 1)
    a, b = track.phrases
    assert a.status == "tempo" and a.tempo == 1.3 and a.dropped == ""
    assert a.spoken == a.text
    assert a.duration == pytest.approx(1.923, abs=0.01)
    assert b.start == round(a.start + a.duration + 0.02, 3)
    assert b.start > b.origin
    assert "сдвинуто" in track.note
    assert "сдвинуто фраз" in log_file.read_text(encoding="utf-8")
    assert "обрезка по границе слова" not in log_file.read_text(encoding="utf-8")


def test_voice_cuts_whole_words_when_nothing_to_borrow(settings, tmp_path, monkeypatch):
    _patch_mix(monkeypatch)
    settings.voice.voice_mode = "fit"
    settings.voice.max_tempo = 1.3
    log_file = tmp_path / "job.log"
    text = "раз два три четыре пять"
    cues = [Cue(0.0, 0.4, text), Cue(1.2, 1.6, text)]
    track = VoiceOver(tmp_path, settings, Log(log_file, echo=False), None,
                      engine=LenEngine()).make(tmp_path, cues, 3.0, 1)
    a = track.phrases[0]
    kept = a.spoken.split()
    assert a.status == "cut" and a.dropped
    assert kept == text.split()[:len(kept)] and len(kept) < len(text.split())
    assert a.duration == pytest.approx(0.5 * len(kept) / 1.3, abs=0.02)
    log = log_file.read_text(encoding="utf-8")
    assert "обрезка по границе слова" in log and a.dropped in log


def test_voice_starts_early_into_previous_pause(settings, tmp_path, monkeypatch):
    _patch_mix(monkeypatch)
    settings.voice.voice_mode = "fit"
    settings.voice.max_tempo = 1.3
    cues = [Cue(0.0, 0.3, "да"), Cue(2.0, 2.4, "раз два три четыре пять")]
    track = VoiceOver(tmp_path, settings, Log(None, echo=False), None,
                      engine=LenEngine()).make(tmp_path, cues, 2.8, 1)
    b = track.phrases[1]
    assert b.origin == 2.0
    assert b.start <= b.origin - 0.5
    assert b.start >= b.origin - VOICE_BORROW - 0.02


def test_captions_follow_shifted_voice(settings):
    words = [Word(0.2, 0.6, "hello"), Word(0.7, 1.1, "there.")]
    tr = Transcript("en", [Segment(0.2, 1.1, "hello there.", words)])
    job = Job.create(settings, "a.mp4")
    renderer = Renderer(job, settings, Log(None, echo=False), tr, {}, FakeLLM())
    cues = renderer.subtitle_cues(words, None)
    assert len(cues) == 1 and cues[0].start == 0.2
    spoken = " ".join(cues[0].text.split()[:-1])
    renderer._voice_phrases = [
        Phrase(0.55, 1.0, cues[0].text, spoken=spoken, duration=1.1, status="cut",
               dropped=cues[0].text.split()[-1], origin=cues[0].start),
    ]
    ass, note = renderer.captions(words, None, geometry("blur", 1920, 1080))
    assert note == "перевод на русский"
    assert ass_time(0.55) in ass
    assert spoken in ass and cues[0].text not in ass
