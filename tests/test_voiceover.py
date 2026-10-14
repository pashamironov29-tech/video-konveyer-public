"""Закадровый перевод: естественный темп, фразы подряд, клип дольше голоса."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from conftest import FakeLLM, make_video, needs_ffmpeg
from konveyer.captions import Cue
from konveyer.hardware import Encoder, encoder_args
from konveyer.jobs import Job
from konveyer.layout import build_graph, geometry
from konveyer.log import Log
from konveyer.render import Renderer
from konveyer.transcript import Segment, Transcript, Word
from konveyer.util import ffprobe
from konveyer.timeline import append_source_tail, keep_intervals, select_expr
from konveyer.voice import (VOICEOVER_GAP, EdgeEngine, Phrase, VoiceOver, VoiceTrack,
                            WordMark, align_cues_to_voice, fade_tail, marks_from_edge_events,
                            plan_clip_tail, sentence_groups, shift_marks, split_audio_parts,
                            write_wav)
from konveyer import voice as voicemod

MIX = 48000


class LenEngine:
    """0.5 с на слово, без своего ускорения."""

    name = "silero"
    rate = MIX
    tag = "len"

    def __init__(self):
        self.calls: list[list[str]] = []

    def synth(self, texts, prosody=None):
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


def test_fade_tail_keeps_length_and_softens_the_end():
    samples = np.ones(MIX, np.float32)
    out = fade_tail(samples, MIX, 0.045)
    assert out.shape == samples.shape
    assert out[0] == pytest.approx(1.0)
    assert abs(float(out[-1])) < 1e-5
    assert out[-10] < out[-2000] < 1.0
    assert fade_tail(np.ones(3, np.float32), MIX).shape == (3,)


def test_plan_clip_tail_prefers_source_then_soft_hold():
    assert plan_clip_tail(30, 10, 10, 14, False) == (4, 0)
    assert plan_clip_tail(12, 10, 10, 14, False) == pytest.approx((2, 2))
    assert plan_clip_tail(10, 10, 10, 14, False) == pytest.approx((0, 4))
    # Паузы внутри клипа хвост не отменяют: исходник продолжается с конца клипа.
    assert plan_clip_tail(30, 10, 8, 11, True) == pytest.approx((3, 0))
    assert plan_clip_tail(12, 10, 8, 14, True) == pytest.approx((2, 4))
    assert plan_clip_tail(30, 10, 10, 10.01, False) == (0.0, 0.0)


def test_append_source_tail_starts_where_the_clip_ended():
    body = [(0.0, 0.5), (1.4, 2.0)]
    assert append_source_tail(body, 2.0, 1.25) == [(0.0, 0.5), (1.4, 2.0), (2.0, 3.25)]
    assert append_source_tail(body, 2.0, 0.0) == body
    assert append_source_tail(None, 2.0, 1.0) is None


def test_build_graph_pads_video_and_audio():
    g = geometry("blur", 1920, 1080)
    graph, has_a = build_graph(
        g, fps=30, src_h=1080, text_filters=[], ass_file=None, keep_expr=None,
        crop_x_expr=None, has_audio=True, duck="0.5", voice_in="1:a",
        pad_video=0.6, pad_audio=0.6, out_dur=3.0)
    assert has_a
    assert "tpad=stop_mode=clone:stop_duration=0.600" in graph
    assert "gblur=sigma=16:enable='gte(t,2.280)'" in graph
    assert "fade=t=out:st=2.400:d=0.600:color=black" in graph
    assert "afade=t=out:st=1.850:d=0.550" in graph
    assert graph.index("afade=") < graph.index("apad=")
    assert "apad=pad_dur=0.600" in graph
    assert "duration=longest" in graph
    with_subs, _ = build_graph(
        g, fps=30, src_h=1080, text_filters=[], ass_file="captions.ass", keep_expr=None,
        crop_x_expr=None, has_audio=False, pad_video=0.6, out_dur=3.0)
    assert with_subs.index("fade=t=out") < with_subs.index("ass=")


def test_voiceover_places_in_order_and_keeps_every_word(settings, tmp_path, monkeypatch):
    _patch_mix(monkeypatch)
    settings.voice.voice_mode = "voiceover"
    log_file = tmp_path / "job.log"
    llm = FakeLLM()
    cues = [Cue(0.0, 0.4, "раз два три"), Cue(0.5, 0.9, "четыре пять")]
    track = VoiceOver(tmp_path, settings, Log(log_file, echo=False), llm,
                      engine=LenEngine()).make(tmp_path, cues, 2.0, 1)
    a, b = track.phrases
    assert a.start == pytest.approx(0.0, abs=0.02)
    assert a.spoken == a.text == "раз два три" and a.dropped == "" and a.tempo == 1.0
    assert a.duration == pytest.approx(1.5, abs=0.03)
    assert b.start == pytest.approx(a.start + a.duration + VOICEOVER_GAP, abs=0.03)
    assert b.start > b.origin and b.status == "lag"
    assert b.spoken == "четыре пять" and b.dropped == ""
    assert track.end == pytest.approx(b.start + b.duration, abs=0.03)
    assert track.end > 2.0
    assert len(track.intervals) == 1
    assert track.intervals[0][1] == pytest.approx(track.end, abs=0.03)
    assert "дольше" in track.note and "отставание" in track.note
    assert llm.calls == []
    log = log_file.read_text(encoding="utf-8")
    assert "обрезка по границе слова" not in log
    assert "отставание фраз: 1" in log
    saved = __import__("json").loads((tmp_path / "voice.json").read_text(encoding="utf-8"))
    assert saved["mode"] == "voiceover"
    assert saved["phrases"][0]["dropped"] == ""

    align_cues_to_voice(cues, track.phrases, to_voice=True)
    assert cues[0].text == "раз два три" and cues[0].start == a.start
    assert cues[0].end == pytest.approx(a.start + a.duration, abs=0.02)
    assert cues[1].text == "четыре пять" and cues[1].start == b.start
    assert cues[1].end == pytest.approx(b.start + b.duration, abs=0.02)
    assert cues[0].end <= cues[1].start + 1e-6


def test_voiceover_subtitle_does_not_keep_the_english_window(settings, tmp_path, monkeypatch):
    _patch_mix(monkeypatch)
    settings.voice.voice_mode = "voiceover"
    cues = [Cue(0.0, 5.0, "Коротко.")]
    track = VoiceOver(tmp_path, settings, Log(None, echo=False), None,
                      engine=LenEngine()).make(tmp_path, cues, 6.0, 1)
    align_cues_to_voice(cues, track.phrases, to_voice=True)
    ph = track.phrases[0]
    assert ph.duration == pytest.approx(0.5, abs=0.03)
    assert cues[0].text == "Коротко."
    assert cues[0].start == pytest.approx(ph.start, abs=0.02)
    assert cues[0].end == pytest.approx(ph.start + ph.duration, abs=0.02)
    assert cues[0].end < 1.0


def test_voiceover_waits_for_english_start_and_splits_sentences(settings, tmp_path, monkeypatch):
    _patch_mix(monkeypatch)
    settings.voice.voice_mode = "voiceover"
    cues = [Cue(1.0, 1.3, "Раз."), Cue(3.0, 3.4, "Два слова.")]
    track = VoiceOver(tmp_path, settings, Log(None, echo=False), FakeLLM(),
                      engine=LenEngine()).make(tmp_path, cues, 6.0, 1)
    a, b = track.phrases
    assert a.start == pytest.approx(1.0, abs=0.02) and a.status == "ok"
    assert b.start == pytest.approx(3.0, abs=0.02) and b.status == "ok"
    assert b.start > a.start + a.duration
    assert len(track.intervals) == 2
    assert track.end < 6.0


def test_voiceover_keeps_tiny_slot_and_bridges_one_sentence(settings, tmp_path, monkeypatch):
    _patch_mix(monkeypatch)
    settings.voice.voice_mode = "voiceover"
    cues = [Cue(0.0, 0.08, "Думаю одно"), Cue(0.1, 0.16, "из пользы")]
    track = VoiceOver(tmp_path, settings, Log(None, echo=False), None,
                      engine=LenEngine()).make(tmp_path, cues, 1.0, 1)
    assert [p.spoken for p in track.phrases] == ["Думаю одно", "из пользы"]
    assert all(p.dropped == "" for p in track.phrases)
    assert track.phrases[1].start >= track.phrases[0].start + track.phrases[0].duration + 0.14
    assert len(track.intervals) == 1


def test_voiceover_edge_does_not_speed_up_or_cut(settings, tmp_path, monkeypatch):
    _patch_mix(monkeypatch)
    settings.voice.voice_mode = "voiceover"
    settings.voice.expression["enabled"] = False
    calls = []

    def fn(text, voice, rate, pitch, volume):
        calls.append({"text": text, "rate": rate})
        return np.full(int(MIX * 2.4), 0.25, np.float32)

    eng = EdgeEngine(settings.voice, Log(None, echo=False), tmp_path, synth_fn=fn,
                     sleeper=lambda s: None)
    llm = FakeLLM()
    line = "Думаю, одно из пользы этой мысли"
    log_file = tmp_path / "job.log"
    track = VoiceOver(tmp_path, settings, Log(log_file, echo=False), llm, engine=eng).make(
        tmp_path, [Cue(0.0, 0.3, line)], 1.0, 1)
    assert len(calls) == 1 and calls[0]["rate"] == "+0%"
    assert calls[0]["text"] == "Думаю одно из пользы этой мысли"
    ph = track.phrases[0]
    assert ph.spoken == line and ph.text == line and "," in ph.text
    assert "," not in ph.said and ph.dropped == "" and ph.tempo == 1.0
    assert ph.duration == pytest.approx(2.4, abs=0.05)
    assert track.end == pytest.approx(2.4, abs=0.05)
    assert llm.calls == []
    logged = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
    assert "обрезка" not in logged
    data, _ = voicemod.read_wav(tmp_path / "voice.wav")
    assert data.size == pytest.approx(2.4 * MIX, abs=80)
    assert abs(float(data[1000])) > 0.1
    assert abs(float(data[int(2.4 * MIX) - 1])) > 0.1


def test_voiceover_speaks_ii_as_two_beats(settings, tmp_path, monkeypatch):
    _patch_mix(monkeypatch)
    settings.voice.voice_mode = "voiceover"
    settings.voice.expression["enabled"] = False
    calls = []

    def fn(text, voice, rate, pitch, volume):
        calls.append(text)
        return np.full(int(MIX * 0.4), 0.2, np.float32)

    eng = EdgeEngine(settings.voice, Log(None, echo=False), tmp_path, synth_fn=fn,
                     sleeper=lambda s: None)
    line = "Это ИИ."
    track = VoiceOver(tmp_path, settings, Log(None, echo=False), None, engine=eng).make(
        tmp_path, [Cue(0.0, 1.0, line)], 4.0, 1)
    assert calls == ["Это и-и."]
    assert track.phrases[0].text == line and track.phrases[0].spoken == line
    assert "и-и" in track.phrases[0].said


def _edge(settings, tmp_path, fn):
    return EdgeEngine(settings.voice, Log(None, echo=False), tmp_path, synth_fn=fn,
                      sleeper=lambda s: None)


def test_sentence_groups_stop_at_punctuation():
    assert sentence_groups(["с точки", "зрения свободы.", "Дальше", "ещё?"]) == [[0, 1], [2, 3]]
    assert sentence_groups(["Одно."]) == [[0]]
    assert continues_quote()


def continues_quote():
    from konveyer.voice import continues_sentence
    return (not continues_sentence("свободы.»") and continues_sentence("с точки")
            and not continues_sentence("Ну…"))


def test_split_audio_uses_word_marks_then_text_length():
    rate = 1000
    audio = np.arange(1000, dtype=np.float32)
    marks = [WordMark("раз", 0.0, 0.2), WordMark("два", 0.2, 0.4), WordMark("три", 0.4, 1.0)]
    parts = split_audio_parts(audio, rate, ["раз два", "три"], marks)
    assert [len(p) for p in parts] == [400, 600]
    assert float(parts[0][0]) == 0 and float(parts[1][0]) == 400
    # чужие границы не подходят — делим по длине текста: 2 и 7 символов
    by_len = split_audio_parts(audio, rate, ["аа", "ббббббб"], [WordMark("чужое", 0.0, 1.0)])
    assert [len(p) for p in by_len] == [222, 778]
    whole = split_audio_parts(audio, rate, ["одно"], marks)
    assert len(whole) == 1 and len(whole[0]) == 1000


def _gray_mean(path, ss: float) -> float:
    raw = path.with_suffix(".gray")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-ss", f"{max(ss, 0):.3f}", "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "gray", str(raw)],
        check=True)
    data = np.fromfile(raw, dtype=np.uint8)
    raw.unlink(missing_ok=True)
    return float(data.mean()) if data.size else 0.0


def test_edge_ticks_and_silence_shift():
    marks = marks_from_edge_events([
        {"type": "audio", "data": b"x"},
        {"type": "WordBoundary", "offset": 4_000_000, "duration": 2_000_000, "text": "раз"},
    ])
    assert len(marks) == 1 and marks[0].text == "раз"
    assert marks[0].start == pytest.approx(0.4) and marks[0].end == pytest.approx(0.6)
    shifted = shift_marks(marks, 0.1, 1.0)
    assert shifted[0].text == "раз"
    assert shifted[0].start == pytest.approx(0.3) and shifted[0].end == pytest.approx(0.5)
    assert shift_marks([WordMark("мимо", 0.0, 0.05)], 0.1, 1.0) == []


def test_voiceover_edge_speaks_one_sentence_in_one_request(settings, tmp_path, monkeypatch):
    _patch_mix(monkeypatch)
    settings.voice.voice_mode = "voiceover"
    calls = []

    def fn(text, voice, rate, pitch, volume):
        calls.append({"text": text, "rate": rate, "pitch": pitch})
        words = text.replace(".", "").replace("?", "").split()
        step = 0.4
        marks = [{"text": w, "start": i * step, "end": (i + 1) * step} for i, w in enumerate(words)]
        return np.full(int(MIX * step * len(words)), 0.3, np.float32), marks

    cues = [Cue(0.0, 0.3, "с точки"), Cue(0.35, 0.6, "зрения свободы.")]
    log_file = tmp_path / "job.log"
    track = VoiceOver(tmp_path, settings, Log(log_file, echo=False), None,
                      engine=_edge(settings, tmp_path, fn)).make(tmp_path, cues, 2.0, 1)
    assert len(calls) == 1
    assert calls[0]["text"] == "с точки зрения свободы."
    assert calls[0]["rate"] == "+0%" and calls[0]["pitch"] == "+0Hz"
    a, b = track.phrases
    assert a.duration == pytest.approx(0.8, abs=0.03)
    assert b.duration == pytest.approx(0.8, abs=0.03)
    assert b.start == pytest.approx(a.start + a.duration, abs=0.03)
    assert a.status == "ok" and b.status == "ok"
    assert len(track.intervals) == 1
    assert track.intervals[0][1] == pytest.approx(b.start + b.duration, abs=0.03)
    data, _rate = voicemod.read_wav(tmp_path / "voice.wav")
    mid = int(a.duration * MIX)
    assert abs(float(data[mid - 30])) > 0.1 and abs(float(data[mid + 30])) > 0.1
    assert abs(float(data[int((b.start + b.duration) * MIX) - 1])) > 0.1
    align_cues_to_voice(cues, track.phrases, to_voice=True)
    assert cues[0].end == pytest.approx(cues[1].start, abs=0.03)
    assert cues[1].text == "зрения свободы."
    assert "предложений одним запросом: 1" in log_file.read_text(encoding="utf-8")

    VoiceOver(tmp_path, settings, Log(None, echo=False), None,
              engine=_edge(settings, tmp_path, fn)).make(tmp_path, cues, 2.0, 1)
    assert len(calls) == 1


def test_voiceover_edge_splits_by_text_length_without_marks(settings, tmp_path, monkeypatch):
    _patch_mix(monkeypatch)
    settings.voice.voice_mode = "voiceover"

    def fn(text, voice, rate, pitch, volume):
        return np.full(int(MIX * 0.9), 0.25, np.float32)

    cues = [Cue(0.0, 0.2, "аа"), Cue(0.2, 0.4, "бббббб.")]
    track = VoiceOver(tmp_path, settings, Log(None, echo=False), None,
                      engine=_edge(settings, tmp_path, fn)).make(tmp_path, cues, 2.0, 1)
    a, b = track.phrases
    assert a.duration == pytest.approx(0.2, abs=0.04)
    assert b.duration == pytest.approx(0.7, abs=0.04)
    assert b.start == pytest.approx(a.start + a.duration, abs=0.03)
    assert len(track.intervals) == 1


def test_voiceover_edge_keeps_gap_between_sentences(settings, tmp_path, monkeypatch):
    _patch_mix(monkeypatch)
    settings.voice.voice_mode = "voiceover"
    calls = []

    def fn(text, voice, rate, pitch, volume):
        calls.append({"text": text, "pitch": pitch})
        return np.full(int(MIX * 0.5), 0.2, np.float32)

    cues = [Cue(0.0, 0.2, "А как"), Cue(0.25, 0.4, "дела?"),
            Cue(2.0, 2.3, "Дальше.")]
    track = VoiceOver(tmp_path, settings, Log(None, echo=False), None,
                      engine=_edge(settings, tmp_path, fn)).make(tmp_path, cues, 4.0, 1)
    by_text = {c["text"]: c for c in calls}
    assert set(by_text) == {"А как дела?", "Дальше."}
    assert by_text["А как дела?"]["pitch"] == "+8Hz"
    assert by_text["Дальше."]["pitch"] == "+0Hz"
    a, b, c = track.phrases
    assert b.start == pytest.approx(a.start + a.duration, abs=0.03)
    assert c.start >= b.start + b.duration + VOICEOVER_GAP - 0.02
    assert c.start == pytest.approx(2.0, abs=0.03)
    assert len(track.intervals) == 2


def test_fit_still_speaks_each_cue(settings, tmp_path, monkeypatch):
    _patch_mix(monkeypatch)
    settings.voice.voice_mode = "fit"
    calls = []

    def fn(text, voice, rate, pitch, volume):
        calls.append(text)
        return np.full(int(MIX * 0.2), 0.2, np.float32)

    cues = [Cue(0.0, 1.5, "с точки"), Cue(2.0, 3.5, "зрения.")]
    VoiceOver(tmp_path, settings, Log(None, echo=False), None,
              engine=_edge(settings, tmp_path, fn)).make(tmp_path, cues, 4.0, 1)
    assert calls == ["с точки", "зрения."]


def _render(settings, tmp_path, seconds, clip_end, extra, tighten=False, words=None):
    settings.voice.voice_mode = "voiceover"
    settings.voice.voice_check = False
    settings.render.width, settings.render.height = 320, 568
    settings.render.caption_size = 28
    settings.render.zoom = 1.0
    settings.render.cover = False
    settings.render.titles = False
    settings.paths.jobs_dir = str(tmp_path / "jobs")
    src = make_video(tmp_path / "src.mp4", seconds=seconds, size="320x240")
    job = Job.create(settings, str(src))
    shutil.copy(src, job.dir / "source.mp4")
    info = ffprobe(job.dir / "source.mp4")
    end = min(clip_end, info["duration"])
    job.data["meta"] = {
        "file": "source.mp4", "width": info["width"], "height": info["height"],
        "duration": info["duration"], "has_audio": True, "fps": 30,
    }
    job.data["clips"] = [{"id": 1, "start": 0.0, "end": end, "kicker": "", "title": "", "bottom": ""}]
    job.save()
    if tighten:
        settings.render.tighten = True
    words = words or [Word(0.2, 0.7, "привет"), Word(0.8, 1.2, "мир")]
    tr = Transcript("ru", [Segment(words[0].start, words[-1].end,
                                   " ".join(w.word for w in words), words)])

    class LongVoice:
        def make(self, folder, cues, total, clip_id):
            voice_end = total + extra
            n = int(round(voice_end * MIX))
            path = folder / "voice.wav"
            write_wav(path, np.full(n, 0.05, np.float32), MIX)
            text = cues[0].text if cues else "привет мир"
            origin = cues[0].start if cues else 0.0
            dur = round(voice_end - origin, 3)
            ph = Phrase(origin, total, text, spoken=text, duration=dur, status="ok",
                        origin=origin, said=text)
            return VoiceTrack(path, [(origin, round(origin + dur, 3))], "brian, закадр", [ph],
                              end=round(origin + dur, 3))

    renderer = Renderer(job, settings, Log(job.log_file, echo=False), tr, {}, None,
                        Encoder("libx264", encoder_args("libx264", 30, "ultrafast")), LongVoice())
    clip = renderer.render(job.data["clips"][0])
    renderer.close()
    return job, clip, info


@needs_ffmpeg
def test_voiceover_extends_with_freeze_when_source_ends(settings, tmp_path):
    job, clip, info = _render(settings, tmp_path, seconds=3, clip_end=99, extra=0.7)
    graph = (job.clip_dir(1) / "filter.txt").read_text(encoding="utf-8")
    assert "fade=t=out" in graph and "color=black" in graph
    assert "gblur=sigma=16" in graph
    assert "afade=t=out" in graph and graph.index("afade=") < graph.index("apad=")
    assert clip["out_duration"] == pytest.approx(info["duration"] + 0.7, abs=0.15)
    probed = ffprobe(job.dir / clip["file"])
    assert probed["duration"] == pytest.approx(clip["out_duration"], abs=0.35)
    log = job.log_file.read_text(encoding="utf-8")
    assert "затемнение" in log and "стоп-кадр" not in log
    ass = (job.clip_dir(1) / "captions.ass").read_text(encoding="utf-8")
    assert "привет" in ass
    picture = job.dir / clip["file"]
    assert _gray_mean(picture, 0.15) > 40
    assert _gray_mean(picture, clip["out_duration"] - 0.04) < 25


@needs_ffmpeg
def test_voiceover_continues_source_before_freeze(settings, tmp_path):
    job, clip, info = _render(settings, tmp_path, seconds=4, clip_end=2.0, extra=1.0)
    graph = (job.clip_dir(1) / "filter.txt").read_text(encoding="utf-8")
    assert "tpad=" not in graph
    assert clip["out_duration"] == pytest.approx(3.0, abs=0.15)
    probed = ffprobe(job.dir / clip["file"])
    assert probed["duration"] == pytest.approx(clip["out_duration"], abs=0.35)
    assert "исходник" in job.log_file.read_text(encoding="utf-8")
    assert "afade=" not in graph and "tpad=" not in graph
    picture = job.dir / clip["file"]
    assert _gray_mean(picture, clip["out_duration"] - 0.08) > 40


@needs_ffmpeg
def test_voiceover_tail_uses_source_even_if_pauses_were_cut(settings, tmp_path):
    words = [Word(0.1, 0.35, "раз"), Word(1.7, 1.95, "два")]
    job, clip, info = _render(settings, tmp_path, seconds=3, clip_end=2.0, extra=1.5,
                              tighten=True, words=words)
    graph = (job.clip_dir(1) / "filter.txt").read_text(encoding="utf-8")
    body = keep_intervals(words, 2.0, settings.render.pause_sec, settings.render.keep_pad)
    assert len(body) > 1
    expr = select_expr(append_source_tail(body, 2.0, 1.0))
    assert expr in graph
    assert "fade=t=out" in graph and "afade=t=out" in graph
    log = job.log_file.read_text(encoding="utf-8")
    assert "исходник" in log and "затемнение" in log and "стоп-кадр" not in log
    assert clip["out_duration"] > kept_ok(body)
    probed = ffprobe(job.dir / clip["file"])
    assert probed["duration"] == pytest.approx(clip["out_duration"], abs=0.4)
    assert info["duration"] > 2.0


def kept_ok(body):
    return sum(b - a for a, b in body)


def test_block_restarts_when_russian_runs_ahead_of_original():
    from konveyer.voice import Phrase, voice_blocks
    # Длинная английская речь без пауз, русский текст короткий: блок не тянется 20 с.
    ph = [Phrase(float(i * 2), 2.0, "Короткая фраза,", origin=float(i * 2),
                 cue_end=float(i * 2) + 1.95, speaker="S2") for i in range(10)]
    blocks = voice_blocks(ph)
    assert len(blocks) > 1
    assert all(ph[b[-1]].origin - ph[b[0]].origin < 8 for b in blocks)


def test_block_restarts_when_russian_falls_behind():
    from konveyer.voice import Phrase, voice_blocks
    long = "Очень длинная русская фраза, которая явно не помещается в две секунды английского."
    ph = [Phrase(float(i * 2), 2.0, long, origin=float(i * 2),
                 cue_end=float(i * 2) + 1.95, speaker="S2") for i in range(6)]
    assert len(voice_blocks(ph)) >= 3

def test_block_never_cuts_after_function_word_or_colon():
    from konveyer.voice import Phrase, voice_blocks, soft_cut_ok, mid_sentence_tail
    texts = ["Именно то, что ты перечислил,", "меня и беспокоит: эти", "огромные рои агентов,",
             "которые сотрудничали друг с другом, тысячи", "их. Конец."]
    ph = [Phrase(float(i * 4), 4.0, t, origin=float(i * 4), cue_end=float(i * 4) + 3.9,
                 speaker="S2") for i, t in enumerate(texts)]
    ends = {b[-1] for b in voice_blocks(ph)}
    assert 1 not in ends and 3 not in ends          # «эти», «тысячи» — не конец куска
    assert not soft_cut_ok("меня и беспокоит: эти", 99, 99)
    assert not soft_cut_ok("рои агентов:", 99, 99)
    assert soft_cut_ok("огромные рои агентов,", 99, 99)
    assert not soft_cut_ok("огромные рои агентов,", 10, 99)
    assert mid_sentence_tail("меня и беспокоит.") == "меня и беспокоит,"


def test_speech_bounds_skip_edge_silence():
    import numpy as np
    from konveyer.voice import _speech_bounds
    a = np.zeros(1000, np.float32)
    a[300:700] = 0.3
    assert _speech_bounds(a) == (300, 700)
    assert _speech_bounds(np.zeros(10, np.float32)) is None


def test_block_prosody_can_slow_down_but_not_below_floor(settings):
    from konveyer.voice import block_prosody
    from konveyer.config import parse_percent
    fast = block_prosody("Фраза.", settings.voice, 1.0, "+0%", 1.2)
    slow = block_prosody("Фраза.", settings.voice, 1.0, "+0%", 1.2, slow=0.5)
    assert parse_percent(slow.rate) < parse_percent(fast.rate)
    assert parse_percent(slow.rate) >= -16


def test_long_block_keeps_voice_pitch(settings):
    from konveyer.voice import block_prosody
    short = block_prosody("Кто?", settings.voice, 1.0, "+0%", 1.2)
    long = block_prosody("Ну не знаю, просто представить что-то вроде: вы пытаетесь установить "
                         "софт в куче разных компаний, да?", settings.voice, 1.0, "+0%", 1.2)
    assert short.pitch == "+8Hz" and long.pitch == "+0Hz" and long.kind == "question"
