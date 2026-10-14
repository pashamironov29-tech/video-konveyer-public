"""Целое видео 16:9: кадр, отрезок, перевод кусками, догон озвучки, пост, Telegram, веб-флаги."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from conftest import FAKE_TOKEN, FakeLLM, FakeSession, FakeTranscriber, ROOT, english_transcript, needs_ffmpeg
from konveyer.captions import Cue
from konveyer.cli import _job_options, build_parser
from konveyer.config import load_settings
from konveyer.fullvideo import (build_wide_graph, chunk_batches, compress_full_video, full_canvas,
                                full_limit_bytes, full_path_text, make_full_post, output_fps,
                                plan_span, render_full, run_watched, span_seconds, tg_encode_commands,
                                tg_scale_filter, topics_from_transcript, translate_long, video_kbps_for)
from konveyer.jobs import Job
from konveyer.log import Log
from konveyer.mediaq import media_issues, tech_issues
from konveyer.pipeline import Pipeline
from konveyer.telegram import Telegram, send_job
from konveyer.transcript import Segment, Transcript, Word
from konveyer.util import KonveyerError, ffprobe, read_json
from konveyer.voice import (Phrase, VoiceOver, capped_boost, read_wav, schedule_voice,
                            strip_speech_punct, voice_blocks)
from konveyer.voicecheck import gate_issues, review_voice_sample, sample_groups
from konveyer.config import parse_percent
from konveyer.voice import EdgeEngine, expression_prosody, write_wav
from konveyer import voice as voicemod

PER_CHAR = 0.06


class FakeEngine:
    name = "silero"
    rate = 48000
    tag = "fake"

    def __init__(self, seconds: float | None = None):
        self.seconds = seconds
        self.calls: list[list[str]] = []

    def synth(self, texts, prosody=None):
        self.calls.append(list(texts))
        out = []
        for i, text in enumerate(texts):
            rate = 1.0
            if prosody is not None:
                rate = max(1.0 + parse_percent(prosody[i].rate) / 100.0, 0.5)
            dur = self.seconds if self.seconds is not None else PER_CHAR * len(text)
            n = max(int(self.rate * dur / rate), 1)
            out.append((0.25 * np.sin(2 * np.pi * 220 * np.arange(n) / self.rate)).astype(np.float32))
        return out


def test_canvas_keeps_source_and_caps_1080():
    same = full_canvas(1280, 720)
    assert (same.width, same.height, same.content_w, same.content_h) == (1280, 720, 1280, 720)
    small = full_canvas(640, 360)
    assert (small.width, small.height) == (640, 360)
    hd = full_canvas(1920, 1080)
    assert (hd.width, hd.height) == (1920, 1080)
    uhd = full_canvas(3840, 2160)
    assert (uhd.width, uhd.height, uhd.content_w, uhd.content_h) == (1920, 1080, 1920, 1080)
    vertical = full_canvas(1080, 1920)
    assert (vertical.width, vertical.height) == (1920, 1080)
    assert vertical.content_w < vertical.width and vertical.content_h <= 1080
    assert vertical.content_w <= 1080 and vertical.content_h <= 1920
    box = full_canvas(800, 600)
    assert box.height == 600 and box.width >= 800 and abs(box.width / box.height - 16 / 9) < 0.02
    assert box.content_w <= 800 and box.content_h <= 600


def test_span_and_parser():
    assert span_seconds(100, None, None) == (0.0, 100.0)
    assert span_seconds(100, 10, 200) == (10.0, 100.0)
    with pytest.raises(KonveyerError):
        span_seconds(10, 8, 8.1)
    a = build_parser().parse_args(["run", "https://youtu.be/x", "--full", "--from", "1:00",
                                   "--to", "2:30", "--voice"])
    opts = _job_options(a)
    assert opts["full"] is True and opts["full_from"] == 60 and opts["full_to"] == 150
    assert opts["voice"] is True
    bare = build_parser().parse_args(["run", "https://youtu.be/x", "--from", "5"])
    with pytest.raises(KonveyerError, match="--full"):
        _job_options(bare)
    bad = build_parser().parse_args(["run", "u", "--full", "--from", "нет"])
    with pytest.raises(KonveyerError):
        _job_options(bad)
    late = build_parser().parse_args(["full", "ролик.mp4", "--to", "12:00", "--no-translate"])
    late.full = True
    opts = _job_options(late)
    assert opts["full"] is True and opts["full_to"] == 720 and opts["translate"] is False


def test_schedule_uses_pause_then_rate_without_growing_forever():
    fit = schedule_voice([0, 5, 10], [2, 2, 2], 15, 2, 1.15)
    assert [p["status"] for p in fit] == ["ok", "ok", "ok"]
    assert [p["start"] for p in fit] == [0, 5, 10]
    assert all(p["tempo"] == 1 for p in fit)

    caught = schedule_voice([0, 10], [8, 2], 20, 2, 1.15)
    assert caught[0]["tempo"] == 1 and caught[0]["status"] == "ok"
    assert caught[1]["start"] == pytest.approx(10, abs=0.05)
    assert caught[1]["lag"] == pytest.approx(0, abs=0.05)
    spilled = schedule_voice([0, 10], [11, 2], 20, 2, 1.15)
    assert spilled[0]["status"] == "pause" and spilled[0]["tempo"] == 1
    assert spilled[1]["lag"] <= 2.05

    origins = [i * 4.0 for i in range(8)]
    steady = schedule_voice(origins, [4.4] * 8, 40, 2.0, 1.15)
    assert max(p["lag"] for p in steady) <= 2.05
    assert all(p["tempo"] <= 1.15 + 1e-6 for p in steady)

    hard = schedule_voice([i * 3.0 for i in range(6)], [6.0] * 6, 30, 2.0, 1.15)
    assert any(p["status"] == "over" for p in hard[:-1])
    assert all(p["tempo"] <= 1.15 + 1e-6 for p in hard)

    tail = schedule_voice([0.0], [5.0], 3.0, 2.0, 1.15)
    assert tail[0]["status"] == "tail" and tail[0]["tempo"] == 1 and tail[0]["duration"] == 5

    skipped = schedule_voice([0, 4, 8], [2, 0, 2], 12, 2, 1.15)
    assert skipped[1]["status"] == "fail"
    assert skipped[2]["start"] == pytest.approx(8, abs=0.05)


def test_capped_boost_respects_edge_max(settings):
    boost = capped_boost(settings.voice, "Смотри!", 1.15)
    factor = 1 + parse_percent(expression_prosody("Смотри!", settings.voice, boost).rate) / 100
    assert factor <= settings.voice.edge_max_rate + 0.011
    assert capped_boost(settings.voice, "Обычная фраза.", 1.0) == 1.0


def test_chunks_and_topics(settings, quiet_log):
    groups = chunk_batches(["а" * 100] * 5, 10, 250)
    assert [len(g) for g in groups] == [2, 2, 1]
    llm = FakeLLM()
    texts = [("hello " * 20).strip()] * 3
    done = translate_long(texts, llm, quiet_log, "", batch=10, max_chars=200)
    assert done == [f"РУ {t}" for t in texts]
    assert len(llm.calls) == 3
    assert all(len(call[1]) < 500 for call in llm.calls)
    tr = english_transcript(30, (100, 100))
    topics = topics_from_transcript(tr, 0, 30, 4)
    assert topics[0]["time"] == "0:00" and len(topics) <= 4
    assert all("—" not in t["time"] and t["title"] for t in topics)


def test_full_config(tmp_path):
    s = load_settings(tmp_path / "нет.toml", root=tmp_path)
    assert s.full.lag_sec == 2 and s.full.telegram_max_mb == 50
    assert s.full.tail_sec == 3 and s.full.voice_rate == "+8%"
    assert s.full.telegram_compress is True and s.full.telegram_target_mb == 48
    assert s.full.min_video_kbps == 300 and s.voice.strip_commas is True
    assert s.full.max_height == 1080 and s.full.voice_check_samples == 4
    text = (ROOT / "konveyer.toml.example").read_text(encoding="utf-8")
    assert "[full]" in text and "telegram_max_mb" in text and "lag_sec" in text
    assert "telegram_compress" in text and "voice_rate" in text and "strip_commas" in text
    toml = tmp_path / "konveyer.toml"
    toml.write_text("[full]\nlag_sec = 1.5\ntelegram_max_mb = 40\n", encoding="utf-8")
    loaded = load_settings(toml, root=tmp_path)
    assert loaded.full.lag_sec == 1.5 and loaded.full.telegram_max_mb == 40
    for bad in ("lag_sec = 0", "max_height = 2160", "telegram_max_mb = 0", "topics = 0"):
        toml.write_text(f"[full]\n{bad}\n", encoding="utf-8")
        with pytest.raises(KonveyerError, match="full"):
            load_settings(toml, root=tmp_path)


def test_graph_is_wide_and_can_hold_the_last_frame():
    graph, has_a = build_wide_graph(
        width=1280, height=720, content_w=1280, content_h=720, fps=30, ass_file="captions.ass",
        has_audio=True, audio_in="0:a", duck="1-0.5*t", voice_in="1:a", out_len=11.2,
        tail=1.2, fade=0.5)
    assert has_a and "pad=1280:720" in graph and "tpad=stop_mode=clone:stop_duration=1.200" in graph
    assert "fade=" in graph and "gblur" not in graph and "crop=" not in graph
    assert "volume='1-0.5*t'" in graph
    plain, audio = build_wide_graph(
        width=640, height=360, content_w=640, content_h=360, fps=30, ass_file=None,
        has_audio=True, audio_in="0:a", duck="", voice_in="", out_len=4, tail=0, fade=0)
    assert audio and "tpad" not in plain and "gblur" not in plain


def test_wide_fps():
    assert output_fps({"fps": 29.97}, 30) == 30
    assert output_fps({"fps": 0}, 25) == 25


def test_heartbeat_writes_progress(tmp_path):
    log = Log(tmp_path / "job.log", echo=False)
    run_watched([sys.executable, "-c", "import time; time.sleep(0.7)"],
                "кодирование", log, tmp_path, 0.15)
    text = (tmp_path / "job.log").read_text(encoding="utf-8")
    assert "кодирование: старт" in text and "ещё идёт" in text


@needs_ffmpeg
def test_make_timed_does_not_cut_words_and_extends(settings, tmp_path):
    settings.voice.edge_max_rate = 1.15
    text = "слово " * 12
    text = text.strip()
    cues = [Cue(0.0, 1.0, text), Cue(2.0, 2.4, "Дальше едем спокойно.")]
    vo = VoiceOver(tmp_path, settings, Log(tmp_path / "job.log", echo=False), None, engine=FakeEngine())
    track = vo.make_timed(tmp_path, cues, 8.0, 1, lag_sec=2.0, max_rate=1.15, progress_every=1)
    assert track is not None
    assert all(not p.dropped and p.spoken == p.text for p in track.phrases if p.status != "fail")
    assert "обрезано" not in track.note
    assert "озвучка: фразы" in (tmp_path / "job.log").read_text(encoding="utf-8")

    long = Cue(0.2, 0.8, ("длинная " * 16).strip())
    vo2 = VoiceOver(tmp_path, settings, Log(None, echo=False), None, engine=FakeEngine())
    tail = vo2.make_timed(tmp_path / "хвост", [long], 2.0, 1, lag_sec=2.0, max_rate=1.15)
    assert tail is not None and tail.phrases[0].status == "tail"
    assert tail.phrases[0].tempo == 1 and tail.phrases[0].dropped == ""
    spoken_words = tail.phrases[0].spoken.split()
    source_words = long.text.split()
    assert spoken_words and spoken_words == source_words[:len(spoken_words)]
    assert len(spoken_words) < len(source_words)
    with wave.open(str(tmp_path / "хвост" / "voice.wav"), "rb") as w:
        dur = w.getnframes() / w.getframerate()
    assert 2.5 < dur <= 2.0 + 3.15


@needs_ffmpeg
def test_make_timed_edge_rate_stays_under_cap(settings, tmp_path):
    class RateEngine(FakeEngine):
        name = "edge"
        rate_fit = True
        tag = "edge:test"

    settings.voice.engine = "edge"
    settings.voice.speaker = "brian"
    settings.voice.edge_max_rate = 1.15
    text = ("фраза " * 20).strip() + "."
    cues = [Cue(0.0, 1.0, text), Cue(3.0, 3.4, "Хватит.")]
    engine = RateEngine()
    vo = VoiceOver(tmp_path, settings, Log(None, echo=False), None, engine=engine)
    track = vo.make_timed(tmp_path, cues, 12.0, 1, lag_sec=2.0, max_rate=1.15)
    assert track is not None
    assert track.phrases[0].dropped == ""
    assert all(len(call) == 1 for call in engine.calls)
    factor = 1 + parse_percent(track.phrases[0].rate) / 100
    assert factor <= 1.15 + 1e-6
    assert track.phrases[1].start - track.phrases[1].origin <= 2.2
    assert all(p.status != "over" for p in track.phrases)
    saved = read_json(tmp_path / "voice.json")
    assert saved["mode"] == "full"


def test_sample_groups_and_review(settings, tmp_path):
    phrases = [Phrase(i * 10, 2, f"фраза {i}", duration=2.0, status="ok", spoken=f"фраза {i}")
               for i in range(10)]
    groups = sample_groups(phrases, 3, 5)
    assert len(groups) >= 2
    flat = [i for group in groups for i in group]
    assert flat == sorted(flat) and len(flat) == len(set(flat))
    assert groups[0][0] < groups[-1][0]
    short = [Phrase(0, 1, "а", duration=1, status="ok"), Phrase(2, 1, "б", duration=1, status="ok")]
    assert sample_groups(short, 4, 18) == [[0, 1]]

    rate = 48000
    write_wav(tmp_path / "voice.wav", np.zeros(rate * 30, np.float32), rate)
    heard = {}

    def hear(path, rows):
        with wave.open(str(path), "rb") as w:
            heard["dur"] = w.getnframes() / w.getframerate()
        return ["фраза"] * len(rows)

    settings.voice.voice_check = True
    result = review_voice_sample(tmp_path, phrases, settings, Log(None, echo=False), FakeLLM(), 1,
                                 samples=3, window=4, hear=hear)
    assert result["sampled"] is True and result["fragments"] >= 2
    assert result["sample_phrases"] < len(phrases)
    assert heard["dur"] < 20
    assert (tmp_path / "voice_check.json").is_file()


def test_gate_reads_full_folder(settings):
    job = Job.create(settings, "a.mp4")
    folder = job.dir / "full"
    folder.mkdir()
    (folder / "voice_check.json").write_text(json.dumps({
        "skipped": False, "score": 3, "phrases": [],
    }), encoding="utf-8")
    job.data["options"] = {"full": True}
    job.data["clips"] = [{"id": 1, "full": True, "file": "full/video.mp4"}]
    settings.voice.voice_check = True
    settings.voice.voice_check_min = 7
    # целое видео оценка не держит и цифру в Telegram не шлёт
    assert gate_issues(job, settings) == []
    from konveyer.voicecheck import voice_notes
    assert "из 10" not in voice_notes(job, settings)


def test_tech_skips_shorts_limit():
    info = {"width": 1280, "height": 720, "duration": 600, "bit_rate": 2_000_000}
    quiet = tech_issues(info, width=1280, height=720, shorts_max=60, shorts_pack=False, readable=True)
    loud = tech_issues(info, width=1280, height=720, shorts_max=60, shorts_pack=True, readable=True)
    assert not any("Shorts" in item for item in quiet)
    assert any("Shorts" in item for item in loud)


def test_fallback_post_has_timecodes(settings, quiet_log):
    job = Job.create(settings, "a.mp4")
    job.data["meta"] = {"title": "Интервью", "uploader": "Автор", "duration": 30, "description": ""}
    job.data["clips"] = [{"id": 1, "start": 0, "end": 30, "full": True}]
    post = make_full_post(job, settings, quiet_log, None, english_transcript(30, (100, 100)))
    text = job.post_md.read_text(encoding="utf-8")
    assert "Таймкоды" in text and "0:00" in text and "#shorts" not in text
    assert post["ai"] is False and post["topics"]


def test_post_from_llm(settings, quiet_log):
    job = Job.create(settings, "a.mp4")
    job.data["meta"] = {"title": "Интервью", "uploader": "Тестовый автор", "duration": 20,
                        "description": ""}
    job.data["clips"] = [{"id": 1, "start": 0, "end": 20, "full": True}]
    post = make_full_post(job, settings, quiet_log, FakeLLM(), english_transcript())
    assert "Целый ролик" in post["post"]
    assert post["topics"][0]["title"] == "Начало"
    assert "#shorts" not in job.post_md.read_text(encoding="utf-8")


@needs_ffmpeg
def test_send_full_video_or_path(settings, wide_video):
    job = Job.create(settings, "a.mp4")
    folder = job.dir / "full"
    folder.mkdir()
    shutil.copy(wide_video, folder / "video.mp4")
    job.data["options"] = {"full": True}
    job.data["meta"] = {"title": "Интервью", "uploader": "Автор"}
    job.data["clips"] = [{"id": 1, "full": True, "file": "full/video.mp4", "width": 640, "height": 360}]
    job.post_md.write_text("Текст поста для ВК", encoding="utf-8")
    job.save()
    session = FakeSession()
    sent = send_job(job, settings, Log(None, echo=False), tg=Telegram(FAKE_TOKEN, "1", session),
                    check=False)
    assert sent["clips"] == [1] and sent["post"] is True
    assert session.posts[0]["url"].endswith("/sendVideo")
    assert not (folder / "video_tg.mp4").exists()

    # повтор без --resend ничего не шлёт: видео и пост уже отмечены в job.json
    session = FakeSession()
    send_job(job, settings, Log(None, echo=False), tg=Telegram(FAKE_TOKEN, "1", session),
             check=False)
    assert session.posts == []

    settings.full.telegram_max_mb = 0.01
    session = FakeSession()
    sent = send_job(job, settings, Log(None, echo=False), tg=Telegram(FAKE_TOKEN, "1", session),
                    check=False, resend=True)
    assert sent["clips"] == [] and sent["post"] is True and "path" in sent
    assert session.posts[0]["url"].endswith("/sendMessage")
    body = session.posts[0]["data"]["text"]
    assert "не сжимаю" in body and "Текст поста для ВК" in body and str(folder / "video.mp4") in body
    assert not any(p["url"].endswith("/sendVideo") for p in session.posts)
    assert not (folder / "video_tg.mp4").exists()
    note = full_path_text(folder / "video.mp4", "пост", settings)
    assert "Файл:" in note


@needs_ffmpeg
def test_media_full_ignores_vertical_and_shorts(settings, wide_video, quiet_log):
    job = Job.create(settings, "a.mp4")
    folder = job.dir / "full"
    folder.mkdir()
    shutil.copy(wide_video, folder / "video.mp4")
    job.data["options"] = {"full": True}
    job.data["clips"] = [{"id": 1, "full": True, "file": "full/video.mp4", "width": 640, "height": 360}]
    settings.render.width, settings.render.height = 1080, 1920
    settings.quality.shorts_max_sec = 5
    settings.post.shorts_pack = True
    text = "\n".join(media_issues(job, settings, quiet_log, None))
    assert "Shorts" not in text and "разрешение" not in text


@needs_ffmpeg
def test_pipeline_full_is_horizontal(settings, wide_video, fast_encoder):
    job = Job.create(settings, str(wide_video), {"full": True})
    log = Log(job.log_file, echo=False)
    Pipeline(job, settings, log, llm=FakeLLM(), transcriber=FakeTranscriber(),
             encoder=fast_encoder).run()
    data = Job.load(job.dir).data
    assert data["state"] == "done"
    assert len(data["clips"]) == 1 and data["clips"][0]["full"] is True
    out = job.dir / "full" / "video.mp4"
    info = ffprobe(out)
    assert (info["width"], info["height"]) == (640, 360)
    assert info["duration"] == pytest.approx(20, abs=0.4)
    assert info["has_audio"]
    graph = (job.dir / "full" / "filter.txt").read_text(encoding="utf-8")
    assert "pad=" in graph and "gblur" not in graph and "crop=" not in graph
    ass = (job.dir / "full" / "captions.ass").read_text(encoding="utf-8")
    assert "PlayResX: 640" in ass and "PlayResY: 360" in ass and ",2,60,60," in ass
    assert "РУ" in ass
    assert not list((job.dir / "full").glob("t_*.txt"))
    post = job.post_md.read_text(encoding="utf-8")
    assert "Целый ролик" in post and "Таймкоды" in post and "#shorts" not in post
    logged = job.log_file.read_text(encoding="utf-8")
    assert "режим целого видео" in logged and "готово за" in logged
    assert "целое видео:" in logged and "кодирование" in logged and "перевод: кусок" in logged
    assert all(data["steps"][name]["sec"] >= 0 for name in data["steps"])


@needs_ffmpeg
def test_pipeline_from_to(settings, wide_video, fast_encoder):
    job = Job.create(settings, str(wide_video), {"full": True, "full_from": 1, "full_to": 6})
    Pipeline(job, settings, Log(job.log_file, echo=False), llm=FakeLLM(),
             transcriber=FakeTranscriber(), encoder=fast_encoder).run()
    clip = Job.load(job.dir).data["clips"][0]
    assert clip["start"] == 1 and clip["end"] == 6
    info = ffprobe(job.dir / clip["file"])
    assert info["duration"] == pytest.approx(5, abs=0.4)
    assert (info["width"], info["height"]) == (640, 360)


def _gray_mean(path, t: float) -> float:
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path),
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ])
    return sum(raw) / max(len(raw), 1)


@needs_ffmpeg
def test_render_holds_last_frame_for_voice(settings, wide_video, fast_encoder):
    job = Job.create(settings, str(wide_video), {"full": True, "full_to": 2, "voice": True})
    log = Log(job.log_file, echo=False)
    from konveyer.ingest import ingest
    ingest(job, settings, log)
    plan_span(job, settings, log, Transcript("en"))
    settings.voice.enabled = True
    settings.voice.voice_check = False
    tr = Transcript("en", [Segment(0.2, 1.2, "Hello there.",
                                   [Word(0.2, 1.2, " Hello there.")])])
    render_full(job, settings, log, tr, FakeLLM(), encoder=fast_encoder,
                voice_engine=FakeEngine(seconds=3.0))
    info = ffprobe(job.dir / "full" / "video.mp4")
    assert info["duration"] > 2.6
    graph = (job.dir / "full" / "filter.txt").read_text(encoding="utf-8")
    assert "tpad=stop_mode=clone" in graph and "fade=t=out" in graph and "gblur" not in graph
    out = job.dir / "full" / "video.mp4"
    assert _gray_mean(out, 1.0) > _gray_mean(out, info["duration"] - 0.08) + 30
    saved = read_json(job.dir / "full" / "voice.json")
    assert saved["phrases"][0]["dropped"] == ""
    assert saved["phrases"][0]["spoken"]


def _patch_mix(monkeypatch):
    def fake_to_mix(path, tempo, workdir):
        data, _rate = voicemod.read_wav(path)
        if tempo > 1.0001:
            data = data[:max(1, int(len(data) / tempo))]
        return np.asarray(data, np.float32)

    monkeypatch.setattr(voicemod, "to_mix", fake_to_mix)


def test_strip_commas_keeps_sentence_marks():
    assert strip_speech_punct("по-русски, как дела?") == "по-русски как дела?"
    assert strip_speech_punct("до конца — и… обратно.") == "до конца и обратно."
    assert strip_speech_punct("конец...") == "конец"
    assert strip_speech_punct("Я просто хочу, чтобы.") == "Я просто хочу чтобы."


def test_strip_commas_off_sends_the_comma(settings, tmp_path, monkeypatch):
    _patch_mix(monkeypatch)
    settings.voice.voice_mode = "voiceover"
    settings.voice.strip_commas = False
    settings.voice.expression["enabled"] = False
    calls = []

    def fn(text, voice, rate, pitch, volume):
        calls.append(text)
        return np.full(int(48000 * 0.3), 0.2, np.float32)

    line = "Думаю, одно."
    VoiceOver(tmp_path, settings, Log(None, echo=False), None,
              engine=EdgeEngine(settings.voice, Log(None, echo=False), tmp_path, synth_fn=fn,
                                sleeper=lambda s: None)).make(
        tmp_path, [Cue(0.0, 0.4, line)], 4.0, 1)
    assert calls == [line]


def test_voice_blocks_split_on_long_pause():
    def phrase(origin, end, text, speaker=""):
        return Phrase(origin, 1.0, text, origin=origin, cue_end=end, speaker=speaker)

    blocks = voice_blocks([
        phrase(0.0, 0.4, "раз два."),
        phrase(0.5, 0.9, "три."),
        phrase(2.0, 2.4, "четыре."),
    ])
    assert blocks == [[0], [1], [2]]          # каждое предложение со своего места
    same = voice_blocks([
        phrase(0.0, 0.4, "раз.", "S1"),
        phrase(0.45, 0.8, "два.", "S1"),
        phrase(0.9, 1.2, "три.", "S2"),
    ])
    assert same == [[0], [1], [2]]
    # долгая пауза посреди предложения кусок не рвёт
    assert voice_blocks([phrase(0.0, 0.4, "раз два"), phrase(2.0, 2.4, "три.")]) == [[0, 1]]


def test_long_phrases_do_not_accumulate_lag(settings, tmp_path, monkeypatch):
    _patch_mix(monkeypatch)
    line = ("Я просто хочу, чтобы эта длинная мысль, наконец, дошла до людей, "
            "и первое слово не пропало.")
    assert "," in line and len(line) > 80
    cues = [Cue(i * 3.0, i * 3.0 + 0.8, line) for i in range(8)]
    vo = VoiceOver(tmp_path, settings, Log(None, echo=False), None, engine=FakeEngine())
    track = vo.make_timed(tmp_path, cues, 24.0, 1, lag_sec=2.0, max_rate=1.15)
    assert track is not None
    lags = [p.start - p.origin for p in track.phrases]
    assert max(lags) <= 2.2
    assert track.end <= 24.0 + 3.15
    assert all(p.status != "over" and p.dropped == "" for p in track.phrases)
    assert all("," in p.text and "," not in p.said for p in track.phrases)
    for prev, nxt in zip(track.phrases, track.phrases[1:]):
        assert nxt.start >= prev.start + prev.duration - 0.02
    data, _rate = read_wav(tmp_path / "voice.wav")
    assert float(np.max(np.abs(data))) < 0.4


def test_block_keeps_quiet_onset_and_does_not_overlap(settings, tmp_path, monkeypatch):
    _patch_mix(monkeypatch)
    settings.voice.expression["enabled"] = False
    # Edge synthesizes phrases in parallel workers: pick the head level by text, not call order.
    heads = {"ту же задачу.": 0.004, "и что масштаб.": 0.006}
    calls = []

    def fn(text, voice, rate, pitch, volume):
        calls.append(text)
        audio = np.full(int(48000 * 2.0), 0.45, np.float32)
        audio[:int(0.08 * 48000)] = heads[text]
        return audio

    cues = [Cue(0.0, 0.4, "ту же задачу."), Cue(1.2, 1.6, "и что масштаб.")]
    eng = EdgeEngine(settings.voice, Log(None, echo=False), tmp_path, synth_fn=fn,
                     sleeper=lambda s: None)
    track = VoiceOver(tmp_path, settings, Log(None, echo=False), None, engine=eng).make_timed(
        tmp_path, cues, 6.0, 1, lag_sec=2.0, max_rate=1.15)
    assert track is not None
    assert sorted(calls) == sorted(heads)
    assert track.phrases[0].text == "ту же задачу."
    data, rate = read_wav(tmp_path / "voice.wav")
    head = int(0.06 * rate)
    assert float(np.mean(np.abs(data[:head]))) == pytest.approx(0.004, abs=0.0008)
    assert abs(float(data[int(1.9 * rate)])) > 0.4
    assert abs(float(data[int(2.02 * rate)])) < 0.001
    onset = int(2.04 * rate) + 10
    assert abs(float(data[onset])) == pytest.approx(0.006, abs=0.0008)
    assert track.phrases[1].start >= track.phrases[0].start + track.phrases[0].duration - 0.02


def test_tg_bitrate_and_commands():
    nine = video_kbps_for(9 * 60, 48 * 1024 * 1024, 128, 300)
    assert nine is not None and nine > 300
    assert video_kbps_for(3 * 3600, 48 * 1024 * 1024, 128, 300) is None
    assert tg_scale_filter(1080) == "scale=-2:720:flags=lanczos"
    assert "," not in tg_scale_filter(1080)
    assert "," not in tg_scale_filter(360)
    cmds = tg_encode_commands(Path("a.mp4"), Path("video_tg.mp4"), height=1080, kbps=600,
                              has_audio=True, nvenc=False, passlog=Path("tgpass"), ffmpeg="ffmpeg")
    assert [c[c.index("-pass") + 1] for c in cmds] == ["1", "2"]
    assert "aac" in cmds[1] and "128k" in cmds[1] and "+faststart" in cmds[1]
    assert "scale=-2:720:flags=lanczos" in cmds[1]
    nv = tg_encode_commands(Path("a.mp4"), Path("video_tg.mp4"), height=1080, kbps=600,
                            has_audio=True, nvenc=True, passlog=Path("tgpass"), ffmpeg="ffmpeg")
    assert len(nv) == 1
    flat = " ".join(nv[0])
    assert "h264_nvenc" in flat and "fullres" in flat and "720" in flat
    assert "aac" in flat and "128k" in flat and "+faststart" in flat


def test_compress_skips_when_bitrate_too_low(settings, tmp_path):
    src = tmp_path / "video.mp4"
    src.write_bytes(b"not a video")
    calls = []

    def probe(_path):
        return {"width": 1920, "height": 1080, "duration": 3 * 3600, "has_audio": True}

    def run(cmd, what):
        calls.append(cmd)

    out = compress_full_video(src, settings, Log(None, echo=False), run=run, probe=probe, nvenc=False)
    assert out is None and calls == []
    assert not (tmp_path / "video_tg.mp4").exists()


@needs_ffmpeg
def test_compress_short_file(settings, tmp_path):
    from conftest import make_video
    src = make_video(tmp_path / "video.mp4", 0.6, "320x240")
    out = compress_full_video(src, settings, Log(None, echo=False), nvenc=False)
    assert out is not None and out.name == "video_tg.mp4"
    info = ffprobe(out)
    assert info["height"] <= 720 and info["has_audio"]
    assert out.stat().st_size <= full_limit_bytes(settings)
    assert not list(tmp_path.glob("tgpass*"))


@needs_ffmpeg
def test_send_full_uses_compressed_copy(settings, wide_video, monkeypatch):
    job = Job.create(settings, "a.mp4")
    folder = job.dir / "full"
    folder.mkdir()
    original = folder / "video.mp4"
    original.write_bytes(b"\0" * (60 * 1024 * 1024))
    job.data["options"] = {"full": True}
    job.data["meta"] = {"title": "Интервью", "uploader": "Автор"}
    job.data["clips"] = [{"id": 1, "full": True, "file": "full/video.mp4"}]
    job.post_md.write_text("Текст поста для ВК", encoding="utf-8")
    job.save()

    def fake_compress(src, settings, log, **kwargs):
        dest = src.with_name("video_tg.mp4")
        shutil.copy(wide_video, dest)
        return dest

    import konveyer.fullvideo as fullvideo
    monkeypatch.setattr(fullvideo, "compress_full_video", fake_compress)
    session = FakeSession()
    sent = send_job(job, settings, Log(None, echo=False), tg=Telegram(FAKE_TOKEN, "1", session),
                    check=False)
    assert sent["clips"] == [1] and sent["post"] is True
    assert sent["compressed"] == "full/video_tg.mp4"
    assert str(original.resolve()) in sent["path"]
    assert session.posts[0]["url"].endswith("/sendVideo")
    assert session.posts[0]["files"]["video"][0] == "video_tg.mp4"
    body = session.posts[1]["data"]["text"]
    assert "сжатая копия" in body and "Текст поста для ВК" in body
    assert str(original.resolve()) in body


def test_full_post_rewritten_then_simple_and_video_still_sent(settings, wide_video, monkeypatch):
    from konveyer import fullvideo, pipeline, quality
    from konveyer.quality import QualityResult
    job = Job.create(settings, "a.mp4")
    folder = job.dir / "full"
    folder.mkdir()
    shutil.copy(wide_video, folder / "video.mp4")
    job.data["options"] = {"full": True}
    job.data["meta"] = {"title": "Интервью", "uploader": "Автор", "duration": 20}
    job.data["clips"] = [{"id": 1, "full": True, "file": "full/video.mp4", "width": 640, "height": 360}]
    job.data["guard"] = {"message": "Сторож: замечания по озвучке\n• отставание: 0:13 (1.7 с)"}
    job.post_md.write_text("Пост с выдумкой про ботнет", encoding="utf-8")
    job.save()
    checks, rewrites = [], []
    monkeypatch.setattr(quality, "assess_publish", lambda *a, **k: checks.append(1) or
                        QualityResult(False, "выдуман ботнет", "deepseek"))
    monkeypatch.setattr(fullvideo, "make_full_post",
                        lambda job, s, log, llm, tr, avoid="": rewrites.append(avoid))
    monkeypatch.setattr(pipeline, "load_transcript", lambda job: english_transcript())
    session = FakeSession()
    sent = send_job(job, settings, Log(None, echo=False), tg=Telegram(FAKE_TOKEN, "1", session),
                    llm=object())
    assert rewrites == ["выдуман ботнет", "выдуман ботнет"] and len(checks) == 4
    assert sent["clips"] == [1] and sent["post"] is True
    assert session.posts[0]["url"].endswith("/sendVideo")
    texts = [p["data"].get("text", "") for p in session.posts if p["url"].endswith("/sendMessage")]
    assert texts[0].startswith("🤖 Интервью") and "⏱ Таймкоды" in texts[0]   # простой пост
    assert "#ИИ #нейросети" in texts[0]
    assert "Видео целиком: Автор" in texts[0]
    assert texts[-1].startswith("Сторож:") and "Пост: проверка не пройдена" in texts[-1]
    assert not any("На проверку" in x for x in texts)


def test_fabrication_needs_quote_from_english_or_russian():
    from konveyer.quality import unsupported_claims
    en = "They hacked OpenAI itself. The damage was minimal."
    ru = "Взломали саму Опен Эй Ай. Ущерб был минимальным."
    data = {"claims": [
        {"claim": "взлом OpenAI", "quote": "They hacked OpenAI itself"},
        {"claim": "ущерб минимален", "quote": "Ущерб был минимальным"},
        {"claim": "ущерб в сотни миллиардов", "quote": ""},
        {"claim": "ботнет", "quote": "a botnet took down banks"},
    ]}
    assert unsupported_claims(data, en, ru) == ["ущерб в сотни миллиардов", "ботнет"]


def test_topics_sorted_unique_within_length():
    from konveyer.fullvideo import parse_topics
    raw = [{"time": "0:00", "title": "А"}, {"time": "1:40", "title": "Б"},
           {"time": "9:00", "title": "Лишнее"}, {"time": "0:08", "title": "В"},
           {"time": "0:08", "title": "Повтор"}]
    got = parse_topics(raw, 10, 240.0)
    assert [x["time"] for x in got] == ["0:00", "0:08", "1:40"]


def test_fabrication_check_accepts_description_and_plain_headings():
    from konveyer.quality import unsupported_claims
    en = "They hacked OpenAI's monitoring infrastructure."
    meta = "Описание: Amodei is arguing that the race is in dire need of a deliberate slowdown."
    data = {"claims": [
        {"claim": "призывает замедлиться", "quote": "in dire need of a deliberate slowdown"},
        {"claim": "Таймкод 5:10 — Агенты взламывают инфраструктуру OpenAI", "quote": ""},
        {"claim": "Таймкод 6:30 — Призыв к замедлению разработки", "quote": ""},
        {"claim": "Таймкод 7:00 — Microsoft теряет 500 миллиардов", "quote": ""},
    ]}
    assert unsupported_claims(data, en, "", meta) == ["Таймкод 7:00 — Microsoft теряет 500 миллиардов"]
