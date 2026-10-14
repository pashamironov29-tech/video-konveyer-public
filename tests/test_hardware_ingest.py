"""Выбор кодировщика и устройства, скачивание (команда yt-dlp) и локальные файлы."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import needs_ffmpeg
from konveyer import transcribe as trmod
from konveyer.hardware import ENCODER_ORDER, Encoder, detect_encoder, encoder_args
from konveyer.ingest import ingest, meta_from_info, take_local, ytdlp_command, ytdlp_format
from konveyer.jobs import Job
from konveyer.transcribe import WhisperTranscriber, pick_device, transcribe_job
from konveyer.transcript import Transcript
from konveyer.util import KonveyerError, read_json


# ------------------------------------------------------------------ кодировщик

def runner_ok(*names):
    tried = []

    def run(cmd):
        name = cmd[cmd.index("-c:v") + 1]
        tried.append(name)
        return 0 if name in names else 1
    run.tried = tried
    return run


def test_detect_prefers_nvenc():
    r = runner_ok("h264_nvenc", "h264_qsv")
    assert detect_encoder(runner=r).name == "h264_nvenc" and r.tried == ["h264_nvenc"]


def test_detect_laptop_qsv():
    r = runner_ok("h264_qsv")
    enc = detect_encoder(runner=r)
    assert enc.name == "h264_qsv" and enc.hardware and enc.title == "Intel Quick Sync"
    assert r.tried == ["h264_nvenc", "h264_qsv"]
    assert "nv12" in enc.args


def test_detect_falls_back_to_cpu():
    r = runner_ok()
    enc = detect_encoder(runner=r)
    assert enc.name == "libx264" and not enc.hardware
    assert r.tried == list(ENCODER_ORDER[:-1])


def test_detect_explicit_preference():
    assert detect_encoder("h264_amf", runner=runner_ok("h264_amf")).name == "h264_amf"
    assert detect_encoder("h264_amf", runner=runner_ok("h264_nvenc")).name == "libx264"
    r = runner_ok("h264_nvenc")
    assert detect_encoder("libx264", runner=r).name == "libx264" and r.tried == []


def test_encoder_args_quality():
    assert encoder_args("libx264", 18, "slow")[-4:] == ["-crf", "18", "-pix_fmt", "yuv420p"]
    assert "-cq" in encoder_args("h264_nvenc", 20) and "21" in encoder_args("h264_nvenc", 20)
    assert Encoder("h264_amf").title == "AMD AMF"


@needs_ffmpeg
def test_detect_real_ffmpeg():
    enc = detect_encoder()
    assert enc.name in ENCODER_ORDER


# ------------------------------------------------------------------ устройство Whisper

def test_pick_device(settings, monkeypatch):
    monkeypatch.setattr(trmod, "cuda_devices", lambda: 0)
    assert pick_device(settings) == ("cpu", "small", "int8")
    monkeypatch.setattr(trmod, "cuda_devices", lambda: 1)
    assert pick_device(settings) == ("cuda", "large-v3", "float16")
    settings.whisper.device = "cpu"
    assert pick_device(settings)[0] == "cpu"


def test_whisper_falls_back_to_cpu(settings, quiet_log, monkeypatch):
    monkeypatch.setattr(trmod, "cuda_devices", lambda: 1)
    tried = []

    def fake_run(self, audio, device, model, compute, duration, log):
        tried.append((device, model))
        if device == "cuda":
            raise RuntimeError("no kernel image for sm_120")
        return Transcript("ru", [])
    monkeypatch.setattr(WhisperTranscriber, "_run", fake_run)
    tr = WhisperTranscriber(settings).transcribe(np.zeros(10, np.float32), 1.0, quiet_log)
    assert tr.language == "ru" and tried == [("cuda", "large-v3"), ("cpu", "small")]


def test_whisper_cpu_error_is_readable(settings, quiet_log, monkeypatch):
    monkeypatch.setattr(trmod, "cuda_devices", lambda: 0)
    monkeypatch.setattr(WhisperTranscriber, "_run", lambda *a: (_ for _ in ()).throw(RuntimeError("bad")))
    with pytest.raises(KonveyerError, match="faster-whisper"):
        WhisperTranscriber(settings).transcribe(np.zeros(1, np.float32), 1.0, quiet_log)


# ------------------------------------------------------------------ yt-dlp

def test_ytdlp_command(settings, tmp_path):
    cmd = ytdlp_command("https://x.com/a/status/1", tmp_path, settings)
    assert cmd[-1] == "https://x.com/a/status/1"
    assert "--write-info-json" in cmd and "--no-playlist" in cmd
    assert cmd[cmd.index("-o") + 1] == "source.%(ext)s" and cmd[cmd.index("-P") + 1] == str(tmp_path)
    assert "--cookies-from-browser" not in cmd
    settings.ingest.cookies_from_browser = "firefox"
    cmd = ytdlp_command("u", tmp_path, settings)
    assert cmd[cmd.index("--cookies-from-browser") + 1] == "firefox"


def test_ytdlp_format():
    f = ytdlp_format(720)
    assert f.startswith("bv*[height<=720][ext=mp4]+ba[ext=m4a]/") and f.endswith("/b")


def test_meta_from_info():
    m = meta_from_info({"title": " Робот ", "channel": "Канал", "description": "о" * 5000,
                        "webpage_url": "https://u", "extractor_key": "Youtube", "tags": list(range(30))})
    assert m["title"] == "Робот" and m["uploader"] == "Канал" and len(m["description"]) == 4000
    assert m["site"] == "Youtube" and len(m["tags"]) == 20
    assert meta_from_info({})["uploader"] == ""


def test_take_local_missing(settings):
    job = Job.create(settings, "нет.mp4")
    with pytest.raises(KonveyerError, match="не найден"):
        take_local(r"C:\нет\такого.mp4", job)


@needs_ffmpeg
def test_ingest_local_cyrillic(settings, quiet_log, wide_video):
    job = Job.create(settings, f'"{wide_video}"')
    ingest(job, settings, quiet_log)
    meta = Job.load(job.dir).data["meta"]
    assert meta["file"] == "source.mp4" and (job.dir / "source.mp4").exists()
    assert meta["width"] == 640 and meta["has_audio"] and meta["title"] == "ролик 16x9"
    assert meta["site"] == "local"


@needs_ffmpeg
def test_transcribe_job_no_audio(settings, quiet_log, silent_video, fake_transcriber):
    job = Job.create(settings, str(silent_video))
    ingest(job, settings, quiet_log)
    tr = transcribe_job(job, settings, quiet_log, fake_transcriber)
    assert tr.empty and fake_transcriber.calls == 0
    saved = read_json(job.transcript_file)
    assert saved["language"] == "" and saved["segments"] == []
    assert saved["reliable"] is False and saved["speech_reason"] == "звука нет"
    meta = Job.load(job.dir).data["meta"]
    assert meta["speech_ok"] is False
    assert meta["speech_note"] == "речи нет — субтитры пропущены"


@needs_ffmpeg
def test_transcribe_job_calls_transcriber(settings, quiet_log, wide_video, fake_transcriber):
    job = Job.create(settings, str(wide_video))
    ingest(job, settings, quiet_log)
    tr = transcribe_job(job, settings, quiet_log, fake_transcriber)
    assert fake_transcriber.calls == 1 and tr.language == "en"
    assert Job.load(job.dir).data["meta"]["language"] == "en"
    assert Transcript.from_dict(read_json(job.transcript_file)) == tr
