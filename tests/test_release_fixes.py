"""Исправления перед публичным релизом: веб, ingest, озвучка, граф ffmpeg, язык CLI."""
from __future__ import annotations

import collections
import http.client
import subprocess
import threading
import time

import numpy as np
import pytest

from conftest import needs_ffmpeg
from konveyer import cli, i18n, util
from konveyer import voice as voicemod
from konveyer.captions import Cue
from konveyer.config import _validate
from konveyer.ingest import ingest, take_local
from konveyer.jobs import Job
from konveyer.log import Log
from konveyer.util import KonveyerError, filter_complex_args, ffmpeg_filter_path
from konveyer.voice import EdgeEngine, VoiceOver
from konveyer.web.server import App, Busy, allowed_host, make_server


class Gate:
    def __init__(self):
        self.gate = threading.Event()
        self.calls = []

    def __call__(self, settings, job, kind, data):
        self.gate.wait(5)
        self.calls.append(kind)
        job.set_state("done")


@pytest.fixture
def server(settings):
    rec = Gate()
    rec.gate.set()
    app = App(settings, runner=rec)
    srv, _ = make_server(settings, app, port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv.server_address[1], app, rec, settings
    rec.gate.set()
    srv.shutdown()
    srv.server_close()
    app.stop()


def raw(port, method, path, headers=None, body=b""):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
    for k, v in (headers or {}).items():
        conn.putheader(k, v)
    conn.endheaders(body or None)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


# ---------- веб: Host, источник, Content-Length, очередь

def test_allowed_host():
    assert allowed_host("127.0.0.1:8765", 8765)
    assert allowed_host("LOCALHOST:8765", 8765)
    assert allowed_host("[::1]:8765", 8765)
    assert not allowed_host("evil.example:8765", 8765)
    assert not allowed_host("127.0.0.1:9999", 8765)
    assert not allowed_host("", 8765) and not allowed_host(None, 8765)


def test_foreign_host_header_is_refused(server):
    port, *_ = server
    assert raw(port, "GET", "/api/status", {"Host": f"127.0.0.1:{port}"})[0] == 200
    assert raw(port, "GET", "/api/status", {"Host": f"localhost:{port}"})[0] == 200
    code, body = raw(port, "GET", "/api/status", {"Host": f"rebind.example:{port}"})
    assert code == 403 and "Host".encode() in body
    assert raw(port, "POST", "/api/jobs", {"Host": f"rebind.example:{port}", "X-Konveyer": "1",
                                           "Content-Length": "2"}, b"{}")[0] == 403


def test_bad_content_length(server):
    port, *_ = server
    h = {"Host": f"127.0.0.1:{port}", "X-Konveyer": "1", "Content-Type": "application/json"}
    assert raw(port, "POST", "/api/jobs", {**h, "Content-Length": "-5"})[0] == 400
    assert raw(port, "POST", "/api/jobs", {**h, "Content-Length": "abc"})[0] == 400


def test_web_source_must_be_video_or_link(server, tmp_path):
    port, app, _, _ = server
    secret = tmp_path / "notes.txt"
    secret.write_text("личное", encoding="utf-8")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00" * 16)
    h = {"Host": f"127.0.0.1:{port}", "X-Konveyer": "1", "Content-Type": "application/json"}

    def post(source):
        import json
        body = json.dumps({"source": source}).encode()
        return raw(port, "POST", "/api/jobs", {**h, "Content-Length": str(len(body))}, body)

    assert post(str(secret))[0] == 400
    assert post(str(tmp_path / "нет.mp4"))[0] == 400
    assert post(str(video))[0] == 201
    app.tasks.join()


def test_status_hides_jobs_dir(server):
    port, *_ = server
    _, body = raw(port, "GET", "/api/status", {"Host": f"127.0.0.1:{port}"})
    assert b"jobs_dir" not in body


def test_try_submit_is_atomic(settings):
    rec = Gate()
    app = App(settings, runner=rec)
    try:
        job = Job.create(settings, "https://youtu.be/x")
        assert app.try_submit(job, "run", {})
        assert not app.try_submit(job, "run", {})
        with pytest.raises(Busy):
            app.submit(job, "run", {})
        assert app.busy(job.id)
    finally:
        rec.gate.set()
        app.tasks.join()
        app.stop()
    assert rec.calls == ["run"] and not app.busy(job.id)


def test_post_text_blocked_only_while_post_step_runs(server):
    port, app, rec, settings = server
    import json
    job = Job.create(settings, "https://youtu.be/x")
    rec.gate.clear()
    app.submit(job, "run", {})
    h = {"Host": f"127.0.0.1:{port}", "X-Konveyer": "1", "Content-Type": "application/json"}
    body = json.dumps({"text": "Мой пост"}).encode()
    h["Content-Length"] = str(len(body))
    job.data["current"] = "render"
    job.save()
    assert raw(port, "POST", f"/api/jobs/{job.id}/post_text", h, body)[0] == 200
    job.data["current"] = "post"
    job.save()
    assert raw(port, "POST", f"/api/jobs/{job.id}/post_text", h, body)[0] == 409
    rec.gate.set()
    app.tasks.join()


def test_web_host_must_be_loopback(settings):
    settings.web.host = "0.0.0.0"
    with pytest.raises(KonveyerError, match="web.host"):
        _validate(settings)
    settings.web.host = "localhost"
    _validate(settings)
    settings.web.port = 70000
    with pytest.raises(KonveyerError, match="web.port"):
        _validate(settings)


# ---------- ingest

def test_take_local_copies_by_default_and_links_on_request(settings, tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"video")
    a = Job.create(settings, str(src))
    dst, _ = take_local(str(src), a)
    dst.write_bytes(b"changed")
    assert src.read_bytes() == b"video"                 # копия: исходник цел
    b = Job.create(settings, str(src))
    dst2, _ = take_local(str(src), b, link=True)
    assert dst2.read_bytes() == b"video"
    try:
        same = dst2.stat().st_ino == src.stat().st_ino and src.stat().st_nlink > 1
    except OSError:
        same = False
    assert same or dst2.is_file()                       # FS без жёстких ссылок — откат на копию


@needs_ffmpeg
def test_non_video_is_removed_after_probe(settings, tmp_path):
    fake = tmp_path / "secret.mp4"
    fake.write_text("это не видео", encoding="utf-8")
    job = Job.create(settings, str(fake))
    with pytest.raises(KonveyerError):
        ingest(job, settings, Log(None, echo=False))
    assert not list(job.dir.glob("source*"))
    assert fake.is_file()


def test_job_create_same_second_gets_own_folder(settings, monkeypatch):
    monkeypatch.setattr("konveyer.jobs.time.strftime", lambda fmt, *a: "20260101-000000")
    a = Job.create(settings, "https://youtu.be/x", label="одно")
    b = Job.create(settings, "https://youtu.be/x", label="одно")
    assert a.dir != b.dir and a.dir.is_dir() and b.dir.is_dir()
    assert b.dir.name.endswith("-2")


# ---------- озвучка

def test_write_wav_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "a.wav"
    voicemod.write_wav(path, np.zeros(480, np.float32), 48000)
    before = path.read_bytes()

    def boom(*a, **k):
        raise OSError("диск занят")

    monkeypatch.setattr(voicemod.os, "replace", boom)
    with pytest.raises(OSError):
        voicemod.write_wav(path, np.ones(960, np.float32) * 0.5, 48000)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.wav"]


def test_prefetch_synthesizes_duplicate_phrase_once(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(voicemod, "to_mix",
                        lambda path, tempo, workdir: np.asarray(voicemod.read_wav(path)[0], np.float32))
    settings.voice.voice_mode = "voiceover"
    settings.voice.voice_check = False
    settings.voice.expression["enabled"] = False
    settings.voice.edge_workers = 4
    asked = collections.Counter()
    lock = threading.Lock()

    def fn(text, voice, rate, pitch, volume):
        with lock:
            asked[text] += 1
        time.sleep(0.1)                                 # потоки должны пересечься
        return np.full(int(48000 * 0.4), 0.1, np.float32)

    log = Log(None, echo=False)
    eng = EdgeEngine(settings.voice, log, tmp_path / "edge", synth_fn=fn, sleeper=lambda s: None)
    texts = ["Повтор фразы.", "Первая другая.", "Повтор фразы.", "Вторая другая.", "Повтор фразы."]
    cues = [Cue(i * 3.0, i * 3.0 + 1.5, t) for i, t in enumerate(texts)]
    VoiceOver(tmp_path, settings, log, None, engine=eng).make(tmp_path, cues, 16.0, 1)
    assert sorted(asked) == sorted(set(texts)) and max(asked.values()) == 1


# ---------- граф ffmpeg

def test_filter_complex_args_short_inline_long_from_file(monkeypatch):
    monkeypatch.setattr(util, "_script_flag", "-/filter_complex")
    assert filter_complex_args("[0:v]null[v]") == ["-filter_complex", "[0:v]null[v]"]
    long = "[0:v]" + ",".join(["null"] * 3000) + "[v]"
    assert filter_complex_args(long) == ["-/filter_complex", "filter.txt"]


@needs_ffmpeg
def test_long_graph_from_file_runs(tmp_path):
    graph = "[0:v]" + ",".join(["null"] * 3000) + "[v]"
    (tmp_path / "filter.txt").write_text(graph, encoding="utf-8")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
           "-i", "color=c=black:s=64x64:d=0.1", *filter_complex_args(graph), "-map", "[v]", "out.mp4"]
    r = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "out.mp4").stat().st_size > 0


@needs_ffmpeg
def test_filter_path_with_apostrophe_and_brackets(tmp_path):
    folder = tmp_path / "O'Brien, [x]; y"
    folder.mkdir()
    ass = folder / "sub's.ass"
    ass.write_text("[Script Info]\nScriptType: v4.00+\nPlayResX: 64\nPlayResY: 64\n\n[V4+ Styles]\n"
                   "Format: Name, Fontname, Fontsize\nStyle: Default,Arial,20\n\n[Events]\n"
                   "Format: Layer, Start, End, Style, Text\n"
                   "Dialogue: 0,0:00:00.00,0:00:01.00,Default,hi\n", encoding="utf-8")
    graph = f"[0:v]ass='{ffmpeg_filter_path(ass)}'[v]"
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                        "-i", "color=c=black:s=64x64:d=0.1", "-filter_complex", graph, "-map", "[v]",
                        str(tmp_path / "out.mp4")], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr


# ---------- язык CLI

def test_ui_lang_from_config_translates_help(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "konveyer.toml"
    cfg.write_text('[ui]\nlang = "en"\n', encoding="utf-8-sig")   # с BOM, как сохраняет PowerShell 5
    monkeypatch.setattr(i18n, "_lang", None)
    monkeypatch.delenv("KONVEYER_LANG", raising=False)
    try:
        with pytest.raises(SystemExit):
            cli.main(["--config", str(cfg), "--help"])
    finally:
        i18n.set_lang("ru")
    out = capsys.readouterr().out
    assert "usage" in out.lower()
    assert "путь к konveyer.toml" not in out
