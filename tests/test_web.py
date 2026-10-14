"""Веб-интерфейс: API, отдача файлов с Range, защита от выхода за папку и от чужих страниц."""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

from konveyer.jobs import Job
from konveyer.web.server import App, make_server


class Recorder:
    def __init__(self):
        self.calls = []
        self.gate = threading.Event()
        self.gate.set()

    def __call__(self, settings, job, kind, data):
        self.gate.wait(5)
        self.calls.append((job.id, kind, data))
        if kind == "boom":
            raise RuntimeError("упало специально")
        job.set_state("done")


@pytest.fixture
def web(settings):
    rec = Recorder()
    app = App(settings, runner=rec)
    server, _ = make_server(settings, app, port=0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield base, app, rec, settings
    rec.gate.set()
    server.shutdown()
    server.server_close()
    app.stop()


def req(url, body=None, header=True, headers=None):
    h = dict(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        h["Content-Type"] = "application/json"
        if header:
            h["X-Konveyer"] = "1"
    url = urllib.parse.quote(url, safe=":/?=&%")       # как браузер: кириллица в %XX
    r = urllib.request.Request(url, data=data, headers=h, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def jbody(raw):
    return json.loads(raw.decode("utf-8"))


def wait_idle(app):
    app.tasks.join()


def test_index_and_status(web):
    base, app, _, _ = web
    code, raw, headers = req(base + "/")
    assert code == 200 and "Видео-<b>конвейер</b>" in raw.decode("utf-8")
    assert headers["Content-Type"].startswith("text/html")
    code, raw, _ = req(base + "/api/status")
    st = jbody(raw)
    assert code == 200 and st["secrets"] == {
        "DEEPSEEK_API_KEY": False, "TG_BOT_TOKEN": False, "TG_CHAT_ID": False,
        "HF_TOKEN": False, "HUGGING_FACE_HUB_TOKEN": False}


def test_status_never_returns_secret_values(web, monkeypatch):
    base, *_ = web
    monkeypatch.setenv("TG_BOT_TOKEN", "123:секретный-токен")
    code, raw, _ = req(base + "/api/status")
    assert "секретный" not in raw.decode("utf-8") and jbody(raw)["secrets"]["TG_BOT_TOKEN"] is True


def test_post_requires_header(web):
    base, _, rec, _ = web
    code, raw, _ = req(base + "/api/jobs", {"source": "https://youtu.be/x"}, header=False)
    assert code == 403 and rec.calls == []


def test_create_full_job(web):
    base, app, rec, settings = web
    code, raw, _ = req(base + "/api/jobs", {
        "source": "https://youtu.be/full", "full": True, "full_from": "1:00", "full_to": "90",
        "count": 5, "layout": "blur", "voice": True, "zoom": 1.8,
    })
    assert code == 201
    job = Job.load(settings.jobs_dir / jbody(raw)["id"])
    opts = job.data["options"]
    assert opts["full"] is True and opts["full_from"] == 60 and opts["full_to"] == 90
    assert opts["voice"] is True
    assert "count" not in opts and "layout" not in opts and "zoom" not in opts
    code, raw, _ = req(base + "/")
    assert "целое видео" in raw.decode("utf-8")
    assert req(base + "/api/jobs", {"source": "x", "full": True, "full_from": "нет"})[0] == 400
    assert req(base + "/api/jobs", {"source": "x", "full_from": "20", "full_to": "5"})[0] == 400
    code, raw, _ = req(base + "/api/jobs", {"source": "x", "full_from": "1", "full_to": "5"})
    assert code == 201
    opts = Job.load(settings.jobs_dir / jbody(raw)["id"]).data["options"]
    assert opts["full"] is True and opts["full_from"] == 1 and opts["full_to"] == 5
    wait_idle(app)
    assert rec.calls


def test_create_job_queues_run(web):
    base, app, rec, settings = web
    code, raw, _ = req(base + "/api/jobs", {"source": "https://youtu.be/abc", "count": "2", "layout": "smart",
                                            "tighten": True, "no_llm": True, "tg": True, "max": ""})
    assert code == 201
    job_id = jbody(raw)["id"]
    wait_idle(app)
    assert rec.calls == [(job_id, "run", {"tg": True})]
    opts = Job.load(settings.jobs_dir / job_id).data["options"]
    assert opts == {"count": 2, "layout": "smart", "tighten": True, "llm": False}
    code, raw, _ = req(base + "/api/jobs")
    assert [j["id"] for j in jbody(raw)] == [job_id]
    code, raw, _ = req(base + f"/api/jobs/{job_id}")
    d = jbody(raw)
    assert d["job"]["state"] == "done" and d["post"] == "" and "ingest" in d["steps"]


def test_create_job_without_speakers(web):
    base, app, rec, settings = web
    code, raw, _ = req(base + "/api/jobs", {"source": "https://youtu.be/abc",
                                            "no_speakers": True, "one_voice": True})
    assert code == 201
    wait_idle(app)
    opts = Job.load(settings.jobs_dir / jbody(raw)["id"]).data["options"]
    assert opts["speakers"] is False and opts["per_speaker"] is False
    assert rec.calls[-1][1] == "run"


def test_create_job_without_titles(web):
    base, app, rec, settings = web
    code, raw, _ = req(base + "/api/jobs", {"source": "https://youtu.be/abc", "no_titles": True})
    assert code == 201
    job_id = jbody(raw)["id"]
    wait_idle(app)
    opts = Job.load(settings.jobs_dir / job_id).data["options"]
    assert opts["titles"] is False
    assert rec.calls[-1][1] == "run"


def test_create_job_with_voice(web):
    base, app, rec, settings = web
    code, raw, _ = req(base + "/api/jobs", {"source": "https://youtu.be/abc", "voice": True,
                                            "voice_speaker": "baya", "voice_no_subs": True,
                                            "separate": True})
    assert code == 201
    wait_idle(app)
    opts = Job.load(settings.jobs_dir / jbody(raw)["id"]).data["options"]
    assert opts == {"voice": True, "voice_speaker": "baya", "voice_subs": False, "separate": True}

    code, raw, _ = req(base + "/api/jobs", {"source": "https://youtu.be/abd", "voice": False,
                                            "voice_speaker": "aidar", "separate": True})
    wait_idle(app)
    opts = Job.load(settings.jobs_dir / jbody(raw)["id"]).data["options"]
    assert "voice" not in opts and "separate" not in opts

    code, raw, _ = req(base + "/api/jobs", {"source": "https://youtu.be/abe", "voice": True,
                                            "voice_speaker": "чужой"})
    wait_idle(app)
    opts = Job.load(settings.jobs_dir / jbody(raw)["id"]).data["options"]
    assert opts == {"voice": True}


def test_web_page_has_voice_controls(web):
    base, *_ = web
    code, raw, _ = req(base + "/")
    page = raw.decode("utf-8")
    assert "Русская озвучка" in page and "один стиль субтитров" in page
    assert "одним голосом на всех" in page
    options = page.split('id="voice_speaker"', 1)[1].split("</select>", 1)[0]
    order = ["brian", "dmitry", "svetlana", "ava", "emma", "florian", "seraphina"]
    positions = [options.index(f'value="{name}"') for name in order]
    assert positions == sorted(positions)
    assert "andrew" not in options.lower() and "aidar" not in options
    assert 'id="zoom"' in page and 'id="subs_style"' in page
    assert 'value="big"' in page and 'id="score"' in page and 'id="vision_check"' in page
    assert 'id="cover_mode"' in page and "без обложки" in page and "coverStrip" in page


def test_status_defaults_and_zoom_job(web):
    base, app, rec, settings = web
    st = jbody(req(base + "/api/status")[1])
    assert st["defaults"]["zoom"] == 1.0 and st["defaults"]["subs_style"] == "normal"
    code, raw, _ = req(base + "/api/jobs", {"source": "https://youtu.be/zoom", "zoom": 1.7,
                                            "subs_style": "big", "score": True, "vision_check": True})
    assert code == 201
    wait_idle(app)
    opts = Job.load(settings.jobs_dir / jbody(raw)["id"]).data["options"]
    assert opts["zoom"] == 1.7 and opts["subs_style"] == "big"
    assert opts["score"] is True and opts["vision_check"] is True
    assert req(base + "/api/jobs", {"source": "x", "zoom": 3})[0] == 400


def test_clip_zoom_saved(web):
    base, app, rec, settings = web
    job = Job.create(settings, "a.mp4")
    job.data["clips"] = [{"id": 1, "start": 0.0, "end": 8.0, "title": "старый"}]
    job.save()
    code, _, _ = req(base + f"/api/jobs/{job.id}/clips/1", {"zoom": "1.2"})
    assert code == 200
    wait_idle(app)
    clip = Job.load(job.dir).data["clips"][0]
    assert clip["zoom"] == 1.2
    assert req(base + f"/api/jobs/{job.id}/clips/1", {"zoom": 2.5})[0] == 400


def test_create_job_cover_options(web):
    base, app, rec, settings = web
    code, raw, _ = req(base + "/api/jobs", {"source": "https://youtu.be/cover", "cover_mode": "ai",
                                            "no_cover": True})
    assert code == 201
    wait_idle(app)
    opts = Job.load(settings.jobs_dir / jbody(raw)["id"]).data["options"]
    assert opts["cover_mode"] == "ai" and opts["cover"] is False
    code, raw, _ = req(base + "/api/jobs", {"source": "x", "cover_mode": "sideways"})
    assert code == 201
    wait_idle(app)
    opts = Job.load(settings.jobs_dir / jbody(raw)["id"]).data["options"]
    assert "cover_mode" not in opts


def test_job_page_serves_cover_preview(web):
    base, app, _, settings = web
    job = Job.create(settings, "a.mp4")
    folder = job.clip_dir(1)
    folder.mkdir(parents=True)
    from PIL import Image
    Image.new("RGB", (80, 140), (20, 80, 200)).save(folder / "clip_01_cover_frame.jpg", "JPEG")
    job.data["clips"] = [{
        "id": 1, "start": 0.0, "end": 5.0, "title": "кадр", "file": "",
        "covers": {"frame": "clips/clip_01/clip_01_cover_frame.jpg", "ai": "", "title": "ИИ ВЗЛОМАЛ"},
    }]
    job.save()
    code, raw, headers = req(base + f"/files/{job.id}/clips/clip_01/clip_01_cover_frame.jpg")
    assert code == 200 and headers["Content-Type"].startswith("image/jpeg") and raw[:2] == b"\xff\xd8"
    page = req(base + "/")[1].decode("utf-8")
    assert "обложка" in page and "кадр" in page


def test_create_job_validation(web):
    base, *_ = web
    assert req(base + "/api/jobs", {"source": "  "})[0] == 400
    assert req(base + "/api/jobs", {"source": "x", "count": "много"})[0] == 400


def test_clip_edit_and_busy(web):
    base, app, rec, settings = web
    job = Job.create(settings, "a.mp4")
    job.data["clips"] = [{"id": 1, "start": 0.0, "end": 10.0, "title": "старый"}]
    job.save()
    rec.gate.clear()
    code, _, _ = req(base + f"/api/jobs/{job.id}/clips/1",
                     {"title": "Новый|заголовок", "start": "2", "end": "9", "layout": "full"})
    assert code == 200
    code, raw, _ = req(base + f"/api/jobs/{job.id}/clips/1", {"title": "ещё"})
    assert code == 409
    rec.gate.set()
    wait_idle(app)
    clip = Job.load(job.dir).data["clips"][0]
    assert clip["title"] == "Новый|заголовок" and (clip["start"], clip["end"]) == (2.0, 9.0)
    assert clip["layout_req"] == "full"
    assert rec.calls[-1] == (job.id, "rerender", {"clips": [1]})
    assert req(base + f"/api/jobs/{job.id}/clips/1", {"start": 9, "end": 3})[0] == 400
    code, raw, _ = req(base + f"/api/jobs/{job.id}/clips/7", {"title": "x"})
    assert code == 400 and "нет клипа" in jbody(raw)["error"]


def test_post_text_redo_and_send(web, monkeypatch):
    base, app, rec, settings = web
    job = Job.create(settings, "a.mp4")
    assert req(base + f"/api/jobs/{job.id}/post_text", {"text": "Мой пост"})[0] == 200
    assert job.post_md.read_text(encoding="utf-8") == "Мой пост"
    assert req(base + f"/api/jobs/{job.id}/redo", {"from": "нет"})[0] == 400
    assert req(base + f"/api/jobs/{job.id}/redo", {"from": "render"})[0] == 200
    wait_idle(app)
    code, raw, _ = req(base + f"/api/jobs/{job.id}/send", {})
    assert code == 400 and "TG_BOT_TOKEN" in jbody(raw)["error"]
    monkeypatch.setenv("TG_BOT_TOKEN", "123:abc-fake")
    monkeypatch.setenv("TG_CHAT_ID", "42")
    assert req(base + f"/api/jobs/{job.id}/send", {"clips": False, "only": [2]})[0] == 200
    wait_idle(app)
    assert [c[1:] for c in rec.calls] == [("redo", {"from": "render"}),
                                         ("send", {"clips": False, "post": True, "only": [2]})]


def test_worker_error_marks_job(web):
    base, app, rec, settings = web
    job = Job.create(settings, "a.mp4")
    app.submit(job, "boom")
    wait_idle(app)
    data = Job.load(job.dir).data
    assert data["state"] == "failed" and "упало специально" in data["error"]
    assert "RuntimeError" in job.log_file.read_text(encoding="utf-8")
    assert job.id not in app.pending


def test_files_range_and_traversal(web):
    base, _, _, settings = web
    job = Job.create(settings, "a.mp4")
    f = job.clip_dir(1) / "clip.mp4"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"0123456789")
    url = base + f"/files/{job.id}/clips/clip_01/clip.mp4"
    code, raw, h = req(url)
    assert code == 200 and raw == b"0123456789" and h["Content-Type"] == "video/mp4"
    assert h["Accept-Ranges"] == "bytes"
    code, raw, h = req(url, headers={"Range": "bytes=2-5"})
    assert code == 206 and raw == b"2345" and h["Content-Range"] == "bytes 2-5/10"
    code, raw, h = req(url, headers={"Range": "bytes=-3"})
    assert code == 206 and raw == b"789"
    code, raw, h = req(url, headers={"Range": "bytes=50-"})
    assert code == 416
    (settings.jobs_dir / "секрет.txt").write_text("нельзя", encoding="utf-8")
    assert req(base + f"/files/{job.id}/%2e%2e/секрет.txt")[0] == 404
    assert req(base + f"/files/{job.id}/..%5c..%5cсекрет.txt")[0] == 404
    assert req(base + "/files/..%2f/job.json")[0] == 404
    assert req(base + f"/files/{job.id}/нет.mp4")[0] == 404
    assert req(base + "/static/../server.py")[0] == 404


def test_unknown_routes(web):
    base, _, _, settings = web
    assert req(base + "/api/nope")[0] == 404
    assert req(base + "/api/jobs/нет-такого")[0] == 404
    job = Job.create(settings, "a.mp4")
    assert req(base + f"/api/jobs/{job.id}/дичь", {})[0] == 404
