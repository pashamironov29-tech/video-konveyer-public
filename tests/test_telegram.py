"""Telegram: только отправка, токен нигде не светится."""
from __future__ import annotations

import shutil

import pytest

from conftest import FAKE_TOKEN, FakeSession, needs_ffmpeg
from konveyer.jobs import Job
from konveyer.log import Log
from konveyer.telegram import Telegram, clip_caption, make_telegram, send_job, split_message
from konveyer.util import KonveyerError


def test_split_message_short_and_long():
    assert split_message("  привет  ") == ["привет"]
    assert split_message("") == []
    text = "\n".join(f"строка {i} " + "х" * 50 for i in range(200))
    parts = split_message(text, 1000)
    assert all(len(p) <= 1000 for p in parts) and len(parts) > 5
    assert "\n".join(parts).replace("\n", "") == text.replace("\n", "")


def test_split_message_huge_line():
    parts = split_message("я" * 2500, 1000)
    assert [len(p) for p in parts] == [1000, 1000, 500]


def test_needs_env(settings):
    with pytest.raises(KonveyerError, match="TG_BOT_TOKEN"):
        make_telegram(settings, FakeSession())


def test_send_message(settings, monkeypatch):
    monkeypatch.setenv("TG_BOT_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("TG_CHAT_ID", "-100500")
    s = FakeSession()
    tg = make_telegram(settings, s)
    assert tg.send_message("пост " * 2000) == 3
    assert all(p["url"].endswith("/sendMessage") and p["data"]["chat_id"] == "-100500" for p in s.posts)


def test_error_hides_token():
    s = FakeSession(fail_with=f"Unauthorized for bot{FAKE_TOKEN}")
    tg = Telegram(FAKE_TOKEN, "1", s)
    with pytest.raises(KonveyerError) as e:
        tg.send_message("x")
    assert FAKE_TOKEN not in str(e.value) and "***" in str(e.value) and "400" in str(e.value)


def test_network_error_hides_token():
    class Broken:
        def post(self, url, **kw):
            raise ConnectionError(f"cannot reach {url}")
    with pytest.raises(KonveyerError) as e:
        Telegram(FAKE_TOKEN, "1", Broken()).send_message("x")
    assert FAKE_TOKEN not in str(e.value)


def test_clip_caption():
    cap = clip_caption({"id": 2, "title": "Робот|учится", "bottom": "Без|паники"}, {"uploader": "Канал"})
    assert cap == "Клип 2: Робот учится\nБез паники\nАвтор: Канал"
    assert clip_caption({"id": 1}, {}) == "Клип 1:"


@needs_ffmpeg
def test_send_job(settings, wide_video):
    # фикстура 640×360: проверка сравнивает с размером из настроек
    settings.render.width, settings.render.height = 640, 360
    job = Job.create(settings, "a.mp4")
    clip_dir = job.clip_dir(1)
    clip_dir.mkdir(parents=True)
    shutil.copy(wide_video, clip_dir / "clip.mp4")
    job.data["meta"] = {"uploader": "Канал"}
    job.data["clips"] = [{"id": 1, "title": "Т", "bottom": "", "file": "clips/clip_01/clip.mp4"},
                         {"id": 2, "title": "Не готов", "file": ""}]
    job.post_md.write_text("Текст поста", encoding="utf-8")
    settings.telegram.shorts_post = True        # общий пост после Shorts — только по настройке
    s = FakeSession()
    tg = Telegram(FAKE_TOKEN, "1", s)
    sent = send_job(job, settings, Log(None, echo=False), tg=tg)
    assert sent == {"clips": [1], "post": True}
    video = s.posts[0]
    assert video["url"].endswith("/sendVideo")
    assert video["data"]["width"] == 640 and video["data"]["duration"] == 20
    assert video["files"]["video"][0] == "clip.mp4" and video["files"]["video"][1] > 1000
    assert s.posts[1]["data"]["text"] == "Текст поста"
    assert Job.load(job.dir).data["sent"]["clips"] == [1]

    s2 = FakeSession()
    send_job(job, settings, Log(None, echo=False), clips=False, tg=Telegram(FAKE_TOKEN, "1", s2))
    assert s2.posts == []                       # пост уже ушёл — повтор только с resend
    send_job(job, settings, Log(None, echo=False), clips=False, tg=Telegram(FAKE_TOKEN, "1", s2),
             resend=True)
    assert [p["url"].rsplit("/", 1)[1] for p in s2.posts] == ["sendMessage"]


@needs_ffmpeg
def test_send_video_shrinks_big_file(tmp_path, wide_video):
    path = tmp_path / "клип.mp4"
    shutil.copy(wide_video, path)
    size = path.stat().st_size
    s = FakeSession()
    tg = Telegram(FAKE_TOKEN, "1", s, max_mb=size * 0.8 / 2**20)
    tg.send_video(path)
    name, sent_size, _ = s.posts[0]["files"]["video"]
    shrunk = tmp_path / "клип_tg.mp4"
    assert name == "клип_tg.mp4" and shrunk.is_file()
    assert sent_size == shrunk.stat().st_size <= tg.max_bytes < size


@needs_ffmpeg
def test_send_video_refuses_when_shrink_cannot_fit(tmp_path, wide_video):
    path = tmp_path / "клип.mp4"
    shutil.copy(wide_video, path)
    s = FakeSession()
    tg = Telegram(FAKE_TOKEN, "1", s, max_mb=0.2)
    with pytest.raises(KonveyerError, match="не влезает"):
        tg.send_video(path)
    assert s.posts == []


def _tg_env(monkeypatch):
    monkeypatch.setenv("TG_BOT_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("TG_CHAT_ID", "-100500")


def test_send_job_twice_sends_one_post(settings, monkeypatch):
    _tg_env(monkeypatch)
    job = Job.create(settings, "a.mp4")
    job.post_md.write_text("Пост про ИИ", encoding="utf-8")
    session = FakeSession()
    tg = make_telegram(settings, session)
    log = Log(None, echo=False)
    send_job(job, settings, log, tg=tg, check=False)
    send_job(job, settings, log, tg=tg, check=False)
    posts = [p for p in session.posts if p["url"].endswith("/sendMessage")]
    assert len(posts) == 1
    mark = job.data["delivered"]["post"]
    assert mark["sig"] and mark["at"]
    assert not (job.dir / "send.lock").exists()
    send_job(job, settings, log, tg=tg, check=False, resend=True)
    assert len([p for p in session.posts if p["url"].endswith("/sendMessage")]) == 2


def test_send_lock_blocks_second_sender(settings, monkeypatch):
    _tg_env(monkeypatch)
    job = Job.create(settings, "a.mp4")
    job.post_md.write_text("Пост", encoding="utf-8")
    (job.dir / "send.lock").write_text("1", encoding="utf-8")
    session = FakeSession()
    with pytest.raises(KonveyerError, match="уже идёт"):
        send_job(job, settings, Log(None, echo=False), tg=make_telegram(settings, session),
                 check=False)
    assert session.posts == []


def test_clip_caption_source_handle():
    cap = clip_caption({"id": 1, "title": "Робот"}, {"uploader": "Alerta", "uploader_id": "AlertaNews24"})
    assert cap == "Клип 1: Робот\nИсточник: @AlertaNews24"
