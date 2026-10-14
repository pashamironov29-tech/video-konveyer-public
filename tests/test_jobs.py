"""Задания: папки, состояние, поиск."""
from __future__ import annotations

import pytest

from konveyer.jobs import STEPS, Job, find_job, label_from_source, list_jobs
from konveyer.util import KonveyerError, read_json


@pytest.mark.parametrize("src, label", [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10", "youtube-dQw4w9WgXcQ"),
    ("https://youtu.be/abc123", "youtube-abc123"),
    ("https://x.com/user/status/1790000000000", "x-1790000000000"),
    ("https://twitter.com/user/status/42", "x-42"),
    ("https://www.instagram.com/reel/C9xyz/", "insta-C9xyz"),
    ("https://m.vk.com/video-1_2", "vk-video-1_2"),
    ("https://example.org/", "example"),
    (r"C:\видео\Мой ролик.mp4", "Мой ролик"),
])
def test_label_from_source(src, label):
    assert label_from_source(src) == label


def test_create_and_load(settings):
    job = Job.create(settings, "https://youtu.be/abc", {"count": 2})
    assert job.dir.parent == settings.jobs_dir
    assert job.id.endswith("_youtube-abc")
    data = read_json(job.file)
    assert data["source"] == "https://youtu.be/abc" and data["options"] == {"count": 2}
    assert data["state"] == "new" and data["clips"] == []
    again = Job.load(job.dir)
    assert again.data == job.data


def test_create_same_second_gets_suffix(settings):
    a = Job.create(settings, "https://youtu.be/same")
    b = Job.create(settings, "https://youtu.be/same")
    assert a.dir != b.dir and b.dir.exists()


def test_label_is_translit(settings):
    job = Job.create(settings, r"C:\x\Котики.mp4")
    assert job.id.endswith("_kotiki")


def test_steps_and_invalidate(settings):
    job = Job.create(settings, "a.mp4")
    for s in STEPS:
        job.mark(s, "done", sec=1)
    assert all(job.done(s) for s in STEPS)
    job.invalidate_from("moments")
    assert job.done("analyze") and not job.done("moments") and not job.done("post")
    assert not Job.load(job.dir).done("render")


def test_state_and_summary(settings):
    job = Job.create(settings, "a.mp4")
    job.data["meta"] = {"title": "Название", "uploader": "Автор"}
    job.data["clips"] = [{"id": 1}, {"id": 2}]
    job.set_state("failed", "render", "упало")
    s = Job.load(job.dir).summary()
    assert s["state"] == "failed" and s["current"] == "render" and s["error"] == "упало"
    assert s["title"] == "Название" and s["clips"] == 2
    assert job.clip(2) == {"id": 2}
    with pytest.raises(KonveyerError, match="нет клипа"):
        job.clip(5)


def test_paths(settings):
    job = Job.create(settings, "a.mp4")
    assert job.source_video is None
    job.data["meta"] = {"file": "source.mp4"}
    assert job.source_video == job.dir / "source.mp4"
    assert job.clip_dir(3) == job.dir / "clips" / "clip_03"
    assert job.post_md.name == "post.md" and job.log_file.name == "job.log"


def test_find_job(settings):
    with pytest.raises(KonveyerError, match="заданий пока нет"):
        find_job(settings, "last")
    a = Job.create(settings, "https://youtu.be/first", label="первый")
    b = Job.create(settings, "https://youtu.be/second", label="второй")
    assert {j.id for j in list_jobs(settings)} == {a.id, b.id}
    assert find_job(settings, a.id).id == a.id
    assert find_job(settings, "pervyy").id == a.id
    assert find_job(settings, str(b.dir)).id == b.id
    assert find_job(settings, "last").id == max(a.id, b.id)
    with pytest.raises(KonveyerError, match="не найдено"):
        find_job(settings, "нет-такого")
    with pytest.raises(KonveyerError, match="несколько"):
        find_job(settings, "_")


def test_list_skips_broken(settings):
    Job.create(settings, "a.mp4")
    (settings.jobs_dir / "мусор").mkdir()
    (settings.jobs_dir / "битый").mkdir()
    (settings.jobs_dir / "битый" / "job.json").write_text("", encoding="utf-8")
    assert len(list_jobs(settings)) == 1
