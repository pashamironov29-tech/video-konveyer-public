"""Автоочистка: после успешной отправки — исходники и временные файлы, старые задания — целиком."""
from __future__ import annotations

import time

import pytest

from konveyer import cleanup
from konveyer.jobs import Job
from konveyer.log import Log


class StubTelegram:
    def __init__(self, fail: bool = False):
        self.sent_ids: list[int] = []
        self.fail = fail

    def _ok(self) -> int:
        self.sent_ids.append(len(self.sent_ids) + 100)
        return 1

    def send_video(self, path, caption, log=None, thumb=None):
        if self.fail:
            raise RuntimeError("сеть упала")
        return self._ok()

    def send_message(self, text):
        return self._ok()

    def send_photo(self, path, caption=""):
        return self._ok()


def _write(path, data: bytes = b"x" * 1000):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def make_job(settings, label="t", delivered=True, days_old=0.0):
    job = Job.create(settings, "https://x.com/a/status/1", {}, label)
    d = job.dir
    _write(d / "source.mp4", b"s" * 50_000)
    _write(d / "source.info.json")
    _write(d / "voice_cache" / "a.wav")
    _write(d / "frames" / "f1.jpg")
    _write(d / "transcript.json", b"{}")
    _write(d / "post.md", "пост".encode())
    c = d / "clips" / "clip_01"
    _write(c / "clip.mp4", b"c" * 20_000)
    _write(c / "clip_tg.mp4")
    _write(c / "voice.wav")
    _write(c / "t_0.txt")
    _write(c / "filter.txt")
    _write(c / "description.md", "описание".encode())
    _write(c / "cover.jpg")
    _write(c / "captions.ass")
    _write(c / "frames" / "x.png")
    job.data["clips"] = [{"id": 1, "file": "clips/clip_01/clip.mp4"}]
    if delivered:
        job.data["delivered"] = {"clip:1": {"message_ids": [1]}}
        job.data["sent"] = {"clips": [1], "post": False}
    if days_old:
        job.data["created"] = time.strftime("%Y-%m-%d %H:%M:%S",
                                            time.localtime(time.time() - days_old * 86400))
    job.save()
    return job


KEPT = ["job.json", "post.md", "transcript.json", "clips/clip_01/clip.mp4",
        "clips/clip_01/description.md", "clips/clip_01/cover.jpg", "clips/clip_01/captions.ass"]
GONE = ["source.mp4", "source.info.json", "voice_cache", "frames", "clips/clip_01/clip_tg.mp4",
        "clips/clip_01/voice.wav", "clips/clip_01/t_0.txt", "clips/clip_01/filter.txt",
        "clips/clip_01/frames"]


def test_defaults(settings):
    assert settings.cleanup.enabled is True
    assert settings.cleanup.keep_days == 3


def test_plan_lists_sources_and_temp_only(settings):
    job = make_job(settings)
    plan = cleanup.plan_job(job.dir, job.data)
    names = {p.relative_to(job.dir).as_posix() for p in plan.paths}
    assert names == set(GONE)
    assert plan.size >= 50_000


def test_dry_run_deletes_nothing(settings):
    job = make_job(settings)
    plan = cleanup.clean_job(job, None, dry_run=True)
    assert plan.paths
    for name in GONE + KEPT:
        assert (job.dir / name).exists(), name


def test_clean_keeps_results(settings, quiet_log):
    job = make_job(settings)
    cleanup.clean_job(job, quiet_log)
    for name in KEPT:
        assert (job.dir / name).exists(), name
    for name in GONE:
        assert not (job.dir / name).exists(), name
    assert Job.load(job.dir).data["cleanup"]["removed"] == len(GONE)


def test_referenced_clip_file_is_protected(settings, quiet_log):
    job = make_job(settings)
    _write(job.dir / "best.mp4")
    job.data["clips"][0]["file"] = "best.mp4"
    job.save()
    cleanup.clean_job(job, quiet_log)
    assert (job.dir / "best.mp4").exists()


def test_nothing_deleted_when_not_delivered(settings, quiet_log):
    job = make_job(settings, delivered=False)
    plan = cleanup.clean_job(job, quiet_log)
    assert plan.paths == []
    assert (job.dir / "source.mp4").exists()


def test_send_job_cleans_after_success(settings, monkeypatch):
    from konveyer.telegram import send_job

    monkeypatch.delenv("KONVEYER_CLEANUP", raising=False)
    job = make_job(settings, delivered=False)
    send_job(job, settings, Log(None, echo=False), post=False, tg=StubTelegram(), check=False)
    assert "clip:1" in Job.load(job.dir).data["delivered"]
    assert not (job.dir / "source.mp4").exists()
    assert (job.dir / "clips" / "clip_01" / "clip.mp4").exists()


def test_send_job_failure_keeps_everything(settings, monkeypatch):
    from konveyer.telegram import send_job

    monkeypatch.delenv("KONVEYER_CLEANUP", raising=False)
    job = make_job(settings, delivered=False)
    with pytest.raises(RuntimeError):
        send_job(job, settings, Log(None, echo=False), post=False, tg=StubTelegram(fail=True),
                 check=False)
    for name in GONE + KEPT:
        assert (job.dir / name).exists(), name


def test_disabled_in_config(settings, monkeypatch):
    from konveyer.telegram import send_job

    monkeypatch.delenv("KONVEYER_CLEANUP", raising=False)
    settings.cleanup.enabled = False
    job = make_job(settings, delivered=False)
    send_job(job, settings, Log(None, echo=False), post=False, tg=StubTelegram(), check=False)
    assert (job.dir / "source.mp4").exists()


def test_old_jobs_removed_only_if_sent(settings, quiet_log, monkeypatch):
    monkeypatch.delenv("KONVEYER_CLEANUP", raising=False)
    old_sent = make_job(settings, "old-sent", days_old=4)
    old_unsent = make_job(settings, "old-unsent", delivered=False, days_old=5)
    fresh = make_job(settings, "fresh", days_old=1)
    dry = cleanup.remove_old(settings, quiet_log, dry_run=True)
    assert [o.dir for o in dry] == [old_sent.dir]
    assert old_sent.dir.exists()
    cleanup.auto_cleanup(settings, quiet_log)
    assert not old_sent.dir.exists()
    assert old_unsent.dir.exists() and fresh.dir.exists()


def test_env_switch_disables_auto(settings, quiet_log, monkeypatch):
    monkeypatch.setenv("KONVEYER_CLEANUP", "0")
    job = make_job(settings, days_old=10)
    assert cleanup.auto_cleanup(settings, quiet_log) == []
    assert job.dir.exists()


def test_report_dry_run_lists_and_sums(settings):
    job = make_job(settings)
    old = make_job(settings, "old", days_old=4)
    text = cleanup.report(settings, dry_run=True)
    assert job.dir.name in text and old.dir.name in text
    assert "Итого будет удалено" in text
    assert (job.dir / "source.mp4").exists() and old.dir.exists()


def test_cli_cleanup_dry_run(settings, monkeypatch, capsys):
    from konveyer import cli

    job = make_job(settings)
    monkeypatch.setattr(cli, "load_settings", lambda *a, **k: settings)
    assert cli.main(["cleanup", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert job.dir.name in out
    assert (job.dir / "source.mp4").exists()
