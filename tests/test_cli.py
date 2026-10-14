"""Командная строка: разбор флагов, doctor не печатает секретов, команды по заданиям."""
from __future__ import annotations

import json

import pytest

from conftest import FAKE_KEY, FAKE_TOKEN
from konveyer import cli
from konveyer import hardware
from konveyer.hardware import Encoder
from konveyer.jobs import Job


@pytest.fixture
def jobs_dir(tmp_path):
    return tmp_path / "jobs"


def run_cli(capsys, jobs_dir, *args):
    code = cli.main(["--set", f"paths.jobs_dir={jobs_dir}", *args])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_parser_run_flags():
    a = cli.build_parser().parse_args(["run", "https://youtu.be/x", "--count", "2", "--layout", "smart",
                                       "--tighten", "--no-llm", "--cpu", "--no-translate", "--tg"])
    assert cli._job_options(a) == {"count": 2, "layout": "smart", "tighten": True, "translate": False,
                                   "llm": False, "device": "cpu"}
    assert a.tg


def test_overrides_parse():
    assert cli._overrides(["render.crf=18", "moments.count = 2"]) == {"render.crf": "18", "moments.count": "2"}
    with pytest.raises(cli.KonveyerError):
        cli._overrides(["crf=18"])


def test_doctor_hides_secrets(capsys, jobs_dir, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    monkeypatch.setenv("TG_BOT_TOKEN", FAKE_TOKEN)
    monkeypatch.setattr(hardware, "cuda_devices", lambda: 0)
    monkeypatch.setattr(hardware, "detect_encoder", lambda *a, **k: Encoder("h264_qsv"))
    code, out, err = run_cli(capsys, jobs_dir, "doctor")
    assert FAKE_KEY not in out + err and FAKE_TOKEN not in out + err
    assert "DEEPSEEK_API_KEY     задан" in out and "TG_CHAT_ID           не задан" in out
    assert "Intel Quick Sync" in out and "CUDA для Whisper" in out


def test_config_json(capsys, jobs_dir):
    code, out, _ = run_cli(capsys, jobs_dir, "--set", "render.crf=17", "config")
    data = json.loads(out)
    assert code == 0 and data["render"]["crf"] == 17 and data["paths"]["jobs_dir"] == str(jobs_dir)


def test_bad_setting_exit_code(capsys, jobs_dir):
    code, _, err = run_cli(capsys, jobs_dir, "--set", "render.layout=круг", "config")
    assert code == 1 and "ОШИБКА" in err


def test_jobs_show_empty_and_filled(capsys, jobs_dir, tmp_path):
    code, out, _ = run_cli(capsys, jobs_dir, "jobs")
    assert code == 0 and "Заданий пока нет" in out
    code, _, err = run_cli(capsys, jobs_dir, "show")
    assert code == 1 and "заданий пока нет" in err

    from konveyer.config import load_settings
    s = load_settings(tmp_path / "нет.toml", {"paths.jobs_dir": str(jobs_dir)}, root=tmp_path)
    job = Job.create(s, "https://youtu.be/abc")
    job.data["meta"] = {"title": "Ролик про робота", "width": 1920, "height": 1080, "duration": 125}
    job.data["clips"] = [{"id": 1, "start": 3.0, "end": 20.0, "title": "Робот|учится", "file": ""}]
    job.mark("ingest", "done", sec=4.2)
    job.set_state("done")
    code, out, _ = run_cli(capsys, jobs_dir, "jobs")
    assert job.id in out and "готово" in out and "Ролик про робота" in out
    code, out, _ = run_cli(capsys, jobs_dir, "show", "last")
    assert "1920x1080, 2:05" in out and "клип 1: 3.0–20.0" in out and "(не отрендерен)" in out


def test_clip_rejects_bad_bounds(capsys, jobs_dir, tmp_path):
    from konveyer.config import load_settings
    s = load_settings(tmp_path / "нет.toml", {"paths.jobs_dir": str(jobs_dir)}, root=tmp_path)
    job = Job.create(s, "a.mp4")
    job.data["clips"] = [{"id": 1, "start": 3.0, "end": 20.0}]
    job.save()
    code, _, err = run_cli(capsys, jobs_dir, "clip", job.id, "1", "--start", "30")
    assert code == 1 and "позже начала" in err


def test_send_without_env(capsys, jobs_dir, tmp_path):
    from konveyer.config import load_settings
    s = load_settings(tmp_path / "нет.toml", {"paths.jobs_dir": str(jobs_dir)}, root=tmp_path)
    Job.create(s, "a.mp4")
    code, _, err = run_cli(capsys, jobs_dir, "send")
    assert code == 1 and "TG_BOT_TOKEN" in err


def test_run_missing_file(capsys, jobs_dir):
    code, out, err = run_cli(capsys, jobs_dir, "run", r"C:\нет\файла.mp4", "--no-llm")
    assert code == 1 and "файл не найден" in err
    assert any(jobs_dir.iterdir())
