"""Shorts по ручным таймкодам: разбор кусков, границы по предложениям, переиспользование."""
from __future__ import annotations

import pytest

from konveyer.jobs import Job
from konveyer.manual import (find_base, is_spans, manual_job, parse_spans, snap_end,
                             snap_start)
from konveyer.pipeline import Pipeline
from konveyer.util import KonveyerError, write_json

from conftest import english_transcript


def test_parse_spans():
    assert parse_spans(["2:10-2:50", "300–340", "0:33.5-1:10"]) == [
        (130.0, 170.0), (300.0, 340.0), (33.5, 70.0)]
    assert is_spans(["2:10-2:50", "5:00-5:40"]) and not is_spans(["2"])
    with pytest.raises(KonveyerError):
        parse_spans(["2:50-2:10"])
    with pytest.raises(KonveyerError):
        parse_spans(["полторы минуты"])


def test_bounds_snap_to_sentence_ends():
    words = english_transcript().words          # точка после каждого 6-го слова
    # 7-е слово (3.2 с) начинает предложение, 6-е кончается на 3.1 с
    assert snap_start(3.9, words) == pytest.approx(3.1)
    # 12-е слово кончает предложение на 6.1 с, следующее — с 6.2 с
    assert snap_end(6.5, words, 20.0) == pytest.approx(6.2)
    # дальше 1,5 с от конца предложения — хотя бы не режем слово
    assert snap_start(8.4, words, window=0.3) == pytest.approx(8.1)


def _base_job(settings, tmp_path):
    base = Job.create(settings, "https://youtu.be/abc", {"full": True, "voice": True})
    (base.dir / "src.mp4").write_bytes(b"video")
    base.data["meta"] = {"file": "src.mp4", "duration": 20.0, "title": "Ролик",
                         "uploader": "CNN", "speaker_cast": {"S2": "ru-RU-DmitryNeural"}}
    write_json(base.transcript_file, english_transcript().to_dict())
    write_json(base.dir / "translations.json", {"word1": "слово1"})
    for step in ("ingest", "transcribe"):
        base.mark(step, "done")
    base.save()
    return base


def test_manual_job_reuses_download_and_transcript(settings, tmp_path, quiet_log, fake_llm):
    base = _base_job(settings, tmp_path)
    assert find_base(settings, "https://youtu.be/abc").id == base.id
    job, found = manual_job(settings, base.id, [(3.9, 6.5)], {"voice": True})
    assert found.id == base.id and job.id != base.id
    assert (job.dir / "src.mp4").read_bytes() == b"video"
    assert job.done("ingest") and job.done("transcribe") and not job.done("analyze")
    assert (job.dir / "translations.json").is_file()
    opts = job.data["options"]
    assert "full" not in opts and opts["manual"] == [[3.9, 6.5]] and opts["voice"] is True
    assert job.data["meta"]["speaker_cast"] == {"S2": "ru-RU-DmitryNeural"}
    assert not base.data["options"].get("manual")      # старое задание не трогаем

    Pipeline(job, settings, quiet_log, llm=fake_llm)._step("moments")
    clips = job.data["clips"]
    assert len(clips) == 1
    assert clips[0]["start"] == pytest.approx(3.1) and clips[0]["end"] == pytest.approx(6.2)
    assert clips[0]["title"]
