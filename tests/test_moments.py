"""Выбор моментов: разбор ответа ИИ, эвристика, подгонка границ, шаг целиком."""
from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import FakeLLM, english_transcript
from konveyer.config import MomentsCfg
from konveyer.jobs import Job
from konveyer.moments import (Moment, compact_transcript, default_texts, fit_moment,
                              heuristic_moments, moments_prompt, parse_moments, parse_time,
                              pick_moments, remove_overlaps)
from konveyer.transcript import Segment, Transcript, Word
from konveyer.util import read_json


@pytest.mark.parametrize("value, sec", [
    (12, 12.0), ("12.5", 12.5), ("12,5", 12.5), ("1:23", 83.0), ("1:02:03.5", 3723.5), (" 0:07 ", 7.0),
])
def test_parse_time(value, sec):
    assert parse_time(value) == sec


def test_parse_time_bad():
    with pytest.raises(ValueError):
        parse_time("минута")


def test_parse_moments_filters_garbage():
    data = {"moments": [
        {"start": "0:10", "end": "0:40", "score": 8, "title": "А|Б"},
        {"start": 50, "end": 50.5},                   # слишком короткий
        {"start": "когда-то", "end": 10},              # непонятное время
        "мусор",
        {"start": 90, "end": 200, "score": "много"},   # обрежется по длительности
        {"start": -5, "end": 20, "score": 99},
    ]}
    ms = parse_moments(data, 120.0)
    assert [(m.start, m.end) for m in ms] == [(10, 40), (90, 120), (0, 20)]
    assert ms[0].title == "А|Б" and ms[1].score == 5.0 and ms[2].score == 10.0
    assert parse_moments([{"start": 1, "end": 5}], 10)[0].end == 5
    assert parse_moments("ерунда", 10) == []


def test_compact_transcript_fits_limit():
    segs = [Segment(i * 2.0, i * 2.0 + 1.8, f"фраза номер {i} " * 3) for i in range(300)]
    tr = Transcript("ru", segs)
    full = compact_transcript(tr, 10 ** 9)
    assert full.startswith("[0.0-1.8] фраза номер 0")
    short = compact_transcript(tr, 5000)
    assert len(short) <= 5000 and short.startswith("[0.0-")
    assert compact_transcript(Transcript(), 100) == ""


def test_moments_prompt_has_rules(settings):
    tr = english_transcript(20)
    system, user = moments_prompt(tr, {"title": "Видео", "uploader": "Автор"}, settings.moments,
                                  "моя группа", 2000, 20)
    assert "моя группа" in system and "от 15 до 60 секунд" in system
    assert "Название: Видео" in user and "Автор: Автор" in user and "[0.2-" in user


def _analysis(duration=300.0, peak=(120, 150), hop=0.5):
    n = int(duration / hop)
    loud = np.full(n, -30.0)
    loud[int(peak[0] / hop):int(peak[1] / hop)] = -10.0
    loud[:20] = -70.0
    speech = np.zeros(n)
    speech[int(peak[0] / hop):int(peak[1] / hop)] = 4.0
    return {"hop": hop, "loudness": loud.tolist(), "speech": speech.tolist(),
            "scenes": [130.0, 135.0, 140.0]}


def test_heuristic_finds_loud_talky_part():
    cfg = MomentsCfg(count=2, min_sec=20, max_sec=40)
    ms = heuristic_moments(_analysis(), Transcript(), 300.0, cfg)
    assert len(ms) == 2
    best = max(ms, key=lambda m: m.score)
    assert best.start < 150 and best.end > 120
    assert "громкий пик" in best.reason and "много речи" in best.reason
    assert abs(ms[0].start - ms[1].start) >= 30


def test_heuristic_avoids_outro():
    """Громкие титры в конце длинного ролика проигрывают обычной громкой середине."""
    a = _analysis(duration=600.0, peak=(300, 330))
    loud = np.array(a["loudness"])
    loud[int(570 / 0.5):] = -10.0
    a["loudness"] = loud.tolist()
    a["speech"] = [0.0] * len(loud)
    a["scenes"] += [575.0, 580.0, 585.0, 590.0]
    cfg = MomentsCfg(count=1, min_sec=20, max_sec=40)
    ms = heuristic_moments(a, Transcript(), 600.0, cfg)
    assert ms[0].start < 330 and ms[0].end > 300
    import konveyer.moments as mm
    old, mm.OUTRO_SHARE = mm.OUTRO_SHARE, 0.0
    try:
        assert heuristic_moments(a, Transcript(), 600.0, cfg)[0].end > 570   # без штрафа — титры
    finally:
        mm.OUTRO_SHARE = old


def test_heuristic_short_video_whole():
    ms = heuristic_moments({"hop": 0.5, "loudness": [-20] * 20}, Transcript(), 10.0, MomentsCfg())
    assert len(ms) == 1 and (ms[0].start, ms[0].end) == (0.0, 10.0)


def test_fit_moment_snaps_to_words():
    words = [Word(9.7, 10.4, " Начало"), Word(10.5, 11.0, " фразы"), Word(29.0, 30.2, " конец."),
             Word(30.6, 31.0, " Дальше")]
    cfg = MomentsCfg(min_sec=15, max_sec=60)
    m = fit_moment(Moment(10.0, 30.0), words, [], 100.0, cfg)
    assert m.start == pytest.approx(9.55)
    assert m.end == pytest.approx(30.5)


def test_fit_moment_scenes_and_limits():
    cfg = MomentsCfg(min_sec=15, max_sec=30)
    m = fit_moment(Moment(10.3, 29.6), [], [10.0, 30.0], 100.0, cfg)
    assert m.start == 10.0 and m.end == pytest.approx(29.96)
    assert fit_moment(Moment(0, 90), [], [], 100.0, cfg).end == 30.0
    short = fit_moment(Moment(50, 52), [], [], 100.0, cfg)
    assert short.end - short.start == pytest.approx(15.0)
    assert fit_moment(Moment(3, 5), [], [], 10.0, cfg) == Moment(0.0, 10.0)


def test_remove_overlaps_keeps_best():
    ms = [Moment(0, 30, 5), Moment(10, 40, 9), Moment(60, 90, 7), Moment(85, 115, 6)]
    kept = remove_overlaps(ms)
    assert [(m.start, m.score) for m in kept] == [(10, 9), (60, 7), (85, 6)]


@pytest.mark.parametrize("title, short", [
    ("Big Buck Bunny 60fps 4K - Official Blender Foundation Short Film", "Big Buck Bunny 60fps 4K"),
    ("Нейросеть научилась печь хлеб", "Нейросеть научилась печь хлеб"),
    ("GPT-6: что умеет новая модель и зачем она нужна людям", "GPT-6: что умеет новая модель и"),
    ("ИИ | Новости недели", "ИИ | Новости недели"),
    ("Очень длинное название без всяких разделителей которое не влезает", "Очень длинное название без всяких"),
    ("Смотри https://t.co/x #ии #shorts", "Смотри"),
])
def test_short_title(title, short):
    from konveyer.moments import short_title
    assert short_title(title) == short
    assert len(short_title(title)) <= 36


def test_default_texts():
    m = Moment(0, 10)
    default_texts(m, {"uploader": "Канал", "title": "Нейросеть   научилась печь хлеб https://t.co/x"})
    assert m.kicker == "Канал"
    assert m.title.replace("|", " ") == "Нейросеть научилась печь хлеб"
    keep = Moment(0, 10, title="свой", kicker="мой")
    default_texts(keep, {"uploader": "Канал", "title": "x"})
    assert (keep.kicker, keep.title) == ("мой", "свой")


def _job(settings, duration):
    job = Job.create(settings, "a.mp4")
    job.data["meta"] = {"duration": duration, "title": "Ролик", "uploader": "Автор"}
    job.save()
    return job


def test_pick_moments_short_video_whole(settings, quiet_log):
    job = _job(settings, 40.0)
    ms = pick_moments(job, settings, quiet_log, None, Transcript(), {})
    assert len(ms) == 1 and (ms[0].start, ms[0].end) == (0.0, 40.0)
    assert job.data["clips"][0]["id"] == 1 and job.data["clips"][0]["title"]
    assert read_json(job.moments_file)[0]["end"] == 40.0


def test_pick_moments_with_llm(settings, quiet_log):
    settings.moments.whole_if_shorter = 5
    settings.moments.min_sec, settings.moments.max_sec = 5, 8
    job = _job(settings, 20.0)
    llm = FakeLLM()
    ms = pick_moments(job, settings, quiet_log, llm, english_transcript(), {"scenes": []})
    assert len(ms) == 2
    assert ms[0].title == "Робот|учится" and ms[0].kicker == "Тестовый автор"
    assert ms[1].title == "Заголовок|2"           # дописано вторым запросом
    assert all(5 * 0.8 <= m.duration <= 8.5 for m in ms)
    assert len(llm.calls) == 2


def test_pick_moments_llm_garbage_falls_back(settings, quiet_log):
    settings.moments.whole_if_shorter = 5
    settings.moments.min_sec, settings.moments.max_sec = 5, 8
    job = _job(settings, 20.0)
    analysis = {"hop": 0.5, "loudness": [-20.0] * 40, "speech": [2.0] * 40, "scenes": []}
    ms = pick_moments(job, settings, quiet_log, FakeLLM(broken=True), english_transcript(), analysis)
    assert ms and all(m.reason.startswith("эвристика") for m in ms)
    assert all(m.title == "Ролик" for m in ms)


def test_pick_moments_respects_job_count(settings, quiet_log):
    settings.moments.whole_if_shorter = 5
    settings.moments.min_sec, settings.moments.max_sec = 5, 6
    job = _job(settings, 60.0)
    job.data["options"] = {"count": 1}
    analysis = {"hop": 0.5, "loudness": [-20.0] * 120, "speech": [1.0] * 120, "scenes": []}
    assert len(pick_moments(job, settings, quiet_log, None, Transcript(), analysis)) == 1
    assert json.loads(job.file.read_text(encoding="utf-8"))["clips"][0]["id"] == 1
