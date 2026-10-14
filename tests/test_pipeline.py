"""Сквозной прогон: синтетическое видео -> клипы 9:16 -> пост. Настоящий ffmpeg,
поддельные Whisper и DeepSeek."""
from __future__ import annotations

import pytest

from conftest import FakeLLM, FakeTranscriber, english_transcript, needs_ffmpeg
from konveyer.jobs import STEPS, Job
from konveyer.log import Log
from konveyer.pipeline import Pipeline, settings_for_job
from konveyer.transcript import Segment, Transcript, Word
from konveyer.util import KonveyerError, ffprobe, read_json

pytestmark = needs_ffmpeg


@pytest.fixture
def small(settings):
    """Маленький кадр и короткие клипы, чтобы тест шёл секунды."""
    settings.render.width, settings.render.height = 540, 960
    settings.render.caption_size = 40
    settings.moments.whole_if_shorter = 5
    settings.moments.min_sec, settings.moments.max_sec = 5, 8
    return settings


def run_job(settings, source, options=None, llm=None, transcriber=None, encoder=None):
    job = Job.create(settings, str(source), options or {})
    p = Pipeline(job, settings, Log(job.log_file, echo=False), llm=llm,
                 transcriber=transcriber or FakeTranscriber(), encoder=encoder)
    p.run()
    return job, p


def test_full_run_with_ai(small, wide_video, fast_encoder):
    llm = FakeLLM()
    job, p = run_job(small, wide_video, llm=llm, encoder=fast_encoder)
    data = Job.load(job.dir).data
    assert data["state"] == "done" and all(data["steps"][s]["status"] == "done" for s in STEPS)
    assert len(data["clips"]) == 2
    for clip in data["clips"]:
        out = job.dir / clip["file"]
        info = ffprobe(out)
        assert (info["width"], info["height"]) == (540, 960) and info["has_audio"]
        assert info["duration"] == pytest.approx(clip["end"] - clip["start"], abs=0.25)
        assert clip["layout"] == "blur" and clip["encoder"] == "libx264"
        assert clip["captions"] == "перевод на русский"
        assert (job.dir / clip["cover"]).stat().st_size > 1000
        ass = (out.parent / "captions.ass").read_text(encoding="utf-8")
        assert "РУ word" in ass
        assert not (out.parent / "clip.part.mp4").exists()
    t0 = (job.dir / "clips/clip_01/t_0.txt").read_text(encoding="utf-8")
    assert t0 == "Тестовый автор"
    assert read_json(job.dir / "translations.json")
    post = job.post_md.read_text(encoding="utf-8")
    assert "Тестовый пост" in post and "#shorts #россия" in post
    analysis = read_json(job.analysis_file)
    assert analysis["loudness"] and len(analysis["speech"]) == 40


def test_rerun_skips_done_steps(small, wide_video, fast_encoder):
    tr = FakeTranscriber()
    job, _ = run_job(small, wide_video, llm=None, transcriber=tr, encoder=fast_encoder)
    assert tr.calls == 1
    Pipeline(job, small, Log(job.log_file, echo=False), llm=None, transcriber=tr,
             encoder=fast_encoder).run()
    assert tr.calls == 1
    assert "уже готово, пропускаю" in job.log_file.read_text(encoding="utf-8")


def test_no_llm_heuristic_and_template(small, wide_video, fast_encoder):
    job, _ = run_job(small, wide_video, {"llm": False}, llm=None, encoder=fast_encoder)
    data = Job.load(job.dir).data
    assert data["state"] == "done" and data["clips"]
    assert all(c["reason"].startswith("эвристика") for c in data["clips"])
    assert all(c["captions"] == "оригинал (en)" for c in data["clips"])
    assert "Черновик без ИИ" in job.post_md.read_text(encoding="utf-8")


def test_rerender_one_clip_with_new_title(small, wide_video, fast_encoder):
    job, p = run_job(small, wide_video, llm=FakeLLM(), encoder=fast_encoder)
    clip = job.clip(1)
    clip.update(title="Новый|заголовок", start=1.0, end=7.0, layout_req="full")
    job.save()
    p.rerender([1])
    data = Job.load(job.dir).data
    c1 = data["clips"][0]
    assert c1["layout"] == "full"
    assert ffprobe(job.dir / c1["file"])["duration"] == pytest.approx(6.0, abs=0.25)
    texts = [f.read_text(encoding="utf-8") for f in sorted((job.dir / "clips/clip_01").glob("t_*.txt"))]
    assert "Новый" in texts and "заголовок" in texts
    assert data["clips"][1]["layout"] == "blur"


def test_tighten_cuts_pauses(small, wide_video, fast_encoder):
    words = [Word(0.2, 0.6, " раз"), Word(0.7, 1.1, " два"), Word(3.5, 3.9, " три"), Word(4.0, 4.4, " четыре"),
             Word(7.5, 7.9, " пять")]
    tr = FakeTranscriber(Transcript("ru", [Segment(0.2, 7.9, "раз два три четыре пять", words)]))
    small.moments.whole_if_shorter = 60
    job, _ = run_job(small, wide_video, {"tighten": True}, llm=None, transcriber=tr, encoder=fast_encoder)
    clip = Job.load(job.dir).data["clips"][0]
    assert clip["out_duration"] < 20 - 10
    assert ffprobe(job.dir / clip["file"])["duration"] == pytest.approx(clip["out_duration"], abs=0.3)
    assert clip["captions"] == "русские, подсветка слов"


def test_vertical_source_full_layout(small, vertical_video, fast_encoder):
    small.moments.whole_if_shorter = 60
    job, _ = run_job(small, vertical_video, llm=None,
                     transcriber=FakeTranscriber(english_transcript(6, (10, 10))), encoder=fast_encoder)
    clip = Job.load(job.dir).data["clips"][0]
    assert clip["layout"] == "full" and (clip["start"], clip["end"]) == (0.0, pytest.approx(6.0, abs=0.1))


def test_smart_layout_runs(small, wide_video, fast_encoder):
    from konveyer.facetrack import FaceFinder

    finder = FaceFinder()
    try:
        has_detector = finder.available
    finally:
        finder.close()
    small.moments.whole_if_shorter = 60
    job, _ = run_job(small, wide_video, {"layout": "smart", "captions": False}, llm=None,
                     encoder=fast_encoder)
    clip = Job.load(job.dir).data["clips"][0]
    # Нет MediaPipe и каскада (OpenCV 5 без cv2/data) — не падаем, уходим в blur.
    assert clip["layout"] == ("smart" if has_detector else "blur")
    assert clip["captions"] == "выключены"
    if not has_detector:
        assert "раскладка blur" in job.log_file.read_text(encoding="utf-8")
    assert ffprobe(job.dir / clip["file"])["width"] == 540


def test_failure_marks_step(settings):
    job = Job.create(settings, r"C:\нет\такого\файла.mp4")
    with pytest.raises(KonveyerError, match="не найден"):
        Pipeline(job, settings, Log(job.log_file, echo=False), llm=None).run()
    data = Job.load(job.dir).data
    assert data["state"] == "failed" and data["current"] == "ingest"
    assert data["steps"]["ingest"]["status"] == "failed"
    assert "ОШИБКА на шаге «скачивание»" in job.log_file.read_text(encoding="utf-8")


def test_no_titles_skips_overlay_and_title_prompt(small, wide_video, fast_encoder):
    llm = FakeLLM()
    job, _ = run_job(small, wide_video, {"titles": False}, llm=llm, encoder=fast_encoder)
    data = Job.load(job.dir).data
    assert all(c["title"] == "" and c["kicker"] == "" and c["bottom"] == "" for c in data["clips"])
    folder = job.dir / "clips" / "clip_01"
    assert list(folder.glob("t_*.txt")) == []
    assert "drawtext=" not in (folder / "filter.txt").read_text(encoding="utf-8")
    systems = [s for s, _ in llm.calls]
    assert systems and all("kicker" not in s for s in systems)
    assert not any(s.startswith("Ты пишешь тексты на экран") for s in systems)


def test_settings_for_job_options(settings):
    job = Job.create(settings, "a.mp4", {"count": 5, "layout": "full", "llm": False, "device": "cpu",
                                         "tighten": True, "unknown": 1, "max": ""})
    s = settings_for_job(settings, job)
    assert s.moments.count == 5 and s.render.layout == "full" and not s.moments.use_llm
    assert s.whisper.device == "cpu" and s.render.tighten
    assert settings.moments.count == 3 and settings.render.layout == "auto"
