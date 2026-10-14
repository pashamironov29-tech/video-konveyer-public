"""По умолчанию Shorts из горизонтального видео без зума: кадр целиком, текст на размытом фоне."""
from __future__ import annotations

from conftest import needs_ffmpeg
from konveyer.config import Settings
from konveyer.layout import choose_layout, geometry, text_layout


def test_default_is_whole_frame_with_text_on_background():
    s = Settings()
    assert s.render.zoom == 1.0
    mode = choose_layout(s.render.layout, 1920, 1080, s.render.width, s.render.height)
    assert mode == "blur"
    geo = geometry(mode, 1920, 1080, s.render.width, s.render.height, s.render.caption_size,
                   zoom=s.render.zoom)
    assert geo.zoom == 1.0 and geo.fg_w == 0                     # без приближения, бока целы
    assert geo.video_h == 608 and geo.video_y == (1920 - geo.video_h) // 2
    assert geo.caption_margin_v >= geo.video_y + geo.video_h     # субтитры ниже картинки
    items = text_layout(geo, "CNN", "Заголовок|в две строки", "Подпись", "a.ttf", "b.ttf")
    for it in items:
        above = it.y + it.size <= geo.video_y
        below = it.y >= geo.video_y + geo.video_h
        assert above or below, it                                 # надписи не на картинке


def test_old_zoom_still_available():
    geo = geometry("blur", 1920, 1080, 1080, 1920, 74, zoom=1.5)
    assert geo.zoom == 1.5 and geo.fg_w > 1080


def test_zoom_center_flag_reaches_settings():
    from konveyer import cli
    from konveyer.pipeline import settings_for_job

    a = cli.build_parser().parse_args(["clip", "u", "0:00-0:30", "--zoom", "1.3", "--zoom-center"])
    opts = cli._job_options(a)
    assert opts["zoom"] == 1.3 and opts["zoom_center"] is True
    a = cli.build_parser().parse_args(["run", "u", "--zoom-center"])
    assert cli._job_options(a)["zoom_center"] is True
    s = Settings()

    class J:
        data = {"options": opts}
    tuned = settings_for_job(s, J())
    assert tuned.render.zoom == 1.3 and tuned.render.zoom_center is True


@needs_ffmpeg
def test_zoom_center_skips_subject_tracking(settings, tmp_path, monkeypatch, wide_video, fast_encoder):
    import shutil
    from konveyer import render as rmod
    from konveyer.jobs import Job
    from konveyer.log import Log
    from konveyer.transcript import Transcript
    from konveyer.util import ffprobe

    def boom(*a, **k):
        raise AssertionError("слежение не должно вызываться")
    monkeypatch.setattr(rmod, "subject_frac_expr", boom)
    settings.render.width, settings.render.height = 320, 568
    settings.render.zoom, settings.render.zoom_center = 1.3, True
    settings.render.cover = settings.render.titles = settings.render.captions = False
    settings.voice.enabled = False
    job = Job.create(settings, str(wide_video))
    shutil.copy(wide_video, job.dir / "source.mp4")
    info = ffprobe(job.dir / "source.mp4")
    job.data["meta"] = {"file": "source.mp4", "width": info["width"], "height": info["height"],
                        "duration": info["duration"], "has_audio": True, "fps": 30}
    job.data["clips"] = [{"id": 1, "start": 0.0, "end": 2.0, "kicker": "", "title": "", "bottom": ""}]
    job.save()
    r = rmod.Renderer(job, settings, Log(job.log_file, echo=False), Transcript("en", []), {}, None,
                      fast_encoder)
    clip = r.render(job.data["clips"][0])
    r.close()
    graph = (job.clip_dir(1) / "filter.txt").read_text(encoding="utf-8")
    assert clip["zoom"] == 1.3 and "crop=320:" in graph
    assert "по центру кадра" in job.log_file.read_text(encoding="utf-8")
