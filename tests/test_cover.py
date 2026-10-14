"""Обложки: шрифт и перенос, выбор кадра, заголовок, оба размера. Сеть замокана."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image, ImageFont

from conftest import ROOT, needs_ffmpeg
from konveyer import cli
from konveyer.config import load_settings
from konveyer.cover import (bundled_font, clean_title, compose_cover, cover_rels,
                            crop_cover, display_title, eye_open_score, fallback_title,
                            fit_title, grab_frames, guess_speaker, invent_prompt,
                            invent_title, laplacian_variance, make_covers, pick_best,
                            pollinations_url, punch, resolve_font, sample_times,
                            save_jpeg, score_frame, shade, text_region, wrap_words,
                            PollinationsProvider)
from konveyer.jobs import Job
from konveyer.log import Log
from konveyer.pipeline import settings_for_job
from konveyer.telegram import Telegram, send_job
from konveyer.util import KonveyerError
from conftest import FAKE_TOKEN, FakeSession, make_video


FONT = bundled_font()


class NoFace:
    available = False

    def boxes(self, frame):
        return []

    def close(self):
        pass


class StubLLM:
    def __init__(self, title="ИИ взломал сам себя и всех остальных",
                 prompt="neon abstract city, no people"):
        self.title = title
        self.prompt = prompt
        self.calls = []

    def chat(self, system, user, json_mode=True, temperature=None):
        self.calls.append(system)
        if "thumbnail" in system:
            return json.dumps({"prompt": self.prompt})
        return json.dumps({"title": self.title})


class Boom:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, width, height):
        self.calls += 1
        raise OSError("тест не ходит в сеть")


class Paint:
    def __init__(self, color=(20, 40, 180)):
        self.color = color
        self.calls = []

    def generate(self, prompt, width, height):
        self.calls.append((prompt, width, height))
        img = Image.new("RGB", (width, height), self.color)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        return buf.getvalue()


def _skin(h, w, seed=0, noise=6):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 176, np.int16)
    img += rng.integers(0, noise, img.shape, dtype=np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def _eyes(img, box, open_eyes=True):
    h, w = img.shape[:2]
    x, y, bw, bh = box
    if not open_eyes:
        return img
    for cx in (x + bw * 0.28, x + bw * 0.72):
        cy = y + bh * 0.36
        rad = max(int(min(bw * w, bh * h) * 0.06), 2)
        px, py = int(cx * w), int(cy * h)
        yy, xx = np.ogrid[:h, :w]
        mask = (yy - py) ** 2 + (xx - px) ** 2 <= rad ** 2
        img[mask] = (12, 12, 12)
    return img


def _blur(img, k=17):
    """Настоящее размытие, не мозаика: у мозаики лапласиан как раз высокий."""
    kernel = np.ones(k, np.float32) / k
    out = np.empty(img.shape, np.float32)
    for c in range(3):
        horiz = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, img[:, :, c])
        out[:, :, c] = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 0, horiz)
    return np.clip(out, 0, 255).astype(np.uint8)


def _job(settings, *, full=False, clips=None, meta=None):
    job = Job.create(settings, "ролик.mp4", {"full": True} if full else {})
    (job.dir / "source.mp4").write_bytes(b"x")
    job.data["meta"] = {"file": "source.mp4", "width": 640, "height": 360,
                        "title": "Ролик", "uploader": "", **(meta or {})}
    job.data["clips"] = clips or [{
        "id": 1, "start": 1.0, "end": 8.0, "title": "Робот учится", "file": "",
        "full": full,
    }]
    job.save()
    return job


def _grab(color=(30, 90, 40)):
    def grabber(video, start, end, count):
        frame = _skin(180, 320, seed=1)
        frame[:] = color
        return [(1.5, frame)]
    return grabber


# ------------------------------------------------------------------ шрифт и перенос

def test_bundled_russo_one_is_in_the_repo():
    assert FONT.is_file()
    assert (FONT.parent / "OFL.txt").is_file()
    assert "Open Font License" in (FONT.parent / "OFL.txt").read_text(encoding="utf-8")
    font = ImageFont.truetype(str(FONT), 48)
    assert font.getlength("ИИ") > font.getlength("I")


def test_resolve_font_prefers_custom_then_bundle(settings, tmp_path):
    assert resolve_font(settings) == FONT
    own = tmp_path / "мой.ttf"
    own.write_bytes(FONT.read_bytes())
    settings.cover.font = str(own)
    assert resolve_font(settings) == own
    settings.cover.font = "нет-такого.ttf"
    log = Log(tmp_path / "font.log", echo=False)
    assert resolve_font(settings, log) == FONT
    assert "не найден" in (tmp_path / "font.log").read_text(encoding="utf-8")


def test_resolve_font_can_download_once(settings, tmp_path, monkeypatch):
    dest = tmp_path / "fonts" / "RussoOne-Regular.ttf"
    monkeypatch.setattr("konveyer.cover.bundled_font", lambda: dest)
    payload = FONT.read_bytes()

    def fetch(url):
        assert "russoone" in url
        return payload

    found = resolve_font(settings, fetch=fetch)
    assert found == dest and dest.read_bytes() == payload


def test_wrap_and_fit_keep_every_word_inside_the_box():
    text = "ИИ ВЗЛОМАЛ САМ СЕБЯ"
    lines, size = fit_title(text, FONT, max_width=280, max_height=220, max_size=90, min_size=20)
    assert " ".join(lines) == text
    assert len(lines) >= 2
    font = ImageFont.truetype(str(FONT), size)
    assert all(font.getlength(line) <= 280.5 for line in lines)
    short_lines, short_size = fit_title("ИИ", FONT, 600, 200, 90, min_size=20)
    assert short_size > size
    # узкая колонка уменьшает кегль, слова не выкидывает
    tight, tight_size = fit_title("НЕЙРОСЕТЬНЕЙРОСЕТЬ", FONT, 120, 80, 80, min_size=12)
    assert " ".join(tight) == "НЕЙРОСЕТЬНЕЙРОСЕТЬ"
    assert tight_size < 80
    assert wrap_words("а б в", lambda s: len(s) * 10, 35) == ["а б", "в"]


def test_compose_is_exact_size_and_under_two_megabytes(tmp_path):
    frame = _skin(720, 1280, seed=2, noise=12)
    img = compose_cover(frame, "ИИ взломал сам себя?", "Илон Маск", FONT, side="left")
    assert img.size == (1280, 720)
    path = tmp_path / "cover.jpg"
    save_jpeg(img, path)
    assert path.stat().st_size < 2 * 1024 * 1024
    assert path.read_bytes()[:2] == b"\xff\xd8"
    back = Image.open(path)
    assert back.size == (1280, 720)
    tall = _skin(1920, 1080, seed=3, noise=4)
    face = (0.3, 0.48, 0.4, 0.4)
    vertical = compose_cover(tall, "Альтман против Маска", "", FONT, face=face, side="top")
    assert vertical.size == (1080, 1920)
    box = text_region(1080, 1920, face, "top")
    face_top = int(face[1] * 1920)
    assert box[1] + box[3] <= face_top + 1


def test_shade_darkens_the_text_side():
    img = np.full((40, 80, 3), 180, np.uint8)
    left = shade(img, "left")
    assert left[:, :20].mean() < left[:, 60:].mean()
    top = shade(img, "top")
    assert top[:10].mean() < top[30:].mean()
    colored = np.zeros((20, 20, 3), np.uint8)
    colored[:, :] = (180, 40, 40)
    richer = punch(colored)
    assert richer[:, :, 0].mean() >= colored[:, :, 0].mean() - 1


# ------------------------------------------------------------------ кадр

def test_pick_sharp_open_eyes_over_blur_closed_and_small():
    big = (0.22, 0.12, 0.42, 0.72)
    sharp = _eyes(_skin(180, 240, seed=4), big, open_eyes=True)
    closed = _eyes(_skin(180, 240, seed=4), big, open_eyes=False)
    blurry = _blur(sharp, 10)
    small_box = (0.72, 0.62, 0.08, 0.12)
    small = _eyes(_skin(180, 240, seed=4), small_box, open_eyes=True)
    assert eye_open_score(sharp, big) > eye_open_score(closed, big)
    assert laplacian_variance(sharp) > laplacian_variance(blurry) * 2
    assert score_frame(sharp, [big]) > score_frame(closed, [big])
    assert score_frame(sharp, [big]) > score_frame(blurry, [big])
    assert score_frame(sharp, [big]) > score_frame(small, [small_box])
    assert score_frame(sharp, [big]) > score_frame(np.zeros_like(sharp), [big])
    cut = (0.0, 0.1, 0.4, 0.5)
    assert score_frame(sharp, [cut]) < score_frame(sharp, [big])
    assert pick_best([blurry, closed, small, sharp], [[big], [big], [small_box], [big]]) == 3


def test_crop_puts_face_aside_and_vertical_face_below_the_title():
    frame = np.zeros((360, 640, 3), np.uint8)
    box = (0.04, 0.18, 0.28, 0.62)
    y0, y1 = int(box[1] * 360), int((box[1] + box[3]) * 360)
    x0, x1 = int(box[0] * 640), int((box[0] + box[2]) * 640)
    frame[y0:y1, x0:x1] = (240, 240, 240)
    wide, face, side = crop_cover(frame, box, 320, 180)
    assert wide.shape == (180, 320, 3)
    assert side == "right"
    bright = wide.mean(axis=2) > 100
    xs = np.where(bright.any(axis=0))[0]
    assert xs.mean() < 320 * 0.48
    region = text_region(320, 180, face, side)
    face_px = (face[0] * 320, face[1] * 180, face[2] * 320, face[3] * 180)
    assert region[0] >= face_px[0] + face_px[2] - 8

    tall, face_t, side_t = crop_cover(frame, box, 180, 320)
    assert tall.shape == (320, 180, 3) and side_t == "top"
    bright_y = np.where((tall.mean(axis=2) > 80).any(axis=1))[0]
    assert bright_y.mean() > 320 * 0.4
    top_box = text_region(180, 320, face_t, "top")
    assert top_box[1] + top_box[3] <= face_t[1] * 320 + 2


def test_sample_times_stay_inside_the_clip():
    times = sample_times(10, 70, 5)
    assert len(times) == 5
    assert times[0] > 10 and times[-1] < 70
    assert times == sorted(times)
    assert sample_times(0, 0.2, 4) == [0.0]


@needs_ffmpeg
def test_grab_frames_reads_real_video(wide_video):
    frames = grab_frames(wide_video, 1, 6, 3, 640, 360, 320)
    assert len(frames) == 3
    assert frames[0][1].ndim == 3 and frames[0][1].shape[1] <= 640


# ------------------------------------------------------------------ заголовок

def test_clean_title_limits_words_and_drops_clickbait():
    assert clean_title("«ИИ взломал сам себя?»", 5) == "ИИ взломал сам себя?"
    cut = clean_title("ИИ взломал сам себя прямо сейчас навсегда?", 5)
    assert len(cut.split()) == 5 and cut.endswith("?")
    assert clean_title("ШОК это нельзя смотреть", 5) == ""
    assert display_title("ии взломал") == "ИИ ВЗЛОМАЛ"
    assert len(fallback_title({}, "", "Робот|учится", 5).split()) == 2


def test_guess_speaker_from_title_or_uploader():
    assert guess_speaker({"title": "Интервью с Илоном Маском: про Марс"}) == "Илоном Маском"
    assert guess_speaker({"title": "Sam Altman: Why AGI"}) == "Sam Altman"
    assert guess_speaker({"title": "Альтман против Маска"}) == ""
    assert guess_speaker({"title": "Про нейросети", "uploader": "Lex Fridman"}) == "Lex Fridman"
    assert guess_speaker({"title": "Про нейросети", "uploader": "AI News"}) == ""
    assert guess_speaker({"title": "про нейросети: ответ"}) == ""


def test_invent_title_uses_llm_and_falls_back(settings, quiet_log):
    llm = StubLLM()
    title = invent_title(llm, settings, {"title": "Ролик"}, "речь про модели", "", quiet_log)
    assert title == "ИИ взломал сам себя и" and len(title.split()) == 5
    assert any("заголовок обложки" in c for c in llm.calls)
    own = invent_title(llm, settings, {}, "", "", quiet_log,
                       override="Свой заголовок тут лишний совсем лишний")
    assert own == "Свой заголовок тут лишний совсем"
    broken = StubLLM(title="ШОК сенсация")
    assert invent_title(broken, settings, {"title": "Робот учится говорить"}, "", "", quiet_log).startswith("Робот")
    prompt = invent_prompt(llm, {"title": "Ролик"}, "речь", title, quiet_log)
    assert "no people" in prompt and "neon" in prompt


def test_prompt_url_retries_and_timeout():
    url = pollinations_url("https://image.pollinations.ai/prompt", "red square?", 1280, 720)
    assert url.startswith("https://image.pollinations.ai/prompt/red%20square%3F?")
    assert "width=1280" in url and "height=720" in url and "nologo=true" in url
    sleeps = []

    class Session:
        def __init__(self):
            self.calls = 0

        def get(self, url, timeout=None):
            assert timeout == 2.5
            self.calls += 1
            if self.calls < 3:
                raise OSError("обрыв")
            return type("R", (), {"status_code": 200, "content": b"x" * 800})()

    provider = PollinationsProvider(timeout=2.5, attempts=3, session=Session(), sleep=sleeps.append)
    assert len(provider.generate("a b", 100, 80)) == 800
    assert sleeps == [1, 2]
    bad = PollinationsProvider(timeout=1, attempts=2, session=type("S", (), {
        "get": lambda *a, **k: (_ for _ in ()).throw(OSError("нет"))})(), sleep=lambda s: None)
    with pytest.raises(KonveyerError, match="не сгенерировалась"):
        bad.generate("x", 10, 10)


# ------------------------------------------------------------------ оба режима и оба размера

def _covers(settings, *, full, mode, provider, llm=None, meta=None, title=None):
    settings.cover.mode = mode
    job = _job(settings, full=full, meta=meta)
    make_covers(job, settings, Log(None, echo=False), llm, title=title,
                grabber=_grab(), provider=provider, finder=NoFace())
    return job


def test_shorts_and_full_both_modes(settings, tmp_path):
    llm = StubLLM()
    paint = Paint()
    job = _covers(settings, full=False, mode="both", provider=paint, llm=llm,
                  meta={"title": "Интервью с Илоном Маском"})
    covers = job.data["clips"][0]["covers"]
    assert covers["speaker"] == "Илоном Маском"
    assert covers["title"].startswith("ИИ взломал")
    frame = job.dir / covers["frame"]
    ai = job.dir / covers["ai"]
    assert frame.name == "clip_01_cover_frame.jpg"
    assert ai.name == "clip_01_cover_ai.jpg"
    assert Image.open(frame).size == (1080, 1920)
    assert Image.open(ai).size == (1080, 1920)
    assert frame.stat().st_size < 2 * 1024 * 1024
    assert paint.calls and paint.calls[0][1:] == (1080, 1920)
    assert abs(np.array(Image.open(ai)).mean() - np.array(Image.open(frame)).mean()) > 5

    settings.cover.mode = "frame"
    bomb = Boom()
    job = _covers(settings, full=True, mode="frame", provider=bomb, llm=llm)
    covers = job.data["clips"][0]["covers"]
    assert covers["ai"] == "" and bomb.calls == 0
    assert Image.open(job.dir / covers["frame"]).size == (1280, 720)
    assert cover_rels({"id": 1}, True)["frame"] == "full/cover_frame.jpg"

    paint = Paint()
    job = _covers(settings, full=True, mode="both", provider=paint, llm=llm)
    covers = job.data["clips"][0]["covers"]
    assert Image.open(job.dir / covers["frame"]).size == (1280, 720)
    assert Image.open(job.dir / covers["ai"]).size == (1280, 720)
    assert paint.calls and paint.calls[0][1:] == (1280, 720)
    assert (job.dir / covers["ai"]).name == "cover_ai.jpg"


def test_ai_failure_falls_back_to_frame(settings, quiet_log):
    bomb = Boom()
    job = _covers(settings, full=True, mode="ai", provider=bomb)
    covers = job.data["clips"][0]["covers"]
    assert covers["fallback"] is True and covers["ai"] == ""
    assert (job.dir / covers["frame"]).is_file()
    assert bomb.calls == 1
    both = _covers(settings, full=False, mode="both", provider=Boom())
    data = both.data["clips"][0]["covers"]
    assert data["frame"] and data["ai"] == "" and data["fallback"] is True


def test_custom_title_skips_the_model(settings):
    llm = StubLLM()
    job = _covers(settings, full=False, mode="frame", provider=Boom(), llm=llm,
                  title="Цифра 5 и модель")
    assert job.data["clips"][0]["covers"]["title"] == "Цифра 5 и модель"
    assert llm.calls == []


def test_cover_flags_and_config(settings, tmp_path):
    parser = cli.build_parser()
    args = parser.parse_args(["run", "u", "--no-cover", "--cover-mode", "ai", "--full"])
    opts = cli._job_options(args, settings)
    assert opts["cover"] is False and opts["cover_mode"] == "ai" and opts["full"] is True
    tuned = settings_for_job(settings, Job.create(settings, "a.mp4", opts))
    assert tuned.cover.enabled is False and tuned.cover.mode == "ai"
    cover = parser.parse_args(["cover", "last", "--title", "Свой текст", "--cover-mode", "frame",
                                "--clip", "2"])
    assert cover.title == "Свой текст" and cover.cover_mode == "frame" and cover.clip == 2
    toml = tmp_path / "k.toml"
    toml.write_text('[cover]\nmode = "frame"\nmax_words = 4\ntext_color = "#fff"\n', encoding="utf-8")
    loaded = load_settings(toml, root=tmp_path)
    assert loaded.cover.mode == "frame" and loaded.cover.max_words == 4
    assert "[cover]" in (ROOT / "konveyer.toml.example").read_text(encoding="utf-8")
    toml.write_text('[cover]\nmode = "nope"\n', encoding="utf-8")
    with pytest.raises(KonveyerError, match="cover.mode"):
        load_settings(toml, root=tmp_path)


def test_disabled_cover_does_nothing(settings, tmp_path):
    settings.cover.enabled = False
    job = _job(settings)
    log = Log(tmp_path / "off.log", echo=False)
    make_covers(job, settings, log, finder=NoFace())
    assert "выключены" in (tmp_path / "off.log").read_text(encoding="utf-8")
    assert "covers" not in job.data["clips"][0]


@needs_ffmpeg
def test_telegram_sends_photos_before_video_and_a_thumb(settings):
    job = _job(settings, full=False)
    clip_dir = job.clip_dir(1)
    clip_dir.mkdir(parents=True)
    make_video(clip_dir / "clip.mp4", 2, "320x180")
    frame = clip_dir / "clip_01_cover_frame.jpg"
    ai = clip_dir / "clip_01_cover_ai.jpg"
    Image.new("RGB", (1080, 1920), (10, 10, 10)).save(frame, "JPEG", quality=70)
    Image.new("RGB", (1080, 1920), (200, 20, 20)).save(ai, "JPEG", quality=70)
    job.data["clips"] = [{
        "id": 1, "title": "Т", "bottom": "", "file": "clips/clip_01/clip.mp4",
        "covers": {"frame": "clips/clip_01/clip_01_cover_frame.jpg",
                   "ai": "clips/clip_01/clip_01_cover_ai.jpg", "title": "ИИ"},
    }]
    job.data["meta"] = {"uploader": "Канал", "width": 320, "height": 180}
    settings.render.width, settings.render.height = 320, 180
    session = FakeSession()
    send_job(job, settings, Log(None, echo=False), tg=Telegram(FAKE_TOKEN, "1", session),
             post=False, check=False)
    methods = [p["url"].rsplit("/", 1)[1] for p in session.posts]
    assert methods == ["sendPhoto", "sendPhoto", "sendVideo"]
    assert "кадр" in session.posts[0]["data"]["caption"]
    assert "нейросеть" in session.posts[1]["data"]["caption"]
    thumb = session.posts[2]["files"]["thumbnail"]
    assert thumb[0].endswith("_tg.jpg") and 0 < thumb[1] <= 200 * 1024
    opened = Image.open(clip_dir / thumb[0])
    assert opened.size[0] <= 320 and opened.size[1] <= 320
