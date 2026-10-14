"""util, envkeys, config, textfit, transcript."""
from __future__ import annotations

import sys

import pytest

from conftest import FAKE_KEY, FAKE_TOKEN, needs_ffmpeg
from konveyer import config as cfgmod
from konveyer.config import apply, load_settings
from konveyer.envkeys import get_secret, has_secret, mask
from konveyer.textfit import balance_two, fit_font_size, split_text_lines, wrap_lines
from konveyer.transcript import Segment, Transcript, Word
from konveyer.util import (KonveyerError, ffmpeg_filter_path, ffprobe, fmt_time, is_url,
                           read_json, run, slugify, write_json)


# ------------------------------------------------------------------ envkeys

def test_secret_from_env_only(monkeypatch):
    assert get_secret("TG_BOT_TOKEN") == ""
    assert not has_secret("TG_BOT_TOKEN")
    monkeypatch.setenv("TG_BOT_TOKEN", f"  {FAKE_TOKEN}  ")
    assert get_secret("TG_BOT_TOKEN") == FAKE_TOKEN
    assert has_secret("TG_BOT_TOKEN")


def test_mask_hides_all_secrets(monkeypatch):
    monkeypatch.setenv("TG_BOT_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    text = f"url https://api.telegram.org/bot{FAKE_TOKEN}/send key={FAKE_KEY}"
    out = mask(text)
    assert FAKE_TOKEN not in out and FAKE_KEY not in out
    assert out.count("***") == 2


def test_mask_ignores_short_values(monkeypatch):
    monkeypatch.setenv("TG_CHAT_ID", "12")
    assert mask("цена 12 рублей") == "цена 12 рублей"
    assert mask("") == ""


# ------------------------------------------------------------------ util

@pytest.mark.parametrize("sec, text", [(0, "0:00"), (75.4, "1:15"), (3725, "1:02:05"), (-3, "0:00")])
def test_fmt_time(sec, text):
    assert fmt_time(sec) == text


def test_slugify_translit():
    assert slugify("Привет, Мир! 2026") == "privet-mir-2026"
    assert slugify("щука ёж") == "schuka-ezh"
    assert slugify("!!!") == "video"
    assert len(slugify("а" * 100, 10)) <= 10


def test_ffmpeg_filter_path_escapes_colon():
    assert ffmpeg_filter_path(r"C:\Windows\Fonts\arial.ttf") == r"C\:/Windows/Fonts/arial.ttf"
    assert ffmpeg_filter_path("it's.txt") == r"it\'s.txt"


def test_json_roundtrip_cyrillic(tmp_path):
    p = tmp_path / "папка" / "данные.json"
    write_json(p, {"текст": "привет", "n": [1, 2]})
    assert read_json(p) == {"текст": "привет", "n": [1, 2]}
    assert read_json(tmp_path / "нет.json", {"d": 1}) == {"d": 1}
    assert not list(p.parent.glob("*.tmp"))


def test_is_url():
    assert is_url("https://youtu.be/x") and is_url("HTTP://x.com/a")
    assert not is_url(r"C:\видео\a.mp4") and not is_url("")


def test_run_error_masks_secret(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    code = f"import sys; sys.stderr.write('bad key {FAKE_KEY}'); sys.exit(3)"
    with pytest.raises(KonveyerError) as e:
        run([sys.executable, "-c", code], "проба")
    assert "код выхода 3" in str(e.value)
    assert FAKE_KEY not in str(e.value) and "***" in str(e.value)


def test_run_missing_program():
    with pytest.raises(KonveyerError):
        run(["нет-такой-программы-xyz"], "проба")


@needs_ffmpeg
def test_ffprobe(wide_video, silent_video):
    info = ffprobe(wide_video)
    assert (info["width"], info["height"]) == (640, 360)
    assert 19.5 < info["duration"] < 20.5
    assert info["has_audio"] and info["fps"] == 30
    assert not ffprobe(silent_video)["has_audio"]


# ------------------------------------------------------------------ config

def test_defaults(settings, tmp_path):
    assert settings.render.width == 1080 and settings.render.height == 1920
    assert settings.llm.model == "deepseek-chat"
    assert settings.web.host == "127.0.0.1"
    assert settings.jobs_dir == tmp_path / "jobs"
    assert "root" not in settings.to_dict()


def test_toml_and_overrides(tmp_path):
    (tmp_path / "k.toml").write_text(
        '[render]\nlayout = "full"\ncrf = 18\n[moments]\ncount = 5\n', encoding="utf-8")
    s = load_settings(tmp_path / "k.toml", {"render.tighten": "да", "moments.max_sec": "45"}, root=tmp_path)
    assert s.render.layout == "full" and s.render.crf == 18
    assert s.moments.count == 5 and s.moments.max_sec == 45.0
    assert s.render.tighten is True


def test_config_from_env_var(tmp_path, monkeypatch):
    cfg = tmp_path / "свой.toml"
    cfg.write_text('[web]\nport = 9999\n', encoding="utf-8")
    monkeypatch.setenv("KONVEYER_CONFIG", str(cfg))
    assert load_settings(root=tmp_path).web.port == 9999


@pytest.mark.parametrize("override, msg", [
    ({"нет.ключа": "1"}, "раздела"),
    ({"render.нет": "1"}, "нет ключа"),
    ({"render.crf": "много"}, "целое"),
    ({"render.tighten": "может"}, "да/нет"),
    ({"render.layout": "diagonal"}, "render.layout"),
    ({"moments.min_sec": "100"}, "min_sec"),
    ({"render.width": "1081"}, "чётными"),
])
def test_bad_settings(tmp_path, override, msg):
    with pytest.raises(KonveyerError, match=msg):
        load_settings(tmp_path / "нет.toml", override, root=tmp_path)


def test_bad_toml(tmp_path):
    (tmp_path / "k.toml").write_text("[render\n", encoding="utf-8")
    with pytest.raises(KonveyerError, match="ошибка в k.toml"):
        load_settings(tmp_path / "k.toml", root=tmp_path)


def test_apply_list(settings):
    apply(settings, "post", "style_files", "a.md")
    assert settings.post.style_files == ["a.md"]
    assert "post" in cfgmod.SECTIONS


# ------------------------------------------------------------------ textfit

def test_wrap_and_balance():
    assert wrap_lines("раз два три четыре", 9) == ["раз два", "три", "четыре"]
    assert wrap_lines("раз два три четыре", 9, 2) == ["раз два", "три четыре"]
    assert balance_two("коротко", 18) == ["коротко"]
    a, b = balance_two("нейросеть научилась готовить борщ", 20)
    assert abs(len(a) - len(b)) <= 8
    assert balance_two("", 10) == []
    two = balance_two("Нейросеть научилась печь хлеб", 18)
    assert len(two) == 2 and " ".join(two) == "Нейросеть научилась печь хлеб"


def test_split_never_loses_words():
    text = "очень длинный заголовок который точно не влезает никуда"
    for n in (1, 2, 3):
        lines = split_text_lines(text, n, 12)
        assert len(lines) <= n and " ".join(lines) == text


def test_split_text_lines_pipe():
    assert split_text_lines("первая|вторая|третья", 2, 18) == ["первая", "вторая"]
    assert split_text_lines(r"раз\nдва", 2, 18) == ["раз", "два"]


def test_fit_font_size_limits():
    assert fit_font_size("a", 960, 118) == 118
    assert fit_font_size("x" * 200, 960, 118) == 28
    assert fit_font_size("x" * 20, 960, 118, 0.7) == int(960 / (20 * 0.7))


# ------------------------------------------------------------------ transcript

def test_transcript_roundtrip():
    tr = Transcript("en", [Segment(0, 2, "hello world", [Word(0, 0.5, " hello"), Word(0.6, 1.2, " world")]),
                           Segment(3, 4, "no words")])
    assert len(tr.words) == 3 and tr.words[2].word == " no words"
    assert tr.text() == "hello world no words"
    back = Transcript.from_dict(tr.to_dict())
    assert back == tr
    assert [w.word for w in tr.words_between(0.5, 1.0)] == [" world"]
    assert Transcript.from_dict(None).empty
