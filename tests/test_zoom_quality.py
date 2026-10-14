"""Зум, крупные субтитры, проверка клипа и текст для озвучки.

Тяжёлые модели и Ollama не запускаются: DeepSeek — поддельный клиент,
Ollama — подмена requests.
"""
from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import needs_ffmpeg
from konveyer import cli
from konveyer.captions import Cue, Word, big_anchor, build_ass, make_big_cues, spread_phrase
from konveyer.config import load_settings
from konveyer.facetrack import fraction_track, smooth_frac_expr, subject_x
from konveyer.jobs import Job
from konveyer.layout import build_graph, even, geometry, text_layout
from konveyer.llm import DeepSeek
from konveyer.log import Log
from konveyer.mediaq import (OLLAMA_DOWN, audio_issues, face_cut, frame_issues, is_black,
                             ollama_status, parse_silence, parse_volume, reset_ollama_warning,
                             score_issues, sharpness, subtitle_issues, tech_issues, vision_issues)
from konveyer.quality import assess_publish
from konveyer.saytext import prepare_for_speech
from konveyer.usage import UsageBook
from konveyer.util import KonveyerError, ffprobe
from konveyer.voice import VoiceOver

FONT = "C:/Windows/Fonts/arial.ttf"


def _w(a, b, text):
    return Word(a, b, text)


# ------------------------------------------------------------------ зум

def test_geometry_zoom_one_matches_old():
    g = geometry("blur", 1920, 1080, zoom=1.0)
    assert (g.video_h, g.video_y, g.fg_w, g.zoom) == (608, 656, 0, 1.0)
    assert g.caption_margin_v == 1284
    smart = geometry("smart", 1920, 1080)
    assert smart.crop_w == 608 and smart.zoom == 1.0


def test_geometry_zoom_one_and_half():
    g = geometry("blur", 1920, 1080, zoom=1.5)
    assert g.video_h == even(608 * 1.5) == 912
    assert g.video_y == even((1920 - 912) / 2) == 504
    assert g.fg_w == even(1080 * 1.5) == 1620
    assert g.fg_w > g.W and g.video_h > 608
    items = text_layout(g, "Автор канала", "Робот|учится", "Без паники", FONT, FONT)
    assert items[0].text == "Автор канала"
    assert all(4 <= it.y and it.y + it.size <= g.H - 4 for it in items)
    smart = geometry("smart", 1920, 1080, zoom=1.5)
    assert smart.crop_w == even(608 / 1.5) < 608


def test_zoom_graph_crops_sides_and_plain_graph_stays():
    plain = geometry("blur", 1920, 1080)
    graph, _ = build_graph(plain, fps=30, src_h=1080, text_filters=[], ass_file=None,
                           keep_expr=None, crop_x_expr=None, has_audio=False)
    assert "overlay=0:656" in graph and "scale=1080:608" in graph
    assert "crop=1080:608" not in graph

    g = geometry("blur", 1920, 1080, zoom=1.5)
    frac = smooth_frac_expr([(0.0, 0.25), (1.2, 0.7)])
    graph, has_a = build_graph(g, fps=30, src_h=1080, text_filters=[], ass_file=None,
                               keep_expr=None, crop_x_expr=None, has_audio=True, zoom_frac=frac)
    assert has_a and f"scale={g.fg_w}:{g.video_h}" in graph
    assert f"crop={g.W}:{g.video_h}:" in graph and f"overlay=0:{g.video_y}" in graph
    assert "if(lt(t," in graph and graph.count("'") >= 2

    full = geometry("full", 1920, 1080, zoom=1.5)
    graph, _ = build_graph(full, fps=30, src_h=1080, text_filters=[], ass_file=None,
                           keep_expr=None, crop_x_expr=None, has_audio=False, zoom_frac="0.4")
    assert "scale=1620:2880" in graph and "crop=1080:1920:" in graph


def test_fraction_track_smooth_and_subject():
    track = fraction_track([(0.0, 0.2), (0.4, 0.21), (2.2, 0.8)], [1.5], 3.0)
    assert track[0] == (0.0, 0.205) or abs(track[0][1] - 0.205) < 0.02
    assert track[-1][1] > 0.7
    expr = smooth_frac_expr([(0.0, 0.2), (1.0, 0.8)])
    assert expr.startswith("if(lt(t,") and "0.2000" in expr and "0.8000" in expr
    same = fraction_track([(0.0, 0.50), (1.0, 0.51)], [], 2.0)
    assert len(same) == 1

    base = np.full((48, 80, 3), 30, np.uint8)
    base[:, 8:18] = 180
    cx, prev, kind = subject_x(base, None)
    assert kind == "object" and cx < 0.4
    moved = base.copy()
    moved[:, 60:74] = 230
    cx2, _, kind2 = subject_x(moved, prev)
    assert kind2 == "motion" and cx2 > 0.55


@needs_ffmpeg
def test_zoom_filter_encodes(tmp_path, wide_video):
    g = geometry("blur", 640, 360, 540, 960, zoom=1.5)
    frac = smooth_frac_expr([(0.0, 0.3), (0.6, 0.72)])
    graph, _ = build_graph(g, fps=30, src_h=360, text_filters=[], ass_file=None,
                           keep_expr=None, crop_x_expr=None, has_audio=True, zoom_frac=frac)
    out = tmp_path / "зум.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
                    "-t", "1", "-i", str(wide_video), "-filter_complex", graph,
                    "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", out.name], check=True, cwd=tmp_path)
    info = ffprobe(out)
    assert (info["width"], info["height"]) == (540, 960) and info["has_audio"]
    assert info["duration"] == pytest.approx(1.0, abs=0.2)


# ------------------------------------------------------------------ крупные субтитры

def test_big_cues_one_or_two_words_on_whisper_times():
    words = [_w(0.10, 0.40, " сотни"), _w(0.42, 0.80, "мест"),
             _w(1.40, 1.70, "оченьдлинноеслово"), _w(1.72, 2.00, "ещё"),
             _w(2.50, 2.80, "конец."), _w(2.82, 3.10, "точка")]
    cues = make_big_cues(words)
    assert [c.text for c in cues] == ["сотни мест", "оченьдлинноеслово", "ещё", "конец.", "точка"]
    assert (cues[0].start, cues[0].end) == (0.10, 0.80)
    assert cues[0].end < cues[1].start
    spread = spread_phrase("сотни свободных мест тут", 1.0, 3.0)
    assert spread[0].text == "сотни свободных" and spread[-1].end == 3.0
    assert spread[0].start == 1.0 and len(spread) >= 2


def test_big_ass_is_bold_white_lower_half_and_normal_stays():
    cues = make_big_cues([_w(0.2, 0.6, "сотни"), _w(0.62, 1.0, "мест")])
    x, y = big_anchor(1080, 1920, video_y=504, video_h=912)
    assert y >= 960 and y < 504 + 912
    ass = build_ass(cues, 1080, 1920, style="big", anchor=(x, y))
    assert ",-1,0,0,0," in ass and "&H00FFFFFF" in ass and "&H00000000" in ass
    assert r"\an5\pos(" in ass and "СОТНИ МЕСТ" in ass
    assert "Dialogue:" in ass
    normal = build_ass(cues, 1080, 1920)
    assert r"\an5\pos" not in normal and "сотни" in normal and "мест" in normal
    assert subtitle_issues(ass) == []
    assert subtitle_issues(normal) == []


def test_subtitle_overflow_and_face():
    ass = ("[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n\n"
           "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
           "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
           "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
           "Style: Cap,Arial,74,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,"
           "1,6,2,2,60,60,400,1\n\n[Events]\n"
           "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
           r"Dialogue: 0,0:00:00.00,0:00:01.00,Cap,,0,0,0,,{\an5\pos(2000,100)\fs200}СЛОВО" "\n")
    assert subtitle_issues(ass) == ["субтитры: вылезают за край"]
    low = build_ass([Cue(0.1, 1.0, "низ")], 1080, 1920, align=2, margin_v=200, karaoke=False)
    # маленькое лицо прямо на нижней строке: пересечение больше трети лица
    face = [(0.45, 0.85, 0.10, 0.05)]
    assert "субтитры: перекрывают лицо" in subtitle_issues(low, face)


# ------------------------------------------------------------------ проверка клипа

def test_frames_black_blur_and_cut_face():
    black = np.zeros((40, 40, 3), np.uint8)
    sharp = np.zeros((40, 40, 3), np.uint8)
    sharp[:, ::2] = 255
    flat = np.full((40, 40, 3), 40, np.uint8)
    assert is_black(black) and not is_black(sharp)
    assert sharpness(sharp) > 1 and sharpness(flat) < 1
    assert "кадры: чёрный экран" in frame_issues([black, black, black, sharp])[0]
    assert "кадры: сильное размытие" in frame_issues([flat, flat, flat])[0]
    assert face_cut((0.0, 0.2, 0.3, 0.3))
    assert not face_cut((0.2, 0.2, 0.3, 0.3))
    msg, _ = frame_issues([sharp, sharp], [[(0.0, 0.1, 0.2, 0.3)], []])
    assert msg == ["кадры: лицо обрезано краем"]


def test_audio_and_tech_messages():
    vol = parse_volume("mean_volume: -40.2 dB\nmax_volume: -0.1 dB\n")
    assert vol == {"mean_db": -40.2, "max_db": -0.1}
    sil = parse_silence("silence_start: 0\nsilence_end: 1.4\nsilence_start: 8.2\n")
    assert sil[0] == (0.0, 1.4) and sil[1][1] is None
    stats = {"mean_db": -40.0, "max_db": -0.1, "silence": sil, "audio_duration": 12.0}
    found = audio_issues(stats, 10.0)
    assert "звук: слишком тихо" in found and "звук: клиппинг" in found
    assert "звук: длинная тишина в начале" in found and "звук: длинная тишина в конце" in found
    assert "звук: длина не совпадает с видео" in found
    assert audio_issues({}, 5, has_audio=False) == ["звук: нет дорожки"]
    ok = tech_issues({"width": 1080, "height": 1920, "bit_rate": 2_000_000, "duration": 40},
                     width=1080, height=1920, shorts_max=60, shorts_pack=True, readable=True)
    assert ok == []
    bad = tech_issues({"width": 640, "height": 360, "bit_rate": 100_000, "duration": 70},
                      width=1080, height=1920, shorts_max=60, shorts_pack=True, readable=True)
    assert any(x.startswith("техника: разрешение") for x in bad)
    assert "техника: низкий битрейт" in bad and "техника: длиннее 60 с для Shorts" in bad
    assert tech_issues({}, width=1080, height=1920, shorts_max=60, shorts_pack=True,
                       readable=False) == ["техника: файл не читается целиком"]


def test_score_off_and_on_books_tokens(settings):
    job = Job.create(settings, "a.mp4")
    job.data["clips"] = [{"id": 2, "title": "Робот учится говорить", "kicker": "Автор",
                          "bottom": "", "reason": "про роботов", "file": ""}]
    job.save()
    log = Log(job.log_file, echo=False)

    class Chatty:
        def __init__(self):
            self.calls = 0

        def chat(self, *a, **k):
            self.calls += 1
            return "{}"

    quiet = Chatty()
    assert score_issues(job, settings, log, quiet) == [] and quiet.calls == 0

    settings.quality.score = True
    settings.quality.min_score = 6
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"score": 3, "reason": "скучно"}'))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4, prompt_cache_hit_tokens=1),
    ))))
    book = UsageBook(job, settings, log)
    llm = DeepSeek(settings.llm, "sk-test", client=client, sleep=lambda s: None, book=book)
    found = score_issues(job, settings, log, llm)
    assert any("клип 2: оценка 3 из 10, ниже порога 6" in x and "скучно" in x for x in found)
    book.flush(final=True)
    data = Job.load(job.dir).data["deepseek"]
    assert data["stages"]["score"]["prompt_tokens"] == 11
    assert "DeepSeek [оценка]:" in job.log_file.read_text(encoding="utf-8")


def test_vision_down_is_one_line_and_bad_frame(settings, monkeypatch):
    import requests

    monkeypatch.setattr("konveyer.mediaq._b64_jpeg", lambda frame: "aGVsbG8=")
    reset_ollama_warning()
    settings.quality.vision = True
    log = Log(None, echo=False)
    lines = []
    log.info = lines.append
    frame = np.zeros((8, 8, 3), np.uint8)

    def down(url, payload):
        raise requests.ConnectionError("нет")

    assert vision_issues([frame], settings, log, poster=down) == []
    assert vision_issues([frame], settings, log, poster=down) == []
    assert lines == [OLLAMA_DOWN]

    class Ok:
        status_code = 200

        def json(self):
            return {"message": {"content": '{"ok": false, "reason": "лицо обрезано"}'}}

    reset_ollama_warning()
    assert vision_issues([frame], settings, log, poster=lambda u, p: Ok()) == ["кадры ИИ: лицо обрезано"]


def test_ollama_status(settings, monkeypatch):
    import requests

    def down(*a, **k):
        raise requests.ConnectionError("нет")

    monkeypatch.setattr(requests, "get", down)
    ok, note = ollama_status(settings)
    assert ok is False and "не запущена" in note

    class Resp:
        status_code = 200

        def json(self):
            return {"models": [{"name": "llava:latest"}]}

    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp())
    ok, note = ollama_status(settings)
    assert ok and "qwen2.5vl" in note and "llava" in note


@needs_ffmpeg
def test_assess_blocks_wrong_frame(settings, wide_video):
    job = Job.create(settings, "a.mp4")
    folder = job.clip_dir(1)
    folder.mkdir(parents=True)
    shutil.copy(wide_video, folder / "clip.mp4")
    job.data["meta"] = {"title": "Ролик", "uploader": "Автор", "description": "Коротко."}
    job.data["clips"] = [{"id": 1, "title": "Новости дня", "kicker": "Автор", "bottom": "Спокойно",
                          "file": "clips/clip_01/clip.mp4"}]
    job.post_md.write_text("Автор выложил короткий ролик без выдуманного сюжета.", encoding="utf-8")
    job.save()
    log = Log(job.log_file, echo=False)
    result = assess_publish(job, settings, log, None)
    assert not result.ok and result.source == "local"
    assert "разрешение" in result.reason
    assert "проверка перед отправкой не пройдена" in job.log_file.read_text(encoding="utf-8")


# ------------------------------------------------------------------ озвучка: цифры и латиница

def test_prepare_for_speech_cases():
    assert prepare_for_speech("В 2024 году вышел GPT") == \
        "В две тысячи двадцать четвёртом году вышел джи-пи-ти"
    assert prepare_for_speech("до 3 дней") == "до трёх дней"
    assert prepare_for_speech("к 5 часам") == "к пяти часам"
    assert prepare_for_speech("AI и GPT") == "эй-ай и джи-пи-ти"
    assert prepare_for_speech("Windows 11") == "виндовс одиннадцать"
    assert "девяностого" in prepare_for_speech("после 1990 года")
    assert "двухтысячному" in prepare_for_speech("к 2000 году")
    assert prepare_for_speech("рост на 50%") == "рост на пятьдесят процентов"
    assert prepare_for_speech("RTX 5060") == "эр-ти-экс пять тысяч шестьдесят"
    assert "чат джи-пи-ти" in prepare_for_speech("Это ChatGPT")
    assert prepare_for_speech("вышел GPT4") == "вышел джи-пи-ти четыре"
    said = prepare_for_speech("В 2024 году вышел GPT", {"GPT": "гпт"})
    assert said.endswith("гпт") and "джи-пи-ти" not in said
    assert prepare_for_speech(said, {"GPT": "гпт"}) == said
    assert prepare_for_speech("З") == "зэ"
    assert prepare_for_speech("буква З") == "буква зэ"
    assert prepare_for_speech("Выбери З.") == "Выбери зэ."
    assert "три" not in prepare_for_speech("Это буква З, не цифра")
    assert prepare_for_speech("3") == "три"
    assert prepare_for_speech("цифра 3 и буква З") == "цифра три и буква зэ"
    assert prepare_for_speech("в городе к реке") == "в городе к реке"
    assert prepare_for_speech("фильм о войне") == "фильм о войне"
    assert prepare_for_speech("я и ты с улицы") == "я и ты с улицы"
    assert prepare_for_speech("вариант Б") == "вариант бэ"
    assert prepare_for_speech("Это ИИ.") == "Это и-и."
    assert prepare_for_speech("код СС") == "код эс-эс"
    assert prepare_for_speech("ее") == "ее"
    said_ii = prepare_for_speech("про ИИ")
    assert said_ii == "про и-и"
    assert prepare_for_speech(said_ii) == said_ii
    ze = prepare_for_speech("буква З")
    assert prepare_for_speech(ze) == ze


def test_prepare_fallback_without_num2words(monkeypatch):
    import builtins
    real = builtins.__import__

    def blocked(name, *a, **k):
        if name == "num2words" or name.startswith("num2words."):
            raise ImportError("нет")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert prepare_for_speech("было 5 штук") == "было пять штук"


def test_voice_changes_speech_not_cue(tmp_path, settings):
    class FakeEngine:
        name = "silero"
        rate = 48000
        tag = "fake:aidar"

        def __init__(self):
            self.calls = []

        def synth(self, texts):
            self.calls.append(list(texts))
            n = self.rate // 5
            tone = (0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / self.rate)).astype(np.float32)
            return [tone for _ in texts]

    cue = Cue(0.2, 1.2, "В 2024 году вышел GPT")
    engine = FakeEngine()
    voice = VoiceOver(tmp_path, settings, Log(None, echo=False), None, engine=engine)
    voice.make(tmp_path, [cue], 3.0, 1)
    assert cue.text == "В 2024 году вышел GPT"
    said = engine.calls[0][0]
    assert "2024" not in said and "GPT" not in said and "джи-пи-ти" in said
    import json
    saved = json.loads((tmp_path / "voice.json").read_text(encoding="utf-8"))
    assert saved["phrases"][0]["text"] == "В 2024 году вышел GPT"
    assert saved["phrases"][0]["said"] == said


def test_config_and_cli_flags(tmp_path):
    (tmp_path / "k.toml").write_text(
        '[render]\nzoom = 1.2\n[subs]\nstyle = "big"\n'
        '[quality]\nmin_score = 8\nvision_model = "llava"\n'
        '[voice.replacements]\nGPT = "гпт"\n', encoding="utf-8")
    s = load_settings(tmp_path / "k.toml", root=tmp_path)
    assert s.render.zoom == 1.2 and s.subs.style == "big"
    assert s.quality.min_score == 8 and s.quality.score is False and s.quality.vision is False
    assert s.quality.vision_model == "llava" and s.voice.replacements["GPT"] == "гпт"
    with pytest.raises(KonveyerError):
        load_settings(tmp_path / "нет.toml", {"render.zoom": 3}, root=tmp_path)

    a = cli.build_parser().parse_args(["run", "u", "--zoom", "1.8", "--subs-style", "big",
                                       "--score", "--vision-check"])
    opts = cli._job_options(a)
    assert opts["zoom"] == 1.8 and opts["subs_style"] == "big"
    assert opts["score"] is True and opts["vision_check"] is True
    bare = cli.build_parser().parse_args(["run", "u"])
    assert "zoom" not in cli._job_options(bare) and "score" not in cli._job_options(bare)
    with pytest.raises(KonveyerError):
        cli._job_options(cli.build_parser().parse_args(["run", "u", "--zoom", "0.5"]))


def test_pronounce_dictionary_words_endings_and_user_file(tmp_path):
    from konveyer.saytext import latin_left
    assert prepare_for_speech("Атака на Hugging Face") == "Атака на Хаггинг Фейс"
    assert prepare_for_speech("в OPENAI и google-а") == "в Опен Эй Ай и Гугла"
    assert prepare_for_speech("Это ИИ и AI") == "Это и-и и эй-ай"
    assert prepare_for_speech("With") == "вит"          # «it» внутри слова не трогаем
    assert prepare_for_speech("Dario  Amodei, CEO Anthropic") == "Дарио Амодей, гендиректор Энтропик"
    assert prepare_for_speech("Claude", {"Claude": "Клауд"}) == "Клауд"
    assert latin_left("Hugging Face и Foo, OpenAI") == ["Foo"]
    (tmp_path / "pronunciations.toml").write_text('Foo = "Фу"\n', encoding="utf-8")
    (tmp_path / "konveyer.toml").write_text('[voice.pronounce]\nBar = "Бар"\n', encoding="utf-8")
    s = load_settings(tmp_path / "konveyer.toml", root=tmp_path)
    assert s.voice.pronounce == {"Bar": "Бар", "Foo": "Фу"}
    assert prepare_for_speech("Foo и Bar", s.voice.pronounce) == "Фу и Бар"