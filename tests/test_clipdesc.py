"""Своё описание к каждому Shorts: формат, хештеги, подпись или сообщение, без повторов."""
from __future__ import annotations

import shutil

from conftest import FAKE_TOKEN, FakeLLM, FakeSession, needs_ffmpeg
from konveyer.clipdesc import (SHORTS_TAG, fits_caption, make_descriptions, parse_desc,
                               read_description, utf16_len)
from konveyer.jobs import Job
from konveyer.log import Log
from konveyer.postfmt import paragraphs, starts_with_emoji
from konveyer.quality import collect_texts
from konveyer.telegram import Telegram, send_job

ASS = ("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
       "Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,{\\c&H00FFFF&}Агенты жертвовали собой\n"
       "Dialogue: 0,0:00:02.00,0:00:04.00,Default,,0,0,0,,и передавали задачу дальше\n")


def test_parse_desc_format_and_tags(settings):
    data = {"text": "Агенты передают задачу.\n\n⚠️ Амодеи на CNN.\n\nЕщё абзац.\n\nИ ещё.\n\n"
                    "Страшно или круто? Пиши в комменты.\n\n#Лишний",
            "hashtags": ["#Anthropic", "#Амодеи", "#CNN", "#ИИ"]}
    text = parse_desc(data, settings, {"uploader": "CNN"})
    body, tags_line = text.rsplit("\n\n", 1)
    blocks = paragraphs(body)
    assert 2 <= len(blocks) <= 4 and all(starts_with_emoji(b) for b in blocks)
    assert "?" in blocks[-1] and blocks[-1].startswith("👇")
    tags = tags_line.split()
    assert tags[:5] == settings.post.hashtags.split() and tags[-1] == SHORTS_TAG
    assert "#Anthropic" in tags and "#Амодеи" in tags and len(tags) <= 10
    assert tags.count("#ИИ") == 1 and fits_caption(text)


def test_tags_given_as_one_string_are_split(settings):
    text = parse_desc({"text": "Абзац.\n\nЧто думаешь?",
                       "hashtags": ["#Anthropic #OpenAI,#HuggingFace", "#CNN"]}, settings)
    tags = text.rsplit("\n\n", 1)[1].split()
    assert {"#Anthropic", "#OpenAI", "#HuggingFace", "#CNN"} <= set(tags)
    assert len(tags) <= 10 and tags[-1] == SHORTS_TAG


def test_question_added_and_long_text_trimmed(settings):
    text = parse_desc({"text": "Абзац про агентов.\n\nВторой абзац."}, settings)
    assert "Пиши в комменты" in paragraphs(text)[-2]
    long = "\n\n".join(["Агенты " * 60] * 3 + ["Что думаешь?"])
    out = parse_desc({"text": long}, settings)
    assert fits_caption(out) and utf16_len(out) <= 1024 and "Что думаешь?" in out


def _job(settings, video=None):
    job = Job.create(settings, "a.mp4")
    folder = job.clip_dir(1)
    folder.mkdir(parents=True)
    (folder / "captions.ass").write_text(ASS, encoding="utf-8")
    if video:
        shutil.copy(video, folder / "clip.mp4")
    job.data["meta"] = {"title": "Интервью", "uploader": "CNN", "width": 640, "height": 360}
    job.data["clips"] = [{"id": 1, "start": 0, "end": 4, "title": "Рой|агентов", "bottom": "",
                          "file": "clips/clip_01/clip.mp4"}]
    job.save()
    return job


def test_make_descriptions_uses_clip_subtitles(settings):
    job = _job(settings)
    llm = FakeLLM()
    out = make_descriptions(job, settings, Log(None, echo=False), llm)
    assert "Агенты жертвовали собой" in llm.calls[-1][1]          # промпт — по субтитрам клипа
    text = (job.clip_dir(1) / "description.md").read_text(encoding="utf-8").strip()
    assert text == out[1] and text.startswith("🐝") and text.endswith(SHORTS_TAG)
    assert job.data["clips"][0]["description"] == "clips/clip_01/description.md"
    # без ИИ — запасное описание по надписям клипа
    make_descriptions(job, settings, Log(None, echo=False), None)
    plain = read_description(job, job.data["clips"][0])
    assert plain.startswith("🤖 Рой агентов") and "#ИИ" in plain and plain.endswith(SHORTS_TAG)
    # проверка на выдумки видит описание, но без эмодзи и хештегов
    descs = dict(collect_texts(job))["описания Shorts"]
    assert "Рой агентов" in descs and "#" not in descs and "🤖" not in descs


@needs_ffmpeg
def test_send_description_as_caption_no_post_no_repeat(settings, wide_video):
    settings.render.width, settings.render.height = 640, 360
    job = _job(settings, wide_video)
    make_descriptions(job, settings, Log(None, echo=False), FakeLLM())
    job.post_md.write_text("Общий пост", encoding="utf-8")
    s = FakeSession()
    sent = send_job(job, settings, Log(None, echo=False), tg=Telegram(FAKE_TOKEN, "1", s), check=False)
    assert [p["url"].rsplit("/", 1)[1] for p in s.posts] == ["sendVideo"]   # общий пост выключен
    assert s.posts[0]["data"]["caption"].startswith("🐝") and sent["post"] is False
    assert job.data["delivered"]["clip:1"]["desc"]
    s2 = FakeSession()
    send_job(job, settings, Log(None, echo=False), tg=Telegram(FAKE_TOKEN, "1", s2), check=False)
    assert s2.posts == []                                                     # повтор не шлётся
    settings.telegram.shorts_post = True
    send_job(job, settings, Log(None, echo=False), tg=Telegram(FAKE_TOKEN, "1", s2), check=False)
    assert [p["data"]["text"] for p in s2.posts] == ["Общий пост"]           # по настройке — да


@needs_ffmpeg
def test_long_description_goes_right_after_video(settings, wide_video):
    settings.render.width, settings.render.height = 640, 360
    job = _job(settings, wide_video)
    long = "🐝 " + "Очень длинное описание. " * 60
    (job.clip_dir(1) / "description.md").write_text(long, encoding="utf-8")
    s = FakeSession()
    send_job(job, settings, Log(None, echo=False), tg=Telegram(FAKE_TOKEN, "1", s), check=False)
    assert [p["url"].rsplit("/", 1)[1] for p in s.posts] == ["sendVideo", "sendMessage"]
    assert s.posts[0]["data"]["caption"].startswith("Клип 1:")
    assert s.posts[1]["data"]["text"] == long.strip()
    assert "desc:1" in job.data["delivered"]
    s2 = FakeSession()
    send_job(job, settings, Log(None, echo=False), tg=Telegram(FAKE_TOKEN, "1", s2), check=False)
    assert s2.posts == []


def test_desc_source_handle_no_platforms(settings):
    from konveyer.clipdesc import fallback_desc

    meta = {"uploader": "Alerta", "uploader_id": "AlertaNews24", "title": "Робот в X"}
    data = {"text": "🤖 Робот ударил стримера, видео в X и Instagram.\n\nИсточник: Instagram\n\n"
                    "👇 Кто виноват? Пиши в комменты.", "hashtags": ["#Westcol", "#Instagram"]}
    out = parse_desc(data, settings, meta)
    body = out.rsplit("\n\n", 1)[0]
    assert "Instagram" not in out and "в X" not in out
    assert body.splitlines()[-1] == "Источник: @AlertaNews24"
    assert out.index("👇") < out.index("Источник") and "#Westcol" in out
    fb = fallback_desc({"id": 1, "title": "Робот"}, meta, settings)
    assert "Источник: @AlertaNews24" in fb and "в X" not in fb
