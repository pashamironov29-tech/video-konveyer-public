"""Оформление постов: эмодзи у абзацев, ⏱ у таймкодов, 6–10 хештегов, проверка их не трогает."""
from __future__ import annotations

from konveyer.fullvideo import render_full_md
from konveyer.post import fallback_post, parse_post
from konveyer.postfmt import decorate_post, split_tags, starts_with_emoji, strip_decor
from konveyer.quality import unsupported_claims

BASE = ["#ИИ", "#нейросети", "#искусственныйинтеллект", "#интервью", "#технологии"]


def test_every_paragraph_gets_emoji_and_tags(settings):
    body, tags = decorate_post("Амодеи рассказал CNN о рое.\n\nЭто опасно?\n\n⚠️ Уже с эмодзи.",
                               settings.post, ["Anthropic", "#OpenAI"], {"uploader": "CNN"})
    paras = body.split("\n\n")
    assert len(paras) == 3 and all(starts_with_emoji(p) for p in paras)
    assert paras[2] == "⚠️ Уже с эмодзи."
    assert tags[:5] == BASE and "#Anthropic" in tags and "#OpenAI" in tags
    assert 6 <= len(tags) <= 10


def test_model_without_hashtags_still_gets_base_tags(settings):
    post = parse_post({"post": "Текст поста"}, settings, {"uploader": "CNN"})
    body, tags = split_tags(post["post"])
    assert starts_with_emoji(body)
    assert tags[:5] == BASE and len(tags) >= 6          # + тематический из автора
    draft = fallback_post({"uploader": "CNN", "title": "ролик"}, [], settings)
    assert split_tags(draft["post"])[1][:5] == BASE


def test_hashtags_can_be_switched_off(settings):
    settings.post.hashtags_on = False
    settings.post.emoji = False
    body, tags = decorate_post("Текст.\n\nЕщё.", settings.post, ["#X"])
    assert tags == [] and body == "Текст.\n\nЕщё."


def test_full_post_has_clock_and_no_emoji_in_timecodes(settings):
    body, tags = decorate_post("Пост.", settings.post, ["#Anthropic"])
    md = render_full_md({"post": body, "topics": [{"time": "0:00", "title": "Начало"}],
                         "hashtags": tags, "clock": True})
    assert "\n⏱ Таймкоды:\n\n0:00 — Начало\n" in md
    assert md.rstrip().splitlines()[-1].startswith("#ИИ")


def test_fabrication_check_ignores_hashtags_and_emoji():
    assert strip_decor("🤖 Амодеи на CNN\n\n#ИИ #Anthropic") .strip() == "Амодеи на CNN"
    data = {"claims": [{"claim": "#Anthropic #OpenAI", "quote": ""},
                       {"claim": "🐝", "quote": ""},
                       {"claim": "выдуманный факт", "quote": ""}]}
    assert unsupported_claims(data, "speech", "речь") == ["выдуманный факт"]


def test_no_platform_names_source_is_handle(settings):
    from konveyer.postfmt import clean_tag, fix_source, source_handle, source_line

    meta = {"uploader": "Alerta News 24", "uploader_id": "AlertaNews24",
            "url": "https://x.com/AlertaNews24/status/1"}
    assert source_handle(meta) == "@AlertaNews24" and source_line(meta) == "Источник: @AlertaNews24"
    assert source_handle({"uploader_id": "123", "handle": "Sara_Clips"}) == "@Sara_Clips"
    assert source_handle({"url": "https://x.com/Kwesi/status/5"}) == "@Kwesi"
    assert source_handle({"uploader": "Канал с пробелами"}) == ""
    text = fix_source("🤖 Ролик разлетелся в X за сутки, потом в Instagram.\n\n"
                      "Tesla Model X — ни при чём.\n\nИсточник: X / @AlertaNews24", meta)
    assert "в X" not in text and "Instagram" not in text and "Model X" in text
    assert text.splitlines()[-1] == "Источник: @AlertaNews24"
    assert clean_tag("#Instagram") == "" and clean_tag("X") == "" and clean_tag("#роботы") == "#роботы"
    post = parse_post({"post": "Видео из соцсети X (бывший Twitter) про робота.",
                       "hashtags": ["#X", "#роботы"], "shorts_title": "Робот в Instagram",
                       "shorts_description": "Робот.\nАвтор: Alerta в X"}, settings, meta)
    for field in ("post", "shorts_title", "shorts_description"):
        assert "Twitter" not in post[field] and "Instagram" not in post[field] and " X" not in post[field]
    assert "Источник: @AlertaNews24" in post["post"] and "#X" not in post["hashtags"]
    assert "Источник: @AlertaNews24" in post["shorts_description"]
