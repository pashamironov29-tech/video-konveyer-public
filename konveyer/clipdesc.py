"""Описание к каждому Shorts: 2–3 коротких абзаца с эмодзи строго по этому клипу,
вопрос для комментов и строка хештегов (постоянные из [post] hashtags + тематические + #shorts).

Пишет DeepSeek по русскому тексту клипа (его субтитрам), факты — только из клипа.
Файл — clips/clip_NN/description.md. До 1024 символов уходит подписью к видео,
длиннее — отдельным сообщением сразу после видео.
"""
from __future__ import annotations

import json
from pathlib import Path

from konveyer.config import Settings
from konveyer.jobs import Job
from konveyer.llm import chat_json, set_stage
from konveyer.log import Log
from konveyer.postfmt import (SOURCE_RULE, clean_tag, decorate_body, fix_source, merge_tags,
                              meta_tags, model_tags, paragraphs, scrub_platforms, source_handle,
                              source_line, split_tags, starts_with_emoji, with_tags)
from konveyer.util import KonveyerError

DESC_NAME = "description.md"
CAPTION_LIMIT = 1024          # подпись к видео в Telegram
SHORTS_TAG = "#shorts"
MAX_TAGS = 10
TEXT_LIMIT = 700              # просим у модели: с хештегами влезает в подпись
ASK = "👇 А ты что думаешь? Пиши в комменты."
SAMPLE = (
    "🐝 ИИ-агенты жертвовали собой ради других, как пчёлы, только хуже: один не успевал "
    "с задачей, сделал часть и передал эстафету следующему.\n\n"
    "⚠️ Амодеи на CNN о том, как быстро агенты научились координироваться без человека.\n\n"
    "👇 Страшно или круто? Пиши в комменты."
)
DESC_SYSTEM = (
    "Пишешь описание к одному короткому вертикальному ролику (Shorts) для подписи под видео "
    "в Telegram. Язык русский, на «ты», ирония сухая, без восклицательных знаков. "
    "2–3 коротких абзаца, каждый начинается с одного подходящего по смыслу эмодзи, абзацы "
    "через пустую строку: первый — хук по самому яркому моменту именно этого клипа; второй "
    "(и, если нужно, третий) — кто говорит, где и о чём, одной-двумя фразами. Последний "
    "абзац — короткий вопрос зрителям и «Пиши в комменты», начинается с 👇. "
    "Факты, числа и имена — только из текста клипа ниже, его надписей, названия ролика и "
    "автора; чего нет в клипе — не добавляй, ни слова о других частях интервью. "
    "Кто именно сказал фразу, пиши только если это ясно из текста; в интервью вопросы "
    "задаёт ведущий — его слова гостю не приписывай. "
    "Весь текст не длиннее {limit} символов. Хештеги в текст не пиши — дай отдельно в поле "
    "\"hashtags\" 2–4 тематических по этому клипу: компании, имена, слитно, без пробелов "
    "(например #Anthropic #OpenAI #Амодеи #CNN). "
    + SOURCE_RULE.replace("{", "{{").replace("}", "}}") + " "
    "Ответ — только JSON: {{\"text\": \"...\", \"hashtags\": [\"#...\"]}}.\n\n"
    "Образец формы (факты из него не брать):\n{sample}"
)


def utf16_len(text: str) -> int:
    """Длина так, как считает Telegram (эмодзи — две единицы)."""
    return len((text or "").encode("utf-16-le")) // 2


def fits_caption(text: str) -> bool:
    return 0 < utf16_len(text.strip()) <= CAPTION_LIMIT


def desc_path(job: Job, clip: dict) -> Path:
    rel = str(clip.get("file") or "")
    folder = (job.dir / rel).parent if rel else job.clip_dir(int(clip["id"]))
    return folder / DESC_NAME


def read_description(job: Job, clip: dict) -> str:
    path = desc_path(job, clip)
    try:
        return path.read_text(encoding="utf-8").strip() if path.is_file() else ""
    except OSError:
        return ""


def clip_text(job: Job, clip: dict, tr=None, limit: int = 3000) -> str:
    """Русский текст клипа: субтитры (captions.ass), иначе фразы озвучки, иначе расшифровка."""
    from konveyer.quality import ass_plain

    folder = desc_path(job, clip).parent
    ass = folder / "captions.ass"
    if ass.is_file():
        text = ass_plain(ass)
        if text.strip():
            return text[:limit]
    voice = folder / "voice.json"
    if voice.is_file():
        try:
            data = json.loads(voice.read_text(encoding="utf-8"))
            text = " ".join(str(p.get("text") or "") for p in data.get("phrases") or [])
            if text.strip():
                return " ".join(text.split())[:limit]
        except (OSError, ValueError, AttributeError):
            pass
    # ненадёжная речь (музыка, шум) — расшифровку не показываем, иначе модель сочинит
    if (tr is not None and hasattr(tr, "words_between") and getattr(tr, "reliable", True)
            and not getattr(tr, "empty", False)):
        words = tr.words_between(float(clip.get("start", 0)), float(clip.get("end", 0)))
        return " ".join("".join(w.word for w in words).split())[:limit]
    return ""


def desc_tags(post_cfg, model=(), meta: dict | None = None) -> list[str]:
    """Постоянные + тематические + #shorts последним, всего не больше 10."""
    if not getattr(post_cfg, "hashtags_on", True):
        return []
    tags = [t for t in merge_tags(post_cfg.hashtags, model, meta_tags(meta or {}),
                                  limit=MAX_TAGS) if t.casefold() != SHORTS_TAG]
    return tags[:MAX_TAGS - 1] + [SHORTS_TAG]


def shape_text(text: str, emoji: bool = True) -> str:
    """2–3 абзаца по делу + вопрос последним; каждый с эмодзи."""
    blocks = paragraphs(text)
    if not blocks:
        return ""
    ask = blocks[-1] if "?" in blocks[-1] else ""
    body = blocks[:-1] if ask else blocks
    body = body[:3]
    if not ask:
        ask = ASK
    elif emoji and not starts_with_emoji(ask):
        ask = f"👇 {ask}"
    return decorate_body("\n\n".join(body + [ask]), emoji)


def finish(text: str, tags: list[str], source: str = "", max_blocks: int = 4) -> str:
    """Склеить и убедиться, что влезает в подпись: лишние абзацы и теги — долой.
    source («Источник: @ник») — последней строкой последнего абзаца (через один перевод строки),
    перед хештегами. Не больше max_blocks абзацев; вопрос (последний абзац) остаётся."""
    blocks = paragraphs(text)

    def joined(bs: list[str]) -> str:
        body = "\n\n".join(bs)
        if source:
            body = f"{body}\n{source}" if body else source
        return body

    out = with_tags(joined(blocks), tags)
    while len(blocks) > 2 and (len(blocks) > max_blocks or not fits_caption(out)):
        blocks = blocks[:-2] + blocks[-1:]          # выкинуть предпоследний абзац, вопрос оставить
        out = with_tags(joined(blocks), tags)
    return out.strip()


def _split_source(text: str, meta: dict | None) -> tuple[str, str]:
    """Чистка платформ; строку «Источник» вынуть из текста (её ставит finish)."""
    clean = fix_source(text, meta)
    keep, src = [], source_line(meta)
    for line in clean.splitlines():
        if line.startswith("Источник:"):
            src = src or line
            continue
        keep.append(line)
    return "\n".join(keep).strip(), src


def parse_desc(data, settings: Settings, meta: dict | None = None) -> str:
    if not isinstance(data, dict) or not str(data.get("text", "")).strip():
        raise KonveyerError("модель не вернула описание клипа")
    body, body_tags = split_tags(str(data["text"]))
    body, src = _split_source(body, meta)
    body = shape_text(body, getattr(settings.post, "emoji", True))
    tags = desc_tags(settings.post, model_tags(data) + [clean_tag(t) for t in body_tags], meta)
    return finish(body, tags, src)


def fallback_desc(clip: dict, meta: dict, settings: Settings) -> str:
    """Без ИИ: надписи клипа и откуда он."""
    title = " ".join(str(clip.get("title") or "").replace("|", " ").split())
    bottom = " ".join(str(clip.get("bottom") or "").replace("|", " ").split())
    hook = " ".join(x for x in (title, bottom) if x) or str(meta.get("title") or "Фрагмент ролика")
    who = str(meta.get("uploader") or "").strip()
    src = str(meta.get("title") or "").strip()
    second = f"🎬 Фрагмент ролика «{src}»" + (f", {who}." if who else ".") if src else ""
    body = shape_text(scrub_platforms("\n\n".join(x for x in (f"🤖 {hook}", second) if x)),
                      getattr(settings.post, "emoji", True))
    return finish(body, desc_tags(settings.post, (), meta), source_line(meta))


def desc_prompt(job: Job, clip: dict, text: str) -> tuple[str, str]:
    meta = job.data.get("meta") or {}
    system = DESC_SYSTEM.format(limit=TEXT_LIMIT, sample=SAMPLE)
    marks = " / ".join(" ".join(str(clip.get(k) or "").replace("|", " ").split())
                       for k in ("kicker", "title", "bottom") if clip.get(k))
    user = (f"Ролик: {meta.get('title') or '-'}\nАвтор: {meta.get('uploader') or '-'}"
            f" ({source_handle(meta) or 'ника нет'})\n"
            f"Клип {clip['id']}, надписи на экране: {marks or '-'}\n\n"
            f"Русский текст клипа:\n{text or '(речи нет — о содержании не выдумывай)'}")
    return system, user


def make_description(job: Job, clip: dict, settings: Settings, log: Log, llm, tr=None) -> str:
    meta = job.data.get("meta") or {}
    text = clip_text(job, clip, tr)
    desc = ""
    if llm is not None and text:
        set_stage(llm, "post")
        try:
            system, user = desc_prompt(job, clip, text)
            desc = parse_desc(chat_json(llm, system, user), settings, meta)
        except KonveyerError as e:
            log.warn(f"описание клипа {clip['id']} через ИИ не получилось: {e}")
    if not desc:
        desc = fallback_desc(clip, meta, settings)
    path = desc_path(job, clip)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(desc + "\n", encoding="utf-8")
    clip["description"] = path.relative_to(job.dir).as_posix()
    how = "подпись к видео" if fits_caption(desc) else "отдельным сообщением (длиннее 1024)"
    log.info(f"описание клипа {clip['id']}: {path.name}, {utf16_len(desc)} символов — {how}")
    return desc


def make_descriptions(job: Job, settings: Settings, log: Log, llm, tr=None,
                      only: list[int] | None = None) -> dict[int, str]:
    if not getattr(settings.post, "clip_desc", True):
        return {}
    out: dict[int, str] = {}
    for clip in job.data.get("clips") or []:
        if only and clip["id"] not in only:
            continue
        out[clip["id"]] = make_description(job, clip, settings, log, llm, tr)
    job.save()
    return out
