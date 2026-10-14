"""Шаг 6: текст поста ВК и пакет для YouTube Shorts (название, описание, коммент).

Стиль берётся из ваших файлов [post] style_files (prompts/post_prompt.md, post_example.md), если они есть.
Факты — только из названия, описания и расшифровки ролика.
"""
from __future__ import annotations

from pathlib import Path

from konveyer.config import Settings
from konveyer.jobs import Job
from konveyer.llm import LLM, chat_json, set_stage
from konveyer.log import Log
from konveyer.moments import meta_block
from konveyer.speech import NO_SPEECH_MATERIAL
from konveyer.transcript import Transcript
from konveyer.util import KonveyerError, write_json

POST_SYSTEM = (
    "Ты ведёшь ВК-группу «{group}». Пишешь пост к ролику. Язык русский, на «ты», "
    "ирония сухая, как у взрослого паблика, без восклицательных знаков и без детского тона. "
    "Структура: хук с эмодзи; коротко, что это значит; что произошло; список с эмодзи; "
    "самый конкретный кусок; блок «честно, чтобы без паники»; одна фраза-итог; вопрос "
    "и «пиши в комменты». В тексте обязательно имя или ник автора ролика (без ссылки, "
    "без названия платформы). "
    "Ссылок не ставить. Числа, имена и даты — только те, что есть в материалах ниже; "
    "если чего-то нет — не выдумывай. Если в расшифровке сказано, что речи нет — "
    "не выдумывай, о чём ролик: ни жанр, ни сюжет, ни цитаты.\n"
    "Ещё сделай пакет для YouTube Shorts: название — хук и хештеги в конце названия "
    "(обязательно {hashtags} и 1–3 по теме); описание — 2–4 строки и хештеги; "
    "коммент — короткий вопрос, чтобы зрители отвечали.\n"
    "{rules}\n"
    "Ответ — только JSON: {{\"post\": \"...\", \"hashtags\": [\"#...\"], \"shorts_title\": \"...\", "
    "\"shorts_description\": \"...\", \"comment\": \"...\"}}."
)


def load_style(files: list[str], limit: int = 6000) -> str:
    parts = []
    for f in files:
        p = Path(f)
        if p.is_file():
            try:
                parts.append(f"--- {p.name} ---\n" + p.read_text(encoding="utf-8")[:limit])
            except OSError:
                continue
    return "\n\n".join(parts)


def post_prompt(meta: dict, tr: Transcript, clips: list[dict], settings: Settings) -> tuple[str, str]:
    p = settings.post
    from konveyer.postfmt import PROMPT_RULES
    system = POST_SYSTEM.format(group=p.group, hashtags=p.shorts_hashtags,
                                rules=PROMPT_RULES.format(base=p.hashtags))
    style = load_style(p.style_files)
    if style:
        system += "\n\nПравила и пример группы (следуй им):\n" + style
    clip_lines = "\n".join(
        f"- клип {c['id']}: {c.get('title', '').replace('|', ' ')} / {c.get('bottom', '').replace('|', ' ')}"
        for c in clips)
    speech = tr.text()[:8000] if (tr.reliable and not tr.empty) else NO_SPEECH_MATERIAL
    user = (f"{meta_block(meta)}\n\nЯзык речи: {tr.language or 'нет речи'}\n"
            f"Нарезанные клипы:\n{clip_lines or '-'}\n\nРасшифровка речи:\n{speech}")
    return system, user


def _with_hashtags(title: str, hashtags: str) -> str:
    title = " ".join((title or "").split())
    for tag in hashtags.split():
        if tag.lower() not in title.lower():
            title = f"{title} {tag}".strip()
    return title


def parse_post(data, settings: Settings, meta: dict | None = None) -> dict:
    from konveyer.postfmt import decorate_post, model_tags, with_tags

    if not isinstance(data, dict) or not str(data.get("post", "")).strip():
        raise KonveyerError("модель не вернула текст поста")
    from konveyer.postfmt import fix_source, scrub_platforms

    body, tags = decorate_post(fix_source(str(data["post"]), meta, add=True), settings.post,
                               model_tags(data), meta)
    out = {"post": with_tags(body, tags), "hashtags": tags, "ai": True}
    if settings.post.shorts_pack:
        out["shorts_title"] = _with_hashtags(scrub_platforms(str(data.get("shorts_title", ""))),
                                             settings.post.shorts_hashtags)
        out["shorts_description"] = fix_source(str(data.get("shorts_description", "")), meta)
        out["comment"] = scrub_platforms(str(data.get("comment", ""))).strip()
    return out


def fallback_post(meta: dict, clips: list[dict], settings: Settings) -> dict:
    """Черновик без ИИ: скелет по шаблону группы, чтобы дописать руками."""
    who = meta.get("uploader") or "автор"
    title = meta.get("title") or "ролик"
    hook = clips[0].get("title", "").replace("|", " ") if clips else title
    from konveyer.postfmt import decorate_post, with_tags

    post = (f"🤖 {hook}\n\n"
            f"{who} выложил ролик «{title}».\n\n"
            "Что произошло:\n🔹 ...\n🔹 ...\n\n"
            "Честно, чтобы без паники: ...\n\n"
            "Итог: ...\n\n"
            "А ты как думаешь? Пиши в комменты.")
    body, tags = decorate_post(post, settings.post, (), meta)
    out = {"post": with_tags(body, tags), "hashtags": tags, "ai": False}
    if settings.post.shorts_pack:
        out["shorts_title"] = _with_hashtags(hook[:70], settings.post.shorts_hashtags)
        from konveyer.postfmt import scrub_platforms, source_line
        src = source_line(meta) or f"Автор: {who}"
        out["shorts_description"] = f"{scrub_platforms(title)}\n{src}\n{settings.post.shorts_hashtags}"
        out["comment"] = "А ты бы такое попробовал? Пиши в ответах."
    return out


def render_md(post: dict) -> str:
    md = post["post"].strip() + "\n"
    if post.get("shorts_title"):
        md += ("\n\n---\n\n## YouTube Shorts\n\n"
               f"**Название:** {post['shorts_title']}\n\n"
               f"**Описание:**\n\n{post.get('shorts_description', '')}\n\n"
               f"**Коммент (первым под роликом):** {post.get('comment', '')}\n")
    if not post.get("ai", True):
        md += "\n\n> Черновик без ИИ: DeepSeek не был доступен. Допиши точки руками.\n"
    return md


def save_post(job: Job, post: dict) -> None:
    write_json(job.post_json, post)
    job.post_md.write_text(render_md(post), encoding="utf-8")


def make_post(job: Job, settings: Settings, log: Log, llm: LLM | None, tr: Transcript) -> dict:
    meta, clips = job.data["meta"], job.data.get("clips", [])
    post = None
    if llm:
        log.info("пишу пост через DeepSeek")
        set_stage(llm, "post")
        try:
            system, user = post_prompt(meta, tr, clips, settings)
            post = parse_post(chat_json(llm, system, user), settings, meta)
        except KonveyerError as e:
            log.warn(f"пост через ИИ не получился: {e}")
    if post is None:
        log.info("делаю черновик поста по шаблону")
        post = fallback_post(meta, clips, settings)
    save_post(job, post)
    log.info(f"пост: {job.post_md.name} ({len(post['post'])} символов)")
    if clips and settings.post.clip_desc:
        from konveyer.clipdesc import make_descriptions
        make_descriptions(job, settings, log, llm, tr)
    return post
